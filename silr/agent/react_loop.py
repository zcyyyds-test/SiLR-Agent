"""ReActAgent: Bounded ReAct loop with SiLR verification.

Flow per step:
1. Observe → domain observer → compressed JSON
2. Reason+Act → LLM → ActionParser → action dict
3. Verify → SiLRVerifier → PASS/FAIL
4. Retry (up to max_proposals_per_step) or apply
5. Check recovery via observer
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .config import AgentConfig
from .types import (
    Observation, StepRecord, StepOutcome, EpisodeResult,
)
from .llm.base import BaseLLMClient
from .action_parser import ActionParser, ParseError
from .observation import BaseObserver
from .trajectory import _clean_thought
from silr.core.interfaces import BaseSystemManager
from silr.core.config import DomainConfig
from silr.verifier import SiLRVerifier, Verdict

logger = logging.getLogger(__name__)


class ReActAgent:
    """Bounded ReAct agent with SiLR verification loop."""

    def __init__(
        self,
        manager: BaseSystemManager,
        verifier: SiLRVerifier,
        llm_client: BaseLLMClient,
        domain_config: DomainConfig,
        config: AgentConfig = AgentConfig(),
        failsafe: Optional[Any] = None,
        trajectory_recorder: Optional[Any] = None,
        few_shot_context: Optional[str] = None,
    ):
        self._manager = manager
        self._verifier = verifier
        self._llm = llm_client
        self._config = config
        self._failsafe = failsafe
        self._recorder = trajectory_recorder

        dc = domain_config
        self._tools = dc.create_toolset(manager)
        self._tool_schemas = (
            dc.build_tool_schemas(manager)
            if dc.build_tool_schemas else []
        )
        base_prompt = (
            dc.build_system_prompt(manager, self._tool_schemas)
            if dc.build_system_prompt
            else f"You are a {dc.domain_name} recovery agent."
        )
        valid_ids = (
            dc.get_valid_device_ids(manager)
            if dc.get_valid_device_ids else {}
        )
        self._parser = ActionParser(
            allowed_actions=dc.allowed_actions,
            valid_device_ids=valid_ids,
            param_aliases=dc.param_aliases,
        )
        self._observer = (
            dc.create_observer(manager)
            if dc.create_observer
            else _MinimalObserver(manager, self._tools)
        )

        if few_shot_context:
            self._system_prompt = base_prompt + "\n\n" + few_shot_context
        else:
            self._system_prompt = base_prompt

    def run_episode(self, scenario_id: str = "unknown") -> EpisodeResult:
        """Run a complete recovery episode.

        Returns EpisodeResult with full step history.
        """
        result = EpisodeResult(scenario_id=scenario_id)
        messages = [{"role": "system", "content": self._system_prompt}]
        consecutive_step_fails = 0

        for step_num in range(1, self._config.max_steps + 1):
            # 1. Observe
            obs = self._observer.observe()

            # Check if already recovered
            if obs.is_stable:
                record = StepRecord(
                    step_number=step_num,
                    observation=obs,
                    outcome=StepOutcome.RECOVERED,
                )
                result.steps.append(record)
                result.recovered = True
                break

            # 2. Build user message with observation
            user_msg = self._build_observation_message(step_num, obs)
            messages.append({"role": "user", "content": user_msg})

            # 3. Propose + Verify loop
            record = StepRecord(step_number=step_num, observation=obs)
            action_applied = False

            for proposal_idx in range(self._config.max_proposals_per_step):
                result.total_proposals += 1

                # Call LLM
                try:
                    response = self._llm.chat(
                        messages=messages,
                        tools=self._tool_schemas if self._llm.supports_tool_use() else None,
                        temperature=self._config.temperature,
                        seed=self._config.seed,
                    )
                except Exception as e:
                    logger.error(f"LLM call failed: {e}")
                    record.error = f"LLM error: {e}"
                    record.outcome = StepOutcome.FAIL_PARSE
                    break

                # Parse action
                try:
                    thought, action = self._parser.parse(response)
                    thought = _clean_thought(thought)
                    record.thought = thought
                except ParseError as e:
                    logger.warning(f"Parse error (attempt {proposal_idx+1}): {e}")
                    record.error = str(e)
                    result.total_rejections += 1
                    messages.append({
                        "role": "assistant",
                        "content": "(malformed response)",
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[PARSE ERROR] Your previous response could not be parsed. "
                            f"Ignore it entirely and respond with a fresh action as JSON: "
                            f'{{"tool_name": "<tool>", "params": {{...}}}}'
                        ),
                    })
                    continue

                record.proposed_actions.append(action)

                # Handle "none" action (agent thinks system is stable)
                if action.get("tool_name") == "none":
                    record.outcome = StepOutcome.RECOVERED
                    action_applied = True
                    messages.append({
                        "role": "assistant",
                        "content": f"Thought: {thought}\nAction: none (system stable)",
                    })
                    break

                # 4. Verify
                if self._config.enable_verification:
                    vr = self._verifier.verify(action)
                    record.verification_results.append(vr)

                    if vr.verdict == Verdict.PASS:
                        apply_result = self._apply_action(action)
                        record.applied_action = action
                        record.tool_result = apply_result
                        record.outcome = StepOutcome.SUCCESS
                        action_applied = True
                        consecutive_step_fails = 0

                        messages.append({
                            "role": "assistant",
                            "content": f"Thought: {thought}\nAction: {_format_action(action)}",
                        })
                        messages.append({
                            "role": "user",
                            "content": "[SiLR APPROVED] Action applied successfully.",
                        })
                        break
                    else:
                        result.total_rejections += 1
                        feedback = self._format_rejection(vr)
                        messages.append({
                            "role": "assistant",
                            "content": f"Thought: {thought}\nAction: {_format_action(action)}",
                        })
                        messages.append({
                            "role": "user",
                            "content": feedback,
                        })
                        logger.info(
                            f"Step {step_num}, proposal {proposal_idx+1} rejected: "
                            f"{vr.fail_reason}"
                        )
                else:
                    # NoVerify mode — apply directly
                    apply_result = self._apply_action(action)
                    record.applied_action = action
                    record.tool_result = apply_result
                    record.outcome = StepOutcome.SUCCESS
                    action_applied = True
                    messages.append({
                        "role": "assistant",
                        "content": f"Thought: {thought}\nAction: {_format_action(action)}",
                    })
                    break

            if not action_applied:
                record.outcome = StepOutcome.FAIL_VERIFY
                consecutive_step_fails += 1

                # Fail-safe: trigger after N full steps of failure
                if (
                    consecutive_step_fails >= self._config.consecutive_fail_limit
                    and self._failsafe is not None
                ):
                    last_rejected = record.proposed_actions[-1] if record.proposed_actions else None
                    fs_action = self._failsafe.suggest_escalated(obs, last_rejected)
                    if fs_action:
                        apply_result = self._apply_action(fs_action)
                        fs_ok = (
                            apply_result is not None
                            and apply_result.get("status") != "error"
                        )
                        if fs_ok:
                            record.applied_action = fs_action
                            record.tool_result = apply_result
                            record.outcome = StepOutcome.FAILSAFE
                            result.failsafe_triggered = True
                            consecutive_step_fails = 0
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"[FAILSAFE] Applied rule-based action: "
                                    f"{_format_action(fs_action)}"
                                ),
                            })
                        else:
                            logger.warning(
                                f"Failsafe action failed: {fs_action} → {apply_result}"
                            )

            result.steps.append(record)

            # Trim context window
            messages = self._trim_context(messages)

            # Run steady-state solver after action to update system state
            if record.applied_action:
                self._manager.solve()

        # Final observation
        result.final_observation = self._observer.observe()
        result.recovered = result.final_observation.is_stable
        result.total_steps = len(result.steps)

        # Record trajectory
        if self._recorder is not None:
            self._recorder.record_episode(result)

        return result

    def _apply_action(self, action: dict) -> dict | None:
        """Execute action on the main system."""
        tool_name = action["tool_name"]
        params = action.get("params", {})
        tool = self._tools.get(tool_name)
        if tool is None:
            logger.error(f"Tool '{tool_name}' not found")
            return None
        return tool.execute(**params)

    def _build_observation_message(self, step_num: int, obs: Observation) -> str:
        """Build user message containing observation for the LLM."""
        parts = [f"## Step {step_num} — System Observation\n"]
        parts.append(obs.compressed_json)
        if obs.violations:
            parts.append(f"\n{len(obs.violations)} active violation(s) detected.")
        else:
            parts.append("\nNo violations detected.")
        parts.append("\nPropose ONE action to improve the system state.")
        return "\n".join(parts)

    def _format_rejection(self, vr: Any) -> str:
        """Format verification rejection feedback for LLM context."""
        mode = self._config.verification_feedback_mode
        action = vr.action
        action_str = _format_action(action)

        if mode == "full":
            return f"[SiLR REJECTED] {action_str}\n\n{vr.report_text}\n\nPlease propose a revised action."

        if mode == "detailed":
            lines = [f"[SiLR REJECTED] {action_str} FAILED."]
            lines.append(f"Reason: {vr.fail_reason}")
            if vr.check_results:
                for cr in vr.check_results:
                    if not cr.passed:
                        for v in cr.violations:
                            lines.append(f"  - {v.detail}")
            lines.append("Please propose a revised action.")
            return "\n".join(lines)

        # summary mode
        reason = (vr.fail_reason or "unknown")
        reason = reason.split("\n")[0]
        if len(reason) > 120:
            reason = reason[:117] + "..."

        lines = [f"[SiLR REJECTED] {action_str} FAILED."]
        lines.append(f"Reason: {reason}")
        lines.append("Suggestion: Try a more conservative action.")
        return "\n".join(lines)

    def _trim_context(self, messages: list[dict]) -> list[dict]:
        """Sliding window: keep system prompt + last 2 full step pairs."""
        if len(messages) <= 7:
            return messages

        system_msg = messages[0]
        conversation = messages[1:]

        keep_count = 8
        if len(conversation) <= keep_count:
            return messages

        older = conversation[:-keep_count]
        recent = conversation[-keep_count:]

        summary_lines = ["## Previous Steps Summary"]
        step_idx = 0
        for msg in older:
            if msg["role"] == "user" and "System Observation" in msg.get("content", ""):
                step_idx += 1
                content = msg["content"]
                for line in content.split("\n"):
                    if "violation" in line.lower():
                        summary_lines.append(f"Step {step_idx} obs: {line.strip()}")
                        break
            elif msg["role"] == "assistant" and "Action:" in msg.get("content", ""):
                content = msg["content"]
                for line in content.split("\n"):
                    if line.startswith("Action:"):
                        summary_lines.append(f"  action: {line}")
                        break
            elif msg["role"] == "user" and "[SiLR" in msg.get("content", ""):
                content = msg["content"]
                first_line = content.split("\n")[0]
                summary_lines.append(f"  → {first_line}")

        if len(summary_lines) > 1:
            summary = {"role": "user", "content": "\n".join(summary_lines)}
            return [system_msg, summary] + recent
        return [system_msg] + recent


class _MinimalObserver(BaseObserver):
    """Fallback observer when domain doesn't provide one.

    Simply reports system as unstable with empty observation.
    Domains should provide their own observer via DomainConfig.create_observer.
    """

    def __init__(self, manager: BaseSystemManager, tools: dict):
        self._manager = manager
        self._tools = tools

    def observe(self) -> Observation:
        return Observation(
            raw={},
            compressed_json="{}",
            violations=[],
            is_stable=False,
        )


def _format_action(action: dict) -> str:
    """Format action dict as readable string."""
    name = action.get("tool_name", "unknown")
    params = action.get("params", {})
    param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
    return f"{name}({param_str})"
