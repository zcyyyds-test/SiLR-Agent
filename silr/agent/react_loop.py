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
import os
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

        # In bare-text mode (tools=None to API), the domain system prompt may
        # rely on native tool-calling to convey output format. Append an
        # explicit JSON-block emission contract so cross-family models without
        # a matching vLLM tool-call parser still produce parseable actions.
        if not self._llm.supports_tool_use():
            base_prompt = base_prompt + "\n\n" + (
                "## Output Format (REQUIRED)\n\n"
                "After your reasoning, emit EXACTLY ONE action as a JSON code "
                "block on its own line, using this schema:\n\n"
                "```json\n"
                "{\"tool_name\": \"<one of the tools above>\", "
                "\"params\": {\"<param>\": <value>, ...}}\n"
                "```\n\n"
                "Rules: (1) one JSON block per response, no other code blocks; "
                "(2) `tool_name` must exactly match a tool listed above; "
                "(3) `params` must contain every required parameter for that "
                "tool with numeric values as JSON numbers (not strings); "
                "(4) any reasoning may precede the JSON block but must NOT be "
                "inside it. "
                "(5) You MUST emit the JSON block on EVERY response, even if "
                "the observation looks similar to a previous step — never reply "
                "with reasoning alone or 'no action'."
            )
            # Optional few-shot rescue for instruct-tuned models (e.g. Gemma)
            # that emit a valid JSON on step 1 but drop the format in
            # subsequent multi-turn observations. Opt-in via SILR_FEWSHOT=1 so
            # baseline cross-family runs (DSR1-Llama-8B, default Gemma) remain
            # unchanged for already-published results.
            if os.environ.get("SILR_FEWSHOT", "").strip() in ("1", "true", "True"):
                base_prompt = base_prompt + "\n\n" + (
                    "## Example response shape (illustrative only — your actual "
                    "tool names and parameters come from the schema above)\n\n"
                    "User observation: <observation showing a stressed bus / "
                    "branch / storage>\n\n"
                    "Your response:\n\n"
                    "The stressed bus voltage is below v_min; I will reduce a "
                    "nearby renewable generator's active power to relieve the "
                    "over-supply.\n\n"
                    "```json\n"
                    "{\"tool_name\": \"<actual tool from list>\", "
                    "\"params\": {\"<param>\": <number>, ...}}\n"
                    "```\n\n"
                    "Every subsequent observation you receive should be answered "
                    "with the SAME response shape: brief reasoning + one JSON "
                    "block. Do not abbreviate or skip the JSON block."
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
        # Anti-stall liveness state: track the running-minimum
        # post-action violation count across SAFE_PROGRESS-admitted steps.
        # If we go ``stall_progress_budget`` consecutive SAFE_PROGRESS
        # steps without strictly improving on that minimum, terminate.
        stall_progress_budget = self._config.stall_progress_budget
        min_post_viol = None
        stall_streak = 0
        stall_breakout = False

        for step_num in range(1, self._config.max_steps + 1):
            # 1. Observe
            obs = self._observer.observe()

            # Check if already recovered
            if obs.is_stable:
                record = StepRecord(
                    step_number=step_num,
                    observation=obs,
                    pre_penalty=_manager_penalty(self._manager),
                    post_penalty=_manager_penalty(self._manager),
                    outcome=StepOutcome.RECOVERED,
                )
                result.steps.append(record)
                result.recovered = True
                break

            # 2. Build user message with observation.
            # The previous step ends on a `user` turn (verifier feedback, e.g.
            # "[SiLR APPROVED]..."), so naively appending the next observation
            # as another `user` turn produces two consecutive user messages.
            # Permissive chat templates (Qwen, Llama) accept this, but strict
            # ones (Gemma) raise "Conversation roles must alternate" → the
            # serving backend returns HTTP 400 and the model is never called
            # (observed: Gemma-3/4 step-2..N all 400, mistaken for 0/9 model
            # failure). Merge into the trailing user turn to keep strict
            # alternation. Content is identical for permissive models.
            user_msg = self._build_observation_message(step_num, obs)
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] = (
                    messages[-1]["content"] + "\n\n" + user_msg
                )
            else:
                messages.append({"role": "user", "content": user_msg})

            # 3. Propose + Verify loop
            record = StepRecord(
                step_number=step_num,
                observation=obs,
                pre_penalty=_manager_penalty(self._manager),
            )
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

                # Optional raw-response instrumentation (SILR_LOG_RAW=1).
                # Diagnoses elicitation confound: distinguishes "model emitted
                # no JSON" from "model emitted JSON but parser missed it", and
                # whether native tool_calls fired. Default off — zero effect on
                # data; emits one structured log line per proposal.
                if os.environ.get("SILR_LOG_RAW", "").strip() in ("1", "true", "True"):
                    _rc = response.content or ""
                    logger.warning(
                        "[RAW] step=%d prop=%d finish=%s n_tool_calls=%d "
                        "content_len=%d content_head=%r",
                        step_num, proposal_idx, response.finish_reason,
                        len(response.tool_calls), len(_rc), _rc[:400],
                    )

                # Parse action
                try:
                    thought, action = self._parser.parse(response)
                    thought = _clean_thought(thought)
                    record.thought = thought
                    if os.environ.get("SILR_LOG_RAW", "").strip() in ("1", "true", "True"):
                        logger.warning(
                            "[RAW] step=%d prop=%d PARSE_OK tool_name=%s",
                            step_num, proposal_idx, action.get("tool_name"),
                        )
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

                    # Admit both terminal-PASS (zero violations) and
                    # SAFE_PROGRESS (recoverability-preserving step under
                    # the ``progress`` gating policy). Both lead to apply
                    # + advance; episode termination is still gated on
                    # ``observation.is_stable`` (terminal recovery).
                    if vr.verdict in (Verdict.PASS, Verdict.SAFE_PROGRESS):
                        apply_result = self._apply_action(action)
                        record.applied_action = action
                        record.tool_result = apply_result
                        record.outcome = StepOutcome.SUCCESS
                        action_applied = True
                        consecutive_step_fails = 0

                        # Unified APPROVED feedback for both PASS and SAFE_PROGRESS.
                        # The 4-cell multi_1 contrast (2026-05-25 panel) showed the
                        # ADMITTED wording induces a conservatism bias in Qwen3-14B:
                        # the LLM reads "verifier-admitted" as "barely accepted, take
                        # smaller next step" and converges in 6+ actions instead of 2.
                        # Opt-in `SILR_SAFE_PROGRESS_DISTINCT_FEEDBACK=1` keeps the
                        # ADMITTED branch available for studies that explicitly want
                        # to compare PASS vs SAFE_PROGRESS context signaling.
                        if (
                            os.environ.get("SILR_SAFE_PROGRESS_DISTINCT_FEEDBACK")
                            and vr.verdict == Verdict.SAFE_PROGRESS
                        ):
                            feedback = (
                                "[SiLR ADMITTED] Action applied as a verifier-admitted "
                                "recovery step."
                            )
                        else:
                            feedback = "[SiLR APPROVED] Action applied successfully."
                        messages.append({
                            "role": "assistant",
                            "content": f"Thought: {thought}\nAction: {_format_action(action)}",
                        })
                        messages.append({
                            "role": "user",
                            "content": feedback,
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
                    # NoVerify mode — apply directly.
                    # Observer trace: when observe_verification is set, compute Φ
                    # passively for the mechanism metrics WITHOUT gating (the action
                    # is applied regardless of verdict). Mirrors the gated call order
                    # (verify() has no persistent side effect on the manager), so the
                    # greedy ungated trajectory and recovery outcome are unchanged.
                    if self._config.observe_verification:
                        record.verification_results.append(self._verifier.verify(action))
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
            record.post_penalty = _manager_penalty(self._manager)

            # Anti-stall liveness check: count consecutive SAFE_PROGRESS
            # admissions that do not strictly reduce the running-minimum
            # outstanding violation count. The intent is to defeat the
            # denial-of-recovery / stall attacker who games the
            # single-step progress gate with infinitesimal improvements
            # that never reach terminal recovery.
            if stall_progress_budget is not None and record.verification_results:
                vr_last = record.verification_results[-1]
                if (
                    vr_last.verdict == Verdict.SAFE_PROGRESS
                    and record.applied_action is not None
                ):
                    post_viol = sum(
                        len(cr.violations) for cr in vr_last.check_results
                        if not cr.passed
                    )
                    if min_post_viol is None or post_viol < min_post_viol:
                        min_post_viol = post_viol
                        stall_streak = 0
                    else:
                        stall_streak += 1
                        if stall_streak >= stall_progress_budget:
                            record.outcome = StepOutcome.STALL
                            result.stall_terminated = True
                            stall_breakout = True
                            logger.info(
                                "Stall budget exceeded after step %d "
                                "(%d consecutive SAFE_PROGRESS without "
                                "violation-count improvement); "
                                "terminating episode",
                                step_num, stall_streak,
                            )
                else:
                    # Any non-SAFE_PROGRESS outcome (PASS recovery, FAIL,
                    # ERROR) resets the stall streak. PASS exits naturally
                    # via the is_stable check next iteration.
                    stall_streak = 0
                    min_post_viol = None

            if stall_breakout:
                break

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


def _manager_penalty(manager: BaseSystemManager) -> float | None:
    value = getattr(manager, "last_penalty", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
