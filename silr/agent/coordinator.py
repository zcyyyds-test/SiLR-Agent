"""Multi-agent coordinator: LLM-driven specialist dispatch.

The coordinator observes full system state and decides which specialist
agent to activate. Each specialist is a standard ReActAgent with a
restricted DomainConfig (subset of tools/checkers). The SiLRVerifier
uses the full checker set for global safety verification.

Specialists communicate state changes through the shared manager —
no explicit message passing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import AgentConfig
from .multi_types import MultiAgentEpisodeResult, SpecialistActivation
from .observation import BaseObserver
from .react_loop import ReActAgent
from .types import Observation
from .llm.base import BaseLLMClient
from silr.core.config import DomainConfig
from silr.core.interfaces import BaseSystemManager
from silr.verifier import SiLRVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoordinatorConfig:
    """Configuration for the multi-agent coordinator."""

    max_rounds: int = 6
    max_specialist_steps: int = 4
    max_proposals_per_step: int = 3
    enable_verification: bool = True
    temperature: float = 0.0
    seed: int | None = 42
    verification_feedback_mode: str = "detailed"


@dataclass
class SpecialistSpec:
    """Specification for a specialist agent."""

    name: str
    domain_config: DomainConfig
    system_prompt_suffix: str = ""


class CoordinatorAgent:
    """LLM-driven coordinator that dispatches specialist ReActAgents.

    Each round:
    1. Observe full system state
    2. Ask coordinator LLM which specialist to dispatch
    3. Create and run a ReActAgent with the specialist's DomainConfig
    4. Compare pre/post observations to detect constraint changes
    5. Repeat until stable or max_rounds
    """

    def __init__(
        self,
        manager: BaseSystemManager,
        verifier: SiLRVerifier,
        llm_client: BaseLLMClient,
        specialists: list[SpecialistSpec],
        full_domain_config: DomainConfig,
        config: CoordinatorConfig = CoordinatorConfig(),
        specialist_llm_client: BaseLLMClient | None = None,
    ):
        self._manager = manager
        self._verifier = verifier
        self._llm = llm_client
        self._specialist_llm = specialist_llm_client or llm_client
        self._specialists = {s.name: s for s in specialists}
        self._full_config = full_domain_config
        self._config = config

        if full_domain_config.create_observer:
            self._observer: BaseObserver = full_domain_config.create_observer(manager)
        else:
            logger.warning(
                "No observer provided in full_domain_config. "
                "Using fallback that always reports unstable — "
                "coordinator requires a real observer to function."
            )
            self._observer = _FallbackObserver()

    def run_episode(self, scenario_id: str = "unknown") -> MultiAgentEpisodeResult:
        """Run a multi-agent coordinator episode."""
        result = MultiAgentEpisodeResult(scenario_id=scenario_id)
        history: list[dict] = []

        for round_num in range(1, self._config.max_rounds + 1):
            # 1. Observe
            obs = self._observer.observe()

            if obs.is_stable:
                result.recovered = True
                result.final_observation = obs
                break

            # 2. Ask coordinator which specialist to dispatch
            dispatch = self._decide_dispatch(obs, history)

            if dispatch is None or dispatch.get("action") == "done":
                result.final_observation = obs
                break

            specialist_name = dispatch.get("specialist", "")
            thought = dispatch.get("reason", "")

            if specialist_name not in self._specialists:
                logger.warning(
                    f"Coordinator requested unknown specialist '{specialist_name}', "
                    f"available: {list(self._specialists.keys())}"
                )
                result.error = f"Unknown specialist: {specialist_name}"
                result.final_observation = obs
                break

            # 3. Run specialist
            spec = self._specialists[specialist_name]
            pre_obs = obs

            agent_config = AgentConfig(
                max_steps=self._config.max_specialist_steps,
                max_proposals_per_step=self._config.max_proposals_per_step,
                enable_verification=self._config.enable_verification,
                temperature=self._config.temperature,
                seed=self._config.seed,
                verification_feedback_mode=self._config.verification_feedback_mode,
            )

            specialist_agent = ReActAgent(
                manager=self._manager,
                verifier=self._verifier,
                llm_client=self._specialist_llm,
                domain_config=spec.domain_config,
                config=agent_config,
            )

            logger.info(f"Round {round_num}: dispatching '{specialist_name}'")
            ep_result = specialist_agent.run_episode(
                scenario_id=f"{scenario_id}_r{round_num}_{specialist_name}",
            )

            # 4. Post-observation and constraint change detection
            post_obs = self._observer.observe()
            improved, worsened = self._detect_constraint_changes(pre_obs, post_obs)

            activation = SpecialistActivation(
                specialist_name=specialist_name,
                round_number=round_num,
                coordinator_thought=thought,
                episode_result=ep_result,
                pre_observation=pre_obs,
                post_observation=post_obs,
                constraints_improved=improved,
                constraints_worsened=worsened,
            )
            result.activations.append(activation)
            result.total_specialist_steps += ep_result.total_steps
            result.total_proposals += ep_result.total_proposals
            result.total_rejections += ep_result.total_rejections

            # Update history for next round
            history.append({
                "round": round_num,
                "specialist": specialist_name,
                "improved": improved,
                "worsened": worsened,
                "steps": ep_result.total_steps,
                "recovered_by_specialist": ep_result.recovered,
            })

            logger.info(
                f"Round {round_num} done: improved={improved}, worsened={worsened}"
            )

        # Final check
        if result.final_observation is None:
            result.final_observation = self._observer.observe()
        result.recovered = result.final_observation.is_stable
        result.total_rounds = len(result.activations)

        return result

    def _decide_dispatch(
        self, obs: Observation, history: list[dict],
    ) -> dict | None:
        """Ask coordinator LLM which specialist to dispatch."""
        prompt = self._build_coordinator_prompt(obs, history)
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._llm.chat(
                messages=messages,
                temperature=self._config.temperature,
                seed=self._config.seed,
            )
        except Exception as e:
            logger.error(f"Coordinator LLM call failed: {e}")
            return None

        return self._parse_dispatch(response.content)

    def _build_system_prompt(self) -> str:
        specialist_desc = "\n".join(
            f"- {name}: actions={sorted(s.domain_config.allowed_actions)}, "
            f"handles: {', '.join(c.name for c in s.domain_config.checkers)}"
            for name, s in self._specialists.items()
        )
        return (
            "You are a multi-agent coordinator for system fault recovery.\n\n"
            f"Available specialists:\n{specialist_desc}\n\n"
            "Each round, analyze the current system state and decide which "
            "specialist to dispatch next. Respond with JSON:\n"
            '{"specialist": "<name>", "reason": "..."}\n'
            "Or if the system is stable or no further action helps:\n"
            '{"action": "done", "reason": "..."}'
        )

    def _build_coordinator_prompt(
        self, obs: Observation, history: list[dict],
    ) -> str:
        parts = [f"Current system state:\n{obs.compressed_json}"]

        if obs.violations:
            parts.append(f"\n{len(obs.violations)} active violation(s):")
            for v in obs.violations[:5]:
                parts.append(f"  - [{v.get('severity', '?')}] {v.get('detail', v)}")

        if history:
            parts.append("\nPrevious rounds:")
            for h in history:
                improved = ", ".join(h["improved"]) or "none"
                worsened = ", ".join(h["worsened"]) or "none"
                parts.append(
                    f"  Round {h['round']}: dispatched '{h['specialist']}' "
                    f"({h['steps']} steps) → improved: {improved}, worsened: {worsened}"
                )

        parts.append("\nWhich specialist should act next?")
        return "\n".join(parts)

    def _parse_dispatch(self, text: str) -> dict | None:
        """Parse coordinator LLM response into dispatch decision."""
        if not text:
            return None

        # Try JSON block
        m = re.search(r'```json\s*\n?(.*?)\n?\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try bare JSON
        m = re.search(r'\{[^{}]*"(?:specialist|action)"[^{}]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def _detect_constraint_changes(
        self, pre: Observation, post: Observation,
    ) -> tuple[list[str], list[str]]:
        """Compare pre/post observations to find improved/worsened constraints."""
        pre_violations = _group_violations(pre.violations)
        post_violations = _group_violations(post.violations)

        all_types = set(pre_violations.keys()) | set(post_violations.keys())
        improved = []
        worsened = []

        for ct in all_types:
            pre_count = len(pre_violations.get(ct, []))
            post_count = len(post_violations.get(ct, []))
            if post_count < pre_count:
                improved.append(ct)
            elif post_count > pre_count:
                worsened.append(ct)

        return improved, worsened


def _group_violations(violations: list[dict]) -> dict[str, list[dict]]:
    """Group violations by constraint type."""
    groups: dict[str, list[dict]] = {}
    for v in violations:
        ct = v.get("type", "unknown")
        groups.setdefault(ct, []).append(v)
    return groups


class _FallbackObserver(BaseObserver):
    """Fallback when domain doesn't provide an observer."""

    def observe(self) -> Observation:
        return Observation(
            raw={}, compressed_json="{}",
            violations=[], is_stable=False,
        )
