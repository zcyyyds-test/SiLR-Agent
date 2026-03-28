"""Batch evaluation runner — domain-agnostic via factory pattern.

This runner does not import any simulator package directly. Instead, it
accepts factory callables for creating managers and loading scenarios.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from ..core.interfaces import BaseSystemManager
from ..core.config import DomainConfig
from ..agent.config import AgentConfig
from ..agent.react_loop import ReActAgent
from ..agent.types import EpisodeResult
from ..agent.llm.base import BaseLLMClient
from ..agent.trajectory import TrajectoryRecorder
from ..verifier import SiLRVerifier

logger = logging.getLogger(__name__)


class EvalRunner:
    """Run multiple scenarios and collect results.

    Domain-agnostic: the caller provides factory functions for creating
    managers and loading scenarios, rather than hardcoding ANDES paths.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        domain_config: DomainConfig,
        manager_factory: Callable[[], BaseSystemManager],
        scenario_loader: Any,
        config: AgentConfig = AgentConfig(),
        record_trajectories: bool = False,
        few_shot_context: Optional[str] = None,
    ):
        """
        Args:
            llm_client: LLM client for the agent.
            domain_config: Domain configuration bundle.
            manager_factory: Callable that creates a fresh BaseSystemManager.
            scenario_loader: Object with load(id) and load_all() methods.
            config: Agent configuration.
            record_trajectories: Whether to record trajectories for training.
            few_shot_context: Optional few-shot examples to prepend.
        """
        self._llm = llm_client
        self._domain_config = domain_config
        self._manager_factory = manager_factory
        self._loader = scenario_loader
        self._config = config
        self._recorder = TrajectoryRecorder() if record_trajectories else None
        self._results: list[EpisodeResult] = []
        self._few_shot_context = few_shot_context

    def run_scenario(self, scenario: Any) -> EpisodeResult:
        """Run a single scenario and return the result."""
        scenario_id = getattr(scenario, 'id', str(scenario))
        logger.info(f"Running scenario: {scenario_id}")
        t0 = time.perf_counter()

        # Fresh system for each scenario
        manager = self._manager_factory()

        # Apply scenario setup if the loader has a setup_episode method
        if hasattr(self._loader, 'setup_episode'):
            try:
                self._loader.setup_episode(manager, scenario)
            except Exception as e:
                logger.error(f"Scenario setup failed: {e}")
                return EpisodeResult(
                    scenario_id=scenario_id,
                    error=f"Setup failed: {e}",
                )

        # Create verifier
        verifier = SiLRVerifier(
            manager,
            domain_config=self._domain_config,
        )

        # Create failsafe if domain provides one
        failsafe = None
        if self._domain_config.create_failsafe:
            failsafe = self._domain_config.create_failsafe(manager)

        # Create agent
        agent = ReActAgent(
            manager=manager,
            verifier=verifier,
            llm_client=self._llm,
            config=self._config,
            domain_config=self._domain_config,
            failsafe=failsafe,
            trajectory_recorder=self._recorder,
            few_shot_context=self._few_shot_context,
        )

        # Run
        result = agent.run_episode(scenario_id=scenario_id)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Scenario {scenario_id}: recovered={result.recovered}, "
            f"steps={result.total_steps}, time={elapsed:.1f}s"
        )

        self._results.append(result)
        return result

    def run_all(
        self,
        scenario_ids: list[str] | None = None,
    ) -> list[EpisodeResult]:
        """Run all scenarios (or a specified subset).

        Args:
            scenario_ids: Explicit list. If None, loads all from loader.
        """
        if scenario_ids:
            scenarios = [self._loader.load(sid) for sid in scenario_ids]
        else:
            scenarios = self._loader.load_all()

        results = []
        for scenario in scenarios:
            result = self.run_scenario(scenario)
            results.append(result)

        return results

    @property
    def results(self) -> list[EpisodeResult]:
        return self._results

    @property
    def trajectory_recorder(self) -> Optional[TrajectoryRecorder]:
        return self._recorder
