"""Batch evaluation runner for multi-agent coordinator episodes."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from ..core.interfaces import BaseSystemManager
from ..core.config import DomainConfig
from ..agent.coordinator import CoordinatorAgent, CoordinatorConfig, SpecialistSpec
from ..agent.multi_types import MultiAgentEpisodeResult
from ..agent.multi_trajectory import MultiAgentTrajectoryRecorder
from ..agent.llm.base import BaseLLMClient
from ..verifier import SiLRVerifier

logger = logging.getLogger(__name__)


class MultiAgentEvalRunner:
    """Run cascading fault scenarios with multi-agent coordinator."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        full_domain_config: DomainConfig,
        specialists: list[SpecialistSpec],
        manager_factory: Callable[[], BaseSystemManager],
        scenario_loader: Any,
        config: CoordinatorConfig = CoordinatorConfig(),
        specialist_llm_client: BaseLLMClient | None = None,
        record_trajectories: bool = False,
    ):
        self._llm = llm_client
        self._specialist_llm = specialist_llm_client
        self._full_config = full_domain_config
        self._specialists = specialists
        self._manager_factory = manager_factory
        self._loader = scenario_loader
        self._config = config
        self._recorder = MultiAgentTrajectoryRecorder() if record_trajectories else None
        self._results: list[MultiAgentEpisodeResult] = []

    def run_scenario(self, scenario: Any) -> MultiAgentEpisodeResult:
        scenario_id = getattr(scenario, "id", str(scenario))
        logger.info(f"Running multi-agent scenario: {scenario_id}")
        t0 = time.perf_counter()

        manager = self._manager_factory()

        if hasattr(self._loader, "setup_episode"):
            try:
                self._loader.setup_episode(manager, scenario)
            except Exception as e:
                logger.error(f"Scenario setup failed: {e}")
                return MultiAgentEpisodeResult(
                    scenario_id=scenario_id,
                    error=f"Setup failed: {e}",
                )

        verifier = SiLRVerifier(manager, domain_config=self._full_config)

        coordinator = CoordinatorAgent(
            manager=manager,
            verifier=verifier,
            llm_client=self._llm,
            specialists=self._specialists,
            full_domain_config=self._full_config,
            config=self._config,
            specialist_llm_client=self._specialist_llm,
        )

        result = coordinator.run_episode(scenario_id=scenario_id)
        elapsed = time.perf_counter() - t0
        logger.info(
            f"Scenario {scenario_id}: recovered={result.recovered}, "
            f"rounds={result.total_rounds}, time={elapsed:.1f}s"
        )

        self._results.append(result)
        if self._recorder is not None:
            self._recorder.record_episode(result)

        return result

    def run_all(
        self, scenario_ids: list[str] | None = None,
    ) -> list[MultiAgentEpisodeResult]:
        if scenario_ids:
            scenarios = [self._loader.load(sid) for sid in scenario_ids]
        else:
            scenarios = self._loader.load_all()

        return [self.run_scenario(s) for s in scenarios]

    @property
    def results(self) -> list[MultiAgentEpisodeResult]:
        return self._results

    @property
    def trajectory_recorder(self) -> Optional[MultiAgentTrajectoryRecorder]:
        return self._recorder
