"""SiLR Agent: Bounded ReAct loop with SiLR verification."""

from .react_loop import ReActAgent
from .config import AgentConfig
from .types import StepOutcome, Observation, StepRecord, EpisodeResult
from .action_parser import ActionParser, ParseError
from .coordinator import CoordinatorAgent, CoordinatorConfig, SpecialistSpec
from .multi_types import MultiAgentEpisodeResult, SpecialistActivation

__all__ = [
    "ReActAgent",
    "AgentConfig",
    "StepOutcome",
    "Observation",
    "StepRecord",
    "EpisodeResult",
    "ActionParser",
    "ParseError",
    "CoordinatorAgent",
    "CoordinatorConfig",
    "SpecialistSpec",
    "MultiAgentEpisodeResult",
    "SpecialistActivation",
]
