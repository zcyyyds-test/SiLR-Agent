# SiLR-Agent

[![CI](https://github.com/zcyyyds-test/SiLR-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/zcyyyds-test/SiLR-Agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Simulation-in-the-Loop Reasoning for verified LLM agent actions.**

*Any system with state, a solver, and constraints can have a verified LLM agent.*

SiLR clones the system state before every action, executes the proposal on the shadow copy, runs the domain solver, and checks constraint satisfaction — rejecting unsafe actions before they reach the real system.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│            Coordinator (optional)                    │
│  LLM-driven dispatch of specialist agents           │
├──────────────┬──────────────┬───────────────────────┤
│ Specialist A │ Specialist B │  ...                   │
│  (ReAct)     │  (ReAct)     │                        │
├──────────────┴──────────────┴───────────────────────┤
│                  SiLR Verifier                       │
│  shadow copy → execute → solve → check constraints   │
├─────────────────────────────────────────────────────┤
│              Domain Tools & Checkers                 │
├─────────────────────────────────────────────────────┤
│              Domain Environment                      │
│  (any system with state + solver + constraints)      │
└─────────────────────────────────────────────────────┘
```

**Verification pipeline:**
1. **Clone** — `create_shadow_copy()` produces an independent state snapshot
2. **Execute** — the proposed action runs on the shadow copy only
3. **Solve** — `run_pflow()` re-solves the system
4. **Check** — all registered `ConstraintChecker`s evaluate the new state
5. **Verdict** — PASS / FAIL / ERROR

## Why SiLR?

- **Safety guarantee** — every action is pre-verified on a shadow copy before reaching the real system
- **Domain-extensible** — any system with state + solver + constraints can plug in (power grid, GPU cluster, network, thermal, ...)
- **Multi-agent coordination** — LLM coordinator dispatches specialist agents, each with restricted tools, while the verifier enforces global safety
- **Training-ready** — verified trajectories feed directly into SFT / DPO / GRPO pipelines

## Installation

```bash
pip install -e .            # Core framework (zero dependencies)
pip install -e '.[agent]'   # + LLM agent support (OpenAI)
pip install -e '.[grid]'    # + power grid domain (ANDES)
pip install -e '.[training]' # + training (PyTorch + HuggingFace)
pip install -e '.[all]'     # Everything
```

## Quick Start

### Verify an Action (Zero Dependencies)

```python
from domains.network import NetworkManager, build_network_domain_config
from silr.verifier import SiLRVerifier, Verdict

manager = NetworkManager()          # 5-node network topology
config = build_network_domain_config()

manager.fail_link(1, 2)
manager.run_pflow()

verifier = SiLRVerifier(manager, domain_config=config)
result = verifier.verify(
    {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}},
)
print(result.verdict)          # Verdict.PASS
print(result.pflow_converged)  # True
```

### Run the ReAct Agent

```python
from silr.agent import ReActAgent, AgentConfig
from silr.agent.llm.openai_client import OpenAIClient

agent = ReActAgent(
    manager=manager,
    verifier=verifier,
    llm_client=OpenAIClient(model="gpt-4o"),
    domain_config=config,
    config=AgentConfig(max_steps=5),
)
result = agent.run_episode(scenario_id="scenario_01")
print(f"Recovered: {result.recovered}, Steps: {result.total_steps}")
```

### Multi-Agent Coordinator

For cascading faults where constraints conflict, the coordinator dispatches specialist agents — each limited to a subset of tools — while the verifier enforces global safety:

```python
from silr.agent import CoordinatorAgent, CoordinatorConfig, SpecialistSpec
from domains.network import (
    NetworkManager, NetworkScenarioLoader,
    build_network_domain_config,
    build_connectivity_specialist_config,
    build_utilization_specialist_config,
)
from silr.verifier import SiLRVerifier

# Set up a cascading fault scenario
manager = NetworkManager()
loader = NetworkScenarioLoader()
loader.setup_episode(manager, loader.load("cascade_hard"))

full_config = build_network_domain_config()
verifier = SiLRVerifier(manager, domain_config=full_config)

# Define specialists (each is a standard ReActAgent with restricted tools)
specialists = [
    SpecialistSpec(name="connectivity", domain_config=build_connectivity_specialist_config()),
    SpecialistSpec(name="utilization", domain_config=build_utilization_specialist_config()),
]

coordinator = CoordinatorAgent(
    manager=manager,
    verifier=verifier,
    llm_client=llm_client,  # any BaseLLMClient
    specialists=specialists,
    full_domain_config=full_config,
    config=CoordinatorConfig(max_rounds=6),
)
result = coordinator.run_episode(scenario_id="cascade_hard")
print(f"Recovered: {result.recovered}, Rounds: {result.total_rounds}, Conflicts: {result.conflict_count}")
```

The coordinator observes full system state each round, asks the LLM which specialist to activate, then compares pre/post observations to detect cross-constraint conflicts.

### Power Grid Domain

A reference power grid implementation is included under `domains/grid/`, built on the [ANDES](https://docs.andes.app/) simulator.

### GPU Cluster Scheduling Domain

A reference GPU cluster scheduling domain is included under `domains/cluster/`, with a complete SFT → GRPO post-training pipeline using the SiLR verifier as the reward signal.

A trained Qwen3-14B + LoRA agent achieves **94.1% recovery** across 51 evaluation episodes (vs 88.2% for the SFT baseline and 67% for the GPT-5.4 teacher), demonstrating that verifier-gated reward signals can improve LLM agent reliability beyond imitation learning. See [`domains/cluster/README.md`](domains/cluster/README.md) for the full case study, training pipeline, and application context.

## Add Your Own Domain

Four components:

### 1. System Manager

```python
from silr.core.interfaces import BaseSystemManager

class MyManager(BaseSystemManager):
    @property
    def sim_time(self) -> float: ...
    @property
    def base_mva(self) -> float: return 1.0
    @property
    def system_state(self) -> dict: ...
    def run_pflow(self) -> bool: ...
    def create_shadow_copy(self) -> "MyManager": ...
```

### 2. Constraint Checkers

```python
from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult

class TemperatureChecker(BaseConstraintChecker):
    @property
    def name(self) -> str:
        return "temperature_limits"

    def check(self, state: dict, base_mva: float) -> CheckResult:
        violations = [...]  # check state against limits
        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={"max_temp": ...},
            violations=violations,
        )
```

### 3. Domain Tools

```python
from silr.tools.base import BaseTool

class AdjustCoolingTool(BaseTool):
    name = "adjust_cooling"
    description = "Adjust cooling power for a thermal zone."

    def _validate_params(self, zone_id: str = "", delta_kw: float = 0, **kw):
        if not zone_id:
            raise ValidationError("zone_id is required")

    def _run(self, zone_id: str = "", delta_kw: float = 0, **kw) -> dict:
        self.manager.set_cooling(zone_id, delta_kw)
        return {"adjusted": True, "zone_id": zone_id}
```

### 4. Domain Config

```python
from silr.core.config import DomainConfig

def build_my_domain_config():
    return DomainConfig(
        domain_name="thermal_plant",
        checkers=[TemperatureChecker(), PressureChecker()],
        allowed_actions=frozenset(["adjust_cooling", "open_valve"]),
        create_toolset=create_my_toolset,
    )
```

The SiLR verifier, ReAct agent, coordinator, and training pipeline all work with your new domain automatically.

## Project Structure

```
silr/                    # Framework core
├── core/                # ABCs: BaseSystemManager, BaseConstraintChecker, DomainConfig
├── tools/               # BaseTool ABC
├── verifier/            # SiLRVerifier — shadow-copy verification pipeline
├── agent/               # ReAct loop, CoordinatorAgent, LLM clients
│   ├── coordinator.py   # Multi-agent coordinator + specialist dispatch
│   └── react_loop.py    # Single-agent ReAct loop (reused as specialist)
├── training/            # SFT/DPO trainers, reward computation
└── eval/                # EvalRunner, MultiAgentEvalRunner, metrics

domains/                 # Reference implementations
├── network/             # 5-node network (zero dependencies)
│   ├── scenarios.py     # Cascading fault scenarios
│   └── specialists.py   # Specialist agent configs
├── grid/                # Power grid domain (requires ANDES)
└── cluster/             # GPU cluster scheduling (SFT + GRPO case study)
    ├── README.md        # Full case study and training pipeline
    ├── manager.py       # ClusterManager: state, transitions, shadow copy
    ├── observation.py   # Compressed JSON observation builder
    ├── scenarios/       # 17 failure scenarios across 6 categories
    └── checkers.py      # ResourceCapacity, Affinity, RackSpread, Priority, Queue

examples/                # Runnable demos
tests/                   # pytest suite
```

## Affiliation

Developed as part of doctoral research on **LLM agents for industrial applications** at Institute of Science Tokyo (formerly Tokyo Institute of Technology).

## License

MIT

## Citation

```bibtex
@misc{silr-agent,
  author = {Chenyu Zhou},
  title  = {SiLR-Agent: Simulation-in-the-Loop Reasoning for LLM Agents},
  year   = {2026},
  url    = {https://github.com/zcyyyds-test/SiLR-Agent}
}
```
