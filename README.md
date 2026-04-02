# SiLR-Agent

[![CI](https://github.com/zcyyyds-test/SiLR-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/zcyyyds-test/SiLR-Agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Simulation-in-the-Loop Reasoning framework that integrates domain knowledge into LLM agent decision loops.**

*Any domain with a simulator can have a verified LLM agent.*

SiLR-Agent integrates domain simulators and constraint knowledge into LLM-driven decision making. Before any action reaches the real system, SiLR clones the simulator state, executes the proposed action on the shadow copy, runs the domain solver, and checks constraint satisfaction — rejecting unsafe actions before they cause damage.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   ReAct Agent                       │
│  (LLM reasoning loop with bounded retries)          │
├─────────────────────────────────────────────────────┤
│                  SiLR Verifier                      │
│  shadow copy → execute → solve → check constraints  │
├─────────────────────────────────────────────────────┤
│               Domain Tools & Checkers               │
│  (actions, observations, constraint checkers)        │
├─────────────────────────────────────────────────────┤
│              Domain Simulator                       │
│  (any system with state + solver + constraints)      │
└─────────────────────────────────────────────────────┘
```

The verification pipeline:
1. **Clone** — `create_shadow_copy()` produces an independent simulator snapshot
2. **Execute** — the proposed action runs on the shadow copy only
3. **Solve** — `run_pflow()` (and optional `post_solve_hook`) re-solves the system
4. **Check** — all registered `ConstraintChecker`s evaluate the new state
5. **Verdict** — PASS (all checks green), FAIL (violations found), or ERROR (solver diverged)

## Why SiLR?

- **Safety guarantee**: every action is pre-verified on a shadow copy before reaching the real system
- **Domain-extensible**: any simulator with state + solver + constraints can plug in
- **Training-ready**: verified trajectories feed directly into SFT / DPO / GRPO pipelines

## Installation

```bash
# Core framework (zero dependencies)
pip install -e .

# With LLM agent support
pip install -e '.[agent]'

# With power grid domain (requires ANDES simulator)
pip install -e '.[grid]'

# With training support (PyTorch + HuggingFace)
pip install -e '.[training]'

# Everything
pip install -e '.[all]'
```

## Quick Start

### Network Domain (Zero Dependencies)

The toy network domain ships with SiLR as a self-contained demo — no external packages needed.

```python
from domains.network import NetworkManager, build_network_domain_config
from silr.verifier import SiLRVerifier, Verdict

# 1. Set up domain
manager = NetworkManager()          # 5-node network topology
config = build_network_domain_config()

# 2. Inject a fault
manager.fail_link(1, 2)
manager.run_pflow()

# 3. Verify a recovery action (shadow-copy verification)
verifier = SiLRVerifier(manager, domain_config=config)
result = verifier.verify(
    {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}},
)

print(result.verdict)        # Verdict.PASS
print(result.pflow_converged)  # True
```

### Power Grid Domain

A reference power grid implementation is included under `domains/grid/`, built on the [ANDES](https://docs.andes.app/) simulator. See `domains/grid/` for the full integration.

### Running the ReAct Agent

```python
from silr.agent import ReActAgent, AgentConfig
from silr.agent.llm.openai_client import OpenAIClient

agent = ReActAgent(
    manager=manager,
    verifier=verifier,
    llm_client=OpenAIClient(model="gpt-4o"),
    config=AgentConfig(max_steps=5),
    domain_config=config,
)
result = agent.run_episode(scenario_id="scenario_01")
print(f"Recovered: {result.recovered}, Steps: {result.total_steps}")
```

## Multi-Agent Coordinator

For cascading faults where multiple constraints conflict, SiLR supports a multi-agent coordinator that dispatches specialist agents:

```python
from silr.agent import CoordinatorAgent, CoordinatorConfig, SpecialistSpec
from domains.network import (
    build_network_domain_config,
    build_connectivity_specialist_config,
    build_utilization_specialist_config,
)

specialists = [
    SpecialistSpec(name="connectivity", domain_config=build_connectivity_specialist_config()),
    SpecialistSpec(name="utilization", domain_config=build_utilization_specialist_config()),
]

coordinator = CoordinatorAgent(
    manager=manager,
    verifier=verifier,
    llm_client=llm_client,
    specialists=specialists,
    full_domain_config=build_network_domain_config(),
    config=CoordinatorConfig(max_rounds=6),
)
result = coordinator.run_episode(scenario_id="cascade_hard")
print(f"Recovered: {result.recovered}, Rounds: {result.total_rounds}")
```

Each specialist is a standard `ReActAgent` with a restricted `DomainConfig` (subset of tools). The coordinator observes the full system state, dispatches specialists via LLM reasoning, and detects cross-constraint conflicts by comparing pre/post observations.

## Add Your Own Domain

Implementing a new domain requires four components:

### Step 1: System Manager

Subclass `BaseSystemManager` with your simulator's lifecycle:

```python
from silr.core.interfaces import BaseSystemManager

class MyManager(BaseSystemManager):
    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def base_mva(self) -> float:
        return 1.0  # use 1.0 if your domain has no per-unit system

    @property
    def system_state(self) -> dict:
        """Current state snapshot for constraint checkers."""
        ...

    def run_pflow(self) -> bool:
        """Run steady-state solver. Return True if converged."""
        ...

    def create_shadow_copy(self) -> "MyManager":
        """Return an independent deep copy for verification."""
        ...
```

### Step 2: Constraint Checkers

Define what "safe" means in your domain:

```python
from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult

class TemperatureChecker(BaseConstraintChecker):
    @property
    def name(self) -> str:
        return "temperature_limits"

    def check(self, state: dict, base_mva: float) -> CheckResult:
        violations = []
        for zone in state["thermal_zones"]:
            if zone["temp_c"] > 85.0:
                violations.append(...)
        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={"max_temp": max(z["temp_c"] for z in state["thermal_zones"])},
            violations=violations,
        )
```

### Step 3: Domain Tools

Wrap simulator actions as `BaseTool` instances:

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

### Step 4: Domain Config

Bundle everything into a `DomainConfig`:

```python
from silr.core.config import DomainConfig

def build_my_domain_config():
    return DomainConfig(
        domain_name="thermal_plant",
        checkers=[TemperatureChecker(), PressureChecker()],
        allowed_actions=frozenset(["adjust_cooling", "open_valve"]),
        create_toolset=create_my_toolset,  # callable: manager → {name: tool}
    )
```

That's it. The SiLR verifier, ReAct agent, and training pipeline all work with your new domain automatically.

## Project Structure

```
silr/                    # Framework core (pip install silr)
├── core/                # ABCs: BaseSystemManager, BaseConstraintChecker, DomainConfig
├── tools/               # BaseTool ABC
├── verifier/            # SiLRVerifier + shadow-copy verification pipeline
├── agent/               # ReAct loop, CoordinatorAgent, LLM clients
│   ├── coordinator.py   # Multi-agent coordinator + specialist dispatch
│   └── react_loop.py    # Single-agent ReAct loop (also used as specialist)
├── training/            # SFT/DPO trainers, reward computation
└── eval/                # EvalRunner, MultiAgentEvalRunner, metrics

domains/                 # Reference implementations
├── network/             # Toy 5-node network (zero dependencies)
│   ├── scenarios.py     # Cascading fault scenarios for multi-agent testing
│   └── specialists.py   # Connectivity / utilization specialist configs
└── grid/                # Power grid domain (requires ANDES)

examples/                # Runnable demos
tests/                   # pytest suite
```

## License

MIT

## Citation

If you use SiLR-Agent in your research, please cite:

```bibtex
@misc{silr-agent,
  author = {Chenyu Zhou},
  title  = {SiLR-Agent: Simulation-in-the-Loop Reasoning for LLM Agents},
  year   = {2026},
  url    = {https://github.com/zcyyyds-test/SiLR-Agent}
}
```
