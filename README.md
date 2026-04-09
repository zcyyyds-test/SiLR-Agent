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
3. **Solve** — the domain's solver re-computes the new state
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
manager.solve()

verifier = SiLRVerifier(manager, domain_config=config)
result = verifier.verify(
    {"tool_name": "restore_link", "params": {"src": 1, "dst": 2}},
)
print(result.verdict)          # Verdict.PASS
print(result.solver_converged)  # True
```

### Run the ReAct Agent

```python
from domains.network import (
    NetworkManager,
    NetworkScenarioLoader,
    build_network_domain_config,
)
from silr.agent import ReActAgent, AgentConfig
from silr.agent.llm.openai_client import OpenAIClient
from silr.verifier import SiLRVerifier

# Inject a cascading fault: link 1-2 down + link 2-5 near overload
manager = NetworkManager()
loader = NetworkScenarioLoader()
scenario = loader.load("cascade_easy")
loader.setup_episode(manager, scenario)

# Agent needs an observer to see live violations — pass with_observer=True
config = build_network_domain_config(with_observer=True)
verifier = SiLRVerifier(manager, domain_config=config)

agent = ReActAgent(
    manager=manager,
    verifier=verifier,
    llm_client=OpenAIClient(model="gpt-4o"),
    domain_config=config,
    config=AgentConfig(max_steps=5),
)
result = agent.run_episode(scenario_id="cascade_easy")
print(f"Recovered: {result.recovered}, Steps: {result.total_steps}")
```

### Multi-Agent Coordinator

For cascading faults where constraints conflict, the coordinator dispatches specialist agents — each limited to a subset of tools — while the verifier enforces global safety:

```python
from domains.network import (
    NetworkManager,
    NetworkScenarioLoader,
    build_network_domain_config,
    build_connectivity_specialist_config,
    build_utilization_specialist_config,
)
from silr.agent import CoordinatorAgent, CoordinatorConfig, SpecialistSpec
from silr.agent.llm.openai_client import OpenAIClient
from silr.verifier import SiLRVerifier

# Inject a cascading fault where restoring the link worsens an existing overload
manager = NetworkManager()
loader = NetworkScenarioLoader()
scenario = loader.load("cascade_hard")
loader.setup_episode(manager, scenario)

# CoordinatorAgent requires an observer — pass with_observer=True
full_config = build_network_domain_config(with_observer=True)
verifier = SiLRVerifier(manager, domain_config=full_config)

specialists = [
    SpecialistSpec(name="connectivity", domain_config=build_connectivity_specialist_config()),
    SpecialistSpec(name="utilization",  domain_config=build_utilization_specialist_config()),
]
coordinator = CoordinatorAgent(
    manager=manager,
    verifier=verifier,
    llm_client=OpenAIClient(model="gpt-4o"),
    specialists=specialists,
    full_domain_config=full_config,
    config=CoordinatorConfig(max_rounds=4, max_specialist_steps=3),
)
result = coordinator.run_episode(scenario_id="cascade_hard")
```

Each round, the coordinator observes full system state, asks the LLM which specialist to dispatch, then compares pre/post observations to detect cross-constraint conflicts. See `examples/network_cascading.py` for a complete runnable example.

### Power Grid Domain

A reference power grid domain is included under `domains/grid/`, built on the [ANDES](https://docs.andes.app/) transient stability solver. Includes 4 constraint checkers (voltage, frequency, line loading, transient stability), a tool suite for generation, load, and breaker actions, and a scenario library covering N-1 contingencies and cascading faults on the IEEE 39-bus benchmark.

A fine-tuned Qwen3-14B + LoRA agent achieves **97.0% recovery** across 66 fault scenarios, surpassing GPT-5.4 few-shot (94.2%) and zero-shot (82.1%) — showing that verifier-gated SFT alone can carry a domain-specialized open model past frontier baselines on physics-constrained recovery tasks. The same SiLR verifier framework that gates GPU cluster scheduling decisions also gates grid control actions, illustrating the architecture's domain-extensibility. See [`domains/grid/README.md`](domains/grid/README.md) for setup and the runnable example.

### GPU Cluster Scheduling Domain

A reference GPU cluster scheduling domain is included under `domains/cluster/`, with a complete SFT → GRPO post-training pipeline using the SiLR verifier as the reward signal.

A trained Qwen3-14B + LoRA agent achieves **94.1% recovery** across 51 evaluation episodes (vs 88.2% for the SFT baseline and 67% for the GPT-5.4 teacher), demonstrating that verifier-gated reward signals can improve LLM agent reliability beyond imitation learning. See [`domains/cluster/README.md`](domains/cluster/README.md) for the full case study, training pipeline, and application context.

## Add Your Own Domain

A new domain plugs in by implementing four core abstractions:

| Component | Responsibility |
|-----------|----------------|
| `BaseSystemManager` (silr.core) | State, `create_shadow_copy()`, `solve()` |
| `BaseConstraintChecker` (silr.core) | Inspect post-action state, return pass/fail + violations |
| `BaseTool` (silr.tools) | Execute a single action against the manager |
| `DomainConfig` (silr.core.config) | Bundle checkers, tools, and prompts for injection |

A minimal manager looks like this:

```python
from silr.core.interfaces import BaseSystemManager

class MyManager(BaseSystemManager):
    @property
    def sim_time(self) -> float: ...
    @property
    def base_mva(self) -> float: return 1.0
    @property
    def system_state(self) -> dict: ...
    def solve(self) -> bool: ...
    def create_shadow_copy(self) -> "MyManager": ...
```

Once your `DomainConfig` is built, the SiLR verifier, ReAct agent, coordinator, and training pipeline all work with your new domain automatically. See `domains/network/` for a minimal reference implementation, and `domains/cluster/` or `domains/grid/` for richer examples.

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
