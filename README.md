# SiLR-Agent

**Simulation-in-the-Loop Reasoning framework for LLM agents in safety-critical domains.**

*Any domain with a simulator can have a verified LLM agent.*

SiLR-Agent is a domain-agnostic framework that uses physics simulators as safety nets for LLM-driven decision making. Before any action reaches the real system, SiLR clones the simulator state, executes the proposed action on the shadow copy, runs the domain solver, and checks constraint satisfaction — rejecting unsafe actions before they cause damage.

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
- **Domain-agnostic**: any simulator with state + solver + constraints can plug in
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

## Add Your Own Domain

Implementing a new domain requires four components:

### Step 1: System Manager

Subclass `BaseSystemManager` with your simulator's lifecycle:

```python
from silr.core.interfaces import BaseSystemManager

class MyManager(BaseSystemManager):
    def run_pflow(self) -> bool:
        """Run steady-state solver. Return True if converged."""
        ...

    def create_shadow_copy(self) -> "MyManager":
        """Return an independent deep copy for verification."""
        ...

    @property
    def system_state(self) -> dict:
        """Current state snapshot for constraint checkers."""
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

    def execute(self, zone_id: str, delta_kw: float) -> dict:
        self._manager.set_cooling(zone_id, delta_kw)
        return {"status": "success", "data": {"adjusted": True}}
```

### Step 4: Domain Config

Bundle everything into a `DomainConfig`:

```python
from silr.core.config import DomainConfig

def build_my_domain_config(manager):
    return DomainConfig(
        domain_name="thermal_plant",
        checkers=[TemperatureChecker(), PressureChecker()],
        allowed_actions=frozenset(["adjust_cooling", "open_valve"]),
        toolset=create_my_toolset(manager),
        system_prompt="You are a thermal plant operator...",
        tool_schemas=[...],  # OpenAI function-calling format
    )
```

That's it. The SiLR verifier, ReAct agent, and training pipeline all work with your new domain automatically.

## Project Structure

```
silr/                    # Framework core (pip install silr)
├── core/                # ABCs: BaseSystemManager, BaseConstraintChecker, DomainConfig
├── tools/               # BaseTool ABC
├── verifier/            # SiLRVerifier + shadow-copy verification pipeline
├── agent/               # ReAct loop, ActionParser, LLM clients
├── training/            # SFT/DPO trainers, reward computation
└── eval/                # EvalRunner, metrics

domains/                 # Reference implementations
├── network/             # Toy 5-node network (zero dependencies)
└── grid/                # Power grid domain (requires ANDES)

examples/                # Runnable demos
tests/                   # pytest suite
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Zero forced dependencies for `silr/` | Maximizes installability; domain simulators and LLM clients are optional extras |
| `DomainConfig` is required, not optional | Clean API for a new framework — no legacy compat paths |
| Shadow-copy verification | Non-destructive: original simulator state is never modified |
| `post_solve_hook` for domain-specific solvers | Extra solver passes (e.g. time-domain simulation) are domain concepts, not framework concepts |
| `ActionParser` accepts injectable `allowed_actions` | Decouples parsing from any specific domain's action set |

## Testing

```bash
# Install dev dependencies
pip install -e '.[dev]'

# Run tests (network domain only — no external dependencies)
pytest tests/ -v
```

## License

MIT

## Citation

If you use SiLR-Agent in your research, please cite:

```bibtex
@misc{silr-agent,
  author = {Chenyu Zhou},
  title  = {SiLR-Agent: Simulation-in-the-Loop Reasoning for LLM Agents},
  year   = {2025},
  url    = {https://github.com/SciTokyo/SILR-Agent}
}
```
