# Power Grid Domain

A reference power grid domain built on the [ANDES](https://docs.andes.app/) transient stability simulator. Demonstrates SiLR applied to a domain with a real numerical solver, time-domain simulation, and stiff stability constraints.

## Requirements

```bash
pip install -e '.[grid]'
```

This installs ANDES and its dependencies. The grid domain will not load without ANDES installed.

## Architecture

`SystemManager` wraps an `andes.System` instance and exposes the `BaseSystemManager` interface. The lifecycle is a small state machine:

```
IDLE → LOADED → PFLOW_DONE → TDS_INITIALIZED → TDS_RUNNING
```

`solve()` runs the AC power flow. The optional `_tds_hook` extends verification with a short transient simulation, so an action is only accepted if the post-action system also remains dynamically stable.

## Constraints

| Checker | Validates |
|---------|-----------|
| `VoltageChecker` | All bus voltages within `[0.90, 1.10]` p.u. |
| `FrequencyChecker` | System frequency deviation within ±0.5 Hz |
| `LineLoadingChecker` | No line exceeds 100% of its thermal limit |
| `TransientStabilityChecker` | Generator rotor angle separation stays below threshold during TDS |

## Tools

| Category | Examples |
|----------|----------|
| Generation control | `adjust_gen` (re-dispatch active power) |
| Load shedding | `shed_load` (drop load at a bus) |
| Topology | `open_breaker`, `close_breaker` |
| Observation | Voltage, frequency, line loading queries |

Full tool list lives in `domains/grid/tools/`.

## Failure Scenarios

`domains/grid/scenarios/` includes N-1 contingencies, multi-line cascading faults, and pre-stressed loading conditions used in the cascading-fault evaluation suite.

## Usage

```python
import andes
from domains.grid import SystemManager, build_grid_domain_config
from silr.verifier import SiLRVerifier

manager = SystemManager()
manager.load(andes.get_case("ieee39/ieee39_full.xlsx"))
manager.solve()

verifier = SiLRVerifier(manager, domain_config=build_grid_domain_config(pflow_only=True))
result = verifier.verify({"tool_name": "adjust_gen", "params": {"gen_id": "GENROU_1", "delta_p_mw": -20}})
```

See `examples/grid_recovery.py` for a complete runnable demo.
