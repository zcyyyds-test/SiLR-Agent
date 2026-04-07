# Network Domain

A minimal 5-node network domain. Pure Python, **zero external dependencies** — useful as the entry-point reference for understanding how a SiLR domain plugs together.

## Topology

```
    1 ---100--- 2 ---100--- 3
    |           |           |
   80          60          80
    |           |           |
    4 ----100---- 5 --------+
```

5 nodes connected by 6 bidirectional links. Each link has a capacity (Mbps) and current traffic load. The simulator's `solve()` method redistributes traffic along shortest paths until convergence.

## Constraints

| Checker | Validates |
|---------|-----------|
| `LinkUtilizationChecker` | No link exceeds 90% utilisation |
| `ConnectivityChecker` | All node pairs remain reachable |

## Tools

| Tool | Effect |
|------|--------|
| `restore_link` | Bring a downed link back up |
| `reroute_traffic` | Move traffic from a congested link onto an alternate path |

`NetworkManager` also exposes `fail_link()` as a direct method for fault injection in tests and examples — it is intentionally not registered as an agent tool.

## Failure Scenarios

The cascading-fault scenario suite (`scenarios.py`) covers single-link failures, multi-link cascades, and overload pre-stress conditions that force the agent to choose between connectivity and utilisation specialists.

## Usage

```python
from domains.network import NetworkManager, build_network_domain_config
from silr.verifier import SiLRVerifier

manager = NetworkManager()
manager.fail_link(1, 2)
manager.solve()

verifier = SiLRVerifier(manager, domain_config=build_network_domain_config())
result = verifier.verify({"tool_name": "restore_link", "params": {"src": 1, "dst": 2}})
```

See `examples/network_routing.py` and `examples/network_cascading.py` for complete runnable demos, and `domains/network/specialists.py` for the multi-agent specialist configurations used by the coordinator demo.
