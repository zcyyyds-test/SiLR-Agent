"""0-GPU probe: does the cluster domain present a NONEMPTY verifier-level
product-order state Phi at the start of its scenarios?

pillar-2's geometric process reward rewards descent in the product order over
Phi=(S support set, sigma per-branch severity). That requires the *start* state
to carry multiple incomparable verifier-level violations to descend on. For the
cluster domain the verifier-level checkers are resource_capacity + affinity
(config.py wires only these two; queue/priority/rack_spread are observer-only).

This probe sets up every scenario and reports, at the start state, the
verifier-level violation count (capacity+affinity = Phi) vs the observer-level
counts (queue/priority/rack_spread). If Phi is empty across scenarios, the
geometric reward has nothing to descend on -> the domain does not fit the
pillar-2 Phi-descent framing (the verifier is only a placement gate).

Run from repo root (Python 3.10+; TSUBAME silr-vllm env):
    PYTHONPATH=. python scripts/cluster_phi_feasibility_probe.py
"""

from __future__ import annotations

from domains.cluster.manager import ClusterManager
from domains.cluster.scenarios.loader import ClusterScenarioLoader, SCENARIOS
from domains.cluster.checkers import ResourceCapacityChecker, AffinityChecker
import domains.cluster.checkers as C

cap = ResourceCapacityChecker()
aff = AffinityChecker()
extra = [
    getattr(C, n)()
    for n in ("RackSpreadChecker", "PriorityChecker", "QueueChecker")
    if hasattr(C, n)
]
loader = ClusterScenarioLoader()

nonempty = 0
print(f"{'scenario':40s} {'diff':6s} | Phi(cap+aff) | observer(queue/prio/spread)")
for s in SCENARIOS:
    m = ClusterManager()
    loader.setup_episode(m, s)
    st = m.system_state
    vc = len(cap.check(st, 1.0).violations)
    va = len(aff.check(st, 1.0).violations)
    obs = {ch.name: len(ch.check(st, 1.0).violations) for ch in extra}
    phi = vc + va
    nonempty += phi > 0
    tag = "  <-- Phi nonempty" if phi > 0 else ""
    print(f"{s.id:40s} {s.difficulty:6s} | cap={vc} aff={va} (Phi={phi}) | {obs}{tag}")

print(f"\n=== {nonempty}/{len(SCENARIOS)} scenarios have NONEMPTY verifier-level "
      f"Phi at start ===")
print("If 0: geometric Phi-descent reward has no signal at start -> cluster does "
      "not fit pillar-2 (verifier is only a placement gate, not a violation "
      "landscape to descend).")
