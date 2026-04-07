"""NetworkManager: 5-node network simulator for SiLR framework demo.

Topology (5 nodes, 6 bidirectional links):

    1 ---100--- 2 ---100--- 3
    |           |           |
   80          60          80
    |           |           |
    4 ----100---- 5 --------+

Each link has capacity (Mbps) and current traffic load.
Faults = link failures. Recovery = reroute traffic.
Constraints = link utilization < 90%, all demands routable.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from silr.core.interfaces import BaseSystemManager


# Default topology: (src, dst) -> {capacity_mbps, traffic_mbps, up, latency_ms}
_DEFAULT_LINKS = {
    (1, 2): {"capacity": 100, "traffic": 40, "up": True, "latency": 2},
    (2, 3): {"capacity": 100, "traffic": 30, "up": True, "latency": 2},
    (1, 4): {"capacity": 80, "traffic": 20, "up": True, "latency": 3},
    (2, 5): {"capacity": 60, "traffic": 25, "up": True, "latency": 4},
    (3, 5): {"capacity": 80, "traffic": 15, "up": True, "latency": 3},
    (4, 5): {"capacity": 100, "traffic": 35, "up": True, "latency": 2},
}

# Demands: (src, dst) -> mbps  (traffic to route)
_DEFAULT_DEMANDS = {
    (1, 3): 30,
    (1, 5): 20,
    (4, 3): 15,
    (4, 2): 10,
}


class NetworkManager(BaseSystemManager):
    """5-node network simulator. Pure Python, no external dependencies.

    Implements BaseSystemManager for SiLR verification compatibility.
    """

    def __init__(self):
        self._time: float = 0.0
        self._links: dict[tuple[int, int], dict] = copy.deepcopy(_DEFAULT_LINKS)
        self._demands: dict[tuple[int, int], float] = copy.deepcopy(_DEFAULT_DEMANDS)
        self._nodes: set[int] = {1, 2, 3, 4, 5}
        self._converged: bool = True

    # --- BaseSystemManager interface ---

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def base_mva(self) -> float:
        return 1.0  # no per-unit system in networking

    @property
    def system_state(self) -> dict:
        """Domain state for constraint checkers."""
        return {
            "links": self._links,
            "demands": self._demands,
            "nodes": self._nodes,
        }

    def create_shadow_copy(self) -> NetworkManager:
        """Create independent copy for SiLR verification."""
        shadow = NetworkManager()
        shadow._time = self._time
        shadow._links = copy.deepcopy(self._links)
        shadow._demands = copy.deepcopy(self._demands)
        shadow._converged = self._converged
        return shadow

    def solve(self) -> bool:
        """Recalculate traffic distribution using shortest-path routing.

        Returns True if all demands can be satisfied.
        """
        # Reset traffic on all up links
        for link_data in self._links.values():
            if link_data["up"]:
                link_data["traffic"] = 0

        # Route each demand along shortest path (by latency)
        all_routed = True
        for (src, dst), demand in self._demands.items():
            path = self._shortest_path(src, dst)
            if path is None:
                all_routed = False
                continue
            for i in range(len(path) - 1):
                link_key = self._link_key(path[i], path[i + 1])
                if link_key and self._links[link_key]["up"]:
                    self._links[link_key]["traffic"] += demand

        self._converged = all_routed
        self._time += 1.0
        return all_routed

    # --- Domain-specific operations ---

    def fail_link(self, src: int, dst: int) -> bool:
        """Simulate link failure. Returns True if link existed and was up."""
        key = self._link_key(src, dst)
        if key is None:
            return False
        if not self._links[key]["up"]:
            return False
        self._links[key]["up"] = False
        self._links[key]["traffic"] = 0
        return True

    def restore_link(self, src: int, dst: int) -> bool:
        """Restore a failed link. Returns True if link existed and was down."""
        key = self._link_key(src, dst)
        if key is None:
            return False
        if self._links[key]["up"]:
            return False
        self._links[key]["up"] = True
        return True

    def reroute_traffic(self, src: int, dst: int, amount: float) -> bool:
        """Move traffic away from a specific link by rerouting.

        Reduces traffic on (src, dst) and redistributes via alternative paths.
        Returns True if successful.
        """
        key = self._link_key(src, dst)
        if key is None:
            return False
        link = self._links[key]
        if not link["up"]:
            return False
        actual_reduce = min(amount, link["traffic"])
        link["traffic"] -= actual_reduce
        alt_links = [
            k for k, v in self._links.items()
            if v["up"] and k != key
        ]
        if alt_links:
            per_link = actual_reduce / len(alt_links)
            for alt_key in alt_links:
                self._links[alt_key]["traffic"] += per_link
        return True

    def get_link_ids(self) -> list[tuple[int, int]]:
        """Return all link IDs."""
        return list(self._links.keys())

    def get_node_ids(self) -> list[int]:
        """Return all node IDs."""
        return sorted(self._nodes)

    # --- Internal helpers ---

    def _link_key(self, src: int, dst: int) -> Optional[tuple[int, int]]:
        """Find canonical link key (links are stored in one direction only)."""
        if (src, dst) in self._links:
            return (src, dst)
        if (dst, src) in self._links:
            return (dst, src)
        return None

    def _shortest_path(self, src: int, dst: int) -> Optional[list[int]]:
        """Dijkstra's shortest path by latency on up links."""
        import heapq
        adj: dict[int, list[tuple[int, float]]] = {n: [] for n in self._nodes}
        for (a, b), data in self._links.items():
            if data["up"]:
                adj[a].append((b, data["latency"]))
                adj[b].append((a, data["latency"]))

        dist = {n: float("inf") for n in self._nodes}
        prev: dict[int, Optional[int]] = {n: None for n in self._nodes}
        dist[src] = 0
        pq = [(0, src)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            if u == dst:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

        if dist[dst] == float("inf"):
            return None

        path = []
        node = dst
        while node is not None:
            path.append(node)
            node = prev[node]
        return list(reversed(path))
