"""Network domain tools for SiLR verification demo.

4 tools matching the SiLR pattern: observe + act + check.
All inherit from BaseTool for framework compatibility.
"""

from __future__ import annotations

from silr.tools.base import BaseTool
from silr.exceptions import DeviceNotFoundError, ValidationError


class GetLinkStatusTool(BaseTool):
    """Observe all link states: traffic, capacity, utilization."""

    name = "get_link_status"
    description = "Get current status of all network links"

    def _validate_params(self, **kwargs) -> None:
        pass

    def _run(self, **kwargs) -> dict:
        mgr = self.manager
        links = []
        for (src, dst), data in mgr._links.items():
            util = (data["traffic"] / data["capacity"] * 100) if data["capacity"] > 0 else 0
            links.append({
                "link": f"{src}-{dst}",
                "src": src,
                "dst": dst,
                "capacity_mbps": data["capacity"],
                "traffic_mbps": data["traffic"],
                "utilization_pct": round(util, 1),
                "up": data["up"],
                "latency_ms": data["latency"],
            })
        return {"links": links, "total": len(links)}


class CheckNetworkHealthTool(BaseTool):
    """Check overall network health: connectivity and constraint satisfaction."""

    name = "check_network_health"
    description = "Check network health and constraint violations"

    def _validate_params(self, **kwargs) -> None:
        pass

    def _run(self, **kwargs) -> dict:
        mgr = self.manager
        up_links = sum(1 for d in mgr._links.values() if d["up"])
        total_links = len(mgr._links)
        overloaded = []
        for (src, dst), data in mgr._links.items():
            if data["up"] and data["capacity"] > 0:
                util = data["traffic"] / data["capacity"] * 100
                if util > 90:
                    overloaded.append(f"{src}-{dst} ({util:.0f}%)")

        return {
            "up_links": up_links,
            "total_links": total_links,
            "overloaded": overloaded,
            "is_healthy": len(overloaded) == 0 and up_links == total_links,
        }


class RestoreLinkTool(BaseTool):
    """Restore a failed network link."""

    name = "restore_link"
    description = "Restore a failed link between two nodes"

    def _validate_params(self, src: int = 0, dst: int = 0, **kwargs) -> None:
        if src == 0 or dst == 0:
            raise ValidationError("src and dst node IDs are required")
        mgr = self.manager
        key = mgr._link_key(src, dst)
        if key is None:
            raise DeviceNotFoundError(
                f"Link {src}-{dst} not found. "
                f"Available: {[f'{a}-{b}' for a, b in mgr._links.keys()]}"
            )

    def _run(self, src: int = 0, dst: int = 0, **kwargs) -> dict:
        mgr = self.manager
        restored = mgr.restore_link(src, dst)
        return {
            "link": f"{src}-{dst}",
            "restored": restored,
            "message": f"Link {src}-{dst} restored" if restored else f"Link {src}-{dst} was already up",
        }


class RerouteTrafficTool(BaseTool):
    """Reroute traffic away from an overloaded link."""

    name = "reroute_traffic"
    description = "Reduce traffic on a link by rerouting through alternatives"

    def _validate_params(self, src: int = 0, dst: int = 0,
                         amount_mbps: float = 0, **kwargs) -> None:
        if src == 0 or dst == 0:
            raise ValidationError("src and dst node IDs are required")
        if amount_mbps <= 0:
            raise ValidationError("amount_mbps must be positive")
        mgr = self.manager
        key = mgr._link_key(src, dst)
        if key is None:
            raise DeviceNotFoundError(f"Link {src}-{dst} not found")

    def _run(self, src: int = 0, dst: int = 0,
             amount_mbps: float = 0, **kwargs) -> dict:
        mgr = self.manager
        success = mgr.reroute_traffic(src, dst, amount_mbps)
        return {
            "link": f"{src}-{dst}",
            "amount_mbps": amount_mbps,
            "success": success,
        }


def create_network_toolset(manager) -> dict:
    """Create toolset for the toy network domain."""
    tools = [
        GetLinkStatusTool(manager),
        CheckNetworkHealthTool(manager),
        RestoreLinkTool(manager),
        RerouteTrafficTool(manager),
    ]
    return {t.name: t for t in tools}
