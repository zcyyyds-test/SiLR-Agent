"""gym-anm domain constraint checkers for SiLR verification.

Three constraints, taken directly from gym-anm's own operational limits:
1. Bus voltage magnitude within ``[v_min, v_max]`` (per bus, p.u.).
2. Branch apparent-power flow within its rating ``branch.rate`` (p.u.).
3. Storage state of charge within ``[soc_min, soc_max]``.

All read the live ``Simulator`` passed as ``system_state`` (mirrors the grid
domain, which passes the ANDES ``System`` object). Non-finite values from the
solver (NaN / inf) are flagged as critical violations rather than silently
compared (NaN comparisons always return False and would otherwise mask a
divergent-flow as PASS).
"""

from __future__ import annotations

import math
from typing import Any

from silr.core.interfaces import BaseConstraintChecker
from silr.verifier.types import CheckResult, Violation


def _slack_id(sim) -> Any:
    for bid, bus in sim.buses.items():
        if bool(bus.is_slack):
            return bid
    return None


class ANMVoltageChecker(BaseConstraintChecker):
    """Bus voltage magnitudes must stay within each bus's [v_min, v_max].

    The slack bus (fixed at 1.0 p.u. by definition) is excluded from the
    min/max summary stats but still checked for finiteness.
    """

    name = "voltage"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        sim = system_state
        slack = _slack_id(sim)
        violations = []
        nonslack_v = []
        for bid, bus in sim.buses.items():
            v = abs(bus.v)
            if not math.isfinite(v):
                violations.append(
                    Violation(
                        constraint_type="voltage",
                        device_type="bus",
                        device_id=bid,
                        metric="v_pu",
                        value=float("nan"),
                        limit=float(bus.v_min),
                        unit="p.u.",
                        severity="critical",
                        detail=f"Bus {bid}: V is non-finite ({v!r})",
                    )
                )
                continue
            v = float(v)
            if bid != slack:
                nonslack_v.append(v)
            vmin, vmax = float(bus.v_min), float(bus.v_max)
            if v < vmin or v > vmax:
                low = v < vmin
                margin = (vmin - v) if low else (v - vmax)
                violations.append(
                    Violation(
                        constraint_type="voltage",
                        device_type="bus",
                        device_id=bid,
                        metric="v_pu",
                        value=round(v, 4),
                        limit=round(vmin if low else vmax, 4),
                        unit="p.u.",
                        severity="critical" if margin > 0.05 else "violation",
                        detail=(
                            f"Bus {bid}: V = {v:.4f} p.u. "
                            f"outside [{vmin:.3f}, {vmax:.3f}]"
                        ),
                    )
                )

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "min_pu": round(min(nonslack_v), 4) if nonslack_v else None,
                "max_pu": round(max(nonslack_v), 4) if nonslack_v else None,
                "n_violations": len(violations),
            },
            violations=violations,
        )


class ANMBranchLoadingChecker(BaseConstraintChecker):
    """Branch apparent-power flow must stay within its rating. rate<=0 skipped.

    Note: gym-anm's ``branch.s_apparent_max`` is misleadingly named — it is the
    *current* apparent power on the branch, not the rating; ``branch.rate`` is
    the rating. Non-finite flows (divergent power flow) are flagged critical.
    """

    name = "branch_loading"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        sim = system_state
        violations = []
        max_loading = 0.0
        n_rated = 0
        for br in sim.branches.values():
            rate = float(br.rate)
            if rate <= 0:
                continue
            n_rated += 1
            s_val = abs(br.s_apparent_max)
            if not math.isfinite(s_val):
                violations.append(
                    Violation(
                        constraint_type="branch_loading",
                        device_type="branch",
                        device_id=f"{br.f_bus}-{br.t_bus}",
                        metric="loading_pct",
                        value=float("nan"),
                        limit=100.0,
                        unit="%",
                        severity="critical",
                        detail=(
                            f"Branch {br.f_bus}-{br.t_bus}: apparent power is "
                            f"non-finite ({s_val!r}) — solver likely divergent"
                        ),
                    )
                )
                continue
            s = float(s_val)
            loading = s / rate * 100.0
            if loading > max_loading:
                max_loading = loading
            if loading > 100.0:
                violations.append(
                    Violation(
                        constraint_type="branch_loading",
                        device_type="branch",
                        device_id=f"{br.f_bus}-{br.t_bus}",
                        metric="loading_pct",
                        value=round(loading, 1),
                        limit=100.0,
                        unit="%",
                        severity="critical" if loading > 120.0 else "violation",
                        detail=(
                            f"Branch {br.f_bus}-{br.t_bus}: loading = {loading:.1f}% "
                            f"(S = {s:.4f} p.u. > rate {rate:.4f} p.u.)"
                        ),
                    )
                )

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "max_loading_pct": round(max_loading, 1) if n_rated else None,
                "n_rated_branches": n_rated,
                "n_violations": len(violations),
            },
            violations=violations,
        )


class ANMStorageSoCChecker(BaseConstraintChecker):
    """Storage state of charge must stay within ``[soc_min, soc_max]`` (per device).

    Reads ``device.soc`` post-transition (verified writeback on gym-anm 2.0.1).
    Non-finite SoC is flagged critical.
    """

    name = "storage_soc"

    def check(self, system_state: Any, base_mva: float) -> CheckResult:
        sim = system_state
        violations = []
        # gym-anm sets soc/soc_min on every Device subclass (None for non-storage),
        # so filter by class instead of hasattr to avoid sweeping in loads/gens.
        from gym_anm.simulator.components import StorageUnit

        storages = [
            (dev_id, dev)
            for dev_id, dev in sim.devices.items()
            if isinstance(dev, StorageUnit)
        ]
        if not storages:
            return CheckResult(
                checker_name=self.name,
                passed=True,
                summary={"n_storage": 0, "n_violations": 0},
                violations=[],
            )

        soc_summary = {}
        for dev_id, dev in storages:
            raw_soc = dev.soc
            if raw_soc is None:
                # gym-anm leaves soc unset in some early/edge-case transitions;
                # treat as not-yet-observed rather than a violation.
                soc_summary[dev_id] = None
                continue
            try:
                soc = float(raw_soc)
            except (TypeError, ValueError):
                soc_summary[dev_id] = None
                continue
            soc_min = float(dev.soc_min)
            soc_max = float(dev.soc_max)
            soc_summary[dev_id] = round(soc, 4) if math.isfinite(soc) else None

            if not math.isfinite(soc):
                violations.append(
                    Violation(
                        constraint_type="storage_soc",
                        device_type="storage",
                        device_id=dev_id,
                        metric="soc",
                        value=float("nan"),
                        limit=soc_min,
                        unit="pu",
                        severity="critical",
                        detail=f"Storage {dev_id}: SoC is non-finite",
                    )
                )
                continue
            if soc < soc_min or soc > soc_max:
                low = soc < soc_min
                violations.append(
                    Violation(
                        constraint_type="storage_soc",
                        device_type="storage",
                        device_id=dev_id,
                        metric="soc",
                        value=round(soc, 4),
                        limit=round(soc_min if low else soc_max, 4),
                        unit="pu",
                        severity="critical",
                        detail=(
                            f"Storage {dev_id}: SoC = {soc:.4f} "
                            f"outside [{soc_min:.4f}, {soc_max:.4f}]"
                        ),
                    )
                )

        return CheckResult(
            checker_name=self.name,
            passed=len(violations) == 0,
            summary={
                "n_storage": len(storages),
                "soc": soc_summary,
                "n_violations": len(violations),
            },
            violations=violations,
        )
