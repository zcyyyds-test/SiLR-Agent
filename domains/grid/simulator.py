"""ANDES System lifecycle manager.

Manages state machine: IDLE → LOADED → PFLOW_DONE → TDS_INITIALIZED → TDS_RUNNING
Provides device queries, event registration, snapshot save/restore.
"""

import enum
import time
import logging
from typing import Optional, Any

import numpy as np
import domains.grid.utils.platform_fix  # noqa: F401 — multiprocess Pool fix
import andes

from .config import SYSTEM_BASE_MVA, MAX_SNAPSHOTS
from silr.core.interfaces import BaseSystemManager
from silr.exceptions import (
    SystemNotLoadedError,
    SystemStateError,
    DeviceNotFoundError,
    ConvergenceError,
    SnapshotError,
)

logger = logging.getLogger(__name__)


class SystemState(enum.Enum):
    IDLE = "idle"
    LOADED = "loaded"
    PFLOW_DONE = "pflow_done"
    TDS_INITIALIZED = "tds_initialized"
    TDS_RUNNING = "tds_running"


# Ordered for comparison
_STATE_ORDER = list(SystemState)


class SystemManager(BaseSystemManager):
    """Manages ANDES System lifecycle with state machine enforcement.

    All tools operate through this manager. Phase 2 SiLR creates shadow
    copies via create_shadow_copy().

    Implements BaseSystemManager for domain-agnostic framework compatibility.
    """

    def __init__(self):
        self._ss: Optional[andes.System] = None
        self._state = SystemState.IDLE
        self._case_path: Optional[str] = None
        self._addfile_path: Optional[str] = None
        self._snapshots: dict[str, dict] = {}
        self._event_registry: list[dict] = []
        self._base_mva = SYSTEM_BASE_MVA

    # --- Properties ---

    @property
    def ss(self) -> andes.System:
        if self._ss is None:
            raise SystemNotLoadedError("No system loaded. Call load() first.")
        return self._ss

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def sim_time(self) -> float:
        if self._ss is None:
            return 0.0
        return float(self._ss.dae.t)

    @property
    def base_mva(self) -> float:
        return self._base_mva

    @property
    def system_state(self):
        """Domain-specific state object (andes.System) for constraint checkers."""
        return self._ss

    # --- Lifecycle ---

    def load(self, case_path: str, addfile: Optional[str] = None,
             setup: bool = True) -> None:
        """Load an ANDES case file."""
        logger.info(f"Loading case: {case_path}")
        self._case_path = case_path
        self._addfile_path = addfile
        self._event_registry.clear()

        kwargs = {"default_config": True, "no_output": True}
        if addfile:
            kwargs["addfile"] = addfile

        # Always load without setup so we can fix params first.
        self._ss = andes.load(case_path, setup=False, **kwargs)
        self._state = SystemState.LOADED

        if setup:
            self._fix_slack_tgov_vmax()
            self.ss.setup()
            self._base_mva = float(self._ss.config.mva)

        logger.info(f"System loaded. Buses={self.ss.Bus.n}, "
                     f"Lines={self.ss.Line.n}")

    def setup(self) -> None:
        """Run ss.setup() for a system loaded with setup=False."""
        self._require_state(SystemState.LOADED)
        self._fix_slack_tgov_vmax()
        self.ss.setup()
        self._base_mva = float(self.ss.config.mva)
        logger.info("System setup complete.")

    def solve(self) -> bool:
        """Run power flow. Returns True if converged."""
        if self._state in (SystemState.TDS_INITIALIZED, SystemState.TDS_RUNNING):
            raise SystemStateError(
                "Cannot run PFlow after TDS initialization. "
                "Reload the system to start fresh."
            )
        self._require_min_state(SystemState.LOADED)

        self.ss.PFlow.run()
        converged = bool(self.ss.PFlow.converged)

        if converged:
            self._state = SystemState.PFLOW_DONE
            logger.info("Power flow converged.")
        else:
            logger.warning("Power flow did NOT converge.")

        return converged

    def init_tds(self) -> None:
        """Initialize TDS. Requires successful PFlow."""
        self._require_state(SystemState.PFLOW_DONE)
        self.ss.TDS.init()
        self._state = SystemState.TDS_INITIALIZED
        logger.info("TDS initialized.")

    def run_tds(self, t_end: float, step_size: Optional[float] = None) -> None:
        """Run TDS to t_end. Auto-initializes if in PFLOW_DONE state."""
        self._require_min_state(SystemState.PFLOW_DONE)

        if self._state == SystemState.PFLOW_DONE:
            self.init_tds()

        if step_size:
            self.ss.TDS.config.tstep = step_size

        self.ss.TDS.config.tf = t_end
        self.ss.TDS.run()

        if getattr(self.ss.TDS, 'busted', False):
            logger.warning(f"TDS diverged at t = {self.sim_time:.4f}s (busted=True)")
        else:
            self._state = SystemState.TDS_RUNNING
            logger.info(f"TDS completed. t = {self.sim_time:.4f}s")

    # --- Event Registration & Rebuild ---

    def register_event(self, model: str, params: dict) -> None:
        """Register a Fault/Toggle event for next rebuild."""
        self._event_registry.append({"model": model, "params": dict(params)})
        logger.info(f"Event registered: {model} {params}")

    def rebuild_with_events(self) -> None:
        """Reload system with registered events.

        Flow: save DAE → reload(setup=False) → add events → setup() →
              PFlow → restore DAE (if dimensions match).
        """
        if not self._event_registry:
            logger.warning("No events registered. Nothing to rebuild.")
            return

        # Save current DAE state if available
        dae_snapshot = None
        if self._state in (SystemState.PFLOW_DONE,
                           SystemState.TDS_INITIALIZED,
                           SystemState.TDS_RUNNING):
            dae_snapshot = {
                "x": np.copy(self.ss.dae.x),
                "y": np.copy(self.ss.dae.y),
                "t": float(self.ss.dae.t),
            }

        # Reload
        kwargs = {"default_config": True, "no_output": True, "setup": False}
        if self._addfile_path:
            kwargs["addfile"] = self._addfile_path

        self._ss = andes.load(self._case_path, **kwargs)

        # Add all registered events
        for event in self._event_registry:
            self.ss.add(event["model"], event["params"])

        self._fix_slack_tgov_vmax()
        self.ss.setup()
        self._base_mva = float(self.ss.config.mva)

        # PFlow to establish algebraic solution
        self.ss.PFlow.run()
        if not self.ss.PFlow.converged:
            raise ConvergenceError("PFlow failed after rebuild with events.")

        self._state = SystemState.PFLOW_DONE

        # Restore DAE if dimensions match
        if dae_snapshot is not None:
            x_match = len(self.ss.dae.x) == len(dae_snapshot["x"])
            y_match = len(self.ss.dae.y) == len(dae_snapshot["y"])
            if x_match and y_match:
                self.ss.dae.x[:] = dae_snapshot["x"]
                self.ss.dae.y[:] = dae_snapshot["y"]
                self.ss.dae.t = dae_snapshot["t"]
                logger.info("DAE state restored after rebuild.")
            else:
                logger.warning(
                    f"DAE dimension changed: "
                    f"x {len(dae_snapshot['x'])}→{len(self.ss.dae.x)}, "
                    f"y {len(dae_snapshot['y'])}→{len(self.ss.dae.y)}. "
                    f"Using fresh PFlow solution."
                )

        logger.info(f"System rebuilt with {len(self._event_registry)} events.")

    # --- Snapshot Management ---

    def save_snapshot(self, name: str = "default") -> None:
        """Save current DAE state to memory."""
        self._require_min_state(SystemState.PFLOW_DONE)

        if len(self._snapshots) >= MAX_SNAPSHOTS:
            oldest = next(iter(self._snapshots))
            del self._snapshots[oldest]
            logger.warning(f"Max snapshots reached. Removed oldest: {oldest}")

        self._snapshots[name] = {
            "x": np.copy(self.ss.dae.x),
            "y": np.copy(self.ss.dae.y),
            "t": float(self.ss.dae.t),
            "state": self._state,
            "events": [dict(e) for e in self._event_registry],
            "saved_at": time.time(),
        }
        logger.info(f"Snapshot '{name}' saved at t={self.sim_time:.4f}s")

    def restore_snapshot(self, name: str = "default") -> None:
        """Restore system from snapshot by reloading + DAE write-back."""
        if name not in self._snapshots:
            available = list(self._snapshots.keys())
            raise SnapshotError(
                f"Snapshot '{name}' not found. Available: {available}"
            )

        snap = self._snapshots[name]

        # Reload system
        kwargs = {"default_config": True, "no_output": True, "setup": False}
        if self._addfile_path:
            kwargs["addfile"] = self._addfile_path

        self._ss = andes.load(self._case_path, **kwargs)

        # Replay events
        self._event_registry = [dict(e) for e in snap["events"]]
        for event in self._event_registry:
            self.ss.add(event["model"], event["params"])

        self.ss.setup()
        self._base_mva = float(self.ss.config.mva)

        # PFlow to establish algebraic structure
        self.ss.PFlow.run()
        if not self.ss.PFlow.converged:
            raise ConvergenceError("PFlow failed during snapshot restore.")

        # Write back DAE arrays
        x_match = len(self.ss.dae.x) == len(snap["x"])
        y_match = len(self.ss.dae.y) == len(snap["y"])
        if x_match and y_match:
            self.ss.dae.x[:] = snap["x"]
            self.ss.dae.y[:] = snap["y"]
            self.ss.dae.t = snap["t"]
        else:
            logger.warning("DAE dimensions differ. Using fresh PFlow state.")

        self._state = snap["state"]
        logger.info(f"Snapshot '{name}' restored.")

    def list_snapshots(self) -> list[str]:
        return list(self._snapshots.keys())

    # --- Device Query ---

    def get_gen_idx_list(self, model: str = "GENROU") -> list:
        """Get all generator indices for a given model type."""
        self._require_min_state(SystemState.LOADED)
        mdl = getattr(self.ss, model, None)
        if mdl is None or mdl.n == 0:
            return []
        return list(mdl.idx.v)

    def get_all_syn_gen_idx(self) -> list:
        """Get all synchronous generator indices (GENROU + GENCLS)."""
        result = []
        for model_name in ["GENROU", "GENCLS"]:
            result.extend(self.get_gen_idx_list(model_name))
        return result

    def get_gen_model_name(self, gen_idx) -> Optional[str]:
        """Return which model ('GENROU' or 'GENCLS') a generator belongs to."""
        for model_name in ["GENROU", "GENCLS"]:
            mdl = getattr(self.ss, model_name, None)
            if mdl is not None and gen_idx in list(mdl.idx.v):
                return model_name
        return None

    def get_bus_idx_list(self) -> list:
        self._require_min_state(SystemState.LOADED)
        return list(self.ss.Bus.idx.v)

    def get_pq_bus_ids(self) -> list[int]:
        """Get bus IDs that have at least one PQ load attached."""
        self._require_min_state(SystemState.LOADED)
        return sorted(set(int(b) for b in self.ss.PQ.bus.v))

    def get_line_idx_list(self) -> list:
        self._require_min_state(SystemState.LOADED)
        return list(self.ss.Line.idx.v)

    def get_static_gen_for_syn(self, syn_idx, syn_model: str = "GENROU") -> Any:
        """Get the StaticGen (PV/Slack) idx for a synchronous generator.

        GENROU/GENCLS have a 'gen' parameter linking to StaticGen.
        """
        mdl = getattr(self.ss, syn_model)
        idx_list = list(mdl.idx.v)
        if syn_idx not in idx_list:
            raise DeviceNotFoundError(
                f"{syn_model} '{syn_idx}' not found."
            )
        i = idx_list.index(syn_idx)
        return mdl.gen.v[i]

    def get_gen_limits(self, static_gen_idx) -> dict:
        """Get Pmin/Pmax/p0 for a static generator (PV or Slack). In p.u."""
        self._require_min_state(SystemState.LOADED)

        for model_name in ["PV", "Slack"]:
            mdl = getattr(self.ss, model_name, None)
            if mdl is None:
                continue
            idx_list = list(mdl.idx.v)
            if static_gen_idx in idx_list:
                i = idx_list.index(static_gen_idx)
                return {
                    "pmin": float(mdl.pmin.v[i]) if hasattr(mdl, 'pmin') else 0.0,
                    "pmax": float(mdl.pmax.v[i]) if hasattr(mdl, 'pmax') else 999.0,
                    "p0": float(mdl.p0.v[i]),
                }

        raise DeviceNotFoundError(
            f"Static gen '{static_gen_idx}' not found in PV or Slack."
        )

    def get_pq_on_bus(self, bus_idx) -> list[dict]:
        """Get all PQ loads on a given bus. Powers in p.u."""
        self._require_min_state(SystemState.LOADED)
        results = []
        bus_list = list(self.ss.PQ.bus.v)
        idx_list = list(self.ss.PQ.idx.v)

        for i, b in enumerate(bus_list):
            if b == bus_idx:
                p_val = float(self.ss.PQ.p0.v[i])
                q_val = float(self.ss.PQ.q0.v[i])
                results.append({
                    "idx": idx_list[i],
                    "p0": p_val,
                    "q0": q_val,
                })
        return results

    def get_line_status(self, line_idx) -> int:
        """Get line connection status (1=connected, 0=disconnected)."""
        self._require_min_state(SystemState.LOADED)
        idx_list = list(self.ss.Line.idx.v)
        if line_idx not in idx_list:
            raise DeviceNotFoundError(f"Line '{line_idx}' not found.")
        i = idx_list.index(line_idx)
        return int(self.ss.Line.u.v[i])

    # --- Shadow Copy (Phase 2 SiLR) ---

    def create_shadow_copy(self) -> "SystemManager":
        """Create an independent copy by reloading the case + DAE write-back."""
        shadow = SystemManager()
        shadow.load(self._case_path, self._addfile_path, setup=False)

        for event in self._event_registry:
            shadow.ss.add(event["model"], event["params"])

        shadow.setup()
        shadow.solve()

        if self._state in (SystemState.PFLOW_DONE,
                           SystemState.TDS_INITIALIZED,
                           SystemState.TDS_RUNNING):
            x_match = len(shadow.ss.dae.x) == len(self.ss.dae.x)
            y_match = len(shadow.ss.dae.y) == len(self.ss.dae.y)
            if x_match and y_match:
                shadow.ss.dae.x[:] = np.copy(self.ss.dae.x)
                shadow.ss.dae.y[:] = np.copy(self.ss.dae.y)
            else:
                logger.warning(
                    f"Shadow DAE dim mismatch: x={x_match}, y={y_match}. "
                    f"Using fresh PFlow."
                )

        # Sync line connection status (u is a parameter, not in DAE vectors).
        # Without this, tripped lines appear connected in the shadow,
        # causing close_line to be rejected as "already connected".
        shadow.ss.Line.u.v[:] = np.copy(self.ss.Line.u.v)

        return shadow

    # --- Parameter Fixes ---

    def _fix_slack_tgov_vmax(self) -> None:
        """Relax TGOV1N VMAX for the slack bus generator.

        Under prestress (load_scale > 1), PFlow dumps all surplus power onto
        the slack bus, pushing its mechanical torque well beyond the default
        TGOV1N VMAX.  The lead-lag block's LAG_y gets clamped while LL_y
        tracks the true torque, creating a large equation mismatch that fails
        TDS initialization (LL_y=31.56 vs LAG_y=12.59 → mismatch 18.97).

        Fix: set VMAX to 999 for the slack gen's governor.  The slack bus
        represents a large aggregated system, not a single turbine, so
        removing the gate-opening limit is physically reasonable.
        """
        ss = self._ss
        if ss is None:
            return

        tgov = getattr(ss, "TGOV1N", None) or getattr(ss, "TGOV1", None)
        if tgov is None or tgov.n == 0:
            return

        slack = getattr(ss, "Slack", None)
        if slack is None or slack.n == 0:
            return

        genrou = getattr(ss, "GENROU", None)
        if genrou is None or genrou.n == 0:
            return

        slack_buses = set(slack.bus.v)
        gen_idx_list = list(genrou.idx.v)
        gen_bus_list = list(genrou.bus.v)
        tgov_syn_list = list(tgov.syn.v)

        for i, syn_ref in enumerate(tgov_syn_list):
            if syn_ref in gen_idx_list:
                gi = gen_idx_list.index(syn_ref)
                if gen_bus_list[gi] in slack_buses:
                    old = tgov.VMAX.v[i]
                    tgov.VMAX.v[i] = 999.0
                    logger.info(
                        f"Relaxed {tgov.class_name}[{tgov.idx.v[i]}] VMAX "
                        f"{old} -> 999.0 (slack bus gen)"
                    )

    # --- State Machine Helpers ---

    def _require_state(self, required: SystemState) -> None:
        if self._state != required:
            raise SystemStateError(
                f"Operation requires state '{required.value}', "
                f"current is '{self._state.value}'."
            )

    def _require_min_state(self, minimum: SystemState) -> None:
        if _STATE_ORDER.index(self._state) < _STATE_ORDER.index(minimum):
            raise SystemStateError(
                f"Operation requires at least '{minimum.value}', "
                f"current is '{self._state.value}'."
            )
