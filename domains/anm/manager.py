"""GymANMManager: wraps a gym-anm Simulator for the SiLR verification pipeline.

gym-anm (Henry & Ernst, *Energy and AI*, 2021; arXiv:2103.07932) provides public,
peer-reviewed RL environments for Active Network Management (ANM) in electricity
distribution networks. Wrapping its ``Simulator`` gives SiLR a community-standard,
externally-validated power-systems testbed — directly answering the "is your
simulator faithful?" reviewer concern that self-built simulators attract.

Shadow-execution design (verified empirically on gym-anm 2.0.1, see decisions.md):
  - ``Simulator.transition(P_load, P_potential, P_set_points, Q_set_points)`` takes the
    realized load / max-generation as ARGUMENTS. We capture (freeze) the current
    timestep's realized conditions and let the verifier vary only the *action*
    (set-points), re-solving the power flow without advancing the stochastic process.
  - The ``Simulator`` is plain numpy/scipy and deepcopy-isolates cleanly, so
    ``create_shadow_copy() = copy.deepcopy(simulator)`` (proven: a transition on the
    copy leaves the original's bus voltages bit-identical).

Constraints come from gym-anm itself: per-bus voltage limits (``bus.v_min/v_max``),
per-branch apparent-power ratings (``branch.rate``), and storage SoC bounds
(``device.soc_min/soc_max``).

Episode-step semantics (``step()``) advance the env's stochastic process and
re-sample conditions; the verifier never touches ``step()`` — it only deepcopies
the simulator and replays ``transition`` under frozen conditions.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

from silr.core.interfaces import BaseSystemManager

logger = logging.getLogger(__name__)


class GymANMManager(BaseSystemManager):
    """Wrap a gym-anm environment's Simulator as a SiLR system manager.

    The agent acts by setting device set-points (via tools, which mutate
    ``_P_set`` / ``_Q_set``); ``solve()`` re-solves the power flow under the
    frozen conditions, after which constraint checkers read ``system_state``
    (the live ``Simulator``, exposing ``buses`` and ``branches``).
    """

    def __init__(
        self,
        env: Any = None,
        env_name: str = "ANM6Easy",
        seed: Optional[int] = None,
    ):
        if env is None:
            import gym_anm.envs as _envs

            env = getattr(_envs, env_name)()
        self._env = env
        # gymnasium 1.x reset accepts seed kw for reproducibility
        if seed is not None:
            try:
                self._env.reset(seed=int(seed))
            except TypeError:
                # legacy gym.Env without seed kwarg
                self._env.reset()
        else:
            self._env.reset()
        self._sim = env.simulator
        self._seed = seed

        from gym_anm.simulator.components import Generator, Load, StorageUnit

        devs = self._sim.devices
        self._load_ids = [i for i, d in devs.items() if isinstance(d, Load)]
        self._gen_ids = [
            i for i, d in devs.items() if isinstance(d, Generator) and not d.is_slack
        ]
        self._des_ids = [i for i, d in devs.items() if isinstance(d, StorageUnit)]
        # belt-and-suspenders: env enumerates next_vars / step in the same order,
        # but assert symmetric counts in case gym-anm changes its filter convention.
        assert len(self._load_ids) == self._sim.N_load, (
            f"load id count {len(self._load_ids)} != sim.N_load {self._sim.N_load}"
        )
        assert len(self._gen_ids) == self._sim.N_non_slack_gen, (
            f"non-slack gen id count {len(self._gen_ids)} != "
            f"sim.N_non_slack_gen {self._sim.N_non_slack_gen}"
        )

        self._time: float = 0.0
        try:
            self._dt: float = float(env.delta_t)
        except AttributeError:
            self._dt = 1.0
        # Frozen current-timestep conditions: realized load + max generation.
        self._P_load: dict[int, float] = {}
        self._P_pot: dict[int, float] = {}
        # Pending control set-points (mutated by tools, applied by solve()).
        self._P_set: dict[int, float] = {}
        self._Q_set: dict[int, float] = {}
        self._last_reward: float = 0.0
        self._last_penalty: float = 0.0

        self._sample_conditions()
        self._init_default_setpoints()
        if not self.solve():
            logger.warning(
                "GymANMManager: initial transition did not converge "
                "(default set-points may be infeasible for this snapshot)"
            )

    # --- condition / set-point setup ---

    def _sample_conditions(self) -> None:
        """Freeze the current realized load / max-generation from ``next_vars``."""
        v = self._env.next_vars(self._env.state)
        n_load = self._sim.N_load
        n_nsg = self._sim.N_non_slack_gen
        self._P_load = {d: float(v[k]) for k, d in enumerate(self._load_ids)}
        self._P_pot = {d: float(v[n_load + k]) for k, d in enumerate(self._gen_ids)}

    def set_conditions(
        self,
        P_load: dict,
        P_pot: dict,
        reset_setpoints: bool = True,
        solve: bool = True,
    ) -> bool:
        """Override the frozen conditions (used to build stress scenarios).

        Validates that ``P_load`` keys exactly match the manager's load ids and
        ``P_pot`` keys match the non-slack generator ids; raises ValueError on
        mismatch (silent mismatches would feed bad data to ``transition``).

        By default also resets the pending set-points to defaults (renewables
        at full potential, storage idle) and re-solves the power flow so the
        manager state is consistent with the new conditions. Pass
        ``reset_setpoints=False`` or ``solve=False`` to skip those.
        """
        if set(P_load.keys()) != set(self._load_ids):
            raise ValueError(
                f"P_load keys {sorted(P_load.keys())} must equal "
                f"load ids {sorted(self._load_ids)}"
            )
        if set(P_pot.keys()) != set(self._gen_ids):
            raise ValueError(
                f"P_pot keys {sorted(P_pot.keys())} must equal "
                f"non-slack gen ids {sorted(self._gen_ids)}"
            )
        self._P_load = {int(k): float(val) for k, val in P_load.items()}
        self._P_pot = {int(k): float(val) for k, val in P_pot.items()}
        if reset_setpoints:
            self._init_default_setpoints()
        if solve:
            return self.solve()
        return True

    def _init_default_setpoints(self) -> None:
        """Default: renewables output their potential, no curtailment / no storage."""
        self._P_set = {g: self._P_pot.get(g, 0.0) for g in self._gen_ids}
        self._Q_set = {g: 0.0 for g in self._gen_ids}
        for s in self._des_ids:
            self._P_set[s] = 0.0
            self._Q_set[s] = 0.0

    # --- BaseSystemManager interface ---

    @property
    def sim_time(self) -> float:
        return self._time

    @property
    def base_mva(self) -> float:
        return float(self._sim.baseMVA)

    @property
    def system_state(self) -> Any:
        """The live Simulator; checkers read ``buses`` / ``branches`` off it."""
        return self._sim

    def create_shadow_copy(self) -> "GymANMManager":
        """Independent copy for verification. Deepcopies only the Simulator
        (the gym env is not needed and may hold non-copyable handles)."""
        shadow = GymANMManager.__new__(GymANMManager)
        shadow._env = None
        shadow._sim = copy.deepcopy(self._sim)
        self._copy_state_into(shadow)
        return shadow

    def _copy_state_into(self, target: "GymANMManager") -> None:
        """Copy all non-Simulator state into ``target``. Centralises the field
        list so adding a new field doesn't silently break ``create_shadow_copy``."""
        target._seed = self._seed
        target._load_ids = list(self._load_ids)
        target._gen_ids = list(self._gen_ids)
        target._des_ids = list(self._des_ids)
        target._time = self._time
        target._dt = self._dt
        target._P_load = dict(self._P_load)
        target._P_pot = dict(self._P_pot)
        target._P_set = dict(self._P_set)
        target._Q_set = dict(self._Q_set)
        target._last_reward = self._last_reward
        target._last_penalty = self._last_penalty

    def solve(self) -> bool:
        """Re-solve the power flow under frozen conditions + current set-points.

        Pure re-evaluation: does not advance ``sim_time`` or re-sample conditions
        (use ``step()`` for episode advancement). Exceptions are logged and
        reported as non-convergence rather than propagating to the verifier.
        """
        try:
            _, reward, _e_loss, penalty, converged = self._sim.transition(
                self._P_load, self._P_pot, self._P_set, self._Q_set
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "GymANMManager.solve(): transition raised %s: %s "
                "(P_set=%s, Q_set=%s) — treating as non-convergence",
                type(e).__name__, e, self._P_set, self._Q_set,
            )
            return False
        self._last_reward = float(reward)
        self._last_penalty = float(penalty)
        return bool(converged)

    # --- episode-step (advances time + re-samples conditions) ---

    def step(self, reset_setpoints: bool = True) -> bool:
        """Apply current set-points and advance one episode timestep.

        Re-solves the power flow, advances ``sim_time`` by ``delta_t``, and
        re-samples the next stochastic conditions for the next step. Distinct
        from ``solve()`` which never advances time — that separation is what
        lets the verifier replay transitions on a deepcopied simulator without
        side-effecting the real episode.

        ``reset_setpoints=True`` (default) restores defaults after stepping so
        a stale curtailment from step t-1 doesn't silently carry into step t.
        """
        converged = self.solve()
        self._time += self._dt
        # Sync env state so next_vars sees a consistent time-of-day / aux.
        # gym-anm v2 keeps state in self._env.state; refresh from the simulator.
        try:
            self._env.state = self._sim.state_values if hasattr(self._sim, "state_values") else self._env.state
        except Exception:  # noqa: BLE001
            pass
        self._sample_conditions()
        if reset_setpoints:
            self._init_default_setpoints()
        return converged

    # --- domain helpers ---

    @property
    def last_reward(self) -> float:
        """gym-anm native reward of the last solve (−energy_loss − λ·penalty)."""
        return self._last_reward

    @property
    def last_penalty(self) -> float:
        """gym-anm native constraint-violation penalty of the last solve."""
        return self._last_penalty

    def get_generator_ids(self) -> list[int]:
        return list(self._gen_ids)

    def get_storage_ids(self) -> list[int]:
        return list(self._des_ids)
