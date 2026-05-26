from __future__ import annotations

import copy

from silr.core.config import DomainConfig
from silr.core.interfaces import BaseConstraintChecker, BaseSystemManager
from silr.verifier import CheckResult, SiLRVerifier, Verdict, Violation


class _PenaltyManager(BaseSystemManager):
    def __init__(self, penalty: float = 10.0):
        self.last_penalty = penalty
        self.next_penalty = penalty

    @property
    def sim_time(self) -> float:
        return 0.0

    @property
    def base_mva(self) -> float:
        return 1.0

    @property
    def system_state(self):
        return self

    def create_shadow_copy(self):
        return copy.deepcopy(self)

    def solve(self) -> bool:
        self.last_penalty = self.next_penalty
        return True


class _SetPenaltyTool:
    def __init__(self, manager: _PenaltyManager):
        self._manager = manager

    def execute(self, penalty: float):
        self._manager.next_penalty = penalty
        return {"status": "ok"}


class _PenaltyChecker(BaseConstraintChecker):
    name = "penalty"

    def check(self, system_state: _PenaltyManager, base_mva: float):
        if system_state.last_penalty <= 1e-9:
            return CheckResult("penalty", True, {})
        return CheckResult(
            "penalty",
            False,
            {},
            [
                Violation(
                    constraint_type="penalty",
                    device_type="system",
                    device_id="all",
                    metric="penalty",
                    value=system_state.last_penalty,
                    limit=0.0,
                    unit="",
                    severity="violation",
                    detail="synthetic penalty remains positive",
                )
            ],
        )


def _verifier() -> SiLRVerifier:
    cfg = DomainConfig(
        domain_name="synthetic",
        checkers=[_PenaltyChecker()],
        allowed_actions=frozenset({"set_penalty"}),
        create_toolset=lambda manager: {"set_penalty": _SetPenaltyTool(manager)},
        gating_policy="scalar_progress",
    )
    return SiLRVerifier(_PenaltyManager(), cfg)


def test_scalar_progress_admits_non_worsening_nonterminal_step():
    result = _verifier().verify({"tool_name": "set_penalty", "params": {"penalty": 9.5}})

    assert result.verdict == Verdict.SAFE_PROGRESS
    assert "Admissible scalar recovery step" in result.fail_reason


def test_scalar_progress_rejects_scalar_penalty_increase():
    result = _verifier().verify({"tool_name": "set_penalty", "params": {"penalty": 10.2}})

    assert result.verdict == Verdict.FAIL
    assert "Scalar penalty worsened" in result.fail_reason


def test_scalar_progress_relative_slack_is_opt_in(monkeypatch):
    monkeypatch.setenv("SILR_SCALAR_PROGRESS_RELATIVE_SLACK", "0.05")

    admitted = _verifier().verify({
        "tool_name": "set_penalty",
        "params": {"penalty": 10.4},
    })
    rejected = _verifier().verify({
        "tool_name": "set_penalty",
        "params": {"penalty": 10.6},
    })

    assert admitted.verdict == Verdict.SAFE_PROGRESS
    assert rejected.verdict == Verdict.FAIL
    assert "threshold 10.5000" in rejected.fail_reason


def test_scalar_progress_still_passes_terminal_recovery():
    result = _verifier().verify({"tool_name": "set_penalty", "params": {"penalty": 0.0}})

    assert result.verdict == Verdict.PASS
