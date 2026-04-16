"""Portfolio compliance DomainConfig factory."""

from silr.core.config import DomainConfig
from .tools import create_finance_toolset
from .observation import FinanceObserver
from .prompts import build_finance_system_prompt, build_finance_tool_schemas


def build_finance_domain_config(with_observer: bool = True) -> DomainConfig:
    """Build a DomainConfig for the portfolio compliance domain.

    Args:
        with_observer: If True, include FinanceObserver.
            Default True for agent evaluation.
    """
    return DomainConfig(
        domain_name="portfolio_compliance",
        # No verifier checkers — all compliance constraints are global state
        # metrics (position concentration, sector exposure, cash reserve,
        # drawdown) that require multiple trades to resolve under the $30K
        # per-trade limit.  Tool-level validation already prevents invalid
        # actions (insufficient shares/cash, trade size limit).
        # All constraints are checked by the observer for is_stable.
        # Same pattern as cluster domain where queue/priority/rack_spread
        # are observer-only.
        checkers=[],
        allowed_actions=frozenset([
            "adjust_position",
            "liquidate_position",
        ]),
        # Teacher models (esp. Kimi) occasionally invert the `qty_delta` name;
        # redirect common variants so the first proposal doesn't fail on
        # spelling alone. Strategy consistency matters more than "punishing"
        # the typo with rejection feedback — the verifier still rejects
        # semantically wrong moves.
        param_aliases={
            "adjust_position": {
                "delta_qty": "qty_delta",
                "qty_change": "qty_delta",
                "qty": "qty_delta",
                "quantity": "qty_delta",
                "shares": "qty_delta",
            },
        },
        create_toolset=create_finance_toolset,
        create_observer=(lambda mgr: FinanceObserver(mgr)) if with_observer else None,
        build_system_prompt=build_finance_system_prompt,
        build_tool_schemas=build_finance_tool_schemas,
    )
