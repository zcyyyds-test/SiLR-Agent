"""Portfolio compliance DomainConfig factory."""

from silr.core.config import DomainConfig
from .checkers import CashReserveChecker
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
        # Verifier checker: only per-action safety constraints.
        # CashReserve blocks buys that would deplete cash below 5%.
        # Position/Sector/Drawdown are multi-step global constraints
        # handled by observer for episode termination — same pattern as
        # cluster domain's queue/priority/rack_spread.
        checkers=[
            CashReserveChecker(),
        ],
        allowed_actions=frozenset([
            "adjust_position",
            "liquidate_position",
        ]),
        create_toolset=create_finance_toolset,
        create_observer=(lambda mgr: FinanceObserver(mgr)) if with_observer else None,
        build_system_prompt=build_finance_system_prompt,
        build_tool_schemas=build_finance_tool_schemas,
    )
