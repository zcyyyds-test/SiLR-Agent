"""Portfolio compliance DomainConfig factory."""

from silr.core.config import DomainConfig
from .checkers import (
    PositionConcentrationChecker,
    SectorExposureChecker,
    CashReserveChecker,
)
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
        # Verifier checkers: only constraints fixable by trading actions.
        # DrawdownChecker is observer-only — drawdown is a market condition
        # that trading cannot resolve (selling doesn't change portfolio value),
        # same pattern as cluster domain's queue/priority/rack_spread.
        checkers=[
            PositionConcentrationChecker(),
            SectorExposureChecker(),
            CashReserveChecker(),
        ],
        allowed_actions=frozenset([
            "adjust_position",
            "liquidate_position",
            "rebalance_sector",
        ]),
        create_toolset=create_finance_toolset,
        create_observer=(lambda mgr: FinanceObserver(mgr)) if with_observer else None,
        build_system_prompt=build_finance_system_prompt,
        build_tool_schemas=build_finance_tool_schemas,
    )
