"""Demo: Portfolio compliance recovery with SiLR verification.

Shows the domain working end-to-end:
1. Create portfolio + apply market stress scenario
2. Verify a rebalancing action via SiLR
3. Observer produces LLM-readable compliance state
"""

from domains.finance import (
    FinanceManager,
    build_finance_domain_config,
    FinanceScenarioLoader,
)
from silr.verifier import SiLRVerifier


def main():
    # Setup
    mgr = FinanceManager()
    config = build_finance_domain_config()
    loader = FinanceScenarioLoader()

    # Apply scenario: NVDA surge causes concentration breach
    scenario = loader.load("nvda_concentration")
    loader.setup_episode(mgr, scenario)
    print(f"Scenario: {scenario.description}")
    print(f"Difficulty: {scenario.difficulty}")

    # Show initial violations
    state = mgr.system_state
    print(f"\nPortfolio value: ${state['portfolio_value']:,.2f}")
    print(f"Cash: ${state['cash']:,.2f} ({state['cash']/state['portfolio_value']*100:.1f}%)")
    print("Positions:")
    for sym in sorted(state["positions"]):
        pos = state["positions"][sym]
        if pos["qty"] > 0:
            print(f"  {sym}: {pos['qty']} shares @ ${pos['price']:.2f}"
                  f" = ${pos['qty']*pos['price']:,.2f}"
                  f" ({pos['weight']*100:.1f}%)")

    # Create verifier (no checkers — constraints are observer-based)
    verifier = SiLRVerifier(mgr, domain_config=config)

    # Try selling some NVDA to reduce concentration
    action = {
        "tool_name": "adjust_position",
        "params": {"symbol": "NVDA", "qty_delta": -100},
    }
    result = verifier.verify(action)
    print(f"\nAction: sell 100 NVDA")
    print(f"Verdict: {result.verdict.value}")

    # Observer output shows compliance state
    observer = config.create_observer(mgr)
    obs = observer.observe()
    print(f"\nSystem stable: {obs.is_stable}")
    print(f"Active violations: {len(obs.violations)}")
    for v in obs.violations:
        print(f"  [{v['severity']}] {v['type']}: {v['detail']}")
    print(f"\nCompressed observation (first 200 chars):")
    print(f"  {obs.compressed_json[:200]}...")


if __name__ == "__main__":
    main()
