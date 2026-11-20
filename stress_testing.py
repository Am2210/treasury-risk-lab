"""Interest-rate stress testing with predefined and custom scenarios."""

import pandas as pd
from bond_pricing import (
    reprice_portfolio, build_portfolio, portfolio_summary,
    DEFAULT_PORTFOLIO,
)


# Predefined scenarios (shocks in basis points)
PRESET_SCENARIOS = {
    "A: Parallel +100bp": {"2 Yr": 100, "5 Yr": 100, "10 Yr": 100},
    "B: Parallel -100bp": {"2 Yr": -100, "5 Yr": -100, "10 Yr": -100},
    "C: Bear Steepener": {"2 Yr": 25, "5 Yr": 75, "10 Yr": 125},
    "D: Bear Flattener": {"2 Yr": 125, "5 Yr": 75, "10 Yr": 25},
}


def run_stress_scenario(
    yields_dict: dict,
    shock_bp: dict,
    portfolio_spec: list | None = None,
) -> dict:
    """Run a single stress scenario on the portfolio.

    Args:
        yields_dict: Current yields by tenor (percentage, e.g. 4.25)
        shock_bp: Yield shocks by tenor (basis points, e.g. 100)
        portfolio_spec: Portfolio positions

    Returns:
        Dict with scenario results including full repricing and duration-plus-convexity approximation.
    """
    if portfolio_spec is None:
        portfolio_spec = DEFAULT_PORTFOLIO

    port_df = build_portfolio(yields_dict, portfolio_spec)
    original_summary = portfolio_summary(port_df)
    original_mv = original_summary.get("total_market_value", 0.0)

    # Full repricing
    repriced_df = reprice_portfolio(port_df, shock_bp)

    total_new_mv = repriced_df["new_price"].sum()
    total_pnl = total_new_mv - original_mv
    total_pnl_pct = total_pnl / original_mv * 100 if original_mv != 0 else 0

    # Duration-based approximation for the whole portfolio
    dur_approx_pnl = repriced_df["duration_approx_pnl"].sum()
    approx_error = total_pnl - dur_approx_pnl

    # Per-position breakdown
    position_details = []
    for _, row in repriced_df.iterrows():
        position_details.append({
            "tenor": row["tenor"],
            "face_value": row["face_value"],
            "shock_bp": row["shock_bp"],
            "old_price": row["old_price"],
            "new_price": row["new_price"],
            "pnl_dollar": row["pnl"],
            "pnl_pct": row["pnl_pct"],
            "duration_approx_pnl": row["duration_approx_pnl"],
            "approximation_error": row["approximation_error"],
            "risk_contribution_pct": (
                row["pnl"] / total_pnl * 100
                if total_pnl != 0 else None
            ),
        })

    return {
        "original_portfolio_value": original_mv,
        "shocked_portfolio_value": total_new_mv,
        "total_pnl_dollar": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "duration_approx_pnl": dur_approx_pnl,
        "approximation_error": approx_error,
        "approximation_error_pct": (
            approx_error / abs(total_pnl) * 100 if total_pnl != 0 else 0
        ),
        "position_details": pd.DataFrame(position_details),
    }


def run_all_presets(yields_dict: dict,
                    portfolio_spec: list | None = None) -> dict:
    """Run all predefined stress scenarios."""
    results = {}
    for name, shocks in PRESET_SCENARIOS.items():
        results[name] = run_stress_scenario(yields_dict, shocks, portfolio_spec)
    return results


def stress_summary_table(results: dict) -> pd.DataFrame:
    """Create summary comparison table across scenarios."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Scenario": name,
            "Original Value ($)": res["original_portfolio_value"],
            "Shocked Value ($)": res["shocked_portfolio_value"],
            "P&L ($)": res["total_pnl_dollar"],
            "P&L (%)": res["total_pnl_pct"],
            "Duration + Convexity P&L ($)": res["duration_approx_pnl"],
            "Approx Error ($)": res["approximation_error"],
        })
    return pd.DataFrame(rows)