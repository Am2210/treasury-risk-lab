"""Monte Carlo simulation for portfolio risk estimation.

Generates correlated yield-change scenarios from a multivariate normal
distribution estimated from historical data, then fully reprices the portfolio.
"""

import numpy as np
import pandas as pd
from bond_pricing import build_portfolio, scenario_pnl
from data_pipeline import DEFAULT_WINDOW_END, DEFAULT_WINDOW_START, PORTFOLIO_TENORS, scenario_changes
from historical_var import compute_var_es


def estimate_covariance(
    df_yields: pd.DataFrame,
    tenors: list[str] | None = None,
    start_date: str = DEFAULT_WINDOW_START,
    end_date: str = DEFAULT_WINDOW_END,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Estimate mean and covariance of daily yield changes (decimal).

    Returns:
        mean_changes: Mean daily change for each tenor (decimal)
        cov_matrix: Covariance matrix of daily changes (decimal)
        tenors: List of tenor labels used
    """
    if tenors is None:
        tenors = PORTFOLIO_TENORS

    changes = scenario_changes(df_yields, tenors, start_date, end_date)[tenors] / 10_000
    if len(changes) < 2:
        raise ValueError("Monte Carlo calibration requires at least two aligned yield changes.")

    mean_changes = changes.mean().values
    cov_matrix = changes.cov().values

    return mean_changes, cov_matrix, tenors


def run_monte_carlo(
    df_yields: pd.DataFrame,
    yields_dict: dict,
    portfolio_spec: list | None = None,
    n_simulations: int = 10_000,
    start_date: str = DEFAULT_WINDOW_START,
    end_date: str = DEFAULT_WINDOW_END,
    seed: int = 42,
    valuation_date: str | None = None,
) -> dict:
    """Run Monte Carlo simulation with full repricing.

    Args:
        df_yields: Wide-format yield data
        yields_dict: Current yields by tenor (percentage)
        portfolio_spec: Portfolio positions
        n_simulations: Number of simulated scenarios
        start_date: Covariance estimation start
        end_date: Covariance estimation end
        seed: Random seed for reproducibility
        valuation_date: Optional cutoff for information available at valuation
    """
    if n_simulations < 1:
        raise ValueError("At least one Monte Carlo scenario is required.")
    portfolio = build_portfolio(yields_dict, portfolio_spec)
    tenors = portfolio["tenor"].drop_duplicates().tolist()
    if valuation_date is not None:
        end_date = min(pd.Timestamp(valuation_date), pd.Timestamp(end_date))
    mean_changes, cov_matrix, _ = estimate_covariance(df_yields, tenors, start_date, end_date)

    rng = np.random.default_rng(seed)
    shocks = rng.multivariate_normal(mean_changes, cov_matrix, n_simulations)
    pnl_array = scenario_pnl(portfolio, pd.DataFrame(shocks, columns=tenors)).sum(axis=1).to_numpy()
    base_value = portfolio["market_value"].sum()
    risk = compute_var_es(pd.DataFrame({"pnl_dollar": pnl_array, "portfolio_value_before": base_value}))

    return {
        **risk,
        "n_simulations": n_simulations,
        "base_portfolio_value": base_value,
        "pnl_array": pnl_array,
        "mean_changes": mean_changes,
        "cov_matrix": cov_matrix,
        "tenors": tenors,
    }


def compare_methods(hist_results: dict, mc_results: dict) -> pd.DataFrame:
    """Compare Historical Simulation vs Monte Carlo risk estimates."""
    rows = []
    for metric, hist_key, mc_key in [
        ("95% VaR ($)", "var_95_dollar", "var_95_dollar"),
        ("99% VaR ($)", "var_99_dollar", "var_99_dollar"),
        ("95% ES ($)", "es_95_dollar", "es_95_dollar"),
        ("99% ES ($)", "es_99_dollar", "es_99_dollar"),
        ("95% VaR (%)", "var_95_pct", "var_95_pct"),
        ("99% VaR (%)", "var_99_pct", "var_99_pct"),
    ]:
        hist_val = hist_results.get(hist_key, np.nan)
        mc_val = mc_results.get(mc_key, np.nan)
        diff = mc_val - hist_val if not (np.isnan(hist_val) or np.isnan(mc_val)) else np.nan
        rows.append({
            "Metric": metric,
            "Historical Simulation": hist_val,
            "Monte Carlo": mc_val,
            "Difference": diff,
        })
    return pd.DataFrame(rows)