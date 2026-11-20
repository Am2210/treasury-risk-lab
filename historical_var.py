"""Historical one-observation VaR and Expected Shortfall via full repricing."""

import numpy as np
import pandas as pd

from bond_pricing import build_portfolio, scenario_pnl
from data_pipeline import DEFAULT_WINDOW_END, DEFAULT_WINDOW_START, scenario_changes


def _historical_pnl(df_yields, valuation_date, yields_dict, portfolio_spec, start_date, end_date):
    portfolio = build_portfolio(yields_dict, portfolio_spec)
    tenors = portfolio["tenor"].drop_duplicates().tolist()
    cutoff = min(pd.Timestamp(valuation_date), pd.Timestamp(end_date))
    changes = scenario_changes(df_yields, tenors, start_date, cutoff)
    if changes.empty:
        raise ValueError("No aligned yield changes in the historical window.")
    pnl = scenario_pnl(portfolio, changes.set_index("date")[tenors] / 10_000)
    return portfolio, pnl


def compute_historical_scenarios(
    df_yields: pd.DataFrame,
    valuation_date: str,
    yields_dict: dict,
    portfolio_spec: list | None = None,
    start_date: str = DEFAULT_WINDOW_START,
    end_date: str = DEFAULT_WINDOW_END,
) -> pd.DataFrame:
    """Reprice the portfolio under aligned shocks known by the valuation date."""
    portfolio, tenor_pnl = _historical_pnl(
        df_yields, valuation_date, yields_dict, portfolio_spec, start_date, end_date
    )
    base_value = portfolio["market_value"].sum()
    pnl = tenor_pnl.sum(axis=1).to_numpy()
    return pd.DataFrame({
        "scenario_date": tenor_pnl.index,
        "portfolio_value_before": base_value,
        "portfolio_value_after": base_value + pnl,
        "pnl_dollar": pnl,
        "pnl_pct": pnl / base_value * 100 if base_value else 0.0,
    })


def _empirical_tail(losses: np.ndarray, confidence: float) -> tuple[float, np.ndarray]:
    """Return inverse-ECDF VaR and weights for an exact-mass upper tail."""
    if not 0 < confidence < 1:
        raise ValueError("Confidence levels must be between zero and one.")
    ordered = np.sort(losses)
    tail_mass = (1 - confidence) * len(ordered)
    whole = round(tail_mass)
    if whole and abs(tail_mass - whole) <= 1e-12:
        tail_mass = float(whole)
    rank = max(len(ordered) - int(tail_mass), 1)
    var = ordered[rank - 1]

    # Split boundary mass equally among ties so attribution is order invariant.
    # Use the first observation in the ES tail, which can differ from VaR
    # when the requested tail mass is a whole number of observations.
    boundary = ordered[-int(np.ceil(tail_mass))]
    above = losses > boundary
    tied = losses == boundary
    weights = above.astype(float)
    weights[tied] = (tail_mass - above.sum()) / tied.sum()
    return float(var), weights / tail_mass


def compute_var_es(
    scenarios_df: pd.DataFrame, confidence_levels: tuple[float, ...] = (0.95, 0.99)
) -> dict:
    """Summarize empirical losses. Negative VaR or ES denotes a gain."""
    if scenarios_df.empty:
        raise ValueError("At least one repricing scenario is required.")
    pnl = scenarios_df["pnl_dollar"].to_numpy(dtype=float)
    values = scenarios_df["portfolio_value_before"].to_numpy(dtype=float)
    if not np.isfinite(pnl).all() or not np.isfinite(values).all():
        raise ValueError("Scenario P&L and valuation values must be finite.")
    portfolio_value = values[0]
    if portfolio_value <= 0 or not np.allclose(values, portfolio_value, rtol=0, atol=1e-8):
        raise ValueError("Scenarios must share a positive base portfolio value.")

    results = {"n_scenarios": len(pnl), "portfolio_value": portfolio_value}
    losses = -pnl
    for confidence in confidence_levels:
        var, weights = _empirical_tail(losses, confidence)
        es = float(losses @ weights)
        label = f"{confidence * 100:g}"
        results.update({
            f"var_{label}_dollar": var,
            f"var_{label}_pct": var / portfolio_value * 100,
            f"es_{label}_dollar": es,
            f"es_{label}_pct": es / portfolio_value * 100,
        })
    series = pd.Series(pnl)
    results.update(
        pnl_mean=float(pnl.mean()), pnl_std=float(pnl.std()),
        pnl_min=float(pnl.min()), pnl_max=float(pnl.max()),
        pnl_skew=float(series.skew()), pnl_kurtosis=float(series.kurtosis()),
    )
    return results


def risk_contribution_by_tenor(
    df_yields: pd.DataFrame,
    valuation_date: str,
    yields_dict: dict,
    portfolio_spec: list | None = None,
    start_date: str = DEFAULT_WINDOW_START,
    end_date: str = DEFAULT_WINDOW_END,
) -> pd.DataFrame:
    """Report standalone 95% VaR and additive 95% portfolio ES by tenor."""
    portfolio, pnl = _historical_pnl(
        df_yields, valuation_date, yields_dict, portfolio_spec, start_date, end_date
    )
    losses = -pnl
    portfolio_var, weights = _empirical_tail(losses.sum(axis=1).to_numpy(), 0.95)
    standalone = {tenor: _empirical_tail(losses[tenor].to_numpy(), 0.95)[0] for tenor in pnl}
    result = pd.DataFrame({
        "face_value": portfolio.groupby("tenor")["face_value"].sum(),
        "standalone_var_95": pd.Series(standalone),
        "pnl_std": pnl.std(ddof=0),
        "pnl_mean": pnl.mean(),
        "es_contribution_95": losses.T @ weights,
    }).reindex(pnl.columns)
    total_standalone = result["standalone_var_95"].sum()
    # Legacy field is a normalized standalone share, independent of portfolio ES.
    result["var_contribution_pct"] = result["standalone_var_95"] / total_standalone * 100 if total_standalone > 0 else 0
    result["diversification_benefit"] = total_standalone - portfolio_var
    return result.rename_axis("tenor").reset_index()
