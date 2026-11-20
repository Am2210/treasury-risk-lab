"""Bond pricing, duration, convexity, and DV01 calculations.

All calculations use explicit semiannual cash flow discounting.
Par yields are used as coupon rates and approximate YTMs (simplified assumption).
"""

from math import exp, expm1, isfinite, log1p

import numpy as np
import pandas as pd
from typing import Optional


def _coupon_periods(maturity_years: float, ytm: float) -> int:
    periods = 2 * maturity_years
    if not isfinite(periods) or periods < 0 or periods != int(periods):
        raise ValueError("Maturity must be a nonnegative multiple of six months.")
    if not isfinite(ytm) or ytm <= -2:
        raise ValueError("YTM must be finite and greater than -200%.")
    return int(periods)


def bond_price(face: float, coupon_rate: float, ytm: float,
               maturity_years: float) -> float:
    """Price a semiannual coupon bond using present value of cash flows.

    Args:
        face: Face/par value
        coupon_rate: Annual coupon rate (decimal, e.g. 0.0425 for 4.25%)
        ytm: Yield to maturity (decimal)
        maturity_years: Time to maturity in years
    """
    n_periods = _coupon_periods(maturity_years, ytm)
    if n_periods <= 0:
        return face
    semi_coupon = face * coupon_rate / 2.0
    semi_ytm = ytm / 2.0

    if semi_ytm == 0:
        return semi_coupon * n_periods + face

    log_discount = -n_periods * log1p(semi_ytm)
    pv_coupons = semi_coupon * (-expm1(log_discount) / semi_ytm)
    pv_principal = face * exp(log_discount)
    return pv_coupons + pv_principal


def macaulay_duration(face: float, coupon_rate: float, ytm: float,
                      maturity_years: float) -> float:
    """Macaulay duration in years via weighted cash flow timing."""
    n_periods = _coupon_periods(maturity_years, ytm)
    if n_periods <= 0:
        return 0.0
    semi_coupon = face * coupon_rate / 2.0
    semi_ytm = ytm / 2.0
    price = bond_price(face, coupon_rate, ytm, maturity_years)
    if price == 0:
        return 0.0

    weighted_time = 0.0
    for t in range(1, n_periods + 1):
        cf = semi_coupon
        if t == n_periods:
            cf += face
        pv_cf = cf / (1 + semi_ytm) ** t
        weighted_time += (t / 2.0) * pv_cf  # t/2 converts periods to years
    return weighted_time / price


def modified_duration(face: float, coupon_rate: float, ytm: float,
                      maturity_years: float) -> float:
    """Modified duration = Macaulay duration / (1 + ytm/2)."""
    mac_dur = macaulay_duration(face, coupon_rate, ytm, maturity_years)
    return mac_dur / (1 + ytm / 2.0)


def convexity(face: float, coupon_rate: float, ytm: float,
              maturity_years: float) -> float:
    """Dollar convexity per dollar of price, in years^2."""
    n_periods = _coupon_periods(maturity_years, ytm)
    if n_periods <= 0:
        return 0.0
    semi_coupon = face * coupon_rate / 2.0
    semi_ytm = ytm / 2.0
    price = bond_price(face, coupon_rate, ytm, maturity_years)
    if price == 0:
        return 0.0

    conv_sum = 0.0
    for t in range(1, n_periods + 1):
        cf = semi_coupon
        if t == n_periods:
            cf += face
        pv_cf = cf / (1 + semi_ytm) ** t
        conv_sum += t * (t + 1) * pv_cf

    return conv_sum / (price * 4 * (1 + semi_ytm) ** 2)


def dv01(face: float, coupon_rate: float, ytm: float,
         maturity_years: float) -> float:
    """Positive DV01 magnitude using a symmetric one-basis-point bump."""
    price_up = bond_price(face, coupon_rate, ytm + 0.0001, maturity_years)
    price_down = bond_price(face, coupon_rate, ytm - 0.0001, maturity_years)
    return abs(price_down - price_up) / 2.0


def price_bond_details(face: float, coupon_rate: float, ytm: float,
                       maturity_years: float) -> dict:
    """Compute all analytics for a single bond."""
    px = bond_price(face, coupon_rate, ytm, maturity_years)
    mac = macaulay_duration(face, coupon_rate, ytm, maturity_years)
    mod = modified_duration(face, coupon_rate, ytm, maturity_years)
    cvx = convexity(face, coupon_rate, ytm, maturity_years)
    d01 = dv01(face, coupon_rate, ytm, maturity_years)

    return {
        "face_value": face,
        "coupon_rate": coupon_rate,
        "ytm": ytm,
        "maturity_years": maturity_years,
        "price": px,
        "market_value": px,  # for par bonds price ≈ face
        "macaulay_duration": mac,
        "modified_duration": mod,
        "convexity": cvx,
        "dv01": d01,
    }


def reprice_bond(face: float, coupon_rate: float, original_ytm: float,
                 new_ytm: float, maturity_years: float) -> dict:
    """Reprice a bond under a new yield; return old price, new price, and P&L."""
    old_price = bond_price(face, coupon_rate, original_ytm, maturity_years)
    new_price = bond_price(face, coupon_rate, new_ytm, maturity_years)
    pnl = new_price - old_price
    mod_dur = modified_duration(face, coupon_rate, original_ytm, maturity_years)
    cvx = convexity(face, coupon_rate, original_ytm, maturity_years)
    dy = new_ytm - original_ytm
    dur_approx = -mod_dur * dy * old_price
    cvx_approx = 0.5 * cvx * dy ** 2 * old_price
    approx_pnl = dur_approx + cvx_approx

    return {
        "old_price": old_price,
        "new_price": new_price,
        "pnl": pnl,
        "pnl_pct": pnl / old_price * 100 if old_price != 0 else 0,
        "duration_approx_pnl": approx_pnl,
        "approximation_error": pnl - approx_pnl,
    }


# ---------- Portfolio-level ----------

DEFAULT_PORTFOLIO = [
    {"tenor": "2 Yr", "maturity_years": 2, "face_value": 2_000_000, "weight": 0.20},
    {"tenor": "5 Yr", "maturity_years": 5, "face_value": 3_000_000, "weight": 0.30},
    {"tenor": "10 Yr", "maturity_years": 10, "face_value": 5_000_000, "weight": 0.50},
]


def build_portfolio(yields_dict: dict, portfolio_spec: Optional[list] = None) -> pd.DataFrame:
    """Construct portfolio from yield data and position specifications.

    Args:
        yields_dict: {tenor_label: yield_pct} e.g. {"2 Yr": 4.25, "5 Yr": 4.38, "10 Yr": 4.58}
        portfolio_spec: list of dicts with tenor, maturity_years, face_value
    """
    if portfolio_spec is None:
        portfolio_spec = DEFAULT_PORTFOLIO

    if not portfolio_spec:
        raise ValueError("Provide at least one portfolio position.")
    rows = []
    for pos in portfolio_spec:
        tenor = pos["tenor"]
        ytm_pct = yields_dict.get(tenor, np.nan)
        if pd.isna(ytm_pct) or not isfinite(ytm_pct):
            raise ValueError(f"Missing or nonfinite valuation yield for {tenor}.")
        ytm = ytm_pct / 100.0
        coupon = ytm  # par yield = coupon rate assumption
        face = pos["face_value"]
        if not isfinite(face) or face < 0:
            raise ValueError("Portfolio face values must be finite and nonnegative.")
        mat = pos["maturity_years"]
        details = price_bond_details(face, coupon, ytm, mat)
        details["tenor"] = tenor
        rows.append(details)

    df = pd.DataFrame(rows)
    total = df["market_value"].sum()
    df["weight"] = df["market_value"] / total if total else 0.0
    return df


def portfolio_summary(port_df: pd.DataFrame) -> dict:
    """Compute portfolio-level risk metrics from position-level DataFrame."""
    total_mv = port_df["market_value"].sum()
    if total_mv == 0:
        return {}

    mv_weights = port_df["market_value"] / total_mv
    weighted_duration = (mv_weights * port_df["modified_duration"]).sum()
    weighted_convexity = (mv_weights * port_df["convexity"]).sum()
    total_dv01 = port_df["dv01"].sum()
    weighted_mac_dur = (mv_weights * port_df["macaulay_duration"]).sum()

    return {
        "total_face_value": port_df["face_value"].sum(),
        "total_market_value": total_mv,
        "portfolio_modified_duration": weighted_duration,
        "portfolio_macaulay_duration": weighted_mac_dur,
        "portfolio_convexity": weighted_convexity,
        "portfolio_dv01": total_dv01,
        "position_count": len(port_df),
    }


def reprice_portfolio(port_df: pd.DataFrame, yield_shocks_bp: dict) -> pd.DataFrame:
    """Reprice each position under yield shocks (basis points).

    Args:
        port_df: Portfolio DataFrame from build_portfolio()
        yield_shocks_bp: {tenor: shock_in_bp} e.g. {"2 Yr": 100, "5 Yr": 100, "10 Yr": 100}
    """
    rows = []
    for _, pos in port_df.iterrows():
        shock_bp = yield_shocks_bp.get(pos["tenor"], 0)
        shock_decimal = shock_bp / 10000.0
        new_ytm = pos["ytm"] + shock_decimal
        result = reprice_bond(
            pos["face_value"], pos["coupon_rate"],
            pos["ytm"], new_ytm, pos["maturity_years"]
        )
        result["tenor"] = pos["tenor"]
        result["shock_bp"] = shock_bp
        result["face_value"] = pos["face_value"]
        rows.append(result)

    return pd.DataFrame(rows)

def scenario_pnl(portfolio: pd.DataFrame, changes_decimal: pd.DataFrame) -> pd.DataFrame:
    """Reprice fixed coupon cash flows and aggregate scenario P&L by tenor."""
    pnl = {}
    for pos in portfolio.itertuples(index=False):
        position_pnl = np.array([
            bond_price(pos.face_value, pos.coupon_rate, pos.ytm + shock, pos.maturity_years)
            - pos.price
            for shock in changes_decimal[pos.tenor]
        ])
        if pos.tenor in pnl:
            pnl[pos.tenor] += position_pnl
        else:
            pnl[pos.tenor] = position_pnl
    return pd.DataFrame(pnl, index=changes_decimal.index)
