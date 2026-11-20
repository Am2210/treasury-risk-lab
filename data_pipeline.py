"""Treasury yield curve data ingestion, cleaning, and transformation."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Tenor mapping: column name -> numeric years
TENOR_MAP = {
    "1 Mo": 1/12, "2 Mo": 2/12, "3 Mo": 3/12, "4 Mo": 4/12,
    "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0,
    "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}

DATA_PATH = Path(__file__).resolve().parent / "data" / "yield-curve-rates-1990-2024.csv"
DEFAULT_WINDOW_START = "2018-01-01"
DEFAULT_WINDOW_END = "2024-12-31"

CORE_TENORS = ["2 Yr", "5 Yr", "10 Yr", "30 Yr"]
PCA_TENORS = ["2 Yr", "3 Yr", "5 Yr", "7 Yr", "10 Yr", "30 Yr"]
PORTFOLIO_TENORS = ["2 Yr", "5 Yr", "10 Yr"]


def load_raw_csv(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load raw CSV, parse dates, sort chronologically."""
    if filepath is None:
        filepath = DATA_PATH

    df = pd.read_csv(filepath)
    if df.empty:
        raise ValueError("Yield CSV contains no observations.")
    df.columns = df.columns.str.strip()
    df.rename(columns={"Date": "date"}, inplace=True)

    # The archive uses two-digit years within 1990-2024. Use a fixed 1990 pivot.
    date_parts = df["date"].str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{2}|\d{4})$")
    month, day, year_text = (date_parts[col] for col in range(3))
    year = pd.to_numeric(year_text, errors="raise")
    short_year = year_text.str.len().eq(2)
    year = year.where(~short_year, year + np.where(year >= 90, 1900, 2000))
    df["date"] = pd.to_datetime({"year": year, "month": month, "day": day})
    if df["date"].isna().any() or df["date"].duplicated().any():
        raise ValueError("Yield data must have unique, nonmissing observation dates.")

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    yield_cols = [c for c in df.columns if c != "date"]
    for col in yield_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")
        if np.isinf(df[col]).any():
            raise ValueError(f"Nonfinite yield in {col}.")

    return df


def get_wide_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned wide-format yield table (date + tenor columns in %)."""
    return df.copy()


def get_long_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize wide table to long format with tenor metadata."""
    yield_cols = [c for c in df.columns if c != "date"]
    long = df.melt(id_vars=["date"], value_vars=yield_cols,
                   var_name="tenor_label", value_name="yield_pct")
    long["tenor_years"] = long["tenor_label"].map(TENOR_MAP)
    long["yield_decimal"] = long["yield_pct"] / 100.0
    long.dropna(subset=["yield_pct"], inplace=True)
    long.sort_values(["date", "tenor_years"], inplace=True)
    long.reset_index(drop=True, inplace=True)
    return long


def compute_daily_changes(df: pd.DataFrame, tenors: list[str]) -> pd.DataFrame:
    """Compute daily yield changes in basis points for specified tenors."""
    if not tenors or len(set(tenors)) != len(tenors):
        raise ValueError("Provide a nonempty list of unique tenors.")
    subset = df[["date"] + tenors].sort_values("date")
    if subset["date"].isna().any() or subset["date"].duplicated().any():
        raise ValueError("Yield changes require unique, nonmissing observation dates.")
    changes = subset[tenors].diff()
    changes_bp = changes * 100  # percent to basis points
    changes_bp["date"] = subset["date"]
    changes_bp = changes_bp.iloc[1:]  # drop first NaN row
    changes_bp.dropna(subset=tenors, inplace=True)
    changes_bp.reset_index(drop=True, inplace=True)
    return changes_bp


def scenario_changes(
    df: pd.DataFrame, tenors: list[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Return aligned changes in bp with both endpoints inside the window."""
    window = filter_date_range(df, start_date, end_date)
    return compute_daily_changes(window, tenors)


def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """Compute key yield curve spreads in basis points."""
    spreads = pd.DataFrame({"date": df["date"]})
    if "10 Yr" in df.columns and "2 Yr" in df.columns:
        spreads["10Y_2Y"] = (df["10 Yr"] - df["2 Yr"]) * 100
    if "10 Yr" in df.columns and "3 Mo" in df.columns:
        spreads["10Y_3M"] = (df["10 Yr"] - df["3 Mo"]) * 100
    if "30 Yr" in df.columns and "10 Yr" in df.columns:
        spreads["30Y_10Y"] = (df["30 Yr"] - df["10 Yr"]) * 100
    spreads.dropna(how="all", subset=[c for c in spreads.columns if c != "date"], inplace=True)
    return spreads


def compute_rolling_volatility(df: pd.DataFrame, tenors: list[str],
                                window: int = 21) -> pd.DataFrame:
    """Compute rolling annualized yield-change volatility (basis points)."""
    changes_bp = compute_daily_changes(df, tenors)
    vol = changes_bp[tenors].rolling(window=window).std() * np.sqrt(252)
    vol["date"] = changes_bp["date"]
    vol.dropna(inplace=True)
    return vol


def data_quality_report(df: pd.DataFrame) -> dict:
    """Generate a data quality summary."""
    yield_cols = [c for c in df.columns if c != "date"]
    report = {
        "total_rows": len(df),
        "date_range": (df["date"].min().strftime("%Y-%m-%d"),
                       df["date"].max().strftime("%Y-%m-%d")),
        "duplicate_dates": int(df["date"].duplicated().sum()),
        "missing_by_tenor": {col: int(df[col].isna().sum()) for col in yield_cols},
        "coverage_by_tenor": {},
        "yield_stats": {},
    }
    for col in yield_cols:
        valid = df[col].dropna()
        if len(valid) > 0:
            report["coverage_by_tenor"][col] = {
                "first_date": df.loc[valid.index[0], "date"].strftime("%Y-%m-%d"),
                "last_date": df.loc[valid.index[-1], "date"].strftime("%Y-%m-%d"),
                "count": len(valid),
            }
            report["yield_stats"][col] = {
                "min": round(valid.min(), 4),
                "max": round(valid.max(), 4),
                "mean": round(valid.mean(), 4),
                "std": round(valid.std(), 4),
            }
    return report


def validate_known_values(df: pd.DataFrame) -> dict:
    """Validate Dec 31, 2024 yields against known reference values."""
    target = df[df["date"] == pd.Timestamp("2024-12-31")]
    expected = {"2 Yr": 4.25, "5 Yr": 4.38, "10 Yr": 4.58, "30 Yr": 4.78}
    results = {}
    if len(target) == 1:
        row = target.iloc[0]
        for tenor, exp_val in expected.items():
            actual = row.get(tenor)
            results[tenor] = {
                "expected": exp_val,
                "actual": round(actual, 2) if pd.notna(actual) else None,
                "match": np.isclose(actual, exp_val, rtol=0, atol=1e-9) if pd.notna(actual) else False,
            }
    else:
        results["error"] = f"Found {len(target)} rows for 2024-12-31"
    return results


def filter_date_range(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter DataFrame to a date range."""
    mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
    return df[mask].copy()


def get_yield_curve_for_date(df: pd.DataFrame, target_date) -> pd.DataFrame:
    """Get the yield curve from the latest observation on or before the date."""
    target_date = pd.Timestamp(target_date)
    row = df[df["date"] == target_date]
    if len(row) == 0:
        prior = df.loc[df["date"] <= target_date]
        if prior.empty:
            raise ValueError("No yield observation on or before the requested date.")
        row = prior.sort_values("date").tail(1)
    yield_cols = [c for c in df.columns if c != "date"]
    curve = []
    for col in yield_cols:
        val = row[col].values[0]
        if pd.notna(val):
            curve.append({"tenor_label": col, "tenor_years": TENOR_MAP[col],
                         "yield_pct": val, "date": row["date"].values[0]})
    return pd.DataFrame(curve, columns=["tenor_label", "tenor_years", "yield_pct", "date"]).sort_values("tenor_years")


def monthly_averages(df: pd.DataFrame, tenors: list[str]) -> pd.DataFrame:
    """Compute monthly average yields."""
    monthly = df.set_index("date")[tenors].resample("ME").mean()
    monthly.reset_index(inplace=True)
    return monthly


def annual_averages(df: pd.DataFrame, tenors: list[str]) -> pd.DataFrame:
    """Compute annual average yields."""
    annual = df.set_index("date")[tenors].resample("YE").mean()
    annual["year"] = annual.index.year
    annual.reset_index(inplace=True)
    return annual