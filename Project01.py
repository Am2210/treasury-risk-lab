"""Validate the bundled data and default portfolio. Run with python Project01.py."""

from math import isclose

from bond_pricing import bond_price, build_portfolio, portfolio_summary
from data_pipeline import PORTFOLIO_TENORS, data_quality_report, load_raw_csv, validate_known_values


def main() -> int:
    df = load_raw_csv()
    report = data_quality_report(df)
    print(f"Rows loaded: {len(df):,}")
    print(f"Date range: {' to '.join(report['date_range'])}")
    print(f"Duplicate dates: {report['duplicate_dates']}")
    print(f"Missing values by tenor: {report['missing_by_tenor']}")
    validation = validate_known_values(df)
    if (
        len(df) != 8757
        or report["date_range"] != ("1990-01-02", "2024-12-31")
        or report["duplicate_dates"]
        or "error" in validation
        or not all(value["match"] for value in validation.values())
    ):
        print(f"FAIL: Treasury archive validation: {validation}")
        return 1

    print("PASS: 2024-12-31 reference yields")
    for tenor, value in validation.items():
        print(f"  {tenor}: {value['actual']:.2f}%")

    row = df.loc[df["date"].eq("2024-12-31")].iloc[0]
    summary = portfolio_summary(build_portfolio(row[PORTFOLIO_TENORS].to_dict()))
    print(f"Portfolio market value: ${summary['total_market_value']:,.2f}")
    print(f"Modified duration: {summary['portfolio_modified_duration']:.4f} years")
    print(f"DV01: ${summary['portfolio_dv01']:,.2f}/bp")
    print(f"Convexity: {summary['portfolio_convexity']:.4f} years^2")
    higher_yield_price = bond_price(1_000_000, 0.0458, 0.0558, 10)
    if (
        not isclose(summary["total_market_value"], 10_000_000, abs_tol=0.01)
        or not 0 < higher_yield_price < 1_000_000
    ):
        print("FAIL: Portfolio par value or price sensitivity")
        return 1
    print("PASS: Portfolio par value and price sensitivity")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
