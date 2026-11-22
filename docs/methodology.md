# Treasury Risk Lab methodology

[Return to the project README](../README.md)

## Data and valuation date

The bundled [1990–2024 CSV](../data/yield-curve-rates-1990-2024.csv) contains
8,757 observations from January 2, 1990 through December 31, 2024. The yields
are percentages, so 4.25 means 4.25% and a change of 0.01 percentage point
equals 1 bp. [U.S. Treasury CSV archive](https://home.treasury.gov/interest-rates-data-csv-archive/)

The dashboard uses the latest observation on or before the requested date with
complete 2Y, 5Y, and 10Y yields. It uses up to seven years of history ending on
that observation, including its closing yields. Both endpoints of a scenario
fall inside the window and must have all required tenors. Changes are computed
between adjacent published observations, which can span weekends or holidays.
Missing tenor values exclude affected pairs. [Treasury December 2024 rates](https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?field_tdr_date_value_month=202412&type=daily_treasury_yield_curve)

Bundled CSV SHA-256

`258a594eb64e9d21d9e8574b0b8580b0a5230dce56bb284ec6aa49a9a260897f`

## Model and assumptions

- **Pricing and sensitivity.** Each synthetic bond pays semiannual coupons.
  At valuation, coupon and YTM equal the same-tenor Treasury par yield, so
  each bond prices at face. Each cash flow is discounted at the bond's semiannual
  YTM. DV01 is half the price difference from a symmetric ±1 bp bump, reported
  as a positive sensitivity for the long-only portfolio. Portfolio duration and
  convexity are market-value weighted.
- **Stress testing.** Scenario P&L is the shocked price less the initial price.
  Full repricing is compared against the second-order approximation
  `ΔP ≈ P × (−Dmod × Δy + ½ × C × Δy²)`, calculated per bond using decimal
  yield changes and summed across positions.
- **Historical risk.** Apply aligned 2Y, 5Y, and 10Y changes to the same fixed
  portfolio. Loss equals negative P&L. VaR is the inverse empirical CDF at the
  chosen confidence. ES averages exactly the worst `1 − confidence` fraction,
  with fractional weight at the cutoff and equal treatment of ties. At 99%,
  the reference ES uses only 17.49 observation-equivalents. The signed tenor
  contributions use the portfolio tail weights and sum to 95% portfolio ES.
- **PCA and Monte Carlo.** PCA decomposes the sample covariance of centered
  daily yield changes in basis points. Factor labels are interpretations of
  the loadings. Monte Carlo estimates the sample mean and covariance in decimal
  yield units, uses a multivariate normal distribution and seed 42, and reprices
  each scenario using the same risk definitions.
- **Scope.** This is an educational snapshot of rate-shock P&L. Cash flows and
  remaining maturity stay fixed through each shock. The model omits accrued
  interest, carry, roll-down, observed bond prices and a bootstrapped spot curve.
  Gaussian calibration can underrepresent extreme moves. Out-of-sample risk
  backtesting remains future work.
