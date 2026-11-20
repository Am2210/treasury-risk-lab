# Treasury Risk Lab

Explore U.S. Treasury par yield curves and measure rate risk for a hypothetical
$10 million portfolio of 2-, 5-, and 10-year coupon bonds. The dashboard links
market observations, cash-flow pricing, stress tests, historical Value at Risk (VaR)
and Expected Shortfall (ES), principal component analysis (PCA), and Gaussian
Monte Carlo.

## Run locally

Use Python 3.11 or 3.12. Run these commands in the repository root.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe Project01.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

On macOS or Linux, use `.venv/bin/python` for the last four commands. Open
[the local dashboard](http://127.0.0.1:8501). For a first walkthrough, select
**Portfolio**, run the **Stress test** for +100 bp, then inspect **Historical risk**.
Open the sidebar to adjust the valuation date and faces. The historical view
offers a scenario-level CSV export in USD and percent of initial portfolio value.

## Reference run

The default curve is December 31, 2024. Portfolio faces are $2 million, $3 million,
and $5 million in 2Y, 5Y, and 10Y respectively. All six dashboard views use the
selected as-of date. The default historical sample contains 1,749 aligned yield shocks.

- Market value $10,000,000, modified duration 5.6894 years, and DV01 $5,689.44 per bp.
- 95% historical VaR $52,726 and Expected Shortfall $71,697.
- 99% historical VaR $83,932 and Expected Shortfall $101,288.

The validation command checks the archive row count, date range, selected par yields
against published reference values, par pricing, and price sensitivity. It exits nonzero when a check fails.
GitHub Actions is configured to run validation and tests on Ubuntu/Python 3.11 and
Windows/Python 3.12. The local reference run used Python 3.12.7, NumPy 1.26.4,
pandas 2.3.3, Streamlit 1.37.1, and Altair 5.0.1.

## Data and valuation date

The bundled [1990–2024 CSV](data/yield-curve-rates-1990-2024.csv) contains
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

## Repository map

- `data/` holds the bundled public CSV.
- `data_pipeline.py` reads the archive and builds aligned yield changes.
- `bond_pricing.py` handles pricing, sensitivities, and scenario repricing.
- `historical_var.py`, `monte_carlo.py`, `pca_analysis.py`, and
  `stress_testing.py` implement the risk methods.
- `app.py` is the local Streamlit entry point. It was created as a simplified
  replacement after the original interface was truncated during workbook export.
- `Project01.py` is the command-line validation entry point.
- `tests/` covers independent pricing checks, risk regressions, and dashboard interactions.

## Publication status

Code reuse and publication rights for the reconstructed source require confirmation
before public distribution or assignment of a license. The local project uses the
bundled public data and runs without enterprise services or credentials.
