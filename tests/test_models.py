"""Independent pricing checks and rate-risk regression tests."""

import unittest

import numpy as np
import pandas as pd

from bond_pricing import bond_price, build_portfolio, convexity, dv01, modified_duration
from data_pipeline import compute_daily_changes, get_yield_curve_for_date
from historical_var import compute_historical_scenarios, compute_var_es, risk_contribution_by_tenor
from monte_carlo import estimate_covariance, run_monte_carlo
from pca_analysis import prepare_pca_data, run_pca
from stress_testing import run_stress_scenario


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tenors = ["2 Yr", "5 Yr", "10 Yr"]
        self.yields = dict(zip(self.tenors, [4.25, 4.38, 4.58]))
        self.data = pd.DataFrame({
            "date": pd.date_range("2024-01-02", periods=6, freq="B"),
            "2 Yr": [4.20, 4.40, 4.10, 4.25, 4.21, 5.15],
            "5 Yr": [4.30, 4.25, 4.32, 4.38, 4.27, 5.25],
            "10 Yr": [4.35, 4.45, 4.51, 4.58, 4.55, 5.95],
        })

    def test_cash_flow_price_at_zero_near_zero_and_negative_yields(self):
        for ytm in (-0.01, 0, 1e-14, 0.037):
            expected = sum(20 / (1 + ytm / 2) ** n for n in range(1, 7))
            expected += 1000 / (1 + ytm / 2) ** 6
            self.assertAlmostEqual(bond_price(1000, 0.04, ytm, 3), expected, places=9)
        with self.assertRaises(ValueError):
            bond_price(1000, 0.04, 0.03, 2.1)

    def test_duration_convexity_and_dv01_match_price_derivatives(self):
        face, coupon, ytm, maturity = 1000, 0.035, 0.041, 5
        bump = 0.0001
        price = bond_price(face, coupon, ytm, maturity)
        up = bond_price(face, coupon, ytm + bump, maturity)
        down = bond_price(face, coupon, ytm - bump, maturity)
        self.assertAlmostEqual(modified_duration(face, coupon, ytm, maturity),
                               (down - up) / (2 * bump * price), places=5)
        self.assertAlmostEqual(convexity(face, coupon, ytm, maturity),
                               (down + up - 2 * price) / (bump ** 2 * price), places=5)
        self.assertAlmostEqual(dv01(face, coupon, ytm, maturity), (down - up) / 2)

    def test_valuation_date_and_missing_yield(self):
        date = get_yield_curve_for_date(self.data, "2024-01-06")["date"].iloc[0]
        self.assertEqual(date, pd.Timestamp("2024-01-05"))
        with self.assertRaisesRegex(ValueError, "10 Yr"):
            build_portfolio({"2 Yr": 4, "5 Yr": 4})

    def test_missing_observation_never_bridges_daily_change(self):
        data = self.data.copy()
        data.loc[2, "5 Yr"] = np.nan
        changes = compute_daily_changes(data, self.tenors)
        pca_changes, _ = prepare_pca_data(data, self.tenors)
        self.assertEqual(changes["date"].to_list(), data.loc[[1, 4, 5], "date"].to_list())
        pd.testing.assert_frame_equal(changes, pca_changes)
        _, covariance, _ = estimate_covariance(data, self.tenors)
        np.testing.assert_allclose(covariance, (changes[self.tenors] / 10_000).cov())

    def test_historical_cutoff_and_stress_agree_for_same_shock(self):
        scenarios = compute_historical_scenarios(self.data, "2024-01-05", self.yields)
        self.assertEqual(scenarios["scenario_date"].max(), pd.Timestamp("2024-01-05"))
        shocks = ((self.data.loc[1, self.tenors] - self.data.loc[0, self.tenors]) * 100).to_dict()
        stress = run_stress_scenario(self.yields, shocks)
        self.assertAlmostEqual(scenarios["pnl_dollar"].iloc[0], stress["total_pnl_dollar"])
        self.assertAlmostEqual(stress["position_details"]["risk_contribution_pct"].sum(), 100)
        with self.assertRaisesRegex(ValueError, "No aligned"):
            compute_historical_scenarios(self.data, "2024-01-02", self.yields)

    def test_empirical_var_and_fractional_es_with_ties(self):
        scenarios = pd.DataFrame({"pnl_dollar": [4, -10, -10, -40], "portfolio_value_before": 100})
        results = compute_var_es(scenarios, [0.5, 0.625, 0.75])
        self.assertEqual(results["var_50_dollar"], 10)
        self.assertAlmostEqual(results["es_50_dollar"], 25)
        self.assertAlmostEqual(results["es_62.5_dollar"], (40 + 5) / 1.5)
        self.assertEqual(results["var_75_dollar"], 10)
        self.assertAlmostEqual(results["es_75_dollar"], 40)

    def test_es_contributions_reconcile(self):
        scenarios = compute_historical_scenarios(self.data, "2024-01-05", self.yields)
        attribution = risk_contribution_by_tenor(self.data, "2024-01-05", self.yields)
        self.assertAlmostEqual(attribution["es_contribution_95"].sum(),
                               compute_var_es(scenarios)["es_95_dollar"])

    def test_pca_explains_covariance_with_orthonormal_loadings(self):
        results = run_pca(self.data, self.tenors)
        loadings = results["loadings_df"][["PC1", "PC2", "PC3"]].to_numpy()
        np.testing.assert_allclose(loadings.T @ loadings, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(loadings @ np.diag(results["eigenvalues"]) @ loadings.T,
                                   results["cov_matrix"], atol=1e-10)
        self.assertAlmostEqual(results["explained_variance_ratio"].sum(), 1)

    def test_monte_carlo_is_seeded_and_respects_calibration_cutoff(self):
        options = dict(n_simulations=100, seed=42, valuation_date="2024-01-05")
        full = run_monte_carlo(self.data, self.yields, **options)
        filtered = run_monte_carlo(self.data.iloc[:4], self.yields, **options)
        np.testing.assert_array_equal(full["pnl_array"], filtered["pnl_array"])
        self.assertGreaterEqual(full["es_99_dollar"], full["var_99_dollar"])


if __name__ == "__main__":
    unittest.main()
