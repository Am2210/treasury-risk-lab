"""Basic sanity checks for a portfolio demonstration—not a full model audit."""
import unittest

import numpy as np

from historical_var import compute_historical_scenarios, compute_var_es
from data_pipeline import load_raw_csv, validate_known_values
from bond_pricing import bond_price, build_portfolio, portfolio_summary
from stress_testing import run_stress_scenario, PRESET_SCENARIOS

class TreasuryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_raw_csv()
        row = cls.data.loc[cls.data.date.eq("2024-12-31")].iloc[0]
        cls.yields = {t: float(row[t]) for t in ("2 Yr", "5 Yr", "10 Yr")}

    def test_known_values_and_rows(self):
        self.assertEqual(len(self.data), 8757)
        self.assertTrue(all(value["match"] for value in validate_known_values(self.data).values()))
        self.assertEqual(self.data.date.duplicated().sum(), 0)

    def test_bond_price_decreases_when_yield_increases(self):
        p0 = bond_price(1_000_000, 0.0458, 0.0458, 10)
        p1 = bond_price(1_000_000, 0.0458, 0.0558, 10)
        self.assertLess(p1, p0)

    def test_par_bond_pricing(self):
        p = bond_price(1_000_000, 0.0458, 0.0458, 10)
        self.assertAlmostEqual(p, 1_000_000, places=5)

    def test_portfolio_summary(self):
        summary = portfolio_summary(build_portfolio(self.yields))
        self.assertAlmostEqual(summary["total_market_value"], 10_000_000, places=4)
        self.assertGreater(summary["portfolio_dv01"], 0)

    def test_rate_shock_loss(self):
        res = run_stress_scenario(self.yields, PRESET_SCENARIOS["A: Parallel +100bp"])
        self.assertLess(res["total_pnl_dollar"], 0)
        self.assertLess(res["shocked_portfolio_value"], res["original_portfolio_value"])


    def test_reference_risk_matches_independent_cash_flow_repricing(self):
        tenors = ["2 Yr", "5 Yr", "10 Yr"]
        face_amounts = (2_000_000, 3_000_000, 5_000_000)
        coupons = np.array([self.yields[tenor] / 100 for tenor in tenors])

        def present_value(shocks):
            total = 0.0
            for col, (years, face, coupon) in enumerate(zip((2, 5, 10), face_amounts, coupons)):
                periods = np.arange(1, years * 2 + 1)
                cash_flows = np.full(len(periods), face * coupon / 2)
                cash_flows[-1] += face
                yields = coupon + shocks[..., col, None]
                total += (cash_flows / (1 + yields / 2) ** periods).sum(axis=-1)
            return total

        base_value = present_value(np.zeros(3))
        up = present_value(np.full(3, 0.0001))
        down = present_value(np.full(3, -0.0001))
        summary = portfolio_summary(build_portfolio(self.yields))
        duration = (present_value(np.full(3, -0.00001)) - present_value(np.full(3, 0.00001)))
        duration /= 2 * 0.00001 * base_value
        self.assertAlmostEqual(summary["portfolio_modified_duration"], duration, places=6)
        self.assertEqual(round(duration, 4), 5.6894)
        self.assertAlmostEqual(summary["portfolio_dv01"], (down - up) / 2, places=7)
        self.assertEqual(round((down - up) / 2, 2), 5689.44)
        self.assertAlmostEqual(summary["portfolio_convexity"],
                               (down + up - 2 * base_value) / (base_value * 0.0001 ** 2),
                               places=4)

        shock = run_stress_scenario(self.yields, PRESET_SCENARIOS["A: Parallel +100bp"])
        self.assertAlmostEqual(shock["total_pnl_dollar"],
                               present_value(np.full(3, 0.01)) - base_value, places=7)

        window = self.data.loc[self.data.date.between("2018-01-01", "2024-12-31")]
        changes = window.set_index("date")[tenors].diff().dropna() / 100
        independent_pnl = present_value(changes.to_numpy()) - base_value
        scenarios = compute_historical_scenarios(self.data, "2024-12-31", self.yields)
        self.assertEqual(len(scenarios), 1749)
        np.testing.assert_array_equal(scenarios["scenario_date"].to_numpy(), changes.index.to_numpy())
        np.testing.assert_allclose(scenarios["pnl_dollar"], independent_pnl, rtol=0, atol=1e-7)

        risk = compute_var_es(scenarios)
        losses = sorted(-independent_pnl)
        for confidence, rounded in [(95, (52726, 71697)), (99, (83932, 101288))]:
            with self.subTest(confidence=confidence):
                rank = (len(losses) * confidence + 99) // 100
                whole, hundredths = divmod(len(losses) * (100 - confidence), 100)
                fraction = hundredths / 100
                es = (sum(losses[-whole:]) + fraction * losses[-whole - 1]) / (whole + fraction)
                var = losses[rank - 1]
                self.assertAlmostEqual(var, risk[f"var_{confidence}_dollar"], places=7)
                self.assertAlmostEqual(es, risk[f"es_{confidence}_dollar"], places=7)
                self.assertEqual((round(var), round(es)), rounded)


if __name__ == "__main__":
    unittest.main()
