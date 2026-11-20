"""Basic sanity checks for a portfolio demonstration—not a full model audit."""
import unittest

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


if __name__ == "__main__":
    unittest.main()
