"""Exercise Streamlit views and configuration changes with AppTest."""

import unittest
from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class DashboardTests(unittest.TestCase):
    def test_all_views_render(self):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("arrow_vega_lite_chart")), 3)

        for view in ("Portfolio", "Stress test", "Historical risk", "PCA", "Monte Carlo"):
            with self.subTest(view=view):
                app.radio[0].set_value(view).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.warning)

    def test_as_of_date_and_position_changes_update_risk(self):
        app = AppTest.from_file(str(APP), default_timeout=20).run()
        app.radio[0].set_value("Historical risk").run()
        baseline = app.dataframe[0].value["VaR (USD)"].iloc[0]

        app.sidebar.number_input[2].set_value(0).run()
        self.assertFalse(app.exception)
        self.assertLess(app.dataframe[0].value["VaR (USD)"].iloc[0], baseline)

        app.sidebar.date_input[0].set_value(date(2024, 12, 29)).run()
        self.assertFalse(app.exception)
        self.assertIn("2024-12-27", app.info[0].value)
        self.assertTrue(any("aligned daily shocks" in item.value and "2024-12-27" in item.value
                            for item in app.caption))
        self.assertNotEqual(app.dataframe[0].value["VaR (USD)"].iloc[0], baseline)


if __name__ == "__main__":
    unittest.main()
