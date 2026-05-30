import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.generate_dynamic_plot import OUTPUT_DIR, generate_financial_plot
from src.agents.plot_store import GLOBAL_PLOT_DATA


class PlotRenderingTests(unittest.TestCase):
    def setUp(self):
        GLOBAL_PLOT_DATA.clear()

    def test_generate_financial_plot_registers_interactive_plotspec(self):
        result = generate_financial_plot.func(
            data={
                "price_history": {
                    "AAPL": [
                        {"date": "2020-01-01", "close": 100},
                        {"date": "2020-01-02", "close": 101},
                    ]
                }
            },
            plot_type="line",
            title="Test Plot",
        )

        self.assertEqual(result, "Chart ready: Test Plot")
        self.assertIn("default", GLOBAL_PLOT_DATA)
        specs = GLOBAL_PLOT_DATA["default"]
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["plot_type"], "line")
        self.assertEqual(specs[0]["title"], "Test Plot")
        self.assertEqual(specs[0]["series"][0]["name"], "AAPL")


    def test_plot_us_economic_indicators_registers_plotspec(self):
        from src.agents.live_data_tools import plot_us_economic_indicators
        result = plot_us_economic_indicators.func()
        self.assertIn("Chart ready", result)
        self.assertIn("default", GLOBAL_PLOT_DATA)
        specs = GLOBAL_PLOT_DATA["default"]
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["plot_type"], "line")
        self.assertEqual(specs[0]["title"], "US unemployment rate comparison with GDP per capita")
        self.assertIn("recessions", specs[0])
        self.assertIn("yAxis", specs[0])


if __name__ == "__main__":
    unittest.main()

