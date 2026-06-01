import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.generate_dynamic_plot import OUTPUT_DIR, generate_financial_plot


@patch('src.memory.mongodb_memory_layer.MongoMemoryManager')
class PlotRenderingTests(unittest.TestCase):
    def setUp(self):
        from src.agents.plot_store import GLOBAL_PLOT_IDS
        GLOBAL_PLOT_IDS.clear()

    def test_generate_financial_plot_registers_interactive_plotspec(self, mock_mongo_class):
        mock_mongo = mock_mongo_class.return_value
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
        mock_mongo.store_plot.assert_called_once()
        plot_id, spec = mock_mongo.store_plot.call_args[0][:2]
        
        self.assertEqual(spec["plot_type"], "line")
        self.assertEqual(spec["title"], "Test Plot")
        self.assertEqual(spec["series"][0]["name"], "AAPL")

    @patch('src.agents.price_series_tool.load_cached_analysis_dataset')
    def test_generate_financial_plot_from_cache_returns(self, mock_load_cached, mock_mongo_class):
        mock_mongo = mock_mongo_class.return_value
        
        # Setup mock cached dataset containing returns
        mock_load_cached.return_value = {
            "tickers_included": ["AAPL", "MSFT"],
            "returns": {
                "AAPL": [-0.01, 0.02],
                "MSFT": [0.015, -0.005]
            },
            "return_dates_by_ticker": {
                "AAPL": ["2020-01-02", "2020-01-03"],
                "MSFT": ["2020-01-02", "2020-01-03"]
            }
        }
        
        result = generate_financial_plot.func(
            data={
                "analysis_cache_key": "test_cache_key",
                "metric": "returns",
                "y_label": "Log Return"
            },
            plot_type="line",
            title="Log Returns Plot",
        )
        
        self.assertEqual(result, "Chart ready: Log Returns Plot")
        mock_mongo.store_plot.assert_called_once()
        plot_id, spec = mock_mongo.store_plot.call_args[0][:2]
        
        self.assertEqual(spec["plot_type"], "line")
        self.assertEqual(spec["title"], "Log Returns Plot")
        self.assertEqual(spec["y_label"], "Log Return")
        self.assertEqual(len(spec["series"]), 2)
        
        # Verify the mapping of return values and dates to x/y series format
        aapl_series = next(s for s in spec["series"] if s["name"] == "AAPL")
        self.assertEqual(aapl_series["data"], [
            {"x": "2020-01-02", "y": -0.01},
            {"x": "2020-01-03", "y": 0.02}
        ])

    def test_plot_us_economic_indicators_registers_plotspec(self, mock_mongo_class):
        mock_mongo = mock_mongo_class.return_value
        from src.agents.live_data_tools import plot_us_economic_indicators
        result = plot_us_economic_indicators.func()
        
        self.assertIn("Chart ready", result)
        mock_mongo.store_plot.assert_called_once()
        plot_id, spec = mock_mongo.store_plot.call_args[0][:2]
        
        self.assertEqual(spec["plot_type"], "line")
        self.assertEqual(spec["title"], "US unemployment rate comparison with GDP per capita")
        self.assertIn("recessions", spec)
        self.assertIn("yAxis", spec)


if __name__ == "__main__":
    unittest.main()
