import os
import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.generate_dynamic_plot import OUTPUT_DIR, generate_financial_plot
from src.agents.custom_math_plot import generate_custom_math_plot
from src.agents.derived_plot_tools import generate_missing_data_heatmap, generate_ohlc_correlation_heatmap, run_data_analysis_plot


@patch('src.memory.mongodb_memory_layer.MongoMemoryManager')
class PlotRenderingTests(unittest.TestCase):
    def setUp(self):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS
        GLOBAL_PLOT_IDS.clear()
        GLOBAL_PLOT_DATA.clear()

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

    def test_generate_financial_plot_registers_memory_fallback_when_storage_fails(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS

        mock_mongo = mock_mongo_class.return_value
        mock_mongo.store_plot.return_value = False

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
            title="Storage Failure Plot",
        )

        self.assertEqual(result, "Chart ready: Storage Failure Plot")
        self.assertIn("default", GLOBAL_PLOT_IDS)
        self.assertEqual(len(GLOBAL_PLOT_IDS["default"]), 1)
        plot_id = GLOBAL_PLOT_IDS["default"][0]
        self.assertEqual(GLOBAL_PLOT_DATA[plot_id]["plot_type"], "line")
        self.assertEqual(GLOBAL_PLOT_DATA[plot_id]["title"], "Storage Failure Plot")

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

    def test_generate_financial_plot_preserves_row_based_grouped_bar_payload(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS

        mock_mongo = mock_mongo_class.return_value
        mock_mongo.store_plot.return_value = False

        result = generate_financial_plot.func(
            plot_type="bar",
            data={
                "plot_id": "return_volatility_grouped_bar",
                "chart_type": "bar",
                "bar_mode": "grouped",
                "x_axis": "ticker",
                "y_axis": "return_percent",
                "unit": "percent",
                "sort": "descending",
                "skipAnimation": True,
                "animation": {"duration": "600ms", "easing": "ease-out"},
                "series": [
                    {"key": "return_percent", "label": "Return"},
                    {"key": "volatility_percent", "label": "Volatility"},
                ],
                "data": [
                    {"ticker": "AAPL", "return_percent": 12.1, "volatility_percent": 28.8},
                    {"ticker": "MSFT", "return_percent": 10.4, "volatility_percent": 26.9},
                ],
            },
            title="Return vs Volatility",
        )

        self.assertEqual(result, "Chart ready: Return vs Volatility")
        spec = GLOBAL_PLOT_DATA[GLOBAL_PLOT_IDS["default"][0]]
        self.assertEqual(spec["plot_type"], "bar")
        self.assertEqual(spec["bar_mode"], "grouped")
        self.assertEqual(spec["x_axis"], "ticker")
        self.assertEqual(spec["series"][1]["key"], "volatility_percent")
        self.assertTrue(spec["skipAnimation"])
        self.assertEqual(spec["animation"]["duration"], "600ms")

    def test_dense_row_based_bar_spec_uses_svg_batch_defaults(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS

        mock_mongo = mock_mongo_class.return_value
        mock_mongo.store_plot.return_value = False

        result = generate_financial_plot.func(
            plot_type="bar",
            data={
                "plot_id": "dense_grouped_bar",
                "chart_type": "bar",
                "bar_mode": "grouped",
                "x_axis": "ticker",
                "y_axis": "return_percent",
                "unit": "percent",
                "series": [
                    {"key": "return_percent", "label": "Return"},
                    {"key": "volatility_percent", "label": "Volatility"},
                ],
                "data": [
                    {
                        "ticker": f"T{index + 1}",
                        "return_percent": index * 0.1,
                        "volatility_percent": 20 + index * 0.05,
                    }
                    for index in range(260)
                ],
            },
            title="Dense Return vs Volatility",
        )

        self.assertEqual(result, "Chart ready: Dense Return vs Volatility")
        spec = GLOBAL_PLOT_DATA[GLOBAL_PLOT_IDS["default"][0]]
        self.assertEqual(spec["renderer"], "svg-batch")
        self.assertTrue(spec["skipAnimation"])

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

    def test_generate_custom_math_plot_registers_formula_series(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS

        result = generate_custom_math_plot.func(
            formulas=[
                {"name": "Linear", "formula": "x"},
                {"name": "Square", "formula": "x**2"},
            ],
            title="Custom Formula Test",
            x_start=0,
            x_end=2,
            points=3,
            y_label="Formula Value",
        )

        self.assertEqual(result, "Custom math plot ready: Custom Formula Test")
        plot_id = GLOBAL_PLOT_IDS["default"][0]
        spec = GLOBAL_PLOT_DATA[plot_id]
        self.assertEqual(spec["plot_type"], "line")
        self.assertEqual(spec["title"], "Custom Formula Test")
        self.assertEqual(spec["series"][0]["data"], [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 2.0, "y": 2.0}])
        self.assertEqual(spec["series"][1]["data"][-1], {"x": 2.0, "y": 4.0})

    def test_generate_custom_math_plot_blocks_unsafe_expression(self, mock_mongo_class):
        result = generate_custom_math_plot.func(
            formulas=[{"name": "Unsafe", "formula": "__import__('os').system('echo bad')"}],
            title="Unsafe Formula",
            x_start=0,
            x_end=1,
            points=2,
        )

        self.assertIn("Unable to generate custom math plot", result)

    def test_historical_price_plot_downsamples_dense_chat_payload(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS
        from src.agents.live_data_tools import plot_historical_prices

        start = datetime(2020, 1, 1)
        historical_prices = [
            {
                "Date": (start + timedelta(days=index)).strftime("%Y-%m-%d"),
                "Close": 100 + index * 0.1,
            }
            for index in range(1500)
        ]

        with patch("src.agents.live_data_tools._find_price_documents_with_retry") as mock_find:
            mock_find.return_value = [{"ticker": "AAPL", "historical_prices": historical_prices}]
            result = plot_historical_prices.func(
                tickers=["AAPL"],
                start_date="2020-01-01",
                end_date="2025-12-31",
            )

        self.assertIn("Plot generated successfully", result)
        plot_id = GLOBAL_PLOT_IDS["default"][0]
        spec = GLOBAL_PLOT_DATA[plot_id]
        self.assertTrue(spec["density"]["sampled"])
        self.assertLessEqual(len(spec["series"][0]["data"]), 701)
        self.assertEqual(spec["density"]["point_counts"]["AAPL"]["original"], 1500)
        self.assertTrue(spec["skipAnimation"])

    def test_generate_ohlc_correlation_heatmap_registers_heatmap(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS
        from src.agents.price_series_tool import _store_analysis_dataset

        cache_key = _store_analysis_dataset(
            {
                "prices": {
                    "AXP": [
                        {"date": "2020-01-01", "open": 10, "high": 12, "low": 9, "close": 11},
                        {"date": "2020-01-02", "open": 11, "high": 13, "low": 10, "close": 12},
                        {"date": "2020-01-03", "open": 12, "high": 14, "low": 11, "close": 13},
                        {"date": "2020-01-04", "open": 13, "high": 15, "low": 12, "close": 14},
                        {"date": "2020-01-05", "open": 14, "high": 16, "low": 13, "close": 15},
                    ]
                }
            }
        )

        result = generate_ohlc_correlation_heatmap.func(
            ticker="AXP",
            analysis_cache_key=cache_key,
        )

        self.assertEqual(result["status"], "success")
        plot_id = GLOBAL_PLOT_IDS["default"][0]
        spec = GLOBAL_PLOT_DATA[plot_id]
        self.assertEqual(spec["plot_type"], "heatmap")
        self.assertEqual(spec["metadata"]["ticker"], "AXP")
        self.assertEqual(spec["metadata"]["observations"], 5)
        self.assertEqual(spec["matrix"]["open"]["close"], 1.0)
        self.assertEqual(len(spec["series"][0]["data"]), 16)

    def test_generate_missing_data_heatmap_builds_matrix_from_tickers(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS

        docs = [
            {
                "ticker": "AAPL",
                "historical_prices": [
                    {"Date": "2020-01-01", "Close": 10},
                    {"Date": "2020-01-02", "Close": 11},
                    {"Date": "2020-01-03", "Close": 12},
                ],
            },
            {
                "ticker": "MSFT",
                "historical_prices": [
                    {"Date": "2020-01-01", "Close": 20},
                    {"Date": "2020-01-03", "Close": 22},
                ],
            },
        ]

        with patch("src.agents.derived_plot_tools._find_price_documents_with_retry") as mock_find:
            mock_find.return_value = docs
            result = generate_missing_data_heatmap.func(
                tickers=["AAPL", "MSFT"],
                start_date="2020-01-01",
                end_date="2020-01-03",
            )

        self.assertEqual(result["status"], "success")
        plot_id = GLOBAL_PLOT_IDS["default"][0]
        spec = GLOBAL_PLOT_DATA[plot_id]
        self.assertEqual(spec["plot_type"], "heatmap")
        self.assertEqual(spec["metadata"]["heatmap_type"], "missing")
        self.assertEqual(spec["matrix"]["2020-01-02"]["AAPL"], 1)
        self.assertEqual(spec["matrix"]["2020-01-02"]["MSFT"], 0)
        self.assertEqual(spec["metadata"]["missing_counts"]["MSFT"], 1)

    def test_common_analysis_plot_routes_missing_data_heatmap(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS

        docs = [
            {
                "ticker": "AAPL",
                "historical_prices": [
                    {"Date": "2020-01-01", "Close": 10},
                    {"Date": "2020-01-02", "Close": 11},
                ],
            }
        ]

        with patch("src.agents.derived_plot_tools._find_price_documents_with_retry") as mock_find:
            mock_find.return_value = docs
            result = run_data_analysis_plot.func(
                analysis_task="missing_data_heatmap",
                tickers=["AAPL"],
                start_date="2020-01-01",
                end_date="2020-01-02",
            )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["analysis_task"], "missing_data_heatmap")
        spec = GLOBAL_PLOT_DATA[GLOBAL_PLOT_IDS["default"][0]]
        self.assertEqual(spec["plot_type"], "heatmap")

    def test_common_analysis_plot_routes_returns_correlation_heatmap(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS
        from src.agents.price_series_tool import _store_analysis_dataset

        cache_key = _store_analysis_dataset(
            {
                "returns": {"AAPL": [0.01, 0.02, 0.03], "MSFT": [0.03, 0.02, 0.01]},
                "return_dates_by_ticker": {
                    "AAPL": ["2020-01-02", "2020-01-03", "2020-01-04"],
                    "MSFT": ["2020-01-02", "2020-01-03", "2020-01-04"],
                },
            }
        )

        result = run_data_analysis_plot.func(
            analysis_task="returns_correlation_heatmap",
            tickers=["AAPL", "MSFT"],
            analysis_cache_key=cache_key,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["analysis_task"], "returns_correlation_heatmap")
        spec = GLOBAL_PLOT_DATA[GLOBAL_PLOT_IDS["default"][0]]
        self.assertEqual(spec["plot_type"], "heatmap")
        self.assertEqual(spec["metadata"]["tickers"], ["AAPL", "MSFT"])

    def test_common_analysis_plot_routes_returns_box_plot(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS
        from src.agents.price_series_tool import _store_analysis_dataset

        cache_key = _store_analysis_dataset(
            {
                "returns": {
                    "AAPL": [-0.012, -0.006, 0.001, 0.008, 0.014, 0.021],
                    "MSFT": [-0.009, -0.004, 0.002, 0.006, 0.011, 0.018],
                },
                "return_dates_by_ticker": {
                    "AAPL": ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"],
                    "MSFT": ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06", "2020-01-07"],
                },
            }
        )

        result = run_data_analysis_plot.func(
            analysis_task="returns_box_plot",
            tickers=["AAPL", "MSFT"],
            start_date="2020-01-01",
            end_date="2020-01-31",
            analysis_cache_key=cache_key,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["analysis_task"], "returns_box_plot")
        spec = GLOBAL_PLOT_DATA[GLOBAL_PLOT_IDS["default"][0]]
        self.assertEqual(spec["plot_type"], "box")
        self.assertEqual(spec["y_label"], "Daily return (%)")
        self.assertEqual([item["label"] for item in spec["data"]], ["AAPL", "MSFT"])
        for item in spec["data"]:
            self.assertIn("q1", item)
            self.assertIn("median", item)
            self.assertIn("q3", item)
            self.assertGreaterEqual(item["sample_size"], 5)

    def test_common_analysis_plot_routes_default_price_spread_area(self, mock_mongo_class):
        from src.agents.plot_store import GLOBAL_PLOT_DATA, GLOBAL_PLOT_IDS
        from src.agents.price_series_tool import _store_analysis_dataset

        cache_key = _store_analysis_dataset(
            {
                "prices": {
                    "AAPL": [
                        {"date": "2020-01-02", "close": 100},
                        {"date": "2020-01-03", "close": 104},
                        {"date": "2020-01-06", "close": 108},
                    ],
                    "MSFT": [
                        {"date": "2020-01-02", "close": 90},
                        {"date": "2020-01-03", "close": 96},
                        {"date": "2020-01-06", "close": 99},
                    ],
                }
            }
        )

        result = run_data_analysis_plot.func(
            analysis_task="price_spread_area",
            tickers=[],
            start_date="2020-01-01",
            end_date="2025-01-01",
            analysis_cache_key=cache_key,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["analysis_task"], "price_spread_area")
        self.assertEqual(result["tickers"], ["AAPL", "MSFT"])
        spec = GLOBAL_PLOT_DATA[GLOBAL_PLOT_IDS["default"][0]]
        self.assertEqual(spec["plot_type"], "line")
        self.assertEqual(spec["chart_type"], "line_area")
        self.assertTrue(spec["series"][0]["area"])
        self.assertEqual(spec["series"][0]["baseline"], 0)
        self.assertEqual(spec["metadata"]["formula"], "AAPL.close - MSFT.close")
        self.assertEqual(spec["series"][0]["data"][0], {"x": "2020-01-02", "y": 10.0})


if __name__ == "__main__":
    unittest.main()
