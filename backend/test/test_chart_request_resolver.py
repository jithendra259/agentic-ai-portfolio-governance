import os
import sys
import unittest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.decision.chart_request_resolver import build_chart_response, resolve_deterministic_chart_request


class ChartRequestResolverTests(unittest.TestCase):
    def test_area_plot_any_metric_uses_default_spread_area(self):
        with patch("src.decision.chart_request_resolver.run_data_analysis_plot") as mock_tool:
            mock_tool.func.return_value = {
                "status": "success",
                "plot_id": "plot-123",
                "analysis_task": "price_spread_area",
                "tickers": ["AAPL", "MSFT"],
            }

            resolved = resolve_deterministic_chart_request(
                "i need area plot take any metric",
                "session-1",
            )

        self.assertIsNotNone(resolved)
        self.assertEqual(resolved["analysis_task"], "price_spread_area")
        self.assertEqual(resolved["plot_id"], "plot-123")
        call_kwargs = mock_tool.func.call_args.kwargs
        self.assertEqual(call_kwargs["analysis_task"], "price_spread_area")
        self.assertEqual(call_kwargs["tickers"], ["AAPL", "MSFT"])
        self.assertEqual(call_kwargs["start_date"], "2020-01-01")
        self.assertEqual(call_kwargs["end_date"], "2025-01-01")

    def test_box_plot_request_routes_to_returns_box_plot(self):
        with patch("src.decision.chart_request_resolver.run_data_analysis_plot") as mock_tool:
            mock_tool.func.return_value = {
                "status": "success",
                "plot_id": "plot-box",
                "analysis_task": "returns_box_plot",
                "tickers": ["AAPL", "MSFT", "NVDA"],
            }

            resolved = resolve_deterministic_chart_request(
                "box plot of daily returns for AAPL, MSFT, NVDA from 2015 to 2025",
                "session-1",
            )

        self.assertIsNotNone(resolved)
        call_kwargs = mock_tool.func.call_args.kwargs
        self.assertEqual(call_kwargs["analysis_task"], "returns_box_plot")
        self.assertEqual(call_kwargs["tickers"], ["AAPL", "MSFT", "NVDA"])
        self.assertEqual(call_kwargs["start_date"], "2015-01-01")
        self.assertEqual(call_kwargs["end_date"], "2025-12-31")

    def test_close_price_plot_routes_to_price_line_without_line_keyword(self):
        with patch("src.decision.chart_request_resolver.run_data_analysis_plot") as mock_tool:
            mock_tool.func.return_value = {
                "status": "success",
                "plot_id": "plot-line",
                "analysis_task": "price_line",
                "tickers": ["AAPL"],
            }

            resolved = resolve_deterministic_chart_request(
                "plot AAPL close price from 2020 to 2025",
                "session-1",
            )

        self.assertIsNotNone(resolved)
        call_kwargs = mock_tool.func.call_args.kwargs
        self.assertEqual(call_kwargs["analysis_task"], "price_line")
        self.assertEqual(call_kwargs["tickers"], ["AAPL"])
        self.assertEqual(call_kwargs["start_date"], "2020-01-01")
        self.assertEqual(call_kwargs["end_date"], "2025-12-31")

    def test_close_price_plot_for_sector_passes_sector_without_default_tickers(self):
        with patch("src.decision.chart_request_resolver.run_data_analysis_plot") as mock_tool:
            mock_tool.func.return_value = {
                "status": "success",
                "plot_id": "plot-real-estate",
                "analysis_task": "price_line",
                "tickers": ["ARE", "AMT", "CCI"],
            }

            resolved = resolve_deterministic_chart_request(
                "plot the data of closing price of Real Estate",
                "session-1",
            )

        self.assertIsNotNone(resolved)
        call_kwargs = mock_tool.func.call_args.kwargs
        self.assertEqual(call_kwargs["analysis_task"], "price_line")
        self.assertEqual(call_kwargs["tickers"], [])
        self.assertIsNone(call_kwargs["ticker"])
        self.assertEqual(call_kwargs["sector"], "Real Estate")
        self.assertEqual(resolved["sector"], "Real Estate")

    def test_build_chart_response_prefers_sector_scope(self):
        response = build_chart_response(
            {
                "analysis_task": "price_line",
                "tickers": ["ARE", "AMT", "CCI"],
                "sector": "Real Estate",
                "start_date": "2020-01-01",
                "end_date": "2025-01-01",
            }
        )

        self.assertIn("close-price line chart for Real Estate", response)
        self.assertNotIn("ARE, AMT, CCI", response)

    def test_close_price_plot_preserves_exchange_suffix_tickers(self):
        with patch("src.decision.chart_request_resolver.run_data_analysis_plot") as mock_tool:
            mock_tool.func.return_value = {
                "status": "success",
                "plot_id": "plot-nse",
                "analysis_task": "price_line",
                "tickers": ["INFY.NS", "RELIANCE.NS", "TCS.NS"],
            }

            resolved = resolve_deterministic_chart_request(
                "plot the closing price RELIANCE.NS,TCS.NS, INFY.NS",
                "session-1",
            )

        self.assertIsNotNone(resolved)
        call_kwargs = mock_tool.func.call_args.kwargs
        self.assertEqual(call_kwargs["tickers"], ["RELIANCE.NS", "TCS.NS", "INFY.NS"])
        self.assertEqual(call_kwargs["ticker"], "RELIANCE.NS")

    def test_build_chart_response_names_chart_type(self):
        response = build_chart_response(
            {
                "analysis_task": "price_spread_area",
                "tickers": ["AAPL", "MSFT"],
                "start_date": "2020-01-01",
                "end_date": "2025-01-01",
            }
        )

        self.assertIn("close-price spread area chart", response)
        self.assertIn("AAPL, MSFT", response)


if __name__ == "__main__":
    unittest.main()
