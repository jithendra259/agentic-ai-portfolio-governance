import os
import sys
import unittest
from unittest.mock import patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.price_series_tool import get_price_series_for_analysis


class PriceSeriesToolTests(unittest.TestCase):
    def test_returns_structured_prices_returns_and_stats(self):
        docs = [
            {
                "ticker": "AAPL",
                "historical_prices": [
                    {"Date": "2020-01-01", "Close": 100.0},
                    {"Date": "2020-01-02", "Close": 101.0},
                    {"Date": "2020-01-03", "Close": 102.0},
                    {"Date": "2020-01-06", "Close": 103.0},
                    {"Date": "2020-01-07", "Close": 104.0},
                    {"Date": "2020-01-08", "Close": 105.0},
                ],
            },
            {
                "ticker": "MSFT",
                "historical_prices": [
                    {"Date": "2020-01-01", "Close": 50.0},
                    {"Date": "2020-01-02", "Close": 50.5},
                    {"Date": "2020-01-03", "Close": 51.0},
                    {"Date": "2020-01-06", "Close": 52.0},
                    {"Date": "2020-01-07", "Close": 53.0},
                    {"Date": "2020-01-08", "Close": 54.0},
                ],
            },
        ]

        with patch("src.agents.price_series_tool._find_documents_with_retry", return_value=docs):
            result = get_price_series_for_analysis.func(
                tickers=["AAPL", "MSFT"],
                start_date="2020-01-01",
                end_date="2020-01-08",
            )

        self.assertEqual(result["tickers_included"], ["AAPL", "MSFT"])
        self.assertEqual(result["tickers_missing"], [])
        self.assertEqual(len(result["prices"]["AAPL"]), 6)
        self.assertEqual(len(result["returns"]["AAPL"]), 5)
        self.assertIn("annualised_vol", result["stats"]["AAPL"])
        self.assertEqual(result["stats"]["AAPL"]["observations"], 6)

    def test_returns_error_when_no_series_found(self):
        with patch("src.agents.price_series_tool._find_documents_with_retry", return_value=[]):
            result = get_price_series_for_analysis.func(
                tickers=["AAPL"],
                start_date="2020-01-01",
                end_date="2020-01-08",
            )

        self.assertIn("error", result)
        self.assertIn("tickers_missing", result)


if __name__ == "__main__":
    unittest.main()
