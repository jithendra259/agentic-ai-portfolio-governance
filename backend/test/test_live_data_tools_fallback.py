import os
import sys
import unittest
from unittest.mock import patch

from pymongo.errors import NetworkTimeout


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.live_data_tools import get_historical_prices, get_stock_database_snapshot


class LiveDataToolsYFinanceFallbackTests(unittest.TestCase):
    def test_stock_snapshot_falls_back_to_yfinance_when_mongo_lookup_fails(self):
        fallback_doc = {
            "ticker": "MSFT",
            "historical_prices": [
                {"Date": "2025-01-01", "Close": 410.0},
                {"Date": "2025-12-31", "Close": 430.0},
            ],
            "info": {
                "company_name": "Microsoft Corporation",
                "shortName": "Microsoft",
                "longName": "Microsoft Corporation",
                "sector": "Technology",
                "industry": "Software",
                "country": "United States",
                "website": "https://www.microsoft.com",
                "summary": "Builds software and cloud services.",
            },
            "key_stats": {"market_cap": 1000000000, "trailing_pe": 30.0},
            "financials": {},
            "graph_relationships": {},
            "analysis_and_estimates": {},
            "_source": "yfinance_fallback",
        }

        with patch("src.agents.live_data_tools._find_documents_with_retry", side_effect=NetworkTimeout("timeout")):
            with patch("src.agents.live_data_tools._fetch_yfinance_snapshot_doc", return_value=fallback_doc):
                result = get_stock_database_snapshot.func(tickers=["MSFT"])

        self.assertIn("Stock Snapshot", result)
        self.assertIn("yfinance fallback", result)
        self.assertIn("Ticker: MSFT", result)
        self.assertIn("Microsoft Corporation", result)

    def test_historical_prices_fall_back_to_yfinance_when_mongo_lookup_fails(self):
        fallback_row = {
            "ticker": "MSFT",
            "close": 430.12,
            "date": "2025-12-31",
            "source": "yfinance fallback",
        }

        with patch("src.agents.live_data_tools._find_documents_with_retry", side_effect=NetworkTimeout("timeout")):
            with patch("src.agents.live_data_tools._fetch_yfinance_price_on_or_before", return_value=fallback_row):
                result = get_historical_prices.func(tickers=["MSFT"], target_date="2025-12-31")

        self.assertIn("Historical closing prices on or immediately before 2025-12-31:", result)
        self.assertIn("MSFT: close=430.12 on 2025-12-31 (source: yfinance fallback)", result)
        self.assertIn("yfinance fallback used for: MSFT", result)

    def test_historical_prices_keep_mongo_source_when_local_history_is_available(self):
        docs = [
            {
                "ticker": "AAPL",
                "historical_prices": [
                    {"Date": "2025-12-30", "Close": 200.0},
                    {"Date": "2025-12-31", "Close": 201.5},
                ],
            }
        ]

        with patch("src.agents.live_data_tools._find_documents_with_retry", return_value=docs):
            result = get_historical_prices.func(tickers=["AAPL"], target_date="2025-12-31")

        self.assertIn("AAPL: close=201.50 on 2025-12-31 (source: MongoDB)", result)
        self.assertNotIn("yfinance fallback used for", result)


if __name__ == "__main__":
    unittest.main()
