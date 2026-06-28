import json
import os
import sys
import unittest
from unittest.mock import patch

from pymongo.errors import NetworkTimeout


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.live_data_tools import (
    _get_client,
    get_historical_prices,
    get_market_data_bundle,
    get_stock_database_snapshot,
    get_yfinance_market_data,
    plot_historical_prices,
    run_full_governance_pipeline,
)


class FakeMarketMemory:
    def __init__(self, cached_payload=None):
        self.cached_payload = cached_payload
        self.store_calls = []

    def compute_market_data_cache_key(self, **kwargs):
        self.last_cache_key_kwargs = kwargs
        return "cache-key"

    def retrieve_market_data_cache(self, cache_key):
        self.last_retrieve_key = cache_key
        if self.cached_payload is None:
            return None
        return {"payload": self.cached_payload, "source": "yfinance", "fetched_at": "now", "expires_at": "later"}

    def store_market_data_cache(self, **kwargs):
        self.store_calls.append(kwargs)
        return True


class FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.info = {
            "longName": f"{symbol} Corporation",
            "sector": "Technology",
            "industry": "Software",
            "marketCap": 2500000000,
            "currentPrice": 123.45,
        }

    def history(self, **_kwargs):
        import pandas as pd

        return pd.DataFrame(
            [{"Open": 99.0, "High": 101.0, "Low": 98.0, "Close": 100.5, "Volume": 1000}],
            index=pd.to_datetime(["2025-01-02"]),
        )

    @property
    def financials(self):
        import pandas as pd

        return pd.DataFrame(
            {"2025-03-31": [1200000000, 220000000]},
            index=["Total Revenue", "Net Income"],
        )


class FakeYFinance:
    def __init__(self):
        self.ticker_calls = []

    def Ticker(self, symbol):
        self.ticker_calls.append(symbol)
        return FakeTicker(symbol)


class LiveDataToolsYFinanceFallbackTests(unittest.TestCase):
    def tearDown(self):
        _get_client.cache_clear()

    def test_mongo_client_resolves_uri_lazily_from_environment(self):
        _get_client.cache_clear()

        with patch.dict(os.environ, {"MONGO_URI": "mongodb://example.local:27017"}, clear=False):
            with patch("src.agents.live_data_tools.MongoClient") as mongo_client:
                _get_client()

        mongo_client.assert_called_once()
        self.assertEqual(mongo_client.call_args.args[0], "mongodb://example.local:27017")

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

    def test_yfinance_market_data_uses_cache_before_fetching(self):
        cached_payload = {
            "symbol": "AAPL",
            "data_type": "history",
            "source": "yfinance",
            "history": [{"Date": "2025-01-02", "Close": 100.5}],
        }
        fake_memory = FakeMarketMemory(cached_payload=cached_payload)
        fake_yfinance = FakeYFinance()

        with patch("src.agents.live_data_tools.memory_manager", fake_memory):
            with patch("src.agents.live_data_tools._get_yfinance_module", return_value=fake_yfinance):
                result = get_yfinance_market_data.func(symbol="AAPL", data_type="history")

        self.assertIn("Market data for AAPL", result)
        self.assertIn("yfinance cache", result)
        self.assertEqual(fake_yfinance.ticker_calls, [])
        self.assertEqual(fake_memory.store_calls, [])

    def test_yfinance_market_data_fetches_and_stores_cache_miss(self):
        fake_memory = FakeMarketMemory(cached_payload=None)
        fake_yfinance = FakeYFinance()

        with patch("src.agents.live_data_tools.memory_manager", fake_memory):
            with patch("src.agents.live_data_tools._get_yfinance_module", return_value=fake_yfinance):
                result = get_yfinance_market_data.func(symbol="AAPL", data_type="history", period="5d")

        self.assertIn("Market data for AAPL", result)
        self.assertIn("history: 1 rows", result)
        self.assertIn("Cache: stored", result)
        self.assertEqual(fake_yfinance.ticker_calls, ["AAPL"])
        self.assertEqual(fake_memory.store_calls[0]["symbol"], "AAPL")
        self.assertEqual(fake_memory.store_calls[0]["data_type"], "history")

    def test_market_data_bundle_infers_required_payloads_and_returns_comparison_values(self):
        fake_memory = FakeMarketMemory(cached_payload=None)
        fake_yfinance = FakeYFinance()

        with patch("src.agents.live_data_tools.memory_manager", fake_memory):
            with patch("src.agents.live_data_tools._get_yfinance_module", return_value=fake_yfinance):
                result = get_market_data_bundle.func(
                    symbols=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
                    request="compare their revenue and net income values",
                )

        self.assertIn("Market Data Bundle", result)
        self.assertIn("Data fetched: financials, info", result)
        self.assertIn("Currency basis: INFY.NS=INR, RELIANCE.NS=INR, TCS.NS=INR", result)
        self.assertIn("| Total revenue | ₹120.00 crore (2025-03-31)", result)
        self.assertIn("| Net income | ₹22.00 crore (2025-03-31)", result)
        self.assertEqual(
            sorted(call["data_type"] for call in fake_memory.store_calls),
            ["financials", "financials", "financials", "info", "info", "info"],
        )

    def test_market_data_bundle_warns_for_mixed_currency_values(self):
        def fake_payload(symbol, data_type, **_kwargs):
            currency = "INR" if symbol.endswith(".NS") else "USD"
            if data_type == "info":
                return ({"symbol": symbol, "data_type": "info", "info": {"currency": currency, "marketCap": 2500000000000}}, False)
            return (None, False)

        with patch("src.agents.live_data_tools._fetch_cached_yfinance_market_payload", side_effect=fake_payload):
            result = get_market_data_bundle.func(
                symbols=["AAPL", "RELIANCE.NS"],
                request="compare market cap values",
            )

        self.assertIn("Currency basis: AAPL=USD, RELIANCE.NS=INR", result)
        self.assertIn("Warning: monetary values use mixed currencies", result)
        self.assertIn("$2.50T", result)
        self.assertIn("₹2.50 lakh crore", result)

    def test_market_data_bundle_keeps_non_us_non_inr_values_in_native_currency(self):
        def fake_payload(symbol, data_type, **_kwargs):
            if data_type == "info":
                return ({"symbol": symbol, "data_type": "info", "info": {"currency": "JPY", "marketCap": 2500000000000}}, False)
            return (None, False)

        with patch("src.agents.live_data_tools._fetch_cached_yfinance_market_payload", side_effect=fake_payload):
            result = get_market_data_bundle.func(symbols=["7203.T"], request="compare market cap values")

        self.assertIn("Currency basis: 7203.T=JPY", result)
        self.assertIn("¥2,500,000,000,000.00 JPY", result)
        self.assertNotIn("2.50T", result)

    def test_market_data_bundle_fetches_broad_payloads_for_all_available_data(self):
        fake_memory = FakeMarketMemory(cached_payload=None)

        with patch("src.agents.live_data_tools.memory_manager", fake_memory):
            with patch("src.agents.live_data_tools._fetch_cached_yfinance_market_payload", return_value=({"symbol": "AAPL"}, False)):
                result = get_market_data_bundle.func(symbols=["AAPL"], request="fetch all available stock data")

        self.assertIn("history", result)
        self.assertIn("financials", result)
        self.assertIn("options", result)
        self.assertIn("cached for the next analysis step", result)

    def test_plot_historical_prices_falls_back_to_yfinance_for_exchange_suffix_tickers(self):
        import pandas as pd

        fallback_frame = pd.DataFrame(
            [
                {"Date": pd.Timestamp("2025-01-01"), "Close": 2500.0},
                {"Date": pd.Timestamp("2025-01-02"), "Close": 2520.0},
            ]
        )

        class FakePlotMemory:
            def store_plot(self, *_args, **_kwargs):
                return True

        with patch("src.agents.live_data_tools._find_price_documents_with_retry", return_value=[]):
            with patch("src.agents.live_data_tools._cached_yfinance_history_frame", return_value=fallback_frame):
                with patch("src.agents.live_data_tools.MongoMemoryManager", return_value=FakePlotMemory()):
                    result = plot_historical_prices.func(
                        tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
                        start_date="2025-01-01",
                        end_date="2025-01-31",
                    )

        self.assertIn("Historical Price Plot", result)
        self.assertIn("Included tickers: INFY.NS, RELIANCE.NS, TCS.NS", result)
        self.assertIn("yfinance used and cached for: INFY.NS, RELIANCE.NS, TCS.NS", result)

    def test_full_governance_pipeline_uses_yfinance_when_mongo_has_no_nse_tickers(self):
        import pandas as pd

        def fallback_frame(ticker, *_args, **_kwargs):
            offset = {"INFY.NS": 0, "RELIANCE.NS": 20, "TCS.NS": 40}.get(ticker, 0)
            dates = pd.bdate_range("2025-07-01", periods=130)
            return pd.DataFrame(
                {
                    "Date": dates,
                    "Close": [100.0 + offset + index * (1.0 + offset / 200.0) for index in range(len(dates))],
                }
            )

        optimization_payload = {
            "status": "success",
            "weights": {"INFY.NS": 0.3, "RELIANCE.NS": 0.4, "TCS.NS": 0.3},
            "instability_index": 0.2,
            "lambda_t": 0.1,
            "risk_tolerance": "moderate",
        }

        class FakeMemory:
            def store_regime_pattern(self, **_kwargs):
                return None

        with patch("src.agents.live_data_tools._find_documents_with_retry", return_value=[]):
            with patch("src.agents.live_data_tools._cached_yfinance_history_frame", side_effect=fallback_frame):
                with patch("src.agents.live_data_tools._build_optimization_payload", return_value=optimization_payload):
                    with patch("src.agents.live_data_tools._generate_inline_governance_plots", return_value=[]):
                        with patch("src.agents.live_data_tools.memory_manager", FakeMemory()):
                            result = run_full_governance_pipeline.func(
                                tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
                                target_date="2025-12-30",
                                risk_tolerance="moderate",
                            )

        payload = json.loads(result)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["valid_tickers"], ["INFY.NS", "RELIANCE.NS", "TCS.NS"])
        self.assertEqual(payload["data_sources"]["INFY.NS"], "yfinance")
        self.assertIn("yfinance was used and cached", payload["message"])


if __name__ == "__main__":
    unittest.main()
