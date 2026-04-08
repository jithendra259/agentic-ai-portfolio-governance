import os
import sys
import unittest

import pandas as pd


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.time_series_a1 import TimeSeriesAgent


class FakeCollection:
    def __init__(self, docs):
        self.docs = docs

    def find(self, query, projection):
        universe_id = query.get("universes")
        for doc in self.docs:
            universes = doc.get("universes", [])
            if universe_id in universes:
                yield {
                    "ticker": doc.get("ticker"),
                    "historical_prices": doc.get("historical_prices", []),
                }


def _build_history(dates, closes):
    return [
        {"Date": date.strftime("%Y-%m-%d"), "Close": close}
        for date, close in zip(dates, closes)
        if close is not None
    ]


class TimeSeriesAgentTests(unittest.TestCase):
    def test_quality_control_drops_sparse_assets_and_retains_recoverable_assets(self):
        dates = pd.date_range("2020-01-01", periods=40, freq="D")
        docs = [
            {
                "ticker": "AAA",
                "universes": ["U1"],
                "historical_prices": _build_history(dates, [100.0 + i for i in range(40)]),
            },
            {
                "ticker": "BBB",
                "universes": ["U1"],
                "historical_prices": _build_history(
                    dates,
                    [50.0 + i if i != 10 else None for i in range(40)],
                ),
            },
            {
                "ticker": "CCC",
                "universes": ["U1"],
                "historical_prices": _build_history(
                    dates,
                    [20.0 + i if i % 4 else None for i in range(40)],
                ),
            },
        ]

        agent = TimeSeriesAgent(FakeCollection(docs))
        result = agent.execute("U1", "2020-02-09", lookback_days=20)

        self.assertEqual(result["retained_assets"], ["AAA", "BBB"])
        self.assertEqual(result["dropped_assets"], ["CCC"])
        self.assertEqual(list(result["returns_df"].columns), ["AAA", "BBB"])
        self.assertGreaterEqual(len(result["returns_df"]), 20)
        self.assertGreaterEqual(result["instability_index"], 0.0)
        self.assertLessEqual(result["instability_index"], 1.0)


if __name__ == "__main__":
    unittest.main()
