import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


NOTEBOOK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NOTEBOOK_DIR))


class IncrementalMarketDataCacheTests(unittest.TestCase):
    def test_save_falls_back_when_windows_atomic_replace_is_denied(self):
        from market_data_cache import _load_cache, _save_cache

        index = pd.bdate_range("2020-01-01", periods=3)
        frame = pd.DataFrame({"A": [1.0, 2.0, 3.0]}, index=index)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            path.write_text("old\n", encoding="utf-8")
            with patch.object(Path, "replace", side_effect=PermissionError("locked")):
                _save_cache(frame, path)
            loaded = _load_cache(path)

        pd.testing.assert_frame_equal(loaded, frame, check_freq=False)

    def test_retry_saves_successful_ticker_to_cache(self):
        from market_data_cache import update_adjusted_close_cache

        attempts = {"A": 0}
        index = pd.bdate_range("2020-01-01", "2020-01-10")

        def fetch(ticker, start, end):
            attempts[ticker] += 1
            if attempts[ticker] == 1:
                return pd.Series(dtype=float)
            return pd.Series(np.arange(len(index)) + 10.0, index=index, name=ticker)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            prices, audit = update_adjusted_close_cache(
                ["A"], "2020-01-01", "2020-01-11", path,
                fetch_ticker=fetch, max_attempts=2, pause_seconds=0,
            )
            stored = pd.read_csv(path, index_col=0, parse_dates=True)

        self.assertEqual(attempts["A"], 2)
        self.assertEqual(list(prices.columns), ["A"])
        self.assertEqual(list(stored.columns), ["A"])
        self.assertEqual(audit.loc[0, "status"], "downloaded")

    def test_complete_cached_ticker_skips_network_fetch(self):
        from market_data_cache import update_adjusted_close_cache

        index = pd.bdate_range("2020-01-01", "2020-01-10")
        calls = []

        def fetch(ticker, start, end):
            calls.append(ticker)
            raise AssertionError("complete cache must not call network")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prices.csv"
            pd.DataFrame({"A": np.arange(len(index)) + 10.0}, index=index).to_csv(path)
            prices, audit = update_adjusted_close_cache(
                ["A"], "2020-01-01", "2020-01-11", path,
                fetch_ticker=fetch, max_attempts=2, pause_seconds=0,
            )

        self.assertFalse(calls)
        self.assertEqual(len(prices), len(index))
        self.assertEqual(audit.loc[0, "status"], "cached")


if __name__ == "__main__":
    unittest.main()
