from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import json
import unittest

import numpy as np
import pandas as pd

from src.memory.state_sanitizer import SanitizingCheckpointer, sanitize_for_mongodb


@dataclass
class MetricBundle:
    ticker: str
    value: Decimal


class FakeCheckpointer:
    def __init__(self):
        self.put_args = None
        self.put_kwargs = None
        self.put_writes_args = None

    def put(self, *args, **kwargs):
        self.put_args = args
        self.put_kwargs = kwargs
        return {"status": "saved"}

    def put_writes(self, *args, **kwargs):
        self.put_writes_args = args
        return {"status": "writes_saved"}


class StateSanitizerTests(unittest.TestCase):
    def test_sanitizes_common_checkpoint_toxic_objects(self):
        state = {
            "frame": pd.DataFrame(
                [
                    {"ticker": "AAPL", "close": Decimal("201.50")},
                    {"ticker": "MSFT", "close": Decimal("430.12")},
                ]
            ),
            "series": pd.Series({"AAPL": Decimal("0.6"), "MSFT": Decimal("0.4")}),
            "array": np.array([1.1, 2.2]),
            "timestamp": datetime(2026, 6, 14, tzinfo=timezone.utc),
            "symbols": {"MSFT", "AAPL"},
            "bundle": MetricBundle("AAPL", Decimal("1.247")),
        }

        sanitized = sanitize_for_mongodb(state)

        json.dumps(sanitized)
        self.assertEqual(sanitized["frame"][0]["ticker"], "AAPL")
        self.assertEqual(sanitized["frame"][0]["close"], "201.50")
        self.assertEqual(sanitized["series"]["AAPL"], "0.6")
        self.assertEqual(sanitized["array"], [1.1, 2.2])
        self.assertEqual(sanitized["timestamp"], "2026-06-14T00:00:00+00:00")
        self.assertEqual(sorted(sanitized["symbols"]), ["AAPL", "MSFT"])
        self.assertEqual(sanitized["bundle"]["value"], "1.247")

    def test_sanitizing_checkpointer_scrubs_checkpoint_payload(self):
        fake = FakeCheckpointer()
        wrapper = SanitizingCheckpointer(fake)
        config = {"configurable": {"thread_id": "portfolio-thread"}}
        checkpoint = {
            "channel_values": {
                "returns_df": pd.DataFrame([{"AAPL": np.float64(0.01)}]),
                "weights": pd.Series({"AAPL": Decimal("0.65")}),
            }
        }

        result = wrapper.put(
            config,
            checkpoint,
            {"sharpe": Decimal("1.234")},
            {"version": np.array([1, 2])},
        )

        self.assertEqual(result, {"status": "saved"})
        self.assertIs(fake.put_args[0], config)
        json.dumps(fake.put_args[1])
        json.dumps(fake.put_args[2])
        json.dumps(fake.put_args[3])
        self.assertEqual(fake.put_args[1]["channel_values"]["returns_df"][0]["AAPL"], 0.01)
        self.assertEqual(fake.put_args[1]["channel_values"]["weights"]["AAPL"], "0.65")
        self.assertEqual(fake.put_args[2]["sharpe"], "1.234")
        self.assertEqual(fake.put_args[3]["version"], [1, 2])

    def test_sanitizing_checkpointer_scrubs_pending_writes(self):
        fake = FakeCheckpointer()
        wrapper = SanitizingCheckpointer(fake)
        config = {"configurable": {"thread_id": "portfolio-thread"}}

        wrapper.put_writes(config, [("metrics", {"beta": np.float64(1.247)})], "task-1")

        self.assertIs(fake.put_writes_args[0], config)
        json.dumps(fake.put_writes_args[1])
        self.assertEqual(fake.put_writes_args[1][0][1]["beta"], 1.247)


if __name__ == "__main__":
    unittest.main()
