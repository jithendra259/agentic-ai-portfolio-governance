import unittest

import pandas as pd

from src.memory.artifact_store import ArtifactStore


class ArtifactStoreTests(unittest.TestCase):
    def test_save_and_load_dataframe_and_series_without_mongo(self):
        store = ArtifactStore(mongo_uri="")
        frame = pd.DataFrame([{"AAPL": 0.01, "MSFT": 0.02}])
        series = pd.Series({"AAPL": 0.6, "MSFT": 0.4}, name="weights")

        frame_id = store.save(frame, kind="returns_df")
        series_id = store.save(series, kind="weights")

        self.assertTrue(frame_id.startswith("artifact-"))
        self.assertTrue(series_id.startswith("artifact-"))
        loaded_frame = store.load(frame_id)
        loaded_series = store.load(series_id)
        self.assertEqual(float(loaded_frame.iloc[0]["AAPL"]), 0.01)
        self.assertEqual(float(loaded_series["MSFT"]), 0.4)
        self.assertEqual(loaded_series.name, "weights")


if __name__ == "__main__":
    unittest.main()
