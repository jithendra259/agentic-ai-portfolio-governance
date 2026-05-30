import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.generate_dynamic_plot import OUTPUT_DIR, generate_financial_plot
from src.agents.custom_plot_tool import generate_custom_plot
from ui import app as ui_app


class PlotRenderingTests(unittest.TestCase):
    def test_generate_financial_plot_returns_outputs_route_not_filesystem_path(self):
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

        self.assertIn("![Test Plot](", result)
        self.assertIn("](/outputs/", result)
        self.assertNotIn("C:/", result)
        self.assertNotIn("\\outputs\\", result)
        filename = result.rsplit("(", 1)[1].rstrip(")")
        output_path = OUTPUT_DIR / Path(filename).name
        self.assertTrue(output_path.exists())
        self.assertEqual(OUTPUT_DIR, Path(__file__).resolve().parent.parent / "outputs")

    def test_ui_rewrites_absolute_outputs_path_to_backend_url(self):
        markdown = "Plot generated successfully: ![Chart](C:/repo/outputs/test_chart.png)"

        with patch.object(ui_app, "STREAM_API_URL", "http://127.0.0.1:8000/chat/stream"):
            rewritten = ui_app._rewrite_plot_markdown(markdown)

        self.assertEqual(
            rewritten,
            "Plot generated successfully: ![Chart](http://127.0.0.1:8000/outputs/test_chart.png)",
        )

    def test_custom_correlation_heatmap_aligns_uneven_price_histories(self):
        result = generate_custom_plot.func(
            data={
                "prices": {
                    "AAPL": [
                        {"date": "2020-01-01", "close": 100},
                        {"date": "2020-01-02", "close": 101},
                        {"date": "2020-01-03", "close": 102},
                        {"date": "2020-01-06", "close": 103},
                    ],
                    "MSFT": [
                        {"date": "2020-01-02", "close": 50},
                        {"date": "2020-01-03", "close": 51},
                        {"date": "2020-01-06", "close": 52},
                    ],
                }
            },
            description="correlation heatmap for technology sector tickers using log returns",
        )

        self.assertIn("Plot generated successfully", result)
        self.assertIn("aligned daily log-return observations", result)
        filename = result.split("](/outputs/", 1)[1].split(")", 1)[0]
        self.assertTrue((OUTPUT_DIR / filename).exists())

    def test_custom_close_return_plot_uses_deterministic_renderer(self):
        result = generate_custom_plot.func(
            data={
                "returns": {"AAPL": [0.01, -0.005, 0.003]},
                "return_dates_by_ticker": {"AAPL": ["2020-01-02", "2020-01-03", "2020-01-06"]},
            },
            description="close return plot of AAPL",
        )

        self.assertIn("Plot generated successfully", result)
        self.assertIn("Daily close returns", result)
        filename = result.split("](/outputs/", 1)[1].split(")", 1)[0]
        self.assertTrue((OUTPUT_DIR / filename).exists())

    def test_ui_falls_back_when_stream_ends_prematurely(self):
        class StreamResponse:
            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=True):
                raise requests.exceptions.ChunkedEncodingError("Response ended prematurely")

        class FallbackResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "Fallback answer with ![Chart](/outputs/test_chart.png)"}

        with patch.object(ui_app, "STREAM_API_URL", "http://127.0.0.1:8000/chat/stream"), patch.object(
            ui_app, "API_URL", "http://127.0.0.1:8000/chat"
        ), patch.object(requests, "post", side_effect=[StreamResponse(), FallbackResponse()]):
            outputs = list(ui_app.chat_with_api("plot AAPL", [], "session-1"))

        self.assertEqual(len(outputs), 1)
        self.assertIn("Fallback answer", outputs[0])
        self.assertIn("http://127.0.0.1:8000/outputs/test_chart.png", outputs[0])


if __name__ == "__main__":
    unittest.main()
