import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.generate_dynamic_plot import OUTPUT_DIR, generate_financial_plot
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


if __name__ == "__main__":
    unittest.main()
