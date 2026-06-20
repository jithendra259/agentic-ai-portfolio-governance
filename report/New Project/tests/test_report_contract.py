from pathlib import Path
import re
import unittest


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT = REPORT_DIR.parents[1]
MAIN = REPORT_DIR / "main.tex"
ANNEX = REPORT_DIR / "generated_result_figures.tex"
EVIDENCE = REPORT_DIR / "generated_verified_results_tables.tex"
FIGURE_ROOT = ROOT / "notebook" / "figures_universe_analysis"


class ThesisReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = MAIN.read_text(encoding="utf-8")

    def test_verified_study_period_and_protocol_boundaries_are_present(self):
        self.assertIn("Study Period:} 2014--2025", self.text)
        self.assertIn("2014--2019", self.text)
        self.assertIn("2020--2022", self.text)
        self.assertIn("2023--2025", self.text)
        self.assertNotIn("Study Period:} 2012--2025", self.text)
        self.assertNotIn("2005--2025", self.text)

    def test_primary_and_supplemental_objectives_are_distinct(self):
        self.assertIn(
            "Static G-CVaR: fixed graph-aware tail-risk optimizer", self.text
        )
        self.assertIn(
            "Adaptive G-CVaR: instability-gated graph-aware tail-risk optimizer",
            self.text,
        )
        self.assertIn("Supplemental Linear-Centrality Adaptive G-CVaR", self.text)
        self.assertRegex(self.text, r"w\^\{?\\top\}? A_t w")
        self.assertRegex(self.text, r"c_t\^\{?\\top\}? w")

    def test_verified_implementation_components_are_documented(self):
        required = [
            "React 19",
            "Vite 8",
            "MUI X Chat",
            "FastAPI",
            "LangGraph",
            "MongoDB",
            "Supabase PostgreSQL",
            "NDJSON",
            "advisory-only",
            "/chat/stream",
            "/api/analytics",
            "/api/governance/decision",
        ]
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, self.text)

    def test_unsupported_fixed_claims_are_removed(self):
        self.assertNotIn("38.1\\%", self.text)
        self.assertNotIn("25.9\\% CVaR", self.text)
        self.assertNotIn(
            "statistically significant reductions in crisis-period tail risk",
            self.text,
        )
        self.assertNotRegex(self.text.lower(), r"always outperforms|universally superior")

    def test_report_uses_portable_graphic_paths(self):
        preamble = self.text.split(r"\begin{document}", 1)[0]
        self.assertNotRegex(preamble, r"[A-Z]:/")
        self.assertNotRegex(preamble, r"[A-Z]:\\")

    def test_automatic_navigation_replaces_static_page_tables(self):
        self.assertIn(r"\tableofcontents", self.text)
        self.assertIn(r"\listoffigures", self.text)
        self.assertIn(r"\listoftables", self.text)
        self.assertNotIn(r"\section*{Table of Contents}", self.text)

    def test_generated_evidence_is_included_and_auditable(self):
        self.assertTrue(EVIDENCE.exists())
        evidence = EVIDENCE.read_text(encoding="utf-8")
        required = [
            "Executive empirical result summary",
            "Experiment-lane interpretation hierarchy",
            "Reproducibility and execution environment",
            "gcvar_test_governance_ranking.csv",
            "adaptive_graph_cvar_v2_results.csv",
            "universe_data_coverage_audit_2014_2025.csv",
            "final_technical_validation_checks.csv",
            "121",
        ]
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, evidence)
        self.assertIn(r"\input{generated_verified_results_tables.tex}", self.text)

    def test_every_result_png_is_referenced_once(self):
        self.assertTrue(ANNEX.exists())
        annex = ANNEX.read_text(encoding="utf-8")
        pngs = sorted(FIGURE_ROOT.rglob("*.png"))
        self.assertEqual(121, len(pngs))
        references = re.findall(r"\\ResultFigure\{([^}]+)\}", annex)
        self.assertEqual(len(pngs), len(references))
        self.assertEqual(len(references), len(set(references)))
        expected = {
            "../../notebook/figures_universe_analysis/"
            + png.relative_to(FIGURE_ROOT).as_posix()
            for png in pngs
        }
        self.assertEqual(expected, set(references))


if __name__ == "__main__":
    unittest.main()
