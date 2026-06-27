from pathlib import Path
from collections import Counter
import re
import unittest


REPORT_DIR = Path(__file__).resolve().parents[1]
ROOT = REPORT_DIR.parents[1]
MAIN = REPORT_DIR / "main.tex"
ANNEX = REPORT_DIR / "generated_result_figures.tex"
EVIDENCE = REPORT_DIR / "generated_verified_results_tables.tex"
FIGURE_ROOT = ROOT / "notebook" / "figures_universe_analysis"
CHATBOT_IMAGES = REPORT_DIR / "chatbotimages"
EXPECTED_CHATBOT_IMAGES = {
    "Screenshot 2026-06-04 125451.png",
    "Screenshot 2026-06-04 132757.png",
    "Screenshot 2026-06-21 233402.png",
    "Screenshot 2026-06-21 233759.png",
    "Screenshot 2026-06-21 233832.png",
    "Screenshot 2026-06-21 234105.png",
    "Screenshot 2026-06-22 150747.png",
    "Screenshot 2026-06-22 150757.png",
    "Screenshot 2026-06-22 150834.png",
    "Screenshot 2026-06-22 151036.png",
}


def strip_unescaped_latex_comments(text):
    """Remove LaTeX comments while preserving escaped percent signs."""
    active_lines = []
    for line in text.splitlines(keepends=True):
        for index, character in enumerate(line):
            if character != "%":
                continue
            preceding_backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                preceding_backslashes += 1
                cursor -= 1
            if preceding_backslashes % 2 == 0:
                line = line[:index]
                if active_lines or line:
                    line += "\n"
                break
        active_lines.append(line)
    return "".join(active_lines)
REDISTRIBUTED_STEMS = {
    "adaptive_gcvar_evidence_triangle",
    "gcvar_implementation_audit",
    "terminal_value_vs_cvar_tradeoff",
    "time_varying_graph_exposure",
    "cvar_drawdown_improvement_vs_equal_weight",
    "computed_component_contribution_to_sharpe",
    "ablation_composite_score_waterfall",
    "U1_performance_diagnostics",
    "U1_stress_overlay",
    "U2_performance_diagnostics",
    "U2_stress_overlay",
    "crisis_only_governance_comparison",
    "sample_hitl_100000_terminal_value_comparison",
    "sample_hitl_decision_distribution_and_event_impact",
    "sample_hitl_ticker_network_risk_vs_adopted_allocation",
}


class ThesisReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_text = MAIN.read_text(encoding="utf-8")
        cls.text = strip_unescaped_latex_comments(
            re.sub(r"\\iffalse.*?\\fi", "", raw_text, flags=re.DOTALL)
        )

    def chatbot_appendix(self):
        match = re.search(
            r"\\section\{Appendix A --- Chatbot Answer Screenshots\}"
            r"(?P<body>.*?)"
            r"\\section\*\{References\}",
            self.text,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(
            match,
            "expected chatbot appendix between its section heading and References",
        )
        return match.group("body")

    def test_verified_study_period_and_protocol_boundaries_are_present(self):
        self.assertIn("Study Period:} 2014--2025", self.text)
        self.assertIn("2014--2019", self.text)
        self.assertIn("2020--2022", self.text)
        self.assertIn("2023--2025", self.text)
        self.assertNotIn("Study Period:} 2012--2025", self.text)
        self.assertNotIn("Time Period: 2005--2025", self.text)

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
        searchable_evidence = evidence.replace(r"\_", "_")
        required = [
            "Executive empirical result summary",
            "Experiment-lane interpretation hierarchy",
            "Adaptive G-CVaR untouched-test results by universe",
            "Reproducibility and execution environment",
            "gcvar_test_governance_ranking.csv",
            "adaptive_graph_cvar_v2_results.csv",
            "universe_data_coverage_audit_2014_2025.csv",
            "final_technical_validation_checks.csv",
            "121",
        ]
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, searchable_evidence)
        self.assertIn(r"\input{generated_verified_results_tables.tex}", self.text)

    def test_generated_evidence_reports_all_universes_individually(self):
        evidence = EVIDENCE.read_text(encoding="utf-8")
        for index in range(1, 12):
            with self.subTest(universe=index):
                self.assertRegex(evidence, rf"(?m)^U{index} &")

    def test_generated_evidence_has_valid_latex_row_escapes(self):
        self.assertTrue(EVIDENCE.exists())
        evidence = EVIDENCE.read_text(encoding="utf-8")
        self.assertNotIn("\t", evidence)
        self.assertIn(r"\texttt{", evidence)
        row_prefixes = (
            "Assets and universes",
            "Core strategies",
            "Python &",
            "Git commit at report generation",
        )
        for line in evidence.splitlines():
            if line.startswith(row_prefixes):
                with self.subTest(line=line):
                    self.assertTrue(line.endswith(r"\\"), line)

    def test_manifest_inventories_every_redistributed_figure(self):
        self.assertTrue(ANNEX.exists())
        annex = ANNEX.read_text(encoding="utf-8")
        manifest_match = re.search(
            r"\\begin\{longtable\}.*?\\end\{longtable\}",
            annex,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(manifest_match, "generated figure manifest is missing")
        manifest = manifest_match.group(0).replace(r"\allowbreak{}", "")
        self.assertEqual(15, len(REDISTRIBUTED_STEMS))
        for stem in REDISTRIBUTED_STEMS:
            with self.subTest(stem=stem):
                self.assertIn(stem.replace("_", r"\_"), manifest)

    def test_redistributed_figures_render_once_in_body_and_not_annex(self):
        self.assertTrue(ANNEX.exists())
        annex = ANNEX.read_text(encoding="utf-8")
        for stem in REDISTRIBUTED_STEMS:
            with self.subTest(stem=stem):
                self.assertEqual(1, self.text.count(stem + ".png"))
                self.assertNotIn(stem + ".png", annex)

    def test_figure_labels_are_unique(self):
        generated_inputs = [ANNEX, EVIDENCE]
        report_text = self.text + "".join(
            path.read_text(encoding="utf-8")
            for path in generated_inputs
            if path.exists()
        )
        labels = re.findall(r"\\label\{(fig:[^}]+)\}", report_text)
        self.assertEqual(len(labels), len(set(labels)))

    def test_chatbot_image_inventory_matches_appendix_design(self):
        discovered = {path.name for path in CHATBOT_IMAGES.glob("*.png")}
        self.assertEqual(10, len(EXPECTED_CHATBOT_IMAGES))
        self.assertEqual(EXPECTED_CHATBOT_IMAGES, discovered)

    def test_each_chatbot_image_is_referenced_exactly_once(self):
        appendix = self.chatbot_appendix()
        references = re.findall(
            r"\\includegraphics(?:\s*\[[^]]*\])?\s*\{([^}]+)\}",
            appendix,
        )
        actual = Counter(
            target for target in references if target.startswith("chatbotimages/")
        )
        expected = Counter(
            {f"chatbotimages/{filename}": 1 for filename in EXPECTED_CHATBOT_IMAGES}
        )
        self.assertEqual(expected, actual)

    def test_chatbot_appendix_evaluation_and_contact_metadata_are_present(self):
        appendix = self.chatbot_appendix()
        evaluation_claims = {
            "query cohort": r"50 standardized queries",
            "aligned responses": r"35 out of 50",
            "zero-hallucination result": (
                r"(?:70\.0\\%.*?zero-hallucination|"
                r"zero-hallucination.*?70\.0\\%)"
            ),
            "institutional threshold": r"(?:threshold.*?(?:at least\s*)?70\\%|70\\%.*?threshold)",
            "mean latency": r"mean response latency.*?1\.25 seconds",
        }
        for claim, pattern in evaluation_claims.items():
            with self.subTest(claim=claim):
                self.assertRegex(
                    appendix,
                    re.compile(pattern, flags=re.DOTALL | re.IGNORECASE),
                )
        labeled_entries = {
            "Paper 1 repository": "https://github.com/jithendra259/fintech_chatbotk",
            "Paper 2 repository": "https://github.com/jithendra259/agentic-ai-portfolio-governance",
            "Authors": "K J Subramanyam and Sunayana Jadhav",
            "Institution": "K J Somaiya College of Engineering, Mumbai",
            "University": "Somaiya Vidyavihar University",
        }
        for label, value in labeled_entries.items():
            with self.subTest(label=label):
                self.assertRegex(
                    appendix,
                    re.compile(
                        rf"{re.escape(label)}.{{0,100}}?{re.escape(value)}",
                        flags=re.DOTALL,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
