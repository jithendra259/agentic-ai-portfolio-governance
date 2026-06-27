# Chatbot and Analytics Evidence Appendix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add all ten chatbot/analytics screenshots as three contextual evidence groups immediately before References, with accurate explanatory matter and research repository/contact information.

**Architecture:** `main.tex` owns the appendix narrative and multi-panel layouts. `test_report_contract.py` inventories the source directory and enforces exactly one active image reference per screenshot plus the required evaluation and contact metadata. The existing Tectonic and PyMuPDF workflow verifies the integrated PDF after both this appendix and the previously approved figure redistribution.

**Tech Stack:** Python 3.13 `unittest`, LaTeX `graphicx`/`minipage`, Tectonic, PyMuPDF.

---

### Task 1: Add the screenshot appendix contract

**Files:**
- Modify: `report/New Project/tests/test_report_contract.py`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Add the screenshot directory and expected filenames**

```python
CHATBOT_IMAGES = REPORT_DIR / "chatbotimages"
CHATBOT_SCREENSHOTS = {
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
```

- [ ] **Step 2: Add source-inventory, single-reference, and metadata tests**

```python
def test_chatbot_screenshot_inventory_matches_report_contract(self):
    discovered = {path.name for path in CHATBOT_IMAGES.glob("*.png")}
    self.assertEqual(CHATBOT_SCREENSHOTS, discovered)

def test_every_chatbot_screenshot_is_rendered_once(self):
    for filename in CHATBOT_SCREENSHOTS:
        with self.subTest(filename=filename):
            self.assertEqual(1, self.text.count("chatbotimages/" + filename))

def test_chatbot_evidence_appendix_and_contact_metadata_are_present(self):
    required = [
        "Chatbot and Analytics Interaction Evidence",
        "50 standardized queries",
        "70.0\\%",
        "35 out of 50",
        "1.25 seconds",
        "https://github.com/jithendra259/fintech_chatbotk",
        "https://github.com/jithendra259/agentic-ai-portfolio-governance",
        "K J Subramanyam and Sunayana Jadhav",
        "K J Somaiya College of Engineering, Mumbai",
        "Somaiya Vidyavihar University",
    ]
    for value in required:
        with self.subTest(value=value):
            self.assertIn(value, self.text)
```

- [ ] **Step 3: Run the tests and verify the appendix tests fail**

Run:

```powershell
python -m unittest 'report/New Project/tests/test_report_contract.py' -v
```

Expected: the existing 12 tests pass and the new screenshot-reference/metadata tests fail.

### Task 2: Add the grouped appendix and explanatory matter

**Files:**
- Modify: `report/New Project/main.tex:3048`
- Test: `report/New Project/tests/test_report_contract.py`

- [ ] **Step 1: Add the appendix heading and evaluation framing before References**

Insert after the existing appendix content and before `\section*{References}`:

```latex
\clearpage
\section{Chatbot and Analytics Interaction Evidence}
\label{app:chatbot-analytics-evidence}

This appendix presents illustrative chatbot responses, generated charts, and analytical dashboard outputs from the post-hoc evaluation. The aggregate evaluation covered 50 standardized queries. Of these, 35 out of 50 were reported as factually aligned with the MongoDB blackboard ground truth, corresponding to a 70.0\% zero-hallucination rate and meeting the stated institutional threshold of at least 70\%. Mean response latency was 1.25 seconds. These aggregate results come from the evaluation record; the ten screenshots below illustrate representative outputs and do not independently reproduce the complete query-level audit log.
```

- [ ] **Step 2: Add Figure 21 as a four-panel factual-output group**

Use a two-by-two arrangement for `233402`, `233759`, `150834`, and `151036`. Add inline panel descriptors under each image. Caption the combined figure “Factual retrieval, portfolio recommendation, allocation comparison, and conversational explanation outputs.” Add label `fig:chatbot-factual-portfolio-evidence`. Follow it with one paragraph explaining grounding and the illustrative-not-complete-audit limitation.

- [ ] **Step 3: Add Figure 22 as a four-panel network/risk group**

Use a two-by-two arrangement for `125451`, `233832`, `234105`, and `150757`. Caption the combined figure “Network contagion, correlation, and outlier-risk diagnostics exposed through the analytical interface.” Add label `fig:chatbot-network-risk-evidence`. Explain that the panels expose intermediate analytical state but do not independently establish causal contagion or superior performance.

- [ ] **Step 4: Add Figure 23 as a two-panel allocation/volatility group**

Use independently sized minipages for `132757` and `150747`. Caption the combined figure “Optimized allocation and return/volatility diagnostic views.” Add label `fig:chatbot-allocation-volatility-evidence`. Explain that the outputs support comparison and review and must be interpreted with source data and audit records.

- [ ] **Step 5: Add repositories and contact immediately before References**

```latex
\subsection*{Research Repositories and Contact}
\addcontentsline{toc}{subsection}{Research Repositories and Contact}
\begin{description}[leftmargin=4.2cm,style=multiline]
  \item[Paper 1 repository:] \url{https://github.com/jithendra259/fintech_chatbotk}
  \item[Paper 2 repository:] \url{https://github.com/jithendra259/agentic-ai-portfolio-governance}
  \item[Authors:] K J Subramanyam and Sunayana Jadhav
  \item[Institution:] K J Somaiya College of Engineering, Mumbai
  \item[University:] Somaiya Vidyavihar University
\end{description}
```

- [ ] **Step 6: Run the full report contract**

```powershell
python -m unittest 'report/New Project/tests/test_report_contract.py' -v
```

Expected: all tests pass.

### Task 3: Compile and visually verify the combined report

**Files:**
- Regenerate: `report/New Project/main.pdf`
- Inspect: `report/New Project/.latex-build/main.log`
- Inspect: `tmp/pdfs/chatbot-evidence-appendix/`

- [ ] **Step 1: Regenerate the managed figure manifest and run tests**

```powershell
python 'report/New Project/generate_result_figure_appendix.py'
python -m unittest 'report/New Project/tests/test_report_contract.py' -v
```

Expected: the manifest regenerates successfully and all tests pass.

- [ ] **Step 2: Compile with the available Tectonic executable**

From `report/New Project`, run the currently available bundled Tectonic binary with:

```powershell
& 'C:\Users\jithe\.codex\.tmp\bundled-marketplaces\openai-bundled\plugins\latex\bin\tectonic.exe' -X compile main.tex --outdir .latex-build --keep-logs --keep-intermediates
Copy-Item -LiteralPath '.latex-build\main.pdf' -Destination 'main.pdf' -Force
```

Expected: exit code 0 and an updated `main.pdf`.

- [ ] **Step 3: Inspect compile diagnostics**

```powershell
rg -n "error|undefined|multiply defined|overfull|underfull" '.latex-build/main.log'
```

Expected: no errors, undefined references, or multiply defined labels. Investigate layout warnings on the new appendix pages.

- [ ] **Step 4: Render and inspect the new appendix pages**

Use PyMuPDF to locate the “Chatbot and Analytics Interaction Evidence” heading, render through the page before References, and save page images/contact sheets under `tmp/pdfs/chatbot-evidence-appendix/`. Verify all ten images are present once, panel descriptors are readable, chatbot text is legible, captions remain with their groups, repository URLs fit inside the page, and no blank or clipped page is introduced.

- [ ] **Step 5: Verify the entire report integration**

Confirm the redistributed result figures remain at their analytical points of use, the managed placement manifest remains compact, the chatbot appendix occurs immediately before References, and the final PDF page count/content are internally consistent.
