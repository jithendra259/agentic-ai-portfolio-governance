# Chatbot and Analytics Evidence Appendix Design

## Objective

Add all ten images from `report/New Project/chatbotimages/` to the thesis in a concise, evidence-oriented appendix immediately before References. Group related screenshots into readable composite figures, explain the analytical meaning of each group, and add the research repositories and contact information.

## Evidence Framing

The appendix will be titled **Chatbot and Analytics Interaction Evidence**. The source files are a mixture of chatbot responses, generated charts, portfolio diagnostics, and analytical dashboard outputs. They will not be described as twelve dialogue transcripts or as direct visual proof of every query in the 50-query evaluation.

The supplied post-hoc evaluation values will be reported as evaluation-summary results:

- 50 standardized queries;
- 35 factually aligned responses;
- 70.0% zero-hallucination rate;
- institutional threshold of at least 70%;
- mean response latency of 1.25 seconds.

The surrounding text will distinguish these aggregate reported metrics from the ten illustrative screenshots.

## Figure Groups

### Figure 21: Factual Retrieval and Portfolio Outputs

Use the following four screenshots:

- `Screenshot 2026-06-21 233402.png`
- `Screenshot 2026-06-21 233759.png`
- `Screenshot 2026-06-22 150834.png`
- `Screenshot 2026-06-22 151036.png`

These panels demonstrate historical price retrieval, numerical portfolio recommendations, current-versus-advisory allocation comparison, and an explanatory conversational response. The explanatory matter will connect them to factual grounding while noting that screenshots are illustrative outputs rather than the complete 50-query audit log.

### Figure 22: Network Contagion, Regime, and Risk Diagnostics

Use the following four screenshots:

- `Screenshot 2026-06-04 125451.png`
- `Screenshot 2026-06-21 233832.png`
- `Screenshot 2026-06-21 234105.png`
- `Screenshot 2026-06-22 150757.png`

These panels show institutional risk networks, ticker-level contagion detail, a correlation-style diagnostic matrix, and outlier-return detection. The explanation will describe how the interface exposes intermediate analytical state used by the governed workflow without claiming that the screenshots alone establish causal contagion or model superiority.

### Figure 23: Allocation and Volatility Analysis

Use the following two screenshots:

- `Screenshot 2026-06-04 132757.png`
- `Screenshot 2026-06-22 150747.png`

These panels show an optimized allocation view together with return-distribution and rolling-volatility diagnostics. The explanation will emphasize decision-support interpretation, readable visual comparison, and the need to interpret generated charts alongside the underlying data and audit records.

## Layout

- Figures 21 and 22 use a two-by-two panel layout across one or two pages depending on legibility.
- Figure 23 uses a two-panel layout with widths selected independently because the source images have different aspect ratios.
- Each panel receives a short `(a)` to `(d)` descriptor without adding a new LaTeX subcaption dependency.
- Images must remain large enough for chart labels and chatbot text to be legible in the compiled PDF.
- Every source file appears exactly once.

## Research Repositories and Contact

After the screenshot explanations and before References, add:

- Paper 1 repository: `https://github.com/jithendra259/fintech_chatbotk`
- Paper 2 repository: `https://github.com/jithendra259/agentic-ai-portfolio-governance`
- Authors: K J Subramanyam and Sunayana Jadhav
- Institution: K J Somaiya College of Engineering, Mumbai
- University: Somaiya Vidyavihar University

Repository addresses will use clickable `\url{}` formatting.

## Report Integration

- Insert the appendix after the existing experimental appendices and immediately before `\section*{References}`.
- Use automatic figure numbering rather than hard-coded page numbers or batch footers.
- Add the appendix to the table of contents.
- Preserve existing references and appendix content.

## Verification

1. Add contract tests that enumerate all ten source filenames and require exactly one active `\includegraphics` reference per image.
2. Require the appendix heading, aggregate evaluation metrics, repository links, authors, institution, and university.
3. Compile the current thesis with Tectonic.
4. Render the new appendix pages and inspect panel alignment, text size, caption proximity, page breaks, and screenshot legibility.
5. Confirm there are no missing files, duplicate labels, unresolved references, or images repeated elsewhere in the report.

## Acceptance Criteria

- All ten screenshots appear exactly once in three evidence groups.
- Captions and explanatory paragraphs accurately match the visible content.
- The 70.0% result is presented as an aggregate reported evaluation result, not inferred from the screenshots.
- Research repositories and contact information appear immediately before References.
- The final PDF compiles and the new appendix is readable without clipping, overlap, or misleading trial numbering.
