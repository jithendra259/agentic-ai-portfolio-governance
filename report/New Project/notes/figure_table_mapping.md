# Figure and Table Mapping for Thesis

## Purpose of This Note
This note maps extracted notebook outputs to likely thesis figures and tables based on:
- the thesis outline previously provided
- `results/paper1/manifest.md`
- `results/paper2/manifest.md`
- `notes/paper1_analysis.md`
- `notes/paper2_analysis.md`

This is a working mapping, not a final locked numbering scheme.

## Paper 1 Mapping

### Likely Figure Mapping
| Thesis Figure | Target Section | Candidate Asset | Reason |
|---|---|---|---|
| Figure 2 | Sec 4.1.5 | `figures/paper1/cell_61_img_1.png` | likely instability-index time-series or related summary plot |
| Figure 3 | Sec 4.1.5 | `figures/paper1/cell_68_img_1.png` | likely distribution/regime timeline style visual |
| Figure 7 | Sec 4.7.8 | `figures/paper1/cell_27_img_1.png` | explicitly labeled cumulative wealth in manifest |
| Figure 8 | Sec 4.7.8 | `figures/paper1/cell_41_img_1.png` | likely rolling performance plot |
| Figure 9 | Sec 4.7.8 | `figures/paper1/cell_57_img_1.png` | likely crisis/regime comparison bar plot |
| Figure 10 | Sec 4.7.7 | `figures/paper1/cell_47_img_1.png` | likely threshold sensitivity or regime occurrence plot |

### Likely Table Mapping
| Thesis Table | Target Section | Candidate Asset | Reason |
|---|---|---|---|
| T3 | Sec 4.1 | `results/paper1/cell_12_table_1.html` | summary statistics of returns / inputs |
| T5 | Sec 4.7.1 | `results/paper1/cell_39_table_1.html` | rolling results dataframe likely supports overall performance |
| T6 | Sec 4.7.6 | `results/paper1/cell_43_table_1.html` | manifest suggests threshold or parameter table nearby |
| T7 | Sec 4.7.7 | `results/paper1/cell_56_table_1.html` | crisis periods / threshold analysis candidate |
| T9 | Sec 4.7.10 | `results/paper1/cell_52_table_1.html` | universe results table candidate |
| T10 | Sec 4.7.11 | `results/paper1/cell_56_table_1.html` | crisis-period robustness candidate |
| T11 | Sec 4.7.4 | `results/paper1/cell_49_result_1.txt` + `results/paper1/cell_39_table_1.html` | summary stats and governance metrics |

### Additional Paper 1 Support Assets
- `results/paper1/cell_45_result_1.txt` — regime counts
- `results/paper1/cell_67_table_1.html` — likely regime timeline table
- `figures/paper1/cell_54_img_1.png` — likely cross-universe or regime comparison plot

## Paper 2 Mapping

### Likely Figure Mapping
| Thesis Figure | Target Section | Candidate Asset | Reason |
|---|---|---|---|
| Figure 11 | Sec 5.2 | `figures/paper2/cell_16_img_1.png` | manifest mentions NetworkX; likely holdings graph |
| Figure 12 | Sec 5.2 | `figures/paper2/cell_16_img_2.png` | likely centrality distribution / related graph output |
| Figure 13 | Sec 5.3 | `figures/paper2/cell_18_img_1.png` | likely gate/objective plot near mathematical setup |
| Figure 14 | Sec 5.4 | `figures/paper2/cell_26_img_1.png` | likely system architecture / multi-agent figure |
| Figure 15 | Sec 5.7 | `figures/paper2/cell_42_img_1.png` | candidate heatmap / regime activation style figure |
| Figure 16 | Sec 5.10 | `figures/paper2/cell_42_img_2.png` or `figures/paper2/cell_43_img_1.png` | likely ablation comparison |
| Figure 17 | Sec 5.12 | `figures/paper2/cell_47_img_1.png` | likely walk-forward validation plot |
| Figure 18 | Sec 5.14 | `figures/paper2/cell_44_img_1.png` | likely attribution scatter |
| Figure 19 | Sec 5.14 | `figures/paper2/cell_45_img_1.png` | likely trust / governance diagnostic |

### Likely Table Mapping
| Thesis Table | Target Section | Candidate Asset | Reason |
|---|---|---|---|
| T13 | Sec 5.2.5 | `results/paper2/cell_28_table_1.html` | main HTML table in notebook likely centrality/results table |
| T15 | Sec 5.12 | `results/paper2/cell_30_result_2.txt` + validation figures | walk-forward metrics candidate |
| T16 | Sec 5.7.1 | paper PDF values + notebook crisis outputs | crisis-only test appears explicitly in paper |
| T17 | Sec 5.8.1 | notebook logged summary outputs in `results/paper2/` | aggregate results likely in text outputs |
| T18 | Sec 5.9 | paper PDF regime table + notebook crisis outputs | regime-stratified metrics clearly reported in paper |
| T19 | Sec 5.10.2 | ablation outputs near `figures/paper2/cell_42_img_2.png` | ablation-specific support |
| T21 | Sec 5.13 | notebook text outputs + paper cross-universe discussion | cross-universe validation across 11 sectors |
| T22 | Sec 5.14 | notebook governance outputs + paper discussion | trustworthiness/XAI evaluation |

### Additional Paper 2 Support Assets
- `figures/paper2/cell_14_img_1.png`, `figures/paper2/cell_14_img_2.png` — early setup/diagnostic visuals
- `figures/paper2/cell_26_img_2.png`, `figures/paper2/cell_26_img_3.png` — additional architecture/flow candidates
- `figures/paper2/cell_30_img_1.png` — likely validation or summary plot
- `figures/paper2/cell_32_img_1.png` through `figures/paper2/cell_41_img_1.png` — likely regime, optimization, and performance visuals

## Direct Mapping Priorities
If only a small number of assets are selected first, use these:

### Priority Paper 1 Assets
- `figures/paper1/cell_27_img_1.png`
- `figures/paper1/cell_41_img_1.png`
- `figures/paper1/cell_57_img_1.png`
- `results/paper1/cell_39_table_1.html`
- `results/paper1/cell_52_table_1.html`

### Priority Paper 2 Assets
- `figures/paper2/cell_16_img_1.png`
- `figures/paper2/cell_18_img_1.png`
- `figures/paper2/cell_26_img_1.png`
- `figures/paper2/cell_42_img_1.png`
- `results/paper2/cell_28_table_1.html`

## Recommended Workflow Before Thesis Writing
1. Open the priority figure files visually and confirm what each actually shows.
2. Rename confirmed assets into thesis-friendly names.
3. Build chapter-specific notes from the confirmed assets.
4. Only then insert them into the thesis document.

## Suggested Next Notes
- `notes/paper1_section_mapping.md`
- `notes/paper2_section_mapping.md`
- `notes/chapter_4_paper1_methodology.md`
- `notes/chapter_5_paper2_methodology.md`