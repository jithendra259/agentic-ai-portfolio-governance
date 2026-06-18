# Plot Prompt Library Design

**Date:** 2026-06-18
**Status:** Approved for implementation planning

## Objective

Improve plot-generation accuracy by creating a large, structured library of realistic user prompts without expanding the already-large global system prompt. The same catalog will support user-facing suggestions and automated routing evaluations.

## Scope

The library will cover every plot family currently rendered by the chat interface:

- line
- bar
- pie
- scatter
- sparkline
- sankey
- candlestick
- heatmap
- network
- funnel
- radar
- gauge
- radial bar
- radial line
- box plot

The initial target is 12-20 curated cases per family, producing approximately 250 cases. Coverage is more important than reaching an arbitrary count.

## Architecture

### Canonical prompt catalog

Create one machine-readable catalog as the source of truth. Organize entries by plot family and assign each entry a stable identifier. Each entry will contain:

- `id`
- `plot_type`
- `prompt`
- `category`
- `expected_tool`
- `expected_analysis_task`, when applicable
- `expected_defaults`
- `required_payload_fields`
- `ui_visible`
- `tags`

The catalog must describe outcomes rather than duplicate full system-prompt instructions. Backend evaluations and frontend suggestions will select from this shared content through small adapters appropriate to their runtimes.

### Prompt categories

Each plot family will include cases from these categories:

1. Basic explicit request
2. Natural financial request without a chart-type keyword
3. Multi-series or multi-category request
4. Styling or layout customization
5. Follow-up request relying on conversation context
6. Request with omitted optional values that should use safe defaults
7. Ambiguous request that requires a bounded clarification
8. Invalid or incompatible request that should return a useful explanation
9. Synonyms and informal wording
10. Boundary-volume cases appropriate to the renderer

Plot-specific variants will be added where they matter. For example, bar prompts will distinguish single, grouped, stacked, horizontal, and vertical layouts; scatter prompts will include bubble data; and pie prompts will enforce part-to-whole semantics.

### Runtime prompt selection

The global system prompt will retain concise chart-routing principles. Detailed examples will be retrieved by detected intent and plot family, then injected only when relevant. A request for a heatmap must not consume examples for sankey, gauge, or unrelated families.

Selection will be deterministic where routing already identifies a plot family. When routing is uncertain, the selector may provide a small set of examples from the most likely families. The injected example budget must be capped to prevent context growth.

### User-facing suggestions

The chat empty state or plot-help experience will display a small rotating subset of entries marked `ui_visible`. Suggestions will be grouped or filtered by plot family rather than displaying hundreds of prompts simultaneously. Selecting a suggestion will populate or submit the exact catalog prompt.

User-facing examples must be understandable without internal tool names, cache keys, or implementation terminology.

## Data Flow

1. A catalog entry is authored and validated against the schema.
2. Backend evaluation loads the entry and submits its `prompt` to the routing layer.
3. The result is compared with the entry's expected plot type, tool, analysis task, defaults, and payload requirements.
4. Runtime example selection filters catalog entries by detected family and category, then injects a bounded subset.
5. The frontend adapter exposes only entries marked `ui_visible` as prompt suggestions.

## Validation and Error Handling

Catalog validation will reject:

- duplicate identifiers
- unsupported plot types
- missing prompts or expected tools
- incompatible tool and plot combinations
- missing required expectations for analysis-driven plots
- UI suggestions containing internal implementation language

Box plots require explicit handling because the frontend renders `box`, while the general dynamic plot generator does not list it in `SUPPORTED_PLOTS`. Their catalog entries must expect the approved analysis route rather than the generic plot generator.

Invalid-input cases are successful tests when the system produces the expected clarification or bounded error. They must not be counted as successful plot generations.

## Testing Strategy

### Schema tests

Verify that every entry is structurally valid, uniquely identified, and references supported runtime behavior.

### Coverage tests

Require every plot family and core prompt category to have at least one entry. Add family-specific requirements for important variants such as grouped and stacked bars.

### Routing evaluations

Run catalog prompts through the deterministic intent and decision layers where possible. Assert:

- selected chart family
- selected tool
- selected analysis task
- inferred defaults
- required payload fields

LLM-dependent evaluations should report accuracy by family and category and preserve failing prompt IDs for diagnosis. They should not rely solely on exact prose matching.

### Frontend tests

Verify that only `ui_visible` entries appear, selection preserves the catalog text, and filtering by family works. Existing fixture endpoints and the plot gallery remain the visual-rendering verification path; the prompt catalog does not replace renderer tests.

## Success Criteria

- All 15 rendered plot families have catalog coverage.
- The initial catalog contains approximately 250 non-duplicate, meaningful cases.
- Each case has machine-checkable expected behavior.
- Relevant examples can be selected without injecting the entire catalog.
- User-facing suggestions are generated from the canonical catalog.
- Evaluation output identifies accuracy and failures by plot family and category.
- Existing plot and chat tests continue to pass.

## Out of Scope

- Adding new plot renderers
- Changing chart visual design
- Allowing arbitrary Python execution
- Treating prompt count alone as the accuracy metric
- Replacing deterministic routing with prompt-only behavior

## Implementation Boundaries

Implementation should preserve the current dirty worktree and avoid unrelated chart refactoring. The prompt catalog, adapters, selector, UI suggestion integration, and evaluation tests should be separable units with explicit interfaces. Any mismatch discovered between documented and actual chart capability should be recorded and tested rather than hidden by a prompt workaround.
