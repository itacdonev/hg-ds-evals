# Run Notes

## What we changed (notebook-only — the scorer is untouched)

> Both changes are applied **pre-scoring**, inside `import_traces_local.ipynb`. They rewrite the dataframe and the kwargs handed to `ToolParameterScorer.score()`. The scorer class in `ai-data-science/evals/src/.../tool_parameter.py` is **not modified** — once these rules stabilize they can graduate into the repo, and these two cells are the entry-points.

- **`scorers_to_run` augmentation** (cell `ddaf6335`) — drops `tool_parameter` from a row's `scorers_to_run` when the scorer cannot meaningfully run: (a) KB cases whose every expected entry is `knowledge_search` (parameters are fuzzy semantic strings the deterministic scorer can't compare), or (b) off-topic cases with `expected_tool_calls == []` (nothing to score). Routing and tool-usage scoring still happen for these rows. The cell hard-fails if `scorers_to_run` is empty for *every* row — that's an upstream assessment-attachment problem, not the augmentation's job to absorb.

- **Parameter-rule relaxation** (cell `b5a9529a`) — pre-processes `expected_tool_calls` and `actual_tool_calls` for `analyze_transactions` calls before they reach the scorer, applying R1–R8 from the rules table below. Some rules drop keys from comparison (R1–R3, R6); others normalize default values so omitted ↔ explicit-default match (R4, R5, R7). Pair-aware R8 is gated on `PERSONAS_WITHOUT_PRODUCT_FILTER` (empty by default). Dropped keys are emitted in `tool_parameter_expected_excused` / `tool_parameter_actual_excused` columns so the report greys them out in the per-call UI. The CSV columns `expected_tool_calls` / `actual_tool_calls` are NOT modified, so the raw data stays auditable. Flip `RELAXED_PARAMETER_RULES = False` to A/B-compare.

## Dataset issue we worked around (handled by augmentation above)

Cases `smoke-076, 077, 078, 079, 080, 081, 083` are off-topic / chit-chat / ethical questions:
- `expected_tool_calls: []` (no tool needed)
- `actual_tool_calls: []` (correctly handled by the agent)
- but the test case still listed `tool_input_parameters` in `expectations.scorers`

The augmentation drops `tool_parameter` from these rows so they no longer report as "scored with no result". Effect on the headline: params pass 32.9% → 37.5%.

## Relaxation rules (analyze_transactions only)

| Rule   | Description                                                                                        | Reasoning                                                         |
| ------ | -------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **R1** | Ignore `size` on both sides.                                                                       | Upstream pagination; not user-visible.                            |
| **R2** | Ignore `sort_by` when `group_by` is `group_none` or omitted.                                       | With no groups there is nothing to sort.                          |
| **R3** | Ignore `sort` when `visualization_type` is `SUMMARY` or omitted.                                   | Aggregation is order-independent.                                 |
| **R4** | Treat `group_by` omitted ↔ `'group_none'` as equivalent.                                           | `group_none` is the documented default.                           |
| **R5** | Treat `products_filter_mode` omitted ↔ `'PFM_SETTINGS'` as equivalent.                             | `PFM_SETTINGS` is the documented default per the tool docstring.  |
| **R6** | Ignore `limit=1000` when `visualization_type=SUMMARY` on either side.                              | A cap larger than any persona's total transactions is never hit.  |
| **R7** | Treat `visualization_type` omitted ↔ `'SUMMARY'` as equivalent.                                    | `SUMMARY` is the documented default.                              |
| **R8** | When the persona has no product-filter preference, expected `PFM_SETTINGS` accepts actual `'ALL'`. | Identical data returned for those personas. Off until persona set is populated. |

**Ordering matters** within the relaxation: R7/R4 normalize defaults before R3/R6/R2 read them, and R8 (pair-aware) runs before per-call normalization so the rewritten value passes through R5.
