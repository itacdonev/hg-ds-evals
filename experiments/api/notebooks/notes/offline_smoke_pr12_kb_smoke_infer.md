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

Scoped to `tool == "analyze_transactions"` — other tools pass through untouched. R1–R7 are per-call (applied independently to each expected and actual call). R8 is pair-aware (looks at expected and actual together).

| Rule   | Description                                                                                                                                          | Reasoning                                                                                                                 |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **R1** | Drop `size` from comparison (both sides). Key listed as excused.                                                                                     | Upstream pagination; not user-visible.                                                                                    |
| **R2** | Drop `sort_by` when `group_by` is omitted or `'group_none'` (treating omitted as `group_none`).                                                      | With no groups there is nothing to sort.                                                                                  |
| **R3** | Drop `sort` when `visualization_type` is omitted or `'SUMMARY'` (treating omitted as `SUMMARY`).                                                     | Aggregation is order-independent.                                                                                         |
| **R4** | Normalize omitted `group_by` → `'group_none'`. Key stays in scope and is still scored against the normalized value.                                  | `group_none` is the documented default.                                                                                   |
| **R5** | Normalize omitted `products_filter_mode` → `'PFM_SETTINGS'`. Key stays in scope. Implication: an actual call omitting the key now auto-matches an expected `PFM_SETTINGS` without needing R8. | `PFM_SETTINGS` is the documented tool default.                                                                            |
| **R6** | Drop `limit` when value is `1000` AND (post-R7) `visualization_type` is `SUMMARY`. Applied per side independently.                                   | A cap larger than any persona's total transactions is never hit.                                                          |
| **R7** | Normalize omitted `visualization_type` → `'SUMMARY'`. Key stays in scope.                                                                            | `SUMMARY` is the documented default.                                                                                      |
| **R8** | Pair-aware, persona-conditional: when `eval_persona ∈ PERSONAS_WITHOUT_PRODUCT_FILTER` AND the expected call at position `i` has `products_filter_mode='PFM_SETTINGS'`, rewrite the actual call at position `i`'s explicit `'ALL'` → `'PFM_SETTINGS'` (positional pairing). Expected is never modified. | Identical data returned for those personas. `PERSONAS_WITHOUT_PRODUCT_FILTER` defaults to empty → R8 effectively off until the eval team populates it. |

**Excused vs normalized:** R1–R3 and R6 *drop* keys (key listed in `tool_parameter_expected_excused` / `_actual_excused`; report greys them out). R4, R5, R7 *normalize* the value but keep the key in scope for scoring. R8 *rewrites* one actual value pre-comparison (not surfaced as excused).

**Order of application** (matches `_compute_relaxation` → `_relax_one`):
1. **R8** first on the whole actual list (pair-aware positional rewrite) — runs before R5 so the rewritten `PFM_SETTINGS` is still present when R5 checks for omission.
2. Per call, capture `vis_for_conditionals` (treat omitted as `SUMMARY`) and `gb_for_conditionals` (treat omitted as `group_none`) BEFORE any mutation — this is what R2/R3/R6 read.
3. **R4** materializes `group_by` default.
4. **R5** materializes `products_filter_mode` default.
5. **R1** drops `size`.
6. **R2** drops `sort_by` (conditional on `gb_for_conditionals`).
7. **R3** drops `sort` (conditional on `vis_for_conditionals`).
8. **R6** drops `limit=1000` (conditional on `vis_for_conditionals`).
9. **R7** materializes `visualization_type` default (R3/R6 already consulted the post-normalization view via `vis_for_conditionals`, so doing this last is purely cosmetic on the resulting dict).

**A/B toggle:** `RELAXED_PARAMETER_RULES = False` bypasses everything; the scorer sees raw `expected_tool_calls` / `actual_tool_calls`. Per-rule firing counts are emitted in the `RELAX_FIRED` summary printed after scoring.
