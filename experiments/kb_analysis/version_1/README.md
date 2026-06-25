# KB analysis — version_1 (SK feature-and-product knowledge)

Analysis of the updated Slovak KB (`feature-and-product-knowledge.local.csv`, **198 fragments**) and the
updated test set (`SLSP_test_cases.csv`, **604 rows / 199 topics**), focused on three questions you raised:

1. **How disjoint are the ENUM fragments?** Where they overlap, could the agent pick a *different* fragment
   (and give a different answer) across N runs for the *same* question? → **[Report 1](report_01_enum_disjointness.md)**
2. **What additional, high-probability user questions should the test set include** — especially ones the KB
   *cannot* fully answer? → **[Report 2](report_02_test_set_expansion.md)** + **[`new_test_cases.csv`](new_test_cases.csv)**
3. **For each ENUM, is anything missing / under-explained** enough to confuse the agent or make it
   non-deterministic? → **[Report 3](report_03_enum_completeness.md)**

A fourth report collects the **hard data bugs and coverage problems** that surfaced across all three:
**[Report 0](report_00_kb_data_quality.md)**.

## What changed since version_0
version_0 worked on two **separate EN/SK** CSVs and was mostly a **translation-quality** audit + `description`
disjointness. version_1 is **one Slovak KB** with a richer schema (`summary`, `notInScope`, `knowledgeType`,
tags…), so the focus shifted to **selection determinism** and **content completeness**. The version_0
`GIRO@STANDARD_FEES ≡ STANDARD_LIMITS` duplicate bug has a sequel here (now in the *metadata* — Report 0 §A2),
and the "ABOUT vs APPLICATION" overlap pattern persists.

## The reports

| Report | What's in it |
|---|---|
| **[Report 0 — data quality & coverage](report_00_kb_data_quality.md)** | 12 source-verified content bugs (wrong-product text, self-contradictory `notInScope`, 4-vs-5-digit ePIN, Czech-residency rule, truncated text, a visible "TODO: add link"), a Tier-B list of flagged contradictions, the fees/rates/limits deferral-to-price-list design (and its three real gaps: missing links, illustrative numbers, unverified link targets), NOTVALIDATE status, and test↔KB ID drift. |
| **[Report 1 — ENUM disjointness & determinism](report_01_enum_disjointness.md)** | How topically entangled the KB is (by **meaning**, 157/198 have a near-twin ≥0.70, ~half ≥0.75), **two** LLM-judge passes over 76 pairs (40 lexical + 36 semantic-only) → **32 HIGH** determinism risk, the systemic tangles (PFM budgets, loan siblings, the cross-product `DISPOSING_PERSON` family), and prioritised fixes. |
| **[Report 2 — test-set expansion](report_02_test_set_expansion.md)** | 398 proposed questions (**334 gap-exposing**), the missing-information areas by product family, the highest-priority gap questions, and how to use the CSV. |
| **[Report 3 — per-ENUM completeness](report_03_enum_completeness.md)** | The full per-fragment catalogue: 331 issues across 179 fragments (61 HIGH), with the 51 HIGH-impact fragments written up in detail and the rest tabulated. |

## Headline findings
- **The KB is lexically disjoint but topically entangled — and the meaning-based view (now run) makes the entanglement larger.** Only 4/198 fragments are near-verbatim twins and 52/198 have a topical (LSA) near-twin ≥0.80, but on the **semantic** model — the one that mirrors how the agent matches meaning — **157/198 (79%) have a near-twin ≥0.70, 93/198 (47%) ≥0.75, 46/198 ≥0.80**. So by meaning, ~half the KB has a close look-alike, not a quarter; the lexical numbers were a genuine lower bound.
- **The selection mechanism (verified in `ai-orchestrator`):** a LangGraph `retrieve → prune → rerank` pipeline where a **GPT-5.1 LLM reranker** picks fragments seeing only `id · retrieval-scores · `**`description`**. The runtime fragment model is `{id, summary, description}` — **`notInScope` isn't loaded and `summary` is unused at selection** ([Report 1 §0](report_01_enum_disjointness.md)). Both the similarity and the determinism judging therefore use **name + `description` only**; re-judging the 40 lexical pairs strictly on that basis (excluding `notInScope`/`summary`) changed only 2 of 40 verdicts, which offset — totals stay **27/12/1** (lexical) and **32 HIGH** combined.
- **Root cause of most confusability:** fragments **embed content owned by a sibling fragment**, so when their `description`s overlap the reranker has only noisy retrieval scores to break the tie — and it returns a *set*, so overlapping siblings can come back **together** and drop conflicting facts into one answer. Fixing `notInScope` text alone changes nothing today; the live levers are **de-duplicating descriptions** and **surfacing `summary`/`notInScope` into the reranker**.
- **~Half of all content issues (165/331) are a missing concrete number** (fee / limit / rate). By design these defer to the official price list, shown to the customer as a **clickable link** — so this is *not* a wholesale gap. The real residual issues are narrower: a few ENUM fragments cite the price list with **no link** (64 findings on link-less ENUM fragments; 1 explicit case), some embed an **illustrative figure** the agent may quote as real, and the **linked documents' correctness is unverified**.
- **12 verified hard bugs** need fixing first (Report 0 §1) — cheap, mechanical, high impact.

## Deliverable for editing the KB / test set
- **[`new_test_cases.csv`](new_test_cases.csv)** — 398 proposed rows in the exact `SLSP_test_cases.csv` schema (SK+EN), gap cases unanswered with the missing area in `comment`. Append-ready; `PROPOSED-NNNN` ids keep them distinct from validated cases. *(The original test set and KB CSV were not modified.)*

## Method & caveats
- **Disjointness** computed on the full `description` (per your instruction), all 198 fragments, via three methods: char-n-gram **TF-IDF** + **LSA** + a **semantic** multilingual sentence-embedding cosine (`paraphrase-multilingual-MiniLM-L12-v2`). The semantic method was TLS-blocked in version_0 and the first cut of version_1; that block is now **resolved** (user-space, no admin — `truststore` to use the macOS keychain + `HF_HUB_DISABLE_XET=1` to avoid the Xet downloader's own TLS stack; see Report 1 §8). The semantic run confirmed the lexical numbers were a lower bound and surfaced 203 lexically-invisible candidate pairs. The **LLM-judge** verdicts (substitutability, content bugs, completeness) used Claude subagents over this harness — no external API — and do not depend on any cosine method.
- All quantitative artifacts are in **`analysis_outputs/`** (see below). Reports 2 & 3 are generated from those CSVs, so they stay faithful to the underlying analysis.

## Reproduce
```bash
cd experiments/kb_analysis/version_1
# semantic method needs these (one-off; works behind the Erste proxy, no admin):
../../../.venv/bin/python -m pip install sentence-transformers truststore
../../../.venv/bin/python analyze_disjointness.py     # tfidf + lsa + SEMANTIC cosine, clusters, confusable + semantic-only pairs
# (LLM-judge of the 40 lexical pairs, the 36 semantic-only pairs, and the 16-batch content
#  pass were run via Claude subagents; structured outputs cached under analysis_outputs/.
#  build_desconly_judge_batch.py / build_semonly_judge_batch.py rebuild the judging batches.
#  Judging uses name+description ONLY — notInScope/summary are not read by the live agent.)
../../../.venv/bin/python merge_gap_results.py        # consolidate the content pass -> CSVs
../../../.venv/bin/python gen_report_02.py            # regenerate Report 2 from the CSVs
../../../.venv/bin/python gen_report_03.py            # regenerate Report 3 from the CSVs
```

## `analysis_outputs/` artifacts
| File | Contents |
|---|---|
| `disjointness_per_fragment.csv` | per-fragment nearest-neighbour stats (max TF-IDF/LSA/**semantic**, top-5 each) |
| `confusable_pairs.csv` | candidate overlapping pairs (lexical- **or semantic**-filtered; `semantic_only` flag) |
| `semantic_only_pairs.csv` | the 203 pairs the semantic model flags but the lexical methods miss |
| `confusable_pairs_judged_desc_only.csv` | **canonical** — the 40 lexical pairs judged on name+description only (risk, prior_risk, substitutability, fix) |
| `confusable_pairs_judged.csv` | the original (with-`notInScope`) lexical pass; kept for provenance, superseded by the desc-only file |
| `semantic_only_pairs_judged.csv` | the 36 top semantic-only pairs, same judge schema (5 HIGH / 9 MED / 22 LOW) |
| `clusters.csv` | connected-component overlap clusters (lexical **+ semantic** edges) |
| `gap_completeness_issues.csv` | every per-fragment completeness issue (type, impact, detail, fix/missing area) |
| `gap_new_questions.csv` | every proposed question (answerable_by_kb, missing_area, priority) |
| `gap_per_fragment.csv` | per-fragment issue/question counts + one-line note |
| `gap_batches/` | raw per-batch inputs (`*.json`) and LLM outputs (`*_result.json`) + `INSTRUCTIONS.md` |
