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
| **[Report 0 — data quality & coverage](report_00_kb_data_quality.md)** | 12 source-verified content bugs (wrong-product text, self-contradictory `notInScope`, 4-vs-5-digit ePIN, Czech-residency rule, truncated text, a visible "TODO: add link"), a Tier-B list of flagged contradictions, the systemic "exact figures live in an external doc the agent can't read" gap, NOTVALIDATE status, and test↔KB ID drift. |
| **[Report 1 — ENUM disjointness & determinism](report_01_enum_disjointness.md)** | How topically entangled the KB is (45% of fragments have a near-twin), the 40 closest pairs LLM-judged for genuine substitutability (**27 HIGH** determinism risk), the systemic tangles (PFM budgets, loan siblings, ABOUT/APPLICATION), and prioritised fixes. |
| **[Report 2 — test-set expansion](report_02_test_set_expansion.md)** | 398 proposed questions (**334 gap-exposing**), the missing-information areas by product family, the highest-priority gap questions, and how to use the CSV. |
| **[Report 3 — per-ENUM completeness](report_03_enum_completeness.md)** | The full per-fragment catalogue: 331 issues across 179 fragments (61 HIGH), with the 51 HIGH-impact fragments written up in detail and the rest tabulated. |

## Headline findings
- **The KB is lexically disjoint but topically entangled** — only 4/198 fragments are near-verbatim twins, but **52/198 have a topical near-twin** and **89/198 a peer at LSA ≥ 0.70**. Because the agent matches on meaning, that topical overlap is the determinism exposure.
- **The selection mechanism (verified in `ai-orchestrator`):** a LangGraph `retrieve → prune → rerank` pipeline where a **GPT-5.1 LLM reranker** picks fragments seeing only `id · retrieval-scores · `**`description`**. The runtime fragment model is `{id, summary, description}` — **`notInScope` isn't loaded and `summary` is unused at selection** ([Report 1 §0](report_01_enum_disjointness.md)).
- **Root cause of most confusability:** fragments **embed content owned by a sibling fragment**, so when their `description`s overlap the reranker has only noisy retrieval scores to break the tie — and it returns a *set*, so overlapping siblings can come back **together** and drop conflicting facts into one answer. Fixing `notInScope` text alone changes nothing today; the live levers are **de-duplicating descriptions** and **surfacing `summary`/`notInScope` into the reranker**.
- **~Half of all content issues (164/331) are a missing concrete number** (fee / limit / rate), most deferred to the external *Sadzobník* the agent cannot read — so the highest-intent customer questions are unanswerable or non-deterministic.
- **12 verified hard bugs** need fixing first (Report 0 §1) — cheap, mechanical, high impact.

## Deliverable for editing the KB / test set
- **[`new_test_cases.csv`](new_test_cases.csv)** — 398 proposed rows in the exact `SLSP_test_cases.csv` schema (SK+EN), gap cases unanswered with the missing area in `comment`. Append-ready; `PROPOSED-NNNN` ids keep them distinct from validated cases. *(The original test set and KB CSV were not modified.)*

## Method & caveats
- **Disjointness** computed on the full `description` (per your instruction), all 198 fragments, via char-n-gram **TF-IDF** + **LSA** cosine. HuggingFace sentence-transformers are **TLS-blocked** in this environment (same as version_0), so a semantic embedding model could not be used — the LSA numbers are a *lower bound* on what a production semantic router will conflate. The **LLM-judge** verdicts (substitutability, content bugs, completeness) used Claude subagents over this harness — no external API — and do not depend on the cosine method.
- All quantitative artifacts are in **`analysis_outputs/`** (see below). Reports 2 & 3 are generated from those CSVs, so they stay faithful to the underlying analysis.

## Reproduce
```bash
cd experiments/kb_analysis/version_1
../../../.venv/bin/python analyze_disjointness.py     # cosine + clusters + confusable pairs
# (LLM-judge of confusable pairs and the 16-batch content pass were run via Claude subagents;
#  their structured outputs are cached under analysis_outputs/)
../../../.venv/bin/python merge_gap_results.py        # consolidate the content pass -> CSVs
../../../.venv/bin/python gen_report_02.py            # regenerate Report 2 from the CSVs
../../../.venv/bin/python gen_report_03.py            # regenerate Report 3 from the CSVs
```

## `analysis_outputs/` artifacts
| File | Contents |
|---|---|
| `disjointness_per_fragment.csv` | per-fragment nearest-neighbour stats (max TF-IDF/LSA, top-5) |
| `confusable_pairs.csv` | candidate overlapping pairs (cosine-filtered) |
| `confusable_pairs_judged.csv` | the 40 closest pairs with LLM determinism verdicts (risk, substitutability, fix) |
| `clusters.csv` | connected-component overlap clusters |
| `gap_completeness_issues.csv` | every per-fragment completeness issue (type, impact, detail, fix/missing area) |
| `gap_new_questions.csv` | every proposed question (answerable_by_kb, missing_area, priority) |
| `gap_per_fragment.csv` | per-fragment issue/question counts + one-line note |
| `gap_batches/` | raw per-batch inputs (`*.json`) and LLM outputs (`*_result.json`) + `INSTRUCTIONS.md` |
