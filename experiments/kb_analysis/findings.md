# KB analysis — findings

Inputs (only `kb.knowledgeId` and `kb.description` were used; `kb.summary` was dropped):

- `input/KB_GAI_SK_EN_2026-04-20_14h16_phase_1_2.csv` — 163 entries (English)
- `input/KB_GAI_SK_SK_2026-04-20_14h16_phase_1_2.csv` — 164 entries (Slovak)

ID alignment:

- Common to both languages: **163**
- Present only in SK: **1** — `CREATE_STANDING_ORDER` (no English counterpart)
- Present only in EN: **0**

> Methodology note. The precomputed embeddings shipped in the `.jsonl` siblings of these CSVs were built from `kb.summary`, not `kb.description`, and were not used for any computation in this analysis. Disjointness was computed with two corpus-derived methods: (a) character-3-5-gram **TF-IDF** cosine, and (b) **LSA** (word-1-2 TF-IDF + Truncated SVD, 100 components). Translation correctness was judged by Claude (sub-agents over the same harness), reading each EN/SK pair and using only the two texts. No outside knowledge was consulted.

## Headline numbers

| Question | Answer |
|---|---|
| Translation **ok** | 110 / 163 |
| Translation **minor_issues** | 45 / 163 |
| Translation **major_issues** | **8 / 163** |
| Mean faithfulness (1–5) | **4.69** |
| Mean completeness (1–5) | **4.66** |
| Pairs with at least one issue | 58 / 163 |
| EN ids near-duplicate by LSA (max_sim_other ≥ 0.95) | 6 / 163 |
| SK ids near-duplicate by LSA (max_sim_other ≥ 0.95) | 10 / 164 |
| EN ids topically overlapping (LSA ≥ 0.80) | 40 / 163 |
| SK ids topically overlapping (LSA ≥ 0.80) | 40 / 164 |

---

## 1. Translation correctness

### 1.1 The 8 major issues (most actionable)

Each of these EN translations has a problem the LLM judge classified as material. All quotes are paraphrases of the model's findings — see `kb_findings.csv` (column `judge_summary`) and `translation_judgments.jsonl` for the verbatim verdicts.

| knowledgeId | Issue |
|---|---|
| **PORTFOLIO@APPLICATION** | The EN field is **not translated at all** — it contains the Slovak source text verbatim. Confirmed manually (`description_en[:300]` starts with `**Informácie, ako začať investovať…**`). |
| **GIRO@STANDARD_LIMITS** | Both EN and SK contain the **fees** description, byte-identical to `GIRO@STANDARD_FEES`. The "limits" topic implied by the ID is absent from the KB. (See finding #3 below.) |
| **GEORGE@PROBLEM** | EN drops the in-person branch channel that exists in SK and is truncated mid-word at the end of the field. |
| **CARD_DEBIT@REQUEST_CARD** | **Value mismatch**: EN says virtual-card minimum age **18**; SK says **viac ako 15 rokov** (over 15). Wrong fact in the EN KB. |
| **CARD_CREDIT@BENEFITS** | **Adds an unsupported claim**: EN states the credit limit "starts from €300 for VISA Classic" — this figure is not in the SK source. EN also drops the "denied boarding due to overbooking" detail from the Air Refund clause. |
| **GIRO@CHILDREN_STUDENT** | EN drops the entire "turning 18" section and omits payment-method/credit-card details in the "My Reward" conditions. |
| **INSURANCE@HOME_APPLICATION** | Several omissions plus value mismatches in contact details and procedure steps. |
| **START_TRANSFER@FOREIGN** | EN's SEPA country list omits a substantial portion of the SK source list — incomplete coverage. |

### 1.2 What kinds of issues recur (across all 163 pairs)

Issue type counts from the LLM judge (issues per pair are independent — one pair can contribute several):

| Issue type | Count | What it means |
|---|---|---|
| `missing` | 33 | A fact or section in SK has no counterpart in EN |
| `structure` | 17 | Bullets / headers / line-breaks dropped; paragraph runs together; bold lost |
| `value_mismatch` | 14 | A specific number, age, fee, percentage, or contact detail differs |
| `added` | 10 | EN states something not present in SK (worst case: hallucinated values) |
| `terminology` | 10 | A product name or term rendered inconsistently across the EN corpus |
| `ambiguous` | 7 | EN wording introduces ambiguity that SK does not have |

### 1.3 Notes worth fixing in bulk

- **9 pairs** have `consistent_terminology=False` — recurring offender is product-name rendering (e.g. *Rýchle čerpanie* sometimes "Quick Cash", sometimes "Fast Disbursement").
- Several pairs have **untranslated Slovak fragments** at the end of the EN cell (e.g. `ATM@WITHDRAWAL` ends with `'sadzobníka poplatkov..'` and `'webovej stránke banky.'` — looks like truncation during a translation pass).
- **URL alignment**: 11 pairs have a different URL count between EN and SK; in several cases the EN drops the explicit hyperlink and renders the link target as bold text only.
- Number alignment is OK overall (Jaccard of normalized numeric tokens has median 1.0), but the value mismatches we caught are concentrated in fees, ages, and phone/contact numbers — high-impact for a banking RAG.

### 1.4 ID coverage gap

`CREATE_STANDING_ORDER` exists only in the Slovak KB. If the EN agentic RAG is expected to handle "create standing order" queries, this fragment is missing on the EN side.

---

## 2. Within-language disjointness

### 2.1 What was measured

Disjointness is operationalised as **how similar each fragment's `kb.description` is to the most similar OTHER fragment in the same language**. A fragment is "disjoint" if its nearest neighbour is far away; it is "overlapping" if a neighbour is close.

Two complementary similarity methods are computed per language:

| Method | What it captures | Why it's here |
|---|---|---|
| `tfidf` — char-3-5-gram TF-IDF cosine | Surface lexical overlap (shared phrasings, near-duplicate text) | Robust to morphology; flags copy-paste duplicates first |
| `lsa` — word-1-2 TF-IDF + Truncated SVD (100 dims) | Topical / co-occurrence overlap (paraphrases on the same topic) | Picks up "same topic, different wording" cases that TF-IDF misses |

Both are computed **only from this corpus** — no external embedding model, so the analysis is fully reproducible without network access. (A third method, multilingual sentence-transformer embeddings, was attempted but blocked by the corporate TLS proxy on Hugging Face. LSA is the right substitute given the constraint.)

For each `knowledgeId` the findings file contains, per language and per method:

- `max_sim_other_<method>_<lang>` — cosine to its nearest other fragment (the headline disjointness number)
- `nearest_id_<method>_<lang>` — which fragment that is
- `mean_top3_sim_<method>_<lang>` — robustness check vs. a single noisy neighbour
- `n_neighbors_ge_{0.50, 0.70, 0.85}_<method>_<lang>` — counts at three thresholds (how dense is the neighbourhood)
- `top5_neighbors_<method>_<lang>` — the five nearest, with cosines (for triage)

Threshold flags (descriptive, not strict cutoffs):

- `flag_lexical_near_duplicate` — TF-IDF max_sim ≥ 0.85
- `flag_lexical_overlap` — TF-IDF max_sim ≥ 0.60
- `flag_topical_near_duplicate` — LSA max_sim ≥ 0.95
- `flag_topical_overlap` — LSA max_sim ≥ 0.80

### 2.2 Distribution

`max_sim_other` distributions (a fragment's similarity to its nearest peer):

|  | EN tfidf | SK tfidf | EN lsa | SK lsa |
|---|---|---|---|---|
| mean | 0.56 | 0.50 | 0.65 | 0.60 |
| median | 0.59 | 0.49 | 0.65 | 0.62 |
| 75% | 0.64 | 0.62 | 0.80 | 0.80 |
| max | 1.00 | 1.00 | 1.00 | 1.00 |

Reading: most fragments have a peer in the 0.5–0.8 range (LSA), which means topical neighbours are common but not dense. The long right tail is what hurts retrieval — see the next section.

### 2.3 The most overlapping pairs (need attention)

These are the IDs where retrieval is most likely to confuse two fragments. Both methods agree on the top of the list.

**True near-duplicates (TF-IDF cosine ≈ 1.0):**

- `GIRO@STANDARD_FEES` ↔ `GIRO@STANDARD_LIMITS` — **identical text** (KB data bug, see finding #3 below).

**Heavy topical overlap (LSA ≥ 0.95) — same product family, similar prose:**

| Pair | EN LSA | SK LSA | What's going on |
|---|---|---|---|
| `SAVING@DEPOSIT_ABOUT` ↔ `SAVING@DEPOSIT_APPLICATION` | 0.986 | 0.982 | "About" and "how to apply" descriptions overlap heavily on product attributes. |
| `SAVING@KIDS_ABOUT` ↔ `SAVING@KIDS_DISPOSING_PERSON` | 0.958 | (lower) | Disponent rules described in both. |
| `INSURANCE@PERSONAL_ITEMS_AND_CARDS_PROCEDURES` ↔ `INSURANCE@PERSONAL_ITEMS_AND_CARDS_COVERAGE` | 0.929 | similar | "How to claim" and "what is covered" share a lot of preamble. |
| `ROUND_UP_SAVING_APPLICATION` ↔ `SAVING@ROUND_UP_SAVING` | 0.923 | 0.956 | Two IDs for the same feature. |
| `INSURANCE@TRAVELING_LONGTERM_ABOUT` ↔ `INSURANCE@TRAVELING_LONGTERM_APPLICATION` | 0.913 | 0.930 | Same overlap pattern as the deposit pair. |
| `LOAN@UNSECURED_INTEREST_RATES` ↔ `LOAN@CONSOLIDATION_INTEREST_RATE` | 0.907 | (high) | Two loan products share interest-rate text. |
| `GIRO@FOREIGN_ABOUT` ↔ `GIRO@FOREIGN_APPLICATION` | (high) | 0.961 | About-vs-Application split, again. |
| `PORTFOLIO@APPLICATION` ↔ `SEARCH_SECURITIES` | (high) | 0.955 | Investment account / securities search overlap (also note: PORTFOLIO@APPLICATION's EN cell is untranslated, see translation finding). |
| `LOAN@UNSECURED_INSTALMENTS` ↔ `LOAN@CONSOLIDATION_INSTALMENTS` | (high) | 0.927 | Mirror of the loan interest-rate case. |

**Lexical near-duplicates that aren't already in the topical list:**

- `EASY_ACCESS` ↔ `TOKEN_MANAGEMENT` — TF-IDF 0.746 EN / 0.697 SK. Distinct topics on paper but heavy phrase reuse — worth checking.
- `ACCOUNT_STATEMENT_LIST` ↔ `SAVING@ACCOUNT_STATEMENTS` — TF-IDF 0.738 EN. The savings-account variant is largely a copy of the general one.
- `CARD_DEBIT@VALIDITY_PERIOD` ↔ `CARD_CREDIT@VALIDITY_PERIOD` — TF-IDF 0.71 EN. Same prose template across card products.

### 2.4 Recurring overlap pattern: the "ABOUT vs APPLICATION" split

The KB has a systematic structural decision: many products have both a `<PRODUCT>_ABOUT` (or just the product code) and `<PRODUCT>_APPLICATION` fragment. Across the highest-overlap pairs, these splits dominate. They are problematic for retrieval because:

- The "about" text describes attributes the user often searches for (limits, eligibility), and the "application" text describes the same attributes again as part of the application context.
- LSA cosines of 0.93–0.98 mean the agentic retriever is essentially flipping a coin between them.

A practical mitigation: shrink the redundant attribute repetition in the `_APPLICATION` fragments, or merge the two when the only differences are the application channel sentence.

---

## 3. KB data bug: `GIRO@STANDARD_FEES` ≡ `GIRO@STANDARD_LIMITS`

`GIRO@STANDARD_FEES` and `GIRO@STANDARD_LIMITS` have **byte-identical** `kb.description` text in **both** the EN and SK CSVs. The text in both rows describes the *opening fee*, *maintenance fee*, age-based pricing, and the "My Reward" free-account conditions.

Implications for the agentic RAG:

- `GIRO@STANDARD_LIMITS` does not contain any limits-related content. Whatever the agentic RAG returns for "limits" queries on a SPACE account will match a fees fragment.
- For *any* retrieval query that should hit `GIRO@STANDARD_LIMITS`, the system is choosing between two perfect duplicates — the retriever cannot disambiguate, and the answer it returns is on the wrong topic.
- The actual standard-account *limits* description is missing from the KB.

EN sample (both IDs return this exact text):

```
**Fee for opening a SPACE account**
The fee for opening a SPACE account is not charged. Opening the account is free.
**Fee for maintaining a SPACE account**
The account maintenance fee varies based on the client's age:
15 -- 26 years: account free of charge
27 -- 61 years: fee €7
62 years and older: discounted price €3.5
...
```

**Action**: replace `GIRO@STANDARD_LIMITS.kb.description` with the actual standard-account limits content (daily/monthly transaction limits, ATM withdrawal limits, etc.), or remove the ID if the duplication is intentional.

---

## 4. How to read `kb_findings.csv`

One row per `knowledgeId`, including the ID-in-only-one-language case. Columns:

- `knowledgeId`, `description_en`, `description_sk` — the inputs
- `present_in_en`, `present_in_sk`, `flag_only_in_one_lang` — alignment
- `char_len_*`, `word_count_*`, `len_ratio_en_over_sk`, `bullet_count_*`, `header_count_*`, `url_count_*`, `percent_count_*`, `numbers_*`, `numbers_jaccard`, `numbers_only_en`, `numbers_only_sk` — structural alignment of the two descriptions
- `max_sim_other_<method>_<lang>`, `nearest_id_<method>_<lang>`, `mean_top3_sim_<method>_<lang>`, `n_neighbors_ge_<thr>_<method>_<lang>`, `top5_neighbors_<method>_<lang>` — disjointness
- `judge_faithfulness`, `judge_completeness`, `judge_consistent_terminology`, `judge_verdict`, `judge_summary`, `judge_issue_count`, `judge_issue_types`, `judge_issues` — translation verdicts

Threshold flags (T/F columns; convenience for filtering):

- Translation: `flag_only_in_one_lang`, `flag_len_ratio_off`, `flag_numbers_mismatch`, `flag_bullet_mismatch`, `flag_header_mismatch`, `flag_url_mismatch`
- Disjointness: `flag_lexical_overlap_<lang>`, `flag_lexical_near_duplicate_<lang>`, `flag_topical_overlap_<lang>`, `flag_topical_near_duplicate_<lang>`

### Suggested triage queries

```
# All major translation issues
df[df.judge_verdict == "major_issues"]

# Translation issues that involve a wrong value (highest risk)
df[df.judge_issue_types.str.contains("value_mismatch", na=False)]

# Fragments that are topically near-indistinguishable from another
df[df.flag_topical_near_duplicate_en | df.flag_topical_near_duplicate_sk]

# Lexical near-duplicates (likely true copy/paste)
df[df.flag_lexical_near_duplicate_en | df.flag_lexical_near_duplicate_sk]
```

---

## 5. Companion artifacts in this folder

- `kb_clean.jsonl` — `{knowledgeId, description_en, description_sk}` (no embeddings, no summary)
- `kb_findings.csv` — one row per `knowledgeId` with every analysis column (see §4)
- `translation_judgments.jsonl` — full LLM verdict per id including the issue list
- `nearest_neighbors_tfidf_{en,sk}.csv` — top-5 lexical neighbours per id
- `nearest_neighbors_lsa_{en,sk}.csv` — top-5 topical neighbours per id
- `analyze_kb.py` — disjointness + structural-translation feature script
- `judge_translation.py` — direct-API LLM judge (not used in this run; left for reference)
- `merge_judgments.py` — merges sub-agent verdicts into `kb_findings.csv`
- `judge_batches/` — batched inputs and per-batch verdict JSONL files

## 6. Recommended next steps

1. **Fix the 8 major translation issues** listed in §1.1 — at minimum the `PORTFOLIO@APPLICATION` untranslated cell and the `CARD_DEBIT@REQUEST_CARD` age-15-vs-18 value mismatch.
2. **Resolve the `GIRO@STANDARD_FEES`/`GIRO@STANDARD_LIMITS` duplicate** by writing the actual limits content (or deleting the ID).
3. **Address the structural ABOUT/APPLICATION redundancy** for at least the savings-deposit, insurance-traveling-longterm, and loan unsecured/consolidation pairs — these are the highest LSA-cosine clusters and are the most likely retrieval-confusion sources.
4. **Add the missing EN counterpart** for `CREATE_STANDING_ORDER` if EN queries are expected.
5. **Standardise product-name rendering** in EN — the 9 `consistent_terminology=False` cases plus the *Rýchle čerpanie* example suggest a glossary pass would help retrieval and answer quality.
6. **Re-embed the KB on `kb.description`** (the precomputed embeddings on disk were built from `kb.summary`). If embeddings are produced fresh, re-running this analysis with a real multilingual sentence-transformer model would refine the disjointness numbers further.
