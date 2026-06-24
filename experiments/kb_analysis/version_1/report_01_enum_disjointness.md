# Report 1 — ENUM fragment disjointness & answer determinism

**KB:** `version_1/feature-and-product-knowledge.local.csv` — 198 fragments, Slovak, one row per `knowledgeId` ("ENUM fragment").
**Question this report answers:** *To what extent are the ENUM fragments disjoint? Where they overlap, how much — and would that overlap make the agent pick a **different** fragment (and give a different answer) across N runs for the **same** question?*

**What was measured, and on what.** Per your instruction, similarity was computed on the **full `description`** (the text the agent matches the question against), across **all 198** fragments, keyed by `knowledgeId`. Two corpus-derived methods were used because HuggingFace sentence-transformers are still blocked by the corporate TLS proxy (`CERTIFICATE_VERIFY_FAILED`, same as version_0):

| Method | Captures | Role |
|---|---|---|
| char 3–5-gram **TF-IDF** cosine | surface / lexical overlap (copy-paste, shared phrasing) | finds literal near-duplicates |
| word 1–2 **TF-IDF + Truncated SVD (LSA)** cosine | topical / co-occurrence overlap (same subject, different words) | finds "same topic, different wording" — the case a semantic LLM router is most exposed to |

The cosine numbers are only a **candidate filter**. Every candidate pair (cosine ≥ 0.75; 40 pairs) was then read in full by an **LLM judge** (Claude subagents, the same harness version_0 used — no external API) that decided whether the two fragments are genuinely **substitutable for the same realistic user question**, including underspecified questions where the user does not name the exact product. That substitutability — not the cosine — is the determinism verdict.

> **Determinism, defined here:** for the same question, the agent (a) selects the same fragment(s) and (b) returns the same answer, across repeated runs. Two fragments are a *determinism risk* when a plausible question could legitimately be answered from **either**, and nothing in the text deterministically forces one choice.

---

## 0. How fragment selection actually works (verified in `ai-orchestrator`)

The Hey George knowledge tool is a LangGraph pipeline — **`retrieve → prune → rerank`** (`apps/orchestrator/src/orchestrator/adapters/knowledge/`):

1. **Retrieve** — the question + query variations fan out to the KB service (`POST /admin/knowledge-base/query`, hybrid **RRF** dense+sparse, 30 hits/query); results are merged and de-duplicated with provenance.
2. **Prune** — diversity round-robin + `expected_facets` matched against the **`description`** + score-based rescue.
3. **Rerank** — a **GPT-5.1 LLM reranker** (`LLM_RERANK_MODEL = GPT_5_1`) is shown, per candidate, only `id · group · best_rank · query_coverage · best_score · matched_queries · `**`description`** and returns `selected_ids`.
4. **Output** — the agent receives `id: description` for each selected fragment.

Two facts from the code drive everything below:

- **Selection is decided on `description`.** The runtime fragment model is `{id, summary, description}` (`domain/knowledge/models.py`) — **`notInScope` is not even loaded**, and `summary` is loaded but **never used** in prune/rerank/output. The disambiguation fields authored in the KB therefore do **not** influence selection today. *(Caveat: what the KB service indexes/embeds server-side is out of this repo — it may use `summary`; but every step the orchestrator controls uses `description`.)*
- **The GPT-5.1 reranker is the determinism linchpin.** When two candidates have near-identical descriptions, the only things left to separate them are noisy retrieval signals (score/rank/coverage) — so the reranker can return a **different subset or order across runs**. And because it returns a **set**, overlapping fragments can be selected **together**, dropping conflicting facts (e.g. the term-deposit CZK `10 000` vs `12 000` minimum — Report 0 §A9) into a single answer context. This is the concrete mechanism behind "the agent can't choose the same ENUM fragments in N iterations."

---

## 1. Executive summary

1. **The KB is lexically disjoint but topically entangled.** Only **4 / 198** fragments have a near-verbatim twin (char-TF-IDF ≥ 0.80), but **52 / 198 (26%)** have a topical twin at LSA ≥ 0.80 and **89 / 198 (45%)** at LSA ≥ 0.70. Because the agent matches on *meaning*, it is the topical overlap that bites. **Roughly a quarter of the KB has at least one fragment it can be confused with.**

2. **Of the 40 closest pairs, the LLM judge rated 27 HIGH determinism risk, 12 MEDIUM, 1 LOW.** HIGH = a common, realistic question maps to both fragments with no deterministic tie-breaker.

3. **One root cause explains most of it:** a fragment **embeds content that belongs to a dedicated sibling fragment**, while the field meant to draw the boundary — `notInScope` — is **empty, silent on the collision, or self-contradictory**. The agent is then choosing between two fragments that both legitimately answer the question. (And since routing is on `description`, the otherwise-careful `notInScope` text is not even consulted today — see §6.)

4. **Five fragments have outright content/metadata bugs** that break determinism regardless of similarity — including one fragment whose description is about the **wrong product**, and three whose `notInScope` disclaims the very topic the fragment is about. All five are verified against the source text (§3).

5. **Two structural anti-patterns dominate** and are worth fixing as classes, not one-by-one: the **PFM budgeting tangle** (4 fragments all teach "set a budget") and the **loan sibling templates** (refinancing vs "na čokoľvek" pôžička share near-identical limits/rates/instalments/application text with no product tie-breaker).

Artifacts (in `analysis_outputs/`): `disjointness_per_fragment.csv`, `confusable_pairs.csv`, `confusable_pairs_judged.csv` (the judged verdicts), `clusters.csv`, `exact_duplicate_groups.csv`.

---

## 2. How disjoint is the KB overall?

Each fragment's similarity to its **single nearest peer** (`description`):

| | char-TF-IDF (lexical) | LSA (topical) |
|---|---|---|
| mean | 0.51 | 0.65 |
| median | 0.50 | 0.67 |
| 75th pct | 0.63 | 0.82 |
| max | 1.00 | 1.00 |

Fragments having a nearest peer **at or above** a threshold:

| threshold | LSA (topical) | char-TF-IDF (lexical) |
|---|---|---|
| ≥ 0.95 | 6 | 2 |
| ≥ 0.90 | 25 | 2 |
| ≥ 0.85 | 41 | 4 |
| ≥ 0.80 | **52** | 4 |
| ≥ 0.70 | **89** | 19 |

**Reading:** literal duplication is rare (the KB is not full of copy-paste), but *topical* neighbours are everywhere — the median fragment already sits at LSA 0.67 to its nearest peer, and 45% have a peer ≥ 0.70. A retriever that matched on exact words would look fine here; an LLM that matches on meaning will see far more collisions. That gap is precisely the determinism exposure you were worried about.

There are **0 byte-identical descriptions** (so version_0's `STANDARD_FEES≡STANDARD_LIMITS` exact-clone bug is gone), but near-clones remain (§3, pair 0).

---

## 3. Content / metadata bugs that break determinism (verified)

These are not "overlap" in the statistical sense — they are defects where the text itself is wrong or self-contradictory, so the agent cannot be deterministic even with a perfect router. **All five were checked against the raw source.**

| # | Fragment | Defect (verified) | Effect on determinism |
|---|---|---|---|
| 1 | **`GIRO@FOREIGN_ABOUT`** | `knowledgeName`, `summary`, `notInScope` describe a **foreign-currency SPACE account**, but the `description` is a near-verbatim copy of **`GIRO@CHILDREN_ABOUT`** (children's account, ages 6–14, Visa Junior). char-cosine 0.997 / LSA 1.00. | Every foreign-currency-account question is answered with children's-account content. The fragment contradicts its own metadata. **Highest-severity data bug in the KB.** |
| 2 | **`CREATE_STANDING_ORDER`** | Name = *"Nastavenie alebo zmena trvalých príkazov"*; description explains how to **set up** a standing order — but `notInScope` says *"Nerieši sa tu zadanie a zrušenie trvalého príkazu."* (creating is **not** covered here). | If a router ever trusts `notInScope`, it will steer "create a standing order" **away from the only fragment that handles it.** |
| 3 | **`LOAN@CONSOLIDATION_INTEREST_RATE`** | Name = *"…úroková sadzba"*, description = *"Informácie o poplatkoch a úrokovej sadzbe"* — yet `notInScope` says *"Neriešime tu úrokové sadzby…"* (interest rates **not** covered here). | The fragment's own scope field disclaims the fragment's entire purpose. Disambiguator is unusable. |
| 4 | **`START_TRANSFER`** | `notInScope` says *"Neriešia sa tu cezhraničné prevody"* (cross-border **not** covered) — but the description contains a full **SEPA** section and a *"referencia pri zahraničných platbách"* (foreign-payment) section. | Collides with `START_TRANSFER@FOREIGN` on exactly the SEPA/foreign content its `notInScope` claims to exclude — see pair 18. |
| 5 | **`CARD_DEBIT@CAPABILITIES`** | The ePIN is called both *"štvormiestneho kódu"* (4-digit) **and** *"Tento päťmiestny kód"* (5-digit) **within the same description**. (The dedicated `CARD_DEBIT@EPIN` fragment states no length at all.) | Same question ("how long is the ePIN?") can be answered 4 **or** 5 depending on which sentence the agent anchors on — a within-fragment determinism failure. |

**Fixes:** (1) replace `GIRO@FOREIGN_ABOUT.description` with genuine foreign-currency content; (2,3,4) rewrite the three `notInScope` fields so they exclude *neighbouring* topics, not their own; (5) settle the ePIN on one length (the judge believed 5-digit is intended — please confirm) and remove the contradictory sentence.

---

## 4. The determinism-risk landscape (judged pairs)

40 candidate pairs judged. Cross-family = the two fragments belong to **different product families** (different `@`-prefix or unrelated id), which is the more dangerous case because nothing about the product names hints they overlap.

| | HIGH | MEDIUM | LOW |
|---|---|---|---|
| **cross-family** | 12 | 8 | 0 |
| **same-family** | 15 | 4 | 1 |
| **total** | **27** | **12** | **1** |

By relationship type:

| relationship | HIGH | MED | LOW | meaning |
|---|---|---|---|---|
| `scope_overlap` | 17 | 5 | 0 | one fragment bleeds into another's dedicated job |
| `sibling_products` | 5 | 3 | 0 | same template, different product (loans, card types) |
| `about_vs_application_split` | 1 | 4 | 1 | "what it is" vs "how to apply" repeat the same body |
| `content_bug` | 2 | 0 | 0 | pairs 0 & 9 above |
| `near_duplicate` | 2 | 0 | 0 | effectively the same fragment twice |

---

## 5. HIGH-risk pairs, grouped

### 5.1 Cross-family scope bleed (the worst — 12 pairs)
A fragment contains a full how-to that *belongs* to another family's dedicated fragment, and neither `notInScope` mentions the other.

**The PFM budgeting tangle — 4 fragments, one job.** `PFM_SPENDING_BUDGET`, `PFM_CATEGORY_BUDGETS`, `PFM_SUBCATEGORY_BUDGET` and even `PRODUCTDETAIL_CASHFLOW_GRAPH` all walk the user through *"Nastaviť rozpočet"* (pairs 3, 5, 17, 27, 28, 37). The cashflow-graph fragment — nominally a spending **chart** — embeds a complete budget-creation procedure. `notInScope` is **empty** on most of them.
> *"Ako si nastavím rozpočet na kategóriu výdavkov, napr. Bývanie?"* → at least 3 fragments answer this. The agent has no rule to pick one.

**Other cross-family HIGH pairs:**

| Pair | The collision | One realistic question that flips |
|---|---|---|
| `PORTFOLIO@APPLICATION` ↔ `SEARCH_SECURITIES` | PORTFOLIO@APPLICATION (meant: open/close the securities account) embeds a full *"ako kúpiť akcie, ETF, dlhopisy"* walkthrough that duplicates SEARCH_SECURITIES | *"Ako vyhľadám a kúpim akcie alebo ETF cez George?"* |
| `INSTANT_TRANSFER` ↔ `DISPLAY_LIMITS` | INSTANT_TRANSFER carries the full *"kde upravíte limit na okamžité platby"* section that is DISPLAY_LIMITS' job | *"Ako si zmením denný limit pre okamžité platby?"* |
| `SHOW_STANDING_SWEEP_ORDERS_V2` ↔ `CREATE_STANDING_ORDER` | view/change vs create — but names both say *"nastavenie/zmena"*, and CREATE's `notInScope` is broken (bug #2) | *"Ako si nastavím alebo upravím trvalý príkaz?"* |
| `SAVING@DISPOSING_PERSON` ↔ `ACCOUNT@DISPOSING_PERSON` | ACCOUNT version says *"disponent môže byť určený aj na sporiacom účte"*, claiming the savings case the other fragment owns | *"Ako pridám disponenta a aké má práva?"* |
| `APP_NOTIFICATIONS` ↔ `GEORGE@DAILY_BALANCE` | both set up balance/movement notifications; APP_NOTIFICATIONS bleeds into the web channel the other claims | *"Ako si nastavím notifikácie o pohyboch na účte?"* |
| `CALL_BRANCH_AUTHORISED` ↔ `CALL_PHONE_AUTHORISED_V2` | near-duplicate: identical *Kontakty → Klientske centrum → Zavolať* flow | *"Ako cez Georgea zavolám do banky?"* |
| `CARD_DEBIT@REQUEST_CARD` ↔ `VIRTUAL_CARDS` | REQUEST_CARD openly includes *"ako založiť virtuálnu kartu"*, VIRTUAL_CARDS' sole topic | *"Ako si založím virtuálnu kartu?"* |
| `CARD_CREDIT@VALIDITY_PERIOD` ↔ `CARD_DEBIT@VALIDITY_PERIOD` | same validity/auto-reissue text; debit version's `notInScope` is empty | *"Aká je platnosť karty a ako funguje obnova?"* (user rarely says debit/credit) |
| `LOCK_CARD_PERM` ↔ `REQUEST_CARD` | the loss/theft *"zablokovať a prevydať"* flow is fully in both | *"Stratil som kartu — ako ju zablokujem a dostanem novú?"* |

### 5.2 Loan sibling templates (5 HIGH pairs)
`LOAN@CONSOLIDATION_*` (refinancing) and `LOAN@UNSECURED_*` ("pôžička na čokoľvek") share near-identical text for **INSTALMENTS** (7), **INTEREST_RATE** (10), **LIMITS** (13, identical 300–40 000 €) and **APPLICATION** (26). `notInScope` never names the *other* loan. Any question that doesn't say *refinancovanie/konsolidácia* has no deterministic target:
> *"Aká je úroková sadzba na pôžičke?"* · *"Koľko si môžem maximálne požičať?"* · *"Aký je poplatok za predčasné splatenie?"*

### 5.3 Same-family scope splits & near-duplicates (the rest)
`SAVING@DEPOSIT_APPLICATION`↔`SAVING@DEPOSIT_ABOUT` (1, same founding walkthrough on both sides), `SAVING@KIDS_ABOUT`↔`SAVING@KIDS_DISPOSING_PERSON` (14), `START_TRANSFER`↔`START_TRANSFER@FOREIGN` (18, + bug #4), `CARD_DEBIT@CAPABILITIES`↔`CARD_DEBIT@EPIN` (19, + bug #5), `CHANGE_CARD_LIMITS@BLOCKONLINE`↔`CHANGE_CARD_LIMITS` (12), `DEBT@SOLUTION`↔`DEBT@RESTRUCTURING` (29), and `OUTGOING_TRX_CLAIM@INCOMING`↔`OUTGOING_TRX_CLAIM` (30, effectively one fragment twice).

Full per-pair detail (Slovak example question, what currently separates them, and the concrete fix) is in **`analysis_outputs/confusable_pairs_judged.csv`**.

---

## 6. The systemic root cause & why it matters for routing

Across nearly every HIGH pair the judge wrote the same sentence in different words: **the two fragments overlap on a task, and `notInScope` is empty or silent exactly at the collision.** Three compounding facts:

1. **Descriptions are not single-responsibility.** "About" fragments re-teach the application steps; a chart fragment teaches budgeting; an account-opening fragment teaches buying securities. The same procedure lives in 2–4 places, so 2–4 fragments legitimately match the same question.
2. **`notInScope` — the natural tie-breaker — is under-used or wrong.** Many overlapping fragments have empty `notInScope`; several disclaim their *own* topic (§3). It almost never says *"this specific task is handled in fragment X."*
3. **Selection is on `description` only** (verified — §0). The GPT-5.1 reranker never sees `summary` (loaded but unused) or `notInScope` (not loaded at all), so the carefully-written disambiguation fields **do not influence selection today** — the one signal that could separate siblings is invisible to the reranker.

**Implication:** there are two independent levers, and the biggest determinism win comes from pulling both:
- **(A) De-duplicate the descriptions** so each task lives in exactly one fragment (kills the overlap at the source — this is the **only** lever that helps the pipeline *as it runs today*, since the reranker sees nothing but `description` + scores).
- **(B) Surface `summary`/`notInScope` into the reranker prompt *and* fix the broken `notInScope` fields** (Report 0 §1) so siblings get an explicit boundary the reranker can key on (e.g. the word *refinancovanie* vs *na čokoľvek*). This is an orchestrator change, not just a KB edit — today fixing `notInScope` text alone changes nothing at selection time.

---

## 7. Recommendations (prioritised)

**P0 — data bugs (do first; cheap, high impact):**
1. Fix `GIRO@FOREIGN_ABOUT.description` (wrong product entirely).
2. Fix the three self-contradicting `notInScope` fields: `CREATE_STANDING_ORDER`, `LOAN@CONSOLIDATION_INTEREST_RATE`, `START_TRANSFER`.
3. Resolve the 4-vs-5-digit ePIN contradiction in `CARD_DEBIT@CAPABILITIES`.

**P1 — kill the two systemic tangles:**
4. **PFM budgets:** pick one owner per task — overall vs per-category vs per-subcategory — remove the budget how-to from `PRODUCTDETAIL_CASHFLOW_GRAPH`, and add mutual `notInScope` cross-references. (Consider merging into one budgeting fragment with category/subcategory subsections.)
5. **Loan siblings:** factor the shared limits/rates/instalments/application text or add reciprocal `notInScope` lines naming the other product; ensure each fragment's first line names the exact product.

**P2 — single-responsibility pass on the remaining HIGH scope-bleed pairs** (§5.1/5.3): move each embedded how-to (buy-securities, virtual-card creation, instant-payment limits, ePIN setup, standing-order create-vs-view, disponent) into its dedicated fragment and leave a one-line pointer behind.

**P3 — reranker change (architectural):** surface `notInScope` (and `summary`) into the GPT-5.1 reranker prompt — and load `notInScope` into the fragment model first, since it isn't fetched today (§0). Without this, the scope boundaries authored in the KB never reach the selector, and sibling products with identical bodies remain a coin-flip even after the P0–P2 KB edits. (A cheaper interim guard: have the reranker prefer the single best fragment rather than returning overlapping siblings as a set.)

**P4 — regression guard:** re-run `analyze_disjointness.py` after edits; treat any new LSA ≥ 0.90 cross-family pair or any `notInScope` that matches its own description as a CI failure for the KB.

---

## 8. How to read the artifacts

| File | One row per | Key columns |
|---|---|---|
| `disjointness_per_fragment.csv` | knowledgeId | `max_tfidf`, `max_lsa`, `nearest_*`, `n_lsa_ge_0.80`, `top5_lsa` |
| `confusable_pairs.csv` | candidate pair (cosine ≥ 0.45/0.70) | `tfidf`, `lsa`, `max_sim`, `same_family` |
| `confusable_pairs_judged.csv` | the 40 judged pairs | `determinism_risk`, `substitutable`, `relationship`, `content_bug`, `example_question_sk`, `disambiguator`, `fix` |
| `clusters.csv` | overlap cluster | `members`, `families`, `cross_family` |

> Caveat: cosine ran on corpus-derived TF-IDF/LSA (no semantic embedding model available in this environment). The LSA numbers are a *lower bound* on what a production semantic router will conflate — if HuggingFace (or an Azure embedding endpoint) becomes reachable, re-run with a multilingual model to refine the long tail. The LLM-judge verdicts in §3–§5 do not depend on the cosine method and stand on their own.
