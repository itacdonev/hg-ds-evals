# Report 1 — ENUM fragment disjointness & answer determinism

**KB:** `version_1/feature-and-product-knowledge.local.csv` — 198 fragments, Slovak, one row per `knowledgeId` ("ENUM fragment").
**Question this report answers:** *To what extent are the ENUM fragments disjoint? Where they overlap, how much — and would that overlap make the agent pick a **different** fragment (and give a different answer) across N runs for the **same** question?*

**What was measured, and on what.** Per your instruction, similarity was computed on the **full `description`** (the text the agent matches the question against), across **all 198** fragments, keyed by `knowledgeId`. **Three** methods were used — and the meaning-based one, blocked in the first version of this review, has now been run (the corporate-TLS-proxy blocker was a local fix, no admin needed — see §Method note at the end):

| Method | Captures | Role |
|---|---|---|
| char 3–5-gram **TF-IDF** cosine | surface / lexical overlap (copy-paste, shared phrasing) | finds literal near-duplicates |
| word 1–2 **TF-IDF + Truncated SVD (LSA)** cosine | topical / co-occurrence overlap (same subject, different words) | finds "same topic, different wording" |
| **semantic** — multilingual sentence-embedding cosine (`paraphrase-multilingual-MiniLM-L12-v2`) | meaning-based overlap: two fragments that mean the same thing in **different words** | the closest proxy for what a production semantic router actually conflates — and the lever the lexical methods systematically under-count |

The cosine numbers are only a **candidate filter**. Two LLM-judge passes (Claude subagents, the same harness version_0 used — no external API) then read candidate pairs in full and decided whether the two fragments are genuinely **substitutable for the same realistic user question**, including underspecified questions where the user does not name the exact product:
- the original **40 lexical** candidates (TF-IDF/LSA ≥ 0.75);
- the **top 36 semantic-only** candidates — pairs the semantic model flags but that fall *below* both lexical thresholds, i.e. exactly the overlap the lexical methods miss.

That substitutability — not the cosine — is the determinism verdict.

> **Judged on what the agent actually sees.** Both passes assess substitutability using **only the inputs the reranker receives — the fragment `id`/name + `description`** — never `summary` (loaded but unused) or `notInScope` (not loaded at all); see §0. Similarity (§2) was likewise computed on `description` only. An earlier cut of the lexical pass discussed `notInScope` in its reasoning; **re-judging the 40 lexical pairs strictly on name+description changed only 2 verdicts, and they offset** (`CARD_DEBIT@VALIDITY_PERIOD↔CLOSE_CARD` MED→HIGH; `LOAN@CONSOLIDATION_APPLICATION↔LOAN@UNSECURED_APPLICATION` HIGH→MED), so the totals are unchanged at **27/12/1**. The determinism findings therefore do **not** rest on the unused fields — they are driven by `description` overlap. (Canonical verdicts: `confusable_pairs_judged_desc_only.csv`.)

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

1. **The KB is lexically disjoint but topically entangled — and the meaning-based view makes the entanglement much larger.** Only **4 / 198** fragments have a near-verbatim twin (char-TF-IDF ≥ 0.80), and **52 / 198 (26%)** have a topical twin at LSA ≥ 0.80. But on the **semantic** model — the one that mirrors how the agent matches meaning — **157 / 198 (79%)** have a near-twin ≥ 0.70, **93 / 198 (47%)** ≥ 0.75, and **46 / 198 (23%)** ≥ 0.80; the median fragment sits at **0.74** to its nearest peer. The lexical methods were a genuine lower bound: **by meaning, roughly half the KB has a close look-alike, not a quarter.**

2. **Two judge passes, 76 pairs, 32 HIGH determinism risks.** The original 40 lexical candidates → **27 HIGH** / 12 MED / 1 LOW. The top 36 *semantic-only* candidates (invisible to the lexical methods) → **5 HIGH** / 9 MED / 22 LOW. Combined: **32 HIGH, 21 MEDIUM.** HIGH = a common, realistic question maps to both fragments with no deterministic tie-breaker. The semantic-only HIGHs are a distinct, previously-missed class — most are **cross-product same-topic** collisions (§5.4).

3. **One root cause explains most of it:** a fragment **embeds content that belongs to a dedicated sibling fragment**, while the field meant to draw the boundary — `notInScope` — is **empty, silent on the collision, or self-contradictory**. The agent is then choosing between two fragments that both legitimately answer the question. (And since routing is on `description`, the otherwise-careful `notInScope` text is not even consulted today — see §6.)

4. **Two *live* content bugs + three *latent* scope-field bugs.** Two fragments have errors in the `description` itself, so they break determinism at runtime: `GIRO@FOREIGN_ABOUT` (its description is the **wrong product** — children's account) and `CARD_DEBIT@CAPABILITIES` (ePIN given as both 4- and 5-digit). Three more have a `notInScope` that disclaims their own topic — but since `notInScope` is **not read at runtime** (§0), those are **latent authoring bugs, not live determinism breakers** (§3). All five verified against source.

5. **Two structural anti-patterns dominate** and are worth fixing as classes, not one-by-one: the **PFM budgeting tangle** (4 fragments all teach "set a budget") and the **loan sibling templates** (refinancing vs "na čokoľvek" pôžička share near-identical limits/rates/instalments/application text with no product tie-breaker).

Artifacts (in `analysis_outputs/`): `disjointness_per_fragment.csv`, `confusable_pairs.csv`, `confusable_pairs_judged_desc_only.csv` (canonical verdicts — judged on name+description only), `semantic_only_pairs_judged.csv`, `clusters.csv`.

---

## 2. How disjoint is the KB overall?

Each fragment's similarity to its **single nearest peer** (`description`):

| | char-TF-IDF (lexical) | LSA (topical) | **semantic (meaning)** |
|---|---|---|---|
| mean | 0.51 | 0.65 | **0.75** |
| median | 0.50 | 0.67 | **0.74** |
| 75th pct | 0.63 | 0.82 | 0.79 |
| max | 1.00 | 1.00 | 1.00 |

Fragments having a nearest peer **at or above** a threshold:

| threshold | char-TF-IDF | LSA (topical) | **semantic (meaning)** |
|---|---|---|---|
| ≥ 0.95 | 2 | 6 | 2 |
| ≥ 0.90 | 2 | 25 | 9 |
| ≥ 0.85 | 4 | 41 | 20 |
| ≥ 0.80 | 4 | 52 | **46** |
| ≥ 0.75 | 8 | 67 | **93** |
| ≥ 0.70 | 19 | 88 | **157** |

**Reading.** Literal duplication is rare (the KB is not full of copy-paste). The semantic view is the one that matters for an LLM router, and it has a **high floor**: the 25th-percentile fragment is already at 0.71 to its nearest peer, so **157/198 clear 0.70**. That floor is partly just banking-domain relatedness — 0.70 means "about a neighbouring thing," *not* "interchangeable." So the raw semantic counts are a **candidate signal, not a verdict**: the meaningful-overlap tier is ≥ 0.80 (**46/198 ≈ 23%**) and especially ≥ 0.85 (20), and even those are confirmed only by the LLM judge (§4–§5). The honest headline: on words alone ~a quarter of the KB has a close twin; **by meaning, ~half do at the "close" bar (≥0.75) and almost everything is at least loosely related** — which is exactly why the lexical-only numbers were a lower bound.

There are **0 byte-identical descriptions** (so version_0's `STANDARD_FEES≡STANDARD_LIMITS` exact-clone bug is gone), but near-clones remain (§3A, `GIRO@FOREIGN_ABOUT`).

---

## 3. Content / metadata bugs (verified) — split by whether they bite at runtime

Defects where the text itself is wrong or self-contradictory. **Only bugs in the `description` reach the live agent**: `notInScope` is not loaded at selection (§0), so a broken `notInScope` is a *latent* (authoring) defect today, not a live determinism breaker. All verified against the raw source.

### 3A. Live — the bug is in the `description` (breaks determinism at runtime)

| Fragment | Defect (verified) | Effect |
|---|---|---|
| **`GIRO@FOREIGN_ABOUT`** | name/summary describe a **foreign-currency SPACE account**, but the `description` is a near-verbatim copy of **`GIRO@CHILDREN_ABOUT`** (children's account, ages 6–14, Visa Junior). char-cosine 0.997 / LSA 1.00. | Every foreign-currency-account question is answered with children's-account content. **Highest-severity data bug in the KB.** |
| **`CARD_DEBIT@CAPABILITIES`** | the ePIN is called both *"štvormiestneho kódu"* (4-digit) **and** *"Tento päťmiestny kód"* (5-digit) **within the same description**. | "How long is the ePIN?" can be answered 4 **or** 5 — a within-fragment determinism failure. |

**Fix:** (1) replace `GIRO@FOREIGN_ABOUT.description` with genuine foreign-currency content; (2) settle the ePIN on one length (the judge believed 5-digit is intended — please confirm) and remove the contradictory sentence.

### 3B. Latent — the bug is in `notInScope` (not read at runtime; authoring hygiene)

These fragments have a `notInScope` that disclaims their **own** topic. Because the reranker never sees `notInScope` (§0), **they do not affect live selection today** — but they signal mis-authored scope boundaries and would bite if `notInScope` is ever surfaced to the reranker (§7 P3). Fix for KB hygiene, *not* as a live-determinism P0.

| Fragment | The self-contradiction |
|---|---|
| **`CREATE_STANDING_ORDER`** | description explains how to **set up** a standing order; `notInScope` says *"Nerieši sa tu zadanie a zrušenie trvalého príkazu."* (creating not covered here). |
| **`LOAN@CONSOLIDATION_INTEREST_RATE`** | description = *"Informácie o poplatkoch a úrokovej sadzbe"*; `notInScope` says *"Neriešime tu úrokové sadzby…"* (rates not covered here). |
| **`START_TRANSFER`** | `notInScope` says *"Neriešia sa tu cezhraničné prevody"* — but the description has a full **SEPA** + foreign-payment section (so it also collides with `START_TRANSFER@FOREIGN`, pair 18). |

**Fix:** rewrite each `notInScope` to exclude *neighbouring* topics, not its own — but this changes nothing at selection time unless `notInScope` is also surfaced into the reranker (§7 P3).

---

## 4. The determinism-risk landscape (judged pairs)

### 4.1 Lexical pass — 40 candidates (description-only verdicts)
Cross-family = the two fragments belong to **different product families** (different `@`-prefix or unrelated id), which is the more dangerous case because nothing about the product names hints they overlap. Counts below are the **description-only re-judge** (`confusable_pairs_judged_desc_only.csv`).

| | HIGH | MEDIUM | LOW |
|---|---|---|---|
| **cross-family** | 16 | 3 | 1 |
| **same-family** | 11 | 9 | 0 |
| **total** | **27** | **12** | **1** |

Cross-family pairs carry most of the HIGH risk (16 of 27) — overlap between *different* products is both more dangerous and harder to spot by name.

By relationship type:

| relationship | HIGH | MED | LOW | meaning |
|---|---|---|---|---|
| `scope_overlap` | 14 | 2 | 0 | one fragment bleeds into another's dedicated job |
| `duplicate_content` | 4 | 0 | 0 | the same procedure copied into both descriptions |
| `about_vs_application_split` | 3 | 5 | 0 | "what it is" vs "how to apply" repeat the same body |
| `cross_product_same_topic` | 3 | 0 | 0 | same generic procedure written for different products |
| `sibling_products` | 2 | 4 | 0 | same template, different product (loans, card types) |
| `content_bug` | 1 | 0 | 0 | the `GIRO@FOREIGN_ABOUT` wrong-product mis-paste (§3A) |
| `distinct_intents` | 0 | 1 | 1 | adjacent but separable on description |

### 4.2 Semantic-only pass — 36 candidates the lexical methods missed
These pairs scored **below** both lexical thresholds (TF-IDF < 0.45 **and** LSA < 0.70) yet rank high on meaning — the 203 such pairs are in `semantic_only_pairs.csv`; the top 36 were judged (`semantic_only_pairs_judged.csv`).

| | HIGH | MEDIUM | LOW |
|---|---|---|---|
| **cross-family** | 2 | 5 | 8 |
| **same-family** | 3 | 4 | 14 |
| **total** | **5** | **9** | **22** |

The low HIGH-rate (5/36) is the system working as intended: the judge correctly threw out "semantically adjacent but distinct" pairs the embedding model over-scores — lock vs unlock vs close a card, notifications vs devices, activate vs close — and kept only genuine overlap. The relationship mix differs from the lexical pass: the dominant HIGH type here is **`cross_product_same_topic`** — the *same generic procedure written for different products*, which shares little vocabulary (so LSA missed it) but identical meaning. The standout is the **`*_DISPOSING_PERSON`** family (§5.4).

### 4.3 Combined
**76 pairs judged → 32 HIGH, 21 MEDIUM, 23 LOW.** The semantic pass added **5 HIGH and 9 MEDIUM** risks that were invisible to lexical analysis — a ~19% increase in confirmed HIGH-risk pairs, concentrated in cross-product collisions.

---

## 5. HIGH-risk pairs, grouped

### 5.1 Cross-family scope bleed (the worst class — most of the 16 cross-family HIGH pairs)
A fragment contains a full how-to that *belongs* to another family's dedicated fragment. (Their `notInScope` fields don't mention each other either — but that field isn't read at runtime anyway, §0, so the fix is in the descriptions.)

**The PFM budgeting tangle — 4 fragments, one job.** `PFM_SPENDING_BUDGET`, `PFM_CATEGORY_BUDGETS`, `PFM_SUBCATEGORY_BUDGET` and even `PRODUCTDETAIL_CASHFLOW_GRAPH` all walk the user through *"Nastaviť rozpočet"* (pairs 3, 5, 17, 27, 28, 37). The cashflow-graph fragment — nominally a spending **chart** — embeds a complete budget-creation procedure. `notInScope` is **empty** on most of them.
> *"Ako si nastavím rozpočet na kategóriu výdavkov, napr. Bývanie?"* → at least 3 fragments answer this. The agent has no rule to pick one.

**Other cross-family HIGH pairs:**

| Pair | The collision | One realistic question that flips |
|---|---|---|
| `PORTFOLIO@APPLICATION` ↔ `SEARCH_SECURITIES` | PORTFOLIO@APPLICATION (meant: open/close the securities account) embeds a full *"ako kúpiť akcie, ETF, dlhopisy"* walkthrough that duplicates SEARCH_SECURITIES | *"Ako vyhľadám a kúpim akcie alebo ETF cez George?"* |
| `INSTANT_TRANSFER` ↔ `DISPLAY_LIMITS` | INSTANT_TRANSFER carries the full *"kde upravíte limit na okamžité platby"* section that is DISPLAY_LIMITS' job | *"Ako si zmením denný limit pre okamžité platby?"* |
| `SHOW_STANDING_SWEEP_ORDERS_V2` ↔ `CREATE_STANDING_ORDER` | view/change vs create — but names both say *"nastavenie/zmena"*, and CREATE's `notInScope` is broken (§3B, latent) | *"Ako si nastavím alebo upravím trvalý príkaz?"* |
| `SAVING@DISPOSING_PERSON` ↔ `ACCOUNT@DISPOSING_PERSON` | ACCOUNT version says *"disponent môže byť určený aj na sporiacom účte"*, claiming the savings case the other fragment owns | *"Ako pridám disponenta a aké má práva?"* |
| `APP_NOTIFICATIONS` ↔ `GEORGE@DAILY_BALANCE` | both set up balance/movement notifications; APP_NOTIFICATIONS bleeds into the web channel the other claims | *"Ako si nastavím notifikácie o pohyboch na účte?"* |
| `CALL_BRANCH_AUTHORISED` ↔ `CALL_PHONE_AUTHORISED_V2` | near-duplicate: identical *Kontakty → Klientske centrum → Zavolať* flow | *"Ako cez Georgea zavolám do banky?"* |
| `CARD_DEBIT@REQUEST_CARD` ↔ `VIRTUAL_CARDS` | REQUEST_CARD openly includes *"ako založiť virtuálnu kartu"*, VIRTUAL_CARDS' sole topic | *"Ako si založím virtuálnu kartu?"* |
| `CARD_CREDIT@VALIDITY_PERIOD` ↔ `CARD_DEBIT@VALIDITY_PERIOD` | same validity/auto-reissue text; neither description states debit-vs-credit up front | *"Aká je platnosť karty a ako funguje obnova?"* (user rarely says debit/credit) |
| `LOCK_CARD_PERM` ↔ `REQUEST_CARD` | the loss/theft *"zablokovať a prevydať"* flow is fully in both | *"Stratil som kartu — ako ju zablokujem a dostanem novú?"* |
| `CARD_DEBIT@VALIDITY_PERIOD` ↔ `CLOSE_CARD` | both carry the verbatim *"Zrušiť obnovu / prevydanie karty"* step list (MED→**HIGH** on description-only — `notInScope` was its only separator) | *"Ako zruším automatické prevydanie karty?"* |

### 5.2 Loan sibling templates
`LOAN@CONSOLIDATION_*` (refinancing) and `LOAN@UNSECURED_*` ("pôžička na čokoľvek") share near-identical text for **INSTALMENTS** (7), **INTEREST_RATE** (10) and **LIMITS** (13, identical 300–40 000 €) — all HIGH. The **APPLICATION** pair (26) dropped to **MEDIUM** on the description-only re-judge (each application description does name its product), but still overlaps heavily. None of the descriptions reliably names the product on the rate/limit/instalment fragments, so any question that doesn't say *refinancovanie/konsolidácia* has no deterministic target:
> *"Aká je úroková sadzba na pôžičke?"* · *"Koľko si môžem maximálne požičať?"* · *"Aký je poplatok za predčasné splatenie?"*

### 5.3 Same-family scope splits & near-duplicates (the rest)
`SAVING@DEPOSIT_APPLICATION`↔`SAVING@DEPOSIT_ABOUT` (1, same founding walkthrough on both sides), `SAVING@KIDS_ABOUT`↔`SAVING@KIDS_DISPOSING_PERSON` (14), `START_TRANSFER`↔`START_TRANSFER@FOREIGN` (18, + §3B `notInScope`), `CARD_DEBIT@CAPABILITIES`↔`CARD_DEBIT@EPIN` (19, + §3A ePIN bug), `CHANGE_CARD_LIMITS@BLOCKONLINE`↔`CHANGE_CARD_LIMITS` (12), `DEBT@SOLUTION`↔`DEBT@RESTRUCTURING` (29), and `OUTGOING_TRX_CLAIM@INCOMING`↔`OUTGOING_TRX_CLAIM` (30, effectively one fragment twice).

Full per-pair detail (Slovak example question, what currently separates them, and the concrete fix) is in **`analysis_outputs/confusable_pairs_judged_desc_only.csv`**.

### 5.4 Semantic-only HIGH pairs (lexically invisible — only the meaning model caught them)
These 5 pairs sit **below** both lexical thresholds, so the original pass never saw them. They share little surface vocabulary but nearly identical *meaning* — the failure mode a semantic router is most exposed to and a lexical audit is blindest to.

| Pair | sem | The collision | One realistic question that flips |
|---|---|---|---|
| `SAVING@DEPOSIT_DISPOSING_PERSON` ↔ `GIRO@FOREIGN_DISPOSING_PERSON` | 0.86 | the *disponent* (authorised person) definition + "set it up at a branch" sentence is near-identical across two unrelated products | *"Kto je disponent a ako ho pridám k účtu?"* |
| `CARD_CREDIT@REQUEST_CARD` ↔ `CARD_CREDIT@APPLICATION` | 0.82 | the "how to apply for a credit card" paragraph is duplicated almost verbatim — clearest duplicate-content case in this pass | *"Ako požiadam o kreditnú kartu?"* |
| `GIRO@FOREIGN_ADDITIONAL_SERVICES` ↔ `GIRO@FOREIGN_PAYMENT_CARD` | 0.78 | both answer "can I have a card on a foreign-currency account?" and agree on USD-only | *"Môžem mať platobnú kartu k účtu vedenému v cudzej mene?"* |
| `CREATE_SWEEP_ORDER` ↔ `SHOW_STANDING_SWEEP_ORDERS_V2` | 0.78 | the view/change fragment embeds the *create a sweep order* sections that are the other fragment's whole job | *"Ako si nastavím nadlimitný trvalý príkaz?"* |
| `LOAN@MORTGAGE_SPENDING` ↔ `LOAN@MORTGAGE_POST_SIGNING_CONDITIONS` | 0.77 | both cover pre-drawdown conditions + delivering documents to the branch officer | *"Aké podmienky musím splniť pred čerpaním hypotéky a ako čerpanie prebehne?"* |

**The `*_DISPOSING_PERSON` family is the systemic one.** It is a 5-fragment cross-family cluster (cluster #4: `SAVING@…`, `SAVING@DEPOSIT_…`, `ACCOUNT@…`, `GIRO@STANDARD_…`, `GIRO@FOREIGN_…`), all teaching the *same* generic "who is an authorised person / how to add one" content for different account types, at semantic 0.86–0.94. A user almost never names the product (*"ako pridám disponenta?"*), so the choice among five near-identical fragments is a coin-flip. The lexical pass surfaced one edge of this (`SAVING ↔ ACCOUNT`, §5.1); the semantic pass shows it is a whole family. **Fix:** factor the shared disponent definition into one fragment (or make the first sentence of each name its product explicitly), since the only live selection signal is `description` (§6). The 9 MEDIUM pairs (`semantic_only_pairs_judged.csv`) follow the same pattern at lower intensity — e.g. `SHOW_CARD_PIN ↔ SHOW_CARD_PAN_CVC`, `REGULAR@ORDERS ↔ SECURITIES_ORDERS`.

---

## 6. The systemic root cause & why it matters for routing

Across nearly every HIGH pair the same structural fault recurs: **two fragments' `description`s overlap on a task, and nothing *in the description* deterministically separates them.** Three compounding facts:

1. **Descriptions are not single-responsibility.** "About" fragments re-teach the application steps; a chart fragment teaches budgeting; an account-opening fragment teaches buying securities. The same procedure lives in 2–4 places, so 2–4 fragments legitimately match the same question. **This is the live cause** — and the only one the reranker can see.
2. **Selection is on `description` only** (verified — §0). The GPT-5.1 reranker never sees `summary` (loaded but unused) or `notInScope` (not loaded at all), so even where a fragment *has* a clean scope boundary authored, it cannot influence selection today.
3. **`notInScope` — the field that *should* be the tie-breaker — is therefore latent**, and is itself under-used or wrong (many empty; several disclaim their own topic, §3B). It would help only once surfaced into the reranker (lever B).

**Implication:** there are two independent levers, and the biggest determinism win comes from pulling both:
- **(A) De-duplicate the descriptions** so each task lives in exactly one fragment (kills the overlap at the source — this is the **only** lever that helps the pipeline *as it runs today*, since the reranker sees nothing but `description` + scores).
- **(B) Surface `summary`/`notInScope` into the reranker prompt *and* fix the broken `notInScope` fields** (Report 0 §1) so siblings get an explicit boundary the reranker can key on (e.g. the word *refinancovanie* vs *na čokoľvek*). This is an orchestrator change, not just a KB edit — today fixing `notInScope` text alone changes nothing at selection time.

---

## 7. Recommendations (prioritised)

**P0 — live data bugs (do first; cheap, high impact):** these are in the `description`, so they bite at runtime.
1. Fix `GIRO@FOREIGN_ABOUT.description` (wrong product entirely).
2. Resolve the 4-vs-5-digit ePIN contradiction in `CARD_DEBIT@CAPABILITIES`.

> *Not P0:* the three self-contradictory `notInScope` fields (`CREATE_STANDING_ORDER`, `LOAN@CONSOLIDATION_INTEREST_RATE`, `START_TRANSFER`, §3B). `notInScope` is not read at selection, so fixing them changes **nothing live today** — do them as authoring hygiene together with P3.

**P1 — kill the two systemic tangles (description-level — these *do* help today):**
3. **PFM budgets:** pick one owner per task — overall vs per-category vs per-subcategory — and remove the budget how-to from `PRODUCTDETAIL_CASHFLOW_GRAPH`. (Consider merging into one budgeting fragment with subsections.) Make each fragment's first description line name its exact scope.
4. **Loan siblings:** factor out the shared limits/rates/instalments text, and ensure **each fragment's first description line names the exact product** (*refinancovanie/konsolidácia* vs *pôžička na čokoľvek*) — that is the only tie-breaker the reranker can actually see. (`notInScope` cross-references help only once surfaced — P3.)
5. **The `*_DISPOSING_PERSON` family (§5.4):** factor the shared disponent definition into one fragment, or prefix each with its product, so *"ako pridám disponenta?"* has a single target.

**P2 — single-responsibility pass on the remaining HIGH scope-bleed pairs** (§5.1/5.3): move each embedded how-to (buy-securities, virtual-card creation, instant-payment limits, ePIN setup, standing-order create-vs-view, disponent) into its dedicated fragment and leave a one-line pointer behind.

**P3 — reranker change (architectural):** surface `notInScope` (and `summary`) into the GPT-5.1 reranker prompt — and load `notInScope` into the fragment model first, since it isn't fetched today (§0). Without this, the scope boundaries authored in the KB never reach the selector, and sibling products with identical bodies remain a coin-flip even after the P0–P2 KB edits. (A cheaper interim guard: have the reranker prefer the single best fragment rather than returning overlapping siblings as a set.)

**P4 — regression guard:** re-run `analyze_disjointness.py` after edits; treat any new LSA ≥ 0.90 cross-family pair or any `notInScope` that matches its own description as a CI failure for the KB.

---

## 8. How to read the artifacts

| File | One row per | Key columns |
|---|---|---|
| `disjointness_per_fragment.csv` | knowledgeId | `max_tfidf`, `max_lsa`, **`max_semantic`**, `nearest_*`, `n_semantic_ge_0.70/0.80/0.85`, `top5_semantic` |
| `confusable_pairs.csv` | candidate pair (lexical ≥ 0.45/0.70 **or semantic ≥ 0.70**) | `tfidf`, `lsa`, **`semantic`**, `max_sim`, `same_family`, **`lexical_candidate`**, **`semantic_only`** |
| `semantic_only_pairs.csv` | the **203** pairs semantic flags but lexical misses | sorted by `semantic`; the long tail for follow-up judging |
| **`confusable_pairs_judged_desc_only.csv`** | the 40 lexical pairs **re-judged on name+description only** (canonical) | `determinism_risk`, `prior_risk`, `substitutable`, `relationship`, `content_bug`, `example_question_sk`, `disambiguator`, `fix` |
| `confusable_pairs_judged.csv` | the original (with-`notInScope`) lexical pass | kept for provenance; superseded by the desc-only file |
| `semantic_only_pairs_judged.csv` | the 36 semantic-only judged pairs (already description-based) | same schema |
| `clusters.csv` | overlap cluster (edge: tfidf ≥ .60 / lsa ≥ .85 / **sem ≥ .80**) | `members`, `families`, `cross_family` |

> **Method note — the semantic model, and how the block was lifted.** The meaning-based method now runs with `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`, ~470 MB, 50+ languages incl. Slovak). The HuggingFace TLS block (`CERTIFICATE_VERIFY_FAILED`) was **not** a firewall block and needed **no admin**: the Erste "Proxy Certification Authority" CA is already trusted by macOS (curl works), Python just wasn't consulting the keychain. Two user-space fixes, baked into `analyze_disjointness.py`: (1) `truststore.inject_into_ssl()` so Python's `ssl` uses the keychain; (2) `HF_HUB_DISABLE_XET=1` because the new Xet downloader (`hf_xet`, a Rust binary) has its own TLS stack that ignores the keychain and hangs on the inspecting proxy — disabling it routes downloads back through Python's `ssl`.
>
> **What is still a bound.** This is a *generic* multilingual STS model, not the production KB-service embedder (which is out of this repo). It validated Slovak meaning well (paraphrase 0.91, near-duplicate 0.72, unrelated < 0.2) but its cross-lingual axis is weak — irrelevant here since the KB is all Slovak. The production router may conflate somewhat differently; the LLM-judge verdicts in §3–§5 do not depend on any cosine method and stand on their own.
