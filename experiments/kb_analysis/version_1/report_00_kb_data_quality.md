# Report 0 — KB data quality, content bugs & coverage

**KB:** `version_1/feature-and-product-knowledge.local.csv` — 198 fragments, Slovak.
**Test set:** `version_1/SLSP_test_cases.csv` — 604 rows, 199 distinct topic IDs.

This report collects the **hard defects** — text that is wrong, self-contradictory, stale, or unfinished — plus **coverage** problems between the test set and the KB. These break determinism *regardless of retrieval quality* (a perfect router still returns wrong/contradictory text), so they are the cheapest, highest-value fixes. Overlap/confusability between fragments is in [Report 1](report_01_enum_disjointness.md); the exhaustive per-fragment completeness catalogue is in [Report 3](report_03_enum_completeness.md).

> **Severity of "content contradiction" for determinism:** when one fragment states a fact two ways (or two fragments disagree), the agent answers the *same* question differently depending on which sentence/fragment it anchors on. That is the literal failure mode you asked about — "the agent can't give the same answer in N iterations."

---

## 1. Tier A — content bugs verified against the source text

Each row below was checked directly in the CSV (not just flagged by a model). Quotes are from `description`/`summary`/`notInScope`.

| # | Fragment | Defect (verified) | Why it breaks determinism |
|---|---|---|---|
| A1 | **`GIRO@FOREIGN_ABOUT`** | `description` is a near-verbatim copy of **`GIRO@CHILDREN_ABOUT`** (children's SPACE account, ages 6–14, Visa Junior). Name/summary/notInScope correctly describe a **foreign-currency** account. | Every foreign-currency-account question is answered with children's-account text. Worst defect in the KB. |
| A2 | **`GIRO@STANDARD_LIMITS`** | `knowledgeName` (*"SPACE účet - poplatky"*) **and** `summary` are byte-identical to **`GIRO@STANDARD_FEES`** — i.e. the *fees* metadata sits on the *limits* fragment. (This is version_0's `FEES≡LIMITS` bug resurfacing in the metadata.) | A router or label that reads name/summary will treat the limits fragment as a fees duplicate. |
| A3 | **`CREATE_STANDING_ORDER`** | `notInScope` = *"Nerieši sa tu zadanie a zrušenie trvalého príkazu."* — but the fragment's whole job is **setting up** a standing order. | If routing ever trusts `notInScope`, it steers "create a standing order" away from the only fragment that handles it. |
| A4 | **`LOAN@CONSOLIDATION_INTEREST_RATE`** | `notInScope` = *"Neriešime tu úrokové sadzby…"* on a fragment named *"…úroková sadzba"* whose body is *"Informácie o poplatkoch a úrokovej sadzbe"*. | The fragment disclaims its own purpose. |
| A5 | **`START_TRANSFER`** | `notInScope` = *"Neriešia sa tu cezhraničné prevody"* — yet the `description` contains a full **SEPA** section and a *"referencia pri zahraničných platbách"* section. | Collides with `START_TRANSFER@FOREIGN` on the exact content its `notInScope` claims to exclude. |
| A6 | **`CARD_DEBIT@CAPABILITIES`** | The ePIN is called both *"štvormiestneho kódu"* (4-digit) and *"Tento päťmiestny kód"* (5-digit) **in the same description**. (`CARD_DEBIT@EPIN` states no length at all.) | "How long is the ePIN?" → 4 or 5 depending on the sentence the agent anchors on. |
| A7 | **`INVEST@COMPLIANCE`** | Eligibility reads *"fyzické osoby s trvalým pobytom na území **ČR**"* (permanent residence in the **Czech Republic**) — in a Slovenská sporiteľňa KB. | Wrong-country eligibility rule; the agent will mis-answer "can I invest as a Slovak resident?". |
| A8 | **`CARD_CREDIT@ABOUT`** vs **`CARD_CREDIT@REQUEST_CARD`** | Age stated as *"viac ako 18 rokov"* (no upper bound) in one fragment and *"18 až 69 rokov"* in the other. | "Up to what age can I get a credit card?" answers differently by retrieval. |
| A9 | **`SAVING@DEPOSIT_APPLICATION`** vs **`SAVING@DEPOSIT_ABOUT`** | Term-deposit CZK minimum is *"10 000 CZK"* in one and *"12 000 CZK"* in the other (same product). | Numeric answer flips by which sibling is retrieved. |
| A10 | **`LOAN@MORTGAGE_INTEREST_RATES`** | A sentence is truncated mid-word: *"…(výnimkou je neplnenie **pode**"* (should be *"podmienok"*). | Incomplete/garbled answer; signals a broken ingest. |
| A11 | **`GEORGE@WHAT_YOU_DO`** | Ships an editorial placeholder to users: *"Viac sa dozviete **[tu]. - doplniť link** na web (ak bude existovať info na webe)."* | The agent will emit an internal "TODO: add link" to a customer. |
| A12 | **`CARD_CREDIT@APPLICATION_STATUS`** | Duplicated text artifact *"…oň žiadať. oň žiadať."* (flagged; visible stutter in the body). | Cosmetic, but indicates uncontrolled copy-paste in this family. |

**All of A1–A11 are confirmed in the source.** A12 is a visible artifact reported by the content pass.

> **Runtime caveat on the `notInScope` defects (A3–A5).** Per the orchestrator code (see [Report 1 §0](report_01_enum_disjointness.md)), the runtime fragment model is `{id, summary, description}` — **`notInScope` is not loaded and `summary` is unused at selection time.** So the self-contradicting `notInScope` fields do **not** misroute the live agent *today*; they are authoring-hygiene + latent risks (they bite the moment `notInScope` is wired into retrieval/rerank, and may already affect server-side indexing in the KB service, which is out of repo). The defects that **do** hit the live agent are the ones in the `description`/metadata that the pipeline actually reads — A1, A2, A6–A12 — plus A5's *description-level* SEPA overlap (the bad `notInScope` is secondary there).

---

## 2. Tier B — contradictions & scope mismatches flagged by the content pass

The per-fragment content analysis (Report 3) found **36 internal contradictions** and **53 scope-mismatch** issues. The highest-impact ones not already in Tier A are below. These are model-flagged with quoted evidence in `analysis_outputs/gap_completeness_issues.csv` — **recommend a quick human spot-check before editing.**

| Fragment | Type | What's contradictory / mismatched |
|---|---|---|
| `ACCOUNT@DISPOSING_PERSON` | internal_contradiction | Body says disposition rights are **branch-only**, then the closing line says they can be set up **in George without a branch**. |
| `SELL_ORDER` ↔ `PORTFOLIO@ABOUT` | internal_contradiction | Mutual-fund redemption: SELL_ORDER says sell in-app via *"Predať"*; PORTFOLIO@ABOUT says redemption is **branch-only**. |
| `CARD_CREDIT@VALIDITY_PERIOD` | internal_contradiction | Reissue-cancellation rule stated twice with conflicting wording; also conflicts with `CARD_CREDIT@CANCELLATION` (one month vs two months before expiry). |
| `OUTGOING_TRX_CLAIM` family | internal_contradiction | Card-claim window differs across siblings: **15/120 days** vs none vs **30 days/6 months**. |
| `PFM_BUDGETS` | internal_contradiction | Locates the budget list in two different places (*"v hornej časti úvodnej obrazovky"* vs behind the list icon under *"Výdavky"*). |
| `GIRO@FOREIGN_PAYMENT_CARD` | ambiguous | Dual-account (EUR+USD) insufficient-funds rule is self-contradictory ("enough on both" vs "pays from whichever has funds"). |
| `TOKEN_MANAGEMENT` | internal_contradiction | Android face-login: description lists face login for iOS only; summary says *"Face Unlock na Androide"*. |
| `CARD_DEBIT@NEW_CARD_ISSUES` | ambiguous | Two conflicting "treat as undelivered" thresholds (10 working days vs wait until the 15th). |
| `DRAWDOWN` | scope_mismatch | `notInScope` excludes housing loans, but the body explains them. |
| `ATM@TRANSACTION_CLAIM` | scope_mismatch | Name = *"Reklamácia platby kartou"* (card-payment claim) but body covers only ATM-withdrawal claims. |
| `FRAUD@ABOUT` | scope_mismatch | Summary promises account/online-banking blocking, money recovery, fraud definitions; body only says *"call us + block card"*. |
| `START_TRANSFER@FOREIGN` | outdated | Still lists **RUB** as a supported cross-border currency (sanctions-era stale). |
| `ATM@PAYMENT` ≈ `ATM@PAYMENT_TERMINAL` | near-duplicate | Two fragments with the same *"no payment via ATM, use George"* content. |
| `PROFILE` | ambiguous | Alias allowed-character list is garbled (dash artifact) → non-deterministic punctuation answers. |
| `GEORGE_DEVICES` | ambiguous | The "lost phone / no old device" activation flow is grammatically garbled. |

---

## 3. Systemic gap: exact figures are deferred to documents the agent can't read

**This is the single biggest content problem.** Of 331 completeness issues, **164 concern a missing concrete number** (fee, limit, rate, threshold, price), and **at least 50 explicitly defer the answer to the external *Sadzobník*, a PDF, or a web link** that the agent cannot open. The result: the most common, highest-intent customer questions are unanswerable or get a "see the price list" non-answer.

Representative (each is a real, high-frequency question with **no number in the KB**):
- **ATM:** daily/per-withdrawal cash limit in EUR; fee for other-bank ATMs in SK and abroad (`ATM@WITHDRAWAL`); AML deposit threshold (`ATM@DEPOSIT`).
- **Cards:** debit-card fees (`CARD_DEBIT@FEES` — pure pointer, zero figures); foreign-transaction / FX margin (`CARD_DEBIT@ABOUT`); replacement-card fee (`CARD_CREDIT@REQUEST_CARD`); card limit standard/max values (`CHANGE_CARD_LIMITS`).
- **Savings/deposits:** base savings rate as a number (`SAVING@INTEREST_RATES_AND_LIMITS` gives only the +0,50 % bonus); current term-deposit rates (`SAVING@DEPOSIT_INTEREST_RATES` → external PDF + a hypothetical 1,5 % the agent may quote as real); branch cash deposit/withdrawal fee (`SAVING@FEES`).
- **Credit card:** interest rate / APR for revolving debt (`CARD_CREDIT@REPAY`).
- **Current accounts:** SPACE monthly maintenance fee, 2nd-card price, foreign-account opening/FX fees (`GIRO@*`).
- **Investing/insurance:** per-trade commission (`PORTFOLIO@FEES_AND_LIMITS`); life-insurance price & sum insured (`INSURANCE@LIFE_ABOUT`).

**Why it matters for determinism specifically:** where the KB gives a *hypothetical example* number (e.g. the 1,5 % term-deposit illustration), the agent may present it as the real rate in some runs and hedge in others — non-deterministic *and* potentially wrong. **Recommendation:** decide a policy — either ingest the Sadzobník figures into the KB (best), or have fragments return a single deterministic "I can't quote the current fee, here's where to find it" response so the answer is at least stable.

---

## 4. `NOTVALIDATE` fragments (24)

24 fragments carry `knowledgeType = NOTVALIDATE`. Per your instruction they were analyzed alongside the rest, but several of the defects above sit on NOTVALIDATE fragments (e.g. `GIRO@CHILDREN_APPLICATION`, `DEBT@RESTRUCTURING`). **Please confirm whether NOTVALIDATE content is served in production.** If it is, it needs the same QA bar; if not, it should be excluded from retrieval so it can't be selected. Either way the status should be explicit, because today these fragments are indistinguishable from validated ones at retrieval time.

---

## 5. Coverage: test set ↔ KB

**Test set shape:** 604 rows over 199 topic IDs (mean ≈ 3 questions/topic; 1 topic with a single question; max 7). Questions are bilingual (SK + EN); **`expected_answer_EN` is empty for all 604 rows** (only `expected_answer_SK` is populated) — flag this if EN gold answers were intended. 30 rows are marked `golden_standard_question? = Y`; 16 are `DESCOPE = fixed` and cluster on exactly the rates/limits/fees topics that §3 shows the KB can't answer — likely descoped for that reason.

**ID drift — 5 test topic IDs have no KB fragment, and 4 of them are spelling mismatches of an (untested) KB fragment:**

| Test-set ID (no KB match) | Almost certainly means (KB ID) | Note |
|---|---|---|
| `PFM_CATEGORY_BUDGET` | `PFM_CATEGORY_BUDGETS` | singular/plural |
| `PRODUCTS_SETTINGS` | `PRODUCT_SETTINGS` | plural |
| `REGULAR_ORDERS` | `REGULAR@ORDERS` | `_` vs `@` |
| `ROUND_UP_SAVINGS` | `SAVING@ROUND_UP_SAVING` | prefix + plural |
| `GEORGE@TRANSACTION_WITHOUT_CONFIRMATION` | **— no clear KB match —** | genuinely unmapped; either missing KB content or a distinct intent (closest: `TRANSACTION_CLAIM`) |

The first four are the same concepts as the **4 KB fragments that no test case references** (`PFM_CATEGORY_BUDGETS`, `REGULAR@ORDERS`, `SAVING@ROUND_UP_SAVING`, `PRODUCTDETAIL_TRANSACTION_SEARCH@OVERVIEW`). So aligning the spelling closes both gaps at once. **Action:** normalize the topic IDs in the test set to the KB spelling (or vice versa) and add a CI check that every `knowledge_topic_ID` exists in the KB; investigate `GEORGE@TRANSACTION_WITHOUT_CONFIRMATION` separately.

---

## 6. Recommended fix order

1. **Tier A bugs (§1)** — small, mechanical, high impact. Start with A1 (wrong product text) and A2–A5 (wrong/contradictory metadata that can mislead routing).
2. **ID drift (§5)** — one-line renames; unblocks 4 untested fragments and prevents silent test-set misrouting.
3. **Decide the figures policy (§3)** — the biggest answerability lever; pick "ingest the numbers" vs "stable deferral response."
4. **Tier B spot-check & fix (§2)** — verify the flagged contradictions, then correct.
5. **Confirm NOTVALIDATE status (§4)** and exclude from retrieval if not production-ready.

Full evidence: `analysis_outputs/gap_completeness_issues.csv` (every issue, with quoted detail and impact).
