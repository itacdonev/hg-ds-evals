# SK — Knowledge Base Review: Findings

*June 2026*

---

## What we looked at

 We reviewed the current Slovak knowledge base together with the **list of ~600 test questions** used to check the assistant. We set out to answer three questions the team raised:

1. Can the assistant reliably pick the **same** article (and give the **same** answer) when a customer asks the same question more than once?
2. What **additional questions** are customers likely to ask — especially ones we may not be able to answer today?
3. For each article, is anything **missing or unclear** that could confuse the assistant?

This document summarises what we found and what we recommend. A short methodology note is in the appendix; the detailed technical files are available for the data and engineering teams.

---

## The headline

1. **The assistant can give different answers to the same question.** Many articles overlap — they cover the same topic in very similar words — so the step that picks "which article should I use?" can land on a different article from one attempt to the next. About **one in four** articles has a close look-alike.

2. **A number of articles contain wrong or conflicting information.** We confirmed **12 clear errors** by hand (and flagged roughly 90 more for the team to check). Examples include an article about *foreign-currency accounts* that actually describes the *children's account*, and a credit-card age limit stated two different ways.

3. **Exact fees, rates and limits aren't in the knowledge base — by design.** For these, the assistant points the customer to the official price list (*Sadzobník*) as a **link they can open**, which is a reasonable approach. What still needs attention is narrower: a few articles mention the price list but include **no link**, some show an **example figure** that could be mistaken for the real one, and the **accuracy of the linked documents themselves** is something this review could not verify.

None of these are catastrophic, and all are fixable. The rest of this document explains each one with examples, the customer impact, and what we recommend.

---

## 1. The assistant doesn't always give the same answer to the same question

**What's happening.** When a customer asks something, the assistant searches the library, then uses an AI step to decide which article (or articles) best match the question. When two articles are very similar, that decision can go either way — so the customer can get a slightly different answer on a second try, or the assistant can pull in two overlapping articles at once.

**Why.** The library has grown to the point where many articles repeat each other. We found that **about a quarter of all articles have a close look-alike**, and we examined the 40 closest cases in detail — **27 of them are genuinely confusing** (a realistic customer question would fit either article equally well).

**Typical examples (in everyday terms):**

- **Loans.** The "refinancing loan" and "general-purpose loan" articles are nearly identical on rates, limits, instalments and how to apply. A customer who just asks *"What's the interest rate on a loan?"* — without naming which one — could be answered from either.
- **Budgeting.** **Four** different articles all explain "how to set a budget" (overall budget, by category, by sub-category, and the spending-overview screen). The assistant has no clear rule for which to use.
- **Cards.** *"I lost my card — how do I block it and get a new one?"* is described in two separate articles (blocking, and ordering a replacement).
- **Standing orders.** Setting one up and viewing/changing existing ones live in articles whose titles overlap.

**Why it matters.** Inconsistent answers undermine customer trust, make the assistant harder to test and quality-check, and — when two overlapping articles are used together — can place **conflicting facts in the same answer** (see the next section for a real example).

**Bottom line.** The single most effective fix is to make sure **each topic lives in exactly one article**, and that articles don't repeat each other's content.

---

## 2. Some articles contain wrong or conflicting information

These are the highest-priority fixes because they are quick to correct and directly affect answer quality. We verified each of the items below against the actual article text.

| Topic / article | The problem | What a customer might experience |
|---|---|---|
| **Foreign-currency account** | The article's main text is actually the **children's-account** text (it was copied in by mistake). | Asking about a foreign-currency account returns children's-account information. |
| **Credit-card age limit** | One article says **"over 18" (no upper limit)**; another says **"18 to 69"**. | Different age answers depending on the day. |
| **Term deposit – minimum amount** | One article says the CZK minimum is **10,000**; another says **12,000**. | A wrong or inconsistent minimum-deposit figure. |
| **Investing eligibility** | An article states you must have **permanent residence in the Czech Republic** — but this is a Slovak bank. | A Slovak customer could be told (incorrectly) they aren't eligible. |
| **Card ePIN length** | The same article calls the ePIN **both a 4-digit and a 5-digit code**. | The assistant could state either. |
| **"What George can do"** | The text contains a leftover internal note: *"Learn more [here]. – add link"*. | The customer sees an unfinished editorial note. |
| **Mortgage interest rate** | A sentence is **cut off mid-word**. | An incomplete, garbled answer. |

In addition to these confirmed items, the review flagged **around 90 more** likely contradictions or mismatches (for example, an article whose title promises one thing while the body covers another) for the team to verify.

**Why it matters.** Wrong figures on age, fees, minimums or eligibility are not just confusing — for a bank they carry **compliance and mis-selling risk**. These should be corrected first.

**Bottom line.** Fix the confirmed errors now; review the flagged list next.

---

## 3. Fees, rates and limits: handled by a link to the price list (mostly fine, a few gaps)

The questions customers ask most — *"How much can I withdraw per day?"*, *"What's the fee?"*, *"What interest rate will I pay?"* — need a **specific number**. These numbers change often, so the knowledge base deliberately does **not** repeat them. Instead the assistant points the customer to the official **price list (*Sadzobník*)** or product page as a **link they can open**. This is a sensible design, and for most of these questions it is the right behaviour.

Three things still need attention:

- **A few articles refer to the price list but include no link** — so the customer has nothing to click (we confirmed this on at least one loan article). More broadly, a number of fee/rate questions land on articles that give **neither the figure nor a link**, leaving the customer with nothing. Every fees/rates/limits question should reliably surface a working link.
- **Some articles contain an example figure** (for instance an illustrative interest rate). There is a risk the assistant repeats it as if it were the real, current number.
- **We could not check whether the linked documents are correct or up to date.** That is outside what this review could see, and the team has noted it cannot currently confirm it either — so it is worth a separate check.

**Bottom line.** Keep the link-based approach. Just make sure (a) every fees/rates/limits answer reliably shows a **working link**, (b) **example figures** are removed or clearly labelled as illustrative, and (c) the **linked documents are verified** separately for accuracy and currency.

---

## 4. We've prepared ~400 new test questions (ready to use)

To strengthen testing, we drafted **about 400 additional, realistic customer questions** (in Slovak and English), on top of the existing ~600. About **330** are questions the knowledge base doesn't fully answer as written. These split into two useful groups:

- **~120 are fees/rates/limits questions** that should simply trigger the **price-list link** (see Section 3) — a ready-made set for testing that the assistant surfaces the right link.
- **~210 are genuine knowledge gaps** — missing procedures, eligibility rules or conditions worth filling in. Each is labelled with the **area of information that's missing**, so the team can act on it even where we couldn't know the exact detail.

These are delivered as a spreadsheet-ready file that matches the existing test-question format, so they can be reviewed and added directly.

---

## 5. Housekeeping: a few naming mismatches

The test set and the knowledge base disagree on the **spelling of four topic names** (for example, singular vs plural, or a slightly different code). Because of this, four articles look "untested" when they are actually just labelled differently, and a few test questions point to a name that doesn't exist. One topic — roughly *"payment without confirmation"* — has **no matching article at all** and needs a closer look. These are quick, clerical fixes but worth doing so coverage reporting is accurate.

---

## What we recommend (in priority order)

1. **Correct the confirmed errors** in Section 2 — fast, and the highest impact on answer quality and compliance.
2. **Make sure every fees/rates/limits answer reliably shows a working price-list link** (Section 3), remove stray example figures, and arrange a **separate check that the linked documents are correct and current**.
3. **Reduce look-alike articles** (Section 1) — aim for one topic per article, with no repeated content. Start with loans, budgeting and cards.
4. **Fix the naming mismatches** (Section 5) so testing coverage is reported correctly.
5. **(For the technical team)** Today, when the assistant chooses an article it reads only the article's main text — not the short summary or the "what this is **not** about" note that authors carefully wrote. Letting it use those notes (and fixing the ones that are currently wrong) would help it pick the right article more consistently.

---

## What we need from you (decisions)

- **"Not yet validated" articles.** 24 articles are marked as not-yet-validated. Are these live for customers? If yes, they need the same quality bar; if no, they should be hidden so the assistant can't use them.
- **ePIN length.** Is the ePIN **4 or 5 digits**? (The dedicated ePIN article doesn't say, and another article says both.)
- **"Payment without confirmation" topic.** Is this missing knowledge that should be written, or does it belong to an existing topic?

---

## Appendix A — How we assessed this (methodology)

- We worked from two files: the current Slovak knowledge base (~200 articles) and the existing test set (~600 questions).
- **Look-alike articles:** we compared every article against every other one using standard text-similarity techniques to surface the closest pairs, then had the assistant's underlying AI read each close pair in full and judge whether a real customer question could land on either — i.e. whether it is genuinely confusing, not just similar in wording.
- **Article-by-article review:** we had the AI read all ~200 articles (in Slovak) and flag anything missing, vague, contradictory, or off-topic, and propose realistic additional customer questions — noting, for each, whether the knowledge base can answer it and, if not, the area of information that's missing.
- **Hand-checking:** we verified the most important problems (the errors in Section 2) directly against the source text, so those are confirmed rather than estimated. The ~90 further contradictions are flagged for the team to confirm.
- **How the assistant actually chooses articles:** we read the assistant's own code to confirm how it selects articles, so the findings reflect the live system, not an assumption.
- **One known limitation:** a more advanced "meaning-based" similarity tool was blocked by the corporate network, so the look-alike numbers are a **conservative estimate** — the real amount of overlap is likely a little higher, not lower. The hand-verified errors and the article-by-article findings do not depend on that tool.

## Appendix B — Detailed material (for the data & engineering teams)

The full technical detail behind this summary lives alongside this document:

- **Look-alike / consistency analysis** — every confusing article pair, with the customer question that triggers it and the suggested fix.
- **Article-by-article findings** — all gaps and issues per article (about 330 in total).
- **Data-quality & coverage notes** — the confirmed errors, the flagged list, and the naming mismatches.
- **New test questions** — the ~400 proposed questions as an append-ready file.
