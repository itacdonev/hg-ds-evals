# SK — Knowledge Base Review: Findings

*June 2026*

---

## What we analysed

 We reviewed the current SK knowledge base together with the **list of ~600 test questions** used to check the assistant. We set out to answer three questions:

1. What is the level of nondeterminism in our answers -> Can the assistant reliably pick the **same** ENUM fragment (and give the **same** answer) when a customer asks the same question more than once?
2. What **additional questions** are customers likely to ask — especially ones we may not be able to answer today?
3. For each ENUM fragment, is anything **missing or unclear** that could confuse the assistant?

This document summarises what we found and what we recommend. A short methodology note is in the appendix; the detailed technical files are available for the data and engineering teams.

---

## Summary of main findings

1. **The assistant can give different answers to the same question.** Many ENUM fragments overlap — they cover the same topic — so the step that picks "which ENUM fragment should I use?" can land on a different ENUM fragment from one attempt to the next. When we compare by **meaning** (not just shared words), about **half** of all ENUM fragments have a close look-alike.

2. **A number of ENUM fragments contain wrong or conflicting information.** We confirmed **12 clear errors** by hand (and flagged roughly 90 more for the team to check). Examples include an ENUM fragment about *foreign-currency accounts* that actually describes the *children's account*, and a credit-card age limit stated two different ways.

3. **Exact fees, rates and limits aren't in the knowledge base — by design.** For these, the assistant points the customer to the official price list (*Sadzobník*) as a **link they can open**, which is a reasonable approach. What still needs attention is narrower: a few ENUM fragments mention the price list but include **no link**, some show an **example figure** that could be mistaken for the real one, and the **accuracy of the linked documents themselves** is something this review could not verify.

All of these are fixable. The rest of this document explains each one with examples, the customer impact, and what can be the potential solution.

---

## 1. The assistant doesn't always give the same answer to the same question

**What's happening.** When a customer asks something, the assistant searches the library, then uses an AI step to decide which ENUM fragment (or ENUM fragments) best match the question. When two ENUM fragments are very similar, that decision can go either way — so the customer can get a slightly different answer on a second try, or the assistant can pull in two overlapping ENUM fragments at once.

**Why.** The KB has grown to the point where many ENUM fragments repeat each other. Comparing by meaning, **about half of all ENUM fragments have a close look-alike**. We then had the assistant's own AI read the closest cases in full — both the ones that look alike in *wording* and the ones that only look alike in *meaning* — and **at least 32 pairs are genuinely confusing** (a realistic customer question would fit either ENUM fragment equally well). Several of these only show up on the meaning-based comparison — for example, the explanation of an "authorised person" is repeated almost word-for-word across savings, current and foreign-currency accounts, so a question like *"how do I add an authorised person?"* has no single right ENUM fragment to land on.

**Typical examples (in everyday terms):**

- **Loans.** The "refinancing loan" and "general-purpose loan" ENUM fragments are nearly identical on rates, limits, instalments and how to apply. A customer who just asks *"What's the interest rate on a loan?"* — without naming which one — could be answered from either.
- **Budgeting.** **Four** different ENUM fragments all explain "how to set a budget" (overall budget, by category, by sub-category, and the spending-overview screen). The assistant has no clear rule for which to use.
- **Cards.** *"I lost my card — how do I block it and get a new one?"* is described in two separate ENUM fragments (blocking, and ordering a replacement).
- **Standing orders.** Setting one up and viewing/changing existing ones live in ENUM fragments whose titles overlap.

**Why it matters.** Inconsistent answers undermine customer trust, make the assistant harder to test and quality-check, and — when two overlapping ENUM fragments are used together — can place **conflicting facts in the same answer** (see the next section for a real example).

**Bottom line.** The single most effective fix is to make sure **each topic lives in exactly one ENUM fragment**, and that ENUM fragments don't repeat each other's content.

---

## 2. Some ENUM fragments contain wrong or conflicting information

These are the highest-priority fixes because they are quick to correct and directly affect answer quality. We verified each of the items below against the actual ENUM fragment text.

| Topic / ENUM fragment | The problem | What a customer might experience |
|---|---|---|
| **Foreign-currency account** | The ENUM fragment's main text is actually the **children's-account** text (it was copied in by mistake). | Asking about a foreign-currency account returns children's-account information. |
| **Credit-card age limit** | One ENUM fragment says **"over 18" (no upper limit)**; another says **"18 to 69"**. | Different age answers depending on the day. |
| **Term deposit – minimum amount** | One ENUM fragment says the CZK minimum is **10,000**; another says **12,000**. | A wrong or inconsistent minimum-deposit figure. |
| **Investing eligibility** | An ENUM fragment states you must have **permanent residence in the Czech Republic** — but this is a Slovak bank. | A Slovak customer could be told (incorrectly) they aren't eligible. |
| **Card ePIN length** | The same ENUM fragment calls the ePIN **both a 4-digit and a 5-digit code**. | The assistant could state either. |
| **"What George can do"** | The text contains a leftover internal note: *"Learn more [here]. – add link"*. | The customer sees an unfinished editorial note. |
| **Mortgage interest rate** | A sentence is **cut off mid-word**. | An incomplete, garbled answer. |

In addition to these confirmed items, the review flagged **around 90 more** likely contradictions or mismatches (for example, an ENUM fragment whose title promises one thing while the body covers another) for the team to verify.

**Why it matters.** Wrong figures on age, fees, minimums or eligibility are not just confusing — for a bank they carry **compliance and mis-selling risk**. These should be corrected first.

**Bottom line.** Fix the confirmed errors now; review the flagged list next.

---

## 3. Fees, rates and limits: handled by a link to the price list

The questions customers ask most — *"How much can I withdraw per day?"*, *"What's the fee?"*, *"What interest rate will I pay?"* — need a **specific number**. These numbers change often, so the knowledge base deliberately does **not** repeat them. Instead the assistant points the customer to the official **price list (*Sadzobník*)** or product page as a **link they can open**.

Three things still need attention:

- **A few ENUM fragments refer to the price list but include no link** — so the customer has nothing to click (we confirmed this on at least one loan ENUM fragment). More broadly, a number of fee/rate questions land on ENUM fragments that give **neither the figure nor a link**, leaving the customer with nothing. Every fees/rates/limits question should reliably surface a working link.
- **Some ENUM fragments contain an example figure** (for instance an illustrative interest rate). There is a risk the assistant repeats it as if it were the real, current number.
- **We could not check whether the linked documents are correct or up to date.** That is outside what this review could see, and the team has noted it cannot currently confirm it either — so it is worth a separate check.

**Status.** This is currently being worked on — the link-handling side is being investigated.

---

## 4. We've prepared ~400 new test questions (ready for review)

To strengthen testing, we drafted **about 400 additional, realistic customer questions** (in Slovak and English), on top of the existing ~600. About **330** are questions the knowledge base doesn't fully answer as written. These split into two useful groups:

- **~120 are fees/rates/limits questions** that should simply trigger the **price-list link** (see Section 3) — a ready-made set for testing that the assistant surfaces the right link.
- **~210 are genuine knowledge gaps** — missing procedures, eligibility rules or conditions worth filling in. Each is labelled with the **area of information that's missing**, so the team can act on it even where we couldn't know the exact detail.

These are delivered as a spreadsheet-ready file that matches the existing test-question format, so they can be reviewed and added directly.

---

## 5. Housekeeping: a few naming mismatches

The test set and the knowledge base disagree on the **spelling of four topic names** (for example, singular vs plural, or a slightly different code). Because of this, four ENUM fragments look "untested" when they are actually just labelled differently, and a few test questions point to a name that doesn't exist. One topic — roughly *"payment without confirmation"* — has **no matching ENUM fragment at all** and needs a closer look. These are quick, clerical fixes but worth doing so coverage reporting is accurate.

---

## Open questions

- **Where should the price-list / document links live, and how?** Today these links sit inside the ENUM fragment text. It may be better to **add them as a dedicated, curated field in the knowledge base** and/or to **upload the referenced documents (Sadzobník, product PDFs) separately** into a place the assistant can use directly — so a fee/rate answer always carries the right, current link. **This needs investigation** — both the technical approach and who owns keeping the links/documents current.
- **Who verifies the linked documents?** Their accuracy and how up-to-date they are could not be checked in this review, and — per the team — cannot currently be confirmed. This needs an owner.

---

## Appendix A — How we assessed this (methodology)

- We worked from two files: the current Slovak knowledge base (~200 ENUM fragments) and the existing test set (~600 questions).
- **Look-alike ENUM fragments:** we compared every ENUM fragment against every other one, both by **wording** and by **meaning** (a multilingual AI language model that understands Slovak), to surface the closest pairs. We then had the assistant's underlying AI read each close pair in full and judge whether a real customer question could land on either — i.e. whether it is genuinely confusing, not just similar. The meaning-based comparison is important because the assistant itself matches questions by meaning, so it catches look-alikes that share no obvious words.
- **Per-ENUM-fragment review:** we had the AI read all ~200 ENUM fragments (in Slovak) and flag anything missing, vague, contradictory, or off-topic, and propose realistic additional customer questions — noting, for each, whether the knowledge base can answer it and, if not, the area of information that's missing.
- **Hand-checking:** we verified the most important problems (the errors in Section 2) directly against the source text, so those are confirmed rather than estimated. The ~90 further contradictions are flagged for the team to confirm.
- **How the assistant actually chooses ENUM fragments:** we read the assistant's own code to confirm how it selects ENUM fragments. It matches a customer's question against each fragment's **name and main text only** — the internal "scope notes" (the *not-in-scope* field) are **not used** at this step. Our look-alike and consistency analysis therefore compares fragments on that same main text, so the findings reflect the live system, not an assumption.
- **On the meaning-based analysis (now completed):** an earlier version of this review could not run the more advanced "meaning-based" similarity tool because of a corporate-network restriction. That has since been resolved — it was a local configuration fix and needed **no special access**. We have now run it, and it **confirmed and increased** the earlier estimate: by meaning, about **half** of ENUM fragments have a close look-alike (versus about a quarter on wording alone), and it surfaced **~200 additional look-alike pairs** the word-based method had missed entirely. We had the AI review the closest of these, and several were confirmed as genuine consistency risks. So the earlier numbers were, as flagged, conservative. One residual note: the model we used is a general-purpose one, not the exact engine inside the assistant, so it is a close proxy rather than an identical match — but the hand-verified errors and per-ENUM-fragment findings do not depend on it.

## Appendix B — Detailed material (for the data & engineering teams)

The full technical detail behind this summary lives alongside this document:

- **Look-alike / consistency analysis** — every confusing ENUM fragment pair, with the customer question that triggers it and the suggested fix.
- **Per-ENUM-fragment findings** — all gaps and issues per ENUM fragment (about 330 in total).
- **Data-quality & coverage notes** — the confirmed errors, the flagged list, and the naming mismatches.
- **New test questions** — the ~400 proposed questions as an append-ready file.
