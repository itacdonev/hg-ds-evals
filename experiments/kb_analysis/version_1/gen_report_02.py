"""Generate report_02_test_set_expansion.md from the merged gap-analysis data."""
import os, re, pandas as pd

PRICELIST_RE = re.compile(
    r"fee|poplat|limit|rate|sadzb|úrok|urok|price|cena|\beur\b|amount|apr|rpmn|interest", re.I)

# Neutralise the (now-corrected) "agent can't read the link" phrasings in the
# sub-agent-authored text: links are shown to the customer to open (Report 0 §3).
_SCRUB = [
    (r"\s*,?\s*which the agent cannot render as in-app guidance", ""),
    (r"the agent cannot read", "the customer opens it via a link"),
    (r"unanswerable by design", "answered by surfacing the link"),
]
def _scrub(t: str) -> str:
    for pat, rep in _SCRUB:
        t = re.sub(pat, rep, t, flags=re.I)
    return re.sub(r"\s{2,}", " ", t)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_02_test_set_expansion.md")
def fam(k): return k.split("@", 1)[0] if "@" in k else k

nq = pd.read_csv(os.path.join(HERE, "analysis_outputs/gap_new_questions.csv")).fillna("")
nq["family"] = nq.knowledgeId.map(fam)
gap = nq[nq.answerable_by_kb.isin(["partial", "no"])].copy()
ans = nq[nq.answerable_by_kb == "yes"]
gap["_pricelist"] = gap.missing_area.map(lambda m: bool(PRICELIST_RE.search(str(m))))
link_expected = gap[gap._pricelist]
genuine_gap = gap[~gap._pricelist]
PR = {"HIGH": 0, "MED": 1, "LOW": 2}
nq["_p"] = nq.priority.map(lambda x: PR.get(x, 3))
gap["_p"] = gap.priority.map(lambda x: PR.get(x, 3))
gap["_a"] = gap.answerable_by_kb.map({"no": 0, "partial": 1})

L = []; w = L.append
w("# Report 2 — test-set expansion: new questions & KB gaps\n")
w("**Inputs:** the 198-fragment KB and the existing 604-row test set (`SLSP_test_cases.csv`).\n")
w("**Deliverable:** `new_test_cases.csv` — proposed questions in the **exact SLSP schema** (SK + EN), "
  "directly appendable; gap-exposing cases are left unanswered with the missing-information area noted "
  "in `comment`. This report explains and prioritises them.\n")
w("> Per your instruction I did **not** draft `expected_answer_SK` for the answerable additions — those "
  "need an authoritative source. Gap cases have no expected answer **by design**: the point is that the "
  "KB, as written, doesn't fully answer them.\n")
w("\n> **Important framing (corrected).** Exact fees/rates/limits are *intentionally* not in the KB — "
  "they live in the official price list (Sadzobník), shown to the customer as a **clickable link they "
  "open themselves**. So fee/rate/limit questions are **not knowledge gaps**; they are *link-expected* "
  "test cases (the assistant should surface the right link). They are tagged `[LINK-EXPECTED]` in the CSV "
  "and counted separately below. The genuine knowledge gaps are the rest (missing procedures, eligibility, "
  "conditions), tagged `[KB-GAP]`. Where an auto-generated line below says the agent *cannot read* a "
  "PDF/link or that a rate is *unanswerable by design*, read it as: *the figure isn't restated in the "
  "ENUM fragment — it lives in the linked price list the customer opens.*\n")

w("\n## Method\n")
w("For every fragment, the content pass proposed up to 3 *additional* high-probability user questions "
  "that are **not** already in the test set (deduplicated against the existing questions per topic), "
  "prioritising ones the KB **cannot** fully answer. Each is tagged `answerable_by_kb` = "
  "`yes` / `partial` / `no`, a `priority`, and — when not fully answerable — a `missing_area`.\n")

w("\n## Summary\n")
w(f"- **{len(nq)}** proposed questions across **{nq.knowledgeId.nunique()}** fragments.\n")
w(f"- **{len(genuine_gap)} are genuine knowledge gaps** (`[KB-GAP]`) — missing procedures, eligibility "
  "rules or conditions the KB should hold but doesn't. **These are the ones to fill.**\n")
w(f"- **{len(link_expected)} are link-expected** (`[LINK-EXPECTED]`) — fees/rates/limits whose answer is "
  "the price-list link by design; use them to test that the assistant surfaces the right link, not to write KB content.\n")
w(f"- **{len(ans)}** are fully answerable today — useful as extra coverage/regression tests.\n")
w(f"- By priority: HIGH {int((nq.priority=='HIGH').sum())}, MED {int((nq.priority=='MED').sum())}, "
  f"LOW {int((nq.priority=='LOW').sum())}.\n")

w("\n## How to use `new_test_cases.csv`\n")
w("- Columns match `SLSP_test_cases.csv` exactly; `test_case_number` uses a `PROPOSED-NNNN` prefix so "
  "the rows are distinguishable from validated cases until you accept them.\n")
w("- `comment` encodes `PROPOSED <priority>/<answerable_by_kb>`, a tag (`[KB-GAP]` / `[LINK-EXPECTED]` / "
  "`[answerable]`), the `missing_area`, and a one-line rationale.\n")
w("- **Triage suggestion:** `[KB-GAP]` rows drive KB content edits; `[LINK-EXPECTED]` rows test that the "
  "assistant returns the right price-list link (and flag any ENUM fragment missing that link); `[answerable]` rows "
  "can be merged into the eval set as-is after a gold-answer pass.\n")
w("- Rows are sorted so gap-exposing + HIGH-priority appear first.\n")

# ---- Gap areas by family ----
w("\n## 1. Missing-information areas, by product family\n")
w("Each family below lists the **distinct information areas** the KB is missing (named so colleagues can fill them), "
  "with the count of gap-exposing questions that hit that family.\n")
fam_order = gap.family.value_counts().index.tolist()
for f in fam_order:
    sub = gap[gap.family == f]
    areas = []
    seen = set()
    for a in sub.sort_values(["_a", "_p"]).missing_area:
        a = _scrub(str(a).strip())
        key = a.lower()[:40]
        if a and key not in seen:
            seen.add(key); areas.append(a)
    if not areas:
        continue
    w(f"\n### {f} — {len(sub)} gap questions\n")
    for a in areas[:12]:
        if len(a) > 180: a = a[:177] + "..."
        w(f"- {a}\n")
    if len(areas) > 12:
        w(f"- …and {len(areas)-12} more (see CSV).\n")

# ---- Featured highest-priority gap questions ----
w("\n---\n## 2. Highest-priority gap questions (feature list)\n")
w("HIGH-priority questions the KB cannot fully answer — the ones most worth fixing *and* testing. "
  "Full set (398) in `new_test_cases.csv`.\n\n")
w("| Fragment | Question (EN) | Missing |\n|---|---|---|\n")
feat = gap[gap.priority == "HIGH"].sort_values(["_a", "family"])
for _, r in feat.head(45).iterrows():
    q = r.question_en.strip().replace("|", "/")
    m = _scrub(str(r.missing_area).strip().replace("|", "/"))
    if len(q) > 90: q = q[:87] + "..."
    if len(m) > 90: m = m[:87] + "..."
    w(f"| `{r.knowledgeId}` | {q} | {m} |\n")
if len(feat) > 45:
    w(f"\n*(+{len(feat)-45} more HIGH-priority gap questions in the CSV.)*\n")

# ---- answerable additions ----
w("\n---\n## 3. Fully-answerable additions (extra coverage)\n")
w(f"{len(ans)} proposed questions the KB *can* answer — natural high-probability phrasings not yet tested. "
  "Good regression coverage once gold answers are written. Examples:\n\n")
for _, r in ans.head(12).iterrows():
    w(f"- **`{r.knowledgeId}`** — {r.question_en.strip()}\n")
w("\nSee `new_test_cases.csv` (`comment` contains `[answerable]`) for all of them.\n")

open(OUT, "w", encoding="utf-8").write("".join(L))
print("Wrote", OUT, "—", len(nq), "questions,", len(gap), "gap,", len(ans), "answerable,",
      len(fam_order), "families.")
