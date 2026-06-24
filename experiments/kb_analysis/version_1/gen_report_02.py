"""Generate report_02_test_set_expansion.md from the merged gap-analysis data."""
import os, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_02_test_set_expansion.md")
def fam(k): return k.split("@", 1)[0] if "@" in k else k

nq = pd.read_csv(os.path.join(HERE, "analysis_outputs/gap_new_questions.csv")).fillna("")
nq["family"] = nq.knowledgeId.map(fam)
gap = nq[nq.answerable_by_kb.isin(["partial", "no"])].copy()
ans = nq[nq.answerable_by_kb == "yes"]
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
  "KB, as written, cannot fully answer them.\n")

w("\n## Method\n")
w("For every fragment, the content pass proposed up to 3 *additional* high-probability user questions "
  "that are **not** already in the test set (deduplicated against the existing questions per topic), "
  "prioritising ones the KB **cannot** fully answer. Each is tagged `answerable_by_kb` = "
  "`yes` / `partial` / `no`, a `priority`, and — when not fully answerable — a `missing_area`.\n")

w("\n## Summary\n")
w(f"- **{len(nq)}** proposed questions across **{nq.knowledgeId.nunique()}** fragments.\n")
w(f"- **{len(gap)} are gap-exposing** (`partial`/`no`) — the KB cannot fully answer them today; "
  f"**{(gap.answerable_by_kb=='no').sum()}** it cannot answer at all.\n")
w(f"- **{len(ans)}** are fully answerable — useful as extra coverage/regression tests for content the KB *does* hold.\n")
w(f"- By priority: HIGH {int((nq.priority=='HIGH').sum())}, MED {int((nq.priority=='MED').sum())}, "
  f"LOW {int((nq.priority=='LOW').sum())}.\n")

w("\n## How to use `new_test_cases.csv`\n")
w("- Columns match `SLSP_test_cases.csv` exactly; `test_case_number` uses a `PROPOSED-NNNN` prefix so "
  "the rows are distinguishable from validated cases until you accept them.\n")
w("- `comment` encodes `PROPOSED <priority>/<answerable_by_kb>`, the `missing_area` (for gap cases), and a one-line rationale.\n")
w("- **Triage suggestion:** the `[KB-GAP:no]` and `[KB-GAP:partial]` rows are the ones that should drive KB edits; "
  "the `[answerable]` rows can be merged into the eval set as-is (after a gold-answer pass).\n")
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
        a = str(a).strip()
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
    m = str(r.missing_area).strip().replace("|", "/")
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
