"""Generate report_03_enum_completeness.md from the merged gap-analysis data.

Per-fragment completeness catalogue for all 198 ENUM fragments:
 - HIGH-impact fragments get a full write-up (grouped by family)
 - MED/LOW-only fragments get a compact one-line table
 - clean fragments are listed
Prose framing is written here; the catalogue is assembled from the CSVs so it
stays exhaustive and faithful to the source analysis.
"""
import os, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_03_enum_completeness.md")

def fam(k): return k.split("@", 1)[0] if "@" in k else k

kb = pd.read_csv(os.path.join(HERE, "feature-and-product-knowledge.local.csv"), sep="|", dtype=str).fillna("")
kb.columns = [c.replace("feature-and-product-knowledge.", "") for c in kb.columns]
kb["knowledgeId"] = kb["knowledgeId"].str.strip()
name_by = dict(zip(kb.knowledgeId, kb.knowledgeName))
type_by = dict(zip(kb.knowledgeId, kb.knowledgeType))

iss = pd.read_csv(os.path.join(HERE, "analysis_outputs/gap_completeness_issues.csv")).fillna("")
pf = pd.read_csv(os.path.join(HERE, "analysis_outputs/gap_per_fragment.csv")).fillna("")
pf["family"] = pf.knowledgeId.map(fam)
iss["family"] = iss.knowledgeId.map(fam)
RANK = {"HIGH": 0, "MED": 1, "LOW": 2}
iss["_r"] = iss.determinism_impact.map(lambda x: RANK.get(x, 3))

L = []
w = L.append

w("# Report 3 — per-ENUM completeness\n")
w("**KB:** `version_1/feature-and-product-knowledge.local.csv` — all 198 fragments analyzed.\n")
w("This is the exhaustive, per-fragment catalogue you asked for: *for each ENUM, is anything "
  "missing or under-explained enough to confuse the agent or make it non-deterministic for the "
  "same question?* Each fragment's `summary` + `notInScope` + `description` were read in full "
  "(Slovak). Where the exact missing fact can't be known from outside the bank, the **area** of "
  "the missing information is named instead — your colleagues can fill those in.\n")
w("> Read alongside [Report 0](report_00_kb_data_quality.md) (the hard bugs, de-duplicated and "
  "source-verified) and [Report 1](report_01_enum_disjointness.md) (cross-fragment confusability). "
  "Issues here are graded by **determinism impact**: **HIGH** = the agent will likely answer the "
  "same question differently across runs, or cannot answer at all; **MED** = an incomplete but "
  "stable answer; **LOW** = minor.\n")

n_frag = len(pf)
n_clean = int((pf.n_issues == 0).sum())
w("## Summary\n")
w(f"- **{len(iss)}** completeness issues across **{n_frag - n_clean}** fragments "
  f"(**{n_clean}** fragments had no issue).\n")
by_imp = iss.determinism_impact.value_counts()
w(f"- By determinism impact: **HIGH {int(by_imp.get('HIGH',0))}**, "
  f"MED {int(by_imp.get('MED',0))}, LOW {int(by_imp.get('LOW',0))}.\n")
by_type = iss.type.value_counts()
w("- By type: " + ", ".join(f"`{t}` {int(c)}" for t, c in by_type.items()) + ".\n")
w(f"- **{int((pf.n_high_issues>=1).sum())}** fragments carry at least one HIGH-impact issue "
  "(detailed below).\n")
w("\nThe two dominant patterns (see Report 0 §3 for the first): **missing concrete numbers** "
  "(fees/limits/rates deferred to the external Sadzobník) and **summary-vs-body scope mismatch** "
  "(the `summary` promises content the `description` never delivers).\n")
w("\n---\n")

# ---- HIGH-impact fragments, grouped by family ----
w("## 1. Fragments with HIGH-impact issues (full detail)\n")
w("Grouped by product family. Format: **issue type** — detail → *fix / missing area*.\n")
hi_ids = pf[pf.n_high_issues >= 1].sort_values(["family", "n_high_issues"], ascending=[True, False])
for f in sorted(hi_ids.family.unique()):
    w(f"\n### {f}\n")
    for kid in hi_ids[hi_ids.family == f].knowledgeId:
        nm = name_by.get(kid, ""); tp = type_by.get(kid, "")
        tag = f" · `{tp}`" if tp == "NOTVALIDATE" else ""
        w(f"\n**`{kid}`** — {nm}{tag}\n")
        sub = iss[iss.knowledgeId == kid].sort_values("_r")
        for _, r in sub.iterrows():
            if r.determinism_impact not in ("HIGH", "MED"):
                continue
            detail = r.detail.strip().replace("\n", " ")
            fix = r.fix_or_missing_area.strip().replace("\n", " ")
            w(f"- **{r.determinism_impact} · {r.type}** — {detail}" + (f" → *{fix}*" if fix else "") + "\n")

# ---- MED/LOW-only fragments, compact table ----
w("\n---\n")
w("## 2. Fragments with only MED / LOW issues (compact)\n")
w("One line per fragment; full per-issue detail in `analysis_outputs/gap_completeness_issues.csv`.\n\n")
w("| Fragment | #issues | Summary note |\n|---|---|---|\n")
medlow = pf[(pf.n_issues >= 1) & (pf.n_high_issues == 0)].sort_values(["family", "knowledgeId"])
for _, r in medlow.iterrows():
    note = (r.overall_note or "").strip().replace("\n", " ").replace("|", "/")
    if len(note) > 160: note = note[:157] + "..."
    w(f"| `{r.knowledgeId}` | {r.n_issues} | {note} |\n")

# ---- clean fragments ----
w("\n---\n")
w("## 3. Fragments with no completeness issue found\n")
clean = pf[pf.n_issues == 0].sort_values("knowledgeId").knowledgeId.tolist()
w(", ".join(f"`{k}`" for k in clean) + "\n")

w("\n---\n")
w("## How to read the data\n")
w("`analysis_outputs/gap_completeness_issues.csv` — one row per issue "
  "(`knowledgeId, type, determinism_impact, detail, fix_or_missing_area, batch`).\n\n"
  "`analysis_outputs/gap_per_fragment.csv` — one row per fragment "
  "(`n_issues, n_high_issues, n_new_questions, n_gap_questions, overall_note`).\n")

open(OUT, "w", encoding="utf-8").write("".join(L))
print("Wrote", OUT, "—", len(iss), "issues,", len(hi_ids), "HIGH fragments,",
      len(medlow), "MED/LOW fragments,", len(clean), "clean.")
