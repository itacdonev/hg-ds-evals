"""Merge the per-batch gap-analysis result JSONs into consolidated CSVs.

Reads analysis_outputs/gap_batches/*_result.json (one per thematic batch),
validates the schema, and writes three tidy CSVs into analysis_outputs/:
  - gap_completeness_issues.csv   one row per completeness issue
  - gap_new_questions.csv         one row per proposed new question
  - gap_per_fragment.csv          one row per fragment (counts + overall_note)
Also reconciles coverage against the KB so we know every fragment was processed.
"""
from __future__ import annotations
import glob, json, os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
GB = os.path.join(HERE, "analysis_outputs", "gap_batches")
OUT = os.path.join(HERE, "analysis_outputs")

EXPECTED_BATCHES = [
    "savings","loans_1","loans_2","current_accounts_giro_1","current_accounts_giro_2",
    "credit_cards","debit_cards_card_ops_1","debit_cards_card_ops_2","insurance",
    "investing_pension","payments_transfers","pfm_statements","atm_cash",
    "accounts_app_misc_1","accounts_app_misc_2","accounts_app_misc_3",
]

def main():
    kb_ids = set(pd.read_csv(os.path.join(HERE,"feature-and-product-knowledge.local.csv"),
                             sep="|", dtype=str)["feature-and-product-knowledge.knowledgeId"].str.strip())
    issues, questions, perfrag = [], [], []
    seen_ids, present_batches, missing_batches = set(), [], []
    for b in EXPECTED_BATCHES:
        path = os.path.join(GB, f"{b}_result.json")
        if not os.path.exists(path):
            missing_batches.append(b); continue
        present_batches.append(b)
        data = json.load(open(path, encoding="utf-8"))
        for frag in data:
            kid = frag["knowledgeId"]; seen_ids.add(kid)
            ci = frag.get("completeness_issues", []) or []
            nq = frag.get("new_questions", []) or []
            for it in ci:
                issues.append({"knowledgeId": kid, "batch": b,
                    "type": it.get("type",""), "determinism_impact": it.get("determinism_impact",""),
                    "detail": it.get("detail",""), "fix_or_missing_area": it.get("fix_or_missing_area","")})
            for q in nq:
                questions.append({"knowledgeId": kid, "batch": b,
                    "question_sk": q.get("question_sk",""), "question_en": q.get("question_en",""),
                    "answerable_by_kb": q.get("answerable_by_kb",""), "missing_area": q.get("missing_area",""),
                    "priority": q.get("priority",""), "rationale": q.get("rationale","")})
            perfrag.append({"knowledgeId": kid, "batch": b,
                "n_issues": len(ci),
                "n_high_issues": sum(1 for x in ci if x.get("determinism_impact")=="HIGH"),
                "n_new_questions": len(nq),
                "n_gap_questions": sum(1 for x in nq if x.get("answerable_by_kb") in ("partial","no")),
                "overall_note": frag.get("overall_note","")})

    pd.DataFrame(issues).to_csv(os.path.join(OUT,"gap_completeness_issues.csv"), index=False)
    pd.DataFrame(questions).to_csv(os.path.join(OUT,"gap_new_questions.csv"), index=False)
    pd.DataFrame(perfrag).to_csv(os.path.join(OUT,"gap_per_fragment.csv"), index=False)

    print(f"Batches present: {len(present_batches)}/{len(EXPECTED_BATCHES)}")
    if missing_batches: print("  MISSING:", missing_batches)
    print(f"Fragments processed: {len(seen_ids)}  |  KB fragments: {len(kb_ids)}")
    not_proc = kb_ids - seen_ids
    if not_proc: print(f"  NOT processed ({len(not_proc)}):", sorted(not_proc))
    extra = seen_ids - kb_ids
    if extra: print("  Unknown ids in results:", sorted(extra))
    print(f"\nCompleteness issues: {len(issues)}  (HIGH={sum(1 for i in issues if i['determinism_impact']=='HIGH')}, "
          f"MED={sum(1 for i in issues if i['determinism_impact']=='MED')}, "
          f"LOW={sum(1 for i in issues if i['determinism_impact']=='LOW')})")
    if issues:
        print("  by type:", pd.Series([i['type'] for i in issues]).value_counts().to_dict())
    print(f"New questions: {len(questions)}  (gap-exposing partial/no="
          f"{sum(1 for q in questions if q['answerable_by_kb'] in ('partial','no'))}, "
          f"fully answerable yes={sum(1 for q in questions if q['answerable_by_kb']=='yes')})")
    if questions:
        print("  by priority:", pd.Series([q['priority'] for q in questions]).value_counts().to_dict())

if __name__ == "__main__":
    main()
