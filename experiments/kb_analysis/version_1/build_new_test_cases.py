"""Build new_test_cases.csv (append-ready, SLSP schema) from the proposed questions.

Classifies each gap-exposing question as:
  - LINK-EXPECTED : the answer is a fee/rate/limit that, by design, is deferred to the
                    official price list (Sadzobník) and shown to the customer as a link.
                    These are not knowledge gaps; they test that the assistant surfaces a link.
  - KB-GAP        : a genuine knowledge gap (procedure / eligibility / condition not written).
  - answerable    : the KB can answer it as-is (extra coverage).
Dedupes against the existing test set and within the proposal; sorts most-valuable first.
"""
import csv, re
import pandas as pd

PRICELIST_RE = re.compile(
    r"fee|poplat|limit|rate|sadzb|úrok|urok|price|cena|\beur\b|amount|apr|rpmn|interest",
    re.IGNORECASE,
)

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9áäčďéíĺľňóôŕšťúýž ]", "", str(s).lower()).strip()

def main():
    nq = pd.read_csv("analysis_outputs/gap_new_questions.csv").fillna("")
    ts = pd.read_csv("SLSP_test_cases.csv", sep=";", encoding="utf-8-sig").fillna("")
    ts["knowledge_topic_ID"] = ts["knowledge_topic_ID"].str.strip()

    existing: dict[str, set[str]] = {}
    for _, r in ts.iterrows():
        existing.setdefault(r.knowledge_topic_ID, set()).add(norm(r.question_SK))

    ans_rank = {"no": 0, "partial": 1, "yes": 2}
    pri_rank = {"HIGH": 0, "MED": 1, "LOW": 2}
    nq["_a"] = nq.answerable_by_kb.map(lambda x: ans_rank.get(x, 3))
    nq["_p"] = nq.priority.map(lambda x: pri_rank.get(x, 3))
    nq = nq.sort_values(["_a", "_p"])

    seen: set[tuple[str, str]] = set()
    rows = []
    for _, r in nq.iterrows():
        key = (r.knowledgeId, norm(r.question_sk))
        if norm(r.question_sk) in existing.get(r.knowledgeId, set()):
            continue
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)

    out = []
    n_link = n_gap = n_ans = 0
    for i, r in enumerate(rows, 1):
        if r.answerable_by_kb == "yes":
            kind, n_ans = "[answerable]", n_ans + 1
            miss = ""
        elif PRICELIST_RE.search(str(r.missing_area)):
            kind, n_link = "[LINK-EXPECTED] should surface the price-list link", n_link + 1
            miss = f" missing_area: {r.missing_area}"
        else:
            kind, n_gap = "[KB-GAP]", n_gap + 1
            miss = f" missing_area: {r.missing_area}"
        comment = f"PROPOSED {r.priority}/{r.answerable_by_kb} {kind}{miss} | {r.rationale}"
        out.append({
            "test_case_number": f"PROPOSED-{i:04d}", "knowledge_topic_ID": r.knowledgeId,
            "golden_standard_question?": "", "DESCOPE": "",
            "question_SK": r.question_sk, "question_EN": r.question_en,
            "expected_answer_SK": "", "expected_answer_EN": "", "comment": comment,
        })

    cols = ["test_case_number", "knowledge_topic_ID", "golden_standard_question?", "DESCOPE",
            "question_SK", "question_EN", "expected_answer_SK", "expected_answer_EN", "comment"]
    pd.DataFrame(out)[cols].to_csv(
        "new_test_cases.csv", sep=";", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )
    print(f"Wrote new_test_cases.csv: {len(out)} rows")
    print(f"  LINK-EXPECTED (fees/rates/limits → link): {n_link}")
    print(f"  KB-GAP (genuine missing knowledge):       {n_gap}")
    print(f"  answerable (extra coverage):              {n_ans}")

if __name__ == "__main__":
    main()
