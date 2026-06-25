"""Build LLM-judge batches for the top semantic-only (lexically-missed) pairs.

These pairs scored high on the meaning-based model but BELOW the lexical
thresholds (tfidf<0.45 and lsa<0.70), so the original lexical-only pass never
judged them. We judge the top N here with the SAME schema as
confusable_pairs_judged.csv, so the determinism verdicts are comparable.

Writes analysis_outputs/semonly_batches/batch_<k>.json (each a list of pairs with
full fragment text) + SCHEMA.md. The judging itself is done by Claude subagents.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis_outputs", "semonly_batches")
os.makedirs(OUT, exist_ok=True)

TOP_N = 36
PER_BATCH = 12

kb = pd.read_csv(os.path.join(HERE, "feature-and-product-knowledge.local.csv"),
                 sep="|", dtype=str).fillna("")
kb.columns = [c.replace("feature-and-product-knowledge.", "") for c in kb.columns]
kb["knowledgeId"] = kb["knowledgeId"].str.strip()
for col in ("description", "summary", "notInScope", "knowledgeName"):
    kb[col] = kb[col].map(lambda s: s.replace("\\n", "\n").replace("\\t", "\t"))
by_id = {r.knowledgeId: r for _, r in kb.iterrows()}


def frag(kid: str) -> dict:
    r = by_id.get(kid)
    if r is None:
        return {"knowledgeId": kid, "MISSING": True}
    return {
        "knowledgeId": kid,
        "knowledgeName": r.knowledgeName,
        "summary": r.summary,
        "notInScope": r.notInScope,
        "description": r.description,
    }


so = pd.read_csv(os.path.join(HERE, "analysis_outputs", "semantic_only_pairs.csv"))
so = so.sort_values("semantic", ascending=False).head(TOP_N).reset_index(drop=True)

pairs = []
for i, r in so.iterrows():
    pairs.append({
        "pair_index": int(i),
        "id_a": r.id_a, "id_b": r.id_b,
        "semantic": float(r.semantic), "tfidf": float(r.tfidf), "lsa": float(r.lsa),
        "same_family": bool(r.same_family),
        "fragment_a": frag(r.id_a),
        "fragment_b": frag(r.id_b),
    })

n_batches = (len(pairs) + PER_BATCH - 1) // PER_BATCH
for k in range(n_batches):
    chunk = pairs[k * PER_BATCH:(k + 1) * PER_BATCH]
    with open(os.path.join(OUT, f"batch_{k+1}.json"), "w", encoding="utf-8") as fh:
        json.dump(chunk, fh, ensure_ascii=False, indent=2)
    print(f"batch_{k+1}.json: {len(chunk)} pairs")

print(f"\nTotal {len(pairs)} pairs in {n_batches} batches -> {OUT}")
