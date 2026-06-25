"""Re-build judge batches for the 40 lexical pairs using ONLY what the agent sees.

The ai-orchestrator reranker selects on the fragment `id`/name + `description`.
`summary` is loaded-but-unused and `notInScope` is not loaded at all. The original
lexical judging pass evaluated `notInScope` as a disambiguator, which downgraded
several pairs' determinism risk on a signal that does not exist at selection time.
This rebuilds the batches with id + knowledgeName + description ONLY, so the
re-judge reflects live behaviour.

Writes analysis_outputs/desconly_batches/batch_<k>.json + result files later.
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "analysis_outputs", "desconly_batches")
os.makedirs(OUT, exist_ok=True)
PER_BATCH = 10

kb = pd.read_csv(os.path.join(HERE, "feature-and-product-knowledge.local.csv"),
                 sep="|", dtype=str).fillna("")
kb.columns = [c.replace("feature-and-product-knowledge.", "") for c in kb.columns]
kb["knowledgeId"] = kb["knowledgeId"].str.strip()
for col in ("description", "knowledgeName"):
    kb[col] = kb[col].map(lambda s: s.replace("\\n", "\n").replace("\\t", "\t"))
by_id = {r.knowledgeId: r for _, r in kb.iterrows()}


def frag(kid: str) -> dict:
    r = by_id.get(kid)
    if r is None:
        return {"knowledgeId": kid, "MISSING": True}
    # ONLY the fields the reranker actually sees — no summary, no notInScope.
    return {"knowledgeId": kid, "knowledgeName": r.knowledgeName, "description": r.description}


judged = pd.read_csv(os.path.join(HERE, "analysis_outputs", "confusable_pairs_judged.csv"))
pairs = []
for i, r in judged.iterrows():
    pairs.append({
        "pair_index": int(r.pair_index) if "pair_index" in judged.columns else int(i),
        "id_a": r.id_a, "id_b": r.id_b,
        "tfidf": float(r.tfidf), "lsa": float(r.lsa),
        "same_family": bool(r.same_family),
        "prior_risk": r.determinism_risk,          # for reference only; judge independently
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
