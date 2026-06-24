"""ENUM-fragment disjointness / overlap analysis (version_1).

Measures how similar each KB fragment's `description` is to every other
fragment's `description`, so we can flag fragments the agent could confuse
(i.e. pick *different* fragments across N runs for the same question).

Input  : version_1/feature-and-product-knowledge.local.csv  (pipe-delimited, Slovak)
Scope  : all 198 fragments, keyed by knowledgeId, text = description (per user).

Methods (corpus-derived — HuggingFace is TLS-blocked in this env, see findings):
  - tfidf : char-3-5-gram TF-IDF cosine  -> surface/lexical overlap (copy-paste)
  - lsa   : word-1-2 TF-IDF + TruncatedSVD(100) cosine -> topical overlap

Outputs in version_1/analysis_outputs/:
  - disjointness_per_fragment.csv   one row / knowledgeId, nearest-neighbour stats
  - confusable_pairs.csv            undirected pairs ranked by similarity (candidates)
  - exact_duplicate_groups.csv      fragments whose description is byte-identical
  - clusters.csv                    connected-component clusters at the overlap threshold
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

HERE = Path(__file__).resolve().parent
KB_CSV = HERE / "feature-and-product-knowledge.local.csv"
OUT = HERE / "analysis_outputs"
OUT.mkdir(parents=True, exist_ok=True)

PREFIX = "feature-and-product-knowledge."


def _unescape(s: str) -> str:
    return s.replace("\\n", "\n").replace("\\t", "\t")


def load_kb() -> pd.DataFrame:
    df = pd.read_csv(KB_CSV, sep="|", dtype=str).fillna("")
    df.columns = [c.replace(PREFIX, "") for c in df.columns]
    for col in ("description", "summary", "notInScope", "knowledgeName"):
        df[col] = df[col].map(_unescape)
    df["knowledgeId"] = df["knowledgeId"].str.strip()
    return df


def family(kid: str) -> str:
    """Product family = token before '@', else the id itself."""
    return kid.split("@", 1)[0] if "@" in kid else kid


# ---- similarity -----------------------------------------------------------
def tfidf_cos(texts: list[str]) -> np.ndarray:
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1)
    X = vec.fit_transform(texts)
    return cosine_similarity(X)


def lsa_cos(texts: list[str], n_components: int = 100) -> np.ndarray:
    vec = TfidfVectorizer(
        analyzer="word", lowercase=True, min_df=2,
        token_pattern=r"(?u)\b\w[\w']+\b", ngram_range=(1, 2),
    )
    X = vec.fit_transform(texts)
    k = min(n_components, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = normalize(svd.fit_transform(X))
    return Z @ Z.T


def top5(sim_row: np.ndarray, ids: list[str]) -> list[tuple[str, float]]:
    order = np.argsort(-sim_row)[:5]
    return [(ids[j], float(sim_row[j])) for j in order]


def main():
    df = load_kb()
    ids = df["knowledgeId"].tolist()
    texts = df["description"].tolist()
    name_by = dict(zip(df["knowledgeId"], df["knowledgeName"]))
    type_by = dict(zip(df["knowledgeId"], df["knowledgeType"]))
    n = len(ids)
    print(f"Loaded {n} fragments; mean description length "
          f"{int(np.mean([len(t) for t in texts]))} chars.")

    # --- exact duplicates (normalised whitespace) ---
    def norm_txt(t: str) -> str:
        return re.sub(r"\s+", " ", t).strip().lower()

    dup_groups: dict[str, list[str]] = {}
    for kid, t in zip(ids, texts):
        dup_groups.setdefault(norm_txt(t), []).append(kid)
    dups = {k: v for k, v in dup_groups.items() if len(v) > 1}
    dup_rows = []
    for grp in dups.values():
        for kid in grp:
            dup_rows.append({"group_size": len(grp), "knowledgeId": kid,
                             "group_members": "; ".join(grp),
                             "char_len": len(df.loc[df.knowledgeId == kid, "description"].iloc[0])})
    pd.DataFrame(dup_rows).to_csv(OUT / "exact_duplicate_groups.csv", index=False)
    print(f"\nExact-duplicate description groups: {len(dups)}")
    for grp in dups.values():
        print("   ", " == ".join(grp))

    # --- similarity matrices ---
    print("\nComputing TF-IDF (char) and LSA (word+SVD) cosine ...")
    St = tfidf_cos(texts)
    Sl = lsa_cos(texts)
    for S in (St, Sl):
        np.fill_diagonal(S, -1.0)

    # --- per-fragment nearest-neighbour stats ---
    rows = []
    for i, kid in enumerate(ids):
        t5t, t5l = top5(St[i], ids), top5(Sl[i], ids)
        rows.append({
            "knowledgeId": kid,
            "knowledgeName": name_by[kid],
            "knowledgeType": type_by[kid],
            "family": family(kid),
            "char_len": len(texts[i]),
            "max_tfidf": t5t[0][1], "nearest_tfidf": t5t[0][0],
            "max_lsa": t5l[0][1], "nearest_lsa": t5l[0][0],
            "n_tfidf_ge_0.60": int(np.sum(St[i] >= 0.60)),
            "n_tfidf_ge_0.85": int(np.sum(St[i] >= 0.85)),
            "n_lsa_ge_0.80": int(np.sum(Sl[i] >= 0.80)),
            "n_lsa_ge_0.95": int(np.sum(Sl[i] >= 0.95)),
            "top5_tfidf": "; ".join(f"{a}({s:.2f})" for a, s in t5t),
            "top5_lsa": "; ".join(f"{a}({s:.2f})" for a, s in t5l),
        })
    per = pd.DataFrame(rows)
    per["max_any"] = per[["max_tfidf", "max_lsa"]].max(axis=1)
    per = per.sort_values("max_any", ascending=False)
    per.to_csv(OUT / "disjointness_per_fragment.csv", index=False)

    # --- undirected candidate pairs ---
    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            tf, ls = St[i, j], Sl[i, j]
            if tf >= 0.45 or ls >= 0.70:  # candidate threshold
                pair_rows.append({
                    "id_a": ids[i], "id_b": ids[j],
                    "name_a": name_by[ids[i]], "name_b": name_by[ids[j]],
                    "type_a": type_by[ids[i]], "type_b": type_by[ids[j]],
                    "same_family": family(ids[i]) == family(ids[j]),
                    "tfidf": round(float(tf), 3), "lsa": round(float(ls), 3),
                    "max_sim": round(float(max(tf, ls)), 3),
                })
    pairs = pd.DataFrame(pair_rows).sort_values("max_sim", ascending=False)
    pairs.to_csv(OUT / "confusable_pairs.csv", index=False)

    # --- clusters (connected components at overlap threshold) ---
    # edge if char-lexical >= 0.60 OR topical >= 0.85 (strong overlap)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if St[i, j] >= 0.60 or Sl[i, j] >= 0.85:
                union(i, j)
    comp: dict[int, list[str]] = {}
    for i, kid in enumerate(ids):
        comp.setdefault(find(i), []).append(kid)
    clusters = [c for c in comp.values() if len(c) > 1]
    clusters.sort(key=len, reverse=True)
    crow = []
    for ci, c in enumerate(clusters, 1):
        fams = sorted({family(k) for k in c})
        crow.append({"cluster": ci, "size": len(c),
                     "families": "; ".join(fams),
                     "cross_family": len(fams) > 1,
                     "members": " | ".join(c)})
    pd.DataFrame(crow).to_csv(OUT / "clusters.csv", index=False)

    # ---- console summary ----
    print("\n=== max-similarity distribution (each fragment vs its nearest peer) ===")
    print(per[["max_tfidf", "max_lsa"]].describe().round(3).to_string())

    print("\n=== Top 30 most-confusable pairs (by max of the two methods) ===")
    for _, r in pairs.head(30).iterrows():
        fam = "same-fam" if r.same_family else "CROSS-FAM"
        print(f"  tfidf={r.tfidf:.2f} lsa={r.lsa:.2f} [{fam}]  {r.id_a}  <->  {r.id_b}")

    print(f"\n=== Overlap clusters (>=2 fragments, edge: tfidf>=.60 or lsa>=.85): "
          f"{len(clusters)} ===")
    for ci, c in enumerate(clusters, 1):
        fams = sorted({family(k) for k in c})
        tag = "  [CROSS-FAMILY]" if len(fams) > 1 else ""
        print(f"  #{ci} (n={len(c)}){tag}: {' | '.join(c)}")

    n_overlap = int((per["max_any"] >= 0.80).sum())
    n_lex_dup = int((per["max_tfidf"] >= 0.85).sum())
    print(f"\nFragments with a peer at max_any>=0.80: {n_overlap}/{n}")
    print(f"Fragments with a lexical near-duplicate (tfidf>=0.85): {n_lex_dup}/{n}")
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
