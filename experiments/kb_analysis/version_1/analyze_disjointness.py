"""ENUM-fragment disjointness / overlap analysis (version_1).

Measures how similar each KB fragment's `description` is to every other
fragment's `description`, so we can flag fragments the agent could confuse
(i.e. pick *different* fragments across N runs for the same question).

Input  : version_1/feature-and-product-knowledge.local.csv  (pipe-delimited, Slovak)
Scope  : all 198 fragments, keyed by knowledgeId, text = description (per user).

Methods:
  - tfidf    : char-3-5-gram TF-IDF cosine -> surface/lexical overlap (copy-paste)
  - lsa      : word-1-2 TF-IDF + TruncatedSVD(100) cosine -> topical overlap
  - semantic : multilingual sentence-embedding cosine -> meaning-based overlap.
               Captures pairs that mean the same thing in *different words* — what a
               production semantic router actually conflates. The lexical tfidf/lsa
               methods are a lower bound; semantic lifts that bound. Requires
               `sentence-transformers` (+ `truststore`, so the model download works
               behind the Erste TLS-inspecting proxy). If unavailable, the script
               degrades gracefully to tfidf+lsa.

Outputs in version_1/analysis_outputs/:
  - disjointness_per_fragment.csv   one row / knowledgeId, nearest-neighbour stats
  - confusable_pairs.csv            undirected pairs ranked by similarity (candidates)
  - semantic_only_pairs.csv         pairs the lexical methods MISSED but semantic flags
  - exact_duplicate_groups.csv      fragments whose description is byte-identical
  - clusters.csv                    connected-component clusters at the overlap threshold
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

# Two corporate-network fixes so the HuggingFace model download works (user-space,
# no admin). See the analysis findings / Report 1 §Method for the full write-up:
#  1. truststore: make Python's `ssl` trust the OS keychain, where the Erste
#     "Proxy Certification Authority" CA already lives (curl trusts it; Python's
#     bundled certifi does not). Fixes CERTIFICATE_VERIFY_FAILED.
#  2. HF_HUB_DISABLE_XET: the new Xet downloader (`hf_xet`, a Rust binary) has its
#     OWN TLS stack that does NOT consult the keychain, so truststore can't reach it
#     and weight downloads hang on the inspecting proxy. Forcing classic HTTPS
#     downloads routes them back through Python's ssl, which truststore has patched.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Multilingual STS model — 50+ languages incl. Slovak, ~470 MB, 384-dim.
ST_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

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


def semantic_cos(texts: list[str], model_name: str = ST_MODEL) -> np.ndarray | None:
    """Meaning-based cosine via a multilingual sentence-embedding model.

    Returns None (so callers can fall back to tfidf+lsa) if sentence-transformers
    isn't installed or the model can't be loaded. Embeddings are L2-normalised, so
    the dot product is cosine similarity in [-1, 1].
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        print(f"  [semantic] sentence-transformers unavailable ({e}); "
              "falling back to tfidf+lsa only.")
        return None
    try:
        model = SentenceTransformer(model_name)
    except Exception as e:
        print(f"  [semantic] could not load '{model_name}' ({e}); "
              "falling back to tfidf+lsa only.")
        return None
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=False)
    return np.asarray(emb) @ np.asarray(emb).T


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
    print(f"Computing semantic cosine via '{ST_MODEL}' (downloads on first run) ...")
    Ss = semantic_cos(texts)
    has_sem = Ss is not None
    if has_sem:
        print(f"  [semantic] OK — {Ss.shape[0]}x{Ss.shape[1]} meaning-based matrix.")
    mats = [St, Sl] + ([Ss] if has_sem else [])
    for S in mats:
        np.fill_diagonal(S, -1.0)

    # --- per-fragment nearest-neighbour stats ---
    rows = []
    for i, kid in enumerate(ids):
        t5t, t5l = top5(St[i], ids), top5(Sl[i], ids)
        row = {
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
        }
        if has_sem:
            t5s = top5(Ss[i], ids)
            row.update({
                "max_semantic": t5s[0][1], "nearest_semantic": t5s[0][0],
                "n_semantic_ge_0.70": int(np.sum(Ss[i] >= 0.70)),
                "n_semantic_ge_0.80": int(np.sum(Ss[i] >= 0.80)),
                "n_semantic_ge_0.85": int(np.sum(Ss[i] >= 0.85)),
                "top5_semantic": "; ".join(f"{a}({s:.2f})" for a, s in t5s),
            })
        rows.append(row)
    per = pd.DataFrame(rows)
    max_cols = ["max_tfidf", "max_lsa"] + (["max_semantic"] if has_sem else [])
    per["max_any"] = per[max_cols].max(axis=1)
    per = per.sort_values("max_any", ascending=False)
    per.to_csv(OUT / "disjointness_per_fragment.csv", index=False)

    # --- undirected candidate pairs ---
    # A pair is a candidate if any method clears its threshold. The semantic
    # threshold (0.70) lets in "means the same, says it differently" pairs the
    # lexical methods (tfidf>=0.45 or lsa>=0.70) never see.
    SEM_CAND = 0.70
    pair_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            tf, ls = float(St[i, j]), float(Sl[i, j])
            ss = float(Ss[i, j]) if has_sem else float("nan")
            lex_cand = tf >= 0.45 or ls >= 0.70
            sem_cand = has_sem and ss >= SEM_CAND
            if lex_cand or sem_cand:
                sims = [tf, ls] + ([ss] if has_sem else [])
                pair_rows.append({
                    "id_a": ids[i], "id_b": ids[j],
                    "name_a": name_by[ids[i]], "name_b": name_by[ids[j]],
                    "type_a": type_by[ids[i]], "type_b": type_by[ids[j]],
                    "same_family": family(ids[i]) == family(ids[j]),
                    "tfidf": round(tf, 3), "lsa": round(ls, 3),
                    "semantic": round(ss, 3) if has_sem else "",
                    "max_sim": round(max(sims), 3),
                    "lexical_candidate": lex_cand,
                    "semantic_only": bool(sem_cand and not lex_cand),
                })
    pairs = pd.DataFrame(pair_rows).sort_values("max_sim", ascending=False)
    pairs.to_csv(OUT / "confusable_pairs.csv", index=False)

    # Pairs the lexical methods MISSED but semantic flags — these are exactly the
    # ones the "conservative estimate" caveat was about. Ranked by semantic cosine.
    if has_sem:
        sem_only = pairs[pairs.semantic_only].sort_values("semantic", ascending=False)
        sem_only.to_csv(OUT / "semantic_only_pairs.csv", index=False)

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
            strong = St[i, j] >= 0.60 or Sl[i, j] >= 0.85
            if has_sem:
                strong = strong or Ss[i, j] >= 0.80
            if strong:
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
    dist_cols = ["max_tfidf", "max_lsa"] + (["max_semantic"] if has_sem else [])
    print("\n=== max-similarity distribution (each fragment vs its nearest peer) ===")
    print(per[dist_cols].describe().round(3).to_string())

    print("\n=== Top 30 most-confusable pairs (by max across methods) ===")
    for _, r in pairs.head(30).iterrows():
        fam = "same-fam" if r.same_family else "CROSS-FAM"
        sem = f" sem={r.semantic:.2f}" if has_sem else ""
        flag = " [SEM-ONLY]" if (has_sem and r.semantic_only) else ""
        print(f"  tfidf={r.tfidf:.2f} lsa={r.lsa:.2f}{sem} [{fam}]{flag}  "
              f"{r.id_a}  <->  {r.id_b}")

    edge = "tfidf>=.60 or lsa>=.85" + (" or sem>=.80" if has_sem else "")
    print(f"\n=== Overlap clusters (>=2 fragments, edge: {edge}): {len(clusters)} ===")
    for ci, c in enumerate(clusters, 1):
        fams = sorted({family(k) for k in c})
        tag = "  [CROSS-FAMILY]" if len(fams) > 1 else ""
        print(f"  #{ci} (n={len(c)}){tag}: {' | '.join(c)}")

    n_overlap = int((per["max_any"] >= 0.80).sum())
    n_lex_dup = int((per["max_tfidf"] >= 0.85).sum())
    print(f"\nFragments with a peer at max_any>=0.80: {n_overlap}/{n}")
    print(f"Fragments with a lexical near-duplicate (tfidf>=0.85): {n_lex_dup}/{n}")
    print(f"Fragments with a topical near-twin (lsa>=0.70): "
          f"{int((per['max_lsa'] >= 0.70).sum())}/{n}")
    if has_sem:
        print("\n=== SEMANTIC (meaning-based) — the lower-bound lift ===")
        for thr in (0.70, 0.75, 0.80, 0.85):
            cnt = int((per["max_semantic"] >= thr).sum())
            print(f"  fragments with a semantic near-twin >= {thr:.2f}: {cnt}/{n}")
        n_sem_only = int(pairs.semantic_only.sum()) if "semantic_only" in pairs else 0
        n_lex_pairs = int(pairs.lexical_candidate.sum())
        print(f"  candidate pairs — lexical: {n_lex_pairs}, "
              f"semantic-only (lexically missed): {n_sem_only}, total: {len(pairs)}")
        print(f"  -> wrote semantic_only_pairs.csv ({n_sem_only} pairs to LLM-judge)")
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
