"""KB analysis (text-only — does NOT use any precomputed embeddings).

Inputs (only kb.knowledgeId and kb.description are used; kb.summary is dropped):
- input/KB_GAI_SK_EN_*.csv
- input/KB_GAI_SK_SK_*.csv

Outputs in experiments/kb_analysis/:
- kb_clean.jsonl                   {knowledgeId, description_en, description_sk}
- kb_findings.csv                  one row per knowledgeId with all analysis columns
- nearest_neighbors_tfidf_en.csv   per-id top-5 lexical neighbors (TF-IDF, EN)
- nearest_neighbors_tfidf_sk.csv   per-id top-5 lexical neighbors (TF-IDF, SK)
- nearest_neighbors_lsa_en.csv     per-id top-5 topical neighbors (TF-IDF+SVD/LSA, EN)
- nearest_neighbors_lsa_sk.csv     per-id top-5 topical neighbors (TF-IDF+SVD/LSA, SK)

Metrics produced per knowledgeId:
  Disjointness (within-language):
    max_sim_other_{tfidf,lsa}_{en,sk}
    nearest_id_{tfidf,lsa}_{en,sk}
    n_neighbors_ge_{0.50,0.70,0.85}_{tfidf,lsa}_{en,sk}
    top5_neighbors_{tfidf,lsa}_{en,sk}
  Translation consistency (EN vs SK on the same id):
    char_len_en, char_len_sk, len_ratio_en_over_sk
    bullet_count_{en,sk}, bullet_count_diff
    header_count_{en,sk}, header_count_diff
    url_count_{en,sk}, url_count_diff
    numbers_{en,sk}, numbers_jaccard, numbers_only_{en,sk}
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "input"
OUT = ROOT / "experiments" / "kb_analysis"
OUT.mkdir(parents=True, exist_ok=True)

EN_CSV = INPUT / "KB_GAI_SK_EN_2026-04-20_14h16_phase_1_2.csv"
SK_CSV = INPUT / "KB_GAI_SK_SK_2026-04-20_14h16_phase_1_2.csv"

# CSV cells contain literal "\n" (backslash + n), not real newlines.
# Convert to real newlines for any analysis that cares about lines.
def _unescape(s: str) -> str:
    return s.replace("\\n", "\n").replace("\\t", "\t")


def load_csv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            kid = row["kb.knowledgeId"].strip()
            out[kid] = _unescape(row["kb.description"])
    return out


# ---- structural feature extraction --------------------------------------
NUMBER_RE = re.compile(r"\d{1,3}(?:[.,  ]\d{3})*(?:[.,]\d+)?|\d+(?:[.,]\d+)?")
URL_RE = re.compile(r"https?://[^\s)]+")
HEADER_RE = re.compile(r"\*\*([^*]+)\*\*")
BULLET_RE = re.compile(r"(?m)^\s*[-*]\s+")
PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")


def _to_float(token: str) -> float | None:
    t = token.replace(" ", "").replace(" ", "").strip()
    if "," in t and "." in t:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")
        else:
            t = t.replace(",", "")
    elif "," in t:
        head, _, tail = t.rpartition(",")
        if 1 <= len(tail) <= 2:
            t = head + "." + tail
        else:
            t = t.replace(",", "")
    elif "." in t:
        head, _, tail = t.rpartition(".")
        if len(tail) == 3 and head and "." not in head:
            t = head + tail
    try:
        return float(t)
    except ValueError:
        return None


def normalize_numbers(text: str) -> set[float]:
    out: set[float] = set()
    for tok in NUMBER_RE.findall(text):
        v = _to_float(tok)
        if v is not None:
            out.add(v)
    return out


def structural_features(text: str) -> dict:
    return {
        "char_len": len(text),
        "word_count": len(text.split()),
        "line_count": text.count("\n") + 1,
        "bullet_count": len(BULLET_RE.findall(text)),
        "header_count": len(HEADER_RE.findall(text)),
        "url_count": len(URL_RE.findall(text)),
        "number_set": normalize_numbers(text),
        "percent_count": len(PERCENT_RE.findall(text)),
    }


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


# ---- disjointness (per language, two methods) ----------------------------
def neighbors(
    sim: np.ndarray, ids: list[str], lang: str, method: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Given a square similarity matrix (self-sim already removed), return
    (per_id_df, neighbor_long_df)."""
    per_id = []
    nn_rows = []
    for i, kid in enumerate(ids):
        row = sim[i]
        order = np.argsort(-row)[:5]
        top5 = [(ids[j], float(row[j])) for j in order]
        per_id.append({
            "knowledgeId": kid,
            f"max_sim_other_{method}_{lang}": top5[0][1],
            f"nearest_id_{method}_{lang}": top5[0][0],
            f"mean_top3_sim_{method}_{lang}": float(np.mean([s for _, s in top5[:3]])),
            f"n_neighbors_ge_0.50_{method}_{lang}": int(np.sum(row >= 0.50)),
            f"n_neighbors_ge_0.70_{method}_{lang}": int(np.sum(row >= 0.70)),
            f"n_neighbors_ge_0.85_{method}_{lang}": int(np.sum(row >= 0.85)),
            f"top5_neighbors_{method}_{lang}": "; ".join(f"{a}({s:.3f})" for a, s in top5),
        })
        for rank, (other, s) in enumerate(top5, 1):
            nn_rows.append({
                "knowledgeId": kid, "rank": rank, "neighbor": other,
                "cosine": s, "lang": lang, "method": method,
            })
    return pd.DataFrame(per_id), pd.DataFrame(nn_rows)


def tfidf_sim(texts: list[str]) -> np.ndarray:
    """Character-3-5-gram TF-IDF, lowercased — language-agnostic, robust to
    morphology differences in Slovak."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True, min_df=1)
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -1.0)
    return sim


def lsa_sim(texts: list[str], n_components: int = 100) -> np.ndarray:
    """Topical similarity via word-level TF-IDF + Truncated SVD (a.k.a. LSA).
    Fully derived from the supplied corpus — no external model or knowledge.
    Captures co-occurrence-based semantic overlap that pure char-ngram cosine
    misses (e.g. paraphrases that share few characters but the same topic)."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize

    n = len(texts)
    word_vec = TfidfVectorizer(
        analyzer="word", lowercase=True, min_df=2,
        token_pattern=r"(?u)\b\w[\w']+\b",
        ngram_range=(1, 2),
    )
    X = word_vec.fit_transform(texts)
    k = min(n_components, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=0)
    Z = svd.fit_transform(X)
    Z = normalize(Z)
    sim = Z @ Z.T
    np.fill_diagonal(sim, -1.0)
    return sim


# ---- main ----------------------------------------------------------------
def main():
    en_desc = load_csv(EN_CSV)
    sk_desc = load_csv(SK_CSV)

    only_en = sorted(set(en_desc) - set(sk_desc))
    only_sk = sorted(set(sk_desc) - set(en_desc))
    common = sorted(set(en_desc) & set(sk_desc))
    print(f"EN ids: {len(en_desc)}  SK ids: {len(sk_desc)}  common: {len(common)}")
    print(f"Only in EN: {only_en}")
    print(f"Only in SK: {only_sk}")

    # --- write the clean jsonl (no embeddings, no summary) ---------------
    all_ids = sorted(set(en_desc) | set(sk_desc))
    with open(OUT / "kb_clean.jsonl", "w", encoding="utf-8") as f:
        for kid in all_ids:
            f.write(json.dumps({
                "knowledgeId": kid,
                "description_en": en_desc.get(kid),
                "description_sk": sk_desc.get(kid),
            }, ensure_ascii=False) + "\n")

    # --- disjointness, per language, two methods --------------------------
    en_ids = list(en_desc.keys())
    sk_ids = list(sk_desc.keys())
    en_texts = [en_desc[k] for k in en_ids]
    sk_texts = [sk_desc[k] for k in sk_ids]

    print("Computing TF-IDF similarities (EN, SK)...")
    sim_tfidf_en = tfidf_sim(en_texts)
    sim_tfidf_sk = tfidf_sim(sk_texts)

    print("Computing LSA (TF-IDF + Truncated SVD) similarities (EN, SK)...")
    sim_lsa_en = lsa_sim(en_texts)
    sim_lsa_sk = lsa_sim(sk_texts)

    dj_tfidf_en, nn_tfidf_en = neighbors(sim_tfidf_en, en_ids, "en", "tfidf")
    dj_tfidf_sk, nn_tfidf_sk = neighbors(sim_tfidf_sk, sk_ids, "sk", "tfidf")
    dj_lsa_en, nn_lsa_en = neighbors(sim_lsa_en, en_ids, "en", "lsa")
    dj_lsa_sk, nn_lsa_sk = neighbors(sim_lsa_sk, sk_ids, "sk", "lsa")

    nn_tfidf_en.to_csv(OUT / "nearest_neighbors_tfidf_en.csv", index=False)
    nn_tfidf_sk.to_csv(OUT / "nearest_neighbors_tfidf_sk.csv", index=False)
    nn_lsa_en.to_csv(OUT / "nearest_neighbors_lsa_en.csv", index=False)
    nn_lsa_sk.to_csv(OUT / "nearest_neighbors_lsa_sk.csv", index=False)

    # --- per-id structural translation features --------------------------
    rows = []
    for kid in all_ids:
        en_t = en_desc.get(kid)
        sk_t = sk_desc.get(kid)
        row = {
            "knowledgeId": kid,
            "description_en": en_t,
            "description_sk": sk_t,
            "present_in_en": en_t is not None,
            "present_in_sk": sk_t is not None,
        }
        if en_t and sk_t:
            fe = structural_features(en_t)
            fs = structural_features(sk_t)

            def _fmt(v: float) -> str:
                return str(int(v)) if v.is_integer() else f"{v:g}"

            row.update({
                "char_len_en": fe["char_len"],
                "char_len_sk": fs["char_len"],
                "len_ratio_en_over_sk": fe["char_len"] / max(1, fs["char_len"]),
                "word_count_en": fe["word_count"],
                "word_count_sk": fs["word_count"],
                "bullet_count_en": fe["bullet_count"],
                "bullet_count_sk": fs["bullet_count"],
                "bullet_count_diff": fe["bullet_count"] - fs["bullet_count"],
                "header_count_en": fe["header_count"],
                "header_count_sk": fs["header_count"],
                "header_count_diff": fe["header_count"] - fs["header_count"],
                "url_count_en": fe["url_count"],
                "url_count_sk": fs["url_count"],
                "url_count_diff": fe["url_count"] - fs["url_count"],
                "percent_count_en": fe["percent_count"],
                "percent_count_sk": fs["percent_count"],
                "numbers_en": ", ".join(_fmt(v) for v in sorted(fe["number_set"])),
                "numbers_sk": ", ".join(_fmt(v) for v in sorted(fs["number_set"])),
                "numbers_jaccard": jaccard(fe["number_set"], fs["number_set"]),
                "numbers_only_en": ", ".join(_fmt(v) for v in sorted(fe["number_set"] - fs["number_set"])),
                "numbers_only_sk": ", ".join(_fmt(v) for v in sorted(fs["number_set"] - fe["number_set"])),
            })
        rows.append(row)

    findings = pd.DataFrame(rows)

    # merge in disjointness columns
    for df in (dj_tfidf_en, dj_lsa_en, dj_tfidf_sk, dj_lsa_sk):
        findings = findings.merge(df, on="knowledgeId", how="left")

    # threshold flags (purely descriptive)
    findings["flag_only_in_one_lang"] = ~(findings["present_in_en"] & findings["present_in_sk"])
    findings["flag_len_ratio_off"] = (
        (findings["len_ratio_en_over_sk"].lt(0.70)) | (findings["len_ratio_en_over_sk"].gt(1.40))
    )
    findings["flag_numbers_mismatch"] = findings["numbers_jaccard"].lt(0.80)
    findings["flag_bullet_mismatch"] = findings["bullet_count_diff"].abs().gt(1)
    findings["flag_header_mismatch"] = findings["header_count_diff"].abs().gt(2)
    findings["flag_url_mismatch"] = findings["url_count_diff"].abs().gt(0)

    findings["flag_topical_overlap_en"] = findings["max_sim_other_lsa_en"].ge(0.80)
    findings["flag_topical_overlap_sk"] = findings["max_sim_other_lsa_sk"].ge(0.80)
    findings["flag_topical_near_duplicate_en"] = findings["max_sim_other_lsa_en"].ge(0.95)
    findings["flag_topical_near_duplicate_sk"] = findings["max_sim_other_lsa_sk"].ge(0.95)
    findings["flag_lexical_overlap_en"] = findings["max_sim_other_tfidf_en"].ge(0.60)
    findings["flag_lexical_overlap_sk"] = findings["max_sim_other_tfidf_sk"].ge(0.60)
    findings["flag_lexical_near_duplicate_en"] = findings["max_sim_other_tfidf_en"].ge(0.85)
    findings["flag_lexical_near_duplicate_sk"] = findings["max_sim_other_tfidf_sk"].ge(0.85)

    front = ["knowledgeId", "description_en", "description_sk"]
    findings = findings[front + [c for c in findings.columns if c not in front]]
    findings.to_csv(OUT / "kb_findings.csv", index=False)

    # ---- console summary ------------------------------------------------
    print("\n=== TF-IDF disjointness ===")
    print("EN max_sim_other_tfidf_en:")
    print(dj_tfidf_en["max_sim_other_tfidf_en"].describe().to_string())
    print("\nSK max_sim_other_tfidf_sk:")
    print(dj_tfidf_sk["max_sim_other_tfidf_sk"].describe().to_string())

    print("\n=== LSA (TF-IDF + Truncated SVD) disjointness ===")
    print("EN max_sim_other_lsa_en:")
    print(dj_lsa_en["max_sim_other_lsa_en"].describe().to_string())
    print("\nSK max_sim_other_lsa_sk:")
    print(dj_lsa_sk["max_sim_other_lsa_sk"].describe().to_string())

    for label, col, df in [
        ("Top 15 most-overlapping EN (LSA)", "max_sim_other_lsa_en", dj_lsa_en),
        ("Top 15 most-overlapping SK (LSA)", "max_sim_other_lsa_sk", dj_lsa_sk),
        ("Top 15 most-overlapping EN (TF-IDF char)", "max_sim_other_tfidf_en", dj_tfidf_en),
        ("Top 15 most-overlapping SK (TF-IDF char)", "max_sim_other_tfidf_sk", dj_tfidf_sk),
    ]:
        print(f"\n=== {label} ===")
        nearest_col = col.replace("max_sim_other", "nearest_id")
        for _, r in df.sort_values(col, ascending=False).head(15).iterrows():
            print(f"{r[col]:.3f}  {r['knowledgeId']:<46s} -> {r[nearest_col]}")

    print("\nWrote:")
    for name in ("kb_clean.jsonl", "kb_findings.csv",
                 "nearest_neighbors_tfidf_en.csv", "nearest_neighbors_tfidf_sk.csv",
                 "nearest_neighbors_lsa_en.csv", "nearest_neighbors_lsa_sk.csv"):
        print(f"  {OUT/name}")


if __name__ == "__main__":
    main()
