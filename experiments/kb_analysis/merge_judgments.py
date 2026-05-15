"""Merge sub-agent translation verdicts back into kb_findings.csv,
collect a single translation_judgments.jsonl, and emit summary stats."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
batches = OUT / "judge_batches"

# 1) gather all verdicts
verdicts: dict[str, dict] = {}
parse_errors = []
for vp in sorted(batches.glob("verdicts_*.jsonl")):
    with open(vp, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                parse_errors.append((vp.name, ln, str(e), line[:120]))
                continue
            kid = obj.get("knowledgeId")
            if not kid:
                parse_errors.append((vp.name, ln, "missing knowledgeId", line[:120]))
                continue
            verdicts[kid] = obj

print(f"Loaded {len(verdicts)} verdicts from {len(list(batches.glob('verdicts_*.jsonl')))} batch files")
if parse_errors:
    print(f"WARN {len(parse_errors)} parse errors:")
    for name, ln, err, snip in parse_errors[:10]:
        print(f"  {name}:{ln}  {err}  {snip!r}")

# 2) write a consolidated translation_judgments.jsonl
with open(OUT / "translation_judgments.jsonl", "w", encoding="utf-8") as f:
    for kid in sorted(verdicts):
        f.write(json.dumps(verdicts[kid], ensure_ascii=False) + "\n")

# 3) merge into kb_findings.csv
findings_path = OUT / "kb_findings.csv"
df = pd.read_csv(findings_path)


def get(d: dict, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def short_issues(v: dict) -> str:
    issues = v.get("issues") or []
    return " | ".join(f"[{i.get('type', '?')}] {i.get('detail', '')}".strip() for i in issues)


df["judge_faithfulness"] = df["knowledgeId"].map(lambda k: get(verdicts.get(k, {}), "faithfulness"))
df["judge_completeness"] = df["knowledgeId"].map(lambda k: get(verdicts.get(k, {}), "completeness"))
df["judge_consistent_terminology"] = df["knowledgeId"].map(lambda k: get(verdicts.get(k, {}), "consistent_terminology"))
df["judge_verdict"] = df["knowledgeId"].map(lambda k: get(verdicts.get(k, {}), "verdict"))
df["judge_summary"] = df["knowledgeId"].map(lambda k: get(verdicts.get(k, {}), "summary"))
df["judge_issue_count"] = df["knowledgeId"].map(lambda k: len(get(verdicts.get(k, {}), "issues", default=[]) or []))
df["judge_issues"] = df["knowledgeId"].map(lambda k: short_issues(verdicts.get(k, {})))
df["judge_issue_types"] = df["knowledgeId"].map(
    lambda k: ", ".join(sorted({i.get("type", "?") for i in (get(verdicts.get(k, {}), "issues", default=[]) or [])}))
)

df.to_csv(findings_path, index=False)

# 4) summary stats
print("\n=== Translation judge summary ===")
verdict_counts = Counter(df["judge_verdict"].dropna())
for v, c in verdict_counts.most_common():
    print(f"  verdict={v:<14s} {c}")
print(f"\n  faithfulness  mean={df['judge_faithfulness'].mean():.2f}  "
      f"median={df['judge_faithfulness'].median()}  "
      f"min={df['judge_faithfulness'].min()}  max={df['judge_faithfulness'].max()}")
print(f"  completeness  mean={df['judge_completeness'].mean():.2f}  "
      f"median={df['judge_completeness'].median()}  "
      f"min={df['judge_completeness'].min()}  max={df['judge_completeness'].max()}")
print(f"  consistent_terminology=False: {(df['judge_consistent_terminology'] == False).sum()}")
print(f"  rows with >= 1 issue:         {(df['judge_issue_count'] > 0).sum()}")

issue_type_counts: Counter = Counter()
for k, v in verdicts.items():
    for i in v.get("issues") or []:
        issue_type_counts[i.get("type", "?")] += 1
print("\n  issue type counts:")
for t, c in issue_type_counts.most_common():
    print(f"    {t:<16s} {c}")

# Top problem cases
print("\n=== 15 lowest-faithfulness rows ===")
worst = df.dropna(subset=["judge_faithfulness"]).sort_values(
    ["judge_faithfulness", "judge_completeness"]
).head(15)[
    ["knowledgeId", "judge_faithfulness", "judge_completeness", "judge_verdict", "judge_summary"]
]
print(worst.to_string(index=False))

print("\n=== 15 lowest-completeness rows ===")
worst = df.dropna(subset=["judge_completeness"]).sort_values(
    ["judge_completeness", "judge_faithfulness"]
).head(15)[
    ["knowledgeId", "judge_faithfulness", "judge_completeness", "judge_verdict", "judge_summary"]
]
print(worst.to_string(index=False))

# Major issues with full summaries
print("\n=== Rows with verdict=major_issues ===")
major = df[df["judge_verdict"] == "major_issues"][
    ["knowledgeId", "judge_faithfulness", "judge_completeness", "judge_summary"]
]
print(major.to_string(index=False))

print(f"\nMerged judgments into {findings_path}")
print(f"Wrote consolidated {OUT / 'translation_judgments.jsonl'}")
