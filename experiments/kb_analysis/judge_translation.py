"""LLM-based translation judge for the EN<->SK kb.description pairs.

For each knowledgeId present in BOTH languages, asks Claude (Haiku 4.5) to
compare the SK source text and the EN translation. The model is instructed to
use ONLY the two texts (no outside knowledge).

Output: experiments/kb_analysis/translation_judgments.jsonl
  one line per id: {knowledgeId, faithfulness, completeness, summary, issues:[...]}
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from anthropic import Anthropic

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "input"
OUT = ROOT / "experiments" / "kb_analysis"
OUT.mkdir(parents=True, exist_ok=True)

EN_CSV = INPUT / "KB_GAI_SK_EN_2026-04-20_14h16_phase_1_2.csv"
SK_CSV = INPUT / "KB_GAI_SK_SK_2026-04-20_14h16_phase_1_2.csv"
RESULT = OUT / "translation_judgments.jsonl"

MODEL = "claude-haiku-4-5-20251001"

SYSTEM = """You are a bilingual Slovak-English translation reviewer.
You compare a Slovak source text (SK) to its English translation (EN) for a
banking knowledge-base entry. Your judgement uses ONLY the two texts you are
given. Do not use any outside knowledge of banking products, fees, regulations,
or this specific bank. If a fact appears in EN but not in SK, flag it as
"added"; if a fact appears in SK but not in EN, flag it as "missing"; if a
specific value differs (number, percentage, currency amount, age, date, URL,
phone number, product name), flag it as "value_mismatch". You also report
overall faithfulness (semantic correctness) and completeness (no missing
content) on a 1-5 scale. Be terse. Output JSON only, no prose, no fences."""

USER_TEMPLATE = """Compare these two descriptions for the same KB entry.

knowledgeId: {kid}

SK (source):
<sk>
{sk}
</sk>

EN (translation):
<en>
{en}
</en>

Return strictly this JSON schema (no markdown fences, no commentary):
{{
  "faithfulness": <int 1-5>,            // 5 = fully faithful, 1 = wrong/mistranslated
  "completeness": <int 1-5>,            // 5 = all SK content covered, 1 = major omissions
  "consistent_terminology": <bool>,     // does EN use consistent terms across the text
  "verdict": "ok" | "minor_issues" | "major_issues",
  "summary": "<= 1 sentence overall description of the EN vs SK alignment",
  "issues": [
    {{
      "type": "missing" | "added" | "value_mismatch" | "terminology" | "structure" | "ambiguous",
      "detail": "<= 1 short sentence; quote the SK and EN snippets if relevant"
    }}
  ]
}}
If there are no issues, return "issues": []."""


def load_csv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="|"):
            out[row["kb.knowledgeId"].strip()] = row["kb.description"].replace("\\n", "\n")
    return out


def parse_json_loose(text: str) -> dict:
    """Tolerate the model wrapping JSON in ```json fences or returning prose around it."""
    text = text.strip()
    # strip code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # find outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


def judge_one(client: Anthropic, kid: str, sk: str, en: str) -> dict:
    last_err = None
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1500,
                temperature=0.0,
                system=SYSTEM,
                messages=[{"role": "user", "content": USER_TEMPLATE.format(kid=kid, sk=sk, en=en)}],
            )
            text = resp.content[0].text
            obj = parse_json_loose(text)
            obj["knowledgeId"] = kid
            obj["_model"] = MODEL
            return obj
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    return {"knowledgeId": kid, "_error": f"{type(last_err).__name__}: {last_err}"}


def main():
    en = load_csv(EN_CSV)
    sk = load_csv(SK_CSV)
    common = sorted(set(en) & set(sk))
    print(f"Judging {len(common)} EN<->SK pairs with {MODEL}...")

    # resume support: skip ids already in the result file
    done: set[str] = set()
    if RESULT.exists():
        with open(RESULT, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "knowledgeId" in obj and "_error" not in obj:
                        done.add(obj["knowledgeId"])
                except json.JSONDecodeError:
                    pass
    todo = [k for k in common if k not in done]
    print(f"Already done: {len(done)}.  Remaining: {len(todo)}.")

    if not todo:
        print("Nothing to do.")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(2)

    client = Anthropic()
    t0 = time.time()
    n_done = 0
    with open(RESULT, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(judge_one, client, k, sk[k], en[k]): k for k in todo}
        for fut in as_completed(futs):
            kid = futs[fut]
            obj = fut.result()
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            out.flush()
            n_done += 1
            if n_done % 10 == 0 or n_done == len(todo):
                elapsed = time.time() - t0
                rate = n_done / max(0.1, elapsed)
                eta = (len(todo) - n_done) / max(0.1, rate)
                v = obj.get("verdict", obj.get("_error", "?"))
                print(f"[{n_done:>3}/{len(todo)}]  {kid:<46s} -> {v}    ({elapsed:.0f}s, ETA {eta:.0f}s)")

    print(f"Wrote {RESULT}")


if __name__ == "__main__":
    main()
