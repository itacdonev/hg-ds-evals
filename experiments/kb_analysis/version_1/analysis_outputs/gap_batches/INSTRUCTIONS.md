# KB content-gap + test-question analysis — subagent instructions

You analyze a batch of Slovak knowledge-base (KB) fragments for the **Hey George**
conversational banking AI (Slovenská sporiteľňa). You read Slovak fluently. There are
NO ground-truth labels — judge against the fragment text itself and realistic user behaviour.

## Inputs
- Your batch file (a JSON object keyed by `knowledgeId`) is named in the task that spawned you.
  Each value has: `knowledgeName`, `knowledgeType`, `knowledgeTag`, `isFeature`, `family`,
  `summary` (a contents-index), `notInScope` (what the fragment explicitly does NOT cover),
  `description` (the answer body the agent uses), `existing_test_questions` (list of `{q_sk,q_en}`
  already in the test set for this topic), `n_existing_test_questions`.

## How the agent works (context for your judgements)
The agent matches a user question against the fragment `description` (full text) and answers
from it. `summary`/`notInScope` can help it route IF they are clear and consistent with the
description. Determinism = the agent picking the SAME fragment and giving the SAME answer for
the same question across N runs.

## For EACH fragment in your batch, produce two things

### A. completeness_issues  (may be empty list if the fragment is solid)
Each issue = an object:
- `type`: one of
  - `missing_info`        — a fact a user will plausibly ask for is simply absent
  - `underspecified`      — present but too vague to answer deterministically (e.g. "limits apply" with no number)
  - `ambiguous`           — wording lets the agent answer two different ways
  - `internal_contradiction` — the fragment states two conflicting facts (e.g. 4-digit vs 5-digit)
  - `scope_mismatch`      — description/name/summary/notInScope disagree about what this fragment covers
  - `outdated_or_unverifiable` — figure/claim that looks stale or that the agent cannot ground
- `detail`: SPECIFIC, in English. Quote the Slovak phrase if useful.
- `determinism_impact`: `HIGH` | `MED` | `LOW`  (HIGH = agent will likely answer the same question differently across runs, or cannot answer at all)
- `fix_or_missing_area`: concrete fix, OR — if you can't know the exact missing fact but you DO know the topic area that's missing — name that area (e.g. "exact daily ATM withdrawal limit in EUR").

Only report REAL issues. A complete, clear fragment should have an empty list. Do not invent problems.

### B. new_questions  (0–3 per fragment; QUALITY over quantity)
Additional high-probability user questions NOT already covered by `existing_test_questions`.
**Prioritise gap-exposing questions** — natural things a user would ask that the KB CANNOT fully
answer. Skip a fragment (0 questions) if it is well covered and well specified.
Each = an object:
- `question_sk`: natural Slovak user phrasing (how a real customer types/speaks)
- `question_en`: faithful English translation
- `answerable_by_kb`: `yes` | `partial` | `no`  (can THIS KB, as written, fully answer it?)
- `missing_area`: if `partial`/`no`, name the missing information area (else "")
- `priority`: `HIGH` | `MED` | `LOW`  (HIGH = very common question AND/OR exposes an important gap)
- `rationale`: 1 sentence, English — why a user asks this / why it matters.

## Output
Write a JSON array (one object per fragment, IN BATCH ORDER) to a file next to your batch file,
named `<batchstem>_result.json` (e.g. if your batch is `savings.json`, write `savings_result.json`).
Each object:
```
{
  "knowledgeId": "...",
  "completeness_issues": [ ... ],
  "new_questions": [ ... ],
  "overall_note": "one line, English, on this fragment's quality/coverage"
}
```
Validate it parses as JSON. Then return a TERSE final message: fragments processed, count of
HIGH-impact completeness issues, count of `answerable_by_kb != yes` new questions, and a one-line
list of the most important gaps you found. Your final message is data for the orchestrator, not a
user-facing reply.
