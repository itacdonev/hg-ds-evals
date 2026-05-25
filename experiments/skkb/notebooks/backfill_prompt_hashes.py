#!/usr/bin/env python3
"""Backfill prompt-hash columns + prompt sidecar from a raw MLflow traces
JSONL into an existing eval checkpoint CSV — without touching anything else.

Use this when the trace import was run under the old prompt extractor
(every per-trace hash came out as ``d41d8cd98f``, the MD5 of the empty
string) and you've already run the judge against the resulting
checkpoint. Re-importing from scratch would discard those judge scores;
this script updates ONLY:

  * ``main_agent_prompt_hash``           (if the column exists)
  * ``daily_banking_agent_prompt_hash``  (if the column exists)

… joining the JSONL to the checkpoint on ``trace_id``. Every other
column — judge weighted_avg, per-dimension scores, agent_response,
expert_score, … — is left byte-identical. The script also writes the
``prompt_{run_id}.json`` sidecar next to the checkpoint, which is what
populates the report's Prompts tab.

Before any write the script copies the checkpoint to
``<checkpoint>.bak`` (skipped if a ``.bak`` already exists, so re-runs
are safe). The CSV write is atomic — written to a sibling temp file
and renamed into place.

Usage (from this directory):

    ../../../.venv/bin/python backfill_prompt_hashes.py \\
        --checkpoint checkpoints/uat/evals_skkb_..._<run_id>.csv \\
        --traces     ../input/traces_<run_id>.jsonl

The run_id is parsed from the traces filename. Override with
``--run-id`` if your file is named differently.

Pairs with the orchestrator workaround in
``hg_ds_evals/preprocessing/traces.py::_get_agent_chat_system_prompt``
(see the ``TODO(peter-pr)`` note in that file). Re-run this script
whenever you want to re-extract hashes against a newer copy of the
JSONL after the upstream PR lands.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

# ─── Repo bootstrap ─────────────────────────────────────────────────────
# Locate the repo root by walking up from this file until we hit one with
# an `hg_ds_evals` package directory. Lets the script run from any cwd.
_THIS = Path(__file__).resolve()
_REPO = next(
    (p for p in [_THIS.parent, *_THIS.parents] if (p / "hg_ds_evals").is_dir()),
    None,
)
if _REPO is None:
    sys.exit("could not locate repo root (expected an hg_ds_evals/ directory above this script)")
sys.path.insert(0, str(_REPO))

from hg_ds_evals.preprocessing.traces import (  # noqa: E402
    _build_span_children,
    extract_agent_prompt_hashes,
    extract_agent_system_prompts,
    _extract_tool_registry,
)


HASH_COLS = (
    "main_agent_prompt_hash",
    "daily_banking_agent_prompt_hash",
)


def _run_id_from_traces_path(p: Path) -> str:
    """Parse <run_id> out of ``traces_<run_id>.jsonl``."""
    stem = p.stem
    if not stem.startswith("traces_"):
        raise ValueError(
            f"can't infer run_id from {p.name!r}; "
            f"expected traces_<run_id>.jsonl — pass --run-id to override"
        )
    return stem[len("traces_"):]


def _iter_jsonl(path: Path):
    """Yield parsed JSON objects from a .jsonl file, skipping blank lines."""
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _scan_traces(traces_path: Path) -> tuple[dict, str, str, dict]:
    """Walk every trace once and compute:

      * ``per_trace[trace_id]`` → per-trace hash dict for the CSV update.
      * ``main_prompt`` / ``dba_prompt`` → first non-empty system prompt
        of each agent encountered (for the sidecar; runs are uniform).
      * ``tool_descriptions`` → union by tool name (for the sidecar).
    """
    per_trace: dict[str, dict[str, str]] = {}
    main_prompt = dba_prompt = ""
    tool_descriptions: dict[str, str] = {}

    for row in _iter_jsonl(traces_path):
        info = row.get("info") or {}
        data = row.get("data") or {}
        tid = info.get("trace_id")
        if not tid:
            continue
        spans = data.get("spans") or []
        if not spans:
            continue
        children = _build_span_children(spans)

        # Per-trace hashes — what we'll write back to the checkpoint.
        hashes = extract_agent_prompt_hashes(spans, children)
        per_trace[tid] = {
            "main_agent_prompt_hash": hashes["main_agent_prompt_hash"],
            "daily_banking_agent_prompt_hash":
                hashes["daily_banking_agent_prompt_hash"],
        }

        # Sidecar aggregates — collected from whichever trace first
        # surfaces a non-empty value. Runs are uniform in practice, so
        # "first match" matches "all matches".
        if not main_prompt or not dba_prompt:
            prompts = extract_agent_system_prompts(spans, children)
            if not main_prompt:
                main_prompt = prompts["main_agent_system_prompt"]
            if not dba_prompt:
                dba_prompt = prompts["daily_banking_agent_system_prompt"]
        _, descriptions, _ = _extract_tool_registry(spans)
        for name, desc in descriptions.items():
            tool_descriptions.setdefault(name, desc)

    return per_trace, main_prompt, dba_prompt, tool_descriptions


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` atomically.

    Goes via a sibling NamedTemporaryFile + ``os.replace`` so a crash
    mid-write can't corrupt the checkpoint."""
    fd, tmp = tempfile.mkstemp(
        prefix=path.stem + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        # Best-effort cleanup of the temp file; original is still intact.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="path to the eval checkpoint CSV to update in-place")
    ap.add_argument("--traces", type=Path, required=True,
                    help="path to the raw mlflow traces .jsonl")
    ap.add_argument("--run-id", default=None,
                    help="override run_id (otherwise parsed from traces filename)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    ckp = args.checkpoint.expanduser().resolve()
    traces = args.traces.expanduser().resolve()
    if not ckp.exists():
        sys.exit(f"checkpoint not found: {ckp}")
    if not traces.exists():
        sys.exit(f"traces file not found: {traces}")

    run_id = args.run_id or _run_id_from_traces_path(traces)
    print(f"checkpoint: {ckp}")
    print(f"traces:     {traces}")
    print(f"run_id:     {run_id}")

    # ─── Load checkpoint ────────────────────────────────────────────────
    # keep_default_na=False + na_filter=False keeps string columns as-is
    # (no "" → NaN conversion), so round-tripping the unaffected columns
    # produces semantically identical output.
    df = pd.read_csv(ckp, keep_default_na=False, na_filter=False, dtype=str)
    print(f"checkpoint rows: {len(df)}  cols: {len(df.columns)}")
    if "trace_id" not in df.columns:
        sys.exit("checkpoint has no trace_id column; cannot join to traces")
    cols_to_update = [c for c in HASH_COLS if c in df.columns]
    cols_missing = [c for c in HASH_COLS if c not in df.columns]
    if cols_missing:
        print(f"checkpoint is missing these hash columns (will be skipped, not added): {cols_missing}")
    if not cols_to_update:
        sys.exit("none of the hash columns exist in the checkpoint — nothing to update")

    # ─── Walk traces ────────────────────────────────────────────────────
    per_trace, main_prompt, dba_prompt, tool_descriptions = _scan_traces(traces)
    print(f"traces parsed: {len(per_trace)}")
    print(f"sidecar prompts: main={len(main_prompt)} chars, "
          f"dba={len(dba_prompt)} chars, tools={len(tool_descriptions)} entries")

    # ─── Apply per-row updates ──────────────────────────────────────────
    # Pull the existing values so we can report a before/after summary
    # and skip rows whose hash is already correct (lets the run be
    # idempotent without spurious "all rows updated" lines).
    before = {col: df[col].copy() for col in cols_to_update}
    updated = changed_per_col = {col: 0 for col in cols_to_update}.copy()
    missed = 0
    for i, tid in enumerate(df["trace_id"].astype(str)):
        entry = per_trace.get(tid)
        if entry is None:
            missed += 1
            continue
        for col in cols_to_update:
            new_val = entry[col]
            if df.at[i, col] != new_val:
                df.at[i, col] = new_val
                changed_per_col[col] += 1

    for col in cols_to_update:
        distinct_after = df[col].nunique()
        empty_count = int((df[col] == "d41d8cd98f").sum())
        print(f"  {col:<40} updated={changed_per_col[col]:>4}  "
              f"distinct_after={distinct_after}  still_empty_hash={empty_count}")
    print(f"trace_id rows with no matching jsonl entry (skipped): {missed}")

    if args.dry_run:
        print("--dry-run: not writing")
        return

    if all(c == 0 for c in changed_per_col.values()):
        print("nothing to update — checkpoint already has these hashes")
    else:
        # ─── Backup + atomic write ──────────────────────────────────────
        bak = ckp.with_suffix(ckp.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(ckp, bak)
            print(f"backup written: {bak.name}")
        else:
            print(f"backup already exists, leaving alone: {bak.name}")
        _atomic_write_csv(df, ckp)
        print(f"checkpoint updated in place: {ckp.name}")

    # ─── Sidecar JSON (always write — cheap, idempotent) ────────────────
    sidecar = ckp.parent / f"prompt_{run_id}.json"
    sidecar.write_text(
        json.dumps({
            "run_id": run_id,
            "main_agent_system_prompt": main_prompt,
            "daily_banking_agent_system_prompt": dba_prompt,
            "tool_descriptions": tool_descriptions,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"sidecar written: {sidecar}")


if __name__ == "__main__":
    main()
