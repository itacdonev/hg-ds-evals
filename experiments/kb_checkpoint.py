"""Checkpoint CSV loading helpers for KB-pipeline (SKKB / CZKB) evaluation outputs."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

# Checkpoint rows contain large JSON blobs (>128 KB) that exceed the default
# csv module field-size limit (131072).  Raise it at import time so every
# consumer (notebook, standalone script) benefits automatically.
csv.field_size_limit(sys.maxsize)


def _is_score(value: str) -> bool:
    return str(value).strip() in {"0", "1", "2", "0.0", "1.0", "2.0"}


def _repair_row(row: list[str], header: list[str]) -> tuple[list[str], str, bool, str]:
    """Return a rectangular row plus load metadata.

    The checkpoint writer appends one-row DataFrames. If a later result has
    a different set/order of keys than the first row, pandas writes a ragged
    CSV row. This loader keeps the analysis usable and records what happened.
    """
    expected = len(header)
    warning = ""
    is_error = False
    error_message = ""

    if len(row) == expected:
        return row, warning, is_error, error_message

    if len(row) == 2:
        # Observed shape for a failed judge parse: test_case_id + exception text.
        is_error = True
        error_message = row[1]
        warning = f"short checkpoint row with {len(row)} fields; treated as judge error"
        return [row[0]] + [""] * (expected - 1), warning, is_error, error_message

    if len(row) == expected + 1 and "query_clarity_score" in header:
        score_idx = header.index("query_clarity_score")
        if row[score_idx] == "" and _is_score(row[score_idx + 1]):
            warning = f"removed extra empty field before {header[score_idx]}"
            return row[:score_idx] + row[score_idx + 1 :], warning, is_error, error_message

    if len(row) < expected:
        warning = f"short checkpoint row with {len(row)} fields; padded to {expected}"
        return row + [""] * (expected - len(row)), warning, is_error, error_message

    warning = f"long checkpoint row with {len(row)} fields; truncated to {expected}"
    return row[:expected], warning, is_error, error_message


def read_checkpoint_csv(path: str | Path) -> pd.DataFrame:
    """Read a checkpoint CSV, repairing known ragged-row shapes.

    The checkpoint writer uses ``pandas.to_csv(..., quoting=csv.QUOTE_ALL)``
    (see ``hg_ds_evals.common.utils.append_to_checkpoint``), so the round-trip
    path is ``pd.read_csv(..., quoting=csv.QUOTE_ALL, dtype=str)``. We try that
    first; if it fails (ragged rows from a partially-corrupted checkpoint), we
    fall back to the per-row csv.reader repair logic.

    Returns string-valued checkpoint columns plus metadata columns:
    `_csv_record_number`, `_csv_load_warning`, `error`, and `error_message`.
    """
    path = Path(path)

    try:
        df = pd.read_csv(path, quoting=csv.QUOTE_ALL, dtype=str,
                          keep_default_na=False, na_values=[""])
    except Exception:
        df = None

    if df is not None:
        df = df.fillna("")
        df["_csv_record_number"] = range(2, len(df) + 2)
        if "_csv_load_warning" not in df.columns:
            df["_csv_load_warning"] = ""
        if "error" not in df.columns:
            df["error"] = False
        if "error_message" not in df.columns:
            df["error_message"] = ""
        return df

    # Fallback: manually repair ragged rows we know about.
    records: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame()

        for record_number, row in enumerate(reader, start=2):
            repaired, warning, is_error, error_message = _repair_row(row, header)
            record: dict[str, object] = dict(zip(header, repaired, strict=False))
            record["_csv_record_number"] = record_number
            record["_csv_load_warning"] = warning
            if "error" not in record or record["error"] in {"", None}:
                record["error"] = is_error
            if "error_message" not in record or record["error_message"] in {"", None}:
                record["error_message"] = error_message
            records.append(record)

    return pd.DataFrame.from_records(records)
