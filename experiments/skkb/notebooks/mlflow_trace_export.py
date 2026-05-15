"""Utilities for exporting MLflow traces from a run to JSON."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
import json
import math
from pathlib import Path
from typing import Any

import mlflow


def export_traces_to_json(
    run_id: str,
    output_path: str | Path | None = None,
    *,
    tracking_uri: str | None = "databricks",
    locations: Sequence[str] | None = None,
    max_results: int | None = None,
    include_spans: bool = True,
    order_by: Sequence[str] | None = None,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> Path:
    """Export all MLflow traces associated with ``run_id`` to a JSON file.

    This is intended for Databricks clusters, where ``tracking_uri="databricks"``
    and output paths such as ``/dbfs/tmp/traces.json`` are valid.

    Args:
        run_id: MLflow run ID whose associated traces should be exported.
        output_path: Destination JSON path. ``dbfs:/...`` is converted to
            ``/dbfs/...`` for local file writes on Databricks clusters. When
            omitted, the file is written as ``mlflow_traces_<run_id>.json`` in
            the current working directory.
        tracking_uri: MLflow tracking URI. Use ``None`` to keep the current
            process tracking URI unchanged.
        locations: Optional MLflow trace locations. For experiment-backed
            traces, pass experiment IDs. If omitted, the run's experiment ID is
            used automatically.
        max_results: Maximum number of traces to export. ``None`` exports every
            matching trace returned by MLflow pagination.
        include_spans: Include full span payloads. Keep this ``True`` for a full
            trace export.
        order_by: Optional MLflow trace ordering clauses.
        indent: JSON indentation. Use ``None`` for compact JSON.
        ensure_ascii: Passed to ``json.dump``.

    Returns:
        The local filesystem path that was written.
    """

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    resolved_locations = (
        list(locations) if locations is not None else [_get_run_experiment_id(run_id)]
    )
    output_file = _resolve_output_path(output_path, run_id)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    traces = mlflow.search_traces(
        run_id=run_id,
        locations=resolved_locations,
        max_results=max_results,
        order_by=list(order_by) if order_by is not None else None,
        include_spans=include_spans,
        return_type="list",
    )

    payload = {
        "run_id": run_id,
        "tracking_uri": mlflow.get_tracking_uri(),
        "locations": resolved_locations,
        "include_spans": include_spans,
        "trace_count": len(traces),
        "exported_at": datetime.now(UTC).isoformat(),
        "traces": [_trace_to_dict(trace) for trace in traces],
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=indent,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        )
        file.write("\n")

    return output_file


def _get_run_experiment_id(run_id: str) -> str:
    return mlflow.get_run(run_id).info.experiment_id


def _resolve_output_path(output_path: str | Path | None, run_id: str) -> Path:
    if output_path is None:
        return Path(f"mlflow_traces_{_safe_filename(run_id)}.json")

    raw_path = str(output_path)
    if raw_path.startswith("dbfs:/"):
        return Path("/dbfs") / raw_path.removeprefix("dbfs:/").lstrip("/")
    return Path(raw_path)


def _safe_filename(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_" for char in value
    )
    return safe or "run"


def _trace_to_dict(trace: Any) -> dict[str, Any]:
    """Convert an MLflow Trace object to a JSON-compatible dictionary."""

    if hasattr(trace, "to_json"):
        try:
            loaded = json.loads(trace.to_json())
            if isinstance(loaded, dict):
                converted = _json_safe(loaded)
                if isinstance(converted, dict):
                    return converted
        except Exception:
            pass

    if hasattr(trace, "to_dict"):
        try:
            converted = _json_safe(trace.to_dict())
            if isinstance(converted, dict):
                return converted
        except Exception:
            pass

    if hasattr(trace, "to_pandas_dataframe_row"):
        converted = _json_safe(trace.to_pandas_dataframe_row())
        if isinstance(converted, dict):
            return converted

    return {"repr": repr(trace)}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, datetime | date):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]

    if hasattr(value, "to_dictionary"):
        return _json_safe(value.to_dictionary())

    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())

    return str(value)
