"""MLflow trace helpers for notebook-friendly OTEL table generation.

This vendors the narrow in-memory path from the trace-transformer work in
``ai-data-science`` so notebooks can fetch traces from MLflow and materialize
schema-faithful OTEL spans, logs, and metrics DataFrames without writing large
JSON payloads to disk first.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
import json
import math
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import types as T

OTEL_SPANS_TABLE = "mlflow_experiment_trace_otel_spans"
OTEL_LOGS_TABLE = "mlflow_experiment_trace_otel_logs"
OTEL_METRICS_TABLE = "mlflow_experiment_trace_otel_metrics"

_STRING_MAP = T.MapType(T.StringType(), T.StringType())

_RESOURCE_SCHEMA = T.StructType(
    [
        T.StructField("attributes", _STRING_MAP, True),
        T.StructField("dropped_attributes_count", T.IntegerType(), True),
    ]
)

_INSTRUMENTATION_SCOPE_SCHEMA = T.StructType(
    [
        T.StructField("name", T.StringType(), True),
        T.StructField("version", T.StringType(), True),
        T.StructField("attributes", _STRING_MAP, True),
        T.StructField("dropped_attributes_count", T.IntegerType(), True),
    ]
)

_EXEMPLAR_SCHEMA = T.ArrayType(
    T.StructType(
        [
            T.StructField("time_unix_nano", T.LongType(), True),
            T.StructField("value", T.DoubleType(), True),
            T.StructField("span_id", T.StringType(), True),
            T.StructField("trace_id", T.StringType(), True),
            T.StructField("filtered_attributes", _STRING_MAP, True),
        ]
    )
)

SPANS_SCHEMA = T.StructType(
    [
        T.StructField("trace_id", T.StringType(), True),
        T.StructField("span_id", T.StringType(), True),
        T.StructField("trace_state", T.StringType(), True),
        T.StructField("parent_span_id", T.StringType(), True),
        T.StructField("flags", T.IntegerType(), True),
        T.StructField("name", T.StringType(), True),
        T.StructField("kind", T.StringType(), True),
        T.StructField("start_time_unix_nano", T.LongType(), True),
        T.StructField("end_time_unix_nano", T.LongType(), True),
        T.StructField("attributes", _STRING_MAP, True),
        T.StructField("dropped_attributes_count", T.IntegerType(), True),
        T.StructField(
            "events",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("time_unix_nano", T.LongType(), True),
                        T.StructField("name", T.StringType(), True),
                        T.StructField("attributes", _STRING_MAP, True),
                        T.StructField("dropped_attributes_count", T.IntegerType(), True),
                    ]
                )
            ),
            True,
        ),
        T.StructField("dropped_events_count", T.IntegerType(), True),
        T.StructField(
            "links",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("trace_id", T.StringType(), True),
                        T.StructField("span_id", T.StringType(), True),
                        T.StructField("trace_state", T.StringType(), True),
                        T.StructField("attributes", _STRING_MAP, True),
                        T.StructField("dropped_attributes_count", T.IntegerType(), True),
                        T.StructField("flags", T.IntegerType(), True),
                    ]
                )
            ),
            True,
        ),
        T.StructField("dropped_links_count", T.IntegerType(), True),
        T.StructField(
            "status",
            T.StructType(
                [
                    T.StructField("message", T.StringType(), True),
                    T.StructField("code", T.StringType(), True),
                ]
            ),
            True,
        ),
        T.StructField("resource", _RESOURCE_SCHEMA, True),
        T.StructField("resource_schema_url", T.StringType(), True),
        T.StructField("instrumentation_scope", _INSTRUMENTATION_SCOPE_SCHEMA, True),
        T.StructField("span_schema_url", T.StringType(), True),
    ]
)

LOGS_SCHEMA = T.StructType(
    [
        T.StructField("event_name", T.StringType(), True),
        T.StructField("trace_id", T.StringType(), True),
        T.StructField("span_id", T.StringType(), True),
        T.StructField("time_unix_nano", T.LongType(), True),
        T.StructField("observed_time_unix_nano", T.LongType(), True),
        T.StructField("severity_number", T.StringType(), True),
        T.StructField("severity_text", T.StringType(), True),
        T.StructField("body", T.StringType(), True),
        T.StructField("attributes", _STRING_MAP, True),
        T.StructField("dropped_attributes_count", T.IntegerType(), True),
        T.StructField("flags", T.IntegerType(), True),
        T.StructField("resource", _RESOURCE_SCHEMA, True),
        T.StructField("resource_schema_url", T.StringType(), True),
        T.StructField("instrumentation_scope", _INSTRUMENTATION_SCOPE_SCHEMA, True),
        T.StructField("log_schema_url", T.StringType(), True),
    ]
)

METRICS_SCHEMA = T.StructType(
    [
        T.StructField("name", T.StringType(), True),
        T.StructField("description", T.StringType(), True),
        T.StructField("unit", T.StringType(), True),
        T.StructField("metric_type", T.StringType(), True),
        T.StructField(
            "gauge",
            T.StructType(
                [
                    T.StructField("start_time_unix_nano", T.LongType(), True),
                    T.StructField("time_unix_nano", T.LongType(), True),
                    T.StructField("value", T.DoubleType(), True),
                    T.StructField("exemplars", _EXEMPLAR_SCHEMA, True),
                    T.StructField("attributes", _STRING_MAP, True),
                    T.StructField("flags", T.IntegerType(), True),
                ]
            ),
            True,
        ),
        T.StructField(
            "sum",
            T.StructType(
                [
                    T.StructField("start_time_unix_nano", T.LongType(), True),
                    T.StructField("time_unix_nano", T.LongType(), True),
                    T.StructField("value", T.DoubleType(), True),
                    T.StructField("exemplars", _EXEMPLAR_SCHEMA, True),
                    T.StructField("attributes", _STRING_MAP, True),
                    T.StructField("flags", T.IntegerType(), True),
                    T.StructField("aggregation_temporality", T.StringType(), True),
                    T.StructField("is_monotonic", T.BooleanType(), True),
                ]
            ),
            True,
        ),
        T.StructField(
            "histogram",
            T.StructType(
                [
                    T.StructField("start_time_unix_nano", T.LongType(), True),
                    T.StructField("time_unix_nano", T.LongType(), True),
                    T.StructField("count", T.LongType(), True),
                    T.StructField("sum", T.DoubleType(), True),
                    T.StructField("bucket_counts", T.ArrayType(T.LongType()), True),
                    T.StructField("explicit_bounds", T.ArrayType(T.DoubleType()), True),
                    T.StructField("exemplars", _EXEMPLAR_SCHEMA, True),
                    T.StructField("attributes", _STRING_MAP, True),
                    T.StructField("flags", T.IntegerType(), True),
                    T.StructField("min", T.DoubleType(), True),
                    T.StructField("max", T.DoubleType(), True),
                    T.StructField("aggregation_temporality", T.StringType(), True),
                ]
            ),
            True,
        ),
        T.StructField(
            "exponential_histogram",
            T.StructType(
                [
                    T.StructField("attributes", _STRING_MAP, True),
                    T.StructField("start_time_unix_nano", T.LongType(), True),
                    T.StructField("time_unix_nano", T.LongType(), True),
                    T.StructField("count", T.LongType(), True),
                    T.StructField("sum", T.DoubleType(), True),
                    T.StructField("scale", T.IntegerType(), True),
                    T.StructField("zero_count", T.LongType(), True),
                    T.StructField(
                        "positive_bucket",
                        T.StructType(
                            [
                                T.StructField("offset", T.IntegerType(), True),
                                T.StructField("bucket_counts", T.ArrayType(T.LongType()), True),
                            ]
                        ),
                        True,
                    ),
                    T.StructField(
                        "negative_bucket",
                        T.StructType(
                            [
                                T.StructField("offset", T.IntegerType(), True),
                                T.StructField("bucket_counts", T.ArrayType(T.LongType()), True),
                            ]
                        ),
                        True,
                    ),
                    T.StructField("flags", T.IntegerType(), True),
                    T.StructField("exemplars", _EXEMPLAR_SCHEMA, True),
                    T.StructField("min", T.DoubleType(), True),
                    T.StructField("max", T.DoubleType(), True),
                    T.StructField("zero_threshold", T.DoubleType(), True),
                    T.StructField("aggregation_temporality", T.StringType(), True),
                ]
            ),
            True,
        ),
        T.StructField(
            "summary",
            T.StructType(
                [
                    T.StructField("start_time_unix_nano", T.LongType(), True),
                    T.StructField("time_unix_nano", T.LongType(), True),
                    T.StructField("count", T.LongType(), True),
                    T.StructField("sum", T.DoubleType(), True),
                    T.StructField(
                        "quantile_values",
                        T.ArrayType(
                            T.StructType(
                                [
                                    T.StructField("quantile", T.DoubleType(), True),
                                    T.StructField("value", T.DoubleType(), True),
                                ]
                            )
                        ),
                        True,
                    ),
                    T.StructField("attributes", _STRING_MAP, True),
                    T.StructField("flags", T.IntegerType(), True),
                ]
            ),
            True,
        ),
        T.StructField("metadata", _STRING_MAP, True),
        T.StructField("resource", _RESOURCE_SCHEMA, True),
        T.StructField("resource_schema_url", T.StringType(), True),
        T.StructField("instrumentation_scope", _INSTRUMENTATION_SCOPE_SCHEMA, True),
        T.StructField("metric_schema_url", T.StringType(), True),
    ]
)


@dataclass
class OTelTables:
    """Container for the three canonical OTEL DataFrames."""

    spans: DataFrame
    logs: DataFrame
    metrics: DataFrame

    def as_dict(self) -> dict[str, DataFrame]:
        return {
            OTEL_SPANS_TABLE: self.spans,
            OTEL_LOGS_TABLE: self.logs,
            OTEL_METRICS_TABLE: self.metrics,
        }


def read_mlflow_traces(
    *,
    run_id: str | None = None,
    locations: Sequence[str] | None = None,
    tracking_uri: str | None = "databricks",
    max_results: int | None = None,
    include_spans: bool = True,
    order_by: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Read MLflow traces into JSON-safe dictionaries.

    When ``locations`` is omitted, the run's experiment ID is used so the call
    stays on the supported ``locations`` API surface.
    """

    try:
        import mlflow
    except ImportError as exc:
        raise ImportError(
            "mlflow is required to read traces. Install mlflow in the notebook environment."
        ) from exc

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    resolved_locations = _resolve_locations(
        mlflow=mlflow,
        run_id=run_id,
        locations=locations,
    )

    traces = mlflow.search_traces(
        run_id=run_id,
        locations=resolved_locations,
        max_results=max_results,
        order_by=list(order_by) if order_by is not None else None,
        include_spans=include_spans,
        return_type="list",
    )

    client = None
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    except Exception:
        client = None

    serialized: list[dict[str, Any]] = []
    for trace in traces:
        hydrated_trace = _hydrate_trace(client, trace)
        serialized.append(_serialize_mlflow_trace(hydrated_trace, tracking_uri=mlflow.get_tracking_uri()))
    return serialized


def load_mlflow_traces_to_otel_tables(
    spark: SparkSession,
    *,
    run_id: str | None = None,
    locations: Sequence[str] | None = None,
    tracking_uri: str | None = "databricks",
    max_results: int | None = None,
    include_spans: bool = True,
    order_by: Sequence[str] | None = None,
) -> OTelTables:
    """Fetch MLflow traces and materialize OTEL bronze-style DataFrames."""

    traces = read_mlflow_traces(
        run_id=run_id,
        locations=locations,
        tracking_uri=tracking_uri,
        max_results=max_results,
        include_spans=include_spans,
        order_by=order_by,
    )
    return traces_to_otel_tables(spark=spark, traces=traces)


def traces_to_otel_tables(
    spark: SparkSession,
    traces: Sequence[dict[str, Any]],
) -> OTelTables:
    """Create OTEL spans/logs/metrics DataFrames from serialized MLflow traces."""

    span_rows: list[dict[str, Any]] = []
    log_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for trace in traces:
        trace_id = _to_str(trace.get("trace_id"))
        trace_start_ns = _trace_start_ns(trace)
        resource = _default_resource(trace)
        scope = _default_scope(trace)

        spans = trace.get("spans_flat") if isinstance(trace.get("spans_flat"), list) else []
        for span in spans:
            span_rows.append(
                {
                    "trace_id": trace_id,
                    "span_id": _to_str(span.get("span_id")),
                    "trace_state": "",
                    "parent_span_id": _to_parent_span_id(span.get("parent_span_id")),
                    "flags": 1,
                    "name": _to_str(span.get("name")),
                    "kind": _derive_span_kind(span),
                    "start_time_unix_nano": _to_int(span.get("start_time_ns")) or 0,
                    "end_time_unix_nano": _to_int(span.get("end_time_ns"))
                    or _to_int(span.get("start_time_ns"))
                    or 0,
                    "attributes": _stringify_map(span.get("attributes")),
                    "dropped_attributes_count": 0,
                    "events": _normalize_events(span.get("events")),
                    "dropped_events_count": 0,
                    "links": [],
                    "dropped_links_count": 0,
                    "status": _normalize_status_struct(span.get("status")),
                    "resource": resource,
                    "resource_schema_url": "",
                    "instrumentation_scope": scope,
                    "span_schema_url": "",
                }
            )

            metric_rows.extend(
                _derive_span_metrics(
                    trace_id=trace_id,
                    span=span,
                    resource=resource,
                    scope=scope,
                )
            )

        log_rows.extend(
            _build_trace_metadata_logs(
                trace=trace,
                trace_id=trace_id,
                trace_start_ns=trace_start_ns,
                resource=resource,
                scope=scope,
            )
        )

    spans_df = spark.createDataFrame(span_rows, SPANS_SCHEMA)
    logs_df = spark.createDataFrame(log_rows, LOGS_SCHEMA)
    metrics_df = spark.createDataFrame(metric_rows, METRICS_SCHEMA)
    return OTelTables(spans=spans_df, logs=logs_df, metrics=metrics_df)


def _resolve_locations(
    *,
    mlflow: Any,
    run_id: str | None,
    locations: Sequence[str] | None,
) -> list[str]:
    if locations is not None:
        resolved = [str(location) for location in locations if str(location)]
        if resolved:
            return resolved

    if run_id:
        return [mlflow.get_run(run_id).info.experiment_id]

    raise ValueError("Provide locations or run_id so traces can be resolved.")


def _build_trace_metadata_logs(
    trace: dict[str, Any],
    trace_id: str,
    trace_start_ns: int,
    resource: dict[str, Any],
    scope: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    client_request_id = _to_optional_str(trace.get("client_request_id"))
    trace_metadata = _stringify_map(trace.get("trace_metadata"))

    metadata_body = json.dumps(
        {
            "client_request_id": client_request_id,
            "trace_metadata": trace_metadata,
        },
        sort_keys=True,
    )

    rows.append(
        _base_log_row(
            event_name="genai.trace.metadata",
            trace_id=trace_id,
            span_id="",
            time_unix_nano=trace_start_ns,
            body=metadata_body,
            attributes={},
            resource=resource,
            scope=scope,
        )
    )

    tags = _stringify_map(trace.get("trace_tags"))
    for idx, key in enumerate(sorted(tags.keys())):
        rows.append(
            _base_log_row(
                event_name="genai.trace.tag",
                trace_id=trace_id,
                span_id="",
                time_unix_nano=trace_start_ns + idx + 1,
                body=json.dumps({"key": key, "value": tags[key]}, sort_keys=True),
                attributes={},
                resource=resource,
                scope=scope,
            )
        )

    assessments = trace.get("assessments") if isinstance(trace.get("assessments"), list) else []
    for idx, assessment in enumerate(assessments):
        if not isinstance(assessment, dict):
            continue

        assessment_id = _to_str(assessment.get("assessment_id"))
        rows.append(
            _base_log_row(
                event_name="genai.assessments.snapshot",
                trace_id=trace_id,
                span_id=_to_str(assessment.get("span_id")),
                time_unix_nano=trace_start_ns + len(tags) + idx + 1,
                body=json.dumps(
                    {
                        "assessment_id": assessment_id,
                        "trace_id": _to_str(assessment.get("trace_id") or trace_id),
                        "assessment_name": _to_str(assessment.get("name")),
                        "source": _json_dumps_or_none(assessment.get("source")),
                        "create_time": _to_int(assessment.get("create_time_ms")),
                        "last_update_time": _to_int(assessment.get("last_update_time_ms")),
                        "expectation": _json_dumps_or_none(assessment.get("expectation")),
                        "feedback": _json_dumps_or_none(assessment.get("feedback")),
                        "rationale": _to_optional_str(assessment.get("rationale")),
                        "metadata": _json_dumps_or_none(assessment.get("metadata")),
                        "span_id": _to_optional_str(assessment.get("span_id")),
                        "overrides": _to_optional_str(assessment.get("overrides")),
                        "valid": _to_optional_bool_str(assessment.get("valid")),
                    },
                    sort_keys=True,
                ),
                attributes={"assessment_id": assessment_id, "deleted": "false"},
                resource=resource,
                scope=scope,
            )
        )

    return rows


def _derive_span_metrics(
    *,
    trace_id: str,
    span: dict[str, Any],
    resource: dict[str, Any],
    scope: dict[str, Any],
) -> list[dict[str, Any]]:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
    if not isinstance(attrs, dict):
        return []

    start_ns = _to_int(span.get("start_time_ns")) or 0
    end_ns = _to_int(span.get("end_time_ns")) or start_ns
    model_name = _to_optional_str(attrs.get("gen_ai.response.model"))

    token_candidates = {
        "input": _coerce_token(
            attrs.get("gen_ai.usage.input_tokens")
            or _nested_value(attrs.get("mlflow.chat.tokenUsage"), "input_tokens")
            or _nested_value(attrs.get("mlflow.chat.tokenUsage"), "prompt_tokens")
            or _nested_value(attrs.get("mlflow.spanOutputs"), "usage_metadata.input_tokens")
            or _nested_value(attrs.get("mlflow.spanOutputs"), "response_metadata.prompt_tokens")
        ),
        "output": _coerce_token(
            attrs.get("gen_ai.usage.output_tokens")
            or _nested_value(attrs.get("mlflow.chat.tokenUsage"), "output_tokens")
            or _nested_value(attrs.get("mlflow.chat.tokenUsage"), "completion_tokens")
            or _nested_value(attrs.get("mlflow.spanOutputs"), "usage_metadata.output_tokens")
            or _nested_value(attrs.get("mlflow.spanOutputs"), "response_metadata.completion_tokens")
        ),
        "total": _coerce_token(
            attrs.get("gen_ai.usage.total_tokens")
            or _nested_value(attrs.get("mlflow.chat.tokenUsage"), "total_tokens")
            or _nested_value(attrs.get("mlflow.spanOutputs"), "usage_metadata.total_tokens")
            or _nested_value(attrs.get("mlflow.spanOutputs"), "response_metadata.total_tokens")
        ),
    }

    if token_candidates["total"] is None and (
        token_candidates["input"] is not None or token_candidates["output"] is not None
    ):
        token_candidates["total"] = float(
            (token_candidates["input"] or 0) + (token_candidates["output"] or 0)
        )

    rows: list[dict[str, Any]] = []
    for token_type, value in token_candidates.items():
        if value is None:
            continue

        metric_attributes = {
            "trace_id": trace_id,
            "span_id": _to_str(span.get("span_id")),
            "gen_ai.token.type": token_type,
        }
        if model_name is not None:
            metric_attributes["gen_ai.response.model"] = model_name

        rows.append(
            {
                "name": "gen_ai.client.token.usage",
                "description": "Number of tokens used in a GenAI request",
                "unit": "token",
                "metric_type": "sum",
                "gauge": None,
                "sum": {
                    "start_time_unix_nano": start_ns,
                    "time_unix_nano": end_ns,
                    "value": float(value),
                    "exemplars": [],
                    "attributes": metric_attributes,
                    "flags": 0,
                    "aggregation_temporality": "AGGREGATION_TEMPORALITY_CUMULATIVE",
                    "is_monotonic": True,
                },
                "histogram": None,
                "exponential_histogram": None,
                "summary": None,
                "metadata": {},
                "resource": resource,
                "resource_schema_url": "",
                "instrumentation_scope": scope,
                "metric_schema_url": "",
            }
        )

    return rows


def _base_log_row(
    *,
    event_name: str,
    trace_id: str,
    span_id: str,
    time_unix_nano: int,
    body: str,
    attributes: dict[str, str],
    resource: dict[str, Any],
    scope: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event_name": event_name,
        "trace_id": trace_id,
        "span_id": span_id,
        "time_unix_nano": time_unix_nano,
        "observed_time_unix_nano": time_unix_nano,
        "severity_number": None,
        "severity_text": None,
        "body": body,
        "attributes": attributes,
        "dropped_attributes_count": 0,
        "flags": 0,
        "resource": resource,
        "resource_schema_url": "",
        "instrumentation_scope": scope,
        "log_schema_url": "",
    }


def _normalize_status_struct(value: Any) -> dict[str, str]:
    if isinstance(value, Mapping):
        message = _to_str(value.get("message") or value.get("description"))
        code = _to_str(value.get("code") or value.get("status_code") or value.get("statusCode")).upper()
    else:
        message = ""
        code = _to_str(value).upper()

    if "ERROR" in code:
        return {"message": message, "code": "STATUS_CODE_ERROR"}
    if "OK" in code:
        return {"message": message, "code": "STATUS_CODE_OK"}
    return {"message": message, "code": "STATUS_CODE_UNSET"}


def _derive_span_kind(span: dict[str, Any]) -> str:
    attrs = span.get("attributes") if isinstance(span.get("attributes"), dict) else {}
    span_type = _to_str(attrs.get("mlflow.spanType")).upper() if isinstance(attrs, dict) else ""
    if span_type == "CHAIN":
        return "SPAN_KIND_SERVER"
    if span_type in {"AGENT", "TOOL", "CHAT_MODEL", "LLM", "RETRIEVER"}:
        return "SPAN_KIND_INTERNAL"
    return "SPAN_KIND_INTERNAL"


def _normalize_events(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    events: list[dict[str, Any]] = []
    for event in value:
        if not isinstance(event, dict):
            continue
        events.append(
            {
                "time_unix_nano": _to_int(event.get("timestamp_ns")) or 0,
                "name": _to_str(event.get("name")),
                "attributes": _stringify_map(event.get("attributes")),
                "dropped_attributes_count": 0,
            }
        )
    return events


def _default_resource(trace: dict[str, Any]) -> dict[str, Any]:
    experiment_id = _to_optional_str(trace.get("experiment_id"))
    attributes = {"service.name": "mlflow-trace-extractor"}
    if experiment_id is not None:
        attributes["mlflow.experiment_id"] = experiment_id
    return {"attributes": attributes, "dropped_attributes_count": 0}


def _default_scope(trace: dict[str, Any]) -> dict[str, Any]:
    version = _to_optional_str(_nested_value(trace.get("metadata"), "format_version"))
    attributes: dict[str, str] = {}
    if version is not None:
        attributes["extractor.format_version"] = version
    return {
        "name": "mlflow",
        "version": "0.1.0",
        "attributes": attributes,
        "dropped_attributes_count": 0,
    }


def _trace_start_ns(trace: dict[str, Any]) -> int:
    spans = trace.get("spans_flat") if isinstance(trace.get("spans_flat"), list) else []
    start_candidates = [
        _to_int(span.get("start_time_ns")) for span in spans if isinstance(span, dict)
    ]
    non_null = [value for value in start_candidates if value is not None]
    if non_null:
        return int(min(non_null))

    timestamp_ms = _to_int(trace.get("timestamp_ms"))
    if timestamp_ms is not None:
        return int(timestamp_ms * 1_000_000)
    return int(datetime.now(UTC).timestamp() * 1_000_000_000)


def _hydrate_trace(client: Any, trace: Any) -> Any:
    if client is None:
        return trace

    info = getattr(trace, "info", None)
    if info is None:
        return trace

    trace_id = _to_optional_str(getattr(info, "trace_id", None))
    request_id = _to_optional_str(getattr(info, "request_id", None))
    lookup_id = trace_id or request_id
    if lookup_id is None:
        return trace

    try:
        hydrated = client.get_trace(lookup_id)
    except Exception:
        return trace

    hydrated_info = getattr(hydrated, "info", None)
    if hydrated_info is None:
        return trace

    hydrated_trace_id = _to_optional_str(getattr(hydrated_info, "trace_id", None))
    hydrated_request_id = _to_optional_str(getattr(hydrated_info, "request_id", None))
    if lookup_id in {candidate for candidate in [hydrated_trace_id, hydrated_request_id] if candidate}:
        return hydrated
    return trace


def _serialize_mlflow_trace(trace: Any, *, tracking_uri: str) -> dict[str, Any]:
    info = getattr(trace, "info", None)
    data = getattr(trace, "data", None)
    spans = getattr(data, "spans", None) if data is not None else None
    span_list = spans if isinstance(spans, list) else list(spans or [])

    spans_flat = [_strip_children(_serialize_mlflow_span(span)) for span in span_list]
    tree_spans = [copy.deepcopy(span) for span in spans_flat]
    root_span = _build_span_tree(tree_spans)

    trace_metadata = _to_string_map(getattr(info, "trace_metadata", None))
    trace_tags = _to_string_map(getattr(info, "tags", None))
    client_request_id = _to_optional_str(getattr(info, "client_request_id", None))
    assessments = _serialize_assessments(getattr(info, "assessments", None))
    reachable_count = _count_tree_nodes(root_span)
    span_count = len(spans_flat)

    return {
        "trace_id": _extract_trace_id(info),
        "experiment_id": _to_str(getattr(info, "experiment_id", "")),
        "timestamp_ms": _to_int(getattr(info, "timestamp_ms", 0)) or 0,
        "status": _to_str(getattr(info, "status", "UNSET")),
        "root_span": root_span,
        "spans_flat": spans_flat,
        "trace_metadata": trace_metadata,
        "trace_tags": trace_tags,
        "client_request_id": client_request_id,
        "assessments": assessments,
        "metadata": {
            "format_version": 2,
            "extraction_timestamp": datetime.now(UTC).isoformat(),
            "source_uri": tracking_uri,
            "span_count": span_count,
            "root_tree_reachable_count": reachable_count,
            "disconnected_span_count": max(0, span_count - reachable_count),
        },
    }


def _extract_trace_id(info: Any) -> str:
    if info is None:
        return ""

    request_id = getattr(info, "request_id", None)
    if request_id:
        return str(request_id)
    return _to_str(getattr(info, "trace_id", ""))


def _serialize_mlflow_span(span: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if hasattr(span, "attributes") and span.attributes:
        attributes = {
            str(key): _to_json_compatible(value)
            for key, value in span.attributes.items()
        }

    events: list[dict[str, Any]] = []
    if hasattr(span, "events") and span.events:
        for event in span.events:
            event_dict: dict[str, Any] = {
                "name": _to_str(getattr(event, "name", "")),
                "timestamp_ns": _to_int(getattr(event, "timestamp", None)),
            }
            if hasattr(event, "attributes") and event.attributes:
                event_dict["attributes"] = {
                    str(key): _to_json_compatible(value)
                    for key, value in event.attributes.items()
                }
            else:
                event_dict["attributes"] = {}
            events.append(event_dict)

    return {
        "span_id": _to_str(getattr(span, "span_id", "")),
        "parent_span_id": _to_optional_str(
            getattr(span, "parent_id", None) or getattr(span, "parent_span_id", None)
        ),
        "name": _to_str(getattr(span, "name", "")),
        "start_time_ns": _to_int(getattr(span, "start_time_ns", None)) or 0,
        "end_time_ns": _to_int(getattr(span, "end_time_ns", None)),
        "status": _to_str(getattr(span, "status", "UNSET")),
        "attributes": attributes,
        "events": events,
        "children": [],
    }


def _to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_dictionary"):
        try:
            return _to_json_compatible(value.to_dictionary())
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _to_json_compatible(value.to_dict())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_compatible(item) for item in value]
    if hasattr(value, "__dict__"):
        public_fields = {key: item for key, item in vars(value).items() if not key.startswith("_")}
        if public_fields:
            return {str(key): _to_json_compatible(item) for key, item in public_fields.items()}
    if hasattr(value, "value") and isinstance(value.value, (str, int, float, bool)):
        return value.value
    return str(value)


def _to_string_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def _serialize_assessments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    rows: list[dict[str, Any]] = []
    for item in value:
        serialized = _to_json_compatible(item)
        if isinstance(serialized, dict):
            rows.append(serialized)

    rows.sort(
        key=lambda item: (
            str(item.get("assessment_id", "")),
            str(item.get("name", "")),
            str(item.get("create_time_ms", "")),
            str(item.get("last_update_time_ms", "")),
        )
    )
    return rows


def _strip_children(span: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in span.items() if key != "children"}


def _build_span_tree(spans: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not spans:
        return None

    by_id = {span["span_id"]: span for span in spans if span.get("span_id")}
    roots: list[dict[str, Any]] = []
    for span in spans:
        parent_id = _to_optional_str(span.get("parent_span_id"))
        if parent_id and parent_id in by_id and parent_id != span.get("span_id"):
            by_id[parent_id].setdefault("children", []).append(span)
        else:
            roots.append(span)

    if not roots:
        return spans[0]

    roots.sort(key=lambda span: (span.get("start_time_ns") or 0, span.get("span_id") or ""))
    return roots[0]


def _count_tree_nodes(span: dict[str, Any] | None) -> int:
    if not span:
        return 0
    children = span.get("children") if isinstance(span, dict) else None
    if not isinstance(children, list):
        return 1
    return 1 + sum(_count_tree_nodes(child) for child in children if isinstance(child, dict))


def _trace_to_dict(trace: Any) -> dict[str, Any]:
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
    if value is None or isinstance(value, (str, bool, int)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "to_dictionary"):
        return _json_safe(value.to_dictionary())

    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())

    return str(value)


def _json_dumps_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


def _to_parent_span_id(value: Any) -> str:
    parent = _to_optional_str(value)
    return "" if parent is None else parent


def _nested_value(value: Any, dotted_path: str) -> Any:
    if value is None:
        return None

    current = value
    if isinstance(current, str):
        stripped = current.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                current = json.loads(current)
            except Exception:
                return None

    for part in dotted_path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _coerce_token(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _stringify_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}

    normalized: dict[str, str] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, (dict, list)):
            normalized[str(key)] = json.dumps(item, sort_keys=True, default=str)
        else:
            normalized[str(key)] = str(item)
    return normalized


def _to_optional_bool_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"

    text = str(value).strip().lower()
    if text in {"true", "false"}:
        return text
    return None


def _to_optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _to_str(value: Any) -> str:
    return "" if value is None else str(value)


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None