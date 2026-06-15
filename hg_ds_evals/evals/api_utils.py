"""Utilities for deterministic API evaluation scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any


ANALYZE_TRANSACTIONS_TOOL = "analyze_transactions"

_GROUP_NONE = "group_none"
_DEFAULT_GROUP_BY = _GROUP_NONE
_DEFAULT_SORT_BY = "total_sum_desc"
_DEFAULT_VISUALIZATION_TYPE = "SUMMARY"
_DEFAULT_EXCLUDE_OWN_TRANSFERS = True
_DEFAULT_SIZE = 1000
_DEFAULT_PRODUCTS_WITH_SELECTION = "PRODUCT_SELECTION"
_DEFAULT_PRODUCTS_WITHOUT_SELECTION = "ALL"

# ``size`` is documented by ``analyze_transactions`` as an upstream default,
# not as a Python signature default on the MCP wrapper.
_DEFAULTABLE_ACTUAL_KEYS = {
    "exclude_own_transfers",
    "group_by",
    "products_filter_mode",
    "size",
    "sort_by",
    "visualization_type",
}

_TRANSACTION_COLLECTION_ID_KEY = "transaction_collection_id"
_IGNORED_WITH_TRANSACTION_COLLECTION_ID = {
    "account_ids",
    "amount_from",
    "amount_to",
    "date_from",
    "date_to",
    "direction",
    "exclude_own_transfers",
    "excluded_main_category",
    "excluded_sub_category",
    "limit",
    "main_category",
    "products_filter_mode",
    "search_string",
    "size",
    "sort",
    "sub_category",
    "types",
}

_TRANSACTION_SORT_ALIASES = {
    "MAIN_DATE": "EXECUTION_DATE",
    "MAIN_DATE_ASC": "EXECUTION_DATE_ASC",
    "MAIN_DATE_DESC": "EXECUTION_DATE_DESC",
}


def compute_tool_parameter_equivalence(
    expected_tool_calls: object,
    actual_tool_calls: object,
    *,
    eval_persona: object = None,
    personas_without_product_filter: set[str] | None = None,
) -> dict[str, object]:
    """Normalize tool-call parameters by source-backed runtime semantics.

    This is intentionally conservative: expected calls are not populated with
    defaults that the test case did not assert. Actual ``analyze_transactions``
    calls are populated only for defaultable keys that appear in the relaxed
    expected calls, so omitted runtime defaults can match explicit expected
    defaults without inflating the expected-key denominator. Runtime-computed
    date defaults are not materialized.
    """
    rule_counts: Counter[str] = Counter()
    expected_calls = _as_call_list(expected_tool_calls)
    actual_calls = _as_call_list(actual_tool_calls)

    expected_relaxed: list[object] = []
    expected_excused: list[list[str]] = []
    expected_default_keys: set[str] = set()

    for call in expected_calls:
        relaxed_call, excused_keys = _relax_expected_call(call, rule_counts)
        expected_relaxed.append(relaxed_call)
        expected_excused.append(excused_keys)
        if _is_relaxation_tool(relaxed_call):
            params, _ = _params_dict(relaxed_call, side="expected")
            expected_default_keys.update(set(params) & _DEFAULTABLE_ACTUAL_KEYS)

    actual_relaxed: list[object] = []
    actual_excused: list[list[str]] = []

    for call in actual_calls:
        relaxed_call, excused_keys = _relax_actual_call(
            call,
            expected_default_keys=expected_default_keys,
            rule_counts=rule_counts,
        )
        actual_relaxed.append(relaxed_call)
        actual_excused.append(excused_keys)

    actual_relaxed = _apply_persona_product_filter_exception(
        expected_relaxed,
        actual_relaxed,
        eval_persona=eval_persona,
        personas_without_product_filter=personas_without_product_filter or set(),
        rule_counts=rule_counts,
    )

    return {
        "expected_relaxed": expected_relaxed,
        "actual_relaxed": actual_relaxed,
        "expected_excused": expected_excused,
        "actual_excused": actual_excused,
        "rule_counts": dict(rule_counts),
    }


def tool_parameter_kwargs_from_row(
    row: Mapping[str, object],
    *,
    equivalence_enabled: bool,
) -> dict[str, list[object]]:
    """Build ``ToolParameterScorer`` kwargs from a raw or equivalence row."""
    if equivalence_enabled:
        equivalence = row.get("_parameter_equivalence") or row.get("_relaxation")
        if isinstance(equivalence, Mapping):
            return {
                "expected_tool_calls": _as_call_list(equivalence.get("expected_relaxed")),
                "actual_tool_calls": _as_call_list(equivalence.get("actual_relaxed")),
            }
    return {
        "expected_tool_calls": _as_call_list(row.get("expected_tool_calls")),
        "actual_tool_calls": _as_call_list(row.get("actual_tool_calls")),
    }


def find_expected_default_parameters(expected_tool_calls: object) -> list[dict[str, object]]:
    """Find expected tool parameters that duplicate runtime defaults."""
    findings: list[dict[str, object]] = []

    for call_index, call in enumerate(_as_call_list(expected_tool_calls)):
        if not _is_relaxation_tool(call):
            continue
        assert isinstance(call, Mapping)
        params, _ = _params_dict(call, side="expected")

        for parameter, runtime_default, reason in (
            (
                "exclude_own_transfers",
                _DEFAULT_EXCLUDE_OWN_TRANSFERS,
                "matches analyze_transactions runtime default",
            ),
            ("group_by", _DEFAULT_GROUP_BY, "matches analyze_transactions runtime default"),
            (
                "visualization_type",
                _DEFAULT_VISUALIZATION_TYPE,
                "matches analyze_transactions runtime default",
            ),
            ("size", _DEFAULT_SIZE, "matches documented upstream default"),
        ):
            if parameter in params and _same_value(params[parameter], runtime_default):
                findings.append(
                    _expected_default_finding(
                        call_index,
                        parameter,
                        params[parameter],
                        runtime_default,
                        reason,
                    )
                )

        if "sort_by" in params:
            if _effective_group_by(params) == _GROUP_NONE:
                findings.append(
                    _expected_default_finding(
                        call_index,
                        "sort_by",
                        params["sort_by"],
                        "ignored when group_by=group_none",
                        "sort_by has no effect when group_by resolves to group_none",
                    )
                )
            elif _same_value(params["sort_by"], _DEFAULT_SORT_BY):
                findings.append(
                    _expected_default_finding(
                        call_index,
                        "sort_by",
                        params["sort_by"],
                        _DEFAULT_SORT_BY,
                        "matches analyze_transactions runtime default",
                    )
                )

        if "products_filter_mode" in params:
            runtime_default = _default_products_filter_mode(params)
            if _same_value(params["products_filter_mode"], runtime_default):
                selection_reason = (
                    "account_ids present"
                    if _has_account_ids(params.get("account_ids"))
                    else "account_ids absent"
                )
                findings.append(
                    _expected_default_finding(
                        call_index,
                        "products_filter_mode",
                        params["products_filter_mode"],
                        runtime_default,
                        f"matches conditional runtime default ({selection_reason})",
                    )
                )

    return findings


def _as_call_list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _is_relaxation_tool(call: object) -> bool:
    if not isinstance(call, Mapping):
        return False
    tool_name = str(call.get("tool") or call.get("name") or "")
    return tool_name.casefold() == ANALYZE_TRANSACTIONS_TOOL


def _params_dict(call: Mapping[str, object], *, side: str) -> tuple[dict[str, object], str]:
    canonical_key = "parameters" if side == "expected" else "arguments"
    raw_params = call.get(canonical_key)
    if isinstance(raw_params, dict):
        return dict(raw_params), canonical_key

    other_key = "arguments" if canonical_key == "parameters" else "parameters"
    other_params = call.get(other_key)
    if isinstance(other_params, dict):
        return dict(other_params), canonical_key

    return {}, canonical_key


def _with_params(call: Mapping[str, object], params: dict[str, object], key: str) -> dict[str, object]:
    updated = dict(call)
    updated[key] = params
    other_key = "arguments" if key == "parameters" else "parameters"
    if other_key in updated:
        updated[other_key] = dict(params)
    return updated


def _relax_expected_call(
    call: object,
    rule_counts: Counter[str],
) -> tuple[object, list[str]]:
    if not _is_relaxation_tool(call):
        return call, []
    assert isinstance(call, Mapping)

    params, write_key = _params_dict(call, side="expected")
    excused_keys: list[str] = []
    changed = False

    if _has_transaction_collection_id(params):
        dropped_keys = _drop_ignored_collection_filters(params)
        if dropped_keys:
            excused_keys.extend(dropped_keys)
            rule_counts["drop filters for transaction_collection_id (expected)"] += len(
                dropped_keys
            )
            changed = True
    else:
        changed = _normalize_transaction_sort_alias(
            params, side="expected", rule_counts=rule_counts
        )

    if _effective_group_by(params) == _GROUP_NONE and "sort_by" in params:
        params.pop("sort_by", None)
        excused_keys.append("sort_by")
        rule_counts["drop sort_by for group_none (expected)"] += 1
        changed = True

    return (_with_params(call, params, write_key) if changed else call), excused_keys


def _relax_actual_call(
    call: object,
    *,
    expected_default_keys: set[str],
    rule_counts: Counter[str],
) -> tuple[object, list[str]]:
    if not _is_relaxation_tool(call):
        return call, []
    assert isinstance(call, Mapping)

    params, write_key = _params_dict(call, side="actual")
    excused_keys: list[str] = []
    collection_id_present = _has_transaction_collection_id(params)
    changed = False

    if collection_id_present:
        dropped_keys = _drop_ignored_collection_filters(params)
        if dropped_keys:
            excused_keys.extend(dropped_keys)
            rule_counts["drop filters for transaction_collection_id (actual)"] += len(
                dropped_keys
            )
            changed = True
    else:
        changed = _normalize_transaction_sort_alias(
            params, side="actual", rule_counts=rule_counts
        )

    if "group_by" in expected_default_keys and _missing(params.get("group_by")):
        params["group_by"] = _DEFAULT_GROUP_BY
        rule_counts["default group_by=group_none (actual)"] += 1
        changed = True

    if _effective_group_by(params) == _GROUP_NONE and "sort_by" in params:
        params.pop("sort_by", None)
        excused_keys.append("sort_by")
        rule_counts["drop sort_by for group_none (actual)"] += 1
        changed = True
    elif "sort_by" in expected_default_keys and _missing(params.get("sort_by")):
        params["sort_by"] = _DEFAULT_SORT_BY
        rule_counts["default sort_by=total_sum_desc (actual)"] += 1
        changed = True

    if "visualization_type" in expected_default_keys and _missing(params.get("visualization_type")):
        params["visualization_type"] = _DEFAULT_VISUALIZATION_TYPE
        rule_counts["default visualization_type=SUMMARY (actual)"] += 1
        changed = True

    if (
        not collection_id_present
        and "exclude_own_transfers" in expected_default_keys
        and _missing(params.get("exclude_own_transfers"))
    ):
        params["exclude_own_transfers"] = _DEFAULT_EXCLUDE_OWN_TRANSFERS
        rule_counts["default exclude_own_transfers=True (actual)"] += 1
        changed = True

    if not collection_id_present and "size" in expected_default_keys and _missing(params.get("size")):
        params["size"] = _DEFAULT_SIZE
        rule_counts["default size=1000 (actual)"] += 1
        changed = True

    if (
        not collection_id_present
        and "products_filter_mode" in expected_default_keys
        and _missing(params.get("products_filter_mode"))
    ):
        params["products_filter_mode"] = _default_products_filter_mode(params)
        rule_counts[f"default products_filter_mode={params['products_filter_mode']} (actual)"] += 1
        changed = True

    return (_with_params(call, params, write_key) if changed else call), excused_keys


def _has_transaction_collection_id(params: Mapping[str, object]) -> bool:
    return _TRANSACTION_COLLECTION_ID_KEY in params and not _missing(
        params.get(_TRANSACTION_COLLECTION_ID_KEY)
    )


def _drop_ignored_collection_filters(params: dict[str, object]) -> list[str]:
    dropped_keys: list[str] = []
    for key in sorted(_IGNORED_WITH_TRANSACTION_COLLECTION_ID):
        if key in params:
            params.pop(key, None)
            dropped_keys.append(key)
    return dropped_keys


def _normalize_transaction_sort_alias(
    params: dict[str, object],
    *,
    side: str,
    rule_counts: Counter[str],
) -> bool:
    value = params.get("sort")
    if not isinstance(value, str):
        return False
    normalized = _TRANSACTION_SORT_ALIASES.get(value)
    if normalized is None:
        return False
    params["sort"] = normalized
    rule_counts[f"normalize {value}->{normalized} ({side})"] += 1
    return True


def _apply_persona_product_filter_exception(
    expected_calls: list[object],
    actual_calls: list[object],
    *,
    eval_persona: object,
    personas_without_product_filter: set[str],
    rule_counts: Counter[str],
) -> list[object]:
    """Apply an explicitly configured persona-fixture exception.

    This is not a general ``analyze_transactions`` runtime default. It is off by
    default and only fires when the caller provides persona names whose fixture
    setup makes ``ALL`` equivalent to ``PFM_SETTINGS`` for the evaluated case.
    """
    if str(eval_persona or "") not in personas_without_product_filter:
        return actual_calls

    actual_positions = [
        index for index, call in enumerate(actual_calls) if _is_relaxation_tool(call)
    ]
    if not actual_positions:
        return actual_calls

    updated_calls = list(actual_calls)
    analyze_occurrence = 0

    for expected_call in expected_calls:
        if not _is_relaxation_tool(expected_call):
            continue
        assert isinstance(expected_call, Mapping)
        expected_params, _ = _params_dict(expected_call, side="expected")
        if expected_params.get("products_filter_mode") != "PFM_SETTINGS":
            analyze_occurrence += 1
            continue
        if analyze_occurrence >= len(actual_positions):
            break

        actual_index = actual_positions[analyze_occurrence]
        actual_call = updated_calls[actual_index]
        assert isinstance(actual_call, Mapping)
        actual_params, write_key = _params_dict(actual_call, side="actual")
        if actual_params.get("products_filter_mode") == _DEFAULT_PRODUCTS_WITHOUT_SELECTION:
            actual_params["products_filter_mode"] = "PFM_SETTINGS"
            updated_calls[actual_index] = _with_params(actual_call, actual_params, write_key)
            rule_counts["persona product filter ALL->PFM_SETTINGS (actual)"] += 1
        analyze_occurrence += 1

    return updated_calls


def _effective_group_by(params: Mapping[str, object]) -> object:
    value = params.get("group_by")
    return _DEFAULT_GROUP_BY if _missing(value) else value


def _default_products_filter_mode(params: Mapping[str, object]) -> str:
    return (
        _DEFAULT_PRODUCTS_WITH_SELECTION
        if _has_account_ids(params.get("account_ids"))
        else _DEFAULT_PRODUCTS_WITHOUT_SELECTION
    )


def _has_account_ids(value: object) -> bool:
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return value not in (None, "")


def _same_value(left: object, right: object) -> bool:
    return left == right or str(left).casefold() == str(right).casefold()


def _expected_default_finding(
    call_index: int,
    parameter: str,
    expected_value: object,
    runtime_default: object,
    reason: str,
) -> dict[str, object]:
    return {
        "call_index": call_index,
        "tool": ANALYZE_TRANSACTIONS_TOOL,
        "parameter": parameter,
        "expected_value": expected_value,
        "runtime_default": runtime_default,
        "reason": reason,
        "suggested_action": "remove from expected_tool_calls unless this is intentionally non-default",
    }


def _missing(value: object) -> bool:
    return value is None
