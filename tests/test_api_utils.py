from hg_ds_evals.evals.api_utils import (
    compute_tool_parameter_equivalence,
    find_expected_default_parameters,
)


def _single_actual_params(result):
    return result["actual_relaxed"][0]["arguments"]


def _single_expected_params(result):
    return result["expected_relaxed"][0]["parameters"]


def test_products_filter_mode_omitted_without_account_ids_defaults_to_all():
    result = compute_tool_parameter_equivalence(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {"products_filter_mode": "PFM_SETTINGS"},
            }
        ],
        [{"tool": "analyze_transactions", "arguments": {}}],
    )

    assert _single_actual_params(result)["products_filter_mode"] == "ALL"


def test_products_filter_mode_omitted_with_account_ids_defaults_to_product_selection():
    result = compute_tool_parameter_equivalence(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {"products_filter_mode": "PRODUCT_SELECTION"},
            }
        ],
        [
            {
                "tool": "analyze_transactions",
                "arguments": {"account_ids": ["account-1-card-1"]},
            }
        ],
    )

    assert _single_actual_params(result)["products_filter_mode"] == "PRODUCT_SELECTION"


def test_actual_runtime_defaults_are_materialized_only_when_expected_asserts_them():
    result = compute_tool_parameter_equivalence(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {
                    "exclude_own_transfers": True,
                    "group_by": "group_none",
                    "visualization_type": "SUMMARY",
                },
            }
        ],
        [{"tool": "analyze_transactions", "arguments": {}}],
    )

    assert _single_actual_params(result) == {
        "exclude_own_transfers": True,
        "group_by": "group_none",
        "visualization_type": "SUMMARY",
    }


def test_expected_defaults_are_not_added_when_test_case_does_not_assert_them():
    result = compute_tool_parameter_equivalence(
        [{"tool": "analyze_transactions", "parameters": {"direction": "OUTGOING"}}],
        [{"tool": "analyze_transactions", "arguments": {}}],
    )

    assert _single_expected_params(result) == {"direction": "OUTGOING"}
    assert _single_actual_params(result) == {}


def test_sort_by_is_excused_when_group_by_is_group_none():
    result = compute_tool_parameter_equivalence(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {"group_by": "group_none", "sort_by": "date_desc"},
            }
        ],
        [
            {
                "tool": "analyze_transactions",
                "arguments": {"group_by": "group_none", "sort_by": "date_asc"},
            }
        ],
    )

    assert "sort_by" not in _single_expected_params(result)
    assert "sort_by" not in _single_actual_params(result)
    assert result["expected_excused"] == [["sort_by"]]
    assert result["actual_excused"] == [["sort_by"]]


def test_sort_is_preserved_when_group_by_is_group_none():
    result = compute_tool_parameter_equivalence(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {"group_by": "group_none", "sort": "MAIN_DATE_DESC"},
            }
        ],
        [
            {
                "tool": "analyze_transactions",
                "arguments": {"group_by": "group_none", "sort": "EXECUTION_DATE_DESC"},
            }
        ],
    )

    assert _single_expected_params(result)["sort"] == "EXECUTION_DATE_DESC"
    assert _single_actual_params(result)["sort"] == "EXECUTION_DATE_DESC"


def test_main_date_sort_aliases_normalize_to_execution_date():
    result = compute_tool_parameter_equivalence(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {"sort": "MAIN_DATE_DESC"},
            }
        ],
        [
            {
                "tool": "analyze_transactions",
                "arguments": {"sort": "EXECUTION_DATE_DESC"},
            }
        ],
    )

    assert _single_expected_params(result)["sort"] == "EXECUTION_DATE_DESC"
    assert _single_actual_params(result)["sort"] == "EXECUTION_DATE_DESC"


def test_size_default_is_materialized_only_when_expected_asserts_size():
    result = compute_tool_parameter_equivalence(
        [{"tool": "analyze_transactions", "parameters": {"size": 1000}}],
        [{"tool": "analyze_transactions", "arguments": {}}],
    )

    assert _single_actual_params(result)["size"] == 1000


def test_transaction_collection_id_excuses_ignored_filter_parameters():
    result = compute_tool_parameter_equivalence(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {
                    "transaction_collection_id": "collection-1",
                    "date_from": "2026-01-01",
                    "direction": "OUTGOING",
                    "sort": "MAIN_DATE_DESC",
                    "limit": 10,
                    "size": 1000,
                    "exclude_own_transfers": True,
                    "products_filter_mode": "ALL",
                    "group_by": "group_by_month",
                    "sort_by": "date_desc",
                    "visualization_type": "DETAIL",
                },
            }
        ],
        [
            {
                "tool": "analyze_transactions",
                "arguments": {
                    "transaction_collection_id": "collection-1",
                    "date_from": "2025-12-01",
                    "direction": "INCOMING",
                    "sort": "AMOUNT_DESC",
                    "limit": 1,
                    "size": 25,
                    "exclude_own_transfers": False,
                    "products_filter_mode": "PFM_SETTINGS",
                    "group_by": "group_by_month",
                    "sort_by": "date_desc",
                    "visualization_type": "DETAIL",
                },
            }
        ],
    )

    expected_params = _single_expected_params(result)
    actual_params = _single_actual_params(result)
    assert expected_params == {
        "transaction_collection_id": "collection-1",
        "group_by": "group_by_month",
        "sort_by": "date_desc",
        "visualization_type": "DETAIL",
    }
    assert actual_params == expected_params
    assert set(result["expected_excused"][0]) == {
        "date_from",
        "direction",
        "exclude_own_transfers",
        "limit",
        "products_filter_mode",
        "size",
        "sort",
    }
    assert set(result["actual_excused"][0]) == set(result["expected_excused"][0])


def test_non_analyze_tools_pass_through_unchanged():
    expected = [{"tool": "george-gcg-product_getCards", "parameters": {}}]
    actual = [{"tool": "george-gcg-product_getCards", "arguments": {"state": "ACTIVE"}}]

    result = compute_tool_parameter_equivalence(expected, actual)

    assert result["expected_relaxed"] == expected
    assert result["actual_relaxed"] == actual
    assert result["expected_excused"] == [[]]
    assert result["actual_excused"] == [[]]


def test_expected_default_audit_finds_source_backed_defaults():
    findings = find_expected_default_parameters(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {
                    "exclude_own_transfers": True,
                    "group_by": "group_none",
                    "products_filter_mode": "ALL",
                    "size": "1000",
                    "sort_by": "date_desc",
                    "visualization_type": "SUMMARY",
                },
            }
        ]
    )

    assert [finding["parameter"] for finding in findings] == [
        "exclude_own_transfers",
        "group_by",
        "visualization_type",
        "size",
        "sort_by",
        "products_filter_mode",
    ]
    assert findings[4]["runtime_default"] == "ignored when group_by=group_none"


def test_expected_default_audit_does_not_flag_non_default_parameters():
    findings = find_expected_default_parameters(
        [
            {
                "tool": "analyze_transactions",
                "parameters": {
                    "direction": "OUTGOING",
                    "exclude_own_transfers": False,
                    "products_filter_mode": "PFM_SETTINGS",
                    "visualization_type": "DETAIL",
                },
            },
            {"tool": "george-gcg-product_getCards", "parameters": {"state": "ACTIVE"}},
        ]
    )

    assert findings == []
