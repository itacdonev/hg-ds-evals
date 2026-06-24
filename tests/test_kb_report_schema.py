from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import kb_report  # noqa: E402


def test_normalize_checkpoint_schema_maps_old_scorer_columns_to_new_names() -> None:
    df = pd.DataFrame(
        [
            {
                "case_scope": "kb",
                "query_clarity_score": "2",
                "selection_semantic_relevance_score": "1",
                "selection_semantic_relevance_reasoning": "Some useful ENUMs.",
                "selected_context_sufficiency_score": "1",
                "optimal_retrieved_context_adequacy_score": "0",
                "answer_expected_alignment_score": "2",
                "answer_groundedness_score": "2",
                "language_compliance_score": "2",
                "extra_or_distracting_enums": '["DISTRACTING@ENUM"]',
                "missing_facts": '["missing fee limit"]',
                "hallucinated_claims": '["unsupported claim"]',
                "retrieved_pool_inadequacy_description": (
                    "The post-prune pool lacks the required fee limit."
                ),
            }
        ]
    )

    out = kb_report.normalize_checkpoint_schema(df)

    assert out.loc[0, "kb_selection_semantic_relevance_score"] == "1"
    assert out.loc[0, "kb_selection_semantic_relevance_reasoning"] == "Some useful ENUMs."
    assert out.loc[0, "kb_selected_context_sufficiency_score"] == "1"
    assert out.loc[0, "kb_retrieval_pool_adequacy_score"] == "0"
    assert out.loc[0, "kb_answer_groundedness_score"] == "2"
    assert out.loc[0, "answer_language_compliance_score"] == "2"
    assert out.loc[0, "non_useful_reranked_enums"] == '["DISTRACTING@ENUM"]'
    assert out.loc[0, "missing_facts_in_agent_response"] == '["missing fee limit"]'
    assert out.loc[0, "hallucinated_claims_in_agent_response"] == '["unsupported claim"]'
    assert out.loc[0, "post_prune_candidates_context_inadequacy_description"] == (
        "The post-prune pool lacks the required fee limit."
    )
    assert bool(out.loc[0, "retrieved_pool_inadequacy_identified"]) is True


def test_normalize_checkpoint_schema_preserves_existing_new_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "answer_language_compliance_score": "1",
                "language_compliance_score": "2",
            }
        ]
    )

    out = kb_report.normalize_checkpoint_schema(df)

    assert out.loc[0, "answer_language_compliance_score"] == "1"


def test_normalize_checkpoint_schema_fills_blank_new_columns_from_old_aliases() -> None:
    df = pd.DataFrame(
        [
            {
                "answer_language_compliance_score": "",
                "language_compliance_score": "2",
            }
        ]
    )

    out = kb_report.normalize_checkpoint_schema(df)

    assert out.loc[0, "answer_language_compliance_score"] == "2"


def test_dimension_metadata_aliases_fill_canonical_ids() -> None:
    out = kb_report._add_dimension_metadata_aliases(
        {
            "answer_groundedness": {
                "name": "Answer Groundedness",
                "description": "Grounded against selected KB context.",
            }
        }
    )

    assert out["kb_answer_groundedness"]["name"] == "Answer Groundedness"


def test_enrich_hides_missing_scorers_and_records_warning() -> None:
    df = pd.DataFrame(
        [
            {
                "test_case_id": "case_1",
                "query_clarity_score": "2",
                "query_clarity_reasoning": "Clear.",
            }
        ]
    )

    out = kb_report.enrich(df)

    assert list(kb_report._dimension_weights_for_df(out)) == ["query_clarity"]
    assert out.loc[0, "weighted_avg"] == 1.0
    assert any("Missing scorer columns" in w for w in out.attrs["report_schema_warnings"])


def test_enrich_flags_expected_enum_absent_from_supplied_kb_export() -> None:
    df = pd.DataFrame(
        [
            {
                "test_case_id": "case_1",
                "user_query": "Where is this feature?",
                "case_scope": "kb",
                "query_scope": "kb",
                "expected_enums": '["MISSING@ENUM"]',
                "pre_prune_enum_ids": '["OTHER@ENUM"]',
                "post_prune_enum_ids": '["OTHER@ENUM"]',
                "reranked_enum_ids": '["OTHER@ENUM"]',
            }
        ]
    )

    out = kb_report.enrich(
        df,
        kb_description_lookup={"OTHER@ENUM": {"cz": "", "en": "Other text"}},
    )
    summary = kb_report.compute_summary_metrics(out, df_all=out)

    assert out.loc[0, "failure_mode"] == "expected_enum_missing_from_kb"
    assert out.loc[0, "_expected_enums_missing_from_kb"] == [
        {"expected": "MISSING@ENUM", "raw_expected": "MISSING@ENUM"}
    ]
    assert bool(out.loc[0, "_expected_enum_missing_from_kb"]) is True
    assert summary["n_defect"] == 1
    assert summary["n_clean"] == 0


def test_enrich_treats_normalized_kb_export_match_as_naming_mismatch() -> None:
    df = pd.DataFrame(
        [
            {
                "test_case_id": "case_1",
                "user_query": "Where are regular orders?",
                "case_scope": "kb",
                "query_scope": "kb",
                "expected_enums": '["REGULAR_ORDERS"]',
                "pre_prune_enum_ids": "[]",
                "post_prune_enum_ids": "[]",
                "reranked_enum_ids": "[]",
            }
        ]
    )

    out = kb_report.enrich(
        df,
        kb_description_lookup={"REGULAR@ORDERS": {"cz": "", "en": "Regular orders"}},
    )

    assert out.loc[0, "failure_mode"] == "enum_name_mismatch"
    assert out.loc[0, "_expected_enums_missing_from_kb"] == []
    assert out.loc[0, "_enum_naming_mismatches"] == [
        {
            "expected": "REGULAR_ORDERS",
            "kb_form": "REGULAR@ORDERS",
            "raw_expected": "REGULAR_ORDERS",
        }
    ]


def test_baseline_issue_counts_use_full_issues_table_not_top_cards() -> None:
    summary = {
        "top_failures_clean": [
            {"key": "scope_misroute", "n": 9},
        ],
        "failure_modes": [
            {"key": "scope_misroute", "n": 12, "n_fail": 9},
            {"key": "pool_content_gap", "n": 5, "n_fail": 4},
            {"key": "context_use_failure", "n": 2, "n_fail": 0},
        ],
    }

    out = kb_report._baseline_issue_counts_from_summary(summary)

    assert out == {
        "scope_misroute": 9,
        "pool_content_gap": 4,
        "context_use_failure": 0,
    }


def test_top_failures_html_starts_with_total_hard_fail_card() -> None:
    html = kb_report._top_failures_html(
        [
            {
                "key": "scope_misroute",
                "label": "Wrong agent routing",
                "owner": "Agent",
                "info": "Agent answered with the wrong flow.",
                "n": 3,
                "pct": 0.75,
                "ids": ["case_1", "case_2", "case_3"],
            }
        ],
        n_fail_clean=4,
        n_clean=20,
        hard_fail_ids=["case_1", "case_2", "case_3", "case_4"],
        baseline={
            "label": "baseline.csv",
            "n_fail_clean": 6,
            "issue_counts": {"scope_misroute": 2},
        },
    )

    assert "Hard fail" in html
    assert "20.0% of all valid cases" in html
    assert html.find("Hard fail") < html.find("Wrong agent routing")
    assert "data-label='hard fail: clean cases the judge failed'" in html
    assert "title='vs baseline baseline.csv: was 6'" in html
    assert "-2</div>" in html


def test_enum_recall_target_uses_expected_enum_count() -> None:
    expected_counts = pd.Series([1, 1, 2, 3])

    single = kb_report._enum_recall_target(expected_counts, expected_counts.eq(1))
    multiple = kb_report._enum_recall_target(expected_counts, expected_counts.gt(1))
    mixed = kb_report._enum_recall_target(
        expected_counts,
        pd.Series([True, False, True, False], index=expected_counts.index),
    )

    assert single == {"target": 0.90, "basis": "single expected ENUM"}
    assert multiple == {"target": 0.70, "basis": "multiple expected ENUMs"}
    assert mixed == {"target": 0.7666666666666666, "basis": "mixed expected ENUM counts"}
    assert kb_report.RECALL_TARGET_SINGLE_EXPECTED_ENUM == 0.90
    assert kb_report.RECALL_TARGET_MULTI_EXPECTED_ENUM == 0.70
