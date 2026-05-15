# HEY GEORGE - EVALS
# Functions for parsing LLM evaluation responses.

# parsers.py
#=============================================================
import json
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, Markdown
from ds_common.config.config import HGCol as C


def _base_result(res: dict, id_columns: list, passthrough_columns: list) -> dict:
    main_results = {col: res.get(col) for col in id_columns}
    for col in passthrough_columns:
        main_results[col] = res.get(col)
    return main_results


def _error_result(
    res: dict,
    id_columns: list,
    passthrough_columns: list,
    error_message: str,
    error_type: str,
) -> dict:
    main_results = _base_result(res, id_columns, passthrough_columns)
    main_results["error"] = True
    main_results["error_type"] = error_type
    main_results["error_message"] = error_message
    main_results["raw_output_text"] = res.get("output_text")
    return main_results


def parse_single_row_response(res:dict, id_columns:list=None, passthrough_columns:list=None) -> dict:
    """
    Parse a single result row from LLM evaluation (schema-agnostic).
    
    Args:
        res: Dictionary containing response data
        id_columns: List of column names to use as identifiers (e.g., [C.SESSION_ID, C., 'event_id'])
        passthrough_columns: List of column names to carry through from input to
            checkpoint without being sent to the LLM.
    
    Returns:
        Dictionary with parsed results
    """
    # Define and extract ID columns
    if id_columns is None:
        id_columns = [C.SESSION_ID, C.FLOW_SEQUENCE, C.EVENT_ID]
    if passthrough_columns is None:
        passthrough_columns = []

    main_results = _base_result(res, id_columns, passthrough_columns)

    # Handle errors
    if res.get('error'):
        return _error_result(
            res=res,
            id_columns=id_columns,
            passthrough_columns=passthrough_columns,
            error_message=str(res.get("error_message") or res.get("error") or ""),
            error_type=str(res.get("error_type") or "EvaluationError"),
        )

    try:
        # Parse JSON response
        json_result = json.loads(res['output_text'])

        main_results["error"] = False
        main_results["error_type"] = ""
        main_results["error_message"] = ""
        main_results["raw_output_text"] = ""

        # Get all top levels except rubric_scores
        for key,value in json_result.items():
            if key == "rubric_scores":
                continue
            
            if isinstance(value, (list, dict)):
                main_results[key] = json.dumps(value)
            else:
                main_results[key] = value

        # Parse rubric scores
        rubric_scores = json_result.get('rubric_scores', {})
        
        for rubric_name in rubric_scores:
            rubric_data = rubric_scores.get(rubric_name, {})
            main_results[f'{rubric_name}_score'] = rubric_data.get('score')
            main_results[f'{rubric_name}_reasoning'] = rubric_data.get('reasoning')
        
    except json.JSONDecodeError as e:
        return _error_result(
            res=res,
            id_columns=id_columns,
            passthrough_columns=passthrough_columns,
            error_message=str(e),
            error_type=type(e).__name__,
        )
    
    return main_results


def parse_eval_responses(results: list, df: pd.DataFrame, id_columns: list = None) -> pd.DataFrame:
    """
    Parse JSON results for evaluations (schema-agnostic) and merge with original DataFrame.
    
    Args:
        results: List of response dictionaries from LLM
        df: Original pandas DataFrame with input data
        id_columns: List of column names to use as merge keys
    
    Returns:
        DataFrame with parsed evaluation results merged with original data
    """
    if id_columns is None:
        id_columns = [C.SESSION_ID, C.FLOW_SEQUENCE, C.EVENT_IDe]
    
    parsed_results = []
    for res in results:
        parsed_row = parse_single_row_response(res, id_columns)
        parsed_results.append(parsed_row)

    results_df = pd.DataFrame(parsed_results)
    
    final_df = df.merge(
        results_df,
        on=id_columns,
        how='left',
        suffixes=('', '_eval')
    )

    return final_df

def show_case_view(pdf_row: pd.Series) -> None:
    """Pretty-print a single eval row in logical sections."""

    sections = {
        "IDs": [
            f"{C.SESSION_ID}", f"{C.FLOW_SEQUENCE}", f"{C.EVENT_ID}", f"{C.EVENT_DATE}",
            f"{C.INTENT_CATEGORY}", f"{C.INTENT_TOPIC}", "domain"
        ],
        "User / Bot": [
            f"{C.MESSAGE_SENT_EN}", "contexted_query_en",
            "conversation_history_txt", "message_received_en",
            "fallback", "fallback_reasoning"
        ],
        "ENUMs": [
            f"{C.ENUM_PHASE_II}", "ENUM_idx", f"{C.ENUM_PHASE_I}"
        ],
        "Eval results": [
            'optimal_enum_selection', 'optimal_enum_positions',
            'selection_was_optimal', 'root_cause_category', 'kbx_gap_identified',
            'kbx_gap_description', 'kbx_overall_explanation',
            'expected_answer_with_optimal_selection'
        ]
    }

    for title, cols in sections.items():
        cols = [c for c in cols if c in pdf_row.index]
        if not cols:
            continue

        display(Markdown(f"### {title}"))
        data = [
            {"field": c, "value": str(pdf_row[c]) if pdf_row[c] is not None else ""}
            for c in cols
        ]
        display(pd.DataFrame(data))

    # --- Rubrics: keep DataFrame order ---
    rubrics = {}
    for col in pdf_row.index:
        if col.endswith("_score"):
            name = col[:-6]  # remove "_score"
            rubrics.setdefault(name, {"rubric": name})
            rubrics[name]["score"] = pdf_row[col]
        elif col.endswith("_reasoning"):
            name = col[:-10]  # remove "_reasoning"
            rubrics.setdefault(name, {"rubric": name})
            rubrics[name]["reasoning"] = pdf_row[col]

    # preserve first-seen order
    rubric_rows = list(rubrics.values())

    if rubric_rows:
        display(Markdown("### Rubrics"))
        display(pd.DataFrame(rubric_rows))


def build_rubric_viewer(df: pd.DataFrame, index_col: str = C.SESSION_ID) -> None:
    """Interactive viewer for a single eval row with rubric details.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation dataframe (e.g., tmp_results.toPandas()).
    index_col : str, optional
        Column to use for selecting rows (default is "session_id").
    """
    if df.empty:
        display(Markdown("**No rows to display.**"))
        return
    
    if index_col not in df.columns:
        raise ValueError(f"Column '{index_col}' not found in DataFrame.")
    
    # Build dropdown options in DataFrame order
    unique_keys = df[index_col].astype(str).tolist()
    options = [(f"{index_col}={k}", k) for k in unique_keys]
    
    dropdown = widgets.Dropdown(
        options=options,
        description="Session:",
        layout=widgets.Layout(width="60%")
    )
    
    output = widgets.Output()
    
    def render_row(key: str):
        output.clear_output()
        with output:
            # more robust match: cast both sides to string
            mask = df[index_col].astype(str) == str(key)
            row = df[mask]
            if row.empty:
                display(Markdown(f"No row found for {index_col} = `{key}`"))
                return
            show_case_view(row.iloc[0])
    
    def on_change(change):
        if change["name"] == "value" and change["new"] is not None:
            render_row(change["new"])
    
    dropdown.observe(on_change, names="value")
    display(widgets.VBox([dropdown, output]))
    
    # initial render
    render_row(unique_keys[0])
