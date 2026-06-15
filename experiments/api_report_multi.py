"""Generate a self-contained HTML report comparing API trace scorer
results across multiple (typically three) MLflow runs.

This is the multi-run sibling of ``api_report.py``. Where ``api_report.py``
renders one scored run (with an optional baseline overlay), this script
takes N scored enriched-traces CSVs — one per MLflow run — and lays them
side by side so you can see, per test case, how the deterministic scorers
moved across runs. The intended use is non-determinism / regression
inspection over a handful of sequential runs of the same test set.

It deliberately reuses ``api_report``'s data layer (``enrich``,
``case_payload``, ``compute_metrics``, ``scorer_summary``) and CSS design
system so the two reports stay visually consistent and the parsing
contract lives in exactly one place.

Two tabs only:
  • Summary    — headline pass rates, primary-outcome counts and
                 per-scorer counts, one column per run, plus a
                 divergence overview (how many cases differ across runs).
  • Test Cases — one card per test_case_id with a 3-row comparison
                 (Routing / Tools called / Tool parameters) across all
                 runs in a single view. Rows where the runs disagree are
                 highlighted. Expected values are intentionally omitted.

Usage:
    python api_report_multi.py \
        --input .../enriched_traces_<RUN_A>.csv \
        --input .../enriched_traces_<RUN_B>.csv \
        --input .../enriched_traces_<RUN_C>.csv \
        --output reports/api_multi_report.html
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

# Reuse the single source of truth for parsing / enrichment / metrics and
# the shared CSS. Make the import work whether the script is launched from
# experiments/ or from a notebook directory.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_report as ar  # noqa: E402


# ─── Per-run loading ──────────────────────────────────────────────────────

def _run_label(df: pd.DataFrame, path: Path) -> str:
    """Full run identifier for a CSV: the (single) source_run_id, falling
    back to the filename stem. When a CSV mixes several run ids we keep the
    first and warn — the multi-run report assumes one run per file."""
    if "source_run_id" in df.columns:
        vals = sorted({str(v) for v in df["source_run_id"].dropna().unique() if str(v).strip()})
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            print(f"[warn] {path.name} contains {len(vals)} source_run_id values; "
                  f"using the first ({vals[0]}).")
            return vals[0]
    return path.stem


def load_run(path: Path) -> dict[str, Any]:
    """Read one scored CSV and reduce it to everything the report needs:
    the run label, summary metrics, per-scorer counts, primary-outcome
    counts and a test_case_id → case-payload index."""
    df = pd.read_csv(path)
    df = ar.enrich(df)
    if "test_case_id" not in df.columns:
        raise KeyError(f"CSV missing 'test_case_id' column: {path}")
    label = _run_label(df, path)
    cases = {ar._safe_str(c["id"]): c for c in ar.case_payload(df)}
    outcome_counts = (
        df["_issue_bucket"].value_counts()
        .reindex(["pass", "agent_routing", "tool_usage", "tool_parameter"], fill_value=0)
        .to_dict()
    )
    return {
        "path": path,
        "label": label,
        "metrics": ar.compute_metrics(df),
        "summaries": {s["scorer"]: s for s in ar.scorer_summary(df)},
        "outcomes": {k: int(v) for k, v in outcome_counts.items()},
        "cases": cases,
        "n": len(df),
    }


# ─── Signatures used for divergence detection ──────────────────────────────

def _routing_sig(case: dict[str, Any] | None) -> str | None:
    if case is None:
        return None
    return ar._safe_str(case.get("actual_agent"))


def _tools_sig(case: dict[str, Any] | None) -> str | None:
    if case is None:
        return None
    return json.dumps([ar._norm_tool(t) for t in (case.get("actual_tools") or [])])


def _params_sig(case: dict[str, Any] | None) -> str | None:
    """Order-sensitive signature of the actual tool calls' parameters, so
    two runs that called the same tools with the same args collapse to the
    same string regardless of dict key order."""
    if case is None:
        return None
    norm: list[Any] = []
    for call in case.get("actual_tool_calls") or []:
        if not isinstance(call, dict):
            norm.append(["", ar._safe_str(call)])
            continue
        name = call.get("tool") or call.get("name") or ""
        params = call.get("parameters") or call.get("args") or call.get("arguments") or {}
        if isinstance(params, dict):
            items = sorted(
                (str(k), json.dumps(v, sort_keys=True, default=str))
                for k, v in params.items()
            )
        else:
            items = [["", ar._safe_str(params)]]
        norm.append([ar._norm_tool(name), items])
    return json.dumps(norm, sort_keys=True)


def _diverges(values: list[Any]) -> bool:
    """True when ≥2 runs are present and they disagree. ``None`` (case
    absent in that run) is ignored so 'present in 2 of 3 runs' still
    compares the 2 that ran."""
    present = [v for v in values if v is not None]
    return len(set(present)) > 1 and len(present) >= 2


# ─── Test-case ordering (mirrors api_report.case_payload) ───────────────────

def _tcid_key(value: Any) -> tuple[int, str]:
    s = ar._safe_str(value)
    m = re.search(r"\d+", s)
    return (int(m.group()) if m else 10**9, s)


# ─── Score → badge helpers ──────────────────────────────────────────────────

def _binary_badge(score: float | None) -> str:
    if score is None:
        return "<span class='fm-badge fm-na'>n/a</span>"
    if score == 1:
        return "<span class='fm-badge fm-pass'>pass</span>"
    return "<span class='fm-badge fm-routing'>fail</span>"


def _param_badge(bucket: str, score: float | None) -> str:
    label = {
        "pass": "pass", "partial": "partial", "fail": "fail", "na": "n/a",
    }.get(bucket, "n/a")
    cls = {
        "pass": "fm-pass", "partial": "fm-params", "fail": "fm-routing", "na": "fm-na",
    }.get(bucket, "fm-na")
    num = "" if score is None else f" {ar._fmt_num(score, 2)}"
    return f"<span class='fm-badge {cls}'>{label}{ar._h(num)}</span>"


# ─── Summary tab ────────────────────────────────────────────────────────────

def _pct_cell(rate: float | None) -> str:
    cls = ar._rate_class(rate)
    return f"<td class='num-cell {cls}'>{ar._fmt_pct(rate)}</td>"


def _runs_header(runs: list[dict[str, Any]], first: str = "Metric") -> str:
    cols = "".join(
        f"<th class='num-head' data-tip='Run {i+1} — {ar._h(r['label'])}'>"
        f"Run {i+1}</th>"
        for i, r in enumerate(runs)
    )
    return f"<thead><tr><th>{ar._h(first)}</th>{cols}</tr></thead>"


def _headline_compare(runs: list[dict[str, Any]]) -> str:
    rows = [
        ("Test cases", lambda m: f"<td class='num-cell'>{m['n_total']}</td>"),
        ("Routing pass", lambda m: _pct_cell(m["routing_pass_rate"])),
        ("Tool pass", lambda m: _pct_cell(m["usage_pass_rate"])),
        ("Params pass", lambda m: _pct_cell(m["param_pass_rate"])),
    ]
    body = []
    for label, cell in rows:
        cells = "".join(cell(r["metrics"]) for r in runs)
        body.append(f"<tr><td>{ar._h(label)}</td>{cells}</tr>")
    return (
        "<table class='tbl'>"
        + _runs_header(runs)
        + f"<tbody>{''.join(body)}</tbody></table>"
    )


def _outcomes_compare(runs: list[dict[str, Any]]) -> str:
    labels = {
        "pass": ("Pass", "fm-pass"),
        "agent_routing": ("Agent routing fail", "fm-routing"),
        "tool_usage": ("Tool usage fail", "fm-usage"),
        "tool_parameter": ("Tool parameter mismatch", "fm-params"),
    }
    body = []
    for key, (label, cls) in labels.items():
        cells = []
        for r in runs:
            count = r["outcomes"].get(key, 0)
            share = (count / r["n"]) if r["n"] else 0
            cells.append(
                f"<td class='num-cell'>{count} "
                f"<span class='muted-inline'>({ar._fmt_pct(share)})</span></td>"
            )
        body.append(
            f"<tr><td><span class='fm-badge {cls}'>{ar._h(label)}</span></td>"
            f"{''.join(cells)}</tr>"
        )
    return (
        "<table class='tbl'>"
        + _runs_header(runs, first="Primary outcome")
        + f"<tbody>{''.join(body)}</tbody></table>"
    )


def _scorer_compare(runs: list[dict[str, Any]], scorer: str) -> str:
    rows = [
        ("Scored", lambda s: str(s["scored"])),
        ("Pass", lambda s: str(s["passed"])),
        ("Partial", lambda s: ("–" if s["kind"] == "binary" else str(s["partial"]))),
        ("Fail", lambda s: str(s["failed"])),
        ("Pass rate", lambda s: ar._fmt_pct(s["pass_rate"])),
        ("Mean", lambda s: ar._fmt_num(s["mean"], 3)),
    ]
    body = []
    for label, fn in rows:
        cells = []
        for r in runs:
            s = r["summaries"].get(scorer)
            cells.append(f"<td class='num-cell'>{ar._h(fn(s)) if s else '–'}</td>")
        body.append(f"<tr><td>{ar._h(label)}</td>{''.join(cells)}</tr>")
    return (
        "<div class='card'>"
        f"<div class='card-title'><code>{ar._h(scorer)}</code> "
        f"{ar._info_icon(ar.SCORER_MODE_INFO.get(scorer, ''))}</div>"
        "<table class='tbl'>"
        + _runs_header(runs)
        + f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _divergence_overview(div: dict[str, int], n_compared: int) -> str:
    cards = [
        ("Cases compared", str(n_compared),
         "Test cases present in at least two of the supplied runs (only "
         "these can diverge)."),
        ("Routing divergent", str(div["routing"]),
         "Cases where the actual routed agent is not identical across all "
         "runs that scored the case."),
        ("Tools divergent", str(div["tools"]),
         "Cases where the actual tool-call sequence (ordered, "
         "case-insensitive) differs across runs."),
        ("Params divergent", str(div["params"]),
         "Cases where the actual tool-call parameters differ across runs."),
    ]
    out = []
    for label, value, tip in cards:
        out.append(
            "<div class='headline-card'>"
            f"<div class='hc-label'>{ar._h(label)} {ar._info_icon(tip)}</div>"
            f"<div class='hc-value'>{ar._h(value)}</div></div>"
        )
    return "".join(out)


# ─── Test Cases tab ─────────────────────────────────────────────────────────

def _tool_chips(tools: list[Any]) -> str:
    if not tools:
        return "<span class='cmp-none'>(no tools)</span>"
    return "".join(f"<code class='tool-chip'>{ar._h(t)}</code>" for t in tools)


def _calls_detail(calls: list[Any]) -> str:
    """Compact per-call parameter listing from a list of tool calls
    (works for both the expected and actual sides)."""
    if not calls:
        return "<div class='cmp-none'>(no tool calls)</div>"
    blocks = []
    for call in calls:
        if not isinstance(call, dict):
            blocks.append(f"<div class='param-line'>{ar._h(ar._safe_str(call))}</div>")
            continue
        name = call.get("tool") or call.get("name") or "(tool)"
        params = call.get("parameters") or call.get("args") or call.get("arguments") or {}
        if isinstance(params, dict) and params:
            items = "".join(
                f"<div class='param-kv'><span class='param-k'>{ar._h(k)}</span>"
                f"<span class='param-v'>{ar._h(ar._short(json.dumps(v, ensure_ascii=False, default=str), 200))}</span></div>"
                for k, v in params.items()
            )
        else:
            items = "<div class='param-kv'><span class='param-v cmp-none'>(no params)</span></div>"
        blocks.append(
            f"<div class='param-block'><code class='tool-chip'>{ar._h(name)}</code>{items}</div>"
        )
    return "".join(blocks)


# Actual (per-run) cells ────────────────────────────────────────────────────

def _routing_cell(case: dict[str, Any] | None) -> str:
    if case is None:
        return "<td class='cmp-cell'><span class='cmp-absent'>— absent —</span></td>"
    agent = case.get("actual_agent") or "(none)"
    badge = _binary_badge(case["scores"].get("agent_routing"))
    return (
        f"<td class='cmp-cell'><div class='cmp-agent'>{ar._h(agent)}</div>{badge}</td>"
    )


def _tools_cell(case: dict[str, Any] | None) -> str:
    if case is None:
        return "<td class='cmp-cell'><span class='cmp-absent'>— absent —</span></td>"
    badge = _binary_badge(case["scores"].get("tool_usage"))
    return f"<td class='cmp-cell'><div class='cmp-tools'>{_tool_chips(case.get('actual_tools') or [])}</div>{badge}</td>"


def _params_cell(case: dict[str, Any] | None) -> str:
    if case is None:
        return "<td class='cmp-cell'><span class='cmp-absent'>— absent —</span></td>"
    badge = _param_badge(case.get("param_bucket", "na"), case["scores"].get("tool_parameter"))
    detail = _calls_detail(case.get("actual_tool_calls") or [])
    return (
        "<td class='cmp-cell'>"
        f"{badge}"
        f"<details class='param-details' open><summary>parameters</summary>{detail}</details>"
        "</td>"
    )


# Expected (reference) cells — same value across runs, rendered once ──────────

def _exp_routing_cell(ref: dict[str, Any] | None) -> str:
    if ref is None:
        return "<td class='cmp-cell exp-cell'><span class='cmp-none'>—</span></td>"
    agent = ref.get("expected_agent") or "(none)"
    return f"<td class='cmp-cell exp-cell'><div class='cmp-agent'>{ar._h(agent)}</div></td>"


def _exp_tools_cell(ref: dict[str, Any] | None) -> str:
    if ref is None:
        return "<td class='cmp-cell exp-cell'><span class='cmp-none'>—</span></td>"
    return f"<td class='cmp-cell exp-cell'><div class='cmp-tools'>{_tool_chips(ref.get('expected_tools') or [])}</div></td>"


def _exp_params_cell(ref: dict[str, Any] | None) -> str:
    if ref is None:
        return "<td class='cmp-cell exp-cell'><span class='cmp-none'>—</span></td>"
    detail = _calls_detail(ref.get("expected_tool_calls") or [])
    return (
        "<td class='cmp-cell exp-cell'>"
        f"<details class='param-details' open><summary>parameters</summary>{detail}</details>"
        "</td>"
    )


def _case_card(tcid: str, runs: list[dict[str, Any]]) -> str:
    present = [r["cases"].get(tcid) for r in runs]
    # Expected is defined by the test case, so it's identical across runs;
    # take it from the first run that has the case.
    ref = next((c for c in present if c is not None), None)

    routing_div = _diverges([_routing_sig(c) for c in present])
    tools_div = _diverges([_tools_sig(c) for c in present])
    params_div = _diverges([_params_sig(c) for c in present])

    # Pick a query string from the first run that has the case.
    query = ""
    for c in present:
        if c and c.get("query"):
            query = c["query"]
            break

    div_tags = []
    if routing_div:
        div_tags.append("routing")
    if tools_div:
        div_tags.append("tools")
    if params_div:
        div_tags.append("params")
    badges = "".join(
        f"<span class='div-badge'>{ar._h(t)} differs</span>" for t in div_tags
    )
    if not div_tags:
        badges = "<span class='div-badge agree'>all runs agree</span>"

    search_blob = " ".join([
        tcid, query,
        " ".join(ar._safe_str(c.get("actual_agent")) for c in present if c),
        " ".join(t for c in present if c for t in (c.get("actual_tools") or [])),
    ]).lower()

    def row(label: str, diverged: bool, exp_cell: str, cell_fn) -> str:
        cells = exp_cell + "".join(cell_fn(c) for c in present)
        cls = "cmp-row diverge" if diverged else "cmp-row"
        flag = "<span class='row-flag'>≠</span>" if diverged else ""
        return f"<tr class='{cls}'><th>{ar._h(label)}{flag}</th>{cells}</tr>"

    header_cols = "".join(
        f"<th class='num-head' data-tip='Run {i+1} — {ar._h(r['label'])}'>Run {i+1}</th>"
        for i, r in enumerate(runs)
    )

    return (
        f"<div class='tc-card' data-divergent='{' '.join(div_tags)}' "
        f"data-search='{ar._h(search_blob)}'>"
        "<div class='tc-head'>"
        f"<code class='tc-id'>{ar._h(tcid)}</code>{badges}"
        + (f"<div class='tc-query'>{ar._h(query)}</div>" if query else "")
        + "</div>"
        "<table class='tbl cmp-tbl'>"
        f"<thead><tr><th>Field</th><th class='num-head exp-head'>Expected</th>{header_cols}</tr></thead>"
        "<tbody>"
        + row("Routing", routing_div, _exp_routing_cell(ref), _routing_cell)
        + row("Tools called", tools_div, _exp_tools_cell(ref), _tools_cell)
        + row("Tool parameters", params_div, _exp_params_cell(ref), _params_cell)
        + "</tbody></table></div>"
    )


# ─── Document assembly ──────────────────────────────────────────────────────

EXTRA_CSS = """
/* ─── Multi-run comparison additions ─── */
.muted-inline { color: #a3b5c9; font-size: 11px; }
.grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
          gap: 16px; margin-bottom: 16px; }
.fm-na { background: #edf0f4; color: #5c7999; }

.tc-card { background: #fff; border: 1px solid #e4eaf0; border-radius: 10px;
           box-shadow: 0 1px 4px rgba(10,40,92,.07); padding: 14px 16px; margin-bottom: 14px; }
.tc-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
.tc-id { font-size: 13px; font-weight: 600; }
.tc-query { flex-basis: 100%; color: #5c7999; font-size: 12px; line-height: 1.4; }
.div-badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px;
             font-weight: 600; background: #fef4e2; color: #b46504; }
.div-badge.agree { background: #dff5ea; color: #028661; }

.cmp-tbl { table-layout: fixed; }
.cmp-tbl th:first-child { width: 120px; }
.cmp-tbl td, .cmp-tbl th { vertical-align: top; }
.cmp-tbl tbody th { text-align: left; font-weight: 600; color: #0a285c; white-space: nowrap; }
.cmp-row.diverge { background: rgba(242,169,30,0.10); }
.cmp-row.diverge > th { color: #b46504; }
.row-flag { color: #b46504; margin-left: 6px; font-weight: 700; }
.cmp-cell { font-size: 12px; }
/* Expected reference column — distinct neutral tint, left rule separating
   it from the per-run actual columns. */
.cmp-tbl .exp-head { background: #eef1f6; color: #5c7999; text-align: left; }
.cmp-tbl td.exp-cell { background: #f7f9fc; border-left: 2px solid #e4eaf0;
                       border-right: 2px solid #e4eaf0; }
.cmp-row.diverge td.exp-cell { background: #f7f9fc; }
.cmp-agent { font-family: ui-monospace, Menlo, monospace; font-size: 12px;
             margin-bottom: 4px; word-break: break-word; }
.cmp-tools { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 4px; }
.tool-chip { background: #e7effd; color: #0a285c; padding: 1px 5px; border-radius: 4px;
             font-size: 11px; }
.cmp-absent { color: #c0ccd9; font-style: italic; font-size: 11px; }
.cmp-none { color: #a3b5c9; font-style: italic; font-size: 11px; }

.param-details { margin-top: 4px; }
.param-details > summary { cursor: pointer; color: #5c7999; font-size: 11px; }
.param-block { margin: 4px 0; padding: 4px 0; border-top: 1px dashed #edf0f4; }
.param-block:first-child { border-top: none; }
.param-kv { display: flex; gap: 6px; margin-top: 2px; font-size: 11px; }
.param-k { color: #5c7999; font-weight: 600; min-width: 70px; }
.param-v { color: #0a285c; word-break: break-word; font-family: ui-monospace, Menlo, monospace; }

.cmp-toolbar { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
               margin-bottom: 14px; }
.cmp-toolbar input[type=text] { flex: 1; min-width: 220px; padding: 8px 12px;
               border: 1px solid #d3dde8; border-radius: 8px; font: inherit; font-size: 13px; }
.cmp-toolbar label { font-size: 13px; color: #5c7999; display: flex; align-items: center; gap: 6px; }
.cmp-count { color: #5c7999; font-size: 12px; margin-left: auto; }
"""

JS = r"""
// ─── Tabs ────────────────────────────────────────────────────────────
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(btn.dataset.target).classList.add("active");
  });
});

// ─── Global tip-bubble (matches api_report.py) ───────────────────────
const _tipBubble = document.createElement("div");
_tipBubble.className = "tip-bubble";
document.body.appendChild(_tipBubble);
function _showTip(target) {
  const tip = target.dataset.tip;
  if (!tip) return;
  _tipBubble.textContent = tip;
  _tipBubble.classList.add("visible");
  const rect = target.getBoundingClientRect();
  _tipBubble.style.left = "0px";
  _tipBubble.style.top = "-9999px";
  const tw = _tipBubble.offsetWidth, th = _tipBubble.offsetHeight;
  let left = rect.left + rect.width / 2 - tw / 2;
  let top = rect.bottom + 8;
  const margin = 6;
  if (left < margin) left = margin;
  if (left + tw > window.innerWidth - margin) left = window.innerWidth - tw - margin;
  if (top + th > window.innerHeight - margin) top = rect.top - th - 8;
  _tipBubble.style.left = left + "px";
  _tipBubble.style.top = top + "px";
}
function _hideTip() { _tipBubble.classList.remove("visible"); }
document.addEventListener("mouseover", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) _showTip(t);
});
document.addEventListener("mouseout", e => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (t) _hideTip();
});
window.addEventListener("scroll", _hideTip, true);

// ─── Test-case filtering ─────────────────────────────────────────────
const cards = Array.from(document.querySelectorAll(".tc-card"));
const searchEl = document.getElementById("cmp-search");
const divOnlyEl = document.getElementById("cmp-diverge-only");
const countEl = document.getElementById("cmp-count");

function applyFilter() {
  const q = (searchEl.value || "").trim().toLowerCase();
  const divOnly = divOnlyEl.checked;
  let shown = 0;
  cards.forEach(card => {
    const matchesSearch = !q || card.dataset.search.includes(q);
    const isDiv = (card.dataset.divergent || "").length > 0;
    const visible = matchesSearch && (!divOnly || isDiv);
    card.style.display = visible ? "" : "none";
    if (visible) shown++;
  });
  countEl.textContent = shown + " / " + cards.length + " cases";
}
searchEl.addEventListener("input", applyFilter);
divOnlyEl.addEventListener("change", applyFilter);
applyFilter();
"""


def render_html(runs: list[dict[str, Any]], *, output_path: Path) -> str:
    # Divergence overview across the union of cases.
    all_ids = sorted(
        {tcid for r in runs for tcid in r["cases"]},
        key=_tcid_key,
    )
    div = {"routing": 0, "tools": 0, "params": 0}
    n_compared = 0
    for tcid in all_ids:
        present = [r["cases"].get(tcid) for r in runs]
        if len([c for c in present if c is not None]) >= 2:
            n_compared += 1
        if _diverges([_routing_sig(c) for c in present]):
            div["routing"] += 1
        if _diverges([_tools_sig(c) for c in present]):
            div["tools"] += 1
        if _diverges([_params_sig(c) for c in present]):
            div["params"] += 1

    case_cards = "".join(_case_card(tcid, runs) for tcid in all_ids)
    scorer_cards = "".join(_scorer_compare(runs, s) for s in ar.SCORERS)

    css_all = ar.CSS + EXTRA_CSS

    run_lines = "".join(
        f"<span>Run {i+1}: <code>{ar._h(r['label'])}</code> "
        f"<span class='header-aux'>({r['n']} cases)</span></span>"
        for i, r in enumerate(runs)
    )

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>API Evaluation — Multi-Run Comparison</title>
<style>{css_all}</style>
</head>
<body>

<div class="sticky-top">
<header class="report-header">
  <div class="header-inner">
    <h1>API Multi-Run Comparison</h1>
    <div class="header-meta header-meta-grid">
      <div class="header-col">{run_lines}</div>
      <div class="header-footer">
        <span>Runs compared: <strong>{len(runs)}</strong> ·
          Cases (union): <strong>{len(all_ids)}</strong></span>
      </div>
    </div>
  </div>
</header>

<nav class="tab-nav">
  <div class="tab-nav-inner">
    <button class="tab-btn tab-home active" data-target="tab-summary">Summary</button>
    <button class="tab-btn" data-target="tab-cases">Test Cases</button>
  </div>
</nav>
</div>

<main class="content">

  <div id="tab-summary" class="tab-panel active">
    <div class="headline-row">
      {_divergence_overview(div, n_compared)}
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-title">Headline Metrics {ar._info_icon(
            "Pass rates per run. Routing/Tool are binary pass rates; Params is "
            "the share of scored cases with tool_parameter_score == 1. Cell tint: "
            "green ≥ 80%, amber ≥ 50%, red < 50%.")}</div>
        {_headline_compare(runs)}
      </div>
      <div class="card">
        <div class="card-title">Primary Outcomes {ar._info_icon(
            "Per run, priority-assigned outcome per case: routing failure → tool "
            "usage failure → tool-parameter mismatch → pass. Counts with share of "
            "that run's cases.")}</div>
        {_outcomes_compare(runs)}
      </div>
    </div>

    <div class="card-title" style="margin:4px 0 8px">Per-Scorer Comparison</div>
    <div class="grid-3">
      {scorer_cards}
    </div>
  </div>

  <div id="tab-cases" class="tab-panel">
    <div class="cmp-toolbar">
      <input id="cmp-search" type="text" placeholder="Search id, query, agent, tool…">
      <label><input id="cmp-diverge-only" type="checkbox"> Only divergent cases</label>
      <span id="cmp-count" class="cmp-count"></span>
    </div>
    {case_cards}
  </div>

</main>

<script>{JS}</script>
</body>
</html>
"""


# ─── CLI ────────────────────────────────────────────────────────────────────

def default_output_path() -> Path:
    return Path(__file__).resolve().parent / "reports" / "api_multi_report.html"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Scored enriched-traces CSV for one run. Repeat "
                             "--input once per run (typically three times). "
                             "Run order follows the order of these flags.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output HTML report path "
                             "(default: reports/api_multi_report.html).")
    args = parser.parse_args()

    if len(args.input) < 2:
        parser.error("provide at least two --input CSVs to compare.")

    runs: list[dict[str, Any]] = []
    for raw in args.input:
        path = raw.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input CSV does not exist: {path}")
        run = load_run(path)
        print(f"[run {len(runs)+1}] {run['label']} — {run['n']} cases ({path.name})")
        runs.append(run)

    output_path = (args.output.expanduser().resolve() if args.output
                   else default_output_path())

    html_text = render_html(runs, output_path=output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
