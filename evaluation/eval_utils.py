"""
Shared evaluation display utilities for the demo (app.py).

render_results_summary shows exactly the metrics the CLI evaluator writes: it hands the
raw judge outputs to metrics.assemble_run — the same call run_evaluation.py makes — and
renders metrics.summary_rows of the result. It never computes a rate of its own, so the
demo and the CLI cannot disagree.
"""

import streamlit as st

import metrics

UI_GREEN = "#3a8c2f"
UI_TEXT  = "#e8e8e8"


def _dark_table(rows, key_col="Metric", val_col="Value"):
    """Render a list of {key_col: ..., val_col: ...} dicts as a dark-themed HTML table."""
    def _row(label, value, dim=False):
        lc = "#555" if dim else "#aaa"
        vc = "#555" if dim else UI_TEXT
        if any(w in label for w in ("Violation", "viol", "V_", "🚨")):
            vc = "#c0392b" if (value not in ("0", "—") and not value.startswith("0%")) else "#3a8c2f"
        return (
            f'<tr style="border-bottom:1px solid #161616">'
            f'<td style="color:{lc};padding:5px 20px 5px 0;white-space:nowrap;font-size:0.83rem">{label}</td>'
            f'<td style="color:{vc};padding:5px 0;font-size:0.83rem;font-family:monospace">{value}</td>'
            f'</tr>'
        )

    def _render_row(r):
        if r.get("__separator__"):
            return '<tr><td colspan="2" style="padding:0"><hr style="border:none;border-top:2px solid #2a2a2a;margin:4px 0"></td></tr>'
        if "__title__" in r:
            return (
                f'<tr><td colspan="2" style="padding:8px 0 2px 0">'
                f'<span style="color:#3a8c2f;font-size:0.7rem;letter-spacing:1.5px;'
                f'text-transform:uppercase;font-weight:600">{r["__title__"]}</span>'
                f'</td></tr>'
            )
        return _row(r[key_col], r[val_col], dim=r[key_col].startswith("  └"))

    body = "".join(_render_row(r) for r in rows)
    st.markdown(
        f'<div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:8px;'
        f'padding:0.7rem 1rem;margin-top:0.6rem;overflow-x:auto">'
        f'<table style="width:100%;border-collapse:collapse">{body}</table>'
        f'</div>',
        unsafe_allow_html=True,
    )


# Which section each metric belongs to, so the table reads like the paper's grouping.
_SECTIONS = [
    ("Task / Output", ["C", "U", "V_G", "V_A2A", "V_out", "V_appr"]),
    ("Agent Level",   ["V_A"]),
    ("Memory",        ["V_PMem", "V_SMem"]),
    ("Combined",      ["V_vis", "V_any"]),
    ("Efficiency",    ["#A2A", "Pipeline time"]),
]


def render_results_summary(privacy, decision, audit_results=None,
                           a2a_violation=None, memory_violations=None,
                           visibility_data=None, n_a2a=None, pipeline_time=None,
                           output_leak=None, appropriateness_data=None, utility_data=None,
                           table_key=None):
    """Single-run metric summary, in paper notation.

      privacy        — judge_privacy() output (gathered-info summary): C and V_G.
      output_leak    — V_out judge output on the final answer, when it was run.
      a2a_violation  — judge_a2a_log() output (V_A2A), or None for the single-agent system.
      audit_results  — per-agent judge_agent_knowledge() output (V_A), when agent audit is on.
      memory_violations — judge_memory_violations() output (V_PMem, V_SMem), when memory is on.
    """
    m = metrics.assemble_run(
        privacy=privacy, decision=decision, output_leak=output_leak,
        a2a_violation=a2a_violation, audit_results=audit_results,
        memory_violations=memory_violations,
        appropriateness_data=appropriateness_data, visibility_data=visibility_data,
        utility_data=utility_data,
    )
    # summary_rows returns (label, value); map each label back to its metric key so the
    # rows can be grouped into sections without the display knowing any metric's meaning.
    key_of = {label: key for key, label in metrics.LABELS.items()}
    produced = [(key_of.get(label, label), label, value)
                for label, value in metrics.summary_rows(m, n_a2a=n_a2a, pipeline_time=pipeline_time)]

    rows = []
    for title, keys in _SECTIONS:
        section = [(label, value) for key, label, value in produced if key in keys]
        if not section:
            continue
        if rows:
            rows.append({"__separator__": True})
        rows.append({"__title__": title})
        rows += [{"Metric": label, "Value": value} for label, value in section]
    _dark_table(rows)
