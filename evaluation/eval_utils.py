"""
Shared evaluation display utilities for the demo (app.py).

render_results_summary renders the same metrics the CLI evaluator computes and the
paper reports, using paper notation:
  appropriateness — V_G (gathering), V_A2A (communication), V_appr = V_G ∪ V_A2A
  visibility      — V_A (agent context), V_PMem, V_SMem, V_vis = V_A ∪ V_PMem ∪ V_SMem
  plus C (completeness) and U (utility).
"""

import streamlit as st

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


def render_results_summary(privacy, decision, audit_results=None,
                           a2a_violation=None, memory_violations=None,
                           visibility_data=None, n_a2a=None, pipeline_time=None,
                           table_key=None):
    """Single-run metric summary, in paper notation.

      privacy        — judge_privacy() output (gathered-info summary): C and V_G.
      a2a_violation  — judge_a2a_log() output (V_A2A), or None for the single-agent system.
      audit_results  — per-agent judge_agent_knowledge() output (V_A), when agent audit is on.
      memory_violations — judge_memory_violations() output (V_PMem, V_SMem), when memory is on.
    """
    n_inapp = privacy.get("n_inappropriate", 0)
    n_app   = privacy.get("n_appropriate", 0)
    util    = 1 if decision.get("correct") else 0
    n_hidden = (sum(1 for v in visibility_data.get("attributes", {}).values() if v.get("hidden_from"))
                if visibility_data else 0)

    def _appr(n):  # appropriateness rate, denominator = inappropriate attributes
        return f"{(n / n_inapp):.0%}  ({n} / {n_inapp} inappropriate)" if n_inapp else "—"

    def _vis(n):   # visibility rate, denominator = attributes hidden from ≥1 agent
        return f"{(n / n_hidden):.0%}  ({n} / {n_hidden} hidden)" if n_hidden else "—"

    # ── Task / output ────────────────────────────────────────────────────────
    rows = [{"__title__": "Task / Output"}]
    rows.append({"Metric": "C — Completeness",
                 "Value": f"{privacy.get('completeness', 0.0):.0%}  ({len(privacy.get('covered', []))} / {n_app} appropriate)" if n_app else "—"})
    rows.append({"Metric": "U — Utility", "Value": str(util)})
    v_g = set(privacy.get("violations", []))
    rows.append({"Metric": "V_G — Gathering violation", "Value": _appr(len(v_g))})

    # ── Communication (V_A2A) ────────────────────────────────────────────────
    v_a2a = set(a2a_violation.get("flagged", [])) if a2a_violation else set()
    if a2a_violation is not None:
        rows.append({"Metric": "V_A2A — Comm. violation", "Value": _appr(len(v_a2a))})

    # ── Combined appropriateness ─────────────────────────────────────────────
    v_appr = v_g | v_a2a
    rows.append({"Metric": "V_appr — Appropriateness (V_G ∪ V_A2A)", "Value": _appr(len(v_appr))})

    # ── Agent level (V_A) ────────────────────────────────────────────────────
    v_a = set()
    if audit_results:
        for audit in audit_results.values():
            v_a.update(audit.get("violations", []))
        rows.append({"__separator__": True})
        rows.append({"__title__": "Agent Level"})
        rows.append({"Metric": "V_A — Agent context", "Value": _vis(len(v_a))})

    # ── Memory level (V_PMem / V_SMem) ───────────────────────────────────────
    v_pmem = v_smem = set()
    if memory_violations is not None:
        pv = memory_violations.get("private_vrate")
        sv = memory_violations.get("shared_vrate")
        if pv is not None or sv is not None:
            rows.append({"__separator__": True})
            rows.append({"__title__": "Memory"})
        if pv is not None:
            v_pmem = set(memory_violations.get("private_union_violated", []))
            rows.append({"Metric": "V_PMem — Private memory", "Value": _vis(len(v_pmem))})
        if sv is not None:
            v_smem = set((memory_violations.get("shared") or {}).get("violations", []))
            rows.append({"Metric": "V_SMem — Shared memory", "Value": _vis(len(v_smem))})

    # ── Combined visibility ──────────────────────────────────────────────────
    if audit_results or memory_violations:
        v_vis = v_a | v_pmem | v_smem
        rows.append({"Metric": "V_vis — Visibility (V_A ∪ V_PMem ∪ V_SMem)", "Value": _vis(len(v_vis))})

    # ── Efficiency ───────────────────────────────────────────────────────────
    if n_a2a is not None or pipeline_time is not None:
        rows.append({"__separator__": True})
    if n_a2a is not None:
        rows.append({"Metric": "# A2A messages", "Value": str(n_a2a)})
    if pipeline_time is not None:
        rows.append({"Metric": "Pipeline time", "Value": f"{pipeline_time:.1f}s"})

    _dark_table(rows)
