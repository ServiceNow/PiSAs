"""
PiSAs Demo — Animated pipeline visualization.
Run with: streamlit run app.py   (or: -- -d <path/to/scenario_folder>)
"""

import argparse
import json
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

# ── Paths ─────────────────────────────────────────────────────────────────────

_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("-d", "--data", default=None,
                     help="Path to a scenario folder (containing scenario.json, utility.json, "
                          "visibility.json, appropriateness.json), or a folder name under data/.")
_cli_args, _ = _parser.parse_known_args()
_CLI_SCENARIO = _cli_args.data  # None if not passed

def _available_scenarios() -> list[str]:
    return sorted(
        d for d in os.listdir(_DATA_ROOT)
        if os.path.isdir(os.path.join(_DATA_ROOT, d))
    )

def _scenario_paths(scenario_id: str) -> dict:
    # Accept either a direct filesystem path to a scenario folder, or a name under data/.
    base = scenario_id if os.path.isdir(scenario_id) else os.path.join(_DATA_ROOT, scenario_id)
    return {
        "scenario":        os.path.join(base, "scenario.json"),
        "utility":         os.path.join(base, "utility.json"),
        "visibility":      os.path.join(base, "visibility.json"),
        "appropriateness": os.path.join(base, "appropriateness.json"),
    }


def build_cast_context(scenario: dict) -> str:
    """Build a name→role mapping string from scenario cast for use in judge prompts."""
    lines = []
    for member in scenario.get("cast", {}).values():
        name = member.get("name", "")
        role = member.get("role", "").replace("_", " ")
        team = member.get("team", "")
        entry = f"- {name}: {role}"
        if team:
            entry += f" ({team} team)"
        lines.append(entry)
    return "\n".join(lines)


MODELS = [
    "local/gpt-oss-120b",
    "anthropic/claude-opus-4-7",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
    "google/gemini-2.5-pro",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "openai/gpt-4o-mini",
    "qwen/qwen3.6-27b",
    "deepseek/deepseek-chat",
]


UI_GREEN = "#62D84E"   # accent green
UI_BLACK = "#0a0a0a"
UI_TEXT  = "#e8e8e8"   # white/light foreground
UI_DIM   = "#666666"   # neutral dim for secondary text

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="PiSAs Demo", page_icon="🔐", layout="wide")

st.markdown("""
<style>
    /* Black background */
    .stApp                       { background-color: #000000 !important; }
    [data-testid="stHeader"]     { background-color: #000000 !important; }
    /* Pointer cursor on all dropdowns, toggles, sliders, buttons */
    div[data-baseweb="select"] * { cursor: pointer !important; }
    div[data-baseweb="toggle"] * { cursor: pointer !important; }
    input[type="range"]          { cursor: pointer !important; }
    button                       { cursor: pointer !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        f'<div style="padding:0.6rem 0 1rem 0">'
        f'<span style="font-size:1.3rem;font-weight:700;color:{UI_GREEN};letter-spacing:1px">⬡ PiSAs</span><br>'
        f'<span style="font-size:0.72rem;color:#666;letter-spacing:2px;text-transform:uppercase">Multi-Agent Privacy Eval</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">🔑 OpenRouter API Key</p>', unsafe_allow_html=True)
    _env_key = os.getenv("OPENROUTER_API_KEY", "")
    api_key = st.text_input(
        "OpenRouter API Key",
        value=_env_key,
        type="password",
        placeholder="sk-or-...",
        label_visibility="collapsed",
    ) or _env_key

    st.markdown("<hr style='border-color:#222;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">📂 Scenario</p>', unsafe_allow_html=True)
    if _CLI_SCENARIO:
        selected_scenario = _CLI_SCENARIO
        st.markdown(f'<span style="font-family:monospace;font-size:0.8rem;color:{UI_GREEN}">{selected_scenario}</span>', unsafe_allow_html=True)
    else:
        _scenarios = _available_scenarios()
        if not _scenarios:
            st.error(f"No scenarios found in {_DATA_ROOT}")
        selected_scenario = st.selectbox(
            "Scenario",
            options=_scenarios,
            index=None,
            placeholder="Select a scenario…",
            label_visibility="collapsed",
        )
    _paths = _scenario_paths(selected_scenario) if selected_scenario else None

    st.markdown("<hr style='border-color:#222;margin:0.8rem 0'>", unsafe_allow_html=True)
    st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">⚙ System</p>', unsafe_allow_html=True)
    system_design = st.selectbox(
        "Architecture",
        options=["Single", "Decentralized", "Centralized"],
        index=2,
        label_visibility="collapsed",
    )
    use_team_agent        = system_design == "Single"
    use_token_passing     = system_design == "Decentralized"
    use_truly_centralized = system_design == "Centralized"

    agent_llm = st.selectbox("Agent LLM", MODELS,
                             index=MODELS.index("google/gemini-2.5-flash"),
                             label_visibility="collapsed")
    team_llm      = agent_llm if use_team_agent   else MODELS[0]
    siloed_llm    = agent_llm if (use_token_passing or use_truly_centralized) else MODELS[0]

    use_private_memory   = False
    use_shared_memory    = False
    use_memory_cleanup   = False
    shared_memory_writer_flag = "all"
    if not use_team_agent:
        st.markdown("<hr style='border-color:#222;margin:0.8rem 0'>", unsafe_allow_html=True)
        st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">🧠 Memory</p>', unsafe_allow_html=True)
        use_private_memory = st.toggle("Private memory", value=False)
        use_shared_memory  = st.toggle("Shared memory",  value=False)
        if use_token_passing:
            shared_memory_writer_flag = "all"
        else:
            # TC-only radio. "Coordinator only" is the paper's centralized restricted-write
            # variant: only the coordinator writes to shared memory (user agents cannot),
            # mapped to the internal "executor" flag value.
            shared_memory_writer = st.radio(
                "Shared memory write access",
                options=["All team", "Coordinator only"],
                index=0,
                horizontal=True,
                disabled=not use_shared_memory,
                label_visibility="collapsed",
            )
            shared_memory_writer_flag = "all" if shared_memory_writer == "All team" else "executor"
        use_memory_cleanup = st.toggle("Memory cleanup", value=False, disabled=not use_shared_memory)

    st.markdown("<hr style='border-color:#222;margin:0.8rem 0'>", unsafe_allow_html=True)

    st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">⚖ Judge</p>', unsafe_allow_html=True)
    judge_llm = st.selectbox("Judge LLM", MODELS,
                             index=MODELS.index("google/gemini-2.5-flash"),
                             label_visibility="collapsed")
    st.caption("Defaults are cheap-but-capable. The paper uses gemini-2.5-pro as judge "
               "(haiku-4-5 / gpt-4o-mini / gemini-2.5-flash verifiers).")
    lenient_context_matching  = st.toggle("Lenient context matching",    value=True)
    run_agent_level_audit     = st.toggle("Agent-level audit",          value=False, disabled=use_team_agent)

    st.markdown("<hr style='border-color:#222;margin:0.8rem 0'>", unsafe_allow_html=True)

    st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">🛡 Privacy Awareness</p>', unsafe_allow_html=True)
    privacy_level = st.selectbox(
        "Privacy Awareness",
        options=["None", "Low", "Medium", "High"],
        index=3,
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#222;margin:0.8rem 0'>", unsafe_allow_html=True)
    with st.expander("⚙ Advanced", expanded=False):
        write_llm = st.selectbox("Write / Cleanup LLM", MODELS,
                                 index=MODELS.index("google/gemini-2.5-flash-lite"),
                                 help="LLM used for memory write and cleanup calls.")
        st.caption("Verifier ensemble — 3 models, majority vote")
        verifier_model_1 = st.selectbox("Verifier LLM 1", MODELS,
                                        index=MODELS.index("anthropic/claude-haiku-4-5"))
        verifier_model_2 = st.selectbox("Verifier LLM 2", MODELS,
                                        index=MODELS.index("openai/gpt-4o-mini"))
        verifier_model_3 = st.selectbox("Verifier LLM 3", MODELS,
                                        index=MODELS.index("google/gemini-2.5-flash"))

    run_btn = st.button("▶  Run Pipeline", type="primary", use_container_width=True)

from judges import set_verifier_models
set_verifier_models([verifier_model_1, verifier_model_2, verifier_model_3])

# ── Settings change detection — auto-clear stale results ─────────────────────
_RESULT_KEYS = ["team_gathered", "team_response", "team_prompt_decision", "team_pipeline_time", "team_privacy", "team_decision",
                "token_result", "tc_result", "agent_contents", "routing_map",
                "memory_violations"]

_current_settings = {
    "system_design":            system_design,
    "agent_llm":                agent_llm,
    "write_llm":                write_llm,
    "use_private_memory":       use_private_memory,
    "use_shared_memory":        use_shared_memory,
    "shared_memory_writer":     shared_memory_writer_flag,
    "use_memory_cleanup":       use_memory_cleanup,
    "judge_llm":                judge_llm,
    "verifier_model_1":         verifier_model_1,
    "verifier_model_2":         verifier_model_2,
    "verifier_model_3":         verifier_model_3,
    "privacy_level":            privacy_level,
    "lenient_context_matching": lenient_context_matching,
    "run_agent_level_audit":    run_agent_level_audit,
}
if st.session_state.get("_last_settings") != _current_settings:
    st.session_state._last_settings = _current_settings
    st.session_state.pipeline_started = False
    st.session_state.artifact_idx = 0
    for key in _RESULT_KEYS:
        st.session_state.pop(key, None)

# Handle run button immediately before any rendering
if run_btn:
    st.session_state.pipeline_started = True
    st.session_state.artifact_idx = 0
    for key in _RESULT_KEYS:
        st.session_state.pop(key, None)
    st.rerun()

# ── Helpers ───────────────────────────────────────────────────────────────────

def code_box(content: str, raw_html: bool = False) -> str:
    import html as _html
    body = content if raw_html else _html.escape(content, quote=False)
    return f'<div style="background:{UI_BLACK};padding:1rem;border-radius:8px;font-family:monospace;font-size:0.85rem;color:{UI_TEXT};white-space:pre-wrap;overflow-x:auto;border:1px solid #222">{body}</div>'

# Section color mapping: header keyword → (bg, text, label)
_SECTION_COLORS = {
    "your identity":            ("#1a1a1a", "#888888", "fixed"),
    "organizational context":   ("#1a1a1a", "#888888", "fixed"),
    "people":                   ("#1a1a1a", "#888888", "fixed"),
    "relationships":            ("#1a1a1a", "#888888", "fixed"),
    "colleagues you can reach": ("#1a1a1a", "#888888", "fixed"),
    "interactions":             ("#0d1f0d", UI_GREEN,  "preloaded context"),
    "conversation so far":      ("#0d0d2a", "#7090ff", "communication"),
    "communication":            ("#0d0d2a", "#7090ff", "communication"),
    "task":                     ("#1a0d00", "#c87830", "task"),
    "message from":             ("#0d0d2a", "#7090ff", "communication"),
    "my memory":                ("#1a0d2a", "#cc88ff", "memory"),
    "memory":                   ("#1a0d2a", "#cc88ff", "memory"),
    "what you already remember": ("#1a0d2a", "#cc88ff", "memory"),
    "shared memory":            ("#0d0a2a", "#8877ff", "shared memory"),
    "current shared memory":    ("#0d0a2a", "#8877ff", "shared memory"),
    "instructions":             ("#1a1a1a", "#888888", "fixed"),
    "your perspective":         ("#1a1a1a", "#888888", "fixed"),
    "candidates":               ("#0d1a1f", "#55a0bb", "ci candidates"),
    "privacy instruction":      ("#1a0a1a", "#cc66aa",  "privacy standard"),
    "information attributes":   ("#0a1a1a", "#22aaaa",  "attr schema"),
    "attributes":               ("#0d1f0d", UI_GREEN,  "preloaded context"),
    "agent content":            ("#0d0d2a", "#7090ff", "communication"),
    "message":                  ("#0d0d2a", "#7090ff", "communication"),
    "latest message":           ("#0d0d2a", "#7090ff", "communication"),
    "last interaction":         ("#0d0d2a", "#7090ff", "communication"),
    "interaction":              ("#0d1f0d", UI_GREEN,  "preloaded context"),
    "response":                 ("#0d0d2a", "#7090ff", "communication"),
    "allowed decisions":        ("#1a1a1a", "#888888", "fixed"),
    "decision rule":            ("#1a1a1a", "#888888", "fixed"),
}

def colored_prompt_box(prompt: str) -> str:
    import re
    lines   = prompt.split("\n")
    parts   = []
    current_bg   = "#111111"
    current_fg   = UI_TEXT
    current_lines = []

    def flush():
        if current_lines:
            content = "\n".join(current_lines)
            import html as _html
            _A2A_BOUNDARY = "*** AGENT-TO-AGENT COMMUNICATION ***"
            _base = (f'background:{current_bg};padding:0.5rem 0.8rem;margin:1px 0;'
                     f'font-family:monospace;font-size:0.82rem;white-space:pre-wrap;'
                     f'color:{current_fg}')
            if _A2A_BOUNDARY in content:
                idx           = content.index(_A2A_BOUNDARY)
                pre_content   = content[:idx]
                post_content  = content[idx:]
                entries_idx   = post_content.find("\n\n[")
                if entries_idx > 0:
                    warning_content = post_content[:entries_idx]
                    entries_content = post_content[entries_idx:]
                else:
                    warning_content = post_content
                    entries_content = ""
                if pre_content.strip():
                    parts.append(f'<div style="{_base}">{_html.escape(pre_content, quote=False)}</div>')
                if warning_content.strip():
                    parts.append(
                        f'<div style="background:{current_bg};padding:0.2rem 0.8rem;margin:1px 0;'
                        f'font-family:monospace;font-size:0.76rem;white-space:pre-wrap;color:{UI_TEXT}">'
                        f'{_html.escape(warning_content, quote=False)}</div>'
                    )
                if entries_content.strip():
                    parts.append(
                        f'<div style="{_base};font-style:italic">'
                        f'{_html.escape(entries_content, quote=False)}</div>'
                    )
            else:
                parts.append(f'<div style="{_base}">{_html.escape(content, quote=False)}</div>')

    # Sections that contain embedded user/artifact content — unknown ## headers inside
    # them should not trigger re-coloring; only known _SECTION_COLORS headers exit opaque mode.
    _OPAQUE_SECTIONS = {"interactions", "message", "response", "agent content",
                        "communication", "conversation so far", "message from",
                        "my memory", "memory", "what you already remember",
                        "shared memory", "current shared memory", "candidates"}
    inside_opaque = False
    used_labels = set()
    for line in lines:
        # Detect section header (# or ##)
        m = re.match(r"^#{1,2}\s+(.+)", line)
        if m:
            header = m.group(1).lower()
            matched = next((v for k, v in _SECTION_COLORS.items() if k in header), None)
            if inside_opaque and not matched:
                # Unknown header inside opaque section — treat as content, don't re-color
                current_lines.append(line)
                continue
            flush()
            current_lines = [line]
            if matched:
                current_bg, current_fg, lbl = matched
                used_labels.add(lbl)
                inside_opaque = any(s in header for s in _OPAQUE_SECTIONS)
            else:
                current_bg, current_fg = "#111111", UI_TEXT
                inside_opaque = False
        else:
            current_lines.append(line)

    flush()

    _all_legend = {
        "fixed":             ("#1a1a1a", "#888888"),
        "preloaded context": ("#0d1f0d", UI_GREEN),
        "communication":     ("#0d0d2a", "#7090ff"),
        "task":              ("#1a0d00", "#c87830"),
        "memory":            ("#1a0d2a", "#cc88ff"),
        "shared memory":     ("#0d0a2a", "#8877ff"),
        "ci candidates":     ("#0d1a1f", "#55a0bb"),
        "privacy standard":  ("#1a0a1a", "#cc66aa"),
    }
    legend = " &nbsp; ".join(
        f'<span style="background:{bg};color:{fg};padding:1px 6px;border-radius:3px;'
        f'font-size:0.72rem;font-family:monospace">{label}</span>'
        for label, (bg, fg) in _all_legend.items()
        if label in used_labels
    )

    return (
        f'<div style="margin-bottom:4px">{legend}</div>'
        f'<div style="border:1px solid #222;border-radius:8px;overflow:hidden">'
        + "".join(parts) +
        f'</div>'
    )

def _render_mem_entries(entries: dict, empty_color: str = "#555"):
    if entries:
        st.markdown(
            "\n\n".join(
                (
                    f'<span style="color:#cc88ff;font-family:monospace;font-size:0.82rem">{k}</span>'
                    if not v else
                    f'<span style="color:#cc88ff;font-family:monospace;font-size:0.82rem"><b>{k}</b>:</span>'
                    f' <span style="color:{UI_TEXT};font-size:0.82rem">{v}</span>'
                )
                for k, v in entries.items()
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="color:{empty_color};font-size:0.8rem;font-style:italic">(none)</span>',
            unsafe_allow_html=True,
        )




def _render_log_entry_cols(entry: dict, use_private_memory: bool, use_shared_memory: bool,
                           pipeline_time: float = None):
    """Render [Prompt | Message | 0-4 memory cols] for one log entry.

    Memory columns show both this agent's writes AND the peer's writes for the same interaction.
    """
    agent      = entry.get("agent", "")
    peer_name  = entry.get("peer_name", "")

    _priv_w       = entry.get("private_memory_written") or {}
    _shared_w     = entry.get("shared_memory_written") or {}
    _priv_snap    = entry.get("private_memory_snapshot")
    _sh_snap      = entry.get("shared_memory_snapshot")
    _write_prompt = entry.get("write_prompt", "")

    _peer_priv_w      = entry.get("peer_private_memory_written") or {}
    _peer_shared_w    = entry.get("peer_shared_memory_written") or {}
    _peer_priv_snap   = entry.get("peer_private_memory_snapshot")
    _peer_write_prompt = entry.get("peer_write_prompt", "")

    _mem_cols: list = []
    if use_private_memory:
        _mem_cols.append("agent_private")
    if use_shared_memory:
        _mem_cols.append("agent_shared")
    if use_private_memory and peer_name:
        _mem_cols.append("peer_private")
    if use_shared_memory and peer_name:
        _mem_cols.append("peer_shared")

    cols = st.columns(2 + len(_mem_cols))

    with cols[0]:
        with st.expander("📋 Prompt", expanded=False):
            st.markdown(colored_prompt_box(entry.get("prompt", "")), unsafe_allow_html=True)
        if _write_prompt:
            with st.expander("📝 Sender Write Prompt", expanded=False):
                st.markdown(colored_prompt_box(_write_prompt), unsafe_allow_html=True)
        if _peer_write_prompt:
            with st.expander("📝 Receiver Write Prompt", expanded=False):
                st.markdown(colored_prompt_box(_peer_write_prompt), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(code_box(entry.get("message", "")), unsafe_allow_html=True)
        if pipeline_time:
            st.caption(f"⏱ pipeline: {pipeline_time:.1f}s total")

    for _ci, _col_type in enumerate(_mem_cols, start=2):
        with cols[_ci]:
            if _col_type == "agent_private":
                _render_mem_col(f"🔒 {agent} — Private Memory", _priv_w, snapshot=_priv_snap)
            elif _col_type == "agent_shared":
                _render_mem_col(f"🌐 {agent} → Shared", _shared_w, snapshot=_sh_snap)
            elif _col_type == "peer_private":
                _render_mem_col(f"🔒 {peer_name} — Private Memory", _peer_priv_w, snapshot=_peer_priv_snap)
            elif _col_type == "peer_shared":
                _render_mem_col(f"🌐 {peer_name} → Shared", _peer_shared_w, snapshot=_sh_snap)


def _render_mem_col(label: str, entries: dict, snapshot: dict = None):
    st.markdown(
        f'<p style="color:#cc88ff;font-size:0.75rem;letter-spacing:1px;'
        f'text-transform:uppercase;margin-bottom:4px">{label}</p>',
        unsafe_allow_html=True,
    )
    _render_mem_entries(entries)
    if snapshot is not None:
        with st.expander("📚 Updated memory", expanded=False):
            _render_mem_entries(snapshot, empty_color="#444")


def highlight_vars(text: str) -> str:
    import re
    return re.sub(
        r"\{(\w+)\}",
        rf'<span style="color:{UI_GREEN};font-weight:bold">{{{{\1}}}}</span>',
        text,
    )


# ── Shared render helpers ─────────────────────────────────────────────────────

def _attr_bullets(revealed_obj, ground_truth=None):
    if not isinstance(revealed_obj, dict):
        return ""
    if revealed_obj.get("status") == "not_extracted":
        return ""
    v       = revealed_obj.get("value", "")
    e       = revealed_obj.get("explanation", "")
    correct = revealed_obj.get("correct")
    gt_line = (
        f'<br><span style="color:#666;font-size:0.75rem;padding-left:1.2rem">• ground truth: {ground_truth}</span>'
    ) if ground_truth is not None else ""
    if correct == "correct":
        v_icon, v_color = "✅", UI_GREEN
    elif correct == "incorrect":
        v_icon, v_color = "❌", "#c0392b"
    elif correct == "uncertain":
        v_icon, v_color = "⚠️", "#c87830"
    else:
        v_icon, v_color = "", "#aaa"
    verified_line = (
        f'<br><span style="color:{v_color};font-size:0.75rem;padding-left:1.2rem">• verified: {v_icon} {correct}</span>'
    ) if correct is not None else ""
    return (
        f'<br><span style="color:#aaa;font-size:0.75rem;padding-left:1.2rem">• value: {v}</span>'
        f'{gt_line}'
        f'{verified_line}'
        f'<br><span style="color:#aaa;font-style:italic;font-size:0.75rem;padding-left:1.2rem">• explanation: {e}</span>'
    ) if (v or e) else ""

def _attr_state(obj):
    """Returns 'found', 'null', or 'sentinel' for a judge result object."""
    if isinstance(obj, dict) and obj.get("status") == "not_extracted":
        return "sentinel"
    if isinstance(obj, dict) and obj.get("value") is not None:
        return "found"
    return "null"


def _pick_agent_template(use_private_memory, use_shared_memory):
    from agents import _AGENT_BODY_NO_MEM, _AGENT_BODY_PRIV, _AGENT_BODY_SH, _AGENT_BODY_BOTH
    if use_private_memory and use_shared_memory: return _AGENT_BODY_BOTH
    if use_private_memory: return _AGENT_BODY_PRIV
    if use_shared_memory:  return _AGENT_BODY_SH
    return _AGENT_BODY_NO_MEM

def _pick_instructions():
    """Executor + peer instruction blocks for the TP prompt-template preview."""
    from agents import _EXECUTOR_INSTRUCTIONS, _TP_PEER_INSTRUCTIONS
    return _EXECUTOR_INSTRUCTIONS, _TP_PEER_INSTRUCTIONS


def build_preload(arch_prefix, agents_map, timeline, scenario, use_private_memory, use_shared_memory, idx, shared_memory_writer="all", use_memory_cleanup=False, write_llm=None, memory_visibility_instruction=""):
    from agents import preload_artifacts_with_memory, _CLEANUP_LLM_DEFAULT
    _write_llm = write_llm or _CLEANUP_LLM_DEFAULT
    if use_private_memory or use_shared_memory:
        cache_key = (arch_prefix, idx, use_private_memory, use_shared_memory, shared_memory_writer, use_memory_cleanup, _write_llm, memory_visibility_instruction)
        k_key      = f"_{arch_prefix}_preload_key"
        k_mem      = f"_{arch_prefix}_private_memories"
        k_smem     = f"_{arch_prefix}_shared_memory"
        k_log      = f"_{arch_prefix}_preload_log"
        k_artifacts = f"_{arch_prefix}_agent_artifacts"
        if st.session_state.get(k_key) == cache_key:
            # Shared-only: re-populate agent contexts with artifacts (lost when agents_map is rebuilt)
            # Private memory agents rely on memory notes, not raw artifacts in context
            if use_shared_memory and not use_private_memory:
                artifact_by_id = {a["id"]: a for a in timeline}
                for agent_name, artifact_ids in st.session_state.get(k_artifacts, {}).items():
                    if agent_name in agents_map:
                        for art_id in artifact_ids:
                            if art_id in artifact_by_id:
                                agents_map[agent_name].receive_artifact(artifact_by_id[art_id])
            return st.session_state[k_mem], st.session_state[k_smem], st.session_state[k_log]
        private_memories, shared_memory, preload_log = preload_artifacts_with_memory(
            agents_map, timeline, scenario,
            use_private_memory=use_private_memory,
            use_shared_memory=use_shared_memory,
            up_to=idx,
            shared_memory_writer=shared_memory_writer,
            use_memory_cleanup=use_memory_cleanup,
            write_llm=_write_llm,
            memory_visibility_instruction=memory_visibility_instruction,
        )
        st.session_state[k_key]      = cache_key
        st.session_state[k_mem]      = private_memories
        st.session_state[k_smem]     = shared_memory
        st.session_state[k_log]      = preload_log
        st.session_state[k_artifacts] = {
            name: list(agent.received_artifact_ids) for name, agent in agents_map.items()
        }
        return private_memories, shared_memory, preload_log
    private_memories, shared_memory, preload_log = preload_artifacts_with_memory(
        agents_map, timeline, scenario, use_private_memory=False, use_shared_memory=False, up_to=idx
    )
    return private_memories, shared_memory, preload_log


def render_artifact_stream(timeline, idx, agents_map, use_private_memory, use_shared_memory, preload_log, cast_members, key_prefix):
    private_by_artifact  = {}
    shared_by_artifact   = {}
    combined_by_artifact = {}
    for entry in preload_log:
        if entry.get("memory_type") == "combined":
            combined_by_artifact.setdefault(entry["artifact_id"], []).append(entry)
        elif entry.get("memory_type") == "shared":
            shared_by_artifact.setdefault(entry["artifact_id"], []).append(entry)
        else:
            private_by_artifact.setdefault(entry["artifact_id"], []).append(entry)

    for i, artifact in enumerate(timeline[:idx]):
        icon  = ARTIFACT_ICONS.get(artifact["type"], "📌")
        label = f"{icon} **{artifact['id'].upper()}** · {artifact['type'].replace('_', ' ').title()} · {artifact['author']}"
        _all_collapsed = st.session_state.get("artifact_all_collapsed", False)
        with st.expander(label, expanded=(i == idx - 1) and not _all_collapsed):
            render_artifact(artifact)
            receivers = [n for n in artifact["visible_to"] if n in agents_map]
            _load_verb = "Memorised by" if use_private_memory else "Loaded into"
            st.markdown(
                f'<p style="color:{UI_GREEN};font-size:0.75rem;margin-top:0.5rem">'
                f'{_load_verb}: {", ".join(receivers)}</p>',
                unsafe_allow_html=True,
            )
            if use_private_memory and use_shared_memory:
                for pe in combined_by_artifact.get(artifact["id"], []):
                    with st.expander(f"🧠🌐 {pe['agent']} — memory", expanded=False):
                        cols_c = st.columns(5)
                        with cols_c[0]:
                            with st.expander("📋 Preload Prompt", expanded=False):
                                st.markdown(colored_prompt_box(pe["prompt"]), unsafe_allow_html=True)
                        with cols_c[1]:
                            _render_mem_col("🔒 Private Written", pe["memory_written"])
                        with cols_c[2]:
                            _render_mem_col("🧠 Private State", pe.get("memory_state", {}))
                        with cols_c[3]:
                            _render_mem_col("🌐 Shared Written", pe.get("shared_memory_written", {}))
                        with cols_c[4]:
                            _render_mem_col("🗂 Shared State", pe.get("shared_memory_state", {}))
            elif use_private_memory:
                for pe in private_by_artifact.get(artifact["id"], []):
                    with st.expander(f"🧠 {pe['agent']} — private memory", expanded=False):
                        cols_pre = st.columns(3)
                        with cols_pre[0]:
                            with st.expander("📋 Preload Prompt", expanded=False):
                                st.markdown(colored_prompt_box(pe["prompt"]), unsafe_allow_html=True)
                        with cols_pre[1]:
                            _render_mem_col("🔒 Written to Memory", pe["memory_written"])
                        with cols_pre[2]:
                            _render_mem_col("🧠 Memory State", pe.get("memory_state", {}))
            if use_shared_memory and not use_private_memory:
                for pe in shared_by_artifact.get(artifact["id"], []):
                    with st.expander(f"🌐 {pe['agent']} — shared memory", expanded=False):
                        cols_sh = st.columns(3)
                        with cols_sh[0]:
                            with st.expander("📋 Preload Prompt", expanded=False):
                                st.markdown(colored_prompt_box(pe["prompt"]), unsafe_allow_html=True)
                        with cols_sh[1]:
                            _render_mem_col("🌐 Written to Shared Memory", pe["memory_written"])
                        with cols_sh[2]:
                            _render_mem_col("🗂 Shared Memory State", pe.get("memory_state", {}))


def render_nav_buttons(idx, timeline, result_key, extra_reset_pops=None):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 5])
    with c1:
        if st.button("▶ Next", disabled=idx >= len(timeline)):
            st.session_state.artifact_idx += 1
            st.session_state.artifact_all_collapsed = False
            st.session_state.pop(result_key, None)
            st.rerun()
    with c2:
        if st.button("⏭ All"):
            st.session_state.artifact_idx = len(timeline)
            st.session_state.artifact_all_collapsed = True
            st.session_state.pop(result_key, None)
            st.rerun()
    with c3:
        if st.button("↺ Reset"):
            st.session_state.artifact_idx = 0
            st.session_state.artifact_all_collapsed = False
            st.session_state.pop(result_key, None)
            for k in (extra_reset_pops or []):
                st.session_state.pop(k, None)
            st.rerun()
    with c4:
        st.progress(idx / len(timeline), text=f"{idx} / {len(timeline)} artifacts")


def render_task_block(scenario):
    from agents import _resolve_task_text
    task = scenario["task"]
    # Resolve from both sources, exactly like the pipeline: task["description"]
    # combined with the timeline "task" artifact.
    desc = _resolve_task_text(scenario).strip()
    with st.expander(f"📌  {task['type'].replace('_', ' ').title()}", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Description**")
            st.markdown(f"_{desc or '(no description)'}_")
            st.markdown(f"**Subject:** {task['subject']['name']} `{task['subject']['role']}`")
        with col2:
            st.markdown("**Participants**")
            for role, p in task["participants"].items():
                st.markdown(f"- `{role}` → {p['name']} ({p['role']})")


def render_prompt_templates(template, exec_instructions="", peer_instructions="", peer_label="Peer"):
    from agents import _AGENT_HEADER
    with st.expander("📋 Prompt Template", expanded=False):
        col_exec, col_peer = st.columns(2)
        exec_full = _AGENT_HEADER + template.replace("{instructions}", exec_instructions)
        peer_full = _AGENT_HEADER + template.replace("{instructions}", peer_instructions)
        with col_exec:
            st.markdown("**Executor**")
            st.markdown(code_box(highlight_vars(exec_full), raw_html=True), unsafe_allow_html=True)
        with col_peer:
            st.markdown(f"**{peer_label}**")
            st.markdown(code_box(highlight_vars(peer_full), raw_html=True), unsafe_allow_html=True)


def capture_run_state(agents_map, use_private_memory, preload_log):
    st.session_state.agent_contents = {
        name: a.get_full_content() for name, a in agents_map.items()
    }
    if preload_log:
        rm = {name: [] for name in agents_map}
        for e in preload_log:
            if e["agent"] in rm and e["artifact_id"] not in rm[e["agent"]]:
                rm[e["agent"]].append(e["artifact_id"])
        st.session_state.routing_map = rm
    else:
        st.session_state.routing_map = {
            name: a.received_artifact_ids[:] for name, a in agents_map.items()
        }


def render_privacy_evaluation(response, utility_data, appropriateness_data, judge_llm, api_key, privacy=None, gathered_info=None, cast_context=""):
    from judges import judge_privacy
    if privacy is None:
        with st.spinner("Running privacy judge…"):
            privacy = judge_privacy(gathered_info, utility_data, appropriateness_data, judge_llm, api_key, cast_context=cast_context)
    with st.expander("🔒 Privacy Leakage (gathered-info summary)", expanded=True):
        cols = st.columns(2)
        cols[0].metric("🚨 V_G — Gathering violation",
                  f"{privacy['violation_rate']:.0%}",
                  delta=f"{len(privacy['violations'])} / {privacy['n_inappropriate']} inappropriate attrs leaked",
                  delta_color="inverse")
        cols[0].caption(f"⏱ {privacy.get('judge_time', 0):.1f}s")
        cols[1].metric("✅ C — Completeness",
                  f"{privacy['completeness']:.0%}",
                  delta=f"{len(privacy['covered'])} / {privacy['n_appropriate']} appropriate attrs gathered")

        _p_cols = st.columns(2)
        with _p_cols[0]:
            with st.expander("📋 Judge Prompt — V_G", expanded=False):
                st.markdown(colored_prompt_box(privacy["prompt"]), unsafe_allow_html=True)
            with st.expander("💬 LLM Output — V_G", expanded=False):
                st.markdown(code_box(privacy.get("raw", "")), unsafe_allow_html=True)
        with _p_cols[1]:
            with st.expander("📋 Judge Prompt — Completeness", expanded=False):
                st.markdown(colored_prompt_box(privacy.get("completeness_prompt", privacy["prompt"])), unsafe_allow_html=True)
            with st.expander("💬 LLM Output — Completeness", expanded=False):
                _raw = privacy.get("completeness_raw", "")
                st.markdown(code_box("\n\n".join(f"── run {i+1} ──\n{r}" for i, r in enumerate(_raw)) if isinstance(_raw, list) else _raw), unsafe_allow_html=True)

        st.markdown("<hr style='border-color:#1a1a1a;margin:0.6rem 0'>", unsafe_allow_html=True)
        _b_cols  = st.columns(2)
        col_v    = _b_cols[0]
        col_c    = _b_cols[1]

        def _render_attr_list(col, header_html, revealed_dict, attr_filter):
            with col:
                st.markdown(header_html, unsafe_allow_html=True)
                for attr, verdict in appropriateness_data["attributes"].items():
                    if verdict != attr_filter:
                        continue
                    obj     = revealed_dict.get(attr)
                    state   = _attr_state(obj)
                    correct = obj.get("correct") if isinstance(obj, dict) else None
                    if attr_filter == "inappropriate":
                        if state == "found":
                            if correct == "correct":   icon, action, color = "❌", "revealed",              "#c0392b"
                            elif correct == "incorrect": icon, action, color = "✅", "claimed (wrong value)", "#555"
                            else:                      icon, action, color = "⚠️", "revealed (uncertain)",  "#c87830"
                        elif state == "sentinel":      icon, action, color = "⚠️", "not extracted",         "#c87830"
                        else:                          icon, action, color = "✅", "🔒 hidden",              "#555"
                    else:
                        if state == "found":
                            if correct == "correct":   icon, action, color = "✅", "revealed",              UI_GREEN
                            elif correct == "incorrect": icon, action, color = "❌", "revealed (wrong value)","#c0392b"
                            else:                      icon, action, color = "⚠️", "revealed (uncertain)",  "#c87830"
                        elif state == "sentinel":      icon, action, color = "⚠️", "not extracted",         "#c87830"
                        else:                          icon, action, color = "❌", "missing",                "#555"
                    _gt     = utility_data.get("attribute_values", {}).get(attr, {}).get("value")
                    bullets = _attr_bullets(obj, ground_truth=_gt) if state == "found" else ""
                    st.markdown(
                        f'{icon} <span style="color:{UI_TEXT}">{attr}</span>'
                        f' <span style="color:{color};font-size:0.8rem">→ {action}</span>{bullets}',
                        unsafe_allow_html=True,
                    )

        _render_attr_list(
            col_v,
            f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">🚨 V_G — should hide</p>',
            privacy["revealed"],
            "inappropriate",
        )
        _render_attr_list(
            col_c,
            f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">✅ Completeness <span style="color:#555;font-size:0.7rem;text-transform:none">(gathered-info)</span></p>',
            privacy.get("completeness_revealed", privacy["revealed"]),
            "appropriate",
        )
    return privacy


def render_utility_evaluation(response, scenario, utility_data, judge_llm, api_key, decision=None):
    from judges import judge_decision
    with st.expander("🎯 Utility", expanded=True):
        if decision is None:
            with st.spinner("Running decision judge…"):
                decision = judge_decision(response, scenario["task"], utility_data, judge_llm, api_key)
        st.markdown("**Decision Correctness**")
        st.markdown(f"Extracted: `{decision['decision']}`  |  Oracle: `{decision['oracle']}`")
        if decision["correct"]:
            st.success(f"✅ Correct — {decision['explanation']}")
        else:
            st.error(f"❌ Incorrect — {decision['explanation']}")
        st.caption(f"⏱ {decision.get('_time', 0):.1f}s")
        with st.expander("📋 Judge Prompt — Decision", expanded=False):
            st.markdown(colored_prompt_box(decision.get("prompt", "")), unsafe_allow_html=True)
    return decision




def render_agent_level_audit(agent_contents, executor_name, visibility_data, utility_data, appropriateness_data, judge_llm, api_key, lenient=True, cast_context="", system=None, gathered_info=None, use_private_memory=False):
    import pandas as pd
    from judges import judge_agent_knowledge

    audit_results = {}
    with st.expander("🔍 Agent-Level Audit", expanded=True):
        if not agent_contents:
            st.warning("Agent contents not available. Re-run the task to enable audit.")
            return audit_results

        st.markdown(
            "<p style='color:#888;font-size:0.85rem'>For each agent, the judge tries to extract "
            "every attribute from what the agent has seen. Any attribute found that the agent "
            "should NOT know is flagged as a violation.</p>",
            unsafe_allow_html=True,
        )

        # M2 — match the CLI evaluator (run_evaluation.py): in Centralized with memory off, the
        # gathered-info summary the Coordinator hands the executor is part of the executor's
        # context, so fold it in (as a pseudo-A2A from the Coordinator) before auditing V_A.
        # With private memory on, this leakage is already captured via V_PMem/V_SMem.
        if system == "centralized" and gathered_info and not use_private_memory and executor_name in agent_contents:
            from datetime import datetime, timezone
            _ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            agent_contents = dict(agent_contents)
            agent_contents[executor_name] = (
                agent_contents.get(executor_name, "")
                + f"\n\n[{_ts}] agent2agent — Coordinator\n{gathered_info}"
            ).strip()

        for agent_name, content in agent_contents.items():
            with st.spinner(f"Auditing {agent_name}…"):
                audit = judge_agent_knowledge(
                    agent_name, content, visibility_data, utility_data, judge_llm, api_key, lenient=lenient, cast_context=cast_context
                )
            audit_results[agent_name] = audit

            role_tag = "executor" if agent_name == executor_name else "peer"
            # V_A counts every hidden attribute that surfaces in the agent's context, regardless
            # of source (artifact or a2a) — matches the CLI evaluator and the paper's V_A.
            viols    = audit.get("violations", [])
            vcount   = len(viols)
            n_hidden = audit["n_hidden"]
            vrate    = (vcount / n_hidden) if n_hidden else 0.0
            color    = "#c0392b" if vcount > 0 else UI_GREEN

            st.markdown(
                f"<b style='color:{UI_TEXT}'>{agent_name}</b> "
                f"<span style='color:#555;font-size:0.8rem'>({role_tag})</span> — "
                f"<span style='color:{color}'>{vcount} violation{'s' if vcount != 1 else ''} / {n_hidden} hidden attrs "
                f"({vrate:.0%})</span>"
                f"<span style='color:#444;font-size:0.72rem'> ⏱ {audit.get('_time', 0):.1f}s</span>",
                unsafe_allow_html=True,
            )

            with st.expander("📋 Judge Prompt", expanded=False):
                st.markdown(colored_prompt_box(audit["prompt"]), unsafe_allow_html=True)
            with st.expander("💬 LLM Output", expanded=False):
                st.markdown(code_box(audit.get("raw", "")), unsafe_allow_html=True)

            rows = []
            for attr, res in audit["per_attribute"].items():
                should   = res["should_know"]
                known    = res["known"]
                sentinel = res.get("is_sentinel", False)
                if should and known:
                    status = "✅ revealed"
                elif should and not known:
                    status = "⚠️ not extracted" if sentinel else "🔒 hidden"
                elif not should and not known:
                    status = "⚠️ not extracted" if sentinel else "✅ hidden"
                else:
                    status = "❌ revealed"
                _gt      = utility_data.get("attribute_values", {}).get(attr, {}).get("value", "—")
                _correct = res.get("correct")
                _correct_display = {"correct": "✅ correct", "incorrect": "❌ incorrect", "uncertain": "⚠️ uncertain"}.get(_correct, "—") if known else "—"
                rows.append({
                    "attribute":      attr,
                    "should know":    "yes" if should else "no",
                    "ground truth":   _gt,
                    "found by judge": res["extracted_value"] if known else ("(not extracted)" if sentinel else "—"),
                    "verified":       _correct_display,
                    "source":         res.get("source", "—") if known else "—",
                    "status":         status,
                })
            with st.expander("📋 Full Attribute Table", expanded=False):
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            violations_found = [
                (_attr, _res) for _attr, _res in audit["per_attribute"].items()
                if _res["violation"]
            ]
            if violations_found:
                st.markdown(
                    f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;'
                    f'text-transform:uppercase;margin:0.6rem 0 0.4rem 0">🚨 Violations</p>',
                    unsafe_allow_html=True,
                )
            for _attr, _res in violations_found:
                _val     = _res.get("extracted_value", "") or ""
                _src     = _res.get("source") or ""
                _expl    = _res.get("explanation", "") or ""
                _gt      = utility_data.get("attribute_values", {}).get(_attr, {}).get("value", "")
                _correct = _res.get("correct")
                _v_icon  = {"correct": "✅", "incorrect": "❌", "uncertain": "⚠️"}.get(_correct, "")
                _src_type, _, _src_id = _src.partition(" | ")
                _src_id = _src_id.strip()
                if _src_type.strip() == "artifact":
                    _src_html = f'🔴 <span style="color:#ff6b6b">artifact</span>' + (f' <span style="color:#888;font-size:0.72rem">| {_src_id}</span>' if _src_id else "")
                elif _src_type.strip() == "a2a":
                    _src_html = f'🟠 <span style="color:#ffaa44">a2a</span>' + (f' <span style="color:#888;font-size:0.72rem">| {_src_id}</span>' if _src_id else "")
                else:
                    _src_html = '<span style="color:#888">unknown</span>'
                st.markdown(
                    f'❌ <span style="color:{UI_TEXT}">{_attr}</span>'
                    f'<br><span style="color:#aaa;font-size:0.75rem;padding-left:1.2rem">• value: {_val}</span>'
                    f'<br><span style="color:#666;font-size:0.75rem;padding-left:1.2rem">• ground truth: {_gt}</span>'
                    f'<br><span style="color:#aaa;font-size:0.75rem;padding-left:1.2rem">• verified: {_v_icon} {_correct or "—"}</span>'
                    f'<br><span style="color:#aaa;font-size:0.75rem;padding-left:1.2rem">• source: {_src_html}</span>'
                    f'<br><span style="color:#aaa;font-style:italic;font-size:0.75rem;padding-left:1.2rem">• explanation: {_expl}</span>',
                    unsafe_allow_html=True,
                )

            st.markdown("<hr style='border-color:#1a1a1a;margin:0.4rem 0'>", unsafe_allow_html=True)

        # ── Summary table: all attributes, visibility-based ──────────────────
        all_attrs = list(appropriateness_data.get("attributes", {}).keys())
        if all_attrs and audit_results:
            st.markdown(
                f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;'
                f'text-transform:uppercase;margin:1rem 0 0.4rem 0">📊 Attribute Visibility Summary</p>',
                unsafe_allow_html=True,
            )
            summary_rows = []
            for attr in all_attrs:
                appropriateness = appropriateness_data["attributes"].get(attr, "—")
                violating = []
                for agent_name, audit in audit_results.items():
                    res = audit["per_attribute"].get(attr, {})
                    if res.get("violation"):
                        src = res.get("source", "") or ""
                        src_type, _, src_id = src.partition(" | ")
                        src_label = src_type.strip() + (f" ({src_id.strip()})" if src_id.strip() else "")
                        violating.append(f"{agent_name} [{src_label}]")
                violated = len(violating) > 0
                summary_rows.append({
                    "attribute":      attr,
                    "appropriateness": appropriateness,
                    "violated":       "❌ yes" if violated else "✅ no",
                    "revealed by":    ", ".join(violating) if violating else "—",
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    return audit_results


def render_memory_violations(
    private_memories, shared_memory,
    use_private_memory, use_shared_memory,
    visibility_data, utility_data, appropriateness_data,
    judge_llm, api_key, lenient=True, cast_context="",
):
    """Audit memory stores for violations; cache and display results. Returns memory_violations dict."""
    if not use_private_memory and not use_shared_memory:
        return None
    from judges import judge_memory_violations

    if "memory_violations" not in st.session_state:
        _pmems = private_memories if use_private_memory else {}
        _smem  = shared_memory if use_shared_memory else None
        with st.spinner("Auditing memory stores for violations…"):
            st.session_state.memory_violations = judge_memory_violations(
                private_memories=_pmems,
                shared_memory=_smem,
                visibility_data=visibility_data,
                utility_data=utility_data,
                model=judge_llm,
                api_key=api_key,
                lenient=lenient,
                cast_context=cast_context,
            )

    mv = st.session_state.memory_violations

    with st.expander("🔐 Memory Violations", expanded=True):
        st.markdown(
            "<p style='color:#888;font-size:0.85rem'>Checks whether memory stores contain "
            "attributes that should not be there: private memory is audited per-agent against "
            "that agent's visibility constraints; shared memory is audited against the union of "
            "all agents' constraints (any attribute hidden from at least one agent is a "
            "potential violation).</p>",
            unsafe_allow_html=True,
        )

        # ── Private memory ──────────────────────────────────────────────────────
        if use_private_memory:
            st.markdown(
                f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;'
                f'text-transform:uppercase;margin:0.4rem 0">🔑 Private Memory (per-agent)</p>',
                unsafe_allow_html=True,
            )
            private_res = mv.get("private", {})
            if not private_res:
                st.markdown("<span style='color:#555;font-size:0.83rem'>All private memories empty — nothing to audit.</span>", unsafe_allow_html=True)
            else:
                for agent_name, audit in private_res.items():
                    vcount   = len(audit["violations"])
                    n_hidden = audit["n_hidden"]
                    color    = "#c0392b" if vcount > 0 else UI_GREEN
                    st.markdown(
                        f"<b style='color:{UI_TEXT}'>{agent_name}</b> — "
                        f"<span style='color:{color}'>{vcount} violation{'s' if vcount != 1 else ''} / {n_hidden} hidden attrs "
                        f"({audit['violation_rate']:.0%})</span>"
                        f"<span style='color:#444;font-size:0.72rem'> ⏱ {audit.get('_time', 0):.1f}s</span>",
                        unsafe_allow_html=True,
                    )
                    for _attr in audit["violations"]:
                        _res  = audit["per_attribute"][_attr]
                        _val  = _res.get("extracted_value", "") or ""
                        _expl = _res.get("explanation", "") or ""
                        st.markdown(
                            f'❌ <span style="color:{UI_TEXT}">{_attr}</span>'
                            f'<br><span style="color:#aaa;font-size:0.75rem;padding-left:1.2rem">• found: {_val}</span>'
                            f'<br><span style="color:#aaa;font-style:italic;font-size:0.75rem;padding-left:1.2rem">• {_expl}</span>',
                            unsafe_allow_html=True,
                        )

        # ── Shared memory ───────────────────────────────────────────────────────
        if use_shared_memory:
            st.markdown(
                f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;'
                f'text-transform:uppercase;margin:0.8rem 0 0.4rem 0">🌐 Shared Memory</p>',
                unsafe_allow_html=True,
            )
            sr = mv.get("shared")
            if sr is None:
                st.markdown("<span style='color:#555;font-size:0.83rem'>Shared memory empty — nothing to audit.</span>", unsafe_allow_html=True)
            else:
                color = "#c0392b" if sr["violations"] else UI_GREEN
                st.markdown(
                    f"<span style='color:{color}'>{len(sr['violations'])} violation{'s' if len(sr['violations']) != 1 else ''} "
                    f"/ {sr['n_potential']} attrs visible to less than all agents "
                    f"({sr['violation_rate']:.0%})</span>"
                    f"<span style='color:#444;font-size:0.72rem'> ⏱ {sr.get('_time', 0):.1f}s</span>",
                    unsafe_allow_html=True,
                )
                if shared_memory:
                    with st.expander("📋 Shared Memory Contents", expanded=False):
                        st.markdown(code_box(shared_memory.render()), unsafe_allow_html=True)
                for _attr in sr["violations"]:
                    _res  = sr["per_attribute"][_attr]
                    _val  = _res.get("extracted_value", "") or ""
                    _expl = _res.get("explanation", "") or ""
                    st.markdown(
                        f'❌ <span style="color:{UI_TEXT}">{_attr}</span>'
                        f'<br><span style="color:#aaa;font-size:0.75rem;padding-left:1.2rem">• found: {_val}</span>'
                        f'<br><span style="color:#aaa;font-style:italic;font-size:0.75rem;padding-left:1.2rem">• {_expl}</span>',
                        unsafe_allow_html=True,
                    )

    return mv


from eval_utils import render_results_summary


# ── Load (always) ────────────────────────────────────────────────────────────

if not selected_scenario:
    st.info("⬅️ Select a scenario from the sidebar to begin.")
    st.stop()

visibility_data       = json.load(open(_paths["visibility"]))
appropriateness_data  = json.load(open(_paths["appropriateness"]))
scenario              = json.load(open(_paths["scenario"]))
utility_data          = json.load(open(_paths["utility"]))
cast_context          = build_cast_context(scenario)

if "artifact_idx" not in st.session_state:
    st.session_state.artifact_idx = 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — INPUTS
# ═══════════════════════════════════════════════════════════════════════════════

with st.expander("📂  Scenario", expanded=not st.session_state.get("pipeline_started")):

    with st.expander(f"👥  Cast  —  {len(scenario['cast'])} people", expanded=False):
        cast_rows = [
            {
                "Name": m["name"],
                "Role": m["role"],
                "Participant": "✅" if m["participant"] else "—",
                "Team": m["team"] or "—",
                "Background": m["background"],
            }
            for m in scenario["cast"].values()
        ]
        st.dataframe(cast_rows, use_container_width=True, hide_index=True)

    with st.expander(f"🏗  Org  —  {len(scenario['org']['teams'])} teams, {len(scenario['org']['relations'])} relations", expanded=False):
        for tm in scenario["org"]["teams"]:
            st.markdown(
                f'<div style="border-left:2px solid #333;padding:0.4rem 0.8rem;margin:0.3rem 0">'
                f'<b style="color:{UI_GREEN}">{tm["id"]}</b> &nbsp; manager: {tm["manager"]}'
                f'{"  |  members: " + ", ".join(tm["members"]) if tm["members"] else ""}'
                f'{"  |  collaborators: " + ", ".join(tm["collaborators"]) if tm["collaborators"] else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("<hr style='border-color:#1a1a1a;margin:0.5rem 0'>", unsafe_allow_html=True)
        for r in scenario["org"]["relations"]:
            st.markdown(
                f'<div style="padding:0.25rem 0">'
                f'<span style="color:{UI_TEXT}">{r["from"]}</span>'
                f'&nbsp;<span style="color:{UI_GREEN};font-family:monospace"> ──{r["type"]}──▶ </span>'
                f'<span style="color:{UI_TEXT}">{r["to"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with st.expander("🎯  Ground Truth", expanded=False):
        with st.expander("👁  Visibility Matrix", expanded=False):
            vis_rows = [
                {
                    "Attribute": attr,
                    "Visible To": ", ".join(item["visible_to"]) or "—",
                    "Hidden From": ", ".join(item["hidden_from"]) or "—",
                }
                for attr, item in visibility_data["attributes"].items()
            ]
            st.dataframe(vis_rows, use_container_width=True, hide_index=True)

        with st.expander("✅  Appropriateness Matrix", expanded=False):
            app_rows = [
                {
                    "Attribute": attr,
                    "Appropriate": verdict,
                }
                for attr, verdict in appropriateness_data["attributes"].items()
            ]
            st.dataframe(app_rows, use_container_width=True, hide_index=True)

        with st.expander("📍  Artifact Attributes", expanded=False):
            attr_to_artifacts = {}
            for a in scenario["timeline"]:
                for attr in a.get("attributes", []):
                    attr_to_artifacts.setdefault(attr, []).append(a["id"])
            attr_rows = [
                {
                    "Attribute": attr,
                    "Value": utility_data["attribute_values"].get(attr, {}).get("value", "—"),
                    "Appropriate": appropriateness_data["attributes"].get(attr, "—"),
                    "In Artifact(s)": ", ".join(attr_to_artifacts.get(attr, [])) or "—",
                }
                for attr in utility_data["attribute_values"]
            ]
            st.dataframe(attr_rows, use_container_width=True, hide_index=True)

        with st.expander("🎯  Utility", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">Oracle Answer</p>', unsafe_allow_html=True)
                st.markdown(f'`{utility_data["oracle_answer"]}`')
                st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin:8px 0 4px 0">Allowed Answers</p>', unsafe_allow_html=True)
                st.markdown(" · ".join(f"`{a}`" for a in utility_data["allowed_answers"]))
                st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin:8px 0 4px 0">Decision Rule</p>', unsafe_allow_html=True)
                st.markdown(f'_{utility_data["decision_rule"]}_')
            with col2:
                st.markdown(f'<p style="color:{UI_GREEN};font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;margin-bottom:4px">Task-Critical Values</p>', unsafe_allow_html=True)
                appropriate_attrs = [a for a, v in appropriateness_data["attributes"].items() if v == "appropriate"]
                for attr in appropriate_attrs:
                    val = utility_data["attribute_values"][attr]["value"]
                    st.markdown(f"- `{attr}` = **{val}**")

# ── Run guard ─────────────────────────────────────────────────────────────────

if not run_btn and not st.session_state.get("pipeline_started"):
    st.info("Configure the system in the sidebar, then press **▶ Run Pipeline** to begin.")
    st.stop()

if not api_key:
    st.error("OPENROUTER_API_KEY env variable not set.")
    st.stop()

timeline     = scenario["timeline"]
cast_members = [m["name"] for m in scenario["cast"].values()]

ARTIFACT_ICONS = {
    "hr_note": "📝",
    "email": "📧",
    "personal_draft_note": "🔒",
    "slack_thread": "💬",
    "internal_memo": "📄",
    "performance_form": "📋",
    "google_doc": "📃",
    "meeting_notes": "🗒",
}

def render_artifact(artifact):
    icon = ARTIFACT_ICONS.get(artifact["type"], "📌")
    attrs = artifact.get("attributes", [])
    inappropriate_attrs = set(k for k, v in appropriateness_data["attributes"].items() if v == "inappropriate")
    attr_badges = "".join(
        f'<span style="background:#3a0000;color:#ff6b6b;padding:1px 6px;border-radius:3px;'
        f'font-size:0.72rem;font-family:monospace;margin-right:4px">{a}</span>'
        if a in inappropriate_attrs else
        f'<span style="background:#0d1f0d;color:{UI_GREEN};padding:1px 6px;border-radius:3px;'
        f'font-size:0.72rem;font-family:monospace;margin-right:4px">{a}</span>'
        for a in attrs
    )
    attr_line = f'<br>{attr_badges}' if attrs else ""
    st.markdown(
        f'<div style="border-left:3px solid {UI_GREEN};padding:0.6rem 1rem;margin:0.5rem 0;background:#0d0d0d;border-radius:4px">'
        f'<div style="margin-bottom:0.3rem">'
        f'<span style="font-size:1rem">{icon}</span>'
        f'&nbsp;<b style="color:{UI_GREEN}">{artifact["type"].replace("_", " ").title()}</b>'
        f'&nbsp;<small style="color:#666">{artifact["timestamp"]}</small>'
        f'<br><small style="color:#888">from <b>{artifact["author"]}</b> → {", ".join(artifact["visible_to"])}</small>'
        f'{attr_line}'
        f'</div>'
        f'<span style="color:{UI_TEXT};font-family:monospace;font-size:0.82rem;white-space:pre-wrap">{artifact["content"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Single: one round, show all artifacts at once then run ────────────────
if use_team_agent:
    st.markdown("#### 📡 Artifacts")
    for artifact in timeline:
        icon  = ARTIFACT_ICONS.get(artifact["type"], "📌")
        label = f"{icon} **{artifact['id'].upper()}** · {artifact['type'].replace('_', ' ').title()} · {artifact['author']}"
        with st.expander(label, expanded=False):
            render_artifact(artifact)

    st.markdown("#### 🎯 Task")
    render_task_block(scenario)

    from agents import build_team_system, load_artifacts_team, TEAM_PROMPT_TEMPLATE, TEAM_GATHER_PROMPT_TEMPLATE, PRIVACY_INSTRUCTIONS
    from judges import judge_privacy, judge_decision

    privacy_instruction = PRIVACY_INSTRUCTIONS[privacy_level]
    agent = build_team_system(team_llm, api_key)
    load_artifacts_team(agent, timeline)

    with st.expander("📋 Gather Prompt Template", expanded=False):
        st.markdown(code_box(highlight_vars(TEAM_GATHER_PROMPT_TEMPLATE), raw_html=True), unsafe_allow_html=True)
    with st.expander("📋 Gather Prompt", expanded=False):
        st.markdown(colored_prompt_box(agent.build_gather_prompt(scenario)), unsafe_allow_html=True)
    with st.expander("📋 Decision Prompt Template", expanded=False):
        st.markdown(code_box(highlight_vars(TEAM_PROMPT_TEMPLATE), raw_html=True), unsafe_allow_html=True)
    if st.session_state.get("team_prompt_decision"):
        with st.expander("📋 Decision Prompt", expanded=False):
            st.markdown(colored_prompt_box(st.session_state.team_prompt_decision), unsafe_allow_html=True)

    if "team_response" not in st.session_state:
        if st.button("▶ Run Agent Task", type="primary"):
            with st.spinner("Running…"):
                import time as _time
                _t0 = _time.time()
                result = agent.run_task(scenario, privacy_instruction)
                gathered_info = result["gathered_info"]
                response      = result["response"]
                st.session_state.team_gathered          = gathered_info
                st.session_state.team_response          = response
                st.session_state.team_prompt_decision   = result["prompt_decision"]
                st.session_state.team_pipeline_time     = _time.time() - _t0
                st.session_state.team_privacy   = judge_privacy(
                    gathered_info, utility_data, appropriateness_data, judge_llm, api_key,
                    cast_context=cast_context,
                )
                st.session_state.team_decision  = judge_decision(response, scenario["task"], utility_data, judge_llm, api_key)
            st.rerun()
    else:
        gathered_info = st.session_state.team_gathered
        response      = st.session_state.team_response
        privacy       = st.session_state.team_privacy
        decision      = st.session_state.team_decision

        with st.expander("🗂 Stage 1 — Gathered Facts", expanded=False):
            st.markdown(code_box(gathered_info), unsafe_allow_html=True)
        with st.expander(f"🤖 Stage 2 — Final Response  ({team_llm})", expanded=True):
            st.markdown(code_box(response), unsafe_allow_html=True)

        st.markdown("#### 📊 Evaluation")
        render_privacy_evaluation(response, utility_data, appropriateness_data, judge_llm, api_key, privacy=privacy, cast_context=cast_context)
        render_utility_evaluation(response, scenario, utility_data, judge_llm, api_key, decision=decision)

        # Single agent has no agent-to-agent communication → no V_A2A.
        render_results_summary(
            privacy, decision,
            n_a2a=0,
            pipeline_time=st.session_state.get("team_pipeline_time"),
        )


# ── Decentralized ──────────────────────────────────────────────────────
elif use_token_passing:
    from agents import build_siloed_system, run_task_token_passing, PRIVACY_INSTRUCTIONS, MEMORY_VISIBILITY_INSTRUCTIONS
    st.markdown("#### 📡 Artifact Stream")
    idx        = st.session_state.artifact_idx
    agents_map = build_siloed_system(scenario["cast"], siloed_llm, api_key)
    private_memories, shared_memory, preload_log = build_preload("tp", agents_map, timeline, scenario, use_private_memory, use_shared_memory, idx, shared_memory_writer_flag, use_memory_cleanup, write_llm=write_llm, memory_visibility_instruction=MEMORY_VISIBILITY_INSTRUCTIONS[privacy_level])

    render_artifact_stream(timeline, idx, agents_map, use_private_memory, use_shared_memory, preload_log, cast_members, "tp")
    render_nav_buttons(idx, timeline, "token_result")

    if idx >= len(timeline):
        st.markdown("---")
        t             = scenario["task"]
        executor_role = t["executor_role"]
        executor_name = t["participants"][executor_role]["name"]

        st.markdown("#### 🎯 Task")
        render_task_block(scenario)

        _exec_instr, _peer_instr = _pick_instructions()
        render_prompt_templates(_pick_agent_template(use_private_memory, use_shared_memory), _exec_instr, _peer_instr.format(name="{name}"), peer_label="Peer")

        if "token_result" not in st.session_state:
            if st.button("▶ Run Agent Task", type="primary"):
                import time as _time
                with st.spinner("Running decentralized agent task…"):
                    _t0    = _time.time()
                    result = run_task_token_passing(agents_map, executor_name, scenario,
                                                    max_rounds=30,
                                                    use_private_memory=use_private_memory,
                                                    private_memories=private_memories,
                                                    shared_memory=shared_memory if use_shared_memory else None,
                                                    privacy_instruction=PRIVACY_INSTRUCTIONS[privacy_level],
                                                    memory_visibility_instruction=MEMORY_VISIBILITY_INSTRUCTIONS[privacy_level],
                                                    shared_memory_writer=shared_memory_writer_flag,
                                                    use_memory_cleanup=use_memory_cleanup,
                                                    write_llm=write_llm)
                    result["_pipeline_time"] = _time.time() - _t0
                st.session_state.token_result = result
                capture_run_state(agents_map, use_private_memory, preload_log)
                st.rerun()
        else:
            result   = st.session_state.token_result
            response = result["final_response"]

            st.markdown("#### 🎫 Decentralized — Message Log")
            for entry in result["log"]:
                is_final = entry["type"] == "final"
                if entry["type"] == "sent":
                    header = f"🎫 **{entry['agent']}** passes token → **{entry['to']}**"
                elif entry["type"] == "received":
                    # ⚠️ = target fell back to executor (parsing failure / invalid / self), not a deliberate choice
                    warn = " ⚠️ *(fallback — peer did not produce a valid TO:)*" if entry.get("target_was_fallback") else ""
                    header = f"🎫 **{entry['agent']}** holds token → passes to **{entry['to']}**{warn}"
                else:
                    header = f"🏁 **{entry['agent']}** — Final Answer"
                with st.expander(header, expanded=is_final):
                    _render_log_entry_cols(entry, use_private_memory, use_shared_memory)

            st.markdown("#### 📊 Evaluation")
            _gathered           = result.get("gathered_info")
            privacy             = render_privacy_evaluation(response, utility_data, appropriateness_data, judge_llm, api_key, gathered_info=_gathered, cast_context=cast_context)
            decision            = render_utility_evaluation(response, scenario, utility_data, judge_llm, api_key)

            audit_results  = render_agent_level_audit(st.session_state.get("agent_contents", {}), executor_name, visibility_data, utility_data, appropriateness_data, judge_llm, api_key, lenient=lenient_context_matching, cast_context=cast_context, system="decentralized", gathered_info=_gathered, use_private_memory=use_private_memory) if run_agent_level_audit else {}
            mem_violations = render_memory_violations(private_memories, shared_memory, use_private_memory, use_shared_memory, visibility_data, utility_data, appropriateness_data, judge_llm, api_key, lenient=lenient_context_matching, cast_context=cast_context)
            _n_a2a = len([e for e in result["log"] if e["type"] in ("sent", "received")])
            # V_A2A — log-based a2a audit, identical to the CLI evaluator (judge_a2a_log).
            from judges import judge_a2a_log
            _a2a_viol = judge_a2a_log(result["log"], appropriateness_data, utility_data, visibility_data,
                                      judge_llm, api_key, lenient=lenient_context_matching, cast_context=cast_context) if result.get("log") else None
            render_results_summary(privacy, decision, audit_results,
                                   a2a_violation=_a2a_viol, memory_violations=mem_violations,
                                   visibility_data=visibility_data,
                                   n_a2a=_n_a2a, pipeline_time=result.get("_pipeline_time"))


# ── Centralized ─────────────────────────────────────────────────────────
elif use_truly_centralized:
    from agents import build_siloed_system, run_task_truly_centralized, DECISION_PROMPT, PRIVACY_INSTRUCTIONS, MEMORY_VISIBILITY_INSTRUCTIONS
    st.markdown("#### 📡 Artifact Stream")
    idx        = st.session_state.artifact_idx
    agents_map = build_siloed_system(scenario["cast"], siloed_llm, api_key)
    # TC "executor only" → only Coordinator writes shared memory (during run, not preload)
    _tc_preload_shared = use_shared_memory and (shared_memory_writer_flag == "all")
    private_memories, shared_memory, preload_log = build_preload("tc", agents_map, timeline, scenario, use_private_memory, _tc_preload_shared, idx, shared_memory_writer_flag, use_memory_cleanup, write_llm=write_llm, memory_visibility_instruction=MEMORY_VISIBILITY_INSTRUCTIONS[privacy_level])

    render_artifact_stream(timeline, idx, agents_map, use_private_memory, _tc_preload_shared, preload_log, cast_members, "tc")
    render_nav_buttons(idx, timeline, "tc_result")

    if idx >= len(timeline):
        st.markdown("---")
        t             = scenario["task"]
        executor_role = t["executor_role"]
        executor_name = t["participants"][executor_role]["name"]

        st.markdown("#### 🎯 Task")
        render_task_block(scenario)

        from agents import _AGENT_HEADER, _AGENT_BODY_NO_MEM, _AGENT_BODY_SH, _COORDINATOR_INSTRUCTIONS, _PEER_INSTRUCTIONS, _COORDINATOR_NAME as _TC_COORD_NAME
        _body_tpl       = _pick_agent_template(use_private_memory, use_shared_memory)
        _coord_body_tpl = _AGENT_BODY_SH if use_shared_memory else _AGENT_BODY_NO_MEM
        _coord_tpl = _AGENT_HEADER + _coord_body_tpl.replace("{instructions}", _COORDINATOR_INSTRUCTIONS)
        _peer_tpl  = _AGENT_HEADER + _body_tpl.replace("{instructions}", _PEER_INSTRUCTIONS.format(sender="{sender}", name="{name}"))

        with st.expander("📋 Prompt Templates", expanded=False):
            _tc1, _tc2, _tc3 = st.columns(3)
            with _tc1:
                st.markdown("**Coordinator**")
                st.markdown(code_box(highlight_vars(_coord_tpl), raw_html=True), unsafe_allow_html=True)
            with _tc2:
                st.markdown("**Peer**")
                st.markdown(code_box(highlight_vars(_peer_tpl), raw_html=True), unsafe_allow_html=True)
            with _tc3:
                st.markdown("**Decision (Stage 2)**")
                st.markdown(code_box(highlight_vars(DECISION_PROMPT), raw_html=True), unsafe_allow_html=True)

        _tc_log  = (st.session_state.get("tc_result") or {}).get("log", [])
        _coord_e = next((e for e in _tc_log if e.get("type") == "sent" and e.get("agent") == _TC_COORD_NAME), None)
        _peer_e  = next((e for e in _tc_log if e.get("type") == "received"), None)
        _final_e = next((e for e in reversed(_tc_log) if e.get("type") == "final"), None)
        if any([_coord_e, _peer_e, _final_e]):
            with st.expander("📋 Actual Prompts", expanded=False):
                _ap1, _ap2, _ap3 = st.columns(3)
                with _ap1:
                    st.markdown("**Coordinator**")
                    st.markdown(colored_prompt_box(_coord_e["prompt"]) if _coord_e else code_box("(not run yet)"), unsafe_allow_html=True)
                with _ap2:
                    st.markdown("**Peer**")
                    st.markdown(colored_prompt_box(_peer_e["prompt"]) if _peer_e else code_box("(not run yet)"), unsafe_allow_html=True)
                with _ap3:
                    st.markdown("**Decision (Stage 2)**")
                    st.markdown(colored_prompt_box(_final_e["prompt"]) if _final_e else code_box("(not run yet)"), unsafe_allow_html=True)

        if "tc_result" not in st.session_state:
            if st.button("▶ Run Agent Task", type="primary"):
                import time as _time
                with st.spinner("Running centralized agent task…"):
                    _t0    = _time.time()
                    result = run_task_truly_centralized(agents_map, executor_name, scenario,
                                                       max_rounds=30,
                                                       use_private_memory=use_private_memory,
                                                       private_memories=private_memories,
                                                       shared_memory=shared_memory if use_shared_memory else None,
                                                       privacy_instruction=PRIVACY_INSTRUCTIONS[privacy_level],
                                                       memory_visibility_instruction=MEMORY_VISIBILITY_INSTRUCTIONS[privacy_level],
                                                       shared_memory_writer=shared_memory_writer_flag,
                                                       use_memory_cleanup=use_memory_cleanup,
                                                       write_llm=write_llm,
                                                       # coordinator extended-thinking is only wired for Sonnet (Anthropic);
                                                       # matches the CLI's --coordinator-thinking used for TC Sonnet runs.
                                                       coordinator_thinking=("claude-sonnet" in agent_llm.lower()))
                    result["_pipeline_time"] = _time.time() - _t0
                st.session_state.tc_result = result
                capture_run_state(agents_map, use_private_memory, preload_log)
                st.rerun()
        else:
            result   = st.session_state.tc_result
            response = result["final_response"]

            st.markdown("#### 💬 Agent Conversation")
            for entry in result["log"]:
                is_final = entry["type"] == "final"
                if entry["type"] == "step0":
                    continue
                elif entry["type"] == "sent":
                    header = f"💬 **{entry['agent']}** → **{entry['to']}**"
                elif entry["type"] == "received":
                    header = f"💬 **{entry['agent']}** → **{entry['to']}**"
                elif entry["type"] == "gathered":
                    header = f"📋 **{entry['agent']}** — Gathered Facts"
                else:
                    header = f"🏁 **{entry['agent']}** — Final Answer"
                with st.expander(header, expanded=is_final):
                    _render_log_entry_cols(
                        entry, use_private_memory, use_shared_memory,
                        pipeline_time=result.get("_pipeline_time") if is_final else None,
                    )

            st.markdown("#### 📊 Evaluation")
            _gathered           = result.get("gathered_info")
            privacy             = render_privacy_evaluation(response, utility_data, appropriateness_data, judge_llm, api_key, gathered_info=_gathered, cast_context=cast_context)
            decision            = render_utility_evaluation(response, scenario, utility_data, judge_llm, api_key)

            audit_results       = render_agent_level_audit(st.session_state.get("agent_contents", {}), executor_name, visibility_data, utility_data, appropriateness_data, judge_llm, api_key, lenient=lenient_context_matching, cast_context=cast_context, system="centralized", gathered_info=_gathered, use_private_memory=use_private_memory) if run_agent_level_audit else {}
            mem_violations      = render_memory_violations(private_memories, shared_memory, use_private_memory, use_shared_memory, visibility_data, utility_data, appropriateness_data, judge_llm, api_key, lenient=lenient_context_matching, cast_context=cast_context)
            _n_a2a = len([e for e in result["log"] if e["type"] in ("sent", "received")])
            # V_A2A — log-based a2a audit, identical to the CLI evaluator (judge_a2a_log).
            from judges import judge_a2a_log
            _a2a_viol = judge_a2a_log(result["log"], appropriateness_data, utility_data, visibility_data,
                                      judge_llm, api_key, lenient=lenient_context_matching, cast_context=cast_context) if result.get("log") else None
            render_results_summary(privacy, decision, audit_results,
                                   a2a_violation=_a2a_viol, memory_violations=mem_violations,
                                   visibility_data=visibility_data,
                                   n_a2a=_n_a2a, pipeline_time=result.get("_pipeline_time"))


