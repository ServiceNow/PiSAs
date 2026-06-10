"""
PiSAs — Evaluation runner (judging only; consumes pipeline_*.json from run_pipeline.py).

Two modes, selected by which flags you pass:

  Single scenario → -d/--scenario + -o/--output-dir
      python run_evaluation.py -d <scenario_dir> -o <output_dir> --judge-llm google/gemini-2.5-pro

  Batch (every pipeline_*_runN.json in a results folder, in parallel)
      python run_evaluation.py --scenarios-folder data --results-path results/run \\
          --judge-llm google/gemini-2.5-pro [--agent-audit] [--memory-audit]

Writes evaluation_<scenario>.json (single) / evaluation_<scenario>_runN.json (batch) with
the paper metrics in paper notation: V_G, V_A2A, V_appr, V_A, V_PMem, V_SMem, V_vis, C, U.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

log = logging.getLogger("evaluation")
log.setLevel(logging.DEBUG)
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_sh)

_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
W = 72


def _resolve_scenario(scenario_arg):
    if not scenario_arg:
        return None, None
    if os.path.isdir(scenario_arg):
        return os.path.basename(os.path.abspath(scenario_arg)), os.path.abspath(scenario_arg)
    fallback = os.path.join(_DATA_ROOT, scenario_arg)
    if os.path.isdir(fallback):
        return scenario_arg, os.path.abspath(fallback)
    return None, None


def _scenario_paths(base_dir: str) -> dict:
    return {
        "scenario":        os.path.join(base_dir, "scenario.json"),
        "utility":         os.path.join(base_dir, "utility.json"),
        "visibility":      os.path.join(base_dir, "visibility.json"),
        "appropriateness": os.path.join(base_dir, "appropriateness.json"),
    }


def _query_key_usage(api_key: str):
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())["data"]
            return {"usage": data.get("usage", 0.0), "limit": data.get("limit")}
    except Exception:
        return None


# ── Single-scenario evaluation ─────────────────────────────────────────────────

def run_single(args, api_key):
    """Judge one scenario's pipeline_<id>.json and write evaluation_<id>.json."""
    scenario_id, base_dir = _resolve_scenario(args.scenario)
    if scenario_id is None:
        print(f"ERROR: Scenario {args.scenario!r} not found.", file=sys.stderr)
        sys.exit(1)

    output_dir       = Path(args.output_dir).resolve()
    scenario_dir_out = output_dir / scenario_id
    pipeline_path    = scenario_dir_out / f"pipeline_{scenario_id}.json"
    judgment_path    = scenario_dir_out / f"evaluation_{scenario_id}.json"

    if not pipeline_path.exists():
        print(f"ERROR: pipeline file not found: {pipeline_path}\n  Run run_pipeline.py first.", file=sys.stderr)
        sys.exit(1)
    if judgment_path.exists() and not args.force:
        log.info(f"  ↺ {judgment_path.name} exists — skipping (use --force to overwrite).")
        return

    _fh = logging.FileHandler(scenario_dir_out / f"run_{scenario_id}.log", encoding="utf-8", mode="a")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_fh)

    _paths = _scenario_paths(base_dir)
    visibility_data      = json.load(open(_paths["visibility"]))
    appropriateness_data = json.load(open(_paths["appropriateness"]))
    scenario             = json.load(open(_paths["scenario"]))
    utility_data         = json.load(open(_paths["utility"]))

    t_task        = scenario["task"]
    executor_role = t_task["executor_role"]
    executor_name = t_task["participants"][executor_role]["name"]

    cast       = scenario.get("cast", {})
    _cast_list = cast if isinstance(cast, list) else list(cast.values())
    cast_context = "\n".join(
        f"- {m['name']}: {m.get('role', '').replace('_', ' ')}"
        + (f" ({m['team']} team)" if m.get("team") else "")
        for m in _cast_list
    )

    from judges import (judge_privacy, judge_decision,
                        judge_agent_knowledge, judge_memory_violations, judge_a2a_log,
                        set_verifier_models)
    set_verifier_models([args.verifier_llm_1, args.verifier_llm_2, args.verifier_llm_3])

    orch     = json.load(open(pipeline_path))
    orch_cfg = orch.get("config", {})

    log.info("═" * W)
    log.info("  ⬡ PiSAs  —  Evaluation")
    log.info(f"  Scenario    : {scenario_id}")
    log.info(f"  Pipeline    : {pipeline_path.name}")
    log.info(f"  Judge LLM   : {args.judge_llm}")
    log.info(f"  Verifiers   : {args.verifier_llm_1} | {args.verifier_llm_2} | {args.verifier_llm_3}")
    log.info(f"  Output audit: {'on' if args.output_audit else 'off'}  |  "
             f"Agent audit: {'on' if args.agent_audit else 'off'}  |  "
             f"Memory audit: {'on' if args.memory_audit else 'off'}")
    log.info("═" * W)

    response       = orch["final_response"]
    gathered       = orch.get("gathered_info")
    log_           = orch.get("log", [])
    agent_contents = orch.get("agent_contents", {}) or {}

    # ── Output level: C (completeness) + V_G, plus utility (U) ──
    privacy = decision = None
    if args.output_audit:
        log.info("\n  evaluating — privacy (C + V_G)…")
        privacy = judge_privacy(gathered, utility_data, appropriateness_data, args.judge_llm, api_key,
                                cast_context=cast_context)
        log.info("  evaluating — decision (utility)…")
        decision  = judge_decision(response, t_task, utility_data, args.judge_llm, api_key)

    # ── Agent level: V_A (visibility in agent contexts) ──
    audit_results = audit_results_va = {}
    if args.agent_audit and orch_cfg.get("system") != "single":
        log.info("  evaluating — agent audit (V_A)…")
        audit_results = {
            name: judge_agent_knowledge(name, content, visibility_data, utility_data, args.judge_llm, api_key,
                                        lenient=args.violation_leniency, cast_context=cast_context)
            for name, content in agent_contents.items() if content and content.strip()
        }
        if orch_cfg.get("system") == "centralized" and gathered and not orch_cfg.get("private_memory"):
            # Augment executor content with gathered_info only when memory is off; with memory on,
            # the executor's memory write already captures this leakage via V_PMem/V_SMem.
            _ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _exec_aug = (agent_contents.get(executor_name, "") + f"\n\n[{_ts}] agent2agent — Coordinator\n{gathered}").strip()
            _exec_va_audit = judge_agent_knowledge(executor_name, _exec_aug, visibility_data, utility_data,
                                                   args.judge_llm, api_key,
                                                   lenient=args.violation_leniency, cast_context=cast_context)
            audit_results_va = {n: (_exec_va_audit if n == executor_name else r) for n, r in audit_results.items()}
        else:
            audit_results_va = audit_results

    # ── Memory level: V_PMem / V_SMem ──
    memory_violations = None
    if args.memory_audit and orch_cfg.get("system") != "single":
        has_private = orch_cfg.get("private_memory") and orch.get("private_memories_rendered")
        has_shared  = orch_cfg.get("shared_memory")  and orch.get("shared_memory_rendered")
        if has_private or has_shared:
            log.info("  evaluating — memory violations (V_PMem/V_SMem)…")
            priv_mems = {n: r for n, r in orch["private_memories_rendered"].items() if r} if has_private else {}
            memory_violations = judge_memory_violations(
                private_memories=priv_mems,
                shared_memory=orch["shared_memory_rendered"] if has_shared else None,
                visibility_data=visibility_data, utility_data=utility_data,
                model=args.memory_judge_llm, api_key=api_key,
                lenient=args.violation_leniency, cast_context=cast_context,
            )

    # ── Assemble paper-notation metrics ──
    def _gt(attr):
        return utility_data.get("attribute_values", {}).get(attr, {}).get("value")

    def _privacy_attrs(attr_list, revealed_dict):
        return {
            attr: {"ground_truth": _gt(attr),
                   "extracted_value": revealed_dict.get(attr, {}).get("value"),
                   "explanation": revealed_dict.get(attr, {}).get("explanation", "")}
            for attr in attr_list if isinstance(revealed_dict.get(attr, {}), dict)
        }

    def _agent_attrs(attr_list, per_attribute):
        return {
            attr: {"ground_truth": _gt(attr),
                   "extracted_value": per_attribute.get(attr, {}).get("extracted_value"),
                   "explanation": per_attribute.get(attr, {}).get("explanation", "")}
            for attr in attr_list
        }

    n_inapp  = privacy.get("n_inappropriate", 0) if privacy else 0
    n_approp = privacy.get("n_appropriate", 0) if privacy else 0
    _viol = privacy.get("violations", []) if privacy else []   # V_G flagged
    _cov  = privacy.get("covered", [])     if privacy else []
    _miss = privacy.get("missing", [])     if privacy else []

    privacy_section = None
    if privacy:
        privacy_section = {
            "V_G": {
                "flagged": _viol, "n_flagged": len(_viol), "n_inapp": n_inapp,
                "rate": privacy.get("violation_rate"),
                "attributes": _privacy_attrs(_viol, privacy.get("revealed", {})),
            },
            "completeness": {
                "covered": _cov, "missing": _miss, "n_covered": len(_cov), "n_approp": n_approp,
                "rate": privacy.get("completeness"),
                "attributes": _privacy_attrs(_cov + _miss, privacy.get("completeness_revealed", {})),
            },
        }

    # V_A — union of hidden attributes surfacing in agent contexts (augmented audit).
    agent_violation_section = None
    if audit_results_va:
        union_violated = list({attr for res in audit_results_va.values() for attr in res.get("violations", [])})
        _n_any_hidden_va = sum(1 for v in visibility_data.get("attributes", {}).values() if v.get("hidden_from", []))
        agent_violation_section = {
            "total": {
                "flagged": union_violated, "n_flagged": len(union_violated),
                "n_potential": _n_any_hidden_va,
                "rate": len(union_violated) / _n_any_hidden_va if _n_any_hidden_va else 0.0,
            }
        }
        for agent_name, agent_res in audit_results_va.items():
            agent_viols = agent_res.get("violations", [])
            n_hidden    = agent_res.get("n_hidden", 0)
            agent_violation_section[agent_name] = {
                "flagged": agent_viols, "n_flagged": len(agent_viols), "n_hidden": n_hidden,
                "rate": len(agent_viols) / n_hidden if n_hidden else 0.0,
                "attributes": _agent_attrs(agent_viols, agent_res.get("per_attribute", {})),
            }

    # V_A2A — agent-to-agent leakage audited directly from the communication log
    # (robust to memory config; a context-based audit collapses to ~0 under private memory).
    a2a_violation = None
    if log_ and orch_cfg.get("system") != "single":
        log.info("  evaluating — a2a log audit (V_A2A)…")
        a2a_violation = judge_a2a_log(log_, appropriateness_data, utility_data, visibility_data,
                                      args.judge_llm, api_key,
                                      lenient=args.violation_leniency, cast_context=cast_context)

    # V_PMem / V_SMem — hidden attributes surfacing in memory stores.
    memory_section = None
    if memory_violations:
        n_any_hidden_mem = memory_violations.get("n_any_hidden", 0)
        private_raw = memory_violations.get("private", {}) or {}
        shared_raw  = memory_violations.get("shared")

        private_section = None
        if private_raw:
            union_priv = memory_violations.get("private_union_violated", [])
            private_section = {
                "flagged": union_priv, "n_flagged": len(union_priv), "n_hidden": n_any_hidden_mem,
                "rate": memory_violations.get("private_vrate"),
            }
            if args.agent_audit:
                private_section["agents"] = {
                    agent_name: {
                        "flagged": agent_res.get("violations", []),
                        "n_flagged": len(agent_res.get("violations", [])),
                        "n_hidden": agent_res.get("n_hidden", 0),
                        "rate": (len(agent_res.get("violations", [])) / agent_res["n_hidden"]
                                 if agent_res.get("n_hidden") else 0.0),
                        "attributes": _agent_attrs(agent_res.get("violations", []), agent_res.get("per_attribute", {})),
                    }
                    for agent_name, agent_res in private_raw.items()
                }

        shared_section = None
        if shared_raw:
            shared_viols = shared_raw.get("violations", [])
            shared_section = {
                "flagged": shared_viols, "n_flagged": len(shared_viols),
                "n_hidden": shared_raw.get("n_potential", 0), "rate": shared_raw.get("violation_rate"),
                "attributes": _agent_attrs(shared_viols, shared_raw.get("per_attribute", {})),
            }

        memory_section = {k: v for k, v in (("private", private_section), ("shared", shared_section)) if v} or None

    # V_appr = V_G ∪ V_A2A — any-channel appropriateness violation.
    _v_a2a = set(a2a_violation["flagged"]) if a2a_violation else set()
    _vappr_flagged = sorted(set(_viol) | _v_a2a)
    violation_appr = {
        "flagged": _vappr_flagged, "n_flagged": len(_vappr_flagged), "n_inapp": n_inapp,
        "rate": len(_vappr_flagged) / n_inapp if n_inapp else 0.0,
    }

    # V_vis = V_A ∪ V_PMem ∪ V_SMem — any-channel visibility violation.
    _n_any_hidden  = sum(1 for v in visibility_data.get("attributes", {}).values() if v.get("hidden_from", []))
    _va_flagged    = set(agent_violation_section["total"]["flagged"]) if agent_violation_section else set()
    _priv_flagged  = set(memory_section["private"]["flagged"]) if (memory_section and memory_section.get("private")) else set()
    _shared_flagged = set(memory_section["shared"]["flagged"]) if (memory_section and memory_section.get("shared")) else set()
    _vvis_union    = _va_flagged | _priv_flagged | _shared_flagged
    violation_vis = {
        "flagged": sorted(_vvis_union), "n_flagged": len(_vvis_union),
        "n_potential": _n_any_hidden, "rate": len(_vvis_union) / _n_any_hidden if _n_any_hidden else 0.0,
    }

    judgment_data = {
        "config": {
            "scenario": scenario_id, "scenario_dir": base_dir,
            "system": orch_cfg.get("system"), "arch_label": orch_cfg.get("arch_label"),
            "agent_llm": orch_cfg.get("agent_llm"), "write_llm": orch_cfg.get("write_llm"),
            "max_rounds": orch_cfg.get("max_rounds"),
            "private_memory": orch_cfg.get("private_memory"), "shared_memory": orch_cfg.get("shared_memory"),
            "shared_memory_writer": orch_cfg.get("shared_memory_writer"),
            "memory_cleanup": orch_cfg.get("memory_cleanup"), "privacy_level": orch_cfg.get("privacy_level"),
            "judge_llm": args.judge_llm,
            "verifier_llm_1": args.verifier_llm_1, "verifier_llm_2": args.verifier_llm_2,
            "verifier_llm_3": args.verifier_llm_3,
            "violation_leniency": args.violation_leniency,
            "output_audit": args.output_audit, "agent_audit": args.agent_audit, "memory_audit": args.memory_audit,
        },
        "time": datetime.now(timezone.utc).isoformat(),
        "efficiency": {"time": orch.get("pipeline_time"), "rounds": orch.get("n_a2a")},
        # Paper metrics (flat, paper notation). Appropriateness: V_G, V_A2A, V_appr=V_G∪V_A2A.
        # Visibility: V_A, V_PMem, V_SMem, V_vis=V_A∪V_PMem∪V_SMem. Plus C (completeness) and U (utility).
        "utility": {
            "score": 1 if (decision and decision.get("correct")) else 0,
            "correctness": bool(decision.get("correct")) if decision else None,
        },
        "completeness": privacy_section["completeness"] if privacy_section else None,
        "V_G":    privacy_section["V_G"] if privacy_section else None,
        "V_A2A":  a2a_violation,
        "V_appr": violation_appr,
        "V_A":    agent_violation_section,
        "V_PMem": (memory_section or {}).get("private"),
        "V_SMem": (memory_section or {}).get("shared"),
        "V_vis":  violation_vis,
    }

    with open(judgment_path, "w") as f:
        json.dump(judgment_data, f, indent=2, default=str)
    log.info(f"\n  ✓ wrote {judgment_path}")


# ── Batch evaluation (parallel fan-out over pipeline JSONs) ─────────────────────

def _make_folder_name(system, agent_llm, privacy_level, private_memory, shared_memory) -> str:
    model_short = agent_llm.split("/")[-1].lower().replace(".", "_").replace("-", "_")
    if shared_memory and private_memory:
        tag = "memory_both"
    elif shared_memory:
        tag = "memory_shared"
    elif private_memory:
        tag = "memory_private"
    else:
        tag = "memory_no"
    return f"{system}_{model_short}_{privacy_level.lower()}_{tag}_all_nocl"


def _eval_argv(args, api_key) -> list:
    """Single-mode eval flags shared by every batch subprocess (no -d/-o/--force)."""
    argv = ["--judge-llm", args.judge_llm,
            "--verifier-llm-1", args.verifier_llm_1,
            "--verifier-llm-2", args.verifier_llm_2,
            "--verifier-llm-3", args.verifier_llm_3]
    if not args.violation_leniency:
        argv.append("--no-violation-leniency")
    if not args.output_audit:
        argv.append("--no-output-audit")
    if args.agent_audit:
        argv.append("--agent-audit")
    if args.memory_audit:
        argv += ["--memory-audit", "--memory-judge-llm", args.memory_judge_llm]
    if api_key:
        argv += ["--api-key", api_key]
    return argv


def _run_one_eval(args, api_key, scenario_dir: Path, out_dir: Path, pipeline_path: Path) -> bool:
    """Evaluate one pipeline_<name>_runN.json in an isolated subprocess; collect its output."""
    scenario_name = scenario_dir.name
    run_suffix = pipeline_path.stem.rsplit("_run", 1)[-1]
    eval_path  = out_dir / f"evaluation_{scenario_name}_run{run_suffix}.json"
    try:
        with tempfile.TemporaryDirectory(prefix="rbe_tmp_") as tmp:
            tmp_path = Path(tmp)
            tmp_sd = tmp_path / scenario_name
            tmp_sd.mkdir(parents=True)
            shutil.copy2(str(pipeline_path), str(tmp_sd / f"pipeline_{scenario_name}.json"))
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "-d", str(scenario_dir), "-o", str(tmp_path), "--force"] + _eval_argv(args, api_key)
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                log.error(f"  ✗ {scenario_name} run{run_suffix}\n{result.stderr[-400:]}")
                return False
            src = tmp_sd / f"evaluation_{scenario_name}.json"
            if not src.exists():
                log.error(f"  ✗ no evaluation produced for {pipeline_path.name}")
                return False
            shutil.move(str(src), str(eval_path))
            log.info(f"  ✓ {scenario_name} run{run_suffix}")
            return True
    except Exception as e:
        log.error(f"  ✗ {scenario_name} run{run_suffix}: unexpected error: {e}")
        return False


def run_batch(args, api_key):
    """Evaluate every pipeline_*_runN.json in a results folder, in parallel."""
    scenarios_root = Path(args.scenarios_folder).resolve()
    if not scenarios_root.is_dir():
        log.error(f"--scenarios-folder not found: {scenarios_root}")
        sys.exit(1)

    if args.results_path:
        results_root = Path(args.results_path).resolve() / scenarios_root.name
    else:
        folder = _make_folder_name(args.system, args.agent_llm, args.privacy_level,
                                   args.private_memory, args.shared_memory)
        results_root = Path(args.results_base).resolve() / folder / scenarios_root.name
    if not results_root.exists():
        log.error(f"Results folder not found: {results_root}")
        sys.exit(1)

    all_scenario_dirs = sorted(
        p for p in scenarios_root.iterdir() if p.is_dir() and (p / "scenario.json").exists()
    )
    if args.scenarios:
        requested = set(args.scenarios)
        all_scenario_dirs = [p for p in all_scenario_dirs if p.name in requested]
    if not all_scenario_dirs:
        log.error("No matching scenarios found.")
        sys.exit(1)

    run_filter = set(args.run_index.split(",")) if args.run_index else None
    tasks, n_skip = [], 0
    for scenario_dir in all_scenario_dirs:
        scenario_name = scenario_dir.name
        out_dir = results_root / scenario_name
        if not out_dir.is_dir():
            continue
        for pf in sorted(out_dir.glob(f"pipeline_{scenario_name}_run*.json")):
            run_suffix = pf.stem.rsplit("_run", 1)[-1]
            if run_filter is not None and run_suffix not in run_filter:
                continue
            eval_path = out_dir / f"evaluation_{scenario_name}_run{run_suffix}.json"
            if eval_path.exists() and not args.force:
                n_skip += 1
                continue
            tasks.append((scenario_dir, out_dir, pf))

    log.info("═" * W)
    log.info("  ⬡ PiSAs — Batch Evaluation")
    log.info(f"  Scenarios : {scenarios_root}")
    log.info(f"  Results   : {results_root}")
    log.info(f"  Judge     : {args.judge_llm}  |  Verifiers: {args.verifier_llm_1} | {args.verifier_llm_2} | {args.verifier_llm_3}")
    log.info(f"  Output audit: {'on' if args.output_audit else 'off'}  |  "
             f"Agent audit: {'on' if args.agent_audit else 'off'}  |  "
             f"Memory audit: {'on' if args.memory_audit else 'off'}")
    log.info(f"  Tasks     : {len(tasks)} to run, {n_skip} already done  |  Workers: {args.workers}")
    log.info("═" * W)
    if not tasks:
        log.info("  Nothing to do.")
        return

    usage_before = _query_key_usage(api_key)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one_eval, args, api_key, sd, od, pf): (sd, od, pf) for sd, od, pf in tasks}
        results_list = [f.result() for f in as_completed(futures)]

    n_ok   = sum(results_list)
    n_fail = len(results_list) - n_ok
    usage_after = _query_key_usage(api_key)
    log.info(f"\n{'═' * W}")
    log.info(f"  Done.  {n_ok} ok  |  {n_skip} skipped  |  {n_fail} failed")
    if usage_before and usage_after:
        spent = usage_after["usage"] - usage_before["usage"]
        remaining = f"${usage_after['limit'] - usage_after['usage']:.2f}" if usage_after["limit"] is not None else "unlimited"
        log.info(f"  💸 Spent: ${spent:.4f}  |  Remaining: {remaining}")
    log.info("═" * W)
    if n_fail:
        sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="PiSAs evaluation runner — judging only. "
                    "Single mode (-d/-o) or batch mode (--scenarios-folder + --results-path/--results-base).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Judge config (shared).
    p.add_argument("--judge-llm", default="google/gemini-2.5-pro", metavar="MODEL", help="Judge LLM.")
    p.add_argument("--verifier-llm-1", default="anthropic/claude-haiku-4-5", metavar="MODEL")
    p.add_argument("--verifier-llm-2", default="openai/gpt-4o-mini", metavar="MODEL")
    p.add_argument("--verifier-llm-3", default="google/gemini-2.5-flash", metavar="MODEL")
    p.add_argument("--no-violation-leniency", dest="violation_leniency", action="store_false", default=True)
    p.add_argument("--output-audit", action="store_true", default=True, help="Output-level metrics (C, V_G, U). Default on.")
    p.add_argument("--no-output-audit", dest="output_audit", action="store_false")
    p.add_argument("--agent-audit", action="store_true", default=False, help="Per-agent knowledge audit (V_A).")
    p.add_argument("--memory-audit", action="store_true", default=False,
                   help="Memory violation checks (V_PMem/V_SMem; skipped silently if no memory).")
    p.add_argument("--memory-judge-llm", default="google/gemini-2.5-flash", metavar="MODEL",
                   help="Judge LLM for memory checks (cheaper ok — memory contents are shorter).")
    p.add_argument("--api-key", default=None, metavar="KEY", help="OpenRouter API key (overrides OPENROUTER_API_KEY).")
    # Single mode.
    p.add_argument("-d", "--scenario", default=None, metavar="FOLDER",
                   help="[single] Scenario folder under data/, or a direct path.")
    p.add_argument("-o", "--output-dir", default=None, metavar="DIR", help="[single] Folder holding pipeline_<id>.json.")
    p.add_argument("--force", action="store_true", default=False, help="Overwrite existing evaluation_*.json.")
    # Batch mode.
    p.add_argument("--scenarios-folder", default=None, metavar="PATH", help="[batch] Folder of scenario sub-dirs.")
    p.add_argument("--results-path", default=None, metavar="PATH", help="[batch] Results folder with pipeline JSONs.")
    p.add_argument("--results-base", default=None, metavar="PATH",
                   help="[batch] Base dir; folder derived from system/model/privacy/memory flags.")
    p.add_argument("-s", "--system", choices=["decentralized", "single", "centralized"], default="decentralized", help="[batch] Only for --results-base.")
    p.add_argument("--agent-llm", default=None, metavar="MODEL", help="[batch] Only for --results-base.")
    p.add_argument("--privacy-level", choices=["None", "Low", "Medium", "High"], default="High",
                   help="[batch] Only for --results-base.")
    p.add_argument("--private-memory", action="store_true", default=False, help="[batch] Only for --results-base.")
    p.add_argument("--shared-memory", action="store_true", default=False, help="[batch] Only for --results-base.")
    p.add_argument("--workers", type=int, default=8, metavar="N", help="[batch] Parallel workers.")
    p.add_argument("--scenarios", nargs="+", default=None, metavar="NAME", help="[batch] Restrict to these scenario names.")
    p.add_argument("--run-index", default=None, metavar="N[,N,...]", help="[batch] Restrict to these run indices.")
    return p


def main():
    args = build_parser().parse_args()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: No API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    is_batch = bool(args.scenarios_folder or args.results_path or args.results_base)
    if is_batch:
        if not (args.scenarios_folder and (args.results_path or args.results_base)):
            print("ERROR: batch mode needs --scenarios-folder and one of --results-path/--results-base.", file=sys.stderr)
            sys.exit(1)
        if args.results_base and not args.agent_llm:
            print("ERROR: --agent-llm is required with --results-base.", file=sys.stderr)
            sys.exit(1)
        run_batch(args, api_key)
    else:
        if not (args.scenario and args.output_dir):
            print("ERROR: single mode needs both -d/--scenario and -o/--output-dir.", file=sys.stderr)
            sys.exit(1)
        run_single(args, api_key)


if __name__ == "__main__":
    main()
