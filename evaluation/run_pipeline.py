"""
PiSAs — Pipeline runner (orchestration only; runs no judges — use run_evaluation.py).

Two modes, selected by which flags you pass:

  Single scenario  → -d/--scenario + -o/--output-dir
      python run_pipeline.py -d <scenario_dir> -o <output_dir> \\
          -s centralized --agent-llm anthropic/claude-sonnet-4-6 --privacy-level High

  Batch (every scenario in a folder, in parallel) → --scenarios-folder + --results-path
      python run_pipeline.py --scenarios-folder data --results-path results/run \\
          -s centralized --agent-llm anthropic/claude-sonnet-4-6 --privacy-level High --run-index 0,1,2

Batch mode fans out one isolated subprocess (this same script, single mode) per
(scenario, run_index) and collects pipeline_<scenario>_run<N>.json into the results folder.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

log = logging.getLogger("pipeline")
log.setLevel(logging.DEBUG)
_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(_sh)

_DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
ARCH_LABELS = {"decentralized": "Decentralized", "single": "Single", "centralized": "Centralized"}
W = 72


def _resolve_scenario(scenario_arg):
    """Accept a scenario folder path or a name under data/; return (id, abs_dir)."""
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
    """Best-effort OpenRouter usage snapshot for batch cost reporting."""
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


# ── Single-scenario orchestration ─────────────────────────────────────────────

def run_single(args, api_key):
    """Orchestrate one scenario and write pipeline_<id>.json under <output_dir>/<id>/."""
    scenario_id, base_dir = _resolve_scenario(args.scenario)
    if scenario_id is None:
        print(f"ERROR: Scenario {args.scenario!r} not found.", file=sys.stderr)
        sys.exit(1)

    output_dir       = Path(args.output_dir).resolve()
    scenario_dir_out = output_dir / scenario_id
    scenario_dir_out.mkdir(parents=True, exist_ok=True)
    run_path         = scenario_dir_out / f"pipeline_{scenario_id}.json"

    _fh = logging.FileHandler(scenario_dir_out / f"run_{scenario_id}.log", encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(_fh)

    _paths = _scenario_paths(base_dir)
    shutil.copy2(_paths["scenario"], scenario_dir_out / "scenario.json")
    visibility_data      = json.load(open(_paths["visibility"]))
    appropriateness_data = json.load(open(_paths["appropriateness"]))
    scenario             = json.load(open(_paths["scenario"]))
    utility_data         = json.load(open(_paths["utility"]))

    t_task        = scenario["task"]
    executor_role = t_task["executor_role"]
    executor_name = t_task["participants"][executor_role]["name"]

    use_team   = args.system == "single"
    use_tc     = args.system == "centralized"
    arch_label = ARCH_LABELS[args.system]

    # Imported here (not at module top) so batch mode's parent process stays light.
    from agents import (build_siloed_system, run_task_token_passing,
                        run_task_truly_centralized, build_team_system, load_artifacts_team,
                        preload_artifacts_with_memory, PRIVACY_INSTRUCTIONS,
                        MEMORY_VISIBILITY_INSTRUCTIONS)

    log.info("═" * W)
    log.info("  ⬡ PiSAs  —  Pipeline")
    log.info(f"  Scenario    : {scenario_id}")
    log.info(f"  Output dir  : {output_dir}")
    log.info(f"  System      : {arch_label}")
    log.info(f"  Agent LLM   : {args.agent_llm}")
    if not use_team:
        log.info(f"  Write LLM   : {args.write_llm}")
        log.info(f"  Max rounds  : {args.max_rounds}")
        log.info(f"  Priv memory : {'on' if args.private_memory else 'off'}")
        log.info(f"  Shared mem  : {'on' if args.shared_memory else 'off'}")
    log.info(f"  Privacy lvl : {args.privacy_level}")
    log.info("═" * W)

    if run_path.exists() and not args.force:
        log.info(f"\n  ↺ {run_path.name} exists — skipping (use --force to overwrite).")
        return

    t0 = time.time()
    prompt_gather = prompt_decision = None

    if use_team:
        log.info("\n  loading artifacts…")
        team_agent = build_team_system(args.agent_llm, api_key)
        load_artifacts_team(team_agent, scenario["timeline"])
        agents_map       = {executor_name: team_agent}
        private_memories = {}
        shared_memory    = None

        log.info("  executing task (Single)…")
        result          = team_agent.run_task(scenario, PRIVACY_INSTRUCTIONS[args.privacy_level])
        gathered        = result["gathered_info"]
        response        = result["response"]
        prompt_gather   = result["prompt_gather"]
        prompt_decision = result["prompt_decision"]
        exec_ctx        = team_agent.get_full_content()
        n_a2a           = 0
    else:
        log.info("\n  preloading artifacts…")
        agents_map = build_siloed_system(scenario["cast"], args.agent_llm, api_key)
        # TC only preloads shared memory when writes are open to all agents.
        _preload_shared = (args.shared_memory and args.shared_memory_writer == "all") if use_tc else args.shared_memory
        private_memories, shared_memory, _ = preload_artifacts_with_memory(
            agents_map, scenario["timeline"], scenario,
            use_private_memory=args.private_memory,
            use_shared_memory=_preload_shared,
            up_to=len(scenario["timeline"]),
            shared_memory_writer=args.shared_memory_writer,
            use_memory_cleanup=args.memory_cleanup,
            write_llm=args.write_llm,
            memory_visibility_instruction=MEMORY_VISIBILITY_INSTRUCTIONS[args.privacy_level],
        )

        log.info(f"  executing task ({arch_label})…")
        run_fn = run_task_truly_centralized if use_tc else run_task_token_passing
        tc_kwargs = dict(
            coordinator_model_name=args.coordinator_llm,
            coordinator_thinking=args.coordinator_thinking,
        ) if use_tc else {}
        result = run_fn(
            agents_map, executor_name, scenario,
            max_rounds=args.max_rounds,
            use_private_memory=args.private_memory,
            private_memories=private_memories,
            shared_memory=shared_memory if args.shared_memory else None,
            privacy_instruction=PRIVACY_INSTRUCTIONS[args.privacy_level],
            memory_visibility_instruction=MEMORY_VISIBILITY_INSTRUCTIONS[args.privacy_level],
            shared_memory_writer=args.shared_memory_writer,
            use_memory_cleanup=args.memory_cleanup,
            write_llm=args.write_llm,
            **tc_kwargs,
        )
        n_a2a    = len([e for e in result["log"] if e["type"] in ("sent", "received")])
        response = result["final_response"]
        exec_ctx = result.get("gathering_context_at_decision")
        gathered = result.get("gathered_info")

    pipeline_time = time.time() - t0

    # Ground truth carried forward for the evaluator.
    _app_labels = appropriateness_data.get("attributes", {})
    _vis_attrs  = visibility_data.get("attributes", {})
    _attr_vals  = utility_data.get("attribute_values", {})
    _all_attrs  = sorted(set(_app_labels) | set(_attr_vals))
    _gt_attrs = {
        attr: {
            "value":           _attr_vals.get(attr, {}).get("value"),
            "about_attribute": _attr_vals.get(attr, {}).get("about_attribute"),
            "about_value":     _attr_vals.get(attr, {}).get("about_value"),
            "appropriateness": _app_labels.get(attr),
            "visible_to":      _vis_attrs.get(attr, {}).get("visible_to", []),
            "hidden_from":     _vis_attrs.get(attr, {}).get("hidden_from", []),
        }
        for attr in _all_attrs
    }
    _n_inapp = sum(1 for v in _app_labels.values() if v == "inappropriate")
    _n_app   = sum(1 for v in _app_labels.values() if v == "appropriate")

    pipeline_data = {
        "config": {
            "scenario":             scenario_id,
            "scenario_dir":         base_dir,
            "system":               args.system,
            "arch_label":           arch_label,
            "agent_llm":            args.agent_llm,
            "write_llm":            args.write_llm,
            "max_rounds":           args.max_rounds,
            "private_memory":       args.private_memory,
            "shared_memory":        args.shared_memory,
            "shared_memory_writer": args.shared_memory_writer,
            "memory_cleanup":       args.memory_cleanup,
            "privacy_level":        args.privacy_level,
        },
        "task_description":              scenario["task"]["description"],
        "ground_truth": {
            "n_attributes":    len(_all_attrs),
            "n_appropriate":   _n_app,
            "n_inappropriate": _n_inapp,
            "attributes":      _gt_attrs,
        },
        "ran_at_utc":                    datetime.now(timezone.utc).isoformat(),
        "final_response":                response,
        "gathered_info":                 gathered,
        "prompt_gather":                 prompt_gather,
        "prompt_decision":               prompt_decision,
        "gathering_context_at_decision": exec_ctx,
        "log":                           result.get("log", []),
        "agent_contents":                {n: a.get_full_content() for n, a in agents_map.items()},
        "coordinator_content":           result.get("coordinator_content"),
        "coordinator_thinking_trace":    result.get("coordinator_thinking_trace"),
        "private_memories_rendered":     {n: m.render() for n, m in private_memories.items()},
        "shared_memory_rendered":        shared_memory.render() if shared_memory else None,
        "n_a2a":                         n_a2a,
        "pipeline_time":                 pipeline_time,
    }

    with open(run_path, "w") as f:
        json.dump(pipeline_data, f, indent=2, default=str)
    log.info(f"  ✓ wrote {run_path}")
    log.info(f"  Pipeline time: {pipeline_time:.1f}s  |  n_a2a: {n_a2a}")


# ── Batch orchestration (parallel fan-out over scenarios) ──────────────────────

def _pipeline_argv(args, api_key) -> list:
    """Single-mode flags shared by every batch subprocess (no -d/-o/--force)."""
    argv = ["-s", args.system, "--agent-llm", args.agent_llm,
            "--write-llm", args.write_llm, "--max-rounds", str(args.max_rounds)]
    if args.coordinator_llm:
        argv += ["--coordinator-llm", args.coordinator_llm]
    if args.coordinator_thinking:
        argv.append("--coordinator-thinking")
    if args.private_memory:
        argv.append("--private-memory")
    if args.shared_memory:
        argv.append("--shared-memory")
    argv += ["--shared-memory-writer", args.shared_memory_writer]
    if args.memory_cleanup:
        argv.append("--memory-cleanup")
    argv += ["--privacy-level", args.privacy_level]
    if api_key:
        argv += ["--api-key", api_key]
    return argv


def _next_run_index(out_dir: Path, scenario_name: str) -> int:
    indices = []
    for p in out_dir.glob(f"pipeline_{scenario_name}_run*.json"):
        try:
            indices.append(int(p.stem.rsplit("_run", 1)[-1]))
        except (ValueError, IndexError):
            pass
    return max(indices) + 1 if indices else 0


def _run_one_batch(args, api_key, scenario_dir: Path, out_dir: Path, run_index: int) -> bool:
    """Run one (scenario, run_index) in an isolated subprocess; collect its outputs."""
    scenario_name = scenario_dir.name
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="rbp_tmp_") as tmp:
            tmp_path = Path(tmp)
            cmd = [sys.executable, str(Path(__file__).resolve()),
                   "-d", str(scenario_dir), "-o", str(tmp_path), "--force"] + _pipeline_argv(args, api_key)
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                log.error(f"  ✗ {scenario_name} run{run_index}\n{result.stderr[-400:]}")
                return False

            tmp_sd = tmp_path / scenario_name
            if not tmp_sd.is_dir():
                log.error(f"  ✗ output dir not found: {tmp_sd}")
                return False

            suffix = f"_run{run_index}"
            for stem, new_stem in [
                (f"pipeline_{scenario_name}", f"pipeline_{scenario_name}{suffix}"),
                (f"run_{scenario_name}",      f"run_{scenario_name}{suffix}"),
            ]:
                for ext in (".json", ".log"):
                    src = tmp_sd / (stem + ext)
                    if src.exists():
                        shutil.move(str(src), str(out_dir / (new_stem + ext)))
            log.info(f"  ✓ {scenario_name} run{run_index}")
            return True
    except Exception as e:
        log.error(f"  ✗ {scenario_name} run{run_index}: unexpected error: {e}")
        return False


def run_batch(args, api_key):
    """Find every scenario in --scenarios-folder and run each (optionally K runs) in parallel."""
    scenarios_root = Path(args.scenarios_folder).resolve()
    if not scenarios_root.is_dir():
        log.error(f"--scenarios-folder not found: {scenarios_root}")
        sys.exit(1)
    results_root = Path(args.results_path).resolve() / scenarios_root.name
    results_root.mkdir(parents=True, exist_ok=True)

    all_scenario_dirs = sorted(
        p for p in scenarios_root.iterdir()
        if p.is_dir() and (p / "scenario.json").exists()
    )
    if args.scenarios:
        requested = set(args.scenarios)
        all_scenario_dirs = [p for p in all_scenario_dirs if p.name in requested]
    if not all_scenario_dirs:
        log.error("No matching scenarios found.")
        sys.exit(1)

    requested_indices = [int(x.strip()) for x in args.run_index.split(",")] if args.run_index else None

    work, n_skip = [], 0
    for scenario_dir in all_scenario_dirs:
        scenario_name = scenario_dir.name
        out_dir = results_root / scenario_name
        if args.delete and out_dir.is_dir():
            shutil.rmtree(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and out_dir.is_dir() and list(out_dir.glob(f"pipeline_{scenario_name}_run*.json")):
            n_skip += 1
            continue
        if requested_indices is not None:
            work += [(scenario_dir, out_dir, ri) for ri in requested_indices]
        else:
            ri = _next_run_index(out_dir, scenario_name) if out_dir.is_dir() else 0
            work.append((scenario_dir, out_dir, ri))

    log.info("═" * W)
    log.info("  ⬡ PiSAs — Batch Pipeline")
    log.info(f"  Scenarios : {scenarios_root}  ({len(all_scenario_dirs)} found, {n_skip} skipped)")
    log.info(f"  Results   : {results_root}")
    log.info(f"  System    : {args.system}  |  Model: {args.agent_llm}")
    log.info(f"  Privacy   : {args.privacy_level}  |  Memory: {'shared ' if args.shared_memory else ''}{'private' if args.private_memory else ''}")
    log.info(f"  Tasks     : {len(work)}  |  Workers: {args.workers}")
    log.info("═" * W)

    usage_before = _query_key_usage(api_key)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one_batch, args, api_key, sd, od, ri): (sd, od, ri) for sd, od, ri in work}
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
        description="PiSAs pipeline runner — orchestration only, no judging. "
                    "Single mode (-d/-o) or batch mode (--scenarios-folder/--results-path).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Architecture + model config (shared).
    p.add_argument("-s", "--system", choices=["decentralized", "single", "centralized"], default="decentralized", help="Agent architecture.")
    p.add_argument("--agent-llm", required=True, metavar="MODEL", help="Agent backbone LLM.")
    p.add_argument("--coordinator-llm", default=None, metavar="MODEL",
                   help="LLM for the TC coordinator; falls back to --agent-llm if unset.")
    p.add_argument("--coordinator-thinking", action="store_true", default=False,
                   help="Enable extended thinking for the TC coordinator (min budget 1024 tokens).")
    p.add_argument("--write-llm", default="google/gemini-2.5-flash-lite", metavar="MODEL",
                   help="LLM for memory write/cleanup calls.")
    p.add_argument("--max-rounds", type=int, default=30, metavar="R")
    p.add_argument("--private-memory", action="store_true", default=False)
    p.add_argument("--shared-memory", action="store_true", default=False)
    p.add_argument("--shared-memory-writer", choices=["all", "executor"], default="all")
    p.add_argument("--memory-cleanup", action="store_true", default=False)
    p.add_argument("--privacy-level", choices=["None", "Low", "Medium", "High"], default="High")
    p.add_argument("--api-key", default=None, metavar="KEY",
                   help="OpenRouter API key (overrides OPENROUTER_API_KEY).")
    # Single mode.
    p.add_argument("-d", "--scenario", default=None, metavar="FOLDER",
                   help="[single] Scenario folder under data/, or a direct path.")
    p.add_argument("-o", "--output-dir", default=None, metavar="DIR", help="[single] Output directory.")
    p.add_argument("--force", action="store_true", default=False, help="[single] Overwrite existing pipeline_*.json.")
    # Batch mode.
    p.add_argument("--scenarios-folder", default=None, metavar="PATH", help="[batch] Folder of scenario sub-dirs.")
    p.add_argument("--results-path", default=None, metavar="PATH", help="[batch] Output root for results.")
    p.add_argument("--run-index", default=None, metavar="N[,N,...]",
                   help="[batch] Comma-separated run indices (e.g. 0,1,2). Omit to auto-increment.")
    p.add_argument("--workers", type=int, default=32, metavar="N", help="[batch] Parallel workers.")
    p.add_argument("--delete", action="store_true", default=False, help="[batch] Wipe each scenario's output dir first.")
    p.add_argument("--skip-existing", action="store_true", default=False, help="[batch] Skip scenarios already run.")
    p.add_argument("--scenarios", nargs="+", default=None, metavar="NAME", help="[batch] Restrict to these scenario names.")
    return p


def main():
    args = build_parser().parse_args()
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: No API key. Set OPENROUTER_API_KEY or pass --api-key.", file=sys.stderr)
        sys.exit(1)

    is_batch = bool(args.scenarios_folder or args.results_path)
    if is_batch:
        if not (args.scenarios_folder and args.results_path):
            print("ERROR: batch mode needs both --scenarios-folder and --results-path.", file=sys.stderr)
            sys.exit(1)
        run_batch(args, api_key)
    else:
        if not (args.scenario and args.output_dir):
            print("ERROR: single mode needs both -d/--scenario and -o/--output-dir.", file=sys.stderr)
            sys.exit(1)
        run_single(args, api_key)


if __name__ == "__main__":
    main()
