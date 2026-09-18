#!/usr/bin/env python3
"""
PiSAs — one command: orchestrate, judge, aggregate.

The three steps are separate scripts because they are separately re-runnable (re-judge
without re-running, re-aggregate without re-judging). This wrapper is for when you just
want a number:

    # a released task, all three systems, three runs each
    python run_benchmark.py --task uas_flight_readiness --agent-llm anthropic/claude-sonnet-4-6

    # your own scenarios, one system
    python run_benchmark.py --scenarios-folder my_task -s centralized \\
        --agent-llm openai/gpt-5.5 --run-index 0

It calls run_pipeline.py, then run_evaluation.py, then aggregate_results.py for each
system, and prints one table per system at the end. Anything it does not expose is
available by running the three scripts yourself; the flags here are the ones a headline
number needs.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchmark import PISAS_HF_REPO, resolve_task

SYSTEMS = ["single", "centralized", "decentralized"]


def _run(argv, label):
    print(f"\n  → {label}\n    {' '.join(argv[1:])}\n", flush=True)
    rc = subprocess.run(argv, check=False).returncode
    if rc != 0:
        print(f"ERROR: {label} failed (exit {rc})", file=sys.stderr)
    return rc


def main():
    p = argparse.ArgumentParser(
        description="Run, judge and aggregate PiSAs in one command.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("what to run on")
    src.add_argument("--task", default=None, metavar="NAME",
                     help="A task of the Hugging Face benchmark (see run_pipeline.py --list-tasks).")
    src.add_argument("--scenarios-folder", default=None, metavar="PATH",
                     help="Your own folder of scenario bundles, instead of --task.")
    src.add_argument("--hf-repo", default=PISAS_HF_REPO, metavar="REPO_ID")
    src.add_argument("--hf-revision", default=None, metavar="REV")

    p.add_argument("-s", "--systems", default="single,centralized,decentralized", metavar="LIST",
                   help="Comma-separated subset of single,centralized,decentralized.")
    p.add_argument("--agent-llm", required=True, metavar="MODEL")
    p.add_argument("--judge-llm", default="google/gemini-2.5-pro", metavar="MODEL")
    p.add_argument("--judge-mode", choices=["chain", "fact"], default="chain")
    p.add_argument("--privacy-level", choices=["None", "Low", "Medium", "High"], default="High")
    p.add_argument("--run-index", default="0,1,2", metavar="N[,N,...]")
    p.add_argument("--results-path", default="results/PiSAs", metavar="PATH")
    p.add_argument("--workers", type=int, default=8, metavar="N")
    p.add_argument("--hide-task-from-peers", action="store_true", default=False)
    p.add_argument("--no-agent-audit", dest="agent_audit", action="store_false", default=True,
                   help="Skip the per-agent knowledge audit (V_A, V_vis).")
    p.add_argument("--api-key", default=None, metavar="KEY")
    p.add_argument("--skip-existing", action="store_true", default=False)
    args = p.parse_args()

    if not (args.task or args.scenarios_folder):
        p.error("one of --task or --scenarios-folder is required")

    scenarios = (args.scenarios_folder or
                 str(resolve_task(args.task, repo_id=args.hf_repo, revision=args.hf_revision)))
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        p.error(f"unknown system(s): {unknown}; choose from {SYSTEMS}")

    py = sys.executable
    key = ["--api-key", args.api_key] if args.api_key else []
    if not (args.api_key or os.getenv("OPENROUTER_API_KEY")):
        p.error("no API key: set OPENROUTER_API_KEY or pass --api-key")

    failures = []
    for system in systems:
        out = str(Path(args.results_path) / system)

        pipeline = [py, str(HERE / "run_pipeline.py"), "-s", system,
                    "--agent-llm", args.agent_llm, "--scenarios-folder", scenarios,
                    "--results-path", out, "--privacy-level", args.privacy_level,
                    "--run-index", args.run_index, "--workers", str(args.workers)] + key
        if args.hide_task_from_peers:
            pipeline.append("--hide-task-from-peers")
        if args.skip_existing:
            pipeline.append("--skip-existing")
        if _run(pipeline, f"{system}: orchestrate"):
            failures.append(system)
            continue

        evaluate = [py, str(HERE / "run_evaluation.py"), "--scenarios-folder", scenarios,
                    "--results-path", out, "--judge-llm", args.judge_llm,
                    "--judge-mode", args.judge_mode, "--workers", str(args.workers)] + key
        if args.agent_audit:
            evaluate.append("--agent-audit")
        if _run(evaluate, f"{system}: judge"):
            failures.append(system)
            continue

        _run([py, str(HERE / "aggregate_results.py"), "--results-path", out],
             f"{system}: aggregate")

    if failures:
        print(f"\n  failed: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
