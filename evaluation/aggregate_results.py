#!/usr/bin/env python3
"""
PiSAs — Results aggregator.

Reads the per-scenario, per-run evaluation_*.json files produced by run_evaluation.py
in one config folder and reports the paper's metrics as **mean ± standard error over
scenarios**, which is what the paper's tables print.

Aggregation convention (App. D.4 of the paper), applied per scenario first, then
averaged over scenarios:

  U, C      run means within a scenario.
  V_*       an attribute counts as violated if it leaked in ANY of the K runs
            (``--mode any_k``, the default and the paper's main results).
            ``--mode worst`` takes the worst single run, ``--mode mean`` their mean.
  V_S       share of runs in which at least one inappropriate attribute leaked
            anywhere (gathering, agent-to-agent, or the final answer).
  #A2A      mean messages per run.

Metrics:
  C      completeness — appropriate attributes present in the gathered summary
  U      utility — decision matches the oracle
  V_G    inappropriate attributes in the gathered summary
  V_A2A  inappropriate attributes in agent-to-agent messages
  V_out  inappropriate attributes in the final answer
  V_appr V_G ∪ V_A2A  (the paper writes this one V_C)
  V_A    hidden attributes surfacing in an agent's context
  V_PMem / V_SMem  hidden attributes in private / shared memory
  V_vis  V_A ∪ V_PMem ∪ V_SMem
  V_any  everything that leaked, over the union universe (inappropriate ∪ hidden)
  F      share of runs with at least one appropriateness violation (V_appr ≠ ∅)

A metric that nothing measured in this config prints "—" rather than 0.0 %: the
single-agent system has no agent-to-agent channel and no per-agent audit, so V_A2A,
V_A and V_vis are genuinely undefined there, not zero.

Usage:
    python aggregate_results.py --results-path results/PiSAs/uas_flight_readiness
    python aggregate_results.py --results-path <folder> --mode all --json metrics.json
    python aggregate_results.py --results-path <folder> --per-scenario rows.csv
    python aggregate_results.py --results-path <a> --results-path <b>     # pooled
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

# metric -> (json path to its sub-dict, key holding the denominator)
# Appropriateness violations use n_inapp; visibility violations the hidden-from count;
# V_any the union universe of both.
VIOLATION_METRICS = {
    "V_G":    (["V_G"],          "n_inapp"),
    "V_A2A":  (["V_A2A"],        "n_inapp"),
    "V_out":  (["V_out"],        "n_inapp"),
    "V_appr": (["V_appr"],       "n_inapp"),
    "V_A":    (["V_A", "total"], "n_potential"),
    "V_PMem": (["V_PMem"],       "n_hidden"),
    "V_SMem": (["V_SMem"],       "n_hidden"),
    "V_vis":  (["V_vis"],        "n_potential"),
    "V_any":  (["V_any"],        "n_universe"),
}
# Display order for the report.
ROW_ORDER = ["C", "U", "V_G", "V_A2A", "V_out", "V_appr", "V_A", "V_PMem", "V_SMem",
             "V_vis", "V_any", "V_S", "F", "#A2A"]

# A gathered summary longer than this is almost always a model dumping its raw reasoning
# into the gather channel rather than a list of facts.
GATHER_BLOAT_CHARS = 20000


def _dig(d, path):
    """Follow a list of keys; return None if any level is missing/None."""
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _scenario_run(fname: str):
    """evaluation_<scenario>_run<k>.json -> (scenario, run_suffix)."""
    stem = fname[len("evaluation_"):-len(".json")]
    scen, _, run = stem.rpartition("_run")
    return scen, run


def load_runs(results_paths, exclude_degraded=False):
    """Return ({scenario: [run_dict, ...]}, health) for every evaluation_*_run*.json found.

    Each run_dict is the evaluation JSON with the matching pipeline JSON attached under
    "_pipeline" when it exists (for #A2A and run-health checks)."""
    runs = defaultdict(list)
    health = {"runs": 0, "degraded": 0, "no_a2a": 0, "empty_gather": 0,
              "bloated_gather": 0, "verifier_degraded": 0, "excluded": 0}
    for results_path in results_paths:
        for pf in sorted(Path(results_path).rglob("evaluation_*_run*.json")):
            scen, run = _scenario_run(pf.name)
            try:
                ev = json.load(open(pf))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  ! skipping {pf.name}: {e}", file=sys.stderr)
                continue
            pipe_path = pf.with_name(f"pipeline_{scen}_run{run}.json")
            pipe = {}
            if pipe_path.exists():
                try:
                    pipe = json.load(open(pipe_path))
                except (json.JSONDecodeError, OSError):
                    pipe = {}
            ev["_pipeline"] = pipe
            ev["_run"] = run

            gathered = pipe.get("gathered_info") or ""
            degraded = bool(pipe.get("degraded"))
            verif_degraded = bool(_dig(ev, ["verification", "degraded"]))
            health["runs"] += 1
            health["degraded"] += int(degraded)
            health["verifier_degraded"] += int(verif_degraded)
            # The single-agent system has no agent-to-agent channel, so zero messages is
            # its normal state, not a symptom.
            system = (pipe.get("config", {}) or {}).get("system") or _dig(ev, ["config", "system"])
            if system != "single":
                health["no_a2a"] += int(pipe.get("n_a2a") == 0)
            health["empty_gather"] += int(pipe != {} and not gathered.strip())
            health["bloated_gather"] += int(len(gathered) > GATHER_BLOAT_CHARS)

            if exclude_degraded and (degraded or verif_degraded or
                                     (pipe != {} and not gathered.strip())):
                health["excluded"] += 1
                continue
            runs[scen].append(ev)
    return runs, health


def _per_run_rate(sub, denom_key):
    """(flagged set, denominator, rate) for one run's metric sub-dict, or Nones if absent."""
    if not isinstance(sub, dict):
        return None, None, None
    flagged = set(sub.get("flagged", []) or [])
    denom = sub.get(denom_key) or 0
    rate = (len(flagged) / denom) if denom else 0.0
    return flagged, denom, rate


def per_scenario_violation(runs, path, denom_key, mode):
    """One scenario's rate for one violation metric, or None when nothing measured it."""
    subs = [_dig(r, path) for r in runs]
    present = [_per_run_rate(s, denom_key) for s in subs if isinstance(s, dict)]
    if not present:
        return None
    if mode == "any_k":
        union = set().union(*(f for f, _, _ in present))
        denom = next((d for _, d, _ in present if d), 0)
        return len(union) / denom if denom else 0.0
    if mode == "worst":
        return max(r for _, _, r in present)
    return sum(r for _, _, r in present) / len(present)


def per_scenario_scalars(runs):
    """C, U, V_S, F, #A2A for one scenario (run means / run shares)."""
    def _mean(xs):
        return st.mean(xs) if xs else None
    comps = [c for c in (_dig(r, ["completeness", "rate"]) for r in runs) if c is not None]
    utils = [u for u in (_dig(r, ["utility", "score"]) for r in runs) if u is not None]
    a2as = [n for n in (_dig(r, ["efficiency", "rounds"]) for r in runs) if n is not None]

    fails, leaks = [], []
    for r in runs:
        vappr = _dig(r, ["V_appr", "flagged"])
        if vappr is not None:
            fails.append(1 if len(vappr) > 0 else 0)
        # V_S: did anything inappropriate leak on any surface in this run?
        surfaces = [_dig(r, [m, "flagged"]) for m in ("V_G", "V_A2A", "V_out")]
        if any(s is not None for s in surfaces):
            leaks.append(1 if any(s for s in surfaces if s) else 0)
    return {"C": _mean(comps), "U": _mean(utils), "F": _mean(fails),
            "V_S": _mean(leaks), "#A2A": _mean(a2as)}


def compute(runs_by_scenario, mode):
    """Per-scenario rows + mean/SE over scenarios for every metric."""
    rows = []
    for scen in sorted(runs_by_scenario):
        runs = runs_by_scenario[scen]
        row = {"scenario": scen, "n_runs": len(runs)}
        row.update(per_scenario_scalars(runs))
        for name, (path, denom_key) in VIOLATION_METRICS.items():
            row[name] = per_scenario_violation(runs, path, denom_key, mode)
        rows.append(row)

    summary = {}
    for name in ROW_ORDER:
        xs = [r[name] for r in rows if r.get(name) is not None]
        if not xs:
            summary[name] = {"mean": None, "se": None, "n": 0}
            continue
        se = st.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0
        summary[name] = {"mean": st.mean(xs), "se": se, "n": len(xs)}
    return summary, rows


def _fmt(name, cell):
    if cell is None or cell.get("mean") is None:
        return "—"
    m, se = cell["mean"], cell["se"]
    if name == "#A2A":
        return f"{m:.1f}±{se:.1f}"
    return f"{m * 100:.1f}±{se * 100:.1f}"


def _write_per_scenario(path, rows_by_mode, modes):
    """Dump the per-scenario rows (first mode) as CSV or JSON, by extension."""
    rows = rows_by_mode[modes[0]]
    if str(path).lower().endswith(".json"):
        with open(path, "w") as f:
            json.dump({m: rows_by_mode[m] for m in modes}, f, indent=2)
        return
    fields = ["scenario", "n_runs"] + ROW_ORDER
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def main():
    ap = argparse.ArgumentParser(description="Aggregate PiSAs evaluation results into paper metrics.")
    ap.add_argument("--results-path", required=True, metavar="PATH", action="append",
                    help="Config folder containing evaluation_*_run*.json (searched recursively). "
                         "Repeat the flag to pool several folders into one table.")
    ap.add_argument("--mode", choices=["any_k", "worst", "mean", "all"], default="any_k",
                    help="Violation aggregation across runs (default: any_k = paper main results).")
    ap.add_argument("--exclude-degraded", action="store_true", default=False,
                    help="Drop runs whose orchestration produced an empty turn or empty gathered "
                         "summary, or whose verifier committee could not vote.")
    ap.add_argument("--per-scenario", default=None, metavar="FILE",
                    help="Write the per-scenario rows to FILE (.csv or .json).")
    ap.add_argument("--json", default=None, metavar="FILE",
                    help="Also write the computed metrics to this JSON file.")
    args = ap.parse_args()

    results_paths = []
    for p in args.results_path:
        rp = Path(p).resolve()
        if not rp.is_dir():
            print(f"ERROR: not a directory: {rp}", file=sys.stderr)
            sys.exit(1)
        results_paths.append(rp)

    runs_by_scenario, health = load_runs(results_paths, exclude_degraded=args.exclude_degraded)
    if not runs_by_scenario:
        print(f"ERROR: no evaluation_*_run*.json under {', '.join(str(p) for p in results_paths)}",
              file=sys.stderr)
        sys.exit(1)

    n_scen = len(runs_by_scenario)
    n_runs = sum(len(v) for v in runs_by_scenario.values())
    modes = ["any_k", "worst", "mean"] if args.mode == "all" else [args.mode]
    summaries, rows_by_mode = {}, {}
    for m in modes:
        summaries[m], rows_by_mode[m] = compute(runs_by_scenario, m)

    label = results_paths[0].name if len(results_paths) == 1 else f"{len(results_paths)} folders pooled"
    print("═" * 64)
    print(f"  {label}")
    print(f"  scenarios={n_scen}  eval files={n_runs}  mode={args.mode}   (mean ± SE over scenarios)")
    print("═" * 64)
    if len(modes) > 1:
        print("  Metric    " + "".join(f"{m:>16}" for m in modes))
    else:
        print("  Metric              Rate")
    for name in ROW_ORDER:
        cells = "".join(f"{_fmt(name, summaries[m][name]):>16}" for m in modes)
        print(f"  {name:<8}{cells}")
    print("═" * 64)

    # Run health — a degraded run reads as a clean, cautious system unless you look.
    print("  Run health")
    print(f"    runs inspected        : {health['runs']}")
    print(f"    empty agent turn      : {health['degraded']}")
    print(f"    empty gathered summary: {health['empty_gather']}")
    print(f"    no a2a messages       : {health['no_a2a']}   (multi-agent runs only)")
    print(f"    gathered summary >{GATHER_BLOAT_CHARS // 1000}k chars: {health['bloated_gather']}")
    print(f"    verifier could not vote: {health['verifier_degraded']}")
    if args.exclude_degraded:
        print(f"    excluded from the table: {health['excluded']}")
    elif health["degraded"] or health["empty_gather"]:
        print("    → re-run with --exclude-degraded to drop these")
    print("═" * 64)

    if args.per_scenario:
        _write_per_scenario(args.per_scenario, rows_by_mode, modes)
        print(f"  wrote {args.per_scenario}")

    if args.json:
        payload = {"results_paths": [str(p) for p in results_paths],
                   "n_scenarios": n_scen, "n_eval_files": n_runs,
                   "mode": args.mode, "health": health,
                   "metrics": summaries, "per_scenario": rows_by_mode}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
