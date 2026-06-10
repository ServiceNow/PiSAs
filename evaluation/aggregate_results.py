#!/usr/bin/env python3
"""
PiSAs — Results aggregator.

Reads the per-scenario, per-run evaluation_*.json files produced by run_evaluation.py
in one config folder and reports the paper's metrics, aggregated across the K runs and
averaged across scenarios.

Three aggregation modes for violation rates (App. D.4 of the paper):
  any_k  (default, paper main results) — an attribute counts as violated if it leaks in
                                         ANY of the K runs (union across runs per scenario).
  worst                                — the worst single-run rate per scenario.
  mean                                 — the mean single-run rate per scenario.

Completeness (C), Utility (U), failure rate (F) and #A2A are mode-independent:
C/U/#A2A are averaged over all scenario-run pairs; F is the fraction of scenario-run
pairs with at least one appropriateness violation (V_appr non-empty).

Usage:
    python aggregate_results.py --results-path results/paper/jira/tc_..._memory_no
    python aggregate_results.py --results-path <folder> --mode all --json metrics.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# metric -> (json path to its sub-dict, key holding the denominator)
# Appropriateness violations use n_inapp; visibility violations use the hidden-from count.
VIOLATION_METRICS = {
    "V_G":    (["V_G"],          "n_inapp"),
    "V_A2A":  (["V_A2A"],        "n_inapp"),
    "V_appr": (["V_appr"],       "n_inapp"),
    "V_A":    (["V_A", "total"], "n_potential"),
    "V_PMem": (["V_PMem"],       "n_hidden"),
    "V_SMem": (["V_SMem"],       "n_hidden"),
    "V_vis":  (["V_vis"],        "n_potential"),
}
# Display order for the report.
ROW_ORDER = ["C", "U", "V_G", "V_A2A", "V_appr", "V_A", "V_PMem", "V_SMem", "V_vis", "F", "#A2A"]


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


def load_runs(results_path: Path):
    """Return {scenario: [eval_dict, ...]} for every evaluation_*_run*.json found."""
    runs = defaultdict(list)
    for pf in sorted(results_path.rglob("evaluation_*_run*.json")):
        scen, _ = _scenario_run(pf.name)
        try:
            runs[scen].append(json.load(open(pf)))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ! skipping {pf.name}: {e}", file=sys.stderr)
    return runs


def _per_run_rate(sub, denom_key):
    """|flagged| / denom for one run's metric sub-dict, or None if absent."""
    if not isinstance(sub, dict):
        return None, None, None
    flagged = set(sub.get("flagged", []) or [])
    denom = sub.get(denom_key) or 0
    rate = (len(flagged) / denom) if denom else 0.0
    return flagged, denom, rate


def aggregate_violation(runs_by_scenario, path, denom_key, mode):
    """Aggregate one violation metric across scenarios/runs for the given mode.
    Returns (rate or None, n_scenarios_with_data)."""
    per_scenario_rates = []
    for scen, runs in runs_by_scenario.items():
        subs = [_dig(r, path) for r in runs]
        present = [(_per_run_rate(s, denom_key)) for s in subs if isinstance(s, dict)]
        if not present:
            continue
        if mode == "any_k":
            union = set().union(*(f for f, _, _ in present))
            denom = next((d for _, d, _ in present if d), 0)
            per_scenario_rates.append(len(union) / denom if denom else 0.0)
        elif mode == "worst":
            per_scenario_rates.append(max(r for _, _, r in present))
        elif mode == "mean":
            per_scenario_rates.append(sum(r for _, _, r in present) / len(present))
    if not per_scenario_rates:
        return None, 0
    return sum(per_scenario_rates) / len(per_scenario_rates), len(per_scenario_rates)


def aggregate_scalars(runs_by_scenario):
    """C, U, F, #A2A — all averaged over scenario-run pairs (mode-independent)."""
    comps, utils, fails, a2as = [], [], [], []
    for runs in runs_by_scenario.values():
        for r in runs:
            c = _dig(r, ["completeness", "rate"])
            if c is not None:
                comps.append(c)
            u = _dig(r, ["utility", "score"])
            if u is not None:
                utils.append(u)
            vappr = _dig(r, ["V_appr", "flagged"])
            if vappr is not None:
                fails.append(1 if len(vappr) > 0 else 0)
            n = _dig(r, ["efficiency", "rounds"])
            if n is not None:
                a2as.append(n)
    def _avg(xs):
        return sum(xs) / len(xs) if xs else None
    return {"C": _avg(comps), "U": _avg(utils), "F": _avg(fails), "#A2A": _avg(a2as)}


def compute(runs_by_scenario, mode):
    out = {}
    for name, (path, denom_key) in VIOLATION_METRICS.items():
        out[name], _ = aggregate_violation(runs_by_scenario, path, denom_key, mode)
    out.update(aggregate_scalars(runs_by_scenario))
    return out


def _fmt(name, val):
    if val is None:
        return "—"
    if name == "#A2A":
        return f"{val:.1f}"
    return f"{val:.1%}"


def main():
    ap = argparse.ArgumentParser(description="Aggregate PiSAs evaluation results into paper metrics.")
    ap.add_argument("--results-path", required=True, metavar="PATH",
                    help="Config folder containing evaluation_*_run*.json (searched recursively).")
    ap.add_argument("--mode", choices=["any_k", "worst", "mean", "all"], default="any_k",
                    help="Violation aggregation across runs (default: any_k = paper main results).")
    ap.add_argument("--json", default=None, metavar="FILE",
                    help="Also write the computed metrics to this JSON file.")
    args = ap.parse_args()

    results_path = Path(args.results_path).resolve()
    if not results_path.is_dir():
        print(f"ERROR: not a directory: {results_path}", file=sys.stderr)
        sys.exit(1)

    runs_by_scenario = load_runs(results_path)
    if not runs_by_scenario:
        print(f"ERROR: no evaluation_*_run*.json under {results_path}", file=sys.stderr)
        sys.exit(1)

    n_scen = len(runs_by_scenario)
    n_runs = sum(len(v) for v in runs_by_scenario.values())
    modes = ["any_k", "worst", "mean"] if args.mode == "all" else [args.mode]
    results = {m: compute(runs_by_scenario, m) for m in modes}

    print("═" * 56)
    print(f"  {results_path.name}")
    print(f"  scenarios={n_scen}  eval files={n_runs}  mode={args.mode}")
    print("═" * 56)
    # C/U/F/#A2A are mode-independent — take them from the first mode.
    scalar0 = results[modes[0]]
    header = "  Metric    " + "".join(f"{m:>10}" for m in modes)
    print(header if len(modes) > 1 else "  Metric        Rate")
    for name in ROW_ORDER:
        if name in ("C", "U", "F", "#A2A"):
            print(f"  {name:<8}" + (f"{_fmt(name, scalar0[name]):>10}" if len(modes) == 1
                                    else "".join(f"{_fmt(name, scalar0[name]):>10}" for _ in modes)))
        else:
            print(f"  {name:<8}" + "".join(f"{_fmt(name, results[m][name]):>10}" for m in modes))
    print("═" * 56)

    if args.json:
        payload = {"results_path": str(results_path), "n_scenarios": n_scen,
                   "n_eval_files": n_runs, "metrics": results}
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
