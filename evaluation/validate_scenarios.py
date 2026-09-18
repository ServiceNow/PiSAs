#!/usr/bin/env python3
"""
PiSAs — scenario bundle validator.

Checks a folder of scenario bundles for the mistakes that make a run meaningless
rather than merely wrong: attribute ids that disagree between files, a fact no
artifact carries, a name nobody in the cast has, an oracle answer outside the
allowed set, or an executor who already holds everything worth gathering.

Run it before spending money on a batch:

    python validate_scenarios.py my_task/                 # a folder of scenario_NN/
    python validate_scenarios.py my_task/scenario_01      # or one bundle
    python validate_scenarios.py my_task/ --strict        # warnings fail too

Exit code 0 when every bundle passes, 1 otherwise (2 for a usage error), so it
drops straight into a Makefile or CI step.
"""

import argparse
import json
import sys
from pathlib import Path

FILES = ("scenario.json", "utility.json", "visibility.json", "appropriateness.json")


def _cast_names(scenario: dict) -> set:
    cast = scenario.get("cast", {}) or {}
    members = cast if isinstance(cast, list) else list(cast.values())
    return {m.get("name") for m in members if isinstance(m, dict) and m.get("name")}


def _carried_attributes(scenario: dict) -> set:
    """Attribute ids that some timeline artifact declares it carries."""
    out = set()
    for art in scenario.get("timeline", []) or []:
        for key in ("attributes", "attribute_ids", "carries", "reveals"):
            out.update(art.get(key, []) or [])
    return out


def _artifact_ids(scenario: dict) -> set:
    return {art.get("id") for art in scenario.get("timeline", []) or [] if art.get("id")}


def check_bundle(folder: Path):
    """Return (errors, warnings) for one scenario bundle."""
    errors, warnings = [], []

    missing = [f for f in FILES if not (folder / f).exists()]
    if missing:
        return [f"missing file(s): {', '.join(missing)}"], []
    try:
        scenario, utility, visibility, appropriateness = (
            json.load(open(folder / f)) for f in FILES)
    except json.JSONDecodeError as e:
        return [f"invalid JSON: {e}"], []

    attr_values = utility.get("attribute_values", {}) or {}
    ids = set(attr_values)
    appr = appropriateness.get("attributes", {}) or {}
    vis = visibility.get("attributes", {}) or {}
    names = _cast_names(scenario)
    carried = _carried_attributes(scenario)

    # ── The three files must describe the same attributes ──
    if not ids:
        errors.append("utility.json has no attribute_values")
    for label, other in (("appropriateness.json", set(appr)), ("visibility.json", set(vis))):
        only_here, only_there = ids - other, other - ids
        if only_here:
            errors.append(f"{label} is missing {len(only_here)} attribute(s): {sorted(only_here)[:5]}")
        if only_there:
            errors.append(f"{label} has {len(only_there)} attribute(s) absent from utility.json: {sorted(only_there)[:5]}")

    bad_labels = {a: v for a, v in appr.items() if v not in ("appropriate", "inappropriate")}
    if bad_labels:
        errors.append(f"appropriateness must be 'appropriate' or 'inappropriate': {list(bad_labels.items())[:3]}")

    # ── Every fact must actually be somewhere an agent can read it ──
    if carried:
        # An attribute counts as carried if an artifact lists it, or if the attribute
        # names an existing artifact as its source (some tasks record it only that way).
        art_ids = _artifact_ids(scenario)
        sourced = {a for a, info in attr_values.items() if (info or {}).get("source") in art_ids}
        orphans = ids - carried - sourced
        if orphans:
            errors.append(f"{len(orphans)} attribute(s) carried by no timeline artifact: {sorted(orphans)[:5]}")
    else:
        warnings.append("no timeline artifact declares which attributes it carries — "
                        "the 'attributes' field on each artifact is how coverage is checked")

    # ── Visibility has to be self-consistent and name real people ──
    for aid, v in vis.items():
        holder = v.get("holder")
        visible = set(v.get("visible_to", []) or [])
        hidden = set(v.get("hidden_from", []) or [])
        # A holder is usually a person, but some tasks attribute a fact to a system of
        # record instead; that is fine, it just cannot be an agent that holds context.
        if holder and holder in names and holder not in visible:
            errors.append(f"{aid}: holder {holder!r} is not in visible_to")
        elif holder and holder not in names:
            warnings.append(f"{aid}: holder {holder!r} is not a cast member — treated as a non-agent source")
        overlap = visible & hidden
        if overlap:
            errors.append(f"{aid}: {sorted(overlap)} are in both visible_to and hidden_from")
        unknown = (visible | hidden) - names
        if unknown:
            # Not fatal: the harness matches agent names and ignores anything else, and
            # some tasks list a system of record alongside the people. But a typo here
            # silently means "this can never be a violation", so it is worth seeing.
            warnings.append(f"{aid}: visible_to/hidden_from name(s) not in the cast, ignored "
                            f"when scoring: {sorted(unknown)[:5]}")
        if not hidden:
            warnings.append(f"{aid}: hidden_from is empty, so it can never be a visibility violation")

    # ── The decision has to be well posed ──
    allowed = utility.get("allowed_answers") or []
    oracle = utility.get("oracle_answer")
    if not allowed:
        errors.append("utility.json has no allowed_answers")
    elif oracle not in allowed:
        errors.append(f"oracle_answer {oracle!r} is not one of allowed_answers {allowed}")
    if not (utility.get("decision_rule") or "").strip():
        warnings.append("utility.json has no decision_rule — the decision judge is told nothing "
                        "about how the answer follows from the facts")

    # ── The task has to have someone to execute it ──
    task = scenario.get("task", {}) or {}
    role = task.get("executor_role")
    participants = task.get("participants", {}) or {}
    if not role or role not in participants:
        errors.append(f"task.executor_role {role!r} is not a key of task.participants")
    else:
        executor = (participants[role] or {}).get("name")
        if executor not in names:
            errors.append(f"executor {executor!r} is not in the cast")
        else:
            # If the executor already holds every appropriate fact, there is nothing to
            # gather and the multi-agent systems have no work to do. (Being merely
            # *visible* to the executor is normal — the executor is often allowed to
            # learn a fact that a colleague holds.)
            appropriate = {a for a, v in appr.items() if v == "appropriate"}
            held = {a for a in appropriate if vis.get(a, {}).get("holder") == executor}
            if appropriate and held == appropriate:
                warnings.append("the executor already holds every appropriate attribute — "
                                "there is nothing to gather")
    # The instruction the executor receives is task.description, a timeline artifact of
    # type "task", or both concatenated — so only the absence of both is an error.
    task_artifact = any((art.get("type") == "task" and (art.get("content") or "").strip())
                        for art in scenario.get("timeline", []) or [])
    if not (task.get("description") or "").strip() and not task_artifact:
        errors.append("no task instruction: task.description is empty and no timeline artifact "
                      "has type 'task'")

    # ── Scoring needs both classes to exist ──
    if not any(v == "inappropriate" for v in appr.values()):
        warnings.append("no inappropriate attributes: V_G, V_A2A, V_out and V_appr will all be 0")
    if not any(v == "appropriate" for v in appr.values()):
        warnings.append("no appropriate attributes: completeness cannot be computed")

    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description="Validate PiSAs scenario bundles.")
    ap.add_argument("path", help="A scenario folder, or a folder of scenario folders.")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    ap.add_argument("--quiet", action="store_true", help="Only print bundles that have something to report.")
    args = ap.parse_args()

    root = Path(args.path)
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        sys.exit(2)

    bundles = ([root] if (root / "scenario.json").exists()
               else sorted(p for p in root.iterdir() if p.is_dir() and (p / "scenario.json").exists()))
    if not bundles:
        print(f"ERROR: no scenario bundles under {root}", file=sys.stderr)
        sys.exit(2)

    n_bad = n_warn = 0
    for b in bundles:
        errors, warnings = check_bundle(b)
        if errors:
            n_bad += 1
        if warnings:
            n_warn += 1
        if not errors and not warnings:
            if not args.quiet:
                print(f"  ok    {b.name}")
            continue
        print(f"  {'FAIL' if errors else 'warn'}  {b.name}")
        for e in errors:
            print(f"          error:   {e}")
        for w in warnings:
            print(f"          warning: {w}")

    print(f"\n  {len(bundles)} bundle(s): {len(bundles) - n_bad} passed, {n_bad} failed, "
          f"{n_warn} with warnings")
    sys.exit(1 if (n_bad or (args.strict and n_warn)) else 0)


if __name__ == "__main__":
    main()
