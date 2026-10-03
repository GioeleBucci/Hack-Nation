"""Command line for humans: setup, the manual round-0 baseline, approvals, export, benchmark, figures.

    firelab init                      generate the landscape and reference maps
    firelab status                    same as the lab_status tool
    firelab baseline                  round 0: no treatment vs random, by hand, timed
    firelab approve REC1 --by NAME    approve a pending recommendation (manual approval mode)
    firelab reject REC1 --by NAME --reason "..."
    firelab export                    write results/record_export.json, evidence.md, report.md
    firelab bench                     acceleration benchmark (agent vs random/grid search)
    firelab figures EXP-R1-A          maps and effect-size chart for one experiment
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from firelab import config


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_init(args) -> None:
    from firelab.landscape import load_landscape
    from firelab.reference import load_reference
    config.ensure_dirs()
    ls = load_landscape(args.landscape)
    t0 = time.perf_counter()
    ref = load_reference(ls.name)
    _print({"landscape": ls.summary(), "reference_sims": ref.n_sims,
            "reference_mean_burn_prob": float(ref.burn_prob[ls.burnable].mean()),
            "seconds": round(time.perf_counter() - t0, 1)})


def cmd_status(args) -> None:
    from firelab.tools.record import lab_status
    _print(lab_status())


def cmd_baseline(args) -> None:
    """The manual round 0. Time how long the whole cycle takes you, including reading the output."""
    from firelab.bench import MANUAL_PREFIX
    from firelab.landscape import load_landscape
    from firelab.tools._common import get_record
    from firelab.tools.simulate import run_ensemble
    from firelab.tools.stats import compute_stats
    from firelab.tools.strategies import generate_plan

    t0 = time.perf_counter()
    plans = [generate_plan("none", args.budget, 0)]
    plans += [generate_plan("random", args.budget, s) for s in range(1, args.replicates + 1)]
    for p in plans:
        if not p["ok"]:
            sys.exit(p["error"])
    ids = [p["id"] for p in plans]
    n_existing = sum(e["id"].startswith(MANUAL_PREFIX) for e in get_record().get("experiment"))
    exp_id = f"{MANUAL_PREFIX}-{n_existing + 1}"
    run = run_ensemble(exp_id, ids, "train", "train", args.n_ignitions, args.n_weather, seed=0)
    if not run["ok"]:
        sys.exit(run["error"])
    stats = compute_stats(exp_id, ids[1:])
    elapsed = time.perf_counter() - t0
    table = stats["by_metric"]["burned_ha"]["comparisons"]
    _print({"experiment_id": exp_id, "compute_seconds": round(elapsed, 1),
            "results": run["results"],
            "no_treatment_vs_random": [{k: r[k] for k in ("plan_id", "strategy", "rel_change_pct",
                                                            "rel_change_ci95", "verdict")} for r in table]})
    ls = load_landscape()
    mean_s0 = next(r["mean_ba_ha"] for r in run["results"] if r["strategy"] == "none")
    share = 100 * mean_s0 / (ls.n_burnable * ls.cell_ha)
    print(f"\nMean burned area without treatment: {mean_s0:.0f} ha ({share:.1f}% of the burnable area). "
          "If this is below ~1% or above ~40%, tune SpreadParams (docs/simulator.md) before running the lab.",
          file=sys.stderr)


def _decide(args, verdict: str) -> None:
    from firelab.tools._common import get_record
    rec = get_record()
    item = rec.get_one("recommendation", args.recommendation_id)
    if not item:
        sys.exit(f"Unknown recommendation {args.recommendation_id}")
    if item["payload"]["status"] != "pending_approval":
        sys.exit(f"{args.recommendation_id} is already {item['payload']['status']}")
    rec.put("approval", {"recommendation_id": item["id"], "verdict": verdict, "approved_by": args.by,
                         "via": "firelab_cli", "reason": getattr(args, "reason", None)}, author=args.by)
    rec.put("recommendation", {**item["payload"], "status": verdict}, author=args.by)
    print(f"{args.recommendation_id}: {verdict} by {args.by}")


def cmd_export(args) -> None:
    from firelab.report import export
    _print(export())


def cmd_bench(args) -> None:
    from firelab.bench import run_benchmark
    result = run_benchmark(args.budget, args.n_ignitions, args.n_weather, args.repeats)
    _print({k: v for k, v in result.items() if k not in ("space_scores", "agent_trajectory")})
    print(f"\nFull output: {config.RESULTS_DIR / 'acceleration.json'}", file=sys.stderr)


def cmd_figures(args) -> None:
    from firelab.viz import experiment_figures
    _print(experiment_figures(args.experiment_id))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="firelab", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="generate landscape and reference maps")
    p.add_argument("--landscape", default=None)
    p.set_defaults(func=cmd_init)

    sub.add_parser("status", help="lab status").set_defaults(func=cmd_status)

    p = sub.add_parser("baseline", help="manual round 0: no treatment vs random")
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--n-ignitions", type=int, default=50)
    p.add_argument("--n-weather", type=int, default=5)
    p.set_defaults(func=cmd_baseline)

    for name, verdict in (("approve", "approved"), ("reject", "rejected")):
        p = sub.add_parser(name, help=f"{name} a pending recommendation")
        p.add_argument("recommendation_id")
        p.add_argument("--by", required=True)
        if name == "reject":
            p.add_argument("--reason", required=True)
        p.set_defaults(func=lambda a, v=verdict: _decide(a, v))

    sub.add_parser("export", help="export record and report").set_defaults(func=cmd_export)

    p = sub.add_parser("bench", help="acceleration benchmark")
    p.add_argument("--budget", type=float, default=5.0)
    p.add_argument("--n-ignitions", type=int, default=40)
    p.add_argument("--n-weather", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("figures", help="figures for one experiment")
    p.add_argument("experiment_id")
    p.set_defaults(func=cmd_figures)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
