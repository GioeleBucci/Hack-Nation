"""run_ensemble: burn a set of plans under identical scenarios and record the results."""

from __future__ import annotations

import os
import time

import numpy as np

from firelab import config
from firelab.landscape import load_landscape
from firelab.scenarios import ignition_points, weather_scenarios
from firelab.simulator import run_ensemble as simulate
from firelab.tools._common import ToolError, check_id, get_record, log, tool
from firelab.tools.strategies import load_plan_mask

MAX_PLANS_PER_CALL = 16


def _log_mlflow(experiment_id: str, plan: dict, spec: dict, metrics: dict) -> str | None:
    """Log one plan's result as an MLflow run; returns the run id, or None if MLflow is unavailable."""
    if not config.USE_MLFLOW:
        return None
    try:
        import mlflow
    except ImportError:
        return None
    try:
        if "MLFLOW_TRACKING_URI" not in os.environ:   # set it to "databricks" to log to the workspace
            mlflow.set_tracking_uri((config.ROOT / "mlruns").as_uri())
        mlflow.set_experiment("firelab")
        with mlflow.start_run(run_name=f"{experiment_id}/{plan['id']}") as run:
            mlflow.set_tags({"experiment_id": experiment_id, "plan_id": plan["id"], "strategy": plan["strategy"]})
            mlflow.log_params({**{k: v for k, v in spec.items() if k != "plan_ids"},
                               "plan_params": str(plan["params"]), "budget_pct": plan["budget_pct"]})
            mlflow.log_metrics(metrics)
            return run.info.run_id
    except Exception:  # noqa: BLE001 - tracking must never break an experiment
        log.exception("MLflow logging failed")
        return None


def _metrics(burned_ha: np.ndarray, asset_cells: np.ndarray) -> dict:
    return {
        "mean_ba_ha": float(burned_ha.mean()),
        "sd_ba_ha": float(burned_ha.std(ddof=1)) if burned_ha.size > 1 else 0.0,
        "p90_ba_ha": float(np.percentile(burned_ha, 90)),
        "mean_asset_cells": float(asset_cells.mean()),
        "p_asset_loss": float((asset_cells > 0).mean()),
    }


@tool
def run_ensemble(experiment_id: str, plan_ids: list[str], ignition_set: str = "train",
                 weather_set: str = "train", n_ignitions: int = 50, n_weather: int = 5,
                 seed: int = 0, option_id: str | None = None) -> dict:
    """Simulate every plan under the same scenarios. Results are written by this tool, not by agents."""
    rec = get_record()
    ls = load_landscape()
    check_id(experiment_id, "experiment_id")
    if rec.get_one("experiment", experiment_id):
        raise ToolError(f"Experiment '{experiment_id}' already exists; results are immutable. Use a new id.")
    plan_ids = list(dict.fromkeys(plan_ids or []))
    if not 1 <= len(plan_ids) <= MAX_PLANS_PER_CALL:
        raise ToolError(f"Pass between 1 and {MAX_PLANS_PER_CALL} plan_ids")
    plans = [rec.get_one("plan", pid) for pid in plan_ids]
    missing = [pid for pid, p in zip(plan_ids, plans) if p is None]
    if missing:
        raise ToolError(f"Unknown plan ids {missing}; build them with generate_plan first")
    plans = [p["payload"] for p in plans]
    if any(p["landscape_fingerprint"] != ls.fingerprint() for p in plans):
        raise ToolError("Some plans were built on a different landscape")

    try:
        ignitions = ignition_points(ls, ignition_set, int(n_ignitions))
        weathers = weather_scenarios(weather_set, int(n_weather))
    except ValueError as e:
        raise ToolError(str(e)) from e

    # Budget guard (defense in depth: the simulation_budget policy checks the same numbers).
    cost = len(plan_ids) * int(n_ignitions) * int(n_weather)
    if cost > config.MAX_SIMS_PER_CALL:
        raise ToolError(f"Request needs {cost} simulations; the per-call cap is {config.MAX_SIMS_PER_CALL}")
    used = rec.sims_used()
    if used + cost > config.MAX_SIMS:
        raise ToolError(f"Simulation budget exhausted: {used}/{config.MAX_SIMS} used, request needs {cost}. "
                        "Choose a cheaper experiment or stop.")

    spec = {"id": experiment_id, "option_id": option_id, "plan_ids": plan_ids,
            "ignition_set": ignition_set, "weather_set": weather_set,
            "n_ignitions": int(n_ignitions), "n_weather": int(n_weather), "seed": int(seed),
            "cost_sims": cost, "simulator_version": config.SIMULATOR_VERSION,
            "landscape": ls.name, "landscape_fingerprint": ls.fingerprint()}
    rec.put("experiment", {**spec, "status": "running"}, author="run_ensemble")   # reserves the budget

    try:
        masks = np.stack([load_plan_mask(p, ls) for p in plans])
        t0 = time.perf_counter()
        out = simulate(ls, masks, ignitions, weathers, seed=int(seed))
        runtime = time.perf_counter() - t0
    except Exception:
        rec.put("experiment", {**spec, "cost_sims": 0, "status": "failed"}, author="run_ensemble")
        raise

    config.ensure_dirs()
    runs_file = config.RUNS_DIR / f"{experiment_id}.npz"
    np.savez_compressed(runs_file, plan_ids=np.array(plan_ids), burned_ha=out.burned_ha,
                        asset_cells=out.asset_cells, burn_count=out.burn_count.astype(np.uint16),
                        scenarios=np.array(out.scenarios))

    table = []
    for k, plan in enumerate(plans):
        metrics = _metrics(out.burned_ha[k], out.asset_cells[k])
        run_id = _log_mlflow(experiment_id, plan, spec, metrics)
        rec.put("result", {"id": f"{experiment_id}/{plan['id']}", "experiment_id": experiment_id,
                           "plan_id": plan["id"], "strategy": plan["strategy"], "metrics": metrics,
                           "seed": int(seed), "mlflow_run_id": run_id}, author="run_ensemble")
        table.append({"plan_id": plan["id"], "strategy": plan["strategy"], **metrics, "mlflow_run_id": run_id})

    rec.put("experiment", {**spec, "status": "done", "runtime_s": round(runtime, 2),
                           "runs_file": runs_file.relative_to(config.ROOT).as_posix()}, author="run_ensemble")
    used_now = rec.sims_used()
    return {"experiment_id": experiment_id, "cost_sims": cost, "runtime_s": round(runtime, 2),
            "sims_used": used_now, "sims_remaining": config.MAX_SIMS - used_now,
            "n_scenarios": len(out.scenarios),
            "results": sorted(table, key=lambda r: r["mean_ba_ha"]),
            "next": "Call compute_stats with the random control(s) as baseline_plan_ids."}
