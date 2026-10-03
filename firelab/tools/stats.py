"""compute_stats: paired comparison of every plan in an experiment against its controls."""

from __future__ import annotations

import numpy as np

from firelab import config
from firelab.stats import MIN_EFFECT_PCT, compare_plans
from firelab.tools._common import ToolError, get_record, tool

METRICS = {"burned_ha": "burned_ha", "asset_cells": "asset_cells"}


def load_runs(experiment_id: str) -> dict:
    exp = get_record().get_one("experiment", experiment_id)
    if not exp or exp["payload"].get("status") != "done":
        raise ToolError(f"No finished experiment '{experiment_id}'")
    with np.load(config.ROOT / exp["payload"]["runs_file"]) as z:
        return {"plan_ids": [str(p) for p in z["plan_ids"]], "burned_ha": z["burned_ha"],
                "asset_cells": z["asset_cells"], "burn_count": z["burn_count"], "experiment": exp["payload"]}


@tool
def compute_stats(experiment_id: str, baseline_plan_ids: list[str], metrics: list[str] | None = None,
                  n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap CIs, Wilcoxon signed-rank and Holm correction vs the baseline plan(s).

    With several baseline ids (e.g. random placements with different seeds) the baseline is
    their per-scenario mean.
    """
    runs = load_runs(experiment_id)
    baseline_plan_ids = list(dict.fromkeys(baseline_plan_ids or []))
    if not baseline_plan_ids or not set(baseline_plan_ids) <= set(runs["plan_ids"]):
        raise ToolError(f"baseline_plan_ids must be non-empty and among {runs['plan_ids']}")
    metrics = metrics or ["burned_ha", "asset_cells"]
    if not set(metrics) <= set(METRICS):
        raise ToolError(f"metrics must be among {list(METRICS)}")

    rec = get_record()
    plans = {p["id"]: p["payload"] for p in rec.get("plan", runs["plan_ids"])}
    out: dict = {"experiment_id": experiment_id,
                 "scenario_sets": {k: runs["experiment"][k] for k in
                                   ("ignition_set", "weather_set", "n_ignitions", "n_weather", "seed")},
                 "criterion": f"wins = mean burned area >= {MIN_EFFECT_PCT}% lower than baseline, "
                              "95% CI of the relative change below 0, Holm-adjusted Wilcoxon p < 0.05",
                 "by_metric": {}}
    for m in metrics:
        values = {pid: runs[METRICS[m]][k].astype(np.float64) for k, pid in enumerate(runs["plan_ids"])}
        result = compare_plans(values, baseline_plan_ids, n_boot=int(n_boot), seed=int(seed))
        for row in result["comparisons"]:
            row["strategy"] = plans[row["plan_id"]]["strategy"]
        out["by_metric"][m] = result

    # Treatment efficiency relative to no treatment: hectares of burned area avoided per treated hectare.
    s0 = [pid for pid in runs["plan_ids"] if plans[pid]["strategy"] == "none"]
    if s0:
        ba = runs["burned_ha"].mean(axis=1)
        ba_s0 = ba[runs["plan_ids"].index(s0[0])]
        out["efficiency_vs_no_treatment"] = {
            pid: round(float((ba_s0 - ba[k]) / plans[pid]["treated_ha"]), 4)
            for k, pid in enumerate(runs["plan_ids"]) if plans[pid]["treated_ha"] > 0}

    item = rec.put("stats", {**out, "baseline_plan_ids": baseline_plan_ids}, author="compute_stats")
    return {"stats_id": item["id"], **out}
