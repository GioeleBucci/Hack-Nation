"""Acceleration benchmark: simulations needed to find a near-best strategy (docs/acceleration.md).

1. Evaluate every configuration of a fixed strategy space once on a common screening
   ensemble (this is the "ground truth" ranking; it is an offline measurement and does not
   touch the lab's record or budget).
2. Random search: shuffle the space R times; count simulations until a configuration within
   5% of the best reduction is found. Grid search: the fixed enumeration order.
3. Agent: walk the lab's recorded experiments in order, adding up the simulations it actually
   spent, until an experiment contains a plan that reaches the same target (plans are scored on
   the same screening ensemble, so the comparison is like for like).
"""

from __future__ import annotations

import json

import numpy as np

from firelab import config
from firelab.landscape import load_landscape
from firelab.placement import build_plan
from firelab.record import Record
from firelab.scenarios import ignition_points, weather_scenarios
from firelab.simulator import run_ensemble

MANUAL_PREFIX = "R0-manual"

SEARCH_SPACE: list[tuple[str, dict]] = [
    ("fuel_load", {}), ("burn_prob", {}), ("path_centrality", {}),
    *[("asset_buffer", {"upwind_weight": u, "max_distance": d}) for u in (0.0, 0.5, 1.0) for d in (None, 15)],
    *[("wind_strips", {"spacing": s, "width": w, "priority": p})
      for s in (8, 12, 20) for w in (1, 2, 3) for p in ("fuel_load", "burn_prob")],
    *[("hybrid", {"weights": w}) for w in (
        {"burn_prob": 0.5, "asset_proximity": 0.5},
        {"burn_prob": 0.5, "upwind_of_assets": 0.5},
        {"centrality": 0.5, "upwind_of_assets": 0.5},
        {"fuel_load": 0.3, "burn_prob": 0.3, "upwind_of_assets": 0.4},
    )],
]


class Evaluator:
    """Scores plans as % reduction in mean burned area vs no treatment on one fixed ensemble."""

    def __init__(self, budget_pct: float, n_ignitions: int, n_weather: int, seed: int = 0):
        self.ls = load_landscape()
        self.budget_pct = budget_pct
        self.ignitions = ignition_points(self.ls, "train", n_ignitions)
        self.weathers = weather_scenarios("train", n_weather)
        self.seed = seed
        self.cost_per_plan = n_ignitions * n_weather
        self.s0_mean = self._mean_ba([np.zeros(self.ls.shape, dtype=bool)])[0]

    def _mean_ba(self, masks: list[np.ndarray]) -> np.ndarray:
        out = run_ensemble(self.ls, np.stack(masks), self.ignitions, self.weathers, self.seed)
        return out.burned_ha.mean(axis=1)

    def reductions(self, configs: list[tuple[str, dict, int]], batch: int = 16) -> list[float]:
        masks = [build_plan(self.ls, s, self.budget_pct, seed, p)[0] for s, p, seed in configs]
        means = np.concatenate([self._mean_ba(masks[i:i + batch]) for i in range(0, len(masks), batch)])
        return [float(100 * (1 - m / self.s0_mean)) for m in means]


def sims_to_target(order: list[int], scores: list[float], target: float, cost: int, overhead: int) -> int | None:
    for k, idx in enumerate(order, start=1):
        if scores[idx] >= target:
            return overhead + k * cost
    return None


def agent_trajectory(rec: Record, ev: Evaluator) -> list[dict]:
    """Per recorded experiment: cumulative simulations spent and the best screening score so far."""
    traj, spent, best = [], 0, -np.inf
    for exp in rec.get("experiment"):
        p = exp["payload"]
        if p.get("status") != "done" or exp["id"].startswith(MANUAL_PREFIX):
            continue   # the manual round-0 baseline is the human comparison, not agent spend
        spent += p["cost_sims"]
        plans = [x["payload"] for x in rec.get("plan", p["plan_ids"])
                 if x["payload"]["strategy"] not in ("none", "random")]
        if plans:
            scores = ev.reductions([(x["strategy"], x["params"], x["seed"]) for x in plans])
            best = max(best, *scores)
        traj.append({"experiment_id": exp["id"], "cumulative_sims": spent, "best_reduction_pct": best})
    return traj


def run_benchmark(budget_pct: float = 5.0, n_ignitions: int = 40, n_weather: int = 3,
                  repeats: int = 20, seed: int = 0) -> dict:
    ev = Evaluator(budget_pct, n_ignitions, n_weather, seed)
    scores = ev.reductions([(s, p, 0) for s, p in SEARCH_SPACE])
    traj = agent_trajectory(Record(), ev)
    agent_best = max((t["best_reduction_pct"] for t in traj), default=-np.inf)
    best = max(max(scores), agent_best)
    target = 0.95 * best
    cost, overhead = ev.cost_per_plan, ev.cost_per_plan   # overhead = the no-treatment reference run

    rng = np.random.default_rng(seed)
    random_sims = [sims_to_target(list(rng.permutation(len(scores))), scores, target, cost, overhead)
                   for _ in range(repeats)]
    found = [s for s in random_sims if s is not None]
    agent_sims = next((t["cumulative_sims"] for t in traj if t["best_reduction_pct"] >= target), None)

    result = {
        "setup": {"budget_pct": budget_pct, "n_ignitions": n_ignitions, "n_weather": n_weather,
                  "sims_per_config": cost, "space_size": len(SEARCH_SPACE), "repeats": repeats},
        "best_reduction_pct": best,
        "target_reduction_pct": target,
        "space_scores": [{"strategy": s, "params": p, "reduction_pct": round(r, 2)}
                         for (s, p), r in sorted(zip(SEARCH_SPACE, scores), key=lambda x: -x[1])],
        "grid_search_sims": sims_to_target(list(range(len(scores))), scores, target, cost, overhead),
        "random_search_sims": {
            "median": float(np.median(found)) if found else None,
            "iqr": [float(np.percentile(found, 25)), float(np.percentile(found, 75))] if found else None,
            "found_in": f"{len(found)}/{repeats}",
        },
        "agent_sims": agent_sims,
        "agent_trajectory": traj,
    }
    if agent_sims and found:
        result["speedup_vs_random_median"] = round(float(np.median(found)) / agent_sims, 2)
        # Spread of the ratio over the random-search repeats (uncertainty of the comparison).
        ratios = np.array(found, dtype=float) / agent_sims
        result["speedup_vs_random_iqr"] = [round(float(np.percentile(ratios, 25)), 2),
                                           round(float(np.percentile(ratios, 75)), 2)]
    if agent_sims and result["grid_search_sims"]:
        result["speedup_vs_grid"] = round(result["grid_search_sims"] / agent_sims, 2)

    config.ensure_dirs()
    (config.RESULTS_DIR / "acceleration.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
