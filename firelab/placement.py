"""Treatment-placement strategies: turn (strategy, params, budget) into a set of treated cells.

All strategies work the same way: build a score per cell, restrict to eligible cells
(burnable, not an asset, plus any strategy-specific mask) and treat the top-k cells, where
k = budget_pct % of the burnable cells. Ties are broken randomly with the plan seed.
See docs/strategies.md for the rationale behind each strategy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from firelab import config
from firelab.landscape import Landscape
from firelab.scenarios import WEATHER_SETS
from firelab.simulator import downwind_vector, shift

DEFAULT_WIND_FROM = WEATHER_SETS["train"].dominant_from_deg

LAYERS = ("fuel_load", "burn_prob", "centrality", "asset_proximity", "upwind_of_assets")

STRATEGIES: dict[str, dict[str, Any]] = {
    "none": {"code": "S0", "defaults": {},
             "description": "No treatment (control)."},
    "random": {"code": "SR", "defaults": {},
               "description": "Random cells at the same budget (control; use several seeds)."},
    "fuel_load": {"code": "S1", "defaults": {},
                  "description": "Highest fuel load first."},
    "burn_prob": {"code": "S2", "defaults": {},
                  "description": "Highest no-treatment burn probability first (reference map)."},
    "asset_buffer": {"code": "S3",
                     "defaults": {"max_distance": None, "upwind_weight": 0.0, "wind_from_deg": DEFAULT_WIND_FROM},
                     "description": "Cells closest to assets; upwind_weight > 0 favours the upwind side."},
    "wind_strips": {"code": "S4",
                    "defaults": {"spacing": 12, "width": 2, "angle_offset_deg": 0.0,
                                 "priority": "fuel_load", "wind_from_deg": DEFAULT_WIND_FROM},
                    "description": "Parallel strips perpendicular to the dominant wind (rotated by angle_offset_deg); "
                                   "within strips, cells ranked by the 'priority' layer."},
    "path_centrality": {"code": "S5", "defaults": {},
                        "description": "Cells that most often carried fire on to neighbours in large reference fires."},
    "hybrid": {"code": "S6",
               "defaults": {"weights": {"burn_prob": 0.5, "asset_proximity": 0.5}, "wind_from_deg": DEFAULT_WIND_FROM},
               "description": f"Weighted sum of normalized layers {list(LAYERS)}."},
}


def _normalize(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = a[mask]
    if vals.size == 0 or vals.max() == vals.min():
        return np.zeros_like(a, dtype=np.float64)
    return np.clip((a - vals.min()) / (vals.max() - vals.min()), 0, 1)


def asset_proximity(ls: Landscape) -> np.ndarray:
    if not ls.assets.any():
        return np.zeros(ls.shape)
    return 1.0 / (1.0 + distance_transform_edt(~ls.assets))


def upwind_of_assets(ls: Landscape, wind_from_deg: float, reach: int = 30, decay: float = 0.92) -> np.ndarray:
    """High where an asset lies a short distance downwind of the cell."""
    vy, vx = downwind_vector(wind_from_deg)
    targets = binary_dilation(ls.assets, iterations=2).astype(np.float64)
    acc = np.zeros(ls.shape)
    for k in range(1, reach + 1):
        dy, dx = int(round(k * vy)), int(round(k * vx))
        acc += decay ** k * shift(targets, -dy, -dx)    # = targets[y + dy, x + dx]
    return acc


def layer(ls: Landscape, name: str, wind_from_deg: float) -> np.ndarray:
    """One normalized score layer in [0, 1] over treatable cells."""
    if name == "fuel_load":
        raw = ls.fuel
    elif name in ("burn_prob", "centrality"):
        from firelab.reference import load_reference   # lazy: may trigger the one-off reference build
        ref = load_reference(ls.name)
        raw = ref.burn_prob if name == "burn_prob" else ref.centrality
    elif name == "asset_proximity":
        raw = asset_proximity(ls)
    elif name == "upwind_of_assets":
        raw = upwind_of_assets(ls, wind_from_deg)
    else:
        raise ValueError(f"Unknown layer '{name}'. Known: {list(LAYERS)}")
    return _normalize(np.asarray(raw, dtype=np.float64), ls.treatable)


def _top_k(score: np.ndarray, eligible: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(eligible)
    order = np.lexsort((rng.random(idx.size), -score.ravel()[idx]))   # by score, ties random
    mask = np.zeros(score.size, dtype=bool)
    mask[idx[order[:k]]] = True
    return mask.reshape(score.shape)


def resolve_params(strategy: str, params: dict | None) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Known: {list(STRATEGIES)}")
    defaults = STRATEGIES[strategy]["defaults"]
    params = dict(params or {})
    unknown = set(params) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown params {sorted(unknown)} for '{strategy}'. Allowed: {sorted(defaults)}")
    return {**defaults, **params}


def build_plan(ls: Landscape, strategy: str, budget_pct: float, seed: int,
               params: dict | None = None) -> tuple[np.ndarray, dict]:
    """Return the (H, W) treated mask and the fully resolved parameters."""
    params = resolve_params(strategy, params)
    if not 0 <= budget_pct <= config.MAX_BUDGET_PCT:
        raise ValueError(f"budget_pct must be in [0, {config.MAX_BUDGET_PCT}]")
    rng = np.random.default_rng(seed)
    k = int(round(budget_pct / 100 * ls.n_burnable))
    eligible = ls.treatable.copy()
    wind = float(params.get("wind_from_deg", DEFAULT_WIND_FROM))

    if strategy == "none":
        return np.zeros(ls.shape, dtype=bool), params
    if strategy == "random":
        score = rng.random(ls.shape)
    elif strategy == "fuel_load":
        score = layer(ls, "fuel_load", wind)
    elif strategy == "burn_prob":
        score = layer(ls, "burn_prob", wind)
    elif strategy == "path_centrality":
        score = layer(ls, "centrality", wind)
    elif strategy == "asset_buffer":
        score = layer(ls, "asset_proximity", wind) + float(params["upwind_weight"]) * layer(ls, "upwind_of_assets", wind)
        if params["max_distance"] is not None:
            eligible &= distance_transform_edt(~ls.assets) <= float(params["max_distance"])
    elif strategy == "wind_strips":
        spacing, width = float(params["spacing"]), float(params["width"])
        if spacing <= 0 or not 0 < width < spacing:
            raise ValueError("wind_strips needs spacing > 0 and 0 < width < spacing")
        vy, vx = downwind_vector(wind + float(params["angle_offset_deg"]))
        yy, xx = np.mgrid[0:ls.shape[0], 0:ls.shape[1]]
        along_wind = yy * vy + xx * vx          # constant along each strip
        eligible &= np.mod(along_wind, spacing) < width
        score = layer(ls, str(params["priority"]), wind)
    elif strategy == "hybrid":
        weights = params["weights"]
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"hybrid needs a non-empty 'weights' dict over {list(LAYERS)}")
        score = sum(float(w) * layer(ls, name, wind) for name, w in weights.items())
    else:  # pragma: no cover - guarded by resolve_params
        raise ValueError(strategy)

    return _top_k(np.asarray(score, dtype=np.float64), eligible, k, rng), params
