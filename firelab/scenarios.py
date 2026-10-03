"""Frozen ignition and weather sets.

Every plan in an experiment burns under exactly the same scenarios (ignition point x
weather), so all comparisons are paired. Sets are generated deterministically from a fixed
seed and requesting ``n`` items always returns the first ``n`` of the same fixed list, so a
small screening ensemble is a strict subset of a larger confirmation ensemble.

- ``train`` sets are used for screening and refinement.
- ``heldout*`` sets are reserved for the transfer test (a different dominant wind).
- The ``reference`` ignition set is used only to build the no-treatment reference maps
  (burn probability, spread-path centrality), so those maps are not fitted to the very
  ignitions the experiments are scored on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from firelab.landscape import Landscape

MAX_IGNITIONS = 1000
MAX_WEATHER = 50


@dataclass(frozen=True)
class Weather:
    wind_from_deg: float     # meteorological convention: direction the wind blows FROM, clockwise from north
    wind_speed_kmh: float
    dryness: float           # multiplies spread rates; >1 = drier than average fire-season day


@dataclass(frozen=True)
class WeatherSetSpec:
    dominant_from_deg: float
    direction_spread_deg: float
    speed_kmh: tuple[float, float]
    dryness: tuple[float, float]
    seed: int
    description: str


IGNITION_SETS: dict[str, dict] = {
    "train": {"seed": 101, "description": "Uniform over treatable cells; used for screening and refinement."},
    "heldout": {"seed": 202, "description": "Independent uniform draw; used for the transfer test."},
    "reference": {"seed": 303, "description": "Only used to build the no-treatment reference maps."},
}

WEATHER_SETS: dict[str, WeatherSetSpec] = {
    "train": WeatherSetSpec(225, 20, (15, 35), (0.85, 1.15), 11,
                            "High fire-weather days, dominant south-westerly wind."),
    "heldout_nw": WeatherSetSpec(315, 20, (15, 35), (0.85, 1.15), 22,
                                 "Same severity, dominant north-westerly wind (transfer test)."),
    "heldout_extreme": WeatherSetSpec(225, 25, (35, 55), (1.1, 1.35), 33,
                                      "Extreme days: stronger winds and drier fuels (stress test)."),
}


def ignition_points(ls: Landscape, set_name: str, n: int) -> np.ndarray:
    """First ``n`` ignition cells of the named set, as an (n, 2) array of (row, col)."""
    if set_name not in IGNITION_SETS:
        raise ValueError(f"Unknown ignition set '{set_name}'. Known: {sorted(IGNITION_SETS)}")
    if not 1 <= n <= MAX_IGNITIONS:
        raise ValueError(f"n_ignitions must be in [1, {MAX_IGNITIONS}]")
    # Seed also depends on the landscape so a different landscape gets its own valid cells.
    rng = np.random.default_rng([IGNITION_SETS[set_name]["seed"], ls.shape[0], ls.shape[1]])
    candidates = np.flatnonzero(ls.treatable)
    chosen = rng.choice(candidates, size=min(MAX_IGNITIONS, candidates.size), replace=False)[:n]
    return np.stack(np.unravel_index(chosen, ls.shape), axis=1)


def weather_scenarios(set_name: str, n: int) -> list[Weather]:
    """First ``n`` weather scenarios of the named set."""
    if set_name not in WEATHER_SETS:
        raise ValueError(f"Unknown weather set '{set_name}'. Known: {sorted(WEATHER_SETS)}")
    if not 1 <= n <= MAX_WEATHER:
        raise ValueError(f"n_weather must be in [1, {MAX_WEATHER}]")
    spec = WEATHER_SETS[set_name]
    rng = np.random.default_rng(spec.seed)
    directions = (spec.dominant_from_deg + rng.normal(0, spec.direction_spread_deg, MAX_WEATHER)) % 360
    speeds = rng.uniform(*spec.speed_kmh, MAX_WEATHER)
    dryness = rng.uniform(*spec.dryness, MAX_WEATHER)
    return [Weather(round(float(d), 1), round(float(s), 1), round(float(r), 3))
            for d, s, r in zip(directions[:n], speeds[:n], dryness[:n])]


def describe_sets() -> dict:
    """Machine-readable description for agents (exposed through ``lab_status``)."""
    return {
        "ignition_sets": {k: v["description"] for k, v in IGNITION_SETS.items() if k != "reference"},
        "weather_sets": {k: {**asdict(v), "speed_kmh": list(v.speed_kmh), "dryness": list(v.dryness)}
                         for k, v in WEATHER_SETS.items()},
        "max_ignitions": MAX_IGNITIONS,
        "max_weather": MAX_WEATHER,
    }
