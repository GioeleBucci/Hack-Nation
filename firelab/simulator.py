"""Stochastic cellular-automaton fire spread (numpy).

This is the MVP stand-in for Cell2Fire (see docs/simulator.md for the model, its
assumptions and how to swap in Cell2Fire). Design points that matter for the science:

- **Stacked plans.** All plans of an experiment burn together as a (P, H, W) stack.
- **Common random numbers.** For a scenario, the random field drawn at each step depends only
  on (seed, ignition index, weather index), never on the plans. Plans are therefore compared
  under identical randomness: the difference between two plans is caused by the treatment,
  not by luck, and a plan gets the same result whichever experiment it is run in.
- **Treatment = non-burnable.** Treated cells cannot ignite (Cell2Fire "harvested cells").

Spread rule: each step, a burning cell tries to ignite each of its 8 neighbours with
probability ``1 - exp(-rate)``, where
``rate = base_rate * fuel[target] * dryness * exp(wind_coef * speed * cos(angle to downwind))
        * exp(slope_coef * rise / run) / distance``.
A cell burns for one step and is then burned out.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from firelab.landscape import Landscape
from firelab.scenarios import Weather

# (dy, dx) of fire travel: fire moves from (y - dy, x - dx) into (y, x). Row 0 is north.
DIRECTIONS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


@dataclass(frozen=True)
class SpreadParams:
    base_rate: float = 0.42
    wind_coef: float = 0.05      # per km/h; at 30 km/h downwind spread is ~4.5x faster, upwind ~0.2x
    slope_coef: float = 3.0      # per unit of tan(slope)


def downwind_vector(wind_from_deg: float) -> tuple[float, float]:
    """Unit vector (vy, vx) in grid coordinates (row down = south, col right = east)."""
    theta = np.deg2rad(wind_from_deg)
    return float(np.cos(theta)), float(-np.sin(theta))


def shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """``out[..., y, x] = a[..., y - dy, x - dx]``, zero outside the grid."""
    out = np.zeros_like(a)
    h, w = a.shape[-2:]
    out[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        a[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def spread_probabilities(ls: Landscape, weather: Weather, params: SpreadParams = SpreadParams()) -> np.ndarray:
    """(8, H, W) probability that fire in the neighbour behind direction d ignites each cell."""
    vy, vx = downwind_vector(weather.wind_from_deg)
    probs = np.zeros((len(DIRECTIONS), *ls.shape), dtype=np.float32)
    for d, (dy, dx) in enumerate(DIRECTIONS):
        dist = float(np.hypot(dy, dx))
        alignment = (dy * vy + dx * vx) / dist
        wind = np.exp(params.wind_coef * weather.wind_speed_kmh * alignment)
        rise = ls.elevation - shift(ls.elevation, dy, dx)          # target minus source
        slope = np.exp(params.slope_coef * np.clip(rise / (dist * ls.cell_size_m), -1, 1))
        rate = params.base_rate * ls.fuel * weather.dryness * wind * slope / dist
        probs[d] = 1.0 - np.exp(-rate)
    return probs


@dataclass
class EnsembleOutput:
    burned_ha: np.ndarray       # (P, S) burned area per plan and scenario
    asset_cells: np.ndarray     # (P, S) burned asset cells per plan and scenario
    burn_count: np.ndarray      # (P, H, W) number of scenarios in which each cell burned
    centrality: np.ndarray | None = None   # (P, H, W) only when track_spread=True
    scenarios: list[tuple[int, int]] | None = None   # (ignition index, weather index) per column


def simulate_scenario(ls: Landscape, treated: np.ndarray, ignition: tuple[int, int], probs: np.ndarray,
                      rng: np.random.Generator, track_spread: bool = False
                      ) -> tuple[np.ndarray, np.ndarray | None]:
    """Burn one scenario for a stack of plans.

    Returns the (P, H, W) burned mask and, if ``track_spread``, the (P, H, W) mask of cells
    that passed fire on to at least one neighbour.
    """
    unburned = np.broadcast_to(ls.burnable, treated.shape) & ~treated
    burning = np.zeros_like(unburned)
    iy, ix = int(ignition[0]), int(ignition[1])
    burning[:, iy, ix] = unburned[:, iy, ix]   # an ignition on a treated cell goes nowhere
    unburned[:, iy, ix] = False
    burned = np.zeros_like(unburned)
    spreaders = np.zeros_like(unburned) if track_spread else None

    max_steps = 3 * sum(ls.shape)
    for _ in range(max_steps):
        if not burning.any():
            break
        u = rng.random(probs.shape, dtype=np.float32)   # shared by all plans: common random numbers
        hits = u < probs
        new = np.zeros_like(burning)
        for d, (dy, dx) in enumerate(DIRECTIONS):
            ignited = shift(burning, dy, dx) & hits[d] & unburned
            new |= ignited
            if track_spread:
                spreaders |= shift(ignited, -dy, -dx)
        burned |= burning
        unburned &= ~new
        burning = new
    burned |= burning   # only non-empty if max_steps was hit
    return burned, spreaders


def scenario_rng(seed: int, ignition_index: int, weather_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, ignition_index, weather_index]))


def run_ensemble(ls: Landscape, treated: np.ndarray, ignitions: np.ndarray, weathers: list[Weather],
                 seed: int, params: SpreadParams = SpreadParams(), track_spread: bool = False) -> EnsembleOutput:
    """Burn every plan under every (ignition, weather) scenario."""
    treated = np.asarray(treated, dtype=bool)
    if treated.ndim == 2:
        treated = treated[None]
    n_plans = treated.shape[0]
    n_scen = len(ignitions) * len(weathers)
    burned_ha = np.zeros((n_plans, n_scen), dtype=np.float64)
    asset_cells = np.zeros((n_plans, n_scen), dtype=np.int32)
    burn_count = np.zeros((n_plans, *ls.shape), dtype=np.int32)
    centrality = np.zeros((n_plans, *ls.shape), dtype=np.float64) if track_spread else None
    scenarios = []

    s = 0
    for w_idx, weather in enumerate(weathers):
        probs = spread_probabilities(ls, weather, params)
        for i_idx, ignition in enumerate(ignitions):
            rng = scenario_rng(seed, i_idx, w_idx)
            burned, spreaders = simulate_scenario(ls, treated, ignition, probs, rng, track_spread)
            cells = burned.sum(axis=(1, 2))
            burned_ha[:, s] = cells * ls.cell_ha
            asset_cells[:, s] = (burned & ls.assets).sum(axis=(1, 2))
            burn_count += burned
            if track_spread:
                # Cells that carried fire in big fires matter most: weight by fire size.
                centrality += spreaders * cells[:, None, None]
            scenarios.append((i_idx, w_idx))
            s += 1

    if track_spread:
        centrality /= max(n_scen, 1)
    return EnsembleOutput(burned_ha, asset_cells, burn_count, centrality, scenarios)
