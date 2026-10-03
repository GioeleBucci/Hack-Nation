# Landscape, scenarios and the fire-spread simulator

## Landscape (`firelab/landscape.py`)

A landscape is three aligned rasters and a cell size:

| Raster | Meaning |
|---|---|
| `fuel` | Fuel load in [0, 1]. 0 = non-burnable (water, rock). |
| `elevation` | Metres; drives the slope effect. |
| `assets` | Cells to protect (houses). They can burn but are never treated. |

`synthetic_v1` (default) is 120 × 120 cells of 100 m (1 ha each, 14,400 ha), generated
deterministically from a seed:

- smooth hills with a ridge rising to the north-east, and bare rock on the highest ground;
- patchy fuel at two spatial scales;
- a creek running south from the northern edge, ending three-quarters of the way down, so fire can
  go around it;
- two settlements in the east, downwind of the default south-westerly fire weather.

Row 0 is north and column 0 is west. Each landscape has a **fingerprint** (hash of its arrays); plans
and experiments store it, and a run refuses plans built on a different landscape.

`synthetic_small` (60 × 60) is for tests. To use real data, save
`data/landscape_<name>.npz` with arrays `fuel`, `elevation`, `assets` and scalar `cell_size_m`
(e.g. from LANDFIRE fuel models mapped to a [0, 1] load) and set `FIRELAB_LANDSCAPE=<name>`.

## Scenario sets (`firelab/scenarios.py`)

A **scenario** is one ignition point × one weather condition. Sets are frozen: generated from a fixed
seed, and asking for `n` items always returns the first `n` of the same list. A 50-ignition screen
is therefore a strict subset of a 200-ignition confirmation.

| Set | Kind | Content | Use |
|---|---|---|---|
| `train` | ignitions | uniform over treatable cells | screening, refinement |
| `heldout` | ignitions | independent uniform draw | transfer test |
| `reference` | ignitions | independent draw | only for the reference maps |
| `train` | weather | wind from 225° ± 20°, 15–35 km/h, dryness 0.85–1.15 | screening, refinement |
| `heldout_nw` | weather | wind from 315° ± 20°, same severity | transfer test (wind rotated 90°) |
| `heldout_extreme` | weather | wind from 225° ± 25°, 35–55 km/h, dryness 1.1–1.35 | stress test |

Weather is synthetic. `burn_window_days` uses real Open-Meteo (ERA5) data, but only for burn-window
feasibility, not for spread. Weighting ignitions by NASA FIRMS detections and deriving weather sets
from ERA5 fire-season percentiles are natural next steps.

## Spread model (`firelab/simulator.py`)

A stochastic cellular automaton on the 8-neighbour grid. At each step, every burning cell tries to
ignite each unburned neighbour with probability

```
p = 1 − exp(−rate)
rate = base_rate · fuel[target] · dryness
       · exp(wind_coef · wind_speed · cos(angle between spread direction and downwind))
       · exp(slope_coef · rise / run)
       / distance            (1 for edge neighbours, √2 for diagonals)
```

A cell burns for one step and is then burned out. The fire stops when nothing is burning. Defaults
(`SpreadParams`): `base_rate = 0.42`, `wind_coef = 0.05` per km/h (at 30 km/h, spread is about 4.5×
faster downwind and 0.2× upwind), `slope_coef = 3.0`.

**Treatment** is modelled as making cells non-burnable, the same idealization Cell2Fire uses with
"harvested cells". An ignition that lands on a treated cell does not spread.

### Common random numbers

All plans in an experiment are burned as one `(plans, H, W)` stack. For each scenario, the random
field drawn at each step depends only on `(seed, ignition index, weather index)`. Consequences:

- Differences between plans come from the treatment, not from luck. This greatly reduces the
  variance of paired differences, so fewer simulations are needed for the same CI width.
- A plan gets **exactly the same result** whatever other plans share its stack, and whichever
  experiment it appears in, as long as the scenario set and seed are the same. This is tested in
  `tests/test_simulator.py`.

### Outputs per experiment

Saved to `record/runs/<experiment_id>.npz`: per plan and scenario the burned area (ha) and burned
asset cells, plus per-plan burn-count maps (burn probability = count / scenarios) used for figures.

### Reference maps (`firelab/reference.py`)

The `burn_prob` and `path_centrality` strategies need a no-treatment picture of the landscape. It is
built once (`firelab init`) from 100 `reference` ignitions × 5 `train` weathers and cached in
`data/`:

- **burn probability**: share of reference fires that burned the cell;
- **spread centrality**: how often the cell passed fire to a neighbour, weighted by the size of that
  fire, normalized to [0, 1].

The reference uses its own ignition set, so these strategies are not fitted to the exact fires they
are scored on. The reference is treated as frozen input data (like an existing burn-probability
product) and is not charged to the lab's simulation budget.

## Calibration check

`firelab baseline` prints the mean no-treatment burned area as a share of the burnable area. Aim for
roughly 5–30%. If fires are tiny, placement cannot matter; if every fire burns the whole map, it
cannot either. Adjust `SpreadParams.base_rate` first (`firelab/simulator.py`), re-run
`firelab init` (delete `data/reference_*.npz` first) and start a fresh record.

## Assumptions and limits (state them in the demo)

- No spotting, crown fire, fuel moisture dynamics, diurnal weather or suppression.
- Constant weather during each fire.
- Treatment fully removes fuel, and the effect does not decay over time.
- One-step burn duration; spread rates are relative, not calibrated to m/min.
- Synthetic landscape and weather; results hold for this landscape only.

## Swapping in Cell2Fire

The rest of the lab only needs `run_ensemble(ls, treated, ignitions, weathers, seed) → EnsembleOutput`
with per-plan, per-scenario burned area and asset cells. To use Cell2Fire:

1. Export the landscape to a Cell2Fire instance folder (fuel grid, elevation, ignition points,
   weather files per scenario).
2. Pass treated cells as Cell2Fire's harvested-cells file, one run per plan.
3. Parse the final burned grids into the same `EnsembleOutput` arrays, and set
   `SIMULATOR_VERSION` in `firelab/config.py`.

Cell2Fire uses its own random streams, so the common-random-numbers property is lost. Pairing by
scenario still holds; expect wider CIs at the same ensemble size.
