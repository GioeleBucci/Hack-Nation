# Placement strategies (`firelab/placement.py`)

Every strategy follows the same recipe:

1. Compute a **score** per cell.
2. Restrict to **eligible** cells: burnable and not an asset, plus any strategy-specific mask.
3. Treat the **top-k** cells, with `k = budget_pct% × burnable cells`. Ties are broken randomly using
   the plan's seed.

Because the budget is enforced in one place, every strategy is compared at exactly the same treated
area. A strategy whose eligible set is smaller than `k` (e.g. narrow strips) returns fewer cells and
the plan is flagged `under_budget`.

`generate_plan` (the tool) wraps this. It derives a deterministic plan id from the full spec
(e.g. `S4-3fa2c1`), so the same request always returns the same plan, saves the mask to
`record/plans/<id>.npy` and records a `plan` item with the mask hash.

## Strategies

| Code | Name | Score / rule | Parameters (defaults) |
|---|---|---|---|
| S0 | `none` | No treatment (control) | — |
| SR | `random` | Random cells (control; run 3–5 seeds as replicates) | — |
| S1 | `fuel_load` | Highest fuel load | — |
| S2 | `burn_prob` | Highest no-treatment burn probability (reference map) | — |
| S3 | `asset_buffer` | Closest to assets, optionally biased upwind | `max_distance` (None, in cells), `upwind_weight` (0.0), `wind_from_deg` (225) |
| S4 | `wind_strips` | Parallel strips perpendicular to the dominant wind; within strips, ranked by a layer | `spacing` (12), `width` (2), `angle_offset_deg` (0), `priority` (`fuel_load`), `wind_from_deg` (225) |
| S5 | `path_centrality` | Cells that most often carried fire in large reference fires | — |
| S6 | `hybrid` | Weighted sum of normalized layers | `weights` ({`burn_prob`: 0.5, `asset_proximity`: 0.5}), `wind_from_deg` (225) |

Unknown parameters are rejected with the list of allowed ones, so agents get immediate feedback.

## Score layers

All layers are min-max normalized to [0, 1] over treatable cells, so hybrid weights are comparable.

| Layer | Meaning |
|---|---|
| `fuel_load` | The fuel raster |
| `burn_prob` | Reference burn probability ([simulator.md](simulator.md#reference-maps-firelabreferencepy)) |
| `centrality` | Reference spread centrality |
| `asset_proximity` | `1 / (1 + distance to the nearest asset)` |
| `upwind_of_assets` | High where an asset (dilated by 2 cells) lies within ~30 cells downwind, decaying with distance |

## Geometry notes

- Wind direction uses the meteorological convention: `wind_from_deg` is where the wind comes
  **from**, clockwise from north. 225 = south-westerly, which pushes fire to the north-east.
- Strips are lines of constant projection onto the downwind direction, so they cross the fire's
  main path. `angle_offset_deg` rotates them (90 gives strips parallel to the wind).
- Strategies with `wind_from_deg` default to the training weather's dominant wind. The transfer test
  keeps that parameter fixed while the weather changes. That is the point of the test: does a rule
  tuned to the expected wind still help when the wind differs?

## Adding a strategy

1. Add an entry to `STRATEGIES` (code, defaults, description).
2. Add its branch in `build_plan` that produces `score` (and optionally narrows `eligible`).
3. Add a case in `tests/test_placement.py`.

The hypothesis schema, `lab_status` and the benchmark read `STRATEGIES` automatically. The `enum` of
`generate_plan.strategy` in `agents/lab_director.yaml` has to be updated by hand.
