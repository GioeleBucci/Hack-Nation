import numpy as np

from firelab.landscape import generate_synthetic
from firelab.scenarios import ignition_points, weather_scenarios
from firelab.simulator import run_ensemble, shift

LS = generate_synthetic("t", size=40, seed=1)


def test_shift_moves_values_and_zero_fills():
    a = np.arange(9).reshape(3, 3)
    out = shift(a, 1, 0)
    assert out[0].tolist() == [0, 0, 0]
    assert out[1].tolist() == a[0].tolist()
    assert shift(a, 0, -1)[:, 2].tolist() == [0, 0, 0]


def test_treated_cells_never_burn():
    treated = np.zeros(LS.shape, dtype=bool)
    treated[::3, :] = LS.treatable[::3, :]
    out = run_ensemble(LS, treated, ignition_points(LS, "train", 10), weather_scenarios("train", 2), seed=0)
    assert out.burn_count[0][treated].sum() == 0


def test_ignition_on_treated_cell_burns_nothing():
    ign = ignition_points(LS, "train", 1)
    treated = np.zeros(LS.shape, dtype=bool)
    treated[tuple(ign[0])] = True
    out = run_ensemble(LS, treated, ign, weather_scenarios("train", 1), seed=0)
    assert out.burned_ha[0, 0] == 0


def test_deterministic_and_independent_of_stack_companions():
    ign, wx = ignition_points(LS, "train", 5), weather_scenarios("train", 2)
    rng = np.random.default_rng(0)
    plan_a = np.zeros(LS.shape, dtype=bool)
    plan_b = LS.treatable & (rng.random(LS.shape) < 0.1)
    alone = run_ensemble(LS, plan_a, ign, wx, seed=3)
    stacked = run_ensemble(LS, np.stack([plan_b, plan_a]), ign, wx, seed=3)
    assert np.array_equal(alone.burned_ha[0], stacked.burned_ha[1])   # common random numbers
    again = run_ensemble(LS, plan_a, ign, wx, seed=3)
    assert np.array_equal(alone.burned_ha, again.burned_ha)


def test_scenario_sets_are_nested_prefixes():
    assert np.array_equal(ignition_points(LS, "train", 10)[:5], ignition_points(LS, "train", 5))
    assert weather_scenarios("heldout_nw", 5)[:2] == weather_scenarios("heldout_nw", 2)
