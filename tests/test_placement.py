import pytest

from firelab.landscape import generate_synthetic
from firelab.placement import build_plan

LS = generate_synthetic("t", size=40, seed=1)

CASES = [
    ("random", {}),
    ("fuel_load", {}),
    ("asset_buffer", {"upwind_weight": 0.5}),
    ("wind_strips", {"spacing": 6, "width": 2}),
    ("hybrid", {"weights": {"fuel_load": 0.5, "asset_proximity": 0.5}}),
]


@pytest.mark.parametrize("strategy,params", CASES)
def test_plans_respect_budget_and_eligibility(strategy, params):
    mask, _ = build_plan(LS, strategy, 5.0, seed=1, params=params)
    target = round(0.05 * LS.n_burnable)
    assert 0 < mask.sum() <= target
    assert not (mask & ~LS.treatable).any()   # never assets or non-burnable cells


def test_no_treatment_is_empty():
    mask, _ = build_plan(LS, "none", 5.0, seed=0)
    assert mask.sum() == 0


def test_same_seed_same_plan():
    a, _ = build_plan(LS, "random", 5.0, seed=4)
    b, _ = build_plan(LS, "random", 5.0, seed=4)
    assert (a == b).all()


def test_unknown_params_are_rejected():
    with pytest.raises(ValueError):
        build_plan(LS, "wind_strips", 5.0, seed=0, params={"spcing": 3})
