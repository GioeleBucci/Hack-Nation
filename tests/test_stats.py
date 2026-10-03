import numpy as np
import pytest

from firelab.stats import compare_plans, holm


def test_holm_matches_hand_computation():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_clear_improvement_wins_and_null_does_not():
    rng = np.random.default_rng(0)
    base = rng.uniform(50, 150, 200)
    values = {"rand": base, "good": base * 0.7, "same": base.copy()}
    res = compare_plans(values, ["rand"])
    rows = {r["plan_id"]: r for r in res["comparisons"]}
    assert rows["good"]["verdict"] == "wins"
    assert rows["good"]["rel_change_pct"] == pytest.approx(-30)
    assert rows["same"]["verdict"] == "no_clear_difference"
