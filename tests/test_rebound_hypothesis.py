"""The hypothesis for 2008-09 added after the result (scripts/rebound_hypothesis.py)."""

import numpy as np
import pytest

from scripts import direction_diagnostic as dd
from scripts import rebound_hypothesis as rh
from scripts import run_plan


def test_the_grid_runs_in_steps_of_a_hundredth_of_a_percent_with_both_ends():
    g = rh.grid(*rh.MID)
    assert len(g) == 101 and g[0] == 0.02 and g[-1] == 0.03
    np.testing.assert_allclose(np.diff(g), 1e-4)
    h = rh.grid(rh.HIGH_FROM, 0.05537)
    assert h[0] == 0.04 and h[-1] == pytest.approx(0.0553) and len(h) == 154
    assert len(rh.grid(rh.HIGH_FROM, 0.039)) == 0


def test_the_rebound_windows_are_the_after_side_of_the_held_out_block():
    episode = np.array([-1] * 15 + [0] * 3 + [-1] * 30 + [1] * 2 + [-1] * 20)
    y = (episode >= 0).astype(int)
    idx = rh.rebound_windows(y, episode, 1)
    assert idx.tolist() == list(range(50, 62))
    lo, hi = 48 - run_plan.CONTEXT, 50 + run_plan.CONTEXT          # the block of episode 1
    _, after = dd.sides(y[lo:hi])
    assert (np.flatnonzero(after) + lo).tolist() == idx.tolist()
    y[55] = 1
    with pytest.raises(ValueError):
        rh.rebound_windows(y, episode, 1)


def test_the_reading_of_the_two_predictions():
    assert rh.reading(True, True).startswith("both predictions hold")
    assert rh.reading(False, False).startswith("both predictions fail")
    assert rh.reading(True, False).startswith("only prediction 1")
    assert rh.reading(False, True).startswith("only prediction 2")
    assert rh.reading(None, True) == "not computable"


def test_dry_run_on_synthetic_prices():
    pytest.importorskip("gtda")
    from sklearn.ensemble import RandomForestClassifier

    small = lambda s: RandomForestClassifier(n_estimators=20, min_samples_leaf=5,     # noqa: E731
                                             class_weight="balanced", random_state=s)
    res = rh.run(run_plan.synthetic_prices(), seeds=(0, 1), make_model=small)
    t = res["trainings"]
    assert t[rh.PLAN]["n_train"] - t[rh.NO_2020]["n_train"] == run_plan.CONTEXT
    # every other crisis in the training set loses its rebound: 2011, 2015, Q4 2018, 2020 and 2022
    assert t[rh.PLAN]["n_train"] - t[rh.NO_REBOUND]["n_train"] == 5 * run_plan.CONTEXT
    assert t[rh.PLAN]["n_train_crisis"] == t[rh.NO_2020]["n_train_crisis"] == t[rh.NO_REBOUND]["n_train_crisis"]
    assert len(res["2020 rebound windows"]) == run_plan.CONTEXT
    assert all(d > "2020-03-23" for d in res["2020 rebound windows"])
    for name in (rh.PLAN, rh.NO_2020, rh.NO_REBOUND):
        assert len(t[name]["per_seed"]["before"]) == 2
        assert len(t[name]["curve"]) == len(res["grid"])
        assert len(t[name]["block score"]) == len(res["block"]["crisis"])
        assert 0 <= t[name]["probability, 2-3 %"] <= 1
    assert res["prediction 1"]["grid points, 2-3 %"] == 101
    assert res["prediction 2"]["AUC before, plan"] == t[rh.PLAN]["AUC"]["before"]
    assert isinstance(res["prediction 2"]["holds"], bool)
