"""The direction diagnostic added after the result (scripts/direction_diagnostic.py)."""

import numpy as np
import pytest

from scripts import direction_diagnostic as dd
from scripts import run_plan
from src.features import correlation_spectrum, mardia_kurtosis, realized_volatility
from src.persistence import standardize_windows


def _distances(windows):
    return np.linalg.norm(windows[:, :, None, :] - windows[:, None, :, :], axis=-1)


def test_the_plan_features_ignore_the_sign_and_the_order_of_the_days_but_the_mean_return_does_not():
    rng = np.random.default_rng(0)
    w = 0.01 * rng.standard_t(4, size=(3, 40, 7)) + 0.001
    perm = rng.permutation(40)
    for f in (realized_volatility, correlation_spectrum, mardia_kurtosis):
        np.testing.assert_allclose(f(-w), f(w), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(f(w[:, perm]), f(w), rtol=1e-10, atol=1e-12)
    # Vietoris-Rips persistence uses only the distances between the days of a standardised window
    d = _distances(standardize_windows(w))
    np.testing.assert_allclose(_distances(standardize_windows(-w)), d, atol=1e-12)
    np.testing.assert_allclose(_distances(standardize_windows(w[:, perm])), d[:, perm][:, :, perm],
                               atol=1e-12)
    np.testing.assert_allclose(dd.mean_return(-w), -dd.mean_return(w))


def test_the_normal_windows_of_a_block_split_into_before_and_after():
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0])
    before, after = dd.sides(y)
    assert before.tolist() == [True, True, False, False, False, False, False, False]
    assert after.tolist() == [False] * 5 + [True] * 3
    score = np.array([0.1, 0.2, 0.9, 0.8, 0.7, 0.95, 0.85, 0.75])
    a = dd.side_aucs(score, y)
    assert a["before"] == pytest.approx(1.0)
    assert a["after"] == pytest.approx(3 / 9)
    # the AUC on the whole block is the mean of the two sides, weighted by their windows
    assert a["all"] == pytest.approx((2 * a["before"] + 3 * a["after"]) / 5)


def test_the_prediction_needs_all_three_episodes_and_includes_its_bounds():
    table = {ep: {s: {"all": 0.65, "before": 0.9, "after": 0.4} for s in dd.SCORES}
             for ep in dd.PREDICTED}
    p = dd.prediction(table)
    assert p["volatility, forest"]["holds"] and p["volatility, raw"]["holds"]
    table["covid crash"]["volatility, raw"] = {"all": 0.705, "before": 0.9, "after": 0.51}
    p = dd.prediction(table)
    assert p["volatility, forest"]["holds"] and not p["volatility, raw"]["holds"]
    table["covid crash"]["volatility, raw"] = {"all": 0.65, "before": 0.8, "after": 0.5}
    assert dd.prediction(table)["volatility, raw"]["holds"]


def test_dry_run_on_synthetic_prices():
    pytest.importorskip("gtda")
    from sklearn.ensemble import RandomForestClassifier

    small = lambda s: RandomForestClassifier(n_estimators=20, min_samples_leaf=5,     # noqa: E731
                                             class_weight="balanced", random_state=s)
    res = dd.run(run_plan.synthetic_prices(), seeds=(0, 1), make_model=small)
    loco, wf = res["schemes"]["leave one crisis out"], res["schemes"]["walk-forward"]
    assert loco["scorable counted episodes"] == 5 and wf["scorable counted episodes"] == 4
    assert not wf["episodes"]["global financial crisis"]["scorable"]    # nothing to learn from before it
    assert set(loco["prediction"]) == set(dd.SCORES) and "prediction" not in wf
    for r in loco["episodes"].values():
        assert r["n_before"] == r["n_after"] == run_plan.CONTEXT
        raw = r["volatility, raw"]
        assert raw["all"] == pytest.approx((raw["before"] + raw["after"]) / 2)
        assert len(r["volatility, forest"]["per_seed"]["after"]) == 2
        assert set(r["AUC, forest"]) == {"volatility", dd.BASELINE, dd.COMBINED, dd.WITH_MEAN,
                                         dd.COMBINED_MEAN}
