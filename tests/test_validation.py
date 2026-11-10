"""Tests for the leakage-free splits, the labels and the baseline features.
None of them needs giotto-tda or network access."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.classify import make_logistic
from src.data import CRISIS_PERIODS, crisis_labels
from src.features import leading_eigenvalue_share, realized_volatility
from src.persistence import sliding_windows, standardize_windows
from src.validation import (calibrated_threshold, calibration_split, evaluate, first_alarm,
                            leave_one_crisis_out, purge_gap, purged_walk_forward)

WINDOW, STEP = 40, 5


def _days(idx, end):
    """Set of return rows used by the windows ``idx``."""
    return {d for i in idx for d in range(end[i] - WINDOW + 1, end[i] + 1)}


def test_purge_gap_removes_all_shared_days():
    _, end = sliding_windows(np.zeros((400, 3)), WINDOW, STEP)
    gap = purge_gap(WINDOW, STEP)
    for train, test in purged_walk_forward(len(end), n_splits=4, gap=gap):
        assert train.max() < test.min()
        assert not _days(train, end) & _days(test, end)
    # one window less of purge would leak
    train, test = next(purged_walk_forward(len(end), n_splits=4, gap=gap - 1))
    assert _days(train, end) & _days(test, end)


def test_leave_one_crisis_out_holds_out_each_episode_without_shared_days():
    _, end = sliding_windows(np.zeros((1000, 3)), WINDOW, STEP)
    episode = np.full(len(end), -1)
    episode[40:50] = 0
    episode[120:126] = 1
    gap = purge_gap(WINDOW, STEP)
    splits = list(leave_one_crisis_out(episode, gap=gap, context=20))
    assert len(splits) == 2
    for (train, test), e in zip(splits, [0, 1]):
        assert set(np.where(episode == e)[0]) <= set(test)
        assert not set(train) & set(test)
        assert not _days(train, end) & _days(test, end)
        assert set(episode[test]) >= {-1, e}          # both classes in the test block


def test_crisis_labels_use_inclusive_periods():
    dates = pd.to_datetime(["2008-09-12", "2008-09-15", "2009-03-09", "2009-03-10",
                            "2018-12-24", "2018-12-26", "2020-03-01"])
    y, ep = crisis_labels(dates)
    k = list(CRISIS_PERIODS).index
    assert y.tolist() == [0, 1, 1, 0, 1, 0, 1]
    assert ep.tolist() == [-1, 0, 0, -1, k("q4 2018 selloff"), -1, k("covid crash")]


def test_baselines_detect_volatility_and_comovement():
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.01, (1, 200, 5))
    common = rng.normal(0, 0.03, (200, 1))
    stressed = (common + rng.normal(0, 0.005, (200, 5)))[None]
    w = np.concatenate([calm, stressed])
    vol = realized_volatility(w)[:, 0]
    lam = leading_eigenvalue_share(w)[:, 0]
    assert vol[1] > 2 * vol[0]
    assert lam[0] < 0.4 < 0.9 < lam[1] <= 1.0


def test_standardized_windows_remove_scale():
    rng = np.random.default_rng(1)
    w = rng.normal(0, 0.05, (3, 40, 4))
    z = standardize_windows(w)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-12)
    assert np.allclose(z.std(axis=1), 1, atol=1e-6)


def test_shuffled_kfold_overstates_skill_on_overlapping_windows():
    """A feature unrelated to the label, but slowly varying in time (the mean
    of a three-dimensional random walk over each window), scores well under shuffled k-fold —
    neighbouring windows in time are neighbours in feature space and share
    the label — and near chance under purged walk-forward."""
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(2)
    T = 3000
    walk = np.cumsum(rng.normal(size=(T, 3)), axis=0)
    regime = (np.sin(np.arange(T) / 150.0) > 0.6).astype(int)        # slow label, unrelated to the walk
    windows, end = sliding_windows(walk, WINDOW, STEP)
    X = windows.mean(axis=1)
    y = regime[end]
    kfold = list(KFold(5, shuffle=True, random_state=0).split(X))
    from sklearn.neighbors import KNeighborsClassifier
    knn = lambda: KNeighborsClassifier(1)                               # noqa: E731
    leaky = evaluate(knn, X, y, kfold)["mean"]["roc_auc"]
    purged = evaluate(knn, X, y, list(purged_walk_forward(len(y), 5, purge_gap(WINDOW, STEP))))
    assert leaky > 0.9
    assert purged["mean"]["roc_auc"] < 0.65
    assert evaluate(make_logistic, X, y, kfold)["n_scored"] == 5


def test_calibration_block_is_purged_from_the_fitting_part():
    """The calibration block shares no day with the windows the model is fitted
    on, also inside a leave-one-crisis-out training fold with a hole in it."""
    _, end = sliding_windows(np.zeros((1000, 3)), WINDOW, STEP)
    episode = np.full(len(end), -1)
    episode[60:70] = 0
    episode[150:156] = 1
    gap = purge_gap(WINDOW, STEP)
    for train, _ in leave_one_crisis_out(episode, gap=gap, context=20):
        fit, cal = calibration_split(train, gap)
        assert len(cal) == round(0.2 * len(train)) and set(fit) | set(cal) <= set(train)
        assert not _days(fit, end) & _days(cal, end)
    fit, cal = calibration_split(np.arange(100), gap - 1)       # one window less leaks
    assert _days(fit, end) & _days(cal, end)


def test_calibrated_threshold_is_out_of_sample():
    """A random forest scores its own training windows near 0, so a threshold
    from them would be degenerate; on the purged calibration block it is not."""
    from src.classify import make_random_forest
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.15).astype(int)
    X = rng.normal(size=(n, 3)) + 0.8 * y[:, None]
    train = np.arange(n)
    model, thr = calibrated_threshold(lambda: make_random_forest(0), X, y, train, gap=7)
    fit, _ = calibration_split(train, 7)
    in_sample = np.percentile(model.predict_proba(X[fit][y[fit] == 0])[:, 1], 95)
    assert thr > in_sample


def test_frozen_prices_round_trip_and_hash_check(tmp_path):
    from src.data import freeze_prices, load_frozen_prices
    idx = pd.bdate_range("2020-01-01", periods=5)
    prices = pd.DataFrame({"SPY": [300.1, 301.25, 299.9, 302.0, 303.5],
                           "XLF": [30.01, 30.2, 29.95, 30.4, 30.6]}, index=idx)
    f = tmp_path / "p.csv"
    digest = freeze_prices(prices, f)
    back = load_frozen_prices(f, digest)
    np.testing.assert_allclose(back.to_numpy(), prices.to_numpy())
    try:
        load_frozen_prices(f, "0" * 64)
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong hash must be refused")


def test_first_alarm_is_raised_on_the_third_window_of_the_run():
    """No look-ahead: the alarm is the window that completes the run, and a
    run must start at or after the first window searched."""
    s = np.array([1, 1, 0, 1, 1, 1, 1], dtype=float)
    assert first_alarm(s, 0.5, k=3) == 5
    assert first_alarm(s, 0.5, start=4, k=3) == 6
    assert first_alarm(np.array([1, 1, 0, 1, 1]), 0.5, k=3) is None
