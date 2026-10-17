"""Tests for the leakage-free splits, the labels and the baseline features.
None of them needs giotto-tda or network access."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.classify import make_logistic
from src.data import crisis_labels
from src.features import leading_eigenvalue_share, realized_volatility
from src.persistence import sliding_windows, standardize_windows
from src.validation import evaluate, leave_one_crisis_out, purge_gap, purged_walk_forward

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
    dates = pd.to_datetime(["2008-09-12", "2008-09-15", "2009-03-09", "2009-03-10", "2020-03-01"])
    y, ep = crisis_labels(dates)
    assert y.tolist() == [0, 1, 1, 0, 1]
    assert ep.tolist() == [-1, 0, 0, -1, 3]


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
