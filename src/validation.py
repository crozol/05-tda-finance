"""Leakage-free evaluation for classifiers on sliding-window features.

Consecutive windows overlap: with a 40-day window and a 5-day step, two
neighbouring windows share 35 days of returns. A shuffled or plain k-fold
split therefore puts near-copies of every test window in the training set,
and the reported AUC measures memorisation, not detection (López de Prado,
2018, ch. 7). Two split schemes avoid this:

* **Purged walk-forward.** Train on the past, test on the next block; the
  ``gap`` windows between the two are dropped, so no training window shares
  a single day with a test window. Training only ever looks backwards, as a
  real-time detector would.
* **Leave one crisis out.** Each labelled crisis episode is held out in turn,
  with the normal-regime windows around it, and the model is trained on
  everything else minus a purge gap on both sides. This asks whether a
  detector fitted on other crises recognises a new one, which is the
  question that matters with only a handful of episodes.

``evaluate`` scores any feature matrix under a given split, so topological
features and the baselines (realised volatility, the leading eigenvalue of
the correlation matrix) are compared on identical folds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


def purge_gap(window: int, step: int) -> int:
    """Number of windows to drop between train and test so that no training
    window shares a day with a test window: ceil(window / step) - 1 windows
    overlap the first test window, so that many are purged."""
    return int(np.ceil(window / step)) - 1


def purged_walk_forward(n: int, n_splits: int, gap: int, min_train: int | None = None) -> Iterator[Split]:
    """Expanding-window splits: test blocks of equal size at the end of the
    series, training on everything before ``test_start - gap``."""
    if min_train is None:
        min_train = n // (n_splits + 1)
    test_size = (n - min_train) // n_splits
    for k in range(n_splits):
        test_start = min_train + k * test_size
        test_end = n if k == n_splits - 1 else test_start + test_size
        train_end = test_start - gap
        if train_end <= 0:
            continue
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def leave_one_crisis_out(episode: np.ndarray, gap: int, context: int) -> Iterator[Split]:
    """One split per crisis episode.

    ``episode`` holds, for every window, the integer id of the crisis it
    belongs to, or -1 for normal-regime windows. The test set is the episode
    plus ``context`` windows on each side (so it contains both classes); the
    training set is every other window except ``gap`` windows on each side of
    the test block.
    """
    episode = np.asarray(episode)
    n = len(episode)
    idx = np.arange(n)
    for e in sorted(set(episode[episode >= 0].tolist())):
        members = idx[episode == e]
        lo = max(0, members.min() - context)
        hi = min(n, members.max() + context + 1)
        test = np.arange(lo, hi)
        keep = (idx < lo - gap) | (idx >= hi + gap)
        yield idx[keep], test


def calibration_split(train: np.ndarray, gap: int, frac: float = 0.2) -> Split:
    """Split a training fold into a fitting part and a calibration block.

    The calibration block is the last ``frac`` of the fold's windows; the
    ``gap`` windows before it are dropped, so no fitting window shares a day
    with a calibration window. A threshold set on the block is then out of
    sample for a model fitted on the rest. Thresholds taken from the fold's own
    training windows are degenerate for a random forest, which scores them near
    0, and out-of-bag scores do not help: the bootstrap ignores time, so every
    out-of-bag window has in-bag neighbours that share most of its days."""
    train = np.sort(np.asarray(train))
    n_cal = max(1, int(round(frac * len(train))))
    cal = train[-n_cal:]
    return train[train < cal[0] - gap], cal


def calibrated_threshold(make_model: Callable[[], object], X: np.ndarray, y: np.ndarray,
                         train: np.ndarray, gap: int, q: float = 95.0, frac: float = 0.2):
    """(model, threshold): the model fitted on the fitting part of
    ``calibration_split`` and the ``q``-th percentile of its score on the
    normal windows of the calibration block. The same model must score the
    held-out windows. None if the fitting part has a single class or the
    block has no normal window."""
    fit, cal = calibration_split(train, gap, frac)
    X = np.asarray(X).reshape(len(y), -1)
    normal = cal[y[cal] == 0]
    if len(set(y[fit])) < 2 or len(normal) == 0:
        return None
    model = make_model()
    model.fit(X[fit], y[fit])
    return model, float(np.percentile(model.predict_proba(X[normal])[:, 1], q))


def evaluate(make_model: Callable[[], object], X: np.ndarray, y: np.ndarray,
             splits: list[Split]) -> dict:
    """ROC-AUC and average precision per fold. Folds whose test or training
    set contains a single class cannot be scored and are reported as skipped."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    X = np.asarray(X).reshape(len(y), -1)
    folds = []
    for train, test in splits:
        if len(set(y[train])) < 2 or len(set(y[test])) < 2:
            folds.append({"n_test": len(test), "skipped": True})
            continue
        model = make_model()
        model.fit(X[train], y[train])
        score = model.predict_proba(X[test])[:, 1]
        folds.append({"n_test": len(test), "skipped": False,
                      "roc_auc": float(roc_auc_score(y[test], score)),
                      "average_precision": float(average_precision_score(y[test], score))})
    scored = [f for f in folds if not f["skipped"]]
    summary = {k: float(np.mean([f[k] for f in scored])) if scored else float("nan")
               for k in ("roc_auc", "average_precision")}
    return {"folds": folds, "mean": summary, "n_scored": len(scored)}
