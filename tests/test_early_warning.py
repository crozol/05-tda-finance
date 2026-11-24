"""The early-warning study: episodes, labels, held-out blocks, synthetic markets and the closed-form
landscape norms."""

import numpy as np
import pandas as pd

from src.early_warning import (drawdown_episodes, leave_one_episode_out, nearest_peak,
                               past_only_episode_splits, regime_rows, synthetic_indices, window_labels)
from src.features import total_persistence_norms
from src.persistence import sliding_windows
from src.validation import purge_gap


def _path(*legs):
    """A price path through the given (n_days, total return) legs, starting at 100."""
    x = [100.0]
    for n, r in legs:
        x += list(x[-1] * (1 + r) ** (np.arange(1, n + 1) / n))
    return np.array(x)


def test_episodes_need_a_52_week_high_and_a_15_percent_fall():
    x = _path((300, 0.5), (40, -0.20), (60, 0.12), (30, -0.15), (200, 0.6), (20, -0.08), (100, 0.1))
    ep = drawdown_episodes(x)
    assert len(ep) == 1                                # the 8 % dip does not count
    peak, trough = ep[0]
    assert peak == 300 and x[trough] == x[300:].min()  # the rally inside the bear market starts nothing new
    assert trough == 300 + 40 + 60 + 30


def test_two_separate_drawdowns_are_two_episodes():
    x = _path((300, 0.4), (30, -0.2), (300, 0.5), (30, -0.25), (100, 0.3))
    assert [p for p, _ in drawdown_episodes(x)] == [300, 630]


def test_labels_positive_before_the_peak_negative_far_from_any_boundary():
    end = np.arange(0, 2000)
    ep = [(1000, 1100)]
    y = window_labels(end, ep, horizon=60, far=120)
    assert set(np.flatnonzero(y == 1)) == set(range(940, 1000))
    neg = np.flatnonzero(y == 0)
    assert np.all((np.abs(neg - 1000) >= 120) & (np.abs(neg - 1100) >= 120))
    assert y[880] == 0 and y[881] == -1 and y[1219] == -1 and y[1220] == 0
    assert np.all(y[1000:1101] == -1)                  # the drawdown itself is left out


def test_held_out_blocks_share_no_day_with_training():
    _, end = sliding_windows(np.zeros((3000, 2)), 50, 5)
    ep = [(800, 900), (1700, 1750), (2500, 2600)]
    y = window_labels(end, ep)
    block = nearest_peak(end, ep)
    gap = purge_gap(50, 5)
    days = lambda idx: {d for i in idx for d in range(end[i] - 49, end[i] + 1)}      # noqa: E731
    splits = list(leave_one_episode_out(block, y, gap))
    assert len(splits) == 3
    for e, (train, test) in enumerate(splits):
        assert np.all(y[test] >= 0) and np.all(y[train] >= 0) and set(y[test]) == {0, 1}
        assert np.all(block[test] == e)
        assert not days(train) & days(test)
    first_train, _ = next(past_only_episode_splits(block, y, gap))
    assert len(first_train) == 0 or set(y[first_train]) == {0}      # nothing earlier to learn from


def test_synthetic_regimes_and_scenarios():
    dates = pd.bdate_range("2000-01-03", periods=1500)
    ep = [(600, 650), (1200, 1230)]
    reg = regime_rows(len(dates) - 1, ep, window=50)
    assert np.all(reg[600 - 109:600] == "pre-peak") and reg[600 - 110] == "normal"
    assert np.all(reg[600:651] == "crisis") and reg[651] == "rebound"
    pre = reg == "pre-peak"
    out = {}
    for sc in ("null", "heavy-tailed", "tails-only"):
        r = np.diff(np.log(synthetic_indices(1, sc, dates, ep).to_numpy()), axis=0)
        out[sc] = r
        assert r.shape == (len(dates) - 1, 4)
    # null and heavy-tailed share the Gaussian draws outside the pre-peak regime
    np.testing.assert_allclose(out["null"][~pre], out["heavy-tailed"][~pre])
    # tails-only: the normal variance with heavier tails, so a narrower bulk (unit-variance t3: IQR 0.88
    # against 1.35 for a Gaussian) and a larger kurtosis than the null's pre-peak days
    normal = reg == "normal"
    iqr = lambda r: np.subtract(*np.percentile(r, [75, 25]))                       # noqa: E731
    assert iqr(out["tails-only"][pre]) < 0.85 * iqr(out["null"][normal])
    kurt = lambda r: ((r - r.mean()) ** 4).mean() / r.var() ** 2               # noqa: E731
    assert kurt(out["tails-only"][pre][:, 0]) > kurt(out["null"][pre][:, 0]) + 1


def _tent_integral(diagram, p, n=200_001):
    """sum_k ∫ lambda_k^p on a fine grid, for the points of one homology dimension."""
    lo, hi = diagram[:, 0].min(), diagram[:, 1].max()
    x = np.linspace(lo, hi, n)
    t = np.maximum(0.0, np.minimum(x[None, :] - diagram[:, :1], diagram[:, 1:2] - x[None, :]))
    layers = -np.sort(-t, axis=0)
    return np.trapz((layers ** p).sum(axis=0), x)


def test_closed_form_norms_match_the_landscape_integral():
    rng = np.random.default_rng(0)
    b = rng.uniform(0, 1, 7)
    pts = np.column_stack([b, b + rng.uniform(0.05, 0.8, 7)])
    diagrams = np.concatenate([np.column_stack([pts, np.ones(7)]),
                               [[0.3, 0.3, 1.0]]])[None]                     # one diagonal padding point
    out = total_persistence_norms(diagrams, dims=(1,), orders=(1, 2, 3))
    for j, p in enumerate((1, 2, 3)):
        assert out[0, j] == np.float64(out[0, j])
        np.testing.assert_allclose(out[0, j], _tent_integral(pts, p) ** (1 / p), rtol=1e-6)


def test_closed_form_matches_the_nested_loops_example():
    d = np.array([[[0.0, 1.0, 0.0], [1.0, 3.0, 1.0], [1.5, 2.5, 1.0]]])
    out = total_persistence_norms(d)                    # columns: L1 (H0, H1), L2 (H0, H1)
    np.testing.assert_allclose(out[0, 1], 1.25)
    np.testing.assert_allclose(out[0, 3], np.sqrt(0.75))
