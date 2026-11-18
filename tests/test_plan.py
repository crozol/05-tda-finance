"""The pieces of scripts/run_plan.py that decide the result: the walk-forward
splits, the seed average, the win rule and the lead time. The end-to-end dry
run on synthetic prices needs giotto-tda and is skipped without it."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import run_plan
from src.classify import SeedAverage
from src.persistence import sliding_windows
from src.validation import leave_one_crisis_out, past_only_crisis_splits, purge_gap

WINDOW, STEP = 40, 5


def _days(idx, end):
    return {d for i in idx for d in range(end[i] - WINDOW + 1, end[i] + 1)}


def test_past_only_splits_train_on_the_past_with_the_same_test_blocks():
    _, end = sliding_windows(np.zeros((1500, 3)), WINDOW, STEP)
    episode = np.full(len(end), -1)
    episode[60:70], episode[150:160], episode[250:255] = 0, 1, 2
    gap = purge_gap(WINDOW, STEP)
    loco = list(leave_one_crisis_out(episode, gap, context=12))
    past = list(past_only_crisis_splits(episode, gap, context=12))
    assert len(past) == 3
    for (_, t_loco), (train, test) in zip(loco, past):
        assert np.array_equal(t_loco, test)
        assert train.max() < test.min() - gap
        assert not _days(train, end) & _days(test, end)
    assert set(episode[past[0][0]]) == {-1}           # the first crisis has nothing to learn from
    assert set(episode[past[2][0]]) == {-1, 0, 1}


def test_seed_average_is_the_mean_probability():
    class Const:
        def __init__(self, s):
            self.s = s

        def fit(self, X, y):
            return self

        def predict_proba(self, X):
            return np.tile([1 - self.s / 10, self.s / 10], (len(X), 1))

    m = SeedAverage(Const, seeds=[1, 2, 6]).fit(np.zeros((4, 1)), np.array([0, 1, 0, 1]))
    assert np.allclose(m.predict_proba(np.zeros((2, 1)))[:, 1], 0.3)


def test_a_win_needs_the_mean_to_exceed_the_spread_across_seeds():
    delta = np.array([[0.05, 0.010, -0.02, np.nan, 0.03],
                      [0.07, -0.010, -0.01, np.nan, 0.03],
                      [0.06, 0.012, -0.03, np.nan, 0.03]])
    c = run_plan.criterion(delta, counted=np.array([True, True, True, True, False]))
    assert c["win"].tolist() == [True, False, False, False, True]
    assert c["wins_counted"] == 1 and c["scorable_counted"] == 3


def test_lead_time_counts_trading_days_to_the_minimum_and_never_looks_ahead():
    end_rows = np.arange(100, 400, 5)                        # windows ending every 5 rows
    start_row, end_row = 250, 330
    score = np.zeros(len(end_rows))
    score[[0, 1, 2]] = 1                                     # a run before the 60-day look-back: ignored
    score[[30, 31, 32]] = 1                                  # rows 250, 255, 260
    r = run_plan.lead_time(score, 0.5, end_rows, start_row, end_row)
    assert r == {"alarm": True, "lead_days": 330 - 260, "before_start": False, "window": 32}
    score[[25, 26, 27]] = 1                                  # rows 225-235, inside the look-back
    assert run_plan.lead_time(score, 0.5, end_rows, start_row, end_row)["lead_days"] == 330 - 235
    late = np.zeros(len(end_rows))
    late[[45, 46, 47]] = 1                                   # completes at row 335, after the minimum
    assert run_plan.lead_time(late, 0.5, end_rows, start_row, end_row) == {"alarm": False, "lead_days": None}


def test_heavy_tailed_crises_keep_the_gaussian_draws_and_fatten_only_the_crisis_tails():
    g = run_plan.log_returns(run_plan.synthetic_prices(seed=1))
    h = run_plan.log_returns(run_plan.synthetic_prices(seed=1, crisis="heavy-tailed"))
    _, episode = run_plan.crisis_labels(g.index)
    calm = (episode < 0) & (g.index < "2008-01-01")           # well away from every crisis
    np.testing.assert_allclose(g.to_numpy()[calm], h.to_numpy()[calm])

    def kurtosis(x):
        z = (x - x.mean()) / x.std()
        return float((z ** 4).mean())

    crisis_days = (episode >= 0)
    assert kurtosis(h.to_numpy()[crisis_days].ravel()) > kurtosis(g.to_numpy()[crisis_days].ravel()) + 1
    with pytest.raises(ValueError):
        run_plan.synthetic_prices(crisis="other")


def test_tails_only_crises_keep_volatility_and_correlation_and_fatten_the_tails():
    """The positive control: outside the regime the returns are those of the null; inside it
    only the tails change. Kendall's tau depends only on the correlation for any elliptical
    law, 2/pi * arcsin(0.3) = 0.194 here, so it is the same on calm and regime days."""
    from scipy.stats import kendalltau

    g = run_plan.log_returns(run_plan.synthetic_prices(seed=2))
    t = run_plan.log_returns(run_plan.synthetic_prices(seed=2, crisis="tails-only"))
    regime = run_plan.synthetic_regime(g.index)
    np.testing.assert_allclose(t.to_numpy()[~regime], g.to_numpy()[~regime])

    def mean_tau(x):
        return np.mean([kendalltau(x[:, i], x[:, j])[0] for i in range(7) for j in range(i + 1, 7)])

    def kurtosis(x):
        z = (x - x.mean()) / x.std()
        return float((z ** 4).mean())

    x_regime, x_calm = t.to_numpy()[regime], t.to_numpy()[~regime]
    assert mean_tau(x_regime) == pytest.approx(2 / np.pi * np.arcsin(0.3), abs=0.03)
    assert mean_tau(x_calm) == pytest.approx(2 / np.pi * np.arcsin(0.3), abs=0.03)
    assert kurtosis(x_regime.ravel()) > kurtosis(x_calm.ravel()) + 1


def test_the_synthetic_regime_doubles_volatility_and_nearly_decides_the_label():
    from scripts import calibrate_plan

    returns = run_plan.log_returns(run_plan.synthetic_prices(seed=1))
    r, regime = returns.to_numpy(), run_plan.synthetic_regime(returns.index)
    assert r[regime].std() / r[~regime].std() == pytest.approx(2.0, rel=0.05)
    auc = calibrate_plan.regime_fraction_auc()
    assert set(auc) == set(run_plan.CRISIS_PERIODS)
    assert all(0.9 < a <= 1.0 for a in auc.values())


def test_calibration_records_every_comparison(tmp_path):
    pytest.importorskip("gtda")
    from sklearn.ensemble import RandomForestClassifier
    from scripts import calibrate_plan

    small = lambda s: RandomForestClassifier(n_estimators=10, min_samples_leaf=5,     # noqa: E731
                                             class_weight="balanced", random_state=s)
    res = calibrate_plan.criteria(run_plan.synthetic_prices(seed=0), seeds=(0, 1), make_model=small)
    for scheme in calibrate_plan.SCHEMES:
        assert set(res["schemes"][scheme]) == set(calibrate_plan.COMPARISONS)
    for name in calibrate_plan.COMPARISONS:
        assert res["schemes"]["walk-forward"][name]["scorable"] == 4
    assert set(res["success"]) == set(calibrate_plan.COMPARISONS)
    # each side of a comparison is the control plus topology: four topological columns more
    data = run_plan.build_features(run_plan.synthetic_prices(seed=0))
    sets = calibrate_plan.feature_sets(data)
    for with_topology, control in calibrate_plan.COMPARISONS.values():
        assert sets[with_topology].shape[1] == sets[control].shape[1] + 4
        np.testing.assert_array_equal(sets[with_topology][:, 4:], sets[control])
    assert calibrate_plan.clopper_pearson(0, 20)[1] == pytest.approx(1 - 0.025 ** (1 / 20))


def test_dry_run_on_synthetic_prices():
    pytest.importorskip("gtda")
    from sklearn.ensemble import RandomForestClassifier

    small = lambda s: RandomForestClassifier(n_estimators=20, min_samples_leaf=5,     # noqa: E731
                                             class_weight="balanced", random_state=s)
    res = run_plan.run(run_plan.synthetic_prices(), seeds=(0, 1), make_model=small)
    assert len(res["positives"]) == 6        # business days, so more windows than the 1 092 trading-day ones
    loco, wf = res["schemes"]["leave one crisis out"], res["schemes"]["walk-forward"]
    assert loco["scorable"] == 5 and wf["scorable"] == 4      # walk-forward cannot score 2008
    gfc = wf["episodes"]["global financial crisis"]
    assert gfc["delta_auc"]["mean"] is None
    assert not wf["alarms"]["global financial crisis"][run_plan.BASELINE]["scorable"]
    assert set(loco["reported"]) == set(run_plan.REPORTED)
    assert set(loco["reported"]["kurtosis"]) == {"comparison", "wins", "passes"}
    # the injected regimes are easy: every score ranks the held-out crisis well
    for t in loco["episodes"].values():
        assert t["random_forest"]["volatility"]["roc_auc"]["mean"] > 0.8
