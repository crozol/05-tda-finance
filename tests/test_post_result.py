"""The analyses added after the first result (scripts/post_result.py) and the Clayton scenario."""

import numpy as np
import pytest

from scripts import run_plan


def test_matched_auc_uses_only_pairs_close_in_volatility():
    from scripts.post_result import matched_auc

    y = np.array([1, 1, 0, 0, 0])
    log_vol = np.log(np.array([1.00, 2.00, 1.05, 2.05, 5.00]))
    score = np.array([0.9, 0.1, 0.2, 0.5, 0.95])
    # close pairs: (0, 2) and (1, 3); the crisis window wins the first and loses the second
    auc, n = matched_auc(score, y, log_vol)
    assert n == 2
    assert np.isnan(auc)                          # fewer than MIN_PAIRS pairs
    y6 = np.r_[y, 1]
    log_vol6 = np.r_[log_vol, np.log(5.2)]
    score6 = np.r_[score, 0.99]
    from scripts import post_result
    old = post_result.MIN_PAIRS
    post_result.MIN_PAIRS = 2
    try:
        auc, n = matched_auc(score6, y6, log_vol6)
    finally:
        post_result.MIN_PAIRS = old
    assert n == 3                                 # (0, 2), (1, 3) and (5, 4)
    assert auc == pytest.approx(2 / 3)


def test_monte_carlo_p_is_barnards():
    from scripts.post_result import monte_carlo_p

    null = [0.1, 0.2, 0.3, 0.4]
    assert monte_carlo_p(0.5, null) == pytest.approx(1 / 5)
    assert monte_carlo_p(0.3, null) == pytest.approx(3 / 5)          # ties count against
    assert monte_carlo_p(0.0, null) == pytest.approx(1.0)


def test_post_result_reproduces_the_delta_auc_of_the_plan():
    pytest.importorskip("gtda")
    from sklearn.ensemble import RandomForestClassifier
    from scripts import calibrate_plan, post_result

    small = lambda s: RandomForestClassifier(n_estimators=10, min_samples_leaf=5,     # noqa: E731
                                             class_weight="balanced", random_state=s)
    prices = run_plan.synthetic_prices(seed=0)
    new = post_result.dataset_stats(prices, seeds=(0, 1), make_model=small)
    old = calibrate_plan.criteria(prices, seeds=(0, 1), make_model=small)
    for scheme in post_result.SCHEMES:
        for pair, key in (("primary", "spectrum + volatility"), ("kurtosis", "spectrum + volatility + kurtosis")):
            blocks = new[scheme][pair]["blocks"]
            for episode, d in old["schemes"][scheme][key]["delta_auc"].items():
                if d is None:
                    assert episode not in blocks
                else:
                    assert blocks[episode]["dAUC"] == pytest.approx(d, rel=1e-12, abs=1e-15)
            s = new[scheme][pair]["S"]
            assert set(s) == set(post_result.STATS)
            assert s["dLL"] is not None and s["dBrier"] is not None


def test_clayton_crises_keep_volatility_and_kendalls_tau_and_crash_together():
    from scipy.stats import kendalltau

    g = run_plan.log_returns(run_plan.synthetic_prices(seed=4))
    c = run_plan.log_returns(run_plan.synthetic_prices(seed=4, crisis="clayton"))
    regime = run_plan.synthetic_regime(g.index)
    np.testing.assert_allclose(c.to_numpy()[~regime], g.to_numpy()[~regime])
    xg, xc = g.to_numpy()[regime], c.to_numpy()[regime]

    def mean_tau(x):
        return np.mean([kendalltau(x[:, i], x[:, j])[0] for i in range(7) for j in range(i + 1, 7)])

    assert mean_tau(xc) == pytest.approx(2 / np.pi * np.arcsin(0.7), abs=0.03)
    assert xc.std() == pytest.approx(xg.std(), rel=0.05)

    def joint(x, lower):
        q = np.quantile(x, 0.05 if lower else 0.95, axis=0)
        hit = (x < q) if lower else (x > q)
        return np.mean([np.mean(hit[:, i] & hit[:, j]) for i in range(7) for j in range(i + 1, 7)]) / 0.05

    assert joint(xc, lower=True) > joint(xg, lower=True) + 0.15     # joint crashes
    assert joint(xc, lower=False) < joint(xg, lower=False) - 0.15   # rises are not joint
