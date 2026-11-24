"""The early-warning study (analysis plan: docs/early-warning-plan.md).

Positives: 50-day windows of the four US indices that end in the 60 trading days before the peak
of a drawdown of the S&P 500 of at least 15 %; negatives: windows that end at least 120 trading
days from every peak and every trough (src/early_warning.py). Features of each window:

* topology as Gidea & Katz (2018) compute it: Vietoris-Rips persistence of the raw log returns in
  dimension 1, and the L1 and L2 norms of its landscape summed over all layers, in closed form
  (src.features.total_persistence_norms);
* baselines: realised volatility, the eigenvalues of the correlation matrix and the eigenvalues
  of the covariance matrix, the complete Gaussian control for topology on raw windows
  (src.features.covariance_spectrum).

Primary statistic: under leave-one-episode-out, for each held-out block, the class-balanced
log-loss of a random forest on the baselines minus that of a forest on topology plus the
baselines, averaged over 10 seeds; S is the mean over the blocks. Monte Carlo test against 99
synthetic null datasets; power on 20 heavy-tailed and 20 tails-only datasets. Reported, counting
for nothing: ΔAUC and wins per block, the past-only walk-forward, the plan's standardised
topology (dimensions 0 and 1) against volatility and the correlation spectrum, and, on market
data only, logistic regression.

    python -m scripts.early_warning --dry-run                         # one synthetic null dataset, printed
    python -m scripts.early_warning --scenario null --seeds 0 1 2     # -> results/early_warning/null/seed_*.json
    python -m scripts.early_warning --worker 1 --of 4                 # every synthetic dataset, a quarter of them
    python -m scripts.early_warning --market                          # -> results/early_warning/market.json
    python -m scripts.early_warning --summarize                       # -> results/early_warning.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from src.classify import make_logistic, make_random_forest
from src.data import log_returns
from src.early_warning import (FAR, HORIZON, INDICES, SCENARIOS, drawdown_episodes, leave_one_episode_out,
                               load_frozen_indices, nearest_peak, past_only_episode_splits,
                               synthetic_indices, window_labels)
from src.features import (correlation_spectrum, covariance_spectrum, realized_volatility, stack_features,
                          total_persistence_norms)
from src.persistence import persistence_diagrams, sliding_windows, standardize_windows
from src.validation import purge_gap

WINDOW, STEP = 50, 5
GAP = purge_gap(WINDOW, STEP)                       # 9 windows
SEEDS = tuple(range(10))
CLIP, ALPHA = 1e-3, 0.05
N_NULL, N_ALT = 99, 20
SYNTHETIC_SEEDS = {"null": range(N_NULL), "heavy-tailed": range(100, 100 + N_ALT),
                   "tails-only": range(100, 100 + N_ALT)}
COMPARISONS = {                                     # name: (combined, control, scheme)
    "primary": ("gk + baselines", "baselines", "leave one episode out"),
    "walk-forward": ("gk + baselines", "baselines", "walk-forward"),
    "standardised": ("std + volatility + correlation", "volatility + correlation", "leave one episode out"),
}
OUT = Path("results/early_warning")


def market_episodes(prices) -> list[tuple[int, int]]:
    """Episodes in rows of the returns (prices.index[1:]), from the S&P 500 closes."""
    return drawdown_episodes(prices[INDICES[0]].to_numpy()[1:])


def build(prices, episodes) -> dict:
    returns = log_returns(prices)
    windows, end = sliding_windows(returns.to_numpy(), WINDOW, STEP)
    gk = total_persistence_norms(persistence_diagrams(windows, (1,)), dims=(1,))
    std = total_persistence_norms(persistence_diagrams(standardize_windows(windows), (0, 1)), dims=(0, 1))
    vol, corr, cov = realized_volatility(windows), correlation_spectrum(windows), covariance_spectrum(windows)
    sets = {"gk + baselines": stack_features(gk, vol, corr, cov),
            "baselines": stack_features(vol, corr, cov),
            "std + volatility + correlation": stack_features(std, vol, corr),
            "volatility + correlation": stack_features(vol, corr),
            "volatility": vol, "gk": gk}
    return {"dates": returns.index[end], "end": end, "y": window_labels(end, episodes),
            "block": nearest_peak(end, episodes), "sets": sets}


def balanced_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, CLIP, 1 - CLIP)
    return float(0.5 * (-np.log(p[y == 1]).mean() - np.log(1 - p[y == 0]).mean()))


def compare(data: dict, combined: str, control: str, scheme: str, make_model=make_random_forest,
            seeds=SEEDS) -> dict:
    """dLL and ΔAUC of ``combined`` against ``control`` on every held-out block, per seed."""
    from sklearn.metrics import roc_auc_score

    y, block = data["y"], data["block"]
    split = leave_one_episode_out if scheme == "leave one episode out" else past_only_episode_splits
    blocks = []
    for e, (train, test) in enumerate(split(block, y, GAP)):
        yt = y[test]
        if len(set(y[train])) < 2 or len(set(yt)) < 2:
            blocks.append({"episode": e, "scorable": False})
            continue
        rows = {"dLL": [], "dAUC": [], "AUC control": [], "AUC combined": []}
        for s in seeds:
            pc = make_model(s).fit(data["sets"][combined][train], y[train]).predict_proba(data["sets"][combined][test])[:, 1]
            pb = make_model(s).fit(data["sets"][control][train], y[train]).predict_proba(data["sets"][control][test])[:, 1]
            rows["dLL"].append(balanced_log_loss(yt, pb) - balanced_log_loss(yt, pc))
            ac, ab = roc_auc_score(yt, pc), roc_auc_score(yt, pb)
            rows["dAUC"].append(ac - ab)
            rows["AUC control"].append(ab)
            rows["AUC combined"].append(ac)
        blocks.append({"episode": e, "scorable": True, "n_positive": int(yt.sum()), "n_negative": int((yt == 0).sum()),
                       **{k: float(np.mean(v)) for k, v in rows.items()},
                       "dAUC sd": float(np.std(rows["dAUC"], ddof=1)) if len(seeds) > 1 else 0.0})
    scored = [b for b in blocks if b["scorable"]]
    return {"S": float(np.mean([b["dLL"] for b in scored])),
            "dAUC": float(np.mean([b["dAUC"] for b in scored])),
            "wins": int(sum(b["dAUC"] > b["dAUC sd"] for b in scored)), "scorable": len(scored),
            "AUC control": float(np.mean([b["AUC control"] for b in scored])), "blocks": blocks}


def dataset(prices, episodes, seeds=SEEDS, logistic: bool = False, names=tuple(COMPARISONS)) -> dict:
    t0 = time.time()
    data = build(prices, episodes)
    y = data["y"]
    out = {"n_windows": int(len(y)), "n_positive": int((y == 1).sum()), "n_negative": int((y == 0).sum()),
           "comparisons": {name: compare(data, *COMPARISONS[name], seeds=seeds) for name in names}}
    if logistic:
        out["logistic"] = {name: compare(data, a, b, scheme, lambda s: make_logistic(), seeds=(0,))
                           for name, (a, b, scheme) in COMPARISONS.items()}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def synthetic(scenario: str, seed: int) -> dict:
    """One synthetic dataset. The reported comparisons are computed on the first N_ALT null datasets
    and on every alternative one (their p-values use those N_ALT nulls); the primary one on all."""
    prices = load_frozen_indices()
    episodes = market_episodes(prices)
    names = tuple(COMPARISONS) if scenario != "null" or seed < N_ALT else ("primary",)
    res = dataset(synthetic_indices(seed, scenario, prices.index, episodes), episodes, names=names)
    return {"scenario": scenario, "seed": seed, **res}


def jobs() -> list[tuple[str, int]]:
    return [(s, k) for s in SCENARIOS for k in SYNTHETIC_SEEDS[s]]


def mc_p(s: float, null: np.ndarray) -> float:
    """Monte Carlo p-value (Barnard, 1963): (1 + #{null >= s}) / (N + 1)."""
    return float((1 + np.sum(null >= s)) / (len(null) + 1))


def summarize() -> dict:
    """Size, power and the market p-value of every comparison, and the reading of the plan."""
    from scipy.stats import binom

    load = lambda sc: [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(OUT / sc / "seed_*.json")))]  # noqa: E731
    runs = {sc: load(sc) for sc in SCENARIOS}
    market = json.loads((OUT / "market.json").read_text()) if (OUT / "market.json").exists() else None
    out = {"n": {sc: len(v) for sc, v in runs.items()}, "comparisons": {}}
    for name in COMPARISONS:
        null = np.array([r["comparisons"][name]["S"] for r in runs["null"] if name in r["comparisons"]])
        size = np.mean([mc_p(s, np.delete(null, i)) <= ALPHA for i, s in enumerate(null)]) if len(null) > 1 else None
        c = {"n_null": int(len(null)), "size": size}
        for sc in ("heavy-tailed", "tails-only"):
            k = int(sum(mc_p(r["comparisons"][name]["S"], null) <= ALPHA for r in runs[sc]))
            c[f"power {sc}"] = {"k": k, "n": len(runs[sc])}
        if market is not None:
            s = market["comparisons"][name]["S"]
            c["market"] = {"S": s, "p": mc_p(s, null), "dAUC": market["comparisons"][name]["dAUC"],
                           "wins": market["comparisons"][name]["wins"],
                           "scorable": market["comparisons"][name]["scorable"],
                           "AUC control": market["comparisons"][name]["AUC control"],
                           "null AUC control, 5-95 %": [float(q) for q in np.percentile(
                               [r["comparisons"][name]["AUC control"] for r in runs["null"]
                                if name in r["comparisons"]], [5, 95])]}
        out["comparisons"][name] = c
    # the reading of the plan: power first, then the p-value of the primary comparison
    pw = out["comparisons"]["primary"]["power heavy-tailed"]
    min_k = int(next(k for k in range(pw["n"] + 1) if binom.sf(k - 1, pw["n"], ALPHA) <= ALPHA)) if pw["n"] else None
    has_power = pw["n"] > 0 and pw["k"] >= min_k
    reading = {"power needed (of n heavy-tailed datasets)": min_k, "has power": bool(has_power)}
    if market is not None:
        p = out["comparisons"]["primary"]["market"]["p"]
        reading["verdict"] = ("the test cannot answer the question: no power on heavy-tailed synthetic data"
                              if not has_power else
                              ("evidence that topology adds early warning to the baselines" if p <= ALPHA
                               else "no evidence that topology adds early warning to the baselines"))
    out["reading"] = reading
    return out


def _print(res: dict) -> None:
    print(f"{res['n_windows']} windows: {res['n_positive']} positive, {res['n_negative']} negative "
          f"({res['seconds']} s)")
    for name, c in res["comparisons"].items():
        print(f"  {name:13s} S = {c['S']:+.4f}   ΔAUC {c['dAUC']:+.3f}   wins {c['wins']}/{c['scorable']}"
              f"   AUC of the control {c['AUC control']:.3f}")
        for b in c["blocks"]:
            if b["scorable"]:
                print(f"      episode {b['episode']}: n+ {b['n_positive']:3d} n- {b['n_negative']:4d}  "
                      f"dLL {b['dLL']:+.4f}  ΔAUC {b['dAUC']:+.3f} ± {b['dAUC sd']:.3f}  AUC control {b['AUC control']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--scenario", choices=SCENARIOS)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--worker", type=int)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--market", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        _print(synthetic("null", 0))
    elif a.market:
        prices = load_frozen_indices()
        res = {"first_day": str(prices.index[0].date()), "last_day": str(prices.index[-1].date()),
               "episodes": [[str(prices.index[1:][p].date()), str(prices.index[1:][t].date())]
                            for p, t in market_episodes(prices)],
               **dataset(prices, market_episodes(prices), logistic=True)}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "market.json").write_text(json.dumps(res, indent=1))
        _print(res)
    elif a.summarize:
        res = summarize()
        Path("results/early_warning.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
    else:
        todo = ([(a.scenario, k) for k in a.seeds] if a.scenario else
                [j for i, j in enumerate(jobs()) if a.worker is None or i % a.of == a.worker - 1])
        for scenario, seed in todo:
            path = OUT / scenario / f"seed_{seed}.json"
            if path.exists():
                continue
            res = synthetic(scenario, seed)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(res, indent=1))
            print(f"[ok] {path} S = {res['comparisons']['primary']['S']:+.4f} ({res['seconds']} s)", flush=True)
