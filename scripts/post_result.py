"""Analyses added after the first result, declared here before any of them was computed.

Written on 2026-11-19. The first result on market data (analysis plan v0.1.2, commit caf7aaa)
did not meet the pre-registered success criterion; that verdict stands, and nothing here changes
it. An external review of plan v0.1.2, written before it saw that result, showed that the
criterion has almost no power where it matters: on synthetic crises that change volatility,
correlation and the tails at once (the heavy-tailed scenario of scripts/calibrate_plan.py), where
topology does have something to see, the primary comparison passed in 0 of 20 datasets. The
review traced this to the ceiling of the ROC-AUC: the baselines already separate the crisis
windows almost perfectly, so an added predictor can raise it by little. The analyses below,
proposed by that review, measure what the market data can and cannot say. They were written down
and pushed before any of them was computed, on synthetic or market data, and every number they
produce is reported, whatever it shows.

Comparisons: the primary one (topology + spectrum + volatility against spectrum + volatility)
and the same with Mardia's kurtosis on both sides. Schemes: leave-one-crisis-out and the
past-only walk-forward of the plan, with its folds, random forests and 10 seeds. Each statistic is
computed per held-out block for every seed, averaged over the seeds, then averaged over the
counted blocks (5 under leave-one-crisis-out, the 4 scorable ones under the walk-forward):

* dAUC: ROC-AUC of the combined model minus that of the control (the plan's ΔAUC);
* dLL: log-loss of the control minus that of the combined model, probabilities clipped to
  [0.001, 0.999]: a proper scoring rule, still sensitive near perfect separation;
* dBrier: Brier score of the control minus that of the combined model;
* dAUC|vol: the ROC-AUC difference on the pairs of a crisis window and a normal window whose
  realised volatilities differ by at most 0.1 in log, where volatility cannot separate the
  classes; a block with fewer than 5 such pairs is left out.

For each statistic S, a Monte Carlo test (Barnard, 1963): p = (1 + #{S_null >= S}) / (N + 1),
with S_null from N synthetic null datasets: N = 99 (seeds 0-98) in the plan's setting and N = 20
in each setting of Gidea & Katz (so p >= 1/21 there). The size is checked by leaving each null
dataset out in turn. The power is the share of datasets of an alternative with p <= 0.05:
heavy-tailed (the realistic one), tails-only and clayton (crises with lower-tail dependence
through a Clayton copula at the null's Kendall tau; not elliptical, so neither the spectrum nor
the kurtosis describes it: the positive control for the comparison with the kurtosis), 20
datasets each; the settings of Gidea & Katz get heavy-tailed and tails-only.

On market data the script reports S and p for every statistic, comparison, scheme and setting.
If no statistic has power in the heavy-tailed scenario, the market data cannot answer the
question with this design, and that is the conclusion that will be reported.

Usage:
    python -m scripts.post_result --setting plan --scenario null --seeds 0 1 2
    python -m scripts.post_result --setting plan --market
    python -m scripts.post_result --summarize                    # -> results/post_result.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from scripts.run_plan import (BASELINE, COMBINED, CONTEXT, NOT_COUNTED, REPORTED, SEEDS, SETTINGS,
                              build_features, synthetic_prices)
from src.classify import make_random_forest
from src.data import CRISIS_PERIODS, load_frozen_prices
from src.validation import leave_one_crisis_out, past_only_crisis_splits

PAIRS = {"primary": (COMBINED, BASELINE), "kurtosis": REPORTED["kurtosis"]}
SCENARIOS = {"null": "gaussian", "heavy-tailed": "heavy-tailed", "tails-only": "tails-only",
             "clayton": "clayton"}
SCHEMES = ("leave one crisis out", "walk-forward")
STATS = ("dAUC", "dLL", "dBrier", "dAUC|vol")
CLIP, CALIPER, MIN_PAIRS, ALPHA = 1e-3, 0.1, 5, 0.05
N_NULL = {"plan": 99, "gidea-katz-50": 20, "gidea-katz-100": 20}
OUT = "results/post_result"


def matched_auc(score: np.ndarray, y: np.ndarray, log_vol: np.ndarray) -> tuple[float, int]:
    """ROC-AUC over the (crisis, normal) pairs whose log volatilities differ by at most CALIPER;
    NaN when there are fewer than MIN_PAIRS such pairs."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    close = np.abs(log_vol[pos][:, None] - log_vol[neg][None, :]) <= CALIPER
    n = int(close.sum())
    if n < MIN_PAIRS:
        return float("nan"), n
    diff = score[pos][:, None] - score[neg][None, :]
    return float(((diff > 0) + 0.5 * (diff == 0))[close].mean()), n


def block_metrics(X: np.ndarray, y: np.ndarray, log_vol: np.ndarray, train: np.ndarray,
                  test: np.ndarray, seeds, make_model) -> dict:
    """The four raw metrics of one feature set on one held-out block, one value per seed."""
    from sklearn.metrics import roc_auc_score

    out = {"auc": [], "ll": [], "brier": [], "auc_vol": []}
    yt = y[test]
    for s in seeds:
        p = make_model(s).fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        pc = np.clip(p, CLIP, 1 - CLIP)
        out["auc"].append(float(roc_auc_score(yt, p)))
        out["ll"].append(float(-np.mean(yt * np.log(pc) + (1 - yt) * np.log(1 - pc))))
        out["brier"].append(float(np.mean((p - yt) ** 2)))
        a, n = matched_auc(p, yt, log_vol[test])
        out["auc_vol"].append(a)
    out["n_pairs"] = n
    return out


def deltas(a: dict, b: dict) -> dict:
    """Combined (a) against control (b), each statistic positive when topology helps."""
    with np.errstate(invalid="ignore"):
        d = {"dAUC": np.mean(np.subtract(a["auc"], b["auc"])),
             "dLL": np.mean(np.subtract(b["ll"], a["ll"])),
             "dBrier": np.mean(np.subtract(b["brier"], a["brier"])),
             "dAUC|vol": np.mean(np.subtract(a["auc_vol"], b["auc_vol"]))}
    out = {k: (None if np.isnan(v) else float(v)) for k, v in d.items()}
    out["n_pairs"] = a["n_pairs"]
    return out


def dataset_stats(prices, setting: str = "plan", seeds=SEEDS, make_model=make_random_forest) -> dict:
    window, dims = SETTINGS[setting]
    data = build_features(prices, window, dims)
    y, episode, sets = data["y"], data["episode"], data["sets"]
    log_vol = np.log(sets["volatility"].ravel())
    names = list(CRISIS_PERIODS)
    episodes = sorted(set(episode[episode >= 0].tolist()))
    splits = {"leave one crisis out": list(leave_one_crisis_out(episode, data["gap"], CONTEXT)),
              "walk-forward": list(past_only_crisis_splits(episode, data["gap"], CONTEXT))}
    needed = sorted({k for pair in PAIRS.values() for k in pair})
    out = {}
    for scheme in SCHEMES:
        out[scheme] = {pair: {"blocks": {}} for pair in PAIRS}
        for (train, test), e in zip(splits[scheme], episodes):
            if len(set(y[train])) < 2 or len(set(y[test])) < 2:
                continue
            raw = {k: block_metrics(sets[k], y, log_vol, train, test, seeds, make_model) for k in needed}
            for pair, (a, b) in PAIRS.items():
                out[scheme][pair]["blocks"][names[e]] = deltas(raw[a], raw[b])
        for pair in PAIRS:
            blocks = {k: v for k, v in out[scheme][pair]["blocks"].items() if k not in NOT_COUNTED}
            out[scheme][pair]["S"] = {
                st: (float(np.mean(vals)) if (vals := [v[st] for v in blocks.values() if v[st] is not None])
                     else None) for st in STATS}
            out[scheme][pair]["counted_blocks"] = len(blocks)
    return out


def monte_carlo_p(s: float, null: list[float]) -> float:
    """Barnard's Monte Carlo p-value: (1 + #{null >= s}) / (N + 1)."""
    return (1 + sum(v >= s for v in null)) / (len(null) + 1)


def run_synthetic(setting: str, scenario: str, seeds: list[int], out_dir: str = OUT) -> None:
    d = Path(out_dir) / setting / scenario
    d.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f = d / f"seed{seed:02d}.json"
        if f.exists():
            print(f"{setting} {scenario} seed {seed}: already in {f}, skipped", flush=True)
            continue
        t0 = time.time()
        res = {"setting": setting, "scenario": scenario, "seed": seed,
               "schemes": dataset_stats(synthetic_prices(seed, crisis=SCENARIOS[scenario]), setting)}
        f.write_text(json.dumps(res, indent=2))
        s = res["schemes"]["leave one crisis out"]["primary"]["S"]
        print(f"{setting} {scenario} seed {seed}: LOCO primary " + ", ".join(
            f"{k} {v:+.4f}" if v is not None else f"{k} -" for k, v in s.items())
            + f" ({time.time() - t0:.0f} s)", flush=True)


def run_market(setting: str, out_dir: str = OUT) -> None:
    f = Path(out_dir) / setting / "market.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    res = {"setting": setting, "data": "frozen prices", "schemes": dataset_stats(load_frozen_prices(), setting)}
    f.write_text(json.dumps(res, indent=2))
    print(f"[ok] {f}")


def summarize(out_dir: str = OUT, out: str = "results/post_result.json") -> dict:
    summary = {}
    for setting in SETTINGS:
        root = Path(out_dir) / setting
        runs = {sc: [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(root / sc / "seed*.json")))]
                for sc in SCENARIOS}
        nulls = [r for r in runs["null"] if r["seed"] < N_NULL[setting]]
        if not nulls:
            continue
        market = json.loads((root / "market.json").read_text()) if (root / "market.json").exists() else None
        summary[setting] = {"n_null": len(nulls)}
        for scheme in SCHEMES:
            for pair in PAIRS:
                for st in STATS:
                    null = [r["schemes"][scheme][pair]["S"][st] for r in nulls]
                    null = [v for v in null if v is not None]
                    if not null:
                        continue
                    loo = [monte_carlo_p(v, null[:i] + null[i + 1:]) for i, v in enumerate(null)]
                    entry = {"size_leave_one_out": float(np.mean([p <= ALPHA for p in loo]))}
                    for sc in ("heavy-tailed", "tails-only", "clayton"):
                        vals = [r["schemes"][scheme][pair]["S"][st] for r in runs[sc]]
                        vals = [v for v in vals if v is not None]
                        if vals:
                            entry[f"power {sc}"] = {"k": int(sum(monte_carlo_p(v, null) <= ALPHA for v in vals)),
                                                    "n": len(vals)}
                    if market is not None:
                        s = market["schemes"][scheme][pair]["S"][st]
                        entry["market"] = None if s is None else {"S": s, "p": monte_carlo_p(s, null)}
                    summary[setting].setdefault(scheme, {}).setdefault(pair, {})[st] = entry
    Path(out).write_text(json.dumps(summary, indent=2))
    for setting, s in summary.items():
        for scheme in SCHEMES:
            for pair in PAIRS:
                for st, e in s.get(scheme, {}).get(pair, {}).items():
                    pw = "; ".join(f"{k.split()[1]} {v['k']}/{v['n']}" for k, v in e.items() if k.startswith("power"))
                    m = e.get("market")
                    mk = f"market S {m['S']:+.4f} p {m['p']:.3f}" if m else "market -"
                    print(f"{setting:15s} {scheme:21s} {pair:8s} {st:8s} size {e['size_leave_one_out']:.2f}; "
                          f"power {pw}; {mk}")
    print(f"[ok] {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=list(SETTINGS), default="plan")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="null")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--market", action="store_true")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize()
    elif a.market:
        run_market(a.setting)
    else:
        run_synthetic(a.setting, a.scenario, a.seeds)
