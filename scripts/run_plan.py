"""Run the analysis plan of the README.

Every step follows the section "Analysis plan (fixed before any result)":

* 40-day windows with a 5-day step, labelled by their last day
  (``src/data.py``), persistent homology in dimensions 0 and 1 of each
  standardised window, and the ``L¹`` and ``L²`` norms of its persistence
  landscapes (four topological features);
* the primary comparison, topology plus the leading eigenvalue share against
  the eigenvalue share alone, with realised volatility as a secondary
  baseline, under leave-one-crisis-out and under a walk-forward scheme that
  trains only on the past;
* a random forest of 300 trees fitted with seeds 0-9; ROC-AUC and average
  precision are means over the seeds, and an episode is a win when the mean
  ΔAUC (combined minus eigenvalue) exceeds its standard deviation across seeds;
* the alarm rule: a threshold at the 95th percentile of the score on the
  normal windows of a purged calibration block (the last 20 % of the training
  fold), the alarm on the window that completes a run of 3 above it, searched
  from 60 trading days before the crisis, and the lead time in trading days to
  the drawdown minimum;
* logistic regression as a check.

By default it runs on synthetic prices with the crisis periods of
``src/data.py`` injected as high-volatility, high-correlation regimes, a dry
run of the whole pipeline that computes nothing on market data. ``--data
frozen`` runs it on the frozen prices, whose SHA-256 is checked on loading;
that run is the first result of the project.

Usage:
    python -m scripts.run_plan                       # synthetic dry run
    python -m scripts.run_plan --data frozen         # the analysis plan, on market data
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.classify import SeedAverage, make_logistic, make_random_forest
from src.data import CRISIS_PERIODS, crisis_labels, load_frozen_prices, log_returns
from src.features import (landscape_norms, leading_eigenvalue_share, realized_volatility,
                          stack_features)
from src.persistence import persistence_diagrams, sliding_windows, standardize_windows
from src.validation import (calibrated_threshold, evaluate, first_alarm, leave_one_crisis_out,
                            past_only_crisis_splits, purge_gap)

WINDOW, STEP = 40, 5
GAP = purge_gap(WINDOW, STEP)                 # 7 windows
CONTEXT = 12                                  # normal windows on each side of a held-out episode (60 days)
LOOKBACK = 60                                 # alarms are searched from 60 trading days before the start
SEEDS = tuple(range(10))
NOT_COUNTED = ("china devaluation",)          # reported, not counted (2 positive windows)
LOCO_MIN_WINS = 4                             # of the 5 counted episodes
WF_MIN_WINS = 3                               # of the counted episodes the walk-forward can score
COMBINED, BASELINE = "topology + eigenvalue", "eigenvalue"
ALARM_SCORES = (COMBINED, BASELINE, "volatility")


def synthetic_prices(seed: int = 0, start: str = "2005-01-03", end: str = "2026-11-06",
                     n_assets: int = 7) -> pd.DataFrame:
    """One-factor returns on business days, with every crisis period of
    ``src/data.py`` (and the 30 calendar days before it) at twice the
    volatility and a higher factor correlation (0.7 against 0.3)."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    sigma, rho = np.full(len(dates), 0.01), np.full(len(dates), 0.3)
    for s, e in CRISIS_PERIODS.values():
        m = (dates >= pd.Timestamp(s) - pd.Timedelta(days=30)) & (dates <= pd.Timestamp(e))
        sigma[m], rho[m] = 0.02, 0.7
    common = rng.normal(size=(len(dates), 1))
    idio = rng.normal(size=(len(dates), n_assets))
    r = sigma[:, None] * (np.sqrt(rho)[:, None] * common + np.sqrt(1 - rho)[:, None] * idio)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates,
                        columns=[f"S{i}" for i in range(n_assets)])


def build_features(prices: pd.DataFrame) -> dict:
    """Windows, labels and the feature sets compared on identical folds."""
    returns = log_returns(prices)
    windows, end = sliding_windows(returns.to_numpy(), WINDOW, STEP)
    y, episode = crisis_labels(returns.index[end])
    topo = landscape_norms(persistence_diagrams(standardize_windows(windows)))
    eig, vol = leading_eigenvalue_share(windows), realized_volatility(windows)
    sets = {COMBINED: stack_features(topo, eig),
            BASELINE: eig,
            "volatility": vol,
            "topology": topo,
            "volatility + eigenvalue": stack_features(vol, eig),
            "topology + volatility + eigenvalue": stack_features(topo, vol, eig)}
    return {"returns": returns, "end": end, "y": y, "episode": episode, "sets": sets}


def scores_by_episode(sets: dict, y: np.ndarray, splits: list, seeds, make_model) -> dict:
    """ROC-AUC and AP of every feature set on every fold, one row per seed
    (NaN where the fold cannot be scored)."""
    out = {}
    for name, X in sets.items():
        auc = np.full((len(seeds), len(splits)), np.nan)
        ap = np.full_like(auc, np.nan)
        for i, s in enumerate(seeds):
            r = evaluate(lambda s=s: make_model(s), X, y, splits)
            for k, f in enumerate(r["folds"]):
                if not f["skipped"]:
                    auc[i, k], ap[i, k] = f["roc_auc"], f["average_precision"]
        out[name] = {"roc_auc": auc, "average_precision": ap}
    return out


def criterion(delta: np.ndarray, counted: np.ndarray) -> dict:
    """``delta``: ΔAUC, shape (n_seeds, n_episodes). An episode is a win when
    the mean over seeds exceeds the standard deviation across seeds."""
    with warnings.catch_warnings():                 # unscorable episodes are all-NaN columns
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(delta, axis=0)
        sd = np.nanstd(delta, axis=0, ddof=1) if delta.shape[0] > 1 else np.zeros(delta.shape[1])
    scorable = ~np.isnan(mean)
    win = scorable & (mean > sd)
    return {"mean": mean, "sd": sd, "win": win, "scorable": scorable,
            "wins_counted": int((win & counted).sum()),
            "scorable_counted": int((scorable & counted).sum())}


def lead_time(score: np.ndarray, threshold: float, end_rows: np.ndarray,
              start_row: int, end_row: int) -> dict:
    """The alarm rule on one held-out block. ``score`` and ``end_rows`` (the
    last return row of each window) are in time order; the alarm is searched
    from the first window ending ``LOOKBACK`` rows before ``start_row`` (the
    crisis start) and must come before ``end_row`` (the drawdown minimum)."""
    first = int(np.searchsorted(end_rows, start_row - LOOKBACK))
    j = first_alarm(score, threshold, start=first)
    if j is None or end_rows[j] >= end_row:
        return {"alarm": False, "lead_days": None}
    return {"alarm": True, "lead_days": int(end_row - end_rows[j]),
            "before_start": bool(end_rows[j] < start_row), "window": int(j)}


def alarms(data: dict, splits: list, seeds, make_model) -> dict:
    """Lead time of each alarm score on each held-out episode. One score per
    window: the mean probability of the models fitted with every seed."""
    names, idx = list(CRISIS_PERIODS), data["returns"].index
    episodes = sorted(set(data["episode"][data["episode"] >= 0].tolist()))
    out = {}
    for (train, test), e in zip(splits, episodes):
        start, stop = CRISIS_PERIODS[names[e]]
        start_row = int(idx.searchsorted(pd.Timestamp(start)))
        end_row = int(idx.searchsorted(pd.Timestamp(stop), side="right")) - 1
        per = {}
        for name in ALARM_SCORES:
            X = data["sets"][name]
            cal = calibrated_threshold(lambda: SeedAverage(make_model, seeds), X, data["y"], train, GAP)
            if cal is None:
                per[name] = {"scorable": False}
                continue
            model, thr = cal
            s = model.predict_proba(X[test])[:, 1]
            per[name] = {"scorable": True, "threshold": thr,
                         **lead_time(s, thr, data["end"][test], start_row, end_row)}
        out[names[e]] = per
    return out


def run(prices: pd.DataFrame, seeds=SEEDS, make_model=make_random_forest) -> dict:
    t0 = time.time()
    data = build_features(prices)
    y, episode = data["y"], data["episode"]
    names = list(CRISIS_PERIODS)
    episodes = sorted(set(episode[episode >= 0].tolist()))
    counted = np.array([names[e] not in NOT_COUNTED for e in episodes])
    schemes = {"leave one crisis out": list(leave_one_crisis_out(episode, GAP, CONTEXT)),
               "walk-forward": list(past_only_crisis_splits(episode, GAP, CONTEXT))}
    res = {"n_windows": int(len(y)),
           "positives": {names[e]: int((episode == e).sum()) for e in episodes},
           "counted": [names[e] for e, c in zip(episodes, counted) if c],
           "first_day": str(data["returns"].index[0].date()),
           "last_day": str(data["returns"].index[-1].date()),
           "seeds": list(seeds), "schemes": {}}
    for scheme, splits in schemes.items():
        rf = scores_by_episode(data["sets"], y, splits, seeds, make_model)
        lr = scores_by_episode(data["sets"], y, splits, [0], lambda s: make_logistic())
        crit = criterion(rf[COMBINED]["roc_auc"] - rf[BASELINE]["roc_auc"], counted)
        crit_lr = criterion(lr[COMBINED]["roc_auc"] - lr[BASELINE]["roc_auc"], counted)
        need = LOCO_MIN_WINS if scheme == "leave one crisis out" else WF_MIN_WINS
        table = {}
        for k, e in enumerate(episodes):
            table[names[e]] = {
                "n_test": int(len(splits[k][1])), "n_positive": int((episode == e).sum()),
                "counted": bool(counted[k]),
                "random_forest": {name: {m: _stats(v[m][:, k]) for m in v} for name, v in rf.items()},
                "logistic": {name: {m: _num(v[m][0, k]) for m in v} for name, v in lr.items()},
                "delta_auc": {"mean": _num(crit["mean"][k]), "sd": _num(crit["sd"][k]),
                              "win": bool(crit["win"][k])},
                "delta_auc_logistic": _num(crit_lr["mean"][k]),
            }
        res["schemes"][scheme] = {
            "episodes": table,
            "wins": crit["wins_counted"], "scorable": crit["scorable_counted"],
            "min_wins": need, "passes": crit["wins_counted"] >= need,
            "logistic_wins": int(((crit_lr["mean"] > 0) & counted).sum()),
            "alarms": alarms(data, splits, seeds, make_model),
        }
    res["success"] = all(v["passes"] for v in res["schemes"].values())
    res["seconds"] = round(time.time() - t0, 1)
    return res


def _num(x) -> float | None:
    return None if x is None or np.isnan(x) else float(x)


def _stats(v: np.ndarray) -> dict:
    v = v[~np.isnan(v)]
    if not len(v):
        return {"mean": None, "sd": None, "per_seed": []}
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "per_seed": v.tolist()}


def report(res: dict) -> None:
    print(f"{res['n_windows']} windows, {res['first_day']} to {res['last_day']}; "
          f"positives {res['positives']}")
    for scheme, s in res["schemes"].items():
        print(f"\n== {scheme}: {s['wins']} wins of {s['scorable']} scorable counted episodes "
              f"(needs {s['min_wins']}) -> {'PASS' if s['passes'] else 'FAIL'}")
        for ep, t in s["episodes"].items():
            rf = t["random_forest"]
            f = lambda name: (f"{rf[name]['roc_auc']['mean']:.3f}" if rf[name]["roc_auc"]["mean"] is not None
                              else "  -  ")
            d = t["delta_auc"]
            dd = f"{d['mean']:+.3f} ± {d['sd']:.3f}" if d["mean"] is not None else "not scorable"
            print(f"  {ep:26s} n+ {t['n_positive']:3d}{'' if t['counted'] else ' (not counted)':15s}"
                  f" AUC comb {f(COMBINED)}  eig {f(BASELINE)}  vol {f('volatility')}  ΔAUC {dd}"
                  f"{'  win' if d['win'] else ''}")
        for ep, per in s["alarms"].items():
            print(f"  alarm {ep:26s} " + "  ".join(
                f"{n}: {'n/a' if not a['scorable'] else (str(a['lead_days']) + ' d' if a['alarm'] else 'miss')}"
                for n, a in per.items()))
    print(f"\nsuccess: {res['success']}   ({res['seconds']} s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "frozen"], default="synthetic")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    prices = load_frozen_prices() if a.data == "frozen" else synthetic_prices()
    out = Path(a.out or ("results/plan.json" if a.data == "frozen" else "results/synthetic_dry_run.json"))
    res = run(prices)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    report(res)
    print(f"[ok] {out}")
