"""Hypothesis for 2008-09, added after the result and declared here before any of it was computed.

Written on 2026-11-21 at the request of the external review of that day, which read the results of
scripts/direction_diagnostic.py. Under leave-one-crisis-out, the realised volatility separates the
crisis windows of the 2008-09 block from its normal windows (ROC-AUC 0.875; 1.00 against the windows
that end before the crisis), but the forest on volatility alone, trained on the other crises, does
not (0.553; 0.714 against the windows before; results/direction_diagnostic.json). The review's
hypothesis, translated from its Spanish: "The rebound windows of the other crises are also labelled
normal in the training folds. In 2020 those windows have the highest volatility of the whole sample,
2 to 5 % daily, with a normal label. A forest then learns a non-monotone relation: medium-high
volatility -> crisis, very high volatility -> normal. So it scores low the October-November 2008
windows, at 4-5.5 %, which are crisis. The logistic regression is monotone and cannot learn that
shape; hence its 0.875." In figures/features_timeseries.png the highest volatility of the sample is
inside the 2008-09 crisis, which this fold holds out; the run lists the training windows above 4 %.

The review's two predictions, with its thresholds (translated):
1. "In the leave-one-crisis-out fold of 2008-09, evaluate the forest on volatility on a grid of
   values. Prediction: the mean crisis probability for volatility >= 4 % is lower than for the
   range 2-3 %."
2. "Retrain that fold without the 2020 rebound windows (the 12 after 23 March 2020). Prediction: the
   forest's AUC against the earlier windows rises from 0.71 to >= 0.9."
"If both fail, the hypothesis is wrong and 2008-09 remains an unexplained failure of transfer, which
is what the README says today."

How they are computed, fixed here. Data, windows, labels, fold and forests are those of the plan
(scripts/run_plan.py): the frozen prices; 40-day windows with a 5-day step, labelled by their last
day; the 2008-09 held-out block, the crisis with the 12 windows on either side; training on every
other window except the purge gap of 7 windows on each side of the block; random forests of 300
trees (min_samples_leaf=5, balanced class weights) with seeds 0-9, on the realised volatility alone,
the mean over the assets of the standard deviation of the window's daily log returns (0.04 is 4 %
daily, as on the figure).
- Prediction 1. The grid runs from 0.0200 to 0.0300 and from 0.0400 to the highest realised
  volatility of any window in the sample, in steps of 0.0001 (0.01 % daily), both ends included.
  The forest of each seed, trained on the fold's training set, gives a crisis probability at every
  point; its mean over the points of each range is averaged over the seeds. The prediction holds
  when that average is lower for >= 4 % than for 2-3 %.
- Prediction 2. The 2020 rebound windows are the 12 normal windows of the 2020 held-out block that
  end after its last crisis window: the first 12 windows whose last day is after 23 March 2020, the
  "after" side of scripts/direction_diagnostic.py. They are removed from the fold's training set,
  nothing else changes, and the forests are refitted with the same seeds. The prediction holds when
  the mean over the seeds of the ROC-AUC of the 2008-09 crisis windows against the 12 normal windows
  of the block that end before them is at least 0.9. With the full training set it is 0.714, which
  this run must reproduce.
- Reading. Each prediction is reported as holding or not. If both hold, the hypothesis accounts for
  2008-09; if both fail, it is wrong, as the review says; if only one holds, that is reported and
  nothing more is concluded. Nothing here changes the verdict of the plan or the reading of
  scripts/post_result.py.

Reported, counting for nothing: the ROC-AUC on the whole block and against the windows after, with
and without the 2020 rebound windows; prediction 1's averages for the forests trained without them;
the same refit without the 12 rebound windows of every crisis in the training set (2011, 2015,
Q4 2018, 2020 and 2022), the general form of the hypothesis; the forest's crisis probability over a
grid from the lowest to the highest volatility of any window, step 0.0001, averaged over the seeds,
for each of the three training sets; the training windows with volatility >= 0.04 and the 2008-09
crisis windows with volatility >= 0.04, with their dates. The run also recomputes the forest's
per-seed AUCs in the 2008-09 block, which must reproduce results/plan.json (the whole block) and
results/direction_diagnostic.json (whole block, before and after).

Usage:
    python -m scripts.rebound_hypothesis                    # synthetic dry run, no market data
    python -m scripts.rebound_hypothesis --data frozen      # -> results/rebound_hypothesis.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.direction_diagnostic import side_aucs
from scripts.run_plan import CONTEXT, SEEDS, build_features, synthetic_prices
from src.classify import make_random_forest
from src.data import CRISIS_PERIODS, load_frozen_prices
from src.validation import leave_one_crisis_out

HELD_OUT = "global financial crisis"
REBOUND_OF = "covid crash"
GRID_STEP = 1e-4                                   # 0.01 % daily
MID = (0.02, 0.03)                                 # prediction 1: 2-3 % daily ...
HIGH_FROM = 0.04                                   # ... against >= 4 % daily
BEFORE_MIN = 0.9                                   # prediction 2
PLAN, NO_2020, NO_REBOUND = "plan", "without the 2020 rebound", "without every rebound"
PARTS = ("all", "before", "after")


def grid(lo: float, hi: float, step: float = GRID_STEP) -> np.ndarray:
    """lo, lo + step, ... up to hi, both ends included when hi is on the grid (empty if hi < lo)."""
    if hi < lo:
        return np.empty(0)
    n = int(np.floor((hi - lo) / step + 1e-9))
    return np.round(lo + step * np.arange(n + 1), 6)


def rebound_windows(y: np.ndarray, episode: np.ndarray, e: int, context: int = CONTEXT) -> np.ndarray:
    """The ``context`` windows that follow the last crisis window of episode ``e``: the "after" side
    of its held-out block, all of them normal."""
    last = int(np.flatnonzero(np.asarray(episode) == e).max())
    idx = np.arange(last + 1, min(len(y), last + 1 + context))
    if np.asarray(y)[idx].any():
        raise ValueError("a rebound window is labelled crisis")
    return idx


def reading(first: bool | None, second: bool | None) -> str:
    if first is None or second is None:
        return "not computable"
    if first and second:
        return "both predictions hold: the hypothesis accounts for 2008-09"
    if not first and not second:
        return "both predictions fail: the hypothesis is wrong; 2008-09 remains an unexplained failure of transfer"
    return f"only prediction {1 if first else 2} holds: reported, nothing more is concluded"


def _mean(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def run(prices: pd.DataFrame, seeds=SEEDS, make_model=make_random_forest) -> dict:
    t0 = time.time()
    data = build_features(prices)                  # the plan's setting: 40-day windows
    X, y, episode = data["sets"]["volatility"], data["y"], data["episode"]
    vol = X[:, 0]
    dates = data["returns"].index[data["end"]]
    names = list(CRISIS_PERIODS)
    eps = sorted(set(episode[episode >= 0].tolist()))
    splits = dict(zip(eps, leave_one_crisis_out(episode, data["gap"], CONTEXT)))
    held = names.index(HELD_OUT)
    train, test = splits[held]
    rebound_2020 = rebound_windows(y, episode, names.index(REBOUND_OF))
    rebound_all = np.concatenate([rebound_windows(y, episode, e) for e in eps if e != held])
    if not (np.isin(rebound_2020, train).all() and np.isin(rebound_all, train).all()):
        raise ValueError("a rebound window is not in the training set of the fold")
    trainings = {PLAN: train, NO_2020: np.setdiff1d(train, rebound_2020),
                 NO_REBOUND: np.setdiff1d(train, rebound_all)}
    mid, high = grid(*MID), grid(HIGH_FROM, float(vol.max()))
    curve_x = grid(np.floor(vol.min() / GRID_STEP) * GRID_STEP, float(vol.max()))
    yb = y[test]
    out = {}
    for name, tr in trainings.items():
        per_seed = {p: [] for p in (*PARTS, "probability, 2-3 %", "probability, >= 4 %")}
        curve, block = np.zeros(len(curve_x)), np.zeros(len(test))
        for s in seeds:
            m = make_model(s).fit(X[tr], y[tr])
            score = m.predict_proba(X[test])[:, 1]
            block += score / len(seeds)
            for p, a in side_aucs(score, yb).items():
                per_seed[p].append(a)
            per_seed["probability, 2-3 %"].append(float(m.predict_proba(mid[:, None])[:, 1].mean()))
            per_seed["probability, >= 4 %"].append(
                float(m.predict_proba(high[:, None])[:, 1].mean()) if len(high) else None)
            curve += m.predict_proba(curve_x[:, None])[:, 1] / len(seeds)
        out[name] = {"n_train": int(len(tr)), "n_train_crisis": int(y[tr].sum()),
                     "AUC": {p: _mean(per_seed[p]) for p in PARTS},
                     "probability, 2-3 %": _mean(per_seed["probability, 2-3 %"]),
                     "probability, >= 4 %": _mean(per_seed["probability, >= 4 %"]),
                     "per_seed": per_seed, "curve": curve.tolist(), "block score": block.tolist()}
    p1_mid, p1_high = out[PLAN]["probability, 2-3 %"], out[PLAN]["probability, >= 4 %"]
    first = None if p1_high is None else bool(p1_high < p1_mid)
    second = bool(out[NO_2020]["AUC"]["before"] >= BEFORE_MIN)
    hi_train = train[vol[train] >= HIGH_FROM]
    hi_block = test[(y[test] == 1) & (vol[test] >= HIGH_FROM)]
    listing = lambda idx: [{"end": str(dates[i].date()), "crisis": int(y[i]), "volatility": float(vol[i])}  # noqa: E731
                           for i in idx]
    return {
        "setting": {"window": data["window"], "purge_gap": data["gap"], "context": CONTEXT,
                    "held out": HELD_OUT, "grid step": GRID_STEP, "2-3 %": list(MID), ">= 4 %": HIGH_FROM,
                    "before at least": BEFORE_MIN},
        "seeds": list(seeds), "first_day": str(data["returns"].index[0].date()),
        "last_day": str(data["returns"].index[-1].date()),
        "highest volatility of any window": float(vol.max()),
        "2020 rebound windows": [str(dates[i].date()) for i in rebound_2020],
        "every rebound window": [str(dates[i].date()) for i in rebound_all],
        "prediction 1": {"probability, 2-3 %": p1_mid, "probability, >= 4 %": p1_high,
                         "grid points, 2-3 %": int(len(mid)), "grid points, >= 4 %": int(len(high)),
                         "holds": first},
        "prediction 2": {"AUC before, plan": out[PLAN]["AUC"]["before"],
                         "AUC before, without the 2020 rebound": out[NO_2020]["AUC"]["before"],
                         "holds": second},
        "reading": reading(first, second),
        "trainings": out,
        "grid": curve_x.tolist(),
        "block": {"end": [str(d.date()) for d in dates[test]], "crisis": yb.tolist(),
                  "volatility": vol[test].tolist()},
        "training windows with volatility >= 4 %": listing(hi_train),
        "2008-09 crisis windows with volatility >= 4 %": listing(hi_block),
        "seconds": round(time.time() - t0, 1)}


def check_against_plan(res: dict, plan_path: str = "results/plan.json",
                       diag_path: str = "results/direction_diagnostic.json") -> dict:
    """Largest difference between the forest's per-seed AUCs in the 2008-09 block, recomputed here
    with the plan's training set, and those of the plan's run and of the direction diagnostic."""
    plan = json.loads(Path(plan_path).read_text())
    diag = json.loads(Path(diag_path).read_text())
    ours = res["trainings"][PLAN]["per_seed"]
    ref = {"all": plan["schemes"]["leave one crisis out"]["episodes"][HELD_OUT]["random_forest"]
           ["volatility"]["roc_auc"]["per_seed"]}
    side = diag["schemes"]["leave one crisis out"]["episodes"][HELD_OUT]["volatility, forest"]["per_seed"]
    diffs = list(np.abs(np.array(ours["all"]) - np.array(ref["all"])))
    for p in PARTS:
        diffs += list(np.abs(np.array(ours[p]) - np.array(side[p])))
    return {"compared": len(diffs), "max_abs_difference": float(max(diffs))}


def report(res: dict) -> None:
    f = lambda x: "  -  " if x is None else f"{x:.3f}"            # noqa: E731
    print(f"held out: {HELD_OUT}; highest volatility of any window {res['highest volatility of any window']:.4f}")
    for name, t in res["trainings"].items():
        a = t["AUC"]
        print(f"  {name:26s} n_train {t['n_train']} ({t['n_train_crisis']} crisis)"
              f"  AUC all / before / after {f(a['all'])} / {f(a['before'])} / {f(a['after'])}"
              f"  probability 2-3 % {f(t['probability, 2-3 %'])}, >= 4 % {f(t['probability, >= 4 %'])}")
    p1, p2 = res["prediction 1"], res["prediction 2"]
    print(f"prediction 1: {f(p1['probability, >= 4 %'])} < {f(p1['probability, 2-3 %'])}? holds = {p1['holds']}")
    print(f"prediction 2: before {f(p2['AUC before, plan'])} -> {f(p2['AUC before, without the 2020 rebound'])}"
          f" >= {BEFORE_MIN}? holds = {p2['holds']}")
    print(f"reading: {res['reading']}")
    print("training windows with volatility >= 4 %: " + ", ".join(
        f"{w['end']} {'crisis' if w['crisis'] else 'normal'} {100 * w['volatility']:.2f}"
        for w in res["training windows with volatility >= 4 %"]))
    print("2008-09 crisis windows with volatility >= 4 %: " + ", ".join(
        f"{w['end']} {100 * w['volatility']:.2f}" for w in res["2008-09 crisis windows with volatility >= 4 %"]))
    if "reproduces the plan" in res:
        print(f"reproduces the plan: {res['reproduces the plan']}")
    print(f"({res['seconds']} s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "frozen"], default="synthetic")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    frozen = a.data == "frozen"
    # n_jobs only spreads the trees over the cores: with the same seed the forest is the same
    parallel = lambda s: make_random_forest(s).set_params(n_jobs=-1)      # noqa: E731
    res = run(load_frozen_prices() if frozen else synthetic_prices(), make_model=parallel)
    if frozen:
        res["reproduces the plan"] = check_against_plan(res)
    out = Path(a.out or ("results/rebound_hypothesis.json" if frozen
                         else "results/rebound_hypothesis_dry_run.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    report(res)
    print(f"[ok] {out}")
