"""Direction diagnostic, added after the result and declared here before any of it was computed.

Written on 2026-11-20, after the analyses of scripts/post_result.py. The hypothesis: the result on
market data is inconclusive by construction. Every feature of the comparison, the baselines
included, is unchanged when every return r becomes -r and when the days of a window are reordered
(src/features.py, src/persistence.py), while the label, being inside a peak-to-trough drawdown,
depends on the direction of the market. Inside the held-out blocks, moreover, the normal windows
after the trough are rebounds as volatile and as correlated as the crisis, and the windows that end
just after the trough still contain its most turbulent days. Nothing here changes the verdict of the plan
(criterion not met) or the reading of scripts/post_result.py; every number below is reported,
whatever it shows, and counts for nothing.

Data, folds and model are those of the plan (scripts/run_plan.py): the frozen prices, 40-day
windows with a 5-day step labelled by their last day, each held-out block the crisis with the 12
windows (60 trading days) on either side, leave-one-crisis-out and the past-only walk-forward,
random forests of 300 trees with seeds 0-9. The normal windows of a block are split into those
that end before its first crisis window ("before") and those that end after its last ("after").

1. The diagnostic. For every block, the ROC-AUC of the crisis windows against the windows
   before, and against the windows after, for two scores of volatility alone: (i) the forest on
   volatility alone, the "volatility alone" of the README (the mean over the seeds of each seed's
   AUC, as in the plan), and (ii) the realised volatility itself, with no model. And, for every
   block, the ROC-AUC of the forest on the baselines plus the mean return of the window (the mean
   of its daily log returns over days and assets, which changes sign with r) next to that of the
   baselines alone (spectrum + volatility).
2. Also: the primary comparison with the mean return on both sides, topology + spectrum +
   volatility + mean return against spectrum + volatility + mean return. ΔAUC per block, its mean
   and standard deviation over the seeds, and a win when the mean exceeds the standard deviation
   (the plan's rule).

The prediction, written before any of this was run: AUC against the earlier neighbours >= 0.8, and
against the later ones <= 0.5, in 2008-09, Q4 2018 and 2020; with the mean of the returns, the AUC
per block rises substantially. If it holds, the mechanism is established; if not, another
explanation must be sought. Its first part is checked under leave-one-crisis-out, for each of the
two scores of volatility: it holds for a score when, in each of the three
episodes, the AUC against the windows before is at least 0.8 and against the windows after at
most 0.5. The second part names no threshold; its numbers are reported.

Known before running, from results/plan.json (2026-11-19): under leave-one-crisis-out, the
logistic regression on volatility alone has ROC-AUC 0.875 in the 2008-09 block. A logistic
regression on one feature ranks the windows as that feature does, or in reverse if its coefficient
is negative, so the realised volatility has an AUC of 0.875 there (or 0.125). With 12 normal
windows on each side, the AUC on the whole block is the mean of the AUCs against each side, so
against the windows after it is at least 2 x 0.875 - 1 = 0.75. Unless the coefficient is negative,
the prediction cannot hold in 2008-09 for the realised volatility. For the forest on volatility
alone, whose AUC on the whole block is 0.553, it can.

The run also recomputes the plan's AUCs of the forest on volatility alone, on the baselines and on
the combined model, which must reproduce results/plan.json.

Usage:
    python -m scripts.direction_diagnostic                   # synthetic dry run, no market data
    python -m scripts.direction_diagnostic --data frozen     # -> results/direction_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_plan import (BASELINE, COMBINED, CONTEXT, NOT_COUNTED, SEEDS, STEP, build_features,
                              synthetic_prices)
from src.classify import make_random_forest
from src.data import CRISIS_PERIODS, load_frozen_prices
from src.features import stack_features
from src.persistence import sliding_windows
from src.validation import leave_one_crisis_out, past_only_crisis_splits

MEAN = "mean return"
WITH_MEAN = f"{BASELINE} + {MEAN}"                 # the baselines plus the mean return
COMBINED_MEAN = f"{COMBINED} + {MEAN}"             # the combined model plus the mean return
PREDICTED = ("global financial crisis", "q4 2018 selloff", "covid crash")
BEFORE_MIN, AFTER_MAX = 0.8, 0.5                   # the prediction
SCORES = ("volatility, forest", "volatility, raw")
PARTS = ("all", "before", "after")


def mean_return(windows: np.ndarray) -> np.ndarray:
    """Mean daily log return of each window over its days and assets, shape (n_windows, 1): the
    only feature here that changes sign when every return does."""
    return windows.mean(axis=(1, 2))[:, None]


def sides(y_block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Masks of the normal windows of a held-out block, in time order, that end before its first
    crisis window and after its last one."""
    y_block = np.asarray(y_block)
    idx, pos = np.arange(len(y_block)), np.flatnonzero(y_block == 1)
    normal = y_block == 0
    return normal & (idx < pos.min()), normal & (idx > pos.max())


def side_aucs(score: np.ndarray, y_block: np.ndarray) -> dict:
    """ROC-AUC of the crisis windows of a block against all its normal windows, against those
    before and against those after (None for a side without windows)."""
    from sklearn.metrics import roc_auc_score

    score, y_block = np.asarray(score, dtype=float), np.asarray(y_block)
    before, after = sides(y_block)
    out = {"all": float(roc_auc_score(y_block, score))}
    for name, side in (("before", before), ("after", after)):
        keep = (y_block == 1) | side
        out[name] = float(roc_auc_score(y_block[keep], score[keep])) if side.any() else None
    return out


def fold_scores(make_model, X: np.ndarray, y: np.ndarray, splits: list) -> list:
    """The model's crisis probability on every test block, None where the training or the test set
    has a single class (as src.validation.evaluate skips it)."""
    X = np.asarray(X).reshape(len(y), -1)
    out = []
    for train, test in splits:
        if len(set(y[train])) < 2 or len(set(y[test])) < 2:
            out.append(None)
            continue
        model = make_model()
        model.fit(X[train], y[train])
        out.append(model.predict_proba(X[test])[:, 1])
    return out


def prediction(table: dict) -> dict:
    """The first part of the prediction, for each score of volatility alone."""
    out = {}
    for score in SCORES:
        per = {}
        for ep in PREDICTED:
            r = table[ep].get(score)
            if r is None or r["before"] is None or r["after"] is None:
                per[ep] = {"before": None, "after": None, "holds": None}
                continue
            per[ep] = {"before": r["before"], "after": r["after"],
                       "holds": bool(r["before"] >= BEFORE_MIN and r["after"] <= AFTER_MAX)}
        held = [v["holds"] for v in per.values()]
        out[score] = {"per_episode": per, "holds": None if None in held else all(held)}
    return out


def run(prices: pd.DataFrame, seeds=SEEDS, make_model=make_random_forest) -> dict:
    t0 = time.time()
    data = build_features(prices)                       # the plan's setting: 40-day windows, H0 and H1
    windows, _ = sliding_windows(data["returns"].to_numpy(), data["window"], STEP)
    mret, sets = mean_return(windows), data["sets"]
    vol = sets["volatility"][:, 0]
    feats = {"volatility": sets["volatility"], BASELINE: sets[BASELINE], COMBINED: sets[COMBINED],
             WITH_MEAN: stack_features(sets[BASELINE], mret),
             COMBINED_MEAN: stack_features(sets[COMBINED], mret)}
    y, episode = data["y"], data["episode"]
    names = list(CRISIS_PERIODS)
    episodes = sorted(set(episode[episode >= 0].tolist()))
    schemes = {"leave one crisis out": list(leave_one_crisis_out(episode, data["gap"], CONTEXT)),
               "walk-forward": list(past_only_crisis_splits(episode, data["gap"], CONTEXT))}
    res = {"setting": {"window": data["window"], "step": STEP, "homology_dims": list(data["dims"]),
                       "purge_gap": data["gap"], "context": CONTEXT},
           "n_windows": int(len(y)), "seeds": list(seeds),
           "first_day": str(data["returns"].index[0].date()),
           "last_day": str(data["returns"].index[-1].date()), "schemes": {}}
    for scheme, splits in schemes.items():
        # aucs[name][i][k]: side AUCs of the forest with seed i on block k (None if not scorable)
        aucs = {name: [[None if sc is None else side_aucs(sc, y[test])
                        for sc, (_, test) in zip(fold_scores(lambda seed=seed: make_model(seed), X, y,
                                                             splits), splits)]
                       for seed in seeds]
                for name, X in feats.items()}
        table = {}
        for k, e in enumerate(episodes):
            yb = y[splits[k][1]]
            before, after = sides(yb)
            row = {"counted": names[e] not in NOT_COUNTED, "n_positive": int(yb.sum()),
                   "n_before": int(before.sum()), "n_after": int(after.sum()),
                   "volatility, raw": side_aucs(vol[splits[k][1]], yb),
                   "scorable": aucs["volatility"][0][k] is not None}
            if row["scorable"]:
                per_seed = {name: {p: [a[k][p] for a in aucs[name]] for p in PARTS} for name in feats}
                row["volatility, forest"] = {p: _mean(per_seed["volatility"][p]) for p in PARTS}
                row["volatility, forest"]["per_seed"] = per_seed["volatility"]
                row["AUC, forest"] = {name: _mean(per_seed[name]["all"]) for name in feats}
                row["AUC, forest, per seed"] = {name: per_seed[name]["all"] for name in feats}
                d = np.array(per_seed[COMBINED_MEAN]["all"]) - np.array(per_seed[WITH_MEAN]["all"])
                sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
                row["primary with the mean return"] = {"comparison": f"{COMBINED_MEAN} vs {WITH_MEAN}",
                                                       "mean": float(d.mean()), "sd": sd,
                                                       "win": bool(d.mean() > sd)}
            table[names[e]] = row
        counted = [r for r in table.values() if r["counted"] and r["scorable"]]
        res["schemes"][scheme] = {
            "episodes": table,
            "wins of the primary with the mean return (counts for nothing)":
                sum(r["primary with the mean return"]["win"] for r in counted),
            "scorable counted episodes": len(counted)}
        if scheme == "leave one crisis out":
            res["schemes"][scheme]["prediction"] = prediction(table)
    res["seconds"] = round(time.time() - t0, 1)
    return res


def check_against_plan(res: dict, path: str = "results/plan.json") -> dict:
    """Largest difference between the per-seed AUCs recomputed here and those of the plan's run."""
    plan = json.loads(Path(path).read_text())
    diffs = []
    for scheme, s in res["schemes"].items():
        for ep, row in s["episodes"].items():
            if not row["scorable"]:
                continue
            ref = plan["schemes"][scheme]["episodes"][ep]["random_forest"]
            for name in ("volatility", BASELINE, COMBINED):
                diffs += list(np.abs(np.array(row["AUC, forest, per seed"][name])
                                     - np.array(ref[name]["roc_auc"]["per_seed"])))
    return {"compared": len(diffs), "max_abs_difference": float(max(diffs)) if diffs else None}


def _mean(v: list) -> float | None:
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def report(res: dict) -> None:
    f = lambda x: "  -  " if x is None else f"{x:.3f}"            # noqa: E731
    for scheme, s in res["schemes"].items():
        print(f"\n== {scheme}  (AUC: all / before / after)")
        for ep, r in s["episodes"].items():
            raw = r["volatility, raw"]
            line = (f"  {ep:24s} n+ {r['n_positive']:2d} ({r['n_before']}|{r['n_after']})"
                    f"{'' if r['counted'] else ' not counted':12s}"
                    f" vol raw {f(raw['all'])} / {f(raw['before'])} / {f(raw['after'])}")
            if r["scorable"]:
                vf, a, d = r["volatility, forest"], r["AUC, forest"], r["primary with the mean return"]
                line += (f"  vol forest {f(vf['all'])} / {f(vf['before'])} / {f(vf['after'])}"
                         f"  base {f(a[BASELINE])} -> +mean {f(a[WITH_MEAN])}"
                         f"  ΔAUC topology | +mean {d['mean']:+.3f} ± {d['sd']:.3f}{'  win' if d['win'] else ''}")
            print(line)
        print(f"  wins of the primary with the mean return: "
              f"{s['wins of the primary with the mean return (counts for nothing)']} of "
              f"{s['scorable counted episodes']} (counts for nothing)")
        for score, p in s.get("prediction", {}).items():
            print(f"  prediction, {score}: holds = {p['holds']}  " + "; ".join(
                f"{ep}: {f(v['before'])} / {f(v['after'])}" for ep, v in p["per_episode"].items()))
    if "reproduces results/plan.json" in res:
        print(f"\nreproduces results/plan.json: {res['reproduces results/plan.json']}")
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
        res["reproduces results/plan.json"] = check_against_plan(res)
    out = Path(a.out or ("results/direction_diagnostic.json" if frozen
                         else "results/direction_diagnostic_dry_run.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    report(res)
    print(f"[ok] {out}")
