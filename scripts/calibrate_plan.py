"""Operating characteristics of the success criterion, measured on synthetic data only.

How often does the criterion of the analysis plan pass when topology has nothing to add, and how
often when it has? Every dataset comes from scripts.run_plan.synthetic_prices, with the crisis
periods of src/data.py:

* null: crises change only volatility and correlation, and returns stay Gaussian. Topology on
  standardised windows then has nothing to add to the eigenvalue spectrum of the correlation
  matrix (src.features.correlation_spectrum), whatever it adds to the leading eigenvalue alone;
* heavy-tailed: the same, and crisis returns are also multivariate Student t with 3 degrees of
  freedom at the same correlation, a structure the spectrum does not see.

For each dataset the script applies the win rule of the plan (10 random-forest seeds, a win when
the mean ΔAUC exceeds its standard deviation across seeds, leave-one-crisis-out and the
past-only walk-forward) to the primary comparison, topology + spectrum against the spectrum,
and to the secondary one, topology + eigenvalue against the leading eigenvalue (the primary
comparison before amendment 7). No market data are used. --summarize gives the pass rates with
exact (Clopper-Pearson) 95% intervals.

Usage:
    python -m scripts.calibrate_plan --scenario null --seeds 0 1 2 3
    python -m scripts.calibrate_plan --scenario heavy-tailed --seeds 0 1 2 3
    python -m scripts.calibrate_plan --summarize          # -> results/calibration.json
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np

from scripts.run_plan import (BASELINE, COMBINED, CONTEXT, GAP, LOCO_MIN_WINS, NOT_COUNTED,
                              SECONDARY, SEEDS, WF_MIN_WINS, _num, build_features, criterion,
                              scores_by_episode, synthetic_prices)
from src.classify import make_random_forest
from src.data import CRISIS_PERIODS
from src.validation import leave_one_crisis_out, past_only_crisis_splits

COMPARISONS = {"primary": (COMBINED, BASELINE), "secondary": SECONDARY}
SCENARIOS = {"null": "gaussian", "heavy-tailed": "heavy-tailed"}
SCHEMES = ("leave one crisis out", "walk-forward")


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def criteria(prices, seeds=SEEDS, make_model=make_random_forest) -> dict:
    """The win rule of the plan for both comparisons and both schemes, on one dataset."""
    data = build_features(prices)
    y, episode = data["y"], data["episode"]
    names = list(CRISIS_PERIODS)
    episodes = sorted(set(episode[episode >= 0].tolist()))
    counted = np.array([names[e] not in NOT_COUNTED for e in episodes])
    sets = {k: data["sets"][k] for pair in COMPARISONS.values() for k in pair}
    splits = {"leave one crisis out": list(leave_one_crisis_out(episode, GAP, CONTEXT)),
              "walk-forward": list(past_only_crisis_splits(episode, GAP, CONTEXT))}
    out = {}
    for scheme in SCHEMES:
        rf = scores_by_episode(sets, y, splits[scheme], seeds, make_model)
        need = LOCO_MIN_WINS if scheme == "leave one crisis out" else WF_MIN_WINS
        out[scheme] = {}
        for name, (a, b) in COMPARISONS.items():
            c = criterion(rf[a]["roc_auc"] - rf[b]["roc_auc"], counted)
            out[scheme][name] = {
                "wins": c["wins_counted"], "scorable": c["scorable_counted"],
                "passes": c["wins_counted"] >= need,
                "delta_auc": {names[e]: _num(c["mean"][k]) for k, e in enumerate(episodes)},
                "sd": {names[e]: _num(c["sd"][k]) for k, e in enumerate(episodes)}}
    return {"schemes": out,
            "success": {name: all(out[s][name]["passes"] for s in SCHEMES) for name in COMPARISONS}}


def main(scenario: str, seeds: list[int], out_dir: str = "results/calibration") -> None:
    d = Path(out_dir) / scenario
    d.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        f = d / f"seed{seed:02d}.json"
        if f.exists():
            print(f"{scenario} seed {seed}: already in {f}, skipped", flush=True)
            continue
        t0 = time.time()
        res = {"scenario": scenario, "seed": seed,
               **criteria(synthetic_prices(seed, crisis=SCENARIOS[scenario]))}
        f.write_text(json.dumps(res, indent=2))
        s = res["schemes"]
        print(f"{scenario} seed {seed}: primary LOCO {s[SCHEMES[0]]['primary']['wins']}/5, "
              f"WF {s[SCHEMES[1]]['primary']['wins']}/4; secondary LOCO "
              f"{s[SCHEMES[0]]['secondary']['wins']}/5, WF {s[SCHEMES[1]]['secondary']['wins']}/4 "
              f"({time.time() - t0:.0f} s)", flush=True)


def summarize(out_dir: str = "results/calibration",
              out: str = "results/calibration.json") -> dict:
    summary = {}
    for scenario in SCENARIOS:
        runs = [json.loads(Path(f).read_text())
                for f in sorted(glob.glob(str(Path(out_dir) / scenario / "seed*.json")))]
        if not runs:
            continue
        n = len(runs)
        summary[scenario] = {"n_datasets": n, "seeds": [r["seed"] for r in runs]}
        for name in COMPARISONS:
            rates = {}
            for label, passed in (
                    ("leave one crisis out", [r["schemes"][SCHEMES[0]][name]["passes"] for r in runs]),
                    ("walk-forward", [r["schemes"][SCHEMES[1]][name]["passes"] for r in runs]),
                    ("criterion", [r["success"][name] for r in runs])):
                k = int(sum(passed))
                rates[label] = {"passes": k, "rate": k / n, "ci95": clopper_pearson(k, n)}
            wins = {s: np.bincount([r["schemes"][s][name]["wins"] for r in runs],
                                   minlength=6).tolist() for s in SCHEMES}
            summary[scenario][name] = {"comparison": " vs ".join(COMPARISONS[name]),
                                       "rates": rates, "wins_histogram": wins}
            c = rates["criterion"]
            print(f"{scenario:12s} {name:9s} ({' vs '.join(COMPARISONS[name])}): criterion passes "
                  f"{c['passes']}/{n} = {c['rate']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]; "
                  f"LOCO {rates['leave one crisis out']['passes']}/{n}, "
                  f"WF {rates['walk-forward']['passes']}/{n}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=2))
    print(f"[ok] {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", choices=list(SCENARIOS), default="null")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--out-dir", default="results/calibration")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    if a.summarize:
        summarize(a.out_dir)
    else:
        main(a.scenario, a.seeds, a.out_dir)
