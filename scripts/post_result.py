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

Reading of the rule, fixed on 2026-11-20 after the synthetic runs and before any statistic was
computed on market data. The rule above names no threshold. A statistic has power when its power
on heavy-tailed crises exceeds the nominal size, 0.05, by an exact one-sided binomial test at
level 0.05 (at least 4 of 20 datasets). The market evidence is read on the primary comparison in
the plan's setting, on the statistics with power, with Holm's correction across them; every other
p-value is reported and counts for nothing. These are dLL (5 of 20) and dBrier (5 of 20) under
leave-one-crisis-out, and dLL (7 of 20) and dAUC|vol (4 of 20) under the walk-forward.
(summarize() applies this reading to the power it computes and writes it under "reading".)

Added on 2026-11-20, after the market data, on synthetic data only: the power of the reading as a
whole, the share of the synthetic datasets of each scenario in which Holm's correction across the
family finds an adjusted p-value of at most 0.05 (on the null, each dataset is tested against the
others; a statistic that cannot be computed gives p = 1), written under "reading" as
"power (computed after the market data)"; and --tables, the summary as markdown tables with exact
(Clopper-Pearson) 95 % intervals for the power. Neither changes a statistic, a p-value or the
reading.

Usage:
    python -m scripts.post_result --setting plan --scenario null --seeds 0 1 2
    python -m scripts.post_result --setting plan --market
    python -m scripts.post_result --summarize                    # -> results/post_result.json
    python -m scripts.post_result --tables                       # markdown tables of the summary
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom

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
READING = {"setting": "plan", "pair": "primary", "scenario": "heavy-tailed"}   # fixed 2026-11-20


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


def has_power(k: int, n: int, size: float = ALPHA, level: float = ALPHA) -> bool:
    """Power above the nominal size: exact one-sided binomial test of k detections in n."""
    return k > 0 and binom.sf(k - 1, n, size) <= level


def holm(p: dict[str, float]) -> dict[str, float]:
    """Holm's step-down adjustment of a family of p-values (monotone, capped at 1)."""
    out, run = {}, 0.0
    for i, key in enumerate(sorted(p, key=p.get)):
        run = max(run, min(1.0, (len(p) - i) * p[key]))
        out[key] = run
    return out


def reading(summary: dict) -> dict:
    """The reading of the rule fixed on 2026-11-20 (module docstring), applied to a summary."""
    s, pair, sc = summary.get(READING["setting"], {}), READING["pair"], READING["scenario"]
    family = [f"{scheme} {st}" for scheme in SCHEMES for st in STATS
              if (e := s.get(scheme, {}).get(pair, {}).get(st, {}).get(f"power {sc}"))
              and has_power(e["k"], e["n"])]
    res = {**READING, "family": family, "market_holm": None}
    markets = {f: s[f.rsplit(" ", 1)[0]][pair][f.rsplit(" ", 1)[1]].get("market") for f in family}
    if family and all(m is not None for m in markets.values()):
        res["market_holm"] = holm({f: m["p"] for f, m in markets.items()})
    return res


def reading_power(nulls: list[dict], runs: dict[str, list[dict]], family: list[str],
                  pair: str = READING["pair"]) -> dict:
    """Added 2026-11-20, after the market data (module docstring): in how many synthetic datasets of
    each scenario the reading as a whole, Holm across the family, gives an adjusted p <= ALPHA."""
    def pvalues(r: dict, others: list[dict]) -> dict[str, float]:
        out = {}
        for f in family:
            scheme, st = f.rsplit(" ", 1)
            null = [v for o in others if (v := o["schemes"][scheme][pair]["S"][st]) is not None]
            s = r["schemes"][scheme][pair]["S"][st]
            out[f] = 1.0 if s is None else monte_carlo_p(s, null)
        return out

    res = {}
    for sc, rs in [("null", nulls)] + [(sc, runs.get(sc, [])) for sc in ("heavy-tailed", "tails-only", "clayton")]:
        if not rs or not family:
            continue
        k = sum(min(holm(pvalues(r, nulls[:i] + nulls[i + 1:] if sc == "null" else nulls)).values()) <= ALPHA
                for i, r in enumerate(rs))
        res[sc] = {"k": int(k), "n": len(rs)}
    return res


def exact_interval(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Clopper-Pearson interval for a binomial proportion k / n."""
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def tables(path: str = "results/post_result.json") -> str:
    """The summary as markdown tables: size, power (k/n, exact 95 % interval), market S and p."""
    d = json.loads(Path(path).read_text())
    lines = []
    for setting in SETTINGS:
        if not (s := d.get(setting)):
            continue
        lines += [f"**{setting}** ({s['n_null']} null datasets)", "",
                  "| Scheme | Comparison | Statistic | Size | Heavy-tailed | Tails only | Clayton | Market S | p |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for scheme in SCHEMES:
            for pair in PAIRS:
                for st, e in s.get(scheme, {}).get(pair, {}).items():
                    cells = []
                    for sc in ("heavy-tailed", "tails-only", "clayton"):
                        v = e.get(f"power {sc}")
                        cells.append("{}/{} ({:.2f}–{:.2f})".format(v["k"], v["n"], *exact_interval(v["k"], v["n"]))
                                     if v else "–")
                    m = e.get("market")
                    mk = [f"{m['S']:+.4f}".replace("-", "−"), f"{m['p']:.3f}"] if m else ["–", "–"]
                    name = st.replace("|", "\\|")
                    lines.append(f"| {scheme} | {pair} | {name} | {e['size_leave_one_out']:.2f} | "
                                 + " | ".join(cells + mk) + " |")
        lines.append("")
    return "\n".join(lines)


def summarize(out_dir: str = OUT, out: str = "results/post_result.json") -> dict:
    summary, synthetic = {}, {}
    for setting in SETTINGS:
        root = Path(out_dir) / setting
        runs = {sc: [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(root / sc / "seed*.json")))]
                for sc in SCENARIOS}
        nulls = [r for r in runs["null"] if r["seed"] < N_NULL[setting]]
        if not nulls:
            continue
        synthetic[setting] = (nulls, runs)
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
    read = reading(summary)
    if READING["setting"] in synthetic:
        read["power (computed after the market data)"] = reading_power(*synthetic[READING["setting"]],
                                                                       read["family"])
    Path(out).write_text(json.dumps({**summary, "reading": read}, indent=2))
    for setting, s in summary.items():
        for scheme in SCHEMES:
            for pair in PAIRS:
                for st, e in s.get(scheme, {}).get(pair, {}).items():
                    pw = "; ".join(f"{k.split()[1]} {v['k']}/{v['n']}" for k, v in e.items() if k.startswith("power"))
                    m = e.get("market")
                    mk = f"market S {m['S']:+.4f} p {m['p']:.3f}" if m else "market -"
                    print(f"{setting:15s} {scheme:21s} {pair:8s} {st:8s} size {e['size_leave_one_out']:.2f}; "
                          f"power {pw}; {mk}")
    print(f"reading ({read['setting']}, {read['pair']}; power on {read['scenario']} above the size): "
          + (", ".join(read["family"]) or "no statistic has power")
          + ("; market, Holm-adjusted p: " + ", ".join(f"{k} {v:.3f}" for k, v in read["market_holm"].items())
             if read["market_holm"] else "; market -"))
    if pw := read.get("power (computed after the market data)"):
        print("power of the reading (computed after the market data, synthetic data only): "
              + ", ".join(f"{sc} {v['k']}/{v['n']}" for sc, v in pw.items()))
    print(f"[ok] {out}")
    return {**summary, "reading": read}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setting", choices=list(SETTINGS), default="plan")
    p.add_argument("--scenario", choices=list(SCENARIOS), default="null")
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--market", action="store_true")
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--tables", action="store_true")
    a = p.parse_args()
    if a.tables:
        print(tables())
    elif a.summarize:
        summarize()
    elif a.market:
        run_market(a.setting)
    else:
        run_synthetic(a.setting, a.scenario, a.seeds)
