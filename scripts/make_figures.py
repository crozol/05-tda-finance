"""Figures of the README, drawn from the frozen prices and the saved results.

Nothing here computes a new statistic on market data: the pipeline figure shows two windows of the
frozen prices, the leakage figure uses synthetic data, and every other figure reads a results file.

    python -m scripts.make_figures                 # every figure, into figures/
    python -m scripts.make_figures --only market   # one of them
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# portfolio palette (the same as the figures of the other projects)
BG_PANEL, BG_AXES = "#0d1220", "#0c101c"
FG_0, FG_1 = "#e7ecf5", "#9aa3b8"
GRID, SPINE = (1.0, 1.0, 1.0, 0.05), (1.0, 1.0, 1.0, 0.20)
PURPLE, CYAN, AMBER, PINK, GREEN = "#7c5cff", "#22d3ee", "#fbbf24", "#f472b6", "#34d399"
SHORT = {"global financial crisis": "2008–09", "euro debt crisis": "2011", "china devaluation": "2015",
         "q4 2018 selloff": "Q4 2018", "covid crash": "2020", "2022 drawdown": "2022"}
OUT = Path("figures")


def style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BG_PANEL, "savefig.facecolor": BG_PANEL, "axes.facecolor": BG_AXES,
        "axes.edgecolor": SPINE, "axes.labelcolor": FG_0, "axes.titlecolor": FG_0,
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10.5, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "xtick.color": FG_1,
        "ytick.color": FG_1, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "text.color": FG_0,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.family": ["DejaVu Sans", "Segoe UI", "Arial"], "savefig.dpi": 170, "savefig.bbox": "tight"})


def _save(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"[ok] {OUT / name}")


def _clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    from scipy.stats import beta

    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


# ------------------------------------------------------------------ pipeline

def _tents(points: np.ndarray, x: np.ndarray, n_layers: int) -> np.ndarray:
    """Persistence landscape layers lambda_1..lambda_k of the (birth, death) points on x."""
    if len(points) == 0:
        return np.zeros((n_layers, len(x)))
    t = np.maximum(0.0, np.minimum(x[None, :] - points[:, :1], points[:, 1:2] - x[None, :]))
    t = -np.sort(-t, axis=0)
    out = np.zeros((n_layers, len(x)))
    out[:min(n_layers, len(t))] = t[:n_layers]
    return out


def pipeline() -> None:
    """A calm and a crisis window of the frozen prices: the standardised days on their first two
    principal axes, the persistence diagram, and the landscape of the loops (H1)."""
    import matplotlib.pyplot as plt
    from gtda.homology import VietorisRipsPersistence

    from src.data import load_frozen_prices, log_returns
    from src.features import leading_eigenvalue_share, realized_volatility
    from src.persistence import sliding_windows, standardize_windows

    r = log_returns(load_frozen_prices())
    windows, end = sliding_windows(r.to_numpy(), 40, 5)
    dates = r.index[end]
    vol = realized_volatility(windows)[:, 0]
    in_2017 = np.where(dates.year == 2017)[0]
    in_2008 = np.where((dates >= "2008-09-15") & (dates <= "2009-03-09"))[0]
    picks = {"calm": in_2017[np.argmin(vol[in_2017])], "crisis": in_2008[np.argmax(vol[in_2008])]}
    z = standardize_windows(windows[list(picks.values())])
    diagrams = VietorisRipsPersistence(homology_dimensions=[0, 1]).fit_transform(z)
    eig = leading_eigenvalue_share(windows[list(picks.values())])[:, 0]

    style()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), gridspec_kw={"wspace": 0.28, "hspace": 0.42})
    dmax = max(d[:, 1].max() for d in diagrams) * 1.05
    loops = np.concatenate([d[(d[:, 2] == 1) & (d[:, 1] > d[:, 0])][:, :2] for d in diagrams])
    x = np.linspace(loops.min() - 0.15, loops.max() + 0.15, 2000)
    for row, (label, i) in enumerate(picks.items()):
        w, dg = z[row], diagrams[row]
        color = CYAN if label == "calm" else PINK
        start = r.index[end[i] - 39].date()
        u, s, _ = np.linalg.svd(w - w.mean(0), full_matrices=False)
        pc = u[:, :2] * s[:2]
        ax = axes[row, 0]
        ax.scatter(pc[:, 0], pc[:, 1], s=22, color=color, alpha=0.85, edgecolor="none")
        ax.set_aspect("equal", "datalim")
        ax.set_title(f"{label}: {start} → {dates[i].date()}", loc="left")
        ax.set_xlabel("first principal axis")
        ax.set_ylabel("second principal axis")
        ax.text(0.02, 0.02, f"volatility {100 * vol[i]:.2f} % / day\nleading eigenvalue share {eig[row]:.2f}",
                transform=ax.transAxes, fontsize=8.5, color=FG_1, va="bottom")
        h0, h1 = dg[dg[:, 2] == 0][:, :2], dg[dg[:, 2] == 1][:, :2]
        h0, h1 = h0[h0[:, 1] > h0[:, 0]], h1[h1[:, 1] > h1[:, 0]]
        ax = axes[row, 1]
        ax.plot([0, dmax], [0, dmax], color=FG_1, lw=0.8, ls="--", alpha=0.6)
        ax.scatter(h0[:, 0], h0[:, 1], s=18, color=FG_1, alpha=0.8, label="H₀ (components)")
        ax.scatter(h1[:, 0], h1[:, 1], s=26, color=color, label="H₁ (loops)", zorder=3)
        ax.set_xlim(0, dmax)
        ax.set_ylim(0, dmax)
        ax.set_xlabel("birth (distance scale)")
        ax.set_ylabel("death")
        ax.set_title("persistence diagram", loc="left")
        ax.legend(loc="lower right")
        ax = axes[row, 2]
        layers = _tents(h1, x, 5)
        for k, lam in enumerate(layers):
            ax.plot(x, lam, color=color, alpha=1.0 - 0.17 * k, lw=1.4)
        dx = x[1] - x[0]
        ax.text(0.03, 0.95, f"‖λ‖₁ = {dx * layers.sum():.2f}\n{len(h1)} loops", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color=FG_1)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("distance scale")
        ax.set_title("H₁ landscape, layers λ₁…λ₅", loc="left")
    fig.suptitle("From a window of returns to a number: the TDA pipeline on two windows of market data",
                 x=0.06, ha="left", y=0.99, fontsize=13, fontweight="bold")
    _save(fig, "pipeline.png")


# ------------------------------------------------------------------ leakage

def leakage(n_reps: int = 50) -> None:
    """The negative control of tests/test_validation.py, repeated: a feature unrelated to the label
    but slowly varying in time, scored under shuffled k-fold and under purged walk-forward."""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier

    from src.persistence import sliding_windows
    from src.validation import evaluate, purge_gap, purged_walk_forward

    knn = lambda: KNeighborsClassifier(1)                                 # noqa: E731
    res = {"shuffled 5-fold": [], "purged walk-forward": []}
    for seed in range(n_reps):
        rng = np.random.default_rng(seed)
        T = 3000
        walk = np.cumsum(rng.normal(size=(T, 3)), axis=0)
        phase = rng.uniform(0, 2 * np.pi)
        regime = (np.sin(np.arange(T) / 150.0 + phase) > 0.6).astype(int)
        windows, end = sliding_windows(walk, 40, 5)
        X, y = windows.mean(axis=1), regime[end]
        kf = list(KFold(5, shuffle=True, random_state=seed).split(X))
        res["shuffled 5-fold"].append(evaluate(knn, X, y, kf)["mean"]["roc_auc"])
        pw = list(purged_walk_forward(len(y), 5, purge_gap(40, 5)))
        res["purged walk-forward"].append(evaluate(knn, X, y, pw)["mean"]["roc_auc"])
    style()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    rng = np.random.default_rng(0)
    for k, (name, v) in enumerate(res.items()):
        v = np.asarray(v)
        c = PINK if k == 0 else CYAN
        ax.scatter(v, k + rng.uniform(-0.15, 0.15, len(v)), s=16, color=c, alpha=0.75, edgecolor="none")
        ax.plot([np.median(v)] * 2, [k - 0.28, k + 0.28], color=c, lw=2.2)
        ax.text(np.median(v), k + 0.36, f"median {np.median(v):.2f}", ha="center", fontsize=9, color=c)
    ax.axvline(0.5, color=FG_1, lw=0.8, ls="--", alpha=0.6)
    ax.set_yticks([0, 1], list(res))
    ax.set_ylim(-0.6, 1.7)
    ax.set_xlim(0.2, 1.02)
    ax.set_xlabel("ROC-AUC of a feature unrelated to the label (50 synthetic series)")
    ax.set_title("Overlapping windows leak under shuffled k-fold", loc="left")
    _save(fig, "leakage.png")


# ------------------------------------------------------------------ calibration

def calibration() -> None:
    """How often the whole success criterion passed on synthetic data, before any market data."""
    import matplotlib.pyplot as plt

    c = json.loads(Path("results/calibration.json").read_text())
    rows = [("spectrum + volatility", "primary: topology + spectrum + volatility\nvs spectrum + volatility"),
            ("spectrum", "topology + spectrum vs spectrum"),
            ("spectrum + volatility + kurtosis", "primary, kurtosis on both sides"),
            ("eigenvalue", "topology + leading eigenvalue\nvs eigenvalue (first version)")]
    scen = [("null", "Gaussian crises (null)", FG_1), ("heavy-tailed", "heavy-tailed crises", AMBER),
            ("tails-only", "tails only (positive control)", GREEN)]
    style()
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for r, (key, label) in enumerate(rows):
        for s, (sk, sl, color) in enumerate(scen):
            rate = c[sk][key]["rates"]["criterion"]
            k, n = rate["passes"], c[sk]["n_datasets"]
            lo, hi = _clopper_pearson(k, n)
            y = r + (s - 1) * 0.24
            ax.barh(y, k / n, height=0.22, color=color, alpha=0.8, label=sl if r == 0 else None)
            ax.plot([lo, hi], [y, y], color=FG_0, lw=0.9, alpha=0.7)
            ax.text(hi + 0.012, y, f"{k}/{n}", va="center", fontsize=8.5, color=FG_1)
    ax.axvline(0.19, color=PINK, lw=1.0, ls="--")
    ax.text(0.198, 0.5, "chance bound: 0.19 if every crisis were a coin flip", fontsize=8.5,
            color=PINK, va="center")
    ax.set_yticks(range(len(rows)), [l for _, l in rows], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("share of 20 synthetic datasets in which the whole criterion passes (95 % interval)")
    ax.set_title("The criterion, run on synthetic markets before any market data", loc="left", pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    _save(fig, "calibration.png")


# ------------------------------------------------------------------ market result

def market() -> None:
    """ΔAUC per held-out crisis under both schemes, with the logistic-regression check."""
    import matplotlib.pyplot as plt

    p = json.loads(Path("results/plan.json").read_text())
    style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (scheme, s) in zip(axes, p["schemes"].items()):
        eps = list(s["episodes"])
        for k, name in enumerate(eps):
            e = s["episodes"][name]
            d = e["delta_auc"]
            if d["mean"] is None:
                ax.text(k, 0.0, "not\nscorable", ha="center", va="center", fontsize=8, color=FG_1)
                continue
            color = PURPLE if e["counted"] else FG_1
            ax.errorbar(k, d["mean"], yerr=d["sd"], fmt="o", ms=8, color=color, capsize=4,
                        mfc=color if d["win"] else BG_AXES, mew=1.6, zorder=3)
            if e["delta_auc_logistic"] is not None:
                ax.plot(k + 0.22, e["delta_auc_logistic"], "D", ms=5, color=AMBER, alpha=0.9)
        ax.axhline(0, color=FG_1, lw=0.8, ls="--", alpha=0.6)
        ax.set_xlim(-0.6, len(eps) - 0.4)
        ax.set_xticks(range(len(eps)), [SHORT[e] + ("" if s["episodes"][e]["counted"] else "\n(not counted)")
                                        for e in eps], fontsize=9)
        ax.set_title(f"{scheme}: {s['wins']} of {s['scorable']} wins ({s['min_wins']} needed)", loc="left")
    axes[0].set_ylabel("ΔAUC  (with topology − without)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=PURPLE, mfc=PURPLE, label="random forest, win (mean > spread)"),
               Line2D([], [], marker="o", ls="", color=PURPLE, mfc=BG_AXES, label="random forest, no win"),
               Line2D([], [], marker="D", ls="", color=AMBER, label="logistic regression (check)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Market data: does topology add to volatility and the correlation spectrum?",
                 x=0.06, ha="left", y=1.0, fontsize=13, fontweight="bold")
    _save(fig, "market_delta_auc.png")


# ------------------------------------------------------------------ power

def power() -> None:
    """Power of the statistics added after the result, on synthetic crises, and their market p-values."""
    import matplotlib.pyplot as plt

    pr = json.loads(Path("results/post_result.json").read_text())
    reading = pr["reading"]
    rows = []
    for scheme in ("leave one crisis out", "walk-forward"):
        for stat, v in pr["plan"][scheme]["primary"].items():
            key = f"{scheme} {stat}"
            rows.append((("LOCO" if scheme.startswith("leave") else "WF") + f" · {stat}", v,
                         reading["market_holm"].get(key)))
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    for i, (label, v, holm) in enumerate(rows):
        for dy, (sk, color) in zip((-0.2, 0.0, 0.2), (("heavy-tailed", AMBER), ("tails-only", GREEN),
                                                      ("clayton", PURPLE))):
            k, n = v[f"power {sk}"]["k"], v[f"power {sk}"]["n"]
            lo, hi = _clopper_pearson(k, n)
            a1.plot([lo, hi], [i + dy] * 2, color=color, lw=1, alpha=0.6)
            a1.plot(k / n, i + dy, "o", color=color, ms=6, label=sk if i == 0 else None)
        pm = v["market"]["p"]
        a2.plot(pm, i, "o", color=CYAN, ms=7, label="market p" if i == 0 else None)
        if holm is not None:
            first = not any(r[2] is not None for r in rows[:i])
            a2.plot(holm, i, "s", color=PINK, ms=7, label="Holm-adjusted, counted" if first else None)
    a1.axvline(0.05, color=FG_1, lw=0.8, ls="--", alpha=0.6)
    a1.set_yticks(range(len(rows)), [r[0] + ("  ★" if r[2] is not None else "") for r in rows], fontsize=9)
    a1.invert_yaxis()
    a1.set_xlim(0, 1)
    a1.set_xlabel("power: share of 20 synthetic datasets with p ≤ 0.05")
    a1.set_title("Power on synthetic crises", loc="left")
    a1.legend(loc="lower right")
    a2.axvline(0.05, color=FG_1, lw=0.8, ls="--", alpha=0.6)
    a2.set_xscale("log")
    a2.set_xlim(0.005, 1.5)
    a2.set_xlabel("p-value on market data (Monte Carlo, 99 null datasets)")
    a2.set_title("Market data", loc="left")
    a2.legend(loc="lower left")
    fig.text(0.06, -0.04, "★ the four statistics with power on heavy-tailed crises (≥ 4 of 20), the only ones "
             "the reading fixed before the market data counts", fontsize=8.5, color=FG_1)
    fig.suptitle("Tests added after the result: low power, and no evidence where they count",
                 x=0.06, ha="left", y=1.0, fontsize=13, fontweight="bold")
    _save(fig, "power.png")


# ------------------------------------------------------------------ before / after

def before_after() -> None:
    """Realised volatility of every window of each held-out block: before, inside, after the crisis."""
    import matplotlib.pyplot as plt

    from src.data import CRISIS_PERIODS, crisis_labels, load_frozen_prices, log_returns
    from src.features import realized_volatility
    from src.persistence import sliding_windows

    d = json.loads(Path("results/direction_diagnostic.json").read_text())
    eps = d["schemes"]["leave one crisis out"]["episodes"]
    r = log_returns(load_frozen_prices())
    windows, end = sliding_windows(r.to_numpy(), 40, 5)
    vol = 100 * realized_volatility(windows)[:, 0]
    _, episode = crisis_labels(r.index[end])
    names = list(CRISIS_PERIODS)
    counted = [n for n in names if eps[n]["counted"]]
    style()
    fig, axes = plt.subplots(1, len(counted), figsize=(13, 3.6), sharey=True)
    for ax, name in zip(axes, counted):
        e = names.index(name)
        members = np.where(episode == e)[0]
        lo, hi = members.min() - 12, members.max() + 12
        idx = np.arange(lo, hi + 1)
        x = idx - members.min()
        ax.axvspan(-0.5, len(members) - 0.5, color=AMBER, alpha=0.16, lw=0)
        ax.plot(x, vol[idx], color=CYAN, lw=1.3)
        ax.scatter(x[episode[idx] == e], vol[idx][episode[idx] == e], s=10, color=AMBER, zorder=3)
        raw = eps[name]["volatility, raw"]
        ax.set_title(f"{SHORT[name]}\n", loc="left")
        ax.text(0.0, 1.02, f"AUC before {raw['before']:.2f} · after {raw['after']:.2f}",
                transform=ax.transAxes, va="bottom", fontsize=8.5, family="monospace",
                color=PINK if raw["after"] < 0.5 else FG_1)
    axes[0].set_ylabel("realised volatility (% / day)")
    fig.supxlabel("windows from the start of the crisis (step: 5 trading days); crisis windows in amber",
                  fontsize=10, color=FG_1, y=-0.04)
    fig.suptitle("Volatility separates each crisis from the windows before it, not always from those after",
                 x=0.06, ha="left", y=1.1, fontsize=13, fontweight="bold")
    _save(fig, "before_after.png")


# ------------------------------------------------------------------ 2020 rebound

def rebound() -> None:
    """2008-09 fold: the forest's crisis probability against volatility, with and without the
    rebound after the 2020 trough in its training set."""
    import matplotlib.pyplot as plt

    from src.data import CRISIS_PERIODS, crisis_labels, load_frozen_prices, log_returns
    from src.features import realized_volatility
    from src.persistence import sliding_windows

    h = json.loads(Path("results/rebound_hypothesis.json").read_text())
    grid = 100 * np.asarray(h["grid"])
    r = log_returns(load_frozen_prices())
    windows, end = sliding_windows(r.to_numpy(), 40, 5)
    vol = 100 * realized_volatility(windows)[:, 0]
    dates = r.index[end]
    _, episode = crisis_labels(dates)
    crisis08 = vol[episode == list(CRISIS_PERIODS).index("global financial crisis")]
    reb = vol[np.isin(dates, pd.to_datetime(h["2020 rebound windows"]))]
    style()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for key, label, color in (("plan", "trained as in the plan", PINK),
                              ("without the 2020 rebound", "without the 12 windows after the 2020 trough", CYAN)):
        ax.plot(grid, h["trainings"][key]["curve"], color=color, lw=2, label=label)
    ax.scatter(crisis08, np.full(len(crisis08), -0.05), marker="|", s=120, color=AMBER,
               label="2008–09 crisis windows (held out)")
    ax.scatter(reb, np.full(len(reb), -0.11), marker="|", s=120, color=FG_1,
               label="2020 rebound windows (labelled normal)")
    ax.set_xlim(grid.min(), grid.max())
    ax.set_ylim(-0.16, 1.02)
    ax.set_xlabel("realised volatility of the window (% / day)")
    ax.set_ylabel("forest's crisis probability")
    ax.set_title("2008–09: the forest learned from the 2020 rebound that extreme volatility is normal",
                 loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
    _save(fig, "rebound.png")


FIGURES = {"pipeline": pipeline, "leakage": leakage, "calibration": calibration, "market": market,
           "power": power, "before_after": before_after, "rebound": rebound}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=list(FIGURES), action="append")
    a = p.parse_args()
    for name in a.only or FIGURES:
        FIGURES[name]()
