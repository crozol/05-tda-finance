"""Replication of Gidea & Katz (2018), descriptive (docs/early-warning-plan.md, last sections).

Their setting: daily log returns of the S&P 500, Dow Jones, Nasdaq and Russell 2000; sliding windows
of 50 and 100 days with a step of one day, each a cloud of points in R^4; Vietoris-Rips persistence
in dimension 1; the L1 and L2 norms of the persistence landscape summed over all layers (here in
closed form, src.features.total_persistence_norms). On rolling windows of 500 days of each norm
series: the variance, the lag-1 autocorrelation and the mean spectral density at low frequencies;
Kendall's tau of each statistic over the 250 trading days before 2000-03-10 and 2008-09-15.

Here the data start on 1992-01-02, the first day Yahoo has all four indices (they start in December
1987), and end on 2016-12-08 as theirs. The low-frequency band is not given in the paper: the
periodogram of the demeaned 500-day segment is averaged over the frequencies up to 0.05 cycles per
day (periods of 20 days or more). The same statistics are computed on the realised volatility of the
same windows, and for every statistic the share of all 250-day stretches whose tau is at least the
one found before each crash: how often the "warning" appears anyway.

    python -m scripts.gidea_katz            # -> results/gidea_katz.json, figures/gidea_katz.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import log_returns
from src.early_warning import load_frozen_indices
from src.features import realized_volatility, total_persistence_norms
from src.persistence import persistence_diagrams, sliding_windows

END = "2016-12-08"
CRASHES = {"dot-com": "2000-03-10", "Lehman": "2008-09-15"}
WINDOWS = (50, 100)
ROLL, BEFORE, F_MAX = 500, 250, 0.05
REPORTED_TAU = {"dot-com": 0.89, "Lehman": 1.00}       # their Fig. 11: spectral density, L1, w = 50


def norm_series(returns: pd.DataFrame, w: int) -> pd.DataFrame:
    """L1 and L2 norms of the H1 landscape and realised volatility of every window, dated by its last day."""
    windows, end = sliding_windows(returns.to_numpy(), w, 1)
    norms = total_persistence_norms(persistence_diagrams(windows, (1,)), dims=(1,))
    return pd.DataFrame({"L1": norms[:, 0], "L2": norms[:, 1], "volatility": realized_volatility(windows)[:, 0]},
                        index=returns.index[end])


def low_frequency_power(x: np.ndarray, f_max: float = F_MAX) -> float:
    from scipy.signal import periodogram

    f, pxx = periodogram(x - x.mean(), fs=1.0, detrend=False)
    return float(pxx[(f > 0) & (f <= f_max)].mean())


def rolling_statistics(s: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "variance": s.rolling(ROLL).var(),
        "autocorrelation": s.rolling(ROLL).corr(s.shift(1)),
        "spectral density": s.rolling(ROLL).apply(low_frequency_power, raw=True),
    })


def kendall_before(stat: pd.Series, date: str) -> float:
    from scipy.stats import kendalltau

    seg = stat[stat.index < pd.Timestamp(date)].dropna().iloc[-BEFORE:]
    return float(kendalltau(np.arange(len(seg)), seg.to_numpy()).statistic)


def base_rate(stat: pd.Series, tau: float, step: int = 5) -> float:
    """Share of the 250-day stretches (ends every ``step`` days) whose Kendall tau is at least ``tau``."""
    from scipy.stats import kendalltau

    v = stat.dropna().to_numpy()
    taus = [kendalltau(np.arange(BEFORE), v[e - BEFORE:e]).statistic for e in range(BEFORE, len(v) + 1, step)]
    return float(np.mean(np.asarray(taus) >= tau - 1e-12))


def run() -> dict:
    r = log_returns(load_frozen_indices()).loc[:END]
    out = {"first_day": str(r.index[0].date()), "last_day": str(r.index[-1].date()), "windows": {}}
    series = {}
    for w in WINDOWS:
        s = norm_series(r, w)
        series[w] = s
        res = {"correlation with volatility (Spearman)": {
            k: float(s[k].rank().corr(s["volatility"].rank())) for k in ("L1", "L2")}, "statistics": {}}
        for col in ("L1", "L2", "volatility"):
            stats = rolling_statistics(s[col])
            for name in stats:
                key = f"{col} · {name}"
                res["statistics"][key] = {}
                for crash, date in CRASHES.items():
                    tau = kendall_before(stats[name], date)
                    res["statistics"][key][crash] = {"tau": tau, "base rate": base_rate(stats[name], tau)}
        out["windows"][str(w)] = res
    # the band is not given in the paper: tau of the L1 spectral density for other bands (descriptive)
    out["band sensitivity, L1, w = 50"] = {
        str(fm): {crash: kendall_before(series[50]["L1"].rolling(ROLL).apply(
            lambda x, fm=fm: low_frequency_power(x, fm), raw=True), date) for crash, date in CRASHES.items()}
        for fm in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)}
    out["reported by Gidea & Katz"] = {"spectral density of L1, w = 50": REPORTED_TAU}
    return out, series


def plot(res: dict, series: dict, out: str = "figures/gidea_katz.png") -> None:
    import matplotlib.pyplot as plt

    from scripts.make_figures import AMBER, CYAN, FG_1, PINK, PURPLE, style

    style()
    s = series[50]
    fig = plt.figure(figsize=(12, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.25)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(s.index, s["L1"] / s["L1"].max(), color=PURPLE, lw=0.8, label="L¹ norm of the H₁ landscape (50-day windows)")
    ax.plot(s.index, s["volatility"] / s["volatility"].max(), color=CYAN, lw=0.8, alpha=0.75,
            label="realised volatility of the same windows")
    for crash, date in CRASHES.items():
        d = pd.Timestamp(date)
        ax.axvline(d, color=AMBER, lw=1)
        ax.axvspan(s.index[s.index.get_indexer([d], method="nearest")[0] - BEFORE], d, color=AMBER, alpha=0.12, lw=0)
        ax.text(d, 1.02, f" {crash}", color=AMBER, fontsize=9, va="bottom")
    rho = res["windows"]["50"]["correlation with volatility (Spearman)"]["L1"]
    ax.set_title(f"Landscape norm and volatility (Spearman ρ = {rho:.2f}); the 250 days before each crash shaded",
                 loc="left", pad=16)
    ax.set_ylabel("divided by its maximum")
    ax.legend(loc="upper left")
    st = res["windows"]["50"]["statistics"]
    names = [("variance", "variance"), ("spectral density", "low-freq. spectral density"),
             ("autocorrelation", "lag-1 autocorrelation")]
    for j, crash in enumerate(CRASHES):
        ax = fig.add_subplot(gs[1, j])
        y = np.arange(len(names))
        for dy, (col, color, label) in zip((-0.2, 0.0, 0.2), (("L1", PURPLE, "L¹ norm"), ("L2", PINK, "L² norm"),
                                                              ("volatility", CYAN, "volatility"))):
            taus = [st[f"{col} · {n}"][crash]["tau"] for n, _ in names]
            rates = [st[f"{col} · {n}"][crash]["base rate"] for n, _ in names]
            ax.barh(y + dy, taus, height=0.18, color=color, alpha=0.85, label=label)
            for yy, t, b in zip(y + dy, taus, rates):
                ax.text(t + (0.02 if t >= 0 else -0.02), yy, f"{b:.0%}", va="center", fontsize=7.5, color=FG_1,
                        ha="left" if t >= 0 else "right")
        if crash in REPORTED_TAU:
            ax.plot(REPORTED_TAU[crash], 1 - 0.2, marker="*", ms=12, color=AMBER, ls="",
                    label="Gidea & Katz, L¹ spectral density")
        ax.axvline(0, color=FG_1, lw=0.8)
        ax.set_xlim(-1.15, 1.25)
        ax.set_yticks(y, [l for _, l in names] if j == 0 else [""] * len(names))
        ax.invert_yaxis()
        ax.set_xlabel("Kendall τ over the 250 days before the crash")
        ax.set_title(f"{crash} ({CRASHES[crash]})", loc="left")
        if j == 1:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.03), fontsize=8.5)
    fig.text(0.06, -0.035, "labels: share of all 250-day stretches of 1992–2016 with a τ at least as high "
             "(how often the same 'warning' appears anyway)", fontsize=8.5, color=FG_1)
    fig.suptitle("Replicating Gidea & Katz (2018) on 1992–2016", x=0.06, ha="left", y=0.99,
                 fontsize=13, fontweight="bold")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/gidea_katz.json")
    a = ap.parse_args()
    res, series = run()
    Path(a.out).write_text(json.dumps(res, indent=1))
    for w, r in res["windows"].items():
        print(f"w = {w}: Spearman with volatility {r['correlation with volatility (Spearman)']}")
        for k, v in r["statistics"].items():
            print(f"   {k:32s} " + "  ".join(f"{c}: τ {x['tau']:+.2f} (base {x['base rate']:.0%})" for c, x in v.items()))
    plot(res, series)
    print(f"[ok] {a.out}")
