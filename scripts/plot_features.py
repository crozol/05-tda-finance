"""Time series of the window features on the frozen market data, with the crisis periods shaded.

A descriptive figure, drawn after the first result (2026-11-19); it tests nothing. One panel per
feature, all on the same time axis: realised volatility and the leading eigenvalue share (two of the
baselines), and the L1 norms of the persistence landscapes in dimensions 0 and 1 (the topology of the
plan's setting: 40-day standardised windows, step 5). Each point is a window, placed at its last day,
the day that labels it.

    python -m scripts.plot_features          # -> figures/features_timeseries.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_plan import build_features
from src.data import CRISIS_PERIODS, load_frozen_prices

# portfolio palette (the same as the figures of the other projects)
BG_PANEL, BG_AXES = "#0d1220", "#0c101c"
FG_0, FG_1 = "#e7ecf5", "#9aa3b8"
GRID, SPINE = (1.0, 1.0, 1.0, 0.05), (1.0, 1.0, 1.0, 0.20)
PURPLE, CYAN, AMBER = "#7c5cff", "#22d3ee", "#fbbf24"
SHORT = {"global financial crisis": "2008–09", "euro debt crisis": "2011", "china devaluation": "2015",
         "q4 2018 selloff": "Q4 2018", "covid crash": "2020", "2022 drawdown": "2022"}


def _style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": BG_PANEL, "savefig.facecolor": BG_PANEL, "axes.facecolor": BG_AXES,
        "axes.edgecolor": SPINE, "axes.labelcolor": FG_0, "axes.titlecolor": FG_0,
        "axes.titleweight": "bold", "axes.titlesize": 13, "axes.labelsize": 10.5, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "xtick.color": FG_1,
        "ytick.color": FG_1, "xtick.labelsize": 10, "ytick.labelsize": 9.5, "text.color": FG_0,
        "font.family": ["DejaVu Sans", "Segoe UI", "Arial"], "savefig.dpi": 170, "savefig.bbox": "tight"})


def main(out: str = "figures/features_timeseries.png") -> None:
    import matplotlib.pyplot as plt

    data = build_features(load_frozen_prices())
    dates = data["returns"].index[data["end"]]
    s = data["sets"]
    topo = s["topology"]                              # columns: L1 (H0, H1), L2 (H0, H1)
    panels = [("realised volatility, daily (%)", 100 * s["volatility"][:, 0], CYAN),
              ("leading eigenvalue share", s["eigenvalue"][:, 0], CYAN),
              ("landscape L¹ norm, H₀", topo[:, 0], PURPLE),
              ("landscape L¹ norm, H₁", topo[:, 1], PURPLE)]
    _style()
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 8.2), sharex=True,
                             gridspec_kw={"hspace": 0.12})
    for ax, (label, y, color) in zip(axes, panels):
        for start, stop in CRISIS_PERIODS.values():
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(stop), color=AMBER, alpha=0.16, lw=0)
        ax.plot(dates, y, color=color, lw=1.1)
        ax.set_ylabel(label, rotation=0, ha="left", va="bottom", fontsize=10, color=FG_1)
        ax.yaxis.set_label_coords(0.0, 1.02)
        ax.margins(x=0.005)
    top = axes[0]
    for name, (start, stop) in CRISIS_PERIODS.items():
        mid = pd.Timestamp(start) + (pd.Timestamp(stop) - pd.Timestamp(start)) / 2
        top.annotate(SHORT[name], (mid, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 14),
                     textcoords="offset points", ha="center", fontsize=9, color=AMBER)
    fig.suptitle("Window features on market data, 2005–2026 (40-day windows, step 5; crises shaded)",
                 x=0.125, ha="left", y=0.99, fontsize=13, fontweight="bold", color=FG_0)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[ok] {out} ({len(dates)} windows, {dates[0].date()} to {dates[-1].date()})")


if __name__ == "__main__":
    main()
