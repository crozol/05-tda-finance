"""Early warning: the windows that end shortly before the peak of a large drawdown.

The second question of the project (analysis plan: docs/early-warning-plan.md). Gidea & Katz
(2018) report that persistence landscapes of four US indices change in the months before the
2000 and 2008 crashes. Here a window is a positive when it ends in the 60 trading days before
the peak of a drawdown of the S&P 500 of at least 15 %, and a negative when it ends far from
every peak and every trough; the drawdown itself and the rebound after it are left out, so the
sign of the returns, which the features ignore, does not decide the label.

What is known when a window ends. A positive window ends before its peak: that a peak follows
within 60 trading days is known only later, which is the point of a prediction. A negative
window ends at least 120 trading days from every peak and trough, before and after it, so its
label too is fixed by events up to 120 trading days after its last day. The model never sees
those events as features; under the walk-forward scheme the labels of the last training windows
use them (look-ahead in the labels, not in the features), which is why that scheme is reported,
not counted.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd

INDICES = ("^GSPC", "^DJI", "^IXIC", "^RUT")       # S&P 500, Dow Jones, Nasdaq, Russell 2000
# Downloaded with yfinance 1.8.0 on 2026-11-24 (src.data.download_prices); Yahoo's Dow Jones
# series starts on 1992-01-02, the first day of the file.
FROZEN_INDICES = Path("data/indices_2026-11-24.csv")
FROZEN_INDICES_SHA256 = "45e86aaa8581ff96e9d4aa2dd0d0218076376963a0ed492f99c8d8675503cbcf"

DROP, LOOKBACK = 0.15, 252      # a peak: a 52-week closing high followed by a fall of 15 % or more
HORIZON, FAR = 60, 120          # positives end 1-60 trading days before a peak; negatives >= 120 from any


def load_frozen_indices(path: str | Path = FROZEN_INDICES,
                        sha256: str = FROZEN_INDICES_SHA256) -> pd.DataFrame:
    from src.data import load_frozen_prices

    return load_frozen_prices(path, sha256)[list(INDICES)]


def drawdown_episodes(close: np.ndarray, drop: float = DROP, lookback: int = LOOKBACK) -> list[tuple[int, int]]:
    """(peak, trough) rows of every large drawdown of a price series.

    A peak is a close that is the highest of the previous ``lookback`` trading days (itself
    included) and is followed by a close at least ``drop`` below it before any higher close. Each
    peak starts an episode, which ends at the lowest close between the peak and the next peak (or
    the end of the series). A rally inside a long bear market does not start a new episode: it
    never sets a 52-week high."""
    x = np.asarray(close, dtype=float)
    n, peaks = len(x), []
    for t in range(n):
        if x[t] < x[max(0, t - lookback):t + 1].max():
            continue
        j = t + 1
        while j < n and x[j] <= x[t]:
            if x[j] <= (1 - drop) * x[t]:
                peaks.append(t)
                break
            j += 1
    return [(t, t + int(np.argmin(x[t:(peaks[k + 1] if k + 1 < len(peaks) else n)])))
            for k, t in enumerate(peaks)]


def window_labels(end_rows: np.ndarray, episodes: list[tuple[int, int]],
                  horizon: int = HORIZON, far: int = FAR) -> np.ndarray:
    """1 for a window whose last day is 1 to ``horizon`` trading days before a peak, 0 for one whose
    last day is at least ``far`` trading days from every peak and every trough and outside every
    episode, -1 (left out) otherwise."""
    end = np.asarray(end_rows)
    y = np.full(len(end), -1)
    far_ok = np.ones(len(end), dtype=bool)
    for peak, trough in episodes:
        far_ok &= (np.abs(end - peak) >= far) & (np.abs(end - trough) >= far)
        far_ok &= ~((end >= peak) & (end <= trough))
    y[far_ok] = 0
    for peak, _ in episodes:
        y[(end >= peak - horizon) & (end <= peak - 1)] = 1
    return y


def nearest_peak(end_rows: np.ndarray, episodes: list[tuple[int, int]]) -> np.ndarray:
    """For each window, the index of the episode whose peak is nearest to its last day: the
    held-out block of that episode. Blocks are contiguous stretches of windows."""
    peaks = np.array([p for p, _ in episodes])
    return np.argmin(np.abs(np.asarray(end_rows)[:, None] - peaks[None, :]), axis=1)


def leave_one_episode_out(block: np.ndarray, y: np.ndarray, gap: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """One split per episode: the test set is the labelled windows (y >= 0) of its block; the
    training set is the labelled windows of the other blocks, except those within ``gap`` windows
    of the block on either side, so that no training window shares a day with a test window."""
    idx = np.arange(len(y))
    for e in range(int(block.max()) + 1):
        members = idx[block == e]
        lo, hi = members.min(), members.max()
        test = members[y[members] >= 0]
        train = idx[(y >= 0) & ((idx < lo - gap) | (idx > hi + gap))]
        yield train, test


def past_only_episode_splits(block: np.ndarray, y: np.ndarray, gap: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """The same test blocks, trained only on the labelled windows that end before the block, minus
    ``gap`` windows. The first episode has no earlier positive and cannot be scored."""
    idx = np.arange(len(y))
    for e in range(int(block.max()) + 1):
        members = idx[block == e]
        test = members[y[members] >= 0]
        yield idx[(y >= 0) & (idx < members.min() - gap)], test


# ---------------------------------------------------------------------------- synthetic markets

REGIMES = {                     # daily volatility, correlation between the four indices
    "normal": (0.010, 0.80),
    "pre-peak": (0.011, 0.82),  # null and heavy-tailed scenarios; set on synthetic data only (plan)
    "crisis": (0.020, 0.90),
    "rebound": (0.015, 0.85),
}
SCENARIOS = ("null", "heavy-tailed", "tails-only")


def regime_rows(n_days: int, episodes: list[tuple[int, int]], window: int, horizon: int = HORIZON,
                far: int = FAR) -> np.ndarray:
    """The regime of every return row: 'pre-peak' covers every day of a positive window (the
    ``horizon + window - 1`` rows before a peak), 'crisis' runs from peak to trough, 'rebound' the
    ``far`` rows after the trough, 'normal' everything else."""
    r = np.full(n_days, "normal", dtype=object)
    for peak, trough in episodes:
        r[max(0, peak - horizon - window + 1):peak] = "pre-peak"
        r[trough + 1:min(n_days, trough + 1 + far)] = "rebound"
        r[peak:trough + 1] = "crisis"
    return r


def synthetic_indices(seed: int, scenario: str, dates: pd.DatetimeIndex, episodes: list[tuple[int, int]],
                      window: int = 50, n_assets: int = 4, df: float = 3.0) -> pd.DataFrame:
    """Four one-factor return series on the market's trading days, with the market's episodes
    (rows of the returns, that is of ``dates[1:]``).

    null: every regime Gaussian; the pre-peak regime differs from the normal one only in
    volatility and correlation, so topology has nothing to add to the covariance spectrum.
    heavy-tailed: as the null, with the pre-peak days multivariate Student t (``df`` degrees of
    freedom, rescaled to the same covariance). tails-only, the positive control: the pre-peak days
    keep the normal volatility and correlation and only their tails change."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, not {scenario!r}")
    rng = np.random.default_rng(seed)
    n = len(dates) - 1                                   # return rows: one fewer than prices
    reg = regime_rows(n, episodes, window)
    sigma = np.array([REGIMES[k][0] for k in reg])
    rho = np.array([REGIMES[k][1] for k in reg])
    pre = reg == "pre-peak"
    if scenario == "tails-only":
        sigma[pre], rho[pre] = REGIMES["normal"]
    z = (np.sqrt(rho)[:, None] * rng.normal(size=(n, 1))
         + np.sqrt(1 - rho)[:, None] * rng.normal(size=(n, n_assets)))
    w = rng.chisquare(df, size=n)
    if scenario in ("heavy-tailed", "tails-only"):
        z = z * np.where(pre, np.sqrt((df - 2) / w), 1.0)[:, None]
    r = sigma[:, None] * z
    prices = 100 * np.exp(np.vstack([np.zeros((1, n_assets)), np.cumsum(r, axis=0)]))
    return pd.DataFrame(prices, index=dates, columns=[f"S{i}" for i in range(n_assets)])
