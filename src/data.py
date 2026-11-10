"""Price download, log returns, and crisis labels."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ("SPY", "QQQ", "XLF", "XLE", "XLK", "XLY", "XLV")

# Crisis episodes used as positive labels. These dates are a modelling choice,
# fixed before any feature is computed and documented here so that they can be
# changed and the analysis rerun; results must be reported for this exact
# list. Each window is labelled by the date of its *last* day, so a window is
# "crisis" only once the crisis has started, never in anticipation of it.
CRISIS_PERIODS: dict[str, tuple[str, str]] = {
    "global financial crisis": ("2008-09-15", "2009-03-09"),   # Lehman to the market bottom
    "euro debt crisis": ("2011-08-01", "2011-10-03"),          # US downgrade selloff to the October 3 low
    "china devaluation": ("2015-08-11", "2015-08-25"),         # yuan devaluation to the August 25 low
    "q4 2018 selloff": ("2018-09-21", "2018-12-24"),           # day after the S&P 500 peak to the December 24 low
    "covid crash": ("2020-02-20", "2020-03-23"),               # day after the S&P 500 peak to the trough
    "2022 drawdown": ("2022-01-04", "2022-10-12"),             # day after the S&P 500 peak to the October low
}


def download_prices(tickers: tuple[str, ...] = DEFAULT_TICKERS, start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return df.dropna(how="all").ffill().dropna()


# Every result uses this copy: yfinance is unofficial and Yahoo revises adjusted
# prices retroactively, so a new download need not match it. Downloaded with
# yfinance 1.8.0 on 2026-11-10, 2005-01-03 to 2026-11-06 (README, analysis plan).
FROZEN_PRICES = Path("data/prices_2026-11-10.csv")
FROZEN_PRICES_SHA256 = "10535bf43cf79624dd9c81f2284d4520e49235981fa25061e9431c7aa1d6ae29"


def freeze_prices(prices: pd.DataFrame, path: str | Path) -> str:
    """Write the prices in the frozen format and return the file's SHA-256."""
    prices.to_csv(path, float_format="%.10g")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen_prices(path: str | Path = FROZEN_PRICES,
                       sha256: str = FROZEN_PRICES_SHA256) -> pd.DataFrame:
    """Read the frozen prices, refusing a file whose SHA-256 differs."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path}: SHA-256 {digest} differs from the frozen {sha256}")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def crisis_labels(dates: pd.DatetimeIndex,
                  periods: dict[str, tuple[str, str]] = CRISIS_PERIODS) -> tuple[np.ndarray, np.ndarray]:
    """For each date, (label, episode): label 1 inside a crisis period, else 0;
    episode is the index of the period in ``periods`` (-1 outside), as needed
    by ``validation.leave_one_crisis_out``."""
    dates = pd.DatetimeIndex(dates)
    episode = np.full(len(dates), -1)
    for k, (start, end) in enumerate(periods.values()):
        episode[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))] = k
    return (episode >= 0).astype(int), episode
