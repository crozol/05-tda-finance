"""Descarga de precios y cálculo de retornos log normalizados."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_TICKERS = ("SPY", "QQQ", "XLF", "XLE", "XLK", "XLY", "XLV")


def download_prices(tickers: tuple[str, ...] = DEFAULT_TICKERS, start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return df.dropna(how="all").ffill().dropna()


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()
