"""Persistent homology over sliding windows of multivariate returns."""

from __future__ import annotations

import numpy as np


def sliding_windows(returns: np.ndarray, window: int = 40, step: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """returns shape (T, d) → (windows of shape (n_windows, window, d), index of
    the last row of each window). Labels and dates are attached through that
    last index, so a window only ever uses data up to the day it is labelled."""
    n = (returns.shape[0] - window) // step + 1
    out = np.empty((n, window, returns.shape[1]))
    end = np.empty(n, dtype=int)
    for i in range(n):
        start = i * step
        out[i] = returns[start : start + window]
        end[i] = start + window - 1
    return out, end


def standardize_windows(windows: np.ndarray) -> np.ndarray:
    """Z-score every asset inside every window. Without this, the point cloud
    simply grows with volatility and the persistence diagram mostly re-measures
    it; standardised windows isolate the change in the *shape* of the joint
    distribution, which is what topology is supposed to add."""
    mu = windows.mean(axis=1, keepdims=True)
    sd = windows.std(axis=1, keepdims=True) + 1e-12
    return (windows - mu) / sd


def persistence_diagrams(windows: np.ndarray, homology_dims: tuple[int, ...] = (0, 1)):
    """Vietoris-Rips persistence diagram of each window, treating the window's
    rows (days) as points in R^d."""
    from gtda.homology import VietorisRipsPersistence

    vr = VietorisRipsPersistence(homology_dimensions=list(homology_dims))
    return vr.fit_transform(windows)
