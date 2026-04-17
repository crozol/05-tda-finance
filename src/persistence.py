"""Homología persistente sobre ventanas deslizantes de retornos multivariados."""

from __future__ import annotations

import numpy as np


def sliding_windows(returns: np.ndarray, window: int = 40, step: int = 5) -> np.ndarray:
    """returns shape (T, d) → array (n_windows, window, d)."""
    n = (returns.shape[0] - window) // step + 1
    out = np.empty((n, window, returns.shape[1]))
    for i in range(n):
        start = i * step
        out[i] = returns[start : start + window]
    return out


def persistence_diagrams(windows: np.ndarray, homology_dims: tuple[int, ...] = (0, 1)):
    """Calcula diagramas de persistencia Vietoris-Rips para cada ventana."""
    from gtda.homology import VietorisRipsPersistence

    vr = VietorisRipsPersistence(homology_dimensions=list(homology_dims))
    return vr.fit_transform(windows)
