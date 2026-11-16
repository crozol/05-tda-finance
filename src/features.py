"""Window features: topological descriptors and the non-topological baselines
they have to beat."""

from __future__ import annotations

import numpy as np


# ------------------------------- topological -------------------------------

def persistence_entropy(diagrams: np.ndarray) -> np.ndarray:
    from gtda.diagrams import PersistenceEntropy

    return PersistenceEntropy().fit_transform(diagrams)


def persistence_landscape(diagrams: np.ndarray, n_layers: int = 5, n_bins: int = 100) -> np.ndarray:
    from gtda.diagrams import PersistenceLandscape

    pl = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    return pl.fit_transform(diagrams)


def landscape_norms(diagrams: np.ndarray, orders: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """L^p norms of the persistence landscape of each window, the scalar
    summary used by Gidea & Katz (2018) to track the approach of crashes: one
    column per order p and homology dimension, in that order.

    The norm sums over all the layers of the landscape, as in their eq. (2.3):

        ||λ||_p = (Σ_k ||λ_k||_p^p)^(1/p),   ||f||_p = (∫ |f(x)|^p dx)^(1/p),

    where λ_k(x) is the k-th largest of the tent functions
    max(0, min(x - b, d - x)) of the points (b, d) of the diagram. giotto-tda
    keeps only the first layer by default (n_layers = 1). A diagram with n
    points has at most n non-zero layers, and giotto-tda pads every diagram to
    the same number of points per homology dimension, so asking for that many
    layers includes all of them (layers beyond the last non-zero one add
    zeros). The integral is giotto-tda's Riemann sum on its default grid: 100
    points per homology dimension, from the smallest birth to the largest death
    over all the diagrams passed in. The grid uses no label, but its range comes
    from every window passed in."""
    from gtda.diagrams import Amplitude

    dims = diagrams[0, :, 2]
    n_layers = int(max(np.sum(dims == q) for q in np.unique(dims)))
    return np.concatenate(
        [Amplitude(metric="landscape", metric_params={"p": p, "n_layers": n_layers}).fit_transform(diagrams)
         for p in orders], axis=1)


def stack_features(*feature_matrices: np.ndarray) -> np.ndarray:
    return np.concatenate([f.reshape(f.shape[0], -1) for f in feature_matrices], axis=1)


# -------------------------------- baselines --------------------------------
# A crisis raises volatility and makes assets move together. Both are already
# visible without topology, so any topological feature must be compared with
# these two on the same folds.

def realized_volatility(windows: np.ndarray) -> np.ndarray:
    """Mean across assets of the standard deviation of returns in each window
    (raw, unstandardised returns), shape (n_windows, 1)."""
    return windows.std(axis=1, ddof=1).mean(axis=1, keepdims=True)


def leading_eigenvalue_share(windows: np.ndarray) -> np.ndarray:
    """Largest eigenvalue of the correlation matrix of each window divided by
    the number of assets: the fraction of total variance in the first
    principal component, a standard measure of market co-movement
    (the one-component case of the absorption ratio, Kritzman et al. 2011).
    Shape (n_windows, 1)."""
    d = windows.shape[2]
    out = np.empty((windows.shape[0], 1))
    for i, w in enumerate(windows):
        c = np.corrcoef(w, rowvar=False)
        out[i, 0] = np.linalg.eigvalsh(c)[-1] / d
    return out
