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


def total_persistence_norms(diagrams: np.ndarray, dims: tuple[int, ...] = (0, 1),
                            orders: tuple[int, ...] = (1, 2)) -> np.ndarray:
    """The landscape norms of ``landscape_norms``, summed over all layers, in closed form: one column
    per order p and homology dimension, in that order.

    Lemma. Summing over all layers, sum_k lambda_k(x)^p = sum_a f_a(x)^p, since the layers at x are
    the tent values f_a(x) of the points of the diagram in decreasing order. A tent of half-height
    h = (d - b)/2 integrates to 2 h^(p+1)/(p+1), so

        ||lambda||_p^p = sum_a (d_a - b_a)^(p+1) / (2^p (p + 1)),

    a total persistence of order p + 1 (Cohen-Steiner, Edelsbrunner, Harer & Mileyko, 2010). It
    needs no grid: on raw returns, whose scale changes tenfold between calm and crisis, a grid
    common to every window resolves the small diagrams of calm windows with a handful of points.
    Points padded onto the diagonal have zero persistence and add nothing."""
    out = []
    for p in orders:
        for q in dims:
            pers = np.where(diagrams[:, :, 2] == q, diagrams[:, :, 1] - diagrams[:, :, 0], 0.0)
            out.append((np.sum(pers ** (p + 1), axis=1) / (2 ** p * (p + 1))) ** (1 / p))
    return np.stack(out, axis=1)


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


def correlation_spectrum(windows: np.ndarray) -> np.ndarray:
    """All eigenvalues of the correlation matrix of each window, in decreasing
    order, divided by the number of assets (each row sums to 1; the first
    column is ``leading_eigenvalue_share``). Shape (n_windows, n_assets).

    This is the complete co-movement control for topology computed on
    standardised windows. If the days of a window are independent Gaussian
    draws with a common distribution, the standardised window is a uniformly
    random rotation, within the subspace orthogonal to the vector of ones, of
    a matrix fixed by its correlation matrix R; the pairwise distances between
    its days, all that Vietoris-Rips persistence uses, then depend on R only
    through its eigenvalues. Topology can add to the spectrum only through
    structure a Gaussian does not have."""
    d = windows.shape[2]
    out = np.empty((windows.shape[0], d))
    for i, w in enumerate(windows):
        out[i] = np.linalg.eigvalsh(np.corrcoef(w, rowvar=False))[::-1] / d
    return out


def covariance_spectrum(windows: np.ndarray) -> np.ndarray:
    """All eigenvalues of the sample covariance matrix of each window (raw returns, divisor
    n - 1), in decreasing order. Shape (n_windows, n_assets).

    The complete Gaussian control for topology computed on raw, unstandardised windows, as
    Gidea & Katz (2018) compute it. For independent Gaussian days, the centred window is a
    uniformly random frame times the square root of its sample covariance, so the distances
    between its days depend on everything else only through these eigenvalues
    (scripts/spectrum_check.py states and checks the standardised case, whose control is
    ``correlation_spectrum``)."""
    out = np.empty((windows.shape[0], windows.shape[2]))
    for i, w in enumerate(windows):
        out[i] = np.linalg.eigvalsh(np.cov(w, rowvar=False))[::-1]
    return out


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


def mardia_kurtosis(windows: np.ndarray) -> np.ndarray:
    """Mardia's (1970) multivariate kurtosis of each window,
    b_2 = mean_t (z_t^T S^-1 z_t)^2, with z_t the standardised days and S their
    covariance (divisor n). Shape (n_windows, 1). It is affine invariant, so
    standardising changes nothing; for n independent Gaussian days in d
    dimensions its mean is d(d + 2)(n - 1)/(n + 1), about 59.9 for n = 40 and
    d = 7.

    The control for what the correlation spectrum leaves out when days are
    elliptical. For independent elliptical days, a standardised day is
    z_t ≈ r_t R^(1/2) u_t, with u_t uniform on the sphere and independent of
    the radius r_t, up to the sampling error of the standardisation: the
    distances between days, all that Vietoris-Rips persistence uses, then
    depend on R only through its eigenvalues and otherwise on the law of r_t,
    whose leading summary is b_2 (it estimates the mean of r_t^4 when the mean
    of r_t^2 is d). A window that mixes calm and turbulent days also has a
    larger b_2."""
    from .persistence import standardize_windows

    z = standardize_windows(windows)
    out = np.empty((len(z), 1))
    for i, w in enumerate(z):
        s_inv = np.linalg.inv(np.cov(w, rowvar=False, bias=True))
        out[i, 0] = np.mean(np.einsum("ij,jk,ik->i", w, s_inv, w) ** 2)
    return out
