"""The spectral surrogates of scripts/spectrum_check.py, checked without giotto-tda: the total
length of the minimum spanning tree of the days, which is the sum of the deaths of the
H0 diagram, stands in for the topology."""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wilcoxon

from scripts.spectrum_check import one_factor, spectral_surrogates, uniform_frames
from src.persistence import standardize_windows


def mst_length(window: np.ndarray) -> float:
    return float(minimum_spanning_tree(squareform(pdist(window))).sum())


def test_frames_are_orthonormal_and_orthogonal_to_the_ones():
    v = uniform_frames(np.random.default_rng(0), 5)
    for f in v:
        np.testing.assert_allclose(f.T @ f, np.eye(7), atol=1e-10)
        np.testing.assert_allclose(f.T @ np.ones(40), 0, atol=1e-10)


def test_a_surrogate_keeps_the_spectrum_and_the_gram_eigenvalues():
    rng = np.random.default_rng(1)
    z = standardize_windows(rng.normal(size=(3, 40, 7)) @ np.linalg.cholesky(one_factor(0.5)).T)
    lam, s = spectral_surrogates(z, rng)
    for w, l, ws in zip(z, lam, s):
        np.testing.assert_allclose(np.linalg.eigvalsh(w.T @ w / 40), l, atol=1e-10)
        np.testing.assert_allclose(np.sort(np.linalg.eigvalsh(ws.T @ ws / 40)), l, atol=1e-10)
        np.testing.assert_allclose(np.linalg.eigvalsh(w @ w.T)[-7:], np.linalg.eigvalsh(ws @ ws.T)[-7:],
                                   atol=1e-8)


def _paired_p(windows, rng):
    z = standardize_windows(windows)
    _, s = spectral_surrogates(z, rng)
    return wilcoxon([mst_length(a) - mst_length(b) for a, b in zip(z, s)]).pvalue


def test_gaussian_windows_and_their_surrogates_are_exchangeable():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(400, 40, 7)) @ np.linalg.cholesky(one_factor(0.7)).T * np.linspace(0.5, 3, 7) - 0.3
    assert _paired_p(x, rng) > 0.01


def test_heavy_tails_are_not_in_the_spectrum():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(400, 40, 7)) @ np.linalg.cholesky(one_factor(0.3)).T
    x = x * np.sqrt(1 / rng.chisquare(3, size=(400, 40, 1)))
    assert _paired_p(x, rng) < 1e-6
