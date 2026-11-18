"""Landscape norms: every layer counts, as in Gidea & Katz (2018, eq. 2.3)."""

import numpy as np
import pytest

pytest.importorskip("gtda")

from src.features import landscape_norms  # noqa: E402


def tent(x, b, d):
    return np.maximum(0.0, np.minimum(x - b, d - x))


def diagram_with_nested_loops():
    """One window: an H0 point (0, 1) and two nested H1 points, (1, 3) and
    (1.5, 2.5). The smaller loop lies under the larger one, so it is the second
    landscape layer, which a first-layer-only norm would miss."""
    return np.array([[[0.0, 1.0, 0.0], [1.0, 3.0, 1.0], [1.5, 2.5, 1.0]]])


def riemann_norm(layers, x, p):
    """The same Riemann sum giotto-tda uses: (Δx Σ_k Σ_j |λ_k(x_j)|^p)^(1/p)."""
    dx = x[1] - x[0]
    return (dx * sum(np.sum(np.abs(lam) ** p) for lam in layers)) ** (1 / p)


def test_the_norm_sums_over_all_layers():
    d = diagram_with_nested_loops()
    out = landscape_norms(d, orders=(1, 2))           # columns: p=1 (H0, H1), p=2 (H0, H1)
    x1 = np.linspace(1.0, 3.0, 100)                   # H1 grid: smallest birth to largest death
    layers = [tent(x1, 1.0, 3.0), tent(x1, 1.5, 2.5)]
    assert out[0, 1] == pytest.approx(riemann_norm(layers, x1, 1), rel=1e-12)
    assert out[0, 3] == pytest.approx(riemann_norm(layers, x1, 2), rel=1e-12)
    # and it differs from the first layer alone, giotto-tda's default
    assert out[0, 3] > riemann_norm(layers[:1], x1, 2) * 1.05


def test_the_norm_matches_the_integral():
    """∫ λ_1² = 2/3 and ∫ λ_2² = 1/12, so ||λ||_2 = sqrt(3/4); ||λ||_1 = 1 + 1/4."""
    out = landscape_norms(diagram_with_nested_loops(), orders=(1, 2))
    assert out[0, 1] == pytest.approx(1.25, rel=1e-3)
    assert out[0, 3] == pytest.approx(np.sqrt(0.75), rel=1e-3)


def test_correlation_spectrum_is_complete_and_extends_the_leading_eigenvalue():
    from src.features import correlation_spectrum, leading_eigenvalue_share

    rng = np.random.default_rng(1)
    w = rng.normal(size=(6, 40, 7)) @ np.linalg.cholesky(0.4 * np.ones((7, 7)) + 0.6 * np.eye(7)).T
    spec = correlation_spectrum(w)
    assert spec.shape == (6, 7)
    np.testing.assert_allclose(spec.sum(axis=1), 1.0)                     # trace of R is 7
    assert np.all(np.diff(spec, axis=1) <= 1e-12)                         # decreasing
    np.testing.assert_allclose(spec[:, :1], leading_eigenvalue_share(w))
    np.testing.assert_allclose(correlation_spectrum(w * np.arange(1, 8)), spec)   # scale-free


def test_the_gram_matrix_of_the_days_is_a_frame_times_the_spectrum():
    """The decomposition behind the primary comparison. A standardised window z (n days, d
    assets) with correlation matrix R = V Λ V^T satisfies z z^T = n q Λ q^T, where
    q = z V Λ^(-1/2) / sqrt(n) has orthonormal columns orthogonal to the vector of ones. The
    distances between days, all that Vietoris-Rips persistence uses, are a function of z z^T.
    For independent Gaussian days the frame q is uniformly distributed given R (the
    distribution is invariant under orthogonal maps of the days that fix the vector of ones),
    so given R those distances depend on it only through Λ. This test checks the
    decomposition."""
    rng = np.random.default_rng(2)
    n, d = 40, 7
    z = rng.normal(size=(n, d))
    z = (z - z.mean(0)) / z.std(0)                                          # a standardised window
    r = z.T @ z / n
    lam, v = np.linalg.eigh(r)
    gram_from_z = z @ z.T
    # the Gram matrix of the days is fixed by an orthonormal frame q and the spectrum alone
    q = z @ v / np.sqrt(n * lam)                                            # orthonormal columns, q ⟂ 1
    np.testing.assert_allclose(q.T @ q, np.eye(d), atol=1e-10)
    np.testing.assert_allclose(q.T @ np.ones(n), 0, atol=1e-10)
    np.testing.assert_allclose(gram_from_z, n * q @ np.diag(lam) @ q.T, atol=1e-9)


def test_mardia_kurtosis_matches_the_formula_and_its_gaussian_mean():
    from src.features import mardia_kurtosis

    rng = np.random.default_rng(3)
    n, d = 40, 7
    w = rng.normal(size=(1000, n, d)) @ np.linalg.cholesky(0.5 * np.ones((d, d)) + 0.5 * np.eye(d)).T
    b2 = mardia_kurtosis(w)
    assert b2.shape == (1000, 1)
    # the definition, on the raw window: centred days and their covariance with divisor n
    x = w[0] - w[0].mean(0)
    s = x.T @ x / n
    by_hand = np.mean([(xt @ np.linalg.solve(s, xt)) ** 2 for xt in x])
    assert b2[0, 0] == pytest.approx(by_hand, rel=1e-9)
    # affine invariant: rescaling and shifting the assets changes nothing
    np.testing.assert_allclose(mardia_kurtosis(w[:20] * np.arange(1, 8) + 3.0), b2[:20], rtol=1e-9)
    # Gaussian days: E b_2 = d(d + 2)(n - 1)/(n + 1) = 59.93; sd of the mean over 1000 windows ~0.11
    assert b2.mean() == pytest.approx(d * (d + 2) * (n - 1) / (n + 1), abs=0.5)


def test_mardia_kurtosis_sees_heavier_tails_at_the_same_correlation():
    from src.features import mardia_kurtosis

    rng = np.random.default_rng(4)
    g = rng.normal(size=(400, 40, 7))
    t = g * np.sqrt(3 / rng.chisquare(3, size=(400, 40, 1)))      # multivariate t, 3 degrees of freedom
    assert mardia_kurtosis(t).mean() > mardia_kurtosis(g).mean() + 20


def test_asking_for_more_layers_changes_nothing():
    """The number of layers used is enough: requesting more only adds zeros."""
    from gtda.diagrams import Amplitude
    from gtda.homology import VietorisRipsPersistence

    rng = np.random.default_rng(0)
    d = VietorisRipsPersistence(homology_dimensions=[0, 1]).fit_transform(rng.normal(size=(5, 40, 7)))
    out = landscape_norms(d, orders=(2,))
    more = Amplitude(metric="landscape", metric_params={"p": 2, "n_layers": 200}).fit_transform(d)
    np.testing.assert_allclose(out, more, rtol=1e-12)
