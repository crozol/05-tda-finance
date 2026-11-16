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


def test_asking_for_more_layers_changes_nothing():
    """The number of layers used is enough: requesting more only adds zeros."""
    from gtda.diagrams import Amplitude
    from gtda.homology import VietorisRipsPersistence

    rng = np.random.default_rng(0)
    d = VietorisRipsPersistence(homology_dimensions=[0, 1]).fit_transform(rng.normal(size=(5, 40, 7)))
    out = landscape_norms(d, orders=(2,))
    more = Amplitude(metric="landscape", metric_params={"p": 2, "n_layers": 200}).fit_transform(d)
    np.testing.assert_allclose(out, more, rtol=1e-12)
