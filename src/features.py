"""Convertir diagramas de persistencia en vectores de features para ML clásico."""

from __future__ import annotations

import numpy as np


def persistence_entropy(diagrams: np.ndarray) -> np.ndarray:
    from gtda.diagrams import PersistenceEntropy

    return PersistenceEntropy().fit_transform(diagrams)


def persistence_landscape(diagrams: np.ndarray, n_layers: int = 5, n_bins: int = 100) -> np.ndarray:
    from gtda.diagrams import PersistenceLandscape

    pl = PersistenceLandscape(n_layers=n_layers, n_bins=n_bins)
    return pl.fit_transform(diagrams)


def stack_features(*feature_matrices: np.ndarray) -> np.ndarray:
    return np.concatenate([f.reshape(f.shape[0], -1) for f in feature_matrices], axis=1)
