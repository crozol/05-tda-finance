"""Classifier over window features: normal regime vs. crisis.

The classifier is evaluated with the leakage-free splits of
``src/validation.py``, never with shuffled k-fold, and always next to the
baselines, so that any gain can be attributed to the topology rather than to
the volatility or correlation information the topological features also
carry.
"""

from __future__ import annotations

import numpy as np

from .validation import evaluate


def make_random_forest(seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                  class_weight="balanced", random_state=seed)


def make_logistic():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))


def compare_feature_sets(feature_sets: dict[str, np.ndarray], y: np.ndarray,
                         splits: list, make_model=make_random_forest) -> dict:
    """Evaluate every feature set on the same folds. Typical keys: "volatility",
    "leading eigenvalue", "volatility + eigenvalue", "topology",
    "topology + baselines"."""
    return {name: evaluate(make_model, X, y, splits) for name, X in feature_sets.items()}
