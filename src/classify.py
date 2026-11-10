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


class SeedAverage:
    """The mean predicted probability of one model per seed: with random
    forests, a forest of ``len(seeds)`` times as many trees. Used where the
    analysis plan needs one score per window rather than one per seed (the
    alarm rule)."""

    def __init__(self, make_model=make_random_forest, seeds=range(10)):
        self.make_model, self.seeds = make_model, list(seeds)

    def fit(self, X, y):
        self.models_ = [self.make_model(s).fit(X, y) for s in self.seeds]
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models_], axis=0)


def compare_feature_sets(feature_sets: dict[str, np.ndarray], y: np.ndarray,
                         splits: list, make_model=make_random_forest) -> dict:
    """Evaluate every feature set on the same folds. Typical keys: "volatility",
    "leading eigenvalue", "volatility + eigenvalue", "topology",
    "topology + baselines"."""
    return {name: evaluate(make_model, X, y, splits) for name, X in feature_sets.items()}
