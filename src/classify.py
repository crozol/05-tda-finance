"""Clasificador sobre features topológicos: régimen normal vs. crisis."""

from __future__ import annotations

import numpy as np


def train_classifier(X: np.ndarray, y: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    clf = RandomForestClassifier(n_estimators=300, random_state=0)
    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    clf.fit(X, y)
    return clf, scores
