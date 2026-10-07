"""Event-rate models with exposure.

Each model learns the rate `events / km` with the trip distance as the sample weight, which is
the standard Poisson set-up with an exposure offset. The expected count of a trip is
`rate x distance_km`. Preprocessing lives inside the model, so it is fitted on training trips only.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ("baseline", "glm", "hgb")


class RateModel(RegressorMixin, BaseEstimator):
    """`baseline` (one fleet rate), `glm` (Poisson GLM) or `hgb` (gradient boosting, Poisson loss)."""

    def __init__(self, kind: str = "glm", alpha: float = 1.0, learning_rate: float = 0.05,
                 max_iter: int = 300, max_leaf_nodes: int = 15, min_samples_leaf: int = 20, random_state: int = 0):
        self.kind = kind
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def _make(self):
        if self.kind == "glm":
            return Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
                    ("scale", StandardScaler()),
                    ("poisson", PoissonRegressor(alpha=self.alpha, max_iter=3000)),
                ]
            )
        if self.kind == "hgb":
            return HistGradientBoostingRegressor(
                loss="poisson", learning_rate=self.learning_rate, max_iter=self.max_iter,
                max_leaf_nodes=self.max_leaf_nodes, min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
        raise ValueError(f"unknown model {self.kind!r}; choose from {MODEL_NAMES}")

    def fit(self, X, counts, exposure=None):
        counts = np.asarray(counts, dtype=float)
        exposure = np.ones_like(counts) if exposure is None else np.asarray(exposure, dtype=float)
        if np.any(exposure <= 0):
            raise ValueError("exposure must be more than 0")
        if np.any(counts < 0):
            raise ValueError("counts must be 0 or more")
        self.fleet_rate_ = float(counts.sum() / exposure.sum())
        self.n_features_in_ = np.asarray(X).shape[1]
        if self.kind == "baseline":
            self.model_ = None
            return self
        rate = counts / exposure
        model = self._make()
        if self.kind == "glm":
            model.fit(np.asarray(X, dtype=float), rate, poisson__sample_weight=exposure)
        else:
            model.fit(np.asarray(X, dtype=float), rate, sample_weight=exposure)
        self.model_ = model
        return self

    def predict_rate(self, X) -> np.ndarray:
        n = np.asarray(X).shape[0]
        if self.model_ is None:
            return np.full(n, self.fleet_rate_)
        return np.clip(self.model_.predict(np.asarray(X, dtype=float)), 1e-9, None)

    def predict_count(self, X, exposure) -> np.ndarray:
        return self.predict_rate(X) * np.asarray(exposure, dtype=float)

    def predict(self, X):
        return self.predict_rate(X)


def make_model(name: str, seed: int = 0) -> RateModel:
    if name not in MODEL_NAMES:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")
    return RateModel(kind=name, random_state=seed)
