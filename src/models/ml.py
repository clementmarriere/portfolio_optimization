"""Headline forecaster — pooled cross-sectional ML.

One model is trained on ALL assets stacked together (panel), so it learns a
shared mapping feature -> expected return from ~16x more samples than any
per-asset model would see. This is the model put forward in the main pipeline;
`gbm` is the default headline (see config.HEADLINE_MODEL).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import Forecaster, register


class PooledGBMForecaster(Forecaster):
    """Gradient-boosted trees on the pooled panel. Shallow + regularised to keep
    variance in check on a small monthly sample."""

    name = "gbm"

    def __init__(self) -> None:
        self.model = HistGradientBoostingRegressor(
            max_depth=3,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PooledGBMForecaster":
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.to_numpy())


class PooledRidgeForecaster(Forecaster):
    """Linear, standardised, strongly regularised — the honest linear annex."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PooledRidgeForecaster":
        self.model.fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.to_numpy())


register("gbm", PooledGBMForecaster)
register("ridge", PooledRidgeForecaster)
