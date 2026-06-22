"""Annex baselines — deliberately naive references for the headline model.

These are NOT the subject of the project; they exist so the headline model's
edge can be quantified. ARIMA (statsmodels) and the deep sequence models live
in their own modules and are also annex-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster, register


class HistoricalMeanForecaster(Forecaster):
    """Expected return = average realised return seen in training. Constant."""

    name = "historical_mean"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HistoricalMeanForecaster":
        self.mu_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.mu_)


class MovingAverageForecaster(Forecaster):
    """Persistence: next month's return ~ last month's realised return."""

    name = "moving_average"

    def __init__(self, col: str = "ret_1m") -> None:
        self.col = col

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MovingAverageForecaster":
        return self  # nothing to learn

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.col].to_numpy()


register("historical_mean", HistoricalMeanForecaster)
register("moving_average", MovingAverageForecaster)
