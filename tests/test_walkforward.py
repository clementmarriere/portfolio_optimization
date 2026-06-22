"""Tests for the walk-forward engine — the no-leakage guarantee is the key one."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import Forecaster, register
from src.models.walkforward import TARGET, walk_forward


class _TrainPeekForecaster(Forecaster):
    """Records the max training date it ever sees, to detect look-ahead."""

    name = "train_peek"
    seen_max_train_date = None

    def fit(self, X, y):
        # X carries a 'date_marker' column injected by the test.
        m = X["date_marker"].max()
        cls = type(self)
        if cls.seen_max_train_date is None or m > cls.seen_max_train_date:
            cls.seen_max_train_date = m
        return self

    def predict(self, X):
        return np.zeros(len(X))


register("train_peek", _TrainPeekForecaster)


def _table(n_months: int = 24, n_assets: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    rng = np.random.default_rng(1)
    rows = []
    for d in dates:
        for a in range(n_assets):
            rows.append({
                "date": d, "ticker": f"A{a}",
                "feat": rng.normal(), "date_marker": d.value,
                TARGET: rng.normal(),
            })
    return pd.DataFrame(rows).set_index(["date", "ticker"])


def test_warmup_and_forecast_dates():
    table = _table(n_months=24)
    min_train = 6
    fc = walk_forward(table, "historical_mean", ["feat"], min_train)
    forecast_dates = pd.DatetimeIndex(fc["date"].unique()).sort_values()
    all_dates = table.index.get_level_values("date").unique().sort_values()
    # Exactly the dates after the warm-up window are forecast.
    assert len(forecast_dates) == len(all_dates) - min_train
    assert forecast_dates.min() == all_dates[min_train]


def test_no_future_in_training():
    table = _table(n_months=18)
    _TrainPeekForecaster.seen_max_train_date = None
    fc = walk_forward(table, "train_peek", ["feat", "date_marker"], min_train_months=6)
    last_forecast_date = fc["date"].max().value
    # Training never saw a date >= the latest decision date it predicted on.
    assert _TrainPeekForecaster.seen_max_train_date < last_forecast_date
