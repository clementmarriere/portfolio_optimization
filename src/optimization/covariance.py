"""Covariance estimation for the optimiser — causal, monthly-scaled.

Two estimators:
  - ledoit_wolf_monthly : shrinkage covariance (headline risk model, robust)
  - sample_monthly      : plain sample covariance (the naive Markowitz benchmark)

Both use a trailing window of DAILY returns observed up to (and including) the
decision date, then scale to a monthly horizon to match the monthly forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

TRADING_DAYS_PER_MONTH = 21


def _window(daily_returns: pd.DataFrame, asof, lookback: int, tickers: list[str]) -> pd.DataFrame:
    """Trailing `lookback` daily returns up to `asof`, columns in `tickers` order."""
    win = daily_returns.loc[:asof, tickers].tail(lookback).dropna(how="any")
    if len(win) < 2:
        raise ValueError(f"Not enough return history before {asof} ({len(win)} rows).")
    return win


def ledoit_wolf_monthly(daily_returns, asof, lookback, tickers) -> np.ndarray:
    win = _window(daily_returns, asof, lookback, tickers)
    cov_daily = LedoitWolf().fit(win.to_numpy()).covariance_
    return cov_daily * TRADING_DAYS_PER_MONTH


def sample_monthly(daily_returns, asof, lookback, tickers) -> np.ndarray:
    win = _window(daily_returns, asof, lookback, tickers)
    cov_daily = np.cov(win.to_numpy(), rowvar=False)
    return cov_daily * TRADING_DAYS_PER_MONTH
