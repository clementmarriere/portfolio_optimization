"""Couche 1 — walk-forward forecasting engine + evaluation.

Run:
    python -m src.models.walkforward                 # headline + annex models
    python -m src.models.walkforward --models gbm    # a subset

Expanding-window protocol (no leakage)
--------------------------------------
Rebalancing dates are processed in order. At decision date `t`:
  - TRAIN on every sample whose date is strictly before `t`. Those samples'
    targets (month-end -> next month-end returns) are already realised by `t`.
  - PREDICT the forward return for each asset using only features observed at `t`.
The first `TRAIN_MIN_MONTHS` dates are warm-up and produce no forecast.

Outputs:
    data/processed/forecasts.parquet   tidy (date, ticker, model, y_true, y_pred)
    results/metrics/forecast_metrics.csv   one row per model
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.models import baselines, ml  # noqa: F401  (populate the registry)
from src.models.base import get_forecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("forecast")

TARGET = "fwd_ret"


def walk_forward(
    table: pd.DataFrame,
    model_name: str,
    feature_cols: list[str],
    min_train_months: int,
) -> pd.DataFrame:
    """Out-of-sample forecasts for one model over the expanding-window schedule."""
    df = table.reset_index()  # columns: date, ticker, <features>, fwd_ret
    dates = np.sort(df["date"].unique())
    test_dates = dates[min_train_months:]

    out = []
    for t in test_dates:
        train = df[df["date"] < t]
        test = df[df["date"] == t]
        model = get_forecaster(model_name)
        model.fit(train[feature_cols], train[TARGET])
        preds = model.predict(test[feature_cols])
        out.append(pd.DataFrame({
            "date": test["date"].to_numpy(),
            "ticker": test["ticker"].to_numpy(),
            "model": model_name,
            "y_true": test[TARGET].to_numpy(),
            "y_pred": preds,
        }))

    logger.info("[%s] %d forecast dates (%s -> %s)",
                model_name, len(test_dates),
                pd.Timestamp(test_dates[0]).date(), pd.Timestamp(test_dates[-1]).date())
    return pd.concat(out, ignore_index=True)


def evaluate(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Allocation-oriented metrics per model."""
    rows = []
    for name, g in forecasts.groupby("model"):
        err = g["y_pred"] - g["y_true"]
        # Cross-sectional Information Coefficient: rank-corr of pred vs realised
        # within each date, averaged. This is what actually drives ranking-based
        # allocation, far more relevant than raw RMSE.
        ics = (
            g.groupby("date")
            .apply(lambda d: spearmanr(d["y_pred"], d["y_true"]).statistic
                   if d["y_pred"].nunique() > 1 else np.nan, include_groups=False)
            .dropna()
        )
        rows.append({
            "model": name,
            "rmse": float(np.sqrt((err ** 2).mean())),
            "mae": float(err.abs().mean()),
            "ic_mean": float(ics.mean()),
            "ic_ir": float(ics.mean() / ics.std()) if ics.std() > 0 else np.nan,
            "hit_rate": float((np.sign(g["y_pred"]) == np.sign(g["y_true"])).mean()),
            "n_obs": int(len(g)),
        })
    metrics = pd.DataFrame(rows).set_index("model")
    # Headline first, then annex, for readability.
    order = [config.HEADLINE_MODEL] + [m for m in metrics.index if m != config.HEADLINE_MODEL]
    return metrics.loc[[m for m in order if m in metrics.index]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward forecasting.")
    parser.add_argument(
        "--models", nargs="+",
        default=[config.HEADLINE_MODEL, *config.ANNEX_MODELS],
        help="Models to run (first should be the headline).",
    )
    args = parser.parse_args()

    table = pd.read_parquet(config.FEATURES_PATH)
    feature_cols = [c for c in table.columns if c != TARGET]
    logger.info("Loaded %d rows, %d features: %s", len(table), len(feature_cols), feature_cols)

    forecasts = pd.concat(
        [walk_forward(table, m, feature_cols, config.TRAIN_MIN_MONTHS) for m in args.models],
        ignore_index=True,
    )

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(config.FORECASTS_PATH)
    logger.info("Saved forecasts -> %s", config.FORECASTS_PATH)

    metrics = evaluate(forecasts)
    metrics.to_csv(config.FORECAST_METRICS_PATH)
    logger.info("Saved metrics   -> %s", config.FORECAST_METRICS_PATH)
    logger.info("\n%s", metrics.round(4).to_string())


if __name__ == "__main__":
    main()
