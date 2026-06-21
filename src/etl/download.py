"""ETL layer — fetch the investable universe and produce a clean price/return panel.

Run:
    python -m src.etl.download                 # full universe, 2007 -> today
    python -m src.etl.download --start 2010-01-01 --end 2024-12-31

Outputs (parquet, wide format, DatetimeIndex x tickers):
    data/raw/prices_raw.parquet     adjusted close exactly as downloaded
    data/processed/prices.parquet   aligned + gap-filled price panel
    data/processed/returns.parquet  simple daily returns
    data/processed/coverage.csv     per-ticker data-quality report

Missing-data policy
-------------------
Different exchanges keep different holiday calendars, so a naive outer-join
leaves scattered NaNs. We:
  1. align every ticker on the union of observed trading days,
  2. forward-fill short gaps only (<= MAX_FFILL_DAYS) — a stale holiday quote is
     fine, but we never invent weeks of data,
  3. drop each ticker's leading NaNs (pre-inception), then
  4. trim to the longest common window so the optimiser sees a rectangular panel.
Everything is reported in coverage.csv rather than silently dropped.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date

import pandas as pd

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("etl")

# Forward-fill at most one trading week to bridge cross-exchange holidays.
MAX_FFILL_DAYS = 5


def download_prices(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Download adjusted close for every ticker as a wide DataFrame."""
    import yfinance as yf  # deferred: keeps cleaning logic importable without the dep

    end = end or date.today().isoformat()
    logger.info("Downloading %d tickers from %s to %s", len(tickers), start, end)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,  # adjusted close -> total-return-aware prices
        progress=False,
        group_by="column",
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data — check connectivity/tickers.")

    # With multiple tickers, columns are a (field, ticker) MultiIndex.
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.reindex(columns=tickers)  # stable column order; missing -> all-NaN

    missing = [t for t in tickers if prices[t].notna().sum() == 0]
    if missing:
        logger.warning("No data returned for: %s", ", ".join(missing))

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "date"
    return prices.sort_index()


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Align on the union of trading days, bridge short gaps, trim to common window."""
    # 1. Union calendar of all observed trading days.
    panel = prices.reindex(prices.index.unique().sort_values())

    # 2. Forward-fill short gaps only (holidays / single missing prints).
    panel = panel.ffill(limit=MAX_FFILL_DAYS)

    # 3. & 4. Trim to the longest window where every ticker has data.
    first_valid = panel.apply(lambda s: s.first_valid_index()).max()
    last_valid = panel.apply(lambda s: s.last_valid_index()).min()
    panel = panel.loc[first_valid:last_valid]

    remaining = int(panel.isna().sum().sum())
    if remaining:
        logger.warning("%d residual NaNs after trim; back-filling within window.", remaining)
        panel = panel.bfill(limit=MAX_FFILL_DAYS)

    logger.info(
        "Clean panel: %d rows x %d tickers (%s -> %s)",
        len(panel), panel.shape[1],
        panel.index.min().date(), panel.index.max().date(),
    )
    return panel


def coverage_report(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Per-ticker data-quality summary for the README / sanity checks."""
    meta = config.asset_metadata()
    rows = []
    for t in raw.columns:
        s = raw[t]
        rows.append({
            "ticker": t,
            "name": meta[t].name if t in meta else "",
            "asset_class": meta[t].asset_class if t in meta else "",
            "first_obs": s.first_valid_index(),
            "last_obs": s.last_valid_index(),
            "n_obs": int(s.notna().sum()),
            "pct_missing_in_window": round(
                100 * clean[t].isna().mean(), 3
            ) if t in clean.columns else 100.0,
        })
    return pd.DataFrame(rows).sort_values("first_obs").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download & clean the ETF universe.")
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=config.END_DATE)
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = download_prices(config.TICKERS, args.start, args.end)
    raw.to_parquet(config.RAW_PRICES_PATH)
    logger.info("Saved raw prices -> %s", config.RAW_PRICES_PATH)

    clean = clean_prices(raw)
    returns = clean.pct_change().dropna(how="all")

    clean.to_parquet(config.PRICES_PATH)
    returns.to_parquet(config.RETURNS_PATH)
    logger.info("Saved clean prices  -> %s", config.PRICES_PATH)
    logger.info("Saved daily returns -> %s", config.RETURNS_PATH)

    coverage = coverage_report(raw, clean)
    coverage.to_csv(config.COVERAGE_PATH, index=False)
    logger.info("Saved coverage      -> %s", config.COVERAGE_PATH)
    logger.info("\n%s", coverage.to_string(index=False))


if __name__ == "__main__":
    main()
