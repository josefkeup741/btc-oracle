#!/usr/bin/env python3
"""
BTC Oracle - Training Runner
==============================
Imports historical data (first run only), then trains XGBoost models.
Run manually or on a weekly cron.

Usage:
    python run_train.py                    # Train with existing store data
    python run_train.py --import-csv       # Import your Kaggle CSV first, then train
    python run_train.py --backfill         # Backfill FnG + price history from APIs, then train
    python run_train.py --status           # Just show database status
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

from config import LOG_FORMAT, LOG_LEVEL, LOG_DIR
from data.store import Store
from data.collectors.fear_greed import FearGreedCollector
from data.collectors.price import PriceCollector
from data.collectors.technicals import TechnicalsCollector
from model.train import train

log_file = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("run_train")


def import_kaggle_csv(store: Store, csv_path: str):
    """
    Import your btc_with_fgi_4h.csv into the store.
    Maps columns to the btc_price and fear_greed tables.
    """
    import pandas as pd

    logger.info(f"Importing {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {len(df)} rows")

    # Import price data
    price_rows = []
    for _, row in df.iterrows():
        price_rows.append({
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": 0,  # Not in this CSV
        })

    store.upsert_many("btc_price", price_rows)
    logger.info(f"  Imported {len(price_rows)} rows into btc_price")

    # Import FnG data (deduplicated to daily since FnG is daily)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    fng_daily = df.groupby("date").first().reset_index()

    fng_rows = []
    for _, row in fng_daily.iterrows():
        fng_rows.append({
            "timestamp": str(row["timestamp"]),
            "fng_value": row["Fear & Greed Index"],
            "fng_classification": row["Fear & Greed Classification"],
        })

    store.upsert_many("fear_greed", fng_rows)
    logger.info(f"  Imported {len(fng_rows)} rows into fear_greed")


def backfill_from_apis(store: Store):
    """Backfill historical data from free APIs."""
    logger.info("Backfilling from APIs...")

    # Fear & Greed history (free API goes back years)
    try:
        fng = FearGreedCollector()
        rows = fng.collect_historical(days=2000)
        if rows:
            store.upsert_many("fear_greed", rows)
            logger.info(f"  Backfilled {len(rows)} FnG entries")
    except Exception as e:
        logger.error(f"  FnG backfill failed: {e}")

    # Price history from CoinGecko (free: max 365 days at daily granularity)
    try:
        price = PriceCollector()
        rows = price.collect_historical(days=365)
        if rows:
            store.upsert_many("btc_price", rows)
            logger.info(f"  Backfilled {len(rows)} price entries")
    except Exception as e:
        logger.error(f"  Price backfill failed: {e}")


def compute_historical_technicals(store: Store):
    """Compute technicals for all historical price data."""
    import pandas as pd

    logger.info("Computing historical technicals...")
    prices = store.query("btc_price")
    if prices.empty or len(prices) < 60:
        logger.warning("Not enough price data for technicals")
        return

    prices = prices.sort_values("timestamp").reset_index(drop=True)
    tech = TechnicalsCollector()
    result = tech.compute_from_df(prices)

    if result is not None and not result.empty:
        rows = []
        for _, row in result.iterrows():
            d = row.to_dict()
            ts = str(d.pop("timestamp"))
            # Skip rows with NaN (early periods before indicators warm up)
            if pd.notna(d.get("rsi_14")):
                d["timestamp"] = ts
                rows.append(d)

        store.upsert_many("technicals", rows)
        logger.info(f"  Computed technicals for {len(rows)} periods")


def print_status(store: Store):
    """Print database status."""
    print("\n" + "=" * 60)
    print("  BTC ORACLE - DATABASE STATUS")
    print("=" * 60)
    status = store.status()
    for table, info in status.items():
        if info["rows"] > 0:
            print(f"  {table:<20s}  {info['rows']:>7,d} rows  ({info['earliest'][:10]} to {info['latest'][:10]})")
        else:
            print(f"  {table:<20s}  {'empty':>7s}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="BTC Oracle Training")
    parser.add_argument("--import-csv", type=str, help="Path to btc_with_fgi_4h.csv to import")
    parser.add_argument("--backfill", action="store_true", help="Backfill from APIs before training")
    parser.add_argument("--status", action="store_true", help="Show database status and exit")
    parser.add_argument("--no-train", action="store_true", help="Import/backfill only, skip training")
    args = parser.parse_args()

    store = Store()

    if args.status:
        print_status(store)
        return

    # Import Kaggle CSV if provided
    if args.import_csv:
        import_kaggle_csv(store, args.import_csv)

    # Backfill from APIs
    if args.backfill:
        backfill_from_apis(store)

    # Always compute technicals from whatever price data exists
    compute_historical_technicals(store)

    print_status(store)

    if args.no_train:
        logger.info("Skipping training (--no-train flag)")
        return

    # Train
    logger.info("Starting model training...")
    result = train(store)

    if "error" in result:
        logger.error(f"Training failed: {result['error']}")
        sys.exit(1)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
