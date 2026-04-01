#!/usr/bin/env python3
"""
BTC Oracle - Data Collection Runner
=====================================
Run this on a schedule (every 4 hours via cron).
Triggers all collectors and stores results.

Cron example (every 4 hours):
    0 */4 * * * cd /home/user/btc_oracle && /usr/bin/python3 run_collect.py >> logs/collect.log 2>&1
"""
import sys
import logging
from datetime import datetime, timezone

from config import LOG_FORMAT, LOG_LEVEL, LOG_DIR
from data.store import Store
from data.collectors.deribit import DeribitCollector
from data.collectors.ibit import IBITCollector
from data.collectors.fear_greed import FearGreedCollector
from data.collectors.price import PriceCollector
from data.collectors.whale import WhaleCollector
from data.collectors.polymarket import PolymarketCollector
from data.collectors.technicals import TechnicalsCollector

# Set up logging
log_file = LOG_DIR / f"collect_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("run_collect")


def main():
    logger.info("=" * 50)
    logger.info("Starting data collection run")
    logger.info("=" * 50)

    store = Store()
    results = {}

    # Collectors that hit external APIs
    api_collectors = [
        PriceCollector(),
        DeribitCollector(),
        FearGreedCollector(),
        WhaleCollector(),
        PolymarketCollector(),
        IBITCollector(),
    ]

    for collector in api_collectors:
        name = collector.__class__.__name__
        try:
            data = collector.collect_with_timestamp()
            if data is not None:
                ts = data.pop("timestamp")
                store.upsert(collector.table_name, ts, data)
                results[name] = "OK"
                logger.info(f"  ✓ {name}: collected and stored")
            else:
                results[name] = "NO_DATA"
                logger.warning(f"  ✗ {name}: returned no data")
        except Exception as e:
            results[name] = f"ERROR: {e}"
            logger.error(f"  ✗ {name}: {e}", exc_info=True)

    # Technicals collector (reads from store, not an API)
    try:
        tech = TechnicalsCollector(store=store)
        data = tech.collect_with_timestamp()
        if data is not None:
            ts = data.pop("timestamp")
            store.upsert(tech.table_name, ts, data)
            results["TechnicalsCollector"] = "OK"
            logger.info("  ✓ TechnicalsCollector: computed and stored")
        else:
            results["TechnicalsCollector"] = "NO_DATA"
            logger.warning("  ✗ TechnicalsCollector: not enough price history yet")
    except Exception as e:
        results["TechnicalsCollector"] = f"ERROR: {e}"
        logger.error(f"  ✗ TechnicalsCollector: {e}", exc_info=True)

    # Print summary
    logger.info("-" * 50)
    logger.info("Collection Summary:")
    for name, status in results.items():
        icon = "✓" if status == "OK" else "✗"
        logger.info(f"  {icon} {name}: {status}")

    # Print store status
    status = store.status()
    logger.info("\nDatabase Status:")
    for table, info in status.items():
        logger.info(f"  {table}: {info['rows']} rows ({info['earliest']} to {info['latest']})")

    logger.info("Collection run complete\n")


if __name__ == "__main__":
    main()
