#!/usr/bin/env python3
"""
BTC Oracle - Paper Trading Log
================================
The single train/test split can't give a trustworthy accuracy because 80 days
spans only ~2 independent 7-day windows. This module instead accumulates a real,
dated, out-of-sample track record:

  - RECORD: each run logs the model's current 7-day prediction (direction,
    probability, magnitude, confidence) plus the BTC price at that moment.
  - RESOLVE: predictions whose 7-day target has passed are checked against the
    actual price and marked correct / incorrect.
  - REPORT: running hit rate, hit rate by confidence, hit rate when the two
    models agree, magnitude error, and a simulated directional return.

Run once per day via cron. Consecutive daily predictions overlap (their 7-day
windows share 6 of 7 days), so they are NOT statistically independent — the
report notes this. Over a couple of months across different regimes, the hit
rate becomes the cleanest signal of whether the system actually works.

Usage:
    python run_paper_trade.py            # resolve matured + record new + print stats
    python run_paper_trade.py --notify   # also send the stats to Telegram
    python run_paper_trade.py --stats     # just print stats, don't record a new one
"""
import os
import sys
import sqlite3
import argparse
import logging
from datetime import datetime, timezone, timedelta

import pandas as pd

from config import DB_PATH, PREDICTION_HORIZON_DAYS
from data.store import Store
from model.predict import predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("paper_trade")

RESOLVE_TOLERANCE_HOURS = 8  # how far after target_time we'll accept a price row


def _conn():
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA busy_timeout=5000")
    return c


def init_table():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                prediction_time TEXT PRIMARY KEY,
                btc_price_at_prediction REAL,
                predicted_direction TEXT,
                direction_probability REAL,
                predicted_magnitude REAL,
                confidence TEXT,
                models_agree INTEGER,
                target_time TEXT,
                resolved INTEGER DEFAULT 0,
                actual_price_at_target REAL,
                actual_direction TEXT,
                actual_magnitude REAL,
                correct INTEGER
            )
        """)
        c.commit()


def record_prediction(store: Store) -> bool:
    """Generate the current prediction and log it as an unresolved paper trade."""
    pred = predict(store)
    if pred is None:
        logger.warning("No prediction produced; nothing recorded.")
        return False

    price_row = store.latest("btc_price")
    if not price_row:
        logger.warning("No price data; nothing recorded.")
        return False

    btc_price = float(price_row["close"])
    # Anchor the prediction to the latest price row's timestamp so resolution
    # lines up with the 4-hour collection grid.
    anchor_ts = pd.to_datetime(price_row["timestamp"], format="mixed", utc=True)
    target_ts = anchor_ts + timedelta(days=PREDICTION_HORIZON_DAYS)

    pred_time = anchor_ts.isoformat()

    with _conn() as c:
        # Don't double-record for the same anchor timestamp
        existing = c.execute(
            "SELECT 1 FROM paper_trades WHERE prediction_time = ?", (pred_time,)
        ).fetchone()
        if existing:
            logger.info(f"Prediction for {pred_time} already logged; skipping record.")
            return False

        c.execute("""
            INSERT INTO paper_trades
            (prediction_time, btc_price_at_prediction, predicted_direction,
             direction_probability, predicted_magnitude, confidence,
             models_agree, target_time, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
        """, (
            pred_time, btc_price, pred["direction"],
            float(pred["direction_probability"]), float(pred["magnitude_pct"]),
            pred["confidence"], 1 if pred["models_agree"] else 0,
            target_ts.isoformat(),
        ))
        c.commit()

    logger.info(
        f"Recorded: {pred['direction']} ({pred['direction_probability']:.0%}) "
        f"{pred['magnitude_pct']:+.1f}% from ${btc_price:,.0f}, target {target_ts.date()}"
    )
    return True


def resolve_matured(store: Store) -> int:
    """Resolve predictions whose target time has passed using actual price."""
    now = datetime.now(timezone.utc)

    prices = store.query("btc_price")
    if prices.empty:
        return 0
    prices = prices.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], format="mixed", utc=True)
    prices = prices.sort_values("timestamp").reset_index(drop=True)

    resolved_count = 0
    with _conn() as c:
        rows = c.execute(
            "SELECT prediction_time, btc_price_at_prediction, predicted_direction, "
            "predicted_magnitude, target_time FROM paper_trades WHERE resolved = 0"
        ).fetchall()

        for (pred_time, price_at_pred, pred_dir, pred_mag, target_time) in rows:
            target_ts = pd.to_datetime(target_time, format="mixed", utc=True)
            if target_ts > now:
                continue  # not matured yet

            # Find the first price row at or after the target time
            future = prices[prices["timestamp"] >= target_ts]
            if future.empty:
                continue
            nearest = future.iloc[0]
            # Guard against resolving against a price far past the target (data gap)
            if (nearest["timestamp"] - target_ts) > timedelta(hours=RESOLVE_TOLERANCE_HOURS):
                # If there's also a row just before, accept the closest within tolerance
                before = prices[prices["timestamp"] <= target_ts]
                if not before.empty and (target_ts - before.iloc[-1]["timestamp"]) <= timedelta(hours=RESOLVE_TOLERANCE_HOURS):
                    nearest = before.iloc[-1]
                else:
                    continue

            actual_price = float(nearest["close"])
            actual_dir = "UP" if actual_price > price_at_pred else "DOWN"
            actual_mag = ((actual_price - price_at_pred) / price_at_pred) * 100
            correct = 1 if actual_dir == pred_dir else 0

            c.execute("""
                UPDATE paper_trades
                SET resolved = 1, actual_price_at_target = ?, actual_direction = ?,
                    actual_magnitude = ?, correct = ?
                WHERE prediction_time = ?
            """, (actual_price, actual_dir, actual_mag, correct, pred_time))
            resolved_count += 1

        c.commit()

    if resolved_count:
        logger.info(f"Resolved {resolved_count} matured prediction(s).")
    return resolved_count


def get_stats() -> dict:
    """Compute track-record statistics from resolved paper trades."""
    with _conn() as c:
        df = pd.read_sql_query("SELECT * FROM paper_trades", c)

    total = len(df)
    pending = int((df["resolved"] == 0).sum()) if total else 0
    resolved = df[df["resolved"] == 1].copy() if total else df

    stats = {"total": total, "pending": pending, "resolved": len(resolved)}
    if len(resolved) == 0:
        return stats

    stats["accuracy"] = resolved["correct"].mean()
    # Baseline: always guessing the realized majority direction
    up_frac = (resolved["actual_direction"] == "UP").mean()
    stats["baseline"] = max(up_frac, 1 - up_frac)
    stats["mag_mae"] = (resolved["predicted_magnitude"] - resolved["actual_magnitude"]).abs().mean()

    # Accuracy by confidence
    stats["by_confidence"] = {}
    for conf in ["HIGH", "MEDIUM", "LOW"]:
        sub = resolved[resolved["confidence"] == conf]
        if len(sub):
            stats["by_confidence"][conf] = (sub["correct"].mean(), len(sub))

    # Accuracy when models agree vs disagree
    agree = resolved[resolved["models_agree"] == 1]
    disagree = resolved[resolved["models_agree"] == 0]
    if len(agree):
        stats["agree"] = (agree["correct"].mean(), len(agree))
    if len(disagree):
        stats["disagree"] = (disagree["correct"].mean(), len(disagree))

    # Simulated directional return: +mag if predicted UP, -mag if predicted DOWN.
    # (Overlapping windows, no compounding — an edge indicator, not a real P&L.)
    signs = resolved["predicted_direction"].map({"UP": 1, "DOWN": -1})
    stats["directional_return"] = float((signs * resolved["actual_magnitude"]).sum())
    stats["avg_return_per_trade"] = float((signs * resolved["actual_magnitude"]).mean())

    return stats


def format_stats(stats: dict) -> str:
    lines = []
    lines.append("📈 BTC Oracle Paper Trading")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"Predictions logged: {stats['total']}")
    lines.append(f"Resolved: {stats['resolved']}  |  Pending: {stats['pending']}")

    if stats["resolved"] == 0:
        lines.append("\nNo resolved predictions yet — check back after the first 7-day window matures.")
        return "\n".join(lines)

    acc = stats["accuracy"]
    base = stats["baseline"]
    delta = acc - base
    lines.append("")
    lines.append(f"Hit rate: {acc:.1%}  (baseline {base:.1%}, {delta:+.1%})")
    lines.append(f"Magnitude MAE: {stats['mag_mae']:.2f}%")

    if "by_confidence" in stats and stats["by_confidence"]:
        lines.append("\nBy confidence:")
        for conf, (a, n) in stats["by_confidence"].items():
            lines.append(f"  {conf}: {a:.0%} ({n})")

    if "agree" in stats:
        a, n = stats["agree"]
        lines.append(f"\nModels agree: {a:.0%} ({n})")
    if "disagree" in stats:
        a, n = stats["disagree"]
        lines.append(f"Models disagree: {a:.0%} ({n})")

    dr = stats["directional_return"]
    apt = stats["avg_return_per_trade"]
    lines.append("")
    lines.append(f"Simulated directional edge: {dr:+.1f}% total, {apt:+.2f}%/trade")

    if stats["resolved"] < 15:
        lines.append("\n⚠️ Small sample — not yet conclusive. Overlapping daily")
        lines.append("windows mean these aren't fully independent. Keep accumulating.")

    return "\n".join(lines)


def send_telegram(text: str) -> bool:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.warning("Telegram env vars not set; skipping notify.")
        return False
    import requests
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="BTC Oracle Paper Trading")
    parser.add_argument("--notify", action="store_true", help="Send stats to Telegram")
    parser.add_argument("--stats", action="store_true", help="Only print stats; don't record")
    args = parser.parse_args()

    init_table()
    store = Store()

    # Always resolve matured predictions first
    resolve_matured(store)

    # Record a fresh prediction unless --stats
    if not args.stats:
        record_prediction(store)

    stats = get_stats()
    text = format_stats(stats)
    print("\n" + text + "\n")

    if args.notify:
        if send_telegram(text):
            print("✓ Sent to Telegram")
        else:
            print("✗ Telegram send failed")


if __name__ == "__main__":
    main()
