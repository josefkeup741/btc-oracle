"""
Telegram Bot for BTC Oracle
=============================
Sends daily prediction summaries and key feature snapshots to Telegram.
Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables.

Usage:
    python run_notify.py              # Send daily summary
    python run_notify.py --test       # Send a test message
"""
import os
import sys
import json
import argparse
import logging
import requests
from datetime import datetime, timezone

from data.store import Store
from model.predict import predict

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"


def send_message(text: str) -> bool:
    """Send a message via Telegram bot."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.error("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
        return False

    try:
        resp = requests.post(
            TELEGRAM_API_URL.format(token=TELEGRAM_BOT_TOKEN),
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
            },
            timeout=10,
        )
        if resp.status_code == 200:
            logger.info("Telegram message sent successfully")
            return True
        else:
            logger.error(f"Telegram API error: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def build_daily_summary(store: Store) -> str:
    """Build the daily summary message with prediction and key features."""
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%B %d, %Y")

    # Get latest data from each source
    price_data = store.latest("btc_price")
    deribit_data = store.latest("deribit_options")
    fng_data = store.latest("fear_greed")
    whale_data = store.latest("whale_activity")
    ibit_data = store.latest("ibit_options")

    # Start building message
    lines = []
    lines.append(f"🔮 <b>BTC Oracle Daily Brief</b>")
    lines.append(f"📅 {date_str}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━")

    # Price info
    if price_data:
        btc_price = price_data.get("close", 0)
        lines.append(f"💰 <b>BTC Price:</b> ${btc_price:,.0f}")

    # Try to get prediction
    try:
        prediction = predict(store)
        if prediction:
            direction = prediction["direction"]
            prob = prediction["direction_probability"]
            magnitude = prediction["magnitude_pct"]
            confidence = prediction["confidence"]
            agree = prediction["models_agree"]

            emoji = "🟢" if direction == "UP" else "🔴"
            arrow = "↑" if direction == "UP" else "↓"

            lines.append("")
            lines.append(f"<b>7-Day Prediction:</b>")
            lines.append(f"{emoji} Direction: <b>{direction}</b> {arrow} ({prob:.0%})")
            lines.append(f"📏 Expected: <b>{magnitude:+.1f}%</b>")
            lines.append(f"🎯 Confidence: {confidence}")
            if not agree:
                lines.append(f"⚠️ Models disagree — treat with caution")
    except Exception as e:
        lines.append(f"\n⚠️ Prediction unavailable: {str(e)[:50]}")

    # Key signals
    lines.append("")
    lines.append("<b>📊 Key Signals:</b>")

    if deribit_data:
        bias_7d = deribit_data.get("consensus_bias_7d")
        bias_30d = deribit_data.get("consensus_bias_30d")
        bias_90d = deribit_data.get("consensus_bias_90d")
        pcr = deribit_data.get("put_call_ratio")

        if bias_7d is not None:
            lines.append(f"  Deribit 7d consensus: {float(bias_7d):+.1f}%")
        if bias_30d is not None:
            lines.append(f"  Deribit 30d consensus: {float(bias_30d):+.1f}%")
        if bias_90d is not None:
            lines.append(f"  Deribit 90d consensus: {float(bias_90d):+.1f}%")
        if pcr is not None:
            lines.append(f"  Put/Call ratio: {float(pcr):.2f}")

    if fng_data:
        fng_val = fng_data.get("fng_value")
        fng_class = fng_data.get("fng_classification", "")
        if fng_val is not None:
            lines.append(f"  Fear & Greed: {int(float(fng_val))}/100 ({fng_class})")

    if ibit_data:
        ibit_bias = ibit_data.get("consensus_bias_pct")
        if ibit_bias is not None:
            lines.append(f"  IBIT consensus: {float(ibit_bias):+.1f}%")

    # Macro data if available
    macro_data = store.latest("macro")
    if macro_data:
        dxy = macro_data.get("dxy")
        oil = macro_data.get("oil_price")
        m2_yoy = macro_data.get("m2_yoy_growth")
        us10y = macro_data.get("us10y_yield")

        lines.append("")
        lines.append("<b>🌍 Macro:</b>")
        if dxy is not None:
            lines.append(f"  DXY: {float(dxy):.1f}")
        if us10y is not None:
            lines.append(f"  10Y Yield: {float(us10y):.2f}%")
        if oil is not None:
            lines.append(f"  Oil: ${float(oil):.1f}")
        if m2_yoy is not None:
            lines.append(f"  M2 YoY: {float(m2_yoy):+.1f}%")

    # Data health
    lines.append("")
    lines.append("<b>🔧 Collection Health:</b>")
    status = store.status()
    for table, info in status.items():
        if info["rows"] > 0:
            lines.append(f"  ✓ {table}: {info['rows']:,} rows")
        else:
            lines.append(f"  ✗ {table}: empty")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="BTC Oracle Telegram Notifications")
    parser.add_argument("--test", action="store_true", help="Send a test message")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    if args.test:
        success = send_message("🔮 BTC Oracle test message — notifications are working!")
        if success:
            print("✓ Test message sent!")
        else:
            print("✗ Failed to send test message. Check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.")
        return

    store = Store()
    summary = build_daily_summary(store)
    print(summary)
    print()

    success = send_message(summary)
    if success:
        print("✓ Daily summary sent to Telegram")
    else:
        print("✗ Failed to send summary")


if __name__ == "__main__":
    main()
