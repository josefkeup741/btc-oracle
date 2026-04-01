#!/usr/bin/env python3
"""
BTC Oracle - Prediction Runner
================================
Generates a prediction using the latest data and trained models.

Usage:
    python run_predict.py          # Print prediction to stdout
    python run_predict.py --json   # Output as JSON (for integrations)
"""
import sys
import json
import argparse
import logging

from config import LOG_FORMAT, LOG_LEVEL
from data.store import Store
from model.predict import predict

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)


def main():
    parser = argparse.ArgumentParser(description="BTC Oracle Prediction")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    store = Store()
    result = predict(store)

    if result is None:
        print("❌ Prediction failed. Check that models are trained and data is collected.")
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
