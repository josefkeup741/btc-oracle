"""
BTC Oracle Configuration
========================
All API endpoints, thresholds, paths, and model parameters in one place.
"""
import os
from pathlib import Path

# === PATHS ===
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "btc_oracle.db"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# === COLLECTION INTERVAL ===
COLLECT_INTERVAL_HOURS = 4

# === API ENDPOINTS (no keys needed for these) ===
DERIBIT_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
FEAR_GREED_URL = "https://api.alternative.me/fng/"
COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_OHLC_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
COINGECKO_MARKET_CHART_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
POLYMARKET_CLOB_URL = "https://clob.polymarket.com"

# === API KEYS (set these as environment variables) ===
# export WHALE_ALERT_API_KEY="your_key_here"
WHALE_ALERT_API_KEY = os.environ.get("WHALE_ALERT_API_KEY", "")
WHALE_ALERT_URL = "https://api.whale-alert.io/v1/transactions"
WHALE_ALERT_MIN_USD = 500_000  # Only track transactions above $500k

# === IBIT OPTIONS ===
IBIT_TICKER = "IBIT"
IBIT_MIN_OI_THRESHOLD = 2500  # From your original notebook

# === DERIBIT OPTIONS ===
DERIBIT_MIN_OI = 100  # Minimum open interest to include an expiration
DERIBIT_NEAR_TERM_DAYS = 30
DERIBIT_FAR_TERM_DAYS = 90

# === TECHNICAL INDICATORS ===
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
ATR_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
OBV_SLOPE_PERIOD = 5

# === MODEL PARAMETERS ===
PREDICTION_HORIZON_DAYS = 7
PREDICTION_HORIZON_PERIODS = PREDICTION_HORIZON_DAYS * 6  # 4h candles

# Walk-forward validation
WF_TRAIN_MONTHS = 6
WF_TEST_MONTHS = 1

# XGBoost defaults (conservative starting point)
XGB_CLASSIFIER_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42,
    'verbosity': 0,
}

XGB_REGRESSOR_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'eval_metric': 'rmse',
    'random_state': 42,
    'verbosity': 0,
}

EARLY_STOPPING_ROUNDS = 50

# === LOGGING ===
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
LOG_LEVEL = "INFO"
