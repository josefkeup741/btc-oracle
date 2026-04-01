# BTC Oracle

Multi-signal Bitcoin price prediction system using XGBoost. Combines options market data, sentiment indicators, on-chain whale activity, prediction market signals, and technical indicators to forecast 7-day BTC price direction and magnitude.

## How It Works

Every 4 hours, collectors pull data from 7 free sources and store it in a local SQLite database. A feature engineering pipeline builds 25 model features from the raw data. XGBoost models (classifier + regressor) are trained with walk-forward validation to predict whether BTC will be higher or lower in 7 days, and by how much.

### Data Sources (all free, no paid APIs)

| Source | What It Provides | API |
|--------|-----------------|-----|
| Deribit | BTC options OI, put/call ratio, weighted breakevens, consensus price, near/far-term skew, max pain | Deribit public API |
| IBIT (BlackRock ETF) | ETF options flow, put/call ratio, institutional consensus | yfinance |
| Fear & Greed Index | Retail sentiment (0-100) | alternative.me |
| Whale Activity | Large BTC transactions, exchange wallet balance tracking | Blockchain.com + Blockchair |
| Polymarket | Prediction market probabilities for BTC price targets | Polymarket CLOB API |
| BTC Price | OHLCV candles | CoinGecko |
| Technical Indicators | RSI, EMA ratio, ATR, MACD histogram, Bollinger %B, OBV slope | Computed from price data |

### Features (25 total)

| Category | Features |
|----------|----------|
| Sentiment (3) | FnG value, 7-day average, 24h delta |
| Deribit Options (5) | Put/call ratio, consensus bias, near-term skew, far-term skew, skew divergence |
| IBIT Options (2) | Consensus bias, Deribit-IBIT divergence |
| Whale Activity (3) | Net exchange flow, whale volume, 7-day flow trend |
| Prediction Markets (2) | Primary probability, probability delta |
| Technicals (6) | EMA ratio, RSI, ATR, OBV slope, Bollinger %B, MACD histogram |
| Price-derived (3) | 24h returns, 7-day returns, 7-day volatility |

## Quick Start

### 1. Install

```bash
git clone https://github.com/josefkeup741/btc-oracle.git
cd btc-oracle
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Import historical data and train

```bash
# If you have a historical BTC + Fear & Greed CSV:
python run_train.py --import-csv btc_with_fgi_4h.csv

# Or backfill from free APIs:
python run_train.py --backfill

# Or just check database status:
python run_train.py --status
```

### 3. Collect live data

```bash
# Single collection run:
python run_collect.py

# Set up cron for every 4 hours:
crontab -e
# Add: 0 */4 * * * cd /path/to/btc-oracle && /path/to/venv/bin/python run_collect.py >> logs/collect.log 2>&1
```

### 4. Generate predictions

```bash
python run_predict.py          # Pretty-printed
python run_predict.py --json   # JSON output
```

## Project Structure

```
btc_oracle/
├── config.py                  # All settings in one place
├── data/
│   ├── collectors/
│   │   ├── base.py            # Abstract collector class
│   │   ├── deribit.py         # Deribit BTC options (24/7)
│   │   ├── ibit.py            # IBIT ETF options (market hours)
│   │   ├── fear_greed.py      # Fear & Greed Index (daily)
│   │   ├── whale.py           # On-chain whale tracking (free APIs)
│   │   ├── polymarket.py      # Prediction market probabilities
│   │   ├── price.py           # BTC spot price via CoinGecko
│   │   └── technicals.py      # RSI, MACD, EMA, ATR, BB, OBV
│   ├── store.py               # SQLite time-series storage
│   └── features.py            # Feature engineering (25 features)
├── model/
│   ├── targets.py             # Label generation (direction + magnitude)
│   ├── train.py               # XGBoost walk-forward training
│   └── predict.py             # Live inference
├── run_collect.py             # Cron entry point: collect all data
├── run_train.py               # Import data + train models
├── run_predict.py             # Generate predictions
└── requirements.txt
```

## Model Architecture

Two XGBoost models trained on the same feature set:

- **Direction Model** (XGBClassifier): Predicts probability of BTC being higher in 7 days
- **Magnitude Model** (XGBRegressor): Predicts expected % change over 7 days

Training uses **walk-forward validation** (expanding window, 1-month test folds) to avoid look-ahead bias. The model only ever sees past data when making predictions.

### Baseline Results (FnG + technicals only, no options/whale data yet)

```
Samples: 12,855  |  Features: 24
Walk-forward folds: 65

DIRECTION MODEL (7-day):
  Accuracy:  51.6%  (baseline: 53.4%)

TOP FEATURES:
  1. fng_7d_avg          (0.129)
  2. volatility_7d       (0.109)
  3. ema_ratio           (0.104)
  4. atr_14              (0.102)
  5. returns_7d          (0.100)
```

The baseline model with only backward-looking indicators performs at random — which is expected and validates the thesis that forward-looking signals (options flow, whale activity, prediction markets) are needed. Accuracy should improve as those signals accumulate.

## Deployment

Designed to run on a free/cheap cloud VM. Currently deployed on Azure for Students (B2ats_v2, free tier).

### Cron Schedule

```
# Collect data every 4 hours
0 */4 * * * cd ~/btc_oracle && ~/btc_oracle/venv/bin/python run_collect.py >> logs/collect.log 2>&1

# Retrain weekly (optional)
0 3 * * 0 cd ~/btc_oracle && ~/btc_oracle/venv/bin/python run_train.py >> logs/train.log 2>&1
```

## Roadmap

- [ ] Accumulate 4+ weeks of Deribit options data and retrain
- [ ] Add 24h and 72h prediction horizons
- [ ] Hyperparameter tuning with Optuna
- [ ] Improve whale collector (add more exchange addresses, handle rate limits)
- [ ] Improve Polymarket collector (BTC markets may not always be active)
- [ ] Dashboard for visualizing predictions vs. actuals
- [ ] Paper trading log to track live accuracy

## License

MIT
