# BTC Oracle

Multi-signal Bitcoin price prediction system using XGBoost.

## Quick Start

### 1. Install dependencies

```bash
cd btc_oracle
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Import your historical data

```bash
# Import the Kaggle CSV (btc_with_fgi_4h.csv)
python run_train.py --import-csv /path/to/btc_with_fgi_4h.csv --no-train

# Or backfill from free APIs (less historical depth but automated)
python run_train.py --backfill --no-train

# Check what's in the database
python run_train.py --status
```

### 3. Train initial model

```bash
python run_train.py
```

This will:
- Compute technical indicators from stored price data
- Build all 25 features
- Run walk-forward validation
- Train direction (classifier) and magnitude (regressor) models
- Print accuracy metrics and feature importance

### 4. Start collecting live data

```bash
# Single collection run (test it works)
python run_collect.py

# Set up cron for every 4 hours (see Server Setup below)
```

### 5. Generate predictions

```bash
python run_predict.py          # Pretty-printed output
python run_predict.py --json   # JSON output for integrations
```

---

## Server Setup (Oracle Cloud Free Tier)

Your collectors need to run 24/7. Here's how to set that up for $0.

### Step 1: Create an Oracle Cloud account

Go to https://cloud.oracle.com and sign up. The "Always Free" tier includes:
- 1 ARM-based VM (4 cores, 24GB RAM) — way more than you need
- 200GB block storage
- No expiration (unlike AWS/GCP free tiers)

### Step 2: Launch an instance

- Shape: `VM.Standard.A1.Flex` (ARM) — 1 OCPU, 6GB RAM is plenty
- Image: Ubuntu 22.04 or 24.04
- Add your SSH public key

### Step 3: Set up the environment

```bash
# SSH into your instance
ssh ubuntu@<your-instance-ip>

# Install Python and pip
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

# Clone your repo (or scp the files)
git clone <your-repo-url> btc_oracle
cd btc_oracle

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Set up environment variables

```bash
# Add to ~/.bashrc or create .env file
export WHALE_ALERT_API_KEY="your_key_here"
```

### Step 5: Import historical data and train

```bash
# Upload your CSV
scp btc_with_fgi_4h.csv ubuntu@<ip>:~/btc_oracle/

# Import and train
cd ~/btc_oracle
source venv/bin/activate
python run_train.py --import-csv btc_with_fgi_4h.csv
```

### Step 6: Set up cron jobs

```bash
crontab -e
```

Add these lines:

```cron
# Collect data every 4 hours
0 */4 * * * cd /home/ubuntu/btc_oracle && /home/ubuntu/btc_oracle/venv/bin/python run_collect.py >> /home/ubuntu/btc_oracle/logs/collect.log 2>&1

# Retrain model every Sunday at 3am UTC
0 3 * * 0 cd /home/ubuntu/btc_oracle && /home/ubuntu/btc_oracle/venv/bin/python run_train.py >> /home/ubuntu/btc_oracle/logs/train.log 2>&1

# Generate prediction daily at 8am UTC (optional, for logging)
0 8 * * * cd /home/ubuntu/btc_oracle && /home/ubuntu/btc_oracle/venv/bin/python run_predict.py --json >> /home/ubuntu/btc_oracle/logs/predictions.log 2>&1
```

### Step 7: Verify it's running

```bash
# Check cron is active
crontab -l

# Watch the next collection run
tail -f logs/collect.log

# Check database status anytime
source venv/bin/activate && python run_train.py --status
```

---

## Project Structure

```
btc_oracle/
├── config.py                  # All settings in one place
├── data/
│   ├── collectors/
│   │   ├── base.py            # Abstract collector class
│   │   ├── deribit.py         # Deribit BTC options
│   │   ├── ibit.py            # IBIT ETF options
│   │   ├── fear_greed.py      # Fear & Greed Index
│   │   ├── whale.py           # On-chain whale transactions
│   │   ├── polymarket.py      # Prediction market probabilities
│   │   ├── price.py           # BTC spot price (CoinGecko)
│   │   └── technicals.py      # RSI, MACD, EMA, ATR, BB
│   ├── store.py               # SQLite time-series storage
│   └── features.py            # Feature engineering (25 features)
├── model/
│   ├── targets.py             # Label generation (direction + magnitude)
│   ├── train.py               # XGBoost walk-forward training
│   └── predict.py             # Live inference
├── run_collect.py             # Cron: collect data every 4h
├── run_train.py               # Import data + train models
├── run_predict.py             # Generate predictions
├── requirements.txt
└── README.md
```

## Features (25 total)

| # | Feature | Source |
|---|---------|--------|
| 1-3 | FnG value, 7d avg, 24h delta | Fear & Greed API |
| 4-8 | PCR, consensus bias, near/far skew, skew divergence | Deribit Options |
| 9-10 | IBIT consensus bias, options divergence | IBIT Options |
| 11-13 | Net exchange flow, whale volume, 7d flow trend | Whale Alert |
| 14-15 | Prediction market prob, prob delta | Polymarket |
| 16-21 | EMA ratio, RSI, ATR, OBV slope, BB %B, MACD hist | Technicals |
| 22-25 | 24h returns, 7d returns, 7d volatility | Price-derived |

## API Keys Needed

| Service | Cost | Sign up |
|---------|------|---------|
| Whale Alert | Free (10 req/min) | https://whale-alert.io |

All other APIs (Deribit, CoinGecko, Fear & Greed, Polymarket, yfinance) are free with no key required.
