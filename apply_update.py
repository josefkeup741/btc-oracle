"""
Run this on the VM to update store.py and features.py for time-bucketed consensus bias.
Usage: python apply_update.py
"""

def update_store():
    """Add new columns to deribit_options schema in store.py"""
    with open("data/store.py", "r") as f:
        content = f.read()

    old_schema = '''        "deribit_options": {
            "btc_spot": "REAL",
            "total_call_oi": "REAL",
            "total_put_oi": "REAL",
            "put_call_ratio": "REAL",
            "weighted_call_breakeven": "REAL",
            "weighted_put_breakeven": "REAL",
            "consensus_price": "REAL",
            "consensus_bias_pct": "REAL",
            "near_term_skew": "REAL",
            "far_term_skew": "REAL",
            "max_pain_30d": "REAL",
        }'''

    new_schema = '''        "deribit_options": {
            "btc_spot": "REAL",
            "total_call_oi": "REAL",
            "total_put_oi": "REAL",
            "put_call_ratio": "REAL",
            "weighted_call_breakeven": "REAL",
            "weighted_put_breakeven": "REAL",
            "consensus_price": "REAL",
            "consensus_bias_pct": "REAL",
            "consensus_bias_7d": "REAL",
            "consensus_bias_30d": "REAL",
            "consensus_bias_90d": "REAL",
            "near_term_skew": "REAL",
            "far_term_skew": "REAL",
            "max_pain_30d": "REAL",
        }'''

    if old_schema in content:
        content = content.replace(old_schema, new_schema)
        with open("data/store.py", "w") as f:
            f.write(content)
        print("✓ Updated store.py schema")
    else:
        print("⚠ store.py schema not found (may already be updated)")


def update_features():
    """Add new consensus bias features to features.py"""
    with open("data/features.py", "r") as f:
        content = f.read()

    # Add the new time-bucketed features and consensus spread to the derived features section
    old_marker = '    # Options divergence (Deribit vs IBIT)'
    new_derived = '''    # Consensus bias spread (short-term vs long-term sentiment divergence)
    if "consensus_bias_7d" in df.columns and "consensus_bias_90d" in df.columns:
        df["consensus_spread_7d_90d"] = df["consensus_bias_7d"] - df["consensus_bias_90d"]
    else:
        df["consensus_spread_7d_90d"] = np.nan

    # Options divergence (Deribit vs IBIT)'''

    if old_marker in content:
        content = content.replace(old_marker, new_derived)

    # Add the new features to the feature column list
    old_features = '''        # Deribit Options (5)
        "put_call_ratio",       # from deribit_options table
        "consensus_bias_pct",   # from deribit_options table
        "near_term_skew",
        "far_term_skew",
        "deribit_skew_divergence",'''

    new_features = '''        # Deribit Options (8)
        "put_call_ratio",       # from deribit_options table
        "consensus_bias_pct",   # all expirations
        "consensus_bias_7d",    # options expiring within 7 days
        "consensus_bias_30d",   # options expiring within 30 days
        "consensus_bias_90d",   # options expiring within 90 days
        "consensus_spread_7d_90d",  # short vs long term divergence
        "near_term_skew",
        "far_term_skew",
        "deribit_skew_divergence",'''

    if old_features in content:
        content = content.replace(old_features, new_features)
        with open("data/features.py", "w") as f:
            f.write(content)
        print("✓ Updated features.py")
    else:
        print("⚠ features.py feature list not found (may already be updated)")


def update_db():
    """Add new columns to existing SQLite database without losing data."""
    import sqlite3
    import os

    db_path = "btc_oracle.db"
    if not os.path.exists(db_path):
        print("⚠ No database found, columns will be created on next run")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check existing columns
    cursor.execute("PRAGMA table_info(deribit_options)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    new_cols = {
        "consensus_bias_7d": "REAL",
        "consensus_bias_30d": "REAL",
        "consensus_bias_90d": "REAL",
    }

    for col_name, col_type in new_cols.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE deribit_options ADD COLUMN {col_name} {col_type}")
            print(f"  ✓ Added column {col_name} to deribit_options")
        else:
            print(f"  - Column {col_name} already exists")

    conn.commit()
    conn.close()
    print("✓ Database schema updated")


if __name__ == "__main__":
    print("=== Applying BTC Oracle Update: Time-Bucketed Consensus Bias ===\n")
    update_store()
    update_features()
    update_db()
    print("\n=== Update complete! Run 'python run_collect.py' to test ===")
