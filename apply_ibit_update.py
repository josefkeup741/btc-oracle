"""
Run this on the VM to update store.py and features.py for expanded IBIT options.
Usage: python apply_ibit_update.py
"""

def update_store():
    """Add new columns to ibit_options schema in store.py"""
    with open("data/store.py", "r") as f:
        content = f.read()

    old_schema = '''        "ibit_options": {
            "ibit_price": "REAL",
            "total_call_oi": "REAL",
            "total_put_oi": "REAL",
            "put_call_ratio": "REAL",
            "weighted_call_breakeven": "REAL",
            "weighted_put_breakeven": "REAL",
            "consensus_price": "REAL",
            "consensus_bias_pct": "REAL",
        }'''

    new_schema = '''        "ibit_options": {
            "ibit_price": "REAL",
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
        }'''

    if old_schema in content:
        content = content.replace(old_schema, new_schema)
        with open("data/store.py", "w") as f:
            f.write(content)
        print("✓ Updated store.py with expanded IBIT schema")
    else:
        print("⚠ IBIT schema not found in store.py (may already be updated)")


def update_features():
    """Update features.py with expanded IBIT features and cross-market divergences."""
    with open("data/features.py", "r") as f:
        content = f.read()

    # Replace the IBIT feature section in the feature list
    old_features = '''        # IBIT Options (2)
        # These get suffixed by join_all if column names collide
        "consensus_bias_pct_ibit_options" if "consensus_bias_pct_ibit_options" in df.columns else "consensus_bias_pct",
        "options_divergence",'''

    new_features = '''        # IBIT Options (5)
        # These get suffixed by join_all if column names collide
        "consensus_bias_pct_ibit_options" if "consensus_bias_pct_ibit_options" in df.columns else "consensus_bias_pct",
        "consensus_bias_7d_ibit_options" if "consensus_bias_7d_ibit_options" in df.columns else "consensus_bias_7d",
        "consensus_bias_30d_ibit_options" if "consensus_bias_30d_ibit_options" in df.columns else "consensus_bias_30d",
        "near_term_skew_ibit_options" if "near_term_skew_ibit_options" in df.columns else "near_term_skew",
        "options_divergence",'''

    if old_features in content:
        content = content.replace(old_features, new_features)

    # Update the options divergence calculation to use 7d buckets (more precise)
    old_divergence = '''    # Options divergence (Deribit vs IBIT)
    deribit_bias = df.get("consensus_bias_pct")
    ibit_bias = df.get("consensus_bias_pct_ibit_options")
    if deribit_bias is not None and ibit_bias is not None:
        df["options_divergence"] = deribit_bias - ibit_bias
    else:
        df["options_divergence"] = np.nan'''

    new_divergence = '''    # Options divergence (Deribit vs IBIT)
    # Use 30d bucket for divergence since IBIT 7d bucket is often empty
    deribit_30d = df.get("consensus_bias_30d")
    ibit_30d = df.get("consensus_bias_30d_ibit_options")
    if deribit_30d is not None and ibit_30d is not None:
        df["options_divergence"] = deribit_30d - ibit_30d
    else:
        # Fall back to aggregate bias
        deribit_bias = df.get("consensus_bias_pct")
        ibit_bias = df.get("consensus_bias_pct_ibit_options")
        if deribit_bias is not None and ibit_bias is not None:
            df["options_divergence"] = deribit_bias - ibit_bias
        else:
            df["options_divergence"] = np.nan'''

    if old_divergence in content:
        content = content.replace(old_divergence, new_divergence)
        with open("data/features.py", "w") as f:
            f.write(content)
        print("✓ Updated features.py with expanded IBIT features")
    else:
        print("⚠ Options divergence block not found in features.py (may already be updated)")


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

    cursor.execute("PRAGMA table_info(ibit_options)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    new_cols = {
        "consensus_bias_7d": "REAL",
        "consensus_bias_30d": "REAL",
        "consensus_bias_90d": "REAL",
        "near_term_skew": "REAL",
        "far_term_skew": "REAL",
    }

    for col_name, col_type in new_cols.items():
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE ibit_options ADD COLUMN {col_name} {col_type}")
            print(f"  ✓ Added column {col_name} to ibit_options")
        else:
            print(f"  - Column {col_name} already exists")

    conn.commit()
    conn.close()
    print("✓ Database schema updated")


if __name__ == "__main__":
    print("=== Applying IBIT Options Update ===\n")
    update_store()
    update_features()
    update_db()
    print("\n=== Update complete! Run 'python run_collect.py' to test ===")
