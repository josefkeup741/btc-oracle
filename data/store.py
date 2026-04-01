"""
BTC Oracle Storage Layer
========================
SQLite-based time-series storage. All collectors write here.
Handles upserts, queries, and cross-table joins on timestamp.
"""
import sqlite3
import pandas as pd
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

from config import DB_PATH

logger = logging.getLogger(__name__)


class Store:
    """SQLite time-series store. One table per data source."""

    # Schema definitions: table_name -> {column: sql_type}
    TABLES = {
        "btc_price": {
            "open": "REAL",
            "high": "REAL",
            "low": "REAL",
            "close": "REAL",
            "volume": "REAL",
        },
        "fear_greed": {
            "fng_value": "REAL",
            "fng_classification": "TEXT",
        },
        "deribit_options": {
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
        },
        "ibit_options": {
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
        },
        "whale_activity": {
            "large_tx_count": "INTEGER",
            "large_tx_volume_btc": "REAL",
            "exchange_inflow_btc": "REAL",
            "exchange_outflow_btc": "REAL",
            "net_exchange_flow": "REAL",
            "whale_tx_avg_size": "REAL",
        },
        "polymarket": {
            "btc_primary_prob": "REAL",
            "btc_primary_question": "TEXT",
            "market_volume_24h": "REAL",
            "prob_delta_24h": "REAL",
        },
        "technicals": {
            "ema_ratio": "REAL",
            "rsi_14": "REAL",
            "atr_14": "REAL",
            "obv_slope": "REAL",
            "bb_pct_b": "REAL",
            "macd_hist": "REAL",
        },
    }

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent read perf
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Create all tables if they don't exist."""
        with self._conn() as conn:
            for table_name, columns in self.TABLES.items():
                col_defs = ", ".join(f"{name} {dtype}" for name, dtype in columns.items())
                sql = f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        timestamp TEXT PRIMARY KEY,
                        {col_defs}
                    )
                """
                conn.execute(sql)
            logger.info(f"Database initialized at {self.db_path}")

    def upsert(self, table: str, timestamp: str, data: dict):
        """Insert or replace a row. Timestamp should be ISO format UTC string."""
        if table not in self.TABLES:
            raise ValueError(f"Unknown table: {table}. Valid: {list(self.TABLES.keys())}")

        # Only insert columns that exist in the schema
        valid_cols = set(self.TABLES[table].keys())
        filtered = {k: v for k, v in data.items() if k in valid_cols}

        cols = ["timestamp"] + list(filtered.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_str = ", ".join(cols)
        values = [timestamp] + list(filtered.values())

        with self._conn() as conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})",
                values,
            )

    def upsert_many(self, table: str, rows: list[dict]):
        """Bulk upsert. Each dict must have a 'timestamp' key."""
        if not rows:
            return
        if table not in self.TABLES:
            raise ValueError(f"Unknown table: {table}")

        valid_cols = set(self.TABLES[table].keys())
        # Use columns from first row
        sample = rows[0]
        filtered_keys = [k for k in sample.keys() if k in valid_cols]
        cols = ["timestamp"] + filtered_keys
        col_str = ", ".join(cols)
        placeholders = ", ".join(["?"] * len(cols))

        with self._conn() as conn:
            for row in rows:
                values = [row["timestamp"]] + [row.get(k) for k in filtered_keys]
                conn.execute(
                    f"INSERT OR REPLACE INTO {table} ({col_str}) VALUES ({placeholders})",
                    values,
                )
        logger.info(f"Upserted {len(rows)} rows into {table}")

    def query(self, table: str, start: str = None, end: str = None) -> pd.DataFrame:
        """Query a table for a time range. Returns DataFrame."""
        conditions = []
        params = []
        if start:
            conditions.append("timestamp >= ?")
            params.append(start)
        if end:
            conditions.append("timestamp <= ?")
            params.append(end)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._conn() as conn:
            df = pd.read_sql_query(
                f"SELECT * FROM {table} {where} ORDER BY timestamp",
                conn,
                params=params,
            )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        return df

    def latest(self, table: str) -> dict | None:
        """Get the most recent row from a table."""
        with self._conn() as conn:
            cursor = conn.execute(
                f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cols = [desc[0] for desc in cursor.description]
            return dict(zip(cols, row))

    def count(self, table: str) -> int:
        """Row count for a table."""
        with self._conn() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            return cursor.fetchone()[0]

    def join_all(self, start: str = None, end: str = None) -> pd.DataFrame:
        """
        Merge all tables on timestamp using an asof join.
        Uses btc_price as the base timeline (most granular).
        For each other table, finds the most recent value at or before each timestamp.
        """
        base = self.query("btc_price", start, end)
        if base.empty:
            return base

        base = base.set_index("timestamp").sort_index()

        for table_name in self.TABLES:
            if table_name == "btc_price":
                continue

            other = self.query(table_name, start, end)
            if other.empty:
                # Add NaN columns for this table
                for col in self.TABLES[table_name]:
                    base[col] = None
                continue

            other = other.set_index("timestamp").sort_index()

            # Asof merge: for each base timestamp, get the most recent row from other
            base = pd.merge_asof(
                base.reset_index().sort_values("timestamp"),
                other.reset_index().sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                suffixes=("", f"_{table_name}"),
            ).set_index("timestamp")

        return base.reset_index()

    def import_csv(self, table: str, csv_path: str, column_map: dict = None):
        """
        Import a CSV file into a table.
        column_map: {csv_column: db_column} mapping.
        """
        df = pd.read_csv(csv_path)
        if column_map:
            df = df.rename(columns=column_map)

        rows = df.to_dict("records")
        self.upsert_many(table, rows)
        logger.info(f"Imported {len(rows)} rows from {csv_path} into {table}")

    def status(self) -> dict:
        """Return row counts and date ranges for all tables."""
        info = {}
        for table in self.TABLES:
            count = self.count(table)
            if count > 0:
                latest = self.latest(table)
                with self._conn() as conn:
                    cursor = conn.execute(
                        f"SELECT MIN(timestamp) FROM {table}"
                    )
                    earliest = cursor.fetchone()[0]
                info[table] = {
                    "rows": count,
                    "earliest": earliest,
                    "latest": latest["timestamp"],
                }
            else:
                info[table] = {"rows": 0, "earliest": None, "latest": None}
        return info
