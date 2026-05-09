# src/access.py
#
# Four functions you care about:
#   get_price()           — spot or futures close at a given minute
#   get_features()        — full feature dict at a given timestamp
#   get_signals()         — options chain snapshot (weekly or monthly)
#   get_features_batch()  — same as get_features() but for 1000 timestamps at once

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional
import os

from config import WAREHOUSE_DIR

# NSE opens at 9:15, last 1-min bar closes at 15:29, last options snap is at 15:25
MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 29)
OPTIONS_LAST = time(15, 25)

# First call to get_connection() spins it up and registers all the views.
_con: Optional[duckdb.DuckDBPyConnection] = None


def get_connection() -> duckdb.DuckDBPyConnection:
    # Lazy init — only creates the connection once, reuses it after that.
    global _con
    if _con is None:
        _con = duckdb.connect(database=":memory:", read_only=False)
        _register_views(_con)
    return _con


def _register_views(con: duckdb.DuckDBPyConnection) -> None:
    # Point DuckDB at the Parquet folders using glob patterns.
    # Nothing is actually loaded into memory here — DuckDB reads files on demand
    # and uses the hive partition folders (date=.../expiry=...) for pruning.
    spot_glob    = os.path.join(WAREHOUSE_DIR, "spot",    "**", "*.parquet").replace("\\", "/")
    futures_glob = os.path.join(WAREHOUSE_DIR, "futures", "**", "*.parquet").replace("\\", "/")
    options_glob = os.path.join(WAREHOUSE_DIR, "options", "**", "*.parquet").replace("\\", "/")
    vix_glob     = os.path.join(WAREHOUSE_DIR, "vix",     "**", "*.parquet").replace("\\", "/")
    aux_dir      = "aux_data" if os.path.exists(os.path.join(WAREHOUSE_DIR, "aux_data")) else "aux"
    fii_path     = os.path.join(WAREHOUSE_DIR, aux_dir, "fii_dii_flow.parquet").replace("\\", "/")
    cal_path     = os.path.join(WAREHOUSE_DIR, aux_dir, "nse_calendar.parquet").replace("\\", "/")

    con.execute(f"CREATE OR REPLACE VIEW v_spot     AS SELECT * FROM read_parquet('{spot_glob}',    hive_partitioning=true)")
    con.execute(f"CREATE OR REPLACE VIEW v_futures  AS SELECT * FROM read_parquet('{futures_glob}', hive_partitioning=true)")
    con.execute(f"CREATE OR REPLACE VIEW v_options  AS SELECT * FROM read_parquet('{options_glob}', hive_partitioning=true)")
    con.execute(f"CREATE OR REPLACE VIEW v_vix      AS SELECT * FROM read_parquet('{vix_glob}',     hive_partitioning=true)")
    con.execute(f"CREATE OR REPLACE VIEW v_fii_dii  AS SELECT * FROM read_parquet('{fii_path}')")
    con.execute(f"CREATE OR REPLACE VIEW v_calendar AS SELECT * FROM read_parquet('{cal_path}')")


def build_duckdb_views() -> None:
    """Called from run.py right after ingestion finishes.
    Resets the connection so views point at the freshly written warehouse."""
    global _con
    _con = None
    get_connection()
    print("[/] DuckDB views registered over warehouse Parquet files.")


# ── Small helpers used everywhere below ───────────────────────────────────────

def _is_trading_time(ts: datetime) -> bool:
    return MARKET_OPEN <= ts.time() <= MARKET_CLOSE


def _floor_to_minute(ts: datetime) -> datetime:
    # 09:16:45 becomes 09:16:00 — strips sub-minute precision
    return ts.replace(second=0, microsecond=0)


def _floor_to_5min(ts: datetime) -> datetime:
    # 09:17:30 becomes 09:15:00 — finds the containing options snapshot
    minute = (ts.minute // 5) * 5
    return ts.replace(minute=minute, second=0, microsecond=0)


def _is_trading_day(ts: datetime, con: duckdb.DuckDBPyConnection) -> bool:
    d = ts.date().isoformat()
    count = con.execute(
        "SELECT COUNT(*) FROM v_calendar WHERE trade_date = ?", [d]
    ).fetchone()[0]
    return count > 0


# ══════════════════════════════════════════════════════════════════════════════
# get_price
# ══════════════════════════════════════════════════════════════════════════════

def get_price(timestamp: datetime, instrument: str = "spot") -> Optional[float]:
    """
    Give me a timestamp, I'll give you the close price of that 1-minute bar.

    instrument can be 'spot', 'near_futures', or 'mid_futures'.

    A few things worth knowing before you call this:
    - If you pass a time before 9:15, you'll get None back. Pre-open prices
      aren't continuous and shouldn't be used as-is.
    - If you pass a time after 15:29, we snap to the 15:29 close (last bar).
    - If you pass 09:16:45, we look up the 09:16 bar — seconds get stripped.
    - If a bar is genuinely missing from the warehouse (gap in the feed),
      you get None. We never forward-fill silently here.
    - Bad instrument name raises ValueError immediately so you catch it early.
    """
    allowed = {"spot", "near_futures", "mid_futures"}
    if instrument not in allowed:
        raise ValueError(f"instrument must be one of {allowed}, got '{instrument}'")

    con = get_connection()

    if timestamp.time() < MARKET_OPEN:
        return None

    # Anything after the last bar gets capped at 15:29
    if timestamp.time() > MARKET_CLOSE:
        timestamp = timestamp.replace(hour=15, minute=29, second=0, microsecond=0)

    bar_ts   = _floor_to_minute(timestamp)
    date_str = bar_ts.date().isoformat()

    if instrument == "spot":
        row = con.execute(
            "SELECT close FROM v_spot WHERE date = ? AND timestamp = ? LIMIT 1",
            [date_str, bar_ts]
        ).fetchone()

    elif instrument == "near_futures":
        row = con.execute(
            "SELECT near_month_close FROM v_futures WHERE date = ? AND timestamp = ? LIMIT 1",
            [date_str, bar_ts]
        ).fetchone()

    else:
        row = con.execute(
            "SELECT mid_month_close FROM v_futures WHERE date = ? AND timestamp = ? LIMIT 1",
            [date_str, bar_ts]
        ).fetchone()

    # Row is None when the bar doesn't exist — gap in feed, holiday, or wrong date
    if row is None:
        return None

    return float(row[0])


# ══════════════════════════════════════════════════════════════════════════════
# get_features
# ══════════════════════════════════════════════════════════════════════════════

def get_features(timestamp: datetime,lookback_days: int = 30) -> Optional[dict]:
    """
    Returns everything the feature engine needs at a given timestamp, assembled
    from spot, futures, VIX, FII/DII, and the options chain.

    What you get back:
        spot_close        — 1-min close price
        vwap_session      — cumulative VWAP from 9:15 to this bar
        vix               — India VIX at this exact minute
        fii_net           — FII net flow for the day (crores)
        dii_net           — DII net flow for the day (crores)
        near_basis        — near futures close minus spot close
        atm_iv_ce         — IV of the ATM call at the last options snapshot
        atm_iv_pe         — IV of the ATM put at the last options snapshot
        pcr_oi            — total put OI / total call OI at last snapshot
        lookback_complete — False when we have fewer days than lookback_days asks for
        partial           — True when any of the above came back None

    Some edge cases you'll hit in the first week:
    - First 5 minutes of the session: VWAP only covers the bars we have so far,
      not None — a partial calculation is still useful.
    - We only have 7 days of data, so any lookback > 7 sets lookback_complete=False.
      The dict still comes back, just with that flag set.
    - If one source is unavailable (say VIX has a gap), that key is None and
      partial=True, but the rest of the dict is still populated and usable.
    - Timestamp outside trading hours returns None entirely, not a partial dict.
    """
    if not _is_trading_time(timestamp):
        return None

    con      = get_connection()
    bar_ts   = _floor_to_minute(timestamp)
    date_str = bar_ts.date().isoformat()

    result = {
        "timestamp":          bar_ts,
        "spot_close":         None,
        "vwap_session":       None,
        "vix":                None,
        "fii_net":            None,
        "dii_net":            None,
        "near_basis":         None,
        "atm_iv_ce":          None,
        "atm_iv_pe":          None,
        "pcr_oi":             None,
        "lookback_complete":  False,
        "partial":            False,
    }

    # How many trading days do we actually have in the warehouse?
    available_days = con.execute(
        "SELECT COUNT(DISTINCT date) FROM v_spot"
    ).fetchone()[0]
    result["lookback_complete"] = available_days >= lookback_days

    # Spot close
    row = con.execute(
        "SELECT close FROM v_spot WHERE date=? AND timestamp=? LIMIT 1",
        [date_str, bar_ts]
    ).fetchone()
    if row:
        result["spot_close"] = float(row[0])

    # Session VWAP — sum(price * volume) / sum(volume) from open to this bar
    row = con.execute(
        """
        SELECT SUM(close * volume) / NULLIF(SUM(volume), 0)
        FROM v_spot
        WHERE date = ?
          AND timestamp >= CAST(? || ' 09:15:00' AS TIMESTAMP)
          AND timestamp <= ?
        """,
        [date_str, date_str, bar_ts]
    ).fetchone()
    if row and row[0] is not None:
        result["vwap_session"] = float(row[0])

    # VIX — 1-min resolution, same timestamp as spot
    row = con.execute(
        "SELECT vix_close FROM v_vix WHERE date=? AND timestamp=? LIMIT 1",
        [date_str, bar_ts]
    ).fetchone()
    if row:
        result["vix"] = float(row[0])

    # FII/DII — daily data, so we just need the date
    row = con.execute(
        "SELECT fii_net, dii_net FROM v_fii_dii WHERE date=? LIMIT 1",
        [date_str]
    ).fetchone()
    if row:
        result["fii_net"] = float(row[0]) if row[0] is not None else None
        result["dii_net"] = float(row[1]) if row[1] is not None else None

    # Near-month basis = futures close - spot close
    row = con.execute(
        "SELECT near_month_close FROM v_futures WHERE date=? AND timestamp=? LIMIT 1",
        [date_str, bar_ts]
    ).fetchone()
    if row and result["spot_close"] is not None:
        result["near_basis"] = float(row[0]) - result["spot_close"]

    # Options features — use the last 5-min snapshot at or before this bar
    snap_ts = _floor_to_5min(bar_ts)

    if snap_ts.time() >= MARKET_OPEN:
        # Weekly expiry = MIN(expiry) on this date
        expiry_row = con.execute(
            "SELECT MIN(expiry) FROM v_options WHERE date = ? AND expiry >= ?",
            [date_str, date_str]
        ).fetchone()

        if expiry_row and expiry_row[0] is not None:
            expiry_str = str(expiry_row[0])

            # ATM strike = closest listed strike to current spot price
            if result["spot_close"] is not None:
                atm_row = con.execute(
                    """
                    SELECT strike FROM v_options
                    WHERE date=? AND expiry=? AND timestamp=?
                    ORDER BY ABS(strike - ?) ASC
                    LIMIT 1
                    """,
                    [date_str, expiry_str, snap_ts, result["spot_close"]]
                ).fetchone()

                if atm_row:
                    atm_strike = atm_row[0]

                    # ATM call IV — skip rows where the vendor froze IV (iv_reliable=False)
                    ce = con.execute(
                        """
                        SELECT iv FROM v_options
                        WHERE date=? AND expiry=? AND timestamp=?
                          AND strike=? AND side='CE' AND iv_reliable=TRUE
                        LIMIT 1
                        """,
                        [date_str, expiry_str, snap_ts, atm_strike]
                    ).fetchone()
                    if ce and ce[0] is not None:
                        result["atm_iv_ce"] = float(ce[0])

                    # ATM put IV
                    pe = con.execute(
                        """
                        SELECT iv FROM v_options
                        WHERE date=? AND expiry=? AND timestamp=?
                          AND strike=? AND side='PE' AND iv_reliable=TRUE
                        LIMIT 1
                        """,
                        [date_str, expiry_str, snap_ts, atm_strike]
                    ).fetchone()
                    if pe and pe[0] is not None:
                        result["atm_iv_pe"] = float(pe[0])

            # PCR = total put OI / total call OI across all strikes
            pcr = con.execute(
                """
                SELECT
                    SUM(CASE WHEN side='PE' THEN oi ELSE 0 END) * 1.0 /
                    NULLIF(SUM(CASE WHEN side='CE' THEN oi ELSE 0 END), 0)
                FROM v_options
                WHERE date=? AND expiry=? AND timestamp=?
                """,
                [date_str, expiry_str, snap_ts]
            ).fetchone()
            if pcr and pcr[0] is not None:
                result["pcr_oi"] = float(pcr[0])

    # If any feature came back empty, set partial=True so the caller knows
    value_keys = ["spot_close", "vwap_session", "vix", "fii_net", "dii_net",
                  "near_basis", "atm_iv_ce", "atm_iv_pe", "pcr_oi"]
    result["partial"] = any(result[k] is None for k in value_keys)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# get_signals
# ══════════════════════════════════════════════════════════════════════════════

def get_signals(chain_type: str,timestamp: datetime) -> Optional[pd.DataFrame]:
    """
    Returns the full options chain snapshot closest to (but not after) the given timestamp.

    chain_type='weekly'  gives you the nearest expiry.
    chain_type='monthly' gives you the far expiry.

    Things that will trip you up if you don't read this:
    - Options snapshots are every 5 minutes. If you ask for 10:17, you get the
      10:15 snapshot. We never interpolate option prices or IVs — not safe.
    - If you ask for a time before the first snapshot of the day (say 09:13),
      you get None. We won't give you yesterday's chain — the market was closed
      overnight and conditions are completely different.
    - If you ask after 15:25 (last snapshot), you still get the 15:25 data
      but the 'stale' column will be True so you know it's not fresh.
    - If the chain file is missing for that (date, expiry) pair entirely, None.
    - iv_reliable=False rows mean don't use that IV — most relevant on Aug 28
      expiry day where the vendor froze all IVs at 0.1540 all session.
    """
    if chain_type not in {"weekly", "monthly"}:
        raise ValueError(f"chain_type must be 'weekly' or 'monthly', got '{chain_type}'")

    if timestamp.time() < MARKET_OPEN:
        return None

    con      = get_connection()
    date_str = timestamp.date().isoformat()

    # Pick the right expiry based on chain_type
    if chain_type == "weekly":
        expiry_row = con.execute(
            "SELECT MIN(expiry) FROM v_options WHERE date=? AND expiry >= ?",
            [date_str, date_str]
        ).fetchone()
    else:
        expiry_row = con.execute(
            "SELECT MAX(expiry) FROM v_options WHERE date=? AND expiry >= ?",
            [date_str, date_str]
        ).fetchone()

    if not expiry_row or expiry_row[0] is None:
        return None

    expiry_str = str(expiry_row[0])

    # Find the last snapshot that exists at or before our timestamp
    snap_row = con.execute(
        """
        SELECT MAX(timestamp) FROM v_options
        WHERE date=? AND expiry=? AND timestamp <= ?
        """,
        [date_str, expiry_str, timestamp]
    ).fetchone()

    if not snap_row or snap_row[0] is None:
        # Nothing before this time today — don't reach back to yesterday
        return None

    snap_ts = snap_row[0]

    # Pull the full chain at that snapshot timestamp
    df = con.execute(
        """
        SELECT timestamp, strike, side, open, high, low, close,
               volume, oi, iv, iv_reliable, expiry
        FROM v_options
        WHERE date=? AND expiry=? AND timestamp=?
        ORDER BY strike ASC, side ASC
        """,
        [date_str, expiry_str, snap_ts]
    ).df()

    if df.empty:
        return None

    # Mark it stale if the snapshot is older than the bar requested
    df["stale"] = pd.to_datetime(snap_ts) < pd.to_datetime(
        timestamp.replace(second=0, microsecond=0)
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# get_features_batch
# ══════════════════════════════════════════════════════════════════════════════

def get_features_batch(timestamps: list,lookback_days: int = 30) -> pd.DataFrame:
    """
    Same as get_features() but for a list of timestamps, done efficiently.

    Instead of calling get_features() in a loop (which would fire one DuckDB
    query per timestamp per data source = very slow at 1000 timestamps), this
    fires one query per data source across all timestamps at once, then joins
    everything together at the end.

    You always get back exactly as many rows as you put in, in the same order.
    Nothing is dropped. If a timestamp is outside trading hours or has missing
    data, it shows up with valid=False or partial=True and NaN in those columns.
    The caller decides what to do with those rows — we don't make that call.
    """
    if not timestamps:
        return pd.DataFrame()

    con = get_connection()

    feature_cols = [
        "spot_close", "vwap_session", "vix", "fii_net", "dii_net",
        "near_basis", "atm_iv_ce", "atm_iv_pe", "pcr_oi"
    ]

    # Sort timestamps into valid (trading hours) and invalid up front
    valid_ts   = []
    invalid_ts = {}

    for ts in timestamps:
        bar_ts = _floor_to_minute(ts)
        if not _is_trading_time(bar_ts):
            invalid_ts[ts] = "outside_trading_hours"
        else:
            valid_ts.append(bar_ts)

    # Start with a skeleton — one NaN row per input timestamp
    skeleton = pd.DataFrame({
        "timestamp":         [_floor_to_minute(ts) for ts in timestamps],
        **{col: np.nan for col in feature_cols},
        "lookback_complete": False,
        "partial":           True,
        "valid":             False,
        "invalid_reason":    ""
    })

    for ts, reason in invalid_ts.items():
        bar_ts = _floor_to_minute(ts)
        skeleton.loc[skeleton["timestamp"] == bar_ts, "invalid_reason"] = reason

    if not valid_ts:
        return skeleton

    unique_ts     = list(set(valid_ts))
    unique_dates  = list(set(ts.date().isoformat() for ts in unique_ts))
    ts_literals   = ", ".join(f"TIMESTAMP '{ts}'" for ts in unique_ts)
    date_literals = ", ".join(f"'{d}'" for d in unique_dates)

    # Spot — single query across all timestamps
    spot_df = con.execute(
        f"SELECT timestamp, close AS spot_close FROM v_spot WHERE timestamp IN ({ts_literals})"
    ).df()
    spot_df["timestamp"] = pd.to_datetime(spot_df["timestamp"])

    # Session VWAP — cumulative per bar, batched by date
    vwap_rows = []
    for d in unique_dates:
        for bar_ts in [ts for ts in unique_ts if ts.date().isoformat() == d]:
            row = con.execute(
                """
                SELECT ? AS ts,
                       SUM(close * volume) / NULLIF(SUM(volume), 0)
                FROM v_spot
                WHERE date = ?
                  AND timestamp >= CAST(? || ' 09:15:00' AS TIMESTAMP)
                  AND timestamp <= ?
                """,
                [bar_ts, d, d, bar_ts]
            ).fetchone()
            if row:
                vwap_rows.append({"timestamp": pd.Timestamp(row[0]), "vwap_session": row[1]})

    vwap_df = pd.DataFrame(vwap_rows) if vwap_rows else pd.DataFrame(
        columns=["timestamp", "vwap_session"]
    )
    if not vwap_df.empty:
        vwap_df["timestamp"] = pd.to_datetime(vwap_df["timestamp"])

    # VIX
    vix_df = con.execute(
        f"SELECT timestamp, vix_close AS vix FROM v_vix WHERE timestamp IN ({ts_literals})"
    ).df()
    if not vix_df.empty:
        vix_df["timestamp"] = pd.to_datetime(vix_df["timestamp"])

    # FII/DII — daily, join on date
    fii_df = con.execute(
        f"SELECT date, fii_net, dii_net FROM v_fii_dii WHERE date IN ({date_literals})"
    ).df()
    if not fii_df.empty:
        fii_df["date"] = fii_df["date"].astype(str)

    # Futures
    fut_df = con.execute(
        f"SELECT timestamp, near_month_close FROM v_futures WHERE timestamp IN ({ts_literals})"
    ).df()
    if not fut_df.empty:
        fut_df["timestamp"] = pd.to_datetime(fut_df["timestamp"])

    # Options — floor each timestamp to 5-min, then look up PCR per snapshot
    snap_map     = {ts: _floor_to_5min(ts) for ts in unique_ts}
    spot_lookup = {}
    if not spot_df.empty:
        spot_lookup = spot_df.set_index("timestamp")["spot_close"].to_dict()

    opt_rows = []
    for ts, snap_ts in snap_map.items():
        spot_close = spot_lookup.get(pd.Timestamp(ts))
        if snap_ts.time() < MARKET_OPEN or pd.isna(spot_close):
            continue

        d = snap_ts.date().isoformat()
        expiry_row = con.execute(
            "SELECT MIN(expiry) FROM v_options WHERE date=? AND expiry >= ?", [d, d]
        ).fetchone()
        if not expiry_row or expiry_row[0] is None:
            continue
        expiry_str = str(expiry_row[0])

        row = con.execute(
            """
            WITH atm AS (
                SELECT strike
                FROM v_options
                WHERE date=? AND expiry=? AND timestamp=?
                ORDER BY ABS(strike - ?) ASC
                LIMIT 1
            )
            SELECT
                SUM(CASE WHEN side='PE' THEN oi ELSE 0 END) * 1.0 /
                    NULLIF(SUM(CASE WHEN side='CE' THEN oi ELSE 0 END), 0) AS pcr_oi,
                MAX(CASE
                    WHEN side='CE'
                     AND strike=(SELECT strike FROM atm)
                     AND iv_reliable=TRUE
                    THEN iv
                END) AS atm_iv_ce,
                MAX(CASE
                    WHEN side='PE'
                     AND strike=(SELECT strike FROM atm)
                     AND iv_reliable=TRUE
                    THEN iv
                END) AS atm_iv_pe
            FROM v_options
            WHERE date=? AND expiry=? AND timestamp=?
            """,
            [d, expiry_str, snap_ts, float(spot_close), d, expiry_str, snap_ts]
        ).fetchone()
        opt_rows.append({
            "timestamp": ts,
            "pcr_oi": float(row[0]) if row and row[0] is not None else None,
            "atm_iv_ce": float(row[1]) if row and row[1] is not None else None,
            "atm_iv_pe": float(row[2]) if row and row[2] is not None else None,
        })

    opt_df = pd.DataFrame(opt_rows) if opt_rows else pd.DataFrame(
        columns=["timestamp", "pcr_oi", "atm_iv_ce", "atm_iv_pe"]
    )

    # Stitch everything onto the skeleton row by row
    skeleton = skeleton.set_index("timestamp")

    if not spot_df.empty:
        skeleton.update(spot_df.set_index("timestamp")[["spot_close"]])

    if not vwap_df.empty:
        skeleton.update(vwap_df.set_index("timestamp")[["vwap_session"]])

    if not vix_df.empty:
        skeleton.update(vix_df.set_index("timestamp")[["vix"]])

    if not fii_df.empty:
        fii_indexed = fii_df.set_index("date")
        for ts in skeleton.index:
            d_key = ts.date().isoformat()
            if d_key in fii_indexed.index:
                skeleton.at[ts, "fii_net"] = fii_indexed.at[d_key, "fii_net"]
                skeleton.at[ts, "dii_net"] = fii_indexed.at[d_key, "dii_net"]

    if not fut_df.empty and not spot_df.empty:
        merged = fut_df.set_index("timestamp").join(
            spot_df.set_index("timestamp"), how="inner"
        )
        merged["near_basis"] = merged["near_month_close"] - merged["spot_close"]
        skeleton.update(merged[["near_basis"]])

    if not opt_df.empty:
        opt_df["timestamp"] = pd.to_datetime(opt_df["timestamp"])
        skeleton.update(opt_df.set_index("timestamp")[["pcr_oi", "atm_iv_ce", "atm_iv_pe"]])

    # Final pass — mark valid rows and compute partial flag
    skeleton = skeleton.reset_index()
    trading_mask = skeleton["timestamp"].apply(_is_trading_time)
    skeleton.loc[trading_mask, "valid"] = True

    for col in feature_cols:
        skeleton[col] = pd.to_numeric(skeleton[col], errors="coerce")

    skeleton["partial"] = skeleton[feature_cols].isnull().any(axis=1)

    available_days = con.execute(
        "SELECT COUNT(DISTINCT date) FROM v_spot"
    ).fetchone()[0]
    skeleton["lookback_complete"] = available_days >= lookback_days

    # The skeleton was created in input order and can contain duplicate timestamps.
    # Returning it directly preserves the one-output-row-per-input contract.
    return skeleton
