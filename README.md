# Quant Singularity Data Engine Intern Project

This repository builds a small but complete data layer for the supplied NIFTY market-data bundle. It ingests raw vendor CSV files, validates known bad feed conditions, writes a Parquet warehouse, registers DuckDB views over that warehouse, logs the run to MLflow, and exposes typed access functions for downstream strategy and feature workflows.

## Project Structure

```text
.
|-- intern_data_db/              # Supplied raw data bundle
|   `-- aux_data/                # Original aux folder; renamed because aux is reserved on Windows Git
|-- src/
|   |-- config.py                # Project paths
|   |-- validate.py              # Data-quality rules and validation log entries
|   |-- ingest.py                # CSV ingestion and Parquet writing
|   |-- access.py                # DuckDB views and access functions
|   `-- run.py                   # Single-command pipeline entrypoint
|-- warehouse/                   # Generated Parquet warehouse
|-- mlruns/                      # MLflow local tracking artifacts
|-- validation_report.json       # Validation module output
|-- benchmark.py                 # Latency benchmark runner
|-- benchmark.csv                # Latest benchmark output
|-- sai_pavan_sahu_report.pdf    # Written report
`-- requirements.txt
```

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Single Run Command

```powershell
python src\run.py
```

The command performs the full pipeline:

1. Clears and rewrites `warehouse/` for an idempotent run.
2. Reads all CSV files from `intern_data_db/`.
3. Applies validation rules for spot, futures, options, and VIX.
4. Writes partitioned Snappy Parquet files.
5. Registers DuckDB views over the Parquet warehouse.
6. Writes `validation_report.json`.
7. Logs parameters, metrics, and the report artifact to MLflow.

If an existing `validation_report.json` is locked by another process on Windows, the run still completes and writes `validation_report_latest.json`. On a fresh checkout, the normal output is `validation_report.json`.

## Warehouse Layout

```text
warehouse/
|-- spot/date=YYYY-MM-DD/*.parquet
|-- futures/date=YYYY-MM-DD/*.parquet
|-- options/date=YYYY-MM-DD/expiry=YYYY-MM-DD/*.parquet
|-- vix/date=YYYY-MM-DD/*.parquet
`-- aux_data/
    |-- fii_dii_flow.parquet
    `-- nse_calendar.parquet
```

Spot, futures, and VIX are partitioned by trading date. Options are partitioned by trading date and expiry because the common query is a full chain for a given session and contract expiry.

## Access Functions

The main consumer API is in `src/access.py`:

```python
from datetime import datetime
from access import get_price, get_features, get_signals, get_features_batch

price = get_price(datetime(2025, 8, 25, 10, 15), "spot")
features = get_features(datetime(2025, 8, 25, 10, 15))
chain = get_signals("weekly", datetime(2025, 8, 25, 10, 17))
batch = get_features_batch([
    datetime(2025, 8, 25, 10, 15),
    datetime(2025, 8, 25, 10, 16),
])
```

Contracts are intentionally conservative: pre-open timestamps return `None` or invalid rows, options snapshots are floored to the latest available 5-minute snapshot, and partial feature vectors are marked with `partial=True`.

## Benchmarks

Run:

```powershell
python benchmark.py
```

Latest local results are in `benchmark.csv`. The current run measured:

| Function | Median | p99 | Notes |
|---|---:|---:|---|
| `get_price("spot")` | 4.41 ms | 12.86 ms | Simple date/timestamp lookup |
| `get_features` | 57.86 ms | 70.30 ms | Multiple DuckDB lookups plus options-derived fields |
| `get_signals("weekly")` | 23.11 ms | 32.20 ms | Full chain snapshot lookup |
| `get_features_batch(1000)` | 26.03 s total | n/a | Contract-correct, but still not optimized enough for production |

## Known Gaps

- `get_features_batch` preserves one output row per input timestamp, including duplicates, but its VWAP and options loops should be vectorized further before production use.
- The current validation module catches the supplied feed issues but does not yet detect semantic errors such as swapped CE/PE labels or plausible-but-wrong prices.
- File-store MLflow works locally, but MLflow now recommends moving longer-lived tracking to a database-backed store.
