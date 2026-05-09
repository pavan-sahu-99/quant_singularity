# src\ingest.py
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import SPOT_DIR, FUTURES_DIR, OPTIONS_DIR, AUX_DIR, WAREHOUSE_DIR
from validate import validate_spot, validate_futures, validate_options, validate_vix

def write_partitioned_dataset(df, root_name, partition_cols):
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_to_dataset(
        table,
        root_path=os.path.join(WAREHOUSE_DIR, root_name),
        partition_cols=partition_cols,
        compression='snappy',
        basename_template='part-{i}.parquet',
    )

def ingest_spot(report_log):
    print("Ingesting Nifty Spot...")
    for f in sorted(os.listdir(SPOT_DIR)):
        if not f.endswith(".csv"): continue
        file_path = os.path.join(SPOT_DIR, f)
        
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = validate_spot(df, f"nifty_spot/{f}", report_log)
        
        df['date'] = df['timestamp'].dt.date
        df = df.sort_values('timestamp')
        
        write_partitioned_dataset(df, 'spot', ['date'])

def ingest_futures(report_log):
    print("Ingesting Nifty Futures...")
    for f in sorted(os.listdir(FUTURES_DIR)):
        if not f.endswith(".csv"): continue
        file_path = os.path.join(FUTURES_DIR, f)
        
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df = validate_futures(df, f"nifty_futures/{f}", report_log)
        
        df['date'] = df['timestamp'].dt.date
        df = df.sort_values('timestamp')
        
        write_partitioned_dataset(df, 'futures', ['date'])

def ingest_options(report_log):
    print("Ingesting Options Chain...")
    for f in sorted(os.listdir(OPTIONS_DIR)):
        if not f.endswith(".csv"): continue
        file_path = os.path.join(OPTIONS_DIR, f)
        
        df = pd.read_csv(file_path)
        df = validate_options(df, f"options_chain/{f}", report_log)
        
        df['date'] = df['timestamp'].dt.date
        # Extract expiry from filename (format: YYYY-MM-DD_YYYY-MM-DD.csv)
        _, expiry_str = f.replace('.csv', '').split('_')
        df['expiry'] = pd.to_datetime(expiry_str).date()
        
        # Deterministic sorting for strict idempotency
        df = df.sort_values(['timestamp', 'strike', 'side'])
        
        write_partitioned_dataset(df, 'options', ['date', 'expiry'])

def ingest_aux(report_log):
    print("Ingesting Aux Files...")
    aux_out_dir = os.path.join(WAREHOUSE_DIR, 'aux_data')
    os.makedirs(aux_out_dir, exist_ok=True)
    
    # FII DII
    fii = pd.read_csv(os.path.join(AUX_DIR, 'fii_dii_flow.csv'))
    fii.to_parquet(os.path.join(aux_out_dir, 'fii_dii_flow.parquet'), compression='snappy')
    
    # VIX
    vix = pd.read_csv(os.path.join(AUX_DIR, 'india_vix.csv'), parse_dates=['timestamp'])
    vix = validate_vix(vix, "aux/india_vix.csv", report_log)
    vix['date'] = vix['timestamp'].dt.date
    vix = vix.sort_values('timestamp')
    write_partitioned_dataset(vix, 'vix', ['date'])
    
    # Calendar
    cal = pd.read_csv(os.path.join(AUX_DIR, 'nse_calendar.csv'))
    cal.to_parquet(os.path.join(aux_out_dir, 'nse_calendar.parquet'), compression='snappy')

def ingest_all():
    os.makedirs(WAREHOUSE_DIR, exist_ok=True)
    report_log = []
    
    ingest_spot(report_log)
    ingest_futures(report_log)
    ingest_options(report_log)
    ingest_aux(report_log)
    
    return report_log
