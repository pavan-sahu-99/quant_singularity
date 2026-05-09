#src\run.py
import json
import os
import shutil
import mlflow
from datetime import datetime

from config import WAREHOUSE_DIR, BASE_DIR
from ingest import ingest_all
from access import build_duckdb_views

def run_pipeline():
    print("=====================================================")
    print("  Data Engine Ingest & Validate   ")
    print("=====================================================\n")
    
    # Idempotency Requirement: We wipe the warehouse and start fresh so outputs are byte-identical.
    if os.path.exists(WAREHOUSE_DIR):
        print("[!] Clearing existing warehouse for an idempotent run...")
        shutil.rmtree(WAREHOUSE_DIR)
        
    # Setup MLflow Tracking 
    mlflow.set_tracking_uri(f"file:///{BASE_DIR.replace(chr(92), '/')}/mlruns")
    mlflow.set_experiment("Data_Engine_Ingestion")
    
    with mlflow.start_run(run_name=f"Run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log Pipeline Parameters
        mlflow.log_param("data_window_start", "2025-08-22")
        mlflow.log_param("data_window_end", "2025-09-02")
        mlflow.log_param("parquet_compression", "snappy")
        
        # 1. Execute ingestion and validation
        report_log = ingest_all()
        mlflow.log_metric("total_warnings_dropped",
            sum(r['rows_affected'] for r in report_log if r['decision']=='dropped' and r['severity']=='WARNING'))
        mlflow.log_metric("total_issues_fixed",
            sum(r['rows_affected'] for r in report_log if r['decision']=='fixed'))
        mlflow.log_metric("total_findings", len(report_log))
        
        # 2. Build the JSON Validation Report
        report_data = {
            "report_version": "1.0",
            "data_window": {
                "start": "2025-08-22",
                "end": "2025-09-02"
            },
            "summary": {
                "total_critical_errors_dropped": sum([r['rows_affected'] for r in report_log if r['decision'] == 'dropped' and r['severity'] == 'CRITICAL']),
                "total_warnings_dropped": sum([r['rows_affected'] for r in report_log if r['decision'] == 'dropped' and r['severity'] == 'WARNING']),
                "total_issues_fixed": sum([r['rows_affected'] for r in report_log if r['decision'] == 'fixed'])
            },
            "findings": report_log
        }
        
        # 3. Register DuckDB Views
        build_duckdb_views()
        
        report_path = os.path.join(BASE_DIR, "validation_report.json")
        artifact_path = report_path
        try:
            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=4)
        except PermissionError:
            artifact_path = os.path.join(BASE_DIR, "validation_report_latest.json")
            with open(artifact_path, "w") as f:
                json.dump(report_data, f, indent=4)
            print(f"\n[!] Existing validation_report.json is locked; wrote {artifact_path} instead.")
            
        print(f"\n[/] Pipeline complete! Validation report saved to {artifact_path}")
        
        # 3. Log Metrics & Artifacts
        mlflow.log_metric("total_critical_errors_dropped", report_data["summary"]["total_critical_errors_dropped"])
        mlflow.log_artifact(artifact_path)
        print("[/] MLflow Tracking successfully recorded.")

if __name__ == "__main__":
    run_pipeline()
