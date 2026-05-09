import csv
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from access import get_features, get_features_batch, get_price, get_signals


TRADING_DAYS = [
    "2025-08-22",
    "2025-08-25",
    "2025-08-26",
    "2025-08-27",
    "2025-08-28",
    "2025-09-01",
    "2025-09-02",
]


def percentile(values, p):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round((len(ordered) - 1) * p)))
    return ordered[idx]


def ci95(values):
    if len(values) < 2:
        return (values[0], values[0])
    mean = statistics.mean(values)
    stderr = statistics.stdev(values) / (len(values) ** 0.5)
    margin = 1.96 * stderr
    return (mean - margin, mean + margin)


def sample_timestamps(n, seed=42):
    rng = random.Random(seed)
    stamps = []
    for _ in range(n):
        day = rng.choice(TRADING_DAYS)
        minute_offset = rng.randint(0, 374)
        stamps.append(datetime.fromisoformat(day + " 09:15:00") + timedelta(minutes=minute_offset))
    return stamps


def time_call(fn, timestamps):
    latencies = []
    successes = 0
    for ts in timestamps:
        start = time.perf_counter()
        result = fn(ts)
        latencies.append((time.perf_counter() - start) * 1000)
        if result is not None:
            successes += 1
    low, high = ci95(latencies)
    return {
        "samples": len(latencies),
        "successes": successes,
        "median_ms": statistics.median(latencies),
        "p99_ms": percentile(latencies, 0.99),
        "mean_ci95_low_ms": low,
        "mean_ci95_high_ms": high,
    }


def main():
    # Warm DuckDB views once so the benchmark focuses on steady-state access latency.
    get_price(datetime.fromisoformat("2025-08-22 09:15:00"), "spot")

    timestamps_100 = sample_timestamps(100)
    timestamps_1000 = sample_timestamps(1000, seed=7)

    rows = []
    for name, fn in [
        ("get_price_spot", lambda ts: get_price(ts, "spot")),
        ("get_features", lambda ts: get_features(ts)),
        ("get_signals_weekly", lambda ts: get_signals("weekly", ts)),
    ]:
        stats = time_call(fn, timestamps_100)
        rows.append({"metric": name, **stats, "wall_ms": ""})

    start = time.perf_counter()
    batch = get_features_batch(timestamps_1000)
    wall_ms = (time.perf_counter() - start) * 1000
    rows.append(
        {
            "metric": "get_features_batch_1000",
            "samples": len(batch),
            "successes": int(batch["valid"].sum()) if "valid" in batch else 0,
            "median_ms": "",
            "p99_ms": "",
            "mean_ci95_low_ms": "",
            "mean_ci95_high_ms": "",
            "wall_ms": wall_ms,
        }
    )

    rows.append(
        {
            "metric": "machine",
            "samples": platform.processor() or platform.machine(),
            "successes": platform.platform(),
            "median_ms": "",
            "p99_ms": "",
            "mean_ci95_low_ms": "",
            "mean_ci95_high_ms": "",
            "wall_ms": "",
        }
    )

    out_path = ROOT / "benchmark.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
