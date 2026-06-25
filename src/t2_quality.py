"""
T2: Data quality analysis of original TLC parquet files using Dask bags (map-reduce).

For each file: reads with pandas, detects quality issues, groups results by
actual pickup year. Results are aggregated across files by (dataset, year) and
saved to JSON.
"""

import os
import re
import json
import traceback
from pathlib import Path

import pandas as pd
import dask.bag as db
from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# ── Paths ─────────────────────────────────────────────────────────────────────

SHARED_DIR   = Path("/d/hpc/projects/FRI/bigdata/data/Taxi")
DOWNLOAD_DIR = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/taxi_new")
OUTPUT_DIR   = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data")
OUTPUT_FILE  = OUTPUT_DIR / "t2_quality_results.json"

# ── Dataset configs ────────────────────────────────────────────────────────────
# pickup_cols: tried in order until one is found in the file

DATASET_CONFIGS = {
    "yellow_tripdata": {
        "pickup_cols":    ["tpep_pickup_datetime", "pickup_datetime", "Trip_Pickup_DateTime"],
        "dropoff_cols":   ["tpep_dropoff_datetime", "dropoff_datetime", "Trip_Dropoff_DateTime"],
        "distance_col":   "trip_distance",
        "passenger_col":  "passenger_count",
        "fare_col":       "fare_amount",
    },
    "green_tripdata": {
        "pickup_cols":    ["lpep_pickup_datetime"],
        "dropoff_cols":   ["lpep_dropoff_datetime"],
        "distance_col":   "trip_distance",
        "passenger_col":  "passenger_count",
        "fare_col":       "fare_amount",
    },
    "fhv_tripdata": {
        "pickup_cols":    ["pickup_datetime"],
        "dropoff_cols":   ["dropOff_datetime"],
        "distance_col":   None,
        "passenger_col":  None,
        "fare_col":       None,
    },
    "fhvhv_tripdata": {
        "pickup_cols":    ["pickup_datetime"],
        "dropoff_cols":   ["dropoff_datetime"],
        "distance_col":   "trip_miles",
        "passenger_col":  None,
        "fare_col":       "base_passenger_fare",
    },
}

PLACEHOLDER_DROPOFF = pd.Timestamp("1989-01-01 00:00:00")

ISSUE_COLS = [
    "invalid_year",
    "null_pickup",
    "null_dropoff",
    "placeholder_dropoff",
    "same_timestamps",
    "negative_duration",
    "zero_distance",
    "negative_distance",
    "zero_passengers",
    "negative_fare",
    "excessive_duration",
    "high_fare",
]


def _extract_file_year(file_path):
    """Extract the nominal year from either the filename or its parent folders."""
    path = Path(file_path)
    year_pattern = re.compile(r"(19|20)\d{2}")

    # Prefer the filename, then walk up the path until we find a 4-digit year.
    for part in [path.stem, *reversed(path.parts)]:
        match = year_pattern.search(part)
        if match:
            return int(match.group(0))

    raise ValueError(f"Could not extract a year from path: {file_path}")


# ── MAP function ───────────────────────────────────────────────────────────────

def analyze_file(args):
    """
    MAP: (file_path_str, dataset_name) → list of per-year quality dicts.

    Returns a list because a single file may (rarely) span two calendar years.
    On error returns a single error-flagged dict so the job does not crash.
    """
    file_path, dataset_name = args
    file_year = _extract_file_year(file_path)
    cfg = DATASET_CONFIGS[dataset_name]

    try:
        df = pd.read_parquet(file_path)
    except Exception as exc:
        return [{
            "dataset": dataset_name,
            "year": file_year,
            "total_rows": 0,
            "error": str(exc),
            **{col: 0 for col in ISSUE_COLS},
        }]

    total_rows = len(df)

    # ── Resolve actual column names ────────────────────────────────────────────
    cols = set(df.columns)

    pickup_col = next((c for c in cfg["pickup_cols"] if c in cols), None)
    dropoff_col = next((c for c in cfg["dropoff_cols"] if c in cols), None)
    distance_col = cfg["distance_col"] if cfg["distance_col"] in cols else None
    passenger_col = cfg["passenger_col"] if cfg["passenger_col"] in cols else None
    fare_col = cfg["fare_col"] if cfg["fare_col"] in cols else None

    # ── Parse datetime columns ────────────────────────────────────────────────
    if pickup_col is not None:
        pickup = pd.to_datetime(df[pickup_col], errors="coerce")
    else:
        pickup = pd.Series([pd.NaT] * total_rows, dtype="datetime64[ns]")

    if dropoff_col is not None:
        dropoff = pd.to_datetime(df[dropoff_col], errors="coerce")
    else:
        dropoff = pd.Series([pd.NaT] * total_rows, dtype="datetime64[ns]")

    # ── Group by actual pickup year ────────────────────────────────────────────
    pickup_year = pickup.dt.year.fillna(file_year).astype(int)

    results = []
    for year, grp_idx in pickup_year.groupby(pickup_year):
        n = len(grp_idx)
        p = pickup.loc[grp_idx.index]
        d = dropoff.loc[grp_idx.index]
        placeholder_dropoff = int((d == PLACEHOLDER_DROPOFF).sum())
        valid_duration = (
            d.notna()
            & (d != PLACEHOLDER_DROPOFF)
        )

        duration_h = (d - p).dt.total_seconds() / 3600.0

        # 1. invalid_year: pickup year not within ±1 of file_year
        # All rows in this group share the same pickup year, so it's all-or-nothing.
        invalid_year = n if abs(year - file_year) > 1 else 0

        # 2–11 quality checks
        null_pickup        = int(p.isna().sum())
        null_dropoff       = int(d.isna().sum())
        same_timestamps = int(
            ((p == d) & valid_duration).sum()
        )

        negative_duration = int(
            ((d < p) & valid_duration).sum()
        )

        excessive_duration = int(
            ((duration_h > 24) & valid_duration).sum()
        )

        if distance_col is not None:
            dist = pd.to_numeric(df[distance_col].loc[grp_idx.index], errors="coerce")
            zero_distance     = int((dist == 0).sum())
            negative_distance = int((dist < 0).sum())
        else:
            zero_distance = negative_distance = 0

        if passenger_col is not None:
            pax = pd.to_numeric(df[passenger_col].loc[grp_idx.index], errors="coerce")
            zero_passengers = int((pax == 0).sum())
        else:
            zero_passengers = 0

        if fare_col is not None:
            fare = pd.to_numeric(df[fare_col].loc[grp_idx.index], errors="coerce")
            negative_fare = int((fare < 0).sum())
            high_fare     = int((fare > 500).sum())
        else:
            negative_fare = high_fare = 0

        results.append({
            "dataset":            dataset_name,
            "year":               int(year),
            "total_rows":         n,
            "invalid_year":       invalid_year,
            "null_pickup":        null_pickup,
            "null_dropoff":       null_dropoff,
            "placeholder_dropoff": placeholder_dropoff,
            "same_timestamps":    same_timestamps,
            "negative_duration":  negative_duration,
            "zero_distance":      zero_distance,
            "negative_distance":  negative_distance,
            "zero_passengers":    zero_passengers,
            "negative_fare":      negative_fare,
            "excessive_duration": excessive_duration,
            "high_fare":          high_fare,
        })

    return results if results else [{
        "dataset": dataset_name,
        "year": file_year,
        "total_rows": 0,
        **{col: 0 for col in ISSUE_COLS},
    }]


# ── REDUCE functions ───────────────────────────────────────────────────────────

def reduce_key(record):
    return f"{record['dataset']}_{record['year']}"


def _merge_two(acc, record):
    """Accumulate numeric fields from two dicts with the same key."""
    merged = {"dataset": acc["dataset"], "year": acc["year"]}
    merged["total_rows"] = acc["total_rows"] + record["total_rows"]
    for col in ISSUE_COLS:
        merged[col] = acc.get(col, 0) + record.get(col, 0)
    return merged


def reduce_binop(acc, record):
    """Fold a new record into the accumulator (acc starts as first record)."""
    return _merge_two(acc, record)


def reduce_combine(a, b):
    """Merge two partial accumulators (for combine step)."""
    return _merge_two(a, b)


# ── Main ───────────────────────────────────────────────────────────────────────

def build_task_list():
    tasks = []
    for dataset in DATASET_CONFIGS:
        files = sorted(
            list(SHARED_DIR.glob(f"{dataset}_*.parquet")) +
            list(DOWNLOAD_DIR.glob(f"{dataset}_*.parquet"))
        )
        for f in files:
            tasks.append((str(f), dataset))
    return tasks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Start SLURM cluster ────────────────────────────────────────────────────
    cluster = SLURMCluster(
        queue="all",
        cores=1,
        memory="16GB",
        walltime="02:00:00",
        job_extra_directives=[
            f"--output={OUTPUT_DIR}/t2_worker_%j.out"
        ],
    )
    cluster.scale(8)
    client = Client(cluster)
    print(f"Dashboard: {client.dashboard_link}", flush=True)
    print("Waiting for workers...", flush=True)
    client.wait_for_workers(n_workers=4, timeout=300)
    print(f"Workers ready: {len(client.scheduler_info()['workers'])}", flush=True)

    # ── Build task list ────────────────────────────────────────────────────────
    tasks = build_task_list()
    print(f"Total files to process: {len(tasks)}", flush=True)
    for ds_name in DATASET_CONFIGS:
        n = sum(1 for _, d in tasks if d == ds_name)
        print(f"  {ds_name}: {n} files", flush=True)

    # ── Map-reduce via Dask bag ────────────────────────────────────────────────
    bag = db.from_sequence(tasks, npartitions=min(len(tasks), 64))

    # map: each task → list of per-year dicts; flatten so bag holds individual dicts
    flat = bag.map(analyze_file).flatten()

    # foldby reduce: group by (dataset, year), accumulate counts
    folded = flat.foldby(
        key=reduce_key,
        binop=reduce_binop,
        combine=reduce_combine,
    )

    print("Computing...", flush=True)
    results_kv = folded.compute()

    # results_kv is a list of (key, record) tuples — extract values
    results = [record for _, record in results_kv]
    results.sort(key=lambda r: (r["dataset"], r["year"]))

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} records to {OUTPUT_FILE}", flush=True)

    total_rows = sum(r["total_rows"] for r in results)
    print(f"Total rows processed: {total_rows:,}", flush=True)

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()
