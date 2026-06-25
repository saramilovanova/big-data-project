"""
T0: Download missing TLC taxi files (2025-02 to 2026-02) using Dask Bags
in a Map-Reduce manner.

  MAP      : for each file -> download it, extract schema (columns + dtypes)
  REDUCE   : group by dataset prefix, accumulate list of schemas  (foldby)
  FINALIZER: compare schemas within each group, save to schema_comparison.json
"""

import dask.bag as db
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from pathlib import Path
import pyarrow.parquet as pq
import urllib.request
import json

# ── Config ────────────────────────────────────────────────────────────────────

SHARED_DIR   = Path("/d/hpc/projects/FRI/bigdata/data/Taxi")
DOWNLOAD_DIR = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/taxi_new")
BASE_URL     = "https://d37ci6vzurychx.cloudfront.net/trip-data"
OUTPUT_JSON  = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/schema_comparison.json")

DATASETS = {
    "yellow_tripdata": (2012, 1),
    "green_tripdata":  (2014, 1),
    "fhv_tripdata":    (2015, 1),
    "fhvhv_tripdata":  (2019, 2),
}
END = (2026, 2)


def months_range(start, end):
    y, m = start
    while (y, m) <= end:
        yield y, m
        m += 1
        if m > 12:
            m, y = 1, y + 1


def build_task_list():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tasks = []
    for prefix, start in DATASETS.items():
        for year, month in months_range(start, END):
            fname = f"{prefix}_{year}-{month:02d}.parquet"
            if (SHARED_DIR / fname).exists() or (DOWNLOAD_DIR / fname).exists():
                continue
            tasks.append((fname, f"{BASE_URL}/{fname}", str(DOWNLOAD_DIR / fname)))
    return tasks


# ── MAP ───────────────────────────────────────────────────────────────────────

def map_download_schema(item):
    fname, url, dest = item
    prefix = "_".join(fname.split("_")[:2])
    result = {"file": fname, "prefix": prefix, "columns": None, "dtypes": None, "error": None}

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
            f.write(resp.read())
    except Exception as e:
        result["error"] = str(e)
        return result

    try:
        schema = pq.read_schema(dest)
        result["columns"] = schema.names
        result["dtypes"]  = {field.name: str(field.type) for field in schema}
    except Exception as e:
        result["error"] = str(e)

    return result


# ── REDUCE ────────────────────────────────────────────────────────────────────

def reduce_key(record):
    return record["prefix"]

def reduce_binop(accumulator, record):
    return accumulator + [record]

def reduce_combine(acc_a, acc_b):
    return acc_a + acc_b


# ── FINALIZER ─────────────────────────────────────────────────────────────────

def finalizer(grouped_results):
    report = {}

    for prefix, records in sorted(grouped_results):
        ok     = [r for r in records if r["columns"] is not None]
        errors = [r for r in records if r["error"]]
        ok.sort(key=lambda r: r["file"])

        dataset_report = {
            "files_ok":          [r["file"] for r in ok],
            "files_error":       {r["file"]: r["error"] for r in errors},
            "reference_file":    ok[0]["file"] if ok else None,
            "reference_schema":  ok[0]["dtypes"] if ok else {},
            "consistent_columns": True,
            "consistent_dtypes":  True,
            "differences":       [],
        }

        if ok:
            ref_cols   = set(ok[0]["columns"])
            ref_dtypes = ok[0]["dtypes"]
            for r in ok[1:]:
                cols = set(r["columns"])
                if cols != ref_cols:
                    dataset_report["consistent_columns"] = False
                    for col in sorted(cols - ref_cols):
                        dataset_report["differences"].append(
                            {"file": r["file"], "type": "extra_column", "column": col})
                    for col in sorted(ref_cols - cols):
                        dataset_report["differences"].append(
                            {"file": r["file"], "type": "missing_column", "column": col})
                for col, dtype in r["dtypes"].items():
                    if col in ref_dtypes and dtype != ref_dtypes[col]:
                        dataset_report["consistent_dtypes"] = False
                        dataset_report["differences"].append(
                            {"file": r["file"], "type": "dtype_mismatch",
                             "column": col, "found": dtype, "expected": ref_dtypes[col]})

        report[prefix] = dataset_report

    OUTPUT_JSON.write_text(json.dumps(report, indent=2))
    print(f"Schema comparison saved to {OUTPUT_JSON}")


# ── Main ──────────────────────────────────────────────────────────────────────

def schema_from_file(path):
    """Extract schema from an already-downloaded file (no download needed)."""
    fname  = Path(path).name
    prefix = "_".join(fname.split("_")[:2])
    result = {"file": fname, "prefix": prefix, "columns": None, "dtypes": None, "error": None}
    try:
        schema = pq.read_schema(path)
        result["columns"] = schema.names
        result["dtypes"]  = {field.name: str(field.type) for field in schema}
    except Exception as e:
        result["error"] = str(e)
    return result


def main():
    tasks = build_task_list()
    print(f"Missing files to download: {len(tasks)}")
    for fname, _, _ in tasks:
        print(f"  {fname}")

    if tasks:
        cluster = SLURMCluster(
            queue="all",
            cores=1,
            memory="16GB",
            walltime="01:00:00",
        )
        cluster.scale(min(8, len(tasks)))
        client = Client(cluster)
        print(f"Dask dashboard: {client.dashboard_link}")

        bag = db.from_sequence(tasks, npartitions=min(8, len(tasks)))

        # MAP
        mapped = bag.map(map_download_schema)

        # REDUCE
        reduced = mapped.foldby(
            key=reduce_key,
            binop=reduce_binop,
            initial=[],
            combine=reduce_combine,
            combine_initial=[],
        )

        grouped_results = reduced.compute()

        client.close()
        cluster.close()

    else:
        # No new files — build grouped_results from files already in DOWNLOAD_DIR
        print("No new files to download — reading schemas from existing downloads.")
        existing = sorted(DOWNLOAD_DIR.glob("*.parquet"))
        records  = [schema_from_file(p) for p in existing]
        groups   = {}
        for r in records:
            groups.setdefault(r["prefix"], []).append(r)
        grouped_results = list(groups.items())

    # FINALIZER — always runs
    finalizer(grouped_results)





if __name__ == "__main__":
    main()
