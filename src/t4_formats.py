"""
T4: Format comparison for Green Taxi 2024 partition (~18 MB parquet).

Steps:
  1. Read the source parquet partition
  2. Export to: CSV, CSV.gz, HDF5, DuckDB database file
  3. Benchmark reading each format into a Pandas DataFrame (5 runs, median)
  4. Save results (file sizes + read times) to data/t4/benchmark.csv
"""

import time
import os
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import duckdb

# ── Paths ──────────────────────────────────────────────────────────────────────

SOURCE = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data"
              "/partitioned/green_tripdata/2024/part-0.parquet")
OUT    = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data/t4")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "parquet":  OUT / "green_2024.parquet",
    "csv":      OUT / "green_2024.csv",
    "csv_gz":   OUT / "green_2024.csv.gz",
    "hdf5":     OUT / "green_2024.h5",
    "duckdb":   OUT / "green_2024.duckdb",
}

N_RUNS = 5   # number of read repetitions for timing


# ── Step 1: Read source ────────────────────────────────────────────────────────

print("Reading source parquet...", flush=True)
df = pq.read_table(SOURCE).to_pandas()
print(f"  {len(df):,} rows × {len(df.columns)} columns", flush=True)

# Copy parquet to t4 dir (so all files are in one place)
import shutil
shutil.copy(SOURCE, FILES["parquet"])


# ── Step 2: Export to all formats ─────────────────────────────────────────────

print("\nExporting formats...", flush=True)

# CSV
df.to_csv(FILES["csv"], index=False)
print(f"  CSV      saved", flush=True)

# CSV gzipped
df.to_csv(FILES["csv_gz"], index=False, compression="gzip")
print(f"  CSV.gz   saved", flush=True)

# HDF5
df.to_hdf(FILES["hdf5"], key="green_2024", mode="w", format="table",
          complevel=0, complib="blosc")
print(f"  HDF5     saved", flush=True)

# DuckDB database file
if FILES["duckdb"].exists():
    FILES["duckdb"].unlink()
con = duckdb.connect(str(FILES["duckdb"]))
con.register("df_view", df)
con.execute("CREATE TABLE green_2024 AS SELECT * FROM df_view")
con.close()
print(f"  DuckDB   saved", flush=True)


# ── Step 3: File sizes ─────────────────────────────────────────────────────────

print("\nFile sizes:", flush=True)
sizes = {}
for fmt, path in FILES.items():
    size_mb = path.stat().st_size / 1e6
    sizes[fmt] = round(size_mb, 3)
    print(f"  {fmt:10s}  {size_mb:7.2f} MB", flush=True)


# ── Step 4: Read benchmarks ────────────────────────────────────────────────────

print(f"\nBenchmarking reads ({N_RUNS} runs each)...", flush=True)

def bench(fn):
    times = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return round(sorted(times)[N_RUNS // 2], 4)   # median

times = {}

times["parquet"] = bench(lambda: pd.read_parquet(FILES["parquet"]))
print(f"  parquet  {times['parquet']:.4f}s", flush=True)

times["csv"] = bench(lambda: pd.read_csv(FILES["csv"]))
print(f"  csv      {times['csv']:.4f}s", flush=True)

times["csv_gz"] = bench(lambda: pd.read_csv(FILES["csv_gz"], compression="gzip"))
print(f"  csv_gz   {times['csv_gz']:.4f}s", flush=True)

times["hdf5"] = bench(lambda: pd.read_hdf(FILES["hdf5"], key="green_2024"))
print(f"  hdf5     {times['hdf5']:.4f}s", flush=True)

def read_duckdb():
    c = duckdb.connect(str(FILES["duckdb"]), read_only=True)
    result = c.execute("SELECT * FROM green_2024").fetchdf()
    c.close()
    return result

times["duckdb"] = bench(read_duckdb)
print(f"  duckdb   {times['duckdb']:.4f}s", flush=True)


# ── Step 5: Save benchmark results ────────────────────────────────────────────

results = pd.DataFrame({
    "format":      list(sizes.keys()),
    "size_mb":     list(sizes.values()),
    "read_time_s": [times[k] for k in sizes.keys()],
})
results["rows"] = len(df)
results["columns"] = len(df.columns)
results.to_csv(OUT / "benchmark.csv", index=False)

print("\n── Results ──────────────────────────────────────────────")
print(results.to_string(index=False))
print(f"\nSaved to {OUT / 'benchmark.csv'}")
