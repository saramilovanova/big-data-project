"""
T1: Read ALL original parquet files for each TLC dataset, add a 'year' column
derived from the pickup datetime, and write output partitioned by year with
row groups of ~2M rows (~100-200 MB).

Schema drift across files is handled by:
  1. Reading all schemas upfront and unifying them with pa.unify_schemas()
  2. Manually aligning every batch to the unified schema
     (missing columns filled with nulls, existing columns cast to unified types)

All files are processed in a single streaming write_dataset call.
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SHARED_DIR   = Path("/d/hpc/projects/FRI/bigdata/data/Taxi")
DOWNLOAD_DIR = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/taxi_new")
OUTPUT_DIR   = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data/partitioned")

DATASETS = {
    "yellow_tripdata":  "tpep_pickup_datetime",
    "green_tripdata":   "lpep_pickup_datetime",
    "fhv_tripdata":     "pickup_datetime",
    "fhvhv_tripdata":   "pickup_datetime",
}

ROWS_PER_GROUP = 2_000_000
BATCH_SIZE     = 200_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def file_year(path):
    return int(path.stem.split("_")[-1].split("-")[0])


def safe_unify_schemas(schemas):
    """
    Unify schemas field by field.
    - Single type across all files  → keep as-is
    - Numeric promotion possible    → promote (via pa.unify_schemas)
    - Truly incompatible (e.g. string vs int64) → fall back to large_string
    """
    # Collect field names in order of first appearance
    seen, ordered_names = set(), []
    for schema in schemas:
        for field in schema:
            if field.name not in seen:
                ordered_names.append(field.name)
                seen.add(field.name)

    unified_fields = []
    for name in ordered_names:
        types = list({schema.field(name).type for schema in schemas
                      if name in schema.names})
        if len(types) == 1:
            unified_fields.append(pa.field(name, types[0]))
        else:
            try:
                mini = pa.unify_schemas(
                    [pa.schema([pa.field(name, t)]) for t in types],
                    promote_options="permissive",
                )
                unified_fields.append(mini.field(name))
            except Exception:
                # incompatible types → store as string
                unified_fields.append(pa.field(name, pa.large_string()))

    return pa.schema(unified_fields)


def align_batch(batch, unified_schema):
    """
    Return a RecordBatch with exactly unified_schema columns:
      - columns present in batch → cast to unified type (null on failure)
      - columns missing in batch → null array of unified type
    """
    arrays = []
    batch_names = set(batch.schema.names)
    for field in unified_schema:
        if field.name in batch_names:
            col = batch.column(field.name)
            if col.type != field.type:
                try:
                    col = col.cast(field.type, safe=False)
                except Exception:
                    col = pa.nulls(len(batch), type=field.type)
        else:
            col = pa.nulls(len(batch), type=field.type)
        arrays.append(col)
    return pa.RecordBatch.from_arrays(arrays, schema=unified_schema)


# ── Per-dataset processing ────────────────────────────────────────────────────

def process(prefix, pickup_col):
    out_dir = OUTPUT_DIR / prefix
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(
        list(SHARED_DIR.glob(f"{prefix}_*.parquet")) +
        list(DOWNLOAD_DIR.glob(f"{prefix}_*.parquet"))
    )

    if not all_files:
        print(f"\n{prefix}: no files found", flush=True)
        return

    print(f"\n{prefix}: {len(all_files)} files", flush=True)

    # Step 1 — unified schema across all files (reads only footer metadata)
    all_schemas = [pq.read_schema(f) for f in all_files]
    unified = safe_unify_schemas(all_schemas)
    out_schema = unified.append(pa.field("year", pa.int32()))
    print(f"  Unified schema: {len(unified)} columns", flush=True)

    # Step 2 — stream all files through a single generator
    def all_batches():
        for i, f in enumerate(all_files):
            yr = file_year(f)
            print(f"  [{i+1}/{len(all_files)}] {f.name}", flush=True)
            file_schema = pq.read_schema(f)
            dataset = ds.dataset(str(f), format="parquet", schema=file_schema)
            for batch in dataset.to_batches(batch_size=BATCH_SIZE):
                # extract year from pickup col BEFORE aligning (uses original col names)
                if pickup_col in batch.schema.names:
                    pickup   = batch.column(pickup_col)
                    year_arr = pc.if_else(
                        pc.is_valid(pickup),
                        pc.year(pickup).cast(pa.int32()),
                        pa.scalar(yr, pa.int32()),
                    )
                else:
                    year_arr = pa.array([yr] * len(batch), type=pa.int32())

                aligned = align_batch(batch, unified)
                yield aligned.append_column(pa.field("year", pa.int32()), year_arr)

    # Step 3 — single write_dataset call, partitioned by year
    reader = pa.RecordBatchReader.from_batches(out_schema, all_batches())
    ds.write_dataset(
        reader,
        base_dir=str(out_dir),
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("year", pa.int32())])),
        max_rows_per_group=ROWS_PER_GROUP,
        existing_data_behavior="overwrite_or_ignore",
    )

    written = sorted(out_dir.rglob("*.parquet"))
    years   = sorted({p.parent.name for p in written})
    print(f"  Done: {len(written)} output files, years={years}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for prefix, pickup_col in DATASETS.items():
        process(prefix, pickup_col)
    print("\nAll done.")

if __name__ == "__main__":
    main()
