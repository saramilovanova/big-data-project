"""
Yellow Taxi schema normalization.
Renames all historical column variants to canonical names,
merges Airport_fee -> airport_fee, drops artifacts,
casts everything to target types, writes partitioned by year.
"""

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

SHARED_DIR   = Path("/d/hpc/projects/FRI/bigdata/data/Taxi")
DOWNLOAD_DIR = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/taxi_new")
OUTPUT_DIR   = Path("/d/hpc/projects/FRI/bigdata/students/sm_bv/final_project/data/yellow_normalized")

PICKUP_COL   = "tpep_pickup_datetime"
ROWS_PER_GROUP = 2_000_000
BATCH_SIZE     = 200_000

# ── Column rename map (old name -> canonical name) ────────────────────────────

RENAME_MAP = {
    # vendor
    "vendor_name"           : "VendorID",
    "vendor_id"             : "VendorID",
    # pickup datetime
    "Trip_Pickup_DateTime"  : "tpep_pickup_datetime",
    "pickup_datetime"       : "tpep_pickup_datetime",
    # dropoff datetime
    "Trip_Dropoff_DateTime" : "tpep_dropoff_datetime",
    "dropoff_datetime"      : "tpep_dropoff_datetime",
    # passenger / distance
    "Passenger_Count"       : "passenger_count",
    "Trip_Distance"         : "trip_distance",
    # rate code
    "Rate_Code"             : "RatecodeID",
    "rate_code"             : "RatecodeID",
    # store and forward
    "store_and_forward"     : "store_and_fwd_flag",
    # payment
    "Payment_Type"          : "payment_type",
    # fares
    "Fare_Amt"              : "fare_amount",
    "surcharge"             : "extra",
    "Tip_Amt"               : "tip_amount",
    "Tolls_Amt"             : "tolls_amount",
    "Total_Amt"             : "total_amount",
    # location (lat/lon era)
    "Start_Lon"             : "pickup_longitude",
    "Start_Lat"             : "pickup_latitude",
    "End_Lon"               : "dropoff_longitude",
    "End_Lat"               : "dropoff_latitude",
    # airport fee capitalisation fix
    "Airport_fee"           : "airport_fee",
}

COLS_TO_DROP = {"__index_level_0__"}

# ── Target unified schema ─────────────────────────────────────────────────────

UNIFIED_SCHEMA = pa.schema([
    pa.field("VendorID",              pa.float64()),
    pa.field("tpep_pickup_datetime",  pa.timestamp("us")),
    pa.field("tpep_dropoff_datetime", pa.timestamp("us")),
    pa.field("passenger_count",       pa.float64()),
    pa.field("trip_distance",         pa.float64()),
    pa.field("RatecodeID",            pa.float64()),
    pa.field("store_and_fwd_flag",    pa.string()),
    pa.field("pickup_longitude",      pa.float64()),   # null for 2011+
    pa.field("pickup_latitude",       pa.float64()),   # null for 2011+
    pa.field("dropoff_longitude",     pa.float64()),   # null for 2011+
    pa.field("dropoff_latitude",      pa.float64()),   # null for 2011+
    pa.field("PULocationID",          pa.float64()),   # null for 2009-2010
    pa.field("DOLocationID",          pa.float64()),   # null for 2009-2010
    pa.field("payment_type",          pa.float64()),
    pa.field("fare_amount",           pa.float64()),
    pa.field("extra",                 pa.float64()),
    pa.field("mta_tax",               pa.float64()),
    pa.field("tip_amount",            pa.float64()),
    pa.field("tolls_amount",          pa.float64()),
    pa.field("improvement_surcharge", pa.float64()),
    pa.field("total_amount",          pa.float64()),
    pa.field("congestion_surcharge",  pa.float64()),
    pa.field("airport_fee",           pa.float64()),
    pa.field("cbd_congestion_fee",    pa.float64()),
    pa.field("year",                  pa.int32()),
])


# ── Batch normalization ───────────────────────────────────────────────────────

def normalize_batch(batch, file_year):
    # Step 1: rename columns
    names = batch.schema.names
    new_names = [RENAME_MAP.get(n, n) for n in names]
    batch = batch.rename_columns(new_names)

    # Step 2: merge Airport_fee into airport_fee (coalesce)
    # After renaming both map to "airport_fee" - if duplicated, take first non-null
    if batch.schema.names.count("airport_fee") > 1:
        idx = [i for i, n in enumerate(batch.schema.names) if n == "airport_fee"]
        merged = pc.coalesce(*[batch.column(i) for i in idx])
        # Remove duplicate columns and add merged one
        cols_to_keep = [i for i in range(batch.num_columns)
                        if batch.schema.names[i] != "airport_fee" or i == idx[0]]
        arrays = [batch.column(i) for i in cols_to_keep]
        names  = [batch.schema.names[i] for i in cols_to_keep]
        arrays[names.index("airport_fee")] = merged
        batch = pa.RecordBatch.from_arrays(arrays, names=names)

    # Step 3: drop artifact columns
    for col in COLS_TO_DROP:
        if col in batch.schema.names:
            idx = batch.schema.get_field_index(col)
            batch = batch.remove_column(idx)

    # Step 4: derive year from pickup datetime (before type casting)
    if PICKUP_COL in batch.schema.names:
        pickup = batch.column(PICKUP_COL)
        # handle string timestamps (2009-2010 files)
        if pickup.type in (pa.string(), pa.large_string()):
            try:
                pickup = pc.strptime(pickup, format="%Y-%m-%d %H:%M:%S", unit="us")
            except Exception:
                pickup = pickup.cast(pa.timestamp("us"), safe=False)
        year_arr = pc.if_else(
            pc.is_valid(pickup),
            pc.year(pickup).cast(pa.int32()),
            pa.scalar(file_year, pa.int32()),
        )
    else:
        year_arr = pa.array([file_year] * len(batch), type=pa.int32())

    # Step 5: align to UNIFIED_SCHEMA (cast existing, nulls for missing)
    batch_names = {batch.schema.names[i]: i for i in range(batch.num_columns)}
    arrays = []
    for field in UNIFIED_SCHEMA:
        if field.name == "year":
            arrays.append(year_arr)
            continue
        if field.name in batch_names:
            col = batch.column(batch_names[field.name])
            if col.type != field.type:
                try:
                    col = col.cast(field.type, safe=False)
                except Exception:
                    col = pa.nulls(len(batch), type=field.type)
        else:
            col = pa.nulls(len(batch), type=field.type)
        arrays.append(col)

    return pa.RecordBatch.from_arrays(arrays, schema=UNIFIED_SCHEMA)


# ── Main processing ───────────────────────────────────────────────────────────

def main():
    files = sorted(
        list(SHARED_DIR.glob("yellow_tripdata_*.parquet")) +
        list(DOWNLOAD_DIR.glob("yellow_tripdata_*.parquet"))
    )
    print(f"yellow_tripdata: {len(files)} files", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def all_batches():
        for i, f in enumerate(files):
            yr = int(f.stem.split("_")[-1].split("-")[0])
            print(f"  [{i+1}/{len(files)}] {f.name}", flush=True)
            file_schema = pq.read_schema(f)
            dataset = ds.dataset(str(f), format="parquet", schema=file_schema)
            for batch in dataset.to_batches(batch_size=BATCH_SIZE):
                yield normalize_batch(batch, yr)

    reader = pa.RecordBatchReader.from_batches(UNIFIED_SCHEMA, all_batches())

    ds.write_dataset(
        reader,
        base_dir=str(OUTPUT_DIR),
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("year", pa.int32())])),
        max_rows_per_group=ROWS_PER_GROUP,
        existing_data_behavior="overwrite_or_ignore",
    )

    written = sorted(OUTPUT_DIR.rglob("*.parquet"))
    years   = sorted({p.parent.name for p in written})
    print(f"\nDone: {len(written)} files, years={years}", flush=True)


if __name__ == "__main__":
    main()
