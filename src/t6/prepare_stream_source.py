"""
Builds the single source file the Kafka producer streams from:
  - reads T1 output (NOT T5-augmented files) for Yellow Taxi and FHVHV, year 2021
  - aligns both datasets onto one shared schema (column names + units differ
    between the two raw datasets, see the SELECT lists below)
  - joins the static TLC taxi-zone lookup to attach PU/DO borough + zone names
  - applies the data-quality filters identified in T2 (dropoff before pickup,
    zero/garbage distances, timestamps outside the inspected year, etc.) --
    T2 explicitly asks for these insights to be reused in later steps
  - optionally samples a fixed percentage of rows (SAMPLE_PCT in config.py)
    to keep the demo tractable on a laptop, while preserving the temporal
    pattern across the whole year
  - sorts the combined result by pickup_datetime and writes it to a single
    parquet file -- T6 explicitly requires events to be produced "ordered by
    pickup timestamp"

"""

import duckdb
from pathlib import Path

from config import (
    T1_YELLOW_GLOB,
    T1_FHVHV_GLOB,
    TAXI_ZONE_LOOKUP_CSV,
    COMBINED_PARQUET,
    SAMPLE_PCT,
    YEAR,
    BASE_DIR,
)

con = duckdb.connect()

con.execute("PRAGMA memory_limit='4GB'")
con.execute(f"PRAGMA temp_directory='{BASE_DIR / 'duckdb_tmp'}'")
con.execute("PRAGMA threads=4")

Path(COMBINED_PARQUET).parent.mkdir(parents=True, exist_ok=True)

con.execute(f"""
    CREATE OR REPLACE TEMP VIEW zones AS
    SELECT
        CAST(LocationID AS BIGINT) AS LocationID,
        Borough,
        Zone
    FROM read_csv_auto('{TAXI_ZONE_LOOKUP_CSV}')
""")

# --- Yellow Taxi: unify column names/units onto the shared schema ---------
con.execute(f"""
    CREATE OR REPLACE TEMP VIEW yellow_unified AS
    SELECT
        tpep_pickup_datetime  AS pickup_datetime,
        tpep_dropoff_datetime AS dropoff_datetime,
        PULocationID,
        DOLocationID,
        trip_distance,
        fare_amount,
        tip_amount,
        total_amount,
        date_diff('minute', tpep_pickup_datetime, tpep_dropoff_datetime)
            AS trip_duration_min,
        0 AS is_fhvhv,
        'yellow' AS source
    FROM read_parquet('{T1_YELLOW_GLOB}')
    WHERE year(tpep_pickup_datetime) = {YEAR}
""")

# --- FHVHV: same shared schema, different source column names -------------
con.execute(f"""
    CREATE OR REPLACE TEMP VIEW fhvhv_unified AS
    SELECT
        pickup_datetime,
        dropoff_datetime,
        PULocationID,
        DOLocationID,
        trip_miles AS trip_distance,
        base_passenger_fare AS fare_amount,
        tips AS tip_amount,
        (base_passenger_fare + COALESCE(tolls, 0) + COALESCE(bcf, 0)
         + COALESCE(sales_tax, 0) + COALESCE(congestion_surcharge, 0)
         + COALESCE(airport_fee, 0) + COALESCE(tips, 0)) AS total_amount,
        date_diff('minute', pickup_datetime, dropoff_datetime)
            AS trip_duration_min,
        1 AS is_fhvhv,
        'fhvhv' AS source
    FROM read_parquet('{T1_FHVHV_GLOB}')
    WHERE year(pickup_datetime) = {YEAR}
""")

# --- Combine, attach zone/borough names, apply T2-style quality filters ---
con.execute(f"""
    CREATE OR REPLACE TEMP VIEW combined AS
    SELECT
        c.pickup_datetime,
        c.dropoff_datetime,
        c.PULocationID,
        c.DOLocationID,
        puz.Borough AS PU_Borough,
        puz.Zone    AS PU_Zone,
        doz.Borough AS DO_Borough,
        doz.Zone    AS DO_Zone,
        c.trip_distance,
        c.fare_amount,
        c.tip_amount,
        c.total_amount,
        c.trip_duration_min,
        c.is_fhvhv,
        c.source
    FROM (
        SELECT * FROM yellow_unified
        UNION ALL
        SELECT * FROM fhvhv_unified
    ) c
    LEFT JOIN zones puz ON CAST(c.PULocationID AS BIGINT) = puz.LocationID
    LEFT JOIN zones doz ON CAST(c.DOLocationID AS BIGINT) = doz.LocationID
    WHERE c.dropoff_datetime > c.pickup_datetime   -- T2: dropoff before/at pickup
      AND c.trip_distance > 0                       -- T2: zero-distance trips
      AND c.trip_distance < 200                     -- T2: implausible distances
      AND c.fare_amount >= 0
      AND c.total_amount BETWEEN 0 AND 1000
      AND year(c.pickup_datetime) = {YEAR}           -- T2: stray cross-year timestamps
""")

if SAMPLE_PCT < 100:
    con.execute(f"""
        CREATE OR REPLACE TEMP VIEW sampled AS
        SELECT * FROM combined USING SAMPLE {SAMPLE_PCT}% (bernoulli)
    """)
    source_view = "sampled"
else:
    source_view = "combined"

n_before = con.sql("SELECT COUNT(*) FROM combined").fetchone()[0]
print(f"Rows after T2-style quality filters, before sampling: {n_before:,}")

con.execute(f"""
    COPY (
        SELECT * FROM {source_view}
        ORDER BY pickup_datetime
    ) TO '{COMBINED_PARQUET}'
    (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 200000)
""")

n_after = con.sql(f"SELECT COUNT(*) FROM read_parquet('{COMBINED_PARQUET}')").fetchone()[0]
print(f"Rows written to {COMBINED_PARQUET}: {n_after:,} (SAMPLE_PCT={SAMPLE_PCT})")

by_source = con.sql(f"""
    SELECT source, COUNT(*) AS n
    FROM read_parquet('{COMBINED_PARQUET}')
    GROUP BY source
""").df()
print(by_source)