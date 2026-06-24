# config.py
# Shared configuration for the T6 streaming pipeline.
# Edit the paths in the DATA SOURCES section to match your own T1 output layout.
 
from pathlib import Path
 
# All paths below that are *not* explicitly absolute are resolved relative
# to this file's own folder, not to whatever directory you happen to launch
# `python` from -- so it doesn't matter whether you run scripts from
# Assignment_6\ or Assignment_6\project\, the output always lands in the
# same place.
BASE_DIR = Path(__file__).resolve().parent
 
# -----------------------------------------------------------------------
# DATA SOURCES (T1 output -- NOT the T5-augmented files, per the T6 spec:
# "reading data from processed Parquet datasets, as created in T1")
# -----------------------------------------------------------------------
 
YEAR = 2021

# Point these at your T1 output for each dataset. T1 repartitions by year,
# so this is typically something like ".../yellow/year=2021/*.parquet".
# If your T1 layout isn't Hive-partitioned, point at the whole dataset glob
# instead -- prepare_stream_source.py filters by year explicitly either way.
T1_YELLOW_GLOB = "D:\\Projects\\bd\\Project\\data\\processed\\yellow\\2021\\part-0.parquet"
T1_FHVHV_GLOB = "D:\\Projects\\bd\\Project\\data\\processed\\fhvhv\\2021\\part-0.parquet"

# Small static dimension table: LocationID -> Borough/Zone. This is the
# standard TLC taxi_zone_lookup.csv (265 rows), not the heavier T5
# enrichment (weather/schools/businesses) -- just the zone/borough mapping.
TAXI_ZONE_LOOKUP_CSV = "D:\\Projects\\bd\\Project\\data\\taxi_zone_lookup.csv"


# Output of prepare_stream_source.py: a single, globally sorted-by-pickup
# parquet file combining Yellow + FHVHV for YEAR.
COMBINED_PARQUET = str(BASE_DIR / "data" / f"streaming_source_{YEAR}.parquet")
 
# Local hardware is resource-constrained (7.8 GB RAM laptop), and FHVHV
# alone runs into the hundreds of millions of rows/year. SAMPLE_PCT keeps
# the demo tractable while still covering the full year. Set to 100 to use
# every row if you have the time/resources, or run it overnight.
SAMPLE_PCT = 5  # percent, applied uniformly so the temporal pattern across
                # the year (seasonality, day-of-week, hour-of-day) is preserved
 
# -----------------------------------------------------------------------
# KAFKA
# -----------------------------------------------------------------------
 
BOOTSTRAP = "localhost:10000,localhost:10001"
 
TOPIC_RAW = f"taxi-events-{YEAR}"           # combined Yellow + FHVHV stream
TOPIC_BOROUGH_STATS = "taxi-borough-stats"
TOPIC_LOCATION_STATS = "taxi-location-stats"
TOPIC_PY_BOROUGH_STATS = "taxi-borough-stats-python"
TOPIC_PY_LOCATION_STATS = "taxi-location-stats-python"
TOPIC_CLUSTERS = "taxi-clusters"
 
WINDOW_SECONDS = 30
 
# Fill this in using find_top_locations.py (run it once against
# COMBINED_PARQUET and paste the printed list here).
TOP10_LOCATIONS = [265.0, 79.0, 132.0, 237.0, 61.0, 236.0, 161.0, 138.0, 170.0, 234.0]
 
# At least 3 attributes, all present (with consistent meaning) in both the
# Yellow and FHVHV unified schema -- see prepare_stream_source.py.
STATS_FIELDS = ["trip_distance", "fare_amount", "tip_amount"]
 
# Clustering features: T1-level attributes only (no T5 join required).
# is_fhvhv lets the clusters separate by vehicle type as well as trip shape.
CLUSTER_FEATURES = [
    "trip_distance",
    "fare_amount",
    "tip_amount",
    "total_amount",
    "trip_duration_min",
    "is_fhvhv",
]
 
FEATURE_SCALES = {
    "trip_distance": 12.0,
    "fare_amount": 40.0,
    "tip_amount": 8.0,
    "total_amount": 50.0,
    "trip_duration_min": 30.0,
    "is_fhvhv": 1.0,
}
 
N_CLUSTERS = 5
 
# -----------------------------------------------------------------------
# LOCAL OUTPUT (durable, independent of Kafka topic retention)
# -----------------------------------------------------------------------
 
OUTPUT_DIR = str(BASE_DIR / "outputs")