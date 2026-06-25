# Shared configuration for the T6 streaming pipeline.
 
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

YEAR = 2021

T1_YELLOW_GLOB = "D:\\Projects\\bd\\Project\\data\\processed\\yellow\\2021\\part-0.parquet"
T1_FHVHV_GLOB = "D:\\Projects\\bd\\Project\\data\\processed\\fhvhv\\2021\\part-0.parquet"

TAXI_ZONE_LOOKUP_CSV = "D:\\Projects\\bd\\Project\\data\\taxi_zone_lookup.csv"


# Output of prepare_stream_source.py: a single, globally sorted-by-pickup
# parquet file combining Yellow + FHVHV for YEAR.
COMBINED_PARQUET = str(BASE_DIR / "data" / f"streaming_source_{YEAR}.parquet")

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

TOP10_LOCATIONS = [265.0, 79.0, 132.0, 237.0, 61.0, 236.0, 161.0, 138.0, 170.0, 234.0]

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