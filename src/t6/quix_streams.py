"""
Consumes the combined taxi event stream and computes rolling (tumbling
30-second window) descriptive statistics -- count, mean, std, min, max --
for STATS_FIELDS (trip_distance, fare_amount, tip_amount), grouped by:
  - pickup borough (PU_Borough)
  - the 10 highest-volume pickup locations (TOP10_LOCATIONS)

Windows are computed over event time (the trip's pickup_datetime), not over
Kafka's own record timestamp or wall-clock receive time. producer.py
deliberately leaves the Kafka message timestamp at its default (real send
time), since Kafka's log retention also keys off that same metadata and
would otherwise treat 2021 trip data as years "stale" the instant it's
written (see producer.py's docstring). Event time for windowing is instead
extracted directly from the pickup_datetime field in the JSON payload via
a custom timestamp_extractor passed to app.topic().
"""

import math
import argparse

from datetime import timedelta, datetime, timezone

from quixstreams import Application

from config import (
    BOOTSTRAP, TOPIC_RAW, TOPIC_BOROUGH_STATS, TOPIC_LOCATION_STATS,
    WINDOW_SECONDS, TOP10_LOCATIONS, STATS_FIELDS, OUTPUT_DIR,
)
from sinks import JsonlWriter

borough_seen = 0
location_seen = 0
location_matched = 0


def extract_pickup_timestamp(value, headers, timestamp, timestamp_type):
    """Use the trip's pickup_datetime (from the payload) as the event-time
    for windowing, falling back to the Kafka record timestamp if missing
    or unparseable."""
    pickup = value.get("pickup_datetime") if isinstance(value, dict) else None
    if not pickup:
        return timestamp
    try:
        dt = datetime.fromisoformat(pickup)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (TypeError, ValueError):
        return timestamp


def initializer(value):
    """Create the first accumulator for each borough/location in each window."""
    acc = {"n": 0}
    for f in STATS_FIELDS:
        acc[f"sum_{f}"] = 0.0
        acc[f"ssq_{f}"] = 0.0
        acc[f"min_{f}"] = float("inf")
        acc[f"max_{f}"] = float("-inf")
    return reducer(acc, value)


def reducer(acc, value):
    """Update the accumulator with one incoming taxi trip."""
    acc["n"] += 1
    for f in STATS_FIELDS:
        v = value.get(f) or 0.0
        acc[f"sum_{f}"] += v
        acc[f"ssq_{f}"] += v**2
        acc[f"min_{f}"] = min(acc[f"min_{f}"], v)
        acc[f"max_{f}"] = max(acc[f"max_{f}"], v)
    return acc


def finalize(window_result):
    """Calculate mean, standard deviation, min, and max for a completed window."""
    acc = window_result["value"]
    n = acc["n"]
    out = {
        "count": n,
        "window_start": str(window_result["start"]),
        "window_end": str(window_result["end"]),
    }
    for f in STATS_FIELDS:
        mean = acc[f"sum_{f}"] / n
        var = acc[f"ssq_{f}"] / n - mean**2
        out[f"{f}_mean"] = round(mean, 4)
        out[f"{f}_std"] = round(math.sqrt(max(var, 0)), 4)
        out[f"{f}_min"] = round(acc[f"min_{f}"], 4)
        out[f"{f}_max"] = round(acc[f"max_{f}"], 4)
    return out


def print_borough_progress(row):
    global borough_seen
    borough_seen += 1
    if borough_seen % 5000 == 0:
        print(f"Borough pipeline consumed {borough_seen} taxi events")


def print_location_progress(row):
    global location_seen
    location_seen += 1
    if location_seen % 5000 == 0:
        print(f"Location pipeline consumed {location_seen} taxi events before filter")


def print_location_match_progress(row):
    global location_matched
    location_matched += 1
    if location_matched % 1000 == 0:
        print(f"Location pipeline matched {location_matched} top-location taxi events")


def is_top_location(row):
    try:
        location_id = int(row["PULocationID"])
    except (KeyError, TypeError, ValueError):
        return False
    return location_id in TOP10_LOCATIONS


def stringify_location_id(row):
    return {**row, "PULocationID": str(int(row["PULocationID"]))}


def run_borough_stats():
    app = Application(
        broker_address=BOOTSTRAP,
        consumer_group="borough-stats",
        auto_offset_reset="earliest",
    )

    src = app.topic(TOPIC_RAW, value_deserializer="json", timestamp_extractor=extract_pickup_timestamp)
    dst = app.topic(TOPIC_BOROUGH_STATS, value_serializer="json")
    sdf = app.dataframe(src)
    sdf = sdf.update(print_borough_progress)

    sdf = sdf.group_by("PU_Borough")
    sdf = (
        sdf.tumbling_window(timedelta(seconds=WINDOW_SECONDS))
        .reduce(reducer=reducer, initializer=initializer)
        .final()
    )

    sdf = sdf.apply(finalize)

    writer = JsonlWriter(f"{OUTPUT_DIR}/borough_stats.jsonl")
    sdf = sdf.update(lambda value, key, ts, headers: writer.write({**value, "PU_Borough": key}), metadata=True)

    sdf.to_topic(dst)

    print("Running borough statistics...")
    try:
        app.run()
    finally:
        writer.close()


def run_location_stats():
    app = Application(
        broker_address=BOOTSTRAP,
        consumer_group="location-stats",
        auto_offset_reset="earliest",
    )

    src = app.topic(TOPIC_RAW, value_deserializer="json", timestamp_extractor=extract_pickup_timestamp)
    dst = app.topic(TOPIC_LOCATION_STATS, value_serializer="json")
    sdf = app.dataframe(src)
    sdf = sdf.update(print_location_progress)
    sdf = sdf.filter(is_top_location)
    sdf = sdf.update(print_location_match_progress)
    sdf = sdf.apply(stringify_location_id)
    sdf = sdf.group_by("PULocationID")

    sdf = (
        sdf.tumbling_window(timedelta(seconds=WINDOW_SECONDS))
        .reduce(reducer=reducer, initializer=initializer)
        .final()
    )

    sdf = sdf.apply(finalize)

    writer = JsonlWriter(f"{OUTPUT_DIR}/location_stats.jsonl")
    sdf = sdf.update(lambda value, key, ts, headers: writer.write({**value, "PULocationID": key}), metadata=True)

    sdf.to_topic(dst)

    print("Running location statistics...")
    try:
        app.run()
    finally:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one Quix Streams pipeline.")
    parser.add_argument("pipeline", choices=["borough", "location"],
                         help="Pipeline to run. Start both in separate terminals.")
    args = parser.parse_args()

    if args.pipeline == "borough":
        run_borough_stats()
    else:
        run_location_stats()
