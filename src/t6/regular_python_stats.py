"""
regular_python_stats.py

Same rolling-window statistics as quix_streams.py, but implemented with
plain Python + confluent-kafka only (manual tumbling-window accumulators),
writing results to a separate pair of topics (TOPIC_PY_BOROUGH_STATS /
TOPIC_PY_LOCATION_STATS). Keeping this on its own topics lets it run
side-by-side with the Quix Streams pipeline without the two interfering,
so results can be compared directly (see the "why separate topics" note in
the project write-up).
"""

import json
import math
from datetime import datetime, timezone

from confluent_kafka import Consumer, Producer

from config import (
    BOOTSTRAP, TOPIC_RAW, TOPIC_PY_BOROUGH_STATS, TOPIC_PY_LOCATION_STATS,
    WINDOW_SECONDS, TOP10_LOCATIONS, STATS_FIELDS, OUTPUT_DIR,
)
from sinks import JsonlWriter


def parse_pickup_time(value):
    """Parse the pickup timestamp from a Kafka event."""
    if not value:
        return None
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def window_start_for(timestamp):
    """Return the start timestamp of the tumbling window for an event time."""
    epoch = int(timestamp.timestamp())
    return epoch - (epoch % WINDOW_SECONDS)


def new_accumulator():
    acc = {"count": 0}
    for field in STATS_FIELDS:
        acc[f"sum_{field}"] = 0.0
        acc[f"ssq_{field}"] = 0.0
        acc[f"min_{field}"] = float("inf")
        acc[f"max_{field}"] = float("-inf")
    return acc


def clean_number(value):
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def update_accumulator(acc, event):
    acc["count"] += 1
    for field in STATS_FIELDS:
        value = clean_number(event.get(field))
        acc[f"sum_{field}"] += value
        acc[f"ssq_{field}"] += value**2
        acc[f"min_{field}"] = min(acc[f"min_{field}"], value)
        acc[f"max_{field}"] = max(acc[f"max_{field}"], value)


def finalize(group_type, group_value, window_start, acc):
    count = acc["count"]
    window_start_dt = datetime.fromtimestamp(window_start, tz=timezone.utc)
    window_end_dt = datetime.fromtimestamp(window_start + WINDOW_SECONDS, tz=timezone.utc)

    result = {
        "processor": "regular_python",
        "group_type": group_type,
        "group_value": group_value,
        "count": count,
        "window_start": window_start_dt.isoformat(),
        "window_end": window_end_dt.isoformat(),
    }

    for field in STATS_FIELDS:
        mean = acc[f"sum_{field}"] / count
        variance = acc[f"ssq_{field}"] / count - mean**2
        result[f"{field}_mean"] = round(mean, 4)
        result[f"{field}_std"] = round(math.sqrt(max(variance, 0.0)), 4)
        result[f"{field}_min"] = round(acc[f"min_{field}"], 4)
        result[f"{field}_max"] = round(acc[f"max_{field}"], 4)

    return result


def emit_completed_windows(producer, windows, current_window_start, writers):
    completed_keys = [
        key for key in windows
        if key[2] + WINDOW_SECONDS <= current_window_start
    ]

    for group_type, group_value, window_start in completed_keys:
        acc = windows.pop((group_type, group_value, window_start))
        topic = TOPIC_PY_BOROUGH_STATS if group_type == "borough" else TOPIC_PY_LOCATION_STATS
        result = finalize(group_type, group_value, window_start, acc)
        producer.produce(topic, key=str(group_value), value=json.dumps(result))
        writers[group_type].write(result)

    if completed_keys:
        producer.poll(0)


def update_windows(windows, event, window_start):
    borough = event.get("PU_Borough")
    if borough:
        key = ("borough", borough, window_start)
        update_accumulator(windows.setdefault(key, new_accumulator()), event)

    try:
        location_id = int(event.get("PULocationID"))
    except (TypeError, ValueError):
        location_id = None

    if location_id in TOP10_LOCATIONS:
        key = ("location", location_id, window_start)
        update_accumulator(windows.setdefault(key, new_accumulator()), event)


def main():
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": "regular-python-stats",
        "auto.offset.reset": "earliest",
    })
    producer = Producer({"bootstrap.servers": BOOTSTRAP})

    windows = {}
    count = 0
    writers = {
        "borough": JsonlWriter(f"{OUTPUT_DIR}/borough_stats_python.jsonl"),
        "location": JsonlWriter(f"{OUTPUT_DIR}/location_stats_python.jsonl"),
    }

    consumer.subscribe([TOPIC_RAW])
    print("Running regular Python stream statistics...")

    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(msg.error())
                continue

            event = json.loads(msg.value().decode("utf-8"))
            pickup_time = parse_pickup_time(event.get("pickup_datetime"))
            if pickup_time is None:
                continue

            window_start = window_start_for(pickup_time)
            update_windows(windows, event, window_start)
            emit_completed_windows(producer, windows, window_start, writers)

            count += 1
            if count % 5000 == 0:
                print(f"Processed {count} events with regular Python stats")

    except KeyboardInterrupt:
        print("Stopping regular Python stats...")

    finally:
        for group_type, group_value, window_start in list(windows):
            acc = windows.pop((group_type, group_value, window_start))
            topic = TOPIC_PY_BOROUGH_STATS if group_type == "borough" else TOPIC_PY_LOCATION_STATS
            result = finalize(group_type, group_value, window_start, acc)
            producer.produce(topic, key=str(group_value), value=json.dumps(result))
            writers[group_type].write(result)
        for writer in writers.values():
            writer.close()
        consumer.close()
        producer.flush()


if __name__ == "__main__":
    main()