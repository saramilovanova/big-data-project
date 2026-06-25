"""
Quick debugging helper: read up to --limit messages from one of the
pipeline's topics and pretty-print them.

    python read_topic.py raw --limit 5
    python read_topic.py borough --limit 5
    python read_topic.py clusters --limit 5
"""

import argparse
import json
import time

from confluent_kafka import Consumer

from config import (
    BOOTSTRAP, TOPIC_RAW, TOPIC_BOROUGH_STATS, TOPIC_LOCATION_STATS,
    TOPIC_PY_BOROUGH_STATS, TOPIC_PY_LOCATION_STATS, TOPIC_CLUSTERS,
)

TOPICS = {
    "raw": TOPIC_RAW,
    "borough": TOPIC_BOROUGH_STATS,
    "location": TOPIC_LOCATION_STATS,
    "py-borough": TOPIC_PY_BOROUGH_STATS,
    "py-location": TOPIC_PY_LOCATION_STATS,
    "clusters": TOPIC_CLUSTERS,
}


def main():
    parser = argparse.ArgumentParser(description="Read messages from an assignment Kafka topic.")
    parser.add_argument("topic", choices=sorted(TOPICS), help="Topic shortcut to read.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seconds", type=int, default=30)
    args = parser.parse_args()

    topic = TOPICS[args.topic]
    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id": f"{topic}-debug-reader-{int(time.time())}",
        "auto.offset.reset": "earliest",
    })

    consumer.subscribe([topic])
    print(f"Reading up to {args.limit} messages from '{topic}'...")

    rows = []
    deadline = time.time() + args.seconds

    try:
        while len(rows) < args.limit and time.time() < deadline:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                print(msg.error())
                continue
            rows.append(json.loads(msg.value().decode("utf-8")))
    finally:
        consumer.close()

    for i, row in enumerate(rows):
        print(f"\n[{i}]")
        print(json.dumps(row, indent=2))

    if not rows:
        print("No messages read. Start the processor before the producer, or reset topics and try again.")


if __name__ == "__main__":
    main()
