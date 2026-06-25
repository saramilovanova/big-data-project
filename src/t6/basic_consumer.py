"""
Reads raw combined taxi trip events and prints them, for debugging/
demonstrating the raw event flow before the rolling-stats and clustering
pipelines.
"""

import json
from confluent_kafka import Consumer
from config import BOOTSTRAP, TOPIC_RAW

consumer = Consumer({
    "bootstrap.servers": BOOTSTRAP,
    "group.id": "basic-group",
    "auto.offset.reset": "earliest",
})

consumer.subscribe([TOPIC_RAW])

print("Listening for taxi events...")

count = 0

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(msg.error())
            continue

        event = json.loads(msg.value().decode("utf-8"))
        print(
            f"[{count}] {event.get('source')} | {event.get('PU_Borough')} | "
            f"fare={event.get('fare_amount'):.2f} | dist={event.get('trip_distance'):.2f}"
        )
        count += 1

except KeyboardInterrupt:
    print("Stopping consumer...")

finally:
    consumer.close()
