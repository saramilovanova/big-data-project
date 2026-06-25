"""
Deletes and recreates all pipeline topics, so each run starts clean
(no leftover messages or consumer-group offsets from a previous attempt).
Run this before starting the producer for a fresh end-to-end run.
"""

import time

from confluent_kafka.admin import AdminClient, NewTopic

from config import (
    BOOTSTRAP, TOPIC_RAW, TOPIC_BOROUGH_STATS, TOPIC_LOCATION_STATS,
    TOPIC_PY_BOROUGH_STATS, TOPIC_PY_LOCATION_STATS, TOPIC_CLUSTERS,
)

ALL_TOPICS = [
    TOPIC_RAW, TOPIC_BOROUGH_STATS, TOPIC_LOCATION_STATS,
    TOPIC_PY_BOROUGH_STATS, TOPIC_PY_LOCATION_STATS, TOPIC_CLUSTERS,
]

admin = AdminClient({"bootstrap.servers": BOOTSTRAP})

print("Deleting topics...", ALL_TOPICS)
admin.delete_topics(ALL_TOPICS)
time.sleep(3)

print("Recreating topics...")
new_topics = [NewTopic(topic=t, num_partitions=1, replication_factor=1) for t in ALL_TOPICS]
admin.create_topics(new_topics)

print("Done.")
