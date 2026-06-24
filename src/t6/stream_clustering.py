"""
stream_clustering.py  (fixed for disk-space efficiency)

Consumes the combined taxi event stream, applies online (incremental)
K-Means clustering trip-by-trip, and:
  - writes EVERY cluster assignment to a local JSONL file (for the report)
  - sends only periodic CLUSTER-CENTRE SNAPSHOTS to the taxi-clusters Kafka
    topic (not one message per event -- that mirrored the full raw stream
    and consumed 5+ GB of disk for a 16.5M-event run)

Why the change?
  The previous version did `producer.produce(TOPIC_CLUSTERS, ..., value=payload)`
  for each of the ~16.5M events, creating a taxi-clusters topic that was just
  as large as the raw topic.  The T6 requirement is to *apply* stream clustering
  to the data -- it doesn't require storing every individual cluster assignment
  back in Kafka.  Periodic centre snapshots (tiny) fulfil the "write processed
  data back to Kafka" requirement while using negligible disk space.
"""

import json
import math
import time

from confluent_kafka import Consumer, Producer

from config import (
    BOOTSTRAP, TOPIC_RAW, TOPIC_CLUSTERS,
    CLUSTER_FEATURES, FEATURE_SCALES, N_CLUSTERS, OUTPUT_DIR,
)
from sinks import JsonlWriter

# ── How often to push a centre snapshot to Kafka (every N events) ────────────
# At SAMPLE_PCT=1 (~3.3M events) this produces ~3300 Kafka messages -- tiny.
SNAPSHOT_EVERY = 1_000


class OnlineKMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.centers    = []
        self.counts     = []

    def update(self, x):
        """Assign one feature vector to the nearest cluster and update its center."""
        if len(self.centers) < self.n_clusters:
            cluster_id = len(self.centers)
            self.centers.append(list(x))
            self.counts.append(1)
            return cluster_id, 0.0

        distances  = [self._distance(x, c) for c in self.centers]
        cluster_id = min(range(self.n_clusters), key=distances.__getitem__)
        self.counts[cluster_id] += 1

        lr     = 1.0 / self.counts[cluster_id]
        center = self.centers[cluster_id]
        for i, v in enumerate(x):
            center[i] += lr * (v - center[i])

        return cluster_id, distances[cluster_id]

    def centre_summary(self, count):
        """Return a compact snapshot of all cluster centres for Kafka."""
        return {
            "event_count": count,
            "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "centres": [
                {
                    "cluster_id":    i,
                    "count":         self.counts[i] if i < len(self.counts) else 0,
                    **{
                        f: round(self.centers[i][fi] * FEATURE_SCALES.get(f, 1.0), 4)
                        for fi, f in enumerate(CLUSTER_FEATURES)
                        if i < len(self.centers)
                    },
                }
                for i in range(self.n_clusters)
            ],
        }

    @staticmethod
    def _distance(x, y):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x, y)))


def clean_number(value):
    """Convert invalid or missing numeric values to 0.0."""
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return value


def feature_vector(event):
    """Create a scaled numeric feature vector from one taxi trip event."""
    return [
        clean_number(event.get(f)) / FEATURE_SCALES.get(f, 1.0)
        for f in CLUSTER_FEATURES
    ]


def cluster_payload(event, cluster_id, distance):
    """Create the output record for the local JSONL file."""
    return {
        "cluster_id":       cluster_id,
        "cluster_distance": round(distance, 6),
        "pickup_datetime":  event.get("pickup_datetime"),
        "source":           event.get("source"),
        "PULocationID":     event.get("PULocationID"),
        "PU_Borough":       event.get("PU_Borough"),
        "PU_Zone":          event.get("PU_Zone"),
        # Omit full feature dict from JSONL to keep file small; add if needed.
        "trip_distance":    clean_number(event.get("trip_distance")),
        "fare_amount":      clean_number(event.get("fare_amount")),
    }


def delivery_report(err, msg):
    if err:
        print(f"[clustering] Kafka delivery error: {err}")


def main():
    consumer_group = f"stream-clustering-{int(time.time())}"

    consumer = Consumer({
        "bootstrap.servers": BOOTSTRAP,
        "group.id":          consumer_group,
        "auto.offset.reset": "earliest",
    })
    producer = Producer({
        "bootstrap.servers":            BOOTSTRAP,
        "queue.buffering.max.messages": 10_000,
        "linger.ms":                    200,
    })
    model   = OnlineKMeans(N_CLUSTERS)
    writer  = JsonlWriter(f"{OUTPUT_DIR}/clusters.jsonl")
    centres_writer = JsonlWriter(f"{OUTPUT_DIR}/cluster_centres.jsonl")

    consumer.subscribe([TOPIC_RAW])

    print(f"[clustering] Running online k-means (k={N_CLUSTERS}) → '{TOPIC_CLUSTERS}'")
    print(f"[clustering] Consumer group: {consumer_group}")
    print(f"[clustering] Centre snapshots sent to Kafka every {SNAPSHOT_EVERY:,} events")
    print(f"[clustering] Per-event assignments written to: {OUTPUT_DIR}/clusters.jsonl")

    count           = 0
    last_wait_msg   = 0

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                now = time.time()
                if now - last_wait_msg >= 10:
                    print("[clustering] Waiting for taxi events from Kafka...")
                    last_wait_msg = now
                continue

            if msg.error():
                print(f"[clustering] Kafka error: {msg.error()}")
                continue

            event      = json.loads(msg.value().decode("utf-8"))
            vector     = feature_vector(event)
            cluster_id, distance = model.update(vector)
            payload    = cluster_payload(event, cluster_id, distance)

            # ── JSONL: write every assignment (for analysis/report) ───────────
            writer.write(payload)

            count += 1

            # ── Kafka: send a CENTRE SNAPSHOT every SNAPSHOT_EVERY events ─────
            # (NOT one message per event -- that's what killed the disk)
            if count % SNAPSHOT_EVERY == 0:
                snapshot = model.centre_summary(count)
                producer.produce(
                    TOPIC_CLUSTERS,
                    key   = str(count),
                    value = json.dumps(snapshot),
                    callback = delivery_report,
                )
                centres_writer.write(snapshot)
                producer.poll(0)

            if count <= 5:
                print(
                    f"[clustering] Event {count}: cluster={cluster_id}  "
                    f"source={event.get('source')}  "
                    f"zone={event.get('PU_Zone')}  "
                    f"dist={clean_number(event.get('trip_distance')):.2f}  "
                    f"fare={clean_number(event.get('fare_amount')):.2f}"
                )
            if count % 50_000 == 0:
                print(f"[clustering] Clustered {count:,} events")

    except KeyboardInterrupt:
        print("[clustering] Stopping stream clustering...")

    finally:
        # Final centre snapshot
        if model.centers:
            final = model.centre_summary(count)
            producer.produce(TOPIC_CLUSTERS, key="final", value=json.dumps(final))
            centres_writer.write(final)
            print("\n[clustering] Final cluster centres:")
            for c in final["centres"]:
                print(
                    f"  Cluster {c['cluster_id']}: "
                    f"dist={c.get('trip_distance',0):.2f}mi  "
                    f"fare=${c.get('fare_amount',0):.2f}  "
                    f"tip=${c.get('tip_amount',0):.2f}  "
                    f"n={c['count']:,}"
                )

        writer.close()
        centres_writer.close()
        consumer.close()
        producer.flush()
        print(f"\n[clustering] Done. {count:,} events clustered.")
        print(f"  Assignments  → {OUTPUT_DIR}/clusters.jsonl")
        print(f"  Centre log   → {OUTPUT_DIR}/cluster_centres.jsonl")
        print(f"  Kafka topic  → {TOPIC_CLUSTERS}  ({count // SNAPSHOT_EVERY} snapshots)")


if __name__ == "__main__":
    main()
