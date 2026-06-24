"""
producer.py

Streams the combined Yellow Taxi + FHVHV 2021 events (built by
prepare_stream_source.py) into Kafka, one message per trip, in pickup-time
order.

Two things matter beyond the basic "read parquet, send to Kafka" prototype:

1. Memory: the combined file can be tens of millions of rows, far more than
   fits comfortably in pandas on a laptop. We stream it batch-by-batch with
   pyarrow's iter_batches() instead of loading the whole file at once.

2. Event time vs Kafka's own message timestamp: it's tempting to set each
   Kafka message's timestamp to the trip's pickup_datetime so downstream
   windowing "just works" off the Kafka record metadata. Don't -- Kafka's
   time-based log retention also keys off that same metadata (it deletes a
   segment once `now - largest_record_timestamp_in_segment > retention.ms`).
   Since this data is from 2021, pickup_datetime is years in the past
   relative to wall-clock "now," so a message timestamped with its own
   pickup_datetime looks years "stale" to the broker the instant it's
   written, and retention deletes it almost immediately -- regardless of how
   long it's actually been sitting in the topic. Kafka message timestamps
   are left to default to real send time here (correct for retention/ops
   purposes); event time for windowing is instead read from the
   pickup_datetime field already present in the JSON payload (see
   quix_streams.py's timestamp_extractor). --speed lets you optionally
   throttle production to simulate a chosen playback rate for a live demo.
"""

import argparse
import json
import time

import pyarrow.parquet as pq
from confluent_kafka import Producer

from config import BOOTSTRAP, COMBINED_PARQUET, TOPIC_RAW


def row_to_json(row: dict) -> str:
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        elif v is None:
            out[k] = None
        else:
            try:
                if v != v:  # NaN check without importing pandas/numpy
                    out[k] = None
                    continue
            except TypeError:
                pass
            out[k] = v
    return json.dumps(out)


def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed: {err}")


def main():
    parser = argparse.ArgumentParser(description="Stream combined taxi events into Kafka.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only send the first N rows (useful for a quick test run).")
    parser.add_argument("--speed", type=float, default=0.0,
                         help="Playback speed-up factor for a live demo, e.g. 3600 means "
                              "1 simulated hour passes per real second. 0 (default) = "
                              "send as fast as possible, no artificial pacing.")
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    # producer = Producer({"bootstrap.servers": BOOTSTRAP, "linger.ms": 20, "batch.num.messages": 10000})
    producer = Producer({
        "bootstrap.servers": BOOTSTRAP,
        "linger.ms": 20,
        "batch.num.messages": 10000,
        "queue.buffering.max.messages": 500000,
    })

    pf = pq.ParquetFile(COMBINED_PARQUET)
    print(f"Streaming from {COMBINED_PARQUET} -> topic '{TOPIC_RAW}'")
    print(f"Row groups: {pf.num_row_groups}, total rows: {pf.metadata.num_rows:,}")

    sent = 0
    last_pickup_ts = None
    t_last_real = time.monotonic()

    for batch in pf.iter_batches(batch_size=args.batch_size):
        df = batch.to_pandas()

        for row in df.to_dict(orient="records"):
            if args.limit is not None and sent >= args.limit:
                break

            pickup_dt = row["pickup_datetime"]
            pickup_ms = int(pickup_dt.timestamp() * 1000)

            if args.speed > 0 and last_pickup_ts is not None:
                sim_elapsed = (pickup_ms - last_pickup_ts) / 1000.0  # seconds of event time
                real_elapsed = max(sim_elapsed / args.speed, 0.0)
                now = time.monotonic()
                sleep_for = real_elapsed - (now - t_last_real)
                if sleep_for > 0:
                    time.sleep(min(sleep_for, 2.0))  # cap to avoid huge gaps (e.g. overnight)
                t_last_real = time.monotonic()

            last_pickup_ts = pickup_ms

            while True:
                try:
                    producer.produce(
                        TOPIC_RAW,
                        key=str(int(row["PULocationID"])) if row["PULocationID"] is not None else None,
                        value=row_to_json(row),
                        callback=delivery_report,
                    )
                    break

                except BufferError:
                    producer.poll(0.1)

            producer.poll(0)

            sent += 1
            if sent % 50000 == 0:
                producer.flush()
                print(f"Sent {sent:,} events (last pickup_datetime={pickup_dt})")

        if args.limit is not None and sent >= args.limit:
            break

    producer.flush()
    print(f"Done. Streaming complete, {sent:,} events sent.")


if __name__ == "__main__":
    main()
