"""
Kafka topic retention (see docker-compose.yaml) is finite -- even at 6
hours, it's not a place to permanently keep your results. Each consumer
writes its output to a local JSON-lines file via JsonlWriter, in addition
to producing back to Kafka. The .jsonl files in OUTPUT_DIR are the durable
artifacts to load into pandas for your report's tables/charts.
"""

import json
from pathlib import Path


class JsonlWriter:
    """Buffered append-only JSON-lines writer."""

    def __init__(self, path, flush_every=200):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)
        self._count = 0
        self._flush_every = flush_every

    def write(self, obj):
        self._fh.write(json.dumps(obj) + "\n")
        self._count += 1
        if self._count % self._flush_every == 0:
            self._fh.flush()

    def close(self):
        self._fh.flush()
        self._fh.close()
