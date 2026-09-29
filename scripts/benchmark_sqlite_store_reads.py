"""Compare caching costs for repeated checkpoints and sequential event scans.

Run with ``uv run python scripts/benchmark_sqlite_store_reads.py``. This isolates
Python row materialization; it does not measure SQL, model latency, or a live
application. Re-profile the application before attributing end-to-end gains.
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import time
from collections.abc import Callable
from contextlib import closing

from cayu._validation import copy_durable_json_object
from cayu.storage._validated_cache import validated_row_cache
from cayu.storage.sqlite import _checkpoint_from_json, _event_from_row


def _measure(operation: Callable[[], object], *, samples: int) -> dict[str, float | int]:
    operation()
    elapsed = []
    for _ in range(samples):
        started = time.perf_counter()
        operation()
        elapsed.append(1000 * (time.perf_counter() - started))
    return {"median_ms": statistics.median(elapsed), "samples": samples}


def _checkpoint_measurements() -> dict[str, object]:
    value = json.dumps(
        {
            "history": [
                {"message": "x" * 512, "index": index, "metadata": {"ok": True}}
                for index in range(2000)
            ]
        }
    )
    measurements = {}
    for name, decode in (
        (
            "validate_each_read",
            lambda text: copy_durable_json_object(json.loads(text), "checkpoint"),
        ),
        ("cached_validated_copy", _checkpoint_from_json),
    ):
        measurements[name] = _measure(lambda decode=decode: decode(value), samples=30)
    return {"source_bytes": len(value.encode()), "measurements": measurements}


def _event_scan_measurements() -> dict[str, object]:
    payload = json.dumps(
        {
            "items": [
                {"text": "x" * 64, "tags": ["a", "b", "c"], "data": {"ok": True, "n": index}}
                for index in range(100)
            ]
        }
    )
    # Construct immutable row snapshots before timing, so both cases measure
    # the same materialization work without SQL or connection-lease overhead.
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        rows = [
            connection.execute(
                """
                SELECT 'custom.benchmark.scan' AS event_type, 'session' AS session_id,
                       NULL AS interaction_id, ? AS event_id,
                       '2026-01-01T00:00:00+00:00' AS timestamp,
                       NULL AS agent_name, NULL AS environment_name,
                       NULL AS workflow_name, NULL AS tool_name, ? AS payload_json,
                       0 AS input_contract_runtime_owned,
                       0 AS file_attachment_attestations_runtime_owned
                """,
                (f"event-{index}", payload),
            ).fetchone()
            for index in range(100)
        ]
    # Recreate the former event cache to keep the sequential-scan regression
    # reproducible alongside the checkpoint cache's favorable warm-read case.
    measurements = {}
    for name, decode in (
        ("validate_each_event", _event_from_row),
        ("cached_event_scan", validated_row_cache(_event_from_row)),
    ):
        measurements[name] = _measure(
            lambda decode=decode: [decode(row) for row in rows], samples=5
        )
    return {
        "events": len(rows),
        "payload_bytes_each": len(payload.encode()),
        "measurements": measurements,
    }


def main() -> None:
    print(
        json.dumps(
            {"checkpoint": _checkpoint_measurements(), "event_scan": _event_scan_measurements()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
