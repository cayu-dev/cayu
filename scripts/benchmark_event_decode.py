"""Time decoding stored events, trusted and validating, on a production type mix.

Run with ``uv run python scripts/benchmark_event_decode.py``. It builds 10,000
SQLite ``cayu_events`` rows in memory, with event types in the proportions and
payload sizes of a busy review deployment, then times:

- ``event_from_row``, the trusted decode reads use (target: under 0.5 s);
- ``validating_event_from_row``, the full-validation decode every read used
  before, and which ``cayu storage verify-events`` still runs;
- ``copy_event`` of the trusted events, which reuses the stored text.

Pass ``--events N`` for another count and ``--json`` for machine-readable output.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from cayu.events import copy_event
from cayu.storage import _sqlite_records as sqlite_records

#: (event type, share of all events, average payload JSON bytes), from the
#: review.cayu.ai store's 997,415 events.
TYPE_MIX: tuple[tuple[str, int, int], ...] = (
    ("tool.call.completed", 123_408, 14_226),
    ("tool.call.started", 118_410, 578),
    ("model.thinking.delta", 97_924, 299),
    ("runner.exec.completed", 94_923, 1_832),
    ("runner.exec.started", 94_923, 559),
    ("request.footprint.recorded", 84_331, 3_778),
    ("model.started", 84_330, 339),
    ("tool.exposure.recorded", 82_967, 544),
    ("model.completed", 82_875, 7_694),
    ("environment.lifecycle.transition", 55_129, 498),
    ("interaction.started", 7_882, 713),
    ("session.started", 7_873, 828),
    ("turn.completed", 7_860, 479),
    ("interaction.completed", 7_655, 790),
    ("session.completed", 7_655, 2),
)
TARGET_SECONDS = 0.5


def _payload(event_type: str, size: int, rng: random.Random) -> dict[str, Any]:
    """A payload shaped like the type's: a few identifiers, nested data, bulk text."""

    payload: dict[str, Any] = {
        "tool_call_id": f"call_{rng.getrandbits(64):016x}",
        "model_attempt_id": f"attempt-{rng.getrandbits(48):012x}",
        "status": "completed",
        "attempt": rng.randint(1, 3),
        "duration_ms": rng.randint(5, 90_000),
    }
    if event_type == "model.completed":
        payload["usage"] = {"input_tokens": rng.randint(1_000, 90_000), "output_tokens": 800}
        payload["usage_metrics"] = {
            "provider_name": "anthropic",
            "model": "claude-sonnet-5-5",
            "input_tokens": payload["usage"]["input_tokens"],
            "output_tokens": 800,
            "cache": {"read_tokens": 4_000, "write_tokens": 0},
        }
    if event_type in {"request.footprint.recorded", "tool.exposure.recorded"}:
        payload["items"] = [
            {"name": f"tool_{index}", "bytes": rng.randint(10, 9_000), "kept": index % 2 == 0}
            for index in range(12)
        ]
    padding = max(0, size - len(sqlite_records.json_dumps(payload)) - 12)
    line = "+    changed line of the reviewed diff\n"
    payload["output"] = (line * (padding // len(line) + 1))[:padding]
    return payload


def build_rows(count: int, *, seed: int = 3939) -> list[sqlite3.Row]:
    """``count`` stored event rows with the production type mix."""

    rng = random.Random(seed)
    types = [event_type for event_type, _, _ in TYPE_MIX]
    weights = [share for _, share, _ in TYPE_MIX]
    sizes = {event_type: size for event_type, _, size in TYPE_MIX}
    started = datetime(2026, 11, 1, tzinfo=UTC)
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    columns = ("sequence", *sqlite_records.EVENT_COLUMN_NAMES)
    rows: list[sqlite3.Row] = []
    for sequence in range(1, count + 1):
        event_type = rng.choices(types, weights)[0]
        values = {
            "sequence": sequence,
            "session_id": f"review-{sequence // 90:05d}",
            "event_id": f"{rng.getrandbits(128):032x}",
            "interaction_id": f"interaction-{sequence // 30:06d}",
            "event_type": event_type,
            "timestamp": sqlite_records.format_datetime(started + timedelta(seconds=sequence)),
            "agent_name": "reviewer",
            "environment_name": "docker-coding",
            "workflow_name": None,
            "tool_name": "bash" if event_type.startswith("tool.") else None,
            "payload_json": sqlite_records.json_dumps(_payload(event_type, sizes[event_type], rng)),
            "input_contract_runtime_owned": 0,
            "file_attachment_attestations_runtime_owned": 0,
        }
        rows.append(
            connection.execute(
                "SELECT " + ", ".join(f"? AS {name}" for name in columns),
                [values[name] for name in columns],
            ).fetchone()
        )
    connection.close()
    return rows


def measure(count: int) -> dict[str, Any]:
    rows = build_rows(count)
    started = time.perf_counter()
    events = [sqlite_records.event_from_row(row) for row in rows]
    trusted = time.perf_counter() - started
    started = time.perf_counter()
    for row in rows:
        sqlite_records.validating_event_from_row(row)
    validating = time.perf_counter() - started
    started = time.perf_counter()
    for event in events:
        copy_event(event)
    copied = time.perf_counter() - started
    return {
        "events": count,
        "average_payload_bytes": round(sum(len(row["payload_json"]) for row in rows) / count),
        "trusted_decode_seconds": round(trusted, 3),
        "validating_decode_seconds": round(validating, 3),
        "copy_trusted_seconds": round(copied, 3),
        "target_seconds": TARGET_SECONDS * count / 10_000,
        "meets_target": trusted <= TARGET_SECONDS * count / 10_000,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Time decoding stored events.")
    parser.add_argument("--events", type=int, default=10_000)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = measure(args.events)
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}: {value}")
    return 0 if result["meets_target"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
