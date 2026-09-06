"""Synthetic incremental evidence benchmark (no providers).

Run from the repository root:
    uv run python benchmarks/incremental_evidence.py --output /tmp/evidence-benchmark.json
Fixture construction and each measurement run in separate processes so fixture
allocation is not attributed to capture RSS. All payloads are synthetic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import sqlite3
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from cayu import (
    Event,
    EventType,
    Message,
    RunRequest,
    SessionIdentity,
    SessionStatus,
    SessionStore,
    SQLiteSessionStore,
)
from cayu.runtime.evidence_spool import (
    EvidenceSpool,
    IncrementalEvidenceAdmission,
    IncrementalEvidenceError,
    IncrementalEvidenceLimits,
)


async def prepare_session(
    store: SessionStore, session_id: str = "benchmark", *, parent_session_id: str | None = None
) -> None:
    message = Message.text("user", "Synthetic evidence benchmark")
    await store.create(
        RunRequest(
            agent_name="synthetic",
            session_id=session_id,
            parent_session_id=parent_session_id,
            messages=[message],
        ),
        identity=SessionIdentity(provider_name="synthetic", model="synthetic"),
        interaction_started_event=Event(
            type=EventType.INTERACTION_STARTED,
            session_id=session_id,
            interaction_id="interaction",
        ),
        interaction_source_messages=[message],
    )
    await store.replace_initial_transcript_messages(
        session_id,
        [message],
        [message],
        interaction_id="interaction",
    )
    await store.append_transcript_messages(
        session_id,
        [Message.text("assistant", "done")],
        interaction_id="interaction",
    )
    await store.publish_interaction_transition(
        session_id,
        event=Event(
            type=EventType.INTERACTION_COMPLETED,
            session_id=session_id,
            interaction_id="interaction",
        ),
        from_statuses={SessionStatus.RUNNING},
        to_status=SessionStatus.COMPLETED,
    )


async def insert_sqlite_records(
    store: SQLiteSessionStore,
    count: int,
    payload_bytes: int,
    session_id: str = "benchmark",
) -> None:
    payload = json.dumps({"text": "x" * payload_bytes})
    stamp = datetime.now(UTC).isoformat()

    def write(connection: sqlite3.Connection) -> None:
        with connection:
            connection.executemany(
                "INSERT INTO cayu_events(session_id, event_id, event_type, timestamp, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    (session_id, f"synthetic-{i}", "custom.synthetic.evidence", stamp, payload)
                    for i in range(count)
                ),
            )

    # Repository-only synthetic seeding preserves the store's registered SQL
    # authority functions and triggers. It is outside the measured process.
    await store._run_write(write)


async def complete_session(store: SessionStore, session_id: str = "benchmark") -> None:
    await store.append_event(
        session_id,
        Event(type=EventType.SESSION_COMPLETED, session_id=session_id),
    )


async def prepare(path: Path, count: int, payload_bytes: int) -> None:
    store = SQLiteSessionStore(path)
    try:
        await prepare_session(store)
        await insert_sqlite_records(store, count, payload_bytes)
        await complete_session(store)
    finally:
        await store.close()


def rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


async def measure(path: Path, count: int, arrivals: int) -> dict:
    limits = IncrementalEvidenceLimits(max_events=max(count + 10, 10_000), max_seconds=600)
    admission = IncrementalEvidenceAdmission(max_captures=4)
    baseline = rss_bytes()
    started = time.monotonic()
    outcomes: list[dict] = []

    async def capture() -> None:
        try:
            reservation = admission.reserve(limits)
        except IncrementalEvidenceError as exc:
            outcomes.append({"status": exc.code})
            return
        # Admission precedes store/connection and spill allocation.
        store = SQLiteSessionStore(path)
        try:
            with EvidenceSpool(limits) as spool:
                await store.export_terminal_session_evidence("benchmark", spool=spool)
                total = 0
                # Slow consumers do not retain a source transaction or producer queue.
                for _record in spool.events:
                    total += 1
                    if total % 4096 == 0:
                        await asyncio.sleep(0)
                outcomes.append(
                    {
                        "status": "complete",
                        "events": total,
                        "evidence_bytes": spool.boundary.total_bytes,
                        "peak_record_bytes": spool.peak_record_bytes,
                        "peak_page_transport_bytes": spool.peak_page_transport_bytes,
                        "peak_page_records": spool.peak_page_records,
                        "spill_records_read": spool.spill_records_read,
                        "spill_bytes": spool.path.stat().st_size,
                        "digest": spool.evidence_sha256,
                    }
                )
            assert not spool.path.exists()
        finally:
            await store.close()
            reservation.close()

    await asyncio.gather(*(capture() for _ in range(arrivals)))
    assert (
        admission.active_captures
        == admission.reserved_buffer_bytes
        == admission.reserved_spill_bytes
        == 0
    )
    return {
        "synthetic_records": count,
        "arrivals": arrivals,
        "admitted": sum(row["status"] == "complete" for row in outcomes),
        "rejected": sum(row["status"] != "complete" for row in outcomes),
        "seconds": round(time.monotonic() - started, 3),
        "baseline_peak_rss_bytes": baseline,
        "peak_rss_bytes": rss_bytes(),
        "peak_rss_growth_bytes": max(0, rss_bytes() - baseline),
        "per_capture_buffer_reservation_bytes": admission.buffer_envelope(limits),
        "aggregate_buffer_budget_bytes": admission.max_buffer_bytes,
        "aggregate_spill_budget_bytes": admission.max_spill_bytes,
        "limits": limits.model_dump(),
        "captures": outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("prepare", "measure"))
    parser.add_argument("--path", type=Path)
    parser.add_argument("--count", type=int, default=250_000)
    parser.add_argument("--payload-bytes", type=int, default=128)
    parser.add_argument("--arrivals", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "prepare":
        asyncio.run(prepare(args.path, args.count, args.payload_bytes))
        return
    if args.mode == "measure":
        print(json.dumps(asyncio.run(measure(args.path, args.count, args.arrivals))))
        return
    results = []
    for count, payload, arrivals in (
        (10_000, 128, 1),
        (100_000, 128, 1),
        (250_000, 128, 1),
        (64, 800_000, 1),
        (256, 800_000, 1),
        (1000, 128, 100),
    ):
        with tempfile.TemporaryDirectory(prefix="cayu-evidence-benchmark-") as directory:
            path = Path(directory) / "source.sqlite3"
            base = [sys.executable, __file__, "--path", str(path), "--count", str(count)]
            subprocess.run(
                [*base, "--mode", "prepare", "--payload-bytes", str(payload)], check=True
            )
            result = subprocess.run(
                [*base, "--mode", "measure", "--arrivals", str(arrivals)],
                check=True,
                capture_output=True,
                text=True,
            )
            row = json.loads(result.stdout)
            row["payload_bytes"] = payload
            results.append(row)
            print(
                json.dumps(
                    {key: value for key, value in row.items() if key not in {"captures", "limits"}}
                ),
                flush=True,
            )
    document = {"python": sys.version, "platform": sys.platform, "results": results}
    if args.output:
        args.output.write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
