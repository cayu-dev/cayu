"""Compare memory adapter compositions with the same current shared dependencies.

This is a bounded CPU probe, not an end-to-end agent or historical release benchmark.
The baseline adapter source is loaded from Git; no checkout or production state is changed.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASELINE = "7fbcd127e8ffecef6fefeb6add32826962ddf040"
sys.path.insert(0, str(ROOT / "src"))

from cayu.events import Event  # noqa: E402
from cayu.messages import Message  # noqa: E402
from cayu.sessions.event_queries import EventQuery  # noqa: E402
from cayu.sessions.memory import InMemorySessionStore  # noqa: E402
from cayu.sessions.records import SessionIdentity  # noqa: E402
from cayu.sessions.requests import RunRequest  # noqa: E402
from cayu.sessions.transcript_queries import TranscriptQuery  # noqa: E402

NOW = datetime(2026, 1, 1, tzinfo=UTC)


async def setup(cls):
    store = cls(ownership_clock=lambda: NOW)
    await store.create(
        RunRequest(agent_name="assistant", session_id="probe", messages=[]),
        identity=SessionIdentity(provider_name="scripted", model="scripted"),
    )
    await store.append_transcript_messages(
        "probe",
        [Message.text("user", f"message {i}") for i in range(100)],
        interaction_id=None,
    )
    await store.append_events(
        "probe",
        [
            Event(
                id=f"event-{i}",
                session_id="probe",
                type="custom.assessment.event",
                timestamp=NOW,
                payload={"i": i},
            )
            for i in range(100)
        ],
    )
    await store.checkpoint("probe", {"counter": 0})
    return store


async def workload(store, name):
    results = []
    if name == "bounded_reads":
        for _ in range(50):
            events = await store.query_events(
                EventQuery(session_id="probe", event_type="custom.assessment.event", limit=20)
            )
            transcript = await store.query_transcript(TranscriptQuery(session_id="probe", limit=20))
            results.append((events, transcript))
    else:
        for i in range(50):
            await store.append_transcript_messages_and_transform_checkpoint(
                "probe",
                [Message.text("assistant", f"result {i}")],
                lambda session, checkpoint: {"counter": checkpoint["counter"] + 1},
                interaction_id=None,
            )
            await store.publish_checkpoint_and_events(
                "probe",
                checkpoint_transform=lambda session, checkpoint: checkpoint,
                events=[
                    Event(
                        id=f"write-{i}",
                        session_id="probe",
                        type="custom.assessment.event",
                        timestamp=NOW,
                        payload={"i": i},
                    )
                ],
            )
    return results


async def evidence(store, results):
    # Omit randomly generated session identity and initial event timestamps; compare
    # every workload event, transcript row, attribution and checkpoint value.
    events = await store.query_events(EventQuery(session_id="probe", limit=5000))
    snapshot = await store.load_transcript_snapshot("probe")
    material = {
        "results": [
            [[record.model_dump(mode="json") for record in records], page.model_dump(mode="json")]
            for records, page in results
        ],
        "events": [
            r.event.model_dump(mode="json")
            for r in events
            if r.event.type == "custom.assessment.event"
        ],
        "transcript": snapshot.model_dump(mode="json"),
        "checkpoint": await store.load_checkpoint("probe"),
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


async def compare(parent, pairs):
    report = {}
    for name in ("bounded_reads", "transcript_checkpoint_event_writes"):
        samples = []
        hashes = set()
        for pair in range(pairs + 5):
            stores = {"base": await setup(parent), "head": await setup(InMemorySessionStore)}
            timings, outputs = {}, {}
            for variant in ("base", "head") if pair % 2 == 0 else ("head", "base"):
                started = time.process_time_ns()
                results = await workload(stores[variant], name)
                timings[variant] = (time.process_time_ns() - started) / 1000
                outputs[variant] = await evidence(stores[variant], results)
            assert outputs["base"] == outputs["head"], (name, outputs)
            hashes.add(outputs["head"])
            if pair >= 5:
                samples.append(timings)
        assert len(hashes) == 1, (name, hashes)
        before = statistics.median(row["base"] for row in samples)
        after = statistics.median(row["head"] for row in samples)
        report[name] = {
            "baseline_cpu_us": before,
            "current_cpu_us": after,
            "percent_medians": 100 * (after / before - 1),
            "median_paired_delta_us": statistics.median(
                row["head"] - row["base"] for row in samples
            ),
            "matching_material_sha256": next(iter(hashes)),
            "samples_cpu_us": samples,
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=30)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", "src/cayu"], cwd=ROOT, check=True)
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:src/cayu/sessions/memory.py"], cwd=ROOT, text=True
    )
    with tempfile.TemporaryDirectory(prefix="cayu-memory-assessment-") as directory:
        path = Path(directory) / "memory.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location("cayu.sessions._assessment_baseline", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        report = {
            "baseline_adapter": BASELINE,
            "current_revision": subprocess.check_output(
                ["git", "log", "-1", "--format=%H", "HEAD", "--", "src/cayu"], cwd=ROOT, text=True
            ).strip(),
            "current_production_tree": subprocess.check_output(
                ["git", "rev-parse", "HEAD:src/cayu"], cwd=ROOT, text=True
            ).strip(),
            "python": sys.version.split()[0],
            "pairs": args.pairs,
            "warmup_pairs": 5,
            "scope": "Historical memory adapter and current adapter with identical current shared dependencies; setup, imports and evidence hashing excluded from CPU timings.",
            "workloads": asyncio.run(compare(module.InMemorySessionStore, args.pairs)),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
