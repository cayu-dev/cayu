#!/usr/bin/env python3
"""Measure hermetic agent work-context and recall-checkpoint store overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

from cayu import (
    AgentRecallCheckpoint,
    AgentRecallCheckpointMode,
    AgentWorkContext,
    AgentWorkContextStore,
    InMemoryAgentWorkContextStore,
    SQLiteAgentWorkContextStore,
)

_SCHEMA_VERSION = "cayu.agent_work_context_performance.v1"
_DEFAULT_SAMPLES = 50
_NOW = datetime(2026, 8, 28, 9, 0, tzinfo=UTC)
_ACCESS_POLICY_SHA256 = "a" * 64

_CEILINGS = {
    "memory_zero_record_construction_p95_ms": 1.0,
    "sqlite_zero_record_construction_p50_ms": 50.0,
    "sqlite_zero_record_construction_p95_ms": 250.0,
    "memory_current_read_p95_ms": 1.0,
    "sqlite_current_read_p95_ms": 10.0,
    "memory_revision_append_p95_ms": 5.0,
    "sqlite_revision_append_p50_ms": 25.0,
    "sqlite_revision_append_p95_ms": 250.0,
    "memory_checkpoint_advance_p95_ms": 5.0,
    "sqlite_checkpoint_advance_p50_ms": 25.0,
    "sqlite_checkpoint_advance_p95_ms": 250.0,
    "sqlite_storage_bytes_per_durable_record": 32_768,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _storage_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


def _context(revision: int) -> AgentWorkContext:
    return AgentWorkContext.create(
        task_id="agent-work-context-performance",
        goal=f"Process durable shared-memory frontier {revision:05d}",
        revision=revision,
        operation_id=f"context-performance-{revision:05d}",
        published_by="performance-runner",
        published_at=_NOW + timedelta(microseconds=revision),
        scope_ids=("repository:cayu",),
        workflow_id="workflow:memory-v5.1",
        workflow_phase="agent-work-context",
        workflow_iteration=revision,
        entity_ids=(f"frontier:{revision:05d}",),
        repository_paths=("src/cayu",),
        code_symbols=("AgentWorkContextStore",),
    )


def _checkpoint(
    context: AgentWorkContext,
    revision: int,
) -> AgentRecallCheckpoint:
    return AgentRecallCheckpoint(
        agent_id="agent:performance",
        task_id=context.task_id,
        knowledge_namespace="project:cayu",
        access_policy_sha256=_ACCESS_POLICY_SHA256,
        revision=revision,
        work_context_revision=context.revision,
        work_context_sha256=context.content_sha256,
        knowledge_sequence=revision,
        index_readiness_sequence=revision,
        knowledge_high_water_sequence=revision,
        index_readiness_high_water_sequence=revision,
        processing_mode=(
            AgentRecallCheckpointMode.FULL_INDEX
            if revision == 1
            else AgentRecallCheckpointMode.DELTA
        ),
        processing_id=f"checkpoint-processing-{revision:05d}",
        operation_id=f"checkpoint-performance-{revision:05d}",
        updated_by="performance-runner",
        updated_at=_NOW + timedelta(seconds=1, microseconds=revision),
    )


async def _close(store: AgentWorkContextStore) -> None:
    await store.close()


async def _construct_store(backend: str, path: Path) -> AgentWorkContextStore:
    if backend == "memory":
        return InMemoryAgentWorkContextStore(clock=lambda: _NOW)
    return SQLiteAgentWorkContextStore(path, clock=lambda: _NOW)


async def _backend_result(
    backend: str,
    *,
    samples: int,
    directory: Path,
) -> dict[str, Any]:
    path = directory / f"{backend}-agent-work-context.sqlite"
    if backend == "sqlite":
        initializer = await _construct_store(backend, path)
        await _close(initializer)

    construction_latencies: list[float] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        empty = await _construct_store(backend, path)
        await _close(empty)
        construction_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
    zero_record_storage_bytes = _storage_bytes(path) if backend == "sqlite" else None

    store = await _construct_store(backend, path)
    first = _context(1)
    await store.publish_work_context(first, expected_revision=None)

    current_read_latencies: list[float] = []
    for _ in range(samples):
        started = time.perf_counter_ns()
        loaded = await store.load_work_context(first.task_id)
        current_read_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert loaded is not None and loaded.revision == 1

    revision_append_latencies: list[float] = []
    current = first
    for revision in range(2, samples + 2):
        candidate = _context(revision)
        started = time.perf_counter_ns()
        receipt = await store.publish_work_context(
            candidate,
            expected_revision=current.revision,
        )
        revision_append_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert receipt.changed is True
        current = receipt.context

    first_checkpoint = _checkpoint(current, 1)
    await store.advance_recall_checkpoint(first_checkpoint, expected_revision=None)
    checkpoint_advance_latencies: list[float] = []
    for revision in range(2, samples + 2):
        candidate = _checkpoint(current, revision)
        started = time.perf_counter_ns()
        stored = await store.advance_recall_checkpoint(
            candidate,
            expected_revision=revision - 1,
        )
        checkpoint_advance_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert stored == candidate

    await _close(store)
    durable_record_count = (samples + 1) * 2 + (samples + 1)
    populated_storage_bytes = _storage_bytes(path) if backend == "sqlite" else None
    incremental_storage_bytes = (
        None
        if zero_record_storage_bytes is None or populated_storage_bytes is None
        else populated_storage_bytes - zero_record_storage_bytes
    )
    return {
        "backend": backend,
        "sample_count": samples,
        "zero_record_construction_latency": _latency_summary(construction_latencies),
        "current_read_latency": _latency_summary(current_read_latencies),
        "revision_append_latency": _latency_summary(revision_append_latencies),
        "checkpoint_advance_latency": _latency_summary(checkpoint_advance_latencies),
        "durable_record_count": durable_record_count,
        "zero_record_storage_bytes": zero_record_storage_bytes,
        "populated_storage_bytes": populated_storage_bytes,
        "incremental_storage_bytes": incremental_storage_bytes,
        "storage_bytes_per_durable_record": (
            None
            if incremental_storage_bytes is None
            else round(incremental_storage_bytes / durable_record_count, 6)
        ),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_zero_record_construction_p95_ms": result[
                "zero_record_construction_latency"
            ]["p95_ms"],
            f"{backend}_current_read_p95_ms": result["current_read_latency"]["p95_ms"],
            f"{backend}_revision_append_p95_ms": result["revision_append_latency"]["p95_ms"],
            f"{backend}_checkpoint_advance_p95_ms": result["checkpoint_advance_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks.update(
                {
                    "sqlite_zero_record_construction_p50_ms": result[
                        "zero_record_construction_latency"
                    ]["p50_ms"],
                    "sqlite_revision_append_p50_ms": result["revision_append_latency"]["p50_ms"],
                    "sqlite_checkpoint_advance_p50_ms": result["checkpoint_advance_latency"][
                        "p50_ms"
                    ],
                    "sqlite_storage_bytes_per_durable_record": result[
                        "storage_bytes_per_durable_record"
                    ],
                }
            )
        for metric, observed in checks.items():
            ceiling = _CEILINGS[metric]
            if observed > ceiling:
                findings.append(
                    {
                        "backend": backend,
                        "metric": metric,
                        "observed": round(observed, 6),
                        "ceiling": ceiling,
                    }
                )
    return findings


async def _run(samples: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-agent-work-context-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(backend, samples=samples, directory=directory)
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "samples": samples,
            "provider_calls": 0,
            "backends": ["memory", "sqlite"],
        },
        "control": {
            "kind": "current_store_with_zero_work_context_records",
            "historical_pre_feature": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "ceilings": _CEILINGS,
        "results": results,
        "ceiling_findings": findings,
        "within_ceilings": not findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=_DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 1_000:
        parser.error("--samples must be between 1 and 1000")

    report = asyncio.run(_run(args.samples))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
