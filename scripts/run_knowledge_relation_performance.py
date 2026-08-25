#!/usr/bin/env python3
"""Measure hermetic revision-bound knowledge-relation storage overhead."""

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
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRelation,
    KnowledgeRelationKind,
    KnowledgeRelationQuery,
    KnowledgeRevisionRef,
    SQLiteKnowledgeStore,
    prepare_knowledge_relations,
)

_SCHEMA_VERSION = "cayu.knowledge_relation_performance.v1"
_DEFAULT_RELATIONS = 50
_DEFAULT_UNRELATED_RELATIONS = 5_000
_DEFAULT_BATCH_SIZE = 10
_DEFAULT_QUERY_ITERATIONS = 30
_NOW = datetime(2026, 8, 25, 9, 0, tzinfo=UTC)

_CEILINGS = {
    "memory_zero_relation_entry_publish_p95_ms": 10.0,
    "sqlite_zero_relation_entry_publish_p95_ms": 100.0,
    "preparation_p95_ms_per_relation": 2.0,
    "memory_relation_publish_p95_ms_per_relation": 5.0,
    "sqlite_relation_publish_p95_ms_per_relation": 20.0,
    # Query hydration is CPU-scheduler sensitive on shared hosted runners. The
    # checked-in 50-result baselines are below 20 ms; these ceilings retain a
    # multi-fold regression guard without failing ordinary runner variation.
    "memory_bounded_query_p95_ms": 50.0,
    "sqlite_bounded_query_p95_ms": 75.0,
    # A lookup for an isolated endpoint in a store containing thousands of
    # unrelated relations must stay index-bound instead of scanning the store.
    "memory_unrelated_lookup_p95_ms": 2.0,
    "sqlite_unrelated_lookup_p95_ms": 10.0,
    "sqlite_storage_bytes_per_relation": 32_768,
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
        candidate.stat().st_size for candidate in (path, Path(f"{path}-wal")) if candidate.exists()
    )


def _entries(count: int, unrelated_count: int) -> list[KnowledgeEntry]:
    entries = [
        KnowledgeEntry(
            id="relation-performance-anchor",
            text="Canonical anchor for bounded lineage lookup.",
            created_at=_NOW,
            updated_at=_NOW,
        ),
        KnowledgeEntry(
            id="relation-performance-isolated",
            text="Isolated endpoint for unrelated-relation lookup.",
            created_at=_NOW,
            updated_at=_NOW,
        ),
    ]
    entries.extend(
        KnowledgeEntry(
            id=f"relation-performance-target-{index:04d}",
            text=f"Canonical target {index:04d}.",
            created_at=_NOW,
            updated_at=_NOW,
        )
        for index in range(count)
    )
    if unrelated_count:
        entries.extend(
            KnowledgeEntry(
                id=f"relation-performance-background-{index:05d}",
                text=f"Unrelated background endpoint {index:05d}.",
                created_at=_NOW,
                updated_at=_NOW,
            )
            for index in range(unrelated_count + 1)
        )
    return entries


def _relations(count: int, unrelated_count: int) -> list[KnowledgeRelation]:
    relations = [
        KnowledgeRelation(
            id=f"relation-performance-{index:04d}",
            subject=KnowledgeRevisionRef(
                entry_id="relation-performance-anchor",
                revision=1,
            ),
            object=KnowledgeRevisionRef(
                entry_id=f"relation-performance-target-{index:04d}",
                revision=1,
            ),
            kind=(
                KnowledgeRelationKind.SUPERSEDES
                if index % 2 == 0
                else KnowledgeRelationKind.DERIVED_FROM
            ),
            created_by="performance-runner",
            policy_id="hermetic-v1",
            created_at=_NOW + timedelta(microseconds=index),
            metadata={"ordinal": index},
        )
        for index in range(count)
    ]
    relations.extend(
        KnowledgeRelation(
            id=f"relation-performance-background-{index:05d}",
            subject=KnowledgeRevisionRef(
                entry_id=f"relation-performance-background-{index:05d}",
                revision=1,
            ),
            object=KnowledgeRevisionRef(
                entry_id=f"relation-performance-background-{index + 1:05d}",
                revision=1,
            ),
            kind=KnowledgeRelationKind.DERIVED_FROM,
            created_by="performance-runner",
            policy_id="hermetic-v1",
            created_at=_NOW + timedelta(microseconds=count + index),
            metadata={"background_ordinal": index},
        )
        for index in range(unrelated_count)
    )
    return relations


async def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


async def _backend_result(
    backend: str,
    *,
    relation_count: int,
    unrelated_relation_count: int,
    batch_size: int,
    query_iterations: int,
    directory: Path,
) -> dict[str, Any]:
    zero_path = directory / f"{backend}-zero-relations.sqlite"
    populated_path = directory / f"{backend}-populated-relations.sqlite"
    scope = KnowledgeAccessScope.privileged()
    if backend == "memory":
        zero_store = InMemoryKnowledgeStore(access_scope=scope)
        populated_store = InMemoryKnowledgeStore(access_scope=scope)
    else:
        zero_store = SQLiteKnowledgeStore(zero_path, access_scope=scope)
        populated_store = SQLiteKnowledgeStore(populated_path, access_scope=scope)

    entries = _entries(relation_count, unrelated_relation_count)
    zero_relation_entry_latencies: list[float] = []
    for entry in entries:
        started = time.perf_counter_ns()
        await zero_store.create_entry(entry)
        zero_relation_entry_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        await populated_store.create_entry(entry)

    relation_records = _relations(relation_count, unrelated_relation_count)
    preparation_latencies: list[float] = []
    publication_latencies: list[float] = []
    batch_sizes: list[int] = []
    for start in range(0, len(relation_records), batch_size):
        batch = relation_records[start : start + batch_size]
        started = time.perf_counter_ns()
        prepare_knowledge_relations(
            batch,
            operation_id=f"relation-performance-operation-{start:04d}",
        )
        preparation_latencies.append((time.perf_counter_ns() - started) / 1_000_000)

        started = time.perf_counter_ns()
        await populated_store.publish_relations(
            batch,
            operation_id=f"relation-performance-operation-{start:04d}",
        )
        publication_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        batch_sizes.append(len(batch))

    query = KnowledgeRelationQuery(
        reference=KnowledgeRevisionRef(
            entry_id="relation-performance-anchor",
            revision=1,
        ),
        limit=relation_count,
        max_bytes=relation_count * 8_192,
    )
    unrelated_query = KnowledgeRelationQuery(
        reference=KnowledgeRevisionRef(
            entry_id="relation-performance-isolated",
            revision=1,
        ),
        limit=1,
    )
    query_latencies: list[float] = []
    unrelated_query_latencies: list[float] = []
    for _ in range(query_iterations):
        started = time.perf_counter_ns()
        result = await populated_store.read_relations(query)
        query_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if result is None or len(result.relations) != relation_count or result.truncated:
            raise RuntimeError("Bounded relation performance query lost records.")

        started = time.perf_counter_ns()
        unrelated_result = await populated_store.read_relations(unrelated_query)
        unrelated_query_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if unrelated_result is None or unrelated_result.relations or unrelated_result.truncated:
            raise RuntimeError("Unrelated relation performance lookup returned records.")

    await _close(zero_store)
    await _close(populated_store)
    zero_storage_bytes = _storage_bytes(zero_path) if backend == "sqlite" else None
    populated_storage_bytes = _storage_bytes(populated_path) if backend == "sqlite" else None
    preparation_per_relation = [
        latency / size for latency, size in zip(preparation_latencies, batch_sizes, strict=True)
    ]
    publication_per_relation = [
        latency / size for latency, size in zip(publication_latencies, batch_sizes, strict=True)
    ]
    storage_overhead = (
        None
        if zero_storage_bytes is None or populated_storage_bytes is None
        else populated_storage_bytes - zero_storage_bytes
    )
    published_relation_count = relation_count + unrelated_relation_count
    return {
        "backend": backend,
        "matched_relation_count": relation_count,
        "unrelated_relation_count": unrelated_relation_count,
        "published_relation_count": published_relation_count,
        "batch_size": batch_size,
        "batch_count": len(batch_sizes),
        "query_iteration_count": query_iterations,
        "zero_relation_entry_publish_latency": _latency_summary(zero_relation_entry_latencies),
        "preparation_latency_per_relation": _latency_summary(preparation_per_relation),
        "relation_publish_latency_per_relation": _latency_summary(publication_per_relation),
        "bounded_query_latency": _latency_summary(query_latencies),
        "unrelated_lookup_latency": _latency_summary(unrelated_query_latencies),
        "zero_relation_storage_bytes": zero_storage_bytes,
        "populated_relation_storage_bytes": populated_storage_bytes,
        "storage_byte_overhead": storage_overhead,
        "storage_bytes_per_relation": (
            None
            if storage_overhead is None
            else round(storage_overhead / published_relation_count, 6)
        ),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_zero_relation_entry_publish_p95_ms": result[
                "zero_relation_entry_publish_latency"
            ]["p95_ms"],
            "preparation_p95_ms_per_relation": result["preparation_latency_per_relation"]["p95_ms"],
            f"{backend}_relation_publish_p95_ms_per_relation": result[
                "relation_publish_latency_per_relation"
            ]["p95_ms"],
            f"{backend}_bounded_query_p95_ms": result["bounded_query_latency"]["p95_ms"],
            f"{backend}_unrelated_lookup_p95_ms": result["unrelated_lookup_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks["sqlite_storage_bytes_per_relation"] = result["storage_bytes_per_relation"]
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


async def _run(
    relation_count: int,
    unrelated_relation_count: int,
    batch_size: int,
    query_iterations: int,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-knowledge-relation-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                relation_count=relation_count,
                unrelated_relation_count=unrelated_relation_count,
                batch_size=batch_size,
                query_iterations=query_iterations,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "matched_relation_count": relation_count,
            "unrelated_relation_count": unrelated_relation_count,
            "published_relation_count": relation_count + unrelated_relation_count,
            "batch_size": batch_size,
            "query_iterations": query_iterations,
            "provider_calls": 0,
        },
        "control": {
            "kind": "current_runtime_zero_relations",
            "historical_pre_feature": False,
            "relation_count": 0,
            "entry_count": len(_entries(relation_count, unrelated_relation_count)),
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
    parser.add_argument("--relations", type=int, default=_DEFAULT_RELATIONS)
    parser.add_argument(
        "--unrelated-relations",
        type=int,
        default=_DEFAULT_UNRELATED_RELATIONS,
    )
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--query-iterations",
        type=int,
        default=_DEFAULT_QUERY_ITERATIONS,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.relations <= 100:
        parser.error("--relations must be between 1 and 100")
    if not 0 <= args.unrelated_relations <= 10_000:
        parser.error("--unrelated-relations must be between 0 and 10000")
    published_relation_count = args.relations + args.unrelated_relations
    if not 1 <= args.batch_size <= min(published_relation_count, 100):
        parser.error("--batch-size must be between 1 and the published relation count")
    if not 1 <= args.query_iterations <= 1_000:
        parser.error("--query-iterations must be between 1 and 1000")

    report = asyncio.run(
        _run(
            args.relations,
            args.unrelated_relations,
            args.batch_size,
            args.query_iterations,
        )
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
