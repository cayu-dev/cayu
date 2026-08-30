#!/usr/bin/env python3
"""Measure provider-free full, delta, and no-work checkpoint recall overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from cayu import (
    DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID,
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    AgentRecallProcessingMode,
    AgentRecallProcessingRequest,
    AgentRecallProcessor,
    AgentRecallProcessorConfig,
    AgentWorkContext,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeIndexReadinessUpdate,
    KnowledgeIndexState,
    RecallSituation,
    SQLiteKnowledgeStore,
    WeightedReciprocalRankFusionConfig,
    knowledge_chunk_embedding_identity,
)
from cayu.storage import MAX_KNOWLEDGE_REVISION_SEARCH_REFS

_SCHEMA_VERSION = "cayu.checkpoint_recall_performance.v1"
_DEFAULT_SAMPLES = 50
_DEFAULT_EXISTING_RECORDS = 500
_MAX_DELTA_RECORDS = MAX_KNOWLEDGE_REVISION_SEARCH_REFS - 1
_NOW = datetime(2026, 8, 28, 10, 0, tzinfo=UTC)
_NAMESPACE = "benchmark:checkpoint-recall"
_QUERY = "checkpoint aware delta target phrase memory"

_CEILINGS = {
    "memory_full_index_p95_ms": 100.0,
    "memory_delta_p95_ms": 20.0,
    "memory_max_delta_p95_ms": 250.0,
    "memory_no_work_p95_ms": 5.0,
    "sqlite_full_index_p50_ms": 100.0,
    "sqlite_full_index_p95_ms": 500.0,
    "sqlite_delta_p50_ms": 30.0,
    "sqlite_delta_p95_ms": 250.0,
    "sqlite_max_delta_p50_ms": 100.0,
    "sqlite_max_delta_p95_ms": 500.0,
    "sqlite_no_work_p50_ms": 15.0,
    "sqlite_no_work_p95_ms": 100.0,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _scope() -> KnowledgeAccessScope:
    return KnowledgeAccessScope.for_namespace(_NAMESPACE)


def _context() -> AgentWorkContext:
    return AgentWorkContext.create(
        task_id="checkpoint-recall-performance",
        goal="Measure bounded checkpoint-aware cross-agent recall",
        revision=1,
        operation_id="checkpoint-recall-performance-context",
        published_by="performance-runner",
        published_at=_NOW,
        scope_ids=("repository:cayu",),
        repository_paths=("src/cayu/recall_processing.py",),
        code_symbols=("AgentRecallProcessor",),
    )


def _fusion_config() -> WeightedReciprocalRankFusionConfig:
    return WeightedReciprocalRankFusionConfig(
        configuration_version="checkpoint-recall-performance-v1",
        channel_weights={
            KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
            KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
        },
        max_candidates_per_channel=20,
        fused_head_limit=20,
    )


def _request(*, checkpoint, operation_id: str) -> AgentRecallProcessingRequest:
    return AgentRecallProcessingRequest(
        agent_id="agent:performance",
        work_context=_context(),
        situation=RecallSituation(
            query=_QUERY,
            knowledge_access_scope=_scope(),
            knowledge_namespace=_NAMESPACE,
            current_time=_NOW,
        ),
        checkpoint_stream_id=DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID,
        checkpoint=checkpoint,
        processing_id=f"processing:{operation_id}",
        operation_id=operation_id,
        updated_by="performance-runner",
        updated_at=_NOW,
    )


async def _create_entry(store, entry_id: str, text: str) -> None:
    await store.create_entry(
        KnowledgeEntry(id=entry_id, namespace=_NAMESPACE, text=text),
        [
            KnowledgeChunk(
                id=f"{entry_id}-chunk",
                entry_id=entry_id,
                text=text,
                chunk_index=0,
            )
        ],
        access_scope=_scope(),
    )


async def _backend_result(
    backend: str,
    *,
    samples: int,
    existing_records: int,
    directory: Path,
) -> dict[str, Any]:
    store = (
        InMemoryKnowledgeStore(access_scope=_scope())
        if backend == "memory"
        else SQLiteKnowledgeStore(
            directory / "checkpoint-recall.sqlite",
            access_scope=_scope(),
        )
    )
    for index in range(existing_records):
        await _create_entry(
            store,
            f"existing-{index:05d}",
            f"{_QUERY} existing record {index:05d}",
        )
    pending_readiness = []
    for index in range(samples):
        chunks = await store.read_chunks(
            f"existing-{index:05d}",
            access_scope=_scope(),
        )
        identity = knowledge_chunk_embedding_identity(
            chunks[0],
            embedding_model="checkpoint-recall-performance",
            dimensions=3,
        )
        pending_readiness.append(
            await store.publish_index_readiness(
                KnowledgeIndexReadinessUpdate(
                    identity=identity,
                    state=KnowledgeIndexState.PENDING,
                    attempt_id=f"max-delta-attempt-{index:05d}",
                ),
                expected_sequence=None,
                operation_id=f"max-delta-pending-{index:05d}",
                access_scope=_scope(),
            )
        )
    processor = AgentRecallProcessor(
        store,
        fusion_config=_fusion_config(),
        config=AgentRecallProcessorConfig(
            knowledge_change_limit=_MAX_DELTA_RECORDS,
            index_readiness_limit=1,
        ),
    )

    full_latencies: list[float] = []
    full_result = None
    full_request = _request(checkpoint=None, operation_id="full-index")
    for _ in range(samples):
        started = time.perf_counter_ns()
        full_result = await processor.process(full_request)
        full_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert full_result.mode is AgentRecallProcessingMode.FULL_INDEX
    assert full_result is not None and full_result.proposed_checkpoint is not None
    checkpoint = full_result.proposed_checkpoint

    no_work_latencies: list[float] = []
    for index in range(samples):
        started = time.perf_counter_ns()
        no_work = await processor.process(
            _request(checkpoint=checkpoint, operation_id=f"no-work-{index:05d}")
        )
        no_work_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert no_work.mode is AgentRecallProcessingMode.NO_WORK

    delta_latencies: list[float] = []
    for index in range(samples):
        entry_id = f"delta-{index:05d}"
        await _create_entry(store, entry_id, f"{_QUERY} delta record {index:05d}")
        started = time.perf_counter_ns()
        delta = await processor.process(
            _request(checkpoint=checkpoint, operation_id=f"delta-{index:05d}")
        )
        delta_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert delta.mode is AgentRecallProcessingMode.DELTA
        assert [reference.entry_id for reference in delta.eligible_revisions] == [entry_id]
        assert delta.proposed_checkpoint is not None
        checkpoint = delta.proposed_checkpoint

    max_delta_latencies: list[float] = []
    for sample in range(samples):
        expected_entry_ids = [f"existing-{sample:05d}"]
        for record in range(_MAX_DELTA_RECORDS):
            entry_id = f"max-delta-{sample:05d}-{record:05d}"
            expected_entry_ids.append(entry_id)
            await _create_entry(
                store,
                entry_id,
                f"{_QUERY} maximum bounded delta record {sample:05d} {record:05d}",
            )
        pending = pending_readiness[sample]
        await store.publish_index_readiness(
            KnowledgeIndexReadinessUpdate(
                identity=pending.identity,
                state=KnowledgeIndexState.READY,
                attempt_id=pending.attempt_id,
            ),
            expected_sequence=pending.sequence,
            operation_id=f"max-delta-ready-{sample:05d}",
            access_scope=_scope(),
        )
        started = time.perf_counter_ns()
        max_delta = await processor.process(
            _request(checkpoint=checkpoint, operation_id=f"max-delta-{sample:05d}")
        )
        max_delta_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert max_delta.mode is AgentRecallProcessingMode.DELTA
        assert [reference.entry_id for reference in max_delta.eligible_revisions] == sorted(
            expected_entry_ids
        )
        assert len(max_delta.eligible_revisions) == MAX_KNOWLEDGE_REVISION_SEARCH_REFS
        assert max_delta.proposed_checkpoint is not None
        checkpoint = max_delta.proposed_checkpoint

    if isinstance(store, SQLiteKnowledgeStore):
        await store.close()
    return {
        "backend": backend,
        "sample_count": samples,
        "existing_record_count": existing_records,
        "max_delta_record_count": _MAX_DELTA_RECORDS,
        "max_delta_revision_ref_count": MAX_KNOWLEDGE_REVISION_SEARCH_REFS,
        "full_index_latency": _latency_summary(full_latencies),
        "delta_latency": _latency_summary(delta_latencies),
        "max_delta_latency": _latency_summary(max_delta_latencies),
        "no_work_latency": _latency_summary(no_work_latencies),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_full_index_p95_ms": result["full_index_latency"]["p95_ms"],
            f"{backend}_delta_p95_ms": result["delta_latency"]["p95_ms"],
            f"{backend}_max_delta_p95_ms": result["max_delta_latency"]["p95_ms"],
            f"{backend}_no_work_p95_ms": result["no_work_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks.update(
                {
                    "sqlite_full_index_p50_ms": result["full_index_latency"]["p50_ms"],
                    "sqlite_delta_p50_ms": result["delta_latency"]["p50_ms"],
                    "sqlite_max_delta_p50_ms": result["max_delta_latency"]["p50_ms"],
                    "sqlite_no_work_p50_ms": result["no_work_latency"]["p50_ms"],
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


async def _run(samples: int, existing_records: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-checkpoint-recall-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                samples=samples,
                existing_records=existing_records,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "samples": samples,
            "existing_records": existing_records,
            "max_delta_records": _MAX_DELTA_RECORDS,
            "max_delta_revision_refs": MAX_KNOWLEDGE_REVISION_SEARCH_REFS,
            "provider_calls": 0,
            "backends": ["memory", "sqlite"],
        },
        "control": {
            "kind": "current_checkpoint_recall_with_no_provider_calls",
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
    parser.add_argument("--existing-records", type=int, default=_DEFAULT_EXISTING_RECORDS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.samples <= 1_000:
        parser.error("--samples must be between 1 and 1000")
    if not 1 <= args.existing_records <= 100_000:
        parser.error("--existing-records must be between 1 and 100000")
    if args.existing_records < args.samples:
        parser.error("--existing-records must be at least --samples")

    report = asyncio.run(_run(args.samples, args.existing_records))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
