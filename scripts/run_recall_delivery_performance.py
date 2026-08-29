#!/usr/bin/env python3
"""Measure hermetic staged-recall delivery store overhead."""

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
    AgentRecallDelivery,
    AgentRecallDeliveryEvidenceKind,
    AgentRecallDeliveryState,
    AgentRecallProcessingRequest,
    AgentRecallProcessor,
    AgentWorkContext,
    AgentWorkContextStore,
    InMemoryAgentWorkContextStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    RecallSituation,
    SQLiteAgentWorkContextStore,
    WeightedReciprocalRankFusionConfig,
)
from cayu.recall import KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL

_SCHEMA_VERSION = "cayu.recall_delivery_performance.v1"
_DEFAULT_SAMPLES = 50
_NOW = datetime(2026, 8, 29, 9, 0, tzinfo=UTC)
_NAMESPACE = "project:recall-delivery-performance"
_SCOPE = KnowledgeAccessScope.for_namespace(_NAMESPACE)

_CEILINGS = {
    "memory_stage_checkpoint_p50_ms": 25.0,
    "memory_stage_checkpoint_p95_ms": 40.0,
    "memory_no_pending_p50_ms": 2.5,
    "memory_no_pending_p95_ms": 5.0,
    "memory_claim_p50_ms": 30.0,
    "memory_claim_p95_ms": 50.0,
    "memory_acknowledgement_p50_ms": 30.0,
    "memory_acknowledgement_p95_ms": 50.0,
    "sqlite_stage_checkpoint_p50_ms": 40.0,
    "sqlite_stage_checkpoint_p95_ms": 300.0,
    "sqlite_no_pending_p50_ms": 10.0,
    "sqlite_no_pending_p95_ms": 100.0,
    "sqlite_claim_p50_ms": 50.0,
    "sqlite_claim_p95_ms": 200.0,
    "sqlite_acknowledgement_p50_ms": 50.0,
    "sqlite_acknowledgement_p95_ms": 200.0,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _context() -> AgentWorkContext:
    return AgentWorkContext.create(
        task_id="recall-delivery-performance",
        goal="Measure atomic staged recall delivery",
        revision=1,
        operation_id="recall-delivery-performance-context",
        published_by="performance-runner",
        published_at=_NOW,
        scope_ids=("repository:cayu",),
        workflow_id="workflow:memory-v5.1",
        workflow_phase="staged-recall-delivery",
        entity_ids=("feature:recall-delivery",),
        repository_paths=("src/cayu/work_context.py",),
        code_symbols=("AgentRecallDelivery",),
    )


def _processor(store: InMemoryKnowledgeStore) -> AgentRecallProcessor:
    return AgentRecallProcessor(
        store,
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="recall-delivery-performance-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        ),
    )


async def _deliveries(samples: int) -> tuple[AgentWorkContext, list[AgentRecallDelivery]]:
    context = _context()
    knowledge_store = InMemoryKnowledgeStore()
    processor = _processor(knowledge_store)
    previous_checkpoint = None
    deliveries: list[AgentRecallDelivery] = []
    for sample in range(1, samples + 1):
        entry_id = f"recall-delivery-performance-{sample:05d}"
        text = f"staged recall delivery performance evidence {sample:05d}"
        await knowledge_store.create_entry(
            KnowledgeEntry(id=entry_id, namespace=_NAMESPACE, text=text),
            [
                KnowledgeChunk(
                    id=f"{entry_id}:chunk",
                    entry_id=entry_id,
                    text=text,
                    chunk_index=0,
                )
            ],
            access_scope=_SCOPE,
        )
        operation_id = f"recall-delivery-performance-process-{sample:05d}"
        result = await processor.process(
            AgentRecallProcessingRequest(
                agent_id="agent:recall-delivery-performance",
                work_context=context,
                situation=RecallSituation(
                    query="staged recall delivery performance evidence",
                    knowledge_access_scope=_SCOPE,
                    knowledge_namespace=_NAMESPACE,
                    current_time=_NOW,
                ),
                checkpoint=previous_checkpoint,
                processing_id=f"processing:{sample:05d}",
                operation_id=operation_id,
                updated_by="performance-runner",
                updated_at=_NOW + timedelta(seconds=1, microseconds=sample),
            )
        )
        delivery = AgentRecallDelivery.from_processing_result(
            result,
            delivery_id=f"delivery:{sample:05d}",
            expected_checkpoint_revision=(
                None if previous_checkpoint is None else previous_checkpoint.revision
            ),
            staged_by="performance-runner",
            staged_at=_NOW + timedelta(seconds=2, microseconds=sample),
        )
        deliveries.append(delivery)
        previous_checkpoint = delivery.checkpoint
    return context, deliveries


async def _store(backend: str, path: Path) -> AgentWorkContextStore:
    if backend == "memory":
        return InMemoryAgentWorkContextStore(clock=lambda: _NOW + timedelta(seconds=3))
    return SQLiteAgentWorkContextStore(
        path,
        clock=lambda: _NOW + timedelta(seconds=3),
    )


async def _backend_result(
    backend: str,
    *,
    context: AgentWorkContext,
    deliveries: list[AgentRecallDelivery],
    directory: Path,
) -> dict[str, Any]:
    store = await _store(backend, directory / f"{backend}-recall-delivery.sqlite")
    await store.publish_work_context(context, expected_revision=None)

    stage_latencies: list[float] = []
    for delivery in deliveries:
        started = time.perf_counter_ns()
        staged = await store.stage_recall_delivery(delivery)
        stage_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert staged.state is AgentRecallDeliveryState.PENDING

    claim_latencies: list[float] = []
    acknowledgement_latencies: list[float] = []
    key = deliveries[0].key()
    for sample, delivery in enumerate(deliveries, start=1):
        started = time.perf_counter_ns()
        claimed = await store.claim_recall_delivery(
            key,
            claim_id=f"claim:{sample:05d}",
            worker_id="worker:recall-delivery-performance",
            lease_seconds=30,
        )
        claim_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert claimed is not None
        assert claimed.delivery.delivery_id == delivery.delivery_id
        assert claimed.claim is not None

        started = time.perf_counter_ns()
        acknowledged = await store.acknowledge_recall_delivery(
            claimed.claim,
            acknowledgement_id=f"acknowledgement:{sample:05d}",
            evidence_kind=AgentRecallDeliveryEvidenceKind.APPLICATION_HANDOFF,
            evidence_ref=f"handoff:{sample:05d}",
            acknowledged_at=claimed.claim.claimed_at,
        )
        acknowledgement_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert acknowledged.state is AgentRecallDeliveryState.ACKNOWLEDGED

    no_pending_latencies: list[float] = []
    for sample in range(1, len(deliveries) + 1):
        started = time.perf_counter_ns()
        missing = await store.claim_recall_delivery(
            key,
            claim_id=f"no-pending:{sample:05d}",
            worker_id="worker:recall-delivery-performance",
            lease_seconds=30,
        )
        no_pending_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert missing is None

    await store.close()
    return {
        "backend": backend,
        "sample_count": len(deliveries),
        "stage_checkpoint_latency": _latency_summary(stage_latencies),
        "claim_latency": _latency_summary(claim_latencies),
        "acknowledgement_latency": _latency_summary(acknowledgement_latencies),
        "no_pending_latency": _latency_summary(no_pending_latencies),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    lanes = {
        "stage_checkpoint": "stage_checkpoint_latency",
        "claim": "claim_latency",
        "acknowledgement": "acknowledgement_latency",
        "no_pending": "no_pending_latency",
    }
    for result in results:
        backend = result["backend"]
        for metric, lane in lanes.items():
            for percentile in ("p50_ms", "p95_ms"):
                name = f"{backend}_{metric}_{percentile}"
                observed = result[lane][percentile]
                ceiling = _CEILINGS[name]
                if observed > ceiling:
                    findings.append(
                        {
                            "backend": backend,
                            "metric": name,
                            "observed": round(observed, 6),
                            "ceiling": ceiling,
                        }
                    )
    return findings


async def _run(samples: int) -> dict[str, Any]:
    context, deliveries = await _deliveries(samples)
    with tempfile.TemporaryDirectory(prefix="cayu-recall-delivery-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                context=context,
                deliveries=deliveries,
                directory=directory,
            )
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
            "kind": "current_staged_recall_queue_after_all_deliveries_acknowledged",
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
