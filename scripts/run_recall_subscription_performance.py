#!/usr/bin/env python3
"""Measure hermetic idle recall-subscription store overhead."""

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
    AgentRecallProcessingRequest,
    AgentRecallProcessor,
    AgentRecallSubscription,
    AgentRecallSubscriptionEvaluationOutcome,
    AgentWorkContext,
    AgentWorkContextStore,
    AutomaticRecallPolicy,
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

_SCHEMA_VERSION = "cayu.recall_subscription_performance.v1"
_DEFAULT_SAMPLES = 50
_NOW = datetime(2026, 8, 30, 9, 0, tzinfo=UTC)
_NAMESPACE = "project:recall-subscription-performance"
_SCOPE = KnowledgeAccessScope.for_namespace(_NAMESPACE)
_QUERY = "idle recall subscription performance evidence"

_CEILINGS = {
    "memory_zero_due_p50_ms": 2.5,
    "memory_zero_due_p95_ms": 5.0,
    "memory_silent_commit_p50_ms": 15.0,
    "memory_silent_commit_p95_ms": 30.0,
    "memory_wake_publish_p50_ms": 40.0,
    "memory_wake_publish_p95_ms": 80.0,
    "memory_wake_claim_p50_ms": 30.0,
    "memory_wake_claim_p95_ms": 60.0,
    "memory_wake_acknowledgement_p50_ms": 30.0,
    "memory_wake_acknowledgement_p95_ms": 60.0,
    "sqlite_zero_due_p50_ms": 5.0,
    "sqlite_zero_due_p95_ms": 25.0,
    "sqlite_silent_commit_p50_ms": 30.0,
    "sqlite_silent_commit_p95_ms": 100.0,
    "sqlite_wake_publish_p50_ms": 50.0,
    "sqlite_wake_publish_p95_ms": 200.0,
    "sqlite_wake_claim_p50_ms": 40.0,
    "sqlite_wake_claim_p95_ms": 150.0,
    "sqlite_wake_acknowledgement_p50_ms": 40.0,
    "sqlite_wake_acknowledgement_p95_ms": 150.0,
}


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _processor(store: InMemoryKnowledgeStore) -> AgentRecallProcessor:
    return AgentRecallProcessor(
        store,
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="recall-subscription-performance-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        ),
    )


def _policy(result, *, threshold: float) -> AutomaticRecallPolicy:
    assert result.recall is not None
    return AutomaticRecallPolicy(
        calibration_version="recall-subscription-performance-v1",
        fusion_strategy_version=result.recall.fusion.strategy_version,
        fusion_configuration_version=result.recall.fusion.configuration_version,
        minimum_inject_score=threshold,
        minimum_offer_score=threshold,
    )


async def _workloads(samples: int):
    knowledge = InMemoryKnowledgeStore()
    await knowledge.create_entry(
        KnowledgeEntry(
            id="recall-subscription-performance-entry",
            namespace=_NAMESPACE,
            text=_QUERY,
        ),
        [
            KnowledgeChunk(
                id="recall-subscription-performance-entry:chunk",
                entry_id="recall-subscription-performance-entry",
                text=_QUERY,
                chunk_index=0,
            )
        ],
        access_scope=_SCOPE,
    )
    processor = _processor(knowledge)
    workloads = []
    for lane, threshold in (("silent", 1.0), ("wake", 0.0)):
        for sample in range(1, samples + 1):
            suffix = f"{lane}:{sample:05d}"
            context = AgentWorkContext.create(
                task_id=f"recall-subscription-performance:{suffix}",
                goal="Measure bounded idle recall subscription overhead",
                revision=1,
                operation_id=f"context:{suffix}",
                published_by="performance-runner",
                published_at=_NOW,
            )
            situation = RecallSituation(
                query=_QUERY,
                knowledge_access_scope=_SCOPE,
                knowledge_namespace=_NAMESPACE,
                current_time=_NOW,
            )
            policy_result = await processor.process(
                AgentRecallProcessingRequest(
                    agent_id="agent:recall-subscription-performance",
                    work_context=context,
                    situation=situation,
                    checkpoint_stream_id="recall-subscription-performance:policy",
                    checkpoint=None,
                    processing_id=f"policy-processing:{suffix}",
                    operation_id=f"policy-processing-operation:{suffix}",
                    updated_by="performance-runner",
                    updated_at=_NOW + timedelta(seconds=1),
                )
            )
            subscription = AgentRecallSubscription.create(
                subscription_id=f"subscription:{suffix}",
                agent_id="agent:recall-subscription-performance",
                work_context=context,
                knowledge_namespace=_NAMESPACE,
                access_policy_sha256=policy_result.access_policy_sha256,
                admission_policy=_policy(policy_result, threshold=threshold),
                minimum_interval_seconds=60,
                expires_at=_NOW + timedelta(days=1),
                revision=1,
                operation_id=f"subscription-publication:{suffix}",
                published_by="performance-runner",
                published_at=_NOW,
                query=_QUERY,
            )
            result = await processor.process(
                AgentRecallProcessingRequest(
                    agent_id="agent:recall-subscription-performance",
                    work_context=context,
                    situation=situation,
                    checkpoint_stream_id=subscription.checkpoint_stream_id(),
                    checkpoint=None,
                    processing_id=f"processing:{suffix}",
                    operation_id=f"processing-operation:{suffix}",
                    updated_by="performance-runner",
                    updated_at=_NOW + timedelta(seconds=1),
                )
            )
            workloads.append((lane, sample, context, subscription, result))
    return workloads


async def _store(backend: str, path: Path) -> AgentWorkContextStore:
    def clock() -> datetime:
        return _NOW + timedelta(seconds=10)

    if backend == "memory":
        return InMemoryAgentWorkContextStore(clock=clock)
    return SQLiteAgentWorkContextStore(path, clock=clock)


async def _backend_result(
    backend: str,
    *,
    workloads,
    samples: int,
    directory: Path,
) -> dict[str, Any]:
    store = await _store(backend, directory / f"{backend}-recall-subscription.sqlite")
    claims = {}
    for lane, sample, context, subscription, _result in workloads:
        await store.publish_work_context(context, expected_revision=None)
        await store.publish_recall_subscription(subscription, expected_revision=None)
        record = await store.claim_due_recall_subscription(
            subscription.checkpoint_key(),
            claim_id=f"subscription-claim:{lane}:{sample:05d}",
            runner_id="runner:recall-subscription-performance",
            lease_seconds=300,
        )
        assert record is not None and record.claim is not None
        claims[(lane, sample)] = record.claim

    zero_due_latencies: list[float] = []
    claimed_key = workloads[0][3].checkpoint_key()
    for sample in range(1, samples + 1):
        started = time.perf_counter_ns()
        missing = await store.claim_due_recall_subscription(
            claimed_key,
            claim_id=f"zero-due:{sample:05d}",
            runner_id="runner:recall-subscription-performance",
            lease_seconds=300,
        )
        zero_due_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert missing is None

    silent_commit_latencies: list[float] = []
    wake_publish_latencies: list[float] = []
    for lane, sample, _context, _subscription, result in workloads:
        started = time.perf_counter_ns()
        evaluation = await store.commit_recall_subscription_evaluation(
            claims[(lane, sample)],
            result,
            evaluation_id=f"evaluation:{lane}:{sample:05d}",
            delivery_id=(None if lane == "silent" else f"delivery:{sample:05d}"),
            staged_by="performance-runner",
            evaluated_at=_NOW + timedelta(seconds=10),
        )
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        if lane == "silent":
            silent_commit_latencies.append(elapsed)
            assert evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.SILENT
        else:
            wake_publish_latencies.append(elapsed)
            assert evaluation.outcome is AgentRecallSubscriptionEvaluationOutcome.WAKE

    wake_claim_latencies: list[float] = []
    wake_acknowledgement_latencies: list[float] = []
    wake_workloads = [item for item in workloads if item[0] == "wake"]
    for _lane, sample, _context, subscription, _result in wake_workloads:
        started = time.perf_counter_ns()
        wake = await store.claim_recall_subscription_wake(
            subscription.checkpoint_key(),
            claim_id=f"wake-claim:{sample:05d}",
            runner_id="scheduler:recall-subscription-performance",
            lease_seconds=300,
        )
        wake_claim_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert wake is not None and wake.claim is not None

        started = time.perf_counter_ns()
        acknowledged = await store.acknowledge_recall_subscription_wake(
            wake.claim,
            acknowledgement_id=f"wake-acknowledgement:{sample:05d}",
            acknowledged_at=wake.claim.claimed_at,
        )
        wake_acknowledgement_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        assert acknowledged.acknowledgement is not None

    await store.close()
    return {
        "backend": backend,
        "sample_count": samples,
        "zero_due_latency": _latency_summary(zero_due_latencies),
        "silent_commit_latency": _latency_summary(silent_commit_latencies),
        "wake_publish_latency": _latency_summary(wake_publish_latencies),
        "wake_claim_latency": _latency_summary(wake_claim_latencies),
        "wake_acknowledgement_latency": _latency_summary(wake_acknowledgement_latencies),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    lanes = {
        "zero_due": "zero_due_latency",
        "silent_commit": "silent_commit_latency",
        "wake_publish": "wake_publish_latency",
        "wake_claim": "wake_claim_latency",
        "wake_acknowledgement": "wake_acknowledgement_latency",
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
    workloads = await _workloads(samples)
    with tempfile.TemporaryDirectory(prefix="cayu-recall-subscription-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                workloads=workloads,
                samples=samples,
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
            "kind": "current_store_with_bounded_claimed_subscriptions",
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
