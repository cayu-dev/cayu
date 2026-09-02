#!/usr/bin/env python3
"""Measure hermetic semantic-watch evaluation, replay, and storage overhead."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, TypedDict

from cayu import (
    KNOWLEDGE_LEXICAL_CHANNEL,
    KNOWLEDGE_SEMANTIC_CHANNEL,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeEntry,
    KnowledgeRecallSource,
    KnowledgeSemanticWatchConfig,
    KnowledgeSemanticWatchDecision,
    KnowledgeSemanticWatchDisposition,
    KnowledgeSemanticWatchEvaluator,
    KnowledgeSemanticWatchReceipt,
    RecallEngine,
    SQLiteKnowledgeStore,
    WeightedReciprocalRankFusionConfig,
)

_SCHEMA_VERSION = "cayu.knowledge_semantic_watch_performance.v1"
_DEFAULT_OPERATION_COUNT = 10
_NAMESPACE = "performance:semantic-watch"
_SCOPE = KnowledgeAccessScope.for_namespace(_NAMESPACE)
_CEILINGS = {
    "memory_evaluation_p95_ms": 50.0,
    "sqlite_evaluation_p95_ms": 100.0,
    "memory_receipt_replay_p95_ms": 25.0,
    "sqlite_receipt_replay_p95_ms": 50.0,
    "sqlite_storage_bytes_per_outcome": 65_536,
}


class _EmitPolicy:
    def __init__(self) -> None:
        self.calls = 0

    async def decide_semantic_watch(self, request):
        self.calls += 1
        return KnowledgeSemanticWatchDecision(
            request_sha256=request.fingerprint,
            disposition=KnowledgeSemanticWatchDisposition.EMIT,
            policy_identity="performance.semantic-watch-policy",
            policy_version="1",
            code="performance_match",
        )


class _EvaluationValues(TypedDict):
    operation_id: str
    observation_id: str
    observation_source_type: str
    observation_source_id: str
    observation_text: str


def _engine(store) -> RecallEngine:
    return RecallEngine(
        (KnowledgeRecallSource(store),),
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="performance.semantic-watch-recall.v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        ),
    )


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


async def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        await close()


async def _backend_result(
    backend: str,
    *,
    operation_count: int,
    directory: Path,
) -> dict[str, Any]:
    control_path = directory / f"{backend}-control.sqlite"
    watched_path = directory / f"{backend}-watched.sqlite"
    if backend == "memory":
        control_store = InMemoryKnowledgeStore(access_scope=_SCOPE)
        watched_store = InMemoryKnowledgeStore(access_scope=_SCOPE)
    else:
        control_store = SQLiteKnowledgeStore(control_path, access_scope=_SCOPE)
        watched_store = SQLiteKnowledgeStore(watched_path, access_scope=_SCOPE)
    for index in range(operation_count):
        entry = KnowledgeEntry(
            id=f"performance-watch-{index:04d}",
            text=f"semanticwatchmarker{index:04d}",
            namespace=_NAMESPACE,
        )
        await control_store.create_entry(entry)
        await watched_store.create_entry(entry)

    policy = _EmitPolicy()
    evaluator = KnowledgeSemanticWatchEvaluator(
        watched_store,
        _engine(watched_store),
        config=KnowledgeSemanticWatchConfig(
            watch_identity="performance.semantic-watch",
            watch_version="1",
            recall_profile_identity="performance.semantic-watch-recall",
            recall_profile_version="1",
            policy_identity="performance.semantic-watch-policy",
            policy_version="1",
            knowledge_namespace=_NAMESPACE,
        ),
        policy=policy,
    )
    evaluation_latencies: list[float] = []
    replay_latencies: list[float] = []
    receipts: list[tuple[_EvaluationValues, KnowledgeSemanticWatchReceipt]] = []
    for index in range(operation_count):
        values: _EvaluationValues = {
            "operation_id": f"performance-watch-operation-{index:04d}",
            "observation_id": f"performance-observation-{index:04d}",
            "observation_source_type": "performance",
            "observation_source_id": f"performance-source-{index:04d}",
            "observation_text": f"semanticwatchmarker{index:04d}",
        }
        started = time.perf_counter_ns()
        receipt = await evaluator.evaluate(**values)
        evaluation_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if not receipt.authority.evidence.candidates:
            raise RuntimeError("Semantic-watch performance evaluation lost its exact match.")
        receipts.append((values, receipt))

    calls_before_replay = policy.calls
    for values, _receipt in receipts:
        started = time.perf_counter_ns()
        replay = await evaluator.evaluate(**values)
        replay_latencies.append((time.perf_counter_ns() - started) / 1_000_000)
        if not replay.replayed:
            raise RuntimeError("Semantic-watch performance replay was not identified.")
    if policy.calls != calls_before_replay:
        raise RuntimeError("Semantic-watch performance replay called policy again.")

    receipt_bytes = [
        len(receipt.model_dump_json(warnings=False).encode()) for _, receipt in receipts
    ]
    await _close(control_store)
    await _close(watched_store)
    control_bytes = _storage_bytes(control_path) if backend == "sqlite" else None
    watched_bytes = _storage_bytes(watched_path) if backend == "sqlite" else None
    storage_overhead = (
        None if control_bytes is None or watched_bytes is None else watched_bytes - control_bytes
    )
    return {
        "backend": backend,
        "operation_count": operation_count,
        "evaluation_latency": _latency_summary(evaluation_latencies),
        "receipt_replay_latency": _latency_summary(replay_latencies),
        "max_receipt_json_bytes": max(receipt_bytes),
        "control_storage_bytes": control_bytes,
        "watched_storage_bytes": watched_bytes,
        "storage_byte_overhead": storage_overhead,
        "storage_bytes_per_outcome": (
            None if storage_overhead is None else round(storage_overhead / operation_count, 6)
        ),
    }


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_evaluation_p95_ms": result["evaluation_latency"]["p95_ms"],
            f"{backend}_receipt_replay_p95_ms": result["receipt_replay_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks["sqlite_storage_bytes_per_outcome"] = result["storage_bytes_per_outcome"]
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


async def _run(operation_count: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-knowledge-semantic-watch-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                operation_count=operation_count,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
    findings = _ceiling_findings(results)
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "operation_count": operation_count,
            "provider_calls": 0,
            "policy_calls": operation_count,
            "replay_policy_calls": 0,
        },
        "control": {
            "kind": "identical_current_runtime_knowledge_without_watch_outcomes",
            "historical_pre_feature": False,
            "operation_count": operation_count,
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
    parser.add_argument("--operation-count", type=int, default=_DEFAULT_OPERATION_COUNT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.operation_count <= 50:
        parser.error("--operation-count must be between 1 and 50")

    report = asyncio.run(_run(args.operation_count))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 1 if args.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
