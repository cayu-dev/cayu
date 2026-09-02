#!/usr/bin/env python3
"""Measure provider-free durable knowledge-enrichment lifecycle overhead."""

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
    InMemoryKnowledgeStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    KnowledgeCurator,
    KnowledgeCuratorConfig,
    KnowledgeEnrichmentJobStatus,
    KnowledgeEnrichmentQueue,
    KnowledgeEnrichmentQueueConfig,
    KnowledgeEnrichmentRequest,
    KnowledgeEnrichmentTrigger,
    KnowledgeEnrichmentWorker,
    KnowledgeStatus,
    LearningBatch,
    LearningCandidate,
    LearningDecision,
    LearningSignal,
    LearningSourceReference,
    LearningVerdict,
    SQLiteKnowledgeStore,
    SQLiteTaskStore,
    TaskExecutionSource,
    TaskRetryPolicy,
    TaskTerminalizationRetryPolicy,
    TaskTerminalizationUncertain,
)

_SCHEMA_VERSION = "cayu.knowledge_enrichment_jobs_performance.v1"
_DEFAULT_OPERATION_COUNT = 20
_NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
_NAMESPACE = "performance:knowledge-enrichment"
_LABELS = {"suite": "knowledge-enrichment"}
_SCOPE = KnowledgeAccessScope.for_namespace(
    _NAMESPACE,
    required_labels=_LABELS,
    allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE],
)
_CEILINGS = {
    "memory_enqueue_p95_ms": 50.0,
    "memory_processing_p95_ms": 100.0,
    "memory_replay_p95_ms": 50.0,
    "memory_empty_poll_p95_ms": 10.0,
    "sqlite_enqueue_p95_ms": 150.0,
    "sqlite_processing_p95_ms": 250.0,
    "sqlite_replay_p95_ms": 150.0,
    "sqlite_empty_poll_p95_ms": 25.0,
    "sqlite_storage_bytes_per_job": 262_144.0,
}


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
        self.calls += 1
        return [
            LearningCandidate(
                proposal_key=f"proposal:{batch.id}",
                text=f"Retain the bounded procedure learned from {batch.id}.",
                signal_ids=tuple(signal.id for signal in batch.signals),
                kind="procedure",
            )
        ]


class _Evaluator:
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate_candidate(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> LearningDecision:
        self.calls += 1
        return LearningDecision(
            verdict=LearningVerdict.ACCEPTED,
            code="performance_source_supported",
        )


class _AcknowledgementLossTaskStore(InMemoryTaskStore):
    def __init__(self) -> None:
        super().__init__()
        self._lose_acknowledgement = True

    async def settle_task_retry_attempt(self, request):
        receipt = await super().settle_task_retry_attempt(request)
        if self._lose_acknowledgement:
            self._lose_acknowledgement = False
            raise ConnectionError("injected acknowledgement loss")
        return receipt


class _UnavailableSettlementSQLiteTaskStore(SQLiteTaskStore):
    async def complete_task(self, task_id, result, *, worker_id=None, handoff_id=None):
        raise ConnectionError("injected preparation-completion outage")

    async def settle_task_retry_attempt(self, request):
        raise ConnectionError("injected settlement outage")


class _UnavailablePreparationTaskStore(InMemoryTaskStore):
    def __init__(self) -> None:
        super().__init__()
        self._fail_preparation = True

    async def create_task(self, request):
        if request.title == "Knowledge enrichment preparation" and self._fail_preparation:
            self._fail_preparation = False
            raise ConnectionError("injected preparation outage")
        return await super().create_task(request)


def _curator_config() -> KnowledgeCuratorConfig:
    return KnowledgeCuratorConfig(
        candidate_generator_identity="performance.generator.v1",
        evaluator_identity="performance.evaluator.v1",
        namespace=_NAMESPACE,
        labels=_LABELS,
    )


def _request(queue: KnowledgeEnrichmentQueue, index: int) -> KnowledgeEnrichmentRequest:
    identity = f"performance-{index:04d}"
    source = LearningSourceReference(
        source_type="performance_event",
        source_id=identity,
        source_revision="1",
        source_hash=f"sha256:{identity}",
    )
    signal = LearningSignal(
        id=identity,
        deduplication_key=identity,
        kind="performance_observation",
        scope=_NAMESPACE,
        summary=f"Bounded performance observation {index}.",
        source_references=(source,),
        occurred_at=_NOW,
    )
    return KnowledgeEnrichmentRequest(
        operation_id=f"operation:{identity}",
        batch=LearningBatch(id=f"batch:{identity}", signals=(signal,)),
        trigger=KnowledgeEnrichmentTrigger(
            kind="performance_boundary",
            source_type="performance_run",
            source_id=identity,
            source_revision="1",
            occurred_at=_NOW,
        ),
        profile=queue.profile,
        submitted_at=_NOW,
        execution_source=TaskExecutionSource.SDK_TASK,
    )


def _latencies(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, (95 * len(ordered) + 99) // 100 - 1)
    return {
        "p50_ms": round(median(ordered), 6),
        "p95_ms": round(ordered[p95_index], 6),
    }


def _storage_bytes(*paths: Path) -> int:
    return sum(
        candidate.stat().st_size
        for path in paths
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"))
        if candidate.exists()
    )


async def _close(store: Any) -> None:
    close = getattr(store, "close", None)
    if callable(close):
        await close()


def _stores(backend: str, task_path: Path, knowledge_path: Path):
    if backend == "memory":
        return InMemoryTaskStore(), InMemoryKnowledgeStore(access_scope=_SCOPE)
    return (
        SQLiteTaskStore(task_path),
        SQLiteKnowledgeStore(knowledge_path, access_scope=_SCOPE),
    )


async def _backend_result(
    backend: str,
    *,
    operation_count: int,
    directory: Path,
) -> dict[str, Any]:
    task_path = directory / f"{backend}-tasks.sqlite"
    knowledge_path = directory / f"{backend}-knowledge.sqlite"
    config = _curator_config()
    queue_config = KnowledgeEnrichmentQueueConfig()
    tasks, knowledge = _stores(backend, task_path, knowledge_path)
    queue = KnowledgeEnrichmentQueue(
        tasks,
        curator_config=config,
        access_scope=_SCOPE,
        config=queue_config,
    )
    requests = [_request(queue, index) for index in range(operation_count)]
    enqueue: list[float] = []
    for request in requests:
        started = time.perf_counter_ns()
        await queue.submit(request)
        enqueue.append((time.perf_counter_ns() - started) / 1_000_000)

    fresh_store_reopen = backend == "sqlite"
    if fresh_store_reopen:
        await _close(knowledge)
        await _close(tasks)
        tasks, knowledge = _stores(backend, task_path, knowledge_path)
        queue = KnowledgeEnrichmentQueue(
            tasks,
            curator_config=config,
            access_scope=_SCOPE,
            config=queue_config,
        )

    generator = _Generator()
    evaluator = _Evaluator()
    curator = KnowledgeCurator(
        knowledge,
        candidate_generator=generator,
        evaluator=evaluator,
        config=config,
        access_scope=_SCOPE,
    )
    worker = KnowledgeEnrichmentWorker(queue, curator)
    processing: list[float] = []
    max_result_bytes = 0
    for index in range(operation_count):
        started = time.perf_counter_ns()
        job = await worker.process_next(
            worker_id=f"performance-worker-{index:04d}",
            lease_seconds=30,
            reclaim=False,
        )
        processing.append((time.perf_counter_ns() - started) / 1_000_000)
        if job is None or job.status is not KnowledgeEnrichmentJobStatus.COMPLETED:
            raise RuntimeError("Knowledge enrichment performance job did not complete.")
        max_result_bytes = max(max_result_bytes, len(job.model_dump_json().encode("utf-8")))

    calls_before_replay = (generator.calls, evaluator.calls)
    replay: list[float] = []
    for request in requests:
        started = time.perf_counter_ns()
        job = await queue.submit(request)
        replay.append((time.perf_counter_ns() - started) / 1_000_000)
        if job.status is not KnowledgeEnrichmentJobStatus.COMPLETED:
            raise RuntimeError("Exact enrichment replay lost its terminal result.")
    replay_component_calls = (
        generator.calls - calls_before_replay[0],
        evaluator.calls - calls_before_replay[1],
    )

    empty_poll: list[float] = []
    for index in range(operation_count):
        started = time.perf_counter_ns()
        idle = await worker.process_next(
            worker_id=f"empty-worker-{index:04d}",
            lease_seconds=30,
        )
        empty_poll.append((time.perf_counter_ns() - started) / 1_000_000)
        if idle is not None:
            raise RuntimeError("Empty enrichment poll unexpectedly claimed work.")

    await curator.aclose()
    await _close(knowledge)
    await _close(tasks)
    populated_storage = _storage_bytes(task_path, knowledge_path) if backend == "sqlite" else None
    control_storage = None
    storage_per_job = None
    if backend == "sqlite":
        control_tasks_path = directory / "sqlite-control-tasks.sqlite"
        control_knowledge_path = directory / "sqlite-control-knowledge.sqlite"
        control_tasks, control_knowledge = _stores(
            backend,
            control_tasks_path,
            control_knowledge_path,
        )
        await _close(control_knowledge)
        await _close(control_tasks)
        control_storage = _storage_bytes(control_tasks_path, control_knowledge_path)
        assert populated_storage is not None
        storage_per_job = round(
            max(0, populated_storage - control_storage) / operation_count,
            6,
        )

    return {
        "backend": backend,
        "operation_count": operation_count,
        "fresh_store_reopen_before_processing": fresh_store_reopen,
        "enqueue_latency": _latencies(enqueue),
        "processing_latency": _latencies(processing),
        "exact_replay_latency": _latencies(replay),
        "empty_poll_latency": _latencies(empty_poll),
        "generator_calls": generator.calls,
        "evaluator_calls": evaluator.calls,
        "replay_generator_calls": replay_component_calls[0],
        "replay_evaluator_calls": replay_component_calls[1],
        "max_job_result_json_bytes": max_result_bytes,
        "control_storage_bytes": control_storage,
        "populated_storage_bytes": populated_storage,
        "storage_bytes_per_job": storage_per_job,
    }


async def _acknowledgement_loss_probe() -> bool:
    tasks = _AcknowledgementLossTaskStore()
    knowledge = InMemoryKnowledgeStore(access_scope=_SCOPE)
    config = _curator_config()
    queue = KnowledgeEnrichmentQueue(
        tasks,
        curator_config=config,
        access_scope=_SCOPE,
    )
    request = _request(queue, 90_001)
    await queue.submit(request)
    generator = _Generator()
    evaluator = _Evaluator()
    curator = KnowledgeCurator(
        knowledge,
        candidate_generator=generator,
        evaluator=evaluator,
        config=config,
        access_scope=_SCOPE,
    )
    try:
        job = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="ack-loss-worker",
            lease_seconds=30,
        )
        return bool(
            job is not None
            and job.status is KnowledgeEnrichmentJobStatus.COMPLETED
            and generator.calls == 1
            and evaluator.calls == 1
        )
    finally:
        await curator.aclose()


async def _worker_loss_probe(directory: Path) -> bool:
    task_path = directory / "worker-loss-tasks.sqlite"
    knowledge_path = directory / "worker-loss-knowledge.sqlite"
    tasks = _UnavailableSettlementSQLiteTaskStore(task_path)
    knowledge = SQLiteKnowledgeStore(knowledge_path, access_scope=_SCOPE)
    config = _curator_config()
    queue_config = KnowledgeEnrichmentQueueConfig(
        terminalization_retry_policy=TaskTerminalizationRetryPolicy(
            max_attempts=1,
            attempt_timeout_seconds=1.0,
            initial_backoff_seconds=0.0,
            backoff_multiplier=1.0,
            max_backoff_seconds=0.0,
        )
    )
    queue = KnowledgeEnrichmentQueue(
        tasks,
        curator_config=config,
        access_scope=_SCOPE,
        config=queue_config,
    )
    request = _request(queue, 90_002)
    await queue.submit(request)
    generator = _Generator()
    evaluator = _Evaluator()
    curator = KnowledgeCurator(
        knowledge,
        candidate_generator=generator,
        evaluator=evaluator,
        config=config,
        access_scope=_SCOPE,
    )
    try:
        await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="lost-worker",
            lease_seconds=1,
        )
    except TaskTerminalizationUncertain:
        pass
    else:
        await curator.aclose()
        await knowledge.close()
        await tasks.close()
        return False
    await curator.aclose()
    await knowledge.close()
    await tasks.close()
    await asyncio.sleep(1.05)

    recovered_tasks = SQLiteTaskStore(task_path)
    recovered_knowledge = SQLiteKnowledgeStore(knowledge_path, access_scope=_SCOPE)
    recovered_queue = KnowledgeEnrichmentQueue(
        recovered_tasks,
        curator_config=config,
        access_scope=_SCOPE,
        config=queue_config,
    )
    recovered_curator = KnowledgeCurator(
        recovered_knowledge,
        candidate_generator=generator,
        evaluator=evaluator,
        config=config,
        access_scope=_SCOPE,
    )
    try:
        job = await KnowledgeEnrichmentWorker(recovered_queue, recovered_curator).process_next(
            worker_id="recovery-worker",
            lease_seconds=30,
        )
        return bool(
            job is not None
            and job.status is KnowledgeEnrichmentJobStatus.COMPLETED
            and generator.calls == 1
            and evaluator.calls == 1
        )
    finally:
        await recovered_curator.aclose()
        await recovered_knowledge.close()
        await recovered_tasks.close()


async def _preparation_loss_probe() -> bool:
    tasks = _UnavailablePreparationTaskStore()
    knowledge = InMemoryKnowledgeStore(access_scope=_SCOPE)
    config = _curator_config()
    queue = KnowledgeEnrichmentQueue(
        tasks,
        curator_config=config,
        access_scope=_SCOPE,
        config=KnowledgeEnrichmentQueueConfig(
            retry_policy=TaskRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.0,
                backoff_multiplier=1.0,
                max_backoff_seconds=0.0,
            )
        ),
    )
    await queue.submit(_request(queue, 90_003))
    generator = _Generator()
    evaluator = _Evaluator()
    curator = KnowledgeCurator(
        knowledge,
        candidate_generator=generator,
        evaluator=evaluator,
        config=config,
        access_scope=_SCOPE,
    )
    worker = KnowledgeEnrichmentWorker(queue, curator)
    try:
        first = await worker.process_next(
            worker_id="preparation-loss-worker-1",
            lease_seconds=30,
        )
        second = await worker.process_next(
            worker_id="preparation-loss-worker-2",
            lease_seconds=30,
        )
        return bool(
            first is not None
            and first.status is KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
            and second is not None
            and second.status is KnowledgeEnrichmentJobStatus.FAILED
            and generator.calls == 1
            and evaluator.calls == 1
        )
    finally:
        await curator.aclose()


def _ceiling_findings(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for result in results:
        backend = result["backend"]
        checks = {
            f"{backend}_enqueue_p95_ms": result["enqueue_latency"]["p95_ms"],
            f"{backend}_processing_p95_ms": result["processing_latency"]["p95_ms"],
            f"{backend}_replay_p95_ms": result["exact_replay_latency"]["p95_ms"],
            f"{backend}_empty_poll_p95_ms": result["empty_poll_latency"]["p95_ms"],
        }
        if backend == "sqlite":
            checks["sqlite_storage_bytes_per_job"] = result["storage_bytes_per_job"]
        for metric, observed in checks.items():
            ceiling = _CEILINGS[metric]
            if observed > ceiling:
                findings.append(
                    {
                        "backend": backend,
                        "metric": metric,
                        "observed": observed,
                        "ceiling": ceiling,
                    }
                )
    return findings


async def _run(operation_count: int) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cayu-knowledge-enrichment-performance-") as raw:
        directory = Path(raw)
        results = [
            await _backend_result(
                backend,
                operation_count=operation_count,
                directory=directory,
            )
            for backend in ("memory", "sqlite")
        ]
        acknowledgement_loss_reconciled = await _acknowledgement_loss_probe()
        worker_loss_recovered = await _worker_loss_probe(directory)
        preparation_loss_fenced = await _preparation_loss_probe()
    findings = _ceiling_findings(results)
    correctness = {
        "acknowledgement_loss_reconciled": acknowledgement_loss_reconciled,
        "worker_loss_recovered_after_store_reopen_and_lease_expiry": worker_loss_recovered,
        "preparation_write_failure_did_not_redispatch_semantics": preparation_loss_fenced,
        "exact_replay_component_calls": sum(
            result["replay_generator_calls"] + result["replay_evaluator_calls"]
            for result in results
        ),
        "provider_calls": 0,
    }
    return {
        "schema_version": _SCHEMA_VERSION,
        "workload": {
            "operation_count_per_backend": operation_count,
            "candidate_count_per_operation": 1,
            "provider_calls": 0,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "ceilings": _CEILINGS,
        "results": results,
        "correctness": correctness,
        "ceiling_findings": findings,
        "within_ceilings": not findings
        and all(
            (
                acknowledgement_loss_reconciled,
                worker_loss_recovered,
                preparation_loss_fenced,
                correctness["exact_replay_component_calls"] == 0,
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation-count", type=int, default=_DEFAULT_OPERATION_COUNT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if not 1 <= arguments.operation_count <= 50:
        parser.error("--operation-count must be between 1 and 50")

    report = asyncio.run(_run(arguments.operation_count))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        sys.stdout.write(rendered)
    else:
        arguments.output.write_text(rendered, encoding="utf-8")
    return 1 if arguments.check and not report["within_ceilings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
