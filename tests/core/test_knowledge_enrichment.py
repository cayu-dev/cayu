from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tests.core.postgres_contention_support import drop_cayu_tables

from cayu import (
    InMemoryKnowledgeStore,
    InMemoryTaskStore,
    KnowledgeAccessScope,
    KnowledgeCurator,
    KnowledgeCuratorConfig,
    KnowledgeEnrichmentConflict,
    KnowledgeEnrichmentFailureCategory,
    KnowledgeEnrichmentFailureDecision,
    KnowledgeEnrichmentFeedbackAuthorization,
    KnowledgeEnrichmentJobRejected,
    KnowledgeEnrichmentJobStatus,
    KnowledgeEnrichmentQueue,
    KnowledgeEnrichmentQueueConfig,
    KnowledgeEnrichmentRequest,
    KnowledgeEnrichmentTrigger,
    KnowledgeEnrichmentWorker,
    LearningBatch,
    LearningCandidate,
    LearningDecision,
    LearningSignal,
    LearningSourceReference,
    LearningVerdict,
    PostgresTaskStore,
    SQLiteTaskStore,
    TaskClaimLost,
    TaskCreate,
    TaskExecutionSource,
    TaskRetryPolicy,
    TaskStatus,
    TaskTerminalizationRetryPolicy,
    TaskTerminalizationUncertain,
)
from cayu.knowledge_enrichment import _parse_failure_payload, _result_from_task
from cayu.storage.migrations import SchemaMode

_NOW = datetime(2026, 9, 2, 12, 0, tzinfo=UTC)
_ACCESS_SCOPE = KnowledgeAccessScope.privileged()


def _source(source_id: str = "event-1") -> LearningSourceReference:
    return LearningSourceReference(
        source_type="session_event",
        source_id=source_id,
        source_hash=f"sha256:{source_id}",
        locator={"event_id": source_id},
    )


def _signal(
    signal_id: str = "signal-1", *, summary: str = "A release check failed."
) -> LearningSignal:
    return LearningSignal(
        id=signal_id,
        deduplication_key=f"dedupe:{signal_id}",
        kind="release_observation",
        scope="project:cayu",
        summary=summary,
        source_references=(_source(f"event:{signal_id}"),),
        occurred_at=_NOW,
    )


def _batch(*signals: LearningSignal) -> LearningBatch:
    return LearningBatch(id="batch-1", signals=signals or (_signal(),))


def _config(**updates: Any) -> KnowledgeCuratorConfig:
    values = {
        "candidate_generator_identity": "test.generator.v1",
        "evaluator_identity": "test.evaluator.v1",
        "namespace": "project:cayu",
        "labels": {"project": "cayu"},
        **updates,
    }
    return KnowledgeCuratorConfig(**values)


def _queue_config(**updates: Any) -> KnowledgeEnrichmentQueueConfig:
    values = {
        "retry_policy": TaskRetryPolicy(
            max_attempts=3,
            initial_backoff_seconds=0.0,
            backoff_multiplier=1.0,
            max_backoff_seconds=0.0,
        ),
        **updates,
    }
    return KnowledgeEnrichmentQueueConfig(**values)


class _Generator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
        self.calls += 1
        return [
            LearningCandidate(
                proposal_key="release-check",
                text="Run release checks before publishing.",
                signal_ids=tuple(signal.id for signal in batch.signals),
                kind="procedure",
            )
        ]


class _BlockingGenerator(_Generator):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
        self.started.set()
        await self.release.wait()
        return await super().generate_candidates(batch)


class _Evaluator:
    def __init__(self) -> None:
        self.calls = 0

    async def evaluate_candidate(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> LearningDecision:
        self.calls += 1
        assert candidate.signal_ids == tuple(signal.id for signal in signals)
        return LearningDecision(
            verdict=LearningVerdict.ACCEPTED,
            code="supported",
        )


def _curator(
    *,
    generator: _Generator | None = None,
    evaluator: _Evaluator | None = None,
    config: KnowledgeCuratorConfig | None = None,
    knowledge_store: InMemoryKnowledgeStore | None = None,
) -> tuple[KnowledgeCurator, _Generator, _Evaluator]:
    generator = generator or _Generator()
    evaluator = evaluator or _Evaluator()
    curator = KnowledgeCurator(
        knowledge_store or InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
        candidate_generator=generator,
        evaluator=evaluator,
        config=config or _config(),
        access_scope=_ACCESS_SCOPE,
    )
    return curator, generator, evaluator


def _request(
    queue: KnowledgeEnrichmentQueue,
    *,
    operation_id: str = "enrich-release-1",
    batch: LearningBatch | None = None,
    trigger: KnowledgeEnrichmentTrigger | None = None,
    feedback_authorization: KnowledgeEnrichmentFeedbackAuthorization | None = None,
) -> KnowledgeEnrichmentRequest:
    return KnowledgeEnrichmentRequest(
        operation_id=operation_id,
        batch=batch or _batch(),
        trigger=trigger
        or KnowledgeEnrichmentTrigger(
            kind="build_completed",
            source_type="build",
            source_id="build-1",
            source_revision="commit-1",
            occurred_at=_NOW,
        ),
        profile=queue.profile,
        submitted_at=_NOW,
        execution_source=TaskExecutionSource.SDK_TASK,
        feedback_authorization=feedback_authorization,
    )


def _task_store(kind: str, path: Path):
    if kind == "memory":
        return InMemoryTaskStore()
    return SQLiteTaskStore(path)


@pytest.mark.parametrize(
    ("capability", "message"),
    [
        ("supports_delayed_availability", "delayed task availability"),
        ("supports_idempotent_terminalization", "idempotent task terminalization"),
    ],
)
def test_enrichment_queue_requires_every_task_store_capability(
    capability: str,
    message: str,
) -> None:
    store = InMemoryTaskStore()
    setattr(store, capability, False)

    with pytest.raises(ValueError, match=message):
        KnowledgeEnrichmentQueue(
            store,
            curator_config=_config(),
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_enrichment_submission_is_exactly_idempotent(kind: str, tmp_path: Path) -> None:
    async def run() -> None:
        store = _task_store(kind, tmp_path / "tasks.sqlite")
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=_config(),
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        request = _request(queue)
        first, replay = await asyncio.gather(
            queue.submit(request),
            queue.submit(request),
        )

        assert first == replay
        assert replay.status is KnowledgeEnrichmentJobStatus.PENDING
        assert len(await store.list_tasks()) == 1
        assert await queue.submit(request) == first

        changed = request.model_copy(
            update={"batch": _batch(_signal(summary="Different evidence."))}
        )
        with pytest.raises(KnowledgeEnrichmentConflict):
            await queue.submit(changed)
        if isinstance(store, SQLiteTaskStore):
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_worker_completes_and_replay_does_not_rerun_curator(kind: str, tmp_path: Path) -> None:
    async def run() -> None:
        store = _task_store(kind, tmp_path / "tasks.sqlite")
        curator, generator, evaluator = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        await queue.submit(request)

        completed = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="worker-1",
            lease_seconds=30,
        )

        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert completed.result is not None
        assert completed.result.curation.batch_id == request.batch.id
        assert completed.result.curation.candidates[0].entry_id is not None
        assert generator.calls == 1
        assert evaluator.calls == 1

        replay = await queue.submit(request)
        assert replay == completed
        assert generator.calls == 1
        assert evaluator.calls == 1
        assert (
            await KnowledgeEnrichmentWorker(queue, curator).process_next(
                worker_id="worker-2",
                lease_seconds=30,
            )
            is None
        )
        if isinstance(store, SQLiteTaskStore):
            await store.close()

    asyncio.run(run())


def test_sqlite_job_survives_a_fresh_queue_process(tmp_path: Path) -> None:
    async def run() -> None:
        path = tmp_path / "tasks.sqlite"
        config = _config()
        producer_store = SQLiteTaskStore(path)
        producer = KnowledgeEnrichmentQueue(
            producer_store,
            curator_config=config,
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        request = _request(producer)
        submitted = await producer.submit(request)
        await producer_store.close()

        worker_store = SQLiteTaskStore(path)
        curator, _, _ = _curator(config=config)
        worker_queue = KnowledgeEnrichmentQueue(
            worker_store,
            curator_config=config,
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        completed = await KnowledgeEnrichmentWorker(worker_queue, curator).process_next(
            worker_id="fresh-worker",
            lease_seconds=30,
        )
        assert completed is not None
        assert completed.id == submitted.id
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        await worker_store.close()

        reader_store = SQLiteTaskStore(path)
        reader = KnowledgeEnrichmentQueue(
            reader_store,
            curator_config=config,
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        reloaded = await reader.load(request.operation_id)
        assert reloaded == completed
        await reader_store.close()

    asyncio.run(run())


def test_worker_cancellation_keeps_authority_until_curation_is_settled() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        generator = _BlockingGenerator()
        curator, _, _ = _curator(generator=generator)
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        await queue.submit(request)
        processing = asyncio.create_task(
            KnowledgeEnrichmentWorker(queue, curator).process_next(
                worker_id="worker-1",
                lease_seconds=30,
            )
        )
        await generator.started.wait()
        processing.cancel()
        await asyncio.sleep(0)
        assert processing.done() is False

        in_flight = await queue.load(request.operation_id)
        task = await store.load_task(
            in_flight.current_task_id if in_flight is not None else "missing"
        )
        assert in_flight is not None
        assert in_flight.status is KnowledgeEnrichmentJobStatus.PROCESSING
        assert task is not None
        assert task.status is TaskStatus.CLAIMED

        generator.release.set()
        with pytest.raises(asyncio.CancelledError):
            await processing

        job = await queue.load(request.operation_id)
        assert job is not None
        assert job.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert generator.calls == 1

    asyncio.run(run())


def test_operator_cancellation_wins_the_terminal_settlement_race() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        generator = _BlockingGenerator()
        curator, _, _ = _curator(generator=generator)
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        submitted = await queue.submit(request)
        processing = asyncio.create_task(
            KnowledgeEnrichmentWorker(queue, curator).process_next(
                worker_id="worker-1",
                lease_seconds=30,
            )
        )
        await generator.started.wait()
        cancellation_requested = await store.cancel_task(
            submitted.current_task_id,
            {"code": "operator_requested"},
        )
        assert cancellation_requested.status is TaskStatus.CLAIMED

        generator.release.set()
        cancelled = await processing
        assert cancelled is not None
        assert cancelled.status is KnowledgeEnrichmentJobStatus.CANCELLED
        assert cancelled.failure is not None
        assert cancelled.failure.category is KnowledgeEnrichmentFailureCategory.CANCELLED

    asyncio.run(run())


def test_retryable_worker_failure_creates_a_fenced_successor() -> None:
    class FlakyCurator(KnowledgeCurator):
        calls = 0

        async def _commit_prepared_curation(
            self,
            prepared,
            *,
            replay_stable=False,
            validate_publications=True,
        ):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("private database address")
            return await super()._commit_prepared_curation(
                prepared,
                replay_stable=replay_stable,
                validate_publications=validate_publications,
            )

    async def run() -> None:
        store = InMemoryTaskStore()
        base, generator, evaluator = _curator()
        curator = FlakyCurator(
            InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            candidate_generator=generator,
            evaluator=evaluator,
            config=base.config,
            access_scope=base.access_scope,
        )
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        await queue.submit(request)
        worker = KnowledgeEnrichmentWorker(queue, curator)

        scheduled = await worker.process_next(worker_id="worker-1", lease_seconds=30)
        assert scheduled is not None
        assert scheduled.status is KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
        assert len(scheduled.attempts) == 2
        assert scheduled.attempts[0].failure is not None
        assert (
            scheduled.attempts[0].failure.category
            is KnowledgeEnrichmentFailureCategory.DEPENDENCY_UNAVAILABLE
        )
        assert "private database address" not in scheduled.model_dump_json()

        completed = await worker.process_next(worker_id="worker-2", lease_seconds=30)
        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert len(completed.attempts) == 2
        assert curator.calls == 2
        assert generator.calls == 1
        assert evaluator.calls == 1

    asyncio.run(run())


def test_feedback_input_requires_explicit_policy_and_independent_batch_source() -> None:
    store = InMemoryTaskStore()
    queue = KnowledgeEnrichmentQueue(
        store,
        curator_config=_config(),
        access_scope=_ACCESS_SCOPE,
        config=_queue_config(),
    )
    trigger = KnowledgeEnrichmentTrigger.completed_interaction(
        session_id="session-1",
        interaction_id="interaction-1",
        terminal_event_id="event-terminal-1",
        occurred_at=_NOW,
        includes_recalled_material=True,
    )
    with pytest.raises(ValidationError, match="feedback authorization"):
        _request(queue, trigger=trigger)

    source_fingerprint = _batch().signals[0].source_references[0].fingerprint
    authorization = KnowledgeEnrichmentFeedbackAuthorization(
        policy_identity="acme.feedback-policy",
        policy_version="1",
        independent_source_fingerprints=(source_fingerprint,),
    )
    request = _request(
        queue,
        trigger=trigger,
        feedback_authorization=authorization,
    )
    assert request.trigger.metadata == {"session_id": "session-1"}

    unrelated = authorization.model_copy(update={"independent_source_fingerprints": ("0" * 64,)})
    with pytest.raises(ValidationError, match="source references"):
        _request(
            queue,
            trigger=trigger,
            feedback_authorization=unrelated,
        )


def test_queue_rejects_oversized_batch_before_task_creation() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        config = _config(max_batch_bytes=600, max_signal_bytes=600)
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=config,
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        with pytest.raises(ValueError, match="configured byte limit"):
            await queue.submit(
                _request(
                    queue,
                    batch=_batch(_signal(summary="x" * 500)),
                )
            )
        assert await store.list_tasks() == []

    asyncio.run(run())


def test_worker_rejects_a_different_curator_profile() -> None:
    store = InMemoryTaskStore()
    queue = KnowledgeEnrichmentQueue(
        store,
        curator_config=_config(),
        access_scope=_ACCESS_SCOPE,
        config=_queue_config(),
    )
    other, _, _ = _curator(config=_config(evaluator_identity="other.evaluator"))
    with pytest.raises(KnowledgeEnrichmentConflict, match="profile"):
        KnowledgeEnrichmentWorker(queue, other)


def test_queue_configuration_has_an_isolated_execution_domain() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        producer_curator, _, _ = _curator()
        producer = KnowledgeEnrichmentQueue(
            store,
            curator_config=producer_curator.config,
            access_scope=producer_curator.access_scope,
            config=_queue_config(max_reclaims_per_poll=100),
        )
        request = _request(producer)
        await producer.submit(request)

        changed_curator, _, _ = _curator()
        changed = KnowledgeEnrichmentQueue(
            store,
            curator_config=changed_curator.config,
            access_scope=changed_curator.access_scope,
            config=_queue_config(max_reclaims_per_poll=99),
        )
        assert changed.task_type != producer.task_type
        assert await changed.load(request.operation_id) is None
        changed_job = await changed.submit(_request(changed))
        producer_job = await producer.load(request.operation_id)
        assert producer_job is not None
        assert changed_job.id != producer_job.id

        completed = await KnowledgeEnrichmentWorker(producer, producer_curator).process_next(
            worker_id="producer-worker",
            lease_seconds=30,
        )
        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        isolated = await changed.load(request.operation_id)
        assert isolated is not None
        assert isolated.status is KnowledgeEnrichmentJobStatus.PENDING

        changed_completed = await KnowledgeEnrichmentWorker(
            changed,
            changed_curator,
        ).process_next(worker_id="changed-worker", lease_seconds=30)
        assert changed_completed is not None
        assert changed_completed.status is KnowledgeEnrichmentJobStatus.COMPLETED

    asyncio.run(run())


def test_curator_profiles_have_isolated_worker_routes_in_one_task_store() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        first_curator, _, _ = _curator()
        second_curator, _, _ = _curator(config=_config(evaluator_identity="other.evaluator"))
        first = KnowledgeEnrichmentQueue(
            store,
            curator_config=first_curator.config,
            access_scope=first_curator.access_scope,
            config=_queue_config(),
        )
        second = KnowledgeEnrichmentQueue(
            store,
            curator_config=second_curator.config,
            access_scope=second_curator.access_scope,
            config=_queue_config(),
        )
        await first.submit(_request(first, operation_id="shared-operation"))
        await second.submit(_request(second, operation_id="shared-operation"))

        assert first.task_type != second.task_type
        first_completed = await KnowledgeEnrichmentWorker(first, first_curator).process_next(
            worker_id="first-profile-worker",
            lease_seconds=30,
        )
        assert first_completed is not None
        assert first_completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        second_pending = await second.load("shared-operation")
        assert second_pending is not None
        assert second_pending.status is KnowledgeEnrichmentJobStatus.PENDING

    asyncio.run(run())


def test_worker_fails_closed_on_a_curator_result_outside_the_durable_batch() -> None:
    class NoCandidateGenerator:
        async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
            return []

    class WrongScopeCurator(KnowledgeCurator):
        async def _commit_prepared_curation(
            self,
            prepared,
            *,
            replay_stable=False,
            validate_publications=True,
        ):
            result = await super()._commit_prepared_curation(
                prepared,
                replay_stable=replay_stable,
                validate_publications=validate_publications,
            )
            return result.model_copy(update={"scope": "project:other"})

    async def run() -> None:
        store = InMemoryTaskStore()
        config = _config()
        curator = WrongScopeCurator(
            InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            candidate_generator=NoCandidateGenerator(),
            evaluator=_Evaluator(),
            config=config,
            access_scope=_ACCESS_SCOPE,
        )
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=config,
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        request = _request(queue, operation_id="wrong-result-scope")
        await queue.submit(request)

        failed = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="worker-1",
            lease_seconds=30,
        )

        assert failed is not None
        assert failed.status is KnowledgeEnrichmentJobStatus.FAILED
        assert failed.failure is not None
        assert failed.failure.code == "job_contract_conflict"
        assert failed.result is None

    asyncio.run(run())


def test_retry_exhaustion_is_terminal_and_preserves_attempt_diagnostics() -> None:
    class UnavailableCurator(KnowledgeCurator):
        calls = 0

        async def _commit_prepared_curation(
            self,
            prepared,
            *,
            replay_stable=False,
            validate_publications=True,
        ):
            self.calls += 1
            raise ConnectionError("private dependency address")

    async def run() -> None:
        store = InMemoryTaskStore()
        base, generator, evaluator = _curator()
        curator = UnavailableCurator(
            InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            candidate_generator=generator,
            evaluator=evaluator,
            config=base.config,
            access_scope=base.access_scope,
        )
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    initial_backoff_seconds=0.0,
                    backoff_multiplier=1.0,
                    max_backoff_seconds=0.0,
                )
            ),
        )
        request = _request(queue)
        await queue.submit(request)
        worker = KnowledgeEnrichmentWorker(queue, curator)

        first = await worker.process_next(worker_id="worker-1", lease_seconds=30)
        assert first is not None
        assert first.status is KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
        exhausted = await worker.process_next(worker_id="worker-2", lease_seconds=30)

        assert exhausted is not None
        assert exhausted.status is KnowledgeEnrichmentJobStatus.FAILED
        assert len(exhausted.attempts) == 2
        assert exhausted.attempts[-1].failure is not None
        assert exhausted.attempts[-1].failure.retryable is True
        assert exhausted.failure is not None
        assert exhausted.failure.retryable is False
        assert exhausted.failure.annotation == {"terminal_reason": "attempts_exhausted"}
        assert "private dependency address" not in exhausted.model_dump_json()
        assert curator.calls == 2

    asyncio.run(run())


@pytest.mark.parametrize("classifier_kind", ["raises", "async"])
def test_invalid_failure_classifier_fails_closed(classifier_kind: str) -> None:
    class BrokenCurator(KnowledgeCurator):
        async def _commit_prepared_curation(
            self,
            prepared,
            *,
            replay_stable=False,
            validate_publications=True,
        ):
            raise RuntimeError("private component failure")

    def raising_classifier(error: Exception) -> KnowledgeEnrichmentFailureDecision:
        raise RuntimeError("private classifier failure")

    async def async_classifier(error: Exception) -> KnowledgeEnrichmentFailureDecision:
        return KnowledgeEnrichmentFailureDecision(
            code="should_not_run",
            category=KnowledgeEnrichmentFailureCategory.INTERNAL_ERROR,
        )

    async def run() -> None:
        store = InMemoryTaskStore()
        base, generator, evaluator = _curator()
        curator = BrokenCurator(
            InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE),
            candidate_generator=generator,
            evaluator=evaluator,
            config=base.config,
            access_scope=base.access_scope,
        )
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        await queue.submit(request)
        classifier = raising_classifier if classifier_kind == "raises" else async_classifier
        failed = await KnowledgeEnrichmentWorker(
            queue,
            curator,
            exception_classifier=classifier,
        ).process_next(worker_id="worker-1", lease_seconds=30)

        assert failed is not None
        assert failed.status is KnowledgeEnrichmentJobStatus.FAILED
        assert failed.failure is not None
        assert failed.failure.code == (
            "failure_classifier_failed"
            if classifier_kind == "raises"
            else "failure_classifier_invalid"
        )
        assert "private" not in failed.model_dump_json()

    asyncio.run(run())


def test_unsafe_result_limit_is_rejected_before_submission() -> None:
    with pytest.raises(ValidationError, match="max_result_bytes"):
        _queue_config(max_result_bytes=100)


def test_result_fanout_is_rejected_before_enqueue_or_publication() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        curator_config = _config(
            max_candidates=250,
            max_evaluator_notes_bytes=64 * 1024,
        )
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator_config,
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )

        with pytest.raises(ValueError, match="curator bounds can exceed max_result_bytes"):
            await queue.submit(_request(queue, operation_id="unsafe-result-fanout"))

        assert await queue.load("unsafe-result-fanout") is None

    asyncio.run(run())


def test_result_preflight_allows_safe_medium_signal_fanout() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        curator_config = _config(max_signals=100)
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator_config,
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        batch = LearningBatch(
            id="medium-fanout",
            signals=tuple(_signal(f"signal-{index}") for index in range(100)),
        )

        submitted = await queue.submit(
            _request(queue, operation_id="safe-result-fanout", batch=batch)
        )

        assert submitted.status is KnowledgeEnrichmentJobStatus.PENDING
        assert len(await store.list_tasks()) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "retry_update",
    [
        {"max_elapsed_seconds": 60.0},
        {"max_total_tokens": 1_000},
    ],
)
def test_queue_rejects_retry_authority_it_cannot_account(retry_update: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="attempt/backoff bounds only"):
        _queue_config(
            retry_policy=TaskRetryPolicy(
                max_attempts=2,
                initial_backoff_seconds=0.0,
                **retry_update,
            )
        )


def test_heartbeat_authority_loss_cannot_publish_a_terminal_job_result() -> None:
    class LosingTaskStore(InMemoryTaskStore):
        async def heartbeat(
            self,
            task_id: str,
            worker_id: str,
            *,
            lease_expires_at: datetime,
            handoff_id: str | None = None,
            extend_seconds: int = 300,
        ):
            raise TaskClaimLost("lease transferred")

    async def run() -> None:
        store = LosingTaskStore()
        generator = _BlockingGenerator()
        curator, _, _ = _curator(generator=generator)
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        await queue.submit(request)
        with pytest.raises(TaskClaimLost, match="heartbeat lost"):
            await KnowledgeEnrichmentWorker(queue, curator).process_next(
                worker_id="worker-1",
                lease_seconds=1,
            )

        job = await queue.load(request.operation_id)
        assert job is not None
        assert job.status is KnowledgeEnrichmentJobStatus.PROCESSING
        assert job.result is None

    asyncio.run(run())


def test_settlement_store_failure_is_not_reclassified_as_a_curation_failure() -> None:
    class UnavailableSettlementStore(InMemoryTaskStore):
        settlement_calls = 0
        settlement_available = False

        async def settle_task_retry_attempt(self, request):
            self.settlement_calls += 1
            if not self.settlement_available:
                raise ConnectionError("task store unavailable")
            return await super().settle_task_retry_attempt(request)

    async def run() -> None:
        store = UnavailableSettlementStore()
        curator, generator, evaluator = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(
                terminalization_retry_policy=TaskTerminalizationRetryPolicy(
                    max_attempts=2,
                    attempt_timeout_seconds=1.0,
                    initial_backoff_seconds=0.0,
                    backoff_multiplier=1.0,
                    max_backoff_seconds=0.0,
                )
            ),
        )
        request = _request(queue)
        submitted = await queue.submit(request)
        with pytest.raises(TaskTerminalizationUncertain):
            await KnowledgeEnrichmentWorker(queue, curator).process_next(
                worker_id="worker-1",
                lease_seconds=1,
            )

        task = await store.load_task(submitted.current_task_id)
        assert task is not None
        assert task.status is TaskStatus.CLAIMED
        assert task.error is None
        assert store.settlement_calls == 2
        assert generator.calls == 1
        assert evaluator.calls == 1
        preparation = next(
            task
            for task in await store.list_tasks()
            if task.title == "Knowledge enrichment preparation"
        )
        assert preparation.status is TaskStatus.COMPLETED
        assert preparation.result is not None
        durable_curation = preparation.result["curation"]

        store.settlement_available = True
        await asyncio.sleep(1.05)
        completed = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="worker-2",
            lease_seconds=30,
        )
        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert completed.result is not None
        assert completed.result.curation.model_dump(mode="json") == durable_curation
        assert generator.calls == 1
        assert evaluator.calls == 1

    asyncio.run(run())


def test_process_loss_after_publication_reuses_the_durable_semantic_preparation() -> None:
    class RecoveringSettlementStore(InMemoryTaskStore):
        preparation_completion_available = False
        settlement_available = False

        async def complete_task(self, task_id, result, *, worker_id=None, handoff_id=None):
            if not self.preparation_completion_available:
                raise ConnectionError("preparation completion unavailable")
            return await super().complete_task(
                task_id,
                result,
                worker_id=worker_id,
                handoff_id=handoff_id,
            )

        async def settle_task_retry_attempt(self, request):
            if not self.settlement_available:
                raise ConnectionError("task settlement unavailable")
            return await super().settle_task_retry_attempt(request)

    async def run() -> None:
        tasks = RecoveringSettlementStore()
        knowledge = InMemoryKnowledgeStore(access_scope=_ACCESS_SCOPE)
        generator = _Generator()
        evaluator = _Evaluator()
        first_curator, _, _ = _curator(
            generator=generator,
            evaluator=evaluator,
            knowledge_store=knowledge,
        )
        queue = KnowledgeEnrichmentQueue(
            tasks,
            curator_config=first_curator.config,
            access_scope=first_curator.access_scope,
            config=_queue_config(
                terminalization_retry_policy=TaskTerminalizationRetryPolicy(
                    max_attempts=1,
                    attempt_timeout_seconds=1.0,
                    initial_backoff_seconds=0.0,
                    backoff_multiplier=1.0,
                    max_backoff_seconds=0.0,
                )
            ),
        )
        request = _request(queue, operation_id="post-publication-process-loss")
        await queue.submit(request)

        with pytest.raises(TaskTerminalizationUncertain):
            await KnowledgeEnrichmentWorker(queue, first_curator).process_next(
                worker_id="lost-worker",
                lease_seconds=1,
            )
        assert generator.calls == 1
        assert evaluator.calls == 1
        preparation = next(
            task
            for task in await tasks.list_tasks()
            if task.title == "Knowledge enrichment preparation"
        )
        assert preparation.status is TaskStatus.PENDING
        assert preparation.result is None
        await first_curator.aclose()

        tasks.preparation_completion_available = True
        tasks.settlement_available = True
        await asyncio.sleep(1.05)
        recovered_curator, _, _ = _curator(
            generator=generator,
            evaluator=evaluator,
            knowledge_store=knowledge,
        )
        recovered = await KnowledgeEnrichmentWorker(queue, recovered_curator).process_next(
            worker_id="recovery-worker",
            lease_seconds=30,
        )

        assert recovered is not None
        assert recovered.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert recovered.result is not None
        assert generator.calls == 1
        assert evaluator.calls == 1
        candidate = recovered.result.curation.candidates[0]
        assert candidate.entry_id is not None
        assert await knowledge.get_entry(candidate.entry_id) is not None
        await recovered_curator.aclose()

    asyncio.run(run())


def test_semantic_dispatch_acknowledgement_loss_reconciles_without_rerun() -> None:
    class DispatchAcknowledgementLossStore(InMemoryTaskStore):
        lose_acknowledgement = True

        async def create_task(self, request):
            created = await super().create_task(request)
            if (
                request.title == "Knowledge enrichment semantic dispatch"
                and self.lose_acknowledgement
            ):
                self.lose_acknowledgement = False
                raise ConnectionError("semantic dispatch acknowledgement lost")
            return created

    async def run() -> None:
        store = DispatchAcknowledgementLossStore()
        curator, generator, evaluator = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        await queue.submit(_request(queue, operation_id="dispatch-ack-loss"))

        completed = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="worker-1",
            lease_seconds=30,
        )

        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert generator.calls == 1
        assert evaluator.calls == 1

    asyncio.run(run())


def test_dispatch_write_failure_before_commit_retries_before_semantic_work() -> None:
    class DispatchWriteFailureStore(InMemoryTaskStore):
        fail_dispatch = True

        async def create_task(self, request):
            if request.title == "Knowledge enrichment semantic dispatch" and self.fail_dispatch:
                self.fail_dispatch = False
                raise ConnectionError("semantic dispatch unavailable")
            return await super().create_task(request)

    async def run() -> None:
        store = DispatchWriteFailureStore()
        curator, generator, evaluator = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        await queue.submit(_request(queue, operation_id="dispatch-write-failure"))
        worker = KnowledgeEnrichmentWorker(queue, curator)

        scheduled = await worker.process_next(worker_id="worker-1", lease_seconds=30)
        assert scheduled is not None
        assert scheduled.status is KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
        assert generator.calls == 0
        assert evaluator.calls == 0

        completed = await worker.process_next(worker_id="worker-2", lease_seconds=30)
        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert generator.calls == 1
        assert evaluator.calls == 1

    asyncio.run(run())


def test_preparation_write_failure_never_redispatches_semantic_work() -> None:
    class PreparationWriteFailureStore(InMemoryTaskStore):
        fail_preparation = True

        async def create_task(self, request):
            if request.title == "Knowledge enrichment preparation" and self.fail_preparation:
                self.fail_preparation = False
                raise ConnectionError("preparation unavailable")
            return await super().create_task(request)

    async def run() -> None:
        store = PreparationWriteFailureStore()
        curator, generator, evaluator = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(
                retry_policy=TaskRetryPolicy(
                    max_attempts=2,
                    initial_backoff_seconds=0.0,
                    backoff_multiplier=1.0,
                    max_backoff_seconds=0.0,
                )
            ),
        )
        await queue.submit(_request(queue, operation_id="preparation-write-failure"))
        worker = KnowledgeEnrichmentWorker(queue, curator)

        scheduled = await worker.process_next(worker_id="worker-1", lease_seconds=30)
        assert scheduled is not None
        assert scheduled.status is KnowledgeEnrichmentJobStatus.RETRY_SCHEDULED
        assert scheduled.attempts[0].failure is not None
        assert (
            scheduled.attempts[0].failure.category
            is KnowledgeEnrichmentFailureCategory.SEMANTIC_OUTCOME_UNKNOWN
        )
        assert generator.calls == 1
        assert evaluator.calls == 1

        failed = await worker.process_next(worker_id="worker-2", lease_seconds=30)
        assert failed is not None
        assert failed.status is KnowledgeEnrichmentJobStatus.FAILED
        assert failed.failure is not None
        assert failed.failure.code == "semantic_outcome_unknown"
        assert (
            failed.failure.category is KnowledgeEnrichmentFailureCategory.SEMANTIC_OUTCOME_UNKNOWN
        )
        assert failed.failure.retryable is False
        assert generator.calls == 1
        assert evaluator.calls == 1

    asyncio.run(run())


def test_worker_run_counts_and_terminalizes_a_malformed_claim() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        curator, _, _ = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        malformed = await store.create_task(
            TaskCreate(
                task_id="malformed-enrichment",
                type=queue.task_type,
                input={"knowledge_enrichment": {"contract": "invalid"}},
                retry_policy=queue.config.retry_policy,
            )
        )
        request = _request(queue, operation_id="valid-after-malformed")
        valid = await queue.submit(request)

        handled = await KnowledgeEnrichmentWorker(queue, curator).run(
            worker_id="worker-1",
            lease_seconds=30,
            poll_interval_s=0.01,
            max_jobs=1,
        )

        assert handled == 1
        malformed_after = await store.load_task(malformed.id)
        valid_after = await store.load_task(valid.current_task_id)
        assert malformed_after is not None
        assert malformed_after.status is TaskStatus.FAILED
        assert valid_after is not None
        assert valid_after.status is TaskStatus.PENDING

    asyncio.run(run())


def test_process_next_never_reports_a_rejected_claim_as_idle() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        curator, _, _ = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        malformed = await store.create_task(
            TaskCreate(
                task_id="malformed-process-next-enrichment",
                type=queue.task_type,
                input={"knowledge_enrichment": {"contract": "invalid"}},
            )
        )
        assert malformed.retry_series is None

        with pytest.raises(KnowledgeEnrichmentJobRejected, match="durably rejected"):
            await KnowledgeEnrichmentWorker(queue, curator).process_next(
                worker_id="worker-1",
                lease_seconds=30,
            )

        rejected = await store.load_task(malformed.id)
        assert rejected is not None
        assert rejected.status is TaskStatus.FAILED
        assert (
            await KnowledgeEnrichmentWorker(queue, curator).process_next(
                worker_id="worker-2",
                lease_seconds=30,
            )
            is None
        )

    asyncio.run(run())


@pytest.mark.parametrize("location", ["input", "envelope"])
def test_queue_rejects_extra_fields_in_the_durable_request_envelope(location: str) -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=_config(),
            access_scope=_ACCESS_SCOPE,
            config=_queue_config(),
        )
        submitted = await queue.submit(_request(queue, operation_id=f"extra-{location}"))
        task = await store.load_task(submitted.current_task_id)
        assert task is not None
        tampered_input = dict(task.input)
        if location == "input":
            tampered_input["unexpected"] = True
        else:
            envelope = dict(tampered_input["knowledge_enrichment"])
            envelope["unexpected"] = True
            tampered_input["knowledge_enrichment"] = envelope

        with pytest.raises(KnowledgeEnrichmentConflict, match="envelope"):
            queue._request_from_task(task.model_copy(update={"input": tampered_input}))

    asyncio.run(run())


def test_queue_rejects_extra_fields_in_terminal_envelopes() -> None:
    async def run() -> None:
        store = InMemoryTaskStore()
        curator, _, _ = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue, operation_id="extra-terminal-envelope")
        await queue.submit(request)
        completed = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="worker-1",
            lease_seconds=30,
        )
        assert completed is not None
        task = await store.load_task(completed.current_task_id)
        assert task is not None
        assert task.result is not None
        tampered_result = {**task.result, "unexpected": True}

        with pytest.raises(KnowledgeEnrichmentConflict, match="valid result"):
            _result_from_task(
                task.model_copy(update={"result": tampered_result}),
                request=request,
                request_sha256=request.fingerprint,
                max_result_bytes=queue.config.max_result_bytes,
                curator_config=curator.config,
            )

        preparation = next(
            candidate
            for candidate in await store.list_tasks()
            if candidate.id != completed.current_task_id
        )
        with pytest.raises(KnowledgeEnrichmentConflict, match="preparation input"):
            queue._preparation_from_task(
                preparation.model_copy(update={"input": {**preparation.input, "unexpected": True}}),
                request=request,
                request_sha256=request.fingerprint,
            )

        failure = {
            "contract": "cayu.knowledge-enrichment-failure.v1",
            "failure": {
                "schema_version": 1,
                "request_sha256": request.fingerprint,
                "code": "invalid",
                "category": "invalid_job",
                "retryable": False,
                "annotation": {},
            },
            "unexpected": True,
        }
        with pytest.raises(KnowledgeEnrichmentConflict, match="failure envelope"):
            _parse_failure_payload(failure, request_sha256=request.fingerprint)
        malformed_failure = dict(failure)
        malformed_failure.pop("unexpected")
        malformed_failure["failure"] = "invalid"
        with pytest.raises(KnowledgeEnrichmentConflict, match="failure envelope"):
            _parse_failure_payload(
                malformed_failure,
                request_sha256=request.fingerprint,
            )

    asyncio.run(run())


def test_one_worker_instance_serializes_concurrent_curation_calls() -> None:
    class SerialGenerator(_Generator):
        def __init__(self) -> None:
            super().__init__()
            self.first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
            self.calls += 1
            if self.calls == 1:
                self.first_started.set()
                await self.release_first.wait()
            return [
                LearningCandidate(
                    proposal_key=f"proposal:{batch.id}",
                    text="Run the bounded release check.",
                    signal_ids=tuple(signal.id for signal in batch.signals),
                    kind="procedure",
                )
            ]

    async def run() -> None:
        store = InMemoryTaskStore()
        generator = SerialGenerator()
        curator, _, _ = _curator(generator=generator)
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        first_request = _request(queue, operation_id="serial-1")
        second_request = _request(
            queue,
            operation_id="serial-2",
            batch=LearningBatch(id="batch-2", signals=(_signal("signal-2"),)),
        )
        await queue.submit(first_request)
        await queue.submit(second_request)
        worker = KnowledgeEnrichmentWorker(queue, curator)

        first = asyncio.create_task(worker.process_next(worker_id="worker-1", lease_seconds=30))
        await generator.first_started.wait()
        second = asyncio.create_task(worker.process_next(worker_id="worker-1", lease_seconds=30))
        await asyncio.sleep(0.05)
        assert generator.calls == 1
        assert second.done() is False

        generator.release_first.set()
        results = await asyncio.gather(first, second)
        assert all(
            result is not None and result.status is KnowledgeEnrichmentJobStatus.COMPLETED
            for result in results
        )
        assert generator.calls == 2

    asyncio.run(run())


def test_task_settlement_acknowledgement_loss_reconciles_without_rerun() -> None:
    class AcknowledgementLossTaskStore(InMemoryTaskStore):
        lose_acknowledgement = True

        async def settle_task_retry_attempt(self, request):
            result = await super().settle_task_retry_attempt(request)
            if self.lose_acknowledgement:
                self.lose_acknowledgement = False
                raise ConnectionError("private settlement acknowledgement failure")
            return result

    async def run() -> None:
        store = AcknowledgementLossTaskStore()
        curator, generator, evaluator = _curator()
        queue = KnowledgeEnrichmentQueue(
            store,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        await queue.submit(request)
        completed = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="worker-1",
            lease_seconds=30,
        )

        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert generator.calls == 1
        assert evaluator.calls == 1
        assert "private settlement" not in completed.model_dump_json()

    asyncio.run(run())


def test_knowledge_publication_acknowledgement_loss_commits_the_exact_job_result() -> None:
    class AcknowledgementLossKnowledgeStore(InMemoryKnowledgeStore):
        async def publish_entry_revision(
            self,
            entry,
            chunks,
            *,
            evidence=None,
            access_scope=None,
            operation_id,
            expected_revision=None,
            activation_authority=None,
        ):
            await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
                activation_authority=activation_authority,
            )
            raise ConnectionError("private publication acknowledgement failure")

    async def run() -> None:
        tasks = InMemoryTaskStore()
        knowledge = AcknowledgementLossKnowledgeStore(access_scope=_ACCESS_SCOPE)
        curator, generator, evaluator = _curator(knowledge_store=knowledge)
        queue = KnowledgeEnrichmentQueue(
            tasks,
            curator_config=curator.config,
            access_scope=curator.access_scope,
            config=_queue_config(),
        )
        request = _request(queue)
        await queue.submit(request)
        completed = await KnowledgeEnrichmentWorker(queue, curator).process_next(
            worker_id="worker-1",
            lease_seconds=30,
        )

        assert completed is not None
        assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
        assert completed.result is not None
        candidate = completed.result.curation.candidates[0]
        assert candidate.warning_code is None
        assert candidate.entry_id is not None
        assert await knowledge.get_entry(candidate.entry_id) is not None
        assert generator.calls == 1
        assert evaluator.calls == 1
        assert await queue.submit(request) == completed
        assert generator.calls == 1
        assert evaluator.calls == 1
        assert "private publication" not in completed.model_dump_json()

    asyncio.run(run())


def test_postgres_concurrent_submission_and_fresh_worker_store(postgres_dsn: str) -> None:
    async def run() -> None:
        await drop_cayu_tables(postgres_dsn)
        config = _config()
        queue_config = _queue_config()
        first_store = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        second_store = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.CREATE,
        )
        try:
            await first_store.list_tasks()
            await second_store.list_tasks()
            first_queue = KnowledgeEnrichmentQueue(
                first_store,
                curator_config=config,
                access_scope=_ACCESS_SCOPE,
                config=queue_config,
            )
            second_queue = KnowledgeEnrichmentQueue(
                second_store,
                curator_config=config,
                access_scope=_ACCESS_SCOPE,
                config=queue_config,
            )
            request = _request(
                first_queue,
                operation_id=f"postgres-enrichment-{uuid4()}",
            )
            first, second = await asyncio.gather(
                first_queue.submit(request),
                second_queue.submit(request),
            )
            assert first == second
            assert len(await first_store.list_tasks()) == 1
        finally:
            await first_store.close()
            await second_store.close()

        worker_store = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=2,
            schema_mode=SchemaMode.VALIDATE,
        )
        try:
            curator, generator, evaluator = _curator(config=config)
            worker_queue = KnowledgeEnrichmentQueue(
                worker_store,
                curator_config=config,
                access_scope=_ACCESS_SCOPE,
                config=queue_config,
            )
            completed = await KnowledgeEnrichmentWorker(
                worker_queue,
                curator,
            ).process_next(worker_id="postgres-worker", lease_seconds=30)
            assert completed is not None
            assert completed.status is KnowledgeEnrichmentJobStatus.COMPLETED
            assert generator.calls == 1
            assert evaluator.calls == 1
        finally:
            await worker_store.close()
            await drop_cayu_tables(postgres_dsn)

    asyncio.run(run())


@pytest.mark.process
def test_durable_knowledge_enrichment_example_uses_a_fresh_worker_process() -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, str(repository / "examples/durable_knowledge_enrichment.py")],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert "completed" in completed.stdout
    assert "Run database migrations before starting" in completed.stdout
