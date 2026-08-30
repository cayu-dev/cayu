from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from cayu import (
    DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID,
    AgentRecallCheckpoint,
    AgentRecallCheckpointMode,
    AgentRecallDelivery,
    AgentRecallDeliveryConflict,
    AgentRecallDeliveryEvidenceKind,
    AgentRecallDeliveryState,
    AgentRecallProcessingRequest,
    AgentRecallProcessingResult,
    AgentRecallProcessor,
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextStore,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    RecallSituation,
    WeightedReciprocalRankFusionConfig,
)
from cayu.recall import KNOWLEDGE_LEXICAL_CHANNEL, KNOWLEDGE_SEMANTIC_CHANNEL

_ACCESS_POLICY_SHA256 = "a" * 64
_OTHER_ACCESS_POLICY_SHA256 = "b" * 64
_STARTED_AT = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)
_DELIVERY_NAMESPACE = "project:delivery"


async def _create_delivery_knowledge(
    store: InMemoryKnowledgeStore,
    entry_id: str,
) -> None:
    text = f"checkpoint-aware delivery evidence {entry_id}"
    await store.create_entry(
        KnowledgeEntry(id=entry_id, namespace=_DELIVERY_NAMESPACE, text=text),
        [
            KnowledgeChunk(
                id=f"{entry_id}:chunk",
                entry_id=entry_id,
                text=text,
                chunk_index=0,
            )
        ],
        access_scope=KnowledgeAccessScope.for_namespace(_DELIVERY_NAMESPACE),
    )


def _delivery_processor(store: InMemoryKnowledgeStore) -> AgentRecallProcessor:
    return AgentRecallProcessor(
        store,
        fusion_config=WeightedReciprocalRankFusionConfig(
            configuration_version="delivery-conformance-v1",
            channel_weights={
                KNOWLEDGE_LEXICAL_CHANNEL: 1.0,
                KNOWLEDGE_SEMANTIC_CHANNEL: 1.0,
            },
            max_candidates_per_channel=20,
            fused_head_limit=20,
        ),
    )


async def _process_delivery(
    processor: AgentRecallProcessor,
    work_context: AgentWorkContext,
    *,
    operation_id: str,
    checkpoint_value: AgentRecallCheckpoint | None,
    agent_id: str = "agent:delivery",
    checkpoint_stream_id: str = DEFAULT_AGENT_RECALL_CHECKPOINT_STREAM_ID,
) -> AgentRecallProcessingResult:
    return await processor.process(
        AgentRecallProcessingRequest(
            agent_id=agent_id,
            work_context=work_context,
            situation=RecallSituation(
                query="checkpoint-aware delivery evidence",
                knowledge_access_scope=KnowledgeAccessScope.for_namespace(_DELIVERY_NAMESPACE),
                knowledge_namespace=_DELIVERY_NAMESPACE,
                current_time=_STARTED_AT + timedelta(minutes=40),
            ),
            checkpoint_stream_id=checkpoint_stream_id,
            checkpoint=checkpoint_value,
            processing_id=f"processing:{operation_id}",
            operation_id=operation_id,
            updated_by="runtime:delivery-conformance",
            updated_at=_STARTED_AT + timedelta(minutes=41),
        )
    )


def _stageable_delivery(
    result: AgentRecallProcessingResult,
    *,
    delivery_id: str,
    expected_checkpoint_revision: int | None,
    staged_at: datetime,
) -> AgentRecallDelivery:
    return AgentRecallDelivery.from_processing_result(
        result,
        delivery_id=delivery_id,
        expected_checkpoint_revision=expected_checkpoint_revision,
        staged_by="runtime:delivery-conformance",
        staged_at=staged_at,
    )


async def recall_delivery(
    work_context: AgentWorkContext,
    *,
    delivery_id: str,
    operation_id: str,
    entry_ids: tuple[str, ...],
    checkpoint_value: AgentRecallCheckpoint | None = None,
    staged_at: datetime = _STARTED_AT + timedelta(minutes=45),
) -> AgentRecallDelivery:
    knowledge_store = InMemoryKnowledgeStore()
    for entry_id in entry_ids:
        await _create_delivery_knowledge(knowledge_store, entry_id)
    result = await _process_delivery(
        _delivery_processor(knowledge_store),
        work_context,
        operation_id=operation_id,
        checkpoint_value=checkpoint_value,
    )
    return _stageable_delivery(
        result,
        delivery_id=delivery_id,
        expected_checkpoint_revision=(
            None if checkpoint_value is None else checkpoint_value.revision
        ),
        staged_at=staged_at,
    )


def context(
    *,
    revision: int,
    operation_id: str,
    task_id: str = "task-memory-v51",
    goal: str = "Ship durable cross-agent freshness",
    published_at: datetime | None = None,
    entity_ids: tuple[str, ...] = ("feature:agent-work-context",),
) -> AgentWorkContext:
    return AgentWorkContext.create(
        task_id=task_id,
        goal=goal,
        revision=revision,
        operation_id=operation_id,
        published_by="application:memory-coordinator",
        published_at=published_at or (_STARTED_AT + timedelta(minutes=revision)),
        scope_ids=("repository:cayu",),
        workflow_id="workflow:memory-v51",
        workflow_phase="cross-agent-freshness",
        workflow_iteration=0,
        entity_ids=entity_ids,
        artifact_ids=("architecture:v5.1",),
        repository_paths=("src/cayu",),
        code_symbols=("RecallEngine",),
        planned_action_ids=("action:implement-work-context",),
    )


def checkpoint(
    work_context: AgentWorkContext,
    *,
    revision: int,
    operation_id: str,
    agent_id: str = "agent:primary",
    knowledge_namespace: str = "project:cayu",
    access_policy_sha256: str = _ACCESS_POLICY_SHA256,
    knowledge_sequence: int = 10,
    index_readiness_sequence: int = 7,
    knowledge_high_water_sequence: int | None = None,
    index_readiness_high_water_sequence: int | None = None,
    processing_mode: AgentRecallCheckpointMode = AgentRecallCheckpointMode.FULL_INDEX,
    processing_id: str | None = None,
) -> AgentRecallCheckpoint:
    return AgentRecallCheckpoint(
        agent_id=agent_id,
        task_id=work_context.task_id,
        knowledge_namespace=knowledge_namespace,
        access_policy_sha256=access_policy_sha256,
        revision=revision,
        work_context_revision=work_context.revision,
        work_context_sha256=work_context.content_sha256,
        knowledge_sequence=knowledge_sequence,
        index_readiness_sequence=index_readiness_sequence,
        knowledge_high_water_sequence=(
            knowledge_sequence
            if knowledge_high_water_sequence is None
            else knowledge_high_water_sequence
        ),
        index_readiness_high_water_sequence=(
            index_readiness_sequence
            if index_readiness_high_water_sequence is None
            else index_readiness_high_water_sequence
        ),
        processing_mode=processing_mode,
        processing_id=processing_id or f"processing:{operation_id}",
        operation_id=operation_id,
        updated_by="runtime:memory-coordinator",
        updated_at=_STARTED_AT + timedelta(minutes=20 + revision),
    )


async def assert_work_context_store_conformance(
    store: AgentWorkContextStore,
    *,
    advance_clock: Callable[[timedelta], None] | None = None,
) -> None:
    first = context(revision=1, operation_id="context:create")
    assert await store.load_work_context(first.task_id) is None
    assert await store.load_work_context_publication(first.operation_id) is None
    with pytest.raises(ValueError, match="expected_revision"):
        await store.publish_work_context(first, expected_revision=0)
    assert await store.load_work_context(first.task_id) is None

    created = await store.publish_work_context(first, expected_revision=None)
    assert created.changed is True
    assert created.context == first
    loaded_first = await store.load_work_context(first.task_id)
    assert loaded_first == first
    assert loaded_first is not first
    assert await store.load_work_context(first.task_id, revision=1) == first
    assert await store.load_work_context_publication(first.operation_id) == created

    replay = await store.publish_work_context(first, expected_revision=None)
    assert replay == created
    reused = context(
        revision=1,
        operation_id=first.operation_id,
        goal="Different request under the same operation",
    )
    with pytest.raises(AgentWorkContextConflict, match="publication_operation_reused"):
        await store.publish_work_context(reused, expected_revision=None)

    no_change = context(revision=2, operation_id="context:no-change")
    no_change_receipt = await store.publish_work_context(no_change, expected_revision=1)
    assert no_change_receipt.changed is False
    assert no_change_receipt.context == first
    assert no_change_receipt.operation_id == no_change.operation_id
    assert await store.load_work_context(first.task_id) == first
    assert await store.load_work_context(first.task_id, revision=2) is None
    assert await store.load_work_context_publication(no_change.operation_id) == no_change_receipt

    second = context(
        revision=2,
        operation_id="context:append",
        goal="Integrate durable work context with bounded recall",
    )
    appended = await store.publish_work_context(second, expected_revision=1)
    assert appended.changed is True
    assert appended.context == second
    assert await store.load_work_context(first.task_id) == second
    assert await store.load_work_context(first.task_id, revision=1) == first

    stale = context(revision=2, operation_id="context:stale", goal="Stale writer")
    with pytest.raises(AgentWorkContextConflict, match="stale_context_revision"):
        await store.publish_work_context(stale, expected_revision=1)
    assert await store.load_work_context(first.task_id) == second

    competitors = (
        context(revision=3, operation_id="context:concurrent-a", goal="Concurrent A"),
        context(revision=3, operation_id="context:concurrent-b", goal="Concurrent B"),
    )
    outcomes = await asyncio.gather(
        *(store.publish_work_context(candidate, expected_revision=2) for candidate in competitors),
        return_exceptions=True,
    )
    successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], AgentWorkContextConflict)
    current = await store.load_work_context(first.task_id)
    assert current is not None
    assert current.revision == 3
    assert current in competitors

    initial_checkpoint = checkpoint(
        current,
        revision=1,
        operation_id="checkpoint:initial",
    )
    with pytest.raises(ValueError, match="expected_revision"):
        await store.advance_recall_checkpoint(initial_checkpoint, expected_revision=0)
    assert await store.load_recall_checkpoint(initial_checkpoint.key()) is None
    stored_checkpoint = await store.advance_recall_checkpoint(
        initial_checkpoint,
        expected_revision=None,
    )
    assert stored_checkpoint == initial_checkpoint
    assert await store.load_recall_checkpoint(initial_checkpoint.key()) == initial_checkpoint
    assert (
        await store.load_recall_checkpoint(initial_checkpoint.key(), revision=1)
        == initial_checkpoint
    )
    assert (
        await store.advance_recall_checkpoint(initial_checkpoint, expected_revision=None)
        == initial_checkpoint
    )
    for invalid_expected_revision in (True, 1.0):
        with pytest.raises(ValueError, match="expected_revision"):
            await store.advance_recall_checkpoint(
                initial_checkpoint,
                expected_revision=cast("Any", invalid_expected_revision),
            )

    reused_checkpoint = initial_checkpoint.model_copy(
        update={"knowledge_sequence": 11, "knowledge_high_water_sequence": 11}
    )
    with pytest.raises(AgentWorkContextConflict, match="checkpoint_operation_reused"):
        await store.advance_recall_checkpoint(reused_checkpoint, expected_revision=None)

    second_agent = checkpoint(
        current,
        revision=1,
        operation_id="checkpoint:second-agent",
        agent_id="agent:reviewer",
        knowledge_sequence=3,
        index_readiness_sequence=2,
    )
    await store.advance_recall_checkpoint(second_agent, expected_revision=None)
    assert await store.load_recall_checkpoint(second_agent.key()) == second_agent
    assert await store.load_recall_checkpoint(initial_checkpoint.key()) == initial_checkpoint

    delta = checkpoint(
        current,
        revision=2,
        operation_id="checkpoint:delta",
        knowledge_sequence=12,
        index_readiness_sequence=9,
        processing_mode=AgentRecallCheckpointMode.DELTA,
    ).model_copy(update={"updated_at": initial_checkpoint.updated_at - timedelta(minutes=5)})
    assert delta.updated_at < initial_checkpoint.updated_at
    await store.advance_recall_checkpoint(delta, expected_revision=1)
    assert await store.load_recall_checkpoint(delta.key()) == delta
    assert await store.load_recall_checkpoint(delta.key(), revision=1) == initial_checkpoint

    regressive = checkpoint(
        current,
        revision=3,
        operation_id="checkpoint:regressive",
        knowledge_sequence=11,
        index_readiness_sequence=9,
        processing_mode=AgentRecallCheckpointMode.DELTA,
    )
    with pytest.raises(AgentWorkContextConflict, match="knowledge_sequence_regression"):
        await store.advance_recall_checkpoint(regressive, expected_revision=2)
    assert await store.load_recall_checkpoint(delta.key()) == delta

    fourth = context(
        revision=4,
        operation_id="context:task-switch",
        goal="Review a different memory task",
        entity_ids=("task:memory-review",),
    )
    await store.publish_work_context(fourth, expected_revision=3)
    assert await store.publish_work_context(first, expected_revision=None) == created
    assert await store.load_work_context(first.task_id) == fourth
    stale_context_full = checkpoint(
        current,
        revision=3,
        operation_id="checkpoint:stale-context-full",
        knowledge_sequence=13,
        index_readiness_sequence=10,
        processing_mode=AgentRecallCheckpointMode.FULL_INDEX,
    )
    with pytest.raises(AgentWorkContextConflict, match="stale_work_context_revision"):
        await store.advance_recall_checkpoint(stale_context_full, expected_revision=2)
    changed_context_delta = checkpoint(
        fourth,
        revision=3,
        operation_id="checkpoint:changed-context-delta",
        knowledge_sequence=13,
        index_readiness_sequence=10,
        processing_mode=AgentRecallCheckpointMode.DELTA,
    )
    with pytest.raises(AgentWorkContextConflict, match="changed_context_requires_full_index"):
        await store.advance_recall_checkpoint(changed_context_delta, expected_revision=2)

    full_after_change = checkpoint(
        fourth,
        revision=3,
        operation_id="checkpoint:changed-context-full",
        knowledge_sequence=20,
        index_readiness_sequence=15,
        processing_mode=AgentRecallCheckpointMode.FULL_INDEX,
    )
    await store.advance_recall_checkpoint(full_after_change, expected_revision=2)
    assert await store.load_recall_checkpoint(full_after_change.key()) == full_after_change

    checkpoint_competitors = (
        checkpoint(
            fourth,
            revision=4,
            operation_id="checkpoint:concurrent-a",
            knowledge_sequence=21,
            index_readiness_sequence=16,
            processing_mode=AgentRecallCheckpointMode.DELTA,
        ),
        checkpoint(
            fourth,
            revision=4,
            operation_id="checkpoint:concurrent-b",
            knowledge_sequence=22,
            index_readiness_sequence=17,
            processing_mode=AgentRecallCheckpointMode.DELTA,
        ),
    )
    checkpoint_outcomes = await asyncio.gather(
        *(
            store.advance_recall_checkpoint(candidate, expected_revision=3)
            for candidate in checkpoint_competitors
        ),
        return_exceptions=True,
    )
    checkpoint_successes = [
        outcome for outcome in checkpoint_outcomes if not isinstance(outcome, BaseException)
    ]
    checkpoint_failures = [
        outcome for outcome in checkpoint_outcomes if isinstance(outcome, BaseException)
    ]
    assert len(checkpoint_successes) == 1
    assert len(checkpoint_failures) == 1
    assert isinstance(checkpoint_failures[0], AgentWorkContextConflict)
    current_checkpoint = await store.load_recall_checkpoint(full_after_change.key())
    assert current_checkpoint in checkpoint_competitors

    retry_candidate = checkpoint(
        fourth,
        revision=5,
        operation_id="checkpoint:retry",
        knowledge_sequence=23,
        index_readiness_sequence=18,
        processing_mode=AgentRecallCheckpointMode.DELTA,
    )
    assert (
        await store.advance_recall_checkpoint(retry_candidate, expected_revision=4)
        == retry_candidate
    )
    assert (
        await store.advance_recall_checkpoint(retry_candidate, expected_revision=4)
        == retry_candidate
    )
    assert await store.load_recall_checkpoint(retry_candidate.key()) == retry_candidate
    assert (
        await store.advance_recall_checkpoint(initial_checkpoint, expected_revision=None)
        == initial_checkpoint
    )
    assert await store.load_recall_checkpoint(retry_candidate.key()) == retry_candidate

    narrowed_access = checkpoint(
        fourth,
        revision=1,
        operation_id="checkpoint:narrowed-access",
        access_policy_sha256=_OTHER_ACCESS_POLICY_SHA256,
        knowledge_sequence=20,
        index_readiness_sequence=15,
    )
    await store.advance_recall_checkpoint(narrowed_access, expected_revision=None)
    assert await store.load_recall_checkpoint(narrowed_access.key()) == narrowed_access
    assert await store.load_recall_checkpoint(full_after_change.key()) == retry_candidate

    other_namespace = checkpoint(
        fourth,
        revision=1,
        operation_id="checkpoint:other-namespace",
        knowledge_namespace="project:other",
        knowledge_sequence=2,
        index_readiness_sequence=1,
    )
    await store.advance_recall_checkpoint(other_namespace, expected_revision=None)
    assert await store.load_recall_checkpoint(other_namespace.key()) == other_namespace
    assert await store.load_recall_checkpoint(retry_candidate.key()) == retry_candidate

    other_task = context(
        task_id="task-other-memory-v51",
        revision=1,
        operation_id="context:other-task",
    )
    await store.publish_work_context(other_task, expected_revision=None)
    other_task_checkpoint = checkpoint(
        other_task,
        revision=1,
        operation_id="checkpoint:other-task",
        knowledge_sequence=4,
        index_readiness_sequence=3,
    )
    await store.advance_recall_checkpoint(
        other_task_checkpoint,
        expected_revision=None,
    )
    assert await store.load_recall_checkpoint(other_task_checkpoint.key()) == other_task_checkpoint
    assert await store.load_recall_checkpoint(retry_candidate.key()) == retry_candidate

    frontier_checkpoint = checkpoint(
        other_task,
        revision=1,
        operation_id="checkpoint:captured-frontiers",
        agent_id="agent:frontier-auditor",
        knowledge_sequence=4,
        index_readiness_sequence=3,
        knowledge_high_water_sequence=100,
        index_readiness_high_water_sequence=80,
    )
    await store.advance_recall_checkpoint(frontier_checkpoint, expected_revision=None)
    lowered_knowledge_high_water = checkpoint(
        other_task,
        revision=2,
        operation_id="checkpoint:lowered-knowledge-high-water",
        agent_id=frontier_checkpoint.agent_id,
        knowledge_sequence=5,
        index_readiness_sequence=4,
        knowledge_high_water_sequence=5,
        index_readiness_high_water_sequence=80,
        processing_mode=AgentRecallCheckpointMode.DELTA,
    )
    with pytest.raises(AgentWorkContextConflict, match="knowledge_high_water_regression"):
        await store.advance_recall_checkpoint(
            lowered_knowledge_high_water,
            expected_revision=1,
        )
    lowered_index_high_water = checkpoint(
        other_task,
        revision=2,
        operation_id="checkpoint:lowered-index-high-water",
        agent_id=frontier_checkpoint.agent_id,
        knowledge_sequence=5,
        index_readiness_sequence=4,
        knowledge_high_water_sequence=100,
        index_readiness_high_water_sequence=4,
        processing_mode=AgentRecallCheckpointMode.DELTA,
    )
    with pytest.raises(AgentWorkContextConflict, match="index_high_water_regression"):
        await store.advance_recall_checkpoint(
            lowered_index_high_water,
            expected_revision=1,
        )
    assert await store.load_recall_checkpoint(frontier_checkpoint.key()) == frontier_checkpoint

    unknown_context = checkpoint(
        fourth.model_copy(update={"revision": 5}),
        revision=1,
        operation_id="checkpoint:unknown-context",
        agent_id="agent:unknown-context",
    )
    with pytest.raises(AgentWorkContextConflict, match="unknown_work_context"):
        await store.advance_recall_checkpoint(unknown_context, expected_revision=None)

    knowledge_store = InMemoryKnowledgeStore()
    await _create_delivery_knowledge(knowledge_store, "delivery-entry-1")
    processor = _delivery_processor(knowledge_store)
    first_result = await _process_delivery(
        processor,
        fourth,
        operation_id="delivery:process:1",
        checkpoint_value=None,
    )
    first_delivery = _stageable_delivery(
        first_result,
        delivery_id="delivery:1",
        expected_checkpoint_revision=None,
        staged_at=_STARTED_AT + timedelta(minutes=45),
    )
    assert AgentRecallDelivery.model_validate_json(first_delivery.model_dump_json()) == (
        first_delivery
    )
    with pytest.raises(TypeError):
        cast("Any", first_delivery.processing_result)["mode"] = "no_work"
    tampered_delivery = first_delivery.model_dump(mode="python")
    tampered_delivery["processing_result_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="fingerprint does not match"):
        AgentRecallDelivery.model_validate(tampered_delivery)
    assert first_delivery.materialized_result() == first_result
    assert await store.load_recall_checkpoint(first_delivery.key()) is None

    future_staged_delivery = _stageable_delivery(
        first_result,
        delivery_id="delivery:future-staged",
        expected_checkpoint_revision=None,
        staged_at=first_delivery.staged_at + timedelta(days=365),
    )
    with pytest.raises(AgentRecallDeliveryConflict, match="delivery_staged_in_future"):
        await store.stage_recall_delivery(future_staged_delivery)
    assert await store.load_recall_delivery(future_staged_delivery.delivery_id) is None
    assert await store.load_recall_checkpoint(first_delivery.key()) is None

    first_stage = await store.stage_recall_delivery(first_delivery)
    assert first_stage.state is AgentRecallDeliveryState.PENDING
    assert first_stage.delivery.materialized_result() == first_result
    assert await store.load_recall_delivery(first_delivery.delivery_id) == first_stage
    assert await store.load_recall_checkpoint(first_delivery.key()) == first_delivery.checkpoint
    assert await store.stage_recall_delivery(first_delivery) == first_stage

    reused_operation_result = await _process_delivery(
        processor,
        fourth,
        operation_id=first_delivery.operation_id,
        checkpoint_value=None,
        agent_id="agent:delivery-operation-reuse",
    )
    reused_operation_delivery = _stageable_delivery(
        reused_operation_result,
        delivery_id="delivery:operation-reuse",
        expected_checkpoint_revision=None,
        staged_at=_STARTED_AT + timedelta(minutes=45),
    )
    with pytest.raises(AgentRecallDeliveryConflict, match="delivery_operation_reused"):
        await store.stage_recall_delivery(reused_operation_delivery)
    assert await store.load_recall_delivery(reused_operation_delivery.delivery_id) is None
    assert await store.load_recall_checkpoint(reused_operation_delivery.key()) is None

    await knowledge_store.delete_entry(
        "delivery-entry-1",
        expected_revision=1,
        access_scope=KnowledgeAccessScope.for_namespace(_DELIVERY_NAMESPACE),
        hard=True,
    )
    materialized_after_source_deletion = await store.load_recall_delivery(
        first_delivery.delivery_id
    )
    assert materialized_after_source_deletion is not None
    assert materialized_after_source_deletion.delivery.materialized_result() == first_result

    with pytest.raises(AgentRecallDeliveryConflict, match="delivery_id_reused"):
        await store.stage_recall_delivery(
            first_delivery.model_copy(update={"staged_by": "runtime:other"})
        )
    with pytest.raises(AgentRecallDeliveryConflict, match="checkpoint_delivery_exists"):
        await store.stage_recall_delivery(
            first_delivery.model_copy(update={"delivery_id": "delivery:duplicate-checkpoint"})
        )

    await _create_delivery_knowledge(knowledge_store, "delivery-entry-2")
    second_result = await _process_delivery(
        processor,
        fourth,
        operation_id="delivery:process:2",
        checkpoint_value=first_delivery.checkpoint,
    )
    second_delivery = _stageable_delivery(
        second_result,
        delivery_id="delivery:2",
        expected_checkpoint_revision=1,
        staged_at=_STARTED_AT + timedelta(minutes=46),
    )
    second_stage = await store.stage_recall_delivery(second_delivery)
    assert second_stage.state is AgentRecallDeliveryState.PENDING
    assert await store.load_recall_checkpoint(second_delivery.key()) == (second_delivery.checkpoint)

    claim_outcomes = await asyncio.gather(
        store.claim_recall_delivery(
            first_delivery.key(),
            claim_id="claim:concurrent-a",
            worker_id="worker:a",
            lease_seconds=20,
        ),
        store.claim_recall_delivery(
            first_delivery.key(),
            claim_id="claim:concurrent-b",
            worker_id="worker:b",
            lease_seconds=20,
        ),
    )
    claimed_records = [record for record in claim_outcomes if record is not None]
    assert len(claimed_records) == 1
    claimed = claimed_records[0]
    assert claimed.delivery.delivery_id == first_delivery.delivery_id
    assert claimed.state is AgentRecallDeliveryState.CLAIMED
    assert claimed.claim is not None
    original_claim = claimed.claim
    assert (
        await store.claim_recall_delivery(
            first_delivery.key(),
            claim_id=original_claim.claim_id,
            worker_id=original_claim.worker_id,
            lease_seconds=20,
        )
        == claimed
    )
    with pytest.raises(AgentRecallDeliveryConflict, match="claim_id_reused"):
        await store.claim_recall_delivery(
            first_delivery.key(),
            claim_id=original_claim.claim_id,
            worker_id="worker:reused",
            lease_seconds=20,
        )

    renewed = await store.renew_recall_delivery(original_claim, lease_seconds=30)
    assert renewed.claim is not None
    assert renewed.claim.state_revision == original_claim.state_revision + 1
    assert await store.renew_recall_delivery(original_claim, lease_seconds=30) == renewed
    with pytest.raises(AgentRecallDeliveryConflict, match="renewal_reused"):
        await store.renew_recall_delivery(original_claim, lease_seconds=31)

    assert renewed.claim is not None
    future_transition_time = renewed.claim.claimed_at + timedelta(seconds=1)
    with pytest.raises(AgentRecallDeliveryConflict, match="release_from_future"):
        await store.release_recall_delivery(
            renewed.claim,
            release_id="release:future",
            reason="future event cannot advance the delivery clock",
            released_at=future_transition_time,
        )
    with pytest.raises(AgentRecallDeliveryConflict, match="acknowledgement_from_future"):
        await store.acknowledge_recall_delivery(
            renewed.claim,
            acknowledgement_id="ack:future",
            evidence_kind=AgentRecallDeliveryEvidenceKind.APPLICATION_HANDOFF,
            evidence_ref="handoff:future",
            acknowledged_at=future_transition_time,
        )
    assert await store.load_recall_delivery(renewed.delivery.delivery_id) == renewed

    with pytest.raises(AgentRecallDeliveryConflict, match="release_outside_claim_lease"):
        await store.release_recall_delivery(
            renewed.claim,
            release_id="release:outside-lease",
            reason="invalid event after lease authority",
            released_at=renewed.claim.lease_expires_at,
        )
    with pytest.raises(AgentRecallDeliveryConflict, match="acknowledgement_outside_claim_lease"):
        await store.acknowledge_recall_delivery(
            renewed.claim,
            acknowledgement_id="ack:outside-lease",
            evidence_kind=AgentRecallDeliveryEvidenceKind.APPLICATION_HANDOFF,
            evidence_ref="handoff:outside-lease",
            acknowledged_at=renewed.claim.lease_expires_at,
        )

    release_time = _STARTED_AT + timedelta(minutes=60)
    released = await store.release_recall_delivery(
        renewed.claim,
        release_id="release:1",
        reason="retry after downstream handoff interruption",
        released_at=release_time,
    )
    assert released.state is AgentRecallDeliveryState.PENDING
    assert released.release is not None
    assert released.release.reason == "retry after downstream handoff interruption"
    assert (
        await store.release_recall_delivery(
            renewed.claim,
            release_id="release:1",
            reason="retry after downstream handoff interruption",
            released_at=release_time,
        )
        == released
    )
    with pytest.raises(AgentRecallDeliveryConflict, match="release_id_reused"):
        await store.release_recall_delivery(
            renewed.claim,
            release_id="release:1",
            reason="different retry reason",
            released_at=release_time,
        )
    with pytest.raises(AgentRecallDeliveryConflict, match="stale_delivery_claim"):
        await store.acknowledge_recall_delivery(
            renewed.claim,
            acknowledgement_id="ack:stale",
            evidence_kind=AgentRecallDeliveryEvidenceKind.RECALL_RECEIPT,
            evidence_ref="receipt:stale",
            acknowledged_at=_STARTED_AT + timedelta(minutes=60),
        )

    retry = await store.claim_recall_delivery(
        first_delivery.key(),
        claim_id="claim:retry",
        worker_id="worker:retry",
        lease_seconds=20,
    )
    assert retry is not None
    assert retry.claim is not None
    assert retry.claim.attempt == 2
    assert retry.claim.claimed_at == release_time
    assert retry.claim.lease_expires_at == release_time + timedelta(seconds=20)
    with pytest.raises(AgentRecallDeliveryConflict, match="claim_replay_superseded"):
        await store.claim_recall_delivery(
            first_delivery.key(),
            claim_id=original_claim.claim_id,
            worker_id=original_claim.worker_id,
            lease_seconds=20,
        )
    with pytest.raises(AgentRecallDeliveryConflict, match="release_replay_superseded"):
        await store.release_recall_delivery(
            renewed.claim,
            release_id="release:1",
            reason="retry after downstream handoff interruption",
            released_at=release_time,
        )
    acknowledged = await store.acknowledge_recall_delivery(
        retry.claim,
        acknowledgement_id="ack:1",
        evidence_kind=AgentRecallDeliveryEvidenceKind.RECALL_RECEIPT,
        evidence_ref="receipt:1",
        acknowledged_at=_STARTED_AT + timedelta(minutes=60),
    )
    assert acknowledged.state is AgentRecallDeliveryState.ACKNOWLEDGED
    assert acknowledged.acknowledgement is not None
    assert (
        acknowledged.acknowledgement.evidence_kind is AgentRecallDeliveryEvidenceKind.RECALL_RECEIPT
    )
    assert (
        await store.acknowledge_recall_delivery(
            retry.claim,
            acknowledgement_id="ack:1",
            evidence_kind=AgentRecallDeliveryEvidenceKind.RECALL_RECEIPT,
            evidence_ref="receipt:1",
            acknowledged_at=_STARTED_AT + timedelta(minutes=60),
        )
        == acknowledged
    )
    with pytest.raises(AgentRecallDeliveryConflict, match="acknowledgement_reused"):
        await store.acknowledge_recall_delivery(
            retry.claim,
            acknowledgement_id="ack:1",
            evidence_kind=AgentRecallDeliveryEvidenceKind.CONTEXT_EXPOSURE,
            evidence_ref="exposure:1",
            acknowledged_at=_STARTED_AT + timedelta(minutes=60),
        )

    next_claimed = await store.claim_recall_delivery(
        first_delivery.key(),
        claim_id="claim:second",
        worker_id="worker:second",
        lease_seconds=20,
    )
    assert next_claimed is not None
    assert next_claimed.delivery.delivery_id == second_delivery.delivery_id
    assert next_claimed.claim is not None
    await store.acknowledge_recall_delivery(
        next_claimed.claim,
        acknowledgement_id="ack:2",
        evidence_kind=AgentRecallDeliveryEvidenceKind.APPLICATION_HANDOFF,
        evidence_ref="handoff:2",
        acknowledged_at=_STARTED_AT + timedelta(minutes=60),
    )
    assert (
        await store.claim_recall_delivery(
            first_delivery.key(),
            claim_id="claim:none",
            worker_id="worker:none",
            lease_seconds=20,
        )
        is None
    )

    if advance_clock is not None:
        await _create_delivery_knowledge(knowledge_store, "delivery-entry-3")
        third_result = await _process_delivery(
            processor,
            fourth,
            operation_id="delivery:process:3",
            checkpoint_value=second_delivery.checkpoint,
        )
        third_delivery = _stageable_delivery(
            third_result,
            delivery_id="delivery:3",
            expected_checkpoint_revision=2,
            staged_at=_STARTED_AT + timedelta(minutes=47),
        )
        await store.stage_recall_delivery(third_delivery)
        expiring = await store.claim_recall_delivery(
            third_delivery.key(),
            claim_id="claim:expiring",
            worker_id="worker:expiring",
            lease_seconds=10,
        )
        assert expiring is not None
        assert expiring.claim is not None
        renewed_expiring = await store.renew_recall_delivery(
            expiring.claim,
            lease_seconds=10,
        )
        assert renewed_expiring.claim is not None
        advance_clock(timedelta(seconds=11))
        with pytest.raises(AgentRecallDeliveryConflict, match="expired_delivery_claim"):
            await store.claim_recall_delivery(
                third_delivery.key(),
                claim_id="claim:expiring",
                worker_id="worker:expiring",
                lease_seconds=10,
            )
        with pytest.raises(AgentRecallDeliveryConflict, match="expired_delivery_claim"):
            await store.renew_recall_delivery(expiring.claim, lease_seconds=10)
        with pytest.raises(AgentRecallDeliveryConflict, match="expired_delivery_claim"):
            await store.renew_recall_delivery(renewed_expiring.claim, lease_seconds=10)
        with pytest.raises(AgentRecallDeliveryConflict, match="expired_delivery_claim"):
            await store.release_recall_delivery(
                renewed_expiring.claim,
                release_id="release:expired",
                reason="expired worker cannot release",
                released_at=_STARTED_AT + timedelta(minutes=60, seconds=11),
            )
        takeover = await store.claim_recall_delivery(
            third_delivery.key(),
            claim_id="claim:takeover",
            worker_id="worker:takeover",
            lease_seconds=10,
        )
        assert takeover is not None
        assert takeover.claim is not None
        assert takeover.claim.attempt == 2
        with pytest.raises(AgentRecallDeliveryConflict, match="claim_replay_superseded"):
            await store.claim_recall_delivery(
                third_delivery.key(),
                claim_id="claim:expiring",
                worker_id="worker:expiring",
                lease_seconds=10,
            )
        with pytest.raises(AgentRecallDeliveryConflict, match="stale_delivery_claim"):
            await store.acknowledge_recall_delivery(
                renewed_expiring.claim,
                acknowledgement_id="ack:expired",
                evidence_kind=AgentRecallDeliveryEvidenceKind.APPLICATION_HANDOFF,
                evidence_ref="handoff:expired",
                acknowledged_at=_STARTED_AT + timedelta(minutes=60, seconds=11),
            )
        terminal = await store.acknowledge_recall_delivery(
            takeover.claim,
            acknowledgement_id="ack:takeover",
            evidence_kind=AgentRecallDeliveryEvidenceKind.APPLICATION_HANDOFF,
            evidence_ref="handoff:takeover",
            acknowledged_at=_STARTED_AT + timedelta(minutes=60, seconds=11),
        )
        assert terminal.state is AgentRecallDeliveryState.ACKNOWLEDGED
        for acknowledged_delivery_id in ("delivery:1", "delivery:2", "delivery:3"):
            completed_delivery = await store.load_recall_delivery(acknowledged_delivery_id)
            assert completed_delivery is not None
            assert completed_delivery.state is AgentRecallDeliveryState.ACKNOWLEDGED

        durable_states: list[AgentRecallDelivery] = []
        for revision, disposition in enumerate(
            ("claimed", "released", "pending"),
            start=4,
        ):
            entry_id = f"delivery-entry-{revision}"
            await _create_delivery_knowledge(knowledge_store, entry_id)
            result = await _process_delivery(
                processor,
                fourth,
                operation_id=f"delivery:process:{revision}",
                checkpoint_value=None,
                agent_id=f"agent:delivery:{disposition}",
            )
            staged_delivery = _stageable_delivery(
                result,
                delivery_id=f"delivery:{revision}",
                expected_checkpoint_revision=None,
                staged_at=_STARTED_AT + timedelta(minutes=44 + revision),
            )
            await store.stage_recall_delivery(staged_delivery)
            durable_states.append(staged_delivery)
            loaded_staged = await store.load_recall_delivery(staged_delivery.delivery_id)
            assert loaded_staged is not None
            assert loaded_staged.state is AgentRecallDeliveryState.PENDING
            if disposition == "pending":
                continue
            claimed_state = await store.claim_recall_delivery(
                staged_delivery.key(),
                claim_id=f"claim:durable:{revision}",
                worker_id=f"worker:durable:{revision}",
                lease_seconds=30,
            )
            assert claimed_state is not None
            assert claimed_state.claim is not None
            if disposition == "claimed":
                continue
            released_state = await store.release_recall_delivery(
                claimed_state.claim,
                release_id=f"release:durable:{revision}",
                reason="durable retry evidence",
                released_at=_STARTED_AT + timedelta(minutes=60, seconds=11),
            )
            assert released_state.state is AgentRecallDeliveryState.PENDING

        assert tuple(delivery.delivery_id for delivery in durable_states) == (
            "delivery:4",
            "delivery:5",
            "delivery:6",
        )

    stale_base = context(
        task_id="task:stale-delivery-context",
        revision=1,
        operation_id="delivery:stale-context:create",
    )
    await store.publish_work_context(stale_base, expected_revision=None)
    stale_result = await _process_delivery(
        processor,
        stale_base,
        operation_id="delivery:stale-context:process",
        checkpoint_value=None,
        agent_id="agent:stale-delivery-context",
    )
    stale_delivery = _stageable_delivery(
        stale_result,
        delivery_id="delivery:stale-context",
        expected_checkpoint_revision=None,
        staged_at=_STARTED_AT + timedelta(minutes=50),
    )
    stale_successor = context(
        task_id=stale_base.task_id,
        revision=2,
        operation_id="delivery:stale-context:advance",
        goal="Move past the staged processing basis",
    )
    await store.publish_work_context(stale_successor, expected_revision=1)
    with pytest.raises(AgentWorkContextConflict, match="stale_work_context_revision"):
        await store.stage_recall_delivery(stale_delivery)
    assert await store.load_recall_delivery(stale_delivery.delivery_id) is None
    assert await store.load_recall_checkpoint(stale_delivery.key()) is None


__all__ = [
    "assert_work_context_store_conformance",
    "checkpoint",
    "context",
    "recall_delivery",
]
