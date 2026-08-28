from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from cayu import (
    AgentRecallCheckpoint,
    AgentRecallCheckpointMode,
    AgentWorkContext,
    AgentWorkContextConflict,
    AgentWorkContextStore,
)

_ACCESS_POLICY_SHA256 = "a" * 64
_OTHER_ACCESS_POLICY_SHA256 = "b" * 64
_STARTED_AT = datetime(2026, 8, 28, 8, 0, tzinfo=UTC)


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


async def assert_work_context_store_conformance(store: AgentWorkContextStore) -> None:
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


__all__ = ["assert_work_context_store_conformance", "checkpoint", "context"]
