"""Public group preparation shares the task graph's provenance boundary."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from cayu.runtime._task_graphs import _safe_graph_id, prepare_graph_request, validate_graph_request
from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_group_quiescence_capability_is_complete,
)
from cayu.tasks._groups import validate_group_cursor
from cayu.tasks.base import TaskStore
from cayu.tasks.graphs import task_graph_request_sha256
from cayu.tasks.groups import (
    TaskGroupConflict,
    TaskGroupCreate,
    TaskGroupCreationReceipt,
    TaskGroupEvent,
    TaskGroupQuiescenceResolution,
    TaskGroupSnapshot,
    copy_task_group_create,
    task_group_request_sha256,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp

T = TypeVar("T", bound=BaseModel)


def _validated(app: CayuApp, value: object, expected: type[T]) -> T:
    if not isinstance(value, expected) or type(value) is not expected:
        raise TaskGroupConflict("Task store returned invalid group evidence.")
    outcome = capture_sensitive_validation(
        lambda: expected.model_validate(value.model_dump(mode="json", warnings=False)),
        operation_name="Task group evidence validation",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    if outcome.result is None:
        raise TaskGroupConflict("Task store returned no group evidence.")
    return outcome.result


def _store(app: CayuApp) -> TaskStore:
    store = app.task_store
    if store is None or not store.supports_task_groups or not store.supports_task_graphs:
        raise NotImplementedError("A group-capable task store is required.")
    return store


async def create_task_group(app: CayuApp, request: TaskGroupCreate) -> TaskGroupCreationReceipt:
    validation = capture_sensitive_validation(
        lambda request=request: copy_task_group_create(request),
        operation_name="Task group validation",
        redactor=app._secret_redactor,
    )
    del request
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    copied = validation.result
    if copied is None:
        raise ValueError("Task group validation failed.")
    store = _store(app)
    graph = validate_graph_request(app, copied.graph)
    _safe_graph_id(app, copied.group_id)
    if copied.quiescence is not None and not task_store_group_quiescence_capability_is_complete(
        store
    ):
        raise NotImplementedError("Task group quiescence requires an explicitly capable store.")
    copied = TaskGroupCreate(
        group_id=copied.group_id,
        graph=graph,
        member_task_ids=copied.member_task_ids,
        policy=copied.policy,
        quiescence=copied.quiescence,
        finalizer_task_id=copied.finalizer_task_id,
    )
    submitted = task_group_request_sha256(copied)
    existing = await load_task_group(app, copied.group_id)
    if existing is not None:
        receipt = existing.receipt
        if (
            receipt.submitted_request_sha256 != submitted
            or receipt.graph.graph_id != graph.graph_id
            or receipt.member_task_ids != copied.member_task_ids
            or receipt.policy != copied.policy
            or receipt.quiescence != copied.quiescence
            or receipt.finalizer_task_id != copied.finalizer_task_id
            or receipt.graph.task_ids != tuple(node.task.task_id for node in graph.nodes)
            or receipt.graph.submitted_request_sha256 != task_graph_request_sha256(graph)
        ):
            raise TaskGroupConflict("Group identity has different submission content.")
        return receipt
    graph = await prepare_graph_request(app, graph)
    prepared = TaskGroupCreate(
        group_id=copied.group_id,
        graph=graph,
        member_task_ids=copied.member_task_ids,
        policy=copied.policy,
        quiescence=copied.quiescence,
        finalizer_task_id=copied.finalizer_task_id,
    )
    prepared._submitted_request_sha256 = submitted
    expected = task_group_request_sha256(prepared)
    outcome = await capture_task_store_operation(
        partial(store.create_task_group, prepared),
        operation_name="Task group admission",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    receipt = _validated(app, outcome.result, TaskGroupCreationReceipt)
    if (
        receipt.group_id != prepared.group_id
        or receipt.graph.graph_id != graph.graph_id
        or receipt.graph.request_sha256 != task_graph_request_sha256(graph)
        or receipt.graph.submitted_request_sha256 != graph._submitted_request_sha256
        or receipt.graph.task_ids != tuple(n.task.task_id for n in graph.nodes)
        or receipt.request_sha256 != expected
        or receipt.submitted_request_sha256 != submitted
        or receipt.member_task_ids != prepared.member_task_ids
        or receipt.policy != prepared.policy
        or receipt.quiescence != prepared.quiescence
        or receipt.finalizer_task_id != prepared.finalizer_task_id
    ):
        raise TaskGroupConflict("Task store did not preserve exact group admission authority.")
    return receipt


async def load_task_group(app: CayuApp, group_id: str) -> TaskGroupSnapshot | None:
    group_id = _safe_graph_id(app, group_id)
    outcome = await capture_task_store_operation(
        partial(_store(app).load_task_group, group_id),
        operation_name="Task group inspection",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    if outcome.result is None:
        return None
    snapshot = _validated(app, outcome.result, TaskGroupSnapshot)
    if snapshot.receipt.group_id != group_id:
        raise TaskGroupConflict("Task store returned a different group.")
    for identity in (snapshot.receipt.graph.graph_id, *snapshot.receipt.graph.task_ids):
        _safe_graph_id(app, identity)
    for member in snapshot.members:
        for identity in member.prerequisite_task_ids:
            _safe_graph_id(app, identity)
    return snapshot


async def list_task_group_events(
    app: CayuApp, group_id: str, *, after_sequence: int, limit: int
) -> list[TaskGroupEvent]:
    group_id = _safe_graph_id(app, group_id)
    validate_group_cursor(after_sequence, limit)
    outcome = await capture_task_store_operation(
        partial(
            _store(app).list_task_group_events, group_id, after_sequence=after_sequence, limit=limit
        ),
        operation_name="Task group event inspection",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    values = outcome.result
    if type(values) is not list or len(values) > limit:
        raise TaskGroupConflict("Task store returned invalid group events.")
    events = [_validated(app, value, TaskGroupEvent) for value in values]
    for sequence, event in enumerate(events, after_sequence + 1):
        if event.group_id != group_id or event.sequence != sequence:
            raise TaskGroupConflict("Group event identity or order conflicts.")
        if event.task_id is not None:
            _safe_graph_id(app, event.task_id)
        if event.decision is not None:
            for identity in (
                *event.decision.successful_task_ids,
                *event.decision.unsuccessful_task_ids,
            ):
                _safe_graph_id(app, identity)
    return events


async def reconcile_task_group(app: CayuApp, group_id: str) -> TaskGroupSnapshot:
    group_id = _safe_graph_id(app, group_id)
    store = _store(app)
    if not store.supports_task_group_quiescence:
        raise NotImplementedError("Task group quiescence is unavailable.")
    before = await load_task_group(app, group_id)
    if before is not None:
        from cayu.runtime._task_group_invocation import observe_release

        for execution in before.quiescence.executions:
            if (
                execution.worker_id is None
                and execution.settled_at is None
                and execution.invocation is not None
            ):
                await observe_release(
                    store,
                    app.session_store,
                    execution.invocation,
                    redactor=app._secret_redactor,
                )
    outcome = await capture_task_store_operation(
        partial(store.reconcile_task_group, group_id),
        operation_name="Task group quiescence reconciliation",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    result = _validated(app, outcome.result, TaskGroupSnapshot)
    if result.receipt.group_id != group_id:
        raise TaskGroupConflict("Task store reconciled a different group.")
    return result


async def resolve_task_group_quiescence(
    app: CayuApp,
    request: TaskGroupQuiescenceResolution,
) -> TaskGroupSnapshot:
    from cayu.tasks._group_quiescence import prepare_resolution

    prepared = capture_sensitive_validation(
        lambda: prepare_resolution(request)[0],
        operation_name="Task group resolution validation",
        redactor=app._secret_redactor,
    )
    if prepared.failure is not None:
        raise_task_store_operation_failure(prepared.failure)
    copied = prepared.result
    if copied is None:
        raise TaskGroupConflict("Task group resolution has no authority.")
    _safe_graph_id(app, copied.group_id)
    _safe_graph_id(app, copied.idempotency_key)
    store = _store(app)
    if not store.supports_task_group_quiescence:
        raise NotImplementedError("Task group quiescence is unavailable.")
    outcome = await capture_task_store_operation(
        partial(store.resolve_task_group_quiescence, copied),
        operation_name="Task group quiescence resolution",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    result = _validated(app, outcome.result, TaskGroupSnapshot)
    if (
        result.receipt.group_id != copied.group_id
        or result.receipt.request_sha256 != copied.request_sha256
        or result.quiescence.status.value != "quiescent"
    ):
        raise TaskGroupConflict("Task store did not preserve exact group resolution.")
    return result
