"""Application-owned task-graph preparation; stores own atomic admission."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING

from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.runtime._task_store_operation_boundary import (
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime._verified_work_authority import invocation_contains_secret_public_identity
from cayu.sessions.invocation import TaskExecutionSource
from cayu.tasks._graph_admission import prepare_graph_admission
from cayu.tasks.base import TaskInvocationSnapshot, task_create_with_runtime_invocation
from cayu.tasks.graphs import (
    TaskGraphConflict,
    TaskGraphCreate,
    TaskGraphCreationReceipt,
    TaskGraphEvent,
    TaskGraphNode,
    TaskGraphSnapshot,
    copy_task_graph_create,
    graph_identifier,
    task_graph_request_sha256,
    task_graph_with_runtime_admission,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp


def validate_graph_request(app: CayuApp, request: TaskGraphCreate) -> TaskGraphCreate:
    validation = capture_sensitive_validation(
        lambda value=request: copy_task_graph_create(value),
        operation_name="Task graph validation",
        redactor=app._secret_redactor,
    )
    del request
    failure = validation.failure
    copied = validation.result
    del validation
    if failure is not None:
        raise failure from None
    if copied is None:
        raise ValueError("Task graph validation failed.")
    # Public submissions never inherit another SDK call's internal preparation.
    copied = TaskGraphCreate(graph_id=copied.graph_id, nodes=copied.nodes)
    store = app.task_store
    if store is None or not store.supports_task_graphs:
        raise NotImplementedError("A graph-capable task store is required.")
    identities: list[str | None] = [copied.graph_id]
    for node in copied.nodes:
        identities.extend(
            [node.task.task_id, node.task.parent_task_id, *node.prerequisite_task_ids]
        )
        identities.append(node.task.session_id)
        for value, supported, description in (
            (node.task.available_at, store.supports_delayed_availability, "delayed availability"),
            (node.task.schedule_policy, store.supports_task_scheduling, "managed scheduling"),
            (node.task.retry_policy, store.supports_task_retry_series, "retry series"),
            (
                node.task.work_contract,
                store.supports_verified_work_contracts,
                "verified work contracts",
            ),
        ):
            if value is not None and not supported:
                raise NotImplementedError(f"Task graph store does not support {description}.")
        if (
            node.task.retry_policy is not None
            and app._secret_redactor.redact_uppercase_text(node.task.retry_policy.cost_currency)
            != node.task.retry_policy.cost_currency
        ):
            raise ValueError("Task graph accounting authority contains a workload secret.")
        if node.task.work_contract is not None:
            identities.append(node.task.work_contract.contract_id)
        for origin in (node.task.invocation_origin, node.task._verified_invocation_origin):
            if origin is not None:
                identities.extend([origin.subject, origin.tenant])
    if any(
        value is not None and app._secret_redactor.redact_text(value) != value
        for value in identities
    ):
        del identities, copied
        raise ValueError("Task graph identity contains a workload secret.")
    return copied


async def create_task_graph(app: CayuApp, request: TaskGraphCreate) -> TaskGraphCreationReceipt:
    copied = validate_graph_request(app, request)
    store = app.task_store
    assert store is not None
    submitted_digest = task_graph_request_sha256(copied)
    existing = await load_task_graph(app, copied.graph_id)
    if existing is not None:
        if (
            existing.receipt.submitted_request_sha256 != submitted_digest
            or existing.receipt.task_ids != tuple(node.task.task_id for node in copied.nodes)
        ):
            raise TaskGraphConflict("Graph identity has different submission content.")
        return existing.receipt
    prepared_request = await prepare_graph_request(app, copied)
    expected_digest = task_graph_request_sha256(prepared_request)
    outcome = await capture_task_store_operation(
        partial(store.create_task_graph, prepared_request),
        operation_name="Task graph admission",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    result = outcome.result
    if type(result) is not TaskGraphCreationReceipt:
        raise TaskGraphConflict("Task store returned no graph admission receipt.")
    validation = capture_sensitive_validation(
        lambda: TaskGraphCreationReceipt.model_validate(
            result.model_dump(mode="json", warnings=False)
        ),
        operation_name="Task graph receipt validation",
        redactor=app._secret_redactor,
    )
    failure = validation.failure
    receipt = validation.result
    if failure is not None:
        raise failure from None
    if (
        receipt is None
        or receipt.graph_id != copied.graph_id
        or receipt.request_sha256 != expected_digest
        or receipt.submitted_request_sha256 != submitted_digest
        or receipt.task_ids != tuple(node.task.task_id for node in copied.nodes)
    ):
        raise TaskGraphConflict("Task store did not preserve exact graph admission authority.")
    return receipt


def _safe_graph_id(app: CayuApp, graph_id: str) -> str:
    validation = capture_sensitive_validation(
        partial(graph_identifier, graph_id),
        operation_name="Task graph identity",
        redactor=app._secret_redactor,
    )
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    identity = validation.result
    if identity is None or app._secret_redactor.redact_text(identity) != identity:
        raise ValueError("Task graph identity is unsafe.")
    return identity


async def load_task_graph(app: CayuApp, graph_id: str) -> TaskGraphSnapshot | None:
    graph_id = _safe_graph_id(app, graph_id)
    store = app.task_store
    if store is None or not store.supports_task_graphs:
        raise NotImplementedError("A graph-capable task store is required.")
    outcome = await capture_task_store_operation(
        partial(store.load_task_graph, graph_id),
        operation_name="Task graph inspection",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    result = outcome.result
    if result is None:
        return None
    if type(result) is not TaskGraphSnapshot:
        raise TaskGraphConflict("Task store returned an invalid graph snapshot.")
    validation = capture_sensitive_validation(
        lambda: TaskGraphSnapshot.model_validate(result.model_dump(mode="json", warnings=False)),
        operation_name="Task graph snapshot validation",
        redactor=app._secret_redactor,
    )
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    snapshot = validation.result
    if snapshot is None or snapshot.receipt.graph_id != graph_id:
        raise TaskGraphConflict("Task store returned a different graph snapshot.")
    if any(
        app._secret_redactor.redact_text(identity) != identity
        for identity in snapshot.receipt.task_ids
    ):
        raise TaskGraphConflict("Graph snapshot member identity is unsafe.")
    return snapshot


async def list_task_graph_events(
    app: CayuApp,
    graph_id: str,
    *,
    after_sequence: int,
    limit: int,
) -> list[TaskGraphEvent]:
    graph_id = _safe_graph_id(app, graph_id)
    if type(after_sequence) is not int or not 0 <= after_sequence <= MAX_PORTABLE_JSON_INTEGER:
        raise ValueError("Invalid graph event cursor.")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Invalid graph event page size.")
    store = app.task_store
    if store is None or not store.supports_task_graphs:
        raise NotImplementedError("A graph-capable task store is required.")
    outcome = await capture_task_store_operation(
        partial(store.list_task_graph_events, graph_id, after_sequence=after_sequence, limit=limit),
        operation_name="Task graph event inspection",
        redactor=app._secret_redactor,
    )
    if outcome.failure is not None:
        raise_task_store_operation_failure(outcome.failure)
    result = outcome.result
    if (
        type(result) is not list
        or len(result) > limit
        or any(type(event) is not TaskGraphEvent for event in result)
    ):
        raise TaskGraphConflict("Task store returned an invalid graph event page.")
    validation = capture_sensitive_validation(
        lambda: [
            TaskGraphEvent.model_validate(event.model_dump(mode="json", warnings=False))
            for event in result
        ],
        operation_name="Task graph event validation",
        redactor=app._secret_redactor,
    )
    if validation.failure is not None:
        raise_task_store_operation_failure(validation.failure)
    events = validation.result
    if events is None:
        raise TaskGraphConflict("Task store returned no graph event page.")
    for sequence, event in enumerate(events, start=after_sequence + 1):
        if event.graph_id != graph_id or event.sequence != sequence:
            raise TaskGraphConflict("Task store returned contradictory graph event authority.")
        if any(
            identity is not None and app._secret_redactor.redact_text(identity) != identity
            for identity in (event.task_id, *event.prerequisite_task_ids)
        ):
            raise TaskGraphConflict("Task graph event identity is unsafe.")
    return events


async def prepare_graph_request(app: CayuApp, copied: TaskGraphCreate) -> TaskGraphCreate:
    """Shared SDK provenance preparation; never publishes or replays admission."""
    store = app.task_store
    assert store is not None
    submitted_digest = task_graph_request_sha256(copied)
    prepared = []
    for node in copied.nodes:
        task = node.task
        if (
            task.session_id is not None
            and task._verified_invocation_origin is None
            and task._runtime_session_binding is None
        ):
            lookup = await capture_task_store_operation(
                partial(app.session_store.load_invocation_snapshot, task.session_id),
                operation_name="Graph session invocation lookup",
                redactor=app._secret_redactor,
            )
            if lookup.failure is not None:
                raise_task_store_operation_failure(lookup.failure)
            snapshot = lookup.result
            if snapshot is not None:
                bound = capture_sensitive_validation(
                    lambda task=task, snapshot=snapshot: task_create_with_runtime_invocation(
                        task,
                        source=task._runtime_invocation_source or TaskExecutionSource.SDK_TASK,
                        session_invocation=snapshot,
                    ),
                    operation_name="Graph session invocation validation",
                    redactor=app._secret_redactor,
                )
                if bound.failure is not None:
                    raise_task_store_operation_failure(bound.failure)
                if bound.result is None:
                    raise TaskGraphConflict("Graph session invocation is unavailable.")
                task = bound.result
        prepared.append(TaskGraphNode(task=task, prerequisite_task_ids=node.prerequisite_task_ids))
    prepared_request = TaskGraphCreate(graph_id=copied.graph_id, nodes=tuple(prepared))
    member_ids = {node.task.task_id for node in prepared_request.nodes}
    external_parents = sorted(
        {
            node.task.parent_task_id
            for node in prepared_request.nodes
            if node.task.parent_task_id is not None and node.task.parent_task_id not in member_ids
        }
    )
    parents = {}
    for identity in external_parents:
        lookup = await capture_task_store_operation(
            partial(store.load_invocation_snapshot, identity),
            operation_name="Graph parent invocation lookup",
            redactor=app._secret_redactor,
        )
        if lookup.failure is not None:
            raise_task_store_operation_failure(lookup.failure)
        value = lookup.result
        if type(value) is not TaskInvocationSnapshot:
            raise TaskGraphConflict("Graph parent invocation authority is unavailable.")
        checked = capture_sensitive_validation(
            lambda value=value: TaskInvocationSnapshot(
                id=value.id,
                session_id=value.session_id,
                session_instance_id=value.session_instance_id,
                invocation=value.invocation,
            ),
            operation_name="Graph parent invocation validation",
            redactor=app._secret_redactor,
        )
        if checked.failure is not None:
            raise_task_store_operation_failure(checked.failure)
        if checked.result is None or checked.result.id != identity:
            raise TaskGraphConflict("Graph parent invocation identity conflicts.")
        parents[identity] = checked.result
    prepared_request = task_graph_with_runtime_admission(
        prepared_request,
        submitted_request_sha256=submitted_digest,
        parents=tuple(parents[identity] for identity in sorted(parents)),
    )
    expected_digest = task_graph_request_sha256(prepared_request)
    # Validate the whole batch (including inherited lineage and graph fields)
    # before a custom store can publish even its first member. Store-owned
    # contract registration and closure decisions remain in the transaction.
    preview = capture_sensitive_validation(
        lambda: prepare_graph_admission(
            prepared_request,
            digest=expected_digest,
            now=datetime.now(UTC),
            parents=parents,
            validate_task=lambda request: None,
        ),
        operation_name="Graph admission preflight",
        redactor=app._secret_redactor,
    )
    if preview.failure is not None:
        raise_task_store_operation_failure(preview.failure)
    if preview.result is None:
        raise TaskGraphConflict("Graph admission preflight is unavailable.")
    if any(
        invocation_contains_secret_public_identity(task.invocation, app._secret_redactor)
        for task in preview.result.tasks
    ):
        raise ValueError("Task graph inherited invocation authority contains a workload secret.")
    return prepared_request
