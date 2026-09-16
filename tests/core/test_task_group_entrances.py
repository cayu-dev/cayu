"""Alternate task mutation entrances must settle group policy atomically."""

from __future__ import annotations

import warnings

import pytest
from tests.core.test_task_groups import anyio_backend as anyio_backend
from tests.core.test_task_groups import group_request, require_group
from tests.core.test_task_groups import store as store

from cayu import (
    CayuApp,
    TaskCreate,
    TaskGraphCreate,
    TaskGraphNode,
    TaskGroupCreate,
    TaskGroupPolicy,
    TaskGroupStatus,
)
from cayu.tasks.base import TaskQuery, TaskStatus, TaskStore
from cayu.tasks.graphs import TaskGraphEventType
from cayu.tasks.groups import TaskGroupEventType

pytestmark = pytest.mark.anyio


async def test_hold_resume_keeps_group_pending(store):
    request = group_request("first_success", setup=True)
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(request)
    selected = request.member_task_ids[0]
    await store.pause_task(selected, reason="operator")
    await store.complete_task(request.group_id + "setup", {})
    snapshot = await require_group(app, request.group_id)
    assert snapshot.status is TaskGroupStatus.PENDING
    assert next(m for m in snapshot.members if m.task_id == selected).status is TaskStatus.PAUSED
    await store.resume_task(selected)
    assert (await require_group(app, request.group_id)).status is TaskGroupStatus.PENDING
    await store.complete_task(selected, {})
    assert (await require_group(app, request.group_id)).status is TaskGroupStatus.SUCCEEDED


@pytest.mark.parametrize("field", ["policy", "members", "payload", "group_id"])
async def test_mutated_group_has_no_diagnostic_side_channel(store, field, caplog, capsys):
    from cayu.vaults.redaction import SecretRedactor

    secret = "group-diagnostic-secret-canary"

    class HostileValue:
        def __repr__(self):
            return secret

        __str__ = __repr__

    request = group_request()
    identity = request.group_id
    if field == "policy":
        object.__setattr__(request.policy, "kind", HostileValue())
    elif field == "members":
        object.__setattr__(request, "member_task_ids", (HostileValue(),))
    elif field == "group_id":
        object.__setattr__(request, "group_id", secret)
    else:
        request.graph.nodes[0].task.input = {"bad": HostileValue()}
    app = CayuApp(task_store=store, enable_logging=False, secret_redactor=SecretRedactor(secret))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, TypeError)) as raised:
            await app.create_task_group(request)
    pending: list[BaseException] = [raised.value]
    seen: set[int] = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        assert secret not in str(error) + repr(error)
        pending.extend(e for e in (error.__cause__, error.__context__) if e is not None)
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    output = capsys.readouterr()
    assert secret not in caplog.text + output.out + output.err
    assert all(secret not in str(item.message) for item in recorded)
    assert await store.load_task_group(identity) is None
    assert await store.load_task_graph(request.graph.graph_id) is None


async def test_group_verified_completion_is_the_only_success_authority(store: TaskStore) -> None:
    from tests.core.task_invocation_fixtures import unattributed_session_invocation_binding
    from tests.core.test_completion_decision_application import (
        _contract,
        _persist_decision,
        _result,
    )

    from cayu import CayuApp
    from cayu.sessions.invocation import TaskExecutionSource
    from cayu.tasks.base import task_create_with_runtime_invocation
    from cayu.tasks.contracts import CompletionDecisionApplicationRequest, CompletionVerdict

    contract = await store.publish_work_contract(_contract())
    binding = unattributed_session_invocation_binding("graph-verification")
    root = task_create_with_runtime_invocation(
        TaskCreate(
            task_id="verified-root",
            type="verified-test",
            session_id="graph-verification",
            work_contract=contract.reference(),
        ),
        source=TaskExecutionSource.SDK_TASK,
        session_invocation=binding,
    )
    app = CayuApp(task_store=store, enable_logging=False)
    await app.create_task_group(
        TaskGroupCreate(
            group_id="verified-group",
            member_task_ids=("verified-root",),
            policy=TaskGroupPolicy(kind="all"),
            graph=TaskGraphCreate(
                graph_id="verified-graph",
                nodes=(
                    TaskGraphNode(task=root),
                    TaskGraphNode(
                        task=TaskCreate(task_id="verified-join", type="verified-test"),
                        prerequisite_task_ids=("verified-root",),
                    ),
                ),
            ),
        )
    )
    assert (await require_group(app, "verified-group")).status is TaskGroupStatus.PENDING
    running = await store.start_task("verified-root", session_invocation=binding)
    decision_id, reference = await _persist_decision(
        store, task=running, ordinal=1, verdict=CompletionVerdict.ACCEPTED
    )
    request = CompletionDecisionApplicationRequest(
        task_id=running.id,
        decision_id=decision_id,
        idempotency_key="graph-accepted",
        result=_result("1"),
        result_reference=reference,
    )
    completed = await app.apply_completion_decision(request)
    assert completed.status is TaskStatus.COMPLETED
    assert await app.apply_completion_decision(request) == completed
    claimed = await store.claim_task("verified-join-worker", query=TaskQuery(type="verified-test"))
    assert claimed is not None and claimed.id == "verified-join"
    events = await app.list_task_graph_events("verified-graph")
    assert (
        len(
            [
                event
                for event in events
                if event.type is TaskGraphEventType.READY and event.task_id == claimed.id
            ]
        )
        == 1
    )

    snapshot = await require_group(app, "verified-group")
    assert snapshot.status is TaskGroupStatus.SUCCEEDED
    assert snapshot.decision is not None
    assert snapshot.decision.successful_task_ids == ("verified-root",)
    events = await app.list_task_group_events("verified-group")
    assert sum(e.type is TaskGroupEventType.SUCCEEDED for e in events) == 1


async def test_retry_successor_cannot_change_group_decision(store: TaskStore) -> None:
    from cayu.tasks.base import (
        TaskRetryAttemptDisposition,
        TaskRetryPolicy,
        TaskRetrySettlementRequest,
    )

    await store.create_task_group(
        TaskGroupCreate(
            group_id="retry-group",
            member_task_ids=("retry-root",),
            policy=TaskGroupPolicy(kind="all"),
            graph=TaskGraphCreate(
                graph_id="retry-graph",
                nodes=(
                    TaskGraphNode(
                        task=TaskCreate(
                            task_id="retry-root",
                            type="retry-test",
                            retry_policy=TaskRetryPolicy(
                                max_attempts=2,
                                initial_backoff_seconds=0,
                            ),
                        )
                    ),
                    TaskGraphNode(
                        task=TaskCreate(task_id="retry-child", type="retry-test"),
                        prerequisite_task_ids=("retry-root",),
                    ),
                ),
            ),
        )
    )
    claimed = await store.claim_task("retry-worker", query=TaskQuery(type="retry-test"))
    assert claimed is not None and claimed.retry_series is not None
    receipt = await store.settle_task_retry_attempt(
        TaskRetrySettlementRequest(
            task_id=claimed.id,
            worker_id="retry-worker",
            lease_expires_at=claimed.lease_expires_at,
            causal_budget_id=claimed.retry_series.causal_budget_id,
            idempotency_key="retry-failure",
            disposition=TaskRetryAttemptDisposition.RETRYABLE_FAILURE,
            error={"code": "temporary"},
        )
    )
    assert receipt.successor is not None
    assert receipt.successor.graph_id is None
    child = await store.load_task("retry-child")
    assert child is not None and child.status is TaskStatus.DEPENDENCY_SKIPPED
    successor = await store.claim_task("successor-worker", query=TaskQuery(type="retry-test"))
    assert successor is not None and successor.id == receipt.successor.id
    assert successor.retry_series is not None
    await store.settle_task_retry_attempt(
        TaskRetrySettlementRequest(
            task_id=successor.id,
            worker_id="successor-worker",
            lease_expires_at=successor.lease_expires_at,
            causal_budget_id=successor.retry_series.causal_budget_id,
            idempotency_key="retry-success",
            disposition=TaskRetryAttemptDisposition.SUCCEEDED,
            result={"done": True},
        )
    )
    assert await store.load_task("retry-child") == child
    assert await store.claim_task("late-worker", query=TaskQuery(type="retry-test")) is None

    assert (await require_group(store, "retry-group")).status is TaskGroupStatus.FAILED
