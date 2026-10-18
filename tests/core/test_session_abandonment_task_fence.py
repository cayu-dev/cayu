"""Abandonment freezes task admission under the stores' existing closure owner."""

from __future__ import annotations

import asyncio
import sqlite3
from uuid import uuid4

import pytest
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_abandoned_session_resume import _child, _crash_child, _Proposal
from tests.core.test_session_execution_presence import _app, _stores
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import SessionStatus, ToolEffect
from cayu.runtime.session_closure import SessionClosurePolicy, TaskSessionClosureStore
from cayu.sessions.records import SessionStatusConflict
from cayu.storage import PostgresTaskStore, SQLiteTaskStore
from cayu.storage.migrations import SchemaMode
from cayu.tasks import InMemoryTaskStore, TaskCreate, TaskQuery, TaskStatus
from cayu.tasks.graphs import TaskGraphConflict, TaskGraphCreate, TaskGraphNode


@pytest.mark.store_semantics
@pytest.mark.parametrize("task_backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("cross_session_graph", [False, True])
def test_public_abandonment_rechecks_and_fences_concurrent_task_admission(
    task_backend, request, sqlite_resources, tmp_path, monkeypatch, cross_session_graph
):
    dsn = request.getfixturevalue("postgres_dsn") if task_backend == "postgres" else None
    if task_backend == "postgres":
        from psycopg.errors import CheckViolation

        guard_error = CheckViolation
    else:
        guard_error = ValueError if task_backend == "memory" else sqlite3.IntegrityError

    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (sessions, reopen):
            sid, original_id, late_id, unattached_id = (str(uuid4()) for _ in range(4))
            if not cross_session_graph:
                original_id += "x" * 257
            receipt = tmp_path / "receipt"
            await _crash_child(
                _child(sessions, "sqlite", request, receipt, sid, ToolEffect.EXTERNAL, "crash")
            )
            observer = reopen()
            if task_backend == "memory":
                tasks = competitor = InMemoryTaskStore()
            elif task_backend == "sqlite":
                tasks = SQLiteTaskStore(tmp_path / "tasks.db")
                competitor = SQLiteTaskStore(tmp_path / "tasks.db")
            else:
                tasks = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
                competitor = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
            try:
                original = TaskCreate(type=original_id, task_id=original_id, session_id=sid)
                other_id = str(uuid4())
                if cross_session_graph:
                    await tasks.create_task_graph(
                        TaskGraphCreate(
                            graph_id=str(uuid4()),
                            nodes=(
                                TaskGraphNode(task=original),
                                TaskGraphNode(
                                    task=TaskCreate(
                                        type="other", task_id=other_id, session_id=other_id
                                    )
                                ),
                            ),
                        )
                    )
                else:
                    await tasks.create_task(original)
                await tasks.cancel_task(original_id)
                await competitor.create_task(TaskCreate(type=unattached_id, task_id=unattached_id))
                unattached = await competitor.claim_task("owner", TaskQuery(type=unattached_id))
                binding = await task_backed_session_invocation(competitor, unattached_id, sid)
                provider = VersionedFakeProvider([])
                app = _app(
                    observer,
                    provider,
                    tools=[_Proposal(ToolEffect.EXTERNAL, receipt)],
                    task_store=tasks,
                )
                original_claim = tasks.claim_session_closure
                raced = False

                async def race(claim):
                    nonlocal raced
                    if not raced:
                        raced = True
                        await competitor.create_task(
                            TaskCreate(type=late_id, task_id=late_id, session_id=sid)
                        )
                    return await original_claim(claim)

                monkeypatch.setattr(tasks, "claim_session_closure", race)
                with pytest.raises(SessionStatusConflict, match="task set changed"):
                    await app.abandon_session(sid)
                assert raced
                assert (await observer.load(sid)).status is SessionStatus.INTERRUPTED
                assert "pending_tool_round" in await observer.load_checkpoint(sid)
                assert (await competitor.load_task(late_id)).status is TaskStatus.PENDING
                assert await tasks.load_session_closure_claim(sid) is None
                await competitor.cancel_task(late_id)
                result = await app.abandon_session(sid)
                assert result.status is SessionStatus.FAILED
                claim = await competitor.load_session_closure_claim(sid)
                assert set(claim.task_ids) == {original_id, late_id}
                assert claim.purpose == "retention"
                with pytest.raises(guard_error, match="owned by closure"):
                    await competitor.create_task(TaskCreate(type="late", session_id=sid))
                with pytest.raises(guard_error, match="owned by closure"):
                    await competitor.attach_task(
                        unattached_id,
                        session_id=sid,
                        session_invocation=binding,
                        worker_id="owner",
                        lease_expires_at=unattached.lease_expires_at,
                    )
                assert (await competitor.load_task(unattached_id)).session_id is None
                assert not provider.requests
                # Later explicit erasure can use the already-frozen task set;
                # abandonment does not prevent legitimate closure cleanup.
                if cross_session_graph:
                    assert (await competitor.load_task(other_id)).status is TaskStatus.PENDING
                    with pytest.raises(TaskGraphConflict):
                        await tasks.delete_session_tasks(sid, task_ids=claim.task_ids, policy=None)
                    # The independent graph member can still complete; deletion
                    # becomes safe only when its retained graph is terminal.
                    await competitor.complete_task(other_id, {})
                await TaskSessionClosureStore(tasks).erase_session_closure(
                    sid, policy=SessionClosurePolicy(), plan_id="a" * 64
                )
                assert await tasks.load_task(original_id) is None
                assert await tasks.load_session_closure_claim(sid) == claim
            finally:
                if task_backend != "memory":
                    await competitor.close()
                    await tasks.close()

    asyncio.run(scenario())


def _receipt_app(store, provider, receipt, *, task_store=None, crash=False):
    from datetime import UTC, datetime

    from tests.core.test_tool_effect_reconciliation_registration import _spec

    from cayu import AgentSpec, CayuApp
    from cayu.runtime.tool_effects import (
        ToolEffectReceipt,
        ToolEffectReconciliationRegistration,
        ToolEffectReconciliationResult,
    )
    from cayu.sessions import SessionExecutionConfig

    class Reconciler:
        async def reconcile(self, *, context, receipt):
            assert receipt is None
            assert (
                context.idempotency_key
                in receipt_path.with_suffix(".calls").read_text().splitlines()
            )
            return ToolEffectReconciliationResult(
                outcome="completed",
                observation="sent",
                receipt=ToolEffectReceipt(
                    receipt_id="verified-proposal",
                    receipt_schema="deployment",
                    receipt_schema_version=1,
                    tool_call_id=context.tool_call_id,
                    tool_name=context.tool_name,
                    idempotency_key=context.idempotency_key,
                    outcome="completed",
                    message="Verified committed proposal",
                    source="reconciler",
                    observed_at=datetime.now(UTC),
                ),
            )

    receipt_path = receipt
    app = CayuApp(
        session_store=store,
        task_store=task_store,
        enable_logging=False,
        session_execution=SessionExecutionConfig(
            heartbeat_interval_seconds=0.05, lease_seconds=1.5
        ),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model"),
        tools=[_Proposal(ToolEffect.EXTERNAL, receipt, mode="crash" if crash else "normal")],
        tool_effect_reconcilers={
            "propose": ToolEffectReconciliationRegistration(
                reconciler=Reconciler(),
                spec=_spec(supports_lookup=True, integrity_fields=(), resource_version_fields=()),
            )
        },
    )
    return app


def _receipt_child(path, receipt, sid):
    from pathlib import Path

    from cayu import Message, RunRequest
    from cayu.providers.base import ModelStreamEvent
    from cayu.storage import SQLiteSessionStore

    async def scenario():
        app = _receipt_app(
            SQLiteSessionStore(path),
            VersionedFakeProvider(
                [
                    ModelStreamEvent.tool_call(
                        id="proposal", name="propose", arguments={"body": "new"}
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
            ),
            Path(receipt),
            crash=True,
        )
        async for _ in app.run(
            RunRequest(
                agent_name="assistant", session_id=sid, messages=[Message.text("user", "propose")]
            )
        ):
            pass

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("task_backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("failure_mode", ["write_failed", "crashed", "committed_ack_lost"])
def test_abandonment_fence_tracks_the_durable_session_decision(
    task_backend, failure_mode, request, sqlite_resources, tmp_path, monkeypatch
):
    import subprocess
    import sys

    from cayu import EventType
    from cayu.providers.base import ModelStreamEvent
    from cayu.runtime._tool_effect_state import ToolEffectStateOwner
    from cayu.runtime.tool_effects import ToolEffectReconciliationRequest

    dsn = request.getfixturevalue("postgres_dsn") if task_backend == "postgres" else None

    async def scenario():
        async with _stores("sqlite", request, sqlite_resources) as (sessions, reopen):
            sid, original_id = (str(uuid4()) for _ in range(2))
            receipt = tmp_path / "failure-receipt"
            await _crash_child(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        "import sys; from tests.core.test_session_abandonment_task_fence import _receipt_child; _receipt_child(*sys.argv[1:])",
                        str(sessions.path),
                        str(receipt),
                        sid,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
            observer = reopen()
            if task_backend == "memory":
                tasks = InMemoryTaskStore()
            elif task_backend == "sqlite":
                tasks = SQLiteTaskStore(tmp_path / "failure-tasks.db")
            else:
                tasks = PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
            try:
                await tasks.create_task(TaskCreate(type="job", task_id=original_id, session_id=sid))
                await tasks.cancel_task(original_id)
                provider = VersionedFakeProvider(
                    [ModelStreamEvent.completed({"finish_reason": "stop"})]
                )
                app = _receipt_app(observer, provider, receipt, task_store=tasks)
                transition = app._runtime_session_store.transition_status_and_checkpoint
                advance = tasks.advance_session_retention_epoch
                original_claim = tasks.claim_session_closure
                claims = []

                async def remember_claim(claim):
                    result = await original_claim(claim)
                    claims.append(result)
                    return result

                async def fail_after_task_fence(*args, **kwargs):
                    if kwargs.get("to_status") is SessionStatus.FAILED:
                        assert await tasks.load_session_closure_claim(sid) is not None
                        if failure_mode == "committed_ack_lost":
                            await transition(*args, **kwargs)
                        raise RuntimeError("injected abandonment session write failure")
                    return await transition(*args, **kwargs)

                async def crash_before_release(*args, **kwargs):
                    if claims and failure_mode == "crashed":
                        raise RuntimeError("process lost before fence release")
                    return await advance(*args, **kwargs)

                monkeypatch.setattr(tasks, "claim_session_closure", remember_claim)
                monkeypatch.setattr(tasks, "advance_session_retention_epoch", crash_before_release)
                monkeypatch.setattr(
                    app._runtime_session_store,
                    "transition_status_and_checkpoint",
                    fail_after_task_fence,
                )
                with pytest.raises(
                    RuntimeError, match="injected abandonment session write failure"
                ):
                    await app.abandon_session(sid)
                monkeypatch.setattr(tasks, "advance_session_retention_epoch", advance)
                monkeypatch.setattr(
                    app._runtime_session_store, "transition_status_and_checkpoint", transition
                )
                assert len(claims) == 1
                checkpoint = await observer.load_checkpoint(sid)
                if failure_mode == "committed_ack_lost":
                    assert (await observer.load(sid)).status is SessionStatus.FAILED
                    assert "abandoned_session" in checkpoint
                    assert await tasks.load_session_closure_claim(sid) == claims[0]
                    await app.abandon_session(sid)
                    assert await tasks.load_session_closure_claim(sid) == claims[0]
                    return
                assert (await observer.load(sid)).status is SessionStatus.INTERRUPTED
                assert "abandoned_session" not in checkpoint
                assert (await tasks.load_session_closure_claim(sid) is not None) == (
                    failure_mode == "crashed"
                )
                # A fresh app must recover a crash between stores without relying
                # on the failed process's cleanup callback or in-memory state.
                recovered_app = _receipt_app(observer, provider, receipt, task_store=tasks)
                session = await observer.load(sid)
                effect = await ToolEffectStateOwner(
                    recovered_app._runtime_session_store
                ).resolve_call(
                    session,
                    tool_round_id=checkpoint["pending_tool_round"]["tool_round_id"],
                    tool_call_id="proposal",
                )
                recovered = [
                    event
                    async for event in recovered_app.reconcile_tool_effect(
                        ToolEffectReconciliationRequest(
                            **{
                                name: getattr(effect.intent, name)
                                for name in (
                                    "session_id",
                                    "session_instance_id",
                                    "tool_round_id",
                                    "tool_call_id",
                                    "tool_name",
                                    "idempotency_key",
                                )
                            },
                            expected_run_epoch=session.run_epoch,
                            expected_revision=effect.revision,
                            lookup=True,
                        )
                    )
                ]
                assert recovered[-1].type is EventType.SESSION_COMPLETED
                assert await tasks.load_session_closure_claim(sid) is None
                # The old process cannot commit its delayed claim after the new
                # admission, including after the namespace becomes empty again.
                with pytest.raises(ValueError, match="superseded invocation"):
                    await original_claim(claims[0])
                created = await tasks.create_task(TaskCreate(type="continued", session_id=sid))
                assert created.status is TaskStatus.PENDING
                assert receipt.with_suffix(".calls").read_text().count("\n") == 1
            finally:
                if task_backend != "memory":
                    await tasks.close()

    asyncio.run(scenario())
