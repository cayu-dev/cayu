"""Lost queued retained work keeps the queue's exact terminal identity."""

from __future__ import annotations

import asyncio
import base64
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import _Crashing, _orphan, _request, _slot
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import AgentSpec, CayuApp, Event, EventType, Message, ResumeRequest, SessionStatus
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InterruptSessionRequest,
)
from cayu.sessions._pending_approval_reader import pending_approval_from_checkpoint
from cayu.sessions._terminal_evidence import _session_run_operation_from_checkpoint
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate, TaskStatus, reconcile_abandoned_tasks
from cayu.tasks.dispatch import (
    DispatchRequest,
    DispatchStatus,
    InlineDispatcher,
    TaskStoreDispatcher,
    _existing_queued_dispatch_envelope,
    _QueuedDispatchSettlementState,
)
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy
from cayu.tools.subagents import SubagentSpec, SubagentTool
from cayu.tools.user_input import UserInputTool
from cayu.vaults.redaction import SecretRedactor


def _stores(database_url, path):
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("="))
            },
        )
    )
    return open_application_stores(
        database_url, sqlite_path=path, event_watchers=False, public_authority_alias_codec=codec
    )


class _QueuedApprovalModel(VersionedFakeProvider):
    def __init__(self, *, producing=False, fail=False, gate="approval"):
        super().__init__([])
        self.producing = producing
        self.calls = 0
        self.fail = fail
        self.gate = gate

    async def stream(self, request):
        self.calls += 1
        if not self.producing:
            raise AssertionError("Lost retained work must not redispatch its model")
        if self.calls == 1:
            yield ModelStreamEvent.text_delta("initial response")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
        elif self.fail:
            yield ModelStreamEvent.error("queued linked task provider failure")
            yield ModelStreamEvent.completed({"finish_reason": "error"})
        elif self.gate == "foreground" and self.calls == 2:
            yield ModelStreamEvent.tool_call(
                id="spawn", name="subagent", arguments={"agent": "child", "task": "ask"}
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        elif self.gate in {"input", "foreground"}:
            yield ModelStreamEvent.tool_call(
                id="question", name="ask_user", arguments={"question": "Continue?"}
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
        else:
            yield ModelStreamEvent.tool_call(id="approval-call", name="record", arguments={})
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


def _app(
    stores, *, producing=False, fail=False, secret_redactor=None, inline=False, gate="approval"
):
    dispatcher = (
        InlineDispatcher() if inline else TaskStoreDispatcher(stores.task_store, lease_seconds=1)
    )
    app = CayuApp(
        session_store=stores.session_store,
        task_store=stores.task_store,
        dispatcher=dispatcher,
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    provider = _QueuedApprovalModel(producing=producing, fail=fail, gate=gate)
    app.register_provider(provider, default=True)
    tools = [_Crashing()]
    if gate == "input":
        tools.append(UserInputTool())
    if gate == "foreground":
        from tests.core.test_foreground_subagent_recovery import _identity

        tools.append(
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("queued-lost-child"),
            )
        )
        app.register_agent(AgentSpec(name="child", model="fake-model"), tools=[UserInputTool()])
    app.register_agent(
        AgentSpec(name="scanner", model="fake-model"),
        tools=tools,
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=["record"]),
    )
    slot = _slot()
    app.register_environment_factory(slot.environment.spec, slot)
    return app, provider, dispatcher


def _crash_queued_approval():
    async def scenario():
        stores = _stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            os.environ["CAYU_ORPHAN_TEST_STORE"],
        )
        app, _, dispatcher = _app(stores, producing=True, gate=os.environ["CAYU_ORPHAN_TEST_CRASH"])
        async for _ in app.run(_request("scan")):
            pass
        await app.dispatch(
            DispatchRequest(
                session_id="scan",
                dispatch_id="queued-approval",
                messages=[Message.text("user", "queued request")],
            )
        )
        fanout = app._event_writer.fan_out_persisted

        async def crash_after_approval(events, **kwargs):
            result = await fanout(events, **kwargs)
            if any(
                event.type is EventType.INTERACTION_PAUSED and event.session_id == "scan"
                for event in events
            ):
                os._exit(137)
            return result

        app._event_writer.fan_out_persisted = crash_after_approval
        await dispatcher.process_next(app, worker_id="original")
        raise AssertionError("Queued work did not reach its retained approval")

    asyncio.run(scenario())


def _crash_after_queued_linked_task_failure():
    async def scenario():
        stores = _stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            os.environ["CAYU_ORPHAN_TEST_STORE"],
        )
        app, _, dispatcher = _app(stores, producing=True, fail=True)
        async for _ in app.run(_request("scan")):
            pass
        await app.create_task(TaskCreate(task_id="linked", type="job", session_id="scan"))
        submitted = await app.dispatch(
            DispatchRequest(
                session_id="scan",
                dispatch_id="queued-failure",
                task_id="linked",
                messages=[Message.text("user", "queued request")],
            )
        )
        Path(os.environ["CAYU_ORPHAN_TEST_STORE"] + ".queue").write_text(
            submitted.metadata["queue_task_id"]
        )
        fail_task = app._session_finalization.fail_task
        loss_point = os.environ["CAYU_ORPHAN_TEST_CRASH"]
        elect = app._session_outcomes.claim_invocation_terminal_decision

        async def crash_after_failure_election(**kwargs):
            await elect(**kwargs)
            if loss_point == "decision":
                os._exit(137)
            if loss_point == "cancelled_terminal":
                await stores.task_store.cancel_task("linked")

        app._session_outcomes.claim_invocation_terminal_decision = crash_after_failure_election

        async def crash_after_task_commit(**kwargs):
            result = await fail_task(**kwargs)
            if (
                loss_point == "task"
                and result.id == "linked"
                and result.status is TaskStatus.FAILED
            ):
                os._exit(137)
            return result

        app._session_finalization.fail_task = crash_after_task_commit
        fanout = app._event_writer.fan_out_persisted

        async def crash_after_terminal(events, **kwargs):
            result = await fanout(events, **kwargs)
            if loss_point in {"terminal", "cancelled_terminal"} and any(
                e.type is EventType.SESSION_FAILED for e in events
            ):
                os._exit(137)
            return result

        app._event_writer.fan_out_persisted = crash_after_terminal
        await dispatcher.process_next(app, worker_id="original")
        if loss_point == "finished":
            assert (await stores.session_store.load("scan")).status is SessionStatus.FAILED
            os._exit(137)
        raise AssertionError("Queued linked task did not reach its durable failure")

    asyncio.run(scenario())


@pytest.mark.parametrize("loss_point", ["decision", "task", "terminal", "finished"])
@pytest.mark.parametrize("redact", [False, True])
def test_queued_linked_task_failure_restart_keeps_terminal_identity(
    tmp_path, database_url, loss_point, redact
):
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        database_url=database_url,
        module=__name__,
        entry="_crash_after_queued_linked_task_failure",
        crash=loss_point,
    )

    async def scenario():
        stores = _stores(database_url, tmp_path / "cayu.db")
        from cayu.sessions._terminal_evidence import (
            _queued_dispatch_terminal_receipts_from_checkpoint,
        )

        checkpoint = await stores.session_store.load_checkpoint("scan")
        operation = _session_run_operation_from_checkpoint(checkpoint)
        receipt = next(
            iter(_queued_dispatch_terminal_receipts_from_checkpoint(checkpoint).values()), None
        )
        assert operation is not None or receipt is not None or loss_point == "finished"
        queue_task_id = (tmp_path / "cayu.db.queue").read_text()
        queue_task = await stores.task_store.load_task(queue_task_id)
        envelope = _existing_queued_dispatch_envelope(queue_task, task_type=queue_task.type)
        assert envelope is not None
        operation_id = envelope.dispatch_operation_id
        expected_terminal_id = envelope.terminal_event_id
        app, provider, dispatcher = _app(
            stores, fail=True, secret_redactor=SecretRedactor(operation_id[:8]) if redact else None
        )
        try:
            linked = await stores.task_store.load_task("linked")
            assert linked.status is (
                TaskStatus.RUNNING if loss_point == "decision" else TaskStatus.FAILED
            )
            assert (await stores.session_store.load("scan")).status is (
                SessionStatus.RUNNING
                if loss_point in {"decision", "task"}
                else SessionStatus.FAILED
            )
            if redact:
                await app.emit_events(
                    "scan",
                    [
                        Event(
                            type="custom.raw_operation_probe",
                            session_id="scan",
                            payload={"session_run_operation_id": operation_id},
                        )
                    ],
                )
                raw = [
                    e
                    for e in await stores.session_store.load_events("scan")
                    if e.type == "custom.raw_operation_probe"
                ]
                assert len(raw) == 1 and raw[0].payload["session_run_operation_id"] != operation_id
            failures = []
            recover = app._session_recovery.recover_incomplete_session

            async def observe_recovery(*args, **kwargs):
                try:
                    return await recover(*args, **kwargs)
                except Exception as exc:
                    failures.append(exc)
                    raise

            app._session_recovery.recover_incomplete_session = observe_recovery
            await app.reconcile_orphaned_sessions()
            if failures:
                raise failures[0]
            session = await stores.session_store.load("scan")
            assert session.status is SessionStatus.FAILED
            events = await stores.session_store.load_events("scan")
            terminal = [e for e in events if e.type is EventType.SESSION_FAILED]
            assert len(terminal) == 1
            assert terminal[0].id == expected_terminal_id
            assert terminal[0].payload["session_run_operation_id"] == operation_id
            settlement = await app._queued_dispatch_settlement_state(envelope)
            assert settlement.state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE, {
                "epoch": session.run_epoch,
                "keys": list((await stores.session_store.load_checkpoint("scan")).keys()),
            }
            assert settlement.terminal_status is DispatchStatus.FAILED
            if queue_task.lease_expires_at is not None:
                async with asyncio.timeout(10):
                    while queue_task.lease_expires_at > datetime.now(UTC):
                        await asyncio.sleep(0.02)
            await stores.task_store.reclaim_expired()
            await reconcile_abandoned_tasks(
                stores.task_store,
                live_worker_ids=["replacement"],
                validator_id="tests.exited-queued-process",
                validator_version="1",
            )
            assert await dispatcher.process_next(app, worker_id="replacement") is None
            queue_task = await stores.task_store.load_task(queue_task_id)
            assert queue_task.status is (
                TaskStatus.FAILED if loss_point == "finished" else TaskStatus.CANCELLED
            )
            assert not _queued_dispatch_terminal_receipts_from_checkpoint(
                await stores.session_store.load_checkpoint("scan")
            )
            replay = [
                e
                async for e in app.resume(
                    ResumeRequest(
                        session_id="scan", messages=[Message.text("user", "retry failure")]
                    )
                )
            ]
            assert any(e.type is EventType.SESSION_FAILED for e in replay)
            assert await stores.session_store.load_events("scan") == events
            assert provider.calls == 0
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("gate", ["approval", "input", "foreground"])
def test_queued_lost_gate_interruption_keeps_terminal_identity(tmp_path, database_url, gate):
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        database_url=database_url,
        module=__name__,
        entry="_crash_queued_approval",
        crash=gate,
    )

    async def scenario():
        stores = _stores(database_url, tmp_path / "cayu.db")
        app, provider, dispatcher = _app(stores, gate=gate)
        try:
            checkpoint = await stores.session_store.load_checkpoint("scan")
            operation = _session_run_operation_from_checkpoint(checkpoint)
            assert operation is not None
            if gate == "approval":
                assert pending_approval_from_checkpoint(checkpoint) is not None
            task = await stores.task_store.load_task(operation.queue_task_id)
            envelope = _existing_queued_dispatch_envelope(task, task_type=task.type)
            assert envelope is not None
            await app.reconcile_orphaned_sessions()
            settlement = await app._queued_dispatch_settlement_state(envelope)
            assert settlement.state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE
            assert settlement.terminal_status is DispatchStatus.INTERRUPTED
            if task.lease_expires_at is not None:
                async with asyncio.timeout(10):
                    while task.lease_expires_at > datetime.now(UTC):
                        await asyncio.sleep(0.02)
            await stores.task_store.reclaim_expired()
            await reconcile_abandoned_tasks(
                stores.task_store,
                live_worker_ids=["replacement"],
                validator_id="tests.exited-queued-process",
                validator_version="1",
            )
            assert await dispatcher.process_next(app, worker_id="replacement") is None
            assert (
                await stores.task_store.load_task(operation.queue_task_id)
            ).status is TaskStatus.CANCELLED
            from cayu.sessions._terminal_evidence import (
                _queued_dispatch_terminal_receipts_from_checkpoint,
            )

            assert not _queued_dispatch_terminal_receipts_from_checkpoint(
                await stores.session_store.load_checkpoint("scan")
            )
            events = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="scan", reason="stop lost gated work")
                )
            ]
            assert any(event.type is EventType.SESSION_INTERRUPTED for event in events)
            assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
            terminal = [
                event
                for event in await stores.session_store.load_events("scan")
                if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert len(terminal) == 2
            assert terminal[-1].payload["reason"] == "stop lost gated work"
            assert terminal[0].id == operation.terminal_event_id
            assert terminal[0].payload["session_run_operation_id"] == operation.operation_id
            assert provider.calls == 0
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_direct_resumed_linked_failure_keeps_ordinary_terminal_identity(tmp_path, database_url):
    async def scenario():
        stores = _stores(database_url, tmp_path / "cayu.db")
        app, provider, _ = _app(stores, producing=True, fail=True, inline=True)
        observed = []
        fail_task = app._session_finalization.fail_task

        async def observe_direct_operation(**kwargs):
            operation = _session_run_operation_from_checkpoint(
                await stores.session_store.load_checkpoint("scan")
            )
            assert operation is not None
            assert operation.terminal_event_id is None and operation.queue_task_id is None
            observed.append(operation.operation_id)
            return await fail_task(**kwargs)

        app._session_finalization.fail_task = observe_direct_operation
        try:
            async for _ in app.run(_request("scan")):
                pass
            await app.create_task(TaskCreate(task_id="linked", type="job", session_id="scan"))
            handle = await app.dispatch(
                DispatchRequest(
                    session_id="scan",
                    task_id="linked",
                    messages=[Message.text("user", "direct request")],
                )
            )
            assert handle.status is DispatchStatus.FAILED
            task = await stores.task_store.load_task("linked")
            assert task.status is TaskStatus.FAILED and len(observed) == 1
            marker = task.error["runtime_task_failure"]
            assert marker["session_terminal_event_id"] == marker["failure_id"] + ":session_failed"
            assert marker["session_failure_payload"]["session_run_operation_id"] == observed[0]
            events = await stores.session_store.load_events("scan")
            terminal = [e for e in events if e.type is EventType.SESSION_FAILED]
            assert len(terminal) == 1 and terminal[0].id == marker["session_terminal_event_id"]
            replay = [
                e
                async for e in app.resume(
                    ResumeRequest(
                        session_id="scan", messages=[Message.text("user", "retry failure")]
                    )
                )
            ]
            assert any(e.type is EventType.SESSION_FAILED for e in replay)
            assert await stores.session_store.load_events("scan") == events
            assert provider.calls == 2
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "entrance", ["sweep", "public", "public_default_inactivity", "retained_cleanup"]
)
def test_linked_failure_recovery_preserves_a_live_terminal_publisher(
    tmp_path, database_url, entrance
):
    async def scenario():
        stores = _stores(database_url, tmp_path / "cayu.db")
        owner, _, dispatcher = _app(stores, producing=True, fail=True)
        observer, _, _ = _app(stores, fail=True)
        entered, release = asyncio.Event(), asyncio.Event()
        fanout = owner._event_writer.fan_out_persisted

        async def hold_terminal(events, **kwargs):
            if any(e.type is EventType.SESSION_FAILED for e in events):
                entered.set()
                await release.wait()
            return await fanout(events, **kwargs)

        owner._event_writer.fan_out_persisted = hold_terminal
        async for _ in owner.run(_request("scan")):
            pass
        await owner.create_task(TaskCreate(task_id="linked", type="job", session_id="scan"))
        await owner.dispatch(
            DispatchRequest(
                session_id="scan",
                task_id="linked",
                messages=[Message.text("user", "queued request")],
            )
        )
        task = asyncio.create_task(dispatcher.process_next(owner, worker_id="original"))
        try:
            await asyncio.wait_for(entered.wait(), 30)
            before = await stores.session_store.load("scan")
            assert before.status is SessionStatus.FAILED
            if entrance == "retained_cleanup":
                # The owner's run has returned and only its environment cleanup still
                # runs in this process, holding the execution presence.
                owner._session_control.has_active_tasks = lambda *_args, **_kwargs: False
                owner._environment_lifecycle._deferred_factory_cleanup_tasks["scan"] = (
                    asyncio.get_running_loop().create_future()
                )
            checkpoint = await stores.session_store.load_checkpoint("scan")
            events = await stores.session_store.load_events("scan")
            if entrance == "sweep":
                await observer.reconcile_orphaned_sessions()
            else:
                # Public recovery reads the published failure as terminal; it must
                # still leave the owner that is running its failure hooks alone.
                result = await observer.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="scan",
                        inactive_for_seconds=None if entrance == "public_default_inactivity" else 0,
                    )
                )
                skipped = (
                    IncompleteSessionRecoveryAction.SKIPPED_ACTIVE
                    if entrance == "retained_cleanup"
                    else IncompleteSessionRecoveryAction.SKIPPED_EXECUTION_OWNER
                )
                assert result.actions == (skipped,)
            assert await stores.session_store.load("scan") == before
            assert await stores.session_store.load_checkpoint("scan") == checkpoint
            assert await stores.session_store.load_events("scan") == events
            assert not task.done()
        finally:
            owner._environment_lifecycle._deferred_factory_cleanup_tasks.pop("scan", None)
            vars(owner._session_control).pop("has_active_tasks", None)
            release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 30)
            await owner.aclose()
            await observer.aclose()
            await stores.close()

    asyncio.run(scenario())


def test_cancelled_queued_linked_failure_releases_and_acknowledges(tmp_path, database_url):
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        database_url=database_url,
        module=__name__,
        entry="_crash_after_queued_linked_task_failure",
        crash="cancelled_terminal",
    )

    async def scenario():
        from cayu.sessions._invocation_terminal_decision import (
            invocation_terminal_decision_from_checkpoint,
            settled_invocation_terminal_decision_from_checkpoint,
        )
        from cayu.sessions._terminal_evidence import (
            _queued_dispatch_terminal_receipts_from_checkpoint,
        )

        stores = _stores(database_url, tmp_path / "cayu.db")
        app, provider, dispatcher = _app(stores, fail=True)
        try:
            checkpoint = await stores.session_store.load_checkpoint("scan")
            assert invocation_terminal_decision_from_checkpoint(checkpoint) is None
            assert settled_invocation_terminal_decision_from_checkpoint(checkpoint) is not None
            linked = await stores.task_store.load_task("linked")
            assert linked.status is TaskStatus.CANCELLED
            queue_task_id = (tmp_path / "cayu.db.queue").read_text()
            queued = await stores.task_store.load_task(queue_task_id)
            envelope = _existing_queued_dispatch_envelope(queued, task_type=queued.type)
            assert envelope is not None
            before_events = await stores.session_store.load_events("scan")
            terminal = [event for event in before_events if event.type is EventType.SESSION_FAILED]
            assert len(terminal) == 1 and terminal[0].id == envelope.terminal_event_id
            assert terminal[0].payload["session_run_operation_id"] == envelope.dispatch_operation_id
            await app.reconcile_orphaned_sessions()
            settlement = await app._queued_dispatch_settlement_state(envelope)
            assert settlement.state is _QueuedDispatchSettlementState.TERMINAL_EVIDENCE_DURABLE
            assert settlement.terminal_status is DispatchStatus.FAILED
            if queued.lease_expires_at is not None:
                async with asyncio.timeout(10):
                    while queued.lease_expires_at > datetime.now(UTC):
                        await asyncio.sleep(0.02)
            await stores.task_store.reclaim_expired()
            await reconcile_abandoned_tasks(
                stores.task_store,
                live_worker_ids=["replacement"],
                validator_id="tests.exited-queued-process",
                validator_version="1",
            )
            assert await dispatcher.process_next(app, worker_id="replacement") is None
            assert (await stores.task_store.load_task(queue_task_id)).status is TaskStatus.CANCELLED
            assert not _queued_dispatch_terminal_receipts_from_checkpoint(
                await stores.session_store.load_checkpoint("scan")
            )
            assert await stores.task_store.load_task("linked") == linked
            assert await stores.session_store.load_events("scan") == before_events
            assert provider.calls == 0
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())
