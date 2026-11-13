"""Lost accepted gates remain actionable when their tool effects are known."""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest
from tests.runtime.test_orphaned_session_fast_reconciliation import _foreground_application
from tests.runtime.test_orphaned_session_reconciliation import _Crashing, _orphan, _request
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.approvals.user_input import UserInputResponse
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.server import ServerConfig, create_server
from cayu.sessions.queries import SessionQuery
from cayu.storage import open_application_stores
from cayu.tools.base import ToolResult


class _Recorded(_Crashing):
    async def run(self, ctx, args):
        return ToolResult(content="recorded")


def _crash_accepted_gate_at_known_effect_boundary():
    async def run():
        gate, boundary = os.environ["CAYU_ORPHAN_TEST_CRASH"].split(":")
        stores = open_application_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"],
            sqlite_path=os.environ["CAYU_ORPHAN_TEST_STORE"],
            event_watchers=False,
        )
        app = _foreground_application(
            stores, register_environment=True, gated=gate, tool=_Recorded()
        )
        async for _ in app.run(_request("parent")):
            pass
        [child] = (
            await stores.session_store.list_sessions(SessionQuery(parent_session_id="parent"))
        ).sessions
        checkpoint = await stores.session_store.load_checkpoint(child.id)
        if boundary == "before_dispatch":
            # Stop after the accepted continuation's SESSION_RESUMED but before
            # the effect owner admits/publishes its external tool dispatch.
            for method in ("prepare", "prepare_candidate", "prepare_exact_replay"):
                original = getattr(RuntimeEventWriter, method)

                def prepare(self, event, original=original):
                    if event.type is EventType.TOOL_CALL_STARTED and event.tool_name == "record":
                        os._exit(137)
                    return original(self, event)

                setattr(RuntimeEventWriter, method, prepare)
        else:
            observe = RuntimeEventWriter._observe_persisted_event

            def after_results(self, event, persisted_observer):
                observe(self, event, persisted_observer)
                if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == "record":
                    os._exit(137)

            RuntimeEventWriter._observe_persisted_event = after_results
        if gate == "input":
            resolution = app.resolve_user_input(
                UserInputResponse(
                    session_id=child.id,
                    input_id=checkpoint["pending_user_input"]["input_id"],
                    answer="continue",
                )
            )
        else:
            approval = checkpoint["pending_tool_approval"]
            resolution = app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=child.id,
                    approval_id=approval["approval_id"],
                    tool_round_id=approval["tool_round_id"],
                    tool_call_id=approval["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        async for _ in resolution:
            pass
        raise AssertionError("The continuation did not reach its crash boundary")

    asyncio.run(run())


@pytest.mark.parametrize("gate", ["approval", "input"])
@pytest.mark.parametrize("boundary", ["before_dispatch", "after_results"])
def test_lost_accepted_gate_remains_actionable_without_uncertain_effect(
    tmp_path, database_url, gate, boundary, monkeypatch
):
    path = tmp_path / "cayu.db"
    _orphan(
        path,
        "parent",
        database_url=database_url,
        crash=f"{gate}:{boundary}",
        entry="_crash_accepted_gate_at_known_effect_boundary",
        module=__name__,
    )

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        app = _foreground_application(
            stores, register_environment=False, gated=gate, tool=_Recorded()
        )
        try:
            [child] = (
                await stores.session_store.list_sessions(SessionQuery(parent_session_id="parent"))
            ).sessions
            assert child.status is SessionStatus.RUNNING
            before = await stores.session_store.load_checkpoint(child.id)
            key = "pending_tool_approval" if gate == "approval" else "pending_user_input"
            retained = before[key]
            before_events = await stores.session_store.load_events(child.id)
            assert any(event.type is EventType.SESSION_RESUMED for event in before_events)
            record_results = [
                event
                for event in before_events
                if event.type is EventType.TOOL_CALL_COMPLETED and event.tool_name == "record"
            ]
            assert len(record_results) == (1 if boundary == "after_results" else 0)

            async def forbidden(*args, **kwargs):
                raise AssertionError("Recovery must not redispatch the retained gate")

            monkeypatch.setattr(
                app._get_registered_agent("child").tools["record"].tool, "run", forbidden
            )
            monkeypatch.setattr(app._environment_lifecycle, "resolve_factory", forbidden)
            await app.reconcile_orphaned_sessions()
            current = await stores.session_store.load(child.id)
            assert current.status is SessionStatus.INTERRUPTED
            assert (await stores.session_store.load_checkpoint(child.id))[key] == retained
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://runtime.test"
            ) as client:
                response = await client.get("/api/pending-actions", params={"agent_name": "child"})
            assert response.status_code == 200
            assert response.json()["issues"] == []
            [action] = response.json()["actions"]
            assert action["kind"] == ("tool_approval" if gate == "approval" else "user_input")
            assert await app.reconcile_orphaned_sessions() == ()
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("damage", ["missing", "malformed", "missing_rebind", "missing_release"])
def test_recovered_input_stop_requires_exact_native_epoch_lineage(
    tmp_path, database_url, damage, monkeypatch
):
    from copy import deepcopy

    from tests.runtime.test_queued_retained_work_terminal_identity import _app, _stores

    from cayu.runtime import _session_interruption as interruption
    from cayu.sessions import InterruptSessionRequest
    from cayu.sessions._execution_profile_checkpoint import (
        active_invocation_execution_profile_from_checkpoint,
    )
    from cayu.sessions._invocation_lifecycle import (
        INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY,
        InvocationLifecycleCommandKind,
        _invocation_lifecycle_receipt_ledger_from_checkpoint,
        _InvocationLifecycleReceiptLedger,
    )
    from cayu.sessions.authority import SessionRunFenced

    path = tmp_path / "cayu.db"
    _orphan(
        path,
        "scan",
        database_url=database_url,
        module="tests.runtime.test_queued_retained_work_terminal_identity",
        entry="_crash_queued_approval",
        crash="input",
    )

    async def scenario():
        stores = _stores(database_url, path)
        app, provider, _ = _app(stores, gate="input")
        try:
            await app.reconcile_orphaned_sessions()
            before = await stores.session_store.load_checkpoint("scan")
            before_events = await stores.session_store.load_events("scan")
            session = await stores.session_store.load("scan")
            active = active_invocation_execution_profile_from_checkpoint(before)
            assert session.status is SessionStatus.INTERRUPTED
            assert session.run_epoch > before["pending_user_input"]["source_run_epoch"] + 1
            original_factory = (
                interruption._store_time_checkpoint_with_claimed_pending_session_interrupt
            )
            attempts = []

            def factory(*args, **kwargs):
                transform = original_factory(*args, **kwargs)

                def reject_incomplete_lineage(current, checkpoint, store_now):
                    damaged = deepcopy(checkpoint)
                    if damage == "missing":
                        damaged.pop(INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY)
                    elif damage == "malformed":
                        damaged[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = {
                            "schema_version": True
                        }
                    else:
                        ledger = _invocation_lifecycle_receipt_ledger_from_checkpoint(checkpoint)
                        remove_kind = (
                            InvocationLifecycleCommandKind.REBIND
                            if damage == "missing_rebind"
                            else InvocationLifecycleCommandKind.RELEASE
                        )
                        receipts = tuple(
                            receipt
                            for receipt in ledger.receipts
                            if not (
                                receipt.kind is remove_kind
                                and receipt.active_profile.run_epoch == active.run_epoch
                            )
                        )
                        assert len(receipts) < len(ledger.receipts), [
                            (receipt.kind.value, receipt.active_profile.run_epoch)
                            for receipt in ledger.receipts
                        ]
                        latest = max(
                            (
                                receipt
                                for receipt in receipts
                                if receipt.kind
                                in {
                                    InvocationLifecycleCommandKind.CREATE,
                                    InvocationLifecycleCommandKind.ADMIT,
                                    InvocationLifecycleCommandKind.REBIND,
                                }
                            ),
                            key=lambda receipt: receipt.active_profile.run_epoch,
                        )
                        reservation = (
                            None
                            if any(
                                receipt.kind is InvocationLifecycleCommandKind.RELEASE
                                and receipt.active_profile == latest.active_profile
                                for receipt in receipts
                            )
                            else latest.command_identity
                        )
                        damaged[INVOCATION_LIFECYCLE_RECEIPT_CHECKPOINT_KEY] = (
                            _InvocationLifecycleReceiptLedger(
                                receipts=receipts,
                                release_capacity_command_identity=reservation,
                            ).model_dump(mode="json")
                        )
                    attempts.append(damage)
                    return transform(current, damaged, store_now)

                return reject_incomplete_lineage

            monkeypatch.setattr(
                interruption,
                "_store_time_checkpoint_with_claimed_pending_session_interrupt",
                factory,
            )
            with pytest.raises((SessionRunFenced, RuntimeError)):
                async for _ in app.interrupt_session(
                    InterruptSessionRequest(session_id="scan", reason="stop recovered input")
                ):
                    pass
            assert attempts
            after = await stores.session_store.load_checkpoint("scan")
            assert after["pending_user_input"] == before["pending_user_input"]
            assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
            assert await stores.session_store.load_events("scan") == before_events
            assert provider.calls == 0
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())
