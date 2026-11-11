"""Lost-owner sweeps retain profile authority and expose truthful continuation boundaries."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import httpx
import pytest
from tests.core.test_abandoned_session_resume import _runtime_build
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_fast_reconciliation import _fresh_orphan
from tests.runtime.test_orphaned_session_reconciliation import _TOOL, _application, _slot
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import EventType, SessionStatus, Tool
from cayu.build_provenance import RUNTIME_BUILD_PROVENANCE_ENV, current_runtime_build_provenance
from cayu.runtime._invocation_lifecycle import InvocationContext
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import ExecutionProfileMismatchError
from cayu.server import ServerConfig, create_server
from cayu.sessions._model_stage import (
    ModelCompletionManualRecoveryRequest,
    SessionModelCompletionStageConflict,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.storage import open_application_stores
from cayu.tasks import TaskCreate


class _ChangedProvider(VersionedFakeProvider):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:runtime:versioned-fake-provider",
            behavior_version="2",
            implementation_version="2",
        )


class _NoDispatch(Tool):
    spec = _TOOL

    def __init__(self):
        self.calls = 0

    async def run(self, ctx, args):
        self.calls += 1
        raise AssertionError("A reconciliation sweep must not redispatch a tool")


async def _http_boundary(app):
    server = create_server(app, config=ServerConfig.local_development())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server), base_url="http://runtime.test"
    ) as client:
        sessions = await client.get("/api/sessions", params={"include": "execution_profile"})
        actions = await client.get("/api/pending-actions")
    assert sessions.status_code == actions.status_code == 200
    [session] = sessions.json()["sessions"]
    [profile] = sessions.json()["execution_profiles"]
    return session, profile, actions.json()


@pytest.mark.parametrize("crash", ["model", "tool"])
@pytest.mark.parametrize("profile_change", ["none", "runtime", "runtime_and_provider"])
def test_public_sweep_retains_profile_and_continuation_evidence(
    tmp_path, database_url, monkeypatch, crash, profile_change
):
    monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, _runtime_build("a"))
    current_runtime_build_provenance.cache_clear()
    path = tmp_path / "cayu.db"
    _fresh_orphan(path, "scan", database_url, crash=crash)
    if profile_change != "none":
        monkeypatch.setenv(RUNTIME_BUILD_PROVENANCE_ENV, _runtime_build("b"))
        current_runtime_build_provenance.cache_clear()

    async def scenario():
        stores = open_application_stores(database_url, sqlite_path=path, event_watchers=False)
        provider = (
            _ChangedProvider([])
            if profile_change == "runtime_and_provider"
            else VersionedFakeProvider([])
        )
        tool = _NoDispatch()
        app = _application(
            stores.session_store,
            environment=_slot(),
            task_store=stores.task_store,
            provider=lambda _events: provider,
            tool=tool,
        )
        store = stores.session_store
        try:
            await app.create_task(TaskCreate(task_id="finished", type="scan", session_id="scan"))
            await stores.task_store.cancel_task("finished")
            before = await store.load("scan")
            stage = await store.load_active_model_completion_stage("scan")
            if crash == "model" and profile_change != "none":
                assert stage is not None
                # Identical stored stage/profile data does not authenticate a public
                # manual request. Only the sweep's admitted claim may borrow its profile.
                with pytest.raises(ExecutionProfileMismatchError):
                    await app.recover_model_completion_stage(
                        ModelCompletionManualRecoveryRequest(
                            session_id="scan",
                            stage_id=stage.stage.stage_id,
                            expected_run_epoch=before.run_epoch,
                            expected_session_instance_id=before.instance_id,
                            terminal_status=SessionStatus.INTERRUPTED,
                            inactive_for_seconds=0,
                        )
                    )
                unchanged = await store.load("scan")
                assert unchanged.status is before.status
                assert unchanged.run_epoch == before.run_epoch
            if crash == "model" and profile_change == "runtime":
                owner = app._manual_model_completion_recovery
                settle = owner.recover_model_completion_stage

                async def inspect_claim(
                    command,
                    *,
                    invocation_context=None,
                    interruption_reason=None,
                    interruption_payload=None,
                ):
                    assert invocation_context is not None
                    snapshot = (
                        await store.load("scan"),
                        await store.load_checkpoint("scan"),
                        await store.load_events("scan"),
                        await store.load_active_model_completion_stage("scan"),
                    )
                    for damage in ("provenance", "incarnation", "epoch", "claim", "stage"):
                        copied = object.__new__(InvocationContext)
                        for name in InvocationContext.__slots__:
                            object.__setattr__(copied, name, getattr(invocation_context, name))
                        attempted = command
                        if damage == "provenance":
                            object.__setattr__(copied, "_authority_token", None)
                        elif damage == "incarnation":
                            object.__setattr__(
                                copied,
                                "_binding",
                                replace(
                                    copied.binding,
                                    session_instance_id="00000000-0000-4000-8000-000000000001",
                                ),
                            )
                        elif damage == "epoch":
                            stale_epoch = copied.binding.run_epoch + 1
                            object.__setattr__(
                                copied, "_binding", replace(copied.binding, run_epoch=stale_epoch)
                            )
                            object.__setattr__(
                                copied,
                                "_active_profile",
                                copied.active_profile.model_copy(update={"run_epoch": stale_epoch}),
                            )
                        elif damage == "claim":
                            object.__setattr__(copied, "_recovery_claim_id", None)
                        else:
                            attempted = command.model_copy(update={"stage_id": "another-stage"})
                        with pytest.raises(
                            (
                                TypeError,
                                ValueError,
                                SessionRunFenced,
                                SessionModelCompletionStageConflict,
                            )
                        ):
                            await settle(attempted, invocation_context=copied)
                        assert snapshot == (
                            await store.load("scan"),
                            await store.load_checkpoint("scan"),
                            await store.load_events("scan"),
                            await store.load_active_model_completion_stage("scan"),
                        )
                    return await settle(
                        command,
                        invocation_context=invocation_context,
                        interruption_reason=interruption_reason,
                        interruption_payload=interruption_payload,
                    )

                monkeypatch.setattr(owner, "recover_model_completion_stage", inspect_claim)
            original_events = await store.load_events("scan")
            assert before.status is SessionStatus.RUNNING
            assert (await app.inspect_session_execution("scan")).state == "owner_lost"
            [result] = await app.reconcile_orphaned_sessions()
            current = await store.load("scan")
            events = (await store.load_events("scan"))[len(original_events) :]
            rejections = [
                event
                for event in events
                if event.type is EventType.SESSION_EXECUTION_PROFILE_REJECTED
            ]
            assert provider.requests == []
            assert tool.calls == 0
            if profile_change == "runtime_and_provider":
                assert result.status is SessionStatus.RUNNING
                assert current.run_epoch == before.run_epoch
                assert [action.value for action in result.actions] == ["failed"]
                assert (
                    len(
                        [
                            event
                            for event in await store.load_events("scan")
                            if event.type is EventType.SESSION_EXECUTION_PROFILE_REJECTED
                        ]
                    )
                    == 1
                )
                assert not any(event.type is EventType.SESSION_INTERRUPTED for event in events)
                return

            assert result.status is current.status is SessionStatus.INTERRUPTED, result
            assert not rejections
            [interrupted] = [
                event for event in events if event.type is EventType.SESSION_INTERRUPTED
            ]
            assert interrupted.payload["reason"] == "orphaned_session_reconciled"
            session, profile, pending = await _http_boundary(app)
            assert session["status"] == "interrupted"
            assert profile["boundary"] == ("resume" if crash == "model" else "continuation")
            assert pending["issues"] == []
            assert pending["has_more"] is False and pending["next_cursor"] is None
            if crash == "model":
                assert pending["actions"] == []
                assert stage is not None
                settlement = await store.load_model_completion_stage_settlement(
                    "scan", stage.stage.stage_id
                )
                assert settlement is not None
                assert settlement.disposition.value == "provider_effect_outcome_unknown"
                assert settlement.reason_code == "operator_outcome_unknown"
            else:
                [action] = pending["actions"]
                assert action["kind"] == "manual_recovery"
                assert action["session"]["id"] == "scan"
                assert any(event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN for event in events)
            stable_events = await store.load_events("scan")
            assert await app.reconcile_orphaned_sessions() == ()
            assert await store.load_events("scan") == stable_events
        finally:
            await app.aclose()
            await stores.close()

    try:
        asyncio.run(scenario())
    finally:
        current_runtime_build_provenance.cache_clear()
