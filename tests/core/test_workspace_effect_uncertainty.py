"""Workspace observations cannot turn an unknown external tool into a result."""

from __future__ import annotations

import asyncio

import pytest
from tests.core.test_workspace_mutation_receipts import (
    _BlockingThreadWorkspace,
    _BlockingWorkspaceMutationTool,
    _portable_environment_spec,
    _SingleToolProvider,
)

from cayu import CayuConfig, SQLiteSessionStore, ToolExecutionConfig
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu.core import AgentSpec, EventType, ExecutionProfileBehaviorIdentity, Message
from cayu.environments import DeterministicWorkspaceBinding, Environment
from cayu.runners import RunnerExecutionError
from cayu.runtime import (
    CayuApp,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    RunRequest,
)
from cayu.runtime._session_engine import SessionEngine
from cayu.runtime._tool_effect_state import ToolEffectReconciliationRequired, ToolEffectStateOwner
from cayu.runtime.workspace_observation_recovery import (
    is_workspace_observation_recovery_rejected,
    workspace_observations_from_checkpoint,
)


class PortableBlockingMutationTool(_BlockingWorkspaceMutationTool):
    spec = _BlockingWorkspaceMutationTool.spec.model_copy(
        update={
            "execution_profile_identity": ExecutionProfileBehaviorIdentity(
                name="tests:uncertain-workspace:mutation",
                behavior_version="1",
                implementation_version="1",
            )
        }
    )

    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    async def run(self, ctx, args):
        self.calls.append(dict(args))
        return await super().run(ctx, args)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    ("mode", "termination"),
    [
        ("normal", "cancel"),
        ("finalization_failure", "cancel"),
        ("conflicting_evidence", "cancel"),
        ("finalization_failure", "timeout"),
        ("interruption_failure", "timeout"),
    ],
)
def test_unknown_effect_workspace_receipt_survives_recovery(
    tmp_path, backend, mode, termination, monkeypatch
):
    fail_finalization = mode != "normal"

    async def scenario():
        interruption_error = OSError("session interruption unavailable")
        original_interrupt = SessionEngine._handle_session_interrupted
        if mode == "interruption_failure":

            async def fail_interruption(self, **kwargs):
                raise interruption_error
                yield  # Keep the real cleanup owner's async-iterator contract.

            monkeypatch.setattr(SessionEngine, "_handle_session_interrupted", fail_interruption)
        root = tmp_path / "workspace"
        root.mkdir()
        path = tmp_path / "sessions.sqlite"
        store = SQLiteSessionStore(path) if backend == "sqlite" else InMemorySessionStore()
        workspace = _BlockingThreadWorkspace(root, workspace_id="uncertain-workspace")
        session_id = "workspace-unknown-effect"
        calls = []

        def app_for(current_store):
            app = CayuApp(
                session_store=current_store,
                enable_logging=False,
                config=(
                    CayuConfig(tool_execution=ToolExecutionConfig(tool_timeout_seconds=1))
                    if termination == "timeout"
                    else CayuConfig()
                ),
            )
            app.register_provider(
                _SingleToolProvider(tool_name="blocking_workspace_mutation", arguments={}),
                default=True,
            )
            app.register_environment(
                Environment(
                    _portable_environment_spec("local"),
                    workspace=workspace,
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            app.register_agent(
                AgentSpec(name="assistant", model="scripted-model"),
                tools=[PortableBlockingMutationTool(calls)],
            )
            return app

        original_publish = store.publish_runtime_publication
        failures = 0

        async def publish(session_id, *, request, **kwargs):
            nonlocal failures
            if (
                fail_finalization
                and request.kind == "workspace-observation"
                and request.intent.get("phase") == "terminal"
            ):
                failures += 1
                raise OSError("workspace finalization unavailable")
            return await original_publish(session_id, request=request, **kwargs)

        monkeypatch.setattr(store, "publish_runtime_publication", publish)
        live_events = []

        async def consume():
            async for event in app_for(store).run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "write")],
                ),
            ):
                live_events.append(event)
            return live_events

        consumer = asyncio.create_task(consume())
        try:
            assert await asyncio.to_thread(workspace.dispatched.wait, 10)
            assert consumer.cancelling() == 0
            if termination == "cancel":
                consumer.cancel("cancel after dispatch")
                await asyncio.sleep(0.05)
                assert consumer.cancelling() == 1
            else:
                await asyncio.sleep(1.1)
            assert not consumer.done()
            assert not any(
                event.type is EventType.WORKSPACE_MUTATION_RECORDED
                for event in await store.load_events(session_id)
            )
            workspace.release.set()
            if termination == "cancel":
                with pytest.raises(
                    asyncio.CancelledError, match="cancel after dispatch"
                ) as cancelled:
                    await consumer
                assert consumer.cancelled() and consumer.cancelling() == 1
            else:
                with pytest.raises(ExceptionGroup) as failed_cleanup:
                    await consumer
                if mode == "interruption_failure":
                    original_group, later_failure = failed_cleanup.value.exceptions
                    assert later_failure is interruption_error
                    primary, secondary = original_group.exceptions
                else:
                    assert live_events[-1].type is EventType.SESSION_INTERRUPTED
                    primary, secondary = failed_cleanup.value.exceptions
                assert type(primary) is ToolEffectReconciliationRequired
                assert type(secondary) is RunnerExecutionError
                assert secondary.diagnostic["error_type"] == "OSError"
                assert "workspace finalization unavailable" not in repr(failed_cleanup.value)
            assert bool(failures) is fail_finalization
            if fail_finalization and termination == "cancel":
                cause = exception_cause(cancelled.value)
                assert cause is not None
                bounded_failures = [
                    error
                    for error in iter_exception_tree(cause)
                    if type(error) is RunnerExecutionError
                ]
                assert [error.diagnostic["error_type"] for error in bounded_failures] == ["OSError"]
                assert "workspace finalization unavailable" not in repr(cause)
            monkeypatch.setattr(store, "publish_runtime_publication", original_publish)
            monkeypatch.setattr(SessionEngine, "_handle_session_interrupted", original_interrupt)
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(path)

            events = await store.load_events(session_id)
            unknown = [
                event for event in events if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
            ]
            assert len(unknown) == 1
            round_id = unknown[0].payload["tool_round_id"]
            call_id = unknown[0].payload["tool_call_id"]
            session = await store.load(session_id)
            owner = ToolEffectStateOwner(store)
            record = await owner.resolve_call(session, tool_round_id=round_id, tool_call_id=call_id)
            assert record is not None and record.state == "outcome_unknown"
            receipts = [
                event for event in events if event.type is EventType.WORKSPACE_MUTATION_RECORDED
            ]
            assert len(receipts) == 1
            assert receipts[0].payload["status"] == "changed"
            checkpoint = await store.load_checkpoint(session_id)
            assert bool(workspace_observations_from_checkpoint(checkpoint)) is fail_finalization

            recovery_app = app_for(store)
            if mode == "conflicting_evidence":
                original_query = store.query_events
                conflict_reads = 0

                async def conflicting_query(query):
                    nonlocal conflict_reads
                    rows = await original_query(query)
                    if query.event_id == unknown[0].id:
                        conflict_reads += 1
                        return [
                            row.model_copy(
                                update={
                                    "event": row.event.model_copy(
                                        update={
                                            "payload": {
                                                **row.event.payload,
                                                "dispatch_id": "conflicting-dispatch",
                                            }
                                        }
                                    )
                                }
                            )
                            for row in rows
                        ]
                    return rows

                monkeypatch.setattr(store, "query_events", conflicting_query)
                with pytest.raises(Exception) as rejected:
                    await recovery_app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(session_id=session_id)
                    )
                assert is_workspace_observation_recovery_rejected(rejected.value)
                assert conflict_reads > 0
                assert await owner.load(record.intent) == record
                assert workspace_observations_from_checkpoint(
                    await store.load_checkpoint(session_id)
                ) == workspace_observations_from_checkpoint(checkpoint)
                assert not any(
                    event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
                    for event in await store.load_events(session_id)
                )
                assert calls == [{}]
                monkeypatch.setattr(store, "query_events", original_query)
            for _ in range(2):
                await recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                )
                assert calls == [{}]
                pending = await store.load_checkpoint(session_id)
                assert pending is not None and pending.get("pending_tool_round") is not None
            assert await owner.load(record.intent) == record
            recovered_events = await store.load_events(session_id)
            assert [
                event.id
                for event in recovered_events
                if event.type is EventType.WORKSPACE_MUTATION_RECORDED
            ] == [receipts[0].id]
            finalized = [
                event
                for event in recovered_events
                if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
            ]
            assert len(finalized) == 1
            assert finalized[0].payload["status"] == "complete"
            assert not any(
                event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
                for event in recovered_events
            )
            assert not workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            )
            assert (root / "settled.txt").read_bytes() == b"settled"
        finally:
            workspace.release.set()
            if not consumer.done():
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
            if backend == "sqlite":
                await store.close()

    asyncio.run(scenario())
