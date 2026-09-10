from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from tests.core._event_projection_support import private_events_for_public_events
from tests.core.task_invocation_fixtures import task_backed_session_invocation
from tests.core.test_verified_work_contracts import _contract
from tests.environments.sync_ownership_assertions import assert_sync_resources_owned

from cayu import (
    AgentSpec,
    CayuApp,
    EnqueueSessionMessageRequest,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionMessageDeliveryMode,
    SessionStatusConflict,
    SyncBinding,
    TaskCreate,
    TaskStatus,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    VerifiedTaskHandler,
    VerifiedTaskWorker,
)
from cayu.runtime import InMemoryEventSink, SessionStatus
from cayu.runtime import _environment_lifecycle as lifecycle_module
from cayu.runtime._environment_lifecycle import pending_completion_finalization_from_checkpoint
from cayu.runtime.work_attempt_admission import (
    WorkAttemptExecutionRequest,
    WorkAttemptRecoveryRequest,
    WorkAttemptRecoveryRequired,
    WorkAttemptRunRequest,
)
from cayu.storage import SQLiteSessionStore, SQLiteTaskStore
from cayu.workspaces import LocalWorkspace, WorkspaceMutationResult


class FailOnceWriteWorkspace(LocalWorkspace):
    def __init__(self, root: Path, *, workspace_id: str, fail_path: str) -> None:
        super().__init__(root, workspace_id=workspace_id)
        self.fail_path = fail_path
        self.failed_writes = 0

    async def write_bytes(self, path: str, content: bytes) -> None:
        if path == self.fail_path:
            self.fail_path = ""
            self.failed_writes += 1
            raise OSError(f"forced sync write failure: {path}")
        await super().write_bytes(path, content)

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        if path == self.fail_path:
            self.fail_path = ""
            self.failed_writes += 1
            raise OSError(f"forced sync write failure: {path}")
        return await super().replace_bytes(
            path,
            content,
            expected_revision=expected_revision,
        )


class FailingFinalizeEvidenceStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.fail_finalize_evidence = True
        self.commit_finalize_evidence_before_failure = False
        self.block_finalize_evidence = False
        self.finalize_evidence_attempt_ids: list[str] = []
        self.finalize_evidence_append_started = asyncio.Event()
        self.allow_finalize_evidence_append = asyncio.Event()

    async def append_event(self, session_id: str, event: Event) -> None:
        if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED:
            self.finalize_evidence_attempt_ids.append(event.id)
            if self.block_finalize_evidence:
                self.finalize_evidence_append_started.set()
                await self.allow_finalize_evidence_append.wait()
            if self.fail_finalize_evidence:
                if self.commit_finalize_evidence_before_failure:
                    await super().append_event(session_id, event)
                raise RuntimeError("forced finalize evidence failure")
        await super().append_event(session_id, event)


def _sync_durability_test_app(
    tmp_path: Path,
    store: InMemorySessionStore,
    *,
    event_sink: InMemoryEventSink | None = None,
) -> tuple[CayuApp, SyncBinding, FailOnceWriteWorkspace, LocalWorkspace, Path]:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "a-updated.txt").write_text("original-a", encoding="utf-8")
    (source_root / "b-fail.txt").write_text("original-b", encoding="utf-8")
    source = FailOnceWriteWorkspace(
        source_root,
        workspace_id="durability-source",
        fail_path="b-fail.txt",
    )
    target = LocalWorkspace(target_root, workspace_id="fixed-target")
    binding = SyncBinding(target_workspace=target)
    app = CayuApp(
        session_store=store,
        event_sinks=[] if event_sink is None else [event_sink],
        enable_logging=False,
    )
    app.register_provider(
        ScriptedModelProvider(
            [
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ],
            name="durability-provider",
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="sync-durability"),
            workspace=source,
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="sync-durability-agent",
            model="scripted-model",
            provider_name="durability-provider",
        )
    )
    return app, binding, source, target, target_root


class MutateBoundWorkspaceTool(Tool):
    spec = ToolSpec(
        name="mutate_bound_workspace",
        description="Make deterministic file changes in the bound workspace.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if ctx.workspace is None:
            raise RuntimeError("bound workspace is unavailable")
        await ctx.workspace.write_bytes("a-updated.txt", b"updated-a")
        await ctx.workspace.write_bytes("b-fail.txt", b"updated-b")
        await ctx.workspace.write_bytes("created.txt", b"created")
        await ctx.workspace.delete("removed.txt")
        return ToolResult(content="workspace mutated")


def test_sync_binding_failure_blocks_completion_and_recovers_in_fresh_app(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    # LocalWorkspace.list() sorts paths, so the a-file copies before the injected
    # b-file failure and makes the partial-finalization point deterministic.
    (source_root / "a-updated.txt").write_text("original-a", encoding="utf-8")
    (source_root / "b-fail.txt").write_text("original-b", encoding="utf-8")
    (source_root / "removed.txt").write_text("remove-me", encoding="utf-8")
    (target_root / "stale-target.txt").write_text("clean-me", encoding="utf-8")

    source = FailOnceWriteWorkspace(
        source_root,
        workspace_id="durable-source",
        fail_path="b-fail.txt",
    )
    target = LocalWorkspace(target_root, workspace_id="ephemeral-target")
    binding = SyncBinding(
        target_workspace=target,
        source_conflict_policy="require_revision",
        max_file_bytes=1024,
    )
    store_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(store_path)
    tasks = InMemoryTaskStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_mutate_workspace",
                    name="mutate_bound_workspace",
                    arguments={},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Workspace changes are ready."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ],
        name="workspace-sync-fault-provider",
    )
    app = CayuApp(session_store=store, task_store=tasks, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="sync-fault"),
            workspace=source,
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="workspace-sync-assistant",
            model="scripted-model",
            provider_name="workspace-sync-fault-provider",
        ),
        tools=[MutateBoundWorkspaceTool()],
    )

    async def exercise_contract():
        await tasks.create_task(
            TaskCreate(task_id="workspace-sync-failure-task", type="candidate-build")
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="workspace-sync-assistant",
                    session_id="workspace-sync-failure",
                    task_id="workspace-sync-failure-task",
                    messages=[Message.text("user", "Apply the workspace changes.")],
                )
            )
        ]
        failed_task = await tasks.load_task("workspace-sync-failure-task")
        assert failed_task is not None
        assert failed_task.status is TaskStatus.FAILED
        assert failed_task.result is None
        assert failed_task.error == {
            "message": (
                "SyncBinding source revision conflict for 'b-fail.txt' after 1 copy-back mutations."
            ),
            "type": "SyncBindingSourceConflictError",
            "session_id": "workspace-sync-failure",
            "phase": "workspace_finalize",
            "workspace_output_committed": False,
        }
        initial_session = await store.load("workspace-sync-failure")
        initial_events = await store.load_events("workspace-sync-failure")
        initial_checkpoint = await store.load_checkpoint("workspace-sync-failure")
        assert initial_session is not None
        assert initial_session.status is SessionStatus.FAILED
        assert initial_checkpoint is not None
        assert "pending_completion_finalization" in initial_checkpoint
        assert binding._states
        assert len(provider.requests) == 2
        await store.close()

        partial_source = {
            path.name: path.read_text(encoding="utf-8") for path in source_root.iterdir()
        }
        reopened = SQLiteSessionStore(store_path)
        recovered_source = LocalWorkspace(source_root, workspace_id="durable-source")
        recovery_started = asyncio.Event()
        allow_recovery = asyncio.Event()

        class BlockingRecoveryTarget(LocalWorkspace):
            async def list(
                self,
                pattern: str = "**/*",
                *,
                limit: int | None = None,
            ):
                recovery_started.set()
                await allow_recovery.wait()
                return await super().list(pattern, limit=limit)

        recovered_target = BlockingRecoveryTarget(
            target_root,
            workspace_id="ephemeral-target",
        )
        recovered_binding = SyncBinding(
            target_workspace=recovered_target,
            source_conflict_policy="require_revision",
            max_file_bytes=1024,
        )
        recovery_provider = ScriptedModelProvider(
            [],
            name="workspace-sync-fault-provider",
        )
        recovery_app = CayuApp(
            session_store=reopened,
            task_store=tasks,
            enable_logging=False,
        )
        recovery_app.register_provider(recovery_provider, default=True)
        recovery_app.register_environment(
            Environment(
                EnvironmentSpec(name="sync-fault"),
                workspace=recovered_source,
                binding=recovered_binding,
            ),
            default=True,
        )
        recovery_app.register_agent(
            AgentSpec(
                name="workspace-sync-assistant",
                model="scripted-model",
                provider_name="workspace-sync-fault-provider",
            ),
            tools=[MutateBoundWorkspaceTool()],
        )
        contending_store = SQLiteSessionStore(store_path)
        contending_binding = SyncBinding(
            target_workspace=LocalWorkspace(
                target_root,
                workspace_id="ephemeral-target",
            ),
            source_conflict_policy="require_revision",
            max_file_bytes=1024,
        )
        contending_provider = ScriptedModelProvider(
            [],
            name="workspace-sync-fault-provider",
        )
        contending_app = CayuApp(
            session_store=contending_store,
            task_store=tasks,
            enable_logging=False,
        )
        contending_app.register_provider(contending_provider, default=True)
        contending_app.register_environment(
            Environment(
                EnvironmentSpec(name="sync-fault"),
                workspace=LocalWorkspace(
                    source_root,
                    workspace_id="durable-source",
                ),
                binding=contending_binding,
            ),
            default=True,
        )
        contending_app.register_agent(
            AgentSpec(
                name="workspace-sync-assistant",
                model="scripted-model",
                provider_name="workspace-sync-fault-provider",
            ),
            tools=[MutateBoundWorkspaceTool()],
        )
        try:
            recovery_task = asyncio.create_task(
                recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="workspace-sync-failure",
                        reason="workspace_finalization_restart_test",
                    )
                )
            )
            await asyncio.wait_for(recovery_started.wait(), timeout=10)
            skipped_recovery = await contending_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="workspace-sync-failure",
                    reason="workspace_finalization_contention_test",
                )
            )
            allow_recovery.set()
            recovery = await recovery_task
            durable_session = await reopened.load("workspace-sync-failure")
            durable_events = await reopened.load_events("workspace-sync-failure")
            recovered_checkpoint = await reopened.load_checkpoint("workspace-sync-failure")
        finally:
            allow_recovery.set()
            await reopened.close()
            await contending_store.close()

        assert recovery.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,
        )
        assert recovery_provider.requests == []
        assert skipped_recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        assert contending_provider.requests == []
        assert contending_binding._states == {}
        assert recovered_binding._states == {}
        assert recovered_checkpoint is not None
        assert "pending_completion_finalization" not in recovered_checkpoint
        return (
            events,
            partial_source,
            durable_session,
            initial_events,
            durable_events,
        )

    events, partial_source, session, initial_events, durable_events = asyncio.run(
        exercise_contract()
    )

    assert source.failed_writes == 1
    assert partial_source == {
        "a-updated.txt": "updated-a",
        "b-fail.txt": "original-b",
        "removed.txt": "remove-me",
    }

    assert session is not None
    assert session.status == SessionStatus.FAILED
    streamed_types = [event.type for event in events]
    assert EventType.TASK_FAILED in streamed_types
    assert EventType.TURN_COMPLETED in streamed_types
    assert streamed_types[-1] is EventType.SESSION_FAILED
    assert streamed_types.index(EventType.TASK_FAILED) < streamed_types.index(
        EventType.SESSION_FAILED
    )
    assert EventType.TASK_COMPLETED not in {event.type for event in initial_events}
    assert EventType.SESSION_COMPLETED not in {event.type for event in initial_events}
    finalize_failure = next(
        event
        for event in initial_events
        if event.type is EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
    )
    expected_error = (
        "SyncBinding source revision conflict for 'b-fail.txt' after 1 copy-back mutations."
    )
    assert finalize_failure.payload["error"] == expected_error
    assert finalize_failure.payload["error_type"] == "SyncBindingSourceConflictError"
    assert initial_events[-1].payload["binding_finalize_error"] == {
        "error": expected_error,
        "error_type": "SyncBindingSourceConflictError",
        "outcome": "completed",
        "failures": [
            {
                "phase": "workspace_finalize",
                "error": expected_error,
                "error_type": "SyncBindingSourceConflictError",
            }
        ],
    }
    assert initial_events[-1].payload["workspace_output_committed"] is False
    assert EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED in {
        event.type for event in durable_events
    }
    assert {path.name: path.read_text(encoding="utf-8") for path in source_root.iterdir()} == {
        "a-updated.txt": "updated-a",
        "b-fail.txt": "updated-b",
        "created.txt": "created",
    }


def test_sync_binding_retains_owner_until_finalize_failure_evidence_is_durable(
    tmp_path: Path,
) -> None:
    store = FailingFinalizeEvidenceStore()
    sink = InMemoryEventSink()
    app, binding, source, target, target_root = _sync_durability_test_app(
        tmp_path,
        store,
        event_sink=sink,
    )

    async def exercise_contract() -> None:
        run_error: BaseException | None = None
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="sync-durability-agent",
                        session_id="sync-durability-session",
                        messages=[Message.text("user", "Finish.")],
                    )
                )
            ]
        except BaseException as exc:
            run_error = exc

        assert run_error is not None
        # The terminalizer performs one append/reconciliation attempt. Its
        # outer abort defers a retry so the authoritative exception shape is
        # not replaced during the same unwind.
        assert len(store.finalize_evidence_attempt_ids) == 1
        assert len(set(store.finalize_evidence_attempt_ids)) == 1
        assert binding._states
        generation = next(iter(binding._states))
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=True,
        )
        assert "sync-durability-session" in app._environment_lifecycle._active_environment_setups

        target_before_rebind = {
            path.name: path.read_text(encoding="utf-8") for path in target_root.iterdir()
        }
        with pytest.raises(ValueError, match="already bound by an active session"):
            await binding.bind(source, None, session_id="blocked-rebind")
        assert {
            path.name: path.read_text(encoding="utf-8") for path in target_root.iterdir()
        } == target_before_rebind

        # A normal later run—not a private lifecycle call—delivers the bounded
        # cleanup retry. It reuses the stable pending event identity before the
        # new bind is admitted.
        store.fail_finalize_evidence = False
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="sync-durability-agent",
                    session_id="sync-durability-recovery-trigger",
                    messages=[Message.text("user", "Continue.")],
                )
            )
        ]
        assert events[-1].type == EventType.SESSION_COMPLETED
        assert len(store.finalize_evidence_attempt_ids) == 2
        assert len(set(store.finalize_evidence_attempt_ids)) == 1
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=False,
        )
        assert binding._states == {}
        assert (
            "sync-durability-session" not in app._environment_lifecycle._active_environment_setups
        )

        durable_events = await store.load_events("sync-durability-session")
        finalize_failures = [
            event
            for event in durable_events
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
        ]
        assert len(finalize_failures) == 1
        assert finalize_failures[0].id == store.finalize_evidence_attempt_ids[0]
        sink_failures = [
            event
            for event in sink.events
            if event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
        ]
        private_sink_failures = await private_events_for_public_events(store, sink_failures)
        assert [event.id for event in private_sink_failures] == [finalize_failures[0].id]

        rebound = await binding.bind(source, None, session_id="successful-rebind")
        assert_sync_resources_owned(rebound, expected=True)
        binding.abandon(rebound)
        assert_sync_resources_owned(rebound, expected=False)

    asyncio.run(exercise_contract())


def test_sync_binding_retains_owner_after_finalize_failure_commit_is_reconciled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FailingFinalizeEvidenceStore()
    store.commit_finalize_evidence_before_failure = True
    app, binding, source, target, _target_root = _sync_durability_test_app(tmp_path, store)

    async def exercise_contract() -> None:
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="sync-durability-agent",
                    session_id="sync-durability-reconciled",
                    messages=[Message.text("user", "Finish.")],
                )
            )
        ]
        assert source.failed_writes == 1
        assert store.finalize_evidence_attempt_ids
        assert len(set(store.finalize_evidence_attempt_ids)) == 1
        assert binding._states
        generation = next(iter(binding._states))
        assert_sync_resources_owned(source, target, generation=generation, expected=True)
        assert "sync-durability-reconciled" in app._environment_lifecycle._active_environment_setups
        assert (
            sum(
                event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
                for event in await store.load_events("sync-durability-reconciled")
            )
            == 1
        )
        assert events[-1].type == EventType.SESSION_FAILED
        assert EventType.SESSION_COMPLETED not in {event.type for event in events}

        assert (
            pending_completion_finalization_from_checkpoint(
                await store.load_checkpoint("sync-durability-reconciled")
            )
            is not None
        )

        # Recovery must follow positive settlement of the process-local owner;
        # admission only polls retained cleanup for a short, bounded interval.
        lifecycle = app._environment_lifecycle
        original_abort = lifecycle.abort_environment_setup
        cleanup_started = asyncio.Event()
        allow_cleanup = asyncio.Event()

        async def blocked_cleanup(**kwargs: Any) -> None:
            cleanup_started.set()
            await allow_cleanup.wait()
            await original_abort(**kwargs)

        monkeypatch.setattr(lifecycle, "abort_environment_setup", blocked_cleanup)
        async with asyncio.timeout(30):
            drain = asyncio.create_task(app.drain_environment_cleanups(timeout_s=20))
            try:
                await cleanup_started.wait()
                await lifecycle._settle_retained_environment_cleanups()
                assert not drain.done()
                with pytest.raises(ValueError, match="already bound by an active session"):
                    await binding.bind(source, None, session_id="blocked-reconciled-rebind")
                assert_sync_resources_owned(source, target, generation=generation, expected=True)
            finally:
                allow_cleanup.set()
                drained = await drain
            assert drained
        monkeypatch.setattr(lifecycle, "abort_environment_setup", original_abort)

        assert (
            pending_completion_finalization_from_checkpoint(
                await store.load_checkpoint("sync-durability-reconciled")
            )
            is None
        )
        # Draining already finalized the retained workspace and cleared its
        # durable retry marker, so recovery must not repeat that work.
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="sync-durability-reconciled",
                reason="workspace_finalization_reconciliation_test",
            )
        )
        assert recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
        assert binding._states == {}
        assert_sync_resources_owned(source, target, generation=generation, expected=False)
        assert (
            "sync-durability-reconciled"
            not in app._environment_lifecycle._active_environment_setups
        )

        rebound = await binding.bind(source, None, session_id="reconciled-rebind")
        assert_sync_resources_owned(rebound, expected=True)
        binding.abandon(rebound)
        assert_sync_resources_owned(rebound, expected=False)

    asyncio.run(exercise_contract())


def test_concurrent_lazy_cleanup_sweeps_cannot_release_pending_sync_owner_early(
    tmp_path: Path,
) -> None:
    store = FailingFinalizeEvidenceStore()
    app, binding, source, target, _target_root = _sync_durability_test_app(tmp_path, store)

    async def exercise_contract() -> None:
        with pytest.raises(OSError, match="forced sync write failure"):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="sync-durability-agent",
                        session_id="sync-concurrent-settlement",
                        messages=[Message.text("user", "Finish.")],
                    )
                )
            ]
        failed_session = await store.load("sync-concurrent-settlement")
        assert failed_session is not None
        assert failed_session.status is SessionStatus.FAILED
        lifecycle_types = [
            event.type
            for event in await store.load_events("sync-concurrent-settlement")
            if event.type
            in {
                EventType.INTERACTION_STARTED,
                EventType.INTERACTION_COMPLETED,
                EventType.INTERACTION_FAILED,
            }
        ]
        assert lifecycle_types == [
            EventType.INTERACTION_STARTED,
            EventType.INTERACTION_COMPLETED,
        ]
        generation = next(iter(binding._states))
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=True,
        )

        store.fail_finalize_evidence = False
        store.block_finalize_evidence = True
        first_sweep = asyncio.create_task(
            app._environment_lifecycle._settle_retained_environment_cleanups()
        )
        await store.finalize_evidence_append_started.wait()
        await app._environment_lifecycle._settle_retained_environment_cleanups()
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=True,
        )
        assert binding._states
        retained = app._environment_lifecycle._active_environment_setups[
            "sync-concurrent-settlement"
        ]
        settlement_task = retained.cleanup_settlement_task
        assert settlement_task is not None
        assert not settlement_task.done()

        store.allow_finalize_evidence_append.set()
        await first_sweep
        await settlement_task
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=False,
        )
        assert binding._states == {}
        assert len(store.finalize_evidence_attempt_ids) == 2
        assert len(set(store.finalize_evidence_attempt_ids)) == 1

    asyncio.run(exercise_contract())


def test_lazy_cleanup_sweep_keeps_owned_settlement_alive_after_trigger_cancellation(
    tmp_path: Path,
) -> None:
    store = FailingFinalizeEvidenceStore()
    app, binding, source, target, _target_root = _sync_durability_test_app(tmp_path, store)

    async def exercise_contract() -> None:
        with pytest.raises(OSError, match="forced sync write failure"):
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="sync-durability-agent",
                        session_id="sync-cancelled-settlement",
                        messages=[Message.text("user", "Finish.")],
                    )
                )
            ]
        generation = next(iter(binding._states))
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=True,
        )

        store.fail_finalize_evidence = False
        store.block_finalize_evidence = True

        async def trigger_normal_runtime_activity() -> list[Event]:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="sync-durability-agent",
                        session_id="sync-cancelled-settlement-trigger",
                        messages=[Message.text("user", "Continue.")],
                    )
                )
            ]

        trigger_task = asyncio.create_task(trigger_normal_runtime_activity())
        await store.finalize_evidence_append_started.wait()
        trigger_task.cancel("stop cleanup trigger")
        with pytest.raises(asyncio.CancelledError, match="stop cleanup trigger"):
            await asyncio.wait_for(trigger_task, timeout=1)
        assert trigger_task.cancelled()
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=True,
        )
        retained = app._environment_lifecycle._active_environment_setups[
            "sync-cancelled-settlement"
        ]
        settlement_task = retained.cleanup_settlement_task
        assert settlement_task is not None
        assert not settlement_task.done()

        store.allow_finalize_evidence_append.set()
        await settlement_task
        assert_sync_resources_owned(
            source,
            target,
            generation=generation,
            expected=False,
        )
        assert binding._states == {}
        assert len(store.finalize_evidence_attempt_ids) == 2
        assert len(set(store.finalize_evidence_attempt_ids)) == 1

    asyncio.run(exercise_contract())


@pytest.mark.parametrize("store_kind", ["memory", "sqlite"])
def test_sync_completion_stays_running_until_workspace_finalization_commits(
    tmp_path: Path,
    store_kind: str,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "result.txt").write_text("ready", encoding="utf-8")

    class BlockingCompletionBinding(SyncBinding):
        def __init__(self) -> None:
            super().__init__(
                target_workspace=LocalWorkspace(target_root, workspace_id="blocking-target"),
                source_conflict_policy="require_revision",
                max_file_bytes=1024,
            )
            self.finalize_started = asyncio.Event()
            self.allow_finalize = asyncio.Event()

        async def finalize(self, bound, *, outcome=None, metadata=None):
            self.finalize_started.set()
            await self.allow_finalize.wait()
            return await super().finalize(bound, outcome=outcome, metadata=metadata)

    store = (
        InMemorySessionStore()
        if store_kind == "memory"
        else SQLiteSessionStore(tmp_path / "completion-finalization.sqlite")
    )
    binding = BlockingCompletionBinding()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.completed({"finish_reason": "stop"})]],
            name="blocking-finalize-provider",
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="blocking-finalize"),
            workspace=LocalWorkspace(source_root, workspace_id="blocking-source"),
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="blocking-finalize-agent",
            model="scripted-model",
            provider_name="blocking-finalize-provider",
        )
    )

    async def exercise_contract() -> None:
        run_task = asyncio.create_task(
            _collect_run(
                app,
                RunRequest(
                    agent_name="blocking-finalize-agent",
                    session_id="blocking-finalize-session",
                    messages=[Message.text("user", "Finish.")],
                ),
            )
        )
        try:
            await asyncio.wait_for(binding.finalize_started.wait(), timeout=10)
            pending = await store.load("blocking-finalize-session")
            checkpoint = await store.load_checkpoint("blocking-finalize-session")
            assert pending is not None
            assert pending.status is SessionStatus.RUNNING
            assert checkpoint is not None
            assert "pending_completion_finalization" in checkpoint
            with pytest.raises(
                SessionStatusConflict,
                match="completion finalization is pending",
            ):
                await store.enqueue_session_message(
                    EnqueueSessionMessageRequest(
                        session_id="blocking-finalize-session",
                        idempotency_key="late-message",
                        content="late input",
                        delivery_mode=SessionMessageDeliveryMode.ON_IDLE,
                    )
                )

            binding.allow_finalize.set()
            events = await asyncio.wait_for(run_task, timeout=5)
            completed = await store.load("blocking-finalize-session")
            checkpoint = await store.load_checkpoint("blocking-finalize-session")
            assert completed is not None
            assert completed.status is SessionStatus.COMPLETED
            assert checkpoint is not None
            assert "pending_completion_finalization" not in checkpoint
            assert events[-1].type is EventType.SESSION_COMPLETED
        finally:
            binding.allow_finalize.set()
            if not run_task.done():
                run_task.cancel()
                await asyncio.gather(run_task, return_exceptions=True)
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(exercise_contract())


@pytest.mark.parametrize("claimed_task,governed", [(False, False), (True, False), (True, True)])
def test_restarted_completion_finalization_settles_attached_task(
    tmp_path: Path,
    claimed_task: bool,
    governed: bool,
    monkeypatch,
    *,
    worker_cleanup: bool = False,
) -> None:
    source_root = tmp_path / "restarted-source"
    target_root = tmp_path / "restarted-target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "result.txt").write_text("ready", encoding="utf-8")
    session_path = tmp_path / "restarted-sessions.sqlite"
    task_path = tmp_path / "restarted-tasks.sqlite"

    class BlockingCompletionBinding(SyncBinding):
        def __init__(self) -> None:
            super().__init__(
                target_workspace=LocalWorkspace(
                    target_root,
                    workspace_id="restarted-target",
                ),
                source_conflict_policy="require_revision",
                max_file_bytes=1024,
            )
            self.finalize_started = asyncio.Event()
            self.allow_finalize = asyncio.Event()

        async def finalize(self, bound, *, outcome=None, metadata=None):
            self.finalize_started.set()
            await self.allow_finalize.wait()
            return await super().finalize(bound, outcome=outcome, metadata=metadata)

    def register_runtime(
        app: CayuApp,
        *,
        binding: SyncBinding,
        provider: ScriptedModelProvider,
    ) -> None:
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="restarted-finalize"),
                workspace=LocalWorkspace(
                    source_root,
                    workspace_id="restarted-source",
                ),
                binding=binding,
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(
                name="restarted-finalize-agent",
                model="scripted-model",
                provider_name="restarted-finalize-provider",
            )
        )

    async def exercise_contract() -> None:
        ownership_now = [datetime.now(UTC)]
        session_store = SQLiteSessionStore(session_path)
        task_store = (
            SQLiteTaskStore(
                task_path, clock=lambda: ownership_now[0], ownership_clock=lambda: ownership_now[0]
            )
            if governed
            else SQLiteTaskStore(task_path)
        )
        binding = BlockingCompletionBinding()
        provider = ScriptedModelProvider(
            [[ModelStreamEvent.completed({"finish_reason": "stop"})]],
            name="restarted-finalize-provider",
        )
        app = CayuApp(
            session_store=session_store,
            task_store=task_store,
            enable_logging=False,
        )
        register_runtime(app, binding=binding, provider=provider)
        await task_store.create_task(
            TaskCreate(task_id="historical-finalize-task", type="candidate-build")
        )
        await task_store.start_task(
            "historical-finalize-task",
            session_id="restarted-finalize-session",
            session_invocation=await task_backed_session_invocation(
                task_store,
                "historical-finalize-task",
                "restarted-finalize-session",
            ),
        )
        await task_store.fail_task(
            "historical-finalize-task",
            {"code": "historical_failure"},
        )
        contract = _contract(contract_id="restarted-finalize-contract") if governed else None
        if contract is not None:
            await task_store.publish_work_contract(contract)
        await task_store.create_task(
            TaskCreate(
                task_id="restarted-finalize-task",
                type="candidate-build",
                work_contract=None if contract is None else contract.reference(),
            )
        )
        task_worker_id = None
        task_lease_expires_at = None
        if claimed_task and not governed:
            claimed = await task_store.claim_task("crashed-finalize-worker", lease_seconds=1)
            assert claimed is not None
            task_worker_id = claimed.worker_id
            task_lease_expires_at = claimed.lease_expires_at

        run_request = RunRequest(
            agent_name="restarted-finalize-agent",
            session_id="restarted-finalize-session",
            task_id="restarted-finalize-task",
            task_worker_id=task_worker_id,
            task_lease_expires_at=task_lease_expires_at,
            messages=[Message.text("user", "Finish.")],
        )
        if governed:
            admission = await app.admit_work_attempt(
                run_request,
                execution=WorkAttemptExecutionRequest(
                    admission_id="finalize-admission",
                    claim_id="finalize-claim",
                    attempt_id="finalize-attempt",
                    interaction_id="finalize-interaction",
                    worker_id="crashed-finalize-worker",
                    generation=1,
                    lease_seconds=300,
                ),
            )

            async def collect_governed():
                return [
                    event
                    async for event in app._execute_work_attempt(
                        WorkAttemptRunRequest(
                            admission_id=admission.admission_id,
                            claim_id=admission.claim.claim_id,
                            worker_id=admission.claim.worker_id,
                            generation=1,
                            lease_seconds=300,
                        )
                    )
                ]

            run_task = asyncio.create_task(collect_governed())
        else:
            run_task = asyncio.create_task(_collect_run(app, run_request))
        recovery_session_store = SQLiteSessionStore(session_path)
        recovery_task_store = (
            SQLiteTaskStore(
                task_path, clock=lambda: ownership_now[0], ownership_clock=lambda: ownership_now[0]
            )
            if governed
            else SQLiteTaskStore(task_path)
        )
        recovery_provider = ScriptedModelProvider(
            [],
            name="restarted-finalize-provider",
        )
        recovery_binding = SyncBinding(
            target_workspace=LocalWorkspace(
                target_root,
                workspace_id="restarted-target",
            ),
            source_conflict_policy="require_revision",
            max_file_bytes=1024,
        )
        recovery_app = CayuApp(
            session_store=recovery_session_store,
            task_store=recovery_task_store,
            enable_logging=False,
        )
        register_runtime(
            recovery_app,
            binding=recovery_binding,
            provider=recovery_provider,
        )
        settlement = None
        try:
            await asyncio.wait_for(binding.finalize_started.wait(), timeout=10)
            pending_checkpoint = await session_store.load_checkpoint("restarted-finalize-session")
            assert pending_checkpoint is not None
            assert pending_checkpoint["pending_completion_finalization"]["task_id"] == (
                "restarted-finalize-task"
            )
            if governed:
                assert provider.requests
                assert "initial_transcript_pending" not in pending_checkpoint
            if claimed_task and not governed:
                await asyncio.sleep(1.05)
            if governed:
                ownership_now[0] = admission.claim.lease_expires_at + timedelta(seconds=1)
                recovery_request = WorkAttemptRecoveryRequest(
                    admission_id=admission.admission_id,
                    claim_id="finalize-recovery",
                    worker_id="recovery-worker",
                    generation=2,
                    lease_seconds=300,
                )
                stop = SQLiteTaskStore.record_work_attempt_execution_stop
                activate = SQLiteTaskStore.activate_work_attempt_recovery
                stop_ack_lost = activation_ack_lost = False

                async def lose_stop_ack(store, request):
                    nonlocal stop_ack_lost
                    result = await stop(store, request)
                    if not stop_ack_lost:
                        stop_ack_lost = True
                        raise ConnectionError("stop acknowledgement lost")
                    return result

                async def lose_activation_ack(store, request):
                    nonlocal activation_ack_lost
                    result = await activate(store, request)
                    if not activation_ack_lost:
                        activation_ack_lost = True
                        raise ConnectionError("activation acknowledgement lost")
                    return result

                with monkeypatch.context() as patch:
                    patch.setattr(
                        SQLiteTaskStore, "record_work_attempt_execution_stop", lose_stop_ack
                    )
                    patch.setattr(
                        SQLiteTaskStore, "activate_work_attempt_recovery", lose_activation_ack
                    )
                    with pytest.raises(ConnectionError, match="stop acknowledgement lost"):
                        await recovery_app.recover_work_attempt(recovery_request)
                    stopped = await recovery_task_store.load_work_attempt_admission(
                        admission.admission_id
                    )
                    assert stopped.execution_stop is not None
                    assert (
                        "pending_completion_finalization"
                        in await recovery_session_store.load_checkpoint(admission.session_id)
                    )
                    if worker_cleanup:
                        # Start discovery at the actual missing-ack boundary,
                        # without manually completing the stopped cleanup first.
                        from cayu.runtime import verified_task_worker as worker_module

                        class NoNewWork(VerifiedTaskHandler):
                            async def prepare(self, context):
                                pytest.fail("Cleanup must not prepare new work.")

                            async def propose(self, context):
                                pytest.fail("Cleanup must not propose completion.")

                        class DiscoveryClock(datetime):
                            @classmethod
                            def now(cls, tz=None):
                                return ownership_now[0]

                        replacement = CayuApp(
                            session_store=recovery_session_store,
                            task_store=recovery_task_store,
                            enable_logging=False,
                        )
                        register_runtime(
                            replacement, binding=recovery_binding, provider=recovery_provider
                        )
                        assert not await replacement._session_engine.has_recoverable_work_attempt_model_result(
                            stopped
                        )
                        assert (
                            await replacement._session_engine.load_work_attempt_released_recovery_evidence(
                                stopped
                            )
                            is None
                        )
                        ownership_now[0] = stopped.claim.lease_expires_at + timedelta(seconds=1)
                        patch.setattr(worker_module, "datetime", DiscoveryClock)
                        patch.setattr(SQLiteTaskStore, "activate_work_attempt_recovery", activate)
                        async with VerifiedTaskWorker(
                            replacement, NoNewWork(), worker_id="cleanup-replacement"
                        ) as worker:
                            assert await asyncio.wait_for(worker.run(max_tasks=1), 30) == 1
                        current = await recovery_task_store.load_work_attempt_admission(
                            admission.admission_id
                        )
                        settlement = await recovery_task_store.load_work_attempt_lifecycle_receipt(
                            admission.admission_id
                        )
                        assert current.claim.generation == 3
                        assert current.execution_stop == stopped.execution_stop
                        assert current.execution_entry == stopped.execution_entry
                        assert settlement is not None
                        assert settlement.task.status is TaskStatus.NEEDS_ATTENTION
                        assert settlement.task.status_reason == "work_contract_execution_failed"
                        assert not settlement.retired_contract_binding
                        assert (
                            await recovery_task_store.settle_work_attempt_lifecycle(
                                settlement.request
                            )
                            == settlement
                        )
                        assert (
                            await recovery_session_store.load(admission.session_id)
                        ).status is SessionStatus.FAILED
                        assert (
                            "pending_completion_finalization"
                            not in await recovery_session_store.load_checkpoint(
                                admission.session_id
                            )
                        )
                        assert (
                            await recovery_task_store.load_completion_proposal_for_attempt(
                                admission.attempt_id
                            )
                            is None
                        )
                        assert recovery_provider.requests == []
                        assert (target_root / "result.txt").read_text(encoding="utf-8") == "ready"
                        return
                    with pytest.raises(ConnectionError, match="activation acknowledgement lost"):
                        await recovery_app.recover_work_attempt(recovery_request)
                    recovery = await recovery_app.recover_work_attempt(recovery_request)
                assert recovery.execution_stop == stopped.execution_stop
                assert await recovery_app.recover_work_attempt(recovery_request) == recovery
                with pytest.raises(WorkAttemptRecoveryRequired, match="durably stopped"):
                    async for _ in recovery_app._execute_work_attempt(
                        WorkAttemptRunRequest(
                            admission_id=recovery.admission_id,
                            claim_id=recovery.claim.claim_id,
                            worker_id=recovery.claim.worker_id,
                            generation=2,
                            lease_seconds=300,
                        )
                    ):
                        pytest.fail("Stopped workspace execution must not redispatch.")
            else:
                recovery = await recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="restarted-finalize-session",
                        reason="completion_owner_restarted",
                    )
                )
            recovered_session = await recovery_session_store.load("restarted-finalize-session")
            recovered_task = await recovery_task_store.load_task("restarted-finalize-task")
            checkpoint = await recovery_session_store.load_checkpoint("restarted-finalize-session")
            durable_events = await recovery_session_store.load_events("restarted-finalize-session")

            if not governed:
                assert recovery.actions == (
                    IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,
                )
            assert recovery_provider.requests == []
            assert recovered_session is not None
            assert recovered_session.status is SessionStatus.FAILED
            assert recovered_task is not None
            assert recovered_task.status is (TaskStatus.RUNNING if governed else TaskStatus.FAILED)
            if not governed:
                assert recovered_task.error == {
                    "message": (
                        "Workspace output committed during recovery after the original "
                        "completion owner became unavailable."
                    ),
                    "type": "WorkspaceCompletionFinalizationRecovered",
                    "session_id": "restarted-finalize-session",
                    "phase": "workspace_finalize_recovery",
                    "workspace_output_committed": True,
                }
            assert checkpoint is not None
            assert "pending_completion_finalization" not in checkpoint
            assert (
                EventType.TASK_FAILED in {event.type for event in durable_events}
            ) is not governed
            assert EventType.SESSION_FAILED in {event.type for event in durable_events}
            if governed:
                release = await recovery_app._session_engine.load_work_attempt_release_evidence(
                    recovery
                )
                ownership_now[0] = recovery.claim.lease_expires_at + timedelta(seconds=1)
                # No registered provider/environment or surviving runtime context
                # is needed to reconcile an already-settled stopped attempt.
                fresh_app = CayuApp(
                    session_store=recovery_session_store,
                    task_store=recovery_task_store,
                    enable_logging=False,
                )
                next_owner = await fresh_app.recover_work_attempt(
                    WorkAttemptRecoveryRequest(
                        admission_id=admission.admission_id,
                        claim_id="finalize-third-claim",
                        worker_id="third-worker",
                        generation=3,
                        lease_seconds=300,
                    )
                )
                assert next_owner.execution_stop == recovery.execution_stop
                assert next_owner.execution_entry == recovery.execution_entry
                assert (
                    await fresh_app._session_engine.load_work_attempt_release_evidence(next_owner)
                    == release
                )
                assert await recovery_session_store.load(admission.session_id) == recovered_session
                assert (
                    await recovery_session_store.load_checkpoint(admission.session_id) == checkpoint
                )
                from cayu.runtime import verified_task_worker as worker_module

                class NoNewWork(VerifiedTaskHandler):
                    async def prepare(self, context):
                        pytest.fail("Stopped workspace recovery must not prepare new work.")

                    async def propose(self, context):
                        pytest.fail("Stopped workspace recovery must not propose completion.")

                class DiscoveryClock(datetime):
                    @classmethod
                    def now(cls, tz=None):
                        return ownership_now[0]

                # This fixture already controls SQLite's ownership clock. Align
                # only the worker's advisory scan clock with that same timeline.
                ownership_now[0] = next_owner.claim.lease_expires_at + timedelta(seconds=1)
                with monkeypatch.context() as patch:
                    patch.setattr(worker_module, "datetime", DiscoveryClock)
                    async with VerifiedTaskWorker(
                        fresh_app, NoNewWork(), worker_id="finalize-worker"
                    ) as worker:
                        assert await asyncio.wait_for(worker.run(max_tasks=1), 10) == 1
                settlement = await recovery_task_store.load_work_attempt_lifecycle_receipt(
                    next_owner.admission_id
                )
                assert settlement is not None
                current = await recovery_task_store.load_latest_work_attempt_admission(
                    next_owner.task_id
                )
                assert current.claim.generation == 4
                assert current.execution_stop == next_owner.execution_stop
                assert current.execution_entry == next_owner.execution_entry
                assert settlement.request.release_evidence == release
                assert (
                    await recovery_task_store.load_completion_proposal_for_attempt(
                        next_owner.attempt_id
                    )
                    is None
                )
                assert settlement.task.status is TaskStatus.NEEDS_ATTENTION
                assert settlement.task.status_reason == "work_contract_execution_failed"
                assert not settlement.retired_contract_binding
                assert (
                    await recovery_task_store.settle_work_attempt_lifecycle(settlement.request)
                    == settlement
                )
                assert (target_root / "result.txt").read_text(encoding="utf-8") == "ready"
        finally:
            binding.allow_finalize.set()
            await asyncio.gather(run_task, return_exceptions=True)
            if settlement is not None:
                assert (
                    await recovery_task_store.load_task("restarted-finalize-task")
                    == settlement.task
                )
            await recovery_task_store.close()
            await recovery_session_store.close()
            await task_store.close()
            await session_store.close()

    asyncio.run(exercise_contract())


def test_worker_discovers_stopped_workspace_cleanup_after_lost_ack(tmp_path, monkeypatch):
    test_restarted_completion_finalization_settles_attached_task(
        tmp_path, True, True, monkeypatch, worker_cleanup=True
    )


@pytest.mark.parametrize(
    "recovery_control",
    [
        "normal",
        "cancel",
        "cancel_then_clear_failure",
        "same_app_cancel_then_clear_failure",
        "same_app_marker_conflict",
    ],
)
def test_cancelled_completion_finalization_remains_failed_and_recoverable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recovery_control: str,
) -> None:
    source_root = tmp_path / "cancelled-source"
    target_root = tmp_path / "cancelled-target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "result.txt").write_text("ready", encoding="utf-8")

    class CancelOnceCompletionBinding(SyncBinding):
        cancel_once = True
        finalize_calls = 0

        async def finalize(self, bound, *, outcome=None, metadata=None):
            self.finalize_calls += 1
            if self.cancel_once:
                self.cancel_once = False
                raise asyncio.CancelledError("injected finalization cancellation")
            return await super().finalize(bound, outcome=outcome, metadata=metadata)

    store = InMemorySessionStore()
    binding = CancelOnceCompletionBinding(
        target_workspace=LocalWorkspace(target_root, workspace_id="cancelled-target"),
        source_conflict_policy="require_revision",
        max_file_bytes=1024,
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.completed({"finish_reason": "stop"})]],
            name="cancelled-finalize-provider",
        ),
        default=True,
    )
    app.register_environment(
        Environment(
            EnvironmentSpec(name="cancelled-finalize"),
            workspace=LocalWorkspace(source_root, workspace_id="cancelled-source"),
            binding=binding,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="cancelled-finalize-agent",
            model="scripted-model",
            provider_name="cancelled-finalize-provider",
        )
    )

    async def exercise_contract() -> None:
        events = await _collect_run(
            app,
            RunRequest(
                agent_name="cancelled-finalize-agent",
                session_id="cancelled-finalize-session",
                messages=[Message.text("user", "Finish.")],
            ),
        )
        assert EventType.SESSION_COMPLETED not in {event.type for event in events}
        interrupted = await store.load("cancelled-finalize-session")
        checkpoint = await store.load_checkpoint("cancelled-finalize-session")
        assert interrupted is not None
        assert interrupted.status is SessionStatus.FAILED
        assert checkpoint is not None
        assert "pending_completion_finalization" in checkpoint

        expected_action = IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION
        if recovery_control == "normal":
            # A child-originated cancellation may leave owned cleanup settling
            # after the stream returns. Recovery must not race that retained owner.
            assert await app.drain_environment_cleanups()
            after_cleanup = await store.load_checkpoint("cancelled-finalize-session")
            assert after_cleanup is not None
            if "pending_completion_finalization" not in after_cleanup:
                expected_action = IncompleteSessionRecoveryAction.SKIPPED_TERMINAL
        request = IncompleteSessionRecoveryRequest(
            session_id="cancelled-finalize-session",
            reason="cancelled_workspace_finalization",
        )
        recovery = None
        if recovery_control == "normal":
            # Exercise the settled path independently of filesystem scheduling;
            # blocked-owner tests cover admission while cleanup is still live.
            monkeypatch.setattr(
                lifecycle_module, "_LAZY_ENVIRONMENT_CLEANUP_ADMISSION_BUDGET_SECONDS", 1.0
            )
            recovery = await app.recover_incomplete_session(request)
        else:
            # A fresh app enters durable completion recovery, rather than the
            # original app's lazy sweep of its retained finalization owner.
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_provider = ScriptedModelProvider([], name="cancelled-finalize-provider")
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_environment(
                Environment(
                    EnvironmentSpec(name="cancelled-finalize"),
                    workspace=LocalWorkspace(source_root, workspace_id="cancelled-source"),
                    binding=SyncBinding(
                        target_workspace=LocalWorkspace(
                            target_root, workspace_id="cancelled-target"
                        ),
                        source_conflict_policy="require_revision",
                        max_file_bytes=1024,
                    ),
                ),
                default=True,
            )
            recovery_app.register_agent(
                AgentSpec(
                    name="cancelled-finalize-agent",
                    model="scripted-model",
                    provider_name="cancelled-finalize-provider",
                )
            )
            retained_owner = None
            if recovery_control.startswith("same_app"):
                # Keep this caller inside the bounded poll until cancellation
                # is delivered; the production 10ms budget is not a test barrier.
                monkeypatch.setattr(
                    lifecycle_module, "_LAZY_ENVIRONMENT_CLEANUP_ADMISSION_BUDGET_SECONDS", 1.0
                )
                recovery_app = app
                retained_owner = app._environment_lifecycle._active_environment_setups[
                    "cancelled-finalize-session"
                ]
            clear_started = asyncio.Event()
            allow_clear = asyncio.Event()
            clear_failure = OSError("completion marker clear failed after caller cancellation")
            original_clear = recovery_app._environment_lifecycle.clear_completion_finalization

            async def blocked_clear(**kwargs):
                clear_started.set()
                await allow_clear.wait()
                if recovery_control.endswith("cancel_then_clear_failure") or retained_owner:
                    raise clear_failure
                return await original_clear(**kwargs)

            monkeypatch.setattr(
                recovery_app._environment_lifecycle, "clear_completion_finalization", blocked_clear
            )
            recovery_task = asyncio.create_task(recovery_app.recover_incomplete_session(request))
            try:
                await asyncio.wait_for(clear_started.wait(), timeout=10)
                settlement = (
                    None if retained_owner is None else retained_owner.cleanup_settlement_task
                )
                if retained_owner is not None:
                    assert settlement is not None and not settlement.done()
                    assert binding.finalize_calls == 2
                recovery_task.cancel("cancel public completion recovery")
                assert recovery_task.cancelling() == 1
                await asyncio.sleep(0)
                assert not recovery_task.done()
                allow_clear.set()
                with pytest.raises(asyncio.CancelledError) as raised:
                    await recovery_task
                assert raised.value.args == ("cancel public completion recovery",)
                assert recovery_task.cancelled()
                assert recovery_task.cancelling() == 1
                if recovery_control == "cancel_then_clear_failure":
                    seen: set[int] = set()
                    pending: list[BaseException] = [raised.value]
                    originals: list[BaseException] = []
                    while pending:
                        error = pending.pop()
                        if id(error) in seen:
                            continue
                        seen.add(id(error))
                        originals.append(error)
                        if isinstance(error, BaseExceptionGroup):
                            pending.extend(error.exceptions)
                        if error.__cause__ is not None:
                            pending.append(error.__cause__)
                    assert clear_failure in originals
                    retained = await store.load_checkpoint("cancelled-finalize-session")
                    assert retained is not None
                    assert "pending_completion_finalization" in retained
                if retained_owner is not None:
                    assert settlement is not None
                    outcome = await asyncio.wait_for(asyncio.shield(settlement), timeout=10)
                    assert outcome.error is clear_failure
                    assert retained_owner.cleanup_error is clear_failure
                    assert retained_owner.cleanup_requires_finalize_retry is False
                    assert (
                        app._environment_lifecycle._active_environment_setups[
                            "cancelled-finalize-session"
                        ]
                        is retained_owner
                    )
                    retained = await store.load_checkpoint("cancelled-finalize-session")
                    assert retained is not None
                    assert "pending_completion_finalization" in retained
                    assert (
                        retained_owner.pending_completion_marker_clear
                        == retained["pending_completion_finalization"]
                    )
            finally:
                allow_clear.set()
                await asyncio.gather(recovery_task, return_exceptions=True)
                monkeypatch.setattr(
                    recovery_app._environment_lifecycle,
                    "clear_completion_finalization",
                    original_clear,
                )
            if recovery_control == "cancel_then_clear_failure":
                recovery = await recovery_app.recover_incomplete_session(request)
            if retained_owner is not None:
                if recovery_control == "same_app_marker_conflict":
                    expected_marker = retained_owner.pending_completion_marker_clear
                    assert expected_marker is not None
                    conflicting_marker = {
                        **expected_marker,
                        "binding_generation_id": "different-binding-generation",
                    }

                    async def replace_marker(marker):
                        await store.transform_checkpoint(
                            "cancelled-finalize-session",
                            lambda _session, current: {
                                **(current or {}),
                                "pending_completion_finalization": marker,
                            },
                        )

                    await replace_marker(conflicting_marker)
                    assert not await app.drain_environment_cleanups(timeout_s=0.1)
                    assert binding.finalize_calls == 2
                    conflicted = await store.load_checkpoint("cancelled-finalize-session")
                    assert conflicted is not None
                    assert conflicted["pending_completion_finalization"] == conflicting_marker
                    assert retained_owner.pending_completion_marker_clear == expected_marker
                    await replace_marker(expected_marker)
                assert await app.drain_environment_cleanups(timeout_s=10), (
                    retained_owner.cleanup_error,
                    retained_owner.cleanup_requires_finalize_retry,
                    retained_owner.cleanup_release_safe,
                    tuple(app._environment_lifecycle._active_environment_setups),
                    tuple(app._environment_lifecycle._deferred_run_fence_release_tasks),
                )
                assert binding.finalize_calls == 2
                assert "cancelled-finalize-session" not in (
                    app._environment_lifecycle._active_environment_setups
                )
                persisted_events = await store.load_events("cancelled-finalize-session")
                assert sum(event.type is EventType.MODEL_STARTED for event in persisted_events) == 1
            assert recovery_provider.requests == []
        recovered = await store.load("cancelled-finalize-session")
        checkpoint = await store.load_checkpoint("cancelled-finalize-session")
        if recovery is not None:
            assert recovery.actions == (expected_action,)
        assert recovered is not None
        assert recovered.status is SessionStatus.FAILED
        assert checkpoint is not None
        assert "pending_completion_finalization" not in checkpoint

    asyncio.run(exercise_contract())


async def _collect_run(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]
