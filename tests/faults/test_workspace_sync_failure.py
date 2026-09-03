from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tests.core._event_projection_support import private_events_for_public_events
from tests.core.task_invocation_fixtures import task_backed_session_invocation
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
)
from cayu.runtime import InMemoryEventSink, SessionStatus
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
            await asyncio.wait_for(recovery_started.wait(), timeout=5)
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

        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="sync-durability-reconciled",
                reason="workspace_finalization_reconciliation_test",
            )
        )
        assert recovery.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,
        )
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
            await asyncio.wait_for(binding.finalize_started.wait(), timeout=5)
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


@pytest.mark.parametrize("claimed_task", [False, True])
def test_restarted_completion_finalization_settles_attached_task(
    tmp_path: Path,
    claimed_task: bool,
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
        session_store = SQLiteSessionStore(session_path)
        task_store = SQLiteTaskStore(task_path)
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
        await task_store.create_task(
            TaskCreate(task_id="restarted-finalize-task", type="candidate-build")
        )
        task_worker_id = None
        task_lease_expires_at = None
        if claimed_task:
            claimed = await task_store.claim_task("crashed-finalize-worker", lease_seconds=1)
            assert claimed is not None
            task_worker_id = claimed.worker_id
            task_lease_expires_at = claimed.lease_expires_at

        run_task = asyncio.create_task(
            _collect_run(
                app,
                RunRequest(
                    agent_name="restarted-finalize-agent",
                    session_id="restarted-finalize-session",
                    task_id="restarted-finalize-task",
                    task_worker_id=task_worker_id,
                    task_lease_expires_at=task_lease_expires_at,
                    messages=[Message.text("user", "Finish.")],
                ),
            )
        )
        recovery_session_store = SQLiteSessionStore(session_path)
        recovery_task_store = SQLiteTaskStore(task_path)
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
        try:
            await asyncio.wait_for(binding.finalize_started.wait(), timeout=5)
            pending_checkpoint = await session_store.load_checkpoint("restarted-finalize-session")
            assert pending_checkpoint is not None
            assert pending_checkpoint["pending_completion_finalization"]["task_id"] == (
                "restarted-finalize-task"
            )
            if claimed_task:
                await asyncio.sleep(1.05)
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

            assert recovery.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,
            )
            assert recovery_provider.requests == []
            assert recovered_session is not None
            assert recovered_session.status is SessionStatus.FAILED
            assert recovered_task is not None
            assert recovered_task.status is TaskStatus.FAILED
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
            assert EventType.TASK_FAILED in {event.type for event in durable_events}
            assert EventType.SESSION_FAILED in {event.type for event in durable_events}
        finally:
            binding.allow_finalize.set()
            await asyncio.gather(run_task, return_exceptions=True)
            await recovery_task_store.close()
            await recovery_session_store.close()
            await task_store.close()
            await session_store.close()

    asyncio.run(exercise_contract())


def test_cancelled_completion_finalization_remains_failed_and_recoverable(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "cancelled-source"
    target_root = tmp_path / "cancelled-target"
    source_root.mkdir()
    target_root.mkdir()
    (source_root / "result.txt").write_text("ready", encoding="utf-8")

    class CancelOnceCompletionBinding(SyncBinding):
        cancel_once = True

        async def finalize(self, bound, *, outcome=None, metadata=None):
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

        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="cancelled-finalize-session",
                reason="cancelled_workspace_finalization",
            )
        )
        recovered = await store.load("cancelled-finalize-session")
        checkpoint = await store.load_checkpoint("cancelled-finalize-session")
        assert recovery.actions == (
            IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,
        )
        assert recovered is not None
        assert recovered.status is SessionStatus.FAILED
        assert checkpoint is not None
        assert "pending_completion_finalization" not in checkpoint

    asyncio.run(exercise_contract())


async def _collect_run(app: CayuApp, request: RunRequest) -> list[Event]:
    return [event async for event in app.run(request)]
