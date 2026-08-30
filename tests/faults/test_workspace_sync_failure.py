from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from tests.core._event_projection_support import private_events_for_public_events
from tests.environments.sync_ownership_assertions import assert_sync_resources_owned

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SyncBinding,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.runtime import InMemoryEventSink, SessionStatus
from cayu.storage import SQLiteSessionStore
from cayu.workspaces import LocalWorkspace


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


def test_sync_binding_partial_finalize_failure_is_durable_and_runtime_abandons(
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
    binding = SyncBinding(target_workspace=target)
    store_path = tmp_path / "sessions.sqlite"
    store = SQLiteSessionStore(store_path)
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
    app = CayuApp(session_store=store, enable_logging=False)
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
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="workspace-sync-assistant",
                        session_id="workspace-sync-failure",
                        messages=[Message.text("user", "Apply the workspace changes.")],
                    )
                )
            ]
        finally:
            await store.close()

        partial_source = {
            path.name: path.read_text(encoding="utf-8") for path in source_root.iterdir()
        }
        reopened = SQLiteSessionStore(store_path)
        try:
            durable_session = await reopened.load("workspace-sync-failure")
            durable_events = await reopened.load_events("workspace-sync-failure")
        finally:
            await reopened.close()

        # The failure is already durable and the terminal runtime has no public
        # BoundWorkspace handle through which a caller could retry. Its final
        # abort therefore abandons this exact generation instead of leaking the
        # fixed target for the lifetime of the app.
        assert binding._states == {}
        rebound = await binding.bind(source, None, session_id="workspace-sync-rebind")
        assert_sync_resources_owned(rebound, expected=True)
        binding.abandon(rebound)
        assert_sync_resources_owned(rebound, expected=False)
        assert binding._states == {}
        return events, partial_source, durable_session, durable_events

    events, partial_source, session, durable_events = asyncio.run(exercise_contract())

    assert source.failed_writes == 1
    assert partial_source == {
        "a-updated.txt": "updated-a",
        "b-fail.txt": "original-b",
        "removed.txt": "remove-me",
    }

    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    assert [event.type for event in events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    assert [event.type for event in durable_events[-3:]] == [
        EventType.ENVIRONMENT_BINDING_FINALIZE_STARTED,
        EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
        EventType.SESSION_COMPLETED,
    ]
    finalize_failure = durable_events[-2]
    assert finalize_failure.payload["error"] == "forced sync write failure: b-fail.txt"
    assert finalize_failure.payload["error_type"] == "OSError"
    assert durable_events[-1].payload["binding_finalize_error"] == {
        "error": "forced sync write failure: b-fail.txt",
        "error_type": "OSError",
        "outcome": "completed",
        "failures": [
            {
                "phase": "workspace_finalize",
                "error": "forced sync write failure: b-fail.txt",
                "error_type": "OSError",
            }
        ],
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


def test_sync_binding_releases_owner_after_finalize_failure_commit_is_reconciled(
    tmp_path: Path,
) -> None:
    store = FailingFinalizeEvidenceStore()
    store.commit_finalize_evidence_before_failure = True
    app, binding, source, _target, _target_root = _sync_durability_test_app(tmp_path, store)

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
        assert binding._states == {}
        assert (
            "sync-durability-reconciled"
            not in app._environment_lifecycle._active_environment_setups
        )
        assert (
            sum(
                event.type == EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED
                for event in await store.load_events("sync-durability-reconciled")
            )
            == 1
        )
        assert events[-1].type == EventType.SESSION_COMPLETED

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
