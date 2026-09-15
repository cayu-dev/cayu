"""Public fork admission preserves the complete configured candidate authority."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_execution_profiles import RecordingTool
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_failover_stages import _StageMemoryStore, _StageSQLiteStore

from cayu import (
    AgentSpec,
    CayuApp,
    DispatchRequest,
    DispatchStatus,
    EventType,
    ExecutionProfileBehaviorIdentity,
    ForkSessionRequest,
    InMemoryTaskStore,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    ResumeRequest,
    RunRequest,
    ScriptedModelProvider,
    SQLiteTaskStore,
    TaskStoreDispatcher,
    ToolCapabilityCeiling,
)
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    direct_tool_capability_ceiling_component,
    execution_profile_from_session_metadata,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions._model_failover import ModelFailoverSelection
from cayu.sessions.base import session_fork_profile_relationship


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("initial_kind", ["none", "inherited", "changed_cap", "backup_drift"])
def test_routed_fork_narrows_every_candidate_profile_without_dispatch(
    monkeypatch, tmp_path, backend, initial_kind
):

    async def scenario():
        store = (
            _StageMemoryStore()
            if backend == "memory"
            else _StageSQLiteStore(tmp_path / "fork-profile.sqlite")
        )
        primary = ScriptedModelProvider([[ModelStreamEvent.completed()]], name="primary")

        class Backup(ScriptedModelProvider):
            revision = "1"

            @property
            def execution_profile_identity(self):
                return ExecutionProfileBehaviorIdentity(
                    name="tests:fork-backup",
                    behavior_version=self.revision,
                    implementation_version="1",
                )

        backup = Backup([[ModelStreamEvent.completed()]], name="backup")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(
            AgentSpec(name="agent", model="small"),
            tools=[RecordingTool("retained"), RecordingTool("removed")],
        )
        try:
            source_events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="fork-source",
                        messages=[Message.text("user", "source")],
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),)
                        ),
                    )
                )
            ]
            assert source_events[-1].type is EventType.SESSION_COMPLETED
            source = await store.load("fork-source")
            assert source is not None
            source_profile = execution_profile_from_session_metadata(source.metadata)
            source_checkpoint = await store.load_checkpoint(source.id)
            assert source_checkpoint is not None

            def untrusted_transform(_session, checkpoint):
                assert checkpoint is not None and "model_failover" not in checkpoint
                # An identical value is still caller data, not permission to
                # advance the route. Try a changed selection as well.
                checkpoint["model_failover"] = {
                    **source_checkpoint["model_failover"],
                    "candidate_index": 1,
                }
                return checkpoint

            await runtime_checkpoint_session_store(store).transform_checkpoint(
                source.id, untrusted_transform
            )
            assert await store.load_checkpoint(source.id) == source_checkpoint
            # An explicit checkpoint write advances last_activity_at even if the
            # authority projection retains the same state. Fork must not.
            after_callback = await store.load(source.id)
            assert after_callback is not None
            assert (
                after_callback.model_copy(update={"last_activity_at": source.last_activity_at})
                == source
            )
            source = after_callback
            expected_source = await app.snapshot_fork_source(source.id)
            initial_dispatch = initial_kind != "none"
            if initial_kind == "backup_drift":
                # Registration freezes declared identity and names cannot be
                # registered twice. Reconstruct the application as a real
                # deployment with a changed backup, preserving the store.
                backup = Backup([[ModelStreamEvent.completed()]], name="backup")
                backup.revision = "2"
                app = CayuApp(session_store=store, enable_logging=False)
                app.register_provider(primary, default=True)
                app.register_provider(backup)
                app.register_agent(
                    AgentSpec(name="agent", model="small"),
                    tools=[RecordingTool("retained"), RecordingTool("removed")],
                )
            request = ForkSessionRequest(
                source_session_id=source.id,
                session_id="fork-child",
                expected_source=expected_source,
                tool_capability_ceiling=ToolCapabilityCeiling(tool_names=("retained",)),
                initial_invocation=(
                    ResumeRequest(
                        session_id="fork-child",
                        messages=[Message.text("user", "child")],
                        failover=(
                            ModelFailoverPolicy(
                                fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                                max_total_attempts=21,
                            )
                            if initial_kind == "changed_cap"
                            else None
                        ),
                    )
                    if initial_dispatch
                    else None
                ),
                initial_dispatch_id="child-dispatch" if initial_dispatch else None,
            )
            if initial_kind in {"changed_cap", "backup_drift"}:
                with pytest.raises(RuntimeError, match="changed execution authority"):
                    _ = [event async for event in app.fork_session(request)]
                assert await store.load("fork-child") is None
                with pytest.raises(KeyError, match="Session not found"):
                    await store.load_events("fork-child")
                assert await store.load(source.id) == source
                assert await store.load_checkpoint(source.id) == source_checkpoint
                assert len(primary.requests) == 1 and not backup.requests
                return
            events = [event async for event in app.fork_session(request)]
            assert events[-1].type is EventType.SESSION_FORKED
            child = await store.load("fork-child")
            assert child is not None
            relationship = session_fork_profile_relationship(child)
            assert relationship is not None and relationship.source_profile == source_profile
            selected = execution_profile_from_session_metadata(child.metadata)
            assert selected == relationship.selected_profile
            if initial_dispatch:
                assert relationship.initial_invocation_profile is not None
                assert relationship.initial_invocation_profile.model_failover is not None
                assert relationship.initial_dispatch_id == "child-dispatch"
            assert selected.model_failover is not None and source_profile.model_failover is not None
            assert selected.model_failover.plan.max_total_attempts == 20
            assert [
                (entry.provider_name, entry.model, entry.execution_mode)
                for entry in selected.model_failover.plan.candidates
            ] == [("primary", "small", "synchronous"), ("backup", "large", "synchronous")]
            ceiling = direct_tool_capability_ceiling_component(("retained",))
            for before, after in zip(
                source_profile.model_failover.candidate_profiles,
                selected.model_failover.candidate_profiles,
                strict=True,
            ):
                before_profile, after_profile = before.as_profile(), after.as_profile()
                assert (
                    after_profile.component(ExecutionProfileComponentClass.TOOL_VIEW_GRANTS)
                    == ceiling
                )
                assert after_profile.fingerprint != before_profile.fingerprint
                assert [
                    component
                    for component in after_profile.components
                    if component.component_class
                    is not ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
                ] == [
                    component
                    for component in before_profile.components
                    if component.component_class
                    is not ExecutionProfileComponentClass.TOOL_VIEW_GRANTS
                ]
            child_checkpoint = await store.load_checkpoint(child.id)
            assert child_checkpoint is not None
            origin = ModelFailoverSelection.model_validate(child_checkpoint["model_failover"])
            assert origin.session_id == child.id and origin.session_instance_id == child.instance_id
            assert origin.candidate_index == 0 and origin.source_run_epoch == 0
            assert origin.plan == selected.model_failover.plan
            assert not {"stage_id", "attempts_used", "logical_step_id"}.intersection(
                origin.payload()
            )
            assert await store.load_checkpoint(source.id) == source_checkpoint
            assert await store.load(source.id) == source
            replayed = [event async for event in app.fork_session(request)]
            assert [event.id for event in replayed] == [event.id for event in events]
            assert len(primary.requests) == 1 and not backup.requests
        finally:
            if isinstance(store, _StageSQLiteStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("fork_kind", ["full", "partial", "nested", "queued"])
def test_selected_fallback_survives_fork_and_reconstruction(
    monkeypatch, tmp_path, backend, fork_kind, request
):
    case_id = uuid4().hex
    source_id, child_id, grandchild_id = (
        f"{name}-{case_id}" for name in ("source", "child", "grandchild")
    )
    database = tmp_path / "selected-fork.sqlite"
    task_database = tmp_path / "selected-fork-tasks.sqlite"
    if backend == "postgres":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore, PostgresTaskStore

        dsn = request.getfixturevalue("postgres_dsn")

        class StagePostgresStore(PostgresSessionStore):
            model_failover_stage_version = 1
            invocation_lifecycle_command_version = 1

    def fresh_session_store():
        if backend == "postgres":
            return StagePostgresStore(dsn, schema_mode=SchemaMode.CREATE)
        return _StageMemoryStore() if backend == "memory" else _StageSQLiteStore(database)

    def fresh_task_store():
        if fork_kind != "queued":
            return None
        if backend == "postgres":
            return PostgresTaskStore(dsn, schema_mode=SchemaMode.CREATE)
        return InMemoryTaskStore() if backend == "memory" else SQLiteTaskStore(task_database)

    def application(store, tasks):
        dispatcher = None if tasks is None else TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=store, task_store=tasks, dispatcher=dispatcher, enable_logging=False
        )
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.register_agent(AgentSpec(name="agent", model="small"))
        return app, primary, backup, dispatcher

    async def scenario():
        store = fresh_session_store()
        tasks = fresh_task_store()
        app, primary, backup, _ = application(store, tasks)
        try:
            source_events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id=source_id,
                        messages=[Message.text("user", "source")],
                        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                        failover=ModelFailoverPolicy(
                            fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                            max_total_attempts=2,
                        ),
                    )
                )
            ]
            assert source_events[-1].type is EventType.SESSION_COMPLETED
            assert len(primary.requests) == len(backup.requests) == 1
            checkpoint = await store.load_checkpoint(source_id)
            assert checkpoint is not None and checkpoint["model_failover"]["candidate_index"] == 1
            source_session = await store.load(source_id)
            assert source_session is not None
            initial = ResumeRequest(
                session_id=child_id,
                messages=[Message.text("user", "continue child")],
                retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
            )
            request = ForkSessionRequest(
                source_session_id=source_id,
                session_id=child_id,
                expected_source=(
                    None if fork_kind == "partial" else await app.snapshot_fork_source(source_id)
                ),
                transcript_cursor=1 if fork_kind == "partial" else None,
                copy_checkpoint=fork_kind != "partial",
                initial_invocation=initial if fork_kind == "queued" else None,
                initial_dispatch_id="first-child" if fork_kind == "queued" else None,
            )
            events = [event async for event in app.fork_session(request)]
            assert events[-1].payload["model_failover_candidate_index"] == 1
            child = await store.load(child_id)
            assert child is not None
            relationship = session_fork_profile_relationship(child)
            assert relationship is not None and relationship.model_failover_candidate_index == 1
            child_checkpoint = await store.load_checkpoint(child.id)
            assert child_checkpoint is not None
            origin = ModelFailoverSelection.model_validate(child_checkpoint["model_failover"])
            assert origin.session_id == child.id and origin.source_run_epoch == 0
            assert origin.candidate_index == 1
            assert origin.session_instance_id != source_session.instance_id
            assert await store.load_active_model_completion_stage(child.id) is None
            assert await store.load_checkpoint(source_id) == checkpoint
            assert await store.load(source_id) == source_session
            if backend != "memory":
                await store.close()
                store = fresh_session_store()
            if tasks is not None and backend != "memory":
                await tasks.close()
                tasks = fresh_task_store()
            app, resumed_primary, resumed_backup, dispatcher = application(store, tasks)
            assert [event.id async for event in app.fork_session(request)] == [
                event.id for event in events
            ]
            target = child_id
            if fork_kind == "nested":
                nested_request = ForkSessionRequest(
                    source_session_id=child_id,
                    session_id=grandchild_id,
                    expected_source=await app.snapshot_fork_source(child_id),
                )
                nested = [event async for event in app.fork_session(nested_request)]
                assert nested[-1].payload["model_failover_candidate_index"] == 1
                target = grandchild_id
            if fork_kind == "queued":
                assert dispatcher is not None
                with pytest.raises(RuntimeError, match="exact durable dispatch"):
                    _ = [event async for event in app.resume(initial)]
                dispatch_request = DispatchRequest(
                    session_id=initial.session_id,
                    dispatch_id="first-child",
                    messages=initial.messages,
                    max_steps=initial.max_steps,
                    limits=initial.limits,
                    budget_limits=initial.budget_limits,
                    retry_policy=initial.retry_policy,
                    thinking=initial.thinking,
                    structured_output=initial.structured_output,
                    tool_capability_ceiling=initial.tool_capability_ceiling,
                )
                submitted = await app.dispatch(dispatch_request)
                assert submitted.status is DispatchStatus.SUBMITTED
                assert not resumed_primary.requests and not resumed_backup.requests
                result = await dispatcher.process_next(app, worker_id="fork-worker")
                assert result is not None and result.status is DispatchStatus.COMPLETED, result
                replayed = await app.dispatch(dispatch_request)
                assert replayed.task_id == submitted.task_id
                continued = await store.load_events(target)
            else:
                continued = [
                    event
                    async for event in app.resume(
                        ResumeRequest(
                            session_id=target,
                            messages=initial.messages,
                            retry_policy=initial.retry_policy,
                        )
                    )
                ]
            assert continued[-1].type is EventType.SESSION_COMPLETED, continued[-1].payload
            assert not resumed_primary.requests and len(resumed_backup.requests) == 1
            assert resumed_backup.requests[0].model == "large"
            final = await store.load_checkpoint(target)
            assert final is not None
            progress = final["model_failover"]
            assert progress["candidate_index"] == 1 and progress["attempts_used"] == 1
            assert progress["generation"] == 1 and progress["session_id"] == target
            assert progress["stage_id"] != checkpoint["model_failover"]["stage_id"]
            assert await store.load_active_model_completion_stage(target) is None
            finished = await store.load(target)
            assert finished is not None and finished.provider_name == "primary"
        finally:
            if backend != "memory":
                await store.close()
            if tasks is not None and backend != "memory":
                await tasks.close()

    asyncio.run(scenario())
