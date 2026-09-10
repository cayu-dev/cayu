"""Completion-only recovery must not admit tools that it will never dispatch."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from itertools import product

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    EventType,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    InMemorySessionStore,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SessionStatus,
    SQLiteSessionStore,
    SyncBinding,
    Tool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolSpec,
)
from cayu.environments._finalization_disposal import checkpoint_finalization_disposal
from cayu.runtime._environment_exposure import require_environment_exposed
from cayu.runtime.sessions import SessionRunFenced


@pytest.mark.parametrize(
    ("environment_kind", "handoff", "backend", "changed_requirement"),
    [
        *product(["static", "factory"], ["raw_marker"], ["memory", "sqlite"], [False, True]),
        ("factory", "expired_claim", "memory", True),
        ("factory", "expired_claim", "sqlite", True),
        ("disposal", "validated", "memory", True),
        ("disposal", "validated", "sqlite", True),
        ("disposal", "raw_marker", "memory", True),
        ("disposal", "raw_marker", "sqlite", True),
    ],
)
def test_completion_recovery_does_not_require_new_tool_dependency(
    tmp_path, monkeypatch, changed_requirement, backend, handoff, environment_kind
):
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / "result.txt").write_text("committed output", encoding="utf-8")

    class UnusedTool(Tool):
        async def run(self, ctx, args):
            raise AssertionError("Completion recovery cannot dispatch tools")

    class CancelOnceBinding(SyncBinding):
        pending = True

        async def finalize(self, bound, *, outcome=None, metadata=None):
            if self.pending:
                self.pending = False
                (target / "result.txt").write_text("recovered output", encoding="utf-8")
                if environment_kind == "disposal":
                    # Publish output through real SyncBinding before recording
                    # the factory-owned disposal prefix and interrupting it.
                    await super().finalize(bound, outcome=outcome, metadata=metadata)
                    await checkpoint_finalization_disposal({"allocation": "completion-target"})
                    raise OSError("seed pending factory disposal")
                raise asyncio.CancelledError("seed pending completion finalization")
            if environment_kind == "disposal":
                # The actual workspace effect has already settled. This fixture
                # owns no guest resource and must never repeat that effect.
                return None
            return await super().finalize(bound, outcome=outcome, metadata=metadata)

    async def scenario():
        clock_offset = timedelta()

        def clock():
            return datetime.now(UTC) + clock_offset

        store = (
            InMemorySessionStore(ownership_clock=clock)
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "completion.sqlite", ownership_clock=clock)
        )
        initial_store = store
        factory_requests = []
        disposal_requests = []

        def make_app(*, recovering):
            app = CayuApp(session_store=store, enable_logging=False, clock=clock)
            provider = ScriptedModelProvider(
                [] if recovering else [[ModelStreamEvent.completed({"finish_reason": "stop"})]],
                name="completion-provider",
            )
            app.register_provider(provider, default=True)
            binding_type = SyncBinding if recovering else CancelOnceBinding
            environment = Environment(
                EnvironmentSpec(name="completion"),
                workspace=LocalWorkspace(source, workspace_id="completion-source"),
                binding=binding_type(
                    target_workspace=LocalWorkspace(target, workspace_id="completion-target"),
                    source_conflict_policy="require_revision",
                    max_file_bytes=1024,
                ),
            )
            if environment_kind in {"factory", "disposal"}:

                class Factory(EnvironmentFactory):
                    async def recover_finalization_disposal(self, request, state):
                        assert state == {"allocation": "completion-target"}
                        assert request.reconnect_metadata == {"target": "completion-target"}
                        assert request.operation is EnvironmentFactoryOperation.RECONNECT
                        assert not request.execution_requirements.tool_requirements
                        disposal_requests.append(request)

                    async def create(self, request):
                        factory_requests.append((recovering, request))
                        if recovering:
                            assert request.operation is EnvironmentFactoryOperation.RECONNECT
                            assert request.reconnect_metadata == {"target": "completion-target"}
                            assert not request.execution_requirements.tool_requirements
                        return EnvironmentFactoryResult(
                            environment=environment,
                            reconnect_metadata={"target": "completion-target"},
                        )

                app.register_environment_factory(
                    EnvironmentSpec(name="completion"), Factory(), default=True
                )
            else:
                app.register_environment(environment, default=True)
            requirements = ()
            if recovering and changed_requirement:
                requirements = (
                    ToolExecutionRequirement(
                        name="new_dependency",
                        alternatives=(ToolExecutableRequirement(executable="rg"),),
                    ),
                )
            app.register_agent(
                AgentSpec(
                    name="agent", model="scripted-model", provider_name="completion-provider"
                ),
                tools=[UnusedTool(ToolSpec(name="unused", execution_requirements=requirements))],
            )
            return app, provider

        app, _ = make_app(recovering=False)
        events = []
        try:
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="completion",
                    messages=[Message.text("user", "finish")],
                )
            ):
                events.append(event)
        except OSError as error:
            assert environment_kind == "disposal"
            assert str(error) == "seed pending factory disposal"
        if environment_kind != "disposal":
            assert events[-1].type is EventType.SESSION_FAILED
        seeded = await store.load("completion")
        assert seeded is not None and seeded.status is SessionStatus.FAILED
        checkpoint = await store.load_checkpoint("completion")
        assert checkpoint is not None and "pending_completion_finalization" in checkpoint
        if environment_kind == "disposal":
            assert checkpoint["pending_completion_finalization"]["disposal_state"] == {
                "allocation": "completion-target"
            }
        source_before_recovery = (source / "result.txt").read_text(encoding="utf-8")
        if backend == "sqlite":
            # Independent durable client and app reconstruction. The original
            # process-local cleanup owner stays alive until both drains settle.
            store = SQLiteSessionStore(tmp_path / "completion.sqlite", ownership_clock=clock)
        recovery_app, recovery_provider = make_app(recovering=True)
        bind = recovery_app._environment_lifecycle.bind
        cleanup_bindings = []

        async def check_cleanup_binding(**kwargs):
            result = await bind(**kwargs)
            if kwargs.get("completion_recovery") is not None and result.error is None:
                environment = result.registered_environment
                assert environment is not None and environment.environment_exposure is None
                context = kwargs["invocation_context"].with_registered_environment(
                    environment, validated_profile=kwargs["execution_profile"]
                )
                with pytest.raises(RuntimeError, match="runtime-admitted exposure authority"):
                    require_environment_exposed(
                        environment,
                        session=kwargs["session"],
                        invocation_context=context,
                        registered_agent=kwargs["registered_agent"],
                        execution_profile=kwargs["execution_profile"],
                    )
                cleanup_bindings.append(environment)
            return result

        monkeypatch.setattr(recovery_app._environment_lifecycle, "bind", check_cleanup_binding)
        try:
            if handoff != "validated":
                authorize = recovery_app._environment_lifecycle.authorize_completion_recovery

                async def substitute_raw_marker(**kwargs):
                    nonlocal clock_offset
                    authority = await authorize(**kwargs)
                    if handoff == "expired_claim":
                        # Expire the real store-owned lease after token creation;
                        # do not construct a timeout exception or edit its record.
                        clock_offset += timedelta(minutes=6)
                        return authority
                    return kwargs["marker"]

                monkeypatch.setattr(
                    recovery_app._environment_lifecycle,
                    "authorize_completion_recovery",
                    substitute_raw_marker,
                )
                refusal = (
                    "live recovery claim"
                    if handoff == "expired_claim"
                    else "private invocation authority"
                )
                with pytest.raises(SessionRunFenced, match=refusal):
                    await recovery_app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id="completion", reason="reject unauthenticated handoff"
                        )
                    )
                unchanged = await store.load_checkpoint("completion")
                assert unchanged is not None
                assert (
                    unchanged["pending_completion_finalization"]
                    == checkpoint["pending_completion_finalization"]
                )
                assert (source / "result.txt").read_text(encoding="utf-8") == source_before_recovery
                assert cleanup_bindings == []
                assert disposal_requests == []
                assert not any(recovering for recovering, _ in factory_requests)
                return
            result = await recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="completion", reason="settle output")
            )
            assert result.actions == (
                IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_FINALIZATION,
            )
            assert recovery_provider.requests == []
            assert len(cleanup_bindings) == (0 if environment_kind == "disposal" else 1)
            assert len(disposal_requests) == (1 if environment_kind == "disposal" else 0)
            assert sum(recovering for recovering, _ in factory_requests) == (
                1 if environment_kind == "factory" else 0
            )
            session = await store.load("completion")
            assert session is not None and session.status is SessionStatus.FAILED
            checkpoint = await store.load_checkpoint("completion")
            assert checkpoint is not None and "pending_completion_finalization" not in checkpoint
            assert (source / "result.txt").read_text(encoding="utf-8") == "recovered output"
            assert not any(
                event.type is EventType.ENVIRONMENT_LIFECYCLE_TRANSITION
                and event.payload.get("phase") == "exposure"
                for event in result.events
            )
            if changed_requirement:
                ordinary = [
                    event
                    async for event in recovery_app.run(
                        RunRequest(
                            agent_name="agent",
                            session_id="ordinary-new-requirement",
                            messages=[Message.text("user", "run normally")],
                        )
                    )
                ]
                assert ordinary[-1].type is EventType.SESSION_FAILED
                assert not any(event.type is EventType.MODEL_STARTED for event in ordinary)
        finally:
            # Retain and settle the original in-process owner even if the
            # fresh recovery entrance rejects; no test resources are orphaned.
            try:
                assert recovery_provider.requests == []
            finally:
                assert await recovery_app.drain_environment_cleanups(timeout_s=10)
                assert await app.drain_environment_cleanups(timeout_s=10), [
                    (
                        owner.cleanup_error,
                        owner.cleanup_requires_finalize_retry,
                        owner.pending_completion_marker_clear,
                    )
                    for owner in app._environment_lifecycle._active_environment_setups.values()
                ]
                if backend == "sqlite":
                    await store.close()
                    await initial_store.close()

    asyncio.run(scenario())
