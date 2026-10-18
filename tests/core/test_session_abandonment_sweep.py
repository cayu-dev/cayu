from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from tests.core.session_store_conformance.support import _public_authority_alias_codec
from tests.core.test_session_execution_presence import _stores
from tests.core.test_tool_effect_reconciliation_registration import _spec
from tests.core.test_tool_round_execution_identities import _tool_call_response
from tests.runtime.runtime_environment_support import RecordingEnvironmentFactory
from tests.runtime.runtime_provider_support import VersionedFakeProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolEffect,
    ToolSpec,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.tool_effects import (
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.sessions import IncompleteSessionRecoveryAction, InMemorySessionStore
from cayu.storage import SQLiteSessionStore
from cayu.tasks import TaskCreate
from cayu.tasks.memory import InMemoryTaskStore


def _identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class _UnknownEffect(Tool):
    spec = ToolSpec(
        name="record",
        effect=ToolEffect.EXTERNAL,
        execution_profile_identity=_identity("tests:abandonment-sweep:effect"),
    )

    def __init__(self):
        self.calls = 0

    async def run(self, ctx, args):
        self.calls += 1
        raise ConnectionError("External acknowledgement lost.")


class _RetryableReconciler:
    async def reconcile(self, *, context, receipt):
        return ToolEffectReconciliationResult(
            outcome="not_found", observation="outcome_unknown", retryable=True
        )


class _ReconnectFactory(RecordingEnvironmentFactory):
    @property
    def execution_profile_identity(self):
        return _identity("tests:abandonment-sweep:factory")


async def _interrupted_session(store, *, clock=None, factory=False):
    tasks = InMemoryTaskStore()
    app = CayuApp(session_store=store, task_store=tasks, clock=clock, enable_logging=False)
    provider = VersionedFakeProvider(_tool_call_response(7))
    app.register_provider(provider, default=True)
    if factory:
        spec = EnvironmentSpec(
            name="remote",
            execution_profile_identity=_identity("tests:abandonment-sweep:environment"),
        )
        app.register_environment_factory(
            spec,
            _ReconnectFactory(Environment(spec), reconnect_metadata={"resource": "example"}),
            default=True,
        )
    tool = _UnknownEffect()
    app.register_agent(
        AgentSpec(name="agent", model="fake-model"),
        tools=[tool],
        tool_effect_reconcilers={
            "record": ToolEffectReconciliationRegistration(
                reconciler=_RetryableReconciler(), spec=_spec(supports_lookup=True)
            )
        },
    )
    events = [
        event
        async for event in app.run(
            RunRequest(
                session_id="sweep", agent_name="agent", messages=[Message.text("user", "go")]
            )
        )
    ]
    assert (await store.load("sweep")).status is SessionStatus.INTERRUPTED
    started = next(event for event in events if event.type is EventType.TOOL_CALL_STARTED)
    await tasks.create_task(TaskCreate(task_id="job", type="job", session_id="sweep"))
    await tasks.cancel_task("job")
    return app, started, provider, tool


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
def test_abandonment_sweep_converges_with_generic_factory_reconnect(
    backend, request, sqlite_resources
):
    async def scenario():
        async with _stores(backend, request, sqlite_resources) as (store, _):
            app, _, provider, tool = await _interrupted_session(store, factory=True)
            result = await app.abandon_session("sweep")
            assert result.status is SessionStatus.FAILED
            before = await store.load("sweep")
            checkpoints = runtime_checkpoint_session_store(store)
            checkpoint = await checkpoints.load_checkpoint("sweep")
            assert checkpoint["environment_factory_reconnect"] == {
                "remote": {"resource": "example"}
            }
            for _ in range(3):
                assert (
                    await app.recover_abandoned_sessions(
                        inactive_for_seconds=0, abandon_sessions_of_terminal_tasks=True
                    )
                    == ()
                )
            assert await store.load("sweep") == before
            assert await checkpoints.load_checkpoint("sweep") == checkpoint
            assert len(provider.requests) == tool.calls == 1

    asyncio.run(scenario())


@pytest.mark.store_semantics
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("refresh_activity", [False, True])
def test_retained_abandonment_rechecks_inactivity_after_selection(
    backend, refresh_activity, sqlite_resources, monkeypatch
):
    async def scenario():
        now = [datetime.now(UTC).replace(microsecond=0)]
        async with sqlite_resources as resources:
            codec = _public_authority_alias_codec()
            store = (
                InMemorySessionStore(
                    ownership_clock=lambda: now[0], public_authority_alias_codec=codec
                )
                if backend == "memory"
                else resources.own(
                    SQLiteSessionStore(
                        resources.path(),
                        ownership_clock=lambda: now[0],
                        public_authority_alias_codec=codec,
                    )
                )
            )
            app, started, provider, tool = await _interrupted_session(store, clock=lambda: now[0])
            now[0] += timedelta(hours=2)
            original_list = store.list_sessions
            selected = False

            async def interleaved_list(query):
                nonlocal selected
                page = await original_list(query)
                if query.status is SessionStatus.INTERRUPTED and page.sessions and not selected:
                    selected = True
                    if refresh_activity:
                        target = await app.inspect_tool_effect(
                            "sweep",
                            tool_round_id=started.payload["tool_round_id"],
                            tool_call_id=started.payload["tool_call_id"],
                        )
                        request = ToolEffectReconciliationRequest(
                            **target.model_dump(), lookup=True
                        )
                        _ = [event async for event in app.reconcile_tool_effect(request)]
                        current = await store.load("sweep")
                        assert current.status is SessionStatus.INTERRUPTED
                        assert current.last_activity_at == now[0]
                return page

            monkeypatch.setattr(store, "list_sessions", interleaved_list)
            [result] = await app.recover_abandoned_sessions(
                inactive_for_seconds=3600, abandon_sessions_of_terminal_tasks=True
            )
            assert selected
            if refresh_activity:
                assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
                assert (await store.load("sweep")).status is SessionStatus.INTERRUPTED
                checkpoint = await runtime_checkpoint_session_store(store).load_checkpoint("sweep")
                assert "pending_tool_round" in checkpoint
                assert "abandoned_session" not in checkpoint
            else:
                assert IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT in result.actions
                assert (await store.load("sweep")).status is SessionStatus.FAILED
            assert len(provider.requests) == tool.calls == 1

    asyncio.run(scenario())
