"""Retained selected targets remain manageable after background-mode adoption."""

from __future__ import annotations

import asyncio
import multiprocessing

import pytest
from tests.core.test_execution_profiles import RecordingExecutionProfilePolicy
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_provider_operation_offline_recovery import (
    _CancellableOfflineOperationAdapter,
    _IdempotentAmbiguousStartAdapter,
    _SimulatedProcessLoss,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    IncompleteSessionRecoveryRequest,
    InterruptSessionRequest,
    Message,
    ModelFailoverPolicy,
    ModelTarget,
    ResumeRequest,
    RunRequest,
)
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.providers.base import ModelProviderError, ModelStreamEvent
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.providers.operations import (
    ProviderOperationConnection,
    ProviderOperationMode,
    ProviderOperationStatus,
)
from cayu.runtime.execution_profiles import (
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfileMismatchError,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyResult,
)
from cayu.runtime.retry_policy import RetryPolicy
from cayu.sessions.base import InMemorySessionStore
from cayu.storage.sqlite import SQLiteSessionStore


class _Adapter(_CancellableOfflineOperationAdapter):
    committed = None

    async def start(self, request):
        self.start_calls += 1
        self.start_requests.append(request)

        async def events():
            if self.committed is not None:
                self.committed.set()
                await asyncio.Event().wait()
            raise _SimulatedProcessLoss("lost worker after durable selected operation start")
            yield

        return ProviderOperationConnection(
            state=self.state, status=ProviderOperationStatus.IN_PROGRESS, events=events()
        )

    async def cancel(self, state):
        if self.status is ProviderOperationStatus.COMPLETED:
            self.cancel_calls.append(state)
            return await self.retrieve(state)
        return await super().cancel(state)


class _ExactStartAdapter(_IdempotentAmbiguousStartAdapter):
    async def start(self, request):
        self.start_requests.append(request)
        return await super().start(request)


class _Provider(_RecoveryProvider):
    def __init__(self, name, *, background=False, ambiguous_start=False):
        super().__init__(name, behavior_version="background" if background else "synchronous")
        self.background = background
        self.adapter = _ExactStartAdapter() if ambiguous_start else _Adapter()

    @property
    def provider_operation_mode(self):
        return (
            ProviderOperationMode.BACKGROUND
            if self.background
            else ProviderOperationMode.SYNCHRONOUS
        )

    @property
    def provider_operations(self):
        return self.adapter if self.background else None

    @property
    def stream_deadlines(self):
        return ProviderStreamDeadlines(semantic_progress_timeout_s=30)

    async def stream(self, request):
        self.requests.append(request)
        if request.model == "small":
            raise ModelProviderError("busy", provider=self.name, status_code=503, retryable=True)
        yield ModelStreamEvent.text_delta("selected answer")
        yield ModelStreamEvent.completed()


def _app(store, *, background, separate_provider, ambiguous_start=False, changed_target=False):
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        execution_profile_policy=RecordingExecutionProfilePolicy(
            ExecutionProfilePolicyResult(
                action=ExecutionProfilePolicyAction.ADOPT,
                reason="Authorize background execution of the unchanged targets.",
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
            )
        ),
    )
    primary = _Provider("primary", background=background, ambiguous_start=ambiguous_start)
    selected = (
        _Provider("backup", background=background, ambiguous_start=ambiguous_start)
        if separate_provider
        else primary
    )
    if changed_target:
        selected.behavior_version = "unadmitted-replacement"
    app.register_provider(primary, default=True)
    if separate_provider:
        app.register_provider(selected)
    app.register_agent(AgentSpec(name="agent", model="small"))
    return app, primary, selected


async def _seed_selected_session(store, *, separate_provider):
    app, _, selected = _app(store, background=False, separate_provider=separate_provider)
    events = [
        event
        async for event in app.run(
            RunRequest(
                agent_name="agent",
                session_id="adopted",
                messages=[Message.text("user", "first")],
                retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
                failover=ModelFailoverPolicy(
                    fallbacks=(ModelTarget(provider_name=selected.name, model="large"),),
                    max_total_attempts=2,
                ),
            )
        )
    ]
    assert events[-1].type is EventType.SESSION_COMPLETED


def _adoption_request():
    return ResumeRequest(
        session_id="adopted",
        messages=[Message.text("user", "second")],
        retry_policy=RetryPolicy(max_attempts=1, initial_delay_s=0),
        profile_adoption=ExecutionProfileAdoptionIntent(
            idempotency_key="background",
            reason="Use background mode.",
            requested_by=ResolutionActor(
                subject="maintainer", source=ResolutionActorSource.REQUEST
            ),
        ),
    )


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("separate_provider", [False, True])
@pytest.mark.parametrize("action", ["recover", "cancel", "recover_start", "completion_wins"])
def test_public_background_adoption_recovers_selected_target(
    tmp_path, backend, separate_provider, action
):
    async def run():
        path = tmp_path / "adopted.sqlite"
        store = InMemorySessionStore() if backend == "memory" else SQLiteSessionStore(path)
        try:
            await _seed_selected_session(store, separate_provider=separate_provider)
            app, _, selected = _app(
                store,
                background=True,
                separate_provider=separate_provider,
                ambiguous_start=action == "recover_start",
            )
            with pytest.raises(_SimulatedProcessLoss):
                async for _ in app.resume(_adoption_request()):
                    pass
            assert selected.adapter.start_calls == 1
            assert selected.adapter.start_requests[0].request.model == "large"
            stage = await store.load_active_model_completion_stage("adopted")
            assert stage is not None and stage.stage.intent["requested_model"] == "large"
            assert (await store.load("adopted")).model == "small"
            start_keys = selected.adapter.start_keys if action == "recover_start" else None
            before_events = await store.load_events("adopted")
            if backend == "sqlite":
                await store.close()
                store = SQLiteSessionStore(path)
            app, primary, selected = _app(
                store,
                background=True,
                separate_provider=separate_provider,
                ambiguous_start=action == "recover_start",
            )
            if action == "recover":
                before_session = await store.load("adopted")
                before_checkpoint = await store.load_checkpoint("adopted")
                changed_app, _, changed_provider = _app(
                    store, background=True, separate_provider=separate_provider, changed_target=True
                )
                with pytest.raises(ExecutionProfileMismatchError):
                    await changed_app.recover_incomplete_session(
                        IncompleteSessionRecoveryRequest(
                            session_id="adopted", inactive_for_seconds=0
                        )
                    )
                after_session = await store.load("adopted")
                assert after_session.model_dump(exclude={"updated_at", "last_activity_at"}) == (
                    before_session.model_dump(exclude={"updated_at", "last_activity_at"})
                )
                assert await store.load_checkpoint("adopted") == before_checkpoint
                assert await store.load_active_model_completion_stage("adopted") == stage
                assert not changed_provider.adapter.retrieve_calls
                assert not changed_provider.adapter.cancel_calls
                assert changed_provider.adapter.start_calls == 0
                # Execution rejection may publish its profile diagnostic; it
                # must not claim the stage or touch the external operation.
                rejection_events = (await store.load_events("adopted"))[len(before_events) :]
                assert all(
                    event.type is EventType.SESSION_EXECUTION_PROFILE_REJECTED
                    for event in rejection_events
                )
            if action in {"recover", "recover_start"}:
                selected.adapter.status = ProviderOperationStatus.COMPLETED
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="adopted",
                        inactive_for_seconds=0,
                    )
                )
                assert selected.adapter.retrieve_calls
                assert await store.load_active_model_completion_stage("adopted") is None
                if action == "recover_start":
                    assert selected.adapter.recovery_keys == start_keys
            else:
                if action == "completion_wins":
                    selected.adapter.status = ProviderOperationStatus.COMPLETED
                events = [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(
                            session_id="adopted",
                            reason="Cancel the retained selected operation.",
                        )
                    )
                ]
                assert events[-1].type is EventType.SESSION_INTERRUPTED
                assert selected.adapter.cancel_calls == [selected.adapter.state]
            durable = await store.load_events("adopted")
            new_events = [
                event for event in durable if event.id not in {e.id for e in before_events}
            ]
            completed = [event for event in new_events if event.type is EventType.MODEL_COMPLETED]
            assert len(completed) == (0 if action == "cancel" else 1)
            if completed:
                assert completed[0].payload["requested_model"] == "large"
                assert completed[0].payload["provider_name"] == selected.name
            operation_events = [
                event for event in new_events if event.type.value.startswith("provider.operation.")
            ]
            assert operation_events
            assert all(event.payload["model"] == "large" for event in operation_events)
            assert all(event.payload["provider"] == selected.name for event in operation_events)
            assert primary.adapter.start_calls == selected.adapter.start_calls == 0
            assert not primary.requests and not selected.requests
            assert (await store.load("adopted")).model == "small"
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(run())


def _background_worker(path, committed):
    async def run():
        store = SQLiteSessionStore(path)
        try:
            await _seed_selected_session(store, separate_provider=True)
            app, _, selected = _app(store, background=True, separate_provider=True)
            selected.adapter.committed = committed
            async for _ in app.resume(_adoption_request()):
                pass
        finally:
            await store.close()

    asyncio.run(run())


@pytest.mark.parametrize("action", ["recover", "cancel"])
def test_background_adoption_survives_real_process_death(tmp_path, action):
    path = tmp_path / "killed.sqlite"
    context = multiprocessing.get_context("spawn")
    committed = context.Event()
    process = context.Process(target=_background_worker, args=(path, committed))
    process.start()
    try:
        assert committed.wait(120), f"Selected operation did not start; exit={process.exitcode}"
        process.kill()
        process.join(10)
        assert not process.is_alive() and process.exitcode != 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        process.close()

    async def recover():
        store = SQLiteSessionStore(path)
        try:
            app, primary, selected = _app(store, background=True, separate_provider=True)
            stage = await store.load_active_model_completion_stage("adopted")
            assert stage is not None
            assert stage.stage.intent["provider_name"] == "backup"
            assert stage.stage.intent["requested_model"] == "large"
            before = await store.load_events("adopted")
            starts = [
                event for event in before if event.type is EventType.PROVIDER_OPERATION_STARTED
            ]
            assert len(starts) == 1
            if action == "recover":
                selected.adapter.status = ProviderOperationStatus.COMPLETED
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id="adopted", inactive_for_seconds=0)
                )
                assert selected.adapter.retrieve_calls == [selected.adapter.state]
                assert await store.load_active_model_completion_stage("adopted") is None
            else:
                emitted = [
                    event
                    async for event in app.interrupt_session(
                        InterruptSessionRequest(session_id="adopted", reason="Worker died.")
                    )
                ]
                assert emitted[-1].type is EventType.SESSION_INTERRUPTED
                assert selected.adapter.cancel_calls == [selected.adapter.state]
            durable = await store.load_events("adopted")
            assert [
                event for event in durable if event.type is EventType.PROVIDER_OPERATION_STARTED
            ] == starts
            assert sum(event.type is EventType.MODEL_COMPLETED for event in durable) == (
                2 if action == "recover" else 1
            )
            assert primary.adapter.start_calls == selected.adapter.start_calls == 0
            assert not primary.requests and not selected.requests
            assert (await store.load("adopted")).model == "small"
        finally:
            await store.close()

    asyncio.run(recover())


def test_selected_background_completion_wins_live_interruption():
    async def run():
        store = InMemorySessionStore()
        await _seed_selected_session(store, separate_provider=True)
        app, primary, selected = _app(store, background=True, separate_provider=True)
        selected.adapter.status = ProviderOperationStatus.COMPLETED
        selected.adapter.committed = asyncio.Event()

        async def consume():
            return [event async for event in app.resume(_adoption_request())]

        task = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(selected.adapter.committed.wait(), timeout=30)
            emitted = [
                event
                async for event in app.interrupt_session(
                    InterruptSessionRequest(session_id="adopted", reason="Completion race.")
                )
            ]
            run_events = await task
            assert emitted[-1].type is EventType.SESSION_INTERRUPTED
            assert run_events[-1].type is EventType.SESSION_INTERRUPTED
            assert selected.adapter.cancel_calls == [selected.adapter.state]
            assert selected.adapter.start_calls == 1
            assert primary.adapter.start_calls == 0
            assert not primary.requests and not selected.requests
            completed = [
                event
                for event in await store.load_events("adopted")
                if event.type is EventType.MODEL_COMPLETED
            ]
            assert len(completed) == 2
            assert completed[-1].payload["requested_model"] == "large"
            assert completed[-1].payload["provider_name"] == "backup"
            assert await store.load_active_model_completion_stage("adopted") is None
            assert (await store.load("adopted")).model == "small"
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
