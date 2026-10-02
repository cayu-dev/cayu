"""CayuApp shutdown: admission, one shared deadline, and a truthful outcome."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.application_lifecycle import (
    _ENTRANCE_KIND,
    ApplicationAdmissionsSealed,
)
from cayu.sessions.base import InMemorySessionStore, RunRequest, SessionStatus
from cayu.storage.memory import InMemoryKnowledgeStore, KnowledgeAccessScope
from cayu.tools.knowledge import RememberKnowledgeTool

# Entrances that start or continue model/tool execution, verifiers, or workers.
GATED_ENTRANCES = frozenset(
    {
        "admit_work_attempt",
        "compact_session",
        "dispatch",
        "dispatch_inline",
        "execute_participant_session",
        "execute_participant_session_to_wait",
        "execute_producer_output",
        "execute_recovery",
        "fork_session",
        "reconcile_tool_effect",
        "recover_collaboration_wait",
        "recover_incomplete_session",
        "recover_incomplete_sessions",
        "recover_model_completion_stage",
        "recover_persisted_event_side_effects",
        "recover_session_continuation",
        "recover_tool_approval",
        "recover_tool_round",
        "recover_user_input",
        "recover_work_attempt",
        "refresh_mcp_toolset",
        "replay_session",
        "resolve_completion_result",
        "resolve_provider_operation",
        "resolve_tool_approval",
        "resolve_user_input",
        "resume",
        "resume_pending_interruption_cascades",
        "run",
        "run_event_watchers",
        "service_clarification",
        "service_producer_disposition",
        "start_model_policy",
        "verify_completion_proposal",
    }
)

# Private entrances other components use to start execution (workers, dispatch, evals).
PRIVATE_EXECUTION_ENTRANCES = frozenset(
    {
        "_claim_work_attempt_recovery",
        "_compact_session_private",
        "_dispatch_queued",
        "_execute_work_attempt",
        "_recover_claimed_work_attempt",
        "_recover_incomplete_session_private",
        "_resume_private",
        "_run_private",
        "_run_with_public_projection",
    }
)

# The shutdown parts aclose() composes; counting them would make it wait on itself.
SHUTDOWN_PARTS = frozenset(
    {
        "aclose",
        "close_runtime_timing",
        "drain_background_interruptions",
        "drain_collaboration_requests",
        "drain_environment_cleanups",
        "drain_knowledge_publications",
        "drain_provider_operation_cancellations",
        "drain_recovery_cleanups",
        "drain_session_exports",
        "drain_session_recovery_cleanups",
        "drain_verified_completions",
        "flush_runtime_timing",
        "stop_model_policy",
    }
)


class _GatedProvider(ModelProvider):
    """Answers once ``release`` is set, so a run can be held in flight."""

    name = "fake"

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.started.set()
        await self.release.wait()
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def _app(provider: ModelProvider | None = None, **kwargs) -> CayuApp:
    app = CayuApp(enable_logging=False, **kwargs)
    if provider is not None:
        app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return app


def _request(session_id: str) -> RunRequest:
    return RunRequest(
        agent_name="assistant",
        session_id=session_id,
        messages=[Message.text("user", "hello")],
    )


async def _consume(app: CayuApp, session_id: str) -> list:
    return [event async for event in app.run(_request(session_id))]


class _Resource:
    def __init__(self, log: list[str], name: str) -> None:
        self.log = log
        self.name = name

    async def close(self) -> None:
        self.log.append(self.name)


def test_every_async_entrance_is_gated_tracked_or_a_shutdown_part() -> None:
    def kind(name: str) -> str | None:
        return getattr(getattr(CayuApp, name), _ENTRANCE_KIND, None)

    public = {
        name
        for name, member in inspect.getmembers(CayuApp)
        if not name.startswith("_")
        and (inspect.iscoroutinefunction(member) or inspect.isasyncgenfunction(member))
    }
    assert {name for name in public if kind(name) == "admitted"} == GATED_ENTRANCES
    assert {name for name in public if kind(name) is None} == SHUTDOWN_PARTS
    # Everything else stays available during shutdown but is waited for.
    assert {name for name in public if kind(name) == "tracked"} == (
        public - GATED_ENTRANCES - SHUTDOWN_PARTS
    )
    assert all(kind(name) == "admitted" for name in PRIVATE_EXECUTION_ENTRANCES)


def test_idle_app_settles_and_the_context_manager_is_aclose() -> None:
    async def scenario() -> None:
        async with _app() as app:
            assert app.lifecycle_state == "open"
        outcome = app.shutdown_outcome
        assert outcome is not None and outcome.settled
        assert app.lifecycle_state == "closed"
        assert [step.subsystem for step in outcome.steps] == [
            "open_operations",
            "model_policy",
            "background_interruptions",
            "recovery_cleanups",
            "provider_operation_cancellations",
            "environment_cleanups",
            "knowledge_publications",
            "collaboration_requests",
            "session_exports",
            "verified_completions",
            "runtime_timing",
        ]
        assert await app.aclose() is outcome
        with pytest.raises(RuntimeError, match="cannot be entered again"):
            async with app:
                pass

    asyncio.run(scenario())


def test_in_flight_run_finishes_and_persists_while_new_runs_are_refused() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        provider = _GatedProvider()
        app = _app(provider, session_store=store)
        running = asyncio.create_task(_consume(app, "in-flight"))
        await provider.started.wait()
        closing = asyncio.create_task(app.aclose(timeout_s=10))
        await asyncio.sleep(0.05)
        assert app.lifecycle_state == "closing" and not closing.done()
        with pytest.raises(ApplicationAdmissionsSealed):
            await _consume(app, "refused")
        provider.release.set()
        await running
        outcome = await closing
        assert outcome.settled and outcome.open_operations == 0
        session = await store.load("in-flight")
        assert session is not None and session.status is SessionStatus.COMPLETED
        assert await store.load("refused") is None
        # The caller's store was not closed: another app can keep using it.
        provider.release.set()
        await _consume(_app(provider, session_store=store), "after-shutdown")

    asyncio.run(scenario())


def test_an_abandoned_stream_is_reported_until_it_is_closed() -> None:
    async def scenario() -> None:
        provider = _GatedProvider()
        app = _app(provider)
        stream = app.run(_request("abandoned"))
        await anext(stream)
        outcome = await app.aclose(timeout_s=0.2)
        assert outcome.status == "incomplete" and outcome.open_operations == 1
        step = outcome.step("open_operations")
        assert step is not None and step.reason == "open_operations"
        await stream.aclose()
        retried = await app.aclose(timeout_s=10)
        assert retried.settled and retried.attempt == 2

    asyncio.run(scenario())


def test_owned_resources_close_once_in_reverse_order_only_after_settling(monkeypatch) -> None:
    async def scenario() -> None:
        log: list[str] = []
        app = _app(owned_resources=(_Resource(log, "stores"), _Resource(log, "provider")))
        results = [False, True]
        drain = app.drain_recovery_cleanups

        async def draining(*, timeout_s: float) -> bool:
            await drain(timeout_s=timeout_s)
            return results.pop(0)

        monkeypatch.setattr(app, "drain_recovery_cleanups", draining)
        first = await app.aclose(timeout_s=5)
        assert first.status == "incomplete" and first.owned_resources == "retained"
        assert log == []
        second = await app.aclose(timeout_s=5)
        assert second.settled and second.owned_resources == "released"
        assert log == ["provider", "stores"]
        await app.aclose(timeout_s=5)
        assert log == ["provider", "stores"]

    asyncio.run(scenario())


def test_a_stalled_drain_is_bounded_by_the_shared_deadline(monkeypatch) -> None:
    async def scenario() -> None:
        app = _app()
        release = asyncio.Event()

        async def stalled(*, timeout_s: float) -> bool:
            await release.wait()
            return True

        monkeypatch.setattr(app, "drain_environment_cleanups", stalled)
        outcome = await app.aclose(timeout_s=0.3)
        assert outcome.status == "incomplete"
        assert outcome.elapsed_seconds < 2.0
        step = outcome.step("environment_cleanups")
        assert step is not None and step.reason == "overran_budget"
        # Floor-protected cleanup still ran after the deadline was used up.
        timing = outcome.step("runtime_timing")
        assert timing is not None and timing.status == "settled"
        # Steps skipped for the exhausted deadline still refused new work.
        assert app._knowledge_publication_scope.sealed
        assert app._request_coordinator.owners.closed
        assert app._session_export_coordinator.owners.closed
        release.set()
        assert (await app.aclose(timeout_s=5)).settled

    asyncio.run(scenario())


def test_a_failing_drain_fails_the_outcome_but_later_steps_still_run(monkeypatch) -> None:
    async def scenario() -> None:
        app = _app()

        async def broken(*, timeout_s: float) -> bool:
            raise RuntimeError("drain broke")

        monkeypatch.setattr(app, "drain_recovery_cleanups", broken)
        outcome = await app.aclose(timeout_s=5)
        assert outcome.status == "failed"
        failed = outcome.step("recovery_cleanups")
        assert failed is not None and failed.failure_type == "RuntimeError"
        assert all(
            step.status == "settled"
            for step in outcome.steps
            if step.subsystem not in {"recovery_cleanups"}
        )

    asyncio.run(scenario())


def test_caller_cancellation_is_prompt_and_the_attempt_stays_owned(monkeypatch) -> None:
    async def scenario() -> None:
        app = _app()
        release = asyncio.Event()

        async def gated(*, timeout_s: float) -> bool:
            await release.wait()
            return True

        monkeypatch.setattr(app, "drain_recovery_cleanups", gated)
        caller = asyncio.create_task(app.aclose(timeout_s=10))
        await asyncio.sleep(0.05)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert caller.cancelled()
        release.set()
        outcome = await app.aclose(timeout_s=10)
        assert outcome.settled and outcome.attempt == 1

    asyncio.run(scenario())


def test_knowledge_publications_from_runs_belong_to_the_running_app() -> None:
    class StalledStore(InMemoryKnowledgeStore):
        def __init__(self) -> None:
            super().__init__(access_scope=KnowledgeAccessScope.privileged())
            self.dispatched = asyncio.Event()
            self.release = asyncio.Event()

        async def publish_entry_revision(self, entry, chunks, **kwargs):
            self.dispatched.set()
            await self.release.wait()
            return await super().publish_entry_revision(entry, chunks, **kwargs)

    class RememberOnce(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            self.calls += 1
            if self.calls == 1:
                yield ModelStreamEvent.tool_call(
                    id="call_remember", name="remember_knowledge", arguments={"text": "Kept."}
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def scenario() -> None:
        store = StalledStore()
        tool = RememberKnowledgeTool()
        app = CayuApp(enable_logging=False)
        app.register_provider(RememberOnce(), default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="knowledge"), knowledge_store=store), default=True
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[tool])
        running = asyncio.create_task(_consume(app, "remember"))
        await asyncio.wait_for(store.dispatched.wait(), 10)
        # The runtime bound the app's scope to the tool call.
        assert app._knowledge_publication_scope.pending
        store.release.set()
        await running
        outcome = await app.aclose(timeout_s=10)
        assert outcome.settled
        assert tool._publication_owner.sealed is False

    asyncio.run(scenario())


def test_a_background_subagent_started_by_a_run_holds_shutdown_open(tmp_path) -> None:
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.tools.subagents import (
        BackgroundSubagentTaskRegistry,
        SubagentExecutionMode,
        SubagentSpec,
        SubagentTool,
    )

    class Provider(ModelProvider):
        name = "fake"

        def __init__(self) -> None:
            self.child_started = asyncio.Event()
            self.release_child = asyncio.Event()

        async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
            transcript = " ".join(str(message.content) for message in request.messages)
            if "background review" in transcript and "parent task" not in transcript:
                self.child_started.set()
                await self.release_child.wait()
                yield ModelStreamEvent.text_delta("child done")
            elif "subagent" in transcript and "call_child" in transcript:
                yield ModelStreamEvent.text_delta("parent done")
            else:
                yield ModelStreamEvent.tool_call(
                    id="call_child",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": "background review"},
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                return
            yield ModelStreamEvent.completed({"finish_reason": "stop"})

    async def scenario() -> None:
        log: list[str] = []
        store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
        provider = Provider()
        app = CayuApp(
            session_store=store, enable_logging=False, owned_resources=(_ClosingStore(store, log),)
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="reviewer", model="fake-model"))
        registry = BackgroundSubagentTaskRegistry()
        tool = SubagentTool(
            app,
            agents={
                "reviewer": SubagentSpec(
                    agent_name="reviewer", mode=SubagentExecutionMode.BACKGROUND
                )
            },
            background_registry=registry,
        )
        app.register_agent(AgentSpec(name="parent", model="fake-model"), tools=[tool])
        await asyncio.wait_for(
            _consume_request(
                app,
                RunRequest(
                    agent_name="parent",
                    session_id="parent",
                    messages=[Message.text("user", "parent task")],
                ),
            ),
            10,
        )
        await asyncio.wait_for(provider.child_started.wait(), 10)
        # The parent run finished; its background child is still running.
        outcome = await app.aclose(timeout_s=0.3)
        assert outcome.status == "incomplete" and outcome.open_operations == 1
        assert outcome.owned_resources == "retained" and log == []
        provider.release_child.set()
        await asyncio.gather(*registry.active_tasks("parent"))
        settled = await app.aclose(timeout_s=10)
        assert settled.settled and log == ["closed"]

    asyncio.run(scenario())


class _ClosingStore:
    def __init__(self, store, log: list[str]) -> None:
        self.store = store
        self.log = log

    async def close(self) -> None:
        await self.store.close()
        self.log.append("closed")


async def _consume_request(app: CayuApp, request: RunRequest) -> list:
    return [event async for event in app.run(request)]


def test_an_allowed_operation_in_flight_holds_owned_resources_open() -> None:
    from cayu.tasks import TaskCreate
    from cayu.tasks.base import InMemoryTaskStore

    class SlowStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False
            self.writing = asyncio.Event()
            self.release = asyncio.Event()
            self.wrote_after_close = False

        async def create_task(self, request):
            self.writing.set()
            await self.release.wait()
            self.wrote_after_close = self.closed
            return await super().create_task(request)

        async def close(self) -> None:
            self.closed = True

    async def scenario() -> None:
        store = SlowStore()
        app = CayuApp(task_store=store, enable_logging=False, owned_resources=(store,))
        write = asyncio.create_task(app.create_task(TaskCreate(type="demo", title="t")))
        await store.writing.wait()
        outcome = await app.aclose(timeout_s=0.2)
        # Still allowed while closing, but waited for, so the store stays open.
        assert outcome.status == "incomplete" and outcome.open_operations == 1
        assert outcome.owned_resources == "retained" and not store.closed
        store.release.set()
        await write
        assert not store.wrote_after_close
        assert (await app.aclose(timeout_s=5)).settled and store.closed

    asyncio.run(scenario())


def test_aclose_stops_model_policy_workers(monkeypatch) -> None:
    async def scenario() -> None:
        app = _app()
        stops: list[str] = []

        async def stop_model_policy() -> None:
            stops.append("stopped")

        monkeypatch.setattr(app, "stop_model_policy", stop_model_policy)
        outcome = await app.aclose(timeout_s=5)
        assert outcome.settled and stops == ["stopped"]

    asyncio.run(scenario())


def test_no_operation_reaches_owned_resources_while_or_after_they_close() -> None:
    from cayu.tasks import TaskCreate
    from cayu.tasks.base import InMemoryTaskStore

    class ClosingStore(InMemoryTaskStore):
        def __init__(self) -> None:
            super().__init__()
            self.closing = asyncio.Event()
            self.release_close = asyncio.Event()
            self.writes: list[str] = []

        async def create_task(self, request):
            self.writes.append(request.title)
            return await super().create_task(request)

        async def close(self) -> None:
            self.closing.set()
            await self.release_close.wait()

    async def scenario() -> None:
        store = ClosingStore()
        app = CayuApp(task_store=store, enable_logging=False, owned_resources=(store,))
        closing = asyncio.create_task(app.aclose(timeout_s=5))
        await asyncio.wait_for(store.closing.wait(), 5)
        # The store is being closed: a call that would use it is refused.
        with pytest.raises(ApplicationAdmissionsSealed, match="closed"):
            await app.create_task(TaskCreate(type="demo", title="during close"))
        store.release_close.set()
        outcome = await closing
        assert outcome.settled and outcome.open_operations == 0
        with pytest.raises(ApplicationAdmissionsSealed, match="closed"):
            await app.create_task(TaskCreate(type="demo", title="after close"))
        assert store.writes == []

    asyncio.run(scenario())


def test_admission_stays_open_until_work_settles_then_closes_even_if_release_fails() -> None:
    from cayu.tasks import TaskCreate
    from cayu.tasks.base import InMemoryTaskStore

    class FlakyStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise RuntimeError("store did not close")

    async def scenario() -> None:
        provider = _GatedProvider()
        store = FlakyStore()
        app = _app(provider, task_store=store, owned_resources=(store,))
        running = asyncio.create_task(_consume(app, "in-flight"))
        await asyncio.wait_for(provider.started.wait(), 10)
        incomplete = await app.aclose(timeout_s=0.2)
        assert incomplete.status == "incomplete"
        # Work is still in flight, so allowed calls keep working.
        await app.create_task(TaskCreate(type="demo", title="still allowed"))
        provider.release.set()
        await asyncio.wait_for(running, 10)
        failed = await app.aclose(timeout_s=5)
        assert failed.status == "failed" and failed.owned_resources == "failed"
        # Work settled, so admission closed before the release was attempted.
        with pytest.raises(ApplicationAdmissionsSealed, match="closed"):
            await app.create_task(TaskCreate(type="demo", title="refused"))
        settled = await app.aclose(timeout_s=5)
        assert settled.settled and settled.owned_resources == "released"
        assert store.close_calls == 2

    asyncio.run(scenario())


def test_model_policy_stops_even_when_in_flight_work_uses_up_the_deadline(monkeypatch) -> None:
    async def scenario() -> None:
        app = _app()
        stops: list[str] = []
        release = asyncio.Event()

        async def stop_model_policy() -> None:
            stops.append("stop")

        monkeypatch.setattr(app, "stop_model_policy", stop_model_policy)
        holder = asyncio.create_task(app._run_worker_step(release.wait))
        await asyncio.sleep(0)
        outcome = await app.aclose(timeout_s=0.2)
        assert outcome.status == "incomplete" and outcome.open_operations == 1
        step = outcome.step("model_policy")
        assert step is not None and step.status == "settled"
        assert stops == ["stop"]
        release.set()
        await holder

    asyncio.run(scenario())
