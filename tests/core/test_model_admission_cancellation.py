from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from cayu.deadlines import ExecutionDeadline, ExecutionDeadlineExceeded, bind_execution_deadline
from cayu.failure_evidence import exception_evidence
from cayu.providers._credential_boundary import provider_cancellation_failures
from cayu.providers.deadlines import ProviderStreamDeadlineAdmission, ProviderStreamDeadlines
from cayu.runtime._model_step_executor import (
    _admitted_model_provider_events,
    _owned_model_provider_events,
)


@pytest.mark.parametrize(
    "mode",
    [
        "expired",
        "race",
        "next_turn",
        "refresh_cancel",
        "provider_failure",
        "provider_failure_next_turn",
        "read_failure",
        "cooperative",
        "success",
    ],
)
def test_native_admission_and_caller_cancellation(mode):
    async def scenario():
        calls = 0
        closed = False
        task = asyncio.current_task()
        baseline = task.cancelling()
        deadline = ExecutionDeadline.after(0 if mode in {"expired", "race", "next_turn"} else None)

        class Provider:
            def runtime_stream(self, request):
                nonlocal calls
                calls += 1
                if mode in {"read_failure", "cooperative", "success"}:

                    async def events():
                        nonlocal closed
                        try:
                            if mode == "success":
                                return
                            task.cancel()
                            if mode == "read_failure":
                                raise RuntimeError("private provider failure")
                            await asyncio.sleep(0)
                        finally:
                            closed = True
                        if False:
                            yield

                    return events()
                if mode == "provider_failure_next_turn":
                    asyncio.get_running_loop().call_soon(task.cancel)
                else:
                    task.cancel()
                # Public-looking deadline attributes must not authenticate admission.
                error = ExecutionDeadlineExceeded(ExecutionDeadline.after(0), "model")
                error._cayu_native_model_admission = {"deadline": deadline, "token": object()}
                error.native_admission_deadline = deadline
                raise error

        async def refresh():
            if mode == "next_turn":
                asyncio.get_running_loop().call_soon(task.cancel)
            if mode in {"race", "refresh_cancel"}:
                task.cancel()
            if mode == "refresh_cancel":
                await asyncio.sleep(0)

        admission = ProviderStreamDeadlineAdmission(ProviderStreamDeadlines())
        try:
            from contextlib import nullcontext

            with (
                bind_execution_deadline(deadline),
                nullcontext()
                if mode == "success"
                else pytest.raises(
                    ExecutionDeadlineExceeded if mode == "expired" else asyncio.CancelledError
                ) as caught,
            ):
                async for _ in _owned_model_provider_events(
                    lambda: _admitted_model_provider_events(Provider(), None, admission, refresh),
                    cancellation_baseline=baseline,
                    max_concurrent_streams=100,
                ):
                    pass
            dispatched = mode in {
                "provider_failure",
                "provider_failure_next_turn",
                "read_failure",
                "cooperative",
                "success",
            }
            assert calls == int(dispatched)
            assert closed == (mode in {"read_failure", "cooperative", "success"})
            if mode not in {"expired", "success"}:
                diagnostics = provider_cancellation_failures(caught.value)
                assert bool(diagnostics) == (
                    mode in {"provider_failure", "provider_failure_next_turn", "read_failure"}
                )
                assert exception_evidence(caught.value).secondary_failures == bool(diagnostics)
            if mode in {"expired", "race", "next_turn"}:
                evidence = exception_evidence(caught.value)
                assert evidence.classification == "deadline"
                assert evidence.deadline_phase == "admission"
                assert not evidence.secondary_failures
        finally:
            while task.cancelling() > baseline:
                task.uncancel()
            # Consume any not-yet-delivered cancellation before runner teardown.
            with suppress(asyncio.CancelledError):
                await asyncio.sleep(0)
            admission.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("next_turn", [False, True])
def test_native_admission_race_retains_durable_child_evidence(tmp_path, monkeypatch, next_turn):
    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.evals.testing import ScriptedModelProvider
    from cayu.runtime import _model_step_executor as executor
    from cayu.storage.sqlite import SQLiteSessionStore
    from cayu.workflows.base import WorkflowSpec
    from cayu.workflows.workflow import WorkflowBase, step

    original = executor._admitted_model_provider_events

    async def racing_events(
        provider, request, admission, refresh_live_model_semantics, cleanup_observer=None
    ):
        async def refresh():
            await refresh_live_model_semantics()
            if next_turn:
                asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
            else:
                asyncio.current_task().cancel()

        with bind_execution_deadline(ExecutionDeadline.after(0, scope="native-child")):
            async for event in original(provider, request, admission, refresh, cleanup_observer):
                yield event

    monkeypatch.setattr(executor, "_admitted_model_provider_events", racing_events)

    class Workflow(WorkflowBase):
        spec = WorkflowSpec(name="admission-race")

        async def run(self, session_id):
            yield await self.context(session_id).start()

    async def scenario():
        store = SQLiteSessionStore(tmp_path / "race.sqlite")
        provider = ScriptedModelProvider([])
        app = CayuApp(enable_logging=False, session_store=store)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="worker", model="synthetic"))
        ctx = Workflow(app).context("parent")
        await ctx.start()
        try:
            child = asyncio.create_task(
                step(ctx, agent="worker", step_id="child", session_id="child", prompt="test")
            )
            with pytest.raises(asyncio.CancelledError):
                await child
            assert not provider.requests
            events = await store.load_events("child")
            assert not any(event.type.value == "tool.started" for event in events)
            terminals = [event for event in events if event.type.value == "session.interrupted"]
            assert len(terminals) == 1, [event.type for event in events]
            payload = terminals[0].payload
            assert not payload.get("provider_cancellation_failures")
            evidence = payload["failure_evidence"]
            assert evidence["classification"] == "deadline"
            assert evidence["deadline_phase"] == "admission"
            assert evidence["session_id"] == "child"
            assert evidence["run_epoch"] is not None
            assert evidence["settlement"] == "unknown"
            assert not evidence["secondary_failures"]
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("cleanup", ["failed", "unsettled"])
def test_next_turn_admission_cancellation_preserves_cleanup_failure(cleanup):
    from cayu.providers._credential_boundary import provider_cancellation_admission_deadline

    async def scenario():
        task = asyncio.current_task()
        baseline = task.cancelling()
        released = asyncio.Event()
        closed = asyncio.Event()
        deadline = ExecutionDeadline.after(0)
        calls = 0

        class Provider:
            def runtime_stream(self, request):
                nonlocal calls
                calls += 1
                raise AssertionError("Provider must not be dispatched")

        async def refresh():
            asyncio.get_running_loop().call_soon(task.cancel)

        admission = ProviderStreamDeadlineAdmission(ProviderStreamDeadlines())
        source = _admitted_model_provider_events(Provider(), None, admission, refresh)

        class Source:
            def __aiter__(self):
                return self

            async def __anext__(self):
                return await anext(source)

            async def aclose(self):
                try:
                    await source.aclose()
                    if cleanup == "failed":
                        raise RuntimeError("private cleanup failure")
                    await released.wait()
                finally:
                    closed.set()

        try:
            with bind_execution_deadline(deadline), pytest.raises(asyncio.CancelledError) as caught:
                async for _ in _owned_model_provider_events(
                    Source, cancellation_baseline=baseline, max_concurrent_streams=100
                ):
                    pass
            assert calls == 0
            assert provider_cancellation_admission_deadline(caught.value) == deadline
            diagnostics = provider_cancellation_failures(caught.value)
            assert [item["phase"] for item in diagnostics] == ["provider_stream_cleanup"]
            evidence = exception_evidence(caught.value)
            assert evidence.deadline_phase == "admission"
            assert evidence.secondary_failures
            assert evidence.settlement == "unknown"
        finally:
            while task.cancelling() > baseline:
                task.uncancel()
            released.set()
            await closed.wait()
            admission.close()

    asyncio.run(scenario())
