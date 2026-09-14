from __future__ import annotations

import asyncio

import pytest

from cayu import (
    AgentSpec,
    AuxiliaryInferencePolicy,
    CayuApp,
    CayuConfig,
    EventType,
    IncompleteSessionRecoveryRequest,
    InferenceLimits,
    Message,
    ModelRequest,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolExecutionConfig,
    ToolSpec,
)
from cayu.providers import ModelProviderError


def test_stream_close_during_scope_drain_preserves_generator_exit(monkeypatch):
    from cayu import _task_wait
    from cayu._exception_groups import exception_cause
    from cayu.runtime._auxiliary_invocation import AuxiliaryInferenceScope

    async def run():
        entered = asyncio.Event()
        cleanup = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        failure = RuntimeError("callback cleanup failed")
        calls = []

        async def callback(request, purpose, limits):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup.set()
                await release.wait()
                raise failure from None

        scope = AuxiliaryInferenceScope(callback)

        async def stream():
            async with scope.lifetime():
                calls.append(
                    asyncio.create_task(
                        scope.invoke(
                            ModelRequest(model="model", messages=[]),
                            purpose="tool.summary",
                            limits=InferenceLimits(
                                max_input_tokens=1, max_output_tokens=1, timeout_seconds=10
                            ),
                        )
                    )
                )
                await entered.wait()
                yield

        iterator = stream()
        await anext(iterator)
        closer = asyncio.create_task(iterator.aclose())
        original = _task_wait.consume_pending_task_cancellation

        def observe(*args, **kwargs):
            if asyncio.current_task() is closer and closer.cancelling():
                cancelled.set()
            return original(*args, **kwargs)

        monkeypatch.setattr(_task_wait, "consume_pending_task_cancellation", observe)
        try:
            await asyncio.wait_for(cleanup.wait(), 5)
            closer.cancel()
            await asyncio.wait_for(cancelled.wait(), 5)
            release.set()
            with pytest.raises(BaseExceptionGroup) as caught:
                await closer
            assert len(caught.value.exceptions) == 2
            assert isinstance(caught.value.exceptions[0], GeneratorExit)
            cancellation = caught.value.exceptions[1]
            assert isinstance(cancellation, asyncio.CancelledError)
            assert exception_cause(cancellation) is failure
            assert closer.cancelling() == 1 and not closer.cancelled()
        finally:
            release.set()
            if not closer.done():
                closer.cancel()
            await asyncio.gather(closer, *calls, return_exceptions=True)

    asyncio.run(run())


def test_public_cancel_provider_terminal_completion_settles_once():
    async def run():
        entered = asyncio.Event()
        limits = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)

        class Stream:
            state = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.state == 2:
                    raise StopAsyncIteration
                if self.state == 0:
                    self.state = 1
                    return ModelStreamEvent.text_delta("summary")
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.state = 2
                    return ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 3, "output_tokens": 2}}
                    )
                raise AssertionError

            async def aclose(self):
                return None

        stream = Stream()

        class Provider(ScriptedModelProvider):
            def runtime_stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self.requests.append(request.model_copy(deep=True))
                    return stream
                return super().runtime_stream(request)

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="nested",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=limits, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                await ctx.inference.invoke(
                    ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=limits,
                )

        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="p"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ]
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])

        async def consume():
            async for _ in app.run(
                RunRequest(
                    session_id="public-terminal-cancel",
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            ):
                pass

        task = asyncio.create_task(consume())
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
        events = await app.session_store.load_events("public-terminal-cancel")
        settled = [e for e in events if e.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED]
        assert len(settled) == 1 and settled[0].payload["auxiliary_outcome"] == "completed"
        assert settled[0].payload["usage_metrics"]["total_tokens"] == 5
        assert len(provider.requests) == 2
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="public-terminal-cancel", inactive_for_seconds=None
            )
        )
        assert await app.session_store.load_events("public-terminal-cancel") == events

    asyncio.run(run())


@pytest.mark.parametrize(
    "tool_fails,initial_cancel",
    [(False, False), (True, False), (False, True)],
    ids=["return", "tool-error", "earlier-cancellation"],
)
def test_parent_cancel_during_abandoned_inference_accounting_failure(
    monkeypatch, tool_fails, initial_cancel
):
    from contextlib import asynccontextmanager

    from cayu import ToolResult, _task_wait
    from cayu._exception_groups import exception_cause, iter_exception_tree
    from cayu.runtime import _auxiliary_invocation

    async def run():
        dispatched = asyncio.Event()
        accounting = asyncio.Event()
        release = asyncio.Event()
        cancellation_delivered = asyncio.Event()
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=20)
        calls = []
        drain_owners = []
        drained = []
        scope_failures = []
        failure = RuntimeError("auxiliary accounting failed during scope drain")
        tool_failure = ValueError("tool failed while inference was active")

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self.requests.append(request.model_copy(deep=True))
                    dispatched.set()
                    await asyncio.Event().wait()
                    return
                async for event in super().stream(request):
                    yield event

        class Abandon(Tool):
            spec = ToolSpec(
                name="abandon",
                description="Return while managed inference is active",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                calls.append(
                    asyncio.create_task(
                        ctx.inference.invoke(
                            ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                            purpose="tool.summary",
                            limits=bounds,
                        )
                    )
                )
                await dispatched.wait()
                if initial_cancel:
                    await asyncio.Event().wait()
                if tool_fails:
                    raise tool_failure
                return ToolResult(content="returned too soon")

        app = CayuApp(enable_logging=False)
        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="abandon", id="parent", arguments={}),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ]
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Abandon()])
        original_complete = app.session_store.complete_model_completion_stage

        async def fail_terminal(session_id, *, stage_id, publication):
            if publication.kind == "auxiliary-inference":
                accounting.set()
                await release.wait()
                raise failure
            return await original_complete(session_id, stage_id=stage_id, publication=publication)

        original_wait = _auxiliary_invocation.await_shielded_task_outcome

        async def observe_drain(task, **kwargs):
            drain_owners.append(asyncio.current_task())
            outcome = await original_wait(task, **kwargs)
            drained.append(outcome)
            return outcome

        original_consume = _task_wait.consume_pending_task_cancellation

        def observe_cancellation(*args, **kwargs):
            owner = asyncio.current_task()
            if owner in drain_owners and owner.cancelling():
                cancellation_delivered.set()
            return original_consume(*args, **kwargs)

        monkeypatch.setattr(app.session_store, "complete_model_completion_stage", fail_terminal)
        monkeypatch.setattr(_auxiliary_invocation, "await_shielded_task_outcome", observe_drain)
        monkeypatch.setattr(_task_wait, "consume_pending_task_cancellation", observe_cancellation)
        original_lifetime = _auxiliary_invocation.AuxiliaryInferenceScope.lifetime

        @asynccontextmanager
        async def observe_lifetime(scope):
            try:
                async with original_lifetime(scope):
                    yield
            except BaseException as exc:
                retained = list(iter_exception_tree(exc))
                for error in retained:
                    cause = exception_cause(error)
                    if cause is not None and all(cause is not item for item in retained):
                        retained.extend(iter_exception_tree(cause))
                # Capture before the outer tool privacy boundary deliberately
                # severs unauthenticated exception causes on this same object.
                scope_failures.append((isinstance(exc, asyncio.CancelledError), retained))
                raise

        monkeypatch.setattr(
            _auxiliary_invocation.AuxiliaryInferenceScope, "lifetime", observe_lifetime
        )

        async def consume():
            async for _ in app.run(
                RunRequest(
                    session_id="drain-cancel",
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            ):
                pass

        parent = asyncio.create_task(consume())
        try:
            if initial_cancel:
                await asyncio.wait_for(dispatched.wait(), 10)
                parent.cancel()
            await asyncio.wait_for(accounting.wait(), 10)
            parent.cancel()
            await asyncio.wait_for(cancellation_delivered.wait(), 5)
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await parent
            assert parent.cancelled() and parent.cancelling() == 1 + int(initial_cancel)
            assert len(drained) == 1
            assert drained[0].result.error is not None and drained[0].cancellation is not None
            assert len(scope_failures) == 1 and scope_failures[0][0]
            retained = scope_failures[0][1]
            assert sum(error is failure for error in retained) == 1
            assert sum(error is tool_failure for error in retained) == int(tool_fails)
            if tool_fails:
                assert retained.index(tool_failure) < retained.index(failure)
            assert len(provider.requests) == 2
            active = await app.session_store.load_active_model_completion_stage("drain-cancel")
            assert active is not None and active.stage.state == "in_flight"
            monkeypatch.setattr(
                app.session_store, "complete_model_completion_stage", original_complete
            )
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="drain-cancel",
                    inactive_for_seconds=None,
                )
            )
            assert len(provider.requests) == 2
            settlements = [
                event
                for event in await app.session_store.load_events("drain-cancel")
                if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(settlements) == 1
            assert settlements[0].payload["auxiliary_outcome"] == "outcome_unknown"
        finally:
            release.set()
            if not parent.done():
                parent.cancel()
            await asyncio.gather(parent, *calls, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("committed", [False, True], ids=["before-commit", "after-commit"])
def test_parent_cancel_during_auxiliary_terminal_accounting(monkeypatch, committed):
    from examples.runtime_auxiliary_inference import MODEL, Summarize

    async def run():
        entered = asyncio.Event()
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        name="summarize", id="parent", arguments={"text": "Long input"}
                    ),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}}),
                ],
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model=MODEL), tools=[Summarize()])
        complete = app.session_store.complete_model_completion_stage

        async def pause_terminal(session_id, *, stage_id, publication):
            if publication.kind != "auxiliary-inference":
                return await complete(session_id, stage_id=stage_id, publication=publication)
            if committed:
                await complete(session_id, stage_id=stage_id, publication=publication)
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("Cancellation must interrupt the terminal wait")

        monkeypatch.setattr(app.session_store, "complete_model_completion_stage", pause_terminal)

        async def consume():
            async for _ in app.run(
                RunRequest(
                    session_id="terminal-cancel",
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            ):
                pass

        parent = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(entered.wait(), 10)
            parent.cancel()
            done, _ = await asyncio.wait((parent,), timeout=5)
            assert parent in done
            with pytest.raises(asyncio.CancelledError):
                await parent
            assert parent.cancelled() and parent.cancelling() == 1
            assert len(provider.requests) == 2
            active = await app.session_store.load_active_model_completion_stage("terminal-cancel")
            assert active is not None
            assert active.stage.state == ("completed" if committed else "in_flight")
            monkeypatch.setattr(app.session_store, "complete_model_completion_stage", complete)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="terminal-cancel", inactive_for_seconds=None
                )
            )
            assert len(provider.requests) == 2
            events = await app.session_store.load_events("terminal-cancel")
            settlements = [
                event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(settlements) == 1
            assert settlements[0].payload["auxiliary_outcome"] == (
                "completed" if committed else "outcome_unknown"
            )
            usage = await app.get_session_usage("terminal-cancel")
            assert usage.model_steps == 1
            assert usage.usage.total_tokens == (7 if committed else 2)
            assert usage.unmeasured_model_attempts == int(not committed)
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="terminal-cancel", inactive_for_seconds=None
                )
            )
            assert len(provider.requests) == 2
            assert await app.session_store.load_events("terminal-cancel") == events
        finally:
            if not parent.done():
                parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "cancel_count", [0, 1, 2], ids=["tool-timeout", "parent-once", "parent-twice"]
)
def test_parent_cancel_retains_resistant_auxiliary_read_and_observed_usage(cancel_count):
    async def run():
        started = asyncio.Event()
        cancellation_seen = asyncio.Event()
        release = asyncio.Event()
        settled = asyncio.Event()
        closed = asyncio.Event()
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)
        handles = []

        class Events:
            emitted_error = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.emitted_error:
                    self.emitted_error = True
                    error = ModelProviderError("overloaded", provider="scripted", retryable=True)
                    event = ModelStreamEvent.error("overloaded", cause=error)
                    event.payload["usage"] = {"input_tokens": 3, "output_tokens": 2}
                    return event
                started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancellation_seen.set()
                    try:
                        await release.wait()
                        return ModelStreamEvent.text_delta("late value must not become a result")
                    finally:
                        settled.set()
                raise AssertionError("Provider read resumed unexpectedly")

            async def aclose(self):
                closed.set()

        stream = Events()

        class Provider(ScriptedModelProvider):
            def stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self.requests.append(request.model_copy(deep=True))
                    return stream
                return super().stream(request)

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Cancellation-resistant auxiliary transport",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                handles.append(ctx.inference)
                await ctx.inference.invoke(
                    ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=bounds,
                )
                raise AssertionError("Cancelled inference cannot return a tool result")

        app = CayuApp(
            enable_logging=False,
            config=CayuConfig(
                tool_execution=ToolExecutionConfig(
                    tool_timeout_seconds=1 if cancel_count == 0 else None
                )
            ),
        )
        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ]
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])

        async def consume():
            events = []
            async for event in app.run(
                RunRequest(
                    session_id="resistant-auxiliary",
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            ):
                events.append(event)
            return events

        parent = asyncio.create_task(consume())
        try:
            await asyncio.wait_for(started.wait(), 10)
            for _ in range(cancel_count):
                parent.cancel()
            done, _ = await asyncio.wait((parent,), timeout=5)
            assert parent in done, "Parent cancellation waited for the resistant provider"
            if cancel_count:
                try:
                    await parent
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("Parent cancellation was lost")
                assert parent.cancelled() and parent.cancelling() == cancel_count
            else:
                events = await parent
                assert not parent.cancelled() and parent.cancelling() == 0
                failures = [
                    event for event in events if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
                ]
                assert len(failures) == 1, [event.type for event in events]
                # An external tool's timeout is uncertainty, not a fabricated
                # terminal ToolResult; its durable cause still identifies timeout.
                assert failures[0].payload["failure_evidence"]["classification"] == "timeout"
            assert cancellation_seen.is_set() and not settled.is_set() and not closed.is_set()
            assert len(provider.requests) == 2
            active = await app.session_store.load_active_model_completion_stage(
                "resistant-auxiliary"
            )
            assert active is not None and active.stage.state == "completed"
            terminal = active.stage.publication.events[0]
            assert terminal.payload["auxiliary_outcome"] == "cancelled"
            assert terminal.payload["usage_metrics"]["total_tokens"] == 5
            try:
                await handles[0].invoke(
                    ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=bounds,
                )
            except RuntimeError as exc:
                assert "expired" in str(exc)
            else:
                raise AssertionError("Expired handle authorized a competing call")
            assert len(provider.requests) == 2
            release.set()
            await asyncio.wait_for(settled.wait(), 5)
            await asyncio.wait_for(closed.wait(), 5)
            retained = await app.session_store.load_active_model_completion_stage(
                "resistant-auxiliary"
            )
            assert retained == active
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id="resistant-auxiliary", inactive_for_seconds=None
                )
            )
            assert len(provider.requests) == 2
            assert (await app.get_session_usage("resistant-auxiliary")).usage.total_tokens == 7
            events = await app.session_store.load_events("resistant-auxiliary")
            auxiliary = [
                event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(auxiliary) == 1
            assert auxiliary[0].payload["auxiliary_outcome"] == "cancelled"
        finally:
            release.set()
            if not parent.done():
                parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
            if cancellation_seen.is_set():
                await asyncio.wait_for(closed.wait(), 5)

    asyncio.run(run())


@pytest.mark.parametrize(
    "close_at",
    [EventType.MODEL_AUXILIARY_ATTEMPT_STARTED, EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED],
)
def test_public_stream_abandonment_preserves_auxiliary_settlement(close_at):
    from examples.runtime_auxiliary_inference import MODEL, Summarize

    async def run():
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        name="summarize", id="parent", arguments={"text": "Long input"}
                    ),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}}),
                ],
                [ModelStreamEvent.text_delta("must not dispatch"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model=MODEL), tools=[Summarize()])
        stream = app.run(
            RunRequest(
                session_id="abandoned-auxiliary",
                agent_name="assistant",
                messages=[Message.text("user", "go")],
            )
        )
        found = False
        try:
            async for event in stream:
                if event.type is close_at:
                    found = True
                    break
        finally:
            await stream.aclose()
        assert found
        assert len(provider.requests) == 2
        events = await app.session_store.load_events("abandoned-auxiliary")
        settled = [
            event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert len(settled) == 1
        assert settled[0].payload["auxiliary_outcome"] == "completed"
        usage = await app.get_session_usage("abandoned-auxiliary")
        assert usage.model_steps == 1 and usage.usage.total_tokens == 7
        assert (
            await app.session_store.load_active_model_completion_stage("abandoned-auxiliary")
            is None
        )

    asyncio.run(run())
