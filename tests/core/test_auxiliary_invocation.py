from __future__ import annotations

import asyncio
import warnings
from contextlib import aclosing
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from cayu import AgentSpec, CayuApp, Event, EventType, Message, RunRequest, ScriptedModelProvider
from cayu._exception_groups import exception_cause, iter_exception_tree
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.budgets.usage import SessionUsageSummary, session_usage_summary
from cayu.providers import (
    ModelProviderError,
    ModelRequest,
    ModelStreamDeadlineError,
    ModelStreamEvent,
)
from cayu.providers.base import TargetedToolProjectionRequest, ToolDiscoveryProjectionRequest
from cayu.providers.deadlines import ProviderStreamDeadlines
from cayu.providers.response import ModelResponse
from cayu.runtime import _tool_execution
from cayu.runtime._auxiliary_invocation import AuxiliaryInferenceScope, AuxiliaryInvocationPolicy
from cayu.runtime._run_limit_accounting import RunLimitAccountingContext
from cayu.runtime._tool_round_executor import ToolRoundExecutor
from cayu.runtime.evidence import RuntimeEvidenceOperation, RuntimeEvidenceRequest, runtime_evidence
from cayu.runtime.execution_profiles import (
    ExecutionProfileComponentClass,
    ExecutionProfileMismatchError,
)
from cayu.runtime.execution_units import ModelAttemptIdentity
from cayu.runtime.retry_policy import RetryPolicy
from cayu.runtime.stop_policy import RunLimits, auxiliary_token_admission
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.inference import AuxiliaryInferencePolicy, InferenceLimits
from cayu.vaults.redaction import SecretRedactor


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_public_auxiliary_preparation_binds_every_intent_field(
    sqlite_resources, backend, monkeypatch
):
    from cayu.sessions.base import SessionModelCompletionStageConflict

    limits = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=30)
    checked = []

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Check exact prepared authority",
            input_schema={"type": "object"},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=limits, purposes=("tool.summary",)),
        )

        async def run(self, ctx, args):
            response = await ctx.inference.invoke(
                ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                purpose="tool.summary",
                limits=limits,
            )
            return ToolResult(content=response.text)

    def leaves(value, path=()):
        if type(value) is dict and value:
            for key, item in value.items():
                yield from leaves(item, (*path, key))
        elif type(value) is list and value:
            for index, item in enumerate(value):
                yield from leaves(item, (*path, index))
        else:
            yield path, value

    async def scenario():
        store = (
            sqlite_resources.own(SQLiteSessionStore(sqlite_resources.path("authority.sqlite")))
            if backend == "sqlite"
            else None
        )
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            provider = ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(name="summarize", id="parent", arguments={}),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 1, "output_tokens": 1}}
                        ),
                    ],
                    [
                        ModelStreamEvent.text_delta("summary"),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 2, "output_tokens": 1}}
                        ),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ]
            )
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
            original = app._runtime_session_store.prepare_model_completion_stage

            async def check_preparation(session_id, *, request, **kwargs):
                prepared = await original(session_id, request=request, **kwargs)
                if request.purpose != "auxiliary-inference":
                    return prepared
                assert set(request.intent) == {
                    "model_step_id",
                    "model_attempt_id",
                    "auxiliary_inference",
                    "session_instance_id",
                    "invocation",
                    "interaction_id",
                    "source_run_epoch",
                    "execution_profile_fingerprint",
                    "causal_budget_id",
                    "agent_name",
                    "environment_name",
                    "provider_name",
                    "pricing_provider_name",
                    "billing_identity",
                    "budget_reservations",
                    "allocation_fingerprint",
                    "requested_model",
                    "tool_name",
                    "idempotency_key",
                    "request_fingerprint",
                    "limits",
                    "run_limits",
                    "retry_policy",
                    "run_limit_accounting",
                }
                before = await app.session_store.load_events(session_id)
                for path, value in leaves(request.intent):
                    changed = request.model_copy(deep=True)
                    target = changed.intent
                    for key in path[:-1]:
                        target = target[key]
                    replacement = (
                        not value
                        if type(value) is bool
                        else value + 1
                        if type(value) in (int, float)
                        else value + "-conflict"
                        if type(value) is str
                        else "conflict"
                    )
                    target[path[-1]] = replacement
                    with pytest.raises(SessionModelCompletionStageConflict):
                        await original(session_id, request=changed, **kwargs)
                    checked.append(path)
                    assert await app.session_store.load_events(session_id) == before
                    active = await app.session_store.load_active_model_completion_stage(session_id)
                    assert active.stage == prepared.stage
                for field, replacement in (
                    ("logical_step_id", "conflicting-logical-step"),
                    ("dispatch_ordinal", request.dispatch_ordinal + 1),
                    ("reservation_ids", ("conflicting-reservation",)),
                ):
                    changed = request.model_copy(update={field: replacement}, deep=True)
                    with pytest.raises(SessionModelCompletionStageConflict):
                        await original(session_id, request=changed, **kwargs)
                    assert await app.session_store.load_events(session_id) == before
                    active = await app.session_store.load_active_model_completion_stage(session_id)
                    assert active.stage == prepared.stage
                replay = await original(session_id, request=request, **kwargs)
                assert not replay.dispatch_authorized
                return prepared

            monkeypatch.setattr(
                app._runtime_session_store, "prepare_model_completion_stage", check_preparation
            )
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="authority",
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 3
            assert len(checked) >= 30
            assert ("request_fingerprint",) in checked
            assert ("auxiliary_inference", "parent", "tool_round_id") in checked
        finally:
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "case",
    [
        "tool",
        "terminal_tools",
        "oversize",
        "cleanup",
        "error_cleanup",
        "self_cancel",
        "error_self_cancel",
    ],
)
def test_public_auxiliary_output_failure_retains_fence_and_usage(sqlite_resources, backend, case):
    from cayu.runtime import IncompleteSessionRecoveryRequest
    from cayu.sessions.base import SessionModelCompletionStageConflict

    async def scenario():
        bounds = InferenceLimits(
            max_input_tokens=10, max_output_tokens=10, timeout_seconds=10, max_response_bytes=1024
        )
        failures = []
        tool_entries = []
        closed = []
        if case == "tool":
            event = ModelStreamEvent.tool_call(name="summarize", id="recursive", arguments={})
        elif case in {"error_cleanup", "error_self_cancel"}:
            error = ModelProviderError(
                "overloaded", provider="scripted", retryable=True, status_code=503
            )
            event = ModelStreamEvent.error("overloaded", cause=error)
            event.payload["usage"] = {"input_tokens": 3, "output_tokens": 2}
        else:
            event = ModelStreamEvent.completed(
                {
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                    "finish_reason": "tool_calls" if case == "terminal_tools" else "stop",
                    **({"oversize": "x" * 2048} if case == "oversize" else {}),
                }
            )

        class Events:
            yielded = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if case == "self_cancel" or (case == "error_self_cancel" and self.yielded):
                    task = asyncio.current_task()
                    task.cancel()
                    await asyncio.sleep(0)
                if self.yielded:
                    raise StopAsyncIteration
                self.yielded = True
                return event

            async def aclose(self):
                closed.append(True)
                if case in {"cleanup", "error_cleanup"}:
                    raise RuntimeError("provider cleanup failed")

        class Provider(ScriptedModelProvider):
            def stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self.requests.append(request.model_copy(deep=True))
                    return Events()
                return super().stream(request)

        def nested_requests(provider):
            return [
                request
                for request in provider.requests
                if request.messages == [Message.text("user", "nested")]
            ]

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Rejected nested output",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                tool_entries.append(ctx.idempotency_key)
                try:
                    await ctx.inference.invoke(
                        ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                        purpose="tool.summary",
                        limits=bounds,
                    )
                except Exception as failure:
                    assert asyncio.current_task().cancelling() == 0
                    if case == "self_cancel":
                        assert isinstance(failure, ModelProviderError)
                        assert failure.error_code == "provider_stream_cancelled_itself"
                        assert failure.retryable is False
                    elif case == "error_self_cancel":
                        assert isinstance(failure, ExceptionGroup)
                        assert len(failure.exceptions) == 2
                        assert failure.exceptions[0].status_code == 503
                        assert (
                            failure.exceptions[1].error_code == "provider_stream_cancelled_itself"
                        )
                        assert failure.exceptions[1].retryable is False
                    failures.append(failure)
                    return ToolResult(content="rejected", is_error=True)
                raise AssertionError("Invalid output or failed cleanup cannot return a response")

        path = sqlite_resources.path("output-failure.sqlite")
        store = sqlite_resources.own(SQLiteSessionStore(path)) if backend == "sqlite" else None
        try:
            app = CayuApp(session_store=store, enable_logging=False)

            def parent_response(request):
                if any(
                    part.type == "tool_result"
                    for message in request.messages
                    for part in message.content
                ):
                    return [
                        ModelStreamEvent.text_delta("Handled the tool error"),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 1, "output_tokens": 1}}
                        ),
                    ]
                return [
                    ModelStreamEvent.tool_call(name="summarize", id="parent", arguments={}),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ]

            provider = Provider(response_factory=parent_response)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
            try:
                async for _ in app.run(
                    RunRequest(
                        session_id="output-failure",
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                        retry_policy=RetryPolicy(max_attempts=3, initial_delay_s=0.0, jitter_s=0.0),
                    )
                ):
                    pass
            except SessionModelCompletionStageConflict:
                # Continuing after the tool catches the error must still be
                # refused by the retained provider stage, not grant a retry.
                pass
            assert len(failures) == 1 and len(tool_entries) == 1
            assert closed and len(nested_requests(provider)) == 1
            active = await app.session_store.load_active_model_completion_stage("output-failure")
            expected_outcome = "completed" if case == "cleanup" else "failed"
            if case == "cleanup":
                # Accepted completion wins over its cleanup failure. The tool
                # still receives the failure, and the parent may continue.
                assert active is None
                assert len(provider.requests) == 3
                settled = [
                    item
                    for item in await app.session_store.load_events("output-failure")
                    if item.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
                ]
                assert len(settled) == 1
                terminal = settled[0]
            else:
                assert active is not None and active.stage.state == "completed"
                terminal = active.stage.publication.events[0]
                assert len(provider.requests) == 2
            assert terminal.payload["auxiliary_outcome"] == expected_outcome
            assert terminal.payload["usage_status"] == (
                "missing" if case in {"tool", "self_cancel"} else "observed"
            )
            if case not in {"tool", "self_cancel"}:
                assert terminal.payload["usage_metrics"]["total_tokens"] == 5
            assert "retry_decision" not in terminal.payload
            assert terminal.payload["auxiliary_inference"]["tool_call_id"] == "parent"
            if active is not None:
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="output-failure", inactive_for_seconds=None
                    )
                )
                recovered = await app.session_store.load_events("output-failure")
                await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id="output-failure", inactive_for_seconds=None
                    )
                )
                assert await app.session_store.load_events("output-failure") == recovered
            assert len(nested_requests(provider)) == 1
            if store is not None:
                await store.close()
                store = sqlite_resources.own(SQLiteSessionStore(path))
                app = CayuApp(session_store=store, enable_logging=False)
            events = await app.session_store.load_events("output-failure")
            auxiliary = [
                item for item in events if item.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(auxiliary) == 1
            assert auxiliary[0].id == terminal.id
            assert auxiliary[0].payload["auxiliary_outcome"] == expected_outcome
            assert auxiliary[0].payload["auxiliary_inference"]["tool_call_id"] == "parent"
            assert (await app.get_session_usage("output-failure")).usage.total_tokens == (
                2 if case in {"tool", "self_cancel"} else 9 if case == "cleanup" else 7
            )
        finally:
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "model",
        "options",
        "tools",
        "purpose",
        "wide_limit",
        "hostile_limit",
        "hostile_options",
        "hostile_message",
        "hostile_targeted_projection",
        "hostile_discovery_projection",
        "provider_targeted_projection",
        "provider_discovery_projection",
    ],
)
def test_public_inference_input_boundary(
    sqlite_resources, backend, case, capsys, caplog, monkeypatch
):
    canary = "auxiliary-rejected-value-secret-canary"
    bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)
    outcomes = []

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Exercise public auxiliary input validation",
            input_schema={"type": "object"},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("tool.summary",)),
        )

        async def run(self, ctx, args):
            request = ModelRequest(model="model", messages=[Message.text("user", "nested")])
            limits = bounds.model_copy(deep=True)
            purpose = "tool.summary"
            if case == "model":
                request.model = "other-model"
            elif case == "options":
                request.options = {"billing_identity": canary}
            elif case == "tools":
                request.tools = [{"name": "hidden_tool", "input_schema": {}}]
            elif case == "purpose":
                purpose = "tool.undeclared"
            elif case == "wide_limit":
                object.__setattr__(limits, "max_output_tokens", 11)
            elif case == "hostile_limit":
                object.__setattr__(limits, "max_input_tokens", Hostile())
            elif case == "hostile_options":
                request.options = {"untrusted": Hostile()}
            elif case == "hostile_message":
                object.__setattr__(request.messages[0].content[0], "text", Hostile())
            elif case == "hostile_targeted_projection":
                request.targeted_tool_projection = TargetedToolProjectionRequest.model_construct(
                    marker_id=Hostile(), tools=()
                )
            elif case == "hostile_discovery_projection":
                request.tool_discovery_projection = ToolDiscoveryProjectionRequest.model_construct(
                    generation_id=Hostile()
                )
            try:
                response = await ctx.inference.invoke(request, purpose=purpose, limits=limits)
            except (ValueError, TypeError) as exc:
                assert case != "valid"
                assert canary not in str(exc) and canary not in repr(exc)
                outcomes.append("refused")
                return ToolResult(content="request refused")
            assert case == "valid" and response.text == "summary"
            outcomes.append("completed")
            return ToolResult(content=response.text)

    async def scenario():
        path = sqlite_resources.path("input-boundary.sqlite")
        store = sqlite_resources.own(SQLiteSessionStore(path)) if backend == "sqlite" else None
        try:
            app = CayuApp(session_store=store, enable_logging=False)
            batches = [
                [
                    ModelStreamEvent.tool_call(name="summarize", id="parent", arguments={}),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ]
            ]
            if case == "valid":
                batches.append(
                    [
                        ModelStreamEvent.text_delta("summary"),
                        ModelStreamEvent.completed(
                            {"usage": {"input_tokens": 3, "output_tokens": 2}}
                        ),
                    ]
                )
            batches.append(
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ]
            )
            provider = ScriptedModelProvider(batches)
            if case.startswith("provider_"):
                prepare = provider.prepare_auxiliary_request

                def prepare_with_projection(request, *, max_output_tokens):
                    prepared = prepare(request, max_output_tokens=max_output_tokens)
                    if case == "provider_targeted_projection":
                        prepared.targeted_tool_projection = (
                            TargetedToolProjectionRequest.model_construct(
                                marker_id=Hostile(), tools=()
                            )
                        )
                    else:
                        prepared.tool_discovery_projection = (
                            ToolDiscoveryProjectionRequest.model_construct(generation_id=Hostile())
                        )
                    return prepared

                monkeypatch.setattr(provider, "prepare_auxiliary_request", prepare_with_projection)
            app.register_provider(provider, default=True)
            app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="input-boundary",
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert outcomes == ["completed" if case == "valid" else "refused"]
            assert len(provider.requests) == (3 if case == "valid" else 2)
            if store is not None:
                await store.close()
                store = sqlite_resources.own(SQLiteSessionStore(path))
                app = CayuApp(session_store=store, enable_logging=False)
            stored = await app.session_store.load_events("input-boundary")
            auxiliary = [
                event for event in stored if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(auxiliary) == int(case == "valid")
            assert all(canary not in event.model_dump_json() for event in events + stored)
            assert (
                await app.session_store.load_active_model_completion_stage("input-boundary") is None
            )
            usage = await app.get_session_usage("input-boundary")
            assert usage.model_steps == 2 and usage.usage.total_tokens == (
                9 if case == "valid" else 4
            )
        finally:
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        asyncio.run(run())
    assert all(canary not in str(warning.message) for warning in captured)
    assert canary not in caplog.text
    output = capsys.readouterr()
    assert canary not in output.out + output.err


@pytest.mark.parametrize(
    "phase",
    [
        "start_before",
        "start_after",
        "terminal_before",
        "terminal_after",
        "start_and_terminal",
        "start_child_cancel",
        "terminal_child_cancel",
        "start_dispatch_child_cancel",
        "start_prepare_child_cancel",
    ],
)
def test_public_auxiliary_publication_failure_preserves_recovery_owner(monkeypatch, phase):
    from cayu.runtime import IncompleteSessionRecoveryRequest
    from cayu.sessions.base import SessionModelCompletionStageConflict

    async def run():
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)
        failure = RuntimeError("injected auxiliary publication failure")
        secondary_failure = ValueError("injected terminal bookkeeping failure")
        observed = []
        injected = False
        terminal_injected = False
        child_cancellations = []

        async def cancel_child():
            async def child():
                asyncio.current_task().cancel()
                try:
                    await asyncio.sleep(0)
                except asyncio.CancelledError as exc:
                    child_cancellations.append(exc)
                    raise

            task = asyncio.create_task(child())
            try:
                await task
            finally:
                assert task.cancelled() and task.cancelling() == 1
                assert asyncio.current_task().cancelling() == 0

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Publication failure probe",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                try:
                    await ctx.inference.invoke(
                        ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                        purpose="tool.summary",
                        limits=bounds,
                    )
                except Exception as exc:
                    observed.append(exc)
                else:
                    raise AssertionError("Publication fault was not delivered")
                return ToolResult(content="failure handled")

        app = CayuApp(enable_logging=False)
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", id="parent", arguments={}),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}}),
                ],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])
        append = app.session_store.append_event
        complete = app.session_store.complete_model_completion_stage
        dispatch = app.session_store.mark_model_completion_stage_dispatched
        prepare = app.session_store.prepare_model_completion_stage

        async def prepare_stage(session_id, *, request, **kwargs):
            nonlocal injected
            result = await prepare(session_id, request=request, **kwargs)
            if phase == "start_prepare_child_cancel" and request.purpose == "auxiliary-inference":
                injected = True
                await cancel_child()
            return result

        async def dispatch_stage(session_id, *, stage, consume_child_session_notifications=False):
            nonlocal injected
            result = await dispatch(
                session_id,
                stage=stage,
                consume_child_session_notifications=consume_child_session_notifications,
            )
            if phase == "start_dispatch_child_cancel" and stage.purpose == "auxiliary-inference":
                injected = True
                await cancel_child()
            return result

        async def append_event(session_id, event):
            nonlocal injected
            if (
                not injected
                and phase.startswith("start")
                and event.type is EventType.MODEL_AUXILIARY_ATTEMPT_STARTED
            ):
                injected = True
                if phase.endswith("after"):
                    await append(session_id, event)
                if phase.endswith("child_cancel"):
                    await cancel_child()
                raise failure
            return await append(session_id, event)

        async def complete_stage(session_id, *, stage_id, publication):
            nonlocal injected, terminal_injected
            if (
                phase == "start_and_terminal"
                and not terminal_injected
                and publication.events[0].type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ):
                terminal_injected = True
                raise secondary_failure
            if (
                not injected
                and phase.startswith("terminal")
                and publication.events[0].type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ):
                injected = True
                if phase.endswith("after"):
                    await complete(session_id, stage_id=stage_id, publication=publication)
                if phase.endswith("child_cancel"):
                    await cancel_child()
                raise failure
            return await complete(session_id, stage_id=stage_id, publication=publication)

        monkeypatch.setattr(app.session_store, "append_event", append_event)
        monkeypatch.setattr(app.session_store, "complete_model_completion_stage", complete_stage)
        monkeypatch.setattr(
            app.session_store, "mark_model_completion_stage_dispatched", dispatch_stage
        )
        monkeypatch.setattr(app.session_store, "prepare_model_completion_stage", prepare_stage)
        events = []
        fenced = False
        try:
            async for event in app.run(
                RunRequest(
                    session_id="publication-failure",
                    agent_name="assistant",
                    messages=[Message.text("user", "go")],
                )
            ):
                events.append(event)
        except SessionModelCompletionStageConflict:
            # Without a committed terminal, the outer failure transition may
            # also refuse the unresolved exact stage. Recovery, not a fabricated
            # successful terminal write, owns this boundary.
            fenced = True
        assert injected
        if phase == "start_and_terminal":
            assert terminal_injected and len(observed) == 1
            assert isinstance(observed[0], ExceptionGroup)
            assert observed[0].exceptions == (failure, secondary_failure)
        elif phase.endswith("child_cancel"):
            assert len(observed) == 1 and isinstance(observed[0], RuntimeError)
            assert "without caller cancellation" in str(observed[0])
            assert len(child_cancellations) == 1
            assert exception_cause(observed[0]) is child_cancellations[0]
        else:
            assert observed == [failure]
        assert fenced or events[-1].type is EventType.SESSION_FAILED
        assert len(provider.requests) == (1 if phase.startswith("start") else 2)
        active = await app.session_store.load_active_model_completion_stage("publication-failure")
        assert active is not None
        dispatch_receipt = await app.session_store.load_model_completion_stage_dispatch(
            "publication-failure", active.stage.stage_id
        )
        assert (dispatch_receipt is None) == (phase == "start_prepare_child_cancel")
        assert active.stage.state == (
            "in_flight"
            if phase
            in {
                "terminal_before",
                "start_and_terminal",
                "terminal_child_cancel",
                "start_dispatch_child_cancel",
                "start_prepare_child_cancel",
            }
            else "completed"
        )
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="publication-failure", inactive_for_seconds=None
            )
        )
        assert len(provider.requests) == (1 if phase.startswith("start") else 2)
        assert (
            await app.session_store.load_active_model_completion_stage("publication-failure")
            is None
        )
        stored = await app.session_store.load_events("publication-failure")
        settled = [
            event for event in stored if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert len(settled) == (0 if phase == "start_prepare_child_cancel" else 1)
        if settled:
            assert settled[0].payload["auxiliary_outcome"] == (
                "completed"
                if phase == "terminal_after"
                else "outcome_unknown"
                if phase
                in {
                    "terminal_before",
                    "start_and_terminal",
                    "terminal_child_cancel",
                    "start_dispatch_child_cancel",
                }
                else "failed"
            )
        assert (await app.get_session_usage("publication-failure")).usage.total_tokens == (
            7 if phase == "terminal_after" else 2
        )

    asyncio.run(run())


def test_auxiliary_terminal_recovery_settles_before_promoting_parent_unchanged(monkeypatch):
    async def run():
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=5)
        app = CayuApp(
            enable_logging=False,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=1,
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="fake",
                                    model="fake-model",
                                    input_per_million=1,
                                    output_per_million=1,
                                ),
                            )
                        ),
                        reservation=BudgetReservation(
                            max_input_tokens=1000, max_output_tokens=1000
                        ),
                    ),
                )
            ),
        )
        injected = RuntimeError("terminal accounting interrupted")
        original = app._run_limit_controller.reconcile_model_completion_settlements
        failed = False
        recovered = []

        async def interrupt_accounting(event, **kwargs):
            nonlocal failed
            if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED and not failed:
                failed = True
                raise injected
            return await original(event, **kwargs)

        monkeypatch.setattr(
            app._run_limit_controller,
            "reconcile_model_completion_settlements",
            interrupt_accounting,
        )

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Recover terminal accounting",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                checkpoint = await app.session_store.load_checkpoint(ctx.session_id)
                transcript = await app.session_store.load_transcript(ctx.session_id)
                try:
                    await ctx.inference.invoke(
                        ModelRequest(model="fake-model", messages=[Message.text("user", "nested")]),
                        purpose="tool.summary",
                        limits=bounds,
                    )
                except RuntimeError as error:
                    assert error is injected
                else:
                    raise AssertionError("Accounting fault was not delivered")
                active = await app.session_store.load_active_model_completion_stage(ctx.session_id)
                assert active is not None and active.stage.state == "completed"
                session = await app.session_store.load(ctx.session_id)
                boundary = await app._session_engine._recovery_coordinator.reconcile_model_completion_boundary(
                    session
                )
                assert boundary.state == "promoted"
                assert (
                    await app.session_store.load_active_model_completion_stage(ctx.session_id)
                    is None
                )
                assert await app.session_store.load_checkpoint(ctx.session_id) == checkpoint
                assert await app.session_store.load_transcript(ctx.session_id) == transcript
                auxiliary = [
                    event
                    for event in boundary.recovery_events
                    if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
                ]
                assert len(auxiliary) == 1
                recovered.append(auxiliary[0].id)
                return ToolResult(
                    content="Accounting recovered; original response was not replayed"
                )

        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
            ],
            name="fake",
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Summarize()])
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="recover-auxiliary",
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        assert len(recovered) == 1
        assert len(provider.requests) == 3
        events = await app.session_store.load_events("recover-auxiliary")
        assert [
            event.id for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ] == recovered
        assert (await app.get_session_usage("recover-auxiliary")).usage.total_tokens == 9

    asyncio.run(run())


@pytest.mark.parametrize(
    "termination", ["deadline", "provider_deadline", "caller_cancel", "tail_error"]
)
@pytest.mark.parametrize("error_first", [False, True])
def test_public_inference_preserves_deadline_and_cancellation_classification(
    termination, error_first
):
    async def run():
        started = asyncio.Event()
        provider_cancelled = asyncio.Event()
        bounds = InferenceLimits(
            max_input_tokens=10,
            max_output_tokens=10,
            timeout_seconds=10 if termination == "caller_cancel" else 1,
        )
        observed = {}

        class Provider(ScriptedModelProvider):
            @property
            def stream_deadlines(self):
                if termination == "provider_deadline":
                    return ProviderStreamDeadlines(semantic_progress_timeout_s=0.1)
                return super().stream_deadlines

            async def stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self._consume_batch(request)
                    if error_first:
                        error = ModelProviderError(
                            "temporary failure before stalled tail",
                            provider="fake",
                            status_code=503,
                            retryable=True,
                        )
                        event = ModelStreamEvent.error(str(error), cause=error)
                        event.payload["usage"] = {"input_tokens": 3, "output_tokens": 2}
                        yield event
                    started.set()
                    if termination == "tail_error":
                        raise RuntimeError("stream tail failed")
                    try:
                        await asyncio.Event().wait()
                    except asyncio.CancelledError:
                        provider_cancelled.set()
                        raise
                    return
                async for event in super().stream(request):
                    yield event

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Bounded nested call",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                call = asyncio.create_task(
                    ctx.inference.invoke(
                        ModelRequest(model="fake-model", messages=[Message.text("user", "nested")]),
                        purpose="tool.summary",
                        limits=bounds,
                    )
                )
                await asyncio.wait_for(started.wait(), timeout=2)
                if termination == "caller_cancel":
                    call.cancel()
                try:
                    await call
                except BaseException as exc:
                    observed["error"] = exc
                    observed["cancelled"] = call.cancelled()
                    observed["cancelling"] = call.cancelling()
                if termination == "caller_cancel":
                    # Cancelling the inference waiter must signal the provider
                    # while this tool lifetime is still active, not at return.
                    await asyncio.wait_for(provider_cancelled.wait(), timeout=1)
                    assert asyncio.current_task().cancelling() == 0
                return ToolResult(content="Nested operation stopped")

        app = CayuApp(enable_logging=False)
        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [ModelStreamEvent.completed()],
            ],
            name="fake",
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Summarize()])
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="public-stop",
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        assert started.is_set()
        assert len(provider.requests) == 2
        active = await app.session_store.load_active_model_completion_stage("public-stop")
        assert active is not None
        assert active.stage.state == "completed"
        terminal = active.stage.publication.events[0]
        if termination in {"deadline", "provider_deadline"}:
            assert any(
                isinstance(
                    error, TimeoutError if termination == "deadline" else ModelStreamDeadlineError
                )
                for error in iter_exception_tree(observed["error"])
            )
            assert not observed["cancelled"]
            assert observed["cancelling"] == 0
            assert terminal.payload["auxiliary_outcome"] == "timed_out"
        elif termination == "tail_error":
            assert any(
                isinstance(error, RuntimeError) for error in iter_exception_tree(observed["error"])
            )
            assert not observed["cancelled"]
            assert observed["cancelling"] == 0
            assert terminal.payload["auxiliary_outcome"] == "failed"
        else:
            assert isinstance(observed["error"], asyncio.CancelledError)
            assert observed["cancelled"]
            assert observed["cancelling"] == 1
            assert terminal.payload["auxiliary_outcome"] == "cancelled"
        if error_first:
            assert terminal.payload["usage_status"] == "observed"
            assert terminal.payload["provider_error"]["status_code"] == 503
        if error_first and termination != "caller_cancel":
            # Causal evidence is retained separately from the current control
            # signal. asyncio timeout conversion legitimately uses __cause__.
            retained = list(iter_exception_tree(observed["error"]))
            for error in retained:
                cause = exception_cause(error)
                if cause is not None and all(cause is not item for item in retained):
                    retained.extend(iter_exception_tree(cause))
            assert (
                sum(
                    isinstance(error, ModelProviderError) and error.status_code == 503
                    for error in retained
                )
                == 1
            )

        from cayu.runtime import IncompleteSessionRecoveryRequest

        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="public-stop",
                inactive_for_seconds=None,
            )
        )
        assert len(provider.requests) == 2
        assert await app.session_store.load_active_model_completion_stage("public-stop") is None
        assert (await app.get_session_usage("public-stop")).usage.total_tokens == (
            7 if error_first else 2
        )
        report = await runtime_evidence(
            app,
            RuntimeEvidenceRequest(root_session_id="public-stop", max_sessions=10, max_events=1000),
        )
        projected = [
            attempt
            for attempt in report.sessions[0].attempts
            if attempt.operation is RuntimeEvidenceOperation.AUXILIARY_INFERENCE
        ]
        assert len(projected) == 1
        assert projected[0].status.value == terminal.payload["auxiliary_outcome"]
        assert projected[0].attempt_ordinal == 1
        assert projected[0].auxiliary_inference.purpose == "tool.summary"
        assert (
            len(
                [
                    event
                    for event in await app.session_store.load_events("public-stop")
                    if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
                ]
            )
            == 1
        )

    asyncio.run(run())


@pytest.mark.parametrize("child_task", [False, True])
@pytest.mark.parametrize("fail_after_response", [False, True])
def test_public_tool_inference_handle_records_nested_usage_and_expires(
    child_task, fail_after_response
):
    bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=5)
    handles = []

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Summarize with runtime inference",
            input_schema={"type": "object", "properties": {}},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("tool.summary",)),
        )

        async def run(self, ctx, args):
            assert ctx.inference is not None
            handles.append(ctx.inference)
            # Identical serialized public data cannot reconstruct a capability.
            reconstructed = ToolContext.model_validate_json(ctx.model_dump_json())
            assert reconstructed.session_id == ctx.session_id
            assert reconstructed.causal_budget_id == ctx.causal_budget_id
            assert reconstructed.idempotency_key == ctx.idempotency_key
            assert reconstructed.inference is None
            # A legitimate context copy retains the same single-use handle,
            # not authority derived from its editable public identity fields.
            copied = ctx.model_copy(
                update={
                    "session_id": "caller-forged-session",
                    "causal_budget_id": "caller-forged-causal",
                    "idempotency_key": "caller-forged-parent",
                    "metadata": {"model_attempt_id": "caller-forged-attempt"},
                }
            )
            assert copied.inference is ctx.inference
            for field in (
                "billing_identity",
                "principal",
                "causal_budget_id",
                "invocation_id",
                "tool_call_id",
                "model_attempt_id",
            ):
                with pytest.raises(ValueError):
                    ModelRequest(model="fake-model", messages=[], **{field: "caller-forged"})
            call = copied.inference.invoke(
                ModelRequest(model="fake-model", messages=[Message.text("user", "nested")]),
                purpose="tool.summary",
                limits=bounds,
            )
            response = await asyncio.create_task(call) if child_task else await call
            assert response.text == "summary"
            with pytest.raises(RuntimeError, match="consumed"):
                await ctx.inference.invoke(
                    ModelRequest(model="fake-model", messages=[]),
                    purpose="tool.summary",
                    limits=bounds,
                )
            if fail_after_response:
                raise RuntimeError("Tool failed after successful managed inference")
            return ToolResult(content=response.text)

    async def run():
        ledger = InMemoryBudgetLedger()
        app = CayuApp(
            enable_logging=False,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=1,
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="fake",
                                    model="fake-model",
                                    input_per_million=1,
                                    output_per_million=1,
                                ),
                            )
                        ),
                        reservation=BudgetReservation(
                            max_input_tokens=1000, max_output_tokens=1000
                        ),
                    ),
                )
            ),
        )
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
            ],
            name="fake",
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Summarize()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="public-inference",
                    messages=[Message.text("user", "go")],
                    limits=RunLimits(max_total_tokens=100),
                )
            )
        ]
        auxiliary = [
            event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert len(auxiliary) == 1, [
            (event.type, event.payload)
            for event in events
            if "fail" in event.type or "interrupt" in event.type
        ]
        expected_calls = 2 if fail_after_response else 3
        assert len(provider.requests) == expected_calls
        usage = await app.get_session_usage("public-inference")
        # An unhandled failure in this EXTERNAL tool fences its own effect and
        # stops the outer loop; the completed auxiliary usage remains known.
        assert usage.model_steps == (1 if fail_after_response else 2)
        assert usage.usage.total_tokens == (7 if fail_after_response else 9)
        terminal_type = (
            EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
            if fail_after_response
            else EventType.TOOL_CALL_COMPLETED
        )
        assert events.index(auxiliary[0]) < next(
            index for index, event in enumerate(events) if event.type is terminal_type
        )
        reservations = [event for event in events if event.type is EventType.BUDGET_RESERVED]
        assert len(reservations) == expected_calls
        stored_events = await app.session_store.load_events("public-inference")
        [stored_auxiliary] = [
            event
            for event in stored_events
            if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert stored_auxiliary.payload["auxiliary_outcome"] == "completed"
        assert stored_auxiliary.payload["auxiliary_inference"]["tool_call_id"] == "parent"
        assert stored_auxiliary.payload["usage_metrics"]["total_tokens"] == 5
        assert stored_auxiliary.session_id == "public-inference"
        assert stored_auxiliary.payload["provider_name"] == "fake"
        assert "caller-forged" not in stored_auxiliary.model_dump_json()
        stored_reservations = [
            event for event in stored_events if event.type is EventType.BUDGET_RESERVED
        ]
        assert len(stored_reservations) == expected_calls
        for event in stored_reservations:
            record = await ledger.load_reservation(event.payload["reservation_id"])
            assert record.status == "reconciled"
        assert (
            await app.session_store.load_active_model_completion_stage("public-inference") is None
        )
        with pytest.raises(RuntimeError, match="expired"):
            await handles[0].invoke(
                ModelRequest(model="fake-model", messages=[]), purpose="tool.summary", limits=bounds
            )
        assert len(provider.requests) == expected_calls

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["direct", "child", "abandon", "parent_cancel", "tool_timeout"])
def test_inference_scope_ends_with_real_tool_execution(mode):
    async def run():
        started = asyncio.Event()
        settled = asyncio.Event()
        caller_tasks = []
        observed_cancellations = []
        request = ModelRequest(model="fake", messages=[Message.text("user", "input")])
        limits = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=1)

        async def execute(request, purpose, limits):
            started.set()
            try:
                if mode in {"abandon", "parent_cancel", "tool_timeout"}:
                    await asyncio.Event().wait()
                return ModelResponse(events=(ModelStreamEvent.completed(),))
            except asyncio.CancelledError:
                observed_cancellations.append(asyncio.current_task().cancelling())
                raise
            finally:
                settled.set()

        scope = AuxiliaryInferenceScope(execute)
        ctx = ToolContext(session_id="scope-test")
        assert ctx.inference is None
        ctx._bind_runtime_inference(scope)
        assert "inference" not in ctx.model_dump()
        snapshot = ctx.model_copy(deep=True)
        assert snapshot.inference is ctx.inference
        assert ToolContext.model_validate_json(ctx.model_dump_json()).inference is None
        with pytest.raises(RuntimeError, match="expired"):
            await ctx.inference.invoke(request, purpose="tool.summary", limits=limits)

        class Scoped(Tool):
            spec = ToolSpec(
                name="scoped", description="Scoped test", input_schema={"type": "object"}
            )

            async def run(self, ctx, args):
                invocation = ctx.inference.invoke(request, purpose="tool.summary", limits=limits)
                if mode == "direct":
                    await invocation
                else:
                    child = asyncio.create_task(invocation)
                    caller_tasks.append(child)
                    await started.wait()
                    if mode == "child":
                        await child
                    elif mode in {"parent_cancel", "tool_timeout"}:
                        await asyncio.Event().wait()
                return ToolResult(content="done")

        task = asyncio.create_task(
            _tool_execution.run_tool(
                tool=Scoped(),
                effect=ToolEffect.NONE,
                ctx=ctx,
                arguments={},
                redactor=SecretRedactor,
                inference_scope=scope,
                timeout_seconds=0.1 if mode == "tool_timeout" else None,
            )
        )
        if mode == "parent_cancel":
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            assert task.cancelling() == 1
        else:
            outcome = await task
            assert outcome.result.is_error == (mode in {"abandon", "tool_timeout"})
        assert settled.is_set()
        assert observed_cancellations == ([] if mode in {"direct", "child"} else [1])
        if caller_tasks:
            await asyncio.gather(*caller_tasks, return_exceptions=True)
            assert all(child.done() for child in caller_tasks)
        with pytest.raises(RuntimeError, match="expired"):
            await ctx.inference.invoke(request, purpose="tool.summary", limits=limits)
        with pytest.raises(RuntimeError, match="reopened"):
            async with scope.lifetime():
                pytest.fail("Expired scope reopened")
        with pytest.raises(RuntimeError, match="expired"):
            await snapshot.inference.invoke(request, purpose="tool.summary", limits=limits)

    asyncio.run(run())


@pytest.mark.parametrize(
    "cancel_attempt,preparation_failure", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("with_budget", [False, True])
def test_auxiliary_attempt_runs_provider_and_publishes_before_tool_returns(
    monkeypatch, cancel_attempt, preparation_failure, with_budget
):
    """Exercise the live owner while the public ToolContext handle is being wired."""
    bound = {}
    original = ToolRoundExecutor.execute_tool_call
    limits = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=1)
    started = asyncio.Event()

    async def capture(self, **kwargs):
        bound.update(owner=self._auxiliary_inference, kwargs=kwargs)
        async with aclosing(original(self, **kwargs)) as events:
            async for event in events:
                yield event

    monkeypatch.setattr(ToolRoundExecutor, "execute_tool_call", capture)

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Summarize through runtime inference",
            input_schema={"type": "object", "properties": {}},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=limits, purposes=("tool.summary",)),
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            try:
                return await self.invoke_for_test(ctx, args)
            except BaseException as exc:
                bound["failure"] = exc
                raise

        async def invoke_for_test(self, ctx: ToolContext, args: dict) -> ToolResult:
            owner, values = bound["owner"], bound["kwargs"]
            invocation = values["invocation_context"]
            request, accepted = owner.prepare_request(
                invocation=invocation,
                registered_tool=invocation.registered_agent.executable_tool("summarize"),
                request=ModelRequest(
                    model="fake-model", messages=[Message.text("user", "summarize")]
                ),
                purpose="tool.summary",
                limits=limits,
            )
            admission = await owner.evaluate_admission(
                session=values["session"],
                invocation=invocation,
                policy=values["auxiliary_invocation_policy"],
                budget_limits=values["budget_limits"],
                request_limits=accepted,
            )
            assert admission.decision is None
            stage_values = dict(
                session=values["session"],
                invocation=invocation,
                policy=values["auxiliary_invocation_policy"],
                parent=values["tool_round_identity"],
                tool_name="summarize",
                tool_call_id=values["tool_call"].id,
                idempotency_key=ctx.idempotency_key,
                purpose="tool.summary",
                request=request,
                limits=accepted,
                attempt=0,
            )
            stage_request = owner.stage_request(**stage_values)
            setup = await owner.reserve_attempt(
                session=values["session"],
                invocation=invocation,
                budget_limits=values["budget_limits"],
                request_limits=accepted,
                identity=ModelAttemptIdentity(
                    model_step_id=stage_request.intent["model_step_id"],
                    model_attempt_id=stage_request.intent["model_attempt_id"],
                ),
                billing_identity=None,
            )
            assert setup.error is None and setup.failure is None
            bound["reservation_ids"] = tuple(
                item.record.reservation_id for item in setup.reservations
            )
            stage_request = owner.stage_request(**stage_values, reservations=setup.reservations)
            checkpoint = await app.session_store.load_checkpoint(ctx.session_id)
            transcript = await app.session_store.load_transcript(ctx.session_id)
            attempt = asyncio.create_task(
                owner.execute_attempt(
                    invocation=invocation,
                    request=request,
                    stage_request=stage_request,
                    reservations=setup.reservations,
                    limits=accepted,
                    expected_transcript_cursor=len(transcript),
                    redactor=SecretRedactor(),
                )
            )
            if preparation_failure:
                from cayu.runtime import _auxiliary_inference

                failure = RuntimeError("injected capacity exhaustion")

                def reject_capacity(*args):
                    raise failure

                with monkeypatch.context() as patch:
                    patch.setattr(
                        _auxiliary_inference, "reserve_provider_stream_cleanup", reject_capacity
                    )
                    with pytest.raises(RuntimeError) as caught:
                        await attempt
                    assert caught.value is failure
                assert (
                    await app.session_store.load_model_completion_stage(
                        ctx.session_id, stage_request.stage_id
                    )
                    is None
                )
                assert await app.session_store.load_checkpoint(ctx.session_id) == checkpoint
                assert await app.session_store.load_transcript(ctx.session_id) == transcript
                bound["released_before_preparation"] = True
                return ToolResult(content="Auxiliary capacity unavailable")
            if cancel_attempt:
                await asyncio.wait_for(started.wait(), timeout=2)
                attempt.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await attempt
                assert attempt.cancelled()
                assert attempt.cancelling() == 1
                active = await app.session_store.load_active_model_completion_stage(ctx.session_id)
                assert active is not None
                assert active.stage.stage_id == stage_request.stage_id
                assert active.stage.state == "completed"
                assert (
                    active.stage.publication.events[0].payload["auxiliary_outcome"] == "cancelled"
                )
                assert await app.session_store.load_checkpoint(ctx.session_id) == checkpoint
                assert await app.session_store.load_transcript(ctx.session_id) == transcript
                bound["cancelled_stage"] = stage_request.stage_id
                return ToolResult(content="Auxiliary request was interrupted")
            settled = await attempt
            response, events = settled.result, settled.events
            assert isinstance(response, ModelResponse)
            assert response.text == "nested summary"
            assert events[-1].type == EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            assert events[-1].payload["usage_status"] == "observed"
            assert await app.session_store.load_checkpoint(ctx.session_id) == checkpoint
            assert await app.session_store.load_transcript(ctx.session_id) == transcript
            assert (
                await app.session_store.load_active_model_completion_stage(ctx.session_id) is None
            )
            bound["settled"] = events[-1]
            return ToolResult(content=response.text)

    async def run():
        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                if cancel_attempt and request.messages == [Message.text("user", "summarize")]:
                    self._consume_batch(request)
                    started.set()
                    await asyncio.Event().wait()
                    return
                async for event in super().stream(request):
                    yield event

        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent-call"),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("nested summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 3, "output_tokens": 2}}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
            ],
            name="fake",
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Summarize()])
        observed = []
        async for item in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="auxiliary-live",
                messages=[Message.text("user", "go")],
                limits=RunLimits(max_total_tokens=100),
            )
        ):
            observed.append(item)
        if "failure" in bound:
            raise bound["failure"]
        assert len(bound["reservation_ids"]) == int(with_budget)
        for reservation_id in bound["reservation_ids"]:
            reservation = await ledger.load_reservation(reservation_id)
            if preparation_failure:
                assert reservation.status == "released"
                assert reservation.dispatch_id is None
                releases = [
                    event
                    for event in await app.session_store.load_events("auxiliary-live")
                    if event.type is EventType.BUDGET_RESERVATION_RELEASED
                    and event.payload.get("reservation_id") == reservation_id
                ]
                assert len(releases) == 1
                continue
            assert reservation.status == "reconciled"
            assert reservation.dispatch_id is not None
            assert reservation.actual_amount == (
                reservation.reserved_amount if cancel_attempt else Decimal("0.000005")
            )
        if preparation_failure:
            assert bound["released_before_preparation"]
            assert len(provider.requests) == 2
            return
        if cancel_attempt:
            assert "cancelled_stage" in bound
            assert len(provider.requests) == 2
            active = await app.session_store.load_active_model_completion_stage("auxiliary-live")
            assert active is not None
            assert active.stage.stage_id == bound["cancelled_stage"]
            return
        assert "settled" in bound, "\n".join(
            f"{item.type}: {item.payload}"
            for item in observed
            if item.type
            in {
                EventType.TOOL_CALL_BLOCKED,
                EventType.TOOL_CALL_FAILED,
                EventType.SESSION_INTERRUPTED,
            }
        )
        assert len(provider.requests) == 3
        usage = await app.get_session_usage("auxiliary-live")
        assert usage.model_steps == 2
        assert usage.usage.total_tokens == 9
        assert usage.unmeasured_model_attempts == 0

    ledger = InMemoryBudgetLedger()
    policy = (
        BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=1,
                    pricing=PriceBook(
                        prices=(
                            ModelPrice.fixed(
                                provider_name="fake",
                                model="fake-model",
                                input_per_million=1,
                                output_per_million=1,
                            ),
                        )
                    ),
                    reservation=BudgetReservation(max_input_tokens=1000, max_output_tokens=1000),
                ),
            )
        )
        if with_budget
        else None
    )
    app = CayuApp(enable_logging=False, budget_policy=policy, budget_ledger=ledger)
    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("scenario", ["retry", "exhausted", "budget", "backoff_deadline"])
def test_public_auxiliary_retry_retains_each_attempt_and_readmits(
    sqlite_resources, backend, scenario
):
    import base64

    from pydantic import SecretStr

    from cayu.runtime import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring

    bounds = InferenceLimits(
        max_input_tokens=10,
        max_output_tokens=10,
        timeout_seconds=3 if scenario == "backoff_deadline" else 10,
    )
    failures = []

    class Summarize(Tool):
        spec = ToolSpec(
            name="summarize",
            description="Managed retry",
            input_schema={"type": "object", "properties": {}},
            auxiliary_inference=AuxiliaryInferencePolicy(limits=bounds, purposes=("tool.summary",)),
        )

        async def run(self, ctx, args):
            try:
                response = await ctx.inference.invoke(
                    ModelRequest(model="fake-model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=bounds,
                )
            except Exception as exc:
                failures.append(exc)
                # Explicitly handled failure is a tool result, not uncertainty
                # about the enclosing external tool's own effects.
                return ToolResult(content="Auxiliary request refused", is_error=True)
            assert response.text == "summary"
            return ToolResult(content=response.text)

    class Provider(ScriptedModelProvider):
        def __init__(self):
            super().__init__([], name="fake")
            self.nested = 0
            self.outer = 0

        async def stream(self, request):
            self.requests.append(request.model_copy(deep=True))
            if request.options.get("fake", {}).get("max_output_tokens") == 10:
                self.nested += 1
                if self.nested == 1:
                    error = ModelProviderError(
                        "temporary overload",
                        provider="fake",
                        status_code=503,
                        retryable=True,
                        retry_after_s=10 if scenario == "backoff_deadline" else 0,
                    )
                    event = ModelStreamEvent.error("temporary overload", cause=error)
                    event.payload["usage"] = {"input_tokens": 3, "output_tokens": 2}
                    yield event
                else:
                    yield ModelStreamEvent.text_delta("summary")
                    yield ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 7, "output_tokens": 4}}
                    )
            else:
                self.outer += 1
                if self.outer == 1:
                    yield ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent")
                    yield ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    )
                else:
                    yield ModelStreamEvent.text_delta("done")
                    yield ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 1, "output_tokens": 1}}
                    )

    async def exercise():
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(
                    base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
                )
            },
        )
        codec = PublicAuthorityAliasCodec(keyring)
        store = (
            sqlite_resources.own(
                SQLiteSessionStore(
                    sqlite_resources.path("retry.sqlite"), public_authority_alias_codec=codec
                )
            )
            if backend == "sqlite"
            else None
        )
        ledger = InMemoryBudgetLedger()
        app = CayuApp(
            enable_logging=False,
            secret_redactor=SecretRedactor("unpriced_auxiliary_attempts"),
            public_authority_alias_keyring=keyring,
            session_store=store,
            budget_ledger=ledger,
            budget_policy=BudgetPolicy(
                limits=(
                    BudgetLimit(
                        scope="app",
                        max_estimated_cost=1,
                        pricing=PriceBook(
                            prices=(
                                ModelPrice.fixed(
                                    provider_name="fake",
                                    model="fake-model",
                                    input_per_million=1,
                                    output_per_million=1,
                                ),
                            )
                        ),
                        reservation=BudgetReservation(
                            max_input_tokens=1000, max_output_tokens=1000
                        ),
                    ),
                )
            ),
        )
        provider = Provider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Summarize()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="auxiliary-retry",
                    retry_policy=RetryPolicy(
                        max_attempts=1 if scenario == "exhausted" else 2,
                        initial_delay_s=0.0,
                        jitter_s=0.0,
                    ),
                    messages=[Message.text("user", "go")],
                    limits=RunLimits(max_total_tokens=26 if scenario == "budget" else 100),
                )
            )
        ]
        expected_attempts = 2 if scenario == "retry" else 1
        assert provider.nested == expected_attempts, [
            (event.type, event.payload.get("error"), event.payload.get("message"))
            for event in events[-3:]
        ]
        assert provider.outer == 2, [(event.type, event.payload) for event in events[-3:]]
        auxiliary = [
            event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert len(auxiliary) == expected_attempts
        assert [event.payload["auxiliary_outcome"] for event in auxiliary] == (
            ["failed", "completed"] if scenario == "retry" else ["failed"]
        )
        assert auxiliary[0].payload["provider_error"]["status_code"] == 503
        assert auxiliary[0].payload["retry_decision"]["attempt"] == 1
        assert auxiliary[0].payload["retry_decision"]["retry"] is (scenario != "exhausted")
        assert auxiliary[0].payload["usage_status"] == "observed"
        # Public projection deliberately hides private authority. Distinct
        # attempt identity is asserted below against the durable observations.
        from cayu.runtime._event_projection import PRIVATE_EVENT_AUTHORITY

        assert all(
            event.payload["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
            and event.payload["model_step_id"] == PRIVATE_EVENT_AUTHORITY
            for event in auxiliary
        )
        if scenario == "retry":
            assert not failures
        elif scenario == "exhausted":
            assert len(failures) == 1 and isinstance(failures[0], ModelProviderError)
            assert failures[0].status_code == 503
        elif scenario == "backoff_deadline":
            assert len(failures) == 1 and isinstance(failures[0], TimeoutError)
            # The first attempt is conclusively settled. Expiry while waiting
            # for its retry must not invent a second dispatched attempt.
            assert auxiliary[0].payload["retry_decision"]["delay_seconds"] == 10
        else:
            assert len(failures) == 1 and "invocation limits" in str(failures[0])
        stored = await app.session_store.load_events("auxiliary-retry")
        # Fixed runtime schema remains publishable even when a workload secret
        # equals its name; it must survive both durable and public projection.
        for observations in (events, stored):
            checks = [event for event in observations if event.type is EventType.BUDGET_CHECKED]
            assert checks
            assert all(event.payload["unpriced_auxiliary_attempts"] == 0 for event in checks)
        stored_auxiliary = [
            event for event in stored if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
        ]
        assert (
            len({event.payload["model_attempt_id"] for event in stored_auxiliary})
            == expected_attempts
        )
        assert len({event.payload["model_step_id"] for event in stored_auxiliary}) == 1
        assert all(
            event.payload["auxiliary_inference"]
            == stored_auxiliary[0].payload["auxiliary_inference"]
            for event in stored_auxiliary
        )
        assert [event.payload["auxiliary_outcome"] for event in stored_auxiliary] == [
            event.payload["auxiliary_outcome"] for event in auxiliary
        ]
        assert (
            stored_auxiliary[0].payload["retry_decision"] == auxiliary[0].payload["retry_decision"]
        )
        assert len({event.id for event in stored_auxiliary}) == expected_attempts
        usage = await app.get_session_usage("auxiliary-retry")
        assert usage.model_steps == 2
        assert usage.usage.total_tokens == (20 if scenario == "retry" else 9)
        report = await runtime_evidence(
            app,
            RuntimeEvidenceRequest(
                root_session_id="auxiliary-retry",
                include_causal_budget=True,
                max_sessions=10,
                max_events=1000,
            ),
        )
        projected = [
            attempt
            for attempt in report.sessions[0].attempts
            if attempt.operation is RuntimeEvidenceOperation.AUXILIARY_INFERENCE
        ]
        assert len(projected) == expected_attempts
        assert [attempt.attempt_ordinal for attempt in projected] == list(
            range(1, expected_attempts + 1)
        )
        assert [attempt.status.value for attempt in projected] == [
            event.payload["auxiliary_outcome"] for event in auxiliary
        ]
        assert projected[0].auxiliary_inference.purpose == "tool.summary"
        assert projected[0].auxiliary_inference.parent_tool_call_id == "parent"
        assert projected[0].auxiliary_inference.parent_model_step_id != projected[0].model_step_id
        assert report.lineage_totals.model_step_count == 2
        assert report.lineage_totals.attempt_count == expected_attempts + 2
        assert report.lineage_totals.provider_retry_attempt_count == expected_attempts - 1
        assert report.lineage_totals.usage.total_tokens == usage.usage.total_tokens
        assert report.causal_budget_totals == report.lineage_totals
        reservations = [event for event in stored if event.type is EventType.BUDGET_RESERVED]
        assert len(reservations) == expected_attempts + 2
        for event in reservations:
            reservation = await ledger.load_reservation(event.payload["reservation_id"])
            assert reservation.status == "reconciled"
        assert await app.session_store.load_active_model_completion_stage("auxiliary-retry") is None
        transcript = await app.session_store.load_transcript("auxiliary-retry")
        assert all(
            getattr(part, "text", None) != "summary"
            for message in transcript
            if message.role == "assistant"
            for part in message.content
        )
        if store is not None:
            await store.close()
            reopened = sqlite_resources.own(
                SQLiteSessionStore(
                    sqlite_resources.path("retry.sqlite"), public_authority_alias_codec=codec
                )
            )
            try:
                rebuilt_events = await reopened.load_events("auxiliary-retry")
                rebuilt_auxiliary = [
                    event
                    for event in rebuilt_events
                    if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
                ]
                assert [(event.id, event.payload) for event in rebuilt_auxiliary] == [
                    (event.id, event.payload) for event in stored_auxiliary
                ]
                assert session_usage_summary("auxiliary-retry", rebuilt_events).usage == usage.usage
                rebuilt_report = await runtime_evidence(
                    CayuApp(
                        session_store=reopened,
                        enable_logging=False,
                        secret_redactor=SecretRedactor("unpriced_auxiliary_attempts"),
                        public_authority_alias_keyring=keyring,
                    ),
                    RuntimeEvidenceRequest(
                        root_session_id="auxiliary-retry",
                        include_causal_budget=True,
                        max_sessions=10,
                        max_events=1000,
                    ),
                )
                assert rebuilt_report == report
                assert await reopened.load_active_model_completion_stage("auxiliary-retry") is None
            finally:
                await reopened.close()

    async def run():
        async with sqlite_resources:
            await exercise()

    asyncio.run(run())


@pytest.mark.parametrize("mutation_phase", ["unchanged", "tool", "dispatch"])
def test_public_auxiliary_dispatch_rechecks_admitted_provider_profile(monkeypatch, mutation_phase):
    async def run():
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=5)
        observed = []

        class Provider(ScriptedModelProvider):
            deadlines = ProviderStreamDeadlines()
            nested_calls = 0

            @property
            def stream_deadlines(self):
                return self.deadlines

            async def stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self.nested_calls += 1
                async for event in super().stream(request):
                    yield event

        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", arguments={}, id="parent"),
                    ModelStreamEvent.completed(
                        {
                            "finish_reason": "tool_calls",
                            "usage": {"input_tokens": 1, "output_tokens": 1},
                        }
                    ),
                ],
                [
                    ModelStreamEvent.text_delta("summary"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
            ],
            name="fake",
        )
        admitted_deadlines = provider.deadlines

        def change_deadlines():
            provider.deadlines = ProviderStreamDeadlines(semantic_progress_timeout_s=30)

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Profile bound inference",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                if mutation_phase == "tool":
                    change_deadlines()
                try:
                    response = await ctx.inference.invoke(
                        ModelRequest(model="fake-model", messages=[Message.text("user", "nested")]),
                        purpose="tool.summary",
                        limits=bounds,
                    )
                    assert response.text == "summary"
                except Exception as exc:
                    observed.append(exc)
                finally:
                    provider.deadlines = admitted_deadlines
                return ToolResult(content="Handled nested request")

        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Summarize()])
        original = app.session_store.mark_model_completion_stage_dispatched

        async def mark_then_change(*args, **kwargs):
            receipt = await original(*args, **kwargs)
            if mutation_phase == "dispatch" and kwargs["stage"].purpose == "auxiliary-inference":
                change_deadlines()
            return receipt

        monkeypatch.setattr(
            app.session_store, "mark_model_completion_stage_dispatched", mark_then_change
        )
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="provider-drift",
                    messages=[Message.text("user", "go")],
                )
            )
        ]
        assert provider.nested_calls == (1 if mutation_phase == "unchanged" else 0)
        if mutation_phase == "unchanged":
            assert not observed
            assert any(event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED for event in events)
        else:
            assert len(observed) == 1 and isinstance(observed[0], ExecutionProfileMismatchError)
            assert observed[0].changed_component_classes == (
                ExecutionProfileComponentClass.PROVIDER_ADAPTER,
            )
        active = await app.session_store.load_active_model_completion_stage("provider-drift")
        if mutation_phase == "dispatch":
            assert active is not None and active.stage.state == "completed"
            assert active.stage.publication.events[0].payload["auxiliary_outcome"] == "failed"
        else:
            assert active is None

    asyncio.run(run())


def test_auxiliary_invocation_policy_preserves_resolved_run_semantics():
    limits = RunLimits(max_total_tokens=100, scope="run")
    retry = RetryPolicy(max_attempts=2)
    policy = AuxiliaryInvocationPolicy(limits=limits, retry_policy=retry)
    assert policy.limits == limits
    assert policy.limits is not limits
    assert policy.limits is not policy.limits
    assert policy.retry_policy.max_attempts == 2
    # Replacing application/default policy does not affect this invocation.
    limits = RunLimits(max_total_tokens=1000, scope="session")
    retry = RetryPolicy(max_attempts=5)
    assert policy.limits.max_total_tokens == 100
    assert policy.limits.scope == "run"
    assert policy.retry_policy.max_attempts == 2


@pytest.mark.parametrize("limits,retry", [(None, RetryPolicy()), (RunLimits(), None)])
def test_auxiliary_invocation_policy_requires_resolved_values(limits, retry):
    with pytest.raises(TypeError, match="resolved limits and retry policy"):
        AuxiliaryInvocationPolicy(limits=limits, retry_policy=retry)


def test_auxiliary_invocation_policy_preserves_detached_run_baseline():
    accounting = RunLimitAccountingContext(
        started_at=datetime(2026, 7, 1, tzinfo=UTC),
        baseline=SessionUsageSummary(session_id="session", tool_calls=3),
    )
    policy = AuxiliaryInvocationPolicy(
        limits=RunLimits(max_total_tokens=100),
        retry_policy=RetryPolicy(max_attempts=2),
        accounting=accounting,
    )
    restored = RunLimitAccountingContext.model_validate_json(accounting.model_dump_json())
    resumed = AuxiliaryInvocationPolicy(
        limits=policy.limits, retry_policy=policy.retry_policy, accounting=restored
    )
    assert resumed.accounting == policy.accounting == accounting
    assert policy.accounting is not accounting
    assert policy.accounting.baseline is not accounting.baseline
    assert policy.accounting is not policy.accounting


@pytest.mark.parametrize("prepare_auxiliary", [False, True])
def test_public_run_passes_original_accounting_to_tool_execution(monkeypatch, prepare_auxiliary):
    captured = []
    original = ToolRoundExecutor.execute_tool_call
    bounds = InferenceLimits(max_input_tokens=5, max_output_tokens=5, timeout_seconds=1)

    async def observe(self, **kwargs):
        captured.append(kwargs["auxiliary_invocation_policy"])
        if prepare_auxiliary:
            owner = self._auxiliary_inference
            invocation = kwargs["invocation_context"]
            tool = invocation.registered_agent.executable_tool("echo")
            provider = invocation.registered_provider.provider
            dispatched_before = len(provider.requests)
            request = ModelRequest(model="fake-model", messages=[Message.text("user", "nested")])
            prepared, accepted = owner.prepare_request(
                invocation=invocation,
                registered_tool=tool,
                request=request,
                purpose="tool.summary",
                limits=bounds,
            )
            assert accepted == bounds
            assert prepared.messages == request.messages
            assert prepared is not request
            assert request.options == {}
            for invalid, purpose in (
                (request.model_copy(update={"model": "different"}), "tool.summary"),
                (request.model_copy(update={"options": {"api_key": "forbidden"}}), "tool.summary"),
                (request, "tool.undeclared"),
            ):
                with pytest.raises(ValueError):
                    owner.prepare_request(
                        invocation=invocation,
                        registered_tool=tool,
                        request=invalid,
                        purpose=purpose,
                        limits=bounds,
                    )
            prepare = provider.prepare_auxiliary_request

            def replace_content(candidate, *, max_output_tokens):
                candidate.messages = [Message.text("user", "replacement")]
                return prepare(candidate, max_output_tokens=max_output_tokens)

            with monkeypatch.context() as patch:
                patch.setattr(provider, "prepare_auxiliary_request", replace_content)
                with pytest.raises(ValueError, match="changed the authorized request"):
                    owner.prepare_request(
                        invocation=invocation,
                        registered_tool=tool,
                        request=request,
                        purpose="tool.summary",
                        limits=bounds,
                    )
            assert request.messages == [Message.text("user", "nested")]
            admission = await owner.evaluate_admission(
                session=kwargs["session"],
                invocation=invocation,
                policy=kwargs["auxiliary_invocation_policy"],
                budget_limits=kwargs["budget_limits"],
                request_limits=accepted,
            )
            assert admission.decision is not None
            assert admission.decision.limit == "total_tokens"
            assert admission.decision.actual == 12
            stage_kwargs = {
                "session": kwargs["session"],
                "invocation": invocation,
                "policy": kwargs["auxiliary_invocation_policy"],
                "parent": kwargs["tool_round_identity"],
                "tool_name": "echo",
                "tool_call_id": kwargs["tool_call"].id,
                "idempotency_key": "stable-key",
                "purpose": "tool.summary",
                "request": prepared,
                "limits": accepted,
                "attempt": 0,
            }
            stage = owner.stage_request(**stage_kwargs)
            assert owner.stage_request(**stage_kwargs) == stage
            changed = owner.stage_request(
                **{
                    **stage_kwargs,
                    "request": ModelRequest(
                        model="fake-model", messages=[Message.text("user", "changed")]
                    ),
                }
            )
            assert changed.stage_id == stage.stage_id
            assert changed.intent["request_fingerprint"] != stage.intent["request_fingerprint"]
            assert "nested" not in repr(stage.intent)
            retry = owner.stage_request(**{**stage_kwargs, "attempt": 1})
            assert retry.stage_id != stage.stage_id
            assert retry.intent["auxiliary_inference"] == stage.intent["auxiliary_inference"]
            assert stage.intent["session_instance_id"] == kwargs["session"].instance_id
            assert stage.intent["source_run_epoch"] == invocation.binding.run_epoch
            assert stage.intent["execution_profile_fingerprint"] == invocation.profile.fingerprint
            assert stage.intent["causal_budget_id"] == kwargs["session"].causal_budget_id
            with pytest.raises(ValueError, match="ordinal"):
                owner.stage_request(**{**stage_kwargs, "attempt": True})
            pricing = PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="fake",
                        model="fake-model",
                        input_per_million=1,
                        output_per_million=1,
                    ),
                )
            )
            identity = ModelAttemptIdentity(
                model_step_id=stage.intent["model_step_id"],
                model_attempt_id=stage.intent["model_attempt_id"],
            )
            for reservation in (None, BudgetReservation(max_input_tokens=4, max_output_tokens=5)):
                with pytest.raises(ValueError, match="reservation envelope"):
                    await owner.reserve_attempt(
                        session=kwargs["session"],
                        invocation=invocation,
                        budget_limits=(
                            BudgetLimit(
                                scope="app",
                                max_estimated_cost=1,
                                pricing=pricing,
                                reservation=reservation,
                            ),
                        ),
                        request_limits=accepted,
                        identity=identity,
                        billing_identity=None,
                    )
            assert len(provider.requests) == dispatched_before
        async with aclosing(original(self, **kwargs)) as stream:
            async for item in stream:
                yield item

    monkeypatch.setattr(ToolRoundExecutor, "execute_tool_call", observe)

    class Echo(Tool):
        spec = ToolSpec(
            name="echo",
            description="Echo",
            input_schema={"type": "object", "properties": {}},
            auxiliary_inference=(
                AuxiliaryInferencePolicy(limits=bounds, purposes=("tool.summary",))
                if prepare_auxiliary
                else None
            ),
        )

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="echoed")

    async def run():
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(name="echo", arguments={}, id="call"),
                        ModelStreamEvent.completed(
                            {
                                "finish_reason": "tool_calls",
                                **(
                                    {
                                        "usage": {
                                            "input_tokens": 1,
                                            "output_tokens": 1,
                                            "total_tokens": 2,
                                        }
                                    }
                                    if prepare_auxiliary
                                    else {}
                                ),
                            }
                        ),
                    ],
                    [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()],
                ],
                name="fake",
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="assistant", model="fake-model"), tools=[Echo()])
        async for _ in app.run(
            RunRequest(
                agent_name="assistant",
                session_id="auxiliary-policy-test",
                messages=[Message.text("user", "go")],
                limits=RunLimits(max_total_tokens=10 if prepare_auxiliary else 100),
                retry_policy=RetryPolicy(max_attempts=2),
            )
        ):
            pass
        summary = await app.get_session_usage("auxiliary-policy-test")
        assert summary.unmeasured_model_attempts == (1 if prepare_auxiliary else 2)

    asyncio.run(run())
    assert len(captured) == 1
    policy = captured[0]
    assert policy.limits.max_total_tokens == (10 if prepare_auxiliary else 100)
    assert policy.retry_policy.max_attempts == 2
    assert policy.accounting is not None
    assert policy.accounting.baseline.session_id == "auxiliary-policy-test"
    assert policy.accounting.baseline.tool_calls == 0


@pytest.mark.parametrize("maximum,allowed", [(9, False), (10, True), (11, True)])
def test_auxiliary_token_envelope_counts_prior_and_proposed_usage(maximum, allowed):
    usage = session_usage_summary(
        "s",
        [
            Event(
                type=EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED,
                session_id="s",
                payload={"usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5}},
            )
        ],
    )
    assert usage.model_steps == 0
    decision = auxiliary_token_admission(
        limits=RunLimits(max_total_tokens=maximum),
        usage=usage,
        request_limits=InferenceLimits(max_input_tokens=3, max_output_tokens=2, timeout_seconds=1),
    )
    assert (decision is None) is allowed


@pytest.mark.parametrize(
    "event_type", [EventType.MODEL_COMPLETED, EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED]
)
def test_missing_usage_remains_explicit_through_baseline_reconstruction(event_type):
    usage = session_usage_summary("s", [Event(type=event_type, session_id="s", payload={})])
    assert usage.unmeasured_model_attempts == 1
    accounting = RunLimitAccountingContext(
        started_at=datetime(2026, 7, 1, tzinfo=UTC),
        baseline=usage,
    )
    restored = RunLimitAccountingContext.model_validate_json(accounting.model_dump_json())
    assert restored.baseline.unmeasured_model_attempts == 1
    request = InferenceLimits(max_input_tokens=1, max_output_tokens=1, timeout_seconds=1)
    with pytest.raises(ValueError, match="Unmeasured model attempts"):
        auxiliary_token_admission(
            limits=RunLimits(max_total_tokens=100), usage=usage, request_limits=request
        )
    assert (
        auxiliary_token_admission(limits=RunLimits(), usage=usage, request_limits=request) is None
    )
