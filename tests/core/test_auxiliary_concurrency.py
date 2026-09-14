from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from cayu import (
    AgentSpec,
    AuxiliaryInferencePolicy,
    CayuApp,
    EventType,
    InferenceLimits,
    Message,
    ModelRequest,
    ModelStreamEvent,
    RunLimits,
    RunRequest,
    ScriptedModelProvider,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.budgets.base import BudgetLimit, BudgetPolicy, BudgetReservation, InMemoryBudgetLedger
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.sessions.base import (
    ModelCompletionStageRequest,
    SessionModelCompletionStageConflict,
    SessionStatus,
)
from cayu.storage.sqlite import SQLiteSessionStore


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_live_auxiliary_preparation_conflicts_on_every_intent_field(sqlite_resources, backend):
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=60)

        class Provider(ScriptedModelProvider):
            async def stream(self, request):
                if request.messages == [Message.text("user", "nested")]:
                    self.requests.append(request.model_copy(deep=True))
                    entered.set()
                    await release.wait()
                    yield ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 3, "output_tokens": 2}}
                    )
                else:
                    async for event in super().stream(request):
                        yield event

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Live exact preparation",
                input_schema={"type": "object"},
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                await ctx.inference.invoke(
                    ModelRequest(model="model", messages=[Message.text("user", "nested")]),
                    purpose="tool.summary",
                    limits=bounds,
                )
                return ToolResult(content="done")

        store = (
            sqlite_resources.own(SQLiteSessionStore(sqlite_resources.path("exact.sqlite")))
            if backend == "sqlite"
            else None
        )
        ledger = InMemoryBudgetLedger()
        app = CayuApp(
            session_store=store,
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
                                    provider_name="scripted",
                                    model="model",
                                    input_per_million=1,
                                    output_per_million=1,
                                ),
                            )
                        ),
                        reservation=BudgetReservation(max_input_tokens=10, max_output_tokens=10),
                    ),
                )
            ),
        )
        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(name="summarize", id="parent", arguments={}),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
                [ModelStreamEvent.completed({})],  # Nested stream is supplied by the barrier above.
                [ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}})],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="exact-auxiliary",
                        agent_name="assistant",
                        messages=[Message.text("user", "go")],
                        limits=RunLimits(max_total_tokens=100),
                    )
                )
            ]

        task = sqlite_resources.task(consume())
        observer = None
        try:
            await asyncio.wait_for(entered.wait(), 15)
            # A separate SQLite connection reconstructs the live preparation;
            # no private Python provenance is used for the comparison.
            observer = (
                sqlite_resources.own(SQLiteSessionStore(sqlite_resources.path("exact.sqlite")))
                if backend == "sqlite"
                else app.session_store
            )
            active = await observer.load_active_model_completion_stage("exact-auxiliary")
            assert active is not None and active.stage.state == "in_flight"
            stage = active.stage
            request = ModelCompletionStageRequest(
                **{
                    field: getattr(stage, field)
                    for field in ModelCompletionStageRequest.model_fields
                }
            )
            assert request.reservation_ids and request.intent["budget_reservations"]
            checkpoint = await observer.load_checkpoint("exact-auxiliary")
            transcript = await observer.load_transcript("exact-auxiliary")
            events_before = await observer.load_events("exact-auxiliary")

            async def prepare(candidate, **overrides):
                return await observer.prepare_model_completion_stage(
                    "exact-auxiliary",
                    request=candidate,
                    **{
                        "expected_statuses": {SessionStatus.RUNNING},
                        "expected_run_epoch": stage.source_run_epoch,
                        "expected_transcript_cursor": stage.source_transcript_cursor,
                        **overrides,
                    },
                )

            replay = await prepare(request)
            assert not replay.dispatch_authorized and replay.stage == stage

            def leaves(value, path=()):
                if type(value) is dict and value:
                    for key, item in value.items():
                        yield from leaves(item, (*path, key))
                elif type(value) is list and value:
                    for index, item in enumerate(value):
                        yield from leaves(item, (*path, index))
                else:
                    yield path, value

            paths = list(leaves(request.intent))
            assert {path[0] for path, _ in paths} == set(request.intent)
            for path, value in paths:
                material = deepcopy(request.intent)
                parent = material
                for part in path[:-1]:
                    parent = parent[part]
                parent[path[-1]] = (
                    not value
                    if type(value) is bool
                    else value + 1
                    if type(value) in {int, float}
                    else value + "-changed"
                    if type(value) is str
                    else "changed"
                )
                changed = request.model_copy(update={"intent": material})
                with pytest.raises(SessionModelCompletionStageConflict):
                    await prepare(changed)
                assert (
                    await observer.load_active_model_completion_stage("exact-auxiliary") == active
                ), path
            for field, value in (
                ("logical_step_id", request.logical_step_id + "-changed"),
                ("dispatch_ordinal", request.dispatch_ordinal + 1),
                ("purpose", "assistant-turn"),
                ("reservation_ids", ("different-reservation",)),
            ):
                with pytest.raises(SessionModelCompletionStageConflict):
                    await prepare(request.model_copy(update={field: value}))
            for field, value in (
                ("expected_statuses", {SessionStatus.RUNNING, SessionStatus.PENDING}),
                ("expected_run_epoch", stage.source_run_epoch + 1),
                ("expected_transcript_cursor", stage.source_transcript_cursor + 1),
            ):
                with pytest.raises(SessionModelCompletionStageConflict):
                    await prepare(request, **{field: value})
            assert await observer.load_checkpoint("exact-auxiliary") == checkpoint
            assert await observer.load_transcript("exact-auxiliary") == transcript
            assert await observer.load_events("exact-auxiliary") == events_before
            assert len(provider.requests) == 2
            release.set()
            events = await asyncio.wait_for(task, 15)
            assert events[-1].type is EventType.SESSION_COMPLETED
            assert len(provider.requests) == 3
            assert (await app.get_session_usage("exact-auxiliary")).usage.total_tokens == 9
            for identity in request.reservation_ids:
                assert (await ledger.load_reservation(identity)).status == "reconciled"
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if backend == "sqlite" and observer is not None:
                await observer.close()
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize(
    "maximum,expected_calls,queued_deadline",
    [(26, 1, False), (27, 2, False), (28, 2, False), (100, 1, True)],
)
def test_parallel_auxiliary_requests_serialize_and_readmit(
    sqlite_resources, backend, maximum, expected_calls, queued_deadline
):
    async def scenario():
        bounds = InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=10)
        both_tools = asyncio.Event()
        both_prepared = asyncio.Event()
        first_dispatch = asyncio.Event()
        release = asyncio.Event()
        refusal_seen = asyncio.Event()
        entered = []
        refused = []
        prepared = []
        active = 0
        peak = 0
        nested_calls = 0

        class Provider(ScriptedModelProvider):
            def prepare_auxiliary_request(self, request, *, max_output_tokens):
                prepared.append(request)
                if len(prepared) == 2:
                    both_prepared.set()
                return super().prepare_auxiliary_request(
                    request, max_output_tokens=max_output_tokens
                )

            async def stream(self, request):
                nonlocal active, peak, nested_calls
                if request.options.get(self.name, {}).get("max_output_tokens") != 10:
                    # Auxiliary calls are recorded too, so select the outer
                    # script independently of ScriptedModelProvider's call index.
                    batch = self._batches[0 if not self.requests else 1]
                    self.requests.append(request.model_copy(deep=True))
                    for event in batch:
                        yield event
                    return
                self.requests.append(request.model_copy(deep=True))
                nested_calls += 1
                active += 1
                peak = max(peak, active)
                try:
                    if nested_calls == 1:
                        first_dispatch.set()
                        await release.wait()
                    yield ModelStreamEvent.text_delta("summary")
                    yield ModelStreamEvent.completed(
                        {"usage": {"input_tokens": 3, "output_tokens": 2}}
                    )
                finally:
                    active -= 1

        class Summarize(Tool):
            spec = ToolSpec(
                name="summarize",
                description="Concurrent managed request",
                input_schema={"type": "object", "properties": {"tag": {"type": "string"}}},
                parallel_safe=True,
                auxiliary_inference=AuxiliaryInferencePolicy(
                    limits=bounds, purposes=("tool.summary",)
                ),
            )

            async def run(self, ctx, args):
                entered.append(args["tag"])
                if len(entered) == 2:
                    both_tools.set()
                await both_tools.wait()
                if queued_deadline and args["tag"] == "two":
                    await first_dispatch.wait()
                request_limits = (
                    InferenceLimits(max_input_tokens=10, max_output_tokens=10, timeout_seconds=3)
                    if queued_deadline and args["tag"] == "two"
                    else bounds
                )
                try:
                    result = await ctx.inference.invoke(
                        ModelRequest(model="model", messages=[Message.text("user", args["tag"])]),
                        purpose="tool.summary",
                        limits=request_limits,
                    )
                except (RuntimeError, TimeoutError) as exc:
                    refused.append(exc)
                    refusal_seen.set()
                    return ToolResult(content="refused", is_error=True)
                return ToolResult(content=result.text)

        store = (
            sqlite_resources.own(SQLiteSessionStore(sqlite_resources.path("parallel.sqlite")))
            if backend == "sqlite"
            else None
        )
        app = CayuApp(session_store=store, enable_logging=False)
        provider = Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        name="summarize", id="one", arguments={"tag": "one"}
                    ),
                    ModelStreamEvent.tool_call(
                        name="summarize", id="two", arguments={"tag": "two"}
                    ),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"usage": {"input_tokens": 1, "output_tokens": 1}}),
                ],
            ]
        )
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="assistant", model="model"), tools=[Summarize()])

        async def consume():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="parallel-auxiliary",
                        messages=[Message.text("user", "go")],
                        limits=RunLimits(max_total_tokens=maximum),
                    )
                )
            ]

        task = sqlite_resources.task(consume())
        try:
            await asyncio.wait_for(first_dispatch.wait(), 10)
            await asyncio.wait_for(both_prepared.wait(), 10)
            assert sorted(entered) == ["one", "two"]
            assert nested_calls == 1 and active == 1
            stage = await app.session_store.load_active_model_completion_stage("parallel-auxiliary")
            assert stage is not None and stage.stage.state == "in_flight"
            if queued_deadline:
                await asyncio.wait_for(refusal_seen.wait(), 8)
                assert len(refused) == 1 and isinstance(refused[0], TimeoutError)
                assert nested_calls == 1 and active == 1
                assert (
                    await app.session_store.load_active_model_completion_stage("parallel-auxiliary")
                    == stage
                )
            release.set()
            events = await asyncio.wait_for(task, 15)
            assert events[-1].type is EventType.SESSION_COMPLETED, [
                (event.type, event.payload) for event in events if "fail" in event.type
            ]
            assert nested_calls == expected_calls and peak == 1 and active == 0
            assert len(refused) == 2 - expected_calls
            if not queued_deadline:
                assert all("invocation limits" in str(error) for error in refused)
            settled = [
                event for event in events if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len(settled) == expected_calls
            assert len({event.id for event in settled}) == expected_calls
            # Runtime identity is intentionally private in the public stream.
            # Check exact parent/attempt identities in their durable owner.
            settled = [
                event
                for event in await app.session_store.load_events("parallel-auxiliary")
                if event.type is EventType.MODEL_AUXILIARY_ATTEMPT_SETTLED
            ]
            assert len({event.payload["model_attempt_id"] for event in settled}) == expected_calls
            assert (
                len({event.payload["auxiliary_inference"]["tool_call_id"] for event in settled})
                == expected_calls
            )
            if queued_deadline:
                assert settled[0].payload["auxiliary_inference"]["tool_call_id"] == "one"
            usage = await app.get_session_usage("parallel-auxiliary")
            assert usage.model_steps == 2
            assert usage.usage.total_tokens == 4 + 5 * expected_calls
            assert (
                await app.session_store.load_active_model_completion_stage("parallel-auxiliary")
                is None
            )
        finally:
            release.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if store is not None:
                await store.close()

    async def run():
        async with sqlite_resources:
            await scenario()

    asyncio.run(run())
