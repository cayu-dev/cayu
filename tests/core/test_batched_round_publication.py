"""A round's staged results publish in one write when no after-tool hook runs.

The crash tests stop the process after each publication step, recover in a
fresh application, and compare the recovered session with what result-by-
result publication (forced by a no-op after-tool hook) reaches from its own
crash points.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import AsyncIterator
from typing import Any

import pytest
from tests.core._execution_profile_fixtures import (
    interrupt_and_release_test_invocation,
    versioned_test_provider_identity,
)
from tests.core._runtime_hook_fixtures import NoopAfterToolHook

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.recovery import IncompleteSessionRecoveryRequest
from cayu.sessions.requests import InterruptSessionRequest, RunRequest
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec

_TERMINALS = {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
_CALL_IDS = ("call-a", "call-b")


class _ProcessLoss(BaseException):
    pass


class _CountingTool(Tool):
    spec = ToolSpec(
        name="count",
        input_schema={"type": "object", "properties": {"label": {"type": "string"}}},
        max_terminal_payload_bytes=1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:batched-round-count", behavior_version="1", implementation_version="1"
        ),
    )

    def __init__(self, runs: dict[str, int]) -> None:
        super().__init__()
        self.runs = runs

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        label = str(args["label"])
        self.runs[label] = self.runs.get(label, 0) + 1
        return ToolResult(content=f"counted {label}")


class _TwoCallProvider(ModelProvider):
    name = "two-calls"

    def __init__(self) -> None:
        self.requests = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests == 1:
            for call_id in _CALL_IDS:
                yield ModelStreamEvent.tool_call(
                    id=call_id, name="count", arguments={"label": call_id}
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _PublicationLossStore(InMemorySessionStore):
    """Lose the process (or an acknowledgement) at one result-publication step."""

    invocation_lifecycle_command_version = 1

    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase
        self.failed = False
        self.batched_publications: list[list[str]] = []
        self.terminal_appends = 0
        self.started_stages = 0
        self.batched_stage_writes = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        # What the first application did, before recovery publishes anything.
        self.live: tuple[int, int] = (0, 0)

    def _lose(self, phase: str, *, acknowledgement: bool = False) -> None:
        if self.failed or self.phase != phase:
            return
        self.failed = True
        if acknowledgement:
            raise RuntimeError(f"acknowledgement lost after {phase}")
        raise _ProcessLoss(phase)

    async def append_events(self, session_id, events):
        terminal_ids = [
            event.payload["tool_call_id"] for event in events if event.type in _TERMINALS
        ]
        if len(terminal_ids) < 2:
            # One result's own append (when the store has no atomic claim).
            await super().append_events(session_id, events)
            if terminal_ids:
                self.terminal_appends += 1
                self._lose(f"append-{self.terminal_appends}")
                self._lose(f"append-{self.terminal_appends}-acknowledgement", acknowledgement=True)
            return
        self.batched_publications.append(terminal_ids)
        if self.phase == "block-batch" and not self.failed:
            self.failed = True
            self.started.set()
            await self.release.wait()
        self._lose("before-events")
        await super().append_events(session_id, events)
        self._lose("after-events")
        self._lose("events-acknowledgement", acknowledgement=True)

    async def append_event_and_claim_side_effect(self, session_id, event, **kwargs):
        result = await super().append_event_and_claim_side_effect(session_id, event, **kwargs)
        if event.type in _TERMINALS:
            self.terminal_appends += 1
            self._lose(f"append-{self.terminal_appends}")
            self._lose(f"append-{self.terminal_appends}-acknowledgement", acknowledgement=True)
        return result

    async def transform_checkpoint(self, session_id, transform):
        session = await super().load(session_id)
        before = await super().load_checkpoint(session_id)
        # Stage transforms are pure, so a dry run shows what this write does.
        predicted = transform(session, copy.deepcopy(before))
        started = _started_stage_count(predicted) - _started_stage_count(before)
        if started >= 2:
            self.batched_stage_writes += 1
            self._lose("before-stages")
            result = await super().transform_checkpoint(session_id, transform)
            self._lose("after-stages")
            return result
        result = await super().transform_checkpoint(session_id, transform)
        if started:
            self.started_stages += 1
            self._lose(f"start-{self.started_stages}")
        return result

    async def publish_runtime_publication(self, session_id, *, request, **kwargs):
        if request.kind == "tool-round":
            self._lose("before-round-commit")
        return await super().publish_runtime_publication(session_id, request=request, **kwargs)


def _started_stage_count(checkpoint: dict[str, Any] | None) -> int:
    pending = (checkpoint or {}).get("pending_tool_round")
    if type(pending) is not dict:
        return 0
    return sum(
        1
        for stage in pending.get("staged_terminals") or ()
        if type(stage) is dict and stage.get("publication_started_at") is not None
    )


def _application(store, provider, runs, *, hooks: bool) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[_CountingTool(runs)],
        runtime_hooks=[NoopAfterToolHook()] if hooks else None,
    )
    return app


def _outcome(phase: str, *, hooks: bool) -> tuple[dict[str, Any], _PublicationLossStore]:
    async def run() -> tuple[dict[str, Any], _PublicationLossStore]:
        store = _PublicationLossStore(phase)
        runs: dict[str, int] = {}
        session_id = f"session-batched-round-{phase}-{hooks}"
        crashed = False
        try:
            _ = [
                event
                async for event in _application(store, _TwoCallProvider(), runs, hooks=hooks).run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "count twice")],
                    )
                )
            ]
        except _ProcessLoss:
            crashed = True
        store.live = (store.terminal_appends, store.started_stages)
        recovery_requests = 0
        if crashed:
            await interrupt_and_release_test_invocation(store, session_id)
            provider = _TwoCallProvider()
            await _application(store, provider, runs, hooks=hooks).recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id=session_id)
            )
            recovery_requests = provider.requests
        durable = [
            record.event for record in await store.query_events(EventQuery(session_id=session_id))
        ]
        terminals = sorted(
            (
                event.payload["tool_call_id"],
                event.type.value,
                event.payload["result"]["content"],
            )
            for event in durable
            if event.type in _TERMINALS
        )
        transcript = await store.load_transcript(session_id)
        tool_results = sorted(
            (part.tool_call_id, part.content)
            for message in transcript
            for part in message.content
            if getattr(part, "tool_call_id", None) is not None and hasattr(part, "content")
        )
        checkpoint = await store.load_checkpoint(session_id)
        return (
            {
                "crashed": crashed,
                "runs": runs,
                "terminals": terminals,
                "tool_results": tool_results,
                "pending_round_left": bool(checkpoint and "pending_tool_round" in checkpoint),
                "recovery_requests": recovery_requests,
            },
            store,
        )

    return asyncio.run(run())


_RECOVERED = {
    "crashed": True,
    "runs": {"call-a": 1, "call-b": 1},
    "terminals": [
        ("call-a", "tool.call.completed", "counted call-a"),
        ("call-b", "tool.call.completed", "counted call-b"),
    ],
    "tool_results": [("call-a", "counted call-a"), ("call-b", "counted call-b")],
    "pending_round_left": False,
    "recovery_requests": 0,
}


@pytest.mark.parametrize(
    "phase",
    ["start-1", "start-2", "append-1", "append-2", "before-round-commit"],
)
def test_result_by_result_publication_recovers_each_crash(phase: str) -> None:
    outcome, store = _outcome(phase, hooks=True)

    assert store.batched_publications == []
    assert outcome == _RECOVERED


@pytest.mark.parametrize(
    ("phase", "result_by_result_phase"),
    [
        ("before-stages", "start-1"),
        # Every result's hooks are complete and none is appended: the state a
        # loss just before a result-by-result append leaves that result in.
        ("after-stages", "start-2"),
        ("before-events", "start-2"),
        ("after-events", "append-2"),
        ("before-round-commit", "before-round-commit"),
    ],
)
def test_batched_publication_recovers_each_crash_as_result_by_result_did(
    phase: str, result_by_result_phase: str
) -> None:
    batched, store = _outcome(phase, hooks=False)
    reference, _ = _outcome(result_by_result_phase, hooks=True)

    # No result was started or appended on its own, and recovery reached the
    # same session.
    assert store.live == (0, 0)
    assert store.batched_stage_writes == 1
    assert batched == reference == _RECOVERED


def test_batched_publication_writes_the_round_results_twice_in_all() -> None:
    outcome, store = _outcome("none", hooks=False)

    assert store.batched_stage_writes == 1
    assert store.batched_publications == [list(_CALL_IDS)]
    assert store.live == (0, 0)
    assert outcome == {**_RECOVERED, "crashed": False}


def test_a_lost_batch_acknowledgement_fails_the_run_as_a_lost_append_did() -> None:
    batched, store = _outcome("events-acknowledgement", hooks=False)
    reference, _ = _outcome("append-2-acknowledgement", hooks=True)

    assert store.batched_publications == [list(_CALL_IDS)]
    # Both runs fail with every result durable and the round left for recovery.
    assert batched == reference
    assert batched["terminals"] == _RECOVERED["terminals"]
    assert batched["pending_round_left"] is True


def test_after_tool_hooks_keep_result_by_result_publication() -> None:
    outcome, store = _outcome("none", hooks=True)

    assert store.batched_publications == []
    assert store.terminal_appends == 2
    assert outcome == {**_RECOVERED, "crashed": False}


def test_batched_events_reach_the_stream_once_in_call_order() -> None:
    async def run() -> list[Event]:
        store = _PublicationLossStore("none")
        return [
            event
            async for event in _application(store, _TwoCallProvider(), {}, hooks=False).run(
                RunRequest(
                    agent_name="assistant",
                    session_id="session-batched-round-stream",
                    messages=[Message.text("user", "count twice")],
                )
            )
        ]

    events = asyncio.run(run())

    terminals = [event for event in events if event.type in _TERMINALS]
    assert len(terminals) == 2 and len({event.id for event in terminals}) == 2
    assert [event.type for event in events].index(EventType.TOOL_CALL_COMPLETED) > max(
        index for index, event in enumerate(events) if event.type is EventType.TOOL_CALL_STARTED
    )


def test_interruption_during_the_batched_write_publishes_each_result_once() -> None:
    async def run():
        async def collect(stream):
            return [event async for event in stream]

        store = _PublicationLossStore("block-batch")
        runs: dict[str, int] = {}
        app = _application(store, _TwoCallProvider(), runs, hooks=False)
        session_id = "session-batched-round-interrupted"
        consumer = asyncio.create_task(
            collect(
                app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        messages=[Message.text("user", "count twice")],
                    )
                )
            )
        )
        try:
            await asyncio.wait_for(store.started.wait(), timeout=30)
            interruption = asyncio.create_task(
                collect(
                    app.interrupt_session(
                        InterruptSessionRequest(session_id=session_id, reason="operator stop")
                    )
                )
            )
            for _ in range(100):
                if consumer.cancelling():
                    break
                await asyncio.sleep(0)
        finally:
            # Release a write the interruption did not cancel.
            store.release.set()
        public, interrupted = await asyncio.gather(consumer, interruption)
        await _application(store, _TwoCallProvider(), runs, hooks=False).recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=session_id)
        )
        durable = [
            record.event for record in await store.query_events(EventQuery(session_id=session_id))
        ]
        return runs, public, interrupted, durable, await store.load_checkpoint(session_id)

    runs, _public, interrupted, durable, checkpoint = asyncio.run(run())

    assert runs == {"call-a": 1, "call-b": 1}
    assert sorted(
        event.payload["tool_call_id"] for event in durable if event.type in _TERMINALS
    ) == list(_CALL_IDS)
    assert [event.type for event in interrupted] == [EventType.SESSION_INTERRUPTED]
    assert not (checkpoint and "pending_tool_round" in checkpoint)
