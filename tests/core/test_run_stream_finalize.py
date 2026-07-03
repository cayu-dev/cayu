"""An abandoned run/resume event stream must finalize — not strand — the session.

When a consumer closes the event stream mid-run (client disconnect, abandoned async
generator), GeneratorExit reaches the run generator. The runtime must transition the
still-RUNNING session to INTERRUPTED and persist a terminal event, so the session
stays observable and resumable instead of being stranded in RUNNING forever.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import NamedTuple

from cayu.core import AgentSpec, Message
from cayu.core.events import Event, EventType
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    BudgetLimit,
    BudgetPolicy,
    BudgetReservation,
    CayuApp,
    InMemorySessionStore,
    InterruptSessionRequest,
    ModelPricing,
    PricingCatalog,
    ResumeRequest,
    RunRequest,
    RuntimeHook,
    RuntimeHookContext,
    SessionIdentity,
    SessionStatus,
)


class FakeProvider(ModelProvider):
    name = "fake"

    def __init__(self, batches: list[list[ModelStreamEvent]]) -> None:
        self.event_batches = batches
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        batch_index = len(self.requests)
        self.requests.append(request)
        if batch_index >= len(self.event_batches):
            raise AssertionError(f"No fake provider event batch for request {batch_index}")
        for event in self.event_batches[batch_index]:
            yield event


class Harness(NamedTuple):
    app: CayuApp
    store: InMemorySessionStore
    provider: FakeProvider


def _batch(text: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.text_delta(text),
        ModelStreamEvent.completed({"finish_reason": "stop"}),
    ]


def _build(batches: list[list[ModelStreamEvent]]) -> Harness:
    store = InMemorySessionStore()
    provider = FakeProvider(batches)
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))
    return Harness(app, store, provider)


def _pricing() -> PricingCatalog:
    return PricingCatalog(
        prices=(
            ModelPricing(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )


def _abandoned_terminal_event(events: list[Event]) -> Event:
    interrupted = [event for event in events if event.type == EventType.SESSION_INTERRUPTED]
    assert len(interrupted) == 1
    return interrupted[0]


async def _close_event_stream(stream: AsyncIterator[Event]) -> None:
    close = getattr(stream, "aclose", None)
    assert close is not None
    await close()


def test_abandoned_run_stream_finalizes_running_session() -> None:
    # The first provider batch is consumed by the post-abandon resume, proving the
    # abandoned run never reached the model and the session stayed resumable.
    h = _build([_batch("answer after abandonment")])

    async def scenario() -> tuple[Event, list[Event]]:
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_run",
                messages=[Message.text("user", "hello")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_STARTED
        # The consumer walks away mid-run (e.g. SSE client disconnect).
        await _close_event_stream(stream)
        events = await h.store.load_events("sess_abandoned_run")
        return first_event, events

    _, events = asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_abandoned_run"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    terminal = _abandoned_terminal_event(events)
    assert terminal.payload == {
        "interruption_type": "runtime_interrupted",
        "reason": "event_stream_closed",
        "abandoned": True,
    }

    # The finalized session is resumable — it was not stranded in RUNNING.
    async def resume() -> None:
        async for _ in h.app.resume(
            ResumeRequest(
                session_id="sess_abandoned_run",
                messages=[Message.text("user", "continue")],
            )
        ):
            pass

    asyncio.run(resume())
    resumed = asyncio.run(h.store.load("sess_abandoned_run"))
    assert resumed is not None
    assert resumed.status == SessionStatus.COMPLETED
    assert len(h.provider.requests) == 1


def test_abandoned_resume_stream_finalizes_running_session() -> None:
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        async for _ in h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_resume",
                messages=[Message.text("user", "hello")],
            )
        ):
            pass
        stream = h.app.resume(
            ResumeRequest(
                session_id="sess_abandoned_resume",
                messages=[Message.text("user", "continue")],
            )
        )
        first_event = await anext(stream)
        assert first_event.type == EventType.SESSION_RESUMED
        await _close_event_stream(stream)

    asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_abandoned_resume"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    events = asyncio.run(h.store.load_events("sess_abandoned_resume"))
    terminal = _abandoned_terminal_event(events)
    assert terminal.payload["reason"] == "event_stream_closed"
    assert terminal.payload["abandoned"] is True


def test_finalize_abandoned_session_by_id_finalizes_and_is_idempotent() -> None:
    # The shared finalizer that the run-factory-window and tool-approval GeneratorExit
    # guards call to close a session stranded RUNNING before _run_session's own finalizer.
    h = _build([_batch("x")])

    async def scenario() -> None:
        await h.store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_strand",
                messages=[Message.text("user", "hi")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await h.store.transition_status(
            "sess_strand",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )

        await h.app._finalize_abandoned_session_by_id("sess_strand")
        first = await h.store.load("sess_strand")
        assert first is not None and first.status == SessionStatus.INTERRUPTED

        # Idempotent: a second call (e.g. also reached by _run_session's finalizer) no-ops.
        await h.app._finalize_abandoned_session_by_id("sess_strand")
        second = await h.store.load("sess_strand")
        assert second is not None and second.status == SessionStatus.INTERRUPTED

        # Unknown session id is a safe no-op.
        await h.app._finalize_abandoned_session_by_id("does-not-exist")

        events = await h.store.load_events("sess_strand")
        interrupted = [e for e in events if e.type == EventType.SESSION_INTERRUPTED]
        assert len(interrupted) == 1
        assert interrupted[0].payload["abandoned"] is True

    asyncio.run(scenario())


def test_completed_run_stream_close_is_a_no_op() -> None:
    # Closing an already-finished stream must not rewrite the terminal status.
    h = _build([_batch("first answer")])

    async def scenario() -> None:
        stream = h.app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_completed_close",
                messages=[Message.text("user", "hello")],
            )
        )
        async for _ in stream:
            pass
        await _close_event_stream(stream)

    asyncio.run(scenario())

    session = asyncio.run(h.store.load("sess_completed_close"))
    assert session is not None
    assert session.status == SessionStatus.COMPLETED
    events = asyncio.run(h.store.load_events("sess_completed_close"))
    assert not [event for event in events if event.type == EventType.SESSION_INTERRUPTED]


def test_abandoned_run_stream_releases_active_budget_reservation() -> None:
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    }
                )
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("1"),
                    pricing=_pricing(),
                    reservation=BudgetReservation(
                        max_input_tokens=1_000_000,
                        max_output_tokens=0,
                    ),
                ),
            )
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> list[Event]:
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_budget",
                messages=[Message.text("user", "hello")],
            )
        )
        observed = [await anext(stream), await anext(stream), await anext(stream)]
        assert [event.type for event in observed] == [
            EventType.SESSION_STARTED,
            EventType.BUDGET_CHECKED,
            EventType.BUDGET_RESERVED,
        ]
        await _close_event_stream(stream)
        retry_events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="sess_after_abandoned_budget",
                    messages=[Message.text("user", "retry")],
                )
            )
        ]
        return retry_events

    retry_events = asyncio.run(scenario())

    assert EventType.BUDGET_RESERVED in [event.type for event in retry_events]
    assert retry_events[-1].type == EventType.SESSION_COMPLETED
    assert len(provider.requests) == 1


def test_abandoned_run_stream_settles_remaining_budget_reservations() -> None:
    store = InMemorySessionStore()
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 1,
                            "output_tokens": 0,
                            "total_tokens": 1,
                        },
                    }
                )
            ],
        ]
    )
    pricing = _pricing()
    reservation = BudgetReservation(
        max_input_tokens=1_000_000,
        max_output_tokens=0,
    )
    app = CayuApp(
        session_store=store,
        budget_policy=BudgetPolicy(
            limits=(
                BudgetLimit(
                    scope="app",
                    max_estimated_cost=Decimal("10"),
                    pricing=pricing,
                    reservation=reservation,
                ),
                BudgetLimit(
                    scope="agent",
                    key="assistant",
                    max_estimated_cost=Decimal("10"),
                    pricing=pricing,
                    reservation=reservation,
                ),
            )
        ),
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> list[Event]:
        stream = app.run(
            RunRequest(
                agent_name="assistant",
                session_id="sess_abandoned_budget_reconcile",
                messages=[Message.text("user", "hello")],
            )
        )
        observed: list[Event] = []
        while True:
            event = await anext(stream)
            observed.append(event)
            if event.type == EventType.BUDGET_RECONCILED:
                break
        await _close_event_stream(stream)
        return await store.load_events("sess_abandoned_budget_reconcile")

    events = asyncio.run(scenario())
    event_types = [event.type for event in events]

    assert event_types.count(EventType.BUDGET_RESERVED) == 2
    assert event_types.count(EventType.BUDGET_RECONCILED) == 2


def test_interrupt_stream_close_drains_terminal_hooks() -> None:
    class RecordingInterruptHook(RuntimeHook):
        def __init__(self) -> None:
            self.sessions: list[str] = []

        async def after_session_interrupted(self, context: RuntimeHookContext) -> None:
            self.sessions.append(context.session.id)

    hook = RecordingInterruptHook()
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, runtime_hooks=[hook], enable_logging=False)
    app.register_provider(FakeProvider([]), default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_interrupt_close",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        stream = app.interrupt_session(
            InterruptSessionRequest(
                session_id="sess_interrupt_close",
                reason="operator",
            )
        )
        first = await anext(stream)
        assert first.type == EventType.SESSION_INTERRUPTED
        await _close_event_stream(stream)

    asyncio.run(scenario())

    assert hook.sessions == ["sess_interrupt_close"]


def test_finalize_abandoned_session_by_id_tolerates_missing_environment() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake-model"))

    async def scenario() -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_environment",
                environment_name="removed-env",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.transition_status(
            "sess_missing_environment",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        await app._finalize_abandoned_session_by_id("sess_missing_environment")

    asyncio.run(scenario())

    session = asyncio.run(store.load("sess_missing_environment"))
    events = asyncio.run(store.load_events("sess_missing_environment"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert events[-1].type == EventType.SESSION_INTERRUPTED
    assert events[-1].environment_name == "removed-env"


def test_finalize_abandoned_session_by_id_records_terminal_event_without_agent() -> None:
    store = InMemorySessionStore()
    app = CayuApp(session_store=store, enable_logging=False)

    async def scenario() -> list[Event]:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id="sess_missing_agent",
                messages=[Message.text("user", "hello")],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.transition_status(
            "sess_missing_agent",
            from_statuses={SessionStatus.PENDING},
            to_status=SessionStatus.RUNNING,
        )
        await app._finalize_abandoned_session_by_id("sess_missing_agent")
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        return [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="sess_missing_agent",
                    reason="operator",
                )
            )
        ]

    interrupt_events = asyncio.run(scenario())

    session = asyncio.run(store.load("sess_missing_agent"))
    stored_events = asyncio.run(store.load_events("sess_missing_agent"))
    assert session is not None
    assert session.status == SessionStatus.INTERRUPTED
    assert [event.type for event in stored_events] == [EventType.SESSION_INTERRUPTED]
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert interrupt_events[0].payload["reason"] == "event_stream_closed"
