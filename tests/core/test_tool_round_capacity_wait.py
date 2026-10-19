from __future__ import annotations

import asyncio
import logging
import time

import pytest
from tests.runtime.runtime_provider_support import FakeProvider

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.sessions import _pending_tool_round_reader as pending_round_reader
from cayu.sessions.invocation import InvocationOrigin, SessionExecutionSource, SessionInvocation
from cayu.sessions.memory import InMemorySessionStore
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import RunRequest
from cayu.tools.base import Tool, ToolEffect, ToolResult, ToolSpec
from cayu.tools.terminal_publication import (
    ToolRoundAdmissionTimeoutError,
    ToolTerminalPublicationGovernor,
)


class _ExternalTool(Tool):
    def __init__(self, name: str, calls: list[str]) -> None:
        # No declared terminal bound, so the round needs the exclusive lease.
        self.spec = ToolSpec(
            name=name,
            description="Record one external dispatch.",
            effect=ToolEffect.EXTERNAL,
        )
        self._calls = calls

    async def run(self, ctx, args):
        self._calls.append(ctx.idempotency_key)
        return ToolResult(content="done")


def _app(store, governor, calls):
    provider = FakeProvider(
        [
            [
                ModelStreamEvent.tool_call(id="first", name="first_tool", arguments={}),
                ModelStreamEvent.tool_call(id="second", name="second_tool", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app._tool_round_executor.invocation.terminals.governor = governor
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="root", model="fake-model"),
        tools=[_ExternalTool("first_tool", calls), _ExternalTool("second_tool", calls)],
    )
    return app


async def _drain(app):
    async for _event in app.run(
        RunRequest(agent_name="root", session_id="root", messages=[Message.text("user", "go")])
    ):
        pass


async def _wait_for_round_waiter(
    governor: ToolTerminalPublicationGovernor, *, count: int = 1
) -> None:
    for _ in range(2000):
        if governor.snapshot().round_reservation_waiters >= count:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("The tool round never waited for publication capacity.")


def _family(root_session_id: str) -> SessionInvocation:
    return SessionInvocation(
        origin=InvocationOrigin(trust="unattributed"),
        root_invocation_id="00000000-0000-4000-8000-000000000002",
        root_session_id=root_session_id,
        source=SessionExecutionSource.SDK_RUN,
    )


class _ReleaseRecordingGovernor(ToolTerminalPublicationGovernor):
    def __init__(self, **options) -> None:
        super().__init__(**options)
        self.released: list[tuple[str, str]] = []

    def release_round(self, *, session_id: str, tool_round_id: str) -> None:
        self.released.append((session_id, tool_round_id))
        super().release_round(session_id=session_id, tool_round_id=tool_round_id)


def test_cancelled_round_waiting_for_capacity_closes_without_waiting_again():
    async def scenario():
        store = InMemorySessionStore()
        # No deadline: the cancellation alone must settle the round.
        governor = _ReleaseRecordingGovernor(round_admission_timeout_s=None)
        # Another session's unbounded round owns the process-wide lease.
        await governor.reserve_round(
            session_id="other", tool_round_id="other-round", maximum_bytes=None
        )
        calls: list[str] = []
        app = _app(store, governor, calls)

        run = asyncio.create_task(_drain(app))
        await _wait_for_round_waiter(governor)
        persisted = await store.load_events("root")
        assert persisted[-1].type is EventType.MODEL_COMPLETED
        assert not any(event.type is EventType.TOOL_CALL_STARTED for event in persisted)
        tool_round_id = persisted[-1].payload["tool_round_id"]

        started = time.monotonic()
        run.cancel()
        await asyncio.gather(run, return_exceptions=True)
        # Before the fix, closing the round queued for the same lease again and
        # only stopped at the 30 s retained-cleanup step timeout.
        assert time.monotonic() - started < 10

        persisted = await store.load_events("root")
        assert not any(event.type is EventType.TOOL_CALL_STARTED for event in persisted)
        assert [
            event.payload["tool_call_id"]
            for event in persisted
            if event.type is EventType.TOOL_CALL_FAILED
        ] == ["first", "second"]
        assert (await store.load("root")).status is SessionStatus.INTERRUPTED
        _, pending_round = await pending_round_reader.load_pending_tool_round(store, "root")
        assert pending_round is None
        assert calls == []
        # A close with nothing staged still drops any lease the round kept.
        assert ("root", tool_round_id) in governor.released
        snapshot = governor.snapshot()
        assert snapshot.round_reservation_waiters == 0
        assert snapshot.active_round_reservations == 1

    asyncio.run(scenario())


def test_round_waiting_past_admission_deadline_fails_before_any_tool_runs():
    async def scenario():
        store = InMemorySessionStore()
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=0.2)
        await governor.reserve_round(
            session_id="other", tool_round_id="other-round", maximum_bytes=None
        )
        calls: list[str] = []
        app = _app(store, governor, calls)

        await asyncio.wait_for(_drain(app), timeout=20)

        persisted = await store.load_events("root")
        assert not any(event.type is EventType.TOOL_CALL_STARTED for event in persisted)
        failure = persisted[-1]
        assert failure.type is EventType.SESSION_FAILED
        assert failure.payload["error_type"] == "ToolRoundAdmissionTimeoutError"
        assert "terminal publication capacity" in failure.payload["error"]
        assert "exclusive_groups=1" in failure.payload["error"]
        assert (await store.load("root")).status is SessionStatus.FAILED
        assert calls == []
        snapshot = governor.snapshot()
        assert snapshot.round_reservation_waiters == 0
        assert snapshot.active_round_reservations == 1

    asyncio.run(scenario())


def test_related_rounds_of_unknown_lineage_cannot_hold_a_waiter_forever(caplog):
    async def scenario():
        governor = ToolTerminalPublicationGovernor(
            round_admission_timeout_s=0.3, round_admission_notice_s=0.1
        )
        family = _family("builder-parent")

        async def reserve_sibling(index: int) -> None:
            await governor._reserve_invocation_round(
                session_id=f"builder-{index % 2}",
                tool_round_id=f"builder-round-{index}",
                maximum_bytes=None,
                invocation=family,
            )

        await reserve_sibling(0)
        waiter = asyncio.create_task(
            governor.reserve_round(
                session_id="root", tool_round_id="root-round", maximum_bytes=None
            )
        )
        await _wait_for_round_waiter(governor)
        # Without lineage every related round may be awaited by the family's
        # holder, so each one joins the lease ahead of the queued root and the
        # family never drains.
        for index in range(1, 12):
            await reserve_sibling(index)
            governor.release_round(
                session_id=f"builder-{(index - 1) % 2}",
                tool_round_id=f"builder-round-{index - 1}",
            )
            await asyncio.sleep(0.05)
        with pytest.raises(ToolRoundAdmissionTimeoutError, match="other_reservation_groups=1"):
            await asyncio.wait_for(waiter, timeout=5)
        assert governor.snapshot().round_reservation_waiters == 0
        governor.release_round(session_id="builder-1", tool_round_id="builder-round-11")
        assert governor.snapshot().active_round_reservations == 0

    with caplog.at_level(logging.WARNING, logger="cayu.tools.terminal_publication"):
        asyncio.run(scenario())
    assert any(
        "still waiting for terminal publication capacity" in record.getMessage()
        and "tool_round_id=root-round" in record.getMessage()
        for record in caplog.records
    )


def test_sibling_rounds_queue_behind_a_waiter_when_no_ancestor_holds_a_lease():
    async def scenario():
        # No deadline: the waiter must be admitted on its own merits.
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=None)
        family = _family("builder-parent")

        async def reserve_sibling(index: int) -> None:
            session_id = f"builder-{index % 2}"
            await governor._reserve_invocation_round(
                session_id=session_id,
                tool_round_id=f"builder-round-{index}",
                maximum_bytes=None,
                invocation=family,
                lineage=(session_id, "builder-parent"),
            )

        await reserve_sibling(0)
        waiter = asyncio.create_task(
            governor.reserve_round(
                session_id="root", tool_round_id="root-round", maximum_bytes=None
            )
        )
        await _wait_for_round_waiter(governor)
        # The other sibling starts a round while the first one still holds the
        # family's lease. It takes its turn behind the root instead of keeping
        # the lease alive.
        sibling = asyncio.create_task(reserve_sibling(1))
        await _wait_for_round_waiter(governor, count=2)
        assert governor.snapshot().round_reservation_waiters == 2
        governor.release_round(session_id="builder-0", tool_round_id="builder-round-0")
        await asyncio.wait_for(waiter, timeout=2)
        assert not sibling.done()
        governor.release_round(session_id="root", tool_round_id="root-round")
        await asyncio.wait_for(sibling, timeout=2)
        governor.release_round(session_id="builder-1", tool_round_id="builder-round-1")
        snapshot = governor.snapshot()
        assert snapshot.active_round_reservations == 0
        assert snapshot.round_reservation_waiters == 0

    asyncio.run(scenario())


def test_rounds_nested_under_a_held_family_lease_still_skip_queued_waiters():
    async def scenario():
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=None)
        family = _family("parent")
        await governor._reserve_invocation_round(
            session_id="parent",
            tool_round_id="parent-round",
            maximum_bytes=None,
            invocation=family,
            lineage=("parent",),
        )
        waiter = asyncio.create_task(
            governor.reserve_round(
                session_id="root", tool_round_id="root-round", maximum_bytes=None
            )
        )
        await _wait_for_round_waiter(governor)
        # The parent may be awaiting any of its descendants, so they must not
        # wait behind a round that is itself waiting for the parent to finish.
        for session_id, lineage in (
            ("child-0", ("child-0", "parent")),
            ("child-1", ("child-1", "parent")),
            ("grandchild", ("grandchild", "child-0", "parent")),
        ):
            await asyncio.wait_for(
                governor._reserve_invocation_round(
                    session_id=session_id,
                    tool_round_id=f"{session_id}-round",
                    maximum_bytes=None,
                    invocation=family,
                    lineage=lineage,
                ),
                timeout=2,
            )
        assert not waiter.done()
        for session_id in ("grandchild", "child-1", "child-0", "parent"):
            governor.release_round(session_id=session_id, tool_round_id=f"{session_id}-round")
        await asyncio.wait_for(waiter, timeout=2)
        governor.release_round(session_id="root", tool_round_id="root-round")
        assert governor.snapshot().active_round_reservations == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("lineage", [(), ("other",), ("builder", ""), ["builder"]])
def test_round_lineage_must_start_with_the_requesting_session(lineage):
    async def scenario():
        governor = ToolTerminalPublicationGovernor()
        with pytest.raises(ValueError, match="lineage must start with session_id"):
            await governor._reserve_invocation_round(
                session_id="builder",
                tool_round_id="builder-round",
                maximum_bytes=None,
                invocation=_family("parent"),
                lineage=lineage,
            )
        assert governor.snapshot().active_round_reservations == 0

    asyncio.run(scenario())


def test_round_admitted_before_its_deadline_keeps_waiting_normally():
    async def scenario():
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=5)
        await governor.reserve_round(
            session_id="other", tool_round_id="other-round", maximum_bytes=None
        )
        waiter = asyncio.create_task(
            governor.reserve_round(
                session_id="root", tool_round_id="root-round", maximum_bytes=None
            )
        )
        await _wait_for_round_waiter(governor)
        governor.release_round(session_id="other", tool_round_id="other-round")
        await asyncio.wait_for(waiter, timeout=2)
        assert governor.snapshot().active_round_reservations == 1
        governor.release_round(session_id="root", tool_round_id="root-round")

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "options",
    [
        {"round_admission_timeout_s": 0},
        {"round_admission_timeout_s": float("inf")},
        {"round_admission_timeout_s": True},
        {"round_admission_notice_s": -1},
        {"round_admission_notice_s": None},
    ],
)
def test_round_admission_timing_must_be_positive_finite_seconds(options):
    with pytest.raises(ValueError, match="positive finite number of seconds"):
        ToolTerminalPublicationGovernor(**options)
