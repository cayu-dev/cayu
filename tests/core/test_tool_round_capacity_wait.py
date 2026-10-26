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


def test_round_waiting_past_admission_deadline_fails_before_any_tool_runs(caplog):
    holder = "tenant-b-session"

    async def scenario():
        store = InMemorySessionStore()
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=0.2)
        await governor.reserve_round(
            session_id=holder,
            tool_round_id="holder-round",
            maximum_bytes=None,
            unbounded_tool_names=("legacy_export",),
        )
        calls: list[str] = []
        app = _app(store, governor, calls)

        await asyncio.wait_for(_drain(app), timeout=20)

        persisted = await store.load_events("root")
        assert not any(event.type is EventType.TOOL_CALL_STARTED for event in persisted)
        failure = persisted[-1]
        assert failure.type is EventType.SESSION_FAILED
        assert failure.payload["error_type"] == "ToolRoundAdmissionTimeoutError"
        error = failure.payload["error"]
        assert "terminal publication capacity" in error
        assert "exclusive_groups=1" in error
        # The failure names only this session's tools. Other tenants' active
        # tool names and session IDs belong only in operator logs.
        assert "exclusive_holder_unbounded_tools" not in error
        assert "waiting_round_unbounded_tools=first_tool,second_tool." in error
        assert "declare the largest result each named tool returns." in error
        assert "exclusive_holder_sessions" not in error
        assert all(holder not in event.model_dump_json() for event in persisted)
        assert all("legacy_export" not in event.model_dump_json() for event in persisted)
        assert (await store.load("root")).status is SessionStatus.FAILED
        assert calls == []
        snapshot = governor.snapshot()
        assert snapshot.round_reservation_waiters == 0
        assert snapshot.active_round_reservations == 1

    with caplog.at_level(logging.WARNING, logger="cayu.tools.terminal_publication"):
        asyncio.run(scenario())
    (logged,) = [
        record
        for record in caplog.records
        if "failed waiting for terminal publication capacity" in record.getMessage()
    ]
    assert logged.levelno == logging.WARNING
    assert f"exclusive_holder_sessions={holder} " in logged.getMessage()
    assert "session_id=root " in logged.getMessage()
    assert "exclusive_holder_unbounded_tools=legacy_export " in logged.getMessage()


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


def test_capacity_wait_names_holders_and_unbounded_tools_within_bounds(caplog):
    async def scenario():
        governor = ToolTerminalPublicationGovernor(
            round_admission_timeout_s=0.2, round_admission_notice_s=0.05
        )
        family = _family("parent")
        tools = tuple(f"tool_{index:02}" for index in range(10))
        for index in range(6):
            await governor._reserve_invocation_round(
                session_id=f"family-{index}",
                tool_round_id=f"family-round-{index}",
                maximum_bytes=None,
                invocation=family,
                unbounded_tool_names=tools[index : index + 5],
            )
        with pytest.raises(ToolRoundAdmissionTimeoutError) as raised:
            await governor.reserve_round(
                session_id="root",
                tool_round_id="root-round",
                maximum_bytes=1024,
            )
        message = str(raised.value)
        assert message.endswith("round_waiters=1")
        assert "exclusive_holder_unbounded_tools" not in message
        assert all(tool not in message for tool in tools)
        assert "family-" not in message
        assert "waiting_round_unbounded_tools" not in message
        for index in range(6):
            governor.release_round(
                session_id=f"family-{index}", tool_round_id=f"family-round-{index}"
            )

    with caplog.at_level(logging.WARNING, logger="cayu.tools.terminal_publication"):
        asyncio.run(scenario())
    notice, failed = [record.getMessage() for record in caplog.records]
    assert notice.startswith("Tool round is still waiting")
    assert failed.startswith("Tool round failed waiting")
    for logged in (notice, failed):
        assert "session_id=root tool_round_id=root-round" in logged
        assert (
            "exclusive_holder_sessions=family-0,family-1,family-2,family-3,+2_more "
            "exclusive_holder_unbounded_tools="
            "tool_00,tool_01,tool_02,tool_03,tool_04,tool_05,tool_06,tool_07,+2_more."
        ) in logged


def test_capacity_wait_without_unbounded_tools_reports_only_the_lease():
    async def scenario():
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=0.1)
        # A declared round larger than capacity is exclusive for its size alone.
        await governor.reserve_round(
            session_id="oversized",
            tool_round_id="oversized-round",
            maximum_bytes=governor.staged_capacity_bytes + 1,
        )
        with pytest.raises(ToolRoundAdmissionTimeoutError) as raised:
            await governor.reserve_round(
                session_id="root", tool_round_id="root-round", maximum_bytes=1024
            )
        message = str(raised.value)
        assert message.endswith("round_waiters=1")
        assert "oversized" not in message
        governor.release_round(session_id="oversized", tool_round_id="oversized-round")

        # Declared rounds that merely fill capacity hold no exclusive lease.
        await governor.reserve_round(
            session_id="full",
            tool_round_id="full-round",
            maximum_bytes=governor.staged_capacity_bytes,
        )
        with pytest.raises(ToolRoundAdmissionTimeoutError) as raised:
            await governor.reserve_round(
                session_id="root",
                tool_round_id="root-round",
                maximum_bytes=None,
                unbounded_tool_names=("report", "report"),
            )
        message = str(raised.value)
        assert "exclusive_holder" not in message
        assert "exclusive_groups=0" in message
        assert "waiting_round_unbounded_tools=report." in message

    asyncio.run(scenario())


def test_unbounded_tool_names_explain_a_reservation_without_changing_it():
    async def scenario():
        governor = ToolTerminalPublicationGovernor()
        await governor.reserve_round(
            session_id="root",
            tool_round_id="root-round",
            maximum_bytes=None,
            unbounded_tool_names=("report",),
        )
        # A retry of the same round is the same reservation whatever it names.
        await governor.reserve_round(
            session_id="root", tool_round_id="root-round", maximum_bytes=None
        )
        assert governor.snapshot().active_round_reservations == 1
        for names in (["report"], ("",), (7,)):
            with pytest.raises(ValueError, match="unbounded_tool_names"):
                await governor.reserve_round(
                    session_id="other",
                    tool_round_id="other-round",
                    maximum_bytes=None,
                    unbounded_tool_names=names,
                )
        assert governor.snapshot().round_reservation_waiters == 0

    asyncio.run(scenario())


def test_capacity_advice_excludes_mcp_and_foreground_subagents():
    from tests.core.test_mcp import _fake_toolset

    from cayu.mcp.tools import McpToolAdapter
    from cayu.runtime._runtime_records import ToolCallRequest
    from cayu.runtime._tool_round_staging import _tool_terminal_payload_limits
    from cayu.tools.subagents import SubagentSpec, SubagentTool

    async def scenario():
        app = CayuApp(enable_logging=False)
        toolset = _fake_toolset()
        app.register_agent(
            AgentSpec(name="worker", model="test-model"),
            tools=[SubagentTool(app, agents={"child": SubagentSpec(agent_name="child")})],
            mcp_toolsets=[toolset],
        )
        registered = app._get_registered_agent("worker")
        tools = [
            tool
            for tool in registered.tools.values()
            if isinstance(tool.tool, (McpToolAdapter, SubagentTool))
        ]
        assert len(tools) == 2
        governor = ToolTerminalPublicationGovernor(round_admission_timeout_s=0.01)
        calls = [
            ToolCallRequest(id=str(index), name=tool.name, arguments={})
            for index, tool in enumerate(tools)
        ]
        limits = await _tool_terminal_payload_limits(
            registered, calls, publication_governor=governor
        )
        assert all(limits[call.id] is None for call in calls)
        assert limits.undeclared_tool_names == ()
        await governor.reserve_round(
            session_id="holder",
            tool_round_id="held",
            maximum_bytes=None,
            unbounded_tool_names=limits.undeclared_tool_names,
        )
        with pytest.raises(ToolRoundAdmissionTimeoutError) as raised:
            await governor.reserve_round(
                session_id="waiter", tool_round_id="waiting", maximum_bytes=1024
            )
        assert "exclusive_groups=1" in str(raised.value)
        assert "declare the largest result" not in str(raised.value)
        assert "ToolSpec.max_terminal_payload_bytes" not in str(raised.value)
        governor.release_round(session_id="holder", tool_round_id="held")
        await toolset.close()

    asyncio.run(scenario())
