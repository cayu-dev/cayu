from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from tests.core.test_mcp_stdio_modern import _client, _expected_request_meta, _requests

from cayu import (
    AgentSpec,
    CayuApp,
    McpIdleTimeoutError,
    McpProtocolEra,
    McpProtocolError,
    McpServerSpec,
    McpToolsetRefreshState,
    McpToolsetUnavailable,
    McpTransportLimits,
    StdioMcpClient,
    ToolContext,
    connect_mcp_toolset,
)
from cayu.mcp._stdio_process import stdio_mcp_parent_death_containment_platform_candidate
from cayu.mcp._subscriptions import ModernToolSubscription

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _spec(log: Path, mode: str = "normal") -> McpServerSpec:
    return McpServerSpec(
        name="stdio",
        connection_id="stdio-subscription",
        command=(sys.executable, str(FIXTURES / "fake_mcp_stdio_subscription_server.py")),
        env={"CAYU_FAKE_MCP_REQUEST_LOG": str(log), "CAYU_SUBSCRIPTION_MODE": mode},
    )


async def _until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0.001)


def test_modern_stdio_subscription_refreshes_and_rejects_stale_dispatch(tmp_path):
    log = tmp_path / "requests"

    async def run():
        toolset = await connect_mcp_toolset(_spec(log), client=_client())
        app = CayuApp(enable_logging=False)
        old = toolset.tools[0]
        try:
            app.register_agent(AgentSpec(name="agent", model="unused"), mcp_toolsets=(toolset,))
            assert toolset.refresh_state is McpToolsetRefreshState.DIRTY
            await _until(lambda: toolset.refresh_state is McpToolsetRefreshState.READY)
            await toolset.session.read_resource("control://change")
            await _until(lambda: "mcp__stdio__added" in app.get_agent("agent").tools)
            with pytest.raises(McpToolsetUnavailable):
                await old.run(ToolContext(session_id="test", agent_name="agent"), {})
            result = await toolset.session.call_tool("search", {})
            assert result.content[0]["text"] == "ok"
        finally:
            await toolset.close()
        assert toolset.session.process.returncode is not None

    asyncio.run(run())
    requests = _requests(log)
    listens = [r for r in requests if r["method"] == "subscriptions/listen"]
    assert len(listens) == 1
    assert listens[0]["params"] == {
        "notifications": {"toolsListChanged": True},
        "_meta": _expected_request_meta(),
    }
    cancellations = [r for r in requests if r["method"] == "notifications/cancelled"]
    assert cancellations[0]["params"]["requestId"] == listens[0]["id"]
    assert cancellations[0]["params"]["_meta"] == _expected_request_meta()


@pytest.mark.parametrize(
    "action",
    [
        "wrong-id",
        "boolean-id",
        "duplicate-ack",
        "unrequested",
        "bad-complete",
        "oversize",
        "malformed",
        "exit",
    ],
)
def test_modern_stdio_subscription_fault_fences_shared_connection(tmp_path, action):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        continuity = []
        changes = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True])
            session.transport_limits = McpTransportLimits(max_message_bytes=4096)
            with pytest.raises(McpProtocolError):
                await session.read_resource("control://" + action)
            await _until(lambda: session._closed)
            assert continuity[-1] is False
            assert changes == []
            with pytest.raises(McpProtocolError):
                await session.list_tools()
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_stdio_subscription_graceful_completion_relists_without_restarting(tmp_path):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        continuity = []
        pid = session.process.pid
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: continuity == [False, True])
            first = session._tool_subscription.state.request_id
            await session.read_resource("control://complete")
            await _until(lambda: continuity == [False, True, False, True])
            assert session._tool_subscription.state.request_id != first
            assert session.process.pid == pid
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_stdio_subscription_detach_reattach_ignores_cancelled_frames(tmp_path):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        continuity = []
        changes = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True])
            first = session._tool_subscription.state.request_id
            session._set_tools_list_changed_handler(None)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True, False, True])
            assert session._tool_subscription.state.request_id != first
            await session.read_resource("control://old")
            await session.read_resource("control://uncorrelated")
            assert changes == []
            await session.read_resource("control://change")
            await _until(lambda: len(changes) == 10)
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_stdio_subscription_unsupported_filter_stays_manually_refreshable(tmp_path):
    log = tmp_path / "requests"

    async def run():
        session = await _client().connect(_spec(log, "unsupported"))
        continuity = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: pytest.fail("unexpected change"))
            await _until(lambda: session._tool_subscription.task is None)
            assert continuity[-1] is True
            assert len(await session.list_tools()) == 1
            session._set_tools_list_changed_handler(None)
            session._set_tools_list_changed_handler(lambda: None)
            assert continuity[-1] is True
            assert len(await session.list_tools()) == 1
        finally:
            await session.close()

    asyncio.run(run())
    assert sum(r["method"] == "subscriptions/listen" for r in _requests(log)) == 1


@pytest.mark.parametrize("mode", ["no-ack", "pre-ack"])
def test_modern_stdio_subscription_establishment_cannot_be_extended_by_other_replies(
    tmp_path, mode
):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests", mode))
        session.transport_limits = McpTransportLimits(total_call_timeout_s=0.1, idle_timeout_s=1)
        continuity = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            async with asyncio.timeout(2):
                while not session._closed:
                    try:
                        await session.list_tools()
                    except McpProtocolError:
                        break
                    await asyncio.sleep(0.005)
            assert True not in continuity
            assert session._closed
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_stdio_subscription_survives_quiet_beyond_call_deadlines(tmp_path):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        session.transport_limits = McpTransportLimits(total_call_timeout_s=0.1, idle_timeout_s=0.3)
        continuity = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: continuity == [False, True])
            await asyncio.sleep(0.4)
            assert not session._closed
            await session.list_tools()
            assert continuity == [False, True]
        finally:
            await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("contained", [False, True])
def test_modern_stdio_subscription_official_sdk_interoperability(tmp_path, contained):
    if contained and not stdio_mcp_parent_death_containment_platform_candidate():
        pytest.skip("Default process containment requires supported Linux enforcement.")
    log = tmp_path / "requests"
    spec = McpServerSpec(
        name="sdk",
        connection_id="sdk-stdio-subscriptions",
        command=(sys.executable, str(FIXTURES / "official_mcp_stdio_subscription_server.py")),
        env={"CAYU_SDK_MCP_REQUEST_LOG": str(log)},
    )
    client = (
        StdioMcpClient(protocol_era=McpProtocolEra.MODERN_2026_07_28) if contained else _client()
    )

    async def run():
        toolset = await connect_mcp_toolset(spec, client=client)
        app = CayuApp(enable_logging=False)
        try:
            app.register_agent(AgentSpec(name="agent", model="unused"), mcp_toolsets=(toolset,))
            await _until(lambda: toolset.refresh_state is McpToolsetRefreshState.READY)
            toolset.session.transport_limits = McpTransportLimits(
                idle_timeout_s=0.2, total_call_timeout_s=2
            )
            await asyncio.sleep(0.3)
            assert not toolset.session._closed
            old = toolset.tools[0]
            await toolset.session.read_resource("control://advance")
            await _until(lambda: "mcp__sdk__added" in app.get_agent("agent").tools)
            assert not old._dispatch_authority_is_current()
            # Stop just the listener. The SDK must process cancellation while
            # the process is still alive and ordinary requests remain usable.
            toolset.session._set_tools_list_changed_handler(None)
            await _until(lambda: any(r.get("subscriptionClosed") for r in _requests(log)))
            assert toolset.session.process.returncode is None
            assert len(await toolset.session.list_tools()) == 2
        finally:
            await toolset.close()
        assert toolset.session.process.returncode is not None

    asyncio.run(run())
    requests = _requests(log)
    assert sum(r.get("method") == "subscriptions/listen" for r in requests) == 1
    assert all(r.get("method") not in {"initialize", "notifications/initialized"} for r in requests)


@pytest.mark.parametrize("during_cancel", [False, True])
def test_modern_stdio_subscription_retains_interrupted_writer_without_close_deadlock(
    tmp_path, during_cancel
):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        continuity = []
        entered = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        original_write = session._write

        async def resistant_write(payload):
            target = "notifications/cancelled" if during_cancel else "subscriptions/listen"
            if payload.get("method") != target:
                await original_write(payload)
                return
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            finally:
                finished.set()

        try:
            session._write = resistant_write
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            if during_cancel:
                await _until(lambda: continuity == [False, True])
                session._set_tools_list_changed_handler(None)
            await asyncio.wait_for(entered.wait(), 5)
            if not during_cancel:
                session._set_tools_list_changed_handler(None)
            await asyncio.wait_for(cancelled.wait(), 5)
            await _until(lambda: session.process.returncode is not None)
            assert session._closed
            assert not finished.is_set()
            release.set()
            await asyncio.wait_for(session.close(), 5)
            assert finished.is_set()
            assert session._tool_subscription.task is None
        finally:
            release.set()
            await session.close()

    asyncio.run(run())


@pytest.mark.parametrize("cancel_call", [False, True])
def test_modern_stdio_subscription_does_not_steal_or_retry_concurrent_call(tmp_path, cancel_call):
    log = tmp_path / "requests"

    async def run():
        session = await _client().connect(_spec(log))
        continuity = []
        changes = []
        call = None
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True])
            call = asyncio.create_task(session.call_tool("search", {"defer": True}))
            await _until(lambda: any(r["method"] == "tools/call" for r in _requests(log)))
            assert not call.done()
            if cancel_call:
                call.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await call
                assert session._closed
                assert continuity[-1] is False
            else:
                await session.read_resource("control://release")
                result = await asyncio.wait_for(call, 5)
                assert result.content[0]["text"] == "settled"
                assert changes == [True]
                assert not session._closed
        finally:
            if call is not None:
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
            await session.close()

    asyncio.run(run())
    assert sum(r["method"] == "tools/call" for r in _requests(log)) == 1


@pytest.mark.parametrize("settle", [False, True])
def test_modern_stdio_subscription_does_not_shorten_an_ordinary_calls_idle_budget(tmp_path, settle):
    log = tmp_path / "requests"

    async def run():
        session = await _client().connect(_spec(log))
        session.transport_limits = McpTransportLimits(idle_timeout_s=0.5, total_call_timeout_s=2)
        continuity = []
        call = None
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: continuity == [False, True])
            await asyncio.sleep(0.35)
            call = asyncio.create_task(session.call_tool("search", {"defer": True}))
            await _until(lambda: any(r["method"] == "tools/call" for r in _requests(log)))
            if settle:
                await asyncio.sleep(0.3)
                assert not call.done()
                await session.read_resource("control://release")
                result = await asyncio.wait_for(call, 2)
                assert result.content[0]["text"] == "settled"
                assert not session._closed
            else:
                with pytest.raises(McpIdleTimeoutError):
                    await asyncio.wait_for(call, 2)
                assert session._closed
        finally:
            if call is not None:
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
            await session.close()

    asyncio.run(run())


def test_modern_stdio_subscription_admits_changes_after_quiet_idle_gap(tmp_path):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        continuity = []
        changes = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: changes.append(True))
            await _until(lambda: continuity == [False, True])
            owner = session._tool_subscription
            session._last_expired_idle_gap = (
                owner.started_at,
                owner.started_at + session.transport_limits.idle_timeout_s + 1,
            )
            owner.consume(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/tools/list_changed",
                    "params": {
                        "_meta": {"io.modelcontextprotocol/subscriptionId": owner.state.request_id}
                    },
                }
            )
            assert changes == [True]
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_stdio_subscription_retired_ids_and_lifetime_bytes_are_bounded(tmp_path):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        session.transport_limits = McpTransportLimits(
            max_message_bytes=1024, max_response_bytes=1024
        )
        continuity = []
        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: continuity == [False, True])
            first = session._tool_subscription.state.request_id
            for _ in range(66):
                previous = session._tool_subscription.state.request_id
                session._set_tools_list_changed_handler(None)
                session._set_tools_list_changed_handler(lambda: None)
                await _until(
                    lambda previous=previous: (
                        session._tool_subscription.state is not None
                        and session._tool_subscription.state.request_id != previous
                        and session._tool_subscription.state.acknowledged
                    )
                )
            owner = session._tool_subscription
            assert len(owner.retired) == 64
            assert first not in owner.retired
            # Aggregate notifications exceed the finite response limit; each
            # frame is bounded independently and no backlog is accumulated.
            await session.read_resource("control://change")
            assert not session._closed
            with pytest.raises(McpProtocolError, match="subscription ID"):
                owner.consume(
                    {
                        "jsonrpc": "2.0",
                        "method": "notifications/tools/list_changed",
                        "params": {"_meta": {"io.modelcontextprotocol/subscriptionId": first}},
                    }
                )
        finally:
            await session.close()

    asyncio.run(run())


def test_modern_stdio_close_settles_calls_before_waiting_for_subscription_cancel(tmp_path):
    log = tmp_path / "requests"

    async def run():
        session = await _client().connect(_spec(log))
        continuity = []
        entered = asyncio.Event()
        release = asyncio.Event()
        original_notify = session._notify
        call = close = None

        async def delayed_notify(method, params):
            if method == "notifications/cancelled":
                entered.set()
                await release.wait()
            await original_notify(method, params)

        try:
            session._set_tools_list_changed_continuity_handler(continuity.append)
            session._set_tools_list_changed_handler(lambda: None)
            await _until(lambda: continuity == [False, True])
            call = asyncio.create_task(session.call_tool("search", {"defer": True}))
            await _until(lambda: any(r["method"] == "tools/call" for r in _requests(log)))
            session._notify = delayed_notify
            close = asyncio.create_task(session.close())
            await asyncio.wait_for(entered.wait(), 5)
            with pytest.raises(McpProtocolError, match="session closed"):
                await asyncio.wait_for(asyncio.shield(call), 0.1)
            assert not close.done()
        finally:
            release.set()
            if close is not None:
                await asyncio.wait_for(close, 5)
            await session.close()
            if call is not None:
                await asyncio.gather(call, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("graceful", [False, True])
def test_modern_stdio_unsupported_filter_cannot_admit_buffered_change(tmp_path, graceful):
    async def run():
        session = await _client().connect(_spec(tmp_path / "requests"))
        changes = []
        try:
            # Drive two already-parsed frames in the same reader turn, before
            # the cancellation owner gets a chance to retire the identity.
            owner = session._tool_subscription
            owner.state = ModernToolSubscription(999)
            owner.started_at = asyncio.get_running_loop().time()
            owner.deadline = owner.started_at + 1
            session._tools_list_changed_handler = lambda: changes.append(True)
            meta = {"io.modelcontextprotocol/subscriptionId": 999}
            owner.consume(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/subscriptions/acknowledged",
                    "params": {"_meta": meta, "notifications": {}},
                }
            )
            if graceful:
                owner.consume(
                    {
                        "jsonrpc": "2.0",
                        "id": 999,
                        "result": {"resultType": "complete", "_meta": meta},
                    }
                )
                assert owner.complete and owner.unsupported
            else:
                with pytest.raises(McpProtocolError, match="terminal filter"):
                    owner.consume(
                        {
                            "jsonrpc": "2.0",
                            "method": "notifications/tools/list_changed",
                            "params": {"_meta": meta},
                        }
                    )
            assert changes == []
        finally:
            await session.close()

    asyncio.run(run())
