from __future__ import annotations

import asyncio
import json
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    DEFAULT_MCP_CLIENT_VERSION,
    MCP_MODERN_PROTOCOL_VERSION,
    McpCallDeadlineExceededError,
    McpIdleTimeoutError,
    McpMessageTooLargeError,
    McpPeerClosedError,
    McpProtocolEra,
    McpProtocolError,
    McpServerSpec,
    McpTransportLimits,
    StdioMcpClient,
    StdioMcpProcessLifetime,
    StdioMcpSession,
    ToolContext,
    connect_mcp_toolset,
)
from cayu.mcp._stdio_process import (
    ContainedStdioMcpProcess,
    stdio_mcp_parent_death_containment_platform_candidate,
)
from cayu.mcp.base import _mcp_session_close_task

_FAKE_SERVER = Path(__file__).resolve().parents[1] / "fixtures" / "fake_mcp_2026_stdio_server.py"


def _server_spec(
    request_log: Path,
    *,
    result_overrides: dict[str, object] | None = None,
) -> McpServerSpec:
    env = {"CAYU_FAKE_MCP_REQUEST_LOG": str(request_log)}
    if result_overrides is not None:
        env["CAYU_FAKE_MCP_RESULT_OVERRIDES"] = json.dumps(result_overrides)
    return McpServerSpec(
        name="modern-stdio",
        command=(sys.executable, str(_FAKE_SERVER)),
        env=env,
    )


def _client(**kwargs: Any) -> StdioMcpClient:
    return StdioMcpClient(
        protocol_era=McpProtocolEra.MODERN_2026_07_28,
        process_lifetime=StdioMcpProcessLifetime.GRACEFUL_CLEANUP,
        **kwargs,
    )


def _requests(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _expected_request_meta() -> dict[str, Any]:
    return {
        "io.modelcontextprotocol/protocolVersion": MCP_MODERN_PROTOCOL_VERSION,
        "io.modelcontextprotocol/clientInfo": {
            "name": "cayu",
            "version": DEFAULT_MCP_CLIENT_VERSION,
        },
        "io.modelcontextprotocol/clientCapabilities": {},
    }


def _assert_cayu_traceback_does_not_retain(error: BaseException, canary: str) -> None:
    assert canary not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                canary not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next


def test_modern_stdio_discovers_lists_calls_resources_and_closes(tmp_path: Path) -> None:
    request_log = tmp_path / "requests.jsonl"

    async def run():
        session = await _client().connect(_server_spec(request_log))
        try:
            metadata = session.initialize_result
            tools = await session.list_tools()
            tool_result = await session.call_tool("search", {"query": "cayu"})
            resources = await session.list_resources()
            resource = await session.read_resource("file://fixture")
            continuity = session._set_tools_list_changed_continuity_handler(lambda _ready: None)
            listener = session._set_tools_list_changed_handler(lambda: None)
            return metadata, tools, tool_result, resources, resource, continuity, listener
        finally:
            await session.close()

    metadata, tools, tool_result, resources, resource, continuity, listener = asyncio.run(run())

    assert metadata.protocol_version == MCP_MODERN_PROTOCOL_VERSION
    assert metadata.server_name == "modern-stdio-fixture"
    assert metadata.server_version == "1.0"
    assert metadata.instructions == "Use the modern stdio fixture carefully."
    assert [tool.name for tool in tools] == ["search"]
    assert tool_result.content == [{"type": "text", "text": "ok"}]
    assert tool_result.structured_content == ["one", 2]
    assert [resource.uri for resource in resources] == ["file://fixture"]
    assert resource.contents == [{"uri": "file://fixture", "text": "hello"}]
    assert (continuity, listener) == (False, False)

    requests = _requests(request_log)
    assert [request["method"] for request in requests] == [
        "server/discover",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/read",
    ]
    expected_meta = _expected_request_meta()
    assert requests[2]["params"] == {
        "name": "search",
        "arguments": {"query": "cayu"},
        "_meta": expected_meta,
    }
    assert all(request["params"]["_meta"] == expected_meta for request in requests)
    assert all(request["method"] != "initialize" for request in requests)
    assert all(request["method"] != "notifications/initialized" for request in requests)


@pytest.mark.parametrize(
    ("method", "result", "operation", "error_match"),
    [
        (
            "server/discover",
            {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "private",
                "supportedVersions": ["2025-06-18"],
                "capabilities": {},
            },
            "connect",
            "does not support pinned protocol version",
        ),
        (
            "server/discover",
            {
                "resultType": "complete",
                "cacheScope": "private",
                "supportedVersions": [MCP_MODERN_PROTOCOL_VERSION],
                "capabilities": {},
            },
            "connect",
            "ttlMs",
        ),
        (
            "tools/list",
            {
                "resultType": "complete",
                "ttlMs": 0,
                "cacheScope": "shared",
                "tools": [],
            },
            "tools/list",
            "cacheScope",
        ),
        (
            "tools/call",
            {"resultType": "input_required", "content": []},
            "tools/call",
            "resultType",
        ),
        (
            "resources/read",
            {
                "resultType": "complete",
                "ttlMs": True,
                "cacheScope": "private",
                "contents": [],
            },
            "resources/read",
            "ttlMs",
        ),
    ],
)
def test_modern_stdio_rejects_invalid_wire_results(
    tmp_path: Path,
    method: str,
    result: dict[str, Any],
    operation: str,
    error_match: str,
) -> None:
    request_log = tmp_path / "requests.jsonl"
    server = _server_spec(request_log, result_overrides={method: result})

    async def run() -> None:
        if operation == "connect":
            await _client().connect(server)
            return
        session = await _client().connect(server)
        try:
            if operation == "tools/list":
                await session.list_tools()
            elif operation == "tools/call":
                await session.call_tool("search", {})
            else:
                await session.read_resource("file://fixture")
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match=error_match):
        asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "operation", "result"),
    [
        (
            "server/discover",
            "connect",
            {
                "ttlMs": 0,
                "cacheScope": "private",
                "supportedVersions": [MCP_MODERN_PROTOCOL_VERSION],
                "capabilities": {},
            },
        ),
        (
            "tools/list",
            "tools/list",
            {"ttlMs": 0, "cacheScope": "private", "tools": []},
        ),
        ("tools/call", "tools/call", {"content": [], "structuredContent": None}),
        (
            "resources/list",
            "resources/list",
            {"ttlMs": 0, "cacheScope": "private", "resources": []},
        ),
        (
            "resources/read",
            "resources/read",
            {"ttlMs": 0, "cacheScope": "private", "contents": []},
        ),
    ],
)
def test_modern_stdio_treats_missing_result_type_as_complete(
    tmp_path: Path,
    method: str,
    operation: str,
    result: dict[str, Any],
) -> None:
    request_log = tmp_path / "requests.jsonl"
    server = _server_spec(request_log, result_overrides={method: result})

    async def run() -> None:
        session = await _client().connect(server)
        try:
            if operation == "tools/list":
                await session.list_tools()
            elif operation == "tools/call":
                await session.call_tool("search", {})
            elif operation == "resources/list":
                await session.list_resources()
            elif operation == "resources/read":
                await session.read_resource("file://fixture")
        finally:
            await session.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    ("method", "result", "operation", "canary"),
    [
        (
            "tools/list",
            {
                "resultType": "complete",
                "ttlMs": "modern-stdio-ttl-secret-canary",
                "cacheScope": "private",
                "tools": [],
            },
            "tools/list",
            "modern-stdio-ttl-secret-canary",
        ),
        (
            "tools/call",
            {
                "resultType": "modern-stdio-result-type-secret-canary",
                "content": [],
            },
            "tools/call",
            "modern-stdio-result-type-secret-canary",
        ),
    ],
)
def test_modern_stdio_invalid_control_fields_do_not_retain_raw_values(
    tmp_path: Path,
    method: str,
    result: dict[str, Any],
    operation: str,
    canary: str,
) -> None:
    request_log = tmp_path / "requests.jsonl"

    async def run() -> BaseException:
        session = await _client().connect(
            _server_spec(request_log, result_overrides={method: result})
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                if operation == "tools/list":
                    await session.list_tools()
                else:
                    await session.call_tool("search", {})
            return exc_info.value
        finally:
            await session.close()

    _assert_cayu_traceback_does_not_retain(asyncio.run(run()), canary)


def test_modern_stdio_connects_through_the_public_toolset_adapter(tmp_path: Path) -> None:
    request_log = tmp_path / "requests.jsonl"

    async def run():
        toolset = await connect_mcp_toolset(
            _server_spec(request_log),
            client=_client(),
        )
        try:
            tool = toolset.tools[0]
            result = await tool.run(
                ToolContext(session_id="modern-stdio", agent_name="assistant"),
                {"query": "cayu"},
            )
            return tool.name, result
        finally:
            await toolset.close()

    name, result = asyncio.run(run())
    assert name == "mcp__modern-stdio__search"
    assert result.structured["mcp_structured_content"] == ["one", 2]


def test_modern_stdio_counts_excluded_tools_toward_wire_limit(tmp_path: Path) -> None:
    request_log = tmp_path / "requests.jsonl"

    async def run() -> None:
        session = await _client(max_list_items=1).connect(_server_spec(request_log))
        try:
            await session.list_tools()
        finally:
            await session.close()

    with pytest.raises(McpProtocolError, match="max_list_items=1"):
        asyncio.run(run())


def test_modern_stdio_catalogue_overflow_drops_raw_traceback_values(tmp_path: Path) -> None:
    canary = "modern-catalogue-overflow-private-value"
    result = {
        "ttlMs": 0,
        "cacheScope": "private",
        "tools": [
            {"name": "first", "description": canary, "inputSchema": {}},
            {"name": "second", "description": canary, "inputSchema": {}},
        ],
    }

    async def run() -> McpProtocolError:
        session = await _client(max_list_items=1).connect(
            _server_spec(tmp_path / "requests.jsonl", result_overrides={"tools/list": result})
        )
        try:
            with pytest.raises(McpProtocolError, match="max_list_items=1") as error:
                await session.list_tools()
            return error.value
        finally:
            await session.close()

    _assert_cayu_traceback_does_not_retain(asyncio.run(run()), canary)


def test_modern_stdio_cancellation_sends_stamped_cancel_then_closes(tmp_path: Path) -> None:
    request_log = tmp_path / "requests.jsonl"

    async def wait_for_request(method: str) -> None:
        for _ in range(200):
            if any(request.get("method") == method for request in _requests(request_log)):
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"timed out waiting for {method}")

    async def run() -> None:
        session = await _client().connect(_server_spec(request_log))
        call = asyncio.create_task(session.call_tool("search", {"defer_response": True}))
        await wait_for_request("tools/call")
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        await session.close()

    asyncio.run(run())
    requests = _requests(request_log)
    assert [request["method"] for request in requests] == [
        "server/discover",
        "tools/call",
        "notifications/cancelled",
    ]
    cancellation = requests[-1]
    assert cancellation["params"]["requestId"] == requests[1]["id"]
    assert cancellation["params"]["_meta"] == _expected_request_meta()


def test_stdio_client_requires_explicit_protocol_era_enum() -> None:
    assert StdioMcpClient().protocol_era is McpProtocolEra.LEGACY
    with pytest.raises(TypeError, match="protocol_era"):
        StdioMcpClient(protocol_era="2026-07-28")  # type: ignore[arg-type]


def test_stdio_client_revalidates_mutated_protocol_era_before_launch() -> None:
    client = StdioMcpClient()
    client.protocol_era = "2026-07-28"  # type: ignore[assignment]

    async def run() -> None:
        await client.connect(McpServerSpec(name="must-not-launch", command=("must-not-launch",)))

    with pytest.raises(TypeError, match="protocol_era"):
        asyncio.run(run())


@pytest.mark.parametrize("era", list(McpProtocolEra))
def test_direct_stdio_session_protocol_era_is_fixed_before_initialization(
    tmp_path: Path, era: McpProtocolEra
) -> None:
    request_log = tmp_path / "requests.jsonl"

    async def run() -> None:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_FAKE_SERVER),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"CAYU_FAKE_MCP_REQUEST_LOG": str(request_log)},
        )
        session = StdioMcpSession(
            server=_server_spec(request_log),
            process=process,
            request_timeout_s=5,
            write_timeout_s=5,
            graceful_shutdown_timeout_s=1,
            cancellation_notification_timeout_s=1,
            client_name="cayu",
            client_version=DEFAULT_MCP_CLIENT_VERSION,
            protocol_era=era,
        )
        try:
            other = (
                McpProtocolEra.LEGACY
                if era is McpProtocolEra.MODERN_2026_07_28
                else McpProtocolEra.MODERN_2026_07_28
            )
            with pytest.raises(AttributeError):
                session.protocol_era = other  # type: ignore[misc]
            assert session.protocol_era is era
            assert _requests(request_log) == []
            if era is McpProtocolEra.MODERN_2026_07_28:
                await session.initialize()
                assert _requests(request_log)[0]["params"]["_meta"] == _expected_request_meta()
        finally:
            await session.close()
        assert process.returncode is not None

    asyncio.run(run())


@pytest.mark.parametrize("method", ["server/discover", "tools/call"])
@pytest.mark.parametrize(
    ("fault", "error_type"),
    [
        ("disconnect", McpPeerClosedError),
        ("stderr-crash", McpPeerClosedError),
        ("malformed-json", McpProtocolError),
        ("oversized-response", McpMessageTooLargeError),
        ("idle", McpIdleTimeoutError),
        ("stream-until-deadline", McpCallDeadlineExceededError),
    ],
)
def test_modern_stdio_failure_reaps_process_and_settles_readers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    fault: str,
    error_type: type[McpProtocolError],
) -> None:
    request_log = tmp_path / "requests.jsonl"
    spec = _server_spec(request_log)
    spec = spec.model_copy(
        update={
            "env": {
                **spec.env,
                "CAYU_FAKE_MCP_FAULT": fault,
                "CAYU_FAKE_MCP_FAULT_METHOD": method,
            }
        }
    )
    sessions: list[StdioMcpSession] = []
    original_initialize = StdioMcpSession.initialize

    async def capture_session(session: StdioMcpSession) -> None:
        sessions.append(session)
        await original_initialize(session)

    monkeypatch.setattr(StdioMcpSession, "initialize", capture_session)

    async def run() -> None:
        # Allow process startup its own generous budget. The shorter call
        # deadline still expires before idle timeout for the streaming peer.
        limits = McpTransportLimits(
            max_message_bytes=2048,
            max_response_bytes=4096,
            idle_timeout_s=2 if fault == "stream-until-deadline" else 1,
            total_call_timeout_s=1 if fault == "stream-until-deadline" else 5,
        )
        try:
            with pytest.raises(error_type) as error:
                session = await _client(transport_limits=limits).connect(spec)
                await session.call_tool("search", {})
            if fault == "stderr-crash":
                assert "fatal: fixture configuration unavailable" in str(error.value)
            assert len(sessions) == 1
            failed = sessions[0]
            cleanup = _mcp_session_close_task(error.value) or failed._close_task
            assert cleanup is not None
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=10)
            await asyncio.wait_for(failed.process.wait(), timeout=5)
            assert failed._closed
            assert not failed._pending
            assert failed._reader_task.done()
            assert failed._stderr_task.done()
            with pytest.raises(McpProtocolError, match="closed"):
                await failed.call_tool("search", {})
        finally:
            for session in sessions:
                await session.close()

    asyncio.run(run())
    requests = _requests(request_log)
    assert requests[0]["method"] == "server/discover"
    assert all(request["params"]["_meta"] == _expected_request_meta() for request in requests)
    assert not any(request["method"] == "initialize" for request in requests)


def test_modern_stdio_outbound_limit_includes_metadata_and_preserves_session(
    tmp_path: Path,
) -> None:
    request_log = tmp_path / "requests.jsonl"

    async def run() -> None:
        session = await _client().connect(_server_spec(request_log))
        try:
            # Arguments fit, but the modern envelope pushes the request over
            # the byte limit. Rejection must happen before anything is written.
            session.transport_limits = McpTransportLimits(
                max_message_bytes=400, max_response_bytes=4096
            )
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("search", {"query": "x" * 250})
            assert [request["method"] for request in _requests(request_log)] == ["server/discover"]
            result = await session.call_tool("search", {})
            assert result.structured_content == ["one", 2]
        finally:
            await session.close()
        assert session.process.returncode is not None

    asyncio.run(run())


@pytest.mark.skipif(
    not stdio_mcp_parent_death_containment_platform_candidate(),
    reason="parent-death containment requires supported Linux enforcement",
)
def test_modern_stdio_default_containment_discovers_and_reaps(tmp_path: Path) -> None:
    async def run() -> None:
        client = StdioMcpClient(protocol_era=McpProtocolEra.MODERN_2026_07_28)
        assert client.process_lifetime is StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT
        session = await client.connect(_server_spec(tmp_path / "requests.jsonl"))
        assert isinstance(session, StdioMcpSession)
        try:
            assert isinstance(session.process, ContainedStdioMcpProcess)
            assert (await session.call_tool("search", {})).structured_content == ["one", 2]
        finally:
            await session.close()
        assert session.process.returncode is not None
        assert session._reader_task.done()
        assert session._stderr_task.done()

    asyncio.run(run())


def test_modern_stdio_interoperates_with_official_sdk(tmp_path: Path) -> None:
    # This is a locked dev dependency, not a silently skipped optional tier.
    assert version("mcp") == "2.1.1"
    request_log = tmp_path / "sdk-requests.jsonl"
    fixture = _FAKE_SERVER.with_name("official_mcp_2026_stdio_server.py")
    spec = McpServerSpec(
        name="official-stdio",
        command=(sys.executable, str(fixture)),
        env={"CAYU_SDK_MCP_REQUEST_LOG": str(request_log)},
    )

    async def run() -> None:
        session = await _client().connect(spec)
        try:
            metadata = session.initialize_result
            assert metadata.protocol_version == MCP_MODERN_PROTOCOL_VERSION
            assert metadata.server_name == "official-stdio-fixture"
            assert metadata.server_version == "1.0"
            tools = await session.list_tools()
            assert [tool.name for tool in tools] == ["search"]
            result = await session.call_tool("search", {"query": "sdk proof"})
            assert result.content == [{"type": "text", "text": "sdk proof"}]
            assert result.structured_content == ["sdk proof", 2, None]
            resources = await session.list_resources()
            assert [resource.uri for resource in resources] == ["fixture://message"]
            resource = await session.read_resource("fixture://message")
            assert resource.contents[0]["text"] == "hello from the official SDK"
        finally:
            await session.close()
        assert session.process.returncode == 0

    asyncio.run(run())
    requests = _requests(request_log)
    assert [request["method"] for request in requests] == [
        "server/discover",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/read",
    ]
    assert all(request["params"]["_meta"] == _expected_request_meta() for request in requests)


def test_modern_stdio_stderr_read_failure_does_not_prevent_cleanup(tmp_path: Path) -> None:
    async def run() -> None:
        session = await _client().connect(_server_spec(tmp_path / "requests.jsonl"))
        assert isinstance(session, StdioMcpSession)
        try:
            assert session.process.stderr is not None
            session.process.stderr.set_exception(OSError("fixture stderr read failure"))
            with pytest.raises(OSError, match="fixture stderr read failure"):
                await session._stderr_task
        finally:
            await asyncio.wait_for(session.close(), timeout=10)
        assert session.process.returncode is not None
        assert session._reader_task.done()
        assert session._stderr_task.done()

    asyncio.run(run())


def test_modern_stdio_discovery_cancellation_reaps_without_legacy_messages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request_log = tmp_path / "requests.jsonl"
    spec = _server_spec(request_log)
    spec = spec.model_copy(
        update={
            "env": {
                **spec.env,
                "CAYU_FAKE_MCP_FAULT": "idle",
                "CAYU_FAKE_MCP_FAULT_METHOD": "server/discover",
            }
        }
    )
    sessions: list[StdioMcpSession] = []
    original_initialize = StdioMcpSession.initialize

    async def capture_session(session: StdioMcpSession) -> None:
        sessions.append(session)
        await original_initialize(session)

    monkeypatch.setattr(StdioMcpSession, "initialize", capture_session)

    async def run() -> None:
        connection = asyncio.create_task(_client().connect(spec))
        try:
            async with asyncio.timeout(10):
                while not _requests(request_log):
                    await asyncio.sleep(0.01)
            connection.cancel()
            with pytest.raises(asyncio.CancelledError) as error:
                await connection
            assert len(sessions) == 1
            cleanup = _mcp_session_close_task(error.value)
            assert cleanup is not None
            await asyncio.wait_for(asyncio.shield(cleanup), timeout=10)
            assert sessions[0].process.returncode is not None
            assert sessions[0]._reader_task.done()
            assert sessions[0]._stderr_task.done()
        finally:
            connection.cancel()
            await asyncio.gather(connection, return_exceptions=True)
            for session in sessions:
                await session.close()

    asyncio.run(run())
    assert [request["method"] for request in _requests(request_log)] == ["server/discover"]
