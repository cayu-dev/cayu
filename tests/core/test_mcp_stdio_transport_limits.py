from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from tests.provider_traceback_assertions import is_cayu_source_filename

from cayu import (
    DEFAULT_MCP_MAX_MESSAGE_BYTES,
    DEFAULT_MCP_MAX_RESPONSE_BYTES,
    HttpMcpClient,
    McpCallDeadlineExceededError,
    McpIdleTimeoutError,
    McpMessageTooLargeError,
    McpPeerClosedError,
    McpProtocolError,
    McpServerSpec,
    McpToolDefinition,
    McpToolset,
    McpTransportLimits,
    SecretRedactor,
    SecretRef,
    StaticVault,
    StdioMcpProcessLifetime,
    StdioMcpSession,
    ToolContext,
)
from cayu import (
    StdioMcpClient as _StdioMcpClient,
)
from cayu.mcp import stdio as mcp_stdio_module
from cayu.mcp import tools as mcp_tools_module
from cayu.mcp._stdio_process import stdio_mcp_parent_death_containment_platform_candidate
from cayu.mcp.base import _mcp_session_close_task
from cayu.mcp.stdio import _StdioPendingTiming
from cayu.vaults import REDACTED_SECRET

_FAKE_SERVER = Path(__file__).parents[1] / "fixtures" / "fake_mcp_server.py"


def StdioMcpClient(*args: Any, **kwargs: Any) -> _StdioMcpClient:
    """Keep transport tests portable without weakening the product default."""

    if (
        "process_lifetime" not in kwargs
        and not stdio_mcp_parent_death_containment_platform_candidate()
    ):
        kwargs["process_lifetime"] = StdioMcpProcessLifetime.GRACEFUL_CLEANUP
    return _StdioMcpClient(*args, **kwargs)


def _server_spec() -> McpServerSpec:
    return McpServerSpec(
        name="bounded-stdio",
        command=[sys.executable, str(_FAKE_SERVER)],
    )


def _limits(
    *,
    max_message_bytes: int = 1_024,
    idle_timeout_s: float = 1.0,
    total_call_timeout_s: float = 2.0,
) -> McpTransportLimits:
    return McpTransportLimits(
        max_message_bytes=max_message_bytes,
        max_response_bytes=max(max_message_bytes, 2_048),
        idle_timeout_s=idle_timeout_s,
        total_call_timeout_s=total_call_timeout_s,
    )


def _deep_json_arguments(depth: int, canary: str) -> dict[str, Any]:
    value: Any = canary
    for _ in range(depth):
        value = [value]
    return {"nested": value}


def _assert_cayu_traceback_does_not_retain(
    error: BaseException,
    *canaries: str,
) -> None:
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            for value in traceback_cursor.tb_frame.f_locals.values():
                rendered = repr(value)
                assert all(canary not in rendered for canary in canaries)
        traceback_cursor = traceback_cursor.tb_next


def test_transport_limits_are_strict_immutable_and_coherent() -> None:
    limits = _limits()

    with pytest.raises(ValidationError, match="frozen"):
        limits.max_message_bytes = 1  # type: ignore[misc]
    with pytest.raises(TypeError, match="max_message_bytes"):
        McpTransportLimits(max_message_bytes=True)
    with pytest.raises(ValueError, match="finite"):
        McpTransportLimits(idle_timeout_s=float("inf"))
    with pytest.raises(ValueError, match="at least"):
        McpTransportLimits(max_message_bytes=2, max_response_bytes=1)


def test_transport_clients_preserve_legacy_defaults_with_shared_byte_limits() -> None:
    stdio_limits = StdioMcpClient().transport_limits
    http_limits = HttpMcpClient().transport_limits

    assert stdio_limits.max_message_bytes == DEFAULT_MCP_MAX_MESSAGE_BYTES == 16 * 1024 * 1024
    assert stdio_limits.max_response_bytes == DEFAULT_MCP_MAX_RESPONSE_BYTES == 64 * 1024 * 1024
    assert (stdio_limits.idle_timeout_s, stdio_limits.total_call_timeout_s) == (30.0, 30.0)
    assert (http_limits.idle_timeout_s, http_limits.total_call_timeout_s) == (120.0, 120.0)


def test_stdio_rejects_conflicting_legacy_and_limits_configuration() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        StdioMcpClient(request_timeout_s=1, transport_limits=_limits())


def test_stdio_legacy_timeout_mutation_updates_idle_and_total_limits() -> None:
    async def run() -> tuple[dict[str, Any] | None, dict[str, Any] | None, tuple[float, float]]:
        session = await StdioMcpClient(request_timeout_s=0.2).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session.request_timeout_s = 0.75
        deferred_call: asyncio.Task[Any] | None = None
        try:
            deferred_call = asyncio.create_task(session.call_tool("echo", {"defer_response": True}))
            await asyncio.sleep(0.25)
            trigger_result = await session.call_tool("echo", {"text": "release"})
            deferred_result = await deferred_call
            return (
                deferred_result.structured_content,
                trigger_result.structured_content,
                (
                    session.transport_limits.idle_timeout_s,
                    session.transport_limits.total_call_timeout_s,
                ),
            )
        finally:
            if deferred_call is not None and not deferred_call.done():
                deferred_call.cancel()
                await asyncio.gather(deferred_call, return_exceptions=True)
            await session.close()

    deferred, trigger, timeouts = asyncio.run(run())

    assert deferred == {"echoed": ""}
    assert trigger == {"echoed": "release"}
    assert timeouts == (0.75, 0.75)


def test_stdio_active_call_keeps_its_legacy_timeout_snapshot() -> None:
    async def run() -> tuple[dict[str, Any] | None, float, tuple[float, float]]:
        session = await StdioMcpClient(request_timeout_s=0.75).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        request_written = asyncio.Event()
        original_write = session._write_with_timeout
        deferred_call: asyncio.Task[Any] | None = None

        async def capture_write(
            payload: dict[str, Any],
            *,
            timeout_message: str,
            call_deadline: float | None = None,
        ) -> None:
            await original_write(
                payload,
                timeout_message=timeout_message,
                call_deadline=call_deadline,
            )
            if payload.get("method") == "tools/call" and payload.get("params", {}).get(
                "arguments", {}
            ).get("defer_response"):
                request_written.set()

        try:
            session._write_with_timeout = capture_write
            started_at = asyncio.get_running_loop().time()
            deferred_call = asyncio.create_task(session.call_tool("echo", {"defer_response": True}))
            await request_written.wait()
            session.request_timeout_s = 0.2
            await asyncio.sleep(0.25)
            await session.call_tool("echo", {"text": "release"})
            deferred_result = await deferred_call
            return (
                deferred_result.structured_content,
                asyncio.get_running_loop().time() - started_at,
                (
                    session.transport_limits.idle_timeout_s,
                    session.transport_limits.total_call_timeout_s,
                ),
            )
        finally:
            if deferred_call is not None and not deferred_call.done():
                deferred_call.cancel()
                await asyncio.gather(deferred_call, return_exceptions=True)
            await session.close()

    result, elapsed, timeouts = asyncio.run(run())

    assert result == {"echoed": ""}
    assert elapsed > 0.2
    assert timeouts == (0.2, 0.2)


@pytest.mark.parametrize("invalid_timeout", [float("nan"), float("inf")])
def test_stdio_rejected_legacy_timeout_mutation_is_atomic(invalid_timeout: float) -> None:
    async def run() -> tuple[float, McpTransportLimits, McpTransportLimits]:
        session = await StdioMcpClient(request_timeout_s=0.75).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        original_limits = session.transport_limits
        try:
            with pytest.raises(ValueError, match="finite"):
                session.request_timeout_s = invalid_timeout
            return session.request_timeout_s, original_limits, session.transport_limits
        finally:
            await session.close()

    timeout_s, original_limits, retained_limits = asyncio.run(run())

    assert timeout_s == 0.75
    assert retained_limits is original_limits


@pytest.mark.parametrize(
    ("method", "list_method", "mapping_attribute"),
    [
        ("tools/list", "list_tools", "_tool_transport_names"),
        ("resources/list", "list_resources", "_resource_transport_uris"),
    ],
)
def test_stdio_later_discovery_failure_clears_partial_page_state(
    method: str,
    list_method: str,
    mapping_attribute: str,
) -> None:
    secret = f"mcp-stdio-partial-{method}-authority-secret"
    page_canary = f"mcp-stdio-partial-{method}-page-canary"
    spec = McpServerSpec(
        name="bounded-stdio-pagination-failure",
        command=[sys.executable, str(_FAKE_SERVER)],
        env={
            "CAYU_FAKE_MCP_PAGINATED_FAILURE_METHOD": method,
            "CAYU_FAKE_MCP_PAGINATED_PAGE_CANARY": page_canary,
        },
        secret_env={
            "CAYU_FAKE_MCP_PAGINATED_PRIVATE_IDENTITY": SecretRef(name="identity"),
        },
    )

    async def run() -> tuple[BaseException, dict[str, str]]:
        session = await StdioMcpClient(secret_resolver=StaticVault({"identity": secret})).connect(
            spec
        )
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await getattr(session, list_method)()
            mapping = dict(getattr(session, mapping_attribute))
            return exc_info.value, mapping
        finally:
            await session.close()

    error, mapping = asyncio.run(run())

    assert mapping == {}
    _assert_cayu_traceback_does_not_retain(error, secret, page_canary)


@pytest.mark.parametrize("response_bytes", [1_024, 1_025])
def test_stdio_message_limit_accepts_exact_boundary_and_rejects_next_byte(
    response_bytes: int,
) -> None:
    async def run() -> tuple[bool, int | None]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        try:
            if response_bytes == 1_024:
                result = await session.call_tool(
                    "echo",
                    {"exact_response_bytes": response_bytes},
                )
                return len(result.content[0]["text"]) > 0, session.process.returncode
            with pytest.raises(McpMessageTooLargeError, match="1024 bytes"):
                await session.call_tool(
                    "echo",
                    {"exact_response_bytes": response_bytes},
                )
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return False, session.process.returncode
        finally:
            await session.close()

    succeeded, returncode = asyncio.run(run())

    assert succeeded is (response_bytes == 1_024)
    if response_bytes > 1_024:
        assert returncode is not None


def test_stdio_fatal_overflow_settles_every_concurrent_waiter_and_closes_process() -> None:
    async def run() -> tuple[list[BaseException], dict[int, Any], int | None]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        try:
            outcomes = await asyncio.gather(
                session.call_tool("echo", {"text": "first", "defer_response": True}),
                session.call_tool("echo", {"exact_response_bytes": 1_025}),
                return_exceptions=True,
            )
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return (
                [item for item in outcomes if isinstance(item, BaseException)],
                dict(session._pending),
                session.process.returncode,
            )
        finally:
            await session.close()

    errors, pending, returncode = asyncio.run(run())

    assert len(errors) == 2
    assert all(isinstance(error, McpMessageTooLargeError) for error in errors)
    assert errors[0] is not errors[1]
    assert errors[0].__traceback__ is not errors[1].__traceback__
    diagnostic_canary = "first-stdio-waiter-diagnostic-only"
    errors[0].add_note(diagnostic_canary)
    assert diagnostic_canary not in "".join(traceback.format_exception(errors[1]))
    assert pending == {}
    assert returncode is not None


def test_stdio_oversized_outbound_message_is_rejected_without_poisoning_session() -> None:
    async def run() -> tuple[bool, bool]:
        session = await StdioMcpClient(transport_limits=_limits(max_message_bytes=512)).connect(
            _server_spec()
        )
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("echo", {"text": "x" * 1_000})
            result = await session.call_tool("echo", {"text": "small"})
            return result.structured_content == {
                "echoed": "small"
            }, session.process.returncode is None
        finally:
            await session.close()

    succeeded, remained_running = asyncio.run(run())

    assert succeeded is True
    assert remained_running is True


def test_stdio_oversized_outbound_is_rejected_before_copy_or_serialization(monkeypatch) -> None:
    def payload_copy_must_not_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("oversized request reached defensive copying")

    async def run() -> bool:
        session = await StdioMcpClient(transport_limits=_limits(max_message_bytes=512)).connect(
            _server_spec()
        )
        assert isinstance(session, StdioMcpSession)
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    mcp_stdio_module,
                    "jsonrpc_request_payload",
                    payload_copy_must_not_run,
                )
                with pytest.raises(McpMessageTooLargeError):
                    await session.call_tool("echo", {"text": "x" * 1_000})
            result = await session.call_tool("echo", {"text": "small"})
            return result.structured_content == {"echoed": "small"}
        finally:
            await session.close()

    assert asyncio.run(run()) is True


def test_stdio_adapter_outbound_overflow_drops_known_secret_from_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-outbound-overflow-secret-canary"

    async def run() -> BaseException:
        session = await StdioMcpClient(
            transport_limits=_limits(max_message_bytes=512),
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
        toolset = McpToolset(
            server=session.server,
            session=session,
            definitions=(definition,),
        )
        try:
            with pytest.raises(McpMessageTooLargeError) as exc_info:
                await toolset.tools[0].run(
                    ToolContext(session_id="outbound-overflow", agent_name="test"),
                    {"text": (secret + "-") * 20},
                )
            return exc_info.value
        finally:
            await toolset.close()

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_adapter_rejects_oversized_arguments_before_defensive_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> bool:
        session = await StdioMcpClient(
            transport_limits=_limits(max_message_bytes=512),
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
        toolset = McpToolset(server=session.server, session=session, definitions=(definition,))
        original_copy = mcp_stdio_module.copy_json_value

        def copied_request_must_fit(value: Any, field_name: str) -> Any:
            if field_name == "params":
                raise AssertionError("oversized adapter arguments reached defensive copying")
            return original_copy(value, field_name)

        def adapter_must_delegate_nesting_preflight(_value: Any) -> bool:
            raise AssertionError("built-in adapter performed an unbounded nesting scan")

        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(mcp_stdio_module, "copy_json_value", copied_request_must_fit)
                patcher.setattr(mcp_tools_module, "copy_json_value", copied_request_must_fit)
                patcher.setattr(
                    mcp_tools_module,
                    "mcp_json_value_nesting_too_deep",
                    adapter_must_delegate_nesting_preflight,
                )
                with pytest.raises(McpMessageTooLargeError):
                    await toolset.tools[0].run(
                        ToolContext(session_id="adapter-overflow", agent_name="test"),
                        {"values": ["x" * 16 for _ in range(100)]},
                    )
            request_id = session._next_id
            payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": "echo", "arguments": {"padding": ""}},
            }
            encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            exact_arguments = {"padding": "x" * (512 - len(encoded))}
            result = await toolset.tools[0].run(
                ToolContext(session_id="adapter-boundary", agent_name="test"),
                exact_arguments,
            )
            return result.structured["mcp_structured_content"] == {"echoed": ""}
        finally:
            await toolset.close()

    assert asyncio.run(run()) is True


@pytest.mark.parametrize("through_adapter", [False, True])
def test_stdio_invalid_outbound_arguments_are_typed_and_redacted(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
    through_adapter: bool,
) -> None:
    secret = "mcp-stdio-invalid-outbound-secret-canary"

    async def run() -> tuple[BaseException, bool]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        definition = McpToolDefinition(name="echo", input_schema={"type": "object"})
        toolset = McpToolset(server=session.server, session=session, definitions=(definition,))
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                if through_adapter:
                    await toolset.tools[0].run(
                        ToolContext(session_id="invalid-outbound", agent_name="test"),
                        {secret: object()},
                    )
                else:
                    await session.call_tool("echo", {secret: object()})
            if through_adapter:
                result = await toolset.tools[0].run(
                    ToolContext(session_id="valid-outbound", agent_name="test"),
                    {"text": "still-open"},
                )
                reused = result.structured["mcp_structured_content"] == {"echoed": "still-open"}
            else:
                direct_result = await session.call_tool("echo", {"text": "still-open"})
                reused = direct_result.structured_content == {"echoed": "still-open"}
            return exc_info.value, reused
        finally:
            await toolset.close()

    with caplog.at_level(logging.DEBUG):
        error, reused = asyncio.run(run())

    assert reused is True
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_pre_dispatch_deadline_scrubs_private_request_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-pre-dispatch-deadline-secret-canary"
    original_preflight = mcp_stdio_module.mcp_jsonrpc_request_preflight

    def delayed_preflight(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original_preflight(*args, **kwargs)

    async def run() -> tuple[BaseException, bool, int | None]:
        session = await StdioMcpClient(
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=1),
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session.transport_limits = _limits(idle_timeout_s=0.5, total_call_timeout_s=0.02)
        session._secret_redactor = SecretRedactor(secret)
        session._tool_transport_names = {"echo": secret}
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    mcp_stdio_module,
                    "mcp_jsonrpc_request_preflight",
                    delayed_preflight,
                )
                with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                    await session.call_tool("echo", {})
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return exc_info.value, session._closed, session.process.returncode
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, session_closed, returncode = asyncio.run(run())

    assert session_closed is True
    assert returncode is not None
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_large_integer_preflight_returns_typed_overflow_without_dispatch() -> None:
    async def run() -> bool:
        session = await StdioMcpClient(transport_limits=_limits(max_message_bytes=512)).connect(
            _server_spec()
        )
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpMessageTooLargeError):
                await session.call_tool("echo", {"value": 10**5_000})
            result = await session.call_tool("echo", {"text": "still-open"})
            return result.structured_content == {"echoed": "still-open"}
        finally:
            await session.close()

    assert asyncio.run(run()) is True


def test_stdio_invalid_circular_arguments_keep_validation_error_and_session_reusable() -> None:
    async def run() -> bool:
        session = await StdioMcpClient(transport_limits=_limits(max_message_bytes=512)).connect(
            _server_spec()
        )
        assert isinstance(session, StdioMcpSession)
        arguments: dict[str, Any] = {}
        arguments["self"] = arguments
        try:
            with pytest.raises(McpProtocolError, match="circular references"):
                await session.call_tool("echo", arguments)
            result = await session.call_tool("echo", {"text": "still-open"})
            return result.structured_content == {"echoed": "still-open"}
        finally:
            arguments.clear()
            await session.close()

    assert asyncio.run(run()) is True


@pytest.mark.parametrize(
    ("max_message_bytes", "error_type"),
    [
        (256, McpMessageTooLargeError),
        (8_192, McpProtocolError),
    ],
)
def test_stdio_deep_outbound_is_safely_rejected_without_dispatch_and_session_reuses(
    max_message_bytes: int,
    error_type: type[McpProtocolError],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = f"mcp-stdio-deep-{max_message_bytes}-secret-canary"

    async def run() -> tuple[BaseException, bool, bool, int]:
        session = await StdioMcpClient(
            transport_limits=_limits(max_message_bytes=max_message_bytes),
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        arguments = _deep_json_arguments(1_500, secret)
        write_count = 0
        original_write = session._write_with_timeout

        async def count_write(*args: Any, **kwargs: Any) -> None:
            nonlocal write_count
            write_count += 1
            await original_write(*args, **kwargs)

        session._write_with_timeout = count_write
        try:
            with pytest.raises(error_type) as exc_info:
                await session.call_tool("echo", arguments)
            result = await session.call_tool("echo", {"text": "small"})
            return (
                exc_info.value,
                result.structured_content == {"echoed": "small"},
                session.process.returncode is None,
                write_count,
            )
        finally:
            arguments.clear()
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, succeeded, remained_running, write_count = asyncio.run(run())

    assert succeeded is True
    assert remained_running is True
    assert write_count == 1
    assert secret not in "".join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert error.__context__ is None
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize("depth", [300, 1_500])
def test_stdio_deep_inbound_is_typed_secret_safe_and_settles_every_waiter(
    depth: int,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = f"mcp-stdio-deep-inbound-{depth}-secret-canary"

    async def run() -> tuple[list[BaseException], dict[int, Any], bool, int | None]:
        session = await StdioMcpClient(
            transport_limits=_limits(max_message_bytes=8_192),
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        try:
            outcomes = await asyncio.gather(
                session.call_tool("echo", {"text": "first", "defer_response": True}),
                session.call_tool(
                    "echo",
                    {
                        "deep_response_depth": depth,
                        "deep_response_canary": secret,
                    },
                ),
                return_exceptions=True,
            )
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return (
                [outcome for outcome in outcomes if isinstance(outcome, BaseException)],
                dict(session._pending),
                session._closed,
                session.process.returncode,
            )
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        errors, pending, session_closed, returncode = asyncio.run(run())

    assert len(errors) == 2
    assert all(isinstance(error, McpProtocolError) for error in errors)
    assert all("supported JSON nesting" in str(error) for error in errors)
    assert errors[0] is not errors[1]
    assert pending == {}
    assert session_closed is True
    assert returncode is not None
    for error in errors:
        assert error.__cause__ is None
        assert error.__context__ is None
        assert secret not in "".join(traceback.format_exception(error))
        _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_idle_timeout_is_typed_and_fences_uncertain_process() -> None:
    async def run() -> tuple[dict[int, Any], bool, int | None]:
        session = await StdioMcpClient(
            transport_limits=_limits(idle_timeout_s=0.1, total_call_timeout_s=0.5)
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpIdleTimeoutError, match="idle timeout"):
                await session.call_tool("echo", {"text": "idle", "defer_response": True})
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("echo", {"text": "after-idle"})
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return (
                dict(session._pending),
                session._closed,
                session.process.returncode,
            )
        finally:
            await session.close()

    pending, session_closed, returncode = asyncio.run(run())

    assert pending == {}
    assert session_closed is True
    assert returncode is not None


def test_stdio_active_byte_stream_still_hits_total_call_deadline() -> None:
    async def run() -> None:
        session = await StdioMcpClient(
            transport_limits=_limits(idle_timeout_s=0.1, total_call_timeout_s=0.15)
        ).connect(_server_spec())
        try:
            with pytest.raises(McpCallDeadlineExceededError, match="timed out"):
                await session.call_tool(
                    "echo",
                    {"exact_response_bytes": 1_024, "slow_response_delay_s": 0.02},
                )
        finally:
            await session.close()

    asyncio.run(run())


def test_stdio_total_deadline_remains_authoritative_during_response_processing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_redact = mcp_stdio_module.safely_redact_jsonrpc_response

    def delayed_redaction(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.04)
        return original_redact(*args, **kwargs)

    async def run() -> tuple[bool, int | None]:
        session = await StdioMcpClient(
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.5)
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session.transport_limits = _limits(
            idle_timeout_s=0.5,
            total_call_timeout_s=0.02,
        )
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(
                    mcp_stdio_module,
                    "safely_redact_jsonrpc_response",
                    delayed_redaction,
                )
                with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                    await session.call_tool("echo", {})
            cleanup_task = _mcp_session_close_task(exc_info.value)
            assert cleanup_task is not None
            await cleanup_task
            return session._closed, session.process.returncode
        finally:
            await session.close()

    session_closed, returncode = asyncio.run(run())

    assert session_closed is True
    assert returncode is not None


def test_stdio_total_deadline_includes_tool_result_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parser = mcp_stdio_module.tool_result_from_payload

    def delayed_parser(value: object) -> Any:
        time.sleep(0.04)
        return original_parser(value)

    async def run() -> tuple[bool, int | None]:
        session = await StdioMcpClient(
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.5)
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session.transport_limits = _limits(
            idle_timeout_s=0.5,
            total_call_timeout_s=0.02,
        )
        try:
            with monkeypatch.context() as patcher:
                patcher.setattr(mcp_stdio_module, "tool_result_from_payload", delayed_parser)
                with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                    await session.call_tool("echo", {})
            cleanup_task = _mcp_session_close_task(exc_info.value)
            assert cleanup_task is not None
            await cleanup_task
            return session._closed, session.process.returncode
        finally:
            await session.close()

    session_closed, returncode = asyncio.run(run())

    assert session_closed is True
    assert returncode is not None


def test_stdio_completed_response_cannot_win_after_absolute_deadline() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        future.set_result({})
        session = object.__new__(StdioMcpSession)
        session.transport_limits = _limits(
            idle_timeout_s=10,
            total_call_timeout_s=1,
        )
        session._last_read_activity = 100.0
        session._pending_timing = {}
        real_time = loop.time
        calls = 0

        def boundary_time() -> float:
            nonlocal calls
            calls += 1
            return 100.0 if calls == 1 else 102.0

        loop.time = boundary_time  # type: ignore[method-assign]
        try:
            with pytest.raises(McpCallDeadlineExceededError, match="total call deadline"):
                await session._wait_for_response(
                    future,
                    request_id=1,
                    started_at=100.0,
                    call_deadline=101.0,
                    idle_timeout_s=10.0,
                )
        finally:
            loop.time = real_time  # type: ignore[method-assign]

    asyncio.run(run())


def test_stdio_response_arriving_after_idle_deadline_cannot_win_wake_race() -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        settled_at = loop.time()
        started_at = settled_at - 2
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        future.set_result({})
        session = object.__new__(StdioMcpSession)
        session.transport_limits = _limits(
            idle_timeout_s=1,
            total_call_timeout_s=10,
        )
        session._last_read_activity = settled_at
        session._pending_timing = {
            1: _StdioPendingTiming(
                settled_at=settled_at,
                last_read_activity=settled_at,
                expired_idle_gap=(started_at, settled_at),
                response_received=True,
            )
        }

        with pytest.raises(McpIdleTimeoutError, match="idle timeout"):
            await session._wait_for_response(
                future,
                request_id=1,
                started_at=started_at,
                call_deadline=settled_at + 5,
                idle_timeout_s=1.0,
            )

    asyncio.run(run())


def test_stdio_expired_idle_gap_remains_authoritative_after_unrelated_activity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        session = object.__new__(StdioMcpSession)
        session.transport_limits = _limits(idle_timeout_s=1, total_call_timeout_s=10)
        session._last_read_activity = started_at
        session._last_expired_idle_gap = None
        session._pending_timing = {}
        wait_calls = 0

        async def record_late_activity(*args, **kwargs):
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls > 1:
                raise AssertionError("an expired idle gap was incorrectly reset")
            activity_resumed_at = started_at + 2
            session._last_read_activity = activity_resumed_at
            session._last_expired_idle_gap = (started_at, activity_resumed_at)
            return set(), set()

        monkeypatch.setattr(asyncio, "wait", record_late_activity)
        with pytest.raises(McpIdleTimeoutError, match="idle timeout"):
            await session._wait_for_response(
                future,
                request_id=1,
                started_at=started_at,
                call_deadline=started_at + 10,
                idle_timeout_s=1.0,
            )
        assert wait_calls == 1

    asyncio.run(run())


def test_stdio_initialization_deadline_does_not_wait_for_process_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delayed_server = _server_spec().model_copy(
        update={"env": {"CAYU_FAKE_MCP_INITIALIZE_DELAY_S": "0.05"}}
    )

    async def run() -> bool:
        close_started = asyncio.Event()
        close_finished = asyncio.Event()
        release_close = asyncio.Event()
        original_close = StdioMcpSession.close

        async def blocking_close(session: StdioMcpSession) -> None:
            close_started.set()
            await release_close.wait()
            await original_close(session)
            close_finished.set()

        monkeypatch.setattr(StdioMcpSession, "close", blocking_close)
        client = StdioMcpClient(
            transport_limits=_limits(idle_timeout_s=0.5, total_call_timeout_s=0.01),
            graceful_shutdown_timeout_s=0.1,
        )
        connect_task = asyncio.create_task(client.connect(delayed_server))
        await asyncio.wait_for(close_started.wait(), timeout=1.0)
        done, _ = await asyncio.wait({connect_task}, timeout=0.05)
        returned_before_close = connect_task in done and not close_finished.is_set()
        release_close.set()
        with pytest.raises(McpCallDeadlineExceededError, match="total call deadline"):
            await connect_task
        await asyncio.wait_for(close_finished.wait(), timeout=1.0)
        return returned_before_close

    assert asyncio.run(run()) is True


def test_stdio_initialize_failure_attaches_redacted_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "mcp-stdio-initialize-cleanup-secret-canary"
    captured_session: StdioMcpSession | None = None
    original_close = StdioMcpSession.close

    async def failing_initialize(session: StdioMcpSession) -> None:
        nonlocal captured_session
        captured_session = session
        session._secret_redactor = SecretRedactor(secret)
        raise McpProtocolError("invalid initialization response")

    async def failing_close(session: StdioMcpSession) -> None:
        raise RuntimeError(f"initialization cleanup exposed {secret}")

    monkeypatch.setattr(StdioMcpSession, "initialize", failing_initialize)
    monkeypatch.setattr(StdioMcpSession, "close", failing_close)

    async def run() -> BaseException:
        try:
            with pytest.raises(McpProtocolError, match="invalid initialization") as exc_info:
                await StdioMcpClient().connect(_server_spec())
            return exc_info.value
        finally:
            if captured_session is not None:
                await original_close(captured_session)

    error = asyncio.run(run())

    assert isinstance(error.__cause__, McpProtocolError)
    assert secret not in "".join(traceback.format_exception(error))


def test_stdio_total_call_deadline_includes_request_write() -> None:
    async def run() -> tuple[bool, bool, int | None]:
        session = await StdioMcpClient(request_timeout_s=1).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session.request_timeout_s = 0.02
        write_started = asyncio.Event()

        async def block_write(payload: dict[str, Any]) -> None:
            write_started.set()
            await asyncio.Event().wait()

        try:
            session._write = block_write
            with pytest.raises(McpCallDeadlineExceededError, match="total call deadline"):
                await session.call_tool("echo", {})
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return write_started.is_set(), session._closed, session.process.returncode
        finally:
            await session.close()

    write_started, session_closed, returncode = asyncio.run(run())

    assert write_started is True
    assert session_closed is True
    assert returncode is not None


def test_stdio_completed_response_deadline_redacts_private_payload(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-completed-response-deadline-secret"
    spec = _server_spec().model_copy(
        update={"secret_env": {"CAYU_FAKE_MCP_LIMIT_CANARY": SecretRef(name="token")}}
    )

    async def run() -> BaseException:
        session = await StdioMcpClient(
            transport_limits=_limits(total_call_timeout_s=1),
            secret_resolver=StaticVault({"token": secret}),
        ).connect(spec)
        assert isinstance(session, StdioMcpSession)
        session.transport_limits = _limits(
            idle_timeout_s=1,
            total_call_timeout_s=0.2,
        )
        original_handle_message = session._handle_message

        async def delay_completed_response(message: dict[str, Any]) -> None:
            if message.get("id") == 2:
                # Queue the blocking callback before set_result() wakes the request
                # waiter, reproducing a real response/deadline scheduling race.
                asyncio.get_running_loop().call_soon(time.sleep, 0.25)
            await original_handle_message(message)

        session._handle_message = delay_completed_response
        try:
            with pytest.raises(McpCallDeadlineExceededError) as exc_info:
                await session.call_tool(
                    "echo",
                    {
                        "exact_response_bytes": 512,
                        "include_limit_canary": True,
                    },
                )
            cleanup_task = _mcp_session_close_task(exc_info.value)
            assert cleanup_task is not None
            await cleanup_task
            return exc_info.value
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        if is_cayu_source_filename(traceback_cursor.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in traceback_cursor.tb_frame.f_locals.values()
            )
        traceback_cursor = traceback_cursor.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


@pytest.mark.parametrize(
    ("call_deadline", "write_timeout_s", "error_type", "message"),
    [
        (101.0, 10.0, McpCallDeadlineExceededError, "total call deadline"),
        (110.0, 1.0, TimeoutError, "write timed out"),
    ],
)
def test_stdio_completed_write_cannot_win_after_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
    call_deadline: float,
    write_timeout_s: float,
    error_type: type[BaseException],
    message: str,
) -> None:
    async def run() -> None:
        class BoundaryClock:
            calls = 0

            def time(self) -> float:
                self.calls += 1
                return 100.0 if self.calls == 1 else 102.0

        session = object.__new__(StdioMcpSession)
        session.write_timeout_s = write_timeout_s
        captured: list[tuple[asyncio.Task[None], BaseException]] = []

        async def completed_write(payload: dict[str, Any]) -> None:
            payload.clear()

        async def wait_must_not_run_after_checkpoint_deadline(
            tasks: set[asyncio.Task[None]],
            *,
            timeout: float,
        ) -> tuple[set[asyncio.Task[None]], set[asyncio.Task[None]]]:
            del tasks, timeout
            raise AssertionError("expired write deadline reached asyncio.wait")

        def capture_cleanup(
            task: asyncio.Task[None],
            *,
            primary_error: BaseException,
        ) -> None:
            captured.append((task, primary_error))

        session._write = completed_write
        session._schedule_interrupted_write_cleanup = capture_cleanup
        monkeypatch.setattr(mcp_stdio_module.asyncio, "get_running_loop", lambda: BoundaryClock())
        monkeypatch.setattr(
            mcp_stdio_module.asyncio,
            "wait",
            wait_must_not_run_after_checkpoint_deadline,
        )

        with pytest.raises(error_type, match=message) as exc_info:
            await session._write_with_timeout(
                {},
                timeout_message="MCP test write timed out.",
                call_deadline=call_deadline,
            )
        assert len(captured) == 1
        assert captured[0][0].done() is True
        assert captured[0][1] is exc_info.value

    asyncio.run(run())


def test_stdio_total_deadline_does_not_wait_for_cancellation_resistant_writer() -> None:
    async def run() -> tuple[bool, bool, bool]:
        session = await StdioMcpClient(request_timeout_s=1).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session.request_timeout_s = 0.02
        write_started = asyncio.Event()
        writer_cancelled = asyncio.Event()
        release_writer = asyncio.Event()
        call_task: asyncio.Task[Any] | None = None

        async def cancellation_resistant_write(payload: dict[str, Any]) -> None:
            del payload
            write_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                writer_cancelled.set()
                await release_writer.wait()

        try:
            session._write = cancellation_resistant_write
            call_task = asyncio.create_task(session.call_tool("echo", {}))
            await write_started.wait()
            done, _ = await asyncio.wait({call_task}, timeout=0.1)
            returned_before_writer = call_task in done and not release_writer.is_set()
            with pytest.raises(McpCallDeadlineExceededError):
                await call_task
            cleanup_task = _mcp_session_close_task(call_task.exception())
            assert cleanup_task is not None
            cleanup_retained_writer = not cleanup_task.done()
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("echo", {"text": "competing"})
            release_writer.set()
            await asyncio.wait_for(cleanup_task, timeout=0.2)
            return returned_before_writer, writer_cancelled.is_set(), cleanup_retained_writer
        finally:
            release_writer.set()
            if call_task is not None and not call_task.done():
                call_task.cancel()
                await asyncio.gather(call_task, return_exceptions=True)
            await session.close()

    returned, writer_cancelled, retained = asyncio.run(run())

    assert returned is True
    assert writer_cancelled is True
    assert retained is True


def test_stdio_caller_cancellation_does_not_wait_for_resistant_writer() -> None:
    async def run() -> tuple[int, bool, bool, bool]:
        session = await StdioMcpClient(request_timeout_s=1).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        write_started = asyncio.Event()
        writer_cancelled = asyncio.Event()
        release_writer = asyncio.Event()
        call_task: asyncio.Task[Any] | None = None

        async def cancellation_resistant_write(payload: dict[str, Any]) -> None:
            del payload
            write_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                writer_cancelled.set()
                await release_writer.wait()

        try:
            session._write = cancellation_resistant_write
            call_task = asyncio.create_task(session.call_tool("echo", {}))
            await write_started.wait()
            call_task.cancel("cancel blocked stdio write")
            cancelling = call_task.cancelling()
            done, _ = await asyncio.wait({call_task}, timeout=0.05)
            returned_before_writer = call_task in done and not release_writer.is_set()
            with pytest.raises(
                asyncio.CancelledError, match="cancel blocked stdio write"
            ) as exc_info:
                await call_task
            cleanup_task = _mcp_session_close_task(exc_info.value)
            assert cleanup_task is not None
            cleanup_retained_writer = not cleanup_task.done()
            with pytest.raises(McpProtocolError, match="closed"):
                await session.call_tool("echo", {"text": "competing"})
            release_writer.set()
            await asyncio.wait_for(cleanup_task, timeout=0.2)
            return (
                cancelling,
                call_task.cancelled(),
                returned_before_writer,
                cleanup_retained_writer,
            )
        finally:
            release_writer.set()
            if call_task is not None and not call_task.done():
                call_task.cancel()
                await asyncio.gather(call_task, return_exceptions=True)
            await session.close()

    cancelling, cancelled, returned, retained = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert returned is True
    assert retained is True


def test_stdio_real_task_cancellation_preserves_cancelled_state_and_cleanup() -> None:
    class HostileCancellationDetail:
        render_calls = 0

        def __str__(self) -> str:
            self.render_calls += 1
            return "stdio cancellation detail must not render"

    cancellation_detail = HostileCancellationDetail()

    async def run() -> tuple[int, bool, dict[int, Any], bool, BaseException]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        request_written = asyncio.Event()
        original_write = session._write_with_timeout

        async def capture_write(
            payload: dict[str, Any],
            *,
            timeout_message: str,
            call_deadline: float | None = None,
        ) -> None:
            await original_write(
                payload,
                timeout_message=timeout_message,
                call_deadline=call_deadline,
            )
            if payload.get("method") == "tools/call":
                request_written.set()

        try:
            session._write_with_timeout = capture_write
            task = asyncio.create_task(
                session.call_tool("echo", {"text": "cancel", "defer_response": True})
            )
            await request_written.wait()
            task.cancel(cancellation_detail)
            cancelling = task.cancelling()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            cleanup_task = _mcp_session_close_task(exc_info.value)
            assert cleanup_task is not None
            await cleanup_task
            return (
                cancelling,
                task.cancelled(),
                dict(session._pending),
                session._closed,
                exc_info.value,
            )
        finally:
            await session.close()

    cancelling, cancelled, pending, session_closed, error = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert pending == {}
    assert session_closed is True
    assert cancellation_detail.render_calls == 0
    assert error.args == ("MCP operation cancelled",)


def test_stdio_real_task_cancellation_redacts_numeric_secret_argument(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "4829017351642089"

    async def run() -> tuple[int, bool, dict[int, Any], bool, BaseException]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        request_written = asyncio.Event()
        original_write = session._write_with_timeout

        async def capture_write(
            payload: dict[str, Any],
            *,
            timeout_message: str,
            call_deadline: float | None = None,
        ) -> None:
            await original_write(
                payload,
                timeout_message=timeout_message,
                call_deadline=call_deadline,
            )
            if payload.get("method") == "tools/call":
                request_written.set()

        try:
            session._write_with_timeout = capture_write
            task = asyncio.create_task(
                session.call_tool("echo", {"text": "cancel", "defer_response": True})
            )
            await request_written.wait()
            task.cancel(int(secret))
            cancelling = task.cancelling()
            with pytest.raises(asyncio.CancelledError) as exc_info:
                await task
            cleanup_task = _mcp_session_close_task(exc_info.value)
            assert cleanup_task is not None
            await cleanup_task
            return (
                cancelling,
                task.cancelled(),
                dict(session._pending),
                session._closed,
                exc_info.value,
            )
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        cancelling, cancelled, pending, session_closed, error = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert pending == {}
    assert session_closed is True
    assert error.args == (REDACTED_SECRET,)
    assert secret not in repr(error)
    assert secret not in "".join(traceback.format_exception(error))
    _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_preserves_cancellation_pending_before_request_dispatch() -> None:
    async def cancelled_call(session: StdioMcpSession) -> None:
        current = asyncio.current_task()
        assert current is not None
        current.cancel("stdio cancellation pending before dispatch")
        await session.call_tool("echo", {"text": "must-not-dispatch"})

    async def run() -> tuple[int, bool, dict[int, Any], bool, dict[str, Any] | None]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        try:
            task = asyncio.create_task(cancelled_call(session))
            with pytest.raises(
                asyncio.CancelledError,
                match="stdio cancellation pending before dispatch",
            ):
                await task
            followup = await session.call_tool("echo", {"text": "still reusable"})
            return (
                task.cancelling(),
                task.cancelled(),
                dict(session._pending),
                session._closed,
                followup.structured_content,
            )
        finally:
            await session.close()

    cancelling, cancelled, pending, session_closed, followup = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert pending == {}
    assert session_closed is False
    assert followup == {"echoed": "still reusable"}


def test_stdio_cancelling_one_inflight_request_settles_every_shared_waiter() -> None:
    async def run() -> tuple[BaseException, BaseException, int | None]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        writes = 0
        both_written = asyncio.Event()
        original_write = session._write_with_timeout

        async def capture_write(
            payload: dict[str, Any],
            *,
            timeout_message: str,
            call_deadline: float | None = None,
        ) -> None:
            nonlocal writes
            await original_write(
                payload,
                timeout_message=timeout_message,
                call_deadline=call_deadline,
            )
            if payload.get("method") == "tools/call":
                writes += 1
                if writes == 2:
                    both_written.set()

        try:
            session._write_with_timeout = capture_write
            first = asyncio.create_task(
                session.call_tool("echo", {"text": "first", "defer_response": True})
            )
            second = asyncio.create_task(
                session.call_tool("echo", {"text": "second", "defer_response": True})
            )
            await both_written.wait()
            first.cancel()
            outcomes = await asyncio.gather(first, second, return_exceptions=True)
            await asyncio.wait_for(session.process.wait(), timeout=1)
            assert isinstance(outcomes[0], BaseException)
            assert isinstance(outcomes[1], BaseException)
            return outcomes[0], outcomes[1], session.process.returncode
        finally:
            await session.close()

    first_error, second_error, returncode = asyncio.run(run())

    assert isinstance(first_error, asyncio.CancelledError)
    assert isinstance(second_error, McpProtocolError)
    assert returncode is not None


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        ({"invalid_utf8": True}, McpProtocolError, "invalid UTF-8"),
        ({"close_stdout": True}, McpPeerClosedError, "closed stdout"),
    ],
)
def test_stdio_malformed_or_closed_peer_is_fatal(
    arguments: dict[str, Any],
    error_type: type[BaseException],
    message: str,
) -> None:
    async def run() -> int | None:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(error_type, match=message):
                await session.call_tool("echo", arguments)
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return session.process.returncode
        finally:
            await session.close()

    assert asyncio.run(run()) is not None


def test_stdio_peer_closure_precedes_optional_stderr_enrichment_deadline() -> None:
    async def run() -> tuple[BaseException, int | None]:
        session = await StdioMcpClient(
            transport_limits=_limits(
                idle_timeout_s=0.5,
                total_call_timeout_s=0.1,
            )
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpPeerClosedError) as exc_info:
                await session.call_tool(
                    "echo",
                    {"close_stdout_keep_stderr_s": 0.3},
                )
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return exc_info.value, session.process.returncode
        finally:
            await session.close()

    error, returncode = asyncio.run(run())

    assert "closed stdout" in str(error)
    assert returncode is not None


@pytest.mark.skipif(
    not stdio_mcp_parent_death_containment_platform_candidate(),
    reason="parent-death containment requires supported Linux process controls",
)
def test_contained_stdio_peer_closure_is_not_masked_by_anchor_ownership() -> None:
    async def run() -> tuple[BaseException, int | None]:
        session = await _StdioMcpClient(
            process_lifetime=StdioMcpProcessLifetime.PARENT_DEATH_CONTAINMENT,
            transport_limits=_limits(
                idle_timeout_s=0.5,
                total_call_timeout_s=0.1,
            ),
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        try:
            with pytest.raises(McpPeerClosedError) as exc_info:
                await session.call_tool(
                    "echo",
                    {"close_stdout_keep_stderr_s": 0.3},
                )
            await asyncio.wait_for(session.process.wait(), timeout=1)
            return exc_info.value, session.process.returncode
        finally:
            await session.close()

    error, returncode = asyncio.run(run())

    assert "closed stdout" in str(error)
    assert returncode is not None


def test_stdio_broken_pipe_detaches_raw_failure_and_clears_serialized_request() -> None:
    secret = "mcp-stdio-broken-pipe-secret-canary"
    raw_failure = BrokenPipeError("peer closed")

    class BrokenPipeWriter:
        def __init__(self, underlying) -> None:
            self.underlying = underlying

        def write(self, data: bytes) -> None:
            raise raw_failure

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.underlying.close()

        async def wait_closed(self) -> None:
            await self.underlying.wait_closed()

    async def run() -> McpPeerClosedError:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        assert session.process.stdin is not None
        session.process.stdin = BrokenPipeWriter(session.process.stdin)  # type: ignore[assignment]
        session._stderr_task.cancel()
        await asyncio.gather(session._stderr_task, return_exceptions=True)
        try:
            with pytest.raises(McpPeerClosedError) as exc_info:
                await session.call_tool("echo", {"text": secret})
            return exc_info.value
        finally:
            await session.close()

    error = asyncio.run(run())

    assert error.__cause__ is None
    assert error.__context__ is None
    raw_traceback = raw_failure.__traceback__
    assert raw_traceback is not None
    while raw_traceback is not None:
        if is_cayu_source_filename(raw_traceback.tb_frame.f_code.co_filename):
            assert all(
                secret not in repr(value) for value in raw_traceback.tb_frame.f_locals.values()
            )
        raw_traceback = raw_traceback.tb_next


@pytest.mark.parametrize("write_error_type", [OSError, McpMessageTooLargeError])
def test_stdio_post_dispatch_write_failure_fences_retry_and_redacts_diagnostics(
    write_error_type: type[Exception],
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-post-dispatch-write-secret-canary"

    class PostDispatchFailingWriter:
        def __init__(self, underlying, dispatched: asyncio.Event) -> None:
            self.underlying = underlying
            self.dispatched = dispatched

        def write(self, data: bytes) -> None:
            self.underlying.write(data)
            self.dispatched.set()

        async def drain(self) -> None:
            await self.underlying.drain()
            raise write_error_type(secret)

        def close(self) -> None:
            self.underlying.close()

        async def wait_closed(self) -> None:
            await self.underlying.wait_closed()

    async def run() -> tuple[BaseException, BaseException]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        assert session.process.stdin is not None
        dispatched = asyncio.Event()
        session.process.stdin = PostDispatchFailingWriter(  # type: ignore[assignment]
            session.process.stdin,
            dispatched,
        )
        try:
            with pytest.raises(McpProtocolError, match="transport write failed") as exc_info:
                await session.call_tool("echo", {"text": "first call may have executed"})
            error = exc_info.value
            assert dispatched.is_set()
            assert session._closed is True
            assert session._pending == {}

            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {"text": "must not retry"})

            cleanup_task = _mcp_session_close_task(error)
            assert cleanup_task is not None
            cleanup_outcome = (await asyncio.gather(cleanup_task, return_exceptions=True))[0]
            assert isinstance(cleanup_outcome, McpProtocolError)
            assert session.process.returncode is not None
            return error, cleanup_outcome
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, cleanup_outcome = asyncio.run(run())

    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in repr(error)
    assert secret not in repr(cleanup_outcome)
    _assert_cayu_traceback_does_not_retain(error, secret)
    _assert_cayu_traceback_does_not_retain(cleanup_outcome, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_grouped_post_dispatch_write_failure_fences_retry_and_settles(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-grouped-write-secret-canary"
    raw_failure = BaseExceptionGroup(
        f"grouped write exposed {secret}",
        [
            BaseExceptionGroup(
                f"nested write failure exposed {secret}",
                [
                    asyncio.CancelledError(f"write cancelled with {secret}"),
                    RuntimeError(f"write failed with {secret}"),
                ],
            )
        ],
    )

    class GroupedPostDispatchFailingWriter:
        def __init__(self, underlying, dispatched: asyncio.Event) -> None:
            self.underlying = underlying
            self.dispatched = dispatched

        def write(self, data: bytes) -> None:
            self.underlying.write(data)
            self.dispatched.set()

        async def drain(self) -> None:
            await self.underlying.drain()
            raise raw_failure

        def close(self) -> None:
            self.underlying.close()

        async def wait_closed(self) -> None:
            await self.underlying.wait_closed()

    async def run() -> tuple[BaseExceptionGroup, BaseException, int]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        assert session.process.stdin is not None
        dispatched = asyncio.Event()
        session.process.stdin = GroupedPostDispatchFailingWriter(  # type: ignore[assignment]
            session.process.stdin,
            dispatched,
        )
        try:
            with pytest.raises(BaseExceptionGroup) as exc_info:
                await session.call_tool("echo", {"text": "first call may have executed"})
            error = exc_info.value
            current = asyncio.current_task()
            assert current is not None
            assert dispatched.is_set()
            assert session._closed is True
            assert session._pending == {}
            assert session._pending_timing == {}

            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {"text": "must not retry"})

            cleanup_task = _mcp_session_close_task(error)
            assert cleanup_task is not None
            cleanup_outcome = (await asyncio.gather(cleanup_task, return_exceptions=True))[0]
            assert isinstance(cleanup_outcome, BaseExceptionGroup)
            assert session.process.returncode is not None
            return error, cleanup_outcome, current.cancelling()
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, cleanup_outcome, cancelling = asyncio.run(run())

    assert cancelling == 0
    assert error is not raw_failure
    assert isinstance(error.exceptions[0], BaseExceptionGroup)
    nested_error = error.exceptions[0]
    # A historical cancellation leaf in a completed writer is diagnostic
    # evidence, not a cancellation request delivered to this caller.
    assert isinstance(nested_error.exceptions[0], McpProtocolError)
    assert isinstance(nested_error.exceptions[1], McpProtocolError)
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in repr(error)
    assert secret not in repr(cleanup_outcome)
    _assert_cayu_traceback_does_not_retain(error, secret)
    _assert_cayu_traceback_does_not_retain(cleanup_outcome, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_custom_scalar_post_dispatch_failure_is_typed_redacted_and_fenced(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-custom-scalar-secret-canary"

    class CustomScalarSignal(BaseException):
        pass

    raw_failure = CustomScalarSignal(f"transport exposed {secret}")

    class CustomScalarPostDispatchWriter:
        def __init__(self, underlying, dispatched: asyncio.Event) -> None:
            self.underlying = underlying
            self.dispatched = dispatched

        def write(self, data: bytes) -> None:
            self.underlying.write(data)
            self.dispatched.set()

        async def drain(self) -> None:
            await self.underlying.drain()
            raise raw_failure

        def close(self) -> None:
            self.underlying.close()

        async def wait_closed(self) -> None:
            await self.underlying.wait_closed()

    async def run() -> tuple[BaseException, BaseException, bool, int | None]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        assert session.process.stdin is not None
        dispatched = asyncio.Event()
        session.process.stdin = CustomScalarPostDispatchWriter(  # type: ignore[assignment]
            session.process.stdin,
            dispatched,
        )
        try:
            with pytest.raises(McpProtocolError) as exc_info:
                await session.call_tool("echo", {"text": "first call may have executed"})
            error = exc_info.value
            assert dispatched.is_set()
            assert session._closed is True
            assert session._pending == {}
            assert session._pending_timing == {}

            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {"text": "must not retry"})

            cleanup_task = _mcp_session_close_task(error)
            assert cleanup_task is not None
            cleanup_outcome = (await asyncio.gather(cleanup_task, return_exceptions=True))[0]
            assert isinstance(cleanup_outcome, McpProtocolError)
            return error, cleanup_outcome, session._closed, session.process.returncode
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, cleanup_outcome, session_closed, returncode = asyncio.run(run())

    assert type(error) is McpProtocolError
    assert error is not raw_failure
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in repr(error)
    assert secret not in repr(cleanup_outcome)
    assert REDACTED_SECRET in repr(error)
    _assert_cayu_traceback_does_not_retain(error, secret)
    _assert_cayu_traceback_does_not_retain(cleanup_outcome, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert session_closed is True
    assert returncode is not None


@pytest.mark.parametrize("fatal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_stdio_scalar_fatal_post_dispatch_write_failure_fences_and_settles(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
    fatal_type: type[BaseException],
) -> None:
    secret = "mcp-stdio-scalar-fatal-write-secret-canary"
    raw_failure = fatal_type(f"transport exposed {secret}")

    class ScalarFatalPostDispatchWriter:
        def __init__(self, underlying, dispatched: asyncio.Event) -> None:
            self.underlying = underlying
            self.dispatched = dispatched

        def write(self, data: bytes) -> None:
            self.underlying.write(data)
            self.dispatched.set()

        async def drain(self) -> None:
            await self.underlying.drain()
            raise raw_failure

        def close(self) -> None:
            self.underlying.close()

        async def wait_closed(self) -> None:
            await self.underlying.wait_closed()

    async def run() -> tuple[BaseException, BaseException, bool, int | None]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        assert session.process.stdin is not None
        dispatched = asyncio.Event()
        session.process.stdin = ScalarFatalPostDispatchWriter(  # type: ignore[assignment]
            session.process.stdin,
            dispatched,
        )
        try:
            with pytest.raises(fatal_type) as exc_info:
                await session.call_tool("echo", {"text": "first call may have executed"})
            error = exc_info.value
            assert dispatched.is_set()
            assert session._closed is True
            assert session._pending == {}
            assert session._pending_timing == {}

            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {"text": "must not retry"})

            cleanup_task = _mcp_session_close_task(error)
            assert cleanup_task is not None
            cleanup_outcome = (await asyncio.gather(cleanup_task, return_exceptions=True))[0]
            assert isinstance(cleanup_outcome, McpProtocolError)
            return error, cleanup_outcome, session._closed, session.process.returncode
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error, cleanup_outcome, session_closed, returncode = asyncio.run(run())

    assert error is not raw_failure
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in repr(error)
    assert secret not in repr(cleanup_outcome)
    _assert_cayu_traceback_does_not_retain(error, secret)
    _assert_cayu_traceback_does_not_retain(cleanup_outcome, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
    assert session_closed is True
    assert returncode is not None


def test_stdio_retained_callback_consumes_mixed_failure_group() -> None:
    async def run() -> tuple[BaseException, list[dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        diagnostics: list[dict[str, Any]] = []
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: diagnostics.append(context))

        async def fail_with_mixed_group() -> None:
            raise BaseExceptionGroup(
                "mixed stdio cleanup failure",
                [
                    asyncio.CancelledError("historical cleanup cancellation"),
                    RuntimeError("ordinary cleanup failure"),
                ],
            )

        try:
            task = asyncio.create_task(fail_with_mixed_group())
            task.add_done_callback(mcp_stdio_module._consume_task_result)
            outcome = (await asyncio.gather(task, return_exceptions=True))[0]
            await asyncio.sleep(0)
            return outcome, diagnostics
        finally:
            loop.set_exception_handler(previous_handler)

    outcome, diagnostics = asyncio.run(run())

    assert isinstance(outcome, BaseExceptionGroup)
    assert diagnostics == []


def test_stdio_close_failure_still_terminates_process_and_settles_pending_call(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-close-failure-secret-canary"
    raw_failure = RuntimeError(f"stdin close exposed {secret}")

    class CloseFailingWriter:
        def __init__(self, underlying, dispatched: asyncio.Event) -> None:
            self.underlying = underlying
            self.dispatched = dispatched
            self.close_calls = 0

        def write(self, data: bytes) -> None:
            self.underlying.write(data)
            self.dispatched.set()

        async def drain(self) -> None:
            await self.underlying.drain()

        def close(self) -> None:
            self.close_calls += 1
            raise raw_failure

        async def wait_closed(self) -> None:
            await self.underlying.wait_closed()

    async def run() -> tuple[BaseException, BaseException, BaseException, int, bool, bool, int]:
        session = await StdioMcpClient(
            transport_limits=_limits(),
            graceful_shutdown_timeout_s=0.01,
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        session._secret_redactor = SecretRedactor(secret)
        assert session.process.stdin is not None
        dispatched = asyncio.Event()
        writer = CloseFailingWriter(session.process.stdin, dispatched)
        session.process.stdin = writer  # type: ignore[assignment]
        pending_call = asyncio.create_task(
            session.call_tool("echo", {"text": "pending", "defer_response": True})
        )
        await dispatched.wait()

        with pytest.raises(McpProtocolError, match="session cleanup failed") as first_info:
            await session.close()
        with pytest.raises(McpProtocolError, match="session cleanup failed") as retry_info:
            await session.close()
        pending_outcome = (await asyncio.gather(pending_call, return_exceptions=True))[0]
        assert isinstance(pending_outcome, McpProtocolError)
        return (
            first_info.value,
            retry_info.value,
            pending_outcome,
            session.process.returncode,
            session._reader_task.done(),
            session._stderr_task.done(),
            writer.close_calls,
        )

    with caplog.at_level(logging.DEBUG):
        (
            first_error,
            retry_error,
            pending_error,
            returncode,
            reader_done,
            stderr_done,
            close_calls,
        ) = asyncio.run(run())

    assert returncode is not None
    assert reader_done is True
    assert stderr_done is True
    assert close_calls == 1
    for error in (first_error, retry_error, pending_error):
        assert secret not in repr(error)
        _assert_cayu_traceback_does_not_retain(error, secret)
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)


def test_stdio_close_bounds_cancellation_resistant_shutdown_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BlockingCloseWriter:
        def __init__(
            self,
            underlying: Any,
            wait_until_released: Callable[[str], Awaitable[None]],
        ) -> None:
            self.underlying = underlying
            self.wait_until_released = wait_until_released

        def write(self, data: bytes) -> None:
            self.underlying.write(data)

        async def drain(self) -> None:
            await self.wait_until_released("drain")

        def close(self) -> None:
            # Model a pipe transport that starts closure but cannot flush its
            # buffered bytes while the child refuses to read stdin.
            return None

        async def wait_closed(self) -> None:
            await self.wait_until_released("stdin")

    async def run() -> None:
        session = await StdioMcpClient(
            write_timeout_s=0.01,
            graceful_shutdown_timeout_s=0.05,
        ).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        assert session.process.stdin is not None
        retained_before = set(mcp_stdio_module._RETAINED_STDIO_SHUTDOWN_TASKS)
        release = asyncio.Event()
        cancelled_phases: set[str] = set()
        process_signals = {"terminate": 0, "kill": 0}

        async def wait_until_released(phase: str) -> None:
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled_phases.add(phase)

        writer = BlockingCloseWriter(session.process.stdin, wait_until_released)
        session.process.stdin = writer  # type: ignore[assignment]

        original_wait = session.process.wait
        original_kill = session.process.kill

        async def cancellation_resistant_process_wait() -> int:
            await wait_until_released("process")
            return await original_wait()

        def ignore_terminate() -> None:
            process_signals["terminate"] += 1

        def record_kill() -> None:
            process_signals["kill"] += 1
            original_kill()

        monkeypatch.setattr(session.process, "wait", cancellation_resistant_process_wait)
        monkeypatch.setattr(session.process, "terminate", ignore_terminate)
        monkeypatch.setattr(session.process, "kill", record_kill)

        try:
            with pytest.raises(TimeoutError, match="write timed out") as exc_info:
                await session.call_tool("echo", {"text": "request may have been dispatched"})
            cleanup_task = _mcp_session_close_task(exc_info.value)
            assert cleanup_task is not None
            assert session._closed is True
            with pytest.raises(McpProtocolError, match="session is closed"):
                await session.call_tool("echo", {"text": "must not retry"})
            assert session._close_task is not None
            await asyncio.wait_for(asyncio.shield(session._close_task), timeout=1)
            assert cleanup_task.done() is False
            retained = tuple(
                task
                for task in mcp_stdio_module._RETAINED_STDIO_SHUTDOWN_TASKS
                if task not in retained_before
            )
            assert retained
            assert session.process.returncode is not None
            returncode = session.process.returncode
            reader_done = session._reader_task.done()
            stderr_done = session._stderr_task.done()
            release.set()
            cleanup_outcomes = await asyncio.gather(
                cleanup_task,
                *retained,
                return_exceptions=True,
            )
            assert cleanup_outcomes[0] is None
            assert not any(isinstance(outcome, BaseException) for outcome in cleanup_outcomes)
            await asyncio.sleep(0)
            assert not (mcp_stdio_module._RETAINED_STDIO_SHUTDOWN_TASKS - retained_before)
            assert process_signals == {"terminate": 1, "kill": 1}
            assert cancelled_phases == {"drain", "stdin", "process"}
            assert reader_done and stderr_done
            assert returncode is not None
        finally:
            release.set()
            if session.process.returncode is None:
                original_kill()
            await original_wait()

    asyncio.run(run())


def test_stdio_caller_cancellation_during_broken_pipe_diagnostics_still_fences() -> None:
    class BrokenPipeWriter:
        def __init__(self, underlying) -> None:
            self.underlying = underlying

        def write(self, data: bytes) -> None:
            del data
            raise BrokenPipeError("peer closed")

        async def drain(self) -> None:
            return None

        def close(self) -> None:
            self.underlying.close()

        async def wait_closed(self) -> None:
            await self.underlying.wait_closed()

    async def run() -> tuple[int, bool, bool, int | None]:
        session = await StdioMcpClient(transport_limits=_limits()).connect(_server_spec())
        assert isinstance(session, StdioMcpSession)
        assert session.process.stdin is not None
        session.process.stdin = BrokenPipeWriter(session.process.stdin)  # type: ignore[assignment]
        session._stderr_task.cancel()
        await asyncio.gather(session._stderr_task, return_exceptions=True)
        diagnostics_wait_started = asyncio.Event()

        async def blocked_stderr_drain() -> None:
            await asyncio.Event().wait()

        session._stderr_task = asyncio.create_task(blocked_stderr_drain())
        original_await_stderr_drain = session._await_stderr_drain

        async def capture_diagnostics_wait(*, deadline: float | None = None) -> None:
            diagnostics_wait_started.set()
            await original_await_stderr_drain(deadline=deadline)

        session._await_stderr_drain = capture_diagnostics_wait
        call_task = asyncio.create_task(session.call_tool("echo", {}))
        await diagnostics_wait_started.wait()
        call_task.cancel("cancel broken-pipe diagnostics")
        cancelling = call_task.cancelling()
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel broken-pipe diagnostics",
        ) as exc_info:
            await call_task
        cleanup_task = _mcp_session_close_task(exc_info.value)
        assert cleanup_task is not None
        cleanup_outcome = (await asyncio.gather(cleanup_task, return_exceptions=True))[0]
        assert isinstance(cleanup_outcome, McpProtocolError)
        assert not isinstance(cleanup_outcome, BrokenPipeError)
        return (
            cancelling,
            call_task.cancelled(),
            session._closed,
            session.process.returncode,
        )

    cancelling, cancelled, session_closed, returncode = asyncio.run(run())

    assert cancelling == 1
    assert cancelled is True
    assert session_closed is True
    assert returncode is not None


def test_stdio_oversize_failure_does_not_expose_secret_diagnostics(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    recwarn: pytest.WarningsRecorder,
) -> None:
    secret = "mcp-stdio-limit-secret-canary"
    spec = McpServerSpec(
        name="bounded-secret-stdio",
        command=[sys.executable, str(_FAKE_SERVER)],
        secret_env={"CAYU_FAKE_MCP_LIMIT_CANARY": SecretRef(name="token")},
    )

    async def run() -> BaseException:
        session = await StdioMcpClient(
            transport_limits=_limits(),
            secret_resolver=StaticVault({"token": secret}),
        ).connect(spec)
        try:
            with pytest.raises(McpMessageTooLargeError) as excinfo:
                await session.call_tool(
                    "echo",
                    {
                        "exact_response_bytes": 1_025,
                        "include_limit_canary": True,
                    },
                )
            return excinfo.value
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        error = asyncio.run(run())

    assert secret not in repr(error)
    traceback = error.__traceback__
    while traceback is not None:
        if is_cayu_source_filename(traceback.tb_frame.f_code.co_filename):
            assert all(secret not in repr(value) for value in traceback.tb_frame.f_locals.values())
        traceback = traceback.tb_next
    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert all(secret not in record.getMessage() for record in caplog.records)
    assert all(secret not in str(warning.message) for warning in recwarn)
