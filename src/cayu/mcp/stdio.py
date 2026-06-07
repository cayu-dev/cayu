from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from contextlib import suppress
from typing import Any, cast

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank
from cayu.mcp.base import (
    McpClient,
    McpInitializeResult,
    McpResourceDefinition,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
)

MCP_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_MCP_REQUEST_TIMEOUT_S = 30.0
DEFAULT_MCP_WRITE_TIMEOUT_S = 5.0
DEFAULT_MCP_GRACEFUL_SHUTDOWN_TIMEOUT_S = 2.0
DEFAULT_MCP_CANCELLATION_NOTIFICATION_TIMEOUT_S = 1.0
DEFAULT_MCP_CLIENT_NAME = "cayu"
DEFAULT_MCP_CLIENT_VERSION = "0.1.0"
_JSONRPC_METHOD_NOT_FOUND = -32601


class McpProtocolError(RuntimeError):
    """Raised when an MCP server violates the expected JSON-RPC contract."""


class StdioMcpClient(McpClient):
    """MCP client for local stdio servers."""

    def __init__(
        self,
        *,
        request_timeout_s: float = DEFAULT_MCP_REQUEST_TIMEOUT_S,
        write_timeout_s: float = DEFAULT_MCP_WRITE_TIMEOUT_S,
        graceful_shutdown_timeout_s: float = DEFAULT_MCP_GRACEFUL_SHUTDOWN_TIMEOUT_S,
        cancellation_notification_timeout_s: float = DEFAULT_MCP_CANCELLATION_NOTIFICATION_TIMEOUT_S,
        client_name: str = DEFAULT_MCP_CLIENT_NAME,
        client_version: str = DEFAULT_MCP_CLIENT_VERSION,
        inherit_env: bool = False,
    ) -> None:
        self.request_timeout_s = _validate_positive_number(
            request_timeout_s,
            "request_timeout_s",
        )
        self.write_timeout_s = _validate_positive_number(
            write_timeout_s,
            "write_timeout_s",
        )
        self.graceful_shutdown_timeout_s = _validate_positive_number(
            graceful_shutdown_timeout_s,
            "graceful_shutdown_timeout_s",
        )
        self.cancellation_notification_timeout_s = _validate_positive_number(
            cancellation_notification_timeout_s,
            "cancellation_notification_timeout_s",
        )
        self.client_name = require_clean_nonblank(client_name, "client_name")
        self.client_version = require_clean_nonblank(client_version, "client_version")
        if type(inherit_env) is not bool:
            raise TypeError("inherit_env must be a bool.")
        self.inherit_env = inherit_env

    async def connect(self, server: McpServerSpec) -> McpSession:
        if type(server) is not McpServerSpec:
            raise TypeError("server must be an McpServerSpec.")
        if server.command is None:
            raise ValueError("StdioMcpClient requires an MCP server command.")
        if server.url is not None:
            raise ValueError("StdioMcpClient does not support URL MCP servers.")
        if server.secret_env:
            raise ValueError(
                "StdioMcpClient cannot resolve MCP secret_env values yet. "
                "Resolve secrets before constructing the server spec."
            )
        if server.secret_headers:
            raise ValueError("StdioMcpClient does not support MCP secret_headers.")
        child_env = dict(os.environ) if self.inherit_env else {}
        child_env.update(server.env)
        process = await asyncio.create_subprocess_exec(
            *server.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )
        session = StdioMcpSession(
            server=server,
            process=process,
            request_timeout_s=self.request_timeout_s,
            write_timeout_s=self.write_timeout_s,
            graceful_shutdown_timeout_s=self.graceful_shutdown_timeout_s,
            cancellation_notification_timeout_s=self.cancellation_notification_timeout_s,
            client_name=self.client_name,
            client_version=self.client_version,
        )
        try:
            await session.initialize()
        except asyncio.CancelledError:
            await _close_session_after_failed_connect(session)
            raise
        except Exception:
            await _close_session_after_failed_connect(session)
            raise
        return session


class StdioMcpSession(McpSession):
    def __init__(
        self,
        *,
        server: McpServerSpec,
        process: asyncio.subprocess.Process,
        request_timeout_s: float,
        write_timeout_s: float,
        graceful_shutdown_timeout_s: float,
        cancellation_notification_timeout_s: float,
        client_name: str,
        client_version: str,
    ) -> None:
        self.server = server
        self.process = process
        self.request_timeout_s = request_timeout_s
        self.write_timeout_s = write_timeout_s
        self.graceful_shutdown_timeout_s = graceful_shutdown_timeout_s
        self.cancellation_notification_timeout_s = cancellation_notification_timeout_s
        self.client_name = client_name
        self.client_version = client_version
        self._initialize_result: McpInitializeResult | None = None
        self._next_id = 1
        self._closed = False
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._close_task: asyncio.Task[None] | None = None

    @property
    def initialize_result(self) -> McpInitializeResult:
        if self._initialize_result is None:
            raise McpProtocolError("MCP session has not been initialized.")
        return self._initialize_result

    async def initialize(self) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": self.client_name,
                    "version": self.client_version,
                },
            },
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP initialize result must be an object.")
        self._initialize_result = _initialize_result(result)
        if self._initialize_result.protocol_version != MCP_PROTOCOL_VERSION:
            raise McpProtocolError(
                "MCP server negotiated unsupported protocol version "
                f"{self._initialize_result.protocol_version!r}."
            )
        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        result = await self._request("tools/list", {})
        if type(result) is not dict:
            raise McpProtocolError("MCP tools/list result must be an object.")
        tools = result.get("tools", [])
        if not isinstance(tools, list):
            raise McpProtocolError("MCP tools/list result tools must be a list.")
        return tuple(_tool_definition(tool, self.server.name) for tool in tools)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        tool_name = require_clean_nonblank(name, "tool name")
        copied_arguments = copy_json_value(arguments, "arguments")
        if type(copied_arguments) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        result = await self._request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": copied_arguments,
            },
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP tools/call result must be an object.")
        return _tool_result(result)

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        result = await self._request("resources/list", {})
        if type(result) is not dict:
            raise McpProtocolError("MCP resources/list result must be an object.")
        resources = result.get("resources", [])
        if not isinstance(resources, list):
            raise McpProtocolError("MCP resources/list result resources must be a list.")
        return tuple(_resource_definition(resource, self.server.name) for resource in resources)

    async def read_resource(self, uri: str) -> McpResourceResult:
        resource_uri = require_clean_nonblank(uri, "resource uri")
        result = await self._request("resources/read", {"uri": resource_uri})
        if type(result) is not dict:
            raise McpProtocolError("MCP resources/read result must be an object.")
        contents = result.get("contents", [])
        if not isinstance(contents, list):
            raise McpProtocolError("MCP resources/read result contents must be a list.")
        return McpResourceResult(contents=contents)

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._close_impl())
        cleanup_task = self._close_task
        was_cancelled = False
        while True:
            try:
                await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError:
                was_cancelled = True
                if cleanup_task.done():
                    break
        if was_cancelled:
            raise asyncio.CancelledError

    async def _close_impl(self) -> None:
        if self.process.returncode is None:
            await self._close_stdin_for_graceful_shutdown()
            try:
                await asyncio.wait_for(
                    self.process.wait(),
                    timeout=self.graceful_shutdown_timeout_s,
                )
            except TimeoutError:
                self.process.terminate()
                try:
                    await asyncio.wait_for(
                        self.process.wait(),
                        timeout=self.graceful_shutdown_timeout_s,
                    )
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
        self._fail_pending(McpProtocolError("MCP stdio session closed."))
        await self._cancel_background_task(self._reader_task)
        await self._cancel_background_task(self._stderr_task)

    async def _request(self, method: str, params: dict[str, Any]) -> Any:
        if self._closed:
            raise McpProtocolError("MCP stdio session is closed.")
        method_name = require_clean_nonblank(method, "method")
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        request_written = False
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method_name,
            "params": copy_json_value(params, "params"),
        }
        try:
            await self._write_with_timeout(
                payload,
                timeout_message=f"MCP request {request_id} write timed out.",
            )
            request_written = True
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise
        except TimeoutError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise
        except Exception:
            self._pending.pop(request_id, None)
            raise
        try:
            response = await asyncio.wait_for(future, timeout=self.request_timeout_s)
        except TimeoutError:
            self._pending.pop(request_id, None)
            if request_written:
                await self._send_request_cancelled_notification(
                    request_id,
                    method_name=method_name,
                    reason="Cayu request timed out.",
                )
            raise TimeoutError(f"MCP request {request_id} timed out.") from None
        except asyncio.CancelledError:
            self._pending.pop(request_id, None)
            future.cancel()
            if request_written:
                await self._send_request_cancelled_notification(
                    request_id,
                    method_name=method_name,
                    reason="Cayu caller cancelled the request.",
                )
            raise
        if "error" in response:
            error = response["error"]
            if isinstance(error, Mapping):
                message = error.get("message", "MCP request failed.")
                raise McpProtocolError(f"MCP {method_name} failed: {message}")
            raise McpProtocolError(f"MCP {method_name} failed.")
        if "result" not in response:
            raise McpProtocolError(f"MCP {method_name} response missing result.")
        return copy_json_value(response["result"], "result")

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        method_name = require_clean_nonblank(method, "method")
        await self._write_with_timeout(
            {
                "jsonrpc": "2.0",
                "method": method_name,
                "params": copy_json_value(params, "params"),
            },
            timeout_message=f"MCP notification {method_name} write timed out.",
        )

    async def _write_with_timeout(
        self,
        payload: dict[str, Any],
        *,
        timeout_message: str,
    ) -> None:
        try:
            await asyncio.wait_for(self._write(payload), timeout=self.write_timeout_s)
        except TimeoutError:
            await self._close_after_interrupted_transport_write()
            raise TimeoutError(timeout_message) from None
        except asyncio.CancelledError:
            await self._close_after_interrupted_transport_write()
            raise

    async def _close_after_interrupted_transport_write(self) -> None:
        if asyncio.current_task() is self._reader_task:
            close_task = asyncio.create_task(self.close())
            close_task.add_done_callback(_consume_task_result)
            return
        await _close_session_after_failed_connect(self)

    async def _write(self, payload: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise McpProtocolError("MCP stdio process stdin is unavailable.")
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        async with self._write_lock:
            self.process.stdin.write(data + b"\n")
            await self.process.stdin.drain()

    async def _close_stdin_for_graceful_shutdown(self) -> None:
        stdin = self.process.stdin
        if stdin is None:
            return
        stdin.close()
        wait_closed = getattr(stdin, "wait_closed", None)
        if wait_closed is not None:
            with suppress(BrokenPipeError, ConnectionResetError):
                await wait_closed()

    async def _cancel_background_task(self, task: asyncio.Task) -> None:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _send_request_cancelled_notification(
        self,
        request_id: int,
        *,
        method_name: str,
        reason: str,
    ) -> None:
        if method_name == "initialize":
            return
        notify_task = asyncio.create_task(
            self._notify(
                "notifications/cancelled",
                {"requestId": request_id, "reason": reason},
            )
        )
        try:
            await asyncio.wait_for(
                asyncio.shield(notify_task),
                timeout=self.cancellation_notification_timeout_s,
            )
        except (Exception, asyncio.CancelledError):
            notify_task.cancel()
            notify_task.add_done_callback(_consume_task_result)
            await self._close_after_interrupted_transport_write()

    async def _read_loop(self) -> None:
        error: BaseException | None = None
        try:
            while True:
                message = await self._read_message()
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        finally:
            self._fail_pending(
                error if error is not None else McpProtocolError("MCP stdio reader stopped."),
            )

    async def _handle_message(self, message: dict[str, Any]) -> None:
        message_id = message.get("id")
        if "method" in message:
            if message_id is not None:
                await self._write_server_request_error(message)
            return
        if message_id is None:
            return
        if type(message_id) is not int:
            return
        future = self._pending.pop(message_id, None)
        if future is None or future.done():
            return
        future.set_result(message)

    async def _write_server_request_error(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        method_name = method if isinstance(method, str) else "unknown"
        await self._write_with_timeout(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": _JSONRPC_METHOD_NOT_FOUND,
                    "message": f"Cayu does not support MCP server request: {method_name}",
                },
            },
            timeout_message=f"MCP server request rejection for {method_name} write timed out.",
        )

    def _fail_pending(self, error: BaseException) -> None:
        pending = self._pending
        self._pending = {}
        for future in pending.values():
            if not future.done():
                future.set_exception(error)

    async def _read_message(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise McpProtocolError("MCP stdio process stdout is unavailable.")
        line = await self.process.stdout.readline()
        if not line:
            raise McpProtocolError("MCP stdio process closed stdout.")
        try:
            payload = json.loads(line.decode("utf-8"))
        except ValueError as exc:
            raise McpProtocolError("MCP stdio process wrote invalid JSON.") from exc
        if type(payload) is not dict:
            raise McpProtocolError("MCP JSON-RPC message must be an object.")
        if payload.get("jsonrpc") != "2.0":
            raise McpProtocolError("MCP JSON-RPC message must use jsonrpc='2.0'.")
        return payload

    async def _drain_stderr(self) -> None:
        if self.process.stderr is None:
            return
        while True:
            line = await self.process.stderr.readline()
            if not line:
                return


def _tool_definition(payload: object, server_name: str) -> McpToolDefinition:
    if type(payload) is not dict:
        raise McpProtocolError("MCP tool definitions must be objects.")
    payload = cast("dict[str, Any]", payload)
    name = _mapping_string(payload, "name")
    description = _optional_mapping_string(payload, "description") or ""
    input_schema = payload.get("inputSchema", {})
    if type(input_schema) is not dict:
        raise McpProtocolError("MCP tool inputSchema must be an object.")
    annotations = payload.get("annotations", {})
    if type(annotations) is not dict:
        raise McpProtocolError("MCP tool annotations must be an object.")
    return McpToolDefinition(
        name=name,
        description=description,
        input_schema=input_schema,
        annotations={
            **annotations,
            "mcp_server": server_name,
        },
    )


def _initialize_result(payload: dict[str, Any]) -> McpInitializeResult:
    protocol_version = payload.get("protocolVersion")
    if not isinstance(protocol_version, str):
        raise McpProtocolError("MCP initialize protocolVersion must be a string.")
    capabilities = payload.get("capabilities", {})
    if type(capabilities) is not dict:
        raise McpProtocolError("MCP initialize capabilities must be an object.")
    server_info = payload.get("serverInfo", {})
    if server_info is None:
        server_info = {}
    if type(server_info) is not dict:
        raise McpProtocolError("MCP initialize serverInfo must be an object.")
    instructions = payload.get("instructions")
    if instructions is not None and type(instructions) is not str:
        raise McpProtocolError("MCP initialize instructions must be a string.")
    return McpInitializeResult(
        protocol_version=protocol_version,
        server_name=_optional_mapping_string(server_info, "name"),
        server_version=_optional_mapping_string(server_info, "version"),
        instructions=instructions,
        capabilities=capabilities,
    )


def _tool_result(payload: dict[str, Any]) -> McpToolResult:
    content = payload.get("content", [])
    if not isinstance(content, list):
        raise McpProtocolError("MCP tool result content must be a list.")
    structured_content = payload.get("structuredContent")
    if structured_content is not None and type(structured_content) is not dict:
        raise McpProtocolError("MCP structuredContent must be an object.")
    is_error = payload.get("isError", False)
    if type(is_error) is not bool:
        raise McpProtocolError("MCP tool result isError must be a bool.")
    return McpToolResult(
        content=content,
        structured_content=structured_content,
        is_error=is_error,
    )


def _resource_definition(payload: object, server_name: str) -> McpResourceDefinition:
    if type(payload) is not dict:
        raise McpProtocolError("MCP resource definitions must be objects.")
    payload = cast("dict[str, Any]", payload)
    uri = _mapping_string(payload, "uri")
    metadata = {
        "mcp_server": server_name,
    }
    return McpResourceDefinition(
        uri=uri,
        name=_optional_mapping_string(payload, "name"),
        description=_optional_mapping_string(payload, "description"),
        mime_type=_optional_mapping_string(payload, "mimeType"),
        metadata=metadata,
    )


def _mapping_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise McpProtocolError(f"MCP {key} must be a string.")
    return require_clean_nonblank(value, key)


def _optional_mapping_string(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise McpProtocolError(f"MCP {key} must be a string.")
    return require_nonblank(value, key)


def _consume_task_result(task: asyncio.Task) -> None:
    with suppress(Exception, asyncio.CancelledError):
        task.result()


async def _close_session_after_failed_connect(session: McpSession) -> None:
    close_task = asyncio.create_task(session.close())
    while True:
        try:
            await asyncio.shield(close_task)
            return
        except Exception:
            return
        except asyncio.CancelledError:
            if close_task.done():
                return


def _validate_positive_number(value: float, field_name: str) -> float:
    if type(value) not in {float, int}:
        raise TypeError(f"{field_name} must be a number.")
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return numeric
