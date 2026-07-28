from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from typing import Any

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.mcp._jsonrpc import (
    DEFAULT_MCP_CLIENT_NAME,
    DEFAULT_MCP_CLIENT_VERSION,
    DEFAULT_MCP_MAX_LIST_ITEMS,
    DEFAULT_MCP_MAX_LIST_PAGES,
    DEFAULT_MCP_REQUEST_TIMEOUT_S,
    JSONRPC_METHOD_NOT_FOUND,
    JsonrpcAuthorityMappingResult,
    McpPaginatedPage,
    McpProtocolError,
    collect_paginated,
    initialize_params,
    initialize_result_from_payload,
    jsonrpc_authority_mapping,
    jsonrpc_notification_payload,
    jsonrpc_request_payload,
    merge_jsonrpc_authority_mapping,
    resource_definition_from_payload,
    result_from_jsonrpc_response,
    safely_redact_jsonrpc_response,
    tool_definition_from_payload,
    tool_result_from_payload,
    validate_negotiated_protocol_version,
    validate_positive_integer,
    validate_positive_number,
)
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
from cayu.vaults import (
    REDACTED_SECRET,
    SecretRedactor,
    SecretResolver,
    resolve_secret_env,
    validate_secret_resolver,
)

DEFAULT_MCP_WRITE_TIMEOUT_S = 5.0
DEFAULT_MCP_GRACEFUL_SHUTDOWN_TIMEOUT_S = 2.0
DEFAULT_MCP_CANCELLATION_NOTIFICATION_TIMEOUT_S = 1.0

# When we do not inherit the full parent environment, npx/uvx and other stdio
# launchers still need a handful of variables (a PATH to find the binary, a HOME
# for their package caches, and a locale) or they fail to start at all. Copy only
# this minimal safelist through so the child stays isolated from the rest of the
# parent env while remaining launchable.
_MINIMAL_ENV_SAFELIST = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SYSTEMROOT",
    "APPDATA",
    "USERPROFILE",
)

# Retain at most this many bytes of the child's most recent stderr output so a
# startup crash surfaces in the resulting protocol error instead of being lost.
DEFAULT_MCP_STDERR_CAPTURE_BYTES = 8192
# Best-effort grace period to let the stderr drain finish (and reach EOF) after
# the child closes stdout, so a crash message lands in the captured tail.
DEFAULT_MCP_STDERR_DRAIN_GRACE_S = 0.2


def _base_child_env(inherit_env: bool) -> dict[str, str]:
    """Build the base child environment before per-server overrides.

    Inherits the full parent env when requested; otherwise copies only the
    minimal safelist so launchers such as npx/uvx can still be found and run.
    """
    if inherit_env:
        return dict(os.environ)
    env: dict[str, str] = {}
    for name in _MINIMAL_ENV_SAFELIST:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


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
        max_list_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
        max_list_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
        inherit_env: bool = False,
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        self.request_timeout_s = validate_positive_number(
            request_timeout_s,
            "request_timeout_s",
        )
        self.write_timeout_s = validate_positive_number(
            write_timeout_s,
            "write_timeout_s",
        )
        self.graceful_shutdown_timeout_s = validate_positive_number(
            graceful_shutdown_timeout_s,
            "graceful_shutdown_timeout_s",
        )
        self.cancellation_notification_timeout_s = validate_positive_number(
            cancellation_notification_timeout_s,
            "cancellation_notification_timeout_s",
        )
        self.client_name = require_clean_nonblank(client_name, "client_name")
        self.client_version = require_clean_nonblank(client_version, "client_version")
        self.max_list_pages = validate_positive_integer(max_list_pages, "max_list_pages")
        self.max_list_items = validate_positive_integer(max_list_items, "max_list_items")
        if type(inherit_env) is not bool:
            raise TypeError("inherit_env must be a bool.")
        self.inherit_env = inherit_env
        if secret_resolver is not None:
            validate_secret_resolver(secret_resolver)
        self.secret_resolver = secret_resolver

    async def connect(self, server: McpServerSpec) -> McpSession:
        if type(server) is not McpServerSpec:
            raise TypeError("server must be an McpServerSpec.")
        if server.command is None:
            raise ValueError("StdioMcpClient requires an MCP server command.")
        if server.url is not None:
            raise ValueError("StdioMcpClient does not support URL MCP servers.")
        if server.secret_env and self.secret_resolver is None:
            raise ValueError(
                "StdioMcpClient cannot resolve MCP secret_env without a secret_resolver. "
                "Pass secret_resolver= (a Vault or CredentialProxy) to the client."
            )
        if server.secret_headers:
            raise ValueError("StdioMcpClient does not support MCP secret_headers.")
        child_env = _base_child_env(self.inherit_env)
        child_env.update(server.env)
        secret_redactor = SecretRedactor()
        if server.secret_env and self.secret_resolver is not None:
            # Secret values go straight into the child process env — never into
            # argv — and stay wrapped until this final injection point.
            resolved = await resolve_secret_env(
                server.secret_env,
                self.secret_resolver,
                scope={"mcp_server": server.name},
            )
            for name, secret in resolved.items():
                child_env[name] = secret.value.get_secret_value()
            # A hostile server can echo these values back through tool output; scrub them.
            secret_redactor = SecretRedactor(tuple(resolved.values()))
            resolved.clear()
        try:
            process = await asyncio.create_subprocess_exec(
                *server.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=child_env,
            )
        finally:
            # The child owns its copied environment after a successful spawn. Do
            # not retain injected values if spawning or initialize later fails.
            child_env.clear()
        session = StdioMcpSession(
            server=server,
            process=process,
            request_timeout_s=self.request_timeout_s,
            write_timeout_s=self.write_timeout_s,
            graceful_shutdown_timeout_s=self.graceful_shutdown_timeout_s,
            cancellation_notification_timeout_s=self.cancellation_notification_timeout_s,
            client_name=self.client_name,
            client_version=self.client_version,
            max_list_pages=self.max_list_pages,
            max_list_items=self.max_list_items,
            secret_redactor=secret_redactor,
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
        max_list_pages: int = DEFAULT_MCP_MAX_LIST_PAGES,
        max_list_items: int = DEFAULT_MCP_MAX_LIST_ITEMS,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        self.server = server
        self.process = process
        self._secret_redactor = secret_redactor or SecretRedactor()
        self.request_timeout_s = request_timeout_s
        self.write_timeout_s = write_timeout_s
        self.graceful_shutdown_timeout_s = graceful_shutdown_timeout_s
        self.cancellation_notification_timeout_s = cancellation_notification_timeout_s
        self.client_name = client_name
        self.client_version = client_version
        self.max_list_pages = validate_positive_integer(max_list_pages, "max_list_pages")
        self.max_list_items = validate_positive_integer(max_list_items, "max_list_items")
        self._initialize_result: McpInitializeResult | None = None
        self._next_id = 1
        self._closed = False
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._write_lock = asyncio.Lock()
        self._stderr_tail = bytearray()
        self._reader_task = asyncio.create_task(self._read_loop())
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._close_task: asyncio.Task[None] | None = None
        self._tool_transport_names: dict[str, str] = {}
        self._resource_transport_uris: dict[str, str] = {}
        self._authority_mapping_lock = asyncio.Lock()

    @property
    def initialize_result(self) -> McpInitializeResult:
        if self._initialize_result is None:
            raise McpProtocolError("MCP session has not been initialized.")
        return self._initialize_result

    async def initialize(self) -> None:
        result = await self._request(
            "initialize",
            initialize_params(self.client_name, self.client_version),
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP initialize result must be an object.")
        self._initialize_result = initialize_result_from_payload(result)
        validation_error: McpProtocolError | None = None
        try:
            validate_negotiated_protocol_version(self._initialize_result.protocol_version)
        except McpProtocolError as exc:
            validation_error = McpProtocolError(self._secret_redactor.redact_text(str(exc)))
        if validation_error is not None:
            raise validation_error
        await self._notify("notifications/initialized", {})

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        transport_names: dict[str, str] = {}

        async def request_page(method: str, params: dict[str, Any]) -> Any:
            return await self._request(
                method,
                params,
                authority_mapping=transport_names,
                paginated=True,
            )

        tools = await collect_paginated(
            request_page,
            "tools/list",
            "tools",
            max_pages=self.max_list_pages,
            max_items=self.max_list_items,
            redactor=self._secret_redactor,
        )
        definitions: tuple[McpToolDefinition, ...] = ()
        definition_error = False
        try:
            definitions = tuple(
                tool_definition_from_payload(tool, self.server.name) for tool in tools
            )
        except (McpProtocolError, TypeError, ValueError):
            definition_error = True
        if definition_error:
            transport_names.clear()
            tools.clear()
            raise McpProtocolError("MCP tools/list returned an invalid tool definition.") from None
        async with self._authority_mapping_lock:
            merge_result = merge_jsonrpc_authority_mapping(
                self._tool_transport_names,
                transport_names,
                max_items=self.max_list_items,
            )
            if merge_result.error is not None:
                transport_names.clear()
                raise McpProtocolError(
                    self._secret_redactor.redact_text(merge_result.error)
                ) from None
            self._tool_transport_names = {
                public: raw for public, raw in merge_result.mapping.items() if public != raw
            }
            merge_result.mapping.clear()
            transport_names.clear()
        return definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        tool_name = require_clean_nonblank(name, "tool name")
        copied_arguments = copy_json_value(arguments, "arguments")
        if type(copied_arguments) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        result = await self._request(
            "tools/call",
            {
                "name": self._tool_transport_names.get(tool_name, tool_name),
                "arguments": copied_arguments,
            },
        )
        if type(result) is not dict:
            raise McpProtocolError("MCP tools/call result must be an object.")
        return tool_result_from_payload(result)

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        transport_uris: dict[str, str] = {}

        async def request_page(method: str, params: dict[str, Any]) -> Any:
            return await self._request(
                method,
                params,
                authority_mapping=transport_uris,
                paginated=True,
            )

        resources = await collect_paginated(
            request_page,
            "resources/list",
            "resources",
            max_pages=self.max_list_pages,
            max_items=self.max_list_items,
            redactor=self._secret_redactor,
        )
        definitions: tuple[McpResourceDefinition, ...] = ()
        definition_error = False
        try:
            definitions = tuple(
                resource_definition_from_payload(resource, self.server.name)
                for resource in resources
            )
        except (McpProtocolError, TypeError, ValueError):
            definition_error = True
        if definition_error:
            transport_uris.clear()
            resources.clear()
            raise McpProtocolError(
                "MCP resources/list returned an invalid resource definition."
            ) from None
        async with self._authority_mapping_lock:
            merge_result = merge_jsonrpc_authority_mapping(
                self._resource_transport_uris,
                transport_uris,
                max_items=self.max_list_items,
            )
            if merge_result.error is not None:
                transport_uris.clear()
                raise McpProtocolError(
                    self._secret_redactor.redact_text(merge_result.error)
                ) from None
            self._resource_transport_uris = {
                public: raw for public, raw in merge_result.mapping.items() if public != raw
            }
            merge_result.mapping.clear()
            transport_uris.clear()
        return definitions

    async def read_resource(self, uri: str) -> McpResourceResult:
        resource_uri = require_clean_nonblank(uri, "resource uri")
        result = await self._request(
            "resources/read",
            {"uri": self._resource_transport_uris.get(resource_uri, resource_uri)},
        )
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

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        authority_mapping: dict[str, str] | None = None,
        paginated: bool = False,
    ) -> Any:
        if self._closed:
            raise McpProtocolError("MCP stdio session is closed.")
        method_name = require_clean_nonblank(method, "method")
        request_id = self._next_id
        self._next_id += 1
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        request_written = False
        payload = jsonrpc_request_payload(request_id, method_name, params)
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
        redaction_result = safely_redact_jsonrpc_response(
            response,
            method=method_name,
            redactor=self._secret_redactor,
        )
        redacted_response = redaction_result.response
        mapping_result = JsonrpcAuthorityMappingResult({})
        mapping_error = redaction_result.error
        private_cursor: Any = None
        raw_result: Any = None
        if paginated and mapping_error is None:
            raw_result = response.get("result")
            if type(raw_result) is dict and "nextCursor" in raw_result:
                try:
                    private_cursor = copy_json_value(
                        raw_result["nextCursor"],
                        "nextCursor",
                    )
                except (TypeError, ValueError):
                    mapping_error = f"MCP {method_name} response has an invalid nextCursor."
        if mapping_error is None:
            mapping_result = jsonrpc_authority_mapping(
                response,
                redacted_response,
                method=method_name,
            )
            mapping_error = mapping_result.error
        if mapping_error is None and authority_mapping is not None:
            merge_result = merge_jsonrpc_authority_mapping(
                authority_mapping,
                mapping_result.mapping,
                max_items=self.max_list_items,
            )
            mapping_error = merge_result.error
            if mapping_error is None:
                authority_mapping.clear()
                authority_mapping.update(merge_result.mapping)
            merge_result.mapping.clear()
        if mapping_error is not None and authority_mapping is not None:
            authority_mapping.clear()
        mapping_result.mapping.clear()
        raw_result = None
        response.clear()
        response = {}
        if mapping_error is not None:
            private_cursor = None
            raise McpProtocolError(
                self._secret_redactor.redact_text_bounded(
                    mapping_error,
                    max_bytes=4096,
                )
            ) from None
        try:
            result = result_from_jsonrpc_response(redacted_response, method_name)
        except BaseException:
            private_cursor = None
            raise
        if not paginated:
            return result
        if type(result) is dict:
            result.pop("nextCursor", None)
        return McpPaginatedPage(
            result=result,
            next_cursor=private_cursor,
        )

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        method_name = require_clean_nonblank(method, "method")
        await self._write_with_timeout(
            jsonrpc_notification_payload(method_name, params),
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
        except (BrokenPipeError, ConnectionResetError) as exc:
            await self._await_stderr_drain()
            await self._close_after_interrupted_transport_write()
            raise self._protocol_error("MCP stdio process closed stdout.") from exc
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
                sanitized_error: McpProtocolError | None = None
                try:
                    await self._handle_message(message)
                except Exception as exc:
                    sanitized_error = McpProtocolError(
                        self._secret_redactor.redact_text_bounded(
                            str(exc),
                            max_bytes=4096,
                        )
                    )
                finally:
                    message.clear()
                    message = {}
                if sanitized_error is not None:
                    raise sanitized_error from None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        finally:
            # The reader loop only exits once the transport is dead. Latch the
            # session closed so subsequent requests fast-fail immediately instead
            # of blocking for the full request timeout on a future no reader will
            # ever resolve.
            self._closed = True
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
        future = self._pending.get(message_id)
        if future is None or future.done():
            return
        # Detach the request result so the reader can clear its raw transport
        # frame before waiting for the next message.
        copy_error: McpProtocolError | None = None
        try:
            copied_message = copy_json_value(message, "MCP stdio response")
        except ValueError:
            copied_message = None
            copy_error = McpProtocolError("MCP response contained invalid portable JSON.")
        self._pending.pop(message_id, None)
        if copy_error is not None:
            future.set_exception(copy_error)
            return
        if type(copied_message) is not dict:
            raise AssertionError("MCP stdio response copy returned a non-object.")
        future.set_result(copied_message)

    async def _write_server_request_error(self, message: dict[str, Any]) -> None:
        request_id = message.get("id")
        method = message.get("method")
        method_name = method if isinstance(method, str) else "unknown"
        safe_method_name = self._secret_redactor.redact_text(method_name)
        await self._write_with_timeout(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": JSONRPC_METHOD_NOT_FOUND,
                    "message": f"Cayu does not support MCP server request: {safe_method_name}",
                },
            },
            timeout_message=f"MCP server request rejection for {safe_method_name} write timed out.",
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
            # A closed stdout usually means the child crashed on startup. Give
            # the stderr drain a moment to finish so the crash detail lands in
            # the captured tail, then attach it to the error.
            await self._await_stderr_drain()
            raise self._protocol_error("MCP stdio process closed stdout.")
        decoded_line = line.decode("utf-8", "replace")
        payload: Any = None
        protocol_error: str | None = None
        try:
            payload = json.loads(decoded_line)
        except ValueError:
            protocol_error = "MCP stdio process wrote invalid JSON."
        # Drop the transport document before structural validation raises. A
        # valid JSON value with an invalid JSON-RPC envelope remains untrusted
        # and may contain a resolved secret echoed by the server.
        line = b""
        decoded_line = ""
        if protocol_error is None:
            if type(payload) is not dict:
                protocol_error = "MCP JSON-RPC message must be an object."
            elif payload.get("jsonrpc") != "2.0":
                protocol_error = "MCP JSON-RPC message must use jsonrpc='2.0'."
        if protocol_error is not None:
            payload = None
            raise self._protocol_error(protocol_error)
        return payload

    async def _drain_stderr(self) -> None:
        stderr = self.process.stderr
        if stderr is None:
            return
        while True:
            chunk = await stderr.read(4096)
            if not chunk:
                return
            self._append_stderr(chunk)

    def _append_stderr(self, chunk: bytes) -> None:
        buffer = self._stderr_tail
        buffer.extend(chunk)
        capture_bytes = (
            DEFAULT_MCP_STDERR_CAPTURE_BYTES + self._secret_redactor.max_secret_utf8_bytes
        )
        overflow = len(buffer) - capture_bytes
        if overflow > 0:
            del buffer[:overflow]

    def _stderr_snapshot(self) -> str:
        if not self._stderr_tail:
            return ""
        tail = bytes(self._stderr_tail).decode("utf-8", "replace")
        redacted = self._secret_redactor.redact_text(tail)
        encoded = redacted.encode("utf-8", "replace")
        if len(encoded) > DEFAULT_MCP_STDERR_CAPTURE_BYTES:
            cut = len(encoded) - DEFAULT_MCP_STDERR_CAPTURE_BYTES
            marker = REDACTED_SECRET.encode()
            marker_start = encoded.rfind(marker, 0, cut + len(marker))
            if marker_start >= 0 and marker_start + len(marker) > cut:
                suffix = encoded[marker_start + len(marker) :]
                encoded = marker + suffix[-(DEFAULT_MCP_STDERR_CAPTURE_BYTES - len(marker)) :]
            else:
                encoded = encoded[cut:]
        return encoded.decode("utf-8", "ignore").strip()

    def _protocol_error(self, message: str) -> McpProtocolError:
        tail = self._stderr_snapshot()
        if not tail:
            return McpProtocolError(message)
        return McpProtocolError(f"{message} MCP server stderr (tail): {tail}")

    async def _await_stderr_drain(self) -> None:
        task = self._stderr_task
        if task is None or task.done():
            return
        with suppress(Exception):
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=DEFAULT_MCP_STDERR_DRAIN_GRACE_S,
            )


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
