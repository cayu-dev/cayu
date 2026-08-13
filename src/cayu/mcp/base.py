from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from contextlib import suppress
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from cayu._exception_groups import exception_cause, set_exception_cause
from cayu._validation import (
    copy_durable_json_object,
    copy_durable_json_value,
    require_clean_nonblank_keys,
    require_durable_clean_nonblank,
    require_durable_nonblank,
    require_durable_text,
)
from cayu.mcp._exception_handoffs import (
    attach_mcp_session_close_task,
    copy_mcp_failure_handoffs,
    mcp_session_close_task,
)
from cayu.vaults import SecretRedactor, SecretRef, copy_secret_ref


class McpServerSpec(BaseModel):
    """Configuration for an external MCP server."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str
    # Stable, operator-assigned identity for this logical connection. Runtime
    # admission requires it whenever this server's tools are exposed to a model;
    # direct discovery/client use may leave it unset.
    connection_id: str | None = None
    command: list[str] | None = None
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    secret_env: dict[str, SecretRef] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    secret_headers: dict[str, SecretRef] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("secret_env", "secret_headers", mode="before")
    @classmethod
    def validate_secret_config_keys(cls, value, info):
        copied = require_clean_nonblank_keys(value, info.field_name)
        for key in copied:
            require_durable_text(key, f"{info.field_name} key")
        return copied

    @field_validator("secret_env", "secret_headers")
    @classmethod
    def copy_secret_config_data(cls, value):
        return {key: copy_secret_ref(ref) for key, ref in value.items()}

    @field_validator("env", "headers", "metadata", mode="before")
    @classmethod
    def copy_json_config_data(cls, value, info):
        copied = copy_durable_json_object(value, info.field_name)
        if info.field_name in {"env", "headers"}:
            require_clean_nonblank_keys(copied, info.field_name)
        return copied

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("connection_id", "url")
    @classmethod
    def validate_optional_nonblank_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("command")
    @classmethod
    def validate_command_items(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        return [
            require_durable_nonblank(item, f"command[{index}]") for index, item in enumerate(value)
        ]

    @model_validator(mode="after")
    def validate_transport(self) -> McpServerSpec:
        if bool(self.command) == bool(self.url):
            raise ValueError("MCP server must define exactly one of command or url.")
        return self

    @model_validator(mode="after")
    def validate_secret_config_collisions(self) -> McpServerSpec:
        env_collisions = sorted(set(self.env) & set(self.secret_env))
        if env_collisions:
            raise ValueError(
                f"MCP server env and secret_env declare the same variables: {env_collisions}"
            )
        header_names = {name.lower() for name in self.headers}
        header_collisions = sorted(
            name for name in self.secret_headers if name.lower() in header_names
        )
        if header_collisions:
            raise ValueError(
                f"MCP server headers and secret_headers declare the same headers: "
                f"{header_collisions}"
            )
        return self


def copy_mcp_server_spec(spec: McpServerSpec) -> McpServerSpec:
    """Revalidate and copy a public server spec before any connection side effect."""

    if type(spec) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    return McpServerSpec.model_validate(spec.model_dump(mode="python", warnings=False))


class McpInitializeResult(BaseModel):
    """Server metadata returned by MCP initialize."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    protocol_version: str
    server_name: str | None = None
    server_version: str | None = None
    instructions: str | None = None
    capabilities: dict[str, Any] = Field(default_factory=dict)

    @field_validator("protocol_version")
    @classmethod
    def validate_protocol_version(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("server_name", "server_version", "instructions")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, info.field_name)

    @field_validator("capabilities", mode="before")
    @classmethod
    def copy_capabilities(cls, value):
        return copy_durable_json_object(value, "capabilities")


class McpToolDefinition(BaseModel):
    """Tool definition advertised by an MCP server."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str) -> str:
        if type(value) is not str:
            raise TypeError("description must be a string.")
        return require_durable_text(value, "description")

    @field_validator("input_schema", "annotations", mode="before")
    @classmethod
    def copy_json_data(cls, value, info):
        return copy_durable_json_object(value, info.field_name)


class McpToolResult(BaseModel):
    """Result returned by an MCP tools/call request."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    content: list[dict[str, Any]] = Field(default_factory=list)
    structured_content: dict[str, Any] | None = None
    is_error: StrictBool = False

    @field_validator("content", "structured_content", mode="before")
    @classmethod
    def copy_json_data(cls, value, info):
        return copy_durable_json_value(value, info.field_name)


class McpResourceDefinition(BaseModel):
    """Resource definition advertised by an MCP server."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    uri: str
    name: str | None = None
    description: str | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("name", "description", "mime_type")
    @classmethod
    def validate_optional_strings(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_durable_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value):
        return copy_durable_json_object(value, "metadata")


class McpResourceResult(BaseModel):
    """Result returned by an MCP resources/read request."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    contents: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("contents", mode="before")
    @classmethod
    def copy_contents(cls, value):
        return copy_durable_json_value(value, "contents")


class McpSession(ABC):
    """Initialized connection to one MCP server."""

    # Default: no injected secrets, so nothing to redact. Concrete sessions that inject
    # secret_env/secret_headers set an instance-level redactor built from the resolved
    # values (see StdioMcpSession/HttpMcpSession).
    _secret_redactor: SecretRedactor = SecretRedactor()

    @property
    def secret_redactor(self) -> SecretRedactor:
        """Redactor for secrets injected into this session (empty if none).

        A hostile or buggy MCP server can echo injected secrets back through tool
        content, structured output, stderr, or protocol errors, so the toolset scrubs
        results with this before they reach model-visible context.
        """
        return self._secret_redactor

    @property
    @abstractmethod
    def initialize_result(self) -> McpInitializeResult:
        """Metadata returned by MCP initialize."""

    @abstractmethod
    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        """Return tools advertised by the server."""

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        """Call one server tool."""

    @abstractmethod
    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        """Return resources advertised by the server."""

    @abstractmethod
    async def read_resource(self, uri: str) -> McpResourceResult:
        """Read one server resource."""

    @abstractmethod
    async def close(self) -> None:
        """Close the server connection."""

    def _fence_before_retained_close(self) -> bool:
        """Synchronously reject new work before close continues in the background.

        This is an internal, fail-closed opt-in. Third-party sessions keep the safe
        default so a discovery failure is not returned until their close completes.
        """
        return False


class McpClient(ABC):
    """Factory for initialized MCP sessions."""

    @abstractmethod
    async def connect(self, server: McpServerSpec) -> McpSession:
        """Connect to one MCP server."""


_RETAINED_SESSION_CLOSE_TASKS: set[asyncio.Task[None]] = set()


class _McpCallerCancellationBoundary:
    """Distinguish current caller cancellation from child-only cancellation.

    ``Task.cancelling()`` is an absolute request count, so a request already
    pending when a boundary snapshots it is otherwise indistinguishable from a
    historical, already-delivered request. The real cancellation checkpoint
    below resolves that ambiguity before the owned await begins.
    """

    __slots__ = ("_caller_task", "_checkpoint_delivered", "_request_count")

    def __init__(self) -> None:
        self._caller_task = asyncio.current_task()
        self._checkpoint_delivered = False
        self._request_count = self._caller_task.cancelling() if self._caller_task is not None else 0

    async def checkpoint(self) -> None:
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            self._checkpoint_delivered = True
            raise
        self._request_count = self._caller_task.cancelling() if self._caller_task is not None else 0

    def caller_cancelled(self) -> bool:
        if self._checkpoint_delivered:
            return True
        return (
            self._caller_task is not None
            and asyncio.current_task() is self._caller_task
            and self._caller_task.cancelling() > self._request_count
        )


def _credential_safe_mcp_session_cleanup_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
    context: str,
) -> BaseException:
    # Import lazily: _transport imports _jsonrpc, which imports this base module.
    from cayu.mcp._transport import credential_safe_mcp_transport_failure

    return credential_safe_mcp_transport_failure(
        error,
        redactor=redactor,
        context=context,
        preserve_cause=True,
    )


def _attach_mcp_session_cleanup_failure(
    primary_error: BaseException,
    cleanup_error: BaseException,
) -> None:
    prior_cause = exception_cause(primary_error)
    if prior_cause is cleanup_error:
        return
    combined = (
        cleanup_error
        if prior_cause is None
        else BaseExceptionGroup(
            "MCP session cleanup failures.",
            [prior_cause, cleanup_error],
        )
    )
    set_exception_cause(primary_error, combined)


async def _await_mcp_session_cleanup_task(
    cleanup_task: asyncio.Task[None],
    *,
    redactor: SecretRedactor,
    context: str,
) -> None:
    """Await owned cleanup while keeping caller cancellation authoritative."""

    caller_cancellation: asyncio.CancelledError | None = None
    cleanup_failure: BaseException | None = None
    while True:
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            await asyncio.shield(cleanup_task)
            break
        except asyncio.CancelledError as error:
            received_caller_cancellation = cancellation_boundary.caller_cancelled()
            if received_caller_cancellation:
                if caller_cancellation is None:
                    caller_cancellation = error
                error = None
                if not cleanup_task.done():
                    continue
                try:
                    cleanup_task.result()
                except BaseException as task_error:
                    cleanup_failure = _credential_safe_mcp_session_cleanup_failure(
                        task_error,
                        redactor=redactor,
                        context=context,
                    )
                    task_error = None
                break
            cleanup_failure = _credential_safe_mcp_session_cleanup_failure(
                error,
                redactor=redactor,
                context=context,
            )
            error = None
            break
        except BaseException as error:
            cleanup_failure = _credential_safe_mcp_session_cleanup_failure(
                error,
                redactor=redactor,
                context=context,
            )
            error = None
            break
    del cleanup_task
    if caller_cancellation is not None:
        if cleanup_failure is not None:
            _attach_mcp_session_cleanup_failure(caller_cancellation, cleanup_failure)
        raise caller_cancellation
    if cleanup_failure is not None:
        raise cleanup_failure


async def _close_mcp_session_with_safe_failure(session: McpSession) -> None:
    safe_failure: BaseException | None = None
    try:
        await session.close()
    except BaseException as error:
        safe_failure = _credential_safe_mcp_session_cleanup_failure(
            error,
            redactor=session.secret_redactor,
            context="MCP retained session cleanup failed",
        )
        error = None
    if safe_failure is not None:
        raise safe_failure


def _retain_mcp_session_close(
    session: McpSession,
    *,
    primary_error: BaseException,
) -> asyncio.Task[None]:
    """Start failed-connect cleanup without delaying its public control signal.

    An inner transport boundary may already have attached a more exact settlement
    owner (for example, a cancellation-resistant stdio writer). Retain this close
    waiter as well, but never replace that exact handoff on the public error.
    """

    close_task = asyncio.create_task(_close_mcp_session_with_safe_failure(session))
    _RETAINED_SESSION_CLOSE_TASKS.add(close_task)
    if _mcp_session_close_task(primary_error) is None:
        _attach_mcp_session_cleanup_task(primary_error, close_task)

    def completed(task: asyncio.Task[None]) -> None:
        _RETAINED_SESSION_CLOSE_TASKS.discard(task)
        # This is the terminal observer for deliberately retained cleanup. Mixed
        # BaseExceptionGroups (for example, cancellation plus an ordinary close
        # failure) must not escape into the event loop as a second diagnostic.
        with suppress(BaseException):
            task.result()

    close_task.add_done_callback(completed)
    return close_task


def _attach_mcp_session_cleanup_task(
    primary_error: BaseException,
    cleanup_task: asyncio.Task[None],
) -> None:
    attach_mcp_session_close_task(primary_error, cleanup_task)


def _mcp_session_close_task(error: BaseException) -> asyncio.Task[None] | None:
    return mcp_session_close_task(error)


def _retain_mcp_session_close_if_fenced(
    session: McpSession,
    *,
    primary_error: BaseException,
) -> bool:
    """Retain close only after the session positively proves synchronous fencing."""

    # Fencing authority belongs to the concrete runtime class. A descendant may
    # add work or reuse entrances that an inherited proof does not cover. Invoke
    # the inherited hook so it can still fence entrances it does own, but do not
    # accept that result as proof for the descendant's complete session boundary.
    concrete_hook_owner = "_fence_before_retained_close" in type(session).__dict__
    try:
        fenced = session._fence_before_retained_close()
    except asyncio.CancelledError:
        # This hook is synchronous, so CancelledError cannot be delivery of a new
        # task cancellation. Treat an extension-raised signal as failed fencing
        # and finish close before returning the primary failure.
        return False
    except BaseException:
        # Extension fencing is only a synchronous proof probe. Any signal raised
        # by the probe means fencing was not proved; finish close through the
        # ordinary owned cleanup path before returning the primary failure.
        return False
    if not concrete_hook_owner or fenced is not True:
        return False
    _retain_mcp_session_close(session, primary_error=primary_error)
    return True


async def _close_mcp_session_after_primary_failure(
    session: McpSession,
    *,
    primary_error: BaseException,
    primary_context: str,
    cleanup_context: str,
) -> asyncio.CancelledError | None:
    """Finish unfenced cleanup and retain ordered, credential-safe diagnostics."""

    close_task = asyncio.create_task(_capture_mcp_session_close_failure(session))
    caller_cancellation: asyncio.CancelledError | None = None
    cleanup_failure: BaseException | None = None
    while True:
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            raw_cleanup_failure = await asyncio.shield(close_task)
            if raw_cleanup_failure is not None:
                cleanup_failure = _credential_safe_mcp_session_cleanup_failure(
                    raw_cleanup_failure,
                    redactor=session.secret_redactor,
                    context=cleanup_context,
                )
                raw_cleanup_failure = None
            break
        except asyncio.CancelledError as error:
            received_caller_cancellation = cancellation_boundary.caller_cancelled()
            if received_caller_cancellation:
                if caller_cancellation is None:
                    caller_cancellation = error
                error = None
                if not close_task.done():
                    continue
                raw_cleanup_failure = close_task.result()
                if raw_cleanup_failure is not None:
                    cleanup_failure = _credential_safe_mcp_session_cleanup_failure(
                        raw_cleanup_failure,
                        redactor=session.secret_redactor,
                        context=cleanup_context,
                    )
                    raw_cleanup_failure = None
                break
            cleanup_failure = _credential_safe_mcp_session_cleanup_failure(
                error,
                redactor=session.secret_redactor,
                context=cleanup_context,
            )
            error = None
            break
        except BaseException as error:
            cleanup_failure = _credential_safe_mcp_session_cleanup_failure(
                error,
                redactor=session.secret_redactor,
                context=cleanup_context,
            )
            error = None
            break

    if caller_cancellation is not None:
        safe_cancellation = _credential_safe_mcp_cancellation(
            caller_cancellation,
            redactor=session.secret_redactor,
        )
        safe_primary = _credential_safe_mcp_session_cleanup_failure(
            primary_error,
            redactor=session.secret_redactor,
            context=primary_context,
        )
        _attach_mcp_session_cleanup_failure(safe_cancellation, safe_primary)
        if cleanup_failure is not None:
            _attach_mcp_session_cleanup_failure(safe_cancellation, cleanup_failure)
        return safe_cancellation
    if cleanup_failure is not None:
        _attach_mcp_session_cleanup_failure(primary_error, cleanup_failure)
    return None


async def _capture_mcp_session_close_failure(
    session: McpSession,
) -> BaseException | None:
    """Capture extension process signals before they can escape an asyncio task."""

    try:
        await session.close()
    except BaseException as error:
        return error
    return None


def _credential_safe_mcp_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor,
) -> asyncio.CancelledError:
    from cayu.mcp._transport import _credential_safe_mcp_exception_argument_value

    try:
        args = BaseException.__dict__["args"].__get__(cancellation, BaseException)
    except BaseException:
        args = ()
    copied: list[Any] = []
    if type(args) is tuple:
        for value in args:
            copied.append(
                _credential_safe_mcp_exception_argument_value(
                    value,
                    redactor=redactor,
                    max_bytes=2048,
                    non_text_diagnostic="MCP operation cancelled",
                )
            )
    safe_cancellation = asyncio.CancelledError(*copied)
    copy_mcp_failure_handoffs(cancellation, safe_cancellation)
    cause = exception_cause(cancellation)
    if cause is not None and cause is not cancellation:
        safe_cause = _credential_safe_mcp_session_cleanup_failure(
            cause,
            redactor=redactor,
            context="MCP cancellation diagnostic",
        )
        set_exception_cause(safe_cancellation, safe_cause)
    return safe_cancellation
