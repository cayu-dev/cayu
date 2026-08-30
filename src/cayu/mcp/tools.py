from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha1
from threading import Lock
from typing import Any, NoReturn

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.mcp._jsonrpc import McpProtocolError
from cayu.mcp._transport import (
    credential_safe_mcp_fatal_signal,
    credential_safe_mcp_transport_failure,
    mcp_json_value_nesting_too_deep,
)
from cayu.mcp.base import (
    McpClient,
    McpInitializeResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
    _attach_mcp_session_cleanup_failure,
    _close_mcp_session_after_primary_failure,
    _credential_safe_mcp_cancellation,
    _mcp_server_advertises_tools_list_changed,
    _mcp_tool_private_contract_hash,
    _McpCallerCancellationBoundary,
    _McpToolDiscovery,
    _McpToolDispatchSignal,
    _retain_mcp_session_close_if_fenced,
    copy_mcp_server_spec,
)
from cayu.mcp.http import HttpMcpClient, HttpMcpSession
from cayu.mcp.stdio import StdioMcpClient, StdioMcpSession
from cayu.vaults import SecretRedactor

_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_UNSAFE_TOOL_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9_-]+")
_MAX_STRUCTURED_CONTENT_TEXT_BYTES = 20_000
_MAX_SERVER_INSTRUCTIONS_DESCRIPTION_CHARS = 1_000
_MAX_MCP_DISCOVERY_ERROR_BYTES = 4096
_MCP_SESSION_SOURCE_ATTRIBUTE = "_cayu_internal_mcp_tool_source_v1"
_MCP_SESSION_SOURCE_BIND_LOCK = Lock()
_MISSING_MCP_SESSION_SOURCE = object()
_MCP_LIST_CHANGED_COALESCE_DELAY_S = 0.05


class McpToolsetRefreshState(StrEnum):
    """Live dispatch state for one refreshable MCP source."""

    READY = "ready"
    DIRTY = "dirty"
    REFRESHING = "refreshing"
    QUARANTINED = "quarantined"
    CLOSED = "closed"


class McpToolsetUnavailable(McpProtocolError):
    """Raised when an MCP snapshot no longer has live dispatch authority."""


class McpToolsetRefreshBlocked(RuntimeError):
    """Raised when application policy rejects a refreshed MCP manifest."""


@dataclass(frozen=True, slots=True)
class McpToolsetManifestDiff:
    """Bounded application-visible difference between two MCP snapshots."""

    server_changed: bool
    added_tools: tuple[str, ...] = ()
    removed_tools: tuple[str, ...] = ()
    changed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.server_changed) is not bool:
            raise TypeError("server_changed must be a bool.")
        for field_name in ("added_tools", "removed_tools", "changed_tools"):
            values = getattr(self, field_name)
            if type(values) is not tuple:
                raise TypeError(f"{field_name} must be a tuple.")
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{field_name} must be unique and sorted.")
            for value in values:
                require_clean_nonblank(value, field_name)
        categorized_names = self.added_tools + self.removed_tools + self.changed_tools
        if len(categorized_names) != len(set(categorized_names)):
            raise ValueError("MCP manifest tool changes must have disjoint categories.")

    @property
    def changed(self) -> bool:
        return bool(
            self.server_changed or self.added_tools or self.removed_tools or self.changed_tools
        )

    def policy_input(self) -> dict[str, Any]:
        """Return a detached value accepted by ``McpManifestPolicy``."""

        return {
            "server_changed": self.server_changed,
            "added_tools": list(self.added_tools),
            "removed_tools": list(self.removed_tools),
            "changed_tools": list(self.changed_tools),
        }


@dataclass(frozen=True, slots=True)
class McpToolsetRefreshResult:
    """Accepted result of one trigger-independent MCP catalogue refresh."""

    toolset: McpToolset
    status: str
    previous_generation: int
    generation: int
    previous_manifest_hash: str
    manifest_hash: str
    diff: McpToolsetManifestDiff
    policy_action: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.toolset, McpToolset):
            raise TypeError("toolset must be a McpToolset.")
        if self.status not in {"accepted", "unchanged"}:
            raise ValueError("status must be accepted or unchanged.")
        if type(self.previous_generation) is not int or self.previous_generation < 1:
            raise ValueError("previous_generation must be a positive integer.")
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("generation must be a positive integer.")
        if self.status == "unchanged" and self.generation != self.previous_generation:
            raise ValueError("An unchanged refresh cannot advance the source generation.")
        if self.status == "accepted" and self.generation != self.previous_generation + 1:
            raise ValueError("An accepted refresh must advance the source generation once.")
        for field_name in ("previous_manifest_hash", "manifest_hash"):
            value = getattr(self, field_name)
            if (
                type(value) is not str
                or len(value) != 71
                or not value.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in value[7:])
            ):
                raise ValueError(f"{field_name} must be a SHA-256 identifier.")
        if not isinstance(self.diff, McpToolsetManifestDiff):
            raise TypeError("diff must be an McpToolsetManifestDiff.")
        if self.toolset.generation != self.generation:
            raise ValueError("toolset generation must match the refresh result.")
        if self.status == "unchanged" and self.diff.changed:
            raise ValueError("An unchanged refresh cannot report manifest changes.")
        if self.status == "accepted" and not self.diff.changed:
            raise ValueError("An accepted refresh must report a manifest change.")
        if self.status == "unchanged" and self.previous_manifest_hash != self.manifest_hash:
            raise ValueError("An unchanged refresh must retain its manifest hash.")
        if self.policy_action not in {None, "allow", "alert"}:
            raise ValueError("policy_action must be allow, alert, or None.")


class _McpToolSource:
    """Own one MCP session and the dispatch fence shared by immutable snapshots."""

    def __init__(self, session: McpSession) -> None:
        self.session = session
        self.lock = asyncio.Lock()
        self.generation = 1
        self.state = McpToolsetRefreshState.READY
        self.refresh_owner: object | None = None
        self.static_owners: set[object] = set()
        self._dirty_epoch = 0
        self._admitted_dirty_epoch = 0
        self._failed_dirty_epoch = 0
        self._refresh_idle = asyncio.Event()
        self._refresh_idle.set()
        self._notification_refresh: Callable[[], Awaitable[None]] | None = None
        self._notification_refresh_task: asyncio.Task[None] | None = None
        self._notification_handler_installed = False
        self._notification_continuity_handler_installed = False
        self._notification_refresh_ready = True
        self._notification_activation_dirty_epoch: int | None = None

    def claim_refresh_owner(
        self,
        owner: object,
        *,
        notification_refresh: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        if self.static_owners:
            raise ValueError(
                "An MCP toolset with static registrations cannot acquire refresh ownership."
            )
        if self.refresh_owner is not None and self.refresh_owner is not owner:
            raise ValueError("An MCP toolset can be refresh-owned by only one CayuApp.")
        if self.refresh_owner is owner:
            return
        self.refresh_owner = owner
        if notification_refresh is None or not _mcp_server_advertises_tools_list_changed(
            self.session.initialize_result
        ):
            return
        self._notification_refresh = notification_refresh
        try:
            continuity_installed = self.session._set_tools_list_changed_continuity_handler(
                self._observe_tools_list_changed_continuity
            )
            self._notification_continuity_handler_installed = continuity_installed
            self._notification_refresh_ready = True
            installed = self.session._set_tools_list_changed_handler(
                self._observe_tools_list_changed
            )
        except BaseException:
            self._clear_notification_refresh_owner()
            self.refresh_owner = None
            raise
        if installed:
            self._notification_handler_installed = True
            return
        self._clear_notification_refresh_owner()

    def release_refresh_owner(self, owner: object) -> None:
        if self.refresh_owner is owner:
            self._clear_notification_refresh_owner()
            self.refresh_owner = None

    def _clear_notification_refresh_owner(self) -> None:
        if self._notification_handler_installed:
            self.session._set_tools_list_changed_handler(None)
            self._notification_handler_installed = False
        if self._notification_continuity_handler_installed:
            self.session._set_tools_list_changed_continuity_handler(None)
            self._notification_continuity_handler_installed = False
        activation_dirty_epoch = self._notification_activation_dirty_epoch
        self._notification_activation_dirty_epoch = None
        if (
            activation_dirty_epoch is not None
            and self.state is McpToolsetRefreshState.DIRTY
            and self._dirty_epoch == activation_dirty_epoch
        ):
            self._dirty_epoch -= 1
            self.state = McpToolsetRefreshState.READY
        self._notification_refresh_ready = True
        self._notification_refresh = None
        task = self._notification_refresh_task
        self._notification_refresh_task = None
        if task is not None and not task.done():
            task.cancel()

    def _observe_tools_list_changed(self) -> None:
        """Fence the source synchronously, then coalesce refresh ownership."""

        if (
            self.refresh_owner is None
            or self._notification_refresh is None
            or self.state is McpToolsetRefreshState.CLOSED
        ):
            return
        self._dirty_epoch += 1
        if self.state is not McpToolsetRefreshState.REFRESHING:
            self.state = McpToolsetRefreshState.DIRTY
        if not self._notification_refresh_ready:
            return
        self._ensure_notification_refresh_task()

    def _observe_tools_list_changed_continuity(self, ready: bool) -> None:
        """Fence listener gaps and reconcile once continuity is re-established."""

        if type(ready) is not bool:
            raise TypeError("ready must be a bool.")
        if (
            self.refresh_owner is None
            or self._notification_refresh is None
            or self.state is McpToolsetRefreshState.CLOSED
        ):
            return
        if not ready:
            if self._notification_refresh_ready:
                self._dirty_epoch += 1
                if (
                    not self._notification_handler_installed
                    and self.state is McpToolsetRefreshState.READY
                ):
                    self._notification_activation_dirty_epoch = self._dirty_epoch
            self._notification_refresh_ready = False
            if self.state is not McpToolsetRefreshState.REFRESHING:
                self.state = McpToolsetRefreshState.DIRTY
            return
        if self._notification_refresh_ready:
            return
        self._notification_activation_dirty_epoch = None
        self._dirty_epoch += 1
        self._notification_refresh_ready = True
        if self.state is not McpToolsetRefreshState.REFRESHING:
            self.state = McpToolsetRefreshState.DIRTY
        self._ensure_notification_refresh_task()

    def _ensure_notification_refresh_task(self) -> None:
        task = self._notification_refresh_task
        if task is None or task.done():
            task = asyncio.create_task(self._run_notification_refresh())
            self._notification_refresh_task = task
            task.add_done_callback(self._notification_refresh_completed)

    def _notification_refresh_completed(self, task: asyncio.Task[None]) -> None:
        if self._notification_refresh_task is task:
            self._notification_refresh_task = None
        with suppress(BaseException):
            task.result()

    def _automatic_refresh_is_needed(self) -> bool:
        return (
            self.refresh_owner is not None
            and self._notification_refresh is not None
            and self._notification_refresh_ready
            and self.state is not McpToolsetRefreshState.CLOSED
            and self._dirty_epoch > self._admitted_dirty_epoch
            and self._dirty_epoch > self._failed_dirty_epoch
        )

    async def _run_notification_refresh(self) -> None:
        try:
            await asyncio.sleep(_MCP_LIST_CHANGED_COALESCE_DELAY_S)
            while self._automatic_refresh_is_needed():
                await self._refresh_idle.wait()
                if not self._automatic_refresh_is_needed():
                    return
                refresh = self._notification_refresh
                if refresh is None:
                    return
                attempted_epoch = self._dirty_epoch
                try:
                    await refresh()
                except asyncio.CancelledError:
                    return
                except BaseException:
                    if self._dirty_epoch > attempted_epoch:
                        continue
                    return
        finally:
            refresh = None

    def claim_static_owner(self, owner: object) -> bool:
        if self.refresh_owner is not None:
            raise ValueError(
                "A refresh-owned MCP toolset cannot also be registered through static tools."
            )
        if owner in self.static_owners:
            return False
        self.static_owners.add(owner)
        return True

    def release_static_owner(self, owner: object) -> None:
        self.static_owners.discard(owner)

    def require_refresh_owner(self, owner: object) -> None:
        if self.refresh_owner is not owner:
            raise ValueError("The CayuApp does not own refresh authority for this MCP toolset.")

    def dispatch_authority_is_current(self, generation: int) -> bool:
        """Return whether one immutable snapshot can still enter governed work."""

        return (
            self.state is McpToolsetRefreshState.READY
            and self._notification_refresh_ready
            and self.generation == generation
        )

    def registration_authority_is_current(self, generation: int) -> bool:
        """Allow configuration to finish while first-listener activation is fenced."""

        return self.generation == generation and (
            self.state is McpToolsetRefreshState.READY
            or (
                self.state is McpToolsetRefreshState.DIRTY
                and self._notification_activation_dirty_epoch is not None
                and not self._notification_refresh_ready
            )
        )

    async def begin_refresh(self, *, owner: object, expected_generation: int) -> int:
        async with self.lock:
            self.require_refresh_owner(owner)
            if self.state is McpToolsetRefreshState.CLOSED:
                raise McpToolsetUnavailable("MCP toolset is closed.")
            listener_failure = self.session._tools_list_changed_listener_failure_message()
            if listener_failure is not None:
                if type(listener_failure) is not str:
                    raise RuntimeError("MCP notification listener failure evidence is invalid.")
                safe_listener_failure = self.session.secret_redactor.redact_text_bounded(
                    listener_failure,
                    max_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                )
                listener_failure = ""
                raise McpProtocolError(safe_listener_failure)
            if self._notification_handler_installed and not self._notification_refresh_ready:
                raise McpToolsetUnavailable(
                    "MCP toolset notification continuity is not established."
                )
            if self.generation != expected_generation:
                raise McpToolsetUnavailable("MCP toolset refresh authority is stale.")
            if self.state is McpToolsetRefreshState.REFRESHING:
                raise McpToolsetUnavailable("MCP toolset refresh is already in progress.")
            dirty_epoch = self._dirty_epoch
            self.state = McpToolsetRefreshState.REFRESHING
            self._refresh_idle.clear()
            return dirty_epoch

    def quarantine_refresh(
        self,
        *,
        owner: object,
        expected_generation: int,
        expected_dirty_epoch: int,
    ) -> None:
        self.require_refresh_owner(owner)
        if (
            self.state is McpToolsetRefreshState.REFRESHING
            and self.generation == expected_generation
        ):
            if self._dirty_epoch != expected_dirty_epoch:
                self.state = McpToolsetRefreshState.DIRTY
            else:
                self.state = McpToolsetRefreshState.QUARANTINED
                self._failed_dirty_epoch = max(
                    self._failed_dirty_epoch,
                    expected_dirty_epoch,
                )
            self._refresh_idle.set()

    def require_refresh_current(
        self,
        *,
        owner: object,
        expected_generation: int,
        expected_dirty_epoch: int,
    ) -> None:
        self.require_refresh_owner(owner)
        if (
            self.state is not McpToolsetRefreshState.REFRESHING
            or self.generation != expected_generation
        ):
            raise McpToolsetUnavailable("MCP toolset refresh authority changed.")
        if self._dirty_epoch != expected_dirty_epoch:
            self.state = McpToolsetRefreshState.DIRTY
            self._refresh_idle.set()
            raise McpToolsetUnavailable("MCP toolset refresh was superseded by a newer signal.")

    async def finish_unchanged(
        self,
        *,
        owner: object,
        expected_generation: int,
        expected_dirty_epoch: int,
        publish: Callable[[], Awaitable[None]],
    ) -> None:
        async with self.lock:
            self.require_refresh_current(
                owner=owner,
                expected_generation=expected_generation,
                expected_dirty_epoch=expected_dirty_epoch,
            )
            try:
                await publish()
            except BaseException:
                if self.state is McpToolsetRefreshState.REFRESHING:
                    self.state = McpToolsetRefreshState.QUARANTINED
                    self._failed_dirty_epoch = max(
                        self._failed_dirty_epoch,
                        expected_dirty_epoch,
                    )
                    self._refresh_idle.set()
                raise
            self.require_refresh_current(
                owner=owner,
                expected_generation=expected_generation,
                expected_dirty_epoch=expected_dirty_epoch,
            )
            self._admitted_dirty_epoch = max(
                self._admitted_dirty_epoch,
                expected_dirty_epoch,
            )
            self.state = McpToolsetRefreshState.READY
            self._refresh_idle.set()

    async def publish_refresh(
        self,
        *,
        owner: object,
        expected_generation: int,
        generation: int,
        expected_dirty_epoch: int,
        publish: Callable[[], Awaitable[None]],
    ) -> None:
        async with self.lock:
            self.require_refresh_current(
                owner=owner,
                expected_generation=expected_generation,
                expected_dirty_epoch=expected_dirty_epoch,
            )
            if generation != expected_generation + 1:
                raise McpToolsetUnavailable("MCP toolset refresh authority changed.")
            try:
                await publish()
            except BaseException:
                if self.state is McpToolsetRefreshState.REFRESHING:
                    self.state = McpToolsetRefreshState.QUARANTINED
                    self._failed_dirty_epoch = max(
                        self._failed_dirty_epoch,
                        expected_dirty_epoch,
                    )
                    self._refresh_idle.set()
                raise
            self.require_refresh_current(
                owner=owner,
                expected_generation=expected_generation,
                expected_dirty_epoch=expected_dirty_epoch,
            )
            self.generation = generation
            self._admitted_dirty_epoch = max(
                self._admitted_dirty_epoch,
                expected_dirty_epoch,
            )
            self.state = McpToolsetRefreshState.READY
            self._refresh_idle.set()

    async def call_tool(
        self,
        *,
        generation: int,
        name: str,
        arguments: dict[str, Any],
    ) -> McpToolResult:
        call_name = name
        call_arguments = arguments
        name = ""
        arguments = {}
        call_task: asyncio.Task[McpToolResult] | None = None
        dispatch_signal: _McpToolDispatchSignal | None = None
        try:
            async with self.lock:
                if (
                    self.state is not McpToolsetRefreshState.READY
                    or not self._notification_refresh_ready
                ):
                    raise McpToolsetUnavailable("MCP toolset is not ready for dispatch.")
                if self.generation != generation:
                    raise McpToolsetUnavailable("MCP toolset snapshot is stale.")
                dispatch_signal = _McpToolDispatchSignal()
                call_task = asyncio.create_task(
                    self.session._call_tool_with_dispatch_signal(
                        call_name,
                        call_arguments,
                        dispatch_signal=dispatch_signal,
                    )
                )
                call_name = ""
                call_arguments = {}
                cancellation_boundary = _McpCallerCancellationBoundary()
                caller_cancellation: asyncio.CancelledError | None = None
                try:
                    await cancellation_boundary.checkpoint()
                    done, _pending = await asyncio.wait(
                        (call_task, dispatch_signal.future),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                except asyncio.CancelledError as cancellation:
                    if cancellation_boundary.caller_cancelled():
                        caller_cancellation = cancellation
                    else:
                        raise
                except BaseException:
                    call_task.cancel()
                    with suppress(BaseException):
                        await call_task
                    raise
                if caller_cancellation is not None:
                    settlement = _raise_caller_cancellation_after_mcp_call_settles(
                        call_task,
                        caller_cancellation,
                        redactor=self.session.secret_redactor,
                    )
                    caller_cancellation = None
                    try:
                        await settlement
                    finally:
                        settlement = None
                    raise AssertionError("MCP cancellation settlement returned unexpectedly.")
                if call_task in done:
                    return call_task.result()
            caller_cancellation = None
            try:
                return await call_task
            except asyncio.CancelledError as cancellation:
                if cancellation_boundary.caller_cancelled():
                    caller_cancellation = cancellation
                else:
                    raise
            if caller_cancellation is None:
                raise AssertionError("MCP caller cancellation evidence was lost.")
            settlement = _raise_caller_cancellation_after_mcp_call_settles(
                call_task,
                caller_cancellation,
                redactor=self.session.secret_redactor,
            )
            caller_cancellation = None
            try:
                await settlement
            finally:
                settlement = None
            raise AssertionError("MCP cancellation settlement returned unexpectedly.")
        finally:
            if dispatch_signal is not None:
                dispatch_signal.close()
            dispatch_signal = None
            call_task = None
            call_name = ""
            call_arguments = {}

    async def close(self) -> None:
        if self._notification_handler_installed:
            self.session._set_tools_list_changed_handler(None)
            self._notification_handler_installed = False
        if self._notification_continuity_handler_installed:
            self.session._set_tools_list_changed_continuity_handler(None)
        self._notification_continuity_handler_installed = False
        self._notification_activation_dirty_epoch = None
        self._notification_refresh_ready = True
        self._notification_refresh = None
        notification_task = self._notification_refresh_task
        self._notification_refresh_task = None
        if (
            notification_task is not None
            and notification_task is not asyncio.current_task()
            and not notification_task.done()
        ):
            notification_task.cancel()
        async with self.lock:
            self.state = McpToolsetRefreshState.CLOSED
            self._refresh_idle.set()
        if notification_task is not None and notification_task is not asyncio.current_task():
            with suppress(BaseException):
                await notification_task
        notification_task = None
        await self.session.close()


async def _raise_caller_cancellation_after_mcp_call_settles(
    call_task: asyncio.Task[McpToolResult],
    caller_cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor,
) -> NoReturn:
    """Settle one cancelled child while keeping caller cancellation authoritative."""

    call_task.cancel()
    while not call_task.done():
        cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            await cancellation_boundary.checkpoint()
            await asyncio.shield(call_task)
        except asyncio.CancelledError:
            if cancellation_boundary.caller_cancelled():
                continue
            if not call_task.done():
                continue
            break
        except BaseException:
            if not call_task.done():
                continue
            break

    settlement_failure: BaseException | None = None
    try:
        call_task.result()
    except asyncio.CancelledError:
        pass
    except BaseException as error:
        settlement_failure = credential_safe_mcp_transport_failure(
            error,
            redactor=redactor,
            context="MCP tool cancellation settlement failed",
            preserve_cause=True,
        )
        error = None

    safe_cancellation = _credential_safe_mcp_cancellation(
        caller_cancellation,
        redactor=redactor,
    )
    del caller_cancellation
    has_settlement_failure = settlement_failure is not None
    if settlement_failure is not None:
        _attach_mcp_session_cleanup_failure(safe_cancellation, settlement_failure)
    settlement_failure = None
    del call_task, redactor
    if has_settlement_failure:
        raise safe_cancellation
    raise safe_cancellation from None


def _mcp_tool_source_for_session(session: McpSession, *, generation: int) -> _McpToolSource:
    """Return the unique generation fence attached to one live session."""

    with _MCP_SESSION_SOURCE_BIND_LOCK:
        session_state = vars(session)
        existing = session_state.get(
            _MCP_SESSION_SOURCE_ATTRIBUTE,
            _MISSING_MCP_SESSION_SOURCE,
        )
        if existing is _MISSING_MCP_SESSION_SOURCE:
            source = _McpToolSource(session)
            object.__setattr__(session, _MCP_SESSION_SOURCE_ATTRIBUTE, source)
            return source
        if not isinstance(existing, _McpToolSource) or existing.session is not session:
            raise RuntimeError("MCP session source ownership state is invalid.")
        if existing.generation != generation:
            raise ValueError("A live MCP session cannot be wrapped as an older toolset generation.")
        if existing.state is McpToolsetRefreshState.CLOSED:
            raise McpToolsetUnavailable("A closed MCP session cannot create another toolset.")
        return existing


def _require_mcp_session_source(session: McpSession, source: _McpToolSource) -> None:
    with _MCP_SESSION_SOURCE_BIND_LOCK:
        if vars(session).get(_MCP_SESSION_SOURCE_ATTRIBUTE) is not source:
            raise RuntimeError("MCP session lost its immutable source ownership state.")


@dataclass(frozen=True, slots=True)
class _McpManifestToolEvidence:
    cayu_name: str
    mcp_name: str
    contract_hash: str


@dataclass(frozen=True, slots=True)
class _McpRefreshBindingEvidence:
    """Private contract evidence indexed by its sanitized public identity."""

    public_cayu_name: str
    contract_hash: str


@dataclass(frozen=True, slots=True)
class _McpManifestSnapshot:
    identity_is_explicit: bool
    identity: str
    manifest_hash: str
    server_hash: str
    tools: tuple[_McpManifestToolEvidence, ...]
    tool_count: int


@dataclass(frozen=True, slots=True)
class _McpAdapterBinding:
    """Immutable construction-time authority for one MCP adapter."""

    toolset: McpToolset
    mcp_name: str
    source_contract_hash: str
    manifest_mcp_name: str
    manifest_contract_hash: str


class McpToolAdapter(Tool):
    """Expose one MCP server tool as a Cayu tool."""

    def __init__(
        self,
        *,
        toolset: McpToolset,
        definition: McpToolDefinition,
        name: str | None = None,
    ) -> None:
        if not isinstance(toolset, McpToolset):
            raise TypeError("toolset must be an McpToolset.")
        if type(definition) is not McpToolDefinition:
            raise TypeError("definition must be an McpToolDefinition.")
        binding = toolset._bind_adapter_definition(definition)
        public_definition = _redact_tool_definition(
            definition,
            redactor=toolset.secret_redactor,
        )
        tool_name = name or mcp_cayu_tool_name(
            toolset.server.name,
            public_definition.name,
        )
        if not _TOOL_NAME_RE.fullmatch(tool_name):
            raise ValueError(
                "MCP Cayu tool names must contain 1-64 letters, numbers, underscores, or hyphens."
            )
        self.__binding = binding
        self.__mcp_manifest_hash = toolset.manifest_hash
        self.__server = toolset.server
        self.__definition = public_definition
        super().__init__(
            spec=ToolSpec(
                name=tool_name,
                description=_tool_description(toolset, self.__definition),
                input_schema=self.__definition.input_schema,
                parallel_safe=_mcp_tool_parallel_safe(self.__definition),
                effect=_mcp_tool_effect(self.__definition),
            )
        )

    @property
    def _manifest_binding(self) -> _McpAdapterBinding:
        """Return the immutable dispatch binding used by runtime admission."""

        return self.__binding

    def _dispatch_authority_is_current(self) -> bool:
        """Return whether this snapshot still has live source-generation authority."""

        toolset = self.__binding.toolset
        return toolset._refresh_source.dispatch_authority_is_current(toolset.generation)

    @property
    def toolset(self) -> McpToolset:
        return self.__binding.toolset

    @property
    def mcp_manifest_hash(self) -> str:
        return self.__mcp_manifest_hash

    @property
    def server(self) -> McpServerSpec:
        return self.__server.model_copy(deep=True)

    @property
    def definition(self) -> McpToolDefinition:
        return self.__definition.model_copy(deep=True)

    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        if type(args) is not dict:
            raise TypeError("MCP tool arguments must be an object.")
        call = self.__binding.toolset.call_tool(
            self.__binding.mcp_name,
            args,
        )
        args = {}
        try:
            result = await call
        except BaseException:
            call = None
            raise
        call = None
        redactor = self.__binding.toolset.secret_redactor
        mcp_content = result.content
        mcp_structured_content = result.structured_content
        if redactor.has_values:
            # Redact the complete server values before rendering can truncate a
            # secret across the model-visible structured-content byte boundary.
            mcp_content = _redact_mcp_content(result.content, redactor=redactor)
            mcp_structured_content = redactor.redact_json(result.structured_content)
        content = _mcp_tool_result_text(
            mcp_content,
            structured_content=mcp_structured_content,
        )
        if redactor.has_values:
            # A hostile MCP server can echo injected secrets (secret_env/secret_headers)
            # back through its result. Keep a final text pass as defense in depth.
            content = redactor.redact_text(content)
        return ToolResult(
            content=content,
            structured={
                "mcp_server": self.__server.name,
                "mcp_tool": self.__definition.name,
                "mcp_manifest_hash": self.__mcp_manifest_hash,
                "mcp_content": mcp_content,
                "mcp_structured_content": mcp_structured_content,
            },
            is_error=result.is_error,
        )


class McpToolset:
    """Persistent established MCP server connection plus Cayu tool adapters."""

    def __init__(
        self,
        *,
        server: McpServerSpec,
        session: McpSession,
        definitions: tuple[McpToolDefinition, ...],
    ) -> None:
        self.__initialize(
            server=server,
            session=session,
            definitions=definitions,
            private_contract_hashes=None,
            source=None,
            generation=1,
        )

    @classmethod
    def _from_discovery(
        cls,
        *,
        server: McpServerSpec,
        session: McpSession,
        definitions: tuple[McpToolDefinition, ...],
        private_contract_hashes: tuple[str, ...],
    ) -> McpToolset:
        toolset = object.__new__(cls)
        toolset.__initialize(
            server=server,
            session=session,
            definitions=definitions,
            private_contract_hashes=private_contract_hashes,
            source=None,
            generation=1,
        )
        return toolset

    @classmethod
    def _from_refresh(
        cls,
        *,
        server: McpServerSpec,
        session: McpSession,
        definitions: tuple[McpToolDefinition, ...],
        private_contract_hashes: tuple[str, ...],
        source: _McpToolSource,
        generation: int,
    ) -> McpToolset:
        toolset = object.__new__(cls)
        toolset.__initialize(
            server=server,
            session=session,
            definitions=definitions,
            private_contract_hashes=private_contract_hashes,
            source=source,
            generation=generation,
        )
        return toolset

    def __initialize(
        self,
        *,
        server: McpServerSpec,
        session: McpSession,
        definitions: tuple[McpToolDefinition, ...],
        private_contract_hashes: tuple[str, ...] | None,
        source: _McpToolSource | None,
        generation: int,
    ) -> None:
        if type(server) is not McpServerSpec:
            raise TypeError("server must be a McpServerSpec.")
        if not isinstance(session, McpSession):
            raise TypeError("session must be a McpSession.")
        if type(definitions) is not tuple or any(
            type(definition) is not McpToolDefinition for definition in definitions
        ):
            raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
        if type(generation) is not int or generation < 1:
            raise ValueError("generation must be a positive integer.")
        if source is None:
            if generation != 1:
                raise ValueError("A new MCP source must begin at generation 1.")
            resolved_source = _mcp_tool_source_for_session(
                session,
                generation=generation,
            )
        else:
            if not isinstance(source, _McpToolSource):
                raise TypeError("source must be an _McpToolSource.")
            if source.session is not session:
                raise ValueError("Refreshed MCP snapshots must retain the same session.")
            _require_mcp_session_source(session, source)
            if source.state is not McpToolsetRefreshState.REFRESHING:
                raise ValueError("A refreshed MCP snapshot requires an active source refresh.")
            if generation != source.generation + 1:
                raise ValueError("A refreshed MCP snapshot must be the next source generation.")
            resolved_source = source
        self.__source = resolved_source
        self.__generation = generation
        self.__session = resolved_source.session
        redactor = session.secret_redactor
        raw_definitions = tuple(definition.model_copy(deep=True) for definition in definitions)
        if private_contract_hashes is None:
            resolved_private_contract_hashes = tuple(
                _mcp_tool_private_contract_hash(definition) for definition in raw_definitions
            )
        else:
            if type(private_contract_hashes) is not tuple:
                raise TypeError("private_contract_hashes must be a tuple.")
            resolved_private_contract_hashes = private_contract_hashes
        if len(resolved_private_contract_hashes) != len(raw_definitions):
            raise ValueError("Private MCP contract evidence must match the tool definitions.")
        for contract_hash in resolved_private_contract_hashes:
            if (
                type(contract_hash) is not str
                or len(contract_hash) != 71
                or not contract_hash.startswith("sha256:")
                or any(character not in "0123456789abcdef" for character in contract_hash[7:])
            ):
                raise ValueError("Private MCP contract evidence must contain SHA-256 identifiers.")
        raw_server = server.model_copy(deep=True)
        raw_initialize_result = session.initialize_result
        self.__binding_server = raw_server
        self.__server = _redact_server_spec(raw_server, redactor=redactor)
        self.__initialize_result = _redact_initialize_result(
            raw_initialize_result,
            redactor=redactor,
        )
        self.__definitions = tuple(
            _redact_tool_definition(definition, redactor=redactor) for definition in raw_definitions
        )
        binding_manifest_hash = mcp_tool_manifest_hash(
            server=raw_server,
            initialize_result=raw_initialize_result,
            definitions=raw_definitions,
        )
        manifest_identity = mcp_tool_manifest_identity(
            server=raw_server,
        )
        binding_server_hash = mcp_tool_manifest_server_hash(
            server=raw_server,
            initialize_result=raw_initialize_result,
        )
        binding_manifest_tools = mcp_tool_manifest_tools(
            server=raw_server,
            definitions=raw_definitions,
        )
        self.__binding_snapshot = _McpManifestSnapshot(
            identity_is_explicit=raw_server.connection_id is not None,
            identity=manifest_identity,
            manifest_hash=binding_manifest_hash,
            server_hash=binding_server_hash,
            tools=tuple(
                _McpManifestToolEvidence(
                    cayu_name=entry["cayu_name"],
                    mcp_name=entry["mcp_name"],
                    contract_hash=entry["hash"],
                )
                for entry in binding_manifest_tools
            ),
            tool_count=len(raw_definitions),
        )
        source_tool_keys = [
            (entry.cayu_name, entry.mcp_name) for entry in self.__binding_snapshot.tools
        ]
        if len(source_tool_keys) != len(set(source_tool_keys)):
            raise ValueError("MCP tool definitions must not contain duplicate tools.")

        manifest_hash = mcp_tool_manifest_hash(
            server=self.__server,
            initialize_result=self.__initialize_result,
            definitions=self.__definitions,
        )
        manifest_server_hash = mcp_tool_manifest_server_hash(
            server=self.__server,
            initialize_result=self.__initialize_result,
        )
        manifest_tools = mcp_tool_manifest_tools(
            server=self.__server,
            definitions=self.__definitions,
        )
        self.__manifest_snapshot = _McpManifestSnapshot(
            identity_is_explicit=raw_server.connection_id is not None,
            identity=manifest_identity,
            manifest_hash=manifest_hash,
            server_hash=manifest_server_hash,
            tools=tuple(
                _McpManifestToolEvidence(
                    cayu_name=entry["cayu_name"],
                    mcp_name=entry["mcp_name"],
                    contract_hash=entry["hash"],
                )
                for entry in manifest_tools
            ),
            tool_count=len(self.__definitions),
        )
        self.__refresh_binding_evidence = tuple(
            sorted(
                (
                    _McpRefreshBindingEvidence(
                        public_cayu_name=mcp_cayu_tool_name(
                            self.__server.name,
                            public_definition.name,
                        ),
                        contract_hash=private_contract_hash,
                    )
                    for public_definition, private_contract_hash in zip(
                        self.__definitions,
                        resolved_private_contract_hashes,
                        strict=True,
                    )
                ),
                key=lambda entry: entry.public_cayu_name,
            )
        )
        self.__tools = tuple(
            McpToolAdapter(toolset=self, definition=definition) for definition in raw_definitions
        )
        _validate_unique_tool_names(list(self.__tools))

    @property
    def server(self) -> McpServerSpec:
        return self.__server.model_copy(deep=True)

    @property
    def session(self) -> McpSession:
        return self.__session

    @property
    def generation(self) -> int:
        """Immutable source generation represented by this toolset snapshot."""

        return self.__generation

    @property
    def definitions(self) -> tuple[McpToolDefinition, ...]:
        return tuple(definition.model_copy(deep=True) for definition in self.__definitions)

    @property
    def tools(self) -> tuple[McpToolAdapter, ...]:
        return self.__tools

    @property
    def refresh_state(self) -> McpToolsetRefreshState:
        """Return the source's current dispatch state."""

        return self.__source.state

    @property
    def _refresh_source(self) -> _McpToolSource:
        return self.__source

    async def _prepare_refresh(self) -> tuple[McpToolset, _McpToolDiscovery]:
        cancellation_boundary = _McpCallerCancellationBoundary()
        definitions: tuple[McpToolDefinition, ...] = ()
        discovery: _McpToolDiscovery | None = None
        try:
            await cancellation_boundary.checkpoint()
            discovery = await self.__session._discover_tools_for_toolset()
            definitions = discovery.definitions
            if self.__source.state is McpToolsetRefreshState.CLOSED:
                definitions = ()
                raise McpToolsetUnavailable("MCP toolset closed during refresh.")
            candidate = McpToolset._from_refresh(
                server=self.__binding_server,
                session=self.__session,
                definitions=definitions,
                private_contract_hashes=discovery.private_contract_hashes,
                source=self.__source,
                generation=self.__generation + 1,
            )
            prepared = (candidate, discovery)
            discovery = None
            return prepared
        except McpToolsetUnavailable:
            definitions = ()
            raise
        except asyncio.CancelledError as exc:
            definitions = ()
            if cancellation_boundary.caller_cancelled():
                safe_cancellation = _credential_safe_mcp_cancellation(
                    exc,
                    redactor=self.__session.secret_redactor,
                )
                raise safe_cancellation from None
            public_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=self.__session.secret_redactor,
                context="MCP tool refresh was cancelled unexpectedly",
                max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                preserve_cause=True,
            )
            raise public_error from None
        except TimeoutError as exc:
            definitions = ()
            if not self.__session.secret_redactor.has_values:
                raise
            public_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=self.__session.secret_redactor,
                context="MCP tool refresh failed",
                max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                preserve_cause=True,
            )
            raise public_error from None
        except (BaseExceptionGroup, Exception) as exc:
            definitions = ()
            if not self.__session.secret_redactor.has_values:
                raise
            public_error = credential_safe_mcp_transport_failure(
                exc,
                redactor=self.__session.secret_redactor,
                context="MCP tool refresh failed",
                max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                preserve_cause=True,
            )
            raise public_error from None
        except BaseException as fatal:
            definitions = ()
            public_error = credential_safe_mcp_fatal_signal(
                fatal,
                redactor=self.__session.secret_redactor,
                context="MCP tool refresh failed",
                max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
            )
            raise public_error from None
        finally:
            if discovery is not None:
                discovery.discard()

    def _bind_adapter_definition(self, definition: McpToolDefinition) -> _McpAdapterBinding:
        """Bind an adapter only to a definition advertised by this toolset."""

        entry = mcp_tool_manifest_tools(
            server=self.__binding_server,
            definitions=(definition.model_copy(deep=True),),
        )[0]
        matches = [
            candidate
            for candidate in self.__binding_snapshot.tools
            if candidate.mcp_name == entry["mcp_name"] and candidate.contract_hash == entry["hash"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "MCP adapters must bind exactly one definition advertised by their toolset."
            )
        public_definition = _redact_tool_definition(
            definition,
            redactor=self.secret_redactor,
        )
        public_entry = mcp_tool_manifest_tools(
            server=self.__server,
            definitions=(public_definition,),
        )[0]
        public_matches = [
            candidate
            for candidate in self.__manifest_snapshot.tools
            if candidate.mcp_name == public_entry["mcp_name"]
            and candidate.contract_hash == public_entry["hash"]
        ]
        if len(public_matches) != 1:
            raise ValueError("MCP adapters must map to exactly one sanitized manifest definition.")
        return _McpAdapterBinding(
            toolset=self,
            mcp_name=matches[0].mcp_name,
            source_contract_hash=matches[0].contract_hash,
            manifest_mcp_name=public_matches[0].mcp_name,
            manifest_contract_hash=public_matches[0].contract_hash,
        )

    @property
    def _manifest_snapshot(self) -> _McpManifestSnapshot:
        """Return the immutable construction-time evidence used by runtime admission."""

        return self.__manifest_snapshot

    @property
    def _binding_manifest_snapshot(self) -> _McpManifestSnapshot:
        """Return private construction-time evidence for refresh comparison."""

        return self.__binding_snapshot

    @property
    def _refresh_binding_snapshot(self) -> tuple[_McpRefreshBindingEvidence, ...]:
        """Return private contract hashes keyed only by sanitized public names."""

        return self.__refresh_binding_evidence

    @property
    def manifest_identity_is_explicit(self) -> bool:
        """Whether the cached manifest identity came from an explicit connection ID."""

        return self.__manifest_snapshot.identity_is_explicit

    @property
    def manifest_identity(self) -> str:
        return self.__manifest_snapshot.identity

    @property
    def manifest_hash(self) -> str:
        return self.__manifest_snapshot.manifest_hash

    @property
    def manifest_server_hash(self) -> str:
        return self.__manifest_snapshot.server_hash

    @property
    def manifest_tools(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "cayu_name": entry.cayu_name,
                "mcp_name": entry.mcp_name,
                "hash": entry.contract_hash,
            }
            for entry in self.__manifest_snapshot.tools
        )

    @classmethod
    async def connect(
        cls,
        server: McpServerSpec,
        *,
        client: McpClient | None = None,
    ) -> McpToolset:
        authoritative_server = copy_mcp_server_spec(server)
        mcp_client = client if client is not None else _default_client_for(authoritative_server)
        session = await mcp_client.connect(copy_mcp_server_spec(authoritative_server))
        sanitized_error: BaseException | None = None
        discovery: _McpToolDiscovery | None = None
        discovery_cancellation_boundary = _McpCallerCancellationBoundary()
        try:
            try:
                await discovery_cancellation_boundary.checkpoint()
                discovery = await session._discover_tools_for_toolset()
                definitions = discovery.definitions
                toolset = cls._from_discovery(
                    server=authoritative_server,
                    session=session,
                    definitions=definitions,
                    private_contract_hashes=discovery.private_contract_hashes,
                )
                await discovery.commit()
                discovery = None
                return toolset
            except BaseException:
                if discovery is not None:
                    discovery.discard()
                    discovery = None
                raise
        except asyncio.CancelledError as exc:
            if not discovery_cancellation_boundary.caller_cancelled():
                public_error = credential_safe_mcp_transport_failure(
                    exc,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery was cancelled unexpectedly",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                    preserve_cause=True,
                )
                if not _retain_mcp_session_close_if_fenced(
                    session,
                    primary_error=public_error,
                ):
                    cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                        session,
                        primary_error=public_error,
                        primary_context="MCP tool discovery failed",
                        cleanup_context="MCP tool discovery cleanup failed",
                    )
                    if cleanup_cancellation is not None:
                        sanitized_error = cleanup_cancellation
                if sanitized_error is None:
                    sanitized_error = public_error
                definitions = ()
            else:
                public_cancellation = _credential_safe_mcp_cancellation(
                    exc,
                    redactor=session.secret_redactor,
                )
                if not _retain_mcp_session_close_if_fenced(
                    session,
                    primary_error=public_cancellation,
                ):
                    cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                        session,
                        primary_error=public_cancellation,
                        primary_context="MCP tool discovery was cancelled",
                        cleanup_context="MCP tool discovery cleanup failed",
                    )
                    if cleanup_cancellation is not None:
                        sanitized_error = cleanup_cancellation
                        definitions = ()
                    else:
                        sanitized_error = public_cancellation
                        definitions = ()
                else:
                    sanitized_error = public_cancellation
                    definitions = ()
        except TimeoutError as exc:
            public_error: BaseException = exc
            if session.secret_redactor.has_values:
                public_error = credential_safe_mcp_transport_failure(
                    exc,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery failed",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                    preserve_cause=True,
                )
            if not _retain_mcp_session_close_if_fenced(
                session,
                primary_error=public_error,
            ):
                cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                    session,
                    primary_error=public_error,
                    primary_context="MCP tool discovery timed out",
                    cleanup_context="MCP tool discovery cleanup failed",
                )
                if cleanup_cancellation is not None:
                    sanitized_error = cleanup_cancellation
                    definitions = ()
            if sanitized_error is None:
                if public_error is exc:
                    raise
                sanitized_error = public_error
                definitions = ()
        except (BaseExceptionGroup, Exception) as exc:
            cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                session,
                primary_error=exc,
                primary_context="MCP tool discovery failed",
                cleanup_context="MCP tool discovery cleanup failed",
            )
            if cleanup_cancellation is not None:
                sanitized_error = cleanup_cancellation
                definitions = ()
            elif not session.secret_redactor.has_values:
                raise
            else:
                sanitized_error = credential_safe_mcp_transport_failure(
                    exc,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery failed",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                    preserve_cause=True,
                )
                definitions = ()
        except BaseException as fatal:
            cleanup_cancellation = await _close_mcp_session_after_primary_failure(
                session,
                primary_error=fatal,
                primary_context="MCP tool discovery failed",
                cleanup_context="MCP tool discovery cleanup failed",
            )
            if cleanup_cancellation is not None:
                sanitized_error = cleanup_cancellation
            else:
                sanitized_error = credential_safe_mcp_fatal_signal(
                    fatal,
                    redactor=session.secret_redactor,
                    context="MCP tool discovery failed",
                    max_message_bytes=_MAX_MCP_DISCOVERY_ERROR_BYTES,
                )
            definitions = ()
        if sanitized_error is not None:
            raise sanitized_error
        raise AssertionError("MCP tool discovery returned without a toolset or error.")

    @property
    def initialize_result(self) -> McpInitializeResult:
        return self.__initialize_result.model_copy(deep=True)

    @property
    def secret_redactor(self) -> SecretRedactor:
        """Redactor for secrets injected into this server's session (empty if none)."""
        return self.__session.secret_redactor

    @property
    def process_capability_evidence(self):
        """Configured process-lifecycle evidence for stdio-backed toolsets."""

        if isinstance(self.__session, StdioMcpSession):
            return self.__session.process_capability_evidence
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        # Built-in sessions own a bounded preflight before their defensive copy.
        # Third-party sessions have no common limits contract, so retain the
        # historical toolset-owned copy before handing arguments to extensions.
        call_name = name
        call_arguments = arguments
        name = ""
        arguments = {}
        preparation_error: BaseException | None = None
        if type(self.__session) not in {HttpMcpSession, StdioMcpSession}:
            # Built-in sessions combine nesting and byte validation in their
            # bounded transport preflight. Custom sessions have no shared size
            # contract, so retain the adapter's historical validation boundary.
            if mcp_json_value_nesting_too_deep(call_arguments):
                call_name = ""
                call_arguments = {}
                raise McpProtocolError(
                    "MCP tool arguments exceeded the supported JSON nesting."
                ) from None
            try:
                call_arguments = copy_json_value(call_arguments, "arguments")
            except (RecursionError, TypeError, ValueError) as error:
                preparation_error = credential_safe_mcp_transport_failure(
                    error,
                    redactor=self.secret_redactor,
                    context="MCP tool arguments were invalid",
                )
                error = None
            if preparation_error is not None:
                call_name = ""
                call_arguments = {}
                raise preparation_error from None
            if type(call_arguments) is not dict:
                call_name = ""
                call_arguments = {}
                raise TypeError("MCP tool arguments must be an object.")
        call = self.__source.call_tool(
            generation=self.__generation,
            name=call_name,
            arguments=call_arguments,
        )
        call_name = ""
        call_arguments = {}
        try:
            return await call
        except BaseException:
            call = None
            raise
        finally:
            call = None

    async def close(self) -> None:
        await self.__source.close()


async def connect_mcp_toolset(
    server: McpServerSpec,
    *,
    client: McpClient | None = None,
) -> McpToolset:
    """Connect to one MCP server and return its initialized toolset."""

    return await McpToolset.connect(server, client=client)


def mcp_toolset_manifest_diff(
    previous: McpToolset,
    current: McpToolset,
) -> McpToolsetManifestDiff:
    """Compare two immutable snapshots of the same MCP source."""

    if not isinstance(previous, McpToolset) or not isinstance(current, McpToolset):
        raise TypeError("MCP manifest diff requires two McpToolset snapshots.")
    if previous._refresh_source is not current._refresh_source:
        raise ValueError("MCP manifest diff snapshots must share one source session.")
    if current.generation != previous.generation + 1:
        raise ValueError("MCP manifest diff requires the next source generation.")

    previous_tools = {
        entry.cayu_name: entry.contract_hash for entry in previous._manifest_snapshot.tools
    }
    current_tools = {
        entry.cayu_name: entry.contract_hash for entry in current._manifest_snapshot.tools
    }
    added_keys = current_tools.keys() - previous_tools.keys()
    removed_keys = previous_tools.keys() - current_tools.keys()
    shared_keys = current_tools.keys() & previous_tools.keys()
    changed_keys = {key for key in shared_keys if current_tools[key] != previous_tools[key]}

    # Binding evidence is intentionally private because it can contain values
    # redacted from the public snapshot. A private-only change still needs to
    # trip manifest policy, but its raw identity must not cross the boundary.
    previous_binding = {
        entry.public_cayu_name: entry.contract_hash for entry in previous._refresh_binding_snapshot
    }
    current_binding = {
        entry.public_cayu_name: entry.contract_hash for entry in current._refresh_binding_snapshot
    }
    binding_changed_keys = {
        key
        for key in previous_binding.keys() & current_binding.keys()
        if previous_binding[key] != current_binding[key]
    }
    changed_names = changed_keys | binding_changed_keys

    return McpToolsetManifestDiff(
        server_changed=(
            previous._manifest_snapshot.server_hash != current._manifest_snapshot.server_hash
            or previous._binding_manifest_snapshot.server_hash
            != current._binding_manifest_snapshot.server_hash
        ),
        added_tools=tuple(sorted(added_keys)),
        removed_tools=tuple(sorted(removed_keys)),
        changed_tools=tuple(sorted(changed_names)),
    )


def _redact_server_spec(
    server: McpServerSpec,
    *,
    redactor: SecretRedactor,
) -> McpServerSpec:
    payload = redactor.redact_json_values(server.model_dump(mode="json"))
    if type(payload) is not dict:
        raise AssertionError("MCP server redaction returned a non-object.")
    return McpServerSpec(**payload)


def _redact_initialize_result(
    result: McpInitializeResult,
    *,
    redactor: SecretRedactor,
) -> McpInitializeResult:
    payload = redactor.redact_json_values(
        result.model_dump(mode="json"),
        preserve_string_fields={"protocol_version"},
    )
    if type(payload) is not dict:
        raise AssertionError("MCP initialize-result redaction returned a non-object.")
    return McpInitializeResult(**payload)


def _redact_tool_definition(
    definition: McpToolDefinition,
    *,
    redactor: SecretRedactor,
) -> McpToolDefinition:
    payload = redactor.redact_json_values(definition.model_dump(mode="json"))
    if type(payload) is not dict:
        raise AssertionError("MCP tool-definition redaction returned a non-object.")
    return McpToolDefinition(**payload)


def _default_client_for(server: McpServerSpec) -> McpClient:
    """Pick the transport from the spec: a URL server uses HTTP, a command server stdio."""
    if server.url is not None:
        return HttpMcpClient()
    return StdioMcpClient()


def mcp_cayu_tool_name(server_name: str, tool_name: str) -> str:
    server_slug = _tool_name_slug(server_name, "server_name")
    tool_slug = _tool_name_slug(tool_name, "tool_name")
    candidate = f"mcp__{server_slug}__{tool_slug}"
    if len(candidate) <= 64:
        return candidate
    digest = sha1(candidate.encode("utf-8")).hexdigest()[:10]
    budget = 64 - len("mcp__") - len("__") - len("_") - len(digest)
    server_budget = max(8, budget // 3)
    tool_budget = max(8, budget - server_budget)
    return f"mcp__{server_slug[:server_budget]}__{tool_slug[:tool_budget]}_{digest}"


def mcp_tool_manifest_hash(
    *,
    server: McpServerSpec,
    initialize_result: McpInitializeResult,
    definitions: tuple[McpToolDefinition, ...],
) -> str:
    """Return a stable hash of the MCP tool contract Cayu exposes."""

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if type(initialize_result) is not McpInitializeResult:
        raise TypeError("initialize_result must be an McpInitializeResult.")
    if not isinstance(definitions, tuple):
        raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
    payload = _mcp_tool_manifest_payload(
        server=server,
        initialize_result=initialize_result,
        definitions=definitions,
    )
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def mcp_tool_manifest_identity(
    *,
    server: McpServerSpec,
    definitions: tuple[McpToolDefinition, ...] | None = None,
) -> str:
    """Return an opaque candidate identity for one MCP server connection.

    ``definitions`` remains accepted for source compatibility with the original
    helper, but manifest contents intentionally do not participate in identity.
    Only an explicit ``McpServerSpec.connection_id`` is authoritative for
    runtime manifest history. The server-name form exists solely so a rejected
    ID-less toolset can carry bounded, non-identifying audit evidence.
    """

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if definitions is not None:
        if not isinstance(definitions, tuple):
            raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
        if any(type(definition) is not McpToolDefinition for definition in definitions):
            raise TypeError("definitions must contain McpToolDefinition instances.")
    payload = {
        "schema": "cayu.mcp.connection_identity.v1",
        # Domain-separate authoritative explicit identities from the audit-only
        # server-name candidate. An explicit ID equal to the presentation name
        # must still establish a genuinely new authorization namespace.
        "identity_kind": ("connection_id" if server.connection_id is not None else "server_name"),
        "connection_id": (
            server.connection_id if server.connection_id is not None else server.name
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def mcp_tool_manifest_tools(
    *,
    server: McpServerSpec,
    definitions: tuple[McpToolDefinition, ...],
) -> tuple[dict[str, Any], ...]:
    """Return compact per-tool manifest entries for drift auditing."""

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if not isinstance(definitions, tuple):
        raise TypeError("definitions must be a tuple of McpToolDefinition instances.")
    entries: list[dict[str, Any]] = []
    for definition in definitions:
        if type(definition) is not McpToolDefinition:
            raise TypeError("definitions must contain McpToolDefinition instances.")
        cayu_name = mcp_cayu_tool_name(server.name, definition.name)
        payload = {
            "cayu_name": cayu_name,
            "mcp_name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
            "annotations": definition.annotations,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        entries.append(
            {
                "cayu_name": cayu_name,
                "mcp_name": definition.name,
                "hash": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
            }
        )
    entries.sort(key=lambda entry: (entry["cayu_name"], entry["mcp_name"]))
    return tuple(entries)


def mcp_tool_manifest_server_hash(
    *,
    server: McpServerSpec,
    initialize_result: McpInitializeResult,
) -> str:
    """Return a stable hash of MCP server metadata that affects the manifest."""

    if type(server) is not McpServerSpec:
        raise TypeError("server must be an McpServerSpec.")
    if type(initialize_result) is not McpInitializeResult:
        raise TypeError("initialize_result must be an McpInitializeResult.")
    payload = {
        "name": server.name,
        "protocol_version": initialize_result.protocol_version,
        "server_name": initialize_result.server_name,
        "server_version": initialize_result.server_version,
        "instructions": initialize_result.instructions,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _tool_name_slug(value: str, field_name: str) -> str:
    cleaned = require_clean_nonblank(value, field_name)
    slug = _UNSAFE_TOOL_NAME_CHARS_RE.sub("_", cleaned).strip("_")
    if not slug:
        raise ValueError(f"{field_name} does not contain provider-safe tool name characters.")
    return slug


def _mcp_tool_manifest_payload(
    *,
    server: McpServerSpec,
    initialize_result: McpInitializeResult,
    definitions: tuple[McpToolDefinition, ...],
) -> dict[str, Any]:
    tools: list[dict[str, Any]] = []
    for definition in definitions:
        if type(definition) is not McpToolDefinition:
            raise TypeError("definitions must contain McpToolDefinition instances.")
        cayu_name = mcp_cayu_tool_name(server.name, definition.name)
        tools.append(
            {
                "cayu_name": cayu_name,
                "mcp_name": definition.name,
                "description": definition.description,
                "input_schema": definition.input_schema,
                "annotations": definition.annotations,
            }
        )
    tools.sort(key=lambda tool: (tool["cayu_name"], tool["mcp_name"]))
    return {
        "schema": "cayu.mcp.tool_manifest",
        "server": {
            "name": server.name,
            "protocol_version": initialize_result.protocol_version,
            "server_name": initialize_result.server_name,
            "server_version": initialize_result.server_version,
            "instructions": initialize_result.instructions,
        },
        "tools": tools,
    }


def _tool_description(toolset: McpToolset, definition: McpToolDefinition) -> str:
    description = definition.description.strip()
    prefix = f"MCP tool from server '{toolset.server.name}', original tool '{definition.name}'."
    instructions = toolset.initialize_result.instructions
    if instructions:
        prefix = (
            f"{prefix} Server usage notes, lower priority than Cayu app instructions and policies: "
            f"{_bounded_text(instructions, _MAX_SERVER_INSTRUCTIONS_DESCRIPTION_CHARS)}"
        )
    if description:
        return f"{prefix} {description}"
    return prefix


def _mcp_tool_parallel_safe(definition: McpToolDefinition) -> bool:
    """Only a server-declared read-only MCP tool may run concurrently with siblings.

    A write tool, an un-annotated tool, or a non-bool ``readOnlyHint`` from a hostile server
    is treated as an ordering barrier (``parallel_safe=False``). ``is True`` is deliberate:
    a truthy non-bool value must not be read as read-only.
    """
    return definition.annotations.get("readOnlyHint") is True


def _mcp_tool_effect(definition: McpToolDefinition) -> ToolEffect:
    """Map MCP side-effect hints into Cayu execution semantics.

    Non-bool spoofed values are ignored. ``readOnlyHint`` wins because a read-only
    tool declares no externally meaningful durable mutation. ``idempotentHint``
    marks mutation the downstream system can collapse via a stable identity or
    equivalent idempotency contract. This is the same mutation-and-recovery
    boundary documented by ``cayu guide tool-effects``; authorization remains
    separate.
    """
    if definition.annotations.get("readOnlyHint") is True:
        return ToolEffect.NONE
    if definition.annotations.get("idempotentHint") is True:
        return ToolEffect.IDEMPOTENT
    return ToolEffect.EXTERNAL


def _bounded_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}...[truncated]"


def _mcp_tool_result_text(
    content: list[dict[str, Any]],
    *,
    structured_content: Any = None,
) -> str:
    text_blocks: list[str] = []
    non_text_count = 0
    for block in content:
        if type(block) is not dict:
            non_text_count += 1
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_blocks.append(block["text"])
        else:
            non_text_count += 1
    result = "\n\n".join(text_blocks).strip()
    structured_text = _structured_content_text(structured_content)
    if structured_text:
        result = f"{result}\n\n{structured_text}".strip() if result else structured_text
    if non_text_count:
        note = f"[MCP returned {non_text_count} non-text content block(s).]"
        result = f"{result}\n\n{note}".strip() if result else note
    return result


def _redact_mcp_content(
    content: list[dict[str, Any]],
    *,
    redactor: SecretRedactor,
) -> list[dict[str, Any]]:
    """Redact MCP blocks while preserving the typed text envelope used for rendering."""

    redacted: list[dict[str, Any]] = []
    for block in content:
        if type(block) is not dict:
            raise AssertionError("MCP tool content must contain objects.")
        block_type = block.get("type")
        text = block.get("text")
        if block_type == "text" and type(text) is str:
            untrusted = {key: value for key, value in block.items() if key not in {"type", "text"}}
            redacted_untrusted = redactor.redact_json(untrusted)
            if type(redacted_untrusted) is not dict:
                raise AssertionError("MCP text block redaction returned a non-object.")
            redacted.append(
                {
                    "type": "text",
                    "text": redactor.redact_text(text),
                    **redacted_untrusted,
                }
            )
            continue
        redacted_block = redactor.redact_json(block)
        if type(redacted_block) is not dict:
            raise AssertionError("MCP content block redaction returned a non-object.")
        redacted.append(redacted_block)
    return redacted


def _structured_content_text(structured_content: Any) -> str:
    if structured_content is None:
        return ""
    encoded = json.dumps(
        structured_content,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    data = encoded.encode("utf-8")
    if len(data) <= _MAX_STRUCTURED_CONTENT_TEXT_BYTES:
        return f"Structured MCP content:\n{encoded}"
    truncated = data[:_MAX_STRUCTURED_CONTENT_TEXT_BYTES].decode("utf-8", errors="replace")
    return f"Structured MCP content:\n{truncated}\n\n[structured content truncated]"


def _validate_unique_tool_names(adapters: list[McpToolAdapter]) -> None:
    names = [adapter.name for adapter in adapters]
    if len(names) != len(set(names)):
        raise ValueError("Discovered MCP tools produced duplicate Cayu tool names.")
