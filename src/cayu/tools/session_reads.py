"""Scoped, read-only tools that let an agent look at earlier durable sessions.

Applications register ``ListSessionsTool`` and ``ReadSessionTool`` explicitly
and give each a ``SessionReadAccess`` policy. The policy is evaluated against
the calling session on every call; a session it does not admit is reported
exactly like a session that does not exist. Every read also applies the calling
run's resource-access scope, and the tools refuse to run without one unless the
application declares a single-tenant store with ``unscoped=True``.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from cayu._validation import copy_label_map, require_clean_nonblank
from cayu.messages import (
    CitationPart,
    FilePart,
    HostedToolCallPart,
    Message,
    MessageRole,
    PeerContentPart,
    StructuredOutputPart,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from cayu.resource_access import (
    _runtime_owned_store_access,
    current_binding,
    current_data_bounds,
    model_data_access,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.public_authority import SESSION_REFERENCE_FIELD_NAME
from cayu.session_metadata import SUMMARY_METADATA_KEY, TITLE_METADATA_KEY
from cayu.sessions._access_rules import _SessionAccessBounds
from cayu.sessions.access import SessionAccessDenied
from cayu.sessions.queries import (
    MAX_SESSION_LIST_CURSOR_BYTES,
    LabelSelectorOperator,
    LabelSelectorRequirement,
    SessionOrder,
    SessionQuery,
    encode_session_cursor,
)
from cayu.sessions.records import Session, SessionStatus, _require_bounded_session_id
from cayu.sessions.store import SessionStore
from cayu.sessions.transcript_queries import (
    LATEST_TRANSCRIPT_TEXT_MAX_CHARS,
    TranscriptTextReadLimitExceeded,
)
from cayu.tools._errors import (
    invalid_tool_arguments_result,
    reject_unknown_tool_arguments,
    structured_invalid_arguments,
    tool_argument_validation,
)
from cayu.tools._redaction import active_secret_redactor
from cayu.tools._terminal_bounds import (
    DEFAULT_RECORD_TOOL_TERMINAL_PAYLOAD_BYTES,
    DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES,
    bounded_terminal_result,
    tool_terminal_payload_bound,
)
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.vaults import SecretRedactor

DEFAULT_LIST_SESSIONS_LIMIT = 10
MAX_LIST_SESSIONS_LIMIT = 50
DEFAULT_SESSION_OUTCOME_CHARS = 500
MAX_SESSION_OUTCOME_CHARS = 4_000
MAX_SESSION_TEXT_FILTER_CHARS = 256
DEFAULT_READ_SESSION_LIMIT = 20
MAX_READ_SESSION_LIMIT = 50
DEFAULT_SESSION_MESSAGE_CHARS = 2_000
MAX_SESSION_MESSAGE_CHARS = 20_000
# Stored message bytes one read loads: per message, and for the whole window.
# A larger message is listed with its size instead of being loaded.
READ_SESSION_MAX_RECORD_BYTES = 1024 * 1024
READ_SESSION_MAX_WINDOW_BYTES = 4 * 1024 * 1024
# Store pages one listing call may read while the policy filters rows out.
MAX_LIST_SESSIONS_STORE_PAGES = 5
# Trailing transcript messages, and their stored bytes, searched for the final
# structured output.
STRUCTURED_OUTPUT_SCAN_MESSAGES = 8
STRUCTURED_OUTPUT_SCAN_MAX_BYTES = 512 * 1024

_LIST_ARGUMENTS = frozenset(
    {
        "agent_name",
        "labels",
        "status",
        "text",
        "created_after",
        "last_activity_before",
        "limit",
        "cursor",
        "max_outcome_chars",
    }
)
_READ_ARGUMENTS = frozenset({"session_id", "offset", "limit", "max_message_chars"})
_NOT_FOUND = "Session not found, or this session may not read it."
_LIST_CURSOR_PREFIX = "after."
_STRUCTURED_OUTPUT_NOTES = {
    "not_found_within_limit": (
        f"(no structured output in the last {STRUCTURED_OUTPUT_SCAN_MESSAGES} messages)"
    ),
    "too_large": (
        "(structured output search stopped: messages exceed the "
        f"{STRUCTURED_OUTPUT_SCAN_MAX_BYTES // 1024} KiB read limit)"
    ),
}
_CITATION_NOTE = (
    "Cite the session_id of every session you rely on. Session content records "
    "what happened then and may be out of date."
)


@runtime_checkable
class SessionReadAccess(Protocol):
    """Decide, on every call, which durable sessions a calling session may read.

    ``narrow`` restricts a listing query to the sessions ``caller`` may read, or
    returns ``None`` when it may read none. ``allows`` decides one loaded
    session. The tools apply ``allows`` to every listed session as well, so it
    is the authority; ``narrow`` should match it to avoid extra scans. Page
    cursors name only sessions returned to the caller. Return
    ``False`` for the caller's own session unless the policy means to show it.
    """

    async def narrow(self, caller: Session, query: SessionQuery) -> SessionQuery | None: ...

    async def allows(self, caller: Session, session: Session) -> bool: ...


class OwnTreeSessionAccess:
    """Sessions in the caller's root session tree: the root and its descendants."""

    __slots__ = ("_include_caller",)

    def __init__(self, *, include_caller: bool = False) -> None:
        if type(include_caller) is not bool:
            raise TypeError("include_caller must be a boolean.")
        self._include_caller = include_caller

    @property
    def include_caller(self) -> bool:
        return self._include_caller

    async def narrow(self, caller: Session, query: SessionQuery) -> SessionQuery | None:
        root_session_id = caller.invocation.root_session_id
        if query.root_session_id not in {None, root_session_id}:
            return None
        return query.model_copy(update={"root_session_id": root_session_id})

    async def allows(self, caller: Session, session: Session) -> bool:
        if session.id == caller.id and not self._include_caller:
            return False
        return session.invocation.root_session_id == caller.invocation.root_session_id


class MatchingSessionAccess:
    """Sessions run by the named agents and carrying every label, minus exclusions.

    ``agent_names`` and ``labels`` select sessions; at least one is required, so
    the policy never admits every session in the store. ``exclude_labels``
    removes sessions carrying any of its key/value pairs, for example to keep
    a verifier from reading the work it checks.
    """

    __slots__ = ("_agent_names", "_exclude_labels", "_include_caller", "_labels")

    def __init__(
        self,
        *,
        agent_names: Collection[str] = (),
        labels: Mapping[str, str] | None = None,
        exclude_labels: Mapping[str, str] | None = None,
        include_caller: bool = False,
    ) -> None:
        if isinstance(agent_names, str | bytes):
            raise TypeError("agent_names must be a collection of agent names.")
        if type(include_caller) is not bool:
            raise TypeError("include_caller must be a boolean.")
        selection = SessionQuery(agent_names=tuple(agent_names), labels=dict(labels or {}))
        if not selection.agent_names and not selection.labels:
            raise ValueError(
                "MatchingSessionAccess requires agent_names or labels; "
                "it never admits every session."
            )
        self._agent_names = selection.agent_names
        self._labels = dict(selection.labels)
        self._exclude_labels = copy_label_map(dict(exclude_labels or {}), "exclude_labels")
        self._include_caller = include_caller

    @property
    def agent_names(self) -> tuple[str, ...]:
        return self._agent_names

    @property
    def labels(self) -> dict[str, str]:
        return dict(self._labels)

    @property
    def exclude_labels(self) -> dict[str, str]:
        return dict(self._exclude_labels)

    @property
    def include_caller(self) -> bool:
        return self._include_caller

    async def narrow(self, caller: Session, query: SessionQuery) -> SessionQuery | None:
        agent_names = query.agent_names
        if self._agent_names:
            if query.agent_name is not None and query.agent_name not in self._agent_names:
                return None
            agent_names = tuple(
                name for name in agent_names or self._agent_names if name in self._agent_names
            )
            if not agent_names:
                return None
        labels = dict(query.labels)
        for key, value in self._labels.items():
            if labels.setdefault(key, value) != value:
                return None
        selectors = (
            *query.label_selectors,
            *(
                LabelSelectorRequirement(
                    key=key, operator=LabelSelectorOperator.NOT_IN, values=(value,)
                )
                for key, value in self._exclude_labels.items()
            ),
        )
        return query.model_copy(
            update={"agent_names": agent_names, "labels": labels, "label_selectors": selectors}
        )

    async def allows(self, caller: Session, session: Session) -> bool:
        if session.id == caller.id and not self._include_caller:
            return False
        if self._agent_names and session.agent_name not in self._agent_names:
            return False
        if any(session.labels.get(key) != value for key, value in self._labels.items()):
            return False
        return not any(
            session.labels.get(key) == value for key, value in self._exclude_labels.items()
        )


class _SessionReadTool(Tool):
    def __init__(
        self,
        session_store: SessionStore,
        access: SessionReadAccess,
        spec: ToolSpec,
        unscoped: bool,
    ) -> None:
        if type(unscoped) is not bool:
            raise TypeError("unscoped must be a boolean.")
        if not isinstance(session_store, SessionStore):
            raise TypeError(f"{type(self).__name__} requires a SessionStore.")
        if not isinstance(access, SessionReadAccess):
            raise TypeError(
                f"{type(self).__name__} requires a SessionReadAccess policy, such as "
                "OwnTreeSessionAccess() or MatchingSessionAccess(...)."
            )
        if (
            not session_store.supports_public_authority_aliases
            or session_store.public_authority_alias_codec is None
        ):
            raise ValueError(
                f"{type(self).__name__} returns opaque session aliases and needs a session "
                "store with a public authority alias keyring, for example "
                "CayuApp(public_authority_alias_keyring=...)."
            )
        if not session_store.can_register_public_authority_aliases:
            raise ValueError(
                f"{type(self).__name__} requires a writable session store for alias "
                "registration; read-only and diagnostic stores are unsupported."
            )
        self._session_store = session_store
        self._access = access
        self._unscoped = unscoped
        super().__init__(spec)

    async def _alias(self, caller: Session, session_id: str) -> str:
        """Return the calling session's stable opaque alias for ``session_id``.

        Aliases are scoped to the calling session: another session gets a
        different alias for the same session and cannot resolve this one.
        """

        codec = self._session_store.public_authority_alias_codec
        assert codec is not None
        alias = codec.encode(
            session_id, field_name=SESSION_REFERENCE_FIELD_NAME, session_id=caller.id
        )
        # Alias bookkeeping is Cayu-owned; it publishes no session content.
        async with _runtime_owned_store_access():
            await self._session_store.register_public_authority_alias(
                alias,
                field_name=SESSION_REFERENCE_FIELD_NAME,
                private_value=session_id,
                scope_session_id=caller.id,
            )
        return alias

    async def _resolve(self, caller: Session, alias: str) -> str | None:
        try:
            async with _runtime_owned_store_access():
                return await self._session_store.resolve_public_authority_alias(
                    alias, field_name=SESSION_REFERENCE_FIELD_NAME, scope_session_id=caller.id
                )
        except (TypeError, ValueError):
            return None

    async def _row(
        self,
        caller: Session,
        session: Session,
        *,
        bounds: _SessionAccessBounds | None,
        redactor: SecretRedactor,
        summary_chars: int,
        parents: dict[str, str | None],
    ) -> dict[str, Any]:
        """Project one visible session; a hidden parent reads like no parent."""

        parent_id = session.parent_session_id
        parent_alias = None
        if parent_id is not None:
            if parent_id not in parents:
                parent = await self._load(parent_id)
                parents[parent_id] = (
                    await self._alias(caller, parent.id)
                    if parent is not None and await self._allows(caller, parent, bounds)
                    else None
                )
            parent_alias = parents[parent_id]
        return _session_row(
            session,
            alias=await self._alias(caller, session.id),
            parent_alias=parent_alias,
            redactor=redactor,
            summary_chars=summary_chars,
        )

    async def _admit(self) -> tuple[ToolResult | None, _SessionAccessBounds | None]:
        """Resolve the run's resource-access bounds, or refuse without a scope.

        Call inside ``model_data_access`` so store reads apply the same bounds.
        """

        if current_binding() is None:
            if self._unscoped:
                return None, None
            return _scope_required(self.spec.name), None
        try:
            bounds = await current_data_bounds()
        except SessionAccessDenied:
            return _scope_denied(), None
        if type(bounds) is not _SessionAccessBounds:
            return _scope_denied(), None
        return None, bounds

    async def _load(self, session_id: str) -> Session | None:
        try:
            return await self._session_store.load(session_id)
        except SessionAccessDenied:
            return None

    async def _allows(
        self, caller: Session, session: Session, bounds: _SessionAccessBounds | None
    ) -> bool:
        # The store already applies these bounds; checking again keeps a custom
        # store or policy from widening the run's resource-access scope.
        if bounds is not None and not bounds.matches(session.labels):
            return False
        allowed = await self._access.allows(caller, session)
        if type(allowed) is not bool:
            raise TypeError("SessionReadAccess.allows must return a boolean.")
        return allowed

    async def _outcome(
        self,
        session: Session,
        *,
        max_chars: int,
        redactor: SecretRedactor,
        always_structured: bool = False,
    ) -> dict[str, Any] | None:
        """Final assistant text and, when asked or there is no text, the final structured output."""

        try:
            projection = await self._session_store.load_latest_transcript_text(
                session.id,
                role=MessageRole.ASSISTANT,
                max_chars=min(max_chars, LATEST_TRANSCRIPT_TEXT_MAX_CHARS),
            )
        except (KeyError, SessionAccessDenied):
            return None
        except (NotImplementedError, TranscriptTextReadLimitExceeded):
            return {"available": False}
        outcome: dict[str, Any] = {"available": True, "text": "", "text_truncated": False}
        if projection is not None and projection[0]:
            text, truncated = projection
            outcome["text"] = redactor.redact_cut_text(text, cut_tail=truncated)
            outcome["text_truncated"] = truncated
            if not always_structured:
                return outcome
        status, structured = await self._final_structured_output(session.id)
        outcome["structured_output_status"] = status
        if structured is not None:
            budget = _CharBudget(redactor, max_chars)
            outcome["structured_output_name"] = structured.name
            outcome["structured_output"] = budget.take(
                _compact_json(redactor.redact_json(structured.output))
            )
            outcome["structured_output_truncated"] = budget.truncated
        return outcome

    async def _final_structured_output(
        self, session_id: str
    ) -> tuple[str, StructuredOutputPart | None]:
        """Search the newest messages, newest first, within message and byte limits.

        All reads go through ``load_bounded_transcript_window``, so they apply
        the calling run's resource-access bounds exactly as the transcript
        window does. The status is ``found``, ``none`` (the whole transcript was
        searched), ``not_found_within_limit``, ``too_large`` (a newer message
        could not be read within the bytes allowed) or ``unavailable``.
        """

        try:
            # A one-byte bound measures the first row without loading it and
            # yields the append cursor.
            extent = await self._session_store.load_bounded_transcript_window(
                session_id, start_index=0, limit=1, max_record_bytes=1, max_total_bytes=1
            )
            if extent.cursor == 0:
                return "none", None
            start = max(extent.cursor - STRUCTURED_OUTPUT_SCAN_MESSAGES, 0)
            measured = await self._session_store.load_bounded_transcript_window(
                session_id,
                start_index=start,
                limit=STRUCTURED_OUTPUT_SCAN_MESSAGES,
                max_record_bytes=1,
                max_total_bytes=1,
            )
            # Select the newest suffix using sizes only; older tool output must
            # not consume the budget before the final answer is considered.
            remaining = STRUCTURED_OUTPUT_SCAN_MAX_BYTES
            selected = []
            byte_limited = False
            for record in reversed(measured.records):
                if record.stored_bytes > remaining:
                    byte_limited = True
                    break
                remaining -= record.stored_bytes
                selected.append(record)
            if not selected:
                if byte_limited:
                    return "too_large", None
                return ("none" if start == 0 else "not_found_within_limit"), None
            window = await self._session_store.load_bounded_transcript_window(
                session_id,
                start_index=selected[-1].index,
                limit=len(selected),
                max_record_bytes=STRUCTURED_OUTPUT_SCAN_MAX_BYTES,
                max_total_bytes=STRUCTURED_OUTPUT_SCAN_MAX_BYTES,
            )
        except (KeyError, NotImplementedError, SessionAccessDenied):
            return "unavailable", None
        for record in reversed(window.records):
            if record.message is None:
                return "too_large", None
            part = next(
                (
                    item
                    for item in reversed(record.message.content)
                    if type(item) is StructuredOutputPart
                ),
                None,
            )
            if part is not None:
                return "found", part
        if byte_limited:
            return "too_large", None
        return ("none" if start == 0 else "not_found_within_limit"), None


class ListSessionsTool(_SessionReadTool):
    """List bounded pages of earlier sessions the calling session may read.

    The tool refuses to run unless the calling run has a resource-access scope.
    Pass ``unscoped=True`` only for a single-tenant session store.
    """

    def __init__(
        self,
        session_store: SessionStore,
        *,
        access: SessionReadAccess,
        name: str = "list_sessions",
        description: str | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
        max_terminal_payload_bytes: int | None = None,
        unscoped: bool = False,
    ) -> None:
        super().__init__(
            session_store,
            access,
            ToolSpec(
                name=require_clean_nonblank(name, "name"),
                effect=ToolEffect.NONE,
                description=description
                or (
                    "List earlier sessions this session may read, newest first, with each "
                    "session's opaque id, agent, title, status, labels, timestamps, summary and "
                    "a short final outcome. Filter by agent, labels, status, time or "
                    "title text, and pass next_cursor back to continue. Use read_session "
                    "for a transcript. " + _CITATION_NOTE
                ),
                execution_profile_identity=execution_profile_identity,
                max_terminal_payload_bytes=tool_terminal_payload_bound(
                    max_terminal_payload_bytes,
                    default=DEFAULT_RECORD_TOOL_TERMINAL_PAYLOAD_BYTES,
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "agent_name": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Only sessions run by this agent.",
                        },
                        "labels": {
                            "type": "object",
                            "additionalProperties": {"type": "string", "minLength": 1},
                            "description": "Only sessions carrying every one of these labels.",
                        },
                        "status": {
                            "type": "string",
                            "enum": [status.value for status in SessionStatus],
                            "description": "Only sessions in this status.",
                        },
                        "text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_SESSION_TEXT_FILTER_CHARS,
                            "description": (
                                "Case-insensitive text matched against the title, session "
                                "id, agent name and labels."
                            ),
                        },
                        "created_after": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Only sessions created after this instant.",
                        },
                        "last_activity_before": {
                            "type": "string",
                            "format": "date-time",
                            "description": "Only sessions last active at or before this instant.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_LIST_SESSIONS_LIMIT,
                            "default": DEFAULT_LIST_SESSIONS_LIMIT,
                        },
                        "cursor": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_SESSION_LIST_CURSOR_BYTES,
                            "description": "next_cursor from the previous page.",
                        },
                        "max_outcome_chars": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": MAX_SESSION_OUTCOME_CHARS,
                            "default": DEFAULT_SESSION_OUTCOME_CHARS,
                            "description": (
                                "Characters of each summary and final outcome; 0 omits them."
                            ),
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            unscoped,
        )

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_LIST_ARGUMENTS)
            limit = _int_argument(
                args,
                "limit",
                default=DEFAULT_LIST_SESSIONS_LIMIT,
                minimum=1,
                maximum=MAX_LIST_SESSIONS_LIMIT,
            )
            max_outcome_chars = _int_argument(
                args,
                "max_outcome_chars",
                default=DEFAULT_SESSION_OUTCOME_CHARS,
                minimum=0,
                maximum=MAX_SESSION_OUTCOME_CHARS,
            )
            text = args.get("text")
            if text is not None and (
                type(text) is not str or len(text) > MAX_SESSION_TEXT_FILTER_CHARS
            ):
                raise ValueError(
                    f"text must be a string of at most {MAX_SESSION_TEXT_FILTER_CHARS} characters."
                )
            cursor = args.get("cursor")
            if cursor is not None and (
                type(cursor) is not str
                or not cursor.startswith(_LIST_CURSOR_PREFIX)
                or len(cursor) > MAX_SESSION_LIST_CURSOR_BYTES
            ):
                raise ValueError("cursor must be a next_cursor value from list_sessions.")
            query = SessionQuery(
                agent_name=args.get("agent_name"),
                labels=args.get("labels") or {},
                status=args.get("status"),
                q=None,  # Search only the redacted projection, never hidden stored text.
                created_after=_instant_argument(args, "created_after"),
                last_activity_before=_instant_argument(args, "last_activity_before"),
                limit=limit,
                order_by=SessionOrder.CREATED_AT_DESC,
            )
        async with model_data_access():
            return await self._list(
                ctx,
                query,
                limit=limit,
                cursor=cursor,
                text=text,
                max_outcome_chars=max_outcome_chars,
            )

    async def _list(
        self,
        ctx: ToolContext,
        query: SessionQuery,
        *,
        limit: int,
        cursor: str | None,
        text: str | None,
        max_outcome_chars: int,
    ) -> ToolResult:
        refusal, bounds = await self._admit()
        if refusal is not None:
            return refusal
        if bounds is not None and _protected_label_keys(bounds) & set(query.labels):
            # Tenancy comes from the run's scope, never from model arguments.
            return invalid_tool_arguments_result(
                ValueError("labels cannot filter on a key reserved by the resource-access scope.")
            )
        redactor = active_secret_redactor(ctx)
        caller = await self._load(ctx.session_id)
        store_cursor = None
        if caller is not None and cursor is not None:
            # A cursor names a session this caller was shown; anything else is invalid.
            after_id = await self._resolve(caller, cursor.removeprefix(_LIST_CURSOR_PREFIX))
            after = None if after_id is None else await self._load(after_id)
            if after is None:
                return invalid_tool_arguments_result(
                    ValueError("cursor is not a valid next_cursor for this session.")
                )
            store_cursor = encode_session_cursor(after, SessionOrder.CREATED_AT_DESC)
        # Exact filters must not become a second way to probe a registered
        # secret. A filter containing hidden text matches no public row.
        filters = {"agent_name": query.agent_name, "labels": dict(query.labels)}
        filters_visible = redactor.redact_json(filters) == filters
        narrowed = (
            None
            if caller is None or not filters_visible
            else await self._access.narrow(caller, query)
        )
        if narrowed is not None and type(narrowed) is not SessionQuery:
            raise TypeError("SessionReadAccess.narrow must return a SessionQuery or None.")
        visible: list[Session] = []
        more = scan_limited = False
        if caller is not None and narrowed is not None:
            visible, more, scan_limited = await self._visible_page(
                caller,
                narrowed,
                limit=limit,
                cursor=store_cursor,
                text=text,
                redactor=redactor,
                bounds=bounds,
            )
        sessions: list[dict[str, Any]] = []
        parents: dict[str, str | None] = {}
        for session in visible:
            assert caller is not None
            row = await self._row(
                caller,
                session,
                bounds=bounds,
                redactor=redactor,
                summary_chars=max_outcome_chars,
                parents=parents,
            )
            if max_outcome_chars:
                row["outcome"] = await self._outcome(
                    session, max_chars=max_outcome_chars, redactor=redactor
                )
            sessions.append(row)
        # The cursor names only a returned session, by its alias, never one the policy hid.
        next_cursor = (
            _LIST_CURSOR_PREFIX + sessions[-1]["session_id"] if more and sessions else None
        )
        return ToolResult(
            content=_render_session_list(sessions, next_cursor, scan_limited=scan_limited),
            structured={
                "sessions": sessions,
                "next_cursor": next_cursor,
                "limit": limit,
                "scan_limited": scan_limited,
            },
        )

    async def _visible_page(
        self,
        caller: Session,
        query: SessionQuery,
        *,
        limit: int,
        cursor: str | None,
        text: str | None,
        redactor: SecretRedactor,
        bounds: _SessionAccessBounds | None,
    ) -> tuple[list[Session], bool, bool]:
        """Collect up to ``limit`` allowed sessions from a bounded number of store pages.

        Returns the sessions, whether more may follow them, and whether the
        page budget ran out before the listing was exhausted.
        """

        visible: list[Session] = []
        store_cursor = cursor
        for _ in range(MAX_LIST_SESSIONS_STORE_PAGES):
            page = await self._session_store.list_sessions(
                query.model_copy(
                    update={
                        "limit": limit,
                        "offset": 0,
                        "cursor": store_cursor,
                        "include_total_count": False,
                        "order_by": SessionOrder.CREATED_AT_DESC,
                    }
                )
            )
            for index, session in enumerate(page.sessions):
                if not await self._allows(caller, session, bounds):
                    continue
                if text:
                    title = session.metadata.get(TITLE_METADATA_KEY)
                    labels = redactor.redact_json(dict(session.labels))
                    searchable = [
                        redactor.redact_text(session.agent_name),
                        redactor.redact_text(title) if type(title) is str else "",
                        *labels,
                        *labels.values(),
                    ]
                    if not any(text.casefold() in value.casefold() for value in searchable):
                        continue
                visible.append(session)
                if len(visible) == limit:
                    more = index + 1 < len(page.sessions) or page.next_cursor is not None
                    return visible, more, False
            if page.next_cursor is None:
                return visible, False, False
            store_cursor = page.next_cursor
        return visible, True, True


class ReadSessionTool(_SessionReadTool):
    """Read a bounded transcript window of one session the caller may read.

    The tool refuses to run unless the calling run has a resource-access scope.
    Pass ``unscoped=True`` only for a single-tenant session store.
    """

    def __init__(
        self,
        session_store: SessionStore,
        *,
        access: SessionReadAccess,
        name: str = "read_session",
        description: str | None = None,
        execution_profile_identity: ExecutionProfileBehaviorIdentity | None = None,
        max_terminal_payload_bytes: int | None = None,
        unscoped: bool = False,
    ) -> None:
        super().__init__(
            session_store,
            access,
            ToolSpec(
                name=require_clean_nonblank(name, "name"),
                effect=ToolEffect.NONE,
                description=description
                or (
                    "Read one earlier session this session may read: its details, final "
                    "outcome and a window of its transcript, including tool calls and "
                    "results with secrets masked. Messages are numbered from 0; pass "
                    "next_offset back as offset to continue. " + _CITATION_NOTE
                ),
                execution_profile_identity=execution_profile_identity,
                max_terminal_payload_bytes=tool_terminal_payload_bound(
                    max_terminal_payload_bytes,
                    default=DEFAULT_TEXT_TOOL_TERMINAL_PAYLOAD_BYTES,
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Opaque session id from list_sessions.",
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Index of the first transcript message to read.",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_READ_SESSION_LIMIT,
                            "default": DEFAULT_READ_SESSION_LIMIT,
                        },
                        "max_message_chars": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_SESSION_MESSAGE_CHARS,
                            "default": DEFAULT_SESSION_MESSAGE_CHARS,
                            "description": "Characters kept from each message.",
                        },
                    },
                    "required": ["session_id"],
                    "additionalProperties": False,
                },
            ),
            unscoped,
        )

    @bounded_terminal_result
    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        with tool_argument_validation():
            reject_unknown_tool_arguments(
                args, allowed=_READ_ARGUMENTS, required=frozenset({"session_id"})
            )
            session_id = args.get("session_id")
            if type(session_id) is not str:
                raise ValueError("session_id must be a string.")
            session_id = _require_bounded_session_id(session_id, "session_id")
            offset = _int_argument(args, "offset", default=0, minimum=0, maximum=2**53 - 1)
            limit = _int_argument(
                args,
                "limit",
                default=DEFAULT_READ_SESSION_LIMIT,
                minimum=1,
                maximum=MAX_READ_SESSION_LIMIT,
            )
            max_message_chars = _int_argument(
                args,
                "max_message_chars",
                default=DEFAULT_SESSION_MESSAGE_CHARS,
                minimum=1,
                maximum=MAX_SESSION_MESSAGE_CHARS,
            )
        async with model_data_access():
            return await self._read(
                ctx,
                session_id,
                offset=offset,
                limit=limit,
                max_message_chars=max_message_chars,
            )

    async def _read(
        self,
        ctx: ToolContext,
        session_id: str,
        *,
        offset: int,
        limit: int,
        max_message_chars: int,
    ) -> ToolResult:
        refusal, bounds = await self._admit()
        if refusal is not None:
            return refusal
        redactor = active_secret_redactor(ctx)
        caller = await self._load(ctx.session_id)
        target_id = None if caller is None else await self._resolve(caller, session_id)
        session = None if target_id is None else await self._load(target_id)
        if caller is None or session is None or not await self._allows(caller, session, bounds):
            return _not_found()
        try:
            window = await self._session_store.load_bounded_transcript_window(
                session.id,
                start_index=offset,
                limit=limit,
                max_record_bytes=READ_SESSION_MAX_RECORD_BYTES,
                max_total_bytes=READ_SESSION_MAX_WINDOW_BYTES,
            )
        except (KeyError, SessionAccessDenied):
            return _not_found()
        except NotImplementedError:
            return ToolResult(
                content="This session store cannot read transcripts within byte limits.",
                structured={"error": "transcript_unavailable"},
                is_error=True,
            )
        outcome = await self._outcome(
            session, max_chars=max_message_chars, redactor=redactor, always_structured=True
        )
        if outcome is None:
            return _not_found()
        messages = [
            _message_payload(record.index, record.message, redactor, max_message_chars)
            if record.message is not None
            else {
                "index": record.index,
                "role": record.role.value,
                "parts": [],
                "omitted": True,
                "stored_bytes": record.stored_bytes,
            }
            for record in window.records
        ]
        last = window.records[-1].index if window.records else None
        next_offset = last + 1 if last is not None and last + 1 < window.cursor else None
        row = await self._row(
            caller,
            session,
            bounds=bounds,
            redactor=redactor,
            summary_chars=max_message_chars,
            parents={},
        )
        row["outcome"] = outcome
        structured = {
            "session": row,
            "messages": messages,
            "offset": offset,
            "next_offset": next_offset,
            "total_messages": window.cursor,
            "max_message_chars": max_message_chars,
        }
        return ToolResult(content=_render_session_read(structured), structured=structured)


class _CharBudget:
    """Characters left for one message; text is redacted before it is cut."""

    __slots__ = ("redactor", "remaining", "truncated")

    def __init__(self, redactor: SecretRedactor, chars: int) -> None:
        self.redactor = redactor
        self.remaining = chars
        self.truncated = False

    def take(self, text: str) -> str:
        if self.remaining <= 0:
            self.truncated = self.truncated or bool(text)
            return ""
        redacted = self.redactor.redact_text(text)
        kept = redacted[: self.remaining]
        self.remaining -= len(kept)
        if len(kept) < len(redacted):
            self.truncated = True
        return kept


def _message_payload(
    index: int,
    message: Message,
    redactor: SecretRedactor,
    max_chars: int,
) -> dict[str, Any]:
    budget = _CharBudget(redactor, max_chars)
    parts: list[dict[str, Any]] = []
    omitted = 0
    for part in message.content:
        if type(part) is TextPart:
            parts.append({"type": "text", "text": budget.take(part.text)})
        elif type(part) is ToolCallPart:
            parts.append(
                {
                    "type": "tool_call",
                    "tool_name": part.tool_name,
                    "arguments": (
                        budget.take(_compact_json(redactor.redact_json(part.arguments)))
                        if part.arguments_state == "finalized"
                        else None
                    ),
                    "arguments_state": part.arguments_state,
                }
            )
        elif type(part) is ToolResultPart:
            payload: dict[str, Any] = {
                "type": "tool_result",
                "tool_name": part.tool_name,
                "is_error": part.is_error,
                "content": budget.take(part.content),
            }
            if part.structured is not None:
                payload["structured"] = budget.take(
                    _compact_json(redactor.redact_json(part.structured))
                )
            parts.append(payload)
        elif type(part) is StructuredOutputPart:
            parts.append(
                {
                    "type": "structured_output",
                    "name": part.name,
                    "output": budget.take(_compact_json(redactor.redact_json(part.output))),
                }
            )
        elif type(part) is PeerContentPart:
            parts.append({"type": "peer_content", "text": budget.take(part.text)})
        elif type(part) is FilePart:
            parts.append(
                {
                    "type": "file",
                    "filename": budget.take(str(part.attachment.get("filename", ""))),
                    "content_type": part.attachment.get("content_type"),
                }
            )
        elif type(part) is HostedToolCallPart:
            parts.append(
                {"type": "hosted_tool_call", "hosted_tool": part.hosted_tool, "status": part.status}
            )
        elif type(part) is CitationPart:
            parts.append(
                {
                    "type": "citation",
                    "url": budget.take(part.url),
                    "title": None if part.title is None else budget.take(part.title),
                }
            )
        else:
            # Reasoning and provider protocol state are not part of the record shown.
            omitted += 1
    payload = {"index": index, "role": message.role.value, "parts": parts}
    if budget.truncated:
        payload["truncated"] = True
    if omitted:
        payload["omitted_parts"] = omitted
    return payload


def _session_row(
    session: Session,
    *,
    alias: str,
    parent_alias: str | None,
    redactor: SecretRedactor,
    summary_chars: int,
) -> dict[str, Any]:
    title = session.metadata.get(TITLE_METADATA_KEY)
    summary = session.metadata.get(SUMMARY_METADATA_KEY)
    row: dict[str, Any] = {
        "session_id": alias,
        "parent_session_id": parent_alias,
        "agent_name": redactor.redact_text(session.agent_name),
        "title": redactor.redact_text(title) if type(title) is str else None,
        "status": session.status.value,
        "labels": redactor.redact_json(dict(session.labels)),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }
    if summary_chars:
        row["summary"] = (
            _CharBudget(redactor, summary_chars).take(summary) if type(summary) is str else None
        )
    return row


def _not_found() -> ToolResult:
    # The requested id is not echoed, so a result never repeats a foreign id.
    return ToolResult(content=_NOT_FOUND, structured={"error": "session_not_found"}, is_error=True)


def _scope_required(tool_name: str) -> ToolResult:
    return ToolResult(
        content=(
            f"{tool_name} needs a resource-access scope for this run, and none is active. "
            "Run the agent under a ResourceAccessPolicy binding so session reads stay "
            "inside one tenant, or construct the tool with unscoped=True only for a "
            "single-tenant session store."
        ),
        structured={"error": "resource_scope_required"},
        is_error=True,
    )


def _scope_denied() -> ToolResult:
    return ToolResult(
        content="The run's resource-access scope no longer admits session reads.",
        structured={"error": "resource_scope_denied"},
        is_error=True,
    )


def _protected_label_keys(bounds: _SessionAccessBounds) -> frozenset[str]:
    return bounds.admitted.protected_keys() | bounds.current.protected_keys()


def _render_session_list(
    sessions: list[dict[str, Any]], next_cursor: str | None, *, scan_limited: bool
) -> str:
    if not sessions:
        lines = ["No readable sessions on this page."]
    else:
        lines = [f"{len(sessions)} readable session(s):"]
        for row in sessions:
            lines.extend(_session_lines(row, bullet=True))
    if scan_limited:
        lines.append(
            "Stopped after scanning the most sessions one call reads; narrow the filters "
            "to see sessions further back."
        )
    if next_cursor is not None:
        lines.append(f'More sessions may match: call again with cursor="{next_cursor}".')
    return "\n".join(lines)


def _render_session_read(read: dict[str, Any]) -> str:
    row = read["session"]
    lines = _session_lines(row, bullet=False)
    messages = read["messages"]
    total = read["total_messages"]
    if messages:
        lines.append(
            f"Transcript messages {messages[0]['index']}-{messages[-1]['index']} of {total}:"
        )
    else:
        lines.append(f"No transcript messages at offset {read['offset']} ({total} in total).")
    for message in messages:
        lines.extend(_message_lines(message))
    if read["next_offset"] is not None:
        lines.append(f"More messages follow: call again with offset={read['next_offset']}.")
    return "\n".join(lines)


def _session_lines(row: dict[str, Any], *, bullet: bool) -> list[str]:
    line = (
        f"session_id={row['session_id']} agent={row['agent_name']} "
        f"status={row['status']} created_at={row['created_at']}"
    )
    if row.get("title"):
        line += f' title="{row["title"]}"'
    lines = [("- " if bullet else "") + line]
    indent = "  " if bullet else ""
    if row.get("parent_session_id"):
        lines.append(indent + "parent_session_id: " + row["parent_session_id"])
    if row.get("labels"):
        lines.append(indent + "labels: " + _labels_text(row["labels"]))
    if row.get("summary"):
        lines.append(indent + "summary: " + row["summary"])
    outcome = _outcome_text(row.get("outcome"))
    if outcome:
        lines.append(indent + "outcome: " + outcome)
    return lines


def _labels_text(labels: dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(labels.items()))


def _outcome_text(outcome: dict[str, Any] | None) -> str:
    if not outcome:
        return ""
    if not outcome.get("available"):
        return "(final outcome unavailable)"
    pieces: list[str] = []
    if outcome.get("text"):
        pieces.append(outcome["text"] + (" [truncated]" if outcome.get("text_truncated") else ""))
    if "structured_output" in outcome:
        name = outcome.get("structured_output_name")
        prefix = f"structured output {name}: " if name else "structured output: "
        truncated = " [truncated]" if outcome.get("structured_output_truncated") else ""
        pieces.append(prefix + outcome["structured_output"] + truncated)
    status = outcome.get("structured_output_status")
    if status in _STRUCTURED_OUTPUT_NOTES:
        pieces.append(_STRUCTURED_OUTPUT_NOTES[status])
    return " | ".join(pieces)


def _message_lines(message: dict[str, Any]) -> list[str]:
    head = f"[{message['index']}] {message['role']}:"
    if message.get("omitted"):
        return [
            f"{head} (message omitted: {message['stored_bytes']} stored bytes exceed the "
            f"{READ_SESSION_MAX_RECORD_BYTES // 1024} KiB per-message read limit)"
        ]
    pieces: list[str] = []
    for part in message["parts"]:
        kind = part["type"]
        if kind in {"text", "peer_content"}:
            pieces.append(part["text"])
        elif kind == "tool_call":
            arguments = part["arguments"]
            pieces.append(
                f"tool_call {part['tool_name']} "
                + (arguments if arguments is not None else "(arguments unavailable)")
            )
        elif kind == "tool_result":
            error = " error" if part["is_error"] else ""
            text = f"tool_result {part['tool_name']}{error}: {part['content']}"
            if part.get("structured"):
                text += f" structured={part['structured']}"
            pieces.append(text)
        elif kind == "structured_output":
            name = f" {part['name']}" if part["name"] else ""
            pieces.append(f"structured_output{name}: {part['output']}")
        elif kind == "file":
            pieces.append(f"file {part['filename']}")
        elif kind == "hosted_tool_call":
            pieces.append(f"{part['hosted_tool']} {part['status']}")
        elif kind == "citation":
            pieces.append(f"citation {part['url']}")
    if message.get("truncated"):
        pieces.append("[message truncated]")
    if not pieces:
        return [head]
    return [f"{head} {pieces[0]}", *(f"    {piece}" for piece in pieces[1:])]


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _int_argument(
    args: dict[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = args.get(name, default)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    return value


def _instant_argument(args: dict[str, Any], name: str) -> datetime | None:
    value = args.get(name)
    if value is None:
        return None
    if type(value) is not str:
        raise ValueError(f"{name} must be an ISO 8601 date-time string.")
    try:
        instant = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"{name} must be an ISO 8601 date-time string.") from None
    if instant.tzinfo is None:
        raise ValueError(f"{name} must include a timezone offset.")
    return instant


__all__ = [
    "ListSessionsTool",
    "MatchingSessionAccess",
    "OwnTreeSessionAccess",
    "ReadSessionTool",
    "SessionReadAccess",
]
