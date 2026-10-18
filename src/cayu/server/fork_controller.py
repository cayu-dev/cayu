"""HTTP entrance for enrolled fork controllers.

Mount this router in a host application that authenticates its callers. The
host's ``authenticate`` callback names the subject; request bodies never select
the controller or carry its authority. ``ForkControllerBearerAuth`` is a stock
``authenticate`` callback that maps bearer tokens to subjects. Read routes use
the control plane's serializers and byte limits; the store applies the
controller's read limit to every read.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from contextlib import aclosing, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, ValidationError

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    json_utf8_size_within_limit,
    require_durable_clean_nonblank,
)
from cayu.applications import CayuApp
from cayu.approvals.actors import ResolutionActor, ResolutionActorSource
from cayu.budgets.base import BudgetLimit, BudgetReservation
from cayu.events import EventType
from cayu.fork_controllers import (
    FORK_CONTROLLER_READ_DEFAULT_BYTES,
    FORK_CONTROLLER_READ_MAX_BYTES,
    ForkController,
    ForkControllerAccess,
    ForkControllerDenied,
    ForkControllerEnrollment,
    ForkControllerMode,
)
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionRejected
from cayu.server.auth import AuthConfigurationError
from cayu.server.contracts import (
    ApiEventRecord,
    ApiSessionDetail,
    ApiSupervisionEvent,
    ListSessionsResponse,
    SupervisionEventsRequest,
)
from cayu.server.routes import (
    _BoundedPrivateJsonBodyRoute,
    _BoundedSupervisionEventsRoute,
    _serialize_event_record,
    _serialize_session_base,
    _serialize_session_cursor,
    _serialize_session_detail,
    _serialize_supervision_event,
    _supervision_event_query_from_request,
)
from cayu.sessions._correlation import CORRELATION_FIELDS
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.event_queries import EventOrder, EventQuery, EventQueryResultTooLarge
from cayu.sessions.forks import ForkRequestConflict
from cayu.sessions.queries import SessionOrder, SessionQuery
from cayu.sessions.records import EventRecord, SessionStatus
from cayu.sessions.requests import ForkSessionRequest, ResumeRequest
from cayu.sessions.supervision_events import (
    SupervisionEventCursorInvalid,
    SupervisionEventResultTooLarge,
)

_DENIED = "Fork controller access denied."
_NO_STORE = {"Cache-Control": "private, no-store"}
FORK_CONTROLLER_EVENT_PAGE_MAX = 1000
FORK_CONTROLLER_EVENT_QUERY_MAX_SESSION_IDS = 500
FORK_CONTROLLER_EVENT_QUERY_MAX_REQUEST_BYTES = 64 * 1024
FORK_CONTROLLER_TOKENS_VARIABLE = "CAYU_FORK_CONTROLLER_TOKENS"
FORK_CONTROLLER_TOKEN_MIN_CHARS = 32
FORK_CONTROLLER_MAX_TOKENS = 64
# RFC 6750 b64token: the only credential syntax a bearer header may carry.
_BEARER_CREDENTIAL = re.compile(r"[A-Za-z0-9\-._~+/]+=*")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("ascii")).digest()


class ForkControllerBearerAuth:
    """Stock ``authenticate`` callback mapping bearer tokens to controller subjects.

    ``tokens`` maps each token to the subject it authenticates; two tokens may
    name one subject, for rotation. Only SHA-256 digests of the tokens are kept,
    and a presented token is compared with every entry in constant time. A
    missing, malformed, repeated or unknown ``Authorization`` header returns
    ``""``, which the router answers with 401. Tokens never appear in errors,
    logs or ``repr``.
    """

    __slots__ = ("_entries",)

    def __init__(self, tokens: Mapping[str, str]) -> None:
        if not isinstance(tokens, Mapping):
            raise TypeError("tokens must map bearer tokens to subjects.")
        _require_token_count(len(tokens))
        entries: list[tuple[bytes, str]] = []
        for index, (token, subject) in enumerate(tokens.items()):
            if (
                type(token) is not str
                or len(token) < FORK_CONTROLLER_TOKEN_MIN_CHARS
                or _BEARER_CREDENTIAL.fullmatch(token) is None
            ):
                raise AuthConfigurationError(
                    f"Fork-controller bearer token {index} must be a bearer credential of at "
                    f"least {FORK_CONTROLLER_TOKEN_MIN_CHARS} characters."
                )
            entries.append((_token_digest(token), _entry_subject(subject, index)))
        self._entries = tuple(entries)

    @classmethod
    def from_token_digests(cls, digests: Mapping[str, str]) -> ForkControllerBearerAuth:
        """Map lowercase hex SHA-256 digests of bearer tokens to subjects.

        Use it when the host holds only digests, so the tokens themselves never
        reach the process. Cayu can't check a token's length from its digest;
        issue tokens of at least ``FORK_CONTROLLER_TOKEN_MIN_CHARS`` random
        characters. Digests never appear in errors, logs or ``repr``.
        """

        if not isinstance(digests, Mapping):
            raise TypeError("digests must map token digests to subjects.")
        _require_token_count(len(digests))
        entries: list[tuple[bytes, str]] = []
        for index, (digest, subject) in enumerate(digests.items()):
            if type(digest) is not str or _SHA256_HEX.fullmatch(digest) is None:
                raise AuthConfigurationError(
                    f"Fork-controller token digest {index} must be a lowercase hex SHA-256 digest."
                )
            entries.append((bytes.fromhex(digest), _entry_subject(subject, index)))
        auth = cls.__new__(cls)
        auth._entries = tuple(entries)
        return auth

    @classmethod
    def from_environment(
        cls,
        variable: str = FORK_CONTROLLER_TOKENS_VARIABLE,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ForkControllerBearerAuth:
        """Read a JSON object of ``{"<token>": "<subject>"}`` from one environment variable.

        The variable is read once, when this method is called. ``environ``
        replaces ``os.environ`` for tests and for callers that resolve secrets
        into a mapping first.
        """

        source: Mapping[str, str] = os.environ if environ is None else environ
        raw = source.get(variable)
        if not isinstance(raw, str) or not raw.strip():
            raise AuthConfigurationError(f"{variable} must be set to a JSON object of tokens.")
        return cls(_token_mapping(raw, variable))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> ForkControllerBearerAuth:
        """Read a JSON object of ``{"<token>": "<subject>"}`` from a file, such as a mounted secret."""

        # Decode errors hold the file's bytes, tokens included, so the
        # configuration error is raised outside the handler: it must not chain them.
        raw: str | None = None
        with suppress(OSError, UnicodeDecodeError):
            raw = Path(path).read_text(encoding="utf-8")
        if raw is None:
            raise AuthConfigurationError("The fork-controller token file could not be read.")
        return cls(_token_mapping(raw, "The fork-controller token file"))

    async def __call__(self, request: Request) -> str:
        headers = request.headers.getlist("authorization")
        if len(headers) != 1:
            return ""
        scheme, separator, credential = headers[0].partition(" ")
        if (
            separator != " "
            or scheme.lower() != "bearer"
            or _BEARER_CREDENTIAL.fullmatch(credential) is None
        ):
            return ""
        presented = _token_digest(credential)
        subject = ""
        # Compare with every entry, without an early exit, so timing does not
        # reveal which entry matched.
        for digest, entry_subject in self._entries:
            if hmac.compare_digest(presented, digest):
                subject = entry_subject
        return subject

    def __repr__(self) -> str:
        return f"ForkControllerBearerAuth(tokens={len(self._entries)})"


def _require_token_count(count: int) -> None:
    if not count:
        raise AuthConfigurationError("At least one fork-controller bearer token is required.")
    if count > FORK_CONTROLLER_MAX_TOKENS:
        raise AuthConfigurationError(
            f"At most {FORK_CONTROLLER_MAX_TOKENS} fork-controller bearer tokens are allowed."
        )


def _entry_subject(subject: object, index: int) -> str:
    if type(subject) is not str:
        raise AuthConfigurationError(
            f"The subject for fork-controller bearer token {index} must be a string."
        )
    try:
        return require_durable_clean_nonblank(subject, "subject")
    except ValueError:
        raise AuthConfigurationError(
            f"The subject for fork-controller bearer token {index} is not a valid subject."
        ) from None


def _token_mapping(raw: str, source_name: str) -> dict[str, str]:
    # A JSON error keeps the whole document, tokens included, so it is not chained.
    parsed = True
    try:
        value = json.loads(raw)
    except ValueError:
        parsed = False
    if not parsed:
        raise AuthConfigurationError(f"{source_name} is not valid JSON.")
    if type(value) is not dict:
        raise AuthConfigurationError(f"{source_name} must be a JSON object of tokens to subjects.")
    return value


async def _stream(events):
    # Admission failures surface before any successful HTTP headers are sent.
    # Details are fixed: an absent session and a foreign one look the same.
    try:
        first = await anext(events)
    except StopAsyncIteration:
        return StreamingResponse(iter(()), media_type="text/event-stream")
    except (ForkControllerDenied, KeyError):
        await events.aclose()
        raise HTTPException(status_code=403, detail=_DENIED) from None
    except ExecutionProfileAdoptionRejected:
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The fork's execution profile was not adopted."
        ) from None
    except ForkRequestConflict:
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The fork destination exists with a different request."
        ) from None
    except SessionRunFenced:
        # For example, the source moved after the controller's exact snapshot.
        await events.aclose()
        raise HTTPException(
            status_code=409, detail="The session changed during the request; read it again."
        ) from None
    except ValueError:
        await events.aclose()
        raise HTTPException(status_code=422, detail="Invalid fork-controller request.") from None

    async def body():
        async with aclosing(events):
            yield "data: " + first.model_dump_json() + "\n\n"
            async for event in events:
                yield "data: " + event.model_dump_json() + "\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


class ForkControllerEventQueryRequest(BaseModel):
    """Cross-session event query for a controller; host-authority fields do not exist here.

    ``since`` may be omitted; the page is bounded by ``after_sequence``,
    ``limit`` and ``max_bytes``. A query that is not for exactly one session
    reads in ascending order only and starts after ``after_sequence``, 0 when
    omitted. Session ids the controller may not read match nothing.
    """

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    session_id: str | None = None
    session_ids: tuple[str, ...] = Field(
        default=(), max_length=FORK_CONTROLLER_EVENT_QUERY_MAX_SESSION_IDS
    )
    event_type: EventType | str | None = None
    event_types: tuple[EventType | str, ...] = Field(default=(), max_length=len(EventType))
    exclude_event_types: tuple[EventType | str, ...] = Field(default=(), max_length=len(EventType))
    agent_name: str | None = None
    environment_name: str | None = None
    tool_name: str | None = None
    since: datetime | None = None
    until: datetime | None = None
    after_sequence: StrictInt | None = None
    before_sequence: StrictInt | None = None
    limit: StrictInt = Field(default=100, ge=1, le=FORK_CONTROLLER_EVENT_PAGE_MAX)
    order_by: EventOrder = EventOrder.SEQUENCE_ASC
    exclude_forked_sessions: StrictBool = False
    max_bytes: StrictInt = Field(
        default=FORK_CONTROLLER_READ_DEFAULT_BYTES, ge=1, le=FORK_CONTROLLER_READ_MAX_BYTES
    )


class ForkControllerEventsResponse(BaseModel):
    """One page of events; ``next_sequence`` continues it in ``order_by`` order."""

    events: list[ApiEventRecord]
    order_by: EventOrder
    next_sequence: StrictInt | None
    has_more: StrictBool


class ForkControllerSessionEventsResponse(ForkControllerEventsResponse):
    session_id: str


class ForkControllerSupervisionEvent(ApiSupervisionEvent):
    """One supervision touch plus keyed correlation values that join its events.

    ``tool_call_correlation_ids`` follow the touched calls in order and equal
    their tool events' ``tool_call_correlation_id``. The other values equal the
    same-named fields on the session's tool and model events. None of them is
    resolution authority. Without a configured alias keyring the four fields
    are absent from the response.
    """

    tool_call_correlation_ids: list[str] = Field(default_factory=list)
    tool_round_correlation_id: str | None = None
    model_step_correlation_id: str | None = None
    model_attempt_correlation_id: str | None = None


class ForkControllerSupervisionEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[ForkControllerSupervisionEvent]
    next_cursor: str
    has_more: StrictBool


class _BoundedEventQueryRoute(_BoundedPrivateJsonBodyRoute):
    """Bound controller event queries before FastAPI parses their filters."""

    max_request_bytes = FORK_CONTROLLER_EVENT_QUERY_MAX_REQUEST_BYTES
    invalid_request_detail = "Invalid fork-controller event query."
    oversized_request_detail = "Fork-controller event query exceeds the server byte limit."


def _too_large() -> HTTPException:
    return HTTPException(
        status_code=413,
        detail="The result exceeds the byte limit; request a smaller page.",
        headers=_NO_STORE,
    )


async def _read(operation: Awaitable[Any]) -> Any:
    """Map read failures to fixed details; an absent session and a foreign one look the same."""

    try:
        return await operation
    except ForkControllerDenied:
        raise HTTPException(status_code=403, detail=_DENIED, headers=_NO_STORE) from None
    except NotImplementedError:
        raise HTTPException(
            status_code=501,
            detail="The configured session store does not support fork-controller reads.",
            headers=_NO_STORE,
        ) from None
    except (EventQueryResultTooLarge, SupervisionEventResultTooLarge):
        raise _too_large() from None
    except SupervisionEventCursorInvalid:
        raise HTTPException(
            status_code=422, detail="Invalid supervision event cursor.", headers=_NO_STORE
        ) from None
    except (ValueError, ValidationError):
        raise HTTPException(
            status_code=422, detail="Invalid fork-controller read.", headers=_NO_STORE
        ) from None


def _controller_event(
    app: CayuApp, handle: ForkControllerAccess, record: EventRecord
) -> dict[str, Any]:
    """Serialize like the control plane, then publish only derived correlation values.

    Every correlation field is dropped first, including the control plane's own
    ``tool_call_correlation_id``, which a host without a keyring derives from a
    process-local key; only keyed values are added back.
    """

    serialized = _serialize_event_record(app, record)
    payload = serialized["payload"]
    for field in CORRELATION_FIELDS:
        payload.pop(field, None)
    payload.update(handle.event_correlation_ids(record))
    return serialized


def _event_page(
    app: CayuApp,
    handle: ForkControllerAccess,
    page: tuple[list[EventRecord], bool],
    *,
    order_by: EventOrder,
    after_sequence: int | None,
    before_sequence: int | None,
    max_bytes: int,
) -> dict[str, Any]:
    records, has_more = page
    cursor = after_sequence if order_by is EventOrder.SEQUENCE_ASC else before_sequence
    payload = {
        "events": [_controller_event(app, handle, record) for record in records],
        "order_by": order_by,
        "next_sequence": records[-1].sequence if records else cursor,
        "has_more": has_more,
    }
    if not json_utf8_size_within_limit(payload["events"], max_bytes):
        raise _too_large()
    return payload


def _with_authenticated_actor(body, subject: str):
    """Adoption provenance comes from authentication, never from the request body."""

    intent = body.profile_adoption
    if intent is None:
        return body
    if intent.requested_by.subject != subject:
        # Never silently replace a claimed actor; the audit trail would disagree.
        raise HTTPException(status_code=403, detail=_DENIED)
    return body.model_copy(
        update={
            "profile_adoption": intent.model_copy(
                update={
                    "requested_by": ResolutionActor(
                        subject=subject, source=ResolutionActorSource.HTTP_AUTH
                    )
                }
            )
        }
    )


def create_fork_controller_router(
    app: CayuApp,
    *,
    authenticate: Callable[[Request], Awaitable[str]],
    prefix: str = "/fork-controller",
) -> APIRouter:
    """Mount enrollment, source snapshots, forks, resumes and read routes for controllers."""

    if not callable(authenticate):
        raise TypeError("A trusted authentication callback is required.")
    router = APIRouter(prefix=prefix)

    async def access(request: Request) -> ForkControllerAccess:
        subject = await authenticate(request)
        if type(subject) is not str or not subject.strip() or subject != subject.strip():
            raise HTTPException(status_code=401, detail="Authentication required.")
        try:
            return app.fork_controller_access(subject)
        except ForkControllerDenied:
            raise HTTPException(status_code=403, detail=_DENIED) from None

    access_dependency = Depends(access)

    @router.get("/enrollment", response_model=ForkControllerEnrollment)
    async def enrollment(handle: ForkControllerAccess = access_dependency):
        return handle.enrollment()

    @router.get("/sources/{session_id}/snapshot")
    async def snapshot(session_id: str, handle: ForkControllerAccess = access_dependency):
        try:
            return await handle.snapshot_source(session_id)
        except (ForkControllerDenied, KeyError, ValueError):
            # A malformed id is refused exactly like an absent or foreign one.
            raise HTTPException(status_code=403, detail=_DENIED) from None

    @router.post("/fork")
    async def fork(body: ForkSessionRequest, handle: ForkControllerAccess = access_dependency):
        return await _stream(handle.fork(_with_authenticated_actor(body, handle.subject)))

    @router.post("/resume")
    async def resume(body: ResumeRequest, handle: ForkControllerAccess = access_dependency):
        return await _stream(handle.resume(_with_authenticated_actor(body, handle.subject)))

    @router.get("/sessions", response_model=ListSessionsResponse)
    async def sessions(
        limit: Annotated[int, Query(ge=1, le=1000)] = 50,
        cursor: Annotated[str | None, Query(min_length=1, max_length=4096)] = None,
        status: SessionStatus | None = None,
        agent_name: Annotated[str | None, Query(min_length=1, max_length=256)] = None,
        created_after: datetime | None = None,
        order_by: SessionOrder = SessionOrder.UPDATED_AT_DESC,
        handle: ForkControllerAccess = access_dependency,
    ):
        async def listed():
            return await handle.sessions(
                SessionQuery(
                    status=status,
                    agent_name=agent_name,
                    created_after=created_after,
                    limit=limit,
                    cursor=cursor,
                    include_total_count=True,
                    order_by=order_by,
                )
            )

        result = await _read(listed())
        return {
            "sessions": [_serialize_session_base(app, session) for session in result.sessions],
            "next_cursor": _serialize_session_cursor(app, result.next_cursor),
            "total_count": result.total_count,
        }

    @router.get("/sessions/{session_id}", response_model=ApiSessionDetail)
    async def session(session_id: str, handle: ForkControllerAccess = access_dependency):
        return _serialize_session_detail(app, await _read(handle.session(session_id)))

    @router.get("/sessions/{session_id}/events", response_model=ForkControllerSessionEventsResponse)
    async def session_events(
        session_id: str,
        after_sequence: Annotated[int | None, Query(ge=0, le=MAX_DURABLE_JSON_INTEGER)] = None,
        before_sequence: Annotated[int | None, Query(ge=1, le=MAX_DURABLE_JSON_INTEGER)] = None,
        order_by: EventOrder = EventOrder.SEQUENCE_ASC,
        limit: Annotated[int, Query(ge=1, le=FORK_CONTROLLER_EVENT_PAGE_MAX)] = 100,
        max_bytes: Annotated[
            int, Query(ge=1, le=FORK_CONTROLLER_READ_MAX_BYTES)
        ] = FORK_CONTROLLER_READ_DEFAULT_BYTES,
        handle: ForkControllerAccess = access_dependency,
    ):
        loaded = await _read(handle.session(session_id))

        async def queried():
            # Built inside the guarded read, so an invalid range is a 422, not a 500.
            return await handle._query_event_page(
                EventQuery(
                    session_id=loaded.id,
                    after_sequence=after_sequence,
                    before_sequence=before_sequence,
                    order_by=order_by,
                    limit=limit,
                ),
                exclude_forked_sessions=False,
                max_bytes=max_bytes,
            )

        page = await _read(queried())
        return {
            "session_id": app.project_session_id_for_exposure(loaded.id),
            **_event_page(
                app,
                handle,
                page,
                order_by=order_by,
                after_sequence=after_sequence,
                before_sequence=before_sequence,
                max_bytes=max_bytes,
            ),
        }

    async def query_events(
        body: ForkControllerEventQueryRequest, handle: ForkControllerAccess = access_dependency
    ):
        fields = body.model_dump(
            exclude={"exclude_forked_sessions", "max_bytes", "limit"}, exclude_defaults=True
        )

        async def queried():
            return await handle._query_event_page(
                EventQuery(**fields, limit=body.limit),
                exclude_forked_sessions=body.exclude_forked_sessions,
                max_bytes=body.max_bytes,
            )

        page = await _read(queried())
        return _event_page(
            app,
            handle,
            page,
            order_by=body.order_by,
            after_sequence=body.after_sequence,
            before_sequence=body.before_sequence,
            max_bytes=body.max_bytes,
        )

    router.add_api_route(
        "/events/query",
        query_events,
        methods=["POST"],
        response_model=ForkControllerEventsResponse,
        route_class_override=_BoundedEventQueryRoute,
    )

    async def query_supervision_events(
        body: SupervisionEventsRequest, handle: ForkControllerAccess = access_dependency
    ):
        async def paged():
            return await handle._supervision_page_with_anchors(
                _supervision_event_query_from_request(body)
            )

        page, anchors = await _read(paged())
        events = [
            {
                **_serialize_supervision_event(app, event, anchors[event.sequence]),
                **handle.supervision_correlation_ids(event),
            }
            for event in page.events
        ]
        response = ForkControllerSupervisionEventsResponse.model_validate(
            {"events": events, "next_cursor": page.next_cursor, "has_more": page.has_more}
        )
        # The store bounds the whole private page; correlation values add bytes,
        # so the whole published page, envelope included, is bounded again in
        # the exact form the route returns.
        if not json_utf8_size_within_limit(
            response.model_dump(mode="json", exclude_unset=True), body.max_bytes
        ):
            raise _too_large()
        return response

    router.add_api_route(
        "/supervision-events/query",
        query_supervision_events,
        methods=["POST"],
        response_model=ForkControllerSupervisionEventsResponse,
        # The serializer sets every control-plane field, so only correlation
        # fields a keyless host withholds are left unset and omitted.
        response_model_exclude_unset=True,
        route_class_override=_BoundedSupervisionEventsRoute,
    )

    return router


FORK_CONTROLLER_NAME_VARIABLE = "CAYU_FORK_CONTROLLER_NAME"
FORK_CONTROLLER_AGENTS_VARIABLE = "CAYU_FORK_CONTROLLER_AGENTS"
FORK_CONTROLLER_TOKEN_SHA256_VARIABLE = "CAYU_FORK_CONTROLLER_TOKEN_SHA256"
FORK_CONTROLLER_MODE_VARIABLE = "CAYU_FORK_CONTROLLER_MODE"
FORK_CONTROLLER_HOST_TOOLS_VARIABLE = "CAYU_FORK_CONTROLLER_HOST_TOOLS"
FORK_CONTROLLER_ALLOW_MODEL_RUNS_VARIABLE = "CAYU_FORK_CONTROLLER_ALLOW_MODEL_RUNS"
FORK_CONTROLLER_FORK_BUDGET_USD_VARIABLE = "CAYU_FORK_CONTROLLER_FORK_BUDGET_USD"
FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS_VARIABLE = (
    "CAYU_FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS"
)
FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS_VARIABLE = (
    "CAYU_FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS"
)
_FORK_BUDGET_VARIABLES = (
    FORK_CONTROLLER_FORK_BUDGET_USD_VARIABLE,
    FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS_VARIABLE,
    FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS_VARIABLE,
)
#: The variables ``fork_controller_from_environment`` reads, all reserved for it.
FORK_CONTROLLER_ENVIRONMENT_VARIABLES = (
    FORK_CONTROLLER_NAME_VARIABLE,
    FORK_CONTROLLER_AGENTS_VARIABLE,
    FORK_CONTROLLER_TOKEN_SHA256_VARIABLE,
    FORK_CONTROLLER_MODE_VARIABLE,
    FORK_CONTROLLER_HOST_TOOLS_VARIABLE,
    FORK_CONTROLLER_ALLOW_MODEL_RUNS_VARIABLE,
    *_FORK_BUDGET_VARIABLES,
)
#: A controller enrolled from the environment acts as ``fork-controller:<name>``.
ENVIRONMENT_FORK_CONTROLLER_SUBJECT_PREFIX = "fork-controller:"


@dataclass(frozen=True, slots=True, repr=False)
class EnvironmentForkController:
    """One controller and its bearer authentication, read from the environment."""

    controller: ForkController
    authenticate: ForkControllerBearerAuth

    def __repr__(self) -> str:
        return f"EnvironmentForkController(controller={self.controller.name!r})"


def _environment_names(raw: str, variable: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not raw.strip():
        if allow_empty:
            return ()
        raise ValueError(f"{variable} must name at least one entry.")
    names = tuple(item.strip() for item in raw.split(","))
    if any(not name for name in names):
        raise ValueError(f"{variable} must be a comma-separated list without empty entries.")
    return names


def _environment_fork_budget_limit(configured: Mapping[str, str]) -> BudgetLimit | None:
    """Build the per-fork cap from its three variables, or nothing when none is set."""

    present = [name for name in _FORK_BUDGET_VARIABLES if name in configured]
    if not present:
        return None
    if len(present) != len(_FORK_BUDGET_VARIABLES):
        raise ValueError("The per-fork cap needs all of " + ", ".join(_FORK_BUDGET_VARIABLES) + ".")
    tokens: list[int] = []
    for variable in (
        FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS_VARIABLE,
        FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS_VARIABLE,
    ):
        raw = configured[variable].strip()
        if not raw.isdigit():
            raise ValueError(f"{variable} must be a whole number of tokens.")
        tokens.append(int(raw))
    try:
        return BudgetLimit.usd(
            configured[FORK_CONTROLLER_FORK_BUDGET_USD_VARIABLE].strip(),
            reservation=BudgetReservation(max_input_tokens=tokens[0], max_output_tokens=tokens[1]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The per-fork cap is invalid: {exc}") from None


def fork_controller_from_environment(
    environ: Mapping[str, str] | None = None,
) -> EnvironmentForkController | None:
    """Read one fork controller from the reserved ``CAYU_FORK_CONTROLLER_*`` variables.

    Returns ``None`` when none of them is set. ``CAYU_FORK_CONTROLLER_NAME``,
    ``CAYU_FORK_CONTROLLER_AGENTS`` (comma-separated) and
    ``CAYU_FORK_CONTROLLER_TOKEN_SHA256`` (comma-separated lowercase hex
    SHA-256 digests of the bearer tokens) are required. ``_MODE`` (``observe``),
    ``_HOST_TOOLS`` (comma-separated, default none) and ``_ALLOW_MODEL_RUNS``
    (``true`` or ``false``, default ``false``) are optional. ``_ALLOW_MODEL_RUNS=true``
    needs the per-fork cap: ``_FORK_BUDGET_USD`` and the reservation's
    ``_FORK_RESERVATION_INPUT_TOKENS`` and ``_FORK_RESERVATION_OUTPUT_TOKENS``,
    priced with Cayu's bundled price book. The controller acts
    as subject ``fork-controller:<name>``. Any malformed or partial
    configuration raises, and errors never include digests. ``environ``
    replaces ``os.environ``.
    """

    source: Mapping[str, str] = os.environ if environ is None else environ
    configured = {
        name: source[name] for name in FORK_CONTROLLER_ENVIRONMENT_VARIABLES if name in source
    }
    if not configured:
        return None
    name = configured.get(FORK_CONTROLLER_NAME_VARIABLE, "").strip()
    if not name:
        raise ValueError(
            f"{FORK_CONTROLLER_NAME_VARIABLE} must be set when any CAYU_FORK_CONTROLLER_* "
            "enrollment variable is set."
        )
    for required in (FORK_CONTROLLER_AGENTS_VARIABLE, FORK_CONTROLLER_TOKEN_SHA256_VARIABLE):
        if required not in configured:
            raise ValueError(f"{required} must be set to enroll a fork controller.")
    agents = _environment_names(
        configured[FORK_CONTROLLER_AGENTS_VARIABLE],
        FORK_CONTROLLER_AGENTS_VARIABLE,
        allow_empty=False,
    )
    host_tools = _environment_names(
        configured.get(FORK_CONTROLLER_HOST_TOOLS_VARIABLE, ""),
        FORK_CONTROLLER_HOST_TOOLS_VARIABLE,
        allow_empty=True,
    )
    allow_model_runs = configured.get(FORK_CONTROLLER_ALLOW_MODEL_RUNS_VARIABLE, "false")
    if allow_model_runs not in {"true", "false"}:
        raise ValueError(f"{FORK_CONTROLLER_ALLOW_MODEL_RUNS_VARIABLE} must be true or false.")
    mode = configured.get(FORK_CONTROLLER_MODE_VARIABLE, ForkControllerMode.OBSERVE.value)
    if mode not in {item.value for item in ForkControllerMode}:
        raise ValueError(f"{FORK_CONTROLLER_MODE_VARIABLE} must be 'observe'.")
    try:
        controller = ForkController(
            name=name,
            subjects=(ENVIRONMENT_FORK_CONTROLLER_SUBJECT_PREFIX + name,),
            agents=agents,
            host_tools=host_tools,
            mode=ForkControllerMode(mode),
            allow_model_runs=allow_model_runs == "true",
            fork_budget_limit=_environment_fork_budget_limit(configured),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"The CAYU_FORK_CONTROLLER_* enrollment is invalid: {exc}") from None
    raw_digests = configured[FORK_CONTROLLER_TOKEN_SHA256_VARIABLE]
    # Digests are split without echoing them; each is validated by position only.
    digests = [item.strip() for item in raw_digests.split(",")] if raw_digests.strip() else []
    if len(digests) != len(set(digests)):
        raise AuthConfigurationError(f"{FORK_CONTROLLER_TOKEN_SHA256_VARIABLE} repeats a digest.")
    subject = controller.subjects[0]
    try:
        authenticate = ForkControllerBearerAuth.from_token_digests(dict.fromkeys(digests, subject))
    except AuthConfigurationError as exc:
        raise AuthConfigurationError(f"{FORK_CONTROLLER_TOKEN_SHA256_VARIABLE}: {exc}") from None
    return EnvironmentForkController(controller=controller, authenticate=authenticate)


def environment_fork_controller_router(
    app: CayuApp,
    *,
    prefix: str = "/fork-controller",
    environ: Mapping[str, str] | None = None,
) -> APIRouter | None:
    """Enroll the environment's fork controller on ``app`` and return its router.

    Returns ``None`` when the environment names no controller. The application's
    agents must already be registered. Calling it again with the same
    configuration reuses the enrollment; a controller enrolled in code under the
    same name with a different enrollment is a conflict.
    """

    configured = fork_controller_from_environment(environ)
    if configured is None:
        return None
    controller = configured.controller
    try:
        existing = app._agent_registry.fork_controller(controller.name)
    except KeyError:
        existing = None
    if existing is None:
        try:
            app.enroll_fork_controller(controller)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"The fork controller named by {FORK_CONTROLLER_NAME_VARIABLE} cannot be "
                f"enrolled: {exc}"
            ) from None
    elif existing != controller:
        raise ValueError(
            f"{FORK_CONTROLLER_NAME_VARIABLE} names a fork controller the application already "
            "enrolled differently."
        )
    return create_fork_controller_router(app, authenticate=configured.authenticate, prefix=prefix)


__all__ = [
    "ENVIRONMENT_FORK_CONTROLLER_SUBJECT_PREFIX",
    "FORK_CONTROLLER_AGENTS_VARIABLE",
    "FORK_CONTROLLER_ALLOW_MODEL_RUNS_VARIABLE",
    "FORK_CONTROLLER_ENVIRONMENT_VARIABLES",
    "FORK_CONTROLLER_FORK_BUDGET_USD_VARIABLE",
    "FORK_CONTROLLER_FORK_RESERVATION_INPUT_TOKENS_VARIABLE",
    "FORK_CONTROLLER_FORK_RESERVATION_OUTPUT_TOKENS_VARIABLE",
    "FORK_CONTROLLER_HOST_TOOLS_VARIABLE",
    "FORK_CONTROLLER_MODE_VARIABLE",
    "FORK_CONTROLLER_NAME_VARIABLE",
    "FORK_CONTROLLER_TOKENS_VARIABLE",
    "FORK_CONTROLLER_TOKEN_SHA256_VARIABLE",
    "EnvironmentForkController",
    "ForkControllerBearerAuth",
    "ForkControllerEventQueryRequest",
    "ForkControllerEventsResponse",
    "ForkControllerSessionEventsResponse",
    "ForkControllerSupervisionEvent",
    "ForkControllerSupervisionEventsResponse",
    "create_fork_controller_router",
    "environment_fork_controller_router",
    "fork_controller_from_environment",
]
