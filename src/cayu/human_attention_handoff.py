"""Provider-neutral handoff of human-attention observations to an HTTP receiver.

An application that wants people told about paused actions composes three parts:

* ``HumanAttentionHandoffSink``: an ``EventSink`` that, on a committed attention lifecycle
  event, projects the session's current pending actions and hands the observations to the
  receiver. ``emit`` returns only after the receiver acknowledged durable acceptance, so
  Runtime's persisted side-effect delivery retries anything else.
* ``HumanAttentionHandoff.reconcile_once``: bounded repair. A paginated
  ``query_pending_actions`` scan resuming from a cursor the receiver persisted, an exact
  ``get_human_attention_state`` check of every request the receiver still holds open, and
  a pass report.
* ``HumanAttentionReceiver``: the HTTP client for the ``cayu.human-attention-handoff/v1``
  receiver protocol documented in ``docs/runtime-contracts.md``.

Only identities and Runtime-owned facts leave the process: the exact attention reference,
Runtime's fixed summary, the observed state and reason, the agent name, and an
application-owned requester reference read from a session label. Never question text,
choices, tool arguments or exception text. Nothing here answers, approves, retries a tool
or resumes a session; resolution stays with Runtime's exact-action APIs.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

import httpx

from cayu._version import __version__
from cayu.observability.events import EventSink
from cayu.runtime.human_attention import HumanAttentionReference, HumanAttentionRequest
from cayu.sessions.base import (
    PendingActionKind,
    PendingActionQuery,
    PendingActionRecord,
    PendingActionResultTooLarge,
    PendingActionSession,
)

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.events import Event

PROTOCOL = "cayu.human-attention-handoff/v1"
URL_ENVIRONMENT = "CAYU_ATTENTION_HANDOFF_URL"
TOKEN_ENVIRONMENT = "CAYU_ATTENTION_HANDOFF_TOKEN"
MAX_OBSERVATIONS = 200
DEFAULT_RECONCILE_INTERVAL_SECONDS = 300.0

# Events that can open or settle an attention request: always synchronized.
LIFECYCLE_EVENTS = frozenset(
    {
        "session.awaiting_user_input",
        "session.interrupted",
        "session.delegated_action.updated",
        "session.resumed",
        "session.completed",
        "session.failed",
        "tool.call.approval_requested",
        "tool.call.approved",
        "tool.call.approval_denied",
        "tool.call.approval_expired",
    }
)
# Frequent events that can only settle a request (manual recovery): synchronized only for
# sessions this process knows to hold an open request. The reconciler covers the rest.
FOLLOW_UP_EVENTS = frozenset(
    {"session.checkpointed", "tool.call.completed", "tool.call.blocked", "tool.call.failed"}
)
# A receiver declining a producer or batch. Retrying the same event cannot succeed.
DECLINED_STATUSES = frozenset({401, 403, 409})
_REQUESTER_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+=-]{0,199}")
_MAX_PARENT_DEPTH = 8
_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")


class HumanAttentionHandoffError(RuntimeError):
    """The receiver did not durably accept a call. Never contains the credential."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code

    @property
    def declined(self) -> bool:
        return self.status in DECLINED_STATUSES


class HumanAttentionReceiver:
    """Authenticated client for one ``cayu.human-attention-handoff/v1`` receiver."""

    def __init__(
        self,
        *,
        url: str,
        token: str,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if not url.startswith(("https://", "http://")):
            raise ValueError("The human-attention receiver URL must be http(s).")
        if not token:
            raise ValueError("The human-attention receiver token must not be empty.")
        self._url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}
        self._transport = transport
        self._timeout = timeout_seconds

    def __repr__(self) -> str:
        return f"HumanAttentionReceiver(url={self._url!r})"

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=self._timeout, headers=self._headers
            ) as client:
                response = await client.request(method, f"{self._url}/{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise HumanAttentionHandoffError(
                f"The human-attention receiver is unreachable ({type(exc).__name__})."
            ) from exc
        if not 200 <= response.status_code < 300:
            code = None
            with contextlib.suppress(ValueError, AttributeError, TypeError):
                detail = response.json().get("detail")
                code = detail.get("code") if isinstance(detail, dict) else None
            raise HumanAttentionHandoffError(
                f"The human-attention receiver answered {response.status_code}.",
                status=response.status_code,
                code=code if isinstance(code, str) else None,
            )
        if response.status_code == 204 or not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise HumanAttentionHandoffError(
                "The human-attention receiver answered with invalid JSON."
            ) from exc

    async def state(self) -> dict[str, Any]:
        return _mapping(await self._call("GET", "state"))

    async def observations(
        self,
        observations: list[dict[str, Any]],
        *,
        session_scope: Mapping[str, Any] | None = None,
        scan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if len(observations) > MAX_OBSERVATIONS:
            raise ValueError(f"Submit at most {MAX_OBSERVATIONS} observations at once.")
        body: dict[str, Any] = {
            "observations": observations,
            "protocol": PROTOCOL,
            "runtime_version": __version__,
        }
        if session_scope is not None:
            body["session_scope"] = dict(session_scope)
        if scan is not None:
            body["scan"] = dict(scan)
        return _mapping(await self._call("POST", "observations", json=body))

    async def references(
        self, *, after: str | None = None, session_id: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if after is not None:
            params["after"] = after
        if session_id is not None:
            params["session_id"] = session_id
        return _mapping(await self._call("GET", "references", params=params))

    async def reconciliation(self, report: Mapping[str, Any]) -> dict[str, Any]:
        body = {"protocol": PROTOCOL, "runtime_version": __version__, **report}
        return _mapping(await self._call("POST", "reconciliations", json=body))


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HumanAttentionHandoffError("The human-attention receiver answered a non-object.")
    return value


def _delegated(action: PendingActionRecord) -> bool:
    return action.kind is PendingActionKind.DELEGATED_ACTION


class HumanAttentionHandoffSink(EventSink):
    """Hands committed attention lifecycle events to the receiver before acknowledging."""

    def __init__(self, handoff: HumanAttentionHandoff) -> None:
        self._handoff = handoff

    async def emit(self, event: Event) -> None:
        event_type = str(event.type)
        session_id = event.session_id
        if not session_id or event_type not in LIFECYCLE_EVENTS | FOLLOW_UP_EVENTS:
            return
        if event_type in FOLLOW_UP_EVENTS and not self._handoff.may_be_open(session_id):
            return
        try:
            await self._handoff.sync_session(session_id)
        except HumanAttentionHandoffError as exc:
            if exc.declined:
                # Revoked credential, unsupported protocol or identity conflict: retrying
                # this event cannot succeed. The reconciler reports the condition.
                _LOG.warning(
                    "Human-attention receiver declined a handoff (status %s, code %s).",
                    exc.status,
                    exc.code,
                )
                return
            # Otherwise Runtime's persisted delivery retries the event; nothing is lost.
            raise


class HumanAttentionHandoff:
    """Hand committed human-attention observations to a receiver, and repair the gaps."""

    def __init__(
        self,
        receiver: HumanAttentionReceiver | None,
        *,
        requester_label: str | None = "requester",
        page_size: int = 200,
        max_pages: int = 32,
        failure_backoff_seconds: float = 30.0,
        recover_event_side_effects: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= page_size <= 200 or not 1 <= max_pages <= 1000:
            raise ValueError("page_size must be 1-200 and max_pages 1-1000.")
        if failure_backoff_seconds < 0:
            raise ValueError("failure_backoff_seconds must not be negative.")
        self._receiver = receiver
        self._requester_label = requester_label
        self._page_size = page_size
        self._max_pages = max_pages
        self._failure_backoff = failure_backoff_seconds
        self._recover = recover_event_side_effects
        self._clock = clock
        self._app: CayuApp | None = None
        self._failed_until = 0.0
        self._open_sessions: set[str] = set()
        self._wake = asyncio.Event()
        self._verify_after: str | None = None

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None, **options: Any
    ) -> HumanAttentionHandoff:
        """Configure from ``CAYU_ATTENTION_HANDOFF_URL`` and ``..._TOKEN``; inert without."""

        environ = os.environ if environ is None else environ
        url = environ.get(URL_ENVIRONMENT, "").strip()
        token = environ.get(TOKEN_ENVIRONMENT, "").strip()
        if bool(url) != bool(token):
            raise ValueError(f"{URL_ENVIRONMENT} and {TOKEN_ENVIRONMENT} must be set together.")
        receiver = HumanAttentionReceiver(url=url, token=token) if url else None
        return cls(receiver, **options)

    @property
    def enabled(self) -> bool:
        return self._receiver is not None

    def event_sinks(self) -> list[EventSink]:
        """The sink for ``CayuApp(event_sinks=...)``; empty when not configured."""

        return [HumanAttentionHandoffSink(self)] if self.enabled else []

    def bind(self, app: CayuApp) -> None:
        self._app = app

    def may_be_open(self, session_id: str) -> bool:
        return session_id in self._open_sessions

    # Handoff -----------------------------------------------------------------------------

    def _require(self) -> tuple[HumanAttentionReceiver, CayuApp]:
        if self._receiver is None or self._app is None:
            raise RuntimeError("HumanAttentionHandoff must be configured and bound to a CayuApp.")
        if self._clock() < self._failed_until:
            # Fail fast while the receiver is failing; Runtime retries the delivery later.
            raise HumanAttentionHandoffError(
                "The human-attention receiver failed recently; retrying later."
            )
        return self._receiver, self._app

    async def _guarded(self, call: Awaitable[_T]) -> _T:
        try:
            return await call
        except HumanAttentionHandoffError as exc:
            if exc.status is None or exc.status >= 500:
                self._failed_until = self._clock() + self._failure_backoff
            raise

    async def sync_session(self, session_id: str) -> None:
        """Synchronize one session's current actions; raise unless the receiver accepted."""

        receiver, app = self._require()
        store = app.session_store
        if await store.load(session_id) is None:
            # Not a session identity this store knows: leave it to the repair scan.
            self._wake.set()
            return
        page = await store.query_pending_actions(
            PendingActionQuery(session_id=session_id, limit=200)
        )
        complete = not page.issues and not page.has_more
        observations: list[dict[str, Any]] = []
        for action in page.actions:
            observed = await self._observe_action(action)
            if observed is None:
                complete = complete and _delegated(action)
                continue
            if observed["state"] == "unavailable":
                complete = False
            observations.append(observed)
        response = await self._guarded(
            receiver.observations(
                observations, session_scope={"complete": complete, "session_id": session_id}
            )
        )
        verify = [await self._observe_reference(item) for item in _items(response, "verify")]
        if verify:
            await self._guarded(receiver.observations(verify))
        if any(item["state"] == "active" for item in observations + verify):
            self._open_sessions.add(session_id)
        elif complete:
            self._open_sessions.discard(session_id)

    async def _observe_action(self, action: PendingActionRecord) -> dict[str, Any] | None:
        assert self._app is not None
        request = HumanAttentionRequest.from_pending_action(action)
        if request is None:
            # A delegated parent is navigation only; its child is observed under the
            # child's own attention identity, so one request never appears twice.
            return None
        state = await self._app.get_human_attention_state(request.reference)
        return {
            "agent_name": action.session.agent_name,
            "reason": state.reason,
            "reference": request.reference.model_dump(mode="json"),
            "requester_ref": await self._requester(action.session)
            if state.state == "active"
            else None,
            "state": state.state,
            "summary": request.summary,
        }

    async def _observe_reference(self, item: object) -> dict[str, Any]:
        assert self._app is not None
        # Validation recomputes the attention identity; Runtime rechecks the incarnation.
        reference = HumanAttentionReference.model_validate(item)
        state = await self._app.get_human_attention_state(reference)
        return {
            "reason": state.reason,
            "reference": reference.model_dump(mode="json"),
            "state": state.state,
        }

    async def _requester(self, session: PendingActionSession) -> str | None:
        """The application-owned requester reference from a session label, if any.

        Set it from application code (``RunRequest(labels=...)``), never from model output.
        A delegated child inherits its nearest ancestor's reference.
        """

        label = self._requester_label
        if label is None or self._app is None:
            return None
        current, parent = dict(session.labels), session.parent_session_id
        for _ in range(_MAX_PARENT_DEPTH):
            value = current.get(label)
            if value is not None:
                if _REQUESTER_REF.fullmatch(value) is None:
                    _LOG.warning("Ignoring a requester label the receiver protocol cannot carry.")
                    return None
                return value
            if parent is None:
                return None
            loaded = await self._app.session_store.load(parent)
            if loaded is None:
                return None
            current, parent = dict(loaded.labels), loaded.parent_session_id
        return None

    # Repair ------------------------------------------------------------------------------

    async def reconcile_once(self) -> dict[str, Any]:
        """One bounded repair pass. Returns the receiver's answer to the pass report."""

        receiver, app = self._require()
        if self._recover:
            await app.recover_persisted_event_side_effects(limit=200)
        state = await self._guarded(receiver.state())
        cursor = state.get("scan_cursor")
        cursor = cursor if isinstance(cursor, str) and cursor else None
        issues: set[str] = set()
        pages = observed = 0
        scan_complete = False
        clean = True
        reset = False
        while pages < self._max_pages:
            try:
                page = await app.session_store.query_pending_actions(
                    PendingActionQuery(cursor=cursor, limit=self._page_size)
                )
            except PendingActionResultTooLarge:
                issues.add("page_too_large")
                break
            except ValueError:
                if cursor is None or reset:
                    raise
                # A cursor from an older pass the store no longer accepts: start over once.
                cursor, reset, clean = None, True, True
                issues.add("scan_cursor_reset")
                continue
            pages += 1
            if page.issues:
                clean = False
                issues.add("pending_action_issues")
            observations = []
            for action in page.actions:
                item = await self._observe_action(action)
                if item is None:
                    if not _delegated(action):
                        clean = False
                        issues.add("unprojectable_action")
                    continue
                if item["state"] == "unavailable":
                    clean = False
                    issues.add("attention_unavailable")
                elif item["state"] == "active":
                    self._open_sessions.add(item["reference"]["session_id"])
                observations.append(item)
            next_cursor = page.next_cursor
            if next_cursor is not None and next_cursor == cursor:
                issues.add("repeated_cursor")
                break
            done = next_cursor is None
            if done and page.has_more:
                clean = False
                issues.add("incomplete_page")
            scan_complete = done and clean
            await self._guarded(
                receiver.observations(
                    observations,
                    scan={"complete": scan_complete, "cursor": None if done else next_cursor},
                )
            )
            observed += len(observations)
            cursor = next_cursor
            if done:
                break
        # The exact state of everything the receiver still holds open, independent of
        # the scan, so a missed closure event is repaired too.
        verify_complete = False
        verify_clean = True
        after = self._verify_after
        for _ in range(self._max_pages):
            listed = await self._guarded(receiver.references(after=after, limit=100))
            observations = [await self._observe_reference(item) for item in _items(listed, "items")]
            for item in observations:
                if item["state"] == "unavailable":
                    verify_clean = False
                    issues.add("attention_unavailable")
                elif item["state"] == "active":
                    self._open_sessions.add(item["reference"]["session_id"])
            if observations:
                await self._guarded(receiver.observations(observations))
                observed += len(observations)
            next_after = listed.get("next_after")
            if isinstance(next_after, str) and next_after == after:
                issues.add("repeated_reference_cursor")
                break
            # Advance only after durable acceptance. Keep bounded passes moving past
            # long-lived open requests; wrap at the end to revisit earlier entries.
            after = next_after if isinstance(next_after, str) else None
            self._verify_after = after
            if after is None:
                verify_complete = verify_clean
                break
        return await self._guarded(
            receiver.reconciliation(
                {
                    "issues": sorted(issues),
                    "observed": observed,
                    "pages": pages,
                    "scan_complete": scan_complete,
                    "verify_complete": verify_complete,
                }
            )
        )

    async def run(self, *, interval_seconds: float | None = None) -> None:
        """Repair forever: once at start-up (enrollment backfill), then on a cadence.

        The cadence is ``interval_seconds``, else the receiver's
        ``reconcile_interval_seconds``, else 300 s, bounded to 10 s-1 h with jitter.
        Failures back off and never stop the loop. Inert when not configured.
        """

        if self._receiver is None:
            return
        failures = 0
        interval = interval_seconds or DEFAULT_RECONCILE_INTERVAL_SECONDS
        while True:
            try:
                await self.reconcile_once()
                failures = 0
                if interval_seconds is None:
                    state = await self._receiver.state()
                    advertised = state.get("reconcile_interval_seconds")
                    if isinstance(advertised, int | float) and not isinstance(advertised, bool):
                        interval = float(advertised)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failures += 1
                _LOG.warning(
                    "Human-attention repair pass failed (%s); retrying.", type(exc).__name__
                )
                wait = min(interval, 15.0 * 2 ** min(failures, 5))
            else:
                wait = interval
            wait = max(10.0, min(wait, 3600.0)) * random.uniform(0.9, 1.1)
            self._wake.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=wait)


def _items(response: Mapping[str, Any], key: str) -> list[Any]:
    value = response.get(key, [])
    if not isinstance(value, list):
        raise HumanAttentionHandoffError(f"The human-attention receiver's {key} is not a list.")
    return value


__all__ = [
    "DECLINED_STATUSES",
    "FOLLOW_UP_EVENTS",
    "LIFECYCLE_EVENTS",
    "PROTOCOL",
    "TOKEN_ENVIRONMENT",
    "URL_ENVIRONMENT",
    "HumanAttentionHandoff",
    "HumanAttentionHandoffError",
    "HumanAttentionHandoffSink",
    "HumanAttentionReceiver",
]
