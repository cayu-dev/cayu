"""Where `cayu inbox` reads and settles an agent's inbox.

The commands use one ``InboxClient`` and never build requests themselves. There are two:

- ``LocalInbox`` serves the project's own application in this process, through the same
  inbox and human-review routes ``cayu serve`` mounts (``docs/inbox.md``), so a decision
  here is ``POST /api/inbox/{id}/settle`` run in process. The approved call and the rest
  of the turn then run in this process too, so ``after_settle`` waits until the session
  pauses or ends.
- ``CloudInbox`` reads a Cloud Agent's inbox through Cayu Cloud's
  ``/v1/applications/{id}/inbox`` routes with the CLI's login. Cloud carries a decision to
  the Agent as the signed-in person, and the Agent continues on its own.

Both return the Runtime's own documents: inbox items, item states, and the
``HumanReviewView`` the application's human-review policy builds for the caller. The
review is the only place arguments appear, and only the ones that policy shows.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.cli._cloud_api import CloudApiClient, CloudApiError
    from cayu.runtime.loop_policies import LoopPolicy

#: The most items one listing reads, however many pages that takes; it lists the oldest.
MAX_LISTED_ITEMS = 10_000
_PAGE_LIMIT = 100
#: How often the local client checks whether a continued session is still running.
_CONTINUATION_POLL_SECONDS = 0.2
_RUNNING = frozenset({"pending", "running", "interrupting"})


class InboxClientError(RuntimeError):
    """A refused or failed inbox request; ``code`` is the inbox's own code when it gave one.

    ``item`` is the item's current state when the refusal carried it (a 409 settle).
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        item: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.item = item


@dataclass(frozen=True)
class InboxListing:
    """The items waiting, oldest first, and whether the listing holds all of them."""

    items: list[dict[str, Any]]
    complete: bool
    #: Why the listing may be incomplete, in plain words.
    notes: list[str] = field(default_factory=list)


class InboxClient(Protocol):
    #: ``local`` or ``cloud``, for the output.
    where: str
    #: The human-review purpose the application names for its reviews, when it says.
    purpose: str | None

    async def list_items(
        self, *, kind: str | None, session_id: str | None, agent_name: str | None
    ) -> InboxListing: ...

    async def item(self, item_id: str) -> dict[str, Any]:
        """The item's state: ``pending`` with the item, ``settled`` with who decided, and so on."""
        ...

    async def review(self, item_id: str, session_id: str, *, purpose: str) -> dict[str, Any]:
        """The caller's protected review of the session's pending interaction."""
        ...

    async def long_field(
        self, item_id: str, session_id: str, *, reference: dict[str, Any], key: str
    ) -> dict[str, Any]:
        """One long review field, whole."""
        ...

    async def settle(self, item_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """``{"replayed", "item"}`` once the decision was durably accepted."""
        ...

    async def after_settle(self, session_id: str | None) -> str | None:
        """Wait for work the decision continues in this process; the session's status then."""
        ...


def _refusal(status_code: int, body: Any) -> InboxClientError:
    """The Runtime's or Cloud's refusal, with its code, message and the item's state."""

    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        item = detail.get("item")
        return InboxClientError(
            str(detail.get("code") or f"http_{status_code}"),
            str(detail.get("message") or f"The inbox answered HTTP {status_code}."),
            status_code=status_code,
            item=item if isinstance(item, dict) else None,
        )
    if isinstance(detail, str) and detail:
        message = detail
    elif isinstance(detail, list) and detail and isinstance(detail[0], dict):
        message = str(detail[0].get("msg") or f"The inbox answered HTTP {status_code}.")
    else:
        message = f"The inbox answered HTTP {status_code}."
    code = {403: "forbidden", 404: "item_not_found", 422: "invalid_request"}.get(
        status_code, f"http_{status_code}"
    )
    return InboxClientError(code, message, status_code=status_code)


class LocalInbox:
    """The project application's inbox, served in this process.

    ``reviewer`` is who the decisions are recorded as: the subject the in-process server
    authenticates every request as, and the recipient the human-review policy sees.
    ``continuation_loop_policies`` is what a project's public service gives the sessions it
    started when they continue (``CayuService``), as ``cayu serve`` mounts it.
    ``wait_seconds`` bounds the whole wait for work a decision continues here: waiting for
    the session, then closing the application with what is left (``remaining_wait_seconds``).
    """

    where = "local"

    def __init__(
        self,
        app: CayuApp,
        *,
        reviewer: str,
        continuation_loop_policies: Callable[[str], Awaitable[tuple[LoopPolicy, ...]]]
        | None = None,
        wait_seconds: float | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._wait_seconds = wait_seconds
        self._monotonic = monotonic
        self._wait_deadline: float | None = None
        import httpx

        try:
            from fastapi import FastAPI

            from cayu.server.auth import AuthContext
            from cayu.server.routes import create_router
        except ModuleNotFoundError as exc:
            raise InboxClientError(
                "server_extra_missing",
                "The local inbox serves Cayu's inbox routes in this process, which needs the "
                'server extra. Install it with: pip install "cayu[server]"',
            ) from exc

        identity = AuthContext(subject=reviewer)
        api = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
        api.include_router(
            create_router(
                cayu_app=app,
                session_store=app.session_store,
                task_store=app.task_store,
                knowledge_store=app.knowledge_store,
                knowledge_access_scope=app.knowledge_access_scope,
                knowledge_review_namespace=app.knowledge_review_namespace,
                knowledge_review_labels=app.knowledge_review_labels,
                auth=lambda _request: identity,
                openapi_url=None,
                continuation_loop_policy_provider=continuation_loop_policies,
            )
        )
        self._http = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api), base_url="http://cayu.local"
        )
        self.purpose: str | None = app.human_review_purpose

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await self._http.request(method, "/api" + path, params=params, json=payload)
        try:
            body = response.json()
        except ValueError:
            body = None
        if not 200 <= response.status_code < 300:
            raise _refusal(response.status_code, body)
        if not isinstance(body, dict):
            raise InboxClientError("invalid_response", "The inbox answered without a JSON object.")
        return body

    async def list_items(
        self, *, kind: str | None, session_id: str | None, agent_name: str | None
    ) -> InboxListing:
        filters = {
            name: value
            for name, value in (
                ("kind", kind),
                ("session_id", session_id),
                ("agent_name", agent_name),
            )
            if value is not None
        }
        items: list[dict[str, Any]] = []
        notes: list[str] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {**filters, "limit": _PAGE_LIMIT}
            if cursor is not None:
                params["cursor"] = cursor
            page = await self._json("GET", "/inbox", params=params)
            items.extend(row for row in page.get("items", []) if isinstance(row, dict))
            notes.extend(_issue_note(issue) for issue in page.get("issues", []))
            cursor = page.get("next_cursor")
            if cursor is not None and len(items) >= MAX_LISTED_ITEMS:
                notes.append(f"Only the oldest {MAX_LISTED_ITEMS} of the items read are listed.")
            if cursor is None or len(items) >= MAX_LISTED_ITEMS:
                # The inbox pages by session activity; list oldest first, as Cloud does.
                items.sort(
                    key=lambda item: (str(item.get("created_at") or ""), str(item.get("id") or ""))
                )
                return InboxListing(items[:MAX_LISTED_ITEMS], complete=not notes, notes=notes)

    async def item(self, item_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/inbox/{quote(item_id, safe='')}")

    async def review(self, item_id: str, session_id: str, *, purpose: str) -> dict[str, Any]:
        return await self._json(
            "GET",
            f"/sessions/{quote(session_id, safe='')}/human-review",
            params={"purpose": purpose},
        )

    async def long_field(
        self, item_id: str, session_id: str, *, reference: dict[str, Any], key: str
    ) -> dict[str, Any]:
        return await self._json(
            "POST",
            f"/sessions/{quote(session_id, safe='')}/human-review/long-field",
            payload={"review_reference": reference, "key": key},
        )

    async def settle(self, item_id: str, body: dict[str, Any]) -> dict[str, Any]:
        return await self._json("POST", f"/inbox/{quote(item_id, safe='')}/settle", payload=body)

    async def after_settle(self, session_id: str | None) -> str | None:
        """The settle returned once the decision was recorded; the session continues here."""

        if session_id is None:
            return None
        if self._wait_seconds is not None and self._wait_deadline is None:
            self._wait_deadline = self._monotonic() + self._wait_seconds
        while True:
            session = await self._json("GET", f"/sessions/{quote(session_id, safe='')}")
            status = session.get("status")
            if status not in _RUNNING or self.remaining_wait_seconds() == 0:
                # Past the deadline the session is reported as it is; closing stops it.
                return status if isinstance(status, str) else None
            await asyncio.sleep(_CONTINUATION_POLL_SECONDS)

    def remaining_wait_seconds(self) -> float | None:
        """What is left of ``wait_seconds`` since the first wait began; ``None`` without one."""

        if self._wait_seconds is None:
            return None
        if self._wait_deadline is None:
            return self._wait_seconds
        return max(0.0, self._wait_deadline - self._monotonic())


class CloudInbox:
    """A Cloud Agent's inbox, through Cayu Cloud's API with the CLI's login."""

    where = "cloud"
    # Cloud doesn't say which purpose an Agent's policy accepts.
    purpose = None

    def __init__(
        self,
        client: CloudApiClient,
        application_id: str,
        *,
        idempotency_key: Callable[[], str] = lambda: secrets.token_urlsafe(24),
    ) -> None:
        self._client = client
        self._base = f"/v1/applications/{quote(application_id, safe='')}/inbox"
        self._idempotency_key = idempotency_key

    def _request(self, method: str, path: str, **options: Any) -> dict[str, Any]:
        from cayu.cli._cloud_api import CloudApiError

        try:
            return self._client.request(method, self._base + path, **options)
        except CloudApiError as exc:
            raise _cloud_refusal(exc) from None

    async def list_items(
        self, *, kind: str | None, session_id: str | None, agent_name: str | None
    ) -> InboxListing:
        answer = await asyncio.to_thread(self._request, "GET", "")
        items = [
            row
            for row in answer.get("items", [])
            if isinstance(row, dict)
            and (kind is None or row.get("kind") == kind)
            and (session_id is None or row.get("session_id") == session_id)
            and (agent_name is None or row.get("agent_name") == agent_name)
        ]
        notes = [
            str(agent.get("detail") or agent.get("reason") or agent.get("state"))
            for agent in answer.get("agents", [])
            if isinstance(agent, dict) and agent.get("state") != "read"
        ]
        for agent in answer.get("agents", []):
            if isinstance(agent, dict):
                notes.extend(_issue_note(issue) for issue in agent.get("issues", []))
        return InboxListing(items, complete=answer.get("complete") is True, notes=notes)

    async def item(self, item_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._request, "GET", f"/{quote(item_id, safe='')}")

    async def review(self, item_id: str, session_id: str, *, purpose: str) -> dict[str, Any]:
        try:
            answer = await asyncio.to_thread(
                self._request,
                "GET",
                f"/{quote(item_id, safe='')}/review",
                query={"purpose": purpose},
            )
        except InboxClientError as exc:
            # The item was just read, so a plain 404 is a Cloud without this route.
            if exc.code == "http_404":
                raise _review_unsupported() from None
            raise
        review = answer.get("review")
        if not isinstance(review, dict):
            raise InboxClientError("invalid_response", "Cayu Cloud answered without a review.")
        return review

    async def long_field(
        self, item_id: str, session_id: str, *, reference: dict[str, Any], key: str
    ) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(
                self._request,
                "POST",
                f"/{quote(item_id, safe='')}/review/long-field",
                payload={"review_reference": reference, "key": key},
            )
        except InboxClientError as exc:
            if exc.code == "http_404":
                raise _review_unsupported() from None
            raise

    async def settle(self, item_id: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            answer = await asyncio.to_thread(
                self._request,
                "POST",
                f"/{quote(item_id, safe='')}/settle",
                payload=body,
                idempotency_key=self._idempotency_key(),
            )
        except InboxClientError as exc:
            if exc.code == "review_reference_unsupported":
                raise _review_unsupported() from None
            raise
        return {"replayed": answer.get("replayed") is True, "item": answer.get("item")}

    async def after_settle(self, session_id: str | None) -> str | None:
        # The Agent continues on its own service.
        return None


def _cloud_refusal(exc: CloudApiError) -> InboxClientError:
    body = exc.body or {}
    item = body.get("item")
    if exc.status_code == 422 and any(
        path == "body.review_reference" for path, _ in exc.rejected_fields
    ):
        code = "review_reference_unsupported"
    elif exc.code is not None:
        code = exc.code
    elif exc.status_code is not None:
        code = f"http_{exc.status_code}"
    else:
        code = exc.category
    return InboxClientError(
        code,
        str(exc),
        status_code=exc.status_code,
        item=item if isinstance(item, dict) else None,
    )


def _review_unsupported() -> InboxClientError:
    return InboxClientError(
        "review_unsupported",
        "This Cayu Cloud can't show or decide with an Agent's protected review yet, so it "
        "shows no arguments. Approve or deny in the Agent's Developer dashboard instead.",
    )


def _issue_note(issue: Any) -> str:
    if not isinstance(issue, dict):
        return "The inbox reported a problem reading one source."
    where = issue.get("session_id")
    detail = issue.get("detail") or issue.get("code") or "unreadable"
    return f"{where}: {detail}" if where else str(detail)
