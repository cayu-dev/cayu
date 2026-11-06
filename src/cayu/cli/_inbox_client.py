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
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import quote

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.cli._cloud_api import CloudApiClient, CloudApiError
    from cayu.runtime.loop_policies import LoopPolicy

#: The most items one listing reads, however many pages that takes; it lists the oldest.
MAX_LISTED_ITEMS = 10_000
_PAGE_LIMIT = 100
#: The page a Cloud listing reads through Cloud at a time: one request to the Agent each.
_CLOUD_PAGE_LIMIT = 50
_KINDS = ("approval", "question", "recovery", "knowledge")
#: How many items an older Cloud left out the CLI reads at once, by id.
_LEFT_OUT_READS = 8
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
    """The items waiting and whether the listing holds all of them.

    A whole listing is oldest first. One page (``limit``) is in the inbox's own order, and
    ``next_cursor`` is where the next page starts.
    """

    items: list[dict[str, Any]]
    complete: bool
    #: Why the listing may be incomplete, in plain words.
    notes: list[str] = field(default_factory=list)
    next_cursor: str | None = None


@dataclass(frozen=True)
class InboxCounts:
    """How many items of each kind wait, or ``None`` with why the inbox couldn't say."""

    counts: dict[str, int] | None
    unavailable_reason: str | None = None

    @classmethod
    def read(cls, answer: dict[str, Any]) -> InboxCounts:
        counts = answer.get("counts")
        if (
            answer.get("complete") is not True
            or not isinstance(counts, dict)
            or any(type(counts.get(kind, 0)) is not int for kind in _KINDS)
        ):
            reason = answer.get("unavailable_reason")
            return cls(None, reason if isinstance(reason, str) else "unavailable")
        return cls({kind: counts.get(kind, 0) for kind in _KINDS})

    @classmethod
    def of(
        cls, items: list[dict[str, Any]], *, kinds: tuple[str, ...] | None = None
    ) -> InboxCounts:
        """Counted from a complete listing of ``kinds`` (every kind by default)."""

        counts = dict.fromkeys(_KINDS if kinds is None else kinds, 0)
        for item in items:
            if item.get("kind") in counts:
                counts[item["kind"]] += 1
        return cls(counts)


class InboxClient(Protocol):
    #: ``local`` or ``cloud``, for the output.
    where: str
    #: The human-review purpose the application names for its reviews, when it says.
    purpose: str | None

    async def list_items(
        self,
        *,
        kind: str | None,
        session_id: str | None,
        agent_name: str | None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> InboxListing:
        """Every item, oldest first; with ``limit`` one page, from ``cursor``."""
        ...

    async def counts(self, *, session_id: str | None, agent_name: str | None) -> InboxCounts:
        """How many items of each kind wait, without listing them."""
        ...

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
        self,
        *,
        kind: str | None,
        session_id: str | None,
        agent_name: str | None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> InboxListing:
        filters = _filters(kind, session_id, agent_name)
        items: list[dict[str, Any]] = []
        notes: list[str] = []
        while True:
            params: dict[str, Any] = {**filters, "limit": limit or _PAGE_LIMIT}
            if cursor is not None:
                params["cursor"] = cursor
            page = await self._json("GET", "/inbox", params=params)
            items.extend(row for row in page.get("items", []) if isinstance(row, dict))
            notes.extend(_issue_note(issue) for issue in page.get("issues", []))
            cursor = page.get("next_cursor")
            if limit is not None:
                return InboxListing(
                    items, complete=cursor is None and not notes, notes=notes, next_cursor=cursor
                )
            if cursor is not None and len(items) >= MAX_LISTED_ITEMS:
                notes.append(f"Only the oldest {MAX_LISTED_ITEMS} of the items read are listed.")
            if cursor is None or len(items) >= MAX_LISTED_ITEMS:
                # The inbox pages by session activity; list oldest first, as Cloud does.
                items.sort(key=_oldest_first)
                return InboxListing(items[:MAX_LISTED_ITEMS], complete=not notes, notes=notes)

    async def counts(self, *, session_id: str | None, agent_name: str | None) -> InboxCounts:
        answer = await self._json(
            "GET", "/inbox/counts", params=_filters(None, session_id, agent_name)
        )
        return InboxCounts.read(answer)

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
        self,
        *,
        kind: str | None,
        session_id: str | None,
        agent_name: str | None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> InboxListing:
        """Every item Cloud reads from the Agent, including those asked before its Release.

        Cloud reads one page of the Agent's inbox per request (``limit``, ``cursor``), and the
        CLI follows ``next_cursor`` until the last page, so a large inbox is read whole. With
        ``limit`` it is that one page. An older Cloud answers with one bounded read instead.

        The filters go to Cloud too, so a Cloud that reads one session's items directly
        does; an older one answers with every item and the CLI filters them here. Items
        asked before the Agent's serving Release carry ``asked_before_release: true``. An
        older Cloud leaves them out of ``items`` and names only their ids
        (``agents[].earlier_release_items``), so each is read by its id.
        """

        query = _filters(kind, session_id, agent_name)
        rows: list[Any] = []
        agents: list[dict[str, Any]] = []
        complete = True
        paged = True
        followed: set[str] = set()
        while True:
            paging: dict[str, Any] = {"limit": limit or _CLOUD_PAGE_LIMIT}
            if cursor is not None:
                paging["cursor"] = cursor
            answer = await asyncio.to_thread(self._request, "GET", "", query={**query, **paging})
            rows.extend(answer.get("items", []))
            read = [agent for agent in answer.get("agents", []) if isinstance(agent, dict)]
            agents.extend(read)
            # An older Cloud answers with one bounded read and no cursor.
            paged = paged and "next_cursor" in answer
            cursor = answer.get("next_cursor")
            if not isinstance(cursor, str) or not cursor:
                cursor = None
            elif cursor in followed:
                # A Cloud that keeps answering with the same cursor would never end.
                cursor, complete = None, False
            if cursor is None:
                complete = complete and answer.get("complete") is True
                break
            followed.add(cursor)
            # A page that goes on is whole when the Agent was read and named no issues.
            complete = complete and all(
                agent.get("state") == "read" and not agent.get("issues") for agent in read
            )
            if limit is not None or len(rows) >= MAX_LISTED_ITEMS:
                complete = False
                break

        def wanted(row: object) -> bool:
            if not isinstance(row, dict):
                return False
            row = cast("dict[str, Any]", row)
            return (
                (kind is None or row.get("kind") == kind)
                and (session_id is None or row.get("session_id") == session_id)
                and (agent_name is None or row.get("agent_name") == agent_name)
            )

        items = [row for row in rows if wanted(row)]
        notes = list(
            dict.fromkeys(
                str(agent.get("detail") or agent.get("reason") or agent.get("state"))
                for agent in agents
                if agent.get("state") != "read"
            )
        )
        if limit is None and cursor is not None:
            notes.append(f"Only the first {len(rows)} items read are listed.")
        if limit is not None and not paged:
            notes.append(
                "This Cayu Cloud doesn't page an Agent's inbox yet, so this is its one bounded "
                "read, oldest first."
            )
        for agent in agents:
            notes.extend(_issue_note(issue) for issue in agent.get("issues", []))
        listed = {row.get("id") for row in items}
        left_out = [
            item_id
            for agent in agents
            for item_id in agent.get("earlier_release_items") or ()
            if isinstance(item_id, str) and item_id not in listed
        ]
        earlier, unread = await self._read_left_out(left_out)
        items.extend(item for item in earlier if wanted(item))
        if unread:
            notes.append(
                f"{unread} item(s) asked before the Agent's current Release couldn't be read; "
                "list the inbox again."
            )
        if limit is None or not paged:
            items.sort(key=_oldest_first)
        return InboxListing(
            items, complete=complete and not unread, notes=notes, next_cursor=cursor
        )

    async def counts(self, *, session_id: str | None, agent_name: str | None) -> InboxCounts:
        try:
            answer = await asyncio.to_thread(
                self._request,
                "GET",
                "/counts",
                query=_filters(None, session_id, agent_name) or None,
            )
        except InboxClientError as exc:
            # A Cloud without the route reads `counts` as an item id.
            if exc.status_code in {404, 422}:
                return InboxCounts(None, "not_served")
            raise
        return InboxCounts.read(answer)

    async def _read_left_out(self, item_ids: list[str]) -> tuple[list[dict[str, Any]], int]:
        """The pending items among ids an older Cloud left out, and how many weren't read."""

        limit = asyncio.Semaphore(_LEFT_OUT_READS)

        async def read(item_id: str) -> dict[str, Any] | None:
            async with limit:
                state = await self.item(item_id)
            item = state.get("item")
            if state.get("status") == "unavailable":
                # Not proof it was settled: its current state can't be read.
                raise InboxClientError(
                    "unavailable", "This earlier-release item's current state can't be read."
                )
            if state.get("status") != "pending" or not isinstance(item, dict):
                return None
            return {**item, "asked_before_release": True}

        answers = await asyncio.gather(
            *(read(item_id) for item_id in dict.fromkeys(item_ids)), return_exceptions=True
        )
        unread = sum(1 for answer in answers if isinstance(answer, InboxClientError))
        for answer in answers:
            if isinstance(answer, BaseException) and not isinstance(answer, InboxClientError):
                raise answer
        return [answer for answer in answers if isinstance(answer, dict)], unread

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
            if exc.code == "profile_adoption_unsupported":
                raise InboxClientError(
                    "profile_adoption_unsupported",
                    "This Cayu Cloud can't settle an approval with a changed execution profile "
                    "adopted yet. Settle it in the Agent's Developer dashboard instead.",
                ) from None
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
    elif exc.status_code == 422 and any(
        path == "body.profile_adoption" for path, _ in exc.rejected_fields
    ):
        code = "profile_adoption_unsupported"
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


def _filters(kind: str | None, session_id: str | None, agent_name: str | None) -> dict[str, str]:
    """The listing filters sent upstream, those given."""

    return {
        name: value
        for name, value in (("kind", kind), ("session_id", session_id), ("agent_name", agent_name))
        if value is not None
    }


def _oldest_first(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("created_at") or ""), str(item.get("id") or "")


def _issue_note(issue: Any) -> str:
    if not isinstance(issue, dict):
        return "The inbox reported a problem reading one source."
    where = issue.get("session_id")
    detail = issue.get("detail") or issue.get("code") or "unreadable"
    return f"{where}: {detail}" if where else str(detail)
