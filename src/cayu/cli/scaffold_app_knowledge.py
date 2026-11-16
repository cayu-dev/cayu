"""The Knowledge area of an Agent's web app, for a project with the `knowledge` capability.

What the Agent knows, for the people who use it: plain statements, searchable, with where each
came from and when it was learned, the suggestions waiting in Cayu's Inbox, and correcting a
statement or having it forgotten. It adds, on top of the web app:

- `integrations/http/generated/knowledge.py`, the generated `/app/knowledge` routes;
- the knowledge layer of the web app (`scaffold_app_web.KNOWLEDGE`): the Knowledge area in
  `web/src/generated/knowledge/`, listed in the app's area registry, and its smoke test;
- the routes in the Agent's own router (`integrations/http/__init__.py`), with the access its
  people have there: `operator_knowledge` for its operators (`scaffold_web_app.py`), and
  `member_knowledge`, through Cayu's resource-access policy and checked per organization, for
  the organizations' members (`scaffold_organizations.py`);
- `tests/test_knowledge_area.py`, which checks who reads, searches, corrects and forgets what.

The Developer dashboard's knowledge pages stay as they are: the full view, for builders.
"""

from __future__ import annotations

# Where the generated routes go in the project.
KNOWLEDGE_ROUTES = "integrations/http/generated/knowledge.py"
KNOWLEDGE_TEST = "tests/test_knowledge_area.py"

_KNOWLEDGE_PY = '''"""GENERATED. `/app/knowledge`: what the Agent knows, for the people who use it.

- `GET /app/knowledge` lists what the Agent knows (its kept knowledge) as plain statements,
  latest first, or the ones that match `q`. Each says where it came from, when it was
  learned, whether a person corrected it, and whether the signed-in person may correct it or
  have it forgotten.
- `GET /app/knowledge/suggestions` lists what the Agent asked to remember that nobody has kept
  or dropped yet: the records Cayu's Inbox lists as knowledge proposals, and links there.
- `POST /app/knowledge/correct` replaces a statement's text with the person's correction. It
  stays kept, and says who corrected it.
- `POST /app/knowledge/forget` archives a statement, so the Agent no longer finds or recalls
  it. The archived entry records who forgot it and why, as the Developer dashboard's archive
  does.

A `KnowledgeAccess` says who reads what and who changes what. `operator_knowledge` is for an
Agent whose people are its operators: they read all of its knowledge, and those whose role may
change things correct and forget. `member_knowledge` is for one whose people reach knowledge
through Cayu's resource-access policy: with organizations, each person reads their own
organizations' knowledge only, and `may_change` decides for each organization. Nothing here
widens what Cayu allows.

The Developer dashboard (/cayu/) stays the full view, with revisions, chunks, sources and
recall evidence; this one shows none of them. The reads answer with an ETag
(`cayu.server.etag`), so reading an unchanged list costs a 304 with no body.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from urllib.parse import urlencode
from uuid import uuid4

from cayu import CayuApp
from cayu.resource_access import ResourceAccessDenied
from cayu.server.etag import tagged_json
from cayu.storage import (
    KnowledgeAccessScope,
    KnowledgeActivationConflict,
    KnowledgeActorType,
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeIndexer,
    KnowledgeIndexRequest,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgePublicationConflict,
    KnowledgeQuery,
    KnowledgeReviewWorkflow,
    KnowledgeRevisionConflict,
    KnowledgeSearchMode,
    KnowledgeSearchResult,
    KnowledgeStatus,
)
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from integrations.http.generated.whoami import Person, PersonDependency

# Where suggestions are decided: Cayu's Inbox, on the Developer dashboard.
INBOX_PAGE = "/cayu/inbox"
# How many statements a read lists at most, and how many by default.
MAX_STATEMENTS = 500
DEFAULT_STATEMENTS = 50
# The longest correction a person may write.
MAX_CORRECTION = 8000
# Where a correction is recorded in the entry's metadata.
CORRECTION = "corrected"
# The reviewer version a forgotten entry records, as the Developer dashboard's archive does.
REVIEWER_VERSION = "1"
# A page's statements are read up to this many bytes.
_READ_BYTES = 1_048_576

_UNAVAILABLE = "That statement isn't there any more, or you can't see it."
_CHANGED = "This statement changed since you opened it. Reload it, then try again."


class KnowledgeReader(Protocol):
    """What the routes read and correct through: Cayu's knowledge store under one scope, or
    the person's scoped access (`(await app.access(subject)).knowledge(...)`)."""

    async def list_entries(self, query: KnowledgeListQuery) -> KnowledgeListResult: ...

    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult: ...

    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None: ...

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
    ) -> KnowledgeEntry: ...


@dataclass(frozen=True, slots=True)
class KnowledgeAccess:
    """Who reads which knowledge, and who may change which statement.

    ``scope`` and ``namespace`` are the Agent's (`knowledge/retrieval.py`). ``read`` gives the
    reader for the signed-in person; ``may_change`` says whether they may correct or forget
    one entry. ``group_label`` names the label that says who an entry belongs to (with
    organizations, `organization`), which the page shows. ``review_page`` is where
    suggestions are decided, or None when the people of this app can't open Cayu's Inbox.
    ``link_conversations`` links a statement to the conversation it was learned in."""

    scope: KnowledgeAccessScope
    namespace: str
    read: Callable[[CayuApp, Person, KnowledgeAccessScope], Awaitable[KnowledgeReader]]
    may_change: Callable[[Person, KnowledgeEntry], Awaitable[bool]]
    group_label: str | None = None
    review_page: str | None = INBOX_PAGE
    link_conversations: bool = True


class _StoreReader:
    """Cayu's knowledge store, read and written under one scope."""

    def __init__(self, app: CayuApp, scope: KnowledgeAccessScope) -> None:
        if app.knowledge_store is None:
            raise HTTPException(404, "This Agent keeps no knowledge.")
        self._store = app.knowledge_store
        self._scope = scope

    async def list_entries(self, query: KnowledgeListQuery) -> KnowledgeListResult:
        return await self._store.list_entries(query, access_scope=self._scope)

    async def search(self, query: KnowledgeQuery) -> KnowledgeSearchResult:
        return await self._store.search(query, access_scope=self._scope)

    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None:
        return await self._store.get_entry(entry_id, access_scope=self._scope)

    async def append_entry_revision(
        self,
        entry: KnowledgeEntry,
        chunks: list[KnowledgeChunk] | None = None,
        *,
        expected_revision: int,
    ) -> KnowledgeEntry:
        return await self._store.append_entry_revision(
            entry, chunks, expected_revision=expected_revision, access_scope=self._scope
        )


async def _every_entry(
    app: CayuApp, person: Person, scope: KnowledgeAccessScope
) -> KnowledgeReader:
    return _StoreReader(app, scope)


async def _their_entries(
    app: CayuApp, person: Person, scope: KnowledgeAccessScope
) -> KnowledgeReader:
    if app.knowledge_store is None:
        raise HTTPException(404, "This Agent keeps no knowledge.")
    access = await app.access(person.subject)
    return access.knowledge(access_scope=scope)


async def _by_role(person: Person, entry: KnowledgeEntry) -> bool:
    return person.may_change


def operator_knowledge(
    *, scope: KnowledgeAccessScope, namespace: str
) -> KnowledgeAccess:
    """For an Agent whose people are its operators, who see all of its work on the Developer
    dashboard: everyone reads all of its knowledge, and a role that may change things (Cayu
    Cloud's `admin` and `developer`) corrects and forgets."""

    return KnowledgeAccess(
        scope=scope, namespace=namespace, read=_every_entry, may_change=_by_role
    )


def member_knowledge(
    *,
    scope: KnowledgeAccessScope,
    namespace: str,
    label: str,
    may_change: Callable[[Person, str], Awaitable[bool]],
) -> KnowledgeAccess:
    """For an Agent whose people reach knowledge through Cayu's resource-access policy: each
    person reads only what the policy grants them (with organizations, their organizations'
    knowledge), and ``may_change(person, value)`` says whether they may correct and forget
    the entries whose ``label`` is ``value`` (an organization's admins, say). An entry without
    the label changes for nobody. They can't open Cayu's Inbox, so suggestions don't link
    there, and statements don't link to the conversation they were learned in, which may be
    another person's."""

    async def allowed(person: Person, entry: KnowledgeEntry) -> bool:
        value = entry.labels.get(label)
        return value is not None and await may_change(person, value)

    return KnowledgeAccess(
        scope=scope,
        namespace=namespace,
        read=_their_entries,
        may_change=allowed,
        group_label=label,
        review_page=None,
        link_conversations=False,
    )


SourceKind = Literal["conversation", "document", "person", "agent", "app"]


class SourceView(BaseModel):
    kind: SourceKind
    # What the page says, such as "A conversation" or "handbook.pdf".
    label: str
    # The conversation it was learned in, when the person may open it from here.
    conversation: str | None = None


class CorrectionView(BaseModel):
    by: str
    at: str


class Statement(BaseModel):
    id: str
    # What a correction or forget names, so it changes only what the person saw.
    version: int
    text: str
    source: SourceView
    learned_at: str
    corrected: CorrectionView | None
    # Who the entry belongs to, such as an organization; None when the app has no groups.
    group: str | None
    may_change: bool
    # False when the text shown has secrets taken out: a correction would store those.
    may_correct: bool


class KnowledgeView(BaseModel):
    statements: list[Statement]
    query: str | None
    # More statements than were read: ask for more with a higher `limit`.
    more: bool


class Suggestion(BaseModel):
    id: str
    text: str
    source: SourceView
    suggested_at: str
    group: str | None
    # Where to keep or drop it: Cayu's Inbox, filtered to it.
    review: str | None


class SuggestionsView(BaseModel):
    suggestions: list[Suggestion]
    more: bool
    # Who keeps or drops suggestions, in a sentence the page shows.
    reviewed_by: str


class CorrectRequest(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    version: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=MAX_CORRECTION)


class ForgetRequest(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    version: int = Field(ge=1)
    reason: str | None = Field(default=None, max_length=500)


class Changed(BaseModel):
    id: str
    version: int
    status: Literal["corrected", "forgotten"]


def router(
    app: CayuApp, *, person: PersonDependency, access: KnowledgeAccess
) -> APIRouter:
    """The knowledge routes, relative to /app."""

    routes = APIRouter()

    async def reader(signed_in: Person) -> KnowledgeReader:
        try:
            return await access.read(app, signed_in, access.scope)
        except ResourceAccessDenied:
            raise HTTPException(
                404, "This Agent keeps no knowledge you can see."
            ) from None

    @routes.get("/knowledge")
    async def list_knowledge(
        request: Request,
        response: Response,
        signed_in: Person = Depends(person),  # noqa: B008
        q: str | None = Query(default=None, max_length=200),
        limit: int = Query(default=DEFAULT_STATEMENTS, ge=1, le=MAX_STATEMENTS),
    ) -> KnowledgeView:
        source = await reader(signed_in)
        query = (q or "").strip() or None
        try:
            if query is None:
                entries, more = await _listed(
                    source, access, KnowledgeStatus.ACTIVE, limit
                )
            else:
                entries, more = await _matching(source, access, query, limit)
        except (ResourceAccessDenied, PermissionError):
            entries, more = [], False
        changes = _Changes(access, signed_in)
        view = KnowledgeView(
            statements=[
                _statement(app, access, entry, may_change=await changes(entry))
                for entry in entries
            ],
            query=query,
            more=more,
        )
        # An unchanged list costs a 304 with no body.
        return tagged_json(request, response, view)

    @routes.get("/knowledge/suggestions")
    async def list_suggestions(
        request: Request,
        response: Response,
        signed_in: Person = Depends(person),  # noqa: B008
        limit: int = Query(default=DEFAULT_STATEMENTS, ge=1, le=MAX_STATEMENTS),
    ) -> SuggestionsView:
        source = await reader(signed_in)
        try:
            entries, more = await _listed(
                source, access, KnowledgeStatus.PENDING, limit
            )
        except (ResourceAccessDenied, PermissionError):
            entries, more = [], False
        view = SuggestionsView(
            suggestions=[_suggestion(app, access, entry) for entry in entries],
            more=more,
            reviewed_by=(
                "Keep or drop them in Cayu's Inbox."
                if access.review_page is not None
                else "The Agent's operators keep or drop them."
            ),
        )
        return tagged_json(request, response, view)

    @routes.post("/knowledge/correct")
    async def correct(
        body: CorrectRequest,
        signed_in: Person = Depends(person),  # noqa: B008
    ) -> Changed:
        source = await reader(signed_in)
        entry = await _changeable(access, source, signed_in, body.id, body.version)
        if app.redact_json(entry.text) != entry.text:
            raise HTTPException(
                409,
                "This statement holds a secret the page doesn't show, so it can't be "
                "corrected here. Correct it on the Developer dashboard.",
            )
        text = body.text.strip()
        if not text:
            raise HTTPException(422, "Write the corrected statement.")
        if text == entry.text:
            return Changed(id=entry.id, version=entry.revision, status="corrected")
        successor, chunks = _corrected(entry, text, signed_in)
        try:
            saved = await source.append_entry_revision(
                successor, chunks, expected_revision=entry.revision
            )
        except (KnowledgeRevisionConflict, KnowledgeActivationConflict):
            raise HTTPException(409, _CHANGED) from None
        except (ResourceAccessDenied, PermissionError):
            raise HTTPException(404, _UNAVAILABLE) from None
        return Changed(id=saved.id, version=saved.revision, status="corrected")

    @routes.post("/knowledge/forget")
    async def forget(
        body: ForgetRequest,
        signed_in: Person = Depends(person),  # noqa: B008
    ) -> Changed:
        source = await reader(signed_in)
        entry = await _changeable(access, source, signed_in, body.id, body.version)
        labels = {}
        if access.group_label is not None:
            labels[access.group_label] = entry.labels[access.group_label]
        workflow = KnowledgeReviewWorkflow(
            app.knowledge_store,
            access_scope=access.scope,
            namespace=access.namespace,
            labels=labels,
        )
        reason = (
            body.reason or ""
        ).strip() or "Forgotten from the app's Knowledge page."
        try:
            archived = await workflow.archive_reviewed(
                entry.id,
                expected_revision=entry.revision,
                operation_id=f"app-knowledge-forget-{uuid4().hex}",
                reviewer_identity=signed_in.subject,
                reviewer_version=REVIEWER_VERSION,
                reason=reason,
                annotations={"channel": "app", "page": "knowledge"},
            )
        except (
            KnowledgeRevisionConflict,
            KnowledgeActivationConflict,
            KnowledgePublicationConflict,
        ):
            raise HTTPException(409, _CHANGED) from None
        except (KeyError, PermissionError):
            raise HTTPException(404, _UNAVAILABLE) from None
        return Changed(id=entry.id, version=archived.entry.revision, status="forgotten")

    return routes


async def _listed(
    source: KnowledgeReader,
    access: KnowledgeAccess,
    status: KnowledgeStatus,
    limit: int,
) -> tuple[list[KnowledgeEntry], bool]:
    """The newest ``limit`` entries in ``status`` (kept, or waiting for review), and whether
    there are more."""

    listed = await source.list_entries(
        KnowledgeListQuery(
            namespace=access.namespace,
            statuses=[status],
            limit=limit,
            max_bytes=_READ_BYTES,
            count_total=False,
        )
    )
    return [item.entry for item in listed.entries], _more(listed)


async def _matching(
    source: KnowledgeReader, access: KnowledgeAccess, query: str, limit: int
) -> tuple[list[KnowledgeEntry], bool]:
    try:
        search = KnowledgeQuery(
            text=query,
            namespace=access.namespace,
            statuses=[KnowledgeStatus.ACTIVE],
            mode=KnowledgeSearchMode.KEYWORD,
            limit=limit,
            max_bytes=_READ_BYTES,
        )
    except ValueError:
        # No word to search for ("?", a lone quote): nothing matches it.
        return [], False
    found = await source.search(search)
    # A hit is one part of an entry; an entry shows once, at its best part's place.
    entries: dict[str, KnowledgeEntry] = {}
    for hit in found.hits:
        entries.setdefault(hit.entry.id, hit.entry)
    return list(entries.values()), found.truncated


def _more(listed: KnowledgeListResult) -> bool:
    return listed.truncated and (
        listed.total_entries_known is None
        or listed.total_entries_known > len(listed.entries)
    )


class _Changes:
    """Whether the person may change each entry, asking `may_change` once per group."""

    def __init__(self, access: KnowledgeAccess, person: Person) -> None:
        self._access = access
        self._person = person
        self._known: dict[str | None, bool] = {}

    async def __call__(self, entry: KnowledgeEntry) -> bool:
        group = _group(self._access, entry)
        if group not in self._known:
            self._known[group] = await self._access.may_change(self._person, entry)
        return self._known[group]


async def _changeable(
    access: KnowledgeAccess,
    source: KnowledgeReader,
    person: Person,
    entry_id: str,
    version: int,
) -> KnowledgeEntry:
    """The kept entry the person saw, which they may change: 404 when they can't see it, 403
    when they may not change it, 409 when it changed since they read it."""

    try:
        entry = await source.get_entry(entry_id)
    except (ResourceAccessDenied, PermissionError, ValueError):
        entry = None
    if (
        entry is None
        or entry.status is not KnowledgeStatus.ACTIVE
        or entry.namespace != access.namespace
    ):
        raise HTTPException(404, _UNAVAILABLE)
    if not await access.may_change(person, entry):
        raise HTTPException(
            403,
            "Only the people who may change this Agent correct or forget what it knows.",
        )
    if entry.revision != version:
        raise HTTPException(409, _CHANGED)
    return entry


def _corrected(
    entry: KnowledgeEntry, text: str, person: Person
) -> tuple[KnowledgeEntry, list[KnowledgeChunk]]:
    """The entry's next revision with ``text``, indexed afresh, kept, and saying who
    corrected it; where it came from and when it was learned stay as they were."""

    now = datetime.now(UTC)
    built = KnowledgeIndexer().build(
        KnowledgeIndexRequest(
            text=text,
            entry_id=entry.id,
            namespace=entry.namespace,
            labels=entry.labels,
            kind=entry.kind,
            visibility=entry.visibility,
            status=entry.status,
            created_by_type=entry.created_by_type,
            created_by=entry.created_by,
            source_type=entry.source_type,
            source_uri=entry.source_uri,
            source_id=entry.source_id,
            aspects=entry.aspects,
            impact_targets=entry.impact_targets,
            importance=entry.importance,
            importance_source=entry.importance_source,
            confidence=entry.confidence,
            expires_at=entry.expires_at,
            title=entry.title,
            metadata={
                **entry.metadata,
                CORRECTION: {"by": person.subject, "at": now.isoformat()},
            },
        )
    )
    if built.truncated or built.entry_text_truncated:
        raise HTTPException(422, "That correction is too long to keep.")
    revision = entry.revision + 1
    successor = built.entry.model_copy(
        update={
            "revision": revision,
            "created_at": entry.created_at,
            "updated_at": max(now, entry.updated_at),
        }
    )
    chunks = [
        chunk.model_copy(
            update={
                "id": f"{entry.id}:r{revision}:{chunk.chunk_index}",
                "entry_revision": revision,
            }
        )
        for chunk in built.chunks
    ]
    return successor, chunks


def _statement(
    app: CayuApp, access: KnowledgeAccess, entry: KnowledgeEntry, *, may_change: bool
) -> Statement:
    text = app.redact_json(entry.text)
    return Statement(
        id=entry.id,
        version=entry.revision,
        text=text,
        source=_source(app, access, entry),
        learned_at=_iso(entry.created_at),
        corrected=_correction(app, entry),
        group=_group(access, entry),
        may_change=may_change,
        may_correct=may_change and text == entry.text,
    )


def _suggestion(
    app: CayuApp, access: KnowledgeAccess, entry: KnowledgeEntry
) -> Suggestion:
    text = app.redact_json(entry.text)
    review = None
    if access.review_page is not None:
        # Cayu's Inbox finds a knowledge proposal by its title, else the start of its text.
        shown = app.redact_json(entry.title) if entry.title else text
        first = shown.strip().splitlines()[0][:80].strip() if shown.strip() else ""
        query = {"kind": "knowledge", "q": first} if first else {"kind": "knowledge"}
        review = f"{access.review_page}?{urlencode(query)}"
    return Suggestion(
        id=entry.id,
        text=text,
        source=_source(app, access, entry),
        suggested_at=_iso(entry.created_at),
        group=_group(access, entry),
        review=review,
    )


def _source(app: CayuApp, access: KnowledgeAccess, entry: KnowledgeEntry) -> SourceView:
    """Where an entry came from, in a few words."""

    if entry.source_type in ("tool", "cayu.interaction"):
        conversation = None
        if access.link_conversations and entry.source_id:
            conversation = app.project_session_id_for_exposure(entry.source_id)
        return SourceView(
            kind="conversation", label="A conversation", conversation=conversation
        )
    if entry.created_by_type is KnowledgeActorType.USER:
        return SourceView(kind="person", label=app.redact_json(entry.created_by))
    if entry.source_uri and not entry.source_uri.startswith("cayu://"):
        name = entry.source_uri.rstrip("/").rsplit("/", 1)[-1] or entry.source_uri
        return SourceView(kind="document", label=app.redact_json(name))
    if entry.created_by_type is KnowledgeActorType.MODEL:
        return SourceView(kind="agent", label="The Agent")
    return SourceView(kind="app", label="The Agent's builders")


def _correction(app: CayuApp, entry: KnowledgeEntry) -> CorrectionView | None:
    found: Any = entry.metadata.get(CORRECTION)
    if not isinstance(found, dict):
        return None
    by, at = found.get("by"), found.get("at")
    if not isinstance(by, str) or not isinstance(at, str):
        return None
    return CorrectionView(by=app.redact_json(by), at=at)


def _group(access: KnowledgeAccess, entry: KnowledgeEntry) -> str | None:
    if access.group_label is None:
        return None
    return entry.labels.get(access.group_label)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()
'''

_OPERATOR_TEST_PY = '''"""The Knowledge area's routes: every operator reads what the Agent knows; some change it.

Deployed, the routes use the auth `[tool.cayu.serve].auth` names; here a stand-in returns the
operator each test signs in as, as in `tests/test_web_app.py`. Cayu Cloud's `admin` and
`developer` correct a statement or have it forgotten; a `member` reads.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from cayu import CayuApp
from cayu.server.auth import AuthContext
from cayu.sessions import InMemorySessionStore
from cayu.storage import (
    InMemoryKnowledgeStore,
    KnowledgeActorType,
    KnowledgeEntry,
    KnowledgeStatus,
)
from integrations.http import build_router
from knowledge.retrieval import KNOWLEDGE_NAMESPACE, build_knowledge_scope

PAID = "Invoices are paid on the 15th of each month."
TONE = "Customers prefer short answers."
SUGGESTED = "The office closes at noon on Fridays."


def _app() -> CayuApp:
    store = InMemoryKnowledgeStore(access_scope=build_knowledge_scope())
    return CayuApp(session_store=InMemorySessionStore(), knowledge_store=store)


def _remember(client: TestClient, app: CayuApp) -> None:
    entries = (
        KnowledgeEntry(
            id="paid",
            namespace=KNOWLEDGE_NAMESPACE,
            text=PAID,
            source_type="tool",
            source_id="a-conversation",
            created_by_type=KnowledgeActorType.MODEL,
            created_by="agent",
        ),
        KnowledgeEntry(
            id="tone",
            namespace=KNOWLEDGE_NAMESPACE,
            text=TONE,
            created_by_type=KnowledgeActorType.USER,
            created_by="ada",
        ),
        KnowledgeEntry(
            id="fridays",
            namespace=KNOWLEDGE_NAMESPACE,
            text=SUGGESTED,
            status=KnowledgeStatus.PENDING,
        ),
    )
    for entry in entries:
        client.portal.call(app.knowledge_store.create_entry, entry)


def _signed_in(role: str | None):
    """A stand-in for the auth target: the caller is `ada`, with `role`, or nobody."""

    async def auth(request: Request) -> AuthContext:
        if role == "nobody":
            raise HTTPException(status_code=401, detail="Authentication required.")
        return AuthContext(subject="ada", claims={} if role is None else {"role": role})

    return auth


def _client(app: CayuApp, role: str | None) -> TestClient:
    server = FastAPI()
    server.include_router(build_router(app, development=False, auth=_signed_in(role)))
    return TestClient(server)


def _statements(client: TestClient, **query: str) -> dict[str, dict]:
    response = client.get("/app/knowledge", params=query)
    assert response.status_code == 200, response.text
    return {item["id"]: item for item in response.json()["statements"]}


def test_every_knowledge_route_needs_a_signed_in_operator() -> None:
    with _client(_app(), "nobody") as client:
        for path in ("knowledge", "knowledge/suggestions"):
            assert client.get(f"/app/{path}").status_code == 401, path
        for path in ("knowledge/correct", "knowledge/forget"):
            body = {"id": "paid", "version": 1, "text": "x"}
            assert client.post(f"/app/{path}", json=body).status_code == 401, path


def test_every_operator_reads_and_searches_what_the_agent_knows() -> None:
    app = _app()
    with _client(app, "member") as client:
        _remember(client, app)
        known = _statements(client)
        # Kept knowledge only: the suggestion waits for someone to keep it.
        assert set(known) == {"paid", "tone"}
        assert known["paid"]["text"] == PAID
        assert known["paid"]["source"]["kind"] == "conversation"
        assert known["tone"]["source"] == {
            "kind": "person",
            "label": "ada",
            "conversation": None,
        }
        # A member reads; they don't change what the Agent knows.
        assert not any(item["may_change"] for item in known.values())
        assert set(_statements(client, q="invoice")) == {"paid"}
        assert _statements(client, q="holidays") == {}
        # A search with no word in it matches nothing, rather than failing.
        for nothing in ("?", "\u20ac", '"', "..."):
            assert _statements(client, q=nothing) == {}, nothing

        suggestions = client.get("/app/knowledge/suggestions").json()
        (suggestion,) = suggestions["suggestions"]
        assert suggestion["text"] == SUGGESTED
        # It links to the same record in Cayu's Inbox.
        assert suggestion["review"].startswith("/cayu/inbox?kind=knowledge&q=")

        # An unchanged list costs a 304 with no body.
        tag = client.get("/app/knowledge").headers["etag"]
        unchanged = client.get("/app/knowledge", headers={"if-none-match": tag})
        assert unchanged.status_code == 304


def test_only_admins_and_developers_correct_and_forget() -> None:
    app = _app()
    with _client(app, "member") as client:
        _remember(client, app)
        correct = {"id": "paid", "version": 1, "text": "Invoices are paid on the 20th."}
        assert client.post("/app/knowledge/correct", json=correct).status_code == 403
        forget = {"id": "paid", "version": 1}
        assert client.post("/app/knowledge/forget", json=forget).status_code == 403
    with _client(app, "developer") as client:
        assert all(item["may_change"] for item in _statements(client).values())
        corrected = client.post("/app/knowledge/correct", json=correct)
        assert corrected.status_code == 200, corrected.text
        paid = _statements(client)["paid"]
        assert paid["text"] == "Invoices are paid on the 20th."
        assert paid["corrected"]["by"] == "ada"
        # Where it was learned stays as it was.
        assert paid["source"]["kind"] == "conversation"
        assert set(_statements(client, q="20th")) == {"paid"}
        assert _statements(client, q="15th") == {}
        # A change to what someone saw before it changed is refused: a correction or a
        # forget names the version the person opened, not the one the page read since.
        stale = {**correct, "text": "Invoices are paid on the 1st."}
        assert client.post("/app/knowledge/correct", json=stale).status_code == 409
        opened = {"id": "paid", "version": 1}
        assert client.post("/app/knowledge/forget", json=opened).status_code == 409
        assert _statements(client)["paid"]["text"] == "Invoices are paid on the 20th."
    with _client(app, "admin") as client:
        forget = {"id": "paid", "version": paid["version"], "reason": "Out of date."}
        assert client.post("/app/knowledge/forget", json=forget).status_code == 200
        assert set(_statements(client)) == {"tone"}
        assert _statements(client, q="invoice") == {}
        # A suggestion isn't forgotten here: it is kept or dropped in Cayu's Inbox.
        suggestion = {"id": "fridays", "version": 1}
        assert client.post("/app/knowledge/forget", json=suggestion).status_code == 404
        missing = {"id": "no-such-entry", "version": 1, "text": "x"}
        assert client.post("/app/knowledge/correct", json=missing).status_code == 404
'''

_MEMBER_TEST_PY = '''"""The Knowledge area with organizations: each member reads only their organizations' knowledge.

A member reads and searches what the Agent knows in their organizations, and the suggestions
waiting there; an organization's admins correct its statements and have them forgotten. Nobody
sees, corrects or forgets another organization's. These read the application's memberships
(`CAYU_TEST_POSTGRES_DSN`), with the demonstration people of `operations/demo_seed.py`: alice is
acme's admin and carol its member, dave is globex's admin and bob its member.
"""

from __future__ import annotations

from alembic import command
from fastapi import FastAPI
from fastapi.testclient import TestClient
from test_app_database import ALEMBIC, requires_postgres

from cayu import CayuApp
from cayu.sessions import InMemorySessionStore
from cayu.storage import InMemoryKnowledgeStore, KnowledgeEntry, KnowledgeStatus
from integrations.http import build_router
from knowledge.retrieval import KNOWLEDGE_NAMESPACE, build_knowledge_scope
from operations.demo_seed import seed
from policies.access import build_access_policy

ACME = "Acme's invoices are paid on the 15th of each month."
GLOBEX = "Globex's invoices are paid every quarter."


def _client() -> tuple[TestClient, InMemoryKnowledgeStore]:
    store = InMemoryKnowledgeStore(access_scope=build_knowledge_scope())
    app = CayuApp(
        session_store=InMemorySessionStore(),
        knowledge_store=store,
        resource_access_policy=build_access_policy(),
    )
    server = FastAPI()
    server.include_router(build_router(app, development=True))
    return TestClient(server), store


def _remember(client: TestClient, store: InMemoryKnowledgeStore) -> None:
    """Each organization's kept statement and suggestion."""

    for entry in (
        _entry("acme-invoices", ACME, "acme"),
        _entry("globex-invoices", GLOBEX, "globex"),
        _entry("acme-euros", "Acme wants invoices in euros.", "acme", pending=True),
        _entry("globex-yen", "Globex wants invoices in yen.", "globex", pending=True),
    ):
        client.portal.call(store.create_entry, entry)


def _entry(identifier: str, text: str, organization: str, *, pending: bool = False):
    return KnowledgeEntry(
        id=identifier,
        namespace=KNOWLEDGE_NAMESPACE,
        text=text,
        labels={"organization": organization},
        status=KnowledgeStatus.PENDING if pending else KnowledgeStatus.ACTIVE,
    )


def _sign_in(client: TestClient, subject: str) -> None:
    client.cookies.clear()
    assert client.post("/app/dev/sign-in", json={"subject": subject}).status_code == 204


def _statements(client: TestClient, **query: str) -> dict[str, dict]:
    response = client.get("/app/knowledge", params=query)
    assert response.status_code == 200, response.text
    return {item["id"]: item for item in response.json()["statements"]}


def _suggested(client: TestClient) -> set[str]:
    answer = client.get("/app/knowledge/suggestions").json()
    return {item["id"] for item in answer["suggestions"]}


@requires_postgres
def test_a_member_never_sees_corrects_or_forgets_another_organizations_knowledge(
    database,
) -> None:
    command.upgrade(ALEMBIC, "head")
    seed()
    client, store = _client()
    with client:
        _remember(client, store)
        for subject in ("dave", "bob"):
            _sign_in(client, subject)
            assert set(_statements(client)) == {"globex-invoices"}
            assert set(_statements(client, q="invoices")) == {"globex-invoices"}
            assert _statements(client, q="acme") == {}
            assert _suggested(client) == {"globex-yen"}
            # Acme's statement looks missing to globex's people, admins included.
            correct = {"id": "acme-invoices", "version": 1, "text": "Wrong."}
            corrected = client.post("/app/knowledge/correct", json=correct)
            assert corrected.status_code == 404
            forget = {"id": "acme-invoices", "version": 1}
            assert client.post("/app/knowledge/forget", json=forget).status_code == 404
        _sign_in(client, "alice")
        assert _statements(client)["acme-invoices"]["text"] == ACME
        assert _statements(client)["acme-invoices"]["group"] == "acme"


@requires_postgres
def test_an_organizations_admins_correct_and_forget_its_knowledge(database) -> None:
    command.upgrade(ALEMBIC, "head")
    seed()
    client, store = _client()
    with client:
        _remember(client, store)
        _sign_in(client, "carol")
        acme = _statements(client)["acme-invoices"]
        # A member reads their organization's knowledge but doesn't change it.
        assert acme["may_change"] is False
        correct = {"id": "acme-invoices", "version": 1, "text": "Paid on the 20th."}
        assert client.post("/app/knowledge/correct", json=correct).status_code == 403
        forget = {"id": "acme-invoices", "version": 1}
        assert client.post("/app/knowledge/forget", json=forget).status_code == 403
        # Suggestions are kept or dropped by the Agent's operators, not from the app.
        answer = client.get("/app/knowledge/suggestions").json()
        assert [item["review"] for item in answer["suggestions"]] == [None]

        _sign_in(client, "alice")
        assert _statements(client)["acme-invoices"]["may_change"] is True
        corrected = client.post("/app/knowledge/correct", json=correct)
        assert corrected.status_code == 200, corrected.text
        acme = _statements(client)["acme-invoices"]
        assert (acme["text"], acme["corrected"]["by"]) == ("Paid on the 20th.", "alice")
        forget = {"id": "acme-invoices", "version": acme["version"]}
        assert client.post("/app/knowledge/forget", json=forget).status_code == 200
        assert _statements(client) == {}

        # Globex's knowledge is as it was.
        _sign_in(client, "dave")
        assert _statements(client)["globex-invoices"]["text"] == GLOBEX
'''

# Where the generated routes' line goes in the Agent's router docstring, and what it says.
_ROUTES_LINE = "  from its environment (`fixed_settings.py`).\n"
_ROUTES_LINE_ORGANIZATIONS = "  reads from its environment (`fixed_settings.py`).\n"
_KNOWLEDGE_LINE = (
    "- /app/knowledge: what the Agent knows, for the web app's Knowledge area\n"
    "  (integrations/http/generated/knowledge.py). __WHO__\n"
)
_OPERATORS_WHO = (
    "Every operator reads it; `admin` and\n  `developer` correct it and have it forgotten."
)
_MEMBERS_WHO = (
    "Each member reads their organizations';\n"
    "  an organization's admins correct it and have it forgotten."
)

_OPERATOR_IMPORTS = (
    "from integrations.http.generated.task_runs import operator_tasks\n",
    "from integrations.http.generated.knowledge import operator_knowledge\n"
    "from integrations.http.generated.knowledge import router as knowledge_router\n"
    "from integrations.http.generated.task_runs import operator_tasks\n",
)
_OPERATOR_IMPORTS_AFTER = (
    "from integrations.http.overview_metrics import overview_metrics\n",
    "from integrations.http.overview_metrics import overview_metrics\n"
    "from knowledge.retrieval import KNOWLEDGE_NAMESPACE, build_knowledge_scope\n",
)
_OPERATOR_ROUTES = (
    '\n    @router.get("/agent")\n',
    """
    # What the Agent knows, for the web app's Knowledge area: every operator reads it, and a
    # role that may change things (`admin`, `developer`) corrects it and has it forgotten.
    scope = build_knowledge_scope()
    if scope is not None:
        router.include_router(
            knowledge_router(
                app,
                person=person,
                access=operator_knowledge(scope=scope, namespace=KNOWLEDGE_NAMESPACE),
            )
        )

    @router.get("/agent")
""",
)

_MEMBER_IMPORTS = (
    "from integrations.http.generated import Person, build_generated_router\n",
    "from integrations.http.generated import Person, build_generated_router\n"
    "from integrations.http.generated.knowledge import member_knowledge\n"
    "from integrations.http.generated.knowledge import router as knowledge_router\n",
)
_MEMBER_IMPORTS_AFTER = (
    "    build_authenticate,\n)\n",
    "    build_authenticate,\n)\n"
    "from knowledge.retrieval import KNOWLEDGE_NAMESPACE, build_knowledge_scope\n",
)
_MEMBER_ROUTES = (
    '\n    @router.get("/me")\n',
    """
    async def organization_admin(signed_in: Person, organization: str) -> bool:
        record = await _membership(signed_in.subject, organization)
        return record is not None and record.role == "admin"

    # What the Agent knows, for the web app's Knowledge area: each member reads their
    # organizations' knowledge, through Cayu's resource-access policy, and an organization's
    # admins correct its statements and have them forgotten.
    scope = build_knowledge_scope()
    if scope is not None:
        router.include_router(
            knowledge_router(
                app,
                person=person,
                access=member_knowledge(
                    scope=scope,
                    namespace=KNOWLEDGE_NAMESPACE,
                    label="organization",
                    may_change=organization_admin,
                ),
            )
        )

    @router.get("/me")
""",
)

GUIDANCE = """
## The Knowledge area

The web app's Knowledge area (`web/src/generated/knowledge/`, reading `/app/knowledge` from
`integrations/http/generated/knowledge.py`) shows the app's people what the Agent knows, as
plain statements they can search, with where each came from and when it was learned. It lists
the suggestions waiting to be kept, which are the same records as the knowledge proposals in
Cayu's Inbox. __WHO_CHANGES__ It shows no revisions, chunks or recall evidence: the Developer
dashboard (`/cayu/`) stays the full view. `tests/test_knowledge_area.py` checks who reads and
changes what.
"""
_OPERATORS_CHANGE = (
    "Cayu Cloud's `admin` and `developer` (and you under `cayu serve --dev`) correct a statement "
    "or have it forgotten; a forgotten statement is archived with who forgot it and why."
)
_MEMBERS_CHANGE = (
    "Each member sees only their organizations' knowledge, through the resource-access policy, "
    "and an organization's admins correct its statements or have them forgotten. The Agent's "
    "operators keep or drop the suggestions."
)


def _replace_once(text: str, old: str, new: str, path: str) -> str:
    """``text`` with ``old`` replaced by ``new``, which must be in it exactly once."""

    if text.count(old) != 1:
        raise RuntimeError(f"{path} isn't the web app's router the Knowledge area goes into.")
    return text.replace(old, new)


def knowledge_route_files() -> dict[str, str]:
    """The generated `/app/knowledge` routes, by their path in the project."""

    return {KNOWLEDGE_ROUTES: _KNOWLEDGE_PY}


def with_operator_knowledge(router: str) -> str:
    """The operator app's router (`scaffold_web_app.py`) serving the Knowledge area's
    routes to its operators."""

    path = "integrations/http/__init__.py"
    for old, new in (_OPERATOR_IMPORTS, _OPERATOR_IMPORTS_AFTER, _OPERATOR_ROUTES):
        router = _replace_once(router, old, new, path)
    return _replace_once(
        router,
        _ROUTES_LINE,
        _ROUTES_LINE + _KNOWLEDGE_LINE.replace("__WHO__", _OPERATORS_WHO),
        path,
    )


def with_member_knowledge(router: str) -> str:
    """The organizations app's router (`scaffold_organizations.py`) serving the Knowledge
    area's routes to the organizations' members."""

    path = "integrations/http/__init__.py"
    for old, new in (_MEMBER_IMPORTS, _MEMBER_IMPORTS_AFTER, _MEMBER_ROUTES):
        router = _replace_once(router, old, new, path)
    return _replace_once(
        router,
        _ROUTES_LINE_ORGANIZATIONS,
        _ROUTES_LINE_ORGANIZATIONS + _KNOWLEDGE_LINE.replace("__WHO__", _MEMBERS_WHO),
        path,
    )


def operator_knowledge_test() -> str:
    return _OPERATOR_TEST_PY


def member_knowledge_test() -> str:
    return _MEMBER_TEST_PY


def guidance(*, members: bool) -> str:
    """The README and AGENTS section about the Knowledge area."""

    return GUIDANCE.replace("__WHO_CHANGES__", _MEMBERS_CHANGE if members else _OPERATORS_CHANGE)
