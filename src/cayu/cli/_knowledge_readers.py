"""One read seam for `cayu knowledge`: a local knowledge store or a Cayu server's API.

Both readers return entries in the shape of the Cayu server's knowledge API (``GET
/api/knowledge``, ``/api/knowledge/pending`` and their ``/{entry_id}`` details), so the
command renders one shape whichever it read. ``StoreKnowledgeReader`` takes any
``KnowledgeStore``; ``open_store_knowledge_reader`` opens a SQLite or Postgres one
read-only. ``ApiKnowledgeReader`` takes any ``CayuApi``, such as a Cayu Cloud Agent's.

Both readers apply the namespace, kind and label filters to the rows they return as
well, so a server whose Cayu predates those filters still answers correctly (with
sparser pages).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast
from urllib.parse import quote

from cayu.cli._cloud_agent_api import CayuApi, CayuApiError, CayuApiNotFound
from cayu.cli.store_targets import SessionStoreBackend, SessionStoreTarget
from cayu.knowledge.base import KnowledgeStore
from cayu.knowledge.records import KnowledgeChunk, KnowledgeEntry, KnowledgeStatus
from cayu.knowledge.scopes import KnowledgeAccessScope
from cayu.knowledge.search import (
    KnowledgeListQuery,
    decode_knowledge_list_cursor,
    encode_knowledge_list_cursor,
)

# The server's kept-knowledge list takes at most 100 entries per page.
MAX_KNOWLEDGE_PAGE = 100
# What the server's detail routes return at most.
_DETAIL_MAX_CHUNKS = 50
_DETAIL_MAX_BYTES = 128_000
# Chunks are read whole (up to this much) and redacted before `_DETAIL_MAX_BYTES` applies.
_CHUNK_READ_BYTES = 1_048_576
_LIST_MAX_BYTES = 1_048_576
_NO_KNOWLEDGE_STORE = "Knowledge store is not configured."
# Statuses Cayu's API lists: kept (active) entries and pending proposals.
API_STATUSES = (KnowledgeStatus.ACTIVE, KnowledgeStatus.PENDING)


@dataclass(frozen=True)
class KnowledgeListRequest:
    status: KnowledgeStatus = KnowledgeStatus.ACTIVE
    namespace: str | None = None
    kinds: tuple[str, ...] = ()
    labels: Mapping[str, str] = field(default_factory=dict)
    limit: int = 50
    cursor: str | None = None


@dataclass(frozen=True)
class KnowledgePage:
    entries: list[dict[str, Any]]
    next_cursor: str | None
    # More entries exist that no cursor reaches (Cayu's pending list has no cursor).
    truncated: bool = False


class KnowledgeReader(Protocol):
    source: dict[str, Any]

    async def list_entries(self, request: KnowledgeListRequest) -> KnowledgePage: ...

    async def load_entry(self, entry_id: str) -> dict[str, Any] | None:
        """The entry with its text and chunks, whatever its status, or ``None``."""
        ...

    async def close(self) -> None: ...


class StoreKnowledgeReader:
    """Reads a ``KnowledgeStore`` the caller opened; ``close`` closes it."""

    def __init__(
        self,
        store: KnowledgeStore,
        *,
        access_scope: KnowledgeAccessScope,
        source: dict[str, Any],
    ) -> None:
        self._store = store
        self._scope = access_scope
        self.source = source

    async def list_entries(self, request: KnowledgeListRequest) -> KnowledgePage:
        if request.cursor is not None:
            try:
                decode_knowledge_list_cursor(request.cursor)
            except ValueError:
                raise ValueError("--cursor isn't one this listing returned.") from None
        listed = await self._store.list_entries(
            KnowledgeListQuery(
                namespace=request.namespace,
                labels=dict(request.labels),
                kinds=list(request.kinds) or None,
                statuses=[request.status],
                cursor=request.cursor,
                limit=request.limit,
                max_bytes=_LIST_MAX_BYTES,
            ),
            access_scope=self._scope,
        )
        more = listed.truncated and (
            listed.total_entries_known is None or listed.total_entries_known > len(listed.entries)
        )
        entries = [
            {
                **entry_projection(item.entry),
                "chunk_count": item.chunk_count,
                "text_preview": item.text_preview or "",
            }
            for item in listed.entries
        ]
        return KnowledgePage(
            entries=entries,
            next_cursor=(
                encode_knowledge_list_cursor(listed.entries[-1].entry)
                if more and listed.entries
                else None
            ),
            truncated=more and not listed.entries,
        )

    async def load_entry(self, entry_id: str) -> dict[str, Any] | None:
        entry = await self._store.get_entry(entry_id, access_scope=self._scope)
        if entry is None:
            return None
        chunks = await self._store.read_chunks(
            entry.id,
            revision=entry.revision,
            access_scope=self._scope,
            max_chunks=_DETAIL_MAX_CHUNKS,
            max_bytes=_CHUNK_READ_BYTES,
        )
        return {
            **entry_projection(entry),
            "text": entry.text,
            "metadata": entry.metadata,
            "expires_at": None if entry.expires_at is None else entry.expires_at.isoformat(),
            "chunks": _redacted_chunks(chunks),
            "chunk_limit": _DETAIL_MAX_CHUNKS,
            "chunk_max_bytes": _DETAIL_MAX_BYTES,
        }

    async def close(self) -> None:
        close = getattr(self._store, "close", None)
        if close is not None:
            await close()


class ApiKnowledgeReader:
    """Reads a Cayu server's knowledge API; ``close`` closes the API client."""

    def __init__(self, api: CayuApi, *, source: dict[str, Any]) -> None:
        self._api = api
        self.source = source

    async def list_entries(self, request: KnowledgeListRequest) -> KnowledgePage:
        if request.status not in API_STATUSES:
            raise CayuApiError(
                "unsupported",
                f"{self._api.description}'s Cayu API lists only active and pending knowledge.",
            )
        query: dict[str, str | list[str]] = {"limit": str(request.limit)}
        if request.namespace is not None:
            query["namespace"] = request.namespace
        if request.kinds:
            query["kind"] = list(request.kinds)
        if request.labels:
            query["label"] = [f"{key}={value}" for key, value in request.labels.items()]
        pending = request.status is KnowledgeStatus.PENDING
        if pending:
            if request.cursor is not None:
                raise ValueError("Cayu's pending knowledge list has no next page to continue.")
            body = self._get("/knowledge/pending", query, "pending knowledge")
        else:
            if request.cursor is not None:
                query["cursor"] = request.cursor
            body = self._get("/knowledge", query, "knowledge")
        rows = body.get("entries") if isinstance(body, dict) else None
        if not isinstance(rows, list) or not all(_is_entry(row) for row in rows):
            raise CayuApiError(
                "not_cayu", f"{self._api.description} returned an invalid knowledge list."
            )
        cursor = None if pending else body.get("next_cursor")
        return KnowledgePage(
            entries=[row for row in rows if _matches(row, request)],
            next_cursor=cursor if isinstance(cursor, str) and cursor else None,
            truncated=pending and body.get("truncated") is True,
        )

    async def load_entry(self, entry_id: str) -> dict[str, Any] | None:
        quoted = quote(entry_id, safe="")
        kept_unserved = False
        try:
            return self._entry(
                self._get(f"/knowledge/kept/{quoted}", None, "kept knowledge entries")
            )
        except CayuApiNotFound:
            pass
        except CayuApiError as exc:
            # A Cayu older than `GET /knowledge/kept/{entry_id}` still shows pending entries.
            if exc.category not in {"unsupported", "not_served"}:
                raise
            kept_unserved = True
        try:
            return self._entry(
                self._get(f"/knowledge/pending/{quoted}", None, "pending knowledge entries")
            )
        except CayuApiError as exc:
            # The pending route answers 403 for an entry outside the review scope and 409
            # for one that isn't pending; the kept route already said it isn't kept and
            # visible, so either means there's no entry to show.
            if not isinstance(exc, CayuApiNotFound) and (
                kept_unserved or exc.status_code not in {403, 409}
            ):
                raise
            if kept_unserved:
                raise CayuApiError(
                    "unsupported",
                    f"{self._api.description} runs a Cayu that shows only pending knowledge "
                    "entries; deploy it with a newer Cayu to show kept ones.",
                ) from None
            return None

    async def close(self) -> None:
        self._api.close()

    def _get(self, path: str, query: dict[str, str | list[str]] | None, resource: str) -> Any:
        try:
            return self._api.get(path, query=query, resource=resource)
        except CayuApiNotFound as exc:
            if str(exc) == _NO_KNOWLEDGE_STORE:
                raise CayuApiError(
                    "unsupported", f"{self._api.description} has no knowledge store."
                ) from None
            raise

    def _entry(self, body: object) -> dict[str, Any]:
        if not _is_entry(body):
            raise CayuApiError(
                "not_cayu", f"{self._api.description} returned an invalid knowledge entry."
            )
        return cast("dict[str, Any]", body)


def open_store_knowledge_reader(target: SessionStoreTarget) -> StoreKnowledgeReader:
    """Open the knowledge store in ``target``'s database read-only, seeing every entry."""

    from cayu.storage import migrations as schema

    scope = KnowledgeAccessScope.privileged()
    if target.backend is SessionStoreBackend.SQLITE:
        path = target.sqlite_path
        if path is None:
            raise AssertionError("Resolved SQLite target has no path.")
        if not path.is_file():
            raise ValueError(f"No SQLite database exists at {path}.")
        from cayu.storage import SQLiteKnowledgeStore

        store: KnowledgeStore = SQLiteKnowledgeStore(
            path, schema_mode=schema.SchemaMode.VALIDATE, access_scope=scope, read_only=True
        )
        return StoreKnowledgeReader(
            store, access_scope=scope, source={"kind": "sqlite", "path": str(path)}
        )
    if target.postgres_dsn is None:
        raise AssertionError("Resolved Postgres target has no DSN.")
    from cayu.storage import PostgresKnowledgeStore

    store = PostgresKnowledgeStore(
        target.postgres_dsn,
        schema_mode=schema.SchemaMode.VALIDATE,
        read_only=True,
        access_scope=scope,
    )
    return StoreKnowledgeReader(store, access_scope=scope, source={"kind": "postgres"})


def entry_projection(entry: KnowledgeEntry) -> dict[str, Any]:
    """A local entry's fields in the server's public knowledge shape."""

    return {
        "entry_id": entry.id,
        "revision": entry.revision,
        "namespace": entry.namespace,
        "kind": entry.kind,
        "visibility": entry.visibility.value,
        "status": entry.status.value,
        "title": entry.title,
        "labels": dict(entry.labels),
        "aspects": list(entry.aspects),
        "impact_targets": list(entry.impact_targets),
        "source_type": entry.source_type,
        "source_uri": entry.source_uri,
        "source_id": entry.source_id,
        "created_by_type": entry.created_by_type.value,
        "created_by": entry.created_by,
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
        "importance": None if entry.importance is None else str(entry.importance),
        "importance_source": entry.importance_source,
        "confidence": None if entry.confidence is None else str(entry.confidence),
    }


def _redacted_chunks(chunks: list[KnowledgeChunk]) -> list[dict[str, Any]]:
    """Whole chunks redacted first, then cut to `_DETAIL_MAX_BYTES`, as the server's detail does.

    A redaction pattern (such as a private key's BEGIN and END markers) can only match text
    it sees whole, so a chunk the store may have cut at the read bound is left out.
    """

    from cayu.cli.session import _redact_sensitive
    from cayu.knowledge._retrieval_results import _truncate_text_to_bytes

    # Stores cut UTF-8 at a character boundary, up to three bytes short of the bound.
    if sum(len(chunk.text.encode("utf-8")) for chunk in chunks) > _CHUNK_READ_BYTES - 4:
        chunks = chunks[:-1]
    rows: list[dict[str, Any]] = []
    remaining = _DETAIL_MAX_BYTES
    for chunk in chunks:
        if remaining <= 0:
            break
        row = _chunk_projection(chunk)
        text = _redact_sensitive(chunk.text)
        if text != chunk.text:
            row["content_hash"] = None
        size = len(text.encode("utf-8"))
        if size > remaining:
            text = _truncate_text_to_bytes(text, remaining)
            row["content_hash"] = None
            size = remaining
        remaining -= size
        row["text"] = text
        rows.append(row)
    return rows


def _chunk_projection(chunk: KnowledgeChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "entry_id": chunk.entry_id,
        "entry_revision": chunk.entry_revision,
        "chunk_index": chunk.chunk_index,
        "text": chunk.text,
        "content_hash": chunk.content_hash,
        "source_uri": chunk.source_uri,
        "metadata": dict(chunk.metadata),
    }


def _matches(row: dict[str, Any], request: KnowledgeListRequest) -> bool:
    if request.namespace is not None and row.get("namespace") != request.namespace:
        return False
    if request.kinds and row.get("kind") not in request.kinds:
        return False
    labels = row.get("labels")
    labels = labels if isinstance(labels, dict) else {}
    return all(labels.get(key) == value for key, value in request.labels.items())


def _is_entry(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    row = cast("dict[str, Any]", value)
    return all(isinstance(row.get(key), str) for key in ("entry_id", "namespace", "kind"))
