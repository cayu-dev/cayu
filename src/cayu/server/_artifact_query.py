"""Complete public artifact queries over optional bounded native inventories."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import datetime
from heapq import heappush, heapreplace
from typing import Any

from fastapi import HTTPException

from cayu.artifacts._index import ArtifactIndexOrder, artifact_kind
from cayu.artifacts._scan import ArtifactScanFilter
from cayu.artifacts.base import (
    ArtifactListResult,
    ArtifactMetadata,
    ArtifactStore,
    ArtifactStoreUnavailableError,
)
from cayu.session_metadata import validate_title
from cayu.sessions.display import SessionTitleQuery, SessionTitleSnapshot
from cayu.sessions.store import SessionStore

ArtifactOrder = ArtifactIndexOrder
ARTIFACT_QUERY_SCAN_MAX = 100_000
ARTIFACT_QUERY_BATCH_SIZE = 100


def supports_artifact_search(stores: dict[str, ArtifactStore], sessions: SessionStore) -> bool:
    return (
        bool(stores)
        and sessions.supports_session_titles is True
        and all(store.supports_metadata_scan is True for store in stores.values())
    )


def artifact_search_available(stores: dict[str, ArtifactStore], sessions: SessionStore) -> bool:
    """The dashboard can search across every registered store.

    With an artifact index, search reads the index; until a store's index rows
    are filled it walks the store, which needs a metadata scan.
    """
    return supports_artifact_search(stores, sessions) or (
        bool(stores) and getattr(sessions, "supports_artifact_index", False) is True
    )


def artifact_search_values(
    filename: Any,
    content_type: Any,
    agent_name: Any,
    environment_name: Any,
    session_title: Any,
    artifact_id: Any,
) -> tuple[Any, ...]:
    """The values an artifact search looks in, in the order the dashboard shows them."""

    return (
        filename,
        artifact_kind(content_type),
        content_type,
        agent_name,
        environment_name,
        session_title,
        artifact_id,
    )


def matches_artifact_search(
    needle: str, values: Iterable[Any], redact: Callable[[Any], Any]
) -> bool:
    """A literal, case-insensitive match in a value as it is shown, after redaction.

    A match only inside a redacted secret does not count, so a search can never
    confirm a guess at one.
    """

    return any(type(value) is str and needle in value.lower() for value in map(redact, values))


async def read_titles(
    sessions: SessionStore, ids: tuple[str, ...], redact: Callable[[Any], Any]
) -> dict[str, str | None]:
    if not ids or sessions.supports_session_titles is not True:
        return dict.fromkeys(ids)
    query = SessionTitleQuery(session_ids=ids)
    try:
        result = await sessions.read_session_titles(query)
    except Exception:
        raise HTTPException(503, "Artifact session titles are temporarily unavailable.") from None
    try:
        if type(result) is not SessionTitleSnapshot:
            raise ValueError("Invalid title snapshot.")
        result = result.validate_for_query(query)
    except (TypeError, ValueError, AttributeError):
        raise HTTPException(500, "The session store returned inconsistent titles.") from None
    titles = {}
    for row in result.sessions:
        try:
            titles[row.session_id] = validate_title(redact(row.title))
        except ValueError:
            titles[row.session_id] = None
    return titles


class _Selected:
    """Worst retained row at the root; ties use the exact cross-store identity."""

    def __init__(self, key: tuple[Any, ...], row: dict[str, Any], descending: bool):
        self.key, self.row, self.descending = key, row, descending

    def __lt__(self, other: _Selected) -> bool:
        return self.key < other.key if self.descending else self.key > other.key


async def query_artifacts(
    stores: dict[str, ArtifactStore],
    sessions: SessionStore,
    *,
    serialize: Callable[[ArtifactMetadata, str], dict[str, Any]],
    redact: Callable[[Any], Any],
    filters: ArtifactScanFilter,
    q: str | None,
    order_by: ArtifactOrder,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    if not supports_artifact_search(stores, sessions):
        raise HTTPException(
            501, "Global artifact search and sorting are unavailable for these stores."
        )
    needle = (q or "").strip().lower()
    descending = order_by.endswith("_desc")
    selected: list[_Selected] = []
    total = 0
    scanned = 0
    title_cache: OrderedDict[str, str | None] = OrderedDict()
    # Identity validation is bounded by the scan ceiling, not total remote inventory.
    seen: set[tuple[str, str]] = set()

    async def scan_store(store_id: str, store: ArtifactStore) -> None:
        delivered = 0
        active = True

        async def consume(batch: tuple[ArtifactMetadata, ...]) -> None:
            nonlocal delivered, scanned, total
            if not active:
                raise HTTPException(500, "Artifact scan consumer is no longer active.")
            if type(batch) is not tuple or not 1 <= len(batch) <= ARTIFACT_QUERY_BATCH_SIZE:
                raise HTTPException(500, "Artifact store returned an invalid metadata batch.")
            scanned += len(batch)
            if scanned > ARTIFACT_QUERY_SCAN_MAX:
                raise HTTPException(
                    503, "Artifact query is too large; select a narrower artifact store."
                )
            detached = []
            for artifact in batch:
                if type(artifact) is not ArtifactMetadata:
                    raise HTTPException(500, "Artifact store returned invalid metadata.")
                # Do not serialize extension-controlled malformed objects into diagnostics.
                try:
                    copied = ArtifactMetadata.model_validate(artifact.model_dump(warnings=False))
                except (TypeError, ValueError):
                    raise HTTPException(500, "Artifact store returned invalid metadata.") from None
                identity = (store_id, copied.id)
                if identity in seen:
                    raise HTTPException(500, "Artifact store returned inconsistent inventory.")
                seen.add(identity)
                if filters.matches(copied):
                    detached.append(copied)
            missing = tuple(
                dict.fromkeys(
                    row.session_id
                    for row in detached
                    if row.session_id is not None and row.session_id not in title_cache
                )
            )
            titles = await read_titles(sessions, missing, redact)
            if not active:
                raise HTTPException(500, "Artifact scan consumer is no longer active.")
            title_cache.update(titles)
            for artifact in detached:
                row = serialize(artifact, store_id)
                row["session_title"] = (
                    title_cache.get(artifact.session_id) if artifact.session_id else None
                )
                text = artifact_search_values(
                    row["filename"],
                    row["content_type"],
                    row["agent_name"],
                    row["environment_name"],
                    row["session_title"],
                    row["id"],
                )
                if needle and not matches_artifact_search(needle, text, redact):
                    continue
                total += 1
                if order_by.startswith("filename_"):
                    primary = row["filename"].lower()
                elif order_by.startswith("size_bytes_"):
                    primary = row["size_bytes"]
                else:
                    primary = datetime.fromisoformat(row["created_at"]).timestamp()
                entry = _Selected((primary, store_id, row["id"]), row, descending)
                if len(selected) < offset + limit:
                    heappush(selected, entry)
                elif selected[0] < entry:
                    heapreplace(selected, entry)
            while len(title_cache) > 5000:
                title_cache.popitem(last=False)
            delivered += len(batch)

        try:
            count = await store.scan_metadata(
                consume,
                batch_size=ARTIFACT_QUERY_BATCH_SIZE,
                max_records=max(1, ARTIFACT_QUERY_SCAN_MAX - scanned),
            )
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(
                503, "Artifact inventory is unavailable or exceeds query bounds."
            ) from None
        finally:
            active = False
        if type(count) is not int or count != delivered:
            raise HTTPException(500, "Artifact store did not prove a complete inventory.")

    for store_id, store in stores.items():
        await scan_store(store_id, store)

    rows = [
        entry.row for entry in sorted(selected, key=lambda entry: entry.key, reverse=descending)
    ]
    has_more = offset + limit < total
    return {
        "artifacts": rows[offset : offset + limit],
        "total_count": total,
        "truncated": has_more,
        "limit": limit,
        "offset": offset,
        "next_offset": offset + limit if has_more and offset + limit <= 10_000 else None,
    }


async def list_store_artifacts(
    stores: dict[str, ArtifactStore],
    sessions: SessionStore,
    *,
    serialize: Callable[[ArtifactMetadata, str], dict[str, Any]],
    redact_title: Callable[[Any], Any],
    filters: ArtifactScanFilter,
    offset: int,
    limit: int,
    offset_max: int,
) -> dict[str, Any]:
    """Newest artifacts first, read from each store's own listing (the reference path)."""

    artifacts: list[dict[str, Any]] = []
    total_count: int | None = 0
    truncated = False
    per_store_limit = offset + limit
    for store_id, store in stores.items():
        try:
            result = await store.list(
                scope=filters.scope,
                session_id=filters.session_id,
                agent_name=filters.agent_name,
                environment_name=filters.environment_name,
                limit=per_store_limit,
            )
            if type(result) is not ArtifactListResult:
                raise TypeError("Artifact stores must return ArtifactListResult from list().")
            page = ArtifactListResult(
                artifacts=result.artifacts,
                total_count=result.total_count,
                truncated=result.truncated,
            )
        except (ArtifactStoreUnavailableError, OSError) as exc:
            raise HTTPException(
                status_code=503,
                detail="Artifact store is unavailable.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=500,
                detail="Artifact store returned invalid artifact data.",
            ) from exc
        artifacts.extend(serialize(artifact, store_id) for artifact in page.artifacts)
        if page.total_count is None:
            total_count = None
            truncated = True
        elif total_count is not None:
            total_count += page.total_count
        truncated = truncated or page.truncated

    artifacts.sort(
        key=lambda row: (str(row["created_at"]), str(row["artifact_store_id"]), str(row["id"])),
        reverse=True,
    )
    page_artifacts = artifacts[offset : offset + limit]
    titles = await read_titles(
        sessions,
        tuple(dict.fromkeys(row["session_id"] for row in page_artifacts if row["session_id"])),
        redact_title,
    )
    for row in page_artifacts:
        row["session_title"] = titles.get(row["session_id"])
    next_offset = None
    has_more = False
    if total_count is None:
        if len(artifacts) > offset + limit or truncated:
            has_more = True
            candidate_next_offset = offset + limit
            if candidate_next_offset <= offset_max:
                next_offset = candidate_next_offset
    elif offset + limit < total_count:
        has_more = True
        candidate_next_offset = offset + limit
        if candidate_next_offset <= offset_max:
            next_offset = candidate_next_offset
    truncated = has_more
    return {
        "artifacts": page_artifacts,
        "total_count": total_count,
        "truncated": truncated,
        "limit": limit,
        "offset": offset,
        "next_offset": next_offset,
    }
