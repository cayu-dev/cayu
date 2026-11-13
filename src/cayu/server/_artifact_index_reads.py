"""Artifact lists and counts served from the session store's artifact index.

A store's index rows serve reads once the index has been filled from that store.
The first read that meets a store not yet filled starts filling it in the
background, one walk per store, and is answered from the store itself meanwhile.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import aclosing
from dataclasses import replace
from typing import Any

from fastapi import APIRouter, HTTPException

from cayu.artifacts._index import (
    ARTIFACT_INDEX_PAGE_MAX,
    ArtifactIndexCursor,
    ArtifactIndexPage,
    ArtifactIndexQuery,
    ArtifactIndexRow,
    reindex_artifact_store,
)
from cayu.artifacts._scan import ArtifactScanFilter
from cayu.artifacts.base import ArtifactMetadata, ArtifactStore, ArtifactStoreUnavailableError
from cayu.server._artifact_query import (
    ARTIFACT_QUERY_SCAN_MAX,
    artifact_search_values,
    matches_artifact_search,
)
from cayu.server.contracts import ArtifactReindexBody, ArtifactReindexResponse
from cayu.session_metadata import validate_title
from cayu.sessions.store import SessionStore

_LOGGER = logging.getLogger("cayu.server.artifact_index")
#: How long after a failed fill a read may start another for the same store.
BACKFILL_RETRY_SECONDS = 60.0


class ArtifactIndexReads:
    def __init__(self, sessions: SessionStore) -> None:
        self.sessions = sessions
        self._fills: dict[str, asyncio.Task[Any]] = {}
        self._failed_at: dict[str, float] = {}

    @property
    def available(self) -> bool:
        return getattr(self.sessions, "supports_artifact_index", False) is True

    async def filled(self, stores: dict[str, ArtifactStore]) -> set[str]:
        """The stores whose index rows can serve reads now; starts filling the others."""

        if not self.available or not stores:
            return set()
        try:
            states = await self.sessions.read_artifact_index_stores(tuple(stores))
        except Exception:
            _LOGGER.warning("Could not read the artifact index state.", exc_info=True)
            return set()
        ready = {store_id for store_id in stores if states.get(store_id) is not None}
        for store_id, store in stores.items():
            if store_id not in ready:
                self._start_fill(store_id, store)
        return ready

    def _start_fill(self, store_id: str, store: ArtifactStore) -> None:
        running = self._fills.get(store_id)
        if running is not None and not running.done():
            return
        failed_at = self._failed_at.get(store_id)
        if failed_at is not None and time.monotonic() - failed_at < BACKFILL_RETRY_SECONDS:
            return
        task = asyncio.get_running_loop().create_task(
            reindex_artifact_store(self.sessions, store_id, store),
            name=f"cayu-artifact-index-fill:{store_id}",
        )
        self._fills[store_id] = task

        def settled(done: asyncio.Task[Any]) -> None:
            if done.cancelled():
                return
            failure = done.exception()
            if failure is None:
                self._failed_at.pop(store_id, None)
                return
            self._failed_at[store_id] = time.monotonic()
            _LOGGER.warning(
                "Could not fill the artifact index from store %r; lists read the store.",
                store_id,
                exc_info=failure,
            )

        task.add_done_callback(settled)

    def register_reindex_route(
        self,
        router: APIRouter,
        *,
        stores: Callable[[], dict[str, ArtifactStore]],
        dependencies: list[Any],
        responses: dict[int | str, dict[str, Any]],
    ) -> None:
        """``POST /artifacts/index/reindex``, which `cayu artifact reindex` calls."""

        async def reindex_artifacts(body: ArtifactReindexBody) -> dict[str, Any]:
            if not self.available:
                raise HTTPException(501, "This session store has no artifact index.")
            selected = stores()
            if body.artifact_store_id is not None:
                store = selected.get(body.artifact_store_id)
                if store is None:
                    raise HTTPException(status_code=404, detail="Artifact store not found")
                selected = {body.artifact_store_id: store}
            reports = []
            for store_id, store in selected.items():
                try:
                    drift = await reindex_artifact_store(
                        self.sessions, store_id, store, check=body.check
                    )
                except (ArtifactStoreUnavailableError, OSError) as exc:
                    raise HTTPException(503, "Artifact store is unavailable.") from exc
                reports.append(drift.as_json())
            return {"stores": reports}

        router.add_api_route(
            "/artifacts/index/reindex",
            reindex_artifacts,
            methods=["POST"],
            response_model=ArtifactReindexResponse,
            responses=responses,
            dependencies=dependencies,
            description=(
                "Compare the artifact index with each artifact store, or one, by walking "
                "the store once. With check, only report drift: an artifact with no index "
                "row, a row with no artifact, or differing metadata. Without it, also "
                "repair the rows and mark the store filled, so lists read the index."
            ),
        )

    async def close(self) -> None:
        """Stop fills still running when the server stops; the next start resumes them."""

        running = [task for task in self._fills.values() if not task.done()]
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        self._fills.clear()


def _title(redact: Callable[[Any], Any], title: str | None) -> str | None:
    if title is None:
        return None
    try:
        return validate_title(redact(title))
    except ValueError:
        return None


def _shown_match(needle: str, row: ArtifactIndexRow, redact: Callable[[Any], Any]) -> bool:
    artifact = row.artifact
    title = _title(redact, row.title) if artifact.session_id is not None else None
    values = artifact_search_values(
        artifact.filename,
        artifact.content_type,
        artifact.agent_name,
        artifact.environment_name,
        title,
        artifact.id,
    )
    return matches_artifact_search(needle, values, redact)


async def _index_page(sessions: SessionStore, query: ArtifactIndexQuery) -> ArtifactIndexPage:
    try:
        page = await sessions.query_artifact_index(query)
    except Exception:
        raise HTTPException(503, "The artifact index is temporarily unavailable.") from None
    if type(page) is not ArtifactIndexPage:
        raise HTTPException(500, "The session store returned an invalid artifact page.")
    return page


async def _shown_matches(
    sessions: SessionStore,
    query: ArtifactIndexQuery,
    needle: str,
    redact: Callable[[Any], Any],
    after: ArtifactIndexCursor | None,
) -> AsyncGenerator[ArtifactIndexRow, None]:
    """The rows the index matched whose shown values still match, in order."""

    read = 0
    while True:
        page = await _index_page(
            sessions,
            replace(
                query, after=after, offset=0, limit=ARTIFACT_INDEX_PAGE_MAX, include_total=False
            ),
        )
        for row in page.rows:
            if _shown_match(needle, row, redact):
                yield row
        read += len(page.rows)
        if page.next_cursor is None:
            return
        if read >= ARTIFACT_QUERY_SCAN_MAX:
            raise HTTPException(503, "Artifact search matches too many artifacts; narrow it.")
        after = page.next_cursor


async def _redacted_search_page(
    sessions: SessionStore,
    query: ArtifactIndexQuery,
    needle: str,
    redact: Callable[[Any], Any],
) -> ArtifactIndexPage:
    """A search page, and its total, over only the rows that match once redacted.

    The index matches stored text, which may hold a configured secret. Counting
    or paging by those matches would let a reader test guesses at the secret,
    so every match is checked as shown, as the store walk does.
    """

    wanted = query.offset + query.limit
    collected: list[ArtifactIndexRow] = []
    more = False
    total = 0
    async with aclosing(_shown_matches(sessions, query, needle, redact, query.after)) as matches:
        async for row in matches:
            if len(collected) == wanted:
                more = True
                break
            collected.append(row)
        rows = collected[query.offset :]
        if query.after is None:
            # Counting goes on from where the page ended.
            total = len(collected) + int(more)
            if more:
                async for _ in matches:
                    total += 1
    if query.after is not None:
        # A later page still reports the total of the whole search.
        async with aclosing(_shown_matches(sessions, query, needle, redact, None)) as everything:
            async for _ in everything:
                total += 1
    next_cursor = ArtifactIndexCursor.after(query.order_by, rows[-1]) if more else None
    return ArtifactIndexPage(rows=tuple(rows), total_count=total, next_cursor=next_cursor)


async def query_indexed_artifacts(
    sessions: SessionStore,
    store_ids: Sequence[str],
    *,
    serialize: Callable[[ArtifactMetadata, str], dict[str, Any]],
    redact: Callable[[Any], Any],
    redacts_secrets: bool,
    filters: ArtifactScanFilter,
    q: str | None,
    order_by: str,
    offset: int,
    cursor: str | None,
    limit: int,
    offset_max: int,
) -> dict[str, Any]:
    """A page of artifacts from the index, in the response shape of the store walk.

    ``redacts_secrets`` says whether the app redacts any configured secret; only
    then can a search's stored match differ from what a reader is shown.
    """

    try:
        after = None if cursor is None else ArtifactIndexCursor.decode(cursor, order_by=order_by)
        query = ArtifactIndexQuery(
            store_ids=tuple(store_ids),
            scope=filters.scope,
            session_id=filters.session_id,
            agent_name=filters.agent_name,
            environment_name=filters.environment_name,
            q=q,
            order_by=order_by,
            after=after,
            offset=offset,
            limit=limit,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    if query.q is not None and redacts_secrets:
        page = await _redacted_search_page(sessions, query, query.q, redact)
    else:
        page = await _index_page(sessions, query)
    rows = []
    for indexed in page.rows:
        row = serialize(indexed.artifact, indexed.store_id)
        row["session_title"] = (
            _title(redact, indexed.title) if indexed.artifact.session_id is not None else None
        )
        rows.append(row)
    has_more = page.next_cursor is not None
    next_offset = None
    if has_more and after is None and offset + limit <= offset_max:
        next_offset = offset + limit
    return {
        "artifacts": rows,
        "total_count": page.total_count,
        "truncated": has_more,
        "limit": limit,
        "offset": query.offset,
        "next_offset": next_offset,
        "next_cursor": None if page.next_cursor is None else page.next_cursor.encode(),
    }
