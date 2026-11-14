"""The artifact index: artifact metadata kept in the session store.

Lists, searches and counts of artifacts read ``cayu_artifact_index`` instead of
walking an artifact store. The index holds one row per artifact a store holds,
keyed by the store's id and the artifact's id, with its metadata, the title of
the session that saved it, and a deletion time once it is deleted.

Writing it: :func:`bind_artifact_index` connects a store to an index writer (a
session store that ``supports_artifact_index``). From then on every committed
``put_bytes`` records its metadata, and every ``delete`` and claimed deletion
records a tombstone. The hook lives in :class:`~cayu.artifacts.base.ArtifactStore`
itself, so every backend, built in or not, is covered. ``CayuApp`` binds each
registered store to its session store.

An artifact store is external storage, so its commit and the index row cannot
share a transaction. A row that fails to land, a write made outside Cayu, or a
late completion after cancellation leaves drift. :func:`reindex_artifact_store`
compares the index with a store and reports or repairs it (``cayu artifact
reindex``). Reads use the index for a store only once it has been filled from
that store; until then they fall back to the store itself.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import functools
import hashlib
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, get_args

from cayu._validation import require_clean_nonblank

if TYPE_CHECKING:
    from cayu.artifacts.base import ArtifactMetadata, ArtifactScope, ArtifactStore

_LOGGER = logging.getLogger("cayu.artifacts.index")

ArtifactIndexOrder = Literal[
    "created_at_desc",
    "created_at_asc",
    "filename_asc",
    "filename_desc",
    "size_bytes_desc",
    "size_bytes_asc",
]
ARTIFACT_INDEX_ORDERS: tuple[str, ...] = get_args(ArtifactIndexOrder)
#: The most rows one index page returns, and the most sessions one count reads.
ARTIFACT_INDEX_PAGE_MAX = 1000
#: The most rows a write or a scan of the index handles at once.
ARTIFACT_INDEX_BATCH_MAX = 500
#: The longest search text a query takes, as the artifact routes allow.
ARTIFACT_INDEX_QUERY_MAX = 256
#: Separates the fields of a row's search text, so a match never spans two.
SEARCH_FIELD_SEPARATOR = "\x1f"


def artifact_kind(content_type: str) -> str:
    """The short kind the dashboard shows for a media type, and that search matches."""

    kind = content_type.split(";", 1)[0].strip().lower()
    if kind in {"image/gif", "image/jpeg", "image/png", "image/webp"}:
        return f"{kind[6:].upper()} image"
    if kind.startswith("image/"):
        return "Image"
    if kind == "application/json" or kind.endswith("+json"):
        return "JSON"
    if kind == "text/markdown":
        return "Markdown"
    if kind == "text/csv":
        return "CSV"
    if kind.startswith("text/"):
        return "Text"
    if kind == "application/pdf":
        return "PDF"
    if any(part in kind for part in ("zip", "tar", "gzip", "x-7z", "compressed")):
        return "Archive"
    return "File"


def format_index_time(value: datetime) -> str:
    """Fixed-width UTC text, so SQLite orders it as it orders times."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_index_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def normalize_search(q: str | None) -> str | None:
    """The literal, case-folded text a search matches, or None for no search."""

    if q is None:
        return None
    if type(q) is not str:
        raise TypeError("Artifact search text must be a string.")
    needle = "".join(char for char in q.strip().lower() if char >= " " and char != "\x7f")
    if len(needle) > ARTIFACT_INDEX_QUERY_MAX:
        raise ValueError(
            f"Artifact search text must be at most {ARTIFACT_INDEX_QUERY_MAX} characters."
        )
    return needle or None


@dataclass(frozen=True, slots=True)
class ArtifactIndexValues:
    """One artifact's index columns, computed the same way for every backend."""

    store_id: str
    artifact_id: str
    scope: str
    session_id: str | None
    agent_name: str | None
    environment_name: str | None
    created_at: datetime
    filename: str
    filename_key: str
    content_type: str
    size_bytes: int
    search_text: str
    metadata_json: str

    @classmethod
    def of(cls, store_id: str, artifact: ArtifactMetadata) -> ArtifactIndexValues:
        from cayu.artifacts.base import ArtifactMetadata

        if type(artifact) is not ArtifactMetadata:
            raise TypeError("The artifact index records ArtifactMetadata.")
        search_text = SEARCH_FIELD_SEPARATOR.join(
            value.lower()
            for value in (
                artifact.filename,
                artifact_kind(artifact.content_type),
                artifact.content_type,
                artifact.agent_name or "",
                artifact.environment_name or "",
                artifact.id,
            )
        )
        return cls(
            store_id=store_id,
            artifact_id=artifact.id,
            scope=artifact.scope.value,
            session_id=artifact.session_id,
            agent_name=artifact.agent_name,
            environment_name=artifact.environment_name,
            created_at=artifact.created_at,
            filename=artifact.filename,
            filename_key=artifact.filename.lower(),
            content_type=artifact.content_type,
            size_bytes=artifact.size_bytes,
            search_text=search_text,
            metadata_json=metadata_json(artifact),
        )


def metadata_json(artifact: ArtifactMetadata) -> str:
    return json.dumps(
        artifact.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def decode_metadata(value: str | Mapping[str, Any]) -> ArtifactMetadata:
    from cayu.artifacts.base import ArtifactMetadata

    if isinstance(value, str):
        return ArtifactMetadata.model_validate_json(value)
    return ArtifactMetadata.model_validate(value)


@dataclass(frozen=True, slots=True)
class ArtifactIndexCursor:
    """The position after the last row of a page, in its query's order.

    It names only that row; the index reads the row's sort value when the next
    page is asked for, so a cursor never carries a filename or other metadata
    that reads redact.
    """

    order_by: str
    store_id: str
    artifact_id: str

    def encode(self) -> str:
        raw = json.dumps(
            [self.order_by, self.store_id, self.artifact_id],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @classmethod
    def decode(cls, value: str, *, order_by: str) -> ArtifactIndexCursor:
        try:
            raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
            decoded = json.loads(raw)
        except (binascii.Error, ValueError, UnicodeDecodeError):
            raise ValueError("Artifact cursor is invalid.") from None
        if (
            type(decoded) is not list
            or len(decoded) != 3
            or decoded[0] != order_by
            or type(decoded[1]) is not str
            or type(decoded[2]) is not str
        ):
            raise ValueError("Artifact cursor is invalid for this order.")
        return cls(
            order_by,
            require_clean_nonblank(decoded[1], "artifact_store_id"),
            require_clean_nonblank(decoded[2], "artifact_id"),
        )

    @classmethod
    def after(cls, order_by: str, row: ArtifactIndexRow) -> ArtifactIndexCursor:
        return cls(order_by, row.store_id, row.artifact.id)


@dataclass(frozen=True, slots=True)
class ArtifactIndexQuery:
    """One page of live indexed artifacts in some stores, filtered, searched and ordered.

    ``after`` continues from a cursor (keyset paging); ``offset`` skips rows
    instead and is only for the bounded offsets the public routes accept.
    """

    store_ids: tuple[str, ...]
    scope: ArtifactScope | None = None
    session_id: str | None = None
    agent_name: str | None = None
    environment_name: str | None = None
    q: str | None = None
    order_by: str = "created_at_desc"
    after: ArtifactIndexCursor | None = None
    offset: int = 0
    limit: int = 100
    include_total: bool = True

    def __post_init__(self) -> None:
        from cayu.artifacts.base import ArtifactScope

        stores = tuple(
            dict.fromkeys(require_clean_nonblank(value, "store_id") for value in self.store_ids)
        )
        if not stores or len(stores) > ARTIFACT_INDEX_PAGE_MAX:
            raise ValueError("An artifact index query names 1..1000 stores.")
        object.__setattr__(self, "store_ids", stores)
        if self.scope is not None and type(self.scope) is not ArtifactScope:
            raise TypeError("Artifact scope must be an ArtifactScope.")
        for name in ("session_id", "agent_name", "environment_name"):
            value = getattr(self, name)
            if value is not None:
                require_clean_nonblank(value, name)
        object.__setattr__(self, "q", normalize_search(self.q))
        if self.order_by not in ARTIFACT_INDEX_ORDERS:
            raise ValueError("Unknown artifact order.")
        if self.after is not None and (
            type(self.after) is not ArtifactIndexCursor or self.after.order_by != self.order_by
        ):
            raise ValueError("Artifact cursor is invalid for this order.")
        if type(self.limit) is not int or not 1 <= self.limit <= ARTIFACT_INDEX_PAGE_MAX:
            raise ValueError("Artifact index page limit must be 1..1000.")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("Artifact index offset must be non-negative.")
        if self.after is not None and self.offset:
            raise ValueError("Artifact index pages take a cursor or an offset, not both.")

    @property
    def descending(self) -> bool:
        return self.order_by.endswith("_desc")

    @property
    def sort_column(self) -> str:
        if self.order_by.startswith("filename_"):
            return "filename_key"
        if self.order_by.startswith("size_bytes_"):
            return "size_bytes"
        return "created_at"


@dataclass(frozen=True, slots=True)
class ArtifactIndexRow:
    store_id: str
    artifact: ArtifactMetadata
    #: The current title of the session that saved it, when it has one.
    title: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactIndexPage:
    rows: tuple[ArtifactIndexRow, ...]
    #: Every live row the query matches, when the query asked for it.
    total_count: int | None
    next_cursor: ArtifactIndexCursor | None = None


@dataclass(frozen=True, slots=True)
class ArtifactIndexDrift:
    """How one store's index rows differ from what the store holds."""

    store_id: str
    store_count: int
    index_count: int
    #: Artifacts the store holds with no live row.
    missing: tuple[str, ...] = ()
    #: Live rows with no artifact in the store.
    orphaned: tuple[str, ...] = ()
    #: Live rows whose metadata differs from the store's.
    changed: tuple[str, ...] = ()
    repaired: bool = False
    backfilled_at: datetime | None = None

    @property
    def drift(self) -> bool:
        return bool(self.missing or self.orphaned or self.changed)

    def as_json(self, *, sample: int = 20) -> dict[str, Any]:
        return {
            "artifact_store_id": self.store_id,
            "store_count": self.store_count,
            "index_count": self.index_count,
            "missing_count": len(self.missing),
            "orphaned_count": len(self.orphaned),
            "changed_count": len(self.changed),
            "missing": list(self.missing[:sample]),
            "orphaned": list(self.orphaned[:sample]),
            "changed": list(self.changed[:sample]),
            "drift": self.drift,
            "repaired": self.repaired,
            "backfilled_at": None if self.backfilled_at is None else self.backfilled_at.isoformat(),
        }


class ArtifactIndexWriter(Protocol):
    """What the commit hook writes to: a session store that supports the index."""

    async def record_indexed_artifacts(
        self,
        store_id: str,
        artifacts: Sequence[ArtifactMetadata],
        *,
        observed_at: datetime,
    ) -> None: ...

    async def record_deleted_artifacts(
        self, store_id: str, artifact_ids: Sequence[str], *, deleted_at: datetime
    ) -> None: ...

    async def mark_artifact_index_store(
        self, store_id: str, *, backfilled_at: datetime | None
    ) -> None: ...


# ---------------------------------------------------------------------------
# The commit hook.

_WRITERS_ATTRIBUTE = "_cayu_artifact_index_writers"
_HOOKED_ATTRIBUTE = "__cayu_artifact_index_hook__"
_recording: ContextVar[frozenset[int]] = ContextVar(
    "cayu_artifact_index_recording", default=frozenset()
)


def bind_artifact_index(store: ArtifactStore, writer: ArtifactIndexWriter) -> None:
    """Record ``store``'s commits and deletions in ``writer`` from now on. Idempotent."""

    try:
        attributes = vars(store)
    except TypeError:
        # A store without an instance dictionary (``__slots__``) cannot be bound;
        # reads keep walking it, as for any store whose index is not filled.
        _LOGGER.warning("Artifact store %r cannot record commits in the artifact index.", store)
        return
    writers: list[ArtifactIndexWriter] = attributes.setdefault(_WRITERS_ATTRIBUTE, [])
    if not any(existing is writer for existing in writers):
        writers.append(writer)


def unbind_artifact_index(store: ArtifactStore, writer: ArtifactIndexWriter) -> None:
    writers: list[ArtifactIndexWriter] = getattr(store, "__dict__", {}).get(_WRITERS_ATTRIBUTE, [])
    writers[:] = [existing for existing in writers if existing is not writer]


def bound_artifact_index_writers(store: ArtifactStore) -> tuple[ArtifactIndexWriter, ...]:
    return tuple(getattr(store, "__dict__", {}).get(_WRITERS_ATTRIBUTE, ()))


async def _mark_unfilled(writer: ArtifactIndexWriter, store_id: str) -> None:
    """Have reads fall back to the store until it is filled again; the fill repairs."""

    try:
        await asyncio.shield(writer.mark_artifact_index_store(store_id, backfilled_at=None))
    except Exception:
        _LOGGER.warning(
            "Could not mark artifact store %r for reindexing; run "
            "`cayu artifact reindex` once the session store is reachable.",
            store_id,
            exc_info=True,
        )


async def _deliver(
    store: ArtifactStore,
    action: str,
    write: Callable[[ArtifactIndexWriter, str], Awaitable[None]],
) -> None:
    writers = bound_artifact_index_writers(store)
    if not writers:
        return
    store_id = store.id
    for writer in writers:
        # The artifact already committed. Finish the row even if the caller is
        # cancelled now, or a list would miss an artifact the store holds.
        try:
            await asyncio.shield(write(writer, store_id))
        except Exception:
            _LOGGER.warning(
                "Could not record an artifact %s in the artifact index for store %r; "
                "lists read the store until it is reindexed.",
                action,
                store_id,
                exc_info=True,
            )
            await _mark_unfilled(writer, store_id)


#: Index repairs still running after the store call that started them returned,
#: such as one waiting for a store's unsettled writes; kept until they finish.
_PENDING_REPAIRS: set[asyncio.Task[None]] = set()
#: How long a failed store call waits for its repair before it raises anyway.
REPAIR_WAIT_SECONDS = 5.0


def _keep(task: asyncio.Task[None]) -> None:
    _PENDING_REPAIRS.add(task)
    task.add_done_callback(_PENDING_REPAIRS.discard)


async def _finish_before_raising(repair: Awaitable[None]) -> asyncio.CancelledError | None:
    """Run ``repair`` to its end, through cancellation, within a bound.

    The store call's exception is raised once the repair finishes, so a read
    that follows it sees the store unfilled. A repair still running at the
    bound goes on in the background. Returns a cancellation that arrived
    meanwhile, for the caller to raise.
    """

    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(repair)
    deadline = loop.time() + REPAIR_WAIT_SECONDS
    absorbed: asyncio.CancelledError | None = None
    while not task.done():
        remaining = deadline - loop.time()
        if remaining <= 0:
            _keep(task)
            break
        try:
            await asyncio.wait({task}, timeout=remaining)
        except asyncio.CancelledError as cancellation:
            absorbed = cancellation
    return absorbed


def _may_have_changed(store: ArtifactStore, kind: str, error: BaseException) -> bool:
    """Whether a store call that raised ``error`` may still have changed the store.

    Built-in stores attach settlement evidence to the exception: a write can
    commit and still raise the caller's cancellation, or still be running when
    it does. Without evidence, a cancellation or an unexpected failure may have
    changed the store; a rejected argument has not.
    """

    from cayu.artifacts.settlement import (
        ArtifactWriteSettlementStatus,
        artifact_store_identity_sha256,
        artifact_write_settlements,
    )

    try:
        identity = artifact_store_identity_sha256(store.id)
        records = [
            record
            for record in artifact_write_settlements(error)
            if record.store_identity_sha256 == identity
        ]
    except Exception:
        records = []
    if records:
        return any(record.status is not ArtifactWriteSettlementStatus.ABSENT for record in records)
    if isinstance(error, asyncio.CancelledError):
        return True
    # A scoped access denial is raised before the store operation runs.
    if isinstance(error, (TypeError, ValueError, PermissionError)):
        return False
    return not (kind == "delete" and isinstance(error, FileNotFoundError))


async def _invalidate(store: ArtifactStore) -> None:
    """Mark the store unfilled now, and again once the writes still running settle.

    A fill that runs before a late write lands could otherwise mark the store
    filled without it; the second mark has the next read fill it again.
    """

    writers = bound_artifact_index_writers(store)
    store_id = store.id
    for writer in writers:
        await _mark_unfilled(writer, store_id)
    # Built-in stores keep their dispatched writes, late ones included, here.
    pending_writes = getattr(getattr(store, "_write_registry", None), "pending", None)
    pending = pending_writes() if callable(pending_writes) else ()
    if not pending:
        return

    async def after_settlement() -> None:
        await asyncio.wait(pending)
        for writer in writers:
            await _mark_unfilled(writer, store_id)

    _keep(
        asyncio.get_running_loop().create_task(
            after_settlement(), name=f"cayu-artifact-index-settlement:{store_id}"
        )
    )


def _hook(method: Callable[..., Any], kind: str) -> Callable[..., Any]:
    if getattr(method, _HOOKED_ATTRIBUTE, False):
        return method

    @functools.wraps(method)
    async def hooked(self: ArtifactStore, *args: Any, **kwargs: Any) -> Any:
        active = _recording.get()
        if id(self) in active:
            # A subclass calling super(): the outermost call records.
            return await method(self, *args, **kwargs)
        token = _recording.set(active | {id(self)})
        try:
            result = await method(self, *args, **kwargs)
        except BaseException as error:
            _recording.reset(token)
            # The store may hold a change the index never heard of. The caller's
            # exception, cancellation included, still propagates unchanged.
            if bound_artifact_index_writers(self) and _may_have_changed(self, kind, error):
                cancelled = await _finish_before_raising(_invalidate(self))
                if cancelled is not None and not isinstance(error, asyncio.CancelledError):
                    raise cancelled from error
            raise
        _recording.reset(token)
        # The time the operation finished orders it against other writes of the
        # same artifact. The store held this state from then on, so it outranks
        # a reindex that began earlier and may have seen the store before it.
        finished_at = datetime.now(UTC)
        if kind == "put":
            from cayu.artifacts.base import ArtifactMetadata

            if type(result) is ArtifactMetadata:
                artifact = result

                async def write(writer: ArtifactIndexWriter, store_id: str) -> None:
                    await writer.record_indexed_artifacts(
                        store_id, (artifact,), observed_at=finished_at
                    )

                await _deliver(self, "commit", write)
        else:
            artifact_id = args[-1] if args else kwargs.get("artifact_id")
            if type(artifact_id) is str:

                async def write(writer: ArtifactIndexWriter, store_id: str) -> None:
                    await writer.record_deleted_artifacts(
                        store_id, (artifact_id,), deleted_at=finished_at
                    )

                await _deliver(self, "deletion", write)
        return result

    setattr(hooked, _HOOKED_ATTRIBUTE, True)
    return hooked


#: The store methods the hook wraps, and what each records.
HOOKED_METHODS: Mapping[str, str] = {
    "put_bytes": "put",
    "delete": "delete",
    "delete_session_closure_artifact": "delete",
}


def install_artifact_index_hooks(cls: type) -> None:
    """Wrap the commit and deletion methods ``cls`` defines itself."""

    for name, kind in HOOKED_METHODS.items():
        method = cls.__dict__.get(name)
        if method is None or getattr(method, "__isabstractmethod__", False):
            continue
        if callable(method):
            setattr(cls, name, _hook(method, kind))


# ---------------------------------------------------------------------------
# Backfill and drift.


def _digest(value: str) -> bytes:
    return hashlib.blake2b(value.encode(), digest_size=16).digest()


async def _store_inventory(
    store: ArtifactStore,
    consume: Callable[[tuple[ArtifactMetadata, ...]], Awaitable[None]],
    restart: Callable[[], None],
) -> None:
    """Pass the store's whole inventory to ``consume`` in batches.

    A store over the metadata scan's bound is read again through ``list``;
    ``restart`` is called first, so ``consume`` forgets what the scan delivered.
    """

    from cayu.artifacts._scan import ARTIFACT_SCAN_RECORD_MAX
    from cayu.artifacts.base import ArtifactStoreUnavailableError

    delivered = 0

    async def counted(batch: tuple[ArtifactMetadata, ...]) -> None:
        nonlocal delivered
        await consume(batch)
        delivered += len(batch)

    if store.supports_metadata_scan is True:
        try:
            count = await store.scan_metadata(
                counted, batch_size=ARTIFACT_INDEX_BATCH_MAX, max_records=ARTIFACT_SCAN_RECORD_MAX
            )
        except ArtifactStoreUnavailableError:
            if delivered < ARTIFACT_SCAN_RECORD_MAX:
                raise
            # Over the scan bound: read the whole inventory instead.
            restart()
        else:
            if type(count) is not int or count != delivered:
                raise ArtifactStoreUnavailableError(
                    "Artifact store did not prove a complete inventory."
                )
            return
    result = await store.list(limit=None)
    if result.truncated:
        raise ArtifactStoreUnavailableError("Artifact store cannot list its whole inventory.")
    artifacts = result.artifacts
    for start in range(0, len(artifacts), ARTIFACT_INDEX_BATCH_MAX):
        await consume(artifacts[start : start + ARTIFACT_INDEX_BATCH_MAX])


async def reindex_artifact_store(
    index: Any,
    store_id: str,
    store: ArtifactStore,
    *,
    check: bool = False,
) -> ArtifactIndexDrift:
    """Compare ``store``'s inventory with its index rows; repair them unless ``check``.

    Reads the store's live rows, then walks the store once. Repair writes a row
    for each artifact the store holds (a deletion that finished after the walk
    began wins), tombstones each row with no artifact, and marks the store filled, so
    reads use the index for it, when the store is bound to ``index`` (otherwise its
    later commits would not reach the rows) and no commit marked it as needing
    filling while the walk ran (that commit may be missing from the walk). The report describes what differed
    before repair.
    """

    from cayu.artifacts.base import ArtifactStoreUnavailableError

    if index.supports_artifact_index is not True:
        raise NotImplementedError("This session store has no artifact index.")
    started_at = datetime.now(UTC)
    if not check:
        # A commit that lands during the walk may be missing from it. Its hook marks
        # the store as needing filling, which forgets this claim, so the fill below
        # then leaves the store unfilled for the next read to fill again. A fill that
        # claims the store after this one replaces the claim, and only it completes.
        await index.claim_artifact_index_fill(store_id, claimed_at=started_at)
    indexed: dict[str, bytes] = {}
    after: str | None = None
    while True:
        rows = await index.scan_artifact_index(
            store_id, after=after, limit=ARTIFACT_INDEX_BATCH_MAX
        )
        if not rows:
            break
        for artifact_id, stored_json in rows:
            indexed[artifact_id] = _digest(stored_json)
        after = rows[-1][0]
    seen: set[str] = set()
    missing: list[str] = []
    changed: list[str] = []

    def restart() -> None:
        # Rows the first pass already wrote are rewritten with the same values.
        seen.clear()
        missing.clear()
        changed.clear()

    async def consume(batch: tuple[ArtifactMetadata, ...]) -> None:
        for artifact in batch:
            if artifact.id in seen:
                raise ArtifactStoreUnavailableError("Artifact store listed an artifact twice.")
            seen.add(artifact.id)
            expected = indexed.get(artifact.id)
            if expected is None:
                missing.append(artifact.id)
            elif expected != _digest(metadata_json(artifact)):
                changed.append(artifact.id)
        if not check:
            await index.record_indexed_artifacts(store_id, batch, observed_at=started_at)

    await _store_inventory(store, consume, restart)
    orphaned = [artifact_id for artifact_id in indexed if artifact_id not in seen]
    if not check:
        for start in range(0, len(orphaned), ARTIFACT_INDEX_BATCH_MAX):
            await index.record_deleted_artifacts(
                store_id, orphaned[start : start + ARTIFACT_INDEX_BATCH_MAX], deleted_at=started_at
            )
        # Reads serve a store from the index only while its commits land there.
        bound = any(writer is index for writer in bound_artifact_index_writers(store))
        backfilled_at: datetime | None = None
        if bound:
            filled_at = datetime.now(UTC)
            if await index.complete_artifact_index_fill(
                store_id, claimed_at=started_at, backfilled_at=filled_at
            ):
                backfilled_at = filled_at
        else:
            await index.mark_artifact_index_store(store_id, backfilled_at=None)
    else:
        backfilled_at = (await index.read_artifact_index_stores((store_id,))).get(store_id)
    return ArtifactIndexDrift(
        store_id=store_id,
        store_count=len(seen),
        index_count=len(indexed),
        missing=tuple(sorted(missing)),
        orphaned=tuple(sorted(orphaned)),
        changed=tuple(sorted(changed)),
        repaired=not check,
        backfilled_at=backfilled_at,
    )
