from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from cayu._validation import copy_label_map, require_clean_nonblank
from cayu.storage.memory import (
    DEFAULT_KNOWLEDGE_LIMIT,
    DEFAULT_KNOWLEDGE_MAX_BYTES,
    KnowledgeEntry,
    KnowledgeListQuery,
    KnowledgeListResult,
    KnowledgeStatus,
    KnowledgeVisibility,
)

_KNOWLEDGE_REVIEW_STORE_METHODS = (
    "get_entry",
    "transition_entry_status",
    "list_entries",
)


class _KnowledgeReviewStore(Protocol):
    async def get_entry(self, entry_id: str) -> KnowledgeEntry | None: ...

    async def transition_entry_status(
        self,
        entry_id: str,
        *,
        from_status: KnowledgeStatus,
        to_status: KnowledgeStatus,
        expected_namespace: str | None = None,
        expected_labels: dict[str, str] | None = None,
    ) -> KnowledgeEntry: ...

    async def list_entries(self, query: KnowledgeListQuery) -> KnowledgeListResult: ...


class KnowledgeReviewWorkflow:
    """App-side workflow for reviewing model-authored pending knowledge."""

    def __init__(
        self,
        store: _KnowledgeReviewStore,
        *,
        namespace: str | None = None,
        labels: dict[str, str] | None = None,
        default_limit: int = DEFAULT_KNOWLEDGE_LIMIT,
        default_max_bytes: int = DEFAULT_KNOWLEDGE_MAX_BYTES,
    ) -> None:
        _validate_review_store(store)
        self.store = store
        self.namespace = (
            require_clean_nonblank(namespace, "namespace") if namespace is not None else None
        )
        self.labels = copy_label_map(labels or {}, "labels")
        self.default_limit = _positive_int(default_limit, "default_limit")
        self.default_max_bytes = _positive_int(default_max_bytes, "default_max_bytes")

    async def list_pending(
        self,
        *,
        namespace: str | None = None,
        labels: dict[str, str] | None = None,
        kinds: Iterable[str] | None = None,
        visibilities: Iterable[KnowledgeVisibility] | None = None,
        aspects: Iterable[str] | None = None,
        source_type: str | None = None,
        source_id: str | None = None,
        include_expired: bool = False,
        limit: int | None = None,
        max_bytes: int | None = None,
    ) -> KnowledgeListResult:
        """List pending entries within the workflow scope."""

        query = KnowledgeListQuery(
            namespace=self._scoped_namespace(namespace),
            labels=self._scoped_labels(labels),
            kinds=_string_list(kinds, "kinds") if kinds is not None else None,
            statuses=[KnowledgeStatus.PENDING],
            visibilities=list(visibilities) if visibilities is not None else None,
            aspects=_string_list(aspects, "aspects"),
            source_type=source_type,
            source_id=source_id,
            include_expired=include_expired,
            limit=self.default_limit if limit is None else limit,
            max_bytes=self.default_max_bytes if max_bytes is None else max_bytes,
        )
        return await self.store.list_entries(query)

    async def get_pending(self, entry_id: str) -> KnowledgeEntry:
        """Load one pending entry after status and scope checks."""

        return await self._require_pending_entry(entry_id)

    async def approve(self, entry_id: str) -> KnowledgeEntry:
        """Approve one pending entry, making it visible to normal recall."""

        entry = await self._require_pending_entry(entry_id)
        return await self.store.transition_entry_status(
            entry.id,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ACTIVE,
            expected_namespace=self.namespace,
            expected_labels=self.labels,
        )

    async def reject(self, entry_id: str) -> KnowledgeEntry:
        """Reject one pending entry while retaining it for audit."""

        entry = await self._require_pending_entry(entry_id)
        return await self.store.transition_entry_status(
            entry.id,
            from_status=KnowledgeStatus.PENDING,
            to_status=KnowledgeStatus.ARCHIVED,
            expected_namespace=self.namespace,
            expected_labels=self.labels,
        )

    async def _require_pending_entry(self, entry_id: str) -> KnowledgeEntry:
        clean_id = require_clean_nonblank(entry_id, "entry_id")
        entry = await self.store.get_entry(clean_id)
        if entry is None:
            raise KeyError(f"Knowledge entry {clean_id!r} does not exist.")
        self._require_entry_in_scope(entry)
        if entry.status is not KnowledgeStatus.PENDING:
            raise ValueError(
                f"Knowledge entry {clean_id!r} is {entry.status.value!r}, not 'pending'."
            )
        return entry

    def _require_entry_in_scope(self, entry: KnowledgeEntry) -> None:
        if self.namespace is not None and entry.namespace != self.namespace:
            raise PermissionError(
                f"Knowledge entry {entry.id!r} is outside review namespace {self.namespace!r}."
            )
        for key, value in self.labels.items():
            if entry.labels.get(key) != value:
                raise PermissionError(
                    f"Knowledge entry {entry.id!r} is outside review label {key}={value!r}."
                )

    def _scoped_namespace(self, namespace: str | None) -> str | None:
        if namespace is None:
            return self.namespace
        clean_namespace = require_clean_nonblank(namespace, "namespace")
        if self.namespace is not None and clean_namespace != self.namespace:
            raise ValueError(
                f"namespace {clean_namespace!r} conflicts with review namespace {self.namespace!r}."
            )
        return clean_namespace

    def _scoped_labels(self, labels: dict[str, str] | None) -> dict[str, str]:
        scoped = dict(self.labels)
        extra = copy_label_map(labels or {}, "labels")
        for key, value in extra.items():
            if key in scoped and scoped[key] != value:
                raise ValueError(
                    f"label {key}={value!r} conflicts with review label {key}={scoped[key]!r}."
                )
            scoped[key] = value
        return scoped


def _validate_review_store(store: Any) -> None:
    for method_name in _KNOWLEDGE_REVIEW_STORE_METHODS:
        if not callable(getattr(store, method_name, None)):
            raise TypeError("store must implement the knowledge review store methods.")


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int:
        raise ValueError(f"`{name}` must be an integer.")
    if value <= 0:
        raise ValueError(f"`{name}` must be greater than 0.")
    return value


def _string_list(value: Iterable[str] | None, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str | bytes):
        raise ValueError(f"`{name}` must be an iterable of strings.")
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str:
            raise ValueError(f"`{name}[{index}]` must be a string.")
        result.append(require_clean_nonblank(item, f"{name}[{index}]"))
    return list(dict.fromkeys(result))
