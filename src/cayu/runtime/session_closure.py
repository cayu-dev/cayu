"""Bounded, explicit session-closure inspection and erasure contracts.

Closure is deliberately separate from ``SessionStore.delete_session``.  A
session store owns its own cascade, while tasks, artifacts, budgets, knowledge,
and application stores have independent ownership and retention rules.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    DurableValueError,
    canonical_durable_json_bytes,
    copy_bounded_durable_json_value,
    copy_durable_metadata,
    require_durable_clean_nonblank,
)
from cayu.artifacts import ArtifactListResult, ArtifactMetadata, ArtifactScope
from cayu.artifacts._closure import copy_artifact_closure_claim, encode_artifact_closure_claim
from cayu.memory.evidence import (
    MAX_MEMORY_EVIDENCE_ID_CHARS,
    MIN_MEMORY_EVIDENCE_PAGE_BYTES,
    RecallEvidenceQuery,
)
from cayu.runtime._session_closure_records import (
    SESSION_CLOSURE_NATIVE_CLASSES,
    ClosureRecordsBuilder,
    ClosureRecordsTooLarge,
    validate_closure_records,
)
from cayu.sessions.base import SessionLineageQuery, SessionQuery
from cayu.storage._knowledge_closure import (
    KnowledgeClosureQuery,
    validate_knowledge_closure_inventory,
)
from cayu.tasks.base import (
    TaskQuery,
    TaskSessionClosureClaim,
    TaskStatus,
    copy_task_session_closure_claim,
)
from cayu.vaults.redaction import SecretRedactor

SESSION_CLOSURE_SCHEMA_VERSION = 1
SESSION_CLOSURE_DEFAULT_MAX_RECORDS = 10_000
SESSION_CLOSURE_DEFAULT_MAX_BYTES = 16 * 1024 * 1024
SESSION_CLOSURE_MAX_FIELD_CHARS = 4096
SESSION_CLOSURE_DEFAULT_MAX_DESCENDANTS = 10_000


class SessionClosureDisposition(StrEnum):
    """The authoritative disposition of one discovered record class."""

    OWNED_ELIGIBLE = "owned_eligible"
    RETAINED = "retained"
    SHARED = "shared"
    APPLICATION_OWNED = "application_owned"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    TRUNCATED = "truncated"
    ABSENT = "absent"
    ERASED = "erased"


class SessionClosureOperation(StrEnum):
    INSPECT = "inspect"
    EXPORT = "export"
    ERASE = "erase"


class SessionClosureChildPolicy(StrEnum):
    REJECT = "reject"
    DETACH = "detach"
    RECURSIVE = "recursive"


class SessionClosureBudgetDisposition(StrEnum):
    RETAIN = "retain"


class _DescendantBoundExceeded(Exception):
    """Internal signal for a typed, pre-mutation truncation result."""


class SessionClosurePolicy(BaseModel):
    """Bounded caller policy; it never silently expands store enumeration."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    child_policy: SessionClosureChildPolicy = SessionClosureChildPolicy.REJECT
    budget_disposition: SessionClosureBudgetDisposition = SessionClosureBudgetDisposition.RETAIN
    include_artifact_metadata: bool = True
    max_records: StrictInt = Field(default=SESSION_CLOSURE_DEFAULT_MAX_RECORDS, ge=1, le=100_000)
    max_descendants: StrictInt = Field(
        default=SESSION_CLOSURE_DEFAULT_MAX_DESCENDANTS,
        ge=1,
        le=100_000,
    )
    max_bytes: StrictInt = Field(
        default=SESSION_CLOSURE_DEFAULT_MAX_BYTES,
        ge=1,
        le=256 * 1024 * 1024,
    )

    @model_validator(mode="after")
    def validate_policy(self) -> SessionClosurePolicy:
        return self


class SessionClosureRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, hide_input_in_errors=True, revalidate_instances="always"
    )

    store_id: str = Field(max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    record_class: str = Field(max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    disposition: SessionClosureDisposition
    count: StrictInt = Field(default=0, ge=0)
    bytes: StrictInt = Field(default=0, ge=0)
    detail: str | None = Field(default=None, max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    capability: str | None = Field(default=None, max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    erasure_blocked: StrictBool = False


class SessionClosureManifest(BaseModel):
    """Versioned bounded inventory; records never imply unlisted erasure."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: StrictInt = SESSION_CLOSURE_SCHEMA_VERSION
    session_id: str = Field(max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    operation: SessionClosureOperation
    plan_id: str = Field(max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    generated_at: datetime
    complete: bool
    records: tuple[SessionClosureRecord, ...] = Field(max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("Session closure metadata must be an object.")
        return bounded_session_closure_metadata(cast("dict[str, Any]", value))

    @model_validator(mode="after")
    def validate_manifest(self) -> SessionClosureManifest:
        if self.generated_at.tzinfo is None or self.generated_at.utcoffset() is None:
            raise ValueError("Session closure manifest time must be timezone-aware.")
        if self.schema_version != SESSION_CLOSURE_SCHEMA_VERSION:
            raise ValueError("Unsupported session closure manifest version.")
        if not self.plan_id or any(
            not record.store_id or not record.record_class for record in self.records
        ):
            raise ValueError("Session closure manifest identities must be nonblank.")
        return self


class SessionClosureReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: StrictInt = SESSION_CLOSURE_SCHEMA_VERSION
    session_id: str
    plan_id: str
    operation: SessionClosureOperation
    complete: bool
    already_absent: bool = False
    manifest: SessionClosureManifest
    error: str | None = None


class SessionClosureExportIncomplete(ValueError):
    """A normal export could not establish completeness; no record data is returned."""

    def __init__(self, manifest: SessionClosureManifest) -> None:
        super().__init__(
            "Session closure export is incomplete; partial export requires explicit opt-in."
        )
        self.manifest = manifest.model_copy(update={"operation": SessionClosureOperation.EXPORT})


class SessionClosureExport(BaseModel):
    """Bounded export of native records and explicitly supported dependents.

    Public application export applies redaction; this is not a restorable backup.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: StrictInt = SESSION_CLOSURE_SCHEMA_VERSION
    manifest: SessionClosureManifest
    content_redacted: bool = True
    session_records: dict[str, Any] = Field(default_factory=dict)
    _max_bytes: int = PrivateAttr(default=SESSION_CLOSURE_DEFAULT_MAX_BYTES)

    @field_validator("session_records", mode="before")
    @classmethod
    def copy_session_records(cls, value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("Session closure export records must be an object.")
        return copy_bounded_durable_json_value(
            value,
            "session_closure.export.records",
            max_bytes=256 * 1024 * 1024,
            max_nodes=1_000_000,
        )

    def to_bytes(self) -> bytes:
        # The export remains caller-visible and can contain post-construction
        # mutations. Walk owned fields with the effective byte limit before any
        # full serialization; model_dump() would allocate the entire candidate.
        builder = ClosureRecordsBuilder(max_records=1, max_bytes=self._max_bytes)
        builder.add_class("export", (self,))
        value = builder.records["export"][0]
        encoded = canonical_durable_json_bytes(
            value,
            "session_closure.export",
            max_bytes=self._max_bytes,
            max_nodes=1_000_000,
        )
        if len(encoded) > self._max_bytes:
            raise ValueError("Session closure export exceeds the bounded byte limit.")
        return encoded


class SessionClosureProgress(BaseModel):
    """Strict durable authority for one target or recursive closure plan."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: StrictInt = SESSION_CLOSURE_SCHEMA_VERSION
    root_session_id: str = Field(min_length=1, max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    plan_id: str = Field(min_length=1, max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    policy_digest: str = Field(min_length=64, max_length=64)
    max_records: StrictInt = Field(ge=1, le=100_000)
    max_bytes: StrictInt = Field(ge=1, le=256 * 1024 * 1024)
    phase: str = Field(pattern="^(target|reject|recursive)$")
    descendants: tuple[dict[str, str], ...] = Field(max_length=100_000)
    completed: tuple[str, ...] = Field(max_length=100_000)

    @field_validator("descendants", mode="before")
    @classmethod
    def copy_descendants(cls, value: object) -> tuple[dict[str, str], ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("Closure progress descendants must be a list.")
        result = []
        for item in value:
            if not isinstance(item, dict) or set(item) != {"session_id", "parent_session_id"}:
                raise TypeError("Closure progress descendant identity is malformed.")
            if any(type(part) is not str or not part for part in item.values()):
                raise TypeError("Closure progress descendant identity is malformed.")
            result.append(dict(item))
        return tuple(result)

    @field_validator("completed", mode="before")
    @classmethod
    def copy_completed(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)) or any(
            type(item) is not str or not item for item in value
        ):
            raise TypeError("Closure progress completed identities are malformed.")
        return cast("tuple[str, ...]", tuple(value))

    @model_validator(mode="after")
    def validate_identity(self) -> SessionClosureProgress:
        if self.schema_version != SESSION_CLOSURE_SCHEMA_VERSION:
            raise ValueError("Unsupported closure progress schema version.")
        if len(set(item["session_id"] for item in self.descendants)) != len(self.descendants):
            raise ValueError("Closure progress descendants must be unique.")
        descendant_ids = {item["session_id"] for item in self.descendants}
        if (
            len(set(self.completed)) != len(self.completed)
            or not set(self.completed) <= descendant_ids
        ):
            raise ValueError("Closure progress completed identities conflict with descendants.")
        return self


class SessionClosureStore(Protocol):
    """Optional independent-store adapter owned by an application boundary."""

    store_id: str

    async def inspect_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
    ) -> SessionClosureRecord: ...

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ) -> SessionClosureRecord: ...

    async def export_session_closure(
        self, session_id: str, *, policy: SessionClosurePolicy
    ) -> dict[str, Any]: ...


class SessionClosureLineageStore(Protocol):
    """Atomic session-store operations required by descendant closure."""

    async def claim_session_closure_progress(self, progress: dict[str, Any]) -> None: ...

    async def detach_session_children(
        self,
        parent_session_id: str,
        child_session_ids: tuple[str, ...],
        *,
        closure_receipt: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]: ...

    async def load_session_closure_tombstones(
        self, root_session_id: str, plan_id: str
    ) -> tuple[dict[str, Any], ...]: ...


class UnsupportedSessionClosureStore:
    """Explicit capability marker used when an owned store lacks closure APIs."""

    def __init__(self, store_id: str) -> None:
        self.store_id = store_id

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        raise NotImplementedError(f"{self.store_id} does not support session closure inspection.")

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ):
        raise NotImplementedError(f"{self.store_id} does not support session closure erasure.")


class RetainedSessionClosureStore:
    """Explicit non-destructive owner for data retained by closure policy."""

    def __init__(self, store_id: str, record_class: str = "records") -> None:
        self.store_id = store_id
        self.record_class = record_class

    def _record(self) -> SessionClosureRecord:
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class=self.record_class,
            disposition=SessionClosureDisposition.RETAINED,
            detail="retained by the configured closure policy",
        )

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        return self._record()

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ):
        return self._record()

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        return {"disposition": "retained", "record_class": self.record_class}


class BudgetSessionClosureStore(RetainedSessionClosureStore):
    """Explicit retain-only boundary for financial and accounting records.

    Budget history is not session-owned deletion material.  Keeping this as a
    distinct adapter prevents a future policy extension from accidentally
    treating the generic retained-store implementation as support for a
    destructive budget disposition.
    """

    def __init__(self, store_id: str = "budget-store", record_class: str = "budget_ledger") -> None:
        super().__init__(store_id, record_class)

    def _require_retention(self, policy: SessionClosurePolicy) -> None:
        if policy.budget_disposition is not SessionClosureBudgetDisposition.RETAIN:
            raise ValueError(
                "Budget closure supports retention only; destructive budget dispositions "
                "require an explicit budget adapter."
            )

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        self._require_retention(policy)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class=self.record_class,
            disposition=SessionClosureDisposition.RETAINED,
            detail="budget history is retained by the accounting policy",
            capability="budget.retention-only",
        )

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ):
        self._require_retention(policy)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class=self.record_class,
            disposition=SessionClosureDisposition.RETAINED,
            detail="budget history is retained by the accounting policy",
            capability="budget.retention-only",
        )

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        self._require_retention(policy)
        return {
            "disposition": SessionClosureDisposition.RETAINED.value,
            "record_class": self.record_class,
            "capability": "budget.retention-only",
        }


class SharedSessionClosureStore(RetainedSessionClosureStore):
    """Reference-only owner whose records are never session-owned."""

    def _record(self) -> SessionClosureRecord:
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class=self.record_class,
            disposition=SessionClosureDisposition.SHARED,
            detail="shared or independently retained; not deleted by session closure",
        )


class KnowledgeSessionClosureStore:
    """Retain shared knowledge and inventory exact supported source references."""

    store_id = "knowledge-store"

    def __init__(self, store, session_store, artifact_stores=()) -> None:
        self._store = store
        self._session_store = session_store
        self._artifact_stores = tuple(artifact_stores)

    async def _inventory(self, session_id: str, policy: SessionClosurePolicy):
        loader = getattr(self._session_store, "load_session_closure_records", None)
        inspect_sources = getattr(self._store, "inspect_closure_sources", None)
        if loader is None or inspect_sources is None:
            raise NotImplementedError("Knowledge closure source enumeration is unavailable.")
        snapshot = validate_closure_records(
            await loader(session_id, max_records=policy.max_records, max_bytes=policy.max_bytes),
            max_records=policy.max_records,
            max_bytes=policy.max_bytes,
        )
        sessions = snapshot["records"]["session"]
        if len(sessions) > 1 or (sessions and sessions[0].get("id") != session_id):
            raise ValueError("Knowledge closure source session conflicts.")
        sources = [("session", session_id), ("tool", session_id)]
        for stored_event in snapshot["records"]["events"]:
            if type(stored_event) is not dict:
                raise ValueError("Knowledge closure stored event is malformed.")
            event = stored_event.get("event")
            if type(event) is not dict or event.get("session_id") != session_id:
                raise ValueError("Knowledge closure event ownership conflicts.")
            sources.append(("session_event", event.get("id")))
        for artifact_store in self._artifact_stores:
            artifacts, truncated = await artifact_store._source_inventory(session_id, policy)
            if truncated:
                raise ValueError("Knowledge closure artifact sources are truncated.")
            for artifact in artifacts:
                if len(sources) >= policy.max_records:
                    raise ValueError("Knowledge closure sources exceed their bound.")
                sources.append(("artifact", artifact.id))
        query = KnowledgeClosureQuery(
            sources=tuple(sources),
            max_records=policy.max_records,
            max_bytes=policy.max_bytes,
            source_uris=tuple(
                (kind, f"cayu://sessions/{session_id}") for kind in ("session", "tool")
            ),
        )
        return validate_knowledge_closure_inventory(await inspect_sources(query), query)

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        inventory = await self._inventory(session_id, policy)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="knowledge_references",
            disposition=SessionClosureDisposition.SHARED,
            count=sum(inventory["counts"].values()),
            bytes=len(canonical_durable_json_bytes(inventory, "knowledge closure inventory")),
            detail="shared knowledge is retained; export contains references, not content",
            capability="knowledge.inspect_closure_sources",
        )

    async def erase_session_closure(
        self, session_id: str, *, policy: SessionClosurePolicy, plan_id: str
    ):
        return await self.inspect_session_closure(session_id, policy=policy)

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        return await self._inventory(session_id, policy)


class SessionEvidenceClosureStore:
    """Inventory #946 receipt/exposure references owned by SessionStore."""

    store_id = "session-store-evidence"

    def __init__(self, store: Any) -> None:
        self._store = store

    async def _inventory(
        self, session_id: str, policy: SessionClosurePolicy
    ) -> tuple[dict[str, Any], int, bool]:
        list_receipts = getattr(self._store, "list_recall_receipts", None)
        list_exposures = getattr(self._store, "list_context_exposures", None)
        if list_receipts is None or list_exposures is None:
            raise NotImplementedError("Session evidence enumeration is unavailable.")
        records: dict[str, Any] = {"recall_receipts": [], "context_exposures": []}
        size = len(canonical_durable_json_bytes(records, "session_closure.evidence"))
        count = 0
        if policy.max_bytes < max(size, MIN_MEMORY_EVIDENCE_PAGE_BYTES):
            return records, 0, True
        for key, identity_field, reader in (
            ("recall_receipts", "receipt_id", list_receipts),
            ("context_exposures", "exposure_id", list_exposures),
        ):
            cursor = None
            seen_cursors: set[str] = set()
            seen_ids: set[str] = set()
            while True:
                limit = min(100, policy.max_records - count + 1)
                page = await reader(
                    RecallEvidenceQuery(
                        session_id=session_id,
                        limit=limit,
                        max_bytes=min(policy.max_bytes, 1_000_000),
                        cursor=cursor,
                    )
                )
                if len(page.items) > limit or type(page.truncated) is not bool:
                    raise ValueError("Invalid session evidence page.")
                for item in page.items:
                    identity = getattr(item, identity_field, None)
                    if (
                        type(identity) is not str
                        or not identity
                        or len(identity) > MAX_MEMORY_EVIDENCE_ID_CHARS
                        or getattr(item, "session_id", None) != session_id
                        or identity in seen_ids
                    ):
                        raise ValueError("Session evidence identity or ownership conflicts.")
                    if count == policy.max_records:
                        return records, size, True
                    summary = {"id_digest": _export_identity(identity)}
                    item_size = len(
                        canonical_durable_json_bytes(summary, "session_closure.evidence.item")
                    ) + bool(records[key])
                    if size + item_size > policy.max_bytes:
                        return records, size, True
                    records[key].append(summary)
                    seen_ids.add(identity)
                    size += item_size
                    count += 1
                next_cursor = page.next_cursor
                if not page.truncated:
                    if next_cursor is not None:
                        raise ValueError("Session evidence cursor conflicts with completeness.")
                    break
                if (
                    type(next_cursor) is not str
                    or not next_cursor
                    or next_cursor in seen_cursors
                    or not page.items
                ):
                    raise ValueError("Session evidence pagination is truncated or invalid.")
                seen_cursors.add(next_cursor)
                cursor = next_cursor
        return records, size, False

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        records, size, truncated = await self._inventory(session_id, policy)
        count = sum(len(items) for items in records.values())
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="recall_receipts_context_exposures",
            disposition=(
                SessionClosureDisposition.TRUNCATED
                if truncated
                else SessionClosureDisposition.OWNED_ELIGIBLE
                if count
                else SessionClosureDisposition.ABSENT
            ),
            count=count,
            bytes=size,
            detail=(
                "session-owned evidence is deleted by the final SessionStore cascade"
                if count
                else None
            ),
            capability="session.list_recall_receipts/list_context_exposures",
        )

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ):
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="recall_receipts_context_exposures",
            disposition=SessionClosureDisposition.OWNED_ELIGIBLE,
            detail="awaiting the final SessionStore deletion",
            capability="session-store.cascade",
        )

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        records, _size, truncated = await self._inventory(session_id, policy)
        if truncated:
            raise ValueError("Session evidence export is truncated.")
        return records


@dataclass(frozen=True)
class _ClosureArtifact:
    id: str
    size_bytes: int


class ArtifactSessionClosureStore:
    """Bounded adapter for one registered session-scoped artifact store."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.store_id = f"artifact-store:{store.id}"

    async def _retained_claim(self, session_id, policy):
        if getattr(self._store, "supports_session_closure_claims", False) is not True:
            return None
        loader = getattr(self._store, "load_session_closure_claim", None)
        if loader is None:
            return None
        raw = await loader(session_id)
        if raw is None:
            return None
        claim = copy_artifact_closure_claim(raw)
        if claim.store_id != self._store.id or claim.session_id != session_id:
            raise ValueError("Artifact closure claim authority conflicts.")
        if (
            len(claim.artifacts) > policy.max_records
            or len(encode_artifact_closure_claim(claim)) > policy.max_bytes
        ):
            raise ValueError("Artifact closure claim exceeds its requested bounds.")
        return claim

    async def _source_inventory(self, session_id, policy):
        # A retained claim remains positive source-reference evidence after
        # partial deletion. It does not assert the artifacts still exist or
        # authorize a different closure plan to mutate them.
        if getattr(self._store, "supports_session_closure_claims", False) is True:
            raw = await self._store.load_session_closure_claim(session_id)
            if raw is not None:
                claim = copy_artifact_closure_claim(raw)
                if claim.store_id != self._store.id or claim.session_id != session_id:
                    raise ValueError("Artifact closure source claim conflicts.")
                if (
                    len(claim.artifacts) > policy.max_records
                    or len(encode_artifact_closure_claim(claim)) > policy.max_bytes
                ):
                    raise ValueError("Artifact closure sources exceed their bound.")
                return tuple(
                    _ClosureArtifact(item.artifact_id, item.size_bytes) for item in claim.artifacts
                ), False
        return await self._inventory(session_id, policy)

    async def claim_session_closure(self, session_id, policy, plan_id):
        retained = await self._retained_claim(session_id, policy)
        if retained is not None:
            if retained.plan_id != plan_id:
                raise ValueError("Artifact closure claim authority conflicts.")
        else:
            _artifacts, truncated = await self._inventory(session_id, policy)
            if truncated:
                raise ValueError("Artifact closure inventory is truncated.")
        if getattr(self._store, "supports_session_closure_claims", False) is not True:
            raise NotImplementedError("Artifact store cannot fence session closure.")
        claim = copy_artifact_closure_claim(
            await self._store.claim_session_closure(
                session_id, plan_id, max_records=policy.max_records, max_bytes=policy.max_bytes
            )
        )
        if (
            claim.store_id != self._store.id
            or claim.session_id != session_id
            or claim.plan_id != plan_id
            or (retained is not None and claim != retained)
        ):
            raise ValueError("Artifact closure claim authority conflicts.")
        if (
            len(claim.artifacts) > policy.max_records
            or len(encode_artifact_closure_claim(claim)) > policy.max_bytes
        ):
            raise ValueError("Artifact closure claim exceeds its requested bounds.")
        return claim

    async def _inventory(
        self, session_id: str, policy: SessionClosurePolicy
    ) -> tuple[tuple[_ClosureArtifact, ...], bool]:
        result = await self._store.list(
            scope=ArtifactScope.SESSION,
            session_id=session_id,
            limit=policy.max_records,
        )
        if type(result) is not ArtifactListResult:
            raise ValueError("Artifact closure requires a typed inventory.")
        artifacts, total, truncated = result.artifacts, result.total_count, result.truncated
        if type(artifacts) is not tuple:
            raise ValueError("Artifact closure requires a typed inventory.")
        if (
            type(truncated) is not bool
            or len(artifacts) > policy.max_records
            or (
                total is not None
                and (
                    type(total) is not int
                    or not len(artifacts) <= total <= MAX_DURABLE_JSON_INTEGER
                )
            )
            or (not truncated and total != len(artifacts))
        ):
            raise ValueError("Artifact closure inventory has inconsistent bounds or counts.")
        selected: list[_ClosureArtifact] = []
        seen: set[str] = set()
        remaining = policy.max_bytes
        content_bytes = 0
        for artifact in artifacts:
            if type(artifact) is not ArtifactMetadata:
                raise ValueError("Artifact closure inventory has invalid ownership evidence.")
            artifact_id, scope, owner, size = (
                artifact.id,
                artifact.scope,
                artifact.session_id,
                artifact.size_bytes,
            )
            if (
                scope is not ArtifactScope.SESSION
                or type(owner) is not str
                or owner != session_id
                or type(artifact_id) is not str
                or type(size) is not int
                or not 0 <= size <= MAX_DURABLE_JSON_INTEGER
            ):
                raise ValueError("Artifact closure inventory has invalid ownership evidence.")
            identity = require_durable_clean_nonblank(artifact_id, "artifact identity")
            if identity in seen:
                raise ValueError("Artifact closure inventory contains duplicate identities.")
            content_bytes += size
            if content_bytes > MAX_DURABLE_JSON_INTEGER:
                raise ValueError("Artifact closure content count exceeds its durable bound.")
            # Only these validated fields govern closure. Do not serialize an
            # extension-owned model (including unused metadata or filename).
            try:
                encoded = canonical_durable_json_bytes(
                    {"id": identity, "size_bytes": size},
                    "artifact closure identity",
                    max_bytes=max(1, remaining),
                    max_nodes=4,
                )
            except DurableValueError:
                return tuple(selected), True
            if len(encoded) > remaining:
                return tuple(selected), True
            remaining -= len(encoded)
            seen.add(identity)
            selected.append(_ClosureArtifact(identity, size))
        return tuple(selected), truncated

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        claim = await self._retained_claim(session_id, policy)
        if claim is not None:
            return SessionClosureRecord(
                store_id=self.store_id,
                record_class="artifact_cleanup_claim",
                disposition=SessionClosureDisposition.OWNED_ELIGIBLE,
                count=len(claim.artifacts),
                bytes=sum(item.size_bytes for item in claim.artifacts),
                detail="Retained cleanup authority; counts describe the original claimed set, not live artifacts.",
                capability="artifact.load_session_closure_claim",
            )
        artifacts, truncated = await self._inventory(session_id, policy)
        if truncated:
            return SessionClosureRecord(
                store_id=self.store_id,
                record_class="session_artifacts",
                disposition=SessionClosureDisposition.TRUNCATED,
                count=len(artifacts),
                bytes=sum(item.size_bytes for item in artifacts),
                detail="artifact enumeration exceeded the closure bound",
                capability="artifact.list",
            )
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="session_artifacts",
            disposition=(
                SessionClosureDisposition.OWNED_ELIGIBLE
                if artifacts
                else SessionClosureDisposition.ABSENT
            ),
            count=len(artifacts),
            bytes=sum(item.size_bytes for item in artifacts),
            capability="artifact.list/delete",
        )

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ):
        claim = await self.claim_session_closure(session_id, policy, plan_id)
        for artifact in claim.artifacts:
            await self._store.delete_session_closure_artifact(claim, artifact.artifact_id)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="session_artifacts",
            disposition=SessionClosureDisposition.ERASED,
            count=len(claim.artifacts),
            bytes=sum(item.size_bytes for item in claim.artifacts),
            capability="artifact.claim_session_closure/delete_session_closure_artifact",
        )

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        if not policy.include_artifact_metadata:
            return {"disposition": "omitted", "reason": "artifact metadata excluded by policy"}
        artifacts, truncated = await self._inventory(session_id, policy)
        if truncated:
            raise ValueError("Artifact closure export is truncated.")
        return {
            "artifacts": [
                {
                    "id_digest": _export_identity(artifact.id),
                    "size_bytes": artifact.size_bytes,
                    "scope": ArtifactScope.SESSION.value,
                }
                for artifact in artifacts
            ]
        }


class TaskSessionClosureStore:
    """Bounded task-session adapter; deletion requires an explicit store seam."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.store_id = "task-store"

    async def _tasks(self, session_id: str, policy: SessionClosurePolicy) -> tuple[Any, ...]:
        tasks = []
        while len(tasks) <= policy.max_records:
            page_size = min(1000, policy.max_records + 1 - len(tasks))
            page = await self._store.list_tasks(
                TaskQuery(session_id=session_id, limit=page_size, offset=len(tasks))
            )
            if len(page) > page_size or any(task.session_id != session_id for task in page):
                raise ValueError("Task store returned invalid closure enumeration.")
            tasks.extend(page)
            if len(page) < page_size:
                break
        return tuple(tasks)

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        tasks = await self._tasks(session_id, policy)
        truncated = len(tasks) > policy.max_records
        visible = tasks[: policy.max_records]
        active = any(
            task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
            or task.worker_id is not None
            or task.lease_expires_at is not None
            for task in visible
        )
        if not getattr(self._store, "supports_session_closure_claims", False) or (
            visible and not getattr(self._store, "supports_session_closure_deletion", False)
        ):
            disposition = SessionClosureDisposition.UNSUPPORTED
        elif truncated:
            disposition = SessionClosureDisposition.TRUNCATED
        elif active:
            disposition = SessionClosureDisposition.RETAINED
        elif visible:
            disposition = SessionClosureDisposition.OWNED_ELIGIBLE
        else:
            disposition = SessionClosureDisposition.ABSENT
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="session_tasks",
            disposition=disposition,
            count=len(visible),
            detail=("one or more tasks are still active" if active else None),
            capability="task.list",
            erasure_blocked=active,
        )

    async def claim_session_closure(
        self, session_id: str, *, policy: SessionClosurePolicy, plan_id: str
    ) -> TaskSessionClosureClaim:
        if not getattr(self._store, "supports_session_closure_claims", False):
            raise NotImplementedError("Task store does not support session closure claims.")
        existing = await self._store.load_session_closure_claim(session_id)
        if existing is not None:
            claim = copy_task_session_closure_claim(existing)
            if claim.session_id != session_id or claim.plan_id != plan_id:
                raise ValueError("Task closure claim conflicts with its retained authority.")
        else:
            tasks = await self._tasks(session_id, policy)
            claim = TaskSessionClosureClaim(
                session_id=session_id, plan_id=plan_id, task_ids=tuple(task.id for task in tasks)
            )
        if len(claim.task_ids) > policy.max_records:
            raise ValueError("Task closure inventory is truncated.")
        admitted = copy_task_session_closure_claim(await self._store.claim_session_closure(claim))
        if admitted != claim:
            raise ValueError("Task closure admission returned conflicting authority.")
        return admitted

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ):
        claim = await self.claim_session_closure(session_id, policy=policy, plan_id=plan_id)
        delete = getattr(self._store, "delete_session_tasks", None)
        if delete is None:
            raise NotImplementedError("Task store does not support session task deletion.")
        await delete(session_id, task_ids=claim.task_ids, policy=policy)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="session_tasks",
            disposition=SessionClosureDisposition.ERASED,
            count=len(claim.task_ids),
            capability="task.list/delete",
        )

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        tasks = await self._tasks(session_id, policy)
        if len(tasks) > policy.max_records:
            raise ValueError("Task closure export is truncated.")
        return {
            "tasks": [
                {
                    "id_digest": _export_identity(task.id),
                    "type_digest": _export_identity(task.type),
                    "status": getattr(task.status, "value", task.status),
                    "session_id_digest": _export_identity(task.session_id),
                }
                for task in tasks[: policy.max_records]
            ]
        }


class SessionClosureCoordinator:
    """Coordinate independent closure adapters without pretending atomicity."""

    def __init__(
        self,
        session_store: Any,
        *,
        dependent_stores: tuple[SessionClosureStore, ...] = (),
        clock: Any = datetime.now,
        secret_redactor: SecretRedactor | None = None,
    ) -> None:
        self._session_store = session_store
        self._dependent_stores = tuple(dependent_stores)
        store_ids: set[str] = set()
        for store in self._dependent_stores:
            store_id = store.store_id
            if (
                type(store_id) is not str
                or not store_id
                or len(store_id) > SESSION_CLOSURE_MAX_FIELD_CHARS
                or store_id == "session-store"
                or store_id.startswith("session-store/")
                or store_id in store_ids
            ):
                raise ValueError("Closure dependent store identity is invalid or reserved.")
            require_durable_clean_nonblank(store_id, "closure store identity")
            store_ids.add(store_id)
        self._session_cascade_store_ids = frozenset(
            store.store_id
            for store in self._dependent_stores
            if type(store) is SessionEvidenceClosureStore
        )
        self._clock = clock
        self._secret_redactor = secret_redactor or SecretRedactor()
        self._completed_plan_ids: set[str] = set()
        self._erase_lock = asyncio.Lock()

    def _copy_adapter_record(self, value: object) -> SessionClosureRecord:
        # Revalidate before serialization, including extension-owned instances
        # mutated after construction. Only detached, validated text is redacted.
        record = SessionClosureRecord.model_validate(value)
        return SessionClosureRecord.model_validate(
            record.model_copy(
                update={
                    "detail": None
                    if record.detail is None
                    else self._secret_redactor.redact_text(record.detail),
                    "capability": None
                    if record.capability is None
                    else self._secret_redactor.redact_text(record.capability),
                }
            )
        )

    def _plan_id(self, session_id: str, policy: SessionClosurePolicy) -> str:
        store_ids = ("session-store", *(store.store_id for store in self._dependent_stores))
        return session_closure_plan_id(session_id, policy=policy, store_ids=store_ids)

    async def _load_descendants(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
    ) -> tuple[tuple[str, str], ...] | None:
        """Return bounded ``(child, parent)`` identities in stable tree order.

        ``None`` means the store could not prove a complete bounded lineage.  No
        caller may treat that result as an empty tree.
        """

        query_lineage = getattr(self._session_store, "query_session_lineage", None)
        if query_lineage is None or not getattr(
            self._session_store, "supports_session_lineage", False
        ):
            list_sessions = getattr(self._session_store, "list_sessions", None)
            if list_sessions is None:
                return None
            descendants: list[tuple[str, str]] = []
            descendant_bytes = 0
            cursor = None
            while True:
                limit = min(1000, policy.max_descendants - len(descendants) + 1)
                if limit < 1:
                    raise _DescendantBoundExceeded
                page = await list_sessions(
                    SessionQuery(
                        parent_session_id=session_id,
                        limit=limit,
                        offset=0,
                        cursor=cursor,
                    )
                )
                children = list(getattr(page, "sessions", page))
                for child in children:
                    descendants.append((child.id, session_id))
                    descendant_bytes += len(
                        canonical_durable_json_bytes(
                            {"id": child.id, "parent_session_id": session_id},
                            "session_closure.descendant",
                        )
                    )
                    if (
                        len(descendants) > policy.max_descendants
                        or descendant_bytes > policy.max_bytes
                    ):
                        raise _DescendantBoundExceeded
                next_cursor = getattr(page, "next_cursor", None)
                if next_cursor is None:
                    # A non-empty bare list has no way to prove completeness;
                    # never treat an unpaged response as authoritative.
                    if not hasattr(page, "sessions"):
                        return None if children else tuple(descendants)
                    return tuple(descendants)
                if next_cursor == cursor or not children:
                    return None
                cursor = next_cursor
        pending = [session_id]
        descendants: list[tuple[str, str]] = []
        descendant_bytes = 0
        while pending:
            parent_id = pending.pop(0)
            cursor = None
            while True:
                page = await query_lineage(
                    SessionLineageQuery(
                        parent_session_id=parent_id,
                        cursor=cursor,
                        limit=min(100, policy.max_descendants + 1),
                    )
                )
                for child in page.children:
                    descendants.append((child.id, parent_id))
                    descendant_bytes += len(
                        canonical_durable_json_bytes(
                            {"id": child.id, "parent_session_id": parent_id},
                            "session_closure.descendant",
                        )
                    )
                    if len(descendants) > policy.max_descendants:
                        raise _DescendantBoundExceeded
                    if descendant_bytes > policy.max_bytes:
                        raise _DescendantBoundExceeded
                    if len(descendants) == len({item[0] for item in descendants}):
                        pending.append(child.id)
                    else:
                        return None
                if not page.has_more:
                    break
                cursor = page.next_cursor
        return tuple(descendants)

    async def _save_progress(self, progress: dict[str, Any]) -> None:
        save = getattr(self._session_store, "save_session_closure_progress", None)
        if (
            not getattr(self._session_store, "supports_session_closure_progress", False)
            or save is None
        ):
            raise ValueError("Session store does not support durable closure progress.")
        try:
            await save(progress)
        except NotImplementedError as exc:
            raise ValueError("Session store does not support durable closure progress.") from exc

    async def _load_progress(self, session_id: str, plan_id: str) -> dict[str, Any] | None:
        load = getattr(self._session_store, "load_session_closure_progress", None)
        if load is None:
            return None
        try:
            return await load(session_id, plan_id)
        except NotImplementedError:
            return None

    async def _claim_progress(
        self,
        session_id: str,
        plan_id: str,
        policy: SessionClosurePolicy,
        *,
        descendants: tuple[tuple[str, str], ...] = (),
        phase: str = "target",
    ) -> dict[str, Any]:
        progress = {
            "schema_version": 1,
            "root_session_id": session_id,
            "plan_id": plan_id,
            "policy_digest": sha256(
                canonical_durable_json_bytes(
                    policy.model_dump(mode="json"), "session_closure.policy"
                )
            ).hexdigest(),
            "phase": phase,
            "max_records": policy.max_records,
            "max_bytes": policy.max_bytes,
            "descendants": [
                {"session_id": child_id, "parent_session_id": parent_id}
                for child_id, parent_id in descendants
            ],
            "completed": [],
        }
        claim = getattr(self._session_store, "claim_session_closure_progress", None)
        if claim is None:
            raise ValueError("Session store does not support closure lineage claims.")
        await claim(progress)
        return progress

    async def _validated_progress(
        self,
        progress: dict[str, Any],
        *,
        session_id: str,
        plan_id: str,
        policy: SessionClosurePolicy,
        descendants: tuple[tuple[str, str], ...],
    ) -> SessionClosureProgress:
        value = SessionClosureProgress.model_validate(progress)
        expected_digest = sha256(
            canonical_durable_json_bytes(policy.model_dump(mode="json"), "session_closure.policy")
        ).hexdigest()
        if (
            value.max_records != policy.max_records
            or value.max_bytes != policy.max_bytes
            or value.root_session_id != session_id
            or value.plan_id != plan_id
            or value.policy_digest != expected_digest
        ):
            raise ValueError("Durable closure progress does not match the admitted plan.")
        planned = {(item["session_id"], item["parent_session_id"]) for item in value.descendants}
        current = set(descendants)
        if not current <= planned or any(child in value.completed for child, _ in current):
            raise ValueError("Recursive closure lineage conflicts with the saved plan.")
        completed = set(value.completed)
        load_receipt = getattr(self._session_store, "load_session_closure_receipt", None)
        for child_id, parent_id in planned - current:
            child_plan_id = session_closure_target_plan_id(plan_id, child_id)
            receipt = None if load_receipt is None else await load_receipt(child_id, child_plan_id)
            if (
                receipt is None
                or receipt.get("root_session_id") != session_id
                or receipt.get("target_session_id") != child_id
                or receipt.get("original_parent_session_id") != parent_id
                or receipt.get("plan_id") != child_plan_id
                or receipt.get("operation") != "recursive"
                or receipt.get("complete") is not True
                or await self._session_store.load(child_id) is not None
            ):
                raise ValueError("Missing descendant has no matching closure completion receipt.")
            completed.add(child_id)
        return value.model_copy(update={"completed": tuple(sorted(completed))})

    async def _child_record(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
    ) -> tuple[SessionClosureRecord, tuple[tuple[str, str], ...] | None]:
        try:
            descendants = await self._load_descendants(session_id, policy=policy)
        except _DescendantBoundExceeded:
            return (
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="child_sessions",
                    disposition=SessionClosureDisposition.TRUNCATED,
                    count=policy.max_descendants,
                    detail="descendant bound exceeded before mutation",
                    capability="session.query_session_lineage",
                ),
                None,
            )
        if descendants is None:
            return (
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="child_sessions",
                    disposition=SessionClosureDisposition.UNSUPPORTED,
                    capability="session.query_session_lineage",
                ),
                None,
            )
        if len(descendants) > policy.max_descendants:
            return (
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="child_sessions",
                    disposition=SessionClosureDisposition.TRUNCATED,
                    count=policy.max_descendants,
                    detail="descendant bound exceeded before mutation",
                    capability="session.query_session_lineage",
                ),
                None,
            )
        if (
            descendants
            and policy.child_policy is SessionClosureChildPolicy.DETACH
            and not getattr(self._session_store, "supports_session_closure_detachment", False)
        ):
            return (
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="child_sessions",
                    disposition=SessionClosureDisposition.UNSUPPORTED,
                    count=len(descendants),
                    capability="session.supports_session_closure_detachment",
                ),
                descendants,
            )
        if (
            descendants
            and policy.child_policy is SessionClosureChildPolicy.RECURSIVE
            and not all(
                getattr(self._session_store, name, False)
                for name in (
                    "supports_session_closure_recursive_deletion",
                    "supports_session_closure_progress",
                )
            )
        ):
            return (
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="child_sessions",
                    disposition=SessionClosureDisposition.UNSUPPORTED,
                    count=len(descendants),
                    capability="session.supports_session_closure_recursive_deletion+progress",
                ),
                descendants,
            )
        disposition = (
            SessionClosureDisposition.RETAINED
            if descendants and policy.child_policy is SessionClosureChildPolicy.REJECT
            else SessionClosureDisposition.OWNED_ELIGIBLE
            if descendants
            else SessionClosureDisposition.ABSENT
        )
        return (
            SessionClosureRecord(
                store_id="session-store",
                record_class="child_sessions",
                disposition=disposition,
                count=len(descendants),
                detail=(
                    "child policy must explicitly handle descendants"
                    if descendants and policy.child_policy is SessionClosureChildPolicy.REJECT
                    else None
                ),
                capability="session.query_session_lineage",
            ),
            descendants,
        )

    async def _require_quiescent_session(self, session_id: str) -> Any:
        session = await self._session_store.load(session_id)
        if session is None:
            raise ValueError("Closure target disappeared during admission.")
        if session.status.value in {"running", "interrupting"}:
            raise ValueError("Session closure requires terminal sessions.")
        admission = getattr(self._session_store, "validate_session_closure_admission", None)
        if admission is None:
            raise NotImplementedError("Session closure admission is unsupported.")
        await admission(session_id)
        return session

    async def _require_native_admission(
        self, session_id: str, policy: SessionClosurePolicy
    ) -> None:
        native = validate_closure_records(
            await self._session_store.load_session_closure_records(
                session_id, max_records=policy.max_records, max_bytes=policy.max_bytes
            ),
            max_records=policy.max_records,
            max_bytes=policy.max_bytes,
        )
        if (
            native["counts"]["session"] != 1
            or type(native["records"]["session"][0]) is not dict
            or native["records"]["session"][0].get("id") != session_id
        ):
            raise ValueError("Closure session disappeared during native admission.")

    async def _require_dependent_admission(
        self, session_id: str, policy: SessionClosurePolicy
    ) -> None:
        for store in self._dependent_stores:
            try:
                record = self._copy_adapter_record(
                    await store.inspect_session_closure(session_id, policy=policy)
                )
            except Exception:
                raise ValueError("Dependent closure admission is unavailable.") from None
            if (
                record.store_id != store.store_id
                or record.erasure_blocked
                or record.disposition
                in {
                    SessionClosureDisposition.UNSUPPORTED,
                    SessionClosureDisposition.UNAVAILABLE,
                    SessionClosureDisposition.TRUNCATED,
                }
            ):
                raise ValueError("Dependent closure admission is blocked or incomplete.")

    async def _claim_dependent_sets(
        self, session_id: str, policy: SessionClosurePolicy, plan_id: str
    ) -> None:
        for store in self._dependent_stores:
            if type(store) is TaskSessionClosureStore:
                await store.claim_session_closure(session_id, policy=policy, plan_id=plan_id)
            elif type(store) is ArtifactSessionClosureStore:
                await store.claim_session_closure(session_id, policy, plan_id)

    def _validated_settlement(
        self, store: SessionClosureStore, result: Any
    ) -> SessionClosureRecord:
        record = self._copy_adapter_record(result)
        if record.store_id != store.store_id or record.erasure_blocked:
            raise ValueError("Dependent closure settlement authority conflicts.")
        allowed = {
            SessionClosureDisposition.ERASED,
            SessionClosureDisposition.ABSENT,
            SessionClosureDisposition.RETAINED,
            SessionClosureDisposition.SHARED,
            SessionClosureDisposition.APPLICATION_OWNED,
        }
        if store.store_id in self._session_cascade_store_ids:
            allowed.add(SessionClosureDisposition.OWNED_ELIGIBLE)
        if record.disposition not in allowed:
            raise ValueError("Dependent closure settlement is incomplete.")
        return record

    async def inspect(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy | None = None,
    ) -> SessionClosureManifest:
        """Build a bounded inventory; unsupported adapters remain visible."""

        policy = SessionClosurePolicy() if policy is None else policy
        plan_id = self._plan_id(session_id, policy)
        generated_at = self._clock()
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            generated_at = generated_at.replace(tzinfo=UTC)
        session = await self._session_store.load(session_id)
        records: list[SessionClosureRecord] = []
        descendant_ids: tuple[str, ...] = ()
        descendant_pairs: tuple[tuple[str, str], ...] = ()
        if session is None:
            records.append(
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="session",
                    disposition=SessionClosureDisposition.ABSENT,
                )
            )
            complete = True
            records.append(
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="child_sessions",
                    disposition=SessionClosureDisposition.ABSENT,
                    capability="session.query_session_lineage",
                )
            )
        else:
            source_complete = True
            try:
                load_records = getattr(self._session_store, "load_session_closure_records", None)
                if load_records is None:
                    raise NotImplementedError
                native = validate_closure_records(
                    await load_records(
                        session_id, max_records=policy.max_records, max_bytes=policy.max_bytes
                    ),
                    max_records=policy.max_records,
                    max_bytes=policy.max_bytes,
                )
                if (
                    native["counts"]["session"] != 1
                    or type(native["records"]["session"][0]) is not dict
                    or native["records"]["session"][0].get("id") != session_id
                ):
                    raise ValueError("Closure session disappeared during enumeration.")
                records.extend(
                    SessionClosureRecord(
                        store_id="session-store",
                        record_class=name,
                        disposition=SessionClosureDisposition.OWNED_ELIGIBLE
                        if native["counts"][name]
                        else SessionClosureDisposition.ABSENT,
                        count=native["counts"][name],
                        bytes=native["record_bytes"][name],
                        capability="session.load_session_closure_records",
                    )
                    for name in SESSION_CLOSURE_NATIVE_CLASSES
                )
            except Exception as exc:
                source_complete = False
                disposition = (
                    SessionClosureDisposition.UNSUPPORTED
                    if isinstance(exc, NotImplementedError)
                    else SessionClosureDisposition.TRUNCATED
                    if isinstance(exc, ClosureRecordsTooLarge)
                    or (
                        isinstance(exc, DurableValueError)
                        and exc.code in {"json_value_too_large", "too_many_json_nodes"}
                    )
                    else SessionClosureDisposition.UNAVAILABLE
                )
                records.extend(
                    SessionClosureRecord(
                        store_id="session-store",
                        record_class=name,
                        disposition=disposition,
                        capability="session.load_session_closure_records",
                    )
                    for name in SESSION_CLOSURE_NATIVE_CLASSES
                )
            child_record, _descendants = await self._child_record(session_id, policy=policy)
            records.append(child_record)
            if _descendants is not None:
                descendant_pairs = _descendants
                descendant_ids = tuple(child_id for child_id, _ in _descendants)
            complete = source_complete and child_record.disposition not in {
                SessionClosureDisposition.UNSUPPORTED,
                SessionClosureDisposition.UNAVAILABLE,
                SessionClosureDisposition.TRUNCATED,
            }
        for store in self._dependent_stores:
            try:
                inspect_store = getattr(store, "inspect_session_closure", None)
                if inspect_store is None:
                    raise NotImplementedError
                record = self._copy_adapter_record(await inspect_store(session_id, policy=policy))
                if record.store_id != store.store_id:
                    raise ValueError("Closure adapter returned conflicting store authority.")
                records.append(record)
                if record.disposition in {
                    SessionClosureDisposition.UNSUPPORTED,
                    SessionClosureDisposition.UNAVAILABLE,
                    SessionClosureDisposition.TRUNCATED,
                }:
                    complete = False
            except NotImplementedError:
                records.append(
                    SessionClosureRecord(
                        store_id=store.store_id,
                        record_class="session_dependents",
                        disposition=SessionClosureDisposition.UNSUPPORTED,
                        capability="inspect_session_closure",
                    )
                )
                complete = False
            except Exception:
                records.append(
                    SessionClosureRecord(
                        store_id=store.store_id,
                        record_class="session_dependents",
                        disposition=SessionClosureDisposition.UNAVAILABLE,
                    )
                )
                complete = False
        return SessionClosureManifest(
            session_id=session_id,
            operation=SessionClosureOperation.INSPECT,
            plan_id=plan_id,
            generated_at=generated_at,
            complete=complete,
            records=tuple(records),
            metadata={
                "child_policy": policy.child_policy.value,
                "descendant_session_ids": list(descendant_ids),
                "descendant_dispositions": [
                    {
                        "session_id": child_id,
                        "parent_session_id": parent_id,
                        "disposition": (
                            "retained"
                            if policy.child_policy is SessionClosureChildPolicy.DETACH
                            else "owned_eligible"
                        ),
                    }
                    for child_id, parent_id in descendant_pairs
                ],
            },
        )

    async def erase(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy | None = None,
        expected_plan_id: str | None = None,
    ) -> SessionClosureReport:
        async with self._erase_lock:
            return await self._erase(
                session_id,
                policy=policy,
                expected_plan_id=expected_plan_id,
            )

    async def validate(self, session_id: str, *, policy: SessionClosurePolicy | None = None):
        """Run closure admission checks without mutating any owned resource."""
        policy = SessionClosurePolicy() if policy is None else policy
        inspection = await self.inspect(session_id, policy=policy)
        if not inspection.complete:
            return inspection
        session = await self._session_store.load(session_id)
        if session is None:
            return inspection
        await self._require_quiescent_session(session_id)
        await self._require_dependent_admission(session_id, policy)
        child_record = next(
            (record for record in inspection.records if record.record_class == "child_sessions"),
            None,
        )
        if child_record is not None and child_record.count:
            if policy.child_policy is SessionClosureChildPolicy.REJECT:
                raise ValueError("Session closure requires an explicit child-session policy.")
            descendants = await self._load_descendants(session_id, policy=policy)
            if descendants is None:
                raise ValueError("Session closure could not prove bounded descendant lineage.")
            if policy.child_policy is SessionClosureChildPolicy.DETACH:
                if not getattr(self._session_store, "supports_session_closure_detachment", False):
                    raise ValueError("Session store does not support durable child detachment.")
            else:
                for child_id, _parent_id in descendants:
                    await self._require_native_admission(child_id, policy)
                    await self._require_quiescent_session(child_id)
                    await self._require_dependent_admission(child_id, policy)
        await self._require_quiescent_session(session_id)
        return inspection

    async def _erase(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy | None = None,
        expected_plan_id: str | None = None,
    ) -> SessionClosureReport:
        """Erase dependents first and the session-store row last."""

        policy = SessionClosurePolicy() if policy is None else policy
        plan_id = self._plan_id(session_id, policy)
        if expected_plan_id is not None and expected_plan_id != plan_id:
            raise ValueError("Session closure plan identity conflict.")
        load_receipt = getattr(self._session_store, "load_session_closure_receipt", None)
        if load_receipt is not None:
            try:
                receipt = await load_receipt(session_id, plan_id)
            except NotImplementedError:
                receipt = None
            if receipt is not None:
                stored_report = SessionClosureReport.model_validate(receipt)
                if stored_report.plan_id != plan_id or stored_report.session_id != session_id:
                    raise ValueError("Session closure receipt identity conflict.")
                if await self._session_store.load(session_id) is not None:
                    raise ValueError("Session closure receipt conflicts with a live session.")
                return stored_report.model_copy(update={"already_absent": True})
        inspection = await self.inspect(session_id, policy=policy)
        if not inspection.complete:
            return SessionClosureReport(
                session_id=session_id,
                plan_id=plan_id,
                operation=SessionClosureOperation.ERASE,
                complete=False,
                already_absent=all(
                    record.disposition is SessionClosureDisposition.ABSENT
                    for record in inspection.records
                ),
                manifest=inspection,
                error="Closure inventory is incomplete; no destructive work was started.",
            )
        session = await self._session_store.load(session_id)
        if session is None:
            if plan_id in self._completed_plan_ids:
                return SessionClosureReport(
                    session_id=session_id,
                    plan_id=plan_id,
                    operation=SessionClosureOperation.ERASE,
                    complete=True,
                    already_absent=True,
                    manifest=inspection,
                )
            return SessionClosureReport(
                session_id=session_id,
                plan_id=plan_id,
                operation=SessionClosureOperation.ERASE,
                complete=False,
                already_absent=True,
                manifest=inspection,
                error="The session is absent without a durable closure receipt.",
            )
        await self._require_quiescent_session(session_id)
        await self._require_dependent_admission(session_id, policy)
        saved_progress = (
            await self._load_progress(session_id, plan_id)
            if policy.child_policy is SessionClosureChildPolicy.RECURSIVE
            else None
        )
        detached_edges: tuple[dict[str, Any], ...] = ()
        completed_descendants: tuple[tuple[str, str], ...] = ()
        if policy.child_policy is SessionClosureChildPolicy.DETACH:
            load_tombstones = getattr(self._session_store, "load_session_closure_tombstones", None)
            if load_tombstones is not None:
                detached_edges = await load_tombstones(session_id, plan_id)
        child_record = next(
            (record for record in inspection.records if record.record_class == "child_sessions"),
            None,
        )
        if (
            policy.child_policy is SessionClosureChildPolicy.REJECT
            and child_record is not None
            and child_record.count
        ):
            raise ValueError("Session closure requires an explicit child-session policy.")
        if saved_progress is None and (
            policy.child_policy is not SessionClosureChildPolicy.RECURSIVE
            or child_record is None
            or not child_record.count
        ):
            # A leaf/default closure can conflict with an ancestor's recursive
            # plan too. Own its target before any dependent-store call.
            await self._claim_progress(
                session_id,
                plan_id,
                policy,
                phase="recursive"
                if policy.child_policy is SessionClosureChildPolicy.RECURSIVE
                else "reject"
                if policy.child_policy is SessionClosureChildPolicy.REJECT
                else "target",
            )
            await self._claim_dependent_sets(session_id, policy, plan_id)
        if (
            (child_record is not None and child_record.count)
            or saved_progress is not None
            or detached_edges
        ):
            if policy.child_policy.value == "reject":
                raise ValueError("Session closure requires an explicit child-session policy.")
            descendants = await self._load_descendants(session_id, policy=policy)
            if descendants is None:
                return SessionClosureReport(
                    session_id=session_id,
                    plan_id=plan_id,
                    operation=SessionClosureOperation.ERASE,
                    complete=False,
                    manifest=inspection,
                    error="Closure could not prove bounded descendant lineage.",
                )
            if policy.child_policy is SessionClosureChildPolicy.DETACH:
                detach = getattr(self._session_store, "detach_session_children", None)
                if detach is None:
                    return SessionClosureReport(
                        session_id=session_id,
                        plan_id=plan_id,
                        operation=SessionClosureOperation.ERASE,
                        complete=False,
                        manifest=inspection,
                        error="Session store does not support durable child detachment.",
                    )
                direct_children = tuple(
                    child_id for child_id, parent_id in descendants if parent_id == session_id
                )
                if detached_edges:
                    if descendants:
                        raise ValueError("Child lineage changed after committed detachment.")
                    direct_children = tuple(item["child_session_id"] for item in detached_edges)
                detached_edges = await detach(
                    session_id,
                    direct_children,
                    closure_receipt={
                        "root_session_id": session_id,
                        "plan_id": plan_id,
                        "operation": "detach",
                    },
                )
            else:
                progress = saved_progress
                if progress is None:
                    for child_id, _parent_id in descendants:
                        await self._require_native_admission(child_id, policy)
                    progress = await self._claim_progress(
                        session_id, plan_id, policy, descendants=descendants, phase="recursive"
                    )
                try:
                    validated_progress = await self._validated_progress(
                        progress,
                        session_id=session_id,
                        plan_id=plan_id,
                        policy=policy,
                        descendants=descendants,
                    )
                except Exception:
                    return SessionClosureReport(
                        session_id=session_id,
                        plan_id=plan_id,
                        operation=SessionClosureOperation.ERASE,
                        complete=False,
                        manifest=inspection,
                        error="Recursive closure progress could not be validated.",
                    )
                completed = set(validated_progress.completed)
                progress["completed"] = sorted(completed)
                planned_descendants = tuple(
                    (item["session_id"], item["parent_session_id"])
                    for item in validated_progress.descendants
                )
                try:
                    for child_id, _parent_id in planned_descendants:
                        if child_id not in completed:
                            await self._require_quiescent_session(child_id)
                            await self._require_dependent_admission(child_id, policy)
                    await self._claim_dependent_sets(session_id, policy, plan_id)
                    for child_id, _parent_id in planned_descendants:
                        if child_id not in completed:
                            await self._claim_dependent_sets(
                                child_id, policy, session_closure_target_plan_id(plan_id, child_id)
                            )
                    for child_id, _parent_id in reversed(planned_descendants):
                        if child_id in completed:
                            continue
                        child_plan_id = session_closure_target_plan_id(plan_id, child_id)
                        load_child_receipt = getattr(
                            self._session_store, "load_session_closure_receipt", None
                        )
                        if load_child_receipt is not None:
                            child_receipt = await load_child_receipt(child_id, child_plan_id)
                            if child_receipt is not None:
                                if (
                                    child_receipt.get("root_session_id") != session_id
                                    or child_receipt.get("target_session_id") != child_id
                                    or child_receipt.get("original_parent_session_id") != _parent_id
                                    or child_receipt.get("plan_id") != child_plan_id
                                    or child_receipt.get("operation") != "recursive"
                                    or child_receipt.get("complete") is not True
                                ):
                                    raise ValueError("Recursive child receipt identity conflict.")
                                completed.add(child_id)
                                progress["completed"] = sorted(completed)
                                await self._save_progress(progress)
                                continue
                        await self._require_quiescent_session(child_id)
                        for store in self._dependent_stores:
                            outcome = await store.erase_session_closure(
                                child_id,
                                policy=policy,
                                plan_id=child_plan_id,
                            )
                            self._validated_settlement(store, outcome)
                        await self._session_store.delete_session(
                            child_id,
                            closure_receipt={
                                "root_session_id": session_id,
                                "target_session_id": child_id,
                                "original_parent_session_id": _parent_id,
                                "plan_id": child_plan_id,
                                "operation": "recursive",
                                "complete": True,
                            },
                        )
                        completed.add(child_id)
                        progress["completed"] = sorted(completed)
                        await self._save_progress(progress)
                    # Completion describes the whole admitted operation, including
                    # descendants authenticated from receipts on an earlier attempt.
                    completed_descendants = planned_descendants
                except Exception:
                    return SessionClosureReport(
                        session_id=session_id,
                        plan_id=plan_id,
                        operation=SessionClosureOperation.ERASE,
                        complete=False,
                        manifest=inspection,
                        error="Recursive closure stopped before durable completion.",
                    )
        load_stage = getattr(self._session_store, "load_active_model_completion_stage", None)
        if load_stage is not None:
            try:
                if await load_stage(session_id) is not None:
                    raise ValueError("Session closure requires no active model operation.")
            except NotImplementedError:
                pass
        records = list(inspection.records)
        try:
            for store in self._dependent_stores:
                erased_record = await store.erase_session_closure(
                    session_id,
                    policy=policy,
                    plan_id=plan_id,
                )
                for record_index, existing in enumerate(records):
                    if existing.store_id == store.store_id:
                        records[record_index] = self._validated_settlement(store, erased_record)
                        break
            completion_metadata = dict(inspection.metadata)
            if policy.child_policy is SessionClosureChildPolicy.RECURSIVE:
                completion_metadata.update(
                    descendant_session_ids=[child_id for child_id, _ in completed_descendants],
                    descendant_dispositions=[
                        {"session_id": child_id, "parent_session_id": parent_id}
                        for child_id, parent_id in completed_descendants
                    ],
                )
            erased = inspection.model_copy(
                update={
                    "operation": SessionClosureOperation.ERASE,
                    "metadata": {
                        **completion_metadata,
                        "detached_edges": list(detached_edges),
                        "descendant_dispositions": [
                            {
                                **item,
                                "disposition": "retained"
                                if policy.child_policy is SessionClosureChildPolicy.DETACH
                                else "erased",
                            }
                            for item in completion_metadata.get("descendant_dispositions", [])
                        ],
                    },
                    "records": tuple(
                        record.model_copy(
                            update={
                                "disposition": SessionClosureDisposition.RETAINED
                                if record.record_class == "child_sessions"
                                and policy.child_policy is SessionClosureChildPolicy.DETACH
                                else SessionClosureDisposition.ERASED,
                                **(
                                    {
                                        "count": len(completed_descendants)
                                        if policy.child_policy
                                        is SessionClosureChildPolicy.RECURSIVE
                                        else max(record.count, len(detached_edges))
                                    }
                                    if record.record_class == "child_sessions"
                                    else {}
                                ),
                            }
                        )
                        if record.store_id == "session-store"
                        or record.store_id in self._session_cascade_store_ids
                        else record
                        for record in records
                    ),
                }
            )
            await self._session_store.delete_session(
                session_id,
                closure_receipt=SessionClosureReport(
                    session_id=session_id,
                    plan_id=plan_id,
                    operation=SessionClosureOperation.ERASE,
                    complete=True,
                    manifest=erased,
                ).model_dump(mode="json"),
            )
        except Exception:
            partial = inspection.model_copy(
                update={
                    "operation": SessionClosureOperation.ERASE,
                    "records": tuple(records),
                    "complete": False,
                }
            )
            return SessionClosureReport(
                session_id=session_id,
                plan_id=plan_id,
                operation=SessionClosureOperation.ERASE,
                complete=False,
                manifest=partial,
                error="Closure stopped at an independent store boundary.",
            )
        if "erased" not in locals():
            return SessionClosureReport(
                session_id=session_id,
                plan_id=plan_id,
                operation=SessionClosureOperation.ERASE,
                complete=False,
                manifest=inspection,
                error="Session closure store did not publish a durable erase receipt.",
            )
        self._completed_plan_ids.add(plan_id)
        return SessionClosureReport(
            session_id=session_id,
            plan_id=plan_id,
            operation=SessionClosureOperation.ERASE,
            complete=True,
            manifest=erased,
        )

    async def export(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy | None = None,
        allow_partial: bool = False,
    ) -> SessionClosureExport:
        """Export bounded records; incomplete diagnostic exports require explicit opt-in."""

        if type(allow_partial) is not bool:
            raise TypeError("allow_partial must be a boolean.")
        policy = SessionClosurePolicy() if policy is None else policy
        manifest = await self.inspect(session_id, policy=policy)
        if not manifest.complete and not allow_partial:
            raise SessionClosureExportIncomplete(manifest)
        records: dict[str, Any] = {}
        records_bytes = 2  # The containing JSON object, before any entries.

        async def collect(
            key: str,
            store_id: str,
            record_class: str | None,
            capability: str,
            read: Callable[[], Awaitable[Any]],
            *,
            allow_absent: bool = False,
        ) -> None:
            nonlocal manifest, records_bytes
            if not manifest.complete and not allow_partial:
                raise SessionClosureExportIncomplete(manifest)
            failure = None
            try:
                value = await read()
                if value is None and not allow_absent:
                    raise ValueError("Closure export record is unavailable.")
                if value is not None:
                    if not isinstance(value, dict):
                        raise TypeError("Closure export record must be an object.")
                    if key in records:
                        raise ValueError("Closure export store identity conflicts.")
                    entry_overhead = (
                        len(canonical_durable_json_bytes(key, "session_closure.export.key"))
                        + 1
                        + bool(records)
                    )
                    remaining = policy.max_bytes - records_bytes - entry_overhead
                    if remaining <= 0:
                        raise ClosureRecordsTooLarge()
                    copied = copy_bounded_durable_json_value(
                        value,
                        "session_closure.export.store",
                        max_bytes=remaining,
                        max_nodes=1_000_000,
                    )
                    size = len(
                        canonical_durable_json_bytes(
                            copied,
                            "session_closure.export.store",
                            max_bytes=remaining,
                            max_nodes=1_000_000,
                        )
                    )
                    records[key] = copied
                    records_bytes += entry_overhead + size
            except NotImplementedError:
                failure = SessionClosureDisposition.UNSUPPORTED
            except ClosureRecordsTooLarge:
                failure = SessionClosureDisposition.TRUNCATED
            except DurableValueError as exc:
                failure = (
                    SessionClosureDisposition.TRUNCATED
                    if exc.code in {"json_value_too_large", "too_many_json_nodes"}
                    else SessionClosureDisposition.UNAVAILABLE
                )
            except Exception:
                # Extension/storage failures are represented by fixed diagnostics.
                # Cancellation and fatal BaseException signals remain authoritative.
                failure = SessionClosureDisposition.UNAVAILABLE
            if failure is not None:
                manifest = manifest.model_copy(
                    update={
                        "complete": False,
                        "records": tuple(
                            item.model_copy(
                                update={
                                    "disposition": failure,
                                    "capability": capability,
                                    "detail": "export capability did not provide complete records",
                                }
                            )
                            if item.store_id == store_id
                            and (record_class is None or item.record_class == record_class)
                            else item
                            for item in manifest.records
                        ),
                    }
                )
                # The manifest carries the failure even if no diagnostic entry
                # fits in the remaining record budget. Never overwrite an
                # already collected store's authoritative records.
                diagnostic = {"disposition": failure.value, "capability": capability}
                diagnostic_size = len(
                    canonical_durable_json_bytes(diagnostic, "session_closure.export.failure")
                )
                overhead = (
                    len(canonical_durable_json_bytes(key, "session_closure.export.key"))
                    + 1
                    + bool(records)
                )
                if (
                    key not in records
                    and records_bytes + overhead + diagnostic_size <= policy.max_bytes
                ):
                    records[key] = diagnostic
                    records_bytes += overhead + diagnostic_size

        if policy.child_policy is SessionClosureChildPolicy.DETACH:

            async def read_tombstones():
                load = getattr(self._session_store, "load_session_closure_tombstones", None)
                if load is None:
                    raise NotImplementedError
                return {
                    "disposition": SessionClosureDisposition.RETAINED.value,
                    "items": list(await load(session_id, manifest.plan_id)),
                }

            await collect(
                "session-store/lineage_tombstones",
                "session-store",
                "child_sessions",
                "session.load_session_closure_tombstones",
                read_tombstones,
            )
        if policy.child_policy is SessionClosureChildPolicy.RECURSIVE:

            async def read_progress():
                nonlocal manifest
                load = getattr(self._session_store, "load_session_closure_progress", None)
                if load is None:
                    raise NotImplementedError
                progress = await load(session_id, manifest.plan_id)
                if progress is None:
                    return None
                progress_model = SessionClosureProgress.model_validate(progress)
                if set(progress_model.completed) != {
                    item["session_id"] for item in progress_model.descendants
                }:
                    manifest = manifest.model_copy(update={"complete": False})
                return progress_model.model_dump(mode="json")

            await collect(
                "session-store/closure_progress",
                "session-store",
                "child_sessions",
                "session.load_session_closure_progress",
                read_progress,
                allow_absent=True,
            )

        async def read_snapshot():
            nonlocal manifest
            load = getattr(self._session_store, "load_session_closure_records", None)
            if load is None:
                raise NotImplementedError
            snapshot = validate_closure_records(
                await load(session_id, max_records=policy.max_records, max_bytes=policy.max_bytes),
                max_records=policy.max_records,
                max_bytes=policy.max_bytes,
            )
            if (
                snapshot["counts"]["session"] != 1
                or type(snapshot["records"]["session"][0]) is not dict
                or snapshot["records"]["session"][0].get("id") != session_id
            ):
                raise ValueError("Session export snapshot is unavailable.")
            manifest = manifest.model_copy(
                update={
                    "records": tuple(
                        record.model_copy(
                            update={
                                "count": snapshot["counts"][record.record_class],
                                "bytes": snapshot["record_bytes"][record.record_class],
                                "disposition": SessionClosureDisposition.OWNED_ELIGIBLE
                                if snapshot["counts"][record.record_class]
                                else SessionClosureDisposition.ABSENT,
                            }
                        )
                        if record.store_id == "session-store"
                        and record.record_class in SESSION_CLOSURE_NATIVE_CLASSES
                        else record
                        for record in manifest.records
                    )
                }
            )
            return snapshot

        await collect(
            "session-store/session",
            "session-store",
            None,
            "session.load_session_closure_records",
            read_snapshot,
        )
        for store in self._dependent_stores:

            async def read_dependent(store=store):
                exporter = getattr(store, "export_session_closure", None)
                if exporter is None:
                    raise NotImplementedError
                result = await exporter(session_id, policy=policy)
                if not isinstance(result, dict):
                    raise TypeError("Closure exporter must return an object.")
                return result

            await collect(
                store.store_id, store.store_id, None, "export_session_closure", read_dependent
            )
        if not manifest.complete and not allow_partial:
            raise SessionClosureExportIncomplete(manifest)
        export = SessionClosureExport(
            manifest=manifest.model_copy(update={"operation": SessionClosureOperation.EXPORT}),
            session_records=records,
        )
        export._max_bytes = policy.max_bytes
        export.to_bytes()
        return export


def session_closure_plan_id(
    session_id: str,
    *,
    policy: SessionClosurePolicy,
    store_ids: tuple[str, ...],
) -> str:
    """Derive the exact idempotency identity for one closure plan."""

    material = {
        "schema_version": SESSION_CLOSURE_SCHEMA_VERSION,
        "session_id": session_id,
        "policy": policy.model_dump(mode="json"),
        "store_ids": list(store_ids),
    }
    return sha256(canonical_durable_json_bytes(material, "session_closure.plan")).hexdigest()


def session_closure_target_plan_id(root_plan_id: str, target_session_id: str) -> str:
    """Derive a stable per-descendant receipt identity from the root plan."""

    return sha256(
        canonical_durable_json_bytes(
            {"root_plan_id": root_plan_id, "target_session_id": target_session_id},
            "session_closure.target_plan",
        )
    ).hexdigest()


def bounded_session_closure_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Detach caller metadata under the standard metadata ceiling."""

    return copy_durable_metadata(value, "session_closure.metadata")


def _export_identity(value: object) -> str:
    """Expose stable correlation without exporting caller-controlled text."""

    if not isinstance(value, str):
        value = str(value)
    return sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


__all__ = [
    "SESSION_CLOSURE_DEFAULT_MAX_BYTES",
    "SESSION_CLOSURE_DEFAULT_MAX_RECORDS",
    "SESSION_CLOSURE_SCHEMA_VERSION",
    "ArtifactSessionClosureStore",
    "BudgetSessionClosureStore",
    "RetainedSessionClosureStore",
    "SessionClosureBudgetDisposition",
    "SessionClosureChildPolicy",
    "SessionClosureCoordinator",
    "SessionClosureDisposition",
    "SessionClosureExport",
    "SessionClosureExportIncomplete",
    "SessionClosureLineageStore",
    "SessionClosureManifest",
    "SessionClosureOperation",
    "SessionClosurePolicy",
    "SessionClosureProgress",
    "SessionClosureRecord",
    "SessionClosureReport",
    "SessionClosureStore",
    "SessionEvidenceClosureStore",
    "SharedSessionClosureStore",
    "UnsupportedSessionClosureStore",
    "bounded_session_closure_metadata",
    "session_closure_plan_id",
    "session_closure_target_plan_id",
]
