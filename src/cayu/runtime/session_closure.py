"""Bounded, explicit session-closure inspection and erasure contracts.

Closure is deliberately separate from ``SessionStore.delete_session``.  A
session store owns its own cascade, while tasks, artifacts, budgets, knowledge,
and application stores have independent ownership and retention rules.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StrictInt,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_bounded_durable_json_value,
    copy_durable_metadata,
)
from cayu.artifacts import ArtifactScope
from cayu.memory_evidence import RecallEvidenceQuery
from cayu.runtime.sessions import SessionQuery
from cayu.runtime.tasks import TaskQuery, TaskStatus

SESSION_CLOSURE_SCHEMA_VERSION = 1
SESSION_CLOSURE_DEFAULT_MAX_RECORDS = 10_000
SESSION_CLOSURE_DEFAULT_MAX_BYTES = 16 * 1024 * 1024
SESSION_CLOSURE_MAX_FIELD_CHARS = 4096


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


class SessionClosureBudgetDisposition(StrEnum):
    RETAIN = "retain"


class SessionClosurePolicy(BaseModel):
    """Bounded caller policy; it never silently expands store enumeration."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    child_policy: SessionClosureChildPolicy = SessionClosureChildPolicy.REJECT
    budget_disposition: SessionClosureBudgetDisposition = SessionClosureBudgetDisposition.RETAIN
    include_artifact_metadata: bool = True
    max_records: StrictInt = Field(default=SESSION_CLOSURE_DEFAULT_MAX_RECORDS, ge=1, le=100_000)
    max_bytes: StrictInt = Field(
        default=SESSION_CLOSURE_DEFAULT_MAX_BYTES,
        ge=1,
        le=256 * 1024 * 1024,
    )

    @model_validator(mode="after")
    def validate_policy(self) -> SessionClosurePolicy:
        return self


class SessionClosureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    store_id: str = Field(max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    record_class: str = Field(max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    disposition: SessionClosureDisposition
    count: StrictInt = Field(default=0, ge=0)
    bytes: StrictInt = Field(default=0, ge=0)
    detail: str | None = Field(default=None, max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)
    capability: str | None = Field(default=None, max_length=SESSION_CLOSURE_MAX_FIELD_CHARS)


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


class SessionClosureExport(BaseModel):
    """Bounded, content-free export envelope for closure inspection."""

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
            max_bytes=SESSION_CLOSURE_DEFAULT_MAX_BYTES,
            max_nodes=1_000_000,
        )

    def to_bytes(self) -> bytes:
        value = self.model_dump(mode="json", warnings=False)
        encoded = canonical_durable_json_bytes(value, "session_closure.export")
        if len(encoded) > self._max_bytes:
            raise ValueError("Session closure export exceeds the bounded byte limit.")
        return encoded


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


class SharedSessionClosureStore(RetainedSessionClosureStore):
    """Reference-only owner whose records are never session-owned."""

    def _record(self) -> SessionClosureRecord:
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class=self.record_class,
            disposition=SessionClosureDisposition.SHARED,
            detail="shared or independently retained; not deleted by session closure",
        )


class SessionEvidenceClosureStore:
    """Inventory #946 receipt/exposure references owned by SessionStore."""

    store_id = "session-store-evidence"

    def __init__(self, store: Any) -> None:
        self._store = store

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        list_receipts = getattr(self._store, "list_recall_receipts", None)
        list_exposures = getattr(self._store, "list_context_exposures", None)
        if list_receipts is None or list_exposures is None:
            raise NotImplementedError("Session evidence enumeration is unavailable.")
        limit = min(policy.max_records, 100)
        query = RecallEvidenceQuery(
            session_id=session_id,
            limit=limit,
            max_bytes=min(policy.max_bytes, 1_000_000),
        )
        receipts = await list_receipts(query)
        exposures = await list_exposures(query)
        receipt_count = len(receipts.items)
        exposure_count = len(exposures.items)
        truncated = (
            receipts.truncated
            or exposures.truncated
            or receipts.next_cursor is not None
            or exposures.next_cursor is not None
        )
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="knowledge_evidence_projection_receipts_exposures",
            disposition=(
                SessionClosureDisposition.TRUNCATED
                if truncated
                else SessionClosureDisposition.OWNED_ELIGIBLE
                if receipt_count or exposure_count
                else SessionClosureDisposition.ABSENT
            ),
            count=receipt_count + exposure_count,
            detail=(
                "session-owned evidence is deleted by the final SessionStore cascade"
                if receipt_count or exposure_count
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
            record_class="knowledge_evidence_projection_receipts_exposures",
            disposition=SessionClosureDisposition.ERASED,
            detail="settled by the final SessionStore deletion",
            capability="session-store.cascade",
        )

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        list_receipts = getattr(self._store, "list_recall_receipts", None)
        list_exposures = getattr(self._store, "list_context_exposures", None)
        if list_receipts is None or list_exposures is None:
            raise NotImplementedError("Session evidence enumeration is unavailable.")
        query = RecallEvidenceQuery(
            session_id=session_id,
            limit=min(policy.max_records, 100),
            max_bytes=min(policy.max_bytes, 1_000_000),
        )
        receipts = await list_receipts(query)
        exposures = await list_exposures(query)
        return {
            "recall_receipts": [
                {"id_digest": _export_identity(getattr(item, "id", ""))} for item in receipts.items
            ],
            "context_exposures": [
                {"id_digest": _export_identity(getattr(item, "id", ""))} for item in exposures.items
            ],
        }


class ArtifactSessionClosureStore:
    """Bounded adapter for one registered session-scoped artifact store."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self.store_id = f"artifact-store:{store.id}"

    async def inspect_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        result = await self._store.list(
            scope=ArtifactScope.SESSION,
            session_id=session_id,
            limit=policy.max_records,
        )
        artifacts = tuple(result.artifacts)
        truncated = bool(result.truncated) or (
            result.total_count is not None and result.total_count > len(artifacts)
        )
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
        result = await self._store.list(
            scope=ArtifactScope.SESSION,
            session_id=session_id,
            limit=policy.max_records,
        )
        artifacts = tuple(result.artifacts)
        if bool(result.truncated) or (
            result.total_count is not None and result.total_count > len(artifacts)
        ):
            raise ValueError("Artifact closure inventory is truncated.")
        for artifact in artifacts:
            await self._store.delete(artifact.id)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="session_artifacts",
            disposition=SessionClosureDisposition.ERASED,
            count=len(artifacts),
            bytes=sum(item.size_bytes for item in artifacts),
            capability="artifact.list/delete",
        )

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        result = await self._store.list(
            scope=ArtifactScope.SESSION, session_id=session_id, limit=policy.max_records
        )
        return {
            "artifacts": [
                {
                    "id_digest": _export_identity(artifact.id),
                    "size_bytes": artifact.size_bytes,
                    "scope": getattr(artifact.scope, "value", artifact.scope),
                }
                for artifact in result.artifacts
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
            for task in visible
        )
        if visible and not getattr(self._store, "supports_session_closure_deletion", False):
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
        )

    async def erase_session_closure(
        self,
        session_id: str,
        *,
        policy: SessionClosurePolicy,
        plan_id: str,
    ):
        tasks = await self._tasks(session_id, policy)
        if len(tasks) > policy.max_records:
            raise ValueError("Task closure inventory is truncated.")
        if any(
            task.status not in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}
            for task in tasks
        ):
            raise ValueError("Task closure has active task ownership.")
        delete = getattr(self._store, "delete_session_tasks", None)
        if delete is None:
            raise NotImplementedError("Task store does not support session task deletion.")
        await delete(session_id, task_ids=tuple(task.id for task in tasks), policy=policy)
        return SessionClosureRecord(
            store_id=self.store_id,
            record_class="session_tasks",
            disposition=SessionClosureDisposition.ERASED,
            count=len(tasks),
            capability="task.list/delete",
        )

    async def export_session_closure(self, session_id: str, *, policy: SessionClosurePolicy):
        tasks = await self._tasks(session_id, policy)
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
    ) -> None:
        self._session_store = session_store
        self._dependent_stores = tuple(dependent_stores)
        self._clock = clock
        self._completed_plan_ids: set[str] = set()
        self._erase_lock = asyncio.Lock()

    def _plan_id(self, session_id: str, policy: SessionClosurePolicy) -> str:
        store_ids = ("session-store", *(store.store_id for store in self._dependent_stores))
        return session_closure_plan_id(session_id, policy=policy, store_ids=store_ids)

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
        if session is None:
            records.append(
                SessionClosureRecord(
                    store_id="session-store",
                    record_class="session",
                    disposition=SessionClosureDisposition.ABSENT,
                )
            )
            complete = True
            list_sessions = getattr(self._session_store, "list_sessions", None)
            if list_sessions is None:
                records.append(
                    SessionClosureRecord(
                        store_id="session-store",
                        record_class="child_sessions",
                        disposition=SessionClosureDisposition.UNSUPPORTED,
                        capability="session.list_sessions",
                    )
                )
                complete = False
            else:
                children = []
                while len(children) <= policy.max_records:
                    page_size = min(1000, policy.max_records + 1 - len(children))
                    page = await list_sessions(
                        SessionQuery(
                            parent_session_id=session_id,
                            limit=page_size,
                            offset=len(children),
                        )
                    )
                    page_sessions = list(getattr(page, "sessions", page))
                    children.extend(page_sessions)
                    if len(page_sessions) < page_size:
                        break
                truncated = len(children) > policy.max_records
                records.append(
                    SessionClosureRecord(
                        store_id="session-store",
                        record_class="child_sessions",
                        disposition=(
                            SessionClosureDisposition.TRUNCATED
                            if truncated
                            else SessionClosureDisposition.RETAINED
                            if children
                            else SessionClosureDisposition.ABSENT
                        ),
                        count=min(len(children), policy.max_records),
                        detail=(
                            "child policy must explicitly handle descendants" if children else None
                        ),
                        capability="session.list_sessions",
                    )
                )
                if truncated:
                    complete = False
        else:
            records.extend(
                SessionClosureRecord(
                    store_id="session-store",
                    record_class=record_class,
                    disposition=SessionClosureDisposition.OWNED_ELIGIBLE,
                    count=1,
                    detail="owned by the configured SessionStore cascade",
                )
                for record_class in (
                    "session",
                    "labels",
                    "metadata",
                    "events",
                    "transcript",
                    "checkpoint",
                    "queued_messages",
                    "session_operations",
                    "event_side_effect_deliveries",
                )
            )
            complete = True
            list_sessions = getattr(self._session_store, "list_sessions", None)
            if list_sessions is None:
                records.append(
                    SessionClosureRecord(
                        store_id="session-store",
                        record_class="child_sessions",
                        disposition=SessionClosureDisposition.UNSUPPORTED,
                        capability="session.list_sessions",
                    )
                )
                complete = False
            else:
                children = []
                while len(children) <= policy.max_records:
                    page_size = min(1000, policy.max_records + 1 - len(children))
                    page = await list_sessions(
                        SessionQuery(
                            parent_session_id=session_id, limit=page_size, offset=len(children)
                        )
                    )
                    page_sessions = list(getattr(page, "sessions", page))
                    children.extend(page_sessions)
                    if len(page_sessions) < page_size:
                        break
                truncated = len(children) > policy.max_records
                records.append(
                    SessionClosureRecord(
                        store_id="session-store",
                        record_class="child_sessions",
                        disposition=(
                            SessionClosureDisposition.TRUNCATED
                            if truncated
                            else SessionClosureDisposition.RETAINED
                            if children
                            else SessionClosureDisposition.ABSENT
                        ),
                        count=min(len(children), policy.max_records),
                        detail=(
                            "child policy must explicitly handle descendants" if children else None
                        ),
                        capability="session.list_sessions",
                    )
                )
                if truncated:
                    complete = False
        for store in self._dependent_stores:
            try:
                inspect_store = getattr(store, "inspect_session_closure", None)
                if inspect_store is None:
                    raise NotImplementedError
                record = SessionClosureRecord.model_validate(
                    await inspect_store(session_id, policy=policy)
                )
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
        if session.status.value == "running":
            raise ValueError("Session closure requires a terminal session.")
        load_checkpoint = getattr(self._session_store, "load_checkpoint", None)
        if load_checkpoint is not None:
            checkpoint = await load_checkpoint(session_id)
            operations = (
                checkpoint.get("session_operations") if isinstance(checkpoint, dict) else None
            )
            if isinstance(operations, dict) and operations.get("active_operation_id") is not None:
                raise ValueError(
                    "Cannot delete a session while durable operation "
                    f"{operations['active_operation_id']} is active: {session_id}"
                )
        child_record = next(
            (record for record in inspection.records if record.record_class == "child_sessions"),
            None,
        )
        if child_record is not None and child_record.count:
            if policy.child_policy is SessionClosureChildPolicy.REJECT:
                raise ValueError("Session closure requires an explicit child-session policy.")
            raise NotImplementedError("Child detach and recursive closure are not implemented.")
        load_stage = getattr(self._session_store, "load_active_model_completion_stage", None)
        if load_stage is not None:
            try:
                if await load_stage(session_id) is not None:
                    raise ValueError("Session closure requires no active model operation.")
            except NotImplementedError:
                pass
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
        if session.status.value == "running":
            raise ValueError("Session closure requires a terminal session.")
        load_checkpoint = getattr(self._session_store, "load_checkpoint", None)
        if load_checkpoint is not None:
            checkpoint = await load_checkpoint(session_id)
            operations = (
                checkpoint.get("session_operations") if isinstance(checkpoint, dict) else None
            )
            if isinstance(operations, dict) and operations.get("active_operation_id") is not None:
                raise ValueError(
                    "Cannot delete a session while durable operation "
                    f"{operations['active_operation_id']} is active: {session_id}"
                )
        child_record = next(
            (record for record in inspection.records if record.record_class == "child_sessions"),
            None,
        )
        if child_record is not None and child_record.count:
            if policy.child_policy.value == "reject":
                raise ValueError("Session closure requires an explicit child-session policy.")
            raise NotImplementedError(
                "This closure coordinator does not yet implement child detachment or recursion."
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
                        records[record_index] = SessionClosureRecord.model_validate(erased_record)
                        break
            erased = inspection.model_copy(
                update={
                    "operation": SessionClosureOperation.ERASE,
                    "records": tuple(
                        record.model_copy(update={"disposition": SessionClosureDisposition.ERASED})
                        if record.store_id == "session-store"
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
    ) -> SessionClosureExport:
        """Export only the bounded closure manifest, never session contents."""

        policy = SessionClosurePolicy() if policy is None else policy
        manifest = await self.inspect(session_id, policy=policy)
        records: dict[str, Any] = {}
        load_snapshot = getattr(self._session_store, "load_session_export_snapshot", None)
        if manifest.complete and load_snapshot is not None:
            from cayu.runtime.exports import SessionExportLimits

            snapshot = await load_snapshot(
                session_id,
                limits=SessionExportLimits(
                    max_bytes=policy.max_bytes,
                    max_record_bytes=min(policy.max_bytes, 8 * 1024 * 1024),
                ),
            )
            if snapshot is None:
                manifest = manifest.model_copy(update={"complete": False})
            else:
                records = {"session-store/session": snapshot.document()}
        elif manifest.complete:
            manifest = manifest.model_copy(update={"complete": False})
        export_inventory_complete = manifest.complete
        for store in self._dependent_stores:
            exporter = getattr(store, "export_session_closure", None)
            if exporter is None:
                manifest = manifest.model_copy(update={"complete": False})
                records[f"{store.store_id}/export"] = {
                    "disposition": SessionClosureDisposition.UNSUPPORTED.value,
                    "capability": "export_session_closure",
                }
            elif export_inventory_complete:
                try:
                    exported = await exporter(session_id, policy=policy)
                    records[store.store_id] = copy_bounded_durable_json_value(
                        exported,
                        "session_closure.export.store",
                        max_bytes=policy.max_bytes,
                        max_nodes=1_000_000,
                    )
                except NotImplementedError:
                    manifest = manifest.model_copy(update={"complete": False})
                    records[f"{store.store_id}/export"] = {
                        "disposition": SessionClosureDisposition.UNSUPPORTED.value,
                        "capability": "export_session_closure",
                    }
                except Exception:
                    manifest = manifest.model_copy(update={"complete": False})
                    records[f"{store.store_id}/export"] = {
                        "disposition": SessionClosureDisposition.UNAVAILABLE.value,
                        "capability": "export_session_closure",
                    }
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
    "RetainedSessionClosureStore",
    "SessionClosureBudgetDisposition",
    "SessionClosureChildPolicy",
    "SessionClosureCoordinator",
    "SessionClosureDisposition",
    "SessionClosureExport",
    "SessionClosureManifest",
    "SessionClosureOperation",
    "SessionClosurePolicy",
    "SessionClosureRecord",
    "SessionClosureReport",
    "SessionClosureStore",
    "SessionEvidenceClosureStore",
    "SharedSessionClosureStore",
    "UnsupportedSessionClosureStore",
    "bounded_session_closure_metadata",
    "session_closure_plan_id",
]
