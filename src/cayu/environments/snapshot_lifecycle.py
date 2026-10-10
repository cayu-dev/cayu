"""Durable, fenced execution snapshot capture, restoration and retirement.

The ArtifactStore is deliberately separate from model-visible environment
artifacts. Memory dumps can contain secrets. Do not register this store with
artifact tools or the server's general artifact download inventory.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, Literal

from cayu._validation import canonical_durable_json_bytes, require_durable_clean_nonblank
from cayu.artifacts import ArtifactReadResult, ArtifactScope, ArtifactStore
from cayu.environments.snapshots import (
    CapturedExecutionSnapshot,
    ExecutionSnapshotAdapter,
    ExecutionSnapshotArtifact,
    ExecutionSnapshotConflict,
    ExecutionSnapshotError,
    ExecutionSnapshotFidelity,
    ExecutionSnapshotOperation,
    ExecutionSnapshotOutcomeUnknown,
    ExecutionSnapshotPolicy,
    ExecutionSnapshotPosition,
    ExecutionSnapshotRecord,
)
from cayu.events import Event, EventType
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.sessions._checkpoint_preservation import _execution_snapshot_authority_mutation_scope
from cayu.sessions.base import Session, SessionStatus, SessionStore
from cayu.sessions.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
    decode_runtime_checkpoint,
)

EXECUTION_SNAPSHOTS_KEY = "execution_snapshots"
# A continuation rejected by the snapshot gate may leave the session failed.
# It still has no live run owner, and recovery retains the exact epoch fence.
_IDLE = {
    SessionStatus.PENDING,
    SessionStatus.INTERRUPTED,
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
}
_PENDING_SNAPSHOT_TASKS: set[asyncio.Task] = set()


def _bounded_operation(method):
    @wraps(method)
    async def observe(self, session_id, environment_name, *args, **kwargs):
        task = asyncio.create_task(method(self, session_id, environment_name, *args, **kwargs))
        _PENDING_SNAPSHOT_TASKS.add(task)

        def settled(done):
            _PENDING_SNAPSHOT_TASKS.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(settled)
        try:
            completed, _ = await asyncio.wait({task}, timeout=self.policy.timeout_seconds)
        except asyncio.CancelledError:
            task.cancel()
            await self._revoke_request(session_id, environment_name, kwargs)
            raise
        if not completed:
            task.cancel()
            await self._revoke_request(session_id, environment_name, kwargs)
            raise ExecutionSnapshotOutcomeUnknown(
                "Snapshot deadline expired; inspect durable operation state."
            )
        return task.result()

    return observe


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_durable_json_bytes(value, "execution_snapshot")).hexdigest()


def _position_digest(checkpoint: dict[str, Any] | None) -> str:
    return _digest(
        {
            k: v
            for k, v in (checkpoint or {}).items()
            if k not in {EXECUTION_SNAPSHOTS_KEY, CHECKPOINT_SCHEMA_VERSION_KEY}
        }
    )


def _registry(checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    raw = (checkpoint or {}).get(EXECUTION_SNAPSHOTS_KEY, {})
    if type(raw) is not dict or set(raw) - {"environments"}:
        raise ExecutionSnapshotError("Invalid execution snapshot registry.")
    environments = raw.get("environments", {})
    if type(environments) is not dict:
        raise ExecutionSnapshotError("Invalid execution snapshot environment registry.")
    return {"environments": dict(environments)}


def _environment(registry: dict[str, Any], name: str) -> dict[str, Any]:
    raw = registry["environments"].get(name)
    if raw is None:
        return {"binding": None, "operations": {}, "snapshots": {}}
    if type(raw) is not dict or set(raw) != {"binding", "operations", "snapshots"}:
        raise ExecutionSnapshotError("Invalid execution snapshot environment state.")
    if type(raw["operations"]) is not dict or type(raw["snapshots"]) is not dict:
        raise ExecutionSnapshotError("Invalid execution snapshot records.")
    binding = raw["binding"]
    if binding is not None:
        if type(binding) is not dict or set(binding) != {"generation", "allocation_sha256"}:
            raise ExecutionSnapshotError("Invalid execution snapshot binding.")
        require_durable_clean_nonblank(binding["generation"], "snapshot binding generation")
        if len(binding["generation"].encode()) > 512 or (
            type(binding["allocation_sha256"]) is not str
            or len(binding["allocation_sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in binding["allocation_sha256"]
            )
        ):
            raise ExecutionSnapshotError("Invalid execution snapshot binding identity.")
    for key, value in raw["operations"].items():
        if ExecutionSnapshotOperation.model_validate(value).id != key:
            raise ExecutionSnapshotError("Execution snapshot operation identity mismatch.")
    for key, value in raw["snapshots"].items():
        if ExecutionSnapshotRecord.model_validate(value).id != key:
            raise ExecutionSnapshotError("Execution snapshot identity mismatch.")
    return {
        "binding": raw["binding"],
        "operations": dict(raw["operations"]),
        "snapshots": dict(raw["snapshots"]),
    }


async def _projected_registry(store: SessionStore, session_id: str) -> dict[str, Any]:
    # Snapshot metadata only; never loads or decodes the full controller checkpoint.
    return _registry(
        decode_runtime_checkpoint(
            await store.load_execution_snapshot_checkpoint(session_id), session_id=session_id
        )
    )


def _compact(records: dict[str, Any], limit: int, evictable, order) -> None:
    # Make room for one more record by dropping the oldest evictable tombstones.
    excess = len(records) - limit + 1
    if excess > 0:
        for value in sorted(filter(evictable, records.values()), key=order)[:excess]:
            del records[value["id"]]


def _manifest(record: ExecutionSnapshotRecord) -> str:
    return _digest(record.model_dump(mode="json", exclude={"manifest_sha256", "retention"}))


def _unsubmitted(op: ExecutionSnapshotOperation) -> bool:
    # Submission records the execution position. Recovery claims preserve it,
    # including its absence when no submission has been recorded.
    return op.state == "intent" and op.position is None


def _ordered_operations(state):
    # JSONB stores do not preserve object insertion order. Unknown work must
    # remain visible even when a bounded page omits older terminal records.
    return sorted(
        state["operations"].values(),
        key=lambda value: (
            value["state"] != "succeeded",
            value.get("created_at") or "",
            value["id"],
        ),
    )


class ExecutionSnapshots:
    """Explicit operator API for one session's restorable execution state.

    Capture/restore require an idle session and an exact expected run epoch and
    binding generation. Restore targets a separately provisioned, empty adapter;
    allocation creation/disposal remains owned by its runner/factory. Existing
    controller state is retained, and must match the captured execution position.
    """

    def __init__(
        self,
        session_store: SessionStore,
        snapshot_store: ArtifactStore,
        *,
        policy: ExecutionSnapshotPolicy | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(snapshot_store, ArtifactStore) or not snapshot_store.supports_pins:
            raise ExecutionSnapshotError("Execution snapshots require durable artifact pins.")
        self._store = runtime_checkpoint_session_store(session_store)
        self._artifacts = snapshot_store
        self.policy = ExecutionSnapshotPolicy.model_validate(
            (policy or ExecutionSnapshotPolicy()).model_dump()
        )
        self._clock = clock or (lambda: datetime.now(UTC))

    async def _revoke_request(self, session_id, name, request):
        # Revoke a cancellation-opaque adapter's publication authority. Keep
        # its task alive until settlement without blocking the caller forever.
        with suppress(Exception):
            session = await self._session(session_id, request["expected_run_epoch"])
            operation_id = (
                "esop_" + _digest([session.instance_id, name, request["idempotency_key"]])[:32]
            )
            for _ in range(2):
                state = _environment(await _projected_registry(self._store, session_id), name)
                raw = state["operations"].get(operation_id)
                if raw is None:
                    return
                op = ExecutionSnapshotOperation.model_validate(raw)
                if op.kind == "delete" or op.state not in {"intent", "submitted"}:
                    return
                if not _unsubmitted(op):
                    await self._unknown(session, name, op)
                    return
                if await self._abandon(session, name, op):
                    return
                # It was submitted concurrently; settle it as unknown instead.

    async def inspect(self, session_id: str, environment_name: str) -> dict[str, Any]:
        """Return bounded metadata only; no bytes or provider-private handles."""
        state = _environment(
            _registry(await self._store.load_checkpoint(session_id)), environment_name
        )
        return {
            "binding": state["binding"],
            "operations": _ordered_operations(state)[-100:],
            "snapshots": [
                {
                    k: v
                    for k, v in record.items()
                    if k not in {"artifacts", "pin_owner", "session_instance_id"}
                }
                for record in sorted(
                    state["snapshots"].values(),
                    key=lambda value: (value["created_at"], value["id"]),
                )[-100:]
            ],
            "truncated": len(state["operations"]) > 100 or len(state["snapshots"]) > 100,
        }

    async def _session(self, session_id: str, run_epoch: int) -> Session:
        if type(run_epoch) is not int or run_epoch < 0:
            raise ValueError("Expected run epoch must be a nonnegative integer.")
        session = await self._store.load(session_id)
        if session is None:
            raise KeyError(session_id)
        if session.run_epoch != run_epoch or session.status not in _IDLE:
            raise ExecutionSnapshotConflict("Snapshot operations require the exact idle run epoch.")
        return session

    async def _position(self, session_id: str) -> ExecutionSnapshotPosition:
        return ExecutionSnapshotPosition(
            checkpoint_sha256=_position_digest(await self._store.load_checkpoint(session_id)),
            transcript_cursor=await self._store.load_transcript_cursor(session_id),
        )

    async def _change(
        self,
        session: Session,
        name: str,
        transform,
        *,
        event: Event | None = None,
        position: ExecutionSnapshotPosition | None = None,
    ) -> None:
        def update(current: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
            if current.instance_id != session.instance_id:
                raise ExecutionSnapshotConflict("Session incarnation changed.")
            if position is not None and _position_digest(checkpoint) != position.checkpoint_sha256:
                raise ExecutionSnapshotConflict(
                    "Execution position changed; reconciliation required."
                )
            registry = _registry(checkpoint)
            state = _environment(registry, name)
            transform(state)
            registry["environments"][name] = state
            updated = dict(checkpoint or {})
            updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
            updated[EXECUTION_SNAPSHOTS_KEY] = registry
            return updated

        with _execution_snapshot_authority_mutation_scope():
            await self._store.publish_checkpoint_and_events(
                session.id,
                checkpoint_transform=update,
                events=[] if event is None else [event],
                expected_statuses=_IDLE,
                expected_run_epoch=session.run_epoch,
                expected_transcript_cursor=None if position is None else position.transcript_cursor,
            )

    async def _reserve(
        self,
        session: Session,
        name: str,
        key: str,
        generation: str,
        adapter: ExecutionSnapshotAdapter,
        kind: Literal["capture", "restore", "delete"],
        snapshot_id: str | None = None,
        target_generation: str | None = None,
    ) -> ExecutionSnapshotOperation:
        name = require_durable_clean_nonblank(name, "environment_name")
        key = require_durable_clean_nonblank(key, "idempotency_key")
        generation = require_durable_clean_nonblank(generation, "binding_generation")
        if any(len(value.encode()) > 512 for value in (name, key, generation)):
            raise ValueError("Snapshot request identity exceeds its bound.")
        if adapter.capability.fidelity is ExecutionSnapshotFidelity.UNSUPPORTED:
            raise ExecutionSnapshotError("Environment does not support execution snapshots.")
        identity = _digest([session.instance_id, name, key])[:32]
        op = ExecutionSnapshotOperation(
            id="esop_" + identity,
            kind=kind,
            state="intent",
            request_sha256=_digest(
                [
                    kind,
                    generation,
                    snapshot_id,
                    target_generation,
                    adapter.allocation_sha256,
                    adapter.capability.model_dump(mode="json"),
                ]
            ),
            run_epoch=session.run_epoch,
            binding_generation=generation,
            snapshot_id=snapshot_id or "esnap_" + identity,
            target_generation=target_generation,
            allocation_sha256=adapter.allocation_sha256,
            created_at=self._clock(),
        )
        existing = None

        def reserve(state):
            nonlocal existing
            raw = state["operations"].get(op.id)
            if raw is not None:
                existing = ExecutionSnapshotOperation.model_validate(raw)
                if existing.request_sha256 != op.request_sha256:
                    raise ExecutionSnapshotConflict("Idempotency key was used for another request.")
                return
            if any(value["state"] != "succeeded" for value in state["operations"].values()):
                raise ExecutionSnapshotOutcomeUnknown(
                    "Another snapshot operation requires settlement."
                )
            if kind == "capture":
                if (
                    sum(value["retention"] != "deleted" for value in state["snapshots"].values())
                    >= self.policy.max_records
                ):
                    raise ExecutionSnapshotError(
                        "Snapshot retention limit reached; delete retained snapshots."
                    )
                _compact(
                    state["snapshots"],
                    self.policy.max_records,
                    lambda value: value["retention"] == "deleted",
                    lambda value: (value["created_at"], value["id"]),
                )
            self._compact_operations(state)
            binding = state["binding"]
            if binding is None:
                if kind != "capture":
                    raise ExecutionSnapshotConflict("No source binding is recorded.")
                state["binding"] = {
                    "generation": generation,
                    "allocation_sha256": adapter.allocation_sha256,
                }
            elif binding["generation"] != generation:
                raise ExecutionSnapshotConflict("Expected environment generation changed.")
            elif kind == "capture" and binding["allocation_sha256"] != adapter.allocation_sha256:
                raise ExecutionSnapshotConflict("Capture allocation differs from its binding.")
            state["operations"][op.id] = op.model_dump(mode="json")

        await self._change(session, name, reserve)
        return existing or op

    def _compact_operations(self, state) -> None:
        # Succeeded operations only answer idempotent replays. Keep the newest
        # within max_records so a new operation, including deletion, always fits.
        _compact(
            state["operations"],
            self.policy.max_records,
            lambda value: value["state"] == "succeeded",
            lambda value: (value.get("created_at") or "", value["id"]),
        )

    async def _phase(self, session, name, op, state_name, *, transform=None, position=None):
        replacement = op.model_copy(
            update={"state": state_name, "position": op.position or position}
        )

        def change(state):
            if state["operations"].get(op.id) != op.model_dump(mode="json"):
                raise ExecutionSnapshotConflict("Operation predecessor changed.")
            if state["binding"]["generation"] != op.binding_generation and op.state != "verified":
                raise ExecutionSnapshotConflict("Operation binding changed.")
            if transform:
                transform(state)
            state["operations"][op.id] = replacement.model_dump(mode="json")

        event = Event(
            id="evt_es_" + _digest([op.id, op.run_epoch, state_name]),
            type=EventType.EXECUTION_SNAPSHOT_UPDATED,
            session_id=session.id,
            environment_name=name,
            payload={
                "operation_id": op.id,
                "operation": op.kind,
                "state": state_name,
                "snapshot_id": op.snapshot_id,
                "run_epoch": op.run_epoch,
            },
        )
        await self._change(session, name, change, event=event, position=position)
        return replacement

    def _fence(self, session, name, op):
        async def check():
            # Read-only: the substrate mutation that follows runs outside any
            # transaction anyway, and every phase commit re-checks the run
            # fence, predecessor and execution position transactionally.
            current = await self._store.load(session.id)
            if (
                current is None
                or current.instance_id != session.instance_id
                or current.run_epoch != session.run_epoch
                or current.status not in _IDLE
            ):
                raise ExecutionSnapshotConflict("Snapshot run fence changed.")
            state = _environment(await _projected_registry(self._store, session.id), name)
            if state["operations"].get(op.id) != op.model_dump(mode="json"):
                raise ExecutionSnapshotConflict("Snapshot operation authority changed.")
            expected = (
                op.target_generation
                if op.state == "verified" and op.kind == "restore"
                else op.binding_generation
            )
            if state["binding"] is None or state["binding"]["generation"] != expected:
                raise ExecutionSnapshotConflict("Snapshot binding authority changed.")

        return check

    async def _abandon(self, session, name, op) -> bool:
        """Drop an operation that never reached the substrate; its request stays retryable."""
        if not _unsubmitted(op):
            return False

        def drop(state):
            if state["operations"].get(op.id) != op.model_dump(mode="json"):
                raise ExecutionSnapshotConflict("Operation predecessor changed.")
            del state["operations"][op.id]
            if not state["operations"] and not state["snapshots"]:
                # Only this reservation established the binding.
                state["binding"] = None

        try:
            await self._change(session, name, drop)
        except Exception:
            return False
        return True

    async def _unknown(self, session, name, op):
        # Submitted itself is ambiguous if publication is fenced or loses ACK.
        with suppress(Exception):
            await self._phase(session, name, op, "unknown")

    @_bounded_operation
    async def capture(
        self,
        session_id: str,
        environment_name: str,
        adapter: ExecutionSnapshotAdapter,
        *,
        expected_run_epoch: int,
        expected_generation: str,
        idempotency_key: str,
    ) -> ExecutionSnapshotRecord:
        session = await self._session(session_id, expected_run_epoch)
        op = await self._reserve(
            session, environment_name, idempotency_key, expected_generation, adapter, "capture"
        )
        if op.state == "succeeded":
            state = _environment(
                _registry(await self._store.load_checkpoint(session_id)), environment_name
            )
            return ExecutionSnapshotRecord.model_validate(state["snapshots"][op.snapshot_id])
        if op.state == "verified" and op.run_epoch == expected_run_epoch:
            await adapter.resume_source(op.id, self._fence(session, environment_name, op))
            await self._phase(session, environment_name, op, "succeeded")
            state = _environment(
                _registry(await self._store.load_checkpoint(session_id)), environment_name
            )
            return ExecutionSnapshotRecord.model_validate(state["snapshots"][op.snapshot_id])
        if op.state != "intent" or op.run_epoch != expected_run_epoch:
            raise ExecutionSnapshotOutcomeUnknown("Capture was submitted; do not repeat it.")
        unsubmitted = _unsubmitted(op)
        try:
            if unsubmitted:
                await adapter.preflight_capture(self.policy)
            position = await self._position(session_id)
            if op.position is not None and op.position != position:
                raise ExecutionSnapshotConflict("Controller position changed during recovery.")
            op = await self._phase(session, environment_name, op, "submitted", position=position)
        except BaseException:
            await self._abandon(session, environment_name, op)
            raise
        fence = self._fence(session, environment_name, op)
        try:
            async with asyncio.timeout(self.policy.timeout_seconds):
                bundle = await (
                    adapter.capture(op.id, self.policy, fence)
                    if unsubmitted
                    else adapter.recover_capture(op.id, self.policy, fence)
                )
                if len(bundle.process) + len(bundle.workspace) > self.policy.max_total_bytes:
                    raise ExecutionSnapshotError("Snapshot exceeds aggregate byte policy.")
                artifacts = []
                owner = "execution-snapshot:" + op.snapshot_id
                for role, content in (("process", bundle.process), ("workspace", bundle.workspace)):
                    if not 0 < len(content) <= self.policy.max_artifact_bytes:
                        raise ExecutionSnapshotError("Snapshot artifact exceeds policy.")
                    artifact_id = "art_" + _digest([op.snapshot_id, role])[:32]
                    await fence()
                    await self._artifacts.put_bytes(
                        content,
                        artifact_id=artifact_id,
                        filename="execution-snapshot.bin",
                        content_type="application/octet-stream",
                        scope=ArtifactScope.SESSION,
                        session_id=session.id,
                        environment_name=environment_name,
                    )
                    await self._artifacts.pin(artifact_id, owner=owner)
                    part = ExecutionSnapshotArtifact(
                        role=role,
                        artifact_id=artifact_id,
                        sha256=hashlib.sha256(content).hexdigest(),
                        size_bytes=len(content),
                    )
                    await self._read(part)
                    artifacts.append(part)
                total = sum(part.size_bytes for part in artifacts)
                if total > self.policy.max_total_bytes:
                    raise ExecutionSnapshotError("Snapshot exceeds aggregate byte policy.")
                now = self._clock()
                record = ExecutionSnapshotRecord(
                    id=op.snapshot_id,
                    session_id=session.id,
                    session_instance_id=session.instance_id,
                    environment_name=environment_name,
                    binding_generation=expected_generation,
                    allocation_sha256=adapter.allocation_sha256,
                    created_run_epoch=session.run_epoch,
                    capability=adapter.capability,
                    position=position,
                    created_at=now,
                    expires_at=now + timedelta(seconds=self.policy.retention_seconds),
                    artifacts=tuple(artifacts),
                    total_bytes=total,
                    pin_owner=owner,
                    manifest_sha256="0" * 64,
                )
                record = record.model_copy(update={"manifest_sha256": _manifest(record)})

                def publish(state):
                    state["snapshots"][record.id] = record.model_dump(mode="json")

                op = await self._phase(
                    session, environment_name, op, "verified", transform=publish, position=position
                )
                await adapter.resume_source(op.id, self._fence(session, environment_name, op))
                op = await self._phase(
                    session, environment_name, op, "succeeded", position=position
                )
                return record
        except BaseException:
            if op.state == "submitted":
                await self._unknown(session, environment_name, op)
            raise

    @_bounded_operation
    async def reconcile(
        self,
        session_id: str,
        environment_name: str,
        operation_id: str,
        adapter: ExecutionSnapshotAdapter,
        *,
        expected_run_epoch: int,
        expected_generation: str,
        idempotency_key: str,
        target_generation: str | None = None,
    ) -> ExecutionSnapshotOperation:
        """Fence the old controller and reconcile its exact already-owned operation.

        Never resubmit an ambiguous checkpoint/restart. A capable adapter must
        prove its held capture or inactive restored process. The new epoch is
        returned in durable operation inspection and fences every late publisher.
        """
        session = await self._session(session_id, expected_run_epoch)
        state = _environment(
            _registry(await self._store.load_checkpoint(session_id)), environment_name
        )
        original = ExecutionSnapshotOperation.model_validate(state["operations"][operation_id])
        identity = "esop_" + _digest([session.instance_id, environment_name, idempotency_key])[:32]
        if identity != operation_id or original.allocation_sha256 != adapter.allocation_sha256:
            raise ExecutionSnapshotConflict("Recovery must attach the exact operation allocation.")
        fingerprint = _digest(
            [
                original.kind,
                original.binding_generation,
                None if original.kind == "capture" else original.snapshot_id,
                original.target_generation,
                adapter.allocation_sha256,
                adapter.capability.model_dump(mode="json"),
            ]
        )
        if fingerprint != original.request_sha256:
            raise ExecutionSnapshotConflict("Recovery adapter compatibility changed.")
        if original.state == "succeeded":
            return original
        if original.kind == "delete":
            raise ExecutionSnapshotConflict(
                "Deletion is retried through its idempotent deletion API."
            )
        if state["binding"]["generation"] != expected_generation:
            raise ExecutionSnapshotConflict(
                "Recovery requires the recorded binding and execution position."
            )
        position = original.position or await self._position(session_id)
        if await self._position(session_id) != position:
            raise ExecutionSnapshotConflict(
                "Controller position changed; reconciliation cannot establish continuation."
            )
        target_generation = target_generation or original.target_generation
        claimed = original.model_copy(
            update={
                "run_epoch": session.run_epoch + 1,
                # Preserve submission provenance even if this claim loses its ACK.
                "position": original.position,
                "state": "verified" if original.state == "verified" else "intent",
                "target_generation": target_generation if original.kind == "restore" else None,
                "request_sha256": _digest(
                    [
                        original.kind,
                        original.binding_generation,
                        None if original.kind == "capture" else original.snapshot_id,
                        target_generation if original.kind == "restore" else None,
                        adapter.allocation_sha256,
                        adapter.capability.model_dump(mode="json"),
                    ]
                ),
            }
        )

        def claim(current, checkpoint):
            if current.instance_id != session.instance_id or current.run_epoch != session.run_epoch:
                raise ExecutionSnapshotConflict("Recovery run epoch changed.")
            registry = _registry(checkpoint)
            actual = _environment(registry, environment_name)
            if actual["operations"].get(operation_id) != original.model_dump(mode="json"):
                raise ExecutionSnapshotConflict("Another controller recovered this operation.")
            if (
                actual["binding"]["generation"] != expected_generation
                or _position_digest(checkpoint) != position.checkpoint_sha256
            ):
                raise ExecutionSnapshotConflict("Recovery predecessor changed.")
            actual["operations"][operation_id] = claimed.model_dump(mode="json")
            if original.kind == "restore" and original.state == "verified":
                actual["binding"] = {
                    "generation": target_generation,
                    "allocation_sha256": adapter.allocation_sha256,
                }
            registry["environments"][environment_name] = actual
            updated = dict(checkpoint or {})
            updated[EXECUTION_SNAPSHOTS_KEY] = registry
            updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
            return updated

        with _execution_snapshot_authority_mutation_scope():
            recovered = await self._store.fence_run_and_transform_checkpoint(
                session_id,
                statuses=_IDLE,
                checkpoint_transform=claim,
            )

        if original.kind == "capture":
            await self.capture(
                session_id,
                environment_name,
                adapter,
                expected_run_epoch=recovered.run_epoch,
                expected_generation=original.binding_generation,
                idempotency_key=idempotency_key,
            )
            if target_generation is not None and target_generation != original.binding_generation:

                def rebind(actual):
                    if actual["binding"]["allocation_sha256"] != adapter.allocation_sha256:
                        raise ExecutionSnapshotConflict("Recovered capture binding changed.")
                    actual["binding"]["generation"] = target_generation

                await self._change(recovered, environment_name, rebind)
            state = _environment(
                _registry(await self._store.load_checkpoint(session_id)), environment_name
            )
            return ExecutionSnapshotOperation.model_validate(state["operations"][operation_id])
        return await self.restore(
            session_id,
            environment_name,
            original.snapshot_id,
            adapter,
            expected_run_epoch=recovered.run_epoch,
            expected_generation=original.binding_generation,
            target_generation=target_generation,
            idempotency_key=idempotency_key,
        )

    async def _read(self, part: ExecutionSnapshotArtifact) -> bytes:
        if part.size_bytes > self.policy.max_artifact_bytes:
            raise ExecutionSnapshotError("Stored artifact exceeds policy.")
        try:
            result = await self._artifacts.read_bytes(part.artifact_id, max_bytes=part.size_bytes)
        except (KeyError, FileNotFoundError):
            raise ExecutionSnapshotError("Snapshot artifact is unavailable.") from None
        if (
            type(result) is not ArtifactReadResult
            or result.truncated
            or result.total_bytes != part.size_bytes
            or len(result.content) != part.size_bytes
            or result.metadata.id != part.artifact_id
            or hashlib.sha256(result.content).hexdigest() != part.sha256
        ):
            raise ExecutionSnapshotError("Snapshot integrity verification failed.")
        return result.content

    @_bounded_operation
    async def restore(
        self,
        session_id: str,
        environment_name: str,
        snapshot_id: str,
        target: ExecutionSnapshotAdapter,
        *,
        expected_run_epoch: int,
        expected_generation: str,
        target_generation: str,
        idempotency_key: str,
    ) -> ExecutionSnapshotOperation:
        session = await self._session(session_id, expected_run_epoch)
        state = _environment(
            _registry(await self._store.load_checkpoint(session_id)), environment_name
        )
        identity = "esop_" + _digest([session.instance_id, environment_name, idempotency_key])[:32]
        op = None
        if identity in state["operations"]:
            # Authenticate the original request before replaying or settling it.
            # A submitted target owns its process state independently of source
            # snapshot retention and artifact availability.
            op = await self._reserve(
                session,
                environment_name,
                idempotency_key,
                expected_generation,
                target,
                "restore",
                snapshot_id,
                target_generation,
            )
            if op.state == "succeeded":
                return op
            if op.run_epoch != expected_run_epoch or op.state in {"submitted", "unknown"}:
                raise ExecutionSnapshotOutcomeUnknown(
                    "Restore was submitted; inspect the inactive target."
                )
        if op is not None and not _unsubmitted(op):
            if op.position is None:
                raise ExecutionSnapshotError("Submitted restore has no execution position.")
            position = op.position
            content = None
        else:
            record = ExecutionSnapshotRecord.model_validate(state["snapshots"][snapshot_id])
            content = await self._restore_content(
                session, environment_name, record, target, expected_generation, target_generation
            )
            position = record.position
        if op is None:
            op = await self._reserve(
                session,
                environment_name,
                idempotency_key,
                expected_generation,
                target,
                "restore",
                snapshot_id,
                target_generation,
            )
        if op.state == "succeeded":
            return op
        if op.run_epoch != expected_run_epoch or op.state in {"submitted", "unknown"}:
            raise ExecutionSnapshotOutcomeUnknown(
                "Restore was submitted; inspect the inactive target."
            )
        try:
            async with asyncio.timeout(self.policy.timeout_seconds):
                if op.state == "intent":
                    if await self._position(session_id) != position:
                        raise ExecutionSnapshotConflict(
                            "Controller advanced since capture; reconciliation required."
                        )
                    unsubmitted = _unsubmitted(op)
                    op = await self._phase(
                        session, environment_name, op, "submitted", position=position
                    )
                    fence = self._fence(session, environment_name, op)
                    if unsubmitted:
                        assert content is not None
                        await target.restore(op.id, content, self.policy, fence)
                    else:
                        await target.verify_restore(op.id, self.policy, fence)

                    def publish(state):
                        state["binding"] = {
                            "generation": target_generation,
                            "allocation_sha256": target.allocation_sha256,
                        }

                    op = await self._phase(
                        session,
                        environment_name,
                        op,
                        "verified",
                        transform=publish,
                        position=position,
                    )
                await target.activate(op.id, self._fence(session, environment_name, op))
                return await self._phase(
                    session, environment_name, op, "succeeded", position=position
                )
        except BaseException:
            if op.state == "submitted":
                await self._unknown(session, environment_name, op)
            elif op.state == "intent":
                await self._abandon(session, environment_name, op)
            raise

    async def _restore_content(
        self, session, environment_name, record, target, expected_generation, target_generation
    ) -> CapturedExecutionSnapshot:
        """Admit a fresh restore before reserving or submitting target work."""
        if (
            record.session_instance_id != session.instance_id
            or record.session_id != session.id
            or record.environment_name != environment_name
            or record.retention != "retained"
            or record.expires_at <= self._clock()
            or _manifest(record) != record.manifest_sha256
            or record.capability != target.capability
            or target.allocation_sha256 == record.allocation_sha256
            or target_generation == expected_generation
        ):
            raise ExecutionSnapshotError(
                "Snapshot is unavailable or incompatible with a fresh target."
            )
        if (
            sum(part.size_bytes for part in record.artifacts) != record.total_bytes
            or record.total_bytes > self.policy.max_total_bytes
        ):
            raise ExecutionSnapshotError("Snapshot accounting mismatch.")
        if sorted(part.role for part in record.artifacts) != ["process", "workspace"]:
            raise ExecutionSnapshotError("Snapshot component set is incomplete.")
        # Validate every component before recording a provider submission.
        return CapturedExecutionSnapshot(
            **{part.role: await self._read(part) for part in record.artifacts}
        )

    @_bounded_operation
    async def delete(
        self,
        session_id: str,
        environment_name: str,
        snapshot_id: str,
        *,
        expected_run_epoch: int,
        expected_generation: str,
        idempotency_key: str,
    ) -> None:
        session = await self._session(session_id, expected_run_epoch)
        state = _environment(
            _registry(await self._store.load_checkpoint(session_id)), environment_name
        )
        record = ExecutionSnapshotRecord.model_validate(state["snapshots"][snapshot_id])
        identity = "esop_" + _digest([session.instance_id, environment_name, idempotency_key])[:32]
        if record.retention == "deleted" and identity not in state["operations"]:
            return
        if state["binding"] is None:
            raise ExecutionSnapshotConflict("No source binding is recorded.")
        op = ExecutionSnapshotOperation(
            id=identity,
            kind="delete",
            state="intent",
            request_sha256=_digest(["delete", snapshot_id, expected_generation]),
            run_epoch=expected_run_epoch,
            binding_generation=expected_generation,
            snapshot_id=snapshot_id,
            allocation_sha256=state["binding"]["allocation_sha256"],
            created_at=self._clock(),
        )
        existing = None

        def reserve(state):
            nonlocal existing
            if state["binding"]["generation"] != expected_generation:
                raise ExecutionSnapshotConflict("Deletion binding generation changed.")
            if op.id in state["operations"]:
                existing = ExecutionSnapshotOperation.model_validate(state["operations"][op.id])
                if existing.request_sha256 != op.request_sha256:
                    raise ExecutionSnapshotConflict("Deletion identity conflicts.")
                if existing.state != "succeeded" and existing.run_epoch != session.run_epoch:
                    # A rejected continuation fences the old owner. Adopt its
                    # idempotent deletion under the current transaction's fence.
                    existing = existing.model_copy(update={"run_epoch": session.run_epoch})
                    state["operations"][op.id] = existing.model_dump(mode="json")
                return
            if any(value["state"] != "succeeded" for value in state["operations"].values()):
                raise ExecutionSnapshotConflict("Another snapshot operation owns the binding.")
            self._compact_operations(state)
            state["operations"][op.id] = op.model_dump(mode="json")

        await self._change(session, environment_name, reserve)
        op = existing or op
        if op.state == "succeeded":
            return
        if op.run_epoch != expected_run_epoch:
            raise ExecutionSnapshotConflict("Deletion belongs to another run epoch.")

        def retire(state):
            state["snapshots"][snapshot_id] = record.model_copy(
                update={"retention": "deleting"}
            ).model_dump(mode="json")

        if op.state == "intent":
            op = await self._phase(session, environment_name, op, "submitted", transform=retire)
        # Pin release/delete are idempotent and the binding remains unavailable
        # to restoration throughout deletion, including acknowledgement loss.
        async with asyncio.timeout(self.policy.timeout_seconds):
            for part in record.artifacts:
                await self._fence(session, environment_name, op)()
                # A prior attempt may have deleted this component before losing
                # its ACK. Pin APIs need not accept an already-absent artifact.
                with suppress(KeyError, FileNotFoundError):
                    await self._artifacts.release_pin(part.artifact_id, owner=record.pin_owner)
                await self._artifacts.delete(part.artifact_id)

            def deleted(state):
                state["snapshots"][snapshot_id] = record.model_copy(
                    update={"retention": "deleted"}
                ).model_dump(mode="json")

            await self._phase(session, environment_name, op, "succeeded", transform=deleted)

    async def release_binding(
        self,
        session_id: str,
        environment_name: str,
        adapter: ExecutionSnapshotAdapter | None,
        *,
        expected_run_epoch: int,
        expected_generation: str,
        target_generation: str | None,
    ) -> None:
        """Accept that the bound workload is gone and expose a different allocation.

        Use when the bound allocation was lost and restoration is not wanted or
        not possible, for example after the session advanced past every
        snapshot. With an adapter, its allocation becomes the binding; retained
        snapshots stay restorable from their recorded execution position.
        Without one the binding is cleared, which requires deleting retained
        snapshots first. Repeating a completed release is a no-op.
        """
        session = await self._session(session_id, expected_run_epoch)
        environment_name = require_durable_clean_nonblank(environment_name, "environment_name")
        expected_generation = require_durable_clean_nonblank(
            expected_generation, "binding_generation"
        )
        replacement = None
        if adapter is not None:
            if adapter.capability.fidelity is ExecutionSnapshotFidelity.UNSUPPORTED:
                raise ExecutionSnapshotError("Environment does not support execution snapshots.")
            if target_generation is None:
                raise ValueError("Rebinding to an adapter requires its target generation.")
            replacement = {
                "generation": require_durable_clean_nonblank(
                    target_generation, "target_generation"
                ),
                "allocation_sha256": adapter.allocation_sha256,
            }

        def release(state):
            if state["binding"] == replacement:
                return
            if state["binding"] is None or state["binding"]["generation"] != expected_generation:
                raise ExecutionSnapshotConflict("Expected environment generation changed.")
            if any(value["state"] != "succeeded" for value in state["operations"].values()):
                raise ExecutionSnapshotOutcomeUnknown(
                    "Another snapshot operation requires settlement."
                )
            if replacement is None and any(
                value["retention"] != "deleted" for value in state["snapshots"].values()
            ):
                raise ExecutionSnapshotConflict(
                    "Delete retained snapshots before clearing the binding."
                )
            state["binding"] = replacement

        await self._change(session, environment_name, release)


async def ensure_execution_snapshot_binding(
    store: SessionStore, session: Session, registered
) -> None:
    """Fail closed before any model/tool exposure to an unsettled restored state.

    This runs on every model step and tool call, so it reads only the bounded
    snapshot projection and returns early when nothing was ever bound.
    """
    _require_execution_snapshot_binding(await _projected_registry(store, session.id), registered)


def _require_execution_snapshot_binding(registry, registered) -> None:
    state = _environment(registry, registered.spec.name)
    if state["binding"] is None:
        return
    if any(value["state"] != "succeeded" for value in state["operations"].values()):
        raise ExecutionSnapshotOutcomeUnknown("Execution snapshot operation requires settlement.")
    adapter = registered.environment.execution_snapshot_adapter
    if (
        adapter is None
        or state["binding"]["generation"] != registered.binding_generation_id
        or state["binding"]["allocation_sha256"] != adapter.allocation_sha256
    ):
        raise ExecutionSnapshotConflict(
            "Environment requires explicit snapshot restoration or binding release before exposure."
        )


async def inspect_execution_snapshots(
    store: SessionStore,
    session_id: str,
    *,
    environment_name: str | None = None,
    limit: int = 25,
):
    """Bounded private-state projection shared by CLI, server and dashboard."""
    from cayu.environments.snapshots import ExecutionSnapshotInspection, ExecutionSnapshotSummary

    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("Snapshot inspection limit must be between 1 and 100.")
    if await store.load_state(session_id) is None:
        raise KeyError(session_id)
    registry = _registry(
        decode_runtime_checkpoint(
            await store.load_execution_snapshot_checkpoint(session_id), session_id=session_id
        )
    )
    names = [environment_name] if environment_name is not None else sorted(registry["environments"])
    results = []
    for name in names[:limit]:
        state = _environment(registry, name)
        operations = tuple(
            ExecutionSnapshotOperation.model_validate(value)
            for value in _ordered_operations(state)[-limit:]
        )
        records = tuple(
            ExecutionSnapshotRecord.model_validate(value)
            for value in sorted(
                state["snapshots"].values(), key=lambda value: (value["created_at"], value["id"])
            )[-limit:]
        )
        results.append(
            ExecutionSnapshotInspection(
                environment_name=name,
                binding_generation=None
                if state["binding"] is None
                else state["binding"]["generation"],
                operations=operations,
                snapshots=tuple(
                    ExecutionSnapshotSummary(
                        id=record.id,
                        capability=record.capability,
                        created_at=record.created_at,
                        expires_at=record.expires_at,
                        retention=record.retention,
                        expired=record.expires_at <= datetime.now(UTC),
                        total_bytes=record.total_bytes,
                    )
                    for record in records
                ),
                truncated=len(names) > limit
                or len(state["operations"]) > limit
                or len(state["snapshots"]) > limit,
            )
        )
    return tuple(results)
