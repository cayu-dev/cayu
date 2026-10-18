"""PostgreSQL agent snapshot store."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

try:
    from psycopg.rows import dict_row, tuple_row
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without the extra
    raise RuntimeError(
        "Cayu's Postgres stores require the optional psycopg packages. "
        'Install them with `pip install "cayu[postgres]"`.'
    ) from exc

from cayu.snapshots.base import (
    AgentSnapshot,
    AgentSnapshotAccess,
    AgentSnapshotAuthorizationError,
    AgentSnapshotClosureInspection,
    AgentSnapshotGCPlan,
    AgentSnapshotGCReceipt,
    AgentSnapshotGCRequest,
    AgentSnapshotIdentityBinding,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationProgress,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotMaterializedComponent,
    AgentSnapshotNode,
    AgentSnapshotPinReceipt,
    AgentSnapshotPinRequest,
    AgentSnapshotProtection,
    AgentSnapshotPutReceipt,
    AgentSnapshotReleaseReceipt,
    AgentSnapshotReleaseRequest,
    AgentSnapshotResultBinding,
    AgentSnapshotStore,
    AgentSnapshotStoreConflict,
    AgentSnapshotTrialBinding,
    AgentSnapshotVerificationError,
    _claim_progress,
    _clean,
    _complete_progress,
    _content_sha256,
    _create_gc_plan,
    _create_gc_receipt,
    _final_progress,
    _gc_deletable_nodes,
    _PreparedSnapshotPut,
    _releasable_pin,
    _releasable_protection,
    _replayed_operation,
    _require_initial_progress,
    _require_loaded_record,
    _require_saved_record,
    _require_scope_materialization,
    _require_stored_pin,
    _same_record_identity,
    _scope_key,
    _sha256_hex,
    _snapshot_accesses_by_root,
    _snapshot_model_bytes,
    _snapshot_model_json,
    _snapshot_unresolved_bindings,
    _validate_snapshot_binding,
    _validate_store_record,
    _validated_progress_row,
    _verified_binding_row,
    _verified_root_closure,
    _verified_stored_binding,
)
from cayu.storage._agent_snapshot_schema import AGENT_SNAPSHOT_REVISION, AGENT_SNAPSHOT_TABLES
from cayu.storage._postgres_base import _PostgresStoreBase
from cayu.storage._retention_sql import (
    PostgresRetentionSql,
    held_snapshot_identifiers,
    snapshot_pin_text,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

# Every write takes one transaction-scoped exclusive lock and every read the
# shared form, the PostgreSQL equivalent of the SQLite store's BEGIN IMMEDIATE:
# a read sees one committed state, and the check-then-write sequences below run
# as if alone. Transaction-scoped locks are safe behind transaction pooling.
_LOCK_KEY = "cayu-agent-snapshot-store"
_WRITE_LOCK = "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))"
_READ_LOCK = "SELECT pg_advisory_xact_lock_shared(hashtextextended(%s, 0))"
# Storage retention fences snapshot publication with LOCK TABLE ... IN SHARE MODE
# over the tables it reads, in sorted name order, without the advisory lock. A
# write takes ROW EXCLUSIVE on every snapshot table up front in that same order,
# so it either waits for retention before touching a table or retention waits
# for it; the two can never hold locks the other is waiting for.
_WRITE_TABLE_LOCK = f"LOCK TABLE {', '.join(sorted(AGENT_SNAPSHOT_TABLES))} IN ROW EXCLUSIVE MODE"

_PROGRESS_COLUMNS = (
    "snapshot_fingerprint, candidate_id, state_scope_id, state_mode, "
    "progress_id, revision, document, materialization_fingerprint"
)
_PROGRESS_BY_SCOPE = (
    f"SELECT {_PROGRESS_COLUMNS} FROM cayu_agent_snapshot_materialization_progress "
    "WHERE snapshot_fingerprint = %s AND candidate_id = %s "
    "AND state_scope_id = %s AND state_mode = %s"
)
_PROGRESS_TRANSITION = (
    "UPDATE cayu_agent_snapshot_materialization_progress "
    "SET revision = %s, document = %s, materialization_fingerprint = %s "
    "WHERE snapshot_fingerprint = %s AND candidate_id = %s "
    "AND state_scope_id = %s AND state_mode = %s "
    "AND progress_id = %s AND revision = %s AND document = %s "
    "AND materialization_fingerprint IS NOT DISTINCT FROM %s"
)


class PostgresAgentSnapshotStore(_PostgresStoreBase, AgentSnapshotStore):
    """Snapshot manifests and evaluation lineage shared by processes on PostgreSQL.

    The store keeps the same records and enforces the same checks as
    ``SQLiteAgentSnapshotStore``: content-addressed nodes and roots, authorized
    identity bindings, pins and protections, idempotent lifecycle operations,
    GC plans and receipts, and materialization progress. Provider objects are
    not stored here; compose an ``AgentSnapshotObjectStore`` for them.

    The tables join Cayu's shared revision history (revision 120). The default
    ``schema_mode`` is ``VALIDATE``; ``cayu storage migrate`` creates them.
    """

    _min_required_revision = AGENT_SNAPSHOT_REVISION

    @asynccontextmanager
    async def _transaction(self, *, write: bool, row_factory: Any = dict_row) -> AsyncIterator[Any]:
        async with self._ready_connection() as conn:
            try:
                async with conn.cursor(row_factory=row_factory) as cur:
                    await cur.execute(_WRITE_LOCK if write else _READ_LOCK, (_LOCK_KEY,))
                    if write:
                        await cur.execute(_WRITE_TABLE_LOCK)
                    yield cur
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def retention_held_identifiers(self) -> frozenset[str]:
        async with self._transaction(write=False, row_factory=tuple_row) as cur:
            text = await snapshot_pin_text(PostgresRetentionSql(cur))
        return held_snapshot_identifiers(text.split("\0") if text else ())

    # -- closures ---------------------------------------------------------------

    async def _load_access_closure(
        self,
        cur: Any,
        access: AgentSnapshotAccess,
    ) -> tuple[
        AgentSnapshot,
        AgentSnapshotIdentityBinding,
        tuple[AgentSnapshotNode, ...],
        dict[str, Any],
    ]:
        validated_access = AgentSnapshotAccess.model_validate(access.model_dump(mode="json"))
        await cur.execute(
            "SELECT snapshot_root, authority_scope_fingerprint, binding_document, "
            "snapshot_document FROM cayu_agent_snapshot_bindings WHERE binding_id = %s",
            (validated_access.binding_id,),
        )
        binding, snapshot = _verified_binding_row(validated_access, await cur.fetchone())
        root = validated_access.snapshot.snapshot_root
        await cur.execute(
            "SELECT manifest_document, manifest_bytes, closure_bytes, object_count "
            "FROM cayu_agent_snapshot_roots WHERE snapshot_root = %s",
            (root,),
        )
        root_row = await cur.fetchone()
        rows: list[dict[str, Any]] = []
        if root_row is not None:
            await cur.execute(
                "SELECT link.node_digest, link.ordinal, node.node_kind, node.schema_id, "
                "node.byte_count, node.document "
                "FROM cayu_agent_snapshot_root_nodes AS link "
                "LEFT JOIN cayu_agent_snapshot_nodes AS node ON node.digest = link.node_digest "
                "WHERE link.snapshot_root = %s ORDER BY link.ordinal",
                (root,),
            )
            rows = await cur.fetchall()
        closure = _verified_root_closure(snapshot, root_row, rows)
        return snapshot, binding, closure, cast("dict[str, Any]", root_row)

    async def _put(self, cur: Any, put: _PreparedSnapshotPut) -> str:
        await cur.execute(
            "SELECT binding_document, snapshot_document, put_receipt_document "
            "FROM cayu_agent_snapshot_bindings WHERE binding_id = %s",
            (put.binding.binding_id,),
        )
        existing = await cur.fetchone()
        if existing is not None:
            put.require_replay(existing)
            await self._load_access_closure(cur, put.access)
            return cast("str", existing["put_receipt_document"])
        node_rows = put.node_rows
        await cur.executemany(
            "INSERT INTO cayu_agent_snapshot_nodes "
            "(digest, node_kind, schema_id, byte_count, document) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (digest) DO NOTHING",
            node_rows,
        )
        await cur.execute(
            "SELECT digest, node_kind, schema_id, byte_count, document "
            "FROM cayu_agent_snapshot_nodes WHERE digest = ANY(%s)",
            ([row[0] for row in node_rows],),
        )
        stored_nodes = {row["digest"]: row for row in await cur.fetchall()}
        for node_row in node_rows:
            put.require_stored_node(stored_nodes.get(node_row[0]), node_row)
        root = put.snapshot.snapshot_root
        await cur.execute(
            "SELECT manifest_document, manifest_bytes, closure_bytes, object_count "
            "FROM cayu_agent_snapshot_roots WHERE snapshot_root = %s",
            (root,),
        )
        root_row = await cur.fetchone()
        if root_row is None:
            await cur.execute(
                "INSERT INTO cayu_agent_snapshot_roots "
                "(snapshot_root, manifest_document, manifest_bytes, closure_bytes, object_count) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    root,
                    put.root_manifest_document,
                    put.manifest_bytes,
                    put.closure_bytes,
                    len(put.closure),
                ),
            )
            await cur.executemany(
                "INSERT INTO cayu_agent_snapshot_root_nodes "
                "(snapshot_root, node_digest, ordinal) VALUES (%s, %s, %s)",
                [(root, digest, ordinal) for ordinal, digest in enumerate(put.node_digests)],
            )
        else:
            await cur.execute(
                "SELECT node_digest FROM cayu_agent_snapshot_root_nodes "
                "WHERE snapshot_root = %s ORDER BY ordinal",
                (root,),
            )
            indexed = tuple(row["node_digest"] for row in await cur.fetchall())
            put.require_stored_root(root_row, indexed)
        await cur.execute(
            "INSERT INTO cayu_agent_snapshot_bindings "
            "(binding_id, snapshot_root, authority_scope_fingerprint, binding_document, "
            "snapshot_document, put_receipt_document) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                put.binding.binding_id,
                root,
                put.binding.authority_scope_fingerprint,
                put.binding_document,
                put.snapshot_document,
                put.receipt_document,
            ),
        )
        await cur.execute(
            "INSERT INTO cayu_agent_snapshot_records (record_kind, fingerprint, document) "
            "VALUES ('snapshot', %s, %s) ON CONFLICT (record_kind, fingerprint) DO NOTHING",
            (root, put.snapshot_document),
        )
        await cur.execute(
            "SELECT document FROM cayu_agent_snapshot_records "
            "WHERE record_kind = 'snapshot' AND fingerprint = %s",
            (root,),
        )
        put.require_stored_record(await cur.fetchone())
        return put.receipt_document

    async def put_snapshot(
        self,
        snapshot: AgentSnapshot,
        binding: AgentSnapshotIdentityBinding,
    ) -> AgentSnapshotPutReceipt:
        put = _PreparedSnapshotPut(snapshot, binding)
        async with self._transaction(write=True) as cur:
            document = await self._put(cur, put)
        return AgentSnapshotPutReceipt.model_validate_json(document)

    async def put_snapshot_and_pin(
        self,
        snapshot: AgentSnapshot,
        binding: AgentSnapshotIdentityBinding,
        pin: AgentSnapshotPinRequest,
    ) -> tuple[AgentSnapshotPutReceipt, AgentSnapshotPinReceipt]:
        validated_snapshot, validated_binding = _validate_snapshot_binding(snapshot, binding)
        validated_pin = AgentSnapshotPinRequest.model_validate(pin.model_dump(mode="json"))
        put = _PreparedSnapshotPut(validated_snapshot, validated_binding)
        if validated_pin.access != put.access:
            raise AgentSnapshotAuthorizationError(
                "Atomic snapshot import pin names another binding or scope."
            )
        async with self._transaction(write=True) as cur:
            put_document = await self._put(cur, put)
            pin_document = await self._pin(cur, validated_pin)
        return (
            AgentSnapshotPutReceipt.model_validate_json(put_document),
            AgentSnapshotPinReceipt.model_validate_json(pin_document),
        )

    async def load_identity_binding(self, binding_id: str) -> AgentSnapshotIdentityBinding | None:
        binding_id = _sha256_hex(binding_id, "binding_id")
        async with self._transaction(write=False) as cur:
            await cur.execute(
                "SELECT snapshot_root, authority_scope_fingerprint, binding_document "
                "FROM cayu_agent_snapshot_bindings WHERE binding_id = %s",
                (binding_id,),
            )
            document = _verified_stored_binding(await cur.fetchone(), binding_id)
        return (
            None if document is None else AgentSnapshotIdentityBinding.model_validate_json(document)
        )

    async def get_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshot:
        async with self._transaction(write=False) as cur:
            snapshot, _, _, _ = await self._load_access_closure(cur, access)
        return AgentSnapshot.model_validate_json(_snapshot_model_json(snapshot))

    async def enumerate_snapshot_closure(
        self, access: AgentSnapshotAccess
    ) -> tuple[AgentSnapshotNode, ...]:
        async with self._transaction(write=False) as cur:
            _, _, closure, _ = await self._load_access_closure(cur, access)
        return tuple(
            AgentSnapshotNode.model_validate_json(_snapshot_model_json(node)) for node in closure
        )

    async def inspect_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshotClosureInspection:
        async with self._transaction(write=False) as cur:
            snapshot, _, closure, root_row = await self._load_access_closure(cur, access)
            sizes = {
                node.digest: len(_snapshot_model_bytes(node, "agent_snapshot_node"))
                for node in closure
            }
            await cur.execute(
                "SELECT node_digest, count(*) AS count FROM cayu_agent_snapshot_root_nodes "
                "WHERE node_digest = ANY(%s) GROUP BY node_digest",
                (list(sizes),),
            )
            counts = {row["node_digest"]: row["count"] for row in await cur.fetchall()}
        unique = sum(size for digest, size in sizes.items() if counts.get(digest) == 1)
        inspection = AgentSnapshotClosureInspection(
            snapshot=access.snapshot,
            root_manifest_bytes=cast("int", root_row["manifest_bytes"]),
            logical_closure_bytes=cast("int", root_row["closure_bytes"]),
            unique_stored_bytes=unique,
            shared_bytes=cast("int", root_row["closure_bytes"]) - unique,
            object_count=len(closure),
            node_digests=tuple(sorted(sizes)),
            unresolved_external_bindings=_snapshot_unresolved_bindings(snapshot),
        )
        return AgentSnapshotClosureInspection.model_validate_json(_snapshot_model_json(inspection))

    # -- lifecycle operations ---------------------------------------------------

    async def _operation_replay(
        self,
        cur: Any,
        *,
        kind: str,
        operation_id: str,
        request_material: object,
        model: type[BaseModel],
    ) -> BaseModel | None:
        await cur.execute(
            "SELECT request_digest, response_model, response_document "
            "FROM cayu_agent_snapshot_lifecycle_operations "
            "WHERE operation_kind = %s AND operation_id = %s",
            (kind, operation_id),
        )
        return _replayed_operation(
            await cur.fetchone(),
            kind=kind,
            request_digest=_content_sha256(request_material, f"{kind}_request"),
            model=model,
        )

    async def _record_operation(
        self,
        cur: Any,
        *,
        kind: str,
        operation_id: str,
        request_material: object,
        response: BaseModel,
    ) -> None:
        await cur.execute(
            "INSERT INTO cayu_agent_snapshot_lifecycle_operations "
            "(operation_kind, operation_id, request_digest, response_model, response_document) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                kind,
                operation_id,
                _content_sha256(request_material, f"{kind}_request"),
                type(response).__name__,
                _snapshot_model_json(response),
            ),
        )

    async def _pin(self, cur: Any, request: AgentSnapshotPinRequest) -> str:
        validated = AgentSnapshotPinRequest.model_validate(request.model_dump(mode="json"))
        replay = await self._operation_replay(
            cur,
            kind="pin",
            operation_id=validated.operation_id,
            request_material=validated.identity_material(),
            model=AgentSnapshotPinReceipt,
        )
        if replay is not None:
            return _snapshot_model_json(replay)
        await self._load_access_closure(cur, validated.access)
        receipt = AgentSnapshotPinReceipt.from_request(validated)
        await cur.execute(
            "INSERT INTO cayu_agent_snapshot_pins "
            "(pin_id, snapshot_root, binding_id, document, released) "
            "VALUES (%s, %s, %s, %s, 0) ON CONFLICT (pin_id) DO NOTHING",
            (
                receipt.pin_id,
                receipt.snapshot.snapshot_root,
                receipt.binding_id,
                _snapshot_model_json(receipt),
            ),
        )
        await cur.execute(
            "SELECT document FROM cayu_agent_snapshot_pins WHERE pin_id = %s",
            (receipt.pin_id,),
        )
        _require_stored_pin(await cur.fetchone(), receipt)
        await self._record_operation(
            cur,
            kind="pin",
            operation_id=validated.operation_id,
            request_material=validated.identity_material(),
            response=receipt,
        )
        return _snapshot_model_json(receipt)

    async def pin_snapshot(self, request: AgentSnapshotPinRequest) -> AgentSnapshotPinReceipt:
        async with self._transaction(write=True) as cur:
            document = await self._pin(cur, request)
        return AgentSnapshotPinReceipt.model_validate_json(document)

    async def release_snapshot_pin(
        self, request: AgentSnapshotReleaseRequest
    ) -> AgentSnapshotReleaseReceipt:
        validated = AgentSnapshotReleaseRequest.model_validate(request.model_dump(mode="json"))
        async with self._transaction(write=True) as cur:
            replay = await self._operation_replay(
                cur,
                kind="release",
                operation_id=validated.operation_id,
                request_material=validated.identity_material(),
                model=AgentSnapshotReleaseReceipt,
            )
            if replay is not None:
                document = _snapshot_model_json(replay)
            else:
                await self._load_access_closure(cur, validated.access)
                await cur.execute(
                    "SELECT snapshot_root, binding_id, document FROM cayu_agent_snapshot_pins "
                    "WHERE pin_id = %s",
                    (validated.pin_id,),
                )
                pin = _releasable_pin(await cur.fetchone(), validated)
                receipt = AgentSnapshotReleaseReceipt.from_request(validated)
                await cur.execute(
                    "UPDATE cayu_agent_snapshot_pins SET released = 1 WHERE pin_id = %s",
                    (pin.pin_id,),
                )
                await self._record_operation(
                    cur,
                    kind="release",
                    operation_id=validated.operation_id,
                    request_material=validated.identity_material(),
                    response=receipt,
                )
                document = _snapshot_model_json(receipt)
        return AgentSnapshotReleaseReceipt.model_validate_json(document)

    async def protect_snapshot(
        self, protection: AgentSnapshotProtection
    ) -> AgentSnapshotProtection:
        validated = AgentSnapshotProtection.model_validate(protection.model_dump(mode="json"))
        material = validated.model_dump(mode="json")
        async with self._transaction(write=True) as cur:
            replay = await self._operation_replay(
                cur,
                kind="protect",
                operation_id=validated.operation_id,
                request_material=material,
                model=AgentSnapshotProtection,
            )
            if replay is not None:
                document = _snapshot_model_json(replay)
            else:
                await self._load_access_closure(cur, validated.access)
                document = _snapshot_model_json(validated)
                await cur.execute(
                    "INSERT INTO cayu_agent_snapshot_protections "
                    "(protection_id, snapshot_root, binding_id, kind, document, released) "
                    "VALUES (%s, %s, %s, %s, %s, 0) ON CONFLICT (protection_id) DO NOTHING",
                    (
                        validated.protection_id,
                        validated.access.snapshot.snapshot_root,
                        validated.access.binding_id,
                        validated.kind.value,
                        document,
                    ),
                )
                await cur.execute(
                    "SELECT document FROM cayu_agent_snapshot_protections WHERE protection_id = %s",
                    (validated.protection_id,),
                )
                row = await cur.fetchone()
                if row is None or row["document"] != document:
                    raise AgentSnapshotStoreConflict("Snapshot protection identity conflicts.")
                await self._record_operation(
                    cur,
                    kind="protect",
                    operation_id=validated.operation_id,
                    request_material=material,
                    response=validated,
                )
        return AgentSnapshotProtection.model_validate_json(document)

    async def release_snapshot_protection(
        self,
        *,
        operation_id: str,
        access: AgentSnapshotAccess,
        protection_id: str,
    ) -> AgentSnapshotProtection:
        operation_id = _clean(operation_id, "operation_id", max_chars=256)
        protection_id = _sha256_hex(protection_id, "protection_id")
        request_material = {
            "operation_id": operation_id,
            "access": access.model_dump(mode="json"),
            "protection_id": protection_id,
        }
        async with self._transaction(write=True) as cur:
            replay = await self._operation_replay(
                cur,
                kind="unprotect",
                operation_id=operation_id,
                request_material=request_material,
                model=AgentSnapshotProtection,
            )
            if replay is not None:
                document = _snapshot_model_json(replay)
            else:
                await self._load_access_closure(cur, access)
                await cur.execute(
                    "SELECT document FROM cayu_agent_snapshot_protections WHERE protection_id = %s",
                    (protection_id,),
                )
                protection = _releasable_protection(await cur.fetchone(), access)
                await cur.execute(
                    "UPDATE cayu_agent_snapshot_protections SET released = 1 "
                    "WHERE protection_id = %s",
                    (protection_id,),
                )
                await self._record_operation(
                    cur,
                    kind="unprotect",
                    operation_id=operation_id,
                    request_material=request_material,
                    response=protection,
                )
                document = _snapshot_model_json(protection)
        return AgentSnapshotProtection.model_validate_json(document)

    # -- garbage collection -----------------------------------------------------

    async def _root_is_protected(self, cur: Any, snapshot_root: str) -> bool:
        await cur.execute(
            "SELECT EXISTS (SELECT 1 FROM cayu_agent_snapshot_pins "
            "WHERE snapshot_root = %s AND released = 0) "
            "OR EXISTS (SELECT 1 FROM cayu_agent_snapshot_protections "
            "WHERE snapshot_root = %s AND released = 0) AS held",
            (snapshot_root, snapshot_root),
        )
        row = await cur.fetchone()
        return bool(row is not None and row["held"])

    async def _binding_ids(self, cur: Any, snapshot_root: str) -> set[str]:
        await cur.execute(
            "SELECT binding_id FROM cayu_agent_snapshot_bindings WHERE snapshot_root = %s",
            (snapshot_root,),
        )
        return {row["binding_id"] for row in await cur.fetchall()}

    async def _roots_by_node(self, cur: Any, roots: set[str]) -> dict[str, set[str]]:
        """Every root that reaches each node of ``roots``."""

        if not roots:
            return {}
        await cur.execute(
            "SELECT DISTINCT node_digest FROM cayu_agent_snapshot_root_nodes "
            "WHERE snapshot_root = ANY(%s)",
            (sorted(roots),),
        )
        candidates = [row["node_digest"] for row in await cur.fetchall()]
        if not candidates:
            return {}
        await cur.execute(
            "SELECT node_digest, snapshot_root FROM cayu_agent_snapshot_root_nodes "
            "WHERE node_digest = ANY(%s)",
            (candidates,),
        )
        reached: dict[str, set[str]] = {digest: set() for digest in candidates}
        for row in await cur.fetchall():
            reached[row["node_digest"]].add(row["snapshot_root"])
        return reached

    async def _node_bytes(self, cur: Any, digests: set[str]) -> dict[str, int]:
        if not digests:
            return {}
        await cur.execute(
            "SELECT digest, byte_count FROM cayu_agent_snapshot_nodes WHERE digest = ANY(%s)",
            (sorted(digests),),
        )
        return {row["digest"]: row["byte_count"] for row in await cur.fetchall()}

    async def plan_snapshot_gc(self, request: AgentSnapshotGCRequest) -> AgentSnapshotGCPlan:
        validated = AgentSnapshotGCRequest.model_validate(request.model_dump(mode="json"))
        request_digest = _content_sha256(
            validated.model_dump(mode="json"), "agent_snapshot_gc_request"
        )
        async with self._transaction(write=True) as cur:
            await cur.execute(
                "SELECT plan_id, request_digest, plan_document "
                "FROM cayu_agent_snapshot_gc_plans WHERE operation_id = %s",
                (validated.operation_id,),
            )
            existing = await cur.fetchone()
            if existing is not None:
                if existing["request_digest"] != request_digest:
                    raise AgentSnapshotStoreConflict(
                        "Snapshot GC operation_id is already bound to another request."
                    )
                document = cast("str", existing["plan_document"])
            else:
                for access in validated.candidates:
                    await self._load_access_closure(cur, access)
                blocked: list[str] = []
                collectable: list[str] = []
                for root, accesses in _snapshot_accesses_by_root(validated.candidates).items():
                    authorized = {access.binding_id for access in accesses}
                    if await self._root_is_protected(cur, root) or authorized != (
                        await self._binding_ids(cur, root)
                    ):
                        blocked.append(root)
                    else:
                        collectable.append(root)
                deleting, retained = _gc_deletable_nodes(
                    set(collectable), await self._roots_by_node(cur, set(collectable))
                )
                sizes = await self._node_bytes(cur, deleting)
                if set(sizes) != deleting:
                    raise AgentSnapshotVerificationError(
                        "AgentSnapshot GC candidate contains a missing node."
                    )
                plan = _create_gc_plan(
                    operation_id=validated.operation_id,
                    authorized_candidates=validated.candidates,
                    collectable_roots=collectable,
                    blocked_roots=blocked,
                    node_digests_to_delete=deleting,
                    retained_shared_node_digests=retained,
                    bytes_to_delete=sum(sizes.values()),
                )
                document = _snapshot_model_json(plan)
                await cur.execute(
                    "INSERT INTO cayu_agent_snapshot_gc_plans "
                    "(operation_id, plan_id, request_digest, request_document, plan_document) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        validated.operation_id,
                        plan.plan_id,
                        request_digest,
                        _snapshot_model_json(validated),
                        document,
                    ),
                )
        return AgentSnapshotGCPlan.model_validate_json(document)

    async def execute_snapshot_gc(self, plan: AgentSnapshotGCPlan) -> AgentSnapshotGCReceipt:
        validated = AgentSnapshotGCPlan.model_validate(plan.model_dump(mode="json"))
        async with self._transaction(write=True) as cur:
            await cur.execute(
                "SELECT receipt_document FROM cayu_agent_snapshot_gc_receipts WHERE plan_id = %s",
                (validated.plan_id,),
            )
            receipt_row = await cur.fetchone()
            if receipt_row is not None:
                return AgentSnapshotGCReceipt.model_validate_json(receipt_row["receipt_document"])
            document = await self._execute_gc(cur, validated)
        return AgentSnapshotGCReceipt.model_validate_json(document)

    async def _execute_gc(self, cur: Any, validated: AgentSnapshotGCPlan) -> str:
        await cur.execute(
            "SELECT plan_id, plan_document FROM cayu_agent_snapshot_gc_plans "
            "WHERE operation_id = %s",
            (validated.operation_id,),
        )
        plan_row = await cur.fetchone()
        if plan_row is None:
            raise AgentSnapshotStoreConflict("Snapshot GC plan was not durably prepared.")
        try:
            stored = AgentSnapshotGCPlan.model_validate_json(cast("str", plan_row["plan_document"]))
        except Exception as error:
            raise AgentSnapshotStoreConflict("Stored snapshot GC plan is invalid.") from error
        if stored != validated or plan_row["plan_id"] != validated.plan_id:
            raise AgentSnapshotStoreConflict("Snapshot GC plan differs from durable state.")
        collectable = set(validated.collectable_roots)
        accesses_by_root = _snapshot_accesses_by_root(validated.authorized_candidates)
        for root in sorted(collectable):
            accesses = accesses_by_root[root]
            for access in accesses:
                await self._load_access_closure(cur, access)
            if {access.binding_id for access in accesses} != await self._binding_ids(cur, root):
                raise AgentSnapshotStoreConflict(
                    "Snapshot GC binding reachability changed after planning."
                )
            if await self._root_is_protected(cur, root):
                raise AgentSnapshotStoreConflict(
                    "Snapshot GC root gained protection after planning."
                )
            await cur.execute(
                "SELECT 1 AS present FROM cayu_agent_snapshot_roots WHERE snapshot_root = %s",
                (root,),
            )
            if await cur.fetchone() is None:
                raise AgentSnapshotStoreConflict("Snapshot GC root disappeared before execution.")
        deleting, _ = _gc_deletable_nodes(collectable, await self._roots_by_node(cur, collectable))
        if deleting != set(validated.node_digests_to_delete):
            raise AgentSnapshotStoreConflict("Snapshot GC reachability changed after planning.")
        sizes = await self._node_bytes(cur, deleting)
        if set(sizes) != deleting:
            raise AgentSnapshotStoreConflict("Snapshot GC node disappeared.")
        deleted_bytes = sum(sizes.values())
        if deleted_bytes != validated.bytes_to_delete:
            raise AgentSnapshotStoreConflict("Snapshot GC byte plan changed.")
        roots = sorted(collectable)
        if roots:
            await cur.execute(
                "DELETE FROM cayu_agent_snapshot_bindings WHERE snapshot_root = ANY(%s)",
                (roots,),
            )
            await cur.execute(
                "DELETE FROM cayu_agent_snapshot_records "
                "WHERE record_kind = 'snapshot' AND fingerprint = ANY(%s)",
                (roots,),
            )
            await cur.execute(
                "DELETE FROM cayu_agent_snapshot_root_nodes WHERE snapshot_root = ANY(%s)",
                (roots,),
            )
            await cur.execute(
                "DELETE FROM cayu_agent_snapshot_roots WHERE snapshot_root = ANY(%s)",
                (roots,),
            )
        if deleting:
            await cur.execute(
                "DELETE FROM cayu_agent_snapshot_nodes AS node WHERE node.digest = ANY(%s) "
                "AND NOT EXISTS (SELECT 1 FROM cayu_agent_snapshot_root_nodes AS link "
                "WHERE link.node_digest = node.digest)",
                (sorted(deleting),),
            )
        receipt = _create_gc_receipt(
            plan=validated,
            deleted_roots=collectable,
            deleted_node_digests=deleting,
            deleted_bytes=deleted_bytes,
        )
        document = _snapshot_model_json(receipt)
        await cur.execute(
            "INSERT INTO cayu_agent_snapshot_gc_receipts (plan_id, receipt_document) "
            "VALUES (%s, %s)",
            (validated.plan_id, document),
        )
        return document

    # -- content-addressed records ----------------------------------------------

    async def _save(
        self,
        kind: str,
        fingerprint: str,
        value: BaseModel,
        model: type[BaseModel],
    ) -> BaseModel:
        validated = _validate_store_record(value)
        if getattr(validated, "fingerprint", None) != fingerprint:
            raise AgentSnapshotStoreConflict(
                "Record fingerprint does not match its content-addressed key."
            )
        async with self._transaction(write=True) as cur:
            await cur.execute(
                "INSERT INTO cayu_agent_snapshot_records (record_kind, fingerprint, document) "
                "VALUES (%s, %s, %s) ON CONFLICT (record_kind, fingerprint) DO NOTHING",
                (kind, fingerprint, _snapshot_model_json(validated)),
            )
            stored_document = await self._load_record(cur, kind, fingerprint)
        if stored_document is None:
            raise RuntimeError("PostgreSQL failed to persist the snapshot record.")
        _require_saved_record(stored_document, validated, fingerprint)
        return model.model_validate_json(stored_document)

    async def _load_record(self, cur: Any, kind: str, fingerprint: str) -> str | None:
        await cur.execute(
            "SELECT document FROM cayu_agent_snapshot_records "
            "WHERE record_kind = %s AND fingerprint = %s",
            (kind, fingerprint),
        )
        row = await cur.fetchone()
        return None if row is None else cast("str", row["document"])

    async def _load(
        self,
        kind: str,
        fingerprint: str,
        model: type[BaseModel],
    ) -> BaseModel | None:
        _sha256_hex(fingerprint, "fingerprint")
        async with self._transaction(write=False) as cur:
            document = await self._load_record(cur, kind, fingerprint)
        if document is None:
            return None
        _require_loaded_record(document, model, fingerprint)
        return model.model_validate_json(document)

    async def save_snapshot(self, snapshot: AgentSnapshot) -> AgentSnapshot:
        await self.put_snapshot(snapshot, snapshot.identity_binding)
        return await self.get_snapshot(
            AgentSnapshotAccess(
                snapshot=snapshot.ref,
                binding_id=snapshot.identity_binding.binding_id,
                authority_scope_fingerprint=snapshot.identity_binding.authority_scope_fingerprint,
            )
        )

    async def load_snapshot(self, snapshot_root: str) -> AgentSnapshot | None:
        return cast(
            "AgentSnapshot | None", await self._load("snapshot", snapshot_root, AgentSnapshot)
        )

    async def save_trial(self, trial: AgentSnapshotTrialBinding) -> AgentSnapshotTrialBinding:
        return cast(
            "AgentSnapshotTrialBinding",
            await self._save("trial", trial.fingerprint, trial, AgentSnapshotTrialBinding),
        )

    async def load_trial(self, fingerprint: str) -> AgentSnapshotTrialBinding | None:
        return cast(
            "AgentSnapshotTrialBinding | None",
            await self._load("trial", fingerprint, AgentSnapshotTrialBinding),
        )

    async def save_result(self, result: AgentSnapshotResultBinding) -> AgentSnapshotResultBinding:
        return cast(
            "AgentSnapshotResultBinding",
            await self._save("result", result.fingerprint, result, AgentSnapshotResultBinding),
        )

    async def load_result(self, fingerprint: str) -> AgentSnapshotResultBinding | None:
        return cast(
            "AgentSnapshotResultBinding | None",
            await self._load("result", fingerprint, AgentSnapshotResultBinding),
        )

    # -- materialization --------------------------------------------------------

    async def begin_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterializationProgress:
        validated = cast("AgentSnapshotMaterializationProgress", _validate_store_record(progress))
        _require_initial_progress(validated)
        document = _snapshot_model_json(validated)
        scope = _scope_key(validated)
        async with self._transaction(write=True) as cur:
            await cur.execute(
                "INSERT INTO cayu_agent_snapshot_materialization_progress "
                f"({_PROGRESS_COLUMNS}) VALUES (%s, %s, %s, %s, %s, %s, %s, NULL) "
                "ON CONFLICT DO NOTHING",
                (*scope, validated.progress_id, validated.revision, document),
            )
            await cur.execute(_PROGRESS_BY_SCOPE, scope)
            row = await cur.fetchone()
        if row is None:
            raise RuntimeError("PostgreSQL failed to persist materialization progress.")
        stored = _validated_progress_row(row, expected_scope=scope)
        if row["progress_id"] != validated.progress_id or not _same_record_identity(
            stored, validated
        ):
            raise AgentSnapshotStoreConflict(
                "Materialization scope is already bound to another operation plan."
            )
        return stored

    async def load_materialization_progress_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterializationProgress | None:
        if type(request) is not AgentSnapshotMaterializationRequest:
            raise TypeError("request must be an AgentSnapshotMaterializationRequest.")
        scope = _scope_key(request)
        async with self._transaction(write=False) as cur:
            await cur.execute(_PROGRESS_BY_SCOPE, scope)
            row = await cur.fetchone()
        if row is None:
            return None
        return _validated_progress_row(row, expected_scope=scope)

    async def _transition(
        self,
        expected: AgentSnapshotMaterializationProgress,
        updated: AgentSnapshotMaterializationProgress,
    ) -> AgentSnapshotMaterializationProgress:
        expected = cast("AgentSnapshotMaterializationProgress", _validate_store_record(expected))
        updated = cast("AgentSnapshotMaterializationProgress", _validate_store_record(updated))
        if (
            expected.progress_id != updated.progress_id
            or _scope_key(expected) != _scope_key(updated)
            or updated.revision != expected.revision + 1
        ):
            raise AgentSnapshotStoreConflict("Invalid materialization progress transition.")
        document = _snapshot_model_json(updated)
        async with self._transaction(write=True) as cur:
            await cur.execute(
                _PROGRESS_TRANSITION,
                (
                    updated.revision,
                    document,
                    updated.materialization_fingerprint,
                    *_scope_key(expected),
                    expected.progress_id,
                    expected.revision,
                    _snapshot_model_json(expected),
                    expected.materialization_fingerprint,
                ),
            )
            if cur.rowcount != 1:
                raise AgentSnapshotStoreConflict("Materialization progress compare-and-set failed.")
        return AgentSnapshotMaterializationProgress.model_validate_json(document)

    async def claim_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
    ) -> AgentSnapshotMaterializationProgress:
        return await self._transition(progress, _claim_progress(progress, operation_id))

    async def complete_materialization_operation(
        self,
        progress: AgentSnapshotMaterializationProgress,
        operation_id: str,
        component: AgentSnapshotMaterializedComponent,
    ) -> AgentSnapshotMaterializationProgress:
        validated = AgentSnapshotMaterializedComponent.model_validate(
            component.model_dump(mode="json")
        )
        return await self._transition(
            progress, _complete_progress(progress, operation_id, validated)
        )

    async def _finalize(
        self,
        cur: Any,
        progress: AgentSnapshotMaterializationProgress,
        materialization: AgentSnapshotMaterialization,
    ) -> str:
        validated = cast("AgentSnapshotMaterialization", _validate_store_record(materialization))
        progress = cast("AgentSnapshotMaterializationProgress", _validate_store_record(progress))
        scope = _scope_key(progress)
        await cur.execute(_PROGRESS_BY_SCOPE, scope)
        current_row = await cur.fetchone()
        if current_row is None:
            raise AgentSnapshotStoreConflict("Materialization scope progress is missing.")
        current = _validated_progress_row(current_row, expected_scope=scope)
        if progress.materialization_fingerprint is not None:
            if current != progress or progress.materialization_fingerprint != validated.fingerprint:
                raise AgentSnapshotStoreConflict(
                    "Materialization scope is already finalized with another identity."
                )
            return await self._finalized_record(cur, validated)
        updated = _final_progress(progress, validated)
        await cur.execute(
            "INSERT INTO cayu_agent_snapshot_records (record_kind, fingerprint, document) "
            "VALUES ('materialization', %s, %s) ON CONFLICT (record_kind, fingerprint) DO NOTHING",
            (validated.fingerprint, _snapshot_model_json(validated)),
        )
        stored_document = await self._load_record(cur, "materialization", validated.fingerprint)
        if stored_document is None:
            raise RuntimeError("PostgreSQL failed to persist the materialization record.")
        try:
            stored = AgentSnapshotMaterialization.model_validate_json(stored_document)
        except Exception as error:
            raise AgentSnapshotStoreConflict(
                "Stored materialization is invalid for its fingerprint."
            ) from error
        if not _same_record_identity(stored, validated):
            raise AgentSnapshotStoreConflict(
                "Materialization fingerprint is bound to another logical identity."
            )
        await cur.execute(
            _PROGRESS_TRANSITION,
            (
                updated.revision,
                _snapshot_model_json(updated),
                validated.fingerprint,
                *scope,
                progress.progress_id,
                progress.revision,
                _snapshot_model_json(progress),
                None,
            ),
        )
        if cur.rowcount != 1 and current != updated:
            raise AgentSnapshotStoreConflict("Materialization finalization compare-and-set failed.")
        return stored_document

    async def _finalized_record(
        self,
        cur: Any,
        materialization: AgentSnapshotMaterialization,
    ) -> str:
        document = await self._load_record(cur, "materialization", materialization.fingerprint)
        if document is None:
            raise AgentSnapshotStoreConflict("Finalized materialization record is missing.")
        _require_loaded_record(document, AgentSnapshotMaterialization, materialization.fingerprint)
        existing = AgentSnapshotMaterialization.model_validate_json(document)
        if not _same_record_identity(existing, materialization):
            raise AgentSnapshotStoreConflict(
                "Finalized materialization record has a conflicting identity."
            )
        return document

    async def finalize_materialization(
        self,
        progress: AgentSnapshotMaterializationProgress,
        materialization: AgentSnapshotMaterialization,
    ) -> AgentSnapshotMaterialization:
        async with self._transaction(write=True) as cur:
            document = await self._finalize(cur, progress, materialization)
        return AgentSnapshotMaterialization.model_validate_json(document)

    async def save_materialization(
        self, materialization: AgentSnapshotMaterialization
    ) -> AgentSnapshotMaterialization:
        async with self._transaction(write=True) as cur:
            await cur.execute(
                f"SELECT {_PROGRESS_COLUMNS} FROM cayu_agent_snapshot_materialization_progress "
                "WHERE progress_id = %s",
                (materialization.progress_id,),
            )
            row = await cur.fetchone()
            if row is None:
                raise AgentSnapshotStoreConflict(
                    "Materialization has no durable pre-effect progress record."
                )
            progress = _validated_progress_row(row)
            if progress.materialization_fingerprint is None:
                document = await self._finalize(cur, progress, materialization)
            else:
                if progress.materialization_fingerprint != materialization.fingerprint:
                    raise AgentSnapshotStoreConflict(
                        "Materialization scope is already finalized with another identity."
                    )
                validated = cast(
                    "AgentSnapshotMaterialization", _validate_store_record(materialization)
                )
                _final_progress(progress, validated)
                document = await self._finalized_record(cur, validated)
        return AgentSnapshotMaterialization.model_validate_json(document)

    async def load_materialization(self, fingerprint: str) -> AgentSnapshotMaterialization | None:
        return cast(
            "AgentSnapshotMaterialization | None",
            await self._load("materialization", fingerprint, AgentSnapshotMaterialization),
        )

    async def load_materialization_for_scope(
        self,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterialization | None:
        if type(request) is not AgentSnapshotMaterializationRequest:
            raise TypeError("request must be an AgentSnapshotMaterializationRequest.")
        scope = _scope_key(request)
        async with self._transaction(write=False) as cur:
            await cur.execute(
                "SELECT progress.snapshot_fingerprint, progress.candidate_id, "
                "progress.state_scope_id, progress.state_mode, progress.progress_id, "
                "progress.revision, progress.document, "
                "progress.materialization_fingerprint, "
                "record.document AS materialization_document "
                "FROM cayu_agent_snapshot_materialization_progress AS progress "
                "LEFT JOIN cayu_agent_snapshot_records AS record "
                "ON record.record_kind = 'materialization' "
                "AND record.fingerprint = progress.materialization_fingerprint "
                "WHERE progress.snapshot_fingerprint = %s AND progress.candidate_id = %s "
                "AND progress.state_scope_id = %s AND progress.state_mode = %s",
                scope,
            )
            row = await cur.fetchone()
        if row is None:
            return None
        progress = _validated_progress_row(row, expected_scope=scope)
        if progress.materialization_fingerprint is None:
            return None
        document = row["materialization_document"]
        if document is None:
            raise AgentSnapshotStoreConflict(
                "Materialization scope points to a missing durable record."
            )
        try:
            materialization = AgentSnapshotMaterialization.model_validate_json(document)
        except Exception as error:
            raise AgentSnapshotStoreConflict(
                "Materialization scope points to an invalid durable record."
            ) from error
        _require_scope_materialization(request, progress, materialization)
        return AgentSnapshotMaterialization.model_validate_json(document)


__all__ = ["PostgresAgentSnapshotStore"]
