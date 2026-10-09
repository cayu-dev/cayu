"""Shared PostgreSQL pool lifecycle, schema validation and migration execution."""

from __future__ import annotations

import asyncio
import hmac
import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, LiteralString, cast

from cayu.storage import _postgres_budget_schema as postgres_budget_schema
from cayu.storage import _postgres_catalog as postgres_catalog
from cayu.storage import _postgres_eval_schema as postgres_eval_schema
from cayu.storage import _postgres_knowledge_schema as postgres_knowledge_schema
from cayu.storage import _postgres_memory_evidence_schema as postgres_memory_evidence_schema
from cayu.storage import _postgres_producer_schema as postgres_producer_schema
from cayu.storage import _postgres_session_schema as postgres_session_schema
from cayu.storage import _postgres_support as pg_support
from cayu.storage import _postgres_task_schema as postgres_task_schema
from cayu.storage import _postgres_transcript_schema as postgres_transcript_schema
from cayu.storage import _postgres_verified_work_schema as postgres_verified_work_schema
from cayu.storage import _postgres_work_context_schema as postgres_work_context_schema
from cayu.storage._phase_timing import PostgresTimingScope

try:
    from psycopg import sql
    from psycopg.errors import DeadlockDetected, DuplicateTable, UniqueViolation
    from psycopg_pool import AsyncConnectionPool
except ModuleNotFoundError as exc:
    raise RuntimeError(
        'Cayu\'s Postgres stores require the optional psycopg packages. Install them with `pip install "cayu[postgres]"`.'
    ) from exc
from cayu._validation import require_nonblank
from cayu.knowledge.records import MAX_KNOWLEDGE_CHUNK_ID_BYTES, MAX_KNOWLEDGE_ENTRY_ID_BYTES
from cayu.sessions.transcript_queries import TRANSCRIPT_SEARCH_TOKENIZER_VERSION
from cayu.storage import _postgres_schema_history as postgres_schema_history
from cayu.storage import migrations as schema
from cayu.storage._collaboration_schema import validate_postgres_collaboration_schema
from cayu.storage._collaboration_wait_schema import validate_postgres_wait_discovery
from cayu.storage._context_selection_schema import validate_postgres_context_selection_schema
from cayu.storage._diagnostic_inspection import current_diagnostic_store_inspection
from cayu.storage._participant_bindings_schema import validate_postgres_participant_bindings
from cayu.storage._product_operation_schema import validate_postgres_product_operation_schema
from cayu.storage.knowledge_transition import require_empty_knowledge_revision_transition
from cayu.tasks.handoff import TaskInterruptedHandoffRequest, prepare_interrupted_task_handoff
from cayu.tasks.records import Task, TaskStatus

# A fixed 63-bit advisory-lock key. Every Cayu store sharing a database takes this
# lock before touching schema, so concurrent creators/migrators (the production
# PostgresSessionStore + PostgresTaskStore on one pool) serialize: one runs the
# DDL, the rest wait and then validate (ADR 0001, Decision 4). The value is the
# ASCII bytes of "cayuschm" masked to stay positive (signed bigint); its only
# requirement is being a stable constant unlikely to collide with app locks.
_SCHEMA_ADVISORY_LOCK_KEY = 0x6361_7975_7363_686D & 0x7FFF_FFFF_FFFF_FFFF
_SCHEMA_ADVISORY_LOCK_POLL_SECONDS = 0.25
_POSTGRES_MIN_REQUIRED_REVISION = 18
_INTERRUPTED_HANDOFF_MIGRATION_BATCH_SIZE = 256


async def read_schema_state(cur: Any) -> schema.SchemaState:
    """Read the recorded schema state from an open cursor without applying DDL.

    Returns :data:`schema.UNINITIALIZED` (rather than raising) when the
    bookkeeping table is absent, so it is safe to call against any database.
    """
    # to_regclass returns NULL (not an error) when the table is absent, so an
    # uninitialized database reads as UNINITIALIZED rather than raising.
    await cur.execute("SELECT to_regclass('cayu_schema_migrations')")
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        return schema.SchemaState(revision=schema.UNINITIALIZED, compatible_from=0)
    await cur.execute(
        "SELECT revision, compatible_from FROM cayu_schema_migrations "
        "ORDER BY revision DESC LIMIT 1"
    )
    latest = await cur.fetchone()
    if latest is None:
        return schema.SchemaState(revision=schema.UNINITIALIZED, compatible_from=0)
    return schema.SchemaState(revision=latest[0], compatible_from=latest[1])


async def read_pending_migration_receipt(cur: Any) -> tuple[str, dict[str, object]] | None:
    """Read the one undelivered CLI migration receipt without applying DDL."""

    await cur.execute("SELECT to_regclass('cayu_schema_migration_receipts')")
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        return None
    await cur.execute(
        "SELECT operation_sha256, receipt_json "
        "FROM cayu_schema_migration_receipts WHERE singleton = TRUE"
    )
    row = await cur.fetchone()
    if row is None:
        return None
    value = json.loads(row[1]) if isinstance(row[1], str) else row[1]
    if not isinstance(value, dict):
        raise RuntimeError("Postgres durable migration receipt is invalid.")
    return str(row[0]), dict(value)


async def discard_pending_migration_receipt(cur: Any, operation_sha256: str) -> None:
    """Forget a CLI receipt only after its final rendering completed."""

    await cur.execute(
        "DELETE FROM cayu_schema_migration_receipts "
        "WHERE singleton = TRUE AND operation_sha256 = %s",
        (operation_sha256,),
    )


async def preflight_migration(
    cur: Any,
    state: schema.SchemaState | None = None,
    *,
    allow_empty_recall_reset: bool = False,
) -> schema.SchemaState:
    """Validate every Postgres clean break before migration bookkeeping DDL."""

    if state is None:
        state = await read_schema_state(cur)
    schema.validate_migration_input(state)
    current = state.revision
    planned = schema.pending(current)
    if (
        current != schema.UNINITIALIZED
        and current < 26
        and any(revision.revision == 26 for revision in planned)
    ):
        await _reject_populated_pre_interaction_database(cur)
    if (
        current != schema.UNINITIALIZED
        and current < 36
        and any(revision.revision == 36 for revision in planned)
    ):
        await _reject_populated_pre_invocation_database(cur)
    if (
        current != schema.UNINITIALIZED
        and current < 39
        and any(revision.revision == 39 for revision in planned)
    ):
        await _reject_populated_pre_task_invocation_database(cur)
    if (
        current != schema.UNINITIALIZED
        and current < 41
        and any(revision.revision == 41 for revision in planned)
    ):
        await _reject_populated_pre_knowledge_access_snapshot_database(cur)
    if current < 42 and any(revision.revision == 42 for revision in planned):
        await _reject_populated_pre_knowledge_revision_database(cur)
    if current < 46 and any(revision.revision == 46 for revision in planned):
        await _reject_populated_pre_transcript_search_database(cur)
    if (
        current != schema.UNINITIALIZED
        and current < 52
        and any(revision.revision == 52 for revision in planned)
    ):
        await _reject_populated_pre_targeted_tool_grant_database(cur)
    if (
        current != schema.UNINITIALIZED
        and current < 58
        and any(revision.revision == 58 for revision in planned)
    ):
        await _reject_populated_pre_verifier_profile_database(cur)
    if current < 59 and any(revision.revision == 59 for revision in planned):
        await _reject_populated_pre_result_resolver_database(cur)
    if current < 84 and any(revision.revision == 84 for revision in planned):
        await _reject_pre_worker_admission_history(cur)
    if current < 60 and any(revision.revision == 60 for revision in planned):
        await _reject_populated_pre_knowledge_relation_database(cur)
    if current < 63 and any(revision.revision == 63 for revision in planned):
        await _reject_populated_pre_knowledge_maintenance_database(cur)
    if current < 65 and any(revision.revision == 65 for revision in planned):
        await _reject_populated_pre_bounded_knowledge_entry_database(cur)
    if (
        current != schema.UNINITIALIZED
        and current < 73
        and any(revision.revision == 73 for revision in planned)
    ):
        if allow_empty_recall_reset:
            await preflight_empty_recall_state_reset(cur)
        else:
            await _reject_populated_pre_recall_subscription_database(cur)
    if (
        current != schema.UNINITIALIZED
        and current < 75
        and any(revision.revision == 75 for revision in planned)
    ):
        await _reject_populated_pre_knowledge_activation_database(cur)
    return state


async def _reject_pre_worker_admission_history(cur: Any) -> None:
    for table, authority in (
        ("cayu_work_attempt_admissions", "work-attempt admissions"),
        ("cayu_completion_verification_claims", "verification claims"),
    ):
        await cur.execute("SELECT to_regclass(%s) IS NOT NULL", (table,))
        row = await cur.fetchone()
        if row is None or row[0] is not True:
            continue
        await cur.execute(f"SELECT EXISTS(SELECT 1 FROM {table})")
        row = await cur.fetchone()
        if row is None or row[0] is not False:
            raise RuntimeError(
                "Postgres revision 84 cannot reconstruct executable settings for existing "
                f"{authority}. Recreate the pre-release database before migrating."
            )


async def _reject_populated_pre_interaction_database(cur: Any) -> None:
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 26 is a clean prerelease break and cannot migrate a "
            "populated Cayu session database. Recreate the Cayu database before "
            "starting this build."
        )


async def _reject_populated_pre_invocation_database(cur: Any) -> None:
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 36 requires invocation provenance for every session and "
            "cannot migrate a populated Cayu session database. Recreate the Cayu "
            "database before starting this build."
        )


async def _reject_populated_pre_targeted_tool_grant_database(cur: Any) -> None:
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_sessions)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 52 is a clean prerelease break and cannot migrate a "
            "populated Cayu session database. Recreate the Cayu database before "
            "starting this build."
        )


async def _reject_populated_pre_task_invocation_database(cur: Any) -> None:
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_tasks)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 39 requires invocation provenance for every task and "
            "cannot migrate a populated Cayu task database. Recreate the Cayu "
            "database before starting this build."
        )


_EMPTY_RECALL_RESET_TABLES = (
    "cayu_agent_recall_delivery_states",
    "cayu_agent_recall_delivery_releases",
    "cayu_agent_recall_delivery_claims",
    "cayu_agent_recall_deliveries",
    "cayu_agent_recall_checkpoint_heads",
    "cayu_agent_recall_checkpoints",
)


async def preflight_empty_recall_state_reset(cur: Any) -> None:
    """Lock and prove the prerelease recall tables contain no durable rows."""

    existing: list[str] = []
    for table in _EMPTY_RECALL_RESET_TABLES:
        await cur.execute("SELECT to_regclass(%s)", (table,))
        registered = await cur.fetchone()
        if registered is not None and registered[0] is not None:
            existing.append(table)
    if existing:
        await cur.execute(
            sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(
                sql.SQL(", ").join(sql.Identifier(table) for table in existing)
            )
        )
    for table in existing:
        await cur.execute(sql.SQL("SELECT EXISTS(SELECT 1 FROM {})").format(sql.Identifier(table)))
        row = await cur.fetchone()
        if row is not None and row[0] is True:
            raise schema.SchemaTooOld(
                "Storage revision 73 can rebuild prerelease recall state only when all "
                f"six checkpoint/delivery tables are empty; {table!r} is populated."
            )


async def reset_empty_recall_state(cur: Any) -> None:
    """Rebuild empty revision-69/71 recall tables from this Runtime's DDL."""

    await preflight_empty_recall_state_reset(cur)
    for table in _EMPTY_RECALL_RESET_TABLES:
        await cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table)))
    for revision in (69, 71):
        for statement in postgres_schema_history._MIGRATION_STEPS[revision]:
            await cur.execute(cast("LiteralString", statement))


async def _reject_populated_pre_recall_subscription_database(cur: Any) -> None:
    await cur.execute("SELECT to_regclass('cayu_agent_recall_checkpoints')")
    checkpoint_registered = await cur.fetchone()
    if checkpoint_registered is not None and checkpoint_registered[0] is not None:
        await cur.execute(
            """
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name = 'cayu_agent_recall_checkpoints'
              AND column_name = 'checkpoint_stream_id'
            """
        )
        if await cur.fetchone() is None:
            raise schema.SchemaTooOld(
                "Storage revision 73 introduces independent recall checkpoint streams and "
                "does not migrate the prerelease checkpoint schema. Recreate the Cayu "
                "database before starting this build."
            )
    await cur.execute("SELECT to_regclass('cayu_agent_recall_deliveries')")
    delivery_registered = await cur.fetchone()
    if delivery_registered is None or delivery_registered[0] is None:
        return
    await cur.execute("LOCK TABLE cayu_agent_recall_deliveries IN SHARE ROW EXCLUSIVE MODE")
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_agent_recall_deliveries)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 73 binds recall results to exact subscription input "
            "and cannot migrate a populated recall-delivery database without "
            "inventing missing retrieval authority. Recreate the Cayu database before "
            "starting this build."
        )


async def _reject_populated_pre_knowledge_access_snapshot_database(cur: Any) -> None:
    await cur.execute("SELECT to_regclass('cayu_knowledge_publication_receipts')")
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        return
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_knowledge_publication_receipts)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 41 requires an authorization snapshot for every "
            "knowledge publication receipt and cannot infer one for existing "
            "receipts. Recreate the Cayu database before starting this build."
        )


async def _reject_populated_pre_knowledge_revision_database(cur: Any) -> None:
    candidates = (
        "cayu_knowledge_entries",
        "cayu_knowledge_labels",
        "cayu_knowledge_aspects",
        "cayu_knowledge_impact_targets",
        "cayu_knowledge_chunks",
        "cayu_knowledge_publication_receipts",
        "cayu_knowledge_embeddings",
    )
    counts: dict[str, int] = {}
    for table in candidates:
        await cur.execute("SELECT to_regclass(%s)", (table,))
        registered = await cur.fetchone()
        if registered is None or registered[0] is None:
            continue
        await cur.execute(
            sql.SQL("SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)").format(sql.Identifier(table))
        )
        row = await cur.fetchone()
        counts[table] = 1 if row is not None and row[0] is True else 0
    if not counts:
        return
    require_empty_knowledge_revision_transition(
        counts,
        required_tables=counts,
    )


_KNOWLEDGE_RELATION_CLEAN_BREAK_TABLES = (
    "cayu_knowledge_entries",
    "cayu_knowledge_revisions",
    "cayu_knowledge_chunks",
    "cayu_knowledge_chunks_fts",
    "cayu_knowledge_labels",
    "cayu_knowledge_aspects",
    "cayu_knowledge_impact_targets",
    "cayu_knowledge_evidence",
    "cayu_knowledge_publication_receipts",
    "cayu_knowledge_relations",
    "cayu_knowledge_relation_publication_receipts",
    "cayu_knowledge_changes",
    "cayu_knowledge_change_audiences",
    "cayu_knowledge_change_labels",
    "cayu_knowledge_change_consumers",
    "cayu_knowledge_change_acknowledgements",
    "cayu_knowledge_index_readiness_events",
    "cayu_knowledge_index_readiness_current",
    "cayu_knowledge_embeddings",
)
_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES = (
    *_KNOWLEDGE_RELATION_CLEAN_BREAK_TABLES,
    "cayu_knowledge_maintenance_decisions",
    "cayu_knowledge_maintenance_proposals",
)
_KNOWLEDGE_ACTIVATION_CLEAN_BREAK_TABLES = (
    *_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES,
    "cayu_knowledge_activation_receipts",
    "cayu_knowledge_activation_retirements",
)


async def _reject_populated_pre_knowledge_relation_database(cur: Any) -> None:
    await _reject_populated_pre_knowledge_contract_database(
        cur,
        candidates=_KNOWLEDGE_RELATION_CLEAN_BREAK_TABLES,
        revision=60,
        contract="knowledge-lineage",
    )


async def _reject_populated_pre_knowledge_maintenance_database(cur: Any) -> None:
    await _reject_populated_pre_knowledge_contract_database(
        cur,
        candidates=_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES,
        revision=63,
        contract="reviewed-maintenance",
    )


async def _reject_populated_pre_bounded_knowledge_entry_database(cur: Any) -> None:
    await _reject_populated_pre_knowledge_contract_database(
        cur,
        candidates=_KNOWLEDGE_MAINTENANCE_CLEAN_BREAK_TABLES,
        revision=65,
        contract="bounded-entry-read",
    )


async def _reject_populated_pre_knowledge_activation_database(cur: Any) -> None:
    await _reject_populated_pre_knowledge_contract_database(
        cur,
        candidates=_KNOWLEDGE_ACTIVATION_CLEAN_BREAK_TABLES,
        revision=75,
        contract="knowledge-activation-authority",
    )


async def _reject_populated_pre_knowledge_contract_database(
    cur: Any,
    *,
    candidates: tuple[str, ...],
    revision: int,
    contract: str,
) -> None:
    existing: list[str] = []
    for table in candidates:
        await cur.execute("SELECT to_regclass(%s)", (table,))
        registered = await cur.fetchone()
        if registered is None or registered[0] is None:
            continue
        existing.append(table)
    if existing:
        await cur.execute(
            sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(
                sql.SQL(", ").join(sql.Identifier(table) for table in sorted(existing))
            )
        )
    for table in existing:
        await cur.execute(
            sql.SQL("SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)").format(sql.Identifier(table))
        )
        row = await cur.fetchone()
        if row is not None and row[0] is True:
            raise schema.SchemaTooOld(
                f"Storage revision {revision} is a clean prerelease {contract} break "
                "and cannot migrate a populated Cayu knowledge database. Recreate "
                "the Cayu knowledge database before starting this build."
            )


async def _reject_populated_pre_transcript_search_database(cur: Any) -> None:
    await cur.execute("SELECT to_regclass('cayu_transcript_messages')")
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        return
    # Fence transcript writers through the revision transaction so a clean
    # preflight cannot race an old writer before the non-null column is added.
    await cur.execute("LOCK TABLE cayu_transcript_messages IN SHARE ROW EXCLUSIVE MODE")
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_transcript_messages)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 46 requires the final transcript-search projection "
            "on every transcript row and deliberately does not backfill earlier "
            "data. Recreate the Cayu database before starting this build."
        )


async def _reject_populated_pre_verifier_profile_database(cur: Any) -> None:
    for table in (
        "cayu_completion_verification_claims",
        "cayu_completion_decisions",
    ):
        await cur.execute("SELECT to_regclass(current_schema() || %s)", (f".{table}",))
        registered = await cur.fetchone()
        if registered is None or registered[0] is None:
            continue
        await cur.execute(
            sql.SQL("SELECT EXISTS(SELECT 1 FROM {} LIMIT 1)").format(sql.Identifier(table))
        )
        row = await cur.fetchone()
        if row is not None and row[0] is True:
            raise RuntimeError(
                "Postgres migration revision 58 cannot attribute existing completion-"
                "verification records to immutable verifier profiles. Recreate the "
                "pre-release database before migrating."
            )


async def _reject_populated_pre_result_resolver_database(cur: Any) -> None:
    await cur.execute("SELECT to_regclass('cayu_work_contracts')")
    registered = await cur.fetchone()
    if registered is None or registered[0] is None:
        return
    await cur.execute("LOCK TABLE cayu_work_contracts IN SHARE ROW EXCLUSIVE MODE")
    await cur.execute("SELECT EXISTS(SELECT 1 FROM cayu_work_contracts)")
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 59 requires an exact result-resolver identity for every "
            "verified-work contract and cannot infer one for existing contracts. "
            "Recreate the Cayu task database before starting this build."
        )


async def _reject_revision_43_knowledge_identity_overflow(cur: Any) -> None:
    await cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM cayu_knowledge_entries
            WHERE octet_length(id) > %s
            UNION ALL
            SELECT 1
            FROM cayu_knowledge_chunks
            WHERE octet_length(id) > %s
        )
        """,
        (MAX_KNOWLEDGE_ENTRY_ID_BYTES, MAX_KNOWLEDGE_CHUNK_ID_BYTES),
    )
    row = await cur.fetchone()
    if row is not None and row[0] is True:
        raise schema.SchemaTooOld(
            "Storage revision 43 bounds knowledge entry and chunk identities for "
            "portable indexed storage. Shorten out-of-contract revision-42 identities "
            "or recreate the Cayu database before migration."
        )


async def _disable_prepared_statements(conn: Any) -> None:
    """Pool ``configure`` hook: disable psycopg3 server-side prepared statements.

    Required when the store's own pool runs behind a transaction-pooling pgbouncer
    (e.g. Fly Managed Postgres), where prepared statements raise
    "prepared statement ... already exists". Harmless on a direct connection.
    """
    conn.prepare_threshold = None


async def _configure_store_connection(conn: Any) -> None:
    await _disable_prepared_statements(conn)


async def _acquire_schema_transaction_lock(
    conn: Any,
    cur: Any,
    *,
    read_only: bool = False,
) -> None:
    """Acquire the schema lock without leaving a waiting transaction open.

    Concurrent index DDL holds the same key as a session advisory lock. A
    blocking transaction-lock request would retain its virtual transaction ID
    while waiting, which ``CREATE INDEX CONCURRENTLY`` can in turn wait on and
    deadlock. End every unsuccessful try before polling again.
    """
    while True:
        await cur.execute(
            "SELECT pg_try_advisory_xact_lock(%s)",
            (_SCHEMA_ADVISORY_LOCK_KEY,),
        )
        row = await cur.fetchone()
        if row is not None and row[0] is True:
            return
        await conn.rollback()
        if read_only:
            await conn.execute("SET TRANSACTION READ ONLY")
        await asyncio.sleep(_SCHEMA_ADVISORY_LOCK_POLL_SECONDS)


class _PostgresStoreBase:
    """Shared async connection-pool management for Postgres-backed stores.

    The pool is created eagerly (closed) and opened lazily on first use so that
    it is bound to the event loop that actually drives the store. This mirrors
    the way the SQLite store opens its connection in ``__init__`` while keeping
    psycopg's async pool happy about running inside a live loop.
    """

    _min_required_revision = _POSTGRES_MIN_REQUIRED_REVISION
    _supports_read_only = False

    def __init__(
        self,
        conninfo: str | None = None,
        *,
        pool: AsyncConnectionPool | None = None,
        min_size: int = 1,
        max_size: int = 8,
        schema_mode: schema.SchemaMode = schema.SchemaMode.VALIDATE,
        read_only: bool = False,
        migration_reset_empty_recall_state: bool = False,
        migration_expected_input_state: schema.SchemaState | None = None,
        migration_operation_sha256: str | None = None,
        migration_receipt_json: str | None = None,
    ) -> None:
        if not isinstance(schema_mode, schema.SchemaMode):
            raise TypeError("schema_mode must be a SchemaMode.")
        if type(read_only) is not bool:
            raise TypeError("read_only must be a bool.")
        diagnostic_inspection = current_diagnostic_store_inspection() is not None
        if diagnostic_inspection:
            schema_mode = schema.SchemaMode.VALIDATE
            read_only = True
        if type(migration_reset_empty_recall_state) is not bool:
            raise TypeError("migration_reset_empty_recall_state must be a bool.")
        if migration_reset_empty_recall_state and schema_mode is not schema.SchemaMode.MIGRATE:
            raise ValueError("migration_reset_empty_recall_state requires schema_mode=MIGRATE.")
        if (migration_expected_input_state is None) != (migration_operation_sha256 is None):
            raise ValueError(
                "migration_expected_input_state and migration_operation_sha256 "
                "must be provided together."
            )
        if migration_expected_input_state is not None:
            if not isinstance(migration_expected_input_state, schema.SchemaState):
                raise TypeError("migration_expected_input_state must be a SchemaState.")
            if schema_mode is not schema.SchemaMode.MIGRATE:
                raise ValueError("migration_expected_input_state requires schema_mode=MIGRATE.")
        if (
            migration_operation_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", migration_operation_sha256) is None
        ):
            raise ValueError("migration_operation_sha256 must be a lowercase SHA-256 digest.")
        if migration_receipt_json is not None:
            if type(migration_receipt_json) is not str:
                raise TypeError("migration_receipt_json must be a string.")
            if migration_operation_sha256 is None:
                raise ValueError("migration_receipt_json requires migration_operation_sha256.")
            try:
                decoded_receipt = json.loads(migration_receipt_json)
            except json.JSONDecodeError as exc:
                raise ValueError("migration_receipt_json must be valid JSON.") from exc
            if not isinstance(decoded_receipt, dict):
                raise ValueError("migration_receipt_json must encode a JSON object.")
            migration_receipt_json = json.dumps(
                decoded_receipt,
                sort_keys=True,
                separators=(",", ":"),
            )
        if read_only and not self._supports_read_only and not diagnostic_inspection:
            raise ValueError("read_only is only supported by PostgresSessionStore.")
        if read_only and schema_mode is not schema.SchemaMode.VALIDATE:
            raise ValueError("Read-only Postgres stores require schema_mode=VALIDATE.")
        self._schema_mode = schema_mode
        self._read_only = read_only
        self._migration_reset_empty_recall_state = migration_reset_empty_recall_state
        self._migration_expected_input_state = migration_expected_input_state
        self._migration_operation_sha256 = migration_operation_sha256
        self._migration_receipt_json = migration_receipt_json
        if pool is not None:
            if read_only:
                raise ValueError("read_only requires a store-owned Postgres connection pool.")
            if not isinstance(pool, AsyncConnectionPool):
                raise TypeError("pool must be an AsyncConnectionPool.")
            self._pool = pool
            self._owns_pool = False
            self._conninfo = None
        else:
            if type(conninfo) is not str:
                raise TypeError("conninfo must be a string.")
            self._conninfo = require_nonblank(conninfo, "conninfo")
            self._pool = AsyncConnectionPool(
                self._conninfo,
                min_size=min_size,
                max_size=max_size,
                open=False,
                # Disable server-side prepared statements so the store works behind
                # a transaction-pooling pgbouncer (e.g. Fly Managed Postgres), where
                # prepared statements raise "prepared statement already exists".
                configure=_configure_store_connection,
            )
            self._owns_pool = True
        self._open_lock = asyncio.Lock()
        self._opened = False
        self._schema_ready = False

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[Any]:
        async with PostgresTimingScope(self._pool.connection()) as conn:
            if self._read_only:
                # Keep the guard in the same transaction as the store operation.
                # Session defaults are not stable behind transaction-pooled PgBouncer.
                await conn.execute("SET TRANSACTION READ ONLY")
            yield conn

    @asynccontextmanager
    async def _ready_connection(self) -> AsyncIterator[Any]:
        """Acquire an operation connection after opening and validating the store."""
        await self._ensure_ready()
        async with self._connection() as conn:
            yield conn

    async def ensure_schema(self) -> None:
        """Open the pool and reconcile the schema now (per ``schema_mode``).

        Normally reconciliation happens lazily on first use; the ``cayu storage``
        CLI calls this to run a ``migrate`` (or ``validate``) as an explicit step.
        """
        await self._ensure_ready()

    async def _ensure_ready(self) -> None:
        if self._opened and self._schema_ready:
            return
        async with self._open_lock:
            if not self._opened:
                await self._pool.open()
                self._opened = True
            if not self._schema_ready:
                await self._reconcile_schema()
                self._schema_ready = True

    async def _reconcile_schema(self) -> None:
        """Reconcile the database schema with this binary per ``schema_mode``.

        Concurrent stores serialize transactional schema work on one
        transaction-scoped advisory lock (ADR 0001, Decision 4). Concurrent index
        DDL necessarily runs outside that transaction, so it polls the same key as
        a short-lived session advisory lock while validating or building each
        index. The lock is held only on this dedicated migration connection, which
        keeps normal store traffic safe behind transaction-pooled PgBouncer:

        - ``validate``: read the recorded revision and fail fast unless this binary
          can operate against it. Never runs DDL.
        - ``create``: initialize the baseline schema on an empty database; otherwise
          validate. The dev/test/local default.
        - ``migrate``: apply pending forward revisions under the lock, then validate.
        """
        if self._schema_mode is schema.SchemaMode.MIGRATE:
            await self._migrate_schema()
            return

        async with self._connection() as conn:
            async with conn.cursor() as cur:
                await _acquire_schema_transaction_lock(
                    conn,
                    cur,
                    read_only=self._read_only,
                )
                if self._schema_mode is not schema.SchemaMode.VALIDATE:
                    await cur.execute(postgres_schema_history.MIGRATIONS_TABLE_DDL)
                state = await self._read_schema_state(cur)
                if self._schema_mode is schema.SchemaMode.VALIDATE:
                    schema.validate(
                        state,
                        app_min_supported=self._min_required_revision,
                    )
                    await self._validate_postgres_schema(cur, state)
                elif self._schema_mode is schema.SchemaMode.CREATE:
                    if state.revision == schema.UNINITIALIZED:
                        await self._apply_pending(cur, state)
                    else:
                        schema.validate(
                            state,
                            app_min_supported=self._min_required_revision,
                        )
                        await self._validate_postgres_schema(cur, state)
            await conn.commit()

    async def _migrate_schema(self) -> None:
        relation_preflight_complete = False
        maintenance_preflight_complete = False
        bounded_entry_preflight_complete = False
        activation_preflight_complete = False
        empty_recall_reset_complete = False
        while True:
            concurrent_revision: schema.Revision | None = None
            concurrent_indexes: tuple[postgres_schema_history._ConcurrentIndexMigration, ...] = ()
            recorded_indexes: tuple[postgres_schema_history._ConcurrentIndexMigration, ...] = ()
            async with PostgresTimingScope(self._pool.connection()) as conn:
                async with conn.cursor() as cur:
                    await _acquire_schema_transaction_lock(conn, cur)
                    # Resolve the exact input and every clean-break refusal
                    # before even creating migration bookkeeping. Revision
                    # transactions repeat the checks that need a writer fence.
                    await preflight_migration(
                        cur,
                        allow_empty_recall_reset=(self._migration_reset_empty_recall_state),
                    )
                    await self._preflight_migration_authority(cur)
                    await cur.execute(postgres_schema_history.MIGRATIONS_TABLE_DDL)
                    state = await self._read_schema_state(cur)
                    await self._validate_migration_operation_progress(cur, state)
                    await self._persist_migration_receipt(cur)
                    current = state.revision
                    if (
                        self._migration_reset_empty_recall_state
                        and not empty_recall_reset_complete
                        and 69 <= current < 73
                        and any(revision.revision == 73 for revision in schema.pending(current))
                    ):
                        await reset_empty_recall_state(cur)
                        empty_recall_reset_complete = True
                    if (
                        current != schema.UNINITIALIZED
                        and current < 26
                        and any(revision.revision == 26 for revision in schema.pending(current))
                    ):
                        # Reject before applying any earlier pending revision so
                        # the clean break cannot leave a database half-migrated.
                        await _reject_populated_pre_interaction_database(cur)
                    if (
                        current != schema.UNINITIALIZED
                        and current < 36
                        and any(revision.revision == 36 for revision in schema.pending(current))
                    ):
                        await _reject_populated_pre_invocation_database(cur)
                    if (
                        current != schema.UNINITIALIZED
                        and current < 39
                        and any(revision.revision == 39 for revision in schema.pending(current))
                    ):
                        await _reject_populated_pre_task_invocation_database(cur)
                    if (
                        current != schema.UNINITIALIZED
                        and current < 41
                        and any(revision.revision == 41 for revision in schema.pending(current))
                    ):
                        await _reject_populated_pre_knowledge_access_snapshot_database(cur)
                    if current < 42 and any(
                        revision.revision == 42 for revision in schema.pending(current)
                    ):
                        await _reject_populated_pre_knowledge_revision_database(cur)
                    if current < 46 and any(
                        revision.revision == 46 for revision in schema.pending(current)
                    ):
                        await _reject_populated_pre_transcript_search_database(cur)
                    if (
                        current != schema.UNINITIALIZED
                        and current < 52
                        and any(revision.revision == 52 for revision in schema.pending(current))
                    ):
                        await _reject_populated_pre_targeted_tool_grant_database(cur)
                    if (
                        current != schema.UNINITIALIZED
                        and current < 58
                        and any(revision.revision == 58 for revision in schema.pending(current))
                    ):
                        # Reject before applying any earlier pending revision so the
                        # clean break cannot leave a database half-migrated.
                        await _reject_populated_pre_verifier_profile_database(cur)
                    if current < 59 and any(
                        revision.revision == 59 for revision in schema.pending(current)
                    ):
                        await _reject_populated_pre_result_resolver_database(cur)
                    if current < 84 and any(
                        revision.revision == 84 for revision in schema.pending(current)
                    ):
                        await _reject_pre_worker_admission_history(cur)
                    if (
                        current < 60
                        and any(revision.revision == 60 for revision in schema.pending(current))
                        and (not relation_preflight_complete or current == 59)
                    ):
                        await _reject_populated_pre_knowledge_relation_database(cur)
                        relation_preflight_complete = True
                    if (
                        current < 63
                        and any(revision.revision == 63 for revision in schema.pending(current))
                        and (not maintenance_preflight_complete or current == 62)
                    ):
                        await _reject_populated_pre_knowledge_maintenance_database(cur)
                        maintenance_preflight_complete = True
                    if (
                        current < 65
                        and any(revision.revision == 65 for revision in schema.pending(current))
                        and (not bounded_entry_preflight_complete or current == 64)
                    ):
                        await _reject_populated_pre_bounded_knowledge_entry_database(cur)
                        bounded_entry_preflight_complete = True
                    if (
                        current != schema.UNINITIALIZED
                        and current < 73
                        and any(revision.revision == 73 for revision in schema.pending(current))
                    ):
                        await _reject_populated_pre_recall_subscription_database(cur)
                    if (
                        current != schema.UNINITIALIZED
                        and current < 75
                        and any(revision.revision == 75 for revision in schema.pending(current))
                        and (not activation_preflight_complete or current == 74)
                    ):
                        await _reject_populated_pre_knowledge_activation_database(cur)
                        activation_preflight_complete = True
                    if current == schema.UNINITIALIZED:
                        await self._apply_baseline(cur)
                        current = schema.BASELINE_REVISION
                    pending = schema.pending(current)
                    if not pending:
                        current_state = await self._read_schema_state(cur)
                        schema.validate(
                            current_state,
                            app_min_supported=self._min_required_revision,
                        )
                        self._validate_postgres_revision(current_state)
                        if current_state.revision >= 108:
                            await validate_postgres_context_selection_schema(cur)
                        if current_state.revision >= 110:
                            await postgres_producer_schema._validate_producer_cleanup_receipts(cur)
                        if current_state.revision >= 96:
                            await validate_postgres_participant_bindings(cur)
                        if current_state.revision >= 111:
                            await validate_postgres_wait_discovery(cur)
                        if current_state.revision >= 112:
                            await validate_postgres_product_operation_schema(cur)
                        if self._min_required_revision >= 36:
                            await postgres_session_schema._validate_session_invocation_column(cur)
                        if self._min_required_revision >= 38:
                            await postgres_task_schema._validate_task_terminalization_receipt_table(
                                cur
                            )
                        if self._min_required_revision >= 39:
                            await postgres_task_schema._validate_task_invocation_column(cur)
                        if self._min_required_revision >= 41:
                            await postgres_knowledge_schema._validate_knowledge_publication_access_snapshot_column(
                                cur
                            )
                        if self._min_required_revision >= 42:
                            await postgres_knowledge_schema._validate_knowledge_revision_schema(
                                cur,
                                allow_revision_43=current_state.revision >= 43,
                                require_payload_bytes=current_state.revision >= 65,
                            )
                        if self._min_required_revision >= 43:
                            await postgres_knowledge_schema._validate_knowledge_change_schema(
                                cur,
                                relation_aware=current_state.revision >= 60,
                            )
                        if self._min_required_revision >= 44:
                            await postgres_knowledge_schema._validate_knowledge_index_readiness_schema(
                                cur
                            )
                        if current_state.revision >= 60:
                            await postgres_knowledge_schema._validate_knowledge_relation_schema(cur)
                        if current_state.revision >= 63:
                            await postgres_knowledge_schema._validate_knowledge_maintenance_schema(
                                cur
                            )
                        if current_state.revision >= 67:
                            await postgres_knowledge_schema._validate_knowledge_maintenance_proposal_schema(
                                cur
                            )
                        if current_state.revision >= 69:
                            await postgres_work_context_schema._validate_agent_work_context_schema(
                                cur
                            )
                        if current_state.revision >= 71:
                            await (
                                postgres_work_context_schema._validate_agent_recall_delivery_schema(
                                    cur,
                                    require_processing_schema_version=(
                                        current_state.revision >= 73
                                    ),
                                )
                            )
                        if current_state.revision >= 73:
                            await postgres_work_context_schema._validate_agent_recall_subscription_schema(
                                cur
                            )
                        if self._min_required_revision >= 45:
                            await postgres_task_schema._validate_task_retry_series_schema(cur)
                        if self._min_required_revision >= 46:
                            await postgres_transcript_schema._validate_transcript_search_document_column(
                                cur, tokenizer_version=TRANSCRIPT_SEARCH_TOKENIZER_VERSION
                            )
                        if self._min_required_revision >= 47:
                            await postgres_eval_schema._validate_eval_result_baseline_schema(cur)
                        if self._min_required_revision >= 48:
                            await postgres_eval_schema._validate_captured_eval_case_schema(cur)
                        if self._min_required_revision >= 49:
                            await postgres_verified_work_schema._validate_verified_work_schema(
                                cur,
                                require_verifier_profiles=current_state.revision >= 58,
                            )
                        if self._min_required_revision >= 50:
                            await postgres_eval_schema._validate_eval_run_invocation_column(cur)
                        if self._min_required_revision >= 51:
                            await postgres_memory_evidence_schema._validate_memory_evidence_schema(
                                cur
                            )
                        if self._min_required_revision >= 52:
                            await postgres_session_schema._validate_targeted_tool_grant_schema(cur)
                        if self._min_required_revision >= 53:
                            await postgres_eval_schema._validate_eval_scenario_schema(cur)
                        if self._min_required_revision >= 55:
                            await postgres_task_schema._validate_task_retry_reconciliation_schema(
                                cur
                            )
                        if self._min_required_revision >= 56:
                            await postgres_eval_schema._validate_eval_run_scenario_progress_column(
                                cur
                            )
                        if self._min_required_revision >= 57:
                            await postgres_session_schema._validate_session_message_queue_typed_message_column(
                                cur
                            )
                        if self._min_required_revision >= 83:
                            await (
                                postgres_session_schema._validate_session_message_lifecycle_columns(
                                    cur
                                )
                            )
                        if self._min_required_revision >= 59:
                            await postgres_session_schema._validate_session_instance_schema(cur)
                        if self._min_required_revision >= 61:
                            await postgres_verified_work_schema._validate_work_attempt_admission_schema(
                                cur
                            )
                        if self._min_required_revision >= 84:
                            await postgres_verified_work_schema._validate_work_attempt_lifecycle_schema(
                                cur
                            )
                        if self._min_required_revision >= 62:
                            await postgres_verified_work_schema._validate_deferred_interaction_input_payloads(
                                cur
                            )
                            await postgres_verified_work_schema._validate_work_attempt_continuation_authority(
                                cur
                            )
                        if self._min_required_revision >= 64:
                            await postgres_eval_schema._validate_eval_authored_suite_schema(cur)
                        if self._min_required_revision >= 66:
                            await postgres_task_schema._validate_local_execution_attempt_schema(cur)
                        if self._min_required_revision >= 70:
                            await postgres_task_schema._validate_interrupted_task_handoff_schema(
                                cur
                            )
                        if self._min_required_revision >= 74:
                            await postgres_eval_schema._validate_eval_run_trial_checkpoint_schema(
                                cur
                            )
                        if self._min_required_revision >= 75:
                            await postgres_knowledge_schema._validate_knowledge_activation_schema(
                                cur
                            )
                        if self._min_required_revision >= 76:
                            await postgres_task_schema._validate_interrupted_handoff_generation_column(
                                cur
                            )
                        if self._min_required_revision >= 77:
                            await postgres_knowledge_schema._validate_knowledge_maintenance_governance_schema(
                                cur
                            )
                        if self._min_required_revision >= 78:
                            await (
                                postgres_knowledge_schema._validate_knowledge_semantic_watch_schema(
                                    cur
                                )
                            )
                        if self._min_required_revision >= 88:
                            await postgres_task_schema._validate_task_closure_guard(
                                cur,
                                expected_guard_sql=postgres_schema_history._MIGRATION_STEPS[88][1],
                            )
                        if current_state.revision >= 23:
                            await postgres_budget_schema._validate_budget_reservation_identity_registry(
                                cur,
                                require=True,
                            )
                        if current_state.revision >= 28:
                            await postgres_session_schema._validate_public_authority_alias_registry(
                                cur
                            )
                        recorded_indexes = postgres_schema_history._required_concurrent_indexes(
                            current_state.revision
                        )
                    else:
                        revision = pending[0]
                        if revision.revision == 43:
                            await _reject_revision_43_knowledge_identity_overflow(cur)
                        if revision.revision == 65:
                            await _reject_populated_pre_bounded_knowledge_entry_database(cur)
                        if revision.revision == 73:
                            await _reject_populated_pre_recall_subscription_database(cur)
                        if revision.revision == 75:
                            await _reject_populated_pre_knowledge_activation_database(cur)
                        concurrent_indexes = (
                            postgres_schema_history._CONCURRENT_INDEX_MIGRATIONS.get(
                                revision.revision,
                                (),
                            )
                        )
                        if concurrent_indexes:
                            # A revision may pair small transactional objects
                            # with hot-table indexes that must be built outside a
                            # transaction. Record it only after both phases pass.
                            for statement in postgres_schema_history._MIGRATION_STEPS.get(
                                revision.revision, ()
                            ):
                                await cur.execute(cast("LiteralString", statement))
                            concurrent_revision = revision
                        else:
                            for statement in postgres_schema_history._MIGRATION_STEPS.get(
                                revision.revision, ()
                            ):
                                await cur.execute(cast("LiteralString", statement))
                            await self._validate_revision_schema_objects(cur, revision)
                            await self._record_revision(cur, revision)
                await conn.commit()
            if concurrent_revision is None:
                if not pending:
                    for index in recorded_indexes:
                        async with PostgresTimingScope(self._pool.connection()) as conn:
                            await self._ensure_concurrent_index(conn, index)
                    return
                continue

            if concurrent_revision.revision == 17:
                await self._backfill_revision_seventeen()

            for index in concurrent_indexes:
                async with PostgresTimingScope(self._pool.connection()) as conn:
                    await self._ensure_concurrent_index(
                        conn,
                        index,
                        pending_revision=concurrent_revision.revision,
                    )

            # Record the revision only after every non-transactional object is
            # valid. A competing migrator may have recorded it while this process
            # built or waited for the same index, so re-read under the xact lock.
            async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
                await _acquire_schema_transaction_lock(conn, cur)
                await self._preflight_migration_authority(cur)
                state = await self._read_schema_state(cur)
                await self._validate_migration_operation_progress(cur, state)
                if state.revision < concurrent_revision.revision:
                    await self._validate_revision_schema_objects(cur, concurrent_revision)
                    if concurrent_revision.revision == 23:
                        await postgres_budget_schema._validate_budget_reservation_identity_registry(
                            cur,
                            require=True,
                            verify_event_ownership=True,
                        )
                    await self._record_revision(cur, concurrent_revision)
                await conn.commit()

    async def _preflight_migration_authority(self, cur: Any) -> None:
        """Validate backend-specific migration authority under the schema fence."""

    async def _persist_migration_receipt(self, cur: Any) -> None:
        receipt_json = self._migration_receipt_json
        operation_sha256 = self._migration_operation_sha256
        if receipt_json is None or operation_sha256 is None:
            return
        await cur.execute(postgres_schema_history.MIGRATION_RECEIPTS_TABLE_DDL)
        await cur.execute(
            "INSERT INTO cayu_schema_migration_receipts "
            "(singleton, operation_sha256, receipt_json) "
            "VALUES (TRUE, %s, %s::jsonb) ON CONFLICT (singleton) DO NOTHING",
            (operation_sha256, receipt_json),
        )
        await cur.execute(
            "SELECT operation_sha256, receipt_json "
            "FROM cayu_schema_migration_receipts WHERE singleton = TRUE"
        )
        row = await cur.fetchone()
        if row is None:
            raise RuntimeError("Postgres durable migration receipt was not persisted.")
        observed = json.loads(row[1]) if isinstance(row[1], str) else row[1]
        observed_json = json.dumps(observed, sort_keys=True, separators=(",", ":"))
        if not hmac.compare_digest(str(row[0]), operation_sha256) or not hmac.compare_digest(
            observed_json,
            receipt_json,
        ):
            raise RuntimeError(
                "Postgres has an undelivered receipt for another migration operation; "
                "recover that receipt before starting a new migration."
            )

    async def _validate_migration_operation_progress(
        self,
        cur: Any,
        state: schema.SchemaState,
    ) -> None:
        expected = self._migration_expected_input_state
        operation_sha256 = self._migration_operation_sha256
        if expected is None or operation_sha256 is None:
            return
        if state.revision < expected.revision or (
            state.revision == expected.revision
            and state.compatible_from != expected.compatible_from
        ):
            raise RuntimeError(
                "Postgres migration input changed after preflight; retry from the new revision."
            )
        if state.revision == expected.revision:
            return

        expected_revisions = tuple(
            revision.revision
            for revision in schema.pending(expected.revision)
            if revision.revision <= state.revision
        )
        await cur.execute(
            "SELECT revision, checksum FROM cayu_schema_migrations "
            "WHERE revision > %s AND revision <= %s ORDER BY revision",
            (expected.revision, state.revision),
        )
        observed = tuple((int(row[0]), row[1]) for row in await cur.fetchall())
        if tuple(revision for revision, _checksum in observed) != expected_revisions or any(
            checksum != operation_sha256 for _revision, checksum in observed
        ):
            raise RuntimeError(
                "Postgres migration input changed after preflight: committed revision "
                "progress belongs to another migration operation. Retry from the new revision."
            )

    async def _backfill_revision_seventeen(self) -> None:
        await self._run_resumable_checkpoint_backfill(
            postgres_schema_history._REVISION_17_CHECKPOINT_BACKFILL_SQL,
            "SELECT EXISTS(SELECT 1 FROM cayu_checkpoints WHERE NOT pending_action_metrics_ready)",
        )
        await self._run_resumable_sequence_backfill(
            postgres_schema_history._REVISION_17_EVENT_BACKFILL_SMALL_SQL,
            postgres_schema_history._REVISION_17_EVENT_BACKFILL_SMALL_REMAINING_SQL,
        )
        await self._run_resumable_sequence_backfill(
            postgres_schema_history._REVISION_17_EVENT_BACKFILL_LARGE_SQL,
            postgres_schema_history._REVISION_17_EVENT_BACKFILL_LARGE_REMAINING_SQL,
        )

    async def _run_resumable_checkpoint_backfill(
        self,
        batch_sql: str,
        remaining_sql: str,
    ) -> None:
        after_session_id: str | None = None
        while True:
            async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
                await cur.execute(
                    cast("LiteralString", batch_sql),
                    (after_session_id, after_session_id),
                )
                updated = await cur.fetchall()
                if updated:
                    after_session_id = max(str(row[0]) for row in updated)
                    await conn.commit()
                    continue
                await cur.execute(cast("LiteralString", remaining_sql))
                row = await cur.fetchone()
                remaining = row is not None and row[0] is True
                await conn.commit()
            if not remaining:
                return
            # Catch rows skipped behind the local cursor because another
            # migrator held them. A crash simply restarts this scan from zero.
            after_session_id = None
            await asyncio.sleep(0.05)

    async def _run_resumable_sequence_backfill(
        self,
        batch_sql: str,
        remaining_sql: str,
    ) -> None:
        after_sequence = 0
        while True:
            async with PostgresTimingScope(self._pool.connection()) as conn, conn.cursor() as cur:
                await cur.execute(cast("LiteralString", batch_sql), (after_sequence,))
                updated = await cur.fetchall()
                if updated:
                    after_sequence = max(int(row[0]) for row in updated)
                    await conn.commit()
                    continue
                await cur.execute(cast("LiteralString", remaining_sql))
                row = await cur.fetchone()
                remaining = row is not None and row[0] is True
                await conn.commit()
            if not remaining:
                return
            # Catch rows skipped behind the local cursor because another
            # migrator held them. A crash simply restarts this scan from zero.
            after_sequence = 0
            await asyncio.sleep(0.05)

    def _validate_postgres_revision(self, state: schema.SchemaState) -> None:
        if state.revision < self._min_required_revision:
            raise schema.SchemaTooOld(
                f"Postgres schema is at revision {state.revision}; this build requires "
                f">= {self._min_required_revision}. Run `cayu storage migrate` before "
                "starting."
            )

    async def _validate_postgres_schema(self, cur: Any, state: schema.SchemaState) -> None:
        self._validate_postgres_revision(state)
        if state.revision >= 108:
            await validate_postgres_context_selection_schema(cur)
        if state.revision >= 110:
            await postgres_producer_schema._validate_producer_cleanup_receipts(cur)
        if state.revision >= 96:
            await validate_postgres_participant_bindings(cur)
        if state.revision >= 111:
            await validate_postgres_wait_discovery(cur)
        if state.revision >= 112:
            await validate_postgres_product_operation_schema(cur)
        if state.revision >= 93:
            await validate_postgres_collaboration_schema(
                cur,
                lifecycle=state.revision >= 94,
                requests=state.revision >= 95,
                clarifications=state.revision >= 105,
                planning=state.revision >= 107,
            )
        if self._min_required_revision >= 36:
            await postgres_session_schema._validate_session_invocation_column(cur)
        if self._min_required_revision >= 38:
            await postgres_task_schema._validate_task_terminalization_receipt_table(cur)
        if self._min_required_revision >= 39:
            await postgres_task_schema._validate_task_invocation_column(cur)
        if self._min_required_revision >= 41:
            await postgres_knowledge_schema._validate_knowledge_publication_access_snapshot_column(
                cur
            )
        if self._min_required_revision >= 42:
            await postgres_knowledge_schema._validate_knowledge_revision_schema(
                cur,
                allow_revision_43=state.revision >= 43,
                require_payload_bytes=state.revision >= 65,
            )
        if self._min_required_revision >= 43:
            await postgres_knowledge_schema._validate_knowledge_change_schema(
                cur,
                relation_aware=state.revision >= 60,
            )
        if self._min_required_revision >= 44:
            await postgres_knowledge_schema._validate_knowledge_index_readiness_schema(cur)
        if state.revision >= 60:
            await postgres_knowledge_schema._validate_knowledge_relation_schema(cur)
        if state.revision >= 63:
            await postgres_knowledge_schema._validate_knowledge_maintenance_schema(cur)
        if state.revision >= 67:
            await postgres_knowledge_schema._validate_knowledge_maintenance_proposal_schema(cur)
        if state.revision >= 69:
            await postgres_work_context_schema._validate_agent_work_context_schema(cur)
        if state.revision >= 71:
            await postgres_work_context_schema._validate_agent_recall_delivery_schema(
                cur,
                require_processing_schema_version=state.revision >= 73,
            )
        if state.revision >= 73:
            await postgres_work_context_schema._validate_agent_recall_subscription_schema(cur)
        if self._min_required_revision >= 45:
            await postgres_task_schema._validate_task_retry_series_schema(cur)
        if self._min_required_revision >= 46:
            await postgres_transcript_schema._validate_transcript_search_document_column(
                cur, tokenizer_version=TRANSCRIPT_SEARCH_TOKENIZER_VERSION
            )
        if self._min_required_revision >= 47:
            await postgres_eval_schema._validate_eval_result_baseline_schema(cur)
        if self._min_required_revision >= 48:
            await postgres_eval_schema._validate_captured_eval_case_schema(cur)
        if self._min_required_revision >= 49:
            await postgres_verified_work_schema._validate_verified_work_schema(
                cur,
                require_verifier_profiles=state.revision >= 58,
            )
        if self._min_required_revision >= 50:
            await postgres_eval_schema._validate_eval_run_invocation_column(cur)
        if self._min_required_revision >= 51:
            await postgres_memory_evidence_schema._validate_memory_evidence_schema(cur)
        if self._min_required_revision >= 52:
            await postgres_session_schema._validate_targeted_tool_grant_schema(cur)
        if self._min_required_revision >= 53:
            await postgres_eval_schema._validate_eval_scenario_schema(cur)
        if self._min_required_revision >= 55:
            await postgres_task_schema._validate_task_retry_reconciliation_schema(cur)
        if self._min_required_revision >= 56:
            await postgres_eval_schema._validate_eval_run_scenario_progress_column(cur)
        if self._min_required_revision >= 57:
            await postgres_session_schema._validate_session_message_queue_typed_message_column(cur)
        if self._min_required_revision >= 83:
            await postgres_session_schema._validate_session_message_lifecycle_columns(cur)
        if self._min_required_revision >= 61:
            await postgres_verified_work_schema._validate_work_attempt_admission_schema(cur)
        if self._min_required_revision >= 84:
            await postgres_verified_work_schema._validate_work_attempt_lifecycle_schema(cur)
        if self._min_required_revision >= 62:
            await postgres_verified_work_schema._validate_deferred_interaction_input_payloads(cur)
            await postgres_verified_work_schema._validate_work_attempt_continuation_authority(cur)
        if self._min_required_revision >= 64:
            await postgres_eval_schema._validate_eval_authored_suite_schema(cur)
        if self._min_required_revision >= 68:
            await postgres_eval_schema._validate_eval_judge_calibration_schema(cur)
        if self._min_required_revision >= 72:
            await postgres_eval_schema._validate_eval_run_max_concurrency_schema(cur)
        if self._min_required_revision >= 70:
            await postgres_task_schema._validate_interrupted_task_handoff_schema(cur)
        if self._min_required_revision >= 76:
            await postgres_task_schema._validate_interrupted_handoff_generation_column(cur)
        if self._min_required_revision >= 74:
            await postgres_eval_schema._validate_eval_run_trial_checkpoint_schema(cur)
        if self._min_required_revision >= 75:
            await postgres_knowledge_schema._validate_knowledge_activation_schema(cur)
        if self._min_required_revision >= 77:
            await postgres_knowledge_schema._validate_knowledge_maintenance_governance_schema(cur)
        if self._min_required_revision >= 78:
            await postgres_knowledge_schema._validate_knowledge_semantic_watch_schema(cur)
        if self._min_required_revision >= 79:
            await postgres_session_schema._validate_child_session_lifecycle_schema(cur)
        if self._min_required_revision >= 88:
            await postgres_task_schema._validate_task_closure_guard(
                cur, expected_guard_sql=postgres_schema_history._MIGRATION_STEPS[88][1]
            )
        if state.revision >= 23:
            await postgres_budget_schema._validate_budget_reservation_identity_registry(
                cur,
                require=True,
            )
        if state.revision >= 28:
            await postgres_session_schema._validate_public_authority_alias_registry(cur)
        for index in postgres_schema_history._required_concurrent_indexes(state.revision):
            existing = await self._concurrent_index_state(cur, index)
            if existing is None:
                raise RuntimeError(
                    f"Required Cayu Postgres index is missing: {index.index_name}. "
                    "Run `cayu storage migrate` to repair the schema."
                )
            valid, building = existing
            if not valid or building:
                raise RuntimeError(
                    f"Required Cayu Postgres index is not ready: {index.index_name}. "
                    "Run `cayu storage migrate` to repair the schema."
                )

    async def _validate_revision_schema_objects(
        self,
        cur: Any,
        revision: schema.Revision,
    ) -> None:
        """Validate non-index objects before recording their owning revision."""

        if revision.revision == 115:
            from cayu.storage._continuation_index_migration import (
                migrate_postgres_continuation_indexes,
            )

            await migrate_postgres_continuation_indexes(cur)
        if revision.revision == 108:
            await validate_postgres_context_selection_schema(cur)
        if revision.revision == 110:
            await postgres_producer_schema._validate_producer_cleanup_receipts(cur)
        if revision.revision == 111:
            await validate_postgres_wait_discovery(cur)
        if revision.revision == 112:
            await validate_postgres_product_operation_schema(cur)
        if revision.revision == 102:
            await validate_postgres_participant_bindings(cur)
        if revision.revision == 36:
            await postgres_session_schema._validate_session_invocation_column(cur)
        if revision.revision == 38:
            await postgres_task_schema._validate_task_terminalization_receipt_table(cur)
        if revision.revision == 39:
            await postgres_task_schema._validate_task_invocation_column(cur)
        if revision.revision == 41:
            await postgres_knowledge_schema._validate_knowledge_publication_access_snapshot_column(
                cur
            )
        if revision.revision == 42:
            await postgres_knowledge_schema._validate_knowledge_revision_schema(cur)
        if revision.revision == 43:
            await postgres_knowledge_schema._validate_knowledge_change_schema(cur)
        if revision.revision == 44:
            await postgres_knowledge_schema._validate_knowledge_index_readiness_schema(cur)
        if revision.revision == 45:
            await postgres_task_schema._validate_task_retry_series_schema(cur)
        if revision.revision == 46:
            await postgres_transcript_schema._validate_transcript_search_document_column(
                cur, tokenizer_version=TRANSCRIPT_SEARCH_TOKENIZER_VERSION
            )
        if revision.revision == 47:
            await postgres_eval_schema._validate_eval_result_baseline_schema(cur)
        if revision.revision == 48:
            await postgres_eval_schema._validate_captured_eval_case_schema(cur)
        if revision.revision == 50:
            await postgres_eval_schema._validate_eval_run_invocation_column(cur)
        if revision.revision == 51:
            await postgres_memory_evidence_schema._validate_memory_evidence_schema(cur)
        if revision.revision == 52:
            await postgres_session_schema._validate_targeted_tool_grant_schema(cur)
        if revision.revision == 53:
            await postgres_eval_schema._validate_eval_scenario_schema(cur)
        if revision.revision == 55:
            await postgres_task_schema._validate_task_retry_reconciliation_schema(cur)
        if revision.revision == 56:
            await postgres_eval_schema._validate_eval_run_scenario_progress_column(cur)
        if revision.revision == 57:
            await postgres_session_schema._validate_session_message_queue_typed_message_column(cur)
        if revision.revision == 83:
            await postgres_session_schema._validate_session_message_lifecycle_columns(cur)
        if revision.revision == 58:
            await postgres_verified_work_schema._validate_verified_work_schema(
                cur,
                require_verifier_profiles=True,
            )
        if revision.revision == 59:
            await _reject_populated_pre_result_resolver_database(cur)
            await postgres_session_schema._validate_session_instance_schema(cur)
        if revision.revision == 60:
            await postgres_knowledge_schema._validate_knowledge_change_schema(
                cur, relation_aware=True
            )
            await postgres_knowledge_schema._validate_knowledge_relation_schema(cur)
        if revision.revision == 61:
            await postgres_verified_work_schema._validate_work_attempt_admission_schema(cur)
        if revision.revision == 84:
            await postgres_verified_work_schema._validate_work_attempt_lifecycle_schema(cur)
        if revision.revision == 62:
            await postgres_verified_work_schema._validate_deferred_interaction_input_payloads(cur)
            await postgres_verified_work_schema._validate_work_attempt_continuation_authority(cur)
        if revision.revision == 63:
            await postgres_knowledge_schema._validate_knowledge_maintenance_schema(cur)
        if revision.revision == 64:
            await postgres_eval_schema._validate_eval_authored_suite_schema(cur)
        if revision.revision == 65:
            await postgres_knowledge_schema._validate_knowledge_revision_schema(
                cur,
                allow_revision_43=True,
                require_payload_bytes=True,
            )
        if revision.revision == 66:
            await postgres_task_schema._validate_local_execution_attempt_schema(cur)
        if revision.revision == 67:
            await postgres_knowledge_schema._validate_knowledge_maintenance_proposal_schema(cur)
        if revision.revision == 68:
            await postgres_eval_schema._validate_eval_judge_calibration_schema(cur)
        if revision.revision == 69:
            await postgres_work_context_schema._validate_agent_work_context_schema(cur)
        if revision.revision == 70:
            await postgres_task_schema._validate_interrupted_task_handoff_schema(cur)
        if revision.revision == 71:
            await postgres_work_context_schema._validate_agent_recall_delivery_schema(cur)
        if revision.revision == 72:
            await postgres_eval_schema._validate_eval_run_max_concurrency_schema(cur)
        if revision.revision == 73:
            await postgres_work_context_schema._validate_agent_recall_delivery_schema(
                cur,
                require_processing_schema_version=True,
            )
            await postgres_work_context_schema._validate_agent_recall_subscription_schema(cur)
        if revision.revision == 74:
            await postgres_eval_schema._validate_eval_run_trial_checkpoint_schema(cur)
        if revision.revision == 75:
            await postgres_knowledge_schema._validate_knowledge_activation_schema(cur)
        if revision.revision == 76:
            await self._backfill_interrupted_handoff_generations(cur)
            await postgres_task_schema._validate_interrupted_handoff_generation_column(cur)
        if revision.revision == 77:
            await postgres_knowledge_schema._validate_knowledge_maintenance_governance_schema(cur)
        if revision.revision == 78:
            await postgres_knowledge_schema._validate_knowledge_semantic_watch_schema(cur)
        if revision.revision == 79:
            await postgres_session_schema._validate_child_session_lifecycle_schema(cur)
        if revision.revision == 88:
            await postgres_task_schema._validate_task_closure_guard(
                cur, expected_guard_sql=postgres_schema_history._MIGRATION_STEPS[88][1]
            )
        if revision.revision == 93:
            await validate_postgres_collaboration_schema(cur)
        if revision.revision == 94:
            await validate_postgres_collaboration_schema(cur, lifecycle=True)

    async def _backfill_interrupted_handoff_generations(self, cur: Any) -> None:
        """Carry unambiguous revision-70 handoff authority into revision 76."""

        cursor: tuple[str, datetime, str] | None = None
        active_task_id: str | None = None
        active_current_task: Task | None = None
        active_matching_generations: list[str] = []

        async def finalize_active_task() -> None:
            if active_task_id is None or not active_matching_generations:
                return
            if len(active_matching_generations) != 1:
                raise RuntimeError(
                    "Postgres revision-76 migration cannot determine one "
                    f"interrupted-task handoff generation for task {active_task_id!r}. "
                    "Resolve the ambiguous recovery receipts before migrating."
                )
            await cur.execute(
                "UPDATE cayu_tasks SET interrupted_handoff_id = %s WHERE id = %s",
                (active_matching_generations[0], active_task_id),
            )

        while True:
            after_sql = ""
            after_params: tuple[object, ...] = ()
            if cursor is not None:
                after_sql = "WHERE (task_id, committed_at, handoff_id) > (%s, %s, %s)"
                after_params = cursor
            await cur.execute(
                f"""
                SELECT task_id, handoff_id, request_sha256, request_json, task_json,
                       committed_at
                FROM cayu_task_interrupted_handoff_receipts
                {after_sql}
                ORDER BY task_id, committed_at, handoff_id
                LIMIT %s
                """,
                (*after_params, _INTERRUPTED_HANDOFF_MIGRATION_BATCH_SIZE),
            )
            receipts = tuple(await cur.fetchall())
            if not receipts:
                break
            task_ids = list(dict.fromkeys(str(row[0]) for row in receipts))
            await cur.execute(
                f"SELECT {pg_support.TASK_COLUMNS} FROM cayu_tasks WHERE id = ANY(%s) FOR UPDATE",
                (task_ids,),
            )
            current_tasks = {
                task.id: task
                for task in (pg_support.task_from_row(row) for row in await cur.fetchall())
            }
            receipt_updates: list[tuple[str, str, str]] = []
            for receipt_row in receipts:
                task_id = str(receipt_row[0])
                if task_id != active_task_id:
                    await finalize_active_task()
                    active_task_id = task_id
                    active_current_task = current_tasks.get(task_id)
                    active_matching_generations = []
                try:
                    request = TaskInterruptedHandoffRequest.model_validate(
                        pg_support._json_obj(receipt_row[3])
                    )
                    request, request_sha256 = prepare_interrupted_task_handoff(request)
                    receipt_task = Task.model_validate(pg_support._json_obj(receipt_row[4]))
                    if (
                        request.task_id != task_id
                        or request.handoff_id != receipt_row[1]
                        or request_sha256 != receipt_row[2]
                        or receipt_task.id != request.task_id
                        or receipt_task.status is not TaskStatus.RUNNING
                        or receipt_task.session_id != request.session_id
                        or receipt_task.session_instance_id != request.session_instance_id
                        or receipt_task.worker_id is not None
                        or receipt_task.lease_expires_at is not None
                        or receipt_task.interrupted_handoff_id is not None
                    ):
                        raise ValueError("receipt conflicts with its pre-76 handoff authority")
                except Exception as exc:
                    raise RuntimeError(
                        "Postgres revision-76 migration found malformed interrupted-task "
                        f"handoff authority for task {task_id!r}. Restore the database "
                        "from known-good recovery evidence."
                    ) from exc
                upgraded_task = receipt_task.model_copy(
                    update={"interrupted_handoff_id": request.handoff_id},
                    deep=True,
                )
                upgraded_task = Task.model_validate(upgraded_task.model_dump(mode="python"))
                receipt_updates.append(
                    (
                        pg_support._dumps(upgraded_task.model_dump(mode="json", warnings=False)),
                        task_id,
                        request.handoff_id,
                    )
                )
                if active_current_task == receipt_task:
                    active_matching_generations.append(request.handoff_id)
            await cur.executemany(
                """
                UPDATE cayu_task_interrupted_handoff_receipts
                SET task_json = %s
                WHERE task_id = %s AND handoff_id = %s
                """,
                receipt_updates,
            )
            last = receipts[-1]
            cursor = (str(last[0]), last[5], str(last[1]))
        await finalize_active_task()

    async def _read_schema_state(self, cur: Any) -> schema.SchemaState:
        return await read_schema_state(cur)

    async def _apply_baseline(self, cur: Any) -> None:
        for statement in postgres_schema_history.SCHEMA_STATEMENTS:
            await cur.execute(statement)
        await self._record_revision(cur, schema.revision(schema.BASELINE_REVISION))

    async def _apply_pending(self, cur: Any, state: schema.SchemaState) -> None:
        current = state.revision
        if (
            current != schema.UNINITIALIZED
            and current < 26
            and any(revision.revision == 26 for revision in schema.pending(current))
        ):
            await _reject_populated_pre_interaction_database(cur)
        if (
            current != schema.UNINITIALIZED
            and current < 36
            and any(revision.revision == 36 for revision in schema.pending(current))
        ):
            await _reject_populated_pre_invocation_database(cur)
        if (
            current != schema.UNINITIALIZED
            and current < 39
            and any(revision.revision == 39 for revision in schema.pending(current))
        ):
            await _reject_populated_pre_task_invocation_database(cur)
        if (
            current != schema.UNINITIALIZED
            and current < 41
            and any(revision.revision == 41 for revision in schema.pending(current))
        ):
            await _reject_populated_pre_knowledge_access_snapshot_database(cur)
        if current < 42 and any(revision.revision == 42 for revision in schema.pending(current)):
            await _reject_populated_pre_knowledge_revision_database(cur)
        if current < 46 and any(revision.revision == 46 for revision in schema.pending(current)):
            await _reject_populated_pre_transcript_search_database(cur)
        if (
            current != schema.UNINITIALIZED
            and current < 52
            and any(revision.revision == 52 for revision in schema.pending(current))
        ):
            await _reject_populated_pre_targeted_tool_grant_database(cur)
        if current < 58 and any(revision.revision == 58 for revision in schema.pending(current)):
            await _reject_populated_pre_verifier_profile_database(cur)
        if current < 59 and any(revision.revision == 59 for revision in schema.pending(current)):
            await _reject_populated_pre_result_resolver_database(cur)
        if current < 65 and any(revision.revision == 65 for revision in schema.pending(current)):
            await _reject_populated_pre_bounded_knowledge_entry_database(cur)
        if (
            current != schema.UNINITIALIZED
            and current < 73
            and any(revision.revision == 73 for revision in schema.pending(current))
        ):
            await _reject_populated_pre_recall_subscription_database(cur)
        if current == schema.UNINITIALIZED:
            await self._apply_baseline(cur)
            current = schema.BASELINE_REVISION
        for rev in schema.pending(current):
            if rev.revision == 43:
                await _reject_revision_43_knowledge_identity_overflow(cur)
            if rev.revision == 65:
                await _reject_populated_pre_bounded_knowledge_entry_database(cur)
            if rev.revision == 73:
                await _reject_populated_pre_recall_subscription_database(cur)
            for statement in postgres_schema_history._MIGRATION_STEPS.get(rev.revision, ()):
                await cur.execute(statement)
            # Fresh CREATE owns empty tables under the schema lock, so hot-table
            # indexes can be built transactionally. Existing databases still use
            # the non-transactional CONCURRENTLY path in ``_migrate_schema``.
            for index in postgres_schema_history._CONCURRENT_INDEX_MIGRATIONS.get(rev.revision, ()):
                if index.replace_existing:
                    existing = await self._concurrent_index_state(
                        cur, index, allow_replacement=True
                    )
                    if existing is not None and not existing[0]:
                        await cur.execute(index.drop_statement.replace("CONCURRENTLY", ""))
                await cur.execute(index.transactional_create_statement())
            await self._validate_revision_schema_objects(cur, rev)
            await self._record_revision(cur, rev)

    async def _ensure_concurrent_index(
        self,
        conn: Any,
        index: postgres_schema_history._ConcurrentIndexMigration,
        *,
        pending_revision: int | None = None,
    ) -> None:
        await conn.set_autocommit(True)
        lock_acquired = False
        try:
            # CREATE INDEX CONCURRENTLY cannot run under the transaction-level
            # schema lock. Poll a session lock with pg_try_advisory_lock: a
            # blocking advisory-lock statement would hold a virtual xid while
            # it waits and can deadlock the winning CREATE INDEX CONCURRENTLY.
            # Each failed try completes its autocommit transaction before sleep.
            while not lock_acquired:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_try_advisory_lock(%s)",
                        (_SCHEMA_ADVISORY_LOCK_KEY,),
                    )
                    row = await cur.fetchone()
                    lock_acquired = row is not None and row[0] is True
                if not lock_acquired:
                    await asyncio.sleep(_SCHEMA_ADVISORY_LOCK_POLL_SECONDS)

            if pending_revision is not None:
                async with conn.cursor() as cur:
                    state = await self._read_schema_state(cur)
                if state.revision >= pending_revision:
                    # A peer may have finished this revision and replaced the
                    # index again while we waited. Never apply a stale definition.
                    # The migration loop re-reads progress and validates the final
                    # required indexes before admitting the store.
                    return

            while True:
                async with conn.cursor() as cur:
                    existing = await self._concurrent_index_state(
                        cur,
                        index,
                        allow_replacement=True,
                    )
                    if existing == (True, False):
                        return
                    if existing is not None and existing[1]:
                        await asyncio.sleep(_SCHEMA_ADVISORY_LOCK_POLL_SECONDS)
                        continue
                    if existing is not None:
                        await cur.execute(index.drop_statement)
                    try:
                        await cur.execute(index.create_statement)
                    except (DeadlockDetected, DuplicateTable):
                        continue
                    except UniqueViolation as exc:
                        if not index.unique:
                            continue
                        raise RuntimeError(
                            f"Postgres migration cannot create required unique index "
                            f"{index.index_name}: durable rows contain duplicate identities."
                        ) from exc
                    created = await self._concurrent_index_state(
                        cur,
                        index,
                    )
                    if created == (True, False):
                        return
                    if created is None or not created[1]:
                        raise RuntimeError(
                            f"Postgres migration did not create a valid index: {index.index_name}"
                        )
        finally:
            if lock_acquired:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT pg_advisory_unlock(%s)",
                        (_SCHEMA_ADVISORY_LOCK_KEY,),
                    )
            await conn.set_autocommit(False)

    async def _concurrent_index_state(
        self,
        cur: Any,
        index: postgres_schema_history._ConcurrentIndexMigration,
        *,
        allow_replacement: bool = False,
    ) -> tuple[bool, bool] | None:
        await cur.execute(
            """
            SELECT
                index_definition.indexrelid IS NOT NULL,
                COALESCE(index_definition.indisvalid, FALSE),
                COALESCE(
                    table_class.relnamespace = namespace.oid
                    AND table_class.relname = %s,
                    FALSE
                ),
                COALESCE(access_method.amname = %s, FALSE),
                ARRAY(
                    SELECT pg_get_indexdef(
                        index_definition.indexrelid,
                        key_position,
                        FALSE
                    )
                    FROM generate_series(
                        1,
                        index_definition.indnkeyatts
                    ) AS key_position
                    ORDER BY key_position
                ),
                ARRAY(
                    SELECT index_collation.collname
                    FROM unnest(index_definition.indcollation::oid[])
                         WITH ORDINALITY AS key_collation(
                             collation_oid,
                             key_position
                         )
                    LEFT JOIN pg_catalog.pg_collation AS index_collation
                      ON index_collation.oid = key_collation.collation_oid
                    WHERE key_collation.key_position
                          <= index_definition.indnkeyatts
                    ORDER BY key_collation.key_position
                ),
                pg_get_expr(
                    index_definition.indpred,
                    index_definition.indrelid,
                    FALSE
                ),
                COALESCE(index_definition.indisunique, FALSE),
                EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_stat_progress_create_index AS progress
                    WHERE progress.index_relid = index_class.oid
                )
            FROM pg_catalog.pg_class AS index_class
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = index_class.relnamespace
            LEFT JOIN pg_catalog.pg_index AS index_definition
              ON index_definition.indexrelid = index_class.oid
            LEFT JOIN pg_catalog.pg_class AS table_class
              ON table_class.oid = index_definition.indrelid
            LEFT JOIN pg_catalog.pg_am AS access_method
              ON access_method.oid = index_class.relam
            WHERE namespace.nspname = current_schema()
              AND index_class.relname = %s
            """,
            (index.table_name, index.access_method, index.index_name),
        )
        row = await cur.fetchone()
        if row is None:
            return None
        key_definitions = tuple(
            postgres_catalog._normalize_postgres_index_expression(str(value))
            for value in (row[4] or [])
        )
        expected_keys = tuple(
            postgres_catalog._normalize_postgres_index_expression(value)
            for value in index.key_definitions
        )
        key_collations = tuple(None if value is None else str(value) for value in (row[5] or []))
        required_key_collations = index.required_key_collations
        collations_match = not required_key_collations or (
            len(key_collations) == len(required_key_collations)
            and all(
                required is None or actual == required
                for actual, required in zip(
                    key_collations,
                    required_key_collations,
                    strict=True,
                )
            )
        )
        predicate = postgres_catalog._normalize_postgres_index_expression(row[6])
        expected_predicate = postgres_catalog._normalize_postgres_index_expression(
            index.predicate_definition
        )
        expected_definition = (
            bool(row[0])
            and bool(row[2])
            and bool(row[3])
            and key_definitions == expected_keys
            and collations_match
            and predicate == expected_predicate
            and bool(row[7]) is index.unique
        )
        if not expected_definition:
            replaceable_definition = (
                allow_replacement
                and index.replace_existing
                and bool(row[0])
                and bool(row[2])
                and bool(row[3])
                and key_definitions == expected_keys
                and (
                    predicate == expected_predicate
                    or predicate
                    in {
                        postgres_catalog._normalize_postgres_index_expression(value)
                        for value in index.replacement_predicates
                    }
                )
                and bool(row[7]) is index.unique
            )
            if replaceable_definition:
                return False, bool(row[8])
            columns = ", ".join(
                (f'{key} COLLATE "{collation}"' if collation is not None else key)
                for key, collation in zip(
                    index.key_definitions,
                    index.required_key_collations or (None,) * len(index.key_definitions),
                    strict=True,
                )
            )
            index_kind = "unique B-tree" if index.unique else "B-tree"
            raise RuntimeError(
                f"Postgres schema object {index.index_name!r} conflicts with the required "
                f"{index_kind} index on {index.table_name}({columns}). Remove or rename the "
                "conflicting object, then rerun `cayu storage migrate`. "
                f"Observed keys={key_definitions!r}, predicate={predicate!r}; "
                f"expected keys={expected_keys!r}, predicate={expected_predicate!r}."
            )
        return bool(row[1]), bool(row[8])

    async def _record_revision(self, cur: Any, rev: schema.Revision) -> None:
        await cur.execute(
            "INSERT INTO cayu_schema_migrations "
            "(revision, kind, compatible_from, checksum, applied_at) "
            "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (revision) DO NOTHING",
            (
                rev.revision,
                str(rev.kind),
                rev.compatible_from,
                self._migration_operation_sha256,
                datetime.now(UTC),
            ),
        )

    async def close(self) -> None:
        if self._owns_pool and self._opened:
            await self._pool.close()
            self._opened = False
