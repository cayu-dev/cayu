"""Cross-backend schema versioning + migration model (ADR 0001).

Backend-agnostic core: the revision history, the additive/breaking compatibility
model, and the validate/plan logic. Backend adapters (SQLite, Postgres) own the
DDL execution, the ``cayu_schema_migrations`` table CRUD, and the coordination lock;
they read this module's revision list and reuse :func:`validate` / :func:`pending`.

Compatibility model (ADR 0001, Decision 7):

- Every revision is ``additive`` (forward-compatible — only adds tables/columns/
  indexes; older binaries keep working because the stores select explicit columns)
  or ``breaking`` (rename/drop/retype/semantic change).
- Each revision records a ``compatible_from`` floor: the oldest app revision that
  can still operate against a database at that revision. An additive revision
  inherits the prior revision's floor; a breaking revision sets the floor to itself.
- ``validate`` passes iff ``app_latest >= db.compatible_from`` (binary new enough
  for the DB) and ``db.revision >= app_min_supported`` (DB not ancient).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# Prefix for every Cayu-owned table, so Cayu state never collides with an app's own
# tables in a shared database (ADR 0001, Decision 5).
TABLE_PREFIX = "cayu_"

#: Name of the migration-bookkeeping table (already prefixed).
MIGRATIONS_TABLE = f"{TABLE_PREFIX}schema_migrations"


class RevisionKind(StrEnum):
    """Whether a revision keeps older binaries working (additive) or not (breaking)."""

    ADDITIVE = "additive"
    BREAKING = "breaking"


class SchemaMode(StrEnum):
    """How a store reconciles its code schema with the database at startup."""

    #: Create the baseline schema if the database is empty; otherwise validate.
    #: Default for SQLite / in-memory (dev, tests, local).
    CREATE = "create"
    #: Check compatibility only; never run DDL. Default for production Postgres.
    VALIDATE = "validate"
    #: Apply pending forward revisions under the backend lock, then validate.
    #: The explicit deploy step.
    MIGRATE = "migrate"


@dataclass(frozen=True)
class Revision:
    """One schema revision in the append-only history."""

    revision: int
    kind: RevisionKind
    #: Oldest app revision that can operate against a DB at this revision.
    compatible_from: int


# Append-only migration history. Greenfield baseline = revision 1 (the full current
# schema). Future schema changes append a Revision here; an additive one keeps
# ``compatible_from`` at the prior value, a breaking one sets it to its own number.
REVISIONS: tuple[Revision, ...] = (
    Revision(revision=1, kind=RevisionKind.BREAKING, compatible_from=1),
    # Revisions 2-6 are purely additive (new tables/columns/indexes only): they
    # add cayu_session_labels (2), cayu_event_watcher_state (3), the task
    # worker/lease columns (4), the task status_reason/payload columns (5), and
    # the knowledge tables (6). Older binaries keep working because the stores
    # select explicit columns, so each inherits revision 1's floor rather than
    # raising it.
    Revision(revision=2, kind=RevisionKind.ADDITIVE, compatible_from=1),
    Revision(revision=3, kind=RevisionKind.ADDITIVE, compatible_from=1),
    Revision(revision=4, kind=RevisionKind.ADDITIVE, compatible_from=1),
    Revision(revision=5, kind=RevisionKind.ADDITIVE, compatible_from=1),
    Revision(revision=6, kind=RevisionKind.ADDITIVE, compatible_from=1),
    Revision(revision=7, kind=RevisionKind.ADDITIVE, compatible_from=1),
    # Budget ledger DDL moves into the migration machinery and the table is
    # renamed to the cayu_ prefix (breaking: rename + ownership change).
    Revision(revision=8, kind=RevisionKind.BREAKING, compatible_from=8),
    # Drop the redundant cayu_events.event_json column: the full serialized Event
    # duplicated the individual indexed columns plus payload_json (write
    # amplification + unbounded growth). The store now reconstructs Events from
    # those columns, so an older binary that still SELECTs event_json can no
    # longer read the table (breaking: floor rises to itself).
    Revision(revision=9, kind=RevisionKind.BREAKING, compatible_from=9),
    # Add cayu_sessions.event_seq, a per-session monotonic counter, so the
    # Postgres append path reserves session_order values with a single
    # UPDATE ... RETURNING instead of a SELECT ... FOR UPDATE + COALESCE(MAX())
    # scan on the hottest write path. How session_order is assigned is now
    # welded to that counter: a pre-10 binary appending via MAX() would leave
    # the counter stale, so a rev-10 binary must not share the database with one
    # (breaking: floor rises to itself). SQLite is single-connection-serialized
    # and keeps MAX(); the revision carries no SQLite DDL.
    Revision(revision=10, kind=RevisionKind.BREAKING, compatible_from=10),
    # Add cayu_event_watcher_dead_letters: a durable, replayable record per event
    # that exhausted its delivery attempts, replacing the lossy single
    # dead_lettered_count counter + overwritten last_error on the watcher state.
    # Purely additive (new table only) — older binaries never touch it and keep
    # working, so the floor stays at revision 10's compatible_from.
    Revision(revision=11, kind=RevisionKind.ADDITIVE, compatible_from=10),
    # Add cayu_knowledge_embeddings.embedding_space_version (Postgres/pgvector only; SQLite has no
    # embeddings table, so this revision carries no SQLite DDL). Purely additive (a nullable-with-
    # default column), so the floor stays at revision 10's compatible_from and older binaries keep
    # working.
    Revision(revision=12, kind=RevisionKind.ADDITIVE, compatible_from=10),
    # Add cayu_events.insert_xid so Postgres cross-session event consumers can
    # avoid advancing an after_sequence cursor past events inserted by still-open
    # transactions. This is a Postgres-only additive DDL revision; SQLite has no
    # DDL for it and older SQLite DBs remain compatible with this binary.
    Revision(revision=13, kind=RevisionKind.ADDITIVE, compatible_from=10),
    # Activity timestamps and run epochs are additive columns. New SessionStore
    # implementations still require this revision before use, while older binaries
    # can continue operating against the expanded schema.
    Revision(revision=14, kind=RevisionKind.ADDITIVE, compatible_from=10),
    # Index durable interruption-cascade markers so server startup discovers
    # recoverable roots without scanning every historical interrupted session.
    Revision(revision=15, kind=RevisionKind.ADDITIVE, compatible_from=10),
)

#: The revision an empty database is initialized to.
BASELINE_REVISION = REVISIONS[0].revision
#: The newest revision this binary knows how to produce.
LATEST_REVISION = REVISIONS[-1].revision
#: The oldest DB revision this binary can still operate against. Equals the
#: ``compatible_from`` of the latest revision (older DBs must ``migrate``).
MIN_SUPPORTED_REVISION = REVISIONS[-1].compatible_from

#: Sentinel revision for an empty / uninitialized database.
UNINITIALIZED = 0


class SchemaError(RuntimeError):
    """Base class for schema-compatibility failures."""


class SchemaUninitialized(SchemaError):
    """The database has no Cayu schema yet (needs create/migrate)."""


class SchemaTooOld(SchemaError):
    """The database is older than this binary supports (needs migrate)."""


class SchemaTooNew(SchemaError):
    """The database was migrated past what this binary understands (upgrade the app)."""


@dataclass(frozen=True)
class SchemaState:
    """The schema state read from a database."""

    #: Current revision, or :data:`UNINITIALIZED` (0) when no Cayu schema exists.
    revision: int
    #: ``compatible_from`` floor recorded for the current revision.
    compatible_from: int


def revision(number: int) -> Revision:
    """Look up a known revision by number."""
    for rev in REVISIONS:
        if rev.revision == number:
            return rev
    raise ValueError(f"Unknown schema revision: {number}")


def pending(current: int) -> tuple[Revision, ...]:
    """Revisions newer than ``current`` that a ``migrate`` would apply, in order."""
    return tuple(rev for rev in REVISIONS if rev.revision > current)


def validate(
    state: SchemaState,
    *,
    app_latest: int = LATEST_REVISION,
    app_min_supported: int = MIN_SUPPORTED_REVISION,
) -> None:
    """Fail fast unless this binary can safely operate against ``state``.

    Raises :class:`SchemaUninitialized`, :class:`SchemaTooOld`, or
    :class:`SchemaTooNew` with an actionable message; returns ``None`` on success.
    """
    if state.revision == UNINITIALIZED:
        raise SchemaUninitialized(
            "Cayu schema is not initialized. Run `cayu storage migrate` "
            "(or create the store with schema_mode=create on an empty database)."
        )
    if app_latest < state.compatible_from:
        raise SchemaTooNew(
            f"Database is at schema revision {state.revision}, which requires an app "
            f"that understands revision >= {state.compatible_from}; this build supports "
            f"up to {app_latest}. Upgrade the application."
        )
    if state.revision < app_min_supported:
        raise SchemaTooOld(
            f"Database is at schema revision {state.revision}; this build requires "
            f">= {app_min_supported}. Run `cayu storage migrate` before starting."
        )


__all__ = [
    "BASELINE_REVISION",
    "LATEST_REVISION",
    "MIGRATIONS_TABLE",
    "MIN_SUPPORTED_REVISION",
    "REVISIONS",
    "TABLE_PREFIX",
    "UNINITIALIZED",
    "Revision",
    "RevisionKind",
    "SchemaError",
    "SchemaMode",
    "SchemaState",
    "SchemaTooNew",
    "SchemaTooOld",
    "SchemaUninitialized",
    "pending",
    "revision",
    "validate",
]
