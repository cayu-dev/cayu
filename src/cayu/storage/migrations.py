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
    # Index a session's durable global event sequence so newest-first history
    # pages and exclusive before_sequence cursors do not scan other sessions.
    Revision(revision=16, kind=RevisionKind.ADDITIVE, compatible_from=10),
    # Persist bounded-query metadata beside checkpoint state and relevant event
    # rows, then index checkpoints and normalized event identifiers. The deploy
    # migration backfills explicit readiness markers in resumable committed
    # batches. Every checkpoint and event writer must maintain that metadata
    # atomically, so pre-17 binaries must not write against a revision-17 database.
    Revision(revision=17, kind=RevisionKind.BREAKING, compatible_from=17),
    # Normalize terminal durable-operation records out of checkpoint state.
    # Completion now atomically updates the checkpoint, appends events, and
    # writes the replay record, so older writers must not share this schema.
    Revision(revision=18, kind=RevisionKind.BREAKING, compatible_from=18),
    # Durable queued session input adds a runtime-required table and changes the
    # session terminalization contract. A pre-19 worker can otherwise complete a
    # session after a revision-19 process accepts queued input, permanently
    # stranding that input. Raise the compatibility floor so mixed-version
    # session workers cannot share a revision-19 database.
    Revision(revision=19, kind=RevisionKind.BREAKING, compatible_from=19),
    # Persist one delivery handoff beside every new runtime event so budget and
    # sink fan-out can resume after a process loss. Revision-20 writers insert
    # it in the event transaction; older writers ignore the new table and keep
    # their existing live fan-out, so the schema remains additive and preserves
    # revision 19's compatibility floor.
    Revision(revision=20, kind=RevisionKind.ADDITIVE, compatible_from=19),
    # Budget reservations now persist the provider billing identity used for
    # admission, and model.completed usage payloads add billing/cache dimensions
    # that pre-21 readers reject. The SQL is additive, but the durable event
    # contract is not safe for mixed-version readers, so raise the compatibility
    # floor and block rolling deploys or app-only rollbacks across this revision.
    Revision(revision=21, kind=RevisionKind.BREAKING, compatible_from=21),
    # MCP manifest authorization now depends on a store-atomic accepted
    # baseline. Pre-22 workers do not maintain that state, so mixed-version
    # session workers could silently bypass drift policy.
    Revision(revision=22, kind=RevisionKind.BREAKING, compatible_from=22),
    # Execution-unit identities make budget reservations attributable to one
    # exact effective limit, logical model step, and provider attempt. Reservation
    # ids also become permanent ledger-wide reconciliation keys: a non-cascading
    # ownership registry claims each id before publication, while a unique event
    # index prevents concurrent sessions from publishing the same id. All writers
    # must maintain these invariants, so mixed-version workers are unsafe.
    Revision(revision=23, kind=RevisionKind.BREAKING, compatible_from=23),
    # Add the composite direct-child traversal index used by bounded workflow
    # topology reads. The index is purely additive and older revision-23
    # binaries continue to operate against the expanded schema.
    Revision(revision=24, kind=RevisionKind.ADDITIVE, compatible_from=23),
    # Budget dispatch fencing and the terminal-settlement outbox change when an
    # expired reservation may release capacity. Every writer must persist the
    # dispatch fence and atomically materialize terminal audit evidence, so a
    # pre-25 worker must not share this ledger.
    Revision(revision=25, kind=RevisionKind.BREAKING, compatible_from=25),
    # Add durable interaction attribution, database-owned transcript ordinals,
    # bounded latest-lifecycle projection, deferred admission, and replay-safe
    # queue/transition records. This prerelease contract intentionally does not
    # migrate populated pre-interaction session databases.
    Revision(revision=26, kind=RevisionKind.BREAKING, compatible_from=26),
    # Add composite session/task and parent-task traversal indexes used by
    # bounded task-topology reads. Both indexes are additive; revision-26
    # writers continue to maintain every indexed source column.
    Revision(revision=27, kind=RevisionKind.ADDITIVE, compatible_from=26),
    # Public authority aliases become durable, indexed store state. Every
    # process that can publish an alias must register it before exposure;
    # pre-28 writers do not maintain that reverse index, so mixed-version
    # workers could publish aliases that no revision-28 worker can resolve.
    Revision(revision=28, kind=RevisionKind.BREAKING, compatible_from=28),
    # Add partial workflow-journal indexes for attempt-fenced step replay and
    # attempt-marker lookup. Existing writers already persist every indexed
    # field, so the revision remains additive for revision-28 binaries.
    Revision(revision=29, kind=RevisionKind.ADDITIVE, compatible_from=28),
    # Rebuild the PostgreSQL direct-child session index with bytewise identifier
    # collation so its keyset order matches memory, SQLite, and Python validation.
    # The indexed source columns and write contract are unchanged.
    Revision(revision=30, kind=RevisionKind.ADDITIVE, compatible_from=28),
    # Runtime-attested fresh-input markers add a private event-payload field and
    # an explicit SQL proof bit. Pre-31 readers do not remove that marker at the
    # public projection boundary, so they must not share a revision-31 database.
    Revision(revision=31, kind=RevisionKind.BREAKING, compatible_from=31),
    # Add the embedded eval catalog, immutable published results, and fenced run
    # lifecycle tables. Existing stores neither read nor write these new tables,
    # so the revision is additive; revision-32 EvalStore requires these tables.
    Revision(revision=32, kind=RevisionKind.ADDITIVE, compatible_from=31),
    # Add target-leading eval run catalog and claim indexes. Existing writers
    # already maintain target_key and every lifecycle column in these indexes.
    Revision(revision=33, kind=RevisionKind.ADDITIVE, compatible_from=31),
    # Persist the optional task availability gate and index eligible queue
    # selection. Pre-34 task workers ignore the gate and could claim future work
    # early, so they must not share a revision-34 database.
    Revision(revision=34, kind=RevisionKind.BREAKING, compatible_from=34),
    # Model-authored knowledge publication now commits an immutable operation
    # receipt beside its entry and chunks. Pre-35 workers can still compensate
    # ambiguous writes by deleting a shared deterministic entry id, so they must
    # not share a database with revision-35 knowledge writers.
    Revision(revision=35, kind=RevisionKind.BREAKING, compatible_from=35),
    # Every session now persists its immutable root invocation origin and immediate
    # execution source. Pre-36 writers cannot populate the required value, so they
    # must not share a revision-36 database.
    Revision(revision=36, kind=RevisionKind.BREAKING, compatible_from=36),
    # SQLite knowledge chunks now own a stable integer key shared with their FTS5
    # rows. Pre-37 SQLite writers do not preserve that relationship, so a mixed
    # deployment could strand stale search rows after an update or deletion. The
    # Postgres ledger advances without DDL, but the cross-backend compatibility
    # floor still prevents unsafe old SQLite writers from sharing a migrated DB.
    Revision(revision=37, kind=RevisionKind.BREAKING, compatible_from=37),
    # Add an immutable receipt table for the opt-in idempotent task
    # terminalization operation. Older task writers keep using the legacy
    # completion/failure methods and never claim replay safety, so the new table
    # does not change their write contract.
    Revision(revision=38, kind=RevisionKind.ADDITIVE, compatible_from=37),
    # Every task now persists immutable root invocation provenance. Pre-39 task
    # writers cannot populate the required value, so mixed-version task workers
    # are unsafe and populated historical task tables require a clean rebuild.
    Revision(revision=39, kind=RevisionKind.BREAKING, compatible_from=39),
    # Index the two live checkpoint markers that own queued-dispatch terminal
    # handoff reconciliation. Revision-40 workers also persist a new profiled
    # queued-dispatch task payload that older workers cannot interpret. Operators
    # must quiesce revision-39 and older producers/workers and settle their tasks
    # before migration; the compatibility floor prevents later old-binary startup
    # or rollback, but cannot revoke an already-running worker's cached schema.
    Revision(revision=40, kind=RevisionKind.BREAKING, compatible_from=40),
    # Knowledge publication receipts now retain the immutable authorization
    # projection of their entry. Without it, a receipt cannot be safely replayed
    # after hard deletion. Existing receipts are deliberately not inferred or
    # backfilled; populated receipt tables require an operator-approved rebuild.
    Revision(revision=41, kind=RevisionKind.BREAKING, compatible_from=41),
    # Replace mutable knowledge rows and replace-in-place chunks with stable
    # logical identities, immutable numbered revisions, revision-bound chunks,
    # and CAS publication receipts. This prerelease contract intentionally
    # refuses populated pre-revision knowledge databases instead of maintaining
    # a backfill or dual read/write path.
    Revision(revision=42, kind=RevisionKind.BREAKING, compatible_from=42),
    # Every canonical knowledge mutation now writes revision-bound evidence and
    # one metadata-only ordered change in the same transaction. Pre-43 writers
    # do not maintain that outbox, so mixed-version knowledge writers are unsafe.
    # The DDL is additive and preserves revision-42 knowledge; no historical
    # changes are fabricated during migration.
    Revision(revision=43, kind=RevisionKind.BREAKING, compatible_from=43),
    # Derived knowledge projections now use complete immutable embedding
    # identities and independently sequenced, CAS-fenced readiness events.
    # Canonical revision-43 knowledge is preserved, while pre-identity vector
    # rows are rebuildable derived data and are deliberately discarded.
    Revision(revision=44, kind=RevisionKind.BREAKING, compatible_from=44),
    # Retry-series workers persist cumulative authority on every attempt and
    # atomically create delayed successors with an immutable settlement receipt.
    # Older task workers would ignore that authority and could renew the series.
    Revision(revision=45, kind=RevisionKind.BREAKING, compatible_from=45),
    # Session stores now promise indexed, narrative-only transcript search.
    # Revision 46 installs the SQLite triggers/FTS table and PostgreSQL
    # application-computed projection/GIN index required to uphold that public
    # capability. Populated transcript tables must be recreated: the transition
    # deliberately has no historical projection backfill or compatibility path.
    Revision(revision=46, kind=RevisionKind.BREAKING, compatible_from=46),
    # Add an origin-aware immutable eval-result index plus actor-attributed,
    # idempotent baseline CAS records. Every fresh-result writer must maintain
    # the index atomically; pre-47 EvalStore workers would publish unindexed
    # results, so mixed-version writers are unsafe.
    Revision(revision=47, kind=RevisionKind.BREAKING, compatible_from=47),
    # Captured-only eval cases deliberately carry no runnable input. Their
    # catalog rows therefore record zero messages, while corpus-v1 fresh cases
    # retain the existing one-to-sixteen bound. Older EvalStore writers cannot
    # publish this representation, so mixed-version writers are unsafe.
    Revision(revision=48, kind=RevisionKind.BREAKING, compatible_from=48),
    # Revision 49 persists the complete verified-work authority lifecycle in
    # task stores. Pre-49 task workers ignore contract bindings and could
    # complete contracted work through an ordinary terminalization entrance,
    # so mixed-version task workers are unsafe even though existing ordinary
    # tasks remain valid.
    Revision(revision=49, kind=RevisionKind.BREAKING, compatible_from=49),
    # Revision 50 persists the authenticated eval invocation projection and bounded
    # execution contractions needed to recreate the same runtime authority after
    # a durable worker restart. Pre-50 workers would silently lose that contract,
    # so mixed-version eval workers are unsafe.
    Revision(revision=50, kind=RevisionKind.BREAKING, compatible_from=50),
    # Add bounded recall receipts, exact context-exposure lifecycle records,
    # and per-item exposure evidence. Revision-50 writers do not touch these
    # tables, so they remain compatible until runtime wiring
    # makes pre-dispatch evidence mandatory in a later breaking revision.
    Revision(revision=51, kind=RevisionKind.ADDITIVE, compatible_from=50),
    # Add first-class interaction-scoped targeted tool grants and their permanent
    # digest-only use bindings. Older session writers cannot preserve fork-reset,
    # pruning, export, or lifecycle evidence, so mixed-version operation is unsafe.
    # Populated pre-grant session stores must be recreated rather than carrying
    # incomplete pre-grant fork evidence or a permanent compatibility path.
    Revision(revision=52, kind=RevisionKind.BREAKING, compatible_from=52),
    # Revision 53 adds an independent immutable scenario-v2 document catalog.
    # Revision-52 writers do not touch this table, so mixed-version processes
    # remain safe while scenario-aware EvalStore implementations require the
    # new additive object.
    Revision(revision=53, kind=RevisionKind.ADDITIVE, compatible_from=52),
    # Runtime-attested resume and queued-input markers extend the private
    # input_contract payload to event types that pre-54 readers would expose as
    # ordinary public payload. Exact model-facing file digests also gain an
    # independent durable provenance bit. Mixed-version readers and session
    # writers are therefore unsafe.
    Revision(revision=54, kind=RevisionKind.BREAKING, compatible_from=54),
    # Rejected retry-cancellation reconciliation requests now bind their
    # idempotency keys durably without mutating the fenced task. Revision-54
    # task workers do not maintain that registry, so mixed-version task writers
    # could accept changed evidence under a previously rejected key.
    Revision(revision=55, kind=RevisionKind.BREAKING, compatible_from=55),
    # Controlled scenario runs persist their resumable execution cursor and any
    # operator approval against the fenced eval-run claim. Revision-55 writers
    # ignore the nullable column, so this is an additive compatibility step;
    # scenario-aware EvalStore implementations require revision 56 explicitly.
    Revision(revision=56, kind=RevisionKind.ADDITIVE, compatible_from=55),
    # Typed queued user messages retain multimodal artifact references across
    # durable delivery. Pre-57 workers would ignore that payload and deliver
    # only its text projection, so mixed-version session workers are unsafe.
    Revision(revision=57, kind=RevisionKind.BREAKING, compatible_from=57),
    # Completion-verifier claims and decisions now require a proposal-scoped
    # immutable verifier profile prepared before dispatch. Older task workers
    # cannot maintain that authority chain, so mixed-version operation is unsafe.
    Revision(revision=58, kind=RevisionKind.BREAKING, compatible_from=58),
    # Every verified-work contract now binds the exact application-owned result
    # resolver needed to reconstruct accepted result content. Existing contract
    # rows cannot acquire that authority by inference, so populated pre-59
    # verified-work registries are a clean prerelease break.
    Revision(revision=59, kind=RevisionKind.BREAKING, compatible_from=59),
    # Exact cross-entry knowledge lineage adds immutable revision-bound
    # relations, idempotent publication receipts, and relation events to the
    # canonical knowledge outbox. Populated pre-60 knowledge stores are
    # deliberately not rewritten or backfilled in this prerelease transition.
    Revision(revision=60, kind=RevisionKind.BREAKING, compatible_from=60),
    # Contract-bound execution now requires one durable admission intent and an
    # immutable history of process-aware execution claims before any governed
    # session work may begin. Pre-61 task workers do not maintain those fences,
    # so mixed-version task execution is unsafe.
    Revision(revision=61, kind=RevisionKind.BREAKING, compatible_from=61),
    # Deferred interaction input now carries an authenticated complete initial
    # transcript projection in addition to its source-only fallback, and
    # continuation receipts bind the exact predecessor admission. Pre-62
    # workers do not preserve those payload/authority fields and cannot share
    # the migrated store safely.
    Revision(revision=62, kind=RevisionKind.BREAKING, compatible_from=62),
    # Reviewed maintenance decisions bind exact proposals to atomic replacement,
    # archival, lineage, outbox, and receipt commits. Populated pre-63 knowledge
    # stores are deliberately not rewritten or backfilled.
    Revision(revision=63, kind=RevisionKind.BREAKING, compatible_from=63),
    # Add immutable authored-suite documents for Evals V3. Pre-64 writers do
    # not touch this independent catalog, so mixed-version processes remain
    # safe while suite-authoring EvalStore implementations require revision 64.
    Revision(revision=64, kind=RevisionKind.ADDITIVE, compatible_from=63),
    # Canonical knowledge-entry payload sizes make bounded reads enforceable
    # before entry content crosses the storage boundary. Existing populated
    # prerelease knowledge rows are deliberately not measured or backfilled.
    Revision(revision=65, kind=RevisionKind.BREAKING, compatible_from=65),
    # General local execution attempts add a durable tree-ownership fence to
    # task claiming and lease reclamation. Pre-66 task workers do not consult
    # that fence, so mixed-version task execution is unsafe.
    Revision(revision=66, kind=RevisionKind.BREAKING, compatible_from=66),
    # Accepted maintenance plans now publish an immutable pending replacement,
    # exact review proposal, and replay receipt in one transaction. Existing
    # prerelease knowledge rows are deliberately not inferred or backfilled.
    Revision(revision=67, kind=RevisionKind.BREAKING, compatible_from=67),
    # Add immutable fixed-evidence judge-calibration reports. Existing writers
    # do not touch this independent catalog, so revision-67 processes remain
    # compatible while calibration-aware EvalStore implementations require 68.
    Revision(revision=68, kind=RevisionKind.ADDITIVE, compatible_from=67),
    # Durable agent work contexts and per-agent/task recall checkpoints are new
    # independent authoritative records. Pre-69 binaries never read or write
    # these tables, so the transition is additive and needs no inferred data or
    # compatibility path.
    Revision(revision=69, kind=RevisionKind.ADDITIVE, compatible_from=67),
    # Interrupted-task worker release receipts are independent exact-operation
    # evidence. Older writers do not touch the table, so this remains additive.
    Revision(revision=70, kind=RevisionKind.ADDITIVE, compatible_from=67),
    # Checkpoint-aware recall delivery now stages the exact materialized result
    # and advances its checkpoint in one commit, then fences every delivery
    # claim and terminal handoff. Pre-71 checkpoint writers do not maintain
    # that atomic stage, so mixed-version operation is unsafe. Existing
    # checkpoints remain valid frontiers; no delivery records are inferred.
    Revision(revision=71, kind=RevisionKind.BREAKING, compatible_from=71),
    # Corpus execution may now persist per-run concurrency above 32 through the
    # shared signed-32-bit portable representation maximum. Widening the durable
    # eval-run constraint lets newer writers persist values that older binaries
    # reject, so mixed-version operation is unsafe.
    Revision(revision=72, kind=RevisionKind.BREAKING, compatible_from=72),
    # Idle-task recall subscriptions bind processing results to their exact
    # retrieval input and add scheduler-wake authority beside revision 71.
    # Mixed pre-73 processing-result writers would publish the old contract,
    # so this is a clean break. No subscription or wake state is inferred.
    Revision(revision=73, kind=RevisionKind.BREAKING, compatible_from=73),
    # Eval workers now checkpoint each terminal trial before aggregate
    # publication and reuse those slots after lease recovery. Authored-suite
    # launch parts also use durable concurrency lanes so their shared ceiling
    # holds across coordinators without serializing independent work. Pre-74
    # workers do not consult the recovery or lane fences, so mixed-version eval
    # execution is unsafe.
    Revision(revision=74, kind=RevisionKind.BREAKING, compatible_from=74),
    # High-level knowledge activation now records exact application authority
    # atomically beside generated publication or reviewed approval. Populated
    # pre-75 knowledge stores are rejected rather than assigned inferred
    # governance history; older writers do not maintain the new invariant.
    Revision(revision=75, kind=RevisionKind.BREAKING, compatible_from=75),
    # Interrupted attached-task continuation now carries a durable, consumable
    # handoff generation on the task row. Pre-76 task workers neither publish nor
    # consume that fence and could reauthorize an ABA workerless snapshot, so
    # mixed-version task execution is unsafe.
    Revision(revision=76, kind=RevisionKind.BREAKING, compatible_from=76),
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
