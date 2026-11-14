"""``cayu_session_summaries``: one row per session, kept current by triggers.

A session's overview is derived from many sources: its row, its events, its
checkpoint, its children and the sessions it waits on. Triggers on each source
keep two kinds of column in the same transaction as the write, for every
writer, older binaries included:

- Copies of the session row's list columns (agent, environment, status, times,
  title, eval mark).
- ``source_generation``, a counter every write that can change the session's
  overview advances. A child's creation, deletion or relineage advances its
  parent's, because the parent counts its child runs. On Postgres it advances
  at most once per writing transaction, which is all a settlement in another
  transaction can tell apart.

The derived overview (state, outcome, usage, fork lineage, waits) is computed
by the reference reader, ``SessionOverviewReader``, and stored with the
generation it was computed from. ``cayu_session_summary_dependencies`` names
every other session the computation read, with the stamp (incarnation and
generation) it had before the read. Every session a computation reads is
recorded, including those it reads only to follow another, so the stamps cover
the whole chain. A settlement stores a computation only if the row's generation
is still the one it read, together with its dependencies. A stored overview is
current while ``derived_generation = source_generation`` and every dependency
still has its recorded stamp (``dependencies_current_sql``), so a list can
filter on ``state`` in SQL. Writers never update another session's row to
invalidate it, so writers to related sessions never wait on each other. ``derived_generation`` is ``NULL`` until the first
computation; the migration fills the trigger-maintained columns for every
retained session and leaves the derived columns to be computed on first read.

``usage_epoch`` advances when one of the session's events is changed or
deleted. The stored usage summary is extended with only the events after it
while the epoch is unchanged, and recomputed whole otherwise.

``artifact_count`` is maintained by the artifact index.
"""

from __future__ import annotations

#: The storage revision that adds ``cayu_session_summaries``. Several revisions
#: are in flight at once; this is the one number to change when renumbering.
SESSION_SUMMARY_REVISION = 155

_SQLITE_TITLE = (
    "CASE WHEN json_type({row}.metadata_json, '$.\"cayu:title\"') = 'text' "
    "THEN json_extract({row}.metadata_json, '$.\"cayu:title\"') END"
)
_SQLITE_EVAL = (
    "CASE WHEN json_type({row}.metadata_json, '$.\"cayu:eval_trial\"') = 'true' THEN 1 ELSE 0 END"
)

#: Summary columns copied from ``cayu_sessions``, with the SQLite expression over a
#: session row named ``{row}``.
_SQLITE_COPIED = (
    ("instance_id", "{row}.instance_id"),
    ("parent_session_id", "{row}.parent_session_id"),
    ("agent_name", "{row}.agent_name"),
    ("environment_name", "{row}.environment_name"),
    ("provider_name", "{row}.provider_name"),
    ("model", "{row}.model"),
    ("causal_budget_id", "{row}.causal_budget_id"),
    ("status", "{row}.status"),
    ("eval", _SQLITE_EVAL),
    ("title", _SQLITE_TITLE),
    ("created_at", "{row}.created_at"),
    ("updated_at", "{row}.updated_at"),
    ("last_activity_at", "{row}.last_activity_at"),
)

# A child's row changes its parent's child-run count only through these.
_SQLITE_RELINEAGE = (
    "OLD.instance_id IS NOT NEW.instance_id "
    "OR OLD.parent_session_id IS NOT NEW.parent_session_id "
    "OR OLD.metadata_json IS NOT NEW.metadata_json "
    "OR OLD.invocation_json IS NOT NEW.invocation_json"
)


def _sqlite_copied(row: str) -> tuple[tuple[str, str], ...]:
    return tuple((name, expression.format(row=row)) for name, expression in _SQLITE_COPIED)


def _sqlite_insert_from(row: str, source: str) -> str:
    columns = _sqlite_copied(row)
    names = ", ".join(name for name, _ in columns)
    values = ", ".join(expression for _, expression in columns)
    return f"""INSERT OR REPLACE INTO cayu_session_summaries (session_id, {names})
    SELECT {row}.id, {values}
    {source}"""


def _sqlite_touch(session: str) -> str:
    return (
        "UPDATE cayu_session_summaries SET source_generation = source_generation + 1 "
        f"WHERE session_id = {session}"
    )


_SQLITE_SESSION_SET = ",\n        ".join(
    f"{name} = {expression}" for name, expression in _sqlite_copied("NEW")
)

SQLITE_SESSION_SUMMARIES_DDL = f"""
CREATE TABLE IF NOT EXISTS cayu_session_summaries (
    session_id TEXT PRIMARY KEY,
    instance_id TEXT NOT NULL,
    parent_session_id TEXT,
    agent_name TEXT NOT NULL,
    environment_name TEXT,
    provider_name TEXT NOT NULL,
    model TEXT NOT NULL,
    causal_budget_id TEXT NOT NULL,
    status TEXT NOT NULL,
    eval INTEGER NOT NULL CHECK (eval IN (0, 1)),
    title TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    usage_epoch INTEGER NOT NULL DEFAULT 0,
    source_generation INTEGER NOT NULL DEFAULT 0,
    derived_generation INTEGER,
    state TEXT,
    forked INTEGER CHECK (forked IN (0, 1)),
    artifact_count INTEGER NOT NULL DEFAULT 0 CHECK (artifact_count >= 0),
    derived_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_unsettled
    ON cayu_session_summaries(session_id)
    WHERE derived_generation IS NULL OR derived_generation <> source_generation;

-- Lists select by overview state and counts select by creation time and status.
CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_state
    ON cayu_session_summaries(state, session_id);

CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_created
    ON cayu_session_summaries(created_at);

CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_status
    ON cayu_session_summaries(status, created_at);

-- Each session another session's overview read, with the stamp it had first.
-- A stored overview is current only while every one still has that stamp; a
-- writer never touches another session's row for it.
CREATE TABLE IF NOT EXISTS cayu_session_summary_dependencies (
    session_id TEXT NOT NULL,
    depends_on TEXT NOT NULL,
    instance_id TEXT,
    source_generation INTEGER,
    PRIMARY KEY (session_id, depends_on)
);

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_session_insert
AFTER INSERT ON cayu_sessions
FOR EACH ROW
BEGIN
    {_sqlite_insert_from("NEW", "")};
    {_sqlite_touch("NEW.parent_session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_session_update
AFTER UPDATE OF id, instance_id, agent_name, provider_name, model, parent_session_id,
    causal_budget_id, runtime_name, runtime_version, environment_name, status,
    created_at, updated_at, last_activity_at, run_epoch, invocation_json, metadata_json
ON cayu_sessions
FOR EACH ROW
BEGIN
    UPDATE cayu_session_summaries SET
        {_SQLITE_SESSION_SET},
        source_generation = source_generation + 1
    WHERE session_id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_session_relineage
AFTER UPDATE OF instance_id, parent_session_id, invocation_json, metadata_json
ON cayu_sessions
FOR EACH ROW
WHEN {_SQLITE_RELINEAGE}
BEGIN
    {_sqlite_touch("OLD.parent_session_id")};
    UPDATE cayu_session_summaries SET source_generation = source_generation + 1
    WHERE session_id = NEW.parent_session_id
      AND NEW.parent_session_id IS NOT OLD.parent_session_id;
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_session_delete
AFTER DELETE ON cayu_sessions
FOR EACH ROW
BEGIN
    DELETE FROM cayu_session_summaries WHERE session_id = OLD.id;
    DELETE FROM cayu_session_summary_dependencies WHERE session_id = OLD.id;
    {_sqlite_touch("OLD.parent_session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_event_insert
AFTER INSERT ON cayu_events
FOR EACH ROW
BEGIN
    {_sqlite_touch("NEW.session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_event_delete
AFTER DELETE ON cayu_events
FOR EACH ROW
BEGIN
    UPDATE cayu_session_summaries SET
        usage_epoch = usage_epoch + 1,
        source_generation = source_generation + 1
    WHERE session_id = OLD.session_id;
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_event_rewrite
AFTER UPDATE OF sequence, session_id, event_type, timestamp, payload_json ON cayu_events
FOR EACH ROW
BEGIN
    UPDATE cayu_session_summaries SET
        usage_epoch = usage_epoch + 1,
        source_generation = source_generation + 1
    WHERE session_id IN (OLD.session_id, NEW.session_id);
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_event_pending
AFTER UPDATE OF pending_action_lookup_key, pending_action_projection_json ON cayu_events
FOR EACH ROW
BEGIN
    UPDATE cayu_session_summaries SET source_generation = source_generation + 1
    WHERE session_id IN (OLD.session_id, NEW.session_id);
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_checkpoint_insert
AFTER INSERT ON cayu_checkpoints
FOR EACH ROW
BEGIN
    {_sqlite_touch("NEW.session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_checkpoint_update
AFTER UPDATE ON cayu_checkpoints
FOR EACH ROW
BEGIN
    UPDATE cayu_session_summaries SET source_generation = source_generation + 1
    WHERE session_id IN (OLD.session_id, NEW.session_id);
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_checkpoint_delete
AFTER DELETE ON cayu_checkpoints
FOR EACH ROW
BEGIN
    {_sqlite_touch("OLD.session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_ledger_insert
AFTER INSERT ON cayu_invocation_receipt_ledgers
FOR EACH ROW
BEGIN
    {_sqlite_touch("NEW.session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_ledger_update
AFTER UPDATE ON cayu_invocation_receipt_ledgers
FOR EACH ROW
BEGIN
    UPDATE cayu_session_summaries SET source_generation = source_generation + 1
    WHERE session_id IN (OLD.session_id, NEW.session_id);
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_ledger_delete
AFTER DELETE ON cayu_invocation_receipt_ledgers
FOR EACH ROW
BEGIN
    {_sqlite_touch("OLD.session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_operation_insert
AFTER INSERT ON cayu_session_operations
FOR EACH ROW
BEGIN
    {_sqlite_touch("NEW.session_id")};
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_operation_update
AFTER UPDATE ON cayu_session_operations
FOR EACH ROW
BEGIN
    UPDATE cayu_session_summaries SET source_generation = source_generation + 1
    WHERE session_id IN (OLD.session_id, NEW.session_id);
END;

CREATE TRIGGER IF NOT EXISTS cayu_session_summaries_operation_delete
AFTER DELETE ON cayu_session_operations
FOR EACH ROW
BEGIN
    {_sqlite_touch("OLD.session_id")};
END;

{_sqlite_insert_from("source", "FROM cayu_sessions AS source")};
"""


_POSTGRES_COPIED = (
    ("instance_id", "{row}.instance_id"),
    ("parent_session_id", "{row}.parent_session_id"),
    ("agent_name", "{row}.agent_name"),
    ("environment_name", "{row}.environment_name"),
    ("provider_name", "{row}.provider_name"),
    ("model", "{row}.model"),
    ("causal_budget_id", "{row}.causal_budget_id"),
    ("status", "{row}.status"),
    ("eval", "COALESCE({row}.metadata -> 'cayu:eval_trial' = 'true'::jsonb, false)"),
    (
        "title",
        "CASE WHEN jsonb_typeof({row}.metadata -> 'cayu:title') = 'string' "
        "THEN {row}.metadata ->> 'cayu:title' END",
    ),
    ("created_at", "{row}.created_at"),
    ("updated_at", "{row}.updated_at"),
    ("last_activity_at", "{row}.last_activity_at"),
)


def _postgres_copied(row: str) -> tuple[tuple[str, str], ...]:
    return tuple((name, expression.format(row=row)) for name, expression in _POSTGRES_COPIED)


def _postgres_insert_from(row: str, source: str, conflict: str) -> str:
    columns = _postgres_copied(row)
    names = ", ".join(name for name, _ in columns)
    values = ", ".join(expression for _, expression in columns)
    return f"""INSERT INTO cayu_session_summaries (session_id, {names})
    SELECT {row}.id, {values}
    {source}
    {conflict}"""


_POSTGRES_REPLACE = (
    "ON CONFLICT (session_id) DO UPDATE SET "
    + ", ".join(f"{name} = EXCLUDED.{name}" for name, _ in _POSTGRES_COPIED)
    + (
        ", usage_epoch = cayu_session_summaries.usage_epoch + 1"
        ", source_generation = cayu_session_summaries.source_generation + 1"
        ", generation_txid = txid_current(), epoch_txid = txid_current()"
        ", derived_generation = NULL, state = NULL, forked = NULL, derived_json = NULL"
    )
)

_POSTGRES_SESSION_SET = ",\n                ".join(
    f"{name} = {expression}" for name, expression in _postgres_copied("NEW")
)

_POSTGRES_TOUCH = "PERFORM cayu_session_summary_advance({session}, false)"

POSTGRES_SESSION_SUMMARIES_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS cayu_session_summaries (
        session_id TEXT PRIMARY KEY,
        instance_id TEXT NOT NULL,
        parent_session_id TEXT,
        agent_name TEXT NOT NULL,
        environment_name TEXT,
        provider_name TEXT NOT NULL,
        model TEXT NOT NULL,
        causal_budget_id TEXT NOT NULL,
        status TEXT NOT NULL,
        eval BOOLEAN NOT NULL,
        title TEXT,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        last_activity_at TIMESTAMPTZ NOT NULL,
        usage_epoch BIGINT NOT NULL DEFAULT 0,
        source_generation BIGINT NOT NULL DEFAULT 0,
        -- The transactions that last advanced the generation and the epoch.
        generation_txid BIGINT,
        epoch_txid BIGINT,
        derived_generation BIGINT,
        state TEXT,
        forked BOOLEAN,
        artifact_count BIGINT NOT NULL DEFAULT 0 CHECK (artifact_count >= 0),
        derived_json JSONB
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_unsettled
        ON cayu_session_summaries(session_id)
        WHERE derived_generation IS NULL OR derived_generation <> source_generation
    """,
    # Lists select by overview state and counts select by creation time and status.
    "CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_state "
    "ON cayu_session_summaries(state, session_id)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_created "
    "ON cayu_session_summaries(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_session_summaries_status "
    "ON cayu_session_summaries(status, created_at)",
    # Each session another session's overview read, with the stamp it had first.
    # A stored overview is current only while every one still has that stamp; a
    # writer never touches another session's row for it, so writers to related
    # sessions never wait on each other's summary rows.
    """
    CREATE TABLE IF NOT EXISTS cayu_session_summary_dependencies (
        session_id TEXT NOT NULL,
        depends_on TEXT NOT NULL,
        instance_id TEXT,
        source_generation BIGINT,
        PRIMARY KEY (session_id, depends_on)
    )
    """,
    # Postgres keeps every version of a row a transaction updates until it ends,
    # so a write of many events in one transaction would update the same summary
    # row once per event, each update slower than the last. A row's generation
    # and epoch advance at most once per transaction instead: either change is
    # all a later settlement has to see.
    """
    CREATE OR REPLACE FUNCTION cayu_session_summary_advance(target TEXT, epoch BOOLEAN)
    RETURNS VOID AS $$
    DECLARE
        current_txid BIGINT := txid_current();
    BEGIN
        UPDATE cayu_session_summaries SET
            source_generation = source_generation
                + CASE WHEN generation_txid IS DISTINCT FROM current_txid THEN 1 ELSE 0 END,
            generation_txid = current_txid,
            usage_epoch = usage_epoch
                + CASE WHEN epoch AND epoch_txid IS DISTINCT FROM current_txid THEN 1 ELSE 0 END,
            epoch_txid = CASE WHEN epoch THEN current_txid ELSE epoch_txid END
        WHERE session_id = target
          AND (generation_txid IS DISTINCT FROM current_txid
               OR (epoch AND epoch_txid IS DISTINCT FROM current_txid));
    END;
    $$ LANGUAGE plpgsql
    """,
    f"""
    CREATE OR REPLACE FUNCTION cayu_session_summary_session_changed()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'INSERT' THEN
            {_postgres_insert_from("NEW", "", _POSTGRES_REPLACE)};
            {_POSTGRES_TOUCH.format(session="NEW.parent_session_id")};
        ELSIF TG_OP = 'UPDATE' THEN
            -- Every update fires this; an event or transcript counter alone changes
            -- nothing summarized. Checked here, not with UPDATE OF, so the trigger
            -- holds no dependency on the columns.
            IF (OLD.id, OLD.instance_id, OLD.agent_name, OLD.provider_name, OLD.model,
                OLD.parent_session_id, OLD.causal_budget_id, OLD.runtime_name,
                OLD.runtime_version, OLD.environment_name, OLD.status, OLD.created_at,
                OLD.updated_at, OLD.last_activity_at, OLD.run_epoch, OLD.invocation,
                OLD.metadata)
                IS NOT DISTINCT FROM
               (NEW.id, NEW.instance_id, NEW.agent_name, NEW.provider_name, NEW.model,
                NEW.parent_session_id, NEW.causal_budget_id, NEW.runtime_name,
                NEW.runtime_version, NEW.environment_name, NEW.status, NEW.created_at,
                NEW.updated_at, NEW.last_activity_at, NEW.run_epoch, NEW.invocation,
                NEW.metadata)
            THEN
                RETURN NULL;
            END IF;
            UPDATE cayu_session_summaries SET
                {_POSTGRES_SESSION_SET},
                source_generation = source_generation
                    + CASE WHEN generation_txid IS DISTINCT FROM txid_current() THEN 1 ELSE 0 END,
                generation_txid = txid_current()
            WHERE session_id = NEW.id;
            IF OLD.instance_id IS DISTINCT FROM NEW.instance_id
                OR OLD.parent_session_id IS DISTINCT FROM NEW.parent_session_id
                OR OLD.metadata IS DISTINCT FROM NEW.metadata
                OR OLD.invocation IS DISTINCT FROM NEW.invocation
            THEN
                -- In ID order, so two relineages between the same parents lock
                -- them in the same order.
                PERFORM cayu_session_summary_advance(
                    LEAST(OLD.parent_session_id, NEW.parent_session_id), false
                );
                PERFORM cayu_session_summary_advance(
                    GREATEST(OLD.parent_session_id, NEW.parent_session_id), false
                );
            END IF;
        ELSE
            DELETE FROM cayu_session_summaries WHERE session_id = OLD.id;
            DELETE FROM cayu_session_summary_dependencies WHERE session_id = OLD.id;
            {_POSTGRES_TOUCH.format(session="OLD.parent_session_id")};
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    """
    CREATE OR REPLACE FUNCTION cayu_session_summary_event_changed()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP = 'INSERT' THEN
            PERFORM cayu_session_summary_advance(NEW.session_id, false);
        ELSIF TG_OP = 'UPDATE'
            AND (OLD.sequence, OLD.session_id, OLD.event_type, OLD.timestamp, OLD.payload,
                 OLD.event)
                IS NOT DISTINCT FROM
                (NEW.sequence, NEW.session_id, NEW.event_type, NEW.timestamp, NEW.payload,
                 NEW.event)
        THEN
            -- Only the pending-action lookup columns can still matter.
            IF (OLD.pending_action_lookup_key, OLD.pending_action_projection)
                IS DISTINCT FROM (NEW.pending_action_lookup_key, NEW.pending_action_projection)
            THEN
                PERFORM cayu_session_summary_advance(NEW.session_id, false);
            END IF;
        ELSE
            PERFORM cayu_session_summary_advance(OLD.session_id, true);
            IF TG_OP = 'UPDATE' AND NEW.session_id <> OLD.session_id THEN
                PERFORM cayu_session_summary_advance(NEW.session_id, true);
            END IF;
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    """
    CREATE OR REPLACE FUNCTION cayu_session_summary_session_row_changed()
    RETURNS TRIGGER AS $$
    BEGIN
        IF TG_OP IN ('UPDATE', 'DELETE') THEN
            PERFORM cayu_session_summary_advance(OLD.session_id, false);
        END IF;
        IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND NEW.session_id <> OLD.session_id) THEN
            PERFORM cayu_session_summary_advance(NEW.session_id, false);
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_session_summaries_session ON cayu_sessions",
    """
    CREATE TRIGGER cayu_session_summaries_session
    AFTER INSERT OR DELETE OR UPDATE ON cayu_sessions
    FOR EACH ROW
    EXECUTE FUNCTION cayu_session_summary_session_changed()
    """,
    "DROP TRIGGER IF EXISTS cayu_session_summaries_event ON cayu_events",
    """
    CREATE TRIGGER cayu_session_summaries_event
    AFTER INSERT OR DELETE OR UPDATE ON cayu_events
    FOR EACH ROW
    EXECUTE FUNCTION cayu_session_summary_event_changed()
    """,
    *(
        statement
        for table in (
            "cayu_checkpoints",
            "cayu_invocation_receipt_ledgers",
            "cayu_session_operations",
        )
        for statement in (
            f"DROP TRIGGER IF EXISTS cayu_session_summaries_source ON {table}",
            f"""
    CREATE TRIGGER cayu_session_summaries_source
    AFTER INSERT OR UPDATE OR DELETE ON {table}
    FOR EACH ROW
    EXECUTE FUNCTION cayu_session_summary_session_row_changed()
    """,
        )
    ),
    _postgres_insert_from(
        "source", "FROM cayu_sessions AS source", "ON CONFLICT (session_id) DO NOTHING"
    ),
)


def dependencies_current_sql(summary: str, *, postgres: bool) -> str:
    """SQL true while every dependency of the summary row ``summary`` has its recorded stamp."""

    distinct = "IS DISTINCT FROM" if postgres else "IS NOT"
    return (
        "NOT EXISTS (SELECT 1 FROM cayu_session_summary_dependencies AS dependency "
        "LEFT JOIN cayu_session_summaries AS other ON other.session_id = dependency.depends_on "
        f"WHERE dependency.session_id = {summary}.session_id "
        f"AND (other.instance_id {distinct} dependency.instance_id "
        f"OR other.source_generation {distinct} dependency.source_generation))"
    )


__all__ = [
    "POSTGRES_SESSION_SUMMARIES_DDL",
    "SESSION_SUMMARY_REVISION",
    "SQLITE_SESSION_SUMMARIES_DDL",
    "dependencies_current_sql",
]
