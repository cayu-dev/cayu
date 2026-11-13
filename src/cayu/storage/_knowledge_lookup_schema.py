"""Trigger-maintained knowledge lookup counters, kept beside the event log.

Knowledge lists show how often each entry was looked up and when last. Folding
every ``knowledge.lookup`` event on each read grows with history, so the session
stores keep two projections of those events, written by triggers in the same
transaction as the events themselves, for every writer, older binaries included:

``cayu_knowledge_lookups``
    One row per lookup event and distinct entry ID it returned: the narrow copy
    the totals are folded from, so a deleted or rewritten event can be taken out
    exactly, and ``cayu knowledge lookups check`` can compare both with the events.

``cayu_knowledge_lookup_totals``
    One row per namespace and entry: lookups, distinct sessions and the latest
    lookup time. Readers serve list pages from it with a primary-key read.

An event counts as ``fold_knowledge_lookup_totals`` counts it: its type is
``knowledge.lookup``, its payload ``namespace`` is a string and its ``entry_ids``
an array, and each distinct string ID in it counts once.

PostgreSQL writers run concurrently. A lookup's per-event rows are written and
deleted at once, but the totals change only at commit: a deferred trigger folds
every row the transaction added, and every counted row it deleted (queued in
``cayu_knowledge_lookup_removals``), in one statement in key order. A
transaction holds no totals row lock before that statement, and takes all of
them in it, so concurrent writers can't deadlock on them, however many events
they write or delete. Writers of the same session already serialize on its row,
so the distinct-session count is exact. A totals row reaches zero only inside
the fold, which deletes it before commit.
"""

from __future__ import annotations

from cayu.events import EventType

#: The storage revision that adds these tables. Kept here and in
#: ``migrations.REVISIONS``; the schema histories key their steps by it.
KNOWLEDGE_LOOKUP_TOTALS_REVISION = 152

_LOOKUP = str(EventType.KNOWLEDGE_LOOKUP)
#: Transaction-local setting: ``folded`` once this transaction's changes are in
#: the totals, cleared by every later change to the lookup rows.
_FOLDED = "cayu.knowledge_lookups_folded"


def _sqlite_insert_lookups(row: str) -> str:
    return f"""
    INSERT OR IGNORE INTO cayu_knowledge_lookups (
        sequence, entry_id, namespace, session_id, looked_up_at
    )
    SELECT DISTINCT
        {row}.sequence, returned.value,
        json_extract({row}.payload_json, '$.namespace'),
        {row}.session_id, {row}.timestamp
    FROM json_each({row}.payload_json, '$.entry_ids') AS returned
    WHERE {row}.event_type = '{_LOOKUP}'
      AND json_type({row}.payload_json, '$.namespace') = 'text'
      AND json_type({row}.payload_json, '$.entry_ids') = 'array'
      AND returned.type = 'text';
    """


SQLITE_KNOWLEDGE_LOOKUP_DDL = f"""
CREATE TABLE IF NOT EXISTS cayu_knowledge_lookups (
    sequence INTEGER NOT NULL,
    entry_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    session_id TEXT NOT NULL,
    looked_up_at TEXT NOT NULL,
    PRIMARY KEY (sequence, entry_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_lookups_entry_time
    ON cayu_knowledge_lookups(namespace, entry_id, looked_up_at);

CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_lookups_entry_session
    ON cayu_knowledge_lookups(namespace, entry_id, session_id);

CREATE TABLE IF NOT EXISTS cayu_knowledge_lookup_totals (
    namespace TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    lookups INTEGER NOT NULL CHECK (lookups > 0),
    sessions INTEGER NOT NULL CHECK (sessions > 0),
    last_lookup_at TEXT NOT NULL,
    PRIMARY KEY (namespace, entry_id)
) WITHOUT ROWID;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_lookup_totals_add
AFTER INSERT ON cayu_knowledge_lookups
FOR EACH ROW
BEGIN
    INSERT INTO cayu_knowledge_lookup_totals (
        namespace, entry_id, lookups, sessions, last_lookup_at
    )
    VALUES (NEW.namespace, NEW.entry_id, 1, 1, NEW.looked_up_at)
    ON CONFLICT (namespace, entry_id) DO UPDATE SET
        lookups = lookups + 1,
        sessions = sessions + (NOT EXISTS (
            SELECT 1 FROM cayu_knowledge_lookups AS other
            WHERE other.namespace = NEW.namespace
              AND other.entry_id = NEW.entry_id
              AND other.session_id = NEW.session_id
              AND other.sequence <> NEW.sequence
        )),
        last_lookup_at = max(last_lookup_at, excluded.last_lookup_at);
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_lookup_totals_remove
AFTER DELETE ON cayu_knowledge_lookups
FOR EACH ROW
BEGIN
    UPDATE cayu_knowledge_lookup_totals SET
        lookups = lookups - 1,
        sessions = sessions - (NOT EXISTS (
            SELECT 1 FROM cayu_knowledge_lookups AS other
            WHERE other.namespace = OLD.namespace
              AND other.entry_id = OLD.entry_id
              AND other.session_id = OLD.session_id
        )),
        last_lookup_at = COALESCE((
            SELECT MAX(other.looked_up_at) FROM cayu_knowledge_lookups AS other
            WHERE other.namespace = OLD.namespace AND other.entry_id = OLD.entry_id
        ), last_lookup_at)
    WHERE namespace = OLD.namespace AND entry_id = OLD.entry_id AND lookups > 1;
    DELETE FROM cayu_knowledge_lookup_totals
    WHERE namespace = OLD.namespace AND entry_id = OLD.entry_id AND NOT EXISTS (
        SELECT 1 FROM cayu_knowledge_lookups AS other
        WHERE other.namespace = OLD.namespace AND other.entry_id = OLD.entry_id
    );
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_lookups_insert
AFTER INSERT ON cayu_events
FOR EACH ROW
WHEN NEW.event_type = '{_LOOKUP}'
BEGIN
    {_sqlite_insert_lookups("NEW")}
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_lookups_delete
AFTER DELETE ON cayu_events
FOR EACH ROW
WHEN OLD.event_type = '{_LOOKUP}'
BEGIN
    DELETE FROM cayu_knowledge_lookups WHERE sequence = OLD.sequence;
END;

CREATE TRIGGER IF NOT EXISTS cayu_knowledge_lookups_update
AFTER UPDATE OF sequence, session_id, event_type, timestamp, payload_json ON cayu_events
FOR EACH ROW
WHEN OLD.event_type = '{_LOOKUP}' OR NEW.event_type = '{_LOOKUP}'
BEGIN
    DELETE FROM cayu_knowledge_lookups WHERE sequence = OLD.sequence;
    {_sqlite_insert_lookups("NEW")}
END;

INSERT OR IGNORE INTO cayu_knowledge_lookups (
    sequence, entry_id, namespace, session_id, looked_up_at
)
SELECT DISTINCT
    source.sequence, returned.value, json_extract(source.payload_json, '$.namespace'),
    source.session_id, source.timestamp
FROM cayu_events AS source, json_each(source.payload_json, '$.entry_ids') AS returned
WHERE source.event_type = '{_LOOKUP}'
  AND json_type(source.payload_json, '$.namespace') = 'text'
  AND json_type(source.payload_json, '$.entry_ids') = 'array'
  AND returned.type = 'text';
"""
"""Revision 152 for SQLite. The backfill's inserts fire the totals trigger, so
the totals come out exactly as a writer would have left them."""


def _postgres_returned(row: str) -> str:
    return f"""
        SELECT DISTINCT
            {row}.sequence, returned.value #>> '{{}}' AS entry_id,
            {row}.payload ->> 'namespace' AS namespace,
            {row}.session_id, {row}.timestamp
        FROM jsonb_array_elements(
            CASE WHEN jsonb_typeof({row}.payload -> 'entry_ids') = 'array'
                 THEN {row}.payload -> 'entry_ids' ELSE '[]'::jsonb END
        ) AS returned(value)
        WHERE {row}.event_type = '{_LOOKUP}'
          AND jsonb_typeof({row}.payload -> 'namespace') = 'string'
          AND jsonb_typeof(returned.value) = 'string'"""


POSTGRES_KNOWLEDGE_LOOKUP_DDL: tuple[str, ...] = (
    # Writers wait while the retained lookups are copied, so none is counted twice.
    "LOCK TABLE cayu_events IN SHARE MODE",
    """
    CREATE TABLE IF NOT EXISTS cayu_knowledge_lookups (
        sequence BIGINT NOT NULL,
        entry_id TEXT NOT NULL,
        namespace TEXT NOT NULL,
        session_id TEXT NOT NULL,
        looked_up_at TIMESTAMPTZ NOT NULL,
        counted BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY (sequence, entry_id)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_lookups_entry_time
        ON cayu_knowledge_lookups(namespace, entry_id, looked_up_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_lookups_entry_session
        ON cayu_knowledge_lookups(namespace, entry_id, session_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_cayu_knowledge_lookups_uncounted
        ON cayu_knowledge_lookups(namespace, entry_id) WHERE NOT counted
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_knowledge_lookup_totals (
        namespace TEXT NOT NULL,
        entry_id TEXT NOT NULL,
        lookups BIGINT NOT NULL CHECK (lookups >= 0),
        sessions BIGINT NOT NULL CHECK (sessions >= 0),
        last_lookup_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (namespace, entry_id)
    )
    """,
    # Empty outside a transaction: the fold at commit takes its own rows out.
    """
    CREATE TABLE IF NOT EXISTS cayu_knowledge_lookup_removals (
        sequence BIGINT NOT NULL,
        entry_id TEXT NOT NULL,
        namespace TEXT NOT NULL,
        session_id TEXT NOT NULL,
        looked_up_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (sequence, entry_id)
    )
    """,
    f"""
    INSERT INTO cayu_knowledge_lookups (
        sequence, entry_id, namespace, session_id, looked_up_at, counted
    )
    SELECT returned.*, TRUE
    FROM cayu_events AS source
    CROSS JOIN LATERAL ({_postgres_returned("source")}) AS returned
    WHERE source.event_type = '{_LOOKUP}'
    ON CONFLICT (sequence, entry_id) DO NOTHING
    """,
    """
    INSERT INTO cayu_knowledge_lookup_totals (
        namespace, entry_id, lookups, sessions, last_lookup_at
    )
    SELECT namespace, entry_id, COUNT(*), COUNT(DISTINCT session_id), MAX(looked_up_at)
    FROM cayu_knowledge_lookups
    GROUP BY namespace, entry_id
    ON CONFLICT (namespace, entry_id) DO UPDATE SET
        lookups = EXCLUDED.lookups,
        sessions = EXCLUDED.sessions,
        last_lookup_at = EXCLUDED.last_lookup_at
    """,
    f"""
    CREATE OR REPLACE FUNCTION cayu_track_knowledge_lookup()
    RETURNS TRIGGER
    SET search_path FROM CURRENT
    AS $$
    BEGIN
        PERFORM set_config('{_FOLDED}', '', TRUE);
        IF TG_OP IN ('UPDATE', 'DELETE') THEN
            DELETE FROM cayu_knowledge_lookups WHERE sequence = OLD.sequence;
        END IF;
        IF TG_OP IN ('INSERT', 'UPDATE') AND NEW.event_type = '{_LOOKUP}' THEN
            INSERT INTO cayu_knowledge_lookups (
                sequence, entry_id, namespace, session_id, looked_up_at
            )
            {_postgres_returned("NEW")}
            ON CONFLICT (sequence, entry_id) DO NOTHING;
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    # Fold this transaction's changes into the totals at commit: its uncounted
    # rows, and the counted rows it deleted (queued in the removals table). One
    # statement takes every totals row lock the transaction needs, in key order;
    # the two after it only touch rows it already holds. Rows other writers have
    # not committed are invisible here, and every writer folds its own before it
    # commits.
    f"""
    CREATE OR REPLACE FUNCTION cayu_fold_knowledge_lookups()
    RETURNS TRIGGER
    SET search_path FROM CURRENT
    AS $$
    DECLARE
        emptied_namespaces TEXT[];
        emptied_entries TEXT[];
        shrunk_namespaces TEXT[];
        shrunk_entries TEXT[];
    BEGIN
        -- Commit runs this once per queued row; the first run folds them all.
        IF current_setting('{_FOLDED}', TRUE) = 'folded' THEN
            RETURN NULL;
        END IF;
        PERFORM set_config('{_FOLDED}', 'folded', TRUE);
        WITH added_rows AS (
            UPDATE cayu_knowledge_lookups SET counted = TRUE
            WHERE NOT counted
            RETURNING namespace, entry_id, session_id, looked_up_at
        ),
        removed_rows AS (
            DELETE FROM cayu_knowledge_lookup_removals
            RETURNING namespace, entry_id, session_id, looked_up_at
        ),
        added AS (
            SELECT namespace, entry_id, COUNT(*) AS lookups, MAX(looked_up_at) AS last_at
            FROM added_rows GROUP BY namespace, entry_id
        ),
        removed AS (
            SELECT namespace, entry_id, COUNT(*) AS lookups, MAX(looked_up_at) AS last_at
            FROM removed_rows GROUP BY namespace, entry_id
        ),
        -- A session counts for an entry before this transaction if it had a
        -- counted row (still here, or removed now), and after if any row is left.
        -- This statement's snapshot sees its own rows still uncounted.
        session_changes AS (
            SELECT pair.namespace, pair.entry_id, SUM(
                (EXISTS (
                    SELECT 1 FROM cayu_knowledge_lookups AS kept
                    WHERE kept.namespace = pair.namespace
                      AND kept.entry_id = pair.entry_id
                      AND kept.session_id = pair.session_id
                ))::INTEGER
                - (EXISTS (
                    SELECT 1 FROM cayu_knowledge_lookups AS earlier
                    WHERE earlier.namespace = pair.namespace
                      AND earlier.entry_id = pair.entry_id
                      AND earlier.session_id = pair.session_id
                      AND earlier.counted
                ) OR EXISTS (
                    SELECT 1 FROM removed_rows AS gone
                    WHERE gone.namespace = pair.namespace
                      AND gone.entry_id = pair.entry_id
                      AND gone.session_id = pair.session_id
                ))::INTEGER
            ) AS sessions
            FROM (
                SELECT namespace, entry_id, session_id FROM added_rows
                UNION
                SELECT namespace, entry_id, session_id FROM removed_rows
            ) AS pair
            GROUP BY pair.namespace, pair.entry_id
        ),
        changes AS (
            SELECT namespace, entry_id,
                   COALESCE(added.lookups, 0) - COALESCE(removed.lookups, 0) AS lookups,
                   session_changes.sessions,
                   added.last_at AS added_last_at,
                   removed.last_at AS removed_last_at
            FROM added
            FULL JOIN removed USING (namespace, entry_id)
            JOIN session_changes USING (namespace, entry_id)
        ),
        applied AS (
            INSERT INTO cayu_knowledge_lookup_totals AS totals (
                namespace, entry_id, lookups, sessions, last_lookup_at
            )
            SELECT namespace, entry_id, GREATEST(lookups, 0), GREATEST(sessions, 0),
                   COALESCE(added_last_at, removed_last_at)
            FROM changes
            ORDER BY namespace, entry_id
            ON CONFLICT (namespace, entry_id) DO UPDATE SET
                lookups = totals.lookups + (
                    SELECT changes.lookups FROM changes
                    WHERE changes.namespace = EXCLUDED.namespace
                      AND changes.entry_id = EXCLUDED.entry_id
                ),
                sessions = totals.sessions + (
                    SELECT changes.sessions FROM changes
                    WHERE changes.namespace = EXCLUDED.namespace
                      AND changes.entry_id = EXCLUDED.entry_id
                ),
                last_lookup_at = GREATEST(totals.last_lookup_at, EXCLUDED.last_lookup_at)
            RETURNING totals.namespace, totals.entry_id, totals.lookups
        )
        SELECT
            (SELECT array_agg(namespace ORDER BY namespace, entry_id)
             FROM applied WHERE lookups = 0),
            (SELECT array_agg(entry_id ORDER BY namespace, entry_id)
             FROM applied WHERE lookups = 0),
            (SELECT array_agg(namespace ORDER BY namespace, entry_id)
             FROM changes WHERE removed_last_at IS NOT NULL),
            (SELECT array_agg(entry_id ORDER BY namespace, entry_id)
             FROM changes WHERE removed_last_at IS NOT NULL)
        INTO emptied_namespaces, emptied_entries, shrunk_namespaces, shrunk_entries;

        IF shrunk_namespaces IS NOT NULL THEN
            -- A removal may have taken the newest lookup. This transaction holds
            -- these rows, so no other writer's lookups of them have committed
            -- since, and a new snapshot sees every committed one.
            UPDATE cayu_knowledge_lookup_totals AS totals SET
                last_lookup_at = COALESCE((
                    SELECT MAX(other.looked_up_at) FROM cayu_knowledge_lookups AS other
                    WHERE other.namespace = totals.namespace
                      AND other.entry_id = totals.entry_id
                ), totals.last_lookup_at)
            FROM unnest(shrunk_namespaces, shrunk_entries) AS shrunk(namespace, entry_id)
            WHERE totals.namespace = shrunk.namespace
              AND totals.entry_id = shrunk.entry_id
              AND totals.lookups > 0;
        END IF;
        IF emptied_namespaces IS NOT NULL THEN
            DELETE FROM cayu_knowledge_lookup_totals AS totals
            USING unnest(emptied_namespaces, emptied_entries) AS emptied(namespace, entry_id)
            WHERE totals.namespace = emptied.namespace
              AND totals.entry_id = emptied.entry_id
              AND totals.lookups = 0;
        END IF;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    # A deleted counted row waits in the removals table for the fold at commit,
    # so deleting events takes no totals row locks before then.
    f"""
    CREATE OR REPLACE FUNCTION cayu_queue_knowledge_lookup_removals()
    RETURNS TRIGGER
    SET search_path FROM CURRENT
    AS $$
    BEGIN
        PERFORM set_config('{_FOLDED}', '', TRUE);
        INSERT INTO cayu_knowledge_lookup_removals (
            sequence, entry_id, namespace, session_id, looked_up_at
        )
        SELECT sequence, entry_id, namespace, session_id, looked_up_at
        FROM removed_lookups
        WHERE counted;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_lookups_fold ON cayu_knowledge_lookups",
    """
    CREATE CONSTRAINT TRIGGER cayu_knowledge_lookups_fold
    AFTER INSERT ON cayu_knowledge_lookups
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW
    WHEN (NOT NEW.counted)
    EXECUTE FUNCTION cayu_fold_knowledge_lookups()
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_lookup_removals_fold ON cayu_knowledge_lookup_removals",
    """
    CREATE CONSTRAINT TRIGGER cayu_knowledge_lookup_removals_fold
    AFTER INSERT ON cayu_knowledge_lookup_removals
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW
    EXECUTE FUNCTION cayu_fold_knowledge_lookups()
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_lookups_unfold ON cayu_knowledge_lookups",
    """
    CREATE TRIGGER cayu_knowledge_lookups_unfold
    AFTER DELETE ON cayu_knowledge_lookups
    REFERENCING OLD TABLE AS removed_lookups
    FOR EACH STATEMENT
    EXECUTE FUNCTION cayu_queue_knowledge_lookup_removals()
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_lookups_insert ON cayu_events",
    f"""
    CREATE TRIGGER cayu_knowledge_lookups_insert
    AFTER INSERT ON cayu_events
    FOR EACH ROW
    WHEN (NEW.event_type = '{_LOOKUP}')
    EXECUTE FUNCTION cayu_track_knowledge_lookup()
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_lookups_delete ON cayu_events",
    f"""
    CREATE TRIGGER cayu_knowledge_lookups_delete
    AFTER DELETE ON cayu_events
    FOR EACH ROW
    WHEN (OLD.event_type = '{_LOOKUP}')
    EXECUTE FUNCTION cayu_track_knowledge_lookup()
    """,
    "DROP TRIGGER IF EXISTS cayu_knowledge_lookups_update ON cayu_events",
    f"""
    CREATE TRIGGER cayu_knowledge_lookups_update
    AFTER UPDATE OF sequence, session_id, event_type, timestamp, payload ON cayu_events
    FOR EACH ROW
    WHEN (OLD.event_type = '{_LOOKUP}' OR NEW.event_type = '{_LOOKUP}')
    EXECUTE FUNCTION cayu_track_knowledge_lookup()
    """,
)
"""Revision 152 for PostgreSQL: tables, the backfill, then the triggers."""
