"""Revision-156 summary tables that the Evals pages read instead of whole documents.

Every table here is a projection of an eval document the store already keeps,
written in the same transaction as that document:

- ``cayu_eval_suite_catalog``: one row per corpus suite, with the suite and its
  case specifications cut out of the corpus document. Triggers on
  ``cayu_eval_suites`` maintain it for every writer, older binaries included.
- ``cayu_eval_trial_checkpoint_summaries``: the status, score, diagnostic code
  and size of each trial checkpoint, so a live run view never parses a
  checkpoint document. Triggers on ``cayu_eval_run_trial_checkpoints`` maintain
  it, and it is deleted with its checkpoint.
- ``cayu_eval_result_summaries`` and ``cayu_eval_case_outcomes``: what a run view
  shows about one immutable result, and about each of its cases. The store
  writes them with the result record, because their content is the result's
  canonical presentation, which only Python computes. A result that an older
  binary published has none; readers then derive the same rows from the result
  document, and ``cayu eval runs rebuild-summaries --write`` writes them.
- ``cayu_eval_result_comparisons``: the comparison of two immutable results, a
  pure function of the pair. Readers record it the first time they compare a
  pair, and publication records it against the suite baseline.

Summary rows of a result are deleted with its result record. Each projection has
a table of its own, so the revision stays additive: binaries from revision 146 on
check the exact shape of the existing eval tables and still open the database.
"""

from __future__ import annotations

EVAL_VIEW_SUMMARIES_REVISION = 156

_CATALOG_COLUMNS = (
    "corpus_revision, suite_id, target_key, suite_revision, name, description, "
    "case_count, suite_json, cases_json, created_at"
)
_CHECKPOINT_SUMMARY_COLUMNS = (
    "run_id, case_id, trial_number, status, score, diagnostic_code, document_bytes"
)


def _sqlite_catalog_select(suite: str) -> str:
    """Catalog values of suite row ``suite`` joined to its corpus as ``corpora``."""
    return f"""
        SELECT
            corpora.revision, {suite}.suite_id, corpora.target_key, {suite}.suite_revision,
            {suite}.name, {suite}.description, {suite}.case_count,
            (SELECT json(item.value) FROM json_each(corpora.document_json, '$.suites') AS item
             WHERE json_extract(item.value, '$.id') = {suite}.suite_id),
            (SELECT json_group_array(json(item.value))
             FROM json_each(corpora.document_json, '$.cases') AS item
             WHERE json_extract(item.value, '$.suite_id') = {suite}.suite_id),
            corpora.created_at
    """


def _sqlite_checkpoint_select(checkpoint: str) -> str:
    return f"""
        SELECT
            {checkpoint}.run_id, {checkpoint}.case_id, {checkpoint}.trial_number,
            json_extract({checkpoint}.checkpoint_json, '$.result.status'),
            json_extract({checkpoint}.checkpoint_json, '$.result.score'),
            json_extract({checkpoint}.checkpoint_json, '$.public_data.diagnostic_code'),
            {checkpoint}.document_bytes
    """


SQLITE_EVAL_VIEW_SUMMARIES_DDL = f"""
    CREATE TABLE IF NOT EXISTS cayu_eval_suite_catalog (
        corpus_revision TEXT NOT NULL,
        suite_id TEXT COLLATE BINARY NOT NULL,
        target_key TEXT NOT NULL,
        suite_revision TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        case_count INTEGER NOT NULL,
        suite_json TEXT NOT NULL CHECK (json_valid(suite_json)),
        cases_json TEXT NOT NULL CHECK (json_valid(cases_json)),
        created_at TEXT NOT NULL,
        PRIMARY KEY (corpus_revision, suite_id),
        FOREIGN KEY (corpus_revision, suite_id)
            REFERENCES cayu_eval_suites(corpus_revision, suite_id) ON DELETE CASCADE
    );
    CREATE TRIGGER IF NOT EXISTS cayu_eval_suite_catalog_insert
    AFTER INSERT ON cayu_eval_suites
    BEGIN
        INSERT OR REPLACE INTO cayu_eval_suite_catalog ({_CATALOG_COLUMNS})
        {_sqlite_catalog_select("NEW")}
        FROM cayu_eval_corpora AS corpora WHERE corpora.revision = NEW.corpus_revision;
    END;
    INSERT OR IGNORE INTO cayu_eval_suite_catalog ({_CATALOG_COLUMNS})
    {_sqlite_catalog_select("suites")}
    FROM cayu_eval_suites AS suites
    JOIN cayu_eval_corpora AS corpora ON corpora.revision = suites.corpus_revision;

    CREATE TABLE IF NOT EXISTS cayu_eval_trial_checkpoint_summaries (
        run_id TEXT COLLATE BINARY NOT NULL,
        case_id TEXT COLLATE BINARY NOT NULL,
        trial_number INTEGER NOT NULL,
        -- Copied from the document as is: the trigger never refuses a checkpoint
        -- the checkpoint table accepts, and readers validate what they read.
        status TEXT,
        score REAL,
        diagnostic_code TEXT,
        document_bytes INTEGER NOT NULL,
        PRIMARY KEY (run_id, case_id, trial_number),
        FOREIGN KEY (run_id, case_id, trial_number)
            REFERENCES cayu_eval_run_trial_checkpoints(run_id, case_id, trial_number)
            ON DELETE CASCADE
    );
    CREATE TRIGGER IF NOT EXISTS cayu_eval_trial_checkpoint_summaries_insert
    AFTER INSERT ON cayu_eval_run_trial_checkpoints
    BEGIN
        INSERT OR REPLACE INTO cayu_eval_trial_checkpoint_summaries ({_CHECKPOINT_SUMMARY_COLUMNS})
        {_sqlite_checkpoint_select("NEW")};
    END;
    CREATE TRIGGER IF NOT EXISTS cayu_eval_trial_checkpoint_summaries_update
    AFTER UPDATE OF checkpoint_json, document_bytes ON cayu_eval_run_trial_checkpoints
    BEGIN
        INSERT OR REPLACE INTO cayu_eval_trial_checkpoint_summaries ({_CHECKPOINT_SUMMARY_COLUMNS})
        {_sqlite_checkpoint_select("NEW")};
    END;
    INSERT OR IGNORE INTO cayu_eval_trial_checkpoint_summaries ({_CHECKPOINT_SUMMARY_COLUMNS})
    {_sqlite_checkpoint_select("checkpoints")}
    FROM cayu_eval_run_trial_checkpoints AS checkpoints;

    CREATE TABLE IF NOT EXISTS cayu_eval_result_summaries (
        revision TEXT COLLATE BINARY PRIMARY KEY
            REFERENCES cayu_eval_result_records(revision) ON DELETE CASCADE,
        summary_json TEXT NOT NULL CHECK (json_valid(summary_json))
    );
    CREATE TABLE IF NOT EXISTS cayu_eval_case_outcomes (
        result_revision TEXT COLLATE BINARY NOT NULL
            REFERENCES cayu_eval_result_summaries(revision) ON DELETE CASCADE,
        case_id TEXT COLLATE BINARY NOT NULL,
        status TEXT NOT NULL,
        score REAL,
        passed_trials INTEGER NOT NULL,
        total_trials INTEGER NOT NULL,
        duration_ms INTEGER,
        outcome_json TEXT NOT NULL CHECK (json_valid(outcome_json)),
        evidence_json TEXT CHECK (evidence_json IS NULL OR json_valid(evidence_json)),
        PRIMARY KEY (result_revision, case_id)
    );
    CREATE TABLE IF NOT EXISTS cayu_eval_result_comparisons (
        current_revision TEXT COLLATE BINARY NOT NULL
            REFERENCES cayu_eval_result_records(revision) ON DELETE CASCADE,
        previous_revision TEXT COLLATE BINARY NOT NULL
            REFERENCES cayu_eval_result_records(revision) ON DELETE CASCADE,
        comparison_json TEXT NOT NULL CHECK (json_valid(comparison_json)),
        PRIMARY KEY (current_revision, previous_revision)
    );
    CREATE INDEX IF NOT EXISTS idx_cayu_eval_result_comparisons_previous
        ON cayu_eval_result_comparisons(previous_revision);
    CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_result_revision
        ON cayu_eval_runs(result_revision);
"""


def _postgres_catalog_select(suite: str) -> str:
    return f"""
        SELECT
            corpora.revision, {suite}.suite_id, corpora.target_key, {suite}.suite_revision,
            {suite}.name, {suite}.description, {suite}.case_count,
            (SELECT item FROM jsonb_array_elements(corpora.document::jsonb -> 'suites') AS item
             WHERE item ->> 'id' = {suite}.suite_id LIMIT 1),
            COALESCE((
                SELECT jsonb_agg(item ORDER BY position)
                FROM jsonb_array_elements(corpora.document::jsonb -> 'cases')
                    WITH ORDINALITY AS cases(item, position)
                WHERE item ->> 'suite_id' = {suite}.suite_id
            ), '[]'::jsonb),
            corpora.created_at
    """


def _postgres_checkpoint_select(checkpoint: str) -> str:
    return f"""
        SELECT
            {checkpoint}.run_id, {checkpoint}.case_id, {checkpoint}.trial_number,
            document #>> '{{result,status}}',
            CASE WHEN jsonb_typeof(document #> '{{result,score}}') = 'number'
                THEN (document #>> '{{result,score}}')::double precision END,
            document #>> '{{public_data,diagnostic_code}}',
            {checkpoint}.document_bytes
        FROM (SELECT {checkpoint}.checkpoint_json::jsonb AS document) AS parsed
    """


POSTGRES_EVAL_VIEW_SUMMARIES_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS cayu_eval_suite_catalog (
        corpus_revision TEXT NOT NULL,
        suite_id TEXT COLLATE "C" NOT NULL,
        target_key TEXT NOT NULL,
        suite_revision TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT,
        case_count INTEGER NOT NULL,
        suite_json JSONB NOT NULL,
        cases_json JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (corpus_revision, suite_id),
        FOREIGN KEY (corpus_revision, suite_id)
            REFERENCES cayu_eval_suites(corpus_revision, suite_id) ON DELETE CASCADE
    )
    """,
    f"""
    CREATE OR REPLACE FUNCTION cayu_track_eval_suite_catalog()
    RETURNS TRIGGER
    SET search_path FROM CURRENT
    AS $$
    BEGIN
        INSERT INTO cayu_eval_suite_catalog ({_CATALOG_COLUMNS})
        {_postgres_catalog_select("NEW")}
        FROM cayu_eval_corpora AS corpora WHERE corpora.revision = NEW.corpus_revision
        ON CONFLICT (corpus_revision, suite_id) DO NOTHING;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_eval_suite_catalog_insert ON cayu_eval_suites",
    """
    CREATE TRIGGER cayu_eval_suite_catalog_insert
    AFTER INSERT ON cayu_eval_suites
    FOR EACH ROW EXECUTE FUNCTION cayu_track_eval_suite_catalog()
    """,
    f"""
    INSERT INTO cayu_eval_suite_catalog ({_CATALOG_COLUMNS})
    {_postgres_catalog_select("suites")}
    FROM cayu_eval_suites AS suites
    JOIN cayu_eval_corpora AS corpora ON corpora.revision = suites.corpus_revision
    ON CONFLICT (corpus_revision, suite_id) DO NOTHING
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_eval_trial_checkpoint_summaries (
        run_id TEXT COLLATE "C" NOT NULL,
        case_id TEXT COLLATE "C" NOT NULL,
        trial_number INTEGER NOT NULL,
        -- Copied from the document as is: the trigger never refuses a checkpoint
        -- the checkpoint table accepts, and readers validate what they read.
        status TEXT,
        score DOUBLE PRECISION,
        diagnostic_code TEXT,
        document_bytes BIGINT NOT NULL,
        PRIMARY KEY (run_id, case_id, trial_number),
        FOREIGN KEY (run_id, case_id, trial_number)
            REFERENCES cayu_eval_run_trial_checkpoints(run_id, case_id, trial_number)
            ON DELETE CASCADE
    )
    """,
    f"""
    CREATE OR REPLACE FUNCTION cayu_track_eval_trial_checkpoint_summary()
    RETURNS TRIGGER
    SET search_path FROM CURRENT
    AS $$
    BEGIN
        INSERT INTO cayu_eval_trial_checkpoint_summaries ({_CHECKPOINT_SUMMARY_COLUMNS})
        {_postgres_checkpoint_select("NEW")}
        ON CONFLICT (run_id, case_id, trial_number) DO UPDATE SET
            status = EXCLUDED.status,
            score = EXCLUDED.score,
            diagnostic_code = EXCLUDED.diagnostic_code,
            document_bytes = EXCLUDED.document_bytes;
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql
    """,
    "DROP TRIGGER IF EXISTS cayu_eval_trial_checkpoint_summaries_write "
    "ON cayu_eval_run_trial_checkpoints",
    """
    CREATE TRIGGER cayu_eval_trial_checkpoint_summaries_write
    AFTER INSERT OR UPDATE OF checkpoint_json, document_bytes
    ON cayu_eval_run_trial_checkpoints
    FOR EACH ROW EXECUTE FUNCTION cayu_track_eval_trial_checkpoint_summary()
    """,
    f"""
    INSERT INTO cayu_eval_trial_checkpoint_summaries ({_CHECKPOINT_SUMMARY_COLUMNS})
    SELECT summary.* FROM cayu_eval_run_trial_checkpoints AS checkpoints
    CROSS JOIN LATERAL ({_postgres_checkpoint_select("checkpoints")}) AS summary
    ON CONFLICT (run_id, case_id, trial_number) DO NOTHING
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_eval_result_summaries (
        revision TEXT COLLATE "C" PRIMARY KEY
            REFERENCES cayu_eval_result_records(revision) ON DELETE CASCADE,
        summary_json JSONB NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_eval_case_outcomes (
        result_revision TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_eval_result_summaries(revision) ON DELETE CASCADE,
        case_id TEXT COLLATE "C" NOT NULL,
        status TEXT NOT NULL,
        score DOUBLE PRECISION,
        passed_trials INTEGER NOT NULL,
        total_trials INTEGER NOT NULL,
        duration_ms BIGINT,
        outcome_json JSONB NOT NULL,
        evidence_json TEXT,
        PRIMARY KEY (result_revision, case_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cayu_eval_result_comparisons (
        current_revision TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_eval_result_records(revision) ON DELETE CASCADE,
        previous_revision TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_eval_result_records(revision) ON DELETE CASCADE,
        comparison_json JSONB NOT NULL,
        PRIMARY KEY (current_revision, previous_revision)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cayu_eval_result_comparisons_previous "
    "ON cayu_eval_result_comparisons(previous_revision)",
    "CREATE INDEX IF NOT EXISTS idx_cayu_eval_runs_result_revision "
    "ON cayu_eval_runs(result_revision)",
)

__all__ = [
    "EVAL_VIEW_SUMMARIES_REVISION",
    "POSTGRES_EVAL_VIEW_SUMMARIES_DDL",
    "SQLITE_EVAL_VIEW_SUMMARIES_DDL",
]
