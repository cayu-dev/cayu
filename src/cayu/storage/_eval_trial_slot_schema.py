"""Revision-138 schema for eval run trials that several workers claim."""

from __future__ import annotations

EVAL_RUN_TRIAL_SLOTS_TABLE = "cayu_eval_run_trial_slots"
EVAL_RUN_TRIAL_SLOTS_REVISION = 138

# One row per trial of a run that its owner opened to other workers, in the
# order trials are handed out. ``run_epoch`` is the run ownership epoch that
# opened the slot: a later owner reopens the slots under its own epoch, which
# fences every trial claim taken under the earlier one. A slot is claimed while
# ``claim_id`` is set and its lease has not expired; its terminal result is the
# run's ordinary trial checkpoint, after which the claim is cleared.
SQLITE_EVAL_RUN_TRIAL_SLOT_DDL = f"""
    CREATE TABLE IF NOT EXISTS {EVAL_RUN_TRIAL_SLOTS_TABLE} (
        run_id TEXT COLLATE BINARY NOT NULL
            REFERENCES cayu_eval_runs(run_id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        case_id TEXT COLLATE BINARY NOT NULL,
        trial_number INTEGER NOT NULL CHECK (trial_number BETWEEN 1 AND 100),
        run_epoch INTEGER NOT NULL CHECK (run_epoch >= 1),
        claim_id TEXT COLLATE BINARY,
        claim_epoch INTEGER NOT NULL DEFAULT 0 CHECK (claim_epoch >= 0),
        lease_expires_at TEXT,
        PRIMARY KEY (run_id, position),
        UNIQUE (run_id, case_id, trial_number),
        CHECK ((claim_id IS NULL) = (lease_expires_at IS NULL))
    );
"""

POSTGRES_EVAL_RUN_TRIAL_SLOT_DDL: tuple[str, ...] = (
    f"""CREATE TABLE IF NOT EXISTS {EVAL_RUN_TRIAL_SLOTS_TABLE} (
        run_id TEXT COLLATE "C" NOT NULL
            REFERENCES cayu_eval_runs(run_id) ON DELETE CASCADE,
        position INTEGER NOT NULL CHECK (position >= 0),
        case_id TEXT COLLATE "C" NOT NULL,
        trial_number INTEGER NOT NULL CHECK (trial_number BETWEEN 1 AND 100),
        run_epoch BIGINT NOT NULL CHECK (run_epoch >= 1),
        claim_id TEXT COLLATE "C",
        claim_epoch BIGINT NOT NULL DEFAULT 0 CHECK (claim_epoch >= 0),
        lease_expires_at TIMESTAMPTZ,
        PRIMARY KEY (run_id, position),
        UNIQUE (run_id, case_id, trial_number),
        CHECK ((claim_id IS NULL) = (lease_expires_at IS NULL))
    )""",
)


__all__ = [
    "EVAL_RUN_TRIAL_SLOTS_REVISION",
    "EVAL_RUN_TRIAL_SLOTS_TABLE",
    "POSTGRES_EVAL_RUN_TRIAL_SLOT_DDL",
    "SQLITE_EVAL_RUN_TRIAL_SLOT_DDL",
]
