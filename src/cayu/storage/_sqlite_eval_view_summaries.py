"""SQLite statements for the revision-153 Evals summary tables."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

from cayu.evals.read_views import EvalTrialViewOutcome
from cayu.evals.view_summaries import (
    EvalCaseOutcomePage,
    EvalCaseOutcomeQuery,
    EvalCaseOutcomeV1,
    EvalResultViewRows,
    EvalResultViewSummaryV1,
    EvalSuiteCatalogEntryV1,
    EvalViewComparisonRecordV1,
)


def write_view_rows(
    connection: sqlite3.Connection, revision: str, rows: EvalResultViewRows
) -> bool:
    """Insert a result's summary and case outcomes unless it already has them."""
    if rows.summary.revision != revision:
        raise ValueError("Eval result summary belongs to another result.")
    inserted = connection.execute(
        "INSERT OR IGNORE INTO cayu_eval_result_summaries (revision, summary_json) VALUES (?, ?)",
        (revision, rows.summary_json),
    ).rowcount
    if not inserted:
        return False
    connection.executemany(
        """INSERT INTO cayu_eval_case_outcomes (
            result_revision, case_id, status, score, passed_trials, total_trials,
            duration_ms, outcome_json, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [(revision, *row) for row in rows.outcome_rows()],
    )
    return True


def load_summaries(
    connection: sqlite3.Connection, revisions: Iterable[str]
) -> dict[str, EvalResultViewSummaryV1]:
    wanted = tuple(dict.fromkeys(revisions))
    if not wanted:
        return {}
    marks = ",".join("?" for _ in wanted)
    rows = connection.execute(
        f"SELECT revision, summary_json FROM cayu_eval_result_summaries WHERE revision IN ({marks})",
        wanted,
    ).fetchall()
    return {str(row[0]): EvalResultViewSummaryV1.model_validate_json(row[1]) for row in rows}


def list_case_outcomes(
    connection: sqlite3.Connection, query: EvalCaseOutcomeQuery
) -> EvalCaseOutcomePage | None:
    """A keyset page of outcomes, or None when the result has no summary rows."""
    if (
        connection.execute(
            "SELECT 1 FROM cayu_eval_result_summaries WHERE revision = ?", (query.revision,)
        ).fetchone()
        is None
    ):
        return None
    clauses = ["result_revision = ?"]
    params: list[object] = [query.revision]
    if query.case_ids is not None:
        if not query.case_ids:
            return EvalCaseOutcomePage(items=(), total=0, has_more=False)
        clauses.append(f"case_id IN ({','.join('?' for _ in query.case_ids)})")
        params.extend(query.case_ids)
    if query.case_filter == "failing":
        clauses.append("status IN ('failed', 'error', 'unavailable')")
    elif query.case_filter == "flaky":
        clauses.append("passed_trials > 0 AND passed_trials < total_trials")
    where = " AND ".join(clauses)
    total = connection.execute(
        f"SELECT COUNT(*) FROM cayu_eval_case_outcomes WHERE {where}", params
    ).fetchone()[0]
    page_where, page_params = where, list(params)
    if query.after is not None:
        page_where += " AND case_id > ?"
        page_params.append(query.after)
    evidence = "evidence_json" if query.include_evidence else "NULL"
    sql = (
        f"SELECT outcome_json, {evidence} FROM cayu_eval_case_outcomes "
        f"WHERE {page_where} ORDER BY case_id ASC"
    )
    if query.limit is not None:
        sql += " LIMIT ?"
        page_params.append(query.limit + 1)
    rows = connection.execute(sql, page_params).fetchall()
    more = query.limit is not None and len(rows) > query.limit
    items = []
    for outcome_json, evidence_json in rows[: query.limit]:
        outcome = EvalCaseOutcomeV1.model_validate_json(outcome_json)
        if evidence_json is not None:
            outcome = outcome.model_copy(update={"evidence": json.loads(evidence_json)})
        items.append(outcome)
    return EvalCaseOutcomePage(items=tuple(items), total=int(total), has_more=more)


def load_comparison(
    connection: sqlite3.Connection, current: str, previous: str
) -> EvalViewComparisonRecordV1 | None:
    row = connection.execute(
        "SELECT comparison_json FROM cayu_eval_result_comparisons "
        "WHERE current_revision = ? AND previous_revision = ?",
        (current, previous),
    ).fetchone()
    return None if row is None else EvalViewComparisonRecordV1.model_validate_json(row[0])


def save_comparison(
    connection: sqlite3.Connection,
    current: str,
    previous: str,
    record: EvalViewComparisonRecordV1,
) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO cayu_eval_result_comparisons
        (current_revision, previous_revision, comparison_json)
        SELECT ?, ?, ? WHERE EXISTS (
            SELECT 1 FROM cayu_eval_result_records WHERE revision = ?
        ) AND EXISTS (SELECT 1 FROM cayu_eval_result_records WHERE revision = ?)""",
        (current, previous, record.model_dump_json(), current, previous),
    )


def load_catalog(
    connection: sqlite3.Connection, keys: Iterable[tuple[str, str]]
) -> dict[tuple[str, str], EvalSuiteCatalogEntryV1]:
    wanted = tuple(dict.fromkeys(keys))
    entries: dict[tuple[str, str], EvalSuiteCatalogEntryV1] = {}
    for corpus_revision, suite_id in wanted:
        row = connection.execute(
            """SELECT target_key, suite_revision, name, description, case_count,
                      suite_json, cases_json
            FROM cayu_eval_suite_catalog WHERE corpus_revision = ? AND suite_id = ?""",
            (corpus_revision, suite_id),
        ).fetchone()
        if row is None:
            continue
        entries[(corpus_revision, suite_id)] = EvalSuiteCatalogEntryV1(
            target_key=row[0],
            corpus_revision=corpus_revision,
            suite_id=suite_id,
            suite_revision=row[1],
            name=row[2],
            description=row[3],
            case_count=row[4],
            suite=json.loads(row[5]),
            cases=tuple(json.loads(row[6])),
        )
    return entries


def load_trial_outcomes(
    connection: sqlite3.Connection, run_id: str
) -> tuple[EvalTrialViewOutcome, ...]:
    rows = connection.execute(
        """SELECT case_id, trial_number, status, score, diagnostic_code
        FROM cayu_eval_trial_checkpoint_summaries WHERE run_id = ?
        ORDER BY case_id ASC, trial_number ASC""",
        (run_id,),
    ).fetchall()
    return tuple(
        EvalTrialViewOutcome(
            case_id=row[0],
            trial_number=row[1],
            status=row[2],
            score=row[3],
            diagnostic_code=row[4],
        )
        for row in rows
    )


__all__ = [
    "list_case_outcomes",
    "load_catalog",
    "load_comparison",
    "load_summaries",
    "load_trial_outcomes",
    "save_comparison",
    "write_view_rows",
]
