"""Postgres statements for the revision-153 Evals summary tables."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

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


def _json(value: object) -> Any:
    """A JSONB column comes back decoded; a text column comes back as text."""
    return json.loads(value) if isinstance(value, str) else value


async def write_view_rows(cur: Any, revision: str, rows: EvalResultViewRows) -> bool:
    """Insert a result's summary and case outcomes unless it already has them."""
    if rows.summary.revision != revision:
        raise ValueError("Eval result summary belongs to another result.")
    await cur.execute(
        "INSERT INTO cayu_eval_result_summaries (revision, summary_json) VALUES (%s, %s::jsonb) "
        "ON CONFLICT (revision) DO NOTHING RETURNING revision",
        (revision, rows.summary_json),
    )
    if await cur.fetchone() is None:
        return False
    await cur.executemany(
        """INSERT INTO cayu_eval_case_outcomes (
            result_revision, case_id, status, score, passed_trials, total_trials,
            duration_ms, outcome_json, evidence_json
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)""",
        [(revision, *row) for row in rows.outcome_rows()],
    )
    return True


async def load_summaries(cur: Any, revisions: Iterable[str]) -> dict[str, EvalResultViewSummaryV1]:
    wanted = list(dict.fromkeys(revisions))
    if not wanted:
        return {}
    await cur.execute(
        "SELECT revision, summary_json::text FROM cayu_eval_result_summaries "
        "WHERE revision = ANY(%s)",
        (wanted,),
    )
    return {
        str(row[0]): EvalResultViewSummaryV1.model_validate_json(row[1])
        for row in await cur.fetchall()
    }


async def list_case_outcomes(cur: Any, query: EvalCaseOutcomeQuery) -> EvalCaseOutcomePage | None:
    await cur.execute(
        "SELECT 1 FROM cayu_eval_result_summaries WHERE revision = %s", (query.revision,)
    )
    if await cur.fetchone() is None:
        return None
    clauses = ["result_revision = %s"]
    params: list[object] = [query.revision]
    if query.case_ids is not None:
        if not query.case_ids:
            return EvalCaseOutcomePage(items=(), total=0, has_more=False)
        clauses.append("case_id = ANY(%s)")
        params.append(list(query.case_ids))
    if query.case_filter == "failing":
        clauses.append("status IN ('failed', 'error', 'unavailable')")
    elif query.case_filter == "flaky":
        clauses.append("passed_trials > 0 AND passed_trials < total_trials")
    where = " AND ".join(clauses)
    await cur.execute(f"SELECT COUNT(*) FROM cayu_eval_case_outcomes WHERE {where}", params)
    total = (await cur.fetchone())[0]
    page_where, page_params = where, list(params)
    if query.after is not None:
        page_where += ' AND case_id > %s COLLATE "C"'
        page_params.append(query.after)
    evidence = "evidence_json" if query.include_evidence else "NULL"
    sql = (
        f"SELECT outcome_json::text, {evidence} FROM cayu_eval_case_outcomes "
        f'WHERE {page_where} ORDER BY case_id COLLATE "C" ASC'
    )
    if query.limit is not None:
        sql += " LIMIT %s"
        page_params.append(query.limit + 1)
    await cur.execute(sql, page_params)
    rows = await cur.fetchall()
    more = query.limit is not None and len(rows) > query.limit
    items = []
    for outcome_json, evidence_json in rows[: query.limit]:
        outcome = EvalCaseOutcomeV1.model_validate_json(outcome_json)
        if evidence_json is not None:
            outcome = outcome.model_copy(update={"evidence": json.loads(evidence_json)})
        items.append(outcome)
    return EvalCaseOutcomePage(items=tuple(items), total=int(total), has_more=more)


async def load_comparison(
    cur: Any, current: str, previous: str
) -> EvalViewComparisonRecordV1 | None:
    await cur.execute(
        "SELECT comparison_json::text FROM cayu_eval_result_comparisons "
        "WHERE current_revision = %s AND previous_revision = %s",
        (current, previous),
    )
    row = await cur.fetchone()
    return None if row is None else EvalViewComparisonRecordV1.model_validate_json(row[0])


async def save_comparison(
    cur: Any, current: str, previous: str, record: EvalViewComparisonRecordV1
) -> None:
    await cur.execute(
        """INSERT INTO cayu_eval_result_comparisons
        (current_revision, previous_revision, comparison_json)
        SELECT %s, %s, %s::jsonb WHERE EXISTS (
            SELECT 1 FROM cayu_eval_result_records WHERE revision = %s
        ) AND EXISTS (SELECT 1 FROM cayu_eval_result_records WHERE revision = %s)
        ON CONFLICT (current_revision, previous_revision) DO NOTHING""",
        (current, previous, record.model_dump_json(), current, previous),
    )


async def load_catalog(
    cur: Any, keys: Iterable[tuple[str, str]]
) -> dict[tuple[str, str], EvalSuiteCatalogEntryV1]:
    entries: dict[tuple[str, str], EvalSuiteCatalogEntryV1] = {}
    for corpus_revision, suite_id in dict.fromkeys(keys):
        await cur.execute(
            """SELECT target_key, suite_revision, name, description, case_count,
                      suite_json, cases_json
            FROM cayu_eval_suite_catalog WHERE corpus_revision = %s AND suite_id = %s""",
            (corpus_revision, suite_id),
        )
        row = await cur.fetchone()
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
            suite=_json(row[5]),
            cases=tuple(_json(row[6])),
        )
    return entries


async def load_trial_outcomes(cur: Any, run_id: str) -> tuple[EvalTrialViewOutcome, ...]:
    await cur.execute(
        """SELECT case_id, trial_number, status, score, diagnostic_code
        FROM cayu_eval_trial_checkpoint_summaries WHERE run_id = %s
        ORDER BY case_id COLLATE "C" ASC, trial_number ASC""",
        (run_id,),
    )
    return tuple(
        EvalTrialViewOutcome(
            case_id=row[0],
            trial_number=row[1],
            status=row[2],
            score=row[3],
            diagnostic_code=row[4],
        )
        for row in await cur.fetchall()
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
