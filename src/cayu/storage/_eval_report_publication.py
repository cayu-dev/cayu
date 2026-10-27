"""Report persistence inside the existing SQLite/PostgreSQL mutation owners."""

from __future__ import annotations

from cayu.evals._report_publication import (
    replay_report,
    report_association_key,
    report_association_selector,
    validate_report_association,
    validate_report_record,
)
from cayu.evals.corpus import EvalCorpusDocument
from cayu.evals.reported import EvalReportRecord
from cayu.storage import _sqlite_connection as sqlite_connection


def report_values(record: EvalReportRecord, *, postgres: bool):
    timestamp = record.receipt.accepted_at
    association = report_association_selector(record)
    return (
        record.receipt.id,
        record.receipt.publisher_id,
        record.receipt.report_id,
        record.series.selector.series_id,
        record.report.target_key,
        None if association is None else report_association_key(association),
        record.model_dump_json(),
        timestamp if postgres else timestamp.isoformat(),
    )


def load_report_document(document: str | None) -> EvalReportRecord | None:
    if document is None:
        return None
    return validate_report_record(EvalReportRecord.model_validate_json(document))


def publish_sqlite_report(connection, record: EvalReportRecord) -> EvalReportRecord:
    with sqlite_connection._transaction(connection):
        row = connection.execute(
            "SELECT record_json FROM cayu_eval_reports WHERE report_key = ?",
            (record.receipt.id,),
        ).fetchone()
        if row is not None:
            return replay_report(EvalReportRecord.model_validate_json(row[0]), record)
        association = record.report.association
        corpus = None
        if association is not None:
            row = connection.execute(
                "SELECT document_json FROM cayu_eval_corpora WHERE revision = ?",
                (association.corpus_revision,),
            ).fetchone()
            corpus = None if row is None else EvalCorpusDocument.model_validate_json(row[0])
        validate_report_association(record, corpus)
        connection.execute(
            "INSERT INTO cayu_eval_report_series (series_id, target_key, selector_json, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (series_id) DO NOTHING",
            (
                record.series.selector.series_id,
                record.report.target_key,
                record.series.id,
                record.receipt.accepted_at.isoformat(),
            ),
        )
        connection.execute(
            "INSERT INTO cayu_eval_reports "
            "(report_key, publisher_id, operation_id, series_id, target_key, association_key, "
            "record_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            report_values(record, postgres=False),
        )
        return record


async def publish_postgres_report(cursor, record: EvalReportRecord) -> EvalReportRecord:
    # Insert fences the stable publisher/operation identity before any secondary write.
    await cursor.execute(
        "INSERT INTO cayu_eval_reports "
        "(report_key, publisher_id, operation_id, series_id, target_key, association_key, "
        "record_json, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (report_key) DO NOTHING RETURNING report_key",
        report_values(record, postgres=True),
    )
    if await cursor.fetchone() is None:
        await cursor.execute(
            "SELECT record_json FROM cayu_eval_reports WHERE report_key = %s",
            (record.receipt.id,),
        )
        existing = await cursor.fetchone()
        if existing is None:
            raise RuntimeError("Immutable report disappeared during publication.")
        return replay_report(EvalReportRecord.model_validate_json(existing[0]), record)
    corpus = None
    association = record.report.association
    if association is not None:
        await cursor.execute(
            "SELECT document FROM cayu_eval_corpora WHERE revision = %s FOR SHARE",
            (association.corpus_revision,),
        )
        row = await cursor.fetchone()
        corpus = None if row is None else EvalCorpusDocument.model_validate_json(row[0])
    validate_report_association(record, corpus)
    await cursor.execute(
        "INSERT INTO cayu_eval_report_series (series_id, target_key, selector_json, created_at) "
        "VALUES (%s, %s, %s, %s) ON CONFLICT (series_id) DO NOTHING",
        (
            record.series.selector.series_id,
            record.report.target_key,
            record.series.id,
            record.receipt.accepted_at,
        ),
    )
    return record
