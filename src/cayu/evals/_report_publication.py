"""One canonical immutable report operation shared by all publication adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from cayu._validation import require_durable_clean_nonblank
from cayu.evals.corpus import EvalCorpusDocument
from cayu.evals.published import _validate_published_eval_run_for_corpus
from cayu.evals.read_views import EvalSuiteViewIdentity, run_view_id
from cayu.evals.reported import (
    EVAL_REPORT_MAX_BYTES,
    EvalCorpusReportEvidence,
    EvalReport,
    EvalReportConflict,
    EvalReportReceipt,
    EvalReportRecord,
    report_canonical_bytes,
    report_fingerprint,
    report_operation_id,
)
from cayu.evals.store import EvalStoreResultTooLarge, _exact_model, _require_publication_safe


def prepare_report_record(
    report: EvalReport,
    publisher_id: str,
    *,
    redact_json: Callable[[Any], Any],
) -> EvalReportRecord:
    if type(publisher_id) is not str or len(publisher_id) > 256:
        raise ValueError("Invalid report publisher identity.")
    require_durable_clean_nonblank(publisher_id, "publisher_id")
    report = _exact_model(report, EvalReport, "report")
    document = report.model_dump(mode="json")
    _require_publication_safe(
        {"report": document, "publisher_id": publisher_id},
        redact_json=redact_json,
        resource_name="Eval report",
    )
    if len(report_canonical_bytes(report)) > EVAL_REPORT_MAX_BYTES:
        raise EvalStoreResultTooLarge(EVAL_REPORT_MAX_BYTES)
    identity = report_operation_id(publisher_id, report.report_id)
    return EvalReportRecord(
        report=report,
        receipt=EvalReportReceipt(
            id=identity,
            report_id=report.report_id,
            publisher_id=publisher_id,
            target_key=report.target_key,
            fingerprint=report_fingerprint(report),
            accepted_at=datetime.now(UTC),
            run_key=run_view_id("report", identity),
        ),
    )


def validate_report_record(record: EvalReportRecord) -> EvalReportRecord:
    record = _exact_model(record, EvalReportRecord, "report record")
    expected = prepare_report_record(
        record.report, record.receipt.publisher_id, redact_json=lambda value: value
    ).receipt
    if record.receipt.model_dump(exclude={"accepted_at"}) != expected.model_dump(
        exclude={"accepted_at"}
    ):
        raise EvalReportConflict("Stored report receipt does not match its immutable evidence.")
    if record.receipt.accepted_at.utcoffset() is None:
        raise EvalReportConflict("Stored report receipt timestamp requires a timezone.")
    return record


def replay_report(existing: EvalReportRecord, expected: EvalReportRecord) -> EvalReportRecord:
    existing = validate_report_record(existing)
    if (
        existing.receipt.id != expected.receipt.id
        or existing.receipt.publisher_id != expected.receipt.publisher_id
        or report_canonical_bytes(existing.report) != report_canonical_bytes(expected.report)
    ):
        raise EvalReportConflict("Report operation already names different immutable evidence.")
    return existing


def report_association_selector(record: EvalReportRecord) -> str | None:
    association = record.report.association
    if association is None:
        return None
    return EvalSuiteViewIdentity(
        source="corpus",
        target_key=record.report.target_key,
        revision=association.corpus_revision,
        suite_id=association.suite_id,
    ).id


def validate_report_association(
    record: EvalReportRecord, corpus: EvalCorpusDocument | None
) -> None:
    association = record.report.association
    if association is None:
        return
    if corpus is None or type(record.report.evidence) is not EvalCorpusReportEvidence:
        raise EvalReportConflict("Requested native corpus association is unavailable.")
    corpus = _exact_model(corpus, EvalCorpusDocument, "report corpus association")
    if (
        corpus.target_key != record.report.target_key
        or corpus.revision != association.corpus_revision
    ):
        raise EvalReportConflict("Requested native corpus association disagrees with stored data.")
    try:
        _validate_published_eval_run_for_corpus(corpus, record.report.evidence.result.run)
    except (TypeError, ValueError):
        raise EvalReportConflict(
            "Report does not match the exact stored corpus contract."
        ) from None


def report_association_key(selector: str) -> str:
    """A bounded secondary index key, never a native suite revision."""
    return hashlib.sha256(selector.encode("utf-8")).hexdigest()


def validate_report_key(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("Invalid eval report identity.")
    return value
