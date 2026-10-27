"""Read-only external evidence; selectors never grant native result authority."""

from __future__ import annotations

import hashlib
import json

from fastapi import HTTPException

from cayu.evals.read_views import (
    EvalReportedCaseView,
    EvalReportedRunDetail,
    EvalReportedRunView,
    EvalReportedSuiteDetail,
    EvalReportedSuiteView,
    EvalSuiteViewIdentity,
    EvalViewCaseCounts,
    EvalViewComparison,
    EvalViewPage,
    EvalViewQuery,
    decode_run_view,
    decode_view_token,
    encode_view_token,
    validate_view_index,
)
from cayu.evals.reported import (
    EvalDirectReportEvidence,
    EvalReportedSeriesSelector,
    EvalReportRecord,
)


def _reported_case_page(items, *, scope, cursor, limit):
    # Report records are immutable. A content/scope-bound position keeps the
    # continuation bounded independently of the admitted Unicode case identities.
    identity = hashlib.sha256(json.dumps(scope, separators=(",", ":")).encode()).hexdigest()
    offset = 0
    ordered = sorted(items, key=lambda item: item.id)
    if cursor is not None:
        value = decode_view_token(cursor)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or value[:2] != ["reported-cases", identity]
            or type(value[2]) is not int
            or not 0 < value[2] < len(ordered)
        ):
            raise ValueError("Reported case cursor scope mismatch.")
        offset = value[2]
    selected = ordered[offset : offset + limit]
    end = offset + len(selected)
    more = end < len(ordered)
    return EvalViewPage(
        items=tuple(selected),
        total=len(ordered),
        has_more=more,
        next_cursor=encode_view_token(["reported-cases", identity, end]) if more else None,
    )


def reported_case_view(case) -> EvalReportedCaseView:
    # A case is indivisible in cursor pagination. Keep its authoritative summary
    # readable even when full trial evidence alone exceeds the response ceiling.
    omitted = len(case.model_dump_json().encode("utf-8")) > 64 << 10
    return EvalReportedCaseView(
        id=case.case_id,
        status=case.status,
        score=case.score,
        trial_count=len(case.trials),
        evidence=None if omitted else case,
        evidence_omitted=omitted,
    )


def report_comparison(compared_with=None):
    if compared_with is not None:
        decode_run_view(compared_with)
    return EvalViewComparison(
        state="unavailable",
        compared_with=compared_with,
        unavailable_reasons=("external_report_comparison_unavailable",),
    )


class EvalReportViewReader:
    def __init__(self, store, registry):
        self.store = store
        self.registry = registry

    def target(self, key):
        target = self.registry.get(key)
        if target is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        return target

    async def record(self, selector: str) -> EvalReportRecord:
        source, key = decode_run_view(selector)
        if source != "report" or not self.store.reports:
            raise HTTPException(status_code=404, detail="Eval report not found.")
        record = await self.store.load_report(key)
        if record is None:
            raise HTTPException(status_code=404, detail="Eval report not found.")
        if record.receipt.id != key or record.receipt.run_key != selector:
            raise HTTPException(status_code=409, detail="Eval report identity mismatch.")
        self.target(record.report.target_key)
        return record

    def summary(self, record):
        report = record.report
        direct = type(report.evidence) is EvalDirectReportEvidence
        run = report.evidence.run if direct else report.evidence.result.run
        statuses = [case.status for case in run.cases]
        counts = EvalViewCaseCounts(
            total=len(statuses),
            passed=statuses.count("passed"),
            failed=statuses.count("failed"),
            error=statuses.count("error"),
            unavailable=statuses.count("unavailable") + statuses.count("skipped"),
            flaky=sum(
                any(trial.status == "passed" for trial in case.trials)
                and any(trial.status != "passed" for trial in case.trials)
                for case in run.cases
            ),
        )
        association = (
            None
            if report.association is None
            else EvalSuiteViewIdentity(
                source="corpus",
                target_key=report.target_key,
                revision=report.association.corpus_revision,
                suite_id=report.association.suite_id,
            )
        )
        return EvalReportedRunView(
            id=record.receipt.run_key,
            report_id=report.report_id,
            publisher_id=record.receipt.publisher_id,
            suite=record.series,
            specialist=self.target(report.target_key).request_base.agent_name,
            declared_suite_revision=report.declared_suite_revision,
            origin=report.origin,
            status=run.status,
            score=run.score,
            created_at=record.receipt.accepted_at,
            started_at=run.started_at if direct else None,
            finished_at=run.completed_at if direct else None,
            cases=counts,
            comparison=report_comparison(),
            association=association,
            download_url=f"evals/reports/{record.receipt.run_key}/download",
        )

    async def run_detail(
        self, selector, *, compared_with=None, case_filter=None, cases_cursor=None, cases_limit=50
    ):
        record = await self.record(selector)
        evidence = record.report.evidence
        run = (
            evidence.run if isinstance(evidence, EvalDirectReportEvidence) else evidence.result.run
        )
        selected = list(run.cases)
        if case_filter in {"regressed", "improved"}:
            selected = []
        elif case_filter == "failing":
            selected = [
                case for case in selected if case.status in {"failed", "error", "unavailable"}
            ]
        elif case_filter == "flaky":
            selected = [
                case
                for case in selected
                if any(trial.status == "passed" for trial in case.trials)
                and any(trial.status != "passed" for trial in case.trials)
            ]
        comparison = report_comparison(compared_with)
        return EvalReportedRunDetail(
            run=self.summary(record).model_copy(update={"comparison": comparison}),
            comparison=comparison,
            results=_reported_case_page(
                [reported_case_view(case) for case in selected],
                scope=[selector, record.receipt.fingerprint, compared_with, case_filter],
                cursor=cases_cursor,
                limit=cases_limit,
            ),
        )

    async def history(self, identity, *, cursor=None, limit=20):
        query = EvalViewQuery(
            target_keys=tuple(sorted(self.registry.target_keys)),
            suite=identity,
            cursor=cursor,
            limit=limit,
        )
        page = validate_view_index(await self.store.list_view_index(query), query)
        runs = []
        for item in page.items:
            if item.source != "report":
                raise HTTPException(status_code=409, detail="Invalid reported history entry.")
            from cayu.evals.read_views import run_view_id

            record = await self.record(run_view_id("report", item.revision))
            if record.series.selector != identity:
                raise HTTPException(status_code=409, detail="Report series identity mismatch.")
            runs.append(self.summary(record))
        return EvalViewPage(items=tuple(runs), next_cursor=page.next_cursor, has_more=page.has_more)

    async def suite_summary(self, identity: EvalReportedSeriesSelector):
        page = await self.history(identity, limit=1)
        if not page.items:
            raise HTTPException(status_code=404, detail="Reported suite not found.")
        last = page.items[0]
        return EvalReportedSuiteView(
            id=identity.id,
            identity=last.suite,
            name=last.suite.suite_id,
            specialist=last.specialist,
            case_count=last.cases.total,
            last_run=last,
        )

    async def suite_detail(self, identity, *, runs_cursor=None, runs_limit=20, **unused):
        return EvalReportedSuiteDetail(
            suite=await self.suite_summary(identity),
            runs=await self.history(identity, cursor=runs_cursor, limit=runs_limit),
        )
