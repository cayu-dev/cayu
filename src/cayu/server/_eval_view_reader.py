"""Read orchestration for suite views; no worker or baseline mutation authority."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel

from cayu.evals.corpus import EvalCorpusDocument, eval_suite_trial_policy
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.published import _validate_published_eval_run_for_corpus
from cayu.evals.read_views import (
    EvalCaseHistoryQuery,
    EvalCaseResultReference,
    EvalCaseView,
    EvalReportedRunView,
    EvalRunCaseFilter,
    EvalRunPlacement,
    EvalRunViewDetail,
    EvalRunViewMetadata,
    EvalRunViewSnapshot,
    EvalRunViewSummary,
    EvalSuiteViewDetail,
    EvalSuiteViewIdentity,
    EvalSuiteViewSummary,
    EvalViewIndexPage,
    EvalViewPage,
    EvalViewQuery,
    decode_run_view,
    decode_suite_view,
    run_view_id,
    validate_view_index,
)
from cayu.evals.reported import EvalReportedSeriesSelector
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.results import CapturedEvaluationResultV1, validate_captured_result_for_corpus
from cayu.evals.store import (
    EvalBaselineKey,
    EvalBaselineRecord,
    EvalResultRecord,
    EvalStoreResultTooLarge,
    _exact_model,
)
from cayu.evals.suite_authoring import _validated_eval_suite_document
from cayu.server._eval_report_views import EvalReportViewReader, report_comparison
from cayu.server._eval_result_loader import load_catalog_eval_result
from cayu.server._eval_view_comparison import EvalViewComparisonData, compare_view_results
from cayu.server._eval_view_projection import (
    EvalRunViewData,
    case_page,
    case_results,
    case_view,
    run_metadata,
    run_progress,
)

if TYPE_CHECKING:
    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


ViewModel = TypeVar("ViewModel", bound=BaseModel)


def enforce_view_bytes(value: ViewModel, maximum: int) -> ViewModel:
    if len(value.model_dump_json().encode()) > maximum:
        raise EvalStoreResultTooLarge(maximum)
    return value


class EvalViewReader:
    def __init__(self, store: EvalStore, registry: EvalTargetRegistry):
        self.store = store
        self.registry = registry
        self.reports = EvalReportViewReader(store, registry)
        self._baseline_snapshots: dict[tuple[str, str, str], EvalBaselineRecord | None] = {}
        self._baseline_results: dict[str, EvalRunViewMetadata | None] = {}
        self._detail_snapshots: dict[
            tuple[str, str | None, str | None, int, EvalRunCaseFilter | None], EvalRunViewDetail
        ] = {}
        self._corpus_documents: dict[tuple[str, str], EvalCorpusDocument] = {}

    async def index(self, query: EvalViewQuery) -> EvalViewIndexPage:
        return validate_view_index(await self.store.list_view_index(query), query)

    def target(self, key):
        target = self.registry.get(key)
        if target is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        return target

    async def suite_document(self, identity):
        if not isinstance(identity, EvalSuiteViewIdentity):
            raise HTTPException(
                status_code=422, detail="Reported series has no native suite document."
            )
        self.target(identity.target_key)
        if identity.source == "authored":
            document = await self.store.load_authored_suite(identity.revision)
            if document is None:
                raise HTTPException(status_code=404, detail="Eval suite not found.")
            document = await asyncio.to_thread(_validated_eval_suite_document, document)
            if document.revision != identity.revision or document.target_key != identity.target_key:
                raise HTTPException(status_code=409, detail="Eval suite identity mismatch.")
            return document.suite, document.cases
        document = await self.corpus_document(identity)
        suite = next((item for item in document.suites if item.id == identity.suite_id), None)
        if suite is None:
            raise HTTPException(status_code=404, detail="Eval suite not found.")
        return suite, tuple(case for case in document.cases if case.suite_id == identity.suite_id)

    async def corpus_document(self, identity: EvalSuiteViewIdentity) -> EvalCorpusDocument:
        self.target(identity.target_key)
        key = (identity.target_key, identity.revision)
        if key in self._corpus_documents:
            return self._corpus_documents[key]
        document = await self.store.load_corpus(identity.revision)
        if document is None:
            raise HTTPException(status_code=404, detail="Eval suite not found.")
        document = await asyncio.to_thread(_exact_model, document, EvalCorpusDocument, "corpus")
        if document.revision != identity.revision or document.target_key != identity.target_key:
            raise HTTPException(status_code=409, detail="Eval suite identity mismatch.")
        if len(self._corpus_documents) == 2:
            self._corpus_documents.pop(next(iter(self._corpus_documents)))
        self._corpus_documents[key] = document
        return document

    async def baseline(self, identity: EvalSuiteViewIdentity) -> EvalBaselineRecord | None:
        if identity.source == "authored":
            return None
        assert identity.suite_id is not None
        key = EvalBaselineKey(
            target_key=identity.target_key,
            corpus_revision=identity.revision,
            suite_id=identity.suite_id,
        )
        cache_key = (key.target_key, key.corpus_revision, key.suite_id)
        if cache_key in self._baseline_snapshots:
            return self._baseline_snapshots[cache_key]
        baseline = await self.store.load_baseline(key)
        if baseline is not None and baseline.key != key:
            raise HTTPException(status_code=409, detail="Eval baseline identity mismatch.")
        self._baseline_snapshots[cache_key] = baseline
        return baseline

    async def baseline_result(self, identity: EvalSuiteViewIdentity) -> EvalRunViewMetadata | None:
        if identity.id in self._baseline_results:
            return self._baseline_results[identity.id]
        baseline = await self.baseline(identity)
        value = None
        if baseline is not None:
            selector = run_view_id("result", baseline.result_revision)
            data = await self.run_data(selector)
            if data.identity != identity:
                raise HTTPException(
                    status_code=409, detail="Eval baseline result identity mismatch."
                )
            value = run_metadata(
                data,
                baseline=baseline,
                specialist=self.target(identity.target_key).request_base.agent_name,
            )
        self._baseline_results[identity.id] = value
        return value

    async def result(
        self, revision: str, identity: EvalSuiteViewIdentity
    ) -> tuple[EvalResultRecord, CorpusExecutionResult | CapturedEvaluationResultV1]:
        record, result = await load_catalog_eval_result(revision, self.store, self.registry)
        if (
            record.target.target_key != identity.target_key
            or record.corpus_revision != identity.revision
            or record.suite_id != identity.suite_id
        ):
            raise HTTPException(status_code=409, detail="Eval result suite identity mismatch.")
        suite, _ = await self.suite_document(identity)
        if record.suite_revision != suite.revision:
            raise HTTPException(status_code=409, detail="Eval result suite revision mismatch.")
        return record, result

    async def run_data(self, selector: str) -> EvalRunViewData:
        source, revision = decode_run_view(selector)
        record = result = snapshot = None
        if source == "run":
            snapshot = await self.store.load_run_view_snapshot(revision)
            if snapshot is None:
                raise HTTPException(status_code=404, detail="Eval run not found.")
        else:
            record, result = await load_catalog_eval_result(revision, self.store, self.registry)
            if type(result) is CorpusExecutionResult:
                snapshot = await self.store.load_result_run_view_snapshot(revision)
        if snapshot is not None:
            snapshot = await asyncio.to_thread(
                _exact_model, snapshot, EvalRunViewSnapshot, "snapshot"
            )
            run = snapshot.run
            if source == "run" and run.id != revision:
                raise HTTPException(status_code=409, detail="Eval run identity mismatch.")
            if source == "result" and (run.result is None or run.result.revision != revision):
                raise HTTPException(status_code=409, detail="Eval result run linkage mismatch.")
            identity = EvalSuiteViewIdentity(
                source="corpus",
                target_key=run.spec.target_key,
                revision=run.spec.corpus_revision,
                suite_id=run.spec.suite_id,
            )
            suite, cases = await self.suite_document(identity)
            if run.spec.suite_revision != suite.revision:
                raise HTTPException(status_code=409, detail="Eval run suite identity mismatch.")
            if run.result is not None:
                record, result = await self.result(run.result.revision, identity)
        else:
            assert record is not None
            identity = EvalSuiteViewIdentity(
                source="corpus",
                target_key=record.target.target_key,
                revision=record.corpus_revision,
                suite_id=record.suite_id,
            )
            suite, cases = await self.suite_document(identity)
            if record.suite_revision != suite.revision:
                raise HTTPException(status_code=409, detail="Eval result suite identity mismatch.")
        if result is not None:
            corpus = await self.corpus_document(identity)
            try:
                if type(result) is CorpusExecutionResult:
                    await asyncio.to_thread(
                        _validate_published_eval_run_for_corpus, corpus, result.run
                    )
                elif type(result) is CapturedEvaluationResultV1:
                    await asyncio.to_thread(validate_captured_result_for_corpus, result, corpus)
                else:
                    raise TypeError("Unsupported eval result document.")
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=409, detail="Eval result contradicts its corpus contract."
                ) from exc
        presentation = (
            None if result is None else await asyncio.to_thread(present_eval_result, result)
        )
        return EvalRunViewData(
            selector, snapshot, record, result, identity, suite, cases, presentation
        )

    async def run_detail(
        self,
        selector: str,
        *,
        compared_with: str | None = None,
        case_filter: EvalRunCaseFilter | None = None,
        cases_cursor: str | None = None,
        cases_limit: int = 50,
    ) -> EvalRunViewDetail:
        if decode_run_view(selector)[0] == "report":
            return await self.reports.run_detail(
                selector,
                compared_with=compared_with,
                case_filter=case_filter,
                cases_cursor=cases_cursor,
                cases_limit=cases_limit,
            )
        key = (selector, compared_with, cases_cursor, cases_limit, case_filter)
        if key in self._detail_snapshots:
            return self._detail_snapshots[key]
        value = await self._read_run_detail(
            selector,
            compared_with=compared_with,
            case_filter=case_filter,
            cases_cursor=cases_cursor,
            cases_limit=cases_limit,
        )
        if len(self._detail_snapshots) == 2:
            self._detail_snapshots.pop(next(iter(self._detail_snapshots)))
        self._detail_snapshots[key] = value
        return value

    async def _read_run_detail(
        self,
        selector: str,
        *,
        compared_with: str | None = None,
        case_filter: EvalRunCaseFilter | None = None,
        cases_cursor: str | None = None,
        cases_limit: int = 50,
    ) -> EvalRunViewDetail:
        if compared_with is not None:
            decode_run_view(compared_with)
        data = await self.run_data(selector)
        baseline = await self.baseline(data.identity)
        selected = compared_with or (
            run_view_id("result", baseline.result_revision) if baseline else None
        )
        external_comparison = selected is not None and decode_run_view(selected)[0] == "report"
        if external_comparison:
            await self.reports.record(selected)
        previous = (
            None if selected is None or external_comparison else await self.run_data(selected)
        )
        compared_run = None
        if previous is not None:
            if compared_with is None and previous.identity != data.identity:
                raise HTTPException(
                    status_code=409, detail="Eval baseline result identity mismatch."
                )
            compared_run = run_metadata(
                previous,
                baseline=await self.baseline(previous.identity),
                specialist=self.target(previous.identity.target_key).request_base.agent_name,
            )
            if (
                baseline is not None
                and previous.identity == data.identity
                and previous.record is not None
                and previous.record.revision == baseline.result_revision
            ):
                self._baseline_results[data.identity.id] = compared_run
        comparison = await asyncio.to_thread(
            compare_view_results,
            data.result,
            data.snapshot,
            None if previous is None else previous.result,
            None if previous is None else previous.snapshot,
            selected,
        )
        if external_comparison:
            comparison = EvalViewComparisonData(report_comparison(selected))
        metadata = run_metadata(
            data,
            baseline=baseline,
            specialist=self.target(data.identity.target_key).request_base.agent_name,
        )
        return EvalRunViewDetail(
            run=EvalRunViewSummary(
                **metadata.model_dump(mode="python"),
                comparison=comparison.summary,
                baseline_result=await self.baseline_result(data.identity),
            ),
            compared_run=compared_run,
            comparison=comparison.summary,
            progress=run_progress(data),
            results=case_page(
                case_results(data, previous, comparison, case_filter=case_filter),
                scope=[
                    selector,
                    None if data.record is None else data.record.revision,
                    selected,
                    None
                    if previous is None or previous.record is None
                    else previous.record.revision,
                    None if baseline is None else [baseline.result_revision, baseline.generation],
                    case_filter,
                ],
                cursor=cases_cursor,
                limit=cases_limit,
            ),
        )

    async def with_history_scope(self, run: EvalRunViewMetadata, identity: EvalSuiteViewIdentity):
        if type(run) is EvalReportedRunView:
            record = await self.reports.record(run.id)
            if identity != record.series.selector and identity != run.association:
                raise HTTPException(status_code=409, detail="Reported history identity mismatch.")
            return run
        if not isinstance(identity, EvalSuiteViewIdentity):
            raise HTTPException(
                status_code=409, detail="Native runs cannot belong to reported series."
            )
        matches = (
            run.suite == identity
            if identity.source == "corpus"
            else (
                run.suite.target_key == identity.target_key
                and run.authored_suite_revision == identity.revision
            )
        )
        if matches:
            return run
        association = (
            await self.store.load_result_addition(run.result_revision, identity.id)
            if self.store.session_additions and run.result_revision is not None
            else None
        )
        if association is None:
            raise HTTPException(status_code=409, detail="Eval history identity mismatch.")
        plan = association.receipt.plan
        if (
            plan.mode != "score"
            or plan.destination != identity
            or plan.captured_result_revision != run.result_revision
            or plan.execution_case_count != 1
            or run.origin != "captured_session"
            or run.suite.target_key != identity.target_key
        ):
            raise HTTPException(status_code=409, detail="Eval destination association is invalid.")
        return run.model_copy(
            update={
                "placement": EvalRunPlacement(
                    destination=identity,
                    operation_id=association.receipt.operation_id,
                    case_id=plan.case_id,
                    case_revision=plan.case_revision,
                )
            }
        )

    async def suite_summary(self, identity: EvalSuiteViewIdentity) -> EvalSuiteViewSummary:
        if identity.source == "reported":
            return await self.reports.suite_summary(identity)
        suite, cases = await self.suite_document(identity)
        history = await self.index(
            EvalViewQuery(target_key=identity.target_key, suite=identity, limit=1)
        )
        last = None
        if history.items:
            item = history.items[0]
            last = (
                await self.run_detail(run_view_id(item.source, item.revision), cases_limit=1)
            ).run
            last = await self.with_history_scope(last, identity)
        return EvalSuiteViewSummary(
            id=identity.id,
            identity=identity,
            source=identity.source,
            name=suite.name,
            description=suite.description,
            specialist=self.target(identity.target_key).request_base.agent_name,
            case_count=len(cases),
            trials=eval_suite_trial_policy(suite),
            baseline=await self.baseline(identity),
            baseline_result=await self.baseline_result(identity),
            baseline_scope="execution_group" if identity.source == "authored" else "suite",
            last_run=last,
        )

    async def catalog(self, query: EvalViewQuery) -> EvalViewPage[EvalSuiteViewSummary]:
        if query.target_key is not None:
            self.target(query.target_key)
        query = EvalViewQuery.model_validate(
            query.model_copy(
                update={"target_keys": tuple(sorted(self.registry.target_keys))}
            ).model_dump(mode="python")
        )
        page = await self.index(query)
        items = []
        for entry in page.items:
            if entry.source == "reported_suite":
                identity = decode_suite_view(entry.revision)
                if not isinstance(identity, EvalReportedSeriesSelector):
                    raise HTTPException(
                        status_code=409, detail="Reported catalog identity mismatch."
                    )
                reported = await self.reports.suite_summary(identity)
                if reported.identity.target_key != entry.target_key:
                    raise HTTPException(status_code=409, detail="Reported catalog target mismatch.")
                items.append(reported)
                continue
            if entry.source not in {"authored", "corpus"}:
                raise HTTPException(status_code=409, detail="Invalid eval suite catalog entry.")
            identity = EvalSuiteViewIdentity(
                source="authored" if entry.source == "authored" else "corpus",
                target_key=entry.target_key,
                revision=entry.revision,
                suite_id=entry.suite_id,
            )
            items.append(await self.suite_summary(identity))
        return EvalViewPage(
            items=tuple(items), next_cursor=page.next_cursor, has_more=page.has_more
        )

    async def suite_detail(
        self,
        identity: EvalSuiteViewIdentity,
        *,
        cases_cursor: str | None = None,
        cases_limit: int = 50,
        runs_cursor: str | None = None,
        runs_limit: int = 20,
    ) -> EvalSuiteViewDetail:
        if identity.source == "reported":
            return await self.reports.suite_detail(
                identity, runs_cursor=runs_cursor, runs_limit=runs_limit
            )
        summary = await self.suite_summary(identity)
        _, definitions = await self.suite_document(identity)
        history = await self.index(
            EvalViewQuery(
                target_key=identity.target_key, suite=identity, cursor=runs_cursor, limit=runs_limit
            )
        )
        runs = []
        for item in history.items:
            detail = await self.run_detail(run_view_id(item.source, item.revision), cases_limit=1)
            runs.append(await self.with_history_scope(detail.run, identity))
        page = case_page(
            [case_view(case) for case in definitions],
            scope=identity.id,
            cursor=cases_cursor,
            limit=cases_limit,
        )
        cases: list[EvalCaseView] = []
        latest = (
            ()
            if not page.items
            else await self.store.load_latest_case_results(
                EvalCaseHistoryQuery(suite=identity, case_ids=tuple(case.id for case in page.items))
            )
        )
        if type(latest) is not tuple or len(latest) > len(page.items):
            raise HTTPException(
                status_code=409, detail="Eval case history exceeds its query bounds."
            )
        requested_ids = {case.id for case in page.items}
        latest_by_id = {}
        for item in latest:
            item = _exact_model(item, EvalCaseResultReference, "case result reference")
            if item.case_id not in requested_ids or item.case_id in latest_by_id:
                raise HTTPException(status_code=409, detail="Eval case history identity mismatch.")
            latest_by_id[item.case_id] = item
        for case in page.items:
            item = latest_by_id.get(case.id)
            outcome = None
            if item is not None:
                detail = await self.run_detail(
                    run_view_id(item.source, item.revision), cases_limit=1000
                )
                await self.with_history_scope(detail.run, identity)
                outcome = next(
                    (
                        item.presentation
                        for item in detail.results.items
                        if item.presentation.case_id == case.id
                    ),
                    None,
                )
                if outcome is None:
                    raise HTTPException(
                        status_code=409, detail="Eval case history identity mismatch."
                    )
            cases.append(case.model_copy(update={"last": outcome}))
        return EvalSuiteViewDetail(
            suite=summary,
            cases=EvalViewPage(
                items=tuple(cases),
                next_cursor=page.next_cursor,
                has_more=page.has_more,
                total=page.total,
            ),
            runs=EvalViewPage(
                items=tuple(runs), next_cursor=history.next_cursor, has_more=history.has_more
            ),
        )
