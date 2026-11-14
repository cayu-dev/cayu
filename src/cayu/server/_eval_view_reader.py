"""Read orchestration for suite views; no worker or baseline mutation authority.

Pages read the store's summary tables (revision 154): the suite catalog, result
summaries, case outcomes, trial checkpoint summaries and recorded comparisons.
A whole result document is decoded only when a store keeps no summary of it,
and then the reader derives the same rows with the functions the store's
writers use and records them. Pass ``use_summaries=False`` to read every value
from the documents instead; that is the reference the summaries are checked
against.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from fastapi import HTTPException
from pydantic import BaseModel

from cayu.evals.corpus import EvalCorpusDocument, EvalSuiteSpec, eval_suite_trial_policy
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.published import _validate_published_eval_run_for_corpus
from cayu.evals.read_views import (
    EvalCaseHistoryQuery,
    EvalCaseResultReference,
    EvalCaseResultView,
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
    EvalViewComparison,
    EvalViewIndexPage,
    EvalViewPage,
    EvalViewQuery,
    decode_run_view,
    decode_suite_view,
    run_view_id,
    validate_view_index,
)
from cayu.evals.reported import EvalReportedSeriesSelector
from cayu.evals.results import (
    CapturedEvaluationResultV1,
    EvalResultOrigin,
    validate_captured_result_for_corpus,
)
from cayu.evals.store import (
    EvalBaselineKey,
    EvalBaselineRecord,
    EvalResultRecord,
    EvalStoreResultTooLarge,
    _exact_model,
)
from cayu.evals.suite_authoring import _validated_eval_suite_document
from cayu.evals.view_comparison import (
    EvalViewComparisonData,
    compare_view_results,
    comparison_from_record,
    comparison_record,
    pending_comparison,
)
from cayu.evals.view_summaries import (
    EvalCaseOutcomePage,
    EvalCaseOutcomeQuery,
    EvalCaseOutcomeV1,
    EvalResultViewRows,
    EvalResultViewSummaryV1,
    page_case_outcomes,
    summarize_result,
)
from cayu.server._eval_report_views import EvalReportViewReader, report_comparison
from cayu.server._eval_result_loader import load_catalog_eval_result
from cayu.server._eval_view_projection import (
    EvalCaseDefinitionData,
    EvalRunViewData,
    case_cursor,
    case_cursor_after,
    case_definition,
    case_definitions,
    case_outcome_query,
    case_page,
    case_results,
    run_metadata,
    run_progress,
)

if TYPE_CHECKING:
    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


ViewModel = TypeVar("ViewModel", bound=BaseModel)
_RECORD_MISMATCH = "Eval result catalog metadata does not match its stored document."
_LOGGER = logging.getLogger(__name__)
SuiteData = tuple[EvalSuiteSpec, tuple[EvalCaseDefinitionData, ...]]


@dataclass(frozen=True, slots=True)
class _RunRead:
    run: EvalRunViewSummary
    data: EvalRunViewData
    comparison: EvalViewComparisonData
    compared_run: EvalRunViewMetadata | None
    previous: EvalRunViewData | None
    selected: str | None


def enforce_view_bytes(value: ViewModel, maximum: int) -> ViewModel:
    if len(value.model_dump_json().encode()) > maximum:
        raise EvalStoreResultTooLarge(maximum)
    return value


class EvalViewReader:
    def __init__(
        self,
        store: EvalStore,
        registry: EvalTargetRegistry,
        *,
        use_summaries: bool | None = None,
    ):
        self.store = store
        self.registry = registry
        self.reports = EvalReportViewReader(store, registry)
        self.use_summaries = store.view_summaries if use_summaries is None else use_summaries
        if self.use_summaries and not store.view_summaries:
            raise ValueError("This EvalStore keeps no view summaries.")
        self._baseline_snapshots: dict[tuple[str, str, str], EvalBaselineRecord | None] = {}
        self._baseline_results: dict[str, EvalRunViewMetadata | None] = {}
        self._run_data: dict[str, EvalRunViewData] = {}
        self._view_rows: dict[str, EvalResultViewRows] = {}
        self._suites: dict[tuple[str, str, str | None], SuiteData] = {}
        self._corpus_documents: dict[tuple[str, str], EvalCorpusDocument] = {}

    async def index(self, query: EvalViewQuery) -> EvalViewIndexPage:
        return validate_view_index(await self.store.list_view_index(query), query)

    def target(self, key):
        target = self.registry.get(key)
        if target is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        return target

    def specialist(self, identity: EvalSuiteViewIdentity) -> str:
        return self.target(identity.target_key).request_base.agent_name

    async def resolve_corpus_suite(
        self, revision: str, suite_id: str, target_key: str | None
    ) -> EvalSuiteViewIdentity:
        document = await self.store.load_corpus(revision)
        if document is None:
            raise HTTPException(status_code=404, detail="Eval suite not found.")
        document = await asyncio.to_thread(_exact_model, document, EvalCorpusDocument, "corpus")
        if document.revision != revision:
            raise HTTPException(status_code=409, detail="Eval suite identity mismatch.")
        self.target(document.target_key)
        if target_key is not None and target_key != document.target_key:
            raise HTTPException(status_code=404, detail="Eval suite not found.")
        if not any(suite.id == suite_id for suite in document.suites):
            raise HTTPException(status_code=404, detail="Eval suite not found.")
        return EvalSuiteViewIdentity(
            source="corpus", target_key=document.target_key, revision=revision, suite_id=suite_id
        )

    async def suite_document(self, identity):
        """The suite spec and case specs from the stored document itself."""
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

    async def suite_data(self, identity: EvalSuiteViewIdentity) -> SuiteData:
        """The suite spec and what views need of its cases, from the catalog when kept."""
        key = (identity.target_key, identity.revision, identity.suite_id)
        if key in self._suites:
            return self._suites[key]
        self.target(identity.target_key)
        value = None
        if self.use_summaries and identity.source == "corpus":
            assert identity.suite_id is not None
            catalog = await self.store.load_suite_catalog(((identity.revision, identity.suite_id),))
            entry = catalog.get((identity.revision, identity.suite_id))
            if entry is not None:
                if entry.target_key != identity.target_key:
                    raise HTTPException(status_code=409, detail="Eval suite identity mismatch.")
                suite = await asyncio.to_thread(EvalSuiteSpec.model_validate, entry.suite)
                if suite.id != identity.suite_id or suite.revision != entry.suite_revision:
                    raise HTTPException(status_code=409, detail="Eval suite identity mismatch.")
                value = suite, tuple(case_definition(case) for case in entry.cases)
        if value is None:
            suite, cases = await self.suite_document(identity)
            value = suite, case_definitions(cases)
        self._suites[key] = value
        return value

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
            data = await self.run_data(run_view_id("result", baseline.result_revision))
            if data.identity != identity:
                raise HTTPException(
                    status_code=409, detail="Eval baseline result identity mismatch."
                )
            value = run_metadata(data, baseline=baseline, specialist=self.specialist(identity))
        self._baseline_results[identity.id] = value
        return value

    async def result_record(self, revision: str) -> EvalResultRecord:
        if not self.store.captured_results:
            raise HTTPException(status_code=409, detail="Eval result catalog is unavailable.")
        try:
            record = await self.store.load_result_record(revision)
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=409, detail="Eval result catalog is unavailable."
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid eval result revision.") from exc
        if record is None:
            raise HTTPException(status_code=404, detail="Eval result not found.")
        try:
            record = await asyncio.to_thread(_exact_model, record, EvalResultRecord, "record")
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=_RECORD_MISMATCH) from exc
        if record.revision != revision:
            raise HTTPException(status_code=409, detail=_RECORD_MISMATCH)
        if self.registry.get(record.target.target_key) is None:
            raise HTTPException(status_code=404, detail="Eval result not found.")
        return record

    async def view_rows(
        self, record: EvalResultRecord, identity: EvalSuiteViewIdentity
    ) -> EvalResultViewRows:
        """Derive a result's rows from its document, checked against its corpus."""
        if record.revision in self._view_rows:
            return self._view_rows[record.revision]
        loaded, result = await load_catalog_eval_result(record.revision, self.store, self.registry)
        if loaded != record:
            raise HTTPException(status_code=409, detail=_RECORD_MISMATCH)
        corpus = await self.corpus_document(identity)
        try:
            if type(result) is CorpusExecutionResult:
                await asyncio.to_thread(_validate_published_eval_run_for_corpus, corpus, result.run)
            elif type(result) is CapturedEvaluationResultV1:
                await asyncio.to_thread(validate_captured_result_for_corpus, result, corpus)
            else:
                raise TypeError("Unsupported eval result document.")
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail="Eval result contradicts its corpus contract."
            ) from exc
        rows = await asyncio.to_thread(summarize_result, result)
        if self.use_summaries:
            # The rows are a function of the immutable result, so recording them
            # here is the same as the writer having recorded them.
            await self._record(self.store.save_result_view_rows(rows))
        if len(self._view_rows) == 4:
            self._view_rows.pop(next(iter(self._view_rows)))
        self._view_rows[record.revision] = rows
        return rows

    async def result_summary(
        self, record: EvalResultRecord, identity: EvalSuiteViewIdentity
    ) -> EvalResultViewSummaryV1:
        if self.use_summaries:
            stored = await self.store.load_result_view_summaries((record.revision,))
            summary = stored.get(record.revision)
            if summary is not None:
                if not summary.describes(record):
                    raise HTTPException(status_code=409, detail=_RECORD_MISMATCH)
                return summary
        return (await self.view_rows(record, identity)).summary

    async def case_outcomes(
        self, data: EvalRunViewData, query: EvalCaseOutcomeQuery
    ) -> EvalCaseOutcomePage:
        assert data.record is not None
        if self.use_summaries and data.record.revision not in self._view_rows:
            page = await self.store.list_case_outcomes(query)
            if page is not None:
                return page
        rows = await self.view_rows(data.record, data.identity)
        return page_case_outcomes(rows.outcomes, query)

    async def run_data(self, selector: str) -> EvalRunViewData:
        if selector in self._run_data:
            return self._run_data[selector]
        source, revision = decode_run_view(selector)
        record = snapshot = None
        if source == "run":
            snapshot = await self.store.load_run_view_snapshot(revision)
            if snapshot is None:
                raise HTTPException(status_code=404, detail="Eval run not found.")
        else:
            record = await self.result_record(revision)
            if record.origin is EvalResultOrigin.FRESH_EXECUTION:
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
            suite, definitions = await self.suite_data(identity)
            if run.spec.suite_revision != suite.revision:
                raise HTTPException(status_code=409, detail="Eval run suite identity mismatch.")
            if run.result is not None and record is None:
                record = await self.result_record(run.result.revision)
        else:
            assert record is not None
            identity = EvalSuiteViewIdentity(
                source="corpus",
                target_key=record.target.target_key,
                revision=record.corpus_revision,
                suite_id=record.suite_id,
            )
            suite, definitions = await self.suite_data(identity)
        summary = None
        if record is not None:
            # The record must describe its document before it may place the run.
            summary = await self.result_summary(record, identity)
            if (
                record.target.target_key != identity.target_key
                or record.corpus_revision != identity.revision
                or record.suite_id != identity.suite_id
            ):
                raise HTTPException(status_code=409, detail="Eval result suite identity mismatch.")
            if record.suite_revision != suite.revision:
                raise HTTPException(status_code=409, detail="Eval result suite revision mismatch.")
        data = EvalRunViewData(
            selector=selector,
            snapshot=snapshot,
            record=record,
            identity=identity,
            suite=suite,
            definitions=definitions,
            summary=summary,
            fresh=record is not None and record.origin is EvalResultOrigin.FRESH_EXECUTION,
        )
        if len(self._run_data) == 64:
            self._run_data.pop(next(iter(self._run_data)))
        self._run_data[selector] = data
        return data

    async def comparison(
        self, data: EvalRunViewData, previous: EvalRunViewData | None, selected: str | None
    ) -> EvalViewComparisonData:
        if data.record is None or (
            selected is not None and (previous is None or previous.record is None)
        ):
            return pending_comparison(
                data.snapshot,
                None if previous is None else previous.snapshot,
                current_has_result=data.record is not None,
                selected=selected,
            )
        if previous is None or previous.record is None:
            return EvalViewComparisonData(EvalViewComparison(state="no_baseline"))
        current, before = data.record.revision, previous.record.revision
        if self.use_summaries:
            record = await self.store.load_result_comparison(current, before)
            if record is not None:
                return comparison_from_record(record, selected)
        _, current_result = await load_catalog_eval_result(current, self.store, self.registry)
        _, previous_result = await load_catalog_eval_result(before, self.store, self.registry)
        value = await asyncio.to_thread(
            compare_view_results,
            current_result,
            data.snapshot,
            previous_result,
            previous.snapshot,
            selected,
        )
        if self.use_summaries:
            await self._record(
                self.store.save_result_comparison(current, before, comparison_record(value))
            )
        return value

    @staticmethod
    async def _record(write) -> None:
        """Record a derived row; a store that cannot write (read-only) still serves reads."""
        try:
            await write
        except Exception:
            _LOGGER.warning("Could not record an eval view summary row.", exc_info=True)

    async def run_summary(self, selector: str, *, compared_with: str | None = None) -> _RunRead:
        """A run's summary row with its comparison, without any of its cases."""
        if compared_with is not None:
            decode_run_view(compared_with)
        data = await self.run_data(selector)
        baseline = await self.baseline(data.identity)
        selected = compared_with or (
            run_view_id("result", baseline.result_revision) if baseline else None
        )
        external = selected is not None and decode_run_view(selected)[0] == "report"
        previous = None
        if external:
            await self.reports.record(selected)
        elif selected is not None:
            previous = await self.run_data(selected)
        compared_run = None
        if previous is not None:
            if compared_with is None and previous.identity != data.identity:
                raise HTTPException(
                    status_code=409, detail="Eval baseline result identity mismatch."
                )
            compared_run = run_metadata(
                previous,
                baseline=await self.baseline(previous.identity),
                specialist=self.specialist(previous.identity),
            )
            if (
                baseline is not None
                and previous.identity == data.identity
                and previous.record is not None
                and previous.record.revision == baseline.result_revision
            ):
                self._baseline_results[data.identity.id] = compared_run
        if external:
            comparison = EvalViewComparisonData(report_comparison(selected))
        else:
            comparison = await self.comparison(data, previous, selected)
        metadata = run_metadata(data, baseline=baseline, specialist=self.specialist(data.identity))
        summary = EvalRunViewSummary(
            **metadata.model_dump(mode="python"),
            comparison=comparison.summary,
            baseline_result=await self.baseline_result(data.identity),
        )
        return _RunRead(summary, data, comparison, compared_run, previous, selected)

    async def run_detail(
        self,
        selector: str,
        *,
        compared_with: str | None = None,
        case_filter: EvalRunCaseFilter | None = None,
        cases_cursor: str | None = None,
        cases_limit: int = 50,
        max_bytes: int | None = None,
    ) -> EvalRunViewDetail:
        if decode_run_view(selector)[0] == "report":
            return await self.reports.run_detail(
                selector,
                compared_with=compared_with,
                case_filter=case_filter,
                cases_cursor=cases_cursor,
                cases_limit=cases_limit,
            )
        read = await self.run_summary(selector, compared_with=compared_with)
        summary, data, comparison, previous = read.run, read.data, read.comparison, read.previous
        selected, baseline = read.selected, summary.baseline
        scope = [
            selector,
            data.result_revision,
            selected,
            None if previous is None else previous.result_revision,
            None if baseline is None else [baseline.result_revision, baseline.generation],
            case_filter,
        ]
        after = case_cursor_after(cases_cursor, scope)
        query = case_outcome_query(
            data, comparison, case_filter=case_filter, after=after, limit=cases_limit
        )
        rows: list[EvalCaseResultView] = []
        total, more = 0, False
        if query is not None:
            page = await self.case_outcomes(data, query)
            compared: dict[str, EvalCaseOutcomeV1] = {}
            if (
                page.items
                and comparison.summary.state == "compared"
                and previous is not None
                and previous.record is not None
            ):
                compared_page = await self.case_outcomes(
                    previous,
                    EvalCaseOutcomeQuery(
                        revision=previous.record.revision,
                        case_ids=tuple(item.case_id for item in page.items),
                    ),
                )
                compared = {item.case_id: item for item in compared_page.items}
            rows = case_results(data, page.items, compared, comparison)
            total, more = page.total, page.has_more

        def detail(kept: list[EvalCaseResultView], has_more: bool) -> EvalRunViewDetail:
            return EvalRunViewDetail(
                run=summary,
                compared_run=read.compared_run,
                comparison=comparison.summary,
                progress=run_progress(data),
                results=EvalViewPage(
                    items=tuple(kept),
                    total=total,
                    has_more=has_more,
                    next_cursor=case_cursor(scope, kept[-1].case.id) if has_more else None,
                ),
            )

        result = detail(rows, more)
        if max_bytes is None:
            return result
        # A page over the byte budget keeps the cases that fit and pages on to the
        # rest, so a run with large evidence still opens; only a single case over
        # the budget is too large to show.
        size = len(result.model_dump_json().encode())
        while size > max_bytes and len(rows) > 1:
            excess = size - max_bytes
            while len(rows) > 1 and excess > 0:
                excess -= len(rows.pop().model_dump_json().encode())
            result = detail(rows, True)
            size = len(result.model_dump_json().encode())
        return enforce_view_bytes(result, max_bytes)

    async def history_run(self, item, identity: EvalSuiteViewIdentity):
        selector = run_view_id(item.source, item.revision)
        if item.source == "report":
            # A corpus suite's history includes the reports associated with it.
            run = self.reports.summary(await self.reports.record(selector))
        else:
            run = (await self.run_summary(selector)).run
        return await self.with_history_scope(run, identity)

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
        suite, cases = await self.suite_data(identity)
        history = await self.index(
            EvalViewQuery(target_key=identity.target_key, suite=identity, limit=1)
        )
        last = await self.history_run(history.items[0], identity) if history.items else None
        return EvalSuiteViewSummary(
            id=identity.id,
            identity=identity,
            source=identity.source,
            name=suite.name,
            description=suite.description,
            specialist=self.specialist(identity),
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

    async def latest_case_outcome(
        self, item: EvalCaseResultReference, identity: EvalSuiteViewIdentity
    ):
        selector = run_view_id(item.source, item.revision)
        cached = self._run_data.get(selector)
        if cached is not None and cached.record is None:
            # The latest-result query can see a publication after history cached
            # this run as live. Refresh that snapshot before loading its outcome.
            self._run_data.pop(selector)
        await self.with_history_scope((await self.run_summary(selector)).run, identity)
        data = await self.run_data(selector)
        if data.record is None or data.summary is None:
            return None
        page = await self.case_outcomes(
            data,
            EvalCaseOutcomeQuery(
                revision=data.record.revision, case_ids=(item.case_id,), include_evidence=False
            ),
        )
        return page.items[0].presentation if page.items else None

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
        _, definitions = await self.suite_data(identity)
        history = await self.index(
            EvalViewQuery(
                target_key=identity.target_key, suite=identity, cursor=runs_cursor, limit=runs_limit
            )
        )
        runs = [await self.history_run(item, identity) for item in history.items]
        page = case_page(
            [case.view for case in definitions],
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
                outcome = await self.latest_case_outcome(item, identity)
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
