"""Stored evaluation results, reports, comparisons, and baseline selection routes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from cayu._validation import compact_json_utf8_size
from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.execution_comparison import compare_eval_results
from cayu.evals.execution_reporting import eval_result_report_to_json, render_eval_result_html
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.results import CapturedEvaluationResultV1, EvalResultOrigin
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_DEFAULT_PAGE_SIZE,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_IDENTIFIER_CHARS,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalBaselineConflict,
    EvalBaselineKey,
    EvalBaselineUpdate,
    EvalResultPage,
    EvalResultQuery,
    EvalResultRecord,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.server._http_json import (
    _json_request_openapi,
    _model_json_response,
    _render_utf8,
    _validated_private_json_body,
)
from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    EvalBaselineSelectionRequest,
    EvalBaselineSelectionResponse,
    EvalResultComparisonRequest,
    EvalResultComparisonResponse,
    EvalResultDetailResponse,
)

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.applications import CayuApp
    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


def _eval_result_record_matches_document(
    record: EvalResultRecord,
    result: CorpusExecutionResult | CapturedEvaluationResultV1,
) -> bool:
    """Fail closed when a custom store returns mismatched result metadata."""

    if type(result) is CorpusExecutionResult:
        origin = EvalResultOrigin.FRESH_EXECUTION
        target = result.target
        corpus_revision = result.run.corpus_revision
        suite_id = result.run.suite_id
        suite_revision = result.run.suite_revision
        status = result.run.status
        score = result.run.score
    elif type(result) is CapturedEvaluationResultV1:
        origin = EvalResultOrigin.CAPTURED_SESSION
        target = result.target
        corpus_revision = result.corpus_revision
        suite_id = result.suite_id
        suite_revision = result.suite_revision
        status = result.score.status
        score = result.score.score
    else:
        return False
    document_bytes = compact_json_utf8_size(result.model_dump(mode="json"))
    return (
        record.revision == result.revision
        and record.origin == origin
        and record.target.target_key == target.target_key
        and record.target.application_release_id == target.application_release_id
        and record.target.app_manifest_schema_version == target.app_manifest_schema_version
        and record.target.app_manifest_fingerprint == target.app_manifest_fingerprint
        and record.corpus_revision == corpus_revision
        and record.suite_id == suite_id
        and record.suite_revision == suite_revision
        and record.status == status
        and record.score == score
        and record.document_bytes == document_bytes
    )


def register_evaluation_result_routes(
    bounded_captured_evaluation_router: APIRouter,
    *,
    cayu_app: CayuApp,
    eval_registry: EvalTargetRegistry,
    captured_eval_store: EvalStore | None,
    protected: Sequence[Depends],
    optional_auth_context: Depends,
) -> None:
    """Register result routes with the shared bounded router and authentication wiring."""

    # FastAPI uses the shared dependency object as a default for typed auth context.
    baseline_auth_context = cast("AuthContext | None", optional_auth_context)

    @bounded_captured_evaluation_router.get(
        "/evals/results",
        response_model=EvalResultPage,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_results(
        target_key: Annotated[str, Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS)],
        cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
        limit: Annotated[int, Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE)] = (
            EVAL_STORE_DEFAULT_PAGE_SIZE
        ),
        max_result_bytes: Annotated[
            int,
            Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
        ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
        origin: Annotated[EvalResultOrigin | None, Query()] = None,
    ) -> EvalResultPage:
        if captured_eval_store is None or not captured_eval_store.captured_results:
            raise HTTPException(status_code=409, detail="Eval result catalog is unavailable.")
        if eval_registry.get(target_key) is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        try:
            return await captured_eval_store.list_results(
                EvalResultQuery(
                    target_key=target_key,
                    origin=origin,
                    cursor=cursor,
                    limit=limit,
                    max_result_bytes=max_result_bytes,
                )
            )
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=409, detail="Eval result catalog is unavailable."
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval result catalog exceeds the requested byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc

    async def _load_catalog_eval_result(
        result_revision: str,
    ) -> tuple[EvalResultRecord, CorpusExecutionResult | CapturedEvaluationResultV1]:
        if captured_eval_store is None or not captured_eval_store.captured_results:
            raise HTTPException(status_code=409, detail="Eval result catalog is unavailable.")
        try:
            record = await captured_eval_store.load_result_record(result_revision)
            result = await captured_eval_store.load_result_by_revision(result_revision)
        except NotImplementedError as exc:
            raise HTTPException(
                status_code=409, detail="Eval result catalog is unavailable."
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(status_code=413, detail="Eval result is too large.") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid eval result revision.") from exc
        if record is None or result is None:
            raise HTTPException(status_code=404, detail="Eval result not found.")
        if not await asyncio.to_thread(
            _eval_result_record_matches_document,
            record,
            result,
        ):
            raise HTTPException(
                status_code=409,
                detail="Eval result catalog metadata does not match its stored document.",
            )
        if eval_registry.get(record.target.target_key) is None:
            raise HTTPException(status_code=404, detail="Eval result not found.")
        return record, result

    @bounded_captured_evaluation_router.get(
        "/evals/results/{result_revision}",
        response_model=EvalResultDetailResponse,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_eval_result(result_revision: str) -> EvalResultDetailResponse:
        record, result = await _load_catalog_eval_result(result_revision)
        store = captured_eval_store
        if store is None:
            raise RuntimeError("Loaded eval result has no captured-result store.")
        key = EvalBaselineKey(
            target_key=record.target.target_key,
            corpus_revision=record.corpus_revision,
            suite_id=record.suite_id,
        )
        baseline = await store.load_baseline(key)
        presentation = await asyncio.to_thread(present_eval_result, result)
        return await asyncio.to_thread(
            EvalResultDetailResponse,
            record=record,
            result=result,
            presentation=presentation,
            baseline=baseline,
        )

    @bounded_captured_evaluation_router.get(
        "/evals/results/{result_revision}/report.json",
        response_class=Response,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def download_catalog_eval_json_report(result_revision: str) -> Response:
        record, result = await _load_catalog_eval_result(result_revision)
        report = await asyncio.to_thread(_render_utf8, eval_result_report_to_json, result)
        filename = f"{record.revision.removeprefix('sha256:')}.eval-result.json"
        return Response(
            content=report,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @bounded_captured_evaluation_router.get(
        "/evals/results/{result_revision}/report.html",
        response_class=Response,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def download_catalog_eval_html_report(result_revision: str) -> Response:
        record, result = await _load_catalog_eval_result(result_revision)
        report = await asyncio.to_thread(_render_utf8, render_eval_result_html, result)
        filename = f"{record.revision.removeprefix('sha256:')}.eval-report.html"
        return Response(
            content=report,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @bounded_captured_evaluation_router.post(
        "/evals/result-comparisons",
        response_model=EvalResultComparisonResponse,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
        openapi_extra=_json_request_openapi(EvalResultComparisonRequest),
    )
    async def compare_catalog_eval_results(request: Request) -> Response:
        body = await _validated_private_json_body(
            request,
            EvalResultComparisonRequest,
            invalid_detail="Invalid Evals request.",
        )
        baseline_record, baseline = await _load_catalog_eval_result(body.baseline_result_revision)
        if body.current_result_revision == body.baseline_result_revision:
            current_record, current = baseline_record, baseline
        else:
            current_record, current = await _load_catalog_eval_result(body.current_result_revision)
        comparison = await asyncio.to_thread(
            compare_eval_results,
            baseline,
            current,
            score_tolerance=body.score_tolerance,
        )
        response = await asyncio.to_thread(
            EvalResultComparisonResponse,
            baseline=baseline_record,
            current=current_record,
            comparison=comparison,
        )
        return await _model_json_response(response, EvalResultComparisonResponse)

    @bounded_captured_evaluation_router.post(
        "/evals/results/{result_revision}/baseline",
        response_model=EvalBaselineSelectionResponse,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    )
    async def select_eval_baseline(
        result_revision: str,
        body: EvalBaselineSelectionRequest,
        auth_context: AuthContext | None = baseline_auth_context,
    ) -> EvalBaselineSelectionResponse:
        if captured_eval_store is None or not captured_eval_store.captured_results:
            raise HTTPException(status_code=409, detail="Eval baselines are unavailable.")
        if body.result_revision != result_revision:
            raise HTTPException(
                status_code=400,
                detail="Result revision path and request body do not match.",
            )
        try:
            record = await captured_eval_store.load_result_record(result_revision)
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid eval result revision.") from exc
        if record is None or eval_registry.get(record.target.target_key) is None:
            raise HTTPException(status_code=404, detail="Eval result not found.")
        actor_id = (
            "cayu:trusted-local-development" if auth_context is None else auth_context.subject
        )
        key = EvalBaselineKey(
            target_key=record.target.target_key,
            corpus_revision=record.corpus_revision,
            suite_id=record.suite_id,
        )
        try:
            mutation = await captured_eval_store.set_baseline(
                EvalBaselineUpdate(
                    key=key,
                    result_revision=result_revision,
                    expected_generation=body.expected_generation,
                    operation_id=body.operation_id,
                    actor_id=actor_id,
                ),
                redact_json=cayu_app.redact_json,
            )
        except EvalBaselineConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="The authenticated baseline actor cannot cross the public boundary.",
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid baseline selection.") from exc
        baseline = await captured_eval_store.load_baseline(key)
        if baseline is None:
            raise RuntimeError("Committed eval baseline is unavailable.")
        return EvalBaselineSelectionResponse(baseline=baseline, mutation=mutation)
