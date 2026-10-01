"""Durable evaluation run management, result and report HTTP routes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response

from cayu.evals.execution_comparison import compare_corpus_execution_results
from cayu.evals.execution_reporting import eval_result_report_to_json, render_corpus_execution_html
from cayu.evals.result_presentation import present_eval_result
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_DEFAULT_PAGE_SIZE,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_IDENTIFIER_CHARS,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalBaselineKey,
    EvalRunPage,
    EvalRunQuery,
    EvalRunRecord,
    EvalRunStateConflict,
    EvalRunStatus,
    EvalScenarioApprovalSubmission,
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
    EVALS_ENDPOINT_RESPONSES,
    EvalComparisonRequest,
    EvalComparisonResponse,
    EvalResultResponse,
    EvalScenarioApprovalRequest,
)

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


def register_evaluation_run_routes(
    bounded_evals_router: APIRouter,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    captured_eval_store: EvalStore | None,
    protected: Sequence[Depends],
    optional_auth_context: Depends,
) -> None:
    """Register run management with explicit storage and shared HTTP boundaries."""

    # FastAPI uses the shared dependency object as a default for typed auth context.
    approval_auth_context = cast("AuthContext | None", optional_auth_context)

    async def _load_eval_run(run_id: str):
        try:
            run = await eval_store.load_run(run_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc
        if run is None:
            raise HTTPException(status_code=404, detail="Eval run not found.")
        if active_eval_registry.get(run.spec.target_key) is None:
            raise HTTPException(status_code=404, detail="Eval run not found.")
        return run

    async def _load_eval_result(run_id: str):
        # Authorize the run before hydrating its potentially large result.
        await _load_eval_run(run_id)
        try:
            result = await eval_store.load_result(run_id)
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval result exceeds the server byte limit.",
            ) from exc
        if result is None:
            raise HTTPException(
                status_code=409,
                detail="Eval run has no completed result.",
            )
        # Result publication and run terminalization are one store transaction.
        # Reload after the result becomes visible so a concurrent publication
        # cannot pair it with the active record observed above.
        run = await _load_eval_run(run_id)
        return run, result

    @bounded_evals_router.get(
        "/evals/runs",
        response_model=EvalRunPage,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_runs(
        target_key: Annotated[
            str | None,
            Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS),
        ] = None,
        status: EvalRunStatus | None = None,
        corpus_revision: str | None = None,
        cursor: Annotated[str | None, Query(max_length=EVAL_STORE_MAX_CURSOR_BYTES)] = None,
        limit: Annotated[
            int,
            Query(ge=1, le=EVAL_STORE_MAX_PAGE_SIZE),
        ] = EVAL_STORE_DEFAULT_PAGE_SIZE,
        max_result_bytes: Annotated[
            int,
            Query(ge=1_024, le=EVAL_STORE_MAX_PAGE_BYTES),
        ] = EVAL_STORE_DEFAULT_PAGE_BYTES,
    ):
        selected_key = active_eval_registry.default_target_key if target_key is None else target_key
        eval_target = active_eval_registry.get(selected_key)
        if eval_target is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        try:
            return await eval_store.list_runs(
                EvalRunQuery(
                    target_key=eval_target.key,
                    status=status,
                    corpus_revision=corpus_revision,
                    cursor=cursor,
                    limit=limit,
                    max_result_bytes=max_result_bytes,
                )
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval run page exceeds the requested byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc

    @bounded_evals_router.get(
        "/evals/runs/{run_id}",
        response_model=EvalRunRecord,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_eval_run(run_id: str):
        return await _load_eval_run(run_id)

    @bounded_evals_router.post(
        "/evals/runs/{run_id}/scenario-approval",
        response_model=EvalRunRecord,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def submit_eval_scenario_approval(
        run_id: str,
        body: EvalScenarioApprovalRequest,
        auth_context: AuthContext | None = approval_auth_context,
    ) -> EvalRunRecord:
        run = await _load_eval_run(run_id)
        if run.spec.invocation.scenario is None:
            raise HTTPException(status_code=409, detail="Eval run is not a scenario run.")
        actor_id = (
            "cayu:trusted-local-development" if auth_context is None else auth_context.subject
        )
        try:
            return await eval_store.submit_scenario_approval(
                run_id,
                EvalScenarioApprovalSubmission(
                    expected_progress_revision=body.expected_progress_revision,
                    trial_number=body.trial_number,
                    event_id=body.event_id,
                    decision=body.decision,
                    reason=body.reason,
                    actor_id=actor_id,
                ),
            )
        except EvalRunStateConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="Scenario approval checkpoint changed before submission.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid scenario approval.") from exc

    @bounded_evals_router.post(
        "/evals/runs/{run_id}/cancel",
        response_model=EvalRunRecord,
        status_code=202,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def cancel_eval_run(run_id: str):
        await _load_eval_run(run_id)
        try:
            return await eval_store.request_cancel(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Eval run not found.") from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc

    @bounded_evals_router.get(
        "/evals/runs/{run_id}/result",
        response_model=EvalResultResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_eval_result(run_id: str) -> Response:
        run, result = await _load_eval_result(run_id)
        trial_evidence_links = await eval_store.load_trial_evidence_links(run_id)
        baseline = None
        if captured_eval_store is not None and captured_eval_store.captured_results:
            baseline = await captured_eval_store.load_baseline(
                EvalBaselineKey(
                    target_key=run.spec.target_key,
                    corpus_revision=run.spec.corpus_revision,
                    suite_id=run.spec.suite_id,
                )
            )
        presentation = await asyncio.to_thread(present_eval_result, result)
        response = await asyncio.to_thread(
            EvalResultResponse,
            run=run,
            result=result,
            presentation=presentation,
            baseline=baseline,
            trial_evidence_links=trial_evidence_links,
        )
        return await _model_json_response(response, EvalResultResponse)

    @bounded_evals_router.get(
        "/evals/runs/{run_id}/report.json",
        response_class=Response,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def download_eval_json_report(run_id: str) -> Response:
        _, result = await _load_eval_result(run_id)
        report = await asyncio.to_thread(
            _render_utf8,
            eval_result_report_to_json,
            result,
        )
        return Response(
            content=report,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.eval-result.json"'},
        )

    @bounded_evals_router.get(
        "/evals/runs/{run_id}/report.html",
        response_class=Response,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def download_eval_html_report(run_id: str) -> Response:
        _, result = await _load_eval_result(run_id)
        report = await asyncio.to_thread(
            _render_utf8,
            render_corpus_execution_html,
            result,
        )
        return Response(
            content=report,
            media_type="text/html",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.eval-report.html"'},
        )

    @bounded_evals_router.post(
        "/evals/comparisons",
        response_model=EvalComparisonResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
        openapi_extra=_json_request_openapi(EvalComparisonRequest),
    )
    async def compare_eval_runs(request: Request) -> Response:
        body = await _validated_private_json_body(
            request,
            EvalComparisonRequest,
            invalid_detail="Invalid Evals request.",
        )
        baseline_run, baseline = await _load_eval_result(body.baseline_run_id)
        if body.current_run_id == body.baseline_run_id:
            current_run, current = baseline_run, baseline
        else:
            current_run, current = await _load_eval_result(body.current_run_id)
        comparison = await asyncio.to_thread(
            compare_corpus_execution_results,
            baseline,
            current,
        )
        response = await asyncio.to_thread(
            EvalComparisonResponse,
            baseline=baseline_run,
            current=current_run,
            comparison=comparison,
        )
        return await _model_json_response(response, EvalComparisonResponse)
