"""Memory experiment reports from exact stored execution evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from cayu.evals.execution import CorpusExecutionResult
from cayu.evals.memory_reporting import (
    MemoryExperimentReport,
    MemoryExperimentReportRequest,
    build_memory_experiment_report,
    render_memory_experiment_report_html,
)
from cayu.evals.store import EvalRunStatus, EvalStoreResultTooLarge
from cayu.server._http_json import _model_json_response, _validated_private_json_body
from cayu.server.contracts import CAPTURED_EVALUATION_ENDPOINT_RESPONSES

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalRunRecord, EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


def register_memory_report_routes(
    bounded_memory_report_router: APIRouter,
    *,
    captured_eval_store: EvalStore | None,
    active_eval_registry: EvalTargetRegistry,
    protected: Sequence[Depends],
) -> None:
    """Register report endpoints with their existing private HTTP boundary."""

    async def _build_stored_memory_experiment_report(
        body: MemoryExperimentReportRequest,
    ) -> MemoryExperimentReport:
        store = captured_eval_store
        if store is None:
            raise HTTPException(status_code=409, detail="Eval result storage is unavailable.")
        runs_by_result_revision: dict[str, EvalRunRecord] = {}
        for evidence in body.published_results:
            try:
                run = await store.load_run(evidence.run_id)
                stored = await store.load_result(evidence.run_id)
                current_run = await store.load_run(evidence.run_id)
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Eval result is too large.",
                ) from exc
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid memory experiment result identity.",
                ) from exc
            if run is None or current_run is None or stored is None:
                raise HTTPException(status_code=404, detail="Eval result not found.")
            if (
                current_run.status is not EvalRunStatus.COMPLETED
                or current_run.result is None
                or current_run.result.revision != evidence.result.revision
                or run != current_run
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Memory report eval run evidence changed during readback.",
                )
            if active_eval_registry.get(current_run.spec.target_key) is None:
                raise HTTPException(status_code=404, detail="Eval result not found.")
            if type(stored) is not CorpusExecutionResult:
                raise HTTPException(
                    status_code=409,
                    detail="Memory reports require fresh corpus execution results.",
                )
            if stored != evidence.result:
                raise HTTPException(
                    status_code=409,
                    detail=("Memory report evidence does not match the exact stored result."),
                )
            runs_by_result_revision[evidence.result.revision] = current_run
        variants = {item.variant_id: item for item in body.variants}
        for trial in body.trials:
            revision = trial.published_result_revision
            if revision is None:
                continue
            run = runs_by_result_revision[revision]
            variant = variants[trial.variant_id]
            invocation = run.spec.invocation
            if (
                invocation.execution_profile != variant.execution_profile_binding
                or invocation.execution_profile_snapshot != variant.execution_profile
            ):
                raise HTTPException(
                    status_code=409,
                    detail=("Memory report profile evidence does not match the exact eval run."),
                )
        try:
            return await asyncio.to_thread(build_memory_experiment_report, body)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422,
                detail="Invalid memory experiment report request.",
            ) from exc

    @bounded_memory_report_router.post(
        "/evals/memory-reports",
        response_model=MemoryExperimentReport,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def build_stored_memory_report(request: Request) -> Response:
        body = await _validated_private_json_body(
            request,
            MemoryExperimentReportRequest,
            invalid_detail="Invalid memory experiment report request.",
        )
        report = await _build_stored_memory_experiment_report(body)
        return await _model_json_response(report, MemoryExperimentReport)

    @bounded_memory_report_router.post(
        "/evals/memory-reports/report.html",
        response_class=Response,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def build_stored_memory_report_html(request: Request) -> Response:
        body = await _validated_private_json_body(
            request,
            MemoryExperimentReportRequest,
            invalid_detail="Invalid memory experiment report request.",
        )
        report = await _build_stored_memory_experiment_report(body)
        rendered = await asyncio.to_thread(render_memory_experiment_report_html, report)
        return Response(content=rendered, media_type="text/html")
