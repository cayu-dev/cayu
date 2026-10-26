"""Eval target catalog and captured-evaluation preview, save, export and launch routes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Annotated, TypeAlias, cast

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response

from cayu.evals.models import Trajectory
from cayu.evals.promotion import (
    CapturedEvaluationCandidateV1,
    SessionPromotionError,
    build_promotion_candidate,
    corpus_from_captured_evaluation_candidate,
    corpus_from_promotion_candidate,
    export_captured_evaluation_corpus,
    runnable_promotion_candidate,
    score_promotion_candidate,
)
from cayu.evals.results import CapturedEvaluationResultV1, EvalResultTargetIdentityV1
from cayu.evals.scenario_capture import capture_eval_scenario_from_session
from cayu.evals.store import (
    EvalCorpusConflict,
    EvalResultConflict,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.server._captured_evaluation_source import CapturedEvaluationSource
from cayu.server._eval_run_admission import (
    admit_eval_run,
    bind_eval_admission_request,
    eval_run_invocation,
    prepare_eval_run,
    replay_eval_run,
)
from cayu.server._evaluation_promotion_routes import (
    _promotion_error_detail,
)
from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
    EVALS_ENDPOINT_RESPONSES,
    CapturedEvaluationExportRequest,
    CapturedEvaluationLaunchRequest,
    CapturedEvaluationLaunchResponse,
    CapturedEvaluationPreviewRequest,
    CapturedEvaluationPreviewResponse,
    CapturedEvaluationSaveRequest,
    CapturedEvaluationSaveResponse,
    EvalTargetCatalogResponse,
)
from cayu.server.evals_registry import EvalTargetRegistration, EvalTargetRegistry

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.applications import CayuApp
    from cayu.evals.store import EvalStore

CapturedCandidateValidator: TypeAlias = Callable[
    [str, CapturedEvaluationCandidateV1, str],
    Awaitable[tuple[Trajectory, EvalTargetRegistration, CapturedEvaluationCandidateV1]],
]


def register_captured_evaluation_routes(
    bounded_evals_router: APIRouter,
    bounded_captured_evaluation_router: APIRouter,
    *,
    cayu_app: CayuApp,
    eval_registry: EvalTargetRegistry,
    captured_eval_store: EvalStore | None,
    resolve_public_session_id: Callable[[str], Awaitable[str]],
    protected: Sequence[Depends],
) -> CapturedEvaluationSource:
    """Register captured routes and share their current-candidate validator with launch."""

    @bounded_evals_router.get(
        "/evals/targets",
        response_model=EvalTargetCatalogResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_targets() -> EvalTargetCatalogResponse:
        return await eval_registry.resolved_catalog()

    source = CapturedEvaluationSource(cayu_app, eval_registry, resolve_public_session_id)

    @bounded_captured_evaluation_router.post(
        "/evals/sessions/{session_id}/evaluation/save",
        response_model=CapturedEvaluationSaveResponse,
        status_code=201,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def save_captured_evaluation(
        session_id: str,
        body: CapturedEvaluationSaveRequest,
    ) -> CapturedEvaluationSaveResponse:
        if captured_eval_store is None or not captured_eval_store.captured_results:
            raise HTTPException(
                status_code=409,
                detail="Durable captured-result persistence is not available.",
            )
        trajectory, registration, _ = await source.require_current(
            session_id,
            body.candidate,
            body.expected_candidate_revision,
        )
        try:
            score = source.score(
                trajectory,
                registration,
                body.candidate,
            )
            corpus = corpus_from_captured_evaluation_candidate(body.candidate)
            result = CapturedEvaluationResultV1.create(
                corpus=corpus,
                target=EvalResultTargetIdentityV1(
                    target_key=body.candidate.target_key,
                    application_release_id=body.candidate.source.application_release_id,
                    app_manifest_schema_version=(body.candidate.source.app_manifest_schema_version),
                    app_manifest_fingerprint=(body.candidate.source.app_manifest_fingerprint),
                ),
                score=score,
            )
            record = await captured_eval_store.save_captured_result(
                corpus,
                result,
                redact_json=cayu_app.redact_json,
            )
        except (EvalCorpusConflict, EvalResultConflict) as exc:
            raise HTTPException(
                status_code=409,
                detail="The immutable captured evaluation conflicts with stored content.",
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="The captured evaluation contains unsafe public data.",
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="The captured evaluation exceeds the server byte limit.",
            ) from exc
        return CapturedEvaluationSaveResponse(record=record, result=result)

    @bounded_captured_evaluation_router.post(
        "/evals/sessions/{session_id}/evaluation/preview",
        response_model=CapturedEvaluationPreviewResponse,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def preview_captured_evaluation(
        session_id: str,
        body: CapturedEvaluationPreviewRequest,
    ) -> CapturedEvaluationPreviewResponse:
        trajectory, registration, baseline = await source.load(session_id)
        candidate = baseline if body.draft is None else source.from_draft(baseline, body.draft)
        captured_score = source.score(
            trajectory,
            registration,
            candidate,
        )
        source_session = trajectory.session
        if source_session is None:
            raise HTTPException(status_code=409, detail="Captured session evidence is absent.")
        scenario_conversion = await capture_eval_scenario_from_session(
            cayu_app,
            source_session.id,
            target_key=registration.target.key,
            source_agent_name=registration.target.request_base.agent_name,
            source=baseline.source.case_source(),
            name=candidate.case.name,
            description=candidate.case.description,
        )
        return CapturedEvaluationPreviewResponse(
            baseline_revision=baseline.revision,
            candidate=candidate,
            captured_score=captured_score,
            runnable_conversion=source.runnable_conversion(trajectory, registration),
            scenario_conversion=scenario_conversion,
        )

    @bounded_captured_evaluation_router.post(
        "/evals/sessions/{session_id}/evaluation/export",
        response_class=Response,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def export_captured_evaluation(
        session_id: str,
        body: CapturedEvaluationExportRequest,
    ) -> Response:
        trajectory, registration, _ = await source.require_current(
            session_id,
            body.candidate,
            body.expected_candidate_revision,
        )
        source.score(
            trajectory,
            registration,
            body.candidate,
        )
        content = export_captured_evaluation_corpus(body.candidate)
        return Response(
            content=content,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{body.candidate.target_key}-captured.eval.json"'
                )
            },
        )

    return source


def register_captured_evaluation_launch_routes(
    bounded_captured_evaluation_router: APIRouter,
    *,
    cayu_app: CayuApp,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    require_current_captured_candidate: CapturedCandidateValidator,
    protected: Sequence[Depends],
    optional_auth_context: Depends,
) -> None:
    """Register durable launch with the shared candidate and admission boundaries."""

    # Preserve FastAPI's shared dependency object as the typed auth-context default.
    launch_auth_context = cast("AuthContext | None", optional_auth_context)

    @bounded_captured_evaluation_router.post(
        "/evals/sessions/{session_id}/evaluation/launch",
        response_model=CapturedEvaluationLaunchResponse,
        status_code=202,
        responses=CAPTURED_EVALUATION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def launch_captured_evaluation(
        session_id: str,
        body: CapturedEvaluationLaunchRequest,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=512),
        ],
        auth_context: AuthContext | None = launch_auth_context,
    ) -> CapturedEvaluationLaunchResponse:
        if not eval_store.captured_results:
            raise HTTPException(
                status_code=409,
                detail="Durable captured-result persistence is not available.",
            )
        trajectory, registration, _ = await require_current_captured_candidate(
            session_id,
            body.candidate,
            body.expected_candidate_revision,
        )
        target = registration.target
        try:
            runnable_baseline = build_promotion_candidate(
                cayu_app,
                trajectory,
                target_key=target.key,
                source_agent_name=target.request_base.agent_name,
                application_release_id=target.application_release_id,
                evidence_policy=target.evidence_policy,
                pricing=target.price_book,
                project_root=registration.manifest_project_root,
            )
        except SessionPromotionError as exc:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "source_ineligible",
                    str(exc),
                    reason=exc.code.value,
                ),
            ) from exc
        try:
            runnable_candidate = runnable_promotion_candidate(
                body.candidate,
                runnable_baseline,
                trial_request=body.trial_request,
            )
            score = score_promotion_candidate(
                cayu_app,
                trajectory,
                runnable_candidate,
                target_key=target.key,
                source_agent_name=target.request_base.agent_name,
                application_release_id=target.application_release_id,
                pricing=target.price_book,
                project_root=registration.manifest_project_root,
            )
            corpus = corpus_from_promotion_candidate(runnable_candidate)
            result = CapturedEvaluationResultV1.create(
                corpus=corpus,
                target=EvalResultTargetIdentityV1(
                    target_key=runnable_candidate.target_key,
                    application_release_id=(runnable_candidate.source.application_release_id),
                    app_manifest_schema_version=(
                        runnable_candidate.source.app_manifest_schema_version
                    ),
                    app_manifest_fingerprint=(runnable_candidate.source.app_manifest_fingerprint),
                ),
                score=score,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "candidate_rejected",
                    "The reviewed evaluation cannot be converted to runnable work.",
                ),
            ) from exc
        invocation = eval_run_invocation(
            auth_context,
            max_steps=body.max_steps,
            limits=body.limits,
            cost_budget=body.cost_budget,
        )
        invocation = bind_eval_admission_request(
            invocation,
            kind="captured",
            target_key=target.key,
            resource_identity={"session_id": session_id},
            body=body,
        )
        admission_request_revision = invocation.admission_request_revision
        if admission_request_revision is None:
            raise RuntimeError("Captured eval launch lost its admission request revision.")
        replayed = await replay_eval_run(
            eval_store=eval_store,
            target_key=target.key,
            idempotency_key=idempotency_key,
            admission_request_revision=admission_request_revision,
        )
        if replayed is not None:
            record = await eval_store.load_result_record(result.revision)
            if record is None:
                raise RuntimeError("Replayed captured eval result is unavailable.")
            return CapturedEvaluationLaunchResponse(
                captured=CapturedEvaluationSaveResponse(record=record, result=result),
                run=replayed,
            )
        eval_target, compiled, invocation = await prepare_eval_run(
            active_eval_registry=active_eval_registry,
            corpus=corpus,
            suite_id=runnable_candidate.suite.id,
            max_concurrency=body.max_concurrency,
            invocation=invocation,
            expected_execution_profile_revision=(body.expected_execution_profile_revision),
        )
        try:
            record = await eval_store.save_captured_result(
                corpus,
                result,
                redact_json=target.app.redact_json,
            )
        except (EvalCorpusConflict, EvalResultConflict) as exc:
            raise HTTPException(
                status_code=409,
                detail="The immutable runnable evaluation conflicts with stored content.",
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="The runnable evaluation contains unsafe public data.",
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="The runnable evaluation exceeds the server byte limit.",
            ) from exc
        run = await admit_eval_run(
            eval_store=eval_store,
            corpus=corpus,
            max_concurrency=body.max_concurrency,
            invocation=invocation,
            idempotency_key=idempotency_key,
            eval_target=eval_target,
            compiled=compiled,
        )
        return CapturedEvaluationLaunchResponse(
            captured=CapturedEvaluationSaveResponse(record=record, result=result),
            run=run,
        )
