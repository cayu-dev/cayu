"""Promotion preview/export HTTP handlers and shared promotion error handling."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, NoReturn

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from cayu.budgets.pricing import PriceBook
from cayu.evals.corpus import EvalCaseSpec, EvalSuiteSpec
from cayu.evals.models import Trajectory
from cayu.evals.promotion import (
    PromotionCandidateV1,
    SessionPromotionError,
    build_promotion_candidate,
    corpus_from_promotion_candidate,
    export_promotion_corpus,
    score_promotion_candidate,
)
from cayu.evals.trajectory import (
    SessionTrajectoryError,
    SessionTrajectoryErrorCode,
    trajectory_from_session,
)
from cayu.server.config import EvaluationPromotionConfig
from cayu.server.contracts import (
    EVALUATION_PROMOTION_ENDPOINT_RESPONSES,
    EvaluationPromotionDraft,
    EvaluationPromotionExportRequest,
    EvaluationPromotionPreviewRequest,
    EvaluationPromotionPreviewResponse,
)
from cayu.sessions.terminal_evidence import TerminalSessionEvidenceErrorCode

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.applications import CayuApp


def _promotion_error_detail(
    code: str,
    message: str,
    *,
    reason: str | None = None,
) -> dict[str, str]:
    detail = {"code": code, "message": message}
    if reason is not None:
        detail["reason"] = reason
    return detail


def _raise_promotion_trajectory_error(exc: SessionTrajectoryError) -> NoReturn:
    terminal_code = exc.terminal_code
    reason = terminal_code.value if terminal_code is not None else exc.code.value
    if terminal_code is TerminalSessionEvidenceErrorCode.SESSION_NOT_FOUND:
        status_code = 404
        code = "session_not_found"
    elif terminal_code in {
        TerminalSessionEvidenceErrorCode.EVENT_LIMIT_EXCEEDED,
        TerminalSessionEvidenceErrorCode.TRANSCRIPT_LIMIT_EXCEEDED,
        TerminalSessionEvidenceErrorCode.RECORD_BYTES_EXCEEDED,
        TerminalSessionEvidenceErrorCode.TOTAL_BYTES_EXCEEDED,
        TerminalSessionEvidenceErrorCode.TRANSPORT_BYTES_EXCEEDED,
    } or exc.code in {
        SessionTrajectoryErrorCode.SESSION_LIMIT_EXCEEDED,
        SessionTrajectoryErrorCode.DEPTH_LIMIT_EXCEEDED,
    }:
        status_code = 413
        code = "evidence_limit_exceeded"
    else:
        status_code = 409
        code = "source_ineligible"
    raise HTTPException(
        status_code=status_code,
        detail=_promotion_error_detail(code, str(exc), reason=reason),
    ) from exc


def _require_safe_promotion_document(
    cayu_app: CayuApp,
    document: dict[str, Any],
    *,
    code: str,
    failure_subject: str,
) -> None:
    try:
        redacted_document = cayu_app.redact_json(document)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=_promotion_error_detail(
                code,
                f"{failure_subject} could not cross the application redaction boundary.",
            ),
        ) from exc
    if redacted_document != document:
        raise HTTPException(
            status_code=400,
            detail=_promotion_error_detail(
                code,
                f"{failure_subject} contains a workload secret.",
            ),
        )


def register_evaluation_promotion_routes(
    router: APIRouter,
    *,
    cayu_app: CayuApp,
    evaluation_promotion: EvaluationPromotionConfig,
    evaluation_promotion_pricing: PriceBook | None,
    resolve_public_session_id: Callable[[str], Awaitable[str]],
    protected: Sequence[Depends],
) -> None:
    """Register the enabled feature on its bounded router using shared access wiring."""

    async def _load_promotion_baseline(
        public_session_id: str,
    ) -> tuple[Trajectory, PromotionCandidateV1]:
        assert evaluation_promotion is not None
        private_session_id = await resolve_public_session_id(public_session_id)
        try:
            trajectory = await trajectory_from_session(cayu_app, private_session_id)
        except SessionTrajectoryError as exc:
            _raise_promotion_trajectory_error(exc)
        try:
            candidate = build_promotion_candidate(
                cayu_app,
                trajectory,
                target_key=evaluation_promotion.target_key,
                source_agent_name=evaluation_promotion.source_agent_name,
                application_release_id=evaluation_promotion.application_release_id,
                evidence_policy=evaluation_promotion.evidence_policy,
                pricing=evaluation_promotion_pricing,
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
        return trajectory, candidate

    def _promotion_candidate_from_draft(
        baseline: PromotionCandidateV1,
        draft: EvaluationPromotionDraft,
    ) -> PromotionCandidateV1:
        if draft.expected_baseline_revision != baseline.revision:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The promotion baseline changed; preview the session again.",
                ),
            )
        _require_safe_promotion_document(
            cayu_app,
            draft.model_dump(mode="json"),
            code="draft_rejected",
            failure_subject="The edited candidate",
        )
        try:
            suite = EvalSuiteSpec.create(
                id=draft.suite.id,
                name=draft.suite.name,
                description=draft.suite.description,
                trial_request=draft.suite.trial_request,
            )
            case = EvalCaseSpec.create(
                id=draft.case.id,
                suite_id=draft.case.suite_id,
                name=draft.case.name,
                description=draft.case.description,
                source=baseline.source.case_source(),
                input=draft.case.input,
                assertions=draft.case.assertions,
            )
            candidate = PromotionCandidateV1.create(
                target_key=baseline.target_key,
                source=baseline.source,
                evidence_policy=baseline.evidence_policy,
                pricing_profile=baseline.pricing_profile,
                evidence=baseline.evidence,
                suite=suite,
                case=case,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "draft_rejected",
                    "The edited candidate violates the promotion contract.",
                ),
            ) from exc
        try:
            # A successful dashboard preview is also the export gate. Validate
            # corpus-only invariants here so the UI never presents a current
            # preview that the unchanged export route must reject later.
            corpus_from_promotion_candidate(candidate)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "draft_rejected",
                    "The edited candidate is incompatible with the configured corpus limits or "
                    "pricing profile.",
                ),
            ) from exc
        return candidate

    def _promotion_server_fields_match(
        candidate: PromotionCandidateV1,
        baseline: PromotionCandidateV1,
    ) -> bool:
        return (
            candidate.target_key == baseline.target_key
            and candidate.source == baseline.source
            and candidate.evidence_policy == baseline.evidence_policy
            and candidate.pricing_profile == baseline.pricing_profile
            and candidate.evidence == baseline.evidence
            and candidate.warnings == baseline.warnings
        )

    @router.post(
        "/evals/promotion/sessions/{session_id}/preview",
        response_model=EvaluationPromotionPreviewResponse,
        responses=EVALUATION_PROMOTION_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def preview_evaluation_promotion(
        session_id: str,
        body: EvaluationPromotionPreviewRequest,
    ) -> EvaluationPromotionPreviewResponse:
        trajectory, baseline = await _load_promotion_baseline(session_id)
        candidate = (
            baseline
            if body.draft is None
            else _promotion_candidate_from_draft(baseline, body.draft)
        )
        try:
            captured_score = score_promotion_candidate(
                cayu_app,
                trajectory,
                candidate,
                target_key=evaluation_promotion.target_key,
                source_agent_name=evaluation_promotion.source_agent_name,
                application_release_id=evaluation_promotion.application_release_id,
                pricing=evaluation_promotion_pricing,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The captured evidence changed; preview the session again.",
                ),
            ) from exc
        return EvaluationPromotionPreviewResponse(
            baseline_revision=baseline.revision,
            candidate=candidate,
            captured_score=captured_score,
        )

    @router.post(
        "/evals/promotion/sessions/{session_id}/export",
        responses=EVALUATION_PROMOTION_ENDPOINT_RESPONSES,
        dependencies=protected,
        response_class=Response,
    )
    async def export_evaluation_promotion(
        session_id: str,
        body: EvaluationPromotionExportRequest,
    ) -> Response:
        if body.expected_candidate_revision != body.candidate.revision:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The candidate changed after preview; preview it again before export.",
                ),
            )
        _require_safe_promotion_document(
            cayu_app,
            body.candidate.model_dump(mode="json"),
            code="candidate_rejected",
            failure_subject="The candidate",
        )
        trajectory, baseline = await _load_promotion_baseline(session_id)
        if not _promotion_server_fields_match(body.candidate, baseline):
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The captured evidence or configured promotion identity changed.",
                ),
            )
        try:
            score_promotion_candidate(
                cayu_app,
                trajectory,
                body.candidate,
                target_key=evaluation_promotion.target_key,
                source_agent_name=evaluation_promotion.source_agent_name,
                application_release_id=evaluation_promotion.application_release_id,
                pricing=evaluation_promotion_pricing,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The candidate is no longer exportable; preview it again.",
                ),
            ) from exc
        try:
            corpus_bytes = export_promotion_corpus(body.candidate)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "candidate_rejected",
                    "The candidate cannot be exported as a portable corpus.",
                ),
            ) from exc
        return Response(
            content=corpus_bytes,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{evaluation_promotion.target_key}.eval.json"'
                )
            },
        )
