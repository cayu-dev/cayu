"""Server-resolved capture evidence, draft rebuilding and current-candidate checks."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from fastapi import HTTPException

from cayu.evals.corpus import EvalCaseSpec, EvalSuiteSpec
from cayu.evals.models import Trajectory
from cayu.evals.promotion import (
    CapturedEvaluationCandidateV1,
    CapturedRunScoreV1,
    SessionPromotionError,
    build_captured_evaluation_candidate,
    build_promotion_candidate,
    corpus_from_captured_evaluation_candidate,
    score_captured_evaluation_candidate,
)
from cayu.evals.trajectory import SessionTrajectoryError, trajectory_from_session
from cayu.server._evaluation_promotion_routes import (
    _promotion_error_detail,
    _raise_promotion_trajectory_error,
    _require_safe_promotion_document,
)
from cayu.server.contracts import (
    CapturedEvaluationConversion,
    CapturedEvaluationDraft,
)
from cayu.server.evals_registry import EvalTargetRegistration, EvalTargetRegistry

if TYPE_CHECKING:
    from cayu.applications import CayuApp


class CapturedEvaluationSource:
    """One owner for server-resolved capture evidence and current-candidate validation."""

    def __init__(
        self,
        app: CayuApp,
        registry: EvalTargetRegistry,
        resolve_session: Callable[[str], Awaitable[str]],
    ):
        self.app = app
        self.registry = registry
        self.resolve_session = resolve_session

    async def load(
        self,
        public_session_id: str,
    ) -> tuple[Trajectory, EvalTargetRegistration, CapturedEvaluationCandidateV1]:
        private_session_id = await self.resolve_session(public_session_id)
        try:
            trajectory = await trajectory_from_session(self.app, private_session_id)
        except SessionTrajectoryError as exc:
            _raise_promotion_trajectory_error(exc)
        session = trajectory.session
        if session is None:
            raise HTTPException(status_code=409, detail="Captured session evidence is absent.")
        registration = self.registry.registration_for_agent(session.agent_name)
        if registration is None:
            raise HTTPException(
                status_code=409,
                detail="The session agent has no unambiguous published eval target.",
            )
        target = registration.target
        try:
            candidate = build_captured_evaluation_candidate(
                self.app,
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
        return trajectory, registration, candidate

    def from_draft(
        self,
        baseline: CapturedEvaluationCandidateV1,
        draft: CapturedEvaluationDraft,
    ) -> CapturedEvaluationCandidateV1:
        if draft.expected_baseline_revision != baseline.revision:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The captured evidence changed; preview the session again.",
                ),
            )
        _require_safe_promotion_document(
            self.app,
            draft.model_dump(mode="json"),
            code="draft_rejected",
            failure_subject="The edited captured evaluation",
        )
        try:
            suite = EvalSuiteSpec.create(
                id=draft.suite.id,
                name=draft.suite.name,
                description=draft.suite.description,
            )
            case = EvalCaseSpec.create(
                id=draft.case.id,
                suite_id=draft.case.suite_id,
                name=draft.case.name,
                description=draft.case.description,
                source=baseline.source.case_source(),
                input=None,
                assertions=draft.case.assertions,
            )
            candidate = CapturedEvaluationCandidateV1.create(
                target_key=baseline.target_key,
                source=baseline.source,
                evidence_policy=baseline.evidence_policy,
                pricing_profile=baseline.pricing_profile,
                evidence=baseline.evidence,
                suite=suite,
                case=case,
            )
            corpus_from_captured_evaluation_candidate(candidate)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "draft_rejected",
                    "The edited captured evaluation violates its portable contract.",
                ),
            ) from exc
        return candidate

    def server_fields_match(
        self,
        candidate: CapturedEvaluationCandidateV1,
        baseline: CapturedEvaluationCandidateV1,
    ) -> bool:
        return (
            candidate.target_key == baseline.target_key
            and candidate.source == baseline.source
            and candidate.evidence_policy == baseline.evidence_policy
            and candidate.pricing_profile == baseline.pricing_profile
            and candidate.evidence == baseline.evidence
            and candidate.warnings == baseline.warnings
        )

    def runnable_conversion(
        self,
        trajectory: Trajectory,
        registration: EvalTargetRegistration,
    ) -> CapturedEvaluationConversion:
        target = registration.target
        try:
            build_promotion_candidate(
                self.app,
                trajectory,
                target_key=target.key,
                source_agent_name=target.request_base.agent_name,
                application_release_id=target.application_release_id,
                evidence_policy=target.evidence_policy,
                pricing=target.price_book,
                project_root=registration.manifest_project_root,
            )
        except SessionPromotionError as exc:
            return CapturedEvaluationConversion(
                available=False,
                reason_code=exc.code.value,
            )
        except (TypeError, ValueError):
            return CapturedEvaluationConversion(
                available=False,
                reason_code="conversion_contract_unavailable",
            )
        return CapturedEvaluationConversion(available=True)

    async def require_current(
        self,
        session_id: str,
        candidate: CapturedEvaluationCandidateV1,
        expected_revision: str,
    ) -> tuple[Trajectory, EvalTargetRegistration, CapturedEvaluationCandidateV1]:
        if expected_revision != candidate.revision:
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The evaluation changed after preview; preview it again.",
                ),
            )
        _require_safe_promotion_document(
            self.app,
            candidate.model_dump(mode="json"),
            code="candidate_rejected",
            failure_subject="The captured evaluation",
        )
        trajectory, registration, baseline = await self.load(session_id)
        if not self.server_fields_match(candidate, baseline):
            raise HTTPException(
                status_code=409,
                detail=_promotion_error_detail(
                    "preview_stale",
                    "The captured evidence or target identity changed.",
                ),
            )
        return trajectory, registration, baseline

    def score(
        self,
        trajectory: Trajectory,
        registration: EvalTargetRegistration,
        candidate: CapturedEvaluationCandidateV1,
    ) -> CapturedRunScoreV1:
        """Revalidate one current candidate through the side-effect-free scorer."""

        target = registration.target
        try:
            return score_captured_evaluation_candidate(
                self.app,
                trajectory,
                candidate,
                target_key=target.key,
                source_agent_name=target.request_base.agent_name,
                application_release_id=target.application_release_id,
                pricing=target.price_book,
                project_root=registration.manifest_project_root,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail=_promotion_error_detail(
                    "candidate_rejected",
                    "The captured evaluation cannot be scored from its retained evidence.",
                ),
            ) from exc
