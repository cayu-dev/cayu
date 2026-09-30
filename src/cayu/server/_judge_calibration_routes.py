"""Fixed-evidence judge calibration preview, execution and stored reports."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from decimal import Decimal
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from cayu.evals.calibration import (
    EVAL_JUDGE_CALIBRATION_MAX_BYTES,
    EvalJudgeCalibrationDefinitionV1,
    EvalJudgeCalibrationReportV1,
    PreparedEvalJudgeCalibration,
    compile_eval_judge_calibration_draft,
    prepare_eval_judge_calibration,
    run_eval_judge_calibration_trial,
)
from cayu.evals.corpus import PrivateJudgeReferenceV1, _canonical_decimal_text
from cayu.evals.execution import _candidate_judge_route_relation
from cayu.evals.store import (
    EvalJudgeCalibrationConflict,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.server._http_json import _model_json_response
from cayu.server.contracts import (
    EVALS_ENDPOINT_RESPONSES,
    EvalJudgeCalibrationDiagnostic,
    EvalJudgeCalibrationPreviewRequest,
    EvalJudgeCalibrationPreviewResponse,
    EvalJudgeCalibrationRunRequest,
    EvalJudgeCalibrationRunResponse,
    EvalJudgeCalibrationWorkPreview,
)

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


def register_judge_calibration_routes(
    bounded_evals_router: APIRouter,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    protected: Sequence[Depends],
) -> None:
    """Register calibration routes with one lock set per router composition."""

    calibration_run_locks = tuple(asyncio.Lock() for _ in range(64))

    async def _preview_judge_calibration_definition(
        definition: EvalJudgeCalibrationDefinitionV1,
    ) -> tuple[
        EvalJudgeCalibrationPreviewResponse,
        PreparedEvalJudgeCalibration | None,
    ]:
        diagnostics: list[EvalJudgeCalibrationDiagnostic] = []
        registration = active_eval_registry.registration(definition.target_key)
        if registration is None:
            diagnostics.append(
                EvalJudgeCalibrationDiagnostic(
                    code="target_unavailable",
                    message="The calibration target is not currently published.",
                )
            )
            return (
                EvalJudgeCalibrationPreviewResponse(
                    definition=definition,
                    ready=False,
                    diagnostics=tuple(diagnostics),
                ),
                None,
            )
        target = registration.target
        judge = next(
            (
                item
                for item in target.model_judges
                if item.key == definition.assertion.judge_profile_key
            ),
            None,
        )
        if judge is None:
            diagnostics.append(
                EvalJudgeCalibrationDiagnostic(
                    code="judge_profile_unavailable",
                    message="The selected judge profile is not currently published.",
                )
            )
        else:
            current_profile = next(
                (
                    item
                    for item in registration.catalog_entry.judge_profiles
                    if item.key == judge.key
                ),
                None,
            )
            if current_profile is None:
                diagnostics.append(
                    EvalJudgeCalibrationDiagnostic(
                        code="judge_profile_unavailable",
                        message="The selected judge profile has no public identity.",
                    )
                )
            elif current_profile.revision != definition.assertion.judge_profile_revision:
                diagnostics.append(
                    EvalJudgeCalibrationDiagnostic(
                        code="judge_profile_changed",
                        message="The selected judge profile changed after the draft was authored.",
                    )
                )
            else:
                if (
                    _candidate_judge_route_relation(target, current_profile) == "same_model"
                    and current_profile.same_model_use != "allowed_and_labeled"
                ):
                    diagnostics.append(
                        EvalJudgeCalibrationDiagnostic(
                            code="same_model_forbidden",
                            message=(
                                "The selected judge profile forbids the current same-model route."
                            ),
                        )
                    )
                evidence = definition.assertion.evidence
                requested_evidence = {"final_output"}
                if evidence.include_transcript:
                    requested_evidence.add("transcript")
                reference = definition.assertion.reference
                if reference is not None:
                    requested_evidence.add(
                        "public_reference"
                        if reference.kind == "public_reference"
                        else "private_reference"
                    )
                if not requested_evidence.issubset(current_profile.allowed_evidence):
                    diagnostics.append(
                        EvalJudgeCalibrationDiagnostic(
                            code="evidence_not_allowed",
                            message=(
                                "The selected judge profile does not permit the requested "
                                "candidate or reference evidence."
                            ),
                        )
                    )
                if type(reference) is PrivateJudgeReferenceV1 and not any(
                    item.key == reference.key
                    and item.revision == reference.revision
                    and item.privacy_policy_key == reference.privacy_policy_key
                    and item.privacy_policy_revision == reference.privacy_policy_revision
                    for item in judge.private_references
                ):
                    diagnostics.append(
                        EvalJudgeCalibrationDiagnostic(
                            code="reference_unavailable",
                            message=(
                                "The exact private reference is unavailable under the "
                                "selected judge policy."
                            ),
                        )
                    )
        public_material = definition.model_dump(mode="json")
        try:
            redacted_material = await asyncio.to_thread(
                target.app.redact_json,
                public_material,
            )
        except Exception:
            redacted_material = None
        if redacted_material != public_material:
            diagnostics.append(
                EvalJudgeCalibrationDiagnostic(
                    code="unsafe_public_material",
                    message=(
                        "Calibration evidence or labels contain material rejected by the "
                        "application publication boundary."
                    ),
                )
            )
        if diagnostics:
            return (
                EvalJudgeCalibrationPreviewResponse(
                    definition=definition,
                    ready=False,
                    diagnostics=tuple(diagnostics),
                ),
                None,
            )
        try:
            prepared = await asyncio.to_thread(
                prepare_eval_judge_calibration,
                definition,
                target,
            )
        except (TypeError, ValueError):
            return (
                EvalJudgeCalibrationPreviewResponse(
                    definition=definition,
                    ready=False,
                    diagnostics=(
                        EvalJudgeCalibrationDiagnostic(
                            code="invalid_contract",
                            message=(
                                "The calibration contract is incompatible with current "
                                "trusted judge authority."
                            ),
                        ),
                    ),
                ),
                None,
            )
        profile = prepared.judge_profile
        total_cost = (
            None
            if profile.max_estimated_cost is None
            else _canonical_decimal_text(
                format(Decimal(profile.max_estimated_cost) * definition.trials, "f"),
                "calibration max estimated cost",
            )
        )
        work = EvalJudgeCalibrationWorkPreview(
            judge_calls=definition.trials,
            max_input_tokens=profile.max_input_tokens * definition.trials,
            max_output_tokens=profile.max_output_tokens * definition.trials,
            max_total_tokens=profile.max_total_tokens * definition.trials,
            max_estimated_cost=total_cost,
            cost_currency=(profile.cost_currency if total_cost is not None else None),
        )
        return (
            EvalJudgeCalibrationPreviewResponse(
                definition=definition,
                ready=True,
                judge_profile=profile,
                candidate_route_relation=prepared.candidate_route_relation,
                work=work,
            ),
            prepared,
        )

    @bounded_evals_router.post(
        "/evals/judge-calibrations/preview",
        response_model=EvalJudgeCalibrationPreviewResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def preview_eval_judge_calibration(
        body: EvalJudgeCalibrationPreviewRequest,
    ) -> EvalJudgeCalibrationPreviewResponse:
        try:
            definition = compile_eval_judge_calibration_draft(body.draft)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Judge calibration draft is invalid.",
            ) from exc
        preview, _ = await _preview_judge_calibration_definition(definition)
        return preview

    async def _execute_eval_judge_calibration(
        body: EvalJudgeCalibrationRunRequest,
    ) -> EvalJudgeCalibrationRunResponse:
        try:
            existing = await eval_store.load_judge_calibration_by_run_id(
                body.run_id,
                max_bytes=EVAL_JUDGE_CALIBRATION_MAX_BYTES,
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Judge calibration exceeds the server byte limit.",
            ) from exc
        if existing is not None:
            if existing.definition != body.definition:
                raise HTTPException(
                    status_code=409,
                    detail="Judge calibration run ID is bound to different reviewed input.",
                )
            return EvalJudgeCalibrationRunResponse(report=existing)
        preview, prepared = await _preview_judge_calibration_definition(body.definition)
        if not preview.ready or prepared is None:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "calibration_not_ready",
                    "diagnostics": [item.model_dump(mode="json") for item in preview.diagnostics],
                },
            )
        trials = []
        for sequence in range(1, body.definition.trials + 1):
            trials.append(
                await run_eval_judge_calibration_trial(
                    prepared,
                    sequence=sequence,
                )
            )
        report = EvalJudgeCalibrationReportV1.create(
            run_id=body.run_id,
            prepared=prepared,
            trials=tuple(trials),
        )
        registration = active_eval_registry.registration(body.definition.target_key)
        if registration is None:
            raise HTTPException(
                status_code=409,
                detail="The calibration target became unavailable before persistence.",
            )
        try:
            stored = await eval_store.save_judge_calibration(
                report,
                redact_json=registration.target.app.redact_json,
            )
        except EvalJudgeCalibrationConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="Judge calibration identity conflicts with stored content.",
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="Judge calibration contains unsafe public data.",
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Judge calibration exceeds the server byte limit.",
            ) from exc
        return EvalJudgeCalibrationRunResponse(report=stored)

    @bounded_evals_router.post(
        "/evals/judge-calibrations",
        response_model=EvalJudgeCalibrationRunResponse,
        status_code=201,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def run_eval_judge_calibration(
        body: EvalJudgeCalibrationRunRequest,
    ) -> EvalJudgeCalibrationRunResponse:
        if not eval_store.judge_calibrations:
            raise HTTPException(
                status_code=409,
                detail="Durable judge calibration persistence is not available.",
            )
        if body.definition.revision != body.expected_definition_revision:
            raise HTTPException(
                status_code=409,
                detail="Judge calibration changed after the reviewed preview.",
            )
        digest = hashlib.sha256(body.run_id.encode("ascii")).digest()
        lock = calibration_run_locks[int.from_bytes(digest[:2], "big") % len(calibration_run_locks)]
        async with lock:
            return await _execute_eval_judge_calibration(body)

    @bounded_evals_router.get(
        "/evals/judge-calibrations/{calibration_revision}",
        response_model=EvalJudgeCalibrationReportV1,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_eval_judge_calibration(calibration_revision: str):
        if not eval_store.judge_calibrations:
            raise HTTPException(
                status_code=409,
                detail="Durable judge calibration persistence is not available.",
            )
        try:
            report = await eval_store.load_judge_calibration(
                calibration_revision,
                max_bytes=EVAL_JUDGE_CALIBRATION_MAX_BYTES,
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Judge calibration exceeds the server byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc
        if report is None:
            raise HTTPException(status_code=404, detail="Judge calibration not found.")
        return await _model_json_response(report, EvalJudgeCalibrationReportV1)
