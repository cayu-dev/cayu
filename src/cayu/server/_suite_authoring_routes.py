"""Authored evaluation suite validation, publication and catalog HTTP routes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from cayu.evals.corpus import PrivateJudgeReferenceV1, StructuredModelJudgeAssertionSpec
from cayu.evals.execution import _candidate_judge_route_relation
from cayu.evals.scenario import EvalScenarioDocumentV2
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_DEFAULT_PAGE_SIZE,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_IDENTIFIER_CHARS,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalAuthoredSuiteCatalogPage,
    EvalAuthoredSuiteCatalogQuery,
    EvalAuthoredSuiteConflict,
    EvalAuthoredSuiteReferenceError,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
    authored_suite_scenario_cases,
)
from cayu.evals.suite_authoring import (
    EvalSuiteDocument,
    compile_eval_suite_authoring_draft,
    eval_suite_document_to_json,
    eval_suite_selection,
    validate_expected_eval_suite_revision,
)
from cayu.server._http_json import _model_json_response, _render_utf8
from cayu.server.contracts import (
    EVALS_ENDPOINT_RESPONSES,
    EvalSuiteAuthoringDiagnostic,
    EvalSuitePreviewRequest,
    EvalSuitePreviewResponse,
    EvalSuiteSaveRequest,
    EvalSuiteSaveResponse,
)

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


async def load_authored_suite(
    revision: str,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
) -> EvalSuiteDocument:
    """Load one visible suite revision for authoring reads and run launch."""
    if not eval_store.suite_authoring:
        raise HTTPException(
            status_code=409,
            detail="Durable authored-suite persistence is not available.",
        )
    try:
        suite = await eval_store.load_authored_suite(revision)
    except EvalStoreResultTooLarge as exc:
        raise HTTPException(
            status_code=413,
            detail="Authored eval suite exceeds the server byte limit.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc
    if suite is None or active_eval_registry.get(suite.target_key) is None:
        raise HTTPException(status_code=404, detail="Authored eval suite not found.")
    return suite


def register_suite_authoring_routes(
    bounded_evals_router: APIRouter,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    protected: Sequence[Depends],
) -> None:
    """Register suite authoring and catalog routes with the shared HTTP boundary."""

    async def _authored_suite_diagnostics(
        suite: EvalSuiteDocument,
    ) -> tuple[EvalSuiteAuthoringDiagnostic, ...]:
        diagnostics: list[EvalSuiteAuthoringDiagnostic] = []
        registration = active_eval_registry.registration(suite.target_key)
        if registration is None:
            diagnostics.append(
                EvalSuiteAuthoringDiagnostic(
                    code="target_unavailable",
                    message="The authored suite target is not currently published.",
                )
            )
        else:
            target = registration.target
            profiles = {
                profile.key: profile for profile in registration.catalog_entry.judge_profiles
            }
            judges = {judge.key: judge for judge in target.model_judges}
            for case in suite.cases:
                for assertion in case.assertions:
                    if type(assertion) is not StructuredModelJudgeAssertionSpec:
                        continue
                    judge = judges.get(assertion.judge_profile_key)
                    profile = profiles.get(assertion.judge_profile_key)
                    if judge is None or profile is None:
                        diagnostics.append(
                            EvalSuiteAuthoringDiagnostic(
                                code="judge_profile_unavailable",
                                case_id=case.id,
                                message="The selected judge profile is not currently published.",
                            )
                        )
                        continue
                    if profile.revision != assertion.judge_profile_revision:
                        diagnostics.append(
                            EvalSuiteAuthoringDiagnostic(
                                code="judge_profile_changed",
                                case_id=case.id,
                                message=(
                                    "The selected judge profile changed after this case "
                                    "was authored."
                                ),
                            )
                        )
                        continue
                    requested_evidence = {"final_output"}
                    if assertion.evidence.include_transcript:
                        requested_evidence.add("transcript")
                    if assertion.reference is not None:
                        requested_evidence.add(
                            "public_reference"
                            if assertion.reference.kind == "public_reference"
                            else "private_reference"
                        )
                    if not requested_evidence.issubset(profile.allowed_evidence):
                        diagnostics.append(
                            EvalSuiteAuthoringDiagnostic(
                                code="judge_evidence_not_allowed",
                                case_id=case.id,
                                message=(
                                    "The selected judge profile does not permit the requested "
                                    "evidence."
                                ),
                            )
                        )
                    reference = assertion.reference
                    if type(reference) is PrivateJudgeReferenceV1 and not any(
                        item.key == reference.key
                        and item.revision == reference.revision
                        and item.privacy_policy_key == reference.privacy_policy_key
                        and item.privacy_policy_revision == reference.privacy_policy_revision
                        for item in judge.private_references
                    ):
                        diagnostics.append(
                            EvalSuiteAuthoringDiagnostic(
                                code="judge_reference_unavailable",
                                case_id=case.id,
                                message="The exact private judge reference is unavailable.",
                            )
                        )
                    if (
                        _candidate_judge_route_relation(target, profile) == "same_model"
                        and profile.same_model_use != "allowed_and_labeled"
                    ):
                        diagnostics.append(
                            EvalSuiteAuthoringDiagnostic(
                                code="same_model_forbidden",
                                case_id=case.id,
                                message="The selected judge forbids this same-model route.",
                            )
                        )
            public_material = suite.model_dump(mode="json")
            try:
                redacted_material = await asyncio.to_thread(
                    target.app.redact_json,
                    public_material,
                )
            except Exception:
                redacted_material = None
            if redacted_material != public_material:
                diagnostics.append(
                    EvalSuiteAuthoringDiagnostic(
                        code="unsafe_public_material",
                        message=(
                            "The authored suite contains material rejected by the "
                            "application publication boundary."
                        ),
                    )
                )
        scenario_cases = authored_suite_scenario_cases(suite)
        if scenario_cases and not eval_store.scenarios:
            return (
                *diagnostics,
                *(
                    EvalSuiteAuthoringDiagnostic(
                        code="scenario_store_unavailable",
                        case_id=case.id,
                        message="Durable scenario persistence is not available.",
                    )
                    for case, _ in scenario_cases
                ),
            )
        scenario_by_revision: dict[str, EvalScenarioDocumentV2 | None] = {}
        if scenario_cases:
            load_limit = asyncio.Semaphore(16)

            async def load_scenario_for_preview(revision: str):
                async with load_limit:
                    try:
                        return await eval_store.load_scenario(revision)
                    except (EvalStoreResultTooLarge, TypeError, ValueError):
                        return None

            unique_revisions = tuple(
                sorted({reference.scenario_revision for _, reference in scenario_cases})
            )
            loaded = await asyncio.gather(
                *(load_scenario_for_preview(revision) for revision in unique_revisions)
            )
            scenario_by_revision = dict(zip(unique_revisions, loaded, strict=True))
        for case, reference in scenario_cases:
            scenario = scenario_by_revision.get(reference.scenario_revision)
            if scenario is None:
                diagnostics.append(
                    EvalSuiteAuthoringDiagnostic(
                        code="scenario_unavailable",
                        case_id=case.id,
                        message="The exact scenario revision is unavailable.",
                    )
                )
            elif scenario.id != reference.scenario_id:
                diagnostics.append(
                    EvalSuiteAuthoringDiagnostic(
                        code="scenario_id_mismatch",
                        case_id=case.id,
                        message="The scenario ID does not match its stored revision.",
                    )
                )
            elif scenario.target_key != suite.target_key:
                diagnostics.append(
                    EvalSuiteAuthoringDiagnostic(
                        code="scenario_target_mismatch",
                        case_id=case.id,
                        message="The scenario target does not match its authored suite.",
                    )
                )
            else:
                if case.source is not None and case.source != scenario.source:
                    diagnostics.append(
                        EvalSuiteAuthoringDiagnostic(
                            code="scenario_source_mismatch",
                            case_id=case.id,
                            message=("The case source does not match its scenario provenance."),
                        )
                    )
        return tuple(diagnostics)

    @bounded_evals_router.post(
        "/evals/suites/preview",
        response_model=EvalSuitePreviewResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def preview_eval_suite(
        body: EvalSuitePreviewRequest,
    ) -> EvalSuitePreviewResponse:
        try:
            suite = compile_eval_suite_authoring_draft(body.draft)
            selection = eval_suite_selection(suite)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Authored eval suite draft is invalid.",
            ) from exc
        diagnostics = await _authored_suite_diagnostics(suite)
        return EvalSuitePreviewResponse(
            suite=suite,
            full_selection=selection,
            ready=not diagnostics,
            diagnostics=diagnostics,
        )

    @bounded_evals_router.post(
        "/evals/suites",
        response_model=EvalSuiteSaveResponse,
        status_code=201,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def save_eval_suite(body: EvalSuiteSaveRequest) -> EvalSuiteSaveResponse:
        if not eval_store.suite_authoring:
            raise HTTPException(
                status_code=409,
                detail="Durable authored-suite persistence is not available.",
            )
        try:
            suite = validate_expected_eval_suite_revision(
                body.suite,
                body.expected_suite_revision,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Authored eval suite changed after the reviewed revision.",
            ) from exc
        diagnostics = await _authored_suite_diagnostics(suite)
        if diagnostics:
            raise HTTPException(
                status_code=409,
                detail="Authored eval suite is not ready to save.",
            )
        registration = active_eval_registry.registration(suite.target_key)
        assert registration is not None
        try:
            entry = await eval_store.save_authored_suite(
                suite,
                redact_json=registration.target.app.redact_json,
            )
        except EvalAuthoredSuiteConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="Authored eval suite revision conflicts with stored content.",
            ) from exc
        except EvalAuthoredSuiteReferenceError as exc:
            raise HTTPException(
                status_code=409,
                detail="Authored eval suite scenario references changed before save.",
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="Authored eval suite contains unsafe public data.",
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Authored eval suite exceeds the server byte limit.",
            ) from exc
        return EvalSuiteSaveResponse(
            entry=entry,
            suite=suite,
            full_selection=eval_suite_selection(suite),
        )

    @bounded_evals_router.get(
        "/evals/suites",
        response_model=EvalAuthoredSuiteCatalogPage,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_authored_suites(
        target_key: Annotated[
            str | None,
            Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS),
        ] = None,
        suite_id: Annotated[
            str | None,
            Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS),
        ] = None,
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
        if not eval_store.suite_authoring:
            raise HTTPException(
                status_code=409,
                detail="Durable authored-suite persistence is not available.",
            )
        selected_key = active_eval_registry.default_target_key if target_key is None else target_key
        eval_target = active_eval_registry.get(selected_key)
        if eval_target is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        try:
            return await eval_store.list_authored_suites(
                EvalAuthoredSuiteCatalogQuery(
                    target_key=eval_target.key,
                    suite_id=suite_id,
                    cursor=cursor,
                    limit=limit,
                    max_result_bytes=max_result_bytes,
                )
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Authored eval suite catalog exceeds the requested byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc

    @bounded_evals_router.get(
        "/evals/suites/{suite_revision}",
        response_model=EvalSuiteDocument,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_eval_authored_suite(suite_revision: str):
        suite = await load_authored_suite(
            suite_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        return await _model_json_response(suite, type(suite))

    @bounded_evals_router.get(
        "/evals/suites/{suite_revision}/download",
        response_class=Response,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def download_eval_authored_suite(suite_revision: str) -> Response:
        suite = await load_authored_suite(
            suite_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        suite_json = await asyncio.to_thread(
            _render_utf8,
            eval_suite_document_to_json,
            suite,
        )
        return Response(
            content=suite_json,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{suite.target_key}-'
                    f'{suite.revision[7:19]}.eval-suite.json"'
                )
            },
        )
