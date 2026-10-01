"""Authored evaluation suite launch preview, preparation and admission HTTP routes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, Literal, cast

from fastapi import APIRouter, Header, HTTPException

from cayu.evals.corpus import (
    ArtifactAssertionSpec,
    ToolArgumentsContainAssertionSpec,
    ToolResultContainsAssertionSpec,
    eval_suite_trial_policy,
    pricing_profile_identity,
)
from cayu.evals.execution import compile_corpus_suite
from cayu.evals.execution_profiles import EvalExecutionProfileV1
from cayu.evals.scenario import EvalScenarioDocumentV2
from cayu.evals.scenario_preflight import ScenarioLaunchBindingV2, preflight_eval_scenario
from cayu.evals.store import (
    EvalCorpusConflict,
    EvalRunCostBudget,
    EvalRunRecord,
    EvalScenarioArtifactReference,
    EvalScenarioRunInvocation,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.evals.suite_authoring import (
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocument,
    eval_suite_selection,
)
from cayu.evals.suite_execution import (
    authored_suite_launch_settings,
    corpus_for_authored_scenario_case,
    corpus_for_authored_simple_selection,
)
from cayu.evals.suite_preflight import (
    EvalCandidateLaunchExposure,
    allocate_authored_suite_launch_concurrency,
    compile_authored_suite_run_exposure,
)
from cayu.server._eval_run_admission import (
    admit_eval_run,
    bind_eval_admission_request,
    eval_idempotency_digest,
    eval_run_invocation,
    prepare_eval_run,
    replay_eval_run,
)
from cayu.server._suite_authoring_routes import load_authored_suite
from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    EVALS_ENDPOINT_RESPONSES,
    EvalAuthoredSuiteAdmittedRun,
    EvalAuthoredSuiteLaunchDiagnostic,
    EvalAuthoredSuiteLaunchPlanItem,
    EvalAuthoredSuiteRunLaunchRequest,
    EvalAuthoredSuiteRunLaunchResponse,
    EvalAuthoredSuiteRunPreviewResponse,
    EvalAuthoredSuiteRunSelectionRequest,
)
from cayu.server.evals_registry import target_for_eval_invocation

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


def _narrow_eval_cost_budget(
    current: EvalRunCostBudget | None,
    requested: EvalRunCostBudget | None,
) -> EvalRunCostBudget | None:
    if current is None:
        return requested
    if requested is None:
        return current
    if current.currency != requested.currency:
        raise ValueError("Eval cost-budget contractions must use one currency.")
    return EvalRunCostBudget(
        max_estimated_cost=min(
            current.max_estimated_cost,
            requested.max_estimated_cost,
        ),
        currency=current.currency,
    )


async def _preview_authored_suite_launch(
    suite: EvalSuiteDocument,
    request: EvalAuthoredSuiteRunSelectionRequest,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
) -> tuple[
    EvalAuthoredSuiteRunPreviewResponse,
    dict[str, tuple[EvalScenarioDocumentV2, ScenarioLaunchBindingV2]],
]:
    try:
        selection = eval_suite_selection(suite, request.case_ids)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Authored eval suite selection is invalid.",
        ) from exc
    selected_ids = {item.id for item in selection.cases}
    selected_cases = tuple(case for case in suite.cases if case.id in selected_ids)
    simple_cases = tuple(
        case for case in selected_cases if type(case.stimulus) is EvalSimpleInputStimulusV1
    )
    scenario_cases = tuple(
        case for case in selected_cases if type(case.stimulus) is EvalScenarioStimulusV1
    )
    launches: list[EvalAuthoredSuiteLaunchPlanItem] = []
    if simple_cases:
        launches.append(
            EvalAuthoredSuiteLaunchPlanItem(
                kind="simple_input",
                case_ids=tuple(case.id for case in simple_cases),
            )
        )
    launches.extend(
        EvalAuthoredSuiteLaunchPlanItem(
            kind="scenario",
            case_ids=(case.id,),
            scenario_revision=case.stimulus.scenario_revision,
        )
        for case in scenario_cases
        if type(case.stimulus) is EvalScenarioStimulusV1
    )
    diagnostics: list[EvalAuthoredSuiteLaunchDiagnostic] = []
    if not eval_store.trial_checkpointing:
        diagnostics.append(
            EvalAuthoredSuiteLaunchDiagnostic(
                code="trial_checkpointing_unavailable",
                message=(
                    "Restart-safe terminal trial checkpointing is not available in this deployment."
                ),
            )
        )
    trial_policy = eval_suite_trial_policy(suite.suite)
    concurrency_allocations = allocate_authored_suite_launch_concurrency(
        trial_policy,
        len(launches),
    )
    concurrency_by_case = {
        case_id: allocation
        for launch, allocation in zip(
            launches,
            concurrency_allocations,
            strict=True,
        )
        for case_id in launch.case_ids
    }
    execution_profiles_by_case: dict[str, EvalExecutionProfileV1] = {}
    registration = active_eval_registry.registration(suite.target_key)
    if registration is None:
        raise HTTPException(status_code=404, detail="Authored eval suite not found.")
    if (
        trial_policy.trial_count > registration.execution_profile_policy.max_trials
        or max(item.max_concurrency for item in concurrency_allocations)
        > registration.execution_profile_policy.max_concurrency
    ):
        diagnostics.append(
            EvalAuthoredSuiteLaunchDiagnostic(
                code="trial_policy_exceeds_execution_profile",
                message=(
                    "The suite trial count or concurrency exceeds the current "
                    "server-published execution profile. Select an isolated profile "
                    "with sufficient ceilings or reduce the policy."
                ),
            )
        )
    execution_target = registration.execution_target()
    for case in selected_cases:
        if (
            any(type(assertion) is ToolResultContainsAssertionSpec for assertion in case.assertions)
            and not execution_target.evidence_policy.include_tool_results
        ):
            diagnostics.append(
                EvalAuthoredSuiteLaunchDiagnostic(
                    code="tool_result_evidence_unavailable",
                    case_id=case.id,
                    message=(
                        "This case requires retained public-safe tool results, but the "
                        "selected target does not publish result evidence. Choose a target "
                        "profile with result retention or remove the result assertion."
                    ),
                )
            )
        if (
            any(
                type(assertion) is ToolArgumentsContainAssertionSpec
                for assertion in case.assertions
            )
            and not execution_target.evidence_policy.include_tool_arguments
        ):
            diagnostics.append(
                EvalAuthoredSuiteLaunchDiagnostic(
                    code="tool_argument_evidence_unavailable",
                    case_id=case.id,
                    message=(
                        "This case requires public tool arguments, but the selected target "
                        "does not publish argument evidence. Choose a compatible target "
                        "profile or remove the argument assertion."
                    ),
                )
            )
        if (
            any(
                type(assertion) is ArtifactAssertionSpec and assertion.text_contains is not None
                for assertion in case.assertions
            )
            and not execution_target.evidence_policy.include_artifact_text
        ):
            diagnostics.append(
                EvalAuthoredSuiteLaunchDiagnostic(
                    code="artifact_text_evidence_unavailable",
                    case_id=case.id,
                    message=(
                        "This case requires retained public-safe artifact text, but the "
                        "selected target does not publish artifact text evidence. Choose "
                        "a compatible target profile or remove the text expectation."
                    ),
                )
            )
    if simple_cases and not diagnostics:
        try:
            simple_invocation = eval_run_invocation(
                None,
                max_steps=None,
                limits=None,
                cost_budget=request.cost_budget,
                authored_suite_revision=suite.revision,
                authored_suite_selection_revision=selection.revision,
            )
            effective_simple_target = target_for_eval_invocation(
                execution_target,
                simple_invocation,
            )
            simple_selection = eval_suite_selection(
                suite,
                tuple(case.id for case in simple_cases),
            )
            corpus = await asyncio.to_thread(
                corpus_for_authored_simple_selection,
                suite,
                simple_selection,
                effective_simple_target,
                project_root=registration.manifest_project_root,
            )
            await asyncio.to_thread(
                compile_corpus_suite,
                corpus,
                effective_simple_target,
                suite.suite.id,
            )
        except (TypeError, ValueError):
            diagnostics.append(
                EvalAuthoredSuiteLaunchDiagnostic(
                    code="simple_launch_not_ready",
                    message=(
                        "The selected simple cases are incompatible with the current "
                        "target, evidence policy, pricing, or execution limits."
                    ),
                )
            )
        else:
            try:
                prepared_profile = await active_eval_registry.prepare_execution_profile(
                    suite.target_key,
                    effective_target=effective_simple_target,
                )
            except Exception:
                diagnostics.append(
                    EvalAuthoredSuiteLaunchDiagnostic(
                        code="execution_profile_unavailable",
                        message=(
                            "The selected simple cases have no currently executable "
                            "server-published profile."
                        ),
                    )
                )
            else:
                if (
                    trial_policy.trial_count > prepared_profile.snapshot.ceilings.max_trials
                    or concurrency_by_case[simple_cases[0].id].max_concurrency
                    > prepared_profile.snapshot.ceilings.max_concurrency
                ):
                    diagnostics.append(
                        EvalAuthoredSuiteLaunchDiagnostic(
                            code="trial_policy_exceeds_execution_profile",
                            message=(
                                "The suite trial count or concurrency exceeds the current "
                                "server-published execution profile. Select an isolated "
                                "profile with sufficient ceilings or reduce the policy."
                            ),
                        )
                    )
                else:
                    execution_profiles_by_case.update(
                        (case.id, prepared_profile.snapshot) for case in simple_cases
                    )
                    launches[0] = launches[0].model_copy(
                        update={"execution_profile_revision": (prepared_profile.snapshot.revision)}
                    )
    prepared_scenarios: dict[
        str,
        tuple[EvalScenarioDocumentV2, ScenarioLaunchBindingV2],
    ] = {}
    if scenario_cases and not eval_store.scenario_execution:
        diagnostics.append(
            EvalAuthoredSuiteLaunchDiagnostic(
                code="scenario_execution_unavailable",
                message="Durable scenario execution is not available in this deployment.",
            )
        )
    elif scenario_cases and not diagnostics:
        settings = authored_suite_launch_settings(suite)
        preflight_limit = asyncio.Semaphore(16)

        async def prepare_scenario(case):
            stimulus = case.stimulus
            assert type(stimulus) is EvalScenarioStimulusV1
            case_settings = settings.model_copy(
                update={"max_concurrency": concurrency_by_case[case.id].max_concurrency}
            )
            async with preflight_limit:
                try:
                    scenario = await eval_store.load_scenario(stimulus.scenario_revision)
                    if scenario is None:
                        raise ValueError("scenario unavailable")
                    preflight = await preflight_eval_scenario(
                        scenario,
                        execution_target,
                        case_settings,
                        actor_authorized=True,
                        project_root=registration.manifest_project_root,
                    )
                    if not preflight.ready or preflight.binding is None:
                        detail = (
                            preflight.diagnostics[0].message
                            if preflight.diagnostics
                            else "Current scenario launch requirements are not ready."
                        )
                        return case.id, None, None, detail
                    scenario_invocation = EvalScenarioRunInvocation(
                        scenario_revision=scenario.revision,
                        binding_revision=preflight.binding.revision,
                        authored_suite_revision=suite.revision,
                        authored_case_revision=case.revision,
                        environment_name=preflight.binding.environment_name,
                        trials=trial_policy.trial_count,
                        timeout_seconds=preflight.binding.timeout_seconds,
                        artifact_references=tuple(
                            EvalScenarioArtifactReference(
                                requirement_id=item.requirement_id,
                                artifact_id=item.artifact_id,
                            )
                            for item in preflight.binding.artifacts
                        ),
                    )
                    profile_invocation = eval_run_invocation(
                        None,
                        max_steps=preflight.binding.max_steps,
                        limits=preflight.binding.operator_run_limits,
                        cost_budget=_narrow_eval_cost_budget(
                            preflight.binding.cost_budget,
                            request.cost_budget,
                        ),
                        scenario=scenario_invocation,
                        authored_suite_revision=suite.revision,
                        authored_suite_selection_revision=selection.revision,
                    )
                    effective_target = target_for_eval_invocation(
                        execution_target,
                        profile_invocation,
                    )
                    prepared_profile = await active_eval_registry.prepare_execution_profile(
                        suite.target_key,
                        effective_target=effective_target,
                    )
                    if (
                        trial_policy.trial_count > prepared_profile.snapshot.ceilings.max_trials
                        or concurrency_by_case[case.id].max_concurrency
                        > prepared_profile.snapshot.ceilings.max_concurrency
                    ):
                        return (
                            case.id,
                            None,
                            None,
                            "The suite trial policy exceeds the current execution profile.",
                        )
                    corpus = await asyncio.to_thread(
                        corpus_for_authored_scenario_case,
                        suite,
                        case.id,
                        scenario,
                        preflight.binding,
                        execution_target,
                        project_root=registration.manifest_project_root,
                    )
                    await asyncio.to_thread(
                        compile_corpus_suite,
                        corpus,
                        execution_target,
                        suite.suite.id,
                    )
                    return (
                        case.id,
                        (scenario, preflight.binding),
                        prepared_profile.snapshot,
                        None,
                    )
                except Exception:
                    return (
                        case.id,
                        None,
                        None,
                        "The exact scenario is unavailable or incompatible with current authority.",
                    )

        prepared = await asyncio.gather(*(prepare_scenario(case) for case in scenario_cases))
        for case_id, material, profile, message in prepared:
            if material is None:
                diagnostics.append(
                    EvalAuthoredSuiteLaunchDiagnostic(
                        code="scenario_launch_not_ready",
                        case_id=case_id,
                        message=message,
                    )
                )
            else:
                assert profile is not None
                prepared_scenarios[case_id] = material
                execution_profiles_by_case[case_id] = profile
                launch_index = next(
                    index for index, launch in enumerate(launches) if launch.case_ids == (case_id,)
                )
                launches[launch_index] = launches[launch_index].model_copy(
                    update={"execution_profile_revision": profile.revision}
                )
    exposure = None
    if not diagnostics:
        try:
            exposure_launches = tuple(
                EvalCandidateLaunchExposure(
                    case_ids=launch.case_ids,
                    execution_profile=execution_profiles_by_case[launch.case_ids[0]],
                    cost_budget=(
                        request.cost_budget
                        if launch.kind == "simple_input"
                        else _narrow_eval_cost_budget(
                            prepared_scenarios[launch.case_ids[0]][1].cost_budget,
                            request.cost_budget,
                        )
                    ),
                )
                for launch in launches
            )
            candidate_pricing_fingerprint = (
                None
                if execution_target.price_book is None
                else pricing_profile_identity(execution_target.price_book).fingerprint
            )
            exposure = compile_authored_suite_run_exposure(
                suite,
                selection,
                exposure_launches,
                judge_profiles=registration.catalog_entry.judge_profiles,
                candidate_pricing_profile_fingerprint=(candidate_pricing_fingerprint),
            )
        except (KeyError, TypeError, ValueError):
            diagnostics.append(
                EvalAuthoredSuiteLaunchDiagnostic(
                    code="work_exposure_unavailable",
                    message=(
                        "The exact maximum configured candidate and judge work could "
                        "not be proven from the current execution profiles."
                    ),
                )
            )
    return (
        EvalAuthoredSuiteRunPreviewResponse(
            selection=selection,
            ready=not diagnostics,
            launches=tuple(launches),
            diagnostics=tuple(diagnostics),
            exposure=exposure,
        ),
        prepared_scenarios,
    )


def register_suite_launch_routes(
    bounded_evals_router: APIRouter,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    protected: Sequence[Depends],
    optional_auth_context: Depends,
) -> None:
    """Register suite launch preview and start using the shared HTTP boundaries."""

    # Keep FastAPI's shared dependency object as the typed auth-context default.
    launch_auth_context = cast("AuthContext | None", optional_auth_context)

    @bounded_evals_router.post(
        "/evals/suites/{suite_revision}/runs/preview",
        response_model=EvalAuthoredSuiteRunPreviewResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def preview_eval_authored_suite_run(
        suite_revision: str,
        body: EvalAuthoredSuiteRunSelectionRequest,
    ) -> EvalAuthoredSuiteRunPreviewResponse:
        suite = await load_authored_suite(
            suite_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        preview, _ = await _preview_authored_suite_launch(
            suite, body, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        return preview

    @bounded_evals_router.post(
        "/evals/suites/{suite_revision}/runs",
        response_model=EvalAuthoredSuiteRunLaunchResponse,
        status_code=202,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def launch_eval_authored_suite_run(
        suite_revision: str,
        body: EvalAuthoredSuiteRunLaunchRequest,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=512),
        ],
        auth_context: AuthContext | None = launch_auth_context,
    ) -> EvalAuthoredSuiteRunLaunchResponse:
        suite = await load_authored_suite(
            suite_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        replay_probe = bind_eval_admission_request(
            eval_run_invocation(
                auth_context,
                max_steps=None,
                limits=None,
                cost_budget=body.cost_budget,
            ),
            kind="authored_suite",
            target_key=suite.target_key,
            resource_identity={"suite_revision": suite.revision},
            body=body,
        )
        admission_request_revision = replay_probe.admission_request_revision
        if admission_request_revision is None:
            raise RuntimeError("Authored eval launch lost its admission request revision.")
        replayed_first_part = await replay_eval_run(
            eval_store=eval_store,
            target_key=suite.target_key,
            idempotency_key=idempotency_key,
            idempotency_namespace="authored-suite-part-1",
            admission_request_revision=admission_request_revision,
        )
        if replayed_first_part is None:
            replayed_parts: tuple[EvalRunRecord | None, ...] = (None,) * len(
                body.expected_execution_profiles
            )
        else:
            replayed_parts_list: list[EvalRunRecord | None] = [replayed_first_part]
            for index in range(1, len(body.expected_execution_profiles)):
                replayed_parts_list.append(
                    await replay_eval_run(
                        eval_store=eval_store,
                        target_key=suite.target_key,
                        idempotency_key=idempotency_key,
                        idempotency_namespace=f"authored-suite-part-{index + 1}",
                        admission_request_revision=admission_request_revision,
                    )
                )
            replayed_parts = tuple(replayed_parts_list)
        if all(part is not None for part in replayed_parts):
            selection = eval_suite_selection(suite, body.case_ids)
            cases_by_id = {case.id: case for case in suite.cases}
            replayed_runs = []
            for expectation, part in zip(
                body.expected_execution_profiles,
                replayed_parts,
                strict=True,
            ):
                if part is None:
                    raise RuntimeError("Complete authored eval replay lost an admitted part.")
                first_case = cases_by_id[expectation.case_ids[0]]
                kind: Literal["simple_input", "scenario"] = (
                    "scenario"
                    if type(first_case.stimulus) is EvalScenarioStimulusV1
                    else "simple_input"
                )
                replayed_runs.append(
                    EvalAuthoredSuiteAdmittedRun(
                        kind=kind,
                        case_ids=expectation.case_ids,
                        run=part,
                    )
                )
            return EvalAuthoredSuiteRunLaunchResponse(
                selection=selection,
                runs=tuple(replayed_runs),
            )
        partial_replay = any(part is not None for part in replayed_parts)
        preview, prepared_scenarios = await _preview_authored_suite_launch(
            suite,
            body,
            eval_store=eval_store,
            active_eval_registry=active_eval_registry,
        )
        if not preview.ready:
            raise HTTPException(
                status_code=503 if partial_replay else 409,
                detail=(
                    "An earlier authored eval launch attempt admitted only part of this "
                    "request. Restore current launch readiness and retry with the same "
                    "Idempotency-Key."
                    if partial_replay
                    else "Authored eval suite launch requirements are not currently ready."
                ),
            )
        if preview.exposure is None or body.expected_exposure_revision != preview.exposure.revision:
            raise HTTPException(
                status_code=503 if partial_replay else 409,
                detail=(
                    "The authored-suite maximum work or cost exposure changed after "
                    "readiness. Check launch readiness again."
                ),
            )
        expected_profiles = tuple(
            (expectation.case_ids, expectation.execution_profile_revision)
            for expectation in body.expected_execution_profiles
        )
        current_profiles = tuple(
            (plan.case_ids, plan.execution_profile_revision) for plan in preview.launches
        )
        if expected_profiles != current_profiles:
            raise HTTPException(
                status_code=503 if partial_replay else 409,
                detail=(
                    "An earlier authored eval launch attempt admitted only part of this "
                    "request. Restore the reviewed execution profiles and retry with the "
                    "same Idempotency-Key."
                    if partial_replay
                    else "The authored-suite execution profile changed after readiness. "
                    "Check launch readiness again."
                ),
            )
        registration = active_eval_registry.registration(suite.target_key)
        if registration is None:
            raise HTTPException(status_code=404, detail="Authored eval suite not found.")
        execution_target = registration.execution_target()
        trial_policy = eval_suite_trial_policy(suite.suite)
        cases_by_id = {case.id: case for case in suite.cases}
        launch_revision = eval_idempotency_digest(
            suite.target_key,
            idempotency_key,
            namespace="authored-suite-launch",
        )
        concurrency_allocations = allocate_authored_suite_launch_concurrency(
            trial_policy,
            len(preview.launches),
        )
        prepared_runs = []
        for plan, allocation in zip(
            preview.launches,
            concurrency_allocations,
            strict=True,
        ):
            if plan.kind == "simple_input":
                selection = eval_suite_selection(suite, plan.case_ids)
                invocation = eval_run_invocation(
                    auth_context,
                    max_steps=None,
                    limits=None,
                    cost_budget=body.cost_budget,
                    authored_suite_revision=suite.revision,
                    authored_suite_selection_revision=preview.selection.revision,
                    authored_suite_launch_revision=launch_revision,
                    authored_suite_launch_lane=allocation.lane,
                    authored_suite_exposure=preview.exposure,
                )
                effective_target = target_for_eval_invocation(
                    execution_target,
                    invocation,
                )
                corpus = await asyncio.to_thread(
                    corpus_for_authored_simple_selection,
                    suite,
                    selection,
                    effective_target,
                    project_root=registration.manifest_project_root,
                )
            else:
                case_id = plan.case_ids[0]
                case = cases_by_id[case_id]
                scenario, binding = prepared_scenarios[case_id]
                scenario_invocation = EvalScenarioRunInvocation(
                    scenario_revision=scenario.revision,
                    binding_revision=binding.revision,
                    authored_suite_revision=suite.revision,
                    authored_case_revision=case.revision,
                    environment_name=binding.environment_name,
                    trials=trial_policy.trial_count,
                    timeout_seconds=binding.timeout_seconds,
                    artifact_references=tuple(
                        EvalScenarioArtifactReference(
                            requirement_id=item.requirement_id,
                            artifact_id=item.artifact_id,
                        )
                        for item in binding.artifacts
                    ),
                )
                invocation = eval_run_invocation(
                    auth_context,
                    max_steps=binding.max_steps,
                    limits=binding.operator_run_limits,
                    cost_budget=_narrow_eval_cost_budget(
                        binding.cost_budget,
                        body.cost_budget,
                    ),
                    scenario=scenario_invocation,
                    authored_suite_revision=suite.revision,
                    authored_suite_selection_revision=preview.selection.revision,
                    authored_suite_launch_revision=launch_revision,
                    authored_suite_launch_lane=allocation.lane,
                    authored_suite_exposure=preview.exposure,
                )
                effective_target = target_for_eval_invocation(
                    execution_target,
                    invocation,
                )
                corpus = await asyncio.to_thread(
                    corpus_for_authored_scenario_case,
                    suite,
                    case_id,
                    scenario,
                    binding,
                    effective_target,
                    project_root=registration.manifest_project_root,
                )
            if plan.execution_profile_revision is None:
                raise RuntimeError("Ready authored-suite launch lost its profile revision.")
            invocation = bind_eval_admission_request(
                invocation,
                kind="authored_suite",
                target_key=suite.target_key,
                resource_identity={"suite_revision": suite.revision},
                body=body,
            )
            try:
                eval_target, compiled, invocation = await prepare_eval_run(
                    active_eval_registry=active_eval_registry,
                    corpus=corpus,
                    suite_id=suite.suite.id,
                    max_concurrency=allocation.max_concurrency,
                    invocation=invocation,
                    expected_execution_profile_revision=(plan.execution_profile_revision),
                    expect_exact_execution_profile=True,
                )
            except HTTPException as exc:
                if partial_replay and exc.status_code == 409:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "An earlier authored eval launch attempt admitted only part of "
                            "this request. Restore current launch readiness and retry with "
                            "the same Idempotency-Key."
                        ),
                    ) from exc
                raise
            prepared_runs.append((plan, corpus, invocation, eval_target, compiled, allocation))

        for _, corpus, _, eval_target, _, _ in prepared_runs:
            try:
                await eval_store.save_corpus(
                    corpus,
                    redact_json=eval_target.app.redact_json,
                )
            except EvalCorpusConflict as exc:
                raise HTTPException(
                    status_code=409,
                    detail="Derived authored-suite corpus conflicts with stored content.",
                ) from exc
            except EvalStorePublicationRejected as exc:
                raise HTTPException(
                    status_code=422,
                    detail="Derived authored-suite corpus contains unsafe public data.",
                ) from exc
            except EvalStoreResultTooLarge as exc:
                raise HTTPException(
                    status_code=413,
                    detail="Derived authored-suite corpus exceeds the server byte limit.",
                ) from exc

        admitted: list[EvalAuthoredSuiteAdmittedRun] = []
        for index, (
            plan,
            corpus,
            invocation,
            eval_target,
            compiled,
            allocation,
        ) in enumerate(prepared_runs):
            run = await admit_eval_run(
                eval_store=eval_store,
                corpus=corpus,
                max_concurrency=allocation.max_concurrency,
                invocation=invocation,
                idempotency_key=idempotency_key,
                idempotency_namespace=f"authored-suite-part-{index + 1}",
                eval_target=eval_target,
                compiled=compiled,
            )
            admitted.append(
                EvalAuthoredSuiteAdmittedRun(
                    kind=plan.kind,
                    case_ids=plan.case_ids,
                    run=run,
                )
            )
        return EvalAuthoredSuiteRunLaunchResponse(
            selection=preview.selection,
            runs=tuple(admitted),
        )
