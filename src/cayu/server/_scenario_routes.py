"""Evaluation scenario authoring, artifact fixture preparation, catalog and launch routes."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import Response

from cayu.evals.scenario import EvalScenarioDocumentV2, eval_scenario_to_json
from cayu.evals.scenario_authoring import (
    compile_eval_scenario_draft,
    validate_expected_scenario_revision,
)
from cayu.evals.scenario_execution import corpus_for_eval_scenario
from cayu.evals.scenario_preflight import (
    ScenarioArtifactMaterializationError,
    ScenarioLaunchBindingV2,
    ScenarioLaunchSettingsV2,
    materialize_eval_scenario_artifact_fixture,
    preflight_eval_scenario,
)
from cayu.evals.store import (
    EVAL_STORE_DEFAULT_PAGE_BYTES,
    EVAL_STORE_DEFAULT_PAGE_SIZE,
    EVAL_STORE_MAX_CURSOR_BYTES,
    EVAL_STORE_MAX_IDENTIFIER_CHARS,
    EVAL_STORE_MAX_PAGE_BYTES,
    EVAL_STORE_MAX_PAGE_SIZE,
    EvalCorpusConflict,
    EvalRunInvocation,
    EvalRunRecord,
    EvalScenarioArtifactReference,
    EvalScenarioCatalogPage,
    EvalScenarioCatalogQuery,
    EvalScenarioConflict,
    EvalScenarioRunInvocation,
    EvalStorePublicationRejected,
    EvalStoreResultTooLarge,
)
from cayu.server._eval_run_admission import (
    admit_eval_run,
    bind_eval_admission_request,
    eval_run_invocation,
    prepare_eval_run,
    replay_eval_run,
)
from cayu.server._http_json import _model_json_response, _render_utf8
from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    EVALS_ENDPOINT_RESPONSES,
    EvalScenarioArtifactMaterializationRequest,
    EvalScenarioArtifactMaterializationResponse,
    EvalScenarioPreviewRequest,
    EvalScenarioPreviewResponse,
    EvalScenarioRunCreateRequest,
    EvalScenarioSaveRequest,
    EvalScenarioSaveResponse,
)
from cayu.server.evals_registry import EvalTargetRegistration, target_for_eval_invocation

if TYPE_CHECKING:
    from fastapi.params import Depends

    from cayu.evals.store import EvalStore
    from cayu.server.evals_registry import EvalTargetRegistry


async def load_eval_scenario(
    scenario_revision: str,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
) -> EvalScenarioDocumentV2:
    """Load one visible scenario revision for authoring reads and run launch."""
    if not eval_store.scenarios:
        raise HTTPException(
            status_code=409,
            detail="Durable scenario persistence is not available.",
        )
    try:
        scenario = await eval_store.load_scenario(scenario_revision)
    except EvalStoreResultTooLarge as exc:
        raise HTTPException(
            status_code=413,
            detail="Eval scenario exceeds the server byte limit.",
        ) from exc
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc
    if scenario is None or active_eval_registry.get(scenario.target_key) is None:
        raise HTTPException(status_code=404, detail="Eval scenario not found.")
    return scenario


async def preflight_scenario(
    scenario: EvalScenarioDocumentV2,
    settings: ScenarioLaunchSettingsV2,
    *,
    active_eval_registry: EvalTargetRegistry,
):
    """Resolve current target and launch readiness through the private HTTP boundary."""
    registration = active_eval_registry.registration(scenario.target_key)
    if registration is None:
        raise HTTPException(
            status_code=400,
            detail="Eval scenario is incompatible with the attached targets.",
        )
    try:
        execution_target = registration.execution_target()
        result = await preflight_eval_scenario(
            scenario,
            execution_target,
            settings,
            actor_authorized=True,
            project_root=registration.manifest_project_root,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Eval scenario or its launch selections are invalid.",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=409,
            detail="Attached eval target is unavailable for scenario preflight.",
        ) from exc
    return registration, result


def register_scenario_authoring_routes(
    bounded_evals_router: APIRouter,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    protected: Sequence[Depends],
) -> None:
    """Register scenario authoring routes with the shared bounded HTTP boundary."""

    async def _scenario_execution_profile(
        registration: EvalTargetRegistration,
        scenario: EvalScenarioDocumentV2,
        binding: ScenarioLaunchBindingV2,
    ):
        scenario_invocation = EvalScenarioRunInvocation(
            scenario_revision=scenario.revision,
            binding_revision=binding.revision,
            environment_name=binding.environment_name,
            trials=binding.trials,
            timeout_seconds=binding.timeout_seconds,
            artifact_references=tuple(
                EvalScenarioArtifactReference(
                    requirement_id=item.requirement_id,
                    artifact_id=item.artifact_id,
                )
                for item in binding.artifacts
            ),
        )
        invocation = EvalRunInvocation(
            max_steps=binding.max_steps,
            limits=binding.operator_run_limits,
            cost_budget=binding.cost_budget,
            scenario=scenario_invocation,
        )
        effective_target = target_for_eval_invocation(
            registration.execution_target(),
            invocation,
        )
        return await active_eval_registry.prepare_execution_profile(
            registration.target.key,
            effective_target=effective_target,
        )

    @bounded_evals_router.post(
        "/evals/scenarios/preview",
        response_model=EvalScenarioPreviewResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def preview_eval_scenario(
        body: EvalScenarioPreviewRequest,
    ) -> EvalScenarioPreviewResponse:
        try:
            scenario = compile_eval_scenario_draft(body.draft)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Eval scenario draft is invalid.",
            ) from exc
        registration, preflight = await preflight_scenario(
            scenario, body.settings, active_eval_registry=active_eval_registry
        )
        profile_revision = None
        if preflight.ready and preflight.binding is not None:
            try:
                prepared_profile = await _scenario_execution_profile(
                    registration,
                    scenario,
                    preflight.binding,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The current scenario execution profile is unavailable.",
                ) from exc
            profile_revision = prepared_profile.snapshot.revision
        return EvalScenarioPreviewResponse(
            scenario=scenario,
            preflight=preflight,
            execution_profile_revision=profile_revision,
        )

    @bounded_evals_router.post(
        "/evals/scenarios",
        response_model=EvalScenarioSaveResponse,
        status_code=201,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def save_eval_scenario(
        body: EvalScenarioSaveRequest,
    ) -> EvalScenarioSaveResponse:
        if not eval_store.scenarios:
            raise HTTPException(
                status_code=409,
                detail="Durable scenario persistence is not available.",
            )
        try:
            scenario = validate_expected_scenario_revision(
                body.scenario,
                body.expected_scenario_revision,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Eval scenario changed after the reviewed revision.",
            ) from exc
        registration, preflight = await preflight_scenario(
            scenario, body.settings, active_eval_registry=active_eval_registry
        )
        profile_revision = None
        if preflight.ready and preflight.binding is not None:
            try:
                prepared_profile = await _scenario_execution_profile(
                    registration,
                    scenario,
                    preflight.binding,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The current scenario execution profile is unavailable.",
                ) from exc
            profile_revision = prepared_profile.snapshot.revision
        try:
            entry = await eval_store.save_scenario(
                scenario,
                redact_json=registration.target.app.redact_json,
            )
        except EvalScenarioConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="Eval scenario revision conflicts with stored content.",
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="Eval scenario contains unsafe public data.",
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval scenario exceeds the server byte limit.",
            ) from exc
        return EvalScenarioSaveResponse(
            entry=entry,
            scenario=scenario,
            preflight=preflight,
            execution_profile_revision=profile_revision,
        )

    @bounded_evals_router.post(
        "/evals/scenarios/artifacts/{requirement_id}/materialize",
        response_model=EvalScenarioArtifactMaterializationResponse,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def materialize_eval_scenario_artifact(
        requirement_id: str,
        body: EvalScenarioArtifactMaterializationRequest,
    ) -> EvalScenarioArtifactMaterializationResponse:
        try:
            scenario = validate_expected_scenario_revision(
                body.scenario,
                body.expected_scenario_revision,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Eval scenario changed after the reviewed revision.",
            ) from exc
        registration = active_eval_registry.registration(scenario.target_key)
        if registration is None:
            raise HTTPException(
                status_code=400,
                detail="Eval scenario is incompatible with the attached targets.",
            )
        try:
            materialization = await materialize_eval_scenario_artifact_fixture(
                scenario,
                registration.execution_target(),
                requirement_id,
                environment_name=body.settings.environment_name,
                source_artifact_id=body.settings.artifact_references.get(requirement_id),
                project_root=registration.manifest_project_root,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Eval scenario artifact requirement not found.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Eval scenario artifact selection is invalid.",
            ) from exc
        except ScenarioArtifactMaterializationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        references = dict(body.settings.artifact_references)
        references.pop(requirement_id, None)
        settings = ScenarioLaunchSettingsV2.model_validate(
            {
                **body.settings.model_dump(mode="python"),
                "artifact_references": references,
            }
        )
        registration, preflight = await preflight_scenario(
            materialization.scenario,
            settings,
            active_eval_registry=active_eval_registry,
        )
        profile_revision = None
        if preflight.ready and preflight.binding is not None:
            try:
                prepared_profile = await _scenario_execution_profile(
                    registration,
                    materialization.scenario,
                    preflight.binding,
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=409,
                    detail="The current scenario execution profile is unavailable.",
                ) from exc
            profile_revision = prepared_profile.snapshot.revision
        return EvalScenarioArtifactMaterializationResponse(
            materialization=materialization,
            preflight=preflight,
            execution_profile_revision=profile_revision,
        )

    @bounded_evals_router.get(
        "/evals/scenarios",
        response_model=EvalScenarioCatalogPage,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def list_eval_scenarios(
        target_key: Annotated[
            str | None,
            Query(max_length=EVAL_STORE_MAX_IDENTIFIER_CHARS),
        ] = None,
        scenario_id: Annotated[
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
        if not eval_store.scenarios:
            raise HTTPException(
                status_code=409,
                detail="Durable scenario persistence is not available.",
            )
        selected_key = active_eval_registry.default_target_key if target_key is None else target_key
        eval_target = active_eval_registry.get(selected_key)
        if eval_target is None:
            raise HTTPException(status_code=404, detail="Eval target not found.")
        try:
            return await eval_store.list_scenarios(
                EvalScenarioCatalogQuery(
                    target_key=eval_target.key,
                    scenario_id=scenario_id,
                    cursor=cursor,
                    limit=limit,
                    max_result_bytes=max_result_bytes,
                )
            )
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Eval scenario catalog page exceeds the requested byte limit.",
            ) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Invalid Evals query.") from exc

    @bounded_evals_router.get(
        "/evals/scenarios/{scenario_revision}",
        response_model=EvalScenarioDocumentV2,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def get_eval_scenario(scenario_revision: str):
        scenario = await load_eval_scenario(
            scenario_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        return await _model_json_response(scenario, EvalScenarioDocumentV2)

    @bounded_evals_router.get(
        "/evals/scenarios/{scenario_revision}/download",
        response_class=Response,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def download_eval_scenario(scenario_revision: str) -> Response:
        scenario = await load_eval_scenario(
            scenario_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        scenario_json = await asyncio.to_thread(
            _render_utf8,
            eval_scenario_to_json,
            scenario,
        )
        return Response(
            content=scenario_json,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{scenario.target_key}-'
                    f'{scenario.revision[7:19]}.scenario.json"'
                )
            },
        )


def register_scenario_launch_routes(
    bounded_evals_router: APIRouter,
    *,
    eval_store: EvalStore,
    active_eval_registry: EvalTargetRegistry,
    protected: Sequence[Depends],
    optional_auth_context: Depends,
) -> None:
    """Register saved-scenario launch with shared validation and admission boundaries."""

    # Preserve FastAPI's shared dependency object as the typed auth-context default.
    launch_auth_context = cast("AuthContext | None", optional_auth_context)

    @bounded_evals_router.post(
        "/evals/scenarios/{scenario_revision}/runs",
        response_model=EvalRunRecord,
        status_code=202,
        responses=EVALS_ENDPOINT_RESPONSES,
        dependencies=protected,
    )
    async def launch_eval_scenario(
        scenario_revision: str,
        body: EvalScenarioRunCreateRequest,
        idempotency_key: Annotated[
            str,
            Header(alias="Idempotency-Key", min_length=1, max_length=512),
        ],
        auth_context: AuthContext | None = launch_auth_context,
    ) -> EvalRunRecord:
        if not eval_store.scenario_execution:
            raise HTTPException(
                status_code=409,
                detail="Durable scenario execution is not available.",
            )
        scenario = await load_eval_scenario(
            scenario_revision, eval_store=eval_store, active_eval_registry=active_eval_registry
        )
        replay_probe = bind_eval_admission_request(
            eval_run_invocation(
                auth_context,
                max_steps=None,
                limits=None,
                cost_budget=None,
            ),
            kind="scenario",
            target_key=scenario.target_key,
            resource_identity={"scenario_revision": scenario.revision},
            body=body,
        )
        admission_request_revision = replay_probe.admission_request_revision
        if admission_request_revision is None:
            raise RuntimeError("Scenario eval launch lost its admission request revision.")
        replayed = await replay_eval_run(
            eval_store=eval_store,
            target_key=scenario.target_key,
            idempotency_key=idempotency_key,
            admission_request_revision=admission_request_revision,
        )
        if replayed is not None:
            return replayed
        registration, preflight = await preflight_scenario(
            scenario, body.settings, active_eval_registry=active_eval_registry
        )
        binding = preflight.binding
        if not preflight.ready or binding is None:
            raise HTTPException(
                status_code=409,
                detail="Eval scenario launch requirements are not currently ready.",
            )
        if binding.revision != body.expected_binding_revision:
            raise HTTPException(
                status_code=409,
                detail="Eval scenario launch binding changed after review.",
            )
        scenario_invocation = EvalScenarioRunInvocation(
            scenario_revision=scenario.revision,
            binding_revision=binding.revision,
            environment_name=binding.environment_name,
            trials=binding.trials,
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
            cost_budget=binding.cost_budget,
            scenario=scenario_invocation,
        )
        invocation = bind_eval_admission_request(
            invocation,
            kind="scenario",
            target_key=scenario.target_key,
            resource_identity={"scenario_revision": scenario.revision},
            body=body,
        )
        try:
            effective_target = target_for_eval_invocation(
                registration.execution_target(),
                invocation,
            )
            corpus = await asyncio.to_thread(
                corpus_for_eval_scenario,
                scenario,
                binding,
                effective_target,
                project_root=registration.manifest_project_root,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Eval scenario no longer matches its current target binding.",
            ) from exc
        eval_target, compiled, invocation = await prepare_eval_run(
            active_eval_registry=active_eval_registry,
            corpus=corpus,
            suite_id="scenario",
            max_concurrency=binding.max_concurrency,
            invocation=invocation,
            expected_execution_profile_revision=(body.expected_execution_profile_revision),
            expect_exact_execution_profile=True,
        )
        try:
            await eval_store.save_corpus(
                corpus,
                redact_json=eval_target.app.redact_json,
            )
        except EvalCorpusConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="Derived scenario result contract conflicts with stored content.",
            ) from exc
        except EvalStorePublicationRejected as exc:
            raise HTTPException(
                status_code=422,
                detail="Derived scenario result contract contains unsafe public data.",
            ) from exc
        except EvalStoreResultTooLarge as exc:
            raise HTTPException(
                status_code=413,
                detail="Derived scenario result contract exceeds the server byte limit.",
            ) from exc
        return await admit_eval_run(
            eval_store=eval_store,
            corpus=corpus,
            max_concurrency=binding.max_concurrency,
            invocation=invocation,
            idempotency_key=idempotency_key,
            eval_target=eval_target,
            compiled=compiled,
        )
