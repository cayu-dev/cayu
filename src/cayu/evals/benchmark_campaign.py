"""Benchmark admission through the existing authored-suite store and coordinator."""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import Field, StrictFloat, StrictInt, StrictStr, field_validator, model_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.artifacts import ArtifactScope, ArtifactStore
from cayu.artifacts.local import _rename_directory_no_replace
from cayu.evals._admission import LaunchAdmission, admission_scope
from cayu.evals._inspection_documents import ProcessDocuments, write_process_document
from cayu.evals._process_progress import ProcessEvalProgress
from cayu.evals.benchmark_package import (
    BenchmarkPackageV1,
    LoadedBenchmarkPackage,
    benchmark_package_scenarios,
    benchmark_suite_selection,
)
from cayu.evals.capacity import EvalExecutionCapacity
from cayu.evals.corpus import (
    EvalCorpusDocument,
    _content_revision,
    _model_content_revision,
    _PortableModel,
    _sha256_revision,
    eval_suite_trial_policy,
    pricing_profile_identity,
)
from cayu.evals.execution import (
    CorpusTarget,
    WorkflowEvalTarget,
    _copy_corpus_target,
    compile_corpus_suite,
)
from cayu.evals.execution_profiles import EvalExecutionProfilePolicyV1
from cayu.evals.runner import EvalPlan
from cayu.evals.scenario import EvalScenarioDocumentV2
from cayu.evals.scenario_preflight import ScenarioLaunchSettingsV2, preflight_eval_scenario
from cayu.evals.store import (
    EvalRunCostBudget,
    EvalRunInvocation,
    EvalRunRecoveryPolicyV1,
    EvalRunRequest,
    EvalRunRetryLineageV1,
    EvalRunSpec,
    EvalScenarioArtifactReference,
    EvalScenarioRunInvocation,
    EvalStore,
)
from cayu.evals.suite_authoring import (
    EvalScenarioStimulusV1,
    EvalSimpleInputStimulusV1,
    EvalSuiteDocumentV3,
    EvalSuiteDraftV3,
    EvalSuiteSelectionV1,
    EvalSuiteTrialRequestDraftV3,
    compile_eval_suite_draft_v3,
    eval_suite_selection,
)
from cayu.evals.suite_execution import (
    corpus_for_authored_scenario_case,
    corpus_for_authored_simple_selection,
)
from cayu.evals.suite_preflight import (
    EvalCandidateLaunchExposure,
    allocate_authored_suite_launch_concurrency,
    compile_authored_suite_run_exposure,
)
from cayu.runtime.stop_policy import RunLimits
from cayu.sessions.base import ModelTarget, copy_run_request
from cayu.storage.evals_sqlite import SQLiteEvalStore

if TYPE_CHECKING:
    from cayu.server.evals_registry import EvalTargetRegistry

BENCHMARK_CAMPAIGN_MAX_BYTES = 32 * 1024 * 1024


class BenchmarkCampaignSettingsV1(_PortableModel):
    """Explicit caller selections; omitted trial settings inherit the package."""

    trials: StrictInt | None = Field(default=None, ge=1, le=100)
    minimum_passed_trials: StrictInt | None = Field(default=None, ge=1, le=100)
    max_concurrency: StrictInt | None = Field(default=None, ge=1, le=2**31 - 1)
    case_timeout_seconds: StrictInt | None = Field(default=None, ge=1, le=3600)
    max_steps: StrictInt | None = Field(default=None, ge=1)
    limits: RunLimits | None = None
    cost_budget: EvalRunCostBudget | None = None
    model_target: ModelTarget | None = None
    environment_name: StrictStr | None = Field(default=None, min_length=1, max_length=256)
    stagger_seconds: StrictFloat = Field(default=0.0, ge=0, allow_inf_nan=False)
    recovery_policy: EvalRunRecoveryPolicyV1 = Field(default_factory=EvalRunRecoveryPolicyV1)
    max_retry_attempts: StrictInt = Field(default=0, ge=0, le=3)
    retry_backoff_seconds: StrictInt = Field(default=0, ge=0, le=3600)
    retry_categories: tuple[
        Literal[
            "timeout",
            "provider_failure",
            "environment_failure",
            "execution_failure",
            "recovery_blocked",
            "answer_mismatch",
            "scoring_failure",
            "capture_failure",
            "cancelled",
            "evidence_unavailable",
        ],
        ...,
    ] = ("timeout", "provider_failure", "environment_failure", "recovery_blocked")


BenchmarkRetryLineageV1 = EvalRunRetryLineageV1


class BenchmarkCampaignRunV1(_PortableModel):
    spec: EvalRunSpec
    case_ids: tuple[StrictStr, ...] = Field(min_length=1, max_length=1000)


class BenchmarkCampaignV1(_PortableModel):
    """Immutable admission receipt, not another job state or source of ownership."""

    schema_version: Literal[1] = 1
    revision: StrictStr
    id: StrictStr = Field(min_length=1, max_length=128)
    created_at: StrictStr
    package_id: StrictStr
    package_version: StrictStr
    package_revision: StrictStr
    scorer_id: StrictStr
    scorer_version: StrictStr
    launch_revision: StrictStr
    settings: BenchmarkCampaignSettingsV1
    selection: EvalSuiteSelectionV1
    runs: tuple[BenchmarkCampaignRunV1, ...] = Field(min_length=1, max_length=1000)
    retry_of: BenchmarkRetryLineageV1 | None = None

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_schema(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1.")
        return value

    @field_validator("revision", "package_revision", "launch_revision")
    @classmethod
    def validate_revisions(cls, value: str, info) -> str:
        return _sha256_revision(value, info.field_name)

    @field_validator("created_at")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        if datetime.fromisoformat(value).utcoffset() is None:
            raise ValueError("Campaign admission requires a timezone-aware timestamp.")
        return value

    @model_validator(mode="after")
    def validate_contract(self) -> BenchmarkCampaignV1:
        selected_ids = tuple(case.id for case in self.selection.cases)
        case_ids = tuple(case_id for run in self.runs for case_id in run.case_ids)
        if tuple(sorted(case_ids)) != selected_ids:
            raise ValueError("Campaign runs must partition the exact selected cohort.")
        run_ids = tuple(run.spec.id for run in self.runs)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("Campaign run IDs must be unique.")
        for run in self.runs:
            invocation = run.spec.invocation
            if (
                invocation.recovery_policy != self.settings.recovery_policy
                or invocation.retry_of != self.retry_of
                or invocation.authored_suite_exposure
                != self.runs[0].spec.invocation.authored_suite_exposure
                or invocation.authored_suite_launch_revision != self.launch_revision
                or invocation.authored_suite_selection_revision != self.selection.revision
                or invocation.authored_suite_revision != self.selection.suite_document_revision
                or run.spec.suite_revision != self.selection.suite_revision
                or run.spec.suite_id != self.selection.suite_id
                or invocation.execution_profile is None
                or invocation.execution_profile_snapshot is None
                or invocation.authored_suite_exposure is None
            ):
                raise ValueError("Campaign admission is missing its exact native run identity.")
        exposure = self.runs[0].spec.invocation.authored_suite_exposure
        assert exposure is not None
        if exposure.candidate_trials % len(selected_ids) != 0:
            raise ValueError("Campaign trial allowance must cover every selected case equally.")
        if self.retry_of is not None and (
            selected_ids != (self.retry_of.case_id,)
            or exposure.candidate_trials != 1
            or self.settings.max_retry_attempts != 0
        ):
            raise ValueError("A retry successor must contain exactly one original trial slot.")
        if self.revision != _model_content_revision(self, "benchmark campaign"):
            raise ValueError("Campaign revision does not match its admission.")
        encoded = canonical_durable_json_bytes(self.model_dump(mode="json"), "benchmark campaign")
        if len(encoded) > BENCHMARK_CAMPAIGN_MAX_BYTES:
            raise ValueError("Campaign admission exceeds its byte limit.")
        return self


@dataclass(frozen=True)
class PreparedBenchmarkCampaign:
    campaign: BenchmarkCampaignV1
    suite: EvalSuiteDocumentV3
    corpora: tuple[EvalCorpusDocument, ...]
    scenarios: tuple[EvalScenarioDocumentV2, ...]
    target: CorpusTarget
    registry: EvalTargetRegistry
    package: LoadedBenchmarkPackage


def _selected_target(plan: EvalPlan, settings: BenchmarkCampaignSettingsV1) -> CorpusTarget:
    from cayu.server.evals_registry import _copy_target_with

    if type(plan) is not EvalPlan or (plan.corpus_target is None and plan.workflow_target is None):
        raise ValueError(
            "A benchmark package requires a corpus-target or workflow-target EvalPlan."
        )
    if plan.suite is not None:
        raise ValueError("A packaged benchmark cannot also select a direct EvalSuite.")
    original = plan.corpus_target or plan.workflow_target
    assert original is not None
    target = _copy_corpus_target(original)
    request = copy_run_request(target.request_base)
    if settings.model_target is not None:
        request = request.model_copy(update={"target": settings.model_target})
    if settings.environment_name is not None:
        request = request.model_copy(update={"environment_name": settings.environment_name})
    return _copy_target_with(target, request_base=copy_run_request(request))


def _launch_suite(package: BenchmarkPackageV1, settings: BenchmarkCampaignSettingsV1):
    document, _ = benchmark_suite_selection(package)
    draft = EvalSuiteDraftV3.from_document(document)
    current = draft.trial_request
    trial_request = EvalSuiteTrialRequestDraftV3(
        trials=settings.trials if settings.trials is not None else current.trials,
        minimum_passed_trials=(
            settings.minimum_passed_trials
            if settings.minimum_passed_trials is not None
            else current.minimum_passed_trials
        ),
        max_concurrency=(
            settings.max_concurrency
            if settings.max_concurrency is not None
            else current.max_concurrency
        ),
        timeout_seconds=(
            settings.case_timeout_seconds
            if settings.case_timeout_seconds is not None
            else current.timeout_seconds
        ),
    )
    return compile_eval_suite_draft_v3(draft.model_copy(update={"trial_request": trial_request}))


def _validate_requirements(package: BenchmarkPackageV1, target: CorpusTarget) -> None:
    manifest = target.app.describe()
    environments = {item.name for item in manifest.environments}
    agent = next(
        (item for item in manifest.agents if item.name == target.request_base.agent_name), None
    )
    tools = set() if agent is None else {item.name for item in agent.tools}
    if not set(package.requirements.environments).issubset(environments):
        raise ValueError("The selected target is missing required benchmark environments.")
    if not set(package.requirements.tools).issubset(tools):
        raise ValueError("The selected target is missing required benchmark tools.")


async def _materialize_inputs(
    loaded: LoadedBenchmarkPackage,
    target: CorpusTarget,
    selected_scenarios: tuple[EvalScenarioDocumentV2, ...],
) -> dict[str, dict[str, str]]:
    scenarios = {scenario.revision: scenario for scenario in selected_scenarios}
    revisions = dict(
        zip(
            (item.revision for item in loaded.package.scenarios),
            (item.revision for item in benchmark_package_scenarios(loaded.package)),
            strict=True,
        )
    )
    files = tuple(
        (binding, content)
        for binding, content in loaded.file_contents
        if revisions[binding.scenario_revision] in scenarios
    )
    references: dict[str, dict[str, str]] = {revision: {} for revision in scenarios}
    if not files:
        return references
    environment = target.app.get_environment(target.request_base.environment_name)
    if environment.factory is not None or not isinstance(
        environment.environment.artifact_store, ArtifactStore
    ):
        raise ValueError("Benchmark files require an existing environment with an ArtifactStore.")
    store = environment.environment.artifact_store
    for binding, content in files:
        requirement = next(
            item
            for item in scenarios[revisions[binding.scenario_revision]].artifact_requirements
            if item.id == binding.requirement_id
        )
        # Recheck a caller-constructed LoadedBenchmarkPackage at the execution boundary.
        if (
            len(content) != requirement.size_bytes
            or hashlib.sha256(content).hexdigest() != requirement.content_sha256
        ):
            raise ValueError("Benchmark input snapshot does not match its required digest.")
        metadata = await store.put_bytes(
            content,
            filename=requirement.filename,
            content_type=requirement.content_type,
            scope=ArtifactScope.ENVIRONMENT,
            environment_name=environment.spec.name,
        )
        references[revisions[binding.scenario_revision]][binding.requirement_id] = metadata.id
    return references


async def prepare_benchmark_campaign(
    loaded: LoadedBenchmarkPackage,
    plan: EvalPlan,
    *,
    settings: BenchmarkCampaignSettingsV1 | None = None,
    case_ids: Sequence[str] | None = None,
    campaign_id: str | None = None,
    retry_of: BenchmarkRetryLineageV1 | None = None,
) -> PreparedBenchmarkCampaign:
    """Resolve every native launch before admitting or dispatching candidate work.

    The only materialization is bounded input-file publication to the explicitly
    configured ArtifactStore. Providers, workflows, tools, and judges do not run.
    """
    from cayu.server.evals_registry import explicit_eval_target_registry, target_for_eval_invocation

    if type(loaded) is not LoadedBenchmarkPackage:
        raise TypeError("loaded must be an exact LoadedBenchmarkPackage.")
    package = BenchmarkPackageV1.model_validate(loaded.package.model_dump(mode="json"))
    if tuple(binding for binding, _ in loaded.file_contents) != package.files:
        raise ValueError("Benchmark input snapshots must match the exact manifest bindings.")
    if any(type(content) is not bytes for _, content in loaded.file_contents):
        raise TypeError("Benchmark input snapshots must contain immutable bytes.")
    settings = BenchmarkCampaignSettingsV1.model_validate(
        (settings or BenchmarkCampaignSettingsV1()).model_dump(mode="python")
    )
    target = _selected_target(plan, settings)
    if package.suite.target_key != target.key:
        raise ValueError("Benchmark package target key does not match the selected trusted target.")
    _validate_requirements(package, target)
    suite = _launch_suite(package, settings)
    selection = eval_suite_selection(suite, case_ids)
    selected_ids = {item.id for item in selection.cases}
    selected_cases = tuple(case for case in suite.cases if case.id in selected_ids)
    simple_ids = tuple(
        case.id for case in selected_cases if type(case.stimulus) is EvalSimpleInputStimulusV1
    )
    scenario_cases = tuple(
        case for case in selected_cases if type(case.stimulus) is EvalScenarioStimulusV1
    )
    if scenario_cases and type(target) is WorkflowEvalTarget:
        raise ValueError(
            "File and lifecycle scenarios currently require a native agent CorpusTarget."
        )
    policy = plan.execution_profile_policy or EvalExecutionProfilePolicyV1.safe_default()
    registry = explicit_eval_target_registry(target, policy=policy)
    registration = registry.registration(target.key)
    assert registration is not None
    execution_target = registration.execution_target()
    trial_policy = eval_suite_trial_policy(suite.suite)
    if (
        trial_policy.trial_count > policy.max_trials
        or trial_policy.max_concurrency > policy.max_concurrency
    ):
        raise ValueError(
            "Benchmark trials/concurrency exceed the target's declared execution profile."
        )
    groups = ((simple_ids,) if simple_ids else ()) + tuple((case.id,) for case in scenario_cases)
    allocations = allocate_authored_suite_launch_concurrency(trial_policy, len(groups))
    package_scenarios = benchmark_package_scenarios(package)
    scenarios_by_revision = {scenario.revision: scenario for scenario in package_scenarios}
    scenarios = tuple(
        scenarios_by_revision[case.stimulus.scenario_revision] for case in scenario_cases
    )
    references = await _materialize_inputs(loaded, execution_target, scenarios)
    identifier = campaign_id or f"benchmark-{uuid4()}"
    launch_revision = _content_revision({"campaign_id": identifier}, "benchmark launch")
    prepared_parts = []
    for index, (group, allocation) in enumerate(zip(groups, allocations, strict=True)):
        invocation_fields = dict(
            retain_trial_checkpoints=True,
            recovery_policy=settings.recovery_policy,
            retry_of=retry_of,
            max_steps=settings.max_steps,
            limits=settings.limits,
            cost_budget=settings.cost_budget,
            authored_suite_revision=suite.revision,
            authored_suite_selection_revision=selection.revision,
            authored_suite_launch_revision=launch_revision,
            authored_suite_launch_lane=allocation.lane,
        )
        scenario = None
        if not (simple_ids and index == 0):
            case = next(case for case in scenario_cases if case.id == group[0])
            scenario = scenarios_by_revision[case.stimulus.scenario_revision]
            preflight = await preflight_eval_scenario(
                scenario,
                execution_target,
                ScenarioLaunchSettingsV2(
                    environment_name=settings.environment_name,
                    trials=trial_policy.trial_count,
                    max_concurrency=allocation.max_concurrency,
                    timeout_seconds=suite.suite.trial_request.timeout_seconds,
                    max_steps=settings.max_steps,
                    limits=settings.limits,
                    cost_budget=settings.cost_budget,
                    artifact_references=references[scenario.revision],
                ),
                actor_authorized=True,
            )
            if not preflight.ready or preflight.binding is None:
                raise ValueError(
                    "Benchmark scenario preflight failed: "
                    + ", ".join(item.code for item in preflight.diagnostics)
                )
            binding = preflight.binding
            invocation_fields.update(
                max_steps=binding.max_steps,
                scenario=EvalScenarioRunInvocation(
                    scenario_revision=scenario.revision,
                    binding_revision=binding.revision,
                    authored_suite_revision=suite.revision,
                    authored_case_revision=case.revision,
                    environment_name=binding.environment_name,
                    trials=binding.trials,
                    timeout_seconds=binding.timeout_seconds,
                    artifact_references=tuple(
                        EvalScenarioArtifactReference(
                            requirement_id=item.requirement_id, artifact_id=item.artifact_id
                        )
                        for item in binding.artifacts
                    ),
                ),
            )
        invocation = EvalRunInvocation.model_validate(invocation_fields)
        effective = target_for_eval_invocation(execution_target, invocation)
        corpus = (
            corpus_for_authored_simple_selection(
                suite, eval_suite_selection(suite, group), effective
            )
            if scenario is None
            else corpus_for_authored_scenario_case(suite, group[0], scenario, binding, effective)
        )
        compile_corpus_suite(corpus, effective, suite.suite.id)
        profile = await registry.prepare_execution_profile(target.key, effective_target=effective)
        prepared_parts.append((group, allocation, corpus, invocation, profile))
    exposure = compile_authored_suite_run_exposure(
        suite,
        selection,
        tuple(
            EvalCandidateLaunchExposure(
                case_ids=group,
                execution_profile=profile.snapshot,
                cost_budget=invocation.cost_budget,
            )
            for group, _, _, invocation, profile in prepared_parts
        ),
        judge_profiles=registration.catalog_entry.judge_profiles,
        candidate_pricing_profile_fingerprint=(
            None
            if target.price_book is None
            else pricing_profile_identity(target.price_book).fingerprint
        ),
    )
    runs = []
    for index, (group, allocation, corpus, invocation, profile) in enumerate(prepared_parts):
        invocation = EvalRunInvocation.model_validate(
            {
                **invocation.model_dump(mode="python"),
                "execution_profile": profile.binding,
                "execution_profile_snapshot": profile.snapshot,
                "authored_suite_exposure": exposure,
            }
        )
        runs.append(
            BenchmarkCampaignRunV1(
                case_ids=group,
                spec=EvalRunSpec(
                    run_id=f"{identifier}-{index + 1}",
                    corpus_revision=corpus.revision,
                    target_key=target.key,
                    suite_id=suite.suite.id,
                    suite_revision=suite.suite.revision,
                    max_concurrency=allocation.max_concurrency,
                    invocation=invocation,
                ),
            )
        )
    material = dict(
        schema_version=1,
        id=identifier,
        created_at=datetime.now(UTC).isoformat(),
        package_id=package.id,
        package_version=package.version,
        package_revision=package.revision,
        scorer_id=package.scorer_id,
        scorer_version=package.scorer_version,
        launch_revision=launch_revision,
        settings=settings.model_dump(mode="json"),
        selection=selection.model_dump(mode="json"),
        runs=[run.model_dump(mode="json") for run in runs],
        retry_of=None if retry_of is None else retry_of.model_dump(mode="json"),
    )
    campaign = BenchmarkCampaignV1.model_validate(
        {"revision": _content_revision(material, "benchmark campaign"), **material}
    )
    if target.app.redact_json(campaign.model_dump(mode="json")) != campaign.model_dump(mode="json"):
        raise ValueError("Benchmark admission contains configured workload-secret material.")
    return PreparedBenchmarkCampaign(
        campaign=campaign,
        suite=suite,
        corpora=tuple(part[2] for part in prepared_parts),
        scenarios=package_scenarios,
        target=target,
        registry=registry,
        package=loaded,
    )


def load_benchmark_campaign(directory: str | Path) -> BenchmarkCampaignV1:
    documents = ProcessDocuments(Path(directory))
    material = documents.read("campaign.json", required=True)
    return BenchmarkCampaignV1.model_validate(material)


def campaign_run_request(
    campaign: BenchmarkCampaignV1, run: BenchmarkCampaignRunV1
) -> EvalRunRequest:
    return EvalRunRequest.model_validate(
        {
            **run.spec.model_dump(mode="python"),
            "idempotency_key": _content_revision(
                {"campaign": campaign.id, "run": run.spec.id}, "benchmark admission"
            ),
        }
    )


async def admit_benchmark_campaign(
    prepared: PreparedBenchmarkCampaign, directory: str | Path
) -> BenchmarkCampaignV1:
    """Publish a complete admission atomically before any worker can execute it."""

    destination = Path(directory).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # A crashed writer may leave private staging material, but never claims the
    # deterministic retry slot. Competing writers publish with no replacement.
    with TemporaryDirectory(prefix=".benchmark-admission-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "campaign"
        await _write_benchmark_campaign(prepared, staged)
        _rename_directory_no_replace(staged, destination, parent_fd=None)
        if os.name != "nt":
            descriptor = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    return prepared.campaign


async def _write_benchmark_campaign(prepared: PreparedBenchmarkCampaign, destination: Path) -> None:
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    package_root = destination / "package"
    package_root.mkdir(mode=0o700)
    for binding, content in prepared.package.file_contents:
        output = package_root / binding.path
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(content)
    write_process_document(
        package_root / "benchmark.json", prepared.package.package.model_dump(mode="json")
    )
    store = SQLiteEvalStore(destination / "evals.sqlite3")
    try:
        redact = prepared.target.app.redact_json
        for scenario in prepared.scenarios:
            await store.save_scenario(scenario, redact_json=redact)
        await store.save_authored_suite(prepared.suite, redact_json=redact)
        for corpus in prepared.corpora:
            await store.save_corpus(corpus, redact_json=redact)
        # A restart can replay admission from this receipt and the already-saved catalog.
        write_process_document(
            destination / "campaign.json", prepared.campaign.model_dump(mode="json")
        )
        for run in prepared.campaign.runs:
            await store.admit_run(campaign_run_request(prepared.campaign, run), redact_json=redact)
    finally:
        await store.close()


async def resume_benchmark_campaign(directory: str | Path, plan: EvalPlan) -> BenchmarkCampaignV1:
    """Restore admission idempotently, then let native leases arbitrate every worker."""
    from cayu.server.evals_registry import explicit_eval_target_registry, target_for_eval_invocation

    campaign = load_benchmark_campaign(directory)
    target = _selected_target(plan, campaign.settings)
    registry = explicit_eval_target_registry(
        target, policy=plan.execution_profile_policy or EvalExecutionProfilePolicyV1.safe_default()
    )
    store = SQLiteEvalStore(Path(directory) / "evals.sqlite3")
    try:
        # Check the current profile before completing a partially persisted admission.
        for run in campaign.runs:
            effective = target_for_eval_invocation(target, run.spec.invocation)
            current = await registry.prepare_execution_profile(
                target.key, effective_target=effective
            )
            if current.binding != run.spec.invocation.execution_profile:
                raise ValueError("Resume target no longer matches the admitted execution profile.")
        for run in campaign.runs:
            await store.admit_run(
                campaign_run_request(campaign, run), redact_json=target.app.redact_json
            )
        await execute_benchmark_campaign(campaign, directory, registry=registry, store=store)
    finally:
        await store.close()
    return campaign


async def execute_benchmark_campaign(
    campaign: BenchmarkCampaignV1,
    directory: str | Path,
    *,
    registry: EvalTargetRegistry,
    store: EvalStore,
) -> None:
    """Wait for native durable workers; campaign receipts never own trial scheduling."""
    from cayu.server.evals_registry import ResolvedEvalsRuntime
    from cayu.server.evals_worker import EvalRunCoordinator

    exposure = campaign.runs[0].spec.invocation.authored_suite_exposure
    assert exposure is not None
    concurrency = exposure.max_concurrency
    runtime = ResolvedEvalsRuntime(
        registry=registry,
        store=store,
        execution_capacity=EvalExecutionCapacity(concurrency),
        lease_seconds=30,
        poll_interval_seconds=0.1,
        shutdown_grace_seconds=5.0,
    )
    workers = tuple(
        EvalRunCoordinator(runtime, launch_revision=campaign.launch_revision)
        for _ in range(min(len(campaign.runs), concurrency))
    )
    observation_directory = Path(directory) / "observations" / str(uuid4())
    observation_directory.mkdir(mode=0o700, parents=True)
    progress = ProcessEvalProgress(
        observation_directory,
        launch_id=campaign.id,
        index=0,
        fingerprint=campaign.revision,
        case_ids=tuple(case.id for case in campaign.selection.cases),
        max_trials=exposure.candidate_trials // len(campaign.selection.cases),
        retain_trial_revisions=True,
    )

    async def wait(run_id: str) -> None:
        while True:
            try:
                record = await store.wait_for_run_terminal(run_id, timeout_seconds=30.0)
            except TimeoutError:
                continue
            if record is None:
                raise ValueError("A campaign's admitted run is missing from its EvalStore.")
            return

    with progress.activate(), admission_scope(LaunchAdmission(campaign.settings.stagger_seconds)):
        try:
            for worker in workers:
                worker.start()
            async with asyncio.TaskGroup() as group:
                for run in campaign.runs:
                    group.create_task(wait(run.spec.id))
        finally:
            await asyncio.gather(*(worker.stop() for worker in workers))
