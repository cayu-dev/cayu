from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from hashlib import sha256
from pathlib import Path

import pytest

from cayu import (
    AgentSpec,
    ArtifactScope,
    Environment,
    EnvironmentSpec,
    EvalRunInvocation,
    EvalRunRequest,
    EvalScenarioArtifactReference,
    EvalScenarioDocumentV2,
    EvalScenarioRunInvocation,
    InMemoryEvalStore,
    LocalArtifactStore,
    Message,
    RunRequest,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioJsonPartV2,
    ScenarioLaunchSettingsV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    compile_corpus_suite,
    corpus_for_eval_scenario,
    preflight_eval_scenario,
    run_compiled_eval_scenario,
)
from cayu.artifacts.attachments import RESOLVED_FILE_ATTACHMENTS_OPTION
from cayu.core.messages import TextPart
from cayu.evals.corpus import (
    CorpusUserMessageSpec,
    EvalCaseSpec,
    EvalCorpusDocument,
    EvalSuiteSpec,
    EvaluationEvidencePolicySpec,
    EvaluationSourceIdentityV1,
    FinalOutputEqualsAssertionSpec,
    RootStatusAssertionSpec,
    RunInputSpec,
    TrialRequestSpec,
)
from cayu.evals.execution import (
    CorpusExecutionResult,
    CorpusTarget,
    EvaluationTargetIdentity,
    run_corpus_suite,
)
from cayu.evals.execution_comparison import (
    CorpusComparisonReason,
    compare_corpus_execution_results,
)
from cayu.evals.external import (
    EXTERNAL_PROCESS_PROTOCOL_VERSION,
    EXTERNAL_TRIAL_ENVELOPE_PREFIX,
    ExternalBodyReleaseV1,
    ExternalProcessModelProvider,
    ExternalProcessTargetIdentityV1,
    ExternalTrialEnvelopeV1,
    ExternalTrialIdentityV1,
    OpaqueExternalCaseRefV1,
    external_body_content_revision,
    external_trial_envelope_from_request,
    with_external_trial_envelope,
)
from cayu.evals.result_contract import EvalTrialDiagnosticCode
from cayu.evals.store import EvalRunStatus
from cayu.providers import ModelProviderError, ModelRequest, ModelStreamEvent
from cayu.providers.operations import (
    ProviderOperationAdapter,
    ProviderOperationConnection,
    ProviderOperationRecoveryMetadata,
    ProviderOperationSnapshot,
    ProviderOperationStartRequest,
    ProviderOperationState,
    ProviderOperationStatus,
)
from cayu.runtime.app import CayuApp
from cayu.server.config import EvalsConfig
from cayu.server.evals_registry import (
    explicit_eval_target_registry,
    target_for_eval_invocation,
)
from cayu.server.evals_worker import EvalRunCoordinator
from cayu.storage.evals_sqlite import SQLiteEvalStore

_A = "sha256:" + "a" * 64
_B = "sha256:" + "b" * 64
_C = "sha256:" + "c" * 64
_D = "sha256:" + "d" * 64
_E = "sha256:" + "e" * 64
_F = "sha256:" + "f" * 64


async def _empty_events() -> AsyncIterator[ModelStreamEvent]:
    if False:  # pragma: no cover
        yield ModelStreamEvent.text_delta("")


class _Operations(ProviderOperationAdapter):
    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        del request
        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id="operation-one",
                stream_protocol=EXTERNAL_PROCESS_PROTOCOL_VERSION,
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=_empty_events(),
        )

    async def retrieve(self, state: ProviderOperationState) -> ProviderOperationSnapshot:
        return ProviderOperationSnapshot(state=state, status=ProviderOperationStatus.IN_PROGRESS)

    async def reconnect(self, state: ProviderOperationState) -> ProviderOperationConnection:
        return ProviderOperationConnection(
            state=state,
            status=ProviderOperationStatus.IN_PROGRESS,
            events=_empty_events(),
        )


class _CapturingOperations(_Operations):
    def __init__(self, identity: ExternalProcessTargetIdentityV1) -> None:
        self.identity = identity
        self.envelopes: list[ExternalTrialEnvelopeV1] = []
        self.trials: list[ExternalTrialIdentityV1] = []
        self.candidate_requests: list[ModelRequest] = []

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        envelope, candidate_request = external_trial_envelope_from_request(
            request.request,
            expected_target_revision=self.identity.revision,
        )
        self.envelopes.append(envelope)
        self.trials.append(envelope.trial)
        self.candidate_requests.append(candidate_request)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            yield ModelStreamEvent.text_delta("Approved", recovery_metadata={"cursor": 1})
            yield ModelStreamEvent.completed(
                {
                    "finish_reason": "stop",
                    "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                },
                recovery_metadata={"cursor": 2},
            )

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id=envelope.trial.revision,
                stream_protocol=EXTERNAL_PROCESS_PROTOCOL_VERSION,
                recovery_metadata=ProviderOperationRecoveryMetadata(cursor=0),
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _ConcurrentCapturingOperations(_CapturingOperations):
    def __init__(
        self,
        identity: ExternalProcessTargetIdentityV1,
        *,
        expected_concurrency: int,
    ) -> None:
        super().__init__(identity)
        self.expected_concurrency = expected_concurrency
        self.active = 0
        self.maximum_active = 0
        self._gate = asyncio.Event()

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        envelope, candidate_request = external_trial_envelope_from_request(
            request.request,
            expected_target_revision=self.identity.revision,
        )
        self.envelopes.append(envelope)
        self.trials.append(envelope.trial)
        self.candidate_requests.append(candidate_request)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if self.active == self.expected_concurrency:
            self._gate.set()
        await asyncio.wait_for(self._gate.wait(), timeout=5)

        async def events() -> AsyncIterator[ModelStreamEvent]:
            try:
                yield ModelStreamEvent.text_delta("Approved", recovery_metadata={"cursor": 1})
                yield ModelStreamEvent.completed(
                    {
                        "finish_reason": "stop",
                        "usage": {
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                        },
                    },
                    recovery_metadata={"cursor": 2},
                )
            finally:
                self.active -= 1

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id=envelope.trial.revision,
                stream_protocol=EXTERNAL_PROCESS_PROTOCOL_VERSION,
                recovery_metadata=ProviderOperationRecoveryMetadata(cursor=0),
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


class _DispositionOperations(_Operations):
    def __init__(
        self,
        identity: ExternalProcessTargetIdentityV1,
        *,
        error_code: str,
        status: ProviderOperationStatus,
    ) -> None:
        self.identity = identity
        self.error_code = error_code
        self.status = status

    async def start(self, request: ProviderOperationStartRequest) -> ProviderOperationConnection:
        envelope, _ = external_trial_envelope_from_request(
            request.request,
            expected_target_revision=self.identity.revision,
        )

        async def events() -> AsyncIterator[ModelStreamEvent]:
            error = ModelProviderError(
                "External target ended.",
                provider="cayu-external-process",
                error_type="ExternalContainerDisposition",
                error_code=self.error_code,
                retryable=False,
            )
            yield ModelStreamEvent.error(
                "External target ended.",
                cause=error,
                provider_operation_status=self.status,
                recovery_metadata={"cursor": 1},
            )

        return ProviderOperationConnection(
            state=ProviderOperationState(
                operation_id=envelope.trial.revision,
                stream_protocol=EXTERNAL_PROCESS_PROTOCOL_VERSION,
                recovery_metadata=ProviderOperationRecoveryMetadata(cursor=0),
            ),
            status=ProviderOperationStatus.IN_PROGRESS,
            events=events(),
        )


def _body() -> ExternalBodyReleaseV1:
    return ExternalBodyReleaseV1.create(
        content_revision=_A,
        private_runtime_path="private_runtime.py",
        private_runtime_revision=_B,
        launch_protocol_revision=_C,
        entrypoint=("python", "-I", "/opt/body/agent.py"),
    )


def _target() -> ExternalProcessTargetIdentityV1:
    return ExternalProcessTargetIdentityV1.create(
        body=_body(),
        evaluator_runtime_revision=_D,
        target_implementation_revision=_E,
        runner_revision=_F,
        environment_revision=_A,
        reset_contract_revision=_B,
        evidence_policy_revision=_C,
    )


def _trial(target: ExternalProcessTargetIdentityV1 | None = None) -> ExternalTrialIdentityV1:
    selected = target or _target()
    return ExternalTrialIdentityV1.create(
        native_run_id="run-one",
        target_key="external-agent",
        target_revision=selected.revision,
        corpus_revision=_A,
        suite_id="hidden-preservation",
        suite_revision=_B,
        case_id="preserve-attachment",
        case_revision=_C,
        trial_number=2,
    )


def test_external_body_and_target_identities_pin_independent_material() -> None:
    first = _target()
    repeated = _target()
    changed_runtime = ExternalProcessTargetIdentityV1.create(
        body=first.body,
        evaluator_runtime_revision=_D,
        target_implementation_revision=_E,
        runner_revision=_F,
        environment_revision=_A,
        reset_contract_revision=_B,
        evidence_policy_revision=_D,
    )

    assert first == repeated
    assert first.body.content_revision == _A
    assert first.body.private_runtime_path == "private_runtime.py"
    assert first.body.private_runtime_revision == _B
    assert first.body.launch_protocol_revision == _C
    assert changed_runtime.revision != first.revision

    with pytest.raises(ValueError, match="does not match"):
        ExternalBodyReleaseV1.model_validate(
            {**first.body.model_dump(mode="json"), "content_revision": _F}
        )


def test_external_body_rejects_aliases_and_noncanonical_private_runtime_paths(
    tmp_path: Path,
) -> None:
    body = tmp_path / "body"
    body.mkdir()
    (body / "private_runtime.py").write_text("# trusted runtime\n", encoding="utf-8")
    (body / "agent.py").write_text("# candidate\n", encoding="utf-8")
    (body / "alias.py").symlink_to(body / "agent.py")

    with pytest.raises(ValueError, match="symbolic links"):
        external_body_content_revision(body)
    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        ExternalBodyReleaseV1.create(
            content_revision=_A,
            private_runtime_path="../private_runtime.py",
            private_runtime_revision=_B,
            launch_protocol_revision=_C,
            entrypoint=("python", "{body}/agent.py", "{request}"),
        )


def test_external_trial_envelope_is_exact_removed_before_candidate_dispatch() -> None:
    target = _target()
    opaque_ref = OpaqueExternalCaseRefV1(id="arena-case-alias", revision=_F)
    envelope = ExternalTrialEnvelopeV1(
        trial=_trial(target),
        opaque_case_ref=opaque_ref,
    )
    original = [
        Message.text("system", "Follow the packaged-agent protocol."),
        Message.text("user", "Inspect the attachment."),
    ]
    request = ModelRequest(
        model=EXTERNAL_PROCESS_PROTOCOL_VERSION,
        messages=with_external_trial_envelope(original, envelope),
    )

    extracted, candidate_request = external_trial_envelope_from_request(
        request,
        expected_target_revision=target.revision,
    )

    assert extracted == envelope
    assert candidate_request.messages == original
    assert request.messages != original
    assert extracted.trial.trial_number == 2
    assert extracted.trial.revision == _trial(target).revision
    assert extracted.opaque_case_ref == opaque_ref


def test_external_trial_envelope_fails_closed_on_missing_duplicate_or_target_drift() -> None:
    target = _target()
    envelope = ExternalTrialEnvelopeV1(trial=_trial(target))
    marker = envelope.message()

    with pytest.raises(ValueError, match="exactly one"):
        external_trial_envelope_from_request(
            ModelRequest(
                model=EXTERNAL_PROCESS_PROTOCOL_VERSION,
                messages=[Message.text("user", "No marker")],
            ),
            expected_target_revision=target.revision,
        )
    with pytest.raises(ValueError, match="exactly one"):
        external_trial_envelope_from_request(
            ModelRequest(
                model=EXTERNAL_PROCESS_PROTOCOL_VERSION,
                messages=[marker, marker],
            ),
            expected_target_revision=target.revision,
        )
    with pytest.raises(ValueError, match="changed after admission"):
        external_trial_envelope_from_request(
            ModelRequest(
                model=EXTERNAL_PROCESS_PROTOCOL_VERSION,
                messages=[marker],
            ),
            expected_target_revision=_F,
        )


def test_external_process_provider_uses_reconnectable_runtime_operations() -> None:
    target = _target()
    operations = _Operations()
    provider = ExternalProcessModelProvider(identity=target, operations=operations)

    assert provider.provider_operations is operations
    assert provider.provider_operation_mode == "background"
    assert provider.execution_profile_identity.behavior_version == EXTERNAL_PROCESS_PROTOCOL_VERSION
    assert provider.execution_profile_identity.implementation_version == target.revision

    provider.preflight_portable_messages(
        model=EXTERNAL_PROCESS_PROTOCOL_VERSION,
        messages=[Message.text("system", "Trusted envelope"), Message.text("user", "Run")],
        tools=[],
    )
    with pytest.raises(ValueError, match="tool-definition support"):
        provider.preflight_portable_messages(
            model=EXTERNAL_PROCESS_PROTOCOL_VERSION,
            messages=[Message.text("user", "Run")],
            tools=[{"name": "candidate_tool", "description": "", "input_schema": {}}],
        )


def test_external_target_runs_through_native_corpus_lifecycle_with_exact_trial_identity() -> None:
    evidence_policy = EvaluationEvidencePolicySpec.standard()
    base = _target()
    identity = ExternalProcessTargetIdentityV1.create(
        body=base.body,
        evaluator_runtime_revision=base.evaluator_runtime_revision,
        target_implementation_revision=base.target_implementation_revision,
        runner_revision=base.runner_revision,
        environment_revision=base.environment_revision,
        reset_contract_revision=base.reset_contract_revision,
        evidence_policy_revision=evidence_policy.revision,
    )
    operations = _ConcurrentCapturingOperations(identity, expected_concurrency=24)
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ExternalProcessModelProvider(identity=identity, operations=operations),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="external-candidate", model=EXTERNAL_PROCESS_PROTOCOL_VERSION)
    )
    target = CorpusTarget(
        key="external-agent",
        app=app,
        request_base=RunRequest(
            agent_name="external-candidate",
            messages=[],
            max_steps=1,
        ),
        application_release_id="external-release-one",
        evidence_policy=evidence_policy,
        external_process=identity,
    )
    suite = EvalSuiteSpec.create(
        id="external-suite",
        name="External suite",
        trial_request=TrialRequestSpec(trials=24, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id="external-case",
        suite_id=suite.id,
        name="External case",
        source=EvaluationSourceIdentityV1(
            application_release_id="source-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision=_A,
        ),
        input=RunInputSpec(
            opaque_external_case_ref=OpaqueExternalCaseRefV1(
                id="arena-scoped-case-alias",
                revision=_F,
            ),
        ),
        assertions=(
            RootStatusAssertionSpec(id="completed", expected="completed"),
            FinalOutputEqualsAssertionSpec(id="answer", expected="Approved"),
        ),
    )
    corpus = EvalCorpusDocument.create(
        target_key=target.key,
        evidence_policy=evidence_policy,
        suites=(suite,),
        cases=(case,),
    )

    result = asyncio.run(run_corpus_suite(target, corpus, suite.id, max_concurrency=24))

    assert result.target.external_process == identity
    assert result.run.status == "passed"
    assert len(operations.trials) == 24
    assert sorted(operations.trials, key=lambda trial: trial.trial_number) == list(
        result.external_trials
    )
    assert len({trial.native_run_id for trial in result.external_trials}) == 1
    assert sorted(trial.trial_number for trial in operations.trials) == list(range(1, 25))
    assert operations.maximum_active == 24
    assert {trial.case_revision for trial in operations.trials} == {case.revision}
    assert case.input is not None
    assert {envelope.opaque_case_ref for envelope in operations.envelopes} == {
        case.input.opaque_external_case_ref
    }
    assert all(not request.messages for request in operations.candidate_requests)
    assert all(
        not isinstance(part, TextPart) or not part.text.startswith(EXTERNAL_TRIAL_ENVELOPE_PREFIX)
        for request in operations.candidate_requests
        for message in request.messages
        for part in message.content
    )
    for trial in result.run.cases[0].trials:
        assert trial.usage is not None
        assert trial.usage.total_tokens == 3
        assert trial.duration_ms >= 0

    changed_external = ExternalProcessTargetIdentityV1.create(
        body=identity.body,
        evaluator_runtime_revision=identity.evaluator_runtime_revision,
        target_implementation_revision=_D,
        runner_revision=identity.runner_revision,
        environment_revision=identity.environment_revision,
        reset_contract_revision=identity.reset_contract_revision,
        evidence_policy_revision=identity.evidence_policy_revision,
    )
    changed_target = EvaluationTargetIdentity(
        target_key=result.target.target_key,
        application_release_id=result.target.application_release_id,
        app_manifest=result.target.app_manifest,
        external_process=changed_external,
    )
    changed_trials = tuple(
        ExternalTrialIdentityV1.create(
            native_run_id=trial.native_run_id,
            target_key=trial.target_key,
            target_revision=changed_external.revision,
            corpus_revision=trial.corpus_revision,
            suite_id=trial.suite_id,
            suite_revision=trial.suite_revision,
            case_id=trial.case_id,
            case_revision=trial.case_revision,
            trial_number=trial.trial_number,
        )
        for trial in result.external_trials
    )
    changed_result = CorpusExecutionResult.create(
        target=changed_target,
        run=result.run,
        external_trials=changed_trials,
    )
    compatibility = compare_corpus_execution_results(result, changed_result).compatibility
    assert compatibility.reasons == (CorpusComparisonReason.EXTERNAL_TARGET_REVISION_MISMATCH,)


def test_opaque_external_case_reference_is_rejected_for_an_ordinary_target() -> None:
    evidence_policy = EvaluationEvidencePolicySpec.standard()
    base = _target()
    identity = ExternalProcessTargetIdentityV1.create(
        body=base.body,
        evaluator_runtime_revision=base.evaluator_runtime_revision,
        target_implementation_revision=base.target_implementation_revision,
        runner_revision=base.runner_revision,
        environment_revision=base.environment_revision,
        reset_contract_revision=base.reset_contract_revision,
        evidence_policy_revision=evidence_policy.revision,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ExternalProcessModelProvider(identity=identity, operations=_Operations()),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="external-candidate", model=EXTERNAL_PROCESS_PROTOCOL_VERSION)
    )
    target = CorpusTarget(
        key="ordinary-agent",
        app=app,
        request_base=RunRequest(
            agent_name="external-candidate",
            messages=[],
            max_steps=1,
        ),
        application_release_id="ordinary-release",
        evidence_policy=evidence_policy,
    )
    suite = EvalSuiteSpec.create(id="suite", name="Suite")
    case = EvalCaseSpec.create(
        id="case",
        suite_id=suite.id,
        name="Case",
        source=EvaluationSourceIdentityV1(
            application_release_id="source-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision=_A,
        ),
        input=RunInputSpec(
            opaque_external_case_ref=OpaqueExternalCaseRefV1(
                id="private-case-alias",
                revision=_F,
            )
        ),
        assertions=(RootStatusAssertionSpec(id="completed", expected="completed"),),
    )
    corpus = EvalCorpusDocument.create(
        target_key=target.key,
        evidence_policy=evidence_policy,
        suites=(suite,),
        cases=(case,),
    )

    with pytest.raises(ValueError, match="require an external process target"):
        compile_corpus_suite(corpus, target, suite.id)


def test_external_scenario_delivers_structured_input_and_digest_attested_attachment(
    tmp_path: Path,
) -> None:
    async def exercise():
        evidence_policy = EvaluationEvidencePolicySpec.standard()
        base = _target()
        identity = ExternalProcessTargetIdentityV1.create(
            body=base.body,
            evaluator_runtime_revision=base.evaluator_runtime_revision,
            target_implementation_revision=base.target_implementation_revision,
            runner_revision=base.runner_revision,
            environment_revision=base.environment_revision,
            reset_contract_revision=base.reset_contract_revision,
            evidence_policy_revision=evidence_policy.revision,
        )
        operations = _CapturingOperations(identity)
        artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="external-files")
        content = b"%PDF-1.7 preservation fixture\n"
        artifact = await artifact_store.put_bytes(
            content,
            filename="preservation.pdf",
            content_type="application/pdf",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="files",
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ExternalProcessModelProvider(identity=identity, operations=operations),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="external-candidate", model=EXTERNAL_PROCESS_PROTOCOL_VERSION)
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="files"),
                artifact_store=artifact_store,
            ),
            default=True,
        )
        target = CorpusTarget(
            key="external-agent",
            app=app,
            request_base=RunRequest(
                agent_name="external-candidate",
                messages=[],
                max_steps=1,
            ),
            application_release_id="external-release-one",
            evidence_policy=evidence_policy,
            external_process=identity,
        )
        requirement = ScenarioArtifactRequirementV2(
            id="preservation-document",
            source="artifact_reference",
            reference=artifact.id,
            content_sha256=sha256(content).hexdigest(),
            filename=artifact.filename,
            content_type=artifact.content_type,
            size_bytes=artifact.size_bytes,
        )
        scenario = EvalScenarioDocumentV2.create(
            id="structured-preservation",
            target_key=target.key,
            name="Structured preservation",
            events=(
                ScenarioInitialInputEventV2(
                    sequence=0,
                    id="initial",
                    input=ScenarioInputV2.create(
                        (
                            ScenarioUserMessageV2.create(
                                (
                                    ScenarioTextPartV2(text="Inspect this case."),
                                    ScenarioJsonPartV2(value={"preserve": True, "count": 2}),
                                    ScenarioFilePartV2(artifact_requirement_id=requirement.id),
                                )
                            ),
                        )
                    ),
                ),
            ),
            artifact_requirements=(requirement,),
        )
        preflight = await preflight_eval_scenario(
            scenario,
            target,
            ScenarioLaunchSettingsV2(
                environment_name="files",
                trials=1,
                max_concurrency=1,
                timeout_seconds=30,
            ),
            actor_authorized=True,
        )
        assert preflight.binding is not None
        binding = preflight.binding
        corpus = corpus_for_eval_scenario(scenario, binding, target)
        compiled = compile_corpus_suite(corpus, target, "scenario")
        store = InMemoryEvalStore()
        await store.save_scenario(scenario, redact_json=app.redact_json)
        await store.save_corpus(corpus, redact_json=app.redact_json)
        request = EvalRunRequest(
            run_id="scenario-external-run",
            idempotency_key="sha256:" + "9" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id="scenario",
            suite_revision=compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=EvalRunInvocation(
                max_steps=binding.max_steps,
                limits=binding.operator_run_limits,
                cost_budget=binding.cost_budget,
                scenario=EvalScenarioRunInvocation(
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
                ),
            ),
        )
        await store.admit_run(request, redact_json=app.redact_json)
        claimed = await store.claim_run(target_key=target.key, lease_seconds=30)
        assert claimed is not None
        result = await run_compiled_eval_scenario(
            target,
            compiled,
            scenario,
            binding,
            store=store,
            claim=claimed.claim,
            max_concurrency=1,
            poll_seconds=0.001,
        )
        return result, operations, content

    result, operations, content = asyncio.run(exercise())

    trial = result.run.cases[0].trials[0]
    assert result.run.status == "passed", (
        trial.status,
        trial.code,
        trial.message,
        tuple(assertion.outcome for assertion in trial.assertions),
        len(operations.candidate_requests),
    )
    assert result.external_trials[0].native_run_id == "scenario-external-run"
    assert operations.envelopes[0].trial == result.external_trials[0]
    candidate = operations.candidate_requests[0]
    user = candidate.messages[0]
    assert [part.type for part in user.content] == ["text", "text", "file"]
    assert user.content[1].text == '{"count":2,"preserve":true}'
    resolved = candidate.options[RESOLVED_FILE_ATTACHMENTS_OPTION]
    assert len(resolved) == 1
    attached = next(iter(resolved.values()))
    assert attached["content_sha256"] == sha256(content).hexdigest()
    assert "data_base64" in attached


@pytest.mark.parametrize(
    ("provider_code", "operation_status", "published_status", "diagnostic"),
    [
        (
            "external_container_unknown",
            ProviderOperationStatus.UNAVAILABLE,
            "unavailable",
            EvalTrialDiagnosticCode.EXTERNAL_TARGET_UNKNOWN,
        ),
        (
            "external_container_incomplete",
            ProviderOperationStatus.UNAVAILABLE,
            "unavailable",
            EvalTrialDiagnosticCode.EXTERNAL_TARGET_INCOMPLETE,
        ),
        (
            "external_container_cancelled",
            ProviderOperationStatus.CANCELLED,
            "unavailable",
            EvalTrialDiagnosticCode.EXTERNAL_TARGET_CANCELLED,
        ),
        (
            "external_container_failed",
            ProviderOperationStatus.FAILED,
            "error",
            EvalTrialDiagnosticCode.EXTERNAL_TARGET_FAILED,
        ),
    ],
)
def test_external_dispositions_survive_native_publication(
    provider_code: str,
    operation_status: ProviderOperationStatus,
    published_status: str,
    diagnostic: EvalTrialDiagnosticCode,
) -> None:
    evidence_policy = EvaluationEvidencePolicySpec.standard()
    base = _target()
    identity = ExternalProcessTargetIdentityV1.create(
        body=base.body,
        evaluator_runtime_revision=base.evaluator_runtime_revision,
        target_implementation_revision=base.target_implementation_revision,
        runner_revision=base.runner_revision,
        environment_revision=base.environment_revision,
        reset_contract_revision=base.reset_contract_revision,
        evidence_policy_revision=evidence_policy.revision,
    )
    operations = _DispositionOperations(
        identity,
        error_code=provider_code,
        status=operation_status,
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(
        ExternalProcessModelProvider(identity=identity, operations=operations),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="external-candidate", model=EXTERNAL_PROCESS_PROTOCOL_VERSION)
    )
    target = CorpusTarget(
        key="external-agent",
        app=app,
        request_base=RunRequest(
            agent_name="external-candidate",
            messages=[],
            max_steps=1,
        ),
        application_release_id="external-release-one",
        evidence_policy=evidence_policy,
        external_process=identity,
    )
    suite = EvalSuiteSpec.create(
        id="external-suite",
        name="External suite",
        trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
    )
    case = EvalCaseSpec.create(
        id="external-case",
        suite_id=suite.id,
        name="External case",
        source=EvaluationSourceIdentityV1(
            application_release_id="source-release",
            app_manifest_schema_version="7",
            app_manifest_fingerprint="a" * 64,
            evidence_revision=_A,
        ),
        input=RunInputSpec(messages=(CorpusUserMessageSpec(text="Run this."),)),
        assertions=(FinalOutputEqualsAssertionSpec(id="answer", expected="Approved"),),
    )
    corpus = EvalCorpusDocument.create(
        target_key=target.key,
        evidence_policy=evidence_policy,
        suites=(suite,),
        cases=(case,),
    )

    result = asyncio.run(run_corpus_suite(target, corpus, suite.id))
    trial = result.run.cases[0].trials[0]

    assert trial.status == published_status
    assert trial.code is diagnostic
    comparison = compare_corpus_execution_results(result, result)
    assert comparison.cases[0].baseline_trial_diagnostic_codes == (diagnostic,)
    assert comparison.cases[0].current_trial_diagnostic_codes == (diagnostic,)


def test_sqlite_worker_restart_before_external_dispatch_preserves_native_identity(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        evidence_policy = EvaluationEvidencePolicySpec.standard()
        base = _target()
        identity = ExternalProcessTargetIdentityV1.create(
            body=base.body,
            evaluator_runtime_revision=base.evaluator_runtime_revision,
            target_implementation_revision=base.target_implementation_revision,
            runner_revision=base.runner_revision,
            environment_revision=base.environment_revision,
            reset_contract_revision=base.reset_contract_revision,
            evidence_policy_revision=evidence_policy.revision,
        )
        operations = _CapturingOperations(identity)
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ExternalProcessModelProvider(identity=identity, operations=operations),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="external-candidate", model=EXTERNAL_PROCESS_PROTOCOL_VERSION)
        )
        target = CorpusTarget(
            key="external-agent",
            app=app,
            request_base=RunRequest(
                agent_name="external-candidate",
                messages=[],
                max_steps=1,
            ),
            application_release_id="external-release-one",
            evidence_policy=evidence_policy,
            external_process=identity,
        )
        suite = EvalSuiteSpec.create(
            id="external-suite",
            name="External suite",
            trial_request=TrialRequestSpec(trials=1, timeout_seconds=30),
        )
        case = EvalCaseSpec.create(
            id="external-case",
            suite_id=suite.id,
            name="External case",
            source=EvaluationSourceIdentityV1(
                application_release_id="source-release",
                app_manifest_schema_version="7",
                app_manifest_fingerprint="a" * 64,
                evidence_revision=_A,
            ),
            input=RunInputSpec(
                opaque_external_case_ref=OpaqueExternalCaseRefV1(
                    id="arena-scoped-case-alias",
                    revision=_F,
                )
            ),
            assertions=(FinalOutputEqualsAssertionSpec(id="answer", expected="Approved"),),
        )
        corpus = EvalCorpusDocument.create(
            target_key=target.key,
            evidence_policy=evidence_policy,
            suites=(suite,),
            cases=(case,),
        )
        compiled = compile_corpus_suite(corpus, target, suite.id)
        invocation = EvalRunInvocation()
        registry = explicit_eval_target_registry(target)
        effective_target = target_for_eval_invocation(target, invocation)
        prepared = await registry.prepare_execution_profile(
            target.key,
            effective_target=effective_target,
        )
        request = EvalRunRequest(
            run_id="sqlite-external-run",
            idempotency_key="sha256:" + "8" * 64,
            corpus_revision=corpus.revision,
            target_key=target.key,
            suite_id=suite.id,
            suite_revision=compiled.run_contract.suite_revision,
            max_concurrency=1,
            invocation=invocation.model_copy(
                update={"execution_profile": prepared.binding},
                deep=True,
            ),
        )
        path = tmp_path / "external-evals.sqlite"
        admitted_store = SQLiteEvalStore(path)
        await admitted_store.save_corpus(corpus, redact_json=app.redact_json)
        await admitted_store.admit_run(request, redact_json=app.redact_json)
        await admitted_store.close()

        restarted_store = SQLiteEvalStore(path)
        coordinator = EvalRunCoordinator(
            EvalsConfig(
                target=target,
                store=restarted_store,
                lease_seconds=5,
                poll_interval_seconds=0.01,
                shutdown_grace_seconds=1.0,
            )
        )
        coordinator.start()
        try:
            for _ in range(500):
                record = await restarted_store.load_run(request.run_id)
                assert record is not None
                if record.status in {
                    EvalRunStatus.COMPLETED,
                    EvalRunStatus.FAILED,
                    EvalRunStatus.CANCELLED,
                }:
                    break
                await asyncio.sleep(0.01)
            assert record.status is EvalRunStatus.COMPLETED
            result = await restarted_store.load_result(request.run_id)
            assert result is not None
            assert result.external_trials[0].native_run_id == request.run_id
            assert operations.trials == list(result.external_trials)
        finally:
            await coordinator.stop()
            await restarted_store.close()

    asyncio.run(exercise())
