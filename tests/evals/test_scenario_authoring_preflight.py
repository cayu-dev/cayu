from __future__ import annotations

import asyncio
import hashlib
from decimal import Decimal

import pytest

from cayu import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
    DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    ArtifactScope,
    CayuApp,
    CorpusExecutionLimits,
    CorpusTarget,
    Environment,
    EnvironmentSpec,
    EvalRunCostBudget,
    EvalScenarioDocumentV2,
    EvalScenarioDraftV2,
    LocalArtifactStore,
    RunLimits,
    RunRequest,
    ScenarioApprovalCheckpointEventV2,
    ScenarioArtifactMaterializationError,
    ScenarioArtifactRequirementV2,
    ScenarioFilePartV2,
    ScenarioInitialInputEventV2,
    ScenarioInputV2,
    ScenarioLaunchDiagnosticCode,
    ScenarioLaunchSettingsV2,
    ScenarioQueuedInputEventV2,
    ScenarioSecretRequirementV2,
    ScenarioTextPartV2,
    ScenarioUserMessageV2,
    ScriptedModelProvider,
    StaticVault,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
    compile_eval_scenario_draft,
    materialize_eval_scenario_artifact_fixture,
    preflight_eval_scenario,
    validate_expected_scenario_revision,
)


class _ReviewTool(Tool):
    spec = ToolSpec(
        name="review_action",
        description="Perform a reviewed action.",
        input_schema={"type": "object", "properties": {}},
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        return ToolResult(content="reviewed")


def _target(
    *,
    artifact_store: LocalArtifactStore | None = None,
    vault: StaticVault | None = None,
    approval_required: bool | None = None,
    max_file_attachment_bytes: int = DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    max_total_file_attachment_bytes: int = DEFAULT_MAX_TOTAL_FILE_ATTACHMENT_BYTES,
    max_file_attachments_per_request: int = DEFAULT_MAX_FILE_ATTACHMENTS_PER_REQUEST,
) -> CorpusTarget:
    app = CayuApp(
        enable_logging=False,
        max_file_attachment_bytes=max_file_attachment_bytes,
        max_total_file_attachment_bytes=max_total_file_attachment_bytes,
        max_file_attachments_per_request=max_file_attachments_per_request,
    )
    app.register_provider(ScriptedModelProvider([]), default=True)
    agent = AgentSpec(name="assistant", model="scenario-model")
    if approval_required is None:
        app.register_agent(agent)
    elif approval_required:
        app.register_agent(
            agent,
            tools=[_ReviewTool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
    else:
        app.register_agent(agent, tools=[_ReviewTool()])
    environment_name = None
    if artifact_store is not None or vault is not None:
        environment_name = "files"
        app.register_environment(
            Environment(
                EnvironmentSpec(name=environment_name),
                artifact_store=artifact_store,
                vault=vault,
            ),
            default=True,
        )
    return CorpusTarget(
        key="assistant.default",
        app=app,
        request_base=RunRequest(
            agent_name="assistant",
            messages=[],
            environment_name=environment_name,
            max_steps=8,
        ),
        application_release_id="release-current",
        limits=CorpusExecutionLimits(
            max_trials=4,
            max_concurrency=2,
            max_timeout_seconds=600,
        ),
    )


def _input(*parts) -> ScenarioInputV2:
    return ScenarioInputV2.create((ScenarioUserMessageV2.create(parts),))


def _scenario(
    *,
    artifacts: tuple[ScenarioArtifactRequirementV2, ...] = (),
    extra_events=(),
    secrets: tuple[ScenarioSecretRequirementV2, ...] = (),
) -> EvalScenarioDocumentV2:
    parts = [ScenarioTextPartV2(text="Review the retained request.")]
    parts.extend(
        ScenarioFilePartV2(artifact_requirement_id=requirement.id) for requirement in artifacts
    )
    events = (
        ScenarioInitialInputEventV2(
            sequence=0,
            id="initial",
            input=_input(*parts),
        ),
        *tuple(
            event.model_copy(update={"sequence": index})
            for index, event in enumerate(extra_events, start=1)
        ),
    )
    return EvalScenarioDocumentV2.create(
        id="retained-request",
        target_key="assistant.default",
        name="Retained request",
        events=events,
        artifact_requirements=artifacts,
        secret_requirements=secrets,
    )


def test_draft_compilation_is_canonical_and_revision_checked() -> None:
    scenario = _scenario()
    draft = EvalScenarioDraftV2.from_scenario(scenario)

    assert compile_eval_scenario_draft(draft) == scenario
    assert validate_expected_scenario_revision(scenario, scenario.revision) == scenario
    with pytest.raises(ValueError, match="changed after the reviewed revision"):
        validate_expected_scenario_revision(scenario, "sha256:" + "0" * 64)


def test_simple_scenario_preflight_freezes_current_target_bounds() -> None:
    result = asyncio.run(
        preflight_eval_scenario(
            _scenario(),
            _target(),
            ScenarioLaunchSettingsV2(
                trials=2,
                max_concurrency=2,
                timeout_seconds=60,
                max_steps=3,
            ),
            actor_authorized=True,
        )
    )

    assert result.ready is True
    assert result.diagnostics == ()
    assert result.binding is not None
    assert result.binding.trials == 2
    assert result.binding.max_steps == 3
    assert result.binding.approval_behavior == "fresh_decision"
    assert result.binding.revision.startswith("sha256:")


def test_partial_run_limits_preserve_unselected_target_authority() -> None:
    target = _target()
    target = target.model_copy(
        update={
            "request_base": target.request_base.model_copy(
                update={
                    "limits": RunLimits(
                        max_total_tokens=10_000,
                        max_tool_calls=20,
                        scope="run",
                    )
                }
            )
        }
    )

    result = asyncio.run(
        preflight_eval_scenario(
            _scenario(),
            target,
            ScenarioLaunchSettingsV2(
                limits=RunLimits(max_tool_calls=5, scope="run"),
            ),
            actor_authorized=True,
        )
    )

    assert result.ready is True
    assert result.binding is not None
    assert result.binding.target_limits == RunLimits(
        max_total_tokens=10_000,
        max_tool_calls=20,
        scope="run",
    )
    assert result.binding.operator_run_limits == RunLimits(max_tool_calls=5, scope="run")

    broadened = asyncio.run(
        preflight_eval_scenario(
            _scenario(),
            target,
            ScenarioLaunchSettingsV2(
                limits=RunLimits(max_tool_calls=21, scope="run"),
            ),
            actor_authorized=True,
        )
    )
    assert [item.code for item in broadened.diagnostics] == [
        ScenarioLaunchDiagnosticCode.EXECUTION_LIMIT_EXCEEDED
    ]


def test_run_contraction_preserves_session_scoped_target_authority() -> None:
    target = _target()
    target_limits = RunLimits(
        max_total_tokens=10_000,
        max_tool_calls=20,
        scope="session",
    )
    target = target.model_copy(
        update={"request_base": target.request_base.model_copy(update={"limits": target_limits})}
    )
    operator_limits = RunLimits(max_tool_calls=5, scope="run")

    result = asyncio.run(
        preflight_eval_scenario(
            _scenario(),
            target,
            ScenarioLaunchSettingsV2(limits=operator_limits),
            actor_authorized=True,
        )
    )

    assert result.ready is True
    assert result.binding is not None
    assert result.binding.target_limits == target_limits
    assert result.binding.operator_run_limits == operator_limits


def test_preflight_requires_current_pricing_for_an_operator_cost_bound() -> None:
    result = asyncio.run(
        preflight_eval_scenario(
            _scenario(),
            _target(),
            ScenarioLaunchSettingsV2(
                cost_budget=EvalRunCostBudget(
                    max_estimated_cost=Decimal("0.50"),
                    currency="USD",
                )
            ),
            actor_authorized=True,
        )
    )

    assert [item.code for item in result.diagnostics] == [
        ScenarioLaunchDiagnosticCode.PRICING_UNAVAILABLE
    ]


def test_preflight_reports_independent_current_authority_gaps() -> None:
    scenario = _scenario(
        extra_events=(
            ScenarioApprovalCheckpointEventV2(
                sequence=1,
                id="approval",
                tool_name="unavailable_tool",
                occurrence=1,
            ),
        ),
        secrets=(
            ScenarioSecretRequirementV2(
                id="test-account",
                usage="tool",
                purpose="Use the current test account.",
            ),
        ),
    )
    result = asyncio.run(
        preflight_eval_scenario(
            scenario,
            _target(),
            actor_authorized=False,
        )
    )

    assert result.ready is False
    assert result.binding is None
    assert {item.code for item in result.diagnostics} == {
        ScenarioLaunchDiagnosticCode.ACTOR_AUTHORITY_UNAVAILABLE,
        ScenarioLaunchDiagnosticCode.APPROVAL_TOOL_UNAVAILABLE,
        ScenarioLaunchDiagnosticCode.SECRET_REFERENCE_UNAVAILABLE,
    }
    assert (
        next(
            item for item in result.diagnostics if item.code == "approval_tool_unavailable"
        ).event_id
        == "approval"
    )


def test_preflight_binds_named_secrets_without_resolving_or_publishing_values() -> None:
    scenario = _scenario(
        secrets=(
            ScenarioSecretRequirementV2(
                id="test-account",
                usage="tool",
                purpose="Use the current test account.",
            ),
        ),
    )
    secret_value = "must-never-enter-preflight"
    result = asyncio.run(
        preflight_eval_scenario(
            scenario,
            _target(vault=StaticVault({"test-account": secret_value})),
            actor_authorized=True,
        )
    )

    assert result.ready is True
    assert result.binding is not None
    assert result.binding.secrets[0].requirement_id == "test-account"
    assert result.binding.secrets[0].usage == "tool"
    assert secret_value not in result.model_dump_json()
    assert "static:test-account" not in result.model_dump_json()


def test_preflight_requires_current_fresh_approval_policy_coverage() -> None:
    scenario = _scenario(
        extra_events=(
            ScenarioApprovalCheckpointEventV2(
                sequence=1,
                id="approval",
                tool_name="review_action",
                occurrence=1,
            ),
        ),
    )

    allowed = asyncio.run(
        preflight_eval_scenario(
            scenario,
            _target(approval_required=False),
            actor_authorized=True,
        )
    )
    required = asyncio.run(
        preflight_eval_scenario(
            scenario,
            _target(approval_required=True),
            actor_authorized=True,
        )
    )

    assert [item.code for item in allowed.diagnostics] == [
        ScenarioLaunchDiagnosticCode.APPROVAL_POLICY_SELECTION_REQUIRED
    ]
    assert allowed.diagnostics[0].event_id == "approval"
    assert required.ready is True


def test_preflight_materializes_session_artifact_as_idempotent_environment_fixture(
    tmp_path,
) -> None:
    async def exercise():
        content = b"retained production attachment"
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="scenario-files")
        source = await store.put_bytes(
            content,
            filename="request.txt",
            content_type="text/plain",
            scope=ArtifactScope.SESSION,
            session_id="production-session",
            environment_name="files",
        )
        requirement = ScenarioArtifactRequirementV2(
            id="request-file",
            source="artifact_reference",
            reference=source.id,
            content_sha256=hashlib.sha256(content).hexdigest(),
            filename=source.filename,
            content_type=source.content_type,
            size_bytes=source.size_bytes,
        )
        target = _target(artifact_store=store)
        scenario = _scenario(artifacts=(requirement,))
        before = await preflight_eval_scenario(
            scenario,
            target,
            actor_authorized=True,
        )
        first = await materialize_eval_scenario_artifact_fixture(
            scenario,
            target,
            requirement.id,
        )
        second = await materialize_eval_scenario_artifact_fixture(
            scenario,
            target,
            requirement.id,
        )
        repeated = await materialize_eval_scenario_artifact_fixture(
            first.scenario,
            target,
            requirement.id,
        )
        after = await preflight_eval_scenario(
            first.scenario,
            target,
            actor_authorized=True,
        )
        fixture = await store.read_bytes(first.artifact_id)
        return before, first, second, repeated, after, fixture, content

    before, first, second, repeated, after, fixture, content = asyncio.run(exercise())
    assert [item.code for item in before.diagnostics] == [
        ScenarioLaunchDiagnosticCode.ARTIFACT_BINDING_REQUIRED
    ]
    assert first.artifact_id == second.artifact_id
    assert first.artifact_id == repeated.artifact_id
    assert first.scenario == second.scenario
    assert first.scenario == repeated.scenario
    assert first.scenario.revision != _scenario().revision
    assert fixture.content == content
    assert fixture.metadata.scope is ArtifactScope.ENVIRONMENT
    assert after.ready is True
    assert after.binding is not None
    assert after.binding.artifacts[0].artifact_id == first.artifact_id


def test_preflight_accepts_a_reviewed_artifact_selection_without_mutating_scenario(
    tmp_path,
) -> None:
    async def exercise():
        content = b"published fixture"
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="selected-files")
        fixture = await store.put_bytes(
            content,
            filename="fixture.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            agent_name="assistant",
            environment_name="files",
        )
        requirement = ScenarioArtifactRequirementV2(
            id="fixture",
            source="fixture_digest",
            content_sha256=hashlib.sha256(content).hexdigest(),
            filename=fixture.filename,
            content_type=fixture.content_type,
            size_bytes=fixture.size_bytes,
        )
        scenario = _scenario(artifacts=(requirement,))
        result = await preflight_eval_scenario(
            scenario,
            _target(artifact_store=store),
            ScenarioLaunchSettingsV2(
                artifact_references={requirement.id: fixture.id},
            ),
            actor_authorized=True,
        )
        return scenario, result, fixture

    scenario, result, fixture = asyncio.run(exercise())
    assert scenario.artifact_requirements[0].reference is None
    assert result.ready is True
    assert result.binding is not None
    assert result.binding.artifacts[0].artifact_id == fixture.id


def test_preflight_counts_repeated_file_parts_per_runtime_request(tmp_path) -> None:
    async def exercise():
        content = b"one retained file"
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="repeated-files")
        fixture = await store.put_bytes(
            content,
            filename="fixture.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            agent_name="assistant",
            environment_name="files",
        )
        requirement = ScenarioArtifactRequirementV2(
            id="fixture",
            source="fixture_digest",
            reference=None,
            content_sha256=hashlib.sha256(content).hexdigest(),
            filename=fixture.filename,
            content_type=fixture.content_type,
            size_bytes=fixture.size_bytes,
        )
        repeated_file = ScenarioFilePartV2(artifact_requirement_id=requirement.id)
        scenario = EvalScenarioDocumentV2.create(
            id="repeated-file",
            target_key="assistant.default",
            name="Repeated file",
            events=(
                ScenarioInitialInputEventV2(
                    sequence=0,
                    id="initial",
                    input=_input(repeated_file, repeated_file),
                ),
            ),
            artifact_requirements=(requirement,),
        )
        count_result = await preflight_eval_scenario(
            scenario,
            _target(
                artifact_store=store,
                max_file_attachments_per_request=1,
            ),
            ScenarioLaunchSettingsV2(
                artifact_references={requirement.id: fixture.id},
            ),
            actor_authorized=True,
        )
        byte_result = await preflight_eval_scenario(
            scenario,
            _target(
                artifact_store=store,
                max_total_file_attachment_bytes=len(content),
            ),
            ScenarioLaunchSettingsV2(
                artifact_references={requirement.id: fixture.id},
            ),
            actor_authorized=True,
        )
        return count_result, byte_result

    count_result, byte_result = asyncio.run(exercise())
    for result in (count_result, byte_result):
        assert [item.code for item in result.diagnostics] == [
            ScenarioLaunchDiagnosticCode.EXECUTION_LIMIT_EXCEEDED
        ]
        assert result.diagnostics[0].event_id == "initial"


def test_preflight_applies_file_ceilings_independently_to_each_request(tmp_path) -> None:
    async def exercise():
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="staged-files")
        fixtures = []
        requirements = []
        for requirement_id, content in (
            ("initial-file", b"initial retained file"),
            ("queued-file", b"queued retained file"),
        ):
            fixture = await store.put_bytes(
                content,
                filename=f"{requirement_id}.txt",
                content_type="text/plain",
                scope=ArtifactScope.ENVIRONMENT,
                agent_name="assistant",
                environment_name="files",
            )
            fixtures.append(fixture)
            requirements.append(
                ScenarioArtifactRequirementV2(
                    id=requirement_id,
                    source="fixture_digest",
                    reference=None,
                    content_sha256=hashlib.sha256(content).hexdigest(),
                    filename=fixture.filename,
                    content_type=fixture.content_type,
                    size_bytes=fixture.size_bytes,
                )
            )
        scenario = EvalScenarioDocumentV2.create(
            id="staged-file",
            target_key="assistant.default",
            name="Staged file",
            events=(
                ScenarioInitialInputEventV2(
                    sequence=0,
                    id="initial",
                    input=_input(ScenarioFilePartV2(artifact_requirement_id=requirements[0].id)),
                ),
                ScenarioQueuedInputEventV2(
                    sequence=1,
                    id="queued",
                    delivery_mode="next_turn",
                    input=_input(ScenarioFilePartV2(artifact_requirement_id=requirements[1].id)),
                ),
            ),
            artifact_requirements=tuple(requirements),
        )
        return await preflight_eval_scenario(
            scenario,
            _target(
                artifact_store=store,
                max_file_attachments_per_request=1,
                max_total_file_attachment_bytes=max(item.size_bytes for item in requirements),
            ),
            ScenarioLaunchSettingsV2(
                artifact_references={
                    requirement.id: fixture.id
                    for requirement, fixture in zip(requirements, fixtures, strict=True)
                },
            ),
            actor_authorized=True,
        )

    result = asyncio.run(exercise())
    assert result.ready is True


def test_fixture_materialization_rejects_another_environment_namespace(tmp_path) -> None:
    async def exercise() -> None:
        content = b"other environment content"
        store = LocalArtifactStore(tmp_path / "artifacts", store_id="shared-files")
        source = await store.put_bytes(
            content,
            filename="fixture.txt",
            content_type="text/plain",
            scope=ArtifactScope.ENVIRONMENT,
            environment_name="other-environment",
        )
        requirement = ScenarioArtifactRequirementV2(
            id="fixture",
            source="artifact_reference",
            reference=source.id,
            content_sha256=hashlib.sha256(content).hexdigest(),
            filename=source.filename,
            content_type=source.content_type,
            size_bytes=source.size_bytes,
        )
        with pytest.raises(ScenarioArtifactMaterializationError) as raised:
            await materialize_eval_scenario_artifact_fixture(
                _scenario(artifacts=(requirement,)),
                _target(artifact_store=store),
                requirement.id,
            )
        assert raised.value.code is ScenarioLaunchDiagnosticCode.ARTIFACT_CONTENT_INCONSISTENT

    asyncio.run(exercise())
