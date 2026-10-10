"""Host-side execution-profile prediction mirrors Runtime's admission rules."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest

from cayu import (
    CandidateExecutionProfile,
    ExecutionProfileAdmissionBoundary,
    ExecutionProfilePredictionOutcome,
    SessionExecutionProfiles,
    predict_execution_profile_admission,
    session_execution_profiles,
)
from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.budgets import BudgetLimit, BudgetPolicy
from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.egress import (
    EgressAuthorityBindingIdentity,
    EgressAuthorityCutoverStrategy,
    HttpEgressPolicy,
    build_egress_authority_identity,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime.build_provenance import (
    RuntimeBuildArtifactKind,
    RuntimeBuildProvenance,
    RuntimeBuildProvenanceOrigin,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.runtime.execution_profiles import (
    EXECUTION_PROFILE_METADATA_KEY,
    ExecutionProfileComponentClass,
    ExecutionProfileDecisionKind,
    ExecutionProfileIdentity,
    ExecutionProfileMismatchError,
    build_execution_profile_identity,
    execution_profile_with_egress_authority,
)
from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
    execution_profile_session_metadata,
)
from cayu.sessions.base import InMemorySessionStore, ModelTarget, ResumeRequest, RunRequest
from cayu.sessions.checkpoints import ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY
from cayu.sessions.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
    SessionInvocation,
)
from cayu.sessions.records import Session
from cayu.tools.base import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyRequest, ToolPolicyResult

Component = ExecutionProfileComponentClass
Outcome = ExecutionProfilePredictionOutcome
Decision = ExecutionProfileDecisionKind
Boundary = ExecutionProfileAdmissionBoundary


def _provenance(digest: str = "a") -> RuntimeBuildProvenance:
    return RuntimeBuildProvenance.from_artifact_digest(
        origin=RuntimeBuildProvenanceOrigin.EXPLICIT_MANIFEST,
        artifact_kind=RuntimeBuildArtifactKind.OTHER,
        artifact_digest=digest * 64,
    )


def _profile(
    *,
    prompt: str = "system",
    model: str = "fake-model",
    build: str = "a",
    max_steps: int = 16,
    egress_authority=None,
) -> ExecutionProfileIdentity:
    return build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="test",
        provider_name="fake",
        model=model,
        durable_system_prompt=prompt,
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{'c' * 64}",
        finalization={"max_steps": max_steps},
        egress_authority=egress_authority,
        runtime_build_provenance=_provenance(build),
    )


def _egress(*, allow_post: bool):
    endpoints = [("GET", "/v1/items")]
    if allow_post:
        endpoints.append(("POST", "/v1/items"))
    policy = HttpEgressPolicy(
        name="provider",
        allowed_hosts=("api.example.com",),
        allowed_endpoints=endpoints,
    )
    return build_egress_authority_identity(
        policies={policy.name: policy},
        bindings=(
            EgressAuthorityBindingIdentity(
                destination="api.example.com",
                policy_name=policy.name,
                credential_kind="opaque_bearer",
                credential_authority_fingerprint="1" * 64,
            ),
        ),
        generation=1,
        authority_source="trusted-app",
        authority_scope="session",
        policy_version="v1",
        runner_kind="docker",
        cutover_strategy=EgressAuthorityCutoverStrategy.FRESH_AUTHORITY_PATH,
    )


def test_resumed_sessions_keep_their_system_projection_at_every_boundary() -> None:
    expected = _profile(prompt="v1 prompt")
    candidate = _profile(prompt="v2 prompt")

    for boundary in ExecutionProfileAdmissionBoundary:
        prediction = predict_execution_profile_admission(
            expected,
            candidate,
            boundary=boundary,
            application_policy_configured=False,
        )
        assert prediction.outcome is Outcome.EXACT_REUSE
        assert prediction.admits is True
        assert prediction.possible_decisions == (Decision.EXACT_REUSE,)
        assert prediction.changed_component_classes == ()
        assert Component.DURABLE_SYSTEM_PROJECTION in prediction.retained_component_classes
        assert prediction.expected_fingerprint == expected.fingerprint
        assert prediction.candidate_fingerprint == candidate.fingerprint


def test_recorded_invocation_semantics_are_retained_only_for_continuation() -> None:
    expected = _profile(max_steps=7)
    candidate = _profile(max_steps=16)

    continuation = predict_execution_profile_admission(
        expected, candidate, boundary="continuation", application_policy_configured=False
    )
    resume = predict_execution_profile_admission(
        expected, candidate, boundary="resume", application_policy_configured=True
    )

    assert continuation.outcome is Outcome.EXACT_REUSE
    assert Component.FINALIZATION in continuation.retained_component_classes
    assert resume.changed_component_classes == (Component.FINALIZATION,)
    assert resume.authority_changed is True
    assert resume.outcome is Outcome.POLICY_DEPENDENT
    assert resume.admits is False
    assert resume.possible_decisions == (Decision.MIGRATION_REQUIRED, Decision.REJECTED)


def test_runtime_only_change_depends_on_the_application_policy() -> None:
    expected = _profile(build="a")
    candidate = _profile(build="b")

    continuation = predict_execution_profile_admission(expected, candidate, boundary="continuation")
    unknown = predict_execution_profile_admission(expected, candidate, boundary="resume")
    without_policy = predict_execution_profile_admission(
        expected, candidate, boundary="resume", application_policy_configured=False
    )

    assert continuation.outcome is Outcome.REJECTED
    assert continuation.admits is False
    assert continuation.changed_component_classes == (Component.RUNTIME,)
    assert unknown.outcome is Outcome.POLICY_DEPENDENT
    assert unknown.admits is None
    assert unknown.possible_decisions == (
        Decision.COMPATIBLE_REUSE,
        Decision.MIGRATION_REQUIRED,
        Decision.REJECTED,
    )
    assert without_policy.outcome is Outcome.REJECTED
    assert without_policy.possible_decisions == (Decision.REJECTED,)


def test_only_strictly_narrower_egress_can_be_adopted_without_intent() -> None:
    wide = _profile(egress_authority=_egress(allow_post=True))
    narrow = execution_profile_with_egress_authority(wide, _egress(allow_post=False))

    narrowing = predict_execution_profile_admission(wide, narrow, boundary="resume")
    widening = predict_execution_profile_admission(narrow, wide, boundary="resume")

    assert narrowing.changed_component_classes == (Component.EGRESS_AUTHORITY,)
    assert narrowing.egress_authority_change == "narrower"
    assert narrowing.admits is None
    assert narrowing.possible_decisions[0] is Decision.ADOPTED
    assert widening.egress_authority_change == "wider"
    assert widening.admits is False
    assert Decision.ADOPTED not in widening.possible_decisions


def test_a_candidate_for_another_target_is_not_comparable() -> None:
    prediction = predict_execution_profile_admission(
        _profile(model="fake-model"),
        _profile(model="other-model"),
        boundary="continuation",
    )

    assert prediction.outcome is Outcome.NOT_COMPARABLE
    assert prediction.admits is None
    assert prediction.possible_decisions == ()
    assert prediction.changed_component_classes == ()


def test_prediction_rejects_invalid_policy_flags_and_boundaries() -> None:
    profile = _profile()
    with pytest.raises(TypeError):
        predict_execution_profile_admission(
            profile,
            profile,
            boundary="resume",
            application_policy_configured=1,  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError):
        predict_execution_profile_admission(profile, profile, boundary="fork")


def _session(*, run_epoch: int, metadata: dict | None = None) -> Session:
    # The projection reads only id, run_epoch and metadata; the copy skips the
    # record's cross-field checks, which the synthetic profile does not satisfy.
    return Session(
        id="s",
        agent_name="assistant",
        provider_name="fake",
        model="fake-model",
        causal_budget_id="s",
        invocation=SessionInvocation(
            origin=InvocationOrigin(trust=InvocationOriginTrust.UNATTRIBUTED),
            root_invocation_id="00000000-0000-4000-8000-000000000000",
            root_session_id="s",
            source=SessionExecutionSource.SDK_RUN,
        ),
    ).model_copy(update={"run_epoch": run_epoch, "metadata": metadata or {}})


def _active(profile: ExecutionProfileIdentity, *, session_id: str = "s", run_epoch: int = 1):
    snapshot = ActiveInvocationExecutionProfile(
        session_id=session_id, interaction_id="i", run_epoch=run_epoch, profile=profile
    )
    return {ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: snapshot.model_dump(mode="json")}


def test_session_projection_reports_absent_and_malformed_records() -> None:
    assert session_execution_profiles(_session(run_epoch=0), None) == SessionExecutionProfiles()
    damaged = session_execution_profiles(
        _session(run_epoch=1, metadata={EXECUTION_PROFILE_METADATA_KEY: {"record_type": "x"}}),
        {ACTIVE_INVOCATION_EXECUTION_PROFILE_CHECKPOINT_KEY: {"profile": "bad"}},
    )
    assert damaged.expected is None
    assert damaged.active_invocation is None
    assert damaged.boundary is None
    assert damaged.issues == ("expected_profile_invalid", "active_invocation_profile_invalid")


def test_session_projection_reports_the_boundary_runtime_meets_next() -> None:
    profile = _profile()
    metadata = {EXECUTION_PROFILE_METADATA_KEY: execution_profile_session_metadata(profile)}

    released = session_execution_profiles(
        _session(run_epoch=2, metadata=metadata), _active(profile)
    )
    assert released.active_invocation is not None
    assert released.active_invocation.released is True
    assert released.boundary is Boundary.RESUME

    for pending_record in (
        "pending_tool_approval",
        "provider_operation_pending_resolution_disposition",
    ):
        pending = session_execution_profiles(
            _session(run_epoch=2, metadata=metadata),
            {**_active(profile), pending_record: {"id": "a"}},
        )
        assert pending.boundary is Boundary.CONTINUATION
    # A model-completion stage is stored outside the checkpoint.
    staged = session_execution_profiles(
        _session(run_epoch=2, metadata=metadata), _active(profile), model_completion_pending=True
    )
    assert staged.boundary is Boundary.CONTINUATION

    # An unreleased invocation is running or is continued by recovery.
    open_invocation = session_execution_profiles(
        _session(run_epoch=1, metadata=metadata), _active(profile)
    )
    assert open_invocation.active_invocation is not None
    assert open_invocation.active_invocation.released is False
    assert open_invocation.boundary is Boundary.CONTINUATION

    for session, checkpoint in (
        (_session(run_epoch=2, metadata=metadata), _active(profile, session_id="other")),
        (_session(run_epoch=5, metadata=metadata), _active(profile)),
    ):
        conflicting = session_execution_profiles(session, checkpoint)
        assert conflicting.issues == ("active_invocation_epoch_mismatch",)
        assert conflicting.active_invocation is None
        assert conflicting.boundary is None
        assert conflicting.expected == profile


class _EffectTool(Tool):
    def __init__(self, description: str) -> None:
        self.spec = ToolSpec(
            name="side_effect",
            description=description,
            input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
            effect=ToolEffect.EXTERNAL,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="tests:prediction-effect", behavior_version="1", implementation_version="1"
            ),
        )
        super().__init__()
        self.calls: list[dict[str, object]] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content="done")


class _RequireApproval(ToolPolicy):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:prediction-approval", behavior_version="1", implementation_version="1"
        )

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)


def _release(
    store: InMemorySessionStore,
    *,
    prompt: str,
    tool_description: str,
    script: list[ModelStreamEvent],
) -> tuple[CayuApp, _EffectTool]:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(ScriptedModelProvider(script, name="fake"), default=True)
    tool = _EffectTool(tool_description)
    app.register_agent(
        AgentSpec(name="assistant", model="fake-model", system_prompt=prompt),
        tools=[tool],
        tool_policy=_RequireApproval(),
    )
    return app, tool


async def _collect(events: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in events]


def test_prediction_matches_runtime_for_a_paused_approval_across_releases() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        v1, _ = _release(
            store,
            prompt="v1 prompt",
            tool_description="Original effect.",
            script=[
                ModelStreamEvent.tool_call(
                    id="call-1", name="side_effect", arguments={"value": "x"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
        )
        paused = await _collect(
            v1.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="paused",
                    messages=[Message.text("user", "go")],
                    max_steps=7,
                )
            )
        )
        (requested,) = [
            event for event in paused if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
        ]
        approval = requested.payload["approval"]
        stored = await v1.inspect_session_execution_profiles("paused")
        assert stored.issues == ()
        assert stored.boundary is Boundary.CONTINUATION
        assert stored.expected is not None
        assert stored.active_invocation is not None
        assert stored.active_invocation.released is True
        assert stored.active_invocation.interaction_id == requested.interaction_id
        assert v1.execution_profile_policy_identity is None
        with pytest.raises(KeyError):
            await v1.inspect_session_execution_profiles("missing")

        finished = [ModelStreamEvent.completed({"finish_reason": "stop"})]
        tool_change, _ = _release(
            store, prompt="v1 prompt", tool_description="Changed effect.", script=finished
        )
        prompt_change, prompt_tool = _release(
            store, prompt="v2 prompt", tool_description="Original effect.", script=finished
        )
        target = ModelTarget(provider_name="fake", model="fake-model")

        rejected_candidate = await tool_change.inspect_candidate_execution_profile(
            "assistant", target=target
        )
        rejected = predict_execution_profile_admission(
            stored.active_invocation.profile,
            rejected_candidate.execution_profile,
            boundary="continuation",
        )
        assert rejected.outcome is Outcome.REJECTED
        approve = ToolApprovalRequest(
            session_id="paused",
            approval_id=approval["approval_id"],
            tool_round_id=approval["tool_round_id"],
            tool_call_id=approval["tool_call_id"],
            decision=ToolApprovalDecision.APPROVE,
        )
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(tool_change.resolve_tool_approval(approve))
        assert caught.value.changed_component_classes == rejected.changed_component_classes

        candidate = await prompt_change.inspect_candidate_execution_profile("assistant")
        assert type(candidate) is CandidateExecutionProfile
        assert (candidate.agent_name, candidate.provider_name, candidate.model) == (
            "assistant",
            "fake",
            "fake-model",
        )
        assert candidate.environment_name is None
        predicted = predict_execution_profile_admission(
            stored.active_invocation.profile,
            candidate.execution_profile,
            boundary="continuation",
        )
        # The prompt and the run's recorded max_steps differ from the candidate,
        # but continuation keeps both from the paused invocation.
        assert predicted.outcome is Outcome.EXACT_REUSE
        assert (
            candidate.execution_profile.fingerprint != stored.active_invocation.profile.fingerprint
        )
        resumed = await _collect(prompt_change.resolve_tool_approval(approve))
        assert resumed[-1].type is EventType.SESSION_COMPLETED
        assert prompt_tool.calls == [{"value": "x"}]
        finished_profiles = await prompt_change.inspect_session_execution_profiles("paused")
        assert finished_profiles.boundary is Boundary.RESUME

    asyncio.run(scenario())


def _budget_release(
    store: InMemorySessionStore,
    *,
    budget_policy: BudgetPolicy | None = None,
    environment: Environment | None = None,
) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False, budget_policy=budget_policy)
    app.register_provider(
        ScriptedModelProvider(
            [[ModelStreamEvent.completed({"finish_reason": "stop"})]] * 2, name="fake"
        ),
        default=True,
    )
    if environment is not None:
        app.register_environment(environment, default=True)
    app.register_agent(AgentSpec(name="assistant", model="fake-model", system_prompt="p"))
    return app


def _causal_limit(key: str) -> BudgetLimit:
    return BudgetLimit(
        scope="causal",
        key=key,
        max_estimated_cost=Decimal("100"),
        pricing=PriceBook(
            prices=(
                ModelPrice.fixed(
                    provider_name="fake",
                    model="fake-model",
                    match="exact",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("1"),
                ),
            )
        ),
    )


def test_prediction_resolves_the_candidate_for_the_session_causal_budget() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        v1 = _budget_release(store)
        for session_id, causal_budget_id in (("limited", "K"), ("unlimited", None)):
            await _collect(
                v1.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id=session_id,
                        causal_budget_id=causal_budget_id,
                        messages=[Message.text("user", "hi")],
                    )
                )
            )
        v2 = _budget_release(store, budget_policy=BudgetPolicy(limits=(_causal_limit("K"),)))
        target = ModelTarget(provider_name="fake", model="fake-model")
        policy_configured = v2.execution_profile_policy_identity is not None

        async def predict(session_id: str):
            session = await store.load(session_id)
            assert session is not None
            stored = await v2.inspect_session_execution_profiles(session_id)
            assert stored.boundary is Boundary.RESUME
            assert stored.expected is not None
            candidate = await v2.inspect_candidate_execution_profile(
                "assistant", target=target, causal_budget_id=session.causal_budget_id
            )
            assert candidate.causal_budget_id == session.causal_budget_id
            return predict_execution_profile_admission(
                stored.expected,
                candidate.execution_profile,
                boundary=stored.boundary,
                application_policy_configured=policy_configured,
            )

        limited = await predict("limited")
        assert limited.outcome is Outcome.REJECTED
        assert limited.changed_component_classes == (Component.APPLICATION_BUDGET_POLICY,)
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            await _collect(
                v2.resume(
                    ResumeRequest(session_id="limited", messages=[Message.text("user", "again")])
                )
            )
        assert caught.value.changed_component_classes == limited.changed_component_classes

        # The limit is keyed to another causal id, so this session is unaffected.
        unlimited = await predict("unlimited")
        assert unlimited.outcome is Outcome.EXACT_REUSE
        resumed = await _collect(
            v2.resume(
                ResumeRequest(session_id="unlimited", messages=[Message.text("user", "again")])
            )
        )
        assert resumed[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())


def test_a_session_without_an_environment_is_predicted_without_one() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        v1 = _budget_release(store)
        await _collect(
            v1.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="bare",
                    messages=[Message.text("user", "hi")],
                )
            )
        )
        session = await store.load("bare")
        assert session is not None and session.environment_name is None
        v2 = _budget_release(store, environment=Environment(EnvironmentSpec(name="workspace")))
        assert v2.default_environment_name == "workspace"
        stored = await v2.inspect_session_execution_profiles("bare")
        assert stored.expected is not None
        target = ModelTarget(provider_name="fake", model="fake-model")

        candidate = await v2.inspect_candidate_execution_profile(
            "assistant",
            environment_name=session.environment_name,
            target=target,
            causal_budget_id=session.causal_budget_id,
        )
        assert candidate.environment_name is None
        prediction = predict_execution_profile_admission(
            stored.expected, candidate.execution_profile, boundary="resume"
        )
        assert prediction.outcome is Outcome.EXACT_REUSE
        # A new run would get the default environment; this session does not.
        fresh = await v2.inspect_candidate_execution_profile(
            "assistant", environment_name=v2.default_environment_name, target=target
        )
        assert fresh.environment_name == "workspace"
        assert fresh.execution_profile.fingerprint != candidate.execution_profile.fingerprint
        resumed = await _collect(
            v2.resume(ResumeRequest(session_id="bare", messages=[Message.text("user", "again")]))
        )
        assert resumed[-1].type is EventType.SESSION_COMPLETED

    asyncio.run(scenario())


class _UnreadSessionStore(InMemorySessionStore):
    async def load(self, session_id: str):
        raise AssertionError(f"inspection loaded session {session_id!r}")

    async def load_checkpoint(self, session_id: str):
        raise AssertionError(f"inspection loaded checkpoint {session_id!r}")


def test_candidate_and_run_inspection_never_read_the_session_store() -> None:
    async def scenario() -> None:
        app = _budget_release(_UnreadSessionStore())
        candidate = await app.inspect_candidate_execution_profile("assistant")
        fingerprint = await app.inspect_run_execution_profile(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "hi")])
        )
        assert len(candidate.execution_profile.fingerprint) == len(fingerprint) == 64

    asyncio.run(scenario())


class _StagedSessionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.staged_session_ids: set[str] = set()

    async def load_active_model_completion_stage(self, session_id: str):
        if session_id in self.staged_session_ids:
            return object()
        return await super().load_active_model_completion_stage(session_id)


def test_a_session_with_an_active_model_completion_stage_continues() -> None:
    async def scenario() -> None:
        store = _StagedSessionStore()
        app = _budget_release(store)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="staged",
                    messages=[Message.text("user", "hi")],
                )
            )
        )
        released = await app.inspect_session_execution_profiles("staged")
        assert released.boundary is Boundary.RESUME
        assert released.active_invocation is not None and released.active_invocation.released
        # Runtime continues the invocation that owns the stage, so the active
        # invocation profile, not the expected one, must be reused exactly.
        store.staged_session_ids.add("staged")
        staged = await app.inspect_session_execution_profiles("staged")
        assert staged.boundary is Boundary.CONTINUATION

    asyncio.run(scenario())
