from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any, cast

import pytest
from examples.agent_snapshot_stateful_evaluation import run_reference_experiment
from pydantic import ValidationError

from cayu import (
    AgentSnapshot,
    AgentSnapshotAccess,
    AgentSnapshotAuthorityRef,
    AgentSnapshotAuthorizationError,
    AgentSnapshotCaptureError,
    AgentSnapshotCaptureRequest,
    AgentSnapshotCompleteness,
    AgentSnapshotComponentCapture,
    AgentSnapshotComponentKind,
    AgentSnapshotComponentProvider,
    AgentSnapshotComponentRef,
    AgentSnapshotComponentSelector,
    AgentSnapshotConsistency,
    AgentSnapshotCoordinator,
    AgentSnapshotExecutionProfileComponent,
    AgentSnapshotExecutionProfileRef,
    AgentSnapshotGCRequest,
    AgentSnapshotLearningDisposition,
    AgentSnapshotLogicalRef,
    AgentSnapshotMaterialization,
    AgentSnapshotMaterializationCapability,
    AgentSnapshotMaterializationError,
    AgentSnapshotMaterializationOperation,
    AgentSnapshotMaterializationProgress,
    AgentSnapshotMaterializationRequest,
    AgentSnapshotMaterializedComponent,
    AgentSnapshotOverlayKind,
    AgentSnapshotOverlayRef,
    AgentSnapshotRedaction,
    AgentSnapshotResultBinding,
    AgentSnapshotStoreConflict,
    AgentSnapshotSubject,
    AgentSnapshotTerminalDisposition,
    AgentSnapshotTrialBinding,
    AgentSnapshotTrialStateMode,
    AgentSnapshotVerificationError,
    AgentSpec,
    CayuApp,
    InMemoryAgentSnapshotStore,
    MemoryStateRef,
    SQLiteAgentSnapshotStore,
    Trajectory,
    WorkspaceIdentity,
    WorkspacePathRevision,
    WorkspaceRevisionObservation,
    WorkspaceRevisionObservationStatus,
    agent_snapshot_from_json,
    agent_snapshot_to_json,
    app_body_snapshot_ref,
    build_execution_profile_identity,
    execution_profile_snapshot_ref,
    trajectory_snapshot_ref,
    workspace_snapshot_ref,
)


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def _async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


def _ref(
    value: str,
    *,
    revision: str | None = None,
    frontier: str | None = None,
    scope: str | None = None,
    source_ref: str | None = None,
) -> AgentSnapshotLogicalRef:
    return AgentSnapshotLogicalRef(
        fingerprint=_digest(value),
        revision=revision or f"revision:{value}",
        frontier=frontier,
        scope_fingerprint=scope,
        source_ref=source_ref,
    )


def _profile(value: str = "profile-v1") -> AgentSnapshotExecutionProfileRef:
    return AgentSnapshotExecutionProfileRef(
        schema_version=5,
        fingerprint=_digest(value),
        components=(
            AgentSnapshotExecutionProfileComponent(
                name="automatic_recall",
                fingerprint=_digest(f"{value}:recall"),
                availability="available",
            ),
            AgentSnapshotExecutionProfileComponent(
                name="execution_environment",
                fingerprint=_digest(f"{value}:environment"),
                availability="available",
            ),
            AgentSnapshotExecutionProfileComponent(
                name="provider_target",
                fingerprint=_digest(f"{value}:provider"),
                availability="available",
            ),
            AgentSnapshotExecutionProfileComponent(
                name="runtime",
                fingerprint=_digest(f"{value}:runtime"),
                availability="available",
            ),
        ),
    )


def _memory(scope: str, value: str = "memory-v1") -> MemoryStateRef:
    return MemoryStateRef.create(
        knowledge=_ref(
            f"{value}:knowledge",
            frontier="knowledge-change:41",
            scope=scope,
            source_ref="cayu-ref:knowledge-package:41",
        ),
        transcript_evidence=_ref(
            f"{value}:transcript",
            frontier="session-event:29",
            scope=scope,
        ),
        work_context=_ref(
            f"{value}:work-context",
            revision="checkpoint:8",
            scope=scope,
        ),
        recall_policy=_ref(f"{value}:recall-policy", scope=scope),
        admission_policy=_ref(f"{value}:admission-policy", scope=scope),
        context_projection_policy=_ref(f"{value}:projection-policy", scope=scope),
        recall_receipts=_ref(f"{value}:receipts", frontier="recall-receipt:17", scope=scope),
        context_exposures=_ref(f"{value}:exposures", frontier="context-exposure:14", scope=scope),
        index_readiness=_ref(f"{value}:index", frontier="index-readiness:22", scope=scope),
        learning_disposition=AgentSnapshotLearningDisposition.ISOLATED,
        limitations=("interaction_focus_not_requested",),
    )


def _component(
    kind: AgentSnapshotComponentKind,
    logical: AgentSnapshotLogicalRef,
    *,
    required: bool = True,
    materialization: AgentSnapshotMaterializationCapability = (
        AgentSnapshotMaterializationCapability.REFERENCE_ONLY
    ),
    consistency: AgentSnapshotConsistency = AgentSnapshotConsistency.FRONTIER_CONSISTENT,
    completeness: AgentSnapshotCompleteness = AgentSnapshotCompleteness.COMPLETE,
    provider_id: str | None = None,
) -> AgentSnapshotComponentRef:
    return AgentSnapshotComponentRef(
        kind=kind,
        provider_id=provider_id or f"test.{kind.value}.v1",
        logical=logical,
        consistency=consistency,
        completeness=completeness,
        redaction=AgentSnapshotRedaction.BOUNDED_PROJECTION,
        materialization=materialization,
        required=required,
    )


def _subject() -> AgentSnapshotSubject:
    return AgentSnapshotSubject(
        agent_id="agent-1",
        application_id="application-1",
        project_id="project-1",
        body_release=_ref("body-v1", source_ref="cayu-ref:body-release:v1"),
    )


def _snapshot(
    *,
    captured_at: datetime | None = None,
    capture_request_id: str = "capture-1",
    memory_value: str = "memory-v1",
    body_source_ref: str = "cayu-ref:body-release:v1",
    evaluator: AgentSnapshotAuthorityRef | None = None,
) -> AgentSnapshot:
    scope = _digest("scope-1")
    subject = _subject().model_copy(
        update={
            "body_release": _ref(
                "body-v1",
                source_ref=body_source_ref,
            )
        }
    )
    profile = _profile()
    memory = _memory(scope, memory_value)
    components = (
        _component(AgentSnapshotComponentKind.BODY, subject.body_release),
        _component(
            AgentSnapshotComponentKind.EXECUTION_PROFILE,
            _ref("profile-v1"),
        ),
        _component(
            AgentSnapshotComponentKind.MEMORY,
            AgentSnapshotLogicalRef(
                fingerprint=memory.fingerprint,
                frontier="memory-frontier:1",
                scope_fingerprint=scope,
                source_ref="cayu-ref:memory-package:v1",
            ),
            materialization=AgentSnapshotMaterializationCapability.RESTORABLE,
        ),
        _component(
            AgentSnapshotComponentKind.WORKSPACE,
            _ref("workspace-v1", scope=scope, source_ref="cayu-ref:workspace-package:v1"),
            materialization=AgentSnapshotMaterializationCapability.RESTORABLE,
        ),
    )
    return AgentSnapshot.create(
        capture_request_id=capture_request_id,
        captured_at=captured_at or datetime(2026, 8, 23, tzinfo=UTC),
        subject=subject,
        authority_scope_fingerprint=scope,
        execution_profile=profile,
        memory_state=memory,
        components=components,
        evaluator=evaluator,
    )


def _access(snapshot: AgentSnapshot) -> AgentSnapshotAccess:
    return AgentSnapshotAccess(
        snapshot=snapshot.ref,
        binding_id=snapshot.identity_binding.binding_id,
        authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
    )


def test_agent_snapshot_identity_is_deterministic_and_relocation_stable() -> None:
    first = _snapshot()
    recaptured = _snapshot(
        capture_request_id="capture-2",
        captured_at=datetime(2026, 8, 24, tzinfo=UTC),
        body_source_ref="cayu-ref:relocated-body-package:v1",
    )

    assert recaptured.capture_request_id != first.capture_request_id
    assert recaptured.captured_at != first.captured_at
    assert recaptured.subject.body_release.source_ref != first.subject.body_release.source_ref
    assert recaptured.fingerprint == first.fingerprint

    changed = _snapshot(memory_value="memory-v2")
    assert changed.fingerprint != first.fingerprint


def test_every_included_component_revision_changes_snapshot_identity() -> None:
    snapshot = _snapshot()

    for changed_component in snapshot.components:
        logical = changed_component.logical.model_copy(
            update={"revision": f"changed:{changed_component.kind.value}"}
        )
        components = tuple(
            component.model_copy(update={"logical": logical})
            if component.kind is changed_component.kind
            else component
            for component in snapshot.components
        )
        subject = (
            snapshot.subject.model_copy(update={"body_release": logical})
            if changed_component.kind is AgentSnapshotComponentKind.BODY
            else snapshot.subject
        )
        changed = AgentSnapshot.create(
            capture_request_id=snapshot.capture_request_id,
            captured_at=snapshot.captured_at,
            subject=subject,
            authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
            execution_profile=snapshot.execution_profile,
            memory_state=snapshot.memory_state,
            components=components,
        )
        assert changed.fingerprint != snapshot.fingerprint


def test_component_provider_and_transaction_group_change_snapshot_identity() -> None:
    snapshot = _snapshot()
    provider_changed_components = tuple(
        component.model_copy(update={"provider_id": "test.body.v2"})
        if component.kind is AgentSnapshotComponentKind.BODY
        else component
        for component in snapshot.components
    )
    provider_changed = AgentSnapshot.create(
        capture_request_id=snapshot.capture_request_id,
        captured_at=snapshot.captured_at,
        subject=snapshot.subject,
        authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
        execution_profile=snapshot.execution_profile,
        memory_state=snapshot.memory_state,
        components=provider_changed_components,
    )
    assert provider_changed.fingerprint != snapshot.fingerprint

    def transactional_snapshot(group: str) -> AgentSnapshot:
        return AgentSnapshot.create(
            capture_request_id=snapshot.capture_request_id,
            captured_at=snapshot.captured_at,
            subject=snapshot.subject,
            authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
            execution_profile=snapshot.execution_profile,
            memory_state=snapshot.memory_state,
            components=tuple(
                component.model_copy(
                    update={
                        "consistency": AgentSnapshotConsistency.TRANSACTIONAL,
                        "consistency_group": group,
                    }
                )
                for component in snapshot.components
            ),
        )

    assert (
        transactional_snapshot("transaction-a").fingerprint
        != transactional_snapshot("transaction-b").fingerprint
    )


@_async_test
async def test_snapshot_stores_reject_same_fingerprint_different_document(tmp_path) -> None:
    snapshot = _snapshot()
    relocated = _snapshot(
        capture_request_id="relocated-capture",
        captured_at=datetime(2026, 8, 24, tzinfo=UTC),
        body_source_ref="cayu-ref:relocated-body-package:v1",
    )
    changed_components = tuple(
        component.model_copy(update={"provider_id": "test.body.v2"})
        if component.kind is AgentSnapshotComponentKind.BODY
        else component
        for component in snapshot.components
    )
    colliding = snapshot.model_copy(update={"components": changed_components})

    for store in (
        InMemoryAgentSnapshotStore(),
        SQLiteAgentSnapshotStore(tmp_path / "collision.db"),
    ):
        assert await store.save_snapshot(snapshot) == snapshot
        assert (await store.save_snapshot(relocated)).fingerprint == snapshot.fingerprint
        with pytest.raises(AgentSnapshotStoreConflict, match="fingerprint"):
            await store.save_snapshot(colliding)


@_async_test
async def test_snapshot_stores_reject_key_document_substitution(tmp_path) -> None:
    first = _snapshot()
    second = _snapshot(memory_value="memory-v2")
    in_memory = InMemoryAgentSnapshotStore()
    sqlite = SQLiteAgentSnapshotStore(tmp_path / "key-document-substitution.db")
    for store in (in_memory, sqlite):
        await store.save_snapshot(first)
        await store.save_snapshot(second)

    in_memory._records[("snapshot", first.fingerprint)] = second
    with sqlite3.connect(sqlite.path) as connection:
        second_row = connection.execute(
            "SELECT document FROM cayu_agent_snapshot_records "
            "WHERE record_kind = 'snapshot' AND fingerprint = ?",
            (second.fingerprint,),
        ).fetchone()
        assert second_row is not None
        connection.execute(
            "UPDATE cayu_agent_snapshot_records SET document = ? "
            "WHERE record_kind = 'snapshot' AND fingerprint = ?",
            (second_row[0], first.fingerprint),
        )

    for store in (in_memory, sqlite):
        with pytest.raises(AgentSnapshotStoreConflict, match="fingerprint"):
            await store.load_snapshot(first.fingerprint)


@_async_test
async def test_materialization_rejects_substituted_starting_snapshot() -> None:
    requested = _snapshot()
    substituted = _snapshot(memory_value="memory-v2")

    class SubstitutingStore(InMemoryAgentSnapshotStore):
        async def get_snapshot(self, access: AgentSnapshotAccess) -> AgentSnapshot:
            if access.snapshot == requested.ref:
                return substituted
            return await super().get_snapshot(access)

    store = SubstitutingStore()
    await store.save_snapshot(requested)
    coordinator = AgentSnapshotCoordinator(_providers(substituted), store=store)
    request = AgentSnapshotMaterializationRequest(
        access=_access(requested),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )

    with pytest.raises(AgentSnapshotMaterializationError, match="Starting snapshot"):
        await coordinator.materialize(request)


@_async_test
async def test_materialization_requires_authorized_snapshot_access() -> None:
    snapshot = _snapshot()
    providers = _providers(snapshot)
    store = InMemoryAgentSnapshotStore()
    await store.save_snapshot(snapshot)
    coordinator = AgentSnapshotCoordinator(providers, store=store)
    request = AgentSnapshotMaterializationRequest(
        access=_access(snapshot).model_copy(update={"binding_id": "0" * 64}),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )

    with pytest.raises(AgentSnapshotAuthorizationError):
        await coordinator.materialize(request)

    assert sum(provider.materialize_calls for provider in providers) == 0


@_async_test
async def test_coordinator_rejects_substituted_store_save_results() -> None:
    first = _snapshot()
    second = _snapshot(memory_value="memory-v2")

    class SubstitutingSaveStore(InMemoryAgentSnapshotStore):
        replacement_snapshot: AgentSnapshot | None = second
        replacement_trial: AgentSnapshotTrialBinding | None = None
        replacement_loaded_trial: AgentSnapshotTrialBinding | None = None
        replacement_result: AgentSnapshotResultBinding | None = None

        async def save_snapshot(self, snapshot: AgentSnapshot) -> AgentSnapshot:
            if self.replacement_snapshot is not None:
                return self.replacement_snapshot
            return await super().save_snapshot(snapshot)

        async def save_trial(self, trial: AgentSnapshotTrialBinding) -> AgentSnapshotTrialBinding:
            if self.replacement_trial is not None:
                return self.replacement_trial
            return await super().save_trial(trial)

        async def load_trial(self, fingerprint: str) -> AgentSnapshotTrialBinding | None:
            if self.replacement_loaded_trial is not None:
                return self.replacement_loaded_trial
            return await super().load_trial(fingerprint)

        async def save_result(
            self, result: AgentSnapshotResultBinding
        ) -> AgentSnapshotResultBinding:
            if self.replacement_result is not None:
                return self.replacement_result
            return await super().save_result(result)

    store = SubstitutingSaveStore()
    coordinator = AgentSnapshotCoordinator(_providers(first), store=store)
    with pytest.raises(AgentSnapshotStoreConflict, match="Snapshot save"):
        await coordinator.capture(_capture_request(first))

    evaluator_fingerprint = _digest("evaluator-a")
    declared = _snapshot(
        evaluator=AgentSnapshotAuthorityRef(
            identity=AgentSnapshotLogicalRef(
                fingerprint=evaluator_fingerprint,
                revision="evaluator-a",
            )
        )
    )
    store.replacement_snapshot = None
    coordinator = AgentSnapshotCoordinator(_providers(declared), store=store)
    captured = await coordinator.capture(_capture_request(declared))
    request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    materialization = await coordinator.materialize(request)
    store.replacement_trial = AgentSnapshotTrialBinding.create(
        materialization=materialization,
        case_id="substituted-case",
        trial_id="trial-1",
        evaluator_fingerprint=evaluator_fingerprint,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    with pytest.raises(AgentSnapshotStoreConflict, match="Trial save"):
        await coordinator.begin_trial(
            materialization,
            case_id="requested-case",
            trial_id="trial-1",
            evaluator_fingerprint=evaluator_fingerprint,
        )

    store.replacement_trial = None
    trial = await coordinator.begin_trial(
        materialization,
        case_id="requested-case",
        trial_id="trial-1",
        evaluator_fingerprint=evaluator_fingerprint,
    )
    result = AgentSnapshotResultBinding.create(
        trial=trial,
        session_id="session-a",
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest("runtime-a"),
        eval_result_revision=_digest("eval-a"),
        recorded_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    store.replacement_loaded_trial = AgentSnapshotTrialBinding.create(
        materialization=materialization,
        case_id="substituted-loaded-case",
        trial_id="trial-1",
        evaluator_fingerprint=evaluator_fingerprint,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    with pytest.raises(AgentSnapshotMaterializationError, match="exact durable trial"):
        await coordinator.record_result(result)

    store.replacement_loaded_trial = None
    store.replacement_result = AgentSnapshotResultBinding.create(
        trial=trial,
        session_id="session-b",
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest("runtime-b"),
        eval_result_revision=_digest("eval-b"),
        recorded_at=datetime(2026, 8, 23, tzinfo=UTC),
    )
    with pytest.raises(AgentSnapshotStoreConflict, match="Result save"):
        await coordinator.record_result(result)


def test_agent_snapshot_round_trip_is_strict_and_stably_ordered() -> None:
    snapshot = _snapshot()
    encoded = agent_snapshot_to_json(snapshot)

    assert agent_snapshot_from_json(encoded) == snapshot
    assert encoded == agent_snapshot_to_json(agent_snapshot_from_json(encoded))
    assert '"captured_at":"2026-08-23T00:00:00Z"' in encoded

    forged = snapshot.model_dump(mode="json")
    forged["components"] = list(reversed(forged["components"]))
    with pytest.raises(ValidationError, match="sorted by kind"):
        AgentSnapshot.model_validate(forged)

    forged = snapshot.model_dump(mode="json")
    forged["snapshot_root"] = _digest("forged")
    with pytest.raises(ValidationError, match="snapshot_root does not match"):
        AgentSnapshot.model_validate(forged)


def test_snapshot_contract_excludes_unsafe_source_references_and_hidden_data() -> None:
    with pytest.raises(ValidationError, match="bounded cayu-ref"):
        _ref("unsafe", source_ref="/private/data/cayu.db")
    with pytest.raises(ValidationError, match="pathname"):
        _ref("unsafe", source_ref="cayu-ref:https://example.test?token=secret")
    with pytest.raises(ValidationError, match="pathname"):
        _ref("unsafe", source_ref="cayu-ref:token=secret")

    encoded = agent_snapshot_to_json(_snapshot())
    for excluded in (
        "api_key",
        "credential_value",
        "expected_answer",
        "judge_prompt",
        "raw_provider_state_value",
        "/private/",
    ):
        assert excluded not in encoded


def test_snapshot_rejects_unsupported_schema_and_nested_memory_scope_broadening() -> None:
    snapshot = _snapshot()
    unsupported = json.loads(agent_snapshot_to_json(snapshot))
    unsupported["schema_version"] = 1
    with pytest.raises(ValidationError, match="Input should be 3"):
        AgentSnapshot.model_validate(unsupported)

    assert snapshot.memory_state is not None
    memory = snapshot.memory_state
    assert memory.knowledge is not None
    broadened = MemoryStateRef.create(
        knowledge=memory.knowledge.model_copy(
            update={"scope_fingerprint": _digest("broader-scope")}
        ),
        transcript_evidence=memory.transcript_evidence,
        artifact_evidence=memory.artifact_evidence,
        work_context=memory.work_context,
        recall_policy=memory.recall_policy,
        admission_policy=memory.admission_policy,
        context_projection_policy=memory.context_projection_policy,
        interaction_focus=memory.interaction_focus,
        recall_receipts=memory.recall_receipts,
        context_exposures=memory.context_exposures,
        index_readiness=memory.index_readiness,
        learning_disposition=memory.learning_disposition,
        limitations=memory.limitations,
    )
    components = tuple(
        component.model_copy(
            update={
                "logical": component.logical.model_copy(
                    update={"fingerprint": broadened.fingerprint}
                )
            }
        )
        if component.kind is AgentSnapshotComponentKind.MEMORY
        else component
        for component in snapshot.components
    )
    with pytest.raises(ValidationError, match="cannot broaden"):
        AgentSnapshot.create(
            capture_request_id=snapshot.capture_request_id,
            captured_at=snapshot.captured_at,
            subject=snapshot.subject,
            authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
            execution_profile=snapshot.execution_profile,
            memory_state=broadened,
            components=components,
        )


def test_existing_cayu_identity_adapters_feed_the_snapshot_contract() -> None:
    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="test-model"))
    body = app_body_snapshot_ref(
        app.describe(),
        application_release_id="release-17",
        agent_id="assistant",
        source_ref="cayu-ref:body-package:release-17",
    )
    assert body.revision == "application-release:release-17"

    profile = build_execution_profile_identity(
        runtime_name="cayu",
        runtime_version="0.3.0",
        provider_name="test-provider",
        model="test-model",
        durable_system_prompt="private prompt is hashed by the source contract",
        direct_tools=(),
        tool_catalogue_revision=f"sha256:{_digest('empty-tool-catalogue')}",
    )
    projected_profile = execution_profile_snapshot_ref(profile)
    assert projected_profile.fingerprint == profile.fingerprint
    assert tuple(component.name for component in projected_profile.components) == tuple(
        sorted(component.component_class.value for component in profile.components)
    )

    scope = _digest("scope-1")
    observation = WorkspaceRevisionObservation(
        identity=WorkspaceIdentity(workspace_id="/physical/path/one", observer="test"),
        status=WorkspaceRevisionObservationStatus.SUPPORTED,
        revision="workspace-content-v1",
        paths=(
            WorkspacePathRevision(
                path="README.md",
                working_tree="present",
                present=True,
                kind="file",
                content_sha256=_digest("readme"),
            ),
        ),
        total_paths=1,
    )
    workspace = workspace_snapshot_ref(observation, scope_fingerprint=scope)
    relocated = workspace_snapshot_ref(
        observation.model_copy(
            update={
                "identity": WorkspaceIdentity(
                    workspace_id="/physical/path/two",
                    observer="another-adapter",
                )
            }
        ),
        scope_fingerprint=scope,
    )
    assert workspace.fingerprint == relocated.fingerprint

    trajectory = trajectory_snapshot_ref(
        Trajectory(final_output="bounded result"),
        scope_fingerprint=scope,
        source_ref="cayu-ref:trajectory-package:1",
    )
    assert trajectory.revision == f"trajectory:{trajectory.fingerprint}"


class _Provider(AgentSnapshotComponentProvider):
    def __init__(
        self,
        capture: AgentSnapshotComponentCapture,
        *,
        verified: bool = True,
        operation_results: dict[str, AgentSnapshotMaterializedComponent] | None = None,
        operation_locks: dict[str, asyncio.Lock] | None = None,
        effect_calls: dict[AgentSnapshotComponentKind, int] | None = None,
    ) -> None:
        self.kind = capture.component.kind
        self.provider_id = capture.component.provider_id
        self.captured = capture
        self.verified = verified
        self.materialize_calls = 0
        self.materialize_invocations = 0
        self.recover_operation_calls = 0
        self.recover_calls = 0
        self.relocated_materialization_ref: str | None = None
        self.relocated_overlay_source_ref: str | None = None
        self.overlay_mutations: dict[str, list[str]] = {}
        self.operation_results = operation_results if operation_results is not None else {}
        self.operation_locks = operation_locks if operation_locks is not None else {}
        self.effect_calls = effect_calls if effect_calls is not None else {}
        self.effect_delay_s = 0.0
        self.effect_started_event: asyncio.Event | None = None
        self.effect_release_event: asyncio.Event | None = None

    async def capture(
        self,
        request: AgentSnapshotCaptureRequest,
        selector: AgentSnapshotComponentSelector,
    ) -> AgentSnapshotComponentCapture:
        return self.captured

    async def verify(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
    ) -> bool:
        return self.verified

    async def materialize(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        self.materialize_invocations += 1
        return await self._materialize_once(component, request, operation)

    async def recover_materialization_operation(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        self.recover_operation_calls += 1
        return await self._materialize_once(component, request, operation)

    async def _materialize_once(
        self,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
        operation: AgentSnapshotMaterializationOperation,
    ) -> AgentSnapshotMaterializedComponent:
        lock = self.operation_locks.setdefault(operation.operation_id, asyncio.Lock())
        async with lock:
            existing = self.operation_results.get(operation.operation_id)
            if existing is not None:
                return AgentSnapshotMaterializedComponent.model_validate(
                    existing.model_dump(mode="json")
                )
            if self.effect_started_event is not None:
                self.effect_started_event.set()
            if self.effect_release_event is not None:
                await self.effect_release_event.wait()
            if self.effect_delay_s:
                await asyncio.sleep(self.effect_delay_s)
            self.materialize_calls += 1
            self.effect_calls[self.kind] = self.effect_calls.get(self.kind, 0) + 1
            result = self._new_materialization(component, request)
            self.operation_results[operation.operation_id] = result
            return AgentSnapshotMaterializedComponent.model_validate(result.model_dump(mode="json"))

    def _new_materialization(
        self,
        component: AgentSnapshotComponentRef,
        request: AgentSnapshotMaterializationRequest,
    ) -> AgentSnapshotMaterializedComponent:
        overlay = None
        if self.kind in {
            AgentSnapshotComponentKind.MEMORY,
            AgentSnapshotComponentKind.WORKSPACE,
        }:
            overlay_kind = (
                AgentSnapshotOverlayKind.MEMORY
                if self.kind is AgentSnapshotComponentKind.MEMORY
                else AgentSnapshotOverlayKind.WORKSPACE
            )
            overlay_id = f"{self.kind.value}-{request.state_scope_id[:16]}"
            overlay = AgentSnapshotOverlayRef.create(
                kind=overlay_kind,
                overlay_id=overlay_id,
                baseline_fingerprint=component.logical.fingerprint,
                candidate_id=request.candidate_id,
                state_scope_id=request.state_scope_id,
                source_ref=f"cayu-ref:{overlay_id}",
            )
            self.overlay_mutations.setdefault(overlay.fingerprint, [])
        return AgentSnapshotMaterializedComponent(
            kind=self.kind,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
            materialization_ref=f"cayu-ref:materialized:{self.kind.value}",
            overlay=overlay,
        )

    async def recover(
        self,
        snapshot: AgentSnapshot,
        component: AgentSnapshotComponentRef,
        materialized: AgentSnapshotMaterializedComponent,
        materialization,
    ) -> AgentSnapshotMaterializedComponent:
        self.recover_calls += 1
        updates: dict[str, object] = {}
        if self.relocated_materialization_ref is not None:
            updates["materialization_ref"] = self.relocated_materialization_ref
        if self.relocated_overlay_source_ref is not None and materialized.overlay is not None:
            updates["overlay"] = materialized.overlay.model_copy(
                update={"source_ref": self.relocated_overlay_source_ref}
            )
        if not updates:
            return materialized
        return materialized.model_copy(update=updates)


def _providers(
    snapshot: AgentSnapshot,
    *,
    operation_results: dict[str, AgentSnapshotMaterializedComponent] | None = None,
    operation_locks: dict[str, asyncio.Lock] | None = None,
    effect_calls: dict[AgentSnapshotComponentKind, int] | None = None,
) -> list[_Provider]:
    shared_results = operation_results if operation_results is not None else {}
    shared_locks = operation_locks if operation_locks is not None else {}
    shared_effect_calls = effect_calls if effect_calls is not None else {}
    providers: list[_Provider] = []
    for component in snapshot.components:
        providers.append(
            _Provider(
                AgentSnapshotComponentCapture(
                    component=component,
                    execution_profile=(
                        snapshot.execution_profile
                        if component.kind is AgentSnapshotComponentKind.EXECUTION_PROFILE
                        else None
                    ),
                    memory_state=(
                        snapshot.memory_state
                        if component.kind is AgentSnapshotComponentKind.MEMORY
                        else None
                    ),
                ),
                operation_results=shared_results,
                operation_locks=shared_locks,
                effect_calls=shared_effect_calls,
            )
        )
    return providers


def _capture_request(snapshot: AgentSnapshot) -> AgentSnapshotCaptureRequest:
    return AgentSnapshotCaptureRequest(
        capture_request_id="capture-through-coordinator",
        subject=snapshot.subject,
        authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
        components=tuple(
            AgentSnapshotComponentSelector(kind=component.kind, required=component.required)
            for component in snapshot.components
        ),
        evaluator=snapshot.evaluator,
        promotion_authority=snapshot.promotion_authority,
    )


@_async_test
async def test_strong_capture_fails_closed_on_unavailable_or_inconsistent_state() -> None:
    snapshot = _snapshot()
    components = list(snapshot.components)
    memory = components[2].model_copy(update={"consistency": AgentSnapshotConsistency.BEST_EFFORT})
    components[2] = memory
    captures = []
    for component in components:
        captures.append(
            AgentSnapshotComponentCapture(
                component=component,
                execution_profile=(
                    snapshot.execution_profile
                    if component.kind is AgentSnapshotComponentKind.EXECUTION_PROFILE
                    else None
                ),
                memory_state=(
                    snapshot.memory_state
                    if component.kind is AgentSnapshotComponentKind.MEMORY
                    else None
                ),
            )
        )
    coordinator = AgentSnapshotCoordinator(_Provider(capture) for capture in captures)

    with pytest.raises(AgentSnapshotCaptureError, match="required_consistency_unavailable"):
        await coordinator.capture(_capture_request(snapshot))

    unavailable = components[2].model_copy(
        update={
            "completeness": AgentSnapshotCompleteness.UNAVAILABLE,
            "materialization": AgentSnapshotMaterializationCapability.UNAVAILABLE,
        }
    )
    captures[2] = AgentSnapshotComponentCapture(
        component=unavailable,
        memory_state=snapshot.memory_state,
    )
    coordinator = AgentSnapshotCoordinator(_Provider(capture) for capture in captures)
    with pytest.raises(AgentSnapshotCaptureError, match="required_component_unavailable"):
        await coordinator.capture(_capture_request(snapshot))


@_async_test
async def test_capture_verifies_every_component_and_rejects_scope_broadening() -> None:
    snapshot = _snapshot()
    providers = _providers(snapshot)
    providers[-1].verified = False
    coordinator = AgentSnapshotCoordinator(providers)

    with pytest.raises(AgentSnapshotVerificationError, match="workspace"):
        await coordinator.capture(_capture_request(snapshot))

    workspace = snapshot.components[-1].model_copy(
        update={
            "logical": snapshot.components[-1].logical.model_copy(
                update={"scope_fingerprint": _digest("broader-scope")}
            )
        }
    )
    providers = _providers(snapshot)
    providers[-1].captured = AgentSnapshotComponentCapture(component=workspace)
    coordinator = AgentSnapshotCoordinator(providers)
    with pytest.raises(AgentSnapshotCaptureError, match="authority_scope_broadened"):
        await coordinator.capture(_capture_request(snapshot))


@_async_test
async def test_requested_unavailable_component_is_explicit_and_can_be_optional() -> None:
    snapshot = _snapshot()
    components = tuple(
        component.model_copy(
            update={
                "required": False,
                "completeness": AgentSnapshotCompleteness.UNAVAILABLE,
                "materialization": AgentSnapshotMaterializationCapability.UNAVAILABLE,
                "limitations": ("portable_workspace_adapter_unavailable",),
            }
        )
        if component.kind is AgentSnapshotComponentKind.WORKSPACE
        else component
        for component in snapshot.components
    )
    declared = AgentSnapshot.create(
        capture_request_id=snapshot.capture_request_id,
        captured_at=snapshot.captured_at,
        subject=snapshot.subject,
        authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
        execution_profile=snapshot.execution_profile,
        memory_state=snapshot.memory_state,
        components=components,
    )
    providers = _providers(declared)
    coordinator = AgentSnapshotCoordinator(providers)
    captured = await coordinator.capture(_capture_request(declared))

    workspace = captured.component(AgentSnapshotComponentKind.WORKSPACE)
    assert workspace.required is False
    assert workspace.completeness is AgentSnapshotCompleteness.UNAVAILABLE
    assert workspace.materialization is AgentSnapshotMaterializationCapability.UNAVAILABLE

    providers_without_workspace = tuple(
        provider
        for provider in providers
        if provider.kind is not AgentSnapshotComponentKind.WORKSPACE
    )
    restarted = AgentSnapshotCoordinator(
        providers_without_workspace,
        store=coordinator.store,
    )
    assert await restarted.verify(captured) == captured
    materialized = await restarted.materialize(
        AgentSnapshotMaterializationRequest(
            access=_access(captured),
            candidate_id="candidate-a",
            trial_id="trial-1",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
    )
    assert AgentSnapshotComponentKind.WORKSPACE not in {
        component.kind for component in materialized.components
    }

    with pytest.raises(AgentSnapshotCaptureError, match="provider_unavailable"):
        await restarted.capture(_capture_request(declared))


@_async_test
async def test_candidates_receive_private_memory_and_workspace_overlays() -> None:
    snapshot = _snapshot()
    providers = _providers(snapshot)
    coordinator = AgentSnapshotCoordinator(providers)
    captured = await coordinator.capture(_capture_request(snapshot))

    first = await coordinator.materialize(
        AgentSnapshotMaterializationRequest(
            access=_access(captured),
            candidate_id="candidate-a",
            trial_id="trial-1",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
    )
    second = await coordinator.materialize(
        AgentSnapshotMaterializationRequest(
            access=_access(captured),
            candidate_id="candidate-b",
            trial_id="trial-1",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
    )

    first_overlays = tuple(
        component.overlay for component in first.components if component.overlay is not None
    )
    second_overlays = tuple(
        component.overlay for component in second.components if component.overlay is not None
    )
    assert {overlay.kind for overlay in first_overlays} == {
        AgentSnapshotOverlayKind.MEMORY,
        AgentSnapshotOverlayKind.WORKSPACE,
    }
    assert {overlay.fingerprint for overlay in first_overlays}.isdisjoint(
        overlay.fingerprint for overlay in second_overlays
    )
    assert all(
        overlay.baseline_fingerprint
        in {
            snapshot.component(AgentSnapshotComponentKind.MEMORY).logical.fingerprint,
            snapshot.component(AgentSnapshotComponentKind.WORKSPACE).logical.fingerprint,
        }
        for overlay in (*first_overlays, *second_overlays)
    )

    memory_provider = next(
        provider for provider in providers if provider.kind is AgentSnapshotComponentKind.MEMORY
    )
    memory_provider.overlay_mutations[first_overlays[0].fingerprint].append("candidate-a-write")
    assert memory_provider.overlay_mutations[second_overlays[0].fingerprint] == []
    assert captured == await coordinator.store.load_snapshot(captured.fingerprint)


@_async_test
async def test_trial_state_accumulation_is_explicit() -> None:
    snapshot = _snapshot()
    providers = _providers(snapshot)
    coordinator = AgentSnapshotCoordinator(providers)
    captured = await coordinator.capture(_capture_request(snapshot))

    accumulated_one = await coordinator.materialize(
        AgentSnapshotMaterializationRequest(
            access=_access(captured),
            candidate_id="candidate-a",
            trial_id="trial-1",
            state_mode=AgentSnapshotTrialStateMode.ACCUMULATE_WITHIN_CANDIDATE,
        )
    )
    accumulated_two = await coordinator.materialize(
        AgentSnapshotMaterializationRequest(
            access=_access(captured),
            candidate_id="candidate-a",
            trial_id="trial-2",
            state_mode=AgentSnapshotTrialStateMode.ACCUMULATE_WITHIN_CANDIDATE,
        )
    )
    assert sum(provider.materialize_calls for provider in providers) == len(snapshot.components)
    assert sum(provider.recover_calls for provider in providers) == 2 * len(snapshot.components)
    reset_two = await coordinator.materialize(
        AgentSnapshotMaterializationRequest(
            access=_access(captured),
            candidate_id="candidate-a",
            trial_id="trial-2",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
    )

    assert accumulated_one.state_scope_id == accumulated_two.state_scope_id
    assert accumulated_one.fingerprint == accumulated_two.fingerprint
    assert reset_two.state_scope_id != accumulated_one.state_scope_id
    assert reset_two.fingerprint != accumulated_one.fingerprint
    assert sum(provider.materialize_calls for provider in providers) == 2 * len(snapshot.components)


def test_default_materialization_identity_preserves_base_record_shape() -> None:
    snapshot = _snapshot()
    component = snapshot.component(AgentSnapshotComponentKind.MEMORY)
    legacy_request = AgentSnapshotMaterializationRequest(
        access=_access(snapshot),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    legacy_operation = AgentSnapshotMaterializationOperation.create(
        request=legacy_request,
        component=component,
    )
    legacy_payload = legacy_operation.model_dump(mode="json")

    assert "state_partition_fingerprint" not in legacy_request.model_dump(mode="json")
    assert "state_partition_fingerprint" not in legacy_payload
    assert AgentSnapshotMaterializationOperation.model_validate(legacy_payload) == legacy_operation

    partitioned_request = legacy_request.model_copy(
        update={"state_partition_fingerprint": _digest("partition-a")}
    )
    partitioned_operation = AgentSnapshotMaterializationOperation.create(
        request=partitioned_request,
        component=component,
    )
    assert partitioned_request.state_scope_id != legacy_request.state_scope_id
    assert partitioned_operation.operation_id != legacy_operation.operation_id
    assert partitioned_operation.model_dump(mode="json")["state_partition_fingerprint"] == _digest(
        "partition-a"
    )


@_async_test
async def test_fresh_process_recovery_and_result_lineage_are_idempotent(tmp_path) -> None:
    now = [datetime(2026, 8, 23, tzinfo=UTC)]
    evaluator_fingerprint = _digest("hidden-evaluator-v1")
    snapshot = _snapshot(
        evaluator=AgentSnapshotAuthorityRef(
            identity=AgentSnapshotLogicalRef(
                fingerprint=evaluator_fingerprint,
                revision="hidden-evaluator-v1",
            )
        )
    )
    providers = _providers(snapshot)
    path = tmp_path / "snapshots.db"
    first_store = SQLiteAgentSnapshotStore(path)
    first = AgentSnapshotCoordinator(providers, store=first_store, clock=lambda: now[0])
    captured = await first.capture(_capture_request(snapshot))
    materialization_request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    materialization = await first.materialize(materialization_request)
    trial = await first.begin_trial(
        materialization,
        case_id="hidden-case-1",
        trial_id="trial-1",
        evaluator_fingerprint=evaluator_fingerprint,
    )
    result = AgentSnapshotResultBinding.create(
        trial=trial,
        session_id="candidate-session-1",
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest("runtime-evidence"),
        eval_result_revision=_digest("eval-result"),
        memory_evidence_fingerprint=_digest("memory-evidence"),
        usage_fingerprint=_digest("usage"),
        cost_fingerprint=_digest("cost"),
        recorded_at=now[0],
    )
    assert await first.record_result(result) == result

    now[0] += timedelta(minutes=5)
    restarted_providers = _providers(snapshot)
    restarted_providers[0].relocated_materialization_ref = "cayu-ref:relocated:body"
    workspace_provider = next(
        provider
        for provider in restarted_providers
        if provider.kind is AgentSnapshotComponentKind.WORKSPACE
    )
    workspace_provider.relocated_overlay_source_ref = "cayu-ref:relocated:workspace-overlay"
    restarted = AgentSnapshotCoordinator(
        restarted_providers,
        store=SQLiteAgentSnapshotStore(path),
        clock=lambda: now[0],
    )
    recovered = await restarted.materialize(materialization_request)

    assert recovered.fingerprint == materialization.fingerprint
    assert recovered != materialization
    assert recovered.components[0].materialization_ref == "cayu-ref:relocated:body"
    recovered_workspace = next(
        component
        for component in recovered.components
        if component.kind is AgentSnapshotComponentKind.WORKSPACE
    )
    assert recovered_workspace.overlay is not None
    assert recovered_workspace.overlay.source_ref == "cayu-ref:relocated:workspace-overlay"
    recovered_trial = await restarted.begin_trial(
        recovered,
        case_id="hidden-case-2",
        trial_id="trial-1",
        evaluator_fingerprint=evaluator_fingerprint,
    )
    assert recovered_trial.materialization_fingerprint == materialization.fingerprint
    assert sum(provider.materialize_calls for provider in restarted_providers) == 0
    assert sum(provider.recover_calls for provider in restarted_providers) == len(
        snapshot.components
    )
    assert await restarted.store.load_trial(trial.fingerprint) == trial
    assert await restarted.store.load_result(result.fingerprint) == result
    trial_metadata = cast(
        "dict[str, object]",
        trial.session_metadata()["agent_snapshot_trial"],
    )
    assert trial_metadata["snapshot_fingerprint"] == captured.fingerprint


@_async_test
async def test_sqlite_store_materializes_distinct_trials_concurrently(tmp_path) -> None:
    snapshot = _snapshot()
    store = SQLiteAgentSnapshotStore(tmp_path / "concurrent-materializations.db")
    coordinator = AgentSnapshotCoordinator(_providers(snapshot), store=store)
    captured = await coordinator.capture(_capture_request(snapshot))

    materializations = await asyncio.gather(
        *(
            coordinator.materialize(
                AgentSnapshotMaterializationRequest(
                    access=_access(captured),
                    candidate_id="candidate-a",
                    trial_id=f"trial-{index}",
                    state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
                )
            )
            for index in range(10)
        )
    )

    assert len({item.fingerprint for item in materializations}) == 10
    assert len({item.state_scope_id for item in materializations}) == 10
    with store._connection() as connection:
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30_000


@_async_test
async def test_begin_trial_binds_reset_scope_and_declared_evaluator() -> None:
    evaluator_fingerprint = _digest("evaluator-a")
    snapshot = _snapshot(
        evaluator=AgentSnapshotAuthorityRef(
            identity=AgentSnapshotLogicalRef(
                fingerprint=evaluator_fingerprint,
                revision="evaluator-a",
            )
        )
    )
    coordinator = AgentSnapshotCoordinator(_providers(snapshot))
    captured = await coordinator.capture(_capture_request(snapshot))
    request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    materialization = await coordinator.materialize(request)

    with pytest.raises(AgentSnapshotMaterializationError, match="[Tt]rial"):
        await coordinator.begin_trial(
            materialization,
            case_id="case-1",
            trial_id="trial-2",
            evaluator_fingerprint=evaluator_fingerprint,
        )
    with pytest.raises(AgentSnapshotMaterializationError, match="evaluator"):
        await coordinator.begin_trial(
            materialization,
            case_id="case-1",
            trial_id="trial-1",
            evaluator_fingerprint=_digest("evaluator-b"),
        )

    trial = await coordinator.begin_trial(
        materialization,
        case_id="case-1",
        trial_id="trial-1",
        evaluator_fingerprint=evaluator_fingerprint,
    )
    assert trial.trial_id == "trial-1"
    assert trial.evaluator_fingerprint == evaluator_fingerprint


@_async_test
async def test_begin_trial_requires_evaluator_declared_by_snapshot() -> None:
    snapshot = _snapshot()
    coordinator = AgentSnapshotCoordinator(_providers(snapshot))
    captured = await coordinator.capture(_capture_request(snapshot))
    request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    materialization = await coordinator.materialize(request)

    with pytest.raises(AgentSnapshotMaterializationError, match="evaluator"):
        await coordinator.begin_trial(
            materialization,
            case_id="case-1",
            trial_id="trial-1",
            evaluator_fingerprint=_digest("undeclared-evaluator"),
        )


def test_materialized_component_rejects_overlay_kind_mismatch() -> None:
    baseline = _digest("memory-baseline")
    overlay = AgentSnapshotOverlayRef.create(
        kind=AgentSnapshotOverlayKind.WORKSPACE,
        overlay_id="wrong-kind-overlay",
        baseline_fingerprint=baseline,
        candidate_id="candidate-a",
        state_scope_id=_digest("state-scope"),
    )

    with pytest.raises(ValidationError, match="overlay kind"):
        AgentSnapshotMaterializedComponent(
            kind=AgentSnapshotComponentKind.MEMORY,
            baseline_fingerprint=baseline,
            capability=AgentSnapshotMaterializationCapability.RESTORABLE,
            overlay=overlay,
        )


@_async_test
async def test_coordinator_rejects_provider_overlay_kind_mismatch() -> None:
    snapshot = _snapshot()
    providers = _providers(snapshot)
    memory_provider = next(
        provider for provider in providers if provider.kind is AgentSnapshotComponentKind.MEMORY
    )

    async def wrong_overlay_kind(snapshot, component, request, operation):
        return AgentSnapshotMaterializedComponent.model_construct(
            kind=AgentSnapshotComponentKind.MEMORY,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
            materialization_ref="cayu-ref:wrong-overlay-kind",
            overlay=AgentSnapshotOverlayRef.create(
                kind=AgentSnapshotOverlayKind.WORKSPACE,
                overlay_id="wrong-overlay-kind",
                baseline_fingerprint=component.logical.fingerprint,
                candidate_id=request.candidate_id,
                state_scope_id=request.state_scope_id,
            ),
            limitations=(),
        )

    memory_provider.materialize = cast("Any", wrong_overlay_kind)
    coordinator = AgentSnapshotCoordinator(providers)
    captured = await coordinator.capture(_capture_request(snapshot))

    with pytest.raises(AgentSnapshotMaterializationError, match="invalid memory"):
        await coordinator.materialize(
            AgentSnapshotMaterializationRequest(
                access=_access(captured),
                candidate_id="candidate-a",
                trial_id="trial-1",
                state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            )
        )


@_async_test
async def test_materialization_rejects_overlay_baseline_substitution() -> None:
    snapshot = _snapshot()
    providers = _providers(snapshot)
    workspace = next(
        provider for provider in providers if provider.kind is AgentSnapshotComponentKind.WORKSPACE
    )

    async def substituted_materialize(snapshot, component, request, operation):
        overlay = AgentSnapshotOverlayRef.create(
            kind=AgentSnapshotOverlayKind.WORKSPACE,
            overlay_id="substituted-overlay",
            baseline_fingerprint=_digest("different-baseline"),
            candidate_id=request.candidate_id,
            state_scope_id=request.state_scope_id,
        )
        return AgentSnapshotMaterializedComponent(
            kind=component.kind,
            baseline_fingerprint=component.logical.fingerprint,
            capability=component.materialization,
            overlay=overlay,
        )

    workspace.materialize = cast("Any", substituted_materialize)
    coordinator = AgentSnapshotCoordinator(providers)
    captured = await coordinator.capture(_capture_request(snapshot))

    with pytest.raises(AgentSnapshotMaterializationError, match="Overlay baseline"):
        await coordinator.materialize(
            AgentSnapshotMaterializationRequest(
                access=_access(captured),
                candidate_id="candidate-a",
                trial_id="trial-1",
                state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            )
        )


@_async_test
async def test_restart_resumes_after_later_provider_failure_without_replaying_completed(
    tmp_path,
) -> None:
    snapshot = _snapshot()
    path = tmp_path / "later-provider-failure.db"
    operation_results: dict[str, AgentSnapshotMaterializedComponent] = {}
    operation_locks: dict[str, asyncio.Lock] = {}
    effect_calls: dict[AgentSnapshotComponentKind, int] = {}
    first_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    workspace = next(
        provider
        for provider in first_providers
        if provider.kind is AgentSnapshotComponentKind.WORKSPACE
    )

    async def fail_before_effect(snapshot, component, request, operation):
        raise RuntimeError("injected later-provider failure")

    workspace.materialize = cast("Any", fail_before_effect)
    first = AgentSnapshotCoordinator(first_providers, store=SQLiteAgentSnapshotStore(path))
    captured = await first.capture(_capture_request(snapshot))
    request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )

    with pytest.raises(AgentSnapshotMaterializationError, match="workspace"):
        await first.materialize(request)

    access = AgentSnapshotAccess(
        snapshot=captured.ref,
        binding_id=captured.identity_binding.binding_id,
        authority_scope_fingerprint=captured.authority_scope_fingerprint,
    )
    protected_plan = await first.store.plan_snapshot_gc(
        AgentSnapshotGCRequest(
            operation_id="failed-materialization-gc",
            candidates=(access,),
        )
    )
    assert protected_plan.blocked_roots == (captured.snapshot_root,)
    assert protected_plan.collectable_roots == ()

    restarted_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    restarted = AgentSnapshotCoordinator(
        restarted_providers,
        store=SQLiteAgentSnapshotStore(path),
    )
    materialization = await restarted.materialize(request)

    assert len(materialization.components) == len(snapshot.components)
    for kind in (
        AgentSnapshotComponentKind.BODY,
        AgentSnapshotComponentKind.EXECUTION_PROFILE,
        AgentSnapshotComponentKind.MEMORY,
    ):
        assert effect_calls[kind] == 1
    assert effect_calls[AgentSnapshotComponentKind.WORKSPACE] == 1
    assert sum(provider.recover_operation_calls for provider in restarted_providers) == 1
    assert sum(provider.materialize_invocations for provider in restarted_providers) == 0
    released_plan = await restarted.store.plan_snapshot_gc(
        AgentSnapshotGCRequest(
            operation_id="recovered-materialization-gc",
            candidates=(access,),
        )
    )
    assert released_plan.collectable_roots == (captured.snapshot_root,)


class _LoseWorkspaceProgressAcknowledgementOnceStore(SQLiteAgentSnapshotStore):
    def __init__(self, path) -> None:
        super().__init__(path)
        self.failures = 1

    async def complete_materialization_operation(
        self,
        progress,
        operation_id,
        component,
    ):
        if self.failures and component.kind is AgentSnapshotComponentKind.WORKSPACE:
            self.failures -= 1
            raise RuntimeError("injected lost component acknowledgement")
        return await super().complete_materialization_operation(
            progress,
            operation_id,
            component,
        )


@_async_test
async def test_restart_recovers_effect_completed_before_store_acknowledgement(tmp_path) -> None:
    snapshot = _snapshot()
    path = tmp_path / "lost-acknowledgement.db"
    operation_results: dict[str, AgentSnapshotMaterializedComponent] = {}
    operation_locks: dict[str, asyncio.Lock] = {}
    effect_calls: dict[AgentSnapshotComponentKind, int] = {}
    first_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    first = AgentSnapshotCoordinator(
        first_providers,
        store=_LoseWorkspaceProgressAcknowledgementOnceStore(path),
    )
    captured = await first.capture(_capture_request(snapshot))
    request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )

    with pytest.raises(RuntimeError, match="lost component acknowledgement"):
        await first.materialize(request)

    blocked_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    blocked_workspace = next(
        provider
        for provider in blocked_providers
        if provider.kind is AgentSnapshotComponentKind.WORKSPACE
    )

    async def outcome_unknown(snapshot, component, request, operation):
        raise RuntimeError("provider cannot prove the operation outcome")

    blocked_workspace.recover_materialization_operation = cast("Any", outcome_unknown)
    blocked = AgentSnapshotCoordinator(
        blocked_providers,
        store=SQLiteAgentSnapshotStore(path),
    )
    with pytest.raises(AgentSnapshotMaterializationError, match="recover workspace"):
        await blocked.materialize(request)
    assert sum(provider.materialize_invocations for provider in blocked_providers) == 0

    restarted_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    restarted = AgentSnapshotCoordinator(
        restarted_providers,
        store=SQLiteAgentSnapshotStore(path),
    )
    await restarted.materialize(request)

    assert set(effect_calls.values()) == {1}
    assert sum(provider.recover_operation_calls for provider in restarted_providers) == 1
    assert sum(provider.materialize_invocations for provider in restarted_providers) == 0


@_async_test
async def test_same_sqlite_scope_contention_does_not_duplicate_provider_effects(tmp_path) -> None:
    snapshot = _snapshot()
    path = tmp_path / "same-scope-race.db"
    bootstrap = AgentSnapshotCoordinator(
        _providers(snapshot),
        store=SQLiteAgentSnapshotStore(path),
    )
    captured = await bootstrap.capture(_capture_request(snapshot))
    request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    operation_results: dict[str, AgentSnapshotMaterializedComponent] = {}
    operation_locks: dict[str, asyncio.Lock] = {}
    effect_calls: dict[AgentSnapshotComponentKind, int] = {}
    first_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    second_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    effect_started = asyncio.Event()
    effect_release = asyncio.Event()
    first_providers[0].effect_started_event = effect_started
    first_providers[0].effect_release_event = effect_release
    for provider in second_providers:
        provider.relocated_materialization_ref = f"cayu-ref:second-worker:{provider.kind.value}"
    first_store = SQLiteAgentSnapshotStore(path)
    second_store = SQLiteAgentSnapshotStore(path)
    first = AgentSnapshotCoordinator(first_providers, store=first_store)
    second = AgentSnapshotCoordinator(second_providers, store=second_store)

    first_task = asyncio.create_task(first.materialize(request))
    await asyncio.wait_for(effect_started.wait(), timeout=1)
    second_task = asyncio.create_task(second.materialize(request))
    for _ in range(1_000):
        if sum(provider.recover_operation_calls for provider in second_providers) > 0:
            break
        await asyncio.sleep(0.001)
    else:
        raise AssertionError("Second SQLite coordinator did not reconcile the active operation.")
    effect_release.set()
    second_result = await second_task
    first_result = await first_task

    assert first_result.fingerprint == second_result.fingerprint
    assert set(effect_calls.values()) == {1}
    assert all(
        component.materialization_ref == f"cayu-ref:second-worker:{component.kind.value}"
        for component in second_result.components
    )
    assert sum(provider.recover_operation_calls for provider in second_providers) > 0


@_async_test
async def test_same_in_memory_scope_contention_does_not_duplicate_provider_effects() -> None:
    snapshot = _snapshot()
    store = InMemoryAgentSnapshotStore()
    bootstrap = AgentSnapshotCoordinator(_providers(snapshot), store=store)
    captured = await bootstrap.capture(_capture_request(snapshot))
    request = AgentSnapshotMaterializationRequest(
        access=_access(captured),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    operation_results: dict[str, AgentSnapshotMaterializedComponent] = {}
    operation_locks: dict[str, asyncio.Lock] = {}
    effect_calls: dict[AgentSnapshotComponentKind, int] = {}
    first_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    second_providers = _providers(
        snapshot,
        operation_results=operation_results,
        operation_locks=operation_locks,
        effect_calls=effect_calls,
    )
    effect_started = asyncio.Event()
    effect_release = asyncio.Event()
    first_providers[0].effect_started_event = effect_started
    first_providers[0].effect_release_event = effect_release
    first = AgentSnapshotCoordinator(first_providers, store=store)
    second = AgentSnapshotCoordinator(second_providers, store=store)

    first_task = asyncio.create_task(first.materialize(request))
    await asyncio.wait_for(effect_started.wait(), timeout=1)
    second_task = asyncio.create_task(second.materialize(request))
    for _ in range(1_000):
        if sum(provider.recover_operation_calls for provider in second_providers) > 0:
            break
        await asyncio.sleep(0.001)
    else:
        raise AssertionError("Second in-memory coordinator did not reconcile the active operation.")
    effect_release.set()
    second_result = await second_task
    first_result = await first_task

    assert first_result.fingerprint == second_result.fingerprint
    assert set(effect_calls.values()) == {1}
    assert sum(provider.recover_operation_calls for provider in second_providers) > 0


@_async_test
async def test_store_noop_claim_cannot_dispatch_provider_effect() -> None:
    snapshot = _snapshot()
    effect_calls: dict[AgentSnapshotComponentKind, int] = {}

    class NoopClaimStore(InMemoryAgentSnapshotStore):
        async def claim_materialization_operation(
            self,
            progress: AgentSnapshotMaterializationProgress,
            operation_id: str,
        ) -> AgentSnapshotMaterializationProgress:
            return progress

    store = NoopClaimStore()
    coordinator = AgentSnapshotCoordinator(
        _providers(snapshot, effect_calls=effect_calls),
        store=store,
    )
    captured = await coordinator.capture(_capture_request(snapshot))

    with pytest.raises(AgentSnapshotStoreConflict, match="claim|monotonic"):
        await coordinator.materialize(
            AgentSnapshotMaterializationRequest(
                access=_access(captured),
                candidate_id="candidate-a",
                trial_id="trial-1",
                state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            )
        )
    assert effect_calls == {}


@_async_test
async def test_store_rollback_refresh_cannot_redispatch_completed_provider() -> None:
    snapshot = _snapshot()

    class RollbackRefreshStore(InMemoryAgentSnapshotStore):
        initial: AgentSnapshotMaterializationProgress | None = None
        rollback = False

        async def begin_materialization(
            self,
            progress: AgentSnapshotMaterializationProgress,
        ) -> AgentSnapshotMaterializationProgress:
            self.initial = progress
            return await super().begin_materialization(progress)

        async def claim_materialization_operation(
            self,
            progress: AgentSnapshotMaterializationProgress,
            operation_id: str,
        ) -> AgentSnapshotMaterializationProgress:
            if progress.revision == 2 and not self.rollback:
                self.rollback = True
                raise AgentSnapshotStoreConflict("injected rollback conflict")
            if self.rollback and progress.revision == 0:
                return AgentSnapshotMaterializationProgress.model_validate(
                    progress.model_copy(
                        update={
                            "revision": 1,
                            "active_operation_id": operation_id,
                        }
                    ).model_dump(mode="json")
                )
            return await super().claim_materialization_operation(progress, operation_id)

        async def load_materialization_progress_for_scope(
            self,
            request: AgentSnapshotMaterializationRequest,
        ) -> AgentSnapshotMaterializationProgress | None:
            if self.rollback:
                assert self.initial is not None
                return self.initial
            return await super().load_materialization_progress_for_scope(request)

    providers = _providers(snapshot)
    coordinator = AgentSnapshotCoordinator(providers, store=RollbackRefreshStore())
    captured = await coordinator.capture(_capture_request(snapshot))

    with pytest.raises(AgentSnapshotStoreConflict):
        await coordinator.materialize(
            AgentSnapshotMaterializationRequest(
                access=_access(captured),
                candidate_id="candidate-a",
                trial_id="trial-1",
                state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
            )
        )
    body_provider = next(
        provider for provider in providers if provider.kind is AgentSnapshotComponentKind.BODY
    )
    assert body_provider.materialize_invocations == 1


@_async_test
async def test_materialization_progress_cas_rejects_forged_same_revision_state(
    tmp_path,
) -> None:
    snapshot = _snapshot()
    request = AgentSnapshotMaterializationRequest(
        access=_access(snapshot),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    initial = AgentSnapshotMaterializationProgress.create(
        request=request,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
        components=snapshot.components,
    )
    first_operation = initial.operations[0]
    forged_component = AgentSnapshotMaterializedComponent(
        kind=first_operation.component_kind,
        baseline_fingerprint=first_operation.baseline_fingerprint,
        capability=first_operation.capability,
        materialization_ref="cayu-ref:forged-progress",
    )
    forged = AgentSnapshotMaterializationProgress.model_validate(
        initial.model_copy(update={"components": (forged_component,)}).model_dump(mode="json")
    )

    for store in (
        InMemoryAgentSnapshotStore(),
        SQLiteAgentSnapshotStore(tmp_path / "forged-progress.db"),
    ):
        persisted = await store.begin_materialization(initial)
        assert persisted == initial
        with pytest.raises(AgentSnapshotStoreConflict, match="compare-and-set"):
            await store.claim_materialization_operation(
                forged,
                initial.operations[1].operation_id,
            )


@_async_test
async def test_materialization_finalization_replay_is_idempotent_in_both_stores(
    tmp_path,
) -> None:
    snapshot = _snapshot()
    request = AgentSnapshotMaterializationRequest(
        access=_access(snapshot),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    providers = {provider.kind: provider for provider in _providers(snapshot)}

    for store in (
        InMemoryAgentSnapshotStore(),
        SQLiteAgentSnapshotStore(tmp_path / "finalization-replay.db"),
    ):
        progress = await store.begin_materialization(
            AgentSnapshotMaterializationProgress.create(
                request=request,
                created_at=datetime(2026, 8, 23, tzinfo=UTC),
                components=snapshot.components,
            )
        )
        for operation in progress.operations:
            progress = await store.claim_materialization_operation(
                progress,
                operation.operation_id,
            )
            component = snapshot.component(operation.component_kind)
            result = providers[operation.component_kind]._new_materialization(
                component,
                request,
            )
            progress = await store.complete_materialization_operation(
                progress,
                operation.operation_id,
                result,
            )
        prefinal_progress = progress
        materialization = AgentSnapshotMaterialization.create(
            progress_id=progress.progress_id,
            request=request,
            created_at=progress.created_at,
            components=progress.components,
        )

        first = await store.finalize_materialization(progress, materialization)
        assert await store.finalize_materialization(prefinal_progress, materialization) == first
        final_progress = await store.load_materialization_progress_for_scope(request)
        assert final_progress is not None
        assert await store.finalize_materialization(final_progress, materialization) == first
        assert await store.save_materialization(materialization) == first

        relocated_components = tuple(
            component.model_copy(
                update={
                    "materialization_ref": f"cayu-ref:relocated:{component.kind.value}",
                    "overlay": (
                        None
                        if component.overlay is None
                        else component.overlay.model_copy(
                            update={
                                "source_ref": (f"cayu-ref:relocated:{component.kind.value}-overlay")
                            }
                        )
                    ),
                }
            )
            for component in materialization.components
        )
        relocated = AgentSnapshotMaterialization.create(
            progress_id=progress.progress_id,
            request=request,
            created_at=progress.created_at,
            components=relocated_components,
        )
        assert relocated.fingerprint == materialization.fingerprint
        assert relocated != materialization
        assert await store.save_materialization(relocated) == first

        changed_components = list(materialization.components)
        changed_components[0] = changed_components[0].model_copy(
            update={"baseline_fingerprint": _digest("different-baseline")}
        )
        different = AgentSnapshotMaterialization.create(
            progress_id=progress.progress_id,
            request=request,
            created_at=progress.created_at,
            components=changed_components,
        )
        with pytest.raises(AgentSnapshotStoreConflict, match="another identity"):
            await store.save_materialization(different)


@_async_test
async def test_sqlite_scope_load_rejects_redirected_materialization_pointer(tmp_path) -> None:
    snapshot = _snapshot()

    for forge_progress_document in (False, True):
        path = tmp_path / f"scope-redirect-{forge_progress_document}.db"
        store = SQLiteAgentSnapshotStore(path)
        coordinator = AgentSnapshotCoordinator(_providers(snapshot), store=store)
        captured = await coordinator.capture(_capture_request(snapshot))
        first_request = AgentSnapshotMaterializationRequest(
            access=_access(captured),
            candidate_id="candidate-a",
            trial_id="trial-1",
            state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
        )
        second_request = first_request.model_copy(update={"candidate_id": "candidate-b"})
        first = await coordinator.materialize(first_request)
        second = await coordinator.materialize(second_request)
        assert first.fingerprint != second.fingerprint
        first_progress = await store.load_materialization_progress_for_scope(first_request)
        assert first_progress is not None

        document = None
        if forge_progress_document:
            redirected = AgentSnapshotMaterializationProgress.model_validate(
                first_progress.model_copy(
                    update={"materialization_fingerprint": second.fingerprint}
                ).model_dump(mode="json")
            )
            document = json.dumps(
                redirected.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        with sqlite3.connect(path) as connection:
            if document is None:
                connection.execute(
                    "UPDATE cayu_agent_snapshot_materialization_progress "
                    "SET materialization_fingerprint = ? WHERE state_scope_id = ?",
                    (second.fingerprint, first_request.state_scope_id),
                )
            else:
                connection.execute(
                    "UPDATE cayu_agent_snapshot_materialization_progress "
                    "SET document = ?, materialization_fingerprint = ? "
                    "WHERE state_scope_id = ?",
                    (document, second.fingerprint, first_request.state_scope_id),
                )

        with pytest.raises(AgentSnapshotStoreConflict, match="scope"):
            await store.load_materialization_for_scope(first_request)
        with pytest.raises(
            (AgentSnapshotStoreConflict, AgentSnapshotMaterializationError),
            match="scope",
        ):
            await coordinator.materialize(first_request)


@_async_test
async def test_final_scope_rejects_snapshot_inconsistent_progress_plan(tmp_path) -> None:
    starting = _snapshot()
    substituted = _snapshot(memory_value="memory-v2")
    store = SQLiteAgentSnapshotStore(tmp_path / "snapshot-inconsistent-plan.db")
    await store.save_snapshot(starting)
    request = AgentSnapshotMaterializationRequest(
        access=_access(starting),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    progress = await store.begin_materialization(
        AgentSnapshotMaterializationProgress.create(
            request=request,
            created_at=datetime(2026, 8, 23, tzinfo=UTC),
            components=substituted.components,
        )
    )
    substituted_providers = {provider.kind: provider for provider in _providers(substituted)}
    for operation in progress.operations:
        progress = await store.claim_materialization_operation(
            progress,
            operation.operation_id,
        )
        component = substituted.component(operation.component_kind)
        result = substituted_providers[operation.component_kind]._new_materialization(
            component,
            request,
        )
        progress = await store.complete_materialization_operation(
            progress,
            operation.operation_id,
            result,
        )
    forged = AgentSnapshotMaterialization.create(
        progress_id=progress.progress_id,
        request=request,
        created_at=progress.created_at,
        components=progress.components,
    )
    await store.finalize_materialization(progress, forged)

    coordinator = AgentSnapshotCoordinator(_providers(starting), store=store)
    with pytest.raises(AgentSnapshotStoreConflict, match="operation plan"):
        await coordinator.materialize(request)


@_async_test
async def test_final_scope_rejects_same_scope_foreign_progress_materialization() -> None:
    snapshot = _snapshot()
    request = AgentSnapshotMaterializationRequest(
        access=_access(snapshot),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    providers = {provider.kind: provider for provider in _providers(snapshot)}
    components = tuple(
        providers[component.kind]._new_materialization(component, request)
        for component in snapshot.components
    )
    foreign = AgentSnapshotMaterialization.create(
        progress_id=_digest("foreign-progress"),
        request=request,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
        components=components,
    )
    expected = AgentSnapshotMaterializationProgress.create(
        request=request,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
        components=snapshot.components,
    )
    redirected = AgentSnapshotMaterializationProgress.model_validate(
        expected.model_copy(
            update={
                "revision": len(expected.operations) * 2 + 1,
                "components": components,
                "materialization_fingerprint": foreign.fingerprint,
            }
        ).model_dump(mode="json")
    )

    class RedirectedFinalStore(InMemoryAgentSnapshotStore):
        async def begin_materialization(
            self,
            progress: AgentSnapshotMaterializationProgress,
        ) -> AgentSnapshotMaterializationProgress:
            return redirected

        async def load_materialization(
            self, fingerprint: str
        ) -> AgentSnapshotMaterialization | None:
            if fingerprint == foreign.fingerprint:
                return foreign
            return await super().load_materialization(fingerprint)

    store = RedirectedFinalStore()
    await store.save_snapshot(snapshot)
    coordinator = AgentSnapshotCoordinator(_providers(snapshot), store=store)

    with pytest.raises(AgentSnapshotStoreConflict, match="scope"):
        await coordinator.materialize(request)


@_async_test
async def test_recovery_rejects_snapshot_inconsistent_materialized_components() -> None:
    starting = _snapshot()
    substituted = _snapshot(memory_value="memory-v2")
    request = AgentSnapshotMaterializationRequest(
        access=_access(starting),
        candidate_id="candidate-a",
        trial_id="trial-1",
        state_mode=AgentSnapshotTrialStateMode.RESET_EACH_TRIAL,
    )
    substituted_providers = {provider.kind: provider for provider in _providers(substituted)}
    components = tuple(
        substituted_providers[component.kind]._new_materialization(component, request)
        for component in substituted.components
    )
    forged = AgentSnapshotMaterialization.create(
        progress_id=_digest("forged-progress"),
        request=request,
        created_at=datetime(2026, 8, 23, tzinfo=UTC),
        components=components,
    )

    class ForgedMaterializationStore(InMemoryAgentSnapshotStore):
        async def load_materialization(
            self, fingerprint: str
        ) -> AgentSnapshotMaterialization | None:
            if fingerprint == forged.fingerprint:
                return forged
            return await super().load_materialization(fingerprint)

    store = ForgedMaterializationStore()
    await store.save_snapshot(starting)
    coordinator = AgentSnapshotCoordinator(_providers(starting), store=store)

    with pytest.raises(AgentSnapshotMaterializationError, match="baseline"):
        await coordinator.recover_materialization(
            forged.fingerprint,
            access=_access(starting),
        )


@_async_test
async def test_api_key_free_reference_experiment_proves_bounded_workflow(tmp_path) -> None:
    result = await run_reference_experiment(tmp_path / "reference-experiment.db")

    assert result["baseline_unchanged"] is True
    assert result["cross_candidate_contamination"] is False
    assert result["hidden_evaluator_truth_exported"] is False
    assert result["production_activation"] == "not_requested"
    assert len(result["candidate_results"]) == 2
    assert {item["score"] for item in result["candidate_results"]} == {0, 1}
    assert len({item["memory_overlay_fingerprint"] for item in result["candidate_results"]}) == 2
    assert len({item["workspace_overlay_fingerprint"] for item in result["candidate_results"]}) == 2
    assert result["recovery"]["fresh_process"] is True
    assert result["recovery"]["candidate_effects_replayed"] == 0
    assert result["recovery"]["runtime_effects_replayed"] == 0
    assert result["recovery"]["evaluator_effects_replayed"] == 0
    assert {item["overlay_disposition"] for item in result["candidate_results"]} == {
        "quarantined",
        "retained_for_review",
    }
    assert result["recommendations"] == [
        {
            "candidate_id": "candidate-b",
            "result_fingerprint": result["candidate_results"][1]["result_fingerprint"],
        }
    ]
    portable_result = json.dumps(result, sort_keys=True)
    assert "expected_answer" not in portable_result
    assert "api_key" not in portable_result
