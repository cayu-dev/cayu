"""Shared session-store workspace observations contract cases."""

from __future__ import annotations

import asyncio

import pytest
from tests.core._execution_profile_fixtures import (
    interrupt_and_release_test_invocation,
)
from tests.core.session_store_conformance.support import (
    _admit_test_workspace_observation_intent,
    _close_store,
    _identity,
    _open_store,
    _public_authority_alias_codec,
    _reopen_store,
    _SimulatedProcessLoss,
    _workspace_observation_recovery_environment_spec,
    _WorkspaceObservationProcessLossStoreMixin,
    _WorkspaceObservationRecoveryProvider,
    _WorkspaceObservationRecoveryTool,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.artifacts.local import LocalArtifactStore
from cayu.budgets.base import (
    InMemoryBudgetStore,
)
from cayu.environments.base import Environment
from cayu.environments.bindings import (
    DeterministicWorkspaceBinding,
)
from cayu.events import (
    Event,
    EventType,
)
from cayu.messages import (
    Message,
)
from cayu.observability.events import EventSink, InMemoryEventSink
from cayu.runtime._event_projection import (
    REDACTED_CUSTOM_EVENT_TYPE,
    public_event_id,
    public_event_sequence,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime.authority import SessionRunFenced
from cayu.sessions._checkpoint_mutations import (
    RuntimePublicationMutation,
    runtime_publication_checkpoint_value_digest,
)
from cayu.sessions._runtime_publication import (
    RuntimePublicationRequest,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionStatus
from cayu.sessions.recovery import IncompleteSessionRecoveryAction, IncompleteSessionRecoveryRequest
from cayu.sessions.requests import (
    RunRequest,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces.local import LocalWorkspace
from cayu.workspaces.observation_recovery import (
    WorkspaceObservationArtifact,
    WorkspaceObservationArtifactState,
    WorkspaceObservationEvidenceState,
    WorkspaceObservationLifecycle,
    WorkspaceObservationPhase,
    WorkspaceObservationTerminalStatus,
    publish_workspace_observation_transition,
    workspace_observation_event_digest,
    workspace_observations_from_checkpoint,
)

pytestmark = pytest.mark.postgres


def test_session_store_conformance_reconstructs_workspace_mutation_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"workspace-receipt-{session_store_case[0]}"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            observed_before = Event(
                id=f"evt-workspace-before-{session_store_case[0]}",
                type=EventType.WORKSPACE_REVISION_OBSERVED,
                session_id=session_id,
                payload={
                    "phase": "before",
                    "window_id": "wmut-conformance",
                    "session_run_epoch": 1,
                    "model_step": 2,
                    "tool_call_id": "call-workspace",
                    "workspace_id": "workspace-conformance",
                    "observer": "GitRepositoryBinding",
                    "status": "supported",
                    "revision": "sha256:before",
                    "head_revision": "before-head",
                    "branch": "main",
                    "paths": [],
                    "total_paths": 0,
                    "detail_code": None,
                    "model_step_id": "step-conformance",
                    "model_attempt_id": "attempt-conformance",
                    "tool_round_id": "round-conformance",
                },
            )
            observed_after = observed_before.model_copy(
                update={
                    "id": f"evt-workspace-after-{session_store_case[0]}",
                    "payload": {
                        **observed_before.payload,
                        "phase": "after",
                        "revision": "sha256:after",
                    },
                },
                deep=True,
            )
            receipt = Event(
                id=f"evt-workspace-receipt-{session_store_case[0]}",
                type=EventType.WORKSPACE_MUTATION_RECORDED,
                session_id=session_id,
                payload={
                    "window_id": "wmut-conformance",
                    "before_observation_id": observed_before.id,
                    "after_observation_id": observed_after.id,
                    "session_run_epoch": 1,
                    "model_step": 2,
                    "tool_call_id": "call-workspace",
                    "workspace_id": "workspace-conformance",
                    "observer": "GitRepositoryBinding",
                    "status": "changed",
                    "before_revision": "sha256:before",
                    "after_revision": "sha256:after",
                    "paths": [
                        {
                            "path": "created.txt",
                            "change": "added",
                            "renamed_from": None,
                        }
                    ],
                    "total_paths": 1,
                    "head_changed": False,
                    "branch_changed": False,
                    "detail_code": None,
                    "attribution": {
                        "confidence": "concurrent_ambiguity",
                        "writer_isolation": "shared",
                        "overlap_detected": True,
                        "direct_reconciliation": "consistent",
                        "detail_code": "overlapping_workspace_mutation_windows",
                    },
                    "writer_isolation": {
                        "before": {
                            "status": "shared",
                            "mechanism": "conformance-shared-writer",
                            "generation": None,
                            "detail_code": "shared_writer",
                        },
                        "after": {
                            "status": "shared",
                            "mechanism": "conformance-shared-writer",
                            "generation": None,
                            "detail_code": "shared_writer",
                        },
                    },
                    "direct_mutations": {
                        "operations": [
                            {
                                "sequence": 0,
                                "method": "create_bytes",
                                "path_sha256": "a" * 64,
                                "result_valid": True,
                                "result_operation": "create",
                                "result_evidence_sha256": "b" * 64,
                            }
                        ],
                        "retained_operations": 1,
                        "total_operations": 1,
                        "truncated": False,
                    },
                    "pre_window_change": {
                        "attribution_confidence": "external_or_unknown",
                        "status": "changed",
                        "before_revision": "sha256:earlier",
                        "after_revision": "sha256:before",
                        "paths": [{"path_sha256": "c" * 64, "change": "modified"}],
                        "retained_paths": 1,
                        "total_paths": 1,
                        "truncated": False,
                        "head_changed": False,
                        "branch_changed": False,
                        "detail_code": None,
                    },
                    "model_step_id": "step-conformance",
                    "model_attempt_id": "attempt-conformance",
                    "tool_round_id": "round-conformance",
                },
            )
            expected = [observed_before, observed_after, receipt]
            await store.append_events(session_id, expected)
            store = await _reopen_store(session_store_case, store)

            restored = await store.query_events(EventQuery(session_id=session_id))

            assert [record.event.model_dump(mode="json") for record in restored] == [
                event.model_dump(mode="json") for event in expected
            ]
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_workspace_observation_lifecycle_is_exact_and_replayable(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"workspace-observation-lifecycle-{session_store_case[0]}"
            session = await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            intent = WorkspaceObservationLifecycle(
                session_id=session_id,
                window_id="wmut-conformance-lifecycle",
                source_run_epoch=session.run_epoch,
                binding_generation_id="wbind_conformance_lifecycle",
                workspace_id="workspace-conformance-lifecycle",
                observer="ConformanceWorkspaceBinding",
                observer_authority="configured",
                artifact_store_id="artifact-conformance-lifecycle",
                agent_name="assistant",
                environment_name="local",
                tool_name="mutate",
                tool_call_id="call-workspace",
                model_step_id="mstep-workspace",
                model_attempt_id="matt-workspace",
                tool_round_id="tround-workspace",
                model_step=1,
            )
            skipped_stage = WorkspaceObservationLifecycle.model_validate(
                {
                    **intent.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                    "before_state": WorkspaceObservationEvidenceState.CAPTURED_PRIVATE.value,
                    "tool_outcome_event_id": "evt-illegal-stage-skip",
                    "tool_outcome_event_digest": "d" * 64,
                }
            )
            with pytest.raises(ValueError, match="invalid phase edge"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=intent,
                    current=skipped_stage,
                    phase="tool-outcome",
                )

            premature_before_capture = WorkspaceObservationLifecycle.model_validate(
                {
                    **intent.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.BEFORE_CAPTURED.value,
                    "before_state": WorkspaceObservationEvidenceState.CAPTURED_PRIVATE.value,
                }
            )
            premature_after_artifact = WorkspaceObservationArtifact(
                evidence_kind="revision-after",
                artifact_id="artifact-premature-after",
                sha256="e" * 64,
                size_bytes=128,
                state=WorkspaceObservationArtifactState.INTENT,
            )
            with pytest.raises(ValueError, match="invalid evidence phase"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=premature_before_capture,
                    current=premature_before_capture.model_copy(
                        update={"artifacts": (premature_after_artifact,)}
                    ),
                    phase="artifact-revision-after-intent",
                )

            staged_without_before_evidence = WorkspaceObservationLifecycle.model_validate(
                {
                    **premature_before_capture.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                    "tool_outcome_event_id": "evt-staged-without-before-evidence",
                    "tool_outcome_event_digest": "f" * 64,
                }
            )
            with pytest.raises(ValueError, match="preceded durable before evidence"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=staged_without_before_evidence,
                    current=staged_without_before_evidence.model_copy(
                        update={"artifacts": (premature_after_artifact,)}
                    ),
                    phase="artifact-revision-after-intent",
                )

            original_load_checkpoint = store.load_checkpoint
            child_cancelled = False

            async def load_checkpoint_with_child_cancellation(session_id: str):
                nonlocal child_cancelled
                if not child_cancelled:
                    child_cancelled = True
                    raise asyncio.CancelledError("store-owned checkpoint cancellation")
                return await original_load_checkpoint(session_id)

            store.load_checkpoint = load_checkpoint_with_child_cancellation  # type: ignore[method-assign]
            with pytest.raises(
                RuntimeError,
                match="Workspace observation checkpoint read was cancelled",
            ):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=None,
                    current=intent,
                    phase="intent",
                    intent_admission=_admit_test_workspace_observation_intent(intent),
                )
            current_task = asyncio.current_task()
            assert current_task is not None
            assert current_task.cancelling() == 0
            assert current_task.cancelled() is False

            async def load_checkpoint_with_child_cancellation_group(session_id: str):
                del session_id
                raise BaseExceptionGroup(
                    "store-owned grouped failure",
                    [
                        asyncio.CancelledError("store-owned grouped cancellation"),
                        RuntimeError("store-owned grouped error"),
                    ],
                )

            store.load_checkpoint = (  # type: ignore[method-assign]
                load_checkpoint_with_child_cancellation_group
            )
            with pytest.raises(
                RuntimeError,
                match="Workspace observation checkpoint read reported multiple operational failures",
            ):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=None,
                    current=intent,
                    phase="intent",
                    intent_admission=_admit_test_workspace_observation_intent(intent),
                )
            assert current_task.cancelling() == 0
            assert current_task.cancelled() is False
            store.load_checkpoint = original_load_checkpoint  # type: ignore[method-assign]

            original_publish = store.publish_runtime_publication
            sibling = intent.model_copy(
                update={
                    "window_id": "wmut-unrelated-sibling",
                    "tool_call_id": "call-unrelated-sibling",
                }
            )

            async def publish_with_sibling_mutation(session_id: str, **kwargs):
                request = kwargs["request"]
                operations = []
                workspace_observations_value = None
                for operation in request.mutation.operations:
                    if operation.key != "workspace_observations":
                        operations.append(operation)
                        continue
                    value = dict(operation.value)
                    value[sibling.window_id] = sibling.model_dump(mode="json")
                    workspace_observations_value = value
                    operations.append(operation.model_copy(update={"value": value}))
                assert workspace_observations_value is not None
                conflicting_intent = dict(request.intent)
                conflicting_intent["target_root_digest"] = (
                    runtime_publication_checkpoint_value_digest(workspace_observations_value)
                )
                conflicting_request = RuntimePublicationRequest(
                    publication_id=request.publication_id,
                    kind=request.kind,
                    interaction_id=request.interaction_id,
                    intent=conflicting_intent,
                    mutation=RuntimePublicationMutation(operations=tuple(operations)),
                    transcript_messages=request.transcript_messages,
                    events=request.events,
                    referenced_events=request.referenced_events,
                )
                return await original_publish(
                    session_id,
                    request=conflicting_request,
                    **{key: value for key, value in kwargs.items() if key != "request"},
                )

            store.publish_runtime_publication = (  # type: ignore[method-assign]
                publish_with_sibling_mutation
            )
            with pytest.raises(ValueError, match="changed a sibling lifecycle"):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=None,
                    current=intent,
                    phase="intent",
                    intent_admission=_admit_test_workspace_observation_intent(intent),
                )
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]

            returned_conflicting_acknowledgement = False

            async def publish_with_conflicting_acknowledgement(session_id: str, **kwargs):
                nonlocal returned_conflicting_acknowledgement
                result = await original_publish(session_id, **kwargs)
                if (
                    kwargs["request"].intent.get("phase") != "intent"
                    or returned_conflicting_acknowledgement
                ):
                    return result
                returned_conflicting_acknowledgement = True
                return result.model_copy(
                    update={
                        "receipt": result.receipt.model_copy(
                            update={
                                "kind": "tool-round",
                                "request_digest": "f" * 64,
                            }
                        )
                    }
                )

            store.publish_runtime_publication = (  # type: ignore[method-assign]
                publish_with_conflicting_acknowledgement
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=None,
                current=intent,
                phase="intent",
                intent_admission=_admit_test_workspace_observation_intent(intent),
            )
            assert returned_conflicting_acknowledgement is True
            store.publish_runtime_publication = original_publish  # type: ignore[method-assign]
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: intent}

            await store.checkpoint(session_id, {"generic_checkpoint_marker": "retained"})
            checkpoint_after_generic_write = await store.load_checkpoint(session_id)
            assert checkpoint_after_generic_write is not None
            assert checkpoint_after_generic_write["generic_checkpoint_marker"] == "retained"
            assert workspace_observations_from_checkpoint(checkpoint_after_generic_write) == {
                intent.window_id: intent
            }

            def replace_ordinary_checkpoint(_session, current):
                assert current is not None
                current.pop("workspace_observations", None)
                current["generic_transform_marker"] = "retained"
                return current

            await store.transform_checkpoint(session_id, replace_ordinary_checkpoint)
            checkpoint_after_generic_transform = await store.load_checkpoint(session_id)
            assert checkpoint_after_generic_transform is not None
            assert checkpoint_after_generic_transform["generic_transform_marker"] == "retained"
            assert workspace_observations_from_checkpoint(checkpoint_after_generic_transform) == {
                intent.window_id: intent
            }

            before_captured = WorkspaceObservationLifecycle.model_validate(
                {
                    **intent.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.BEFORE_CAPTURED.value,
                    "before_state": WorkspaceObservationEvidenceState.CAPTURED_PRIVATE.value,
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=intent,
                current=before_captured,
                phase="before-capture",
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: before_captured}

            staged = WorkspaceObservationLifecycle.model_validate(
                {
                    **before_captured.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.TOOL_OUTCOME_STAGED.value,
                    "tool_outcome_event_id": "evt-tool-workspace",
                    "tool_outcome_event_digest": "a" * 64,
                }
            )
            await asyncio.gather(
                *(
                    publish_workspace_observation_transition(
                        session_store=store,
                        event_writer=event_writer,
                        session=session,
                        previous=before_captured,
                        current=staged,
                        phase="tool-outcome",
                    )
                    for _ in range(2)
                )
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: staged}

            artifact_intent = WorkspaceObservationArtifact(
                evidence_kind="revision-before",
                artifact_id="artifact-workspace-before",
                sha256="b" * 64,
                size_bytes=128,
                state=WorkspaceObservationArtifactState.INTENT,
            )
            artifact_intended = WorkspaceObservationLifecycle.model_validate(
                {
                    **staged.model_dump(mode="json"),
                    "artifacts": [artifact_intent.model_dump(mode="json")],
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=staged,
                current=artifact_intended,
                phase="artifact-revision-before-intent",
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: artifact_intended}

            artifact_published = WorkspaceObservationLifecycle.model_validate(
                {
                    **artifact_intended.model_dump(mode="json"),
                    "artifacts": [
                        artifact_intent.model_copy(
                            update={"state": WorkspaceObservationArtifactState.PUBLISHED}
                        ).model_dump(mode="json")
                    ],
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=artifact_intended,
                current=artifact_published,
                phase="artifact-revision-before-published",
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: artifact_published}

            def observation_event(
                *,
                event_id: str,
                event_type: EventType,
                phase: str | None = None,
                status: str | None = None,
            ) -> Event:
                payload = {
                    "window_id": intent.window_id,
                    "session_run_epoch": intent.source_run_epoch,
                    "binding_generation_id": intent.binding_generation_id,
                    "workspace_id": intent.workspace_id,
                    "observer": intent.observer,
                    "artifact_store_id": intent.artifact_store_id,
                    "tool_call_id": intent.tool_call_id,
                    "model_step_id": intent.model_step_id,
                    "model_attempt_id": intent.model_attempt_id,
                    "tool_round_id": intent.tool_round_id,
                    "model_step": intent.model_step,
                }
                if phase is not None:
                    payload["phase"] = phase
                    payload["status"] = "supported"
                    payload["path_scope"] = "complete"
                    if phase == "before":
                        payload.update(
                            {
                                "manifest_artifact_id": artifact_intent.artifact_id,
                                "manifest_artifact_sha256": artifact_intent.sha256,
                                "manifest_artifact_size_bytes": artifact_intent.size_bytes,
                            }
                        )
                if status is not None:
                    payload["status"] = status
                if event_type is EventType.WORKSPACE_MUTATION_RECORDED:
                    payload.update(
                        {
                            "before_observation_id": before_published.before_observation_id,
                            "after_observation_id": after_captured.after_observation_id,
                            "tool_outcome_event_id": staged.tool_outcome_event_id,
                            "tool_outcome_event_digest": staged.tool_outcome_event_digest,
                        }
                    )
                return Event(
                    id=event_id,
                    type=event_type,
                    session_id=session_id,
                    agent_name=intent.agent_name,
                    environment_name=intent.environment_name,
                    tool_name=intent.tool_name,
                    payload=payload,
                )

            before_event = observation_event(
                event_id="evt-workspace-observation-lifecycle-before",
                event_type=EventType.WORKSPACE_REVISION_OBSERVED,
                phase="before",
            )
            before_published = WorkspaceObservationLifecycle.model_validate(
                {
                    **artifact_published.model_dump(mode="json"),
                    "before_state": WorkspaceObservationEvidenceState.PUBLISHED.value,
                    "before_observation_id": before_event.id,
                    "artifacts": [
                        artifact_intent.model_copy(
                            update={"state": WorkspaceObservationArtifactState.REFERENCED}
                        ).model_dump(mode="json")
                    ],
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=artifact_published,
                current=before_published,
                phase="before-evidence",
                events=(before_event,),
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: before_published}

            after_event = observation_event(
                event_id="evt-workspace-observation-lifecycle-after",
                event_type=EventType.WORKSPACE_REVISION_OBSERVED,
                phase="after",
            )
            after_captured = WorkspaceObservationLifecycle.model_validate(
                {
                    **before_published.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.AFTER_CAPTURED.value,
                    "after_state": WorkspaceObservationEvidenceState.PUBLISHED.value,
                    "after_observation_id": after_event.id,
                }
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=before_published,
                current=after_captured,
                phase="after-capture",
                events=(after_event,),
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: after_captured}

            delta_event = observation_event(
                event_id="evt-workspace-observation-lifecycle-delta",
                event_type=EventType.WORKSPACE_MUTATION_RECORDED,
                status="changed",
            )
            delta_published = WorkspaceObservationLifecycle.model_validate(
                {
                    **after_captured.model_dump(mode="json"),
                    "phase": WorkspaceObservationPhase.DELTA_PUBLISHED.value,
                    "delta_state": WorkspaceObservationEvidenceState.PUBLISHED.value,
                    "mutation_event_id": delta_event.id,
                    "mutation_event_digest": workspace_observation_event_digest(delta_event),
                }
            )
            conflicting_delta_event = delta_event.model_copy(
                update={
                    "payload": {
                        **delta_event.payload,
                        "before_observation_id": "evt-conflicting-before-observation",
                    }
                },
                deep=True,
            )
            conflicting_delta = delta_published.model_copy(
                update={
                    "mutation_event_digest": workspace_observation_event_digest(
                        conflicting_delta_event
                    )
                }
            )
            with pytest.raises(
                ValueError,
                match=r"intent\.before_observation_id",
            ):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=after_captured,
                    current=conflicting_delta,
                    phase="delta-publication",
                    events=(conflicting_delta_event,),
                )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=after_captured,
                current=delta_published,
                phase="delta-publication",
                events=(delta_event,),
            )
            store = await _reopen_store(session_store_case, store)
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(),
            )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: delta_published}

            await store.release_run_fence(session_id)
            replacement_owner = await store.fence_run_and_transform_checkpoint(
                session_id,
                statuses={SessionStatus.PENDING},
                checkpoint_transform=lambda _session, checkpoint: checkpoint or {},
            )
            terminal_event = Event(
                id="evt-workspace-observation-lifecycle-terminal",
                type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                session_id=session_id,
                agent_name=intent.agent_name,
                environment_name=intent.environment_name,
                tool_name=intent.tool_name,
                payload={
                    "window_id": intent.window_id,
                    "session_run_epoch": intent.source_run_epoch,
                    "binding_generation_id": intent.binding_generation_id,
                    "workspace_id": intent.workspace_id,
                    "observer": intent.observer,
                    "artifact_store_id": intent.artifact_store_id,
                    "tool_call_id": intent.tool_call_id,
                    "model_step_id": intent.model_step_id,
                    "model_attempt_id": intent.model_attempt_id,
                    "tool_round_id": intent.tool_round_id,
                    "model_step": intent.model_step,
                    "before_observation_id": delta_published.before_observation_id,
                    "after_observation_id": delta_published.after_observation_id,
                    "tool_outcome_event_id": delta_published.tool_outcome_event_id,
                    "tool_outcome_event_digest": delta_published.tool_outcome_event_digest,
                    "mutation_event_id": delta_published.mutation_event_id,
                    "mutation_event_digest": delta_published.mutation_event_digest,
                    "revision_before_artifact_id": artifact_intent.artifact_id,
                    "revision_before_artifact_sha256": artifact_intent.sha256,
                    "revision_before_artifact_size_bytes": artifact_intent.size_bytes,
                    "revision_before_artifact_state": "referenced",
                    "referenced_artifact_count": 1,
                    "failed_artifact_count": 0,
                    "status": "incomplete",
                    "detail_code": "worker_lost_before_workspace_observation_completed",
                },
            )
            with pytest.raises(SessionRunFenced):
                await publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=delta_published,
                    current=None,
                    phase="terminal",
                    terminal_status=WorkspaceObservationTerminalStatus.INCOMPLETE,
                    terminal_detail_code="worker_lost_before_workspace_observation_completed",
                    terminal_artifacts=delta_published.artifacts,
                    events=(terminal_event,),
                )
            assert workspace_observations_from_checkpoint(
                await store.load_checkpoint(session_id)
            ) == {intent.window_id: delta_published}

            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=replacement_owner,
                previous=delta_published,
                current=None,
                phase="terminal",
                terminal_status=WorkspaceObservationTerminalStatus.INCOMPLETE,
                terminal_detail_code="worker_lost_before_workspace_observation_completed",
                terminal_artifacts=delta_published.artifacts,
                events=(terminal_event,),
            )
            store = await _reopen_store(session_store_case, store)
            assert (
                workspace_observations_from_checkpoint(await store.load_checkpoint(session_id))
                == {}
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("crash_phase", "expected_status", "tool_ran", "recovery_repairs_observation"),
    [
        ("intent", "ambiguous", False, True),
        ("before-capture", "ambiguous", False, True),
        ("terminal-stage", "incomplete", True, True),
        ("tool-outcome", "incomplete", True, True),
        ("artifact-revision-after-intent", "incomplete", True, True),
        ("artifact-revision-after-published", "incomplete", True, True),
        ("before-evidence", "incomplete", True, True),
        ("after-capture", "incomplete", True, True),
        ("delta-publication", "complete", True, True),
        ("terminal", "complete", True, False),
    ],
)
def test_session_store_conformance_recovers_workspace_observation_through_public_runtime(
    session_store_case,
    crash_phase: str,
    expected_status: str,
    tool_ran: bool,
    recovery_repairs_observation: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(
            session_store_case,
            workspace_observation_crash_phase=crash_phase,
        )
        workspace_root = session_store_case[1] / f"workspace-observation-runtime-{crash_phase}"
        workspace_root.mkdir(exist_ok=True)
        artifact_root = session_store_case[1] / f"workspace-observation-artifacts-{crash_phase}"
        try:
            session_id = f"workspace-observation-runtime-{crash_phase}-{session_store_case[0]}"
            first_provider = _WorkspaceObservationRecoveryProvider()
            first_tool = _WorkspaceObservationRecoveryTool()
            first_app = CayuApp(session_store=store, enable_logging=False)
            first_app.register_provider(first_provider, default=True)
            first_app.register_environment(
                Environment(
                    _workspace_observation_recovery_environment_spec(),
                    workspace=LocalWorkspace(
                        workspace_root,
                        workspace_id="workspace-observation-runtime",
                    ),
                    artifact_store=LocalArtifactStore(
                        artifact_root,
                        store_id="workspace-observation-artifacts",
                    ),
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[first_tool],
            )

            assert isinstance(store, _WorkspaceObservationProcessLossStoreMixin)
            try:
                with pytest.raises(_SimulatedProcessLoss, match=crash_phase):
                    _ = [
                        event
                        async for event in first_app.run(
                            RunRequest(
                                agent_name="assistant",
                                session_id=session_id,
                                messages=[Message.text("user", "write once")],
                            )
                        )
                    ]
            finally:
                lost_process = store.disarm_workspace_observation_process_loss()
            assert lost_process is True
            assert first_provider.requests == 1
            assert first_tool.calls == (1 if tool_ran else 0)
            assert (workspace_root / "recovered.txt").exists() is tool_ran

            await interrupt_and_release_test_invocation(store, session_id)
            store = await _reopen_store(session_store_case, store)

            recovery_provider = _WorkspaceObservationRecoveryProvider()
            recovery_tool = _WorkspaceObservationRecoveryTool()
            recovery_app = CayuApp(session_store=store, enable_logging=False)
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_environment(
                Environment(
                    _workspace_observation_recovery_environment_spec(),
                    workspace=LocalWorkspace(
                        workspace_root,
                        workspace_id="workspace-observation-runtime",
                    ),
                    artifact_store=LocalArtifactStore(
                        artifact_root,
                        store_id="workspace-observation-artifacts",
                    ),
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[recovery_tool],
            )
            competing_provider = _WorkspaceObservationRecoveryProvider()
            competing_tool = _WorkspaceObservationRecoveryTool()
            competing_app = CayuApp(session_store=store, enable_logging=False)
            competing_app.register_provider(competing_provider, default=True)
            competing_app.register_environment(
                Environment(
                    _workspace_observation_recovery_environment_spec(),
                    workspace=LocalWorkspace(
                        workspace_root,
                        workspace_id="workspace-observation-runtime",
                    ),
                    artifact_store=LocalArtifactStore(
                        artifact_root,
                        store_id="workspace-observation-artifacts",
                    ),
                    binding=DeterministicWorkspaceBinding(),
                ),
                default=True,
            )
            competing_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[competing_tool],
            )
            recoveries = await asyncio.gather(
                recovery_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                ),
                competing_app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=session_id)
                ),
            )
            durable = await store.query_events(EventQuery(session_id=session_id))
            finalized = [
                record.event
                for record in durable
                if record.event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
            ]
            assert len(finalized) == 1
            assert finalized[0].payload["status"] == expected_status
            if crash_phase == "artifact-revision-after-intent":
                assert finalized[0].payload["revision_after_artifact_state"] == "intent"
            elif crash_phase == "artifact-revision-after-published":
                assert finalized[0].payload["revision_after_artifact_state"] == "orphaned"
            assert sum(
                IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in recovery.actions
                for recovery in recoveries
            ) == (1 if recovery_repairs_observation else 0)
            assert recovery_provider.requests == 0
            assert competing_provider.requests == 0
            assert recovery_tool.calls == 0
            assert competing_tool.calls == 0
            assert (
                workspace_observations_from_checkpoint(await store.load_checkpoint(session_id))
                == {}
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("publication_process_signal", [False, True])
def test_session_store_conformance_workspace_observation_commit_then_cancel_fans_out(
    session_store_case,
    publication_process_signal: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            session_id = f"workspace-observation-cancel-{session_store_case[0]}"
            session = await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            sink = InMemoryEventSink()
            event_writer = RuntimeEventWriter(
                session_store=store,
                budget_store=InMemoryBudgetStore(),
                event_sinks=(sink,),
            )
            lifecycle = WorkspaceObservationLifecycle(
                session_id=session_id,
                window_id="wmut-cancel-after-commit",
                source_run_epoch=session.run_epoch,
                binding_generation_id="wbind-cancel-after-commit",
                workspace_id="workspace-cancel-after-commit",
                observer="ConformanceWorkspaceBinding",
                observer_authority="configured",
                agent_name="assistant",
                environment_name="local",
                tool_name="mutate",
                tool_call_id="call-workspace-cancel",
                model_step_id="mstep-workspace-cancel",
                model_attempt_id="matt-workspace-cancel",
                tool_round_id="tround-workspace-cancel",
                model_step=1,
            )
            await publish_workspace_observation_transition(
                session_store=store,
                event_writer=event_writer,
                session=session,
                previous=None,
                current=lifecycle,
                phase="intent",
                intent_admission=_admit_test_workspace_observation_intent(lifecycle),
            )
            terminal = Event(
                id="evt-workspace-observation-cancel-terminal",
                type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                session_id=session_id,
                agent_name=lifecycle.agent_name,
                environment_name=lifecycle.environment_name,
                tool_name=lifecycle.tool_name,
                payload={
                    "window_id": lifecycle.window_id,
                    "session_run_epoch": lifecycle.source_run_epoch,
                    "binding_generation_id": lifecycle.binding_generation_id,
                    "workspace_id": lifecycle.workspace_id,
                    "observer": lifecycle.observer,
                    "artifact_store_id": None,
                    "tool_call_id": lifecycle.tool_call_id,
                    "model_step_id": lifecycle.model_step_id,
                    "model_attempt_id": lifecycle.model_attempt_id,
                    "tool_round_id": lifecycle.tool_round_id,
                    "model_step": lifecycle.model_step,
                    "referenced_artifact_count": 0,
                    "failed_artifact_count": 0,
                    "status": "ambiguous",
                    "detail_code": "worker_lost_before_tool_outcome_was_durable",
                },
            )
            original_publish = store.publish_runtime_publication
            committed = asyncio.Event()
            release_acknowledgement = asyncio.Event()
            process_signal_raised = False

            async def commit_then_hold_acknowledgement(session_id: str, **kwargs):
                nonlocal process_signal_raised
                result = await original_publish(session_id, **kwargs)
                if (
                    kwargs["request"].intent.get("phase") == "terminal"
                    and not process_signal_raised
                ):
                    committed.set()
                    await release_acknowledgement.wait()
                    if publication_process_signal:
                        process_signal_raised = True
                        raise SystemExit(17)
                return result

            store.publish_runtime_publication = (  # type: ignore[method-assign]
                commit_then_hold_acknowledgement
            )
            publishing = asyncio.create_task(
                publish_workspace_observation_transition(
                    session_store=store,
                    event_writer=event_writer,
                    session=session,
                    previous=lifecycle,
                    current=None,
                    phase="terminal",
                    terminal_status=WorkspaceObservationTerminalStatus.AMBIGUOUS,
                    terminal_detail_code="worker_lost_before_tool_outcome_was_durable",
                    events=(terminal,),
                )
            )
            await committed.wait()
            publishing.cancel("workspace observation acknowledgement cancelled")
            release_acknowledgement.set()
            if publication_process_signal:
                with pytest.raises(BaseExceptionGroup) as raised_group:
                    await publishing
                assert len(raised_group.value.exceptions) == 2
                process_signal, cancellation = raised_group.value.exceptions
                assert isinstance(process_signal, SystemExit)
                assert process_signal.args == (17,)
                assert isinstance(cancellation, asyncio.CancelledError)
                assert cancellation.args == ("workspace observation acknowledgement cancelled",)
                assert publishing.cancelling() == 1
                assert publishing.cancelled() is False
            else:
                with pytest.raises(asyncio.CancelledError) as raised:
                    await publishing
                assert raised.value.args == ("workspace observation acknowledgement cancelled",)
                assert publishing.cancelling() == 1
                assert publishing.cancelled() is True
            assert [event.type for event in sink.events] == [
                EventType.WORKSPACE_OBSERVATION_FINALIZED
            ]
            assert (
                workspace_observations_from_checkpoint(await store.load_checkpoint(session_id))
                == {}
            )
            receipt = await store.load_runtime_publication_receipt(
                session_id,
                f"workspace-observation:{lifecycle.window_id}:terminal",
            )
            assert receipt is not None
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_event_projection_preserves_private_authority(
    session_store_case,
) -> None:
    class RecordingSink(EventSink):
        def __init__(self) -> None:
            self.events: list[Event] = []

        async def emit(self, event: Event) -> None:
            self.events.append(event.model_copy(deep=True))

    async def run() -> None:
        store = await _open_store(
            session_store_case,
            public_authority_alias_codec=_public_authority_alias_codec(),
        )
        try:
            session_id = f"eventprojection{session_store_case[0]}"
            await store.create(
                RunRequest(
                    session_id=session_id,
                    agent_name="assistant",
                    messages=[],
                ),
                identity=_identity(),
            )
            sink = RecordingSink()
            app = CayuApp(
                session_store=store,
                event_sinks=[sink],
                enable_logging=False,
                secret_redactor=SecretRedactor(["-", "step", "legacycanary", "explicitsecret"]),
            )
            emitted = await app.emit_event(
                Event(
                    type=EventType.MODEL_STARTED,
                    session_id=session_id,
                    payload={"step": 1},
                )
            )
            with pytest.raises(ValueError, match=r"event\.event_id"):
                await app.emit_event(
                    Event(
                        id="caller-explicitsecret-event",
                        type=EventType.MODEL_STARTED,
                        session_id=session_id,
                        payload={"step": 2},
                    )
                )
            with pytest.raises(ValueError, match=r"event\.payload\.tool_call_id"):
                await app.emit_event(
                    Event(
                        type=EventType.TOOL_CALL_STARTED,
                        session_id=session_id,
                        payload={"tool_call_id": "legacycanary"},
                    )
                )
            legacy = Event(
                type="custom.legacycanary",
                session_id=session_id,
                payload={"legacycanary": "legacycanary"},
            )
            await store.append_event(session_id, legacy)
            recovered = await app.recover_persisted_event_side_effects()
            records = await store.query_events(EventQuery(session_id=session_id))

            assert public_event_sequence(emitted.id) == records[0].sequence
            assert [record.event.id for record in records[1:]] == [legacy.id]
            assert [event.id for event in sink.events] == [
                public_event_id(records[0].sequence),
                public_event_id(records[1].sequence),
            ]
            assert recovered[0].id == public_event_id(records[1].sequence)
            assert sink.events[0].payload == {"step": 1}
            assert sink.events[1].type == REDACTED_CUSTOM_EVENT_TYPE
            assert "legacycanary" not in repr(sink.events[1].model_dump(mode="json"))
        finally:
            await _close_store(store)

    asyncio.run(run())
