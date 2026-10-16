"""Every durable record that carries a session id admits the full admitted id."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from tests.core.test_agent_snapshots import _snapshot
from tests.core.test_model_failover_records import _progress, _selection
from tests.runtime.test_egress_authority_transitions import _authority

from cayu._browser_recording_store import BrowserRecordingStore
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.browser_recording import BrowserRecordingIdentity, BrowserRecordingPolicy
from cayu.budgets.quality import CostQualityAttemptOperation, PairedCostAttempt
from cayu.egress import EgressAuthorityTransitionState
from cayu.egress.authority import compare_egress_authority
from cayu.egress.transitions import _build_transition_record
from cayu.sessions.creation_claims import RuntimeSessionCreateClaimReference
from cayu.sessions.messaging import (
    SessionMessageActionRequest,
    SessionMessageInspection,
    SessionMessageQuery,
    SessionMessageSource,
)
from cayu.sessions.records import MAX_SESSION_ID_BYTES
from cayu.sessions.requests import RunRequest
from cayu.snapshots.base import (
    AgentSnapshotCaptureRequest,
    AgentSnapshotComponentSelector,
    AgentSnapshotResultBinding,
    AgentSnapshotTerminalDisposition,
)
from cayu.tasks.records import TaskSessionClosureClaim
from cayu.tools._browser_recording_guest import recording_configuration

LONG_SESSION_ID = "s" * MAX_SESSION_ID_BYTES
INSTANCE_ID = str(uuid4())


def _capture_request(session_id: str) -> AgentSnapshotCaptureRequest:
    snapshot = _snapshot()
    return AgentSnapshotCaptureRequest(
        capture_request_id="capture",
        subject=snapshot.subject,
        authority_scope_fingerprint=snapshot.authority_scope_fingerprint,
        components=tuple(
            AgentSnapshotComponentSelector(kind=component.kind, required=component.required)
            for component in snapshot.components
        ),
        session_id=session_id,
    )


def _egress_transition(session_id: str):
    expected = _authority(1, allow_post=False)
    target = _authority(2, allow_post=True)
    return _build_transition_record(
        transition_id="transition",
        session_id=session_id,
        environment_name="default",
        revision=1,
        state=EgressAuthorityTransitionState.AUTHORIZED,
        owner_fingerprint="1" * 64,
        expected_authority=expected,
        target_authority=target,
        classification=compare_egress_authority(expected, target),
        policy_identity="policy",
        actor=ResolutionActor(subject="operator", source=ResolutionActorSource.SYSTEM),
        authorization_reason="Authorized.",
        reason="Authorized.",
        strategy=target.cutover_strategy,
        source_environment_fingerprint="2" * 64,
    )


_RECORDS = {
    "model_failover_selection": lambda sid: _selection(session_id=sid),
    "model_failover_progress": lambda sid: _progress(session_id=sid),
    "task_session_closure_claim": lambda sid: TaskSessionClosureClaim(
        session_id=sid, plan_id="a" * 64, task_ids=("task",)
    ),
    "session_message_source": lambda sid: SessionMessageSource(
        session_id=sid, session_instance_id=INSTANCE_ID, run_epoch=0, transcript_cursor=0
    ),
    "session_message_query": lambda sid: SessionMessageQuery(session_id=sid),
    "session_message_action": lambda sid: SessionMessageActionRequest(
        session_id=sid,
        session_instance_id=INSTANCE_ID,
        queue_id="queue",
        idempotency_key="key",
        expected_revision="b" * 64,
        action="withdraw",
    ),
    "session_message_inspection": lambda sid: SessionMessageInspection(
        session_id=sid, session_instance_id=INSTANCE_ID
    ),
    "session_create_claim_reference": lambda sid: RuntimeSessionCreateClaimReference(
        session_id=sid,
        operation_id="operation",
        request_authority_key_id="key",
        request_authority_hmac_sha256="c" * 64,
        claim_id="d" * 64,
    ),
    "snapshot_capture_request": _capture_request,
    "snapshot_result_binding": lambda sid: AgentSnapshotResultBinding.create(
        trial=SimpleNamespace(fingerprint="e" * 64),
        session_id=sid,
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint="f" * 64,
        eval_result_revision="0" * 64,
        recorded_at=datetime(2026, 10, 15, tzinfo=UTC),
    ),
    "paired_cost_attempt": lambda sid: PairedCostAttempt(
        attempt_id="attempt",
        operation_id="operation",
        session_id=sid,
        role="baseline",
        operation=CostQualityAttemptOperation.AGENT_STEP,
        attempt_ordinal=1,
        usage_unavailable_reason="not reported",
    ),
    "egress_authority_transition": _egress_transition,
    "browser_recording_identity": lambda sid: BrowserRecordingIdentity(
        session_id=sid,
        session_instance_id=INSTANCE_ID,
        allocation_fingerprint="a" * 64,
        browser_id="browser",
        worker_instance="worker",
    ),
}


@pytest.mark.parametrize("build", _RECORDS.values(), ids=_RECORDS.keys())
def test_records_admit_every_admitted_session_id(build) -> None:
    RunRequest(agent_name="agent", session_id=LONG_SESSION_ID, messages=[])

    assert build(LONG_SESSION_ID).session_id == LONG_SESSION_ID
    with pytest.raises(ValueError, match=str(MAX_SESSION_ID_BYTES)):
        RunRequest(agent_name="agent", session_id=LONG_SESSION_ID + "s", messages=[])


def test_browser_recording_authorizes_every_admitted_session_id(tmp_path) -> None:
    async def check() -> None:
        store = BrowserRecordingStore(tmp_path / "recordings.sqlite")
        selected = BrowserRecordingPolicy(
            scope="run", allowed_origins=("https://public.test",), retention_seconds=60
        )
        config = await store.authorize_capture(
            session_id=LONG_SESSION_ID,
            policy=selected,
            guest_endpoint="wss://recording.test/browser-recordings/guest",
        )
        assert config.session_id == LONG_SESSION_ID
        guest = {
            **config.private_guest_configuration(),
            "identity": {
                "session_id": LONG_SESSION_ID,
                "session_instance_id": INSTANCE_ID,
                "allocation_fingerprint": "a" * 64,
            },
        }
        assert recording_configuration(guest) == guest
        await store.begin(
            selected,
            {
                **guest["identity"],
                "browser_id": "browser",
                "worker_instance": "worker",
            },
        )
        with pytest.raises(ValueError):
            await store.authorize_capture(
                session_id=LONG_SESSION_ID + "s",
                policy=selected.model_copy(update={"scope": "other"}),
                guest_endpoint="wss://recording.test/browser-recordings/guest",
            )

    asyncio.run(check())


def _declared_max_length(model: type, field_name: str) -> int | None:
    return next(
        (
            item.max_length
            for item in model.model_fields[field_name].metadata
            if hasattr(item, "max_length")
        ),
        None,
    )


def test_session_id_fields_declare_the_admitted_bound() -> None:
    from cayu.evals.operation_outcomes import OperationOutcomeEvidence
    from cayu.evals.process_inspection import EvalSessionReferenceV1
    from cayu.memory.execution import (
        MemoryInterventionExecutionRecord,
        MemoryInterventionRuntimeResult,
    )
    from cayu.messages import PeerContentPart
    from cayu.runtime.session_steering import StopAfterCurrentToolRoundRequest
    from cayu.sessions.invocation_release import InvocationReleaseEvidence
    from cayu.tools.shared_artifacts import (
        SharedArtifactGrant,
        SharedArtifactMaterializationReceipt,
        SharedArtifactRef,
    )
    from cayu.workspaces.branches import WorkspaceBranchAuthority, WorkspaceBranchRecoveryRequest

    fields = (
        (StopAfterCurrentToolRoundRequest, "session_id"),
        (MemoryInterventionExecutionRecord, "session_id"),
        (MemoryInterventionRuntimeResult, "session_id"),
        (WorkspaceBranchAuthority, "session_id"),
        (WorkspaceBranchRecoveryRequest, "session_id"),
        (EvalSessionReferenceV1, "session_id"),
        (PeerContentPart, "sender_session_id"),
        (SharedArtifactRef, "source_session_id"),
        (SharedArtifactGrant, "root_session_id"),
        (SharedArtifactMaterializationReceipt, "destination_session_id"),
        (InvocationReleaseEvidence, "session_id"),
        (OperationOutcomeEvidence, "session_id"),
    )
    assert {
        f"{model.__name__}.{name}": _declared_max_length(model, name) for model, name in fields
    } == {f"{model.__name__}.{name}": MAX_SESSION_ID_BYTES for model, name in fields}


def test_memory_intervention_runtime_result_accepts_every_admitted_session_id() -> None:
    from tests.core.test_memory_intervention_execution import _digest, _empty_attribution

    from cayu.evals.memory_attribution import standard_eval_memory_attribution_bounds
    from cayu.memory.execution import MemoryInterventionRuntimeResult

    result = MemoryInterventionRuntimeResult(
        session_id=LONG_SESSION_ID,
        terminal_disposition=AgentSnapshotTerminalDisposition.COMPLETED,
        runtime_evidence_fingerprint=_digest("runtime"),
        terminal_evidence_available=True,
        expected_receipt_count=0,
        expected_exposure_count=0,
        effective_attribution_bounds=standard_eval_memory_attribution_bounds(),
        attribution=_empty_attribution(),
    )

    assert result.session_id == LONG_SESSION_ID


def test_runtime_evidence_reports_every_admitted_session_id() -> None:
    from tests.core.test_dispatch import FakeProvider, _batch

    from cayu.agents import AgentSpec
    from cayu.applications import CayuApp
    from cayu.messages import Message
    from cayu.runtime.evidence import RuntimeEvidenceRequest, runtime_evidence
    from cayu.sessions.memory import InMemorySessionStore

    async def run() -> None:
        app = CayuApp(session_store=InMemorySessionStore(), enable_logging=False)
        app.register_provider(FakeProvider([_batch("done")]), default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake-model"))
        async for _event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=LONG_SESSION_ID,
                messages=[Message.text("user", "hello")],
            )
        ):
            pass

        report = await runtime_evidence(
            app,
            RuntimeEvidenceRequest(root_session_id=LONG_SESSION_ID, max_sessions=1, max_events=100),
        )

        assert report.root_session_id == LONG_SESSION_ID
        assert [session.session_id for session in report.sessions] == [LONG_SESSION_ID]
        assert report.scope.causal_budget_id == LONG_SESSION_ID

    asyncio.run(run())
