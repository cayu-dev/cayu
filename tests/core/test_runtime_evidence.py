from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from cayu import (
    CayuApp,
    Event,
    EventType,
    InMemorySessionStore,
    InMemoryTaskStore,
    Message,
    ModelPrice,
    PostgresSessionStore,
    PriceBook,
    PricingContextSelector,
    RunRequest,
    RuntimeEvidenceCostStatus,
    RuntimeEvidenceError,
    RuntimeEvidenceErrorCode,
    RuntimeEvidenceOperation,
    RuntimeEvidenceRequest,
    RuntimeEvidenceWarningCode,
    SessionIdentity,
    SessionLineageNode,
    SessionLineageResult,
    SessionStatus,
    SessionStore,
    SQLiteSessionStore,
    TaskCreate,
    runtime_evidence,
)
from cayu.runtime.sessions import EventQueryResultTooLarge, SessionListResult, SessionQuery
from cayu.storage.migrations import SchemaMode


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="provider", model="model")


async def _reset_postgres_event_sequence(postgres_dsn: str) -> None:
    import psycopg

    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        async with connection.cursor() as cursor:
            await cursor.execute("TRUNCATE cayu_events RESTART IDENTITY CASCADE")
        await connection.commit()


async def _create_session(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
    causal_budget_id: str = "budget-1",
    labels: dict[str, str] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    await store.create(
        RunRequest(
            agent_name=f"agent-{session_id}",
            session_id=session_id,
            parent_session_id=parent_session_id,
            causal_budget_id=causal_budget_id,
            labels=labels or {},
            messages=[Message.text("user", f"secret prompt for {session_id}")],
            metadata=metadata or {"raw_prompt": f"secret metadata for {session_id}"},
        ),
        identity=_identity(),
    )
    await store.append_event(
        session_id,
        Event(
            id=f"{session_id}-started",
            type=EventType.SESSION_STARTED,
            session_id=session_id,
            payload={"messages": [f"secret event prompt for {session_id}"]},
        ),
    )


def test_runtime_evidence_request_requires_explicit_scope_bounds() -> None:
    with pytest.raises(ValidationError):
        RuntimeEvidenceRequest.model_validate({"root_session_id": "root"})


def test_runtime_evidence_can_project_only_the_selected_root_session() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "root")
        await _create_session(store, "child", parent_session_id="root")
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="root",
                max_sessions=10,
                max_events=20,
                include_descendants=False,
            ),
        )

    report = asyncio.run(scenario())

    assert tuple(session.session_id for session in report.sessions) == ("root",)
    assert report.scope.descendant_session_ids == ("root",)
    assert report.scope.causal_budget_session_ids is None


def test_runtime_evidence_fails_with_a_typed_error_when_scopes_exceed_report_capacity() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store, "root")
        for index in range(499):
            await _create_session(
                store,
                f"descendant-{index}",
                parent_session_id="root",
                causal_budget_id=f"descendant-budget-{index}",
            )
        await _create_session(store, "causal-peer")

        with pytest.raises(RuntimeEvidenceError) as caught:
            await runtime_evidence(
                CayuApp(session_store=store, enable_logging=False),
                RuntimeEvidenceRequest(
                    root_session_id="root",
                    max_sessions=500,
                    max_events=2_000,
                    include_causal_budget=True,
                    max_causal_budget_sessions=2,
                ),
            )

        assert caught.value.code is RuntimeEvidenceErrorCode.SESSION_LIMIT_EXCEEDED
        assert caught.value.limit == 500
        assert caught.value.observed == 501

    asyncio.run(scenario())


def test_runtime_evidence_allows_a_session_to_change_while_causal_scope_is_listed() -> None:
    class ConcurrentCausalStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        def __init__(self) -> None:
            super().__init__()
            self.updated = False

        async def list_sessions(
            self,
            query: SessionQuery | None = None,
        ) -> SessionListResult:
            assert query is not None
            if query.causal_budget_id is not None and not self.updated:
                self.updated = True
                await self.update_status("root", SessionStatus.COMPLETED)
            return await super().list_sessions(query)

    async def scenario():
        store = ConcurrentCausalStore()
        await _create_session(store, "root")
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="root",
                max_sessions=10,
                max_events=20,
                include_causal_budget=True,
                max_causal_budget_sessions=10,
            ),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].status is SessionStatus.COMPLETED


def test_runtime_evidence_keeps_partial_task_rows_and_session_links_consistent() -> None:
    async def scenario():
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        await _create_session(store, "root")
        for index in range(501):
            await tasks.create_task(
                TaskCreate(
                    task_id=f"task-{index}",
                    type="test",
                    title=f"Task {index}",
                    description="runtime evidence task",
                    session_id="root",
                    input={},
                )
            )
        return await runtime_evidence(
            CayuApp(session_store=store, task_store=tasks, enable_logging=False),
            RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=20),
        )

    report = asyncio.run(scenario())
    root = report.sessions[0]

    task_ids = {task.task_id for task in report.tasks}
    assert len(task_ids) == 500
    assert set(root.task_ids) == task_ids
    assert RuntimeEvidenceWarningCode.TASK_EVIDENCE_PARTIAL in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_labels_missing_tool_call_identity_as_malformed() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "root")
        await store.append_event(
            "root",
            Event(
                id="tool-call-without-identity",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="root",
            ),
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=20),
        )

    report = asyncio.run(scenario())

    assert RuntimeEvidenceWarningCode.MALFORMED_TOOL_CALL in {
        warning.code for warning in report.warnings
    }
    assert RuntimeEvidenceWarningCode.UNKNOWN_EVENT_TYPE not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_receipt_reconciliation_supersedes_recorded_state() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "root")
        await store.append_events(
            "root",
            [
                Event(
                    id="receipt-recorded",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="root",
                    payload={
                        "tool_call_id": "call-1",
                        "result": {
                            "structured": {
                                "receipt_id": "receipt-1",
                                "reconciliation_state": "recorded",
                            }
                        },
                    },
                ),
                Event(
                    id="receipt-reconciled",
                    type=EventType.PROVIDER_OPERATION_RECONCILED,
                    session_id="root",
                    payload={
                        "receipt_id": "receipt-1",
                        "reconciliation_state": "reconciled",
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=20),
        )

    report = asyncio.run(scenario())
    receipt = report.sessions[0].receipts[0]

    assert receipt.tool_call_id == "call-1"
    assert receipt.reconciliation_state == "reconciled"
    assert receipt.source_ref.event_id == "receipt-reconciled"
    assert RuntimeEvidenceWarningCode.MALFORMED_RECEIPT not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_projects_safe_workspace_mutation_and_finalization() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-root")
        common = {
            "window_id": "window-1",
            "workspace_id": "workspace-1",
            "tool_call_id": "call-1",
            "tool_round_id": "round-1",
        }
        await store.append_events(
            "workspace-root",
            [
                Event(
                    id="workspace-before",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-root",
                    payload={
                        **common,
                        "phase": "before",
                        "status": "supported",
                        "revision": "revision-before",
                        "branch": "secret-feature-branch",
                        "path_scope": "complete",
                        "paths": [{"path": "secret-plan.txt", "kind": "file"}],
                        "total_paths": 1,
                        "detail_code": None,
                    },
                ),
                Event(
                    id="workspace-after",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-root",
                    payload={
                        **common,
                        "phase": "after",
                        "status": "supported",
                        "revision": "revision-after",
                        "branch": "secret-feature-branch",
                        "path_scope": "changed",
                        "paths": [{"path": "secret-plan.txt", "kind": "file"}],
                        "total_paths": 1,
                        "detail_code": None,
                    },
                ),
                Event(
                    id="workspace-delta",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-root",
                    payload={
                        **common,
                        "status": "changed",
                        "before_revision": "revision-before",
                        "after_revision": "revision-after",
                        "paths": [{"path": "secret-plan.txt", "change": "modified"}],
                        "total_paths": 1,
                        "head_changed": False,
                        "branch_changed": False,
                        "detail_code": None,
                        "attribution": {
                            "confidence": "exclusive_tool",
                            "writer_isolation": "exclusive",
                            "overlap_detected": False,
                            "direct_reconciliation": "consistent",
                            "detail_code": "exclusive_writer_isolation_verified",
                        },
                        "writer_isolation": {
                            "before": {
                                "status": "exclusive",
                                "mechanism": "secret-lease-mechanism",
                                "generation": "secret-generation",
                            },
                            "after": {
                                "status": "exclusive",
                                "mechanism": "secret-lease-mechanism",
                                "generation": "secret-generation",
                            },
                        },
                        "direct_mutations": {
                            "operations": [{"path": "secret-plan.txt", "method": "write_bytes"}]
                        },
                        "manifest_artifact_id": "artifact-delta",
                        "manifest_artifact_sha256": "a" * 64,
                        "manifest_artifact_size_bytes": 123,
                    },
                ),
                Event(
                    id="workspace-terminal",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-root",
                    payload={
                        **common,
                        "status": "complete",
                        "detail_code": None,
                        "mutation_event_id": "workspace-delta",
                        "referenced_artifact_count": 1,
                        "failed_artifact_count": 0,
                    },
                ),
                Event(
                    id="workspace-finalized",
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                    session_id="workspace-root",
                    payload={
                        "final_revision": {
                            "workspace_id": "workspace-1",
                            "observer": "GitRepositoryBinding",
                            "status": "supported",
                            "revision": "revision-final",
                            "head_revision": "head-secret",
                            "branch": "secret-feature-branch",
                            "path_scope": "changed",
                            "total_paths": 2,
                            "detail_code": None,
                            "finalization_delta": {
                                "attribution_confidence": ("unattributed_finalization_change"),
                                "status": "changed",
                                "before_revision": "revision-after",
                                "after_revision": "revision-final",
                                "paths": [{"path_sha256": "b" * 64, "change": "added"}],
                                "retained_paths": 1,
                                "total_paths": 1,
                                "truncated": False,
                                "head_changed": True,
                                "branch_changed": False,
                                "detail_code": None,
                            },
                        }
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-root",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    session = report.sessions[0]

    assert report.schema_version == 4
    assert len(session.workspace_mutations) == 1
    mutation = session.workspace_mutations[0]
    assert mutation.window_id == "window-1"
    assert mutation.before is not None and mutation.before.revision == "revision-before"
    assert mutation.after is not None and mutation.after.revision == "revision-after"
    assert mutation.delta is not None
    assert mutation.delta.status == "changed"
    assert mutation.delta.total_paths == 1
    assert mutation.attribution is not None
    assert mutation.attribution.confidence == "exclusive_tool"
    assert mutation.terminal is not None and mutation.terminal.status == "complete"
    assert mutation.artifacts[0].artifact_id == "artifact-delta"
    assert [ref.event_id for ref in mutation.source_refs] == [
        "workspace-before",
        "workspace-after",
        "workspace-delta",
        "workspace-terminal",
    ]
    assert session.workspace_finalization is not None
    assert session.workspace_finalization.revision == "revision-final"
    assert session.workspace_finalization.delta is not None
    assert session.workspace_finalization.delta.attribution_confidence == (
        "unattributed_finalization_change"
    )
    assert session.workspace_finalization.source_refs[0].event_id == "workspace-finalized"

    serialized = report.model_dump_json()
    for secret in (
        "secret-plan.txt",
        "secret-feature-branch",
        "head-secret",
        "secret-lease-mechanism",
        "secret-generation",
    ):
        assert secret not in serialized


def test_runtime_evidence_warns_and_omits_conflicting_workspace_evidence() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-conflict")
        common = {
            "window_id": "window-conflict",
            "workspace_id": "workspace-1",
            "tool_call_id": "call-1",
            "tool_round_id": "round-1",
            "phase": "before",
            "status": "supported",
            "path_scope": "complete",
            "paths": [],
            "total_paths": 0,
            "detail_code": None,
        }
        await store.append_events(
            "workspace-conflict",
            [
                Event(
                    id="before-one",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-conflict",
                    payload={**common, "revision": "revision-one"},
                ),
                Event(
                    id="before-two",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-conflict",
                    payload={**common, "revision": "revision-two"},
                ),
                Event(
                    id="bad-workspace-delta",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-conflict",
                    payload={
                        "window_id": "window-bad",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-2",
                        "status": "changed",
                        "before_revision": "before",
                        "after_revision": "after",
                        "total_paths": -1,
                        "head_changed": False,
                        "branch_changed": False,
                    },
                ),
                Event(
                    id="final-revision-one",
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                    session_id="workspace-conflict",
                    payload={
                        "final_revision": {
                            "workspace_id": "workspace-1",
                            "status": "supported",
                            "revision": "final-one",
                            "path_scope": "complete",
                            "total_paths": 0,
                            "detail_code": None,
                        }
                    },
                ),
                Event(
                    id="final-revision-two",
                    type=EventType.SESSION_COMPLETED,
                    session_id="workspace-conflict",
                    payload={
                        "final_revision": {
                            "workspace_id": "workspace-1",
                            "status": "supported",
                            "revision": "final-two",
                            "path_scope": "complete",
                            "total_paths": 0,
                            "detail_code": None,
                        }
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-conflict",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].workspace_mutations == ()
    assert report.sessions[0].workspace_finalization is None
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_workspace_duplicate_replay_converges() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-replay")
        common = {
            "window_id": "window-replay",
            "workspace_id": "workspace-1",
            "tool_call_id": "call-1",
            "tool_round_id": "round-1",
        }
        before = {
            **common,
            "phase": "before",
            "status": "unsupported",
            "revision": None,
            "path_scope": "complete",
            "paths": [],
            "total_paths": 0,
            "detail_code": "revision_observation_unsupported",
        }
        final_revision = {
            "workspace_id": "workspace-1",
            "status": "unsupported",
            "revision": None,
            "path_scope": "complete",
            "total_paths": 0,
            "detail_code": "revision_observation_unsupported",
            "finalization_delta": {
                "attribution_confidence": "unattributed_finalization_change",
                "status": "incomplete",
                "before_revision": None,
                "after_revision": None,
                "total_paths": 0,
                "head_changed": False,
                "branch_changed": False,
                "detail_code": "finalization_baseline_unavailable",
            },
        }
        await store.append_events(
            "workspace-replay",
            [
                Event(
                    id="before-original",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-replay",
                    payload=before,
                ),
                Event(
                    id="before-replayed",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-replay",
                    payload=before,
                ),
                Event(
                    id="terminal-recovered",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-replay",
                    payload={
                        **common,
                        "status": "incomplete",
                        "detail_code": "workspace_revision_evidence_incomplete",
                    },
                ),
                Event(
                    id="finalize-completed",
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                    session_id="workspace-replay",
                    payload={"final_revision": final_revision},
                ),
                Event(
                    id="session-completed",
                    type=EventType.SESSION_COMPLETED,
                    session_id="workspace-replay",
                    payload={"final_revision": final_revision},
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-replay",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    session = report.sessions[0]

    assert [ref.event_id for ref in session.workspace_mutations[0].source_refs] == [
        "before-original",
        "before-replayed",
        "terminal-recovered",
    ]
    assert session.workspace_finalization is not None
    assert [ref.event_id for ref in session.workspace_finalization.source_refs] == [
        "finalize-completed",
        "session-completed",
    ]
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_keeps_recovered_workspace_terminal_evidence_distinct() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-recovered")
        await store.append_event(
            "workspace-recovered",
            Event(
                id="recovered-terminal",
                type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                session_id="workspace-recovered",
                payload={
                    "window_id": "window-recovered",
                    "workspace_id": "workspace-1",
                    "tool_call_id": "call-1",
                    "tool_round_id": "round-1",
                    "status": "ambiguous",
                    "detail_code": "worker_lost_before_tool_outcome_was_durable",
                    "attribution": {
                        "confidence": "concurrent_ambiguity",
                        "writer_isolation": "unknown",
                        "overlap_detected": False,
                        "direct_reconciliation": "not_observed",
                        "detail_code": "workspace_attribution_recovery_incomplete",
                    },
                    "revision_before_artifact_id": "artifact-before",
                    "revision_before_artifact_sha256": "c" * 64,
                    "revision_before_artifact_size_bytes": 17,
                    "revision_before_artifact_state": "missing",
                },
            ),
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-recovered",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    recovered = report.sessions[0].workspace_mutations[0]

    assert recovered.before is None
    assert recovered.after is None
    assert recovered.delta is None
    assert recovered.terminal is not None and recovered.terminal.status == "ambiguous"
    assert recovered.attribution is not None
    assert recovered.attribution.confidence == "concurrent_ambiguity"
    assert recovered.artifacts[0].state == "missing"


def test_runtime_evidence_recovery_supersedes_a_referenced_artifact_state() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-artifact-recovery")
        common = {
            "window_id": "window-recovery",
            "workspace_id": "workspace-1",
            "tool_call_id": "call-1",
        }
        await store.append_events(
            "workspace-artifact-recovery",
            [
                Event(
                    id="before-referenced",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-artifact-recovery",
                    payload={
                        **common,
                        "phase": "before",
                        "status": "supported",
                        "revision": "revision-before",
                        "path_scope": "complete",
                        "total_paths": 1,
                        "detail_code": None,
                        "manifest_artifact_id": "artifact-before",
                        "manifest_artifact_sha256": "d" * 64,
                        "manifest_artifact_size_bytes": 18,
                    },
                ),
                Event(
                    id="terminal-missing",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-artifact-recovery",
                    payload={
                        **common,
                        "status": "incomplete",
                        "detail_code": "referenced_workspace_artifact_missing",
                        "revision_before_artifact_id": "artifact-before",
                        "revision_before_artifact_sha256": "d" * 64,
                        "revision_before_artifact_size_bytes": 18,
                        "revision_before_artifact_state": "missing",
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-artifact-recovery",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    mutation = report.sessions[0].workspace_mutations[0]

    assert mutation.artifacts[0].state == "missing"
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_keeps_workspace_delta_degradation_states_distinct() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-states")
        events = []
        details = {
            "unsupported": "observation_not_complete",
            "truncated": "workspace_evidence_quarantined",
            "failed": "revision_comparison_failed",
            "incomplete": "observation_not_complete",
        }
        for status in ("no_change", "unsupported", "truncated", "failed", "incomplete"):
            complete = status == "no_change"
            events.append(
                Event(
                    id=f"delta-{status}",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-states",
                    payload={
                        "window_id": f"window-{status}",
                        "workspace_id": "workspace-1",
                        "tool_call_id": f"call-{status}",
                        "status": status,
                        "before_revision": "same" if complete else None,
                        "after_revision": "same" if complete else None,
                        "total_paths": 0,
                        "head_changed": False,
                        "branch_changed": False,
                        "detail_code": details.get(status),
                    },
                )
            )
        await store.append_events("workspace-states", events)
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-states",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())

    assert tuple(
        mutation.delta.status
        for mutation in report.sessions[0].workspace_mutations
        if mutation.delta is not None
    ) == ("no_change", "unsupported", "truncated", "failed", "incomplete")


def test_runtime_evidence_accepts_current_workspace_producer_detail_codes() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-producer-details")
        await store.append_events(
            "workspace-producer-details",
            [
                Event(
                    id="file-limit-before",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-producer-details",
                    payload={
                        "window_id": "window-file-limit",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-file-limit",
                        "phase": "before",
                        "status": "truncated",
                        "revision": None,
                        "path_scope": "complete",
                        "total_paths": 1,
                        "detail_code": "file_byte_limit_exceeded",
                    },
                ),
                Event(
                    id="file-limit-terminal",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-producer-details",
                    payload={
                        "window_id": "window-file-limit",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-file-limit",
                        "status": "incomplete",
                        "detail_code": "workspace_revision_evidence_incomplete",
                    },
                ),
                Event(
                    id="artifact-write-delta",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-producer-details",
                    payload={
                        "window_id": "window-artifact-write",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-artifact-write",
                        "status": "truncated",
                        "before_revision": None,
                        "after_revision": None,
                        "total_paths": 0,
                        "head_changed": False,
                        "branch_changed": False,
                        "detail_code": "manifest_artifact_write_failed",
                    },
                ),
                Event(
                    id="artifact-write-terminal",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-producer-details",
                    payload={
                        "window_id": "window-artifact-write",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-artifact-write",
                        "status": "incomplete",
                        "detail_code": "workspace_revision_evidence_incomplete",
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-producer-details",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    first, second = report.sessions[0].workspace_mutations

    assert first.before is not None
    assert first.before.detail_code == "file_byte_limit_exceeded"
    assert second.delta is not None
    assert second.delta.detail_code == "manifest_artifact_write_failed"
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE not in {
        warning.code for warning in report.warnings
    }


@pytest.mark.parametrize(
    ("delta_before", "delta_after"),
    [
        ("different-before", "observed-after"),
        ("observed-before", "different-after"),
    ],
)
def test_runtime_evidence_rejects_contradictory_workspace_revision_chain(
    delta_before: str,
    delta_after: str,
) -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-chain-conflict")
        common = {
            "window_id": "window-conflict",
            "workspace_id": "workspace-1",
            "tool_call_id": "call-1",
        }
        await store.append_events(
            "workspace-chain-conflict",
            [
                Event(
                    id="chain-before",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-chain-conflict",
                    payload={
                        **common,
                        "phase": "before",
                        "status": "supported",
                        "revision": "observed-before",
                        "path_scope": "complete",
                        "total_paths": 0,
                        "detail_code": None,
                    },
                ),
                Event(
                    id="chain-after",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-chain-conflict",
                    payload={
                        **common,
                        "phase": "after",
                        "status": "supported",
                        "revision": "observed-after",
                        "path_scope": "complete",
                        "total_paths": 0,
                        "detail_code": None,
                    },
                ),
                Event(
                    id="chain-delta",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-chain-conflict",
                    payload={
                        **common,
                        "status": "changed",
                        "before_revision": delta_before,
                        "after_revision": delta_after,
                        "total_paths": 1,
                        "head_changed": False,
                        "branch_changed": False,
                        "detail_code": None,
                    },
                ),
                Event(
                    id="chain-terminal",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-chain-conflict",
                    payload={**common, "status": "complete", "detail_code": None},
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-chain-conflict",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].workspace_mutations == ()
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_rejects_complete_terminal_for_incomplete_delta() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-terminal-conflict")
        common = {
            "window_id": "window-terminal-conflict",
            "workspace_id": "workspace-1",
            "tool_call_id": "call-1",
        }
        await store.append_events(
            "workspace-terminal-conflict",
            [
                Event(
                    id="incomplete-delta",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-terminal-conflict",
                    payload={
                        **common,
                        "status": "incomplete",
                        "before_revision": None,
                        "after_revision": None,
                        "total_paths": 0,
                        "head_changed": False,
                        "branch_changed": False,
                        "detail_code": "observation_not_complete",
                    },
                ),
                Event(
                    id="false-complete-terminal",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-terminal-conflict",
                    payload={**common, "status": "complete", "detail_code": None},
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-terminal-conflict",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].workspace_mutations == ()
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE in {
        warning.code for warning in report.warnings
    }


@pytest.mark.parametrize(
    ("delta_after", "attribution_confidence"),
    [
        ("different-final", "unattributed_finalization_change"),
        ("final-observation", "exclusive_tool"),
    ],
)
def test_runtime_evidence_rejects_incoherent_workspace_finalization_delta(
    delta_after: str,
    attribution_confidence: str,
) -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-finalization-conflict")
        await store.append_event(
            "workspace-finalization-conflict",
            Event(
                id="incoherent-finalization",
                type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                session_id="workspace-finalization-conflict",
                payload={
                    "binding_generation_id": "generation-1",
                    "final_revision": {
                        "workspace_id": "workspace-1",
                        "status": "supported",
                        "revision": "final-observation",
                        "path_scope": "complete",
                        "total_paths": 0,
                        "detail_code": None,
                        "finalization_delta": {
                            "attribution_confidence": attribution_confidence,
                            "status": "changed",
                            "before_revision": "before",
                            "after_revision": delta_after,
                            "total_paths": 1,
                            "head_changed": False,
                            "branch_changed": False,
                            "detail_code": None,
                        },
                    },
                },
            ),
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-finalization-conflict",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].workspace_finalization is None
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_selects_latest_workspace_binding_generation() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-resumed-finalization")

        def final_revision(revision: str) -> dict[str, object]:
            return {
                "workspace_id": "workspace-1",
                "status": "supported",
                "revision": revision,
                "path_scope": "complete",
                "total_paths": 0,
                "detail_code": None,
            }

        await store.append_events(
            "workspace-resumed-finalization",
            [
                Event(
                    id="generation-1-finalized",
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                    session_id="workspace-resumed-finalization",
                    payload={
                        "binding_generation_id": "generation-1",
                        "final_revision": final_revision("revision-1"),
                    },
                ),
                Event(
                    id="generation-2-finalized",
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                    session_id="workspace-resumed-finalization",
                    payload={
                        "binding_generation_id": "generation-2",
                        "final_revision": final_revision("revision-2"),
                    },
                ),
                Event(
                    id="generation-1-replayed",
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                    session_id="workspace-resumed-finalization",
                    payload={
                        "binding_generation_id": "generation-1",
                        "final_revision": final_revision("revision-1"),
                    },
                ),
                Event(
                    id="generation-2-session-terminal",
                    type=EventType.SESSION_COMPLETED,
                    session_id="workspace-resumed-finalization",
                    payload={"final_revision": final_revision("revision-2")},
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-resumed-finalization",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    finalization = report.sessions[0].workspace_finalization

    assert finalization is not None
    assert finalization.binding_generation_id == "generation-2"
    assert finalization.revision == "revision-2"
    assert [ref.event_id for ref in finalization.source_refs] == [
        "generation-2-finalized",
        "generation-2-session-terminal",
    ]
    assert RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_retains_workspace_terminal_recovery_epochs() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-terminal-epochs")
        events = []
        for window_id, recovery_run_epoch in (("direct", 1), ("recovered", 2)):
            common = {
                "window_id": f"window-{window_id}",
                "workspace_id": "workspace-1",
                "tool_call_id": f"call-{window_id}",
            }
            events.extend(
                [
                    Event(
                        id=f"{window_id}-delta",
                        type=EventType.WORKSPACE_MUTATION_RECORDED,
                        session_id="workspace-terminal-epochs",
                        payload={
                            **common,
                            "status": "changed",
                            "before_revision": f"{window_id}-before",
                            "after_revision": f"{window_id}-after",
                            "total_paths": 1,
                            "head_changed": False,
                            "branch_changed": False,
                            "detail_code": None,
                        },
                    ),
                    Event(
                        id=f"{window_id}-terminal",
                        type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                        session_id="workspace-terminal-epochs",
                        payload={
                            **common,
                            "session_run_epoch": 1,
                            "recovery_run_epoch": recovery_run_epoch,
                            "binding_generation_id": "generation-1",
                            "status": "complete",
                            "detail_code": None,
                        },
                    ),
                ]
            )
        await store.append_events("workspace-terminal-epochs", events)
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-terminal-epochs",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    direct, recovered = report.sessions[0].workspace_mutations

    assert direct.terminal is not None
    assert direct.terminal.session_run_epoch == direct.terminal.recovery_run_epoch == 1
    assert direct.terminal.binding_generation_id == "generation-1"
    assert recovered.terminal is not None
    assert recovered.terminal.session_run_epoch == 1
    assert recovered.terminal.recovery_run_epoch == 2
    assert recovered.terminal.binding_generation_id == "generation-1"


def test_runtime_evidence_warns_for_malformed_workspace_shapes_without_crashing() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "workspace-malformed")
        await store.append_events(
            "workspace-malformed",
            [
                Event(
                    id="invalid-enum",
                    type=EventType.WORKSPACE_REVISION_OBSERVED,
                    session_id="workspace-malformed",
                    payload={
                        "window_id": "enum-window",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-1",
                        "phase": "middle",
                        "status": "supported",
                        "revision": "revision",
                        "path_scope": "complete",
                        "total_paths": 0,
                    },
                ),
                Event(
                    id="invalid-digest",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-malformed",
                    payload={
                        "window_id": "digest-window",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-2",
                        "status": "changed",
                        "before_revision": "before",
                        "after_revision": "after",
                        "total_paths": 1,
                        "head_changed": False,
                        "branch_changed": False,
                        "manifest_artifact_id": "artifact-1",
                        "manifest_artifact_sha256": "not-a-digest",
                        "manifest_artifact_size_bytes": 10,
                    },
                ),
                Event(
                    id="invalid-identifier",
                    type=EventType.WORKSPACE_OBSERVATION_FINALIZED,
                    session_id="workspace-malformed",
                    payload={
                        "window_id": " ",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-3",
                        "status": "failed",
                        "detail_code": "receipt_publication_failed",
                    },
                ),
                Event(
                    id="invalid-count",
                    type=EventType.WORKSPACE_MUTATION_RECORDED,
                    session_id="workspace-malformed",
                    payload={
                        "window_id": "count-window",
                        "workspace_id": "workspace-1",
                        "tool_call_id": "call-4",
                        "status": "changed",
                        "before_revision": "before",
                        "after_revision": "after",
                        "total_paths": -1,
                        "head_changed": False,
                        "branch_changed": False,
                    },
                ),
                Event(
                    id="invalid-finalization",
                    type=EventType.ENVIRONMENT_BINDING_FINALIZE_FAILED,
                    session_id="workspace-malformed",
                    payload={
                        "final_revision": {
                            "workspace_id": "workspace-1",
                            "status": "mystery",
                            "revision": None,
                            "path_scope": "complete",
                            "total_paths": 0,
                        }
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-malformed",
                max_sessions=10,
                max_events=20,
            ),
        )

    report = asyncio.run(scenario())
    session = report.sessions[0]
    malformed = [
        warning
        for warning in report.warnings
        if warning.code is RuntimeEvidenceWarningCode.MALFORMED_WORKSPACE_EVIDENCE
    ]

    assert session.workspace_mutations == ()
    assert session.workspace_finalization is None
    assert {warning.event_id for warning in malformed} == {
        "invalid-enum",
        "invalid-digest",
        "invalid-identifier",
        "invalid-count",
        "invalid-finalization",
    }


def test_runtime_evidence_workspace_projection_matches_memory_and_sqlite(tmp_path: Path) -> None:
    async def build(store: SessionStore):
        await _create_session(store, "workspace-parity")
        await store.append_event(
            "workspace-parity",
            Event(
                id="parity-delta",
                type=EventType.WORKSPACE_MUTATION_RECORDED,
                session_id="workspace-parity",
                payload={
                    "window_id": "window-parity",
                    "workspace_id": "workspace-1",
                    "tool_call_id": "call-1",
                    "status": "no_change",
                    "before_revision": "same",
                    "after_revision": "same",
                    "total_paths": 0,
                    "head_changed": False,
                    "branch_changed": False,
                    "detail_code": None,
                },
            ),
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="workspace-parity",
                max_sessions=10,
                max_events=20,
            ),
        )

    async def scenario():
        memory = InMemorySessionStore()
        sqlite = SQLiteSessionStore(tmp_path / "workspace-parity.sqlite")
        memory_report = await build(memory)
        sqlite_report = await build(sqlite)
        await sqlite.close()
        return memory_report, sqlite_report

    memory_report, sqlite_report = asyncio.run(scenario())

    assert memory_report.model_dump(mode="json") == sqlite_report.model_dump(mode="json")


def test_runtime_evidence_projects_bounded_lineage_attempts_and_safe_totals() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "root")
        await _create_session(store, "child", parent_session_id="root")
        await store.append_events(
            "root",
            [
                Event(
                    id="root-model-started",
                    type=EventType.MODEL_STARTED,
                    session_id="root",
                    payload={
                        "provider": "provider",
                        "model": "model",
                        "step": 1,
                        "attempt": 1,
                        "model_step_id": "step-1",
                        "model_attempt_id": "attempt-1",
                    },
                ),
                Event(
                    id="root-model-completed",
                    type=EventType.MODEL_COMPLETED,
                    session_id="root",
                    payload={
                        "model_step_id": "step-1",
                        "model_attempt_id": "attempt-1",
                        "usage_metrics": {
                            "provider_name": "provider",
                            "requested_model": "model",
                            "model": "model",
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "reasoning_output_tokens": 0,
                            "cache": {
                                "read_tokens": 2,
                                "write_tokens": 0,
                                "cached_input_tokens": 2,
                                "uncached_input_tokens": 8,
                            },
                        },
                        "text": "secret model output",
                    },
                ),
                Event(
                    id="root-completed",
                    type=EventType.SESSION_COMPLETED,
                    session_id="root",
                    payload={"output": "secret terminal output"},
                ),
            ],
        )
        await store.append_event(
            "child",
            Event(
                id="child-completed",
                type=EventType.SESSION_COMPLETED,
                session_id="child",
            ),
        )
        await store.update_status("root", SessionStatus.COMPLETED)
        await store.update_status("child", SessionStatus.COMPLETED)

        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=20),
        )

    report = asyncio.run(scenario())

    assert report.schema_version == 4
    assert report.scope.descendant_session_ids == ("root", "child")
    assert [session.session_id for session in report.sessions] == ["root", "child"]
    assert report.sessions[1].parent_session_id == "root"
    assert report.sessions[0].last_event_cursor.sequence == 5
    assert report.sessions[0].attempts[0].operation is RuntimeEvidenceOperation.AGENT_STEP
    assert report.sessions[0].attempts[0].usage is not None
    assert report.sessions[0].attempts[0].usage.input_tokens == 10
    assert report.sessions[0].totals.usage.total_tokens == 15
    assert report.lineage_totals.usage.cache.read_tokens == 2
    assert report.whole_workflow_totals is None

    serialized = report.model_dump_json()
    assert "secret prompt" not in serialized
    assert "secret metadata" not in serialized
    assert "secret event prompt" not in serialized
    assert "secret model output" not in serialized
    assert "secret terminal output" not in serialized


def test_runtime_evidence_projects_complete_safe_accounting_scope() -> None:
    def completed_payload(
        attempt_id: str,
        step_id: str,
        tokens: int,
        *,
        operation: str | None = None,
        purpose: str | None = None,
        cache_read: int = 0,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model_step_id": step_id,
            "model_attempt_id": attempt_id,
            "usage_metrics": {
                "provider_name": "provider",
                "requested_model": "model",
                "model": "model",
                "input_tokens": tokens,
                "output_tokens": 0,
                "total_tokens": tokens,
                "reasoning_output_tokens": 0,
                "cache": {
                    "read_tokens": cache_read,
                    "write_tokens": 0,
                    "cached_input_tokens": cache_read,
                    "uncached_input_tokens": tokens - cache_read,
                },
            },
            "text": "secret completion body",
        }
        if operation is not None:
            payload["operation"] = operation
        if purpose is not None:
            payload["purpose"] = purpose
        return payload

    async def scenario():
        store = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        await _create_session(
            store,
            "root",
            metadata={
                "cayu:taint_labels": ["untrusted_web"],
                "secret": "secret session metadata",
            },
        )
        await _create_session(store, "child", parent_session_id="root")
        await _create_session(store, "causal-peer")
        await tasks.create_task(
            TaskCreate(
                task_id="task-root",
                type="secret task type",
                title="secret task title",
                description="secret task description",
                session_id="root",
                input={"secret": "secret task input"},
            )
        )
        events = [
            Event(
                id="attempt-1-started",
                type=EventType.MODEL_STARTED,
                session_id="root",
                payload={
                    "model_step_id": "step-agent",
                    "model_attempt_id": "attempt-1",
                    "attempt": 1,
                },
            ),
            Event(
                id="attempt-1-retry",
                type=EventType.MODEL_RETRY,
                session_id="root",
                payload={
                    "model_step_id": "step-agent",
                    "model_attempt_id": "attempt-1",
                    "attempt": 1,
                    "next_attempt": 2,
                    "error": "secret provider error",
                },
            ),
            Event(
                id="attempt-1-discarded",
                type=EventType.MODEL_ATTEMPT_DISCARDED,
                session_id="root",
                payload={
                    "model_step_id": "step-agent",
                    "model_attempt_id": "attempt-1",
                    "attempt": 1,
                    "error": "secret discarded output",
                },
            ),
            Event(
                id="attempt-2-started",
                type=EventType.MODEL_STARTED,
                session_id="root",
                payload={
                    "model_step_id": "step-agent",
                    "model_attempt_id": "attempt-2",
                    "attempt": 2,
                },
            ),
            Event(
                id="attempt-2-completed",
                type=EventType.MODEL_COMPLETED,
                session_id="root",
                payload=completed_payload("attempt-2", "step-agent", 10),
            ),
            Event(
                id="structured-retry",
                type=EventType.STRUCTURED_OUTPUT_RETRY,
                session_id="root",
                payload={"error": "secret invalid structure"},
            ),
            Event(
                id="structured-started",
                type=EventType.MODEL_STARTED,
                session_id="root",
                payload={
                    "model_step_id": "step-structured",
                    "model_attempt_id": "attempt-structured-1",
                    "attempt": 1,
                },
            ),
            Event(
                id="structured-provider-retry",
                type=EventType.MODEL_RETRY,
                session_id="root",
                payload={
                    "model_step_id": "step-structured",
                    "model_attempt_id": "attempt-structured-1",
                    "attempt": 1,
                    "next_attempt": 2,
                },
            ),
            Event(
                id="structured-discarded",
                type=EventType.MODEL_ATTEMPT_DISCARDED,
                session_id="root",
                payload={
                    "model_step_id": "step-structured",
                    "model_attempt_id": "attempt-structured-1",
                    "attempt": 1,
                },
            ),
            Event(
                id="structured-retry-started",
                type=EventType.MODEL_STARTED,
                session_id="root",
                payload={
                    "model_step_id": "step-structured",
                    "model_attempt_id": "attempt-structured-2",
                    "attempt": 2,
                },
            ),
            Event(
                id="structured-completed",
                type=EventType.MODEL_COMPLETED,
                session_id="root",
                payload=completed_payload(
                    "attempt-structured-2",
                    "step-structured",
                    5,
                    operation="structured_output_repair",
                ),
            ),
            Event(
                id="compaction-started",
                type=EventType.CONTEXT_COMPACTION_STARTED,
                session_id="root",
                payload={"operation_id": "compact-1", "summary": "secret summary"},
            ),
            Event(
                id="compaction-model-started",
                type=EventType.MODEL_STARTED,
                session_id="root",
                payload={
                    "model_step_id": "step-compaction",
                    "model_attempt_id": "attempt-compaction",
                    "attempt": 1,
                    "purpose": "context_compaction",
                },
            ),
            Event(
                id="compaction-model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id="root",
                payload=completed_payload(
                    "attempt-compaction",
                    "step-compaction",
                    4,
                    purpose="context_compaction",
                    cache_read=2,
                ),
            ),
            Event(
                id="compaction-completed",
                type=EventType.CONTEXT_COMPACTION_COMPLETED,
                session_id="root",
                payload={"summary": "secret compacted transcript"},
            ),
            Event(
                id="checkpoint-1",
                type=EventType.SESSION_CHECKPOINTED,
                session_id="root",
                payload={
                    "checkpoint": "context_compaction",
                    "compacted_transcript_cursor": 9,
                    "summary": "secret checkpoint contents",
                },
            ),
        ]
        for index, operation in enumerate(("evaluation", "repair", "comparison_control"), start=1):
            events.extend(
                [
                    Event(
                        id=f"{operation}-started",
                        type=EventType.MODEL_STARTED,
                        session_id="root",
                        payload={
                            "model_step_id": f"step-{operation}",
                            "model_attempt_id": f"attempt-{operation}",
                            "attempt": 1,
                            "operation": operation,
                        },
                    ),
                    Event(
                        id=f"{operation}-completed",
                        type=EventType.MODEL_COMPLETED,
                        session_id="root",
                        payload=completed_payload(
                            f"attempt-{operation}",
                            f"step-{operation}",
                            4 - index,
                            operation=operation,
                        ),
                    ),
                ]
            )
        events.extend(
            [
                Event(
                    id="tool-started",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="root",
                    tool_name="charge_card",
                    payload={
                        "tool_round_id": "round-1",
                        "tool_call_id": "call-1",
                        "idempotency_key": "idempotency-1",
                        "arguments": {"card": "secret-card"},
                    },
                ),
                Event(
                    id="approval-requested",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id="root",
                    payload={
                        "approval_id": "approval-1",
                        "tool_round_id": "round-1",
                        "tool_call_id": "call-1",
                        "reason": "secret approval reason",
                    },
                ),
                Event(
                    id="approval-approved",
                    type=EventType.TOOL_CALL_APPROVED,
                    session_id="root",
                    payload={
                        "approval_id": "approval-1",
                        "tool_round_id": "round-1",
                        "tool_call_id": "call-1",
                        "reason": "secret operator note",
                    },
                ),
                Event(
                    id="tool-completed",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="root",
                    tool_name="charge_card",
                    payload={
                        "tool_round_id": "round-1",
                        "tool_call_id": "call-1",
                        "idempotency_key": "idempotency-1",
                        "manual_recovery": True,
                        "result": {
                            "content": "secret tool result",
                            "structured": {
                                "receipt_id": "receipt-1",
                                "reconciliation_state": "reconciled",
                                "secret": "secret receipt body",
                            },
                        },
                    },
                ),
                Event(
                    id="tool-blocked",
                    type=EventType.TOOL_CALL_BLOCKED,
                    session_id="root",
                    tool_name="publish",
                    payload={
                        "tool_round_id": "round-1",
                        "tool_call_id": "call-2",
                        "idempotency_key": "idempotency-2",
                        "denied_by": "taint_policy",
                        "decision": "deny",
                        "metadata": {
                            "matched_taint_labels": ["untrusted_web"],
                            "secret": "secret policy metadata",
                        },
                        "reason": "secret policy reason",
                    },
                ),
                Event(
                    id="interrupted",
                    type=EventType.SESSION_INTERRUPTED,
                    session_id="root",
                    payload={
                        "manual_recovery_required": True,
                        "error": "secret interruption body",
                    },
                ),
                Event(
                    id="resumed",
                    type=EventType.SESSION_RESUMED,
                    session_id="root",
                    payload={"note": "secret recovery note"},
                ),
                Event(
                    id="future-custom",
                    type="custom.future.evidence",
                    session_id="root",
                    payload={
                        "manual_recovery": True,
                        "manual_recovery_required": True,
                        "secret": "secret future event",
                    },
                ),
            ]
        )
        await store.append_events("root", events)
        pricing = PriceBook(
            price_book_version="test-v1",
            prices=(
                ModelPrice.fixed(
                    provider_name="provider",
                    model="model",
                    input_per_million=Decimal("1"),
                    output_per_million=Decimal("2"),
                ),
            ),
        )
        return await runtime_evidence(
            CayuApp(session_store=store, task_store=tasks, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="root",
                max_sessions=10,
                max_events=100,
                include_causal_budget=True,
                max_causal_budget_sessions=10,
                pricing=pricing,
            ),
        )

    report = asyncio.run(scenario())
    root = next(session for session in report.sessions if session.session_id == "root")

    assert report.scope.descendant_session_ids == ("root", "child")
    assert report.scope.causal_budget_session_ids == ("root", "child", "causal-peer")
    assert report.tasks[0].task_id == "task-root"
    assert root.task_ids == ("task-root",)
    assert root.checkpoints[0].checkpoint_id == "checkpoint-1"
    assert root.checkpoints[0].compacted_transcript_cursor == 9
    assert root.compaction_count == 1
    assert root.compactions[0].compaction_id == "compact-1"
    assert root.compactions[0].status == "completed"
    assert [ref.event_id for ref in root.compactions[0].source_refs] == [
        "compaction-started",
        "compaction-completed",
    ]
    assert [attempt.operation for attempt in root.attempts] == [
        RuntimeEvidenceOperation.AGENT_STEP,
        RuntimeEvidenceOperation.AGENT_STEP,
        RuntimeEvidenceOperation.STRUCTURED_OUTPUT_REPAIR,
        RuntimeEvidenceOperation.STRUCTURED_OUTPUT_REPAIR,
        RuntimeEvidenceOperation.COMPACTION,
        RuntimeEvidenceOperation.EVALUATION,
        RuntimeEvidenceOperation.REPAIR,
        RuntimeEvidenceOperation.COMPARISON_CONTROL,
    ]
    assert root.totals.attempt_count == 8
    assert root.totals.provider_retry_attempt_count == 2
    assert root.totals.usage.total_tokens == 25
    assert root.totals.first_attempt_usage.total_tokens == 10
    assert root.totals.provider_retry_usage.total_tokens == 15
    assert root.totals.compaction_usage.cache.read_tokens == 2
    assert root.totals.structured_output_repair_usage.total_tokens == 5
    assert root.totals.evaluation_usage.total_tokens == 3
    assert root.totals.repair_usage.total_tokens == 2
    assert root.totals.comparison_control_usage.total_tokens == 1
    assert root.totals.priced_attempt_count == 6
    assert root.attempts[0].cost.status is RuntimeEvidenceCostStatus.MISSING_USAGE
    assert root.attempts[1].cost.status is RuntimeEvidenceCostStatus.PRICED
    assert len(root.tool_calls) == 2
    assert root.tool_calls[0].idempotency_key == "idempotency-1"
    assert root.approvals[0].decision == "approved"
    assert root.effective_taint_labels == ("untrusted_web",)
    assert root.policy_decisions[0].decision == "deny"
    assert root.receipts[0].receipt_id == "receipt-1"
    assert root.receipts[0].reconciliation_state == "reconciled"
    assert root.recovery.interruption_count == 1
    assert root.recovery.resume_count == 1
    assert root.recovery.manual_recovery_required_count == 1
    assert root.recovery.manual_reconciliation_count == 1
    assert {ref.event_id for ref in root.recovery.source_refs} == {
        "tool-completed",
        "interrupted",
        "resumed",
    }
    assert report.causal_budget_totals is not None
    assert report.causal_budget_totals.session_count == 3
    assert report.whole_workflow_totals == report.causal_budget_totals
    assert report.pricing_catalog_version == "test-v1"
    assert report.pricing_catalog_generated_at == "unspecified"
    assert RuntimeEvidenceWarningCode.MISSING_USAGE in {warning.code for warning in report.warnings}
    assert RuntimeEvidenceWarningCode.UNKNOWN_EVENT_TYPE in {
        warning.code for warning in report.warnings
    }
    serialized = report.model_dump_json()
    assert "secret" not in serialized


def test_runtime_evidence_marks_legacy_attempt_operation_unknown() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "legacy")
        await store.append_event(
            "legacy",
            Event(
                id="legacy-model-started",
                type=EventType.MODEL_STARTED,
                session_id="legacy",
                payload={"provider": "provider", "model": "model"},
            ),
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(root_session_id="legacy", max_sessions=10, max_events=20),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].attempts[0].operation is RuntimeEvidenceOperation.UNKNOWN
    assert {
        RuntimeEvidenceWarningCode.LEGACY_ATTEMPT_IDENTITY,
        RuntimeEvidenceWarningCode.UNKNOWN_OPERATION,
    } <= {warning.code for warning in report.warnings}


def test_runtime_evidence_later_explicit_operation_overrides_fallback() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "explicit-operation")
        await store.append_events(
            "explicit-operation",
            [
                Event(
                    id="evaluation-started",
                    type=EventType.MODEL_STARTED,
                    session_id="explicit-operation",
                    payload={
                        "model_step_id": "evaluation-step",
                        "model_attempt_id": "evaluation-attempt",
                        "attempt": 1,
                    },
                ),
                Event(
                    id="evaluation-completed",
                    type=EventType.MODEL_COMPLETED,
                    session_id="explicit-operation",
                    payload={
                        "model_step_id": "evaluation-step",
                        "model_attempt_id": "evaluation-attempt",
                        "attempt": 1,
                        "operation": "evaluation",
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="explicit-operation", max_sessions=10, max_events=20
            ),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].attempts[0].operation is RuntimeEvidenceOperation.EVALUATION


def test_runtime_evidence_request_bounds_optional_pricing_catalog() -> None:
    prices = tuple(
        ModelPrice.fixed(
            provider_name="provider",
            model=f"model-{index}",
            input_per_million=Decimal("1"),
            output_per_million=Decimal("1"),
        )
        for index in range(513)
    )
    with pytest.raises(ValueError, match="512 catalog rows"):
        RuntimeEvidenceRequest(
            root_session_id="root",
            max_sessions=10,
            max_events=20,
            pricing=PriceBook(prices=prices),
        )

    one_price = prices[:1]
    with pytest.raises(ValueError, match="1048576 canonical JSON bytes"):
        RuntimeEvidenceRequest(
            root_session_id="root",
            max_sessions=10,
            max_events=20,
            pricing=PriceBook(
                price_book_version="x" * 1_048_576,
                prices=one_price,
            ),
        )
    with pytest.raises(ValueError, match="1024 characters"):
        RuntimeEvidenceRequest(
            root_session_id="root",
            max_sessions=10,
            max_events=20,
            pricing=PriceBook(
                price_book_version="x" * 1_025,
                prices=one_price,
            ),
        )
    with pytest.raises(ValueError, match="1024 characters"):
        RuntimeEvidenceRequest(
            root_session_id="root",
            max_sessions=10,
            max_events=20,
            pricing=PriceBook(
                prices=(
                    ModelPrice.fixed(
                        provider_name="contextual-provider",
                        model="contextual-model",
                        input_per_million=Decimal("1"),
                        output_per_million=Decimal("1"),
                        match="exact",
                        pricing_context=PricingContextSelector(
                            dimensions={"x" * 1_025: ("value",)},
                        ),
                    ),
                ),
            ),
        )


def test_runtime_evidence_preserves_missing_compaction_and_unpriced_model() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "pricing-evidence")
        await store.append_events(
            "pricing-evidence",
            [
                Event(
                    id="compaction-started",
                    type=EventType.MODEL_STARTED,
                    session_id="pricing-evidence",
                    payload={
                        "model_step_id": "compaction-step",
                        "model_attempt_id": "compaction-attempt",
                        "attempt": 1,
                        "purpose": "context_compaction",
                    },
                ),
                Event(
                    id="compaction-completed",
                    type=EventType.MODEL_COMPLETED,
                    session_id="pricing-evidence",
                    payload={
                        "model_step_id": "compaction-step",
                        "model_attempt_id": "compaction-attempt",
                        "attempt": 1,
                        "purpose": "context_compaction",
                    },
                ),
                Event(
                    id="unpriced-started",
                    type=EventType.MODEL_STARTED,
                    session_id="pricing-evidence",
                    payload={
                        "model_step_id": "unpriced-step",
                        "model_attempt_id": "unpriced-attempt",
                        "attempt": 1,
                    },
                ),
                Event(
                    id="unpriced-completed",
                    type=EventType.MODEL_COMPLETED,
                    session_id="pricing-evidence",
                    payload={
                        "model_step_id": "unpriced-step",
                        "model_attempt_id": "unpriced-attempt",
                        "attempt": 1,
                        "usage_metrics": {
                            "provider_name": "provider",
                            "requested_model": "unknown-model",
                            "model": "unknown-model",
                            "input_tokens": 3,
                            "output_tokens": 2,
                            "total_tokens": 5,
                            "reasoning_output_tokens": 0,
                        },
                    },
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="pricing-evidence",
                max_sessions=10,
                max_events=20,
                pricing=PriceBook(
                    prices=(
                        ModelPrice.fixed(
                            provider_name="provider",
                            model="priced-model",
                            input_per_million=Decimal("1"),
                            output_per_million=Decimal("1"),
                        ),
                    ),
                ),
            ),
        )

    report = asyncio.run(scenario())
    attempts = report.sessions[0].attempts

    assert attempts[0].operation is RuntimeEvidenceOperation.COMPACTION
    assert attempts[0].cost.status is RuntimeEvidenceCostStatus.MISSING_USAGE
    assert attempts[1].cost.status is RuntimeEvidenceCostStatus.UNPRICED
    assert report.sessions[0].totals.missing_usage_attempt_count == 1
    assert report.sessions[0].totals.unpriced_attempt_count == 1
    assert {
        RuntimeEvidenceWarningCode.MISSING_USAGE,
        RuntimeEvidenceWarningCode.UNPRICED_USAGE,
    } <= {warning.code for warning in report.warnings}


def test_runtime_evidence_retains_abandoned_compaction_retry_starts() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "compaction-retry")
        await store.append_events(
            "compaction-retry",
            [
                Event(
                    id="compaction-start-1",
                    type=EventType.CONTEXT_COMPACTION_STARTED,
                    session_id="compaction-retry",
                    payload={"operation_id": "compact-1"},
                ),
                Event(
                    id="compaction-start-2",
                    type=EventType.CONTEXT_COMPACTION_STARTED,
                    session_id="compaction-retry",
                    payload={"operation_id": "compact-1"},
                ),
                Event(
                    id="compaction-done",
                    type=EventType.CONTEXT_COMPACTION_COMPLETED,
                    session_id="compaction-retry",
                    payload={"operation_id": "compact-1"},
                ),
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="compaction-retry", max_sessions=10, max_events=20
            ),
        )

    report = asyncio.run(scenario())
    compaction = report.sessions[0].compactions[0]

    assert compaction.status == "completed"
    assert [ref.event_id for ref in compaction.source_refs] == [
        "compaction-start-1",
        "compaction-start-2",
        "compaction-done",
    ]
    assert RuntimeEvidenceWarningCode.MALFORMED_COMPACTION not in {
        warning.code for warning in report.warnings
    }


def test_runtime_evidence_handles_many_distinct_model_steps_linearly() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "many-steps")
        await store.append_events(
            "many-steps",
            [
                Event(
                    id=f"attempt-{index}",
                    type=EventType.MODEL_STARTED,
                    session_id="many-steps",
                    payload={
                        "model_step_id": f"step-{index}",
                        "model_attempt_id": f"attempt-{index}",
                        "attempt": 1,
                    },
                )
                for index in range(2_000)
            ],
        )
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(
                root_session_id="many-steps",
                max_sessions=10,
                max_events=2_100,
            ),
        )

    report = asyncio.run(scenario())

    assert report.sessions[0].totals.model_step_count == 2_000
    assert report.sessions[0].totals.attempt_count == 2_000


@pytest.mark.parametrize(
    ("case", "expected_code", "expected_limit", "expected_observed"),
    [
        ("session", RuntimeEvidenceErrorCode.SESSION_LIMIT_EXCEEDED, 1, 2),
        ("event", RuntimeEvidenceErrorCode.EVENT_LIMIT_EXCEEDED, 1, 2),
        (
            "causal",
            RuntimeEvidenceErrorCode.CAUSAL_BUDGET_LIMIT_EXCEEDED,
            1,
            2,
        ),
    ],
)
def test_runtime_evidence_fails_closed_on_each_scope_bound(
    case: str,
    expected_code: RuntimeEvidenceErrorCode,
    expected_limit: int,
    expected_observed: int,
) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        await _create_session(store, "root")
        if case == "session":
            await _create_session(store, "child", parent_session_id="root")
            request = RuntimeEvidenceRequest(root_session_id="root", max_sessions=1, max_events=20)
        elif case == "event":
            await store.append_event(
                "root",
                Event(
                    id="second-event",
                    type=EventType.SESSION_COMPLETED,
                    session_id="root",
                ),
            )
            request = RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=1)
        else:
            await _create_session(store, "causal-peer")
            request = RuntimeEvidenceRequest(
                root_session_id="root",
                max_sessions=10,
                max_events=20,
                include_causal_budget=True,
                max_causal_budget_sessions=1,
            )
        with pytest.raises(RuntimeEvidenceError) as caught:
            await runtime_evidence(CayuApp(session_store=store, enable_logging=False), request)
        assert caught.value.code is expected_code
        assert caught.value.limit == expected_limit
        assert caught.value.observed == expected_observed

    asyncio.run(scenario())


def test_runtime_evidence_reports_typed_missing_root_and_cycle_errors() -> None:
    class CyclicLineageStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_session_lineage(self, query):
            if query.parent_session_id != "child":
                return await super().query_session_lineage(query)
            root = await self.load("root")
            assert root is not None
            return SessionLineageResult(
                parent_session_id="child",
                children=(
                    SessionLineageNode(
                        id="root",
                        parent_session_id="child",
                        created_at=root.created_at,
                    ),
                ),
            )

    class ContradictoryLineageStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_session_lineage(self, query):
            if query.parent_session_id != "root":
                return await super().query_session_lineage(query)
            child = await self.load("detached-child")
            assert child is not None
            return SessionLineageResult(
                parent_session_id="root",
                children=(
                    SessionLineageNode(
                        id="detached-child",
                        parent_session_id="root",
                        created_at=child.created_at,
                    ),
                ),
            )

    async def scenario() -> None:
        missing_store = InMemorySessionStore()
        with pytest.raises(RuntimeEvidenceError) as missing:
            await runtime_evidence(
                CayuApp(session_store=missing_store, enable_logging=False),
                RuntimeEvidenceRequest(root_session_id="absent", max_sessions=10, max_events=20),
            )
        assert missing.value.code is RuntimeEvidenceErrorCode.ROOT_NOT_FOUND

        cyclic = CyclicLineageStore()
        await _create_session(cyclic, "root")
        await _create_session(cyclic, "child", parent_session_id="root")
        with pytest.raises(RuntimeEvidenceError) as cycle:
            await runtime_evidence(
                CayuApp(session_store=cyclic, enable_logging=False),
                RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=20),
            )
        assert cycle.value.code is RuntimeEvidenceErrorCode.CYCLE_DETECTED

        contradictory = ContradictoryLineageStore()
        await _create_session(contradictory, "root")
        await _create_session(contradictory, "detached-child")
        with pytest.raises(RuntimeEvidenceError) as corrupt:
            await runtime_evidence(
                CayuApp(session_store=contradictory, enable_logging=False),
                RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=20),
            )
        assert corrupt.value.code is RuntimeEvidenceErrorCode.PARENT_CONTRADICTION

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("store_type", "expected_code"),
    [
        ("unsupported_lineage", RuntimeEvidenceErrorCode.STORE_UNSUPPORTED),
        ("unsupported_event_read", RuntimeEvidenceErrorCode.STORE_UNSUPPORTED),
        ("event_source_bytes", RuntimeEvidenceErrorCode.EVENT_SOURCE_BYTES_EXCEEDED),
        ("event_read_failure", RuntimeEvidenceErrorCode.EVIDENCE_READ_FAILED),
    ],
)
def test_runtime_evidence_reports_each_typed_store_read_failure(
    store_type: str,
    expected_code: RuntimeEvidenceErrorCode,
) -> None:
    class UnsupportedLineageStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1
        supports_session_lineage = False

    class UnsupportedEventReadStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_events_bounded(self, query, *, max_bytes):
            raise NotImplementedError

    class EventSourceBytesStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_events_bounded(self, query, *, max_bytes):
            raise EventQueryResultTooLarge(max_bytes)

    class EventReadFailureStore(InMemorySessionStore):
        invocation_lifecycle_command_version = 1

        async def query_events_bounded(self, query, *, max_bytes):
            raise RuntimeError("read failed")

    stores = {
        "unsupported_lineage": UnsupportedLineageStore,
        "unsupported_event_read": UnsupportedEventReadStore,
        "event_source_bytes": EventSourceBytesStore,
        "event_read_failure": EventReadFailureStore,
    }

    async def scenario() -> None:
        store = stores[store_type]()
        if store_type != "unsupported_lineage":
            await _create_session(store, "root")
        with pytest.raises(RuntimeEvidenceError) as caught:
            await runtime_evidence(
                CayuApp(session_store=store, enable_logging=False),
                RuntimeEvidenceRequest(root_session_id="root", max_sessions=10, max_events=20),
            )
        assert caught.value.code is expected_code

    asyncio.run(scenario())


async def _minimal_golden_report(
    store: SessionStore,
    *,
    session_id: str = "golden-root",
):
    event_prefix = "golden" if session_id == "golden-root" else session_id
    await _create_session(store, session_id)
    await store.append_events(
        session_id,
        [
            Event(
                id=f"{event_prefix}-model-started",
                type=EventType.MODEL_STARTED,
                session_id=session_id,
                payload={
                    "model_step_id": "golden-step",
                    "model_attempt_id": "golden-attempt",
                    "attempt": 1,
                },
            ),
            Event(
                id=f"{event_prefix}-model-completed",
                type=EventType.MODEL_COMPLETED,
                session_id=session_id,
                payload={
                    "model_step_id": "golden-step",
                    "model_attempt_id": "golden-attempt",
                    "usage_metrics": {
                        "provider_name": "provider",
                        "requested_model": "model",
                        "model": "model",
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                        "reasoning_output_tokens": 0,
                        "cache": {
                            "read_tokens": 0,
                            "write_tokens": 0,
                            "cached_input_tokens": 0,
                            "uncached_input_tokens": 2,
                        },
                    },
                },
            ),
        ],
    )
    return await runtime_evidence(
        CayuApp(session_store=store, enable_logging=False),
        RuntimeEvidenceRequest(root_session_id=session_id, max_sessions=10, max_events=20),
    )


def test_runtime_evidence_sqlite_restart_and_v4_golden_are_exact(tmp_path: Path) -> None:
    async def scenario():
        database = tmp_path / "runtime-evidence.sqlite"
        first_store = SQLiteSessionStore(database)
        first = await _minimal_golden_report(first_store)
        await first_store.close()
        reopened_store = SQLiteSessionStore(database)
        second = await runtime_evidence(
            CayuApp(session_store=reopened_store, enable_logging=False),
            RuntimeEvidenceRequest(root_session_id="golden-root", max_sessions=10, max_events=20),
        )
        await reopened_store.close()
        return first, second

    first, second = asyncio.run(scenario())
    assert first == second
    golden_path = Path(__file__).parents[1] / "fixtures" / "runtime_evidence_v4.json"
    assert first.model_dump(mode="json") == json.loads(golden_path.read_text())


def test_runtime_evidence_postgres_matches_sqlite_projection(
    postgres_dsn: str, tmp_path: Path
) -> None:
    async def scenario():
        session_id = f"runtime-evidence-{uuid4().hex}"
        sqlite_store = SQLiteSessionStore(tmp_path / "runtime-evidence.sqlite")
        postgres_store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        try:
            await postgres_store.ensure_schema()
            await _reset_postgres_event_sequence(postgres_dsn)
            sqlite_report = await _minimal_golden_report(sqlite_store, session_id=session_id)
            postgres_report = await _minimal_golden_report(postgres_store, session_id=session_id)
        finally:
            await sqlite_store.close()
            await postgres_store.close()
            await _reset_postgres_event_sequence(postgres_dsn)
        return sqlite_report, postgres_report

    sqlite_report, postgres_report = asyncio.run(scenario())

    assert postgres_report == sqlite_report


def test_runtime_evidence_branch_ordering_v1_golden_is_exact() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await _create_session(store, "ordering-root")
        await _create_session(store, "branch-first", parent_session_id="ordering-root")
        await _create_session(store, "branch-second", parent_session_id="ordering-root")
        await _create_session(store, "grandchild", parent_session_id="branch-first")

        async def append_usage(session_id: str, total_tokens: int) -> None:
            await store.append_events(
                session_id,
                [
                    Event(
                        id=f"{session_id}-model-started",
                        type=EventType.MODEL_STARTED,
                        session_id=session_id,
                        payload={
                            "model_step_id": f"{session_id}-step",
                            "model_attempt_id": f"{session_id}-attempt",
                            "attempt": 1,
                        },
                    ),
                    Event(
                        id=f"{session_id}-model-completed",
                        type=EventType.MODEL_COMPLETED,
                        session_id=session_id,
                        payload={
                            "model_step_id": f"{session_id}-step",
                            "model_attempt_id": f"{session_id}-attempt",
                            "attempt": 1,
                            "usage_metrics": {
                                "provider_name": "provider",
                                "requested_model": "model",
                                "model": "model",
                                "input_tokens": total_tokens,
                                "output_tokens": 0,
                                "total_tokens": total_tokens,
                                "reasoning_output_tokens": 0,
                            },
                        },
                    ),
                ],
            )

        await append_usage("branch-first", 2)
        await append_usage("branch-second", 5)
        await append_usage("grandchild", 3)
        return await runtime_evidence(
            CayuApp(session_store=store, enable_logging=False),
            RuntimeEvidenceRequest(root_session_id="ordering-root", max_sessions=10, max_events=20),
        )

    report = asyncio.run(scenario())
    ordering = {
        "sessions": [
            {
                "session_id": session.session_id,
                "total_tokens": int(session.totals.usage.total_tokens),
            }
            for session in report.sessions
        ],
        "branches": [
            {
                "branch_root_session_id": branch.branch_root_session_id,
                "session_ids": list(branch.session_ids),
                "session_count": int(branch.totals.session_count),
                "total_tokens": int(branch.totals.usage.total_tokens),
            }
            for branch in report.branch_totals
        ],
    }
    golden_path = Path(__file__).parents[1] / "fixtures" / "runtime_evidence_ordering_v1.json"
    assert ordering == json.loads(golden_path.read_text())
