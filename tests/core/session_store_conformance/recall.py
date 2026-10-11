"""Shared session-store recall contract cases."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError
from tests.core.session_store_conformance.support import (
    _approval_recovery_environment_spec,
    _ApprovalRecoveryProvider,
    _ApprovalRecoveryTool,
    _automatic_recall_recovery_policy,
    _AutomaticRecallRecoveryKnowledgeStore,
    _ChangingApprovalPolicy,
    _close_store,
    _identity,
    _large_recall_receipt,
    _memory_evidence_digest,
    _new_postgres_store,
    _open_store,
    _pending_approval_for_public_event,
    _planned_context_exposure,
    _recall_receipt,
    _reopen_store,
    _reset_postgres_data,
)

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.approvals.tools import (
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from cayu.context.footprints import RequestFootprintConfig
from cayu.environments.base import Environment
from cayu.events import (
    EventType,
)
from cayu.memory.attribution import MemoryAttributionStatus
from cayu.memory.evidence import (
    ContextExposure,
    ContextExposureState,
    ContextExposureTransitionConflict,
    ContextExposureTransitionRequest,
    RecallEvidenceConflict,
    RecallEvidenceQuery,
    RecallReceipt,
)
from cayu.messages import (
    Message,
    MessageRole,
    TextPart,
)
from cayu.runtime.evidence import RuntimeEvidenceRequest, runtime_evidence
from cayu.sessions.records import SessionStatus
from cayu.sessions.requests import (
    RunRequest,
)
from cayu.storage.memory import KnowledgeEntry
from cayu.tools.policy import (
    ToolPolicyDecision,
)

pytestmark = pytest.mark.postgres


def test_session_store_conformance_persists_fenced_recall_exposure_evidence(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            assert store.supports_recall_evidence is True
            with pytest.raises(ValueError, match="character bound"):
                await store.load_recall_receipt("memory-evidence-session", "r" * 257)
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id="memory-evidence-session",
                    messages=[],
                ),
                identity=_identity(),
            )
            first_receipt = _recall_receipt("a")
            second_receipt = _recall_receipt("b")
            assert await store.create_recall_receipt(first_receipt) == first_receipt
            assert await store.create_recall_receipt(first_receipt) == first_receipt
            await store.create_recall_receipt(second_receipt)

            conflicting_receipt = first_receipt.model_copy(
                update={
                    "frontier_fingerprint": first_receipt.frontier_fingerprint.model_copy(
                        update={"digest": _memory_evidence_digest("different-frontier")}
                    )
                }
            )
            with pytest.raises(RecallEvidenceConflict):
                await store.create_recall_receipt(conflicting_receipt)
            missing_parent_conflict = first_receipt.model_copy(
                update={"session_id": "missing-memory-evidence-session"}
            )
            with pytest.raises(RecallEvidenceConflict):
                await store.create_recall_receipt(missing_parent_conflict)

            page = await store.list_recall_receipts(
                RecallEvidenceQuery(
                    session_id=first_receipt.session_id,
                    interaction_id=first_receipt.interaction_id,
                    limit=1,
                )
            )
            assert [receipt.receipt_id for receipt in page.items] == ["receipt-a"]
            assert page.truncated is True
            assert page.next_cursor is not None
            second_page = await store.list_recall_receipts(
                RecallEvidenceQuery(
                    session_id=first_receipt.session_id,
                    interaction_id=first_receipt.interaction_id,
                    limit=1,
                    cursor=page.next_cursor,
                )
            )
            assert [receipt.receipt_id for receipt in second_page.items] == ["receipt-b"]
            assert second_page.truncated is False
            assert second_page.next_cursor is None
            combined_scope_page = await store.list_recall_receipts(
                RecallEvidenceQuery(
                    session_id=first_receipt.session_id,
                    interaction_id=first_receipt.interaction_id,
                    model_step_id=first_receipt.model_step_id,
                )
            )
            assert [receipt.receipt_id for receipt in combined_scope_page.items] == [
                "receipt-a",
                "receipt-b",
            ]
            with pytest.raises(ValueError, match="does not match"):
                await store.list_recall_receipts(
                    RecallEvidenceQuery(
                        session_id=first_receipt.session_id,
                        model_step_id=f"mstep_{'2' * 32}",
                        cursor=page.next_cursor,
                    )
                )

            large_receipts = (
                _large_recall_receipt("byte-a"),
                _large_recall_receipt("byte-b"),
            )
            for large_receipt in large_receipts:
                await store.create_recall_receipt(large_receipt)
            byte_page = await store.list_recall_receipts(
                RecallEvidenceQuery(
                    session_id=first_receipt.session_id,
                    interaction_id="memory-evidence-byte-page",
                    max_bytes=256_002,
                )
            )
            assert [receipt.receipt_id for receipt in byte_page.items] == ["receipt-byte-a"]
            assert byte_page.truncated is True
            assert byte_page.next_cursor is not None
            final_byte_page = await store.list_recall_receipts(
                RecallEvidenceQuery(
                    session_id=first_receipt.session_id,
                    interaction_id="memory-evidence-byte-page",
                    max_bytes=256_002,
                    cursor=byte_page.next_cursor,
                )
            )
            assert [receipt.receipt_id for receipt in final_byte_page.items] == ["receipt-byte-b"]
            assert final_byte_page.truncated is False
            assert final_byte_page.next_cursor is None

            exposure, item_exposures = _planned_context_exposure("a", first_receipt)
            assert await store.create_context_exposure(exposure, item_exposures) == exposure
            assert await store.create_context_exposure(exposure, item_exposures) == exposure
            missing_parent_exposure_conflict = exposure.model_copy(
                update={"session_id": "missing-memory-evidence-session"}
            )
            with pytest.raises(RecallEvidenceConflict):
                await store.create_context_exposure(
                    missing_parent_exposure_conflict,
                    item_exposures,
                )
            assert (
                await store.load_recall_item_exposures(
                    exposure.session_id,
                    exposure.exposure_id,
                )
                == item_exposures
            )

            later_step_exposure, later_step_items = _planned_context_exposure(
                "later-step",
                first_receipt,
            )
            later_step_exposure = later_step_exposure.model_copy(
                update={"model_step_id": f"mstep_{'3' * 32}"}
            )
            await store.create_context_exposure(later_step_exposure, later_step_items)
            assert (
                await store.load_context_exposure(
                    later_step_exposure.session_id,
                    later_step_exposure.exposure_id,
                )
                == later_step_exposure
            )

            mixed_key_exposure, mixed_key_items = _planned_context_exposure(
                "mixed-key",
                first_receipt,
            )
            mixed_key_exposure = mixed_key_exposure.model_copy(
                update={
                    field_name: getattr(mixed_key_exposure, field_name).model_copy(
                        update={"key_id": "another-memory-evidence-key"}
                    )
                    for field_name in (
                        "composition_fingerprint",
                        "execution_profile_fingerprint",
                        "context_policy_fingerprint",
                        "tool_exposure_fingerprint",
                        "request_contract_fingerprint",
                    )
                }
            )
            with pytest.raises(ValueError, match="one evidence-key identity"):
                await store.create_context_exposure(
                    mixed_key_exposure,
                    mixed_key_items,
                )

            second_attempt_owner, second_attempt_items = _planned_context_exposure(
                "attempt-owner-b",
                first_receipt,
            )
            await store.create_context_exposure(
                second_attempt_owner,
                second_attempt_items,
            )
            dual_collision, dual_collision_items = _planned_context_exposure(
                "dual-collision",
                first_receipt,
            )
            dual_collision = dual_collision.model_copy(
                update={
                    "model_attempt_id": exposure.model_attempt_id,
                    "provider_attempt_id": second_attempt_owner.provider_attempt_id,
                }
            )
            with pytest.raises(RecallEvidenceConflict) as dual_attempt_conflict:
                await store.create_context_exposure(
                    dual_collision,
                    dual_collision_items,
                )
            assert dual_attempt_conflict.value.record_type == "Model-attempt exposure"
            assert dual_attempt_conflict.value.record_id == exposure.exposure_id

            collision_with_missing_receipt, _ = _planned_context_exposure(
                "collision-with-missing-receipt",
                first_receipt,
            )
            collision_with_missing_receipt = collision_with_missing_receipt.model_copy(
                update={
                    "model_attempt_id": exposure.model_attempt_id,
                    "receipt_ids": ("missing-recall-receipt",),
                }
            )
            with pytest.raises(RecallEvidenceConflict) as parent_order_conflict:
                await store.create_context_exposure(
                    collision_with_missing_receipt,
                    (),
                )
            assert parent_order_conflict.value.record_type == "Model-attempt exposure"
            assert parent_order_conflict.value.record_id == exposure.exposure_id

            provider_attempt_collision, provider_collision_items = _planned_context_exposure(
                "provider-collision", first_receipt
            )
            provider_attempt_collision = provider_attempt_collision.model_copy(
                update={"provider_attempt_id": exposure.provider_attempt_id}
            )
            with pytest.raises(RecallEvidenceConflict) as provider_collision:
                await store.create_context_exposure(
                    provider_attempt_collision,
                    provider_collision_items,
                )
            assert provider_collision.value.record_type == "Provider-attempt exposure"

            model_attempt_collision, model_collision_items = _planned_context_exposure(
                "model-collision", first_receipt
            )
            model_attempt_collision = model_attempt_collision.model_copy(
                update={"model_attempt_id": exposure.model_attempt_id}
            )
            with pytest.raises(RecallEvidenceConflict) as model_collision:
                await store.create_context_exposure(
                    model_attempt_collision,
                    model_collision_items,
                )
            assert model_collision.value.record_type == "Model-attempt exposure"

            duplicate_exposure, duplicate_items = _planned_context_exposure(
                "duplicate-item",
                first_receipt,
            )
            duplicate_item = duplicate_items[0].model_copy(update={"ordinal": 1})
            with pytest.raises(ValueError, match="only once"):
                await store.create_context_exposure(
                    duplicate_exposure,
                    (*duplicate_items, duplicate_item),
                )

            malformed_item = item_exposures[0].model_copy(
                update={"content_sha256": _memory_evidence_digest("wrong-content")}
            )
            invalid_exposure, _ = _planned_context_exposure("invalid", first_receipt)
            malformed_item = malformed_item.model_copy(
                update={"exposure_id": invalid_exposure.exposure_id}
            )
            with pytest.raises(ValueError, match="immutable receipt item"):
                await store.create_context_exposure(invalid_exposure, (malformed_item,))
            assert (
                await store.load_context_exposure(
                    invalid_exposure.session_id,
                    invalid_exposure.exposure_id,
                )
                is None
            )

            later_receipt = _recall_receipt(
                "later",
                created_at=first_receipt.created_at + timedelta(seconds=1),
            )
            await store.create_recall_receipt(later_receipt)
            early_exposure, early_items = _planned_context_exposure("early", later_receipt)
            earlier_at = first_receipt.created_at
            early_exposure = early_exposure.model_copy(
                update={
                    "created_at": earlier_at,
                    "updated_at": earlier_at,
                    "transitions": (
                        early_exposure.transitions[0].model_copy(
                            update={"occurred_at": earlier_at}
                        ),
                    ),
                }
            )
            with pytest.raises(ValueError, match="cannot predate"):
                await store.create_context_exposure(early_exposure, early_items)

            transition_time = exposure.created_at
            transition_requests = (
                ContextExposureTransitionRequest(
                    transition_id="transition-a-prepared",
                    expected_state="planned",
                    expected_revision=0,
                    state="prepared",
                    occurred_at=transition_time + timedelta(seconds=1),
                    evidence_kind="request_prepared",
                    evidence_ref="prepared-a",
                ),
                ContextExposureTransitionRequest(
                    transition_id="transition-a-dispatched",
                    expected_state="prepared",
                    expected_revision=1,
                    state="dispatch_started",
                    occurred_at=transition_time + timedelta(seconds=2),
                    evidence_kind="dispatch_intent_committed",
                    evidence_ref="dispatch-a",
                ),
                ContextExposureTransitionRequest(
                    transition_id="transition-a-acknowledged",
                    expected_state="dispatch_started",
                    expected_revision=2,
                    state="acknowledged",
                    occurred_at=transition_time + timedelta(seconds=3),
                    evidence_kind="provider_acknowledgement",
                    evidence_ref="ack-a",
                    provider_request_id="provider-request-a",
                ),
                ContextExposureTransitionRequest(
                    transition_id="transition-a-completed",
                    expected_state="acknowledged",
                    expected_revision=3,
                    state="completed",
                    occurred_at=transition_time + timedelta(seconds=4),
                    evidence_kind="provider_completion",
                    evidence_ref="completed-a",
                    provider_request_id="provider-request-a",
                ),
            )
            current = exposure
            for request in transition_requests:
                current = await store.transition_context_exposure(
                    exposure.session_id,
                    exposure.exposure_id,
                    request,
                )
            assert current.state is ContextExposureState.COMPLETED
            assert current.state_revision == 4
            assert len(current.transitions) == 5
            assert await store.create_context_exposure(exposure, item_exposures) == current
            assert (
                await store.transition_context_exposure(
                    exposure.session_id,
                    exposure.exposure_id,
                    transition_requests[0],
                )
                == current
            )
            fenced_exposure, fenced_items = _planned_context_exposure(
                "fenced-replay",
                first_receipt,
            )
            await store.create_context_exposure(fenced_exposure, fenced_items)
            fenced_prepared = ContextExposureTransitionRequest(
                transition_id="transition-fenced-prepared",
                expected_state="planned",
                expected_revision=0,
                state="prepared",
                occurred_at=transition_time + timedelta(seconds=1),
                evidence_kind="request_prepared",
                evidence_ref="fenced-prepared",
            )
            await store.transition_context_exposure(
                fenced_exposure.session_id,
                fenced_exposure.exposure_id,
                fenced_prepared,
            )
            fenced_failed = ContextExposureTransitionRequest(
                transition_id="transition-fenced-failed",
                expected_state="prepared",
                expected_revision=1,
                state="failed",
                occurred_at=transition_time + timedelta(seconds=2),
                evidence_kind="conclusive_failure",
                evidence_ref="fenced-failure",
            )
            failed_exposure = await store.transition_context_exposure(
                fenced_exposure.session_id,
                fenced_exposure.exposure_id,
                fenced_failed,
            )
            assert (
                await store.transition_context_exposure(
                    fenced_exposure.session_id,
                    fenced_exposure.exposure_id,
                    fenced_failed,
                )
                == failed_exposure
            )
            with pytest.raises(RecallEvidenceConflict):
                await store.transition_context_exposure(
                    fenced_exposure.session_id,
                    fenced_exposure.exposure_id,
                    fenced_failed.model_copy(
                        update={"expected_state": ContextExposureState.DISPATCH_STARTED}
                    ),
                )
            with pytest.raises(ContextExposureTransitionConflict):
                await store.transition_context_exposure(
                    exposure.session_id,
                    exposure.exposure_id,
                    ContextExposureTransitionRequest(
                        transition_id="transition-a-stale",
                        expected_state="planned",
                        expected_revision=0,
                        state="prepared",
                        occurred_at=transition_time + timedelta(seconds=1),
                        evidence_kind="request_prepared",
                        evidence_ref="stale-a",
                    ),
                )
            with pytest.raises(ValidationError, match="lifecycle transition"):
                ContextExposureTransitionRequest(
                    transition_id="transition-a-after-terminal",
                    expected_state="completed",
                    expected_revision=4,
                    state="failed",
                    occurred_at=transition_time + timedelta(seconds=5),
                    evidence_kind="conclusive_failure",
                    evidence_ref="impossible-a",
                )

            concurrent_exposure, concurrent_items = _planned_context_exposure(
                "concurrent",
                first_receipt,
            )
            await store.create_context_exposure(concurrent_exposure, concurrent_items)
            concurrent_results = await asyncio.gather(
                *(
                    store.transition_context_exposure(
                        concurrent_exposure.session_id,
                        concurrent_exposure.exposure_id,
                        ContextExposureTransitionRequest(
                            transition_id=f"transition-concurrent-{winner}",
                            expected_state="planned",
                            expected_revision=0,
                            state="prepared",
                            occurred_at=transition_time + timedelta(seconds=1),
                            evidence_kind="request_prepared",
                            evidence_ref=f"prepared-{winner}",
                        ),
                    )
                    for winner in ("one", "two")
                ),
                return_exceptions=True,
            )
            assert sum(isinstance(result, ContextExposure) for result in concurrent_results) == 1
            assert (
                sum(
                    isinstance(result, ContextExposureTransitionConflict)
                    for result in concurrent_results
                )
                == 1
            )

            store = await _reopen_store(session_store_case, store)
            assert (
                await store.load_recall_receipt(
                    first_receipt.session_id,
                    first_receipt.receipt_id,
                )
                == first_receipt
            )
            assert (
                await store.load_context_exposure(
                    exposure.session_id,
                    exposure.exposure_id,
                )
                == current
            )
            exposure_page = await store.list_context_exposures(
                RecallEvidenceQuery(
                    session_id=exposure.session_id,
                    interaction_id=exposure.interaction_id,
                    model_step_id=exposure.model_step_id,
                )
            )
            assert [item.exposure_id for item in exposure_page.items] == [
                "exposure-a",
                "exposure-attempt-owner-b",
                "exposure-concurrent",
                "exposure-fenced-replay",
            ]

            competing_receipt = _recall_receipt("competing-create")
            changed_competing_receipt = competing_receipt.model_copy(
                update={
                    "frontier_fingerprint": competing_receipt.frontier_fingerprint.model_copy(
                        update={"digest": _memory_evidence_digest("competing-frontier")}
                    )
                }
            )
            receipt_create_results = await asyncio.gather(
                store.create_recall_receipt(competing_receipt),
                store.create_recall_receipt(changed_competing_receipt),
                return_exceptions=True,
            )
            assert sum(isinstance(result, RecallReceipt) for result in receipt_create_results) == 1
            assert (
                sum(isinstance(result, RecallEvidenceConflict) for result in receipt_create_results)
                == 1
            )

            first_create_race, first_create_race_items = _planned_context_exposure(
                "create-race-a",
                first_receipt,
            )
            second_create_race, second_create_race_items = _planned_context_exposure(
                "create-race-b",
                first_receipt,
            )
            second_create_race = second_create_race.model_copy(
                update={
                    "model_attempt_id": first_create_race.model_attempt_id,
                    "provider_attempt_id": first_create_race.provider_attempt_id,
                }
            )
            exposure_create_results = await asyncio.gather(
                store.create_context_exposure(
                    first_create_race,
                    first_create_race_items,
                ),
                store.create_context_exposure(
                    second_create_race,
                    second_create_race_items,
                ),
                return_exceptions=True,
            )
            assert (
                sum(isinstance(result, ContextExposure) for result in exposure_create_results) == 1
            )
            assert (
                sum(
                    isinstance(result, RecallEvidenceConflict) for result in exposure_create_results
                )
                == 1
            )

            await store.delete_session(exposure.session_id)
            assert (
                await store.load_recall_receipt(
                    first_receipt.session_id,
                    first_receipt.receipt_id,
                )
                is None
            )
            assert (
                await store.load_context_exposure(
                    exposure.session_id,
                    exposure.exposure_id,
                )
                is None
            )
            assert not (
                await store.list_recall_receipts(
                    RecallEvidenceQuery(session_id=first_receipt.session_id)
                )
            ).items
            assert not (
                await store.list_context_exposures(
                    RecallEvidenceQuery(session_id=first_receipt.session_id)
                )
            ).items

            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=exposure.session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            assert await store.create_recall_receipt(first_receipt) == first_receipt
            assert await store.create_context_exposure(exposure, item_exposures) == exposure
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_projects_public_memory_attribution(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            store_kind = session_store_case[0]
            session_id = f"memory-attribution-projection-{store_kind}"
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            receipt = _recall_receipt(
                f"projection-{store_kind}",
                session_id=session_id,
            )
            await store.create_recall_receipt(receipt)
            exposure, item_exposures = _planned_context_exposure(
                f"projection-{store_kind}",
                receipt,
            )
            await store.create_context_exposure(exposure, item_exposures)

            report = await runtime_evidence(
                CayuApp(
                    session_store=store,
                    request_footprint=RequestFootprintConfig(
                        fingerprint_key_id="memory-attribution-conformance",
                        fingerprint_key=SecretStr("memory-attribution-conformance-key-material"),
                    ),
                    enable_logging=False,
                ),
                RuntimeEvidenceRequest(
                    root_session_id=session_id,
                    max_sessions=1,
                    max_events=10,
                ),
            )
            attribution = report.sessions[0].memory_attribution
            assert attribution.status is MemoryAttributionStatus.COMPLETE
            assert attribution.observed_receipt_count == 1
            assert attribution.observed_exposure_count == 1
            assert attribution.observed_item_count == 2
            assert (
                attribution.receipts[0].receipt_alias == attribution.exposures[0].receipt_aliases[0]
            )
            assert (
                attribution.receipts[0].items[0].item_alias
                == attribution.exposures[0].items[0].item_alias
            )
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_postgres_context_exposure_creation_locks_session_before_receipts(
    conformance_postgres_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cayu.storage import _postgres_memory_evidence as evidence_storage_module

    async def bypass_identity_lock(_cur: Any, _lock_id: str) -> None:
        # Identity serialization is orthogonal to the session/receipt row-lock
        # order this regression isolates.
        return None

    monkeypatch.setattr(
        evidence_storage_module,
        "_lock_evidence_id",
        bypass_identity_lock,
    )

    async def run() -> None:
        import psycopg

        await _reset_postgres_data(conformance_postgres_dsn)
        store = _new_postgres_store(conformance_postgres_dsn)
        blocker = await psycopg.AsyncConnection.connect(conformance_postgres_dsn)
        creation: asyncio.Task[ContextExposure] | None = None
        try:
            receipt = _recall_receipt("lock-order")
            await store.create(
                RunRequest(
                    agent_name="assistant",
                    session_id=receipt.session_id,
                    messages=[],
                ),
                identity=_identity(),
            )
            await store.create_recall_receipt(receipt)
            exposure, item_exposures = _planned_context_exposure("lock-order", receipt)

            async with blocker.cursor() as cur:
                await cur.execute("SELECT pg_backend_pid()")
                blocker_pid_row = await cur.fetchone()
                assert blocker_pid_row is not None
                blocker_pid = int(blocker_pid_row[0])
                await cur.execute(
                    "SELECT id FROM cayu_sessions WHERE id = %s FOR UPDATE",
                    (receipt.session_id,),
                )
                creation = asyncio.create_task(
                    store.create_context_exposure(exposure, item_exposures)
                )
                for _ in range(500):
                    await cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_stat_activity
                            WHERE datname = current_database()
                              AND pid <> pg_backend_pid()
                              AND wait_event_type = 'Lock'
                              AND %s = ANY(pg_blocking_pids(pid))
                        )
                        """,
                        (blocker_pid,),
                    )
                    waiting = await cur.fetchone()
                    if waiting is not None and bool(waiting[0]):
                        break
                    await asyncio.sleep(0.01)
                else:
                    await cur.execute(
                        """
                        SELECT state, wait_event_type, wait_event, query,
                               pg_blocking_pids(pid)
                        FROM pg_stat_activity
                        WHERE datname = current_database()
                          AND pid <> pg_backend_pid()
                        ORDER BY pid
                        """
                    )
                    activity = await cur.fetchall()
                    raise AssertionError(
                        "Context exposure creation did not wait on the session lock first: "
                        f"{activity!r}"
                    )

                await cur.execute(
                    "DELETE FROM cayu_sessions WHERE id = %s",
                    (receipt.session_id,),
                )
            await blocker.commit()

            with pytest.raises(KeyError, match="Session not found"):
                await asyncio.wait_for(creation, timeout=2)
        finally:
            if creation is not None and not creation.done():
                creation.cancel()
                await asyncio.gather(creation, return_exceptions=True)
            await blocker.rollback()
            await blocker.close()
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_recovers_one_frozen_automatic_recall_frame(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        knowledge = _AutomaticRecallRecoveryKnowledgeStore()
        await knowledge.create_entry(
            KnowledgeEntry(
                id="automatic-recall-recovery",
                text="Atlas recovery evidence says Friday.",
            )
        )
        policy_calls: list[ToolPolicyDecision] = []
        tool_calls: list[dict[str, Any]] = []
        session_id = f"automatic-recall-recovery-{session_store_case[0]}"
        try:
            first_provider = _ApprovalRecoveryProvider()
            first_app = CayuApp(
                session_store=store,
                request_footprint=RequestFootprintConfig(
                    fingerprint_key_id="automatic-recall-recovery",
                    fingerprint_key="automatic-recall-recovery-key-material",
                ),
                enable_logging=False,
            )
            first_app.register_provider(first_provider, default=True)
            first_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    knowledge_store=knowledge,
                ),
                default=True,
            )
            first_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
                context_policy=_automatic_recall_recovery_policy(),
            )
            first_events = [
                event
                async for event in first_app.run(
                    RunRequest(
                        session_id=session_id,
                        agent_name="assistant",
                        messages=[Message.text("user", "When is Atlas released?")],
                    )
                )
            ]
            requested = next(
                event
                for event in first_events
                if event.type is EventType.TOOL_CALL_APPROVAL_REQUESTED
            )
            approval = await _pending_approval_for_public_event(store, requested)
            first_manifest = next(
                part.text
                for message in first_provider.requests[0].messages
                if message.role is MessageRole.USER
                for part in message.content
                if type(part) is TextPart
                and part.text.startswith('<cayu_automatic_memory version="2">')
            )
            assert knowledge.search_count == 1
            assert [event.type for event in first_events].count(
                EventType.AUTOMATIC_RECALL_ADMITTED
            ) == 1

            store = await _reopen_store(session_store_case, store)
            recovery_provider = _ApprovalRecoveryProvider(complete_without_tools=True)
            recovery_app = CayuApp(
                session_store=store,
                request_footprint=RequestFootprintConfig(
                    fingerprint_key_id="automatic-recall-recovery",
                    fingerprint_key="automatic-recall-recovery-key-material",
                ),
                enable_logging=False,
            )
            recovery_app.register_provider(recovery_provider, default=True)
            recovery_app.register_environment(
                Environment(
                    _approval_recovery_environment_spec(),
                    knowledge_store=knowledge,
                ),
                default=True,
            )
            recovery_app.register_agent(
                AgentSpec(name="assistant", model="fake-model"),
                tools=[_ApprovalRecoveryTool(tool_calls)],
                tool_policy=_ChangingApprovalPolicy(policy_calls),
                context_policy=_automatic_recall_recovery_policy(),
            )
            recovered_events = [
                event
                async for event in recovery_app.resolve_tool_approval(
                    ToolApprovalRequest(
                        session_id=session_id,
                        approval_id=approval.approval_id,
                        tool_round_id=approval.tool_round_id,
                        tool_call_id=approval.tool_call_id,
                        decision=ToolApprovalDecision.APPROVE,
                    )
                )
            ]
            recovered_manifest = next(
                part.text
                for message in recovery_provider.requests[0].messages
                if message.role is MessageRole.USER
                for part in message.content
                if type(part) is TextPart
                and part.text.startswith('<cayu_automatic_memory version="2">')
            )

            assert recovered_manifest == first_manifest
            assert knowledge.search_count == 1
            assert EventType.AUTOMATIC_RECALL_STARTED not in [
                event.type for event in recovered_events
            ]
            assert EventType.AUTOMATIC_RECALL_ADMITTED not in [
                event.type for event in recovered_events
            ]
            assert tool_calls == [{"value": "must remain gated"}]
            session = await store.load(session_id)
            assert session is not None
            assert session.status is SessionStatus.COMPLETED
            receipts = (
                await store.list_recall_receipts(RecallEvidenceQuery(session_id=session_id))
            ).items
            exposures = (
                await store.list_context_exposures(RecallEvidenceQuery(session_id=session_id))
            ).items
            assert len(receipts) == 1
            assert len(exposures) == 2
            assert all(exposure.state is ContextExposureState.COMPLETED for exposure in exposures)
            assert {exposure.receipt_ids for exposure in exposures} == {(receipts[0].receipt_id,)}
            assert len({exposure.exposure_id for exposure in exposures}) == 2
            assert len({exposure.composition_fingerprint.digest for exposure in exposures}) == 2
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_session_store_conformance_persists_carried_recall_receipts(
    session_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(session_store_case)
        try:
            for session_id in ("memory-evidence-session", "memory-evidence-other-session"):
                await store.create(
                    RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                    identity=_identity(),
                )
            earlier = _recall_receipt(
                "earlier",
                interaction_id="memory-evidence-earlier-interaction",
                created_at=datetime(2026, 8, 22, 11, 0, tzinfo=UTC),
            )
            current = _recall_receipt("current")
            same_interaction = _recall_receipt("same-interaction")
            other_session = _recall_receipt(
                "other-session",
                session_id="memory-evidence-other-session",
                interaction_id="memory-evidence-earlier-interaction",
                created_at=datetime(2026, 8, 22, 11, 0, tzinfo=UTC),
            )
            for receipt in (earlier, current, same_interaction, other_session):
                await store.create_recall_receipt(receipt)

            exposure, items = _planned_context_exposure("carried", current)
            exposure = exposure.model_copy(update={"carried_receipt_ids": (earlier.receipt_id,)})
            created = await store.create_context_exposure(exposure, items)
            assert created.carried_receipt_ids == (earlier.receipt_id,)
            loaded = await store.load_context_exposure(exposure.session_id, exposure.exposure_id)
            assert loaded == created
            page = await store.list_context_exposures(
                RecallEvidenceQuery(session_id=exposure.session_id)
            )
            assert [item.carried_receipt_ids for item in page.items] == [(earlier.receipt_id,)]
            # Re-anchoring reads item exposures, which cover only linked receipts.
            assert {
                item.receipt_id
                for item in await store.load_recall_item_exposures(
                    exposure.session_id, exposure.exposure_id
                )
            } == {current.receipt_id}

            for label, carried_id, error in (
                ("carried-same-interaction", same_interaction.receipt_id, ValueError),
                ("carried-other-session", other_session.receipt_id, ValueError),
                ("carried-missing", "receipt-missing", KeyError),
            ):
                rejected, rejected_items = _planned_context_exposure(label, current)
                rejected = rejected.model_copy(update={"carried_receipt_ids": (carried_id,)})
                with pytest.raises(error):
                    await store.create_context_exposure(rejected, rejected_items)
                assert (
                    await store.load_context_exposure(rejected.session_id, rejected.exposure_id)
                    is None
                )
        finally:
            await _close_store(store)

    asyncio.run(run())
