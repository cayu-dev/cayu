"""Shared session-store retention scenarios.

Each scenario receives a ``RetentionHarness``: the store under test, its
controllable ownership clock, and a raw fixture executor that seeds reference
rows the way other Cayu stores sharing the database would. Fixture rows only
describe references; no production behavior is patched.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from cayu.events import Event, EventType
from cayu.messages import Message
from cayu.sessions.base import (
    MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY,
    RunRequest,
    SessionIdentity,
)
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.execution import _ExecutionOwner
from cayu.sessions.records import SessionStatus
from cayu.storage.retention import (
    RetentionAuditState,
    RetentionDisposition,
    RetentionMode,
    RetentionPhase,
    RetentionProgress,
    RetentionProtection,
    RetentionReport,
    SessionRetentionPolicy,
)
from cayu.tasks.creation import TaskCreate

T0 = datetime(2026, 1, 1, tzinfo=UTC)
TOOL_OUTPUT_BYTES = 20_000


class RetentionClock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now


@dataclass
class RetentionHarness:
    store: Any
    tasks: Any = None
    clock: RetentionClock | None = None
    execute: Callable[..., Awaitable[None]] | None = None
    snapshot_tables: Callable[[], Awaitable[None]] | None = None
    postgres: bool = False

    def now(self) -> datetime:
        return datetime.now(UTC) if self.clock is None else self.clock.now

    async def age(self, days: int = 3) -> None:
        """Make every existing session ``days`` older than the store's clock."""

        if self.clock is not None:
            self.clock.now += timedelta(days=days)
            return
        assert self.execute is not None
        await self.execute(
            "UPDATE cayu_sessions SET updated_at = updated_at - make_interval(days => ?)",
            (days,),
        )

    async def sql(self, sql: str, parameters: tuple[Any, ...] = (), **options: Any) -> None:
        assert self.execute is not None
        await self.execute(sql, parameters, **options)


def policy(**overrides: Any) -> SessionRetentionPolicy:
    return SessionRetentionPolicy(older_than=timedelta(days=1), **overrides)


async def create_retention_session(
    harness: RetentionHarness,
    session_id: str,
    *,
    parent: str | None = None,
    status: SessionStatus = SessionStatus.COMPLETED,
    deliver: bool = True,
) -> None:
    store = harness.store
    await store.create(
        RunRequest(
            agent_name="retention", messages=[], session_id=session_id, parent_session_id=parent
        ),
        identity=SessionIdentity(provider_name="test", model="test"),
    )
    await store.append_transcript_messages(
        session_id, [Message.text("user", "hello"), Message.text("assistant", "hi")]
    )
    for index, event_type in enumerate(
        (
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_TEXT_DELTA,
            EventType.MODEL_THINKING_DELTA,
        )
    ):
        await store.append_event(
            session_id,
            Event(
                id=f"{session_id}-delta-{index}",
                session_id=session_id,
                type=event_type,
                payload={"delta": "partial text"},
            ),
        )
    await store.append_event(
        session_id,
        Event(
            id=f"{session_id}-tool",
            session_id=session_id,
            type=EventType.TOOL_CALL_COMPLETED,
            tool_name="read_file",
            payload={
                "tool_call_id": "call-1",
                "result": {
                    "content": "x" * TOOL_OUTPUT_BYTES,
                    "structured": {"path": "notes.txt"},
                    "artifacts": [],
                    "is_error": False,
                },
            },
        ),
    )
    await store.append_event(
        session_id,
        Event(
            id=f"{session_id}-model",
            session_id=session_id,
            type=EventType.MODEL_COMPLETED,
            payload={"usage": {"input_tokens": 11, "output_tokens": 7}},
        ),
    )
    if status is SessionStatus.COMPLETED:
        await complete_retention_session(harness, session_id, deliver=deliver)
        return
    if status is not SessionStatus.PENDING:
        await store.update_status(session_id, status)
    if deliver:
        await deliver_side_effects(store)


async def complete_retention_session(
    harness: RetentionHarness, session_id: str, *, deliver: bool = True
) -> None:
    await harness.store.update_status(session_id, SessionStatus.COMPLETED)
    await harness.store.append_event(
        session_id,
        Event(
            id=f"{session_id}-completed",
            session_id=session_id,
            type=EventType.SESSION_COMPLETED,
            payload={},
        ),
    )
    if deliver:
        await deliver_side_effects(harness.store)


async def deliver_side_effects(store: Any) -> None:
    while (claim := await store.claim_persisted_event_side_effect()) is not None:
        await store.mark_persisted_event_side_effect_delivered(claim)


def item_ids(report: RetentionReport) -> list[str]:
    return [item.item_id for item in report.items]


def protections(report: RetentionReport) -> dict[str, set[RetentionProtection]]:
    return {item.item_id: set(item.protections) for item in report.protected}


async def assert_protected(
    harness: RetentionHarness,
    session_id: str,
    protection: RetentionProtection,
    *,
    mode: RetentionMode = RetentionMode.DELETE,
) -> None:
    """Neither a dry run nor an apply touches the session; the reason is reported."""

    for dry_run in (True, False):
        report = await harness.store.apply_retention_policy(policy(mode=mode, dry_run=dry_run))
        assert session_id not in item_ids(report)
        assert protection in protections(report)[session_id], protections(report)
    assert await harness.store.load(session_id) is not None
    events = await harness.store.load_events(session_id)
    assert any(event.type == EventType.MODEL_TEXT_DELTA for event in events)


# -- scenarios ---------------------------------------------------------------


async def assert_policy_contract() -> None:
    with pytest.raises(ValueError, match="terminal"):
        SessionRetentionPolicy(older_than=timedelta(days=1), statuses={SessionStatus.INTERRUPTED})
    with pytest.raises(ValueError, match="positive"):
        SessionRetentionPolicy(older_than=timedelta(0))
    with pytest.raises(ValueError):
        SessionRetentionPolicy(older_than=timedelta(days=1), max_items=10_001)
    default = SessionRetentionPolicy(older_than=timedelta(days=30))
    assert default.dry_run is True
    assert default.mode is RetentionMode.COMPACT
    assert default.statuses == {SessionStatus.COMPLETED, SessionStatus.FAILED}


async def assert_dry_run_matches_apply_and_compaction_keeps_record(
    harness: RetentionHarness,
) -> None:
    store = harness.store
    await create_retention_session(harness, "old")
    usage_before = await store.read_usage_accounting(EventQuery(session_id="old"))
    transcript_before = await store.load_transcript("old")
    await harness.age()
    await create_retention_session(harness, "fresh")

    dry_run = await store.apply_retention_policy(policy())
    assert dry_run.audit_id is None
    assert item_ids(dry_run) == ["old"]
    [planned] = dry_run.items
    assert planned.disposition is RetentionDisposition.SELECTED
    assert planned.counts == {"delta_events_removed": 3, "tool_outputs_compacted": 1}
    assert len(await store.load_events("old")) == 6, "a dry run changes nothing"

    applied = await store.apply_retention_policy(policy(dry_run=False))
    assert [(item.item_id, item.counts, item.bytes) for item in applied.items] == [
        (planned.item_id, planned.counts, planned.bytes)
    ]
    assert applied.items[0].disposition is RetentionDisposition.APPLIED

    events = await store.load_events("old")
    assert [event.type for event in events] == [
        EventType.TOOL_CALL_COMPLETED,
        EventType.MODEL_COMPLETED,
        EventType.SESSION_COMPLETED,
    ]
    result = events[0].payload["result"]
    assert result["content"].startswith(
        f"[cayu storage retention removed {TOOL_OUTPUT_BYTES} bytes of tool output"
    )
    assert result["structured"] == {"path": "notes.txt"}
    assert await store.load_transcript("old") == transcript_before
    assert await store.read_usage_accounting(EventQuery(session_id="old")) == usage_before
    assert (await store.load("old")).status is SessionStatus.COMPLETED
    assert len(await store.load_events("fresh")) == 6

    again = await store.apply_retention_policy(policy(dry_run=False))
    assert again.items == (), "compaction is idempotent"


async def assert_delete_removes_session_and_audits(harness: RetentionHarness) -> None:
    store = harness.store
    await create_retention_session(harness, "doomed")
    await harness.age()
    dry_run = await store.apply_retention_policy(policy(mode=RetentionMode.DELETE))
    applied = await store.apply_retention_policy(policy(mode=RetentionMode.DELETE, dry_run=False))
    assert [(item.item_id, item.counts, item.bytes) for item in applied.items] == [
        (item.item_id, item.counts, item.bytes) for item in dry_run.items
    ]
    assert applied.items[0].counts["sessions_removed"] == 1
    assert applied.items[0].counts["events_removed"] == 6
    assert applied.items[0].counts["transcript_messages_removed"] == 2
    assert applied.items[0].bytes > TOOL_OUTPUT_BYTES
    assert await store.load("doomed") is None

    record = await store.load_retention_audit(applied.audit_id)
    assert record is not None
    assert record.state is RetentionAuditState.COMPLETED
    assert record.mode is RetentionMode.DELETE
    assert [(entry.item_id, entry.bytes) for entry in record.entries] == [
        ("doomed", applied.items[0].bytes)
    ]
    assert record.summary["totals"]["sessions_removed"] == 1
    [listed] = await store.list_retention_audits(item_id="doomed")
    assert listed.audit_id == applied.audit_id


async def assert_every_apply_writes_an_audit(harness: RetentionHarness) -> None:
    applied = await harness.store.apply_retention_policy(policy(dry_run=False))
    assert applied.items == ()
    record = await harness.store.load_retention_audit(applied.audit_id)
    assert record is not None and record.state is RetentionAuditState.COMPLETED
    assert record.summary["totals"]["items"] == 0


async def assert_non_terminal_sessions_are_never_selected(harness: RetentionHarness) -> None:
    for session_id, status in (
        ("pending", SessionStatus.PENDING),
        ("running", SessionStatus.RUNNING),
        ("interrupted", SessionStatus.INTERRUPTED),
    ):
        await create_retention_session(harness, session_id, status=status)
    await harness.age(days=365)
    for mode in RetentionMode:
        report = await harness.store.apply_retention_policy(policy(mode=mode, dry_run=False))
        assert report.items == ()
        assert report.protected == ()
    for session_id in ("pending", "running", "interrupted"):
        assert len(await harness.store.load_events(session_id)) == 5


async def assert_live_task_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "tasked")
    await harness.tasks.create_task(TaskCreate(type="test", task_id="task-1", session_id="tasked"))
    await harness.age()
    await assert_protected(harness, "tasked", RetentionProtection.LIVE_TASK)
    await harness.tasks.complete_task("task-1", {})
    report = await harness.store.apply_retention_policy(policy(mode=RetentionMode.DELETE))
    assert item_ids(report) == ["tasked"]


async def assert_execution_lease_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "leased")
    session = await harness.store.load("leased")
    await harness.age()

    def owner(expires_at: datetime) -> str:
        now = harness.now()
        return _ExecutionOwner(
            session_id="leased",
            session_instance_id=session.instance_id,
            run_epoch=session.run_epoch,
            token="token",
            owner_kind="in_process_runner",
            owner_id="worker-1",
            owner_label=None,
            operation_id=None,
            claimed_at=now,
            heartbeat_at=now,
            lease_expires_at=expires_at,
            last_progress_at=now,
        ).model_dump_json()

    await harness.sql(
        "INSERT INTO cayu_session_execution_owners (session_id, owner_json) VALUES (?, ?)",
        ("leased", owner(harness.now() + timedelta(hours=1))),
    )
    await assert_protected(harness, "leased", RetentionProtection.EXECUTION_LEASE)
    await harness.sql(
        "UPDATE cayu_session_execution_owners SET owner_json = ? WHERE session_id = ?",
        (owner(harness.now() - timedelta(hours=1)), "leased"),
    )
    report = await harness.store.apply_retention_policy(policy(mode=RetentionMode.DELETE))
    assert item_ids(report) == ["leased"]


async def assert_pending_action_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "awaiting")
    await harness.age()
    state = "state" if harness.postgres else "state_json"
    await harness.sql(
        f"INSERT INTO cayu_checkpoints (session_id, {state}, updated_at, pending_action_flags) "
        "VALUES (?, '{}', ?, 1) ON CONFLICT(session_id) DO UPDATE SET pending_action_flags = 1",
        ("awaiting", T0.isoformat()),
    )
    await assert_protected(harness, "awaiting", RetentionProtection.PENDING_ACTION)


async def assert_pending_clarification_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "participant")
    session = await harness.store.load("participant")
    await harness.age()
    await harness.sql(
        "INSERT INTO cayu_participant_session_bindings (creation_key, request_commitment, "
        "session_id, session_instance_id, application_scope, participant_owner_id, "
        "participant_owner_incarnation, participant_id, participant_incarnation, "
        "lifecycle_revision, configuration_revision, admission_generation, creator_commitment, "
        "authorization_commitment, initial_input_commitment, execution_profile_commitment, "
        "binding_json, receipt_json) VALUES (?, 'c', ?, ?, 'app', 'owner', '1', 'helper', '1', "
        "1, 1, 1, 'c', 'c', 'c', 'c', '{}', '{}')",
        ("creation-1", "participant", session.instance_id),
    )
    await harness.sql(
        "INSERT INTO cayu_collaboration_clarification_questions (scope, namespace, generation, "
        "caller_key, request_id, request_incarnation, participant_id, state, next_due_at_ms, "
        "lineage_namespace, lineage_generation, lineage_key, document) "
        "VALUES ('app', 'ns', 1, 'question-1', 'request-1', '1', 'helper', 'open', 0, "
        "'ns', 1, 'lineage', '{}')",
    )
    await assert_protected(harness, "participant", RetentionProtection.PENDING_CLARIFICATION)
    await harness.sql("UPDATE cayu_collaboration_clarification_questions SET state = 'answered'")
    report = await harness.store.apply_retention_policy(policy(mode=RetentionMode.DELETE))
    assert item_ids(report) == ["participant"]


async def assert_lineage_is_pruned_whole(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "root")
    await create_retention_session(harness, "fork", parent="root")
    await create_retention_session(
        harness, "running-child", parent="fork", status=SessionStatus.RUNNING
    )
    await create_retention_session(harness, "parent-running", status=SessionStatus.RUNNING)
    await create_retention_session(harness, "done-child", parent="parent-running")
    await harness.age()
    await assert_protected(harness, "root", RetentionProtection.LINEAGE)
    await assert_protected(harness, "fork", RetentionProtection.LINEAGE)
    await assert_protected(harness, "done-child", RetentionProtection.LINEAGE)

    await complete_retention_session(harness, "running-child")
    await complete_retention_session(harness, "parent-running")
    await harness.age()
    report = await harness.store.apply_retention_policy(
        policy(mode=RetentionMode.DELETE, dry_run=False)
    )
    ordered = item_ids(report)
    assert report.protected == ()
    assert set(ordered) == {"root", "fork", "running-child", "parent-running", "done-child"}
    # Children are removed before their parents.
    assert ordered.index("running-child") < ordered.index("fork") < ordered.index("root")
    assert ordered.index("done-child") < ordered.index("parent-running")


async def assert_budget_takes_children_first(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "a-root")
    await create_retention_session(harness, "a-child", parent="a-root")
    await harness.age(days=1)
    await create_retention_session(harness, "b-solo")
    await harness.age()
    report = await harness.store.apply_retention_policy(
        policy(mode=RetentionMode.DELETE, max_items=1, dry_run=False)
    )
    assert item_ids(report) == ["a-child"]
    assert report.deferred_count == 2
    assert await harness.store.load("a-root") is not None
    report = await harness.store.apply_retention_policy(
        policy(mode=RetentionMode.DELETE, max_bytes=1, dry_run=False)
    )
    assert item_ids(report) == ["a-root"]


async def assert_checkpoint_dependency_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "viewed")
    session = await harness.store.load("viewed")
    await harness.age()
    await harness.sql(
        "INSERT INTO cayu_context_views (view_id, publication_key, source_owner_scope, "
        "source_owner_id, source_owner_incarnation, source_session_id, "
        "source_session_instance_id, transcript_cursor, projection_schema, "
        "extension_set_commitment, manifest_json) "
        "VALUES ('view-1', 'pub-1', 'app', 'owner', '1', ?, ?, 2, 'v1', 'c', '{}')",
        ("viewed", session.instance_id),
    )
    await harness.sql(
        "INSERT INTO cayu_context_view_selections (selection_key, request_commitment, view_id, "
        "owner_scope, owner_id, owner_incarnation, state, pin_commitment, expires_at_ms, "
        "receipt_json) VALUES ('sel-1', 'c', 'view-1', 'app', 'reader', '1', 'adopted', 'p', 0, "
        "'{}')",
    )
    await assert_protected(harness, "viewed", RetentionProtection.CHECKPOINT_DEPENDENCY)


async def assert_snapshot_pin_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "captured")
    await harness.age()
    if harness.snapshot_tables is None:
        pytest.skip("The agent snapshot store has no tables in this backend's database.")
    await harness.snapshot_tables()
    await harness.sql(
        "INSERT INTO cayu_agent_snapshot_bindings (binding_id, snapshot_root, "
        "authority_scope_fingerprint, binding_document, snapshot_document, "
        "put_receipt_document) VALUES ('binding-1', 'root-1', 'f', '{}', ?, '{}')",
        (json.dumps({"capture": {"session_id": "captured"}}),),
    )
    await harness.sql(
        "INSERT INTO cayu_agent_snapshot_pins (pin_id, snapshot_root, binding_id, document, "
        "released) VALUES ('pin-1', 'root-1', 'binding-1', '{}', 0)",
    )
    await assert_protected(harness, "captured", RetentionProtection.SNAPSHOT_PIN)
    await harness.sql("UPDATE cayu_agent_snapshot_pins SET released = 1")
    report = await harness.store.apply_retention_policy(policy(mode=RetentionMode.DELETE))
    assert item_ids(report) == ["captured"]


async def assert_eval_reference_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "evaluated")
    await harness.age()
    document = json.dumps({"cases": [{"trials": [{"session_id": "evaluated"}]}]})
    await harness.sql(
        "INSERT INTO cayu_eval_run_trial_checkpoints (run_id, case_id, trial_number, "
        "checkpoint_json, document_bytes) VALUES ('run-1', 'case-1', 1, ?, ?)",
        (document, len(document.encode())),
        foreign_keys=False,
    )
    await assert_protected(harness, "evaluated", RetentionProtection.EVAL_REFERENCE)


async def assert_knowledge_evidence_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "cited")
    await harness.age()
    locator, metadata = (
        ("locator", "metadata")
        if harness.postgres
        else (
            "locator_json",
            "metadata_json",
        )
    )
    await harness.sql(
        "INSERT INTO cayu_knowledge_evidence (id, entry_id, entry_revision, role, source_type, "
        f"source_id, source_revision, {locator}, disposition, created_at, {metadata}) "
        "VALUES ('evidence-1', 'entry-1', 1, 'origin', 'session_event', 'cited-tool', 'r1', "
        "'{}', 'live', ?, '{}')",
        (T0.isoformat(),),
        foreign_keys=False,
    )
    await assert_protected(harness, "cited", RetentionProtection.KNOWLEDGE_EVIDENCE)
    await harness.sql("UPDATE cayu_knowledge_evidence SET disposition = 'detached'")
    report = await harness.store.apply_retention_policy(policy(mode=RetentionMode.DELETE))
    assert item_ids(report) == ["cited"]


async def assert_product_operation_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "product")
    await harness.age()
    await harness.sql(
        "INSERT INTO cayu_product_operations (work_id, public_id, tenant_id, subject_id, "
        "idempotency_key, request_fingerprint, session_id, task_id, request_text, status) "
        "VALUES ('work-1', 'public-1', 'tenant', 'subject', 'key', 'f', 'product', 'task-x', "
        "'do it', 'pending')",
    )
    await assert_protected(harness, "product", RetentionProtection.PRODUCT_OPERATION)


async def assert_undelivered_events_protect(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "undelivered", deliver=False)
    await harness.age()
    await assert_protected(harness, "undelivered", RetentionProtection.EVENT_DELIVERY_BACKLOG)
    await deliver_side_effects(harness.store)
    report = await harness.store.apply_retention_policy(policy(mode=RetentionMode.DELETE))
    assert item_ids(report) == ["undelivered"]


async def assert_session_export_protects_compaction(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "exported")
    await harness.age()
    record = "record" if harness.postgres else "record_json"
    await harness.sql(
        f"INSERT INTO cayu_session_operations (session_id, idempotency_key, {record}, "
        "updated_at) VALUES ('exported', 'session-export:test', '{}', ?)",
        (T0.isoformat(),),
    )
    await assert_protected(
        harness, "exported", RetentionProtection.SESSION_EXPORT, mode=RetentionMode.COMPACT
    )


async def assert_closure_in_progress_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "closing")
    await harness.age()
    await harness.sql(
        "INSERT INTO cayu_task_session_closure_claims (session_id, plan_id, claim_json) "
        "VALUES ('closing', ?, '{}')",
        ("a" * 64,),
    )
    await assert_protected(harness, "closing", RetentionProtection.CLOSURE_IN_PROGRESS)


async def assert_store_erasure_guard_protects(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "staged")
    await harness.age()
    record = "record" if harness.postgres else "record_json"
    await harness.sql(
        f"INSERT INTO cayu_session_operations (session_id, idempotency_key, {record}, "
        "updated_at) VALUES ('staged', ?, '{}', ?)",
        (MODEL_COMPLETION_ACTIVE_STAGE_STORAGE_KEY, T0.isoformat()),
    )
    for mode in RetentionMode:
        await assert_protected(harness, "staged", RetentionProtection.ERASURE_GUARD, mode=mode)


async def assert_caller_protected_ids(harness: RetentionHarness) -> None:
    await create_retention_session(harness, "external-root")
    await create_retention_session(harness, "external-child", parent="external-root")
    await harness.age()
    report = await harness.store.apply_retention_policy(
        policy(mode=RetentionMode.DELETE, dry_run=False),
        protected_session_ids=["external-root"],
    )
    assert report.items == ()
    assert protections(report) == {
        "external-root": {RetentionProtection.CALLER_PROTECTED},
        "external-child": {RetentionProtection.LINEAGE},
    }


async def assert_apply_releases_the_store_between_lineages(harness: RetentionHarness) -> None:
    for session_id in ("batch-a", "batch-b", "batch-c"):
        await create_retention_session(harness, session_id)
    await harness.age()
    seen: list[RetentionProgress] = []

    async def progress(event: RetentionProgress) -> None:
        seen.append(event)
        if event.phase is RetentionPhase.BATCH and event.batch_index == 0:
            # Another writer proceeds while the apply is between lineages.
            await asyncio.wait_for(create_retention_session(harness, "concurrent"), timeout=10)
            # A lineage planned for a later batch becomes protected before it runs.
            await harness.tasks.create_task(
                TaskCreate(type="test", task_id="late-task", session_id="batch-c")
            )

    report = await harness.store.apply_retention_policy(
        policy(mode=RetentionMode.DELETE, dry_run=False), progress=progress
    )
    assert [event.phase for event in seen] == [
        RetentionPhase.PLANNED,
        RetentionPhase.BATCH,
        RetentionPhase.BATCH,
        RetentionPhase.BATCH,
    ]
    assert seen[0].planned_items == 3
    assert item_ids(report) == ["batch-a", "batch-b"]
    assert protections(report) == {"batch-c": {RetentionProtection.LIVE_TASK}}
    assert seen[3].protected[0].item_id == "batch-c"
    assert await harness.store.load("batch-c") is not None
    assert await harness.store.load("concurrent") is not None
    record = await harness.store.load_retention_audit(report.audit_id)
    assert [entry.item_id for entry in record.entries] == ["batch-a", "batch-b"]


async def assert_same_size_eval_reference_update_protects(harness: RetentionHarness) -> None:
    from tests.evals.eval_store_conformance import _scenario
    from tests.evals.test_corpus_execution import _corpus

    from cayu.evals.store import (
        EvalRunInvocation,
        EvalRunRequest,
        EvalScenarioRunInvocation,
        EvalScenarioRunProgress,
        EvalScenarioTrialPhase,
        EvalScenarioTrialProgress,
    )
    from cayu.storage.evals_sqlite import SQLiteEvalStore
    from cayu.vaults.redaction import SecretRedactor

    await create_retention_session(harness, "target")
    await harness.age()
    if harness.postgres:
        from cayu.storage.evals_postgres import PostgresEvalStore

        evals = PostgresEvalStore(harness.store._conninfo, min_size=1, max_size=1)
    else:
        evals = SQLiteEvalStore(harness.store.path)
    try:
        corpus = _corpus(trials=1)
        scenario = _scenario(corpus, text="controlled retention test")
        redact = SecretRedactor().redact_json
        await evals.save_corpus(corpus, redact_json=redact)
        await evals.save_scenario(scenario, redact_json=redact)
        suite = corpus.suites[0]
        binding = "sha256:" + "b" * 64
        await evals.admit_run(
            EvalRunRequest(
                run_id="retention-reference-run",
                idempotency_key="sha256:" + "9" * 64,
                corpus_revision=corpus.revision,
                target_key=corpus.target_key,
                suite_id=suite.id,
                suite_revision=suite.revision,
                max_concurrency=1,
                invocation=EvalRunInvocation(
                    scenario=EvalScenarioRunInvocation(
                        scenario_revision=scenario.revision,
                        binding_revision=binding,
                        trials=1,
                        timeout_seconds=30,
                    )
                ),
            ),
            redact_json=redact,
        )
        claimed = await evals.claim_run(lease_seconds=120)
        assert claimed is not None
        trial = EvalScenarioTrialProgress(
            trial_number=1,
            phase=EvalScenarioTrialPhase.PENDING,
            session_id="others",
            next_event_sequence=0,
        )
        initial = EvalScenarioRunProgress.create(
            scenario_revision=scenario.revision,
            binding_revision=binding,
            attempt=claimed.claim.epoch,
            trials=(trial,),
        )
        await evals.initialize_scenario_progress(claimed.claim, initial)

        async def progress(update: RetentionProgress) -> None:
            if update.phase is RetentionPhase.PLANNED:
                assert update.planned_items == 1
                updated = await evals.update_scenario_trial(
                    claimed.claim, trial.model_copy(update={"session_id": "target"})
                )
                assert updated.scenario_progress is not None
                assert len(initial.model_dump_json()) == len(
                    updated.scenario_progress.model_dump_json()
                )

        report = await harness.store.apply_retention_policy(
            policy(mode=RetentionMode.DELETE, dry_run=False), progress=progress
        )
        assert report.items == ()
        assert protections(report) == {"target": {RetentionProtection.EVAL_REFERENCE}}
        assert await harness.store.load("target") is not None
    finally:
        await evals.close()


SCENARIOS: tuple[Callable[[RetentionHarness], Awaitable[None]], ...] = (
    assert_dry_run_matches_apply_and_compaction_keeps_record,
    assert_delete_removes_session_and_audits,
    assert_every_apply_writes_an_audit,
    assert_non_terminal_sessions_are_never_selected,
    assert_live_task_protects,
    assert_execution_lease_protects,
    assert_pending_action_protects,
    assert_pending_clarification_protects,
    assert_lineage_is_pruned_whole,
    assert_budget_takes_children_first,
    assert_checkpoint_dependency_protects,
    assert_snapshot_pin_protects,
    assert_eval_reference_protects,
    assert_same_size_eval_reference_update_protects,
    assert_knowledge_evidence_protects,
    assert_product_operation_protects,
    assert_undelivered_events_protect,
    assert_session_export_protects_compaction,
    assert_closure_in_progress_protects,
    assert_store_erasure_guard_protects,
    assert_caller_protected_ids,
    assert_apply_releases_the_store_between_lineages,
)
