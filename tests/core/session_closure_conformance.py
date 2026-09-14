"""Shared descendant closure authority and atomicity scenarios."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from cayu import CayuApp, SessionClosurePolicy
from cayu.events import Event, EventType
from cayu.memory.evidence import RecallReceipt
from cayu.messages import Message
from cayu.runtime.session_closure import SessionClosureRecord
from cayu.runtime.session_message_lifecycle import SessionMessageActionRequest, SessionMessageQuery
from cayu.sessions.base import (
    EnqueueSessionMessageRequest,
    RunRequest,
    RuntimePublicationRequest,
    SessionIdentity,
    SessionOperationPublication,
    SessionStatus,
    _checkpoint_with_session_run_operation,
    runtime_publication_checkpoint_mutation,
)


async def create_closure_session(store, session_id, parent=None):
    await store.create(
        RunRequest(
            agent_name="closure", messages=[], session_id=session_id, parent_session_id=parent
        ),
        identity=SessionIdentity(provider_name="test", model="test"),
    )
    await store.update_status(session_id, SessionStatus.COMPLETED)
    await store.append_event(
        session_id,
        Event(
            id=f"{session_id}-completed",
            session_id=session_id,
            type=EventType.SESSION_COMPLETED,
            payload={},
        ),
    )


async def assert_closure_evidence_pagination(store):
    session_id = "paginated-closure-evidence"
    await create_closure_session(store, session_id)
    fingerprints = {
        field: {"algorithm": "hmac-sha256", "domain": domain, "key_id": "test", "digest": "a" * 64}
        for field, domain in (
            ("situation_fingerprint", "recall_situation"),
            ("source_configuration_fingerprint", "recall_source_configuration"),
            ("admission_policy_fingerprint", "recall_admission_policy"),
            ("access_scope_fingerprint", "recall_access_scope"),
            ("frontier_fingerprint", "recall_frontier"),
        )
    }
    for index in range(101):
        await store.create_recall_receipt(
            RecallReceipt.model_validate(
                {
                    "receipt_id": f"closure-receipt-{index:03d}",
                    "session_id": session_id,
                    "interaction_id": "closure-interaction",
                    "model_step_id": "mstep_" + "1" * 32,
                    "created_at": datetime(2026, 9, 1, tzinfo=UTC),
                    "engine_version": "test.v1",
                    **fingerprints,
                    "sources": [
                        {
                            "source": "knowledge",
                            "required": True,
                            "channels": ["knowledge.lexical"],
                            "state": "complete",
                            "inspected_count": 0,
                            "candidate_limit": 1,
                        }
                    ],
                    "items": [],
                    **{
                        field: 0
                        for field in (
                            "inspected_count",
                            "eligible_count",
                            "admitted_count",
                            "offered_count",
                            "silent_count",
                            "omitted_count",
                        )
                    },
                    "truncated": False,
                }
            )
        )
    app = CayuApp(session_store=store)
    # Receipt records share the native budget with all session-owned records,
    # including the persisted-event delivery row created with the terminal event.
    snapshot = await store.load_session_closure_records(
        session_id, max_records=1000, max_bytes=16 * 1024 * 1024
    )
    total_records = sum(snapshot["counts"].values())
    policy = SessionClosurePolicy(max_records=total_records)
    manifest = await app.inspect_session_closure(session_id, policy=policy)
    evidence = next(
        record for record in manifest.records if record.record_class == "recall_receipts"
    )
    assert evidence.count == 101
    assert evidence.disposition.value == "owned_eligible"
    exported = await app.export_session_closure(session_id, policy=policy)
    records = exported.session_records["session-store/session"]["records"]["recall_receipts"]
    assert len(records) == len({item["receipt_id"] for item in records}) == 101
    lower = await app.inspect_session_closure(
        session_id, policy=SessionClosurePolicy(max_records=total_records - 1)
    )
    assert not lower.complete
    assert await store.load_recall_receipt(session_id, "closure-receipt-100") is not None


async def assert_native_closure_snapshot(store):
    session_id = "native-closure-snapshot"
    await create_closure_session(store, session_id)
    await store.update_labels(session_id, {"one": "first", "two": "second"})
    snapshot = await store.load_session_closure_records(
        session_id, max_records=100, max_bytes=100_000
    )
    assert snapshot["counts"]["session"] == 1
    assert snapshot["counts"]["labels"] == 2
    assert snapshot["counts"]["events"] == 1
    assert snapshot["counts"]["transcript"] == 0
    assert snapshot["counts"]["queued_messages"] == 0
    assert snapshot["counts"]["checkpoint"] == 0
    manifest = await CayuApp(session_store=store).inspect_session_closure(session_id)
    native_records = {
        record.record_class: record
        for record in manifest.records
        if record.store_id == "session-store" and record.record_class != "child_sessions"
    }
    assert manifest.complete
    for name, count in snapshot["counts"].items():
        assert native_records[name].count == count
        assert native_records[name].bytes == snapshot["record_bytes"][name]
        assert native_records[name].disposition.value == ("owned_eligible" if count else "absent")
    with pytest.raises(ValueError):
        await store.load_session_closure_records(session_id, max_records=1, max_bytes=100_000)
    with pytest.raises(ValueError):
        await store.load_session_closure_records(session_id, max_records=100, max_bytes=1)
    assert (await store.load(session_id)).labels == {"one": "first", "two": "second"}


async def assert_populated_closure_export(store):
    session_id = "populated-closure-export"
    await create_closure_session(store, session_id)
    await store.update_status(session_id, SessionStatus.RUNNING)
    accepted = await store.enqueue_session_message(
        EnqueueSessionMessageRequest(
            session_id=session_id,
            idempotency_key="closure-queue",
            content="queued closure input",
            delivery_mode="on_idle",
        )
    )
    batch = await store.deliver_queued_session_messages(
        session_id,
        include_on_idle=True,
        limit=1,
        interaction_id="closure-export-interaction",
        interaction_started_event=Event(
            id="closure-export-start",
            session_id=session_id,
            interaction_id="closure-export-interaction",
            type=EventType.INTERACTION_STARTED,
        ),
    )

    def publish(session, checkpoint, current):
        return SessionOperationPublication(
            checkpoint=checkpoint or {},
            operation_records={"closure-proof": {"proof": "durable operation content"}},
        )

    await store.publish_session_operation(
        session_id, idempotency_key="closure-proof", operation_transform=publish, events=[]
    )
    native = await store.load_session_closure_records(
        session_id, max_records=100, max_bytes=1_000_000
    )
    assert native["counts"]["queue_deliveries"] == 1
    exported = await CayuApp(session_store=store).export_session_closure(session_id)
    snapshot = exported.session_records["session-store/session"]
    assert snapshot["counts"]["queued_messages"] == 1
    message = snapshot["records"]["queued_messages"][0]["message"]
    assert message["queue_id"] == accepted.message.queue_id
    assert message["content"] == "queued closure input"
    assert snapshot["counts"]["transcript"] == 1
    assert snapshot["counts"]["queue_deliveries"] == 1
    delivery = snapshot["records"]["queue_deliveries"][0]
    assert delivery["delivery_id"] == batch.delivery_id
    assert delivery["queue_ids"] == [accepted.message.queue_id]
    assert delivery["include_on_idle"] is True
    assert delivery["has_more"] is False
    assert {
        row["idempotency_key"]: row["record"] for row in snapshot["records"]["session_operations"]
    }["closure-proof"] == {"proof": "durable operation content"}
    for record in exported.manifest.records:
        if record.store_id == "session-store" and record.record_class in snapshot["counts"]:
            assert record.count == snapshot["counts"][record.record_class]
            assert record.bytes == snapshot["record_bytes"][record.record_class]


async def assert_native_closure_admission(store):
    class Dependent:
        store_id = "admission-canary"

        def __init__(self):
            self.mutations = []

        async def inspect_session_closure(self, session_id, *, policy):
            return SessionClosureRecord(
                store_id=self.store_id,
                record_class="records",
                disposition="owned_eligible",
                count=1,
            )

        async def erase_session_closure(self, session_id, *, policy, plan_id):
            self.mutations.append(session_id)
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="erased", count=1
            )

    for kind in ("recovery", "terminal", "budget"):
        for child_policy in ("detach", "recursive"):
            root = f"admission-{kind}-{child_policy}"
            child = f"{root}-child"
            await create_closure_session(store, root)
            await create_closure_session(store, child, root)
            if kind == "recovery":
                now = datetime.now(UTC)
                await store.checkpoint(
                    root,
                    {
                        "incomplete_session_recovery_claim": {
                            "version": 1,
                            "claim_id": "closure-recovery-claim",
                            "claimed_at": now.isoformat(),
                            "claim_expires_at": (now + timedelta(minutes=5)).isoformat(),
                        }
                    },
                )
            elif kind == "terminal":
                await store.transition_status_and_checkpoint(
                    root,
                    from_statuses={SessionStatus.COMPLETED},
                    to_status=SessionStatus.RUNNING,
                    checkpoint_transform=lambda session, checkpoint: (
                        _checkpoint_with_session_run_operation(
                            checkpoint=checkpoint,
                            current_session=session,
                            operation_id="closure-terminal-publication",
                        )
                    ),
                )
                await store.update_status(root, SessionStatus.INTERRUPTED)
            else:
                event = Event(
                    type=EventType.BUDGET_RESERVED,
                    session_id=root,
                    payload={"reservation_id": f"reservation-{root}"},
                )
                await store.claim_budget_reservation_identity(
                    reservation_id=event.payload["reservation_id"],
                    publication_session_id=root,
                    publication_id=event.id,
                )
                await store.append_event(root, event)
            dependent = Dependent()
            app = CayuApp(session_store=store, session_closure_stores=(dependent,))
            before = await store.load_session_closure_records(
                root, max_records=100, max_bytes=1_000_000
            )
            with pytest.raises(ValueError):
                await app.erase_session_closure(
                    root, policy=SessionClosurePolicy(child_policy=child_policy)
                )
            assert not dependent.mutations
            assert (await store.load(child)).parent_session_id == root
            assert (
                await store.load_session_closure_records(root, max_records=100, max_bytes=1_000_000)
                == before
            )


async def assert_detach_closure_conformance(store, competitor=None):
    competitor = store if competitor is None else competitor
    await create_closure_session(store, "root")
    await create_closure_session(store, "other")
    await create_closure_session(store, "first", "root")
    await create_closure_session(store, "second", "root")
    intent = {"root_session_id": "root", "plan_id": "a" * 64, "operation": "detach"}
    with pytest.raises(ValueError):
        await store.detach_session_children("root", ("first", "missing"), closure_receipt=intent)
    assert (await store.load("first")).parent_session_id == "root"
    assert await store.load_session_closure_tombstones("root", "a" * 64) == ()
    first = await store.detach_session_children("root", ("first", "second"), closure_receipt=intent)
    replay = await store.detach_session_children(
        "root", ("second", "first"), closure_receipt=intent
    )
    assert {item["child_session_id"] for item in first} == {"first", "second"}
    assert sorted(first, key=lambda item: item["child_session_id"]) == list(replay)
    for parent, children in (("other", ("first", "second")), ("root", ("first",)), ("root", ())):
        with pytest.raises(ValueError, match="identity conflict"):
            await store.detach_session_children(parent, children, closure_receipt=intent)
    await create_closure_session(store, "late", "root")
    with pytest.raises(ValueError, match="parent identity conflict"):
        await store.delete_session(
            "late",
            closure_receipt={
                "operation": "recursive",
                "original_parent_session_id": "other",
                "root_session_id": "root",
                "target_session_id": "late",
                "plan_id": "b" * 64,
            },
        )
    with pytest.raises(ValueError, match="child edges"):
        await store.delete_session("root", closure_receipt={"plan_id": "delete-plan"})
    assert (await store.load("late")).parent_session_id == "root"
    assert await store.load("root") is not None
    await assert_closure_retry_conformance(store)
    await assert_closure_lineage_ownership(store, competitor)
    await assert_concurrent_detach_intent(store, competitor)
    await assert_leaf_closure_owns_target(store, competitor)
    await assert_closure_evidence_pagination(store)
    await assert_native_closure_snapshot(store)
    await assert_populated_closure_export(store)
    await assert_native_closure_admission(store)
    await assert_closure_publication_replay(store, competitor)
    for deferred in (False, True):
        await assert_closure_queue_admission(store, competitor, deferred=deferred)
    for leased_target in ("root", "child"):
        await assert_closure_event_delivery_ownership(store, competitor, leased_target)


async def assert_closure_event_delivery_ownership(store, competitor, leased_target):
    root = f"closure-delivery-{leased_target}"
    child = f"{root}-child"
    await create_closure_session(store, root)
    await create_closure_session(store, child, parent=root)
    for session_id in (root, child):
        await store.append_event(
            session_id,
            Event(id=f"{session_id}-pending", type="custom.closure.pending", session_id=session_id),
        )
    target = root if leased_target == "root" else child
    claim = await competitor.claim_persisted_event_side_effect(
        session_id=target, event_id=f"{target}-completed", lease_seconds=0.01
    )
    assert claim is not None
    dispatched, settle, deleting, proceed = (asyncio.Event() for _ in range(4))
    deleted = []

    async def external_handler():
        dispatched.set()
        await settle.wait()
        await competitor.mark_persisted_event_side_effect_delivered(claim)

    class Dependent:
        store_id = "delivery-dependent"

        async def inspect_session_closure(self, session_id, *, policy):
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="owned_eligible"
            )

        async def erase_session_closure(self, session_id, *, policy, plan_id):
            deleted.append(session_id)
            deleting.set()
            await proceed.wait()
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="erased"
            )

    app = CayuApp(session_store=store, session_closure_stores=(Dependent(),))
    policy = SessionClosurePolicy(child_policy="recursive")
    handler = asyncio.create_task(external_handler())
    closing = None
    try:
        await asyncio.wait_for(dispatched.wait(), 5)
        # Expiration permits delivery recovery, but does not prove that this
        # already-dispatched handler has stopped. Closure must remain denied.
        expired = asyncio.Event()
        asyncio.get_running_loop().call_later(0.02, expired.set)
        await asyncio.wait_for(expired.wait(), 5)
        with pytest.raises(ValueError, match="event side-effect"):
            await app.erase_session_closure(root, policy=policy)
        assert not handler.done() and not deleted
        assert await store.load(root) is not None and await store.load(child) is not None
        settle.set()
        await handler
        closing = asyncio.create_task(app.erase_session_closure(root, policy=policy))
        closing.add_done_callback(lambda _: deleting.set())
        await asyncio.wait_for(deleting.wait(), 5)
        if closing.done():
            report = await closing
            pytest.fail(f"Closure did not reach dependent cleanup: {report.error}")
        for session_id in (root, child):
            assert (
                await competitor.claim_persisted_event_side_effect(
                    session_id=session_id, event_id=f"{session_id}-pending"
                )
                is None
            )
        unrelated = f"{root}-unrelated"
        await create_closure_session(competitor, unrelated)
        other = await competitor.claim_persisted_event_side_effect(
            session_id=unrelated, event_id=f"{unrelated}-completed"
        )
        assert other is not None
        await competitor.mark_persisted_event_side_effect_delivered(other)
        # The unfiltered worker must skip fenced candidates, not stop at one.
        await competitor.append_event(
            unrelated,
            Event(id=f"{unrelated}-pending", type="custom.closure.pending", session_id=unrelated),
        )
        next_claim = await competitor.claim_persisted_event_side_effect()
        assert next_claim is not None and next_claim.session_id not in {root, child}
        await competitor.mark_persisted_event_side_effect_delivered(next_claim)
        proceed.set()
        assert (await closing).complete
        assert deleted == [child, root]
    finally:
        settle.set()
        proceed.set()
        await asyncio.gather(handler, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)


async def assert_closure_queue_admission(store, competitor, *, deferred):
    session_id = f"closure-pending-queue-{deferred}"
    interaction_id = "closure-pending-interaction"
    source_messages = [Message.text("user", "deferred input")]
    await store.create(
        RunRequest(agent_name="closure", messages=source_messages, session_id=session_id),
        identity=SessionIdentity(provider_name="test", model="test"),
        interaction_started_event=(
            Event(
                id="closure-pending-started",
                type=EventType.INTERACTION_STARTED,
                session_id=session_id,
                interaction_id=interaction_id,
            )
            if deferred
            else None
        ),
        interaction_source_messages=source_messages if deferred else None,
    )
    request = EnqueueSessionMessageRequest(
        session_id=session_id,
        idempotency_key="original",
        content="accepted input",
        delivery_mode="on_idle",
    )
    accepted = await store.enqueue_session_message(request)
    session = await store.load(session_id)
    # Retain a terminal action for exact replay while another item remains queued.
    previous = await store.enqueue_session_message(
        request.model_copy(update={"idempotency_key": "previous"})
    )
    queue = await store.inspect_session_messages(SessionMessageQuery(session_id=session_id))
    revisions = {item.queue_id: item.revision for item in queue.records}
    action_request = SessionMessageActionRequest(
        session_id=session_id,
        session_instance_id=session.instance_id,
        queue_id=previous.message.queue_id,
        expected_revision=revisions[previous.message.queue_id],
        idempotency_key="previous-withdrawal",
        action="withdraw",
    )
    withdrawn = await store.apply_session_message_action(action_request)
    if deferred:
        await store.publish_interaction_transition(
            session_id,
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.INTERRUPTED,
            event=Event(
                id="closure-pending-interrupted",
                type=EventType.INTERACTION_INTERRUPTED,
                session_id=session_id,
                interaction_id=interaction_id,
            ),
        )
        await store.append_event(
            session_id,
            Event(
                id="closure-pending-terminal",
                type=EventType.SESSION_INTERRUPTED,
                session_id=session_id,
            ),
        )
    entered, proceed = asyncio.Event(), asyncio.Event()

    class Dependent:
        store_id = "queue-dependent"

        async def inspect_session_closure(self, session_id, *, policy):
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="owned_eligible"
            )

        async def erase_session_closure(self, session_id, *, policy, plan_id):
            entered.set()
            await proceed.wait()
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="erased"
            )

    app = CayuApp(session_store=store, session_closure_stores=(Dependent(),))
    task = asyncio.create_task(app.erase_session_closure(session_id))
    task.add_done_callback(lambda _: entered.set())
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if task.done():
            report = await task
            pytest.fail(f"Closure completed without reaching its dependent: {report.error}")
        before = await competitor.load_session_closure_records(
            session_id, max_records=100, max_bytes=1_000_000
        )
        replay = await competitor.enqueue_session_message(request)
        assert replay.replayed and replay.message == accepted.message
        action_replay = await competitor.apply_session_message_action(action_request)
        assert action_replay.replayed and action_replay.event == withdrawn.event
        if deferred:
            with pytest.raises(ValueError, match="owned by"):
                await competitor.materialize_deferred_interaction_input(
                    session_id, interaction_id=interaction_id
                )

            def forbidden_transform(*args):
                raise AssertionError("Closed input must not invoke a transcript transform.")

            with pytest.raises(ValueError, match="owned by"):
                await competitor.replace_initial_transcript_messages(
                    session_id,
                    source_messages,
                    source_messages,
                    interaction_id=interaction_id,
                    checkpoint_transform=forbidden_transform,
                )
        for action in ("withdraw", "quarantine"):
            with pytest.raises(ValueError, match="owned by"):
                await competitor.apply_session_message_action(
                    action_request.model_copy(
                        update={
                            "queue_id": accepted.message.queue_id,
                            "expected_revision": revisions[accepted.message.queue_id],
                            "idempotency_key": f"late-{action}",
                            "action": action,
                        }
                    )
                )
        with pytest.raises(ValueError):
            await competitor.enqueue_session_message(
                request.model_copy(update={"content": "conflict"})
            )
        with pytest.raises(ValueError, match="owned by"):
            await competitor.enqueue_session_message(
                request.model_copy(update={"idempotency_key": "new"})
            )
        assert (
            await competitor.load_session_closure_records(
                session_id, max_records=100, max_bytes=1_000_000
            )
            == before
        )
        proceed.set()
        assert (await task).complete
    finally:
        proceed.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def assert_closure_publication_replay(store, competitor):
    session_id = "closure-publication-replay"
    await create_closure_session(store, session_id)
    checkpoint = {"proof": "committed"}
    request = RuntimePublicationRequest(
        publication_id="closure-original",
        kind="model-step",
        intent={"proof": 1},
        mutation=runtime_publication_checkpoint_mutation(None, checkpoint),
        transcript_messages=(),
        events=(
            Event(
                id="closure-original-event", type=EventType.MODEL_COMPLETED, session_id=session_id
            ),
        ),
    )
    published = await store.publish_runtime_publication(session_id, request=request)
    assert not published.replayed
    entered, proceed = asyncio.Event(), asyncio.Event()

    class Dependent:
        store_id = "publication-dependent"

        async def inspect_session_closure(self, session_id, *, policy):
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="owned_eligible"
            )

        async def erase_session_closure(self, session_id, *, policy, plan_id):
            entered.set()
            await proceed.wait()
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="erased"
            )

    app = CayuApp(session_store=store, session_closure_stores=(Dependent(),))
    task = asyncio.create_task(app.erase_session_closure(session_id))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        before = await competitor.load_session_closure_records(
            session_id, max_records=100, max_bytes=1_000_000
        )
        replay = await competitor.publish_runtime_publication(session_id, request=request)
        assert replay.replayed and replay.receipt == published.receipt
        with pytest.raises(ValueError):
            await competitor.publish_runtime_publication(
                session_id, request=request.model_copy(update={"intent": {"proof": 2}})
            )
        replacement = RuntimePublicationRequest(
            publication_id="closure-replacement",
            kind="model-step",
            intent={"proof": 2},
            mutation=runtime_publication_checkpoint_mutation(checkpoint, {"proof": "replaced"}),
            transcript_messages=(),
            events=(
                Event(
                    id="closure-replacement-event",
                    type=EventType.MODEL_COMPLETED,
                    session_id=session_id,
                ),
            ),
        )
        with pytest.raises(ValueError, match="owned by"):
            await competitor.publish_runtime_publication(session_id, request=replacement)
        assert (
            await competitor.load_session_closure_records(
                session_id, max_records=100, max_bytes=1_000_000
            )
            == before
        )
        proceed.set()
        assert (await task).complete
    finally:
        proceed.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def assert_leaf_closure_owns_target(store, competitor):
    root, child = "leaf-first-root", "leaf-first-child"
    await create_closure_session(store, root)
    await create_closure_session(store, child, root)
    entered, proceed = asyncio.Event(), asyncio.Event()
    calls = []

    class Dependent:
        store_id = "leaf-dependent"

        async def inspect_session_closure(self, session_id, *, policy):
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="owned_eligible"
            )

        async def erase_session_closure(self, session_id, *, policy, plan_id):
            calls.append(session_id)
            entered.set()
            await proceed.wait()
            return SessionClosureRecord(
                store_id=self.store_id, record_class="records", disposition="erased"
            )

    dependent = Dependent()
    leaf_app = CayuApp(session_store=store, session_closure_stores=(dependent,))
    ancestor_app = CayuApp(session_store=competitor, session_closure_stores=(dependent,))
    task = asyncio.create_task(leaf_app.erase_session_closure(child))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await assert_closure_blocks_session_admission(competitor, child)
        with pytest.raises(ValueError, match="owned by"):
            await ancestor_app.erase_session_closure(
                root, policy=SessionClosurePolicy(child_policy="recursive")
            )
        assert calls == [child]
        assert (await competitor.load(child)).parent_session_id == root
        proceed.set()
        assert (await task).complete
        assert await store.load(root) is not None
        assert await store.load(child) is None
    finally:
        proceed.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def assert_closure_blocks_session_admission(store, session_id):
    before = await store.load(session_id)
    records_before = await store.load_session_closure_records(
        session_id, max_records=100, max_bytes=1_000_000
    )
    checkpoint_before = await store.load_checkpoint(session_id)
    transformed = []

    def transform(session, checkpoint):
        transformed.append(session.id)
        return checkpoint or {}

    with pytest.raises(ValueError, match="owned by"):
        await store.update_status(session_id, SessionStatus.RUNNING)
    with pytest.raises(ValueError, match="owned by"):
        await store.transition_status_and_checkpoint(
            session_id,
            from_statuses={SessionStatus.COMPLETED},
            to_status=SessionStatus.RUNNING,
            checkpoint_transform=transform,
        )

    def transform_at_time(session, checkpoint, now):
        return transform(session, checkpoint)

    def operation(session, checkpoint, current):
        transformed.append(session.id)
        return SessionOperationPublication(
            checkpoint={"closure-race": "must-not-persist"},
            operation_records={"closure-race": {"value": "must-not-persist"}},
        )

    writes = (
        lambda: store.append_events(
            session_id, [Event(type=EventType.MODEL_STARTED, session_id=session_id)]
        ),
        lambda: store.append_transcript_messages(
            session_id, [Message.text("user", "must not append")]
        ),
        lambda: store.checkpoint(session_id, {"closure-race": "must-not-persist"}),
        lambda: store.transform_checkpoint(session_id, transform),
        lambda: store.transform_checkpoint_with_store_time(session_id, transform_at_time),
        lambda: store.append_transcript_messages_and_transform_checkpoint(
            session_id, [], transform
        ),
        lambda: store.publish_checkpoint_and_events(
            session_id, checkpoint_transform=transform, events=[]
        ),
        lambda: store.publish_session_operation(
            session_id, idempotency_key="closure-race", operation_transform=operation, events=[]
        ),
    )
    for write in writes:
        with pytest.raises(ValueError, match="owned by"):
            await write()
    await store.append_events(session_id, [])
    await store.append_transcript_messages(session_id, [])
    assert await store.load_session_operation(session_id, "closure-race") is None
    assert not transformed
    child_id = f"{session_id}-rejected-new-child"
    with pytest.raises(ValueError, match="owned by"):
        await store.create(
            RunRequest(
                agent_name="closure", messages=[], session_id=child_id, parent_session_id=session_id
            ),
            identity=SessionIdentity(provider_name="test", model="test"),
        )
    assert await store.load(child_id) is None
    assert await store.load(session_id) == before
    assert await store.load_checkpoint(session_id) == checkpoint_before
    assert (
        await store.load_session_closure_records(session_id, max_records=100, max_bytes=1_000_000)
        == records_before
    )


async def assert_closure_lineage_ownership(store, competitor):
    for cancel in (False, True):
        root, child = f"owned-{cancel}-root", f"owned-{cancel}-child"
        ancestor = f"owned-{cancel}-ancestor"
        await create_closure_session(store, ancestor)
        await create_closure_session(store, root, ancestor)
        await create_closure_session(store, child, root)
        entered, proceed = asyncio.Event(), asyncio.Event()
        erased = []

        class BlockingDependent:
            store_id = "blocking-dependent"

            async def inspect_session_closure(self, session_id, *, policy):
                return SessionClosureRecord(
                    store_id=self.store_id,
                    record_class="test-records",
                    disposition="owned_eligible",
                    count=1,
                )

            async def erase_session_closure(
                self,
                session_id,
                *,
                policy,
                plan_id,
                child=child,
                entered=entered,
                proceed=proceed,
                erased=erased,
            ):
                if session_id == child:
                    entered.set()
                    await proceed.wait()
                erased.append(session_id)
                return SessionClosureRecord(
                    store_id=self.store_id,
                    record_class="test-records",
                    disposition="erased",
                    count=1,
                )

        dependent = BlockingDependent()
        app = CayuApp(session_store=store, session_closure_stores=(dependent,))
        policy = SessionClosurePolicy(child_policy="recursive")
        task = asyncio.create_task(app.erase_session_closure(root, policy=policy))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled() and task.cancelling() == 1
            await assert_closure_blocks_session_admission(competitor, root)
            await assert_closure_blocks_session_admission(competitor, child)
            with pytest.raises(ValueError, match="owned by"):
                await competitor.detach_session_children(
                    root,
                    (child,),
                    closure_receipt={
                        "root_session_id": root,
                        "plan_id": "c" * 64,
                        "operation": "detach",
                    },
                )
            with pytest.raises(ValueError, match="owned by"):
                await competitor.delete_session(child)
            with pytest.raises(ValueError, match="owned by"):
                await competitor.delete_session(ancestor)
            competing_deletions = []

            class CompetingDependent(BlockingDependent):
                async def erase_session_closure(
                    self, session_id, *, policy, plan_id, calls=competing_deletions
                ):
                    calls.append(session_id)
                    return SessionClosureRecord(
                        store_id=self.store_id,
                        record_class="test-records",
                        disposition="erased",
                        count=1,
                    )

            competing_app = CayuApp(
                session_store=competitor, session_closure_stores=(CompetingDependent(),)
            )
            for competing_policy in ("reject", "detach", "recursive"):
                with pytest.raises(ValueError, match="owned by"):
                    await competing_app.erase_session_closure(
                        child, policy=SessionClosurePolicy(child_policy=competing_policy)
                    )
            assert not competing_deletions
            assert not erased
            assert (await store.load(child)).parent_session_id == root
            proceed.set()
            if cancel:
                app = CayuApp(session_store=competitor, session_closure_stores=(dependent,))
                result = await app.erase_session_closure(root, policy=policy)
            else:
                result = await task
            assert result.complete
            assert erased == [child, root]
            assert await store.load(child) is None
        finally:
            proceed.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def assert_concurrent_detach_intent(store, competitor):
    root = "concurrent-detach-root"
    await create_closure_session(store, root)
    children = ("concurrent-first", "concurrent-second")
    for child in children:
        await create_closure_session(store, child, root)
    intent = {"root_session_id": root, "plan_id": "d" * 64, "operation": "detach"}
    ready = asyncio.Event()

    async def detach(selected_store, child):
        await ready.wait()
        return await selected_store.detach_session_children(root, (child,), closure_receipt=intent)

    tasks = [
        asyncio.create_task(detach(selected, child))
        for selected, child in zip((store, competitor), children, strict=True)
    ]
    ready.set()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    assert sum(isinstance(result, ValueError) for result in results) == 1
    winner = next(result for result in results if not isinstance(result, BaseException))
    selected = winner[0]["child_session_id"]
    assert await store.detach_session_children(root, (selected,), closure_receipt=intent) == winner
    assert len(await store.load_session_closure_tombstones(root, intent["plan_id"])) == 1
    other = next(child for child in children if child != selected)
    assert (await store.load(other)).parent_session_id == root


async def assert_closure_retry_conformance(store):
    for mode in ("detach", "recursive"):
        root, child, grandchild = (f"{mode}-{name}" for name in ("root", "child", "grandchild"))
        for target, parent in ((root, None), (child, root), (grandchild, child)):
            await create_closure_session(store, target, parent)
        policy = SessionClosurePolicy(child_policy=mode)
        original = store.delete_session
        failed = False

        async def lose_root_ack(session_id, original=original, root=root, **kwargs):
            nonlocal failed
            await original(session_id, **kwargs)
            if session_id == root and not failed:
                failed = True
                raise RuntimeError("root deletion acknowledgement lost")

        store.delete_session = lose_root_ack
        try:
            first = await CayuApp(session_store=store).erase_session_closure(root, policy=policy)
            assert not first.complete
            replay = await CayuApp(session_store=store).erase_session_closure(root, policy=policy)
            assert replay.complete and replay.already_absent
            receipt = await store.load_session_closure_receipt(root, replay.plan_id)
            assert receipt is not None
            if mode == "detach":
                assert (await store.load(child)).parent_session_id is None
                assert (await store.load(grandchild)).parent_session_id == child
                children = next(
                    r
                    for r in receipt["manifest"]["records"]
                    if r["record_class"] == "child_sessions"
                )
                assert children["disposition"] == "retained"
                assert all(
                    item["disposition"] == "retained"
                    for item in receipt["manifest"]["metadata"]["descendant_dispositions"]
                )
            else:
                assert await store.load(child) is None
                assert await store.load(grandchild) is None
        finally:
            store.delete_session = original
