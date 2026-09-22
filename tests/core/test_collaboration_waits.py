"""Public qualification for bounded collaboration waits."""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core._execution_profile_fixtures import create_admitted_session
from tests.core.test_collaboration_request_foundation import public_setup
from tests.core.test_participant_identity import app as collaboration_app
from tests.core.test_session_continuation import _context

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.collaboration._capabilities import CapabilityDescriptor
from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    OwnerRef,
)
from cayu.collaboration.access import CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.collaboration.requests import RequestControl
from cayu.collaboration.waits import (
    CollaborationWait,
    WaitEvidence,
    WaitSnapshot,
    request_object_ref,
    source_key,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runtime._session_continuation import (
    ContinuationNamespace,
    ContinuationTicket,
    ContinuationWait,
    continuation_namespace_id,
)
from cayu.runtime._session_continuation_owner import LATCH_FAMILY, SessionContinuationOwner
from cayu.sessions.base import InMemorySessionStore, RunRequest
from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


def wait_for(receipt, initialized, *, predicate="ALL_SUCCESS", threshold=None, targets=None):
    return CollaborationWait(
        operation=initialized.operation("wait"),
        source_owner=initialized.owner,
        targets=(receipt.expected,) if targets is None else tuple(targets),
        predicate=predicate,
        predicate_version=1,
        threshold=threshold,
        deadline=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        failure_policy="settle",
        service_policy="external_observer",
        projection=None,
        wait_edge_revision=1,
        initiator=receipt.expected.initiator,
    )


async def test_wait_registration_is_exact_and_observe_elects(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    initialized = values[1]
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("cancel-source"),
            expected=receipt.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    wait = wait_for(receipt, values[1])

    registered = await application.register_collaboration_wait(wait, context=resolver.context)
    assert registered.state == "pending"
    assert registered.source_pins
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        anchor = await store._anchor(tx, initialized, SecretRedactor())
    assert anchor.reserved_events >= registered.reserved_events
    assert anchor.reserved_bytes >= registered.reserved_bytes
    assert (
        await application.register_collaboration_wait(wait, context=resolver.context)
    ) == registered
    lookup = await application.lookup_collaboration_wait(wait, context=resolver.context)
    assert isinstance(lookup, ExactMatch)
    assert lookup.receipt == registered.registration
    changed = wait.model_copy(update={"failure_policy": "changed"})
    assert isinstance(
        await application.lookup_collaboration_wait(changed, context=resolver.context),
        ExactConflict,
    )
    missing = wait.model_copy(
        update={"operation": wait.operation.model_copy(update={"caller_key": "missing"})}
    )
    assert isinstance(
        await application.lookup_collaboration_wait(missing, context=resolver.context),
        ExactNotFound,
    )
    observed = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert observed.state == "elected"
    assert observed.election is not None
    assert observed.election.result == "failure"
    assert observed.source_pins == ()
    assert observed.reserved_events == 0
    assert observed.reserved_bytes == 0
    assert await application.inspect_collaboration_wait(wait, context=resolver.context) == observed


async def test_wait_registration_binds_authenticated_initiator(stores):
    application, resolver, values = await public_setup(stores())
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1]).model_copy(
        update={"initiator": source.expected.initiator.model_copy(update={"principal": "forged"})}
    )
    with pytest.raises(CollaborationAccessDenied):
        await application.register_collaboration_wait(wait, context=resolver.context)
    assert (
        await application.inspect_collaboration_wait(
            wait_for(source, values[1]), context=resolver.context
        )
        is None
    )


async def test_wait_cancel_releases_source_pin_without_election(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1], predicate="ANY_SUCCESS")
    await application.register_collaboration_wait(wait, context=resolver.context)

    cancelled = await application.cancel_collaboration_wait(wait, context=resolver.context)
    assert cancelled.state == "cancelled"
    assert cancelled.election is None
    assert cancelled.source_pins == ()
    assert await application.cancel_collaboration_wait(wait, context=resolver.context) == cancelled


async def test_wait_registration_replays_after_acknowledgement_loss(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    original = store.register_wait
    state = {"lost": False}

    async def lose_ack(*args, **kwargs):
        result = await original(*args, **kwargs)
        if not state["lost"]:
            state["lost"] = True
            raise RuntimeError("wait registration acknowledgement lost")
        return result

    store.register_wait = lose_ack
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        await application.register_collaboration_wait(wait, context=resolver.context)
    store.register_wait = original
    replay = await application.register_collaboration_wait(wait, context=resolver.context)
    assert replay.registration.registration_digest
    assert replay.revision == 1


async def test_wait_mutation_cancellation_drains_active_transaction_before_close(
    stores, monkeypatch
):
    store = stores()
    application, resolver, values = await public_setup(store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    await application.register_collaboration_wait(wait, context=resolver.context)
    original = store._transaction
    started = asyncio.Event()
    release = asyncio.Event()

    @asynccontextmanager
    async def blocked(scope, *, write):
        async with original(scope, write=write) as tx:
            if write:
                assert asyncio.current_task() in store._owners.pending
                started.set()
                await release.wait()
            yield tx

    monkeypatch.setattr(store, "_transaction", blocked)
    mutation_task = asyncio.create_task(
        application.cancel_collaboration_wait(wait, context=resolver.context)
    )
    try:
        await asyncio.wait_for(started.wait(), 5)
        mutation_task.cancel("first cancellation")
        mutation_task.cancel("second cancellation")
        with pytest.raises(asyncio.CancelledError):
            await mutation_task
        assert mutation_task.cancelled()
        assert mutation_task.cancelling() == 2
        assert len(store._owners.pending) == 1
        close_task = asyncio.create_task(store.close())

        async def wait_for_closing():
            while not store._owners.closed:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait_for_closing(), 5)
        assert not close_task.done()
        with pytest.raises(CollaborationUnavailable, match="closing"):
            await store.load_wait(values[1], wait, redactor=SecretRedactor())
        release.set()
        await asyncio.wait_for(close_task, 10)
        assert not store._owners.pending
    finally:
        release.set()
        await asyncio.gather(mutation_task, return_exceptions=True)
        monkeypatch.setattr(store, "_transaction", original)

    reopened = stores()
    if reopened is store:
        # Memory has no reopen protocol; inspect the committed owner state.
        from cayu.collaboration.waits import wait_operation_key

        raw = store._scopes[values[1].binding.application_scope][
            ("operations", wait_operation_key(wait))
        ]
        settled = WaitSnapshot.model_validate(raw)
    else:
        replay_app = collaboration_app(
            reopened,
            values[0]._participant_coordinator._registration,
            collaboration_requests=application._request_coordinator._registration,
        )
        await replay_app.initialize_collaboration()
        settled = await replay_app.cancel_collaboration_wait(wait, context=resolver.context)
        assert (
            await replay_app.inspect_collaboration_wait(wait, context=resolver.context) == settled
        )
    assert settled.state == "cancelled"
    assert not settled.source_pins
    assert [event.kind for event in settled.events].count("cancelled") == 1


async def test_wait_evidence_replays_without_duplicate_after_acknowledgement_loss(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1], predicate="ALL_SETTLED")
    await application.register_collaboration_wait(wait, context=resolver.context)
    original = store.record_wait_evidence
    state = {"lost": False}

    async def lose_ack(*args, **kwargs):
        result = await original(*args, **kwargs)
        if not state["lost"]:
            state["lost"] = True
            raise RuntimeError("wait evidence acknowledgement lost")
        return result

    store.record_wait_evidence = lose_ack
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        await application.observe_collaboration_wait(wait, context=resolver.context)
    store.record_wait_evidence = original
    replay = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert replay.state == "pending"
    assert len(replay.evidence) == 1
    assert len(replay.events) == 2


async def test_wait_observation_cancellation_preserves_committed_evidence(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1], predicate="ALL_SETTLED")
    await application.register_collaboration_wait(wait, context=resolver.context)
    original = store.record_wait_evidence
    committed = asyncio.Event()
    release = asyncio.Event()

    async def pause_after_commit(*args, **kwargs):
        result = await original(*args, **kwargs)
        committed.set()
        await release.wait()
        return result

    store.record_wait_evidence = pause_after_commit
    task = asyncio.create_task(
        application.observe_collaboration_wait(wait, context=resolver.context)
    )
    await committed.wait()
    task.cancel("cancel wait observation after evidence commit")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    store.record_wait_evidence = original
    replay = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert replay.state == "pending"
    assert len(replay.evidence) == 1
    assert len(replay.events) == 2


async def test_wait_source_release_retries_after_acknowledgement_loss(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    await application.register_collaboration_wait(wait, context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("release-ack-loss"),
            expected=source.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    original = store.release_wait_sources
    state = {"lost": False}

    async def lose_ack(*args, **kwargs):
        if not state["lost"]:
            state["lost"] = True
            raise RuntimeError("wait release acknowledgement lost")
        return await original(*args, **kwargs)

    store.release_wait_sources = lose_ack
    with pytest.raises(RuntimeError, match="acknowledgement lost"):
        await application.observe_collaboration_wait(wait, context=resolver.context)
    store.release_wait_sources = original
    replay = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert replay.state == "elected"
    assert replay.source_pins == ()


@pytest.mark.parametrize("publication", ["before", "concurrent", "after"])
async def test_wait_registration_catches_publication_race(stores, publication):
    store = stores()
    application, resolver, values = await public_setup(store)
    other = collaboration_app(
        store,
        values[0]._participant_coordinator._registration,
        collaboration_requests=application._request_coordinator._registration,
    )
    await other.initialize_collaboration()
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    control = RequestControl(
        operation=values[1].operation(f"wait-publication-{publication}"),
        expected=source.expected,
        expected_revision=1,
        kind="cancel",
    )

    async def register():
        return await application.register_collaboration_wait(wait, context=resolver.context)

    async def publish():
        return await other.control_collaboration_request(control, context=resolver.context)

    if publication == "before":
        await publish()
        await register()
    elif publication == "after":
        await register()
        await publish()
    else:
        await asyncio.gather(register(), publish())
    observed = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert observed.state == "elected"
    assert observed.election is not None
    assert observed.election.result == "failure"
    assert observed.source_pins == ()


async def test_wait_election_is_idempotent_across_independent_workers(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    other = collaboration_app(
        store,
        values[0]._participant_coordinator._registration,
        collaboration_requests=application._request_coordinator._registration,
    )
    await other.initialize_collaboration()
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    await application.register_collaboration_wait(wait, context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("independent-worker-source"),
            expected=source.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    first, second = await asyncio.gather(
        application.observe_collaboration_wait(wait, context=resolver.context),
        other.observe_collaboration_wait(wait, context=resolver.context),
    )
    assert first == second
    assert first.state == "elected"
    assert len(first.events) == 4


async def test_wait_elects_distinct_multi_target_batch(stores):
    application, resolver, values = await public_setup(stores())
    first = await application.accept_collaboration_request(values[4], context=resolver.context)
    second_request = values[4].model_copy(
        update={"operation": values[1].operation("second-wait-target")}
    )
    second = await application.accept_collaboration_request(
        second_request, context=resolver.context
    )
    for index, receipt in enumerate((first, second)):
        await application.control_collaboration_request(
            RequestControl(
                operation=values[1].operation(f"multi-target-cancel-{index}"),
                expected=receipt.expected,
                expected_revision=1,
                kind="cancel",
            ),
            context=resolver.context,
        )
    wait = wait_for(
        first,
        values[1],
        predicate="ALL_SETTLED",
        targets=(first.expected, second.expected),
    )
    registered = await application.register_collaboration_wait(wait, context=resolver.context)
    assert len(registered.registration.wait.targets) == 2
    observed = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert observed.state == "elected"
    assert observed.election is not None
    assert observed.election.result == "settled"
    assert {item.status for item in observed.evidence} == {"settled"}
    assert observed.source_pins == ()


async def test_wait_normalizes_identical_duplicate_targets(stores):
    application, resolver, values = await public_setup(stores())
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1], targets=(source.expected, source.expected))

    registered = await application.register_collaboration_wait(wait, context=resolver.context)

    assert len(registered.registration.wait.targets) == 1


async def test_wait_rejects_maximum_terminal_envelope_before_registration(stores):
    application, resolver, values = await public_setup(stores())
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    targets = []
    for index in range(64):
        operation = values[1].operation(f"maximum-wait-target-{index}")
        request = values[4].model_copy(update={"operation": operation})
        targets.append(
            source.expected.model_copy(
                update={
                    "operation": operation,
                    "intent": source.expected.intent.model_copy(update={"request": request}),
                }
            )
        )
    with pytest.raises(ValueError, match="Invalid or oversized"):
        wait_for(source, values[1], targets=targets)


async def test_wait_source_pin_blocks_cross_generation_retirement_until_release(stores):
    from cayu.collaboration.lifecycle import NamespaceRetire, NamespaceRotate

    application, resolver, values = await public_setup(stores())
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    await application.register_collaboration_wait(wait, context=resolver.context)
    current = (
        await application.inspect_collaboration_namespace(context=identity_tests.CONTEXT)
    ).current
    rotated = await application.rotate_collaboration_namespace(
        NamespaceRotate(
            operation=current.reference.operation("wait-rotate"),
            namespace=current.reference,
            expected_revision=current.revision,
        ),
        context=identity_tests.CONTEXT,
    )
    retirement = NamespaceRetire(
        operation=rotated.successor.reference.operation("wait-retire"),
        namespace=rotated.namespace.reference,
        expected_revision=rotated.namespace.revision,
        expected_retired_through=0,
    )
    with pytest.raises(CollaborationConflict):
        await application.retire_collaboration_namespace(retirement, context=identity_tests.CONTEXT)

    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("cross-generation-source-cancel"),
            expected=source.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    elected = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert elected.source_pins == ()
    retired = await application.retire_collaboration_namespace(
        retirement, context=identity_tests.CONTEXT
    )
    assert retired.namespace.state == "retired"
    assert (
        await application.inspect_collaboration_wait(wait, context=resolver.context)
    ).state == "elected"


async def test_wait_deadline_elects_explicit_unavailable_evidence(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1]).model_copy(
        update={
            "deadline": (datetime.now(UTC) + timedelta(seconds=2)).isoformat(),
        }
    )
    await application.register_collaboration_wait(wait, context=resolver.context)
    await asyncio.sleep(2.1)
    expired = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert expired.state == "elected"
    assert expired.election is not None
    assert expired.election.result == "unavailable"
    assert expired.evidence[0].status == "unavailable"
    assert expired.source_pins == ()


async def test_wait_deadline_retries_prior_evidence_after_deadline(stores):
    """A pre-deadline unavailable observation must not suppress deadline election."""
    store = stores()
    application, resolver, values = await public_setup(store)
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1]).model_copy(
        update={
            "deadline": (datetime.now(UTC) + timedelta(seconds=2)).isoformat(),
        }
    )
    await application.register_collaboration_wait(wait, context=resolver.context)
    prior = WaitEvidence(
        target=receipt.expected.intent.selection.reference,
        status="unavailable",
        source_sequence=1,
        accepted_at_ms=1,
        observed_at_ms=1,
        receipt_digest=sha256(source_key(receipt.expected).encode()).hexdigest(),
    )
    await store.record_wait_evidence(values[1], wait, prior, redactor=SecretRedactor())
    await asyncio.sleep(2.1)
    expired = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert expired.state == "elected"
    assert expired.election is not None
    assert expired.election.result == "unavailable"
    assert len(expired.evidence) == 1


async def test_wait_cancel_after_deadline_is_expiry(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1]).model_copy(
        update={"deadline": (datetime.now(UTC) + timedelta(seconds=2)).isoformat()}
    )
    await application.register_collaboration_wait(wait, context=resolver.context)
    await asyncio.sleep(2.1)
    expired = await application.cancel_collaboration_wait(wait, context=resolver.context)
    assert expired.state == "expired"
    assert expired.terminal_at_ms is not None


async def test_wait_election_reopens_from_sqlite_without_source_reacquisition(tmp_path):
    path = tmp_path / "collaboration-wait-reopen.sqlite"
    store = SQLiteCollaborationStore(path)
    application, resolver, values = await public_setup(store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1])
    await application.register_collaboration_wait(wait, context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("reopen-wait-cancel"),
            expected=source.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    elected = await application.observe_collaboration_wait(wait, context=resolver.context)
    lookup = await application.lookup_collaboration_wait(wait, context=resolver.context)
    registration = application._participant_coordinator._registration
    requests = application._request_coordinator._registration
    await store.close()
    reopened_store = SQLiteCollaborationStore(path)
    reopened = collaboration_app(
        reopened_store,
        registration,
        collaboration_requests=requests,
    )
    try:
        await reopened.initialize_collaboration()
        assert await reopened.lookup_collaboration_wait(wait, context=resolver.context) == lookup
        assert await reopened.inspect_collaboration_wait(wait, context=resolver.context) == elected
    finally:
        await reopened_store.close()


async def test_wait_observes_source_terminal_after_pending_read(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1], predicate="ALL_SETTLED")
    await application.register_collaboration_wait(wait, context=resolver.context)
    pending = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert pending.state == "pending"
    assert len(pending.evidence) == 1
    assert await application.observe_collaboration_wait(wait, context=resolver.context) == pending
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("later-source-cancel"),
            expected=receipt.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    elected = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert elected.state == "elected"
    assert elected.election.result == "settled"
    assert len(elected.evidence) == 1
    assert elected.evidence[0].status == "settled"
    assert not elected.source_pins


async def test_wait_refuses_premature_expiry_without_mutation(stores):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1])
    registered = await application.register_collaboration_wait(wait, context=resolver.context)
    with pytest.raises(CollaborationConflict):
        await application.cancel_collaboration_wait(wait, context=resolver.context, expired=True)
    assert (
        await application.inspect_collaboration_wait(wait, context=resolver.context) == registered
    )


async def test_wait_receiver_binds_complete_latch_to_retained_ticket(stores):
    """Receiver-boundary test, not qualification of real session publication."""
    from cayu.collaboration._wait_coordinator import _elected_latch
    from cayu.collaboration.waits import request_object_ref
    from cayu.runtime._session_continuation import (
        ContinuationConflict,
        ContinuationNamespace,
        ContinuationTicket,
        continuation_namespace_id,
    )

    store = stores()
    application, resolver, values = await public_setup(store)
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(receipt, values[1])
    ticket = ContinuationTicket(
        namespace=ContinuationNamespace(
            session_id="destination",
            session_instance_id="instance",
            owner=values[1].owner,
            namespace_id=continuation_namespace_id("destination", "instance", values[1].owner),
            generation=1,
        ),
        session_id="destination",
        session_instance_id="instance",
        owner=values[1].owner,
        registration_key="wait-ticket",
        targets=(request_object_ref(receipt.expected.intent.selection.reference),),
        predicate_kind=wait.predicate,
        predicate_version=1,
        deadline=wait.deadline,
        failure_policy=wait.failure_policy,
        service_policy=wait.service_policy,
        wait_edge_revision=1,
        interaction_id="interaction",
        writer_generation=1,
        purpose="receive review outcome",
        state="ARMING",
        revision=1,
    )
    wait = wait.model_copy(update={"delivery_ticket": ticket})
    await application.register_collaboration_wait(wait, context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("cancel-for-latch"),
            expected=receipt.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    elected = await application.observe_collaboration_wait(wait, context=resolver.context)
    with pytest.raises(CollaborationConflict):
        await store.release_wait_sources(values[1], wait, redactor=SecretRedactor())
    receiver = application.collaboration_wait_latch_receiver()
    latch = _elected_latch(elected, SecretRedactor())
    assert await receiver.authenticate_continuation_latch(latch) == latch
    for changed in (
        {"ticket": ticket.model_copy(update={"interaction_id": "other-interaction"})},
        {"ticket": ticket.model_copy(update={"writer_generation": 2})},
        {"latch_key": "different-key"},
        {"accepted_at": "2030-01-01T00:00:00+00:00"},
        {"wait_operation": wait.operation.model_copy(update={"application_scope": "other-scope"})},
    ):
        with pytest.raises((ContinuationConflict, PermissionError)):
            await receiver.authenticate_continuation_latch(latch.model_copy(update=changed))
    assert await application.inspect_collaboration_wait(wait, context=resolver.context) == elected

    # A valid digest alone cannot authenticate a contradictory retained election.
    from cayu.collaboration.waits import election_digest, wait_operation_key

    election = elected.election
    assert election is not None
    foreign = receipt.expected.intent.selection.reference.model_copy(
        update={"request_id": "foreign-request"}
    )
    for changed in (
        {"selected": (foreign,)},
        {"evidence": ()},
        {"result": "success"},
        {"sequence": election.sequence + 1},
        {"elected_at_ms": 1},
    ):
        forged = election.model_copy(update=changed)
        forged = forged.model_copy(
            update={
                "outcome_digest": election_digest(
                    forged.result,
                    forged.selected,
                    forged.evidence,
                    forged.sequence,
                    forged.elected_at_ms,
                )
            }
        )
        async with store._transaction(values[1].binding.application_scope, write=True) as tx:
            await tx.put(
                "operations",
                wait_operation_key(wait),
                elected.model_copy(update={"election": forged}),
                insert=False,
            )
        for read in (
            application.inspect_collaboration_wait,
            application.observe_collaboration_wait,
        ):
            with pytest.raises(ValueError):
                await read(wait, context=resolver.context)
        with pytest.raises(ValueError):
            await receiver.authenticate_continuation_latch(latch)
    async with store._transaction(values[1].binding.application_scope, write=True) as tx:
        await tx.put("operations", wait_operation_key(wait), elected, insert=False)
    assert await receiver.authenticate_continuation_latch(latch) == latch


async def test_wait_delivery_snapshot_requires_positive_pending_responsibility(stores):
    application, resolver, values = await public_setup(stores())
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1], predicate="ANY_SUCCESS")
    await application.register_collaboration_wait(wait, context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("pending-delivery-invariant"),
            expected=source.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    elected = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert elected.state == "elected"
    assert elected.delivery == "none"
    from cayu.collaboration.waits import WaitSnapshot

    malformed = elected.model_dump(mode="json")
    malformed.update({"delivery": "pending", "source_pins": []})
    with pytest.raises(ValueError, match="Pending wait delivery"):
        WaitSnapshot.model_validate(malformed)

    malformed = elected.model_dump(mode="json")
    malformed.update(
        {
            "state": "cancelled",
            "election": None,
            "delivery": "accepted",
            "delivery_receipt_digest": "0" * 64,
            "terminal_at_ms": 1,
        }
    )
    with pytest.raises(ValueError, match="Accepted wait delivery"):
        WaitSnapshot.model_validate(malformed)

    malformed = elected.model_dump(mode="json")
    malformed.update({"source_pins": list(elected.registration.wait.source_keys)})
    with pytest.raises(ValueError, match="Terminal wait without delivery"):
        WaitSnapshot.model_validate(malformed)


@pytest.mark.parametrize("settlement", ["accepted", "excluded"])
async def test_wait_delivery_uses_real_session_continuation_owner(stores, settlement):
    """The public wait handoff must reach the real session owner and receiver."""
    collaboration_store = stores()
    application, resolver, values = await public_setup(collaboration_store)
    source = await application.accept_collaboration_request(values[4], context=resolver.context)
    wait = wait_for(source, values[1], predicate="ANY_SUCCESS")

    session_store = InMemorySessionStore()
    session_app = CayuApp(session_store=session_store, enable_logging=False)
    provider = ScriptedModelProvider(
        (ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})),
        name="wait-destination-provider",
    )
    session_app.register_provider(provider, default=True)
    session_app.register_agent(AgentSpec(name="assistant", model="wait-destination-model"))
    admitted = await create_admitted_session(
        session_store,
        app=session_app,
        request=RunRequest(
            agent_name="assistant",
            session_id="wait-destination",
            messages=[Message.text("user", "prepare a wait")],
        ),
        provider_name=provider.name,
        model="wait-destination-model",
    )
    interaction = admitted.active_invocation_profile.interaction_id
    session_owner = OwnerRef(
        application_scope="tests",
        owner_id=admitted.session.agent_name,
        incarnation=admitted.session.instance_id,
    )
    ticket = ContinuationTicket(
        namespace=ContinuationNamespace(
            session_id=admitted.session.id,
            session_instance_id=admitted.session.instance_id,
            owner=session_owner,
            namespace_id=continuation_namespace_id(
                admitted.session.id, admitted.session.instance_id, session_owner
            ),
            generation=1,
        ),
        session_id=admitted.session.id,
        session_instance_id=admitted.session.instance_id,
        owner=session_owner,
        registration_key="wait-delivery",
        targets=(request_object_ref(source.expected.intent.selection.reference),),
        predicate_kind="ANY_SUCCESS",
        predicate_version=1,
        deadline=wait.deadline,
        failure_policy=wait.failure_policy,
        service_policy=wait.service_policy,
        wait_edge_revision=1,
        interaction_id=interaction,
        writer_generation=admitted.session.run_epoch,
        purpose="receive collaboration result",
        state="ARMING",
        revision=1,
    )
    invocation = await _context(session_store, admitted.session.id, app=session_app)
    receiver = application.collaboration_wait_latch_receiver()
    owner = SessionContinuationOwner(
        store=session_store,
        owner=session_owner,
        receiver=receiver,
        receiver_capability=CapabilityDescriptor(
            owner=values[1].owner,
            mutations=(),
            readbacks=(LATCH_FAMILY,),
        ),
        redactor=SecretRedactor(),
    )
    intent = ContinuationWait.model_validate(
        ticket.model_dump(include=set(ContinuationWait.model_fields))
    )
    prepared = await owner.prepare(intent, invocation=invocation)
    waiting = await owner.park(prepared.ticket, invocation=invocation)
    wait = wait.model_copy(update={"delivery_ticket": waiting.ticket})
    await application.register_collaboration_wait(wait, context=resolver.context)
    if settlement == "accepted":
        await application.control_collaboration_request(
            RequestControl(
                operation=values[1].operation("wait-delivery-cancel"),
                expected=source.expected,
                expected_revision=1,
                kind="cancel",
            ),
            context=resolver.context,
        )
        elected = await application.observe_collaboration_wait(wait, context=resolver.context)
        assert elected.delivery == "pending"
        delivered = await application.deliver_collaboration_wait(
            wait,
            context=resolver.context,
            continuation_owner=owner,
        )
        assert delivered.delivery == "accepted"
        assert delivered.delivery_receipt_digest is not None
        assert (
            await application.deliver_collaboration_wait(
                wait,
                context=resolver.context,
                continuation_owner=owner,
            )
            == delivered
        )
    else:
        cancelled = await application.cancel_collaboration_wait(wait, context=resolver.context)
        assert cancelled.state == "cancelled"
        delivered = await application.exclude_collaboration_wait(
            wait,
            context=resolver.context,
            continuation_owner=owner,
            invocation=invocation,
        )
        assert delivered.delivery == "excluded"
        assert (
            await application.exclude_collaboration_wait(
                wait,
                context=resolver.context,
                continuation_owner=owner,
                invocation=invocation,
            )
            == delivered
        )
    retained = await session_store.load_continuation_ticket(
        admitted.session.id,
        registration_key=ticket.registration_key,
        session_instance_id=admitted.session.instance_id,
    )
    assert retained is not None
    if settlement == "accepted":
        assert retained.latch is not None
    else:
        assert retained.ticket.state == "RETIRED"
    replay_wait = wait.model_copy(update={"delivery_ticket": retained.ticket})
    assert (
        await application.inspect_collaboration_wait(replay_wait, context=resolver.context)
        == delivered
    )
    await owner.drain()


@pytest.mark.parametrize(
    ("predicate", "threshold", "expected"),
    [
        ("ALL_SUCCESS", None, "failure"),
        ("ALL_SETTLED", None, "settled"),
        ("ANY_SUCCESS", None, "failure"),
        ("QUORUM_SUCCESS", 1, "failure"),
    ],
)
async def test_wait_predicate_contracts_are_recorded(stores, predicate, threshold, expected):
    application, resolver, values = await public_setup(stores())
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation(f"cancel-{predicate.lower()}"),
            expected=receipt.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    wait = wait_for(receipt, values[1], predicate=predicate, threshold=threshold)
    registered = await application.register_collaboration_wait(wait, context=resolver.context)
    assert registered.registration.wait.predicate == predicate
    assert registered.registration.wait.threshold == threshold
    observed = await application.observe_collaboration_wait(wait, context=resolver.context)
    assert observed.election is not None
    assert observed.election.result == expected
