"""Controlled receiving-owner evidence through real collaboration transactions."""

import asyncio
import json
import warnings
from contextlib import asynccontextmanager

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_participant_lifecycle import change

from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactMatch,
    ExactNotFound,
    InitiatorBinding,
    ObjectRef,
)
from cayu.collaboration._permits import (
    PermitCommand,
    PermitIntent,
    PermitRegistration,
    PermitSettlementReader,
    ReceivingSettlementReceipt,
)
from cayu.collaboration.lifecycle import NamespaceRetire, NamespaceRotate
from cayu.collaboration.participants import CollaborationCapacityExceeded, CollaborationUnavailable
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
stores = identity_tests.stores
REDACTOR = SecretRedactor()


def permit(initialized, participant, *, key="permit", revision=1, generation=1):
    request = PermitRegistration(
        operation=initialized.operation(key),
        participant=participant,
        expected_lifecycle_revision=revision,
        admission_generation=generation,
        source_operation=initialized.operation(f"source-{key}"),
        target=ObjectRef(
            owner=initialized.owner, kind="fixture", object_id=f"target-{key}", incarnation="one"
        ),
        target_state="future",
        effect_scope="execute",
        required_settlement="quiescence",
        settlement_operation=initialized.operation(f"settle-{key}"),
    )
    return PermitCommand(
        operation=request.operation,
        source=initialized.owner,
        destination=initialized.owner,
        initiator=InitiatorBinding(
            issuer=initialized.owner,
            principal="admission-owner",
            participant=None,
            mandate=None,
            invocation_id=None,
            interaction_id=None,
        ),
        intent=PermitIntent(request=request, limits=initialized.binding.limits),
    )


class Receiver(PermitSettlementReader):
    def __init__(self, expected):
        self.expected = expected
        self.calls = 0
        self.available = True
        self.entered = asyncio.Event()
        self.release = None
        self.outcome = "quiescent"
        self.admission_excluded = False

    @property
    def owner(self):
        return self.expected.intent.request.target.owner

    async def lookup(self, expected):
        assert expected == self.expected
        self.calls += 1
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if not self.available:
            return ExactNotFound()
        return ExactMatch[ReceivingSettlementReceipt](
            receipt=ReceivingSettlementReceipt(
                expected=expected,
                receiving_owner=self.owner,
                receipt_id="receiver-receipt",
                outcome=self.outcome,
                admission_excluded=self.admission_excluded,
            )
        )


@pytest.mark.parametrize("registration_wins", [False, True])
@pytest.mark.parametrize("required_settlement", ["exclusion", "quiescence"])
async def test_receiving_exclusion_fences_or_settles_concurrent_registration(
    stores, registration_wins, required_settlement
):
    from cayu.collaboration._permits import (
        PermitExclusion,
        PermitSettlement,
        RetiredPermitExclusion,
    )
    from cayu.collaboration.lifecycle import CollaborationNamespaceRetired, NamespacePrune

    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    original = permit(initialized, ref)
    expected = original.model_copy(
        update={
            "intent": original.intent.model_copy(
                update={
                    "request": original.intent.request.model_copy(
                        update={"required_settlement": required_settlement}
                    )
                }
            )
        }
    )
    reader = Receiver(expected)
    if required_settlement == "quiescence":
        # Quiescence without permanent admission exclusion cannot create a
        # negative-admission tombstone, even from an authenticated receiver.
        with pytest.raises(CollaborationUnavailable, match="exclusion"):
            await store._exclude_permit(initialized, expected, reader=reader, redactor=REDACTOR)
        reader.admission_excluded = True
        reader.entered.clear()
    else:
        reader.outcome = "excluded"
    reader.release = asyncio.Event()
    exclusion = asyncio.create_task(
        store._exclude_permit(initialized, expected, reader=reader, redactor=REDACTOR)
    )
    await reader.entered.wait()
    if registration_wins:
        await stores()._register_permit(initialized, expected, redactor=REDACTOR)
    reader.release.set()
    result = await exclusion
    assert isinstance(result, PermitSettlement if registration_wins else PermitExclusion)
    assert result.receiving_receipt.outcome == reader.outcome
    inspected = await application.inspect_participant(ref, context=CONTEXT)
    assert inspected.outstanding_obligations == 0
    assert inspected.issued_permit_frontier == int(registration_wins)
    reopened = stores()
    assert (
        await reopened._exclude_permit(initialized, expected, reader=reader, redactor=REDACTOR)
        == result
    )
    if not registration_wins:
        with pytest.raises(CollaborationConflict):
            await reopened._register_permit(initialized, expected, redactor=REDACTOR)
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    rotated = await application.rotate_collaboration_namespace(
        NamespaceRotate(
            operation=before.current.reference.operation("rotate"),
            namespace=before.current.reference,
            expected_revision=before.current.revision,
        ),
        context=CONTEXT,
    )
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    retained = await application.inspect_collaboration_namespace(context=CONTEXT)
    pruned = await application.prune_collaboration_namespace(
        NamespacePrune(
            operation=rotated.successor.reference.operation("prune"),
            namespace=rotated.namespace.reference,
            expected_retention_revision=retained.retention_revision,
            max_records=32,
        ),
        context=CONTEXT,
    )
    assert pruned.complete
    rejected = await reopened._exclude_permit(
        initialized, expected, reader=reader, redactor=REDACTOR
    )
    assert isinstance(rejected, RetiredPermitExclusion)
    assert rejected.expected == expected
    with pytest.raises(CollaborationNamespaceRetired):
        await reopened._register_permit(initialized, expected, redactor=REDACTOR)


async def test_disable_is_accepted_before_settlement_and_reenable_does_not_revive_permits(stores):
    store = stores()
    reg = registration()
    application = app(store, reg)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    admitted = await store._register_permit(initialized, expected, redactor=REDACTOR)
    request = change(initialized, ref, key="disable", revision=1, state="disabled")
    disabled = await application.change_participant_lifecycle(request, context=CONTEXT)
    assert disabled.participant.covered_permit_frontier == admitted.position == 1
    inspected = await application.inspect_participant(ref, context=CONTEXT)
    assert inspected.participant.lifecycle == "disabled"
    assert inspected.settlement == "unsettled" and inspected.outstanding_obligations == 1
    with pytest.raises(CollaborationConflict):
        await application.change_participant_lifecycle(
            change(
                initialized,
                ref,
                key="retire",
                revision=2,
                state="retired",
            ),
            context=CONTEXT,
        )
    with pytest.raises(CollaborationConflict):
        await store._register_permit(
            initialized, permit(initialized, ref, key="late"), redactor=REDACTOR
        )
    enabled = await application.change_participant_lifecycle(
        change(
            initialized,
            ref,
            key="enable",
            revision=2,
            state="active",
        ),
        context=CONTEXT,
    )
    assert enabled.participant.admission_generation == 2
    with pytest.raises(CollaborationConflict):
        await store._register_permit(
            initialized, permit(initialized, ref, key="stale"), redactor=REDACTOR
        )
    assert await store._register_permit(initialized, expected, redactor=REDACTOR) == admitted
    reader = Receiver(expected)
    reader.available = False
    with pytest.raises(CollaborationUnavailable):
        await store._settle_permit(initialized, expected, reader=reader, redactor=REDACTOR)
    reconstructed = app(stores(), reg)
    await reconstructed.initialize_collaboration()
    assert (
        await reconstructed.inspect_participant(ref, context=CONTEXT)
    ).outstanding_obligations == 1
    reader.available = True
    settled = await store._settle_permit(initialized, expected, reader=reader, redactor=REDACTOR)
    assert settled.receiving_receipt.outcome == "quiescent"
    assert (await reconstructed.inspect_participant(ref, context=CONTEXT)).settlement == "settled"
    assert await reconstructed.change_participant_lifecycle(request, context=CONTEXT) == disabled
    calls = reader.calls
    assert (
        await stores()._settle_permit(initialized, expected, reader=reader, redactor=REDACTOR)
        == settled
    )
    assert reader.calls == calls


async def test_unsettled_old_generation_cannot_be_skipped_and_settlement_survives_sealing(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    expected = permit(initialized, created.participants[0].reference)
    await store._register_permit(initialized, expected, redactor=REDACTOR)
    before = await application.inspect_collaboration_namespace(context=CONTEXT)
    rotated = await application.rotate_collaboration_namespace(
        NamespaceRotate(
            operation=before.current.reference.operation("rotate"),
            namespace=before.current.reference,
            expected_revision=before.current.revision,
        ),
        context=CONTEXT,
    )
    assert rotated.namespace.outstanding_obligations == 1
    request = NamespaceRetire(
        operation=rotated.successor.reference.operation("retire"),
        namespace=rotated.namespace.reference,
        expected_revision=rotated.namespace.revision,
        expected_retired_through=0,
    )
    with pytest.raises(CollaborationConflict):
        await application.retire_collaboration_namespace(request, context=CONTEXT)
    await store._settle_permit(initialized, expected, reader=Receiver(expected), redactor=REDACTOR)
    retired = await application.retire_collaboration_namespace(request, context=CONTEXT)
    assert retired.namespace.outstanding_obligations == 0


@pytest.mark.parametrize("winner", ["permit", "disable"])
async def test_registration_and_disable_share_exact_transaction_order(stores, monkeypatch, winner):
    first, second = stores(), stores()
    reg = registration()
    application = app(first, reg)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    request = change(initialized, ref, key="disable", revision=1, state="disabled")
    owner = second if winner == "permit" else first
    original = owner._transaction
    entered, release = asyncio.Event(), asyncio.Event()
    injected = False

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal injected
        async with original(scope, write=write) as tx:
            hold = write and not injected
            if hold:
                injected = True
            yield tx
            if hold:
                entered.set()
                await release.wait()

    monkeypatch.setattr(owner, "_transaction", transaction)
    operations = {
        "permit": lambda: second._register_permit(initialized, expected, redactor=REDACTOR),
        "disable": lambda: application.change_participant_lifecycle(request, context=CONTEXT),
    }
    winning = asyncio.create_task(operations[winner]())
    losing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        loser = "disable" if winner == "permit" else "permit"
        losing = asyncio.create_task(operations[loser]())
        await asyncio.sleep(0)
        assert not losing.done()
        release.set()
        result = await winning
        if winner == "disable":
            with pytest.raises(CollaborationConflict):
                await losing
            closed = result
        else:
            closed = await losing
    finally:
        release.set()
        await asyncio.gather(
            winning, *(() if losing is None else (losing,)), return_exceptions=True
        )
    assert closed.participant.covered_permit_frontier == int(winner == "permit")
    assert (
        await application.inspect_participant(ref, context=CONTEXT)
    ).outstanding_obligations == int(winner == "permit")


@pytest.mark.parametrize("loss", ["cancellation", "timeout"])
async def test_receiving_lookup_is_outside_transaction_and_survives_cancelled_observer(
    stores, loss
):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    await store._register_permit(initialized, expected, redactor=REDACTOR)
    reader = Receiver(expected)
    reader.release = asyncio.Event()
    if loss == "timeout":
        store._owners.observation_timeout = 0.1
    caller = asyncio.create_task(
        store._settle_permit(initialized, expected, reader=reader, redactor=REDACTOR)
    )
    retry = None
    try:
        await asyncio.wait_for(reader.entered.wait(), 5)
        if loss == "cancellation":
            caller.cancel()
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 2
        else:
            with pytest.raises(CollaborationUnavailable):
                await caller
            assert not caller.cancelled() and caller.cancelling() == 0
            store._owners.observation_timeout = 10
        assert len(store._owners.pending) == 1
        closed = await asyncio.wait_for(
            application.change_participant_lifecycle(
                change(
                    initialized,
                    ref,
                    key="disable",
                    revision=1,
                    state="disabled",
                ),
                context=CONTEXT,
            ),
            5,
        )
        assert closed.participant.covered_permit_frontier == 1
        assert (
            await application.inspect_participant(ref, context=CONTEXT)
        ).settlement == "unsettled"
        retry = asyncio.create_task(
            store._settle_permit(initialized, expected, reader=reader, redactor=REDACTOR)
        )
        await asyncio.sleep(0)
        reader.release.set()
        await retry
        assert reader.calls == 1
        assert (await application.inspect_participant(ref, context=CONTEXT)).settlement == "settled"
    finally:
        reader.release.set()
        await asyncio.gather(caller, *(() if retry is None else (retry,)), return_exceptions=True)


@pytest.mark.parametrize(
    "missing", ["issued_permit_frontier", "outstanding_obligations", "settlement", "all"]
)
async def test_public_inspection_requires_explicit_permit_settlement_evidence(
    stores, monkeypatch, missing
):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    await store._register_permit(initialized, permit(initialized, ref), redactor=REDACTOR)
    complete = await application.inspect_participant(ref, context=CONTEXT)
    assert complete.outstanding_obligations == 1 and complete.settlement == "unsettled"
    # A current active participant has no disabled frontier yet, so defaulting an
    # incomplete wrapper response to zeros would incorrectly prove settlement.
    partial = complete.model_dump(mode="json")
    for field in (
        ("issued_permit_frontier", "outstanding_obligations", "settlement")
        if missing == "all"
        else (missing,)
    ):
        partial.pop(field)

    async def incomplete(*args, **kwargs):
        return partial

    monkeypatch.setattr(store, "inspect", incomplete)
    with pytest.raises(CollaborationContractError):
        await application.inspect_participant(ref, context=CONTEXT)


@pytest.mark.parametrize(
    "field",
    [
        "expected_lifecycle_revision",
        "admission_generation",
        "source_operation",
        "target",
        "target_state",
        "effect_scope",
        "required_settlement",
        "settlement_operation",
        "initiator",
        "limits",
    ],
)
async def test_fixed_permit_key_compares_complete_authority_without_mutation(stores, field):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    receipt = await store._register_permit(initialized, expected, redactor=REDACTOR)
    before = await application.inspect_participant(ref, context=CONTEXT)
    events = await application.list_participant_events(context=CONTEXT)
    request = expected.intent.request
    replacements = {
        "expected_lifecycle_revision": 2,
        "admission_generation": 2,
        "source_operation": initialized.operation("another-source"),
        "target": request.target.model_copy(update={"incarnation": "another-incarnation"}),
        "target_state": "existing",
        "effect_scope": "another-effect",
        "required_settlement": "exclusion",
        "settlement_operation": initialized.operation("another-settlement"),
    }
    if field == "initiator":
        changed = expected.model_copy(
            update={
                "initiator": expected.initiator.model_copy(update={"principal": "another-actor"}),
            }
        )
    elif field == "limits":
        changed = expected.model_copy(
            update={
                "intent": expected.intent.model_copy(
                    update={
                        "limits": expected.intent.limits.model_copy(update={"obligations": 32}),
                    }
                )
            }
        )
    else:
        changed = expected.model_copy(
            update={
                "intent": expected.intent.model_copy(
                    update={
                        "request": request.model_copy(update={field: replacements[field]}),
                    }
                )
            }
        )
    with pytest.raises(CollaborationConflict):
        await store._register_permit(initialized, changed, redactor=REDACTOR)
    assert await store._register_permit(initialized, expected, redactor=REDACTOR) == receipt
    assert await application.inspect_participant(ref, context=CONTEXT) == before
    assert await application.list_participant_events(context=CONTEXT) == events


@pytest.mark.parametrize("boundary", ["event", "escaping", "escaped_event", "admitted"])
async def test_admission_covers_complete_escaped_settlement_envelopes(stores, boundary):
    from cayu.collaboration._contracts import MAX_ENVELOPE_BYTES, MAX_ID_BYTES, OwnerRef
    from cayu.collaboration._permit_store import prepare_permit
    from cayu.collaboration._permits import PermitSnapshot
    from cayu.collaboration._preparation import prepare_contract

    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference

    def candidate(length):
        text = "\x01" * length
        target = ObjectRef(
            owner=OwnerRef(application_scope=text, owner_id=text, incarnation=text),
            kind="fixture",
            object_id=text,
            incarnation=text,
        )
        base = permit(initialized, ref)
        return PermitCommand.model_validate(
            base.model_copy(
                update={
                    "initiator": base.initiator.model_copy(
                        update={
                            "principal": text,
                            "participant": target,
                            "mandate": target,
                            "invocation_id": text,
                            "interaction_id": text,
                        }
                    ),
                    "intent": base.intent.model_copy(
                        update={
                            "request": base.intent.request.model_copy(update={"target": target}),
                        }
                    ),
                }
            )
        )

    def representations(expected, receipt_id):
        command = expected.model_dump(mode="json")
        receiving = {
            "expected": command,
            "receiving_owner": command["intent"]["request"]["target"]["owner"],
            "receipt_id": receipt_id,
            "outcome": "quiescent",
            "admission_excluded": False,
        }
        snapshot = {
            "expected": command,
            "position": 2**53 - 1,
            "state": "settled",
            "settlement": receiving,
        }
        receipt = {
            "record_type": "permit_settled",
            "expected": command,
            "receiving_receipt": receiving,
            "event": {
                "id": "f" * 32,
                "sequence": 2**53 - 1,
                "operation": command["intent"]["request"]["settlement_operation"],
                "type": "permit_settled",
                "participants": [ref.model_dump(mode="json")],
            },
        }
        return snapshot, receipt

    def size(value):
        return len(
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        )

    # Pick the largest command accepted by each claimed bound, independently of
    # the production preflight. All identifiers stay within their byte limits.
    receipt_id = (
        "\x01" * MAX_ID_BYTES if boundary in ("escaped_event", "admitted") else "r" * MAX_ID_BYTES
    )
    index = 0 if boundary in ("event", "escaped_event") else 1
    low, high = 1, MAX_ID_BYTES
    while low < high:
        middle = (low + high + 1) // 2
        if size(representations(candidate(middle), receipt_id)[index]) <= MAX_ENVELOPE_BYTES:
            low = middle
        else:
            high = middle - 1
    expected = candidate(low)
    plain_snapshot, plain_receipt = representations(expected, "r" * MAX_ID_BYTES)
    escaped_snapshot, escaped_receipt = representations(expected, "\x01" * MAX_ID_BYTES)
    # This is exactly the old admission probe, and it accepts both rejected cases.
    prepare_contract(PermitSnapshot, plain_snapshot, redactor=REDACTOR)
    before = await application.inspect_participant(ref, context=CONTEXT)
    events = await application.list_participant_events(context=CONTEXT)
    if boundary != "admitted":
        if boundary == "event":
            assert size(plain_receipt) > MAX_ENVELOPE_BYTES
        elif boundary == "escaped_event":
            assert size(escaped_snapshot) <= MAX_ENVELOPE_BYTES < size(escaped_receipt)
        else:
            assert size(plain_receipt) <= MAX_ENVELOPE_BYTES < size(escaped_receipt)
        with pytest.raises(CollaborationContractError):
            await store._register_permit(initialized, expected, redactor=REDACTOR)
        assert await application.inspect_participant(ref, context=CONTEXT) == before
        assert await application.list_participant_events(context=CONTEXT) == events
        assert not (
            await application.list_participant_obligations(ref, context=CONTEXT)
        ).obligations
        # Rejection did not reserve either key or charge capacity.
        await store._register_permit(initialized, permit(initialized, ref), redactor=REDACTOR)
        return

    assert max(size(escaped_snapshot), size(escaped_receipt)) <= MAX_ENVELOPE_BYTES
    assert size(escaped_receipt) > MAX_ENVELOPE_BYTES - 512
    assert prepare_permit(initialized, expected, REDACTOR) == expected
    registered = await store._register_permit(initialized, expected, redactor=REDACTOR)

    class EscapedReceiver(Receiver):
        async def lookup(self, expected):
            return ExactMatch[ReceivingSettlementReceipt](
                receipt=ReceivingSettlementReceipt(
                    expected=expected,
                    receiving_owner=self.owner,
                    receipt_id="\x01" * MAX_ID_BYTES,
                    outcome="quiescent",
                )
            )

    settled = await store._settle_permit(
        initialized, expected, reader=EscapedReceiver(expected), redactor=REDACTOR
    )
    assert settled.receiving_receipt.receipt_id == "\x01" * MAX_ID_BYTES
    assert (await application.inspect_participant(ref, context=CONTEXT)).settlement == "settled"
    assert await stores()._register_permit(initialized, expected, redactor=REDACTOR) == registered
    assert (
        await stores()._settle_permit(
            initialized, expected, reader=EscapedReceiver(expected), redactor=REDACTOR
        )
        == settled
    )


async def test_optional_capacity_cannot_consume_reserved_settlement(stores):
    from cayu.collaboration._capacity import PERMIT_SETTLEMENT_BYTES
    from cayu.collaboration._preparation import prepare_contract
    from cayu.collaboration.base import _Anchor

    store = stores()
    limits = registration().bootstrap.limits.model_copy(
        update={
            "operations": 8,
            "control_operations": 2,
            "events": 8,
            "control_events": 2,
        }
    )
    application = app(store, registration(limits=limits))
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    admitted = await store._register_permit(initialized, expected, redactor=REDACTOR)

    async def anchor():
        async with store._transaction(initialized.binding.application_scope, write=False) as tx:
            return prepare_contract(_Anchor, await tx.get("anchors", ()), redactor=REDACTOR)

    before = await anchor()
    assert before.reserved_bytes == PERMIT_SETTLEMENT_BYTES and before.reserved_events == 1
    assert await store._register_permit(initialized, expected, redactor=REDACTOR) == admitted
    assert await anchor() == before
    for number in range(10):
        try:
            await create(application, initialized, f"extra-{number}")
        except CollaborationCapacityExceeded:
            break
    else:
        pytest.fail("Ordinary admission did not enforce reserved capacity")
    await application.change_participant_lifecycle(
        change(
            initialized,
            ref,
            key="disable",
            revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    await store._settle_permit(initialized, expected, reader=Receiver(expected), redactor=REDACTOR)
    after = await anchor()
    assert after.reserved_bytes == after.reserved_events == 0
    assert after.event_count <= limits.events and after.operation_count <= limits.operations
    assert after.retained_bytes <= limits.retained_bytes
    assert (await application.inspect_participant(ref, context=CONTEXT)).settlement == "settled"


async def test_raw_receiving_receipt_is_not_an_authenticated_reader(stores):
    store = stores()
    application = app(store, registration())
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    expected = permit(initialized, created.participants[0].reference)
    await store._register_permit(initialized, expected, redactor=REDACTOR)
    trusted = Receiver(expected)
    raw = (await trusted.lookup(expected)).receipt
    with pytest.raises(CollaborationConflict):
        await store._settle_permit(initialized, expected, reader=raw, redactor=REDACTOR)
    assert (
        await application.inspect_participant(created.participants[0].reference, context=CONTEXT)
    ).outstanding_obligations == 1
    await store._settle_permit(initialized, expected, reader=trusted, redactor=REDACTOR)


@pytest.mark.parametrize(
    "corruption", ["different_effect", "wrong_owner", "excluded", "secret_id", "malformed_id"]
)
async def test_receiving_evidence_is_revalidated_before_settlement(
    stores,
    corruption,
    caplog,
    capsys,
):
    secret = "receiver-secret-canary"
    redactor = SecretRedactor(secret)
    store = stores()
    application = app(store, registration(), secret_redactor=redactor)
    initialized = await application.initialize_collaboration()
    _, created = await create(application, initialized)
    ref = created.participants[0].reference
    expected = permit(initialized, ref)
    await store._register_permit(initialized, expected, redactor=redactor)
    before = await application.inspect_participant(ref, context=CONTEXT)
    events = await application.list_participant_events(context=CONTEXT)
    valid = (await Receiver(expected).lookup(expected)).receipt

    class Hostile:
        def __repr__(self):
            return secret

        def __str__(self):
            return secret

    values = {
        "wrong_owner": {
            "receiving_owner": expected.source.model_copy(update={"incarnation": "wrong"})
        },
        "excluded": {"outcome": "excluded"},
        "secret_id": {"receipt_id": secret},
        "malformed_id": {"receipt_id": Hostile()},
        "different_effect": {
            "expected": expected.model_copy(
                update={
                    "intent": expected.intent.model_copy(
                        update={
                            "request": expected.intent.request.model_copy(
                                update={"effect_scope": "another"}
                            )
                        }
                    ),
                }
            )
        },
    }
    forged = valid.model_copy(update=values[corruption])

    class IncompleteReceiver(Receiver):
        async def lookup(self, expected):
            return ExactMatch[ReceivingSettlementReceipt].model_construct(receipt=forged)

    with (
        warnings.catch_warnings(record=True) as recorded,
        pytest.raises((CollaborationConflict, CollaborationContractError)) as caught,
    ):
        await store._settle_permit(
            initialized, expected, reader=IncompleteReceiver(expected), redactor=redactor
        )
    output = capsys.readouterr()
    assert not recorded
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert (
        secret not in str(caught.value) + repr(caught.value) + caplog.text + output.out + output.err
    )
    assert await application.inspect_participant(ref, context=CONTEXT) == before
    assert await application.list_participant_events(context=CONTEXT) == events
    await store._settle_permit(initialized, expected, reader=Receiver(expected), redactor=redactor)
    assert (await application.inspect_participant(ref, context=CONTEXT)).settlement == "settled"
