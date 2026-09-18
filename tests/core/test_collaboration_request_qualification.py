"""Public request admission races, complete expectations and capacity boundaries."""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_request_foundation import REDACTOR, public_setup
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_participant_lifecycle import change

from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactMatch,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration.participants import (
    CollaborationCapacityExceeded,
    ParticipantAliasChange,
)
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import RequestCommand, RequestControl, RequestControlCommand

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


async def anchor(store, initialized):
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        return await store._anchor(tx, initialized, REDACTOR)


@pytest.mark.parametrize("mutation", ["disable", "alias"])
@pytest.mark.parametrize("accepted_first", [False, True])
async def test_public_acceptance_serializes_with_participant_mutation(
    stores, monkeypatch, mutation, accepted_first
):
    store = stores()
    application, resolver, values = await public_setup(store)
    initialized, recipient, request = values[1], values[3], values[4]
    _, replacement = await create(application, initialized, "replacement")
    original = store._transaction
    entered, release = asyncio.Event(), asyncio.Event()
    armed = True

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal armed
        intercept = write and armed
        if intercept:
            armed = False
        if intercept and not accepted_first:
            entered.set()
            await release.wait()
        async with original(scope, write=write) as tx:
            yield tx
        if intercept and accepted_first:
            entered.set()
            await release.wait()

    monkeypatch.setattr(store, "_transaction", transaction)
    caller = asyncio.create_task(
        application.accept_collaboration_request(request, context=resolver.context)
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if mutation == "disable":
            await application.change_participant_lifecycle(
                change(
                    initialized, recipient.reference, key="disable", revision=1, state="disabled"
                ),
                context=CONTEXT,
            )
        else:
            await application.change_participant_alias(
                ParticipantAliasChange(
                    operation=initialized.operation("rebind"),
                    alias="reviewer",
                    expected_alias_revision=1,
                    expected_target=recipient.reference,
                    target=replacement.participants[0].reference,
                ),
                context=CONTEXT,
            )
        after_mutation = await anchor(store, initialized)
        release.set()
        if accepted_first:
            receipt = await asyncio.wait_for(caller, 5)
            assert receipt.expected.intent.selection.recipient == recipient
            assert (
                await application.accept_collaboration_request(request, context=resolver.context)
                == receipt
            )
            assert isinstance(
                await application.lookup_collaboration_request(
                    receipt.expected, context=resolver.context
                ),
                ExactMatch,
            )
        else:
            with pytest.raises(CollaborationConflict):
                await asyncio.wait_for(caller, 5)
        assert await anchor(store, initialized) == after_mutation
        current = await application.inspect_participant(recipient.reference, context=CONTEXT)
        assert current.outstanding_obligations == int(accepted_first)
        assert (
            await application.inspect_participant(
                replacement.participants[0].reference, context=CONTEXT
            )
        ).outstanding_obligations == 0
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await application.drain_collaboration_requests()


def leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    else:
        yield path, value


def changed_leaf(document, path, value):
    result = deepcopy(document)
    parent = result
    for part in path[:-1]:
        parent = parent[part]
    parent[path[-1]] = (
        not value
        if type(value) is bool
        else value + 1
        if type(value) is int
        else value + "-different"
        if type(value) is str
        else "different"
    )
    return result


@pytest.mark.parametrize("family", ["acceptance", "control"])
async def test_public_exact_lookup_checks_every_expected_leaf(stores, family):
    store = stores()
    application, resolver, values = await public_setup(store)
    receipt = await application.accept_collaboration_request(values[4], context=resolver.context)
    if family == "control":
        receipt = await application.control_collaboration_request(
            RequestControl(
                operation=values[1].operation("control"),
                expected=receipt.expected,
                expected_revision=1,
                kind="cancel",
            ),
            context=resolver.context,
        )
    schema = RequestCommand if family == "acceptance" else RequestControlCommand
    original = receipt.expected.model_dump(mode="json")
    before = await anchor(store, values[1])
    conflicts = rejected = 0
    for path, value in leaves(original):
        # The complete operation key remains fixed, including its duplicate in
        # the original intent. Every other serialized operand is varied alone.
        if "operation" in path:
            continue
        altered = changed_leaf(original, path, value)
        assert altered["operation"] == original["operation"]
        try:
            expected = prepare_contract(schema, altered, redactor=REDACTOR)
        except CollaborationContractError:
            # Ill-formed authority must also be rejected at the public entrance,
            # including post-construction mutation of an otherwise typed value.
            expected = schema.model_construct(**altered)
            with pytest.raises(CollaborationContractError):
                await application.lookup_collaboration_request(expected, context=resolver.context)
            rejected += 1
        else:
            result = await application.lookup_collaboration_request(
                expected, context=resolver.context
            )
            assert isinstance(result, ExactConflict), (path, result)
            conflicts += 1
    assert conflicts > 20 and rejected > 20
    reconstructed = prepare_contract(schema, original, redactor=REDACTOR)
    assert (
        await application.lookup_collaboration_request(reconstructed, context=resolver.context)
    ).receipt == receipt
    assert await anchor(store, values[1]) == before
    await application.drain_collaboration_requests()


async def test_public_original_intent_fields_cannot_reuse_key(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    request = values[4]
    accepted = await application.accept_collaboration_request(request, context=resolver.context)
    ref = request.delivery_contract
    changes = {
        "kind": {"kind": "contribution", "output_contract": None},
        "sender": {"sender": values[3].reference},
        "target": {"target": values[3].reference},
        "content": {"content": "Changed"},
        "inputs": {"inputs": (ref,)},
        "context_hint": {"context_hint": ref},
        "delivery_contract": {"delivery_contract": ref.model_copy(update={"revision": 2})},
        "output_contract": {"output_contract": ref.model_copy(update={"revision": 2})},
        "independence_policy": {"independence_policy": ref.model_copy(update={"revision": 2})},
        "disclosure_policy": {"disclosure_policy": ref.model_copy(update={"revision": 2})},
        "ttl_ms": {"ttl_ms": request.ttl_ms + 1},
        "cancellation": {"cancellation": "stop"},
    }
    assert set(changes) == set(type(request).model_fields) - {"operation"}
    before = await anchor(store, values[1])
    for changed in changes.values():
        with pytest.raises(CollaborationConflict):
            await application.accept_collaboration_request(
                request.model_copy(update=changed), context=resolver.context
            )
    assert (
        await application.accept_collaboration_request(request, context=resolver.context)
        == accepted
    )
    assert await anchor(store, values[1]) == before
    await application.drain_collaboration_requests()


@pytest.mark.parametrize("field", ["issuer", "principal", "participant", "mandate"])
async def test_authorized_reader_cannot_replay_as_original_initiator(stores, field):
    class Policy(identity_tests.Policy):
        def authorize(self, context, **kwargs):
            # Explicitly allow both application-authenticated principal names.
            assert context.principal in ("operator", "reader")
            return super().authorize(context.model_copy(update={"principal": "operator"}), **kwargs)

    store = stores()
    application, resolver, values = await public_setup(store, reg=registration(policy=Policy()))
    request = values[4]
    accepted = await application.accept_collaboration_request(request, context=resolver.context)
    original_context = resolver.context
    replacement = {
        "issuer": original_context.issuer.model_copy(update={"incarnation": "other"}),
        "principal": "reader",
        "participant": values[3].reference,
        "mandate": original_context.mandate.model_copy(update={"revision": 2}),
    }[field]
    context = original_context.model_copy(update={field: replacement})
    if field == "issuer":
        context = context.model_copy(
            update={"mandate": context.mandate.model_copy(update={"owner": context.issuer})}
        )
        resolver._ref = resolver.ref.model_copy(update={"owner": context.issuer})
    resolver.context = context
    leaf = resolver.resolution.chain.entries[-1].model_copy(
        update={
            "issuer": context.issuer,
            "principal": context.principal,
            "participant": context.participant,
            "reference": context.mandate,
            "root": context.mandate,
        }
    )
    resolver.resolution = resolver.resolution.model_copy(
        update={
            "principal": resolver.resolution.principal.model_copy(
                update={
                    "resolver": resolver.ref,
                    "issuer": context.issuer,
                    "principal": context.principal,
                    "participants": (context.participant,),
                }
            ),
            "chain": resolver.resolution.chain.model_copy(update={"entries": (leaf,)}),
        }
    )
    # A different issuer is an explicitly registered replacement, not mutation
    # of the resolver identity pinned by the original application.
    application = app(
        store,
        values[0]._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    await application.initialize_collaboration()
    before = await anchor(store, values[1])
    result = await application.lookup_collaboration_request(accepted.expected, context=context)
    assert isinstance(result, ExactMatch) and result.receipt == accepted
    with pytest.raises(CollaborationConflict):
        await application.accept_collaboration_request(request, context=context)
    assert await anchor(store, values[1]) == before
    await application.drain_collaboration_requests()


@pytest.mark.parametrize("dimension", ["operations", "events", "retained_bytes"])
@pytest.mark.parametrize("kind", ["cancel", "expire"])
async def test_full_ordinary_capacity_preserves_public_control(
    stores, monkeypatch, dimension, kind
):
    limits = registration().bootstrap.limits.model_copy(
        update={dimension: 16 if dimension != "retained_bytes" else 750_000}
    )
    store = stores()
    application, resolver, values = await public_setup(store, reg=registration(limits=limits))
    initialized = values[1]
    receipts = []
    for index in range(32):
        before = await anchor(store, initialized)
        request = values[4].model_copy(
            update={"operation": initialized.operation(f"request-{index}")}
        )
        try:
            receipt = await application.accept_collaboration_request(
                request, context=resolver.context
            )
        except CollaborationCapacityExceeded:
            assert await anchor(store, initialized) == before
            break
        receipts.append(receipt)
    else:
        pytest.fail("Configured ordinary capacity was not enforced")
    assert receipts
    full = await anchor(store, initialized)
    # Pending reservations and retained evidence are each below the bound; the
    # combined admission, including the next request, must still refuse.
    if dimension == "operations":
        assert full.operation_count < limits.operations
        assert full.reserved_operations < limits.operations
    elif dimension == "events":
        assert full.event_count < limits.events
        assert full.reserved_events < limits.events
    else:
        assert full.retained_bytes < limits.retained_bytes
        assert full.reserved_bytes < limits.retained_bytes
    # Consume the remaining ordinary slots with real identity publications.
    for index in range(64):
        before = await anchor(store, initialized)
        try:
            await create(application, initialized, f"fill-{index}")
        except CollaborationCapacityExceeded:
            assert await anchor(store, initialized) == before
            break
    else:
        pytest.fail("Ordinary filler did not reach the admission bound")
    full = await anchor(store, initialized)
    if dimension == "operations":
        assert (
            full.operation_count + full.reserved_operations
            == limits.operations - limits.control_operations
        )
    elif dimension == "events":
        assert full.event_count + full.reserved_events == limits.events - limits.control_events
    if kind == "expire":
        original = store._transaction
        deadline = max(item.expected.intent.selection.expires_at_ms for item in receipts)

        @asynccontextmanager
        async def transaction(scope, *, write):
            async with original(scope, write=write) as tx:

                async def now():
                    return deadline

                tx.now_ms = now
                yield tx

        monkeypatch.setattr(store, "_transaction", transaction)
    for index, receipt in enumerate(receipts):
        request = RequestControl(
            operation=initialized.operation(f"control-{index}"),
            expected=receipt.expected,
            expected_revision=1,
            kind=kind,
        )
        result = await application.control_collaboration_request(request, context=resolver.context)
        assert result.state == ("expired" if kind == "expire" else "cancelled")
        assert (
            await application.control_collaboration_request(request, context=resolver.context)
            == result
        )
    final = await anchor(store, initialized)
    assert final.reserved_operations == final.reserved_events == final.reserved_bytes == 0
    assert (
        await application.inspect_participant(values[3].reference, context=CONTEXT)
    ).outstanding_obligations == 0
    await application.drain_collaboration_requests()


async def test_public_envelope_boundary_rejects_without_responsibility_or_settles(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    initialized, request = values[1], values[4]
    hint = request.delivery_contract.model_copy(update={"object_id": "\\" * 512})
    low, high = 0, 4096
    largest = None
    rejected = []
    while low <= high:
        size = (low + high) // 2
        probe = request.model_copy(
            update={
                "operation": initialized.operation(f"size-{size}"),
                "content": "\\😀" * size,
                "context_hint": hint,
            }
        )
        # Each original request is valid. Any refusal is caused by a composed
        # command/receipt/future control envelope, not an oversized input field.
        prepare_contract(type(request), probe, redactor=REDACTOR)
        before = await anchor(store, initialized)
        try:
            accepted = await application.accept_collaboration_request(
                probe, context=resolver.context
            )
        except CollaborationContractError:
            rejected.append(size)
            assert await anchor(store, initialized) == before
            high = size - 1
            continue
        control = RequestControl(
            operation=initialized.operation("\\" * 500 + str(size)),
            expected=accepted.expected,
            expected_revision=1,
            kind="cancel",
        )
        settled = await application.control_collaboration_request(control, context=resolver.context)
        snapshot = await application.inspect_collaboration_request(
            accepted.expected, context=resolver.context
        )
        assert snapshot.terminal == settled
        assert len(contract_bytes(snapshot, redactor=REDACTOR)) <= MAX_ENVELOPE_BYTES
        assert (
            await application.lookup_collaboration_request(
                settled.expected, context=resolver.context
            )
        ).receipt == settled
        largest = size
        low = size + 1
    assert largest is not None and rejected
    assert min(rejected) == largest + 1
    final = await anchor(store, initialized)
    assert final.reserved_operations == final.reserved_events == final.reserved_bytes == 0
    assert (
        await application.inspect_participant(values[3].reference, context=CONTEXT)
    ).outstanding_obligations == 0
    await application.drain_collaboration_requests()
