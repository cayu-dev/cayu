"""Public restricted readback and mandatory-control representation bounds."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_collaboration_request_foundation import REDACTOR, public_setup
from tests.core.test_collaboration_request_qualification import anchor

from cayu.collaboration._contracts import CollaborationContractError, ExactMatch
from cayu.collaboration._preparation import contract_bytes, prepare_contract
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.access import CollaborationAccessDenied, CollaborationAccessGrant
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import MAX_CONTROL_INITIATOR_BYTES, RequestControl

pytestmark = pytest.mark.anyio
stores = identity_tests.stores


@pytest.mark.parametrize("restricted_action", ["request_readback", "request_accept"])
async def test_fresh_alias_hides_missing_and_inaccessible_targets(
    stores, monkeypatch, restricted_action
):
    store = stores()
    application, resolver, values = await public_setup(store)
    initialized, sender, hidden, request = values[1:5]
    _, visible = await identity_tests.create(
        application, initialized, "visible", alias="visible", expected_alias_revision=1
    )
    visible_ref = visible.participants[0].reference
    policy = application._participant_coordinator._registration.access_policy
    authorize = policy.authorize

    def restricted(context, *, application_scope, action):
        if action == restricted_action:
            return CollaborationAccessGrant(
                application_scope=application_scope,
                participants=(sender.reference, visible_ref),
            )
        return authorize(context, application_scope=application_scope, action=action)

    monkeypatch.setattr(policy, "authorize", restricted)
    load = store._participant

    async def guarded_load(tx, reference, owner, redactor):
        assert reference != hidden.reference, "Unreadable recipient snapshot was loaded"
        return await load(tx, reference, owner, redactor)

    monkeypatch.setattr(store, "_participant", guarded_load)
    before = await anchor(store, initialized)
    failures = []
    for alias in ("reviewer", "missing"):
        probe = request.model_copy(
            update={"target": request.target.model_copy(update={"alias": alias})}
        )
        with pytest.raises(CollaborationAccessDenied) as caught:
            await application.accept_collaboration_request(probe, context=resolver.context)
        error = caught.value
        failures.append((type(error), str(error), type(error.__cause__), str(error.__cause__)))
        assert await anchor(store, initialized) == before
    assert failures[0] == failures[1]
    allowed = request.model_copy(
        update={"target": request.target.model_copy(update={"alias": "visible"})}
    )
    receipt = await application.accept_collaboration_request(allowed, context=resolver.context)
    assert receipt.expected.intent.selection.recipient.reference == visible_ref
    assert (
        await application.accept_collaboration_request(allowed, context=resolver.context) == receipt
    )
    assert isinstance(
        await application.lookup_collaboration_request(receipt.expected, context=resolver.context),
        ExactMatch,
    )
    await application.drain_collaboration_requests()


@pytest.mark.parametrize("family", ["accept", "control"])
async def test_concurrent_key_creation_checks_retained_read_permission(stores, monkeypatch, family):
    store = stores()
    winner, resolver, values = await public_setup(store)
    initialized, sender, _recipient, request = values[1:5]
    allowed = request.model_copy(update={"target": sender.reference})
    if family == "control":
        excluded_receipt = await winner.accept_collaboration_request(
            request, context=resolver.context
        )
        allowed_receipt = await winner.accept_collaboration_request(
            allowed.model_copy(update={"operation": initialized.operation("allowed")}),
            context=resolver.context,
        )
        winning = RequestControl(
            operation=initialized.operation("race"),
            expected=excluded_receipt.expected,
            expected_revision=1,
            kind="cancel",
        )
        losing = winning.model_copy(update={"expected": allowed_receipt.expected})
    else:
        winning, losing = request, allowed
    restricted = identity_tests.Policy()
    restricted.allowed = (sender.reference,)
    loser = identity_tests.app(
        store,
        replace(values[0]._participant_coordinator._registration, access_policy=restricted),
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    await loser.initialize_collaboration()
    entered, release = asyncio.Event(), asyncio.Event()
    original = store._transaction
    armed = True

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal armed
        if write and armed:
            armed = False
            entered.set()
            await release.wait()
        async with original(scope, write=write) as tx:
            yield tx

    monkeypatch.setattr(store, "_transaction", transaction)

    async def invoke(application, value):
        method = (
            application.accept_collaboration_request
            if family == "accept"
            else application.control_collaboration_request
        )
        return await method(value, context=resolver.context)

    caller = asyncio.create_task(invoke(loser, losing))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        result = await invoke(winner, winning)
        before = await anchor(store, initialized)
        release.set()
        with pytest.raises(CollaborationAccessDenied):
            await asyncio.wait_for(caller, 5)
        assert await anchor(store, initialized) == before
        assert await invoke(winner, winning) == result
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await loser.drain_collaboration_requests()
        await winner.drain_collaboration_requests()


@pytest.mark.parametrize("family", ["accept", "acceptance_lookup", "control_lookup"])
async def test_restricted_readback_never_exposes_exact_conflict(stores, monkeypatch, family):
    store = stores()
    application, resolver, values = await public_setup(store)
    initialized, sender, _recipient, request = values[1:5]
    receipt = await application.accept_collaboration_request(request, context=resolver.context)
    control = RequestControl(
        operation=initialized.operation("control"),
        expected=receipt.expected,
        expected_revision=1,
        kind="cancel",
    )
    terminal = await application.control_collaboration_request(control, context=resolver.context)
    allowed_receipt = await application.accept_collaboration_request(
        request.model_copy(
            update={"operation": initialized.operation("allowed"), "target": sender.reference}
        ),
        context=resolver.context,
    )
    policy = application._participant_coordinator._registration.access_policy
    authorize = policy.authorize

    def restricted(context, *, application_scope, action):
        if action == "request_readback":
            return CollaborationAccessGrant(
                application_scope=application_scope, participants=(sender.reference,)
            )
        return authorize(context, application_scope=application_scope, action=action)

    monkeypatch.setattr(policy, "authorize", restricted)
    before = await anchor(store, initialized)
    for missing in (False, True):
        for changed in (False, True):
            operation = initialized.operation("missing") if missing else request.operation
            original = request.model_copy(
                update={
                    "operation": operation,
                    "content": "Changed" if changed else request.content,
                }
            )
            expected = receipt.expected.model_copy(
                update={
                    "operation": operation,
                    "intent": receipt.expected.intent.model_copy(update={"request": original}),
                }
            )
            prepare_contract(type(expected), expected, redactor=REDACTOR)
            if family == "accept":
                call = application.accept_collaboration_request(original, context=resolver.context)
            else:
                if family == "control_lookup":
                    expected = terminal.expected.model_copy(
                        update={
                            "intent": control.model_copy(update={"expected": expected}),
                        }
                    )
                call = application.lookup_collaboration_request(expected, context=resolver.context)
            with pytest.raises(CollaborationAccessDenied):
                await call
    # An alias-addressed caller cannot substitute an allowed participant in its
    # claimed frozen selection to obtain a conflict for an excluded stored one.
    if family != "accept":
        forged = receipt.expected.model_copy(
            update={
                "intent": receipt.expected.intent.model_copy(
                    update={
                        "selection": receipt.expected.intent.selection.model_copy(
                            update={"recipient": sender}
                        ),
                    }
                ),
            }
        )
        if family == "control_lookup":
            forged = terminal.expected.model_copy(
                update={
                    "intent": control.model_copy(update={"expected": forged}),
                }
            )
        prepare_contract(type(forged), forged, redactor=REDACTOR)
        with pytest.raises(CollaborationAccessDenied):
            await application.lookup_collaboration_request(forged, context=resolver.context)
    if family == "control_lookup":
        # The outer control key belongs to excluded participants even when its
        # submitted nested acceptance really exists and is readable.
        forged = terminal.expected.model_copy(
            update={
                "intent": control.model_copy(update={"expected": allowed_receipt.expected}),
            }
        )
        prepare_contract(type(forged), forged, redactor=REDACTOR)
        with pytest.raises(CollaborationAccessDenied):
            await application.lookup_collaboration_request(forged, context=resolver.context)
    assert await anchor(store, initialized) == before
    monkeypatch.setattr(policy, "authorize", authorize)
    assert isinstance(
        await application.lookup_collaboration_request(receipt.expected, context=resolver.context),
        ExactMatch,
    )
    await application.drain_collaboration_requests()


def escaped_administrator(resolver, *, oversized=False):
    context = resolver.context.model_copy(
        update={
            "principal": "\x01" * 512,
            "participant": None,
            "mandate": resolver.context.mandate.model_copy(
                update={
                    "object_id": "\x01" * 512,
                    "incarnation": "\x01" * (512 if oversized else 128),
                }
            ),
        }
    )
    if not oversized:
        remaining = MAX_CONTROL_INITIATOR_BYTES - len(
            contract_bytes(_initiator(context), redactor=REDACTOR)
        )
        assert remaining >= 0
        incarnation = (
            context.mandate.incarnation + "\x01" * (remaining // 6) + "x" * (remaining % 6)
        )
        context = context.model_copy(
            update={"mandate": context.mandate.model_copy(update={"incarnation": incarnation})}
        )
    leaf = resolver.resolution.chain.entries[-1].model_copy(
        update={
            "principal": context.principal,
            "participant": None,
            "reference": context.mandate,
            "root": context.mandate,
        }
    )
    resolution = resolver.resolution.model_copy(
        update={
            "principal": resolver.resolution.principal.model_copy(
                update={
                    "principal": context.principal,
                    "participants": (),
                }
            ),
            "chain": resolver.resolution.chain.model_copy(update={"entries": (leaf,)}),
        }
    )
    return (
        prepare_contract(type(context), context, redactor=REDACTOR),
        prepare_contract(type(resolution), resolution, redactor=REDACTOR),
    )


@pytest.mark.parametrize("kind", ["cancel", "expire"])
async def test_escaped_control_boundary_settles_admitted_requests(stores, monkeypatch, kind):
    store = stores()
    application, resolver, values = await public_setup(store)
    initialized, request = values[1], values[4]
    original_context, original_resolution = resolver.context, resolver.resolution
    admin, admin_resolution = escaped_administrator(resolver)
    oversized, oversized_resolution = escaped_administrator(resolver, oversized=True)
    assert len(contract_bytes(_initiator(admin), redactor=REDACTOR)) == MAX_CONTROL_INITIATOR_BYTES
    assert (
        len(contract_bytes(_initiator(oversized), redactor=REDACTOR)) > MAX_CONTROL_INITIATOR_BYTES
    )
    policy = application._participant_coordinator._registration.access_policy
    authorize = policy.authorize

    def administrators(context, **kwargs):
        assert context.principal in (original_context.principal, admin.principal)
        return authorize(
            context.model_copy(update={"principal": original_context.principal}), **kwargs
        )

    monkeypatch.setattr(policy, "authorize", administrators)
    low, high = 0, 4096
    largest = None
    rejected = []
    original_transaction = store._transaction
    for_search_time = None

    @asynccontextmanager
    async def transaction(scope, *, write):
        async with original_transaction(scope, write=write) as tx:
            if for_search_time is not None:

                async def now():
                    return for_search_time

                tx.now_ms = now
            yield tx

    monkeypatch.setattr(store, "_transaction", transaction)
    while low <= high:
        size = (low + high) // 2
        probe = request.model_copy(
            update={
                "operation": initialized.operation(f"size-{size}"),
                "content": "\\😀" * size,
            }
        )
        prepare_contract(type(probe), probe, redactor=REDACTOR)
        before = await anchor(store, initialized)
        try:
            accepted = await application.accept_collaboration_request(
                probe, context=resolver.context
            )
        except CollaborationContractError:
            assert await anchor(store, initialized) == before
            rejected.append(size)
            high = size - 1
            continue
        if kind == "expire":
            for_search_time = accepted.expected.intent.selection.expires_at_ms
        control = RequestControl(
            operation=initialized.operation("\x01" * 500 + str(size)),
            expected=accepted.expected,
            expected_revision=1,
            kind=kind,
        )
        before_control = await anchor(store, initialized)
        resolver.context, resolver.resolution = oversized, oversized_resolution
        with pytest.raises(CollaborationContractError):
            await application.control_collaboration_request(control, context=oversized)
        assert await anchor(store, initialized) == before_control
        resolver.context, resolver.resolution = admin, admin_resolution
        result = await application.control_collaboration_request(control, context=resolver.context)
        assert result.state == ("expired" if kind == "expire" else "cancelled")
        assert (
            await application.control_collaboration_request(control, context=resolver.context)
            == result
        )
        assert (
            await application.inspect_collaboration_request(
                accepted.expected, context=resolver.context
            )
        ).terminal == result
        resolver.context, resolver.resolution = original_context, original_resolution
        largest = size
        low = size + 1
    assert largest is not None and rejected and min(rejected) == largest + 1
    final = await anchor(store, initialized)
    assert final.reserved_operations == final.reserved_events == final.reserved_bytes == 0
    await application.drain_collaboration_requests()
