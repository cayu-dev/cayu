"""Public and transactional qualification for the request foundation slice."""

import asyncio
import time
from contextlib import asynccontextmanager

import pytest
from tests.core import test_participant_identity as identity_tests
from tests.core.test_participant_identity import CONTEXT, app, create, registration
from tests.core.test_participant_lifecycle import change

from cayu.collaboration._contracts import (
    CollaborationConflict,
    ExactMatch,
    InitiatorBinding,
    ObjectRef,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration._request_store import (
    accept_in_transaction,
    control_in_transaction,
    retained_request,
)
from cayu.collaboration.mandates import (
    CollaborationMandate,
    MandateAccessContext,
    MandateChain,
    MandateDenied,
    MandateResolution,
    MandateResolver,
    MandateRestrictions,
    PrincipalResolution,
)
from cayu.collaboration.request_access import RequestRegistration
from cayu.collaboration.requests import (
    CollaborationRequest,
    RequestAlias,
    RequestControl,
    RequestControlCommand,
    RequestControlReceipt,
    RequestReceipt,
    RequestSnapshot,
)
from cayu.storage._collaboration_schema import validate_sqlite_collaboration_schema
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio
stores = identity_tests.stores
REDACTOR = SecretRedactor()


async def setup(store, *, reg=None):
    application = app(store, registration() if reg is None else reg)
    initialized = await application.initialize_collaboration()
    _, first = await create(application, initialized, "sender")
    _, second = await create(application, initialized, "recipient", alias="reviewer")
    sender, recipient = first.participants[0], second.participants[0]
    contract = ObjectRef(
        owner=initialized.owner,
        kind="contract",
        object_id="contract",
        incarnation="one",
        revision=1,
    )
    request = CollaborationRequest(
        operation=initialized.operation("request"),
        kind="question",
        sender=sender.reference,
        target=RequestAlias(owner=initialized.owner, alias="reviewer"),
        content="Review the selected material.",
        inputs=(),
        context_hint=None,
        delivery_contract=contract,
        output_contract=contract,
        independence_policy=contract,
        disclosure_policy=contract,
        ttl_ms=60_000,
        cancellation="detach",
    )
    initiator = InitiatorBinding(
        issuer=initialized.owner,
        principal="operator",
        participant=ObjectRef(
            owner=initialized.owner,
            kind="participant",
            object_id=sender.reference.participant_id,
            incarnation=sender.reference.incarnation,
        ),
        mandate=ObjectRef(
            owner=initialized.owner,
            kind="authority",
            object_id="mandate",
            incarnation="one",
            revision=1,
        ),
        invocation_id=None,
        interaction_id=None,
    )
    return application, initialized, sender, recipient, request, initiator


async def accept(store, values, *, fail=False):
    _, initialized, sender, recipient, request, initiator = values
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        result = await accept_in_transaction(
            store,
            tx,
            initialized,
            request,
            initiator,
            sender=sender,
            recipient=recipient,
            authority=RequestResolver(request).resolution,
            authority_expires_at_ms=time.time_ns() // 1_000_000 + 300_000,
            redactor=REDACTOR,
        )
        if fail:
            raise RuntimeError("injected after request publication")
        return result


async def test_atomic_acceptance_and_exact_replay(stores):
    store = stores()
    values = await setup(store)
    application, initialized, _, recipient, request, initiator = values
    first = await accept(store, values)
    assert first.state == "open"
    assert first.admission == "undecided"
    assert first.delivery == "pending"
    assert first.receipt.expected.intent.selection.recipient == recipient
    assert (
        first.receipt.expected.intent.selection.expires_at_ms
        - first.receipt.expected.intent.selection.accepted_at_ms
        == request.ttl_ms
    )
    assert await accept(store, values) == first
    inspected = await application.inspect_participant(recipient.reference, context=CONTEXT)
    assert inspected.outstanding_obligations == 1
    assert inspected.issued_permit_frontier == 1
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        anchor = await store._anchor(tx, initialized, REDACTOR)
        assert anchor.reserved_operations == 1
        assert anchor.reserved_events == 2
    reopened = stores()
    async with reopened._transaction(initialized.binding.application_scope, write=False) as tx:
        found = await retained_request(reopened, tx, initialized, request, initiator, REDACTOR)
    assert found == first
    assert (
        prepare_contract(RequestSnapshot, first.model_dump(mode="json"), redactor=REDACTOR) == first
    )


async def test_acceptance_rollback_does_not_leak_permit_or_capacity(stores):
    store = stores()
    values = await setup(store)
    application, initialized, _, recipient, request, initiator = values
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initialized, REDACTOR)
    with pytest.raises(RuntimeError, match="injected"):
        await accept(store, values, fail=True)
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initialized, REDACTOR) == before
        assert await retained_request(store, tx, initialized, request, initiator, REDACTOR) is None
    assert (
        await application.inspect_participant(recipient.reference, context=CONTEXT)
    ).outstanding_obligations == 0
    assert (await accept(store, values)).state == "open"


async def test_disable_refuses_fresh_acceptance_but_not_exact_replay(stores):
    store = stores()
    values = await setup(store)
    application, initialized, _, recipient, request, _ = values
    accepted = await accept(store, values)
    await application.change_participant_lifecycle(
        change(initialized, recipient.reference, key="disable", revision=1, state="disabled"),
        context=CONTEXT,
    )
    assert await accept(store, values) == accepted
    fresh = request.model_copy(update={"operation": initialized.operation("fresh")})
    with pytest.raises(CollaborationConflict):
        await accept(store, (*values[:4], fresh, values[5]))
    assert (
        await application.inspect_participant(recipient.reference, context=CONTEXT)
    ).outstanding_obligations == 1


@pytest.mark.parametrize("field", ["content", "ttl_ms", "kind", "cancellation", "initiator"])
async def test_fixed_key_conflicts_include_original_intent_and_initiator(stores, field):
    store = stores()
    values = await setup(store)
    await accept(store, values)
    request, initiator = values[4:]
    if field == "initiator":
        initiator = initiator.model_copy(update={"principal": "different"})
    else:
        changes = {
            "content": {"content": "Another review."},
            "ttl_ms": {"ttl_ms": 30_000},
            "kind": {"kind": "contribution", "output_contract": None},
            "cancellation": {"cancellation": "stop"},
        }
        request = request.model_copy(update=changes[field])
    with pytest.raises(CollaborationConflict):
        await accept(store, (*values[:4], request, initiator))


async def test_owner_time_is_sampled_after_lock_acquisition(stores):
    store = stores()
    values = await setup(store)
    initialized = values[1]
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold():
        async with store._transaction(initialized.binding.application_scope, write=True):
            entered.set()
            await release.wait()

    async def sample():
        async with store._transaction(initialized.binding.application_scope, write=True) as tx:
            return await tx.now_ms()

    owner = asyncio.create_task(hold())
    await entered.wait()
    reader = asyncio.create_task(sample())
    try:
        await asyncio.sleep(0.02)
        assert not reader.done()
        released_at = time.time_ns() // 1_000_000
        release.set()
        await owner
        assert await reader >= released_at
    finally:
        release.set()
        await asyncio.gather(owner, reader, return_exceptions=True)


async def test_request_schema_qualification_includes_due_indexes(tmp_path):
    from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
    from cayu.storage.migrations import SchemaError

    store = SQLiteCollaborationStore(tmp_path / "requests.sqlite")
    try:
        validate_sqlite_collaboration_schema(store._connection, lifecycle=True, requests=True)
        store._connection.execute("DROP INDEX cayu_collaboration_request_due_idx")
        with pytest.raises(SchemaError):
            validate_sqlite_collaboration_schema(store._connection, lifecycle=True, requests=True)
    finally:
        await store.close()


async def control(store, initialized, accepted, initiator, *, kind="cancel", fail=False):
    operation = initialized.operation("control")
    expected = RequestControlCommand(
        operation=operation,
        kind=kind,
        source=initialized.owner,
        destination=initialized.owner,
        initiator=initiator,
        intent=RequestControl(
            operation=operation,
            expected=accepted.receipt.expected,
            expected_revision=1,
            kind=kind,
        ),
    )
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        result = await control_in_transaction(
            store,
            tx,
            initialized,
            expected,
            authority_expires_at_ms=time.time_ns() // 1_000_000 + 300_000,
            redactor=REDACTOR,
        )
        if fail:
            raise RuntimeError("after control")
        return result


async def test_control_settles_permit_atomically_and_replays(stores):
    store = stores()
    values = await setup(store)
    application, initialized, _, recipient, request, initiator = values
    accepted = await accept(store, values)
    with pytest.raises(RuntimeError, match="after control"):
        await control(store, initialized, accepted, initiator, fail=True)
    assert (
        await application.inspect_participant(recipient.reference, context=CONTEXT)
    ).outstanding_obligations == 1
    elected = await control(store, initialized, accepted, initiator)
    assert elected.state == "cancelled"
    assert await control(store, initialized, accepted, initiator) == elected
    assert (
        await application.inspect_participant(recipient.reference, context=CONTEXT)
    ).outstanding_obligations == 0
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        anchor = await store._anchor(tx, initialized, REDACTOR)
        assert anchor.reserved_operations == anchor.reserved_events == anchor.reserved_bytes == 0
        snapshot = await retained_request(store, tx, initialized, request, initiator, REDACTOR)
        assert snapshot.terminal == elected
        assert snapshot.delivery == "excluded"


async def test_expiry_equality_and_control_before_deadline(stores):
    store = stores()
    values = await setup(store)
    _, initialized, _, _, _, initiator = values
    accepted = await accept(store, values)
    with pytest.raises(CollaborationConflict, match="not passed"):
        await control(store, initialized, accepted, initiator, kind="expire")
    operation = initialized.operation("control")
    expected = RequestControlCommand(
        operation=operation,
        kind="expire",
        source=initialized.owner,
        destination=initialized.owner,
        initiator=initiator,
        intent=RequestControl(
            operation=operation,
            expected=accepted.receipt.expected,
            expected_revision=1,
            kind="expire",
        ),
    )
    deadline = accepted.receipt.expected.intent.selection.expires_at_ms
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:

        async def now_ms():
            return deadline

        tx.now_ms = now_ms
        elected = await control_in_transaction(
            store,
            tx,
            initialized,
            expected,
            authority_expires_at_ms=deadline + 1000,
            redactor=REDACTOR,
        )
    assert elected.state == "expired"
    assert elected.elected_at_ms == deadline


class RequestResolver(MandateResolver):
    def __init__(self, request):
        owner = request.sender.owner

        def ref(name):
            return ObjectRef(
                owner=owner, kind="authority", object_id=name, incarnation="one", revision=1
            )

        self._ref = ref("resolver")
        self.denied = False
        self.context = MandateAccessContext(
            issuer=owner, principal="operator", participant=request.sender, mandate=ref("mandate")
        )
        mandate = CollaborationMandate(
            reference=ref("mandate"),
            root=ref("mandate"),
            parent=None,
            issuer=owner,
            principal="operator",
            participant=request.sender,
            audiences=(owner,),
            scopes=(owner.application_scope,),
            actions=("consult", "readback", "administer"),
            resources=(),
            remaining_delegations=0,
            sponsor=None,
            budgets=(),
            restrictions=MandateRestrictions(
                channels=("prompt",),
                excluded_sources=(),
                independence_policy=request.independence_policy,
                disclosure_policy=request.disclosure_policy,
            ),
            expires_at_ms=4102444800000,
            revocation_generation=1,
        )
        self.resolution = MandateResolution(
            principal=PrincipalResolution(
                resolver=self.ref,
                issuer=owner,
                principal="operator",
                participants=(request.sender,),
                audiences=(owner,),
                scopes=(owner.application_scope,),
                actions=mandate.actions,
                expires_at_ms=mandate.expires_at_ms,
            ),
            chain=MandateChain(entries=(mandate,)),
        )

    @property
    def ref(self):
        return self._ref

    @asynccontextmanager
    async def acquire(self, context):
        if self.denied or context != self.context:
            raise MandateDenied()
        yield self.resolution


async def public_setup(store, *, reg=None, **kwargs):
    values = await setup(store, reg=reg)
    resolver = RequestResolver(values[4])
    application = app(
        store,
        values[0]._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
        **kwargs,
    )
    await application.initialize_collaboration()
    return application, resolver, values


@pytest.mark.parametrize("kind", ["question", "contribution"])
async def test_public_accept_inspect_lookup_control(stores, kind):
    store = stores()
    application, resolver, values = await public_setup(store)
    request = values[4]
    if kind == "contribution":
        request = request.model_copy(update={"kind": kind, "output_contract": None})
    receipt = await application.accept_collaboration_request(request, context=resolver.context)
    assert (
        await application.accept_collaboration_request(request, context=resolver.context) == receipt
    )
    snapshot = await application.inspect_collaboration_request(
        receipt.expected, context=resolver.context
    )
    assert snapshot.state == "open"
    assert await application.lookup_collaboration_request(
        receipt.expected, context=resolver.context
    ) == ExactMatch[RequestReceipt](receipt=receipt)
    control_request = RequestControl(
        operation=values[1].operation("control"),
        expected=receipt.expected,
        expected_revision=1,
        kind="cancel",
    )
    result = await application.control_collaboration_request(
        control_request, context=resolver.context
    )
    assert result.state == "cancelled"
    assert await application.lookup_collaboration_request(
        result.expected, context=resolver.context
    ) == ExactMatch[RequestControlReceipt](receipt=result)
    assert (
        await application.control_collaboration_request(control_request, context=resolver.context)
        == result
    )
    assert (
        await application.inspect_participant(values[3].reference, context=CONTEXT)
    ).outstanding_obligations == 0
    await application.drain_collaboration_requests()


async def test_independent_applications_accept_once(stores):
    first, resolver, values = await public_setup(stores())
    second = app(
        stores(),
        values[0]._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    await second.initialize_collaboration()
    try:
        left, right = await asyncio.gather(
            first.accept_collaboration_request(values[4], context=resolver.context),
            second.accept_collaboration_request(values[4], context=resolver.context),
        )
        assert left == right
        assert (
            await second.inspect_participant(values[3].reference, context=CONTEXT)
        ).outstanding_obligations == 1
        page = await second.list_due_collaboration_requests(context=resolver.context)
        assert len(page.items) == 1 and page.items[0].receipt == left
    finally:
        await first.drain_collaboration_requests()
        await second.drain_collaboration_requests()


@pytest.mark.parametrize("mode", ["accept", "inspect", "due"])
async def test_pending_observation_does_not_authenticate_another_application(stores, mode):
    from cayu.collaboration.access import CollaborationAccessDenied

    store = stores()
    first, resolver, values = await public_setup(store)
    receipt = None
    if mode != "accept":
        receipt = await first.accept_collaboration_request(values[4], context=resolver.context)
    denied = RequestResolver(values[4])
    denied.denied = True
    second = app(
        store,
        values[0]._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=denied, max_ttl_ms=300_000),
    )
    await second.initialize_collaboration()
    entered, release = asyncio.Event(), asyncio.Event()
    original = resolver.acquire

    @asynccontextmanager
    async def held(context):
        async with original(context) as authority:
            entered.set()
            await release.wait()
            yield authority

    resolver.acquire = held

    def invoke(application):
        if mode == "accept":
            return application.accept_collaboration_request(values[4], context=resolver.context)
        if mode == "inspect":
            return application.inspect_collaboration_request(
                receipt.expected, context=resolver.context
            )
        return application.list_due_collaboration_requests(context=resolver.context)

    owner = asyncio.create_task(invoke(first))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        with pytest.raises(CollaborationAccessDenied):
            await asyncio.wait_for(invoke(second), 2)
        assert not owner.done()
        release.set()
        assert await owner is not None
        assert (
            await first.inspect_participant(values[3].reference, context=CONTEXT)
        ).outstanding_obligations == 1
    finally:
        release.set()
        await asyncio.gather(owner, return_exceptions=True)
        await first.drain_collaboration_requests()


async def test_acceptance_requires_read_grant_for_both_participants(stores):
    from cayu.collaboration.access import CollaborationAccessDenied, CollaborationAccessGrant

    store = stores()
    application, resolver, values = await public_setup(store)
    policy = application._participant_coordinator._registration.access_policy
    original = policy.authorize

    def restricted(context, *, application_scope, action):
        if action == "request_readback":
            return CollaborationAccessGrant(
                application_scope=application_scope, participants=(values[2].reference,)
            )
        return original(context, application_scope=application_scope, action=action)

    policy.authorize = restricted
    scope = values[1].binding.application_scope
    async with store._transaction(scope, write=False) as tx:
        before = await tx.get("anchors", ())
    try:
        with pytest.raises(CollaborationAccessDenied):
            await application.accept_collaboration_request(values[4], context=resolver.context)
        async with store._transaction(scope, write=False) as tx:
            assert await tx.get("anchors", ()) == before
            assert (
                await tx.get(
                    "requests",
                    (
                        values[4].operation.namespace_incarnation,
                        values[4].operation.generation,
                        values[4].operation.caller_key,
                    ),
                )
                is None
            )
        policy.authorize = original
        assert await application.accept_collaboration_request(values[4], context=resolver.context)
    finally:
        policy.authorize = original
        await application.drain_collaboration_requests()


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("mode", ["accept", "due"])
async def test_store_failure_and_guard_failure_preserve_both(stores, cancel, mode, caplog, capsys):
    import warnings

    from cayu.collaboration.participants import CollaborationUnavailable

    store = stores()
    application, resolver, values = await public_setup(
        store, secret_redactor=SecretRedactor("private-canary")
    )
    original = store._transaction

    @asynccontextmanager
    async def failed_transaction(*args, **kwargs):
        raise ConnectionError("store-primary private-canary")
        yield

    @asynccontextmanager
    async def failed_cleanup(context):
        try:
            yield resolver.resolution
        finally:
            if cancel:
                caller.cancel("private-canary")
                caller.cancel()
            raise OSError("guard-cleanup private-canary")

    resolver.acquire = failed_cleanup
    store._transaction = failed_transaction
    with warnings.catch_warnings(record=True) as recorded:
        caller = asyncio.create_task(
            application.accept_collaboration_request(values[4], context=resolver.context)
            if mode == "accept"
            else application.list_due_collaboration_requests(context=resolver.context)
        )
        try:
            with pytest.raises(
                asyncio.CancelledError if cancel else CollaborationUnavailable
            ) as caught:
                await caller
            assert caller.cancelled() == cancel
            assert caller.cancelling() == (2 if cancel else 0)
            pending, seen, graph = [caught.value], set(), []
            while pending:
                error = pending.pop()
                if id(error) in seen:
                    continue
                seen.add(id(error))
                graph.append(error)
                pending.extend(
                    item for item in (error.__cause__, error.__context__) if item is not None
                )
                if isinstance(error, BaseExceptionGroup):
                    pending.extend(error.exceptions)
            assert sum("store-primary" in str(error) for error in graph) == 1
            assert sum("guard-cleanup" in str(error) for error in graph) == 1
            assert sum(isinstance(error, asyncio.CancelledError) for error in graph) == int(cancel)
            output = capsys.readouterr()
            assert "private-canary" not in (
                repr(graph)
                + caplog.text
                + output.out
                + output.err
                + repr([str(item.message) for item in recorded])
            )
        finally:
            store._transaction = original
            await application.drain_collaboration_requests()


async def test_public_denial_is_not_unavailability_and_has_no_publication(stores):
    from cayu.collaboration.access import CollaborationAccessDenied

    store = stores()
    application, resolver, values = await public_setup(store)
    resolver.denied = True
    with pytest.raises(CollaborationAccessDenied):
        await application.accept_collaboration_request(values[4], context=resolver.context)
    assert (
        await application.inspect_participant(values[3].reference, context=CONTEXT)
    ).outstanding_obligations == 0
    resolver.denied = False
    assert (
        await application.accept_collaboration_request(values[4], context=resolver.context)
    ).event.type == "request_accepted"
    await application.drain_collaboration_requests()


@pytest.mark.parametrize("inspection", [False, True])
async def test_dependency_cancellation_preserves_prior_failure(stores, inspection):
    from cayu.collaboration.participants import CollaborationUnavailable

    store = stores()
    application, resolver, values = await public_setup(store)

    @asynccontextmanager
    async def broken_guard(context):
        try:
            raise RuntimeError("mandate-read-failure")
        finally:
            dependency = asyncio.create_task(asyncio.sleep(60))
            dependency.cancel("dependency-stop")
            await dependency
        yield resolver.resolution

    resolver.acquire = broken_guard
    caller = asyncio.create_task(
        application.list_due_collaboration_requests(context=resolver.context)
        if inspection
        else application.accept_collaboration_request(values[4], context=resolver.context)
    )
    try:
        with pytest.raises(CollaborationUnavailable) as raised:
            await caller
        assert not caller.cancelled() and caller.cancelling() == 0
        pending = [raised.value]
        seen = set()
        messages = []
        while pending:
            error = pending.pop()
            if id(error) in seen:
                continue
            seen.add(id(error))
            messages.append(str(error))
            pending.extend(
                item for item in (error.__cause__, error.__context__) if item is not None
            )
            if isinstance(error, BaseExceptionGroup):
                pending.extend(error.exceptions)
        assert any("mandate-read-failure" in message for message in messages)
        assert (
            await application.inspect_participant(values[3].reference, context=CONTEXT)
        ).outstanding_obligations == 0
    finally:
        await application.drain_collaboration_requests()


async def test_public_cancellation_retains_committed_work_and_exact_retry(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    original = store._transaction
    committed, release = asyncio.Event(), asyncio.Event()
    intercepted = False

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal intercepted
        async with original(scope, write=write) as tx:
            yield tx
        if write and not intercepted:
            intercepted = True
            committed.set()
            await release.wait()

    store._transaction = transaction
    caller = asyncio.create_task(
        application.accept_collaboration_request(values[4], context=resolver.context)
    )
    retry = None
    try:
        await asyncio.wait_for(committed.wait(), 5)
        caller.cancel("sensitive-cancellation")
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        assert caller.cancelled() and caller.cancelling() == 2
        assert len(application._request_coordinator._owners.pending) == 1
        retry = asyncio.create_task(
            application.accept_collaboration_request(values[4], context=resolver.context)
        )
        await asyncio.sleep(0)
        assert not retry.done()
        release.set()
        receipt = await retry
        assert await application.lookup_collaboration_request(
            receipt.expected, context=resolver.context
        ) == ExactMatch[RequestReceipt](receipt=receipt)
        assert (
            await application.inspect_participant(values[3].reference, context=CONTEXT)
        ).outstanding_obligations == 1
    finally:
        release.set()
        await asyncio.gather(caller, *(() if retry is None else (retry,)), return_exceptions=True)
        store._transaction = original
        await application.drain_collaboration_requests()


async def test_public_due_inspection_is_bounded_and_does_not_dispatch(stores):
    store = stores()
    application, resolver, values = await public_setup(store)
    receipts = []
    for index in range(3):
        receipts.append(
            await application.accept_collaboration_request(
                values[4].model_copy(update={"operation": values[1].operation(f"request-{index}")}),
                context=resolver.context,
            )
        )
    page = await application.list_due_collaboration_requests(context=resolver.context, limit=2)
    assert tuple(item.receipt for item in page.items) == tuple(receipts[:2])
    assert page.next_cursor is not None
    second = await application.list_due_collaboration_requests(
        context=resolver.context, cursor=page.next_cursor, limit=2
    )
    assert tuple(item.receipt for item in second.items) == tuple(receipts[2:])
    assert second.next_cursor is None
    assert all(item.admission == "undecided" for item in (*page.items, *second.items))
    await application.drain_collaboration_requests()


async def test_control_replay_needs_read_authority_not_fresh_administration(stores):
    from cayu.collaboration.access import CollaborationAccessDenied

    store = stores()
    application, resolver, values = await public_setup(store)
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    command = RequestControl(
        operation=values[1].operation("control"),
        expected=accepted.expected,
        expected_revision=1,
        kind="cancel",
    )
    first = await application.control_collaboration_request(command, context=resolver.context)
    resolver.resolution = resolver.resolution.model_copy(
        update={
            "principal": resolver.resolution.principal.model_copy(update={"actions": ("readback",)})
        }
    )
    assert (
        await application.control_collaboration_request(command, context=resolver.context) == first
    )
    assert (
        await application.accept_collaboration_request(values[4], context=resolver.context)
        == accepted
    )
    with pytest.raises(CollaborationAccessDenied):
        await application.control_collaboration_request(
            command.model_copy(update={"operation": values[1].operation("new-control")}),
            context=resolver.context,
        )
    await application.drain_collaboration_requests()


@pytest.mark.parametrize("family", ["request", "request_control"])
@pytest.mark.parametrize("malformed", [False, True])
async def test_request_keys_conflict_with_identity_and_lifecycle(stores, family, malformed):
    from tests.core.test_collaboration_namespace import command as lifecycle_command

    from cayu.collaboration._contracts import ExactConflict, ExactUnavailable
    from cayu.collaboration.base import _key
    from cayu.collaboration.participants import CollaborationUnavailable

    store = stores()
    application, resolver, values = await public_setup(store)
    initialized, sender = values[1:3]
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    occupied = accepted
    if family == "request_control":
        occupied = await application.control_collaboration_request(
            RequestControl(
                operation=initialized.operation("control"),
                expected=accepted.expected,
                expected_revision=1,
                kind="cancel",
            ),
            context=resolver.context,
        )
    identity_request, identity_receipt = await create(application, initialized, "probe")
    operation = occupied.expected.operation
    identity_request = identity_request.model_copy(update={"operation": operation})
    identity_expected = identity_receipt.expected.model_copy(
        update={
            "operation": operation,
            "intent": identity_receipt.expected.intent.model_copy(
                update={"request": identity_request}
            ),
        }
    )
    lifecycle_request = change(
        initialized, sender.reference, key=operation.caller_key, revision=1, state="disabled"
    )
    lifecycle_expected = lifecycle_command(initialized, lifecycle_request)
    async with store._transaction(initialized.binding.application_scope, write=True) as tx:
        if malformed:
            raw = await tx.get("operations", _key(occupied.expected))
            del raw["event"]
            await tx.put("operations", _key(occupied.expected), raw, insert=False)
        before = await store._anchor(tx, initialized, REDACTOR)
        retained = await tx.get("operations", _key(occupied.expected))
    outcome = ExactUnavailable if malformed else ExactConflict
    assert isinstance(
        await application.lookup_participant_operation(identity_expected, context=CONTEXT),
        outcome,
    )
    assert isinstance(
        await application.lookup_collaboration_lifecycle_operation(
            lifecycle_expected, context=CONTEXT
        ),
        outcome,
    )
    failure = CollaborationUnavailable if malformed else CollaborationConflict
    with pytest.raises(failure):
        await application.create_participant(identity_request, context=CONTEXT)
    with pytest.raises(failure):
        await application.change_participant_lifecycle(lifecycle_request, context=CONTEXT)
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initialized, REDACTOR) == before
        assert await tx.get("operations", _key(occupied.expected)) == retained
    assert (
        await application.inspect_participant(sender.reference, context=CONTEXT)
    ).participant == sender


async def test_request_replay_requires_only_readback_capability(stores, monkeypatch):
    from cayu.collaboration.base import REQUEST_FAMILY
    from cayu.collaboration.participants import CollaborationUnavailable

    store = stores()
    application, resolver, values = await public_setup(store)
    initialized, request = values[1], values[4]
    accepted = await application.accept_collaboration_request(request, context=resolver.context)
    control = RequestControl(
        operation=initialized.operation("control"),
        expected=accepted.expected,
        expected_revision=1,
        kind="cancel",
    )
    settled = await application.control_collaboration_request(control, context=resolver.context)
    # A separate open request proves that an otherwise valid new control is
    # refused by capability admission, not merely by terminal state.
    open_request = request.model_copy(update={"operation": initialized.operation("open")})
    opened = await application.accept_collaboration_request(open_request, context=resolver.context)
    descriptor = store.capabilities(initialized.owner)
    monkeypatch.setattr(
        store,
        "capabilities",
        lambda owner: descriptor.model_copy(
            update={"mutations": tuple(x for x in descriptor.mutations if x != REQUEST_FAMILY)}
        ),
    )
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        before = await store._anchor(tx, initialized, REDACTOR)
    assert (
        await application.accept_collaboration_request(request, context=resolver.context)
        == accepted
    )
    assert (
        await application.control_collaboration_request(control, context=resolver.context)
        == settled
    )
    assert (
        await application.accept_collaboration_request(open_request, context=resolver.context)
        == opened
    )
    with pytest.raises(CollaborationConflict):
        await application.accept_collaboration_request(
            request.model_copy(update={"content": "Different intent"}), context=resolver.context
        )
    with pytest.raises(CollaborationConflict):
        await application.control_collaboration_request(
            control.model_copy(update={"kind": "expire"}), context=resolver.context
        )
    with pytest.raises(CollaborationUnavailable):
        await application.accept_collaboration_request(
            request.model_copy(update={"operation": initialized.operation("fresh")}),
            context=resolver.context,
        )
    with pytest.raises(CollaborationUnavailable):
        await application.control_collaboration_request(
            control.model_copy(
                update={
                    "operation": initialized.operation("fresh-control"),
                    "expected": opened.expected,
                }
            ),
            context=resolver.context,
        )
    async with store._transaction(initialized.binding.application_scope, write=False) as tx:
        assert await store._anchor(tx, initialized, REDACTOR) == before
    assert (
        await application.inspect_collaboration_request(opened.expected, context=resolver.context)
    ).state == "open"


async def test_settled_request_namespace_can_be_pruned_in_small_batches(stores):
    from tests.core.test_collaboration_namespace import rotate

    from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire

    store = stores()
    application, resolver, values = await public_setup(store)
    accepted = await application.accept_collaboration_request(values[4], context=resolver.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=values[1].operation("control"),
            expected=accepted.expected,
            expected_revision=1,
            kind="cancel",
        ),
        context=resolver.context,
    )
    _, rotated = await rotate(store, values[1])
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    for index in range(10):
        before = await application.inspect_collaboration_namespace(context=CONTEXT)
        result = await application.prune_collaboration_namespace(
            NamespacePrune(
                operation=rotated.successor.reference.operation(f"prune-{index}"),
                namespace=rotated.namespace.reference,
                expected_retention_revision=before.retention_revision,
                max_records=2,
            ),
            context=CONTEXT,
        )
        if result.complete:
            break
    else:
        pytest.fail("Settled request prevented bounded namespace pruning.")
    async with store._transaction(values[1].binding.application_scope, write=False) as tx:
        assert (
            await tx.get(
                "requests",
                (
                    accepted.expected.operation.namespace_incarnation,
                    accepted.expected.operation.generation,
                    accepted.expected.operation.caller_key,
                ),
            )
            is None
        )
    await application.drain_collaboration_requests()


@pytest.mark.parametrize("race", [False, True])
async def test_request_lookup_preserves_partial_history_unavailability(stores, monkeypatch, race):
    from tests.core.test_collaboration_namespace import rotate

    from cayu.collaboration import _request_coordinator as coordinator
    from cayu.collaboration._contracts import ExactNotFound, ExactUnavailable
    from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
    from cayu.collaboration.participants import CollaborationUnavailable

    store = stores()
    application, resolver, values = await public_setup(store)
    initialized = values[1]
    # Keep unrelated history after the request under both byte ordering and
    # locale-aware PostgreSQL ordering. Do not assume punctuation sorts first.
    await create(application, initialized, "zz-retained-history")
    request = values[4].model_copy(update={"operation": initialized.operation("!request")})
    accepted = await application.accept_collaboration_request(request, context=resolver.context)
    control = RequestControl(
        operation=initialized.operation("!control"),
        expected=accepted.expected,
        expected_revision=1,
        kind="cancel",
    )
    terminal = await application.control_collaboration_request(control, context=resolver.context)
    missing_operation = initialized.operation("missing")
    missing = accepted.expected.model_copy(
        update={
            "operation": missing_operation,
            "intent": accepted.expected.intent.model_copy(
                update={"request": request.model_copy(update={"operation": missing_operation})}
            ),
        }
    )
    missing_control = terminal.expected.model_copy(
        update={
            "operation": missing_operation,
            "intent": control.model_copy(update={"operation": missing_operation}),
        }
    )
    for expected in (missing, missing_control):
        assert isinstance(
            await application.lookup_collaboration_request(expected, context=resolver.context),
            ExactNotFound,
        )
    _, rotated = await rotate(store, initialized)
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    caller = None
    if race:
        original_read = coordinator.retained_request
        original_transaction = store._transaction
        paused_owners = set()

        async def read(*args, **kwargs):
            result = await original_read(*args, **kwargs)
            if result is not None:
                paused_owners.add(asyncio.current_task())
            return result

        @asynccontextmanager
        async def transaction(scope, *, write):
            async with original_transaction(scope, write=write) as tx:
                yield tx
            task = asyncio.current_task()
            if task in paused_owners:
                paused_owners.remove(task)
                entered.set()
                await release.wait()

        monkeypatch.setattr(coordinator, "retained_request", read)
        monkeypatch.setattr(store, "_transaction", transaction)
        caller = asyncio.create_task(
            application.lookup_collaboration_request(terminal.expected, context=resolver.context)
        )
    try:
        if caller is not None:
            await asyncio.wait_for(entered.wait(), 5)
        for batch in range(16):
            before = await application.inspect_collaboration_namespace(context=CONTEXT)
            pruned = await application.prune_collaboration_namespace(
                NamespacePrune(
                    operation=rotated.successor.reference.operation(f"prune-{batch}"),
                    namespace=rotated.namespace.reference,
                    expected_retention_revision=before.retention_revision,
                    max_records=2,
                ),
                context=CONTEXT,
            )
            async with store._transaction(initialized.binding.application_scope, write=False) as tx:
                retained = await tx.get(
                    "requests",
                    (
                        accepted.expected.operation.namespace_incarnation,
                        accepted.expected.operation.generation,
                        accepted.expected.operation.caller_key,
                    ),
                )
            if retained is None:
                break
        else:
            pytest.fail("Bounded pruning did not remove the request pair.")
        assert not pruned.complete
        assert pruned.namespace.content == "partial"
        after_pruning = await application.inspect_collaboration_namespace(context=CONTEXT)
        release.set()
        if caller is not None:
            assert isinstance(await asyncio.wait_for(caller, 5), ExactUnavailable)
        for expected in (accepted.expected, terminal.expected, missing, missing_control):
            assert isinstance(
                await application.lookup_collaboration_request(expected, context=resolver.context),
                ExactUnavailable,
            )
        with pytest.raises(CollaborationUnavailable):
            await application.inspect_collaboration_request(
                accepted.expected, context=resolver.context
            )
        with pytest.raises(CollaborationUnavailable):
            await application.accept_collaboration_request(request, context=resolver.context)
        assert after_pruning == await application.inspect_collaboration_namespace(context=CONTEXT)
    finally:
        release.set()
        if caller is not None:
            await asyncio.gather(caller, return_exceptions=True)
        await application.drain_collaboration_requests()
