"""Production receiver qualification through inert creation and public admission."""

import asyncio
from contextlib import asynccontextmanager

import pytest
from tests.core.test_budget_binding import _binding
from tests.core.test_collaboration_request_foundation import RequestResolver, setup
from tests.core.test_participant_identity import CONTEXT, app

from cayu.agents import AgentSpec
from cayu.collaboration._clarification_state import clarification_commitment
from cayu.collaboration._contracts import ExactMatch, ObjectRef
from cayu.collaboration._request_coordinator import _initiator
from cayu.collaboration.mandates import MandateResolver
from cayu.collaboration.memory import InMemoryCollaborationStore
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration
from cayu.collaboration.requests import RequestAdmissionCommand
from cayu.evals.testing import ScriptedModelProvider
from cayu.messages import Message
from cayu.sessions import RunRequest
from cayu.sessions.base import InMemorySessionStore
from cayu.sessions.context_views import RecipientSessionCreationRequest
from cayu.vaults.redaction import SecretRedactor

pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("kind", ["cancel", "expire"])
async def test_admission_lookup_control_key_conflict(native_stores, monkeypatch, kind):
    from cayu.collaboration._contracts import ExactConflict, ExactUnavailable
    from cayu.collaboration.access import CollaborationAccessDenied
    from cayu.collaboration.requests import RequestControl

    application, resolver, command, _, _, initialized = await prepared_scenario(native_stores)
    store = native_stores[0]
    original = store._transaction

    @asynccontextmanager
    async def at_deadline(scope, *, write):
        async with original(scope, write=write) as tx:

            async def now_ms():
                return command.expected.intent.selection.expires_at_ms

            tx.now_ms = now_ms
            yield tx

    if kind == "expire":
        monkeypatch.setattr(store, "_transaction", at_deadline)
    control = await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("control-collision"),
            expected=command.expected,
            expected_revision=1,
            kind=kind,
        ),
        context=resolver.sender.context,
    )
    monkeypatch.setattr(store, "_transaction", original)
    candidate = command.model_copy(update={"operation": control.expected.operation})
    policy = application._participant_coordinator._registration.access_policy
    selected = command.expected.intent.selection
    for allowed in (None, (selected.sender.reference, selected.recipient.reference)):
        policy.allowed = allowed
        assert isinstance(
            await application.lookup_collaboration_admission(
                candidate, context=resolver.sender.context
            ),
            ExactConflict,
        )
    policy.allowed = (selected.sender.reference,)
    with pytest.raises(CollaborationAccessDenied):
        await application.lookup_collaboration_admission(candidate, context=resolver.sender.context)
    policy.allowed = None
    async with original(initialized.owner.application_scope, write=True) as tx:
        await tx.delete("request_events", (control.event.sequence,))
    assert isinstance(
        await application.lookup_collaboration_admission(
            candidate, context=resolver.sender.context
        ),
        ExactUnavailable,
    )


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("replacement", ["changed", "removed"])
async def test_admission_replay_after_receiver_reconfiguration(
    native_stores, monkeypatch, committed, replacement
):
    from dataclasses import replace

    from cayu.collaboration._contracts import CollaborationConflict, ExactMatch
    from cayu.collaboration.access import CollaborationAccessDenied
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, provider, session, _ = await prepared_scenario(native_stores)
    registration = application._request_coordinator._registration
    if committed:
        original = native_stores[0]._transaction
        lost = False

        @asynccontextmanager
        async def lost_ack(scope, *, write):
            nonlocal lost
            async with original(scope, write=write) as tx:
                yield tx
            if write and not lost:
                lost = True
                raise RuntimeError("lost committed admission acknowledgement")

        monkeypatch.setattr(native_stores[0], "_transaction", lost_ack)
        with pytest.raises(CollaborationUnavailable):
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
        monkeypatch.setattr(native_stores[0], "_transaction", original)
        assert lost
    current = RequestRegistration(mandates=resolver, max_ttl_ms=300_000)
    if replacement == "changed":
        current = replace(
            current,
            prepared_admission=PreparedAdmissionRegistration(
                receiver=registration.prepared_admission.receiver.model_copy(update={"revision": 2})
            ),
        )
    reopened = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        collaboration_requests=current,
    )
    await reopened.initialize_collaboration()
    before = await reopened.inspect_participant(command.prepared.recipient, context=CONTEXT)
    if committed:
        found = await reopened.lookup_collaboration_admission(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch)
        assert (
            await reopened.admit_collaboration_request(command, context=resolver.recipient.context)
            == found.receipt
        )
        with pytest.raises(CollaborationConflict):
            await reopened.admit_collaboration_request(
                command.model_copy(update={"proposal_commitment": "conflicting-retry"}),
                context=resolver.recipient.context,
            )
        policy = reopened._participant_coordinator._registration.access_policy
        policy.allowed = (command.expected.intent.selection.sender.reference,)
        with pytest.raises(CollaborationAccessDenied):
            await reopened.admit_collaboration_request(command, context=resolver.recipient.context)
        policy.allowed = None
    else:
        with pytest.raises(
            CollaborationConflict if replacement == "changed" else CollaborationUnavailable
        ):
            await reopened.admit_collaboration_request(command, context=resolver.recipient.context)
        # Refusal must not consume authority: the originally configured receiver
        # can still admit this untouched operation.
        assert (
            await reopened.inspect_participant(command.prepared.recipient, context=CONTEXT)
            == before
        )
        assert (
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
        ).state == "admitted"
    if committed:
        assert (
            await reopened.inspect_participant(command.prepared.recipient, context=CONTEXT)
            == before
        )
    assert provider.requests == []
    assert (await native_stores[1].load(session.id)).status == "pending"


async def test_public_preparation_rejects_malformed_budget_receiver_without_diagnostics(
    native_stores, monkeypatch, caplog, capsys
):
    import warnings

    from cayu.collaboration._contracts import CollaborationContractError
    from cayu.collaboration.prepared_admission import prepared_budget

    application, resolver, command, provider, session, initialized = await prepared_scenario(
        native_stores
    )
    creation = RecipientSessionCreationRequest(
        request=RunRequest(agent_name="reviewer", messages=[Message.text("user", "input")]),
        creation_key="prepared-child:" + initialized.owner.application_scope,
        recipient=command.prepared.recipient,
    )
    original = prepared_budget(command.prepared.budget_binding_json)
    returned = original
    receiver = application._run_limit_controller._budget_binding_receiver

    async def resolve(*, request):
        return returned

    monkeypatch.setattr(receiver, "resolve_budget_binding", resolve)
    before = await application.inspect_collaboration_request(
        command.expected, context=resolver.sender.context
    )
    before_session = await native_stores[1].load(session.id)
    secret = "malformed-receiver-private-canary"

    class Hostile:
        def __repr__(self):
            return secret

        def __str__(self):
            return secret

    invalid = (
        original.model_copy(update={"purpose": [secret]}),
        original.model_copy(update={"purpose": Hostile()}),
        original.model_copy(update={"purpose": [secret], "sponsor": secret}),
        original.model_copy(
            update={"limits": (original.limits[0].model_copy(update={"currency": [secret]}),)}
        ),
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for candidate in invalid:
            returned = candidate
            with pytest.raises(CollaborationContractError) as caught:
                await application.prepare_recipient_admission(creation, context=CONTEXT)
            assert caught.value.__cause__ is None
            assert caught.value.__context__ is None
            assert secret not in str(caught.value) + repr(caught.value)
    assert not captured
    assert secret not in caplog.text
    streams = capsys.readouterr()
    assert secret not in streams.out + streams.err
    assert await native_stores[1].load(session.id) == before_session
    assert (
        await application.inspect_collaboration_request(
            command.expected, context=resolver.sender.context
        )
        == before
    )
    returned = original
    assert (
        await application.prepare_recipient_admission(creation, context=CONTEXT) == command.prepared
    )
    assert (
        await application.admit_collaboration_request(command, context=resolver.recipient.context)
    ).state == "admitted"
    assert provider.requests == []


async def test_participant_scoped_reader_classifies_sibling_request_key(native_stores):
    from cayu.collaboration._contracts import ExactConflict
    from cayu.collaboration.access import CollaborationAccessDenied
    from cayu.collaboration.requests import RequestObservation

    application, resolver, command, _, _, _ = await prepared_scenario(native_stores)
    observed = await application.register_collaboration_observation(
        command.expected,
        RequestObservation(
            key="reader",
            filter_commitment="all",
            projection_commitment="events",
            after_sequence=0,
            coverage_sequence=0,
            revision=1,
        ),
        context=resolver.sender.context,
    )
    policy = application._participant_coordinator._registration.access_policy
    selected = command.expected.intent.selection
    policy.allowed = (selected.sender.reference, selected.recipient.reference)
    candidate = command.model_copy(update={"operation": observed.operation})
    found = await application.lookup_collaboration_admission(
        candidate, context=resolver.sender.context
    )
    assert isinstance(found, ExactConflict)
    policy.allowed = (selected.sender.reference,)
    with pytest.raises(CollaborationAccessDenied):
        await application.lookup_collaboration_admission(candidate, context=resolver.sender.context)


async def test_exact_comparison_binds_every_admission_leaf(native_stores):
    from copy import deepcopy

    from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
    from cayu.collaboration._preparation import require_exact_contract

    application, resolver, command, _, _, _ = await prepared_scenario(native_stores)
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    document = receipt.command.model_dump(mode="json")

    def leaves(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from leaves(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                yield from leaves(child, (*path, index))
        else:
            yield path, value

    compared = 0
    for path, value in leaves(document):
        if path[0] == "operation":
            continue  # Hold the complete scoped replay identity constant.
        changed = deepcopy(document)
        target = changed
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = (
            not value
            if type(value) is bool
            else value + 1
            if type(value) is int
            else "changed"
            if value is None
            else value + "-changed"
        )
        # Invalid coupled fields must fail validation; independently valid
        # changes must conflict. Neither may reconcile as an exact match.
        candidate = RequestAdmissionCommand.model_construct(**changed)
        with pytest.raises((CollaborationConflict, CollaborationContractError)):
            require_exact_contract(receipt.command, candidate, redactor=SecretRedactor())
        compared += 1
    assert compared > 100


async def test_prepared_command_canonical_byte_boundary(native_stores):
    from cayu._validation import canonical_durable_json_bytes
    from cayu.collaboration._contracts import CollaborationContractError
    from cayu.collaboration.prepared_admission import MAX_PREPARED_ADMISSION_BYTES

    _, _, command, _, _, _ = await prepared_scenario(native_stores)
    material = command.model_dump(mode="json")
    material["expected"]["intent"]["request"]["content"] = ""
    base = len(canonical_durable_json_bytes(material, "command"))
    for offset in (-1, 0, 1):
        remaining = MAX_PREPARED_ADMISSION_BYTES + offset - base
        # JSON escaping counts toward the byte bound independently of the
        # accepted request's character bound. No source record is fabricated:
        # these are pure proposal-validation tests, not receiving authority.
        assert 0 <= remaining <= 6 * 8192
        material["expected"]["intent"]["request"]["content"] = "\x01" * (remaining // 6) + "x" * (
            remaining % 6
        )
        assert (
            len(canonical_durable_json_bytes(material, "command"))
            == MAX_PREPARED_ADMISSION_BYTES + offset
        )
        if offset > 0:
            with pytest.raises((CollaborationContractError, ValueError)):
                RequestAdmissionCommand.model_validate(material)
        else:
            assert RequestAdmissionCommand.model_validate(material).prepared == command.prepared


async def test_escaped_secret_in_snapshot_is_rejected_before_publication(
    native_stores, caplog, capsys
):
    import json

    from cayu._validation import canonical_durable_json_bytes
    from cayu.collaboration._contracts import CollaborationContractError, ExactNotFound

    application, resolver, command, _, _, _ = await prepared_scenario(native_stores)
    secret = 'private-preparation-"quote\\slash'
    receiver = application._request_coordinator._registration.receiving_owner
    receiver._redactor = receiver._redactor.with_secret(secret)
    binding = json.loads(command.prepared.budget_binding_json)
    binding["purpose"] = secret
    encoded = canonical_durable_json_bytes(binding, "binding").decode()
    assert secret not in encoded
    candidate = command.model_copy(
        update={
            "prepared": command.prepared.model_copy(update={"budget_binding_json": encoded}),
        }
    )
    with pytest.raises(CollaborationContractError) as caught:
        await application.admit_collaboration_request(candidate, context=resolver.recipient.context)
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value.__cause__)
    streams = capsys.readouterr()
    assert secret not in caplog.text + streams.out + streams.err
    assert isinstance(
        await application.lookup_collaboration_admission(command, context=resolver.sender.context),
        ExactNotFound,
    )


async def test_failure_after_local_permit_settlement_rolls_back_whole_admission(
    native_stores, monkeypatch
):
    from cayu.collaboration._contracts import ExactNotFound
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, _, _, _ = await prepared_scenario(native_stores)
    before = await application.inspect_participant(command.prepared.recipient, context=CONTEXT)
    store = native_stores[0]
    original = store._transaction
    injected = False

    @asynccontextmanager
    async def failing(scope, *, write):
        nonlocal injected
        async with original(scope, write=write) as tx:
            put = tx.put

            async def fail_snapshot(family, key, value, *, insert):
                nonlocal injected
                await put(family, key, value, insert=insert)
                if family == "requests" and getattr(value, "admission", None) == "admitted":
                    injected = True
                    raise RuntimeError("admission publication failed after local permit settlement")

            if write:
                tx.put = fail_snapshot
            yield tx

    monkeypatch.setattr(store, "_transaction", failing)
    with pytest.raises(CollaborationUnavailable):
        await application.admit_collaboration_request(command, context=resolver.recipient.context)
    assert injected
    assert (
        await application.inspect_participant(command.prepared.recipient, context=CONTEXT) == before
    )
    assert isinstance(
        await application.lookup_collaboration_admission(command, context=resolver.sender.context),
        ExactNotFound,
    )
    monkeypatch.setattr(store, "_transaction", original)
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    assert receipt.admission_permit is not None
    assert receipt.admission_permit.position == before.issued_permit_frontier + 1


class PreparationResolver(MandateResolver):
    """Two independently authenticated principals, held through owner use."""

    def __init__(self, request, recipient):
        self.sender = RequestResolver(request)
        self.recipient = RequestResolver(request.model_copy(update={"sender": recipient}))
        for resolver in (self.sender, self.recipient):
            original = resolver.resolution
            actions = (*original.principal.actions, "prepare")
            resolver.resolution = original.model_copy(
                update={
                    "principal": original.principal.model_copy(update={"actions": actions}),
                    "chain": original.chain.model_copy(
                        update={
                            "entries": tuple(
                                entry.model_copy(update={"actions": actions})
                                for entry in original.chain.entries
                            ),
                        }
                    ),
                }
            )

    @property
    def ref(self):
        return self.sender.ref

    @asynccontextmanager
    async def acquire(self, context):
        resolver = self.sender if context == self.sender.context else self.recipient
        async with resolver.acquire(context) as resolution:
            yield resolution


@pytest.fixture(params=["memory", "sqlite", "postgres"])
async def native_stores(request, tmp_path):
    opened = []
    if request.param == "memory":
        collaboration, sessions = InMemoryCollaborationStore(), InMemorySessionStore()
        address = None

        def factory():
            return collaboration
    elif request.param == "sqlite":
        from cayu.storage.collaboration_sqlite import SQLiteCollaborationStore
        from cayu.storage.sqlite import SQLiteSessionStore

        collaboration = SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
        address = str(tmp_path / "collaboration.sqlite")
        sessions = SQLiteSessionStore(tmp_path / "sessions.sqlite")

        def factory():
            return SQLiteCollaborationStore(tmp_path / "collaboration.sqlite")
    else:
        from cayu.storage.collaboration_postgres import PostgresCollaborationStore
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        dsn = request.getfixturevalue("postgres_dsn")
        address = dsn
        collaboration = PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)
        sessions = PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

        def factory():
            return PostgresCollaborationStore(dsn, schema_mode=SchemaMode.CREATE)

    def independent():
        other = factory()
        if other is not collaboration:
            opened.append(other)
        return other

    yield collaboration, sessions, independent, (request.param, address)
    for other in opened:
        await other.close()
    await collaboration.close()
    if hasattr(sessions, "close"):
        await sessions.close()


async def prepared_scenario(
    native_stores,
    *,
    use_example=False,
    provider_events=(),
    request_ttl_ms=None,
    session_exports_factory=None,
    provider_factory=ScriptedModelProvider,
    tools=(),
    tool_policy=None,
    loop_policies=(),
    config=None,
    budget_ledger=None,
    budget_binding_factory=None,
    operation_prefix="",
    planned=False,
    planning_driver=None,
    request_cancellation=None,
    requested_session_id=None,
):
    collaboration, sessions, *_ = native_stores
    original, initialized, _, recipient, request, _initiating = await setup(collaboration)
    if request_cancellation is not None:
        request = request.model_copy(update={"cancellation": request_cancellation})
    if operation_prefix:
        request = request.model_copy(
            update={
                "operation": initialized.operation(operation_prefix + request.operation.caller_key)
            }
        )
    if session_exports_factory is not None:
        request = request.model_copy(update={"ttl_ms": 300_000})
    if request_ttl_ms is not None:
        request = request.model_copy(update={"ttl_ms": request_ttl_ms})
    resolver = PreparationResolver(request, recipient.reference)
    binding = (
        _binding(application_scope=initialized.owner.application_scope)
        if budget_binding_factory is None
        else budget_binding_factory(initialized.owner.application_scope)
    )

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return binding

    exports = (
        None
        if session_exports_factory is None
        else session_exports_factory(initialized, request, resolver)
    )

    def build_application(planning_policies=()):
        return app(
            collaboration,
            original._participant_coordinator._registration,
            config=config,
            session_store=sessions,
            collaboration_requests=RequestRegistration(
                mandates=resolver,
                planning_policies=planning_policies,
                max_ttl_ms=max(300_000, request_ttl_ms or 0),
                prepared_admission=PreparedAdmissionRegistration(
                    receiver=ObjectRef(
                        owner=initialized.owner,
                        kind="request_receiver",
                        object_id="native-fresh",
                        incarnation="one",
                        revision=1,
                    )
                ),
            ),
            budget_binding_receiver=BudgetReceiver(),
            budget_ledger=budget_ledger,
            enable_common_root_budget_binding=True,
            session_exports=exports,
        )

    application = build_application()
    if planned:
        from tests.core._execution_profile_fixtures import versioned_test_provider_identity

        class PlannedProvider(provider_factory):
            @property
            def execution_profile_identity(self):
                return versioned_test_provider_identity(self)

        provider_factory = PlannedProvider
    provider = provider_factory(provider_events, name="provider")

    def register_execution(application):
        application.register_provider(provider, default=True)
        application.register_agent(
            AgentSpec(name="reviewer", model="model", system_prompt="system"),
            tools=tools,
            tool_policy=tool_policy,
            loop_policies=loop_policies,
        )

    register_execution(application)
    await application.initialize_collaboration()
    accepted = await application.accept_collaboration_request(
        request, context=resolver.sender.context
    )
    creation = RecipientSessionCreationRequest(
        request=RunRequest(
            agent_name="reviewer",
            session_id=requested_session_id,
            messages=[Message.text("user", "input")],
        ),
        creation_key="prepared-child:" + operation_prefix + initialized.owner.application_scope,
        recipient=recipient.reference,
    )
    if planned:
        from examples.collaboration.planning import fresh_policy
        from tests.core.test_request_planning_contracts import _policy
        from tests.core.test_request_planning_public import complete_plan

        from cayu.collaboration._planning_store import admission_command
        from cayu.collaboration.planning import RequestPlanningRequest, planning_policy_commitment

        preparation = await application.prepare_recipient_creation(creation, context=CONTEXT)
        policy = fresh_policy(
            _policy().reference.model_copy(update={"owner": initialized.owner}),
            _policy().limits,
            preparation,
        )
        application = build_application((policy,))
        register_execution(application)
        await application.initialize_collaboration()
        planning = RequestPlanningRequest(
            operation=initialized.operation(operation_prefix + "producer-plan"),
            expected=accepted.expected,
            expected_revision=1,
            expected_input_revision=0,
            expected_input_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
            planning_generation=1,
            admission_operation=initialized.operation(operation_prefix + "prepared-admission"),
            admission_generation=1,
            initiator=_initiator(resolver.recipient.context),
            policy=policy.reference,
            policy_sha256=planning_policy_commitment(policy, redactor=SecretRedactor()),
            limits=policy.limits,
            deadline_at_ms=accepted.expected.intent.selection.expires_at_ms,
            predecessor=None,
        )
        retained = await (planning_driver or complete_plan)(
            application, planning, resolver.recipient.context
        )
        assert retained.state == "admitted" and retained.pending_stages == 0
        assert provider.requests == []
        child = await application.lookup_recipient_session(creation, context=CONTEXT)
        assert child is not None
        session, _ = child
        evidence = await application.prepare_recipient_admission(creation, context=CONTEXT)
        command = admission_command(retained, SecretRedactor(), prepared=evidence)
        found = await application.lookup_collaboration_admission(
            command, context=resolver.recipient.context
        )
        assert isinstance(found, ExactMatch) and found.receipt.state == "admitted"
        assert await complete_plan(application, planning, resolver.recipient.context) == retained
        return application, resolver, command, provider, session, initialized

    session, _ = await application.create_recipient_session(creation, context=CONTEXT)
    evidence = await application.prepare_recipient_admission(creation, context=CONTEXT)
    command = RequestAdmissionCommand(
        operation=initialized.operation(operation_prefix + "prepared-admission"),
        expected=accepted.expected,
        expected_revision=1,
        expected_input_revision=0,
        expected_input_sha256=clarification_commitment(accepted.expected, SecretRedactor()),
        generation=1,
        decision="fresh",
        prepared=evidence,
        evidence=(),
        initiator=_initiator(resolver.recipient.context),
    )
    if use_example:
        from examples.collaboration.prepared_admission import admit_prepared_recipient

        await admit_prepared_recipient(
            application,
            creation,
            command,
            creation_context=CONTEXT,
            receiving_context=resolver.recipient.context,
        )
    return application, resolver, command, provider, session, initialized


async def test_public_fresh_admission_and_exact_readback_without_dispatch(native_stores):
    application, resolver, command, provider, session, _ = await prepared_scenario(
        native_stores, use_example=True
    )
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    assert receipt.state == "admitted"
    assert receipt.admission_permit is not None
    assert (
        await application.admit_collaboration_request(command, context=resolver.recipient.context)
        == receipt
    )
    found = await application.lookup_collaboration_admission(
        command, context=resolver.sender.context
    )
    assert isinstance(found, ExactMatch)
    assert found.receipt == receipt
    assert (
        await application.collaboration_admission_reader().lookup(
            command, context=resolver.sender.context
        )
        == found
    )
    assert provider.requests == []
    assert (await native_stores[1].load(session.id)).status == "pending"


@pytest.mark.parametrize("kind", ["cancel", "expire"])
async def test_inert_admission_can_cancel_after_recipient_disable(native_stores, monkeypatch, kind):
    from tests.core.test_participant_lifecycle import change

    from cayu.collaboration.requests import RequestControl

    application, resolver, command, provider, session, initialized = await prepared_scenario(
        native_stores
    )
    await application.admit_collaboration_request(command, context=resolver.recipient.context)
    await application.change_participant_lifecycle(
        change(
            initialized,
            command.prepared.recipient,
            key="disable-for-cleanup",
            revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    control = RequestControl(
        operation=initialized.operation("cancel-inert"),
        expected=command.expected,
        expected_revision=2,
        kind=kind,
    )
    # Cleanup after reconstruction needs no live native child, provider or
    # sponsor receiver; only the registered identity and current admin mandate.
    application = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(
            mandates=resolver,
            max_ttl_ms=300_000,
            prepared_admission=application._request_coordinator._registration.prepared_admission,
        ),
    )
    await application.initialize_collaboration()
    if kind == "expire":
        cleanup_store, _ = application._participant_coordinator._ready()
        original = cleanup_store._transaction

        @asynccontextmanager
        async def at_deadline(scope, *, write):
            async with original(scope, write=write) as tx:

                async def now_ms():
                    return command.expected.intent.selection.expires_at_ms

                tx.now_ms = now_ms
                yield tx

        monkeypatch.setattr(cleanup_store, "_transaction", at_deadline)
    receipt = await application.control_collaboration_request(
        control, context=resolver.sender.context
    )
    assert receipt.state == ("cancelled" if kind == "cancel" else "expired")
    assert (
        await application.control_collaboration_request(control, context=resolver.sender.context)
        == receipt
    )
    assert (
        await application.inspect_participant(command.prepared.recipient, context=CONTEXT)
    ).outstanding_obligations == 0
    assert provider.requests == []
    assert (await native_stores[1].load(session.id)).status == "pending"


@pytest.mark.parametrize("disable_first", [False, True])
async def test_independent_lifecycle_owner_orders_against_admission(native_stores, disable_first):
    from tests.core.test_participant_lifecycle import change

    from cayu.collaboration._contracts import CollaborationConflict, ExactNotFound

    application, resolver, command, provider, session, initialized = await prepared_scenario(
        native_stores
    )
    other = app(native_stores[2](), application._participant_coordinator._registration)
    await other.initialize_collaboration()
    receipt = None
    if not disable_first:
        receipt = await application.admit_collaboration_request(
            command, context=resolver.recipient.context
        )
    await other.change_participant_lifecycle(
        change(
            initialized,
            command.prepared.recipient,
            key="disable-recipient",
            revision=1,
            state="disabled",
        ),
        context=CONTEXT,
    )
    if disable_first:
        with pytest.raises(CollaborationConflict):
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
        found = await application.lookup_collaboration_admission(
            command, context=resolver.sender.context
        )
        assert isinstance(found, ExactNotFound)
    else:
        assert (
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
            == receipt
        )
        found = await application.lookup_collaboration_admission(
            command, context=resolver.sender.context
        )
        assert isinstance(found, ExactMatch)
        assert found.receipt == receipt
    assert provider.requests == []
    assert (await native_stores[1].load(session.id)).status == "pending"


async def test_reconstructed_readback_conflicts_without_budget_or_native_target_reads(
    native_stores,
):
    from cayu.collaboration._contracts import ExactConflict

    application, resolver, command, provider, _, _ = await prepared_scenario(native_stores)
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    reconstructed = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        collaboration_requests=RequestRegistration(mandates=resolver, max_ttl_ms=300_000),
    )
    await reconstructed.initialize_collaboration()
    reader = reconstructed.collaboration_admission_reader()
    found = await reader.lookup(command, context=resolver.sender.context)
    assert isinstance(found, ExactMatch)
    assert found.receipt == receipt
    for updates in (
        {"expected_revision": 2},
        {"expected_input_revision": 1},
        {"expected_input_sha256": "d" * 64},
        {"generation": 2},
        {"proposal_commitment": "changed-proposal"},
        {"prepared": command.prepared.model_copy(update={"configuration_revision": 2})},
        {
            "prepared": command.prepared.model_copy(
                update={
                    "target": command.prepared.target.model_copy(
                        update={"session_instance_id": "replacement"}
                    ),
                }
            )
        },
    ):
        conflict = await reader.lookup(
            command.model_copy(update=updates), context=resolver.sender.context
        )
        assert isinstance(conflict, ExactConflict)
    assert provider.requests == []


@pytest.mark.parametrize("signal", ["lost_ack", "cancel", "repeated_cancel", "timeout"])
async def test_admission_commit_survives_interrupted_acknowledgement(
    native_stores, monkeypatch, signal
):
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, provider, _, _ = await prepared_scenario(native_stores)
    store = native_stores[0]
    original = store._transaction
    committed = asyncio.Event()
    release = asyncio.Event()
    faulted = False
    key = (
        command.operation.namespace_incarnation,
        command.operation.generation,
        command.operation.caller_key,
    )

    @asynccontextmanager
    async def transaction(scope, *, write):
        nonlocal faulted
        interrupt = False
        async with original(scope, write=write) as tx:
            yield tx
            if write and not faulted and await tx.get("operations", key) is not None:
                faulted = interrupt = True
        if interrupt:
            committed.set()
            await release.wait()
            if signal == "lost_ack":
                raise RuntimeError("injected lost admission acknowledgement")

    monkeypatch.setattr(store, "_transaction", transaction)
    operation = asyncio.create_task(
        application.admit_collaboration_request(
            command,
            context=resolver.recipient.context,
        )
    )
    try:
        await asyncio.wait_for(committed.wait(), 20)
        if signal in {"cancel", "repeated_cancel"}:
            count = 2 if signal == "repeated_cancel" else 1
            for _ in range(count):
                operation.cancel()
            assert operation.cancelling() == count
            with pytest.raises(asyncio.CancelledError):
                await operation
            assert operation.cancelled()
        if signal == "timeout":
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(operation, 0.01)
            assert operation.cancelling() == 1
            assert operation.cancelled()
        release.set()
        if signal == "lost_ack":
            with pytest.raises(CollaborationUnavailable):
                await operation
        found = await application.lookup_collaboration_admission(
            command, context=resolver.sender.context
        )
        assert isinstance(found, ExactMatch)
        assert (
            await application.admit_collaboration_request(
                command, context=resolver.recipient.context
            )
            == found.receipt
        )
        assert provider.requests == []
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


async def test_prepared_admission_prunes_parent_before_local_permit(native_stores):
    from tests.core.test_collaboration_namespace import rotate

    from cayu.collaboration.lifecycle import NamespacePrune, NamespaceRetire
    from cayu.collaboration.requests import RequestControl

    application, resolver, command, _, _, initialized = await prepared_scenario(native_stores)
    command = command.model_copy(update={"operation": initialized.operation("zzz-admission")})
    await application.admit_collaboration_request(command, context=resolver.recipient.context)
    await application.control_collaboration_request(
        RequestControl(
            operation=initialized.operation("cancel"),
            expected=command.expected,
            expected_revision=2,
            kind="cancel",
        ),
        context=resolver.sender.context,
    )
    _, rotated = await rotate(native_stores[0], initialized)
    await application.retire_collaboration_namespace(
        NamespaceRetire(
            operation=rotated.successor.reference.operation("retire"),
            namespace=rotated.namespace.reference,
            expected_revision=rotated.namespace.revision,
            expected_retired_through=0,
        ),
        context=CONTEXT,
    )
    for index in range(24):
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
        pytest.fail("Prepared admission stranded namespace reclamation")


async def test_disablement_during_receiver_read_cannot_admit_stale_authority(
    native_stores, monkeypatch
):
    from tests.core.test_participant_lifecycle import change

    from cayu.collaboration._contracts import CollaborationConflict, ExactNotFound

    application, resolver, command, provider, _, initialized = await prepared_scenario(
        native_stores
    )
    receiver = application._request_coordinator._registration.receiving_owner
    original = receiver._native_evidence
    read = asyncio.Event()
    release = asyncio.Event()

    async def paused(value):
        await original(value)
        read.set()
        await release.wait()

    monkeypatch.setattr(receiver, "_native_evidence", paused)
    operation = asyncio.create_task(
        application.admit_collaboration_request(
            command,
            context=resolver.recipient.context,
        )
    )
    try:
        await asyncio.wait_for(read.wait(), 20)
        other = app(native_stores[2](), application._participant_coordinator._registration)
        await other.initialize_collaboration()
        await other.change_participant_lifecycle(
            change(
                initialized,
                command.prepared.recipient,
                key="racing-disable",
                revision=1,
                state="disabled",
            ),
            context=CONTEXT,
        )
        before = await other.inspect_participant(command.prepared.recipient, context=CONTEXT)
        release.set()
        with pytest.raises(CollaborationConflict):
            await operation
        after = await other.inspect_participant(command.prepared.recipient, context=CONTEXT)
        assert after == before
        found = await application.lookup_collaboration_admission(
            command, context=resolver.sender.context
        )
        assert isinstance(found, ExactNotFound)
        state = await application.inspect_collaboration_request(
            command.expected, context=resolver.sender.context
        )
        assert state.revision == 1
        assert state.admission == "undecided"
        assert provider.requests == []
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.parametrize("same_key", [True, False])
async def test_competing_receivers_share_one_durable_admission(native_stores, same_key):
    from dataclasses import replace

    from cayu.collaboration._contracts import CollaborationConflict
    from cayu.collaboration.prepared_admission import prepared_budget

    application, resolver, command, provider, _, initialized = await prepared_scenario(
        native_stores
    )

    class BudgetReceiver:
        async def resolve_budget_binding(self, *, request):
            return prepared_budget(command.prepared.budget_binding_json)

    other = app(
        native_stores[2](),
        application._participant_coordinator._registration,
        session_store=native_stores[1],
        collaboration_requests=replace(
            application._request_coordinator._registration, receiving_owner=None
        ),
        budget_binding_receiver=BudgetReceiver(),
        enable_common_root_budget_binding=True,
    )
    await other.initialize_collaboration()
    competing = (
        command
        if same_key
        else command.model_copy(
            update={
                "operation": initialized.operation("competing-admission"),
            }
        )
    )
    outcomes = await asyncio.gather(
        application.admit_collaboration_request(command, context=resolver.recipient.context),
        other.admit_collaboration_request(competing, context=resolver.recipient.context),
        return_exceptions=True,
    )
    if same_key:
        assert outcomes[0] == outcomes[1]
        assert not isinstance(outcomes[0], BaseException)
    else:
        assert sum(isinstance(value, CollaborationConflict) for value in outcomes) == 1
        assert sum(not isinstance(value, BaseException) for value in outcomes) == 1
    snapshot = await application.inspect_collaboration_request(
        command.expected, context=resolver.sender.context
    )
    assert snapshot.revision == 2
    assert snapshot.admission_generation == 1
    assert provider.requests == []


async def test_exact_readback_in_fresh_process(native_stores):
    import json
    import sys

    from cayu.collaboration.requests import RequestAdmissionReceipt

    backend, address = native_stores[3]
    if backend == "memory":
        # Logical reconstruction is covered separately; an in-memory store
        # deliberately does not claim cross-process persistence.
        return
    application, resolver, command, provider, _, _ = await prepared_scenario(native_stores)
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "tests.recovery.prepared_admission_reader_worker",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            child.communicate(
                json.dumps(
                    {
                        "backend": backend,
                        "address": address,
                        "expected": command.model_dump(mode="json"),
                    }
                ).encode()
            ),
            45,
        )
        assert child.returncode == 0, stderr.decode()
        reconstructed = ExactMatch[RequestAdmissionReceipt].model_validate_json(stdout)
        assert reconstructed.receipt == receipt
        assert provider.requests == []
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


async def test_public_rejects_untrusted_native_claims_and_hostile_mutation(
    native_stores, caplog, capsys
):
    import warnings

    from cayu.collaboration._contracts import CollaborationConflict, CollaborationContractError
    from cayu.collaboration.access import CollaborationAccessDenied
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, provider, _, _ = await prepared_scenario(native_stores)
    before = await application.inspect_collaboration_request(
        command.expected, context=resolver.sender.context
    )
    prepared = command.prepared
    assert prepared is not None

    class Hostile:
        def __repr__(self):
            raise AssertionError("private-preparation-canary repr invoked")

        def __str__(self):
            raise AssertionError("private-preparation-canary str invoked")

        def model_dump(self, **kwargs):
            raise AssertionError("private-preparation-canary serializer invoked")

    invalid = (
        prepared.model_copy(update={"configuration_revision": True}),
        prepared.model_copy(update={"schema_version": 2}),
        prepared.model_copy(update={"budget_binding_json": Hostile()}),
        prepared.model_copy(
            update={"target": prepared.target.model_copy(update={"session_instance_id": "wrong"})}
        ),
        prepared.model_copy(
            update={
                "target": prepared.target.model_copy(
                    update={"initial_input_commitment": "sha256:" + "e" * 64}
                )
            }
        ),
        prepared.model_copy(
            update={
                "target": prepared.target.model_copy(
                    update={"definition_commitment": "sha256:" + "e" * 64}
                )
            }
        ),
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for candidate in invalid:
            with pytest.raises(
                (
                    CollaborationContractError,
                    CollaborationConflict,
                    CollaborationAccessDenied,
                    CollaborationUnavailable,
                )
            ):
                await application.admit_collaboration_request(
                    command.model_copy(update={"prepared": candidate}),
                    context=resolver.recipient.context,
                )
    assert not captured
    assert "private-preparation-canary" not in caplog.text
    streams = capsys.readouterr()
    assert "private-preparation-canary" not in streams.out + streams.err
    assert (
        await application.inspect_collaboration_request(
            command.expected, context=resolver.sender.context
        )
        == before
    )
    assert provider.requests == []


async def test_composed_registration_preserves_real_export_answer(native_stores, tmp_path):
    from tests.core.test_collaboration_request_exports import _admit, _integration

    from cayu.collaboration.requests import RequestOutcomeCommand

    backend, address = native_stores[3]
    case = await _integration(
        native_stores[0],
        tmp_path,
        address if backend == "postgres" else None,
        prepared_receiver=True,
    )
    try:
        accepted, initiator, _ = await _admit(case)
        outcome = RequestOutcomeCommand(
            operation=case.values[1].operation("answer-through-composition"),
            expected=accepted.expected,
            expected_revision=2,
            outcome="answered",
            commitment=case.source.expected.intent.output_commitment,
            source_receipt=case.source,
            initiator=initiator,
        )
        receipt = await case.app.publish_collaboration_outcome(outcome, context=case.context)
        assert receipt.command.outcome == "answered"
        assert (
            await case.app.publish_collaboration_outcome(outcome, context=case.context) == receipt
        )
    finally:
        await case.app.drain_collaboration_requests()
        if hasattr(case.session_store, "close"):
            await case.session_store.close()


async def test_missing_durable_admission_permit_evidence_is_unavailable(native_stores):
    from cayu.collaboration._contracts import ExactUnavailable

    application, resolver, command, _, _, initialized = await prepared_scenario(native_stores)
    receipt = await application.admit_collaboration_request(
        command, context=resolver.recipient.context
    )
    assert receipt.admission_permit is not None
    # Simulate missing receiving-owner evidence, not a caller-shaped receipt.
    # Both ordinary inspection and exact readback must fail closed.
    async with native_stores[0]._transaction(initialized.owner.application_scope, write=True) as tx:
        await tx.delete("events", (receipt.admission_permit.event.sequence,))
    found = await application.lookup_collaboration_admission(
        command, context=resolver.sender.context
    )
    assert isinstance(found, ExactUnavailable)


async def test_receiver_guard_failure_after_commit_keeps_exact_admission(
    native_stores, monkeypatch
):
    from cayu.collaboration.participants import CollaborationUnavailable

    application, resolver, command, _, _, _ = await prepared_scenario(native_stores)
    original = resolver.recipient.acquire

    @asynccontextmanager
    async def failing_exit(context):
        async with original(context) as authority:
            yield authority
        raise RuntimeError("guard exit acknowledgement failure")

    monkeypatch.setattr(resolver.recipient, "acquire", failing_exit)
    with pytest.raises(CollaborationUnavailable):
        await application.admit_collaboration_request(command, context=resolver.recipient.context)
    found = await application.lookup_collaboration_admission(
        command, context=resolver.sender.context
    )
    assert isinstance(found, ExactMatch)
    # Historical replay runs before reacquisition of the failed guard.
    assert (
        await application.admit_collaboration_request(command, context=resolver.recipient.context)
        == found.receipt
    )
