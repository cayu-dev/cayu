"""Recovery reconstructs exact source-owned registration without native execution."""

import warnings
from copy import deepcopy
from hashlib import sha256

import pytest
from tests.core.test_collaboration_request_foundation import RequestResolver
from tests.core.test_participant_identity import CONTEXT, Policy, app, registration
from tests.core.test_prepared_admission_public import native_stores as native_stores
from tests.core.test_producer_output_contracts import output_scenario

from cayu.collaboration import ProducerOutputRecovery
from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExactConflict,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
)
from cayu.collaboration._preparation import contract_bytes
from cayu.collaboration._request_store import operation_key
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.request_access import PreparedAdmissionRegistration, RequestRegistration


def test_producer_contracts_have_matching_public_exports():
    import cayu
    import cayu.collaboration as collaboration

    for name in (
        "ProducerOutputRegistration",
        "ProducerOutputProposal",
        "ProducerProgressOccurrence",
        "ProducerProgressCommand",
        "ProducerProgressReference",
        "ProducerCleanupReclamation",
        "ProducerCleanupRetirement",
        "ProducerOutputRecord",
        "ProducerOutputLimits",
        "ProducerDeliveryDestination",
        "ProducerOutputRecovery",
        "ProducerPendingOutput",
        "ProducerPendingPage",
        "ProducerExportRecord",
        "ProducerDeliveryRecord",
        "ProducerCompletionRecord",
        "ProducerCleanupFinalized",
        "ProducerDeliveryRecovery",
        "ProducerDeliveryStatus",
        "ProducerDispositionStatus",
        "ProducerExportCleanupStatus",
        "ProducerOutputAcceptanceReader",
    ):
        assert name in cayu.__all__ and name in collaboration.__all__
        assert getattr(cayu, name) is getattr(collaboration, name)


@pytest.mark.anyio
async def test_exact_producer_readback_recovers_with_independent_owner_and_current_access(
    native_stores, monkeypatch, caplog, capsys
):
    values = await output_scenario(native_stores)
    original, resolver, _, provider, session, initialized, command, execution = values
    monkeypatch.setattr(original._request_coordinator._owners, "observation_timeout", 60)
    registered = await original.register_producer_output(
        command, execution, context=resolver.recipient.context
    )
    page = await original.pending_producer_outputs(
        command.admission.prepared.recipient, context=CONTEXT
    )
    token = next(
        item.recovery for item in page.items if item.recovery.registration == command.operation
    )
    assert token == ProducerOutputRecovery(
        registration=command.operation,
        registration_commitment="sha256:"
        + sha256(contract_bytes(command, redactor=original._secret_redactor)).hexdigest(),
    )
    policy = Policy()
    other = app(
        native_stores[2](),
        registration(
            scope=command.operation.application_scope,
            limits=command.admission.expected.intent.limits,
            policy=policy,
        ),
        collaboration_requests=RequestRegistration(
            mandates=RequestResolver(command.admission.expected.intent.request),
            prepared_admission=PreparedAdmissionRegistration(receiver=command.receiver),
            max_ttl_ms=300_000,
        ),
    )
    try:
        await other.initialize_collaboration()
        monkeypatch.setattr(other._request_coordinator._owners, "observation_timeout", 60)
        # The new application has no providers or native source session. Recovery
        # uses the source owner's immutable registration, not a process-local handle.
        assert not other._providers
        assert await other.session_store.load(session.id) is None
        for expected in (command, token):
            found = await other.lookup_producer_registration(expected, context=CONTEXT)
            assert isinstance(found, ExactMatch) and found.receipt == command
        assert isinstance(
            await other.lookup_producer_registration(
                token.model_copy(update={"registration_commitment": "sha256:" + "f" * 64}),
                context=CONTEXT,
            ),
            ExactConflict,
        )
        assert isinstance(
            await other.lookup_producer_registration(
                command.model_copy(update={"binding_incarnation": "different-binding"}),
                context=CONTEXT,
            ),
            ExactConflict,
        )
        # Keep the durable operation identity fixed while changing every other
        # scalar leaf. Exercise the PUBLIC snapshot/validation/access/readback
        # pipeline, not just a helper's digest comparison.
        document = command.model_dump(mode="json")

        def leaves(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, (*path, key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    yield from leaves(item, (*path, index))
            else:
                yield path, value

        checked = []
        for path, value in leaves(document):
            if path[0] == "operation":
                continue
            changed = deepcopy(document)
            parent = changed
            for part in path[:-1]:
                parent = parent[part]
            parent[path[-1]] = (
                not value
                if type(value) is bool
                else value + 1
                if type(value) is int
                else "changed"
                if value is None
                else value + "-changed"
            )
            candidate = type(command).model_construct(**changed)
            try:
                result = await other.lookup_producer_registration(candidate, context=CONTEXT)
            except (CollaborationContractError, CollaborationConflict, CollaborationAccessDenied):
                pass
            else:
                assert isinstance(result, ExactConflict), path
            checked.append(path)
        assert len(checked) > 100
        assert (
            await other.lookup_producer_registration(command, context=CONTEXT)
        ).receipt == command
        missing = token.model_copy(
            update={"registration": initialized.operation("not-registered-producer")}
        )
        assert isinstance(
            await other.lookup_producer_registration(missing, context=CONTEXT), ExactNotFound
        )
        with pytest.raises(CollaborationAccessDenied):
            await other.lookup_producer_registration(
                token, context=CollaborationAccessContext(principal="not-the-operator")
            )
        policy.allowed = (command.admission.prepared.recipient,)
        with pytest.raises(CollaborationAccessDenied):
            await other.lookup_producer_registration(token, context=CONTEXT)
        policy.allowed = None
        policy.denied.add("request_readback")
        with pytest.raises(CollaborationAccessDenied):
            await other.lookup_producer_registration(token, context=CONTEXT)
        policy.denied.clear()
        # An index and matching command are insufficient when a required durable
        # receipt is missing. Restoring exact evidence restores historical readback.
        key = operation_key(registered.event.operation)
        async with native_stores[0]._transaction(
            initialized.owner.application_scope, write=True
        ) as tx:
            await tx.delete("operations", key)
        try:
            assert isinstance(
                await other.lookup_producer_registration(token, context=CONTEXT), ExactUnavailable
            )
        finally:
            async with native_stores[0]._transaction(
                initialized.owner.application_scope, write=True
            ) as tx:
                await tx.put("operations", key, registered.event, insert=True)
        found = await other.lookup_producer_registration(token, context=CONTEXT)
        assert isinstance(found, ExactMatch) and found.receipt == command
        secret = "rejected-producer-readback-private-canary"

        class Hostile:
            def __str__(self):
                return secret

            def __repr__(self):
                return secret

        rejected = (
            command.model_copy(update={"binding_incarnation": Hostile()}),
            token.model_copy(update={"registration_commitment": Hostile()}),
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            for value in rejected:
                with pytest.raises(CollaborationContractError) as caught:
                    await other.lookup_producer_registration(value, context=CONTEXT)
                errors = [caught.value]
                seen = set()
                while errors:
                    error = errors.pop()
                    if id(error) in seen:
                        continue
                    seen.add(id(error))
                    assert secret not in str(error) + repr(error)
                    errors.extend(
                        item for item in (error.__cause__, error.__context__) if item is not None
                    )
                    if isinstance(error, BaseExceptionGroup):
                        errors.extend(error.exceptions)
        output = capsys.readouterr()
        assert secret not in caplog.text + output.out + output.err + repr(captured)
        assert isinstance(
            await other.lookup_producer_registration(token, context=CONTEXT), ExactMatch
        )
        assert not provider.requests
        assert (await original.session_store.load(session.id)).status == "pending"
    finally:
        await other.drain_collaboration_requests()
