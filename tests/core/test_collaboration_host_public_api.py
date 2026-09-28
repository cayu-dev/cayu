"""Public host configuration is the native configuration, not another authority layer."""

import asyncio
import dataclasses
import importlib
import warnings

import pytest

import cayu
import cayu.collaboration
from cayu import (
    CayuApp,
    CollaborationAccessContext,
    CollaborationHost,
    HostOwnershipLimits,
    HostProducerSource,
    HostRegistration,
    ParticipantRef,
)
from cayu.collaboration._contracts import OwnerRef
from cayu.vaults.redaction import SecretRedactor

PUBLIC_HOST_NAMES = (
    "CollaborationHost",
    "HostInspection",
    "HostRegistration",
    "HostOwnershipLimits",
    "HostProducerSource",
    "HostProducerMaintenanceRule",
    "HostProducerExecutionRule",
    "HostProducerDisclosure",
    "HostProducerOutputRule",
    "HostPlanningRule",
    "HostProducerRegistrationRule",
    "HostWaitRule",
    "HostContinuationRule",
    "HostClarificationRule",
    "HostClarificationMaintenanceSource",
    "HostRequestMaintenanceSource",
    "HostPlannedProducerRule",
    "HostPlannedProducer",
    "HostProducerExecution",
    "HostProducerMaintenance",
)


@pytest.mark.anyio
@pytest.mark.parametrize("registered", [False, True])
async def test_host_continuation_rejects_mutated_resume_without_diagnostic_disclosure(
    registered, capsys, caplog
):
    from datetime import UTC, datetime

    from tests.core.test_session_continuation import _ready_continuation

    from cayu import (
        ContinuationRecovery,
        ContinuationService,
        HostContinuationRule,
        Message,
        ParticipantSessionReference,
        ResumeRequest,
    )
    from cayu._exception_groups import iter_exception_tree
    from cayu.runtime._session_continuation import continuation_operation_key

    store, ticket, latch = await _ready_continuation()
    secret = "host-resume-private-canary"

    class PrivateValue:
        def __repr__(self):
            return secret

        def __str__(self):
            return secret

    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor(secret if registered else None),
    )
    expected = ContinuationRecovery(
        session=ParticipantSessionReference(
            participant=ParticipantRef(
                owner=ticket.owner, participant_id="participant", incarnation="one"
            ),
            creation_key="creation",
            session_id=ticket.session_id,
            session_instance_id=ticket.session_instance_id,
            receipt_commitment="receipt",
        ),
        registration_key=ticket.registration_key,
        ticket_key=continuation_operation_key(ticket),
        preparation_digest="0" * 64,
    )
    service = ContinuationService(
        ticket=ticket,
        latch=latch,
        continuation_id="next",
        mode="inline",
        accepted_at=datetime.now(UTC).isoformat(),
    )
    valid = ResumeRequest(session_id=ticket.session_id, messages=[Message.text("user", "Continue")])

    def configured(request):
        return HostRegistration(
            limits=HostOwnershipLimits(1, 1, 4, 262144),
            producer_sources=(),
            producer_rules=(),
            continuation_rules=(
                HostContinuationRule(
                    expected, request, service, CollaborationAccessContext(principal="operator")
                ),
            ),
        )

    # Inert registration validates shape, not caller-created execution authority.
    host = CollaborationHost(app, configured(valid))
    assert not host.inspect().pending
    await host.aclose()
    poisoned = valid.model_copy(update={"session_id": PrivateValue()})
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, TypeError)) as caught:
            CollaborationHost(app, configured(poisoned))
    captured = capsys.readouterr()
    diagnostics = [captured.out, captured.err, caplog.text]
    diagnostics.extend(str(item.message) for item in recorded)
    for error in iter_exception_tree(caught.value):
        diagnostics.extend((str(error), repr(error), *getattr(error, "__notes__", ())))
    assert all(secret not in value for value in diagnostics)


@pytest.mark.parametrize("name", PUBLIC_HOST_NAMES)
def test_public_host_exports_use_one_native_type(name):
    module = importlib.import_module("cayu.collaboration.host")
    assert name in vars(cayu)["__all__"]
    assert name in vars(cayu.collaboration)["__all__"]
    assert getattr(cayu, name) is getattr(cayu.collaboration, name) is getattr(module, name)


@pytest.mark.parametrize(
    "name",
    (
        "ContinuationConflict",
        "ContinuationUnavailable",
        "ContinuationDiscoveryPage",
        "ContinuationRecovery",
        "ContinuationRecord",
        "ContinuationService",
        "ParticipantSessionCursor",
        "ParticipantSessionReference",
    ),
)
def test_public_host_reconstruction_contracts_are_exported(name):
    assert name in vars(cayu)["__all__"]
    assert name in vars(cayu.collaboration)["__all__"]
    assert getattr(cayu, name) is getattr(cayu.collaboration, name)


@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("invalid", ["string", "object", "boolean"])
def test_public_host_rejects_mutated_authority_without_diagnostic_disclosure(
    registered, invalid, capsys, caplog
):
    secret = "host-authority-private-canary"

    class PrivateValue:
        def __repr__(self):
            return secret

        def __str__(self):
            return secret

    participant = ParticipantRef(
        owner=OwnerRef(application_scope="host", owner_id="owner", incarnation="one"),
        participant_id=secret,
        incarnation="one",
    )
    poisoned = {"string": secret, "object": PrivateValue(), "boolean": True}[invalid]
    participant = participant.model_copy(update={"owner": poisoned})
    app = CayuApp(
        enable_logging=False, secret_redactor=SecretRedactor(secret if registered else None)
    )
    registration = HostRegistration(
        limits=HostOwnershipLimits(1, 1, 2, 262144),
        producer_sources=(
            HostProducerSource(participant, CollaborationAccessContext(principal="operator")),
        ),
        producer_rules=(),
    )
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        with pytest.raises((TypeError, ValueError)) as caught:
            CollaborationHost(app, registration)
    diagnostics = [str(item.message) for item in recorded]
    pending: list[BaseException] = [caught.value]
    seen = set()
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        diagnostics.extend((str(error), repr(error)))
        diagnostics.extend(getattr(error, "__notes__", ()))
        pending.extend(item for item in (error.__cause__, error.__context__) if item is not None)
        if isinstance(error, BaseExceptionGroup):
            pending.extend(error.exceptions)
    captured = capsys.readouterr()
    diagnostics.extend((captured.out, captured.err, caplog.text))
    assert all(secret not in diagnostic for diagnostic in diagnostics)


def test_public_host_is_inert_and_does_not_close_application(monkeypatch):
    app = CayuApp(enable_logging=False)
    selected = HostRegistration(
        limits=HostOwnershipLimits(1, 1, 2, 262144),
        producer_sources=(
            HostProducerSource(
                ParticipantRef(
                    owner=OwnerRef(
                        application_scope="public-host", owner_id="owner", incarnation="one"
                    ),
                    participant_id="participant",
                    incarnation="one",
                ),
                CollaborationAccessContext(principal="operator"),
            ),
        ),
        producer_rules=(),
    )

    async def forbidden_close(*args, **kwargs):
        pytest.fail("The host does not own application lifetime")

    monkeypatch.setattr(app, "aclose", forbidden_close, raising=False)
    # There is intentionally no running loop during registration/construction.
    host = CollaborationHost(app, selected)
    assert not host.inspect().pending
    assert host.inspect().serviced == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        selected.batch_size = 100  # ty: ignore[invalid-assignment] -- deliberate frozen-value mutation

    async def scenario():
        async with host:
            observed = await host.service_once()
            assert not observed.coverage_complete
            assert observed.serviced == 0
            # A configuration value is not current receiving authority.
            assert observed.source_failures == 1
        assert host.inspect().closing
        assert not host.inspect().pending

    asyncio.run(scenario())
