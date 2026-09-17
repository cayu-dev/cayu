"""PR1 value validation, not qualification of a durable export runtime."""

import inspect
from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from cayu.collaboration._contracts import (
    CollaborationConflict,
    CollaborationContractError,
    ExpectedOperation,
    InitiatorBinding,
    ObjectRef,
    OperationRef,
    OwnerRef,
)
from cayu.collaboration._preparation import prepare_contract, require_exact_contract
from cayu.collaboration.exports import (
    ExportLimits,
    SessionExportAcceptance,
    SessionExportAcceptanceReader,
    SessionExportAccessContext,
    SessionExportAction,
    SessionExportAuthorization,
    SessionExportCapacityExceeded,
    SessionExportConflict,
    SessionExportDenied,
    SessionExportIntent,
    SessionExportNamespace,
    SessionExportPolicy,
    SessionExportProjector,
    SessionExportReceipt,
    SessionExportRef,
    SessionExportRegistration,
    SessionExportRequest,
    SessionExportSettlementReceipt,
    SessionExportSettlementRequest,
    SessionExportUnavailable,
)
from cayu.vaults.redaction import SecretRedactor


def owner():
    return OwnerRef(application_scope="app", owner_id="owner", incarnation="inc")


def object_ref(kind):
    return ObjectRef(owner=owner(), kind=kind, object_id=kind, incarnation="inc", revision=1)


def request():
    return SessionExportRequest(
        ref=SessionExportRef(
            session_id="session",
            session_instance_id="instance",
            operation=OperationRef(
                application_scope="app", namespace_incarnation="ns", generation=1, caller_key="key"
            ),
        ),
        source_indices=(12, 2, 0),
        audience=owner(),
        projector=object_ref("projector"),
        policy=object_ref("policy"),
    )


def limits():
    return ExportLimits(max_exports=1024, max_retained_bytes=64 * 1024 * 1024, max_pending=64)


def receipt():
    req = request()
    return SessionExportReceipt(
        expected=ExpectedOperation[SessionExportIntent](
            operation=req.ref.operation,
            kind="session_export",
            schema_version=1,
            mode="source",
            source=owner(),
            destination=req.audience,
            initiator=InitiatorBinding(
                issuer=owner(),
                principal="principal",
                participant=None,
                mandate=None,
                invocation_id=None,
                interaction_id=None,
            ),
            receipt_stage="published",
            intent=SessionExportIntent(
                request=req,
                limits=limits(),
                source_commitment="a" * 64,
                output_commitment="b" * 64,
                authorization=SessionExportAuthorization(
                    issuer=owner(),
                    principal="principal",
                    policy=req.policy,
                    revision=1,
                    expires_at_ms=1,
                ),
            ),
        ),
        event_id="event",
    )


def prepare(schema, value):
    return prepare_contract(schema, value, redactor=SecretRedactor())


def test_receipt_round_trip_is_detached_and_payload_free():
    original = receipt()
    restored = prepare(SessionExportReceipt, original.model_dump(mode="json"))
    assert restored == original
    assert restored is not original
    assert restored.expected.intent is not original.expected.intent
    assert set(type(restored).model_fields) == {"expected", "event_id"}
    assert restored.expected.intent.request.source_indices == (0, 2, 12)
    assert SessionExportReceipt.model_validate_json(original.model_dump_json()) == original
    with pytest.raises(ValueError):
        restored.event_id = "changed"


@pytest.mark.parametrize("indices", [(), tuple(range(16)), (1000, 0, 20)])
def test_source_selection_boundaries_and_numeric_canonicalization(indices):
    raw = request().model_dump(mode="json")
    raw["source_indices"] = indices
    assert prepare(SessionExportRequest, raw).source_indices == tuple(sorted(indices))


@pytest.mark.parametrize(
    "indices", [(True,), (False,), (-1,), (1.0,), ("1",), (1, 1), tuple(range(17)), (2**53,)]
)
def test_source_selection_rejects_ambiguous_or_oversized_values(indices):
    raw = request().model_dump(mode="json")
    raw["source_indices"] = indices
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportRequest, raw)


@pytest.mark.parametrize(
    "field,maximum",
    [("max_exports", 1024), ("max_retained_bytes", 64 * 1024 * 1024), ("max_pending", 64)],
)
@pytest.mark.parametrize("invalid", [True, False, 0, -1, "1", 1.0, None, "overflow", "missing"])
def test_limits_are_required_strict_positive_and_bounded(field, maximum, invalid):
    raw = limits().model_dump()
    if invalid == "missing":
        del raw[field]
    else:
        raw[field] = maximum + 1 if invalid == "overflow" else invalid
    with pytest.raises(CollaborationContractError):
        prepare(ExportLimits, raw)


def test_limits_accept_minimum():
    assert prepare(ExportLimits, dict(max_exports=1, max_retained_bytes=1, max_pending=1))


@pytest.mark.parametrize("field", ["revision", "expires_at_ms"])
@pytest.mark.parametrize("value", [True, False, 0, -1, "1", 1.0, 2**53])
def test_authorization_requires_strict_positive_revision_and_expiry(field, value):
    raw = receipt().expected.intent.authorization.model_dump(mode="json")
    raw[field] = value
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportAuthorization, raw)


@pytest.mark.parametrize("field", ["source_commitment", "output_commitment"])
@pytest.mark.parametrize(
    "value", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64, True, "a" * 64 + "\n"]
)
def test_commitments_require_canonical_sha256(field, value):
    raw = receipt().expected.intent.model_dump(mode="json")
    raw[field] = value
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportIntent, raw)


@pytest.mark.parametrize("field", ["principal", "context", "authorization", "payload"])
def test_public_request_cannot_embed_trusted_context_or_output(field):
    raw = request().model_dump(mode="json")
    raw[field] = "principal"
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportRequest, raw)
    assert prepare(SessionExportAccessContext, {"principal": "principal"})


@pytest.mark.parametrize(
    "path",
    [
        ("request", "ref", "session_id"),
        ("request", "ref", "session_instance_id"),
        ("request", "ref", "operation", "caller_key"),
        ("request", "audience", "incarnation"),
        ("request", "projector", "incarnation"),
        ("authorization", "principal"),
        ("authorization", "issuer", "incarnation"),
        ("authorization", "revision"),
        ("authorization", "expires_at_ms"),
        ("limits", "max_pending"),
        ("limits", "max_exports"),
        ("limits", "max_retained_bytes"),
        ("source_commitment",),
        ("output_commitment",),
        ("request", "source_indices"),
    ],
)
def test_exact_intent_compares_decision_bearing_material(path):
    original = receipt().expected.intent
    raw = original.model_dump(mode="json")
    parent = raw
    for key in path[:-1]:
        parent = parent[key]
    value = parent[path[-1]]
    parent[path[-1]] = (
        (value - 1 if value > 1 else 2)
        if type(value) is int
        else (
            [1] if type(value) is list else "c" * 64 if path[-1].endswith("commitment") else "other"
        )
    )
    changed = prepare(SessionExportIntent, raw)
    with pytest.raises(CollaborationConflict):
        require_exact_contract(original, changed, redactor=SecretRedactor())


def test_cross_field_conflicts_fail_closed():
    raw = receipt().model_dump(mode="json")
    raw["expected"]["operation"]["caller_key"] = "other"
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportReceipt, raw)
    raw = receipt().expected.intent.model_dump(mode="json")
    raw["authorization"]["policy"]["incarnation"] = "other"
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportIntent, raw)


def test_unsafe_mutation_has_no_diagnostic_side_channels(capsys, caplog):
    class Canary:
        def __repr__(self):
            raise AssertionError("secret-canary")

        __str__ = __repr__

    import warnings

    forged = request().model_copy(update={"source_indices": (Canary(),)})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(CollaborationContractError) as error:
            prepare(SessionExportRequest, forged)
    assert "secret-canary" not in str(error.value)
    assert not caught
    assert not caplog.records
    assert capsys.readouterr() == ("", "")


def test_registered_extensions_are_abstract_and_guard_factory_is_synchronous():
    assert SessionExportPolicy.__abstractmethods__ == {"ref", "acquire"}
    assert SessionExportProjector.__abstractmethods__ == {"ref", "project", "validate"}
    assert not inspect.iscoroutinefunction(SessionExportPolicy.acquire)
    assert set(get_args(SessionExportAction)) == {
        "initialize",
        "release",
        "source",
        "export",
        "readback",
        "expose",
        "retire",
    }
    for extension in (SessionExportPolicy, SessionExportProjector):
        with pytest.raises(TypeError):
            extension()
    assert SessionExportAcceptanceReader.__abstractmethods__ == {"owner", "lookup"}
    assert inspect.iscoroutinefunction(SessionExportAcceptanceReader.lookup)
    parameters = inspect.signature(SessionExportPolicy.acquire).parameters
    assert "ref" not in parameters
    assert "action" not in parameters
    assert parameters["actions"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["actions"].default is inspect.Parameter.empty
    assert parameters["session_id"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["session_instance_id"].kind == inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    "actions",
    [
        ("readback", "source", "export"),
        ("readback", "release"),
        ("readback", "retire"),
        ("readback", "expose"),
        ("initialize",),
    ],
)
def test_policy_signature_accepts_combined_permissions_without_operation_ref(actions):
    signature = inspect.signature(SessionExportPolicy.acquire)
    bound = signature.bind(
        object(),
        SessionExportAccessContext(principal="principal"),
        session_id="session",
        session_instance_id="instance",
        actions=actions,
        audience=owner(),
    )
    assert bound.arguments["actions"] == actions


@pytest.mark.parametrize(
    "error",
    [
        SessionExportDenied,
        SessionExportUnavailable,
        SessionExportConflict,
        SessionExportCapacityExceeded,
    ],
)
def test_export_errors_accept_no_untrusted_message(error):
    assert str(error())
    with pytest.raises(TypeError):
        error("secret-canary")


def test_registration_is_frozen_and_not_a_serialized_authority():
    registration = SessionExportRegistration(
        owner=owner(), policy=None, projectors=(), limits=limits()
    )
    with pytest.raises(FrozenInstanceError):
        registration.owner = owner()
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportRequest, registration)
    assert registration.readers == ()


@pytest.mark.parametrize("generation", [True, False, 0, 2, 1.0, "1", None])
def test_namespace_generation_is_strictly_one(generation):
    raw = dict(
        owner=owner(),
        session_id="session",
        session_instance_id="instance",
        namespace_incarnation="ns",
        generation=generation,
        limits=limits(),
    )
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportNamespace, raw)
    raw["generation"] = 1
    namespace = prepare(SessionExportNamespace, raw)
    assert prepare(SessionExportNamespace, namespace.model_dump(mode="json")) == namespace


def settlement(mode):
    export = receipt()
    return SessionExportSettlementReceipt(
        request=SessionExportSettlementRequest(
            request=export.expected.intent.request,
            operation=export.expected.operation.model_copy(update={"caller_key": "settle"}),
            mode=mode,
        ),
        initiator=export.expected.initiator,
        acceptance=SessionExportAcceptance(
            export_receipt=export,
            receiving_owner=owner(),
            receipt_id="accepted",
        )
        if mode == "release"
        else None,
        event_id="settled",
    )


@pytest.mark.parametrize("mode", ["release", "retire"])
def test_settlement_round_trip_and_exact_principal(mode):
    original = settlement(mode)
    assert prepare(SessionExportSettlementReceipt, original.model_dump(mode="json")) == original
    changed = prepare(
        SessionExportSettlementReceipt,
        original.model_copy(
            update={"initiator": original.initiator.model_copy(update={"principal": "other"})}
        ),
    )
    with pytest.raises(CollaborationConflict):
        require_exact_contract(original, changed, redactor=SecretRedactor())


@pytest.mark.parametrize("change", ["missing", "audience", "request", "retire"])
def test_settlement_acceptance_must_match_release(change):
    raw = settlement("release").model_dump(mode="json")
    if change == "missing":
        raw["acceptance"] = None
    elif change == "audience":
        raw["acceptance"]["receiving_owner"]["incarnation"] = "other"
    elif change == "request":
        raw["request"]["request"]["ref"]["session_instance_id"] = "other"
    else:
        raw["request"]["mode"] = "retire"
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportSettlementReceipt, raw)


def test_public_settlement_request_rejects_receipts_and_unknown_modes():
    raw = settlement("retire").request.model_dump(mode="json")
    raw["acceptance"] = settlement("release").acceptance.model_dump(mode="json")
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportSettlementRequest, raw)
    del raw["acceptance"]
    raw["mode"] = "future"
    with pytest.raises(CollaborationContractError):
        prepare(SessionExportSettlementRequest, raw)
