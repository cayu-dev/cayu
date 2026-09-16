from __future__ import annotations

import json
import logging
import subprocess
import sys
import warnings
from typing import Annotated, Literal

import pytest
from pydantic import StrictBool, StrictStr, ValidationError, field_validator
from tests.core._collaboration_fixture import ProbeCommand, ProbeIntent, ProbeReceipt, command, slot

from cayu._exception_groups import iter_exception_tree
from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.collaboration._capabilities import (
    CapabilityDescriptor,
    CollaborationCapabilityUnavailable,
    FamilyVersion,
    require_capability,
)
from cayu.collaboration._contracts import (
    MAX_ENVELOPE_BYTES,
    CollaborationConflict,
    CollaborationContractError,
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    Generation,
    HandoffIntent,
    OperationRef,
    TransportClaimRef,
)
from cayu.collaboration._preparation import contract_bytes, prepare_contract, require_exact_contract
from cayu.vaults.redaction import SecretRedactor


def test_reference_roundtrip_and_detachment() -> None:
    original = command()
    copied = prepare_contract(ProbeCommand, original, redactor=SecretRedactor())
    assert copied == original and copied is not original
    assert copied.intent is not original.intent
    assert copied == prepare_contract(
        ProbeCommand, json.loads(original.model_dump_json()), redactor=SecretRedactor()
    )
    with pytest.raises(ValidationError):
        copied.intent.target = "changed"
    object.__setattr__(original.intent, "target", "changed")
    assert copied.intent.target == "B"


@pytest.mark.parametrize("value", [True, False, 1.0, "1", 0, -1, MAX_PORTABLE_JSON_INTEGER + 1])
def test_strict_generation_before_canonicalization(value) -> None:
    raw = command().operation.model_dump()
    raw["generation"] = value
    with pytest.raises(CollaborationContractError):
        prepare_contract(OperationRef, raw, redactor=SecretRedactor())


@pytest.mark.parametrize("field", ["application_scope", "namespace_incarnation", "caller_key"])
@pytest.mark.parametrize("value", ["", " padded ", "\x00", "\ud800", "😀" * 129, 123])
def test_invalid_identity(field, value) -> None:
    raw = command().operation.model_dump()
    raw[field] = value
    with pytest.raises(CollaborationContractError):
        prepare_contract(OperationRef, raw, redactor=SecretRedactor())


@pytest.mark.parametrize("length", [511, 512, 513])
def test_identity_byte_boundary(length: int) -> None:
    raw = command().operation.model_dump()
    raw["caller_key"] = "x" * length
    if length <= 512:
        assert (
            prepare_contract(OperationRef, raw, redactor=SecretRedactor()).caller_key
            == raw["caller_key"]
        )
    else:
        with pytest.raises(CollaborationContractError):
            prepare_contract(OperationRef, raw, redactor=SecretRedactor())


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("kind",), "another"),
        (("schema_version",), 2),
        (("mode",), "another"),
        (("receipt_stage",), "other"),
        (("source", "owner_id"), "other"),
        (("source", "incarnation"), "two"),
        (("destination", "application_scope"), "other"),
        (("destination", "owner_id"), "other"),
        (("destination", "incarnation"), "two"),
        (("initiator", "principal"), "other"),
        (("initiator", "issuer", "owner_id"), "other"),
        (("initiator", "issuer", "application_scope"), "other"),
        (("initiator", "issuer", "incarnation"), "two"),
        (("initiator", "invocation_id"), "invocation"),
        (("initiator", "interaction_id"), "interaction"),
        (("intent", "target"), "C"),
        (("intent", "policy"), "policy-2"),
        (("intent", "enabled"), False),
        (("intent", "threshold"), 2),
        (("intent", "selected"), "selection"),
        (("intent", "references"), ["resource"]),
        (("intent", "ordered"), ["first"]),
    ],
)
def test_fixed_key_compares_complete_material(path, replacement) -> None:
    original = command()
    raw = original.model_dump()
    child = raw
    for field in path[:-1]:
        child = child[field]
    child[path[-1]] = replacement
    changed = prepare_contract(ProbeCommand, raw, redactor=SecretRedactor())
    assert changed.operation == original.operation
    with pytest.raises(CollaborationConflict):
        require_exact_contract(original, changed, redactor=SecretRedactor())


@pytest.mark.parametrize("binding", ["participant", "mandate"])
@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("owner", "application_scope"), "other"),
        (("owner", "owner_id"), "other"),
        (("owner", "incarnation"), "two"),
        (("kind",), "other"),
        (("object_id",), "other"),
        (("incarnation",), "two"),
        (("revision",), 2),
        (("revision",), None),
    ],
)
def test_fixed_key_compares_complete_object_bindings(binding, path, replacement) -> None:
    raw = command().model_dump()
    raw["initiator"][binding] = {
        "owner": raw["source"],
        "kind": binding,
        "object_id": "object",
        "incarnation": "one",
        "revision": 1,
    }
    original = prepare_contract(ProbeCommand, raw, redactor=SecretRedactor())
    changed = original.model_dump()
    field = changed["initiator"][binding]
    for part in path[:-1]:
        field = field[part]
    field[path[-1]] = replacement
    reconstructed = prepare_contract(ProbeCommand, changed, redactor=SecretRedactor())
    assert reconstructed.operation == original.operation
    with pytest.raises(CollaborationConflict):
        require_exact_contract(original, reconstructed, redactor=SecretRedactor())


def test_sets_order_absence_and_transport_claims() -> None:
    first = ProbeIntent(target="B", references=("z", "a"), ordered=("z", "a"))
    second = ProbeIntent(target="B", references=("a", "z"), ordered=("z", "a"))
    require_exact_contract(first, second, redactor=SecretRedactor())
    with pytest.raises(CollaborationConflict):
        require_exact_contract(
            first, second.model_copy(update={"ordered": ("a", "z")}), redactor=SecretRedactor()
        )
    with pytest.raises(ValidationError):
        ProbeIntent(target="B", references=("a", "a"))
    claim = TransportClaimRef(
        owner=command().source, worker_id="worker", claim_id="claim", generation=1
    )
    assert claim.model_copy(update={"generation": 2}) != claim
    assert "claim" not in command().model_dump()
    raw = command().initiator.model_dump()
    del raw["mandate"]
    with pytest.raises(CollaborationContractError):
        prepare_contract(type(command().initiator), raw, redactor=SecretRedactor())


def test_handoff_scope_and_source() -> None:
    child = command()
    assert HandoffIntent[ProbeIntent](slot=slot(child), child=child).child == child
    with pytest.raises(ValidationError):
        HandoffIntent[ProbeIntent](slot=slot(command(scope="other")), child=child)


def test_lookup_variants_and_receipt_detachment() -> None:
    receipt = ProbeReceipt(expected=command(), stage="applied", selection="original")
    matched = ExactMatch[ProbeReceipt](receipt=receipt)
    object.__setattr__(receipt, "selection", "changed")
    assert matched.receipt.selection == "original"
    for variant in (ExactNotFound, ExactConflict, ExactUnavailable):
        with pytest.raises(ValidationError):
            variant(receipt=receipt)
    with pytest.raises(ValidationError):
        ExactMatch[ProbeReceipt]()
    assert (
        ExactMatch[ProbeReceipt](
            receipt=ProbeReceipt(
                expected=command(),
                stage="excluded",
                selection="none",
            )
        ).receipt.stage
        == "excluded"
    )


class Payload(ContractValue):
    text: StrictStr
    entries: tuple[StrictStr, ...] = ()


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_aggregate_envelope_boundary(delta: int) -> None:
    empty = len(json.dumps({"text": "", "entries": []}, separators=(",", ":")))
    raw = {"text": "x" * (MAX_ENVELOPE_BYTES - empty + delta), "entries": []}
    if delta <= 0:
        value = prepare_contract(Payload, raw, redactor=SecretRedactor())
        assert len(contract_bytes(value, redactor=SecretRedactor())) == MAX_ENVELOPE_BYTES + delta
    else:
        with pytest.raises(CollaborationContractError):
            prepare_contract(Payload, raw, redactor=SecretRedactor())


def test_split_boundary_and_container_limits() -> None:
    # Each member fits; their aggregate does not.
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            Payload, {"text": "x" * 34000, "entries": ["y" * 34000]}, redactor=SecretRedactor()
        )
    with pytest.raises(CollaborationContractError):
        prepare_contract(Payload, {"text": "x", "entries": ["a"] * 65}, redactor=SecretRedactor())
    nested = {}
    for _ in range(66):
        nested = {"child": nested}
    with pytest.raises(CollaborationContractError):
        prepare_contract(Payload, nested, redactor=SecretRedactor())
    cycle = []
    cycle.append(cycle)
    with pytest.raises(CollaborationContractError):
        prepare_contract(Payload, {"text": "x", "entries": cycle}, redactor=SecretRedactor())


@pytest.mark.parametrize("value", [float("inf"), float("nan"), object()])
def test_unsupported_values(value) -> None:
    with pytest.raises(CollaborationContractError):
        prepare_contract(Payload, {"text": value}, redactor=SecretRedactor())


def test_dynamic_mutable_fields_are_not_frozen_authority() -> None:
    class BadSchema(ContractValue):
        metadata: dict[str, str]

    with pytest.raises(ValidationError):
        BadSchema(metadata={"key": "value"})


def test_secret_diagnostics_and_mutated_models(caplog, capsys) -> None:
    canary = "fixture-private-canary-92817"

    class Hostile:
        def __str__(self):
            raise AssertionError(canary)

        def __repr__(self):
            raise AssertionError(canary)

    mutated = command()
    object.__setattr__(mutated.intent, "target", Hostile())
    raw = command().model_dump()
    raw[canary] = canary
    inputs = [mutated, raw, command().model_copy(update={"mode": canary})]
    redactor = SecretRedactor([canary])
    with warnings.catch_warnings(record=True) as recorded, caplog.at_level(logging.DEBUG):
        for value in inputs:
            with pytest.raises(CollaborationContractError) as caught:
                prepare_contract(ProbeCommand, value, redactor=redactor)
            assert list(iter_exception_tree(caught.value)) == [caught.value]
            assert canary not in str(caught.value) + repr(caught.value)
    assert not recorded
    captured = capsys.readouterr()
    assert canary not in caplog.text + captured.out + captured.err


def test_schema_controls_not_arbitrary_values() -> None:
    assert (
        prepare_contract(ExactNotFound, {}, redactor=SecretRedactor(["not_found", "status"]))
        == ExactNotFound()
    )
    with pytest.raises(CollaborationContractError):
        prepare_contract(Payload, {"text": "status"}, redactor=SecretRedactor(["status"]))


class ComposedControls(ContractValue):
    states: tuple[Literal["pending", "settled"], ...]
    status: Literal["pending"] | None
    fixed: tuple[Literal["pending"], StrictStr]
    nested: tuple[tuple[Annotated[Literal["pending"], "control"], ...], ...]
    nullable: tuple[Literal["pending"], ...] | None
    text: StrictStr = "clean"
    ambiguous: Literal["pending"] | StrictStr = "clean"


def composed_controls_input():
    return {
        "states": ["pending", "settled"],
        "status": "pending",
        "fixed": ["pending", "clean"],
        "nested": [["pending"]],
        "nullable": ["pending"],
    }


@pytest.mark.parametrize("absent", [False, True])
def test_composed_controls_roundtrip_under_secret_collision(absent: bool) -> None:
    raw = composed_controls_input()
    if absent:
        raw.update(status=None, nullable=None)
    redactor = SecretRedactor(["pending", "settled"])
    original = prepare_contract(ComposedControls, raw, redactor=redactor)
    for source in (original, json.loads(original.model_dump_json())):
        reconstructed = prepare_contract(ComposedControls, source, redactor=redactor)
        require_exact_contract(original, reconstructed, redactor=redactor)


@pytest.mark.parametrize("field", ["text", "ambiguous", "fixed"])
def test_composed_controls_do_not_exempt_untrusted_text(field: str) -> None:
    raw = composed_controls_input()
    raw[field] = ["pending", "pending"] if field == "fixed" else "pending"
    with pytest.raises(CollaborationContractError) as caught:
        prepare_contract(ComposedControls, raw, redactor=SecretRedactor(["pending"]))
    assert caught.value.__context__ is None


@pytest.mark.parametrize("field", ["states", "status", "fixed", "nested", "nullable"])
@pytest.mark.parametrize("invalid", ["unknown", True])
def test_composed_controls_reject_invalid_values(field: str, invalid) -> None:
    raw = composed_controls_input()
    raw[field] = (
        invalid
        if field == "status"
        else [invalid, "clean"]
        if field == "fixed"
        else [[invalid]]
        if field == "nested"
        else [invalid]
    )
    with pytest.raises(CollaborationContractError):
        prepare_contract(ComposedControls, raw, redactor=SecretRedactor(["pending"]))


@pytest.mark.parametrize("field", ["states", "status", "nullable"])
def test_composed_controls_check_validator_output(field: str) -> None:
    class ReplacingControls(ComposedControls):
        @field_validator("states", "status", "nullable")
        @classmethod
        def replace(cls, value, info):
            if info.field_name == field:
                return "unknown" if field == "status" else ("unknown",)
            return value

    with pytest.raises(CollaborationContractError) as caught:
        prepare_contract(
            ReplacingControls, composed_controls_input(), redactor=SecretRedactor(["pending"])
        )
    assert caught.value.__context__ is None


class LookupEnvelope(ContractValue):
    result: ExactLookup[ProbeReceipt]


@pytest.mark.parametrize("status", ["match", "not_found", "conflict", "unavailable"])
def test_nested_lookup_controls_survive_secret_collisions(status: str) -> None:
    raw = {"result": {"status": status}}
    if status == "match":
        raw["result"]["receipt"] = ProbeReceipt(
            expected=command(), stage="applied", selection="selected"
        ).model_dump()
    redactor = SecretRedactor([status, "status"])
    value = prepare_contract(LookupEnvelope, raw, redactor=redactor)
    assert value.result.status == status
    if status == "match":
        assert type(value.result.receipt) is ProbeReceipt
    restored = prepare_contract(
        LookupEnvelope, json.loads(value.model_dump_json()), redactor=redactor
    )
    require_exact_contract(value, restored, redactor=redactor)


@pytest.mark.parametrize("selection", [True, 42, {"unexpected": "value"}])
def test_lookup_alias_retains_concrete_receipt_validation(selection) -> None:
    receipt = ProbeReceipt(expected=command(), stage="applied", selection="selected").model_dump()
    receipt["selection"] = selection
    with pytest.raises(CollaborationContractError):
        prepare_contract(
            LookupEnvelope,
            {"result": {"status": "match", "receipt": receipt}},
            redactor=SecretRedactor(),
        )


@pytest.mark.parametrize("canary", ["not_found", "conflict", "unavailable"])
def test_nested_lookup_does_not_exempt_receipt_data(canary: str) -> None:
    raw = {
        "result": {
            "status": "match",
            "receipt": ProbeReceipt(
                expected=command(), stage="applied", selection=canary
            ).model_dump(),
        }
    }
    with pytest.raises(CollaborationContractError) as caught:
        prepare_contract(LookupEnvelope, raw, redactor=SecretRedactor([canary]))
    assert caught.value.__context__ is None
    assert canary not in str(caught.value)


def test_ambiguous_union_cannot_borrow_another_branch_control_exemption() -> None:
    calls = []

    class Fixed(ContractValue):
        status: Literal["not_found"]
        count: Generation

    class Text(ContractValue):
        status: StrictStr
        count: StrictStr

        @field_validator("status")
        @classmethod
        def erase(cls, value):
            calls.append(value)
            return "clean"

    class Envelope(ContractValue):
        result: Fixed | Text

    with pytest.raises(CollaborationContractError) as caught:
        prepare_contract(
            Envelope,
            {"result": {"status": "not_found", "count": "text"}},
            redactor=SecretRedactor(["not_found"]),
        )
    assert not calls
    assert caught.value.__context__ is None


@pytest.mark.parametrize("replacement", ["literal-canary-829182", "unknown", True])
def test_after_validator_cannot_invalidate_literal_control(replacement, caplog, capsys) -> None:
    class Control(ContractValue):
        status: Literal["applied"]

        @field_validator("status")
        @classmethod
        def replace(cls, value):
            return replacement

    canary = "literal-canary-829182"
    with (
        warnings.catch_warnings(record=True) as recorded,
        caplog.at_level(logging.DEBUG),
        pytest.raises(CollaborationContractError) as caught,
    ):
        prepare_contract(Control, {"status": "applied"}, redactor=SecretRedactor([canary]))
    assert list(iter_exception_tree(caught.value)) == [caught.value]
    assert not recorded
    captured = capsys.readouterr()
    assert (
        canary
        not in str(caught.value) + repr(caught.value) + caplog.text + captured.out + captured.err
    )


def test_secret_checks_before_and_after_owner_normalization() -> None:
    canary = "normalization-canary-829182"

    class ErasingSchema(ContractValue):
        value: StrictStr

        @field_validator("value")
        @classmethod
        def erase(cls, value):
            return "clean"

    class IntroducingSchema(ContractValue):
        value: StrictStr

        @field_validator("value")
        @classmethod
        def introduce(cls, value):
            return canary

    for schema, value in ((ErasingSchema, canary), (IntroducingSchema, "clean")):
        with pytest.raises(CollaborationContractError) as caught:
            prepare_contract(schema, {"value": value}, redactor=SecretRedactor([canary]))
        assert canary not in str(caught.value)
        assert caught.value.__context__ is None


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_node_limit_independent_of_byte_limit(delta: int) -> None:
    class Matrix(ContractValue):
        entries: tuple[tuple[tuple[StrictBool, ...], ...], ...]

    # 3 root/key/array + 2 middle arrays + 128 inner arrays + 8059 scalars.
    rows = [[True] * 64 for _ in range(61)] + [[True] * (59 + delta), [], []]
    raw = {"entries": [[[True] * 64 for _ in range(64)], rows]}
    assert len(json.dumps(raw).encode()) < MAX_ENVELOPE_BYTES
    if delta <= 0:
        prepare_contract(Matrix, raw, redactor=SecretRedactor())
    else:
        with pytest.raises(CollaborationContractError):
            prepare_contract(Matrix, raw, redactor=SecretRedactor())


def test_capability_versions_owners_wrappers_and_readback() -> None:
    owner = command().destination
    family = FamilyVersion(family="fixture.apply", version=1)
    descriptor = CapabilityDescriptor(owner=owner, mutations=(family,), readbacks=(family,))
    args = dict(
        expected_owner=owner, required=family, supported=(family,), redactor=SecretRedactor()
    )
    require_capability(descriptor, access="mutation", **args)
    read_only = CapabilityDescriptor(owner=owner, mutations=(), readbacks=(family,))
    require_capability(read_only, access="readback", **args)
    with pytest.raises(CollaborationCapabilityUnavailable):
        require_capability(read_only, access="mutation", **args)
    with pytest.raises(ValidationError):
        CapabilityDescriptor(owner=owner, mutations=(family,), readbacks=())
    for version in (True, "1", 0, -1):
        with pytest.raises(ValidationError):
            FamilyVersion(family="fixture.apply", version=version)
    for replacement in (
        dict(supported=()),
        dict(required=FamilyVersion(family="unknown", version=1)),
        dict(required=FamilyVersion(family=family.family, version=2)),
        dict(expected_owner=owner.model_copy(update={"incarnation": "other"})),
    ):
        with pytest.raises(CollaborationCapabilityUnavailable):
            require_capability(descriptor, access="mutation", **(args | replacement))


def test_supported_capability_controls_do_not_exempt_untrusted_owner_material() -> None:
    owner = command().destination
    family = FamilyVersion(family="fixture.apply", version=1)
    descriptor = CapabilityDescriptor(owner=owner, mutations=(family,), readbacks=(family,))
    args = dict(
        expected_owner=owner,
        required=family,
        supported=(family,),
        redactor=SecretRedactor("fixture.apply"),
        access="mutation",
    )
    require_capability(descriptor, **args)
    with pytest.raises(CollaborationContractError):
        require_capability(
            descriptor.model_copy(
                update={"owner": owner.model_copy(update={"owner_id": "fixture.apply"})}
            ),
            **args,
        )
    unknown = FamilyVersion(family="fixture.apply.extra", version=1)
    with pytest.raises(CollaborationContractError):
        require_capability(
            CapabilityDescriptor(owner=owner, mutations=(family,), readbacks=(family, unknown)),
            **args,
        )


def test_import_is_inert() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, threading; import cayu.collaboration; "
            "assert len(threading.enumerate()) == 1; "
            "assert 'cayu.applications' not in sys.modules; "
            "assert 'cayu.storage.postgres' not in sys.modules; "
            "assert not hasattr(cayu.collaboration, 'request_agent')",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
