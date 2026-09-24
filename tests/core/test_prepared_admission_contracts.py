"""Pure snapshot checks; public receiver qualification lives separately."""

import json

import pytest
from tests.core.test_budget_binding import _binding

from cayu._validation import canonical_durable_json_bytes
from cayu.collaboration._contracts import CollaborationContractError
from cayu.collaboration.prepared_admission import (
    MAX_PREPARED_BUDGET_BYTES,
    MAX_PREPARED_PROFILE_BYTES,
    _snapshot_document,
    prepared_budget,
    prepared_profile,
)


def test_budget_snapshot_retains_every_field_without_resolving_authority():
    original = _binding()
    encoded = canonical_durable_json_bytes(original.model_dump(mode="json"), "binding").decode()
    reconstructed = prepared_budget(encoded)
    assert reconstructed == original
    assert reconstructed is not original
    assert reconstructed.authority_digest == original.authority_digest
    for field, value in (
        ("sponsor", "another-sponsor"),
        ("receiver_generation", 2),
        ("allowance", 1),
        ("settlement_policy", "another-policy"),
    ):
        changed = original.model_dump(mode="json")
        changed[field] = value
        checked = prepared_budget(canonical_durable_json_bytes(changed, "binding").decode())
        assert checked.authority_digest != reconstructed.authority_digest


@pytest.mark.parametrize("snapshot", ["{", "null", "[]", "{}", '{"fingerprint":"private-canary"}'])
@pytest.mark.parametrize("reader", [prepared_budget, prepared_profile])
def test_bad_snapshot_errors_have_no_original_validation_context(snapshot, reader, caplog, capsys):
    with pytest.raises(CollaborationContractError) as caught:
        reader(snapshot)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None
    assert "private-canary" not in str(caught.value)
    assert "private-canary" not in caplog.text
    captured = capsys.readouterr()
    assert "private-canary" not in captured.out + captured.err


def test_snapshot_requires_canonical_complete_data():
    value = _binding().model_dump(mode="json")
    with pytest.raises(CollaborationContractError):
        prepared_budget(json.dumps(value, indent=2))
    del value["ancestor_budget_ids"]
    with pytest.raises(CollaborationContractError):
        prepared_budget(canonical_durable_json_bytes(value, "binding").decode())


@pytest.mark.parametrize("amount", [MAX_PREPARED_BUDGET_BYTES, MAX_PREPARED_BUDGET_BYTES + 1])
def test_oversized_budget_snapshot_is_rejected(amount):
    value = _binding().model_dump(mode="json")
    value["purpose"] = "x" * amount
    with pytest.raises(CollaborationContractError):
        prepared_budget(json.dumps(value))


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_canonical_budget_byte_ceiling_below_at_above(offset):
    value = _binding().model_dump(mode="json")
    value["purpose"] = "x"
    base = canonical_durable_json_bytes(value, "binding")
    value["purpose"] = "x" * (MAX_PREPARED_BUDGET_BYTES + offset - len(base) + 1)
    encoded = canonical_durable_json_bytes(value, "binding").decode()
    assert len(encoded.encode()) == MAX_PREPARED_BUDGET_BYTES + offset
    if offset > 0:
        with pytest.raises(CollaborationContractError):
            prepared_budget(encoded)
    else:
        assert prepared_budget(encoded).purpose == value["purpose"]


def test_prepared_receipts_have_a_breaking_writer_fence():
    from cayu.storage import migrations

    revision = migrations.revision(106)
    assert revision.compatible_from == 106
    with pytest.raises(migrations.SchemaTooNew):
        migrations.validate(
            migrations.SchemaState(revision=106, compatible_from=106),
            app_latest=105,
            app_min_supported=105,
        )
    with pytest.raises(migrations.SchemaTooOld):
        migrations.validate(
            migrations.SchemaState(revision=105, compatible_from=105),
            app_latest=106,
            app_min_supported=106,
        )


def test_old_request_wrapper_does_not_advertise_prepared_admission():
    from cayu.collaboration._contracts import OwnerRef
    from cayu.collaboration.base import REQUEST_FAMILY
    from cayu.collaboration.memory import InMemoryCollaborationStore

    class OldWrapper(InMemoryCollaborationStore):
        request_contract_version = 1

    owner = OwnerRef(application_scope="scope", owner_id="participants", incarnation="one")
    assert REQUEST_FAMILY.version == 2
    assert REQUEST_FAMILY in InMemoryCollaborationStore().capabilities(owner).mutations
    assert REQUEST_FAMILY not in OldWrapper().capabilities(owner).mutations


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_profile_json_preflight_byte_ceiling(offset):
    # This is the bounded decoder before native profile reconstruction, not
    # synthetic profile authority. Real profiles use their existing validator.
    document = {"payload": ""}
    overhead = len(canonical_durable_json_bytes(document, "snapshot"))
    document["payload"] = "x" * (MAX_PREPARED_PROFILE_BYTES + offset - overhead)
    encoded = canonical_durable_json_bytes(document, "snapshot").decode()
    if offset > 0:
        with pytest.raises(CollaborationContractError):
            _snapshot_document(encoded, max_bytes=MAX_PREPARED_PROFILE_BYTES)
    else:
        assert _snapshot_document(encoded, max_bytes=MAX_PREPARED_PROFILE_BYTES) == document
