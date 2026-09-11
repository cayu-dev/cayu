"""Generated application reservation contract; no database/authentication claim."""

import importlib
import warnings
from itertools import combinations
from uuid import NAMESPACE_URL, uuid5

import pytest

from cayu.cli.project import project_context
from tests.qualification.repository_maintenance_application import maintenance_project_files


@pytest.fixture
def identity(tmp_path):
    for name, content in maintenance_project_files().items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    with project_context(tmp_path):
        yield importlib.import_module("domain.maintenance_identity")


def intent(module, **changes):
    return module.MaintenanceRunIntent(
        **{
            "tenant": "tenant-a",
            "subject": "user-a",
            "idempotency_key": "request-1",
            "request_json": '{"case":"closed-integer-range-v1","configuration":"pinned"}',
            **changes,
        }
    )


def test_canonical_request_and_portable_identity(identity):
    original = intent(identity)
    reordered = intent(
        identity,
        request_json=' { "configuration": "pinned", "case": "closed-integer-range-v1" } ',
    )
    assert original == reordered and original.fingerprint == reordered.fingerprint
    allocated = identity.allocate_identity(original)
    assert allocated.intent == original and allocated.intent is not original
    assert (
        identity.MaintenanceRunIdentity.model_validate_json(allocated.model_dump_json())
        == allocated
    )
    assert identity.allocate_identity(original).public_id != allocated.public_id
    with pytest.raises(ValueError):
        allocated.task_id = allocated.public_id
    with pytest.raises(ValueError):
        identity.MaintenanceRunIdentity.model_validate(
            {**allocated.model_dump(), "task_id": allocated.public_id}
        )


def test_phase_ids_are_distinct_and_portable(identity):
    allocated = identity.allocate_identity(intent(identity))
    copied = identity.copy_identity(allocated)
    assert copied == allocated and copied is not allocated
    assert copied.intent is not allocated.intent
    assert {phase.value for phase in identity.MaintenanceTaskPhase} == {
        "coding",
        "git_preparation",
        "git_delivery",
        "github_delivery",
    }
    selected = [identity.task_id_for(copied, phase) for phase in identity.MaintenanceTaskPhase]
    assert selected == [
        copied.task_id,
        copied.git_preparation_task_id,
        copied.git_delivery_task_id,
        copied.github_delivery_task_id,
    ]
    fields = [name for name in type(allocated).model_fields if name.endswith("_id")]
    assert len(fields) == 8
    for first, second in combinations(fields, 2):
        with pytest.raises(ValueError):
            identity.MaintenanceRunIdentity.model_validate(
                {**allocated.model_dump(), second: getattr(allocated, first)}
            )
    for phase in ("coding", "unknown", True, None):
        with pytest.raises(ValueError):
            identity.task_id_for(allocated, phase)


def test_evaluator_owned_uuid5_root_survives_reconstruction(identity):
    root = str(uuid5(NAMESPACE_URL, "qualification-only-workflow-root"))
    allocated = identity.allocate_identity(intent(identity), workflow_session_id=root)
    assert allocated.workflow_session_id == root
    assert identity.copy_identity(allocated) == allocated
    assert (
        identity.MaintenanceRunIdentity.model_validate_json(allocated.model_dump_json())
        == allocated
    )


def test_coding_deadline_is_finite_portable_and_not_refreshed(identity):
    allocated = identity.allocate_identity(intent(identity))
    remaining = allocated.coding_deadline().remaining_seconds()
    assert remaining is not None and 0 < remaining <= 180
    elapsed = identity.allocate_identity(intent(identity), coding_expires_at="2000-01-01T00:00:00Z")
    assert elapsed.coding_expires_at == "2000-01-01T00:00:00+00:00"
    restored = identity.MaintenanceRunIdentity.model_validate_json(elapsed.model_dump_json())
    assert restored == elapsed and restored.coding_deadline().expired
    assert restored.coding_deadline().source == "maintenance"
    assert restored.coding_deadline().scope == "coding"
    for value in ("", "2000-01-01", "2000-01-01T00:00:00+01:00", "x" * 65, True):
        with pytest.raises(ValueError):
            identity.allocate_identity(intent(identity), coding_expires_at=value)
    raw = elapsed.model_dump()
    del raw["coding_expires_at"]
    with pytest.raises(ValueError):
        identity.MaintenanceRunIdentity.model_validate(raw)


@pytest.mark.parametrize("field", ["git_delivery_task_id", "coding_expires_at"])
def test_corrupted_phase_id_is_rejected_before_lookup(identity, caplog, capsys, field):
    class Hostile:
        def __repr__(self):
            raise AssertionError("secret-canary-phase")

    allocated = identity.allocate_identity(intent(identity))
    object.__setattr__(allocated, field, Hostile())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for phase in identity.MaintenanceTaskPhase:
            with pytest.raises(ValueError) as error:
                identity.task_id_for(allocated, phase)
            assert "secret-canary" not in str(error.value)
        with pytest.raises(ValueError) as error:
            allocated.coding_deadline()
        assert "secret-canary" not in str(error.value)
    assert not captured and "secret-canary" not in caplog.text
    output = capsys.readouterr()
    assert not output.out and not output.err


@pytest.mark.parametrize(
    "change",
    [
        {"tenant": "tenant-b"},
        {"subject": "user-b"},
        {"idempotency_key": "request-2"},
        {"request_json": '{"case":"different","configuration":"pinned"}'},
        {"request_json": '{"case":"closed-integer-range-v1","configuration":"changed"}'},
    ],
)
def test_every_intent_field_changes_exact_identity(identity, change):
    original, changed = intent(identity), intent(identity, **change)
    assert original != changed and original.fingerprint != changed.fingerprint


@pytest.mark.parametrize(
    "field,bound", [("tenant", 512), ("subject", 512), ("idempotency_key", 128)]
)
def test_identity_bounds_do_not_normalize_authority(identity, field, bound):
    assert getattr(intent(identity, **{field: "x" * bound}), field) == "x" * bound
    for value in ("", " x", "x ", "x\n", "x\x00", "x" * (bound + 1), "\ud800", True):
        with pytest.raises(ValueError):
            intent(identity, **{field: value})


def test_canonical_json_keeps_numeric_types_distinct(identity):
    integer = intent(identity, request_json='{"value":1}')
    boolean = intent(identity, request_json='{"value":true}')
    decimal = intent(identity, request_json='{"value":1.0}')
    assert len({integer.fingerprint, boolean.fingerprint, decimal.fingerprint}) == 3


@pytest.mark.parametrize(
    "value",
    [
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":1e9999}',
        "[]",
        "{",
        '"text"',
        '{"a":"\\ud800"}',
        " " * 65537,
    ],
)
def test_invalid_json_is_not_reservable(identity, value):
    with pytest.raises(ValueError):
        intent(identity, request_json=value)


@pytest.mark.parametrize("field", ["tenant", "subject", "idempotency_key", "request_json"])
def test_corrupted_intent_is_revalidated_without_diagnostic_disclosure(
    identity, field, caplog, capsys
):
    class Hostile:
        def __repr__(self):
            raise AssertionError("secret-canary-repr")

        def __str__(self):
            raise AssertionError("secret-canary-str")

    original = intent(
        identity,
        tenant="secret-canary-tenant",
        subject="secret-canary-subject",
        request_json='{"private":"secret-canary-request"}',
    )
    object.__setattr__(original, field, Hostile())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        for operation in (identity.copy_intent, identity.allocate_identity):
            with pytest.raises(ValueError) as error:
                operation(original)
            assert "secret-canary" not in str(error.value)
    assert not captured
    assert "secret-canary" not in caplog.text
    output = capsys.readouterr()
    assert not output.out and not output.err
