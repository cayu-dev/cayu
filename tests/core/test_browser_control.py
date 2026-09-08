"""Content-free browser control authority and pure transition contracts.

These are foundation tests, not authenticated takeover acceptance coverage.
"""

from __future__ import annotations

import asyncio
import warnings
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from tests.core._execution_profile_fixtures import create_admitted_session

from cayu import RunRequest, SQLiteSessionStore
from cayu._validation import MAX_PORTABLE_JSON_INTEGER
from cayu.runtime import InMemorySessionStore
from cayu.runtime._browser_control_checkpoint import (
    BrowserControlCheckpointMutation,
    browser_control_checkpoint_mutation_scope,
    browser_control_checkpoint_read_scope,
)
from cayu.runtime._browser_control_model import validate_browser_model_publication
from cayu.runtime._browser_control_publication import BrowserControlPublication
from cayu.runtime._checkpoint_store import runtime_checkpoint_session_store
from cayu.runtime.browser_control import (
    BrowserControlCheckpoint,
    BrowserControlConflict,
    BrowserControlIdentity,
    BrowserControlPage,
    BrowserControlRecord,
    BrowserOperatorIdentity,
    BrowserOperatorPurpose,
    BrowserTakeoverRequest,
    request_browser_takeover,
)
from cayu.runtime.checkpoints import (
    BROWSER_CONTROLS_CHECKPOINT_KEY,
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.sessions import SessionIdentity, SessionOperationPublication


def operator_purpose() -> BrowserOperatorPurpose:
    return BrowserOperatorPurpose(code="login", expected_origins=("https://app-native.test",))


def identity() -> BrowserControlIdentity:
    return BrowserControlIdentity(
        session_id="session",
        session_instance_id="instance",
        run_epoch=1,
        interaction_id="interaction",
        execution_profile_fingerprint="a" * 64,
        environment_name="browser",
        allocation_fingerprint="b" * 64,
        browser_session_id="bs_fixture",
        worker_instance_id="worker",
        operator_purpose=operator_purpose(),
    )


def request() -> BrowserTakeoverRequest:
    return BrowserTakeoverRequest(
        request_id="bt_" + "1" * 32,
        identity=identity(),
        operator=BrowserOperatorIdentity(
            subject="operator",
            tenant="tenant",
            operator_session_id="authenticated-session",
            authorization_policy_fingerprint="c" * 64,
        ),
        expected_record_revision=1,
        expected_control_epoch=1,
        pages=(BrowserControlPage(page_id="page", revision="revision", control_epoch=1),),
        purpose_code="login",
        requested_at_ms=1000,
        expires_at_ms=2000,
        maximum_until_ms=5000,
    )


def test_request_blocks_agent_without_granting_operator_input() -> None:
    record = BrowserControlRecord(identity=identity())
    pending = request_browser_takeover(record, request(), now_ms=1000)
    assert record.state == "agent_controlled"
    assert pending.state == "takeover_requested"
    assert pending.revision == 2
    assert pending.control_epoch == 1
    assert pending.lease_until_ms is None
    # Read-only acknowledgement reconciliation must not extend authority.
    assert request_browser_takeover(pending, request(), now_ms=6000) == pending
    restored = BrowserControlRecord.model_validate_json(pending.model_dump_json())
    assert restored == pending


@pytest.mark.parametrize(
    "counts",
    [
        ({"page_id": "page", "operations": True},),
        ({"page_id": "page", "operations": 0},),
        ({"page_id": "page", "operations": -1},),
        ({"page_id": "page", "operations": 1.0},),
        ({"page_id": "page", "operations": MAX_PORTABLE_JSON_INTEGER + 1},),
        ({"page_id": "private-canary\x00", "operations": 1},),
        ({"page_id": "private-canary\ud800", "operations": 1},),
        ({"page_id": "page", "operations": 1},) * 2,
        ({"page_id": "z", "operations": 1}, {"page_id": "a", "operations": 1}),
        (
            {"page_id": "a", "operations": MAX_PORTABLE_JSON_INTEGER},
            {"page_id": "b", "operations": 1},
        ),
        tuple({"page_id": f"page-{index:03}", "operations": 1} for index in range(129)),
    ],
)
def test_operator_accounting_rejects_malformed_or_unbounded_evidence(counts):
    record = BrowserControlRecord(
        identity=identity(),
        request=request(),
        revision=3,
        control_epoch=2,
        state="operator_controlled",
        lease_until_ms=1500,
    )
    with pytest.raises(ValueError) as failure:
        record.model_copy(update={"operator_page_operations": counts})
    assert "private-canary" not in str(failure.value)
    assert "private-canary" not in repr(failure.value)


def test_operator_accounting_is_cumulative_across_pages_and_takeovers():
    from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
    from cayu.runtime.browser_control import BrowserTextInputIntent

    pages = tuple(
        BrowserControlPage(page_id=name, revision="revision", control_epoch=1)
        for name in ("a", "b")
    )
    takeover = request().model_copy(update={"pages": pages})
    record = BrowserControlRecord(
        identity=identity(),
        request=takeover,
        revision=3,
        control_epoch=2,
        state="operator_controlled",
        lease_until_ms=1500,
    )
    for sequence, page in enumerate((pages[0], pages[1], pages[0]), start=1):
        pending = BrowserControlCoordinator._input_admission_successor(
            record,
            BrowserTextInputIntent(
                identity=record.identity,
                request_id=takeover.request_id,
                expected_record_revision=record.revision,
                expected_control_epoch=record.control_epoch,
                page=page,
                input_sequence=sequence,
            ),
        )
        record = BrowserControlCoordinator._input_successor(pending)
    assert [(item.page_id, item.operations) for item in record.operator_page_operations] == [
        ("a", 2),
        ("b", 1),
    ]
    assert BrowserControlRecord.model_validate_json(record.model_dump_json()) == record
    with pytest.raises(ValueError):
        record.model_copy(update={"operator_page_operations": ()})
    handed_back = BrowserControlCoordinator._handback_successor(
        record.model_copy(update={"revision": record.revision + 1, "state": "handback_pending"})
    )
    again = request_browser_takeover(
        handed_back,
        takeover.model_copy(
            update={
                "request_id": "bt_" + "2" * 32,
                "expected_record_revision": handed_back.revision,
                "expected_control_epoch": handed_back.control_epoch,
            }
        ),
        now_ms=1000,
    )
    assert again.operator_page_operations == record.operator_page_operations
    assert again.settled_input_sequence == record.settled_input_sequence == 3


@pytest.mark.parametrize(
    "field,value",
    [
        ("session_id", "other"),
        ("session_instance_id", "other"),
        ("run_epoch", 2),
        ("interaction_id", "other"),
        ("execution_profile_fingerprint", "d" * 64),
        ("environment_name", "other"),
        ("allocation_fingerprint", "d" * 64),
        ("browser_session_id", "other"),
        ("worker_instance_id", "other"),
        ("profile_checkpoint_policy", "on_close"),
    ],
)
def test_identity_substitution_cannot_acquire(field: str, value: object) -> None:
    altered = request().model_copy(
        update={"identity": identity().model_copy(update={field: value})}
    )
    with pytest.raises(BrowserControlConflict):
        request_browser_takeover(BrowserControlRecord(identity=identity()), altered, now_ms=1000)


@pytest.mark.parametrize(
    "changes",
    [
        {"purpose_code": "support"},
        {"checkpoint_consent": "allow"},
        {"expires_at_ms": 2001},
        {"maximum_until_ms": 6000},
        {"requested_at_ms": 999},
        {"expected_record_revision": 2},
        {"expected_control_epoch": 2},
        {"pages": (BrowserControlPage(page_id="other", revision="revision", control_epoch=1),)},
        {"operator": request().operator.model_copy(update={"tenant": "other"})},
        {"operator": request().operator.model_copy(update={"subject": "other"})},
        {"operator": request().operator.model_copy(update={"operator_session_id": "other"})},
        {
            "operator": request().operator.model_copy(
                update={"authorization_policy_fingerprint": "d" * 64}
            )
        },
    ],
)
def test_same_request_identity_requires_complete_material(changes: dict[str, object]) -> None:
    pending = request_browser_takeover(
        BrowserControlRecord(identity=identity()), request(), now_ms=1000
    )
    with pytest.raises(BrowserControlConflict):
        request_browser_takeover(pending, request().model_copy(update=changes), now_ms=1000)


@pytest.mark.parametrize("now", [999, 2000, 5000])
def test_acquisition_time_is_half_open(now: int) -> None:
    with pytest.raises(BrowserControlConflict):
        request_browser_takeover(BrowserControlRecord(identity=identity()), request(), now_ms=now)


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": True},
        {"schema_version": 2},
        {"revision": True},
        {"control_epoch": 0},
        {"state": "future_state"},
        {"state": "operator_controlled"},
        {"lease_until_ms": 1000},
        {"pending_input_sequence": 1},
        {"sensitive_entry": True},
        {"capture_restricted": 1},
        {"credential": "must-not-be-a-field"},
    ],
)
def test_record_rejects_invalid_or_secret_bearing_shapes(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BrowserControlRecord(identity=identity()).model_copy(update=changes)


def test_copy_owns_pages_and_preserves_sparse_field_set() -> None:
    pages = list(request().pages)
    copied = request().model_copy(update={"pages": pages})
    pages.clear()
    assert len(copied.pages) == 1
    record = BrowserControlRecord(identity=identity())
    assert record.model_copy().model_fields_set == {"identity"}
    assert record.model_copy().model_dump(exclude_unset=True) == record.model_dump(
        exclude_unset=True
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"revision": 1},
        {"control_epoch": 2},
        {"lease_until_ms": 2000},
        {"state": "operator_controlled", "lease_until_ms": 2000},
        {"state": "handback_pending"},
        {"state": "agent_controlled"},
        {"state": "control_uncertain", "control_epoch": 4},
        {"sensitive_entry": True, "capture_restricted": True},
    ],
)
def test_reconstruction_rejects_inconsistent_control_generation(changes: dict[str, object]) -> None:
    pending = request_browser_takeover(
        BrowserControlRecord(identity=identity()), request(), now_ms=1000
    )
    raw = pending.model_dump(mode="json")
    raw.update(changes)
    with pytest.raises(ValidationError):
        BrowserControlRecord.model_validate(raw)


def test_acquired_and_handed_back_record_shapes_roundtrip() -> None:
    pending = request_browser_takeover(
        BrowserControlRecord(identity=identity()), request(), now_ms=1000
    )
    acquired = pending.model_copy(
        update={"state": "operator_controlled", "control_epoch": 2, "lease_until_ms": 2000}
    )
    assert BrowserControlRecord.model_validate_json(acquired.model_dump_json()) == acquired
    handed_back = acquired.model_copy(
        update={
            "state": "agent_controlled",
            "control_epoch": 3,
            "lease_until_ms": None,
            "fresh_observation_required": True,
        }
    )
    assert BrowserControlRecord.model_validate_json(handed_back.model_dump_json()) == handed_back


def test_mutated_nested_model_is_revalidated_without_serialization_warning(capsys, caplog) -> None:
    class Canary:
        def __repr__(self) -> str:
            return "credential-canary-not-for-diagnostics"

    malformed = request()
    object.__setattr__(malformed.operator, "subject", Canary())
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError) as error:
            request_browser_takeover(
                BrowserControlRecord(identity=identity()), malformed, now_ms=1000
            )
    output = capsys.readouterr()
    assert "credential-canary" not in str(error.value)
    assert "credential-canary" not in repr(error.value)
    assert "credential-canary" not in output.out + output.err + caplog.text
    assert not captured


def test_checkpoint_replacement_compares_complete_record() -> None:
    initial = BrowserControlRecord(identity=identity())
    checkpoint = BrowserControlCheckpoint().replace_record(expected=None, desired=initial)
    pending = request_browser_takeover(initial, request(), now_ms=1000)
    changed = checkpoint.replace_record(expected=initial, desired=pending)
    assert changed.records == (pending,)
    assert checkpoint.records == (initial,)
    assert BrowserControlCheckpoint.model_validate_json(changed.model_dump_json()) == changed
    with pytest.raises(BrowserControlConflict):
        changed.replace_record(expected=initial, desired=pending)
    with pytest.raises(BrowserControlConflict):
        checkpoint.replace_record(
            expected=initial.model_copy(update={"capture_restricted": True}), desired=pending
        )


def test_checkpoint_cannot_initialize_with_acquired_or_pending_authority() -> None:
    initial = BrowserControlRecord(identity=identity())
    pending = request_browser_takeover(initial, request(), now_ms=1000)
    with pytest.raises(BrowserControlConflict):
        BrowserControlCheckpoint().replace_record(expected=None, desired=pending)


@pytest.mark.parametrize("field", ["session_id", "session_instance_id"])
def test_checkpoint_rejects_mixed_parent_authority(field: str) -> None:
    first = BrowserControlRecord(identity=identity())
    second = BrowserControlRecord(
        identity=identity().model_copy(update={field: "other", "browser_session_id": "bs_second"})
    )
    with pytest.raises(ValidationError):
        BrowserControlCheckpoint(records=(first, second))


def test_checkpoint_rejects_duplicate_and_unsorted_allocations() -> None:
    first = BrowserControlRecord(identity=identity())
    second = BrowserControlRecord(
        identity=identity().model_copy(update={"browser_session_id": "bs_second"})
    )
    for records in [(first, first), (second, first)]:
        with pytest.raises(ValidationError):
            BrowserControlCheckpoint(records=records)


def test_checkpoint_version_reserves_control_authority() -> None:
    from cayu.runtime.checkpoints import (
        BROWSER_CONTROLS_CHECKPOINT_KEY,
        CHECKPOINT_SCHEMA_VERSION_KEY,
        CURRENT_CHECKPOINT_SCHEMA_VERSION,
        decode_runtime_checkpoint,
        runtime_checkpoint_writer_view,
    )

    controls = BrowserControlCheckpoint(records=(BrowserControlRecord(identity=identity()),))
    checkpoint = {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        BROWSER_CONTROLS_CHECKPOINT_KEY: controls.model_dump(mode="json"),
    }
    assert decode_runtime_checkpoint(checkpoint, session_id="session") == checkpoint
    for writer_version in range(1, CURRENT_CHECKPOINT_SCHEMA_VERSION):
        with pytest.raises(ValueError, match="Browser control"):
            runtime_checkpoint_writer_view(
                checkpoint, writer_version=writer_version, session_id="session"
            )
    old = {**checkpoint, CHECKPOINT_SCHEMA_VERSION_KEY: 8, "application": "preserved"}
    migrated = decode_runtime_checkpoint(old, session_id="session")
    assert migrated == {
        CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
        "application": "preserved",
    }


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_store_control_authority_cannot_be_forged_erased_or_raced(
    backend: str, wrapped: bool, tmp_path
) -> None:
    asyncio.run(_assert_store_control_authority(backend, wrapped, tmp_path))


async def _assert_store_control_authority(backend: str, wrapped: bool, tmp_path) -> None:
    raw_store = (
        InMemorySessionStore()
        if backend == "memory"
        else SQLiteSessionStore(tmp_path / "browser-control.sqlite")
    )
    store = runtime_checkpoint_session_store(raw_store) if wrapped else raw_store
    try:
        await store.create(
            RunRequest(agent_name="agent", session_id="session", messages=[]),
            identity=SessionIdentity(provider_name="provider", model="model"),
        )
        initial = BrowserControlRecord(identity=identity())
        controls = BrowserControlCheckpoint(records=(initial,))
        publication = {
            CHECKPOINT_SCHEMA_VERSION_KEY: CURRENT_CHECKPOINT_SCHEMA_VERSION,
            BROWSER_CONTROLS_CHECKPOINT_KEY: controls.model_dump(mode="json"),
        }
        # An identical caller-authored value cannot create private authority.
        await store.checkpoint("session", publication)
        assert BROWSER_CONTROLS_CHECKPOINT_KEY not in (await store.load_checkpoint("session"))
        with browser_control_checkpoint_mutation_scope(
            BrowserControlCheckpointMutation("session", None, controls)
        ):
            await store.checkpoint("session", publication)

        observed = []

        def ordinary_transform(_session, checkpoint):
            observed.append(checkpoint)
            return {"application": "changed"}

        await store.transform_checkpoint("session", ordinary_transform)
        assert BROWSER_CONTROLS_CHECKPOINT_KEY not in observed[0]
        retained = await store.load_checkpoint("session")
        assert retained[BROWSER_CONTROLS_CHECKPOINT_KEY] == controls.model_dump(mode="json")
        assert retained["application"] == "changed"

        pending = request_browser_takeover(initial, request(), now_ms=1000)
        desired = controls.replace_record(expected=initial, desired=pending)
        update = {**retained, BROWSER_CONTROLS_CHECKPOINT_KEY: desired.model_dump(mode="json")}
        command = BrowserControlCheckpointMutation("session", controls, desired)
        with browser_control_checkpoint_mutation_scope(command):
            await store.checkpoint("session", update)
        # Another publication against the old snapshot cannot consume authority twice.
        with (
            browser_control_checkpoint_mutation_scope(command),
            pytest.raises(BrowserControlConflict),
        ):
            await store.checkpoint("session", update)
        assert (await store.load_checkpoint("session"))[BROWSER_CONTROLS_CHECKPOINT_KEY] == (
            desired.model_dump(mode="json")
        )
    finally:
        if isinstance(raw_store, SQLiteSessionStore):
            await raw_store.close()


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_control_publication_is_bound_to_admitted_invocation(backend: str, tmp_path) -> None:
    asyncio.run(_assert_control_publication(backend, tmp_path))


def test_publication_revalidates_before_receipt_serialization(capsys, caplog) -> None:
    class Canary:
        def __repr__(self) -> str:
            return "private-operator-canary"

    publication = BrowserControlPublication(
        BrowserControlCheckpointMutation(
            "session",
            None,
            BrowserControlCheckpoint(records=(BrowserControlRecord(identity=identity()),)),
        )
    )
    object.__setattr__(
        publication.mutation.desired.records[0].identity, "worker_instance_id", Canary()
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(ValidationError) as error:
            publication.receipt()
        assert "private-operator-canary" not in str(error.value) + repr(error.value)
        with pytest.raises(ValidationError):
            _ = publication.storage_key
    output = capsys.readouterr()
    assert not captured
    assert "private-operator-canary" not in output.out + output.err + caplog.text


async def _assert_control_publication(backend: str, tmp_path) -> None:
    now = datetime(1970, 1, 1, 0, 0, 1, tzinfo=UTC)
    raw_store = (
        InMemorySessionStore(ownership_clock=lambda: now)
        if backend == "memory"
        else SQLiteSessionStore(
            tmp_path / "control-publication.sqlite", ownership_clock=lambda: now
        )
    )
    store = runtime_checkpoint_session_store(raw_store)
    try:
        admitted = await create_admitted_session(
            raw_store,
            request=RunRequest(
                agent_name="agent", session_id="session", environment_name="browser", messages=[]
            ),
            provider_name="provider",
            model="model",
        )
        exact_identity = identity().model_copy(
            update={
                "session_instance_id": admitted.session.instance_id,
                "interaction_id": admitted.active_invocation_profile.interaction_id,
                "execution_profile_fingerprint": admitted.active_invocation_profile.profile.fingerprint,
            }
        )

        async def publish(command: BrowserControlPublication) -> None:
            with command.scope():
                await store.publish_session_operation_guarded_with_store_time(
                    "session",
                    idempotency_key=command.storage_key,
                    operation_transform=command.transform,
                    commit_guard=lambda: None,
                    commit_time_guard=command.validate_commit_time,
                    events=[],
                    expected_run_epoch=admitted.session.run_epoch,
                )

        for field, value in {
            "session_instance_id": "other-instance",
            "run_epoch": 2,
            "interaction_id": "other-interaction",
            "execution_profile_fingerprint": "d" * 64,
            "environment_name": "other-environment",
        }.items():
            wrong = BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    "session",
                    None,
                    BrowserControlCheckpoint(
                        records=(
                            BrowserControlRecord(
                                identity=exact_identity.model_copy(update={field: value})
                            ),
                        )
                    ),
                )
            )
            with pytest.raises(BrowserControlConflict):
                await publish(wrong)
            assert await store.load_session_operation("session", wrong.storage_key) is None
            assert BROWSER_CONTROLS_CHECKPOINT_KEY not in (await store.load_checkpoint("session"))

        initial = BrowserControlRecord(identity=exact_identity)
        controls = BrowserControlCheckpoint(records=(initial,))
        command = BrowserControlPublication(
            BrowserControlCheckpointMutation("session", None, controls)
        )
        await publish(command)
        assert (
            await store.load_session_operation("session", command.storage_key) == command.receipt()
        )
        first_request = request().model_copy(update={"identity": exact_identity})
        second_request = first_request.model_copy(update={"request_id": "bt_" + "2" * 32})
        contenders = [
            BrowserControlPublication(
                BrowserControlCheckpointMutation(
                    "session",
                    controls,
                    controls.replace_record(
                        expected=initial,
                        desired=request_browser_takeover(initial, candidate, now_ms=1000),
                    ),
                )
            )
            for candidate in (first_request, second_request)
        ]
        start = asyncio.Event()

        async def compete(candidate):
            await start.wait()
            await publish(candidate)

        workers = [asyncio.create_task(compete(candidate)) for candidate in contenders]
        start.set()
        outcomes = await asyncio.gather(*workers, return_exceptions=True)
        assert sum(outcome is None for outcome in outcomes) == 1
        assert sum(isinstance(outcome, BrowserControlConflict) for outcome in outcomes) == 1
        winner = contenders[outcomes.index(None)]
        assert await store.load_session_operation("session", winner.storage_key) == winner.receipt()
        retained = await store.load_checkpoint("session")
        assert retained[BROWSER_CONTROLS_CHECKPOINT_KEY] == winner.mutation.desired.model_dump(
            mode="json"
        )
        operation = {
            "record_type": "cayu.browser-operation",
            "schema_version": 1,
            "browser_session_id": exact_identity.browser_session_id,
            "parent_session_id": exact_identity.session_id,
            "parent_run_epoch": exact_identity.run_epoch,
            "execution_profile_fingerprint": exact_identity.execution_profile_fingerprint,
            "environment_name": exact_identity.environment_name,
            "allocation_fingerprint": exact_identity.allocation_fingerprint,
        }

        async def publish_model(state: str) -> None:
            def transform(session, checkpoint, _current):
                records = {"model-operation": {**operation, "state": state}}
                validate_browser_model_publication(
                    checkpoint,
                    session=session,
                    operation_records=records,
                    operation_name="click",
                )
                return SessionOperationPublication(checkpoint=checkpoint, operation_records=records)

            with browser_control_checkpoint_read_scope("session"):
                await store.publish_session_operation(
                    "session",
                    idempotency_key="model-operation",
                    operation_transform=transform,
                    events=[],
                    expected_run_epoch=exact_identity.run_epoch,
                )

        for state in ("intent", "dispatched"):
            with pytest.raises(BrowserControlConflict, match="fenced"):
                await publish_model(state)
            assert await store.load_session_operation("session", "model-operation") is None
        await publish_model("terminal")
        assert (await store.load_session_operation("session", "model-operation"))[
            "state"
        ] == "terminal"
        assert (await store.load_checkpoint("session"))[BROWSER_CONTROLS_CHECKPOINT_KEY] == (
            retained[BROWSER_CONTROLS_CHECKPOINT_KEY]
        )
    finally:
        if isinstance(raw_store, SQLiteSessionStore):
            await raw_store.close()


def test_reconnect_advances_only_the_same_view_only_guest():
    from cayu.runtime.browser_control import rebound_browser_control_successor

    old = BrowserControlRecord(identity=identity(), state="control_uncertain")
    current = old.identity.model_copy(update={"run_epoch": 2, "interaction_id": "resumed"})
    desired = rebound_browser_control_successor(old, current)
    assert desired.control_epoch == old.control_epoch + 1
    assert desired.revision == old.revision + 1
    assert desired.fresh_observation_required and desired.state == "agent_controlled"
    checkpoint = BrowserControlCheckpoint(records=(old,))
    changed = checkpoint.replace_record(expected=old, desired=desired)
    assert changed.records == (desired,)
    with pytest.raises(BrowserControlConflict):
        changed.replace_record(expected=old, desired=desired)
    with pytest.raises(BrowserControlConflict):
        checkpoint.replace_record(
            expected=old, desired=desired.model_copy(update={"fresh_observation_required": False})
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("worker_instance_id", "other-worker"),
        ("allocation_fingerprint", "f" * 64),
        ("execution_profile_fingerprint", "f" * 64),
        ("browser_session_id", "other-browser"),
        ("session_id", "other-session"),
        ("session_instance_id", "other-instance"),
        ("environment_name", "other-environment"),
        ("run_epoch", 1),
    ],
)
def test_reconnect_never_substitutes_an_allocation_or_reuses_an_epoch(field, value):
    from cayu.runtime.browser_control import rebound_browser_control_successor

    old = BrowserControlRecord(identity=identity(), state="control_uncertain")
    desired = old.identity.model_copy(
        update={"run_epoch": 2, "interaction_id": "resumed", field: value}
    )
    with pytest.raises(BrowserControlConflict):
        rebound_browser_control_successor(old, desired)


def test_reconnect_preserves_active_takeover_and_sensitive_entry_fences():
    from cayu.runtime.browser_control import rebound_browser_control_successor

    old = request_browser_takeover(
        BrowserControlRecord(identity=identity()), request(), now_ms=1000
    )
    desired = old.identity.model_copy(update={"run_epoch": 2, "interaction_id": "resumed"})
    with pytest.raises(BrowserControlConflict):
        rebound_browser_control_successor(old, desired)
    fenced = old.model_copy(update={"state": "control_uncertain", "revision": old.revision + 1})
    with pytest.raises(BrowserControlConflict):
        rebound_browser_control_successor(fenced, desired)
    assert fenced.request == old.request
