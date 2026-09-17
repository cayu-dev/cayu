"""Export event metadata and projection provenance, not receipt authorization."""

from __future__ import annotations

import asyncio

import pytest

from cayu.budgets.base import InMemoryBudgetStore
from cayu.events import (
    SESSION_EXPORT_EVENT_FIELDS,
    SESSION_EXPORT_EVENT_TYPES,
    Event,
    EventType,
    copy_event,
    event_payload_authority_is_runtime_generated,
    event_with_runtime_payload_authority,
    validate_session_export_event,
)
from cayu.runtime._event_projection import (
    EVENT_PAYLOAD_POLICIES,
    PRIVATE_EVENT_AUTHORITY,
    prepare_new_runtime_event,
    project_persisted_runtime_event,
    project_runtime_event,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.sessions.base import EventQuery, InMemorySessionStore, RunRequest, SessionIdentity
from cayu.vaults.redaction import SecretRedactor


def _event(event_type: EventType) -> Event:
    return Event(
        type=event_type,
        session_id="export-source-session",
        payload={"export_commitment": "a" * 64, "output_commitment": "b" * 64},
    )


def _attest(event: Event) -> Event:
    return event_with_runtime_payload_authority(event, *SESSION_EXPORT_EVENT_FIELDS)


def test_export_event_family_and_policy_are_exhaustive() -> None:
    assert {event.value for event in SESSION_EXPORT_EVENT_TYPES} == {
        "session.export.published",
        "session.export.released",
        "session.export.retired",
    }
    assert {
        event for event in EventType if event.value.startswith("session.export.")
    } == SESSION_EXPORT_EVENT_TYPES
    assert set(EVENT_PAYLOAD_POLICIES) == set(EventType)
    for event_type in SESSION_EXPORT_EVENT_TYPES:
        policy = EVENT_PAYLOAD_POLICIES[event_type]
        assert policy.owned_keys == SESSION_EXPORT_EVENT_FIELDS
        assert policy.authority_keys == SESSION_EXPORT_EVENT_FIELDS
        assert policy.public_authority_keys == SESSION_EXPORT_EVENT_FIELDS
        assert not policy.owned_nested_paths
        assert not policy.aliased_authority_keys


@pytest.mark.parametrize("event_type", sorted(SESSION_EXPORT_EVENT_TYPES))
def test_writer_rejects_untrusted_export_before_any_persistence(event_type) -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        event = _event(event_type)
        await store.create(
            RunRequest(agent_name="assistant", session_id=event.session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        writer = RuntimeEventWriter(
            session_store=store, budget_store=InMemoryBudgetStore(), event_sinks=[]
        )
        before = await store.query_events(EventQuery(session_id=event.session_id))
        for candidate in (
            event,
            _attest(event).model_copy(update={"payload": {**event.payload, "principal": "raw"}}),
            Event.model_validate_json(_attest(event).model_dump_json()),
        ):
            with pytest.raises(ValueError, match="Session export events"):
                await writer.emit(candidate)
            assert await store.query_events(EventQuery(session_id=event.session_id)) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("event_type", sorted(SESSION_EXPORT_EVENT_TYPES))
@pytest.mark.parametrize("collisions", [False, True])
def test_export_event_prepare_copy_and_public_round_trip(event_type, collisions) -> None:
    event = _attest(_event(event_type))
    redactor = SecretRedactor([*event.payload, *event.payload.values()] if collisions else [])
    prepared = prepare_new_runtime_event(copy_event(event), redactor=redactor)
    assert prepared.payload == event.payload
    assert prepare_new_runtime_event(prepared, redactor=redactor).payload == event.payload
    assert project_runtime_event(prepared, sequence=1, redactor=redactor).payload == event.payload
    decoded = Event.model_validate_json(prepared.model_dump_json())
    assert decoded == prepared
    for key, value in event.payload.items():
        assert not event_payload_authority_is_runtime_generated(
            decoded, field_name=key, value=value
        )
    with pytest.raises(ValueError, match="runtime-attested"):
        prepare_new_runtime_event(decoded, redactor=redactor)
    assert (
        project_persisted_runtime_event(decoded, sequence=1, redactor=redactor).payload
        == event.payload
    )
    assert set(project_runtime_event(decoded, sequence=1, redactor=redactor).payload.values()) == {
        PRIVATE_EVENT_AUTHORITY
    }


@pytest.mark.parametrize("event_type", sorted(SESSION_EXPORT_EVENT_TYPES))
@pytest.mark.parametrize("field", sorted(SESSION_EXPORT_EVENT_FIELDS))
@pytest.mark.parametrize(
    "bad",
    [
        None,
        True,
        1,
        [],
        {},
        "",
        "a" * 63,
        "a" * 65,
        "A" * 64,
        "g" * 64,
        "sha256:" + "a" * 64,
        "a" * 63 + "\n",
    ],
)
def test_export_event_rejects_invalid_commitments(event_type, field, bad) -> None:
    event = _attest(_event(event_type))
    event.payload[field] = bad
    with pytest.raises(ValueError, match="runtime-attested"):
        prepare_new_runtime_event(event, redactor=SecretRedactor())
    for project in (project_runtime_event, project_persisted_runtime_event):
        assert field not in project(event, sequence=1, redactor=SecretRedactor()).payload


@pytest.mark.parametrize("event_type", sorted(SESSION_EXPORT_EVENT_TYPES))
@pytest.mark.parametrize("field", sorted(SESSION_EXPORT_EVENT_FIELDS))
def test_export_event_requires_both_exact_attestations(event_type, field) -> None:
    event = _event(event_type)
    with pytest.raises(ValueError, match="runtime-attested"):
        validate_session_export_event(event)
    event = event_with_runtime_payload_authority(event, field)
    with pytest.raises(ValueError, match="runtime-attested"):
        prepare_new_runtime_event(event, redactor=SecretRedactor())
    event = _attest(event)
    event.payload[field] = "c" * 64
    with pytest.raises(ValueError, match="runtime-attested"):
        prepare_new_runtime_event(event, redactor=SecretRedactor())
    event.payload.pop(field)
    with pytest.raises(ValueError, match="exactly two"):
        prepare_new_runtime_event(event, redactor=SecretRedactor())


@pytest.mark.parametrize("event_type", sorted(SESSION_EXPORT_EVENT_TYPES))
@pytest.mark.parametrize("extra", ["inline_output", "source", "principal", "payload", "receipt"])
def test_export_event_rejects_and_never_projects_extra_content(event_type, extra) -> None:
    event = _attest(_event(event_type))
    event.payload[extra] = {"secret": "source-content-canary"}
    with pytest.raises(ValueError, match="exactly two"):
        prepare_new_runtime_event(event, redactor=SecretRedactor())
    for project in (project_runtime_event, project_persisted_runtime_event):
        projected = project(event, sequence=1, redactor=SecretRedactor())
        assert projected.payload == _event(event_type).payload
        assert "source-content-canary" not in projected.model_dump_json()


@pytest.mark.parametrize("event_type", sorted(SESSION_EXPORT_EVENT_TYPES))
def test_export_event_diagnostics_do_not_render_mutated_values(
    event_type, caplog, capsys, recwarn
) -> None:
    canary = "mutated-export-secret-canary"

    class Hostile:
        def __str__(self):
            raise AssertionError(canary)

        def __repr__(self):
            raise AssertionError(canary)

    event = _attest(_event(event_type))
    event.payload["output_commitment"] = Hostile()
    with pytest.raises(ValueError, match="runtime-attested") as error:
        prepare_new_runtime_event(event, redactor=SecretRedactor())
    for project in (project_runtime_event, project_persisted_runtime_event):
        assert (
            "output_commitment" not in project(event, sequence=1, redactor=SecretRedactor()).payload
        )
    captured = capsys.readouterr()
    assert canary not in str(error.value) + repr(error.value)
    assert canary not in caplog.text + captured.out + captured.err
    assert all(canary not in str(warning.message) for warning in recwarn)
