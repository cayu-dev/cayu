from __future__ import annotations

import asyncio
import base64
import logging
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import SecretStr

import cayu.core.events as events_module
import cayu.runtime._event_projection as event_projection_module
from cayu import CayuApp
from cayu._validation import MAX_DURABLE_JSON_INTEGER
from cayu.core.events import (
    Event,
    EventType,
    event_envelope_authority_is_runtime_generated,
    event_id_is_runtime_generated,
    event_payload_authority_is_runtime_generated,
    event_with_durable_sequence,
    event_with_runtime_envelope_authority,
    event_with_runtime_generated_id,
    event_with_runtime_payload_authority,
)
from cayu.core.tools import ToolEffect
from cayu.runtime._event_projection import (
    EVENT_PAYLOAD_POLICIES,
    PRIVATE_EVENT_AUTHORITY,
    REDACTED_CUSTOM_EVENT_TYPE,
    prepare_new_runtime_event,
    private_event_linkage_value,
    public_event_id,
    public_event_linkage_id,
    public_event_sequence,
)
from cayu.runtime._event_projection import (
    project_runtime_event as _project_runtime_event,
)
from cayu.runtime._event_writer import RuntimeEventWriter
from cayu.runtime._run_limits import _validate_ledger_settlement_record
from cayu.runtime._structured_output_tool_round import (
    _structured_output_event,
    _structured_output_validating_event,
)
from cayu.runtime._tool_identity import tool_idempotency_key
from cayu.runtime.approvals import PendingToolApproval, PendingToolCallApproval
from cayu.runtime.budgets import (
    BudgetReconciliation,
    BudgetSettlementRecord,
    BudgetWindow,
    InMemoryBudgetStore,
    budget_reconciliation_payload,
    budget_settlement_event_id,
    budget_settlement_id,
)
from cayu.runtime.event_sinks import EventSink
from cayu.runtime.execution_units import ToolRoundIdentity
from cayu.runtime.public_authority import (
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
)
from cayu.runtime.sessions import (
    EventQuery,
    InMemorySessionStore,
    RunRequest,
    Session,
    SessionIdentity,
)
from cayu.runtime.structured_output import StructuredOutputSpec, StructuredOutputValidation
from cayu.runtime.user_input import PendingUserInput
from cayu.vaults import REDACTED_SECRET, SecretRedactor


def test_event_equality_uses_only_public_durable_fields() -> None:
    event = Event(
        id="event-equality",
        type=EventType.SESSION_STARTED,
        session_id="session-equality",
        payload={"value": 1},
    )
    same_public_event = event_with_durable_sequence(event, 7)

    assert event == same_public_event
    assert Event.__eq__(event, object()) is NotImplemented
    assert event != event.model_copy(update={"payload": {"value": 2}})


def test_non_secret_turn_interaction_ids_match_public_event_envelopes() -> None:
    interaction_id = "ordinary-interaction"
    event = Event(
        type=EventType.TURN_COMPLETED,
        session_id="ordinary-session",
        payload={"interaction_ids": [interaction_id]},
    )

    public = project_runtime_event(event, sequence=1, redactor=SecretRedactor())

    assert public.payload["interaction_ids"] == [interaction_id]


_TERMINAL_CONTROL_KEYS_FOR_TEST = {
    "terminal_outcome",
    "tool_effect",
    "outcome_unknown",
    "manual_reconciliation_required",
    "durable_value_error_code",
    "durable_value_error_path",
}

_TEST_ALIAS_CODEC = PublicAuthorityAliasCodec(
    PublicAuthorityAliasKeyring(
        active_key_id="test",
        keys={
            "test": SecretStr(
                base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
            )
        },
    )
)


def project_runtime_event(
    event: Event,
    *,
    sequence: int,
    redactor: SecretRedactor,
) -> Event:
    return _project_runtime_event(
        event,
        sequence=sequence,
        redactor=redactor,
        public_authority_alias_codec=_TEST_ALIAS_CODEC,
    )


class _RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.events.append(event.model_copy(deep=True))


def test_event_payload_policies_cover_every_exact_builtin_type() -> None:
    assert set(EVENT_PAYLOAD_POLICIES) == set(EventType)
    assert "step" in EVENT_PAYLOAD_POLICIES[EventType.MODEL_STARTED].owned_keys
    assert "step" not in EVENT_PAYLOAD_POLICIES[EventType.SESSION_STARTED].owned_keys
    for policy in EVENT_PAYLOAD_POLICIES.values():
        if "actor" in policy.owned_keys:
            assert {
                ("actor", "source"),
                ("actor", "subject"),
                ("actor", "tenant"),
            } <= policy.owned_nested_paths


def test_pause_projection_schemas_track_the_typed_checkpoint_models() -> None:
    assert (
        frozenset(PendingToolCallApproval.model_fields) | {"arguments_state"}
        == event_projection_module._PENDING_TOOL_CALL_FIELD_NAMES
    )
    assert (
        frozenset(PendingToolApproval.model_fields) | {"arguments_state"}
        == event_projection_module._PENDING_APPROVAL_FIELD_NAMES
    )
    assert (frozenset(PendingUserInput.model_fields) - {"staged_terminals"}) | {
        "arguments_state"
    } == event_projection_module._PENDING_USER_INPUT_FIELD_NAMES


@pytest.mark.parametrize(
    ("field_name", "action_field_name", "event_type"),
    [
        pytest.param(
            "approval_id",
            "approval_id",
            EventType.TOOL_CALL_APPROVAL_REQUESTED,
            id="approval",
        ),
        pytest.param(
            "input_id",
            "input_id",
            EventType.SESSION_AWAITING_USER_INPUT,
            id="user-input",
        ),
        pytest.param(
            "tool_round_id",
            "round_id",
            EventType.TOOL_CALL_STARTED,
            id="tool-round",
        ),
    ],
)
def test_direct_app_linkage_disambiguates_legacy_raw_values_from_public_aliases(
    field_name: str,
    action_field_name: str,
    event_type: EventType,
) -> None:
    async def scenario() -> None:
        session_id = f"session-direct-linkage-{field_name}"
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        private_value = f"private-{field_name}"
        await store.append_event(
            session_id,
            Event(
                type=event_type,
                session_id=session_id,
                payload={field_name: private_value},
            ),
        )
        records = await store.query_events(EventQuery(session_id=session_id))
        assert len(records) == 1
        alias = public_event_linkage_id(records[0].sequence, field_name)
        pending_value = "cayu_event_legacy"

        async def query_pending_actions(_query):
            return SimpleNamespace(actions=[SimpleNamespace(**{action_field_name: pending_value})])

        cast("Any", store).query_pending_actions = query_pending_actions
        app = CayuApp(session_store=store, enable_logging=False)

        assert (
            await app._resolve_public_action_linkage(
                session_id=session_id,
                value=pending_value,
                field_name=field_name,
            )
            == pending_value
        )
        pending_value = "different-pending-authority"
        assert (
            await app._resolve_public_action_linkage(
                session_id=session_id,
                value=alias,
                field_name=field_name,
            )
            == private_value
        )
        pending_value = alias
        with pytest.raises(ValueError, match="ambiguous"):
            await app._resolve_public_action_linkage(
                session_id=session_id,
                value=alias,
                field_name=field_name,
            )

    asyncio.run(scenario())


def test_direct_app_linkage_rejects_an_alias_from_another_session() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        for session_id in ("source-session", "requested-session"):
            await store.create(
                RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
                identity=SessionIdentity(provider_name="fake", model="fake-model"),
            )
        await store.append_event(
            "source-session",
            Event(
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="source-session",
                payload={"approval_id": "private-approval"},
            ),
        )
        records = await store.query_events(EventQuery(session_id="source-session"))
        alias = public_event_linkage_id(records[0].sequence, "approval_id")
        app = CayuApp(session_store=store, enable_logging=False)

        with pytest.raises(ValueError, match="not found in the requested session"):
            await app._resolve_public_action_linkage(
                session_id="requested-session",
                value=alias,
                field_name="approval_id",
            )

    asyncio.run(scenario())


def test_direct_app_linkage_rejects_an_event_without_private_authority() -> None:
    async def scenario() -> None:
        store = InMemorySessionStore()
        session_id = "session-without-action-authority"
        await store.create(
            RunRequest(agent_name="assistant", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
        await store.append_event(
            session_id,
            Event(
                type=EventType.HOOK_STARTED,
                session_id=session_id,
                payload={},
            ),
        )
        records = await store.query_events(EventQuery(session_id=session_id))
        alias = public_event_linkage_id(records[0].sequence, "approval_id")
        app = CayuApp(session_store=store, enable_logging=False)

        with pytest.raises(ValueError, match="no private durable authority"):
            await app._resolve_public_action_linkage(
                session_id=session_id,
                value=alias,
                field_name="approval_id",
            )

    asyncio.run(scenario())


def test_replacing_a_generated_event_id_loses_generated_authority_provenance() -> None:
    generated = Event(
        type=EventType.SESSION_STARTED,
        session_id="session",
    )
    replaced = generated.model_copy(update={"id": "caller-secret-id"})

    prepare_new_runtime_event(generated, redactor=SecretRedactor("-"))
    with pytest.raises(ValueError, match=r"event\.event_id"):
        prepare_new_runtime_event(
            replaced,
            redactor=SecretRedactor("caller-secret-id"),
        )


def test_runtime_envelope_authority_survives_short_secret_collisions_only_with_provenance() -> None:
    event = Event(
        type=EventType.INTERACTION_STARTED,
        session_id="runtime-session-id",
        interaction_id="runtime-interaction-id",
        payload={"status": "active"},
    )
    attested = event_with_runtime_envelope_authority(
        event,
        "session_id",
        "interaction_id",
    )

    prepared = prepare_new_runtime_event(attested, redactor=SecretRedactor("-"))

    assert prepared.session_id == event.session_id
    assert prepared.interaction_id == event.interaction_id
    assert event_envelope_authority_is_runtime_generated(
        prepared,
        field_name="session_id",
        value=event.session_id,
    )
    with pytest.raises(ValueError, match=r"event\.session_id"):
        prepare_new_runtime_event(event, redactor=SecretRedactor("-"))


def test_runtime_envelope_authority_is_not_acquired_by_deserialization() -> None:
    event = event_with_runtime_envelope_authority(
        Event(
            type=EventType.INTERACTION_STARTED,
            session_id="runtime-session-id",
            interaction_id="runtime-interaction-id",
            payload={"status": "active"},
        ),
        "session_id",
        "interaction_id",
    )

    deserialized = Event.model_validate(event.model_dump(mode="python"))

    with pytest.raises(ValueError, match=r"event\.session_id"):
        prepare_new_runtime_event(deserialized, redactor=SecretRedactor("-"))


@pytest.mark.parametrize("secret", ["step", "tep", "t"])
def test_exact_builtin_key_ownership_does_not_extend_to_custom_events(
    secret: str,
) -> None:
    redactor = SecretRedactor(secret)
    built_in = prepare_new_runtime_event(
        Event(
            type=EventType.MODEL_STARTED,
            session_id="session",
            payload={"step": 1},
        ),
        redactor=redactor,
    )
    assert built_in.payload == {"step": 1}

    with pytest.raises(ValueError, match="object key"):
        prepare_new_runtime_event(
            Event(
                type=EventType.SESSION_STARTED,
                session_id="session",
                payload={"step": 1},
            ),
            redactor=redactor,
        )
    if secret == "step":
        with pytest.raises(ValueError, match="object key"):
            prepare_new_runtime_event(
                Event(
                    type="custom.demo",
                    session_id="session",
                    payload={"step": 1},
                ),
                redactor=redactor,
            )


@pytest.mark.parametrize(
    ("secret", "generated_id"),
    [
        ("-", "12345678-1234-4234-8234-123456789012"),
        ("a", "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
        ("f", "ffffffff-ffff-4fff-8fff-ffffffffffff"),
    ],
)
def test_runtime_generated_event_id_ignores_incidental_short_secret_collision(
    monkeypatch: pytest.MonkeyPatch,
    secret: str,
    generated_id: str,
) -> None:
    monkeypatch.setattr(events_module, "uuid4", lambda: UUID(generated_id))
    event = Event(type=EventType.MODEL_STARTED, session_id="session", payload={"step": 1})

    prepared = prepare_new_runtime_event(event, redactor=SecretRedactor(secret))

    assert prepared.id == generated_id


def test_runtime_generated_event_id_tolerates_multiple_simultaneous_short_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_id = "afffffff-ffff-4fff-8fff-ffffffffffff"
    monkeypatch.setattr(events_module, "uuid4", lambda: UUID(generated_id))

    prepared = prepare_new_runtime_event(
        Event(type=EventType.MODEL_STARTED, session_id="session", payload={"step": 1}),
        redactor=SecretRedactor(["-", "a", "f"]),
    )

    assert prepared.id == generated_id


def test_explicit_runtime_event_and_payload_authority_require_positive_provenance() -> None:
    event_id = "runtime-generated-id"
    payload_id = "runtime-generated-linkage"
    plain = Event(
        id=event_id,
        type=EventType.INTERACTION_STARTED,
        session_id="session",
        interaction_id="interaction",
        payload={"status": "active", "start_event_id": payload_id},
    )

    with pytest.raises(ValueError, match=r"event\.event_id"):
        prepare_new_runtime_event(plain, redactor=SecretRedactor("generated-id"))

    runtime_id_only = event_with_runtime_generated_id(plain)
    with pytest.raises(ValueError, match=r"event\.payload\.start_event_id"):
        prepare_new_runtime_event(
            runtime_id_only,
            redactor=SecretRedactor("generated-linkage"),
        )

    prepared = prepare_new_runtime_event(
        event_with_runtime_payload_authority(runtime_id_only, "start_event_id"),
        redactor=SecretRedactor(["generated-id", "generated-linkage"]),
    )
    assert prepared.id == event_id
    assert prepared.payload["start_event_id"] == payload_id

    public = project_runtime_event(
        prepared,
        sequence=9,
        redactor=SecretRedactor(),
    )
    assert event_id not in repr(public.__pydantic_private__)
    assert payload_id not in repr(public.__pydantic_private__)
    assert not event_id_is_runtime_generated(public)
    assert not event_payload_authority_is_runtime_generated(
        public,
        field_name="start_event_id",
        value=payload_id,
    )


def test_runtime_tool_idempotency_authority_is_recomputed_not_shape_trusted() -> None:
    payload = {
        "model_step_id": "step",
        "model_attempt_id": "attempt",
        "tool_round_id": "round",
        "tool_call_id": "call",
    }
    valid_key = tool_idempotency_key(
        session_id="session",
        tool_round_id="round",
        tool_call_id="call",
    )
    prepared = prepare_new_runtime_event(
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="session",
            payload={**payload, "idempotency_key": valid_key},
        ),
        # The generated key contains hyphens, but its exact derivation is positive
        # runtime evidence rather than a shape-only exemption.
        redactor=SecretRedactor("-"),
    )
    assert prepared.payload["idempotency_key"] == valid_key

    forged = f"cayu-tool:v1:{'a' * 64}"
    with pytest.raises(ValueError, match="runtime-owned tool execution identity"):
        prepare_new_runtime_event(
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="session",
                payload={**payload, "idempotency_key": forged},
            ),
            redactor=SecretRedactor(forged),
        )


def test_pre_execution_projection_removes_every_effective_argument_copy() -> None:
    identity = {
        "model_step_id": "step",
        "model_attempt_id": "attempt",
        "tool_round_id": "round",
        "tool_call_id": "call",
    }
    prepared_start = prepare_new_runtime_event(
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="session",
            payload={
                **identity,
                "arguments": {"private": "original"},
                "effective_arguments": {"private": "modified"},
                "idempotency_key": tool_idempotency_key(
                    session_id="session",
                    tool_round_id="round",
                    tool_call_id="call",
                ),
            },
        ),
        redactor=SecretRedactor(),
    )
    prepared_pause = prepare_new_runtime_event(
        Event(
            type=EventType.SESSION_AWAITING_USER_INPUT,
            session_id="session",
            payload={
                "tool_calls": [
                    {
                        **identity,
                        "tool_name": "safe",
                        "arguments": {"private": "original"},
                        "effective_arguments": {"private": "modified"},
                    }
                ]
            },
        ),
        redactor=SecretRedactor(),
    )

    assert prepared_start.payload["arguments_state"] == "quarantined"
    assert "arguments" not in prepared_start.payload
    assert "effective_arguments" not in prepared_start.payload
    paused_call = prepared_pause.payload["tool_calls"][0]
    assert paused_call["arguments_state"] == "quarantined"
    assert "arguments" not in paused_call
    assert "effective_arguments" not in paused_call


@pytest.mark.parametrize(
    "payload",
    [
        {"arguments_state": "finalized"},
        {"arguments_state": "unavailable", "arguments": {}},
        {"arguments_state": "unavailable", "effective_arguments": {}},
        {"arguments_state": "future", "arguments": {}},
    ],
)
def test_new_terminal_events_reject_contradictory_argument_state(payload: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        prepare_new_runtime_event(
            Event(
                type=EventType.TOOL_CALL_FAILED,
                session_id="session",
                payload={
                    "tool_call_id": "call",
                    "tool_name": "safe",
                    **payload,
                },
            ),
            redactor=SecretRedactor(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"arguments_state": "finalized"},
        {"arguments_state": "unavailable", "arguments": {"private": "value"}},
        {"arguments_state": "unavailable", "effective_arguments": {"private": "value"}},
        {"arguments_state": "future", "arguments": {"private": "value"}},
    ],
)
def test_legacy_terminal_argument_conflicts_project_fail_closed(
    payload: dict[str, Any],
) -> None:
    public = project_runtime_event(
        Event(
            type=EventType.TOOL_CALL_FAILED,
            session_id="session",
            payload={
                "tool_call_id": "call",
                "tool_name": "safe",
                **payload,
            },
        ),
        sequence=1,
        redactor=SecretRedactor(),
    )

    assert public.payload["arguments_state"] == "unavailable"
    assert "arguments" not in public.payload
    assert "effective_arguments" not in public.payload


def test_proxy_authority_is_non_actionable_and_verified_idempotency_survives_reprepare() -> None:
    payload = {
        "approval_id": "approval",
        "input_id": "input",
        "model_step_id": "step",
        "model_attempt_id": "attempt",
        "tool_round_id": "round",
        "tool_call_id": "call",
    }
    idempotency_key = tool_idempotency_key(
        session_id="session",
        tool_round_id="round",
        approval_id="approval",
        pause_id="input",
        tool_call_id="call",
    )
    event = Event(
        type=EventType.CREDENTIAL_PROXY_CHECKED,
        session_id="session",
        payload={**payload, "idempotency_key": idempotency_key},
    )
    redactor = SecretRedactor("cayu-tool:v1")

    prepared = prepare_new_runtime_event(event, redactor=redactor)
    reprepared = prepare_new_runtime_event(prepared, redactor=redactor)

    assert prepared.payload["idempotency_key"] == idempotency_key
    assert reprepared.payload["idempotency_key"] == idempotency_key
    public = project_runtime_event(prepared, sequence=8, redactor=redactor)
    for field_name in ("approval_id", "input_id", "tool_call_id", "tool_round_id"):
        assert public.payload[field_name] == PRIVATE_EVENT_AUTHORITY
        assert private_event_linkage_value(prepared, field_name=field_name) is None
    assert public.payload["idempotency_key"] == PRIVATE_EVENT_AUTHORITY


def test_structured_output_round_events_attest_their_deterministic_ids() -> None:
    session = Session(
        id="run",
        agent_name="bot",
        provider_name="p",
        model="m",
    )
    registered_agent = cast(
        "Any",
        SimpleNamespace(spec=SimpleNamespace(name="bot")),
    )
    spec = StructuredOutputSpec(
        name="output",
        json_schema={"type": "object"},
    )
    identity = ToolRoundIdentity(
        model_step_id=f"mstep_{'1' * 32}",
        model_attempt_id=f"matt_{'2' * 32}",
        tool_round_id=f"tround_{'3' * 32}",
    )
    events = [
        _structured_output_validating_event(
            session=session,
            registered_agent=registered_agent,
            environment_name=None,
            spec=spec,
            step=1,
            attempt=1,
            tool_round_identity=identity,
        ),
        _structured_output_event(
            event_type=EventType.STRUCTURED_OUTPUT_VALIDATED,
            session=session,
            registered_agent=registered_agent,
            environment_name=None,
            spec=spec,
            validation=StructuredOutputValidation(valid=True, output={"ok": True}),
            step=1,
            attempt=1,
            tool_round_identity=identity,
        ),
    ]

    for event in events:
        assert event.id.startswith("structured-output:v1:")
        assert event_id_is_runtime_generated(event)
        prepared = prepare_new_runtime_event(event, redactor=SecretRedactor("s"))
        assert prepared.id == event.id


def test_validated_budget_settlement_reattests_its_deterministic_event_id() -> None:
    reservation_id = "reservation"
    settlement_id = budget_settlement_id(reservation_id)
    settled_at = datetime(2026, 7, 31, tzinfo=UTC)
    reconciliation = BudgetReconciliation(
        reservation_id=reservation_id,
        settlement_id=settlement_id,
        settlement_kind="completed",
        budget_limit_id="blim_" + "1" * 64,
        model_step_id="mstep_" + "2" * 32,
        model_attempt_id="matt_" + "3" * 32,
        status="reconciled",
        reserved_amount=Decimal("1"),
        actual_amount=Decimal("0.5"),
        released_amount=Decimal("0.5"),
        settled_at=settled_at,
    )
    event = Event(
        id=budget_settlement_event_id(settlement_id),
        type=EventType.BUDGET_RECONCILED,
        timestamp=settled_at,
        session_id="run",
        agent_name="bot",
        payload=budget_reconciliation_payload(reconciliation),
    )
    record = BudgetSettlementRecord(
        settlement_id=settlement_id,
        reservation_id=reservation_id,
        settlement_kind="completed",
        session_id="run",
        agent_name="bot",
        reconciliation=reconciliation,
        event=event,
    )

    validated = _validate_ledger_settlement_record(record)

    assert event_id_is_runtime_generated(validated.event)
    prepared = prepare_new_runtime_event(
        validated.event,
        redactor=SecretRedactor("budget-settlement:"),
    )
    assert prepared.id == event.id


@pytest.mark.parametrize(
    ("event_type", "payload", "public_id_field"),
    [
        (
            EventType.SERVER_MUTATION_ACCEPTED,
            {"accepted_event_sequence": 7},
            "accepted_event_id",
        ),
        (
            EventType.RUNTIME_SINK_FAILED,
            {"event_sequence": 7},
            "event_id",
        ),
    ],
)
def test_sequence_derived_public_ids_do_not_become_durable_short_secret_authority(
    event_type: EventType,
    payload: dict[str, int],
    public_id_field: str,
) -> None:
    prepared = prepare_new_runtime_event(
        Event(
            type=event_type,
            session_id="sess",
            payload=payload,
        ),
        redactor=SecretRedactor("a"),
    )
    assert public_id_field not in prepared.payload

    public = project_runtime_event(
        prepared,
        sequence=8,
        redactor=SecretRedactor("a"),
    )
    assert public.payload[public_id_field] == public_event_id(7)


@pytest.mark.parametrize(
    ("event_type", "field_name"),
    [
        (EventType.SERVER_MUTATION_ACCEPTED, "accepted_event_sequence"),
        (EventType.RUNTIME_SINK_FAILED, "event_sequence"),
        (EventType.INTERACTION_COMPLETED, "start_event_sequence"),
    ],
)
@pytest.mark.parametrize("invalid", [True, 0, -1, "1"])
def test_public_alias_sequence_evidence_requires_a_positive_integer(
    event_type: EventType,
    field_name: str,
    invalid: object,
) -> None:
    event = Event(
        type=event_type,
        session_id="sess",
        interaction_id=("interaction" if event_type == EventType.INTERACTION_COMPLETED else None),
        payload={
            field_name: invalid,
            **({"status": "completed"} if event_type == EventType.INTERACTION_COMPLETED else {}),
        },
    )

    with pytest.raises(TypeError, match=field_name):
        prepare_new_runtime_event(event, redactor=SecretRedactor())

    public = project_runtime_event(
        event,
        sequence=8,
        redactor=SecretRedactor(),
    )
    assert field_name not in public.payload
    derived_id = (
        "start_event_id"
        if field_name == "start_event_sequence"
        else "accepted_event_id"
        if field_name == "accepted_event_sequence"
        else "event_id"
    )
    assert derived_id not in public.payload


def test_explicit_secret_bearing_event_id_fails_closed() -> None:
    with pytest.raises(ValueError, match=r"event\.event_id"):
        prepare_new_runtime_event(
            Event(
                id="caller-secret-id",
                type=EventType.MODEL_STARTED,
                session_id="session",
                payload={"step": 1},
            ),
            redactor=SecretRedactor("secret-id"),
        )


@pytest.mark.parametrize("runtime_attested", [False, True])
def test_new_event_ids_cannot_enter_the_public_alias_namespace(
    runtime_attested: bool,
) -> None:
    event = Event(
        id=public_event_id(7),
        type=EventType.SESSION_STARTED,
        session_id="session",
    )
    if runtime_attested:
        event = event_with_runtime_generated_id(event)

    with pytest.raises(ValueError, match="reserved public alias namespace"):
        prepare_new_runtime_event(
            event,
            redactor=SecretRedactor(),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "approval_id",
        "idempotency_key",
        "input_id",
        "model_attempt_id",
        "model_step_id",
        "task_id",
        "tool_call_id",
        "tool_round_id",
    ],
)
def test_every_tool_linkage_authority_is_strict_on_write_and_safe_for_legacy(
    field_name: str,
) -> None:
    secret = f"{field_name}-authority-secret"
    event = Event(
        type=EventType.TOOL_CALL_STARTED,
        session_id="session",
        payload={field_name: secret},
    )

    with pytest.raises(ValueError, match=rf"event\.payload\.{field_name}"):
        prepare_new_runtime_event(event, redactor=SecretRedactor(secret))

    public = project_runtime_event(
        event,
        sequence=3,
        redactor=SecretRedactor(secret),
    )
    expected = (
        public_event_linkage_id(3, field_name)
        if field_name in {"approval_id", "input_id", "tool_call_id", "tool_round_id"}
        else PRIVATE_EVENT_AUTHORITY
    )
    assert public.payload[field_name] == expected
    assert secret not in repr(public.model_dump(mode="json"))


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.TOOL_CALL_APPROVED,
        EventType.TOOL_CALL_APPROVAL_EXPIRED,
    ],
)
def test_approval_resolution_events_keep_all_execution_authority_private(
    event_type: EventType,
) -> None:
    event = Event(
        type=event_type,
        session_id="session",
        payload={
            "approval_id": "private-approval",
            "model_attempt_id": "private-model-attempt",
            "model_step_id": "private-model-step",
            "tool_call_id": "private-tool-call",
            "tool_round_id": "private-tool-round",
        },
    )

    public = project_runtime_event(
        event,
        sequence=11,
        redactor=SecretRedactor(),
    )

    assert public.payload == {
        "approval_id": public_event_linkage_id(11, "approval_id"),
        "model_attempt_id": PRIVATE_EVENT_AUTHORITY,
        "model_step_id": PRIVATE_EVENT_AUTHORITY,
        "tool_call_id": public_event_linkage_id(11, "tool_call_id"),
        "tool_round_id": public_event_linkage_id(11, "tool_round_id"),
    }


def test_manual_recovery_top_level_linkage_is_actionable_without_exposing_authority() -> None:
    event = Event(
        type=EventType.SESSION_INTERRUPTED,
        session_id="session",
        payload={
            "interruption_type": "runtime_interrupted",
            "manual_recovery_required": True,
            "model_attempt_id": "private-model-attempt",
            "model_step_id": "private-model-step",
            "tool_call_id": "private-tool-call",
            "tool_round_id": "private-tool-round",
        },
    )

    public = project_runtime_event(
        event,
        sequence=12,
        redactor=SecretRedactor(),
    )

    assert public.payload["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
    assert public.payload["model_step_id"] == PRIVATE_EVENT_AUTHORITY
    assert public.payload["tool_call_id"] == public_event_linkage_id(12, "tool_call_id")
    assert public.payload["tool_round_id"] == public_event_linkage_id(12, "tool_round_id")
    assert private_event_linkage_value(event, field_name="tool_call_id") == "private-tool-call"
    assert private_event_linkage_value(event, field_name="tool_round_id") == "private-tool-round"


@pytest.mark.parametrize("secret", ["source", "subject", "tenant"])
def test_resolution_actor_schema_keys_survive_short_secret_collisions(secret: str) -> None:
    event = Event(
        type=EventType.TOOL_CALL_APPROVED,
        session_id="session",
        payload={
            "approval_id": "approval",
            "tool_call_id": "call",
            "tool_round_id": "round",
            "resolved_by": {
                "source": "request",
                "subject": "operator",
                "tenant": "tenant-a",
            },
        },
    )

    redactor = SecretRedactor(secret)
    prepared = prepare_new_runtime_event(event, redactor=redactor)

    assert prepared.payload["resolved_by"] == {
        key: redactor.redact_text(value) for key, value in event.payload["resolved_by"].items()
    }


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.SESSION_MESSAGE_QUEUED,
        EventType.SESSION_MESSAGE_DELIVERED,
        EventType.SESSION_CHECKPOINTED,
        EventType.MODEL_COMPLETED,
        EventType.BUDGET_CHECKED,
        EventType.BUDGET_LIMIT_REACHED,
        EventType.BUDGET_RESERVATION_FAILED,
        EventType.BUDGET_RESERVED,
        EventType.BUDGET_RECONCILED,
        EventType.BUDGET_RESERVATION_RELEASED,
        EventType.CONTEXT_COMPACTION_STARTED,
        EventType.CONTEXT_COMPACTION_COMPLETED,
        EventType.CONTEXT_COMPACTION_FAILED,
    ],
)
@pytest.mark.parametrize("secret", ["source", "subject", "tenant"])
def test_every_actor_bearing_policy_owns_typed_actor_keys(
    event_type: EventType,
    secret: str,
) -> None:
    actor = {
        "source": "request",
        "subject": "operator",
        "tenant": "tenant-a",
    }
    payload: dict[str, Any] = {"actor": actor}
    if event_type in {
        EventType.BUDGET_RECONCILED,
        EventType.BUDGET_RESERVATION_RELEASED,
    }:
        released = event_type == EventType.BUDGET_RESERVATION_RELEASED
        reconciliation = BudgetReconciliation(
            reservation_id="reservation",
            settlement_id=budget_settlement_id("reservation"),
            settlement_kind="released" if released else "completed",
            budget_limit_id="blim_" + "1" * 64,
            model_step_id="mstep_" + "2" * 32,
            model_attempt_id="matt_" + "3" * 32,
            status="released" if released else "reconciled",
            reserved_amount=Decimal("1"),
            actual_amount=None if released else Decimal("0.5"),
            released_amount=Decimal("1") if released else Decimal("0.5"),
            settled_at=datetime(2026, 7, 31, tzinfo=UTC),
        )
        payload = {**budget_reconciliation_payload(reconciliation), "actor": actor}
    event = Event(
        type=event_type,
        session_id="session",
        payload=payload,
    )

    redactor = SecretRedactor(secret)
    prepared = prepare_new_runtime_event(event, redactor=redactor)
    public = project_runtime_event(prepared, sequence=3, redactor=redactor)

    expected = {key: redactor.redact_text(value) for key, value in actor.items()}
    assert prepared.payload["actor"] == expected
    assert public.payload["actor"] == expected


def test_typed_pause_payload_keys_survive_exact_short_secret_collisions() -> None:
    pending_call = PendingToolCallApproval(
        tool_call_id="call-private",
        tool_name="reader",
        arguments={"path": "README.md"},
    )
    approval = PendingToolApproval(
        approval_id="approval-private",
        tool_round_id=f"tround_{'1' * 32}",
        model_step_id=f"mstep_{'2' * 32}",
        model_attempt_id=f"matt_{'3' * 32}",
        tool_call_id="call-private",
        tool_name="reader",
        arguments={"path": "README.md"},
        agent_name="assistant",
        tool_calls=[pending_call],
    )
    user_input = PendingUserInput(
        input_id="input-private",
        tool_round_id=f"tround_{'1' * 32}",
        model_step_id=f"mstep_{'2' * 32}",
        model_attempt_id=f"matt_{'3' * 32}",
        tool_call_id="call-private",
        tool_name="reader",
        question="Continue?",
        arguments={"path": "README.md"},
        agent_name="assistant",
        tool_calls=[pending_call],
    )
    approval_event = Event(
        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
        session_id="session",
        payload={
            "approval_id": approval.approval_id,
            "tool_call_id": approval.tool_call_id,
            "tool_round_id": approval.tool_round_id,
            "model_step_id": approval.model_step_id,
            "model_attempt_id": approval.model_attempt_id,
            "approval": approval.model_dump(mode="json"),
        },
    )
    input_event = Event(
        type=EventType.SESSION_INTERRUPTED,
        session_id="session",
        payload={
            "interruption_type": "user_input_required",
            "input_id": user_input.input_id,
            "tool_call_id": user_input.tool_call_id,
            "tool_round_id": user_input.tool_round_id,
            "model_step_id": user_input.model_step_id,
            "model_attempt_id": user_input.model_attempt_id,
            "user_input": user_input.model_dump(mode="json"),
        },
    )

    for container_name, event, field_names in (
        (
            "approval",
            approval_event,
            event_projection_module._PENDING_APPROVAL_FIELD_NAMES,
        ),
        (
            "user_input",
            input_event,
            event_projection_module._PENDING_USER_INPUT_FIELD_NAMES,
        ),
    ):
        private_field_names = {
            "arguments",
            "assistant_publication",
            "assistant_message_state",
            "quarantined_assistant_message",
            "secret_resolution_scope",
        }
        for field_name in field_names:
            prepared = prepare_new_runtime_event(
                event,
                redactor=SecretRedactor(field_name),
            )
            if field_name in private_field_names:
                assert field_name not in prepared.payload[container_name]
                assert prepared.payload[container_name]["arguments_state"] == "quarantined"
            else:
                assert field_name in prepared.payload[container_name]
        for field_name in event_projection_module._PENDING_TOOL_CALL_FIELD_NAMES:
            prepared = prepare_new_runtime_event(
                event,
                redactor=SecretRedactor(field_name),
            )
            if field_name == "arguments":
                assert field_name not in prepared.payload[container_name]["tool_calls"][0]
                assert (
                    prepared.payload[container_name]["tool_calls"][0]["arguments_state"]
                    == "quarantined"
                )
            else:
                assert field_name in prepared.payload[container_name]["tool_calls"][0]

        marker_collision = prepare_new_runtime_event(
            event,
            redactor=SecretRedactor("quarantined"),
        )
        assert marker_collision.payload[container_name]["arguments_state"] == "quarantined"
        assert (
            marker_collision.payload[container_name]["tool_calls"][0]["arguments_state"]
            == "quarantined"
        )


@pytest.mark.parametrize(
    "value",
    [
        "cayu_event_01",
        f"cayu_event_{MAX_DURABLE_JSON_INTEGER + 1}",
        f"cayu_event_{'9' * 5000}",
    ],
)
def test_public_event_sequence_rejects_out_of_range_aliases_without_raising(
    value: str,
) -> None:
    assert public_event_sequence(value) is None


def test_validated_fixed_controls_survive_value_collisions() -> None:
    cases = [
        (
            Event(
                type=EventType.INTERACTION_COMPLETED,
                session_id="session",
                interaction_id="interaction",
                payload={"status": "completed"},
            ),
            "completed",
            lambda event: event.payload["status"],
        ),
        (
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="session",
                payload={"effect": ToolEffect.IDEMPOTENT.value},
            ),
            ToolEffect.IDEMPOTENT.value,
            lambda event: event.payload["effect"],
        ),
        (
            Event(
                type=EventType.STRUCTURED_OUTPUT_VALIDATING,
                session_id="session",
                payload={"strategy": "native"},
            ),
            "native",
            lambda event: event.payload["strategy"],
        ),
        (
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="session",
                payload={
                    "step_classification": {
                        "type": "final",
                        "reason": "safe reason",
                    }
                },
            ),
            "final",
            lambda event: event.payload["step_classification"]["type"],
        ),
        (
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="session",
                payload={
                    "completion": {
                        "finish_reason": "stop",
                        "raw_finish_reason": "stop",
                        "status": "safe",
                        "end_turn": True,
                    }
                },
            ),
            "stop",
            lambda event: event.payload["completion"]["finish_reason"],
        ),
        (
            Event(
                type=EventType.SESSION_INTERRUPTED,
                session_id="session",
                payload={"interruption_type": "operator_requested"},
            ),
            "operator_requested",
            lambda event: event.payload["interruption_type"],
        ),
        (
            Event(
                type=EventType.TOOL_CALL_BLOCKED,
                session_id="session",
                payload={
                    "denied_by": "tool_policy",
                    "decision": "deny",
                    "result": {
                        "content": "safe",
                        "structured": {
                            "decision": "deny",
                            "reason": "safe",
                        },
                        "artifacts": [],
                        "is_error": True,
                    },
                },
            ),
            "deny",
            lambda event: event.payload["result"]["structured"]["decision"],
        ),
    ]

    for event, secret, select in cases:
        prepared = prepare_new_runtime_event(
            event,
            redactor=SecretRedactor(secret),
        )
        assert select(prepared) == secret


def test_fixed_control_registry_covers_mcp_binding_workflow_and_budget_protocols() -> None:
    reconciliation = BudgetReconciliation(
        reservation_id="reservation",
        settlement_id=budget_settlement_id("reservation"),
        settlement_kind="completed",
        budget_limit_id="blim_" + "1" * 64,
        model_step_id="mstep_" + "2" * 32,
        model_attempt_id="matt_" + "3" * 32,
        status="reconciled",
        reserved_amount=Decimal("1"),
        actual_amount=Decimal("0.5"),
        released_amount=Decimal("0.5"),
        settled_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    settlement = budget_reconciliation_payload(reconciliation)
    cases = (
        (
            Event(
                type=EventType.MCP_MANIFEST_CHECKED,
                session_id="session",
                payload={
                    "status": "changed",
                    "outcome": "accepted",
                    "policy": {
                        "action": "alert",
                        "status": "changed",
                        "matched_changes": ["tools_added"],
                        "reason": "safe",
                    },
                },
            ),
            ["changed", "accepted", "alert"],
        ),
        (
            Event(
                type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
                session_id="session",
                payload={
                    "outcome": "completed",
                    "terminal_outcome": "completed",
                    "factory_allocation_action": "preserve",
                },
            ),
            ["completed", "preserve"],
        ),
        (
            Event(
                type=EventType.WORKFLOW_STEP_COMPLETED,
                session_id="session",
                payload={"kind": "gated_loop", "outcome": "pass", "passed": True},
            ),
            ["gated_loop", "pass"],
        ),
        (
            Event(
                type=EventType.STRUCTURED_OUTPUT_VALIDATED,
                session_id="session",
                payload={"strategy": "tool"},
            ),
            ["tool"],
        ),
        (
            Event(
                type=EventType.CONTEXT_COMPACTION_COMPLETED,
                session_id="session",
                payload={
                    "checkpoint": "context_compaction",
                    "coverage_mode": "partial_prefix",
                    "chunk_mode": "message_prefix",
                    "bounded_input": True,
                    "compaction_failed": False,
                },
            ),
            ["context_compaction", "partial_prefix", "message_prefix"],
        ),
        (
            Event(
                type=EventType.SESSION_CHECKPOINTED,
                session_id="session",
                payload={"checkpoint": "pending_tool_approval"},
            ),
            ["pending_tool_approval"],
        ),
        (
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="session",
                payload={"budget_settlements": [settlement]},
            ),
            ["reservation_id", "completed", "reconciled"],
        ),
    )

    for event, secrets in cases:
        prepared = prepare_new_runtime_event(event, redactor=SecretRedactor(secrets))
        assert prepared.payload == event.payload

    public_settlement = project_runtime_event(
        cases[-1][0],
        sequence=7,
        redactor=SecretRedactor(),
    ).payload["budget_settlements"][0]
    assert public_settlement["reservation_id"] == PRIVATE_EVENT_AUTHORITY
    assert public_settlement["settlement_id"] == PRIVATE_EVENT_AUTHORITY
    assert public_settlement["budget_limit_id"] == PRIVATE_EVENT_AUTHORITY
    assert public_settlement["model_step_id"] == PRIVATE_EVENT_AUTHORITY
    assert public_settlement["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
    assert public_settlement["settlement_kind"] == "completed"
    assert public_settlement["status"] == "reconciled"


def test_nested_budget_settlement_authority_rejects_secret_before_redaction() -> None:
    reconciliation = BudgetReconciliation(
        reservation_id="reservation-secret",
        settlement_id=budget_settlement_id("reservation-secret"),
        settlement_kind="completed",
        budget_limit_id="blim_" + "1" * 64,
        model_step_id="mstep_" + "2" * 32,
        model_attempt_id="matt_" + "3" * 32,
        status="reconciled",
        reserved_amount=Decimal("1"),
        actual_amount=Decimal("0.5"),
        released_amount=Decimal("0.5"),
        settled_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    event = Event(
        type=EventType.MODEL_COMPLETED,
        session_id="session",
        payload={"budget_settlements": [budget_reconciliation_payload(reconciliation)]},
    )

    with pytest.raises(ValueError, match="budget_settlements.*reservation_id"):
        prepare_new_runtime_event(
            event,
            redactor=SecretRedactor("reservation-secret"),
        )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda settlement: settlement.update({"unexpected": "value"}),
        lambda settlement: settlement["pricing"].update({"unexpected": "value"}),
        lambda settlement: settlement.update({"settled_at_unix_us": True}),
    ],
)
def test_model_budget_settlement_requires_the_exact_nested_accounting_schema(
    mutate: Any,
) -> None:
    reconciliation = BudgetReconciliation(
        reservation_id="reservation",
        settlement_id=budget_settlement_id("reservation"),
        settlement_kind="completed",
        budget_limit_id="blim_" + "1" * 64,
        model_step_id="mstep_" + "2" * 32,
        model_attempt_id="matt_" + "3" * 32,
        status="reconciled",
        reserved_amount=Decimal("1"),
        actual_amount=Decimal("0.5"),
        released_amount=Decimal("0.5"),
        settled_at=datetime(2026, 7, 31, tzinfo=UTC),
    )
    settlement = budget_reconciliation_payload(reconciliation)
    settlement["pricing"] = {
        "provider_name": "provider",
        "model": "model",
        "match": "exact",
        "provenance": {
            "source": "test",
            "url": "https://example.test/pricing",
            "as_of": "2026-07-31",
        },
        "effective_from": None,
        "effective_through": None,
        "tier_max_input_tokens": None,
    }
    mutate(settlement)

    with pytest.raises(ValueError):
        prepare_new_runtime_event(
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="session",
                payload={"budget_settlements": [settlement]},
            ),
            redactor=SecretRedactor(),
        )


@pytest.mark.parametrize(
    "event",
    [
        Event(
            type=EventType.MCP_MANIFEST_CHECKED,
            session_id="session",
            payload={"status": "future", "outcome": "accepted"},
        ),
        Event(
            type=EventType.ENVIRONMENT_BINDING_FINALIZE_COMPLETED,
            session_id="session",
            payload={"outcome": "future"},
        ),
        Event(
            type=EventType.WORKFLOW_STEP_COMPLETED,
            session_id="session",
            payload={"kind": "gated_loop", "passed": 1},
        ),
    ],
)
def test_declared_fixed_protocol_controls_reject_malformed_new_values(event: Event) -> None:
    with pytest.raises(ValueError):
        prepare_new_runtime_event(event, redactor=SecretRedactor())


def test_custom_session_checkpoint_names_remain_untrusted_extensible_values() -> None:
    event = Event(
        type=EventType.SESSION_CHECKPOINTED,
        session_id="session",
        payload={"checkpoint": "custom-policy", "calls": 1},
    )

    prepared = prepare_new_runtime_event(event, redactor=SecretRedactor())
    assert prepared.payload == {"checkpoint": "custom-policy", "calls": 1}
    assert project_runtime_event(
        prepared,
        sequence=4,
        redactor=SecretRedactor("custom-policy"),
    ).payload == {"checkpoint": REDACTED_SECRET, "calls": 1}


def test_custom_session_checkpoint_name_requires_a_string() -> None:
    event = Event(
        type=EventType.SESSION_CHECKPOINTED,
        session_id="session",
        payload={"checkpoint": 1},
    )

    with pytest.raises(TypeError, match="event.payload.checkpoint has an invalid type"):
        prepare_new_runtime_event(event, redactor=SecretRedactor())


@pytest.mark.parametrize(
    "event_type",
    [
        EventType.SESSION_COMPLETED,
        EventType.SESSION_FAILED,
        EventType.SESSION_INTERRUPTED,
    ],
)
@pytest.mark.parametrize(
    "field_name",
    [
        "binding_finalize_error",
        "binding_finalize_publication_error",
        "environment_factory_release",
    ],
)
def test_every_terminal_event_owns_every_finalization_diagnostic(
    event_type: EventType,
    field_name: str,
) -> None:
    diagnostic = {"code": "ok"}
    event = Event(
        type=event_type,
        session_id="run",
        payload={field_name: diagnostic},
    )

    prepared = prepare_new_runtime_event(event, redactor=SecretRedactor("a"))
    public = project_runtime_event(prepared, sequence=3, redactor=SecretRedactor("a"))

    assert prepared.payload[field_name] == diagnostic
    assert public.payload[field_name] == diagnostic


@pytest.mark.parametrize(
    "event_type",
    [EventType.SESSION_STARTED, EventType.SESSION_RESUMED],
)
def test_session_start_and_resume_own_complete_w3c_trace_context(
    event_type: EventType,
) -> None:
    payload = {
        "traceparent": "00-11111111111111111111111111111111-2222222222222222-01",
        "tracestate": "rojo=1",
    }

    prepared = prepare_new_runtime_event(
        Event(type=event_type, session_id="run", payload=payload),
        redactor=SecretRedactor("trace"),
    )

    assert prepared.payload == payload


def test_validated_terminal_controls_survive_exact_nested_schema_collisions() -> None:
    terminal_controls = {
        "terminal_outcome": "tool_execution_error",
        "tool_effect": ToolEffect.EXTERNAL.value,
        "outcome_unknown": True,
        "manual_reconciliation_required": True,
    }
    event = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session",
        payload={
            **terminal_controls,
            "result": {
                "content": "safe",
                "structured": dict(terminal_controls),
                "artifacts": [],
                "is_error": True,
            },
        },
    )

    prepared = prepare_new_runtime_event(
        event,
        redactor=SecretRedactor(["terminal_outcome", "tool_effect", ToolEffect.EXTERNAL.value]),
    )

    assert {key: prepared.payload[key] for key in terminal_controls} == terminal_controls
    assert prepared.payload["result"]["structured"] == terminal_controls


@pytest.mark.parametrize(
    "event",
    [
        Event(
            type=EventType.INTERACTION_COMPLETED,
            session_id="session",
            interaction_id="interaction",
            payload={"status": "future"},
        ),
        Event(
            type=EventType.TOOL_CALL_STARTED,
            session_id="session",
            payload={"effect": "future"},
        ),
        Event(
            type=EventType.STRUCTURED_OUTPUT_VALIDATING,
            session_id="session",
            payload={"strategy": "future"},
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session",
            payload={"step_classification": {"type": "future"}},
        ),
        Event(
            type=EventType.MODEL_COMPLETED,
            session_id="session",
            payload={"completion": {"finish_reason": "future"}},
        ),
        Event(
            type=EventType.SESSION_INTERRUPTED,
            session_id="session",
            payload={"interruption_type": "future"},
        ),
        Event(
            type=EventType.TOOL_CALL_BLOCKED,
            session_id="session",
            payload={
                "denied_by": "tool_policy",
                "decision": "allow",
            },
        ),
        Event(
            type=EventType.TOOL_CALL_BLOCKED,
            session_id="session",
            payload={
                "denied_by": "command_policy",
                "decision": "deny",
                "result": {
                    "content": "safe",
                    "structured": {
                        "decision": "require_command_approval",
                    },
                    "artifacts": [],
                    "is_error": True,
                },
            },
        ),
    ],
)
def test_malformed_fixed_controls_fail_before_new_persistence(event: Event) -> None:
    with pytest.raises(ValueError):
        prepare_new_runtime_event(event, redactor=SecretRedactor())


def test_interaction_start_linkage_uses_its_durable_sequence_alias() -> None:
    private_id = "private-interaction-start-id"
    event = Event(
        id=private_id,
        type=EventType.INTERACTION_STARTED,
        session_id="session",
        interaction_id="interaction",
        payload={
            "status": "active",
            "start_event_id": private_id,
        },
    )

    public = project_runtime_event(
        event,
        sequence=11,
        redactor=SecretRedactor(),
    )

    assert public.id == public_event_id(11)
    assert public.payload["start_event_id"] == public_event_id(11)
    assert private_id not in repr(public.model_dump(mode="json"))


def test_tool_linkage_is_strict_on_write_and_safe_on_legacy_projection() -> None:
    secret = "tool-linkage-secret"
    legacy = Event(
        type=EventType.TOOL_CALL_STARTED,
        session_id="session",
        payload={
            "effect": ToolEffect.NONE.value,
            "tool_call_id": secret,
            "tool_name": f"reader-{secret}",
            "arguments": {"token": secret},
        },
    )

    with pytest.raises(ValueError, match=r"event\.payload\.tool_call_id"):
        prepare_new_runtime_event(legacy, redactor=SecretRedactor(secret))

    public = project_runtime_event(
        legacy,
        sequence=7,
        redactor=SecretRedactor(secret),
    )
    assert public.id == public_event_id(7)
    assert public.payload["effect"] == ToolEffect.NONE.value
    assert public.payload["tool_call_id"] == public_event_linkage_id(
        7,
        "tool_call_id",
    )
    assert secret not in repr(public.model_dump(mode="json"))


@pytest.mark.parametrize("invalid", ["", True, ["call-private"]])
def test_malformed_legacy_linkage_never_receives_a_public_alias(invalid: object) -> None:
    legacy = Event(
        type=EventType.TOOL_CALL_STARTED,
        session_id="session",
        payload={"tool_call_id": invalid},
    )

    public = project_runtime_event(
        legacy,
        sequence=7,
        redactor=SecretRedactor(),
    )

    assert public.payload["tool_call_id"] == PRIVATE_EVENT_AUTHORITY


def test_conflicting_nested_approval_linkage_fails_closed_for_legacy() -> None:
    secret = "nested-approval-private-id"
    legacy = Event(
        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
        session_id="session",
        payload={
            "approval_id": "approval-top",
            "tool_call_id": "call-top",
            "tool_round_id": "round-top",
            "approval": {
                "approval_id": secret,
                "tool_call_id": "call-nested",
                "tool_round_id": "round-nested",
                "model_step_id": "step-nested",
                "model_attempt_id": "attempt-nested",
                "tool_calls": [
                    {
                        "tool_call_id": "sibling-call",
                        "tool_name": "safe",
                    }
                ],
            },
        },
    )

    with pytest.raises(ValueError, match=r"payload\.approval\.approval_id"):
        prepare_new_runtime_event(legacy, redactor=SecretRedactor(secret))

    public = project_runtime_event(
        legacy,
        sequence=12,
        redactor=SecretRedactor(secret),
    )
    approval = public.payload["approval"]
    assert approval["approval_id"] == PRIVATE_EVENT_AUTHORITY
    assert approval["tool_call_id"] == PRIVATE_EVENT_AUTHORITY
    assert approval["tool_round_id"] == PRIVATE_EVENT_AUTHORITY
    assert approval["model_step_id"] == PRIVATE_EVENT_AUTHORITY
    assert approval["model_attempt_id"] == PRIVATE_EVENT_AUTHORITY
    assert approval["tool_calls"][0]["tool_call_id"] == PRIVATE_EVENT_AUTHORITY
    assert secret not in repr(public.model_dump(mode="json"))


def test_nested_approval_linkage_alias_resolves_only_unambiguous_private_authority() -> None:
    event = Event(
        type=EventType.SESSION_INTERRUPTED,
        session_id="session",
        payload={
            "interruption_type": "tool_approval_required",
            "tool_round_id": "round-private",
            "approval": {
                "approval_id": "approval-private",
                "tool_call_id": "call-private",
                "tool_round_id": "round-private",
                "tool_calls": [
                    {"tool_call_id": "call-private"},
                    {"tool_call_id": "sibling-private"},
                ],
            },
        },
    )

    assert private_event_linkage_value(event, field_name="approval_id") == "approval-private"
    assert private_event_linkage_value(event, field_name="tool_call_id") == "call-private"
    assert private_event_linkage_value(event, field_name="tool_round_id") == "round-private"

    conflicting = event.model_copy(deep=True)
    conflicting.payload["approval"]["tool_round_id"] = "other-round"
    assert private_event_linkage_value(conflicting, field_name="tool_round_id") is None


def test_nested_authority_key_ownership_survives_exact_short_secret_collision() -> None:
    event = Event(
        type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
        session_id="session",
        payload={
            "approval_id": "approval-top",
            "tool_call_id": "call-top",
            "tool_round_id": "round-top",
            "approval": {
                "approval_id": "approval-top",
                "tool_call_id": "call-top",
                "tool_round_id": "round-top",
                "model_step_id": "step-nested",
                "model_attempt_id": "attempt-nested",
                "tool_calls": [
                    {
                        "tool_call_id": "sibling-call",
                        "tool_name": "safe",
                    }
                ],
            },
        },
    )
    redactor = SecretRedactor("tool_call_id")

    prepared = prepare_new_runtime_event(event, redactor=redactor)
    assert prepared.payload["approval"]["tool_call_id"] == "call-top"
    assert prepared.payload["approval"]["tool_calls"][0]["tool_call_id"] == "sibling-call"

    public = project_runtime_event(event, sequence=12, redactor=redactor)
    approval = public.payload["approval"]
    assert approval["tool_call_id"] == public_event_linkage_id(12, "tool_call_id")
    assert approval["tool_calls"][0]["tool_call_id"] == public_event_linkage_id(
        12,
        "tool_call_id",
    )
    assert REDACTED_SECRET not in approval


def test_nested_tool_result_linkage_never_exposes_private_authority() -> None:
    event = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session",
        payload={
            "tool_call_id": "top-call",
            "result": {
                "content": "safe",
                "structured": {
                    "tool_call_id": "nested-call",
                    "model_step_id": "nested-step",
                    "detail": "observable",
                },
                "artifacts": [],
                "is_error": True,
            },
        },
    )

    public = project_runtime_event(
        event,
        sequence=13,
        redactor=SecretRedactor(),
    )

    structured = public.payload["result"]["structured"]
    assert structured["tool_call_id"] == PRIVATE_EVENT_AUTHORITY
    assert structured["model_step_id"] == PRIVATE_EVENT_AUTHORITY
    assert structured["detail"] == "observable"


def test_malformed_fixed_control_rejects_new_write_and_loses_public_authority() -> None:
    malformed = Event(
        type=EventType.MODEL_STARTED,
        session_id="session",
        payload={"step": True},
    )
    with pytest.raises(TypeError, match=r"payload\.step"):
        prepare_new_runtime_event(malformed, redactor=SecretRedactor())

    public = project_runtime_event(
        malformed,
        sequence=8,
        redactor=SecretRedactor(),
    )
    assert public.id == public_event_id(8)
    assert public.payload == {}


def test_nested_control_addresses_never_become_flat_payload_keys() -> None:
    event = Event(
        type=EventType.TOOL_CALL_BLOCKED,
        session_id="session",
        payload={
            "denied_by": "command_policy",
            "decision": "deny",
            "result": {
                "content": "denied",
                "structured": {
                    "decision": "deny",
                    "error": "command_denied",
                },
                "artifacts": [],
                "is_error": True,
            },
        },
    )

    prepared = prepare_new_runtime_event(event, redactor=SecretRedactor("deny"))
    public = project_runtime_event(event, sequence=1, redactor=SecretRedactor("deny"))

    for observed in (prepared, public):
        assert all("." not in key for key in observed.payload)
        assert observed.payload["result"]["structured"]["decision"] == "deny"
        assert observed.payload["result"]["structured"]["error"] == "command_denied"


@pytest.mark.parametrize(
    ("event", "invalid_path"),
    [
        (
            Event(
                type=EventType.INTERACTION_COMPLETED,
                session_id="session",
                payload={"status": "future"},
            ),
            ("status",),
        ),
        (
            Event(
                type=EventType.TOOL_CALL_STARTED,
                session_id="session",
                payload={"effect": "future"},
            ),
            ("effect",),
        ),
        (
            Event(
                type=EventType.STRUCTURED_OUTPUT_VALIDATING,
                session_id="session",
                payload={"strategy": "future"},
            ),
            ("strategy",),
        ),
        (
            Event(
                type=EventType.MODEL_COMPLETED,
                session_id="session",
                payload={
                    "step_classification": {
                        "type": "future",
                        "reason": "still descriptive",
                    }
                },
            ),
            ("step_classification", "type"),
        ),
    ],
)
def test_malformed_legacy_discriminators_are_not_public_authority(
    event: Event,
    invalid_path: tuple[str, ...],
) -> None:
    public = project_runtime_event(
        event,
        sequence=8,
        redactor=SecretRedactor(),
    )

    value: object = public.payload
    for key in invalid_path[:-1]:
        assert type(value) is dict
        value = value[key]
    assert type(value) is dict
    assert invalid_path[-1] not in value


def test_malformed_legacy_terminal_controls_are_not_public_authority() -> None:
    malformed = Event(
        type=EventType.TOOL_CALL_FAILED,
        session_id="session",
        payload={
            "terminal_outcome": "future",
            "tool_effect": ToolEffect.EXTERNAL.value,
            "outcome_unknown": True,
            "manual_reconciliation_required": True,
            "result": {
                "content": "legacy diagnostic",
                "structured": {
                    "terminal_outcome": "future",
                    "tool_effect": ToolEffect.EXTERNAL.value,
                    "outcome_unknown": True,
                    "manual_reconciliation_required": True,
                    "detail": "still observable",
                },
                "artifacts": [],
                "is_error": True,
            },
        },
    )

    public = project_runtime_event(
        malformed,
        sequence=8,
        redactor=SecretRedactor(),
    )

    assert _TERMINAL_CONTROL_KEYS_FOR_TEST.isdisjoint(public.payload)
    structured = public.payload["result"]["structured"]
    assert _TERMINAL_CONTROL_KEYS_FOR_TEST.isdisjoint(structured)
    assert structured["detail"] == "still observable"


def test_legacy_custom_type_and_colliding_keys_have_safe_lossless_projection() -> None:
    secret = "legacy-secret"
    legacy = Event(
        type="custom.legacy-secret",
        session_id="session",
        payload={
            "legacy-secret": "first",
            f"{secret}-other": "second",
            "value": secret,
        },
    )

    public = project_runtime_event(
        legacy,
        sequence=9,
        redactor=SecretRedactor(secret),
    )

    assert public.type == REDACTED_CUSTOM_EVENT_TYPE
    assert public.id == public_event_id(9)
    assert len(public.payload) == 3
    assert list(public.payload.values()) == ["first", "second", REDACTED_SECRET]
    assert secret not in repr(public.model_dump(mode="json"))


def test_writer_keeps_budget_identity_private_and_projects_sink_and_recovery() -> None:
    async def scenario():
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                session_id="projectionwriter",
                agent_name="assistant",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        budget_store = InMemoryBudgetStore()
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=budget_store,
            event_sinks=[sink],
        )
        emitted = await writer.emit(
            Event(type=EventType.MODEL_COMPLETED, session_id="projectionwriter")
        )
        records = await store.query_events()
        budget_events = await budget_store.load_events_for_budget(
            scope="app",
            key=None,
            window=BudgetWindow.all_time(),
        )
        return emitted, records, budget_events, sink.events

    emitted, records, budget_events, sink_events = asyncio.run(scenario())
    assert len(records) == 1
    assert emitted.id == records[0].event.id == budget_events[0].id
    assert sink_events[0].id == public_event_id(records[0].sequence)
    assert sink_events[0].id != emitted.id
    assert emitted.id not in repr(sink_events[0].__pydantic_private__)
    assert "projectionwriter" not in repr(sink_events[0].__pydantic_private__)


def test_writer_rejects_forged_tool_idempotency_before_any_publication() -> None:
    async def scenario() -> tuple[int, int]:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                session_id="idempotency-authority",
                agent_name="assistant",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        sink = _RecordingSink()
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[sink],
            secret_redactor=SecretRedactor("aaaaaaaa"),
        )
        with pytest.raises(ValueError, match="runtime-owned tool execution identity"):
            await writer.emit(
                Event(
                    type=EventType.TOOL_CALL_STARTED,
                    session_id="idempotency-authority",
                    payload={
                        "model_step_id": "step",
                        "model_attempt_id": "attempt",
                        "tool_round_id": "round",
                        "tool_call_id": "call",
                        "idempotency_key": f"cayu-tool:v1:{'a' * 64}",
                    },
                )
            )
        return len(await store.query_events()), len(sink.events)

    durable_count, sink_count = asyncio.run(scenario())
    assert durable_count == 0
    assert sink_count == 0


def test_legacy_sink_recovery_logs_only_the_public_projection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from cayu.observability import LoggingEventSink

    secret = "legacy-log-event-secret"
    logger = logging.getLogger("cayu.test.event-projection")

    async def scenario() -> None:
        store = InMemorySessionStore()
        await store.create(
            RunRequest(
                session_id="projectionlog",
                agent_name="assistant",
                messages=[],
            ),
            identity=SessionIdentity(provider_name="fake", model="fake"),
        )
        await store.append_event(
            "projectionlog",
            Event(
                id=f"private-{secret}-id",
                type=f"custom.{secret}",
                session_id="projectionlog",
                payload={secret: secret},
            ),
        )
        writer = RuntimeEventWriter(
            session_store=store,
            budget_store=InMemoryBudgetStore(),
            event_sinks=[
                LoggingEventSink(
                    logger=logger,
                    redactor=SecretRedactor(secret),
                )
            ],
            secret_redactor=SecretRedactor(secret),
        )
        await writer.recover_persisted_side_effects()

    with caplog.at_level(logging.INFO, logger=logger.name):
        asyncio.run(scenario())

    assert REDACTED_CUSTOM_EVENT_TYPE in caplog.text
    assert secret not in caplog.text
