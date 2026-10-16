"""Public call correlation is opaque and separate from resolution aliases."""

import pytest

from cayu import Event, EventType


def event(kind, **payload):
    return Event(type=kind, session_id="session", payload=payload)


def tool_event(kind, call_id, round_id, **payload):
    return event(kind, tool_call_id=call_id, tool_round_id=round_id, **payload)


@pytest.mark.parametrize("configured", [True, False])
def test_public_correlation_is_scoped_opaque_and_cannot_be_supplied_by_a_producer(configured):
    from pydantic import SecretStr

    from cayu.runtime._event_projection import project_runtime_event
    from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
    from cayu.vaults import SecretRedactor

    codec = (
        PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={"test": SecretStr("A" * 43)},
            )
        )
        if configured
        else None
    )
    redactor = SecretRedactor(["private-call", "private-round"])

    def project(value, sequence):
        return project_runtime_event(
            value, sequence=sequence, redactor=redactor, public_authority_alias_codec=codec
        )

    started = tool_event(
        EventType.TOOL_CALL_STARTED,
        "private-call",
        "private-round",
        tool_call_correlation_id="forged",
    )
    terminal = tool_event(EventType.TOOL_CALL_COMPLETED, "private-call", "private-round")
    public_start, public_end = project(started, 1), project(terminal, 2)
    key = public_start.payload["tool_call_correlation_id"]
    assert key == public_end.payload["tool_call_correlation_id"]
    expired = project(
        tool_event(EventType.TOOL_CALL_APPROVAL_EXPIRED, "private-call", "private-round"), 6
    )
    assert expired.payload["tool_call_correlation_id"] == key
    assert key != "forged"
    assert "private-call" not in key and "private-round" not in key
    assert public_start.payload["tool_call_id"] != public_end.payload["tool_call_id"]
    assert (
        key
        != project(terminal.model_copy(update={"session_id": "other"}), 3).payload[
            "tool_call_correlation_id"
        ]
    )
    assert (
        key
        != project(tool_event(EventType.TOOL_CALL_COMPLETED, "private-call", "other"), 4).payload[
            "tool_call_correlation_id"
        ]
    )
    assert (
        project(
            event("custom.extension", tool_call_correlation_id=key),
            5,
        ).payload["tool_call_correlation_id"]
        == key
    )
