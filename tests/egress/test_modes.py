from __future__ import annotations

from cayu.core.events import EventType
from cayu.credentials import (
    AGENT_READABLE_MODES,
    CredentialMode,
    is_agent_readable,
    normalize_credential_mode,
)


def test_credential_mode_values() -> None:
    assert CredentialMode.RAW_ENV == "raw_env"
    assert CredentialMode.TRUSTED_TOOL == "trusted_tool"
    assert CredentialMode.VIRTUAL_EGRESS == "virtual_egress"


def test_only_raw_env_is_agent_readable() -> None:
    assert is_agent_readable(CredentialMode.RAW_ENV)
    assert is_agent_readable("raw_env")
    assert not is_agent_readable(CredentialMode.TRUSTED_TOOL)
    assert not is_agent_readable(CredentialMode.VIRTUAL_EGRESS)
    assert frozenset({CredentialMode.RAW_ENV}) == AGENT_READABLE_MODES


def test_normalize_credential_mode_accepts_string_values() -> None:
    assert normalize_credential_mode("virtual_egress") is CredentialMode.VIRTUAL_EGRESS


def test_egress_event_types_present() -> None:
    assert EventType.CREDENTIAL_MODE_SELECTED == "credential.mode.selected"
    assert EventType.EGRESS_GRANT_MINTED == "egress.grant.minted"
    assert EventType.EGRESS_GRANT_REVOKED == "egress.grant.revoked"
    assert EventType.EGRESS_REQUEST_AUTHORIZED == "egress.request.authorized"
    assert EventType.EGRESS_REQUEST_DENIED == "egress.request.denied"
