from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from cayu._validation import require_clean_nonblank

ANTHROPIC_REASONING_PROTOCOL = "messages"
BEDROCK_REASONING_PROTOCOL = "converse"
BEDROCK_REASONING_PROTOCOL_VERSION = "1"


@dataclass(frozen=True)
class ReasoningStateProvenance:
    """One immutable authority identity for opaque reasoning-state replay."""

    provider: str
    protocol: str
    protocol_version: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "provider",
            require_clean_nonblank(self.provider, "reasoning_state.provider"),
        )
        object.__setattr__(
            self,
            "protocol",
            require_clean_nonblank(self.protocol, "reasoning_state.protocol"),
        )
        object.__setattr__(
            self,
            "protocol_version",
            require_clean_nonblank(
                self.protocol_version,
                "reasoning_state.protocol_version",
            ),
        )


def reasoning_state(
    state_type: str,
    *,
    provenance: ReasoningStateProvenance,
) -> dict[str, Any]:
    """Build the durable authority envelope for provider-owned reasoning state."""

    if type(provenance) is not ReasoningStateProvenance:
        raise TypeError("provenance must be a ReasoningStateProvenance.")
    return {
        "provider": provenance.provider,
        "protocol": provenance.protocol,
        "protocol_version": provenance.protocol_version,
        "type": require_clean_nonblank(state_type, "reasoning_state.type"),
    }


def reasoning_state_matches(
    state: Mapping[str, Any],
    *,
    provenance: ReasoningStateProvenance,
) -> bool:
    """Return whether opaque state is authoritative for one request protocol."""

    if type(provenance) is not ReasoningStateProvenance:
        return False
    return (
        type(state.get("provider")) is str
        and state["provider"] == provenance.provider
        and type(state.get("protocol")) is str
        and state["protocol"] == provenance.protocol
        and type(state.get("protocol_version")) is str
        and state["protocol_version"] == provenance.protocol_version
    )
