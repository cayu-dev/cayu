from __future__ import annotations

import ipaddress
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from cayu._validation import require_clean_nonblank

EgressProtocol = Literal["https"]
_HOST_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")


@dataclass(frozen=True)
class ApprovedEgressDestination:
    """One credentialless HTTPS destination governed by an egress policy.

    This declaration intentionally has no ``SecretRef`` or guest credential.
    The broker authorizes each request against ``policy_name`` and forwards the
    caller's headers unchanged. Cayu v1 only supports broker-terminated HTTPS on
    the standard port; other transports fail closed instead of being silently
    widened.
    """

    destination: str
    policy_name: str
    protocol: EgressProtocol = "https"
    port: int = 443

    def __post_init__(self) -> None:
        destination = _validated_hostname(self.destination)
        policy_name = require_clean_nonblank(self.policy_name, "policy_name")
        if self.protocol != "https":
            raise ValueError("Approved egress destinations only support protocol='https'.")
        if type(self.port) is not int:
            raise TypeError("Approved egress destination port must be an integer.")
        if self.port != 443:
            raise ValueError("Approved egress destinations only support port=443.")
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "policy_name", policy_name)

    @property
    def authority(self) -> tuple[str, str, int]:
        return (self.destination, self.protocol, self.port)


def _validated_hostname(value: str) -> str:
    hostname = require_clean_nonblank(value, "destination").lower().rstrip(".")
    if any(character in hostname for character in "/:@?#") or any(
        character.isspace() for character in hostname
    ):
        raise ValueError("Approved egress destination must be a bare hostname.")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("Approved egress destination must not be an IP address.")
    if len(hostname) > 253 or "." not in hostname:
        raise ValueError("Approved egress destination must be a fully qualified hostname.")
    if any(_HOST_LABEL.fullmatch(label) is None for label in hostname.split(".")):
        raise ValueError("Approved egress destination is not a valid hostname.")
    return hostname


def validate_approved_destinations(
    destinations: Sequence[ApprovedEgressDestination],
) -> tuple[ApprovedEgressDestination, ...]:
    """Copy and reject duplicate or untyped approved destinations."""

    seen: set[tuple[str, str, int]] = set()
    validated: list[ApprovedEgressDestination] = []
    for destination in destinations:
        if type(destination) is not ApprovedEgressDestination:
            raise TypeError(
                "approved_destinations entries must be ApprovedEgressDestination instances."
            )
        if destination.authority in seen:
            raise ValueError(
                f"Approved egress destination {destination.destination!r} is duplicated."
            )
        seen.add(destination.authority)
        validated.append(destination)
    return tuple(validated)
