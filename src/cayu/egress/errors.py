from __future__ import annotations

import asyncio
from typing import Any


class EgressError(RuntimeError):
    """Base error for the virtual egress subsystem."""


class UnsupportedEgressError(EgressError):
    """A runner cannot enforce or capture egress for ``virtual_egress``.

    This is the fail-closed signal: Cayu must never downgrade ``virtual_egress``
    to raw secret injection when a runner cannot prove that direct provider
    egress is blocked or captured. Adapters raise this instead of silently
    weakening the credential boundary.
    """


class UnsupportedEgressCapabilityError(UnsupportedEgressError):
    """A named enforcement capability is unavailable for one runner kind."""

    def __init__(
        self,
        *,
        runner_kind: str,
        capability: str,
        reason: str,
        remediation: str,
    ) -> None:
        self.runner_kind = runner_kind
        self.capability = capability
        self.reason = reason
        self.remediation = remediation
        super().__init__(
            f"Runner {runner_kind!r} cannot verify required egress capability "
            f"{capability!r}: {reason}. Remediation: {remediation}."
        )


class EgressReconnectError(EgressError):
    """Base error for a fail-closed virtual-egress reconnect attempt."""


class DockerEgressReconnectError(EgressReconnectError):
    """Bounded Docker failure classification without backend diagnostic text."""

    def __init__(self, code: str) -> None:
        if code not in {
            "ownership_conflict",
            "ownership_uncertain",
            "allocation_absent",
            "identity_mismatch",
            "configuration_mismatch",
            "daemon_unavailable",
            "disposal_pending",
            "disposed",
            "fencing_failed",
            "preflight_failed",
            "state_unavailable",
            "unsupported_host",
            "listener_conflict",
            "control_server_unavailable",
            "control_server_alias_conflict",
        }:
            raise ValueError("Unknown Docker reconnect error code.")
        self.code = code
        super().__init__(f"Docker virtual-egress reconnect failed: {code}.")


class InvalidEgressReconnectMetadataError(EgressReconnectError):
    """Durable reconnect metadata is malformed, stale, or out of scope."""


class UnsupportedEgressReconnectError(EgressReconnectError):
    """The selected adapter cannot safely re-establish enforced egress."""


class EgressAuthorityCutoverError(EgressError):
    """A governed egress-authority replacement did not become active."""


class UnsupportedEgressAuthorityCutoverError(EgressAuthorityCutoverError):
    """The selected adapter cannot safely replace its active egress authority."""


class EgressAuthorityCutoverNeedsAttention(EgressAuthorityCutoverError):
    """A backend cutover mutation was dispatched but activation is unproven."""

    def __init__(
        self,
        message: str,
        *,
        replacement_binding: Any,
        environment_fingerprint: str | None,
        target_authority_installed: bool = True,
        settlement_task: Any = None,
        cancellation: asyncio.CancelledError | None = None,
        cancellation_requests_consumed: int = 0,
    ) -> None:
        if type(target_authority_installed) is not bool:
            raise TypeError("target_authority_installed must be a bool.")
        self.replacement_binding = replacement_binding
        self.environment_fingerprint = environment_fingerprint
        self.target_authority_installed = target_authority_installed
        self.settlement_task = settlement_task
        if cancellation is not None and not isinstance(cancellation, asyncio.CancelledError):
            raise TypeError("cancellation must be CancelledError or None.")
        if type(cancellation_requests_consumed) is not int or cancellation_requests_consumed < 0:
            raise ValueError("cancellation_requests_consumed must be non-negative.")
        self.cancellation = cancellation
        self.cancellation_requests_consumed = cancellation_requests_consumed
        super().__init__(message)


class EgressReconnectConflictError(EgressReconnectError):
    """Another owner already holds the reconnectable sandbox boundary."""


class EgressReconnectNotFoundError(EgressReconnectError):
    """The sandbox named by durable reconnect metadata no longer exists."""


class VirtualCredentialError(EgressError):
    """A virtual credential was unknown, expired, or revoked at the broker."""
