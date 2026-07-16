from __future__ import annotations


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


class InvalidEgressReconnectMetadataError(EgressReconnectError):
    """Durable reconnect metadata is malformed, stale, or out of scope."""


class UnsupportedEgressReconnectError(EgressReconnectError):
    """The selected adapter cannot safely re-establish enforced egress."""


class EgressReconnectConflictError(EgressReconnectError):
    """Another owner already holds the reconnectable sandbox boundary."""


class EgressReconnectNotFoundError(EgressReconnectError):
    """The sandbox named by durable reconnect metadata no longer exists."""


class VirtualCredentialError(EgressError):
    """A virtual credential was unknown, expired, or revoked at the broker."""
