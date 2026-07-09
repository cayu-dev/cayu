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


class VirtualCredentialError(EgressError):
    """A virtual credential was unknown, expired, or revoked at the broker."""
