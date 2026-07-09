from __future__ import annotations

from enum import StrEnum


class CredentialMode(StrEnum):
    """How a provider credential is delivered to sandboxed workload code.

    The mode is a security contract, not a convenience flag. It records whether
    the real secret can be read by the agent/app process, so operators and audit
    events can reason about the boundary without inspecting any secret value.
    """

    RAW_ENV = "raw_env"
    """Resolve a vault secret and inject the raw value into the process env.

    This is the existing ``secret_env`` behavior. The value is **readable by the
    agent** (env vars, files, ``/proc``). Redaction reduces accidental log
    leakage but does not create a boundary. Use only for trusted environments.
    """

    TRUSTED_TOOL = "trusted_tool"
    """The secret stays host-side; a trusted tool performs one bounded action.

    The sandbox never receives the secret. Use for narrow host-side operations
    (send one email, create one invoice) rather than transparent app traffic.
    """

    VIRTUAL_EGRESS = "virtual_egress"
    """The sandbox receives only a virtual credential.

    A transparent egress broker outside the sandbox swaps the virtual credential
    for the real secret only after egress policy authorizes the request. The
    real value is **never injected** into the sandbox. Requires a runner that can
    enforce brokered egress; otherwise credential setup fails closed.
    """


CredentialModeInput = CredentialMode | str

AGENT_READABLE_MODES: frozenset[CredentialMode] = frozenset({CredentialMode.RAW_ENV})
"""Modes where the real secret value is readable by the sandboxed process."""


def normalize_credential_mode(mode: CredentialModeInput) -> CredentialMode:
    """Normalize public credential-mode input before security comparisons."""
    if isinstance(mode, CredentialMode):
        return mode
    if type(mode) is str:
        try:
            return CredentialMode(mode)
        except ValueError as exc:
            values = ", ".join(item.value for item in CredentialMode)
            raise ValueError(f"credential_mode must be one of: {values}.") from exc
    raise TypeError("credential_mode must be a CredentialMode or string value.")


def is_agent_readable(mode: CredentialModeInput) -> bool:
    """Whether the real secret is readable inside the sandbox under ``mode``."""
    return normalize_credential_mode(mode) in AGENT_READABLE_MODES
