from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

_STRIPE_TEST_PREFIX = "sk_test_cayu_vc_"
_GENERIC_PREFIX = "cayu_vc_"
_VIRTUAL_VALUE_ENTROPY_BYTES = 24
_VIRTUAL_VALUE_TOKEN_HEX_CHARS = _VIRTUAL_VALUE_ENTROPY_BYTES * 2
_HEX_CHARS = frozenset("0123456789abcdef")

HeaderRewriter = Callable[[str, Mapping[str, str]], dict[str, str]]
CredentialKind: TypeAlias = Literal["stripe_bearer", "opaque_bearer"]


@dataclass(frozen=True)
class CredentialKindDescriptor:
    """Virtual credential behavior for one provider/API credential shape."""

    credential_kind: str
    virtual_prefix: str
    header_rewriter: HeaderRewriter | None = None
    test_mode_real_secret_prefixes: tuple[str, ...] = ()

    def rewrite_headers(self, secret: str, headers: Mapping[str, str]) -> dict[str, str]:
        if self.header_rewriter is None:
            raise ValueError(f"Unsupported credential kind {self.credential_kind!r}.")
        return self.header_rewriter(secret, headers)

    def validate_presented_value(self, value: str) -> None:
        """Reject caller-supplied values outside Cayu's virtual namespace."""
        if not value.startswith(self.virtual_prefix):
            raise ValueError(
                "presented_value must use Cayu's virtual credential prefix for "
                f"{self.credential_kind!r}; refusing to register a raw provider credential."
            )
        suffix = value[len(self.virtual_prefix) :]
        if len(suffix) != _VIRTUAL_VALUE_TOKEN_HEX_CHARS or any(
            char not in _HEX_CHARS for char in suffix
        ):
            raise ValueError(
                "presented_value must be a Cayu virtual credential with a generated "
                "hex entropy suffix."
            )


def _without_header(headers: Mapping[str, str], name: str) -> dict[str, str]:
    lowered = name.lower()
    return {key: value for key, value in headers.items() if key.lower() != lowered}


def _bearer_rewriter(secret: str, headers: Mapping[str, str]) -> dict[str, str]:
    rewritten = _without_header(headers, "authorization")
    rewritten["Authorization"] = f"Bearer {secret}"
    return rewritten


SUPPORTED_CREDENTIAL_KINDS: dict[str, CredentialKindDescriptor] = {
    "stripe_bearer": CredentialKindDescriptor(
        credential_kind="stripe_bearer",
        virtual_prefix=_STRIPE_TEST_PREFIX,
        header_rewriter=_bearer_rewriter,
        test_mode_real_secret_prefixes=("sk_test_", "rk_test_"),
    ),
    "opaque_bearer": CredentialKindDescriptor(
        credential_kind="opaque_bearer",
        virtual_prefix=_GENERIC_PREFIX,
        header_rewriter=_bearer_rewriter,
    ),
}


def virtual_credential_entropy_bytes() -> int:
    return _VIRTUAL_VALUE_ENTROPY_BYTES


def credential_kind_descriptor(credential_kind: str) -> CredentialKindDescriptor:
    """Return descriptor data for a supported virtual credential kind."""
    descriptor = SUPPORTED_CREDENTIAL_KINDS.get(credential_kind)
    if descriptor is not None:
        return descriptor
    raise ValueError(f"Unsupported credential kind {credential_kind!r}.")


def validate_credential_kind(credential_kind: str) -> CredentialKind:
    credential_kind_descriptor(credential_kind)
    return cast("CredentialKind", credential_kind)


def supported_credential_kind_descriptor(
    credential_kind: str,
) -> CredentialKindDescriptor | None:
    return SUPPORTED_CREDENTIAL_KINDS.get(credential_kind)


def validate_presented_value(credential_kind: str, value: str) -> None:
    credential_kind_descriptor(credential_kind).validate_presented_value(value)
