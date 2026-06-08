from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import SecretStr

from cayu._validation import copy_json_value, require_nonblank
from cayu.vaults.base import ResolvedSecret

REDACTED_SECRET = "[REDACTED_SECRET]"


class SecretRedactor:
    """Redacts known resolved secret values from strings and JSON-like data."""

    def __init__(
        self,
        secrets: str
        | SecretStr
        | ResolvedSecret
        | Sequence[str | SecretStr | ResolvedSecret]
        | None = None,
    ) -> None:
        values: set[str] = set()
        for secret in _secret_items(secrets):
            values.add(_secret_value(secret))
        self._values = tuple(sorted(values, key=len, reverse=True))

    def with_secret(self, secret: str | SecretStr | ResolvedSecret) -> SecretRedactor:
        value = _secret_value(secret)
        require_nonblank(value, "secret")
        values = set(self._values)
        values.add(value)
        return SecretRedactor._from_values(tuple(sorted(values, key=len, reverse=True)))

    def redact_text(self, value: str) -> str:
        if type(value) is not str:
            raise TypeError("SecretRedactor.redact_text expects a string.")
        redacted = value
        for secret in self._values:
            redacted = redacted.replace(secret, REDACTED_SECRET)
        return redacted

    def redact_json(self, value: Any) -> Any:
        copied = copy_json_value(value, "value")
        return self._redact_copied_json(copied)

    def _redact_copied_json(self, value: Any) -> Any:
        if type(value) is str:
            return self.redact_text(value)
        if value is None or type(value) in {bool, int, float}:
            return value
        if type(value) is list:
            return [self._redact_copied_json(item) for item in value]
        if type(value) is dict:
            return {key: self._redact_copied_json(item) for key, item in value.items()}
        raise AssertionError("copy_json_value returned non-JSON-compatible data.")

    @classmethod
    def _from_values(cls, values: tuple[str, ...]) -> SecretRedactor:
        redactor = cls()
        redactor._values = values
        return redactor


def _secret_value(secret: str | SecretStr | ResolvedSecret) -> str:
    if type(secret) is str:
        return require_nonblank(secret, "secret")
    if type(secret) is SecretStr:
        return require_nonblank(secret.get_secret_value(), "secret")
    if type(secret) is ResolvedSecret:
        return require_nonblank(secret.value.get_secret_value(), "secret")
    raise TypeError("SecretRedactor secrets must be str, SecretStr, or ResolvedSecret.")


def _secret_items(
    secrets: str | SecretStr | ResolvedSecret | Sequence[str | SecretStr | ResolvedSecret] | None,
) -> tuple[str | SecretStr | ResolvedSecret, ...]:
    if secrets is None:
        return ()
    if type(secrets) is str:
        return (secrets,)
    if type(secrets) is SecretStr:
        return (secrets,)
    if type(secrets) is ResolvedSecret:
        return (secrets,)
    if not isinstance(secrets, Sequence):
        raise TypeError("SecretRedactor secrets must be a secret or a sequence of secrets.")
    items: list[str | SecretStr | ResolvedSecret] = []
    for secret in secrets:
        items.append(_as_secret(secret))
    return tuple(items)


def _as_secret(secret: object) -> str | SecretStr | ResolvedSecret:
    if type(secret) is str:
        return secret
    if type(secret) is SecretStr:
        return secret
    if type(secret) is ResolvedSecret:
        return secret
    raise TypeError("SecretRedactor secrets must be str, SecretStr, or ResolvedSecret.")
