from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from typing import Any

from pydantic import SecretStr

from cayu._validation import collision_safe_json_object, copy_json_value, require_nonblank
from cayu.vaults.base import ResolvedSecret

REDACTED_SECRET = "[REDACTED_SECRET]"


def contains_redacted_secret(value: Any) -> bool:
    """Return whether JSON-compatible data contains the durable redaction marker."""

    copied = copy_json_value(value, "value")
    return _contains_redacted_secret(copied)


def _contains_redacted_secret(value: Any) -> bool:
    if type(value) is str:
        return REDACTED_SECRET in value
    if value is None or type(value) in {bool, int, float}:
        return False
    if type(value) is list:
        return any(_contains_redacted_secret(item) for item in value)
    if type(value) is dict:
        return any(
            REDACTED_SECRET in key or _contains_redacted_secret(item) for key, item in value.items()
        )
    raise AssertionError("copy_json_value returned non-JSON-compatible data.")


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
        self._pattern = _redaction_pattern(self._values)

    @property
    def has_values(self) -> bool:
        return bool(self._values)

    @property
    def max_secret_utf8_bytes(self) -> int:
        """Return the bounded overlap needed to redact across a byte boundary."""

        return max((len(value.encode("utf-8")) for value in self._values), default=0)

    def with_secret(self, secret: str | SecretStr | ResolvedSecret) -> SecretRedactor:
        value = _secret_value(secret)
        require_nonblank(value, "secret")
        values = set(self._values)
        values.add(value)
        return SecretRedactor._from_values(tuple(sorted(values, key=len, reverse=True)))

    def merged_with(self, other: SecretRedactor) -> SecretRedactor:
        """Combine two registries without exposing either registry's values."""

        if not isinstance(other, SecretRedactor):
            raise TypeError("other must be a SecretRedactor.")
        return SecretRedactor._from_values(
            tuple(sorted(set(self._values) | set(other._values), key=len, reverse=True))
        )

    def redact_text(self, value: str) -> str:
        if type(value) is not str:
            raise TypeError("SecretRedactor.redact_text expects a string.")
        if self._pattern is None:
            return value
        return self._pattern.sub(REDACTED_SECRET, value)

    def redact_text_bounded(self, value: str, *, max_bytes: int) -> str:
        """Redact a diagnostic string before applying its UTF-8 byte bound."""

        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer.")
        redacted = self.redact_text(value)
        encoded = redacted.encode("utf-8", "replace")
        if len(encoded) <= max_bytes:
            return redacted
        marker = b"...[truncated]"
        if len(marker) >= max_bytes:
            return marker[:max_bytes].decode("utf-8", "ignore")
        return (encoded[: max_bytes - len(marker)] + marker).decode("utf-8", "ignore")

    def redact_json(self, value: Any) -> Any:
        copied = copy_json_value(value, "value")
        return self._redact_copied_json(copied)

    def redact_json_values(
        self,
        value: Any,
        *,
        preserve_string_fields: Collection[str] = (),
    ) -> Any:
        """Redact JSON string values without rewriting protocol object keys."""

        preserved_fields = frozenset(preserve_string_fields)
        if any(type(field_name) is not str for field_name in preserved_fields):
            raise TypeError("preserve_string_fields must contain strings.")
        copied = copy_json_value(value, "value")
        return self._redact_copied_json_values(
            copied,
            field_name=None,
            depth=0,
            preserve_string_fields=preserved_fields,
        )

    def require_no_secret_keys(
        self,
        value: Any,
        *,
        field_name: str = "value",
        preserve_keys: Collection[str] = (),
        untrusted_container_keys: Collection[str] = (),
        match_short_substrings: bool = False,
    ) -> None:
        """Fail closed when a known secret occurs in an untyped JSON object key."""

        preserved_keys = frozenset(preserve_keys)
        untrusted_keys = frozenset(untrusted_container_keys)
        if any(type(key) is not str for key in preserved_keys):
            raise TypeError("preserve_keys must contain strings.")
        if any(type(key) is not str for key in untrusted_keys):
            raise TypeError("untrusted_container_keys must contain strings.")
        copied = copy_json_value(value, field_name)
        self._require_no_secret_keys(
            copied,
            field_name=field_name,
            preserve_keys=preserved_keys,
            untrusted_container_keys=untrusted_keys,
            inside_untrusted=False,
            match_short_substrings=match_short_substrings,
        )

    def _redact_copied_json(self, value: Any) -> Any:
        if type(value) is str:
            return self.redact_text(value)
        if value is None or type(value) in {bool, int, float}:
            return value
        if type(value) is list:
            return [self._redact_copied_json(item) for item in value]
        if type(value) is dict:
            items: list[tuple[str, Any]] = []
            for key, item in value.items():
                if type(key) is not str:
                    raise AssertionError("copy_json_value returned a non-string object key.")
                redacted_key = self.redact_text(key)
                redacted_item = self._redact_copied_json(item)
                items.append((redacted_key, redacted_item))
            return collision_safe_json_object(items, preserve_input_order=True)
        raise AssertionError("copy_json_value returned non-JSON-compatible data.")

    def _redact_copied_json_values(
        self,
        value: Any,
        *,
        field_name: str | None,
        depth: int,
        preserve_string_fields: frozenset[str],
    ) -> Any:
        if type(value) is str:
            if depth == 1 and field_name in preserve_string_fields:
                return value
            return self.redact_text(value)
        if value is None or type(value) in {bool, int, float}:
            return value
        if type(value) is list:
            return [
                self._redact_copied_json_values(
                    item,
                    field_name=field_name,
                    depth=depth + 1,
                    preserve_string_fields=preserve_string_fields,
                )
                for item in value
            ]
        if type(value) is dict:
            return {
                key: self._redact_copied_json_values(
                    item,
                    field_name=key,
                    depth=depth + 1,
                    preserve_string_fields=preserve_string_fields,
                )
                for key, item in value.items()
            }
        raise AssertionError("copy_json_value returned non-JSON-compatible data.")

    def _require_no_secret_keys(
        self,
        value: Any,
        *,
        field_name: str,
        preserve_keys: frozenset[str],
        untrusted_container_keys: frozenset[str],
        inside_untrusted: bool,
        match_short_substrings: bool,
    ) -> None:
        if value is None or type(value) in {str, bool, int, float}:
            return
        if type(value) is list:
            for item in value:
                self._require_no_secret_keys(
                    item,
                    field_name=field_name,
                    preserve_keys=preserve_keys,
                    untrusted_container_keys=untrusted_container_keys,
                    inside_untrusted=inside_untrusted,
                    match_short_substrings=match_short_substrings,
                )
            return
        if type(value) is dict:
            for key, item in value.items():
                # Public schema keys are trusted only at structural boundaries.
                # Below caller-controlled containers, even a spelling that
                # happens to match a schema key is untrusted data.
                key_is_preserved = key in preserve_keys and not inside_untrusted
                if not key_is_preserved and any(
                    key == secret
                    or (secret in key and (match_short_substrings or len(secret) >= 8))
                    for secret in self._values
                ):
                    raise ValueError(
                        f"{field_name} contains a workload secret in an object key; "
                        "refusing to publish it."
                    )
                self._require_no_secret_keys(
                    item,
                    field_name=field_name,
                    preserve_keys=preserve_keys,
                    untrusted_container_keys=untrusted_container_keys,
                    inside_untrusted=(inside_untrusted or key in untrusted_container_keys),
                    match_short_substrings=match_short_substrings,
                )
            return
        raise AssertionError("copy_json_value returned non-JSON-compatible data.")

    @classmethod
    def _from_values(cls, values: tuple[str, ...]) -> SecretRedactor:
        redactor = cls()
        redactor._values = values
        redactor._pattern = _redaction_pattern(values)
        return redactor


def _redaction_pattern(values: tuple[str, ...]) -> re.Pattern[str] | None:
    if not values:
        return None
    # A secret that starts with the public marker must win at the same position;
    # the marker must otherwise win over shorter secrets contained inside it.
    # ``re.sub`` does not rescan replacements, so one pass remains idempotent.
    marker_prefixed = [
        value for value in values if value != REDACTED_SECRET and value.startswith(REDACTED_SECRET)
    ]
    excluded = {*marker_prefixed, REDACTED_SECRET}
    remaining = [value for value in values if value not in excluded]
    alternatives = [
        *(re.escape(value) for value in marker_prefixed),
        re.escape(REDACTED_SECRET),
        *(re.escape(value) for value in remaining),
    ]
    return re.compile("|".join(alternatives))


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
