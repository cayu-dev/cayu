from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from pydantic import SecretStr

from cayu._validation import require_unicode_scalar_text

PUBLIC_AUTHORITY_ALIAS_PREFIX = "cayu_authority_"
PUBLIC_AUTHORITY_ALIAS_VERSION = "v1"
PUBLIC_AUTHORITY_ALIAS_MAX_KEYS = 4
PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV = "CAYU_PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID"
PUBLIC_AUTHORITY_ALIAS_KEYS_ENV = "CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS"

_KEY_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,31}\Z", flags=re.ASCII)
_FIELD_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}\Z", flags=re.ASCII)
_BASE64URL_PATTERN = re.compile(r"[A-Za-z0-9_-]+\Z", flags=re.ASCII)
_HMAC_CONTEXT = b"cayu.public-authority-alias"
_KEY_FINGERPRINT_CONTEXT = b"cayu.public-authority-alias.key-fingerprint.v1"


@dataclass(frozen=True, slots=True)
class ParsedPublicAuthorityAlias:
    """Canonical, non-secret fields parsed from a public authority alias."""

    version: str
    key_id: str
    field_name: str
    tag: str


@dataclass(frozen=True, slots=True, init=False)
class PublicAuthorityAliasKeyring:
    """An immutable, secret-safe keyring for public authority aliases.

    Key material is supplied as canonical, unpadded base64url text wrapping
    exactly 32 bytes. The active key signs new aliases; retained keys only
    verify aliases produced before rotation.
    """

    active_key_id: str
    _entries: tuple[tuple[str, SecretStr], ...] = field(repr=False)

    def __init__(
        self,
        *,
        active_key_id: str,
        keys: Mapping[str, SecretStr],
    ) -> None:
        active_key_id = _require_key_id(active_key_id, field_name="active_key_id")
        if not isinstance(keys, Mapping):
            raise TypeError("keys must be a mapping of key IDs to SecretStr values.")
        if not 1 <= len(keys) <= PUBLIC_AUTHORITY_ALIAS_MAX_KEYS:
            raise ValueError(
                f"keys must contain between 1 and {PUBLIC_AUTHORITY_ALIAS_MAX_KEYS} entries."
            )

        entries: list[tuple[str, SecretStr]] = []
        for key_id, secret in keys.items():
            key_id = _require_key_id(key_id, field_name="key_id")
            if type(secret) is not SecretStr:
                raise TypeError("Public authority alias keys must be SecretStr values.")
            _decode_key(secret)
            entries.append((key_id, SecretStr(secret.get_secret_value())))
        entries.sort(key=lambda item: item[0])
        if active_key_id not in {key_id for key_id, _secret in entries}:
            raise ValueError("active_key_id must identify one of the configured keys.")

        object.__setattr__(self, "active_key_id", active_key_id)
        object.__setattr__(self, "_entries", tuple(entries))

    @property
    def key_ids(self) -> tuple[str, ...]:
        return tuple(key_id for key_id, _secret in self._entries)

    @property
    def keys(self) -> Mapping[str, SecretStr]:
        """Return an immutable mapping whose values remain masked SecretStr objects."""

        return MappingProxyType(dict(self._entries))

    def rotated(
        self,
        *,
        active_key_id: str,
        key: SecretStr,
        retire_key_ids: tuple[str, ...] = (),
    ) -> PublicAuthorityAliasKeyring:
        """Return a new keyring with a new active key and optional retirements."""

        active_key_id = _require_key_id(active_key_id, field_name="active_key_id")
        if type(key) is not SecretStr:
            raise TypeError("key must be a SecretStr value.")
        _decode_key(key)
        if type(retire_key_ids) is not tuple:
            raise TypeError("retire_key_ids must be a tuple of key IDs.")
        retire = {_require_key_id(key_id, field_name="retire_key_id") for key_id in retire_key_ids}
        if len(retire) != len(retire_key_ids):
            raise ValueError("retire_key_ids must not contain duplicates.")
        if active_key_id in retire:
            raise ValueError("The active key cannot be retired.")

        updated = {
            key_id: SecretStr(secret.get_secret_value())
            for key_id, secret in self._entries
            if key_id not in retire
        }
        existing = updated.get(active_key_id)
        if existing is not None and not hmac.compare_digest(
            _decode_key(existing),
            _decode_key(key),
        ):
            raise ValueError("An existing key ID cannot be assigned different key material.")
        updated[active_key_id] = SecretStr(key.get_secret_value())
        return PublicAuthorityAliasKeyring(active_key_id=active_key_id, keys=updated)

    def _key_bytes(self, key_id: str) -> bytes | None:
        for candidate, secret in self._entries:
            if hmac.compare_digest(candidate.encode("ascii"), key_id.encode("ascii")):
                return _decode_key(secret)
        return None

    def __repr__(self) -> str:
        return (
            "PublicAuthorityAliasKeyring("
            f"active_key_id={self.active_key_id!r}, key_ids={self.key_ids!r})"
        )


@dataclass(frozen=True, slots=True)
class PublicAuthorityAliasCodec:
    """Create and verify versioned, field- and session-scoped HMAC aliases."""

    keyring: PublicAuthorityAliasKeyring

    def __post_init__(self) -> None:
        if type(self.keyring) is not PublicAuthorityAliasKeyring:
            raise TypeError("keyring must be a PublicAuthorityAliasKeyring.")

    def encode(
        self,
        private_value: str,
        *,
        field_name: str,
        session_id: str | None = None,
    ) -> str:
        private_value = _require_nonempty_scalar(private_value, field_name="private_value")
        field_name = _require_field_name(field_name)
        session_id = _require_optional_session_id(session_id)
        return self._encode_with_key_id(
            private_value,
            field_name=field_name,
            session_id=session_id,
            key_id=self.keyring.active_key_id,
        )

    def aliases(
        self,
        private_value: str,
        *,
        field_name: str,
        session_id: str | None = None,
    ) -> tuple[str, ...]:
        """Return aliases for every accepted key, active key first.

        Persistent stores use this during a keyed backfill so aliases issued
        before rotation remain resolvable without request-time scans.
        """

        ordered_key_ids = (
            self.keyring.active_key_id,
            *(key_id for key_id in self.keyring.key_ids if key_id != self.keyring.active_key_id),
        )
        return tuple(
            self._encode_with_key_id(
                private_value,
                field_name=field_name,
                session_id=session_id,
                key_id=key_id,
            )
            for key_id in ordered_key_ids
        )

    def key_fingerprint(self, key_id: str) -> str:
        """Return a non-secret stable fingerprint for deployment validation."""

        key_id = _require_key_id(key_id, field_name="key_id")
        key = self.keyring._key_bytes(key_id)
        if key is None:
            raise ValueError("key_id is not configured in this alias keyring.")
        return hmac.digest(key, _KEY_FINGERPRINT_CONTEXT, "sha256").hex()

    def keyring_fingerprint(self) -> str:
        """Return a stable non-secret digest of the exact configured key set."""

        material = json.dumps(
            [[key_id, self.key_fingerprint(key_id)] for key_id in self.keyring.key_ids],
            separators=(",", ":"),
        ).encode("ascii")
        return hashlib.sha256(material).hexdigest()

    def _encode_with_key_id(
        self,
        private_value: str,
        *,
        field_name: str,
        session_id: str | None,
        key_id: str,
    ) -> str:
        private_value = _require_nonempty_scalar(private_value, field_name="private_value")
        field_name = _require_field_name(field_name)
        session_id = _require_optional_session_id(session_id)
        key = self.keyring._key_bytes(key_id)
        if key is None:  # pragma: no cover - guarded by aliases/encode callers
            raise RuntimeError("The public authority alias key is unavailable.")
        tag = _base64url_encode(
            hmac.digest(
                key,
                _mac_input(
                    key_id=key_id,
                    field_name=field_name,
                    session_id=session_id,
                    private_value=private_value,
                ),
                "sha256",
            )
        )
        return (
            f"{PUBLIC_AUTHORITY_ALIAS_PREFIX}{PUBLIC_AUTHORITY_ALIAS_VERSION}."
            f"{key_id}.{field_name}.{tag}"
        )

    def parse(self, value: str) -> ParsedPublicAuthorityAlias | None:
        """Parse a canonical supported alias without verifying private authority."""

        return parse_public_authority_alias(value)

    def matches(
        self,
        alias: str,
        private_value: str,
        *,
        field_name: str,
        session_id: str | None = None,
    ) -> bool:
        parsed = parse_public_authority_alias(alias)
        if parsed is None:
            return False
        private_value = _require_nonempty_scalar(private_value, field_name="private_value")
        field_name = _require_field_name(field_name)
        session_id = _require_optional_session_id(session_id)
        if parsed.field_name != field_name:
            return False
        key = self.keyring._key_bytes(parsed.key_id)
        if key is None:
            return False
        expected = hmac.digest(
            key,
            _mac_input(
                key_id=parsed.key_id,
                field_name=field_name,
                session_id=session_id,
                private_value=private_value,
            ),
            "sha256",
        )
        tag = _base64url_decode(parsed.tag)
        return tag is not None and hmac.compare_digest(expected, tag)

    def rotated(
        self,
        *,
        active_key_id: str,
        key: SecretStr,
        retire_key_ids: tuple[str, ...] = (),
    ) -> PublicAuthorityAliasCodec:
        """Return an immutable codec using a rotated copy of this keyring."""

        return PublicAuthorityAliasCodec(
            self.keyring.rotated(
                active_key_id=active_key_id,
                key=key,
                retire_key_ids=retire_key_ids,
            )
        )

    @staticmethod
    def is_reserved(value: object) -> bool:
        return public_authority_alias_is_reserved(value)


def parse_public_authority_alias(value: str) -> ParsedPublicAuthorityAlias | None:
    """Return the canonical alias components, or None for malformed input."""

    if type(value) is not str or not value.startswith(PUBLIC_AUTHORITY_ALIAS_PREFIX):
        return None
    components = value.removeprefix(PUBLIC_AUTHORITY_ALIAS_PREFIX).split(".")
    if len(components) != 4:
        return None
    version, key_id, field_name, tag = components
    if version != PUBLIC_AUTHORITY_ALIAS_VERSION:
        return None
    if _KEY_ID_PATTERN.fullmatch(key_id) is None:
        return None
    if _FIELD_NAME_PATTERN.fullmatch(field_name) is None:
        return None
    if _base64url_decode(tag) is None:
        return None
    return ParsedPublicAuthorityAlias(
        version=version,
        key_id=key_id,
        field_name=field_name,
        tag=tag,
    )


def public_authority_alias_is_reserved(value: object) -> bool:
    """Return whether a value occupies the public-authority alias namespace."""

    return type(value) is str and value.startswith(PUBLIC_AUTHORITY_ALIAS_PREFIX)


def public_authority_alias_codec_from_environment(
    environ: Mapping[str, str] | None = None,
) -> PublicAuthorityAliasCodec | None:
    """Load an explicit deployment codec from two environment variables.

    ``CAYU_PUBLIC_AUTHORITY_ALIAS_KEYS`` is a JSON object mapping key IDs to
    canonical base64url key material. The active key ID is supplied separately
    so rotation never depends on mapping order. No variables means aliases are
    unconfigured; partial or malformed configuration fails closed.
    """

    if environ is None:
        import os

        environ = os.environ
    if not isinstance(environ, Mapping):
        raise TypeError("environ must be a string mapping.")
    active_key_id = environ.get(PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV)
    encoded_keys = environ.get(PUBLIC_AUTHORITY_ALIAS_KEYS_ENV)
    if active_key_id is None and encoded_keys is None:
        return None
    if active_key_id is None or encoded_keys is None:
        raise ValueError(
            "Public authority alias deployment configuration requires both "
            f"{PUBLIC_AUTHORITY_ALIAS_ACTIVE_KEY_ID_ENV} and "
            f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV}."
        )
    try:
        raw_keys = json.loads(encoded_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} must be a JSON object.") from exc
    if type(raw_keys) is not dict or not raw_keys:
        raise ValueError(f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} must be a non-empty JSON object.")
    keys: dict[str, SecretStr] = {}
    for key_id, encoded_key in raw_keys.items():
        if type(key_id) is not str or type(encoded_key) is not str:
            raise ValueError(
                f"{PUBLIC_AUTHORITY_ALIAS_KEYS_ENV} must map string IDs to string keys."
            )
        keys[key_id] = SecretStr(encoded_key)
    return PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id=active_key_id,
            keys=keys,
        )
    )


def _require_key_id(value: object, *, field_name: str) -> str:
    if type(value) is not str or _KEY_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical lowercase ASCII key identifier.")
    return value


def _require_field_name(value: object) -> str:
    if type(value) is not str or _FIELD_NAME_PATTERN.fullmatch(value) is None:
        raise ValueError("field_name must be a canonical lowercase ASCII field identifier.")
    return value


def _require_nonempty_scalar(value: object, *, field_name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return require_unicode_scalar_text(value, field_name)


def _require_optional_session_id(value: object) -> str | None:
    if value is None:
        return None
    return _require_nonempty_scalar(value, field_name="session_id")


def _decode_key(secret: SecretStr) -> bytes:
    encoded = secret.get_secret_value()
    decoded = _base64url_decode(encoded)
    if decoded is None:
        raise ValueError(
            "Public authority alias keys must be canonical unpadded base64url values "
            "encoding exactly 32 bytes."
        )
    return decoded


def _base64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64url_decode(value: object) -> bytes | None:
    if type(value) is not str or len(value) != 43 or _BASE64URL_PATTERN.fullmatch(value) is None:
        return None
    try:
        decoded = base64.b64decode(value + "=", altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        return None
    if len(decoded) != 32 or _base64url_encode(decoded) != value:
        return None
    return decoded


def _mac_input(
    *,
    key_id: str,
    field_name: str,
    session_id: str | None,
    private_value: str,
) -> bytes:
    components = (
        _HMAC_CONTEXT,
        PUBLIC_AUTHORITY_ALIAS_VERSION.encode("ascii"),
        key_id.encode("ascii"),
        field_name.encode("ascii"),
        b"0" if session_id is None else b"1",
        b"" if session_id is None else session_id.encode("utf-8"),
        private_value.encode("utf-8"),
    )
    return b"".join(len(component).to_bytes(8, "big") + component for component in components)
