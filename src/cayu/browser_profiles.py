"""Application-owned encrypted state for fresh interactive-browser allocations.

Browser profiles are deliberately separate from live browser-session recovery,
artifacts, workspaces, and the ordinary session store.  The public models in
this module carry only authority and bounded status.  Cookie and web-storage
plaintext is accepted only at the encryption/decryption boundary and encrypted
envelopes are visible only to profile-store implementations.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import secrets
import sqlite3
import threading
import traceback as traceback_module
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Never, Self, TypeVar, cast
from urllib.parse import urlsplit

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._task_wait import (
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
)
from cayu.egress.destinations import normalize_egress_hostname

BROWSER_PROFILE_SCHEMA_VERSION = 1
BROWSER_PROFILE_STATE_SCHEMA_VERSION = 1
BROWSER_PROFILE_ENCRYPTION_ALGORITHM = "AES-256-GCM"
BROWSER_PROFILE_NONCE_BYTES = 12
BROWSER_PROFILE_MAX_ORIGINS = 32
BROWSER_PROFILE_MAX_COOKIES = 256
BROWSER_PROFILE_MAX_STORAGE_ENTRIES = 512
BROWSER_PROFILE_MAX_NAME_BYTES = 1_024
BROWSER_PROFILE_MAX_VALUE_BYTES = 64 * 1_024
BROWSER_PROFILE_MAX_PLAINTEXT_BYTES = 1024 * 1_024
BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES = BROWSER_PROFILE_MAX_PLAINTEXT_BYTES + 16
BROWSER_PROFILE_MAX_IMPORT_EXPORT_SECONDS = 30.0
BROWSER_PROFILE_MAX_LEASE_SECONDS = 3_600
BROWSER_PROFILE_MAX_RECEIPTS = 128

_EMPTY_CONTENT_FINGERPRINT = sha256(b"cayu.browser-profile.empty.v1").hexdigest()
_SHA256 = frozenset("0123456789abcdef")
_SAFE_ID = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:-")
_INCLUDED_CATEGORIES = ("cookies.v1", "origin_storage.v1")
_RESTORE_ERROR_CODES = frozenset(
    {
        "profile_corrupt",
        "profile_incompatible",
        "profile_unavailable",
        "restore_cancelled_before_import",
        "restore_outcome_unknown",
        "restore_rejected",
    }
)
_CHECKPOINT_ERROR_CODES = frozenset(
    {
        "checkpoint_cancelled_before_export",
        "checkpoint_failed",
        "checkpoint_outcome_unknown",
    }
)
_INSPECTION_ERROR_CODES = frozenset(
    {*_RESTORE_ERROR_CODES, *_CHECKPOINT_ERROR_CODES, "writer_lease_expired"}
)
_OMITTED_CATEGORIES = (
    "browser_executable_state",
    "cache",
    "downloads",
    "extensions",
    "history",
    "in_flight_requests",
    "indexed_db",
    "javascript_heap",
    "open_pages",
    "password_manager",
    "profile_directories",
    "refs",
    "renderer_state",
    "saved_passwords",
    "screenshots",
    "service_worker_bodies",
    "session_storage",
    "sockets",
    "totp_seeds",
    "traces",
)
_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ValueT = TypeVar("_ValueT")


def _utc(value: datetime, field_name: str) -> datetime:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime.")
    return value.astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)


def _clean(value: str, field_name: str, *, max_bytes: int = 512) -> str:
    value = require_durable_clean_nonblank(value, field_name)
    if len(value.encode("utf-8")) > max_bytes:
        raise ValueError(f"{field_name} exceeds its byte bound.")
    return value


def _identifier(value: str, field_name: str, *, max_chars: int = 256) -> str:
    value = _clean(value, field_name, max_bytes=max_chars)
    if len(value) > max_chars or any(character not in _SAFE_ID for character in value):
        raise ValueError(f"{field_name} must be a bounded opaque identifier.")
    return value


def _digest(value: str, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256 for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def _fixed_error_code(
    value: str,
    field_name: str,
    *,
    allowed: frozenset[str],
) -> str:
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{field_name} is not a supported browser-profile error code.")
    return value


def _fingerprint(domain: bytes, value: object, field_name: str) -> str:
    return sha256(domain + b"\0" + canonical_durable_json_bytes(value, field_name)).hexdigest()


def _portable_text(value: str, field_name: str, *, max_bytes: int) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string.")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be portable Unicode text.") from exc
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL characters.")
    if len(encoded) > max_bytes:
        raise ValueError(f"{field_name} exceeds its byte bound.")
    return value


def _canonical_origin(value: str, field_name: str = "origin") -> str:
    value = _clean(value, field_name, max_bytes=2_048)
    split = urlsplit(value)
    try:
        port = split.port
    except ValueError as exc:
        raise ValueError(f"{field_name} must be one canonical HTTPS origin.") from exc
    if (
        split.scheme.lower() != "https"
        or split.hostname is None
        or split.username is not None
        or split.password is not None
        or port not in {None, 443}
        or split.path not in {"", "/"}
        or split.query
        or split.fragment
    ):
        raise ValueError(f"{field_name} must be one canonical HTTPS origin.")
    host = normalize_egress_hostname(split.hostname, field_name=field_name)
    return f"https://{host}"


def _origin_host(origin: str) -> str:
    host = urlsplit(origin).hostname
    if host is None:  # pragma: no cover - canonical origin invariant
        raise ValueError("Browser profile origin has no hostname.")
    return host


class _ProfileModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )


class _PrivateProfileStateModel(_ProfileModel):
    """Credential-bearing profile plaintext with content-free diagnostics."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(<private>)"

    def __str__(self) -> str:
        return f"{type(self).__name__}(<private>)"


class BrowserProfileCheckpointPolicy(StrEnum):
    DISABLED = "disabled"
    AFTER_TERMINAL_OPERATION = "after_terminal_operation"
    ON_CLOSE = "on_close"


class BrowserProfileTerminalOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"


class BrowserProfileStatus(StrEnum):
    AVAILABLE = "available"
    EMPTY = "empty"
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INCOMPATIBLE = "incompatible"
    CORRUPT = "corrupt"
    OUTCOME_UNKNOWN = "outcome_unknown"


class BrowserProfileLimits(_ProfileModel):
    max_origins: StrictInt = Field(default=8, ge=1, le=BROWSER_PROFILE_MAX_ORIGINS)
    max_cookies: StrictInt = Field(default=128, ge=1, le=BROWSER_PROFILE_MAX_COOKIES)
    max_storage_entries: StrictInt = Field(
        default=256,
        ge=1,
        le=BROWSER_PROFILE_MAX_STORAGE_ENTRIES,
    )
    max_name_bytes: StrictInt = Field(default=512, ge=1, le=BROWSER_PROFILE_MAX_NAME_BYTES)
    max_value_bytes: StrictInt = Field(
        default=16 * 1_024,
        ge=1,
        le=BROWSER_PROFILE_MAX_VALUE_BYTES,
    )
    max_plaintext_bytes: StrictInt = Field(
        default=256 * 1_024,
        ge=1,
        le=BROWSER_PROFILE_MAX_PLAINTEXT_BYTES,
    )
    max_ciphertext_bytes: StrictInt = Field(
        default=256 * 1_024 + 16,
        ge=17,
        le=BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES,
    )
    import_timeout_seconds: StrictFloat = Field(
        default=10.0,
        gt=0,
        le=BROWSER_PROFILE_MAX_IMPORT_EXPORT_SECONDS,
    )
    export_timeout_seconds: StrictFloat = Field(
        default=10.0,
        gt=0,
        le=BROWSER_PROFILE_MAX_IMPORT_EXPORT_SECONDS,
    )

    @model_validator(mode="after")
    def validate_ciphertext_capacity(self) -> Self:
        if self.max_ciphertext_bytes < self.max_plaintext_bytes + 16:
            raise ValueError("max_ciphertext_bytes must include authenticated-encryption overhead.")
        return self


class BrowserProfileScope(_ProfileModel):
    owner_fingerprint: StrictStr
    sharing_fingerprint: StrictStr

    @field_validator("owner_fingerprint", "sharing_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @classmethod
    def build(
        cls,
        *,
        application_id: str,
        tenant_id: str,
        sharing_scope: str,
    ) -> BrowserProfileScope:
        owner = {
            "application_id": _clean(application_id, "application_id", max_bytes=256),
            "tenant_id": _clean(tenant_id, "tenant_id", max_bytes=256),
        }
        sharing = _clean(sharing_scope, "sharing_scope", max_bytes=512)
        return cls(
            owner_fingerprint=_fingerprint(
                b"cayu.browser-profile.owner.v1",
                owner,
                "browser profile owner",
            ),
            sharing_fingerprint=_fingerprint(
                b"cayu.browser-profile.sharing.v1",
                {
                    "owner_fingerprint": _fingerprint(
                        b"cayu.browser-profile.owner.v1",
                        owner,
                        "browser profile owner",
                    ),
                    "sharing_scope": sharing,
                },
                "browser profile sharing scope",
            ),
        )


class BrowserProfileDestinationPolicy(_ProfileModel):
    schema_version: Literal[1] = 1
    origins: tuple[StrictStr, ...]
    fingerprint: StrictStr

    @field_validator("origins", mode="before")
    @classmethod
    def normalize_origins(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, list | tuple):
            raise TypeError("origins must be an ordered collection.")
        if not value or len(value) > BROWSER_PROFILE_MAX_ORIGINS:
            raise ValueError("origins exceed the browser-profile origin bound.")
        if any(type(item) is not str for item in value):
            raise TypeError("origins must contain strings.")
        copied = tuple(
            _canonical_origin(item, "origin") for item in cast("list[str] | tuple[str, ...]", value)
        )
        if copied != tuple(sorted(set(copied))):
            raise ValueError("origins must be unique and sorted.")
        return copied

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _digest(value, "fingerprint")

    @model_validator(mode="after")
    def validate_content(self) -> Self:
        expected = _fingerprint(
            b"cayu.browser-profile.destinations.v1",
            {"schema_version": self.schema_version, "origins": list(self.origins)},
            "browser profile destinations",
        )
        if self.fingerprint != expected:
            raise ValueError("Destination-policy fingerprint does not match its origins.")
        return self

    @classmethod
    def build(cls, origins: Iterable[str]) -> BrowserProfileDestinationPolicy:
        collected: list[str] = []
        for item in origins:
            if len(collected) >= BROWSER_PROFILE_MAX_ORIGINS:
                raise ValueError("origins exceed the browser-profile origin bound.")
            collected.append(_canonical_origin(item, "origin"))
        normalized = tuple(sorted(set(collected)))
        material = {"schema_version": 1, "origins": list(normalized)}
        return cls(
            schema_version=1,
            origins=normalized,
            fingerprint=_fingerprint(
                b"cayu.browser-profile.destinations.v1",
                material,
                "browser profile destinations",
            ),
        )

    def is_narrower_than(self, recorded: BrowserProfileDestinationPolicy) -> bool:
        return set(self.origins) <= set(recorded.origins)

    def admits_url(self, value: str) -> bool:
        """Return whether one HTTPS URL is within this exact origin policy."""

        try:
            split = urlsplit(_clean(value, "url", max_bytes=8_192))
            if split.hostname is None or split.scheme.lower() != "https":
                return False
            port = split.port
            if (
                split.username is not None
                or split.password is not None
                or port
                not in {
                    None,
                    443,
                }
            ):
                return False
            origin = _canonical_origin(f"https://{split.hostname}")
        except (TypeError, ValueError):
            return False
        return origin in self.origins


class BrowserProfileStorageEntry(_PrivateProfileStateModel):
    name: StrictStr
    value: StrictStr

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _portable_text(value, "storage entry name", max_bytes=BROWSER_PROFILE_MAX_NAME_BYTES)

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return _portable_text(
            value,
            "storage entry value",
            max_bytes=BROWSER_PROFILE_MAX_VALUE_BYTES,
        )


class BrowserProfileOriginStorage(_PrivateProfileStateModel):
    origin: StrictStr
    local_storage: tuple[BrowserProfileStorageEntry, ...]

    @field_validator("origin")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        return _canonical_origin(value)

    @field_validator("local_storage", mode="before")
    @classmethod
    def copy_entries(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("local_storage must be an ordered collection.")
        if len(value) > BROWSER_PROFILE_MAX_STORAGE_ENTRIES:
            raise ValueError("local_storage exceeds the browser-profile entry bound.")
        return tuple(BrowserProfileStorageEntry.model_validate(item) for item in value)

    @model_validator(mode="after")
    def validate_entry_order(self) -> Self:
        names = tuple(item.name for item in self.local_storage)
        if names != tuple(sorted(set(names))):
            raise ValueError("Origin storage entry names must be unique and sorted.")
        return self


class BrowserProfileCookie(_PrivateProfileStateModel):
    name: StrictStr
    value: StrictStr
    domain: StrictStr
    path: StrictStr = "/"
    expires: StrictFloat = -1.0
    http_only: StrictBool = False
    secure: StrictBool = True
    same_site: Literal["Strict", "Lax", "None"] = "Lax"

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = _portable_text(value, "cookie name", max_bytes=BROWSER_PROFILE_MAX_NAME_BYTES)
        if not value or any(character in value for character in "=;\r\n\t "):
            raise ValueError("Cookie name is not portable.")
        return value

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        value = _portable_text(value, "cookie value", max_bytes=BROWSER_PROFILE_MAX_VALUE_BYTES)
        if any(character in value for character in ";\r\n"):
            raise ValueError("Cookie value is not portable.")
        return value

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, value: str) -> str:
        value = _portable_text(value, "cookie domain", max_bytes=253)
        if value.startswith("."):
            raise ValueError("Domain cookies are not supported by browser-profile schema v1.")
        return normalize_egress_hostname(value, field_name="cookie domain")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        value = _portable_text(value, "cookie path", max_bytes=2_048)
        if not value.startswith("/") or "\\" in value or "\r" in value or "\n" in value:
            raise ValueError("Cookie path is not portable.")
        return value

    @field_validator("expires")
    @classmethod
    def validate_expires(cls, value: float) -> float:
        if not math.isfinite(value) or (value != -1.0 and value < 0):
            raise ValueError("Cookie expiry is invalid.")
        return value

    @model_validator(mode="after")
    def validate_security(self) -> Self:
        if not self.secure:
            raise ValueError("Browser-profile cookies must be Secure.")
        if self.same_site == "None" and not self.secure:
            raise ValueError("SameSite=None cookies must be Secure.")
        return self


class BrowserProfileStateV1(_PrivateProfileStateModel):
    schema_version: Literal[1] = 1
    included_categories: tuple[StrictStr, ...] = _INCLUDED_CATEGORIES
    omitted_categories: tuple[StrictStr, ...] = _OMITTED_CATEGORIES
    cookies: tuple[BrowserProfileCookie, ...] = ()
    origins: tuple[BrowserProfileOriginStorage, ...] = ()

    @field_validator("included_categories", mode="before")
    @classmethod
    def validate_included(cls, value: object) -> tuple[str, ...]:
        copied = tuple(value) if isinstance(value, list | tuple) else ()
        if copied != _INCLUDED_CATEGORIES:
            raise ValueError("Browser-profile state categories are unsupported.")
        return _INCLUDED_CATEGORIES

    @field_validator("omitted_categories", mode="before")
    @classmethod
    def validate_omitted(cls, value: object) -> tuple[str, ...]:
        copied = tuple(value) if isinstance(value, list | tuple) else ()
        if copied != _OMITTED_CATEGORIES:
            raise ValueError("Browser-profile omitted categories are incomplete.")
        return _OMITTED_CATEGORIES

    @field_validator("cookies", mode="before")
    @classmethod
    def copy_cookies(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("cookies must be an ordered collection.")
        if len(value) > BROWSER_PROFILE_MAX_COOKIES:
            raise ValueError("cookies exceed the browser-profile cookie bound.")
        return tuple(BrowserProfileCookie.model_validate(item) for item in value)

    @field_validator("origins", mode="before")
    @classmethod
    def copy_origins(cls, value: object) -> object:
        if not isinstance(value, list | tuple):
            raise TypeError("origins must be an ordered collection.")
        if len(value) > BROWSER_PROFILE_MAX_ORIGINS:
            raise ValueError("origins exceed the browser-profile origin bound.")
        return tuple(BrowserProfileOriginStorage.model_validate(item) for item in value)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        cookie_keys = tuple((item.domain, item.path, item.name) for item in self.cookies)
        if cookie_keys != tuple(sorted(set(cookie_keys))):
            raise ValueError("Browser-profile cookies must be unique and sorted.")
        origins = tuple(item.origin for item in self.origins)
        if origins != tuple(sorted(set(origins))):
            raise ValueError("Browser-profile origins must be unique and sorted.")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_durable_json_bytes(
            self.model_dump(mode="json"),
            "browser profile state",
        )


def validate_browser_profile_state(
    state: BrowserProfileStateV1,
    *,
    limits: BrowserProfileLimits,
    current_policy: BrowserProfileDestinationPolicy,
) -> BrowserProfileStateV1:
    """Own and validate one plaintext state against exact configured authority."""

    owned = BrowserProfileStateV1.model_validate(state)
    if len(owned.origins) > limits.max_origins or len(owned.cookies) > limits.max_cookies:
        raise ValueError("Browser-profile state exceeds its category count bound.")
    storage_count = sum(len(item.local_storage) for item in owned.origins)
    if storage_count > limits.max_storage_entries:
        raise ValueError("Browser-profile storage exceeds its entry count bound.")
    admitted_origins = set(current_policy.origins)
    if any(item.origin not in admitted_origins for item in owned.origins):
        raise ValueError("Browser-profile origin storage exceeds current destination authority.")
    admitted_hosts = {_origin_host(origin) for origin in current_policy.origins}
    if any(item.domain not in admitted_hosts for item in owned.cookies):
        raise ValueError("Browser-profile cookie scope exceeds current destination authority.")
    for cookie in owned.cookies:
        if len(cookie.name.encode("utf-8")) > limits.max_name_bytes:
            raise ValueError("Browser-profile cookie name exceeds its configured bound.")
        if len(cookie.value.encode("utf-8")) > limits.max_value_bytes:
            raise ValueError("Browser-profile cookie value exceeds its configured bound.")
    for origin in owned.origins:
        for entry in origin.local_storage:
            if len(entry.name.encode("utf-8")) > limits.max_name_bytes:
                raise ValueError("Browser-profile storage name exceeds its configured bound.")
            if len(entry.value.encode("utf-8")) > limits.max_value_bytes:
                raise ValueError("Browser-profile storage value exceeds its configured bound.")
    if len(owned.canonical_bytes()) > limits.max_plaintext_bytes:
        raise ValueError("Browser-profile plaintext exceeds its configured bound.")
    return owned


def browser_profile_state_from_playwright(
    value: object,
    *,
    limits: BrowserProfileLimits,
    current_policy: BrowserProfileDestinationPolicy,
) -> BrowserProfileStateV1:
    """Validate Playwright's portable storage-state subset without retaining extras."""

    if type(value) is not dict or set(value) != {"cookies", "origins"}:
        raise ValueError("Browser worker returned unsupported profile-state fields.")
    owned_value = cast("dict[str, object]", value)
    raw_cookies = owned_value.get("cookies")
    raw_origins = owned_value.get("origins")
    if type(raw_cookies) is not list or type(raw_origins) is not list:
        raise ValueError("Browser worker returned malformed profile-state categories.")
    cookies: list[BrowserProfileCookie] = []
    for item in raw_cookies:
        expected = {
            "name",
            "value",
            "domain",
            "path",
            "expires",
            "httpOnly",
            "secure",
            "sameSite",
        }
        if type(item) is not dict or set(item) != expected:
            raise ValueError("Browser worker returned an unsupported cookie shape.")
        owned_item = cast("dict[str, Any]", item)
        expires = owned_item["expires"]
        if type(expires) not in {int, float}:
            raise ValueError("Browser worker returned an invalid cookie expiry.")
        cookies.append(
            BrowserProfileCookie(
                name=owned_item["name"],
                value=owned_item["value"],
                domain=owned_item["domain"],
                path=owned_item["path"],
                expires=float(expires),
                http_only=owned_item["httpOnly"],
                secure=owned_item["secure"],
                same_site=owned_item["sameSite"],
            )
        )
    origins: list[BrowserProfileOriginStorage] = []
    for item in raw_origins:
        if type(item) is not dict or set(item) != {"origin", "localStorage"}:
            raise ValueError("Browser worker returned an unsupported origin-storage shape.")
        owned_item = cast("dict[str, Any]", item)
        entries = owned_item["localStorage"]
        if type(entries) is not list:
            raise ValueError("Browser worker returned malformed origin storage.")
        copied_entries: list[BrowserProfileStorageEntry] = []
        for entry in entries:
            if type(entry) is not dict or set(entry) != {"name", "value"}:
                raise ValueError("Browser worker returned unsupported storage-entry fields.")
            owned_entry = cast("dict[str, Any]", entry)
            copied_entries.append(
                BrowserProfileStorageEntry(
                    name=owned_entry["name"],
                    value=owned_entry["value"],
                )
            )
        origins.append(
            BrowserProfileOriginStorage(
                origin=owned_item["origin"],
                local_storage=tuple(sorted(copied_entries, key=lambda entry: entry.name)),
            )
        )
    state = BrowserProfileStateV1(
        cookies=tuple(sorted(cookies, key=lambda item: (item.domain, item.path, item.name))),
        origins=tuple(sorted(origins, key=lambda item: item.origin)),
    )
    return validate_browser_profile_state(
        state,
        limits=limits,
        current_policy=current_policy,
    )


def browser_profile_state_to_playwright(state: BrowserProfileStateV1) -> dict[str, object]:
    """Return only the Playwright storage-state fields admitted by schema v1."""

    return {
        "cookies": [
            {
                "name": item.name,
                "value": item.value,
                "domain": item.domain,
                "path": item.path,
                "expires": item.expires,
                "httpOnly": item.http_only,
                "secure": item.secure,
                "sameSite": item.same_site,
            }
            for item in state.cookies
        ],
        "origins": [
            {
                "origin": item.origin,
                "localStorage": [
                    {"name": entry.name, "value": entry.value} for entry in item.local_storage
                ],
            }
            for item in state.origins
        ],
    }


class BrowserProfileAuthority(_ProfileModel):
    schema_version: Literal[1] = 1
    profile_id: StrictStr
    scope: BrowserProfileScope
    destination_policy: BrowserProfileDestinationPolicy
    browser_protocol: StrictStr
    browser_worker_version: StrictStr
    state_schema_version: Literal[1] = 1
    key_authority_id: StrictStr
    store_id: StrictStr
    created_at: datetime
    expires_at: datetime | None = None
    fingerprint: StrictStr

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _identifier(value, "profile_id")

    @field_validator(
        "browser_protocol",
        "browser_worker_version",
        "key_authority_id",
        "store_id",
    )
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("created_at", "expires_at")
    @classmethod
    def validate_time(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, info.field_name)

    @field_validator("fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _digest(value, "fingerprint")

    def identity_material(self) -> dict[str, object]:
        return self.model_dump(mode="json", exclude={"fingerprint"})

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("Browser-profile expiry must follow creation.")
        expected = _fingerprint(
            b"cayu.browser-profile.authority.v1",
            self.identity_material(),
            "browser profile authority",
        )
        if self.fingerprint != expected:
            raise ValueError("Browser-profile authority fingerprint is invalid.")
        return self

    @classmethod
    def build(
        cls,
        *,
        scope: BrowserProfileScope,
        destination_policy: BrowserProfileDestinationPolicy,
        browser_protocol: str,
        browser_worker_version: str,
        key_authority_id: str,
        store_id: str,
        profile_id: str | None = None,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
    ) -> BrowserProfileAuthority:
        values: dict[str, object] = {
            "schema_version": 1,
            "profile_id": (f"bprof_{secrets.token_hex(16)}" if profile_id is None else profile_id),
            "scope": scope,
            "destination_policy": destination_policy,
            "browser_protocol": browser_protocol,
            "browser_worker_version": browser_worker_version,
            "state_schema_version": 1,
            "key_authority_id": key_authority_id,
            "store_id": store_id,
            "created_at": created_at or _now(),
            "expires_at": expires_at,
        }
        draft_values: dict[str, Any] = {**values, "fingerprint": "0" * 64}
        draft = cls.model_construct(**draft_values)
        return cls(
            **values,
            fingerprint=_fingerprint(
                b"cayu.browser-profile.authority.v1",
                draft.identity_material(),
                "browser profile authority",
            ),
        )


class BrowserProfileRef(_ProfileModel):
    profile_id: StrictStr
    authority_fingerprint: StrictStr
    generation: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    content_fingerprint: StrictStr

    @field_validator("profile_id")
    @classmethod
    def validate_profile_id(cls, value: str) -> str:
        return _identifier(value, "profile_id")

    @field_validator("authority_fingerprint", "content_fingerprint")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)


class BrowserProfileAccess(_ProfileModel):
    profile_id: StrictStr
    authority_fingerprint: StrictStr
    owner_fingerprint: StrictStr
    sharing_fingerprint: StrictStr
    store_id: StrictStr

    @field_validator("profile_id", "store_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("authority_fingerprint", "owner_fingerprint", "sharing_fingerprint")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @classmethod
    def from_authority(cls, authority: BrowserProfileAuthority) -> BrowserProfileAccess:
        return cls(
            profile_id=authority.profile_id,
            authority_fingerprint=authority.fingerprint,
            owner_fingerprint=authority.scope.owner_fingerprint,
            sharing_fingerprint=authority.scope.sharing_fingerprint,
            store_id=authority.store_id,
        )


class BrowserProfileEncryptedEnvelope(_ProfileModel):
    schema_version: Literal[1] = 1
    authority_fingerprint: StrictStr
    generation: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    encryption_algorithm: Literal["AES-256-GCM"] = BROWSER_PROFILE_ENCRYPTION_ALGORITHM
    key_authority_id: StrictStr
    store_id: StrictStr
    plaintext_bytes: StrictInt = Field(ge=1, le=BROWSER_PROFILE_MAX_PLAINTEXT_BYTES)
    ciphertext_bytes: StrictInt = Field(ge=17, le=BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES)
    content_fingerprint: StrictStr
    nonce_base64: StrictStr = Field(repr=False)
    ciphertext_base64: StrictStr = Field(repr=False)

    @field_validator("authority_fingerprint", "content_fingerprint")
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @field_validator("key_authority_id", "store_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @model_validator(mode="after")
    def validate_encoded_payload(self) -> Self:
        try:
            nonce = base64.b64decode(self.nonce_base64, validate=True)
            ciphertext = base64.b64decode(self.ciphertext_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Encrypted browser-profile payload is malformed.") from exc
        if len(nonce) != BROWSER_PROFILE_NONCE_BYTES:
            raise ValueError("Encrypted browser-profile nonce has the wrong size.")
        if len(ciphertext) != self.ciphertext_bytes:
            raise ValueError("Encrypted browser-profile ciphertext length is inconsistent.")
        if self.ciphertext_bytes != self.plaintext_bytes + 16:
            raise ValueError("Encrypted browser-profile length is inconsistent.")
        if sha256(nonce + ciphertext).hexdigest() != self.content_fingerprint:
            raise ValueError("Encrypted browser-profile content fingerprint is invalid.")
        return self

    def nonce(self) -> bytes:
        return base64.b64decode(self.nonce_base64, validate=True)

    def ciphertext(self) -> bytes:
        return base64.b64decode(self.ciphertext_base64, validate=True)

    def aad_material(self, authority: BrowserProfileAuthority) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "encryption_algorithm": self.encryption_algorithm,
            "profile_id": authority.profile_id,
            "authority_fingerprint": self.authority_fingerprint,
            "owner_fingerprint": authority.scope.owner_fingerprint,
            "sharing_fingerprint": authority.scope.sharing_fingerprint,
            "destination_policy_fingerprint": authority.destination_policy.fingerprint,
            "browser_protocol": authority.browser_protocol,
            "browser_worker_version": authority.browser_worker_version,
            "state_schema_version": authority.state_schema_version,
            "generation": self.generation,
            "key_authority_id": self.key_authority_id,
            "store_id": self.store_id,
            "plaintext_bytes": self.plaintext_bytes,
            "ciphertext_bytes": self.ciphertext_bytes,
        }


class BrowserProfileKeyAuthority(ABC):
    """Application-owned encryption boundary; stores never receive key material."""

    @property
    @abstractmethod
    def authority_id(self) -> str:
        pass

    @abstractmethod
    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        pass

    @abstractmethod
    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        pass


class AESGCMBrowserProfileKeyAuthority(BrowserProfileKeyAuthority):
    """Local AES-256-GCM key authority for application-managed key bytes."""

    __slots__ = ("_authority_id", "_key")

    def __init__(self, *, authority_id: str, key: bytes) -> None:
        self._authority_id = _identifier(authority_id, "authority_id")
        if type(key) is not bytes or len(key) != 32:
            raise ValueError("Browser-profile AES-GCM key must contain exactly 32 bytes.")
        self._key = bytes(key)

    @property
    def authority_id(self) -> str:
        return self._authority_id

    async def encrypt(self, *, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
        return AESGCM(self._key).encrypt(nonce, plaintext, aad)

    async def decrypt(self, *, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        return AESGCM(self._key).decrypt(nonce, ciphertext, aad)


class BrowserProfileStoreConflict(RuntimeError):
    """The requested profile mutation conflicts with durable authority."""


class BrowserProfileUnavailable(RuntimeError):
    """The profile cannot be safely restored or checkpointed."""


class _ValidatedStoreMutationResult:
    """Private proof that the store ran the exact validated mutation callback."""

    __slots__ = ("_authority", "_result", "_sealed")

    _authority: object
    _result: Any
    _sealed: bool

    def __init__(self, authority: object, result: Any) -> None:
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Validated browser-profile mutation result is immutable.")
        object.__setattr__(self, name, value)

    @property
    def authority(self) -> object:
        return self._authority

    @property
    def result(self) -> Any:
        return self._result


class _ValidatedStoreMutationFailure(Exception):
    """Carry one fixed runtime failure across an extension-owned store primitive."""

    __slots__ = ("authority", "failure_type", "message")

    def __init__(
        self,
        authority: object,
        failure_type: type[Exception],
        message: str,
    ) -> None:
        super().__init__("Validated browser-profile mutation failed.")
        self.authority = authority
        self.failure_type = failure_type
        self.message = message


class BrowserProfileWriterClaim(_ProfileModel):
    profile_id: StrictStr
    authority_fingerprint: StrictStr
    generation: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    content_fingerprint: StrictStr
    writer_id: StrictStr
    fence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    allocation_fingerprint: StrictStr
    execution_profile_fingerprint: StrictStr
    browser_session_id: StrictStr
    acquired_at: datetime
    expires_at: datetime

    @field_validator("profile_id", "writer_id", "browser_session_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "authority_fingerprint",
        "content_fingerprint",
        "allocation_fingerprint",
        "execution_profile_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @field_validator("acquired_at", "expires_at")
    @classmethod
    def validate_time(cls, value: datetime, info) -> datetime:
        return _utc(value, info.field_name)

    @model_validator(mode="after")
    def validate_expiry(self) -> Self:
        if self.expires_at <= self.acquired_at:
            raise ValueError("Browser-profile writer expiry must follow acquisition.")
        return self


class BrowserProfileRestoreRequest(_ProfileModel):
    operation_id: StrictStr
    access: BrowserProfileAccess
    expected_ref: BrowserProfileRef | None = None
    current_policy_fingerprint: StrictStr
    execution_profile_fingerprint: StrictStr
    allocation_fingerprint: StrictStr
    browser_session_id: StrictStr
    writer_id: StrictStr

    @field_validator("operation_id", "browser_session_id", "writer_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "current_policy_fingerprint",
        "execution_profile_fingerprint",
        "allocation_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    def fingerprint(self) -> str:
        return _fingerprint(
            b"cayu.browser-profile.restore-request.v1",
            self.model_dump(mode="json"),
            "browser profile restore request",
        )

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if self.expected_ref is not None and (
            self.expected_ref.profile_id != self.access.profile_id
            or self.expected_ref.authority_fingerprint != self.access.authority_fingerprint
        ):
            raise ValueError("Expected browser-profile reference has conflicting authority.")
        return self


class BrowserProfileRestoreReceipt(_ProfileModel):
    receipt_id: StrictStr
    operation_id: StrictStr
    request_fingerprint: StrictStr
    profile_ref: BrowserProfileRef
    current_policy_fingerprint: StrictStr
    execution_profile_fingerprint: StrictStr
    allocation_fingerprint: StrictStr
    browser_session_id: StrictStr
    writer_fence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    outcome: BrowserProfileTerminalOutcome
    error_code: StrictStr | None = None
    settled_at: datetime

    @field_validator("receipt_id", "operation_id", "browser_session_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "request_fingerprint",
        "current_policy_fingerprint",
        "execution_profile_fingerprint",
        "allocation_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _fixed_error_code(value, "error_code", allowed=_RESTORE_ERROR_CODES)
        )

    @field_validator("settled_at")
    @classmethod
    def validate_settled_at(cls, value: datetime) -> datetime:
        return _utc(value, "settled_at")

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if self.receipt_id != _receipt_id(
            "restore",
            self.operation_id,
            self.request_fingerprint,
        ):
            raise ValueError("Browser-profile restore receipt identity is invalid.")
        if (self.outcome is BrowserProfileTerminalOutcome.SUCCEEDED) == (
            self.error_code is not None
        ):
            raise ValueError("Browser-profile restore outcome contradicts its error code.")
        return self


class BrowserProfileRestorePreparation(_ProfileModel):
    request: BrowserProfileRestoreRequest
    request_fingerprint: StrictStr
    profile_ref: BrowserProfileRef
    writer_claim: BrowserProfileWriterClaim
    envelope: BrowserProfileEncryptedEnvelope | None = Field(default=None, repr=False)
    existing_receipt: BrowserProfileRestoreReceipt | None = None

    @field_validator("request_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _digest(value, "request_fingerprint")

    @model_validator(mode="after")
    def validate_preparation(self) -> Self:
        if self.request_fingerprint != self.request.fingerprint():
            raise ValueError("Restore preparation request fingerprint is invalid.")
        if (
            self.profile_ref.profile_id != self.request.access.profile_id
            or self.profile_ref.authority_fingerprint != self.request.access.authority_fingerprint
            or self.writer_claim.profile_id != self.profile_ref.profile_id
            or self.writer_claim.authority_fingerprint != self.profile_ref.authority_fingerprint
            or self.writer_claim.generation != self.profile_ref.generation
            or self.writer_claim.content_fingerprint != self.profile_ref.content_fingerprint
            or self.writer_claim.writer_id != self.request.writer_id
            or self.writer_claim.allocation_fingerprint != self.request.allocation_fingerprint
            or self.writer_claim.execution_profile_fingerprint
            != self.request.execution_profile_fingerprint
            or self.writer_claim.browser_session_id != self.request.browser_session_id
        ):
            raise ValueError("Restore preparation has conflicting profile authority.")
        if self.profile_ref.generation == 0 and self.envelope is not None:
            raise ValueError("Empty profile generations cannot carry ciphertext.")
        if self.profile_ref.generation > 0 and self.envelope is None:
            raise ValueError("Non-empty profile generations require ciphertext.")
        receipt = self.existing_receipt
        if receipt is not None and (
            receipt.operation_id != self.request.operation_id
            or receipt.request_fingerprint != self.request_fingerprint
            or receipt.profile_ref.profile_id != self.profile_ref.profile_id
            or receipt.profile_ref.authority_fingerprint != self.profile_ref.authority_fingerprint
            or receipt.current_policy_fingerprint != self.request.current_policy_fingerprint
            or receipt.execution_profile_fingerprint != self.request.execution_profile_fingerprint
            or receipt.allocation_fingerprint != self.request.allocation_fingerprint
            or receipt.browser_session_id != self.request.browser_session_id
            or receipt.writer_fence != self.writer_claim.fence
        ):
            raise ValueError("Restore preparation receipt authority is inconsistent.")
        return self


class BrowserProfileCheckpointRequest(_ProfileModel):
    operation_id: StrictStr
    access: BrowserProfileAccess
    writer_claim: BrowserProfileWriterClaim
    source_revision: StrictStr
    source_operation_receipt_id: StrictStr
    source_operation_fingerprint: StrictStr
    current_policy_fingerprint: StrictStr
    ambiguous_lineage: StrictBool = False

    @field_validator("operation_id", "source_revision", "source_operation_receipt_id")
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator("source_operation_fingerprint", "current_policy_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    def fingerprint(self) -> str:
        return _fingerprint(
            b"cayu.browser-profile.checkpoint-request.v1",
            self.model_dump(mode="json"),
            "browser profile checkpoint request",
        )

    @model_validator(mode="after")
    def validate_authority(self) -> Self:
        if (
            self.writer_claim.profile_id != self.access.profile_id
            or self.writer_claim.authority_fingerprint != self.access.authority_fingerprint
        ):
            raise ValueError("Checkpoint writer has conflicting profile authority.")
        return self


class BrowserProfileCheckpointReservation(_ProfileModel):
    reservation_id: StrictStr
    request: BrowserProfileCheckpointRequest
    request_fingerprint: StrictStr
    reserved_ciphertext_bytes: StrictInt = Field(
        ge=17,
        le=BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES,
    )
    previous_origin_count: StrictInt = Field(ge=0, le=BROWSER_PROFILE_MAX_ORIGINS)
    previous_cookie_count: StrictInt = Field(ge=0, le=BROWSER_PROFILE_MAX_COOKIES)
    previous_storage_entry_count: StrictInt = Field(
        ge=0,
        le=BROWSER_PROFILE_MAX_STORAGE_ENTRIES,
    )
    reserved_at: datetime

    @field_validator("reservation_id")
    @classmethod
    def validate_reservation_id(cls, value: str) -> str:
        return _identifier(value, "reservation_id")

    @field_validator("request_fingerprint")
    @classmethod
    def validate_fingerprint(cls, value: str) -> str:
        return _digest(value, "request_fingerprint")

    @field_validator("reserved_at")
    @classmethod
    def validate_reserved_at(cls, value: datetime) -> datetime:
        return _utc(value, "reserved_at")

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.request_fingerprint != self.request.fingerprint():
            raise ValueError("Checkpoint reservation request fingerprint is invalid.")
        if self.reservation_id != _receipt_id(
            "reserve",
            self.request.operation_id,
            self.request_fingerprint,
        ):
            raise ValueError("Checkpoint reservation identity is invalid.")
        return self


class BrowserProfileCheckpointReceipt(_ProfileModel):
    receipt_id: StrictStr
    operation_id: StrictStr
    request_fingerprint: StrictStr
    previous_ref: BrowserProfileRef
    published_ref: BrowserProfileRef
    current_policy_fingerprint: StrictStr
    execution_profile_fingerprint: StrictStr
    allocation_fingerprint: StrictStr
    browser_session_id: StrictStr
    writer_fence: StrictInt = Field(ge=1, le=MAX_DURABLE_JSON_INTEGER)
    source_revision: StrictStr
    source_operation_receipt_id: StrictStr
    source_operation_fingerprint: StrictStr
    ambiguous_lineage: StrictBool
    origin_count: StrictInt = Field(ge=0, le=BROWSER_PROFILE_MAX_ORIGINS)
    cookie_count: StrictInt = Field(ge=0, le=BROWSER_PROFILE_MAX_COOKIES)
    storage_entry_count: StrictInt = Field(
        ge=0,
        le=BROWSER_PROFILE_MAX_STORAGE_ENTRIES,
    )
    outcome: BrowserProfileTerminalOutcome
    error_code: StrictStr | None = None
    settled_at: datetime

    @field_validator(
        "receipt_id",
        "operation_id",
        "browser_session_id",
        "source_revision",
        "source_operation_receipt_id",
    )
    @classmethod
    def validate_identifier(cls, value: str, info) -> str:
        return _identifier(value, info.field_name)

    @field_validator(
        "request_fingerprint",
        "current_policy_fingerprint",
        "execution_profile_fingerprint",
        "allocation_fingerprint",
        "source_operation_fingerprint",
    )
    @classmethod
    def validate_digest(cls, value: str, info) -> str:
        return _digest(value, info.field_name)

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _fixed_error_code(value, "error_code", allowed=_CHECKPOINT_ERROR_CODES)
        )

    @field_validator("settled_at")
    @classmethod
    def validate_settled_at(cls, value: datetime) -> datetime:
        return _utc(value, "settled_at")

    @model_validator(mode="after")
    def validate_transition(self) -> Self:
        if self.receipt_id != _receipt_id(
            "checkpoint",
            self.operation_id,
            self.request_fingerprint,
        ):
            raise ValueError("Browser-profile checkpoint receipt identity is invalid.")
        if self.previous_ref.profile_id != self.published_ref.profile_id:
            raise ValueError("Checkpoint receipt changes browser-profile identity.")
        if self.previous_ref.authority_fingerprint != self.published_ref.authority_fingerprint:
            raise ValueError("Checkpoint receipt changes browser-profile authority.")
        succeeded = self.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
        if succeeded and self.published_ref.generation != self.previous_ref.generation + 1:
            raise ValueError("Successful checkpoint receipt generation is not consecutive.")
        if not succeeded and self.published_ref != self.previous_ref:
            raise ValueError("Failed checkpoint receipts cannot publish a generation.")
        if succeeded == (self.error_code is not None):
            raise ValueError("Browser-profile checkpoint outcome contradicts its error code.")
        return self


class BrowserProfileInspection(_ProfileModel):
    profile_id: StrictStr
    authority_fingerprint: StrictStr
    owner_fingerprint: StrictStr
    sharing_fingerprint: StrictStr
    destination_policy_fingerprint: StrictStr
    state_schema_version: StrictInt
    browser_protocol: StrictStr
    browser_worker_version: StrictStr
    key_authority_id: StrictStr
    store_id: StrictStr
    generation: StrictInt
    content_fingerprint: StrictStr
    origin_count: StrictInt
    cookie_count: StrictInt
    storage_entry_count: StrictInt
    plaintext_bytes: StrictInt
    ciphertext_bytes: StrictInt
    created_at: datetime
    checkpointed_at: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None
    active_writer: StrictBool
    active_allocation_fingerprint: StrictStr | None
    active_writer_expires_at: datetime | None
    status: BrowserProfileStatus
    last_restore_receipt_id: StrictStr | None
    last_checkpoint_receipt_id: StrictStr | None
    safe_error_code: StrictStr | None

    @field_validator("safe_error_code")
    @classmethod
    def validate_safe_error_code(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _fixed_error_code(
                value,
                "safe_error_code",
                allowed=_INSPECTION_ERROR_CODES,
            )
        )


class _StoredRestoreOperation(_ProfileModel):
    request: BrowserProfileRestoreRequest
    request_fingerprint: StrictStr
    claim: BrowserProfileWriterClaim
    receipt: BrowserProfileRestoreReceipt | None = None


class _StoredCheckpointOperation(_ProfileModel):
    reservation: BrowserProfileCheckpointReservation
    envelope: BrowserProfileEncryptedEnvelope | None = Field(default=None, repr=False)
    receipt: BrowserProfileCheckpointReceipt | None = None
    published: StrictBool = False


class _StoredBrowserProfile(_ProfileModel):
    revision: StrictInt = Field(ge=0, le=MAX_DURABLE_JSON_INTEGER)
    authority: BrowserProfileAuthority
    current_envelope: BrowserProfileEncryptedEnvelope | None = Field(default=None, repr=False)
    writer: BrowserProfileWriterClaim | None = None
    next_fence: StrictInt = Field(default=1, ge=1, le=MAX_DURABLE_JSON_INTEGER)
    revoked_at: datetime | None = None
    checkpointed_at: datetime | None = None
    restore_operations: dict[StrictStr, _StoredRestoreOperation] = Field(default_factory=dict)
    checkpoint_operations: dict[StrictStr, _StoredCheckpointOperation] = Field(default_factory=dict)
    last_restore_receipt_id: StrictStr | None = None
    last_checkpoint_receipt_id: StrictStr | None = None
    safe_error_code: StrictStr | None = None
    origin_count: StrictInt = Field(default=0, ge=0, le=BROWSER_PROFILE_MAX_ORIGINS)
    cookie_count: StrictInt = Field(default=0, ge=0, le=BROWSER_PROFILE_MAX_COOKIES)
    storage_entry_count: StrictInt = Field(
        default=0,
        ge=0,
        le=BROWSER_PROFILE_MAX_STORAGE_ENTRIES,
    )

    @field_validator("safe_error_code")
    @classmethod
    def validate_safe_error_code(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _fixed_error_code(
                value,
                "safe_error_code",
                allowed=_INSPECTION_ERROR_CODES,
            )
        )

    @field_validator("revoked_at", "checkpointed_at")
    @classmethod
    def validate_optional_time(cls, value: datetime | None, info) -> datetime | None:
        return None if value is None else _utc(value, info.field_name)

    @field_validator("last_restore_receipt_id", "last_checkpoint_receipt_id")
    @classmethod
    def validate_optional_receipt_id(cls, value: str | None, info) -> str | None:
        return None if value is None else _identifier(value, info.field_name)

    @model_validator(mode="after")
    def validate_record_authority(self) -> Self:
        authority = self.authority
        expected_access = BrowserProfileAccess.from_authority(authority)
        envelope = self.current_envelope
        if envelope is not None and (
            envelope.authority_fingerprint != authority.fingerprint
            or envelope.key_authority_id != authority.key_authority_id
            or envelope.store_id != authority.store_id
        ):
            raise ValueError("Stored browser-profile envelope authority is inconsistent.")
        if self.next_fence <= (0 if self.writer is None else self.writer.fence):
            raise ValueError("Stored browser-profile writer fence is inconsistent.")
        if self.writer is not None and (
            self.writer.profile_id != authority.profile_id
            or self.writer.authority_fingerprint != authority.fingerprint
            or self.writer.generation != _profile_ref(self).generation
            or self.writer.content_fingerprint != _profile_ref(self).content_fingerprint
        ):
            raise ValueError("Stored browser-profile writer authority is inconsistent.")
        if (
            len(self.restore_operations) > BROWSER_PROFILE_MAX_RECEIPTS
            or len(self.checkpoint_operations) > BROWSER_PROFILE_MAX_RECEIPTS
        ):
            raise ValueError("Stored browser-profile operation history exceeds its bound.")
        restore_claims: dict[int, BrowserProfileWriterClaim] = {}
        restore_receipt_ids: set[str] = set()
        for operation_id, operation in self.restore_operations.items():
            if operation_id != operation.request.operation_id:
                raise ValueError("Stored browser-profile restore key is inconsistent.")
            if operation.request_fingerprint != operation.request.fingerprint():
                raise ValueError("Stored browser-profile restore fingerprint is inconsistent.")
            if (
                operation.request.access != expected_access
                or operation.claim.profile_id != authority.profile_id
                or operation.claim.authority_fingerprint != authority.fingerprint
                or operation.claim.writer_id != operation.request.writer_id
                or operation.claim.allocation_fingerprint
                != operation.request.allocation_fingerprint
                or operation.claim.execution_profile_fingerprint
                != operation.request.execution_profile_fingerprint
                or operation.claim.browser_session_id != operation.request.browser_session_id
            ):
                raise ValueError("Stored browser-profile restore authority is inconsistent.")
            if operation.claim.fence >= self.next_fence:
                raise ValueError("Stored browser-profile restore fence is inconsistent.")
            if operation.claim.fence in restore_claims:
                raise ValueError("Stored browser-profile restore fence is duplicated.")
            restore_claims[operation.claim.fence] = operation.claim
            receipt = operation.receipt
            if receipt is not None and (
                receipt.receipt_id
                != _receipt_id(
                    "restore",
                    operation.request.operation_id,
                    operation.request_fingerprint,
                )
                or receipt.operation_id != operation.request.operation_id
                or receipt.request_fingerprint != operation.request_fingerprint
                or receipt.profile_ref
                != BrowserProfileRef(
                    profile_id=operation.claim.profile_id,
                    authority_fingerprint=operation.claim.authority_fingerprint,
                    generation=operation.claim.generation,
                    content_fingerprint=operation.claim.content_fingerprint,
                )
                or receipt.current_policy_fingerprint
                != operation.request.current_policy_fingerprint
                or receipt.execution_profile_fingerprint
                != operation.request.execution_profile_fingerprint
                or receipt.allocation_fingerprint != operation.request.allocation_fingerprint
                or receipt.browser_session_id != operation.request.browser_session_id
                or receipt.writer_fence != operation.claim.fence
            ):
                raise ValueError("Stored browser-profile restore receipt is inconsistent.")
            if receipt is not None:
                if receipt.receipt_id in restore_receipt_ids:
                    raise ValueError("Stored browser-profile restore receipt is duplicated.")
                restore_receipt_ids.add(receipt.receipt_id)
        if (
            self.last_restore_receipt_id is not None
            and self.last_restore_receipt_id not in restore_receipt_ids
        ):
            raise ValueError("Stored browser-profile last restore receipt is inconsistent.")
        if self.writer is not None:
            restore_claim = restore_claims.get(self.writer.fence)
            if restore_claim is None or not _same_writer_lineage(
                self.writer,
                restore_claim,
            ):
                raise ValueError("Stored browser-profile writer has no restore authority.")
        checkpoint_receipt_ids: set[str] = set()
        current_publications: list[_StoredCheckpointOperation] = []
        for operation_id, operation in self.checkpoint_operations.items():
            request = operation.reservation.request
            if operation_id != request.operation_id:
                raise ValueError("Stored browser-profile checkpoint key is inconsistent.")
            if request.access != expected_access:
                raise ValueError("Stored browser-profile checkpoint access is inconsistent.")
            restore_claim = restore_claims.get(request.writer_claim.fence)
            if restore_claim is None or not _same_writer_lineage(
                request.writer_claim,
                restore_claim,
            ):
                raise ValueError(
                    "Stored browser-profile checkpoint writer authority is inconsistent."
                )
            if operation.envelope is not None and operation.receipt is None:
                raise ValueError("A staged browser-profile envelope requires its receipt.")
            if operation.receipt is not None and (
                operation.receipt.receipt_id
                != _receipt_id(
                    "checkpoint",
                    request.operation_id,
                    operation.reservation.request_fingerprint,
                )
                or operation.receipt.operation_id != request.operation_id
                or operation.receipt.request_fingerprint
                != operation.reservation.request_fingerprint
                or operation.receipt.writer_fence != request.writer_claim.fence
                or operation.receipt.previous_ref
                != BrowserProfileRef(
                    profile_id=request.writer_claim.profile_id,
                    authority_fingerprint=request.writer_claim.authority_fingerprint,
                    generation=request.writer_claim.generation,
                    content_fingerprint=request.writer_claim.content_fingerprint,
                )
                or operation.receipt.current_policy_fingerprint
                != request.current_policy_fingerprint
                or operation.receipt.execution_profile_fingerprint
                != request.writer_claim.execution_profile_fingerprint
                or operation.receipt.allocation_fingerprint
                != request.writer_claim.allocation_fingerprint
                or operation.receipt.browser_session_id != request.writer_claim.browser_session_id
                or operation.receipt.source_revision != request.source_revision
                or operation.receipt.source_operation_receipt_id
                != request.source_operation_receipt_id
                or operation.receipt.source_operation_fingerprint
                != request.source_operation_fingerprint
                or operation.receipt.ambiguous_lineage != request.ambiguous_lineage
            ):
                raise ValueError("Stored browser-profile checkpoint receipt is inconsistent.")
            if operation.envelope is not None and (
                operation.receipt is None
                or operation.receipt.outcome is not BrowserProfileTerminalOutcome.SUCCEEDED
                or operation.receipt.published_ref.generation != operation.envelope.generation
                or operation.receipt.published_ref.content_fingerprint
                != operation.envelope.content_fingerprint
            ):
                raise ValueError("Stored browser-profile staged envelope is inconsistent.")
            if operation.published:
                if (
                    operation.envelope is None
                    or operation.receipt is None
                    or operation.receipt.outcome is not BrowserProfileTerminalOutcome.SUCCEEDED
                    or envelope is None
                    or envelope.generation < operation.envelope.generation
                    or (
                        envelope.generation == operation.envelope.generation
                        and envelope != operation.envelope
                    )
                ):
                    raise ValueError("Stored browser-profile published checkpoint is inconsistent.")
                if operation.envelope == envelope:
                    current_publications.append(operation)
            elif operation.envelope is not None and operation.envelope == envelope:
                raise ValueError("Stored browser-profile unpublished checkpoint is current.")
            if operation.receipt is not None:
                if operation.receipt.receipt_id in checkpoint_receipt_ids:
                    raise ValueError("Stored browser-profile checkpoint receipt is duplicated.")
                checkpoint_receipt_ids.add(operation.receipt.receipt_id)
        if (
            self.last_checkpoint_receipt_id is not None
            and self.last_checkpoint_receipt_id not in checkpoint_receipt_ids
        ):
            raise ValueError("Stored browser-profile last checkpoint receipt is inconsistent.")
        if envelope is not None:
            if len(current_publications) != 1:
                raise ValueError(
                    "Stored browser-profile current checkpoint authority is inconsistent."
                )
            current_receipt = current_publications[0].receipt
            if current_receipt is None:  # pragma: no cover - published invariant above
                raise ValueError(
                    "Stored browser-profile current checkpoint authority is inconsistent."
                )
            if (
                self.checkpointed_at != current_receipt.settled_at
                or self.origin_count != current_receipt.origin_count
                or self.cookie_count != current_receipt.cookie_count
                or self.storage_entry_count != current_receipt.storage_entry_count
            ):
                raise ValueError(
                    "Stored browser-profile current checkpoint evidence is inconsistent."
                )
        elif self.checkpointed_at is not None or any(
            (self.origin_count, self.cookie_count, self.storage_entry_count)
        ):
            raise ValueError("Empty browser profile has checkpoint evidence.")
        return self


def _copy_model(value: object, model_type: type[_ModelT]) -> _ModelT:
    if type(value) is not model_type:
        raise TypeError(f"value must be an exact {model_type.__name__} instance.")
    return model_type.model_validate(value)


def _own_stored_profile_record(
    value: object,
    *,
    expected_profile_id: str | None = None,
) -> _StoredBrowserProfile:
    """Own one extension-supplied record without retaining malformed private data."""

    validation_failure: BaseException | None = None
    try:
        owned = _copy_model(value, _StoredBrowserProfile)
    except (TypeError, ValueError) as failure:
        validation_failure = failure
        owned = None
    value = None
    if validation_failure is not None:
        _clear_inactive_profile_failure_frames(validation_failure)
        validation_failure = None
        raise BrowserProfileUnavailable("Browser-profile record is corrupt.") from None
    if owned is None:  # pragma: no cover - paired validation result invariant
        raise BrowserProfileUnavailable("Browser-profile record is corrupt.")
    if expected_profile_id is not None and owned.authority.profile_id != expected_profile_id:
        owned = None
        raise BrowserProfileUnavailable("Browser-profile record is corrupt.")
    return owned


def _safe_store_primitive_failure(error: BaseException) -> BaseException:
    """Sanitize one failure emitted by an application-supplied store primitive."""

    _clear_inactive_profile_failure_frames(error)
    if _profile_failure_contains_process_control(error):
        return error
    if isinstance(error, asyncio.CancelledError):
        return error
    if isinstance(error, BrowserProfileStoreConflict):
        return BrowserProfileStoreConflict("Browser-profile store operation conflicts.")
    return BrowserProfileUnavailable("Browser-profile store operation failed.")


def _validated_model_update(
    value: _ModelT,
    update: Mapping[str, object],
) -> _ModelT:
    return type(value).model_validate(value.model_copy(update=dict(update)))


def _profile_ref(record: _StoredBrowserProfile) -> BrowserProfileRef:
    envelope = record.current_envelope
    return BrowserProfileRef(
        profile_id=record.authority.profile_id,
        authority_fingerprint=record.authority.fingerprint,
        generation=0 if envelope is None else envelope.generation,
        content_fingerprint=(
            _EMPTY_CONTENT_FINGERPRINT if envelope is None else envelope.content_fingerprint
        ),
    )


def _check_access(record: _StoredBrowserProfile, access: BrowserProfileAccess) -> None:
    authority = record.authority
    if (
        access.profile_id != authority.profile_id
        or access.authority_fingerprint != authority.fingerprint
        or access.owner_fingerprint != authority.scope.owner_fingerprint
        or access.sharing_fingerprint != authority.scope.sharing_fingerprint
        or access.store_id != authority.store_id
    ):
        raise BrowserProfileStoreConflict("Browser-profile access authority does not match.")


def _check_available(record: _StoredBrowserProfile, now: datetime) -> None:
    if record.revoked_at is not None:
        raise BrowserProfileUnavailable("Browser profile is revoked.")
    if record.authority.expires_at is not None and now >= record.authority.expires_at:
        raise BrowserProfileUnavailable("Browser profile is expired.")


def _same_claim(left: BrowserProfileWriterClaim, right: BrowserProfileWriterClaim) -> bool:
    return (
        left.profile_id == right.profile_id
        and left.authority_fingerprint == right.authority_fingerprint
        and left.writer_id == right.writer_id
        and left.fence == right.fence
        and left.generation == right.generation
        and left.content_fingerprint == right.content_fingerprint
        and left.allocation_fingerprint == right.allocation_fingerprint
        and left.execution_profile_fingerprint == right.execution_profile_fingerprint
        and left.browser_session_id == right.browser_session_id
    )


def _same_writer_lineage(
    left: BrowserProfileWriterClaim,
    right: BrowserProfileWriterClaim,
) -> bool:
    """Compare immutable writer authority across renewal and checkpoint generations."""

    return (
        left.profile_id == right.profile_id
        and left.authority_fingerprint == right.authority_fingerprint
        and left.writer_id == right.writer_id
        and left.fence == right.fence
        and left.allocation_fingerprint == right.allocation_fingerprint
        and left.execution_profile_fingerprint == right.execution_profile_fingerprint
        and left.browser_session_id == right.browser_session_id
        and left.acquired_at == right.acquired_at
    )


def _checkpoint_operation_is_settled(operation: _StoredCheckpointOperation) -> bool:
    receipt = operation.receipt
    return receipt is not None and (
        receipt.outcome is not BrowserProfileTerminalOutcome.SUCCEEDED or operation.published
    )


def _restore_receipt(
    operation: _StoredRestoreOperation,
    *,
    profile_ref: BrowserProfileRef,
    outcome: BrowserProfileTerminalOutcome,
    error_code: str | None,
    settled_at: datetime,
) -> BrowserProfileRestoreReceipt:
    request = operation.request
    return BrowserProfileRestoreReceipt(
        receipt_id=_receipt_id(
            "restore",
            request.operation_id,
            operation.request_fingerprint,
        ),
        operation_id=request.operation_id,
        request_fingerprint=operation.request_fingerprint,
        profile_ref=profile_ref,
        current_policy_fingerprint=request.current_policy_fingerprint,
        execution_profile_fingerprint=request.execution_profile_fingerprint,
        allocation_fingerprint=request.allocation_fingerprint,
        browser_session_id=request.browser_session_id,
        writer_fence=operation.claim.fence,
        outcome=outcome,
        error_code=error_code,
        settled_at=settled_at,
    )


def _checkpoint_failure_receipt(
    operation: _StoredCheckpointOperation,
    *,
    outcome: BrowserProfileTerminalOutcome,
    error_code: str,
    settled_at: datetime,
) -> BrowserProfileCheckpointReceipt:
    reservation = operation.reservation
    request = reservation.request
    previous = BrowserProfileRef(
        profile_id=request.writer_claim.profile_id,
        authority_fingerprint=request.writer_claim.authority_fingerprint,
        generation=request.writer_claim.generation,
        content_fingerprint=request.writer_claim.content_fingerprint,
    )
    return BrowserProfileCheckpointReceipt(
        receipt_id=_receipt_id(
            "checkpoint",
            request.operation_id,
            reservation.request_fingerprint,
        ),
        operation_id=request.operation_id,
        request_fingerprint=reservation.request_fingerprint,
        previous_ref=previous,
        published_ref=previous,
        current_policy_fingerprint=request.current_policy_fingerprint,
        execution_profile_fingerprint=request.writer_claim.execution_profile_fingerprint,
        allocation_fingerprint=request.writer_claim.allocation_fingerprint,
        browser_session_id=request.writer_claim.browser_session_id,
        writer_fence=request.writer_claim.fence,
        source_revision=request.source_revision,
        source_operation_receipt_id=request.source_operation_receipt_id,
        source_operation_fingerprint=request.source_operation_fingerprint,
        ambiguous_lineage=request.ambiguous_lineage,
        origin_count=reservation.previous_origin_count,
        cookie_count=reservation.previous_cookie_count,
        storage_entry_count=reservation.previous_storage_entry_count,
        outcome=outcome,
        error_code=error_code,
        settled_at=settled_at,
    )


def _settle_expired_writer(
    record: _StoredBrowserProfile,
    now: datetime,
) -> _StoredBrowserProfile:
    """Fence one abandoned allocation and retain exact unknown-outcome evidence."""

    writer = record.writer
    if writer is None or writer.expires_at > now:
        return record
    restore_operations = dict(record.restore_operations)
    checkpoint_operations = dict(record.checkpoint_operations)
    last_restore_receipt_id = record.last_restore_receipt_id
    last_checkpoint_receipt_id = record.last_checkpoint_receipt_id
    safe_error_code = record.safe_error_code
    current_ref = _profile_ref(record)
    for operation_id, operation in tuple(restore_operations.items()):
        if operation.claim.fence != writer.fence or operation.receipt is not None:
            continue
        receipt = _restore_receipt(
            operation,
            profile_ref=current_ref,
            outcome=BrowserProfileTerminalOutcome.OUTCOME_UNKNOWN,
            error_code="restore_outcome_unknown",
            settled_at=now,
        )
        restore_operations[operation_id] = _validated_model_update(
            operation,
            {"receipt": receipt},
        )
        last_restore_receipt_id = receipt.receipt_id
        safe_error_code = receipt.error_code
    for operation_id, operation in tuple(checkpoint_operations.items()):
        request = operation.reservation.request
        if request.writer_claim.fence != writer.fence:
            continue
        if operation.published:
            continue
        if (
            operation.receipt is not None
            and operation.receipt.outcome is not BrowserProfileTerminalOutcome.SUCCEEDED
        ):
            continue
        receipt = _checkpoint_failure_receipt(
            operation,
            outcome=BrowserProfileTerminalOutcome.OUTCOME_UNKNOWN,
            error_code="checkpoint_outcome_unknown",
            settled_at=now,
        )
        checkpoint_operations[operation_id] = _validated_model_update(
            operation,
            {"envelope": None, "receipt": receipt},
        )
        last_checkpoint_receipt_id = receipt.receipt_id
        safe_error_code = receipt.error_code
    return _validated_model_update(
        record,
        {
            "writer": None,
            "restore_operations": restore_operations,
            "checkpoint_operations": checkpoint_operations,
            "last_restore_receipt_id": last_restore_receipt_id,
            "last_checkpoint_receipt_id": last_checkpoint_receipt_id,
            "safe_error_code": safe_error_code,
        },
    )


class BrowserProfileStore(ABC):
    """Dedicated encrypted profile store with atomic store-timed mutation."""

    def __init__(
        self,
        *,
        store_id: str,
        clock: Callable[[], datetime] = _now,
        max_ciphertext_bytes: int = BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES,
    ) -> None:
        self._store_id = _identifier(store_id, "store_id")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        if (
            type(max_ciphertext_bytes) is not int
            or not 17 <= max_ciphertext_bytes <= BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES
        ):
            raise ValueError("max_ciphertext_bytes is outside the browser-profile bound.")
        self._max_ciphertext_bytes = max_ciphertext_bytes

    @property
    def id(self) -> str:
        return self._store_id

    def _time(self) -> datetime:
        return _utc(self._clock(), "store clock")

    @abstractmethod
    async def _create_record(
        self,
        record: _StoredBrowserProfile,
    ) -> _StoredBrowserProfile:
        pass

    @abstractmethod
    async def _load_record(self, profile_id: str) -> _StoredBrowserProfile | None:
        pass

    @abstractmethod
    async def _mutate_record(
        self,
        profile_id: str,
        operation: Callable[
            [_StoredBrowserProfile, datetime],
            tuple[_StoredBrowserProfile, Any],
        ],
    ) -> Any:
        """Apply one mutation with store-owned time inside its write boundary."""

        pass

    @abstractmethod
    async def _list_records(self) -> tuple[_StoredBrowserProfile, ...]:
        pass

    async def _mutate(
        self,
        profile_id: str,
        operation: Callable[[_StoredBrowserProfile, datetime], tuple[_StoredBrowserProfile, Any]],
    ) -> Any:
        owned_profile_id = _identifier(profile_id, "profile_id")
        mutation_authority = object()

        def validated_operation(
            record: _StoredBrowserProfile,
            observed_at: datetime,
        ) -> tuple[_StoredBrowserProfile, _ValidatedStoreMutationResult]:
            try:
                owned_record = _own_stored_profile_record(
                    record,
                    expected_profile_id=owned_profile_id,
                )
            finally:
                record = None  # ty: ignore[invalid-assignment]
            clock_failure: BaseException | None = None
            owned_time: datetime | None = None
            try:
                owned_time = _utc(observed_at, "store clock")
            except (TypeError, ValueError) as failure:
                clock_failure = failure
            observed_at = None  # ty: ignore[invalid-assignment]
            if clock_failure is not None:
                _clear_inactive_profile_failure_frames(clock_failure)
                clock_failure = None
                raise BrowserProfileUnavailable("Browser-profile record is corrupt.") from None
            if owned_time is None:  # pragma: no cover - paired validation invariant
                raise BrowserProfileUnavailable("Browser-profile record is corrupt.")
            try:
                updated, result = operation(owned_record, owned_time)
            except (BrowserProfileStoreConflict, BrowserProfileUnavailable) as failure:
                failure_type = type(failure)
                message = str(failure)
                _clear_inactive_profile_failure_frames(failure)
                failure = None
                if failure_type not in {
                    BrowserProfileStoreConflict,
                    BrowserProfileUnavailable,
                }:
                    raise BrowserProfileUnavailable(
                        "Browser-profile store operation failed."
                    ) from None
                raise _ValidatedStoreMutationFailure(
                    mutation_authority,
                    failure_type,
                    message,
                ) from None
            except ValueError as failure:
                # The only public ValueError raised from a validated mutation
                # is the fixed revocation-time contract. Validation-library
                # failures can retain complete private models and stay generic.
                message = str(failure)
                _clear_inactive_profile_failure_frames(failure)
                failure = None
                if message != "revoked_at cannot be in the future.":
                    raise BrowserProfileUnavailable("Browser-profile record is corrupt.") from None
                raise _ValidatedStoreMutationFailure(
                    mutation_authority,
                    ValueError,
                    message,
                ) from None
            try:
                owned_updated = _own_stored_profile_record(
                    updated,
                    expected_profile_id=owned_profile_id,
                )
            finally:
                updated = None
            return owned_updated, _ValidatedStoreMutationResult(
                mutation_authority,
                result,
            )

        primitive_failure: BaseException | None = None
        raw_result: object = None
        try:
            raw_result = await self._mutate_record(
                owned_profile_id,
                validated_operation,
            )
        except BaseException as failure:
            primitive_failure = failure
        if primitive_failure is not None:
            if (
                type(primitive_failure) is _ValidatedStoreMutationFailure
                and primitive_failure.authority is mutation_authority
                and primitive_failure.failure_type
                in {
                    BrowserProfileStoreConflict,
                    BrowserProfileUnavailable,
                    ValueError,
                }
            ):
                failure_type = primitive_failure.failure_type
                message = primitive_failure.message
                _clear_inactive_profile_failure_frames(primitive_failure)
                primitive_failure = None
                raise failure_type(message) from None
            safe_failure = _safe_store_primitive_failure(primitive_failure)
            primitive_failure = None
            raise safe_failure from None
        if (
            type(raw_result) is not _ValidatedStoreMutationResult
            or raw_result.authority is not mutation_authority
        ):
            raw_result = None
            raise BrowserProfileUnavailable("Browser-profile store operation failed.") from None
        result = raw_result.result
        raw_result = None
        return result

    async def _load_owned_record(self, profile_id: str) -> _StoredBrowserProfile | None:
        owned_profile_id = _identifier(profile_id, "profile_id")
        primitive_failure: BaseException | None = None
        record: _StoredBrowserProfile | None = None
        try:
            record = await self._load_record(owned_profile_id)
        except BaseException as failure:
            primitive_failure = failure
        if primitive_failure is not None:
            safe_failure = _safe_store_primitive_failure(primitive_failure)
            primitive_failure = None
            raise safe_failure from None
        if record is None:
            return None
        try:
            owned = _own_stored_profile_record(
                record,
                expected_profile_id=owned_profile_id,
            )
        finally:
            record = None
        return owned

    async def _list_owned_records(self) -> tuple[_StoredBrowserProfile, ...]:
        primitive_failure: BaseException | None = None
        records: tuple[_StoredBrowserProfile, ...] | object = ()
        try:
            records = await self._list_records()
        except BaseException as failure:
            primitive_failure = failure
        if primitive_failure is not None:
            safe_failure = _safe_store_primitive_failure(primitive_failure)
            primitive_failure = None
            raise safe_failure from None
        if type(records) is not tuple:
            records = ()
            raise BrowserProfileUnavailable("Browser-profile record is corrupt.") from None
        pending = list(reversed(records))
        records = ()
        owned_records: list[_StoredBrowserProfile] = []
        owned: tuple[_StoredBrowserProfile, ...] = ()
        try:
            while pending:
                record = pending.pop()
                try:
                    owned_records.append(_own_stored_profile_record(record))
                finally:
                    record = None
            owned = tuple(owned_records)
            if len({record.authority.profile_id for record in owned}) != len(owned):
                raise BrowserProfileUnavailable("Browser-profile record is corrupt.")
            return owned
        finally:
            pending.clear()
            owned_records.clear()
            owned = ()

    async def create_profile(self, authority: BrowserProfileAuthority) -> BrowserProfileRef:
        owned = _copy_model(authority, BrowserProfileAuthority)
        if owned.store_id != self.id:
            raise BrowserProfileStoreConflict("Browser-profile store identity does not match.")
        record = _StoredBrowserProfile(revision=0, authority=owned)
        primitive_failure: BaseException | None = None
        returned: _StoredBrowserProfile | None = None
        try:
            returned = await self._create_record(record)
        except BaseException as failure:
            primitive_failure = failure
        if primitive_failure is not None:
            safe_failure = _safe_store_primitive_failure(primitive_failure)
            primitive_failure = None
            raise safe_failure from None
        if returned is None:  # pragma: no cover - primitive return contract
            raise BrowserProfileUnavailable("Browser-profile record is corrupt.")
        try:
            stored = _own_stored_profile_record(
                returned,
                expected_profile_id=owned.profile_id,
            )
        finally:
            returned = None
        if stored.authority != owned:
            stored = None
            raise BrowserProfileUnavailable("Browser-profile record is corrupt.")
        return _profile_ref(stored)

    async def current_ref(self, access: BrowserProfileAccess) -> BrowserProfileRef:
        owned_access = _copy_model(access, BrowserProfileAccess)
        record = await self._load_owned_record(owned_access.profile_id)
        if record is None:
            raise BrowserProfileUnavailable("Browser profile does not exist.")
        _check_access(record, owned_access)
        return _profile_ref(record)

    async def resume_writer(
        self,
        access: BrowserProfileAccess,
        *,
        browser_session_id: str,
        execution_profile_fingerprint: str,
        allocation_fingerprint: str,
        lease_seconds: int,
    ) -> BrowserProfileRestorePreparation:
        """Atomically renew and reconstruct one exact live writer."""

        owned_access = _copy_model(access, BrowserProfileAccess)
        session_id = _identifier(browser_session_id, "browser_session_id")
        execution = _digest(
            execution_profile_fingerprint,
            "execution_profile_fingerprint",
        )
        allocation = _digest(allocation_fingerprint, "allocation_fingerprint")
        if (
            type(lease_seconds) is not int
            or not 1 <= lease_seconds <= BROWSER_PROFILE_MAX_LEASE_SECONDS
        ):
            raise ValueError("lease_seconds is outside the browser-profile bound.")

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileRestorePreparation]:
            _check_access(record, owned_access)
            _check_available(record, now)
            writer = record.writer
            if (
                writer is None
                or writer.expires_at <= now
                or writer.browser_session_id != session_id
                or writer.execution_profile_fingerprint != execution
                or writer.allocation_fingerprint != allocation
            ):
                raise BrowserProfileUnavailable("Browser-profile live writer is unavailable.")
            matching = tuple(
                restore_operation
                for restore_operation in record.restore_operations.values()
                if restore_operation.claim.fence == writer.fence
                and (
                    restore_operation.receipt is None
                    or restore_operation.receipt.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
                )
            )
            if len(matching) != 1:
                raise BrowserProfileStoreConflict(
                    "Browser-profile restore receipt authority is ambiguous."
                )
            restore_operation = matching[0]
            renewed = _validated_model_update(
                writer,
                {"expires_at": now + timedelta(seconds=lease_seconds)},
            )
            updated = _validated_model_update(record, {"writer": renewed})
            return updated, BrowserProfileRestorePreparation(
                request=restore_operation.request,
                request_fingerprint=restore_operation.request_fingerprint,
                profile_ref=_profile_ref(record),
                writer_claim=renewed,
                envelope=record.current_envelope,
                existing_receipt=restore_operation.receipt,
            )

        return await self._mutate(owned_access.profile_id, mutate)

    async def prepare_restore(
        self,
        request: BrowserProfileRestoreRequest,
        *,
        lease_seconds: int,
    ) -> BrowserProfileRestorePreparation:
        owned = _copy_model(request, BrowserProfileRestoreRequest)
        if (
            type(lease_seconds) is not int
            or not 1 <= lease_seconds <= BROWSER_PROFILE_MAX_LEASE_SECONDS
        ):
            raise ValueError("lease_seconds is outside the browser-profile bound.")
        request_fingerprint = owned.fingerprint()

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileRestorePreparation]:
            _check_access(record, owned.access)
            _check_available(record, now)
            record = _settle_expired_writer(record, now)
            current_ref = _profile_ref(record)
            if owned.expected_ref is not None and owned.expected_ref != current_ref:
                raise BrowserProfileStoreConflict("Browser-profile generation changed.")
            existing = record.restore_operations.get(owned.operation_id)
            if existing is not None:
                if existing.request_fingerprint != request_fingerprint or existing.request != owned:
                    raise BrowserProfileStoreConflict("Browser-profile restore identity conflicts.")
                if existing.receipt is None:
                    if (
                        record.writer is None
                        or not _same_claim(record.writer, existing.claim)
                        or existing.claim.expires_at <= now
                    ):
                        raise BrowserProfileUnavailable(
                            "Browser-profile restore outcome is unknown after lease expiry."
                        )
                elif existing.receipt.outcome is BrowserProfileTerminalOutcome.SUCCEEDED:
                    if (
                        record.writer is None
                        or not _same_claim(record.writer, existing.claim)
                        or existing.receipt.profile_ref != current_ref
                        or record.writer.expires_at <= now
                    ):
                        raise BrowserProfileUnavailable(
                            "Browser-profile restore is already settled."
                        )
                elif existing.receipt.profile_ref != current_ref:
                    raise BrowserProfileUnavailable("Browser-profile restore is already settled.")
                return record, BrowserProfileRestorePreparation(
                    request=owned,
                    request_fingerprint=request_fingerprint,
                    profile_ref=current_ref,
                    writer_claim=existing.claim,
                    envelope=record.current_envelope,
                    existing_receipt=existing.receipt,
                )
            writer = record.writer
            if writer is not None:
                raise BrowserProfileStoreConflict(
                    "Browser-profile generation already has an active writer."
                )
            if record.next_fence >= MAX_DURABLE_JSON_INTEGER:
                raise BrowserProfileUnavailable("Browser-profile writer fence is exhausted.")
            claim = BrowserProfileWriterClaim(
                profile_id=owned.access.profile_id,
                authority_fingerprint=owned.access.authority_fingerprint,
                generation=current_ref.generation,
                content_fingerprint=current_ref.content_fingerprint,
                writer_id=owned.writer_id,
                fence=record.next_fence,
                allocation_fingerprint=owned.allocation_fingerprint,
                execution_profile_fingerprint=owned.execution_profile_fingerprint,
                browser_session_id=owned.browser_session_id,
                acquired_at=now,
                expires_at=now + timedelta(seconds=lease_seconds),
            )
            operations = dict(record.restore_operations)
            operations[owned.operation_id] = _StoredRestoreOperation(
                request=owned,
                request_fingerprint=request_fingerprint,
                claim=claim,
            )
            operations, checkpoint_operations = _bounded_restore_history(
                record,
                operations,
            )
            updated = _validated_model_update(
                record,
                {
                    "writer": claim,
                    "next_fence": record.next_fence + 1,
                    "restore_operations": operations,
                    "checkpoint_operations": checkpoint_operations,
                    "safe_error_code": None,
                },
            )
            return updated, BrowserProfileRestorePreparation(
                request=owned,
                request_fingerprint=request_fingerprint,
                profile_ref=current_ref,
                writer_claim=claim,
                envelope=record.current_envelope,
            )

        return await self._mutate(owned.access.profile_id, mutate)

    async def complete_restore(
        self,
        request: BrowserProfileRestoreRequest,
        *,
        outcome: BrowserProfileTerminalOutcome,
        error_code: str | None = None,
    ) -> BrowserProfileRestoreReceipt:
        owned = _copy_model(request, BrowserProfileRestoreRequest)
        if type(outcome) is not BrowserProfileTerminalOutcome:
            raise TypeError("outcome must be a BrowserProfileTerminalOutcome.")
        safe_error = (
            None
            if error_code is None
            else _fixed_error_code(
                error_code,
                "error_code",
                allowed=_RESTORE_ERROR_CODES,
            )
        )
        if (outcome is BrowserProfileTerminalOutcome.SUCCEEDED) == (safe_error is not None):
            raise ValueError("Browser-profile restore outcome contradicts its error code.")
        request_fingerprint = owned.fingerprint()

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileRestoreReceipt]:
            _check_access(record, owned.access)
            existing = record.restore_operations.get(owned.operation_id)
            if (
                existing is None
                or existing.request != owned
                or existing.request_fingerprint != request_fingerprint
            ):
                raise BrowserProfileStoreConflict("Browser-profile restore was not prepared.")
            if existing.receipt is not None:
                if (
                    existing.receipt.outcome is not outcome
                    or existing.receipt.error_code != safe_error
                ):
                    raise BrowserProfileStoreConflict(
                        "Browser-profile restore is already settled differently."
                    )
                return record, existing.receipt
            current = _profile_ref(record)
            if record.writer is None or not _same_claim(record.writer, existing.claim):
                raise BrowserProfileStoreConflict("Browser-profile writer authority changed.")
            if record.writer.expires_at <= now:
                raise BrowserProfileStoreConflict("Browser-profile writer lease expired.")
            receipt = BrowserProfileRestoreReceipt(
                receipt_id=_receipt_id("restore", owned.operation_id, request_fingerprint),
                operation_id=owned.operation_id,
                request_fingerprint=request_fingerprint,
                profile_ref=current,
                current_policy_fingerprint=owned.current_policy_fingerprint,
                execution_profile_fingerprint=owned.execution_profile_fingerprint,
                allocation_fingerprint=owned.allocation_fingerprint,
                browser_session_id=owned.browser_session_id,
                writer_fence=existing.claim.fence,
                outcome=outcome,
                error_code=safe_error,
                settled_at=now,
            )
            operations = dict(record.restore_operations)
            operations[owned.operation_id] = _validated_model_update(
                existing,
                {"receipt": receipt},
            )
            release_writer = outcome is BrowserProfileTerminalOutcome.FAILED
            updated = _validated_model_update(
                record,
                {
                    "restore_operations": operations,
                    "writer": None if release_writer else record.writer,
                    "last_restore_receipt_id": receipt.receipt_id,
                    "safe_error_code": safe_error,
                },
            )
            return updated, receipt

        return await self._mutate(owned.access.profile_id, mutate)

    async def load_restore_receipt(
        self,
        access: BrowserProfileAccess,
        operation_id: str,
    ) -> BrowserProfileRestoreReceipt | None:
        owned_access = _copy_model(access, BrowserProfileAccess)
        owned_operation_id = _identifier(operation_id, "operation_id")
        record = await self._load_owned_record(owned_access.profile_id)
        if record is None:
            raise BrowserProfileUnavailable("Browser profile does not exist.")
        _check_access(record, owned_access)
        operation = record.restore_operations.get(owned_operation_id)
        return None if operation is None else operation.receipt

    async def renew_writer(
        self,
        access: BrowserProfileAccess,
        claim: BrowserProfileWriterClaim,
        *,
        lease_seconds: int,
    ) -> BrowserProfileWriterClaim:
        owned_access = _copy_model(access, BrowserProfileAccess)
        owned_claim = _copy_model(claim, BrowserProfileWriterClaim)
        if (
            type(lease_seconds) is not int
            or not 1 <= lease_seconds <= BROWSER_PROFILE_MAX_LEASE_SECONDS
        ):
            raise ValueError("lease_seconds is outside the browser-profile bound.")

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileWriterClaim]:
            _check_access(record, owned_access)
            _check_available(record, now)
            if record.writer is None or not _same_claim(record.writer, owned_claim):
                raise BrowserProfileStoreConflict("Browser-profile writer authority changed.")
            if record.writer.expires_at <= now:
                raise BrowserProfileStoreConflict("Browser-profile writer lease expired.")
            renewed = _validated_model_update(
                record.writer,
                {"expires_at": now + timedelta(seconds=lease_seconds)},
            )
            return _validated_model_update(record, {"writer": renewed}), renewed

        return await self._mutate(owned_access.profile_id, mutate)

    async def reserve_checkpoint(
        self,
        request: BrowserProfileCheckpointRequest,
        *,
        reserved_ciphertext_bytes: int,
    ) -> BrowserProfileCheckpointReservation:
        owned = _copy_model(request, BrowserProfileCheckpointRequest)
        if (
            type(reserved_ciphertext_bytes) is not int
            or not 17 <= reserved_ciphertext_bytes <= self._max_ciphertext_bytes
        ):
            raise BrowserProfileUnavailable("Browser-profile checkpoint capacity is unavailable.")
        request_fingerprint = owned.fingerprint()

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileCheckpointReservation]:
            _check_access(record, owned.access)
            _check_available(record, now)
            if record.writer is None or not _same_claim(record.writer, owned.writer_claim):
                raise BrowserProfileStoreConflict("Browser-profile writer authority changed.")
            if record.writer.expires_at <= now:
                raise BrowserProfileStoreConflict("Browser-profile writer lease expired.")
            if record.writer.generation >= MAX_DURABLE_JSON_INTEGER:
                raise BrowserProfileUnavailable(
                    "Browser-profile checkpoint generation is exhausted."
                )
            existing = record.checkpoint_operations.get(owned.operation_id)
            if existing is not None:
                reservation = existing.reservation
                if (
                    reservation.request_fingerprint != request_fingerprint
                    or reservation.request != owned
                    or reservation.reserved_ciphertext_bytes != reserved_ciphertext_bytes
                ):
                    raise BrowserProfileStoreConflict("Browser-profile checkpoint conflicts.")
                return record, reservation
            reservation = BrowserProfileCheckpointReservation(
                reservation_id=_receipt_id("reserve", owned.operation_id, request_fingerprint),
                request=owned,
                request_fingerprint=request_fingerprint,
                reserved_ciphertext_bytes=reserved_ciphertext_bytes,
                previous_origin_count=record.origin_count,
                previous_cookie_count=record.cookie_count,
                previous_storage_entry_count=record.storage_entry_count,
                reserved_at=now,
            )
            operations = dict(record.checkpoint_operations)
            operations[owned.operation_id] = _StoredCheckpointOperation(reservation=reservation)
            operations = _bounded_operations(
                operations,
                settled=_checkpoint_operation_is_settled,
                preserve=lambda operation: (
                    _checkpoint_operation_matches_envelope(
                        operation,
                        record.current_envelope,
                    )
                    or (
                        operation.receipt is not None
                        and operation.receipt.receipt_id == record.last_checkpoint_receipt_id
                    )
                ),
            )
            return (
                _validated_model_update(record, {"checkpoint_operations": operations}),
                reservation,
            )

        return await self._mutate(owned.access.profile_id, mutate)

    async def stage_checkpoint(
        self,
        reservation: BrowserProfileCheckpointReservation,
        envelope: BrowserProfileEncryptedEnvelope,
        *,
        origin_count: int,
        cookie_count: int,
        storage_entry_count: int,
    ) -> BrowserProfileCheckpointReceipt:
        owned_reservation = _copy_model(reservation, BrowserProfileCheckpointReservation)
        owned_envelope = _copy_model(envelope, BrowserProfileEncryptedEnvelope)
        if type(origin_count) is not int or not 0 <= origin_count <= BROWSER_PROFILE_MAX_ORIGINS:
            raise ValueError("origin_count is outside the browser-profile bound.")
        if type(cookie_count) is not int or not 0 <= cookie_count <= BROWSER_PROFILE_MAX_COOKIES:
            raise ValueError("cookie_count is outside the browser-profile bound.")
        if (
            type(storage_entry_count) is not int
            or not 0 <= storage_entry_count <= BROWSER_PROFILE_MAX_STORAGE_ENTRIES
        ):
            raise ValueError("storage_entry_count is outside the browser-profile bound.")
        request = owned_reservation.request

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileCheckpointReceipt]:
            _check_access(record, request.access)
            existing = record.checkpoint_operations.get(request.operation_id)
            if existing is None or existing.reservation != owned_reservation:
                raise BrowserProfileStoreConflict("Browser-profile checkpoint was not reserved.")
            if existing.envelope is not None or existing.receipt is not None:
                if (
                    existing.envelope == owned_envelope
                    and existing.receipt is not None
                    and existing.receipt.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
                    and existing.receipt.origin_count == origin_count
                    and existing.receipt.cookie_count == cookie_count
                    and existing.receipt.storage_entry_count == storage_entry_count
                ):
                    return record, existing.receipt
                raise BrowserProfileStoreConflict("Browser-profile staged checkpoint conflicts.")
            _check_available(record, now)
            if record.writer is None or not _same_claim(record.writer, request.writer_claim):
                raise BrowserProfileStoreConflict("Browser-profile writer authority changed.")
            if record.writer.expires_at <= now:
                raise BrowserProfileStoreConflict("Browser-profile writer lease expired.")
            previous = _profile_ref(record)
            if (
                owned_envelope.generation != previous.generation + 1
                or owned_envelope.authority_fingerprint != record.authority.fingerprint
                or owned_envelope.key_authority_id != record.authority.key_authority_id
                or owned_envelope.store_id != self.id
                or owned_envelope.ciphertext_bytes > owned_reservation.reserved_ciphertext_bytes
            ):
                raise BrowserProfileStoreConflict("Browser-profile staged generation is invalid.")
            nonce = owned_envelope.nonce_base64
            prior_envelopes = (
                [record.current_envelope] if record.current_envelope is not None else []
            ) + [
                operation.envelope
                for operation in record.checkpoint_operations.values()
                if operation.envelope is not None
            ]
            if any(item.nonce_base64 == nonce for item in prior_envelopes):
                raise BrowserProfileStoreConflict("Browser-profile encryption nonce was reused.")
            published_ref = BrowserProfileRef(
                profile_id=record.authority.profile_id,
                authority_fingerprint=record.authority.fingerprint,
                generation=owned_envelope.generation,
                content_fingerprint=owned_envelope.content_fingerprint,
            )
            receipt = BrowserProfileCheckpointReceipt(
                receipt_id=_receipt_id(
                    "checkpoint",
                    request.operation_id,
                    owned_reservation.request_fingerprint,
                ),
                operation_id=request.operation_id,
                request_fingerprint=owned_reservation.request_fingerprint,
                previous_ref=previous,
                published_ref=published_ref,
                current_policy_fingerprint=request.current_policy_fingerprint,
                execution_profile_fingerprint=(request.writer_claim.execution_profile_fingerprint),
                allocation_fingerprint=request.writer_claim.allocation_fingerprint,
                browser_session_id=request.writer_claim.browser_session_id,
                writer_fence=request.writer_claim.fence,
                source_revision=request.source_revision,
                source_operation_receipt_id=request.source_operation_receipt_id,
                source_operation_fingerprint=request.source_operation_fingerprint,
                ambiguous_lineage=request.ambiguous_lineage,
                origin_count=origin_count,
                cookie_count=cookie_count,
                storage_entry_count=storage_entry_count,
                outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
                settled_at=now,
            )
            operations = dict(record.checkpoint_operations)
            operations[request.operation_id] = _validated_model_update(
                existing,
                {"envelope": owned_envelope, "receipt": receipt},
            )
            return (
                _validated_model_update(record, {"checkpoint_operations": operations}),
                receipt,
            )

        return await self._mutate(request.access.profile_id, mutate)

    async def fail_checkpoint(
        self,
        reservation: BrowserProfileCheckpointReservation,
        *,
        outcome: BrowserProfileTerminalOutcome,
        error_code: str,
    ) -> BrowserProfileCheckpointReceipt:
        if type(outcome) is not BrowserProfileTerminalOutcome:
            raise TypeError("outcome must be a BrowserProfileTerminalOutcome.")
        if outcome is BrowserProfileTerminalOutcome.SUCCEEDED:
            raise ValueError("Failed checkpoint settlement requires a failure outcome.")
        owned = _copy_model(reservation, BrowserProfileCheckpointReservation)
        safe_error = _fixed_error_code(
            error_code,
            "error_code",
            allowed=_CHECKPOINT_ERROR_CODES,
        )
        request = owned.request

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileCheckpointReceipt]:
            _check_access(record, request.access)
            existing = record.checkpoint_operations.get(request.operation_id)
            if existing is None or existing.reservation != owned:
                raise BrowserProfileStoreConflict("Browser-profile checkpoint was not reserved.")
            if existing.receipt is not None:
                if (
                    existing.envelope is None
                    and existing.receipt.outcome is outcome
                    and existing.receipt.error_code == safe_error
                ):
                    return record, existing.receipt
                raise BrowserProfileStoreConflict("Browser-profile checkpoint is already settled.")
            previous = BrowserProfileRef(
                profile_id=request.writer_claim.profile_id,
                authority_fingerprint=request.writer_claim.authority_fingerprint,
                generation=request.writer_claim.generation,
                content_fingerprint=request.writer_claim.content_fingerprint,
            )
            receipt = BrowserProfileCheckpointReceipt(
                receipt_id=_receipt_id(
                    "checkpoint",
                    request.operation_id,
                    owned.request_fingerprint,
                ),
                operation_id=request.operation_id,
                request_fingerprint=owned.request_fingerprint,
                previous_ref=previous,
                published_ref=previous,
                current_policy_fingerprint=request.current_policy_fingerprint,
                execution_profile_fingerprint=request.writer_claim.execution_profile_fingerprint,
                allocation_fingerprint=request.writer_claim.allocation_fingerprint,
                browser_session_id=request.writer_claim.browser_session_id,
                writer_fence=request.writer_claim.fence,
                source_revision=request.source_revision,
                source_operation_receipt_id=request.source_operation_receipt_id,
                source_operation_fingerprint=request.source_operation_fingerprint,
                ambiguous_lineage=request.ambiguous_lineage,
                origin_count=owned.previous_origin_count,
                cookie_count=owned.previous_cookie_count,
                storage_entry_count=owned.previous_storage_entry_count,
                outcome=outcome,
                error_code=safe_error,
                settled_at=now,
            )
            operations = dict(record.checkpoint_operations)
            operations[request.operation_id] = _validated_model_update(
                existing,
                {"receipt": receipt},
            )
            return (
                _validated_model_update(
                    record,
                    {
                        "checkpoint_operations": operations,
                        "last_checkpoint_receipt_id": receipt.receipt_id,
                        "safe_error_code": safe_error,
                    },
                ),
                receipt,
            )

        return await self._mutate(request.access.profile_id, mutate)

    async def publish_checkpoint(
        self,
        access: BrowserProfileAccess,
        operation_id: str,
    ) -> BrowserProfileCheckpointReceipt:
        owned_access = _copy_model(access, BrowserProfileAccess)
        owned_operation_id = _identifier(operation_id, "operation_id")

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, BrowserProfileCheckpointReceipt]:
            _check_access(record, owned_access)
            existing = record.checkpoint_operations.get(owned_operation_id)
            if existing is None or existing.envelope is None or existing.receipt is None:
                raise BrowserProfileStoreConflict("Browser-profile checkpoint is not staged.")
            receipt = existing.receipt
            if existing.published:
                return record, receipt
            _check_available(record, now)
            previous = _profile_ref(record)
            if previous != receipt.previous_ref:
                raise BrowserProfileStoreConflict("Browser-profile checkpoint generation changed.")
            request = existing.reservation.request
            if record.writer is None or not _same_claim(record.writer, request.writer_claim):
                raise BrowserProfileStoreConflict("Browser-profile writer authority changed.")
            if record.writer.expires_at <= now:
                raise BrowserProfileStoreConflict("Browser-profile writer lease expired.")
            envelope = existing.envelope
            updated_claim = _validated_model_update(
                record.writer,
                {
                    "generation": envelope.generation,
                    "content_fingerprint": envelope.content_fingerprint,
                },
            )
            operations = dict(record.checkpoint_operations)
            operations[owned_operation_id] = _validated_model_update(
                existing,
                {"published": True},
            )
            return (
                _validated_model_update(
                    record,
                    {
                        "current_envelope": envelope,
                        "writer": updated_claim,
                        "checkpoint_operations": operations,
                        "checkpointed_at": receipt.settled_at,
                        "last_checkpoint_receipt_id": receipt.receipt_id,
                        "safe_error_code": None,
                        "origin_count": receipt.origin_count,
                        "cookie_count": receipt.cookie_count,
                        "storage_entry_count": receipt.storage_entry_count,
                    },
                ),
                receipt,
            )

        return await self._mutate(owned_access.profile_id, mutate)

    async def load_checkpoint_receipt(
        self,
        access: BrowserProfileAccess,
        operation_id: str,
    ) -> BrowserProfileCheckpointReceipt | None:
        owned_access = _copy_model(access, BrowserProfileAccess)
        owned_operation_id = _identifier(operation_id, "operation_id")
        record = await self._load_owned_record(owned_access.profile_id)
        if record is None:
            raise BrowserProfileUnavailable("Browser profile does not exist.")
        _check_access(record, owned_access)
        operation = record.checkpoint_operations.get(owned_operation_id)
        if operation is None or operation.receipt is None:
            return None
        if (
            operation.receipt.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
            and not operation.published
        ):
            return None
        return operation.receipt

    async def reconcile_checkpoint(
        self,
        access: BrowserProfileAccess,
        operation_id: str,
    ) -> BrowserProfileCheckpointReceipt | None:
        """Adopt an already staged generation without recapturing plaintext."""

        owned_access = _copy_model(access, BrowserProfileAccess)
        owned_operation_id = _identifier(operation_id, "operation_id")
        record = await self._load_owned_record(owned_access.profile_id)
        if record is None:
            raise BrowserProfileUnavailable("Browser profile does not exist.")
        _check_access(record, owned_access)
        operation = record.checkpoint_operations.get(owned_operation_id)
        if operation is None or operation.receipt is None:
            return None
        if operation.receipt.outcome is not BrowserProfileTerminalOutcome.SUCCEEDED:
            return operation.receipt
        if operation.published:
            return operation.receipt
        return await self.publish_checkpoint(owned_access, owned_operation_id)

    async def release_writer(
        self,
        access: BrowserProfileAccess,
        claim: BrowserProfileWriterClaim,
    ) -> None:
        owned_access = _copy_model(access, BrowserProfileAccess)
        owned_claim = _copy_model(claim, BrowserProfileWriterClaim)

        def mutate(
            record: _StoredBrowserProfile,
            _now_value: datetime,
        ) -> tuple[_StoredBrowserProfile, None]:
            _check_access(record, owned_access)
            if record.writer is None:
                return record, None
            if not _same_claim(record.writer, owned_claim):
                raise BrowserProfileStoreConflict("Browser-profile writer authority changed.")
            unsettled_restore = any(
                operation.claim.fence == owned_claim.fence and operation.receipt is None
                for operation in record.restore_operations.values()
            )
            if unsettled_restore:
                raise BrowserProfileStoreConflict(
                    "Browser-profile writer cannot be released before restore settlement."
                )
            unsettled = any(
                not _checkpoint_operation_is_settled(operation)
                for operation in record.checkpoint_operations.values()
                if operation.reservation.request.writer_claim.fence == owned_claim.fence
            )
            if unsettled:
                raise BrowserProfileStoreConflict(
                    "Browser-profile writer cannot be released before checkpoint settlement."
                )
            return _validated_model_update(record, {"writer": None}), None

        await self._mutate(owned_access.profile_id, mutate)

    async def revoke_profile(
        self,
        access: BrowserProfileAccess,
        *,
        revoked_at: datetime | None = None,
    ) -> BrowserProfileInspection:
        owned_access = _copy_model(access, BrowserProfileAccess)
        implicit_time = revoked_at is None
        observed = None if implicit_time else _utc(revoked_at, "revoked_at")

        def mutate(
            record: _StoredBrowserProfile,
            now: datetime,
        ) -> tuple[_StoredBrowserProfile, None]:
            _check_access(record, owned_access)
            effective = max(
                now if observed is None else observed,
                record.authority.created_at,
            )
            if effective > now:
                raise ValueError("revoked_at cannot be in the future.")
            existing = record.revoked_at
            if existing is not None:
                if not implicit_time and existing != effective:
                    raise BrowserProfileStoreConflict("Browser profile has conflicting revocation.")
                return record, None
            return _validated_model_update(record, {"revoked_at": effective}), None

        await self._mutate(owned_access.profile_id, mutate)
        return await self.inspect_profile(owned_access)

    async def inspect_profile(self, access: BrowserProfileAccess) -> BrowserProfileInspection:
        owned_access = _copy_model(access, BrowserProfileAccess)
        record = await self._load_owned_record(owned_access.profile_id)
        if record is None:
            raise BrowserProfileUnavailable("Browser profile does not exist.")
        _check_access(record, owned_access)
        return _inspection(record, self._inspection_time())

    async def list_profiles(
        self,
        *,
        owner_fingerprint: str,
        sharing_fingerprint: str,
    ) -> tuple[BrowserProfileInspection, ...]:
        owner = _digest(owner_fingerprint, "owner_fingerprint")
        sharing = _digest(sharing_fingerprint, "sharing_fingerprint")
        now = self._inspection_time()
        matched = [
            _inspection(record, now)
            for record in await self._list_owned_records()
            if record.authority.scope.owner_fingerprint == owner
            and record.authority.scope.sharing_fingerprint == sharing
        ]
        return tuple(sorted(matched, key=lambda item: item.profile_id))

    def _inspection_time(self) -> datetime:
        clock_failure: BaseException | None = None
        observed_at: datetime | None = None
        try:
            observed_at = self._time()
        except BaseException as failure:
            clock_failure = failure
        if clock_failure is not None:
            safe_failure = _safe_store_primitive_failure(clock_failure)
            clock_failure = None
            raise safe_failure from None
        if observed_at is None:  # pragma: no cover - clock return contract
            raise BrowserProfileUnavailable("Browser-profile store operation failed.")
        return observed_at


def _bounded_operations(
    value: dict[str, _ValueT],
    *,
    settled: Callable[[_ValueT], bool],
    preserve: Callable[[_ValueT], bool] | None = None,
) -> dict[str, _ValueT]:
    if len(value) <= BROWSER_PROFILE_MAX_RECEIPTS:
        return value
    retained = dict(value)
    for operation_id, operation in tuple(retained.items()):
        if settled(operation) and (preserve is None or not preserve(operation)):
            retained.pop(operation_id)
            if len(retained) <= BROWSER_PROFILE_MAX_RECEIPTS:
                return retained
    raise BrowserProfileUnavailable("Browser-profile unsettled operation capacity is exhausted.")


def _bounded_restore_history(
    record: _StoredBrowserProfile,
    restore_operations: dict[str, _StoredRestoreOperation],
) -> tuple[
    dict[str, _StoredRestoreOperation],
    dict[str, _StoredCheckpointOperation],
]:
    """Prune settled history without removing authority retained by checkpoints."""

    if len(restore_operations) <= BROWSER_PROFILE_MAX_RECEIPTS:
        return restore_operations, dict(record.checkpoint_operations)
    retained_restores = dict(restore_operations)
    retained_checkpoints = dict(record.checkpoint_operations)

    def protected_restore_fences() -> set[int]:
        protected = {
            operation.reservation.request.writer_claim.fence
            for operation in retained_checkpoints.values()
        }
        if record.writer is not None:
            protected.add(record.writer.fence)
        return protected

    def prune_restore() -> bool:
        protected_fences = protected_restore_fences()
        for operation_id, operation in tuple(retained_restores.items()):
            receipt = operation.receipt
            if (
                receipt is not None
                and operation.claim.fence not in protected_fences
                and receipt.receipt_id != record.last_restore_receipt_id
            ):
                retained_restores.pop(operation_id)
                return True
        return False

    while len(retained_restores) > BROWSER_PROFILE_MAX_RECEIPTS:
        if prune_restore():
            continue
        checkpoint_pruned = False
        for operation_id, operation in tuple(retained_checkpoints.items()):
            if (
                _checkpoint_operation_is_settled(operation)
                and not _checkpoint_operation_matches_envelope(
                    operation,
                    record.current_envelope,
                )
                and (
                    operation.receipt is None
                    or operation.receipt.receipt_id != record.last_checkpoint_receipt_id
                )
            ):
                retained_checkpoints.pop(operation_id)
                checkpoint_pruned = True
                break
        # Several checkpoints can protect the same restore. Removing one is
        # progress even when another still holds that restore's authority;
        # re-evaluate the remaining dependencies on the next iteration.
        if not checkpoint_pruned:
            raise BrowserProfileUnavailable(
                "Browser-profile unsettled operation capacity is exhausted."
            )
    return retained_restores, retained_checkpoints


def _checkpoint_operation_matches_envelope(
    operation: _StoredCheckpointOperation,
    envelope: BrowserProfileEncryptedEnvelope | None,
) -> bool:
    return bool(envelope is not None and operation.published and operation.envelope == envelope)


def _receipt_id(kind: str, operation_id: str, request_fingerprint: str) -> str:
    digest = sha256(
        b"cayu.browser-profile.receipt.v1\0"
        + kind.encode("ascii")
        + b"\0"
        + operation_id.encode("utf-8")
        + b"\0"
        + request_fingerprint.encode("ascii")
    ).hexdigest()
    return f"bpr_{digest}"


def _inspection(record: _StoredBrowserProfile, now: datetime) -> BrowserProfileInspection:
    authority = record.authority
    current = _profile_ref(record)
    envelope = record.current_envelope
    active_writer = record.writer is not None and record.writer.expires_at > now
    expired_unsettled_writer = bool(
        record.writer is not None
        and record.writer.expires_at <= now
        and (
            any(
                operation.claim.fence == record.writer.fence and operation.receipt is None
                for operation in record.restore_operations.values()
            )
            or any(
                operation.reservation.request.writer_claim.fence == record.writer.fence
                and not _checkpoint_operation_is_settled(operation)
                for operation in record.checkpoint_operations.values()
            )
        )
    )
    if record.revoked_at is not None:
        status = BrowserProfileStatus.REVOKED
    elif authority.expires_at is not None and now >= authority.expires_at:
        status = BrowserProfileStatus.EXPIRED
    elif record.safe_error_code == "profile_corrupt":
        status = BrowserProfileStatus.CORRUPT
    elif record.safe_error_code == "profile_incompatible":
        status = BrowserProfileStatus.INCOMPATIBLE
    elif expired_unsettled_writer or (
        record.safe_error_code is not None and record.safe_error_code.endswith("outcome_unknown")
    ):
        status = BrowserProfileStatus.OUTCOME_UNKNOWN
    elif active_writer:
        status = BrowserProfileStatus.ACTIVE
    elif envelope is None:
        status = BrowserProfileStatus.EMPTY
    else:
        status = BrowserProfileStatus.AVAILABLE
    return BrowserProfileInspection(
        profile_id=authority.profile_id,
        authority_fingerprint=authority.fingerprint,
        owner_fingerprint=authority.scope.owner_fingerprint,
        sharing_fingerprint=authority.scope.sharing_fingerprint,
        destination_policy_fingerprint=authority.destination_policy.fingerprint,
        state_schema_version=authority.state_schema_version,
        browser_protocol=authority.browser_protocol,
        browser_worker_version=authority.browser_worker_version,
        key_authority_id=authority.key_authority_id,
        store_id=authority.store_id,
        generation=current.generation,
        content_fingerprint=current.content_fingerprint,
        origin_count=record.origin_count,
        cookie_count=record.cookie_count,
        storage_entry_count=record.storage_entry_count,
        plaintext_bytes=0 if envelope is None else envelope.plaintext_bytes,
        ciphertext_bytes=0 if envelope is None else envelope.ciphertext_bytes,
        created_at=authority.created_at,
        checkpointed_at=record.checkpointed_at,
        expires_at=authority.expires_at,
        revoked_at=record.revoked_at,
        active_writer=active_writer,
        active_allocation_fingerprint=(
            record.writer.allocation_fingerprint if active_writer and record.writer else None
        ),
        active_writer_expires_at=(
            record.writer.expires_at if active_writer and record.writer else None
        ),
        status=status,
        last_restore_receipt_id=record.last_restore_receipt_id,
        last_checkpoint_receipt_id=record.last_checkpoint_receipt_id,
        safe_error_code=(
            "writer_lease_expired"
            if expired_unsettled_writer and record.safe_error_code is None
            else record.safe_error_code
        ),
    )


class InMemoryBrowserProfileStore(BrowserProfileStore):
    """Process-local store used for deterministic composition and conformance."""

    def __init__(
        self,
        *,
        store_id: str = "browser-profiles-memory",
        clock: Callable[[], datetime] = _now,
        max_ciphertext_bytes: int = BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES,
    ) -> None:
        super().__init__(
            store_id=store_id,
            clock=clock,
            max_ciphertext_bytes=max_ciphertext_bytes,
        )
        self._records: dict[str, _StoredBrowserProfile] = {}
        self._lock = asyncio.Lock()

    async def _create_record(
        self,
        record: _StoredBrowserProfile,
    ) -> _StoredBrowserProfile:
        owned = _copy_model(record, _StoredBrowserProfile)
        async with self._lock:
            existing = self._records.get(owned.authority.profile_id)
            if existing is not None:
                if existing.authority == owned.authority:
                    return _copy_model(existing, _StoredBrowserProfile)
                raise BrowserProfileStoreConflict("Browser-profile identity already exists.")
            self._records[owned.authority.profile_id] = owned
            return _copy_model(owned, _StoredBrowserProfile)

    async def _load_record(self, profile_id: str) -> _StoredBrowserProfile | None:
        owned_id = _identifier(profile_id, "profile_id")
        async with self._lock:
            record = self._records.get(owned_id)
            return None if record is None else _copy_model(record, _StoredBrowserProfile)

    async def _mutate_record(
        self,
        profile_id: str,
        operation: Callable[
            [_StoredBrowserProfile, datetime],
            tuple[_StoredBrowserProfile, Any],
        ],
    ) -> Any:
        owned_id = _identifier(profile_id, "profile_id")
        async with self._lock:
            current = self._records.get(owned_id)
            if current is None:
                raise BrowserProfileUnavailable("Browser profile does not exist.")
            record = _copy_model(current, _StoredBrowserProfile)
            updated, result = operation(record, self._time())
            desired = _validated_model_update(
                updated,
                {"revision": record.revision + 1},
            )
            self._records[owned_id] = _copy_model(desired, _StoredBrowserProfile)
            return result

    async def _list_records(self) -> tuple[_StoredBrowserProfile, ...]:
        async with self._lock:
            return tuple(
                _copy_model(record, _StoredBrowserProfile) for record in self._records.values()
            )


class SQLiteBrowserProfileStore(BrowserProfileStore):
    """SQLite-backed encrypted profile store with transactional row CAS."""

    def __init__(
        self,
        path: str | Path,
        *,
        store_id: str = "browser-profiles-sqlite",
        clock: Callable[[], datetime] = _now,
        max_ciphertext_bytes: int = BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES,
    ) -> None:
        super().__init__(
            store_id=store_id,
            clock=clock,
            max_ciphertext_bytes=max_ciphertext_bytes,
        )
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cayu_browser_profiles (
                profile_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                record_json BLOB NOT NULL
            )
            """
        )
        self._connection.commit()
        self._lock = threading.RLock()
        self._closed = False

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("SQLite browser-profile store is closed.")
            failure: BaseException | None = None
            suppress_implicit_context = False
            transaction_started = False
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                transaction_started = True
                yield self._connection
                self._connection.commit()
            except BaseException as primary:
                try:
                    in_transaction = self._connection.in_transaction
                except BaseException as state_failure:
                    state_failure.__context__ = None
                    failure = self._fence_failed_transaction(primary, state_failure)
                    suppress_implicit_context = True
                else:
                    if transaction_started or in_transaction:
                        failure = self._settle_failed_transaction(primary, in_transaction)
                        suppress_implicit_context = failure is not primary
                    else:
                        failure = primary
            if failure is not None:
                if suppress_implicit_context:
                    raise failure from None
                raise failure

    def _settle_failed_transaction(
        self,
        primary: BaseException,
        in_transaction: bool,
    ) -> BaseException:
        """Roll back uncertain work or close a connection whose cleanup failed."""

        if not in_transaction:
            # A commit may have completed before acknowledgement failed. The
            # caller's exact receipt/readback path owns that reconciliation.
            return primary
        try:
            self._connection.rollback()
        except BaseException as rollback_failure:
            rollback_failure.__context__ = None
            return self._fence_failed_transaction(primary, rollback_failure)
        return primary

    def _fence_failed_transaction(
        self,
        primary: BaseException,
        cleanup_failure: BaseException,
    ) -> BaseException:
        """Fail closed when rollback cannot prove that the writer was released."""

        failures = [primary, cleanup_failure]
        try:
            self._connection.close()
        except BaseException as close_failure:
            close_failure.__context__ = None
            failures.append(close_failure)
        self._closed = True
        return BaseExceptionGroup(
            "SQLite browser-profile transaction failed and cleanup was uncertain.",
            failures,
        )

    @staticmethod
    def _encode(record: _StoredBrowserProfile) -> bytes:
        return canonical_durable_json_bytes(
            record.model_dump(mode="json"),
            "browser profile record",
        )

    @staticmethod
    def _decode(raw: bytes) -> _StoredBrowserProfile:
        value: object = None
        try:
            value = json.loads(raw.decode("utf-8"))
            return _StoredBrowserProfile.model_validate(value)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            raw = b""
            value = None
            raise BrowserProfileUnavailable("Browser-profile record is corrupt.") from None

    async def _create_record(
        self,
        record: _StoredBrowserProfile,
    ) -> _StoredBrowserProfile:
        owned = _copy_model(record, _StoredBrowserProfile)

        def create() -> _StoredBrowserProfile:
            with self._transaction() as connection:
                existing = connection.execute(
                    "SELECT record_json FROM cayu_browser_profiles WHERE profile_id = ?",
                    (owned.authority.profile_id,),
                ).fetchone()
                if existing is not None:
                    raw = existing[0]
                    existing = None
                    try:
                        decoded = self._decode(raw)
                    finally:
                        raw = b""
                    if decoded.authority == owned.authority:
                        return decoded
                    raise BrowserProfileStoreConflict("Browser-profile identity already exists.")
                connection.execute(
                    "INSERT INTO cayu_browser_profiles(profile_id, revision, record_json) "
                    "VALUES (?, ?, ?)",
                    (owned.authority.profile_id, owned.revision, self._encode(owned)),
                )
                return owned

        return await asyncio.to_thread(create)

    async def _load_record(self, profile_id: str) -> _StoredBrowserProfile | None:
        owned_id = _identifier(profile_id, "profile_id")

        def load() -> _StoredBrowserProfile | None:
            row = None
            raw = b""
            with self._lock:
                if self._closed:
                    raise RuntimeError("SQLite browser-profile store is closed.")
                row = self._connection.execute(
                    "SELECT record_json FROM cayu_browser_profiles WHERE profile_id = ?",
                    (owned_id,),
                ).fetchone()
            if row is None:
                return None
            raw = row[0]
            row = None
            try:
                return self._decode(raw)
            finally:
                raw = b""

        return await asyncio.to_thread(load)

    async def _mutate_record(
        self,
        profile_id: str,
        operation: Callable[
            [_StoredBrowserProfile, datetime],
            tuple[_StoredBrowserProfile, Any],
        ],
    ) -> Any:
        owned_id = _identifier(profile_id, "profile_id")

        def mutate() -> Any:
            row = None
            raw = b""
            record = None
            updated = None
            desired = None
            with self._transaction() as connection:
                row = connection.execute(
                    "SELECT revision, record_json FROM cayu_browser_profiles WHERE profile_id = ?",
                    (owned_id,),
                ).fetchone()
                if row is None:
                    raise BrowserProfileUnavailable("Browser profile does not exist.")
                stored_revision = row[0]
                raw = row[1]
                row = None
                try:
                    record = self._decode(raw)
                finally:
                    raw = b""
                if type(stored_revision) is not int or stored_revision != record.revision:
                    raise BrowserProfileUnavailable("Browser-profile record is corrupt.")
                updated, result = operation(record, self._time())
                desired = _validated_model_update(
                    updated,
                    {"revision": record.revision + 1},
                )
                cursor = connection.execute(
                    "UPDATE cayu_browser_profiles SET revision = ?, record_json = ? "
                    "WHERE profile_id = ? AND revision = ?",
                    (
                        desired.revision,
                        self._encode(desired),
                        owned_id,
                        record.revision,
                    ),
                )
                if cursor.rowcount != 1:  # pragma: no cover - write lock invariant
                    raise BrowserProfileStoreConflict(
                        "Browser-profile mutation lost its transactional authority."
                    )
                record = None
                updated = None
                desired = None
                return result

        return await asyncio.to_thread(mutate)

    async def _list_records(self) -> tuple[_StoredBrowserProfile, ...]:
        def load_all() -> tuple[_StoredBrowserProfile, ...]:
            rows = None
            pending: list[bytes] = []
            raw = b""
            decoded: list[_StoredBrowserProfile] = []
            with self._lock:
                if self._closed:
                    raise RuntimeError("SQLite browser-profile store is closed.")
                rows = self._connection.execute(
                    "SELECT record_json FROM cayu_browser_profiles ORDER BY profile_id"
                ).fetchall()
            pending = [row[0] for row in rows]
            rows = None
            try:
                while pending:
                    raw = pending.pop()
                    decoded.append(self._decode(raw))
                    raw = b""
                return tuple(reversed(decoded))
            finally:
                rows = None
                pending.clear()
                raw = b""
                decoded.clear()

        return await asyncio.to_thread(load_all)

    async def close(self) -> None:
        def close_sync() -> None:
            with self._lock:
                if not self._closed:
                    self._connection.close()
                    self._closed = True

        await asyncio.to_thread(close_sync)


class BrowserProfileRestoreMaterial:
    """Private decrypted state and exact durable authority for one restore."""

    __slots__ = ("_preparation", "_sealed", "_state")

    _preparation: BrowserProfileRestorePreparation
    _sealed: bool
    _state: BrowserProfileStateV1

    def __init__(
        self,
        *,
        preparation: BrowserProfileRestorePreparation,
        state: BrowserProfileStateV1,
    ) -> None:
        object.__setattr__(
            self,
            "_preparation",
            _copy_model(preparation, BrowserProfileRestorePreparation),
        )
        object.__setattr__(self, "_state", _copy_model(state, BrowserProfileStateV1))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("BrowserProfileRestoreMaterial is immutable.")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return "BrowserProfileRestoreMaterial(<private>)"

    @property
    def preparation(self) -> BrowserProfileRestorePreparation:
        return _copy_model(self._preparation, BrowserProfileRestorePreparation)

    @property
    def state(self) -> BrowserProfileStateV1:
        return _copy_model(self._state, BrowserProfileStateV1)

    def _snapshot(
        self,
    ) -> tuple[BrowserProfileRestorePreparation, BrowserProfileStateV1]:
        """Return an owned private snapshot without trusting mutable attributes."""

        validation_failure: BaseException | None = None
        preparation: BrowserProfileRestorePreparation | None = None
        state: BrowserProfileStateV1 | None = None
        try:
            preparation = _copy_model(
                self._preparation,
                BrowserProfileRestorePreparation,
            )
            state = _copy_model(self._state, BrowserProfileStateV1)
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            preparation = None
            state = None
            raise BrowserProfileUnavailable(
                "Browser-profile restore material is invalid."
            ) from None
        if preparation is None or state is None:  # pragma: no cover - paired validation
            raise BrowserProfileUnavailable("Browser-profile restore material is invalid.")
        return preparation, state

    def _replace_claim(
        self,
        claim: BrowserProfileWriterClaim,
        *,
        profile_ref: BrowserProfileRef | None = None,
        state: BrowserProfileStateV1 | None = None,
        envelope: BrowserProfileEncryptedEnvelope | None = None,
    ) -> None:
        preparation, current_state = self._snapshot()
        owned_claim = _copy_model(claim, BrowserProfileWriterClaim)
        owned_ref = (
            preparation.profile_ref
            if profile_ref is None
            else _copy_model(profile_ref, BrowserProfileRef)
        )
        if not _same_writer_lineage(preparation.writer_claim, owned_claim):
            raise BrowserProfileStoreConflict(
                "Browser-profile writer replacement changes immutable authority."
            )
        if (
            owned_claim.generation != owned_ref.generation
            or owned_claim.content_fingerprint != owned_ref.content_fingerprint
            or owned_ref.profile_id != preparation.profile_ref.profile_id
            or owned_ref.authority_fingerprint != preparation.profile_ref.authority_fingerprint
            or owned_claim.generation < preparation.writer_claim.generation
            or owned_claim.generation > preparation.writer_claim.generation + 1
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile writer replacement has conflicting generation authority."
            )
        owned_state = current_state if state is None else _copy_model(state, BrowserProfileStateV1)
        owned_envelope = (
            preparation.envelope
            if envelope is None
            else _copy_model(envelope, BrowserProfileEncryptedEnvelope)
        )
        if (owned_ref.generation == 0) != (owned_envelope is None) or (
            owned_envelope is not None
            and (
                owned_envelope.authority_fingerprint != owned_ref.authority_fingerprint
                or owned_envelope.generation != owned_ref.generation
                or owned_envelope.content_fingerprint != owned_ref.content_fingerprint
            )
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile writer replacement has conflicting ciphertext authority."
            )
        updated = _validated_model_update(
            preparation,
            {
                "writer_claim": owned_claim,
                "profile_ref": owned_ref,
                "envelope": owned_envelope,
            },
        )
        object.__setattr__(self, "_preparation", updated)
        object.__setattr__(self, "_state", owned_state)


class BrowserProfileCheckpointPlan:
    """Capacity reservation held before browser profile plaintext is exported."""

    __slots__ = ("_material", "_reservation", "_sealed")

    _material: BrowserProfileRestoreMaterial
    _reservation: BrowserProfileCheckpointReservation
    _sealed: bool

    def __init__(
        self,
        reservation: BrowserProfileCheckpointReservation,
        material: BrowserProfileRestoreMaterial,
    ) -> None:
        if type(material) is not BrowserProfileRestoreMaterial:
            raise TypeError("material must be an exact BrowserProfileRestoreMaterial instance.")
        owned_reservation = _copy_model(
            reservation,
            BrowserProfileCheckpointReservation,
        )
        preparation, _state = material._snapshot()
        if owned_reservation.request.writer_claim != preparation.writer_claim:
            raise BrowserProfileStoreConflict(
                "Browser-profile checkpoint plan has conflicting writer authority."
            )
        object.__setattr__(self, "_reservation", owned_reservation)
        object.__setattr__(self, "_material", material)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("BrowserProfileCheckpointPlan is immutable.")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return "BrowserProfileCheckpointPlan(<private>)"

    @property
    def reservation(self) -> BrowserProfileCheckpointReservation:
        return _copy_model(self._reservation, BrowserProfileCheckpointReservation)

    @property
    def material(self) -> BrowserProfileRestoreMaterial:
        return self._material

    def _snapshot(
        self,
    ) -> tuple[
        BrowserProfileCheckpointReservation,
        BrowserProfileRestoreMaterial,
        BrowserProfileRestorePreparation,
        BrowserProfileStateV1,
    ]:
        validation_failure: BaseException | None = None
        reservation: BrowserProfileCheckpointReservation | None = None
        material: BrowserProfileRestoreMaterial | None = None
        preparation: BrowserProfileRestorePreparation | None = None
        state: BrowserProfileStateV1 | None = None
        try:
            reservation = _copy_model(
                self._reservation,
                BrowserProfileCheckpointReservation,
            )
            material = self._material
            if type(material) is not BrowserProfileRestoreMaterial:
                raise TypeError("Browser-profile checkpoint material is invalid.")
            preparation, state = material._snapshot()
            if reservation.request.writer_claim != preparation.writer_claim:
                raise BrowserProfileStoreConflict(
                    "Browser-profile checkpoint plan has conflicting writer authority."
                )
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            reservation = None
            material = None
            preparation = None
            state = None
            raise BrowserProfileUnavailable("Browser-profile checkpoint plan is invalid.") from None
        if (
            reservation is None or material is None or preparation is None or state is None
        ):  # pragma: no cover - paired validation
            raise BrowserProfileUnavailable("Browser-profile checkpoint plan is invalid.")
        return reservation, material, preparation, state


async def _settle_profile_task(
    operation: Awaitable[_ValueT],
    *,
    task_name: str,
    cancellation: asyncio.CancelledError | None = None,
    preserve_errors: tuple[type[BaseException], ...] = (),
) -> _ValueT:
    # Start the extension inside the capture task. Starting it in a separate
    # task first would let KeyboardInterrupt/SystemExit escape through asyncio's
    # task runner before this boundary could preserve them as child outcomes.
    captured_task = asyncio.create_task(
        capture_awaitable_outcome(lambda owned_operation=operation: owned_operation),
        name=task_name,
    )
    outcome = await await_shielded_task_outcome(
        captured_task,
        cancellation=cancellation,
    )
    captured = outcome.result
    child_error = outcome.error if captured is None else captured.error
    child_result = None if captured is None else captured.result
    if outcome.cancellation is not None:
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=outcome.cancellation,
        )
        if child_error is not None:
            safe_error = _safe_profile_extension_failure(child_error)
            if _profile_failure_contains_process_control(safe_error):
                cancellation = outcome.cancellation
                aggregate = BaseExceptionGroup(
                    "Browser-profile settlement received cancellation and process control.",
                    [cancellation, safe_error],
                )
                child_error = None
                child_result = None
                del outcome, captured, captured_task, operation, safe_error, cancellation
                raise aggregate from None
            outcome.cancellation.add_note(
                "Browser-profile settlement also failed while cancellation was pending."
            )
            cancellation = outcome.cancellation
            child_error = None
            child_result = None
            del outcome, captured, captured_task, operation
            raise cancellation from safe_error
        cancellation = outcome.cancellation
        del outcome, captured, captured_task, operation
        raise cancellation
    if child_error is not None:
        safe_error = (
            child_error
            if preserve_errors and isinstance(child_error, preserve_errors)
            else _safe_profile_extension_failure(child_error)
        )
        child_error = None
        child_result = None
        del outcome, captured, captured_task, operation
        raise safe_error from None
    del outcome, captured, captured_task, operation
    return cast("_ValueT", child_result)


def _clear_inactive_profile_failure_frames(error: BaseException) -> None:
    """Drop completed extension frames that may retain private profile bytes."""

    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        try:
            traceback_module.clear_frames(current.__traceback__)
            current.__traceback__ = None
        except BaseException:
            # Diagnostic cleanup must never replace the authoritative signal.
            pass


def _profile_failure_contains_process_control(error: BaseException) -> bool:
    if isinstance(error, (GeneratorExit, KeyboardInterrupt, SystemExit)):
        return True
    return isinstance(error, BaseExceptionGroup) and any(
        _profile_failure_contains_process_control(child) for child in error.exceptions
    )


def _safe_profile_extension_failure(error: BaseException) -> BaseException:
    """Detach private extension state and publish only bounded diagnostics."""

    _clear_inactive_profile_failure_frames(error)
    if _profile_failure_contains_process_control(error):
        return error
    if isinstance(error, asyncio.CancelledError):
        return BrowserProfileUnavailable("Browser-profile settlement was cancelled unexpectedly.")
    if isinstance(error, (BrowserProfileStoreConflict, BrowserProfileUnavailable)):
        return error
    return BrowserProfileUnavailable("Browser-profile store operation failed.")


def _own_profile_extension_model(
    value: object,
    model_type: type[_ModelT],
) -> _ModelT:
    """Defensively own one typed value returned by an application extension."""

    validation_failure: BaseException | None = None
    owned: _ModelT | None = None
    try:
        owned = _copy_model(value, model_type)
    except (TypeError, ValueError) as failure:
        validation_failure = failure
    value = None
    if validation_failure is not None:
        _clear_inactive_profile_failure_frames(validation_failure)
        validation_failure = None
        raise BrowserProfileUnavailable(
            "Browser-profile extension returned invalid authority evidence."
        ) from None
    if owned is None:  # pragma: no cover - paired validation
        raise BrowserProfileUnavailable(
            "Browser-profile extension returned invalid authority evidence."
        )
    return owned


def _profile_dependency_identity(value: object, attribute: str) -> str:
    """Read one extension identity without publishing extension diagnostics."""

    identity_failure: BaseException | None = None
    identity: str | None = None
    try:
        identity = _identifier(getattr(value, attribute), attribute)
    except BaseException as failure:
        identity_failure = failure
    value = None
    if identity_failure is not None:
        _clear_inactive_profile_failure_frames(identity_failure)
        if _profile_failure_contains_process_control(identity_failure):
            raise identity_failure from None
        identity_failure = None
        raise BrowserProfileUnavailable(
            "Browser-profile dependency authority identity is unavailable."
        ) from None
    if identity is None:  # pragma: no cover - paired validation
        raise BrowserProfileUnavailable(
            "Browser-profile dependency authority identity is unavailable."
        )
    return identity


class BrowserProfileBinding:
    """Application-selected profile authority bound to one interactive browser tool."""

    __slots__ = (
        "_access",
        "_authority",
        "_checkpoint_policy",
        "_current_policy",
        "_expected_ref",
        "_lease_seconds",
        "_limits",
        "_sealed",
        "key_authority",
        "store",
    )

    def __init__(
        self,
        *,
        authority: BrowserProfileAuthority,
        store: BrowserProfileStore,
        key_authority: BrowserProfileKeyAuthority,
        current_policy: BrowserProfileDestinationPolicy | None = None,
        limits: BrowserProfileLimits | None = None,
        checkpoint_policy: BrowserProfileCheckpointPolicy = (
            BrowserProfileCheckpointPolicy.ON_CLOSE
        ),
        expected_ref: BrowserProfileRef | None = None,
        lease_seconds: int = 1_200,
    ) -> None:
        if type(authority) is not BrowserProfileAuthority:
            raise TypeError("authority must be a BrowserProfileAuthority.")
        if not isinstance(store, BrowserProfileStore):
            raise TypeError("store must be a BrowserProfileStore.")
        if not isinstance(key_authority, BrowserProfileKeyAuthority):
            raise TypeError("key_authority must be a BrowserProfileKeyAuthority.")
        owned_authority = _copy_model(authority, BrowserProfileAuthority)
        store_id = _profile_dependency_identity(store, "id")
        key_authority_id = _profile_dependency_identity(
            key_authority,
            "authority_id",
        )
        if owned_authority.store_id != store_id:
            raise ValueError("Browser-profile store identity does not match authority.")
        if owned_authority.key_authority_id != key_authority_id:
            raise ValueError("Browser-profile key authority identity does not match.")
        selected_policy = (
            owned_authority.destination_policy
            if current_policy is None
            else (_copy_model(current_policy, BrowserProfileDestinationPolicy))
        )
        if not selected_policy.is_narrower_than(owned_authority.destination_policy):
            raise ValueError("Browser-profile destination authority cannot be widened.")
        owned_limits = (
            BrowserProfileLimits()
            if limits is None
            else _copy_model(
                limits,
                BrowserProfileLimits,
            )
        )
        store_capacity = store._max_ciphertext_bytes
        if (
            type(store_capacity) is not int
            or not 17 <= store_capacity <= BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES
        ):
            raise BrowserProfileUnavailable(
                "Browser-profile store capacity authority is unavailable."
            )
        if owned_limits.max_ciphertext_bytes > store_capacity:
            raise ValueError("Browser-profile limits exceed store ciphertext capacity.")
        if not isinstance(checkpoint_policy, BrowserProfileCheckpointPolicy):
            raise TypeError("checkpoint_policy must be a BrowserProfileCheckpointPolicy.")
        if expected_ref is not None:
            expected_ref = _copy_model(expected_ref, BrowserProfileRef)
            if (
                expected_ref.profile_id != owned_authority.profile_id
                or expected_ref.authority_fingerprint != owned_authority.fingerprint
            ):
                raise ValueError("Expected browser-profile generation has wrong authority.")
        if (
            type(lease_seconds) is not int
            or not 1 <= lease_seconds <= BROWSER_PROFILE_MAX_LEASE_SECONDS
        ):
            raise ValueError("lease_seconds is outside the browser-profile bound.")
        self._authority = owned_authority
        self._access = BrowserProfileAccess.from_authority(owned_authority)
        self._current_policy = selected_policy
        self._limits = owned_limits
        self._checkpoint_policy = checkpoint_policy
        self._expected_ref = expected_ref
        self._lease_seconds = lease_seconds
        self.store = store
        self.key_authority = key_authority
        self._sealed = True

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("BrowserProfileBinding is immutable.")
        object.__setattr__(self, name, value)

    @property
    def authority(self) -> BrowserProfileAuthority:
        return _copy_model(self._authority, BrowserProfileAuthority)

    @property
    def access(self) -> BrowserProfileAccess:
        return _copy_model(self._access, BrowserProfileAccess)

    @property
    def current_policy(self) -> BrowserProfileDestinationPolicy:
        return _copy_model(self._current_policy, BrowserProfileDestinationPolicy)

    @property
    def limits(self) -> BrowserProfileLimits:
        return _copy_model(self._limits, BrowserProfileLimits)

    @property
    def checkpoint_policy(self) -> BrowserProfileCheckpointPolicy:
        return self._checkpoint_policy

    @property
    def expected_ref(self) -> BrowserProfileRef | None:
        return (
            None
            if self._expected_ref is None
            else _copy_model(self._expected_ref, BrowserProfileRef)
        )

    @property
    def lease_seconds(self) -> int:
        return self._lease_seconds

    @classmethod
    def build(
        cls,
        *,
        scope: BrowserProfileScope,
        destination_policy: BrowserProfileDestinationPolicy,
        browser_protocol: str,
        browser_worker_version: str,
        store: BrowserProfileStore,
        key_authority: BrowserProfileKeyAuthority,
        profile_id: str | None = None,
        created_at: datetime | None = None,
        expires_at: datetime | None = None,
        **options: Any,
    ) -> BrowserProfileBinding:
        if not isinstance(store, BrowserProfileStore):
            raise TypeError("store must be a BrowserProfileStore.")
        if not isinstance(key_authority, BrowserProfileKeyAuthority):
            raise TypeError("key_authority must be a BrowserProfileKeyAuthority.")
        key_authority_id = _profile_dependency_identity(
            key_authority,
            "authority_id",
        )
        store_id = _profile_dependency_identity(store, "id")
        authority = BrowserProfileAuthority.build(
            scope=scope,
            destination_policy=destination_policy,
            browser_protocol=browser_protocol,
            browser_worker_version=browser_worker_version,
            key_authority_id=key_authority_id,
            store_id=store_id,
            profile_id=profile_id,
            created_at=created_at,
            expires_at=expires_at,
        )
        return cls(
            authority=authority,
            store=store,
            key_authority=key_authority,
            **options,
        )

    async def initialize(self) -> BrowserProfileRef:
        self._require_current_dependencies()
        created = await _settle_profile_task(
            self._call_store("create_profile", self.authority),
            task_name="cayu-browser-profile-initialization",
        )
        self._require_current_dependencies()
        owned_created = _own_profile_extension_model(created, BrowserProfileRef)
        if (
            owned_created.profile_id != self._authority.profile_id
            or owned_created.authority_fingerprint != self._authority.fingerprint
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile initialization has conflicting authority."
            )
        return owned_created

    def _require_current_dependencies(self) -> None:
        """Fail closed when mutable extension authorities drift after binding."""

        self._require_current_store_dependency()
        self._require_current_key_dependency()

    def _require_current_store_dependency(self) -> None:
        """Authenticate the store used by mutation and settlement paths."""

        store_id = _profile_dependency_identity(self.store, "id")
        if store_id != self._authority.store_id:
            raise BrowserProfileUnavailable(
                "Browser-profile dependency authority identity changed."
            )

    def _require_current_key_dependency(self) -> None:
        """Authenticate the key authority before private cryptographic work."""

        key_authority_id = _profile_dependency_identity(
            self.key_authority,
            "authority_id",
        )
        if key_authority_id != self._authority.key_authority_id:
            raise BrowserProfileUnavailable(
                "Browser-profile dependency authority identity changed."
            )

    async def _call_store(
        self,
        method_name: str,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call one store method while treating public overrides as untrusted."""

        inherited_contract = False
        store_failure: BaseException | None = None
        try:
            method = getattr(self.store, method_name)
            inherited_contract = getattr(type(self.store), method_name, None) is getattr(
                BrowserProfileStore,
                method_name,
                None,
            )
            return await method(*args, **kwargs)
        except BaseException as failure:
            store_failure = failure
        if store_failure is None:  # pragma: no cover - exception branch invariant
            raise BrowserProfileUnavailable("Browser-profile store operation failed.")
        _clear_inactive_profile_failure_frames(store_failure)
        if _profile_failure_contains_process_control(store_failure) or isinstance(
            store_failure,
            asyncio.CancelledError,
        ):
            raise store_failure from None
        if inherited_contract and type(store_failure) in {
            BrowserProfileStoreConflict,
            BrowserProfileUnavailable,
        }:
            raise store_failure from None
        if (
            inherited_contract
            and type(store_failure) is ValueError
            and str(store_failure) == "revoked_at cannot be in the future."
        ):
            raise ValueError("revoked_at cannot be in the future.") from None
        store_failure = None
        raise BrowserProfileUnavailable("Browser-profile store operation failed.") from None

    def _snapshot_material(
        self,
        material: BrowserProfileRestoreMaterial,
    ) -> tuple[BrowserProfileRestorePreparation, BrowserProfileStateV1]:
        """Authenticate one process-local material handle before extension activity."""

        if type(material) is not BrowserProfileRestoreMaterial:
            material = None  # ty: ignore[invalid-assignment]
            raise TypeError("material must be an exact BrowserProfileRestoreMaterial instance.")
        preparation, state = material._snapshot()
        request = preparation.request
        if (
            request.access != self._access
            or request.current_policy_fingerprint != self._current_policy.fingerprint
            or request.expected_ref != self._expected_ref
        ):
            preparation = None
            state = BrowserProfileStateV1()
            raise BrowserProfileStoreConflict(
                "Browser-profile restore material has conflicting binding authority."
            )
        return preparation, state

    def _snapshot_plan(
        self,
        plan: BrowserProfileCheckpointPlan,
    ) -> tuple[
        BrowserProfileCheckpointReservation,
        BrowserProfileRestoreMaterial,
        BrowserProfileRestorePreparation,
        BrowserProfileStateV1,
    ]:
        """Authenticate the full reserved checkpoint authority before plaintext work."""

        if type(plan) is not BrowserProfileCheckpointPlan:
            plan = None  # ty: ignore[invalid-assignment]
            raise TypeError("plan must be an exact BrowserProfileCheckpointPlan instance.")
        reservation, material, preparation, state = plan._snapshot()
        request = reservation.request
        if (
            request.access != self._access
            or request.current_policy_fingerprint != self._current_policy.fingerprint
            or request.writer_claim != preparation.writer_claim
            or preparation.request.access != self._access
            or preparation.request.current_policy_fingerprint != self._current_policy.fingerprint
            or preparation.request.expected_ref != self._expected_ref
        ):
            reservation = None
            material = None
            preparation = None
            state = BrowserProfileStateV1()
            raise BrowserProfileStoreConflict(
                "Browser-profile checkpoint plan has conflicting binding authority."
            )
        return reservation, material, preparation, state

    def _own_restore_preparation(
        self,
        value: object,
        *,
        expected_request: BrowserProfileRestoreRequest | None = None,
        browser_session_id: str | None = None,
        execution_profile_fingerprint: str | None = None,
        allocation_fingerprint: str | None = None,
    ) -> BrowserProfileRestorePreparation:
        owned = _own_profile_extension_model(
            value,
            BrowserProfileRestorePreparation,
        )
        request = owned.request
        if (
            request.access != self._access
            or request.current_policy_fingerprint != self._current_policy.fingerprint
            or request.expected_ref != self._expected_ref
            or (expected_request is not None and request != expected_request)
            or (browser_session_id is not None and request.browser_session_id != browser_session_id)
            or (
                execution_profile_fingerprint is not None
                and request.execution_profile_fingerprint != execution_profile_fingerprint
            )
            or (
                allocation_fingerprint is not None
                and request.allocation_fingerprint != allocation_fingerprint
            )
            or (
                owned.envelope is not None
                and (
                    owned.envelope.authority_fingerprint != self._authority.fingerprint
                    or owned.envelope.key_authority_id != self._authority.key_authority_id
                    or owned.envelope.store_id != self._authority.store_id
                )
            )
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile restore preparation has conflicting authority."
            )
        return owned

    def _own_restore_receipt(
        self,
        value: object,
        *,
        request: BrowserProfileRestoreRequest,
        outcome: BrowserProfileTerminalOutcome,
        error_code: str | None,
        writer_claim: BrowserProfileWriterClaim | None = None,
    ) -> BrowserProfileRestoreReceipt:
        owned = _own_profile_extension_model(value, BrowserProfileRestoreReceipt)
        if (
            owned.operation_id != request.operation_id
            or owned.request_fingerprint != request.fingerprint()
            or owned.profile_ref.profile_id != self._authority.profile_id
            or owned.profile_ref.authority_fingerprint != self._authority.fingerprint
            or owned.current_policy_fingerprint != request.current_policy_fingerprint
            or owned.execution_profile_fingerprint != request.execution_profile_fingerprint
            or owned.allocation_fingerprint != request.allocation_fingerprint
            or owned.browser_session_id != request.browser_session_id
            or owned.outcome is not outcome
            or owned.error_code != error_code
            or (
                writer_claim is not None
                and (
                    owned.writer_fence != writer_claim.fence
                    or owned.profile_ref.generation != writer_claim.generation
                    or owned.profile_ref.content_fingerprint != writer_claim.content_fingerprint
                )
            )
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile restore receipt has conflicting authority."
            )
        return owned

    def _own_checkpoint_reservation(
        self,
        value: object,
        *,
        request: BrowserProfileCheckpointRequest,
    ) -> BrowserProfileCheckpointReservation:
        owned = _own_profile_extension_model(
            value,
            BrowserProfileCheckpointReservation,
        )
        if (
            owned.request != request
            or owned.request_fingerprint != request.fingerprint()
            or owned.reserved_ciphertext_bytes != self._limits.max_ciphertext_bytes
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile checkpoint reservation has conflicting authority."
            )
        return owned

    def _own_checkpoint_receipt(
        self,
        value: object,
        *,
        reservation: BrowserProfileCheckpointReservation,
        outcome: BrowserProfileTerminalOutcome | None = None,
        error_code: str | None = None,
        counts: tuple[int, int, int] | None = None,
    ) -> BrowserProfileCheckpointReceipt:
        owned = _own_profile_extension_model(value, BrowserProfileCheckpointReceipt)
        request = reservation.request
        if (
            owned.operation_id != request.operation_id
            or owned.request_fingerprint != reservation.request_fingerprint
            or owned.previous_ref
            != BrowserProfileRef(
                profile_id=request.writer_claim.profile_id,
                authority_fingerprint=request.writer_claim.authority_fingerprint,
                generation=request.writer_claim.generation,
                content_fingerprint=request.writer_claim.content_fingerprint,
            )
            or owned.current_policy_fingerprint != request.current_policy_fingerprint
            or owned.execution_profile_fingerprint
            != request.writer_claim.execution_profile_fingerprint
            or owned.allocation_fingerprint != request.writer_claim.allocation_fingerprint
            or owned.browser_session_id != request.writer_claim.browser_session_id
            or owned.writer_fence != request.writer_claim.fence
            or owned.source_revision != request.source_revision
            or owned.source_operation_receipt_id != request.source_operation_receipt_id
            or owned.source_operation_fingerprint != request.source_operation_fingerprint
            or owned.ambiguous_lineage != request.ambiguous_lineage
            or (outcome is not None and owned.outcome is not outcome)
            or (outcome is not None and owned.error_code != error_code)
            or (
                counts is not None
                and (
                    owned.origin_count,
                    owned.cookie_count,
                    owned.storage_entry_count,
                )
                != counts
            )
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile checkpoint receipt has conflicting authority."
            )
        return owned

    def execution_profile_material(self) -> dict[str, object]:
        return {
            "schema_version": BROWSER_PROFILE_SCHEMA_VERSION,
            "profile_id": self.authority.profile_id,
            "authority_fingerprint": self.authority.fingerprint,
            "owner_fingerprint": self.authority.scope.owner_fingerprint,
            "sharing_fingerprint": self.authority.scope.sharing_fingerprint,
            "recorded_destination_policy_fingerprint": (
                self.authority.destination_policy.fingerprint
            ),
            "current_destination_policy_fingerprint": self.current_policy.fingerprint,
            "browser_protocol": self.authority.browser_protocol,
            "browser_worker_version": self.authority.browser_worker_version,
            "state_schema_version": self.authority.state_schema_version,
            "key_authority_id": self.authority.key_authority_id,
            "store_id": self.authority.store_id,
            "checkpoint_policy": self.checkpoint_policy.value,
            "lease_seconds": self.lease_seconds,
            "expected_generation": (
                None if self.expected_ref is None else self.expected_ref.generation
            ),
            "expected_content_fingerprint": (
                None if self.expected_ref is None else self.expected_ref.content_fingerprint
            ),
            "limits": self.limits.model_dump(mode="json"),
        }

    async def prepare_restore(
        self,
        *,
        operation_id: str,
        execution_profile_fingerprint: str,
        allocation_fingerprint: str,
        browser_session_id: str,
    ) -> BrowserProfileRestoreMaterial:
        validation_failure: BaseException | None = None
        owned_operation_id: str | None = None
        owned_execution: str | None = None
        owned_allocation: str | None = None
        owned_browser_session_id: str | None = None
        try:
            owned_operation_id = _identifier(operation_id, "operation_id")
            owned_execution = _digest(
                execution_profile_fingerprint,
                "execution_profile_fingerprint",
            )
            owned_allocation = _digest(
                allocation_fingerprint,
                "allocation_fingerprint",
            )
            owned_browser_session_id = _identifier(
                browser_session_id,
                "browser_session_id",
            )
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        operation_id = None  # ty: ignore[invalid-assignment]
        execution_profile_fingerprint = None  # ty: ignore[invalid-assignment]
        allocation_fingerprint = None  # ty: ignore[invalid-assignment]
        browser_session_id = None  # ty: ignore[invalid-assignment]
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            raise ValueError("Browser-profile restore identity is invalid.") from None
        if (
            owned_operation_id is None
            or owned_execution is None
            or owned_allocation is None
            or owned_browser_session_id is None
        ):  # pragma: no cover - paired validation
            raise ValueError("Browser-profile restore identity is invalid.")
        writer_id = (
            "bpw_"
            + sha256(
                b"cayu.browser-profile.writer.v1\0"
                + self._authority.profile_id.encode("utf-8")
                + b"\0"
                + owned_allocation.encode("ascii")
                + b"\0"
                + owned_browser_session_id.encode("utf-8")
            ).hexdigest()
        )
        request = BrowserProfileRestoreRequest(
            operation_id=owned_operation_id,
            access=self._access,
            expected_ref=self._expected_ref,
            current_policy_fingerprint=self._current_policy.fingerprint,
            execution_profile_fingerprint=owned_execution,
            allocation_fingerprint=owned_allocation,
            browser_session_id=owned_browser_session_id,
            writer_id=writer_id,
        )
        self._require_current_dependencies()

        async def acquire() -> BrowserProfileRestorePreparation:
            try:
                prepared = await self._call_store(
                    "prepare_restore",
                    request,
                    lease_seconds=self.lease_seconds,
                )
            except BaseException as failure:
                if _profile_failure_contains_process_control(failure):
                    raise
                # The store mutation is exact and idempotent.  A second call
                # is both the readback path for commit-then-raise stores and a
                # bounded retry for a transient pre-commit failure.
                try:
                    prepared = await self._call_store(
                        "prepare_restore",
                        request,
                        lease_seconds=self.lease_seconds,
                    )
                except BaseException as reconciliation_failure:
                    if _profile_failure_contains_process_control(reconciliation_failure):
                        safe_failure = _safe_profile_extension_failure(failure)
                        reconciliation_failure.add_note(
                            "Browser-profile restore preparation first failed before "
                            "process control interrupted reconciliation."
                        )
                        raise reconciliation_failure from safe_failure
                    failure.add_note(
                        "Browser-profile restore preparation reconciliation also failed."
                    )
                    raise failure from reconciliation_failure
            return self._own_restore_preparation(
                prepared,
                expected_request=request,
            )

        acquisition = asyncio.create_task(
            acquire(),
            name="cayu-browser-profile-restore-preparation",
        )
        acquisition_outcome = await await_shielded_task_outcome(acquisition)
        if acquisition_outcome.cancellation is not None:
            settlement_outcome = None
            cancellation = acquisition_outcome.cancellation
            consumed_requests = acquisition_outcome.cancellation_requests_consumed
            settlement_error = acquisition_outcome.error
            if acquisition_outcome.error is None and acquisition_outcome.result is not None:
                settlement = asyncio.create_task(
                    self._complete_restore_request(
                        request,
                        outcome=BrowserProfileTerminalOutcome.FAILED,
                        error_code="restore_cancelled_before_import",
                        writer_claim=acquisition_outcome.result.writer_claim,
                    ),
                    name="cayu-browser-profile-cancelled-preparation-settlement",
                )
                settlement_outcome = await await_shielded_task_outcome(
                    settlement,
                    cancellation=cancellation,
                )
                cancellation = settlement_outcome.cancellation or cancellation
                consumed_requests += settlement_outcome.cancellation_requests_consumed
                settlement_error = settlement_outcome.error
            restore_task_cancellation_requests(
                consumed_requests,
                cancellation=cancellation,
            )
            if settlement_error is not None:
                safe_settlement_error = _safe_profile_extension_failure(settlement_error)
                cancellation.add_note("Browser-profile restore preparation settlement also failed.")
                acquisition = None
                acquisition_outcome = None
                settlement_outcome = None
                del settlement_error
                raise cancellation from safe_settlement_error
            acquisition = None
            acquisition_outcome = None
            settlement_outcome = None
            raise cancellation
        if acquisition_outcome.error is not None:
            safe_error = _safe_profile_extension_failure(acquisition_outcome.error)
            acquisition = None
            acquisition_outcome = None
            raise safe_error from None
        preparation = acquisition_outcome.result
        if preparation is None:  # pragma: no cover - successful store contract
            raise BrowserProfileUnavailable(
                "Browser-profile store returned no restore preparation."
            )
        return await self._restore_material(preparation)

    async def _restore_material(
        self,
        preparation: BrowserProfileRestorePreparation,
    ) -> BrowserProfileRestoreMaterial:
        self._require_current_dependencies()
        preparation = self._own_restore_preparation(preparation)
        request = preparation.request
        writer_claim = preparation.writer_claim
        if (
            preparation.existing_receipt is not None
            and preparation.existing_receipt.outcome is not BrowserProfileTerminalOutcome.SUCCEEDED
        ):
            raise BrowserProfileUnavailable("Browser-profile restore already failed.")
        envelope = preparation.envelope
        if envelope is None:
            state = BrowserProfileStateV1()
        else:
            owned_envelope = envelope
            if (
                owned_envelope.authority_fingerprint != self.authority.fingerprint
                or owned_envelope.key_authority_id != self._authority.key_authority_id
                or owned_envelope.store_id != self._authority.store_id
                or owned_envelope.generation != preparation.profile_ref.generation
                or owned_envelope.content_fingerprint != preparation.profile_ref.content_fingerprint
                or owned_envelope.plaintext_bytes > self.limits.max_plaintext_bytes
                or owned_envelope.ciphertext_bytes > self.limits.max_ciphertext_bytes
            ):
                envelope = None
                preparation = None  # ty: ignore[invalid-assignment]
                del owned_envelope
                await self._fail_restore_material(
                    request,
                    writer_claim=writer_claim,
                    error_code="profile_incompatible",
                    primary=BrowserProfileUnavailable("Browser profile is incompatible."),
                )
            aad = canonical_durable_json_bytes(
                owned_envelope.aad_material(self.authority),
                "browser profile authenticated data",
            )
            plaintext = b""
            current_task = asyncio.current_task()
            cancellation_requests_before_decrypt = (
                0 if current_task is None else current_task.cancelling()
            )
            cancellation_pending_before_decrypt = bool(
                current_task is not None and getattr(current_task, "_must_cancel", False)
            )
            decrypt_error_code: str | None = None
            decrypt_primary: BaseException | None = None
            decrypt_cancellation: asyncio.CancelledError | None = None
            process_control_failure: BaseException | None = None

            async def decrypt_profile(
                encrypted: BrowserProfileEncryptedEnvelope,
            ) -> bytes:
                async with asyncio.timeout(self.limits.import_timeout_seconds):
                    return await self.key_authority.decrypt(
                        nonce=encrypted.nonce(),
                        ciphertext=encrypted.ciphertext(),
                        aad=aad,
                    )

            decryption = decrypt_profile(owned_envelope)
            del decrypt_profile
            try:
                plaintext = await _settle_profile_task(
                    decryption,
                    task_name="cayu-browser-profile-decryption",
                    preserve_errors=(InvalidTag, TimeoutError, TypeError, ValueError),
                )
                if type(plaintext) is not bytes:
                    raise TypeError("Browser-profile decryption returned invalid output.")
                if len(plaintext) != owned_envelope.plaintext_bytes:
                    raise ValueError("Browser-profile plaintext length is inconsistent.")
                state = BrowserProfileStateV1.model_validate_json(plaintext)
            except asyncio.CancelledError as cancellation:
                caller_cancellation = current_task is not None and (
                    cancellation_pending_before_decrypt
                    or current_task.cancelling() > cancellation_requests_before_decrypt
                )
                decrypt_error_code = (
                    "restore_cancelled_before_import"
                    if caller_cancellation
                    else "profile_unavailable"
                )
                decrypt_primary = (
                    cancellation
                    if caller_cancellation
                    else BrowserProfileUnavailable(
                        "Browser-profile decryption was cancelled unexpectedly."
                    )
                )
                decrypt_cancellation = cancellation if caller_cancellation else None
                if not caller_cancellation:
                    _clear_inactive_profile_failure_frames(cancellation)
            except (InvalidTag, TimeoutError, TypeError, ValueError) as failure:
                _clear_inactive_profile_failure_frames(failure)
                decrypt_error_code = "profile_corrupt"
                decrypt_primary = BrowserProfileUnavailable("Browser profile cannot be decrypted.")
            except Exception as failure:
                _clear_inactive_profile_failure_frames(failure)
                decrypt_error_code = "profile_unavailable"
                decrypt_primary = BrowserProfileUnavailable("Browser profile cannot be decrypted.")
            except BaseException as failure:
                _clear_inactive_profile_failure_frames(failure)
                process_control_failure = failure
            if process_control_failure is not None:
                decryption = None
                plaintext = b""
                state = BrowserProfileStateV1()
                aad = b""
                envelope = None
                preparation = None  # ty: ignore[invalid-assignment]
                del owned_envelope
                raise process_control_failure from None
            if decrypt_error_code is not None:
                decryption = None
                plaintext = b""
                state = BrowserProfileStateV1()
                aad = b""
                envelope = None
                preparation = None  # ty: ignore[invalid-assignment]
                del owned_envelope
                if decrypt_primary is None:  # pragma: no cover - paired classification
                    raise BrowserProfileUnavailable("Browser profile cannot be decrypted.")
                await self._fail_restore_material(
                    request,
                    writer_claim=writer_claim,
                    error_code=decrypt_error_code,
                    primary=decrypt_primary,
                    cancellation=decrypt_cancellation,
                )
            decryption = None
            plaintext = b""
            aad = b""
            del owned_envelope
            self._require_current_store_dependency()
            try:
                self._require_current_key_dependency()
            except BrowserProfileUnavailable:
                state = BrowserProfileStateV1()
                envelope = None
                preparation = None  # ty: ignore[invalid-assignment]
                await self._fail_restore_material(
                    request,
                    writer_claim=writer_claim,
                    error_code="profile_unavailable",
                    primary=BrowserProfileUnavailable("Browser profile cannot be decrypted."),
                )
            try:
                state = validate_browser_profile_state(
                    state,
                    limits=self.limits,
                    current_policy=self.current_policy,
                )
            except (TypeError, ValueError):
                state = BrowserProfileStateV1()
                preparation = None  # ty: ignore[invalid-assignment]
                await self._fail_restore_material(
                    request,
                    writer_claim=writer_claim,
                    error_code="profile_incompatible",
                    primary=BrowserProfileUnavailable(
                        "Browser profile exceeds current destination authority."
                    ),
                )
        return BrowserProfileRestoreMaterial(preparation=preparation, state=state)

    async def _fail_restore_material(
        self,
        request: BrowserProfileRestoreRequest,
        *,
        writer_claim: BrowserProfileWriterClaim,
        error_code: str,
        primary: BaseException,
        cancellation: asyncio.CancelledError | None = None,
    ) -> Never:
        """Settle rejected private material without replacing its primary signal."""

        settlement = self._complete_restore_request(
            request,
            outcome=BrowserProfileTerminalOutcome.FAILED,
            error_code=error_code,
            writer_claim=writer_claim,
        )
        try:
            await _settle_profile_task(
                settlement,
                task_name="cayu-browser-profile-restore-material-failure-settlement",
                cancellation=cancellation,
            )
        except BaseException as settlement_failure:
            if cancellation is not None:
                # The settlement boundary restores the exact caller request and
                # keeps it authoritative over ordinary cleanup failure. Process
                # control remains authoritative through the same boundary.
                raise
            if _profile_failure_contains_process_control(settlement_failure):
                settlement_failure.add_note(
                    "Browser-profile restore material was rejected before process control."
                )
                raise settlement_failure from primary
            primary.add_note("Browser-profile restore failure settlement also failed.")
            raise primary from settlement_failure
        raise primary from None

    async def complete_restore(
        self,
        material: BrowserProfileRestoreMaterial,
        *,
        outcome: BrowserProfileTerminalOutcome,
        error_code: str | None = None,
    ) -> BrowserProfileRestoreReceipt:
        preparation, _private_state = self._snapshot_material(material)
        _private_state = BrowserProfileStateV1()
        self._require_current_store_dependency()
        return await self._complete_restore_request(
            preparation.request,
            outcome=outcome,
            error_code=error_code,
            writer_claim=preparation.writer_claim,
        )

    async def _complete_restore_request(
        self,
        request: BrowserProfileRestoreRequest,
        *,
        outcome: BrowserProfileTerminalOutcome,
        error_code: str | None,
        writer_claim: BrowserProfileWriterClaim,
    ) -> BrowserProfileRestoreReceipt:
        if type(outcome) is not BrowserProfileTerminalOutcome:
            raise TypeError("outcome must be a BrowserProfileTerminalOutcome.")
        safe_error = (
            None
            if error_code is None
            else _fixed_error_code(
                error_code,
                "error_code",
                allowed=_RESTORE_ERROR_CODES,
            )
        )
        if (outcome is BrowserProfileTerminalOutcome.SUCCEEDED) == (safe_error is not None):
            raise ValueError("Browser-profile restore outcome contradicts its error code.")

        async def settle() -> BrowserProfileRestoreReceipt:
            try:
                receipt = await self._call_store(
                    "complete_restore",
                    request,
                    outcome=outcome,
                    error_code=safe_error,
                )
                return self._own_restore_receipt(
                    receipt,
                    request=request,
                    outcome=outcome,
                    error_code=safe_error,
                    writer_claim=writer_claim,
                )
            except BaseException as failure:
                if _profile_failure_contains_process_control(failure):
                    raise
                try:
                    existing = await self._call_store(
                        "load_restore_receipt",
                        self.access,
                        request.operation_id,
                    )
                except BaseException as reconciliation_failure:
                    if _profile_failure_contains_process_control(reconciliation_failure):
                        safe_failure = _safe_profile_extension_failure(failure)
                        reconciliation_failure.add_note(
                            "Browser-profile restore settlement first failed before "
                            "process control interrupted reconciliation."
                        )
                        raise reconciliation_failure from safe_failure
                    raise failure from reconciliation_failure
                if existing is not None:
                    owned_existing = self._own_restore_receipt(
                        existing,
                        request=request,
                        outcome=outcome,
                        error_code=safe_error,
                        writer_claim=writer_claim,
                    )
                    if (
                        owned_existing.outcome is outcome
                        and owned_existing.error_code == safe_error
                    ):
                        return owned_existing
                raise

        return await _settle_profile_task(
            settle(),
            task_name="cayu-browser-profile-restore-settlement",
        )

    async def resume_writer(
        self,
        *,
        execution_profile_fingerprint: str,
        allocation_fingerprint: str,
        browser_session_id: str,
    ) -> BrowserProfileRestoreMaterial:
        validation_failure: BaseException | None = None
        session_id: str | None = None
        execution: str | None = None
        allocation: str | None = None
        try:
            session_id = _identifier(browser_session_id, "browser_session_id")
            execution = _digest(
                execution_profile_fingerprint,
                "execution_profile_fingerprint",
            )
            allocation = _digest(allocation_fingerprint, "allocation_fingerprint")
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        browser_session_id = None  # ty: ignore[invalid-assignment]
        execution_profile_fingerprint = None  # ty: ignore[invalid-assignment]
        allocation_fingerprint = None  # ty: ignore[invalid-assignment]
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            raise ValueError("Browser-profile writer identity is invalid.") from None
        if session_id is None or execution is None or allocation is None:
            raise ValueError("Browser-profile writer identity is invalid.")
        self._require_current_dependencies()
        raw_preparation = await _settle_profile_task(
            self._call_store(
                "resume_writer",
                self._access,
                browser_session_id=session_id,
                execution_profile_fingerprint=execution,
                allocation_fingerprint=allocation,
                lease_seconds=self.lease_seconds,
            ),
            task_name="cayu-browser-profile-writer-reconstruction",
        )
        self._require_current_dependencies()
        preparation = self._own_restore_preparation(
            raw_preparation,
            browser_session_id=session_id,
            execution_profile_fingerprint=execution,
            allocation_fingerprint=allocation,
        )
        if preparation.existing_receipt is None:
            return await self._restore_material(preparation)
        return BrowserProfileRestoreMaterial(
            preparation=preparation,
            state=BrowserProfileStateV1(),
        )

    async def renew_writer(
        self,
        material: BrowserProfileRestoreMaterial,
    ) -> BrowserProfileWriterClaim:
        preparation, _private_state = self._snapshot_material(material)
        _private_state = BrowserProfileStateV1()
        self._require_current_dependencies()
        renewed = await _settle_profile_task(
            self._call_store(
                "renew_writer",
                self._access,
                preparation.writer_claim,
                lease_seconds=self.lease_seconds,
            ),
            task_name="cayu-browser-profile-writer-renewal",
        )
        self._require_current_dependencies()
        renewed = _own_profile_extension_model(renewed, BrowserProfileWriterClaim)
        if (
            not _same_writer_lineage(renewed, preparation.writer_claim)
            or renewed.generation != preparation.writer_claim.generation
            or renewed.content_fingerprint != preparation.writer_claim.content_fingerprint
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile writer renewal has conflicting authority."
            )
        material._replace_claim(renewed)
        return renewed

    async def reserve_checkpoint(
        self,
        *,
        material: BrowserProfileRestoreMaterial,
        operation_id: str,
        source_revision: str,
        source_operation_receipt_id: str,
        source_operation_fingerprint: str,
        ambiguous_lineage: bool,
    ) -> BrowserProfileCheckpointPlan:
        preparation, _private_state = self._snapshot_material(material)
        _private_state = BrowserProfileStateV1()
        validation_failure: BaseException | None = None
        request: BrowserProfileCheckpointRequest | None = None
        try:
            request = BrowserProfileCheckpointRequest(
                operation_id=operation_id,
                access=self._access,
                writer_claim=preparation.writer_claim,
                source_revision=source_revision,
                source_operation_receipt_id=source_operation_receipt_id,
                source_operation_fingerprint=source_operation_fingerprint,
                current_policy_fingerprint=self._current_policy.fingerprint,
                ambiguous_lineage=ambiguous_lineage,
            )
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        operation_id = None  # ty: ignore[invalid-assignment]
        source_revision = None  # ty: ignore[invalid-assignment]
        source_operation_receipt_id = None  # ty: ignore[invalid-assignment]
        source_operation_fingerprint = None  # ty: ignore[invalid-assignment]
        ambiguous_lineage = None  # ty: ignore[invalid-assignment]
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            request = None
            raise ValueError("Browser-profile checkpoint identity is invalid.") from None
        if request is None:  # pragma: no cover - paired validation
            raise ValueError("Browser-profile checkpoint identity is invalid.")
        self._require_current_dependencies()

        async def acquire() -> BrowserProfileCheckpointReservation:
            try:
                reserved = await self._call_store(
                    "reserve_checkpoint",
                    request,
                    reserved_ciphertext_bytes=self.limits.max_ciphertext_bytes,
                )
            except BaseException as failure:
                if _profile_failure_contains_process_control(failure):
                    raise
                try:
                    reserved = await self._call_store(
                        "reserve_checkpoint",
                        request,
                        reserved_ciphertext_bytes=self.limits.max_ciphertext_bytes,
                    )
                except BaseException as reconciliation_failure:
                    if _profile_failure_contains_process_control(reconciliation_failure):
                        safe_failure = _safe_profile_extension_failure(failure)
                        reconciliation_failure.add_note(
                            "Browser-profile checkpoint reservation first failed before "
                            "process control interrupted reconciliation."
                        )
                        raise reconciliation_failure from safe_failure
                    failure.add_note(
                        "Browser-profile checkpoint reservation reconciliation also failed."
                    )
                    raise failure from reconciliation_failure
            return self._own_checkpoint_reservation(
                reserved,
                request=request,
            )

        acquisition = asyncio.create_task(
            acquire(),
            name="cayu-browser-profile-checkpoint-reservation",
        )
        acquisition_outcome = await await_shielded_task_outcome(acquisition)
        if acquisition_outcome.cancellation is not None:
            settlement_outcome = None
            cancellation = acquisition_outcome.cancellation
            consumed_requests = acquisition_outcome.cancellation_requests_consumed
            settlement_error = acquisition_outcome.error
            reservation = acquisition_outcome.result
            if acquisition_outcome.error is None and reservation is not None:
                owned_reservation = reservation

                async def settle_cancelled_reservation() -> BrowserProfileCheckpointReceipt:
                    receipt = await self._call_store(
                        "fail_checkpoint",
                        owned_reservation,
                        outcome=BrowserProfileTerminalOutcome.FAILED,
                        error_code="checkpoint_cancelled_before_export",
                    )
                    return self._own_checkpoint_receipt(
                        receipt,
                        reservation=owned_reservation,
                        outcome=BrowserProfileTerminalOutcome.FAILED,
                        error_code="checkpoint_cancelled_before_export",
                        counts=(
                            owned_reservation.previous_origin_count,
                            owned_reservation.previous_cookie_count,
                            owned_reservation.previous_storage_entry_count,
                        ),
                    )

                settlement = asyncio.create_task(
                    settle_cancelled_reservation(),
                    name="cayu-browser-profile-cancelled-reservation-settlement",
                )
                settlement_outcome = await await_shielded_task_outcome(
                    settlement,
                    cancellation=cancellation,
                )
                cancellation = settlement_outcome.cancellation or cancellation
                consumed_requests += settlement_outcome.cancellation_requests_consumed
                settlement_error = settlement_outcome.error
            restore_task_cancellation_requests(
                consumed_requests,
                cancellation=cancellation,
            )
            if settlement_error is not None:
                safe_settlement_error = _safe_profile_extension_failure(settlement_error)
                cancellation.add_note(
                    "Browser-profile checkpoint reservation settlement also failed."
                )
                acquisition = None
                acquisition_outcome = None
                reservation = None
                settlement_outcome = None
                del settlement_error
                raise cancellation from safe_settlement_error
            acquisition = None
            acquisition_outcome = None
            reservation = None
            settlement_outcome = None
            raise cancellation
        if acquisition_outcome.error is not None:
            safe_error = _safe_profile_extension_failure(acquisition_outcome.error)
            acquisition = None
            acquisition_outcome = None
            raise safe_error from None
        reservation = acquisition_outcome.result
        if reservation is None:  # pragma: no cover - successful store contract
            raise BrowserProfileUnavailable(
                "Browser-profile store returned no checkpoint reservation."
            )
        return BrowserProfileCheckpointPlan(reservation, material)

    async def fail_checkpoint(
        self,
        plan: BrowserProfileCheckpointPlan,
        *,
        outcome: BrowserProfileTerminalOutcome,
        error_code: str,
    ) -> BrowserProfileCheckpointReceipt:
        reservation, _material, _preparation, _private_state = self._snapshot_plan(plan)
        _private_state = BrowserProfileStateV1()
        self._require_current_store_dependency()
        if type(outcome) is not BrowserProfileTerminalOutcome:
            raise TypeError("outcome must be a BrowserProfileTerminalOutcome.")
        if outcome is BrowserProfileTerminalOutcome.SUCCEEDED:
            raise ValueError("Failed checkpoint settlement requires a failure outcome.")
        safe_error = _fixed_error_code(
            error_code,
            "error_code",
            allowed=_CHECKPOINT_ERROR_CODES,
        )
        receipt = await _settle_profile_task(
            self._call_store(
                "fail_checkpoint",
                reservation,
                outcome=outcome,
                error_code=safe_error,
            ),
            task_name="cayu-browser-profile-checkpoint-failure-settlement",
        )
        return self._own_checkpoint_receipt(
            receipt,
            reservation=reservation,
            outcome=outcome,
            error_code=safe_error,
            counts=(
                reservation.previous_origin_count,
                reservation.previous_cookie_count,
                reservation.previous_storage_entry_count,
            ),
        )

    async def reconcile_checkpoint(
        self,
        operation_id: str,
        *,
        material: BrowserProfileRestoreMaterial | None = None,
    ) -> BrowserProfileCheckpointReceipt | None:
        """Reconcile publication and optionally settle the live material's adoption.

        When supplied, material adopts the matching writer, encrypted envelope,
        and decrypted state before caller cancellation is propagated.
        """

        if material is not None:
            # Keep receipt reconciliation and local adoption in the same owner:
            # pending caller cancellation must not separate these two steps.
            async def reconcile_and_adopt() -> BrowserProfileCheckpointReceipt | None:
                receipt = await self.reconcile_checkpoint(operation_id)
                if (
                    receipt is not None
                    and receipt.outcome is BrowserProfileTerminalOutcome.SUCCEEDED
                ):
                    await self._adopt_checkpoint(material, receipt)
                return receipt

            return await _settle_profile_task(
                reconcile_and_adopt(),
                task_name="cayu-browser-profile-checkpoint-reconciliation-adoption",
            )
        validation_failure: BaseException | None = None
        owned_operation_id: str | None = None
        try:
            owned_operation_id = _identifier(operation_id, "operation_id")
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        operation_id = None  # ty: ignore[invalid-assignment]
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            raise ValueError("Browser-profile checkpoint operation identity is invalid.") from None
        if owned_operation_id is None:  # pragma: no cover - paired validation
            raise ValueError("Browser-profile checkpoint operation identity is invalid.")
        self._require_current_store_dependency()
        receipt = await _settle_profile_task(
            self._call_store(
                "reconcile_checkpoint",
                self._access,
                owned_operation_id,
            ),
            task_name="cayu-browser-profile-checkpoint-reconciliation",
        )
        if receipt is None:
            return None
        owned_receipt = _own_profile_extension_model(
            receipt,
            BrowserProfileCheckpointReceipt,
        )
        if (
            owned_receipt.operation_id != owned_operation_id
            or owned_receipt.previous_ref.profile_id != self._authority.profile_id
            or owned_receipt.previous_ref.authority_fingerprint != self._authority.fingerprint
            or owned_receipt.current_policy_fingerprint != self._current_policy.fingerprint
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile checkpoint reconciliation has conflicting authority."
            )
        return owned_receipt

    async def _adopt_checkpoint(
        self,
        material: BrowserProfileRestoreMaterial,
        receipt: BrowserProfileCheckpointReceipt,
    ) -> None:
        """Adopt a reconciled generation with its exact live writer and envelope."""

        preparation, _state = self._snapshot_material(material)
        receipt = _own_profile_extension_model(receipt, BrowserProfileCheckpointReceipt)
        if (
            receipt.outcome is not BrowserProfileTerminalOutcome.SUCCEEDED
            or receipt.published_ref.profile_id != preparation.profile_ref.profile_id
            or receipt.published_ref.authority_fingerprint
            != preparation.profile_ref.authority_fingerprint
            or receipt.current_policy_fingerprint != self._current_policy.fingerprint
        ):
            raise BrowserProfileStoreConflict("Browser-profile adoption authority conflicts.")
        if preparation.profile_ref.generation >= receipt.published_ref.generation:
            if (
                preparation.profile_ref.generation == receipt.published_ref.generation
                and preparation.profile_ref != receipt.published_ref
            ):
                raise BrowserProfileStoreConflict("Browser-profile adoption content conflicts.")
            return
        if preparation.profile_ref != receipt.previous_ref:
            raise BrowserProfileStoreConflict("Browser-profile adoption lineage conflicts.")

        async def adopt() -> None:
            restored = await self.resume_writer(
                execution_profile_fingerprint=preparation.writer_claim.execution_profile_fingerprint,
                allocation_fingerprint=preparation.writer_claim.allocation_fingerprint,
                browser_session_id=preparation.writer_claim.browser_session_id,
            )
            current = restored.preparation
            if current.profile_ref != receipt.published_ref:
                raise BrowserProfileStoreConflict("Browser-profile adoption generation conflicts.")
            decrypted = await self._restore_material(current)
            material._replace_claim(
                current.writer_claim,
                profile_ref=current.profile_ref,
                envelope=current.envelope,
                state=decrypted.state,
            )

        await _settle_profile_task(adopt(), task_name="cayu-browser-profile-checkpoint-adoption")

    async def publish_checkpoint(
        self,
        plan: BrowserProfileCheckpointPlan,
        state: BrowserProfileStateV1,
    ) -> BrowserProfileCheckpointReceipt:
        reservation, material, _preparation, _prior_state = self._snapshot_plan(plan)
        validation_failure: BaseException | None = None
        owned_state: BrowserProfileStateV1 | None = None
        try:
            owned_state = validate_browser_profile_state(
                state,
                limits=self._limits,
                current_policy=self._current_policy,
            )
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        state = None  # ty: ignore[invalid-assignment]
        _prior_state = BrowserProfileStateV1()
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            owned_state = None
            raise BrowserProfileUnavailable(
                "Browser-profile checkpoint plaintext is invalid."
            ) from None
        if owned_state is None:  # pragma: no cover - paired validation
            raise BrowserProfileUnavailable("Browser-profile checkpoint plaintext is invalid.")
        self._require_current_dependencies()
        plaintext = owned_state.canonical_bytes()
        plaintext_bytes = len(plaintext)
        if plaintext_bytes > self.limits.max_plaintext_bytes:
            raise BrowserProfileUnavailable("Browser-profile plaintext exceeds capacity.")
        request = reservation.request
        previous = BrowserProfileRef(
            profile_id=self.authority.profile_id,
            authority_fingerprint=self.authority.fingerprint,
            generation=request.writer_claim.generation,
            content_fingerprint=request.writer_claim.content_fingerprint,
        )
        generation = previous.generation + 1
        nonce = secrets.token_bytes(BROWSER_PROFILE_NONCE_BYTES)
        ciphertext_bytes = plaintext_bytes + 16
        aad_material = {
            "schema_version": 1,
            "encryption_algorithm": BROWSER_PROFILE_ENCRYPTION_ALGORITHM,
            "profile_id": self.authority.profile_id,
            "authority_fingerprint": self.authority.fingerprint,
            "owner_fingerprint": self.authority.scope.owner_fingerprint,
            "sharing_fingerprint": self.authority.scope.sharing_fingerprint,
            "destination_policy_fingerprint": (self.authority.destination_policy.fingerprint),
            "browser_protocol": self.authority.browser_protocol,
            "browser_worker_version": self.authority.browser_worker_version,
            "state_schema_version": self.authority.state_schema_version,
            "generation": generation,
            "key_authority_id": self.authority.key_authority_id,
            "store_id": self.authority.store_id,
            "plaintext_bytes": plaintext_bytes,
            "ciphertext_bytes": ciphertext_bytes,
        }
        aad = canonical_durable_json_bytes(
            aad_material,
            "browser profile authenticated data",
        )
        current_task = asyncio.current_task()
        cancellation_requests_before_encryption = (
            0 if current_task is None else current_task.cancelling()
        )
        cancellation_pending_before_encryption = bool(
            current_task is not None and getattr(current_task, "_must_cancel", False)
        )

        async def encrypt_profile(
            *,
            private_nonce: bytes = nonce,
            private_plaintext: bytes = plaintext,
            private_aad: bytes = aad,
        ) -> bytes:
            async with asyncio.timeout(self.limits.export_timeout_seconds):
                return await self.key_authority.encrypt(
                    nonce=private_nonce,
                    plaintext=private_plaintext,
                    aad=private_aad,
                )

        encryption = encrypt_profile()
        del encrypt_profile
        plaintext = b""
        encryption_failure: BaseException | None = None
        ciphertext: object = b""
        try:
            ciphertext = await _settle_profile_task(
                encryption,
                task_name="cayu-browser-profile-encryption",
            )
        except asyncio.CancelledError as cancellation:
            if current_task is not None and (
                cancellation_pending_before_encryption
                or current_task.cancelling() > cancellation_requests_before_encryption
            ):
                _clear_inactive_profile_failure_frames(cancellation)
                encryption_failure = cancellation
            else:
                _clear_inactive_profile_failure_frames(cancellation)
                encryption_failure = BrowserProfileUnavailable(
                    "Browser-profile encryption was cancelled unexpectedly."
                )
        except TimeoutError as failure:
            _clear_inactive_profile_failure_frames(failure)
            encryption_failure = BrowserProfileUnavailable("Browser-profile encryption timed out.")
        except BaseException as failure:
            _clear_inactive_profile_failure_frames(failure)
            if _profile_failure_contains_process_control(failure):
                encryption_failure = failure
            else:
                encryption_failure = BrowserProfileUnavailable("Browser-profile encryption failed.")
        if encryption_failure is not None:
            plaintext = b""
            ciphertext = b""
            nonce = b""
            aad = b""
            aad_material.clear()
            del encryption
            raise encryption_failure from None
        try:
            self._require_current_dependencies()
        except BaseException:
            ciphertext = b""
            nonce = b""
            aad = b""
            aad_material.clear()
            del encryption
            raise
        if type(ciphertext) is not bytes:
            ciphertext = b""
            nonce = b""
            aad = b""
            aad_material.clear()
            del encryption
            raise BrowserProfileUnavailable("Browser-profile encryption returned invalid output.")
        if len(ciphertext) != ciphertext_bytes:
            ciphertext = b""
            nonce = b""
            aad = b""
            aad_material.clear()
            del encryption
            raise BrowserProfileUnavailable("Browser-profile encryption returned invalid output.")
        try:
            envelope = BrowserProfileEncryptedEnvelope(
                authority_fingerprint=self.authority.fingerprint,
                generation=generation,
                key_authority_id=self.authority.key_authority_id,
                store_id=self._authority.store_id,
                plaintext_bytes=plaintext_bytes,
                ciphertext_bytes=len(ciphertext),
                content_fingerprint=sha256(nonce + ciphertext).hexdigest(),
                nonce_base64=base64.b64encode(nonce).decode("ascii"),
                ciphertext_base64=base64.b64encode(ciphertext).decode("ascii"),
            )
        finally:
            ciphertext = b""
            nonce = b""
            aad = b""
            aad_material.clear()
            del encryption
        origin_count = len(owned_state.origins)
        cookie_count = len(owned_state.cookies)
        storage_entry_count = sum(len(origin.local_storage) for origin in owned_state.origins)

        async def stage_and_publish(
            stored_envelope: BrowserProfileEncryptedEnvelope = envelope,
        ) -> tuple[BrowserProfileCheckpointReceipt, BrowserProfileEncryptedEnvelope]:
            stored_receipt = self._own_checkpoint_receipt(
                await self._call_store(
                    "stage_checkpoint",
                    reservation,
                    stored_envelope,
                    origin_count=origin_count,
                    cookie_count=cookie_count,
                    storage_entry_count=storage_entry_count,
                ),
                reservation=reservation,
                outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
                counts=(origin_count, cookie_count, storage_entry_count),
            )
            if (
                stored_receipt.published_ref.generation != stored_envelope.generation
                or stored_receipt.published_ref.content_fingerprint
                != stored_envelope.content_fingerprint
            ):
                raise BrowserProfileStoreConflict(
                    "Browser-profile checkpoint staging changed envelope authority."
                )
            try:
                published_receipt = self._own_checkpoint_receipt(
                    await self._call_store(
                        "publish_checkpoint",
                        self._access,
                        request.operation_id,
                    ),
                    reservation=reservation,
                    outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
                    counts=(origin_count, cookie_count, storage_entry_count),
                )
                if published_receipt != stored_receipt:
                    raise BrowserProfileStoreConflict(
                        "Browser-profile checkpoint publication changed its receipt."
                    )
                return published_receipt, stored_envelope
            except BaseException as failure:
                if _profile_failure_contains_process_control(failure):
                    raise
                try:
                    existing = await self._call_store(
                        "load_checkpoint_receipt",
                        self.access,
                        request.operation_id,
                    )
                except BaseException as reconciliation_failure:
                    if _profile_failure_contains_process_control(reconciliation_failure):
                        safe_failure = _safe_profile_extension_failure(failure)
                        reconciliation_failure.add_note(
                            "Browser-profile checkpoint publication first failed before "
                            "process control interrupted reconciliation."
                        )
                        raise reconciliation_failure from safe_failure
                    raise failure from reconciliation_failure
                if existing is not None:
                    owned_existing = self._own_checkpoint_receipt(
                        existing,
                        reservation=reservation,
                        outcome=BrowserProfileTerminalOutcome.SUCCEEDED,
                        counts=(origin_count, cookie_count, storage_entry_count),
                    )
                    if owned_existing == stored_receipt:
                        return owned_existing, stored_envelope
                raise failure

        publication = stage_and_publish()
        del stage_and_publish
        envelope = None
        try:
            published, published_envelope = await _settle_profile_task(
                publication,
                task_name="cayu-browser-profile-checkpoint-publication",
            )
        except BaseException:
            owned_state = BrowserProfileStateV1()
            publication = None
            raise
        publication = None
        updated_claim = _validated_model_update(
            request.writer_claim,
            {
                "generation": published.published_ref.generation,
                "content_fingerprint": published.published_ref.content_fingerprint,
            },
        )
        try:
            material._replace_claim(
                updated_claim,
                profile_ref=published.published_ref,
                state=owned_state,
                envelope=published_envelope,
            )
        finally:
            published_envelope = None
        return published

    async def release_writer(self, material: BrowserProfileRestoreMaterial) -> None:
        preparation, _private_state = self._snapshot_material(material)
        _private_state = BrowserProfileStateV1()
        self._require_current_store_dependency()
        await _settle_profile_task(
            self._call_store(
                "release_writer",
                self._access,
                preparation.writer_claim,
            ),
            task_name="cayu-browser-profile-writer-release",
        )

    async def revoke(
        self,
        *,
        revoked_at: datetime | None = None,
    ) -> BrowserProfileInspection:
        """Revoke future profile use without claiming to erase loaded cookies."""

        validation_failure: BaseException | None = None
        owned_revoked_at: datetime | None = None
        try:
            owned_revoked_at = None if revoked_at is None else _utc(revoked_at, "revoked_at")
        except (TypeError, ValueError) as failure:
            validation_failure = failure
        revoked_at = None
        if validation_failure is not None:
            _clear_inactive_profile_failure_frames(validation_failure)
            validation_failure = None
            raise ValueError("Browser-profile revocation time is invalid.") from None
        self._require_current_store_dependency()
        inspection = await _settle_profile_task(
            self._call_store(
                "revoke_profile",
                self._access,
                revoked_at=owned_revoked_at,
            ),
            task_name="cayu-browser-profile-revocation",
        )
        owned_inspection = _own_profile_extension_model(
            inspection,
            BrowserProfileInspection,
        )
        if (
            owned_inspection.profile_id != self._authority.profile_id
            or owned_inspection.authority_fingerprint != self._authority.fingerprint
            or owned_inspection.owner_fingerprint != self._authority.scope.owner_fingerprint
            or owned_inspection.sharing_fingerprint != self._authority.scope.sharing_fingerprint
            or owned_inspection.destination_policy_fingerprint
            != self._authority.destination_policy.fingerprint
            or owned_inspection.key_authority_id != self._authority.key_authority_id
            or owned_inspection.store_id != self._authority.store_id
        ):
            raise BrowserProfileStoreConflict(
                "Browser-profile revocation inspection has conflicting authority."
            )
        return owned_inspection


__all__ = [
    "BROWSER_PROFILE_ENCRYPTION_ALGORITHM",
    "BROWSER_PROFILE_MAX_CIPHERTEXT_BYTES",
    "BROWSER_PROFILE_MAX_COOKIES",
    "BROWSER_PROFILE_MAX_IMPORT_EXPORT_SECONDS",
    "BROWSER_PROFILE_MAX_LEASE_SECONDS",
    "BROWSER_PROFILE_MAX_NAME_BYTES",
    "BROWSER_PROFILE_MAX_ORIGINS",
    "BROWSER_PROFILE_MAX_PLAINTEXT_BYTES",
    "BROWSER_PROFILE_MAX_STORAGE_ENTRIES",
    "BROWSER_PROFILE_MAX_VALUE_BYTES",
    "BROWSER_PROFILE_SCHEMA_VERSION",
    "BROWSER_PROFILE_STATE_SCHEMA_VERSION",
    "AESGCMBrowserProfileKeyAuthority",
    "BrowserProfileAccess",
    "BrowserProfileAuthority",
    "BrowserProfileBinding",
    "BrowserProfileCheckpointPolicy",
    "BrowserProfileCheckpointReceipt",
    "BrowserProfileCheckpointRequest",
    "BrowserProfileCheckpointReservation",
    "BrowserProfileCookie",
    "BrowserProfileDestinationPolicy",
    "BrowserProfileEncryptedEnvelope",
    "BrowserProfileInspection",
    "BrowserProfileKeyAuthority",
    "BrowserProfileLimits",
    "BrowserProfileOriginStorage",
    "BrowserProfileRef",
    "BrowserProfileRestorePreparation",
    "BrowserProfileRestoreReceipt",
    "BrowserProfileRestoreRequest",
    "BrowserProfileScope",
    "BrowserProfileStateV1",
    "BrowserProfileStatus",
    "BrowserProfileStorageEntry",
    "BrowserProfileStore",
    "BrowserProfileStoreConflict",
    "BrowserProfileTerminalOutcome",
    "BrowserProfileUnavailable",
    "BrowserProfileWriterClaim",
    "InMemoryBrowserProfileStore",
    "SQLiteBrowserProfileStore",
    "browser_profile_state_from_playwright",
    "browser_profile_state_to_playwright",
    "validate_browser_profile_state",
]
