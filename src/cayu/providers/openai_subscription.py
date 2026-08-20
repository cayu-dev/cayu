"""Experimental ChatGPT-subscription authentication for Cayu.

This module intentionally identifies requests as Cayu. It does not impersonate
Codex or attempt to bypass upstream access controls.
"""

from __future__ import annotations

import asyncio
import base64
import errno
import json
import math
import os
import secrets
import stat
import struct
import sys
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

from cayu._validation import require_clean_nonblank, require_finite
from cayu._version import package_version
from cayu.core.messages import Message
from cayu.providers._credential_boundary import (
    ProviderStreamCleanupError,
    aclosing_provider_stream,
    detach_provider_call_traceback,
    detach_provider_stream_traceback,
)
from cayu.providers._http import (
    aclose_transport,
    copy_headers,
    credential_safe_post_completion_failure,
    credential_sanitization_values,
    safe_provider_exception_type_name,
    sanitize_provider_cancellation,
    validate_base_url,
)
from cayu.providers.base import (
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelContextPressureProfile,
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    ModelStreamEventType,
    UsageDialect,
    _preflight_provider_portable_messages,
    privacy_safe_provider_option_projection,
)
from cayu.providers.hosted import HostedToolCapabilityError, OpenAIWebSearch
from cayu.providers.openai import (
    DEFAULT_OPENAI_STREAM_IDLE_TIMEOUT_SECONDS,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN,
    HttpxOpenAITransport,
    OpenAIAPIError,
    OpenAITransport,
    _effective_openai_request_options,
    _openai_tool,
    _preflight_openai_hosted_tools,
    _validate_openai_tool_name,
    build_openai_payload,
    openai_stream_events,
    preflight_openai_native_structured_output_schema,
)
from cayu.vaults import SecretRedactor

_AUTH_STORE_VERSION = 1
_AUTH_PROVIDER_KEY = "openai_subscription"
_SAFE_AUTH_STORE_ERRORS = frozenset(
    {
        "Cayu auth store must contain a JSON object.",
        "Cayu auth store providers must be a JSON object.",
        "Cayu auth store could not be made durable.",
        "Cayu auth store durability is unsupported on this platform.",
        "Cayu auth store has unsafe permissions or ownership.",
        "Cayu auth-store directory changed while in use.",
        "Cayu auth-store directory has unsafe permissions or ownership.",
        "Cayu auth-store parent is not a directory.",
        "Could not read Cayu auth store.",
        "OpenAI subscription credentials are invalid.",
        "OpenAI subscription credentials must be a JSON object.",
        "Refusing to read a symlinked Cayu auth store.",
        "Refusing to use a symlinked Cayu auth-store directory.",
        "Refusing to write a symlinked Cayu auth store.",
    }
)
DEFAULT_OPENAI_SUBSCRIPTION_REFRESH_SKEW_SECONDS = 120.0
DEFAULT_OPENAI_SUBSCRIPTION_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_OPENAI_SUBSCRIPTION_OAUTH_ISSUER = "https://auth.openai.com"
OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_PROTECTED_SUBSCRIPTION_HEADERS = {
    "authorization",
    "chatgpt-account-id",
    "content-type",
    "originator",
    "user-agent",
}
_AUTH_THREAD_LOCKS_GUARD = threading.Lock()
_AUTH_THREAD_LOCKS: dict[str, threading.RLock] = {}
_AUTH_TEMP_CREATE_ATTEMPTS = 16
_AUTH_ACCESS_TOKEN_MAX_BYTES = 256 * 1024
_AUTH_REFRESH_TOKEN_MAX_BYTES = 256 * 1024
_AUTH_ACCOUNT_ID_MAX_BYTES = 16 * 1024
# Python's fcntl module does not export these stable Darwin sys/fcntl.h ABI values.
_DARWIN_F_PREALLOCATE = 42
_DARWIN_F_ALLOCATEALL = 0x00000004
_DARWIN_F_PEOFPOSMODE = 3
_DARWIN_FSTORE = struct.Struct("@Iiqqq")
_OPEN_NOFOLLOW_FLAG = getattr(os, "O_NOFOLLOW", 0)
_OPEN_DIRECTORY_FLAG = getattr(os, "O_DIRECTORY", 0)
_OPEN_CLOEXEC_FLAG = getattr(os, "O_CLOEXEC", 0)
_OPEN_BINARY_FLAG = getattr(os, "O_BINARY", 0)
_OPEN_NONBLOCK_FLAG = getattr(os, "O_NONBLOCK", 0)
_SUPPORTS_DURABLE_AUTH_STORE = (
    os.name == "posix"
    and bool(_OPEN_DIRECTORY_FLAG)
    and bool(_OPEN_NOFOLLOW_FLAG)
    and bool(_OPEN_NONBLOCK_FLAG)
    and hasattr(os, "fchmod")
    and hasattr(os, "geteuid")
    and os.mkdir in os.supports_dir_fd
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
    and os.unlink in os.supports_dir_fd
)


@dataclass(slots=True)
class _PreparedAuthStoreWrite:
    descriptor: int
    temporary_name: str
    identity: tuple[int, int]
    reserved_bytes: int | None = None
    published: bool = False


class OpenAISubscriptionAuthError(RuntimeError):
    """Raised when subscription credentials are unavailable or cannot refresh."""


class OpenAISubscriptionOAuthTransport(Protocol):
    def refresh(self, refresh_token: str) -> Mapping[str, Any]:
        """Exchange a refresh token for current OpenAI OAuth credentials."""


class OpenAISubscriptionCredentialProvider(Protocol):
    async def credentials(self) -> OpenAISubscriptionCredentials:
        """Return fresh credentials for one provider request."""


@dataclass(frozen=True, slots=True)
class OpenAISubscriptionCredentials:
    """Refreshable credentials issued by OpenAI's ChatGPT OAuth flow."""

    access_token: str = field(repr=False)
    refresh_token: str = field(repr=False)
    expires_at: float
    account_id: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "access_token",
            _require_bounded_auth_value(
                self.access_token,
                "access_token",
                max_bytes=_AUTH_ACCESS_TOKEN_MAX_BYTES,
            ),
        )
        object.__setattr__(
            self,
            "refresh_token",
            _require_bounded_auth_value(
                self.refresh_token,
                "refresh_token",
                max_bytes=_AUTH_REFRESH_TOKEN_MAX_BYTES,
            ),
        )
        if isinstance(self.expires_at, bool) or not isinstance(self.expires_at, int | float):
            raise TypeError("expires_at must be a number.")
        try:
            expires_at = float(self.expires_at)
        except OverflowError:
            raise ValueError("expires_at must be finite and greater than zero.") from None
        if not math.isfinite(expires_at) or expires_at <= 0:
            raise ValueError("expires_at must be finite and greater than zero.")
        object.__setattr__(self, "expires_at", expires_at)
        if self.account_id is not None:
            object.__setattr__(
                self,
                "account_id",
                _require_bounded_auth_value(
                    self.account_id,
                    "account_id",
                    max_bytes=_AUTH_ACCOUNT_ID_MAX_BYTES,
                ),
            )


def _require_bounded_auth_value(
    value: str,
    field_name: str,
    *,
    max_bytes: int,
) -> str:
    value = require_clean_nonblank(value, field_name)
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"`{field_name}` must contain valid Unicode text.") from None
    if len(encoded) > max_bytes:
        raise ValueError(f"`{field_name}` must not exceed {max_bytes} bytes.")
    return value


def _maximum_size_subscription_credentials() -> OpenAISubscriptionCredentials:
    return OpenAISubscriptionCredentials(
        access_token="\0" * _AUTH_ACCESS_TOKEN_MAX_BYTES,
        refresh_token="\0" * _AUTH_REFRESH_TOKEN_MAX_BYTES,
        expires_at=sys.float_info.max,
        account_id="\0" * _AUTH_ACCOUNT_ID_MAX_BYTES,
    )


class OpenAISubscriptionAuthStore:
    """Private local JSON store for a user's ChatGPT OAuth credentials."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_auth_path()

    def load(self) -> OpenAISubscriptionCredentials | None:
        sentinel = object()
        result: OpenAISubscriptionCredentials | None | object = sentinel
        failure_message: str | None = None
        try:
            self._validate_existing_parent()
            if not self.path.exists() and not self.path.is_symlink():
                result = None
            else:
                with self._exclusive_lock() as directory_fd:
                    result = self._load_credentials_unlocked(directory_fd)
        except Exception as exc:
            failure_message = _safe_auth_store_error_message(exc)
        if failure_message is not None:
            result = sentinel
            raise ValueError(failure_message) from None
        if result is sentinel:
            raise ValueError("Could not access Cayu auth store.")
        return result if isinstance(result, OpenAISubscriptionCredentials) else None

    def _load_credentials_unlocked(
        self,
        directory_fd: int | None,
    ) -> OpenAISubscriptionCredentials | None:
        decoded = self._load_document(directory_fd, missing_ok=True)
        return self._credentials_from_document(decoded)

    def _credentials_from_document(
        self,
        decoded: dict[str, Any] | None,
    ) -> OpenAISubscriptionCredentials | None:
        if decoded is None:
            return None
        providers = decoded.get("providers")
        if not isinstance(providers, dict):
            return None
        raw = providers.get(_AUTH_PROVIDER_KEY)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("OpenAI subscription credentials must be a JSON object.")
        credentials: OpenAISubscriptionCredentials | None = None
        with suppress(KeyError, TypeError, ValueError):
            credentials = OpenAISubscriptionCredentials(
                access_token=raw["access_token"],
                refresh_token=raw["refresh_token"],
                expires_at=raw["expires_at"],
                account_id=raw.get("account_id"),
            )
        if credentials is None:
            raise ValueError("OpenAI subscription credentials are invalid.")
        return credentials

    def save(self, credentials: OpenAISubscriptionCredentials) -> None:
        if type(credentials) is not OpenAISubscriptionCredentials:
            raise TypeError("credentials must be OpenAISubscriptionCredentials.")
        failure_message: str | None = None
        try:
            with self._exclusive_lock() as directory_fd:
                self._save_credentials_unlocked(credentials, directory_fd)
        except Exception as exc:
            failure_message = _safe_auth_store_error_message(exc)
        if failure_message is not None:
            del credentials
            raise ValueError(failure_message) from None

    def _save_credentials_unlocked(
        self,
        credentials: OpenAISubscriptionCredentials,
        directory_fd: int | None,
        *,
        prepared_write: _PreparedAuthStoreWrite | None = None,
        document: dict[str, Any] | None = None,
    ) -> None:
        if document is None:
            document = self._load_document(directory_fd)
            assert document is not None
        self._set_credentials_in_document(document, credentials)
        self._write_document(
            document,
            directory_fd,
            prepared_write=prepared_write,
        )

    def _set_credentials_in_document(
        self,
        document: dict[str, Any],
        credentials: OpenAISubscriptionCredentials,
    ) -> None:
        providers = document.setdefault("providers", {})
        if not isinstance(providers, dict):
            raise ValueError("Cayu auth store providers must be a JSON object.")
        providers[_AUTH_PROVIDER_KEY] = {
            "access_token": credentials.access_token,
            "refresh_token": credentials.refresh_token,
            "expires_at": credentials.expires_at,
            "account_id": credentials.account_id,
        }
        document["version"] = _AUTH_STORE_VERSION

    def delete(self) -> bool:
        result: bool | None = None
        failure_message: str | None = None
        try:
            self._validate_existing_parent()
            missing = not self.path.exists() and not self.path.is_symlink()
            if missing and not _supports_durable_auth_store():
                result = False
            else:
                with self._exclusive_lock() as directory_fd:
                    result = self._delete_unlocked(directory_fd)
        except Exception as exc:
            failure_message = _safe_auth_store_error_message(exc)
        if failure_message is not None:
            raise ValueError(failure_message)
        if result is None:
            raise ValueError("Could not access Cayu auth store.")
        return result

    def _delete_unlocked(self, directory_fd: int | None) -> bool:
        document = self._load_document(directory_fd)
        assert document is not None
        providers = document.get("providers")
        if not isinstance(providers, dict) or _AUTH_PROVIDER_KEY not in providers:
            return False
        del providers[_AUTH_PROVIDER_KEY]
        document["version"] = _AUTH_STORE_VERSION
        self._write_document(document, directory_fd)
        return True

    @contextmanager
    def _exclusive_lock(self):
        self._prepare_parent()
        key = os.path.normcase(os.path.abspath(self.path))
        with _AUTH_THREAD_LOCKS_GUARD:
            thread_lock = _AUTH_THREAD_LOCKS.setdefault(key, threading.RLock())
        lock_path = self.path.with_name(f".{self.path.name}.lock")
        with thread_lock:
            if _supports_durable_auth_store():
                with (
                    _open_auth_store_directory(self.path.parent) as directory_fd,
                    _process_file_lock(lock_path, directory_fd=directory_fd),
                ):
                    _require_current_auth_store_directory(self.path.parent, directory_fd)
                    try:
                        yield directory_fd
                    except BaseException:
                        raise
                    else:
                        _require_current_auth_store_directory(self.path.parent, directory_fd)
            else:
                with _process_file_lock(lock_path):
                    yield None

    def _load_document(
        self,
        directory_fd: int | None,
        *,
        missing_ok: bool = False,
    ) -> dict[str, Any] | None:
        raw = _read_auth_store_file(self.path, directory_fd=directory_fd)
        if raw is None:
            return None if missing_ok else {"version": _AUTH_STORE_VERSION, "providers": {}}
        read_failed = object()
        decoded: Any = read_failed
        with suppress(UnicodeDecodeError, ValueError):
            decoded = json.loads(raw)
        if decoded is read_failed:
            raise ValueError("Could not read Cayu auth store.")
        if not isinstance(decoded, dict):
            raise ValueError("Cayu auth store must contain a JSON object.")
        return decoded

    def _write_document(
        self,
        document: dict[str, Any],
        directory_fd: int | None,
        *,
        prepared_write: _PreparedAuthStoreWrite | None = None,
    ) -> None:
        if prepared_write is not None:
            assert directory_fd is not None
            self._publish_prepared_document(document, directory_fd, prepared_write)
            return
        with self._prepare_write_unlocked(directory_fd) as current_write:
            assert directory_fd is not None
            self._publish_prepared_document(document, directory_fd, current_write)

    def _publish_prepared_document(
        self,
        document: dict[str, Any],
        directory_fd: int,
        prepared_write: _PreparedAuthStoreWrite,
    ) -> None:
        descriptor = prepared_write.descriptor
        _require_current_auth_store_directory(self.path.parent, directory_fd)
        _require_private_auth_file(os.fstat(descriptor), prepared_write.identity)
        payload = _auth_store_document_payload(document)
        if (
            prepared_write.reserved_bytes is not None
            and len(payload) > prepared_write.reserved_bytes
        ):
            raise ValueError("Cayu auth store could not be made durable.")
        if prepared_write.reserved_bytes is None:
            _write_auth_store_payload(descriptor, payload)
        else:
            _write_reserved_auth_store_payload(
                descriptor,
                payload,
                reserved_bytes=prepared_write.reserved_bytes,
            )
        _sync_auth_store_descriptor(descriptor)
        _require_private_auth_file(os.fstat(descriptor), prepared_write.identity)
        os.replace(
            prepared_write.temporary_name,
            self.path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        prepared_write.published = True
        _require_published_auth_file(
            self.path.name,
            directory_fd=directory_fd,
            expected_identity=prepared_write.identity,
        )
        _sync_auth_store_directory(directory_fd)
        _require_current_auth_store_directory(self.path.parent, directory_fd)
        _require_published_auth_file(
            self.path.name,
            directory_fd=directory_fd,
            expected_identity=prepared_write.identity,
        )

    @contextmanager
    def _prepare_write_unlocked(
        self,
        directory_fd: int | None,
        *,
        reservation_document: dict[str, Any] | None = None,
    ):
        self._require_durable_write_support_unlocked(directory_fd)
        assert directory_fd is not None
        with _prepare_auth_store_write(
            self.path.name,
            directory_fd=directory_fd,
            reservation_document=reservation_document,
        ) as prepared_write:
            _require_current_auth_store_directory(self.path.parent, directory_fd)
            yield prepared_write

    def _require_durable_write_support_unlocked(self, directory_fd: int | None) -> None:
        if directory_fd is None:
            raise ValueError("Cayu auth store durability is unsupported on this platform.")
        _sync_auth_store_directory_chain(self.path.parent, directory_fd)

    def _prepare_parent(self) -> None:
        if not self.path.parent.exists() and not _supports_durable_auth_store():
            raise ValueError("Cayu auth store durability is unsupported on this platform.")
        self._validate_existing_parent()
        if not self.path.parent.exists():
            _create_private_auth_store_directory_chain(self.path.parent)
        self._validate_existing_parent()

    def _validate_existing_parent(self) -> None:
        if self.path.parent.is_symlink():
            raise ValueError("Refusing to use a symlinked Cayu auth-store directory.")
        if self.path.parent.exists() and not self.path.parent.is_dir():
            raise ValueError("Cayu auth-store parent is not a directory.")


def _safe_auth_store_error_message(exc: Exception) -> str:
    message = str(exc)
    if message in _SAFE_AUTH_STORE_ERRORS:
        return message
    return "Could not access Cayu auth store."


class OpenAISubscriptionAuth:
    """Load and refresh one user's ChatGPT subscription credentials."""

    def __init__(
        self,
        *,
        store: OpenAISubscriptionAuthStore | None = None,
        oauth_transport: OpenAISubscriptionOAuthTransport | None = None,
        refresh_skew_seconds: float = DEFAULT_OPENAI_SUBSCRIPTION_REFRESH_SKEW_SECONDS,
        now: Callable[[], float] = time.time,
    ) -> None:
        if isinstance(refresh_skew_seconds, bool) or not isinstance(
            refresh_skew_seconds, int | float
        ):
            raise TypeError("refresh_skew_seconds must be a number.")
        if refresh_skew_seconds < 0:
            raise ValueError("refresh_skew_seconds must not be negative.")
        self.store = store if store is not None else OpenAISubscriptionAuthStore()
        self.oauth_transport = (
            oauth_transport
            if oauth_transport is not None
            else HttpxOpenAISubscriptionOAuthTransport()
        )
        self.refresh_skew_seconds = float(refresh_skew_seconds)
        self._now = now
        self._refresh_lock = asyncio.Lock()

    @detach_provider_call_traceback
    async def credentials(self) -> OpenAISubscriptionCredentials:
        # Store reads take the same cross-process lock as refresh rotation. Keep
        # that potentially network-length wait off Cayu's event-loop thread.
        credentials: OpenAISubscriptionCredentials | None = None
        cancellation: asyncio.CancelledError | None = None
        try:
            credentials = await asyncio.to_thread(self.store.load)
            if credentials is None:
                raise OpenAISubscriptionAuthError(
                    "OpenAI subscription login is missing. Run `cayu auth openai login`."
                )
            if credentials.expires_at > self._now() + self.refresh_skew_seconds:
                return credentials
            async with self._refresh_lock:
                refreshed: OpenAISubscriptionCredentials | None = None
                with suppress(Exception):
                    refreshed = await asyncio.to_thread(self._refresh_credentials)
                if refreshed is None:
                    raise OpenAISubscriptionAuthError(
                        "OpenAI subscription login could not refresh. "
                        "Run `cayu auth openai login` again."
                    )
                return refreshed
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="OpenAI subscription",
                credential_values=_subscription_credential_values(credentials),
                safe_message="OpenAI subscription request cancelled.",
            )
        credentials = None
        if cancellation is not None:
            raise cancellation from None
        raise AssertionError("subscription credential resolution did not return or fail")

    def _refresh_credentials(self) -> OpenAISubscriptionCredentials:
        # The path lock covers the complete rotating-token transaction, not just
        # the JSON write, so separate providers and processes cannot reuse the
        # same refresh token concurrently.
        with self.store._exclusive_lock() as directory_fd:
            document = self.store._load_document(directory_fd, missing_ok=True)
            credentials = self.store._credentials_from_document(document)
            if credentials is None:
                raise OpenAISubscriptionAuthError(
                    "OpenAI subscription login is missing. Run `cayu auth openai login`."
                )
            if credentials.expires_at > self._now() + self.refresh_skew_seconds:
                return credentials
            assert document is not None
            self.store._set_credentials_in_document(
                document,
                _maximum_size_subscription_credentials(),
            )
            with self.store._prepare_write_unlocked(
                directory_fd,
                reservation_document=document,
            ) as prepared_write:
                response = self.oauth_transport.refresh(credentials.refresh_token)
                refreshed = openai_subscription_credentials_from_token_response(
                    response,
                    now=self._now(),
                    fallback_refresh_token=credentials.refresh_token,
                    fallback_account_id=credentials.account_id,
                )
                self.store._save_credentials_unlocked(
                    refreshed,
                    directory_fd,
                    prepared_write=prepared_write,
                    document=document,
                )
                return refreshed


class OpenAISubscriptionProvider(ModelProvider):
    """Experimental Responses adapter backed by a user's ChatGPT subscription.

    OpenAI does not currently document the raw Codex backend as a third-party
    provider API. Cayu therefore sends an honest ``originator: cayu`` identity
    and treats upstream rejection as a hard compatibility boundary.
    """

    name = "openai_subscription"
    usage_dialect = UsageDialect.OPENAI
    supports_native_structured_output = True

    def preflight_portable_messages(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> None:
        _preflight_provider_portable_messages(
            model=model,
            messages=messages,
            tools=tools,
            supports_system_messages=True,
            supports_tool_history=True,
            supports_tool_definitions=True,
            supports_file_attachments=True,
            tool_name_validator=_validate_openai_tool_name,
            tool_definition_validator=_openai_tool,
        )

    def preflight_hosted_tools(
        self,
        *,
        model: str,
        hosted_tools: tuple[OpenAIWebSearch, ...],
        options: dict[str, Any],
    ) -> None:
        _preflight_openai_hosted_tools(
            model=model,
            hosted_tools=hosted_tools,
            options=options,
            endpoint_supported=self.hosted_web_search_supported,
        )

    def request_footprint_options(self, request: ModelRequest) -> dict[str, Any]:
        projected = privacy_safe_provider_option_projection(
            _effective_openai_request_options(request.options)
        )
        return {"openai": projected} if projected else {}

    def request_fingerprint_options(self, request: ModelRequest) -> dict[str, Any]:
        effective = _effective_openai_request_options(request.options)
        return {"openai": effective} if effective else {}

    def __init__(
        self,
        *,
        auth: OpenAISubscriptionCredentialProvider | None = None,
        name: str = "openai_subscription",
        base_url: str = DEFAULT_OPENAI_SUBSCRIPTION_BASE_URL,
        timeout_s: float = DEFAULT_OPENAI_TIMEOUT_SECONDS,
        stream_idle_timeout_s: float = DEFAULT_OPENAI_STREAM_IDLE_TIMEOUT_SECONDS,
        transport: OpenAITransport | None = None,
        extra_headers: Mapping[str, str] | None = None,
        hosted_web_search_supported: bool | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.auth = auth if auth is not None else OpenAISubscriptionAuth()
        self.base_url = validate_base_url(base_url, provider_label="OpenAI subscription")
        if (
            hosted_web_search_supported is not None
            and type(hosted_web_search_supported) is not bool
        ):
            raise TypeError("hosted_web_search_supported must be a boolean or None.")
        self.hosted_web_search_supported = (
            self.base_url
            == validate_base_url(
                DEFAULT_OPENAI_SUBSCRIPTION_BASE_URL,
                provider_label="OpenAI subscription",
            )
            if hosted_web_search_supported is None
            else hosted_web_search_supported
        )
        if type(timeout_s) not in {int, float}:
            raise TypeError("timeout_s must be a number.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        self.timeout_s = float(timeout_s)
        if type(stream_idle_timeout_s) not in {int, float}:
            raise TypeError("stream_idle_timeout_s must be a number.")
        stream_idle_timeout_s = require_finite(
            float(stream_idle_timeout_s), "stream_idle_timeout_s"
        )
        if stream_idle_timeout_s <= 0:
            raise ValueError("stream_idle_timeout_s must be greater than zero.")
        self.stream_idle_timeout_s = stream_idle_timeout_s
        self.transport = transport if transport is not None else HttpxOpenAITransport()
        self.extra_headers = copy_headers(
            extra_headers,
            protected=_PROTECTED_SUBSCRIPTION_HEADERS,
        )

    @property
    def context_pressure_profile(self) -> ModelContextPressureProfile:
        return ModelContextPressureProfile(
            tool_schema_chars_per_token=OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN,
        )

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        preflight_openai_native_structured_output_schema(json_schema)

    async def count_input_tokens(self, request: ModelRequest) -> InputTokenCountResult | None:
        del request
        return None

    @detach_provider_stream_traceback
    async def stream(self, request: ModelRequest):
        credentials: OpenAISubscriptionCredentials | None = None
        cancellation: asyncio.CancelledError | None = None
        overflow_failure: ModelContextOverflowError | None = None
        post_completion_failure: ModelProviderError | None = None
        capability_failure: HostedToolCapabilityError | None = None
        error_event: ModelStreamEvent | None = None
        completion_emitted = False
        try:
            credentials = await self.auth.credentials()
            payload = build_openai_payload(request, stream=True, reasoning_state="inline")
            raw_events = self.transport.stream_response_events(
                url=f"{self.base_url}/responses",
                headers=self._headers(credentials),
                payload=payload,
                timeout_s=self.timeout_s,
                stream_idle_timeout_s=self.stream_idle_timeout_s,
            )
            events = openai_stream_events(raw_events, reasoning_state="inline")
            async with aclosing_provider_stream(raw_events), aclosing_provider_stream(events):
                async for event in events:
                    completion_emitted = event.type == ModelStreamEventType.COMPLETED
                    yield event
                    if completion_emitted:
                        break
        except asyncio.CancelledError as exc:
            cancellation = sanitize_provider_cancellation(
                exc,
                provider_label="OpenAI subscription",
                credential_values=credential_sanitization_values(
                    *_subscription_credential_values(credentials),
                    extra_headers=self.extra_headers,
                ),
                safe_message="OpenAI subscription request cancelled.",
            )
        except ModelContextOverflowError as exc:
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="OpenAI subscription",
                    provider_name="openai",
                    credential_values=credential_sanitization_values(
                        *_subscription_credential_values(credentials),
                        extra_headers=self.extra_headers,
                    ),
                    safe_message="OpenAI subscription provider failed.",
                )
            else:
                overflow_failure = _safe_subscription_context_overflow(
                    exc,
                    credentials,
                    extra_header_values=tuple(self.extra_headers.values()),
                )
        except OpenAIAPIError as exc:
            if request.hosted_tools and _is_subscription_hosted_tool_rejection(exc):
                capability_failure = HostedToolCapabilityError(
                    "The experimental OpenAI subscription backend rejected hosted web search "
                    "for this target. Disable it or verify the current backend capability."
                )
            elif completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="OpenAI subscription",
                    provider_name="openai",
                    credential_values=credential_sanitization_values(
                        *_subscription_credential_values(credentials),
                        extra_headers=self.extra_headers,
                    ),
                    safe_message="OpenAI subscription provider failed.",
                )
            else:
                error_event = _safe_subscription_error_event(
                    exc,
                    credentials,
                    extra_header_values=tuple(self.extra_headers.values()),
                )
        except Exception as exc:
            if completion_emitted:
                post_completion_failure = credential_safe_post_completion_failure(
                    exc,
                    provider_label="OpenAI subscription",
                    provider_name="openai",
                    credential_values=credential_sanitization_values(
                        *_subscription_credential_values(credentials),
                        extra_headers=self.extra_headers,
                    ),
                    safe_message="OpenAI subscription provider failed.",
                )
            else:
                error_event = _safe_subscription_error_event(
                    exc,
                    credentials,
                    extra_header_values=tuple(self.extra_headers.values()),
                )
        credentials = None
        if cancellation is not None:
            raise cancellation from None
        if overflow_failure is not None:
            raise overflow_failure from None
        if post_completion_failure is not None:
            raise post_completion_failure from None
        if capability_failure is not None:
            raise capability_failure from None
        if error_event is not None:
            yield error_event

    async def aclose(self) -> None:
        await aclose_transport(self.transport)

    def _headers(self, credentials: OpenAISubscriptionCredentials) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {credentials.access_token}",
            "originator": "cayu",
            "user-agent": f"cayu/{_cayu_version()}",
        }
        if credentials.account_id is not None:
            headers["ChatGPT-Account-ID"] = credentials.account_id
        headers.update(self.extra_headers)
        return headers


class HttpxOpenAISubscriptionOAuthTransport:
    """HTTP transport for OpenAI's public Codex OAuth client."""

    def __init__(
        self,
        *,
        issuer: str = DEFAULT_OPENAI_SUBSCRIPTION_OAUTH_ISSUER,
        timeout_s: float = 30.0,
    ) -> None:
        self.issuer = validate_base_url(issuer, provider_label="OpenAI OAuth")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, int | float):
            raise TypeError("timeout_s must be a number.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        self.timeout_s = float(timeout_s)

    def refresh(self, refresh_token: str) -> Mapping[str, Any]:
        refresh_token = require_clean_nonblank(refresh_token, "refresh_token")
        return self._post_form(
            "/oauth/token",
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID,
            },
        )

    def exchange_authorization_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> Mapping[str, Any]:
        return self._post_form(
            "/oauth/token",
            {
                "grant_type": "authorization_code",
                "code": require_clean_nonblank(code, "code"),
                "redirect_uri": require_clean_nonblank(redirect_uri, "redirect_uri"),
                "client_id": OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID,
                "code_verifier": require_clean_nonblank(code_verifier, "code_verifier"),
            },
        )

    def request_device_authorization(self) -> Mapping[str, Any]:
        response = self._post_json(
            "/api/accounts/deviceauth/usercode",
            {"client_id": OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID},
        )
        if response is None:
            raise OpenAISubscriptionAuthError("OpenAI device authorization did not start.")
        return response

    def poll_device_authorization(
        self,
        *,
        device_auth_id: str,
        user_code: str,
    ) -> Mapping[str, Any] | None:
        return self._post_json(
            "/api/accounts/deviceauth/token",
            {
                "device_auth_id": require_clean_nonblank(device_auth_id, "device_auth_id"),
                "user_code": require_clean_nonblank(user_code, "user_code"),
            },
            pending_statuses={403, 404},
        )

    def exchange_device_authorization(
        self,
        *,
        authorization_code: str,
        code_verifier: str,
    ) -> Mapping[str, Any]:
        return self.exchange_authorization_code(
            code=authorization_code,
            redirect_uri=f"{self.issuer}/deviceauth/callback",
            code_verifier=code_verifier,
        )

    def _post_form(self, path: str, data: Mapping[str, str]) -> Mapping[str, Any]:
        response: httpx.Response | None = None
        with suppress(httpx.RequestError):
            response = httpx.post(
                f"{self.issuer}{path}",
                data=dict(data),
                headers={
                    "content-type": "application/x-www-form-urlencoded",
                    "user-agent": f"cayu/{_cayu_version()}",
                },
                timeout=self.timeout_s,
            )
        if response is None:
            raise OpenAISubscriptionAuthError("OpenAI OAuth request failed.")
        return _oauth_response(response)

    def _post_json(
        self,
        path: str,
        data: Mapping[str, str],
        *,
        pending_statuses: set[int] | None = None,
    ) -> Mapping[str, Any] | None:
        response: httpx.Response | None = None
        with suppress(httpx.RequestError):
            response = httpx.post(
                f"{self.issuer}{path}",
                json=dict(data),
                headers={
                    "content-type": "application/json",
                    "user-agent": f"cayu/{_cayu_version()}",
                },
                timeout=self.timeout_s,
            )
        if response is None:
            raise OpenAISubscriptionAuthError("OpenAI OAuth request failed.")
        if pending_statuses is not None and response.status_code in pending_statuses:
            return None
        return _oauth_response(response)


def openai_subscription_credentials_from_token_response(
    response: Mapping[str, Any],
    *,
    now: float,
    fallback_refresh_token: str | None = None,
    fallback_account_id: str | None = None,
) -> OpenAISubscriptionCredentials:
    if not isinstance(response, Mapping):
        raise OpenAISubscriptionAuthError("OpenAI OAuth response must be a JSON object.")
    access_token = response.get("access_token")
    refresh_token = response.get("refresh_token") or fallback_refresh_token
    if not isinstance(access_token, str) or not access_token.strip():
        raise OpenAISubscriptionAuthError("OpenAI OAuth response omitted access_token.")
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        raise OpenAISubscriptionAuthError("OpenAI OAuth response omitted refresh_token.")
    expires_in = response.get("expires_in", 3600)
    if isinstance(expires_in, bool) or not isinstance(expires_in, int | float):
        raise OpenAISubscriptionAuthError("OpenAI OAuth response contained invalid expires_in.")
    try:
        expires_in_seconds = float(expires_in)
    except OverflowError:
        raise OpenAISubscriptionAuthError(
            "OpenAI OAuth response contained invalid expires_in."
        ) from None
    expires_at = now + expires_in_seconds
    if (
        not math.isfinite(expires_in_seconds)
        or expires_in_seconds <= 0
        or not math.isfinite(expires_at)
    ):
        raise OpenAISubscriptionAuthError("OpenAI OAuth response contained invalid expires_in.")
    account_id = _extract_account_id(response) or fallback_account_id
    return OpenAISubscriptionCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        account_id=account_id,
    )


def _extract_account_id(tokens: Mapping[str, Any]) -> str | None:
    for token_key in ("id_token", "access_token"):
        token = tokens.get(token_key)
        if not isinstance(token, str):
            continue
        claims = _jwt_claims(token)
        if claims is None:
            continue
        direct = claims.get("chatgpt_account_id")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()
        auth_claims = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, Mapping):
            nested = auth_claims.get("chatgpt_account_id")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
        organizations = claims.get("organizations")
        if isinstance(organizations, list) and organizations:
            first = organizations[0]
            if isinstance(first, Mapping):
                organization_id = first.get("id")
                if isinstance(organization_id, str) and organization_id.strip():
                    return organization_id.strip()
    return None


def _jwt_claims(token: str) -> Mapping[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    encoded = parts[1]
    padding = "=" * (-len(encoded) % 4)
    try:
        decoded = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (ValueError, UnicodeDecodeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def build_openai_subscription_authorize_url(
    *,
    redirect_uri: str,
    code_challenge: str,
    state: str,
    issuer: str = DEFAULT_OPENAI_SUBSCRIPTION_OAUTH_ISSUER,
) -> str:
    issuer = validate_base_url(issuer, provider_label="OpenAI OAuth")
    query = urlencode(
        {
            "response_type": "code",
            "client_id": OPENAI_SUBSCRIPTION_OAUTH_CLIENT_ID,
            "redirect_uri": require_clean_nonblank(redirect_uri, "redirect_uri"),
            "scope": "openid profile email offline_access",
            "code_challenge": require_clean_nonblank(code_challenge, "code_challenge"),
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": require_clean_nonblank(state, "state"),
            "originator": "cayu",
        }
    )
    return f"{issuer}/oauth/authorize?{query}"


def _oauth_response(response: httpx.Response) -> Mapping[str, Any]:
    if response.status_code < 200 or response.status_code >= 300:
        raise OpenAISubscriptionAuthError(
            f"OpenAI OAuth request failed with HTTP {response.status_code}."
        )
    parse_failed = object()
    decoded: Any = parse_failed
    with suppress(ValueError):
        decoded = response.json()
    if decoded is parse_failed:
        raise OpenAISubscriptionAuthError("OpenAI OAuth response was not valid JSON.")
    if not isinstance(decoded, Mapping):
        raise OpenAISubscriptionAuthError("OpenAI OAuth response must be a JSON object.")
    return decoded


def _safe_subscription_error_event(
    exc: Exception,
    credentials: OpenAISubscriptionCredentials | None,
    *,
    extra_header_values: tuple[str, ...] = (),
) -> ModelStreamEvent:
    """Project a provider failure through an allowlisted, credential-safe shape."""

    if isinstance(exc, OpenAISubscriptionAuthError):
        message = "OpenAI subscription authentication failed. Run `cayu auth openai login` again."
    else:
        message = "OpenAI subscription provider failed."
    payload: dict[str, Any] = {
        "error": message,
        "error_type": safe_provider_exception_type_name(exc),
    }
    if isinstance(exc, ModelProviderError):
        typed_fields = {
            key: value
            for key, value in exc.error_payload_fields().items()
            if key in {"status_code", "retryable", "retry_after_s"}
        }
        typed_fields["provider"] = "openai"
        safe_error_type = _safe_subscription_error_identity(
            exc.error_type,
            field_name="error_type",
        )
        safe_error_code = _safe_subscription_error_identity(
            exc.error_code,
            field_name="error_code",
        )
        if safe_error_type is not None:
            typed_fields["provider_error_type"] = safe_error_type
        if safe_error_code is not None:
            typed_fields["provider_error_code"] = safe_error_code
        payload.update(typed_fields)
    else:
        status_code = getattr(exc, "status_code", None)
        if type(status_code) is int:
            payload["status_code"] = status_code
        retryable = getattr(exc, "retryable", None)
        if type(retryable) is bool:
            payload["retryable"] = retryable
        retry_after_s = getattr(exc, "retry_after_s", None)
        if type(retry_after_s) in {int, float}:
            payload["retry_after_s"] = retry_after_s
    if isinstance(exc, ProviderStreamCleanupError):
        payload["stream_cleanup_failed"] = True
    return ModelStreamEvent(type=ModelStreamEventType.ERROR, payload=payload)


def _is_subscription_hosted_tool_rejection(exc: OpenAIAPIError) -> bool:
    if exc.status_code not in {400, 404, 422}:
        return False
    param = "" if exc.param is None else exc.param.lower()
    code = "" if exc.error_code is None else exc.error_code.lower()
    tool_param = param == "tools" or param.startswith("tools[")
    if code in {"invalid_tool", "unsupported_tool"}:
        return True
    if code in {"invalid_value", "unsupported_value"}:
        return tool_param
    return tool_param


def _safe_subscription_context_overflow(
    exc: ModelContextOverflowError,
    credentials: OpenAISubscriptionCredentials | None,
    *,
    extra_header_values: tuple[str, ...] = (),
) -> ModelContextOverflowError:
    message = "OpenAI subscription context window exceeded."
    del credentials, extra_header_values
    provider = "openai"
    return ModelContextOverflowError(
        message,
        provider=provider,
        status_code=exc.status_code,
        error_type=_safe_subscription_error_identity(
            exc.error_type,
            field_name="error_type",
        ),
        error_code=_safe_subscription_error_identity(
            exc.error_code,
            field_name="error_code",
        ),
        request_id=None,
    )


def _safe_subscription_error_identity(
    value: object,
    *,
    field_name: str,
) -> str | None:
    if type(value) is not str:
        return None
    allowed = (
        {
            "authentication_error",
            "error",
            "invalid_request_error",
            "rate_limit_error",
            "server_error",
        }
        if field_name == "error_type"
        else {
            "context_length_exceeded",
            "internal_error",
            "rate_limit_exceeded",
            "server_error",
        }
    )
    return value if value in allowed else None


def _subscription_credential_redactor(
    credentials: OpenAISubscriptionCredentials,
    *,
    extra_header_values: tuple[str, ...] = (),
) -> SecretRedactor:
    values = [credentials.access_token, credentials.refresh_token]
    if credentials.account_id is not None:
        values.append(credentials.account_id)
    values.extend(extra_header_values)
    return SecretRedactor(values)


def _subscription_credential_values(
    credentials: OpenAISubscriptionCredentials | None,
) -> tuple[str, ...]:
    if credentials is None:
        return ()
    values = [credentials.access_token, credentials.refresh_token]
    if credentials.account_id is not None:
        values.append(credentials.account_id)
    return tuple(values)


def _cayu_version() -> str:
    return package_version()


def _default_auth_path() -> Path:
    configured_home = os.environ.get("CAYU_HOME")
    home = Path(configured_home).expanduser() if configured_home else Path.home() / ".cayu"
    return home / "auth.json"


@contextmanager
def _process_file_lock(path: Path, *, directory_fd: int | None = None):
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if directory_fd is not None:
        flags |= _OPEN_NOFOLLOW_FLAG
    descriptor = os.open(
        path.name if directory_fd is not None else path,
        flags,
        0o600,
        dir_fd=directory_fd,
    )
    try:
        if directory_fd is not None:
            initial = os.fstat(descriptor)
            _require_owned_regular_auth_file(initial)
            os.fchmod(descriptor, 0o600)
            _require_private_auth_file(
                os.fstat(descriptor),
                _stat_identity(initial),
            )
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _supports_durable_auth_store() -> bool:
    return _SUPPORTS_DURABLE_AUTH_STORE and (
        sys.platform != "darwin" or _darwin_full_sync_command() is not None
    )


def _darwin_full_sync_command() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        import fcntl
    except ImportError:
        return None
    command = getattr(fcntl, "F_FULLFSYNC", None)
    return command if isinstance(command, int) else None


@contextmanager
def _open_auth_store_directory(path: Path, *, require_private: bool = True):
    if not _supports_durable_auth_store():
        raise ValueError("Cayu auth store durability is unsupported on this platform.")
    flags = os.O_RDONLY | _OPEN_DIRECTORY_FLAG | _OPEN_NOFOLLOW_FLAG | _OPEN_CLOEXEC_FLAG
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise ValueError("Refusing to use a symlinked Cayu auth-store directory.") from None
        raise exc
    try:
        _require_current_auth_store_directory(
            path,
            descriptor,
            require_private=require_private,
        )
        yield descriptor
    finally:
        os.close(descriptor)


def _require_current_auth_store_directory(
    path: Path,
    directory_fd: int,
    *,
    require_private: bool = True,
) -> None:
    opened = os.fstat(directory_fd)
    try:
        current = os.stat(path, follow_symlinks=False)
    except (OSError, RuntimeError):
        raise ValueError("Cayu auth-store directory changed while in use.") from None
    if not stat.S_ISDIR(opened.st_mode) or _stat_identity(opened) != _stat_identity(current):
        if stat.S_ISLNK(current.st_mode):
            raise ValueError("Refusing to use a symlinked Cayu auth-store directory.")
        raise ValueError("Cayu auth-store directory changed while in use.")
    if require_private and (opened.st_uid != os.geteuid() or stat.S_IMODE(opened.st_mode) & 0o022):
        raise ValueError("Cayu auth-store directory has unsafe permissions or ownership.")


def _create_private_auth_store_directory_chain(path: Path) -> None:
    if not _supports_durable_auth_store():
        raise ValueError("Cayu auth store durability is unsupported on this platform.")
    missing_directories: list[Path] = []
    anchor = path
    while not anchor.exists():
        missing_directories.append(anchor)
        parent = anchor.parent
        if parent == anchor:
            raise ValueError("Cayu auth-store parent is not a directory.")
        anchor = parent
    try:
        resolved_anchor = anchor.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("Cayu auth-store directory changed while in use.") from None
    flags = os.O_RDONLY | _OPEN_DIRECTORY_FLAG | _OPEN_NOFOLLOW_FLAG | _OPEN_CLOEXEC_FLAG
    with _open_auth_store_directory(resolved_anchor, require_private=False) as anchor_fd:
        current_fd = os.dup(anchor_fd)
        try:
            for directory in reversed(missing_directories):
                created = False
                try:
                    os.mkdir(directory.name, mode=0o700, dir_fd=current_fd)
                    created = True
                except FileExistsError:
                    pass
                try:
                    child_fd = os.open(directory.name, flags, dir_fd=current_fd)
                except OSError:
                    raise ValueError("Cayu auth-store directory changed while in use.") from None
                try:
                    if created:
                        os.fchmod(child_fd, 0o700)
                    info = os.fstat(child_fd)
                    _require_private_auth_directory(info, exact_mode=created)
                except BaseException:
                    os.close(child_fd)
                    raise
                os.close(current_fd)
                current_fd = child_fd
        finally:
            os.close(current_fd)


def _require_private_auth_directory(info: os.stat_result, *, exact_mode: bool) -> None:
    mode = stat.S_IMODE(info.st_mode)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or mode & 0o022
        or (exact_mode and mode != 0o700)
    ):
        raise ValueError("Cayu auth-store directory has unsafe permissions or ownership.")


def _read_auth_store_file(path: Path, *, directory_fd: int | None) -> str | None:
    if directory_fd is None:
        if path.is_symlink():
            raise ValueError("Refusing to read a symlinked Cayu auth store.")
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise ValueError("Could not read Cayu auth store.") from None

    flags = (
        os.O_RDONLY
        | _OPEN_NOFOLLOW_FLAG
        | _OPEN_CLOEXEC_FLAG
        | _OPEN_BINARY_FLAG
        | _OPEN_NONBLOCK_FLAG
    )
    try:
        descriptor = os.open(path.name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ValueError("Refusing to read a symlinked Cayu auth store.") from None
        raise ValueError("Could not read Cayu auth store.") from None
    try:
        info = os.fstat(descriptor)
        _require_private_auth_file(info, _stat_identity(info))
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        raise ValueError("Could not read Cayu auth store.") from None
    finally:
        os.close(descriptor)


def _open_auth_store_temporary(leaf_name: str, *, directory_fd: int) -> tuple[int, str]:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | _OPEN_NOFOLLOW_FLAG
        | _OPEN_CLOEXEC_FLAG
        | _OPEN_BINARY_FLAG
    )
    for _ in range(_AUTH_TEMP_CREATE_ATTEMPTS):
        temporary_name = f".{leaf_name}.tmp-{secrets.token_hex(16)}"
        try:
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            continue
        return descriptor, temporary_name
    raise OSError("could not allocate a unique auth-store staging file")


def _auth_store_document_payload(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_auth_store_payload(descriptor: int, payload: bytes) -> None:
    os.ftruncate(descriptor, 0)
    _overwrite_auth_store_payload(descriptor, payload)


def _overwrite_auth_store_payload(descriptor: int, payload: bytes) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("could not write Cayu auth store")
        remaining = remaining[written:]


def _write_reserved_auth_store_payload(
    descriptor: int,
    payload: bytes,
    *,
    reserved_bytes: int,
) -> None:
    if len(payload) > reserved_bytes:
        raise ValueError("Cayu auth store could not be made durable.")
    # Keep every allocated byte live until the complete final record is present.
    # JSON accepts trailing whitespace, so a failed size-only compaction can
    # safely publish the synchronized padded record instead of losing a rotated
    # refresh token after the provider has consumed its predecessor.
    padded_payload = payload.ljust(reserved_bytes, b" ")
    _overwrite_auth_store_payload(descriptor, padded_payload)
    if len(payload) == reserved_bytes:
        return
    try:
        os.ftruncate(descriptor, len(payload))
    except OSError:
        try:
            size = os.fstat(descriptor).st_size
        except OSError:
            raise ValueError("Cayu auth store could not be made durable.") from None
        if size < len(payload) or size > reserved_bytes:
            raise ValueError("Cayu auth store could not be made durable.") from None


def _reserve_auth_store_capacity(descriptor: int, reserved_bytes: int) -> None:
    if sys.platform == "darwin":
        _reserve_darwin_auth_store_capacity(descriptor, reserved_bytes)
    else:
        allocator = getattr(os, "posix_fallocate", None)
        if not callable(allocator):
            raise ValueError("Cayu auth store durability is unsupported on this platform.")
        try:
            allocator(descriptor, 0, reserved_bytes)
        except OSError:
            raise ValueError("Cayu auth store could not be made durable.") from None
    try:
        os.ftruncate(descriptor, reserved_bytes)
        if os.fstat(descriptor).st_size != reserved_bytes:
            raise ValueError("Cayu auth store could not be made durable.")
    except OSError:
        raise ValueError("Cayu auth store could not be made durable.") from None


def _reserve_darwin_auth_store_capacity(descriptor: int, reserved_bytes: int) -> None:
    try:
        import fcntl
    except ImportError:
        raise ValueError("Cayu auth store durability is unsupported on this platform.") from None
    request = _DARWIN_FSTORE.pack(
        _DARWIN_F_ALLOCATEALL,
        _DARWIN_F_PEOFPOSMODE,
        0,
        reserved_bytes,
        0,
    )
    try:
        response = fcntl.fcntl(descriptor, _DARWIN_F_PREALLOCATE, request)
    except OSError:
        raise ValueError("Cayu auth store could not be made durable.") from None
    if not isinstance(response, bytes) or len(response) != _DARWIN_FSTORE.size:
        raise ValueError("Cayu auth store could not be made durable.")
    *_, allocated_bytes = _DARWIN_FSTORE.unpack(response)
    if allocated_bytes < reserved_bytes:
        raise ValueError("Cayu auth store could not be made durable.")


@contextmanager
def _prepare_auth_store_write(
    leaf_name: str,
    *,
    directory_fd: int,
    reservation_document: dict[str, Any] | None,
):
    descriptor, temporary_name = _open_auth_store_temporary(
        leaf_name,
        directory_fd=directory_fd,
    )
    identity: tuple[int, int] | None = None
    prepared_write: _PreparedAuthStoreWrite | None = None
    try:
        identity = _stat_identity(os.fstat(descriptor))
        os.fchmod(descriptor, 0o600)
        _require_private_auth_file(os.fstat(descriptor), identity)
        reserved_bytes: int | None = None
        if reservation_document is not None:
            reservation_payload = _auth_store_document_payload(reservation_document)
            reserved_bytes = len(reservation_payload)
            _reserve_auth_store_capacity(descriptor, reserved_bytes)
            _sync_auth_store_descriptor(descriptor)
            _require_private_auth_file(os.fstat(descriptor), identity)
        prepared_write = _PreparedAuthStoreWrite(
            descriptor=descriptor,
            temporary_name=temporary_name,
            identity=identity,
            reserved_bytes=reserved_bytes,
        )
        yield prepared_write
    finally:
        if identity is not None and not (prepared_write is not None and prepared_write.published):
            _unlink_auth_store_temporary_if_unchanged(
                temporary_name,
                directory_fd=directory_fd,
                expected_identity=identity,
            )
        with suppress(OSError):
            os.close(descriptor)


def _require_private_auth_file(
    info: os.stat_result,
    expected_identity: tuple[int, int],
) -> None:
    _require_owned_regular_auth_file(info)
    if _stat_identity(info) != expected_identity or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("Cayu auth store has unsafe permissions or ownership.")


def _require_owned_regular_auth_file(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
        raise ValueError("Cayu auth store has unsafe permissions or ownership.")


def _require_published_auth_file(
    leaf_name: str,
    *,
    directory_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    try:
        info = os.stat(leaf_name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        raise ValueError("Cayu auth store could not be made durable.") from None
    try:
        _require_private_auth_file(info, expected_identity)
    except ValueError:
        raise ValueError("Cayu auth store could not be made durable.") from None


def _unlink_auth_store_temporary_if_unchanged(
    temporary_name: str,
    *,
    directory_fd: int,
    expected_identity: tuple[int, int],
) -> None:
    try:
        info = os.stat(temporary_name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return
    if stat.S_ISREG(info.st_mode) and _stat_identity(info) == expected_identity:
        with suppress(OSError):
            os.unlink(temporary_name, dir_fd=directory_fd)


def _sync_auth_store_directory(directory_fd: int) -> None:
    _sync_auth_store_descriptor(directory_fd)


def _sync_auth_store_descriptor(descriptor: int) -> None:
    try:
        command = _darwin_full_sync_command()
        if sys.platform == "darwin":
            if command is None:
                raise ValueError("Cayu auth store durability is unsupported on this platform.")
            import fcntl

            fcntl.fcntl(descriptor, command)
        else:
            os.fsync(descriptor)
    except OSError:
        raise ValueError("Cayu auth store could not be made durable.") from None


def _sync_auth_store_directory_path(path: Path) -> None:
    try:
        synchronized_path = path.resolve(strict=True)
    except (OSError, RuntimeError):
        raise ValueError("Cayu auth-store directory changed while in use.") from None
    with _open_auth_store_directory(synchronized_path, require_private=False) as directory_fd:
        _sync_auth_store_directory(directory_fd)


def _sync_auth_store_directory_chain(path: Path, directory_fd: int) -> None:
    _require_current_auth_store_directory(path, directory_fd)
    try:
        resolved_path = path.resolve(strict=True)
        resolved_info = os.stat(resolved_path, follow_symlinks=False)
    except (OSError, RuntimeError):
        raise ValueError("Cayu auth-store directory changed while in use.") from None
    if _stat_identity(resolved_info) != _stat_identity(os.fstat(directory_fd)):
        raise ValueError("Cayu auth-store directory changed while in use.")
    for candidate in (*reversed(resolved_path.parents), resolved_path):
        _sync_auth_store_directory_path(candidate)
    _require_current_auth_store_directory(path, directory_fd)


def _stat_identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino
