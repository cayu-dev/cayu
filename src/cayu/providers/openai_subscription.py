"""Experimental ChatGPT-subscription authentication for Cayu.

This module intentionally identifies requests as Cayu. It does not impersonate
Codex or attempt to bypass upstream access controls.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

from cayu._validation import require_clean_nonblank
from cayu.providers._http import (
    aclose_transport,
    copy_headers,
    exception_message,
    validate_base_url,
)
from cayu.providers.base import (
    InputTokenCountResult,
    ModelContextOverflowError,
    ModelContextPressureProfile,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)
from cayu.providers.openai import (
    DEFAULT_OPENAI_STREAM_IDLE_TIMEOUT_SECONDS,
    DEFAULT_OPENAI_TIMEOUT_SECONDS,
    OPENAI_CONTEXT_PRESSURE_TOOL_SCHEMA_CHARS_PER_TOKEN,
    HttpxOpenAITransport,
    OpenAITransport,
    build_openai_payload,
    openai_stream_events,
    preflight_openai_native_structured_output_schema,
)

_AUTH_STORE_VERSION = 1
_AUTH_PROVIDER_KEY = "openai_subscription"
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
    account_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "access_token",
            require_clean_nonblank(self.access_token, "access_token"),
        )
        object.__setattr__(
            self,
            "refresh_token",
            require_clean_nonblank(self.refresh_token, "refresh_token"),
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
                require_clean_nonblank(self.account_id, "account_id"),
            )


class OpenAISubscriptionAuthStore:
    """Private local JSON store for a user's ChatGPT OAuth credentials."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_auth_path()

    def load(self) -> OpenAISubscriptionCredentials | None:
        self._validate_existing_parent()
        if not self.path.exists() and not self.path.is_symlink():
            return None
        with self._exclusive_lock():
            return self._load_credentials_unlocked()

    def _load_credentials_unlocked(self) -> OpenAISubscriptionCredentials | None:
        if self.path.is_symlink():
            raise ValueError(f"Refusing to read symlinked auth store: {self.path}")
        if not self.path.exists():
            return None
        try:
            decoded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read Cayu auth store {self.path}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"Cayu auth store {self.path} must contain a JSON object.")
        providers = decoded.get("providers")
        if not isinstance(providers, dict):
            return None
        raw = providers.get(_AUTH_PROVIDER_KEY)
        if raw is None:
            return None
        if not isinstance(raw, dict):
            raise ValueError("OpenAI subscription credentials must be a JSON object.")
        try:
            return OpenAISubscriptionCredentials(
                access_token=raw["access_token"],
                refresh_token=raw["refresh_token"],
                expires_at=raw["expires_at"],
                account_id=raw.get("account_id"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("OpenAI subscription credentials are invalid.") from exc

    def save(self, credentials: OpenAISubscriptionCredentials) -> None:
        if type(credentials) is not OpenAISubscriptionCredentials:
            raise TypeError("credentials must be OpenAISubscriptionCredentials.")
        with self._exclusive_lock():
            self._save_credentials_unlocked(credentials)

    def _save_credentials_unlocked(self, credentials: OpenAISubscriptionCredentials) -> None:
        document = self._load_document()
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
        self._write_document(document)

    def delete(self) -> bool:
        with self._exclusive_lock():
            return self._delete_unlocked()

    def _delete_unlocked(self) -> bool:
        document = self._load_document()
        providers = document.get("providers")
        if not isinstance(providers, dict) or _AUTH_PROVIDER_KEY not in providers:
            return False
        del providers[_AUTH_PROVIDER_KEY]
        document["version"] = _AUTH_STORE_VERSION
        self._write_document(document)
        return True

    @contextmanager
    def _exclusive_lock(self):
        self._prepare_parent()
        key = os.path.normcase(os.path.abspath(self.path))
        with _AUTH_THREAD_LOCKS_GUARD:
            thread_lock = _AUTH_THREAD_LOCKS.setdefault(key, threading.RLock())
        with thread_lock, _process_file_lock(self.path.with_name(f".{self.path.name}.lock")):
            yield

    def _load_document(self) -> dict[str, Any]:
        if self.path.is_symlink():
            raise ValueError(f"Refusing to write symlinked auth store: {self.path}")
        if not self.path.exists():
            return {"version": _AUTH_STORE_VERSION, "providers": {}}
        try:
            decoded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Could not read Cayu auth store {self.path}: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"Cayu auth store {self.path} must contain a JSON object.")
        return decoded

    def _write_document(self, document: dict[str, Any]) -> None:
        self._prepare_parent()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )
        temporary_path = Path(temporary_name)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            else:
                temporary_path.chmod(0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            self.path.chmod(0o600)
        except BaseException:
            with suppress(OSError):
                os.close(descriptor)
            temporary_path.unlink(missing_ok=True)
            raise

    def _prepare_parent(self) -> None:
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            self._validate_existing_parent()
            return
        with suppress(OSError):
            self.path.parent.chmod(0o700)

    def _validate_existing_parent(self) -> None:
        if self.path.parent.is_symlink():
            raise ValueError(f"Refusing to use symlinked auth store directory: {self.path.parent}")
        if self.path.parent.exists() and not self.path.parent.is_dir():
            raise ValueError(f"Cayu auth store directory is not a directory: {self.path.parent}")


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

    async def credentials(self) -> OpenAISubscriptionCredentials:
        # Store reads take the same cross-process lock as refresh rotation. Keep
        # that potentially network-length wait off Cayu's event-loop thread.
        credentials = await asyncio.to_thread(self.store.load)
        if credentials is None:
            raise OpenAISubscriptionAuthError(
                "OpenAI subscription login is missing. Run `cayu auth openai login`."
            )
        if credentials.expires_at > self._now() + self.refresh_skew_seconds:
            return credentials
        async with self._refresh_lock:
            try:
                return await asyncio.to_thread(self._refresh_credentials)
            except OpenAISubscriptionAuthError:
                raise
            except Exception as exc:
                raise OpenAISubscriptionAuthError(
                    "OpenAI subscription login could not refresh. "
                    "Run `cayu auth openai login` again."
                ) from exc

    def _refresh_credentials(self) -> OpenAISubscriptionCredentials:
        # The path lock covers the complete rotating-token transaction, not just
        # the JSON write, so separate providers and processes cannot reuse the
        # same refresh token concurrently.
        with self.store._exclusive_lock():
            credentials = self.store._load_credentials_unlocked()
            if credentials is None:
                raise OpenAISubscriptionAuthError(
                    "OpenAI subscription login is missing. Run `cayu auth openai login`."
                )
            if credentials.expires_at > self._now() + self.refresh_skew_seconds:
                return credentials
            response = self.oauth_transport.refresh(credentials.refresh_token)
            refreshed = openai_subscription_credentials_from_token_response(
                response,
                now=self._now(),
                fallback_refresh_token=credentials.refresh_token,
                fallback_account_id=credentials.account_id,
            )
            self.store._save_credentials_unlocked(refreshed)
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
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.auth = auth if auth is not None else OpenAISubscriptionAuth()
        self.base_url = validate_base_url(base_url, provider_label="OpenAI subscription")
        if type(timeout_s) not in {int, float}:
            raise TypeError("timeout_s must be a number.")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be greater than zero.")
        self.timeout_s = float(timeout_s)
        if type(stream_idle_timeout_s) not in {int, float}:
            raise TypeError("stream_idle_timeout_s must be a number.")
        if stream_idle_timeout_s <= 0:
            raise ValueError("stream_idle_timeout_s must be greater than zero.")
        self.stream_idle_timeout_s = float(stream_idle_timeout_s)
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

    async def stream(self, request: ModelRequest):
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
            async for event in openai_stream_events(raw_events, reasoning_state="inline"):
                yield event
        except ModelContextOverflowError:
            raise
        except Exception as exc:
            yield ModelStreamEvent.error(
                exception_message(exc, provider_label="OpenAI subscription"),
                cause=exc,
            )

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
        try:
            response = httpx.post(
                f"{self.issuer}{path}",
                data=dict(data),
                headers={
                    "content-type": "application/x-www-form-urlencoded",
                    "user-agent": f"cayu/{_cayu_version()}",
                },
                timeout=self.timeout_s,
            )
        except httpx.RequestError as exc:
            raise OpenAISubscriptionAuthError("OpenAI OAuth request failed.") from exc
        return _oauth_response(response)

    def _post_json(
        self,
        path: str,
        data: Mapping[str, str],
        *,
        pending_statuses: set[int] | None = None,
    ) -> Mapping[str, Any] | None:
        try:
            response = httpx.post(
                f"{self.issuer}{path}",
                json=dict(data),
                headers={
                    "content-type": "application/json",
                    "user-agent": f"cayu/{_cayu_version()}",
                },
                timeout=self.timeout_s,
            )
        except httpx.RequestError as exc:
            raise OpenAISubscriptionAuthError("OpenAI OAuth request failed.") from exc
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
        detail = ""
        try:
            decoded_error = response.json()
        except ValueError:
            decoded_error = None
        if isinstance(decoded_error, Mapping):
            for key in ("error_description", "error"):
                value = decoded_error.get(key)
                if isinstance(value, str) and value.strip():
                    detail = f": {value.strip()[:240]}"
                    break
        raise OpenAISubscriptionAuthError(
            f"OpenAI OAuth request failed with HTTP {response.status_code}{detail}."
        )
    try:
        decoded = response.json()
    except ValueError as exc:
        raise OpenAISubscriptionAuthError("OpenAI OAuth response was not valid JSON.") from exc
    if not isinstance(decoded, Mapping):
        raise OpenAISubscriptionAuthError("OpenAI OAuth response must be a JSON object.")
    return decoded


def _cayu_version() -> str:
    try:
        return version("cayu")
    except PackageNotFoundError:
        return "0.1.0a3"


def _default_auth_path() -> Path:
    configured_home = os.environ.get("CAYU_HOME")
    home = Path(configured_home).expanduser() if configured_home else Path.home() / ".cayu"
    return home / "auth.json"


@contextmanager
def _process_file_lock(path: Path):
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(path, flags, 0o600)
    try:
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
