"""WorkOS device authentication for the Cayu Cloud CLI."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import sys
import time
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from cayu._filesystem_lock import cooperative_path_lock
from cayu.cli._cloud_private_state import (
    PreparedPrivateJsonWrite,
    prepare_private_json,
    remove_private_json_staging,
    write_private_json,
)

_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
_REFRESH_SKEW_SECONDS = 60.0
_REFRESH_RETRY_DELAYS_SECONDS = (0.25, 0.5)
_REFRESH_RETRY_WINDOW_SECONDS = 25.0
# Bound the complete refresh attempt so one lost acknowledgement still leaves
# time for a replay inside WorkOS's 30-second rotated-token grace period.
_REFRESH_HTTP_ATTEMPT_TIMEOUT_SECONDS = 5.0
_AUTH_LOCK_DIRECTORY = "cayu-cloud-auth-locks-v1"
_AUTH_TEXT_MAX_BYTES = 32_768
_AUTH_REFRESH_STAGING_PREFIX = ".cayu-cloud-auth-refresh-"


class CloudAuthError(RuntimeError):
    """Stable customer-facing Cloud authentication failure."""

    def __init__(self, category: str, message: str) -> None:
        super().__init__(message)
        self.category = category


@dataclass(frozen=True, slots=True)
class CloudAuthCredentials:
    api_url: str
    workos_api_hostname: str
    workos_client_id: str
    access_token: str
    refresh_token: str
    expires_at: float
    organization_id: str
    user_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "api_url", _canonical_http_url(self.api_url, "API URL"))
        for field_name in (
            "workos_api_hostname",
            "workos_client_id",
            "access_token",
            "refresh_token",
            "organization_id",
            "user_id",
        ):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or any(character.isspace() for character in value)
                or len(value.encode()) > _AUTH_TEXT_MAX_BYTES
            ):
                raise ValueError(f"Cayu Cloud {field_name} is invalid.")
        _workos_base_url(self.workos_api_hostname)
        if (
            isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, int | float)
            or not math.isfinite(float(self.expires_at))
            or float(self.expires_at) <= 0
        ):
            raise ValueError("Cayu Cloud access-token expiry is invalid.")
        object.__setattr__(self, "expires_at", float(self.expires_at))


@dataclass(frozen=True, slots=True)
class DeviceAuthorization:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: str
    expires_in: float
    interval: float


class CloudAuthStore:
    """Private local store for one interactive Cayu Cloud session."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _default_auth_path()

    def load(self) -> CloudAuthCredentials | None:
        try:
            payload = json.loads(self.path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CloudAuthError(
                "cloud_auth_unavailable",
                "Could not read the local Cayu Cloud login.",
            ) from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise CloudAuthError(
                "cloud_auth_unavailable",
                "The local Cayu Cloud login is invalid; run `cayu cloud login` again.",
            )
        try:
            return CloudAuthCredentials(
                api_url=payload["api_url"],
                workos_api_hostname=payload["workos_api_hostname"],
                workos_client_id=payload["workos_client_id"],
                access_token=payload["access_token"],
                refresh_token=payload["refresh_token"],
                expires_at=payload["expires_at"],
                organization_id=payload["organization_id"],
                user_id=payload["user_id"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CloudAuthError(
                "cloud_auth_unavailable",
                "The local Cayu Cloud login is invalid; run `cayu cloud login` again.",
            ) from exc

    def save(self, credentials: CloudAuthCredentials) -> None:
        if type(credentials) is not CloudAuthCredentials:
            raise TypeError("credentials must be CloudAuthCredentials.")
        with self._exclusive_lock():
            self._save_unlocked(credentials)

    @contextmanager
    def prepare_refresh(
        self,
        expected: CloudAuthCredentials,
    ) -> Iterator[_CloudAuthRefreshPublication]:
        """Reserve publication capacity for one exact rotating-token refresh."""

        if type(expected) is not CloudAuthCredentials:
            raise TypeError("expected must be CloudAuthCredentials.")
        try:
            with ExitStack() as resources:
                with self._exclusive_lock():
                    current = self.load()
                    prepared = None
                    if current is not None and _same_cloud_auth_credentials(current, expected):
                        prepared = resources.enter_context(
                            prepare_private_json(
                                self.path,
                                _refresh_reservation_payload(current),
                                staging_prefix=_auth_refresh_staging_prefix(self.path),
                            )
                        )
                yield _CloudAuthRefreshPublication(
                    store=self,
                    expected=expected,
                    current=current,
                    prepared=prepared,
                )
        except CloudAuthError:
            raise
        except (OSError, ValueError, ExceptionGroup) as exc:
            raise CloudAuthError(
                "cloud_auth_unavailable",
                "Could not prepare or clean up the local Cayu Cloud login refresh.",
            ) from exc

    def _save_unlocked(self, credentials: CloudAuthCredentials) -> None:
        _write_private_json(self.path, _credentials_payload(credentials))

    def delete(self) -> bool:
        with self._exclusive_lock():
            failures: list[Exception] = []
            existed = True
            try:
                self.path.unlink()
            except FileNotFoundError:
                existed = False
            except OSError as exc:
                failures.append(exc)
            removed_staging = 0
            try:
                removed_staging = remove_private_json_staging(
                    self.path,
                    staging_prefix=_auth_refresh_staging_prefix(self.path),
                )
            except (OSError, ExceptionGroup) as exc:
                failures.append(exc)
            if failures:
                cause: Exception
                if len(failures) == 1:
                    cause = failures[0]
                else:
                    cause = ExceptionGroup(
                        "Cloud authentication cleanup failed.",
                        failures,
                    )
                raise CloudAuthError(
                    "cloud_auth_unavailable",
                    "Could not remove the local Cayu Cloud login.",
                ) from cause
            return existed or removed_staging > 0

    @contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with cooperative_path_lock(
                self.path.parent,
                self.path.name,
                lock_directory_name=_AUTH_LOCK_DIRECTORY,
            ):
                yield
        except OSError as exc:
            raise CloudAuthError(
                "cloud_auth_unavailable",
                "Could not update the local Cayu Cloud login.",
            ) from exc


@dataclass(slots=True)
class _CloudAuthRefreshPublication:
    store: CloudAuthStore
    expected: CloudAuthCredentials
    current: CloudAuthCredentials | None
    prepared: PreparedPrivateJsonWrite | None

    def publish(
        self,
        replacement: CloudAuthCredentials,
    ) -> CloudAuthCredentials | None:
        if type(replacement) is not CloudAuthCredentials:
            raise TypeError("replacement must be CloudAuthCredentials.")
        if self.prepared is None:
            return self.current
        replacement_payload = _credentials_payload(replacement)
        with self.store._exclusive_lock():
            current = self.store.load()
            if current is None or not _same_cloud_auth_credentials(current, self.expected):
                return current
            try:
                self.prepared.publish(replacement_payload)
            except (OSError, ValueError) as publication_error:
                try:
                    published = self.store.load()
                except CloudAuthError:
                    published = None
                if published is not None and _same_cloud_auth_credentials(
                    published,
                    replacement,
                ):
                    try:
                        self.prepared.adopt_observed_publication()
                    except (OSError, ValueError) as synchronization_error:
                        raise CloudAuthError(
                            "cloud_auth_unavailable",
                            "Could not save the refreshed Cayu Cloud login.",
                        ) from synchronization_error
                    return published
                try:
                    self.prepared.retry_publication(replacement_payload)
                except (OSError, ValueError) as retry_error:
                    try:
                        published = self.store.load()
                    except CloudAuthError:
                        published = None
                    if published is not None and _same_cloud_auth_credentials(
                        published,
                        replacement,
                    ):
                        try:
                            self.prepared.adopt_observed_publication()
                        except (OSError, ValueError) as synchronization_error:
                            raise CloudAuthError(
                                "cloud_auth_unavailable",
                                "Could not save the refreshed Cayu Cloud login.",
                            ) from synchronization_error
                        return published
                    raise CloudAuthError(
                        "cloud_auth_unavailable",
                        "Could not save the refreshed Cayu Cloud login.",
                    ) from ExceptionGroup(
                        "Private JSON publication attempts failed.",
                        [
                            publication_error,
                            _detach_historical_exception_context(
                                retry_error,
                                historical=publication_error,
                            ),
                        ],
                    )
                else:
                    return replacement
            return replacement


class WorkOSDeviceAuthClient:
    """Small HTTP client for WorkOS's public device authorization flow."""

    def __init__(self, *, api_url: str, timeout_seconds: float = 30.0) -> None:
        self.api_url = _canonical_http_url(api_url, "API URL")
        if timeout_seconds <= 0:
            raise ValueError("API timeout must be positive.")
        self.timeout_seconds = timeout_seconds

    def portal_config(self) -> tuple[str, str]:
        payload = self._request_json("GET", f"{self.api_url}/v1/portal-config")
        if payload.get("authentication") != "workos":
            raise CloudAuthError(
                "workos_unavailable",
                "This Cayu Cloud does not offer WorkOS CLI authentication.",
            )
        client_id = _required_string(payload, "workos_client_id")
        api_hostname = _required_string(payload, "workos_api_hostname")
        _workos_base_url(api_hostname)
        return client_id, api_hostname

    def authorize_device(
        self,
        *,
        client_id: str,
        api_hostname: str,
    ) -> DeviceAuthorization:
        payload = self._request_json(
            "POST",
            f"{_workos_base_url(api_hostname)}/user_management/authorize/device",
            form={"client_id": client_id},
        )
        expires_in = _required_positive_number(payload, "expires_in")
        interval = _required_positive_number(payload, "interval")
        return DeviceAuthorization(
            device_code=_required_string(payload, "device_code"),
            user_code=_required_string(payload, "user_code"),
            verification_uri=_canonical_http_url(
                _required_string(payload, "verification_uri"),
                "verification URI",
            ),
            verification_uri_complete=_canonical_http_url(
                _required_string(payload, "verification_uri_complete"),
                "complete verification URI",
                allow_query=True,
            ),
            expires_in=expires_in,
            interval=interval,
        )

    def poll_for_credentials(
        self,
        authorization: DeviceAuthorization,
        *,
        client_id: str,
        api_hostname: str,
    ) -> CloudAuthCredentials:
        deadline = time.monotonic() + authorization.expires_in
        interval = authorization.interval
        while time.monotonic() < deadline:
            response = self._request(
                "POST",
                f"{_workos_base_url(api_hostname)}/user_management/authenticate",
                form={
                    "client_id": client_id,
                    "device_code": authorization.device_code,
                    "grant_type": _DEVICE_GRANT,
                },
            )
            payload = _json_object(response)
            if 200 <= response.status_code < 300:
                return _credentials_from_response(
                    payload,
                    api_url=self.api_url,
                    client_id=client_id,
                    api_hostname=api_hostname,
                )
            error = payload.get("error")
            if error == "authorization_pending":
                time.sleep(interval)
                continue
            if error == "slow_down":
                interval += 1.0
                time.sleep(interval)
                continue
            if error == "access_denied":
                raise CloudAuthError("login_denied", "Cayu Cloud login was denied.")
            if error == "expired_token":
                raise CloudAuthError("login_expired", "Cayu Cloud login expired.")
            raise CloudAuthError("login_failed", "Cayu Cloud login failed.")
        raise CloudAuthError("login_expired", "Cayu Cloud login expired.")

    def refresh(self, credentials: CloudAuthCredentials) -> CloudAuthCredentials:
        response: httpx.Response | None = None
        retry_started_monotonic = time.monotonic()
        retry_started_wall = time.time()
        for attempt, retry_delay in enumerate((*_REFRESH_RETRY_DELAYS_SECONDS, None)):
            if attempt > 0 and not _refresh_retry_window_open(
                started_monotonic=retry_started_monotonic,
                started_wall=retry_started_wall,
            ):
                return _transient_refresh_fallback(credentials)
            try:
                attempt_timeout_seconds = min(
                    self.timeout_seconds,
                    _REFRESH_HTTP_ATTEMPT_TIMEOUT_SECONDS,
                )
                response = self._request(
                    "POST",
                    f"{_workos_base_url(credentials.workos_api_hostname)}/user_management/authenticate",
                    form={
                        "client_id": credentials.workos_client_id,
                        "grant_type": "refresh_token",
                        "refresh_token": credentials.refresh_token,
                    },
                    timeout_seconds=attempt_timeout_seconds,
                    total_timeout_seconds=attempt_timeout_seconds,
                )
            except CloudAuthError as exc:
                if exc.category != "login_unavailable":
                    raise
            else:
                if 200 <= response.status_code < 300:
                    break
                if not _transient_refresh_status(response.status_code):
                    if response.status_code == 400 and _workos_error(response) == "invalid_grant":
                        raise CloudAuthError(
                            "login_refresh_failed",
                            "The Cayu Cloud login expired; run `cayu cloud login` again.",
                        )
                    raise CloudAuthError(
                        "login_refresh_failed",
                        "Cayu Cloud authentication service rejected the login refresh.",
                    )
            if retry_delay is None or not _refresh_retry_window_open(
                started_monotonic=retry_started_monotonic,
                started_wall=retry_started_wall,
                additional_seconds=retry_delay,
            ):
                return _transient_refresh_fallback(credentials)
            time.sleep(retry_delay)

        if response is None:  # pragma: no cover - the loop always attempts one request
            raise CloudAuthError(
                "login_refresh_unavailable",
                "The Cayu Cloud login could not refresh temporarily; retry this command.",
            )
        refreshed = _credentials_from_response(
            _json_object(response),
            api_url=credentials.api_url,
            client_id=credentials.workos_client_id,
            api_hostname=credentials.workos_api_hostname,
        )
        if (
            refreshed.organization_id != credentials.organization_id
            or refreshed.user_id != credentials.user_id
        ):
            raise CloudAuthError(
                "login_refresh_failed",
                "The Cayu Cloud login identity changed; run `cayu cloud login` again.",
            )
        return refreshed

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        form: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        response = self._request(method, url, form=form)
        if not 200 <= response.status_code < 300:
            raise CloudAuthError(
                "login_failed",
                "Cayu Cloud authentication service rejected the request.",
            )
        return _json_object(response)

    def _request(
        self,
        method: str,
        url: str,
        *,
        form: dict[str, str] | None = None,
        timeout_seconds: float | None = None,
        total_timeout_seconds: float | None = None,
    ) -> httpx.Response:
        try:
            if total_timeout_seconds is not None:
                return asyncio.run(
                    self._request_with_total_timeout(
                        method,
                        url,
                        form=form,
                        phase_timeout_seconds=(
                            self.timeout_seconds if timeout_seconds is None else timeout_seconds
                        ),
                        total_timeout_seconds=total_timeout_seconds,
                    )
                )
            with httpx.Client(
                follow_redirects=False,
                timeout=(self.timeout_seconds if timeout_seconds is None else timeout_seconds),
            ) as client:
                return client.request(method, url, data=form)
        except (httpx.RequestError, TimeoutError):
            raise CloudAuthError(
                "login_unavailable",
                "Cayu Cloud authentication service is unavailable.",
            ) from None

    async def _request_with_total_timeout(
        self,
        method: str,
        url: str,
        *,
        form: dict[str, str] | None,
        phase_timeout_seconds: float,
        total_timeout_seconds: float,
    ) -> httpx.Response:
        async with asyncio.timeout(total_timeout_seconds):
            async with httpx.AsyncClient(
                follow_redirects=False,
                timeout=phase_timeout_seconds,
            ) as client:
                return await client.request(method, url, data=form)


def fresh_cloud_credentials(
    store: CloudAuthStore,
    *,
    timeout_seconds: float,
) -> CloudAuthCredentials | None:
    credentials = store.load()
    if credentials is None:
        return None
    if credentials.expires_at > time.time() + _REFRESH_SKEW_SECONDS:
        return credentials
    with store.prepare_refresh(credentials) as publication:
        current = publication.current
        if current is None or not _same_cloud_auth_credentials(current, credentials):
            return current
        try:
            refreshed = WorkOSDeviceAuthClient(
                api_url=current.api_url,
                timeout_seconds=timeout_seconds,
            ).refresh(current)
        except CloudAuthError as refresh_error:
            try:
                reconciled = store.load()
            except CloudAuthError as reconciliation_error:
                raise refresh_error from reconciliation_error
            if reconciled is None or not _same_cloud_auth_credentials(
                reconciled,
                current,
            ):
                return reconciled
            raise
        if refreshed is current:
            return store.load()
        return publication.publish(refreshed)


def _transient_refresh_status(status_code: int) -> bool:
    return status_code in {408, 429} or 500 <= status_code < 600


def _refresh_retry_window_open(
    *,
    started_monotonic: float,
    started_wall: float,
    additional_seconds: float = 0.0,
) -> bool:
    monotonic_elapsed = time.monotonic() - started_monotonic
    wall_elapsed = time.time() - started_wall
    return (
        0.0 <= monotonic_elapsed + additional_seconds < _REFRESH_RETRY_WINDOW_SECONDS
        and 0.0 <= wall_elapsed + additional_seconds < _REFRESH_RETRY_WINDOW_SECONDS
    )


def _transient_refresh_fallback(
    credentials: CloudAuthCredentials,
) -> CloudAuthCredentials:
    if credentials.expires_at > time.time():
        return credentials
    raise CloudAuthError(
        "login_refresh_unavailable",
        "The Cayu Cloud login could not refresh temporarily; retry this command.",
    )


def _workos_error(response: httpx.Response) -> str | None:
    try:
        payload = response.json()
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    return error if isinstance(error, str) else None


def _same_cloud_auth_credentials(
    left: CloudAuthCredentials,
    right: CloudAuthCredentials,
) -> bool:
    tokens_match = hmac.compare_digest(
        left.access_token.encode("utf-8"),
        right.access_token.encode("utf-8"),
    ) & hmac.compare_digest(
        left.refresh_token.encode("utf-8"),
        right.refresh_token.encode("utf-8"),
    )
    return tokens_match and (
        left.api_url == right.api_url
        and left.workos_api_hostname == right.workos_api_hostname
        and left.workos_client_id == right.workos_client_id
        and left.expires_at == right.expires_at
        and left.organization_id == right.organization_id
        and left.user_id == right.user_id
    )


def _detach_historical_exception_context(
    error: Exception,
    *,
    historical: Exception,
) -> Exception:
    if error.__context__ is historical:
        error.__context__ = None
    return error


def _credentials_payload(credentials: CloudAuthCredentials) -> dict[str, Any]:
    return {
        "access_token": credentials.access_token,
        "api_url": credentials.api_url,
        "expires_at": credentials.expires_at,
        "organization_id": credentials.organization_id,
        "refresh_token": credentials.refresh_token,
        "schema_version": 1,
        "user_id": credentials.user_id,
        "workos_api_hostname": credentials.workos_api_hostname,
        "workos_client_id": credentials.workos_client_id,
    }


def _auth_refresh_staging_prefix(path: Path) -> str:
    destination_digest = hashlib.sha256(os.fsencode(path.name)).hexdigest()
    return f"{_AUTH_REFRESH_STAGING_PREFIX}{destination_digest}-"


def _refresh_reservation_payload(
    credentials: CloudAuthCredentials,
) -> dict[str, Any]:
    payload = _credentials_payload(credentials)
    maximum_escaped_value = "\0" * _AUTH_TEXT_MAX_BYTES
    payload["access_token"] = maximum_escaped_value
    payload["refresh_token"] = maximum_escaped_value
    payload["expires_at"] = sys.float_info.max
    return payload


def _credentials_from_response(
    payload: dict[str, Any],
    *,
    api_url: str,
    client_id: str,
    api_hostname: str,
) -> CloudAuthCredentials:
    user = payload.get("user")
    if not isinstance(user, dict):
        raise CloudAuthError("login_failed", "WorkOS returned an invalid user identity.")
    access_token = _required_string(payload, "access_token")
    return CloudAuthCredentials(
        api_url=api_url,
        workos_api_hostname=api_hostname,
        workos_client_id=client_id,
        access_token=access_token,
        refresh_token=_required_string(payload, "refresh_token"),
        expires_at=_jwt_expiry(access_token),
        organization_id=_required_string(payload, "organization_id"),
        user_id=_required_string(user, "id"),
    )


def _jwt_expiry(token: str) -> float:
    try:
        _, encoded, _ = token.split(".")
        encoded += "=" * (-len(encoded) % 4)
        claims = json.loads(base64.urlsafe_b64decode(encoded))
        expires_at = claims["exp"]
        if isinstance(expires_at, bool):
            raise TypeError
        expires_at = float(expires_at)
    except (
        binascii.Error,
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        raise CloudAuthError(
            "login_failed",
            "WorkOS returned an invalid access token.",
        ) from None
    if not math.isfinite(expires_at) or expires_at <= 0:
        raise CloudAuthError("login_failed", "WorkOS returned an invalid access token.")
    return expires_at


def _json_object(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        raise CloudAuthError(
            "login_failed",
            "Cayu Cloud authentication service returned invalid JSON.",
        ) from None
    if not isinstance(payload, dict):
        raise CloudAuthError(
            "login_failed",
            "Cayu Cloud authentication service returned invalid JSON.",
        )
    return payload


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(character.isspace() for character in value)
    ):
        raise CloudAuthError("login_failed", "WorkOS authentication response is invalid.")
    return value


def _required_positive_number(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CloudAuthError("login_failed", "WorkOS authentication response is invalid.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise CloudAuthError("login_failed", "WorkOS authentication response is invalid.")
    return result


def _canonical_http_url(value: str, label: str, *, allow_query: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Cayu Cloud {label} is invalid.")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or (parsed.query and not allow_query)
        or parsed.fragment
    ):
        raise ValueError(f"Cayu Cloud {label} is invalid.")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError(f"Cayu Cloud {label} must use HTTPS outside loopback.")
    return value.rstrip("/")


def _workos_base_url(hostname: str) -> str:
    if (
        not isinstance(hostname, str)
        or not hostname
        or hostname != hostname.strip()
        or "://" in hostname
        or "/" in hostname
        or any(character.isspace() for character in hostname)
    ):
        raise ValueError("Cayu Cloud WorkOS API hostname is invalid.")
    parsed = urlsplit(f"//{hostname}")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Cayu Cloud WorkOS API hostname is invalid.")
    scheme = "http" if parsed.hostname in {"127.0.0.1", "::1", "localhost"} else "https"
    return f"{scheme}://{hostname}"


def _default_auth_path() -> Path:
    configured = os.environ.get("CAYU_CLOUD_AUTH")
    if configured:
        return Path(configured).expanduser().resolve()
    configured_state = os.environ.get("CAYU_CLOUD_CONFIG")
    if configured_state:
        return Path(configured_state).expanduser().resolve().with_name("auth.json")
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "cayu-cloud" / "auth.json"


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    try:
        write_private_json(path, payload)
    except OSError as exc:
        raise CloudAuthError(
            "cloud_auth_unavailable",
            "Could not save the local Cayu Cloud login.",
        ) from exc
