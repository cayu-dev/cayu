"""HTTPS client for Cayu Cloud's sandbox broker contract (``cayu.cloud.sandbox`` v1).

The client knows only the wire contract: it creates a sandbox from a catalog
image, polls it, renews its lease, starts, polls and cancels commands, copies
archives in and out, and destroys it. It holds a broker credential (the
Agent's sandbox token or an Organization API key), never a cloud-provider
credential, endpoint or image identity, and it imports no provider SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any, Literal
from urllib.parse import quote, unquote, urlencode, urlsplit

import certifi
import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "BROKER_API_URL_ENV",
    "BROKER_APPLICATION_ID_ENV",
    "BROKER_REQUEST_TIMEOUT_ENV",
    "BROKER_TOKEN_ENV",
    "MAX_ARCHIVE_BYTES",
    "MAX_COMMAND_ARGUMENT_BYTES",
    "MAX_COMMAND_ARGV_BYTES",
    "MAX_COMMAND_ENV_BYTES",
    "MAX_COMMAND_OUTPUT_BYTES",
    "MAX_COMMAND_STDIN_BYTES",
    "MAX_LEASE_SECONDS",
    "MAX_WAIT_SECONDS",
    "MIN_COMMAND_OUTPUT_BYTES",
    "MIN_LEASE_SECONDS",
    "SandboxAllocationFailedError",
    "SandboxArchive",
    "SandboxArchiveTooLargeError",
    "SandboxBrokerClient",
    "SandboxBrokerError",
    "SandboxBrokerRefusedError",
    "SandboxBrokerSettings",
    "SandboxBrokerUnavailableError",
    "SandboxBusyError",
    "SandboxCommand",
    "SandboxEgressMode",
    "SandboxFailure",
    "SandboxGuarantees",
    "SandboxImage",
    "SandboxLeaseLostError",
    "SandboxNotFoundError",
    "SandboxNotReadyError",
    "SandboxQuotaExceededError",
    "SandboxRecord",
]

#: Base URL of the Cloud API, for example ``https://cloud.cayu.dev``. Cloud injects it.
BROKER_API_URL_ENV = "CAYU_CLOUD_API_URL"
#: Per-Agent sandbox credential (Cloud injects it), or an Organization API key with
#: the ``sandboxes`` scope.
BROKER_TOKEN_ENV = "CAYU_CLOUD_SANDBOX_TOKEN"
#: The Agent (Cloud application) whose ``/sandboxes`` routes the client uses. Cloud injects it.
BROKER_APPLICATION_ID_ENV = "CAYU_CLOUD_APPLICATION_ID"
#: Optional per-request timeout for broker calls.
BROKER_REQUEST_TIMEOUT_ENV = "CAYU_CLOUD_SANDBOX_REQUEST_TIMEOUT_SECONDS"

MIN_LEASE_SECONDS = 60
MAX_LEASE_SECONDS = 3_600
MAX_WAIT_SECONDS = 20
MAX_COMMAND_STDIN_BYTES = 1024 * 1024
MIN_COMMAND_OUTPUT_BYTES = 1024
MAX_COMMAND_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_COMMAND_ARGV_ITEMS = 1_024
MAX_COMMAND_ARGV_BYTES = 256 * 1024
#: Linux's per-argument limit (MAX_ARG_STRLEN - 1); larger programs go through stdin.
MAX_COMMAND_ARGUMENT_BYTES = 131_071
MAX_COMMAND_ENV_ITEMS = 50
MAX_COMMAND_ENV_BYTES = 64 * 1024
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_ARCHIVE_PATHS = 1_000
MAX_ARCHIVE_EXCLUDES = 50
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
#: Worst-case growth of one output byte in the response JSON: a control byte
#: becomes a six-byte ``\u00XX`` escape. The broker measures each stream's
#: ``output_limit_bytes`` on that encoded form.
JSON_OUTPUT_EXPANSION = 6
# Each stream's limit is measured as it appears in the JSON, so two full
# streams plus the command's other fields bound a response.
_MAX_RESPONSE_BYTES = 2 * MAX_COMMAND_OUTPUT_BYTES + 64 * 1024
_MAX_ERROR_TEXT = 300
_LABEL_KEY = re.compile(r"[a-z0-9_.-]{1,63}")
_COMMAND_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_MAX_LABELS = 10
_MAX_LABEL_VALUE = 200
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_RETRY_DELAYS_SECONDS = (0.5, 1.0, 2.0)
#: Every code the contract names. A response without one is judged by its status.
_KNOWN_CODES = frozenset(
    {
        "scope_required",
        "sandbox_idempotency_conflict",
        "sandbox_invalid",
        "image_unknown",
        "egress_mode_unsupported",
        "sandbox_quota_exceeded",
        "sandboxes_unavailable",
        "sandbox_not_ready",
        "sandbox_busy",
        "command_id_conflict",
        "archive_too_large",
    }
)

SandboxEgressMode = Literal["none", "public_web"]
SandboxStatus = Literal["provisioning", "ready", "destroying", "destroyed", "failed"]
CommandStatus = Literal["running", "exited", "timed_out", "cancelled", "failed"]


class SandboxBrokerError(RuntimeError):
    """A broker call failed. ``code`` is the contract's error code when one was returned."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.detail = dict(detail or {})


class SandboxBrokerRefusedError(SandboxBrokerError):
    """The broker definitively refused the request with a contract code; nothing changed."""


class SandboxQuotaExceededError(SandboxBrokerRefusedError):
    """The Agent or Organization is at its concurrent sandbox limit."""

    @property
    def scope(self) -> str | None:
        value = self.detail.get("scope")
        return value if type(value) is str else None

    @property
    def limit(self) -> int | None:
        value = self.detail.get("limit")
        return value if type(value) is int else None


class SandboxArchiveTooLargeError(SandboxBrokerRefusedError):
    """An archive is over the per-request bound; ``bytes`` is what it would take."""

    @property
    def bytes(self) -> int | None:
        value = self.detail.get("bytes")
        return value if type(value) is int else None


class SandboxNotFoundError(SandboxBrokerRefusedError):
    """The sandbox or command does not exist for this caller."""


class SandboxNotReadyError(SandboxBrokerRefusedError):
    """The sandbox is not ``ready``, so it accepts no operation."""


class SandboxBusyError(SandboxBrokerRefusedError):
    """The sandbox already runs its maximum number of commands; this one did not start."""


class SandboxBrokerUnavailableError(SandboxBrokerError):
    """The broker could not be reached, or answered 408, 429 or 5xx without a contract code.

    These are retryable. ``outcome_unknown`` is true when the request may have
    been applied before it failed.
    """

    def __init__(self, message: str, *, outcome_unknown: bool = False, **kwargs: Any) -> None:
        super().__init__(message, **kwargs)
        self.outcome_unknown = outcome_unknown


class SandboxAllocationFailedError(SandboxBrokerError):
    """The broker reported the sandbox ``failed`` before it became ready."""


class SandboxLeaseLostError(SandboxBrokerError):
    """The sandbox's lease lapsed or the sandbox was destroyed underneath its user."""


class SandboxFailure(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    code: str
    message: str = ""


class SandboxEgress(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    mode: str


class SandboxGuarantees(BaseModel):
    """What Cloud enforces for a sandbox, as a versioned list of items."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    version: int
    items: tuple[str, ...] = ()

    def has(self, item: str) -> bool:
        # Version 1 items are never removed in later versions.
        return self.version >= 1 and item in self.items


class SandboxRecord(BaseModel):
    """The broker's sandbox record. ``provider`` is informational only."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(min_length=1)
    application_id: str | None = None
    image: str
    egress: SandboxEgress
    status: SandboxStatus
    failure: SandboxFailure | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    provider: str | None = None
    created_at: datetime | None = None
    ready_at: datetime | None = None
    lease_expires_at: datetime | None = None
    max_expires_at: datetime | None = None
    destroyed_at: datetime | None = None
    destroy_reason: str | None = None
    guarantees: SandboxGuarantees | None = None


class SandboxImage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    name: str
    description: str = ""
    toolchain: Any = None
    guarantees: SandboxGuarantees | None = None


class SandboxCommand(BaseModel):
    """One command's state. Output is complete only once ``finished``."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    command_id: str
    status: CommandStatus
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_ms: int | None = None

    @property
    def finished(self) -> bool:
        return self.status != "running"


@dataclass(frozen=True)
class SandboxArchive:
    """A tar read from a sandbox, and the named paths that did not exist."""

    data: bytes = field(repr=False)
    missing_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class SandboxBrokerSettings:
    """Where the broker is and how to authenticate to it.

    ``from_env`` reads ``CAYU_CLOUD_API_URL``, ``CAYU_CLOUD_APPLICATION_ID`` and
    ``CAYU_CLOUD_SANDBOX_TOKEN`` (all required; Cloud injects them into every
    Agent process) and the optional ``CAYU_CLOUD_SANDBOX_REQUEST_TIMEOUT_SECONDS``.
    The API URL must be HTTPS; plain HTTP is accepted only for a loopback host.
    """

    api_url: str
    application_id: str
    token: str = field(repr=False)
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        for name in ("api_url", "application_id", "token"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip() or value != value.strip():
                raise ValueError(f"Sandbox broker {name} must be a nonblank string.")
        parts = urlsplit(self.api_url)
        if parts.scheme not in {"https", "http"} or not parts.hostname:
            raise ValueError("Sandbox broker api_url must be an absolute HTTPS URL.")
        if parts.scheme == "http" and parts.hostname not in _LOCAL_HOSTS:
            raise ValueError("Sandbox broker api_url must use HTTPS outside loopback.")
        if parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Sandbox broker api_url must not carry credentials or a query.")
        if "/" in self.application_id:
            raise ValueError("Sandbox broker application_id must not contain '/'.")
        if any(ord(character) < 0x21 or ord(character) > 0x7E for character in self.token):
            raise ValueError("Sandbox broker token must be printable ASCII without spaces.")
        timeout = self.request_timeout_s
        if type(timeout) not in {int, float} or not isfinite(timeout) or timeout <= 0:
            raise ValueError("Sandbox broker request_timeout_s must be a positive number.")
        object.__setattr__(self, "api_url", self.api_url.rstrip("/"))
        object.__setattr__(self, "request_timeout_s", float(timeout))

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SandboxBrokerSettings:
        env = os.environ if environ is None else environ
        missing = [
            name
            for name in (BROKER_API_URL_ENV, BROKER_APPLICATION_ID_ENV, BROKER_TOKEN_ENV)
            if not env.get(name, "").strip()
        ]
        if missing:
            raise ValueError("Sandbox broker configuration is missing: " + ", ".join(missing) + ".")
        timeout_text = env.get(BROKER_REQUEST_TIMEOUT_ENV, "").strip()
        try:
            timeout = float(timeout_text) if timeout_text else DEFAULT_REQUEST_TIMEOUT_SECONDS
        except ValueError:
            raise ValueError(f"{BROKER_REQUEST_TIMEOUT_ENV} must be a number of seconds.") from None
        return cls(
            api_url=env[BROKER_API_URL_ENV].strip(),
            application_id=env[BROKER_APPLICATION_ID_ENV].strip(),
            token=env[BROKER_TOKEN_ENV].strip(),
            request_timeout_s=timeout,
        )


def validate_labels(labels: Mapping[str, str]) -> dict[str, str]:
    """Check labels against the contract before sending them."""

    if not isinstance(labels, Mapping):
        raise TypeError("Sandbox labels must be a mapping.")
    if len(labels) > _MAX_LABELS:
        raise ValueError(f"Sandbox labels allow at most {_MAX_LABELS} keys.")
    owned: dict[str, str] = {}
    for key, value in labels.items():
        if type(key) is not str or _LABEL_KEY.fullmatch(key) is None:
            raise ValueError("Sandbox label keys must match [a-z0-9_.-]{1,63}.")
        if type(value) is not str or len(value) > _MAX_LABEL_VALUE or not value.isprintable():
            raise ValueError(
                f"Sandbox label values must be printable text of at most {_MAX_LABEL_VALUE} "
                "characters."
            )
        owned[key] = value
    return owned


def validate_lease_seconds(lease_seconds: int) -> int:
    if (
        type(lease_seconds) is not int
        or not MIN_LEASE_SECONDS <= lease_seconds <= MAX_LEASE_SECONDS
    ):
        raise ValueError(
            f"Sandbox lease_seconds must be an integer from {MIN_LEASE_SECONDS} to "
            f"{MAX_LEASE_SECONDS}."
        )
    return lease_seconds


def validate_command_shape(
    argv: Sequence[str], env: Mapping[str, str] | None, stdin: str | None
) -> None:
    """Check a command against the broker's request bounds before it is sent."""

    if not argv or len(argv) > MAX_COMMAND_ARGV_ITEMS:
        raise ValueError(f"Sandbox command argv must hold 1 to {MAX_COMMAND_ARGV_ITEMS} strings.")
    if any(type(item) is not str or "\0" in item for item in argv):
        raise ValueError("Sandbox command argv must be strings without NUL bytes.")
    sizes = [len(item.encode("utf-8")) for item in argv]
    if any(size > MAX_COMMAND_ARGUMENT_BYTES for size in sizes):
        raise ValueError(
            f"Sandbox command argv arguments are limited to {MAX_COMMAND_ARGUMENT_BYTES} bytes "
            "each; pass larger programs through stdin."
        )
    if sum(sizes) > MAX_COMMAND_ARGV_BYTES:
        raise ValueError(f"Sandbox command argv exceeds {MAX_COMMAND_ARGV_BYTES} bytes.")
    variables = dict(env or {})
    if len(variables) > MAX_COMMAND_ENV_ITEMS:
        raise ValueError(f"Sandbox command env allows at most {MAX_COMMAND_ENV_ITEMS} variables.")
    env_bytes = sum(len(key.encode()) + len(value.encode()) + 2 for key, value in variables.items())
    if env_bytes > MAX_COMMAND_ENV_BYTES:
        raise ValueError(f"Sandbox command env exceeds {MAX_COMMAND_ENV_BYTES} bytes.")
    if stdin is not None and len(stdin.encode("utf-8")) > MAX_COMMAND_STDIN_BYTES:
        raise ValueError(f"Sandbox command stdin exceeds {MAX_COMMAND_STDIN_BYTES} bytes.")


def relative_workspace_path(path: str | None) -> str | None:
    """Return ``path`` relative to ``/workspace``, or ``None`` for the workspace itself."""

    if path is None:
        return None
    if type(path) is not str:
        raise TypeError("Sandbox workspace paths must be strings.")
    if path.startswith("/"):
        raise ValueError("Sandbox workspace paths must be relative and stay inside /workspace.")
    # Whitespace is part of a path name; trimming it would alias distinct paths.
    normalized = path.rstrip("/")
    if normalized in {"", "."}:
        return None
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("Sandbox workspace paths must be relative and stay inside /workspace.")
    return normalized


class SandboxBrokerClient:
    """Async client for one Agent's ``/v1/applications/{agent}/sandboxes`` routes.

    Every call is replay-safe by the contract (create by idempotency key, a
    command by its ``command_id``, archive upload by its body), so transient
    failures are retried a few times: transport errors, and responses without
    a contract code whose status is 408, 429 or 5xx. Any other 4xx without a
    code (a revoked credential's 401, another Agent's 404) is terminal.
    """

    def __init__(
        self,
        settings: SandboxBrokerSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if type(settings) is not SandboxBrokerSettings:
            raise TypeError("SandboxBrokerClient requires SandboxBrokerSettings.")
        self.settings = settings
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._base = (
            f"{settings.api_url}/v1/applications/{quote(settings.application_id, safe='')}"
            "/sandboxes"
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> SandboxBrokerClient:
        return cls(SandboxBrokerSettings.from_env(environ))

    def __repr__(self) -> str:
        return (
            f"SandboxBrokerClient(api_url={self.settings.api_url!r}, "
            f"application_id={self.settings.application_id!r})"
        )

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.aclose()

    async def __aenter__(self) -> SandboxBrokerClient:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    # Sandboxes ------------------------------------------------------------

    async def list_images(self) -> list[SandboxImage]:
        body, _ = await self._json("GET", f"{self.settings.api_url}/v1/sandbox-images")
        items = body.get("images", body.get("items")) if isinstance(body, Mapping) else body
        if not isinstance(items, list):
            raise SandboxBrokerError("Sandbox broker image list has an unexpected shape.")
        return [self._model(SandboxImage, item) for item in items]

    async def create(
        self,
        *,
        idempotency_key: str,
        image: str,
        egress_mode: SandboxEgressMode,
        lease_seconds: int,
        labels: Mapping[str, str] | None = None,
    ) -> SandboxRecord:
        if (
            type(idempotency_key) is not str
            or not 1 <= len(idempotency_key) <= 200
            or not idempotency_key.isprintable()
        ):
            raise ValueError("Sandbox idempotency_key must be 1 to 200 printable characters.")
        if type(image) is not str or not image.strip():
            raise ValueError("Sandbox image must name a catalog entry.")
        if egress_mode not in ("none", "public_web"):
            raise ValueError("Sandbox egress_mode must be 'none' or 'public_web'.")
        payload = {
            "idempotency_key": idempotency_key,
            "image": image,
            "egress": {"mode": egress_mode},
            "lease_seconds": validate_lease_seconds(lease_seconds),
            "labels": validate_labels(labels or {}),
        }
        body, _ = await self._json("POST", self._base, payload=payload)
        return self._model(SandboxRecord, body)

    async def get(self, sandbox_id: str) -> SandboxRecord:
        body, _ = await self._json("GET", self._sandbox_url(sandbox_id))
        return self._model(SandboxRecord, body)

    async def list_sandboxes(
        self,
        *,
        status: SandboxStatus | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[SandboxRecord], str | None]:
        params: dict[str, str] = {"limit": str(limit)}
        if status is not None:
            params["status"] = status
        if cursor is not None:
            params["cursor"] = cursor
        body, _ = await self._json("GET", self._base, params=params)
        if not isinstance(body, Mapping) or not isinstance(body.get("items"), list):
            raise SandboxBrokerError("Sandbox broker list has an unexpected shape.")
        next_cursor = body.get("next_cursor")
        return (
            [self._model(SandboxRecord, item) for item in body["items"]],
            next_cursor if type(next_cursor) is str else None,
        )

    async def events(self, sandbox_id: str) -> list[dict[str, Any]]:
        body, _ = await self._json("GET", self._sandbox_url(sandbox_id) + "/events")
        items = body.get("items") if isinstance(body, Mapping) else body
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise SandboxBrokerError("Sandbox broker events have an unexpected shape.")
        return items

    async def wait_until_ready(
        self,
        sandbox_id: str,
        *,
        timeout_s: float,
        poll_interval_s: float = 1.0,
    ) -> SandboxRecord:
        """Poll until ``ready``; raise when the sandbox fails, ends, or the timeout passes.

        Cloud fails a sandbox still provisioning after five minutes, so a
        ``timeout_s`` a little above that always sees a terminal answer.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            record = await self.get(sandbox_id)
            if record.status == "ready":
                return record
            if record.status == "failed":
                failure = record.failure
                raise SandboxAllocationFailedError(
                    "Sandbox allocation failed"
                    + (f": {failure.code}" if failure is not None else "."),
                    code=None if failure is None else failure.code,
                )
            if record.status in {"destroying", "destroyed"}:
                raise SandboxLeaseLostError(
                    f"Sandbox ended before it was ready ({record.destroy_reason or 'unknown'}).",
                    code=record.destroy_reason,
                )
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise SandboxBrokerUnavailableError(
                    f"Sandbox was not ready within {timeout_s:g} seconds."
                )
            await asyncio.sleep(min(poll_interval_s, remaining))

    async def renew(self, sandbox_id: str, *, lease_seconds: int) -> SandboxRecord:
        body, _ = await self._json(
            "POST",
            self._sandbox_url(sandbox_id) + "/renew",
            payload={"lease_seconds": validate_lease_seconds(lease_seconds)},
        )
        return self._model(SandboxRecord, body)

    async def destroy(self, sandbox_id: str) -> SandboxRecord:
        """Destroy the sandbox. Idempotent; an unknown sandbox raises ``SandboxNotFoundError``."""

        body, _ = await self._json("DELETE", self._sandbox_url(sandbox_id))
        return self._model(SandboxRecord, body)

    # Commands -------------------------------------------------------------

    async def start_command(
        self,
        sandbox_id: str,
        command_id: str,
        argv: Sequence[str],
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_seconds: int = 300,
        stdin: str | None = None,
        output_limit_bytes: int = 1024 * 1024,
        retry: bool = True,
    ) -> SandboxCommand:
        """Start a command, or return the existing one started under the same id and body.

        With ``retry=False`` exactly one HTTP request is sent, so a caller that
        tracks which dispatches may have reached the broker sees every one.
        """

        if type(command_id) is not str or _COMMAND_ID.fullmatch(command_id) is None:
            raise ValueError("Sandbox command_id must be 1 to 128 of [A-Za-z0-9._:-].")
        validate_command_shape(argv, env, stdin)
        if type(timeout_seconds) is not int or timeout_seconds < 1:
            raise ValueError("Sandbox command timeout_seconds must be a positive integer.")
        if (
            type(output_limit_bytes) is not int
            or not MIN_COMMAND_OUTPUT_BYTES <= output_limit_bytes <= MAX_COMMAND_OUTPUT_BYTES
        ):
            raise ValueError(
                f"Sandbox command output_limit_bytes must be from {MIN_COMMAND_OUTPUT_BYTES} to "
                f"{MAX_COMMAND_OUTPUT_BYTES}."
            )
        payload: dict[str, Any] = {
            "command_id": command_id,
            "argv": list(argv),
            "env": dict(env or {}),
            "timeout_seconds": timeout_seconds,
            "stdin": stdin,
            "output_limit_bytes": output_limit_bytes,
        }
        relative_cwd = relative_workspace_path(cwd)
        if relative_cwd is not None:
            payload["cwd"] = relative_cwd
        body, _ = await self._json(
            "POST", self._sandbox_url(sandbox_id) + "/commands", payload=payload, retry=retry
        )
        return self._model(SandboxCommand, body)

    async def get_command(
        self, sandbox_id: str, command_id: str, *, wait_seconds: int = 0
    ) -> SandboxCommand:
        if type(wait_seconds) is not int or not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
            raise ValueError(f"wait_seconds must be from 0 to {MAX_WAIT_SECONDS}.")
        body, _ = await self._json(
            "GET",
            self._command_url(sandbox_id, command_id),
            params={"wait_seconds": str(wait_seconds)},
            timeout_s=self.settings.request_timeout_s + wait_seconds,
        )
        return self._model(SandboxCommand, body)

    async def cancel_command(self, sandbox_id: str, command_id: str) -> SandboxCommand:
        body, _ = await self._json("POST", self._command_url(sandbox_id, command_id) + "/cancel")
        return self._model(SandboxCommand, body)

    # Archives -------------------------------------------------------------

    async def put_archive(self, sandbox_id: str, data: bytes, *, path: str | None = None) -> dict:
        """Extract a tar of at most ``MAX_ARCHIVE_BYTES`` under ``/workspace/<path>``."""

        if type(data) is not bytes:
            raise TypeError("Sandbox archive must be bytes.")
        if len(data) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"Sandbox archive exceeds {MAX_ARCHIVE_BYTES} bytes.")
        body, _ = await self._json(
            "PUT",
            self._sandbox_url(sandbox_id) + "/archive",
            params={"path": relative_workspace_path(path) or "."},
            content=data,
            headers={"Content-Type": "application/x-tar"},
        )
        if not isinstance(body, dict):
            raise SandboxBrokerError("Sandbox broker archive upload has an unexpected shape.")
        return body

    async def get_archive(
        self,
        sandbox_id: str,
        *,
        paths: Sequence[str] = (),
        excludes: Sequence[str] = (),
    ) -> SandboxArchive:
        """Read a tar of ``paths`` (files or directories; default the whole workspace).

        ``excludes`` names directory names skipped wherever they occur. Paths
        that don't exist are reported in ``missing_paths``. Beyond the
        per-request bound it raises ``SandboxArchiveTooLargeError`` carrying
        the size the paths would take.
        """

        if len(paths) > MAX_ARCHIVE_PATHS:
            raise ValueError(f"Sandbox archive reads name at most {MAX_ARCHIVE_PATHS} paths.")
        if len(excludes) > MAX_ARCHIVE_EXCLUDES:
            raise ValueError(f"Sandbox archive reads exclude at most {MAX_ARCHIVE_EXCLUDES} names.")
        query: list[tuple[str, str]] = [
            ("path", relative_workspace_path(path) or ".") for path in paths
        ]
        for name in excludes:
            if type(name) is not str or not name or "/" in name:
                raise ValueError("Sandbox archive excludes must be directory names.")
            query.append(("exclude", name))
        url = self._sandbox_url(sandbox_id) + "/archive"
        if query:
            url += "?" + urlencode(query)
        data, headers = await self._request(
            "GET",
            url,
            headers={"Accept": "application/x-tar"},
            max_bytes=MAX_ARCHIVE_BYTES + 64 * 1024,
        )
        missing = tuple(
            unquote(item.strip())
            for item in headers.get("x-cayu-missing-paths", "").split(",")
            if item.strip()
        )
        return SandboxArchive(data=data, missing_paths=missing)

    # Transport ------------------------------------------------------------

    def _sandbox_url(self, sandbox_id: str) -> str:
        if type(sandbox_id) is not str or not sandbox_id or "/" in sandbox_id:
            raise ValueError("Sandbox id must be a nonblank opaque string.")
        return f"{self._base}/{quote(sandbox_id, safe='')}"

    def _command_url(self, sandbox_id: str, command_id: str) -> str:
        if type(command_id) is not str or _COMMAND_ID.fullmatch(command_id) is None:
            raise ValueError("Sandbox command_id must be 1 to 128 of [A-Za-z0-9._:-].")
        return f"{self._sandbox_url(sandbox_id)}/commands/{quote(command_id, safe='')}"

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            if self._transport is not None:
                self._client = httpx.AsyncClient(transport=self._transport)
            else:
                self._client = httpx.AsyncClient(
                    verify=ssl.create_default_context(cafile=certifi.where())
                )
        return self._client

    @staticmethod
    def _model(model: type[BaseModel], body: Any) -> Any:
        try:
            return model.model_validate(body)
        except ValidationError:
            raise SandboxBrokerError(
                f"Sandbox broker returned an invalid {model.__name__}."
            ) from None

    async def _json(self, method: str, url: str, **kwargs: Any) -> tuple[Any, httpx.Headers]:
        raw, headers = await self._request(method, url, max_bytes=_MAX_RESPONSE_BYTES, **kwargs)
        try:
            return json.loads(raw), headers
        except ValueError:
            # The status said the request was applied; only its answer was lost.
            raise SandboxBrokerUnavailableError(
                "Sandbox broker returned invalid JSON.", outcome_unknown=True
            ) from None

    async def _request(
        self,
        method: str,
        url: str,
        *,
        max_bytes: int,
        payload: Any = None,
        params: Mapping[str, str] | None = None,
        content: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
        retry: bool = True,
    ) -> tuple[bytes, httpx.Headers]:
        request_headers = {
            "Authorization": f"Bearer {self.settings.token}",
            "Accept": "application/json",
            **(headers or {}),
        }
        timeout = self.settings.request_timeout_s if timeout_s is None else timeout_s
        attempts = 1 + len(_RETRY_DELAYS_SECONDS) if retry else 1
        for attempt in range(attempts):
            try:
                async with self._http().stream(
                    method,
                    url,
                    json=payload,
                    params=params,
                    content=content,
                    headers=request_headers,
                    timeout=timeout,
                ) as response:
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(body) + len(chunk) > max_bytes:
                            raise SandboxBrokerError(
                                "Sandbox broker response exceeds its byte limit.",
                                status=response.status_code,
                            )
                        body.extend(chunk)
                    status = response.status_code
                    response_headers = response.headers
            except httpx.TransportError as exc:
                if attempt + 1 < attempts:
                    await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt])
                    continue
                # A refused connection never reached the broker; anything later might have.
                raise SandboxBrokerUnavailableError(
                    f"Sandbox broker request failed: {type(exc).__name__}.",
                    outcome_unknown=not isinstance(exc, httpx.ConnectError),
                ) from None
            if status < 400:
                return bytes(body), response_headers
            error = _broker_error(status, bytes(body))
            if isinstance(error, SandboxBrokerUnavailableError) and attempt + 1 < attempts:
                await asyncio.sleep(_RETRY_DELAYS_SECONDS[attempt])
                continue
            raise error
        raise AssertionError("unreachable broker retry loop")


def _broker_error(status: int, body: bytes) -> SandboxBrokerError:
    code: str | None = None
    message = ""
    detail: dict[str, Any] = {}
    try:
        decoded = json.loads(body) if body else {}
    except ValueError:
        decoded = {}
    error = decoded.get("error") if isinstance(decoded, Mapping) else None
    if isinstance(error, Mapping):
        raw_code = error.get("code")
        code = raw_code if type(raw_code) is str and raw_code in _KNOWN_CODES else None
        raw_message = error.get("message")
        if type(raw_message) is str:
            message = raw_message[:_MAX_ERROR_TEXT]
        raw_detail = error.get("detail")
        if isinstance(raw_detail, Mapping):
            detail = {
                key: value
                for key, value in raw_detail.items()
                if type(key) is str and type(value) in {str, int, bool}
            }
    text = f"Sandbox broker answered HTTP {status}" + (f" {code}" if code else "")
    if message:
        text += f": {message}"
    kwargs: dict[str, Any] = {"status": status, "code": code, "detail": detail}
    if code is None:
        # Authentication, a proxy or the ingress answered: judge by status.
        # 408, 429 and 5xx are retryable; every other 4xx is terminal.
        if status in {408, 429} or status >= 500:
            return SandboxBrokerUnavailableError(text, outcome_unknown=True, **kwargs)
        if status == 404:
            return SandboxNotFoundError(text, **kwargs)
        return SandboxBrokerRefusedError(text, **kwargs)
    error_type = {
        "sandbox_quota_exceeded": SandboxQuotaExceededError,
        "archive_too_large": SandboxArchiveTooLargeError,
        "sandbox_not_ready": SandboxNotReadyError,
        "sandbox_busy": SandboxBusyError,
    }.get(code, SandboxBrokerRefusedError)
    return error_type(text, **kwargs)
