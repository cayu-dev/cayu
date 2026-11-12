"""Authenticated client for the versioned Cayu Cloud customer API."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlencode, urlsplit

import httpx

from cayu.cli._cloud_diagnostics import safe_text

_SAFE_OBJECT_STORE_ERROR_CODES = frozenset(
    {
        "AccessDenied",
        "AuthenticationFailed",
        "AuthorizationQueryParametersError",
        "ExpiredToken",
        "InvalidArgument",
        "InvalidRequest",
        "RequestExpired",
        "SignatureDoesNotMatch",
    }
)
_SAFE_API_ERROR_DETAILS = {
    (422, "agent_slug_invalid"): (
        "Agent application slugs must be 8-63 lowercase letters, digits, or interior hyphens."
    ),
    (422, "manifest_invalid"): "Cayu Cloud rejected the manifest resources.",
    (422, "invalid_cursor"): (
        "The cursor isn't one this listing returned; list again without --cursor."
    ),
    (403, "organization_admin_required"): (
        "A signed-in organization administrator is required; run `cayu cloud login`."
    ),
    (409, "application_archived"): "The Agent is archived; it can't be deployed or changed.",
    (409, "application_revision_stale"): (
        "The Agent changed; read its revision again before archiving."
    ),
    (409, "idempotency_conflict"): (
        "The idempotency key was already used for a different request."
    ),
    (409, "operator_credentials_not_provisioned"): (
        "The Agent has no Developer dashboard credentials yet. Cayu Cloud creates them "
        "when it next publishes the Agent; deploy it again with `cayu cloud deploy`."
    ),
    (503, "organization_directory_unavailable"): (
        "Administrator access can't be confirmed right now; nothing changed. Run the same "
        "command again."
    ),
}


# Cayu Cloud's inbox refusals (`/v1/applications/{id}/inbox/...`), including the Agent's own
# conflicts it passes on with the item's current state.
_INBOX_REFUSALS = frozenset(
    {
        (403, "agent_not_yours"),
        (403, "agent_refused"),
        (403, "inbox_decision_forbidden"),
        (403, "person_required"),
        (403, "review_refused"),
        (404, "item_not_found"),
        (409, "agent_not_serving"),
        (409, "agent_sign_in_required"),
        (409, "already_settled"),
        (409, "conflict"),
        (409, "expired"),
        (409, "identity_refused"),
        (409, "not_settleable"),
        (409, "profile_changed"),
        (409, "operator_login_missing"),
        (409, "not_reviewable"),
        (409, "review_not_served"),
        (409, "review_stale"),
        (409, "superseded"),
        (409, "unavailable"),
        (422, "decision_refused"),
        (422, "invalid_decision"),
        (502, "decision_outcome_unknown"),
        (503, "agent_sign_in_unavailable"),
        (503, "agent_starting"),
        (503, "inbox_unavailable"),
        (503, "service_status_unavailable"),
        (503, "timeout"),
        (503, "unreachable"),
    }
)

# Cloud's documented refusals of a candidate release, whose own message names the releases.
CANDIDATE_REFUSAL_CODES = frozenset({"candidate_baseline_changed", "candidate_expired"})

# Cloud's documented refusals whose code and own message the CLI reports, filtered like a
# conflict's reason, since the message says what to fix: Compass's (`cayu cloud compass`),
# a schedule run's (`cayu cloud schedule run`), a settings change's (`cayu cloud
# settings`), an environment change's (`cayu cloud env set|unset` with several names), a
# source or SBOM download's (`cayu cloud deployment source|sbom`) and an inbox decision's
# (`cayu inbox`), whose structured fields (such as a refused settle's `item`) that command
# reads from `CloudApiError.body`.
_REPORTED_API_REFUSALS = frozenset(
    {
        *_INBOX_REFUSALS,
        (403, "compass_disabled"),
        (403, "developer_required"),
        (403, "key_scope_required"),
        (404, "agent_not_watched"),
        (404, "not_enrolled"),
        (404, "release_not_found"),
        (404, "schedule_not_found"),
        (409, "agent_inactive"),
        (409, "agent_not_enrollable"),
        (409, "agents_required"),
        (409, "disabled"),
        (409, "enrollment_incomplete"),
        (409, "fork_controller_conflict"),
        (409, "idempotency_key_reused"),
        (409, "no_schedule"),
        (409, "no_selected_release"),
        (409, "release_source_unreadable"),
        (409, "service_busy"),
        # `cayu cloud service check`: why Cloud couldn't run the Agent's operator check.
        (409, "agent_not_active"),
        (409, "no_web_service"),
        (409, "operator_check_not_declared"),
        (409, "service_not_running"),
        (502, "agent_unreachable"),
        (503, "operator_credentials_unavailable"),
        # `cayu cloud admin arena-enrollments`: why Cloud couldn't list Arena's enrollments;
        # each message names the setting to fix.
        (502, "arena_inventory_refused"),
        (503, "arena_inventory_not_configured"),
        (503, "arena_inventory_unavailable"),
        (503, "arena_inventory_unconfigured"),
        (503, "arena_inventory_unsupported"),
        (504, "operator_check_timeout"),
        # `cayu cloud service database`: why Cloud has no database health for the Agent.
        (404, "agent_database_not_found"),
        (503, "agent_database_health_unavailable"),
        # `cayu cloud applications export|import`: why Cloud refused an export or import.
        (403, "admin_required"),
        (404, "application_not_found"),
        (404, "export_not_found"),
        (404, "import_not_found"),
        (409, "agent_data_transfer_in_progress"),
        (409, "agent_not_deployed"),
        (409, "export_not_ready"),
        (409, "target_not_stopped"),
        (503, "agent_data_transfers_unavailable"),
        # `cayu cloud connections`: why Cloud refused a connection or a binding.
        (404, "agent_not_found"),
        (404, "connection_not_found"),
        (404, "slot_unbound"),
        (409, "connection_bound"),
        (409, "connection_exists"),
        (422, "agent_unknown"),
        (422, "agents_required"),
        (422, "connection_unavailable"),
        (422, "connector_not_connectable"),
        (422, "connector_unavailable"),
        (422, "credential_invalid"),
        (422, "invalid_name"),
        (422, "scope_not_allowed"),
        (422, "scope_required"),
        (422, "too_many_agents"),
        # `cayu cloud models` and `connections bind --model`: why Cloud refused a model slot.
        (409, "model_not_approved"),
        (409, "models_unavailable"),
        (422, "invalid_model"),
        (503, "models_unavailable"),
        (503, "models_unreadable"),
        # `cayu cloud admin runtime-record`: why Cloud refused to read or change the record.
        (404, "runtime_record_not_found"),
        (409, "database_migration_running"),
        (409, "runtime_record_changed"),
        (503, "application_runtime_unavailable"),
        (409, "settings_revision_conflict"),
        (409, "source_not_uploaded"),
        (410, "source_bundle_expired"),
        # `cayu cloud deployment sbom`: why Cloud has no SBOM to send.
        (404, "release_sbom_not_found"),
        (410, "release_sbom_removed"),
        (503, "release_sboms_unavailable"),
        (422, "environment_change_invalid"),
        (422, "invalid_setting"),
        (422, "invalid_settings"),
        (422, "log_query_invalid"),
        (422, "no_objectives"),
        (422, "release_too_large"),
        (503, "agent_domain_unknown"),
        (503, "compass_unavailable"),
        (503, "release_source_unavailable"),
        (503, "source_bundles_unavailable"),
        (504, "compass_outcome_unknown"),
    }
)

# A binary download is written in parts of this size; a refusal is read up to the limit.
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_ERROR_BODY_LIMIT = 64 * 1024


@dataclass(frozen=True)
class DownloadedFile:
    """What `CloudApiClient.download` wrote: the answer's headers, size and SHA-256."""

    headers: dict[str, str]
    size: int
    sha256: str


# Cloud's stable rejection codes are short lowercase identifiers.
_CLOUD_ERROR_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_CONFLICT_DETAIL_LIMIT = 512


def archived_agent_error() -> CloudApiError:
    """The documented refusal for an archived Agent, raised when the CLI itself sees it."""

    return CloudApiError(
        "api_request_rejected",
        _SAFE_API_ERROR_DETAILS[(409, "application_archived")],
        code="application_archived",
    )


class CloudApiError(RuntimeError):
    """Stable customer-facing API failure."""

    def __init__(
        self,
        category: str,
        message: str,
        *,
        status_code: int | None = None,
        detail: str | None = None,
        code: str | None = None,
        rejected_fields: tuple[tuple[str, str], ...] = (),
        body: dict[str, Any] | None = None,
        transport_error: str | None = None,
        request: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        # Cloud's stable rejection code, only when it is one of the documented safe codes.
        self.code = code
        # The server's plain-text rejection reason, when it passes the safe-text filter.
        # Callers opt in to showing it; the default message never includes it.
        self.detail = detail
        # The request fields an HTTP 422 names, as (dotted path, error type) pairs such as
        # ("body.candidate", "extra_forbidden").
        self.rejected_fields = rejected_fields
        # Cloud's structured rejection (FastAPI's `detail` object), only with a documented
        # `code`. Its other fields are Cloud's, so a caller checks each one it reads.
        self.body = body
        # For `api_unavailable`: how the request failed before Cloud answered (a name from
        # `_TRANSPORT_ERRORS`) and which request it was, as "METHOD /path" without the query.
        # Neither carries a credential or the exception's own text.
        self.transport_error = transport_error
        self.request = request


# Most specific first: httpx's timeouts are transport errors too.
_TRANSPORT_ERRORS: tuple[tuple[type[httpx.RequestError], str], ...] = (
    (httpx.ConnectTimeout, "connect_timeout"),
    (httpx.ReadTimeout, "read_timeout"),
    (httpx.WriteTimeout, "write_timeout"),
    (httpx.PoolTimeout, "pool_timeout"),
    (httpx.ConnectError, "connect_failed"),
    (httpx.RemoteProtocolError, "connection_closed"),
    (httpx.ReadError, "read_failed"),
    (httpx.WriteError, "write_failed"),
)


def _unavailable(exc: httpx.RequestError, request: str) -> CloudApiError:
    """Cloud couldn't be reached for ``request``. The exception's own text can carry the full
    URL, so only its kind is kept."""

    transport_error = _transport_error(exc)
    return CloudApiError(
        "api_unavailable",
        f"Cayu Cloud API is unavailable ({transport_error.replace('_', ' ')} on {request}).",
        transport_error=transport_error,
        request=request,
    )


def _transport_error(exc: httpx.RequestError) -> str:
    return next(
        (name for kind, name in _TRANSPORT_ERRORS if isinstance(exc, kind)), "request_failed"
    )


@dataclass(frozen=True)
class CloudApiClient:
    api_url: str
    api_key: str
    timeout_seconds: float = 30.0
    api_key_provider: Callable[[], str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    # Given a key Cloud rejected with HTTP 401, return a replacement (for example a
    # refreshed login token), or the same key when none is available.
    rejected_api_key_provider: Callable[[str], str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        parsed = urlsplit(self.api_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("A canonical HTTP(S) Cayu Cloud API URL is required.")
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
            raise ValueError("Cayu Cloud API URL must use HTTPS outside loopback.")
        if (
            not self.api_key
            or self.api_key != self.api_key.strip()
            or any(character.isspace() for character in self.api_key)
        ):
            raise ValueError("Cayu Cloud API key must be a non-empty canonical value.")
        if self.timeout_seconds <= 0:
            raise ValueError("API timeout must be positive.")

    @classmethod
    def from_key_file(
        cls,
        *,
        api_url: str,
        api_key_file: Path,
        timeout_seconds: float = 30.0,
    ) -> CloudApiClient:
        try:
            api_key = api_key_file.read_text().strip()
        except OSError as exc:
            raise CloudApiError(
                "api_key_unavailable",
                "Could not read the Cayu Cloud API key file.",
            ) from exc
        return cls(
            api_url=api_url.rstrip("/"),
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
        idempotency_key: str | None = None,
        query: Mapping[str, str | Sequence[str]] | None = None,
    ) -> dict[str, Any]:
        api_key = self._current_api_key()
        response = self._send(
            method,
            path,
            api_key=api_key,
            payload=payload,
            idempotency_key=idempotency_key,
            query=query,
        )
        if response.status_code == 401 and self.rejected_api_key_provider is not None:
            # Cloud rejected the credential before acting on the request, so sending it
            # once more with a replacement is safe for every method.
            replacement = self._validated_api_key(self.rejected_api_key_provider(api_key))
            if replacement != api_key:
                response = self._send(
                    method,
                    path,
                    api_key=replacement,
                    payload=payload,
                    idempotency_key=idempotency_key,
                    query=query,
                )
        if not 200 <= response.status_code < 300:
            raise _rejection(response) from None
        if response.status_code == 204:
            return {}
        try:
            result = response.json()
        except (TypeError, ValueError, json.JSONDecodeError):
            raise CloudApiError(
                "api_response_invalid",
                "Cayu Cloud API returned invalid JSON.",
            ) from None
        if not isinstance(result, dict):
            raise CloudApiError(
                "api_response_invalid",
                "Cayu Cloud API returned an unsupported response.",
            )
        return result

    def download(
        self,
        path: str,
        destination: BinaryIO,
        *,
        max_bytes: int,
        accept: str = "application/gzip",
    ) -> DownloadedFile:
        """Stream a binary answer of at most ``max_bytes`` into ``destination``.

        Cloud's refusals are JSON and raise the same `CloudApiError` as `request`. The body
        is written as it arrives, so a large answer is never held in memory, and its SHA-256
        digest is computed on the way.
        """

        api_key = self._current_api_key()
        request = f"GET {path}"
        try:
            with httpx.Client(follow_redirects=False, timeout=self.timeout_seconds) as client:
                for attempt in range(2):
                    with client.stream(
                        "GET",
                        self.api_url.rstrip("/") + path,
                        headers={
                            "Accept": f"{accept}, application/json",
                            "Authorization": f"Bearer {api_key}",
                        },
                    ) as response:
                        if (
                            response.status_code == 401
                            and attempt == 0
                            and self.rejected_api_key_provider is not None
                        ):
                            replacement = self._validated_api_key(
                                self.rejected_api_key_provider(api_key)
                            )
                            if replacement != api_key:
                                api_key = replacement
                                continue
                        if not 200 <= response.status_code < 300:
                            raise _rejection(_bounded_error_response(response)) from None
                        digest = hashlib.sha256()
                        written = 0
                        for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                            written += len(chunk)
                            if written > max_bytes:
                                raise CloudApiError(
                                    "api_response_too_large",
                                    f"Cayu Cloud answered {request} with more than "
                                    f"{max_bytes} bytes.",
                                )
                            digest.update(chunk)
                            destination.write(chunk)
                        return DownloadedFile(
                            # httpx already gives the names in lowercase.
                            headers=dict(response.headers.items()),
                            size=written,
                            sha256=digest.hexdigest(),
                        )
        except httpx.RequestError as exc:
            raise _unavailable(exc, request) from None
        raise AssertionError("unreachable")  # pragma: no cover

    def _send(
        self,
        method: str,
        path: str,
        *,
        api_key: str,
        payload: dict[str, object] | None,
        idempotency_key: str | None,
        query: Mapping[str, str | Sequence[str]] | None,
    ) -> httpx.Response:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=self.timeout_seconds,
            ) as client:
                return client.request(
                    method,
                    # A sequence value repeats its name (`label=a=1&label=b=2`).
                    self.api_url.rstrip("/")
                    + path
                    + (f"?{urlencode(query, doseq=True)}" if query else ""),
                    content=body,
                    headers=headers,
                )
        except httpx.RequestError as exc:
            raise _unavailable(exc, f"{method} {path}") from None

    def _current_api_key(self) -> str:
        return self._validated_api_key(
            self.api_key_provider() if self.api_key_provider is not None else self.api_key
        )

    @staticmethod
    def _validated_api_key(api_key: str) -> str:
        if (
            not api_key
            or api_key != api_key.strip()
            or any(character.isspace() for character in api_key)
        ):
            raise CloudApiError(
                "api_key_unavailable",
                "Could not refresh the Cayu Cloud login.",
            )
        return api_key

    def upload_bytes(
        self,
        upload_url: str,
        content: bytes,
        *,
        content_type: str = "application/gzip",
    ) -> None:
        parsed = urlsplit(upload_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise CloudApiError("source_upload_invalid", "Source upload URL is invalid.")
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
            raise CloudApiError(
                "source_upload_invalid",
                "Source upload URL must use HTTPS outside loopback.",
            )
        try:
            with httpx.Client(
                follow_redirects=False,
                timeout=max(self.timeout_seconds, 120.0),
            ) as client:
                response = client.put(
                    upload_url,
                    content=content,
                    headers={"Content-Type": content_type},
                )
        except httpx.RequestError:
            raise CloudApiError(
                "source_upload_failed",
                "Local source bundle upload failed.",
            ) from None
        if not 200 <= response.status_code < 300:
            detail = _object_store_error(response.content)
            suffix = f": {detail.rstrip('.')}" if detail else ""
            raise CloudApiError(
                "source_upload_rejected",
                f"Local source bundle upload returned HTTP {response.status_code}{suffix}.",
                status_code=response.status_code,
            )


def _rejection(response: httpx.Response) -> CloudApiError:
    """The `CloudApiError` for Cloud's non-2xx answer, keeping only its documented parts."""

    structured = _structured_api_error(response)
    detail = _safe_api_error_detail(response.status_code, structured)
    if detail is None and structured is None:
        detail = _safe_conflict_detail(response)
    suffix = f": {detail}" if detail is not None else "."
    code = _safe_api_error_code(response.status_code, structured)
    return CloudApiError(
        "api_request_rejected",
        f"Cayu Cloud API returned HTTP {response.status_code}{suffix}",
        status_code=response.status_code,
        # A documented candidate refusal's own message is its plain-text reason.
        detail=detail if code in CANDIDATE_REFUSAL_CODES else _plain_api_error_detail(response),
        code=code,
        rejected_fields=_rejected_fields(response),
        body=None if code is None or structured is None else structured[1],
    )


def _bounded_error_response(response: httpx.Response) -> httpx.Response:
    """A streamed refusal's first bytes as a complete response the error helpers can read."""

    content = bytearray()
    for chunk in response.iter_bytes():
        content.extend(chunk)
        if len(content) >= _ERROR_BODY_LIMIT:
            del content[_ERROR_BODY_LIMIT:]
            break
    content_type = response.headers.get("content-type")
    return httpx.Response(
        response.status_code,
        headers={} if content_type is None else {"content-type": content_type},
        content=bytes(content),
    )


def _object_store_error(content: bytes) -> str | None:
    """Return only an allowlisted provider code, never provider-controlled detail."""

    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return None
    code = root.findtext("Code")
    if code is None:
        return None
    normalized = code.strip()
    return normalized if normalized in _SAFE_OBJECT_STORE_ERROR_CODES else None


def _structured_api_error(response: httpx.Response) -> tuple[str, dict[str, Any]] | None:
    try:
        payload = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    detail = payload.get("detail")
    if not isinstance(detail, dict):
        return None
    code = detail.get("code")
    return (code, detail) if type(code) is str else None


def _safe_api_error_code(
    status_code: int, structured: tuple[str, dict[str, Any]] | None
) -> str | None:
    """Return Cloud's rejection code only when it is a documented, non-secret code."""

    if structured is None:
        return None
    code = structured[0]
    documented = (
        (status_code, code) in _SAFE_API_ERROR_DETAILS
        or (status_code, code) in _REPORTED_API_REFUSALS
        or (status_code == 409 and code in CANDIDATE_REFUSAL_CODES)
    )
    return code if documented else None


def _safe_api_error_detail(
    status_code: int, structured: tuple[str, dict[str, Any]] | None
) -> str | None:
    """Return only a versioned, non-secret customer API validation message."""

    if structured is None:
        return None
    code, detail = structured
    if status_code == 422 and code == "manifest_invalid":
        pairs = detail.get("valid_pairs")
        if (
            isinstance(pairs, list)
            and 1 <= len(pairs) <= 3
            and all(
                isinstance(pair, dict)
                and set(pair) == {"cpu_millis", "memory_mb"}
                and type(pair["cpu_millis"]) is int
                and type(pair["memory_mb"]) is int
                and pair["cpu_millis"] in {250, 500, 1000, 2000, 4000, 8000, 16000}
                and 512 <= pair["memory_mb"] <= 122880
                for pair in pairs
            )
        ):
            options = "; ".join(
                f"cpu_millis = {pair['cpu_millis']}, memory_mb = {pair['memory_mb']}"
                for pair in pairs
            )
            return "Agent resources exceed supported sizes. Valid manifest pairs: " + options + "."
    allowlisted = _SAFE_API_ERROR_DETAILS.get((status_code, code))
    if allowlisted is None and (
        (status_code, code) in _REPORTED_API_REFUSALS
        or (status_code == 409 and _CLOUD_ERROR_CODE.fullmatch(code))
    ):
        return _reported_message(detail.get("message"))
    return allowlisted


def _reported_message(value: object) -> str | None:
    """A refusal's message, or its first line when the whole message can't be shown.

    An Agent's refusal that Cloud passes on can run to many lines of guidance (an
    execution-profile mismatch lists every difference and next step), which the filter
    refuses as a whole. Its first line says what happened, so it is shown alone, cut back
    to whole sentences that fit, rather than leaving the caller with a bare "HTTP 409".
    """

    whole = safe_text(value, _CONFLICT_DETAIL_LIMIT)
    if whole is not None or not isinstance(value, str):
        return whole
    first = value.split("\n", 1)[0].strip()
    try:
        while len(first.encode()) > _CONFLICT_DETAIL_LIMIT and ". " in first:
            first = first.rsplit(". ", 1)[0] + "."
    except UnicodeError:
        return None
    return safe_text(first, _CONFLICT_DETAIL_LIMIT)


def _safe_conflict_detail(response: httpx.Response) -> str | None:
    """Cloud's reason for a conflict, filtered the way deployment failure text is.

    Cloud explains a 409 (a stale application revision, a release that has not passed
    its gate, a pending lifecycle action, an idempotency or version conflict) in plain
    text. Without it the CLI can only say "HTTP 409", which leaves the caller unable to
    tell a race from a refusal.
    """

    if response.status_code != 409:
        return None
    return _plain_api_error_detail(response)


# A validation rejection names at most this many fields worth reporting.
_REJECTED_FIELD_LIMIT = 32
_REJECTED_FIELD_PART = re.compile(r"[A-Za-z0-9_]{1,64}")
_REJECTED_FIELD_TYPE = re.compile(r"[a-z][a-z_]{0,63}")


def _rejected_fields(response: httpx.Response) -> tuple[tuple[str, str], ...]:
    """The request fields an HTTP 422 validation rejection names, with each error's type.

    Only the location and type of each error are kept, never its message or input, and only
    when every part of the location is a plain field name or index.
    """

    if response.status_code != 422:
        return ()
    try:
        payload = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    errors = payload.get("detail") if isinstance(payload, dict) else None
    if not isinstance(errors, list):
        return ()
    fields: list[tuple[str, str]] = []
    for error in errors[:_REJECTED_FIELD_LIMIT]:
        if not isinstance(error, dict):
            continue
        location, kind = error.get("loc"), error.get("type")
        if (
            not isinstance(location, list)
            or not location
            or not isinstance(kind, str)
            or _REJECTED_FIELD_TYPE.fullmatch(kind) is None
        ):
            continue
        parts = [str(part) for part in location if isinstance(part, (str, int))]
        if len(parts) == len(location) and all(
            _REJECTED_FIELD_PART.fullmatch(part) for part in parts
        ):
            fields.append((".".join(parts), kind))
    return tuple(fields)


def _plain_api_error_detail(response: httpx.Response) -> str | None:
    """Return a bounded plain-text API rejection reason with no private-looking content."""

    try:
        payload = response.json()
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return safe_text(payload.get("detail"), 512)


def _is_loopback(hostname: str | None) -> bool:
    if hostname is None:
        return False
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False
