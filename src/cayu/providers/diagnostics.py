"""Explicitly private, bounded provider-error capture outside runtime events.

This is not a wire logger. No request body, URL, arbitrary response header, or
successful output is captured. Error text can still contain customer data;
applications own sink access, retention, and the workload-secret registry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from inspect import iscoroutine, iscoroutinefunction
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID, uuid4

from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    import httpx

_MAX_SOURCE_CHARS = 64 * 1024
_MAX_RECORDS = 32
_FIELD_BYTES = {"message": 4096, "param": 256, "request_id": 256, "type": 128, "code": 128}
_STATUS_PATHS = (
    "status_code",
    "error.status_code",
    "response.status_code",
    "response.error.status_code",
)
_Boundary = Literal["http_error", "stream_error", "transport_error"]
_ACTIVE_CAPTURE: ContextVar[ProviderErrorCapture | None] = ContextVar(
    "cayu_provider_error_capture", default=None
)


@dataclass(slots=True)
class ProviderErrorCapture:
    """One explicit capture scope; inspect counters after leaving the scope.

    Use :func:`capture_provider_errors` to create this object. Sink failures are
    counted, never substituted for the original provider failure or retried.
    A sink must complete promptly and must not log the supplied record publicly.
    """

    capture_id: str
    _sink: Callable[[dict[str, Any]], None] = field(repr=False)
    _redactor: SecretRedactor = field(repr=False)
    records_attempted: int = 0
    records_written: int = 0
    sink_failures: int = 0
    records_dropped: int = 0
    _closed: bool = False

    def record_error(
        self,
        error: Mapping[str, Any],
        *,
        boundary: _Boundary,
        headers: Mapping[str, str] | None = None,
        http_status_code: int | None = None,
        provider_request_id: str | None = None,
        request_id_source: Literal["header", "body", "unavailable"] = "unavailable",
        body_state: Literal["available", "unavailable"] = "available",
        status_fields: Mapping[str, Any] | None = None,
    ) -> None:
        """Capture an OpenAI-shaped error from a custom transport, if needed.

        Bundled HTTP transports call this automatically. ``headers`` are request
        headers used ONLY to register credentials for redaction, never persisted.
        ``http_status_code`` is the transport status (possibly 200 for SSE), not
        an inferred error status. ``status_fields`` can supply explicit body
        statuses at the fixed status_code/error/response paths; arbitrary keys
        and non-status values are ignored. By default, error's status_code is
        used. No field here authorizes retries or recovery.
        """
        if self._closed:
            return
        if self.records_attempted >= _MAX_RECORDS:
            self.records_dropped += 1
            return
        if boundary not in {"http_error", "stream_error", "transport_error"}:
            raise ValueError("Unsupported provider diagnostic boundary.")
        if request_id_source not in {"header", "body", "unavailable"}:
            raise ValueError("Unsupported provider request ID source.")
        if body_state not in {"available", "unavailable"}:
            raise ValueError("Unsupported provider diagnostic body state.")
        redactor = self._redactor.merged_with(_header_redactor(headers or {}))
        fields: dict[str, str] = {}
        states: dict[str, str] = {}
        for name, max_bytes in _FIELD_BYTES.items():
            value = provider_request_id if name == "request_id" else error.get(name)
            if name == "request_id" and value is None:
                value = error.get(name)
                if type(value) is str and value:
                    request_id_source = "body"
            if value is None:
                states[name] = "absent"
            elif type(value) is not str:
                states[name] = "invalid_type"
            elif len(value) > _MAX_SOURCE_CHARS:
                # Never truncate before redaction: that could expose a secret
                # fragment straddling the truncation boundary. Omit huge fields.
                states[name] = "omitted_oversize"
            else:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError:
                    # Lossy repair after redaction could reconstruct a known
                    # secret (e.g. a surrogate replaced by a password's '?').
                    states[name] = "invalid_unicode"
                    continue
                redacted = redactor.redact_text(value)
                bounded, truncated = redactor.redact_text_bounded_with_marker(
                    value, max_bytes=max_bytes, truncation_marker="...[truncated]"
                )
                fields[name] = bounded
                states[name] = (
                    "redacted_truncated"
                    if truncated and redacted != value
                    else "truncated"
                    if truncated
                    else "redacted"
                    if redacted != value
                    else "present"
                )
        self.records_attempted += 1
        record: dict[str, Any] = {
            "schema": "cayu.provider-error.v1",
            "capture_id": self.capture_id,
            "sequence": self.records_attempted,
            "boundary": boundary,
            "error": fields,
            "field_states": states,
            "body_state": body_state,
            "request_id_source": request_id_source,
        }
        if type(http_status_code) is int and 100 <= http_status_code <= 599:
            record["http_status_code"] = http_status_code
        source_statuses = (
            {"status_code": error.get("status_code")} if status_fields is None else status_fields
        )
        explicit_statuses = {
            path: status
            for path in _STATUS_PATHS
            if type(status := source_statuses.get(path)) is int and 100 <= status <= 599
        }
        distinct_statuses = set(explicit_statuses.values())
        record["error_status_codes"] = explicit_statuses
        record["error_status_conflict"] = len(distinct_statuses) > 1
        # Never choose one side of a conflict or fold HTTP 200 into SSE error
        # statuses. The fixed paths retain the evidence behind the disagreement.
        if len(distinct_statuses) == 1:
            record["error_status_code"] = next(iter(distinct_statuses))
        try:
            result = self._sink(record)
            if iscoroutine(result):
                result.close()
                raise TypeError("Provider diagnostic sinks must be synchronous.")
            if result is not None:
                raise TypeError("Provider diagnostic sinks must return None.")
        except Exception:
            # No arbitrary sink exception message or traceback is logged. The
            # caller can fail its diagnostic qualification using these counters.
            self.sink_failures += 1
        else:
            self.records_written += 1


@contextmanager
def capture_provider_errors(
    sink: Callable[[dict[str, Any]], None],
    *,
    redactor: SecretRedactor,
    capture_id: str | None = None,
) -> Iterator[ProviderErrorCapture]:
    """Opt into private HTTP/SSE error details for operations in this scope.

    No capture occurs by default. An explicit workload ``SecretRedactor`` is
    required; use an empty registry only when the workload contains no secrets.
    Request header credentials are additionally redacted by bundled transports.
    Unknown customer data is NOT anonymized. Store records separately from
    events/logs/support bundles, with restricted access and short retention.

    Scopes are task-local and nestable. Child tasks inherit the active scope but
    must finish before it exits. The optional local capture ID must be a UUID;
    it is distinct from the upstream provider request ID and is not sent to it.
    This scope makes no requests, adds no headers and changes no retry policy.
    """
    if not callable(sink):
        raise TypeError("sink must be callable.")
    if iscoroutinefunction(sink) or iscoroutinefunction(sink.__call__):
        raise TypeError("sink must be synchronous.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    identifier = uuid4().hex if capture_id is None else UUID(capture_id).hex
    capture = ProviderErrorCapture(identifier, sink, redactor)
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield capture
    finally:
        capture._closed = True
        _ACTIVE_CAPTURE.reset(token)


def _header_redactor(headers: Mapping[str, str]) -> SecretRedactor:
    values = []
    for name, value in headers.items():
        if type(value) is not str or not value.strip():
            continue
        values.append(value)
        if name.lower() in {"authorization", "proxy-authorization"}:
            _, separator, credential = value.partition(" ")
            if separator and credential.strip():
                values.append(credential.strip())
    return SecretRedactor(values)


def _record_http_error(
    response: httpx.Response,
    *,
    headers: Mapping[str, str],
    body_response: httpx.Response | None = None,
) -> None:
    capture = _ACTIVE_CAPTURE.get()
    if capture is None or capture._closed:
        return
    body = None
    if (
        body_response is not None
        and "application/json" in body_response.headers.get("content-type", "")
        and len(body_response.content) <= 64 * 1024
    ):
        with suppress(ValueError):
            body = body_response.json()
    error = body.get("error", body) if isinstance(body, Mapping) else {}
    error = error if isinstance(error, Mapping) else {}
    request_id = response.headers.get("x-request-id")
    source = "header" if request_id is not None else "unavailable"
    if request_id is None and isinstance(body, Mapping):
        request_id = body.get("request_id")
        source = "body" if request_id is not None else "unavailable"
    capture.record_error(
        error,
        boundary="http_error",
        headers=headers,
        http_status_code=response.status_code,
        provider_request_id=request_id,
        request_id_source=source,
        body_state="available" if isinstance(body, Mapping) else "unavailable",
        status_fields=_error_status_fields(body) if isinstance(body, Mapping) else {},
    )


def _record_stream_error(
    event: Mapping[str, Any], *, headers: Mapping[str, str], response: httpx.Response | None = None
) -> None:
    capture = _ACTIVE_CAPTURE.get()
    event_type = event.get("type")
    if (
        capture is None
        or capture._closed
        or type(event_type) is not str
        or event_type not in {"error", "response.failed"}
    ):
        return
    container = event.get("response") if event.get("type") == "response.failed" else event
    container = container if isinstance(container, Mapping) else {}
    error = container.get("error", container)
    error = error if isinstance(error, Mapping) else {}
    status_fields = _error_status_fields(container)
    if event_type == "response.failed":
        status_fields = {
            "status_code": event.get("status_code"),
            **{f"response.{path}": value for path, value in status_fields.items()},
        }
    request_id = response.headers.get("x-request-id") if response is not None else None
    source = "header" if request_id is not None else "unavailable"
    if request_id is None:
        request_id = container.get("request_id", event.get("request_id"))
        source = "body" if request_id is not None else "unavailable"
    capture.record_error(
        error,
        boundary="stream_error",
        headers=headers,
        http_status_code=response.status_code if response is not None else None,
        provider_request_id=request_id,
        request_id_source=source,
        status_fields=status_fields,
    )


def _error_status_fields(envelope: Mapping[str, Any]) -> dict[str, Any]:
    fields = {"status_code": envelope.get("status_code")}
    error = envelope.get("error")
    if isinstance(error, Mapping):
        fields["error.status_code"] = error.get("status_code")
    return fields


def _record_transport_error(
    kind: str, *, headers: Mapping[str, str], response: httpx.Response | None = None
) -> None:
    capture = _ACTIVE_CAPTURE.get()
    if capture is not None:
        # Transport exception text often contains a URL or wire material. Keep
        # only the caller's fixed exception category, not str/repr/tracebacks.
        request_id = response.headers.get("x-request-id") if response is not None else None
        capture.record_error(
            {"type": kind},
            boundary="transport_error",
            headers=headers,
            http_status_code=response.status_code if response is not None else None,
            provider_request_id=request_id,
            request_id_source="header" if request_id is not None else "unavailable",
            body_state="unavailable",
        )
