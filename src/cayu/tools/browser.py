"""Runner-backed Playwright adapter for the stable ``web_fetch`` tool."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError, model_validator

from cayu._validation import require_durable_text
from cayu.core.tools import ToolContext, ToolResult
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionRequirements,
    evaluate_execution_admission,
)
from cayu.runners import ExecCommand, RunnerExecutionError, RunnerUnavailableError
from cayu.tools.web import (
    MAX_WEB_FETCH_TITLE_BYTES,
    MAX_WEB_FETCH_URL_LENGTH,
    WebFetchAdapterRequest,
    _canonicalize_url,
    _error_result,
    _web_fetch_success_result,
)

BROWSER_FETCH_PROTOCOL_VERSION = "cayu.browser-fetch.v2"
BROWSER_FETCH_WORKER_VERSION = "2"
BROWSER_FETCH_PLAYWRIGHT_VERSION = "1.62.0"
DEFAULT_BROWSER_FETCH_WORKER_COMMAND = (
    "/usr/local/bin/python",
    "-I",
    "/opt/cayu-browser/worker.py",
)
DEFAULT_BROWSER_FETCH_MAX_REQUESTS = 128
MAX_BROWSER_FETCH_MAX_REQUESTS = 512
DEFAULT_BROWSER_FETCH_MAX_DOM_NODES = 10_000
MAX_BROWSER_FETCH_MAX_DOM_NODES = 100_000
_BROWSER_FETCH_JSON_OVERHEAD_BYTES = 64 * 1024
_BROWSER_FETCH_ERROR_CODES = frozenset(
    {
        "browser_crash",
        "browser_unavailable",
        "capability_refused",
        "cleanup_failed",
        "destination_denied",
        "dns_failure",
        "fetch_failed",
        "http_status",
        "incompatible_browser",
        "oversized_response",
        "redirect_denied",
        "timeout",
        "unsupported_content",
    }
)
_BROWSER_FETCH_ERROR_MESSAGES = {
    "browser_crash": "The sandboxed browser stopped unexpectedly.",
    "browser_unavailable": "The selected runner does not provide the browser worker.",
    "capability_refused": "The selected runner did not prove the required browser isolation.",
    "cleanup_failed": "The sandboxed browser could not be cleaned up safely.",
    "destination_denied": "The destination was denied by the browser egress policy.",
    "dns_failure": "The destination could not be resolved.",
    "fetch_failed": "The sandboxed browser request failed.",
    "incompatible_browser": "The browser worker or Playwright version is incompatible.",
    "oversized_response": "The browser fetch exceeded its resource limits.",
    "redirect_denied": "The redirect target was denied.",
    "timeout": "The browser fetch timed out.",
    "unsupported_content": "The response content type is unsupported.",
}


@runtime_checkable
class _AdmissionAwareRunnerHandle(Protocol):
    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None: ...


class _BrowserRedirect(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status_code: StrictInt = Field(ge=300, le=399)
    from_url: str = Field(min_length=1, max_length=8192)
    to_url: str = Field(min_length=1, max_length=8192)


class _BrowserSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol_version: Literal["cayu.browser-fetch.v2"]
    worker_version: Literal["2"]
    playwright_version: Literal["1.62.0"]
    kind: Literal["success"]
    requested_url: str = Field(min_length=1, max_length=8192)
    final_url: str = Field(min_length=1, max_length=8192)
    title: str | None = None
    representation: Literal["text", "accessibility"]
    content: str
    redirects: tuple[_BrowserRedirect, ...] = Field(max_length=10)
    truncation_reasons: tuple[Literal["title", "content"], ...] = Field(max_length=2)
    response_bytes: StrictInt = Field(ge=0)
    request_count: StrictInt = Field(ge=1, le=MAX_BROWSER_FETCH_MAX_REQUESTS)

    @model_validator(mode="after")
    def validate_truncation_reasons(self) -> _BrowserSuccess:
        if len(set(self.truncation_reasons)) != len(self.truncation_reasons):
            raise ValueError("truncation_reasons must not contain duplicates.")
        return self


class _BrowserFailure(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol_version: Literal["cayu.browser-fetch.v2"]
    worker_version: Literal["2"]
    playwright_version: str = Field(min_length=1, max_length=32)
    kind: Literal["error"]
    error: str = Field(min_length=1, max_length=64)
    status_code: StrictInt | None = Field(default=None, ge=100, le=599)

    @model_validator(mode="after")
    def validate_error(self) -> _BrowserFailure:
        if self.error not in _BROWSER_FETCH_ERROR_CODES:
            raise ValueError("Unknown browser worker error code.")
        if (self.error == "http_status") != (self.status_code is not None):
            raise ValueError("Only http_status errors carry status_code.")
        return self


class BrowserWebFetchAdapter:
    """Execute ``web_fetch`` through a compatible worker in an admitted runner.

    Selection is explicit: construct ``WebFetchTool(adapter=...)``. This adapter
    never imports Playwright in the host process and never falls back to the
    trusted-process HTTP implementation.
    """

    def __init__(
        self,
        *,
        worker_command: Sequence[str] = DEFAULT_BROWSER_FETCH_WORKER_COMMAND,
        max_requests: int = DEFAULT_BROWSER_FETCH_MAX_REQUESTS,
        max_dom_nodes: int = DEFAULT_BROWSER_FETCH_MAX_DOM_NODES,
    ) -> None:
        self._worker_command = _browser_worker_command(worker_command)
        self.max_requests = _bounded_max_requests(max_requests)
        self.max_dom_nodes = _bounded_max_dom_nodes(max_dom_nodes)

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
        runner = ctx.runner
        if runner is None:
            return _error_result(
                "incompatible_runner",
                "The sandboxed browser requires a session runner.",
            )
        if not isinstance(runner, _AdmissionAwareRunnerHandle):
            return _error_result(
                "incompatible_runner",
                "The selected runner does not expose verified execution capabilities.",
            )
        try:
            candidate = runner.execution_admission_candidate()
        except Exception:
            return _error_result(
                "capability_refused",
                _BROWSER_FETCH_ERROR_MESSAGES["capability_refused"],
            )
        if not _browser_runner_is_admitted(candidate):
            return _error_result(
                "capability_refused",
                _BROWSER_FETCH_ERROR_MESSAGES["capability_refused"],
            )

        payload = json.dumps(
            {
                "protocol_version": BROWSER_FETCH_PROTOCOL_VERSION,
                "worker_version": BROWSER_FETCH_WORKER_VERSION,
                "expected_playwright_version": BROWSER_FETCH_PLAYWRIGHT_VERSION,
                "operation": "fetch",
                "url": request.requested_url,
                "limits": {
                    "max_response_bytes": request.max_response_bytes,
                    "max_content_bytes": request.max_content_bytes,
                    "timeout_seconds": request.timeout_seconds,
                    "max_redirects": request.max_redirects,
                    "max_requests": self.max_requests,
                    "max_dom_nodes": self.max_dom_nodes,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        output_limit = _browser_output_limit(request)
        runner_timeout = max(1, math.ceil(request.timeout_seconds))
        try:
            execution = await runner.exec(
                self._worker_command,
                timeout_s=runner_timeout,
                stdin=payload,
                output_limit_bytes=output_limit,
            )
        except RunnerUnavailableError:
            return _error_result(
                "browser_unavailable",
                _BROWSER_FETCH_ERROR_MESSAGES["browser_unavailable"],
            )
        except RunnerExecutionError:
            return _error_result(
                "browser_crash",
                _BROWSER_FETCH_ERROR_MESSAGES["browser_crash"],
            )
        except TimeoutError:
            return _error_result("timeout", _BROWSER_FETCH_ERROR_MESSAGES["timeout"])
        except Exception:
            return _error_result(
                "browser_crash",
                _BROWSER_FETCH_ERROR_MESSAGES["browser_crash"],
            )

        if execution.timed_out:
            return _error_result("timeout", _BROWSER_FETCH_ERROR_MESSAGES["timeout"])
        if execution.cancelled:
            return _error_result(
                "browser_crash",
                _BROWSER_FETCH_ERROR_MESSAGES["browser_crash"],
            )
        if execution.stdout_truncated:
            return _error_result(
                "oversized_response",
                _BROWSER_FETCH_ERROR_MESSAGES["oversized_response"],
            )
        try:
            stdout_size = len(execution.stdout.encode("utf-8"))
        except UnicodeEncodeError:
            return _error_result(
                "malformed_browser_result",
                "The browser worker returned an invalid result.",
            )
        if stdout_size > output_limit:
            return _error_result(
                "oversized_response",
                _BROWSER_FETCH_ERROR_MESSAGES["oversized_response"],
            )
        if execution.exit_code != 0:
            code = (
                "browser_unavailable" if execution.exit_code in {2, 126, 127} else "browser_crash"
            )
            return _error_result(code, _BROWSER_FETCH_ERROR_MESSAGES[code])
        return _browser_worker_result(
            execution.stdout,
            request=request,
            max_requests=self.max_requests,
        )


def _browser_worker_command(value: Sequence[str]) -> ExecCommand:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("worker_command must be a sequence of process arguments.")
    if not value or len(value) > 16:
        raise ValueError("worker_command must contain between 1 and 16 arguments.")
    copied: list[str] = []
    for item in value:
        if type(item) is not str:
            raise TypeError("worker_command arguments must be strings.")
        if not item or len(item.encode("utf-8")) > 4096:
            raise ValueError("worker_command arguments must be bounded non-empty strings.")
        copied.append(item)
    return ExecCommand.process(*copied)


def _bounded_max_requests(value: int) -> int:
    if type(value) is not int:
        raise TypeError("max_requests must be an integer.")
    if value <= 0 or value > MAX_BROWSER_FETCH_MAX_REQUESTS:
        raise ValueError(f"max_requests must be between 1 and {MAX_BROWSER_FETCH_MAX_REQUESTS}.")
    return value


def _bounded_max_dom_nodes(value: int) -> int:
    if type(value) is not int:
        raise TypeError("max_dom_nodes must be an integer.")
    if value <= 0 or value > MAX_BROWSER_FETCH_MAX_DOM_NODES:
        raise ValueError(f"max_dom_nodes must be between 1 and {MAX_BROWSER_FETCH_MAX_DOM_NODES}.")
    return value


def _browser_output_limit(request: WebFetchAdapterRequest) -> int:
    # JSON can double quotes, backslashes, and normalized line breaks. Account
    # for four-byte UTF-8 characters in every bounded URL in the redirect
    # envelope plus fixed framing rather than truncating a valid worker response
    # at the runner capture boundary.
    string_bytes = (
        request.max_content_bytes
        + MAX_WEB_FETCH_TITLE_BYTES
        + 4 * (2 + 2 * request.max_redirects) * MAX_WEB_FETCH_URL_LENGTH
    )
    return 2 * string_bytes + _BROWSER_FETCH_JSON_OVERHEAD_BYTES


def _browser_runner_is_admitted(candidate: ExecutionAdmissionCandidate | None) -> bool:
    if type(candidate) is not ExecutionAdmissionCandidate:
        return False
    requirements = ExecutionRequirements.trusted(
        network_access="brokered_egress",
        cancellation="confirmed",
        cleanup="confirmed",
        minimum_evidence="available",
    )
    decision = evaluate_execution_admission(
        candidate=candidate.candidate,
        requirements=requirements,
        evidence=candidate.evidence,
    )
    return decision.status == "admitted"


def _browser_worker_result(
    stdout: str,
    *,
    request: WebFetchAdapterRequest,
    max_requests: int,
) -> ToolResult:
    try:
        raw = json.loads(stdout)
    except (TypeError, ValueError, RecursionError):
        return _error_result(
            "malformed_browser_result",
            "The browser worker returned an invalid result.",
        )
    if type(raw) is not dict:
        return _error_result(
            "malformed_browser_result",
            "The browser worker returned an invalid result.",
        )
    if (
        raw.get("protocol_version") != BROWSER_FETCH_PROTOCOL_VERSION
        or raw.get("worker_version") != BROWSER_FETCH_WORKER_VERSION
        or raw.get("playwright_version") != BROWSER_FETCH_PLAYWRIGHT_VERSION
    ):
        return _error_result(
            "incompatible_browser",
            _BROWSER_FETCH_ERROR_MESSAGES["incompatible_browser"],
        )
    try:
        if raw.get("kind") == "error":
            failure = _BrowserFailure.model_validate(raw)
            if failure.error == "http_status":
                return ToolResult(
                    content=f"The HTTPS request returned status {failure.status_code}.",
                    structured={"error": "http_status", "status_code": failure.status_code},
                    is_error=True,
                )
            return _error_result(
                failure.error,
                _BROWSER_FETCH_ERROR_MESSAGES[failure.error],
            )
        success = _BrowserSuccess.model_validate(raw)
    except (KeyError, ValidationError, RecursionError):
        return _error_result(
            "malformed_browser_result",
            "The browser worker returned an invalid result.",
        )

    try:
        requested_url = _canonicalize_url(success.requested_url)
        final_url = _canonicalize_url(success.final_url)
        content = require_durable_text(success.content, "browser content")
        content_bytes = len(content.encode("utf-8"))
        title = (
            None if success.title is None else require_durable_text(success.title, "browser title")
        )
        title_bytes = 0 if title is None else len(title.encode("utf-8"))
        redirects = [
            {
                "status_code": redirect.status_code,
                "from_url": _canonicalize_url(redirect.from_url),
                "to_url": _canonicalize_url(redirect.to_url),
            }
            for redirect in success.redirects
        ]
    except (TypeError, UnicodeError, ValueError):
        return _error_result(
            "malformed_browser_result",
            "The browser worker returned an invalid result.",
        )
    if (
        requested_url != request.requested_url
        or success.response_bytes > request.max_response_bytes
        or content_bytes > request.max_content_bytes
        or title_bytes > MAX_WEB_FETCH_TITLE_BYTES
        or len(redirects) > request.max_redirects
        or success.request_count > max_requests
    ):
        return _error_result(
            "malformed_browser_result",
            "The browser worker returned an invalid result.",
        )
    return _web_fetch_success_result(
        requested_url=requested_url,
        final_url=final_url,
        title=title,
        representation=success.representation,
        content=content,
        redirects=redirects,
        truncation_reasons=success.truncation_reasons,
    )


__all__ = [
    "BROWSER_FETCH_PLAYWRIGHT_VERSION",
    "BROWSER_FETCH_PROTOCOL_VERSION",
    "BROWSER_FETCH_WORKER_VERSION",
    "DEFAULT_BROWSER_FETCH_MAX_DOM_NODES",
    "DEFAULT_BROWSER_FETCH_MAX_REQUESTS",
    "DEFAULT_BROWSER_FETCH_WORKER_COMMAND",
    "MAX_BROWSER_FETCH_MAX_DOM_NODES",
    "MAX_BROWSER_FETCH_MAX_REQUESTS",
    "BrowserWebFetchAdapter",
]
