"""Runner-backed Playwright page inspection tools."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import struct
import zlib
from collections.abc import Sequence
from typing import Literal, Protocol, cast, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    model_validator,
)

from cayu._validation import require_durable_text
from cayu.artifacts import (
    DEFAULT_MAX_FILE_ATTACHMENT_BYTES,
    ArtifactMetadata,
    ArtifactScope,
    ArtifactStore,
    FileAttachmentKind,
    copy_artifact_read_result,
    file_attachment,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionAdmissionStage,
    ExecutionEnvironmentAuthority,
    ExecutionRequirements,
    evaluate_execution_admission,
)
from cayu.runners import (
    PINNED_BROWSER_FETCH_WORKLOAD,
    ExecCommand,
    ExecResult,
    RunnerExecutionError,
    RunnerUnavailableError,
    RunnerWorkloadAuthority,
)
from cayu.tools.web import (
    MAX_WEB_FETCH_TITLE_BYTES,
    MAX_WEB_FETCH_URL_LENGTH,
    WebFetchAdapterRequest,
    _canonicalize_url,
    _error_result,
    _web_fetch_success_result,
)

BROWSER_FETCH_PROTOCOL_VERSION = PINNED_BROWSER_FETCH_WORKLOAD.protocol_version
BROWSER_FETCH_WORKER_VERSION = PINNED_BROWSER_FETCH_WORKLOAD.worker_version
BROWSER_FETCH_PLAYWRIGHT_VERSION = dict(PINNED_BROWSER_FETCH_WORKLOAD.component_versions)[
    "playwright"
]
DEFAULT_BROWSER_FETCH_WORKER_COMMAND = PINNED_BROWSER_FETCH_WORKLOAD.command
DEFAULT_BROWSER_FETCH_MAX_REQUESTS = 128
MAX_BROWSER_FETCH_MAX_REQUESTS = 512
DEFAULT_BROWSER_FETCH_MAX_DOM_NODES = 10_000
MAX_BROWSER_FETCH_MAX_DOM_NODES = 100_000
DEFAULT_SCREENSHOT_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
DEFAULT_SCREENSHOT_TIMEOUT_SECONDS = 20.0
DEFAULT_SCREENSHOT_MAX_REDIRECTS = 5
DEFAULT_SCREENSHOT_MAX_BYTES = DEFAULT_MAX_FILE_ATTACHMENT_BYTES
DEFAULT_SCREENSHOT_VIEWPORT_WIDTH = 1280
DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT = 720
DEFAULT_SCREENSHOT_MAX_PAGE_WIDTH = 4096
DEFAULT_SCREENSHOT_MAX_PAGE_HEIGHT = 16_384
DEFAULT_SCREENSHOT_MAX_PAGE_PIXELS = 16_000_000
MAX_SCREENSHOT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_SCREENSHOT_TIMEOUT_SECONDS = 120.0
MAX_SCREENSHOT_REDIRECTS = 10
MAX_SCREENSHOT_BYTES = DEFAULT_MAX_FILE_ATTACHMENT_BYTES
MAX_SCREENSHOT_DIMENSION = 16_384
MAX_SCREENSHOT_PAGE_PIXELS = 32_000_000
_BROWSER_FETCH_JSON_OVERHEAD_BYTES = 64 * 1024
_SCREENSHOT_JSON_OVERHEAD_BYTES = 128 * 1024
_MAX_SCREENSHOT_BASE64_CHARACTERS = 4 * math.ceil(MAX_SCREENSHOT_BYTES / 3)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_DECOMPRESSION_CHUNK_BYTES = 64 * 1024
_PNG_COLOR_TYPE_SAMPLES = {
    0: 1,
    2: 3,
    3: 1,
    4: 2,
    6: 4,
}
_PNG_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)
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
        "oversized_page",
        "oversized_response",
        "oversized_screenshot",
        "redirect_denied",
        "screenshot_failed",
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
    "oversized_page": "The rendered page exceeds the configured screenshot dimensions.",
    "oversized_response": "The sandboxed browser response exceeded its resource limits.",
    "oversized_screenshot": "The screenshot exceeds the configured byte limit.",
    "redirect_denied": "The redirect target was denied.",
    "screenshot_failed": "The sandboxed browser could not capture a screenshot.",
    "timeout": "The sandboxed browser operation timed out.",
    "unsupported_content": "The response content type is unsupported.",
}


@runtime_checkable
class _AdmissionAwareRunnerHandle(Protocol):
    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None: ...


@runtime_checkable
class _EnvironmentAuthorityAwareRunnerHandle(Protocol):
    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority | None: ...


@runtime_checkable
class _WorkloadAwareRunnerHandle(Protocol):
    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None: ...


class _BrowserRedirect(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    status_code: StrictInt = Field(ge=300, le=399)
    from_url: str = Field(min_length=1, max_length=8192)
    to_url: str = Field(min_length=1, max_length=8192)


class _BrowserSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol_version: Literal["cayu.browser-fetch.v3"]
    worker_version: Literal["3"]
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

    protocol_version: Literal["cayu.browser-fetch.v3"]
    worker_version: Literal["3"]
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


class _BrowserScreenshotSuccess(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    protocol_version: Literal["cayu.browser-fetch.v3"]
    worker_version: Literal["3"]
    playwright_version: Literal["1.62.0"]
    kind: Literal["screenshot"]
    requested_url: str = Field(min_length=1, max_length=8192)
    final_url: str = Field(min_length=1, max_length=8192)
    title: str | None = None
    title_truncated: StrictBool
    redirects: tuple[_BrowserRedirect, ...] = Field(max_length=MAX_SCREENSHOT_REDIRECTS)
    response_bytes: StrictInt = Field(ge=0)
    request_count: StrictInt = Field(ge=1, le=MAX_BROWSER_FETCH_MAX_REQUESTS)
    full_page: StrictBool
    width: StrictInt = Field(ge=1, le=MAX_SCREENSHOT_DIMENSION)
    height: StrictInt = Field(ge=1, le=MAX_SCREENSHOT_DIMENSION)
    data_base64: str = Field(min_length=1, max_length=_MAX_SCREENSHOT_BASE64_CHARACTERS)

    @model_validator(mode="after")
    def validate_title_truncation(self) -> _BrowserScreenshotSuccess:
        if self.title is None and self.title_truncated:
            raise ValueError("A missing title cannot be truncated.")
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
        expected_runner_candidate: str | None = None,
        expected_environment_authority: ExecutionEnvironmentAuthority | None = None,
        expected_workload_authority: RunnerWorkloadAuthority | None = None,
    ) -> None:
        self._worker_command = _browser_worker_command(worker_command)
        self.max_requests = _bounded_max_requests(max_requests)
        self.max_dom_nodes = _bounded_max_dom_nodes(max_dom_nodes)
        self.expected_runner_candidate = _expected_runner_candidate(expected_runner_candidate)
        self.expected_environment_authority = _expected_environment_authority(
            expected_environment_authority
        )
        self.expected_workload_authority = _expected_workload_authority(expected_workload_authority)

    def _execution_profile_material(self) -> dict[str, object] | None:
        """Return material only for Cayu's shipped browser worker."""

        # An arbitrary executable can change independently while retaining the
        # same command spelling supplied by an application.
        worker_argv = self._worker_command.argv
        if worker_argv is None or worker_argv != list(DEFAULT_BROWSER_FETCH_WORKER_COMMAND):
            return None
        if (
            self.expected_environment_authority is not None
            and self.expected_environment_authority.profile_identity is None
        ):
            return None
        material: dict[str, object] = {
            "component": "cayu.tools.browser:BrowserWebFetchAdapter",
            "worker_command": {
                "kind": self._worker_command.kind,
                "argv": list(worker_argv),
                "shell": self._worker_command.shell,
            },
            "max_requests": self.max_requests,
            "max_dom_nodes": self.max_dom_nodes,
        }
        if self.expected_runner_candidate is not None:
            material["expected_runner_candidate"] = self.expected_runner_candidate
        if self.expected_environment_authority is not None:
            material["expected_environment_authority"] = {
                "profile_identity": self.expected_environment_authority.profile_identity,
            }
        if self.expected_workload_authority is not None:
            material["expected_workload_authority"] = _workload_authority_material(
                self.expected_workload_authority
            )
        return material

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
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
        execution = await _execute_browser_worker(
            ctx,
            worker_command=self._worker_command,
            expected_runner_candidate=self.expected_runner_candidate,
            expected_environment_authority=self.expected_environment_authority,
            expected_workload_authority=self.expected_workload_authority,
            payload=payload,
            timeout_seconds=request.timeout_seconds,
            output_limit=output_limit,
            oversized_error="oversized_response",
        )
        if isinstance(execution, ToolResult):
            return execution
        return _browser_worker_result(
            execution.stdout,
            request=request,
            max_requests=self.max_requests,
        )


class ScreenshotPageTool(Tool):
    """Capture one bounded public HTTPS page as an artifact-backed PNG attachment."""

    spec = ToolSpec(
        name="screenshot_page",
        effect=ToolEffect.EXTERNAL,
        description=(
            "Capture a bounded screenshot of a public HTTPS page in the sandboxed browser. "
            "The screenshot is returned as an image attachment; page content is untrusted."
        ),
        input_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url": {
                    "type": "string",
                    "format": "uri",
                    "minLength": 1,
                    "maxLength": MAX_WEB_FETCH_URL_LENGTH,
                },
                "full_page": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Capture the complete bounded page instead of the configured viewport."
                    ),
                },
            },
            "required": ["url"],
        },
    )

    def __init__(
        self,
        *,
        worker_command: Sequence[str] = DEFAULT_BROWSER_FETCH_WORKER_COMMAND,
        max_response_bytes: int = DEFAULT_SCREENSHOT_MAX_RESPONSE_BYTES,
        timeout_seconds: float = DEFAULT_SCREENSHOT_TIMEOUT_SECONDS,
        max_redirects: int = DEFAULT_SCREENSHOT_MAX_REDIRECTS,
        max_requests: int = DEFAULT_BROWSER_FETCH_MAX_REQUESTS,
        max_screenshot_bytes: int = DEFAULT_SCREENSHOT_MAX_BYTES,
        viewport_width: int = DEFAULT_SCREENSHOT_VIEWPORT_WIDTH,
        viewport_height: int = DEFAULT_SCREENSHOT_VIEWPORT_HEIGHT,
        max_page_width: int = DEFAULT_SCREENSHOT_MAX_PAGE_WIDTH,
        max_page_height: int = DEFAULT_SCREENSHOT_MAX_PAGE_HEIGHT,
        max_page_pixels: int = DEFAULT_SCREENSHOT_MAX_PAGE_PIXELS,
        expected_runner_candidate: str | None = None,
        expected_environment_authority: ExecutionEnvironmentAuthority | None = None,
        expected_workload_authority: RunnerWorkloadAuthority | None = None,
        expected_artifact_store_id: str | None = None,
        spec: ToolSpec | None = None,
    ) -> None:
        self._worker_command = _browser_worker_command(worker_command)
        self.max_response_bytes = _bounded_int_configuration(
            max_response_bytes,
            "max_response_bytes",
            maximum=MAX_SCREENSHOT_MAX_RESPONSE_BYTES,
        )
        self.timeout_seconds = _bounded_float_configuration(
            timeout_seconds,
            "timeout_seconds",
            maximum=MAX_SCREENSHOT_TIMEOUT_SECONDS,
        )
        self.max_redirects = _bounded_int_configuration(
            max_redirects,
            "max_redirects",
            minimum=0,
            maximum=MAX_SCREENSHOT_REDIRECTS,
        )
        self.max_requests = _bounded_max_requests(max_requests)
        self.max_screenshot_bytes = _bounded_int_configuration(
            max_screenshot_bytes,
            "max_screenshot_bytes",
            maximum=MAX_SCREENSHOT_BYTES,
        )
        self.viewport_width = _bounded_int_configuration(
            viewport_width,
            "viewport_width",
            maximum=MAX_SCREENSHOT_DIMENSION,
        )
        self.viewport_height = _bounded_int_configuration(
            viewport_height,
            "viewport_height",
            maximum=MAX_SCREENSHOT_DIMENSION,
        )
        self.max_page_width = _bounded_int_configuration(
            max_page_width,
            "max_page_width",
            maximum=MAX_SCREENSHOT_DIMENSION,
        )
        self.max_page_height = _bounded_int_configuration(
            max_page_height,
            "max_page_height",
            maximum=MAX_SCREENSHOT_DIMENSION,
        )
        self.max_page_pixels = _bounded_int_configuration(
            max_page_pixels,
            "max_page_pixels",
            maximum=MAX_SCREENSHOT_PAGE_PIXELS,
        )
        if self.viewport_width > self.max_page_width:
            raise ValueError("viewport_width cannot exceed max_page_width.")
        if self.viewport_height > self.max_page_height:
            raise ValueError("viewport_height cannot exceed max_page_height.")
        if self.viewport_width * self.viewport_height > self.max_page_pixels:
            raise ValueError("The configured viewport exceeds max_page_pixels.")
        self.expected_runner_candidate = _expected_runner_candidate(expected_runner_candidate)
        self.expected_environment_authority = _expected_environment_authority(
            expected_environment_authority
        )
        self.expected_workload_authority = _expected_workload_authority(expected_workload_authority)
        self.expected_artifact_store_id = _expected_artifact_store_id(expected_artifact_store_id)
        super().__init__(spec)

    def _execution_profile_material(self) -> dict[str, object] | None:
        """Return bounded material only for Cayu's shipped browser worker."""

        worker_argv = self._worker_command.argv
        if worker_argv is None or worker_argv != list(DEFAULT_BROWSER_FETCH_WORKER_COMMAND):
            return None
        if (
            self.expected_environment_authority is not None
            and self.expected_environment_authority.profile_identity is None
        ):
            return None
        material: dict[str, object] = {
            "worker_command": {
                "kind": self._worker_command.kind,
                "argv": list(worker_argv),
                "shell": self._worker_command.shell,
            },
            "max_response_bytes": self.max_response_bytes,
            "timeout_seconds": self.timeout_seconds,
            "max_redirects": self.max_redirects,
            "max_requests": self.max_requests,
            "max_screenshot_bytes": self.max_screenshot_bytes,
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "max_page_width": self.max_page_width,
            "max_page_height": self.max_page_height,
            "max_page_pixels": self.max_page_pixels,
        }
        if self.expected_runner_candidate is not None:
            material["expected_runner_candidate"] = self.expected_runner_candidate
        if self.expected_environment_authority is not None:
            material["expected_environment_authority"] = {
                "profile_identity": self.expected_environment_authority.profile_identity,
            }
        if self.expected_workload_authority is not None:
            material["expected_workload_authority"] = _workload_authority_material(
                self.expected_workload_authority
            )
        if self.expected_artifact_store_id is not None:
            material["expected_artifact_store_id"] = self.expected_artifact_store_id
        return material

    async def run(self, ctx: ToolContext, args: dict[str, object]) -> ToolResult:
        try:
            requested_url, full_page = _screenshot_arguments(args)
        except (TypeError, ValueError):
            return _error_result(
                "invalid_arguments",
                "A valid public HTTPS URL and optional boolean full_page are required.",
            )
        artifact_store = _screenshot_artifact_store(ctx)
        if artifact_store is None:
            return _error_result(
                "missing_artifact_store",
                "Screenshot capture requires a configured artifact store.",
            )
        if (
            self.expected_artifact_store_id is not None
            and artifact_store.id != self.expected_artifact_store_id
        ):
            return _error_result(
                "capability_refused",
                "The active artifact store does not match the sandboxed WebBridge profile.",
            )
        payload = json.dumps(
            {
                "protocol_version": BROWSER_FETCH_PROTOCOL_VERSION,
                "worker_version": BROWSER_FETCH_WORKER_VERSION,
                "expected_playwright_version": BROWSER_FETCH_PLAYWRIGHT_VERSION,
                "operation": "screenshot",
                "url": requested_url,
                "full_page": full_page,
                "limits": {
                    "max_response_bytes": self.max_response_bytes,
                    "timeout_seconds": self.timeout_seconds,
                    "max_redirects": self.max_redirects,
                    "max_requests": self.max_requests,
                    "max_screenshot_bytes": self.max_screenshot_bytes,
                    "viewport_width": self.viewport_width,
                    "viewport_height": self.viewport_height,
                    "max_page_width": self.max_page_width,
                    "max_page_height": self.max_page_height,
                    "max_page_pixels": self.max_page_pixels,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        execution = await _execute_browser_worker(
            ctx,
            worker_command=self._worker_command,
            expected_runner_candidate=self.expected_runner_candidate,
            expected_environment_authority=self.expected_environment_authority,
            expected_workload_authority=self.expected_workload_authority,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
            output_limit=_screenshot_output_limit(
                max_screenshot_bytes=self.max_screenshot_bytes,
                max_redirects=self.max_redirects,
            ),
            oversized_error="oversized_screenshot",
        )
        if isinstance(execution, ToolResult):
            return execution
        parsed = _browser_screenshot_result(
            execution.stdout,
            requested_url=requested_url,
            full_page=full_page,
            max_response_bytes=self.max_response_bytes,
            max_requests=self.max_requests,
            max_redirects=self.max_redirects,
            max_screenshot_bytes=self.max_screenshot_bytes,
            viewport_width=self.viewport_width,
            viewport_height=self.viewport_height,
            max_page_width=self.max_page_width,
            max_page_height=self.max_page_height,
            max_page_pixels=self.max_page_pixels,
        )
        if isinstance(parsed, ToolResult):
            return parsed
        success, screenshot = parsed
        artifact = await _store_screenshot_artifact(
            artifact_store,
            ctx=ctx,
            screenshot=screenshot,
            success=success,
        )
        if artifact is None:
            return _error_result(
                "artifact_write_failed",
                "The screenshot could not be stored safely.",
            )
        return _screenshot_success_result(success=success, artifact=artifact)


async def _execute_browser_worker(
    ctx: ToolContext,
    *,
    worker_command: ExecCommand,
    expected_runner_candidate: str | None,
    expected_environment_authority: ExecutionEnvironmentAuthority | None,
    expected_workload_authority: RunnerWorkloadAuthority | None,
    payload: str,
    timeout_seconds: float,
    output_limit: int,
    oversized_error: Literal["oversized_response", "oversized_screenshot"],
) -> ExecResult | ToolResult:
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
    if (
        expected_runner_candidate is not None
        and candidate is not None
        and candidate.candidate != expected_runner_candidate
    ):
        return _error_result(
            "capability_refused",
            "The active runner does not match the sandboxed WebBridge profile.",
        )
    if expected_environment_authority is not None:
        if not isinstance(runner, _EnvironmentAuthorityAwareRunnerHandle):
            active_environment_authority = None
        else:
            try:
                active_environment_authority = runner.execution_environment_authority()
            except Exception:
                active_environment_authority = None
        if active_environment_authority != expected_environment_authority:
            return _error_result(
                "capability_refused",
                "The active environment does not match the sandboxed WebBridge profile.",
            )
    if expected_workload_authority is not None:
        if not isinstance(runner, _WorkloadAwareRunnerHandle):
            active_workload = None
        else:
            try:
                active_workload = runner.workload_authority(expected_workload_authority.name)
            except Exception:
                active_workload = None
        if active_workload != expected_workload_authority:
            return _error_result(
                "capability_refused",
                "The active runner does not provide the sandboxed WebBridge workload.",
            )

    try:
        execution = await runner.exec(
            worker_command,
            timeout_s=max(1, math.ceil(timeout_seconds)),
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
        return _error_result(oversized_error, _BROWSER_FETCH_ERROR_MESSAGES[oversized_error])
    try:
        stdout_size = len(execution.stdout.encode("utf-8"))
    except UnicodeEncodeError:
        return _malformed_browser_result()
    if stdout_size > output_limit:
        return _error_result(oversized_error, _BROWSER_FETCH_ERROR_MESSAGES[oversized_error])
    if execution.exit_code != 0:
        code = "browser_unavailable" if execution.exit_code in {2, 126, 127} else "browser_crash"
        return _error_result(code, _BROWSER_FETCH_ERROR_MESSAGES[code])
    return execution


def _bounded_int_configuration(
    value: object,
    name: str,
    *,
    minimum: int = 1,
    maximum: int,
) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}.")
    return value


def _bounded_float_configuration(
    value: object,
    name: str,
    *,
    minimum: float = 0.001,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}.")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}.") from exc
    if not math.isfinite(normalized) or normalized < minimum or normalized > maximum:
        raise ValueError(f"{name} must be a number between {minimum} and {maximum}.")
    return normalized


def _screenshot_arguments(args: object) -> tuple[str, bool]:
    if type(args) is not dict:
        raise ValueError("Only url and full_page are supported.")
    typed_args = cast("dict[object, object]", args)
    if not set(typed_args).issubset({"url", "full_page"}):
        raise ValueError("Only url and full_page are supported.")
    if "url" not in typed_args:
        raise ValueError("url is required.")
    full_page = typed_args.get("full_page", False)
    if type(full_page) is not bool:
        raise TypeError("full_page must be a boolean.")
    return _canonicalize_url(typed_args["url"]), full_page


def _screenshot_artifact_store(ctx: ToolContext) -> ArtifactStore | None:
    authority = ctx._authoritative_artifact_store_for_builtin()
    artifact_store = authority if authority is not None else ctx.artifact_store
    if artifact_store is None:
        return None
    if not isinstance(artifact_store, ArtifactStore):
        raise TypeError("Tool context artifact_store must implement ArtifactStore.")
    return artifact_store


def _screenshot_output_limit(*, max_screenshot_bytes: int, max_redirects: int) -> int:
    encoded_image_bytes = 4 * math.ceil(max_screenshot_bytes / 3)
    bounded_text_bytes = (
        MAX_WEB_FETCH_TITLE_BYTES + 4 * (2 + 2 * max_redirects) * MAX_WEB_FETCH_URL_LENGTH
    )
    return encoded_image_bytes + 2 * bounded_text_bytes + _SCREENSHOT_JSON_OVERHEAD_BYTES


def _browser_screenshot_result(
    stdout: str,
    *,
    requested_url: str,
    full_page: bool,
    max_response_bytes: int,
    max_requests: int,
    max_redirects: int,
    max_screenshot_bytes: int,
    viewport_width: int,
    viewport_height: int,
    max_page_width: int,
    max_page_height: int,
    max_page_pixels: int,
) -> tuple[_BrowserScreenshotSuccess, bytes] | ToolResult:
    try:
        raw = json.loads(stdout)
    except (TypeError, ValueError, RecursionError):
        return _malformed_browser_result()
    if type(raw) is not dict:
        return _malformed_browser_result()
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
        success = _BrowserScreenshotSuccess.model_validate(raw)
        canonical_requested_url = _canonicalize_url(success.requested_url)
        canonical_final_url = _canonicalize_url(success.final_url)
        title = (
            None if success.title is None else require_durable_text(success.title, "browser title")
        )
        canonical_redirects = tuple(
            _BrowserRedirect(
                status_code=redirect.status_code,
                from_url=_canonicalize_url(redirect.from_url),
                to_url=_canonicalize_url(redirect.to_url),
            )
            for redirect in success.redirects
        )
        if (
            canonical_requested_url != requested_url
            or success.full_page is not full_page
            or success.response_bytes > max_response_bytes
            or success.request_count > max_requests
            or len(canonical_redirects) > max_redirects
            or (title is not None and len(title.encode("utf-8")) > MAX_WEB_FETCH_TITLE_BYTES)
            or (
                not full_page
                and (success.width != viewport_width or success.height != viewport_height)
            )
            or success.width > max_page_width
            or success.height > max_page_height
            or success.width * success.height > max_page_pixels
        ):
            return _malformed_browser_result()
        screenshot = base64.b64decode(success.data_base64, validate=True)
        if not screenshot or len(screenshot) > max_screenshot_bytes:
            return _malformed_browser_result()
        _verified_png_dimensions(
            screenshot,
            expected_width=success.width,
            expected_height=success.height,
            max_width=max_page_width,
            max_height=max_page_height,
            max_pixels=max_page_pixels,
        )
    except (TypeError, UnicodeError, ValueError, ValidationError, binascii.Error, RecursionError):
        return _malformed_browser_result()
    normalized = _BrowserScreenshotSuccess.model_validate(
        {
            **success.model_dump(),
            "requested_url": canonical_requested_url,
            "final_url": canonical_final_url,
            "title": title,
            "redirects": [redirect.model_dump() for redirect in canonical_redirects],
        }
    )
    return normalized, screenshot


class _PngRasterValidator:
    """Incrementally validate one bounded PNG zlib stream and its scanline framing."""

    __slots__ = (
        "_decompressor",
        "_expect_filter",
        "_layout_index",
        "_layouts",
        "_row_bytes_remaining",
        "_rows_remaining",
    )

    def __init__(
        self,
        *,
        width: int,
        height: int,
        bit_depth: int,
        color_type: int,
        interlace: int,
    ) -> None:
        samples = _PNG_COLOR_TYPE_SAMPLES[color_type]
        bits_per_pixel = samples * bit_depth
        self._layouts = _png_raster_layouts(
            width=width,
            height=height,
            bits_per_pixel=bits_per_pixel,
            interlace=interlace,
        )
        self._layout_index = 0
        self._rows_remaining = self._layouts[0][1]
        self._row_bytes_remaining = 0
        self._expect_filter = True
        self._decompressor = zlib.decompressobj()

    def feed(self, compressed: bytes) -> None:
        pending = compressed
        while True:
            try:
                decoded = self._decompressor.decompress(
                    pending,
                    _PNG_DECOMPRESSION_CHUNK_BYTES,
                )
            except zlib.error as exc:
                raise ValueError("PNG raster stream is invalid.") from exc
            self._consume(decoded)
            if self._decompressor.unused_data:
                raise ValueError("PNG raster stream contains trailing compressed data.")
            pending = self._decompressor.unconsumed_tail
            if pending:
                continue
            if len(decoded) == _PNG_DECOMPRESSION_CHUNK_BYTES and not self._decompressor.eof:
                # The output ceiling can leave decoded bytes buffered even when
                # zlib consumed the complete input slice. Drain them without
                # allowing one compressed chunk to allocate the full raster.
                pending = b""
                continue
            return

    def finish(self) -> None:
        self.feed(b"")
        if not self._decompressor.eof:
            raise ValueError("PNG raster stream is incomplete.")
        if self._layout_index != len(self._layouts):
            raise ValueError("PNG raster data length is invalid.")

    def _consume(self, decoded: bytes) -> None:
        offset = 0
        while offset < len(decoded):
            if self._layout_index >= len(self._layouts):
                raise ValueError("PNG raster data length is invalid.")
            if self._expect_filter:
                if decoded[offset] > 4:
                    raise ValueError("PNG raster filter is invalid.")
                offset += 1
                self._row_bytes_remaining = self._layouts[self._layout_index][0]
                self._expect_filter = False
            consumed = min(self._row_bytes_remaining, len(decoded) - offset)
            offset += consumed
            self._row_bytes_remaining -= consumed
            if self._row_bytes_remaining == 0:
                self._rows_remaining -= 1
                if self._rows_remaining == 0:
                    self._layout_index += 1
                    if self._layout_index < len(self._layouts):
                        self._rows_remaining = self._layouts[self._layout_index][1]
                self._expect_filter = True


def _png_raster_layouts(
    *,
    width: int,
    height: int,
    bits_per_pixel: int,
    interlace: int,
) -> tuple[tuple[int, int], ...]:
    if interlace == 0:
        passes = ((width, height),)
    else:
        passes = tuple(
            (
                _png_pass_extent(width, x_start, x_step),
                _png_pass_extent(height, y_start, y_step),
            )
            for x_start, y_start, x_step, y_step in _PNG_ADAM7_PASSES
        )
    layouts = tuple(
        ((pass_width * bits_per_pixel + 7) // 8, pass_height)
        for pass_width, pass_height in passes
        if pass_width > 0 and pass_height > 0
    )
    if not layouts:  # pragma: no cover - positive dimensions always select a pass
        raise ValueError("PNG raster layout is invalid.")
    return layouts


def _png_pass_extent(size: int, start: int, step: int) -> int:
    if size <= start:
        return 0
    return (size - start + step - 1) // step


def _verified_png_dimensions(
    content: bytes,
    *,
    expected_width: int | None = None,
    expected_height: int | None = None,
    max_width: int = MAX_SCREENSHOT_DIMENSION,
    max_height: int = MAX_SCREENSHOT_DIMENSION,
    max_pixels: int = MAX_SCREENSHOT_PAGE_PIXELS,
) -> tuple[int, int]:
    if (expected_width is None) != (expected_height is None):
        raise ValueError("Expected PNG dimensions must be supplied together.")
    if type(content) is not bytes or not content.startswith(_PNG_SIGNATURE):
        raise ValueError("Screenshot is not a PNG image.")
    offset = len(_PNG_SIGNATURE)
    width = 0
    height = 0
    saw_ihdr = False
    saw_idat = False
    saw_plte = False
    ended_idat = False
    saw_iend = False
    color_type = -1
    raster: _PngRasterValidator | None = None
    while offset < len(content):
        if len(content) - offset < 12:
            raise ValueError("PNG chunk is truncated.")
        length = struct.unpack_from(">I", content, offset)[0]
        chunk_type = content[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(content) or any(
            byte not in b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            for byte in chunk_type
        ):
            raise ValueError("PNG chunk is malformed.")
        chunk_data = content[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack_from(">I", content, offset + 8 + length)[0]
        actual_crc = zlib.crc32(chunk_data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ValueError("PNG chunk checksum is invalid.")
        if not saw_ihdr:
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError("PNG must begin with one IHDR chunk.")
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            allowed_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                width <= 0
                or height <= 0
                or width > MAX_SCREENSHOT_DIMENSION
                or height > MAX_SCREENSHOT_DIMENSION
                or width * height > MAX_SCREENSHOT_PAGE_PIXELS
                or width > max_width
                or height > max_height
                or width * height > max_pixels
                or (expected_width is not None and width != expected_width)
                or (expected_height is not None and height != expected_height)
                or bit_depth not in allowed_depths.get(color_type, set())
                or compression != 0
                or filtering != 0
                or interlace not in {0, 1}
            ):
                raise ValueError("PNG header is invalid.")
            saw_ihdr = True
            raster = _PngRasterValidator(
                width=width,
                height=height,
                bit_depth=bit_depth,
                color_type=color_type,
                interlace=interlace,
            )
        elif chunk_type == b"IHDR":
            raise ValueError("PNG contains multiple IHDR chunks.")
        elif chunk_type == b"IDAT":
            if ended_idat or saw_iend or (color_type == 3 and not saw_plte):
                raise ValueError("PNG IDAT chunks are out of order.")
            saw_idat = True
            if raster is None:  # pragma: no cover - IHDR ordering invariant
                raise ValueError("PNG raster stream is invalid.")
            raster.feed(chunk_data)
        elif chunk_type == b"PLTE":
            if saw_plte or saw_idat or length == 0 or length % 3 != 0 or length > 768:
                raise ValueError("PNG palette chunk is invalid.")
            saw_plte = True
        elif chunk_type == b"IEND":
            if length != 0 or not saw_idat:
                raise ValueError("PNG IEND chunk is invalid.")
            saw_iend = True
            offset = chunk_end
            break
        else:
            if saw_idat:
                ended_idat = True
            if chunk_type[0] & 0x20 == 0:
                raise ValueError("PNG contains an unsupported critical chunk.")
        offset = chunk_end
    if not saw_iend or offset != len(content):
        raise ValueError("PNG is missing its terminal IEND chunk.")
    if raster is None:  # pragma: no cover - terminal IHDR invariant
        raise ValueError("PNG raster stream is invalid.")
    raster.finish()
    return width, height


def _screenshot_artifact_id(ctx: ToolContext) -> str | None:
    if ctx.idempotency_key is None:
        return None
    digest = hashlib.sha256(
        b"cayu-screenshot-artifact-v1\0"
        + ctx.session_id.encode("utf-8")
        + b"\0"
        + ctx.idempotency_key.encode("utf-8")
    ).hexdigest()[:32]
    return f"art_{digest}"


async def _store_screenshot_artifact(
    artifact_store: ArtifactStore,
    *,
    ctx: ToolContext,
    screenshot: bytes,
    success: _BrowserScreenshotSuccess,
) -> ArtifactMetadata | None:
    sha256 = hashlib.sha256(screenshot).hexdigest()
    artifact_id = _screenshot_artifact_id(ctx)
    filename = f"screenshot-{sha256[:12]}.png"
    metadata = {
        "operation": "screenshot_page",
        "content_sha256": sha256,
        "width": success.width,
        "height": success.height,
        "full_page": success.full_page,
    }
    try:
        artifact = await artifact_store.put_bytes(
            screenshot,
            artifact_id=artifact_id,
            filename=filename,
            content_type="image/png",
            scope=ArtifactScope.SESSION,
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            environment_name=ctx.environment_name,
            metadata=metadata,
        )
    except Exception:
        if artifact_id is None:
            return None
        try:
            existing = copy_artifact_read_result(
                await artifact_store.read_bytes(
                    artifact_id,
                    max_bytes=len(screenshot) + 1,
                ),
                expected_artifact_id=artifact_id,
                max_content_bytes=len(screenshot) + 1,
            )
        except Exception:
            return None
        if existing.truncated or existing.content != screenshot:
            return None
        artifact = existing.metadata
    if type(artifact) is not ArtifactMetadata:
        return None
    try:
        copied = ArtifactMetadata.model_validate(artifact.model_dump())
    except (TypeError, ValueError, ValidationError):
        return None
    if (
        (artifact_id is not None and copied.id != artifact_id)
        or copied.filename != filename
        or copied.content_type != "image/png"
        or copied.size_bytes != len(screenshot)
        or copied.scope is not ArtifactScope.SESSION
        or copied.session_id != ctx.session_id
        or copied.agent_name != ctx.agent_name
        or copied.environment_name != ctx.environment_name
        or dict(copied.metadata) != metadata
    ):
        return None
    return copied


def _screenshot_success_result(
    *,
    success: _BrowserScreenshotSuccess,
    artifact: ArtifactMetadata,
) -> ToolResult:
    redirects = [redirect.model_dump() for redirect in success.redirects]
    structured = {
        "requested_url": success.requested_url,
        "final_url": success.final_url,
        "title": success.title,
        "title_truncated": success.title_truncated,
        "redirects": redirects,
        "full_page": success.full_page,
        "width": success.width,
        "height": success.height,
        "artifact_id": artifact.id,
        "content_type": artifact.content_type,
        "size_bytes": artifact.size_bytes,
    }
    untrusted_parts = [f"URL: {success.final_url}"]
    if success.title is not None:
        untrusted_parts.append(f"Title: {success.title}")
    untrusted = "\n".join(untrusted_parts).replace(
        "</untrusted_web_content>",
        "<\\/untrusted_web_content>",
    )
    attachment = file_attachment(
        artifact_id=artifact.id,
        kind=FileAttachmentKind.IMAGE,
        filename=artifact.filename,
        content_type=artifact.content_type,
        size_bytes=artifact.size_bytes,
        metadata={
            "source": "screenshot_page",
            "width": success.width,
            "height": success.height,
            "full_page": success.full_page,
        },
    )
    return ToolResult(
        content=(
            "Captured sandboxed page screenshot:\n"
            f"Full page: {'true' if success.full_page else 'false'}\n"
            f"Title truncated: {'true' if success.title_truncated else 'false'}\n"
            f"Dimensions: {success.width}x{success.height}\n\n"
            "<untrusted_web_content>\n"
            f"{untrusted}\n"
            "</untrusted_web_content>"
        ),
        structured=structured,
        artifacts=[attachment],
    )


def _malformed_browser_result() -> ToolResult:
    return _error_result(
        "malformed_browser_result",
        "The browser worker returned an invalid result.",
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


def _expected_runner_candidate(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > 96:
        raise ValueError("expected_runner_candidate must be a bounded capability identity.")
    if any(
        not (character.islower() or character.isdigit() or character in "_-") for character in value
    ):
        raise ValueError("expected_runner_candidate must be a bounded capability identity.")
    return value


def _expected_environment_authority(
    value: ExecutionEnvironmentAuthority | None,
) -> ExecutionEnvironmentAuthority | None:
    if value is None:
        return None
    if type(value) is not ExecutionEnvironmentAuthority:
        raise TypeError(
            "expected_environment_authority must be an ExecutionEnvironmentAuthority or None."
        )
    return ExecutionEnvironmentAuthority(
        identity=value.identity,
        profile_identity=value.profile_identity,
    )


def _expected_workload_authority(
    value: RunnerWorkloadAuthority | None,
) -> RunnerWorkloadAuthority | None:
    if value is None:
        return None
    if type(value) is not RunnerWorkloadAuthority:
        raise TypeError("expected_workload_authority must be RunnerWorkloadAuthority or None.")
    return RunnerWorkloadAuthority(
        name=value.name,
        image=value.image,
        command=value.command,
        protocol_version=value.protocol_version,
        worker_version=value.worker_version,
        component_versions=value.component_versions,
    )


def _workload_authority_material(
    value: RunnerWorkloadAuthority | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "name": value.name,
        "image": value.image,
        "command": list(value.command),
        "protocol_version": value.protocol_version,
        "worker_version": value.worker_version,
        "component_versions": [list(component) for component in value.component_versions],
    }


def _expected_artifact_store_id(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value.strip() or len(value.encode("utf-8")) > 512:
        raise ValueError("expected_artifact_store_id must be a bounded nonblank string.")
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


def _browser_runner_is_admitted(
    candidate: ExecutionAdmissionCandidate | None,
    *,
    stage: ExecutionAdmissionStage = "pre_exposure",
) -> bool:
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
        stage=stage,
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
    "ScreenshotPageTool",
]
