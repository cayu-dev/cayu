"""Narrow JSON browser-inspection worker executed inside a Cayu runner.

This module is intentionally not imported by :mod:`cayu.tools.browser`. The
versioned browser image copies it to ``/opt/cayu-browser/worker.py`` and invokes
it as a standalone program, keeping Playwright and Chromium out of the trusted
Cayu host process.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import ipaddress
import json
import math
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Never, cast
from urllib.parse import urljoin, urlsplit

if TYPE_CHECKING or __package__:
    from ._browser_control_guest import (
        GuestCaptureDenied,
        GuestControlAllocationLost,
        GuestControlChannel,
        GuestControlFailure,
        GuestControlFence,
        GuestFrameOwner,
    )
    from ._browser_control_transport import (
        control_connection_closed_normally,
        open_guest_control_channel,
        validate_control_endpoint,
    )
    from ._browser_visual_guest import (
        VISUAL_FAILURE_CODES,
        VisualGuestFailure,
        VisualPageOwner,
        visual_policy_from_json,
    )
else:
    # Isolated Python deliberately excludes the script directory from sys.path.
    # Load only the shipped sibling, without admitting cwd or PYTHONPATH imports.
    _visual_spec = importlib.util.spec_from_file_location(
        "_browser_visual_guest", Path(__file__).resolve().with_name("_browser_visual_guest.py")
    )
    if _visual_spec is None or _visual_spec.loader is None:
        raise ImportError("The browser visual guest module is unavailable.")
    _visual_module = importlib.util.module_from_spec(_visual_spec)
    sys.modules[_visual_spec.name] = _visual_module
    _visual_spec.loader.exec_module(_visual_module)
    VISUAL_FAILURE_CODES = _visual_module.VISUAL_FAILURE_CODES
    VisualGuestFailure = _visual_module.VisualGuestFailure
    VisualPageOwner = _visual_module.VisualPageOwner
    visual_policy_from_json = _visual_module.visual_policy_from_json
    _control_spec = importlib.util.spec_from_file_location(
        "_browser_control_guest", Path(__file__).resolve().with_name("_browser_control_guest.py")
    )
    if _control_spec is None or _control_spec.loader is None:
        raise ImportError("The browser control guest module is unavailable.")
    _control_module = importlib.util.module_from_spec(_control_spec)
    sys.modules[_control_spec.name] = _control_module
    _control_spec.loader.exec_module(_control_module)
    GuestControlFailure = _control_module.GuestControlFailure
    GuestControlAllocationLost = _control_module.GuestControlAllocationLost
    GuestCaptureDenied = _control_module.GuestCaptureDenied
    GuestControlFence = _control_module.GuestControlFence
    GuestFrameOwner = _control_module.GuestFrameOwner
    GuestControlChannel = _control_module.GuestControlChannel
    _transport_spec = importlib.util.spec_from_file_location(
        "_browser_control_transport",
        Path(__file__).resolve().with_name("_browser_control_transport.py"),
    )
    if _transport_spec is None or _transport_spec.loader is None:
        raise ImportError("The browser control transport module is unavailable.")
    _transport_module = importlib.util.module_from_spec(_transport_spec)
    sys.modules[_transport_spec.name] = _transport_module
    _transport_spec.loader.exec_module(_transport_module)
    open_guest_control_channel = _transport_module.open_guest_control_channel
    validate_control_endpoint = _transport_module.validate_control_endpoint
    control_connection_closed_normally = _transport_module.control_connection_closed_normally

PROTOCOL_VERSION = "cayu.browser-fetch.v4"
WORKER_VERSION = "4"
PLAYWRIGHT_VERSION = "1.62.0"
INTERACTIVE_PROTOCOL_VERSION = "cayu.browser-session.v4"
INTERACTIVE_WORKER_VERSION = "10"
CONTROL_BOOTSTRAP_PROTOCOL = "cayu.browser-control-bootstrap.v1"
_BROKER_ERROR_HEADER = "x-cayu-egress-error"
_MAX_URL_LENGTH = 8192
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_CONTENT_BYTES = 256 * 1024
_MAX_REDIRECTS = 10
_MAX_REQUESTS = 512
_MAX_DOM_NODES = 100_000
_MAX_SCREENSHOT_BYTES = 8 * 1024 * 1024
_MAX_SCREENSHOT_DIMENSION = 16_384
_MAX_SCREENSHOT_PAGE_PIXELS = 32_000_000
_MAX_TIMEOUT_SECONDS = 120.0
_MAX_TITLE_BYTES = 512
_ACCESSIBILITY_SNAPSHOT_DEPTH = 32
_MAX_ACCESSIBILITY_INDENT = 64
_MAX_FRAME_DOCUMENTS = 32
_BROWSER_INSPECTION_WORLD = "cayu-browser-inspection"
_RENDER_SETTLE_MILLISECONDS = 250
_FINAL_NETWORK_SETTLE_SECONDS = 0.25
_PLAYWRIGHT_BROWSERS_PATH = "/ms-playwright"
_MAX_CLEANUP_RESERVE_SECONDS = 5.0
_MIN_CLEANUP_RESERVE_SECONDS = 0.25
_MAX_PROFILE_CLEANUP_RESERVE_SECONDS = 1.0
_MIN_PROFILE_CLEANUP_RESERVE_SECONDS = 0.05
_PROFILE_CLEANUP_ARGUMENT = "--cleanup-profile"
_TEMPORARY_PROFILE_PREFIX = "cayu-browser-"
_TEMPORARY_PROFILE_ROOT = Path("/tmp")
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_HTML_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_TEXT_CONTENT_TYPES = frozenset({"text/plain"})
_INTERACTIVE_DAEMON_ARGUMENT = "--interactive-daemon"
_INTERACTIVE_ROOT = Path("/tmp/cayu-browser-sessions")
_INTERACTIVE_IDLE_SECONDS = 15 * 60
_INTERACTIVE_CONNECT_SECONDS = 5.0
_INTERACTIVE_STARTUP_SETTLEMENT_SECONDS = 5.0
_INTERACTIVE_IDLE_POLL_SECONDS = 0.25
_INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES = 1024 * 1024
_INTERACTIVE_MAX_PROFILE_ORIGINS = 32
_INTERACTIVE_MAX_PROFILE_COOKIES = 256
_INTERACTIVE_MAX_PROFILE_STORAGE_ENTRIES = 512
_INTERACTIVE_MAX_PROFILE_PRIVATE_VALUES = 2 * (
    _INTERACTIVE_MAX_PROFILE_COOKIES + _INTERACTIVE_MAX_PROFILE_STORAGE_ENTRIES
)
_INTERACTIVE_MAX_PROFILE_NAME_BYTES = 1024
_INTERACTIVE_MAX_PROFILE_VALUE_BYTES = 64 * 1024
_INTERACTIVE_MAX_REQUEST_BYTES = 7 * _INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES
_INTERACTIVE_MAX_SNAPSHOT_BYTES = 256 * 1024
_INTERACTIVE_MAX_DOM_NODES = 100_000
_INTERACTIVE_MAX_REFS = 1_024
_INTERACTIVE_MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
_INTERACTIVE_MAX_PAGE_DIMENSION = 16_384
_INTERACTIVE_MAX_PAGE_PIXELS = 32_000_000
_INTERACTIVE_MAX_WAIT_MS = 120_000
_INTERACTIVE_MAX_IDLE_SECONDS = 60 * 60
_INTERACTIVE_MAX_REDIRECTS = 10
_INTERACTIVE_MAX_REQUESTS = 512
_INTERACTIVE_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_INTERACTIVE_MAX_OPERATIONS = 16_384
_INTERACTIVE_MAX_PAGES = 16
_INTERACTIVE_MAX_PROVISIONAL_PAGES = 16
_INTERACTIVE_MAX_PAGE_CREATIONS_PER_OPERATION = 16
_INTERACTIVE_MAX_TOTAL_PAGE_CREATIONS = 128
_INTERACTIVE_MAX_BACKGROUND_LIFETIME_SECONDS = 60 * 60
_INTERACTIVE_MAX_OPERATIONS_PER_PAGE = 16_384
_INTERACTIVE_MAX_OBSERVATIONS_PER_PAGE = 16_384
_INTERACTIVE_MAX_TOTAL_OBSERVATIONS = 16_384
_INTERACTIVE_MAX_REFS_PER_PAGE = 16_384
_INTERACTIVE_MAX_TOTAL_REFS = 16_384
_INTERACTIVE_MAX_TOTAL_REQUESTS = 65_536
_INTERACTIVE_MAX_ARTIFACTS_PER_PAGE = 16_384
_INTERACTIVE_MAX_TOTAL_ARTIFACTS = 16_384
_INTERACTIVE_MAX_PAGE_CLEANUP_OPERATIONS = 16_384
_INTERACTIVE_MAX_POPUP_POLICY_ORIGINS = 64
_INTERACTIVE_POPUP_EFFECT_OPERATIONS = frozenset(
    {
        "navigate",
        "click",
        "fill",
        "select",
        "press",
        "wait",
        "download",
        "click_visual_target",
        "click_visual_point",
        "back",
        "forward",
        "reload",
        "scroll",
        "hover",
        "upload",
    }
)
_INTERACTIVE_MAX_SCROLL_REPEATS = 16
_INTERACTIVE_MAX_UPLOAD_FILES = 16
_INTERACTIVE_MAX_UPLOAD_FILE_BYTES = 32 * 1024 * 1024
_INTERACTIVE_MAX_UPLOAD_TOTAL_BYTES = 32 * 1024 * 1024
_INTERACTIVE_MAX_UPLOAD_FILENAME_BYTES = 255
_INTERACTIVE_MAX_UPLOAD_MATERIALIZATION_MS = 120_000
_INTERACTIVE_MAX_UPLOAD_CONTENT_TYPE_BYTES = 255
_INTERACTIVE_OPERATION_LEDGER_BYTES = 16 * 1024 * 1024
_OBSERVATION_GUARD_SETTLEMENT_SECONDS = 1.0
_INTERACTIVE_MAX_ELEMENT_TEXT_BYTES = 2 * 1024
_INTERACTIVE_MAX_TITLE_ENVELOPE_BYTES = 4 * 1024
_INTERACTIVE_ACCESSIBILITY_SOURCE_MULTIPLIER = 8
_INTERACTIVE_ACCESSIBILITY_SERIALIZATION_MULTIPLIER = 8
_INTERACTIVE_MAX_ACCESSIBILITY_MATERIALIZATION_BYTES = 64 * 1024 * 1024
_INTERACTIVE_ACCESSIBILITY_NODE_ENVELOPE_BYTES = 512
_INTERACTIVE_RETIREMENT_BUCKET_HEX_LENGTH = 3
_INTERACTIVE_PROFILE_OUTPUT_OMITTED = "Browser content omitted because it contained profile state."
_INTERACTIVE_POPUP_GUARD = r"""(configuration => {
    const admissionToken = configuration.token;
    const isSafeInteger = Number.isSafeInteger;
    const apply = Reflect.apply;
    const NativeURL = URL;
    const getAttribute = Element.prototype.getAttribute;
    const querySelector = Document.prototype.querySelector;
    const composedPath = Event.prototype.composedPath;
    const preventDefault = Event.prototype.preventDefault;
    const stopImmediatePropagation = Event.prototype.stopImmediatePropagation;
    const trim = String.prototype.trim;
    const toLowerCase = String.prototype.toLowerCase;
    const urlToString = URL.prototype.toString;
    const stringify = String;
    const Anchor = HTMLAnchorElement;
    const Area = HTMLAreaElement;
    const Form = HTMLFormElement;
    const HtmlElement = HTMLElement;
    let remainingCreations = 0;
    let admittedURLs = [];
    let blockedAttempts = 0;
    const normalized = value => apply(
        toLowerCase,
        apply(trim, stringify(value), []),
        [],
    );
    const attribute = (element, name) => apply(getAttribute, element, [name]);
    const recordBlocked = reason => {
        blockedAttempts |= reason;
        return false;
    };
    const consumeAdmission = () => {
        if (remainingCreations <= 0) {
            return recordBlocked(1);
        }
        remainingCreations -= 1;
        return true;
    };
    const setAdmission = (candidate, count) => {
        if (candidate !== admissionToken || !isSafeInteger(count) || count < 0) {
            return -1;
        }
        const outcome = {blocked: blockedAttempts, urls: admittedURLs};
        remainingCreations = count;
        admittedURLs = [];
        blockedAttempts = 0;
        return outcome;
    };
    Object.defineProperty(window, "__cayuSetPopupAdmission", {
        value: setAdmission,
        writable: false,
        configurable: false,
        enumerable: false,
    });
    const sameContextTarget = target => {
        const value = normalized(target || "");
        return value === "" || value === "_self" ||
            value === "_parent" || value === "_top";
    };
    const popupTargetAllowed = target => normalized(target || "_blank") === "_blank";
    const popupURL = raw => {
        const value = apply(trim, stringify(raw == null ? "" : raw), []);
        if (value === "" || value === "about:blank") return "about:blank";
        try {
            const parsed = new NativeURL(value, document.baseURI);
            if (parsed.protocol !== "https:") return null;
            return apply(urlToString, parsed, []);
        } catch {
            return null;
        }
    };
    const declaredTarget = element => {
        const explicitTarget = attribute(element, "target");
        if (explicitTarget !== null) return explicitTarget;
        const base = apply(querySelector, element.ownerDocument, ["base[target]"]);
        return base === null ? "" : attribute(base, "target");
    };
    const formTarget = (form, submitter) => {
        if (submitter instanceof HtmlElement) {
            const override = attribute(submitter, "formtarget");
            if (override !== null) return override;
        }
        return declaredTarget(form);
    };
    const formURL = (form, submitter) => {
        if (submitter instanceof HtmlElement) {
            const override = attribute(submitter, "formaction");
            if (override !== null) return override;
        }
        return attribute(form, "action");
    };
    const admitPopup = (target, url) => {
        const resolvedURL = popupURL(url);
        if (!popupTargetAllowed(target) || resolvedURL === null) {
            return recordBlocked(2);
        }
        if (!consumeAdmission()) return false;
        admittedURLs.push(resolvedURL);
        return true;
    };
    const stop = event => {
        apply(preventDefault, event, []);
        apply(stopImmediatePropagation, event, []);
    };
    const nativeOpen = window.open;
    const guardedOpen = function(...args) {
        const target = args.length > 1 ? args[1] : "_blank";
        if (sameContextTarget(target)) return apply(nativeOpen, this, args);
        if (!admitPopup(target, args[0])) return null;
        return apply(nativeOpen, this, args);
    };
    Object.defineProperty(Window.prototype, "open", {
        value: guardedOpen,
        writable: false,
        configurable: false,
    });
    Object.defineProperty(window, "open", {
        value: guardedOpen,
        writable: false,
        configurable: false,
    });
    window.addEventListener("click", event => {
        for (const item of apply(composedPath, event, [])) {
            if ((item instanceof Anchor || item instanceof Area) &&
                    !sameContextTarget(declaredTarget(item)) &&
                    !admitPopup(declaredTarget(item), attribute(item, "href"))) {
                stop(event);
                return;
            }
        }
    }, true);
    window.addEventListener("submit", event => {
        if (event.target instanceof Form &&
                !sameContextTarget(formTarget(event.target, event.submitter)) &&
                !admitPopup(
                    formTarget(event.target, event.submitter),
                    formURL(event.target, event.submitter),
                )) {
            stop(event);
        }
    }, true);
    const nativeSubmit = Form.prototype.submit;
    Object.defineProperty(Form.prototype, "submit", {
        value: function(...args) {
            if (!sameContextTarget(declaredTarget(this)) &&
                    !admitPopup(declaredTarget(this), attribute(this, "action"))) {
                return undefined;
            }
            return apply(nativeSubmit, this, args);
        },
        writable: false,
        configurable: false,
    });
})(__CAYU_POPUP_CONFIGURATION__)"""
_INTERACTIVE_RESPONSE_FIXED_BYTES = 1024 * 1024
_INTERACTIVE_REF_ENVELOPE_BYTES = 6 * 128 + 6 * 128 + 6 * _INTERACTIVE_MAX_ELEMENT_TEXT_BYTES + 256
_INTERACTIVE_MAX_MESSAGE_BYTES = (
    4 * ((_INTERACTIVE_MAX_ARTIFACT_BYTES + 2) // 3)
    + 6 * _INTERACTIVE_MAX_SNAPSHOT_BYTES
    + _INTERACTIVE_MAX_REFS * _INTERACTIVE_REF_ENVELOPE_BYTES
    + 6 * (_MAX_URL_LENGTH + _INTERACTIVE_MAX_TITLE_ENVELOPE_BYTES)
    + _INTERACTIVE_MAX_TOTAL_PAGE_CREATIONS
    * (6 * (_MAX_URL_LENGTH + _INTERACTIVE_MAX_TITLE_ENVELOPE_BYTES + 256) + 4_096)
    + 7 * _INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES
    + _INTERACTIVE_RESPONSE_FIXED_BYTES
)
_INTERACTIVE_MAX_PRIVATE_REQUEST_BYTES = (
    4 * ((_INTERACTIVE_MAX_UPLOAD_TOTAL_BYTES + 2) // 3) + _INTERACTIVE_MAX_REQUEST_BYTES
)
_INTERACTIVE_REF_PATTERN = re.compile(
    r"(?P<prefix>(?:[ \t]+\[(?:checked(?:=mixed)?|disabled|expanded|active|"
    r"invalid(?:=grammar|=spelling)?|level=[0-9]+|pressed(?:=mixed)?|selected)\])*"
    r"[ \t]+)\[ref=(?P<ref>[A-Za-z0-9._:-]{1,128})\]"
    r"(?=(?:\s+\[[^\]\r\n]{1,256}\])*(?:\s*:\s*[^\r\n]*)?\s*$)"
)
# Slash-delimited names are literal in AI snapshots. Never consume a value
# separator while finding their closing slash: values may contain forged refs.
_INTERACTIVE_ELEMENT_PATTERN = re.compile(
    r"^\s*-\s+([A-Za-z][A-Za-z0-9_-]{0,127})"
    r'(?:[ \t]+(?:"((?:\\.|[^"\\])*)"|(/(?:[^:\r\n]*/)?)))?'
)
_INTERACTIVE_SAFE_ID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?$")
_MAX_ACCESS_RETRY_AFTER_SECONDS = 24 * 60 * 60


def _interactive_popup_guard(admission_token: str) -> str:
    """Bind one unguessable daemon-owned admission token into the init script."""

    return _INTERACTIVE_POPUP_GUARD.replace(
        "__CAYU_POPUP_CONFIGURATION__",
        json.dumps(
            {"token": admission_token},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


class _GuestFailure(RuntimeError):
    def __init__(
        self,
        code: str,
        *,
        status_code: int | None = None,
        allocation_disposition: Literal["live", "retired", "uncertain"] = "uncertain",
        access: dict[str, Any] | None = None,
        effective_origin: str | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.allocation_disposition = allocation_disposition
        self.access = access
        self.effective_origin = effective_origin
        super().__init__(code)


def _guest_http_access(
    url: str,
    status_code: int,
    headers: Any,
    *,
    source: Literal["browser_response"],
) -> dict[str, Any] | None:
    """Classify bounded response metadata without reading page-controlled text."""

    if type(headers) is not dict:
        headers = {}
    normalized: dict[str, str] = {}
    allowed = {
        "cf-mitigated",
        "retry-after",
        "www-authenticate",
        "x-cayu-access-block",
        "x-cayu-access-requirement",
    }
    for key, value in headers.items():
        if type(key) is str and type(value) is str and key.lower() in allowed:
            normalized[key.lower()] = value.encode("utf-8", errors="replace")[:256].decode(
                "utf-8", errors="ignore"
            )
    outcome: str | None = None
    signal = "status_code"
    retry_after: int | None = None
    retry_after_unrepresentable = False
    if normalized.get("x-cayu-access-requirement", "").lower() == "consent":
        outcome = "consent_required"
        signal = "consent_header"
    elif (
        normalized.get("cf-mitigated", "").lower() == "challenge"
        or normalized.get("x-cayu-access-block", "").lower() == "bot_challenge"
    ):
        outcome = "bot_challenge"
        signal = "challenge_header"
    elif status_code == 401:
        if normalized.get("www-authenticate"):
            outcome = "authentication_required"
            signal = "www_authenticate"
        else:
            outcome = "bot_challenge"
    elif status_code == 407:
        outcome = "authentication_required"
        signal = "www_authenticate"
    elif status_code == 429:
        outcome = "rate_limited"
        retry_after, retry_after_unrepresentable = _guest_retry_after_seconds(
            normalized.get("retry-after")
        )
        signal = (
            "retry_after"
            if retry_after is not None or retry_after_unrepresentable
            else "status_code"
        )
    elif status_code in {403, 451}:
        outcome = "destination_denied"
    elif status_code in {404, 410}:
        outcome = "content_unavailable"
    elif status_code in {408, 425} or status_code >= 500:
        outcome = "transient_transport_failure"
    if outcome is None:
        return None
    effective_origin = _guest_https_origin(url)
    destination = hashlib.sha256(
        b"cayu.web-access-destination.v1\0" + effective_origin.encode("utf-8")
    ).hexdigest()
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "outcome": outcome,
        "source": source,
        "signal": signal,
        "destination_fingerprint": destination,
        "status_code": status_code,
        "retry_after_seconds": retry_after,
        "retry_after_unrepresentable": retry_after_unrepresentable,
    }
    return evidence


def _guest_https_origin(url: str) -> str:
    split = urlsplit(url)
    try:
        port = split.port
    except ValueError as exc:
        raise _GuestFailure("browser_crash") from exc
    if (
        split.scheme.lower() != "https"
        or split.hostname is None
        or split.username is not None
        or split.password is not None
        or port not in {None, 443}
    ):
        raise _GuestFailure("browser_crash")
    try:
        hostname = split.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    except UnicodeError as exc:
        raise _GuestFailure("browser_crash") from exc
    else:
        raise _GuestFailure("browser_crash")
    if not hostname:
        raise _GuestFailure("browser_crash")
    return f"https://{hostname}/"


def _guest_websocket_https_origin(url: Any) -> str:
    if type(url) is not str or not 0 < len(url) <= _MAX_URL_LENGTH:
        raise _GuestFailure("browser_crash")
    split = urlsplit(url)
    try:
        port = split.port
    except ValueError as exc:
        raise _GuestFailure("browser_crash") from exc
    if (
        split.scheme.lower() != "wss"
        or split.hostname is None
        or split.username is not None
        or split.password is not None
        or port not in {None, 443}
    ):
        raise _GuestFailure("browser_crash")
    return _guest_https_origin(f"https://{split.hostname}/")


def _guest_retry_after_seconds(value: str | None) -> tuple[int | None, bool]:
    if value is None:
        return None, False
    stripped = value.strip()
    if not stripped:
        return None, False
    if stripped.isascii() and stripped.isdigit():
        if len(stripped) > 128:
            return None, True
        canonical_digits = stripped.lstrip("0") or "0"
        if len(canonical_digits) > 5:
            return None, True
        seconds = int(canonical_digits)
        if seconds > _MAX_ACCESS_RETRY_AFTER_SECONDS:
            return None, True
        return seconds, False
    if len(stripped) > 128:
        return None, False
    try:
        target = parsedate_to_datetime(stripped)
    except (TypeError, ValueError, OverflowError):
        return None, False
    if target.tzinfo is None:
        return None, False
    delta = math.ceil((target.astimezone(UTC) - datetime.now(UTC)).total_seconds())
    if delta > _MAX_ACCESS_RETRY_AFTER_SECONDS:
        return None, True
    if delta >= 0:
        return delta, False
    return None, False


@dataclass(frozen=True)
class _BrowserLimits:
    max_response_bytes: int
    timeout_seconds: float
    max_redirects: int
    max_requests: int


@dataclass(frozen=True)
class _Limits(_BrowserLimits):
    max_content_bytes: int
    max_dom_nodes: int


@dataclass(frozen=True)
class _ScreenshotLimits(_BrowserLimits):
    max_screenshot_bytes: int
    viewport_width: int
    viewport_height: int
    max_page_width: int
    max_page_height: int
    max_page_pixels: int


@dataclass(frozen=True)
class _Request:
    url: str
    limits: _Limits | _ScreenshotLimits
    operation: Literal["fetch", "screenshot"] = "fetch"
    full_page: bool = False


@dataclass
class _PageState:
    max_response_bytes: int
    max_redirects: int
    max_requests: int
    response_bytes: int = 0
    request_count: int = 0
    limit_exceeded: bool = False
    denied_code: str | None = None
    response_inspection_failed: bool = False
    browser_crashed: bool = False
    cleanup_failed: bool = False
    access_evidence: dict[str, Any] | None = None
    effective_origin: str | None = None
    redirects: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _BrowserCleanupOutcome:
    errors: tuple[BaseException, ...] = ()
    cancellation: asyncio.CancelledError | None = None


@dataclass(frozen=True)
class _FrameIdentity:
    frame_id: str
    loader_id: str
    url: str
    mime_type: str
    parent_index: int | None


@dataclass(frozen=True)
class _ScreenshotDocumentIdentity:
    frame_id: str
    loader_id: str
    url: str


@dataclass(frozen=True)
class _FrameProjection:
    identity: _FrameIdentity
    title: str | None
    text: str
    title_truncated: bool
    text_truncated: bool
    semantic_structure: bool
    node_count: int


@dataclass
class _TemporaryProfileOwner:
    home: Path
    process: subprocess.Popen[bytes]
    control_fd: int | None

    @property
    def pid(self) -> int:
        return self.process.pid


@dataclass(frozen=True)
class _InteractiveLimits:
    max_snapshot_bytes: int
    max_dom_nodes: int
    max_refs: int
    max_artifact_bytes: int
    max_page_width: int
    max_page_height: int
    max_page_pixels: int
    max_wait_ms: int
    idle_timeout_seconds: int
    max_redirects: int
    max_requests: int
    max_response_bytes: int
    max_operations: int
    max_pages: int
    max_provisional_pages: int
    max_page_creations_per_operation: int
    max_total_page_creations: int
    max_background_lifetime_seconds: int
    max_operations_per_page: int
    max_observations_per_page: int
    max_total_observations: int
    max_refs_per_page: int
    max_total_refs: int
    max_total_requests: int
    max_artifacts_per_page: int
    max_total_artifacts: int
    max_page_cleanup_operations: int

    max_scroll_repeats: int
    max_upload_files: int
    max_upload_file_bytes: int
    max_upload_total_bytes: int
    max_upload_filename_bytes: int
    max_upload_materialization_ms: int


@dataclass(frozen=True)
class _InteractivePopupPolicy:
    mode: Literal["deny", "same_origin", "destination_policy"]
    allowed_operations: tuple[str, ...]
    allowed_opener_origins: tuple[str, ...]
    allowed_destination_origins: tuple[str, ...]


@dataclass(frozen=True)
class _InteractiveUploadFile:
    artifact_id: str
    content_sha256: str
    filename: str
    content_type: str
    size_bytes: int
    content: bytes

    def fingerprint_material(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _InteractiveRequest:
    operation: Literal[
        "navigate",
        "observe",
        "observe_visual",
        "click_visual_target",
        "click_visual_point",
        "click",
        "fill",
        "select",
        "press",
        "wait",
        "screenshot",
        "download",
        "list_pages",
        "switch_page",
        "close_page",
        "back",
        "forward",
        "reload",
        "scroll",
        "hover",
        "upload",
        "close",
        "profile_restore",
        "profile_checkpoint",
    ]
    session_id: str
    page_id: str | None
    expected_revision: str | None
    expected_control_epoch: int | None
    ref: str | None
    operation_id: str
    url: str | None
    value: str | None
    key: str | None
    wait_ms: int | None
    full_page: bool
    direction: Literal["up", "down", "left", "right"] | None
    amount: Literal["line", "page"] | None
    repeat_count: int | None
    upload_files: tuple[_InteractiveUploadFile, ...]
    limits: _InteractiveLimits
    multi_page: bool
    popup_policy: _InteractivePopupPolicy
    profile_restore_state: dict[str, Any] | None = field(default=None, repr=False)
    profile_allowed_origins: tuple[str, ...] | None = None
    profile_plaintext_limit: int | None = None
    profile_capture_limit: int | None = None
    profile_timeout_seconds: float | None = None
    visual_policy: dict[str, Any] | None = None
    visual_revision: str | None = None
    visual_ref: str | None = None
    screenshot_sha256: str | None = None
    x: float | None = None
    y: float | None = None
    reconcile_only: bool = False
    invocation_control_epoch: int | None = None


@dataclass(frozen=True)
class _InteractiveOperationRecord:
    fingerprint: str
    response: dict[str, Any]
    size_bytes: int


@dataclass
class _InteractivePage:
    page: Any
    session_id: str
    page_id: str
    creation_epoch: int = 1
    control_epoch: int = 1
    lifecycle: Literal[
        "provisional",
        "admitted",
        "active",
        "background",
        "closing",
        "closed",
        "crashed",
        "uncertain",
    ] = "provisional"
    opener_page_id: str | None = None
    opener_origin: str | None = None
    creating_operation_id_sha256: str | None = None
    last_operation_id_sha256: str | None = None
    terminal_reason: str | None = None
    created_monotonic: float = 0.0
    background_since: float | None = None
    operation_count: int = 0
    observation_count: int = 0
    ref_count: int = 0
    artifact_count: int = 0
    configured: bool = False
    staged_initial_url: str | None = None
    public_url: str | None = None
    title: str | None = None
    cdp: Any = None
    revision: str | None = None
    last_observation_revision: str | None = None
    refs: dict[str, str] = field(default_factory=dict)
    # Playwright's ``aria-ref`` locator is intentionally re-resolvable.  Keep
    # the exact element observed for each opaque ref so a later action cannot
    # silently bind to a replacement node with the same accessible identity.
    # ``None`` exists only for internally constructed/test states that predate
    # an observation; production observations always install a complete map.
    ref_targets: dict[str, Any] | None = None
    visual_owner: VisualPageOwner | None = None
    observation_cleanup_task: asyncio.Task[tuple[BaseException, ...]] | None = None
    observation_cleanup_disposition: Literal["retired", "uncertain"] | None = None
    visual_action_cleanup_disposition: Literal["retired", "uncertain"] | None = None
    request_count: int = 0
    redirect_count: int = 0
    response_bytes: int = 0
    navigation_epoch: int = 0
    limit_exceeded: bool = False
    limit_error_code: Literal["oversized_response", "oversized_snapshot", "resource_exhausted"] = (
        "oversized_response"
    )
    denied_code: str | None = None
    access_evidence: dict[str, Any] | None = None
    limit_abort_task: asyncio.Task[bool] | None = None
    current_document_method: str | None = None
    document_methods: dict[str, str] = field(default_factory=dict)
    history_methods: dict[int, str] = field(default_factory=dict)
    dialog_observed: bool = False
    cleanup_task: asyncio.Task[None] | None = None
    unexpected_download_task: asyncio.Task[bool] | None = None
    authorized_download_operation_id_sha256: str | None = None

    def clear_refs(self) -> None:
        self.refs.clear()
        if self.ref_targets is not None:
            self.ref_targets.clear()


@dataclass
class _InteractivePageDelta:
    created_page_ids: set[str] = field(default_factory=set)
    admitted_page_ids: set[str] = field(default_factory=set)
    closed_page_ids: set[str] = field(default_factory=set)
    crashed_page_ids: set[str] = field(default_factory=set)
    refused: list[dict[str, str]] = field(default_factory=list)
    candidate_count: int = 0
    candidate_identities: set[int] = field(default_factory=set)
    candidate_pages: list[Any] = field(default_factory=list)
    staged_frames: dict[Any, tuple[str, str]] = field(default_factory=dict)


def _interactive_page_failure(state: _InteractivePage) -> _GuestFailure | None:
    if state.denied_code is not None:
        return _GuestFailure(state.denied_code)
    if state.access_evidence is not None:
        return _GuestFailure("access_blocked", access=state.access_evidence)
    if state.limit_exceeded:
        return _GuestFailure(state.limit_error_code)
    return None


async def _settle_interactive_navigation_failure(
    state: _InteractivePage,
) -> _GuestFailure | None:
    """Let already-delivered CDP/proxy evidence outrank Playwright's generic error."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _FINAL_NETWORK_SETTLE_SECONDS
    while True:
        if state.denied_code is not None or state.access_evidence is not None:
            return _interactive_page_failure(state)
        remaining = deadline - loop.time()
        if remaining <= 0:
            return _interactive_page_failure(state)
        await asyncio.sleep(min(0.01, remaining))


def _record_page_denial(state: _PageState, code: str) -> None:
    """Retain the first denial while allowing precise redirect evidence to win."""

    if state.denied_code is None or (
        state.denied_code == "destination_denied" and code == "redirect_denied"
    ):
        state.denied_code = code


def _bounded_int(value: Any, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise _GuestFailure("incompatible_browser")
    return value


def _bounded_float(value: Any, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _GuestFailure("incompatible_browser")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise _GuestFailure("incompatible_browser") from exc
    if not math.isfinite(normalized) or normalized < minimum or normalized > maximum:
        raise _GuestFailure("incompatible_browser")
    return normalized


def _request_from_json(raw: Any) -> _Request:
    if type(raw) is not dict or not {
        "expected_playwright_version",
        "limits",
        "operation",
        "protocol_version",
        "url",
        "worker_version",
    }.issubset(raw):
        raise _GuestFailure("incompatible_browser")
    operation = raw["operation"]
    if (
        raw["protocol_version"] != PROTOCOL_VERSION
        or raw["worker_version"] != WORKER_VERSION
        or raw["expected_playwright_version"] != PLAYWRIGHT_VERSION
        or type(operation) is not str
        or operation not in {"fetch", "screenshot"}
    ):
        raise _GuestFailure("incompatible_browser")
    expected_keys = {
        "expected_playwright_version",
        "limits",
        "operation",
        "protocol_version",
        "url",
        "worker_version",
    }
    if operation == "screenshot":
        expected_keys.add("full_page")
    if set(raw) != expected_keys:
        raise _GuestFailure("incompatible_browser")
    url = raw["url"]
    if type(url) is not str or not 0 < len(url) <= _MAX_URL_LENGTH:
        raise _GuestFailure("destination_denied")
    split = urlsplit(url)
    try:
        port = split.port
    except ValueError as exc:
        raise _GuestFailure("destination_denied") from exc
    if (
        split.scheme != "https"
        or split.hostname is None
        or split.username is not None
        or split.password is not None
        or port not in {None, 443}
        or split.fragment
    ):
        raise _GuestFailure("destination_denied")
    raw_limits = raw["limits"]
    fetch_limit_keys = {
        "max_content_bytes",
        "max_dom_nodes",
        "max_redirects",
        "max_requests",
        "max_response_bytes",
        "timeout_seconds",
    }
    screenshot_limit_keys = {
        "max_page_height",
        "max_page_pixels",
        "max_page_width",
        "max_redirects",
        "max_requests",
        "max_response_bytes",
        "max_screenshot_bytes",
        "timeout_seconds",
        "viewport_height",
        "viewport_width",
    }
    expected_limit_keys = fetch_limit_keys if operation == "fetch" else screenshot_limit_keys
    if type(raw_limits) is not dict or set(raw_limits) != expected_limit_keys:
        raise _GuestFailure("incompatible_browser")
    max_response_bytes = _bounded_int(
        raw_limits["max_response_bytes"],
        minimum=1,
        maximum=_MAX_RESPONSE_BYTES,
    )
    timeout_seconds = _bounded_float(
        raw_limits["timeout_seconds"],
        minimum=0.001,
        maximum=_MAX_TIMEOUT_SECONDS,
    )
    max_redirects = _bounded_int(
        raw_limits["max_redirects"],
        minimum=0,
        maximum=_MAX_REDIRECTS,
    )
    max_requests = _bounded_int(
        raw_limits["max_requests"],
        minimum=1,
        maximum=_MAX_REQUESTS,
    )
    if operation == "fetch":
        limits: _Limits | _ScreenshotLimits = _Limits(
            max_response_bytes=max_response_bytes,
            timeout_seconds=timeout_seconds,
            max_redirects=max_redirects,
            max_requests=max_requests,
            max_content_bytes=_bounded_int(
                raw_limits["max_content_bytes"],
                minimum=1,
                maximum=_MAX_CONTENT_BYTES,
            ),
            max_dom_nodes=_bounded_int(
                raw_limits["max_dom_nodes"],
                minimum=1,
                maximum=_MAX_DOM_NODES,
            ),
        )
        full_page = False
    else:
        limits = _ScreenshotLimits(
            max_response_bytes=max_response_bytes,
            timeout_seconds=timeout_seconds,
            max_redirects=max_redirects,
            max_requests=max_requests,
            max_screenshot_bytes=_bounded_int(
                raw_limits["max_screenshot_bytes"],
                minimum=1,
                maximum=_MAX_SCREENSHOT_BYTES,
            ),
            viewport_width=_bounded_int(
                raw_limits["viewport_width"],
                minimum=1,
                maximum=_MAX_SCREENSHOT_DIMENSION,
            ),
            viewport_height=_bounded_int(
                raw_limits["viewport_height"],
                minimum=1,
                maximum=_MAX_SCREENSHOT_DIMENSION,
            ),
            max_page_width=_bounded_int(
                raw_limits["max_page_width"],
                minimum=1,
                maximum=_MAX_SCREENSHOT_DIMENSION,
            ),
            max_page_height=_bounded_int(
                raw_limits["max_page_height"],
                minimum=1,
                maximum=_MAX_SCREENSHOT_DIMENSION,
            ),
            max_page_pixels=_bounded_int(
                raw_limits["max_page_pixels"],
                minimum=1,
                maximum=_MAX_SCREENSHOT_PAGE_PIXELS,
            ),
        )
        full_page = raw["full_page"]
        if type(full_page) is not bool:
            raise _GuestFailure("incompatible_browser")
        if (
            limits.viewport_width > limits.max_page_width
            or limits.viewport_height > limits.max_page_height
            or limits.viewport_width * limits.viewport_height > limits.max_page_pixels
        ):
            raise _GuestFailure("incompatible_browser")
    return _Request(
        operation=operation,
        url=url,
        limits=limits,
        full_page=full_page,
    )


def _interactive_request_from_json(raw: Any) -> _InteractiveRequest:
    if type(raw) is not dict:
        raise _GuestFailure("incompatible_browser")
    operation = raw.get("operation")
    if (
        raw.get("protocol_version") != INTERACTIVE_PROTOCOL_VERSION
        or raw.get("worker_version") != INTERACTIVE_WORKER_VERSION
        or raw.get("expected_playwright_version") != PLAYWRIGHT_VERSION
        or operation
        not in {
            "navigate",
            "observe",
            "observe_visual",
            "click_visual_target",
            "click_visual_point",
            "click",
            "fill",
            "select",
            "press",
            "wait",
            "back",
            "forward",
            "reload",
            "scroll",
            "hover",
            "upload",
            "screenshot",
            "download",
            "list_pages",
            "switch_page",
            "close_page",
            "close",
            "profile_restore",
            "profile_checkpoint",
        }
    ):
        raise _GuestFailure("incompatible_browser")
    base = {
        "expected_playwright_version",
        "limits",
        "operation",
        "protocol_version",
        "page_policy",
        "session_id",
        "worker_version",
        "visual_policy",
    }
    page = base | {"page_id", "operation_id"}
    revision = page | {"expected_revision", "expected_control_epoch"}
    expected: dict[str, set[str]] = {
        "navigate": base | {"page_id", "operation_id", "url"},
        "observe": page,
        "observe_visual": page,
        "click_visual_target": revision | {"visual_revision", "visual_ref"},
        "click_visual_point": revision | {"visual_revision", "screenshot_sha256", "x", "y"},
        "click": revision | {"ref"},
        "fill": revision | {"ref", "value"},
        "select": revision | {"ref", "value"},
        "press": revision | {"key", "ref"},
        "wait": revision | {"wait_ms"},
        "back": revision,
        "forward": revision,
        "reload": revision,
        "scroll": revision | {"amount", "direction", "repeat_count"},
        "hover": revision | {"ref"},
        "upload": revision | {"artifact_ids", "ref", "upload_files"},
        "screenshot": revision,
        "download": revision | {"ref"},
        "list_pages": base | {"operation_id"},
        "switch_page": page,
        "close_page": page,
        "close": base | {"operation_id"},
        "profile_restore": base | {"operation_id"},
        "profile_checkpoint": base | {"operation_id"},
    }
    allowed = set(expected[operation])
    allowed.add("reconcile_only")
    allowed.add("invocation_control_epoch")
    if operation == "screenshot":
        allowed.add("full_page")
    if "browser_profile" in raw:
        allowed.add("browser_profile")
    if set(raw) - allowed or expected[operation] - set(raw):
        raise _GuestFailure("incompatible_browser")
    raw_limits = raw.get("limits")
    if type(raw_limits) is not dict or set(raw_limits) != {
        "max_artifact_bytes",
        "max_page_height",
        "max_page_pixels",
        "max_page_width",
        "max_refs",
        "max_redirects",
        "max_requests",
        "max_response_bytes",
        "max_operations",
        "max_pages",
        "max_provisional_pages",
        "max_page_creations_per_operation",
        "max_total_page_creations",
        "max_background_lifetime_seconds",
        "max_operations_per_page",
        "max_observations_per_page",
        "max_total_observations",
        "max_refs_per_page",
        "max_total_refs",
        "max_total_requests",
        "max_artifacts_per_page",
        "max_total_artifacts",
        "max_page_cleanup_operations",
        "max_scroll_repeats",
        "max_snapshot_bytes",
        "max_dom_nodes",
        "max_wait_ms",
        "idle_timeout_seconds",
        "max_upload_files",
        "max_upload_file_bytes",
        "max_upload_total_bytes",
        "max_upload_filename_bytes",
        "max_upload_materialization_ms",
    }:
        raise _GuestFailure("incompatible_browser")
    limits = _InteractiveLimits(
        max_snapshot_bytes=_bounded_int(
            raw_limits["max_snapshot_bytes"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_SNAPSHOT_BYTES,
        ),
        max_dom_nodes=_bounded_int(
            raw_limits["max_dom_nodes"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_DOM_NODES,
        ),
        max_refs=_bounded_int(
            raw_limits["max_refs"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_REFS,
        ),
        max_artifact_bytes=_bounded_int(
            raw_limits["max_artifact_bytes"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_ARTIFACT_BYTES,
        ),
        max_page_width=_bounded_int(
            raw_limits["max_page_width"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_PAGE_DIMENSION,
        ),
        max_page_height=_bounded_int(
            raw_limits["max_page_height"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_PAGE_DIMENSION,
        ),
        max_page_pixels=_bounded_int(
            raw_limits["max_page_pixels"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_PAGE_PIXELS,
        ),
        max_wait_ms=_bounded_int(
            raw_limits["max_wait_ms"],
            minimum=0,
            maximum=_INTERACTIVE_MAX_WAIT_MS,
        ),
        idle_timeout_seconds=_bounded_int(
            raw_limits["idle_timeout_seconds"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_IDLE_SECONDS,
        ),
        max_redirects=_bounded_int(
            raw_limits["max_redirects"],
            minimum=0,
            maximum=_INTERACTIVE_MAX_REDIRECTS,
        ),
        max_requests=_bounded_int(
            raw_limits["max_requests"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_REQUESTS,
        ),
        max_response_bytes=_bounded_int(
            raw_limits["max_response_bytes"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_RESPONSE_BYTES,
        ),
        max_operations=_bounded_int(
            raw_limits["max_operations"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_OPERATIONS,
        ),
        max_scroll_repeats=_bounded_int(
            raw_limits["max_scroll_repeats"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_SCROLL_REPEATS,
        ),
        max_upload_files=_bounded_int(
            raw_limits["max_upload_files"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_UPLOAD_FILES,
        ),
        max_upload_file_bytes=_bounded_int(
            raw_limits["max_upload_file_bytes"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_UPLOAD_FILE_BYTES,
        ),
        max_upload_total_bytes=_bounded_int(
            raw_limits["max_upload_total_bytes"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_UPLOAD_TOTAL_BYTES,
        ),
        max_upload_filename_bytes=_bounded_int(
            raw_limits["max_upload_filename_bytes"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_UPLOAD_FILENAME_BYTES,
        ),
        max_upload_materialization_ms=_bounded_int(
            raw_limits["max_upload_materialization_ms"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_UPLOAD_MATERIALIZATION_MS,
        ),
        max_pages=_bounded_int(raw_limits["max_pages"], minimum=1, maximum=_INTERACTIVE_MAX_PAGES),
        max_provisional_pages=_bounded_int(
            raw_limits["max_provisional_pages"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_PROVISIONAL_PAGES,
        ),
        max_page_creations_per_operation=_bounded_int(
            raw_limits["max_page_creations_per_operation"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_PAGE_CREATIONS_PER_OPERATION,
        ),
        max_total_page_creations=_bounded_int(
            raw_limits["max_total_page_creations"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_TOTAL_PAGE_CREATIONS,
        ),
        max_background_lifetime_seconds=_bounded_int(
            raw_limits["max_background_lifetime_seconds"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_BACKGROUND_LIFETIME_SECONDS,
        ),
        max_operations_per_page=_bounded_int(
            raw_limits["max_operations_per_page"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_OPERATIONS_PER_PAGE,
        ),
        max_observations_per_page=_bounded_int(
            raw_limits["max_observations_per_page"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_OBSERVATIONS_PER_PAGE,
        ),
        max_total_observations=_bounded_int(
            raw_limits["max_total_observations"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_TOTAL_OBSERVATIONS,
        ),
        max_refs_per_page=_bounded_int(
            raw_limits["max_refs_per_page"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_REFS_PER_PAGE,
        ),
        max_total_refs=_bounded_int(
            raw_limits["max_total_refs"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_TOTAL_REFS,
        ),
        max_total_requests=_bounded_int(
            raw_limits["max_total_requests"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_TOTAL_REQUESTS,
        ),
        max_artifacts_per_page=_bounded_int(
            raw_limits["max_artifacts_per_page"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_ARTIFACTS_PER_PAGE,
        ),
        max_total_artifacts=_bounded_int(
            raw_limits["max_total_artifacts"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_TOTAL_ARTIFACTS,
        ),
        max_page_cleanup_operations=_bounded_int(
            raw_limits["max_page_cleanup_operations"],
            minimum=1,
            maximum=_INTERACTIVE_MAX_PAGE_CLEANUP_OPERATIONS,
        ),
    )
    raw_page_policy = raw.get("page_policy")
    if type(raw_page_policy) is not dict or set(raw_page_policy) != {"multi_page", "popup"}:
        raise _GuestFailure("incompatible_browser")
    multi_page = raw_page_policy.get("multi_page")
    raw_popup = raw_page_policy.get("popup")
    if (
        type(multi_page) is not bool
        or type(raw_popup) is not dict
        or set(raw_popup)
        != {
            "mode",
            "allowed_operations",
            "allowed_opener_origins",
            "allowed_destination_origins",
        }
    ):
        raise _GuestFailure("incompatible_browser")
    popup_mode = raw_popup.get("mode")
    allowed_operations = raw_popup.get("allowed_operations")
    allowed_opener_origins = raw_popup.get("allowed_opener_origins")
    allowed_destination_origins = raw_popup.get("allowed_destination_origins")
    if (
        popup_mode not in {"deny", "same_origin", "destination_policy"}
        or type(allowed_operations) is not list
        or any(
            type(value) is not str
            or value
            not in {
                "click",
                "fill",
                "select",
                "press",
                "wait",
                "click_visual_target",
                "click_visual_point",
            }
            for value in allowed_operations
        )
        or allowed_operations != sorted(set(allowed_operations))
        or type(allowed_opener_origins) is not list
        or type(allowed_destination_origins) is not list
        or len(allowed_opener_origins) > _INTERACTIVE_MAX_POPUP_POLICY_ORIGINS
        or len(allowed_destination_origins) > _INTERACTIVE_MAX_POPUP_POLICY_ORIGINS
    ):
        raise _GuestFailure("incompatible_browser")
    for origins in (allowed_opener_origins, allowed_destination_origins):
        if any(type(value) is not str or _interactive_origin(value) != value for value in origins):
            raise _GuestFailure("incompatible_browser")
        if origins != sorted(set(origins)):
            raise _GuestFailure("incompatible_browser")
    if popup_mode == "deny":
        if allowed_operations or allowed_opener_origins or allowed_destination_origins:
            raise _GuestFailure("incompatible_browser")
    elif not allowed_operations:
        raise _GuestFailure("incompatible_browser")
    if not multi_page and (
        popup_mode != "deny"
        or limits.max_pages != 1
        or limits.max_provisional_pages != 1
        or limits.max_page_creations_per_operation != 1
        or limits.max_total_page_creations != 1
    ):
        raise _GuestFailure("incompatible_browser")
    if (
        limits.max_upload_file_bytes > limits.max_upload_total_bytes
        or (multi_page and limits.max_pages < 2)
        or limits.max_provisional_pages > limits.max_pages
        or limits.max_page_creations_per_operation > limits.max_provisional_pages
        or limits.max_total_page_creations < limits.max_pages
        or limits.max_background_lifetime_seconds > limits.idle_timeout_seconds
        or limits.max_operations_per_page > limits.max_operations
        or limits.max_observations_per_page > limits.max_total_observations
        or limits.max_refs > limits.max_refs_per_page
        or limits.max_refs_per_page > limits.max_total_refs
        or limits.max_requests > limits.max_total_requests
        or limits.max_artifacts_per_page > limits.max_total_artifacts
    ):
        raise _GuestFailure("incompatible_browser")
    session_id = _interactive_identifier(raw.get("session_id"))
    page_id = None if "page_id" not in raw else _interactive_identifier(raw["page_id"])
    expected_revision = (
        None
        if "expected_revision" not in raw
        else _interactive_identifier(raw["expected_revision"])
    )
    expected_control_epoch = None
    if "expected_control_epoch" in raw:
        expected_control_epoch = _bounded_int(
            raw["expected_control_epoch"], minimum=1, maximum=2**63 - 1
        )
    ref = None if "ref" not in raw else _interactive_identifier(raw["ref"])
    operation_id = _interactive_identifier(raw.get("operation_id"))
    url = None
    if "url" in raw:
        url = raw["url"]
        if type(url) is not str or not _browser_request_is_admissible(url):
            raise _GuestFailure("destination_denied")
    value = _interactive_text(raw.get("value"), maximum=16_384) if "value" in raw else None
    key = _interactive_text(raw.get("key"), maximum=128) if "key" in raw else None
    wait_ms = None
    if "wait_ms" in raw:
        wait_ms = _bounded_int(raw["wait_ms"], minimum=0, maximum=limits.max_wait_ms)
    full_page = raw.get("full_page", False)
    if type(full_page) is not bool:
        raise _GuestFailure("incompatible_browser")
    reconcile_only = raw.get("reconcile_only", False)
    if type(reconcile_only) is not bool:
        raise _GuestFailure("incompatible_browser")
    profile_restore_state: dict[str, Any] | None = None
    profile_allowed_origins: tuple[str, ...] | None = None
    profile_plaintext_limit: int | None = None
    profile_capture_limit: int | None = None
    profile_timeout_seconds: float | None = None
    raw_profile = raw.get("browser_profile")
    if raw_profile is not None:
        if type(raw_profile) is not dict or set(raw_profile) != {
            "allowed_origins",
            "schema_version",
            "capture",
            "max_plaintext_bytes",
            "restore_state",
            "timeout_seconds",
        }:
            raise _GuestFailure("incompatible_browser")
        capture = raw_profile.get("capture")
        restore_state = raw_profile.get("restore_state")
        if raw_profile.get("schema_version") != 1 or type(capture) is not bool:
            raise _GuestFailure("incompatible_browser")
        profile_limit = _bounded_int(
            raw_profile.get("max_plaintext_bytes"),
            minimum=1,
            maximum=_INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES,
        )
        profile_plaintext_limit = profile_limit
        raw_allowed_origins = raw_profile.get("allowed_origins")
        if type(raw_allowed_origins) is not list or not raw_allowed_origins:
            raise _GuestFailure("incompatible_browser")
        try:
            profile_allowed_origins = tuple(
                _guest_https_origin(value).removesuffix("/") for value in raw_allowed_origins
            )
        except (TypeError, ValueError, _GuestFailure):
            raise _GuestFailure("incompatible_browser") from None
        if (
            len(profile_allowed_origins) > _INTERACTIVE_MAX_PROFILE_ORIGINS
            or profile_allowed_origins != tuple(sorted(set(profile_allowed_origins)))
            or tuple(raw_allowed_origins) != profile_allowed_origins
        ):
            raise _GuestFailure("incompatible_browser")
        profile_timeout_seconds = _bounded_float(
            raw_profile.get("timeout_seconds"),
            minimum=0.001,
            maximum=_MAX_TIMEOUT_SECONDS,
        )
        if operation == "profile_restore":
            if capture or type(restore_state) is not dict:
                raise _GuestFailure("incompatible_browser")
            profile_restore_state = _validate_interactive_profile_state(
                restore_state,
                maximum_bytes=profile_limit,
                allowed_origins=profile_allowed_origins,
                require_canonical_order=True,
            )
        elif operation == "profile_checkpoint":
            if not capture or restore_state is not None:
                raise _GuestFailure("incompatible_browser")
            profile_capture_limit = profile_limit
        else:
            raise _GuestFailure("incompatible_browser")
    elif operation in {"profile_restore", "profile_checkpoint"}:
        raise _GuestFailure("incompatible_browser")
    try:
        visual_policy = visual_policy_from_json(raw.get("visual_policy"))
    except (VisualGuestFailure, ValueError) as exc:
        raise _GuestFailure("visual_publication_denied") from exc
    visual_revision = (
        _interactive_identifier(raw["visual_revision"]) if "visual_revision" in raw else None
    )
    visual_ref = _interactive_identifier(raw["visual_ref"]) if "visual_ref" in raw else None
    screenshot_sha256 = raw.get("screenshot_sha256")
    if screenshot_sha256 is not None and (
        type(screenshot_sha256) is not str
        or re.fullmatch(r"[0-9a-f]{64}", screenshot_sha256) is None
    ):
        raise _GuestFailure("incompatible_browser")
    for axis in ("x", "y"):
        if axis in raw and (type(raw[axis]) not in {int, float} or not 0 <= raw[axis] < 1):
            raise _GuestFailure("visual_point_outside_viewport")
    direction = raw.get("direction")
    amount = raw.get("amount")
    repeat_count = raw.get("repeat_count")
    if operation == "scroll":
        if direction not in {"up", "down", "left", "right"}:
            raise _GuestFailure("invalid_scroll", allocation_disposition="live")
        if amount not in {"line", "page"}:
            raise _GuestFailure("invalid_scroll", allocation_disposition="live")
        try:
            repeat_count = _bounded_int(
                repeat_count,
                minimum=1,
                maximum=limits.max_scroll_repeats,
            )
        except _GuestFailure as exc:
            raise _GuestFailure("invalid_scroll", allocation_disposition="live") from exc
    elif direction is not None or amount is not None or repeat_count is not None:
        raise _GuestFailure("incompatible_browser")
    upload_files = (
        _interactive_upload_files(
            raw.get("upload_files"),
            limits,
            require_content=not reconcile_only,
        )
        if operation == "upload"
        else ()
    )
    if operation == "upload":
        raw_artifact_ids = raw.get("artifact_ids")
        if (
            type(raw_artifact_ids) is not list
            or len(raw_artifact_ids) != len(upload_files)
            or tuple(_interactive_identifier(item) for item in raw_artifact_ids)
            != tuple(item.artifact_id for item in upload_files)
        ):
            raise _GuestFailure("artifact_refused")
    invocation_control_epoch = raw.get("invocation_control_epoch")
    if "invocation_control_epoch" in raw and (
        type(invocation_control_epoch) is not int or not 1 <= invocation_control_epoch <= 2**53 - 1
    ):
        raise _GuestFailure("incompatible_browser")
    return _InteractiveRequest(
        operation=operation,
        session_id=session_id,
        page_id=page_id,
        expected_revision=expected_revision,
        expected_control_epoch=expected_control_epoch,
        ref=ref,
        operation_id=operation_id,
        url=url,
        value=value,
        key=key,
        wait_ms=wait_ms,
        full_page=full_page,
        direction=direction,
        amount=amount,
        repeat_count=repeat_count,
        upload_files=upload_files,
        limits=limits,
        multi_page=multi_page,
        popup_policy=_InteractivePopupPolicy(
            mode=popup_mode,
            allowed_operations=tuple(allowed_operations),
            allowed_opener_origins=tuple(allowed_opener_origins),
            allowed_destination_origins=tuple(allowed_destination_origins),
        ),
        profile_restore_state=profile_restore_state,
        profile_allowed_origins=profile_allowed_origins,
        profile_plaintext_limit=profile_plaintext_limit,
        profile_capture_limit=profile_capture_limit,
        profile_timeout_seconds=profile_timeout_seconds,
        reconcile_only=reconcile_only,
        invocation_control_epoch=invocation_control_epoch,
        visual_policy=visual_policy,
        visual_revision=visual_revision,
        visual_ref=visual_ref,
        screenshot_sha256=screenshot_sha256,
        x=raw.get("x"),
        y=raw.get("y"),
    )


def _validate_interactive_profile_state(
    value: object,
    *,
    maximum_bytes: int,
    allowed_origins: tuple[str, ...],
    require_canonical_order: bool = False,
) -> dict[str, Any]:
    """Own only Playwright's portable cookie/local-storage state shape."""

    if type(value) is not dict or set(value) != {"cookies", "origins"}:
        raise _GuestFailure("incompatible_browser")
    owned_value = cast("dict[str, object]", value)
    cookies = owned_value.get("cookies")
    origins = owned_value.get("origins")
    if type(cookies) is not list or type(origins) is not list:
        raise _GuestFailure("incompatible_browser")
    if (
        len(cookies) > _INTERACTIVE_MAX_PROFILE_COOKIES
        or len(origins) > _INTERACTIVE_MAX_PROFILE_ORIGINS
    ):
        raise _GuestFailure("incompatible_browser")
    admitted_hosts = {urlsplit(origin).hostname for origin in allowed_origins}
    cookie_keys: list[tuple[str, str, str]] = []
    for cookie in cookies:
        if type(cookie) is not dict or set(cookie) != {
            "name",
            "value",
            "domain",
            "path",
            "expires",
            "httpOnly",
            "secure",
            "sameSite",
        }:
            raise _GuestFailure("incompatible_browser")
        owned_cookie = cast("dict[str, Any]", cookie)
        name = _interactive_profile_text(
            owned_cookie.get("name"),
            maximum_bytes=_INTERACTIVE_MAX_PROFILE_NAME_BYTES,
        )
        cookie_value = _interactive_profile_text(
            owned_cookie.get("value"),
            maximum_bytes=_INTERACTIVE_MAX_PROFILE_VALUE_BYTES,
        )
        domain = owned_cookie.get("domain")
        path = owned_cookie.get("path")
        expires = owned_cookie.get("expires")
        try:
            normalized_expiry = (
                None
                if isinstance(expires, bool) or not isinstance(expires, (int, float))
                else float(expires)
            )
        except OverflowError:
            normalized_expiry = None
        if (
            not name
            or any(character in name for character in "=;\r\n\t ")
            or any(character in cookie_value for character in ";\r\n")
            or type(domain) is not str
            or not domain
            or domain.startswith(".")
            or type(path) is not str
            or not path.startswith("/")
            or "\\" in path
            or "\r" in path
            or "\n" in path
            or normalized_expiry is None
            or not math.isfinite(normalized_expiry)
            or (normalized_expiry != -1.0 and normalized_expiry < 0)
            or type(owned_cookie.get("httpOnly")) is not bool
            or owned_cookie.get("secure") is not True
            or owned_cookie.get("sameSite") not in {"Strict", "Lax", "None"}
        ):
            raise _GuestFailure("incompatible_browser")
        _interactive_profile_text(
            domain,
            maximum_bytes=253,
        )
        _interactive_profile_text(
            path,
            maximum_bytes=2_048,
        )
        try:
            normalized_domain = (
                _guest_https_origin(f"https://{domain}").removeprefix("https://").removesuffix("/")
            )
        except _GuestFailure:
            raise _GuestFailure("incompatible_browser") from None
        if domain != normalized_domain or domain not in admitted_hosts:
            raise _GuestFailure("incompatible_browser")
        cookie_keys.append((domain, path, name))
    if len(cookie_keys) != len(set(cookie_keys)) or (
        require_canonical_order and cookie_keys != sorted(cookie_keys)
    ):
        raise _GuestFailure("incompatible_browser")
    storage_count = 0
    observed_origins: list[str] = []
    for origin in origins:
        if type(origin) is not dict or set(origin) != {"origin", "localStorage"}:
            raise _GuestFailure("incompatible_browser")
        owned_origin = cast("dict[str, Any]", origin)
        entries = owned_origin.get("localStorage")
        if type(entries) is not list:
            raise _GuestFailure("incompatible_browser")
        storage_count += len(entries)
        if storage_count > _INTERACTIVE_MAX_PROFILE_STORAGE_ENTRIES:
            raise _GuestFailure("incompatible_browser")
        if any(type(entry) is not dict or set(entry) != {"name", "value"} for entry in entries):
            raise _GuestFailure("incompatible_browser")
        raw_origin = owned_origin.get("origin")
        if type(raw_origin) is not str:
            raise _GuestFailure("incompatible_browser")
        try:
            normalized_origin = _guest_https_origin(raw_origin).removesuffix("/")
        except _GuestFailure:
            raise _GuestFailure("incompatible_browser") from None
        if raw_origin != normalized_origin or raw_origin not in allowed_origins:
            raise _GuestFailure("incompatible_browser")
        observed_origins.append(raw_origin)
        entry_names: list[str] = []
        for entry in entries:
            owned_entry = cast("dict[str, Any]", entry)
            entry_names.append(
                _interactive_profile_text(
                    owned_entry.get("name"),
                    maximum_bytes=_INTERACTIVE_MAX_PROFILE_NAME_BYTES,
                )
            )
            _interactive_profile_text(
                owned_entry.get("value"),
                maximum_bytes=_INTERACTIVE_MAX_PROFILE_VALUE_BYTES,
            )
        if len(entry_names) != len(set(entry_names)) or (
            require_canonical_order and entry_names != sorted(entry_names)
        ):
            raise _GuestFailure("incompatible_browser")
    if len(observed_origins) != len(set(observed_origins)) or (
        require_canonical_order and observed_origins != sorted(observed_origins)
    ):
        raise _GuestFailure("incompatible_browser")
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _GuestFailure("incompatible_browser") from exc
    if len(encoded) > maximum_bytes:
        raise _GuestFailure("incompatible_browser")
    return json.loads(encoded)


def _interactive_profile_text(value: Any, *, maximum_bytes: int) -> str:
    if type(value) is not str or "\x00" in value:
        raise _GuestFailure("incompatible_browser")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _GuestFailure("incompatible_browser") from exc
    if len(encoded) > maximum_bytes:
        raise _GuestFailure("incompatible_browser")
    return value


def _interactive_profile_private_values(state: dict[str, Any]) -> tuple[str, ...]:
    """Extract non-empty cookie/storage names and values from bounded private state."""

    values: set[str] = set()
    cookies = state.get("cookies")
    origins = state.get("origins")
    if type(cookies) is not list or type(origins) is not list:
        raise _GuestFailure("incompatible_browser")
    for cookie in cookies:
        if (
            type(cookie) is not dict
            or type(cookie.get("name")) is not str
            or type(cookie.get("value")) is not str
        ):
            raise _GuestFailure("incompatible_browser")
        values.update(item for item in (cookie["name"], cookie["value"]) if item)
    for origin in origins:
        if type(origin) is not dict or type(origin.get("localStorage")) is not list:
            raise _GuestFailure("incompatible_browser")
        for entry in origin["localStorage"]:
            if (
                type(entry) is not dict
                or type(entry.get("name")) is not str
                or type(entry.get("value")) is not str
            ):
                raise _GuestFailure("incompatible_browser")
            values.update(item for item in (entry["name"], entry["value"]) if item)
    return tuple(sorted(values, key=lambda item: (-len(item), item)))


def _bounded_interactive_profile_private_values(
    prior_values: tuple[str, ...],
    current_values: tuple[str, ...],
    *,
    maximum_bytes: int,
) -> tuple[str, ...]:
    """Own one finite anti-reflection history or fail before publishing content."""

    protected_values = tuple(
        sorted(
            set(prior_values) | set(current_values),
            key=lambda item: (-len(item), item),
        )
    )
    if (
        len(protected_values) > _INTERACTIVE_MAX_PROFILE_PRIVATE_VALUES
        or sum(len(item.encode("utf-8")) for item in protected_values) > maximum_bytes
    ):
        raise _GuestFailure("resource_exhausted")
    return protected_values


def _interactive_upload_files(
    raw: Any,
    limits: _InteractiveLimits,
    *,
    require_content: bool = True,
) -> tuple[_InteractiveUploadFile, ...]:
    if type(raw) is not list or not 0 < len(raw) <= limits.max_upload_files:
        raise _GuestFailure("artifact_refused")
    files: list[_InteractiveUploadFile] = []
    total_size = 0
    seen_artifacts: set[str] = set()
    for item in raw:
        expected_fields = {
            "artifact_id",
            "content_sha256",
            "content_type",
            "filename",
            "size_bytes",
        }
        if require_content:
            expected_fields.add("content_base64")
        if type(item) is not dict or set(item) != expected_fields:
            raise _GuestFailure("artifact_refused")
        artifact_id = _interactive_identifier(item.get("artifact_id"))
        if artifact_id in seen_artifacts:
            raise _GuestFailure("artifact_refused")
        seen_artifacts.add(artifact_id)
        content_sha256 = item.get("content_sha256")
        if (
            type(content_sha256) is not str
            or len(content_sha256) != 64
            or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
        ):
            raise _GuestFailure("artifact_refused")
        filename = _interactive_upload_filename(
            item.get("filename"),
            maximum=limits.max_upload_filename_bytes,
        )
        content_type = item.get("content_type")
        if (
            type(content_type) is not str
            or not content_type
            or len(content_type.encode("ascii", errors="ignore")) != len(content_type)
            or len(content_type) > _INTERACTIVE_MAX_UPLOAD_CONTENT_TYPE_BYTES
            or re.fullmatch(r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*", content_type)
            is None
        ):
            raise _GuestFailure("artifact_refused")
        size_bytes = item.get("size_bytes")
        if (
            type(size_bytes) is not int
            or size_bytes < 0
            or size_bytes > limits.max_upload_file_bytes
        ):
            raise _GuestFailure("upload_too_large", allocation_disposition="live")
        total_size += size_bytes
        if total_size > limits.max_upload_total_bytes:
            raise _GuestFailure("upload_too_large", allocation_disposition="live")
        encoded = item.get("content_base64", "")
        if type(encoded) is not str or len(encoded) > 4 * ((limits.max_upload_file_bytes + 2) // 3):
            raise _GuestFailure("upload_too_large", allocation_disposition="live")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise _GuestFailure("artifact_refused") from exc
        if require_content and len(content) != size_bytes:
            raise _GuestFailure("artifact_refused")
        if require_content and hashlib.sha256(content).hexdigest() != content_sha256:
            raise _GuestFailure("artifact_refused")
        files.append(
            _InteractiveUploadFile(
                artifact_id=artifact_id,
                content_sha256=content_sha256,
                filename=filename,
                content_type=content_type,
                size_bytes=size_bytes,
                content=content,
            )
        )
    return tuple(files)


def _interactive_upload_filename(value: Any, *, maximum: int) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise _GuestFailure("artifact_refused")
    if value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise _GuestFailure("artifact_refused")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _GuestFailure("artifact_refused") from exc
    if len(encoded) > maximum:
        raise _GuestFailure("artifact_refused")
    return value


def _bounded_live_browser_private_values(state: Any, *, maximum_bytes: int) -> tuple[str, ...]:
    """Read-only anti-reflection inventory, never profile restore/checkpoint authority.

    A live temporary context may contain ordinary HTTP or domain cookies that
    are deliberately not eligible for durable profile import. Only names and
    values are read here; no origin/domain grants or transferable state emerge.
    """
    if type(state) is not dict:
        raise _GuestFailure("policy_denied")
    cookies, origins = state.get("cookies"), state.get("origins")
    if (
        type(cookies) is not list
        or type(origins) is not list
        or len(cookies) > _INTERACTIVE_MAX_PROFILE_COOKIES
        or len(origins) > _INTERACTIVE_MAX_PROFILE_ORIGINS
    ):
        raise _GuestFailure("policy_denied")
    entries = list(cookies)
    storage_count = 0
    for origin in origins:
        if type(origin) is not dict or type(origin.get("localStorage")) is not list:
            raise _GuestFailure("policy_denied")
        storage = origin["localStorage"]
        storage_count += len(storage)
        if storage_count > _INTERACTIVE_MAX_PROFILE_STORAGE_ENTRIES:
            raise _GuestFailure("policy_denied")
        entries.extend(storage)
    values = []
    used = 0
    for entry in entries:
        if type(entry) is not dict:
            raise _GuestFailure("policy_denied")
        owned_entry = cast("dict[str, Any]", entry)
        for field_name, limit in (
            ("name", _INTERACTIVE_MAX_PROFILE_NAME_BYTES),
            ("value", _INTERACTIVE_MAX_PROFILE_VALUE_BYTES),
        ):
            value = _interactive_profile_text(owned_entry.get(field_name), maximum_bytes=limit)
            used += len(value.encode("utf-8"))
            if used > maximum_bytes:
                raise _GuestFailure("resource_exhausted")
            if value:
                values.append(value)
    return _bounded_interactive_profile_private_values(
        (), tuple(values), maximum_bytes=maximum_bytes
    )


def _interactive_identifier(value: Any) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 128
        or _INTERACTIVE_SAFE_ID.fullmatch(value) is None
    ):
        raise _GuestFailure("incompatible_browser")
    return value


def _interactive_text(value: Any, *, maximum: int) -> str:
    if type(value) is not str or not value:
        raise _GuestFailure("incompatible_browser")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _GuestFailure("incompatible_browser") from exc
    if len(encoded) > maximum:
        raise _GuestFailure("incompatible_browser")
    return value


def _fetch_limits(request: _Request) -> _Limits:
    if request.operation != "fetch" or not isinstance(request.limits, _Limits):
        raise _GuestFailure("incompatible_browser")
    return request.limits


def _screenshot_limits(request: _Request) -> _ScreenshotLimits:
    if request.operation != "screenshot" or not isinstance(request.limits, _ScreenshotLimits):
        raise _GuestFailure("incompatible_browser")
    return request.limits


def _proxy_and_ca() -> tuple[str, Path]:
    upper_proxy = os.environ.get("HTTPS_PROXY")
    lower_proxy = os.environ.get("https_proxy")
    if upper_proxy is not None and lower_proxy is not None and upper_proxy != lower_proxy:
        raise _GuestFailure("capability_refused")
    proxy = upper_proxy or lower_proxy
    if not proxy:
        raise _GuestFailure("capability_refused")
    split = urlsplit(proxy)
    try:
        port = split.port
    except ValueError as exc:
        raise _GuestFailure("capability_refused") from exc
    if (
        split.scheme != "http"
        or split.hostname is None
        or port is None
        or split.username is not None
        or split.password is not None
        or split.path not in {"", "/"}
        or split.query
        or split.fragment
    ):
        raise _GuestFailure("capability_refused")
    ca_value = os.environ.get("SSL_CERT_FILE")
    if not ca_value:
        raise _GuestFailure("capability_refused")
    ca_path = Path(ca_value)
    try:
        ca_size = ca_path.stat().st_size
    except OSError as exc:
        raise _GuestFailure("capability_refused") from exc
    if not ca_path.is_absolute() or not ca_path.is_file() or ca_size <= 0 or ca_size > 64 * 1024:
        raise _GuestFailure("capability_refused")
    return proxy, ca_path


def _sanitize_environment(home: Path, *, proxy: str, ca_path: Path) -> None:
    preserved = {
        "HOME": str(home),
        "HTTPS_PROXY": proxy,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "PLAYWRIGHT_BROWSERS_PATH": _PLAYWRIGHT_BROWSERS_PATH,
        "SSL_CERT_FILE": str(ca_path),
        "TMPDIR": str(home / "tmp"),
        "XDG_CACHE_HOME": str(home / "cache"),
        "XDG_CONFIG_HOME": str(home / "config"),
    }
    for directory in ("tmp", "cache", "config"):
        (home / directory).mkdir(mode=0o700)
    os.environ.clear()
    os.environ.update(preserved)


def _temporary_profile_cleanup_command(
    home: Path,
    *,
    timeout_seconds: float,
) -> tuple[str, ...]:
    return (
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        _PROFILE_CLEANUP_ARGUMENT,
        str(home),
        str(timeout_seconds),
    )


def _temporary_profile_cleanup_main(raw_home: str, raw_timeout_seconds: str) -> int:
    """Delete one worker-owned profile after its parent closes the control pipe."""

    if not shutil.rmtree.avoids_symlink_attacks:
        return 2
    try:
        timeout_seconds = float(raw_timeout_seconds)
    except ValueError:
        return 2
    if (
        not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
        or timeout_seconds > _MAX_PROFILE_CLEANUP_RESERVE_SECONDS
    ):
        return 2
    home = Path(raw_home)
    try:
        root = _TEMPORARY_PROFILE_ROOT.resolve(strict=True)
        parent = home.parent.resolve(strict=True)
    except OSError:
        return 2
    if (
        not home.is_absolute()
        or parent != root
        or not home.name.startswith(_TEMPORARY_PROFILE_PREFIX)
        or len(home.name) <= len(_TEMPORARY_PROFILE_PREFIX)
    ):
        return 2
    try:
        metadata = home.lstat()
    except FileNotFoundError:
        return 0
    except OSError:
        return 1
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        return 2
    try:
        sys.stdout.buffer.write(b"1")
        sys.stdout.buffer.flush()
    except BrokenPipeError:
        pass
    # The parent never writes to this pipe. A byte indicates an invalid caller;
    # EOF means the parent explicitly released the profile or exited.
    if sys.stdin.buffer.read(1) != b"":
        return 2

    def cleanup_timed_out(_signum: int, _frame: Any) -> None:
        raise TimeoutError

    previous_handler = signal.signal(signal.SIGALRM, cleanup_timed_out)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 1
            signal.setitimer(signal.ITIMER_REAL, remaining)
            try:
                shutil.rmtree(home)
                return 0
            except FileNotFoundError:
                return 0
            except TimeoutError:
                return 1
            except OSError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return 1
                time.sleep(min(0.01, remaining))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _validate_temporary_profile_home(home: Path) -> None:
    try:
        root = _TEMPORARY_PROFILE_ROOT.resolve(strict=True)
        parent = home.parent.resolve(strict=True)
        metadata = home.lstat()
    except OSError as exc:
        raise _GuestFailure("cleanup_failed") from exc
    if (
        not home.is_absolute()
        or parent != root
        or not home.name.startswith(_TEMPORARY_PROFILE_PREFIX)
        or len(home.name) <= len(_TEMPORARY_PROFILE_PREFIX)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise _GuestFailure("cleanup_failed")


async def _start_temporary_profile_owner(
    *,
    timeout_seconds: float,
    startup_timeout_seconds: float | None = None,
    existing_home: Path | None = None,
) -> _TemporaryProfileOwner:
    if existing_home is None:
        try:
            home = Path(
                tempfile.mkdtemp(
                    prefix=_TEMPORARY_PROFILE_PREFIX,
                    dir=str(_TEMPORARY_PROFILE_ROOT),
                )
            )
        except OSError as exc:
            raise _GuestFailure("cleanup_failed") from exc
    else:
        home = existing_home
    # Until the cleanup owner starts, retain failed setup material for the
    # allocation's outer teardown. Recursive deletion here is unbounded, and
    # rejected ownership evidence must never authorize deletion.
    _validate_temporary_profile_home(home)
    cleanup_timeout_seconds = max(
        0.001,
        min(
            _MAX_PROFILE_CLEANUP_RESERVE_SECONDS,
            timeout_seconds * 0.75,
        ),
    )
    descriptors: list[int] = []
    try:
        control_read, control_write = os.pipe()
        descriptors.extend((control_read, control_write))
        ready_read, ready_write = os.pipe()
        descriptors.extend((ready_read, ready_write))
    except OSError as exc:
        for descriptor in descriptors:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        raise _GuestFailure("cleanup_failed") from exc
    try:
        command = _temporary_profile_cleanup_command(
            home,
            timeout_seconds=cleanup_timeout_seconds,
        )
        process = subprocess.Popen(
            command,
            stdin=control_read,
            stdout=ready_write,
            stderr=subprocess.DEVNULL,
            env={
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
            },
            close_fds=True,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        for descriptor in (control_read, control_write, ready_read, ready_write):
            with contextlib.suppress(OSError):
                os.close(descriptor)
        # No synchronous fallback is safe here: filesystem deletion is exactly
        # the operation that must not be allowed to block the worker deadline.
        raise _GuestFailure("cleanup_failed") from exc
    owner = _TemporaryProfileOwner(
        home=home,
        process=process,
        control_fd=control_write,
    )
    with contextlib.suppress(OSError):
        os.close(control_read)
    with contextlib.suppress(OSError):
        os.close(ready_write)
    read_failure: BaseException | None = None
    ready = b""
    try:
        os.set_blocking(ready_read, False)
        ready_timeout_seconds = (
            timeout_seconds if startup_timeout_seconds is None else startup_timeout_seconds
        )
        ready_deadline = asyncio.get_running_loop().time() + max(
            0.001,
            ready_timeout_seconds,
        )
        while True:
            try:
                ready = os.read(ready_read, 1)
            except BlockingIOError:
                remaining = ready_deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError(
                        "Temporary browser profile cleanup owner did not become ready."
                    ) from None
                await asyncio.sleep(min(0.001, remaining))
                continue
            break
    except BaseException as exc:
        read_failure = exc
    finally:
        with contextlib.suppress(OSError):
            os.close(ready_read)
    if read_failure is not None:
        await _raise_temporary_profile_start_failure(
            owner,
            primary=read_failure,
            timeout_seconds=timeout_seconds,
        )
    if ready != b"1":
        await _raise_temporary_profile_start_failure(
            owner,
            primary=RuntimeError("Temporary browser profile cleanup owner did not become ready."),
            timeout_seconds=timeout_seconds,
        )
    return owner


async def _wait_temporary_profile_owner(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> int | None:
    deadline = asyncio.get_running_loop().time() + max(0.0, timeout_seconds)
    while True:
        returncode = process.poll()
        if returncode is not None:
            return returncode
        remaining_seconds = deadline - asyncio.get_running_loop().time()
        if remaining_seconds <= 0:
            return None
        await asyncio.sleep(min(0.005, remaining_seconds))


async def _cleanup_temporary_profile_owner(
    owner: _TemporaryProfileOwner,
    *,
    timeout_seconds: float,
) -> tuple[BaseException, ...]:
    """Release and reap the independent profile owner within a finite budget."""

    errors: list[BaseException] = []
    if owner.control_fd is not None:
        control_fd = owner.control_fd
        owner.control_fd = None
        try:
            os.close(control_fd)
        except OSError as exc:
            errors.append(exc)
    timeout_seconds = max(0.0, timeout_seconds)
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    timed_out = False
    returncode: int | None = None
    try:
        # Reserve part of the same deadline for SIGKILL delivery and reaping.
        # Spending it all on graceful exit leaves only an immediate poll after
        # kill(), which can report failure while the child is still exiting.
        returncode = await _wait_temporary_profile_owner(
            owner.process,
            timeout_seconds=timeout_seconds * 0.75,
        )
        timed_out = returncode is None
        if timed_out:
            try:
                owner.process.kill()
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(exc)
            remaining_seconds = max(
                0.0,
                deadline - asyncio.get_running_loop().time(),
            )
            returncode = await _wait_temporary_profile_owner(
                owner.process,
                timeout_seconds=remaining_seconds,
            )
            if returncode is None:
                errors.append(
                    RuntimeError("Temporary browser profile cleanup owner could not be reaped.")
                )
        if timed_out:
            errors.insert(0, TimeoutError("Temporary browser profile cleanup timed out."))
        elif returncode != 0:
            errors.append(RuntimeError("Temporary browser profile cleanup failed."))
        elif owner.home.exists():
            errors.append(
                RuntimeError("Temporary browser profile cleanup did not remove its home.")
            )
    except asyncio.CancelledError:
        with contextlib.suppress(OSError):
            owner.process.kill()
        owner.process.poll()
        raise
    except Exception as exc:
        errors.append(exc)
    return tuple(errors)


async def _raise_temporary_profile_start_failure(
    owner: _TemporaryProfileOwner,
    *,
    primary: BaseException,
    timeout_seconds: float,
) -> Never:
    """Settle a spawned guardian before publishing failed startup."""

    cleanup_task = asyncio.create_task(
        _cleanup_temporary_profile_owner(owner, timeout_seconds=timeout_seconds)
    )
    cleanup_outcome = await _await_browser_cleanup_resisting_cancellation(cleanup_task)
    authoritative_failure = (
        primary if not isinstance(primary, Exception) else cleanup_outcome.cancellation
    )
    cause = _browser_cleanup_evidence(
        None if authoritative_failure is primary else primary,
        cleanup_outcome.errors,
    )
    if authoritative_failure is not None:
        if cause is None:
            raise authoritative_failure
        raise authoritative_failure from cause
    failure = _GuestFailure("cleanup_failed")
    if cause is None:  # pragma: no cover - non-cancellation primary invariant
        raise failure
    raise failure from cause


async def _kill_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
    await process.wait()


async def _install_browser_ca(home: Path, ca_path: Path, *, replace_existing: bool = False) -> None:
    certutil = Path("/usr/bin/certutil")
    if not certutil.is_file() or not os.access(certutil, os.X_OK):
        raise _GuestFailure("incompatible_browser")
    database = home / ".pki" / "nssdb"
    database.mkdir(parents=True, mode=0o700, exist_ok=replace_existing)
    commands = (
        (
            [str(certutil), "-D", "-d", f"sql:{database}", "-n", "Cayu session egress"]
            if replace_existing
            else [str(certutil), "-N", "--empty-password", "-d", f"sql:{database}"]
        ),
        [
            str(certutil),
            "-A",
            "-d",
            f"sql:{database}",
            "-n",
            "Cayu session egress",
            "-t",
            "C,,",
            "-i",
            str(ca_path),
        ],
    )
    for command in commands:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            async with asyncio.timeout(5):
                returncode = await process.wait()
        except asyncio.CancelledError:
            if process is not None:
                await _kill_process(process)
            raise
        except TimeoutError as exc:
            if process is not None:
                await _kill_process(process)
            raise _GuestFailure("timeout") from exc
        except OSError as exc:
            raise _GuestFailure("incompatible_browser") from exc
        if returncode != 0:
            raise _GuestFailure("incompatible_browser")


def _normalized_text(value: str, max_bytes: int, *, preserve_lines: bool) -> tuple[str, bool]:
    safe_parts: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if category == "Cs":
            safe_parts.append("\ufffd")
        elif preserve_lines and character in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029":
            safe_parts.append("\n")
        elif category == "Cc":
            safe_parts.append(" ")
        else:
            safe_parts.append(character)
    safe = "".join(safe_parts)
    if preserve_lines:
        normalized = "\n".join(
            line for line in (" ".join(part.split()) for part in safe.splitlines()) if line
        )
    else:
        normalized = " ".join(safe.split())
    encoded = normalized.encode("utf-8")
    if len(encoded) <= max_bytes:
        return normalized, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _normalized_accessibility_text(value: str, max_bytes: int) -> tuple[str, bool]:
    """Bound an ARIA snapshot while preserving its structural indentation."""

    safe_parts: list[str] = []
    for character in value:
        category = unicodedata.category(character)
        if category == "Cs":
            safe_parts.append("\ufffd")
        elif character in "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029":
            safe_parts.append("\n")
        elif category == "Cc":
            safe_parts.append(" ")
        else:
            safe_parts.append(character)
    lines: list[str] = []
    for raw_line in "".join(safe_parts).splitlines():
        stripped = raw_line.lstrip(" ")
        if not stripped:
            continue
        indent = min(len(raw_line) - len(stripped), _MAX_ACCESSIBILITY_INDENT)
        lines.append((" " * indent) + " ".join(stripped.split()))
    normalized = "\n".join(lines)
    encoded = normalized.encode("utf-8")
    if len(encoded) <= max_bytes:
        return normalized, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


async def _extract_page_representation(
    page: Any,
    cdp: Any,
    request: _Request,
    *,
    operation_timeout_ms: int,
) -> tuple[str | None, str, str, tuple[str, ...]]:
    """Select and bound the canonical model-facing page representation."""

    limits = _fetch_limits(request)
    frame_identities = await _frame_identities(cdp)
    frames = _playwright_frames_for_identities(page, frame_identities)
    evidence_membership = await _frame_evidence_membership(cdp, frame_identities)
    remaining_nodes = limits.max_dom_nodes
    projections: list[_FrameProjection] = []
    for identity in frame_identities:
        projection = await _isolated_frame_projection(
            cdp,
            identity,
            limits,
            max_dom_nodes=remaining_nodes,
        )
        remaining_nodes -= projection.node_count
        projections.append(projection)

    included_indexes = [index for index, included in enumerate(evidence_membership) if included]
    included_projections = [projections[index] for index in included_indexes]
    representation = (
        "accessibility"
        if any(projection.semantic_structure for projection in included_projections)
        else "text"
    )
    frame_contents: list[str] = []
    content_truncated = False
    if representation == "accessibility":
        for index in included_indexes:
            frame = frames[index]
            body = frame.locator("body")
            accessibility_content = await body.aria_snapshot(
                timeout=operation_timeout_ms,
                depth=_ACCESSIBILITY_SNAPSHOT_DEPTH,
                mode="default",
                boxes=False,
            )
            accessibility_depth_probe = await body.aria_snapshot(
                timeout=operation_timeout_ms,
                depth=_ACCESSIBILITY_SNAPSHOT_DEPTH + 1,
                mode="default",
                boxes=False,
            )
            if type(accessibility_content) is not str:
                raise _GuestFailure("browser_crash")
            if type(accessibility_depth_probe) is not str:
                raise _GuestFailure("browser_crash")
            normalized, normalized_truncated = _normalized_accessibility_text(
                accessibility_content,
                limits.max_content_bytes,
            )
            frame_contents.append(normalized)
            content_truncated = (
                content_truncated
                or normalized_truncated
                or accessibility_depth_probe != accessibility_content
            )
    else:
        for projection in included_projections:
            frame_contents.append(projection.text)
            content_truncated = content_truncated or projection.text_truncated

    content, aggregate_truncated = _aggregate_frame_content(
        included_indexes,
        included_projections,
        frame_contents,
        max_bytes=limits.max_content_bytes,
    )
    content_truncated = content_truncated or aggregate_truncated
    stable_identities = await _frame_identities(cdp)
    if stable_identities != frame_identities:
        raise _GuestFailure("fetch_failed")
    stable_frames = _playwright_frames_for_identities(page, stable_identities)
    if (
        any(stable is not original for stable, original in zip(stable_frames, frames, strict=True))
        or await _frame_evidence_membership(cdp, stable_identities) != evidence_membership
    ):
        raise _GuestFailure("fetch_failed")

    title = projections[0].title
    title_truncated = any(projection.title_truncated for projection in included_projections)
    truncation_reasons: list[str] = []
    if title_truncated:
        truncation_reasons.append("title")
    if content_truncated:
        truncation_reasons.append("content")
    return title, representation, content, tuple(truncation_reasons)


async def _isolated_frame_projection(
    cdp: Any,
    identity: _FrameIdentity,
    limits: _Limits,
    *,
    max_dom_nodes: int,
) -> _FrameProjection:
    execution_context_id = await _create_isolated_world(cdp, identity.frame_id)
    encoded_limits = json.dumps(
        {
            "content": limits.max_content_bytes,
            "nodes": max_dom_nodes,
            "title": _MAX_TITLE_BYTES,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    projection = await cdp.send(
        "Runtime.evaluate",
        {
            "contextId": execution_context_id,
            "expression": """(() => {
            const limits = __CAYU_INSPECTION_LIMITS__;
            const body = document.body;
            const semanticTags = new Set([
                "a", "area", "button", "details", "dl", "form", "input",
                "nav", "select", "summary", "table", "textarea",
            ]);
            let nodeCount = body ? 1 : 0;
            let semanticStructure = false;
            if (body) {
                const pending = [body];
                scan: while (pending.length > 0) {
                    const node = pending.pop();
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        const element = node;
                        const tag = element.localName;
                        semanticStructure = semanticStructure
                            || semanticTags.has(tag)
                            || element.hasAttribute("role")
                            || element.hasAttribute("aria-label")
                            || element.hasAttribute("aria-labelledby")
                            || element.hasAttribute("aria-describedby");
                        if (element.shadowRoot) {
                            nodeCount += 1;
                            if (nodeCount > limits.nodes) {
                                break scan;
                            }
                            pending.push(element.shadowRoot);
                        }
                    }
                    for (let child = node.lastChild; child; child = child.previousSibling) {
                        nodeCount += 1;
                        if (nodeCount > limits.nodes) {
                            break scan;
                        }
                        pending.push(child);
                    }
                }
            }
            const value = body ? body.innerText : "";
            const title = document.title;
            return {
                text: value.slice(0, limits.content + 1),
                truncated: value.length > limits.content,
                title: title.slice(0, limits.title + 1),
                title_truncated: title.length > limits.title,
                node_count: nodeCount,
                node_limit_exceeded: nodeCount > limits.nodes,
                semantic_structure: semanticStructure,
            };
        })()""".replace("__CAYU_INSPECTION_LIMITS__", encoded_limits),
            "returnByValue": True,
            "awaitPromise": False,
            "userGesture": False,
        },
    )
    if (
        type(projection) is not dict
        or projection.get("exceptionDetails") is not None
        or type(projection.get("result")) is not dict
        or projection["result"].get("type") != "object"
        or type(projection["result"].get("value")) is not dict
    ):
        raise _GuestFailure("browser_crash")
    extracted = projection["result"]["value"]
    if (
        set(extracted)
        != {
            "node_limit_exceeded",
            "node_count",
            "semantic_structure",
            "text",
            "title",
            "title_truncated",
            "truncated",
        }
        or type(extracted.get("text")) is not str
        or type(extracted.get("truncated")) is not bool
        or type(extracted.get("title")) is not str
        or type(extracted.get("title_truncated")) is not bool
        or type(extracted.get("node_count")) is not int
        or extracted["node_count"] < 0
        or type(extracted.get("node_limit_exceeded")) is not bool
        or type(extracted.get("semantic_structure")) is not bool
    ):
        raise _GuestFailure("browser_crash")
    if extracted["node_limit_exceeded"]:
        raise _GuestFailure("oversized_response")
    if extracted["node_count"] > max_dom_nodes:
        raise _GuestFailure("browser_crash")

    text, text_truncated = _normalized_text(
        extracted["text"],
        limits.max_content_bytes,
        preserve_lines=True,
    )
    text_truncated = text_truncated or extracted["truncated"]
    title, title_truncated = _normalized_text(
        extracted["title"],
        _MAX_TITLE_BYTES,
        preserve_lines=False,
    )
    title_truncated = title_truncated or extracted["title_truncated"]
    return _FrameProjection(
        identity=identity,
        title=title or None,
        text=text,
        title_truncated=title_truncated,
        text_truncated=text_truncated,
        semantic_structure=extracted["semantic_structure"],
        node_count=extracted["node_count"],
    )


async def _create_isolated_world(cdp: Any, frame_id: str) -> int:
    isolated_world = await cdp.send(
        "Page.createIsolatedWorld",
        {
            "frameId": frame_id,
            "worldName": _BROWSER_INSPECTION_WORLD,
            "grantUniveralAccess": False,
        },
    )
    if (
        type(isolated_world) is not dict
        or type(isolated_world.get("executionContextId")) is not int
        or isolated_world["executionContextId"] <= 0
    ):
        raise _GuestFailure("browser_crash")
    return isolated_world["executionContextId"]


async def _isolated_page_title(cdp: Any, frame_id: str) -> tuple[str | None, bool]:
    execution_context_id = await _create_isolated_world(cdp, frame_id)
    projection = await cdp.send(
        "Runtime.evaluate",
        {
            "contextId": execution_context_id,
            "expression": """(() => {
            const title = document.title;
            return {
                title: title.slice(0, __CAYU_TITLE_LIMIT__ + 1),
                title_truncated: title.length > __CAYU_TITLE_LIMIT__,
            };
        })()""".replace("__CAYU_TITLE_LIMIT__", str(_MAX_TITLE_BYTES)),
            "returnByValue": True,
            "awaitPromise": False,
            "userGesture": False,
        },
    )
    if (
        type(projection) is not dict
        or projection.get("exceptionDetails") is not None
        or type(projection.get("result")) is not dict
        or projection["result"].get("type") != "object"
        or type(projection["result"].get("value")) is not dict
    ):
        raise _GuestFailure("browser_crash")
    extracted = projection["result"]["value"]
    if (
        set(extracted) != {"title", "title_truncated"}
        or type(extracted.get("title")) is not str
        or type(extracted.get("title_truncated")) is not bool
    ):
        raise _GuestFailure("browser_crash")
    title, title_truncated = _normalized_text(
        extracted["title"],
        _MAX_TITLE_BYTES,
        preserve_lines=False,
    )
    title_truncated = title_truncated or extracted["title_truncated"]
    return title or None, title_truncated


async def _frame_identities(cdp: Any) -> tuple[_FrameIdentity, ...]:
    frame_tree = await cdp.send("Page.getFrameTree")
    if type(frame_tree) is not dict or type(frame_tree.get("frameTree")) is not dict:
        raise _GuestFailure("browser_crash")
    pending: list[tuple[dict[str, Any], int | None]] = [(frame_tree["frameTree"], None)]
    identities: list[_FrameIdentity] = []
    seen_ids: set[str] = set()
    while pending:
        tree, parent_index = pending.pop()
        if type(tree.get("frame")) is not dict:
            raise _GuestFailure("browser_crash")
        frame = tree["frame"]
        frame_id = frame.get("id")
        loader_id = frame.get("loaderId")
        url = frame.get("url")
        mime_type = frame.get("mimeType")
        if (
            type(frame_id) is not str
            or not 0 < len(frame_id) <= 256
            or frame_id in seen_ids
            or type(loader_id) is not str
            or not 0 < len(loader_id) <= 256
            or type(url) is not str
            or not 0 < len(url) <= _MAX_URL_LENGTH
            or type(mime_type) is not str
        ):
            raise _GuestFailure("browser_crash")
        if not _browser_request_is_admissible(url):
            raise _GuestFailure("destination_denied")
        if mime_type not in _HTML_CONTENT_TYPES | _TEXT_CONTENT_TYPES:
            raise _GuestFailure("unsupported_content")
        if parent_index is not None:
            parent_id = frame.get("parentId")
            if parent_id != identities[parent_index].frame_id:
                raise _GuestFailure("browser_crash")
        seen_ids.add(frame_id)
        current_index = len(identities)
        identities.append(
            _FrameIdentity(
                frame_id=frame_id,
                loader_id=loader_id,
                url=url,
                mime_type=mime_type,
                parent_index=parent_index,
            )
        )
        if len(identities) > _MAX_FRAME_DOCUMENTS:
            raise _GuestFailure("oversized_response")
        raw_children = tree.get("childFrames", [])
        if type(raw_children) is not list:
            raise _GuestFailure("browser_crash")
        children: list[dict[str, Any]] = []
        for child in raw_children:
            if type(child) is not dict:
                raise _GuestFailure("browser_crash")
            children.append(child)
        pending.extend((child, current_index) for child in reversed(children))
    return tuple(identities)


async def _frame_evidence_membership(
    cdp: Any,
    identities: tuple[_FrameIdentity, ...],
) -> tuple[bool, ...]:
    """Return frame documents represented by their owner in Chromium's AX tree."""

    if not identities or identities[0].parent_index is not None:
        raise _GuestFailure("browser_crash")
    included = [True]
    for index, identity in enumerate(identities[1:], start=1):
        parent_index = identity.parent_index
        if parent_index is None or parent_index >= index:
            raise _GuestFailure("browser_crash")
        owner = await cdp.send("DOM.getFrameOwner", {"frameId": identity.frame_id})
        if (
            type(owner) is not dict
            or type(owner.get("backendNodeId")) is not int
            or owner["backendNodeId"] <= 0
        ):
            raise _GuestFailure("browser_crash")
        backend_node_id = owner["backendNodeId"]
        accessibility = await cdp.send(
            "Accessibility.getPartialAXTree",
            {
                "backendNodeId": backend_node_id,
                "fetchRelatives": False,
            },
        )
        if type(accessibility) is not dict or type(accessibility.get("nodes")) is not list:
            raise _GuestFailure("browser_crash")
        matching_nodes = [
            node
            for node in accessibility["nodes"]
            if type(node) is dict and node.get("backendDOMNodeId") == backend_node_id
        ]
        if len(matching_nodes) != 1 or type(matching_nodes[0].get("ignored")) is not bool:
            raise _GuestFailure("browser_crash")
        included.append(included[parent_index] and not matching_nodes[0]["ignored"])
    return tuple(included)


def _playwright_frames_for_identities(
    page: Any,
    identities: tuple[_FrameIdentity, ...],
) -> tuple[Any, ...]:
    pending: list[tuple[Any, int | None]] = [(page.main_frame, None)]
    frames: list[Any] = []
    seen: set[int] = set()
    while pending:
        frame, parent_index = pending.pop()
        if id(frame) in seen or len(frames) >= _MAX_FRAME_DOCUMENTS:
            raise _GuestFailure("fetch_failed")
        seen.add(id(frame))
        current_index = len(frames)
        if current_index >= len(identities):
            raise _GuestFailure("fetch_failed")
        identity = identities[current_index]
        if frame.url != identity.url or parent_index != identity.parent_index:
            raise _GuestFailure("fetch_failed")
        frames.append(frame)
        children = frame.child_frames
        if type(children) is not list:
            raise _GuestFailure("browser_crash")
        pending.extend((child, current_index) for child in reversed(children))
    if len(frames) != len(identities):
        raise _GuestFailure("fetch_failed")
    return tuple(frames)


def _aggregate_frame_content(
    frame_indexes: list[int],
    projections: list[_FrameProjection],
    contents: list[str],
    *,
    max_bytes: int,
) -> tuple[str, bool]:
    if (
        len(frame_indexes) != len(projections)
        or len(projections) != len(contents)
        or not projections
        or frame_indexes[0] != 0
        or any(index < 0 for index in frame_indexes)
        or frame_indexes != sorted(set(frame_indexes))
    ):
        raise _GuestFailure("browser_crash")
    if len(projections) == 1:
        return contents[0], False
    sections: list[str] = []
    output_indexes = {
        source_index: output_index for output_index, source_index in enumerate(frame_indexes)
    }
    for source_index, projection, content in zip(
        frame_indexes,
        projections,
        contents,
        strict=True,
    ):
        output_index = output_indexes[source_index]
        url, _ = _normalized_text(
            projection.identity.url,
            4 * _MAX_URL_LENGTH,
            preserve_lines=False,
        )
        lines = [
            f"[{'Main frame' if output_index == 0 else f'Frame {output_index}'}]",
            f"URL: {url}",
        ]
        if projection.identity.parent_index is not None:
            parent_output_index = output_indexes.get(projection.identity.parent_index)
            if parent_output_index is None:
                raise _GuestFailure("browser_crash")
            lines.append(f"Parent frame: {parent_output_index}")
        if projection.title is not None:
            lines.append(f"Title: {projection.title}")
        if content:
            lines.append(content)
        sections.append("\n".join(lines))
    joined = "\n\n".join(sections)
    encoded = joined.encode("utf-8")
    if len(encoded) <= max_bytes:
        return joined, False
    return encoded[:max_bytes].decode("utf-8", errors="ignore"), True


def _browser_request_is_admissible(url: Any) -> bool:
    if type(url) is not str or not 0 < len(url) <= _MAX_URL_LENGTH:
        return False
    split = urlsplit(url)
    scheme = split.scheme.lower()
    if scheme in {"data", "blob"}:
        return True
    if scheme == "about":
        return url in {"about:blank", "about:srcdoc"}
    if scheme != "https":
        return False
    try:
        port = split.port
    except ValueError:
        return False
    return (
        split.hostname is not None
        and split.username is None
        and split.password is None
        and port in {None, 443}
    )


def _interactive_origin(url: Any) -> str | None:
    if type(url) is not str:
        return None
    try:
        split = urlsplit(url)
        port = split.port
    except (TypeError, ValueError):
        return None
    if (
        split.scheme.lower() != "https"
        or split.hostname is None
        or split.username is not None
        or split.password is not None
        or port not in {None, 443}
    ):
        return None
    try:
        hostname = split.hostname.rstrip(".").encode("idna").decode("ascii").lower()
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    except UnicodeError:
        return None
    else:
        return None
    if not hostname:
        return None
    return f"https://{hostname}/"


async def _capture_page_screenshot(
    page: Any,
    cdp: Any,
    request: _Request,
    *,
    main_frame_id: str,
) -> tuple[str, str | None, bool, int, int, bytes]:
    limits = _screenshot_limits(request)
    # Keep the document timeline fixed across the trusted measurement and the
    # later Playwright capture. Page-authored JavaScript is already disabled.
    freeze_result = await cdp.send("Animation.setPlaybackRate", {"playbackRate": 0})
    if type(freeze_result) is not dict:
        raise _GuestFailure("screenshot_failed")
    document_identity = await _screenshot_document_identity(
        cdp,
        expected_frame_id=main_frame_id,
    )
    title, title_truncated = await _isolated_page_title(cdp, main_frame_id)
    if request.full_page:
        expected_width, expected_height = _screenshot_layout_dimensions(
            await cdp.send("Page.getLayoutMetrics")
        )
    else:
        expected_width = limits.viewport_width
        expected_height = limits.viewport_height
    if (
        expected_width <= 0
        or expected_height <= 0
        or expected_width > limits.max_page_width
        or expected_height > limits.max_page_height
        or expected_width * expected_height > limits.max_page_pixels
    ):
        raise _GuestFailure("oversized_page")
    screenshot_options: dict[str, Any] = {
        "type": "png",
        "full_page": request.full_page,
        # Avoid Playwright's temporary inline caret styling. Page CSS can react
        # to that DOM mutation and change layout after the validated measurement.
        "caret": "initial",
    }
    if request.full_page:
        # Playwright recomputes full-page dimensions immediately before capture.
        # Supplying the already-validated rectangle makes that later measurement
        # unable to expand the bitmap if CSS layout moves between the two steps.
        screenshot_options["clip"] = {
            "x": 0,
            "y": 0,
            "width": expected_width,
            "height": expected_height,
        }
    screenshot = await page.screenshot(**screenshot_options)
    if type(screenshot) is not bytes:
        raise _GuestFailure("screenshot_failed")
    if len(screenshot) > limits.max_screenshot_bytes:
        raise _GuestFailure("oversized_screenshot")
    width, height = _png_header_dimensions(screenshot)
    if (
        width > limits.max_page_width
        or height > limits.max_page_height
        or width * height > limits.max_page_pixels
    ):
        raise _GuestFailure("oversized_page")
    if (width, height) != (expected_width, expected_height):
        raise _GuestFailure("screenshot_failed")
    if request.full_page:
        stable_width, stable_height = _screenshot_layout_dimensions(
            await cdp.send("Page.getLayoutMetrics")
        )
        if (stable_width, stable_height) != (expected_width, expected_height):
            if (
                stable_width > limits.max_page_width
                or stable_height > limits.max_page_height
                or stable_width * stable_height > limits.max_page_pixels
            ):
                raise _GuestFailure("oversized_page")
            raise _GuestFailure("screenshot_failed")
    stable_document_identity = await _screenshot_document_identity(
        cdp,
        expected_frame_id=main_frame_id,
    )
    if stable_document_identity != document_identity:
        raise _GuestFailure("screenshot_failed")
    return document_identity.url, title, title_truncated, width, height, screenshot


async def _screenshot_document_identity(
    cdp: Any,
    *,
    expected_frame_id: str,
) -> _ScreenshotDocumentIdentity:
    frame_tree = await cdp.send("Page.getFrameTree")
    if type(frame_tree) is not dict or type(frame_tree.get("frameTree")) is not dict:
        raise _GuestFailure("screenshot_failed")
    root = frame_tree["frameTree"]
    if type(root.get("frame")) is not dict:
        raise _GuestFailure("screenshot_failed")
    frame = root["frame"]
    frame_id = frame.get("id")
    loader_id = frame.get("loaderId")
    url = frame.get("url")
    if (
        type(expected_frame_id) is not str
        or not 0 < len(expected_frame_id) <= 256
        or type(frame_id) is not str
        or frame_id != expected_frame_id
        or type(loader_id) is not str
        or not 0 < len(loader_id) <= 256
        or type(url) is not str
        or not 0 < len(url) <= _MAX_URL_LENGTH
    ):
        raise _GuestFailure("screenshot_failed")
    return _ScreenshotDocumentIdentity(
        frame_id=frame_id,
        loader_id=loader_id,
        url=url,
    )


def _screenshot_layout_dimensions(metrics: Any) -> tuple[int, int]:
    if type(metrics) is not dict or type(metrics.get("cssContentSize")) is not dict:
        raise _GuestFailure("screenshot_failed")
    css_content_size = metrics["cssContentSize"]
    raw_width = css_content_size.get("width")
    raw_height = css_content_size.get("height")
    if (
        isinstance(raw_width, bool)
        or not isinstance(raw_width, (int, float))
        or not math.isfinite(raw_width)
        or isinstance(raw_height, bool)
        or not isinstance(raw_height, (int, float))
        or not math.isfinite(raw_height)
    ):
        raise _GuestFailure("screenshot_failed")
    return math.ceil(raw_width), math.ceil(raw_height)


def _screenshot_playwright_failure(
    *,
    state: _PageState,
    browser: Any,
    page: Any,
) -> _GuestFailure:
    if state.browser_crashed:
        return _GuestFailure("browser_crash")
    try:
        if not browser.is_connected() or page.is_closed():
            return _GuestFailure("browser_crash")
    except Exception:
        # Failure to inspect ownership after a Playwright transport error is
        # itself evidence that the browser boundary is no longer trustworthy.
        return _GuestFailure("browser_crash")
    return _GuestFailure("screenshot_failed")


def _png_header_dimensions(content: bytes) -> tuple[int, int]:
    if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n" or content[12:16] != b"IHDR":
        raise _GuestFailure("screenshot_failed")
    width = int.from_bytes(content[16:20], "big")
    height = int.from_bytes(content[20:24], "big")
    if width <= 0 or height <= 0:
        raise _GuestFailure("screenshot_failed")
    return width, height


async def _fetch_with_browser(
    request: _Request,
    proxy: str,
    *,
    state: _PageState,
    operation_timeout_ms: int,
    cleanup_timeout_seconds: float,
    cleanup_deadline: float,
) -> dict[str, Any]:
    try:
        from playwright.async_api import Error as PlaywrightError
        from playwright.async_api import TimeoutError as PlaywrightTimeoutError
        from playwright.async_api import async_playwright
    except (ImportError, OSError) as exc:
        raise _GuestFailure("browser_unavailable") from exc

    playwright = None
    browser = None
    context = None
    page = None
    response_observed = None
    unexpected_page_observed = None
    navigation_task: asyncio.Task[Any] | None = None
    violation_task: asyncio.Task[bool] | None = None
    violation_observed = asyncio.Event()
    redirects = state.redirects
    launched = False
    primary: BaseException | None = None
    success_projection: (
        tuple[
            str,
            str | None,
            str,
            str,
            tuple[str, ...],
        ]
        | None
    ) = None
    screenshot_projection: tuple[str, str | None, bool, int, int, bytes] | None = None
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            chromium_sandbox=True,
            proxy={"server": proxy},
            args=[
                "--disable-background-networking",
                "--disable-component-update",
                "--disable-default-apps",
                "--disable-dev-shm-usage",
                "--disable-domain-reliability",
                "--disable-features=AutofillServerCommunication,MediaRouter",
                "--disable-quic",
                "--disable-sync",
                "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                "--metrics-recording-only",
                "--no-first-run",
                "--password-store=basic",
                "--use-mock-keychain",
                "--webrtc-ip-handling-policy=disable_non_proxied_udp",
            ],
            timeout=operation_timeout_ms,
        )
        launched = True
        context_options: dict[str, Any] = {
            "accept_downloads": False,
            "ignore_https_errors": False,
            "java_script_enabled": True,
            "service_workers": "block",
        }
        if request.operation == "screenshot":
            screenshot_limits = _screenshot_limits(request)
            context_options.update(
                viewport={
                    "width": screenshot_limits.viewport_width,
                    "height": screenshot_limits.viewport_height,
                },
                device_scale_factor=1,
            )
        context = await browser.new_context(
            **context_options,
        )
        page = await context.new_page()
        page.set_default_timeout(operation_timeout_ms)
        page.set_default_navigation_timeout(operation_timeout_ms)

        def browser_crashed(*_args: Any) -> None:
            state.browser_crashed = True

        browser.on("disconnected", browser_crashed)
        page.on("crash", browser_crashed)

        def abort_page() -> None:
            violation_observed.set()

        def unexpected_page_observed(_unexpected_page: Any) -> None:
            state.response_inspection_failed = True
            abort_page()

        context.on("page", unexpected_page_observed)

        async def route_request(route: Any, browser_request: Any) -> None:
            state.request_count += 1
            is_navigation_request = browser_request.is_navigation_request()
            try:
                request_frame = browser_request.frame if is_navigation_request else None
            except Exception:
                state.response_inspection_failed = True
                with contextlib.suppress(Exception):
                    await route.abort("blockedbyclient")
                abort_page()
                return
            is_main_navigation = is_navigation_request and request_frame == page.main_frame
            if state.access_evidence is not None:
                await route.abort("blockedbyclient")
                abort_page()
                return
            if state.request_count > state.max_requests:
                state.limit_exceeded = True
                await route.abort("blockedbyclient")
                abort_page()
                return
            if not _browser_request_is_admissible(browser_request.url):
                is_redirected_main_navigation = (
                    is_main_navigation and browser_request.redirected_from is not None
                )
                _record_page_denial(
                    state,
                    ("redirect_denied" if is_redirected_main_navigation else "destination_denied"),
                )
                await route.abort("blockedbyclient")
                abort_page()
                return
            if is_main_navigation and urlsplit(browser_request.url).scheme.lower() == "https":
                with contextlib.suppress(_GuestFailure):
                    state.effective_origin = _guest_https_origin(browser_request.url)
            await route.continue_()

        await context.route("**/*", route_request)

        cdp = await context.new_cdp_session(page)
        await cdp.send("Network.enable")
        frame_tree = await cdp.send("Page.getFrameTree")
        main_frame_id = frame_tree["frameTree"]["frame"]["id"]
        if type(main_frame_id) is not str:
            raise _GuestFailure("browser_crash")
        main_document_request_ids: set[str] = set()
        redirected_main_request_ids: set[str] = set()
        await cdp.send(
            "Fetch.enable",
            {
                "patterns": [
                    {
                        "urlPattern": "*",
                        "resourceType": "Document",
                        "requestStage": "Response",
                    }
                ]
            },
        )

        def request_will_be_sent(params: dict[str, Any]) -> None:
            try:
                if params.get("type") != "Document":
                    return
                request_id = params.get("requestId")
                frame_id = params.get("frameId")
                if type(request_id) is not str or type(frame_id) is not str:
                    raise TypeError("Missing browser navigation identity.")
                redirect_response = params.get("redirectResponse")
                if redirect_response is not None and type(redirect_response) is not dict:
                    raise TypeError("Malformed browser redirect evidence.")
                if frame_id == main_frame_id:
                    if redirect_response is not None or request_id in main_document_request_ids:
                        redirected_main_request_ids.add(request_id)
                    main_document_request_ids.add(request_id)
            except Exception:
                state.response_inspection_failed = True
                abort_page()

        cdp.on("Network.requestWillBeSent", request_will_be_sent)

        async def response_paused(params: dict[str, Any]) -> None:
            request_id = params.get("requestId")
            try:
                if type(request_id) is not str:
                    raise TypeError("Missing paused browser response identity.")
                frame_id = params.get("frameId")
                status_code = params.get("responseStatusCode")
                raw_headers = params.get("responseHeaders")
                if (
                    params.get("resourceType") == "Document"
                    and frame_id == main_frame_id
                    and type(status_code) is int
                    and type(raw_headers) is list
                ):
                    headers = {
                        item["name"]: item["value"]
                        for item in raw_headers
                        if type(item) is dict
                        and type(item.get("name")) is str
                        and type(item.get("value")) is str
                    }
                    broker_code = next(
                        (
                            value
                            for key, value in headers.items()
                            if key.lower() == _BROKER_ERROR_HEADER and value
                        ),
                        None,
                    )
                    if broker_code is not None:
                        network_id = params.get("networkId")
                        _record_page_denial(
                            state,
                            (
                                "redirect_denied"
                                if broker_code == "destination_denied"
                                and network_id in redirected_main_request_ids
                                else broker_code
                            ),
                        )
                        abort_page()
                        await cdp.send(
                            "Fetch.failRequest",
                            {"requestId": request_id, "errorReason": "BlockedByClient"},
                        )
                        return
                    response_url = params.get("request", {}).get("url")
                    if type(response_url) is not str:
                        raise TypeError("Missing paused browser response URL.")
                    access = _guest_http_access(
                        response_url,
                        status_code,
                        headers,
                        source="browser_response",
                    )
                    if access is not None:
                        if state.access_evidence is None:
                            state.access_evidence = access
                            state.effective_origin = _guest_https_origin(response_url)
                        abort_page()
                        await cdp.send(
                            "Fetch.failRequest",
                            {"requestId": request_id, "errorReason": "BlockedByClient"},
                        )
                        return
                await cdp.send("Fetch.continueResponse", {"requestId": request_id})
            except Exception:
                state.response_inspection_failed = True
                abort_page()
                if type(request_id) is str:
                    with contextlib.suppress(Exception):
                        await cdp.send(
                            "Fetch.failRequest",
                            {"requestId": request_id, "errorReason": "BlockedByClient"},
                        )

        cdp.on("Fetch.requestPaused", response_paused)

        def response_extra_info(params: dict[str, Any]) -> None:
            try:
                request_id = params.get("requestId")
                if type(request_id) is not str:
                    raise TypeError("Missing browser response identity.")
                headers = params.get("headers")
                if type(headers) is not dict:
                    raise TypeError("Missing raw browser response headers.")
                broker_code = next(
                    (
                        value
                        for key, value in headers.items()
                        if type(key) is str
                        and key.lower() == _BROKER_ERROR_HEADER
                        and type(value) is str
                        and value
                    ),
                    None,
                )
                if broker_code is not None:
                    _record_page_denial(
                        state,
                        (
                            "redirect_denied"
                            if broker_code == "destination_denied"
                            and request_id in redirected_main_request_ids
                            else broker_code
                        ),
                    )
                    abort_page()
            except Exception:
                state.response_inspection_failed = True
                abort_page()

        cdp.on("Network.responseReceivedExtraInfo", response_extra_info)

        def data_received(params: dict[str, Any]) -> None:
            length = params.get("encodedDataLength")
            if isinstance(length, (int, float)) and not isinstance(length, bool):
                state.response_bytes += max(0, math.ceil(float(length)))
                if state.response_bytes > state.max_response_bytes:
                    state.limit_exceeded = True
                    abort_page()

        cdp.on("Network.dataReceived", data_received)

        def response_observed(response: Any) -> None:
            try:
                headers = response.headers
                if (
                    response.request.is_navigation_request()
                    and response.request.frame == page.main_frame
                ):
                    access = _guest_http_access(
                        response.url,
                        response.status,
                        headers,
                        source="browser_response",
                    )
                    if access is not None:
                        if state.access_evidence is None:
                            state.access_evidence = access
                            state.effective_origin = _guest_https_origin(response.url)
                        abort_page()
                        return
                broker_code = headers.get(_BROKER_ERROR_HEADER)
                if broker_code:
                    is_redirected_main_navigation = (
                        response.request.is_navigation_request()
                        and response.request.frame == page.main_frame
                        and response.request.redirected_from is not None
                    )
                    _record_page_denial(
                        state,
                        (
                            "redirect_denied"
                            if broker_code == "destination_denied" and is_redirected_main_navigation
                            else broker_code
                        ),
                    )
                    abort_page()
                if (
                    response.status in _REDIRECT_STATUS_CODES
                    and response.request.is_navigation_request()
                    and response.request.frame == page.main_frame
                ):
                    location = headers.get("location")
                    if location:
                        redirects.append(
                            {
                                "status_code": response.status,
                                "from_url": response.url,
                                "to_url": urljoin(response.url, location),
                            }
                        )
                        if len(redirects) > state.max_redirects:
                            _record_page_denial(state, "redirect_denied")
                            abort_page()
            except Exception:
                state.response_inspection_failed = True
                abort_page()

        page.on("response", response_observed)
        navigation_task = asyncio.create_task(
            page.goto(
                request.url,
                wait_until="load",
                timeout=operation_timeout_ms,
            )
        )
        violation_task = asyncio.create_task(violation_observed.wait())
        await asyncio.wait(
            {navigation_task, violation_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if violation_observed.is_set():
            await _cancel_task(navigation_task)
            navigation_task = None
            await _cancel_task(violation_task)
            violation_task = None
            raise _page_state_failure(state, redirects=redirects) or _GuestFailure(
                "fetch_failed",
                effective_origin=state.effective_origin,
            )
        await _cancel_task(violation_task)
        violation_task = None
        final_response = await navigation_task
        navigation_task = None
        await page.wait_for_timeout(_RENDER_SETTLE_MILLISECONDS)
        state_failure = _page_state_failure(state, redirects=redirects)
        if state_failure is not None:
            raise state_failure
        if final_response is None:
            raise _GuestFailure("fetch_failed", effective_origin=state.effective_origin)
        final_headers = await final_response.all_headers()
        access = _guest_http_access(
            final_response.url,
            final_response.status,
            final_headers,
            source="browser_response",
        )
        if access is not None:
            raise _GuestFailure(
                "access_blocked",
                access=access,
                effective_origin=_guest_https_origin(final_response.url),
            )
        if final_response.status < 200 or final_response.status >= 300:
            raise _GuestFailure("http_status", status_code=final_response.status)
        content_type = final_headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in _HTML_CONTENT_TYPES | _TEXT_CONTENT_TYPES:
            raise _GuestFailure("unsupported_content")
        # Freeze page-authored JavaScript before inspecting the document. The
        # isolated world below retains Playwright/CDP inspection capability but
        # cannot observe page-world prototype or own-property overrides. This
        # makes the node ceiling and accessibility evidence describe one stable
        # document rather than two page-controlled moments in time.
        await cdp.send("Emulation.setScriptExecutionDisabled", {"value": True})
        await _wait_for_browser_violation(violation_observed)
        state_failure = _page_state_failure(state, redirects=redirects)
        if state_failure is not None:
            raise state_failure
        if request.operation == "fetch":
            title, representation, content, truncation_reasons = await _extract_page_representation(
                page,
                cdp,
                request,
                operation_timeout_ms=operation_timeout_ms,
            )
        else:
            try:
                if main_frame_id is None:
                    raise _GuestFailure("browser_crash")
                screenshot_projection = await _capture_page_screenshot(
                    page,
                    cdp,
                    request,
                    main_frame_id=main_frame_id,
                )
            except PlaywrightTimeoutError:
                raise
            except PlaywrightError as exc:
                raise _screenshot_playwright_failure(
                    state=state,
                    browser=browser,
                    page=page,
                ) from exc
        state_failure = _page_state_failure(state, redirects=redirects)
        if state_failure is not None:
            raise state_failure
        if request.operation == "fetch":
            success_projection = (
                page.url,
                title,
                representation,
                content,
                truncation_reasons,
            )
    except asyncio.CancelledError as exc:
        primary = exc
        raise
    except (PlaywrightTimeoutError, TimeoutError) as exc:
        primary = _page_state_failure(state, redirects=redirects) or _GuestFailure(
            "timeout",
            effective_origin=state.effective_origin,
        )
        raise primary from exc
    except _GuestFailure as exc:
        primary = exc
        raise
    except PlaywrightError as exc:
        primary = _page_state_failure(state, redirects=redirects)
        if primary is None and _is_proxy_tunnel_failure(exc):
            primary = _GuestFailure("fetch_failed", effective_origin=state.effective_origin)
        if primary is None:
            primary = _GuestFailure(
                "browser_crash" if launched else "browser_unavailable",
                effective_origin=state.effective_origin,
            )
        raise primary from exc
    except Exception as exc:
        primary = _page_state_failure(state, redirects=redirects) or _GuestFailure(
            "browser_crash",
            effective_origin=state.effective_origin,
        )
        raise primary from exc
    finally:
        cleanup_task = asyncio.create_task(
            _cleanup_browser_resources(
                violation_task=violation_task,
                navigation_task=navigation_task,
                context=context,
                page=page,
                response_observed=response_observed,
                unexpected_page_observed=unexpected_page_observed,
                browser=browser,
                playwright=playwright,
                timeout_seconds=min(
                    cleanup_timeout_seconds,
                    max(
                        0.0,
                        cleanup_deadline - asyncio.get_running_loop().time(),
                    ),
                ),
            )
        )
        cleanup_outcome = await _await_browser_cleanup_resisting_cancellation(cleanup_task)
        state.cleanup_failed = state.cleanup_failed or bool(cleanup_outcome.errors)
        if cleanup_outcome.cancellation is not None:
            cause = _browser_cleanup_evidence(primary, cleanup_outcome.errors)
            if cause is None:
                raise cleanup_outcome.cancellation
            raise cleanup_outcome.cancellation from cause
        if isinstance(primary, asyncio.CancelledError):
            if cleanup_outcome.errors:
                raise primary from _browser_cleanup_evidence(None, cleanup_outcome.errors)
            # The active try statement republishes this cancellation after the
            # finally block. Do not manufacture a second cancellation request.
        elif state.cleanup_failed:
            cleanup = _GuestFailure("cleanup_failed")
            cause = _browser_cleanup_evidence(primary, cleanup_outcome.errors)
            if cause is None:
                raise cleanup
            raise cleanup from cause
    state_failure = _page_state_failure(state, redirects=redirects)
    if state_failure is not None:
        raise state_failure
    if request.operation == "screenshot":
        if screenshot_projection is None:  # pragma: no cover - success construction invariant
            raise _GuestFailure("browser_crash")
        final_url, title, title_truncated, width, height, screenshot = screenshot_projection
        return {
            "protocol_version": PROTOCOL_VERSION,
            "worker_version": WORKER_VERSION,
            "playwright_version": PLAYWRIGHT_VERSION,
            "kind": "screenshot",
            "requested_url": request.url,
            "final_url": final_url,
            "title": title,
            "title_truncated": title_truncated,
            "redirects": list(redirects),
            "response_bytes": state.response_bytes,
            "request_count": state.request_count,
            "full_page": request.full_page,
            "width": width,
            "height": height,
            "data_base64": base64.b64encode(screenshot).decode("ascii"),
        }
    if success_projection is None:  # pragma: no cover - success construction invariant
        raise _GuestFailure("browser_crash")
    final_url, title, representation, content, final_truncation_reasons = success_projection
    return {
        "protocol_version": PROTOCOL_VERSION,
        "worker_version": WORKER_VERSION,
        "playwright_version": PLAYWRIGHT_VERSION,
        "kind": "success",
        "requested_url": request.url,
        "final_url": final_url,
        "title": title,
        "representation": representation,
        "content": content,
        "redirects": list(redirects),
        "truncation_reasons": list(final_truncation_reasons),
        "response_bytes": state.response_bytes,
        "request_count": state.request_count,
    }


def _page_state_failure(
    state: _PageState,
    *,
    redirects: list[dict[str, Any]],
) -> _GuestFailure | None:
    if state.limit_exceeded:
        return _GuestFailure("oversized_response")
    if state.denied_code is not None:
        if state.denied_code in {
            "destination_denied",
            "dns_failure",
            "fetch_failed",
            "oversized_response",
            "redirect_denied",
            "timeout",
        }:
            return _GuestFailure(
                state.denied_code,
                effective_origin=state.effective_origin,
            )
        return _GuestFailure("fetch_failed", effective_origin=state.effective_origin)
    if state.access_evidence is not None and state.effective_origin is not None:
        return _GuestFailure(
            "access_blocked",
            access=state.access_evidence,
            effective_origin=state.effective_origin,
        )
    if state.response_inspection_failed:
        return _GuestFailure("fetch_failed", effective_origin=state.effective_origin)
    if len(redirects) > state.max_redirects:
        return _GuestFailure("redirect_denied", effective_origin=state.effective_origin)
    return None


async def _cancel_tasks(tasks: tuple[asyncio.Task[Any], ...]) -> None:
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def _cancel_task(task: asyncio.Task[Any]) -> None:
    await _cancel_tasks((task,))


async def _cleanup_browser_resources(
    *,
    violation_task: asyncio.Task[Any] | None,
    navigation_task: asyncio.Task[Any] | None,
    context: Any,
    page: Any,
    response_observed: Any,
    unexpected_page_observed: Any,
    browser: Any,
    playwright: Any,
    timeout_seconds: float,
) -> tuple[BaseException, ...]:
    """Attempt every browser owner within one finite cleanup reserve."""

    pending_tasks = tuple(task for task in (violation_task, navigation_task) if task is not None)
    before_listener_removal: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
    if pending_tasks:

        async def cancel_pending_tasks() -> None:
            await _cancel_tasks(pending_tasks)

        before_listener_removal.append(("pending tasks", cancel_pending_tasks))
    if context is not None:
        before_listener_removal.append(("browser context", context.close))
    after_listener_removal: list[tuple[str, Callable[[], Awaitable[Any]]]] = []
    if browser is not None:
        after_listener_removal.append(("browser", browser.close))
    if playwright is not None:
        after_listener_removal.append(("Playwright driver", playwright.stop))

    async_steps = before_listener_removal + after_listener_removal
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    errors: list[BaseException] = []
    completed_steps = 0

    async def run_steps(steps: list[tuple[str, Callable[[], Awaitable[Any]]]]) -> None:
        nonlocal completed_steps
        for label, operation in steps:
            remaining_steps = len(async_steps) - completed_steps
            remaining_seconds = max(0.0, deadline - asyncio.get_running_loop().time())
            stage_seconds = remaining_seconds / remaining_steps
            completed_steps += 1
            if stage_seconds <= 0:
                errors.append(TimeoutError(f"Browser cleanup stage {label} timed out."))
                continue
            try:
                async with asyncio.timeout(stage_seconds):
                    await operation()
            except TimeoutError as exc:
                timeout = TimeoutError(f"Browser cleanup stage {label} timed out.")
                timeout.__cause__ = exc
                errors.append(timeout)
            except asyncio.CancelledError as exc:
                failure = RuntimeError(
                    f"Browser cleanup stage {label} cancelled without caller cancellation."
                )
                failure.__cause__ = exc
                errors.append(failure)
            except Exception as exc:
                errors.append(exc)

    await run_steps(before_listener_removal)
    if page is not None and response_observed is not None:
        try:
            page.remove_listener("response", response_observed)
        except Exception as exc:
            errors.append(exc)
    if context is not None and unexpected_page_observed is not None:
        try:
            context.remove_listener("page", unexpected_page_observed)
        except Exception as exc:
            errors.append(exc)
    await run_steps(after_listener_removal)
    return tuple(errors)


async def _await_browser_cleanup_resisting_cancellation(
    cleanup_task: asyncio.Task[tuple[BaseException, ...]],
) -> _BrowserCleanupOutcome:
    """Finish bounded cleanup while retaining every caller cancellation request."""

    current_task = asyncio.current_task()
    cancellation_baseline = 0 if current_task is None else current_task.cancelling()
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            # Deliver cancellation already requested before the cleanup await,
            # including the race where cleanup completed in the same loop turn.
            await asyncio.sleep(0)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
        if cleanup_task.done():
            break
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            cancellation_requests = 0 if current_task is None else current_task.cancelling()
            if cancellation is not None or cancellation_requests > cancellation_baseline:
                if cancellation is None:
                    cancellation = exc
                continue
            break
    try:
        errors = cleanup_task.result()
    except asyncio.CancelledError as exc:
        failure = RuntimeError("Browser cleanup task cancelled unexpectedly.")
        failure.__cause__ = exc
        errors = (failure,)
    return _BrowserCleanupOutcome(errors=errors, cancellation=cancellation)


def _browser_cleanup_evidence(
    primary: BaseException | None,
    cleanup_errors: tuple[BaseException, ...],
) -> BaseException | None:
    evidence = tuple(
        error
        for error in (primary, *cleanup_errors)
        if error is not None and not isinstance(error, asyncio.CancelledError)
    )
    if not evidence:
        return None
    if len(evidence) == 1:
        return evidence[0]
    return BaseExceptionGroup("Browser operation and cleanup both failed.", list(evidence))


async def _wait_for_browser_violation(violation_observed: asyncio.Event) -> None:
    try:
        async with asyncio.timeout(_FINAL_NETWORK_SETTLE_SECONDS):
            await violation_observed.wait()
    except TimeoutError:
        pass


def _browser_time_budget(total_seconds: float) -> tuple[int, float]:
    cleanup_seconds = _browser_cleanup_reserve_seconds(total_seconds)
    operation_milliseconds = max(
        1,
        math.floor(max(0.001, total_seconds - cleanup_seconds) * 1000),
    )
    return operation_milliseconds, cleanup_seconds


def _browser_cleanup_reserve_seconds(total_seconds: float) -> float:
    return min(
        total_seconds,
        _MAX_CLEANUP_RESERVE_SECONDS,
        max(_MIN_CLEANUP_RESERVE_SECONDS, total_seconds * 0.1),
    )


def _temporary_profile_cleanup_reserve_seconds(cleanup_seconds: float) -> float:
    return min(
        cleanup_seconds,
        _MAX_PROFILE_CLEANUP_RESERVE_SECONDS,
        max(_MIN_PROFILE_CLEANUP_RESERVE_SECONDS, cleanup_seconds * 0.2),
    )


async def _run(request: _Request) -> dict[str, Any]:
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        raise _GuestFailure("capability_refused")
    try:
        installed_playwright = importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError as exc:
        raise _GuestFailure("browser_unavailable") from exc
    if installed_playwright != PLAYWRIGHT_VERSION:
        raise _GuestFailure("incompatible_browser")
    proxy, ca_path = _proxy_and_ca()
    state = _PageState(
        max_response_bytes=request.limits.max_response_bytes,
        max_redirects=request.limits.max_redirects,
        max_requests=request.limits.max_requests,
    )
    loop = asyncio.get_running_loop()
    total_deadline = loop.time() + request.limits.timeout_seconds
    try:
        original_cwd = os.getcwd()
    except OSError:
        original_cwd = "/"
    profile_cleanup_failed = False
    profile_cleanup_reserve = _temporary_profile_cleanup_reserve_seconds(
        _browser_cleanup_reserve_seconds(request.limits.timeout_seconds)
    )
    try:
        async with asyncio.timeout(request.limits.timeout_seconds):
            temporary_profile: _TemporaryProfileOwner | None = None
            primary: BaseException | None = None
            try:
                try:
                    temporary_profile = await _start_temporary_profile_owner(
                        timeout_seconds=profile_cleanup_reserve,
                        startup_timeout_seconds=max(
                            0.001,
                            total_deadline - loop.time(),
                        ),
                    )
                except asyncio.CancelledError:
                    profile_cleanup_failed = True
                    raise
                home = temporary_profile.home
                _sanitize_environment(home, proxy=proxy, ca_path=ca_path)
                os.chdir(home)
                try:
                    await _install_browser_ca(home, ca_path)
                    remaining_seconds = max(0.001, total_deadline - loop.time())
                    operation_timeout_ms, total_cleanup_seconds = _browser_time_budget(
                        remaining_seconds,
                    )
                    profile_cleanup_reserve = _temporary_profile_cleanup_reserve_seconds(
                        total_cleanup_seconds
                    )
                    browser_cleanup_seconds = max(
                        0.0,
                        total_cleanup_seconds - profile_cleanup_reserve,
                    )
                    operation_seconds = operation_timeout_ms / 1000
                    try:
                        async with asyncio.timeout(operation_seconds):
                            return await _fetch_with_browser(
                                request,
                                proxy,
                                state=state,
                                operation_timeout_ms=operation_timeout_ms,
                                cleanup_timeout_seconds=browser_cleanup_seconds,
                                cleanup_deadline=total_deadline - profile_cleanup_reserve,
                            )
                    except TimeoutError as exc:
                        if state.cleanup_failed:
                            raise _GuestFailure("cleanup_failed") from exc
                        primary = _page_state_failure(state, redirects=state.redirects)
                        raise primary or _GuestFailure("timeout") from exc
                finally:
                    try:
                        os.chdir(original_cwd)
                    except OSError:
                        os.chdir("/")
            except BaseException as exc:
                primary = exc
                raise
            finally:
                if temporary_profile is not None:
                    cleanup_task = asyncio.create_task(
                        _cleanup_temporary_profile_owner(
                            temporary_profile,
                            timeout_seconds=min(
                                profile_cleanup_reserve,
                                max(0.0, total_deadline - loop.time()),
                            ),
                        )
                    )
                    cleanup_outcome = await _await_browser_cleanup_resisting_cancellation(
                        cleanup_task
                    )
                    profile_cleanup_failed = bool(cleanup_outcome.errors)
                    if cleanup_outcome.cancellation is not None:
                        cause = _browser_cleanup_evidence(primary, cleanup_outcome.errors)
                        if cause is None:
                            raise cleanup_outcome.cancellation
                        raise cleanup_outcome.cancellation from cause
                    if isinstance(primary, asyncio.CancelledError):
                        if cleanup_outcome.errors:
                            raise primary from _browser_cleanup_evidence(
                                None,
                                cleanup_outcome.errors,
                            )
                    elif profile_cleanup_failed:
                        cleanup_failure = _GuestFailure("cleanup_failed")
                        cause = _browser_cleanup_evidence(primary, cleanup_outcome.errors)
                        if cause is None:  # pragma: no cover - non-empty error invariant
                            raise cleanup_failure
                        raise cleanup_failure from cause
    except TimeoutError as exc:
        if state.cleanup_failed or profile_cleanup_failed:
            raise _GuestFailure("cleanup_failed") from exc
        primary = _page_state_failure(state, redirects=state.redirects)
        raise primary or _GuestFailure("timeout") from exc


def _interactive_socket_path(session_id: str) -> Path:
    digest = (
        __import__("hashlib")
        .sha256(b"cayu-browser-session-socket-v1\0" + session_id.encode("utf-8"))
        .hexdigest()
    )
    return _INTERACTIVE_ROOT / f"{digest}.sock"


def _interactive_retired_path(session_id: str) -> Path:
    token = _interactive_retirement_token(session_id)
    # The full digest is the durable retirement identity. Short shared buckets
    # allow an unrelated session to overwrite positive quiescence evidence and
    # later resurrect the retired allocation after acknowledgement loss.
    return _INTERACTIVE_ROOT / f"{token}.retired"


def _interactive_retirement_token(session_id: str) -> str:
    return (
        __import__("hashlib")
        .sha256(b"cayu-browser-session-retired-v1\0" + session_id.encode("utf-8"))
        .hexdigest()
    )


def _interactive_retirement_is_recorded(session_id: str) -> bool:
    retired_path = _interactive_retired_path(session_id)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(retired_path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 64:
            return False
        recorded = os.read(descriptor, 65)
    except OSError:
        return False
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)
    return recorded == _interactive_retirement_token(session_id).encode("ascii")


def _record_interactive_retirement(session_id: str) -> bool:
    """Publish one exact marker without aliasing another session identity."""

    retired_path = _interactive_retired_path(session_id)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(retired_path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        token = _interactive_retirement_token(session_id).encode("ascii")
        return os.write(descriptor, token) == len(token)
    except OSError:
        return False
    finally:
        if descriptor is not None:
            with contextlib.suppress(OSError):
                os.close(descriptor)


async def _await_interactive_retirement(session_id: str) -> bool:
    """Wait briefly for a daemon's already-started cleanup to settle."""

    deadline = asyncio.get_running_loop().time() + _INTERACTIVE_CONNECT_SECONDS
    while True:
        if _interactive_retirement_is_recorded(session_id):
            return True
        if asyncio.get_running_loop().time() >= deadline:
            return False
        await asyncio.sleep(0.025)


def _interactive_response_has_uncertain_closure(response: Mapping[str, Any]) -> bool:
    return (
        response.get("kind") == "error"
        and response.get("error") == "session_closed"
        and response.get("allocation_disposition") == "uncertain"
    )


def _interactive_response_claims_retirement(response: Mapping[str, Any]) -> bool:
    return response.get("allocation_disposition") == "retired"


async def _run_interactive_request(raw: Any) -> dict[str, Any]:
    request = _interactive_request_from_json(raw)
    pre_dispatch_disposition: Literal["live", "retired"] = (
        "retired" if request.operation in {"navigate", "profile_restore"} else "live"
    )
    if not hasattr(os, "geteuid") or os.geteuid() == 0:
        raise _GuestFailure(
            "capability_refused",
            allocation_disposition=pre_dispatch_disposition,
        )
    try:
        installed_playwright = importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError as exc:
        raise _GuestFailure(
            "browser_unavailable",
            allocation_disposition=pre_dispatch_disposition,
        ) from exc
    if installed_playwright != PLAYWRIGHT_VERSION:
        raise _GuestFailure(
            "incompatible_browser",
            allocation_disposition=pre_dispatch_disposition,
        )
    # Validate the exact broker and CA authority before consulting or starting
    # a persistent browser allocation.
    try:
        _proxy_and_ca()
    except _GuestFailure as exc:
        raise _GuestFailure(
            exc.code,
            status_code=exc.status_code,
            allocation_disposition=pre_dispatch_disposition,
        ) from exc
    socket_path = _interactive_socket_path(request.session_id)
    try:
        _INTERACTIVE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(_INTERACTIVE_ROOT, 0o700)
    except OSError as exc:
        raise _GuestFailure(
            "browser_unavailable",
            allocation_disposition=pre_dispatch_disposition,
        ) from exc
    response = await _interactive_send(socket_path, raw)
    if response is not None:
        if (
            _interactive_response_claims_retirement(response)
            and not _interactive_retirement_is_recorded(request.session_id)
            and not await _await_interactive_retirement(request.session_id)
        ):
            # Positive retirement may release host-side profile authority. Do
            # not publish it until the daemon's process-loss marker is visible.
            return _interactive_error_payload(_GuestFailure("cleanup_failed"))
        if (
            request.operation != "navigate"
            and _interactive_response_has_uncertain_closure(response)
            and await _await_interactive_retirement(request.session_id)
        ):
            return {**response, "allocation_disposition": "retired"}
        return response
    if request.operation not in {"navigate", "profile_restore"} or request.reconcile_only:
        retired = await _await_interactive_retirement(request.session_id)
        disposition: Literal["retired", "uncertain"] = "retired" if retired else "uncertain"
        return _interactive_error_payload(
            _GuestFailure(
                "allocation_lost" if retired else "session_closed",
                allocation_disposition=disposition,
            )
        )
    if _interactive_retirement_is_recorded(request.session_id):
        raise _GuestFailure(
            "browser_unavailable",
            allocation_disposition="retired",
        )
    await _start_interactive_daemon(request.session_id, socket_path)
    # A daemon can fail near the end of its connection window and then spend
    # its bounded cleanup reserve settling Playwright and Chromium. Keep the
    # launching request as the reconciliation owner through that reserve so a
    # retirement marker published just after the ordinary connection deadline
    # is not stranded behind the parent allocation-capacity gate.
    deadline = (
        asyncio.get_running_loop().time()
        + _INTERACTIVE_CONNECT_SECONDS
        + _INTERACTIVE_STARTUP_SETTLEMENT_SECONDS
    )
    while asyncio.get_running_loop().time() < deadline:
        response = await _interactive_send(socket_path, raw)
        if response is not None:
            if (
                _interactive_response_claims_retirement(response)
                and not _interactive_retirement_is_recorded(request.session_id)
                and not await _await_interactive_retirement(request.session_id)
            ):
                return _interactive_error_payload(_GuestFailure("cleanup_failed"))
            return response
        if _interactive_retirement_is_recorded(request.session_id):
            raise _GuestFailure(
                "browser_unavailable",
                allocation_disposition="retired",
            )
        await asyncio.sleep(0.025)
    if _interactive_retirement_is_recorded(request.session_id):
        raise _GuestFailure(
            "browser_unavailable",
            allocation_disposition="retired",
        )
    raise _GuestFailure("browser_unavailable")


def _private_control_bootstrap(raw: Any) -> tuple[str, dict[str, str]]:
    if (
        type(raw) is not dict
        or set(raw) != {"protocol_version", "session_id", "endpoint", "credential", "scope_sha256"}
        or raw["protocol_version"] != CONTROL_BOOTSTRAP_PROTOCOL
    ):
        raise _GuestFailure("incompatible_browser")
    try:
        session_id = _interactive_identifier(raw["session_id"])
        endpoint = validate_control_endpoint(raw["endpoint"])
        credential = raw["credential"]
        scope = raw["scope_sha256"]
        if type(credential) is not str or re.fullmatch(r"[0-9a-f]{64}", credential) is None:
            raise GuestControlFailure()
        GuestControlFence._digest(scope)
    except (ValueError, TypeError, GuestControlFailure):
        raise _GuestFailure("incompatible_browser") from None
    return session_id, {"endpoint": endpoint, "credential": credential, "scope_sha256": scope}


async def _run_private_control_bootstrap(raw: Any) -> dict[str, Any]:
    session_id, material = _private_control_bootstrap(raw)
    # Bootstrap can attach only to an existing daemon. It must never create or
    # revive a browser allocation after loss or replay a spent capability.
    response = await _interactive_send(
        _interactive_socket_path(session_id),
        {
            "protocol_version": CONTROL_BOOTSTRAP_PROTOCOL,
            "session_id": session_id,
            **material,
        },
    )
    material.clear()
    if (
        type(response) is not dict
        or set(response) != {"schema_version", "bootstrap_accepted"}
        or type(response["schema_version"]) is not int
        or response["schema_version"] != 1
        or response["bootstrap_accepted"] is not True
    ):
        raise _GuestFailure("browser_unavailable")
    return {"schema_version": 1, "bootstrap_accepted": True}


async def _interactive_send(socket_path: Path, raw: Any) -> dict[str, Any] | None:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(
                str(socket_path),
                limit=_INTERACTIVE_MAX_MESSAGE_BYTES + 1,
            ),
            timeout=0.25,
        )
    except (FileNotFoundError, ConnectionError, OSError, TimeoutError):
        return None
    try:
        encoded = json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(encoded) > _INTERACTIVE_MAX_PRIVATE_REQUEST_BYTES:
            raise _GuestFailure("incompatible_browser")
        writer.write(encoded + b"\n")
        await writer.drain()
        response = await asyncio.wait_for(
            reader.readuntil(b"\n"),
            timeout=_INTERACTIVE_CONNECT_SECONDS + _INTERACTIVE_MAX_WAIT_MS / 1000,
        )
        if len(response) > _INTERACTIVE_MAX_MESSAGE_BYTES:
            raise _GuestFailure("browser_crash")
        decoded = json.loads(response.decode("utf-8"))
        if type(decoded) is not dict:
            raise _GuestFailure("browser_crash")
        return decoded
    except (asyncio.LimitOverrunError, json.JSONDecodeError, UnicodeError) as exc:
        raise _GuestFailure("browser_crash") from exc
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


async def _start_interactive_daemon(session_id: str, socket_path: Path) -> None:
    if socket_path.exists():
        with contextlib.suppress(OSError):
            socket_path.unlink()
    try:
        subprocess.Popen(
            [
                sys.executable,
                "-I",
                str(Path(__file__).resolve()),
                _INTERACTIVE_DAEMON_ARGUMENT,
                session_id,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise _GuestFailure("browser_unavailable", allocation_disposition="retired") from exc


class _InteractiveDaemon:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._operator_channel_id: str | None = None
        self._operator_input_task: asyncio.Task[tuple[BaseException, ...]] | None = None
        self.operator_frames = GuestFrameOwner()
        self._operator_bootstrap_task: asyncio.Task[tuple[BaseException, ...]] | None = None
        self._operator_connection: Any = None
        self._operator_frame_count = 0
        self._operator_frame_started = float("-inf")
        self.playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.pages: dict[str, _InteractivePage] = {}
        self.active_page_id: str | None = None
        self.browser_version = "unknown"
        self.lock = asyncio.Lock()
        self.close_requested = asyncio.Event()
        self.closing = False
        self.close_after_response = False
        self.idle_expired = False
        self.operations: dict[str, _InteractiveOperationRecord] = {}
        self.page_cleanup_operations: dict[str, _InteractiveOperationRecord] = {}
        self.session_cleanup_operations: dict[str, _InteractiveOperationRecord] = {}
        self.profile_operations: dict[str, _InteractiveOperationRecord] = {}
        self.operation_ledger_bytes = 0
        self.total_page_creations = 0
        self.total_operations = 0
        self.total_observations = 0
        self.total_refs = 0
        self.total_requests = 0
        self.total_artifacts = 0
        self.cleanup_operation_count = 0
        self.configuration_fingerprint: str | None = None
        self.configuration_limits: _InteractiveLimits | None = None
        self.configuration_multi_page: bool | None = None
        self.configuration_popup_policy: _InteractivePopupPolicy | None = None
        self.popup_guard_token = secrets.token_hex(32)
        self.active_request: _InteractiveRequest | None = None
        self.active_delta: _InteractivePageDelta | None = None
        self.popup_cleanup_task: asyncio.Task[bool] | None = None
        self.popup_candidate_observed = asyncio.Event()
        self.visual_worker_instance = "vw_" + secrets.token_hex(16)
        self.control = GuestControlFence(worker_instance=self.visual_worker_instance)
        self.total_visual_captures = 0
        self.configuration_visual_policy: dict[str, Any] | None = None
        self.popup_effect_opener_page_id: str | None = None
        self.popup_effect_opener_origin: str | None = None
        self.popup_cleanup_pages: list[Any] = []
        self.popup_cleanup_retire = False
        self.session_cleanup_tasks: dict[str, asyncio.Task[Any]] = {}
        self.pending_closed_page_ids: set[str] = set()
        self.pending_crashed_page_ids: set[str] = set()
        self.idle_timeout_seconds = _INTERACTIVE_IDLE_SECONDS
        self.last_activity = 0.0
        self.home: Path | None = None
        self.profile_owner: _TemporaryProfileOwner | None = None
        self.profile_output_values: tuple[str, ...] | None = None
        self.profile_allowed_origins: tuple[str, ...] | None = None
        self.profile_plaintext_limit: int | None = None
        self.profile_timeout_seconds: float | None = None

    async def start(self) -> None:
        try:
            from playwright.async_api import async_playwright
        except (ImportError, OSError) as exc:
            raise _GuestFailure("browser_unavailable") from exc
        proxy, ca_path = _proxy_and_ca()
        try:
            await self._start_profile_home()
            if self.home is None:  # pragma: no cover - profile-owner invariant
                raise _GuestFailure("cleanup_failed")
            _sanitize_environment(self.home, proxy=proxy, ca_path=ca_path)
            await _install_browser_ca(self.home, ca_path)
            self._trusted_ca_digest = hashlib.sha256(Path(ca_path).read_bytes()).hexdigest()
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                chromium_sandbox=True,
                proxy={"server": proxy},
                ignore_default_args=["--disable-popup-blocking"],
                args=[
                    "--disable-background-networking",
                    "--disable-component-update",
                    "--disable-default-apps",
                    "--disable-dev-shm-usage",
                    "--disable-domain-reliability",
                    "--disable-features=AutofillServerCommunication,MediaRouter,BackForwardCache",
                    "--disable-quic",
                    "--disable-sync",
                    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                    "--metrics-recording-only",
                    "--no-first-run",
                    "--password-store=basic",
                    "--use-mock-keychain",
                    "--webrtc-ip-handling-policy=disable_non_proxied_udp",
                ],
            )
            self.browser_version = str(self.browser.version)
            # A context is intentionally not created here.  The first navigate
            # request may carry application-owned profile state, which must be
            # imported before any page or untrusted document exists.
        except _GuestFailure:
            await self.close()
            raise
        except Exception as exc:
            await self.close()
            raise _GuestFailure("browser_unavailable") from exc

    async def _start_profile_home(self) -> None:
        """Allocate a profile home owned by an independent process-loss guardian."""

        if self.profile_owner is not None or self.home is not None:
            raise _GuestFailure("incompatible_browser")
        owner = await _start_temporary_profile_owner(
            timeout_seconds=_MAX_PROFILE_CLEANUP_RESERVE_SECONDS,
        )
        self.profile_owner = owner
        self.home = owner.home

    async def _ensure_context(self, storage_state: dict[str, Any] | None) -> None:
        if self.context is not None:
            if storage_state is not None:
                raise _GuestFailure("incompatible_browser")
            return
        try:
            self.context = await self.browser.new_context(
                accept_downloads=True,
                ignore_https_errors=False,
                java_script_enabled=True,
                service_workers="block",
                viewport={"width": 1280, "height": 720},
                device_scale_factor=1,
                **({"storage_state": storage_state} if storage_state is not None else {}),
            )
            # A live transport to this daemon is not evidence that its browser
            # allocation still exists. Fence native/model actions immediately;
            # the next channel exchange carries the mismatch to the host owner.
            self.browser.on("disconnected", self.control.uncertain)
            if not self.browser.is_connected():
                self.control.uncertain()
                raise _GuestFailure("browser_crash")
        except _GuestFailure:
            await self.close()
            raise
        except Exception as exc:
            raise _GuestFailure("incompatible_browser") from exc

    async def _refresh_egress_trust(self) -> None:
        """Rotate trusted CA after exact-allocation egress recovery, without effect replay."""
        previous = getattr(self, "_trusted_ca_digest", None)
        if previous is None:
            return
        try:
            _proxy, ca_path = _proxy_and_ca()
            current = hashlib.sha256(Path(ca_path).read_bytes()).hexdigest()
            if current == previous:
                return
            if self.home is None:
                raise _GuestFailure("restoration_required")
            await _install_browser_ca(self.home, ca_path, replace_existing=True)
            self._trusted_ca_digest = current
            # Neither a cached observation nor an old ref authorizes a fresh action
            # across an authority transfer. A new observe issues the new epoch.
            for state in self.pages.values():
                state.control_epoch += 1
                state.revision = f"br_{secrets.token_hex(16)}"
                state.clear_refs()
        except asyncio.CancelledError:
            raise
        except Exception:
            raise _GuestFailure("restoration_required") from None

    async def execute(self, request: _InteractiveRequest) -> dict[str, Any]:
        if request.session_id != self.session_id:
            raise _GuestFailure("incompatible_browser")
        async with self.lock:
            if self._operator_bootstrap_task is not None and self.control.binding_sha256 is None:
                raise _GuestFailure("policy_denied", allocation_disposition="live")
            try:
                self.control.check_model(request.invocation_control_epoch, request.operation)
            except GuestControlFailure as exc:
                raise _GuestFailure("policy_denied", allocation_disposition="live") from exc
            if (
                self.control.capture_restricted
                and self.profile_output_values is None
                and request.operation != "close"
            ):
                raise _GuestFailure("policy_denied", allocation_disposition="live")
            fingerprint = _interactive_operation_fingerprint(request)
            existing = self.operations.get(request.operation_id)
            if existing is None:
                existing = self.page_cleanup_operations.get(request.operation_id)
            if existing is None:
                existing = self.session_cleanup_operations.get(request.operation_id)
            if existing is None:
                existing = self.profile_operations.get(request.operation_id)
            if existing is not None:
                if existing.fingerprint != fingerprint:
                    return _interactive_error_payload(_GuestFailure("operation_conflict"))
                return json.loads(json.dumps(existing.response))
            if self.closing or self.close_requested.is_set():
                raise _GuestFailure("session_closed")
            await self._refresh_egress_trust()
            await self._ensure_configuration(request)
            if request.reconcile_only:
                return _interactive_error_payload(
                    _GuestFailure("operation_not_dispatched", allocation_disposition="live")
                )
            operation_records = (
                self.session_cleanup_operations
                if request.operation == "close"
                else self.page_cleanup_operations
                if request.operation == "close_page"
                else self.profile_operations
                if request.operation in {"profile_restore", "profile_checkpoint"}
                else self.operations
            )
            if request.operation == "close":
                if operation_records:
                    return _interactive_error_payload(_GuestFailure("resource_exhausted"))
            elif request.operation == "close_page":
                if len(operation_records) >= request.limits.max_page_cleanup_operations:
                    return _interactive_error_payload(_GuestFailure("resource_exhausted"))
            elif request.operation in {"profile_restore", "profile_checkpoint"}:
                # A profile-bound allocation needs one private restore receipt
                # in addition to one checkpoint for every admitted public
                # operation.  The public max_operations bound must therefore
                # not consume the restore slot at its exact upper boundary.
                if len(operation_records) >= request.limits.max_operations + 1:
                    return _interactive_error_payload(_GuestFailure("resource_exhausted"))
            elif len(operation_records) >= request.limits.max_operations:
                return _interactive_error_payload(_GuestFailure("resource_exhausted"))
            try:
                response = await self._execute_locked(request)
            except _GuestFailure as exc:
                response = (
                    await self._retire_failed_allocation(exc, request)
                    if request.operation == "profile_restore"
                    else _interactive_error_payload(exc)
                )
            except Exception as exc:
                self.control.uncertain()
                failure = _interactive_playwright_error(request.operation, exc)
                response = (
                    await self._retire_failed_allocation(failure, request)
                    if request.operation == "profile_restore"
                    else _interactive_error_payload(failure)
                )
            finally:
                self.last_activity = asyncio.get_running_loop().time()
            encoded = json.dumps(
                response,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            retained = response
            response_bound = (
                _INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES * 7
                if request.operation in {"profile_restore", "profile_checkpoint"}
                else _INTERACTIVE_MAX_SNAPSHOT_BYTES * 2
            )
            if (
                len(encoded) > response_bound
                or self.operation_ledger_bytes + len(encoded) > _INTERACTIVE_OPERATION_LEDGER_BYTES
            ):
                retained = _interactive_error_payload(_GuestFailure("outcome_ambiguous"))
                encoded = json.dumps(
                    retained,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            operation_records[request.operation_id] = _InteractiveOperationRecord(
                fingerprint=fingerprint,
                response=json.loads(json.dumps(retained)),
                size_bytes=len(encoded),
            )
            self.operation_ledger_bytes += len(encoded)
            if (
                request.operation == "observe"
                and retained.get("kind") == "success"
                and type(retained.get("observation")) is dict
                and type(retained["observation"].get("revision")) is str
                and (
                    not self.control.capture_restricted
                    or retained.get("profile_output_protected") is True
                )
            ):
                self.control.fresh_observation_required = False
            return retained

    async def bootstrap_operator_channel(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Private bootstrap only; no ordinary operation record stores the bearer."""
        if type(raw) is not dict or set(raw) != {"endpoint", "credential", "scope_sha256"}:
            raise GuestControlFailure()
        endpoint = validate_control_endpoint(raw["endpoint"])
        credential = raw["credential"]
        scope = raw["scope_sha256"]
        if type(credential) is not str or re.fullmatch(r"[0-9a-f]{64}", credential) is None:
            raise GuestControlFailure()
        GuestControlFence._digest(scope)
        async with self.lock:
            if (
                self.closing
                or self.close_requested.is_set()
                or self._operator_bootstrap_task is not None
            ):
                raise GuestControlFailure()

            async def connect_and_own(token: str) -> tuple[BaseException, ...]:
                try:
                    connection = await open_guest_control_channel(
                        endpoint=endpoint, credential=token
                    )
                    token = ""
                    self._operator_connection = connection
                    if self.closing:
                        await asyncio.shield(self._operator_transport_close())
                        return ()
                    await GuestControlChannel(self, scope_sha256=scope).run(connection)
                    return ()
                except BaseException as failure:
                    self.operator_frames.suspend()
                    self.control.uncertain()
                    if self.closing and control_connection_closed_normally(failure):
                        return ()
                    return (failure,)
                finally:
                    token = ""

            self._operator_bootstrap_task = asyncio.create_task(
                connect_and_own(credential), name="cayu-private-browser-channel"
            )
        credential = ""
        # Acceptance means only that a finite channel owner exists, not that the
        # server has acknowledged a fence or granted operator input.
        return {"schema_version": 1, "bootstrap_accepted": True}

    def _operator_transport_close(self) -> asyncio.Task[Any]:
        task = self.session_cleanup_tasks.get("operator-transport")
        if task is None:
            if self._operator_connection is None:
                raise GuestControlFailure()
            task = asyncio.create_task(self._operator_connection.close())
            self.session_cleanup_tasks["operator-transport"] = task
        return task

    async def bind_operator_control(self, binding_sha256: str) -> dict[str, Any]:
        """Private control-channel handshake, never an ordinary JSON operation."""

        async with self.lock:
            if self.closing or self.close_requested.is_set() or self.context is None:
                raise GuestControlFailure()
            self.control.bind(binding_sha256)
            return self._operator_control_evidence()

    def claim_operator_channel(self, channel_id: str) -> None:
        if (
            self.closing
            or self.close_requested.is_set()
            or getattr(self, "_operator_channel_id", None) is not None
        ):
            raise GuestControlFailure()
        self._operator_channel_id = channel_id

    def release_operator_channel(self, channel_id: str) -> None:
        if getattr(self, "_operator_channel_id", None) != channel_id:
            raise GuestControlFailure()
        self._operator_channel_id = None

    async def operator_text_input(
        self,
        *,
        request_id: str,
        epoch: int,
        sequence: int,
        page_id: str,
        page_epoch: int,
        text: str,
    ) -> dict[str, Any]:
        return await self._operator_input(
            request_id=request_id,
            epoch=epoch,
            sequence=sequence,
            page_id=page_id,
            page_epoch=page_epoch,
            text=text,
            key=None,
        )

    async def operator_key_input(
        self,
        *,
        request_id: str,
        epoch: int,
        sequence: int,
        page_id: str,
        page_epoch: int,
        key: str,
    ) -> dict[str, Any]:
        return await self._operator_input(
            request_id=request_id,
            epoch=epoch,
            sequence=sequence,
            page_id=page_id,
            page_epoch=page_epoch,
            text=None,
            key=key,
        )

    async def _operator_input(
        self,
        *,
        request_id: str,
        epoch: int,
        sequence: int,
        page_id: str,
        page_epoch: int,
        text: str | None,
        key: str | None,
    ) -> dict[str, Any]:
        """Private transient text delivery; never a tool argument or receipt payload.

        Capture must already be settled by the sensitive-entry owner. Retain the
        native task after cancellation/timeout; coroutine cancellation does not
        establish that Chromium stopped processing an input operation.
        """
        keys = {
            "tab": "Tab",
            "backtab": "Shift+Tab",
            "enter": "Enter",
            "escape": "Escape",
            "backspace": "Backspace",
        }
        if type(page_id) is not str or type(page_epoch) is not int:
            raise GuestControlFailure()
        if key is not None:
            if text is not None or type(key) is not str or key not in keys:
                raise GuestControlFailure()
        elif (
            type(text) is not str
            or not 1 <= len(text) <= 4096
            or any(0xD800 <= ord(char) <= 0xDFFF or char == "\x00" for char in text)
        ):
            raise GuestControlFailure()
        async with self.lock:
            self.control.check_operator(request_id=request_id, epoch=epoch)
            state = self.pages.get(page_id)
            limits = self.configuration_limits
            if (
                self._operator_channel_id is None
                or not self.control.sensitive_entry
                or not self.operator_frames.paused
                or self.operator_frames.task is not None
                or state is None
                or state.lifecycle != "active"
                or self.active_page_id != page_id
                or state.control_epoch != page_epoch
                or state.limit_exceeded
                or state.denied_code is not None
                or limits is None
                or self.total_operations >= limits.max_operations
                or state.operation_count >= limits.max_operations_per_page
                or (self._operator_input_task is not None and not self._operator_input_task.done())
            ):
                raise GuestControlFailure()
            if text is not None:
                self._protect_operator_private_values((text,))
            self.control.begin_input(request_id=request_id, epoch=epoch, sequence=sequence)
            self.total_operations += 1
            state.operation_count += 1
            state.clear_refs()
            if state.visual_owner is not None:
                state.visual_owner.invalidate()

            # Playwright insert_text only inserts text at current browser focus;
            # it does not interpret key chords, selectors, scripts, or commands.
            async def deliver() -> tuple[BaseException, ...]:
                try:
                    if key is None:
                        await state.page.keyboard.insert_text(text)
                    else:
                        await state.page.keyboard.press(keys[key])
                except BaseException as failure:
                    return (failure,)
                return ()

            native = asyncio.create_task(deliver(), name="cayu-private-operator-input")
            self._operator_input_task = native
            try:
                _, pending = await asyncio.wait(
                    {native}, timeout=min(5.0, limits.max_wait_ms / 1000)
                )
                if pending:
                    raise GuestControlFailure()
                failures = native.result()
                if failures:
                    failure = failures[0]
                    if isinstance(failure, asyncio.CancelledError):
                        error = GuestControlFailure()
                        error.__cause__ = failure
                        raise error
                    raise failure
                self.control.settle_input(sequence)
                self._operator_input_task = None
            except BaseException:
                self.control.uncertain()
                raise
            return self._operator_control_evidence()

    def _protect_operator_private_values(self, values: tuple[str, ...]) -> None:
        limit = self.profile_plaintext_limit or _INTERACTIVE_MAX_PROFILE_PLAINTEXT_BYTES
        protected = _bounded_interactive_profile_private_values(
            self.profile_output_values or (), values, maximum_bytes=limit
        )
        self.profile_output_values = protected
        self.profile_plaintext_limit = limit
        if self.profile_timeout_seconds is None:
            self.profile_timeout_seconds = 5.0

    async def enter_operator_sensitive_entry(
        self, *, request_id: str, epoch: int
    ) -> dict[str, Any]:
        self.control.check_operator(request_id=request_id, epoch=epoch)
        # Stop new capture before waiting for the lifecycle lock or old frames.
        self.operator_frames.suspend()
        try:
            async with self.lock:
                await self.operator_frames.pause()
                self.control.check_operator(request_id=request_id, epoch=epoch)
                self._protect_operator_private_values(())
                self.control.sensitive_entry = True
                self.control.capture_restricted = True
                return {
                    **self._operator_control_evidence(),
                    "capture_generation": self.operator_frames.generation,
                    "sensitive_entry": True,
                    "capture_restricted": True,
                }
        except BaseException:
            self.control.uncertain()
            raise

    async def capture_operator_frame(
        self,
        *,
        view_id: str,
        epoch: int,
        page_id: str,
        page_epoch: int,
        send: Callable[[int, dict[str, Any], bytes], Awaitable[None]],
    ) -> None:
        """Capture only the bounded viewport into a private transport callback.

        No artifact, tool result, model visual target, title, or URL is produced.
        Viewing uses its own short application-authorized grant, not takeover or
        BrowserVisualPolicy. Sensitive entry excludes capture independently.
        """
        async with self.lock:
            self.control.check_view(view_id=view_id, epoch=epoch)
            # Pace before admitting a capture owner. Native page callbacks can
            # change page authority during this wait, so all admission checks
            # below use the post-wait state. A stale page remains a local refusal.
            delay = 0.5 - (time.monotonic() - self._operator_frame_started)
            if delay > 0:
                await asyncio.sleep(delay)
            self.control.check_view(view_id=view_id, epoch=epoch)
            if self.profile_output_values is not None:
                raise GuestCaptureDenied()
            state = self.pages.get(self.active_page_id or "")
            if (
                type(page_id) is not str
                or type(page_epoch) is not int
                or state is None
                or state.page_id != page_id
                or state.control_epoch != page_epoch
            ):
                # Background/stale view requests grant no authority to capture
                # the active page instead. Return the settled pixel-free refusal.
                raise GuestCaptureDenied()
            if (
                self._operator_channel_id is None
                or self.closing
                or self.close_requested.is_set()
                or state is None
                or state.lifecycle != "active"
                or not state.configured
                or state.limit_exceeded
                or state.denied_code is not None
            ):
                raise GuestControlFailure()
            viewport = state.page.viewport_size
            if (
                type(viewport) is not dict
                or set(viewport) != {"width", "height"}
                or type(viewport["width"]) is not int
                or type(viewport["height"]) is not int
                or not 1 <= viewport["width"] <= 1920
                or not 1 <= viewport["height"] <= 1080
            ):
                raise GuestControlFailure()
            width, height = viewport["width"], viewport["height"]
            page_epoch = state.control_epoch
            if self._operator_frame_count >= 7200:
                raise GuestControlFailure()

            def guard() -> None:
                self.control.check_view(view_id=view_id, epoch=epoch)
                if (
                    self.active_page_id != state.page_id
                    or self.operator_frames.paused
                    or self.profile_output_values is not None
                    or state.lifecycle != "active"
                    or state.control_epoch != page_epoch
                    or state.limit_exceeded
                    or state.denied_code is not None
                    or self.closing
                ):
                    raise GuestControlFailure()

            async def capture() -> bytes:
                self._operator_frame_started = time.monotonic()
                self._operator_frame_count += 1
                frame = await state.page.screenshot(
                    type="png",
                    full_page=False,
                    caret="initial",
                    timeout=5000,
                    clip={"x": 0, "y": 0, "width": width, "height": height},
                )
                if type(frame) is not bytes or _png_header_dimensions(frame) != (width, height):
                    raise GuestControlFailure()
                return frame

            async def deliver(generation: int, frame: bytes) -> None:
                await send(
                    generation,
                    {
                        "page_id": state.page_id,
                        "page_epoch": page_epoch,
                        "control_epoch": epoch,
                        "width": width,
                        "height": height,
                        "content_type": "image/png",
                    },
                    frame,
                )

            await self.operator_frames.capture_one(capture=capture, send=deliver, guard=guard)

    async def grant_operator_view(
        self, *, binding_sha256: str, view_id: str, epoch: int, until_ms: int
    ) -> dict[str, Any]:
        async with self.lock:
            if self._operator_channel_id is None:
                raise GuestControlFailure()
            self.control.grant_view(
                binding_sha256=binding_sha256, view_id=view_id, epoch=epoch, until_ms=until_ms
            )
            self.operator_frames.resume()
            return {
                **self._operator_control_evidence(),
                "view_id": view_id,
                "generation": self.operator_frames.generation,
            }

    async def operator_page_descriptors(self, *, epoch: int) -> dict[str, Any]:
        """Content-free page authority, serialized with model/page mutations."""
        async with self.lock:
            self.control.expire()
            if self.browser is not None and self.browser.is_connected() is False:
                self.control.uncertain()
                raise GuestControlAllocationLost()
            if (
                self._operator_channel_id is None
                or self.control.binding_sha256 is None
                or type(epoch) is not int
                or epoch != self.control.epoch
                or self.control.state
                not in {"agent_controlled", "takeover_requested", "operator_controlled"}
            ):
                raise GuestControlFailure()
            private_values = await self._refresh_profile_output_values()
            observed_pages = tuple(
                page for page in self.pages.values() if page.lifecycle in {"active", "background"}
            )
            pages = [
                {
                    "page_id": page.page_id,
                    "revision": self._operator_page_revision(page),
                    "control_epoch": page.control_epoch,
                }
                for page in observed_pages
            ]
            if any(
                page.lifecycle not in {"active", "background", "closed", "crashed"}
                for page in self.pages.values()
            ):
                raise GuestControlFailure()
            return {
                **self._operator_control_evidence(),
                "active_page_id": self.active_page_id,
                "pages": pages,
                "locations": [
                    {
                        "page": page,
                        "origin": self._operator_page_origin(
                            getattr(state.page, "url", None), private_values
                        ),
                    }
                    for page, state in zip(pages, observed_pages, strict=True)
                ],
            }

    @staticmethod
    def _operator_page_revision(page: _InteractivePage) -> str:
        # Navigation invalidates model observations. Operator census must still
        # name the exact native page generation without restoring model refs or
        # claiming that a protected model observation has occurred.
        if page.revision is not None:
            return page.revision
        return f"bo_unobserved_{page.navigation_epoch}_{page.control_epoch}"

    async def _operator_boundary_locations(self) -> list[dict[str, Any]]:
        # Caller holds the lifecycle lock after settling native action owners.
        private_values = await self._refresh_profile_output_values()
        if any(
            page.lifecycle not in {"active", "background", "closed", "crashed"}
            for page in self.pages.values()
        ):
            raise GuestControlFailure()
        pages = sorted(
            (page for page in self.pages.values() if page.lifecycle in {"active", "background"}),
            key=lambda page: page.page_id,
        )
        if len(pages) > 16:
            raise GuestControlFailure()
        return [
            {
                "page": {
                    "page_id": page.page_id,
                    "revision": self._operator_page_revision(page),
                    "control_epoch": page.control_epoch,
                },
                "origin": self._operator_page_origin(
                    getattr(page.page, "url", None), private_values
                ),
            }
            for page in pages
        ]

    @staticmethod
    def _operator_page_origin(url: object, private_values: tuple[str, ...]) -> str | None:
        if (
            type(url) is not str
            or len(url) > _MAX_URL_LENGTH
            or any(ord(char) <= 32 or 0xD800 <= ord(char) <= 0xDFFF for char in url)
        ):
            return None
        origin = _interactive_origin(url)
        if origin is None:
            return None
        origin = origin.removesuffix("/")
        if len(origin) > 2048 or any(value in origin for value in private_values):
            return None
        return origin

    def _operator_control_evidence(self) -> dict[str, Any]:
        return {
            "worker_instance": self.visual_worker_instance,
            "binding_sha256": self.control.binding_sha256,
            "control_epoch": self.control.epoch,
            "state": self.control.state,
            "settled_sequence": self.control.settled_sequence,
            "pending_sequence": self.control.pending_sequence,
            "fresh_observation_required": self.control.fresh_observation_required,
        }

    async def acquire_operator_control(
        self,
        *,
        binding_sha256: str,
        request_id: str,
        request_sha256: str,
        expected_epoch: int,
        expires_at_ms: int,
        maximum_until_ms: int,
        lease_until_ms: int,
        pages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        # Own the expected page tuple before waiting for model/cleanup owners.
        # A page changed by that work cannot inherit the operator's old consent.
        if type(pages) is not list or len(pages) > 32:
            raise GuestControlFailure()
        expected_pages: dict[str, tuple[str, int]] = {}
        for page in pages:
            if (
                type(page) is not dict
                or set(page) != {"page_id", "revision", "control_epoch"}
                or type(page["page_id"]) is not str
                or not page["page_id"]
                or type(page["revision"]) is not str
                or not page["revision"]
                or type(page["control_epoch"]) is not int
                or not 1 <= page["control_epoch"] < 2**53
                or page["page_id"] in expected_pages
            ):
                raise GuestControlFailure()
            expected_pages[page["page_id"]] = (page["revision"], page["control_epoch"])
        del pages

        def acquisition_evidence() -> dict[str, Any]:
            if self.control.acquisition_audit_json is None:
                raise GuestControlFailure()
            return {
                **self._operator_control_evidence(),
                "request_id": self.control.request_id,
                "request_sha256": self.control.request_sha256,
                "lease_until_ms": self.control.lease_wall_ms,
                "audit": json.loads(self.control.acquisition_audit_json),
            }

        self.control.request(
            binding_sha256=binding_sha256,
            request_id=request_id,
            request_sha256=request_sha256,
            expected_epoch=expected_epoch,
            expires_at_ms=expires_at_ms,
            maximum_until_ms=maximum_until_ms,
        )
        if self.control.state != "takeover_requested":
            return acquisition_evidence()
        try:
            async with asyncio.timeout(self.control.request_remaining_seconds()), self.lock:
                if self.control.state != "takeover_requested":
                    return acquisition_evidence()
                await self._settle_operator_control_owners()
                locations = await self._operator_boundary_locations()
                live_pages = {
                    page.page_id: (self._operator_page_revision(page), page.control_epoch)
                    for page in self.pages.values()
                    if page.lifecycle not in {"closed", "crashed"}
                }
                if live_pages != expected_pages or any(
                    page.lifecycle not in {"active", "background", "closed", "crashed"}
                    for page in self.pages.values()
                ):
                    raise GuestControlFailure()
                self.control.grant(request_id=request_id, lease_until_ms=lease_until_ms)
                self.control.acquisition_audit_json = json.dumps(
                    {
                        "request_id": request_id,
                        "phase": "acquired",
                        "control_epoch": self.control.epoch,
                        "locations": locations,
                    }
                )
                return acquisition_evidence()
        except BaseException:
            self.control.uncertain()
            raise

    async def handback_operator_control(self, *, request_id: str, epoch: int) -> dict[str, Any]:
        def handback_evidence() -> dict[str, Any]:
            if self.control.handback_audit_json is None:
                raise GuestControlFailure()
            return {
                **self._operator_control_evidence(),
                "audit": json.loads(self.control.handback_audit_json),
            }

        if not self.control.begin_handback(request_id=request_id, epoch=epoch):
            return handback_evidence()
        try:
            async with asyncio.timeout(5.0), self.lock:
                if self.control.state == "agent_controlled":
                    return handback_evidence()
                await self._settle_operator_control_owners()
                for state in self.pages.values():
                    state.clear_refs()
                    state.revision = None
                    state.last_observation_revision = None
                    state.control_epoch += 1
                    if state.visual_owner is not None:
                        state.visual_owner.invalidate()
                locations = await self._operator_boundary_locations()
                self.control.finish_handback()
                self.control.handback_audit_json = json.dumps(
                    {
                        "request_id": request_id,
                        "phase": "handed_back",
                        "control_epoch": self.control.epoch,
                        "locations": locations,
                    }
                )
                self.last_activity = asyncio.get_running_loop().time()
                return handback_evidence()
        except BaseException:
            self.control.uncertain()
            raise

    def _operator_control_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        """Census the actual native action, restoration and retirement owners."""

        if self.closing or self.close_requested.is_set() or self.context is None:
            raise GuestControlFailure()
        owners: list[asyncio.Task[Any]] = []
        if self.operator_frames.task is not None:
            owners.append(self.operator_frames.task)
        if self._operator_input_task is not None:
            owners.append(self._operator_input_task)
        if self.popup_cleanup_task is not None:
            owners.append(self.popup_cleanup_task)
        for task in self.session_cleanup_tasks.values():
            if task not in owners:
                owners.append(task)
        for state in self.pages.values():
            if state.lifecycle in {"provisional", "closing", "uncertain"} or (
                state.observation_cleanup_disposition == "uncertain"
                or state.visual_action_cleanup_disposition == "uncertain"
            ):
                raise GuestControlFailure()
            for task in (
                state.cleanup_task,
                state.limit_abort_task,
                state.unexpected_download_task,
                state.observation_cleanup_task,
                None if state.visual_owner is None else state.visual_owner.action_task,
                None if state.visual_owner is None else state.visual_owner.disarm_task,
            ):
                if task is not None and task not in owners:
                    owners.append(task)
        return tuple(owners)

    async def _settle_operator_control_owners(self) -> None:
        """Join owners, including work registered while an earlier owner settles."""

        deadline = asyncio.get_running_loop().time() + 5.0
        owners: list[asyncio.Task[Any]] = []
        while True:
            for task in self._operator_control_tasks():
                if task not in owners:
                    owners.append(task)
            if len(owners) > 4096:
                raise GuestControlFailure()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise GuestControlFailure()
            if owners:
                _, pending = await asyncio.wait(owners, timeout=remaining)
                if pending:
                    raise GuestControlFailure()
            # Existing popup settlement also gives already-enqueued callbacks
            # one scheduling turn. Recheck both owner membership and lifecycle
            # evidence after that turn; a newly registered owner cannot be lost.
            await asyncio.sleep(0)
            if all(task in owners for task in self._operator_control_tasks()):
                break
        if owners:
            failures: list[BaseException] = []
            for owner in owners:
                try:
                    result = owner.result()
                except asyncio.CancelledError as cancellation:
                    failure = RuntimeError(
                        "Browser cleanup was cancelled without proving quiescence."
                    )
                    failure.__cause__ = cancellation
                    failures.append(failure)
                except BaseException as failure:
                    failures.append(failure)
                else:
                    if result is False:
                        failures.append(GuestControlFailure())
                    elif isinstance(result, tuple) and result:
                        if all(isinstance(item, BaseException) for item in result):
                            failures.extend(result)
                        else:
                            failures.append(GuestControlFailure())
            if failures:
                raise BaseExceptionGroup("Browser control cleanup did not settle.", failures)

    async def _ensure_configuration(self, request: _InteractiveRequest) -> None:
        if self.context is None:
            if request.operation == "profile_restore":
                return
            if request.operation != "navigate":
                raise _GuestFailure("browser_crash")
            await self._ensure_context(None)
        material = {
            "limits": asdict(request.limits),
            "visual_policy": request.visual_policy,
            "multi_page": request.multi_page,
            "popup_policy": asdict(request.popup_policy),
        }
        fingerprint = hashlib.sha256(
            b"cayu-browser-page-configuration-v1\0"
            + json.dumps(
                material,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.configuration_fingerprint is not None:
            if self.configuration_fingerprint != fingerprint:
                raise _GuestFailure("incompatible_browser")
            return
        if self.context is None:
            raise _GuestFailure("browser_crash")
        try:
            await self.context.add_init_script(_interactive_popup_guard(self.popup_guard_token))
            await self.context.route("**/*", self._route_interactive_request)
            if self.profile_allowed_origins is not None:
                await self.context.route_web_socket("**/*", self._route_profile_web_socket)
        except asyncio.CancelledError:
            self._mark_popup_guard_uncertain()
            raise
        except Exception as exc:
            self._mark_popup_guard_uncertain()
            raise _GuestFailure("browser_crash") from exc
        self.configuration_fingerprint = fingerprint
        self.configuration_limits = request.limits
        self.configuration_multi_page = request.multi_page
        self.configuration_popup_policy = request.popup_policy
        self.configuration_visual_policy = request.visual_policy

    async def _execute_locked(self, request: _InteractiveRequest) -> dict[str, Any]:
        """Execute while the caller owns the daemon lifecycle lock."""

        self.idle_timeout_seconds = request.limits.idle_timeout_seconds
        if request.operation == "profile_restore":
            if (
                request.profile_restore_state is None
                or request.profile_timeout_seconds is None
                or request.profile_allowed_origins is None
                or request.profile_plaintext_limit is None
                or self.context is not None
                or self.pages
            ):
                raise _GuestFailure("incompatible_browser")
            try:
                async with asyncio.timeout(request.profile_timeout_seconds):
                    await self._ensure_context(request.profile_restore_state)
                self.profile_output_values = _bounded_interactive_profile_private_values(
                    (),
                    _interactive_profile_private_values(request.profile_restore_state),
                    maximum_bytes=request.profile_plaintext_limit,
                )
                self.profile_allowed_origins = request.profile_allowed_origins
                self.profile_plaintext_limit = request.profile_plaintext_limit
                self.profile_timeout_seconds = request.profile_timeout_seconds
                await self._ensure_configuration(request)
            except _GuestFailure:
                raise
            except TimeoutError as exc:
                raise _GuestFailure("timeout") from exc
            except Exception as exc:
                raise _GuestFailure("incompatible_browser") from exc
            return {
                "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
                "worker_version": INTERACTIVE_WORKER_VERSION,
                "playwright_version": PLAYWRIGHT_VERSION,
                "kind": "profile_restore",
                "allocation_disposition": "live",
                "profile_restored": True,
            }
        if request.operation == "profile_checkpoint":
            if (
                self.context is None
                or request.profile_capture_limit is None
                or request.profile_timeout_seconds is None
                or self.profile_allowed_origins is None
                or request.profile_allowed_origins != self.profile_allowed_origins
            ):
                raise _GuestFailure("session_closed")
            try:
                async with asyncio.timeout(request.profile_timeout_seconds):
                    profile_state = await self.context.storage_state(indexed_db=False)
                owned_state = _validate_interactive_profile_state(
                    profile_state,
                    maximum_bytes=request.profile_capture_limit,
                    allowed_origins=self.profile_allowed_origins,
                )
            except _GuestFailure:
                raise
            except TimeoutError as exc:
                raise _GuestFailure("timeout") from exc
            except Exception as exc:
                raise _GuestFailure("browser_crash") from exc
            return {
                "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
                "worker_version": INTERACTIVE_WORKER_VERSION,
                "playwright_version": PLAYWRIGHT_VERSION,
                "kind": "profile_checkpoint",
                "allocation_disposition": "live",
                "profile_state": owned_state,
            }
        if self.profile_output_values is not None and request.operation in {
            "screenshot",
            "download",
            "observe_visual",
            "click_visual_target",
            "click_visual_point",
        }:
            raise _GuestFailure("policy_denied")
        delta = _InteractivePageDelta()
        delta.closed_page_ids.update(self.pending_closed_page_ids)
        delta.crashed_page_ids.update(self.pending_crashed_page_ids)
        self.pending_closed_page_ids.clear()
        self.pending_crashed_page_ids.clear()
        self.active_request = request
        self.active_delta = delta
        state: _InteractivePage | None = None
        created_page = False
        try:
            if (
                request.operation in {"observe_visual", "click_visual_target", "click_visual_point"}
                and request.visual_policy is None
            ):
                raise _GuestFailure("visual_mode_disabled")
            await self._expire_background_pages(request.limits, delta=delta)
            if request.operation == "close":
                self.closing = True
                cleanup_ok = await self.close(
                    timeout_seconds=max(1.0, min(10.0, request.limits.max_wait_ms / 1000))
                )
                self.close_requested.set()
                if not cleanup_ok:
                    return _interactive_error_payload(_GuestFailure("cleanup_failed"))
                return _interactive_closed_payload()
            if request.operation == "list_pages":
                if self.total_operations >= request.limits.max_operations:
                    raise _GuestFailure("resource_exhausted")
                self.total_operations += 1
                return _interactive_success_payload(
                    None,
                    page_set=self._page_set_payload(),
                    page_delta=self._page_delta_payload(delta),
                )
            state = self.pages.get(request.page_id or "")
            if request.operation == "navigate":
                if request.page_id is None or request.url is None:
                    raise _GuestFailure("incompatible_browser")
                if state is not None or self.pages:
                    raise _GuestFailure("resource_exhausted")
                if request.limits.max_total_page_creations < 1:
                    raise _GuestFailure("resource_exhausted")
                try:
                    page = await self.context.new_page()
                except Exception:
                    return await self._retire_failed_allocation(
                        _GuestFailure("browser_crash"),
                        request,
                    )
                self.total_page_creations = 1
                state = _InteractivePage(
                    page=page,
                    session_id=request.session_id,
                    page_id=request.page_id,
                    creation_epoch=1,
                    control_epoch=1,
                    lifecycle="active",
                    created_monotonic=asyncio.get_running_loop().time(),
                    public_url=request.url,
                )
                self.pages[request.page_id] = state
                self.active_page_id = request.page_id
                delta.created_page_ids.add(request.page_id)
                delta.admitted_page_ids.add(request.page_id)
                created_page = True
            elif state is None:
                raise _GuestFailure("session_closed")
            if request.operation == "close_page":
                return await self._close_page(state, request, delta)
            if request.operation == "switch_page":
                return await self._switch_page(state, request, delta)
            if state.lifecycle not in {"active", "admitted", "background"}:
                raise _GuestFailure("session_closed")
            if request.operation != "navigate" and self.active_page_id != state.page_id:
                raise _GuestFailure("missing_element")
            if request.operation not in {"navigate", "observe", "observe_visual"}:
                if (
                    state.revision is None
                    or request.expected_revision != state.revision
                    or request.expected_control_epoch != state.control_epoch
                ):
                    raise _GuestFailure("incompatible_browser")
                if request.ref is not None and request.ref not in state.refs:
                    raise _GuestFailure("missing_element")
            if (
                state.operation_count >= request.limits.max_operations_per_page
                or self.total_operations >= request.limits.max_operations
            ):
                raise _GuestFailure("resource_exhausted")
            state.operation_count += 1
            self.total_operations += 1
            state.last_operation_id_sha256 = hashlib.sha256(
                request.operation_id.encode("utf-8")
            ).hexdigest()
            if request.operation not in {"navigate", "observe", "observe_visual"}:
                state.control_epoch += 1
            if request.operation in {"screenshot", "download", "observe_visual"} and (
                state.artifact_count >= request.limits.max_artifacts_per_page
                or self.total_artifacts >= request.limits.max_total_artifacts
            ):
                raise _GuestFailure("resource_exhausted")
            if created_page:
                await self._configure_page(state, request.limits)
            try:
                response = await self._execute_page(state, request)
            except BaseException as primary_failure:
                if (
                    state.observation_cleanup_disposition is not None
                    or state.visual_action_cleanup_disposition is not None
                ):
                    # Visual cleanup already joined the whole browser owner.
                    # Do not restart popup settlement or spend a second close budget.
                    raise
                try:
                    await self._settle_operation_popups(
                        request,
                        delta,
                        primary_failure=primary_failure,
                    )
                except BaseException as settlement_failure:
                    if not isinstance(primary_failure, Exception):
                        raise primary_failure from _browser_cleanup_evidence(
                            primary_failure.__cause__, (settlement_failure,)
                        )
                    raise settlement_failure from primary_failure
                raise
            await self._settle_operation_popups(request, delta)
            if delta.refused:
                reason = delta.refused[0]["reason"]
                code = "resource_exhausted" if reason == "capacity_refused" else "policy_denied"
                return _interactive_error_payload(
                    _GuestFailure(code, allocation_disposition="live"),
                    page_set=self._page_set_payload(),
                    page_delta=self._page_delta_payload(delta),
                )
            return self._with_page_evidence(response, delta)
        except VisualGuestFailure as exc:
            return _interactive_error_payload(
                _GuestFailure(exc.code, allocation_disposition="live"),
                page_set=self._page_set_payload() if self.pages else None,
                page_delta=self._page_delta_payload(delta),
            )
        except _GuestFailure as exc:
            if state is not None and (
                state.observation_cleanup_disposition is not None
                or state.visual_action_cleanup_disposition is not None
            ):
                return _interactive_error_payload(exc)
            if state is not None and (created_page or state.limit_exceeded):
                return await self._retire_failed_allocation(exc, request)
            return _interactive_error_payload(
                exc,
                page_set=self._page_set_payload() if self.pages else None,
                page_delta=self._page_delta_payload(delta),
            )
        except Exception as exc:
            if state is None:
                return _interactive_error_payload(_GuestFailure("browser_crash"))
            if state.observation_cleanup_disposition is not None:
                return _interactive_error_payload(
                    _GuestFailure(
                        "timeout" if isinstance(exc, TimeoutError) else "cleanup_failed",
                        allocation_disposition=state.observation_cleanup_disposition,
                    )
                )
            if state.denied_code is not None:
                failure = _GuestFailure(state.denied_code)
            elif state.limit_exceeded:
                failure = _GuestFailure(state.limit_error_code)
            else:
                try:
                    browser_connected = bool(self.browser.is_connected())
                except Exception:
                    browser_connected = False
                try:
                    page_open = not state.page.is_closed()
                except Exception:
                    page_open = False
                failure = _interactive_runtime_failure(
                    request.operation,
                    exc,
                    browser_connected=browser_connected,
                    page_open=page_open,
                )
            if created_page or state.limit_exceeded:
                return await self._retire_failed_allocation(failure, request)
            return _interactive_error_payload(
                failure,
                page_set=self._page_set_payload(),
                page_delta=self._page_delta_payload(delta),
            )
        finally:
            self.active_request = None
            self.active_delta = None
            self.popup_effect_opener_page_id = None
            self.popup_effect_opener_origin = None

    def _page_set_payload(self) -> dict[str, Any]:
        pages = []
        for state in sorted(self.pages.values(), key=lambda item: item.creation_epoch):
            url = state.public_url
            if url is None:
                raw_url = getattr(state.page, "url", None)
                if raw_url == "about:blank":
                    url = "about:blank"
                elif _interactive_origin(raw_url) is not None:
                    url = raw_url
            if type(url) is str and len(url.encode("utf-8", errors="replace")) > _MAX_URL_LENGTH:
                url = _interactive_origin(url)
            title = state.title
            if self.profile_output_values is not None:
                # Summaries also cover unobserved/background pages and error
                # paths, where current storage cannot be safely refreshed.
                # Only the protected observation publishes page text/URLs.
                url = None
                title = None
            if title is not None and len(title.encode("utf-8", errors="replace")) > (
                _INTERACTIVE_MAX_TITLE_ENVELOPE_BYTES
            ):
                title = title.encode("utf-8", errors="replace")[
                    :_INTERACTIVE_MAX_TITLE_ENVELOPE_BYTES
                ].decode("utf-8", errors="ignore")
            pages.append(
                {
                    "page_id": state.page_id,
                    "lifecycle": state.lifecycle,
                    "creation_epoch": state.creation_epoch,
                    "control_epoch": state.control_epoch,
                    "opener_page_id": state.opener_page_id,
                    "creating_operation_id_sha256": state.creating_operation_id_sha256,
                    "revision": state.revision,
                    "url": url,
                    "title": title,
                    "load_state": (
                        "failed"
                        if state.lifecycle in {"crashed", "uncertain"}
                        else "loading"
                        if state.lifecycle == "provisional"
                        else "loaded"
                    ),
                    "access_state": (
                        "blocked"
                        if state.access_evidence is not None
                        else "unknown"
                        if state.lifecycle in {"provisional", "uncertain"}
                        else "available"
                    ),
                    "last_observation_revision": state.last_observation_revision,
                    "last_operation_id_sha256": state.last_operation_id_sha256,
                    "terminal_reason": state.terminal_reason,
                    "operation_count": state.operation_count,
                    "observation_count": state.observation_count,
                    "ref_count": state.ref_count,
                    "request_count": state.request_count,
                    "artifact_count": state.artifact_count,
                }
            )
        return {
            "session_id": self.session_id,
            "active_page_id": self.active_page_id,
            "pages": pages,
            "total_page_creations": self.total_page_creations,
            "total_operations": self.total_operations,
            "total_observations": self.total_observations,
            "total_refs": self.total_refs,
            "total_requests": self.total_requests,
            "total_artifacts": self.total_artifacts,
            "cleanup_operation_count": self.cleanup_operation_count,
        }

    @staticmethod
    def _page_delta_payload(delta: _InteractivePageDelta) -> dict[str, Any]:
        return {
            "created_page_ids": sorted(delta.created_page_ids),
            "admitted_page_ids": sorted(delta.admitted_page_ids),
            "closed_page_ids": sorted(delta.closed_page_ids),
            "crashed_page_ids": sorted(delta.crashed_page_ids),
            "refused": sorted(delta.refused, key=lambda item: item["page_id"]),
        }

    def _with_page_evidence(
        self,
        response: dict[str, Any],
        delta: _InteractivePageDelta,
    ) -> dict[str, Any]:
        return {
            **response,
            "page_set": self._page_set_payload(),
            "page_delta": self._page_delta_payload(delta),
        }

    def _mark_page_closed(self, state: _InteractivePage, reason: str) -> None:
        if state.lifecycle in {"closed", "crashed", "uncertain"}:
            return
        closing = state.lifecycle == "closing"
        state.lifecycle = "closed"
        state.terminal_reason = reason
        if not closing:
            state.control_epoch += 1
        state.revision = None
        state.clear_refs()
        if self.active_page_id == state.page_id:
            self.active_page_id = None
            self._select_remaining_active()
        if self.active_delta is not None:
            self.active_delta.closed_page_ids.add(state.page_id)
        else:
            self.pending_closed_page_ids.add(state.page_id)

    def _mark_page_crashed(self, state: _InteractivePage) -> None:
        if state.lifecycle in {"closed", "crashed"}:
            return
        state.lifecycle = "crashed"
        state.terminal_reason = "browser_crash"
        state.control_epoch += 1
        state.revision = None
        state.clear_refs()
        if self.active_delta is not None:
            self.active_delta.crashed_page_ids.add(state.page_id)
        else:
            self.pending_crashed_page_ids.add(state.page_id)
        if self.active_page_id == state.page_id:
            self.active_page_id = None
            self._select_remaining_active()

    def _select_remaining_active(self) -> _InteractivePage | None:
        candidate = next(
            (
                state
                for state in sorted(self.pages.values(), key=lambda item: item.creation_epoch)
                if state.lifecycle in {"admitted", "background"}
            ),
            None,
        )
        if candidate is None:
            return None
        candidate.lifecycle = "active"
        candidate.background_since = None
        candidate.control_epoch += 1
        candidate.revision = f"br_{secrets.token_hex(16)}"
        candidate.clear_refs()
        self.active_page_id = candidate.page_id
        return candidate

    async def _close_page(
        self,
        state: _InteractivePage,
        request: _InteractiveRequest,
        delta: _InteractivePageDelta,
    ) -> dict[str, Any]:
        if state.lifecycle in {"closed", "crashed"}:
            delta.closed_page_ids.add(state.page_id)
            return _interactive_success_payload(
                None,
                page_set=self._page_set_payload(),
                page_delta=self._page_delta_payload(delta),
            )
        if not self._reserve_page_cleanup(request.limits):
            raise _GuestFailure("resource_exhausted")
        state.lifecycle = "closing"
        state.control_epoch += 1
        state.last_operation_id_sha256 = hashlib.sha256(
            request.operation_id.encode("utf-8")
        ).hexdigest()
        state.revision = None
        state.clear_refs()
        cleanup_ok = await self._await_page_close(
            state,
            timeout_seconds=max(0.001, request.limits.max_wait_ms / 1000),
        )
        if not cleanup_ok:
            state.lifecycle = "uncertain"
            state.terminal_reason = "cleanup_failed"
            if self.active_page_id == state.page_id:
                self.active_page_id = None
            return _interactive_error_payload(
                _GuestFailure("cleanup_failed", allocation_disposition="uncertain"),
                page_set=self._page_set_payload(),
                page_delta=self._page_delta_payload(delta),
            )
        state.lifecycle = "closed"
        state.terminal_reason = "closed_by_model"
        delta.closed_page_ids.add(state.page_id)
        if self.active_page_id == state.page_id:
            self.active_page_id = None
            self._select_remaining_active()
        return _interactive_success_payload(
            None,
            page_set=self._page_set_payload(),
            page_delta=self._page_delta_payload(delta),
        )

    async def _switch_page(
        self,
        state: _InteractivePage,
        request: _InteractiveRequest,
        delta: _InteractivePageDelta,
    ) -> dict[str, Any]:
        if state.lifecycle not in {"active", "admitted", "background"}:
            raise _GuestFailure("session_closed")
        if (
            state.operation_count >= request.limits.max_operations_per_page
            or self.total_operations >= request.limits.max_operations
        ):
            raise _GuestFailure("resource_exhausted")
        current = self.pages.get(self.active_page_id or "")
        now = asyncio.get_running_loop().time()
        if current is not None and current.page_id != state.page_id:
            current.lifecycle = "background"
            current.background_since = now
            current.control_epoch += 1
            current.revision = f"br_{secrets.token_hex(16)}"
            current.clear_refs()
        state.lifecycle = "active"
        state.background_since = None
        state.control_epoch += 1
        state.revision = None
        state.clear_refs()
        state.operation_count += 1
        self.total_operations += 1
        state.last_operation_id_sha256 = hashlib.sha256(
            request.operation_id.encode("utf-8")
        ).hexdigest()
        self.active_page_id = state.page_id
        observation = await self._observe_page(state, request.limits)
        return _interactive_success_payload(
            observation,
            page_set=self._page_set_payload(),
            page_delta=self._page_delta_payload(delta),
            profile_output_protected=self.profile_output_values is not None,
        )

    async def _observe_page(
        self,
        state: _InteractivePage,
        limits: _InteractiveLimits,
        *,
        visual_policy: dict[str, Any] | None = None,
        visual_artifacts: list[bytes] | None = None,
    ) -> dict[str, Any]:
        if (
            state.observation_count >= limits.max_observations_per_page
            or self.total_observations >= limits.max_total_observations
        ):
            raise _GuestFailure("resource_exhausted")
        navigation_epoch = state.navigation_epoch
        previous_observation_revision = state.last_observation_revision
        try:
            observation = await _interactive_observation(
                state,
                limits,
                browser_version=self.browser_version,
                visual_policy=visual_policy,
                visual_artifacts=visual_artifacts,
            )
        except BaseException as primary:
            if state.observation_cleanup_disposition is None:
                raise
            self.closing = True
            self.close_requested.set()

            async def retire_observation() -> tuple[BaseException, ...]:
                try:
                    if await self.close(timeout_seconds=5.0):
                        state.observation_cleanup_disposition = "retired"
                        return ()
                except BaseException as error:
                    return (error,)
                return (RuntimeError("Browser observation cleanup remains uncertain."),)

            outcome = await _await_browser_cleanup_resisting_cancellation(
                asyncio.create_task(retire_observation())
            )
            signal = outcome.cancellation
            if signal is None and not isinstance(primary, Exception):
                signal = primary
            cause = _browser_cleanup_evidence(
                primary.__cause__
                if signal is primary or isinstance(primary, asyncio.CancelledError)
                else primary,
                outcome.errors,
            )
            if signal is not None:
                # Keep earlier restoration evidence when the primary is itself
                # cancellation (the generic evidence helper excludes that signal).
                if signal is primary and not outcome.errors:
                    raise primary
                raise signal from cause
            raise _GuestFailure(
                "timeout" if isinstance(primary, TimeoutError) else "cleanup_failed",
                allocation_disposition=state.observation_cleanup_disposition,
            ) from cause
        try:
            observation, _protected = await self._protect_profile_observation(observation)
            if (
                state.access_evidence is None and state.navigation_epoch != navigation_epoch
            ) or state.revision != observation.get("revision"):
                raise _GuestFailure("browser_crash")
        except BaseException:
            state.revision = None
            state.clear_refs()
            state.last_observation_revision = previous_observation_revision
            raise
        if not observation["refs"]:
            state.clear_refs()
        ref_count = len(state.refs)
        if (
            state.ref_count + ref_count > limits.max_refs_per_page
            or self.total_refs + ref_count > limits.max_total_refs
        ):
            state.revision = None
            state.clear_refs()
            state.limit_exceeded = True
            state.limit_error_code = "resource_exhausted"
            raise _GuestFailure("resource_exhausted")
        state.observation_count += 1
        state.ref_count += ref_count
        self.total_observations += 1
        self.total_refs += ref_count
        state.title = observation.get("title")
        return observation

    async def _settle_operation_popups(
        self,
        request: _InteractiveRequest,
        delta: _InteractivePageDelta,
        *,
        primary_failure: BaseException | None = None,
    ) -> None:
        # Playwright can enqueue the popup callback immediately before the
        # originating action resolves.  Give that already-created callback one
        # scheduling turn so this operation's exact delta cannot omit it.
        await asyncio.sleep(0)
        settled_page_ids: set[str] = set()
        while pending_page_ids := sorted(delta.created_page_ids - settled_page_ids):
            page_id = pending_page_ids[0]
            settled_page_ids.add(page_id)
            state = self.pages.get(page_id)
            if state is None or state.lifecycle != "provisional":
                continue
            admission_failure: _GuestFailure | None = None
            try:
                failure = _interactive_page_failure(state)
                if failure is not None:
                    raise failure
                if not state.configured:
                    await self._configure_page(state, request.limits)
                navigation_deadline = asyncio.get_running_loop().time() + max(
                    0.001,
                    request.limits.max_wait_ms / 1000,
                )
                raw_url = getattr(state.page, "url", "about:blank")
                while raw_url == "about:blank" and state.staged_initial_url is None:
                    remaining = navigation_deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise _GuestFailure("fetch_failed")
                    await asyncio.sleep(min(0.01, remaining))
                    raw_url = getattr(state.page, "url", "about:blank")
                if state.staged_initial_url is not None:
                    staged_initial_url = state.staged_initial_url
                    state.staged_initial_url = None
                    try:
                        await state.page.goto(
                            staged_initial_url,
                            wait_until="domcontentloaded",
                            timeout=max(
                                1,
                                int(
                                    1_000
                                    * max(
                                        0.001,
                                        navigation_deadline - asyncio.get_running_loop().time(),
                                    )
                                ),
                            ),
                        )
                    except Exception:
                        failure = _interactive_page_failure(state)
                        if failure is None:
                            raise
                        raise failure from None
                wait_for_load_state = getattr(state.page, "wait_for_load_state", None)
                if callable(wait_for_load_state):
                    await wait_for_load_state(
                        "domcontentloaded",
                        timeout=max(
                            1,
                            int(
                                1_000
                                * max(
                                    0.001,
                                    navigation_deadline - asyncio.get_running_loop().time(),
                                )
                            ),
                        ),
                    )
                raw_url = getattr(state.page, "url", "about:blank")
                failure = _interactive_page_failure(state)
                if failure is not None:
                    raise failure
                if raw_url == "about:blank" or not self._popup_destination_allowed(state, raw_url):
                    raise _GuestFailure("destination_denied")
                state.public_url = (
                    raw_url if _interactive_origin(raw_url) is not None else state.public_url
                )
                state.lifecycle = "background"
                state.background_since = asyncio.get_running_loop().time()
                state.revision = f"br_{secrets.token_hex(16)}"
                delta.admitted_page_ids.add(page_id)
            except asyncio.CancelledError:
                self._schedule_popup_cleanup(state.page)
                raise
            except _GuestFailure as exc:
                refusal_reason = (
                    "capacity_refused"
                    if exc.code
                    in {"oversized_response", "oversized_snapshot", "resource_exhausted"}
                    else "policy_denied"
                    if exc.code == "policy_denied"
                    else "destination_denied"
                )
                if exc.code not in {
                    "destination_denied",
                    "redirect_denied",
                    "fetch_failed",
                    "policy_denied",
                }:
                    admission_failure = exc
                opener_page_id = state.opener_page_id or self.active_page_id or state.page_id
                self._record_popup_refusal(
                    page_id=state.page_id,
                    opener_page_id=opener_page_id,
                    reason=refusal_reason,
                )
                state.lifecycle = "closing"
                state.control_epoch += 1
                state.revision = None
                state.clear_refs()
                if not self._reserve_page_cleanup(request.limits):
                    state.lifecycle = "uncertain"
                    state.terminal_reason = "cleanup_failed"
                    self._schedule_popup_cleanup(state.page, retire=True)
                else:
                    cleanup_ok = await self._await_page_close(
                        state,
                        timeout_seconds=max(0.001, request.limits.max_wait_ms / 1000),
                        close_provisional_target=True,
                    )
                    if cleanup_ok:
                        state.lifecycle = "closed"
                        state.terminal_reason = refusal_reason
                        delta.closed_page_ids.add(state.page_id)
                    else:
                        state.lifecycle = "uncertain"
                        state.terminal_reason = "cleanup_failed"
                if admission_failure is not None:
                    if state.lifecycle == "closed":
                        raise
                    raise _GuestFailure(
                        "cleanup_failed",
                        allocation_disposition="uncertain",
                    ) from admission_failure
            except Exception as exc:
                admission_failure = _GuestFailure("browser_crash")
                state.lifecycle = "closing"
                state.control_epoch += 1
                state.revision = None
                state.clear_refs()
                if not self._reserve_page_cleanup(request.limits):
                    state.lifecycle = "uncertain"
                    state.terminal_reason = "cleanup_failed"
                    self._schedule_popup_cleanup(state.page, retire=True)
                else:
                    cleanup_ok = await self._await_page_close(
                        state,
                        timeout_seconds=max(0.001, request.limits.max_wait_ms / 1000),
                        close_provisional_target=True,
                    )
                    if cleanup_ok:
                        state.lifecycle = "closed"
                        state.terminal_reason = "browser_crash"
                        delta.closed_page_ids.add(state.page_id)
                    else:
                        state.lifecycle = "uncertain"
                        state.terminal_reason = "cleanup_failed"
                if state.lifecycle == "closed":
                    raise admission_failure from exc
                raise _GuestFailure(
                    "cleanup_failed",
                    allocation_disposition="uncertain",
                ) from exc
        cleanup = self.popup_cleanup_task
        if cleanup is not None:
            try:
                async with asyncio.timeout(max(0.001, request.limits.max_wait_ms / 1000)):
                    cleanup_ok = await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, Exception):
                cleanup_ok = False
            if not cleanup_ok:
                self.closing = True
                self.close_after_response = True
                raise _GuestFailure("cleanup_failed", allocation_disposition="uncertain")
        if self.closing:
            # A guard failure can fence the allocation without ever creating a
            # popup cleanup task. Retirement always requires the whole owner to
            # settle before the host is allowed to release allocation capacity.
            authoritative_failure = (
                primary_failure if isinstance(primary_failure, _GuestFailure) else None
            )
            if authoritative_failure is None:
                denied = next(
                    (page.denied_code for page in self.pages.values() if page.denied_code),
                    None,
                )
                access = next(
                    (
                        page.access_evidence
                        for page in self.pages.values()
                        if page.access_evidence is not None
                    ),
                    None,
                )
                if denied is not None:
                    authoritative_failure = _GuestFailure(denied)
                elif access is not None:
                    authoritative_failure = _GuestFailure("access_blocked", access=access)
            cleanup_ok = await self.close(
                timeout_seconds=max(1.0, min(10.0, request.limits.max_wait_ms / 1000))
            )
            if cleanup_ok and authoritative_failure is not None:
                raise _GuestFailure(
                    authoritative_failure.code,
                    status_code=authoritative_failure.status_code,
                    effective_origin=authoritative_failure.effective_origin,
                    access=authoritative_failure.access,
                    allocation_disposition="retired",
                ) from primary_failure
            raise _GuestFailure(
                "resource_exhausted" if cleanup_ok else "cleanup_failed",
                allocation_disposition="retired" if cleanup_ok else "uncertain",
            )

    async def _expire_background_pages(
        self,
        limits: _InteractiveLimits,
        *,
        delta: _InteractivePageDelta | None,
    ) -> None:
        now = asyncio.get_running_loop().time()
        for state in tuple(self.pages.values()):
            if (
                state.lifecycle != "background"
                or state.background_since is None
                or now - state.background_since < limits.max_background_lifetime_seconds
            ):
                continue
            state.lifecycle = "closing"
            state.control_epoch += 1
            state.revision = None
            state.clear_refs()
            if not self._reserve_page_cleanup(limits):
                state.lifecycle = "uncertain"
                state.terminal_reason = "cleanup_failed"
                cleanup_ok = await self._retire_limited_allocation(timeout_seconds=5.0)
                raise _GuestFailure(
                    "resource_exhausted" if cleanup_ok else "cleanup_failed",
                    allocation_disposition="retired" if cleanup_ok else "uncertain",
                )
            cleanup_ok = await self._await_page_close(state, timeout_seconds=1.0)
            if not cleanup_ok:
                state.lifecycle = "uncertain"
                state.terminal_reason = "cleanup_failed"
                raise _GuestFailure(
                    "cleanup_failed",
                    allocation_disposition="uncertain",
                )
            else:
                state.lifecycle = "closed"
                state.terminal_reason = "background_expired"
                if delta is not None:
                    delta.closed_page_ids.add(state.page_id)

    def _reserve_page_cleanup(self, limits: _InteractiveLimits) -> bool:
        if self.cleanup_operation_count >= limits.max_page_cleanup_operations:
            return False
        self.cleanup_operation_count += 1
        return True

    async def _await_page_close(
        self,
        state: _InteractivePage,
        *,
        timeout_seconds: float,
        close_provisional_target: bool = False,
    ) -> bool:
        task = state.cleanup_task
        if task is not None and task.done():
            try:
                task.result()
            except (asyncio.CancelledError, Exception):
                state.cleanup_task = None
                task = None
            else:
                return True
        if task is None:
            task = asyncio.create_task(
                self._close_provisional_page_target(state)
                if close_provisional_target and state.cdp is not None
                else state.page.close()
            )
            state.cleanup_task = task

        async def settle() -> tuple[BaseException, ...]:
            done, _ = await asyncio.wait({task}, timeout=max(0.001, timeout_seconds))
            if not done:
                return (TimeoutError("Browser page cleanup did not settle within its bound."),)
            try:
                task.result()
            except asyncio.CancelledError as exc:
                failure = RuntimeError("Browser page cleanup was cancelled unexpectedly.")
                failure.__cause__ = exc
                return (failure,)
            except Exception as exc:
                return (exc,)
            return ()

        outcome = await _await_browser_cleanup_resisting_cancellation(asyncio.create_task(settle()))
        if outcome.cancellation is not None:
            cause = _browser_cleanup_evidence(None, outcome.errors)
            if cause is None:
                raise outcome.cancellation
            raise outcome.cancellation from cause
        return not outcome.errors

    @staticmethod
    async def _close_provisional_page_target(state: _InteractivePage) -> None:
        target_info = await state.cdp.send("Target.getTargetInfo")
        if type(target_info) is not dict or type(target_info.get("targetInfo")) is not dict:
            raise _GuestFailure("browser_crash")
        target_id = target_info["targetInfo"].get("targetId")
        if type(target_id) is not str or not target_id:
            raise _GuestFailure("browser_crash")
        try:
            closed = await state.cdp.send("Target.closeTarget", {"targetId": target_id})
        except asyncio.CancelledError:
            raise
        except Exception:
            # Closing the target that owns this CDP session can terminate the
            # channel before its acknowledgement is delivered.  Accept only
            # positive readback from the exact Playwright page in that case.
            for _ in range(25):
                try:
                    if state.page.is_closed():
                        return
                except Exception:
                    break
                await asyncio.sleep(0.01)
            raise
        if type(closed) is not dict or closed.get("success") is not True:
            raise _GuestFailure("browser_crash")

    def _schedule_response_limit_abort(self, state: _InteractivePage) -> None:
        if state.limit_abort_task is None:
            state.limit_abort_task = asyncio.create_task(self._abort_limited_page(state))

    def _schedule_popup_limit_abort(self, state: _InteractivePage, popup: Any) -> None:
        """Retire one allocation through one owned task after any extra page appears."""

        del popup
        if not state.limit_exceeded:
            state.limit_exceeded = True
            state.limit_error_code = "resource_exhausted"
        if state.limit_abort_task is None:
            state.limit_abort_task = asyncio.create_task(self._close_context_after_limit_abort())

    async def _abort_limited_page(self, state: _InteractivePage) -> bool:
        current_task = asyncio.current_task()
        cancellation_requests_before = 0 if current_task is None else current_task.cancelling()
        cancellation_pending_before = bool(
            current_task is not None and getattr(current_task, "_must_cancel", False)
        )
        try:
            await state.page.close()
            return True
        except asyncio.CancelledError:
            owner_cancelled = bool(
                current_task is not None
                and (
                    cancellation_pending_before
                    or current_task.cancelling() > cancellation_requests_before
                )
            )
            context_closed = await self._close_context_after_limit_abort()
            if owner_cancelled:
                raise
            return context_closed
        except Exception:
            return await self._close_context_after_limit_abort()

    async def _close_context_after_limit_abort(self) -> bool:
        try:
            if self.context is not None:
                await self.context.close()
                self.context = None
            return True
        except asyncio.CancelledError:
            return False
        except Exception:
            return False

    async def _retire_limited_allocation(
        self,
        *,
        timeout_seconds: float,
    ) -> bool:
        self.closing = True
        self.close_after_response = True
        return await self.close(timeout_seconds=timeout_seconds)

    async def _retire_failed_allocation(
        self,
        failure: _GuestFailure,
        request: _InteractiveRequest,
    ) -> dict[str, Any]:
        """Acknowledge failure only after the provisional allocation is quiescent."""

        cleanup_ok = await self._retire_limited_allocation(
            timeout_seconds=max(1.0, min(10.0, request.limits.max_wait_ms / 1000)),
        )
        if not cleanup_ok:
            return _interactive_error_payload(_GuestFailure("cleanup_failed"))
        return _interactive_error_payload(
            _GuestFailure(
                failure.code,
                status_code=failure.status_code,
                access=failure.access,
                effective_origin=failure.effective_origin,
                allocation_disposition="retired",
            )
        )

    def _state_for_page(self, page: Any) -> _InteractivePage | None:
        return next((state for state in self.pages.values() if state.page == page), None)

    @staticmethod
    def _current_page_origin(state: _InteractivePage) -> str | None:
        """Resolve opener policy from the live page before retained safe metadata."""

        try:
            current = _interactive_origin(state.page.url)
        except Exception:
            current = None
        return current or _interactive_origin(state.public_url)

    def _popup_policy_refusal(
        self,
        opener: _InteractivePage,
        request: _InteractiveRequest | None,
        *,
        opener_origin: str | None,
    ) -> str | None:
        if request is None or not request.multi_page:
            return "policy_denied"
        policy = request.popup_policy
        if policy.mode == "deny":
            return "policy_denied"
        if request.operation not in policy.allowed_operations:
            return "operation_refused"
        if self.active_page_id != opener.page_id or opener.lifecycle != "active":
            return "operation_refused"
        if opener_origin is None:
            return "policy_denied"
        if policy.allowed_opener_origins and opener_origin not in policy.allowed_opener_origins:
            return "policy_denied"
        return None

    def _popup_creation_allowance(
        self,
        opener: _InteractivePage,
        request: _InteractiveRequest,
        *,
        opener_origin: str | None,
    ) -> tuple[int, str | None]:
        refusal = self._popup_policy_refusal(
            opener,
            request,
            opener_origin=opener_origin,
        )
        if refusal is not None:
            return 0, refusal
        limits = request.limits
        live_pages = sum(
            page.lifecycle in {"provisional", "admitted", "active", "background"}
            for page in self.pages.values()
        )
        provisional_pages = sum(page.lifecycle == "provisional" for page in self.pages.values())
        operation_creations = (
            0 if self.active_delta is None else len(self.active_delta.created_page_ids)
        )
        allowance = min(
            limits.max_pages - live_pages,
            limits.max_provisional_pages - provisional_pages,
            limits.max_page_creations_per_operation - operation_creations,
            limits.max_total_page_creations - self.total_page_creations,
        )
        if allowance <= 0:
            return 0, "capacity_refused"
        return allowance, None

    def _mark_popup_guard_uncertain(self) -> None:
        for page in self.pages.values():
            if page.lifecycle in {"provisional", "admitted", "active", "background"}:
                page.lifecycle = "uncertain"
                page.terminal_reason = "popup_guard_failed"
                page.control_epoch += 1
                page.revision = None
                page.clear_refs()
        self.active_page_id = None
        self.closing = True
        self.close_after_response = True

    async def _set_popup_creation_allowance(
        self,
        state: _InteractivePage,
        allowance: int,
    ) -> tuple[int, tuple[str, ...]]:
        try:
            result = await state.page.evaluate(
                "values => globalThis.__cayuSetPopupAdmission(values[0], values[1])",
                [self.popup_guard_token, allowance],
            )
        except asyncio.CancelledError:
            self._mark_popup_guard_uncertain()
            raise
        except Exception as exc:
            self._mark_popup_guard_uncertain()
            raise _GuestFailure("browser_crash") from exc
        if type(result) is not dict or set(result) != {"blocked", "urls"}:
            self._mark_popup_guard_uncertain()
            raise _GuestFailure("incompatible_browser")
        blocked = result["blocked"]
        urls = result["urls"]
        if (
            type(blocked) is not int
            or blocked not in {0, 1, 2, 3}
            or type(urls) is not list
            or len(urls) > _INTERACTIVE_MAX_PAGE_CREATIONS_PER_OPERATION
            or any(
                type(url) is not str
                or len(url.encode("utf-8", errors="replace")) > _MAX_URL_LENGTH
                or (url != "about:blank" and not _browser_request_is_admissible(url))
                for url in urls
            )
        ):
            self._mark_popup_guard_uncertain()
            raise _GuestFailure("incompatible_browser")
        return blocked, tuple(urls)

    async def _begin_popup_effect(
        self,
        state: _InteractivePage,
        request: _InteractiveRequest,
    ) -> str | None:
        opener_origin = self._current_page_origin(state)
        allowance, refusal = self._popup_creation_allowance(
            state,
            request,
            opener_origin=opener_origin,
        )
        self.popup_candidate_observed.clear()
        self.popup_effect_opener_page_id = state.page_id
        self.popup_effect_opener_origin = opener_origin
        try:
            blocked, urls = await self._set_popup_creation_allowance(state, allowance)
        except BaseException:
            self.popup_effect_opener_page_id = None
            self.popup_effect_opener_origin = None
            raise
        if blocked or urls:
            await self._set_popup_creation_allowance(state, 0)
            if urls:
                self._mark_popup_guard_uncertain()
                raise _GuestFailure(
                    "browser_crash",
                    allocation_disposition="uncertain",
                )
            self._record_popup_refusal(
                page_id=f"bp_{secrets.token_hex(16)}",
                opener_page_id=state.page_id,
                reason="policy_denied" if blocked & 2 else "capacity_refused",
            )
            raise _GuestFailure(
                "policy_denied" if blocked & 2 else "resource_exhausted",
                allocation_disposition="live",
            )
        return refusal

    async def _end_popup_effect(
        self,
        state: _InteractivePage,
        refusal: str | None,
    ) -> tuple[str, ...]:
        blocked, urls = await self._set_popup_creation_allowance(state, 0)
        if blocked:
            self._record_popup_refusal(
                page_id=f"bp_{secrets.token_hex(16)}",
                opener_page_id=state.page_id,
                reason=refusal or ("policy_denied" if blocked & 2 else "capacity_refused"),
            )
        return urls

    async def _wait_for_popup_candidates(
        self,
        request: _InteractiveRequest,
        expected_urls: tuple[str, ...],
    ) -> None:
        delta = self.active_delta
        expected_count = len(expected_urls)
        if delta is None or expected_count > request.limits.max_page_creations_per_operation:
            self._mark_popup_guard_uncertain()
            raise _GuestFailure("browser_crash", allocation_disposition="uncertain")
        deadline = asyncio.get_running_loop().time() + max(
            0.001,
            request.limits.max_wait_ms / 1000,
        )
        try:
            while delta.candidate_count < expected_count:
                self.popup_candidate_observed.clear()
                if delta.candidate_count >= expected_count:
                    break
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    await self.popup_candidate_observed.wait()
            if delta.candidate_count != expected_count:
                self._mark_popup_guard_uncertain()
                raise _GuestFailure(
                    "browser_crash",
                    allocation_disposition="uncertain",
                )
            # The guard authenticates the bounded number of creations in this
            # action. Exact Playwright page/frame objects bind their lineage and
            # staged requests. URLs can change immediately (including from
            # about:blank), and callbacks need not follow JavaScript call order.
            if delta.staged_frames:
                self._mark_popup_guard_uncertain()
                raise _GuestFailure("browser_crash", allocation_disposition="uncertain")
        except asyncio.CancelledError:
            self._mark_popup_guard_uncertain()
            raise
        except TimeoutError as exc:
            self._mark_popup_guard_uncertain()
            raise _GuestFailure(
                "cleanup_failed",
                allocation_disposition="uncertain",
            ) from exc

    def _record_popup_refusal(
        self,
        *,
        page_id: str,
        opener_page_id: str,
        reason: str,
    ) -> None:
        delta = self.active_delta
        if delta is None:
            return
        if any(item["page_id"] == page_id for item in delta.refused):
            return
        if len(delta.refused) >= (
            self.active_request.limits.max_page_creations_per_operation
            if self.active_request is not None
            else 1
        ):
            return
        delta.refused.append(
            {
                "page_id": page_id,
                "opener_page_id": opener_page_id,
                "reason": reason,
            }
        )

    def _schedule_popup_cleanup(self, page: Any, *, retire: bool = False) -> None:
        limits = self.configuration_limits
        maximum = 1 if limits is None else limits.max_provisional_pages
        if page not in self.popup_cleanup_pages and len(self.popup_cleanup_pages) < maximum:
            self.popup_cleanup_pages.append(page)
        else:
            retire = True
        self.popup_cleanup_retire = self.popup_cleanup_retire or retire
        if self.popup_cleanup_task is None or self.popup_cleanup_task.done():
            self.popup_cleanup_task = asyncio.create_task(self._drain_popup_cleanup())

    async def _drain_popup_cleanup(self) -> bool:
        cleanup_ok = True
        while self.popup_cleanup_pages:
            page = self.popup_cleanup_pages.pop(0)
            limits = self.configuration_limits
            if limits is None or not self._reserve_page_cleanup(limits):
                cleanup_ok = False
                self.popup_cleanup_retire = True
                break
            try:
                await page.close()
            except asyncio.CancelledError:
                self.popup_cleanup_pages.insert(0, page)
                raise
            except Exception:
                cleanup_ok = False
                self.popup_cleanup_retire = True
        if self.popup_cleanup_retire:
            cleanup_ok = await self._close_context_after_limit_abort() and cleanup_ok
            self.closing = True
            self.close_after_response = True
        return cleanup_ok

    def _register_popup_candidate(
        self,
        opener: _InteractivePage,
        popup: Any,
    ) -> _InteractivePage | None:
        existing = self._state_for_page(popup)
        if existing is not None:
            return existing
        request = self.active_request
        page_id = f"bp_{secrets.token_hex(16)}"
        if request is None or self.active_delta is None:
            if not bool(self.configuration_multi_page):
                self._schedule_popup_limit_abort(opener, popup)
            else:
                self._schedule_popup_cleanup(popup)
            return None
        limits = request.limits
        if self.total_page_creations >= limits.max_total_page_creations:
            refusal = "capacity_refused"
            creation_epoch = None
        else:
            self.total_page_creations += 1
            creation_epoch = self.total_page_creations
            if self.popup_effect_opener_page_id != opener.page_id:
                refusal = "operation_refused"
            else:
                refusal = self._popup_policy_refusal(
                    opener,
                    request,
                    opener_origin=self.popup_effect_opener_origin,
                )
        live_count = sum(
            page.lifecycle in {"provisional", "admitted", "active", "background"}
            for page in self.pages.values()
        )
        provisional_count = sum(page.lifecycle == "provisional" for page in self.pages.values())
        if refusal is None and (
            live_count >= limits.max_pages
            or provisional_count >= limits.max_provisional_pages
            or len(self.active_delta.created_page_ids) >= limits.max_page_creations_per_operation
        ):
            refusal = "capacity_refused"
        if refusal is not None:
            self._record_popup_refusal(
                page_id=page_id,
                opener_page_id=opener.page_id,
                reason=refusal,
            )
            if not request.multi_page:
                self._schedule_popup_limit_abort(opener, popup)
            else:
                self._schedule_popup_cleanup(
                    popup,
                    retire=refusal == "capacity_refused",
                )
            return None
        if creation_epoch is None:  # pragma: no cover - capacity refusal returns above
            return None
        state = _InteractivePage(
            page=popup,
            session_id=self.session_id,
            page_id=page_id,
            creation_epoch=creation_epoch,
            control_epoch=1,
            lifecycle="provisional",
            opener_page_id=opener.page_id,
            opener_origin=self.popup_effect_opener_origin,
            creating_operation_id_sha256=hashlib.sha256(
                request.operation_id.encode("utf-8")
            ).hexdigest(),
            created_monotonic=asyncio.get_running_loop().time(),
            public_url=opener.public_url,
        )
        self.pages[page_id] = state
        self.active_delta.created_page_ids.add(page_id)
        frame = getattr(getattr(popup, "main_frame", None), "_impl_obj", None)
        staged = self.active_delta.staged_frames.pop(frame, None)
        if staged is not None:
            url, method = staged
            state.request_count += 1
            if method != "GET":
                state.denied_code = "policy_denied"
            else:
                state.staged_initial_url = url
        return state

    def _observe_popup_candidate(
        self,
        opener: _InteractivePage,
        popup: Any,
    ) -> None:
        self._register_popup_candidate(opener, popup)
        self._note_popup_candidate(popup)

    def _note_popup_candidate(self, popup: Any) -> None:
        if self.active_request is None or self.active_delta is None:
            return
        identity = id(popup)
        if identity in self.active_delta.candidate_identities:
            return
        self.active_delta.candidate_identities.add(identity)
        self.active_delta.candidate_pages.append(popup)
        self.active_delta.candidate_count += 1
        self.popup_candidate_observed.set()

    def _popup_destination_allowed(
        self,
        state: _InteractivePage,
        request_url: str,
    ) -> bool:
        policy = self.configuration_popup_policy
        if policy is None or policy.mode == "deny":
            return False
        destination = _interactive_origin(request_url)
        opener_origin = state.opener_origin
        if destination is None or opener_origin is None:
            return False
        if policy.mode == "same_origin" and destination != opener_origin:
            return False
        return not policy.allowed_destination_origins or (
            destination in policy.allowed_destination_origins
        )

    async def _route_interactive_request(self, route: Any, browser_request: Any) -> None:
        try:
            request_page = browser_request.frame.page
        except Exception:
            # Pinned Playwright 1.62 exposes the provisional Frame channel
            # before its public Frame.page exists. Retain that exact object,
            # never a URL/order guess, until the popup callback publishes it.
            delta = self.active_delta
            request = self.active_request
            if delta is not None and request is not None and self.popup_effect_opener_page_id:
                implementation = getattr(browser_request, "_impl_obj", None)
                initializer = getattr(implementation, "_initializer", None)
                channel = initializer.get("frame") if type(initializer) is dict else None
                frame = getattr(channel, "_object", None)
                self.total_requests += 1
                if (
                    frame is None
                    or not browser_request.is_navigation_request()
                    or frame in delta.staged_frames
                    or len(delta.staged_frames) >= request.limits.max_page_creations_per_operation
                    or self.total_requests > request.limits.max_total_requests
                    or type(browser_request.url) is not str
                    or len(browser_request.url.encode("utf-8")) > _MAX_URL_LENGTH
                ):
                    self._mark_popup_guard_uncertain()
                else:
                    delta.staged_frames[frame] = (browser_request.url, browser_request.method)
            await route.abort("blockedbyclient")
            return
        state = self._state_for_page(request_page)
        if state is None:
            # A popup's first request can reach the context route before
            # Playwright emits the opener's ``popup`` callback.  Authenticate
            # lineage from Playwright's exact opener object; never infer it
            # from whichever page happens to be active.
            opener_method = getattr(request_page, "opener", None)
            opener_page = None
            if callable(opener_method):
                try:
                    opener_page = await opener_method()
                except Exception:
                    opener_page = None
            opener = self._state_for_page(opener_page)
            if opener is None and self.popup_effect_opener_page_id is not None:
                opener = self.pages.get(self.popup_effect_opener_page_id)
            if opener is not None:
                state = self._register_popup_candidate(opener, request_page)
                self._note_popup_candidate(request_page)
            if state is None:
                await route.abort("blockedbyclient")
                return
        limits = self.configuration_limits
        if limits is None:
            await route.abort("blockedbyclient")
            return
        state.request_count += 1
        self.total_requests += 1
        if state.access_evidence is not None:
            await route.abort("blockedbyclient")
            return
        if (
            state.request_count > limits.max_requests
            or self.total_requests > limits.max_total_requests
        ):
            state.limit_exceeded = True
            state.limit_error_code = "resource_exhausted"
            self._schedule_response_limit_abort(state)
            await route.abort("blockedbyclient")
            return
        is_main_navigation = (
            browser_request.is_navigation_request()
            and browser_request.frame == request_page.main_frame
        )
        is_redirect = is_main_navigation and browser_request.redirected_from is not None
        if is_redirect:
            state.redirect_count += 1
            if state.redirect_count > limits.max_redirects:
                state.denied_code = "redirect_denied"
                await route.abort("blockedbyclient")
                return
        if not _browser_request_is_admissible(browser_request.url):
            state.denied_code = "redirect_denied" if is_redirect else "destination_denied"
            await route.abort("blockedbyclient")
            return
        if self.profile_allowed_origins is not None:
            request_scheme = urlsplit(browser_request.url).scheme.lower()
            if request_scheme == "https":
                try:
                    request_origin = _guest_https_origin(browser_request.url).removesuffix("/")
                except _GuestFailure:
                    request_origin = None
                if request_origin not in self.profile_allowed_origins:
                    state.denied_code = "destination_denied"
                    await route.abort("blockedbyclient")
                    return
        if (
            state.opener_page_id is not None
            and is_main_navigation
            and not self._popup_destination_allowed(state, browser_request.url)
        ):
            state.denied_code = "destination_denied"
            self._record_popup_refusal(
                page_id=state.page_id,
                opener_page_id=state.opener_page_id,
                reason="destination_denied",
            )
            if (
                self.active_request is None
                or self.active_delta is None
                or state.page_id not in self.active_delta.created_page_ids
            ):
                self._schedule_popup_cleanup(state.page)
            await route.abort("blockedbyclient")
            return
        if not state.configured:
            # Do not let the first popup document execute before its CDP
            # response guard exists.  The same operation installs the guard
            # and performs this staged navigation under the ordinary route,
            # redirect, response, egress, and size policies.
            if not is_main_navigation:
                await route.abort("blockedbyclient")
                return
            if browser_request.method != "GET":
                state.denied_code = "policy_denied"
                await route.abort("blockedbyclient")
                return
            state.staged_initial_url = browser_request.url
            await route.abort("blockedbyclient")
            return
        await route.continue_()

    def _route_profile_web_socket(self, web_socket_route: Any) -> None:
        allowed_origins = self.profile_allowed_origins
        limits = self.configuration_limits
        state = self.pages.get(self.active_page_id or "")
        self.total_requests += 1
        if state is not None:
            state.request_count += 1
        if (
            allowed_origins is None
            or limits is None
            or self.total_requests > limits.max_total_requests
            or (state is not None and state.request_count > limits.max_requests)
        ):
            if state is not None:
                state.limit_exceeded = True
                state.limit_error_code = "resource_exhausted"
                self._schedule_response_limit_abort(state)
            web_socket_route.close(code=1008, reason="Request limit exceeded.")
            return
        try:
            request_origin = _guest_websocket_https_origin(web_socket_route.url).removesuffix("/")
        except _GuestFailure:
            request_origin = None
        if request_origin is None or request_origin not in allowed_origins:
            if state is not None:
                state.denied_code = "destination_denied"
            web_socket_route.close(code=1008, reason="Policy denied.")
            return
        web_socket_route.connect_to_server()

    def _handle_page_download(self, state: _InteractivePage, download: Any) -> None:
        request = self.active_request
        expected_operation = state.authorized_download_operation_id_sha256
        if (
            request is not None
            and request.operation == "download"
            and request.page_id == state.page_id
            and expected_operation
            == hashlib.sha256(request.operation_id.encode("utf-8")).hexdigest()
        ):
            return
        state.denied_code = "policy_denied"
        task = state.unexpected_download_task
        if task is not None and not task.done():
            state.limit_exceeded = True
            state.limit_error_code = "resource_exhausted"
            self._schedule_response_limit_abort(state)
            return
        limits = self.configuration_limits
        if limits is None or not self._reserve_page_cleanup(limits):
            state.limit_exceeded = True
            state.limit_error_code = "resource_exhausted"
            self._schedule_response_limit_abort(state)
            return
        state.unexpected_download_task = asyncio.create_task(
            self._cancel_unexpected_download(state, download, limits=limits)
        )

    async def _cancel_unexpected_download(
        self,
        state: _InteractivePage,
        download: Any,
        *,
        limits: _InteractiveLimits,
    ) -> bool:
        timeout_seconds = max(0.001, limits.max_wait_ms / 1000)
        try:
            await asyncio.wait_for(download.cancel(), timeout=timeout_seconds)
        except asyncio.CancelledError:
            raise
        except Exception:
            state.lifecycle = "uncertain"
            state.terminal_reason = "cleanup_failed"
            self.closing = True
            self.close_after_response = True
            return await self._close_context_after_limit_abort()
        state.lifecycle = "closing"
        state.control_epoch += 1
        state.revision = None
        state.clear_refs()
        cleanup_ok = await self._await_page_close(
            state,
            timeout_seconds=timeout_seconds,
        )
        if cleanup_ok:
            state.lifecycle = "closed"
            state.terminal_reason = "policy_denied"
            if self.active_page_id == state.page_id:
                self.active_page_id = None
                self._select_remaining_active()
            if self.active_delta is not None:
                self.active_delta.closed_page_ids.add(state.page_id)
            else:
                self.pending_closed_page_ids.add(state.page_id)
            return True
        state.lifecycle = "uncertain"
        state.terminal_reason = "cleanup_failed"
        self.closing = True
        self.close_after_response = True
        return False

    async def _configure_page(
        self,
        state: _InteractivePage,
        limits: _InteractiveLimits,
    ) -> None:
        page = state.page
        page.set_default_timeout(max(1_000, limits.max_wait_ms))
        page.set_default_navigation_timeout(max(1_000, limits.max_wait_ms))

        state.cdp = await self.context.new_cdp_session(page)
        if self.configuration_visual_policy is not None:
            state.visual_owner = VisualPageOwner(state.cdp, self.visual_worker_instance)
            await state.visual_owner.install()
        await state.cdp.send("Network.enable")
        frame_tree = await state.cdp.send("Page.getFrameTree")
        main_frame_id = frame_tree["frameTree"]["frame"]["id"]
        if type(main_frame_id) is not str:
            raise _GuestFailure("browser_crash")
        state.cdp.on(
            "Network.requestWillBeSent",
            lambda params: _retain_document_request(state, main_frame_id, params),
        )
        # Request URLs omit fragments. After aborting an unguarded initial
        # navigation, Chromium retains the full target on this exact frame.
        # Restore only a fragment; never replace request authority with an
        # unrelated frame URL.
        unreachable_url = frame_tree["frameTree"]["frame"].get("unreachableUrl")
        if (
            state.staged_initial_url is not None
            and type(unreachable_url) is str
            and len(unreachable_url.encode("utf-8")) <= _MAX_URL_LENGTH
            and urlsplit(unreachable_url)._replace(fragment="").geturl() == state.staged_initial_url
        ):
            state.staged_initial_url = unreachable_url
        await state.cdp.send(
            "Fetch.enable",
            {
                "patterns": [
                    {
                        "urlPattern": "*",
                        "resourceType": "Document",
                        "requestStage": "Response",
                    }
                ]
            },
        )

        async def response_paused(params: dict[str, Any]) -> None:
            request_id = params.get("requestId")
            try:
                if type(request_id) is not str:
                    raise TypeError("Missing paused browser response identity.")
                frame_id = params.get("frameId")
                status_code = params.get("responseStatusCode")
                raw_headers = params.get("responseHeaders")
                if (
                    params.get("resourceType") == "Document"
                    and frame_id == main_frame_id
                    and type(status_code) is int
                    and type(raw_headers) is list
                ):
                    headers = {
                        item["name"]: item["value"]
                        for item in raw_headers
                        if type(item) is dict
                        and type(item.get("name")) is str
                        and type(item.get("value")) is str
                    }
                    broker_code = next(
                        (
                            value
                            for key, value in headers.items()
                            if key.lower() == _BROKER_ERROR_HEADER and value
                        ),
                        None,
                    )
                    if broker_code is not None:
                        state.denied_code = (
                            "redirect_denied"
                            if broker_code == "destination_denied" and state.redirect_count > 0
                            else broker_code
                        )
                        await state.cdp.send(
                            "Fetch.failRequest",
                            {"requestId": request_id, "errorReason": "BlockedByClient"},
                        )
                        return
                    response_url = params.get("request", {}).get("url")
                    if type(response_url) is not str:
                        raise TypeError("Missing paused browser response URL.")
                    if state.opener_page_id is not None and status_code in _REDIRECT_STATUS_CODES:
                        location = next(
                            (
                                value
                                for key, value in headers.items()
                                if key.lower() == "location" and value
                            ),
                            None,
                        )
                        if location is not None:
                            redirect_url = urljoin(response_url, location)
                            if not _browser_request_is_admissible(
                                redirect_url
                            ) or not self._popup_destination_allowed(state, redirect_url):
                                state.denied_code = "destination_denied"
                                self._record_popup_refusal(
                                    page_id=state.page_id,
                                    opener_page_id=state.opener_page_id,
                                    reason="destination_denied",
                                )
                                await state.cdp.send(
                                    "Fetch.failRequest",
                                    {
                                        "requestId": request_id,
                                        "errorReason": "BlockedByClient",
                                    },
                                )
                                return
                    access = _guest_http_access(
                        response_url,
                        status_code,
                        headers,
                        source="browser_response",
                    )
                    if access is not None:
                        if state.access_evidence is None:
                            state.access_evidence = access
                            state.public_url = _guest_https_origin(response_url)
                        await state.cdp.send(
                            "Fetch.failRequest",
                            {"requestId": request_id, "errorReason": "BlockedByClient"},
                        )
                        return
                await state.cdp.send("Fetch.continueResponse", {"requestId": request_id})
            except Exception:
                state.denied_code = "fetch_failed"
                if type(request_id) is str:
                    with contextlib.suppress(Exception):
                        await state.cdp.send(
                            "Fetch.failRequest",
                            {"requestId": request_id, "errorReason": "BlockedByClient"},
                        )

        state.cdp.on("Fetch.requestPaused", response_paused)

        def data_received(params: dict[str, Any]) -> None:
            length = params.get("encodedDataLength")
            if isinstance(length, (int, float)) and not isinstance(length, bool):
                state.response_bytes += max(0, math.ceil(float(length)))
                if state.response_bytes > limits.max_response_bytes:
                    state.limit_exceeded = True
                    self._schedule_response_limit_abort(state)

        state.cdp.on("Network.dataReceived", data_received)

        def response_observed(response: Any) -> None:
            try:
                headers = response.headers
                if (
                    response.request.is_navigation_request()
                    and response.request.frame == page.main_frame
                ):
                    access = _guest_http_access(
                        response.url,
                        response.status,
                        headers,
                        source="browser_response",
                    )
                    if access is not None and state.access_evidence is None:
                        state.access_evidence = access
                        state.public_url = _guest_https_origin(response.url)
                broker_code = next(
                    (
                        value
                        for key, value in headers.items()
                        if type(key) is str
                        and key.lower() == _BROKER_ERROR_HEADER
                        and type(value) is str
                        and value
                    ),
                    None,
                )
                if broker_code is not None:
                    state.denied_code = (
                        "redirect_denied"
                        if broker_code == "destination_denied"
                        and response.request.is_navigation_request()
                        and response.request.frame == page.main_frame
                        and response.request.redirected_from is not None
                        else broker_code
                    )
                length = headers.get("content-length")
                if length is not None and state.response_bytes + max(0, int(length)) > (
                    limits.max_response_bytes
                ):
                    state.limit_exceeded = True
                    self._schedule_response_limit_abort(state)
            except Exception:
                state.limit_exceeded = True
                self._schedule_response_limit_abort(state)

        page.on("response", response_observed)
        page.on("download", lambda download: self._handle_page_download(state, download))
        page.on("popup", lambda popup: self._observe_popup_candidate(state, popup))
        page.on("framenavigated", lambda frame: self._mark_page_navigated(state, frame))
        page.on("close", lambda: self._mark_page_closed(state, "closed_by_page"))
        page.on("crash", lambda: self._mark_page_crashed(state))

        async def dialog_observed(dialog: Any) -> None:
            state.dialog_observed = True
            with contextlib.suppress(Exception):
                await dialog.dismiss()

        page.on("dialog", dialog_observed)
        state.configured = True

    def _mark_page_navigated(self, state: _InteractivePage, frame: Any) -> None:
        try:
            if frame != state.page.main_frame:
                return
        except Exception:
            state.denied_code = "fetch_failed"
            return
        state.navigation_epoch += 1
        request = self.active_request
        navigation_owned_by_current_mutation = (
            request is not None
            and request.operation != "observe"
            and request.page_id == state.page_id
            and state.last_operation_id_sha256
            == hashlib.sha256(request.operation_id.encode("utf-8")).hexdigest()
        )
        blocked_navigation_owned_by_current_mutation = (
            state.access_evidence is not None and navigation_owned_by_current_mutation
        )
        if not blocked_navigation_owned_by_current_mutation:
            state.revision = None
            state.clear_refs()
        if not navigation_owned_by_current_mutation:
            if state.control_epoch >= _INTERACTIVE_MAX_OPERATIONS_PER_PAGE:
                state.limit_exceeded = True
                state.limit_error_code = "resource_exhausted"
                self._schedule_response_limit_abort(state)
                return
            state.control_epoch += 1
        limits = self.configuration_limits
        if limits is not None and state.navigation_epoch > limits.max_requests:
            state.limit_exceeded = True
            state.limit_error_code = "resource_exhausted"
            self._schedule_response_limit_abort(state)

    async def _execute_page(
        self,
        state: _InteractivePage,
        request: _InteractiveRequest,
    ) -> dict[str, Any]:
        page = state.page
        operation_evidence: dict[str, Any] | None = None
        failure = _interactive_page_failure(state)
        if failure is not None:
            raise failure
        action_target = None
        visual_action = request.operation in {"click_visual_target", "click_visual_point"}
        if request.operation not in {"navigate", "observe", "observe_visual"}:
            if (
                state.revision is None
                or request.expected_revision != state.revision
                or request.expected_control_epoch != state.control_epoch - 1
            ):
                raise _GuestFailure("incompatible_browser")
            admitted_refs = state.refs
            state.revision = None
            state.refs = {}
            internal_ref = None
            if request.ref is not None:
                internal_ref = admitted_refs.get(request.ref)
                if internal_ref is None:
                    raise _GuestFailure("missing_element")
                navigation_epoch = state.navigation_epoch
                if state.ref_targets is None:
                    # Characterization-only states constructed without going
                    # through observation retain the former lookup seam.
                    locator = page.locator(f"aria-ref={internal_ref}")
                    action_target = await locator.element_handle()
                else:
                    action_target = state.ref_targets.get(request.ref)
                if action_target is None or state.navigation_epoch != navigation_epoch:
                    raise _GuestFailure("missing_element")
                state.revision = None
                state.clear_refs()
                if state.ref_targets is not None:
                    try:
                        target_connected = await action_target.evaluate(
                            "element => element.isConnected"
                        )
                    except Exception as exc:
                        if request.operation == "upload":
                            raise
                        raise _GuestFailure("actionability_failed") from exc
                    if target_connected is not True:
                        raise _GuestFailure("actionability_failed")
            else:
                state.revision = None
                state.clear_refs()
        if state.visual_owner is not None and not visual_action:
            state.visual_owner.invalidate()
        _raise_interactive_dialog_failure(state, request.operation)
        guard_effect = request.multi_page and request.operation in (
            _INTERACTIVE_POPUP_EFFECT_OPERATIONS
        )
        popup_refusal: str | None = None
        if guard_effect:
            popup_refusal = await self._begin_popup_effect(state, request)
        primary_failure: BaseException | None = None
        try:
            if request.operation == "navigate":
                try:
                    await page.goto(
                        request.url,
                        wait_until="load",
                        timeout=max(1_000, request.limits.max_wait_ms),
                    )
                except Exception:
                    failure = await _settle_interactive_navigation_failure(state)
                    if failure is None:
                        raise
                    raise failure from None
            elif request.operation == "click":
                if action_target is None:  # pragma: no cover - parser invariant
                    raise _GuestFailure("incompatible_browser")
                await action_target.click()
                # Playwright can deliver the popup callback immediately after
                # the click await settles. Give that already-dispatched signal
                # one bounded event-loop turn before accepting the operation.
                await asyncio.sleep(0)
            elif visual_action:
                await self._execute_visual_action(state, request)
            elif request.operation == "fill":
                if action_target is None:  # pragma: no cover - parser invariant
                    raise _GuestFailure("incompatible_browser")
                await action_target.fill(request.value)
            elif request.operation == "select":
                if action_target is None:  # pragma: no cover - parser invariant
                    raise _GuestFailure("incompatible_browser")
                await action_target.select_option(request.value)
            elif request.operation == "press":
                if action_target is None:  # pragma: no cover - parser invariant
                    raise _GuestFailure("incompatible_browser")
                await action_target.press(request.key)
            elif request.operation == "wait":
                await page.wait_for_timeout(request.wait_ms)
            elif request.operation == "back":
                await _interactive_history_traversal(state, request, offset=-1)
            elif request.operation == "forward":
                await _interactive_history_traversal(state, request, offset=1)
            elif request.operation == "reload":
                await _interactive_safe_reload(state, request)
            elif request.operation == "scroll":
                operation_evidence = await _interactive_semantic_scroll(state, request)
            elif request.operation == "hover":
                if action_target is None:  # pragma: no cover - parser invariant
                    raise _GuestFailure("incompatible_browser")
                await action_target.hover()
            elif request.operation == "upload":
                if action_target is None:  # pragma: no cover - parser invariant
                    raise _GuestFailure("incompatible_browser")
                operation_evidence = await self._upload_files(
                    state, request, action_target, internal_ref
                )
            elif request.operation == "download":
                if action_target is None:  # pragma: no cover - parser invariant
                    raise _GuestFailure("incompatible_browser")
                return await self._download_and_observe(state, request, action_target)
            elif request.operation not in {"observe", "screenshot", "observe_visual"}:
                raise _GuestFailure("incompatible_browser")
            _raise_interactive_dialog_failure(state, request.operation)
        except BaseException as exc:
            primary_failure = exc
            raise
        finally:
            if (
                guard_effect
                and _interactive_page_failure(state) is None
                and state.visual_action_cleanup_disposition is None
            ):
                try:
                    admitted_urls = await self._end_popup_effect(state, popup_refusal)
                    await self._wait_for_popup_candidates(request, admitted_urls)
                except BaseException as cleanup_failure:
                    # Cancellation can arrive during a successful visual disarm,
                    # without retiring the page. Its authority does not depend
                    # on the retirement marker or on popup cleanup succeeding.
                    if primary_failure is not None and not isinstance(primary_failure, Exception):
                        raise primary_failure from _browser_cleanup_evidence(
                            primary_failure.__cause__, (cleanup_failure,)
                        )
                    raise
        # Associate request methods only with the loader actually installed in
        # the main frame. Before the first document request, there is no method
        # authority to associate (nor necessarily a browser history entry).
        if state.document_methods:
            await _synchronize_interactive_history_method(state)
        failure = _interactive_page_failure(state)
        if failure is not None:
            raise failure
        artifact: dict[str, Any] | None = None
        if request.operation == "screenshot":
            screenshot = await _interactive_screenshot(page, state.cdp, request)
            artifact = {
                "kind": "screenshot",
                "filename": "browser-page.png",
                "content_type": "image/png",
                "content_base64": base64.b64encode(screenshot).decode("ascii"),
            }
            state.artifact_count += 1
            self.total_artifacts += 1
        if request.operation == "observe_visual":
            policy = request.visual_policy
            if policy is None:
                raise VisualGuestFailure("visual_mode_disabled")
            if (
                type(self.profile_owner) is not _TemporaryProfileOwner
                or self.closing
                or self.close_requested.is_set()
            ):
                raise VisualGuestFailure("visual_publication_denied")
            if self.total_visual_captures >= policy["max_captures"]:
                raise _GuestFailure("resource_exhausted")
            self.total_visual_captures += 1
            captures: list[bytes] = []
            async with asyncio.timeout(policy["max_processing_ms"] / 1000):
                observation = await self._observe_page(
                    state,
                    request.limits,
                    visual_policy=policy,
                    visual_artifacts=captures,
                )
            artifact = {
                "kind": "screenshot",
                "filename": "browser-visual.png",
                "content_type": "image/png",
                "content_base64": base64.b64encode(captures[0]).decode("ascii"),
            }
            state.artifact_count += 1
            self.total_artifacts += 1
        else:
            observation = await self._observe_page(state, request.limits)
        if request.operation == "observe":
            state.dialog_observed = False
        failure = _interactive_page_failure(state)
        if failure is not None:
            raise failure
        return _interactive_success_payload(
            observation,
            artifact=artifact,
            operation_evidence=operation_evidence,
            profile_output_protected=self.profile_output_values is not None,
        )

    async def _refresh_profile_output_values(self) -> tuple[str, ...]:
        """Refresh the existing bounded private-value owner before any projection."""
        prior_values = self.profile_output_values
        if prior_values is None:
            return ()
        if (
            self.context is None
            or self.profile_plaintext_limit is None
            or self.profile_timeout_seconds is None
            or (self.profile_allowed_origins is None and not self.control.capture_restricted)
        ):
            raise _GuestFailure("policy_denied")
        try:
            async with asyncio.timeout(self.profile_timeout_seconds):
                current_state = await self.context.storage_state(indexed_db=False)
            if self.profile_allowed_origins is None:
                current_values = _bounded_live_browser_private_values(
                    current_state, maximum_bytes=self.profile_plaintext_limit
                )
            else:
                current_state = _validate_interactive_profile_state(
                    current_state,
                    maximum_bytes=self.profile_plaintext_limit,
                    allowed_origins=self.profile_allowed_origins,
                )
                current_values = _interactive_profile_private_values(current_state)
        except _GuestFailure:
            raise
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            raise _GuestFailure("policy_denied") from exc
        protected_values = _bounded_interactive_profile_private_values(
            prior_values,
            current_values,
            maximum_bytes=self.profile_plaintext_limit,
        )
        self.profile_output_values = protected_values
        return protected_values

    async def _protect_profile_observation(
        self,
        observation: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        """Omit current or historical profile values before guest publication."""
        if self.profile_output_values is None:
            return observation, False
        protected_values = await self._refresh_profile_output_values()
        protected = json.loads(json.dumps(observation, ensure_ascii=False))
        url = protected.get("url")
        if type(url) is not str:
            raise _GuestFailure("policy_denied")
        if any(secret in url for secret in protected_values):
            raise _GuestFailure("policy_denied")
        visible_values = [protected.get("title"), protected.get("snapshot")]
        refs = protected.get("refs")
        if type(refs) is not list:
            raise _GuestFailure("policy_denied")
        visible_values.extend(
            value
            for item in refs
            if type(item) is dict
            for value in (item.get("role"), item.get("name"))
        )
        if any(
            type(value) is str and secret in value
            for secret in protected_values
            for value in visible_values
        ):
            protected["title"] = None
            protected["snapshot"] = _INTERACTIVE_PROFILE_OUTPUT_OMITTED
            protected["refs"] = []
        return protected, True

    async def _download_and_observe(
        self,
        state: _InteractivePage,
        request: _InteractiveRequest,
        locator: Any,
    ) -> dict[str, Any]:
        state.authorized_download_operation_id_sha256 = hashlib.sha256(
            request.operation_id.encode("utf-8")
        ).hexdigest()
        try:
            async with state.page.expect_download(
                timeout=max(1_000, request.limits.max_wait_ms)
            ) as info:
                await locator.click()
            download = await info.value
            path = await _interactive_download_path(download, state, request.limits)
            if type(path) is not str:
                raise _GuestFailure("download_failed")
            metadata = os.stat(path, follow_symlinks=False)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size > request.limits.max_artifact_bytes
            ):
                raise _GuestFailure("oversized_artifact")
            with open(path, "rb") as handle:
                content = handle.read(request.limits.max_artifact_bytes + 1)
            if len(content) > request.limits.max_artifact_bytes:
                raise _GuestFailure("oversized_artifact")
            filename = Path(str(download.suggested_filename)).name or "download.bin"
            filename = filename[:255]
        except _GuestFailure:
            raise
        except Exception as exc:
            raise _interactive_playwright_error("download", exc) from exc
        finally:
            state.authorized_download_operation_id_sha256 = None
        observation = await self._observe_page(state, request.limits)
        failure = _interactive_page_failure(state)
        if failure is not None:
            raise failure
        artifact = {
            "kind": "download",
            "filename": filename,
            "content_type": "application/octet-stream",
            "content_base64": base64.b64encode(content).decode("ascii"),
        }
        state.artifact_count += 1
        self.total_artifacts += 1
        return _interactive_success_payload(
            observation,
            artifact=artifact,
            profile_output_protected=False,
        )

    async def _execute_visual_action(
        self, state: _InteractivePage, request: _InteractiveRequest
    ) -> None:
        owner = state.visual_owner
        policy = request.visual_policy
        if owner is None or policy is None:
            raise VisualGuestFailure("visual_evidence_expired")
        primary: BaseException | None = None
        owner.action_task = asyncio.create_task(owner.click(policy, asdict(request)))
        try:
            done, _ = await asyncio.wait(
                {owner.action_task}, timeout=policy["max_processing_ms"] / 1000
            )
            if not done:
                raise TimeoutError("Browser visual action exceeded its bound.")
            owner.action_task.result()
        except BaseException as exc:
            primary = exc
        finally:
            owner.invalidate()

        if primary is not None and not isinstance(primary, VisualGuestFailure):
            self.closing = True
            self.close_requested.set()
            state.visual_action_cleanup_disposition = "uncertain"
            retired = False

            async def retire_uncertain_input() -> tuple[BaseException, ...]:
                nonlocal retired
                try:
                    retired = await self.close()
                    if retired:
                        state.visual_action_cleanup_disposition = "retired"
                except BaseException as exc:
                    return (exc,)
                return () if retired else (RuntimeError("Browser input settlement is uncertain."),)

            outcome = await _await_browser_cleanup_resisting_cancellation(
                asyncio.create_task(retire_uncertain_input())
            )
            cause = _browser_cleanup_evidence(primary, outcome.errors)
            signal: BaseException | None = outcome.cancellation
            if signal is None and not isinstance(primary, Exception):
                signal = primary
            if signal is not None:
                raise signal from cause
            raise _GuestFailure(
                "timeout" if isinstance(primary, TimeoutError) else "outcome_ambiguous",
                allocation_disposition="retired" if retired else "uncertain",
            ) from cause
        owner.action_task = None

        async def settle_guard() -> tuple[BaseException, ...]:
            owner.disarm_task = asyncio.create_task(owner.disarm(policy))
            done, _ = await asyncio.wait({owner.disarm_task}, timeout=1.0)
            if not done:
                return (TimeoutError("Browser visual guard cleanup exceeded its bound."),)
            try:
                owner.disarm_task.result()
            except BaseException as exc:
                return (exc,)
            owner.disarm_task = None
            return ()

        outcome = await _await_browser_cleanup_resisting_cancellation(
            asyncio.create_task(settle_guard())
        )
        if outcome.errors:
            self.closing = True
            self.close_requested.set()
            state.visual_action_cleanup_disposition = "uncertain"

            # An armed native-input guard is not a reusable page. The retained
            # CDP owner is joined by close; only positive browser settlement can
            # release this allocation.
            async def retire_guard() -> tuple[BaseException, ...]:
                try:
                    if await self.close():
                        state.visual_action_cleanup_disposition = "retired"
                        return ()
                except BaseException as exc:
                    return (exc,)
                return (RuntimeError("Browser input settlement is uncertain."),)

            settlement = await _await_browser_cleanup_resisting_cancellation(
                asyncio.create_task(retire_guard())
            )
            failure = _GuestFailure(
                "cleanup_failed",
                allocation_disposition=state.visual_action_cleanup_disposition,
            )
            cause = _browser_cleanup_evidence(primary, outcome.errors + settlement.errors)
            signal = settlement.cancellation or outcome.cancellation
            if signal is None and primary is not None and not isinstance(primary, Exception):
                signal = primary
            if signal is not None:
                raise signal from cause
            raise failure from cause
        if outcome.cancellation is not None:
            raise outcome.cancellation from _browser_cleanup_evidence(primary, ())
        if primary is not None:
            raise primary

    async def _upload_files(
        self,
        state: _InteractivePage,
        request: _InteractiveRequest,
        locator: Any,
        internal_ref: str | None,
    ) -> dict[str, Any]:
        if internal_ref is None or self.home is None or not request.upload_files:
            raise _GuestFailure("incompatible_browser")
        target = await locator.evaluate(
            """element => ({
                is_file_input: element instanceof HTMLInputElement &&
                    element.type.toLowerCase() === "file",
                multiple: element instanceof HTMLInputElement && element.multiple,
            })"""
        )
        if (
            type(target) is not dict
            or set(target) != {"is_file_input", "multiple"}
            or type(target.get("is_file_input")) is not bool
            or type(target.get("multiple")) is not bool
            or not target["is_file_input"]
            or (len(request.upload_files) > 1 and not target["multiple"])
        ):
            raise _GuestFailure("incompatible_upload_target", allocation_disposition="live")
        # File paths remain browser-backed after selection. Transfer bounded
        # admitted bytes instead, so later submission cannot outlive a temp file.
        await locator.set_input_files(
            [
                {"name": upload.filename, "mimeType": upload.content_type, "buffer": upload.content}
                for upload in request.upload_files
            ]
        )
        return {
            "operation": "upload",
            "selected_file_count": len(request.upload_files),
            "selection_state": "selected",
        }

    async def close(self, *, timeout_seconds: float = 5.0) -> bool:
        self.closing = True
        self.operator_frames.suspend()
        self.control.uncertain()
        deadline = asyncio.get_running_loop().time() + max(0.001, timeout_seconds)

        async def settle_owned_task(
            task: asyncio.Task[Any],
            *,
            label: str,
        ) -> tuple[bool, tuple[BaseException, ...]]:
            if task is asyncio.current_task():
                return True, ()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False, (TimeoutError(f"Browser {label} cleanup exceeded its bound."),)
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if not done:
                # Do not cancel opaque Playwright cleanup.  Cancellation does
                # not prove that the owned mutation stopped; retain the task so
                # a later/final close joins the exact same owner.
                return False, (TimeoutError(f"Browser {label} cleanup exceeded its bound."),)
            try:
                value = task.result()
            except asyncio.CancelledError as exc:
                failure = RuntimeError(f"Browser {label} cleanup was cancelled unexpectedly.")
                failure.__cause__ = exc
                return False, (failure,)
            except BaseException as exc:
                return False, (exc,)
            return value is not False, ()

        async def settle_all() -> tuple[BaseException, ...]:
            errors: list[BaseException] = []
            if self._operator_connection is not None:
                self._operator_transport_close()
            popup_cleanup = self.popup_cleanup_task
            if popup_cleanup is not None:
                _, failures = await settle_owned_task(
                    popup_cleanup,
                    label="provisional-page",
                )
                errors.extend(failures)
                if popup_cleanup.done():
                    self.popup_cleanup_task = None

            for state in tuple(self.pages.values()):
                limit_abort = state.limit_abort_task
                if limit_abort is None:
                    continue
                _, failures = await settle_owned_task(
                    limit_abort,
                    label=f"page-limit-{state.creation_epoch}",
                )
                errors.extend(failures)
                if limit_abort.done():
                    state.limit_abort_task = None

            for state in tuple(self.pages.values()):
                unexpected_download = state.unexpected_download_task
                if unexpected_download is None:
                    continue
                _, failures = await settle_owned_task(
                    unexpected_download,
                    label=f"unexpected-download-{state.creation_epoch}",
                )
                errors.extend(failures)
                if unexpected_download.done():
                    state.unexpected_download_task = None

            for state in tuple(self.pages.values()):
                if state.lifecycle in {"closed", "crashed"}:
                    continue
                state.lifecycle = "closing"
                if self.active_page_id == state.page_id:
                    self.active_page_id = None
                state.control_epoch += 1
                state.revision = None
                state.clear_refs()
                task = state.cleanup_task
                if task is None:
                    task = asyncio.create_task(state.page.close())
                    state.cleanup_task = task
                settled, failures = await settle_owned_task(
                    task,
                    label=f"page-{state.creation_epoch}",
                )
                errors.extend(failures)
                if settled:
                    state.lifecycle = "closed"
                    state.terminal_reason = "session_closed"
                else:
                    state.lifecycle = "uncertain"
                    state.terminal_reason = "cleanup_failed"
                    if task.done():
                        state.cleanup_task = None

            for attribute, operation_name in (
                ("context", "close"),
                ("browser", "close"),
                ("playwright", "stop"),
            ):
                owner = getattr(self, attribute)
                if owner is None:
                    continue
                task = self.session_cleanup_tasks.get(attribute)
                if task is None:
                    task = asyncio.create_task(getattr(owner, operation_name)())
                    self.session_cleanup_tasks[attribute] = task
                settled, failures = await settle_owned_task(task, label=attribute)
                errors.extend(failures)
                if settled:
                    setattr(self, attribute, None)
                    self.session_cleanup_tasks.pop(attribute, None)
                elif task.done():
                    self.session_cleanup_tasks.pop(attribute, None)

            bootstrap = self._operator_bootstrap_task
            if bootstrap is not None:
                settled, failures = await settle_owned_task(bootstrap, label="operator-channel")
                errors.extend(failures)
                if settled:
                    # Invocation completion can normally close the channel before
                    # native shutdown starts. Keep that disconnect fenced until
                    # context/browser closure proves quiescence; its historical
                    # normal-close signal must not then poison every close retry.
                    # An exact pre-dispatch allocation-loss refusal is likewise
                    # historical once native shutdown succeeds. Keep the original
                    # in bootstrap.result(); do not forgive cleanup failures,
                    # chained failures, abnormal transport errors or aggregates.
                    errors.extend(
                        failure
                        for failure in bootstrap.result()
                        if not (
                            self.context is None
                            and self.browser is None
                            and (
                                control_connection_closed_normally(failure)
                                or (
                                    type(failure) is GuestControlAllocationLost
                                    and failure.__cause__ is None
                                    and failure.__context__ is None
                                )
                            )
                        )
                    )
            transport_close = self.session_cleanup_tasks.get("operator-transport")
            if transport_close is not None:
                settled, failures = await settle_owned_task(
                    transport_close, label="operator-transport"
                )
                errors.extend(failures)
                if settled:
                    self._operator_connection = None
                    self.session_cleanup_tasks.pop("operator-transport", None)

            # Close the browser first: a CDP acknowledgement can be permanently
            # stalled while its input is still queued. Waiting for it must not
            # spend the time reserved for stopping that browser.
            for state in tuple(self.pages.values()):
                restoration = state.observation_cleanup_task
                if restoration is not None:
                    _, failures = await settle_owned_task(restoration, label="observation-guard")
                    if restoration.done():
                        state.observation_cleanup_task = None
                        if self.browser is not None:
                            errors.extend(failures)
                            if not failures:
                                errors.extend(restoration.result())
                    else:
                        errors.extend(failures)
                visual_owner = state.visual_owner
                if visual_owner is None:
                    continue
                for task_name in ("action_task", "disarm_task"):
                    task = getattr(visual_owner, task_name)
                    if task is None:
                        continue
                    _, failures = await settle_owned_task(task, label="visual-" + task_name)
                    if task.done():
                        # Browser closure makes failed input acknowledgements
                        # expected. An uncompleted owner must remain retained.
                        setattr(visual_owner, task_name, None)
                        if self.browser is not None:
                            errors.extend(failures)
                    else:
                        errors.extend(failures)

            if self.profile_owner is None and self.home is not None:
                try:
                    remaining = max(0.001, deadline - asyncio.get_running_loop().time())
                    self.profile_owner = await _start_temporary_profile_owner(
                        timeout_seconds=min(
                            remaining,
                            _MAX_PROFILE_CLEANUP_RESERVE_SECONDS,
                        ),
                        startup_timeout_seconds=remaining,
                        existing_home=self.home,
                    )
                except BaseException as exc:
                    errors.append(exc)
            profile_owner = self.profile_owner
            if profile_owner is not None:
                remaining = max(0.0, deadline - asyncio.get_running_loop().time())
                profile_errors = await _cleanup_temporary_profile_owner(
                    profile_owner,
                    timeout_seconds=remaining,
                )
                errors.extend(profile_errors)
                if profile_owner.process.poll() is not None:
                    self.profile_owner = None
                if not profile_owner.home.exists():
                    self.home = None
            return tuple(errors)

        outcome = await _await_browser_cleanup_resisting_cancellation(
            asyncio.create_task(settle_all())
        )
        cleanup_ok = not outcome.errors
        if cleanup_ok:
            self.control.state = "closed"
            self.pages.clear()
            self.active_page_id = None
        if outcome.cancellation is not None:
            cause = _browser_cleanup_evidence(None, outcome.errors)
            if cause is None:
                raise outcome.cancellation
            raise outcome.cancellation from cause
        process_control = next(
            (
                error
                for error in outcome.errors
                if isinstance(error, (GeneratorExit, KeyboardInterrupt, SystemExit))
            ),
            None,
        )
        if process_control is not None:
            raise process_control
        return cleanup_ok


def _raise_interactive_dialog_failure(state: _InteractivePage, operation: str) -> None:
    if not state.dialog_observed:
        return
    state.dialog_observed = False
    if operation == "reload":
        raise _GuestFailure("unsafe_reload", allocation_disposition="live")
    if operation in {"back", "forward"}:
        raise _GuestFailure("history_unavailable", allocation_disposition="live")
    raise _GuestFailure("actionability_failed")


def _retain_document_request(
    state: _InteractivePage, main_frame_id: str, params: dict[str, Any]
) -> None:
    # Request/response arrival is not document commitment (e.g. HTTP 204).
    # Only the loader currently installed in the main frame can authorize reuse.
    if params.get("type") != "Document" or params.get("frameId") != main_frame_id:
        return
    loader_id = params.get("loaderId")
    request = params.get("request")
    if type(loader_id) is not str or not loader_id or type(request) is not dict:
        return
    method = request.get("method")
    state.document_methods.pop(loader_id, None)
    if type(method) is str and method:
        state.document_methods[loader_id] = method.upper()
    # Evicted history loses positive method evidence and fails closed.
    while len(state.document_methods) > 128:
        del state.document_methods[next(iter(state.document_methods))]


async def _committed_document_identity(state: _InteractivePage) -> tuple[str, str]:
    raw = await state.cdp.send("Page.getFrameTree")
    try:
        frame = raw["frameTree"]["frame"]
        identity = (frame["id"], frame["loaderId"])
    except (KeyError, TypeError) as exc:
        raise _GuestFailure("browser_crash") from exc
    if any(type(value) is not str or not value for value in identity):
        raise _GuestFailure("browser_crash")
    return identity


async def _interactive_navigation_history(state: _InteractivePage) -> tuple[int, tuple[int, ...]]:
    if state.cdp is None:
        raise _GuestFailure("browser_crash")
    raw = await state.cdp.send("Page.getNavigationHistory")
    if (
        type(raw) is not dict
        or type(raw.get("currentIndex")) is not int
        or type(raw.get("entries")) is not list
    ):
        raise _GuestFailure("browser_crash")
    entries: list[int] = []
    for item in raw["entries"]:
        if type(item) is not dict or type(item.get("id")) is not int:
            raise _GuestFailure("browser_crash")
        entries.append(item["id"])
    current_index = raw["currentIndex"]
    if current_index < 0 or current_index >= len(entries) or len(set(entries)) != len(entries):
        raise _GuestFailure("browser_crash")
    return current_index, tuple(entries)


async def _synchronize_interactive_history_method(state: _InteractivePage) -> tuple[str, str]:
    document = await _committed_document_identity(state)
    current_index, entries = await _interactive_navigation_history(state)
    if await _committed_document_identity(state) != document:
        raise _GuestFailure("navigation_timeout")
    current_entry = entries[current_index]
    method = state.document_methods.get(document[1])
    state.history_methods.pop(current_entry, None)
    if method is not None:
        state.history_methods[current_entry] = method
    state.current_document_method = method
    return document


async def _interactive_history_traversal(
    state: _InteractivePage,
    request: _InteractiveRequest,
    *,
    offset: Literal[-1, 1],
) -> None:
    await _synchronize_interactive_history_method(state)
    current_index, entries = await _interactive_navigation_history(state)
    target_index = current_index + offset
    if target_index < 0 or target_index >= len(entries):
        raise _GuestFailure("history_unavailable", allocation_disposition="live")
    target_method = state.history_methods.get(entries[target_index])
    if target_method not in {"GET", "HEAD"}:
        raise _GuestFailure("history_unavailable", allocation_disposition="live")
    operation = state.page.go_back if offset < 0 else state.page.go_forward
    await operation(
        wait_until="load",
        timeout=max(1_000, request.limits.max_wait_ms),
    )
    final_index, final_entries = await _interactive_navigation_history(state)
    if final_index != target_index:
        raise _GuestFailure("history_unavailable", allocation_disposition="live")
    state.current_document_method = state.history_methods.get(
        final_entries[final_index],
        target_method,
    )


async def _interactive_safe_reload(
    state: _InteractivePage,
    request: _InteractiveRequest,
) -> None:
    document = await _synchronize_interactive_history_method(state)
    if state.current_document_method not in {"GET", "HEAD"}:
        raise _GuestFailure("unsafe_reload", allocation_disposition="live")
    current_index, _entries = await _interactive_navigation_history(state)
    timeout_ms = max(1_000, request.limits.max_wait_ms)
    # Bind dispatch in Chromium, not another read/check: an autonomous POST can
    # commit after method validation and must never become the reload target.
    async with asyncio.timeout(timeout_ms / 1_000):
        async with state.page.expect_navigation(wait_until="load", timeout=timeout_ms):
            try:
                await state.cdp.send("Page.reload", {"loaderId": document[1]})
            except Exception as exc:
                if str(exc) == (
                    "CDPSession.send: Protocol error (Page.reload): "
                    "Reload was discarded because the page already navigated"
                ):
                    raise _GuestFailure("unsafe_reload", allocation_disposition="live") from exc
                raise
    final_index, _final_entries = await _interactive_navigation_history(state)
    if final_index != current_index:
        raise _GuestFailure("unsafe_reload", allocation_disposition="live")


async def _interactive_semantic_scroll(
    state: _InteractivePage,
    request: _InteractiveRequest,
) -> dict[str, Any]:
    if (
        state.cdp is None
        or request.direction not in {"up", "down", "left", "right"}
        or request.amount not in {"line", "page"}
        or type(request.repeat_count) is not int
        or request.repeat_count < 1
        or request.repeat_count > request.limits.max_scroll_repeats
    ):
        raise _GuestFailure("invalid_scroll", allocation_disposition="live")
    frame_tree = await state.cdp.send("Page.getFrameTree")
    if type(frame_tree) is not dict or type(frame_tree.get("frameTree")) is not dict:
        raise _GuestFailure("browser_crash")
    frame = frame_tree["frameTree"].get("frame")
    if type(frame) is not dict or type(frame.get("id")) is not str:
        raise _GuestFailure("browser_crash")
    context_id = await _create_isolated_world(state.cdp, frame["id"])
    projection = await state.cdp.send(
        "Runtime.evaluate",
        {
            "contextId": context_id,
            "expression": """(() => {
                const direction = __CAYU_DIRECTION__;
                const amount = __CAYU_AMOUNT__;
                const repeats = __CAYU_REPEATS__;
                const horizontal = direction === "left" || direction === "right";
                const sign = direction === "up" || direction === "left" ? -1 : 1;
                const viewport = horizontal ? window.innerWidth : window.innerHeight;
                const unit = amount === "line" ? 40 : Math.max(1, Math.floor(viewport * 0.875));
                const before = horizontal ? window.scrollX : window.scrollY;
                window.scrollBy({
                    left: horizontal ? sign * unit * repeats : 0,
                    top: horizontal ? 0 : sign * unit * repeats,
                    behavior: "instant",
                });
                const after = horizontal ? window.scrollX : window.scrollY;
                const root = document.documentElement;
                const body = document.body;
                const extent = horizontal
                    ? Math.max(root?.scrollWidth || 0, body?.scrollWidth || 0)
                    : Math.max(root?.scrollHeight || 0, body?.scrollHeight || 0);
                const maximum = Math.max(0, extent - viewport);
                const atStart = after <= 0;
                const atEnd = after >= maximum;
                return {
                    moved: after !== before,
                    edge: sign < 0 && atStart ? "start" : sign > 0 && atEnd ? "end" : "none",
                };
            })()""".replace("__CAYU_DIRECTION__", json.dumps(request.direction))
            .replace("__CAYU_AMOUNT__", json.dumps(request.amount))
            .replace("__CAYU_REPEATS__", str(request.repeat_count)),
            "returnByValue": True,
            "awaitPromise": False,
            "userGesture": True,
        },
    )
    if (
        type(projection) is not dict
        or projection.get("exceptionDetails") is not None
        or type(projection.get("result")) is not dict
        or type(projection["result"].get("value")) is not dict
    ):
        raise _GuestFailure("browser_crash")
    evidence = projection["result"]["value"]
    if (
        set(evidence) != {"edge", "moved"}
        or type(evidence.get("moved")) is not bool
        or evidence.get("edge") not in {"start", "end", "none"}
    ):
        raise _GuestFailure("browser_crash")
    return {"operation": "scroll", **evidence}


async def _interactive_screenshot(
    page: Any,
    cdp: Any,
    request: _InteractiveRequest,
) -> bytes:
    if cdp is None:
        raise _GuestFailure("browser_crash")
    if request.full_page:
        width, height = _screenshot_layout_dimensions(await cdp.send("Page.getLayoutMetrics"))
    else:
        width, height = 1280, 720
    limits = request.limits
    if (
        width <= 0
        or height <= 0
        or width > limits.max_page_width
        or height > limits.max_page_height
        or width * height > limits.max_page_pixels
    ):
        raise _GuestFailure("oversized_artifact")
    options: dict[str, Any] = {"type": "png", "full_page": False}
    if request.full_page:
        options["clip"] = {"x": 0, "y": 0, "width": width, "height": height}
    screenshot = await page.screenshot(**options)
    if type(screenshot) is not bytes or len(screenshot) > limits.max_artifact_bytes:
        raise _GuestFailure("oversized_artifact")
    actual_width, actual_height = _png_header_dimensions(screenshot)
    if (
        actual_width > limits.max_page_width
        or actual_height > limits.max_page_height
        or actual_width * actual_height > limits.max_page_pixels
        or (actual_width, actual_height) != (width, height)
    ):
        raise _GuestFailure("oversized_artifact")
    return screenshot


async def _interactive_download_path(
    download: Any,
    state: _InteractivePage,
    limits: _InteractiveLimits,
) -> str:
    path_task = asyncio.create_task(download.path())
    try:
        while not path_task.done():
            if state.denied_code is not None or state.limit_exceeded:
                with contextlib.suppress(Exception):
                    await download.cancel()
                with contextlib.suppress(BaseException):
                    await path_task
                failure = _interactive_page_failure(state)
                if failure is None:  # pragma: no cover - guarded above
                    raise _GuestFailure("browser_crash")
                raise failure
            await asyncio.sleep(0.01)
        path = await path_task
    finally:
        if not path_task.done():
            path_task.cancel()
            with contextlib.suppress(BaseException):
                await path_task
    failure = _interactive_page_failure(state)
    if failure is not None:
        raise failure
    if isinstance(path, Path):
        return str(path)
    if type(path) is not str:
        raise _GuestFailure("download_failed")
    return path


async def _admit_interactive_snapshot_materialization(
    state: _InteractivePage,
    limits: _InteractiveLimits,
) -> None:
    """Bound every frame's DOM before Playwright materializes AI evidence."""

    if state.cdp is None:
        raise _GuestFailure("browser_crash")
    remaining_nodes = limits.max_dom_nodes
    # The returned evidence may be truncated to ``max_snapshot_bytes``. Allow a
    # bounded amount of extra source so ordinary truncation remains useful,
    # while preventing one unbounded accessible scalar from being materialized
    # by Playwright before Cayu can truncate the result.
    remaining_source_bytes = (
        limits.max_snapshot_bytes * _INTERACTIVE_ACCESSIBILITY_SOURCE_MULTIPLIER
    )
    total_nodes = 0
    total_source_bytes = 0
    for frame_id in await _interactive_frame_ids(state.cdp):
        node_count, source_bytes, limit_exceeded = await _interactive_frame_snapshot_census(
            state.cdp,
            frame_id,
            max_nodes=remaining_nodes,
            max_source_bytes=remaining_source_bytes,
            max_scalar_bytes=limits.max_snapshot_bytes,
        )
        if limit_exceeded or node_count > remaining_nodes or source_bytes > remaining_source_bytes:
            raise _GuestFailure("oversized_snapshot")
        total_nodes += node_count
        total_source_bytes += source_bytes
        # One accessibility node can reuse any source-derived accessible name
        # in its frame (for example through aria-labelledby). Bound that
        # expansion before Playwright builds the AI-mode snapshot. The fixed
        # envelope covers roles, states, indentation, refs, and YAML framing;
        # the serialization multiplier covers worst-case escaping.
        materialization_upper_bound = total_nodes * (
            total_source_bytes * _INTERACTIVE_ACCESSIBILITY_SERIALIZATION_MULTIPLIER
            + _INTERACTIVE_ACCESSIBILITY_NODE_ENVELOPE_BYTES
        )
        if materialization_upper_bound > _INTERACTIVE_MAX_ACCESSIBILITY_MATERIALIZATION_BYTES:
            raise _GuestFailure("oversized_snapshot")
        remaining_nodes -= node_count
        remaining_source_bytes -= source_bytes


async def _interactive_frame_ids(cdp: Any) -> tuple[str, ...]:
    """Return a bounded, structurally validated frame-tree identity census."""

    frame_tree = await cdp.send("Page.getFrameTree")
    if type(frame_tree) is not dict or type(frame_tree.get("frameTree")) is not dict:
        raise _GuestFailure("browser_crash")
    pending: list[dict[str, Any]] = [frame_tree["frameTree"]]
    frame_ids: list[str] = []
    seen: set[str] = set()
    while pending:
        tree = pending.pop()
        frame = tree.get("frame")
        if type(frame) is not dict:
            raise _GuestFailure("browser_crash")
        frame_id = frame.get("id")
        if type(frame_id) is not str or not 0 < len(frame_id) <= 256 or frame_id in seen:
            raise _GuestFailure("browser_crash")
        seen.add(frame_id)
        frame_ids.append(frame_id)
        if len(frame_ids) > _MAX_FRAME_DOCUMENTS:
            raise _GuestFailure("oversized_snapshot")
        raw_children = tree.get("childFrames", [])
        if type(raw_children) is not list:
            raise _GuestFailure("browser_crash")
        children: list[dict[str, Any]] = []
        for child in raw_children:
            if type(child) is not dict:
                raise _GuestFailure("browser_crash")
            children.append(child)
        pending.extend(reversed(children))
    return tuple(frame_ids)


async def _interactive_frame_snapshot_census(
    cdp: Any,
    frame_id: str,
    *,
    max_nodes: int,
    max_source_bytes: int,
    max_scalar_bytes: int,
) -> tuple[int, int, bool]:
    """Bound one frame's DOM and accessibility-bearing UTF-8 source material."""

    execution_context_id = await _create_isolated_world(cdp, frame_id)
    projection = await cdp.send(
        "Runtime.evaluate",
        {
            "contextId": execution_context_id,
            "expression": """(() => {
            const nodeLimit = __CAYU_NODE_LIMIT__;
            const sourceLimit = __CAYU_SOURCE_LIMIT__;
            const scalarLimit = __CAYU_SCALAR_LIMIT__;
            const root = document.documentElement;
            let nodeCount = root ? 1 : 0;
            let sourceBytes = 0;
            let limitExceeded = nodeCount > nodeLimit;
            const consume = value => {
                if (typeof value !== "string") return;
                let valueBytes = 0;
                for (let index = 0; index < value.length; index += 1) {
                    const code = value.charCodeAt(index);
                    let encodedBytes = 0;
                    if (code < 0x80) encodedBytes = 1;
                    else if (code < 0x800) encodedBytes = 2;
                    else if (code >= 0xD800 && code <= 0xDBFF &&
                            index + 1 < value.length) {
                        const trailing = value.charCodeAt(index + 1);
                        if (trailing >= 0xDC00 && trailing <= 0xDFFF) {
                            encodedBytes = 4;
                            index += 1;
                        } else encodedBytes = 3;
                    } else encodedBytes = 3;
                    sourceBytes += encodedBytes;
                    valueBytes += encodedBytes;
                    if (sourceBytes > sourceLimit || valueBytes > scalarLimit) {
                        limitExceeded = true;
                        return;
                    }
                }
            };
            const consumeElement = element => {
                for (const attribute of element.attributes) {
                    const name = attribute.name.toLowerCase();
                    if (name === "role" || name === "alt" || name === "title" ||
                            name === "href" ||
                            name === "placeholder" || name === "name" ||
                            name === "value" || name.startsWith("aria-")) {
                        consume(attribute.value);
                        if (limitExceeded) return;
                    }
                }
                if (element instanceof HTMLInputElement ||
                        element instanceof HTMLTextAreaElement ||
                        element instanceof HTMLSelectElement) {
                    consume(element.value);
                }
                if (limitExceeded) return;
                try {
                    for (const pseudo of ["::before", "::after", "::marker"]) {
                        const content = getComputedStyle(element, pseudo).content;
                        if (content !== "none" && content !== "normal") consume(content);
                        if (limitExceeded) return;
                    }
                } catch {
                    // If computed accessibility-bearing content cannot be
                    // inspected, fail closed before snapshot materialization.
                    limitExceeded = true;
                }
            };
            if (root && !limitExceeded) {
                const pending = [root];
                scan: while (pending.length > 0) {
                    const node = pending.pop();
                    if (node.nodeType === Node.TEXT_NODE) {
                        const parentName = node.parentElement?.localName;
                        if (parentName !== "script" && parentName !== "style" &&
                                parentName !== "noscript" && parentName !== "template") {
                            consume(node.nodeValue);
                            if (limitExceeded) break scan;
                        }
                    } else if (node.nodeType === Node.ELEMENT_NODE) {
                        consumeElement(node);
                        if (limitExceeded) break scan;
                    }
                    if (node.nodeType === Node.ELEMENT_NODE && node.shadowRoot) {
                        nodeCount += 1;
                        if (nodeCount > nodeLimit) {
                            limitExceeded = true;
                            break scan;
                        }
                        pending.push(node.shadowRoot);
                    }
                    for (let child = node.lastChild; child; child = child.previousSibling) {
                        nodeCount += 1;
                        if (nodeCount > nodeLimit) {
                            limitExceeded = true;
                            break scan;
                        }
                        pending.push(child);
                    }
                }
            }
            return {
                node_count: nodeCount,
                source_bytes: sourceBytes,
                limit_exceeded: limitExceeded,
            };
        })()""".replace("__CAYU_NODE_LIMIT__", str(max_nodes))
            .replace("__CAYU_SOURCE_LIMIT__", str(max_source_bytes))
            .replace("__CAYU_SCALAR_LIMIT__", str(max_scalar_bytes)),
            "returnByValue": True,
            "awaitPromise": False,
            "userGesture": False,
        },
    )
    if (
        type(projection) is not dict
        or projection.get("exceptionDetails") is not None
        or type(projection.get("result")) is not dict
        or projection["result"].get("type") != "object"
        or type(projection["result"].get("value")) is not dict
    ):
        raise _GuestFailure("browser_crash")
    extracted = projection["result"]["value"]
    if (
        set(extracted) != {"limit_exceeded", "node_count", "source_bytes"}
        or type(extracted.get("node_count")) is not int
        or extracted["node_count"] < 0
        or type(extracted.get("source_bytes")) is not int
        or extracted["source_bytes"] < 0
        or type(extracted.get("limit_exceeded")) is not bool
    ):
        raise _GuestFailure("browser_crash")
    return (
        extracted["node_count"],
        extracted["source_bytes"],
        extracted["limit_exceeded"],
    )


async def _interactive_observation(
    state: _InteractivePage,
    limits: _InteractiveLimits,
    *,
    browser_version: str,
    visual_policy: dict[str, Any] | None = None,
    visual_artifacts: list[bytes] | None = None,
) -> dict[str, Any]:
    page = state.page
    blocked = state.access_evidence is not None
    if state.cdp is None:
        raise _GuestFailure("browser_crash")
    primary_failure: BaseException | None = None
    scripts_disabled = False
    animations_frozen = False
    visual_candidate = None
    visual_evidence = None
    revision = f"br_{secrets.token_hex(16)}"
    try:
        scripts_disabled = True
        await state.cdp.send("Emulation.setScriptExecutionDisabled", {"value": True})
        animations_frozen = True
        freeze_result = await state.cdp.send(
            "Animation.setPlaybackRate",
            {"playbackRate": 0},
        )
        if type(freeze_result) is not dict:
            raise _GuestFailure("browser_crash")
        if blocked:
            snapshot = ""
            refs: dict[str, str] = {}
            ref_targets: dict[str, Any] = {}
            ref_metadata: dict[str, tuple[str, str]] = {}
            truncation: list[str] = []
        else:
            await _admit_interactive_snapshot_materialization(state, limits)
            if visual_policy is not None:
                if state.visual_owner is None:
                    raise VisualGuestFailure("visual_mode_disabled")
                visual_candidate = await state.visual_owner.collect(
                    visual_policy,
                    max_dom_nodes=limits.max_dom_nodes,
                )
            raw_snapshot = await page.locator("body").aria_snapshot(
                mode="ai",
                depth=_ACCESSIBILITY_SNAPSHOT_DEPTH,
                timeout=max(1_000, limits.max_wait_ms),
            )
            if type(raw_snapshot) is not str:
                raise _GuestFailure("browser_crash")
            snapshot, refs, ref_metadata, truncation = _interactive_snapshot(
                raw_snapshot,
                limits,
            )
            ref_targets = {}
            for opaque, internal in refs.items():
                target = await page.locator(f"aria-ref={internal}").element_handle()
                if target is None:
                    raise _GuestFailure("browser_crash")
                ref_targets[opaque] = target
        ref_input_evidence = await _interactive_file_input_evidence(ref_targets)
        url = state.public_url if blocked else page.url
        if type(url) is not str:
            raise _GuestFailure("browser_crash")
        split = urlsplit(url)
        scheme = split.scheme.lower()
        if scheme == "https":
            if blocked:
                if split.hostname is None:
                    raise _GuestFailure("browser_crash")
                url = f"https://{split.hostname.lower()}/"
                truncation.append("url")
            if len(url.encode("utf-8", errors="replace")) > _MAX_URL_LENGTH:
                try:
                    port = split.port
                except ValueError as exc:
                    raise _GuestFailure("browser_crash") from exc
                if split.hostname is None or port not in {None, 443}:
                    raise _GuestFailure("browser_crash")
                url = f"https://{split.hostname.lower()}/"
                truncation.append("url")
            state.public_url = url
        elif scheme in {"data", "blob", "about"}:
            if state.public_url is None:
                raise _GuestFailure("browser_crash")
            url = state.public_url
            truncation.append("url")
        else:
            raise _GuestFailure("browser_crash")
        title = None if blocked else await page.title()
        if title is not None:
            title_bytes = title.encode("utf-8", errors="replace")
            if len(title_bytes) > _MAX_TITLE_BYTES:
                title = title_bytes[:_MAX_TITLE_BYTES].decode("utf-8", errors="ignore")
                truncation.append("title")
        if visual_policy is not None:
            if (
                blocked
                or visual_candidate is None
                or state.visual_owner is None
                or visual_artifacts is None
            ):
                raise VisualGuestFailure("visual_publication_denied")
            screenshot = await page.screenshot(type="png", full_page=False)
            if type(screenshot) is not bytes or len(screenshot) > min(
                limits.max_artifact_bytes, visual_policy["max_bytes"]
            ):
                raise _GuestFailure("oversized_artifact")
            width, height = _png_header_dimensions(screenshot)
            view = visual_candidate["viewport"]
            if (
                width != round(view[0] * view[2])
                or height != round(view[1] * view[2])
                or width * height > visual_policy["max_pixels"]
            ):
                raise VisualGuestFailure("unstable_visual_observation")
            visual_evidence = await state.visual_owner.seal(
                visual_policy,
                visual_candidate,
                screenshot,
                session_id=state.session_id,
                page_id=state.page_id,
                revision=revision,
                control_epoch=state.control_epoch,
            )
            visual_artifacts.append(screenshot)
    except _GuestFailure as exc:
        primary_failure = exc
        if exc.code in {"oversized_response", "oversized_snapshot"}:
            state.limit_exceeded = True
            state.limit_error_code = "oversized_snapshot"
            raise _GuestFailure("oversized_snapshot") from exc
        raise
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        if state.observation_cleanup_task is not None:
            raise RuntimeError("Browser observation already owns guard restoration.")
        cleanup_task = asyncio.create_task(
            _restore_interactive_observation_guards(
                state.cdp,
                animations_frozen=animations_frozen,
                scripts_disabled=scripts_disabled,
            )
        )
        state.observation_cleanup_task = cleanup_task

        async def settle_restoration() -> tuple[BaseException, ...]:
            done, _ = await asyncio.wait(
                {cleanup_task}, timeout=_OBSERVATION_GUARD_SETTLEMENT_SECONDS
            )
            if not done:
                return (TimeoutError("Browser observation guard restoration exceeded its bound."),)
            try:
                return cleanup_task.result()
            except asyncio.CancelledError as error:
                failure = RuntimeError("Browser observation guard restoration was cancelled.")
                failure.__cause__ = error
                return (failure,)
            except BaseException as error:
                return (error,)

        cleanup_outcome = await _await_browser_cleanup_resisting_cancellation(
            asyncio.create_task(settle_restoration())
        )
        if cleanup_outcome.errors:
            state.observation_cleanup_disposition = "uncertain"
            state.revision = None
            state.clear_refs()
            if state.visual_owner is not None:
                state.visual_owner.invalidate()
            if visual_artifacts is not None:
                visual_artifacts.clear()
        else:
            state.observation_cleanup_task = None
        process_control = next(
            (
                error
                for error in cleanup_outcome.errors
                if isinstance(error, (SystemExit, KeyboardInterrupt, GeneratorExit))
            ),
            None,
        )
        if cleanup_outcome.cancellation is None and process_control is not None:
            raise process_control from _browser_cleanup_evidence(
                primary_failure,
                tuple(error for error in cleanup_outcome.errors if error is not process_control),
            )
        if cleanup_outcome.cancellation is not None:
            cause = _browser_cleanup_evidence(primary_failure, cleanup_outcome.errors)
            if cause is None:
                raise cleanup_outcome.cancellation
            raise cleanup_outcome.cancellation from cause
        if isinstance(primary_failure, asyncio.CancelledError):
            if cleanup_outcome.errors:
                raise primary_failure from _browser_cleanup_evidence(
                    None,
                    cleanup_outcome.errors,
                )
        elif cleanup_outcome.errors:
            if primary_failure is None:
                raise _GuestFailure("browser_crash") from _browser_cleanup_evidence(
                    None,
                    cleanup_outcome.errors,
                )
            raise primary_failure from _browser_cleanup_evidence(None, cleanup_outcome.errors)
    state.revision = revision
    state.last_observation_revision = state.revision
    state.refs = refs
    state.ref_targets = ref_targets
    return {
        **({"visual": visual_evidence} if visual_evidence is not None else {}),
        "session_id": state.session_id,
        "page_id": state.page_id,
        "revision": state.revision,
        "creation_epoch": state.creation_epoch,
        "control_epoch": state.control_epoch,
        "url": url,
        "title": title,
        "snapshot": snapshot,
        "refs": [
            {
                "ref": opaque,
                "role": ref_metadata[opaque][0],
                "name": ref_metadata[opaque][1],
                "element_type": (
                    "file_input" if ref_input_evidence[opaque] is not None else "element"
                ),
                "allows_multiple_files": ref_input_evidence[opaque],
            }
            for opaque in refs
        ],
        "load_state": "loaded",
        "access_state": "blocked" if blocked else "available",
        "access": state.access_evidence,
        "idle_timeout_seconds": limits.idle_timeout_seconds,
        "truncation_reasons": list(dict.fromkeys(truncation)),
        "backend_identity": {
            "backend": "playwright",
            "backend_version": PLAYWRIGHT_VERSION,
            "browser": "chromium",
            "browser_version": browser_version[:128] or "unknown",
            "worker_protocol": INTERACTIVE_PROTOCOL_VERSION,
            "worker_version": INTERACTIVE_WORKER_VERSION,
        },
    }


async def _interactive_file_input_evidence(
    targets: Mapping[str, Any],
) -> dict[str, bool | None]:
    evidence: dict[str, bool | None] = {}
    for opaque, target in targets.items():
        raw = await target.evaluate(
            """element => ({
                is_file_input: element instanceof HTMLInputElement &&
                    element.type.toLowerCase() === "file",
                multiple: element instanceof HTMLInputElement && element.multiple,
            })"""
        )
        if (
            type(raw) is not dict
            or set(raw) != {"is_file_input", "multiple"}
            or type(raw.get("is_file_input")) is not bool
            or type(raw.get("multiple")) is not bool
        ):
            raise _GuestFailure("browser_crash")
        evidence[opaque] = raw["multiple"] if raw["is_file_input"] else None
    return evidence


async def _restore_interactive_observation_guards(
    cdp: Any,
    *,
    animations_frozen: bool,
    scripts_disabled: bool,
) -> tuple[BaseException, ...]:
    """Restore every observation guard while retaining exact cleanup failures."""

    errors: list[BaseException] = []
    if animations_frozen:
        try:
            restore_result = await cdp.send(
                "Animation.setPlaybackRate",
                {"playbackRate": 1},
            )
            if type(restore_result) is not dict:
                raise _GuestFailure("browser_crash")
        except asyncio.CancelledError as exc:
            failure = RuntimeError("Browser animation guard cleanup cancelled unexpectedly.")
            failure.__cause__ = exc
            errors.append(failure)
        except BaseException as exc:
            errors.append(exc)
    if scripts_disabled:
        try:
            await cdp.send(
                "Emulation.setScriptExecutionDisabled",
                {"value": False},
            )
        except asyncio.CancelledError as exc:
            failure = RuntimeError("Browser script guard cleanup cancelled unexpectedly.")
            failure.__cause__ = exc
            errors.append(failure)
        except BaseException as exc:
            errors.append(exc)
    return tuple(errors)


def _interactive_snapshot(
    raw_snapshot: str,
    limits: _InteractiveLimits,
) -> tuple[str, dict[str, str], dict[str, tuple[str, str]], list[str]]:
    output: list[str] = []
    refs: dict[str, str] = {}
    ref_metadata: dict[str, tuple[str, str]] = {}
    truncation: list[str] = []
    used_bytes = 0
    for line in raw_snapshot.splitlines():
        element_match = _INTERACTIVE_ELEMENT_PATTERN.match(line)
        ref_match = (
            None
            if element_match is None or element_match.group(1) == "text"
            else _INTERACTIVE_REF_PATTERN.match(line, element_match.end())
        )
        if ref_match is not None and len(refs) >= limits.max_refs:
            truncation.append("refs")
            continue
        replaced = line
        if ref_match is not None:
            internal = ref_match.group("ref")
            if internal not in refs.values():
                if len(refs) >= limits.max_refs:
                    truncation.append("refs")
                    continue
                opaque = f"ref_{secrets.token_hex(12)}"
                refs[opaque] = internal
                ref_metadata[opaque] = _interactive_element_metadata(line)
            else:
                opaque = next(key for key, value in refs.items() if value == internal)
            replaced = (
                line[: ref_match.start()]
                + ref_match.group("prefix")
                + f"[ref={opaque}]"
                + line[ref_match.end() :]
            )
        if not replaced:
            continue
        encoded = (replaced + "\n").encode("utf-8", errors="replace")
        if used_bytes + len(encoded) > limits.max_snapshot_bytes:
            truncation.append("snapshot")
            break
        output.append(replaced)
        used_bytes += len(encoded)
    return "\n".join(output), refs, ref_metadata, truncation


def _interactive_element_metadata(line: str) -> tuple[str, str]:
    matched = _INTERACTIVE_ELEMENT_PATTERN.match(line)
    if matched is None:
        return "element", ""
    role = matched.group(1)
    raw_name = matched.group(2) or ""
    name = (
        matched.group(3)
        if matched.group(3) is not None
        else raw_name.replace(r"\"", '"').replace(r"\\", "\\")
    )
    encoded = name.encode("utf-8", errors="replace")
    if len(encoded) > _INTERACTIVE_MAX_ELEMENT_TEXT_BYTES:
        name = encoded[:_INTERACTIVE_MAX_ELEMENT_TEXT_BYTES].decode("utf-8", errors="ignore")
    return role, name


def _interactive_success_payload(
    observation: dict[str, Any] | None,
    *,
    artifact: dict[str, Any] | None = None,
    page_set: dict[str, Any] | None = None,
    page_delta: dict[str, Any] | None = None,
    profile_output_protected: bool = False,
    operation_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
        "worker_version": INTERACTIVE_WORKER_VERSION,
        "playwright_version": PLAYWRIGHT_VERSION,
        "kind": "success",
        "allocation_disposition": "live",
        "artifacts": [] if artifact is None else [artifact],
        "profile_output_protected": profile_output_protected,
    }
    if observation is not None:
        payload["observation"] = observation
    if page_set is not None:
        payload["page_set"] = page_set
    if page_delta is not None:
        payload["page_delta"] = page_delta
    if operation_evidence is not None:
        payload["operation_evidence"] = operation_evidence
    return payload


def _interactive_closed_payload() -> dict[str, Any]:
    return {
        "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
        "worker_version": INTERACTIVE_WORKER_VERSION,
        "playwright_version": PLAYWRIGHT_VERSION,
        "kind": "closed",
        "allocation_disposition": "retired",
    }


def _interactive_error_payload(
    error: _GuestFailure,
    *,
    page_set: dict[str, Any] | None = None,
    page_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stable = error.code
    if stable not in VISUAL_FAILURE_CODES | {
        "access_blocked",
        "actionability_failed",
        "allocation_lost",
        "artifact_write_failed",
        "browser_crash",
        "browser_unavailable",
        "capability_refused",
        "cleanup_failed",
        "destination_denied",
        "download_failed",
        "fetch_failed",
        "history_unavailable",
        "incompatible_browser",
        "incompatible_upload_target",
        "invalid_scroll",
        "missing_element",
        "navigation_timeout",
        "operation_conflict",
        "operation_not_dispatched",
        "outcome_ambiguous",
        "oversized_artifact",
        "oversized_response",
        "oversized_snapshot",
        "policy_denied",
        "redirect_denied",
        "resource_exhausted",
        "restoration_required",
        "session_closed",
        "timeout",
        "unsafe_reload",
        "artifact_refused",
        "artifact_unavailable",
        "upload_cleanup_failed",
        "upload_materialization_failed",
        "upload_too_large",
    }:
        stable = "browser_crash"
    payload: dict[str, Any] = {
        "protocol_version": INTERACTIVE_PROTOCOL_VERSION,
        "worker_version": INTERACTIVE_WORKER_VERSION,
        "playwright_version": PLAYWRIGHT_VERSION,
        "kind": "error",
        "allocation_disposition": error.allocation_disposition,
        "error": stable,
        **({"access": error.access} if error.access is not None else {}),
    }
    if page_set is not None:
        payload["page_set"] = page_set
    if page_delta is not None:
        payload["page_delta"] = page_delta
    return payload


def _interactive_operation_fingerprint(request: _InteractiveRequest) -> str:
    material = asdict(request)
    material.pop("reconcile_only", None)
    material["upload_files"] = [item.fingerprint_material() for item in request.upload_files]
    encoded = json.dumps(
        material,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return __import__("hashlib").sha256(b"cayu-browser-guest-operation-v1\0" + encoded).hexdigest()


def _interactive_playwright_error(operation: str, error: BaseException) -> _GuestFailure:
    is_target_closed = type(error).__name__ == "TargetClosedError" and type(
        error
    ).__module__.startswith("playwright")
    if is_target_closed:
        return _GuestFailure("browser_crash")
    is_timeout = isinstance(error, TimeoutError) or (
        type(error).__name__ == "TimeoutError" and type(error).__module__.startswith("playwright")
    )
    if operation in {"navigate", "back", "forward", "reload"} and is_timeout:
        return _GuestFailure("navigation_timeout")
    if operation == "scroll" and is_timeout:
        return _GuestFailure("timeout")
    if operation in {"click", "fill", "select", "press", "hover"}:
        return _GuestFailure("actionability_failed")
    if operation == "upload":
        # Selection can run page handlers before an acknowledgement is lost.
        return _GuestFailure("outcome_ambiguous")
    if operation == "download":
        return _GuestFailure("download_failed")
    if operation == "wait" and is_timeout:
        return _GuestFailure("timeout")
    return _GuestFailure("browser_crash")


def _interactive_runtime_failure(
    operation: str,
    error: BaseException,
    *,
    browser_connected: bool,
    page_open: bool,
) -> _GuestFailure:
    if not browser_connected:
        return _GuestFailure("browser_crash")
    if _is_playwright_network_failure(error):
        return _GuestFailure("fetch_failed")
    if (
        operation == "navigate"
        and not page_open
        and type(error).__module__.startswith("playwright")
    ):
        # A failed CONNECT can close only the provisional page while the browser
        # allocation remains healthy. Chromium does not expose whether the proxy
        # rejected policy, capacity, or another transport boundary.
        return _GuestFailure("fetch_failed")
    if not page_open:
        return _GuestFailure("browser_crash")
    return _interactive_playwright_error(operation, error)


def _is_proxy_tunnel_failure(error: BaseException) -> bool:
    """Recognize Chromium's ambiguous signal for a failed CONNECT.

    Chromium does not expose whether the proxy returned a policy 403, capacity
    503, or another CONNECT failure. Callers may use authenticated broker
    evidence when it exists; this signal alone is only a generic fetch failure.
    """

    return _playwright_error_chain_has_signal(error, "ERR_TUNNEL_CONNECTION_FAILED")


def _is_playwright_network_failure(error: BaseException) -> bool:
    return _playwright_error_chain_has_signal(error, "net::ERR_")


def _playwright_error_chain_has_signal(error: BaseException, signal: str) -> bool:
    pending = [error]
    seen: set[int] = set()
    while pending and len(seen) < 32:
        candidate = pending.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if type(candidate).__module__.startswith("playwright") and signal in str(candidate):
            return True
        if isinstance(candidate, BaseExceptionGroup):
            pending.extend(candidate.exceptions)
        if candidate.__cause__ is not None:
            pending.append(candidate.__cause__)
        if candidate.__context__ is not None:
            pending.append(candidate.__context__)
    return False


async def _interactive_daemon_main(session_id: str) -> int:
    session_id = _interactive_identifier(session_id)
    socket_path = _interactive_socket_path(session_id)
    daemon = _InteractiveDaemon(session_id)
    server: asyncio.AbstractServer | None = None
    try:
        await daemon.start()

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                raw_line = await reader.readuntil(b"\n")
                if len(raw_line) > _INTERACTIVE_MAX_PRIVATE_REQUEST_BYTES:
                    raise _GuestFailure("incompatible_browser")
                raw = json.loads(raw_line.decode("utf-8"))
                if type(raw) is dict and raw.get("protocol_version") == CONTROL_BOOTSTRAP_PROTOCOL:
                    target, material = _private_control_bootstrap(raw)
                    raw.clear()
                    raw_line = b""
                    if target != daemon.session_id:
                        raise _GuestFailure("incompatible_browser")
                    try:
                        response = await daemon.bootstrap_operator_channel(material)
                    finally:
                        material.clear()
                else:
                    response = await daemon.execute(_interactive_request_from_json(raw))
            except _GuestFailure as exc:
                response = _interactive_error_payload(exc)
            except (json.JSONDecodeError, UnicodeError, asyncio.LimitOverrunError):
                response = _interactive_error_payload(_GuestFailure("incompatible_browser"))
            except Exception:
                response = _interactive_error_payload(_GuestFailure("browser_crash"))
            encoded = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
            if len(encoded) > _INTERACTIVE_MAX_MESSAGE_BYTES:
                encoded = json.dumps(
                    _interactive_error_payload(_GuestFailure("oversized_response")),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            writer.write(encoded + b"\n")
            with contextlib.suppress(Exception):
                await writer.drain()
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            if daemon.close_after_response:
                daemon.close_requested.set()

        with contextlib.suppress(OSError):
            socket_path.unlink()
        server = await asyncio.start_unix_server(
            handle,
            path=str(socket_path),
            limit=_INTERACTIVE_MAX_PRIVATE_REQUEST_BYTES + 1,
        )
        os.chmod(socket_path, 0o600)
        daemon.last_activity = asyncio.get_running_loop().time()
        await _wait_for_interactive_shutdown(daemon)
        return 0
    finally:
        if server is not None:
            server.close()
            await server.wait_closed()
        cleanup_ok = await daemon.close()
        if cleanup_ok:
            # One full-digest marker preserves positive quiescence evidence for
            # this exact allocation identity across daemon process loss.
            _record_interactive_retirement(session_id)
        with contextlib.suppress(OSError):
            socket_path.unlink()


async def _wait_for_interactive_shutdown(daemon: _InteractiveDaemon) -> None:
    while True:
        if daemon.close_requested.is_set():
            return
        async with daemon.lock:
            if daemon.close_requested.is_set():
                return
            if daemon.configuration_limits is not None:
                try:
                    await daemon._expire_background_pages(
                        daemon.configuration_limits,
                        delta=None,
                    )
                except _GuestFailure:
                    # No request owns this expiry failure.  Retire through the
                    # daemon's final cleanup owner; only a fully settled close
                    # publishes the allocation-retirement marker.
                    daemon.closing = True
                    daemon.close_after_response = True
                    daemon.close_requested.set()
                    return
            remaining = max(
                0.0,
                daemon.last_activity
                + daemon.idle_timeout_seconds
                - asyncio.get_running_loop().time(),
                daemon.control.retirement_deferral_seconds(),
            )
            background_remaining = (
                tuple(
                    max(
                        0.0,
                        state.background_since
                        + daemon.configuration_limits.max_background_lifetime_seconds
                        - asyncio.get_running_loop().time(),
                    )
                    for state in daemon.pages.values()
                    if state.lifecycle == "background" and state.background_since is not None
                )
                if daemon.configuration_limits is not None
                else ()
            )
            if background_remaining:
                remaining = min(remaining, min(background_remaining))
            if remaining == 0:
                daemon.closing = True
                daemon.idle_expired = True
                daemon.close_requested.set()
                return
        await asyncio.sleep(min(remaining, _INTERACTIVE_IDLE_POLL_SECONDS))


def _error_payload(error: _GuestFailure) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "worker_version": WORKER_VERSION,
        "playwright_version": PLAYWRIGHT_VERSION,
        "kind": "error",
        "error": error.code,
    }
    if error.status_code is not None:
        payload["status_code"] = error.status_code
    if error.access is not None:
        payload["access"] = error.access
    if error.effective_origin is not None:
        payload["effective_source_url"] = error.effective_origin
    return payload


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == _PROFILE_CLEANUP_ARGUMENT:
        return _temporary_profile_cleanup_main(sys.argv[2], sys.argv[3])
    if len(sys.argv) == 3 and sys.argv[1] == _INTERACTIVE_DAEMON_ARGUMENT:
        try:
            return asyncio.run(_interactive_daemon_main(sys.argv[2]))
        except Exception:
            return 1
    if len(sys.argv) != 1:
        result = _error_payload(_GuestFailure("incompatible_browser"))
    else:
        try:
            raw_stdin = sys.stdin.buffer.read(_INTERACTIVE_MAX_PRIVATE_REQUEST_BYTES + 1)
            if len(raw_stdin) > _INTERACTIVE_MAX_PRIVATE_REQUEST_BYTES:
                raise _GuestFailure("incompatible_browser")
            raw_request = json.loads(raw_stdin.decode("utf-8"))
            if (
                type(raw_request) is dict
                and raw_request.get("protocol_version") == CONTROL_BOOTSTRAP_PROTOCOL
            ):
                try:
                    result = asyncio.run(_run_private_control_bootstrap(raw_request))
                finally:
                    raw_request.clear()
                    raw_stdin = b""
            elif (
                type(raw_request) is dict
                and raw_request.get("protocol_version") == INTERACTIVE_PROTOCOL_VERSION
            ):
                try:
                    result = asyncio.run(_run_interactive_request(raw_request))
                except _GuestFailure as exc:
                    result = _interactive_error_payload(exc)
            else:
                request = _request_from_json(raw_request)
                result = asyncio.run(_run(request))
        except (json.JSONDecodeError, UnicodeError):
            result = _error_payload(_GuestFailure("incompatible_browser"))
        except _GuestFailure as exc:
            result = _error_payload(exc)
        except Exception:
            result = _error_payload(_GuestFailure("browser_crash"))
    encoded_result = json.dumps(
        result,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    sys.stdout.buffer.write(encoded_result + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
