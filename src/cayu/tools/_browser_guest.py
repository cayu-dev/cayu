"""Narrow JSON browser-fetch worker executed inside a Cayu runner.

This module is intentionally not imported by :mod:`cayu.tools.browser`. The
versioned browser image copies it to ``/opt/cayu-browser/worker.py`` and invokes
it as a standalone program, keeping Playwright and Chromium out of the trusted
Cayu host process.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import math
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Never
from urllib.parse import urljoin, urlsplit

PROTOCOL_VERSION = "cayu.browser-fetch.v1"
WORKER_VERSION = "1"
PLAYWRIGHT_VERSION = "1.62.0"
_BROKER_ERROR_HEADER = "x-cayu-egress-error"
_MAX_URL_LENGTH = 8192
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_CONTENT_BYTES = 256 * 1024
_MAX_REDIRECTS = 10
_MAX_REQUESTS = 512
_MAX_TIMEOUT_SECONDS = 120.0
_MAX_TITLE_BYTES = 512
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


class _GuestFailure(RuntimeError):
    def __init__(self, code: str, *, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


@dataclass(frozen=True)
class _Limits:
    max_response_bytes: int
    max_content_bytes: int
    timeout_seconds: float
    max_redirects: int
    max_requests: int


@dataclass(frozen=True)
class _Request:
    url: str
    limits: _Limits


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
    cleanup_failed: bool = False
    redirects: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class _BrowserCleanupOutcome:
    errors: tuple[BaseException, ...] = ()
    cancellation: asyncio.CancelledError | None = None


@dataclass(frozen=True)
class _TemporaryProfileOwner:
    home: Path
    process: subprocess.Popen[bytes]
    control_fd: int

    @property
    def pid(self) -> int:
        return self.process.pid


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
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < minimum or normalized > maximum:
        raise _GuestFailure("incompatible_browser")
    return normalized


def _request_from_json(raw: Any) -> _Request:
    if type(raw) is not dict or set(raw) != {
        "expected_playwright_version",
        "limits",
        "operation",
        "protocol_version",
        "url",
        "worker_version",
    }:
        raise _GuestFailure("incompatible_browser")
    if (
        raw["protocol_version"] != PROTOCOL_VERSION
        or raw["worker_version"] != WORKER_VERSION
        or raw["expected_playwright_version"] != PLAYWRIGHT_VERSION
        or raw["operation"] != "fetch"
    ):
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
    limits = raw["limits"]
    if type(limits) is not dict or set(limits) != {
        "max_content_bytes",
        "max_redirects",
        "max_requests",
        "max_response_bytes",
        "timeout_seconds",
    }:
        raise _GuestFailure("incompatible_browser")
    return _Request(
        url=url,
        limits=_Limits(
            max_response_bytes=_bounded_int(
                limits["max_response_bytes"],
                minimum=1,
                maximum=_MAX_RESPONSE_BYTES,
            ),
            max_content_bytes=_bounded_int(
                limits["max_content_bytes"],
                minimum=1,
                maximum=_MAX_CONTENT_BYTES,
            ),
            timeout_seconds=_bounded_float(
                limits["timeout_seconds"],
                minimum=0.001,
                maximum=_MAX_TIMEOUT_SECONDS,
            ),
            max_redirects=_bounded_int(
                limits["max_redirects"],
                minimum=0,
                maximum=_MAX_REDIRECTS,
            ),
            max_requests=_bounded_int(
                limits["max_requests"],
                minimum=1,
                maximum=_MAX_REQUESTS,
            ),
        ),
    )


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
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        shutil.rmtree(home)
    except FileNotFoundError:
        return 0
    except (OSError, TimeoutError):
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    return 0


async def _start_temporary_profile_owner(
    *,
    timeout_seconds: float,
) -> _TemporaryProfileOwner:
    try:
        home = Path(
            tempfile.mkdtemp(
                prefix=_TEMPORARY_PROFILE_PREFIX,
                dir=str(_TEMPORARY_PROFILE_ROOT),
            )
        )
    except OSError as exc:
        raise _GuestFailure("cleanup_failed") from exc
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
        command = _temporary_profile_cleanup_command(home, timeout_seconds=timeout_seconds)
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
        while True:
            try:
                ready = os.read(ready_read, 1)
            except BlockingIOError:
                await asyncio.sleep(0.001)
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
    try:
        os.close(owner.control_fd)
    except OSError as exc:
        errors.append(exc)
    timeout_seconds = max(0.0, timeout_seconds)
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    kill_reserve = min(0.1, timeout_seconds * 0.25)
    graceful_seconds = max(0.0, timeout_seconds - kill_reserve)
    timed_out = False
    returncode: int | None = None
    try:
        returncode = await _wait_temporary_profile_owner(
            owner.process,
            timeout_seconds=graceful_seconds,
        )
        timed_out = returncode is None
        if timed_out:
            try:
                owner.process.kill()
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.append(exc)
            remaining_seconds = max(0.0, deadline - asyncio.get_running_loop().time())
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


async def _install_browser_ca(home: Path, ca_path: Path) -> None:
    certutil = Path("/usr/bin/certutil")
    if not certutil.is_file() or not os.access(certutil, os.X_OK):
        raise _GuestFailure("incompatible_browser")
    database = home / ".pki" / "nssdb"
    database.mkdir(parents=True, mode=0o700)
    commands = (
        [str(certutil), "-N", "--empty-password", "-d", f"sql:{database}"],
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
    success_projection: tuple[str, str | None, str, tuple[str, ...]] | None = None
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
        context = await browser.new_context(
            accept_downloads=False,
            ignore_https_errors=False,
            java_script_enabled=True,
            service_workers="block",
        )
        page = await context.new_page()
        page.set_default_timeout(operation_timeout_ms)
        page.set_default_navigation_timeout(operation_timeout_ms)

        def abort_page() -> None:
            violation_observed.set()

        def unexpected_page_observed(_unexpected_page: Any) -> None:
            state.response_inspection_failed = True
            abort_page()

        context.on("page", unexpected_page_observed)

        async def route_request(route: Any, browser_request: Any) -> None:
            state.request_count += 1
            if state.request_count > state.max_requests:
                state.limit_exceeded = True
                await route.abort("blockedbyclient")
                abort_page()
                return
            if not _browser_request_is_admissible(browser_request.url):
                is_redirected_main_navigation = (
                    browser_request.is_navigation_request()
                    and browser_request.frame == page.main_frame
                    and browser_request.redirected_from is not None
                )
                _record_page_denial(
                    state,
                    ("redirect_denied" if is_redirected_main_navigation else "destination_denied"),
                )
                await route.abort("blockedbyclient")
                abort_page()
                return
            await route.continue_()

        await context.route("**/*", route_request)

        cdp = await context.new_cdp_session(page)
        await cdp.send("Network.enable")
        main_frame_id: str | None = None
        main_document_request_ids: set[str] = set()
        redirected_main_request_ids: set[str] = set()

        def request_will_be_sent(params: dict[str, Any]) -> None:
            nonlocal main_frame_id
            try:
                if params.get("type") != "Document":
                    return
                request_id = params.get("requestId")
                frame_id = params.get("frameId")
                if type(request_id) is not str or type(frame_id) is not str:
                    raise TypeError("Missing browser navigation identity.")
                if main_frame_id is None:
                    main_frame_id = frame_id
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
            raise _page_state_failure(state, redirects=redirects) or _GuestFailure("fetch_failed")
        await _cancel_task(violation_task)
        violation_task = None
        final_response = await navigation_task
        navigation_task = None
        await page.wait_for_timeout(_RENDER_SETTLE_MILLISECONDS)
        state_failure = _page_state_failure(state, redirects=redirects)
        if state_failure is not None:
            raise state_failure
        if final_response is None:
            raise _GuestFailure("fetch_failed")
        if final_response.status < 200 or final_response.status >= 300:
            raise _GuestFailure("http_status", status_code=final_response.status)
        final_headers = await final_response.all_headers()
        content_type = final_headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in _HTML_CONTENT_TYPES | _TEXT_CONTENT_TYPES:
            raise _GuestFailure("unsupported_content")
        extracted = await page.evaluate(
            """limits => {
                const value = document.body ? document.body.innerText : "";
                const title = document.title;
                return {
                    text: value.slice(0, limits.content + 1),
                    truncated: value.length > limits.content,
                    title: title.slice(0, limits.title + 1),
                    title_truncated: title.length > limits.title,
                };
            }""",
            {
                "content": request.limits.max_content_bytes,
                "title": _MAX_TITLE_BYTES,
            },
        )
        if (
            type(extracted) is not dict
            or type(extracted.get("text")) is not str
            or type(extracted.get("truncated")) is not bool
            or type(extracted.get("title")) is not str
            or type(extracted.get("title_truncated")) is not bool
        ):
            raise _GuestFailure("browser_crash")
        content, content_truncated = _normalized_text(
            extracted["text"],
            request.limits.max_content_bytes,
            preserve_lines=True,
        )
        content_truncated = content_truncated or extracted["truncated"]
        title, title_truncated = _normalized_text(
            extracted["title"],
            _MAX_TITLE_BYTES,
            preserve_lines=False,
        )
        title_truncated = title_truncated or extracted["title_truncated"]
        truncation_reasons: list[str] = []
        if title_truncated:
            truncation_reasons.append("title")
        if content_truncated:
            truncation_reasons.append("content")
        # Let work synchronously initiated by extraction reach the broker before
        # freezing page-authored JavaScript. Keep the response listeners active
        # while already-dispatched requests settle, then close the context before
        # publishing success so no later page activity can race the final check.
        await _wait_for_browser_violation(violation_observed)
        state_failure = _page_state_failure(state, redirects=redirects)
        if state_failure is not None:
            raise state_failure
        await cdp.send("Emulation.setScriptExecutionDisabled", {"value": True})
        await _wait_for_browser_violation(violation_observed)
        state_failure = _page_state_failure(state, redirects=redirects)
        if state_failure is not None:
            raise state_failure
        success_projection = (
            page.url,
            title or None,
            content,
            tuple(truncation_reasons),
        )
    except asyncio.CancelledError as exc:
        primary = exc
        raise
    except (PlaywrightTimeoutError, TimeoutError) as exc:
        primary = _page_state_failure(state, redirects=redirects) or _GuestFailure("timeout")
        raise primary from exc
    except _GuestFailure as exc:
        primary = exc
        raise
    except PlaywrightError as exc:
        primary = _page_state_failure(state, redirects=redirects) or _GuestFailure(
            "browser_crash" if launched else "browser_unavailable"
        )
        raise primary from exc
    except Exception as exc:
        primary = _GuestFailure("browser_crash")
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
    if success_projection is None:  # pragma: no cover - success construction invariant
        raise _GuestFailure("browser_crash")
    state_failure = _page_state_failure(state, redirects=redirects)
    if state_failure is not None:
        raise state_failure
    final_url, title, content, final_truncation_reasons = success_projection
    return {
        "protocol_version": PROTOCOL_VERSION,
        "worker_version": WORKER_VERSION,
        "playwright_version": PLAYWRIGHT_VERSION,
        "kind": "success",
        "requested_url": request.url,
        "final_url": final_url,
        "title": title,
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
            return _GuestFailure(state.denied_code)
        return _GuestFailure("fetch_failed")
    if state.response_inspection_failed:
        return _GuestFailure("fetch_failed")
    if len(redirects) > state.max_redirects:
        return _GuestFailure("redirect_denied")
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
    return payload


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == _PROFILE_CLEANUP_ARGUMENT:
        return _temporary_profile_cleanup_main(sys.argv[2], sys.argv[3])
    if len(sys.argv) != 1:
        result = _error_payload(_GuestFailure("incompatible_browser"))
    else:
        try:
            raw_stdin = sys.stdin.buffer.read(64 * 1024 + 1)
            if len(raw_stdin) > 64 * 1024:
                raise _GuestFailure("incompatible_browser")
            request = _request_from_json(json.loads(raw_stdin.decode("utf-8")))
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
