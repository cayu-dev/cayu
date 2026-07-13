from __future__ import annotations

import asyncio
import contextlib
import importlib
import posixpath
from collections.abc import Callable, Mapping
from math import isfinite
from types import ModuleType
from typing import Any, Literal

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.runners._cleanup import (
    DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
    DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
    DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
    RunnerCleanupPolicy,
    cleanup_runner_command_with_diagnostic,
    validate_cancel_timeout,
    validate_runner_cleanup_policy,
)
from cayu.runners._subprocess import (
    copy_runner_env,
    validate_output_limit,
    validate_stdin,
    validate_timeout,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    attach_cancellation_artifacts,
)

DEFAULT_MICROSANDBOX_IMAGE = "python:3.13"
DEFAULT_MICROSANDBOX_CWD = "/workspace"
DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS = 5.0
MICROSANDBOX_NAME_MAX_BYTES = 128
_MICROSANDBOX_REMOVE_INITIAL_BACKOFF_SECONDS = 0.05
_MICROSANDBOX_REMOVE_MAX_BACKOFF_SECONDS = 0.5
_MICROSANDBOX_CLEANUP_DIAGNOSTIC_TYPE = "cayu.microsandbox_cleanup.v1"

MicrosandboxCloseAction = Literal["remove", "stop", "detach", "none"]


class MicrosandboxCleanupError(RuntimeError):
    """Terminal bounded Microsandbox lifecycle cleanup failure."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = copy_json_value(diagnostic, "diagnostic")


class _MicrosandboxDeadlineExceeded(TimeoutError):
    pass


class _MicrosandboxCleanupExceptionGroup(BaseExceptionGroup):
    pass


class MicrosandboxRunner(Runner):
    """Executes commands in a Microsandbox microVM sandbox.

    The runner does not inherit the trusted host process environment. Pass
    explicit `env` values, preferably resolved at the environment/vault boundary.
    """

    isolation = "microsandbox"

    def __init__(
        self,
        sandbox: Any,
        *,
        name: str,
        default_cwd: str = DEFAULT_MICROSANDBOX_CWD,
        close_action: MicrosandboxCloseAction = "none",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        remove_timeout_s: float = DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
        env_overlay: Mapping[str, str] | None = None,
        sandbox_module: ModuleType | Any | None = None,
    ) -> None:
        if sandbox is None:
            raise TypeError("MicrosandboxRunner sandbox cannot be None.")
        self.name = _validate_sandbox_name(name)
        self.default_cwd = _validate_guest_root(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.cancel_timeout_s = validate_cancel_timeout(cancel_timeout_s)
        self.cancellation_cleanup = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        self.timeout_cleanup = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        self.remove_timeout_s = _validate_remove_timeout(remove_timeout_s)
        self.env_overlay = dict(env_overlay) if env_overlay else {}
        self._sandbox = sandbox
        self._sandbox_module = sandbox_module
        self._sftp_client: Any = None
        self._sftp: Any = None
        self._sftp_lock = asyncio.Lock()
        self._last_cleanup_diagnostic: dict[str, Any] | None = None
        self._remove_stop_completed = False
        self._remove_stop_status: str | None = None

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        image: Any = DEFAULT_MICROSANDBOX_IMAGE,
        default_cwd: str = DEFAULT_MICROSANDBOX_CWD,
        close_action: MicrosandboxCloseAction = "remove",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        remove_timeout_s: float = DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
        ensure_default_cwd: bool = True,
        env_overlay: Mapping[str, str] | None = None,
        sandbox_module: ModuleType | Any | None = None,
        **sandbox_options: Any,
    ) -> MicrosandboxRunner:
        """Create a sandbox and return a runner bound to it.

        Extra keyword arguments are passed through to `microsandbox.Sandbox.create`
        so applications can use provider-specific options such as volumes,
        network policy, resources, labels, secrets, and replace behavior.
        """

        module = _microsandbox_module(sandbox_module)
        sandbox_name = _validate_sandbox_name(name)
        guest_root = _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        removal_timeout = _validate_remove_timeout(remove_timeout_s)
        if type(ensure_default_cwd) is not bool:
            raise TypeError("MicrosandboxRunner ensure_default_cwd must be a bool.")
        sandbox = await module.Sandbox.create(
            sandbox_name,
            image=image,
            **dict(sandbox_options),
        )
        try:
            if ensure_default_cwd:
                await sandbox.exec("mkdir", ["-p", guest_root], cwd="/")
        except asyncio.CancelledError as exc:
            await _cleanup_created_sandbox_after_failure(
                module,
                sandbox,
                sandbox_name,
                exc,
                "Microsandbox setup was cancelled and cleanup failed.",
                remove_timeout_s=removal_timeout,
            )
            raise
        except Exception as exc:
            await _cleanup_created_sandbox_after_failure(
                module,
                sandbox,
                sandbox_name,
                exc,
                "Microsandbox setup failed and cleanup failed.",
                remove_timeout_s=removal_timeout,
            )
            raise
        return cls(
            sandbox,
            name=sandbox_name,
            default_cwd=guest_root,
            close_action=close_action,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            remove_timeout_s=removal_timeout,
            env_overlay=env_overlay,
            sandbox_module=module,
        )

    @classmethod
    async def from_existing(
        cls,
        name: str,
        *,
        default_cwd: str = DEFAULT_MICROSANDBOX_CWD,
        close_action: MicrosandboxCloseAction = "none",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        remove_timeout_s: float = DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
        env_overlay: Mapping[str, str] | None = None,
        sandbox_module: ModuleType | Any | None = None,
    ) -> MicrosandboxRunner:
        """Attach to an existing Microsandbox sandbox by name."""

        module = _microsandbox_module(sandbox_module)
        sandbox_name = _validate_sandbox_name(name)
        _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        removal_timeout = _validate_remove_timeout(remove_timeout_s)
        handle = await module.Sandbox.get(sandbox_name)
        sandbox = await handle.connect()
        return cls(
            sandbox,
            name=sandbox_name,
            default_cwd=default_cwd,
            close_action=close_action,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            remove_timeout_s=removal_timeout,
            env_overlay=env_overlay,
            sandbox_module=module,
        )

    async def close(self) -> None:
        """Apply the configured lifecycle action once."""

        if self._closed:
            return
        await self._close_sftp_session()
        if self.close_action == "none":
            self._last_cleanup_diagnostic = _microsandbox_cleanup_diagnostic(
                sandbox_name=self.name,
                action="none",
                status="skipped",
                timeout_s=self.remove_timeout_s,
            )
            self._closed = True
            return
        if self.close_action == "detach":
            detach = getattr(self._sandbox, "detach", None)
            try:
                if detach is not None:
                    await detach()
            except Exception as exc:
                self._record_failed_cleanup(action="detach", error=exc)
                raise
            self._last_cleanup_diagnostic = _microsandbox_cleanup_diagnostic(
                sandbox_name=self.name,
                action="detach",
                status="detached",
                timeout_s=self.remove_timeout_s,
            )
            self._closed = True
            return
        if self.close_action in {"stop", "remove"}:
            module = _microsandbox_module(self._sandbox_module)
            if self.close_action == "remove" and self._remove_stop_completed:
                stop_status = self._remove_stop_status
            else:
                try:
                    stop_status, already_removed = await _stop_sandbox(
                        module,
                        self._sandbox,
                        not_found_is_removed=self.close_action == "remove",
                    )
                except Exception as exc:
                    self._record_failed_cleanup(action="stop", error=exc)
                    raise
                if already_removed:
                    self._last_cleanup_diagnostic = _microsandbox_cleanup_diagnostic(
                        sandbox_name=self.name,
                        action="remove",
                        status="removed",
                        timeout_s=self.remove_timeout_s,
                        attempts=[
                            {
                                "attempt": 1,
                                "status": "already_removed",
                                "operation": "stop",
                            }
                        ],
                    )
                    self._closed = True
                    return
                if self.close_action == "remove":
                    self._remove_stop_completed = True
                    self._remove_stop_status = stop_status
            if self.close_action == "remove":
                self._last_cleanup_diagnostic = None
                try:
                    self._last_cleanup_diagnostic = await _remove_stopped_sandbox(
                        module,
                        self.name,
                        timeout_s=self.remove_timeout_s,
                        initial_status=stop_status,
                        record_diagnostic=self._set_last_cleanup_diagnostic,
                    )
                except MicrosandboxCleanupError as exc:
                    self._last_cleanup_diagnostic = copy_json_value(exc.diagnostic, "diagnostic")
                    raise
                except Exception as exc:
                    if self._last_cleanup_diagnostic is None:
                        self._record_failed_cleanup(action="remove", error=exc)
                    raise
            else:
                self._last_cleanup_diagnostic = _microsandbox_cleanup_diagnostic(
                    sandbox_name=self.name,
                    action="stop",
                    status="stopped",
                    timeout_s=self.remove_timeout_s,
                    observed_statuses=[] if stop_status is None else [stop_status],
                )
            self._closed = True
            return
        raise AssertionError(f"Unsupported Microsandbox close action: {self.close_action}")

    @property
    def last_cleanup_diagnostic(self) -> dict[str, Any] | None:
        """Return the latest lifecycle cleanup diagnostic, if close was attempted."""

        if self._last_cleanup_diagnostic is None:
            return None
        return copy_json_value(self._last_cleanup_diagnostic, "last_cleanup_diagnostic")

    def _record_failed_cleanup(self, *, action: str, error: Exception) -> None:
        diagnostic = _microsandbox_cleanup_diagnostic(
            sandbox_name=self.name,
            action=action,
            status="failed",
            timeout_s=self.remove_timeout_s,
            error=error,
        )
        _attach_microsandbox_cleanup_diagnostic(error, diagnostic)
        self._set_last_cleanup_diagnostic(diagnostic)

    def _set_last_cleanup_diagnostic(self, diagnostic: dict[str, Any]) -> None:
        self._last_cleanup_diagnostic = copy_json_value(
            diagnostic,
            "diagnostic",
        )

    def filesystem(self) -> Any:
        """Return the native Microsandbox filesystem API for workspace adapters."""

        if self._closed:
            raise RuntimeError("MicrosandboxRunner is closed.")
        return self._sandbox.fs

    async def real_path(self, path: str) -> str:
        """Resolve a guest path through Microsandbox's SFTP realpath API.

        The SSH client and SFTP channel are opened once and cached for reuse:
        listing a directory resolves one path per entry, and a fresh SSH
        handshake per call made large listings pathologically slow (~500
        handshakes to resolve 500 files). On any session error the cached
        session is dropped and the call is retried once against a fresh
        handshake, so a transient disconnect does not fail the whole listing.
        """

        if self._closed:
            raise RuntimeError("MicrosandboxRunner is closed.")
        async with self._sftp_lock:
            try:
                return await self._sftp_real_path(path)
            except Exception:
                await self._close_sftp_session()
                return await self._sftp_real_path(path)

    async def _sftp_real_path(self, path: str) -> str:
        sftp = await self._ensure_sftp_session()
        resolved = await sftp.real_path(path)
        if type(resolved) is not str or not resolved:
            raise RuntimeError("Microsandbox real_path returned an invalid path.")
        return posixpath.normpath(resolved)

    async def _ensure_sftp_session(self) -> Any:
        if self._sftp is not None:
            return self._sftp
        ssh = self._sandbox.ssh()
        client = await ssh.open_client(sftp=True)
        try:
            sftp = await client.sftp()
        except BaseException:
            await _close_quietly(client)
            raise
        self._sftp_client = client
        self._sftp = sftp
        return sftp

    async def _close_sftp_session(self) -> None:
        sftp = self._sftp
        client = self._sftp_client
        self._sftp = None
        self._sftp_client = None
        if sftp is not None:
            await _close_quietly(sftp)
        if client is not None:
            await _close_quietly(client)

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        if type(command) is not ExecCommand:
            raise TypeError("MicrosandboxRunner command must be an ExecCommand.")
        self._ensure_exec_open()

        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        if self.env_overlay:
            environment.update(self.env_overlay)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        sdk_stdin = standard_input.encode("utf-8") if standard_input is not None else None
        output_limit = validate_output_limit(output_limit_bytes)

        stdout = _LimitedBytes(output_limit)
        stderr = _LimitedBytes(output_limit)
        handle = None
        exit_code: int | None = None

        async def run_command() -> None:
            nonlocal exit_code
            nonlocal handle
            if command.kind == "process":
                if command.argv is None:
                    raise ValueError("Process commands require argv.")
                handle = await self._sandbox.exec_stream(
                    command.argv[0],
                    command.argv[1:],
                    cwd=working_dir,
                    env=environment,
                    timeout=float(timeout) if timeout is not None else None,
                    stdin=sdk_stdin,
                )
            else:
                if command.shell is None:
                    raise ValueError("Shell commands require a script.")
                handle = await self._sandbox.shell_stream(
                    command.shell,
                    cwd=working_dir,
                    env=environment,
                    timeout=float(timeout) if timeout is not None else None,
                    stdin=sdk_stdin,
                )

            async for event in handle:
                event_type = _exec_event_type(event)
                data = getattr(event, "data", None)
                if event_type == "stdout" and data is not None:
                    stdout.append(_event_bytes(data))
                elif event_type == "stderr" and data is not None:
                    stderr.append(_event_bytes(data))
                elif event_type == "exited":
                    code = getattr(event, "code", None)
                    if type(code) is int:
                        exit_code = code

            if exit_code is None:
                collected = await handle.collect()
                exit_code = _exec_output_exit_code(collected)
                _apply_collected_output(stdout, stderr, collected)

        try:
            await asyncio.wait_for(run_command(), timeout=timeout)
        except asyncio.CancelledError as exc:
            start_acknowledged = handle is not None
            cleanup = await cleanup_runner_command_with_diagnostic(
                self._sandbox,
                handle=handle,
                adapter="microsandbox",
                timeout_s=self.cancel_timeout_s,
                policy=self.cancellation_cleanup,
            )
            self._apply_cleanup_result(cleanup)
            if not start_acknowledged and self.cancellation_cleanup == "none":
                self._close_exec(
                    "microsandbox command start was not acknowledged; command state is unknown"
                )
            attach_cancellation_artifacts(exc, [cleanup.artifact])
            raise
        except Exception as exc:
            if not _is_timeout_error(exc):
                raise
            start_acknowledged = handle is not None
            cleanup = await cleanup_runner_command_with_diagnostic(
                self._sandbox,
                handle=handle,
                adapter="microsandbox",
                timeout_s=self.cancel_timeout_s,
                policy=self.timeout_cleanup,
            )
            self._apply_cleanup_result(cleanup)
            if not start_acknowledged and self.timeout_cleanup == "none":
                self._close_exec(
                    "microsandbox command start was not acknowledged; command state is unknown"
                )
            return ExecResult(
                stdout=stdout.text(),
                stderr=stderr.text(),
                exit_code=exit_code if exit_code is not None else -9,
                timed_out=True,
                stdout_truncated=stdout.truncated,
                stderr_truncated=stderr.truncated,
                stdout_bytes=stdout.total_bytes,
                stderr_bytes=stderr.total_bytes,
                artifacts=[cleanup.artifact],
            )

        return ExecResult(
            stdout=stdout.text(),
            stderr=stderr.text(),
            exit_code=exit_code if exit_code is not None else 0,
            timed_out=False,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            stdout_bytes=stdout.total_bytes,
            stderr_bytes=stderr.total_bytes,
        )

    def _apply_cleanup_result(self, cleanup: Any) -> None:
        # Unlike the base contract, a failed command kill does not latch the
        # exec path: the microsandbox supervisor still owns the command, so the
        # runner stays reusable (covered by the adapter's tests).
        if (
            cleanup.artifact.get("action") == "kill_command"
            and cleanup.artifact.get("status") == "unsupported"
        ):
            self._close_exec(
                "microsandbox command cleanup could not identify the command; "
                "command state is unknown"
            )
        if cleanup.close_runner:
            self._close_exec("runner cleanup closed the exec path")
        if (
            cleanup.artifact.get("action") == "kill_sandbox"
            and cleanup.artifact.get("status") == "completed"
        ):
            self._closed = True


class _LimitedBytes:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.content = bytearray()
        self.total_bytes = 0
        self.truncated = False

    def append(self, data: bytes) -> None:
        if not data:
            return
        self.total_bytes += len(data)
        if self.limit is None:
            self.content.extend(data)
            return
        remaining = self.limit - len(self.content)
        if remaining <= 0:
            self.truncated = True
            return
        self.content.extend(data[:remaining])
        if len(data) > remaining:
            self.truncated = True

    def replace(self, data: bytes) -> None:
        self.content.clear()
        self.total_bytes = 0
        self.truncated = False
        self.append(data)

    def text(self) -> str:
        return bytes(self.content).decode("utf-8", errors="replace")


async def _close_quietly(resource: Any) -> None:
    # Closing a stale/broken SSH or SFTP handle must not mask the original
    # error that prompted the teardown, so swallow close-time failures.
    with contextlib.suppress(Exception):
        await resource.close()


def _microsandbox_module(module: ModuleType | Any | None = None) -> ModuleType | Any:
    if module is not None:
        return module
    try:
        return importlib.import_module("microsandbox")
    except ModuleNotFoundError as exc:
        if exc.name != "microsandbox":
            raise
        raise RuntimeError(
            "MicrosandboxRunner requires the optional microsandbox package. "
            "Install it with `pip install cayu[microsandbox]`."
        ) from exc


def _validate_sandbox_name(name: str) -> str:
    sandbox_name = require_clean_nonblank(name, "name")
    if len(sandbox_name.encode("utf-8")) > MICROSANDBOX_NAME_MAX_BYTES:
        raise ValueError(f"`name` must be at most {MICROSANDBOX_NAME_MAX_BYTES} UTF-8 bytes.")
    return sandbox_name


def _validate_guest_root(path: str) -> str:
    root = require_clean_nonblank(path, "default_cwd")
    if not posixpath.isabs(root):
        raise ValueError("MicrosandboxRunner default_cwd must be an absolute guest path.")
    return posixpath.normpath(root)


def _validate_close_action(action: MicrosandboxCloseAction) -> MicrosandboxCloseAction:
    if action not in {"remove", "stop", "detach", "none"}:
        raise ValueError("Microsandbox close_action must be remove, stop, detach, or none.")
    return action


def _validate_remove_timeout(value: float) -> float:
    if type(value) not in {int, float}:
        raise TypeError("MicrosandboxRunner remove_timeout_s must be numeric.")
    if not isfinite(value):
        raise ValueError("MicrosandboxRunner remove_timeout_s must be finite.")
    if value <= 0:
        raise ValueError("MicrosandboxRunner remove_timeout_s must be greater than zero.")
    return float(value)


def _event_bytes(data: Any) -> bytes:
    if type(data) is bytes:
        return data
    if type(data) is str:
        return data.encode("utf-8")
    raise TypeError("Microsandbox exec event data must be bytes or string.")


def _exec_event_type(event: Any) -> str:
    event_type = getattr(event, "event_type", None)
    if type(event_type) is str:
        return event_type
    class_name = type(event).__name__
    normalized = class_name.lower()
    if normalized.endswith("stdoutevent"):
        return "stdout"
    if normalized.endswith("stderrevent"):
        return "stderr"
    if normalized.endswith("exitedevent"):
        return "exited"
    if normalized.endswith("startedevent"):
        return "started"
    return normalized.removesuffix("event")


def _exec_output_exit_code(output: Any) -> int:
    exit_code = getattr(output, "exit_code", None)
    if type(exit_code) is int:
        return exit_code
    raise TypeError("Microsandbox exec output missing integer exit_code.")


def _apply_collected_output(
    stdout: _LimitedBytes,
    stderr: _LimitedBytes,
    output: Any,
) -> None:
    _replace_with_collected_stream(stdout, output, "stdout")
    _replace_with_collected_stream(stderr, output, "stderr")


def _replace_with_collected_stream(
    buffer: _LimitedBytes,
    output: Any,
    stream_name: Literal["stdout", "stderr"],
) -> None:
    data = _collected_stream_bytes(output, stream_name)
    if data is not None and (data or not buffer.content):
        buffer.replace(data)


def _collected_stream_bytes(output: Any, stream_name: Literal["stdout", "stderr"]) -> bytes | None:
    bytes_value = getattr(output, f"{stream_name}_bytes", None)
    if type(bytes_value) is bytes:
        return bytes_value
    text_value = getattr(output, f"{stream_name}_text", None)
    if type(text_value) is str:
        return text_value.encode("utf-8")
    return None


def _is_timeout_error(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or exc.__class__.__name__ == "ExecTimeoutError"


async def _cleanup_created_sandbox(
    module: ModuleType | Any,
    sandbox: Any,
    name: str,
    *,
    remove_timeout_s: float,
) -> None:
    stop_status: str | None = None
    already_removed = False
    stop_error: BaseException | None = None
    removal_error: BaseException | None = None
    try:
        stop_status, already_removed = await _stop_sandbox(
            module,
            sandbox,
            not_found_is_removed=True,
        )
    except (BaseExceptionGroup, Exception, asyncio.CancelledError) as exc:
        stop_error = exc
        _attach_microsandbox_cleanup_diagnostic(
            exc,
            _microsandbox_cleanup_diagnostic(
                sandbox_name=name,
                action="stop",
                status="failed",
                timeout_s=remove_timeout_s,
                error=exc,
            ),
        )

    if not already_removed:
        try:
            await _remove_stopped_sandbox(
                module,
                name,
                timeout_s=remove_timeout_s,
                initial_status=stop_status,
            )
        except (BaseExceptionGroup, Exception, asyncio.CancelledError) as exc:
            removal_error = exc
            if "diagnostic" not in exc.__dict__:
                _attach_microsandbox_cleanup_diagnostic(
                    exc,
                    _microsandbox_cleanup_diagnostic(
                        sandbox_name=name,
                        action="remove",
                        status="failed",
                        timeout_s=remove_timeout_s,
                        error=exc,
                    ),
                )

    cleanup_errors = [error for error in (stop_error, removal_error) if error is not None]
    if len(cleanup_errors) == 1:
        raise cleanup_errors[0]
    if cleanup_errors:
        raise _MicrosandboxCleanupExceptionGroup(
            "Microsandbox stop and removal cleanup both failed.",
            cleanup_errors,
        )


async def _stop_sandbox(
    module: ModuleType | Any,
    sandbox: Any,
    *,
    not_found_is_removed: bool,
) -> tuple[str | None, bool]:
    try:
        stop_and_wait = getattr(sandbox, "stop_and_wait", None)
        if stop_and_wait is not None:
            result = await stop_and_wait()
            return _sandbox_status_value(result), False
        await sandbox.stop()
        wait_until_stopped = getattr(sandbox, "wait_until_stopped", None)
        if wait_until_stopped is None:
            return None, False
        result = await wait_until_stopped()
        return _sandbox_status_value(result), False
    except Exception as exc:
        if _is_microsandbox_error(module, exc, "SandboxNotRunningError"):
            return "stopped", False
        if not_found_is_removed and _is_microsandbox_error(module, exc, "SandboxNotFoundError"):
            return None, True
        raise


async def _remove_stopped_sandbox(
    module: ModuleType | Any,
    name: str,
    *,
    timeout_s: float,
    initial_status: str | None,
    record_diagnostic: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    backoff_s = _MICROSANDBOX_REMOVE_INITIAL_BACKOFF_SECONDS
    attempts: list[dict[str, Any]] = []
    observed_statuses = [] if initial_status is None else [initial_status]

    while True:
        attempt = len(attempts) + 1
        try:
            await _await_before_microsandbox_deadline(
                module.Sandbox.remove(name),
                deadline=deadline,
            )
        except asyncio.CancelledError as exc:
            attempts.append({"attempt": attempt, "status": "cancelled", "operation": "remove"})
            _record_microsandbox_removal_failure(
                name=name,
                timeout_s=timeout_s,
                attempts=attempts,
                observed_statuses=observed_statuses,
                error=exc,
                record_diagnostic=record_diagnostic,
            )
            raise
        except _MicrosandboxDeadlineExceeded as exc:
            attempts.append({"attempt": attempt, "status": "timed_out", "operation": "remove"})
            raise _microsandbox_removal_timeout(
                name=name,
                timeout_s=timeout_s,
                attempts=attempts,
                observed_statuses=observed_statuses,
                error=exc,
                record_diagnostic=record_diagnostic,
            ) from exc
        except Exception as exc:
            if _is_microsandbox_error(module, exc, "SandboxNotFoundError"):
                attempts.append({"attempt": attempt, "status": "already_removed"})
                return _microsandbox_cleanup_diagnostic(
                    sandbox_name=name,
                    action="remove",
                    status="removed",
                    timeout_s=timeout_s,
                    attempts=attempts,
                    observed_statuses=observed_statuses,
                )
            if not _is_microsandbox_error(module, exc, "SandboxStillRunningError"):
                attempts.append({"attempt": attempt, "status": "failed", "operation": "remove"})
                _record_microsandbox_removal_failure(
                    name=name,
                    timeout_s=timeout_s,
                    attempts=attempts,
                    observed_statuses=observed_statuses,
                    error=exc,
                    record_diagnostic=record_diagnostic,
                )
                raise

            try:
                sandbox_status = await _refreshed_sandbox_status(
                    module,
                    name,
                    deadline=deadline,
                )
            except asyncio.CancelledError as status_exc:
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "cancelled",
                        "operation": "status_refresh",
                    }
                )
                _record_microsandbox_removal_failure(
                    name=name,
                    timeout_s=timeout_s,
                    attempts=attempts,
                    observed_statuses=observed_statuses,
                    error=status_exc,
                    record_diagnostic=record_diagnostic,
                )
                raise
            except _MicrosandboxDeadlineExceeded as status_exc:
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "timed_out",
                        "operation": "status_refresh",
                    }
                )
                raise _microsandbox_removal_timeout(
                    name=name,
                    timeout_s=timeout_s,
                    attempts=attempts,
                    observed_statuses=observed_statuses,
                    error=status_exc,
                    record_diagnostic=record_diagnostic,
                ) from status_exc
            except Exception as status_exc:
                if _is_microsandbox_error(module, status_exc, "SandboxNotFoundError"):
                    attempts.append({"attempt": attempt, "status": "already_removed"})
                    return _microsandbox_cleanup_diagnostic(
                        sandbox_name=name,
                        action="remove",
                        status="removed",
                        timeout_s=timeout_s,
                        attempts=attempts,
                        observed_statuses=observed_statuses,
                    )
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "failed",
                        "operation": "status_refresh",
                    }
                )
                _record_microsandbox_removal_failure(
                    name=name,
                    timeout_s=timeout_s,
                    attempts=attempts,
                    observed_statuses=observed_statuses,
                    error=status_exc,
                    record_diagnostic=record_diagnostic,
                )
                raise
            if sandbox_status is not None:
                observed_statuses.append(sandbox_status)
            attempts.append(
                {
                    "attempt": attempt,
                    "status": "deferred",
                    "sandbox_status": sandbox_status,
                }
            )

            remaining_s = deadline - loop.time()
            if remaining_s <= 0:
                raise _microsandbox_removal_timeout(
                    name=name,
                    timeout_s=timeout_s,
                    attempts=attempts,
                    observed_statuses=observed_statuses,
                    error=exc,
                    record_diagnostic=record_diagnostic,
                ) from exc
            try:
                await _sleep_before_microsandbox_retry(min(backoff_s, remaining_s))
            except asyncio.CancelledError as sleep_exc:
                attempts.append(
                    {
                        "attempt": attempt,
                        "status": "cancelled",
                        "operation": "backoff",
                    }
                )
                _record_microsandbox_removal_failure(
                    name=name,
                    timeout_s=timeout_s,
                    attempts=attempts,
                    observed_statuses=observed_statuses,
                    error=sleep_exc,
                    record_diagnostic=record_diagnostic,
                )
                raise
            backoff_s = min(
                backoff_s * 2,
                _MICROSANDBOX_REMOVE_MAX_BACKOFF_SECONDS,
            )
            continue

        attempts.append({"attempt": attempt, "status": "removed"})
        return _microsandbox_cleanup_diagnostic(
            sandbox_name=name,
            action="remove",
            status="removed",
            timeout_s=timeout_s,
            attempts=attempts,
            observed_statuses=observed_statuses,
        )


async def _refreshed_sandbox_status(
    module: ModuleType | Any,
    name: str,
    *,
    deadline: float,
) -> str | None:
    handle = await _await_before_microsandbox_deadline(
        module.Sandbox.get(name),
        deadline=deadline,
    )
    refresh = getattr(handle, "refresh", None)
    if refresh is not None:
        refreshed = await _await_before_microsandbox_deadline(
            refresh(),
            deadline=deadline,
        )
        if refreshed is not None:
            handle = refreshed
    return _sandbox_status_value(handle)


async def _sleep_before_microsandbox_retry(delay_s: float) -> None:
    await asyncio.sleep(delay_s)


async def _await_before_microsandbox_deadline(awaitable: Any, *, deadline: float) -> Any:
    timeout = asyncio.timeout_at(deadline)
    try:
        async with timeout:
            return await awaitable
    except TimeoutError as exc:
        if timeout.expired():
            raise _MicrosandboxDeadlineExceeded from exc
        raise


def _record_microsandbox_removal_failure(
    *,
    name: str,
    timeout_s: float,
    attempts: list[dict[str, Any]],
    observed_statuses: list[str],
    error: BaseException,
    record_diagnostic: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    diagnostic = _microsandbox_cleanup_diagnostic(
        sandbox_name=name,
        action="remove",
        status="failed",
        timeout_s=timeout_s,
        attempts=attempts,
        observed_statuses=observed_statuses,
        error=error,
    )
    _attach_microsandbox_cleanup_diagnostic(error, diagnostic)
    if record_diagnostic is not None:
        record_diagnostic(diagnostic)
    return diagnostic


def _microsandbox_removal_timeout(
    *,
    name: str,
    timeout_s: float,
    attempts: list[dict[str, Any]],
    observed_statuses: list[str],
    error: Exception,
    record_diagnostic: Callable[[dict[str, Any]], None] | None,
) -> MicrosandboxCleanupError:
    diagnostic = _microsandbox_cleanup_diagnostic(
        sandbox_name=name,
        action="remove",
        status="timed_out",
        timeout_s=timeout_s,
        attempts=attempts,
        observed_statuses=observed_statuses,
        error=error,
    )
    if record_diagnostic is not None:
        record_diagnostic(diagnostic)
    return MicrosandboxCleanupError(
        f"Microsandbox {name!r} removal did not settle within {timeout_s:g} seconds.",
        diagnostic=diagnostic,
    )


def _sandbox_status_value(value: Any) -> str | None:
    status = getattr(value, "status", None)
    if callable(status):
        status = status()
    if type(status) is str and status:
        return status
    return None


def _is_microsandbox_error(module: ModuleType | Any, exc: Exception, name: str) -> bool:
    error_type = getattr(module, name, None)
    return isinstance(error_type, type) and isinstance(exc, error_type)


def _microsandbox_cleanup_diagnostic(
    *,
    sandbox_name: str,
    action: str,
    status: str,
    timeout_s: float,
    attempts: list[dict[str, Any]] | None = None,
    observed_statuses: list[str] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    diagnostic: dict[str, Any] = {
        "type": _MICROSANDBOX_CLEANUP_DIAGNOSTIC_TYPE,
        "adapter": "microsandbox",
        "sandbox_name": sandbox_name,
        "action": action,
        "status": status,
        "timeout_s": timeout_s,
        "attempts": [] if attempts is None else attempts,
        "observed_statuses": [] if observed_statuses is None else observed_statuses,
    }
    if error is not None:
        diagnostic["error_type"] = type(error).__name__
        diagnostic["error"] = str(error)
    return copy_json_value(diagnostic, "diagnostic")


def _attach_microsandbox_cleanup_diagnostic(
    error: BaseException,
    diagnostic: dict[str, Any],
) -> None:
    error.__dict__["diagnostic"] = copy_json_value(diagnostic, "diagnostic")


async def _cleanup_created_sandbox_after_failure(
    module: ModuleType | Any,
    sandbox: Any,
    name: str,
    original_error: BaseException,
    message: str,
    *,
    remove_timeout_s: float,
) -> None:
    try:
        await _cleanup_created_sandbox(
            module,
            sandbox,
            name,
            remove_timeout_s=remove_timeout_s,
        )
    except _MicrosandboxCleanupExceptionGroup as cleanup_group:
        raise BaseExceptionGroup(
            message,
            [original_error, *cleanup_group.exceptions],
        ) from cleanup_group
    except (BaseExceptionGroup, Exception, asyncio.CancelledError) as cleanup_error:
        raise BaseExceptionGroup(message, [original_error, cleanup_error]) from cleanup_error
