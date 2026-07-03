from __future__ import annotations

import asyncio
import contextlib
import importlib
import posixpath
from types import ModuleType
from typing import Any, Literal

from cayu._validation import require_clean_nonblank
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
MICROSANDBOX_NAME_MAX_BYTES = 128

MicrosandboxCloseAction = Literal["remove", "stop", "detach", "none"]


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
        self._sandbox = sandbox
        self._sandbox_module = sandbox_module
        self._sftp_client: Any = None
        self._sftp: Any = None
        self._sftp_lock = asyncio.Lock()

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
        ensure_default_cwd: bool = True,
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
            )
            raise
        except Exception as exc:
            await _cleanup_created_sandbox_after_failure(
                module,
                sandbox,
                sandbox_name,
                exc,
                "Microsandbox setup failed and cleanup failed.",
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
            sandbox_module=module,
        )

    async def close(self) -> None:
        """Apply the configured lifecycle action once."""

        if self._closed:
            return
        await self._close_sftp_session()
        if self.close_action == "none":
            self._closed = True
            return
        if self.close_action == "detach":
            detach = getattr(self._sandbox, "detach", None)
            if detach is not None:
                await detach()
            self._closed = True
            return
        if self.close_action in {"stop", "remove"}:
            await _stop_sandbox(self._sandbox)
            if self.close_action == "remove":
                module = _microsandbox_module(self._sandbox_module)
                await module.Sandbox.remove(self.name)
            self._closed = True
            return
        raise AssertionError(f"Unsupported Microsandbox close action: {self.close_action}")

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
            cleanup = await cleanup_runner_command_with_diagnostic(
                self._sandbox,
                handle=handle,
                adapter="microsandbox",
                timeout_s=self.cancel_timeout_s,
                policy=self.cancellation_cleanup,
            )
            self._apply_cleanup_result(cleanup)
            attach_cancellation_artifacts(exc, [cleanup.artifact])
            raise
        except Exception as exc:
            if not _is_timeout_error(exc):
                raise
            cleanup = await cleanup_runner_command_with_diagnostic(
                self._sandbox,
                handle=handle,
                adapter="microsandbox",
                timeout_s=self.cancel_timeout_s,
                policy=self.timeout_cleanup,
            )
            self._apply_cleanup_result(cleanup)
            return ExecResult(
                stdout=stdout.text(),
                stderr=stderr.text(),
                exit_code=exit_code if exit_code is not None else -9,
                timed_out=True,
                stdout_truncated=stdout.truncated,
                stderr_truncated=stderr.truncated,
                artifacts=[cleanup.artifact],
            )

        return ExecResult(
            stdout=stdout.text(),
            stderr=stderr.text(),
            exit_code=exit_code if exit_code is not None else 0,
            timed_out=False,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
        )

    def _apply_cleanup_result(self, cleanup: Any) -> None:
        # Unlike the base contract, a failed command kill does not latch the
        # exec path: the microsandbox supervisor still owns the command, so the
        # runner stays reusable (covered by the adapter's tests).
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
        self.truncated = False

    def append(self, data: bytes) -> None:
        if not data:
            return
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


async def _cleanup_created_sandbox(module: ModuleType | Any, sandbox: Any, name: str) -> None:
    try:
        await _stop_sandbox(sandbox)
    finally:
        await module.Sandbox.remove(name)


async def _stop_sandbox(sandbox: Any) -> None:
    stop_and_wait = getattr(sandbox, "stop_and_wait", None)
    if stop_and_wait is not None:
        await stop_and_wait()
        return
    await sandbox.stop()


async def _cleanup_created_sandbox_after_failure(
    module: ModuleType | Any,
    sandbox: Any,
    name: str,
    original_error: BaseException,
    message: str,
) -> None:
    try:
        await _cleanup_created_sandbox(module, sandbox, name)
    except Exception as cleanup_error:
        raise BaseExceptionGroup(message, [original_error, cleanup_error]) from cleanup_error
