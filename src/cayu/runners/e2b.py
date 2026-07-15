from __future__ import annotations

import asyncio
import contextlib
import importlib
import posixpath
import shlex
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal, cast

from cayu._validation import require_clean_nonblank
from cayu.runners._cleanup import (
    DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
    DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
    DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
    RUNNER_CLEANUP_ARTIFACT_TYPE,
    RunnerCleanupPolicy,
    RunnerCleanupResult,
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
    RunnerWorkspaceCapability,
    RunnerWorkspaceCapabilityT,
    attach_cancellation_artifacts,
)

DEFAULT_E2B_CWD = "/home/user/workspace"
E2B_SANDBOX_ID_MAX_BYTES = 256
E2B_LATE_START_CLEANUP_TIMEOUT_MULTIPLIER = 4.0

E2BCloseAction = Literal["kill", "detach", "none"]


class E2BWorkspaceCapability(RunnerWorkspaceCapability):
    """Native E2B filesystem access without runner lifecycle authority."""

    @property
    @abstractmethod
    def sandbox_id(self) -> str:
        """Provider sandbox id used for truthful workspace identity."""

    @abstractmethod
    async def get_info(
        self,
        path: str,
        *,
        user: str | None,
        request_timeout_s: float | None,
    ) -> E2BWorkspaceEntry:
        """Return a normalized Cayu-owned entry for one guest path."""

    @abstractmethod
    async def list_entries(
        self,
        path: str,
        *,
        depth: int,
        user: str | None,
        request_timeout_s: float | None,
    ) -> Sequence[E2BWorkspaceEntry]:
        """List normalized Cayu-owned entries below one guest path."""


class _E2BWorkspaceCapability(E2BWorkspaceCapability):
    def __init__(self, runner: E2BRunner) -> None:
        self._runner = runner

    @property
    def sandbox_id(self) -> str:
        return self._runner.sandbox_id

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("e2b", self._runner.sandbox_id)

    async def get_info(
        self,
        path: str,
        *,
        user: str | None,
        request_timeout_s: float | None,
    ) -> E2BWorkspaceEntry:
        entry = await self._runner.filesystem().get_info(
            path,
            user=user,
            request_timeout=request_timeout_s,
        )
        return E2BWorkspaceEntry.from_provider_entry(entry)

    async def list_entries(
        self,
        path: str,
        *,
        depth: int,
        user: str | None,
        request_timeout_s: float | None,
    ) -> Sequence[E2BWorkspaceEntry]:
        entries = await self._runner.filesystem().list(
            path,
            depth=depth,
            user=user,
            request_timeout=request_timeout_s,
        )
        return tuple(E2BWorkspaceEntry.from_provider_entry(entry) for entry in entries)


@dataclass(frozen=True)
class E2BWorkspaceEntry:
    """Cayu-owned normalized view of one E2B filesystem entry."""

    path: str | None
    type: str | None
    symlink_target: str | None = None

    @classmethod
    def from_provider_entry(cls, entry: Any) -> E2BWorkspaceEntry:
        path = getattr(entry, "path", None)
        entry_type = getattr(entry, "type", None)
        if type(entry_type) is not str:
            entry_type = getattr(entry_type, "value", None)
        symlink_target = getattr(entry, "symlink_target", None)
        return cls(
            path=path if type(path) is str else None,
            type=entry_type if type(entry_type) is str else None,
            symlink_target=symlink_target if type(symlink_target) is str else None,
        )


class E2BRunner(Runner):
    """Executes commands in an E2B cloud sandbox.

    E2B's command API executes command strings through Bash. Cayu preserves its
    process-form contract by shell-quoting argv with `shlex.join(...)` before
    handing the command to E2B.
    """

    isolation = "e2b"

    def __init__(
        self,
        sandbox: Any,
        *,
        sandbox_id: str | None = None,
        default_cwd: str = DEFAULT_E2B_CWD,
        close_action: E2BCloseAction = "none",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        env_overlay: Mapping[str, str] | None = None,
        exec_user: str | None = None,
        e2b_module: ModuleType | Any | None = None,
    ) -> None:
        if sandbox is None:
            raise TypeError("E2BRunner sandbox cannot be None.")
        resolved_id = sandbox_id or getattr(sandbox, "sandbox_id", None)
        if type(resolved_id) is not str:
            raise ValueError("E2BRunner sandbox_id is required.")
        self.sandbox_id = _validate_sandbox_id(resolved_id)
        self.default_cwd = _validate_guest_root(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.cancel_timeout_s = validate_cancel_timeout(cancel_timeout_s)
        self.cancellation_cleanup = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        self.timeout_cleanup = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        self.env_overlay = dict(env_overlay) if env_overlay else {}
        self.exec_user = _validate_exec_user(exec_user)
        self._sandbox = sandbox
        self._e2b_module = e2b_module
        self._late_start_cleanup_timeout_s = self.cancel_timeout_s * (
            E2B_LATE_START_CLEANUP_TIMEOUT_MULTIPLIER
        )
        self._late_start_cleanup_tasks: set[asyncio.Task[None]] = set()

    @classmethod
    async def create(
        cls,
        *,
        template: str | None = None,
        sandbox_timeout_s: int | None = None,
        default_cwd: str = DEFAULT_E2B_CWD,
        close_action: E2BCloseAction = "kill",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        ensure_default_cwd: bool = True,
        metadata: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
        env_overlay: Mapping[str, str] | None = None,
        exec_user: str | None = None,
        secure: bool = True,
        allow_internet_access: bool = True,
        network: Any | None = None,
        lifecycle: Any | None = None,
        volume_mounts: Any | None = None,
        e2b_module: ModuleType | Any | None = None,
        **api_options: Any,
    ) -> E2BRunner:
        """Create an E2B sandbox and return a runner bound to it.

        Provider-specific options intentionally pass through to E2B so Cayu does
        not invent a weaker abstraction over templates, networking, lifecycle,
        or volumes.
        """

        module = _e2b_module(e2b_module)
        guest_root = _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        if type(ensure_default_cwd) is not bool:
            raise TypeError("E2BRunner ensure_default_cwd must be a bool.")
        if type(secure) is not bool:
            raise TypeError("E2BRunner secure must be a bool.")
        if type(allow_internet_access) is not bool:
            raise TypeError("E2BRunner allow_internet_access must be a bool.")
        timeout = _validate_sandbox_timeout(sandbox_timeout_s)

        create_options: dict[str, Any] = {
            "secure": secure,
            "allow_internet_access": allow_internet_access,
        }
        _set_option(create_options, "template", template)
        _set_option(create_options, "timeout", timeout)
        _set_option(create_options, "metadata", _copy_string_dict(metadata, "metadata"))
        _set_option(create_options, "envs", _copy_string_dict(envs, "envs"))
        _set_option(create_options, "network", network)
        _set_option(create_options, "lifecycle", lifecycle)
        _set_option(create_options, "volume_mounts", volume_mounts)
        create_options.update(dict(api_options))

        sandbox = await module.AsyncSandbox.create(**create_options)
        try:
            if ensure_default_cwd:
                await sandbox.commands.run(
                    f"mkdir -p {shlex.quote(guest_root)}",
                    cwd="/",
                    timeout=60,
                )
        except asyncio.CancelledError as exc:
            await _cleanup_created_sandbox_after_failure(
                sandbox,
                exc,
                "E2B setup was cancelled and cleanup failed.",
            )
            raise
        except Exception as exc:
            await _cleanup_created_sandbox_after_failure(
                sandbox,
                exc,
                "E2B setup failed and cleanup failed.",
            )
            raise
        return cls(
            sandbox,
            default_cwd=guest_root,
            close_action=close_action,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            env_overlay=env_overlay,
            exec_user=exec_user,
            e2b_module=module,
        )

    @classmethod
    async def from_existing(
        cls,
        sandbox_id: str,
        *,
        sandbox_timeout_s: int | None = None,
        default_cwd: str = DEFAULT_E2B_CWD,
        close_action: E2BCloseAction = "none",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        ensure_default_cwd: bool = True,
        env_overlay: Mapping[str, str] | None = None,
        exec_user: str | None = None,
        e2b_module: ModuleType | Any | None = None,
        **api_options: Any,
    ) -> E2BRunner:
        """Attach to an existing E2B sandbox by id."""

        module = _e2b_module(e2b_module)
        resolved_id = _validate_sandbox_id(sandbox_id)
        guest_root = _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        if type(ensure_default_cwd) is not bool:
            raise TypeError("E2BRunner ensure_default_cwd must be a bool.")
        timeout = _validate_sandbox_timeout(sandbox_timeout_s)
        connect_options = dict(api_options)
        _set_option(connect_options, "timeout", timeout)
        sandbox = await module.AsyncSandbox.connect(resolved_id, **connect_options)
        if ensure_default_cwd:
            await sandbox.commands.run(
                f"mkdir -p {shlex.quote(guest_root)}",
                cwd="/",
                timeout=60,
            )
        return cls(
            sandbox,
            sandbox_id=resolved_id,
            default_cwd=guest_root,
            close_action=close_action,
            cancel_timeout_s=cancel_timeout_s,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            env_overlay=env_overlay,
            exec_user=exec_user,
            e2b_module=module,
        )

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("e2b", self.sandbox_id)

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        if capability_type is E2BWorkspaceCapability:
            capability = _E2BWorkspaceCapability(self)
            return cast("RunnerWorkspaceCapabilityT", capability)
        return super().workspace_capability(capability_type)

    async def close(self) -> None:
        """Apply the configured lifecycle action once."""

        if self._closed:
            return
        await self._wait_for_late_start_cleanup_tasks()
        if self.close_action in {"none", "detach"}:
            self._closed = True
            return
        if self.close_action == "kill":
            await self._sandbox.kill()
            self._closed = True
            return
        raise AssertionError(f"Unsupported E2B close action: {self.close_action}")

    def filesystem(self) -> Any:
        """Return the native E2B filesystem API for workspace adapters."""

        if self._closed:
            raise RuntimeError("E2BRunner is closed.")
        return self._sandbox.files

    async def _exec_admin(self, script: str, *, timeout_s: float = 60) -> Any:
        """Run adapter-owned bootstrap code as root before guest handoff."""

        if self._closed:
            raise RuntimeError("E2BRunner is closed.")
        if not script.strip():
            raise ValueError("E2B admin script must be nonblank.")
        return await self._sandbox.commands.run(
            script,
            user="root",
            timeout=timeout_s,
        )

    async def _exec_guest_check(self, script: str, *, timeout_s: float = 30) -> Any:
        """Run an adapter-owned hardening check as the eventual guest user."""

        if self._closed:
            raise RuntimeError("E2BRunner is closed.")
        if not script.strip():
            raise ValueError("E2B guest-check script must be nonblank.")
        if self.exec_user is None:
            raise RuntimeError("E2B guest checks require a pinned exec_user.")
        return await self._sandbox.commands.run(
            script,
            user=self.exec_user,
            timeout=timeout_s,
        )

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
            raise TypeError("E2BRunner command must be an ExecCommand.")
        self._ensure_exec_open()

        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        if self.env_overlay:
            environment.update(self.env_overlay)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        output_limit = validate_output_limit(output_limit_bytes)
        script = _command_to_e2b_script(command)
        stdout = _LimitedText(output_limit)
        stderr = _LimitedText(output_limit)
        handle = None
        start_task: asyncio.Task[Any] | None = None

        async def on_stdout(chunk: str) -> None:
            stdout.append(chunk)

        async def on_stderr(chunk: str) -> None:
            stderr.append(chunk)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout if timeout is not None else None
        try:
            run_options: dict[str, Any] = {
                "background": True,
                "envs": environment,
                "cwd": working_dir,
                "on_stdout": on_stdout,
                "on_stderr": on_stderr,
                "stdin": standard_input is not None,
                "timeout": float(timeout) if timeout is not None else 0,
            }
            if self.exec_user is not None:
                run_options["user"] = self.exec_user
            start_task = asyncio.create_task(self._sandbox.commands.run(script, **run_options))
            handle = await self._await_started_handle(start_task, deadline=deadline)
            if standard_input is not None:
                await handle.send_stdin(standard_input)
                await handle.close_stdin()
            if deadline is None:
                result = await handle.wait()
            else:
                remaining = max(deadline - loop.time(), 0.0)
                result = await asyncio.wait_for(handle.wait(), timeout=remaining)
        except asyncio.CancelledError as exc:
            cleanup = await self._cleanup_interrupted_command(
                start_task,
                current_handle=handle,
                cleanup_policy=self.cancellation_cleanup,
                wait_for_handle_before_cancelling=True,
            )
            attach_cancellation_artifacts(exc, [cleanup.artifact])
            raise
        except Exception as exc:
            if _is_timeout_error(exc):
                cleanup = await self._cleanup_interrupted_command(
                    start_task,
                    current_handle=handle,
                    cleanup_policy=self.timeout_cleanup,
                    wait_for_handle_before_cancelling=False,
                )
                return ExecResult(
                    stdout=stdout.text(),
                    stderr=stderr.text(),
                    exit_code=-9,
                    timed_out=True,
                    stdout_truncated=stdout.truncated,
                    stderr_truncated=stderr.truncated,
                    stdout_bytes=stdout.total_bytes,
                    stderr_bytes=stderr.total_bytes,
                    artifacts=[cleanup.artifact],
                )
            if _is_command_exit(exc):
                return _exec_result_from_e2b_result(exc, stdout, stderr)
            raise

        return _exec_result_from_e2b_result(result, stdout, stderr)

    async def _await_started_handle(
        self,
        start_task: asyncio.Task[Any],
        *,
        deadline: float | None,
    ) -> Any:
        if deadline is None:
            return await asyncio.shield(start_task)
        remaining = max(deadline - asyncio.get_running_loop().time(), 0.0)
        return await asyncio.wait_for(asyncio.shield(start_task), timeout=remaining)

    async def _cleanup_interrupted_command(
        self,
        start_task: asyncio.Task[Any] | None,
        *,
        current_handle: Any | None,
        cleanup_policy: RunnerCleanupPolicy,
        wait_for_handle_before_cancelling: bool,
    ) -> RunnerCleanupResult:
        handle, cleanup_deferred = await self._resolve_started_handle_after_interruption(
            start_task,
            current_handle=current_handle,
            cleanup_policy=cleanup_policy,
            wait_for_handle_before_cancelling=wait_for_handle_before_cancelling,
        )
        if cleanup_deferred:
            return RunnerCleanupResult(
                artifact=_late_start_cleanup_deferred_artifact(
                    timeout_s=self.cancel_timeout_s,
                    late_start_cleanup_timeout_s=self._late_start_cleanup_timeout_s,
                ),
                close_runner=False,
            )
        cleanup = await cleanup_runner_command_with_diagnostic(
            self._sandbox,
            handle=handle,
            adapter="e2b",
            timeout_s=self.cancel_timeout_s,
            policy=cleanup_policy,
        )
        self._apply_cleanup_result(cleanup)
        return cleanup

    def _apply_cleanup_result(self, cleanup: Any) -> None:
        if cleanup.close_runner:
            self._close_exec("runner cleanup closed the exec path")
        if (
            cleanup.artifact.get("action") == "kill_sandbox"
            and cleanup.artifact.get("status") == "completed"
        ):
            self._closed = True

    async def _resolve_started_handle_after_interruption(
        self,
        start_task: asyncio.Task[Any] | None,
        *,
        current_handle: Any | None,
        cleanup_policy: RunnerCleanupPolicy,
        wait_for_handle_before_cancelling: bool,
    ) -> tuple[Any | None, bool]:
        if current_handle is not None:
            return current_handle, False
        if start_task is None:
            return None, False
        if start_task.done():
            try:
                handle = await start_task
                return handle, False
            except asyncio.CancelledError:
                return None, False
            except Exception:
                return None, False
        if not wait_for_handle_before_cancelling:
            return await self._cancel_or_track_start_task(
                start_task,
                cleanup_policy=cleanup_policy,
            )
        try:
            handle = await asyncio.wait_for(
                asyncio.shield(start_task),
                timeout=self.cancel_timeout_s,
            )
            return handle, False
        except TimeoutError:
            return await self._cancel_or_track_start_task(
                start_task,
                cleanup_policy=cleanup_policy,
            )
        except asyncio.CancelledError:
            return await self._cancel_or_track_start_task(
                start_task,
                cleanup_policy=cleanup_policy,
            )
        except Exception:
            return None, False

    async def _cancel_or_track_start_task(
        self,
        start_task: asyncio.Task[Any],
        *,
        cleanup_policy: RunnerCleanupPolicy,
    ) -> tuple[Any | None, bool]:
        start_task.cancel()
        try:
            handle = await asyncio.wait_for(
                asyncio.shield(start_task),
                timeout=self.cancel_timeout_s,
            )
            return handle, False
        except TimeoutError:
            if cleanup_policy == "sandbox":
                return None, False
            if cleanup_policy == "none":
                self._close_exec("E2B command start did not stop; command state is unknown")
                return None, False
            self._track_late_start_cleanup(start_task, cleanup_policy=cleanup_policy)
            return None, True
        except asyncio.CancelledError:
            return None, False
        except Exception:
            return None, False

    def _track_late_start_cleanup(
        self,
        start_task: asyncio.Task[Any],
        *,
        cleanup_policy: RunnerCleanupPolicy,
    ) -> None:
        self._close_exec("E2B command start cleanup is pending")
        cleanup_task = asyncio.create_task(
            self._cleanup_late_started_command(start_task, cleanup_policy=cleanup_policy)
        )
        self._late_start_cleanup_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._late_start_cleanup_tasks.discard)

    async def _cleanup_late_started_command(
        self,
        start_task: asyncio.Task[Any],
        *,
        cleanup_policy: RunnerCleanupPolicy,
    ) -> None:
        try:
            handle = await asyncio.wait_for(
                asyncio.shield(start_task),
                timeout=self._late_start_cleanup_timeout_s,
            )
        except asyncio.CancelledError:
            self._open_exec()
            return
        except TimeoutError:
            self._close_exec(
                "E2B command start did not resolve after interruption or timeout; "
                "command state is unknown"
            )
            return
        except Exception:
            self._open_exec()
            return
        cleanup = await cleanup_runner_command_with_diagnostic(
            self._sandbox,
            handle=handle,
            adapter="e2b",
            timeout_s=self.cancel_timeout_s,
            policy=cleanup_policy,
        )
        self._apply_cleanup_result(cleanup)
        if (
            cleanup.artifact.get("action") == "kill_command"
            and cleanup.artifact.get("status") == "completed"
        ):
            self._open_exec()
            return
        self._close_exec("late-started E2B command cleanup did not complete")

    async def _wait_for_late_start_cleanup_tasks(self) -> None:
        if not self._late_start_cleanup_tasks:
            return
        tasks = tuple(self._late_start_cleanup_tasks)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(*tasks, return_exceptions=True)),
                timeout=self.cancel_timeout_s,
            )


class _LimitedText:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.content = bytearray()
        self.total_bytes = 0
        self.truncated = False

    def append(self, data: str | bytes) -> None:
        if type(data) is str:
            chunk = data.encode("utf-8")
        elif type(data) is bytes:
            chunk = data
        else:
            raise TypeError("E2B command output chunks must be strings or bytes.")
        if not chunk:
            return
        self.total_bytes += len(chunk)
        if self.limit is None:
            self.content.extend(chunk)
            return
        remaining = self.limit - len(self.content)
        if remaining <= 0:
            self.truncated = True
            return
        self.content.extend(chunk[:remaining])
        if len(chunk) > remaining:
            self.truncated = True

    def seed_if_empty(self, data: Any) -> None:
        if self.content:
            return
        if type(data) is str or type(data) is bytes:
            self.append(data)

    def text(self) -> str:
        return bytes(self.content).decode("utf-8", errors="replace")


def _e2b_module(module: ModuleType | Any | None = None) -> ModuleType | Any:
    if module is not None:
        return module
    try:
        return importlib.import_module("e2b")
    except ModuleNotFoundError as exc:
        if exc.name != "e2b":
            raise
        raise RuntimeError(
            "E2BRunner requires the optional e2b package. Install it with `pip install cayu[e2b]`."
        ) from exc


def _validate_sandbox_id(sandbox_id: str) -> str:
    value = require_clean_nonblank(sandbox_id, "sandbox_id")
    if len(value.encode("utf-8")) > E2B_SANDBOX_ID_MAX_BYTES:
        raise ValueError(f"`sandbox_id` must be at most {E2B_SANDBOX_ID_MAX_BYTES} UTF-8 bytes.")
    return value


def _validate_exec_user(user: str | None) -> str | None:
    if user is None:
        return None
    return require_clean_nonblank(user, "exec_user")


def _validate_guest_root(path: str) -> str:
    root = require_clean_nonblank(path, "default_cwd")
    if not posixpath.isabs(root):
        raise ValueError("E2BRunner default_cwd must be an absolute guest path.")
    return posixpath.normpath(root)


def _validate_close_action(action: E2BCloseAction) -> E2BCloseAction:
    if action not in {"kill", "detach", "none"}:
        raise ValueError("E2B close_action must be kill, detach, or none.")
    return action


def _validate_sandbox_timeout(timeout_s: int | None) -> int | None:
    if timeout_s is None:
        return None
    if type(timeout_s) is not int:
        raise TypeError("E2B sandbox_timeout_s must be an integer.")
    if timeout_s <= 0:
        raise ValueError("E2B sandbox_timeout_s must be greater than zero.")
    return timeout_s


def _copy_string_dict(value: dict[str, str] | None, field_name: str) -> dict[str, str] | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise TypeError(f"E2BRunner {field_name} must be a dictionary.")
    copied: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or not key.strip():
            raise ValueError(f"E2BRunner {field_name} keys must be non-empty strings.")
        if type(item) is not str:
            raise ValueError(f"E2BRunner {field_name} values must be strings.")
        copied[key] = item
    return copied


def _set_option(options: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        options[key] = value


def _command_to_e2b_script(command: ExecCommand) -> str:
    if command.kind == "process":
        if command.argv is None:
            raise ValueError("Process commands require argv.")
        return shlex.join(command.argv)
    if command.shell is None:
        raise ValueError("Shell commands require a script.")
    return command.shell


def _exec_result_from_e2b_result(
    result: Any,
    stdout: _LimitedText,
    stderr: _LimitedText,
) -> ExecResult:
    stdout.seed_if_empty(getattr(result, "stdout", None))
    stderr.seed_if_empty(getattr(result, "stderr", None))
    exit_code = getattr(result, "exit_code", None)
    if type(exit_code) is not int:
        raise TypeError("E2B command result missing integer exit_code.")
    return ExecResult(
        stdout=stdout.text(),
        stderr=stderr.text(),
        exit_code=exit_code,
        stdout_truncated=stdout.truncated,
        stderr_truncated=stderr.truncated,
        stdout_bytes=stdout.total_bytes,
        stderr_bytes=stderr.total_bytes,
    )


def _is_command_exit(exc: Exception) -> bool:
    return type(getattr(exc, "exit_code", None)) is int


def _is_timeout_error(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or exc.__class__.__name__ == "TimeoutException"


def _late_start_cleanup_deferred_artifact(
    *,
    timeout_s: float,
    late_start_cleanup_timeout_s: float,
) -> dict[str, Any]:
    return {
        "type": RUNNER_CLEANUP_ARTIFACT_TYPE,
        "adapter": "e2b",
        "action": "kill_command",
        "status": "deferred",
        "timeout_s": timeout_s,
        "late_start_cleanup_timeout_s": late_start_cleanup_timeout_s,
        "reason": "command handle is not available yet; cleanup will continue in background",
    }


async def _cleanup_created_sandbox(sandbox: Any) -> None:
    await sandbox.kill()


async def _cleanup_created_sandbox_after_failure(
    sandbox: Any,
    original_error: BaseException,
    message: str,
) -> None:
    try:
        await _cleanup_created_sandbox(sandbox)
    except Exception as cleanup_error:
        raise BaseExceptionGroup(message, [original_error, cleanup_error]) from cleanup_error
