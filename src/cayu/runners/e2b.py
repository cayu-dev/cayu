from __future__ import annotations

import asyncio
import importlib
import posixpath
import shlex
from types import ModuleType
from typing import Any, Literal

from cayu._validation import require_clean_nonblank, require_nonblank
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
)

DEFAULT_E2B_CWD = "/home/user/workspace"
E2B_SANDBOX_ID_MAX_BYTES = 256

E2BCloseAction = Literal["kill", "detach", "none"]


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
        self._sandbox = sandbox
        self._e2b_module = e2b_module
        self._closed = False

    @classmethod
    async def create(
        cls,
        *,
        template: str | None = None,
        sandbox_timeout_s: int | None = None,
        default_cwd: str = DEFAULT_E2B_CWD,
        close_action: E2BCloseAction = "kill",
        ensure_default_cwd: bool = True,
        metadata: dict[str, str] | None = None,
        envs: dict[str, str] | None = None,
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
        ensure_default_cwd: bool = True,
        e2b_module: ModuleType | Any | None = None,
        **api_options: Any,
    ) -> E2BRunner:
        """Attach to an existing E2B sandbox by id."""

        module = _e2b_module(e2b_module)
        resolved_id = _validate_sandbox_id(sandbox_id)
        guest_root = _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
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
            e2b_module=module,
        )

    async def __aenter__(self) -> E2BRunner:
        return self

    async def __aexit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        await self.close()
        return False

    async def close(self) -> None:
        """Apply the configured lifecycle action once."""

        if self._closed:
            return
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
        if self._closed:
            raise RuntimeError("E2BRunner is closed.")

        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        output_limit = validate_output_limit(output_limit_bytes)
        script = _command_to_e2b_script(command)
        stdout = _LimitedText(output_limit)
        stderr = _LimitedText(output_limit)
        handle = None

        async def on_stdout(chunk: str) -> None:
            stdout.append(chunk)

        async def on_stderr(chunk: str) -> None:
            stderr.append(chunk)

        try:
            handle = await self._sandbox.commands.run(
                script,
                background=True,
                envs=environment,
                cwd=working_dir,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
                stdin=standard_input is not None,
                timeout=float(timeout) if timeout is not None else 0,
            )
            if standard_input is not None:
                await handle.send_stdin(standard_input)
                await handle.close_stdin()
            if timeout is None:
                result = await handle.wait()
            else:
                result = await asyncio.wait_for(handle.wait(), timeout=timeout)
        except asyncio.CancelledError:
            if handle is not None:
                await _kill_handle_best_effort(handle)
            raise
        except Exception as exc:
            if _is_timeout_error(exc):
                if handle is not None:
                    await _kill_handle_best_effort(handle)
                return ExecResult(
                    stdout=stdout.text(),
                    stderr=stderr.text(),
                    exit_code=-9,
                    timed_out=True,
                    stdout_truncated=stdout.truncated,
                    stderr_truncated=stderr.truncated,
                )
            if _is_command_exit(exc):
                return _exec_result_from_e2b_result(exc, stdout, stderr)
            raise

        return _exec_result_from_e2b_result(result, stdout, stderr)

    def resolve_cwd(self, cwd: str | None = None) -> str:
        if cwd is None:
            return self.default_cwd
        relative_cwd = require_nonblank(cwd, "cwd")
        if posixpath.isabs(relative_cwd):
            raise ValueError("Runner cwd must be relative.")
        resolved = posixpath.normpath(posixpath.join(self.default_cwd, relative_cwd))
        if not _is_same_or_child(resolved, self.default_cwd):
            raise ValueError("Runner cwd escapes the runner root.")
        return resolved


class _LimitedText:
    def __init__(self, limit: int | None) -> None:
        self.limit = limit
        self.content = bytearray()
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
    )


def _is_command_exit(exc: Exception) -> bool:
    return type(getattr(exc, "exit_code", None)) is int


def _is_timeout_error(exc: Exception) -> bool:
    return isinstance(exc, TimeoutError) or exc.__class__.__name__ == "TimeoutException"


def _is_same_or_child(path: str, root: str) -> bool:
    if root == "/":
        return posixpath.isabs(path)
    return path == root or path.startswith(f"{root.rstrip('/')}/")


async def _kill_handle_best_effort(handle: Any) -> None:
    kill = getattr(handle, "kill", None)
    if kill is not None:
        try:
            await kill()
        except Exception:
            return


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
