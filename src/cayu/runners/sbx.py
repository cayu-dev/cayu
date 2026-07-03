from __future__ import annotations

import asyncio
import posixpath
import shlex
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from typing import Literal
from uuid import uuid4

from cayu._validation import require_clean_nonblank
from cayu.runners._cleanup import (
    DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
    DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
    DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
    RUNNER_COMMAND_KILL_ATTEMPTS,
    RunnerCleanupPolicy,
    cleanup_runner_command_with_diagnostic,
    validate_cancel_timeout,
    validate_runner_cleanup_policy,
)
from cayu.runners._secrets import (
    merge_secret_env_values,
    normalize_runner_secret_env,
    redact_exec_result,
    runner_env_file,
)
from cayu.runners._subprocess import SubprocessCommand, copy_runner_env, run_subprocess
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    attach_cancellation_artifacts,
)
from cayu.vaults import SecretEnv, SecretRef, SecretResolver, resolve_secret_env

DEFAULT_SBX_AGENT = "shell"
DEFAULT_SBX_CWD = "/workspace"
SBX_COMMAND_STATE_DIR = "/tmp/cayu-sbx-commands"

SbxCloseAction = Literal["remove", "stop", "none"]


def _require_sbx(sbx_path: str | None) -> str:
    candidate = sbx_path or shutil.which("sbx")
    if not candidate:
        raise RuntimeError(
            "sbx CLI not found. Install Docker Sandboxes "
            "(https://docs.docker.com/ai/sandboxes/) or pass sbx_path=."
        )
    return candidate


def _validate_close_action(action: str) -> str:
    if action not in {"remove", "stop", "none"}:
        raise ValueError("close_action must be 'remove', 'stop', or 'none'.")
    return action


def _validate_guest_cwd(cwd: str) -> str:
    value = require_clean_nonblank(cwd, "default_cwd")
    if not posixpath.isabs(value):
        raise ValueError("SbxRunner default_cwd must be an absolute guest path.")
    return posixpath.normpath(value)


def _build_sbx_exec_argv(
    sbx_path: str,
    name: str,
    command: ExecCommand,
    *,
    cwd: str,
    env_file: str | None,
    has_stdin: bool,
    pid_file: str,
) -> list[str]:
    argv: list[str] = [sbx_path, "exec"]
    if has_stdin:
        argv.append("-i")
    argv += ["-w", cwd]
    if env_file is not None:
        # --env-file passes sandbox env values from a private file, keeping them out of
        # host-visible argv AND out of the sbx CLI's own process environment (which a
        # model-controlled env could otherwise use to hijack the CLI).
        argv += ["--env-file", env_file]
    argv.append(name)
    if command.kind == "process":
        if command.argv is None:
            raise ValueError("Process commands require argv.")
        command_script = shlex.join(command.argv)
    else:
        if command.shell is None:
            raise ValueError("Shell commands require a script.")
        command_script = command.shell
    argv += ["sh", "-c", _supervised_command_script(command_script, pid_file)]
    return argv


async def _run_sbx(sbx_path: str, args: list[str], *, timeout_s: int | None = None) -> ExecResult:
    return await run_subprocess(
        SubprocessCommand(argv=[sbx_path, *args]),
        env=copy_runner_env(None, inherit_env=True),
        timeout_s=timeout_s,
    )


def _supervised_command_script(command_script: str, pid_file: str) -> str:
    quoted_state_dir = shlex.quote(posixpath.dirname(pid_file))
    setsid_body = _supervised_command_body(command_script, pid_file=pid_file, process_group=True)
    fallback_body = _supervised_command_body(command_script, pid_file=pid_file, process_group=False)
    return (
        f"mkdir -p {quoted_state_dir}; "
        "if command -v setsid >/dev/null 2>&1; then "
        f"exec setsid sh -c {shlex.quote(setsid_body)}; "
        "else "
        f"exec sh -c {shlex.quote(fallback_body)}; "
        "fi"
    )


def _supervised_command_body(
    command_script: str,
    *,
    pid_file: str,
    process_group: bool,
) -> str:
    quoted_pid_file = shlex.quote(pid_file)
    process_group_flag = "1" if process_group else "0"
    return (
        f'printf \'%s %s\\n\' "$$" "{process_group_flag}" > {quoted_pid_file} || exit 1; '
        f"sh -c {shlex.quote(command_script)}; "
        "status=$?; "
        f"rm -f {quoted_pid_file}; "
        'exit "$status"'
    )


def _kill_supervised_command_script(pid_file: str) -> str:
    quoted_pid_file = shlex.quote(pid_file)
    return (
        "attempts=0; "
        f'while ! test -f {quoted_pid_file} && test "$attempts" -lt 20; do '
        "attempts=$((attempts + 1)); sleep 0.1; "
        "done; "
        f"if ! test -f {quoted_pid_file}; then exit 1; fi; "
        f"read pid process_group < {quoted_pid_file} 2>/dev/null || exit 1; "
        "case \"$pid\" in ''|*[!0-9]*) exit 1 ;; esac; "
        'case "$process_group" in 1) '
        'kill -TERM "-$pid" 2>/dev/null || kill -TERM -- "-$pid" 2>/dev/null || '
        'kill -TERM "$pid" 2>/dev/null || true ;; '
        '*) kill -TERM "$pid" 2>/dev/null || true ;; '
        "esac; "
        "sleep 0.2; "
        'case "$process_group" in 1) '
        'kill -KILL "-$pid" 2>/dev/null || kill -KILL -- "-$pid" 2>/dev/null || '
        'kill -KILL "$pid" 2>/dev/null || true ;; '
        '*) kill -KILL "$pid" 2>/dev/null || true ;; '
        "esac; "
        f"rm -f {quoted_pid_file}; "
        "exit 0"
    )


class _SbxCommandHandle:
    def __init__(self, *, sbx_path: str, name: str, pid_file: str) -> None:
        self.sbx_path = sbx_path
        self.name = name
        self.pid_file = pid_file

    async def kill(self) -> bool:
        for _ in range(RUNNER_COMMAND_KILL_ATTEMPTS):
            result = await _run_sbx(
                self.sbx_path,
                ["exec", self.name, "sh", "-c", _kill_supervised_command_script(self.pid_file)],
            )
            if result.exit_code == 0:
                return True
        return await self._verify_command_not_running()

    async def _verify_command_not_running(self) -> bool:
        # The supervised wrapper writes the pid file before running the command
        # and removes it when the command exits, so `test -f` exiting 1 (file
        # absent) after the kill attempts' wait windows means no tracked
        # command is running — a flaky pid-file wait must not report a live
        # command. Any other exit code (sbx transport failure with the sandbox
        # still up, etc.) stays a failure.
        probe = await _run_sbx(
            self.sbx_path,
            ["exec", self.name, "sh", "-c", f"test -f {shlex.quote(self.pid_file)}"],
        )
        return probe.exit_code == 1


class SbxRunner(Runner):
    """Executes commands inside a Docker Sandbox (sbx) microVM.

    Requires the ``sbx`` CLI (https://docs.docker.com/ai/sandboxes/). The runner
    does not inherit the trusted host environment into the sandbox; pass explicit
    ``env`` per call. Env entries are forwarded by name via ``-e`` with values
    carried in the CLI process env — never in host-visible argv. Declared
    ``secret_env`` entries are resolved through ``secret_resolver`` at exec time
    and redacted from captured output. File I/O is expected via RunnerWorkspace
    (exec-based), so the sandbox guest needs python3.
    """

    isolation = "sbx"

    def __init__(
        self,
        name: str,
        *,
        mount_path: str,
        default_cwd: str = DEFAULT_SBX_CWD,
        close_action: SbxCloseAction = "none",
        sbx_path: str | None = None,
        owns_mount: bool = False,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef] = (),
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.mount_path = require_clean_nonblank(mount_path, "mount_path")
        self.default_cwd = _validate_guest_cwd(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.sbx_path = _require_sbx(sbx_path)
        self.cancel_timeout_s = validate_cancel_timeout(cancel_timeout_s)
        self.cancellation_cleanup = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        self.timeout_cleanup = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        self.secret_env, self.secret_resolver = normalize_runner_secret_env(
            secret_env, secret_resolver
        )
        self._owns_mount = owns_mount

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        default_cwd: str = DEFAULT_SBX_CWD,
        close_action: SbxCloseAction = "remove",
        setup_commands: tuple[str, ...] = (),
        sbx_path: str | None = None,
        replace: bool = True,
        mount_path: str | None = None,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef] = (),
        secret_resolver: SecretResolver | None = None,
    ) -> SbxRunner:
        """Create a sandbox via the sbx CLI and return a runner bound to it.

        A throwaway host dir is mounted only to satisfy `sbx create`; the agent's
        workspace is the isolated in-sandbox `default_cwd`. `setup_commands` run as
        root (e.g. to install whois + python3).
        """
        sbx = _require_sbx(sbx_path)
        name = require_clean_nonblank(name, "name")
        default_cwd = _validate_guest_cwd(default_cwd)
        _validate_close_action(close_action)
        cancel_timeout = validate_cancel_timeout(cancel_timeout_s)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        owns_mount = mount_path is None
        if owns_mount:
            mount_path = tempfile.mkdtemp(prefix="cayu-sbx-")
        try:
            if replace:
                await _run_sbx(sbx, ["rm", "--force", name])
            created = await _run_sbx(sbx, ["create", "--name", name, DEFAULT_SBX_AGENT, mount_path])
            if created.exit_code != 0:
                raise RuntimeError(
                    f"sbx create failed (exit {created.exit_code}): {created.stderr[:300]}"
                )
            # Create the in-sandbox workspace root and make it writable by the
            # sandbox's default (non-root) exec user — mkdir runs as root.
            made = await _run_sbx(
                sbx,
                [
                    "exec",
                    "-u",
                    "root",
                    name,
                    "sh",
                    "-c",
                    f"mkdir -p {shlex.quote(default_cwd)} && chmod 0777 {shlex.quote(default_cwd)}",
                ],
            )
            if made.exit_code != 0:
                raise RuntimeError(f"sbx workspace mkdir failed: {made.stderr[:300]}")
            for cmd in setup_commands:
                res = await _run_sbx(
                    sbx, ["exec", "-u", "root", name, "sh", "-c", cmd], timeout_s=300
                )
                if res.exit_code != 0:
                    raise RuntimeError(f"sbx setup command failed: {cmd!r}: {res.stderr[:300]}")
        except BaseException:
            await _run_sbx(sbx, ["rm", "--force", name])
            if owns_mount:
                shutil.rmtree(mount_path, ignore_errors=True)
            raise
        return cls(
            name,
            mount_path=mount_path,
            default_cwd=default_cwd,
            close_action=close_action,
            sbx_path=sbx,
            owns_mount=owns_mount,
            cancel_timeout_s=cancel_timeout,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            secret_env=secret_env,
            secret_resolver=secret_resolver,
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
            raise TypeError("SbxRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        environment = copy_runner_env(env, inherit_env=False)
        resolved_secrets = (
            await resolve_secret_env(self.secret_env, self.secret_resolver)
            if self.secret_env and self.secret_resolver is not None
            else {}
        )
        environment = merge_secret_env_values(environment, resolved_secrets)
        command_id = uuid4().hex
        pid_file = f"{SBX_COMMAND_STATE_DIR}/{command_id}.pid"
        handle = _SbxCommandHandle(sbx_path=self.sbx_path, name=self.name, pid_file=pid_file)
        with runner_env_file(environment) as env_file:
            argv = _build_sbx_exec_argv(
                self.sbx_path,
                self.name,
                command,
                cwd=self.resolve_cwd(cwd),
                env_file=env_file,
                has_stdin=stdin is not None,
                pid_file=pid_file,
            )
            # The host sbx process runs with a pristine inherited env (PATH/HOME/config
            # only). Sandbox env values ride in --env-file, never in the CLI's own
            # environment, so a model-controlled env cannot hijack the host CLI.
            host_env = copy_runner_env(None, inherit_env=True)
            try:
                result = await run_subprocess(
                    SubprocessCommand(argv=argv),
                    env=host_env,
                    timeout_s=timeout_s,
                    stdin=stdin,
                    output_limit_bytes=output_limit_bytes,
                )
            except asyncio.CancelledError as exc:
                cleanup = await cleanup_runner_command_with_diagnostic(
                    self,
                    handle=handle,
                    adapter="sbx",
                    timeout_s=self.cancel_timeout_s,
                    policy=self.cancellation_cleanup,
                )
                self._apply_cleanup_result(cleanup)
                attach_cancellation_artifacts(exc, [cleanup.artifact])
                raise
        if result.timed_out:
            cleanup = await cleanup_runner_command_with_diagnostic(
                self,
                handle=handle,
                adapter="sbx",
                timeout_s=self.cancel_timeout_s,
                policy=self.timeout_cleanup,
            )
            self._apply_cleanup_result(cleanup)
            result = result.model_copy(update={"artifacts": [*result.artifacts, cleanup.artifact]})
        return redact_exec_result(result, resolved_secrets)

    async def close(self) -> None:
        if self._closed:
            return
        if self.close_action == "remove":
            await self._remove_sandbox()
            if self._owns_mount:
                shutil.rmtree(self.mount_path, ignore_errors=True)
        elif self.close_action == "stop":
            await self._stop_sandbox()
        self._closed = True

    async def kill(self) -> bool:
        """Remove the sbx sandbox for shared runner cleanup diagnostics."""

        if self._closed:
            return True
        await self._remove_sandbox()
        if self._owns_mount:
            shutil.rmtree(self.mount_path, ignore_errors=True)
        self._closed = True
        return True

    async def _remove_sandbox(self) -> None:
        result = await _run_sbx(self.sbx_path, ["rm", "--force", self.name])
        if result.exit_code != 0:
            raise RuntimeError(
                f"sbx rm failed for sandbox '{self.name}' "
                f"(exit {result.exit_code}): {result.stderr[:300]}"
            )

    async def _stop_sandbox(self) -> None:
        result = await _run_sbx(self.sbx_path, ["stop", self.name])
        if result.exit_code != 0:
            raise RuntimeError(
                f"sbx stop failed for sandbox '{self.name}' "
                f"(exit {result.exit_code}): {result.stderr[:300]}"
            )
