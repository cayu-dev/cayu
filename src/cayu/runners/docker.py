from __future__ import annotations

import asyncio
import os
import posixpath
import shlex
import shutil
from collections.abc import Mapping, Sequence
from typing import Literal
from uuid import uuid4

from cayu._validation import require_clean_nonblank
from cayu.credentials import CredentialMode, CredentialModeInput, normalize_credential_mode
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

DEFAULT_DOCKER_IMAGE = "debian:stable-slim"
DEFAULT_DOCKER_CWD = "/workspace"
DOCKER_COMMAND_STATE_DIR = "/tmp/cayu-docker-commands"

DockerCloseAction = Literal["remove", "stop", "none"]


def _require_docker(docker_path: str | None) -> str:
    candidate = docker_path or shutil.which("docker")
    if not candidate:
        raise RuntimeError(
            "docker CLI not found. Install Docker "
            "(https://docs.docker.com/get-docker/) or pass docker_path=."
        )
    return candidate


def _validate_close_action(action: str) -> str:
    if action not in {"remove", "stop", "none"}:
        raise ValueError("close_action must be 'remove', 'stop', or 'none'.")
    return action


def _validate_guest_cwd(cwd: str) -> str:
    value = require_clean_nonblank(cwd, "default_cwd")
    if not posixpath.isabs(value):
        raise ValueError("DockerRunner default_cwd must be an absolute guest path.")
    return posixpath.normpath(value)


def _validate_runtime(runtime: str | None) -> str | None:
    if runtime is None:
        return None
    return require_clean_nonblank(runtime, "runtime")


def _validate_mount_path(mount_path: str) -> str:
    value = require_clean_nonblank(mount_path, "mount_path")
    if not os.path.isabs(value):
        raise ValueError("DockerRunner mount_path must be an absolute host path.")
    if "," in value:
        # docker's ``--mount`` uses commas to separate key=value pairs, so a
        # comma in the path would silently corrupt the bind specification.
        raise ValueError("DockerRunner mount_path must not contain commas.")
    value = os.path.normpath(value)
    if not os.path.isdir(value):
        raise ValueError(f"DockerRunner mount_path must be an existing directory: {value!r}")
    return value


def _validate_ca_mount(ca_mount: tuple[str, str]) -> tuple[str, str]:
    host_path, guest_path = ca_mount
    host_path = require_clean_nonblank(host_path, "ca_mount host path")
    guest_path = require_clean_nonblank(guest_path, "ca_mount guest path")
    if not os.path.isabs(host_path) or not os.path.isfile(host_path):
        raise ValueError("ca_mount host path must be an existing absolute file.")
    if not posixpath.isabs(guest_path):
        raise ValueError("ca_mount guest path must be an absolute guest path.")
    if "," in host_path or "," in guest_path:
        raise ValueError("ca_mount paths must not contain commas.")
    return host_path, guest_path


def _build_docker_exec_argv(
    docker_path: str,
    name: str,
    command: ExecCommand,
    *,
    cwd: str,
    env_file: str | None,
    has_stdin: bool,
    pid_file: str,
) -> list[str]:
    argv: list[str] = [docker_path, "exec"]
    if has_stdin:
        argv.append("-i")
    argv += ["-w", cwd]
    if env_file is not None:
        # --env-file passes container env values from a private file, keeping them out of
        # host-visible argv AND out of the docker CLI's own process environment (which a
        # model-controlled env could otherwise use to hijack the CLI, e.g. DOCKER_HOST).
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


async def _run_docker(
    docker_path: str, args: list[str], *, timeout_s: int | None = None
) -> ExecResult:
    return await run_subprocess(
        SubprocessCommand(argv=[docker_path, *args]),
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
    quoted_command_script = shlex.quote(command_script)
    process_group_flag = "1" if process_group else "0"
    return (
        f'printf \'%s %s\\n\' "$$" "{process_group_flag}" > {quoted_pid_file} || exit 1; '
        f"sh -c {quoted_command_script}; "
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
        'if test "$process_group" = 1; then '
        'kill -TERM "-$pid" 2>/dev/null || kill -TERM -- "-$pid" 2>/dev/null || '
        'kill -TERM "$pid" 2>/dev/null || true; '
        "sleep 0.2; "
        'kill -KILL "-$pid" 2>/dev/null || kill -KILL -- "-$pid" 2>/dev/null || '
        'kill -KILL "$pid" 2>/dev/null || true; '
        "else "
        'kill -TERM "$pid" 2>/dev/null || true; '
        "sleep 0.2; "
        'kill -KILL "$pid" 2>/dev/null || true; '
        "fi; "
        f"rm -f {quoted_pid_file}; "
        "exit 0"
    )


class _DockerCommandHandle:
    def __init__(self, *, docker_path: str, name: str, pid_file: str) -> None:
        self.docker_path = docker_path
        self.name = name
        self.pid_file = pid_file

    async def kill(self) -> bool:
        for _ in range(RUNNER_COMMAND_KILL_ATTEMPTS):
            result = await _run_docker(
                self.docker_path,
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
        # command. Any other exit code (docker transport failure with the
        # container still up, etc.) stays a failure.
        probe = await _run_docker(
            self.docker_path,
            ["exec", self.name, "sh", "-c", f"test -f {shlex.quote(self.pid_file)}"],
        )
        return probe.exit_code == 1


class DockerRunner(Runner):
    """Executes commands inside a plain Docker container via the ``docker`` CLI.

    Isolation is a parameter: pass ``runtime="runsc"`` (gVisor) or ``"kata"``
    (microVM) to ``create`` for a hardened boundary; the default (``runc``) is a
    convenience tier for trusted development, CI, conformance, and packaging,
    **not** a security boundary. Cayu never selects it implicitly for untrusted
    code. The host ``docker`` process
    inherits the host environment (the CLI needs it); the containerized command
    receives only the explicit per-call ``env`` plus declared ``secret_env``,
    carried through a private ``--env-file`` so values are never in
    host-visible argv or the Docker CLI's own process environment. ``secret_env``
    entries are resolved through ``secret_resolver`` at exec time and redacted
    from captured output.
    """

    isolation = "docker"

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("docker", self.name)

    def __init__(
        self,
        name: str,
        *,
        default_cwd: str = DEFAULT_DOCKER_CWD,
        close_action: DockerCloseAction = "none",
        docker_path: str | None = None,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef] = (),
        secret_resolver: SecretResolver | None = None,
        credential_mode: CredentialModeInput = CredentialMode.RAW_ENV,
        allow_raw_secret_env: bool = True,
        env_overlay: Mapping[str, str] | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.default_cwd = _validate_guest_cwd(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.docker_path = _require_docker(docker_path)
        self.credential_mode = normalize_credential_mode(credential_mode)
        self.secret_env, self.secret_resolver = normalize_runner_secret_env(
            secret_env,
            secret_resolver,
            credential_mode=self.credential_mode,
            allow_raw_secret_env=allow_raw_secret_env,
        )
        # Trusted egress overlay (proxy vars + CA trust). Applied last on every
        # exec so model-controlled env cannot unset the enforced egress path.
        self.env_overlay = dict(env_overlay) if env_overlay else {}
        self.cancel_timeout_s = validate_cancel_timeout(cancel_timeout_s)
        self.cancellation_cleanup = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        self.timeout_cleanup = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        image: str = DEFAULT_DOCKER_IMAGE,
        runtime: str | None = None,
        mount_path: str | None = None,
        default_cwd: str | None = None,
        close_action: DockerCloseAction = "remove",
        setup_commands: tuple[str, ...] = (),
        docker_path: str | None = None,
        replace: bool = True,
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef] = (),
        secret_resolver: SecretResolver | None = None,
        credential_mode: CredentialModeInput = CredentialMode.RAW_ENV,
        allow_raw_secret_env: bool = True,
        network: str | None = None,
        extra_hosts: Sequence[str] = (),
        env_overlay: Mapping[str, str] | None = None,
        ca_mount: tuple[str, str] | None = None,
    ) -> DockerRunner:
        """Start a long-lived container and return a runner bound to it.

        With ``mount_path`` the host dir is bind-mounted at the same absolute path
        (convenience; pair with ``LocalWorkspace``). Without it, an in-container
        ``default_cwd`` is created (hardened; pair with ``RunnerWorkspace``, which
        needs python3 — install it via ``setup_commands``). ``runtime`` is passed
        to ``docker run --runtime`` (e.g. ``runsc``/``kata``). ``setup_commands``
        run as root.

        For virtual egress, ``network`` attaches the container to a Docker network
        (e.g. an ``--internal`` one that blocks direct internet), ``extra_hosts``
        adds ``--add-host`` entries, ``ca_mount`` bind-mounts a
        ``(host_path, guest_path)`` CA read-only, and ``env_overlay`` is applied to
        every exec's environment (after model env, so it cannot be unset).
        """
        docker = _require_docker(docker_path)
        name = require_clean_nonblank(name, "name")
        image = require_clean_nonblank(image, "image")
        runtime = _validate_runtime(runtime)
        if mount_path is not None:
            mount_path = _validate_mount_path(mount_path)
        _validate_close_action(close_action)
        cancel_timeout = validate_cancel_timeout(cancel_timeout_s)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        if default_cwd is None:
            default_cwd = mount_path if mount_path is not None else DEFAULT_DOCKER_CWD
        default_cwd = _validate_guest_cwd(default_cwd)
        mode = normalize_credential_mode(credential_mode)
        normalize_runner_secret_env(
            secret_env,
            secret_resolver,
            credential_mode=mode,
            allow_raw_secret_env=allow_raw_secret_env,
        )
        try:
            if replace:
                await _run_docker(docker, ["rm", "-f", name])
            run_argv = ["run", "-d"]
            if runtime:
                run_argv += ["--runtime", runtime]
            run_argv += ["--name", name]
            if network is not None:
                run_argv += ["--network", require_clean_nonblank(network, "network")]
            for host_entry in extra_hosts:
                run_argv += ["--add-host", require_clean_nonblank(host_entry, "extra_hosts")]
            if mount_path is not None:
                run_argv += ["--mount", f"type=bind,source={mount_path},target={mount_path}"]
            if ca_mount is not None:
                ca_host, ca_guest = _validate_ca_mount(ca_mount)
                run_argv += [
                    "--mount",
                    f"type=bind,source={ca_host},target={ca_guest},readonly",
                ]
            run_argv += [image, "sleep", "infinity"]
            started = await _run_docker(docker, run_argv)
            if started.exit_code != 0:
                raise RuntimeError(
                    f"docker run failed (exit {started.exit_code}): {started.stderr[:300]}"
                )
            # Isolated mode: create the in-container workspace root (runs as root;
            # plain docker's default exec user is root, so no chmod needed). Bind
            # mode reuses the existing host dir, so skip (and never chmod the host).
            if mount_path is None:
                made = await _run_docker(
                    docker,
                    [
                        "exec",
                        "-u",
                        "root",
                        name,
                        "sh",
                        "-c",
                        f"mkdir -p {shlex.quote(default_cwd)}",
                    ],
                )
                if made.exit_code != 0:
                    raise RuntimeError(f"docker workspace mkdir failed: {made.stderr[:300]}")
            # Setup runs on the (already-attached) network with the egress
            # overlay applied, so any setup traffic is brokered like the app's —
            # it is subject to the same egress policy, so bake tools that need
            # arbitrary hosts into the image rather than installing them here.
            setup_environment = dict(env_overlay or {})
            for cmd in setup_commands:
                with runner_env_file(setup_environment) as setup_env_file:
                    setup_argv = ["exec", "-u", "root"]
                    if setup_env_file is not None:
                        setup_argv += ["--env-file", setup_env_file]
                    setup_argv += [name, "sh", "-c", cmd]
                    res = await _run_docker(docker, setup_argv, timeout_s=300)
                if res.exit_code != 0:
                    raise RuntimeError(f"docker setup command failed: {cmd!r}: {res.stderr[:300]}")
        except BaseException:
            await _run_docker(docker, ["rm", "-f", name])
            raise
        return cls(
            name,
            default_cwd=default_cwd,
            close_action=close_action,
            docker_path=docker,
            cancel_timeout_s=cancel_timeout,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            secret_env=secret_env,
            secret_resolver=secret_resolver,
            credential_mode=mode,
            allow_raw_secret_env=allow_raw_secret_env,
            env_overlay=env_overlay,
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
            raise TypeError("DockerRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        environment = copy_runner_env(env, inherit_env=False)
        resolved_secrets = (
            await resolve_secret_env(self.secret_env, self.secret_resolver)
            if self.secret_env and self.secret_resolver is not None
            else {}
        )
        environment = merge_secret_env_values(environment, resolved_secrets)
        if self.env_overlay:
            # Applied last: the enforced egress overlay must win over model env.
            environment.update(self.env_overlay)
        command_id = uuid4().hex
        pid_file = f"{DOCKER_COMMAND_STATE_DIR}/{command_id}.pid"
        handle = _DockerCommandHandle(
            docker_path=self.docker_path,
            name=self.name,
            pid_file=pid_file,
        )
        with runner_env_file(environment) as env_file:
            argv = _build_docker_exec_argv(
                self.docker_path,
                self.name,
                command,
                cwd=self.resolve_cwd(cwd),
                env_file=env_file,
                has_stdin=stdin is not None,
                pid_file=pid_file,
            )
            # The host docker process runs with a pristine inherited env (PATH/HOME/
            # DOCKER_HOST/docker config only). Container env values ride in --env-file,
            # never in the CLI's own environment, so a model-controlled env cannot hijack
            # the host CLI (e.g. by setting DOCKER_HOST to an attacker daemon).
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
                    adapter="docker",
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
                adapter="docker",
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
            await self._remove_container()
        elif self.close_action == "stop":
            await self._stop_container()
        self._closed = True

    async def kill(self) -> bool:
        """Remove the Docker container for shared runner cleanup diagnostics."""

        if self._closed:
            return True
        await self._remove_container()
        self._closed = True
        return True

    async def _remove_container(self) -> None:
        result = await _run_docker(self.docker_path, ["rm", "-f", self.name])
        if result.exit_code != 0:
            raise RuntimeError(
                f"docker rm failed for container '{self.name}' "
                f"(exit {result.exit_code}): {result.stderr[:300]}"
            )

    async def _stop_container(self) -> None:
        result = await _run_docker(self.docker_path, ["stop", self.name])
        if result.exit_code != 0:
            raise RuntimeError(
                f"docker stop failed for container '{self.name}' "
                f"(exit {result.exit_code}): {result.stderr[:300]}"
            )
