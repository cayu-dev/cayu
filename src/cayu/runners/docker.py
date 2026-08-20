from __future__ import annotations

import asyncio
import os
import posixpath
import shlex
import shutil
from collections.abc import Mapping, Sequence
from typing import Literal
from uuid import uuid4

from cayu._validation import require_clean_nonblank, require_durable_clean_nonblank
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
from cayu.runners._docker_cli import docker_cli_env, normalize_docker_cli_env_allowlist
from cayu.runners._secrets import (
    merge_secret_env_values,
    normalize_runner_secret_env,
    redact_exec_result,
    resolved_secret_redactor,
    runner_env_file,
    validate_runner_env_file_environment,
    validate_secret_env_collisions,
)
from cayu.runners._subprocess import (
    SubprocessCommand,
    copy_runner_env,
    remove_runner_env,
    run_subprocess,
    validate_output_limit,
    validate_runner_env_remove,
    validate_stdin,
    validate_timeout,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    _clean_runner_preflight,
    _clear_preflight_traceback_frames,
    attach_cancellation_artifacts,
    copy_exec_command,
)
from cayu.runners.workloads import BROWSER_FETCH_WORKLOAD_NAME, PINNED_BROWSER_FETCH_WORKLOAD
from cayu.vaults import (
    SecretEnv,
    SecretRedactor,
    SecretRef,
    SecretResolver,
    resolve_secret_env,
)

DEFAULT_DOCKER_IMAGE = "debian:stable-slim"
DEFAULT_DOCKER_CWD = "/workspace"
DOCKER_COMMAND_STATE_DIR = "/tmp/cayu-docker-commands"
_DOCKER_LIFECYCLE_DIAGNOSTIC_MAX_BYTES = 300

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


def _docker_lifecycle_redactor(
    env_overlay: Mapping[str, str] | None,
    docker_cli_env_allowlist: Sequence[str],
) -> SecretRedactor:
    values = [
        value
        for value in (
            *(env_overlay or {}).values(),
            *(os.environ.get(name) for name in docker_cli_env_allowlist),
        )
        if type(value) is str and value
    ]
    return SecretRedactor(values)


def _docker_lifecycle_detail(value: str, redactor: SecretRedactor) -> str:
    detail = redactor.redact_text_bounded(
        value.strip(),
        max_bytes=_DOCKER_LIFECYCLE_DIAGNOSTIC_MAX_BYTES,
    )
    return detail or "no stderr"


def _validate_guest_cwd(cwd: str) -> str:
    value = require_durable_clean_nonblank(cwd, "default_cwd")
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


def validate_docker_seccomp_profile(path: str | None) -> str | None:
    """Return an owned absolute Docker seccomp-profile path."""

    if path is None:
        return None
    value = require_clean_nonblank(path, "seccomp_profile")
    if not os.path.isabs(value):
        raise ValueError("DockerRunner seccomp_profile must be an absolute host path.")
    value = os.path.realpath(value)
    if not os.path.isfile(value):
        raise ValueError("DockerRunner seccomp_profile must be an existing regular file.")
    if os.path.getsize(value) > 1024 * 1024:
        raise ValueError("DockerRunner seccomp_profile must not exceed 1 MiB.")
    return value


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
    docker_path: str,
    args: list[str],
    *,
    docker_cli_env_allowlist: Sequence[str] = (),
    timeout_s: int | None = None,
) -> ExecResult:
    host_env = docker_cli_env(docker_cli_env_allowlist)
    allowlisted_redactor = SecretRedactor(
        tuple(value for name in docker_cli_env_allowlist if (value := host_env.get(name)))
    )
    return await run_subprocess(
        SubprocessCommand(argv=[docker_path, *args]),
        env=host_env,
        timeout_s=timeout_s,
        output_redactor=allowlisted_redactor,
    )


def _supervised_command_script(command_script: str, pid_file: str) -> str:
    quoted_state_dir = shlex.quote(posixpath.dirname(pid_file))
    setsid_body = _supervised_command_body(command_script, pid_file=pid_file, process_group=True)
    fallback_body = _supervised_command_body(command_script, pid_file=pid_file, process_group=False)
    return (
        f"mkdir -p {quoted_state_dir}; "
        "if setsid -w true >/dev/null 2>&1; then "
        f"exec setsid -w sh -c {shlex.quote(setsid_body)}; "
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
    def __init__(
        self,
        *,
        docker_path: str,
        name: str,
        pid_file: str,
        docker_cli_env_allowlist: Sequence[str],
    ) -> None:
        self.docker_path = docker_path
        self.name = name
        self.pid_file = pid_file
        self.docker_cli_env_allowlist = tuple(docker_cli_env_allowlist)

    async def kill(self) -> bool:
        for _ in range(RUNNER_COMMAND_KILL_ATTEMPTS):
            result = await _run_docker(
                self.docker_path,
                ["exec", self.name, "sh", "-c", _kill_supervised_command_script(self.pid_file)],
                docker_cli_env_allowlist=self.docker_cli_env_allowlist,
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
            docker_cli_env_allowlist=self.docker_cli_env_allowlist,
        )
        return probe.exit_code == 1


class DockerRunner(Runner):
    """Executes commands inside a plain Docker container via the ``docker`` CLI.

    Isolation is a parameter: pass ``runtime="runsc"`` (gVisor) or ``"kata"``
    (microVM) to ``create`` for a hardened boundary; the default (``runc``) is a
    convenience tier for trusted development, CI, conformance, and packaging,
    **not** a security boundary. Cayu never selects it implicitly for untrusted
    code. The host ``docker`` process receives only an explicit operational
    allowlist; the containerized command
    receives only the explicit per-call ``env`` plus declared ``secret_env``,
    carried through a private ``--env-file`` so values are never in
    host-visible argv or the Docker CLI's own process environment. ``secret_env``
    entries are resolved through ``secret_resolver`` at exec time and redacted
    from captured output. Private-registry credential helpers that need additional
    host variables must receive their names explicitly through
    ``docker_cli_env_allowlist``; those trusted grants reach only the host-side
    Docker CLI and its helpers, never the container command.
    """

    isolation = "docker"

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("docker", self.name)

    def _execution_profile_material(self) -> dict[str, object] | None:
        """Return portable configuration when no deployment-local grants are present."""

        # Secret declarations, guest overlays, and host CLI environment grants
        # can carry or resolve deployment-local values. They must not become a
        # durable public hash oracle.
        if self.secret_env or self.env_overlay or self.docker_cli_env_allowlist:
            return None
        return {
            "name": self.name,
            "image": self.image,
            "default_cwd": self.default_cwd,
            "close_action": self.close_action,
            "docker_path": self.docker_path,
            "credential_mode": self.credential_mode.value,
            "allow_raw_secret_env": self._allow_raw_secret_env,
            "cancel_timeout_s": self.cancel_timeout_s,
            "cancellation_cleanup": self.cancellation_cleanup,
            "timeout_cleanup": self.timeout_cleanup,
        }

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
        docker_cli_env_allowlist: Sequence[str] = (),
        image: str | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.default_cwd = _validate_guest_cwd(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.docker_path = _require_docker(docker_path)
        self.image = None if image is None else require_clean_nonblank(image, "image")
        self.credential_mode = normalize_credential_mode(credential_mode)
        self._allow_raw_secret_env = allow_raw_secret_env
        self.secret_env, self.secret_resolver = normalize_runner_secret_env(
            secret_env,
            secret_resolver,
            credential_mode=self.credential_mode,
            allow_raw_secret_env=allow_raw_secret_env,
        )
        # Trusted egress overlay (proxy vars + CA trust). Applied last on every
        # exec so model-controlled env cannot unset the enforced egress path.
        self.env_overlay = dict(env_overlay) if env_overlay else {}
        self.docker_cli_env_allowlist = normalize_docker_cli_env_allowlist(docker_cli_env_allowlist)
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
        seccomp_profile: str | None = None,
        docker_cli_env_allowlist: Sequence[str] = (),
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

        ``seccomp_profile`` applies an explicit host-side Docker seccomp profile
        to the container. ``docker_cli_env_allowlist`` explicitly grants named host environment
        variables to the trusted Docker CLI and its credential helpers. It is for
        registry or daemon authentication only and never enters the guest.
        Managed containers always use Docker's minimal init as PID 1 so cleanup
        descendants orphaned by a worker exit are reaped after they finish.
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
        docker_cli_allowlist = normalize_docker_cli_env_allowlist(docker_cli_env_allowlist)
        seccomp_profile = validate_docker_seccomp_profile(seccomp_profile)
        if default_cwd is None:
            default_cwd = mount_path if mount_path is not None else DEFAULT_DOCKER_CWD
        default_cwd = _validate_guest_cwd(default_cwd)
        mode = normalize_credential_mode(credential_mode)
        validated_secret_env, validated_secret_resolver = normalize_runner_secret_env(
            secret_env,
            secret_resolver,
            credential_mode=mode,
            allow_raw_secret_env=allow_raw_secret_env,
        )
        try:
            if replace:
                await _run_docker(
                    docker,
                    ["rm", "-f", name],
                    docker_cli_env_allowlist=docker_cli_allowlist,
                )
            # `docker exec` commands can intentionally delegate bounded cleanup
            # to a descendant which outlives the command's process group. Keep
            # Docker's minimal init as PID 1 so those orphaned owners are reaped
            # after they finish; plain `sleep infinity` cannot reap zombies.
            run_argv = ["run", "-d", "--init"]
            if runtime:
                run_argv += ["--runtime", runtime]
            if seccomp_profile is not None:
                run_argv += ["--security-opt", f"seccomp={seccomp_profile}"]
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
            started = await _run_docker(
                docker,
                run_argv,
                docker_cli_env_allowlist=docker_cli_allowlist,
            )
            if started.exit_code != 0:
                detail = _docker_lifecycle_detail(
                    started.stderr,
                    _docker_lifecycle_redactor(env_overlay, docker_cli_allowlist),
                )
                raise RuntimeError(f"docker run failed (exit {started.exit_code}): {detail}")
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
                    docker_cli_env_allowlist=docker_cli_allowlist,
                )
                if made.exit_code != 0:
                    detail = _docker_lifecycle_detail(
                        made.stderr,
                        _docker_lifecycle_redactor(env_overlay, docker_cli_allowlist),
                    )
                    raise RuntimeError(f"docker workspace mkdir failed: {detail}")
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
                    res = await _run_docker(
                        docker,
                        setup_argv,
                        docker_cli_env_allowlist=docker_cli_allowlist,
                        timeout_s=300,
                    )
                if res.exit_code != 0:
                    detail = _docker_lifecycle_detail(
                        res.stderr,
                        _docker_lifecycle_redactor(env_overlay, docker_cli_allowlist),
                    )
                    raise RuntimeError(f"docker setup command failed: {detail}")
        except BaseException:
            await _run_docker(
                docker,
                ["rm", "-f", name],
                docker_cli_env_allowlist=docker_cli_allowlist,
            )
            raise
        return cls(
            name,
            image=image,
            default_cwd=default_cwd,
            close_action=close_action,
            docker_path=docker,
            cancel_timeout_s=cancel_timeout,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            secret_env=validated_secret_env,
            secret_resolver=validated_secret_resolver,
            credential_mode=mode,
            allow_raw_secret_env=allow_raw_secret_env,
            env_overlay=env_overlay,
            docker_cli_env_allowlist=docker_cli_allowlist,
        )

    def workload_authority(self, name: str):
        """Declare a shipped workload only for its exact selected image."""

        if name != BROWSER_FETCH_WORKLOAD_NAME or self.image != PINNED_BROWSER_FETCH_WORKLOAD.image:
            return None
        return PINNED_BROWSER_FETCH_WORKLOAD

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        operation = self._exec(
            command,
            output_redactor=SecretRedactor(),
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        return await operation

    async def exec_redacted(
        self,
        command: ExecCommand,
        *,
        redactor: SecretRedactor,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        if not isinstance(redactor, SecretRedactor):
            raise TypeError("DockerRunner redactor must be a SecretRedactor.")
        operation = self._exec(
            command,
            output_redactor=redactor,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del command, redactor, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        return await operation

    @_clean_runner_preflight
    def preflight_exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> None:
        """Validate Docker's exact transport contract without side effects."""

        prepared = self._prepare_exec_request(
            command,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        del prepared

    def _prepare_exec_request(
        self,
        command: ExecCommand,
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        env_remove: tuple[str, ...],
        timeout_s: int | None,
        stdin: str | None,
        output_limit_bytes: int | None,
    ) -> tuple[
        ExecCommand,
        str,
        dict[str, str],
        dict[str, str],
        tuple[str, ...],
        dict[str, SecretRef],
        SecretResolver | None,
        int | None,
        str | None,
        int | None,
    ]:
        """Own one complete Docker request before lookup or dispatch."""

        if type(command) is not ExecCommand:
            raise TypeError("DockerRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        owned_command = copy_exec_command(command)
        environment = copy_runner_env(env, inherit_env=False)
        env_overlay = copy_runner_env(self.env_overlay, inherit_env=False)
        validated_env_remove = validate_runner_env_remove(env_remove)
        working_dir = self.resolve_cwd(cwd)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        output_limit = validate_output_limit(output_limit_bytes)
        validate_runner_env_file_environment(environment)
        validate_runner_env_file_environment(env_overlay)
        declared_secret_env, secret_resolver = normalize_runner_secret_env(
            self.secret_env,
            self.secret_resolver,
            credential_mode=self.credential_mode,
            allow_raw_secret_env=self._allow_raw_secret_env,
        )
        validate_runner_env_file_environment(dict.fromkeys(declared_secret_env, ""))
        validate_secret_env_collisions(environment, declared_secret_env)
        return (
            owned_command,
            working_dir,
            environment,
            env_overlay,
            validated_env_remove,
            declared_secret_env,
            secret_resolver,
            timeout,
            standard_input,
            output_limit,
        )

    async def _exec(
        self,
        command: ExecCommand,
        *,
        output_redactor: SecretRedactor,
        cwd: str | None,
        env: dict[str, str] | None,
        env_remove: tuple[str, ...],
        timeout_s: int | None,
        stdin: str | None,
        output_limit_bytes: int | None,
    ) -> ExecResult:
        try:
            (
                owned_command,
                working_dir,
                environment,
                env_overlay,
                validated_env_remove,
                declared_secret_env,
                secret_resolver,
                timeout,
                standard_input,
                output_limit,
            ) = self._prepare_exec_request(
                command,
                cwd=cwd,
                env=env,
                env_remove=env_remove,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=output_limit_bytes,
            )
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            owned_command = None
            working_dir = ""
            environment = {}
            env_overlay = {}
            validated_env_remove = ()
            declared_secret_env = {}
            secret_resolver = None
            standard_input = None
            cwd = None
            env = None
            env_remove = ()
            stdin = None
            del output_redactor
            raise
        finally:
            del command
        env = None
        stdin = None
        resolved_secrets = (
            await resolve_secret_env(declared_secret_env, secret_resolver)
            if declared_secret_env and secret_resolver is not None
            else {}
        )
        environment = merge_secret_env_values(environment, resolved_secrets)
        environment = remove_runner_env(environment, validated_env_remove)
        invocation_redactor = resolved_secret_redactor(resolved_secrets).merged_with(
            output_redactor
        )
        if env_overlay:
            # Applied last: the enforced egress overlay must win over model env.
            environment.update(env_overlay)
        command_id = uuid4().hex
        pid_file = f"{DOCKER_COMMAND_STATE_DIR}/{command_id}.pid"
        handle = _DockerCommandHandle(
            docker_path=self.docker_path,
            name=self.name,
            pid_file=pid_file,
            docker_cli_env_allowlist=self.docker_cli_env_allowlist,
        )
        with runner_env_file(environment) as env_file:
            environment = {}
            argv = _build_docker_exec_argv(
                self.docker_path,
                self.name,
                owned_command,
                cwd=working_dir,
                env_file=env_file,
                has_stdin=standard_input is not None,
                pid_file=pid_file,
            )
            # The trusted host docker process receives only the bounded operational
            # allowlist. Container env values ride in --env-file,
            # never in the CLI's own environment, so a model-controlled env cannot hijack
            # the host CLI (e.g. by setting DOCKER_HOST to an attacker daemon).
            host_env = docker_cli_env(self.docker_cli_env_allowlist)
            try:
                result = await run_subprocess(
                    SubprocessCommand(argv=argv),
                    env=host_env,
                    timeout_s=timeout,
                    stdin=standard_input,
                    output_limit_bytes=output_limit,
                    output_redactor=invocation_redactor,
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
        result = await _run_docker(
            self.docker_path,
            ["rm", "-f", self.name],
            docker_cli_env_allowlist=self.docker_cli_env_allowlist,
        )
        if result.exit_code != 0:
            detail = _docker_lifecycle_detail(
                result.stderr,
                _docker_lifecycle_redactor(
                    self.env_overlay,
                    self.docker_cli_env_allowlist,
                ),
            )
            raise RuntimeError(
                f"docker rm failed for container '{self.name}' (exit {result.exit_code}): {detail}"
            )

    async def _stop_container(self) -> None:
        result = await _run_docker(
            self.docker_path,
            ["stop", self.name],
            docker_cli_env_allowlist=self.docker_cli_env_allowlist,
        )
        if result.exit_code != 0:
            detail = _docker_lifecycle_detail(
                result.stderr,
                _docker_lifecycle_redactor(
                    self.env_overlay,
                    self.docker_cli_env_allowlist,
                ),
            )
            raise RuntimeError(
                f"docker stop failed for container '{self.name}' "
                f"(exit {result.exit_code}): {detail}"
            )
