from __future__ import annotations

import asyncio
import json
import os
import posixpath
import re
import shlex
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Literal, cast
from uuid import uuid4

from cayu._validation import (
    canonical_durable_json_bytes,
    require_clean_nonblank,
    require_durable_clean_nonblank,
)
from cayu.capabilities import CapabilityDetail
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
from cayu.runners.docker_workload import DockerImageIdentity, DockerWorkloadRestrictions
from cayu.runners.workloads import (
    BROWSER_FETCH_WORKLOAD_NAME,
    BROWSER_SESSION_WORKLOAD_NAME,
    PINNED_BROWSER_FETCH_WORKLOAD,
    PINNED_BROWSER_SESSION_WORKLOAD,
)
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
_DOCKER_CONTAINER_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DOCKER_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_DOCKER_EGRESS_CUTOVER_FENCE_SCRIPT = r"""
import os
import signal
import time

main_candidates = []
for entry in os.listdir("/proc"):
    if not entry.isdigit():
        continue
    pid = int(entry)
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            stat_fields = handle.read().split()
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            command = handle.read()
    except (FileNotFoundError, ProcessLookupError):
        continue
    if int(stat_fields[3]) == 1 and command in {
        b"sleep\x00infinity\x00",
        b"/bin/sleep\x00infinity\x00",
        b"/usr/bin/sleep\x00infinity\x00",
    }:
        main_candidates.append(pid)
if not main_candidates:
    raise SystemExit(71)
protected = {1, min(main_candidates), os.getpid(), os.getppid()}
stable_empty_observations = 0
for _ in range(100):
    candidates = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in protected:
            continue
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
                state = handle.read().split()[2]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if state != "Z":
            candidates.append(pid)
    if not candidates:
        stable_empty_observations += 1
        if stable_empty_observations >= 3:
            raise SystemExit(0)
    else:
        stable_empty_observations = 0
        for pid in candidates:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    time.sleep(0.02)
raise SystemExit(70)
""".strip()

DockerCloseAction = Literal["remove", "stop", "none"]


@dataclass(frozen=True, slots=True)
class _DockerRuntimeEvidence:
    container_id: str
    image_id: str
    image_reference: str
    network_mode: str
    default_cwd: str
    runtime: str | None
    seccomp_profile_sha256: str | None
    restrictions: DockerWorkloadRestrictions
    image_identity: DockerImageIdentity
    toolchain_profile_fingerprint: str | None
    required_executables: tuple[str, ...]
    executable_availability: tuple[tuple[str, bool], ...]
    observed_at: datetime
    valid_until: datetime

    @property
    def environment_fingerprint(self) -> str:
        material = {
            "container_id": self.container_id,
            "image_id": self.image_id,
            "image_reference": self.image_reference,
            "network_mode": self.network_mode,
            "default_cwd": self.default_cwd,
            "runtime": self.runtime,
            "seccomp_profile_sha256": self.seccomp_profile_sha256,
            "restrictions": self.restrictions.model_dump(mode="json"),
            "toolchain_profile_fingerprint": self.toolchain_profile_fingerprint,
            "required_executables": list(self.required_executables),
        }
        return (
            "sha256:"
            + sha256(canonical_durable_json_bytes(material, "docker_runtime_evidence")).hexdigest()
        )

    @property
    def image_fingerprint(self) -> str:
        return self.image_identity.fingerprint


class DockerContainerOwnershipError(RuntimeError):
    """Docker allocation did not yield an exact owned container identity."""


class DockerRuntimeConfigurationError(RuntimeError):
    """The exact created container does not match its declared restrictions."""

    def __init__(self, code: str) -> None:
        self.code = require_durable_clean_nonblank(code, "code")
        super().__init__(f"Docker runtime configuration verification failed: {self.code}.")


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


def _normalize_required_executables(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("required_executables must be a sequence of strings.")
    if len(values) > 64:
        raise ValueError("required_executables must contain at most 64 entries.")
    executables = tuple(
        sorted(
            {require_durable_clean_nonblank(value, "required_executables item") for value in values}
        )
    )
    if any(len(value.encode("utf-8")) > 4096 for value in executables):
        raise ValueError("required_executables entries must not exceed 4096 bytes.")
    return executables


def _seccomp_profile_fingerprint(path: str | None) -> str | None:
    if path is None:
        return None
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _strict_tmpfs_options(
    restrictions: DockerWorkloadRestrictions,
) -> dict[str, frozenset[str]]:
    options: dict[str, frozenset[str]] = {}
    args = restrictions.run_args()
    for index, value in enumerate(args):
        if value != "--tmpfs":
            continue
        target, separator, raw_options = args[index + 1].partition(":")
        if not separator:
            raise AssertionError("Docker tmpfs projection omitted its options.")
        options[target] = frozenset(raw_options.split(","))
    return options


def _require_mapping(value: object, code: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DockerRuntimeConfigurationError(code)
    return cast("Mapping[str, object]", value)


def _require_sequence(value: object, code: str) -> Sequence[object]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise DockerRuntimeConfigurationError(code)
    return value


def _verify_strict_container_inspection(
    inspection: Mapping[str, object],
    *,
    container_id: str,
    image_identity: DockerImageIdentity,
    restrictions: DockerWorkloadRestrictions,
    network_mode: str,
    runtime: str | None,
    seccomp_profile: str | None,
) -> tuple[str, str]:
    if inspection.get("Id") != container_id:
        raise DockerRuntimeConfigurationError("container_identity_drift")
    image_id = inspection.get("Image")
    if type(image_id) is not str or _DOCKER_IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise DockerRuntimeConfigurationError("image_identity_missing")
    if image_identity.content_digest is not None and image_id != image_identity.content_digest:
        raise DockerRuntimeConfigurationError("image_content_digest_mismatch")

    config = _require_mapping(inspection.get("Config"), "container_config_missing")
    image_reference = config.get("Image")
    if type(image_reference) is not str or image_reference != image_identity.reference:
        raise DockerRuntimeConfigurationError("image_reference_mismatch")
    if config.get("User") != restrictions.user:
        raise DockerRuntimeConfigurationError("nonroot_user_drift")

    host = _require_mapping(inspection.get("HostConfig"), "host_config_missing")
    if host.get("NetworkMode") != network_mode:
        raise DockerRuntimeConfigurationError("network_mode_drift")
    if host.get("Privileged") is not False:
        raise DockerRuntimeConfigurationError("privileged_mode_enabled")
    if host.get("ReadonlyRootfs") is not restrictions.read_only_root:
        raise DockerRuntimeConfigurationError("read_only_root_drift")
    if runtime is not None and host.get("Runtime") != runtime:
        raise DockerRuntimeConfigurationError("runtime_drift")
    if host.get("PidsLimit") != restrictions.pids_limit:
        raise DockerRuntimeConfigurationError("pids_limit_drift")
    if host.get("Memory") != restrictions.memory_bytes:
        raise DockerRuntimeConfigurationError("memory_limit_drift")
    if host.get("MemorySwap") != restrictions.memory_swap_bytes:
        raise DockerRuntimeConfigurationError("memory_swap_limit_drift")
    if host.get("CpuPeriod") != restrictions.cpu_period_us:
        raise DockerRuntimeConfigurationError("cpu_period_drift")
    if host.get("CpuQuota") != restrictions.cpu_quota_us:
        raise DockerRuntimeConfigurationError("cpu_quota_drift")
    if host.get("ShmSize") != restrictions.shm_size_bytes:
        raise DockerRuntimeConfigurationError("shm_size_drift")

    security_options = {
        value
        for value in _require_sequence(host.get("SecurityOpt"), "security_options_malformed")
        if type(value) is str
    }
    if restrictions.no_new_privileges and not security_options.intersection(
        {"no-new-privileges", "no-new-privileges=true"}
    ):
        raise DockerRuntimeConfigurationError("no_new_privileges_drift")
    if seccomp_profile is not None and f"seccomp={seccomp_profile}" not in security_options:
        raise DockerRuntimeConfigurationError("seccomp_profile_drift")

    cap_drop = {
        str(value).upper()
        for value in _require_sequence(host.get("CapDrop"), "capability_drop_malformed")
    }
    if cap_drop != {"ALL"}:
        raise DockerRuntimeConfigurationError("capability_drop_drift")
    cap_add = tuple(
        sorted(
            str(value).upper()
            for value in _require_sequence(host.get("CapAdd"), "capability_add_malformed")
        )
    )
    if cap_add != restrictions.capability_add:
        raise DockerRuntimeConfigurationError("capability_add_drift")

    expected_tmpfs = _strict_tmpfs_options(restrictions)
    raw_tmpfs = _require_mapping(host.get("Tmpfs"), "tmpfs_configuration_missing")
    if set(raw_tmpfs) != set(expected_tmpfs):
        raise DockerRuntimeConfigurationError("tmpfs_target_drift")
    for target, expected_options in expected_tmpfs.items():
        raw_options = raw_tmpfs.get(target)
        if type(raw_options) is not str:
            raise DockerRuntimeConfigurationError("tmpfs_options_malformed")
        if frozenset(raw_options.split(",")) != expected_options:
            raise DockerRuntimeConfigurationError("tmpfs_options_drift")

    if _require_sequence(host.get("Binds"), "bind_configuration_malformed"):
        raise DockerRuntimeConfigurationError("host_bind_mount_present")
    if _require_sequence(host.get("Devices"), "device_configuration_malformed"):
        raise DockerRuntimeConfigurationError("host_device_present")
    if _require_sequence(host.get("DeviceRequests"), "device_request_malformed"):
        raise DockerRuntimeConfigurationError("host_device_request_present")
    for mount in _require_sequence(inspection.get("Mounts"), "mounts_malformed"):
        mounted = _require_mapping(mount, "mount_entry_malformed")
        if mounted.get("Type") not in {None, "tmpfs"}:
            raise DockerRuntimeConfigurationError("host_mount_present")
        destination = mounted.get("Destination")
        if destination not in expected_tmpfs:
            raise DockerRuntimeConfigurationError("unexpected_mount_present")
    return image_id, image_reference


async def _inspect_strict_container(
    docker_path: str,
    container_id: str,
    *,
    docker_cli_env_allowlist: Sequence[str],
) -> Mapping[str, object]:
    result = await _run_docker(
        docker_path,
        ["inspect", "--format", "{{json .}}", container_id],
        docker_cli_env_allowlist=docker_cli_env_allowlist,
        timeout_s=30,
    )
    if result.exit_code != 0 or result.timed_out:
        raise DockerRuntimeConfigurationError("container_inspection_unavailable")
    try:
        decoded = json.loads(result.stdout)
    except (TypeError, ValueError):
        raise DockerRuntimeConfigurationError("container_inspection_malformed") from None
    return _require_mapping(decoded, "container_inspection_malformed")


async def _probe_strict_container(
    docker_path: str,
    container_id: str,
    *,
    restrictions: DockerWorkloadRestrictions,
    required_executables: tuple[str, ...],
    docker_cli_env_allowlist: Sequence[str],
) -> tuple[tuple[str, bool], ...]:
    identity = await _run_docker(
        docker_path,
        ["exec", container_id, "sh", "-c", 'printf \'%s:%s\' "$(id -u)" "$(id -g)"'],
        docker_cli_env_allowlist=docker_cli_env_allowlist,
        timeout_s=30,
    )
    if identity.exit_code != 0 or identity.timed_out or identity.stdout != restrictions.user:
        raise DockerRuntimeConfigurationError("nonroot_runtime_identity_drift")
    availability: list[tuple[str, bool]] = []
    for executable in required_executables:
        result = await _run_docker(
            docker_path,
            [
                "exec",
                container_id,
                "sh",
                "-c",
                f"command -v {shlex.quote(executable)} >/dev/null 2>&1",
            ],
            docker_cli_env_allowlist=docker_cli_env_allowlist,
            timeout_s=30,
        )
        availability.append((executable, result.exit_code == 0 and not result.timed_out))
    return tuple(availability)


def _build_docker_exec_argv(
    docker_path: str,
    name: str,
    command: ExecCommand,
    *,
    cwd: str,
    env_file: str | None,
    has_stdin: bool,
    pid_file: str,
    direct_process_supervisor: bool = False,
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
    if command.kind == "process" and direct_process_supervisor:
        if command.argv is None:
            raise ValueError("Process commands require argv.")
        argv += [
            "python3",
            "-c",
            _PYTHON_PROCESS_SUPERVISOR,
            pid_file,
            *command.argv,
        ]
        return argv
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


_PYTHON_PROCESS_SUPERVISOR = """\
import ctypes
import os
import signal
import sys
import time

state_path = sys.argv[1]
command = sys.argv[2:]
if not command:
    raise SystemExit(125)
PR_SET_CHILD_SUBREAPER = 36
if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
    raise SystemExit(125)
os.makedirs(os.path.dirname(state_path), mode=0o700, exist_ok=True)
child = os.fork()
if child == 0:
    try:
        os.setsid()
        os.execvp(command[0], command)
    except BaseException:
        os._exit(127)

def unlink_state():
    try:
        os.unlink(state_path)
    except FileNotFoundError:
        pass

def direct_children():
    task_path = f"/proc/{os.getpid()}/task/{os.getpid()}/children"
    try:
        with open(task_path, encoding="ascii") as handle:
            values = handle.read().split()
    except OSError:
        return None
    if any(not value.isdigit() for value in values):
        return None
    return tuple(int(value) for value in values)

def reap_available():
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return

def terminate_descendants(signum, _frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    try:
        os.killpg(child, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 1.0
    complete = False
    while True:
        reap_available()
        children = direct_children()
        if children is None:
            break
        if not children:
            complete = True
            break
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    if complete:
        unlink_state()
    os._exit(128 + signum if complete else 125)

signal.signal(signal.SIGTERM, terminate_descendants)
signal.signal(signal.SIGINT, terminate_descendants)
signal.signal(signal.SIGHUP, terminate_descendants)
try:
    descriptor = os.open(state_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()} 2\\n".encode("ascii"))
    finally:
        os.close(descriptor)
except BaseException:
    try:
        os.killpg(child, 9)
    except ProcessLookupError:
        pass
    os.waitpid(child, 0)
    raise
try:
    direct_status = None
    while True:
        try:
            pid, status = os.waitpid(-1, 0)
            if pid == child:
                direct_status = status
        except ChildProcessError:
            break
        except InterruptedError:
            continue
finally:
    unlink_state()
if direct_status is None:
    raise SystemExit(125)
if os.WIFEXITED(direct_status):
    raise SystemExit(os.WEXITSTATUS(direct_status))
if os.WIFSIGNALED(direct_status):
    raise SystemExit(128 + os.WTERMSIG(direct_status))
raise SystemExit(125)
"""


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
        'if test "$process_group" = 2; then '
        'kill -TERM "$pid" 2>/dev/null || true; '
        "attempts=0; "
        f'while test -f {quoted_pid_file} && test "$attempts" -lt 20; do '
        "attempts=$((attempts + 1)); sleep 0.1; "
        "done; "
        f"if ! test -f {quoted_pid_file}; then exit 0; fi; "
        'kill -KILL "$pid" 2>/dev/null || true; '
        "exit 1; "
        "fi; "
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
        return ("docker", self.container_id or self.name)

    @property
    def container_reference(self) -> str:
        """Return the exact ID when Cayu owns one, otherwise the legacy name."""

        return self.container_id or self.name

    def _execution_profile_material(self) -> dict[str, object] | None:
        """Return portable configuration when no deployment-local grants are present."""

        # Secret declarations, guest overlays, and host CLI environment grants
        # can carry or resolve deployment-local values. They must not become a
        # durable public hash oracle.
        if self.secret_env or self.env_overlay or self.docker_cli_env_allowlist:
            return None
        if self._runtime_evidence is not None:
            evidence = self._runtime_evidence
            return {
                "image_identity": evidence.image_identity.model_dump(mode="json"),
                "restrictions": evidence.restrictions.model_dump(mode="json"),
                "default_cwd": self.default_cwd,
                "close_action": self.close_action,
                "network_mode": evidence.network_mode,
                "runtime": evidence.runtime,
                "seccomp_profile_sha256": evidence.seccomp_profile_sha256,
                "credential_mode": self.credential_mode.value,
                "allow_raw_secret_env": self._allow_raw_secret_env,
                "cancel_timeout_s": self.cancel_timeout_s,
                "cancellation_cleanup": self.cancellation_cleanup,
                "timeout_cleanup": self.timeout_cleanup,
                "required_executables": list(evidence.required_executables),
                "process_transport": (
                    "python_subreaper_supervisor_v2"
                    if "python3" in evidence.required_executables
                    else "shell_supervisor"
                ),
            }
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
        _env_overlay_secret_values_present: bool | None = None,
        docker_cli_env_allowlist: Sequence[str] = (),
        image: str | None = None,
        _container_id: str | None = None,
        _runtime_evidence: _DockerRuntimeEvidence | None = None,
    ) -> None:
        self.name = require_clean_nonblank(name, "name")
        self.default_cwd = _validate_guest_cwd(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.docker_path = _require_docker(docker_path)
        self.image = None if image is None else require_clean_nonblank(image, "image")
        if _runtime_evidence is not None and not isinstance(
            _runtime_evidence,
            _DockerRuntimeEvidence,
        ):
            raise TypeError("_runtime_evidence must be Docker runtime evidence or None.")
        self._runtime_evidence = _runtime_evidence
        if _container_id is not None and (
            type(_container_id) is not str
            or _DOCKER_CONTAINER_ID_PATTERN.fullmatch(_container_id) is None
        ):
            raise ValueError("_container_id must be a full lowercase Docker container ID.")
        evidence_container_id = (
            None if _runtime_evidence is None else _runtime_evidence.container_id
        )
        if (
            _container_id is not None
            and evidence_container_id is not None
            and _container_id != evidence_container_id
        ):
            raise ValueError("Docker runtime evidence belongs to another container ID.")
        self.container_id = _container_id or evidence_container_id
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
        if (
            _env_overlay_secret_values_present is not None
            and type(_env_overlay_secret_values_present) is not bool
        ):
            raise TypeError("_env_overlay_secret_values_present must be bool or None.")
        self._env_overlay_secret_values_present = (
            bool(self.env_overlay)
            if _env_overlay_secret_values_present is None
            else _env_overlay_secret_values_present
        )
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
        _env_overlay_secret_values_present: bool | None = None,
        ca_mount: tuple[str, str] | None = None,
        seccomp_profile: str | None = None,
        docker_cli_env_allowlist: Sequence[str] = (),
        image_identity: DockerImageIdentity | None = None,
        workload_restrictions: DockerWorkloadRestrictions | None = None,
        required_executables: Sequence[str] = (),
        toolchain_profile_fingerprint: str | None = None,
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

        ``image_identity`` plus ``workload_restrictions`` selects the strict,
        evidence-bearing path. It requires an exact no-network, no-mount,
        credential-free container and ``replace=False``; the resulting runner
        retains the full container ID and verifies the live Docker inspection
        before it can be exposed.
        """
        docker = _require_docker(docker_path)
        name = require_clean_nonblank(name, "name")
        image = require_clean_nonblank(image, "image")
        if (image_identity is None) != (workload_restrictions is None):
            raise ValueError(
                "image_identity and workload_restrictions must be configured together."
            )
        strict_mode = image_identity is not None
        owned_image_identity = (
            None
            if image_identity is None
            else DockerImageIdentity.model_validate(image_identity.model_dump(mode="python"))
        )
        owned_restrictions = (
            None
            if workload_restrictions is None
            else DockerWorkloadRestrictions.model_validate(
                workload_restrictions.model_dump(mode="python")
            )
        )
        executable_requirements = _normalize_required_executables(required_executables)
        if toolchain_profile_fingerprint is not None and (
            type(toolchain_profile_fingerprint) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", toolchain_profile_fingerprint) is None
        ):
            raise ValueError(
                "toolchain_profile_fingerprint must be a lowercase SHA-256 identity or None."
            )
        if toolchain_profile_fingerprint is not None and not strict_mode:
            raise ValueError(
                "toolchain_profile_fingerprint requires strict evidence-bearing Docker creation."
            )
        if executable_requirements and not strict_mode:
            raise ValueError(
                "required_executables require image_identity and workload_restrictions."
            )
        if owned_image_identity is not None:
            if image not in {DEFAULT_DOCKER_IMAGE, owned_image_identity.reference}:
                raise ValueError("image must match image_identity.reference.")
            image = owned_image_identity.reference
        runtime = _validate_runtime(runtime)
        if mount_path is not None:
            mount_path = _validate_mount_path(mount_path)
        _validate_close_action(close_action)
        if type(replace) is not bool:
            raise TypeError("replace must be a bool.")
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
        if strict_mode:
            assert owned_image_identity is not None
            assert owned_restrictions is not None
            if replace:
                raise ValueError("Evidence-bearing Docker creation requires replace=False.")
            if mount_path is not None or ca_mount is not None:
                raise ValueError("Evidence-bearing Docker creation forbids host mounts.")
            if extra_hosts:
                raise ValueError("Evidence-bearing Docker creation forbids extra host mappings.")
            if network != "none":
                raise ValueError("Evidence-bearing Docker creation requires network='none'.")
            if setup_commands:
                raise ValueError(
                    "Evidence-bearing Docker creation requires tools baked into the image."
                )
            if validated_secret_env or env_overlay:
                raise ValueError(
                    "Evidence-bearing Docker creation forbids workload secret injection."
                )
            if mode is CredentialMode.RAW_ENV or allow_raw_secret_env:
                raise ValueError(
                    "Evidence-bearing Docker creation requires a non-readable credential mode."
                )
            if close_action != "remove":
                raise ValueError("Evidence-bearing Docker creation requires close_action='remove'.")
            if cancellation_policy != "sandbox" or timeout_policy != "sandbox":
                raise ValueError(
                    "Evidence-bearing Docker creation requires container-level cleanup."
                )
            if not any(
                default_cwd == mount.target
                or default_cwd.startswith(mount.target.rstrip("/") + "/")
                for mount in owned_restrictions.tmpfs
            ):
                raise ValueError(
                    "Evidence-bearing Docker default_cwd must be inside bounded tmpfs."
                )
        seccomp_profile_sha256 = _seccomp_profile_fingerprint(seccomp_profile)
        owned_container_id: str | None = None
        runtime_evidence: _DockerRuntimeEvidence | None = None
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
            if owned_restrictions is not None:
                run_argv.extend(owned_restrictions.run_args())
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
            if strict_mode:
                # Ignore image-authored ENTRYPOINT/CMD startup hooks. The exact
                # admitted command path starts only through later docker exec.
                run_argv += ["--entrypoint", "sleep", image, "infinity"]
            else:
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
            candidate_container_id = started.stdout.strip()
            if _DOCKER_CONTAINER_ID_PATTERN.fullmatch(candidate_container_id) is not None:
                owned_container_id = candidate_container_id
            elif strict_mode:
                raise DockerContainerOwnershipError(
                    "Docker did not return a full exact container ID; cleanup ownership is "
                    "ambiguous and no name-based cleanup was attempted."
                )
            container_reference = owned_container_id or name
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
                        container_reference,
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
                    setup_argv += [container_reference, "sh", "-c", cmd]
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
            if strict_mode:
                assert owned_container_id is not None
                assert owned_image_identity is not None
                assert owned_restrictions is not None
                inspection = await _inspect_strict_container(
                    docker,
                    owned_container_id,
                    docker_cli_env_allowlist=docker_cli_allowlist,
                )
                image_id, image_reference = _verify_strict_container_inspection(
                    inspection,
                    container_id=owned_container_id,
                    image_identity=owned_image_identity,
                    restrictions=owned_restrictions,
                    network_mode="none",
                    runtime=runtime,
                    seccomp_profile=seccomp_profile,
                )
                executable_availability = await _probe_strict_container(
                    docker,
                    owned_container_id,
                    restrictions=owned_restrictions,
                    required_executables=executable_requirements,
                    docker_cli_env_allowlist=docker_cli_allowlist,
                )
                observed_at = datetime.now(UTC)
                runtime_evidence = _DockerRuntimeEvidence(
                    container_id=owned_container_id,
                    image_id=image_id,
                    image_reference=image_reference,
                    network_mode="none",
                    default_cwd=default_cwd,
                    runtime=runtime,
                    seccomp_profile_sha256=seccomp_profile_sha256,
                    restrictions=owned_restrictions,
                    image_identity=owned_image_identity,
                    toolchain_profile_fingerprint=toolchain_profile_fingerprint,
                    required_executables=executable_requirements,
                    executable_availability=executable_availability,
                    observed_at=observed_at,
                    valid_until=observed_at + timedelta(seconds=300),
                )
        except BaseException as create_error:
            cleanup_reference = owned_container_id or (None if strict_mode else name)
            if cleanup_reference is not None:
                cleanup = await _run_docker(
                    docker,
                    ["rm", "-f", cleanup_reference],
                    docker_cli_env_allowlist=docker_cli_allowlist,
                )
                if cleanup.exit_code != 0 or cleanup.timed_out:
                    cleanup_error = DockerContainerOwnershipError(
                        "Docker creation failed and exact-container cleanup was not confirmed."
                    )
                    raise BaseExceptionGroup(
                        "Docker creation and exact-container cleanup both failed.",
                        [create_error, cleanup_error],
                    ) from None
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
            _env_overlay_secret_values_present=_env_overlay_secret_values_present,
            docker_cli_env_allowlist=docker_cli_allowlist,
            _container_id=owned_container_id,
            _runtime_evidence=runtime_evidence,
        )

    def execution_capability_evidence(self):
        """Describe Docker honestly and bind strict evidence to the exact container."""

        from cayu.environments.admission import (
            ExecutionCapabilityClaim,
            ExecutionCapabilityEvidence,
            ExecutionExecutableEvidence,
            ExecutionToolRequirementEvidence,
        )

        evidence = self._runtime_evidence
        if evidence is None:
            unsupported = {
                "untrusted_code_isolation": "docker_untrusted_isolation_unsupported",
                "real_credential_non_possession": "docker_credential_boundary_unverified",
                "deny_by_default_network": "docker_network_boundary_unverified",
                "brokered_egress": "docker_brokered_egress_unverified",
                "guest_privilege_containment": "docker_privilege_boundary_unverified",
                "unprivileged_guest": "docker_privilege_boundary_unverified",
                "host_filesystem_isolation": "docker_host_filesystem_unverified",
                "read_only_host_inputs": "docker_host_filesystem_unverified",
                "reconnect": "docker_reconnect_unsupported",
            }
            return ExecutionCapabilityEvidence(
                subject="docker",
                claims=(
                    ExecutionCapabilityClaim.available("confirmed_cancellation"),
                    ExecutionCapabilityClaim.available("confirmed_cleanup"),
                    *(
                        ExecutionCapabilityClaim.unsupported(
                            capability,
                            reason_code=reason_code,
                            remediation_code=(
                                "select_reconnectable_execution"
                                if capability == "reconnect"
                                else "use_verified_docker_restrictions"
                            ),
                        )
                        for capability, reason_code in unsupported.items()
                    ),
                ),
            )

        def live_claim(
            capability: str,
            *,
            observation: Literal["denied", "supported"],
            details: tuple[CapabilityDetail, ...] = (),
        ) -> ExecutionCapabilityClaim:
            return ExecutionCapabilityClaim(
                capability=capability,
                state="live_verified",
                proof_source="runtime_preflight",
                observation=observation,
                observed_at=evidence.observed_at,
                valid_until=evidence.valid_until,
                adapter_details=details,
            )

        restrictions = evidence.restrictions
        privilege_details = (
            CapabilityDetail(name="nonroot", value=True),
            CapabilityDetail(name="uid", value=restrictions.uid),
            CapabilityDetail(name="gid", value=restrictions.gid),
            CapabilityDetail(
                name="read_only_root",
                value=restrictions.read_only_root,
            ),
            CapabilityDetail(
                name="no_new_privileges",
                value=restrictions.no_new_privileges,
            ),
            CapabilityDetail(name="cap_drop_all", value=True),
            CapabilityDetail(
                name="capability_add_empty",
                value=not restrictions.capability_add,
            ),
            CapabilityDetail(name="pids_limit", value=restrictions.pids_limit),
            CapabilityDetail(name="memory_bytes", value=restrictions.memory_bytes),
            CapabilityDetail(
                name="memory_swap_bytes",
                value=restrictions.memory_swap_bytes,
            ),
            CapabilityDetail(name="cpu_period_us", value=restrictions.cpu_period_us),
            CapabilityDetail(name="cpu_quota_us", value=restrictions.cpu_quota_us),
            CapabilityDetail(name="shm_size_bytes", value=restrictions.shm_size_bytes),
        )
        if restrictions.supports_strict_privilege_evidence:
            privilege_claims = (
                live_claim(
                    "guest_privilege_containment",
                    observation="supported",
                    details=privilege_details,
                ),
                live_claim(
                    "unprivileged_guest",
                    observation="supported",
                    details=privilege_details,
                ),
            )
        else:
            privilege_claims = (
                ExecutionCapabilityClaim.unsupported(
                    "guest_privilege_containment",
                    reason_code="docker_privilege_restrictions_weakened",
                    remediation_code="use_verified_docker_restrictions",
                ),
                ExecutionCapabilityClaim.unsupported(
                    "unprivileged_guest",
                    reason_code="docker_privilege_restrictions_weakened",
                    remediation_code="use_verified_docker_restrictions",
                ),
            )
        unsupported = (
            ExecutionCapabilityClaim.unsupported(
                "untrusted_code_isolation",
                reason_code="docker_untrusted_isolation_unsupported",
                remediation_code="select_untrusted_isolation",
            ),
            ExecutionCapabilityClaim.unsupported(
                "brokered_egress",
                reason_code="docker_network_disabled",
                remediation_code="select_brokered_egress",
            ),
            ExecutionCapabilityClaim.unsupported(
                "read_only_host_inputs",
                reason_code="docker_host_inputs_not_mounted",
                remediation_code="use_workspace_sync",
            ),
            ExecutionCapabilityClaim.unsupported(
                "reconnect",
                reason_code="docker_reconnect_unsupported",
                remediation_code="select_reconnectable_execution",
            ),
        )
        executable_availability = dict(evidence.executable_availability)
        tool_requirements = ExecutionToolRequirementEvidence(
            environment_fingerprint=evidence.environment_fingerprint,
            image_fingerprint=evidence.image_fingerprint,
            executables=tuple(
                ExecutionExecutableEvidence(
                    executable=executable,
                    state=(
                        "live_verified" if executable_availability[executable] else "unavailable"
                    ),
                    observed_at=(
                        evidence.observed_at if executable_availability[executable] else None
                    ),
                    valid_until=(
                        evidence.valid_until if executable_availability[executable] else None
                    ),
                    reason_code=(
                        None if executable_availability[executable] else "executable_unavailable"
                    ),
                    remediation_code=(
                        None if executable_availability[executable] else "rebuild_trusted_image"
                    ),
                )
                for executable in evidence.required_executables
            ),
        )
        return ExecutionCapabilityEvidence(
            subject="docker",
            environment_fingerprint=evidence.environment_fingerprint,
            image_fingerprint=evidence.image_fingerprint,
            toolchain_profile_fingerprint=evidence.toolchain_profile_fingerprint,
            claims=(
                live_claim("real_credential_non_possession", observation="supported"),
                live_claim(
                    "deny_by_default_network",
                    observation="denied",
                    details=(CapabilityDetail(name="network_none", value=True),),
                ),
                *privilege_claims,
                live_claim(
                    "host_filesystem_isolation",
                    observation="supported",
                    details=(
                        CapabilityDetail(name="host_mount_count", value=0),
                        CapabilityDetail(name="cwd_bounded_tmpfs", value=True),
                    ),
                ),
                live_claim(
                    "confirmed_cancellation",
                    observation="supported",
                    details=(
                        CapabilityDetail(name="container_cleanup", value=True),
                        CapabilityDetail(name="exact_container_owner", value=True),
                    ),
                ),
                live_claim(
                    "confirmed_cleanup",
                    observation="supported",
                    details=(
                        CapabilityDetail(name="close_remove", value=True),
                        CapabilityDetail(name="exact_container_owner", value=True),
                    ),
                ),
                *unsupported,
            ),
            tool_requirements=tool_requirements,
        )

    def execution_admission_candidate(self):
        """Return capability evidence for this exact Docker execution target."""

        from cayu.environments.admission import ExecutionAdmissionCandidate

        return ExecutionAdmissionCandidate(
            candidate="docker",
            evidence=self.execution_capability_evidence(),
        )

    def workload_authority(self, name: str):
        """Declare a shipped workload only for its exact selected image."""

        if (
            name == BROWSER_FETCH_WORKLOAD_NAME
            and self.image == PINNED_BROWSER_FETCH_WORKLOAD.image
        ):
            return PINNED_BROWSER_FETCH_WORKLOAD
        if (
            name == BROWSER_SESSION_WORKLOAD_NAME
            and self.image == PINNED_BROWSER_SESSION_WORKLOAD.image
        ):
            return PINNED_BROWSER_SESSION_WORKLOAD
        return None

    def output_secret_values_present(self) -> bool:
        """Declare whether Docker resolves runner-owned secret environment values."""

        return bool(self.secret_env) or self._env_overlay_secret_values_present

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
            name=self.container_reference,
            pid_file=pid_file,
            docker_cli_env_allowlist=self.docker_cli_env_allowlist,
        )
        with runner_env_file(environment) as env_file:
            environment = {}
            argv = _build_docker_exec_argv(
                self.docker_path,
                self.container_reference,
                owned_command,
                cwd=working_dir,
                env_file=env_file,
                has_stdin=standard_input is not None,
                pid_file=pid_file,
                direct_process_supervisor=(
                    self._runtime_evidence is not None
                    and "python3" in self._runtime_evidence.required_executables
                    and owned_command.kind == "process"
                ),
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

    async def fence_guest_processes_for_egress_cutover(self) -> None:
        """Terminate detached guest work while retaining container/workspace identity."""

        self._ensure_exec_open()
        result = await _run_docker(
            self.docker_path,
            [
                "exec",
                "-u",
                "root",
                self.container_reference,
                "python3",
                "-c",
                _DOCKER_EGRESS_CUTOVER_FENCE_SCRIPT,
            ],
            docker_cli_env_allowlist=self.docker_cli_env_allowlist,
            timeout_s=10,
        )
        if result.exit_code != 0 or result.timed_out:
            raise RuntimeError("Docker guest processes could not be fenced for egress cutover.")

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
            ["rm", "-f", self.container_reference],
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
                f"docker rm failed for the owned container (exit {result.exit_code}): {detail}"
            )

    async def _stop_container(self) -> None:
        result = await _run_docker(
            self.docker_path,
            ["stop", self.container_reference],
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
                f"docker stop failed for the owned container (exit {result.exit_code}): {detail}"
            )
