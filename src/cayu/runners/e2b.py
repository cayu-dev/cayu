from __future__ import annotations

import asyncio
import contextlib
import importlib
import posixpath
import shlex
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import ModuleType
from typing import Any, Literal, cast
from uuid import uuid4

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
DEFAULT_E2B_HANDOFF_TIMEOUT_SECONDS = 120.0
DEFAULT_E2B_HANDOFF_CLEANUP_TIMEOUT_SECONDS = 15.0
DEFAULT_E2B_PROTECTED_FILE_MAX_BYTES = 64 * 1024 * 1024
_E2B_GUEST_HANDOFF_METADATA_KEY = "cayu_guest_handoff_id"
_E2B_CONNECTION_OPTION_KEYS = frozenset(
    {
        "api_headers",
        "api_key",
        "api_url",
        "debug",
        "domain",
        "headers",
        "proxy",
        "request_timeout",
        "sandbox_url",
    }
)
_E2B_PROVISIONER_CONSTRUCTION_TOKEN = object()
_E2B_HANDOFF_COMMAND_ENV = {
    "BASH_ENV": "/dev/null",
    "ENV": "/dev/null",
    "IFS": " \t\n",
    "LD_LIBRARY_PATH": "",
    "LD_PRELOAD": "",
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
    "PYTHONHOME": "",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "",
    "PYTHONSAFEPATH": "1",
}

E2BCloseAction = Literal["kill", "detach", "none"]
E2BGuestHandoffPhase = Literal[
    "hardening",
    "bootstrap",
    "guest_setup",
    "guest_probe",
    "verification",
]

_HARDEN_GUEST_SCRIPT = r"""
set -eu
test "$(id -u)" -eq 0
test -x /usr/sbin/iptables
guest="$CAYU_GUEST_USER"
id "$guest" >/dev/null
/usr/sbin/iptables -I OUTPUT 1 -d 169.254.169.254/32 -j REJECT
for group in sudo wheel; do
  if id -nG "$guest" | tr ' ' '\n' | grep -qx "$group"; then
    gpasswd -d "$guest" "$group"
  fi
done
for path in /usr/bin/sudo /bin/su /usr/bin/su; do
  if [ -e "$path" ]; then
    chown root:root "$path"
    chmod 0700 "$path"
  fi
done
""".strip()

_VERIFY_GUEST_HANDOFF_SCRIPT = r"""
set -u
if [ "$(id -u)" -eq 0 ]; then
  exit 41
fi
if [ "$(id -un)" != "$CAYU_GUEST_USER" ]; then
  exit 46
fi
if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  exit 42
fi
if command -v su >/dev/null 2>&1 && su -c true root >/dev/null 2>&1; then
  exit 43
fi
if /usr/sbin/iptables -D OUTPUT -d 169.254.169.254/32 -j REJECT >/dev/null 2>&1; then
  exit 44
fi
python3 - <<'PY'
import socket
try:
    connection = socket.create_connection(("169.254.169.254", 80), timeout=1)
except OSError:
    raise SystemExit(0)
connection.close()
raise SystemExit(45)
PY
""".strip()

_INSTALL_PROTECTED_DIRECTORY_SCRIPT = r"""
import os
import stat

path = os.environ["CAYU_PROTECTED_PATH"]
mode = int(os.environ["CAYU_PROTECTED_MODE"], 8)
parts = [part for part in path.split("/") if part]
current = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
try:
    for index, part in enumerate(parts):
        final = index == len(parts) - 1
        if final:
            os.mkdir(part, mode, dir_fd=current)
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
        else:
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current,
                )
            except FileNotFoundError:
                os.mkdir(part, 0o755, dir_fd=current)
                child = os.open(
                    part,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=current,
                )
        info = os.fstat(child)
        if info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
            raise PermissionError("protected path has a guest-writable or non-root ancestor")
        if final:
            os.fchown(child, 0, 0)
            os.fchmod(child, mode)
        os.close(current)
        current = child
finally:
    os.close(current)
""".strip()

_INSTALL_PROTECTED_FILE_SCRIPT = r"""
import os
import stat

path = os.environ["CAYU_PROTECTED_PATH"]
stage = os.environ["CAYU_PROTECTED_STAGE"]
mode = int(os.environ["CAYU_PROTECTED_MODE"], 8)
parts = [part for part in path.split("/") if part]
current = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
try:
    for part in parts[:-1]:
        try:
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
        except FileNotFoundError:
            os.mkdir(part, 0o755, dir_fd=current)
            child = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current,
            )
        info = os.fstat(child)
        if info.st_uid != 0 or stat.S_IMODE(info.st_mode) & 0o022:
            raise PermissionError("protected path has a guest-writable or non-root ancestor")
        os.close(current)
        current = child
    source = os.open(stage, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        target = os.open(
            parts[-1],
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            mode,
            dir_fd=current,
        )
        try:
            while chunk := os.read(source, 1024 * 1024):
                view = memoryview(chunk)
                while view:
                    written = os.write(target, view)
                    view = view[written:]
            os.fchown(target, 0, 0)
            os.fchmod(target, mode)
            os.fsync(target)
        finally:
            os.close(target)
    finally:
        os.close(source)
finally:
    os.close(current)
    try:
        os.unlink(stage)
    except FileNotFoundError:
        pass
""".strip()

_VERIFY_PROTECTED_ASSET_MUTATION_SCRIPT = r"""
set -u
path="$CAYU_PROTECTED_PATH"
kind="$CAYU_PROTECTED_KIND"
tmp="$(mktemp /tmp/cayu-protected.XXXXXX)"
if [ "$kind" = "file" ]; then
  if printf x >>"$path" 2>/dev/null; then exit 53; fi
  if rm -f -- "$path" 2>/dev/null; then exit 54; fi
  if mv -- "$path" "$path.cayu-moved" 2>/dev/null; then exit 55; fi
  if mv -f -- "$tmp" "$path" 2>/dev/null; then exit 56; fi
else
  if touch -- "$path/cayu-write-probe" 2>/dev/null; then exit 57; fi
  if rmdir -- "$path" 2>/dev/null; then exit 58; fi
  if mv -- "$path" "$path.cayu-moved" 2>/dev/null; then exit 59; fi
fi
rm -f -- "$tmp"
""".strip()

_VERIFY_PROTECTED_ASSET_METADATA_SCRIPT = r"""
set -u
path="$CAYU_PROTECTED_PATH"
expected_mode="$CAYU_PROTECTED_MODE"
test "$(stat -c %u -- "$path")" = "0" || exit 51
test "$(stat -c %a -- "$path")" = "$expected_mode" || exit 52
""".strip()


class E2BGuestHandoffError(RuntimeError):
    """A secret-safe failure while establishing the E2B guest boundary."""

    def __init__(
        self,
        *,
        phase: E2BGuestHandoffPhase,
        reason: str,
        exit_code: int | None = None,
    ) -> None:
        self.phase = phase
        self.reason = require_clean_nonblank(reason, "reason")
        self.exit_code = exit_code
        detail = f" (exit_code={exit_code})" if exit_code is not None else ""
        super().__init__(f"E2B guest handoff {phase} failed: {self.reason}{detail}.")


class _E2BHandoffRollbackCancelledError(RuntimeError):
    """Rollback stopped itself without a supervising cancellation request."""


@dataclass(frozen=True)
class _E2BProtectedAsset:
    path: str
    mode: int
    kind: Literal["file", "directory"]


class E2BGuestProvisioner:
    """Short-lived typed root capability issued during trusted E2B bootstrap.

    Instances are created only by :meth:`E2BRunner.create_hardened` and are
    sealed before guest callbacks run. The capability intentionally exposes no
    arbitrary root command or native filesystem handle.
    """

    def __init__(
        self,
        sandbox: Any,
        *,
        default_cwd: str,
        guest_user: str,
        operation_timeout_s: float,
        max_file_bytes: int,
        _construction_token: object | None = None,
    ) -> None:
        if _construction_token is not _E2B_PROVISIONER_CONSTRUCTION_TOKEN:
            raise TypeError("E2BGuestProvisioner is issued only by E2BRunner.create_hardened().")
        self._sandbox = sandbox
        self._default_cwd = default_cwd
        self._guest_user = guest_user
        self._operation_timeout_s = operation_timeout_s
        self._max_file_bytes = max_file_bytes
        self._lock = asyncio.Lock()
        self._sealed = False
        self._assets: list[_E2BProtectedAsset] = []
        self._staging_root = f"/root/.cayu-handoff-{uuid4().hex}"

    @classmethod
    def _create(
        cls,
        sandbox: Any,
        *,
        default_cwd: str,
        guest_user: str,
        operation_timeout_s: float,
        max_file_bytes: int,
    ) -> E2BGuestProvisioner:
        return cls(
            sandbox,
            default_cwd=default_cwd,
            guest_user=guest_user,
            operation_timeout_s=operation_timeout_s,
            max_file_bytes=max_file_bytes,
            _construction_token=_E2B_PROVISIONER_CONSTRUCTION_TOKEN,
        )

    @property
    def is_sealed(self) -> bool:
        return self._sealed

    async def install_directory(self, path: str, *, mode: int = 0o755) -> None:
        """Create one protected root-owned directory and register its verification."""

        protected_path = _validate_protected_path(
            path,
            default_cwd=self._default_cwd,
            guest_user=self._guest_user,
        )
        protected_mode = _validate_protected_mode(mode, directory=True)
        async with self._lock:
            self._ensure_active()
            self._ensure_unregistered_path(protected_path)
            await _run_handoff_command(
                self._sandbox.commands.run(
                    f"python3 -c {shlex.quote(_INSTALL_PROTECTED_DIRECTORY_SCRIPT)}",
                    user="root",
                    envs=_handoff_command_env(
                        CAYU_PROTECTED_PATH=protected_path,
                        CAYU_PROTECTED_MODE=f"{protected_mode:o}",
                    ),
                    timeout=self._operation_timeout_s,
                ),
                phase="bootstrap",
                reason="protected directory installation",
            )
            self._ensure_active()
            self._assets.append(
                _E2BProtectedAsset(
                    path=protected_path,
                    mode=protected_mode,
                    kind="directory",
                )
            )

    async def install_file(
        self,
        path: str,
        content: str | bytes,
        *,
        mode: int = 0o444,
    ) -> None:
        """Install one protected root-owned file and register its verification."""

        protected_path = _validate_protected_path(
            path,
            default_cwd=self._default_cwd,
            guest_user=self._guest_user,
        )
        protected_mode = _validate_protected_mode(mode, directory=False)
        if type(content) is str:
            payload = content.encode("utf-8")
        elif type(content) is bytes:
            payload = bytes(content)
        else:
            raise TypeError("E2B protected file content must be str or bytes.")
        if len(payload) > self._max_file_bytes:
            raise ValueError(f"E2B protected file content exceeds {self._max_file_bytes} bytes.")
        async with self._lock:
            self._ensure_active()
            self._ensure_unregistered_path(protected_path)
            stage_path = f"{self._staging_root}/{uuid4().hex}"
            await _run_handoff_command(
                self._sandbox.commands.run(
                    f"/bin/mkdir -p {shlex.quote(self._staging_root)}"
                    f" && /bin/chmod 0700 {shlex.quote(self._staging_root)}",
                    user="root",
                    envs=_handoff_command_env(),
                    timeout=self._operation_timeout_s,
                ),
                phase="bootstrap",
                reason="protected file staging",
            )
            self._ensure_active()
            try:
                await self._sandbox.files.write(
                    stage_path,
                    payload,
                    user="root",
                    request_timeout=self._operation_timeout_s,
                )
            except Exception:
                raise E2BGuestHandoffError(
                    phase="bootstrap",
                    reason="protected file transfer",
                ) from None
            self._ensure_active()
            await _run_handoff_command(
                self._sandbox.commands.run(
                    f"python3 -c {shlex.quote(_INSTALL_PROTECTED_FILE_SCRIPT)}",
                    user="root",
                    envs=_handoff_command_env(
                        CAYU_PROTECTED_PATH=protected_path,
                        CAYU_PROTECTED_STAGE=stage_path,
                        CAYU_PROTECTED_MODE=f"{protected_mode:o}",
                    ),
                    timeout=self._operation_timeout_s,
                ),
                phase="bootstrap",
                reason="protected file installation",
            )
            self._ensure_active()
            self._assets.append(
                _E2BProtectedAsset(
                    path=protected_path,
                    mode=protected_mode,
                    kind="file",
                )
            )

    async def _seal(self) -> tuple[_E2BProtectedAsset, ...]:
        async with self._lock:
            self._sealed = True
            return tuple(self._assets)

    def _invalidate(self) -> None:
        self._sealed = True

    def _ensure_active(self) -> None:
        if self._sealed:
            raise RuntimeError("E2B guest provisioning capability is sealed.")

    def _ensure_unregistered_path(self, path: str) -> None:
        if any(asset.path == path for asset in self._assets):
            raise ValueError(f"E2B protected path {path!r} is already registered.")


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
        resolved_user = self._runner._workspace_filesystem_user(user)
        entry = await self._runner._native_filesystem().get_info(
            path,
            user=resolved_user,
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
        resolved_user = self._runner._workspace_filesystem_user(user)
        entries = await self._runner._native_filesystem().list(
            path,
            depth=depth,
            user=resolved_user,
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
        self._hardened_guest_user: str | None = None

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
    async def create_hardened(
        cls,
        *,
        template: str | None = None,
        sandbox_timeout_s: int | None = None,
        default_cwd: str = DEFAULT_E2B_CWD,
        close_action: E2BCloseAction = "kill",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        metadata: dict[str, str] | None = None,
        env_overlay: Mapping[str, str] | None = None,
        guest_user: str = "user",
        network: Any | None = None,
        lifecycle: Any | None = None,
        volume_mounts: Any | None = None,
        handoff_timeout_s: float = DEFAULT_E2B_HANDOFF_TIMEOUT_SECONDS,
        cleanup_timeout_s: float = DEFAULT_E2B_HANDOFF_CLEANUP_TIMEOUT_SECONDS,
        max_protected_file_bytes: int = DEFAULT_E2B_PROTECTED_FILE_MAX_BYTES,
        bootstrap: Callable[[E2BGuestProvisioner], Awaitable[None]] | None = None,
        guest_setup: Callable[[E2BRunner], Awaitable[None]] | None = None,
        guest_probe: Callable[[E2BRunner], Awaitable[None]] | None = None,
        e2b_module: ModuleType | Any | None = None,
        **api_options: Any,
    ) -> E2BRunner:
        """Create, harden, verify, and complete a one-way capability handoff.

        ``bootstrap`` receives the only public privileged capability. It may
        install protected files or directories, then is sealed before
        ``guest_setup`` and ``guest_probe`` run through the guest-pinned runner.
        The complete create/handoff sequence is bounded by ``handoff_timeout_s``.
        Any failure performs bounded sandbox rollback; an ambiguous create is
        reconciled by a unique provider metadata marker.

        Public Internet access is always disabled. ``network`` remains available
        for explicit provider-native allowlists, such as the one proxy endpoint
        used by Cayu's virtual-egress adapter.
        """

        resolved_guest = _validate_hardened_guest_user(guest_user)
        handoff_timeout = _validate_handoff_timeout(handoff_timeout_s, "handoff_timeout_s")
        cleanup_timeout = _validate_handoff_timeout(cleanup_timeout_s, "cleanup_timeout_s")
        max_file_bytes = _validate_max_protected_file_bytes(max_protected_file_bytes)
        guest_root = _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
        validate_cancel_timeout(cancel_timeout_s)
        validate_runner_cleanup_policy(cancellation_cleanup, "cancellation_cleanup")
        validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        _validate_sandbox_timeout(sandbox_timeout_s)
        _validate_handoff_callback(bootstrap, "bootstrap")
        _validate_handoff_callback(guest_setup, "guest_setup")
        _validate_handoff_callback(guest_probe, "guest_probe")
        reserved_api_options = sorted(
            {
                "allow_internet_access",
                "ensure_default_cwd",
                "envs",
                "exec_user",
                "secure",
            }.intersection(api_options)
        )
        if reserved_api_options:
            raise ValueError(
                "E2B create_hardened owns provider options: " + ", ".join(reserved_api_options)
            )
        supplied_metadata = _copy_string_dict(metadata, "metadata") or {}
        if _E2B_GUEST_HANDOFF_METADATA_KEY in supplied_metadata:
            raise ValueError(f"E2B metadata key {_E2B_GUEST_HANDOFF_METADATA_KEY!r} is Cayu-owned.")
        handoff_id = uuid4().hex
        supplied_metadata[_E2B_GUEST_HANDOFF_METADATA_KEY] = handoff_id
        module = _e2b_module(e2b_module)
        connection_options = {
            key: value for key, value in api_options.items() if key in _E2B_CONNECTION_OPTION_KEYS
        }

        runner: E2BRunner | None = None
        provisioner: E2BGuestProvisioner | None = None
        finished_at: float | None = None
        handoff_revoked = False
        loop = asyncio.get_running_loop()
        deadline = loop.time() + handoff_timeout

        async def perform_handoff() -> E2BRunner:
            nonlocal runner, provisioner, finished_at
            try:
                runner = await cls.create(
                    template=template,
                    sandbox_timeout_s=sandbox_timeout_s,
                    default_cwd=guest_root,
                    close_action=close_action,
                    cancel_timeout_s=cancel_timeout_s,
                    cancellation_cleanup=cancellation_cleanup,
                    timeout_cleanup=timeout_cleanup,
                    ensure_default_cwd=False,
                    metadata=supplied_metadata,
                    env_overlay=env_overlay,
                    exec_user=resolved_guest,
                    secure=True,
                    allow_internet_access=False,
                    network=network,
                    lifecycle=lifecycle,
                    volume_mounts=volume_mounts,
                    e2b_module=module,
                    **api_options,
                )
                if handoff_revoked:
                    # Allocation acknowledgement arrived after the supervising
                    # caller had already timed out or been cancelled. Never run
                    # a sandbox command or publish this late-owned resource.
                    runner._closed = True
                    await _kill_late_handoff_runner(
                        runner,
                        timeout_s=cleanup_timeout,
                    )
                    raise asyncio.CancelledError
                # Pin every guest-facing command and filesystem operation as
                # soon as allocation completes. The runner is never exposed
                # with a mutable user-selection gap between bootstrap phases.
                runner._hardened_guest_user = resolved_guest
                await _run_handoff_command(
                    runner._sandbox.commands.run(
                        f"/bin/mkdir -p {shlex.quote(runner.default_cwd)}",
                        cwd="/",
                        user=resolved_guest,
                        envs=_handoff_command_env(),
                        timeout=handoff_timeout,
                    ),
                    phase="hardening",
                    reason="guest workspace setup",
                )
                provisioner = E2BGuestProvisioner._create(
                    runner._sandbox,
                    default_cwd=runner.default_cwd,
                    guest_user=resolved_guest,
                    operation_timeout_s=handoff_timeout,
                    max_file_bytes=max_file_bytes,
                )
                await _run_handoff_command(
                    runner._sandbox.commands.run(
                        _HARDEN_GUEST_SCRIPT,
                        user="root",
                        envs=_handoff_command_env(CAYU_GUEST_USER=resolved_guest),
                        timeout=handoff_timeout,
                    ),
                    phase="hardening",
                    reason="metadata and privilege hardening",
                )
                if bootstrap is not None:
                    await bootstrap(provisioner)
                protected_assets = await provisioner._seal()
                # Prove the privilege, metadata, and protected-asset boundary
                # before exposing the runner to any guest-controlled callback.
                # A successful hardening command is not itself proof that all
                # postconditions hold in the selected template.
                await _verify_guest_handoff(
                    runner,
                    protected_assets=protected_assets,
                    timeout_s=handoff_timeout,
                )
                if guest_setup is not None:
                    await guest_setup(runner)
                if guest_probe is not None:
                    await guest_probe(runner)
                # Guest callbacks must not weaken the established boundary.
                # Repeat the same proof before publishing the runner.
                await _verify_guest_handoff(
                    runner,
                    protected_assets=protected_assets,
                    timeout_s=handoff_timeout,
                )
                return runner
            finally:
                finished_at = loop.time()

        handoff_task = asyncio.create_task(
            perform_handoff(),
            name=f"cayu-e2b-handoff-{handoff_id}",
        )
        try:
            done, _ = await asyncio.wait({handoff_task}, timeout=handoff_timeout)
            if handoff_task in done and finished_at is not None and finished_at <= deadline:
                return handoff_task.result()
            raise TimeoutError("E2B guest handoff timed out.")
        except BaseException as original:
            handoff_revoked = True
            failure = original
            if (
                handoff_task.done()
                and handoff_task.cancelling() > 0
                and not _contains_cancellation(original)
            ):
                # A callback can request cancellation and synchronously raise a
                # different primary failure before cancellation is injected at
                # an await. Preserve both signals across the supervision task.
                failure = BaseExceptionGroup(
                    "E2B guest handoff failed with pending cancellation.",
                    [original, asyncio.CancelledError()],
                )
            if provisioner is not None:
                provisioner._invalidate()
            if runner is not None:
                # Revoke guest-facing capabilities before cancellation can
                # wake a non-cooperative callback during rollback.
                runner._closed = True
            if not handoff_task.done():
                handoff_task.cancel()
            handoff_task.add_done_callback(_consume_task_outcome)
            await _cleanup_handoff_failure(
                module=module,
                runner=runner,
                handoff_id=handoff_id,
                connection_options=connection_options,
                original_error=failure,
                timeout_s=cleanup_timeout,
            )
            if failure is not original:
                raise failure from None
            raise

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
        if self._hardened_guest_user is not None:
            raise RuntimeError(
                "Raw E2B filesystem access is unavailable after guest handoff; use E2BWorkspace."
            )
        return self._sandbox.files

    def _native_filesystem(self) -> Any:
        if self._closed:
            raise RuntimeError("E2BRunner is closed.")
        return self._sandbox.files

    def _workspace_filesystem_user(self, user: str | None) -> str | None:
        if self._hardened_guest_user is None:
            return user
        if user is not None and user != self._hardened_guest_user:
            raise ValueError("Hardened E2B workspace operations are pinned to the guest user.")
        return self._hardened_guest_user

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
            execution_user = self._hardened_guest_user or self.exec_user
            if execution_user is not None:
                run_options["user"] = execution_user
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


def _validate_hardened_guest_user(user: str) -> str:
    value = require_clean_nonblank(user, "guest_user")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("E2B hardened guest_user cannot contain control characters.")
    if "/" in value or ":" in value:
        raise ValueError("E2B hardened guest_user must be a guest account name.")
    if value == "root" or (value.isdecimal() and int(value) == 0):
        raise ValueError("E2B hardened guest_user must not be root.")
    return value


def _validate_handoff_timeout(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"E2B {field_name} must be numeric.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"E2B {field_name} must be finite and greater than zero.")
    return float(value)


def _validate_max_protected_file_bytes(value: int) -> int:
    if type(value) is not int:
        raise TypeError("E2B max_protected_file_bytes must be an integer.")
    if value <= 0:
        raise ValueError("E2B max_protected_file_bytes must be greater than zero.")
    return value


def _validate_handoff_callback(value: Any, field_name: str) -> None:
    if value is not None and not callable(value):
        raise TypeError(f"E2B {field_name} must be an async callable or None.")


def _handoff_command_env(**values: str) -> dict[str, str]:
    environment = dict(_E2B_HANDOFF_COMMAND_ENV)
    environment.update(values)
    return environment


def _validate_protected_path(
    path: str,
    *,
    default_cwd: str,
    guest_user: str,
) -> str:
    value = require_clean_nonblank(path, "protected path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("E2B protected path cannot contain control characters.")
    if not posixpath.isabs(value):
        raise ValueError("E2B protected path must be absolute.")
    normalized = posixpath.normpath(value)
    if normalized != value or normalized == "/":
        raise ValueError("E2B protected path must be normalized and cannot be root.")
    unsafe_roots = (
        default_cwd,
        f"/home/{guest_user}",
        "/tmp",
        "/var/tmp",
        "/dev",
        "/proc",
        "/sys",
        "/run",
    )
    if any(
        normalized == root or normalized.startswith(f"{root.rstrip('/')}/") for root in unsafe_roots
    ):
        raise ValueError("E2B protected path is inside a guest-writable or special path.")
    return normalized


def _validate_protected_mode(mode: int, *, directory: bool) -> int:
    if type(mode) is not int:
        raise TypeError("E2B protected mode must be an integer.")
    if mode < 0 or mode > 0o777:
        raise ValueError("E2B protected mode must contain only ordinary permission bits.")
    if mode & 0o022:
        raise ValueError("E2B protected mode cannot be group- or other-writable.")
    if directory and not mode & 0o100:
        raise ValueError("E2B protected directory mode must be owner-traversable.")
    return mode


def _require_handoff_command_success(
    result: Any,
    *,
    phase: E2BGuestHandoffPhase,
    reason: str,
) -> None:
    exit_code = getattr(result, "exit_code", None)
    if type(exit_code) is not int:
        raise E2BGuestHandoffError(
            phase=phase,
            reason=f"{reason} returned no integer status",
        )
    if exit_code != 0:
        raise E2BGuestHandoffError(
            phase=phase,
            reason=reason,
            exit_code=exit_code,
        )


async def _run_handoff_command(
    command: Awaitable[Any],
    *,
    phase: E2BGuestHandoffPhase,
    reason: str,
) -> Any:
    """Run one Cayu-owned foreground command without exposing provider diagnostics."""

    try:
        result = await command
    except Exception as exc:
        exit_code = getattr(exc, "exit_code", None)
        raise E2BGuestHandoffError(
            phase=phase,
            reason=reason,
            exit_code=exit_code if type(exit_code) is int else None,
        ) from None
    _require_handoff_command_success(result, phase=phase, reason=reason)
    return result


async def _verify_guest_handoff(
    runner: E2BRunner,
    *,
    protected_assets: tuple[_E2BProtectedAsset, ...],
    timeout_s: float,
) -> None:
    guest_user = runner._hardened_guest_user
    if guest_user is None:
        raise RuntimeError("E2B guest verification requires a pinned handoff user.")
    runner._ensure_exec_open()
    await _run_handoff_command(
        runner._sandbox.commands.run(
            _VERIFY_GUEST_HANDOFF_SCRIPT,
            user=guest_user,
            envs=_handoff_command_env(CAYU_GUEST_USER=guest_user),
            timeout=timeout_s,
        ),
        phase="verification",
        reason="guest privilege and metadata verification",
    )
    for asset in protected_assets:
        runner._ensure_exec_open()
        await _run_handoff_command(
            runner._sandbox.commands.run(
                _VERIFY_PROTECTED_ASSET_MUTATION_SCRIPT,
                user=guest_user,
                envs=_handoff_command_env(
                    CAYU_PROTECTED_PATH=asset.path,
                    CAYU_PROTECTED_KIND=asset.kind,
                ),
                timeout=timeout_s,
            ),
            phase="verification",
            reason=f"protected {asset.kind} mutation verification",
        )
        # Ownership/mode inspection is privileged so nested assets remain
        # verifiable beneath an intentionally root-only (for example 0700)
        # protected directory. The guest mutation probe still establishes the
        # externally relevant denial before this postcondition check.
        runner._ensure_exec_open()
        await _run_handoff_command(
            runner._sandbox.commands.run(
                _VERIFY_PROTECTED_ASSET_METADATA_SCRIPT,
                user="root",
                envs=_handoff_command_env(
                    CAYU_PROTECTED_PATH=asset.path,
                    CAYU_PROTECTED_MODE=f"{asset.mode:o}",
                ),
                timeout=timeout_s,
            ),
            phase="verification",
            reason=f"protected {asset.kind} ownership verification",
        )


async def _cleanup_handoff_failure(
    *,
    module: ModuleType | Any,
    runner: E2BRunner | None,
    handoff_id: str,
    connection_options: Mapping[str, Any],
    original_error: BaseException,
    timeout_s: float,
) -> None:
    cleanup_task = asyncio.create_task(
        _rollback_handoff_sandbox(
            module=module,
            runner=runner,
            handoff_id=handoff_id,
            connection_options=connection_options,
            timeout_s=timeout_s,
        )
    )
    primary_cancellation = (
        original_error if isinstance(original_error, asyncio.CancelledError) else None
    )
    cleanup_cancellation: asyncio.CancelledError | None = None
    cleanup_error: BaseException | None = None
    if primary_cancellation is not None:
        # Consume only the request whose CancelledError entered this rollback.
        # Later cancellation requests must remain pending so the task keeps its
        # native cancelled() / cancelling() semantics after cleanup.
        _uncancel_current_task_once()
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not cleanup_task.done():
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            cleanup_error = TimeoutError("E2B guest handoff rollback timed out.")
            cleanup_task.cancel()
            cleanup_task.add_done_callback(_consume_task_outcome)
            break
        try:
            await asyncio.wait_for(asyncio.shield(cleanup_task), timeout=remaining)
        except asyncio.CancelledError as cancellation:
            current_task = asyncio.current_task()
            if current_task is None or current_task.cancelling() == 0:
                # shield() also raises CancelledError when the cleanup task
                # cancels itself. That is a rollback failure, not evidence
                # that the caller cancelled the supervising handoff.
                cleanup_error = _E2BHandoffRollbackCancelledError(
                    "E2B guest handoff rollback cancelled without caller cancellation."
                )
                break
            if cleanup_cancellation is None:
                cleanup_cancellation = cancellation
            # Do not uncancel this newer request. Once its exception has been
            # delivered, shielded cleanup can continue while Task.cancelling()
            # truthfully retains the caller-owned cancellation count.
        except TimeoutError:
            if cleanup_task.done():
                break
            cleanup_error = TimeoutError("E2B guest handoff rollback timed out.")
            cleanup_task.cancel()
            cleanup_task.add_done_callback(_consume_task_outcome)
            break
        except BaseException:
            if cleanup_task.done():
                break
            raise
    if cleanup_task.cancelled():
        if cleanup_error is None:
            cleanup_error = _E2BHandoffRollbackCancelledError(
                "E2B guest handoff rollback cancelled without caller cancellation."
            )
    elif cleanup_error is None and cleanup_task.done():
        try:
            killed = cleanup_task.result()
            if killed:
                if runner is not None:
                    runner._closed = True
            elif runner is not None:
                # False is the provider's idempotent "already gone" result.
                runner._closed = True
        except BaseException as exc:
            cleanup_error = exc
    if primary_cancellation is not None:
        if cleanup_error is not None:
            primary_cancellation.add_note(_handoff_rollback_diagnostic(cleanup_error))
        raise primary_cancellation from cleanup_error
    if cleanup_cancellation is not None:
        cleanup_cancellation.add_note(
            f"E2B guest handoff also failed: {type(original_error).__name__}."
        )
        cause: BaseException = original_error
        if cleanup_error is not None:
            cleanup_cancellation.add_note(_handoff_rollback_diagnostic(cleanup_error))
            cause = BaseExceptionGroup(
                "E2B guest handoff and rollback failed.",
                [original_error, cleanup_error],
            )
        raise cleanup_cancellation from cause
    if cleanup_error is not None:
        raise BaseExceptionGroup(
            "E2B guest handoff and rollback failed.",
            [original_error, cleanup_error],
        )


async def _kill_late_handoff_runner(
    runner: E2BRunner,
    *,
    timeout_s: float,
) -> None:
    """Bound cleanup for an allocation acknowledged after supervision ended."""

    kill_task = asyncio.create_task(runner._sandbox.kill())
    done, _ = await asyncio.wait({kill_task}, timeout=timeout_s)
    if kill_task in done:
        kill_task.result()
        return
    kill_task.cancel()
    kill_task.add_done_callback(_consume_task_outcome)


async def _rollback_handoff_sandbox(
    *,
    module: ModuleType | Any,
    runner: E2BRunner | None,
    handoff_id: str,
    connection_options: Mapping[str, Any],
    timeout_s: float,
) -> bool:
    if runner is not None:
        return bool(await runner._sandbox.kill())
    return await _kill_handoff_sandboxes_by_metadata(
        module,
        handoff_id=handoff_id,
        connection_options=connection_options,
        timeout_s=timeout_s,
    )


async def _kill_handoff_sandboxes_by_metadata(
    module: ModuleType | Any,
    *,
    handoff_id: str,
    connection_options: Mapping[str, Any],
    timeout_s: float,
) -> bool:
    """Reconcile a create call whose remote allocation result is ambiguous."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    stable_empty_since: float | None = None
    killed_any = False
    settle_seconds = min(1.0, timeout_s / 3)
    completion_margin = min(0.05, timeout_s / 10)
    while True:
        remaining = deadline - loop.time()
        operation_remaining = remaining - completion_margin
        if operation_remaining <= 0:
            # A create that never became observable is inherently ambiguous.
            # Poll for essentially the complete cleanup window rather than
            # treating an empty listing as proof that no allocation can appear
            # later. Exhausting the window must remain externally visible as
            # cleanup uncertainty rather than silently claiming success.
            raise TimeoutError("E2B guest handoff allocation reconciliation remained ambiguous.")
        request_options = dict(connection_options)
        request_timeout = request_options.get("request_timeout")
        if (
            request_timeout is None
            or type(request_timeout) not in {int, float}
            or request_timeout <= 0
            or request_timeout > operation_remaining
        ):
            request_options["request_timeout"] = operation_remaining
        query = module.SandboxQuery(
            metadata={_E2B_GUEST_HANDOFF_METADATA_KEY: handoff_id},
        )
        paginator = module.AsyncSandbox.list(
            query=query,
            limit=100,
            **request_options,
        )
        matching_ids: list[str] = []
        while paginator.has_next:
            for sandbox in await paginator.next_items():
                sandbox_id = getattr(sandbox, "sandbox_id", None)
                if type(sandbox_id) is not str:
                    raise RuntimeError("E2B sandbox list returned an invalid sandbox id.")
                matching_ids.append(_validate_sandbox_id(sandbox_id))
        if matching_ids:
            stable_empty_since = None
            for sandbox_id in matching_ids:
                await module.AsyncSandbox.kill(
                    sandbox_id,
                    **request_options,
                )
                killed_any = True
            continue
        elif stable_empty_since is None:
            stable_empty_since = loop.time()
        elif killed_any and loop.time() - stable_empty_since >= settle_seconds:
            return killed_any
        stable_remaining = (
            settle_seconds
            if stable_empty_since is None or not killed_any
            else max(settle_seconds - (loop.time() - stable_empty_since), 0)
        )
        if stable_remaining <= 0:
            continue
        sleep_seconds = min(
            0.25,
            stable_remaining,
            max(deadline - loop.time() - completion_margin, 0),
        )
        if sleep_seconds <= 0:
            raise TimeoutError("E2B guest handoff allocation reconciliation remained ambiguous.")
        await asyncio.sleep(sleep_seconds)


def _contains_cancellation(error: BaseException) -> bool:
    if isinstance(error, asyncio.CancelledError):
        return True
    if isinstance(error, BaseExceptionGroup):
        return any(_contains_cancellation(item) for item in error.exceptions)
    return False


def _uncancel_current_task_once() -> None:
    current_task = asyncio.current_task()
    if current_task is not None and current_task.cancelling() > 0:
        current_task.uncancel()


def _handoff_rollback_diagnostic(error: BaseException) -> str:
    return f"E2B guest handoff rollback incomplete: {type(error).__name__}."


def _consume_task_outcome(task: asyncio.Task[Any]) -> None:
    with contextlib.suppress(BaseException):
        task.result()


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
