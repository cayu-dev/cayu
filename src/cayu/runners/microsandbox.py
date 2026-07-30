from __future__ import annotations

import asyncio
import contextlib
import importlib
import posixpath
import random
from abc import abstractmethod
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from types import ModuleType
from typing import Any, Literal, cast

from cayu._exception_groups import exception_group_children
from cayu._exception_state import exception_state, set_exception_state
from cayu._task_wait import await_shielded_task_outcome
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
from cayu.runners._redacted_output import RedactedOutputCapture
from cayu.runners._subprocess import (
    copy_runner_env,
    remove_runner_env,
    validate_output_limit,
    validate_stdin,
    validate_timeout,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerExecutionError,
    RunnerUnavailableError,
    RunnerWorkspaceCapability,
    RunnerWorkspaceCapabilityT,
    attach_cancellation_artifacts,
    runner_execution_error,
)
from cayu.vaults import SecretRedactor

DEFAULT_MICROSANDBOX_IMAGE = "python:3.13"
DEFAULT_MICROSANDBOX_CWD = "/workspace"
DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS = 5.0
DEFAULT_MICROSANDBOX_RECONNECT_TIMEOUT_SECONDS = 15.0
MICROSANDBOX_NAME_MAX_BYTES = 128
_MICROSANDBOX_REMOVE_INITIAL_BACKOFF_SECONDS = 0.05
_MICROSANDBOX_REMOVE_MAX_BACKOFF_SECONDS = 0.5
_MICROSANDBOX_SETTLEMENT_MAX_BACKOFF_SECONDS = 30.0
_MICROSANDBOX_SETTLEMENT_JITTER_RATIO = 0.2
_MICROSANDBOX_CLEANUP_DIAGNOSTIC_TYPE = "cayu.microsandbox_cleanup.v1"
MICROSANDBOX_LIVENESS_TIMEOUT_SECONDS = 1.0
_MICROSANDBOX_NO_EXIT_EVENT_ERROR = "runtime error: exec session ended without exit event"
_MICROSANDBOX_UNAVAILABLE_REMEDIATION = (
    "Reconnect to or replace the Microsandbox before executing more commands."
)
_MICROSANDBOX_RECONNECT_SETTLEMENT_TASK_ATTRIBUTE = "_cayu_microsandbox_reconnect_settlement_task"
_MICROSANDBOX_RECONNECT_SETTLEMENT_TASK_TOKEN = object()
_MICROSANDBOX_RECONNECT_SETTLEMENT_TASKS: set[asyncio.Task[None]] = set()

MicrosandboxCloseAction = Literal["remove", "stop", "detach", "none"]


class MicrosandboxReconnectIdentityError(ValueError):
    """The provider handle does not match durable reconnect identity."""


@dataclass(frozen=True, slots=True)
class _MicrosandboxReconnectSettlementTaskHandoff:
    task: asyncio.Task[None]
    token: object


def microsandbox_reconnect_settlement_task(
    error: BaseException,
) -> asyncio.Task[None] | None:
    """Return deferred restart settlement owned by a reconnect failure."""

    handoff = exception_state(
        error,
        _MICROSANDBOX_RECONNECT_SETTLEMENT_TASK_ATTRIBUTE,
    )
    if (
        type(handoff) is not _MicrosandboxReconnectSettlementTaskHandoff
        or handoff.token is not _MICROSANDBOX_RECONNECT_SETTLEMENT_TASK_TOKEN
        or not isinstance(handoff.task, asyncio.Task)
    ):
        return None
    return handoff.task


class MicrosandboxWorkspaceCapability(RunnerWorkspaceCapability):
    """Native Microsandbox filesystem access without runner lifecycle authority."""

    @property
    @abstractmethod
    def sandbox_name(self) -> str:
        """Provider sandbox name used for truthful workspace identity."""

    @abstractmethod
    async def list_entries(self, path: str) -> Sequence[MicrosandboxWorkspaceEntry]:
        """List normalized Cayu-owned entries for one guest directory."""

    @abstractmethod
    async def real_path(self, path: str) -> str:
        """Resolve a canonical guest path through the provider transport."""


class _MicrosandboxWorkspaceCapability(MicrosandboxWorkspaceCapability):
    def __init__(self, runner: MicrosandboxRunner) -> None:
        self._runner = runner

    @property
    def sandbox_name(self) -> str:
        return self._runner.name

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("microsandbox", self._runner.name)

    async def list_entries(self, path: str) -> Sequence[MicrosandboxWorkspaceEntry]:
        entries = await self._runner.filesystem().list(path)
        return tuple(MicrosandboxWorkspaceEntry.from_provider_entry(entry) for entry in entries)

    async def real_path(self, path: str) -> str:
        return await self._runner.real_path(path)


@dataclass(frozen=True)
class MicrosandboxWorkspaceEntry:
    """Cayu-owned normalized view of one Microsandbox filesystem entry."""

    path: str | None
    kind: str | None

    @classmethod
    def from_provider_entry(cls, entry: Any) -> MicrosandboxWorkspaceEntry:
        path = getattr(entry, "path", None)
        kind = getattr(entry, "kind", None)
        return cls(
            path=path if type(path) is str else None,
            kind=kind if type(kind) is str else None,
        )


class MicrosandboxCleanupError(RuntimeError):
    """Terminal bounded Microsandbox lifecycle cleanup failure."""

    def __init__(self, message: str, *, diagnostic: dict[str, Any]) -> None:
        super().__init__(message)
        self.diagnostic = copy_json_value(diagnostic, "diagnostic")


class _MicrosandboxDeadlineExceeded(TimeoutError):
    pass


class _MicrosandboxCleanupExceptionGroup(BaseExceptionGroup):
    pass


class MicrosandboxUnavailableError(RunnerUnavailableError):
    """The Microsandbox guest agent was confirmed unreachable after a command."""

    def __init__(
        self,
        *,
        sandbox_name: str,
        last_command: Mapping[str, Any],
        probe: Mapping[str, Any],
    ) -> None:
        self.sandbox_name = sandbox_name
        self.last_command: dict[str, Any] = copy_json_value(
            dict(last_command),
            "last_command",
        )
        self.probe: dict[str, Any] = copy_json_value(dict(probe), "probe")
        self.probe_status = str(self.probe.get("status", "failed"))
        if self.last_command.get("exit_code") == -9:
            reason = "guest_agent_unavailable_after_signal_9"
        else:
            reason = "guest_agent_unavailable_after_incomplete_exec"
        diagnostic = {
            "type": "cayu.runner_unavailable.v1",
            "adapter": "microsandbox",
            "sandbox_name": sandbox_name,
            "status": "unavailable",
            "reason": reason,
            "last_command": self.last_command,
            "probe": self.probe,
            "remediation": _MICROSANDBOX_UNAVAILABLE_REMEDIATION,
        }
        super().__init__(
            f"Microsandbox guest agent unavailable for {sandbox_name!r} after an abnormal "
            f"command outcome. "
            f"{_MICROSANDBOX_UNAVAILABLE_REMEDIATION}",
            diagnostic=diagnostic,
        )


class MicrosandboxRunner(Runner):
    """Executes commands in a Microsandbox microVM sandbox.

    The runner does not inherit the trusted host process environment. Pass
    explicit `env` values, preferably resolved at the environment/vault boundary.
    Commands are serialized so a guest-agent liveness transition completes
    before another guest launch. `reopen_exec()` does not clear confirmed
    guest-agent unavailability; reconnect with a new runner instead.
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
        liveness_timeout_s: float = MICROSANDBOX_LIVENESS_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        remove_timeout_s: float = DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
        env_overlay: Mapping[str, str] | None = None,
        sandbox_module: ModuleType | Any | None = None,
        _restarted_from_stopped: bool = False,
    ) -> None:
        if sandbox is None:
            raise TypeError("MicrosandboxRunner sandbox cannot be None.")
        if type(_restarted_from_stopped) is not bool:
            raise TypeError("MicrosandboxRunner restart evidence must be a bool.")
        self.name = _validate_sandbox_name(name)
        self.default_cwd = _validate_guest_root(default_cwd)
        self.close_action = _validate_close_action(close_action)
        self.cancel_timeout_s = validate_cancel_timeout(cancel_timeout_s)
        self.liveness_timeout_s = _validate_liveness_timeout(liveness_timeout_s)
        self.cancellation_cleanup = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        self.timeout_cleanup = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        self.remove_timeout_s = _validate_remove_timeout(remove_timeout_s)
        self.env_overlay = dict(env_overlay) if env_overlay else {}
        self._sandbox = sandbox
        self._sandbox_module = sandbox_module
        self._restarted_from_stopped = _restarted_from_stopped
        self._sftp_client: Any = None
        self._sftp: Any = None
        self._sftp_lock = asyncio.Lock()
        self._exec_lock = asyncio.Lock()
        self._last_cleanup_diagnostic: dict[str, Any] | None = None
        self._remove_stop_completed = False
        self._remove_stop_status: str | None = None
        self._unavailable_last_command: dict[str, Any] | None = None
        self._unavailable_probe: dict[str, Any] | None = None

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        image: Any = DEFAULT_MICROSANDBOX_IMAGE,
        default_cwd: str = DEFAULT_MICROSANDBOX_CWD,
        close_action: MicrosandboxCloseAction = "remove",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        liveness_timeout_s: float = MICROSANDBOX_LIVENESS_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        remove_timeout_s: float = DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
        ensure_default_cwd: bool = True,
        env_overlay: Mapping[str, str] | None = None,
        sandbox_module: ModuleType | Any | None = None,
        **sandbox_options: Any,
    ) -> MicrosandboxRunner:
        """Create a sandbox and return a runner bound to it.

        Guest networking defaults to `microsandbox.Network.none()`. Pass an
        explicit provider network policy to opt into another creation-time
        network contract.

        Other provider-specific options are passed through unchanged, including
        volumes, resources, labels, secrets, and replace behavior. An explicit
        network policy is also forwarded unchanged.
        """

        module = _microsandbox_module(sandbox_module)
        sandbox_name = _validate_sandbox_name(name)
        guest_root = _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        liveness_timeout = _validate_liveness_timeout(liveness_timeout_s)
        removal_timeout = _validate_remove_timeout(remove_timeout_s)
        if type(ensure_default_cwd) is not bool:
            raise TypeError("MicrosandboxRunner ensure_default_cwd must be a bool.")
        create_options = dict(sandbox_options)
        if "network" not in create_options:
            network_type = getattr(module, "Network", None)
            deny_all = getattr(network_type, "none", None)
            if not isinstance(network_type, type) or not callable(deny_all):
                raise RuntimeError(
                    "The supported microsandbox SDK does not provide Network.none()."
                )
            network = deny_all()
            if not isinstance(network, network_type):
                raise TypeError("microsandbox.Network.none() returned an invalid network policy.")
            create_options["network"] = network
        sandbox = await module.Sandbox.create(
            sandbox_name,
            image=image,
            **create_options,
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
            liveness_timeout_s=liveness_timeout,
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
        liveness_timeout_s: float = MICROSANDBOX_LIVENESS_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        remove_timeout_s: float = DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
        env_overlay: Mapping[str, str] | None = None,
        sandbox_module: ModuleType | Any | None = None,
        reconnect_timeout_s: float = DEFAULT_MICROSANDBOX_RECONNECT_TIMEOUT_SECONDS,
        expected_created_at: float | None = None,
    ) -> MicrosandboxRunner:
        """Attach to an existing Microsandbox sandbox by name.

        The sandbox creator owns its creation-time network contract. Attaching
        does not inspect, replace, or strengthen that network policy.
        """

        return await cls._from_existing(
            name,
            default_cwd=default_cwd,
            close_action=close_action,
            cancel_timeout_s=cancel_timeout_s,
            liveness_timeout_s=liveness_timeout_s,
            cancellation_cleanup=cancellation_cleanup,
            timeout_cleanup=timeout_cleanup,
            remove_timeout_s=remove_timeout_s,
            env_overlay=env_overlay,
            sandbox_module=sandbox_module,
            reconnect_timeout_s=reconnect_timeout_s,
            expected_created_at=expected_created_at,
            lifecycle_only=False,
        )

    @classmethod
    async def _from_existing_for_lifecycle(
        cls,
        name: str,
        *,
        close_action: Literal["stop", "remove"],
        sandbox_module: ModuleType | Any | None,
        reconnect_timeout_s: float,
        expected_created_at: float,
    ) -> MicrosandboxRunner:
        """Attach only to apply a non-executable lifecycle transition."""

        return await cls._from_existing(
            name,
            close_action=close_action,
            sandbox_module=sandbox_module,
            reconnect_timeout_s=reconnect_timeout_s,
            expected_created_at=expected_created_at,
            lifecycle_only=True,
        )

    @classmethod
    async def _from_existing(
        cls,
        name: str,
        *,
        default_cwd: str = DEFAULT_MICROSANDBOX_CWD,
        close_action: MicrosandboxCloseAction = "none",
        cancel_timeout_s: float | None = DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS,
        liveness_timeout_s: float = MICROSANDBOX_LIVENESS_TIMEOUT_SECONDS,
        cancellation_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_CANCELLATION_CLEANUP_POLICY,
        timeout_cleanup: RunnerCleanupPolicy = DEFAULT_RUNNER_TIMEOUT_CLEANUP_POLICY,
        remove_timeout_s: float = DEFAULT_MICROSANDBOX_REMOVE_TIMEOUT_SECONDS,
        env_overlay: Mapping[str, str] | None = None,
        sandbox_module: ModuleType | Any | None = None,
        reconnect_timeout_s: float = DEFAULT_MICROSANDBOX_RECONNECT_TIMEOUT_SECONDS,
        expected_created_at: float | None = None,
        lifecycle_only: bool,
    ) -> MicrosandboxRunner:
        module = _microsandbox_module(sandbox_module)
        sandbox_name = _validate_sandbox_name(name)
        _validate_guest_root(default_cwd)
        _validate_close_action(close_action)
        cancellation_policy = validate_runner_cleanup_policy(
            cancellation_cleanup, "cancellation_cleanup"
        )
        timeout_policy = validate_runner_cleanup_policy(timeout_cleanup, "timeout_cleanup")
        liveness_timeout = _validate_liveness_timeout(liveness_timeout_s)
        removal_timeout = _validate_remove_timeout(remove_timeout_s)
        reconnect_timeout = _validate_reconnect_timeout(reconnect_timeout_s)
        if type(lifecycle_only) is not bool:
            raise TypeError("lifecycle_only must be a bool.")
        restarted_from_stopped = False
        loop = asyncio.get_running_loop()
        reconnect_deadline = loop.time() + reconnect_timeout
        try:
            async with asyncio.timeout_at(reconnect_deadline):
                handle = await module.Sandbox.get(sandbox_name)
            if expected_created_at is not None:
                actual_created_at = _validate_provider_created_at(
                    getattr(handle, "created_at", None)
                )
                if actual_created_at != expected_created_at:
                    raise MicrosandboxReconnectIdentityError(
                        "Microsandbox reconnect provider incarnation does not match."
                    )
            status = _sandbox_status_value(handle)
            if status is not None and status.lower() == "stopped":
                if lifecycle_only:
                    # Lifecycle-only cleanup must never make guest code
                    # executable merely to obtain a connected runner.
                    sandbox = handle
                else:
                    start = getattr(module.Sandbox, "start", None)
                    if start is None:
                        raise RuntimeError("Microsandbox SDK cannot restart a stopped sandbox.")
                    restarted_sandbox: Any = None
                    start_task = asyncio.create_task(
                        start(sandbox_name, detached=True),
                        name=f"cayu-microsandbox-reconnect-start-{sandbox_name}",
                    )
                    try:
                        start_outcome = await await_shielded_task_outcome(
                            start_task,
                            timeout_s=max(reconnect_deadline - loop.time(), 0.0),
                        )
                        if start_outcome.timed_out:
                            settlement_task = _defer_reconnect_start_restoration(
                                module,
                                handle,
                                sandbox_name,
                                start_task,
                            )
                            signal: BaseException
                            if start_outcome.cancellation is not None:
                                signal = start_outcome.cancellation
                            else:
                                signal = TimeoutError(
                                    "Microsandbox stopped-sandbox restart exceeded its "
                                    "reconnect deadline."
                                )
                            _attach_reconnect_settlement_task(signal, settlement_task)
                            raise signal
                        restarted_sandbox = start_outcome.result
                        if (
                            start_outcome.error is not None
                            and start_outcome.cancellation is not None
                        ):
                            restart_error = BaseExceptionGroup(
                                "Microsandbox stopped-sandbox restart failed after "
                                "caller cancellation.",
                                [start_outcome.cancellation, start_outcome.error],
                            )
                            raise restart_error from start_outcome.cancellation
                        if start_outcome.error is not None:
                            raise start_outcome.error
                        if start_outcome.cancellation is not None:
                            raise start_outcome.cancellation
                        restarted_from_stopped = True
                        try:
                            async with asyncio.timeout_at(reconnect_deadline):
                                handle = await module.Sandbox.get(sandbox_name)
                                if expected_created_at is not None:
                                    restarted_created_at = _validate_provider_created_at(
                                        getattr(handle, "created_at", None)
                                    )
                                    if restarted_created_at != expected_created_at:
                                        raise MicrosandboxReconnectIdentityError(
                                            "Microsandbox restarted provider incarnation does "
                                            "not match."
                                        )
                                sandbox = (
                                    restarted_sandbox
                                    if restarted_sandbox is not None
                                    else await handle.connect()
                                )
                        except TimeoutError as reconnect_error:
                            settlement_task = _defer_reconnect_restoration(
                                module,
                                restarted_sandbox if restarted_sandbox is not None else handle,
                                sandbox_name,
                            )
                            _attach_reconnect_settlement_task(
                                reconnect_error,
                                settlement_task,
                            )
                            raise
                    except BaseException as reconnect_error:
                        if microsandbox_reconnect_settlement_task(reconnect_error) is None:
                            await _stop_restarted_sandbox_after_failed_reconnect(
                                module,
                                restarted_sandbox if restarted_sandbox is not None else handle,
                                sandbox_name,
                                reconnect_error,
                                deadline=reconnect_deadline,
                            )
                        raise
            else:
                async with asyncio.timeout_at(reconnect_deadline):
                    sandbox = await handle.connect()
        except TimeoutError as exc:
            error = TimeoutError(
                f"Microsandbox reconnect did not attach within {reconnect_timeout:g} seconds."
            )
            settlement_task = microsandbox_reconnect_settlement_task(exc)
            if settlement_task is not None:
                _attach_reconnect_settlement_task(error, settlement_task)
            raise error from exc
        return cls(
            sandbox,
            name=sandbox_name,
            default_cwd=default_cwd,
            close_action=close_action,
            cancel_timeout_s=cancel_timeout_s,
            liveness_timeout_s=liveness_timeout,
            cancellation_cleanup=cancellation_policy,
            timeout_cleanup=timeout_policy,
            remove_timeout_s=removal_timeout,
            env_overlay=env_overlay,
            sandbox_module=module,
            _restarted_from_stopped=restarted_from_stopped,
        )

    @property
    def resource_key(self) -> tuple[object, ...]:
        return ("microsandbox", self.name)

    @property
    def closed(self) -> bool:
        """Whether this runner instance has completed its lifecycle action."""

        return self._closed

    @property
    def restarted_from_stopped(self) -> bool:
        """Whether this attachment made a stopped allocation executable."""

        return self._restarted_from_stopped

    def _defer_stopped_boundary_restoration(
        self,
        initial_task: asyncio.Task[Any],
    ) -> asyncio.Task[None]:
        """Retain retry ownership after a restarted allocation fails to stop."""

        if not self._restarted_from_stopped:
            raise RuntimeError(
                "Stopped-boundary restoration requires a runner restarted from stopped."
            )
        if not isinstance(initial_task, asyncio.Task):
            raise TypeError("Stopped-boundary restoration requires an asyncio Task.")
        return _defer_reconnect_restoration(
            _microsandbox_module(self._sandbox_module),
            self._sandbox,
            self.name,
            initial_task=initial_task,
        )

    def _defer_terminal_removal(
        self,
        initial_task: asyncio.Task[Any] | None = None,
    ) -> asyncio.Task[None]:
        """Retain retry ownership until a fresh allocation is removed."""

        if self.close_action != "remove":
            raise RuntimeError("Terminal removal settlement requires close_action='remove'.")
        if initial_task is not None and not isinstance(initial_task, asyncio.Task):
            raise TypeError("Terminal removal settlement requires an asyncio Task.")
        return _defer_microsandbox_settlement(
            _microsandbox_module(self._sandbox_module),
            self.close,
            self.name,
            operation="terminal-removal",
            initial_task=initial_task,
        )

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        if capability_type is MicrosandboxWorkspaceCapability:
            capability = _MicrosandboxWorkspaceCapability(self)
            return cast("RunnerWorkspaceCapabilityT", capability)
        return super().workspace_capability(capability_type)

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
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        async with self._exec_lock:
            return await self._exec_serialized(
                command,
                output_redactor=SecretRedactor(),
                cwd=cwd,
                env=env,
                env_remove=env_remove,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=output_limit_bytes,
            )

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
            raise TypeError("MicrosandboxRunner redactor must be a SecretRedactor.")
        async with self._exec_lock:
            return await self._exec_serialized(
                command,
                output_redactor=redactor,
                cwd=cwd,
                env=env,
                env_remove=env_remove,
                timeout_s=timeout_s,
                stdin=stdin,
                output_limit_bytes=output_limit_bytes,
            )

    async def _exec_serialized(
        self,
        command: ExecCommand,
        *,
        output_redactor: SecretRedactor,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        if type(command) is not ExecCommand:
            raise TypeError("MicrosandboxRunner command must be an ExecCommand.")
        self._ensure_agent_available()
        self._ensure_exec_open()

        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        environment = remove_runner_env(environment, env_remove)
        if self.env_overlay:
            environment.update(self.env_overlay)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        sdk_stdin = standard_input.encode("utf-8") if standard_input is not None else None
        output_limit = validate_output_limit(output_limit_bytes)

        stdout = RedactedOutputCapture(redactor=output_redactor, limit=output_limit)
        stderr = RedactedOutputCapture(redactor=output_redactor, limit=output_limit)
        handle = None
        exit_code: int | None = None
        execution_failure: RunnerExecutionError | None = None
        liveness_diagnostic: dict[str, Any] | None = None

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
            stdout.abort()
            stderr.abort()
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
                stdout.abort()
                stderr.abort()
                module = _microsandbox_module(self._sandbox_module)
                if _is_opaque_microsandbox_exec_failure(module, exc):
                    liveness_diagnostic = {
                        "exit_code": exit_code,
                        "timed_out": False,
                        "cancelled": False,
                        "stdout_bytes": stdout.total_bytes,
                        "stderr_bytes": stderr.total_bytes,
                        "error_type": type(exc).__name__,
                    }
                execution_failure = runner_execution_error(
                    exc,
                    adapter="microsandbox",
                    stdout_bytes=stdout.total_bytes,
                    stderr_bytes=stderr.total_bytes,
                )
            else:
                stdout.abort()
                stderr.abort()
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

        if liveness_diagnostic is not None:
            await self._confirm_agent_available(liveness_diagnostic)
        if execution_failure is not None:
            raise execution_failure

        stdout.finish_complete()
        stderr.finish_complete()
        result = ExecResult(
            stdout=stdout.text(),
            stderr=stderr.text(),
            exit_code=exit_code if exit_code is not None else 0,
            timed_out=False,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            stdout_bytes=stdout.total_bytes,
            stderr_bytes=stderr.total_bytes,
        )
        if result.exit_code == -9:
            await self._confirm_agent_available(
                {
                    "exit_code": result.exit_code,
                    "timed_out": result.timed_out,
                    "cancelled": result.cancelled,
                    "stdout_bytes": result.stdout_bytes,
                    "stderr_bytes": result.stderr_bytes,
                    "error_type": None,
                }
            )
        return result

    def _ensure_agent_available(self) -> None:
        if self._closed:
            return
        if self._unavailable_last_command is None or self._unavailable_probe is None:
            return
        raise MicrosandboxUnavailableError(
            sandbox_name=self.name,
            last_command=self._unavailable_last_command,
            probe=self._unavailable_probe,
        )

    async def _confirm_agent_available(self, last_command: Mapping[str, Any]) -> None:
        module = _microsandbox_module(self._sandbox_module)
        ping_error: Exception | None = None
        status_error: Exception | None = None
        registry_status: str | None = None
        try:
            async with asyncio.timeout(self.liveness_timeout_s):
                try:
                    await self._sandbox.ping()
                except Exception as exc:
                    ping_error = exc
                    try:
                        handle = await module.Sandbox.get(self.name)
                        status = getattr(handle, "status", None)
                        if status is not None:
                            registry_status = str(status)
                    except Exception as exc:
                        status_error = exc
                else:
                    return
        except TimeoutError:
            if ping_error is None:
                ping_error = TimeoutError("Microsandbox guest-agent liveness probe timed out.")
                probe_status = "timed_out"
            else:
                status_error = TimeoutError("Microsandbox registry status probe timed out.")
                probe_status = "failed"
        else:
            probe_status = "failed"

        if ping_error is None:
            raise AssertionError("Failed Microsandbox liveness probe did not record an error.")
        probe = {
            "method": "Sandbox.ping",
            "status": probe_status,
            "timeout_s": self.liveness_timeout_s,
            "registry_status": registry_status,
            "error_type": type(ping_error).__name__,
            "status_error_type": type(status_error).__name__ if status_error is not None else None,
        }
        error = MicrosandboxUnavailableError(
            sandbox_name=self.name,
            last_command=last_command,
            probe=probe,
        )
        self._unavailable_last_command = copy_json_value(error.last_command, "last_command")
        self._unavailable_probe = copy_json_value(error.probe, "probe")
        self._close_exec("microsandbox guest agent unavailable after an abnormal command outcome")
        raise error from None

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


def _validate_reconnect_timeout(value: float) -> float:
    if type(value) not in {int, float}:
        raise TypeError("Microsandbox reconnect timeout must be numeric.")
    if not isfinite(value) or value <= 0:
        raise ValueError("Microsandbox reconnect timeout must be finite and greater than zero.")
    return float(value)


def _validate_provider_created_at(value: Any) -> float:
    if type(value) not in {int, float} or not isfinite(value) or value <= 0:
        raise MicrosandboxReconnectIdentityError(
            "Microsandbox provider created_at must be a positive finite number."
        )
    return float(value)


def _validate_liveness_timeout(value: float) -> float:
    if type(value) not in {int, float}:
        raise TypeError("MicrosandboxRunner liveness_timeout_s must be numeric.")
    if not isfinite(value):
        raise ValueError("MicrosandboxRunner liveness_timeout_s must be finite.")
    if value <= 0:
        raise ValueError("MicrosandboxRunner liveness_timeout_s must be greater than zero.")
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
    stdout: RedactedOutputCapture,
    stderr: RedactedOutputCapture,
    output: Any,
) -> None:
    _replace_with_collected_stream(stdout, output, "stdout")
    _replace_with_collected_stream(stderr, output, "stderr")


def _replace_with_collected_stream(
    buffer: RedactedOutputCapture,
    output: Any,
    stream_name: Literal["stdout", "stderr"],
) -> None:
    data = _collected_stream_bytes(output, stream_name)
    if data is not None and (data or buffer.total_bytes == 0):
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


def _is_opaque_microsandbox_exec_failure(module: ModuleType | Any, exc: Exception) -> bool:
    error_type = getattr(module, "MicrosandboxError", None)
    # The SDK maps many unrelated variants to this exact base class. Only the
    # observed no-exit-event error is evidence for a guest-agent liveness probe.
    return (
        isinstance(error_type, type)
        and type(exc) is error_type
        and str(exc) == _MICROSANDBOX_NO_EXIT_EVENT_ERROR
    )


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
    # The supported SDK exposes SandboxStatus as a str-backed enum. Accept
    # that authoritative representation without broad coercion of arbitrary
    # status objects.
    if isinstance(status, str) and status:
        return str(status)
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
        cleanup_failures = exception_group_children(cleanup_group)
        raise BaseExceptionGroup(
            message,
            [
                original_error,
                *(cleanup_failures if cleanup_failures is not None else (cleanup_group,)),
            ],
        ) from cleanup_group
    except (BaseExceptionGroup, Exception, asyncio.CancelledError) as cleanup_error:
        raise BaseExceptionGroup(message, [original_error, cleanup_error]) from cleanup_error


def _attach_reconnect_settlement_task(
    error: BaseException,
    task: asyncio.Task[None],
) -> None:
    handoff = _MicrosandboxReconnectSettlementTaskHandoff(
        task=task,
        token=_MICROSANDBOX_RECONNECT_SETTLEMENT_TASK_TOKEN,
    )
    if not set_exception_state(
        error,
        _MICROSANDBOX_RECONNECT_SETTLEMENT_TASK_ATTRIBUTE,
        handoff,
    ):
        raise RuntimeError("Could not attach Microsandbox reconnect settlement ownership.")


def _retain_reconnect_settlement_task(task: asyncio.Task[None]) -> None:
    """Keep deferred provider settlement alive and consume its terminal error."""

    _MICROSANDBOX_RECONNECT_SETTLEMENT_TASKS.add(task)

    def settled(completed: asyncio.Task[None]) -> None:
        _MICROSANDBOX_RECONNECT_SETTLEMENT_TASKS.discard(completed)
        if completed.cancelled():
            # Cancelled tasks do not emit unhandled-exception warnings. Reading
            # result() here would consume the cancellation message before the
            # lifecycle owner observes the original signal.
            return
        with contextlib.suppress(BaseException):
            completed.result()

    task.add_done_callback(settled)


def _defer_reconnect_start_restoration(
    module: ModuleType | Any,
    handle: Any,
    name: str,
    start_task: asyncio.Task[Any],
) -> asyncio.Task[None]:
    """Own an accepted restart until it settles and the guest is stopped."""

    async def settle() -> None:
        start_outcome = await await_shielded_task_outcome(start_task)
        sandbox = start_outcome.result if start_outcome.result is not None else handle
        await _retry_reconnect_restoration(
            module,
            sandbox,
        )

    task = asyncio.create_task(
        settle(),
        name=f"cayu-microsandbox-reconnect-settlement-{name}",
    )
    _retain_reconnect_settlement_task(task)
    return task


def _defer_reconnect_restoration(
    module: ModuleType | Any,
    sandbox: Any,
    name: str,
    *,
    initial_task: asyncio.Task[Any] | None = None,
) -> asyncio.Task[None]:
    """Retain one retrying owner until a restarted guest is stopped."""

    return _defer_microsandbox_settlement(
        module,
        lambda: _stop_sandbox(
            module,
            sandbox,
            not_found_is_removed=True,
        ),
        name,
        operation="reconnect-restoration",
        initial_task=initial_task,
    )


async def _retry_reconnect_restoration(
    module: ModuleType | Any,
    sandbox: Any,
) -> None:
    """Retry transient provider failures while the exact reconnect owner is retained."""

    await _retry_microsandbox_settlement(
        module,
        lambda: _stop_sandbox(
            module,
            sandbox,
            not_found_is_removed=True,
        ),
    )


def _defer_microsandbox_settlement(
    module: ModuleType | Any,
    operation_call: Callable[[], Awaitable[Any]],
    name: str,
    *,
    operation: str,
    initial_task: asyncio.Task[Any] | None = None,
) -> asyncio.Task[None]:
    """Retain one owner while a positively transient cleanup is retried."""

    async def settle() -> None:
        if initial_task is not None:
            try:
                await asyncio.shield(initial_task)
                return
            except BaseException as error:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                if not _is_retryable_microsandbox_settlement_failure(module, error):
                    raise
            await _sleep_before_microsandbox_settlement_retry(
                _MICROSANDBOX_REMOVE_INITIAL_BACKOFF_SECONDS
            )
        await _retry_microsandbox_settlement(module, operation_call)

    if initial_task is not None and not initial_task.done():
        _retain_reconnect_settlement_task(initial_task)
    task = asyncio.create_task(
        settle(),
        name=f"cayu-microsandbox-{operation}-settlement-{name}",
    )
    _retain_reconnect_settlement_task(task)
    return task


async def _retry_microsandbox_settlement(
    module: ModuleType | Any,
    operation_call: Callable[[], Awaitable[Any]],
) -> None:
    """Retry only failures carrying positive transient provider evidence."""

    backoff_s = _MICROSANDBOX_REMOVE_INITIAL_BACKOFF_SECONDS
    while True:
        try:
            await operation_call()
            return
        except BaseException as error:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            if not _is_retryable_microsandbox_settlement_failure(module, error):
                raise
        await _sleep_before_microsandbox_settlement_retry(backoff_s)
        backoff_s = min(
            backoff_s * 2,
            _MICROSANDBOX_SETTLEMENT_MAX_BACKOFF_SECONDS,
        )


async def _sleep_before_microsandbox_settlement_retry(delay_s: float) -> None:
    jitter_s = delay_s * _MICROSANDBOX_SETTLEMENT_JITTER_RATIO
    await _sleep_before_microsandbox_retry(random.uniform(delay_s - jitter_s, delay_s + jitter_s))


def _is_retryable_microsandbox_settlement_failure(
    module: ModuleType | Any,
    error: BaseException,
) -> bool:
    """Return whether every failure leaf positively permits an automatic retry."""

    pending = [error]
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if not children:
                return False
            pending.extend(children)
            continue
        if isinstance(candidate, asyncio.CancelledError):
            continue
        if not isinstance(candidate, Exception):
            # Fatal signals and unknown BaseException leaves remain authoritative.
            return False
        if isinstance(candidate, PermissionError):
            return False
        if isinstance(candidate, (MicrosandboxCleanupError, TimeoutError, ConnectionError)):
            continue
        if _is_microsandbox_error(module, candidate, "SandboxStillRunningError"):
            continue
        # Unknown, configuration, authentication, and unsupported-operation
        # failures require an explicit later recovery attempt. Automatically
        # retrying them would turn retained ownership into an unbounded provider
        # request loop.
        return False
    return True


async def _stop_restarted_sandbox_after_failed_reconnect(
    module: ModuleType | Any,
    sandbox: Any,
    name: str,
    original_error: BaseException,
    *,
    deadline: float,
) -> None:
    """Restore the stopped boundary before a failed reconnect releases ownership."""

    async def restore_stopped_boundary() -> None:
        await _stop_sandbox(
            module,
            sandbox,
            not_found_is_removed=True,
        )

    stop_task = asyncio.create_task(
        restore_stopped_boundary(),
        name=f"cayu-microsandbox-reconnect-stop-{name}",
    )
    outcome = await await_shielded_task_outcome(
        stop_task,
        cancellation=(
            original_error if isinstance(original_error, asyncio.CancelledError) else None
        ),
        timeout_s=max(deadline - asyncio.get_running_loop().time(), 0.0),
    )
    failures: list[BaseException] = [original_error]
    if outcome.error is not None and outcome.error is not original_error:
        failures.append(outcome.error)
    if (
        outcome.cancellation is not None
        and outcome.cancellation is not original_error
        and outcome.cancellation is not outcome.error
    ):
        failures.append(outcome.cancellation)
    if outcome.timed_out:
        failures.append(
            TimeoutError(
                "Microsandbox stopped-sandbox restoration exceeded its reconnect deadline."
            )
        )
    if len(failures) > 1:
        restoration_error = BaseExceptionGroup(
            "Microsandbox reconnect failed and restoring the stopped allocation also failed.",
            failures,
        )
        settlement_task = _defer_reconnect_restoration(
            module,
            sandbox,
            name,
            initial_task=stop_task,
        )
        # The live retry task is the concrete ownership evidence downstream
        # uses to keep the reconnect claim fenced until restoration succeeds.
        _attach_reconnect_settlement_task(restoration_error, settlement_task)
        raise restoration_error from failures[-1]
