"""Runner core shared by every host that drives the Cayu command sidecar.

The sidecar runs agent commands, owner fencing, command carry-over,
interactive stdio and port forwards behind one protocol. A runner built on
:class:`SidecarProtocolRunner` supplies how it authenticates to the sidecar,
what identifies its sandbox, and its error classes; everything else here is
common.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import re
import secrets
import uuid
from abc import abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from math import isfinite
from typing import Any, ClassVar, Literal, cast

from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import require_clean_nonblank
from cayu.runners._admission_probes import EXECUTABLE_AVAILABILITY_SCRIPT
from cayu.runners._cleanup import (
    RunnerCleanupPolicy,
    RunnerCleanupResult,
    cleanup_runner_command_with_diagnostic,
)
from cayu.runners._redacted_output import redact_completed_exec_result
from cayu.runners._sidecar_transport import (
    SIDECAR_ERRORS,
    SIDECAR_MAX_ENCODED_OUTPUT_BYTES,
    SIDECAR_MAX_OUTPUT_BYTES,
    SIDECAR_MAX_STDIN_BYTES,
    SIDECAR_PROTOCOL_VERSION,
    SidecarEndpointThrottled,
    SidecarEndpointTransientError,
    SidecarEndpointUnauthorized,
    SidecarErrors,
    SidecarOwnershipSuperseded,
    SidecarStaleOwnerEpoch,
)
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
    RunnerExecutionAdmissionObserver,
    RunnerInteractiveProcess,
    RunnerInteractiveProcessCapability,
    RunnerPortForward,
    RunnerPortForwardCapability,
    _clean_runner_preflight,
    _clear_preflight_traceback_frames,
    attach_cancellation_artifacts,
    copy_exec_command,
)
from cayu.vaults import SecretRedactor

SIDECAR_EXEC_TIMEOUT_GRACE_SECONDS = 5.0
SIDECAR_MIN_POLL_INTERVAL_SECONDS = 0.01
# Interactive stdio moves in bounded chunks. A stdout read long-polls the sidecar
# for at most this long, well inside one endpoint request timeout.
SIDECAR_INTERACTIVE_CHUNK_BYTES = 1024 * 1024
SIDECAR_INTERACTIVE_READ_WAIT_SECONDS = 10.0
# A forwarded connection moves in chunks no larger than this, so one chunk of
# model output reaches the guest as soon as the host service sends it.
SIDECAR_FORWARD_CHUNK_BYTES = 256 * 1024
SIDECAR_ADMISSION_PROBE_TIMEOUT_SECONDS = 5
_ADMISSION_PROBE_OUTPUT_LIMIT_BYTES = 1024
_FORWARD_METHODS = (
    "open_forward",
    "accept_forward",
    "read_forward",
    "write_forward",
    "close_forward_connection",
    "close_forward",
)
_OWNER_SUPERSEDED_CANCEL_REASON = "owner_superseded"
# The interactive process tree whose teardown stops only the command's process group.
_PROCESS_GROUP_TREE = "process_group"
_POLL_ATTEMPTS = 3
# A throttled endpoint call waits 0.25 s, then doubles to at most 4 s, for six attempts in all
# (five pauses, 7.75 s; up to 20 s when Retry-After asks for longer), never past the call's
# deadline. Only _endpoint_call retries a throttle; the outer transient-error retries re-raise
# it so this bound holds.
_THROTTLE_ATTEMPTS = 6
_THROTTLE_PAUSE_S = 0.25
_THROTTLE_MAX_PAUSE_S = 4.0
_CommandPollTask = asyncio.Task[CapturedAwaitableOutcome[Mapping[str, Any]]]
_PENDING_SIDECAR_POLLS: set[_CommandPollTask] = set()
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SidecarLimits:
    """The byte and time bounds one runner applies to sidecar commands."""

    max_output_bytes: int = SIDECAR_MAX_OUTPUT_BYTES
    max_encoded_output_bytes: int = SIDECAR_MAX_ENCODED_OUTPUT_BYTES
    max_stdin_bytes: int = SIDECAR_MAX_STDIN_BYTES
    exec_timeout_grace_s: float = SIDECAR_EXEC_TIMEOUT_GRACE_SECONDS


class SidecarProtocolRunner(Runner):
    """Execute commands through the Cayu command sidecar.

    A subclass sets the sidecar's ``endpoint``, the timeouts, cleanup
    policies and ``env_overlay`` the shared code reads, an endpoint transport
    in ``_endpoint_transport``, ``_owner_epoch``, ``_carry_requested`` and
    ``_env_overlay_secret_values_present``, and calls
    :meth:`_init_sidecar_state`. It then supplies the endpoint credential
    (:meth:`_endpoint_token`), what identifies its sandbox
    (:attr:`_resource_id`, :meth:`_admission_local_identity`), its admission
    observer, and its error classes and labels.
    """

    max_exec_output_bytes = SIDECAR_MAX_OUTPUT_BYTES
    _errors: ClassVar[SidecarErrors] = SIDECAR_ERRORS
    # Published as ``adapter`` in cleanup receipts and diagnostics.
    _adapter: ClassVar[str]
    # Names the runner in its own error messages.
    _runner_name: ClassVar[str]
    _logger: ClassVar[logging.Logger] = _LOGGER
    # Interactive process trees whose teardown this runner accepts; see
    # ``_SidecarInteractiveProcess.close``.
    _interactive_process_trees: ClassVar[frozenset[str]] = frozenset({"pid_namespace"})

    endpoint: str
    env_overlay: dict[str, str]
    poll_interval_s: float
    request_timeout_s: float
    cancel_timeout_s: float
    cancellation_cleanup: RunnerCleanupPolicy
    timeout_cleanup: RunnerCleanupPolicy
    _endpoint_transport: Any
    _owner_epoch: int | None
    _carry_requested: tuple[str, ...]
    _env_overlay_secret_values_present: bool

    def _init_sidecar_state(self) -> None:
        """Start with no claim, no carried commands and no health observed yet."""

        self._owner_epoch_verified = False
        # What the sidecar's health last advertised, and what this runner's claim took over.
        self._command_carry_supported = False
        self._carried_commands: tuple[str, ...] = ()
        # Set by ``detach_commands``: interrupted waits leave their guest commands running.
        self._commands_detached = False
        self._command_poll_tasks: dict[str, tuple[float, _CommandPollTask]] = {}
        # Memory-only bearer claim; the sidecar keeps only its digest.
        self._owner_claim = secrets.token_hex(32)
        self._owner_claimed = False
        self._owner_superseded = False
        self._owner_generation: int | None = None

    @property
    @abstractmethod
    def _resource_id(self) -> str:
        """The sandbox's identifier, the prefix of :attr:`owner_fence_generation`."""

    @abstractmethod
    async def _endpoint_token(self, *, force_refresh: bool = False) -> str:
        """The credential for the next endpoint call; ``force_refresh`` after a rejection."""

    @abstractmethod
    def _admission_local_identity(self) -> tuple[object, ...]:
        """Local configuration a retained admission candidate stays valid for."""

    @abstractmethod
    def _admission_observer(self, requirements) -> _SidecarAdmissionObserver:
        """A new admission observer of this runner for ``requirements``."""

    def _sidecar_limits(self) -> SidecarLimits:
        return SidecarLimits()

    def _needs_health_before_claim(self) -> bool:
        """Whether the first claim must wait for a health check that passed."""

        return self._owner_epoch is not None and not self._owner_epoch_verified

    def _observe_health(self, response: Mapping[str, Any], capabilities: object) -> None:
        """Check a healthy sidecar's response further; raise to refuse it."""

    def _detached_reason(self) -> str:
        return f"{self._errors.label} commands were detached for a successor owner"

    def execution_admission_observer(self, requirements):
        """Own live executable evidence for this exact sandbox and requirement set."""

        if not requirements.executable_names():
            return super().execution_admission_observer(requirements)
        return self._admission_observer(requirements)

    def execution_admission_candidate_for(self, requirements):
        if not requirements.executable_names():
            return self.execution_admission_candidate()
        return self._admission_observer(requirements).snapshot()

    async def refresh_execution_admission(self) -> None:
        """Refuse renewal for a sandbox that cannot execute; claims are request-scoped.

        Executable evidence is renewed by the request's admission observer,
        which re-reads the complete identity. This hook adds no claim.
        """

        self._require_admission_open()

    def _require_admission_open(self) -> None:
        if self._owner_superseded:
            raise self._errors.superseded(f"{self._errors.label} has a newer owner.")
        self._ensure_exec_open()

    def output_secret_values_present(self) -> bool:
        """Declare whether command output can contain runner-owned secret values."""

        return self._env_overlay_secret_values_present

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
        command_id: str | None = None,
    ) -> ExecResult:
        """Run one agent command to its end.

        ``command_id`` names the command in the sidecar instead of a random ID,
        so a successor can name it in ``carry_commands`` (1 to 128 of
        ``[A-Za-z0-9._:-]``). The caller keeps it unique within the sandbox:
        the sidecar answers a repeated ID with that command's existing state.
        """
        operation = self._exec(
            command,
            execution_profile="agent",
            output_redactor=None,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
            command_id=_validate_command_id(command_id),
        )
        del command, cwd, env, env_remove, timeout_s, stdin, output_limit_bytes
        return await operation

    @property
    def command_carry_supported(self) -> bool:
        """Whether the sidecar advertised ``carry_commands`` at its last health check.

        A runner holding commands a successor may carry over checks this before
        ``detach_commands``; without it, a successor's claim cancels them.
        """

        return self._command_carry_supported

    @property
    def carried_commands(self) -> tuple[str, ...]:
        """The commands this runner's claim took over from an earlier owner (``carry_commands``).

        Empty when none were asked for, the sidecar can't carry commands, or
        the claim's epoch wasn't strictly newer than every earlier one.
        """

        return self._carried_commands

    def detach_commands(self) -> None:
        """Stop serving this runner's commands without stopping them, for a successor to carry.

        From now on the runner never starts a command again (``reopen_exec``
        refuses and a resume doesn't reopen it), and an ``exec`` or
        ``resume_exec`` whose wait is cancelled or fails returns or raises
        without cancelling its guest command, which keeps running until a
        successor claims the sidecar. A successor that names it in
        ``carry_commands`` takes it over; any other claim cancels it, as claims
        always have. Interactive processes aren't carried over and still stop
        when closed. Cancel the waits still running after this call; the runner
        doesn't end them itself.
        """

        self._commands_detached = True
        self._close_exec(self._detached_reason())

    def reopen_exec(self) -> None:
        if self._commands_detached:
            raise RuntimeError(f"{self._runner_name} is closed: {self._detached_reason()}")
        super().reopen_exec()

    def _open_exec(self) -> None:
        super()._open_exec()
        if self._commands_detached:
            # Detaching is final: a resume doesn't make the runner run commands again.
            self._close_exec(self._detached_reason())

    async def resume_exec(self, command_id: str, *, timeout_s: int | None = None) -> ExecResult:
        """Wait for a command this runner carried over (``carried_commands``) and return its result.

        It behaves as the rest of an ``exec`` of that command would: cancelled,
        it stops the command; at its end the result is read and released.
        ``timeout_s`` bounds this wait (the command's own timeout, which the
        sidecar enforces, still applies); ``None`` waits for the command's end.
        A directly constructed successor that asked to carry the command claims
        ownership first, which carries it over.
        """

        claim_first = not self._owner_claimed and command_id in self._carry_requested
        if command_id not in self._carried_commands and not claim_first:
            raise ValueError("resume_exec takes only a command this runner carried over.")
        timeout = validate_timeout(timeout_s)
        self._ensure_exec_open()
        if claim_first:
            await self._claim_to_resume(command_id)
            if command_id not in self._carried_commands:
                raise ValueError("resume_exec takes only a command this runner carried over.")
        return await self._await_command(
            command_id,
            payload=None,
            timeout=timeout,
            output_redactor=None,
            output_limit=self._sidecar_limits().max_output_bytes,
        )

    async def _claim_to_resume(self, command_id: str) -> None:
        """Claim ownership, which carries ``command_id`` over, before resuming it.

        A cancellation waits for the claim's outcome: once the claim has carried
        the command, it is stopped as a cancelled wait for it would stop it.
        """

        outcome = await await_shielded_task_outcome(asyncio.ensure_future(self._claim_owner()))
        if isinstance(outcome.error, SidecarOwnershipSuperseded):
            self._mark_superseded()
        if outcome.cancellation is not None and command_id in self._carried_commands:
            cleanup = await self._cleanup_exec_command(
                handle=_SidecarCommandHandle(self, command_id),
                policy=self.cancellation_cleanup,
                start_acknowledged=True,
                cancellation=outcome.cancellation,
            )
            attach_cancellation_artifacts(outcome.cancellation, [cleanup.artifact])
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.error is not None:
            raise outcome.error

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
            raise TypeError(f"{self._runner_name} redactor must be a SecretRedactor.")
        operation = self._exec(
            command,
            execution_profile="agent",
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
        """Validate the complete sidecar request without provider activity."""

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
    ) -> tuple[ExecCommand, str, dict[str, str], int | None, str | None, int | None]:
        """Own one complete sidecar request before dispatch admission."""

        if type(command) is not ExecCommand:
            raise TypeError(f"{self._runner_name} command must be an ExecCommand.")
        self._ensure_exec_open()
        owned_command = copy_exec_command(command)
        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        env_overlay = copy_runner_env(self.env_overlay, inherit_env=False)
        environment = remove_runner_env(environment, env_remove)
        if env_overlay:
            # Applied last: enforced egress configuration must win over model env.
            environment.update(env_overlay)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        limits = self._sidecar_limits()
        if (
            standard_input is not None
            and len(standard_input.encode("utf-8")) > limits.max_stdin_bytes
        ):
            raise ValueError(
                f"{self._errors.label} stdin exceeds the sidecar limit of "
                f"{limits.max_stdin_bytes} bytes."
            )
        output_limit = validate_output_limit(output_limit_bytes)
        output_limit = min(
            limits.max_output_bytes,
            limits.max_output_bytes if output_limit is None else output_limit,
        )
        return owned_command, working_dir, environment, timeout, standard_input, output_limit

    async def _exec(
        self,
        command: ExecCommand,
        *,
        execution_profile: Literal["agent", "trusted"],
        output_redactor: SecretRedactor | None,
        cwd: str | None,
        env: dict[str, str] | None,
        env_remove: tuple[str, ...],
        timeout_s: int | None,
        stdin: str | None,
        output_limit_bytes: int | None,
        command_id: str | None = None,
    ) -> ExecResult:
        try:
            (
                owned_command,
                working_dir,
                environment,
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
            if command_id is None:
                command_id = f"cmd-{uuid.uuid4()}"
            payload: dict[str, Any] = {
                "execution_profile": execution_profile,
                "kind": owned_command.kind,
                "cwd": working_dir,
                "env": environment,
                "stdin_base64": (
                    base64.b64encode(standard_input.encode("utf-8")).decode("ascii")
                    if standard_input is not None
                    else None
                ),
                "timeout_s": timeout,
                "output_limit_bytes": output_limit,
                # The sidecar does not receive the workload-secret registry. When
                # host-side redaction is required, it must therefore omit a
                # pre-truncated channel whose missing suffix could complete a
                # secret. Ordinary/system executions retain their established
                # bounded-output contract.
                "omit_truncated_output": output_redactor is not None,
            }
            if owned_command.kind == "process":
                payload["argv"] = list(owned_command.argv or [])
            else:
                payload["shell"] = owned_command.shell
        except BaseException as error:
            _clear_preflight_traceback_frames(error)
            owned_command = None
            working_dir = ""
            environment = {}
            standard_input = None
            payload = {}
            cwd = None
            env = None
            env_remove = ()
            stdin = None
            output_redactor = None
            raise
        finally:
            del command
        return await self._await_command(
            command_id,
            payload=payload,
            timeout=timeout,
            output_redactor=output_redactor,
            output_limit=output_limit,
        )

    async def _await_command(
        self,
        command_id: str,
        *,
        payload: dict[str, Any] | None,
        timeout: int | None,
        output_redactor: SecretRedactor | None,
        output_limit: int | None,
    ) -> ExecResult:
        """Start the command (unless ``payload`` is None: it already runs) and wait for its end."""

        handle = _SidecarCommandHandle(self, command_id)
        start_acknowledged = payload is None
        release_after_read = False
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + timeout + self._sidecar_limits().exec_timeout_grace_s
            if timeout is not None
            else None
        )
        try:
            async with asyncio.timeout_at(deadline):
                if payload is not None:
                    await self._endpoint_start(command_id, payload)
                    start_acknowledged = True
                while True:
                    response = await self._endpoint_get(command_id)
                    state = _required_response_string(response, "state", self._errors)
                    if state in {"completed", "cancelled", "failed"}:
                        if response.get("cancel_reason") == _OWNER_SUPERSEDED_CANCEL_REASON:
                            self._mark_superseded()
                            raise self._errors.superseded(
                                f"A newer {self._errors.label} owner cancelled this command."
                            )
                        result = _exec_result(response, self._errors, self._sidecar_limits())
                        del response
                        release_after_read = True
                        if output_redactor is not None:
                            result = redact_completed_exec_result(
                                result,
                                redactor=output_redactor,
                                output_limit_bytes=output_limit,
                                omit_pretruncated=True,
                            )
                        if result.timed_out:
                            cleanup = await self._cleanup_exec_command(
                                handle=handle,
                                policy=self.timeout_cleanup,
                                start_acknowledged=start_acknowledged,
                            )
                            result.artifacts.append(cleanup.artifact)
                        break
                    if state not in {"accepted", "running"}:
                        raise self._errors.protocol(
                            f"{self._errors.label} command returned unsupported state: {state}"
                        )
                    await asyncio.sleep(
                        max(
                            self.poll_interval_s,
                            SIDECAR_MIN_POLL_INTERVAL_SECONDS,
                        )
                    )
        except asyncio.CancelledError as exc:
            cleanup = await self._cleanup_exec_command(
                handle=handle,
                policy=self.cancellation_cleanup,
                start_acknowledged=start_acknowledged,
                cancellation=exc,
            )
            attach_cancellation_artifacts(exc, [cleanup.artifact])
            raise
        except TimeoutError:
            cleanup = await self._cleanup_exec_command(
                handle=handle,
                policy=self.timeout_cleanup,
                start_acknowledged=start_acknowledged,
            )
            result = await self._host_timeout_result(
                command_id,
                cleanup,
                output_redactor=output_redactor,
                output_limit_bytes=output_limit,
            )
            result.artifacts.append(cleanup.artifact)
            return result
        except Exception:
            await self._cleanup_exec_command(
                handle=handle,
                policy=self.cancellation_cleanup,
                start_acknowledged=start_acknowledged,
            )
            raise
        if release_after_read:
            await self._release_command(command_id)
        return result

    async def _release_command(self, command_id: str) -> None:
        """Ask the sidecar to drop a terminal result the host has now read.

        Browser profile and upload traffic carries cookies and file contents, so
        the delivered output should not stay in sidecar memory (or in a suspend
        snapshot) until the result TTL expires. Failure leaves that TTL as the
        bound and never changes the command's outcome.
        """

        if not callable(getattr(self._endpoint_transport, "release_command", None)):
            return
        try:
            await self._endpoint_call(
                "release_command", command_id=command_id, **self._carried_owner(command_id)
            )
        except Exception:
            self._logger.debug("%s command release failed", self._errors.label, exc_info=True)

    async def _cleanup_exec_command(
        self,
        *,
        handle: _SidecarCommandHandle,
        policy: RunnerCleanupPolicy,
        start_acknowledged: bool,
        cancellation: asyncio.CancelledError | None = None,
    ) -> RunnerCleanupResult:
        if self._commands_detached:
            # Handed to a successor (``detach_commands``): its claim decides the command's fate.
            policy = "none"
        if not start_acknowledged and policy == "none":
            self._poison_exec()
        cleanup = await self._settle_command_cleanup(
            lambda: cleanup_runner_command_with_diagnostic(
                self,
                handle=handle,
                adapter=self._adapter,
                timeout_s=self.cancel_timeout_s,
                policy=policy,
            ),
            adapter=self._adapter,
            timeout_s=self.cancel_timeout_s,
            policy=policy,
            cancellation=cancellation,
        )
        self._apply_cleanup_result(cleanup)
        if not start_acknowledged and policy == "none":
            self._close_exec(
                f"{self._errors.label} command start was not acknowledged; command state is unknown"
            )
        return cleanup

    async def _host_timeout_result(
        self,
        command_id: str,
        cleanup: RunnerCleanupResult,
        *,
        output_redactor: SecretRedactor | None,
        output_limit_bytes: int | None,
    ) -> ExecResult:
        result = ExecResult(exit_code=-9, timed_out=True)
        artifact = cleanup.artifact
        if artifact.get("action") != "kill_command" or artifact.get("status") != "completed":
            return result
        try:
            response = await asyncio.wait_for(
                self._endpoint_get(command_id),
                timeout=self.cancel_timeout_s,
            )
            state = _required_response_string(response, "state", self._errors)
            if state not in {"completed", "cancelled", "failed"}:
                return result
            terminal = _exec_result(response, self._errors, self._sidecar_limits())
            del response
            if output_redactor is not None:
                terminal = redact_completed_exec_result(
                    terminal,
                    redactor=output_redactor,
                    output_limit_bytes=output_limit_bytes,
                    omit_pretruncated=True,
                )
        except Exception:
            return result
        await self._release_command(command_id)
        return ExecResult(
            stdout=terminal.stdout,
            stderr=terminal.stderr,
            exit_code=-9,
            timed_out=True,
            stdout_truncated=terminal.stdout_truncated,
            stderr_truncated=terminal.stderr_truncated,
            stdout_bytes=terminal.stdout_bytes,
            stderr_bytes=terminal.stderr_bytes,
        )

    async def _endpoint_health(self) -> None:
        response = await self._endpoint_call("health")
        status = _required_response_string(response, "status", self._errors)
        if status != "ok":
            raise self._errors.protocol(f"{self._errors.label} health response was not ready.")
        version = response.get("protocol_version")
        if version != SIDECAR_PROTOCOL_VERSION:
            reported_version = version if type(version) is str else repr(version)
            raise self._errors.version_mismatch(
                f"{self._errors.label} sidecar protocol version mismatch: "
                f"expected {SIDECAR_PROTOCOL_VERSION}, got {reported_version}."
            )

        capabilities = response.get("capabilities")
        if self._owner_epoch is not None and (
            not isinstance(capabilities, list) or "owner_epoch" not in capabilities
        ):
            raise self._errors.version_mismatch(
                f"{self._errors.label} sidecar does not advertise owner_epoch support."
            )

        self._observe_health(response, capabilities)
        self._command_carry_supported = (
            isinstance(capabilities, list) and "carry_commands" in capabilities
        )
        self._owner_epoch_verified = True

    async def _claim_owner(self) -> None:
        """Become the sidecar's only command owner; later claims fence this runner.

        A runner claims once. Readiness after resume only confirms the existing
        claim, so a stale runner can never take the sidecar back from its
        successor; a sidecar that lost its fence also reports it as superseded.
        """

        if self._owner_superseded:
            raise self._errors.superseded(f"{self._errors.label} has a newer owner.")
        if self._owner_claimed:
            response = await self._endpoint_call("check_owner", claim_id=self._owner_claim)
            if response.get("current") is not True:
                self._mark_superseded()
                raise self._errors.superseded(f"{self._errors.label} has a newer owner.")
            return
        if self._needs_health_before_claim():
            await self._endpoint_health()
        options: dict[str, Any] = (
            {} if self._owner_epoch is None else {"owner_epoch": self._owner_epoch}
        )
        # Only a sidecar that advertises it is asked; any other cancels them as before.
        carry = self._carry_requested if self._command_carry_supported else ()
        if carry:
            options["carry_commands"] = list(carry)
        try:
            response = await self._endpoint_call(
                "claim_owner", claim_id=self._owner_claim, **options
            )
        except SidecarStaleOwnerEpoch:
            self._mark_superseded()
            raise
        generation = response.get("generation")
        if type(generation) is not int or generation <= 0:
            raise self._errors.protocol(f"{self._errors.label} owner claim returned no generation.")
        carried: tuple[str, ...] = ()
        if carry:
            reported = response.get("carried_commands")
            if (
                not isinstance(reported, list)
                or any(type(command_id) is not str for command_id in reported)
                or not set(reported) <= set(carry)
            ):
                raise self._errors.protocol(
                    f"{self._errors.label} owner claim returned invalid carried commands."
                )
            carried = tuple(dict.fromkeys(cast("list[str]", reported)))
        self._owner_generation = generation
        self._carried_commands = carried
        self._owner_claimed = True

    def interactive_process_capability(self) -> RunnerInteractiveProcessCapability | None:
        """Stream a process's stdio through sidecar protocol 6.

        The sidecar must report a process tree this runner accepts
        (``_interactive_process_trees``). Endpoint transports without the
        interactive methods report no capability.
        """

        transport = self._endpoint_transport
        if not all(
            callable(getattr(transport, name, None))
            for name in ("write_command_stdin", "read_command_stdout")
        ):
            return None
        return _SidecarInteractiveProcessCapability(self)

    def port_forward_capability(self) -> RunnerPortForwardCapability | None:
        """Relay guest connections to a host service through sidecar protocol 7.

        The sidecar listens inside the sandbox; each connection travels over
        the owner-fenced sidecar endpoint, and the host opens the matching
        connection to the target. Endpoint transports without the forward
        methods report no capability.
        """

        transport = self._endpoint_transport
        if not all(callable(getattr(transport, name, None)) for name in _FORWARD_METHODS):
            return None
        return _SidecarPortForwardCapability(self)

    @property
    def owner_fence_generation(self) -> str | None:
        """Non-secret identity of this runner's current sidecar owner claim.

        ``None`` until the runner has claimed the sidecar or once any guest
        operation reported that a successor superseded it. Because every guest
        operation is fenced, a successor that claims inside a caller's window
        makes some operation in that window fail and clears this value.
        """

        if not self._owner_claimed or self._owner_superseded or self._owner_generation is None:
            return None
        return f"{self._resource_id}:{self._owner_generation}"

    @property
    def owner_superseded(self) -> bool:
        """Whether a guest operation has reported that a successor owns this sidecar.

        False before the first claim and while ownership has not been reported
        lost. Once true, it stays true for this runner; do not retry or reattach
        it. This is observed state, not a live ownership check. Unlike
        :attr:`owner_fence_generation`, it distinguishes an unclaimed runner
        from one that has learned its claim was superseded.
        """

        return self._owner_superseded

    def _mark_superseded(self) -> None:
        self._owner_superseded = True
        self._poison_exec(f"{self._errors.label} ownership was superseded")

    async def _endpoint_start(self, command_id: str, payload: dict[str, Any]) -> Mapping[str, Any]:
        try:
            if not self._owner_claimed:
                # A directly constructed runner claims before its first command.
                await self._claim_owner()
            response = await self._endpoint_call(
                "start_command",
                command_id=command_id,
                payload={**payload, "owner_claim": self._owner_claim},
            )
        except SidecarOwnershipSuperseded:
            self._mark_superseded()
            raise
        returned_id = _required_response_string(response, "command_id", self._errors)
        if returned_id != command_id:
            raise self._errors.protocol(
                f"{self._errors.label} start returned the wrong command id."
            )
        return response

    async def _endpoint_get(self, command_id: str) -> Mapping[str, Any]:
        loop = asyncio.get_running_loop()
        pending = self._command_poll_tasks.get(command_id)
        if pending is None:
            deadline = loop.time() + self.request_timeout_s
            task = asyncio.create_task(
                capture_awaitable_outcome(
                    lambda: self._poll_command_until_deadline(command_id, deadline)
                )
            )
            self._command_poll_tasks[command_id] = (deadline, task)
            _PENDING_SIDECAR_POLLS.add(task)

            def retire(completed: _CommandPollTask) -> None:
                _PENDING_SIDECAR_POLLS.discard(completed)
                if self._command_poll_tasks.get(command_id) == (deadline, completed):
                    del self._command_poll_tasks[command_id]

            task.add_done_callback(retire)
        else:
            deadline, task = pending
        outcome = await await_shielded_task_outcome(
            task,
            timeout_s=max(0, deadline - loop.time()),
            timeout_after_cancellation_s=0,
        )
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
        )
        if outcome.cancellation is not None:
            raise outcome.cancellation
        if outcome.timed_out:
            raise TimeoutError(f"{self._errors.label} command-state poll exceeded its deadline.")
        captured = outcome.result
        failure = outcome.error if captured is None else captured.error
        if isinstance(failure, asyncio.CancelledError):
            raise self._errors.error(
                f"{self._errors.label} command-state read was cancelled by its dependency."
            )
        if failure is not None:
            raise failure
        if captured is None or captured.result is None:
            raise self._errors.protocol(
                f"{self._errors.label} command-state read returned no result."
            )
        return captured.result

    async def _poll_command_until_deadline(
        self, command_id: str, deadline: float
    ) -> Mapping[str, Any]:
        loop = asyncio.get_running_loop()
        for attempt in range(_POLL_ATTEMPTS):
            if loop.time() >= deadline:
                raise TimeoutError(
                    f"{self._errors.label} command-state poll exceeded its deadline."
                )
            try:
                result = await self._endpoint_call(
                    "get_command", deadline=deadline, command_id=command_id
                )
            except SidecarEndpointThrottled:
                raise
            except SidecarEndpointTransientError:
                if attempt == _POLL_ATTEMPTS - 1:
                    raise
            else:
                if loop.time() >= deadline:
                    raise TimeoutError(
                        f"{self._errors.label} command-state poll exceeded its deadline."
                    )
                return result
            await asyncio.sleep(min(0.05 * (2**attempt), max(0, deadline - loop.time())))
        raise AssertionError("unreachable command-state retry loop")

    async def _endpoint_cancel(self, command_id: str) -> Mapping[str, Any]:
        return await self._endpoint_call(
            "cancel_command", command_id=command_id, **self._carried_owner(command_id)
        )

    def _carried_owner(self, command_id: str) -> dict[str, Any]:
        # The sidecar lets only the current owner cancel or release a carried command.
        if command_id in self._carried_commands:
            return {"owner_claim": self._owner_claim}
        return {}

    async def _idempotent_owner_call(self, method_name: str, **kwargs: Any) -> Mapping[str, Any]:
        for attempt in range(_POLL_ATTEMPTS):
            try:
                return await self._endpoint_call(
                    method_name, owner_claim=self._owner_claim, **kwargs
                )
            except SidecarOwnershipSuperseded:
                self._mark_superseded()
                raise
            except SidecarEndpointThrottled:
                raise
            except SidecarEndpointTransientError:
                if attempt == _POLL_ATTEMPTS - 1:
                    raise
            await asyncio.sleep(0.05 * (2**attempt))
        raise AssertionError("unreachable owner-call retry loop")

    async def _endpoint_call(
        self, method_name: str, *, deadline: float | None = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        method = getattr(self._endpoint_transport, method_name)
        rejected = False
        refresh_token = False
        throttled = 0
        loop = asyncio.get_running_loop()
        while True:
            token = await self._endpoint_token(force_refresh=refresh_token)
            # Refresh once per rejection; a throttle retry reuses the refreshed token.
            refresh_token = False
            timeout_s = self.request_timeout_s
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"{self._errors.label} command-state poll exceeded its deadline."
                    )
                timeout_s = min(timeout_s, remaining)
            try:
                result = await method(
                    endpoint=self.endpoint,
                    token=token,
                    timeout_s=timeout_s,
                    **kwargs,
                )
            except SidecarEndpointUnauthorized:
                # One fresh token per call, so a token rotated mid-command doesn't fail it.
                if rejected:
                    raise
                rejected = True
                refresh_token = True
                continue
            except SidecarEndpointThrottled as exc:
                # Refused before the guest acted, so sending it again can't repeat any effect.
                throttled += 1
                if throttled >= _THROTTLE_ATTEMPTS:
                    raise
                pause = min(
                    _THROTTLE_MAX_PAUSE_S,
                    max(
                        exc.retry_after_s or 0.0,
                        _THROTTLE_PAUSE_S * 2 ** (throttled - 1),
                    ),
                )
                if deadline is not None and loop.time() + pause >= deadline:
                    raise
                await asyncio.sleep(pause)
                continue
            if not isinstance(result, Mapping):
                raise self._errors.protocol(
                    f"{self._errors.label} {method_name} response must be an object."
                )
            return result


class _SidecarInteractiveProcessCapability(RunnerInteractiveProcessCapability):
    """Interactive processes in one sandbox; the runner keeps its own lifecycle."""

    def __init__(self, runner: SidecarProtocolRunner) -> None:
        self._runner = runner
        self._errors = runner._errors

    @property
    def secret_values(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(copy_runner_env(self._runner.env_overlay, inherit_env=False).values())
        )

    async def open_interactive_process(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> RunnerInteractiveProcess:
        runner = self._runner
        owned_command, working_dir, environment, timeout, _stdin, _limit = (
            runner._prepare_exec_request(
                command,
                cwd=cwd,
                env=env,
                env_remove=(),
                timeout_s=timeout_s,
                stdin=None,
                output_limit_bytes=None,
            )
        )
        payload: dict[str, Any] = {
            "execution_profile": "agent",
            "kind": owned_command.kind,
            "cwd": working_dir,
            "env": environment,
            "stdin_base64": None,
            "timeout_s": timeout,
            "interactive": True,
        }
        if owned_command.kind == "process":
            payload["argv"] = list(owned_command.argv or [])
        else:
            payload["shell"] = owned_command.shell
        process = _SidecarInteractiveProcess(
            runner, f"cmd-{uuid.uuid4()}", secret_values=self.secret_values
        )
        del environment
        try:
            response = await runner._endpoint_start(process.command_id, payload)
            if response.get("process_tree") not in runner._interactive_process_trees:
                raise self._errors.protocol(
                    f"{self._errors.label} sidecar cannot confirm teardown of an interactive "
                    "process tree; rebuild the image from this Cayu release's sidecar."
                )
        except BaseException as error:
            # The start may have reached the sidecar. Stop it before reporting,
            # even if the caller is being cancelled.
            outcome = await await_shielded_task_outcome(asyncio.ensure_future(process.close()))
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
            )
            if outcome.error is not None:
                self.retain_failed_start(error, process)
                runner._poison_exec(
                    f"{self._errors.label} interactive command cleanup is unconfirmed"
                )
            raise
        finally:
            del payload
        return process


class _SidecarInteractiveProcess(RunnerInteractiveProcess):
    """Offset-addressed stdio of one sidecar interactive command.

    Every stdin write and stdout read names its byte offset, so a request that
    timed out is retried without duplicating or losing bytes.
    """

    def __init__(
        self, runner: SidecarProtocolRunner, command_id: str, *, secret_values: tuple[str, ...] = ()
    ) -> None:
        self._runner = runner
        self._errors = runner._errors
        self.command_id = command_id
        self._secret_values = secret_values
        self._stdin_offset = 0
        self._stdout_offset = 0
        self._stdin_closed = False
        self._stdout_eof = False
        self._terminal: Mapping[str, Any] | None = None
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()

    @property
    def secret_values(self) -> tuple[str, ...]:
        return self._secret_values

    async def write(self, data: bytes) -> None:
        async with self._write_lock:
            if self._stdin_closed:
                raise BrokenPipeError("Interactive process stdin is closed.")
            data = bytes(data)
            for start in range(0, len(data), SIDECAR_INTERACTIVE_CHUNK_BYTES):
                chunk = data[start : start + SIDECAR_INTERACTIVE_CHUNK_BYTES]
                await self._write_chunk(chunk, eof=False)
                if self._stdin_closed:
                    raise BrokenPipeError("Interactive process stdin is closed.")

    async def close_stdin(self) -> None:
        async with self._write_lock:
            if not self._stdin_closed:
                await self._write_chunk(b"", eof=True)
                self._stdin_closed = True

    async def _write_chunk(self, chunk: bytes, *, eof: bool) -> None:
        response = await self._stdio_call(
            "write_command_stdin",
            offset=self._stdin_offset,
            data_base64=base64.b64encode(chunk).decode("ascii"),
            eof=eof,
        )
        closed = response.get("closed")
        offset = response.get("offset")
        if type(closed) is not bool or type(offset) is not int:
            raise self._errors.protocol(f"{self._errors.label} stdin response is malformed.")
        if closed:
            self._stdin_closed = True
        elif offset != self._stdin_offset + len(chunk):
            raise self._errors.protocol(f"{self._errors.label} stdin offset is inconsistent.")
        self._stdin_offset = offset

    async def read(self) -> bytes:
        async with self._read_lock:
            wait_s = min(SIDECAR_INTERACTIVE_READ_WAIT_SECONDS, self._runner.request_timeout_s / 2)
            while not self._stdout_eof:
                response = await self._stdio_call(
                    "read_command_stdout",
                    offset=self._stdout_offset,
                    max_bytes=SIDECAR_INTERACTIVE_CHUNK_BYTES,
                    wait_s=wait_s,
                )
                encoded = response.get("data_base64")
                eof = response.get("eof")
                if (
                    response.get("offset") != self._stdout_offset
                    or type(encoded) is not str
                    or type(eof) is not bool
                    or len(encoded) > 4 * ((SIDECAR_INTERACTIVE_CHUNK_BYTES + 2) // 3)
                ):
                    raise self._errors.protocol(
                        f"{self._errors.label} stdout response is malformed."
                    )
                try:
                    data = base64.b64decode(encoded, validate=True)
                except ValueError:
                    raise self._errors.protocol(
                        f"{self._errors.label} stdout response is malformed."
                    ) from None
                self._stdout_offset += len(data)
                self._stdout_eof = eof
                if data:
                    return data
            return b""

    async def wait(self) -> int:
        delay = max(self._runner.poll_interval_s, SIDECAR_MIN_POLL_INTERVAL_SECONDS)
        while self._terminal is None:
            response = await self._runner._endpoint_get(self.command_id)
            if _required_response_string(response, "state", self._errors) in {
                "completed",
                "cancelled",
                "failed",
            }:
                self._terminal = self._settled(response)
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, 1.0)
        exit_code = self._settled(self._terminal).get("exit_code")
        if type(exit_code) is not int:
            raise self._errors.protocol(f"{self._errors.label} command returned no exit code.")
        return exit_code

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            runner = self._runner
            self._stdin_closed = True
            if self._terminal is None or self._terminal.get("process_tree_stopped") is not True:
                response = await runner._endpoint_cancel(self.command_id)
                state = _required_response_string(response, "state", self._errors)
                deadline = asyncio.get_running_loop().time() + 2 * (runner.cancel_timeout_s or 5.0)
                while state not in {"completed", "cancelled", "failed", "not_found"}:
                    if asyncio.get_running_loop().time() >= deadline:
                        raise self._errors.error(
                            f"{self._errors.label} interactive command did not stop in time."
                        )
                    await asyncio.sleep(max(runner.poll_interval_s, 0.05))
                    response = await runner._endpoint_get(self.command_id)
                    state = _required_response_string(response, "state", self._errors)
                if state == "not_found":
                    # The start never reached the sidecar: the cancellation tombstone
                    # it just recorded keeps a delayed start from ever running.
                    self._closed = True
                    return
                self._terminal = self._settled(response, cleanup=True)
            if not _process_tree_settled(self._terminal, runner._interactive_process_trees):
                raise self._errors.error(
                    f"{self._errors.label} could not confirm that the interactive process tree exited."
                )
            self._closed = True
        await runner._release_command(self.command_id)

    def _settled(self, response: Mapping[str, Any], *, cleanup: bool = False) -> Mapping[str, Any]:
        if response.get("cancel_reason") == _OWNER_SUPERSEDED_CANCEL_REASON:
            self._runner._mark_superseded()
            if not cleanup:
                raise self._errors.superseded(
                    f"A newer {self._errors.label} owner cancelled this command."
                )
        return {
            key: response[key]
            for key in (
                "state",
                "exit_code",
                "process_tree",
                "process_tree_stopped",
                "cancel_reason",
            )
            if key in response
        }

    async def _stdio_call(self, method_name: str, **kwargs: Any) -> Mapping[str, Any]:
        return await self._runner._idempotent_owner_call(
            method_name, command_id=self.command_id, **kwargs
        )


class _SidecarPortForwardCapability(RunnerPortForwardCapability):
    """Guest listeners in one sandbox; the runner keeps its own lifecycle."""

    def __init__(self, runner: SidecarProtocolRunner) -> None:
        self._runner = runner
        self._errors = runner._errors

    async def open_port_forward(
        self,
        *,
        host: str,
        port: int,
        timeout_s: int | None = None,
    ) -> RunnerPortForward:
        runner = self._runner
        target_host = require_clean_nonblank(host, "host")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Port forward target port must be an integer from 1 to 65535.")
        lifetime = validate_timeout(timeout_s)
        runner._ensure_exec_open()
        if not runner._owner_claimed:
            # A directly constructed runner claims before its first guest operation.
            try:
                await runner._claim_owner()
            except SidecarOwnershipSuperseded:
                runner._mark_superseded()
                raise
        forward = _SidecarPortForward(runner, f"fwd-{uuid.uuid4()}", (target_host, port))
        try:
            response = await forward._call("open_forward", lifetime_s=lifetime)
            guest_host = response.get("host")
            guest_port = response.get("port")
            if (
                response.get("forward_id") != forward.forward_id
                or type(guest_host) is not str
                or not guest_host
                or type(guest_port) is not int
                or not 1 <= guest_port <= 65535
            ):
                raise self._errors.protocol(f"{self._errors.label} forward response is malformed.")
        except BaseException as error:
            # The open may have reached the sidecar. Close it, or leave a
            # tombstone that refuses a delayed open, before reporting.
            outcome = await await_shielded_task_outcome(asyncio.ensure_future(forward.close()))
            restore_task_cancellation_requests(
                outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
            )
            if outcome.error is not None:
                self.retain_failed_open(error, forward)
            raise
        forward._start(guest_host, guest_port)
        return forward


class _SidecarPortForward(RunnerPortForward):
    """Host half of one sidecar forward: accepts guest connections and pumps their bytes.

    Each guest connection gets a host connection to the target. Guest bytes are
    long-polled by offset and host bytes are written by offset, so a request
    that timed out is retried without duplicating or losing bytes.
    """

    def __init__(
        self, runner: SidecarProtocolRunner, forward_id: str, target: tuple[str, int]
    ) -> None:
        self._runner = runner
        self._errors = runner._errors
        self.forward_id = forward_id
        self.target = target
        self._guest_host = ""
        self._guest_port = 0
        self._accepting: asyncio.Task[None] | None = None
        self._connections: set[asyncio.Task[None]] = set()
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def guest_host(self) -> str:
        return self._guest_host

    @property
    def guest_port(self) -> int:
        return self._guest_port

    def _start(self, guest_host: str, guest_port: int) -> None:
        self._guest_host = guest_host
        self._guest_port = guest_port
        self._accepting = asyncio.create_task(self._accept())

    def _wait_s(self) -> float:
        return min(SIDECAR_INTERACTIVE_READ_WAIT_SECONDS, self._runner.request_timeout_s / 2)

    async def _accept(self) -> None:
        after = 0
        while not self._closing:
            try:
                response = await self._call("accept_forward", after=after, wait_s=self._wait_s())
            except SidecarOwnershipSuperseded:
                return
            except SidecarEndpointTransientError:
                # The cursor makes polling safe to resume after an endpoint outage.
                await asyncio.sleep(0.5)
                continue
            except Exception:
                self._runner._logger.warning(
                    "%s forward stopped accepting connections.", self._errors.label
                )
                return
            connections = response.get("connections")
            if type(connections) is not list or any(
                type(identifier) is not int for identifier in connections
            ):
                self._runner._logger.warning(
                    "%s forward accept response is malformed.", self._errors.label
                )
                return
            for identifier in connections:
                if identifier > after:
                    task = asyncio.create_task(self._relay(identifier))
                    self._connections.add(task)
                    task.add_done_callback(self._connections.discard)
            after = max([after, *connections])
            if response.get("closed") is True:
                return

    async def _relay(self, connection: int) -> None:
        writer: asyncio.StreamWriter | None = None
        try:
            try:
                reader, writer = await asyncio.open_connection(*self.target)
            except OSError:
                return
            guest_to_host = asyncio.create_task(self._guest_to_host(connection, writer))
            host_to_guest = asyncio.create_task(self._host_to_guest(connection, reader))
            pending: set[asyncio.Task[None]] = {guest_to_host, host_to_guest}
            try:
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    if any(task.exception() is not None for task in done):
                        break
            finally:
                for task in (guest_to_host, host_to_guest):
                    task.cancel()
                await asyncio.gather(guest_to_host, host_to_guest, return_exceptions=True)
        except Exception:
            self._runner._logger.debug(
                "%s forwarded connection ended with an error.", self._errors.label
            )
        finally:
            if writer is not None:
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            await self._close_connection(connection)

    async def _close_connection(self, connection: int) -> None:
        # Keep the relay task owned until its guest slot is released. Losing
        # this obligation after an endpoint outage eventually fills the listener.
        while not self._closing:
            try:
                response = await self._call("close_forward_connection", connection=connection)
                if (
                    type(response.get("connection")) is not int
                    or response.get("connection") != connection
                    or response.get("closed") is not True
                ):
                    raise self._errors.protocol(
                        f"{self._errors.label} did not confirm the forwarded connection closed."
                    )
                return
            except SidecarOwnershipSuperseded:
                # Ownership replacement closes the old owner's entire forward.
                return
            except Exception:
                self._runner._logger.debug(
                    "%s forwarded connection cleanup will retry.", self._errors.label
                )
            await asyncio.sleep(0.5)

    async def _guest_to_host(self, connection: int, writer: asyncio.StreamWriter) -> None:
        offset = 0
        while True:
            response = await self._call(
                "read_forward",
                connection=connection,
                offset=offset,
                max_bytes=SIDECAR_FORWARD_CHUNK_BYTES,
                wait_s=self._wait_s(),
            )
            encoded = response.get("data_base64")
            eof = response.get("eof")
            if (
                response.get("offset") != offset
                or type(encoded) is not str
                or type(eof) is not bool
                or len(encoded) > 4 * ((SIDECAR_FORWARD_CHUNK_BYTES + 2) // 3)
            ):
                raise self._errors.protocol(f"{self._errors.label} forward read is malformed.")
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError:
                raise self._errors.protocol(
                    f"{self._errors.label} forward read is malformed."
                ) from None
            offset += len(data)
            if data:
                writer.write(data)
                await writer.drain()
            if eof:
                if writer.can_write_eof():
                    writer.write_eof()
                return

    async def _host_to_guest(self, connection: int, reader: asyncio.StreamReader) -> None:
        offset = 0
        while True:
            data = await reader.read(SIDECAR_FORWARD_CHUNK_BYTES)
            response = await self._call(
                "write_forward",
                connection=connection,
                offset=offset,
                data_base64=base64.b64encode(data).decode("ascii"),
                eof=not data,
            )
            closed = response.get("closed")
            written = response.get("offset")
            if type(closed) is not bool or type(written) is not int:
                raise self._errors.protocol(f"{self._errors.label} forward write is malformed.")
            if closed:
                return
            if written != offset + len(data):
                raise self._errors.protocol(f"{self._errors.label} forward offset is inconsistent.")
            offset = written
            if not data:
                return

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            tasks = tuple(self._connections)
            if self._accepting is not None:
                tasks = (self._accepting, *tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                response = await self._call("close_forward")
            except SidecarOwnershipSuperseded:
                # A bare refusal does not say whether the takeover's teardown
                # of this forward finished, so it is not proof of closure.
                raise self._errors.error(
                    f"{self._errors.label} could not confirm the forward closed after a newer owner "
                    "claimed it; rebuild the image from this Cayu release's sidecar."
                ) from None
            superseded = response.get("owner_superseded") is True
            if superseded:
                self._runner._mark_superseded()
            closed = response.get("closed")
            if closed is not True:
                if superseded and closed is False:
                    raise self._errors.error(
                        f"{self._errors.label} could not confirm the forward closed: the newer "
                        "owner's takeover has not finished tearing it down."
                    )
                raise self._errors.protocol(
                    f"{self._errors.label} did not confirm the forward closed."
                )
            self._closed = True

    async def _call(self, method_name: str, **kwargs: Any) -> Mapping[str, Any]:
        return await self._runner._idempotent_owner_call(
            method_name, forward_id=self.forward_id, **kwargs
        )


class _SidecarCommandHandle:
    def __init__(self, runner: SidecarProtocolRunner, command_id: str) -> None:
        self.runner = runner
        self.command_id = command_id

    async def kill(self) -> None:
        errors = self.runner._errors
        response = await self.runner._endpoint_cancel(self.command_id)
        state = _required_response_string(response, "state", errors)
        if state not in {"cancelled", "completed", "failed", "not_found"}:
            raise errors.protocol(
                f"{errors.label} cancellation did not reach a terminal state: {state}"
            )


@dataclass
class _SidecarAdmissionState:
    candidate: Any = None
    identity: Any = None


def _process_tree_settled(terminal: Mapping[str, Any], accepted: frozenset[str]) -> bool:
    """Whether a stopped interactive command's teardown is as complete as the runner accepts.

    A PID namespace proves every descendant exited (``process_tree_stopped``).
    A runner that accepts process-group teardown takes the end of the command
    and the signalled process group instead: a descendant that left the group
    is not covered.
    """

    if terminal.get("process_tree_stopped") is True:
        return True
    return _PROCESS_GROUP_TREE in accepted and terminal.get("process_tree") == _PROCESS_GROUP_TREE


@dataclass(frozen=True, repr=False)
class _SidecarAdmissionObserver(RunnerExecutionAdmissionObserver):
    """Probe executables through the agent lane of one exact, unchanged sandbox.

    Identity is read before and after the probes (``_observe_identity``); any
    change fails closed. Probes use ``runner.exec``, so every dispatched
    command is settled, or the runner is fenced, by the runner's own command
    cleanup before this observer returns or raises.
    """

    runner: SidecarProtocolRunner
    state: _SidecarAdmissionState = field(default_factory=_SidecarAdmissionState)

    # Names the candidate and its evidence subject.
    _candidate_name: ClassVar[str]

    def _fingerprint(self, identity: Any) -> str:
        raise NotImplementedError

    async def _observe_identity(self) -> Any:
        """One complete identity observation; it must carry the runner's ``local`` identity."""

        raise NotImplementedError

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _candidate(self, identity: Any, claims=()):
        from cayu.environments.admission import (
            ExecutionAdmissionCandidate,
            ExecutionCapabilityEvidence,
            ExecutionToolRequirementEvidence,
        )

        fingerprint = self._fingerprint(identity)
        return ExecutionAdmissionCandidate(
            candidate=self._candidate_name,
            evidence=ExecutionCapabilityEvidence(
                subject=self._candidate_name,
                unclaimed_reason_code="security_unclaimed",
                environment_fingerprint=fingerprint,
                tool_requirements=ExecutionToolRequirementEvidence(
                    environment_fingerprint=fingerprint,
                    executables=claims,
                ),
            ),
        )

    def snapshot(self):
        from cayu.environments.admission import ExecutionExecutableEvidence

        self.runner._require_admission_open()
        identity = self.state.identity
        candidate = self.state.candidate
        if (
            candidate is not None
            and identity is not None
            and identity.local == self.runner._admission_local_identity()
        ):
            return candidate
        probes = {probe.executable: probe for probe in self.requirements.executable_probes()}
        return self._candidate(
            None,
            tuple(
                ExecutionExecutableEvidence(
                    executable=name,
                    state="declared",
                    requirement_fingerprint=None
                    if name not in probes
                    else probes[name].fingerprint,
                )
                for name in self.requirements.executable_names()
            ),
        )

    async def _probe(self, command: ExecCommand) -> int:
        result = await self.runner.exec(
            command,
            timeout_s=SIDECAR_ADMISSION_PROBE_TIMEOUT_SECONDS,
            output_limit_bytes=_ADMISSION_PROBE_OUTPUT_LIMIT_BYTES,
        )
        if result.timed_out or result.cancelled:
            raise self.runner._errors.error(
                f"{self.runner._errors.label} admission probe did not complete."
            )
        return result.exit_code

    async def collect(self):
        from cayu.environments.admission import (
            EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS,
            ExecutionExecutableEvidence,
        )

        self.state.candidate = None
        self.state.identity = None
        # Timestamp the start of the complete observation, not its end.
        observed_at = self._now()
        valid_until = observed_at + timedelta(seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS)
        initial = await self._observe_identity()
        probes = {probe.executable: probe for probe in self.requirements.executable_probes()}
        claims = []
        for name in self.requirements.executable_names():
            probe = probes.get(name)
            arguments = None if probe is None else probe.probe_arguments
            command = (
                ExecCommand.process(
                    "/bin/sh", "-c", EXECUTABLE_AVAILABILITY_SCRIPT, "cayu-admission", name
                )
                if arguments is None
                else ExecCommand.process(name, *arguments)
            )
            code = await self._probe(command)
            available = code in ((0,) if probe is None else probe.accepted_exit_codes)
            claims.append(
                ExecutionExecutableEvidence(
                    executable=name,
                    state="live_verified" if available else "unavailable",
                    observed_at=observed_at if available else None,
                    valid_until=valid_until if available else None,
                    requirement_fingerprint=None if probe is None else probe.fingerprint,
                    reason_code=None if available else "executable_unavailable",
                    remediation_code=None if available else "install_executable",
                )
            )
        final = await self._observe_identity()
        if final != initial:
            raise self.runner._errors.identity_drift(
                f"{self.runner._errors.label} identity changed during admission observation."
            )
        self.runner._require_admission_open()
        candidate = self._candidate(final, tuple(claims))
        self.state.identity = final
        self.state.candidate = candidate
        return candidate

    async def refresh(self):
        await self.collect()


def _exec_result(
    response: Mapping[str, Any], errors: SidecarErrors, limits: SidecarLimits
) -> ExecResult:
    exit_code = response.get("exit_code")
    if type(exit_code) is not int:
        raise errors.protocol(f"{errors.label} result exit_code must be an integer.")
    stdout_bytes = _output_byte_count(response, "stdout_bytes", errors)
    stderr_bytes = _output_byte_count(response, "stderr_bytes", errors)
    return ExecResult(
        stdout=_decode_output(response, "stdout_base64", stdout_bytes, errors, limits),
        stderr=_decode_output(response, "stderr_base64", stderr_bytes, errors, limits),
        exit_code=exit_code,
        timed_out=_required_bool(response, "timed_out", errors),
        cancelled=_optional_bool(response, "cancelled", False, errors),
        stdout_truncated=_required_bool(response, "stdout_truncated", errors),
        stderr_truncated=_required_bool(response, "stderr_truncated", errors),
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
    )


def _output_byte_count(response: Mapping[str, Any], key: str, errors: SidecarErrors) -> int:
    value = response.get(key)
    if type(value) is not int or value < 0:
        raise errors.protocol(f"{errors.label} result {key} must be a nonnegative integer.")
    return value


def _decode_output(
    response: Mapping[str, Any],
    key: str,
    byte_count: int,
    errors: SidecarErrors,
    limits: SidecarLimits,
) -> str:
    raw = response.get(key, "")
    if type(raw) is not str:
        raise errors.protocol(f"{errors.label} result {key} must be a string.")
    if len(raw) > limits.max_encoded_output_bytes:
        raise errors.protocol(f"{errors.label} encoded output exceeds its byte ceiling.")
    padding = 2 if raw.endswith("==") else 1 if raw.endswith("=") else 0
    if (len(raw) // 4) * 3 - padding > limits.max_output_bytes:
        raise errors.protocol(f"{errors.label} decoded output exceeds its byte ceiling.")
    try:
        decoded = base64.b64decode(raw, validate=True)
    except ValueError as exc:
        raise errors.protocol(f"{errors.label} result {key} was invalid base64.") from exc
    if byte_count < len(decoded):
        raise errors.protocol(
            f"{errors.label} result byte total for {key} must cover decoded output bytes."
        )
    return decoded.decode("utf-8", errors="replace")


def _required_response_string(response: Mapping[str, Any], key: str, errors: SidecarErrors) -> str:
    value = response.get(key)
    if type(value) is not str or not value.strip():
        raise errors.protocol(f"{errors.label} response {key} must be a string.")
    return value


def _response_string(response: Any, key: str) -> str | None:
    if not isinstance(response, Mapping):
        return None
    value = response.get(key)
    return value if type(value) is str and value.strip() else None


def _required_bool(response: Mapping[str, Any], key: str, errors: SidecarErrors) -> bool:
    value = response.get(key)
    if type(value) is not bool:
        raise errors.protocol(f"{errors.label} response {key} must be a boolean.")
    return value


def _optional_bool(
    response: Mapping[str, Any], key: str, default: bool, errors: SidecarErrors
) -> bool:
    value = response.get(key, default)
    if type(value) is not bool:
        raise errors.protocol(f"{errors.label} response {key} must be a boolean.")
    return value


def _positive_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero.")
    return float(value)


def _nonnegative_float(value: float, field_name: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{field_name} must be a number.")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field_name} must be finite and non-negative.")
    return float(value)


def _validate_owner_epoch(value: int | None) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("owner_epoch must be a nonnegative integer or None.")
    return value


_COMMAND_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_MAX_CARRIED_COMMANDS = 256


def _validate_command_id(value: str | None) -> str | None:
    # "." and ".." would name another route once the transport puts the ID in a URL path.
    if value is not None and (
        type(value) is not str or value in {".", ".."} or not _COMMAND_ID_PATTERN.fullmatch(value)
    ):
        raise ValueError("command_id must be 1 to 128 of [A-Za-z0-9._:-], other than . and ..")
    return value


def _validate_carry_commands(value: Sequence[str], owner_epoch: int | None) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("carry_commands must be a sequence of command IDs.")
    carried = tuple(dict.fromkeys(value))
    if len(carried) > _MAX_CARRIED_COMMANDS:
        raise ValueError(f"carry_commands names at most {_MAX_CARRIED_COMMANDS} commands.")
    for command_id in carried:
        _validate_command_id(command_id)
    if carried and owner_epoch is None:
        raise ValueError("carry_commands requires an owner_epoch.")
    return carried
