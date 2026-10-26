"""Run commands and file operations in a brokered sandbox through the broker's API."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import logging
import posixpath
import re
import secrets
import tarfile
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, BinaryIO, Literal, TypeVar
from urllib.parse import quote

from cayu._validation import canonical_durable_json_bytes
from cayu.environments.admission import (
    ExecutionAdmissionCandidate,
    ExecutionCapabilityClaim,
    ExecutionCapabilityEvidence,
    ExecutionExecutableEvidence,
    ExecutionToolRequirementEvidence,
)
from cayu.environments.bindings import _SYNC_GIT_MODE_TAR_OWNER as SYNC_GIT_MODE_TAR_OWNER
from cayu.runners._admission_probes import EXECUTABLE_AVAILABILITY_SCRIPT
from cayu.runners._cleanup import _cleanup_artifact
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
    RunnerSystemExecutionMode,
    attach_cancellation_artifacts,
    copy_exec_command,
)
from cayu.sandbox_broker.client import (
    JSON_OUTPUT_EXPANSION,
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_PATHS,
    MAX_COMMAND_OUTPUT_BYTES,
    MAX_STDIO_CHUNK_BYTES,
    MAX_WAIT_SECONDS,
    MIN_COMMAND_OUTPUT_BYTES,
    SandboxArchiveTooLargeError,
    SandboxBrokerClient,
    SandboxBrokerError,
    SandboxBrokerRefusedError,
    SandboxBrokerUnavailableError,
    SandboxBusyError,
    SandboxCommand,
    SandboxCopy,
    SandboxCopyFailedError,
    SandboxGuarantees,
    SandboxLeaseLostError,
    SandboxNotFoundError,
    SandboxNotReadyError,
    SandboxRecord,
    SandboxStagedCopies,
    _CopyInDispatch,
    validate_command_shape,
)
from cayu.sandbox_broker.lease import SandboxLeaseKeeper
from cayu.workspaces import RunnerWorkspace
from cayu.workspaces.base import TarStreamReadResult

__all__ = [
    "BROKERED_SANDBOX_CANDIDATE",
    "BrokeredSandboxRunner",
    "BrokeredSandboxWorkspace",
    "StagedCopiesMode",
    "sandbox_capability_claims",
]

_LOGGER = logging.getLogger(__name__)
_T = TypeVar("_T")
BROKERED_SANDBOX_ROOT = "/workspace"
#: Admission candidate and evidence subject for brokered sandboxes.
BROKERED_SANDBOX_CANDIDATE = "cayu-cloud-sandbox"
_BUSY_RETRY_SECONDS = 120.0
_START_RECOVERY_SECONDS = 120.0
_POLL_RECOVERY_GRACE_SECONDS = 120.0
_CANCEL_SETTLE_SECONDS = 30.0
_DEFAULT_TIMEOUT_SECONDS = 300
# A command without a timeout runs until this long before the sandbox's end of
# life by the Agent's clock. The broker checks the timeout against its own,
# later and possibly skewed, clock and refuses one past the remaining lifetime.
_LIFETIME_CLOCK_MARGIN_SECONDS = 60
# How the broker refuses a timeout past the sandbox's remaining lifetime.
_REMAINING_LIFETIME_REFUSAL = re.compile(r"remaining lifetime \((\d+) seconds\)")
_ADMISSION_PROBE_TIMEOUT_SECONDS = 10
_SHA256_IDENTITY = re.compile(r"sha256:[0-9a-f]{64}\Z")
# Leave room in each archive for tar headers, long-name records and padding.
_ARCHIVE_MEMBER_OVERHEAD = 4 * 1024
_ARCHIVE_END_BYTES = 1024
_ARCHIVE_QUERY_BUDGET = 6 * 1024
#: The largest file the brokered workspace can move in one archive.
MAX_TRANSFER_FILE_BYTES = MAX_ARCHIVE_BYTES - 64 * 1024
#: What a staged copy of one file may take beyond the file: its tar headers and padding.
STAGED_FILE_OVERHEAD_BYTES = 64 * 1024
#: When a brokered workspace uses the sandbox's staged copies.
StagedCopiesMode = Literal["auto", "never", "always"]


def sandbox_capability_claims(
    guarantees: SandboxGuarantees | None,
    *,
    egress_mode: str,
    declared: bool,
) -> tuple[ExecutionCapabilityClaim, ...]:
    """Map Cloud's sandbox guarantees onto Runtime execution capabilities.

    ``declared`` builds pre-create claims from the configuration; otherwise a
    capability is ``available`` only when the sandbox record carries the
    guarantee item that backs it, and ``unsupported`` when it doesn't.

    ``deny_by_default_network`` follows the record's egress guarantee:

    - from guarantees version 2 on, the ``egress_deny_by_default`` item (with
      ``egress_only_via_broker``) backs it. Cloud sets that item on ``none``
      sandboxes, which have no path out, and on ``allowlist`` sandboxes, which
      reach only their listed hosts through Cloud's logging proxy with no
      credential added;
    - a version 1 record backs it only for ``egress_mode="none"``, with
      ``egress_only_via_broker``, so an ``allowlist`` record without the
      version 2 item fails closed.

    ``public_web`` reaches the open internet through a proxy that logs but
    denies nothing, so it never backs ``deny_by_default_network``, whatever
    its record says. Pre-create (``declared``) claims declare it for ``none``
    and ``allowlist``; the record must then prove it. No mode backs
    ``brokered_egress``, which needs a Runtime policy broker in the network
    path.
    """

    items = set() if guarantees is None or guarantees.version < 1 else set(guarantees.items)
    backing: dict[str, tuple[str, ...]] = {
        "untrusted_code_isolation": ("untrusted_code_isolation",),
        "host_filesystem_isolation": ("untrusted_code_isolation",),
        "real_credential_non_possession": ("no_provider_credentials",),
        "guest_privilege_containment": ("unprivileged_commands",),
        "unprivileged_guest": ("unprivileged_commands",),
    }
    if egress_mode != "public_web":
        if declared:
            if egress_mode in ("none", "allowlist"):
                backing["deny_by_default_network"] = ()
        elif guarantees is not None and guarantees.version >= 2:
            backing["deny_by_default_network"] = (
                "egress_only_via_broker",
                "egress_deny_by_default",
            )
        elif egress_mode == "none":
            backing["deny_by_default_network"] = ("egress_only_via_broker",)
    claims: list[ExecutionCapabilityClaim] = []
    for capability, required in backing.items():
        if declared:
            claims.append(ExecutionCapabilityClaim.declared(capability))
        elif items.issuperset(required):
            claims.append(ExecutionCapabilityClaim.available(capability))
        else:
            claims.append(
                ExecutionCapabilityClaim.unsupported(
                    capability,
                    reason_code="sandbox_guarantee_missing",
                    remediation_code="select_guaranteed_sandbox",
                )
            )
    if egress_mode == "public_web":
        claims.append(
            ExecutionCapabilityClaim.unsupported(
                "deny_by_default_network",
                reason_code="sandbox_egress_public_web",
                remediation_code="select_no_egress",
            )
        )
    elif "deny_by_default_network" not in backing:
        # An allowlist (or a mode this Runtime doesn't know) on a record without
        # the version 2 egress item: nothing proves the network is denied.
        claims.append(
            ExecutionCapabilityClaim.unsupported(
                "deny_by_default_network",
                reason_code="sandbox_guarantee_missing",
                remediation_code="select_guaranteed_sandbox",
            )
        )
    claims.append(
        ExecutionCapabilityClaim.unsupported(
            "brokered_egress",
            reason_code="sandbox_egress_unbrokered",
            remediation_code="select_brokered_egress",
        )
    )
    # Cancellation is confirmed by the broker or the sandbox is destroyed, and
    # destroy is the cleanup; neither depends on a guarantee item.
    for capability in ("confirmed_cancellation", "confirmed_cleanup"):
        claims.append(
            ExecutionCapabilityClaim.declared(capability)
            if declared
            else ExecutionCapabilityClaim.available(capability)
        )
    claims.append(
        ExecutionCapabilityClaim.unsupported(
            "reconnect",
            reason_code="reconnect_unsupported",
            remediation_code="select_reconnectable_execution",
        )
    )
    return tuple(claims)


class BrokeredSandboxRunner(Runner):
    """A ``Runner`` over one brokered sandbox, operated only through the broker API.

    Each command starts under a fresh ``command_id`` and is polled until it
    finishes, so its length is bounded only by its timeout and the sandbox's
    lifetime (a command without a timeout may run until a minute before the
    sandbox's maximum lifetime, by the Agent's clock; a broker that names a
    shorter remaining lifetime when it refuses the timeout is asked once more
    with that number). A dropped connection is recovered by replaying the same id,
    which the broker never runs twice. A command cancelled by its caller is
    cancelled in the sandbox; only if the broker can't confirm it stopped is
    the runner's exec path closed for good. Bounds that can't be met (stdin
    over 1 MiB, argv over 256 KiB, env over 64 KiB) are refused, and output
    is captured up to 2 MiB per stream as the broker encodes it in JSON
    (fewer raw bytes when the output is full of control characters or
    non-ASCII text). Each argv argument is limited to 131,071 bytes.

    ``close`` stops lease renewal and destroys the sandbox. If the destroy call
    fails, the sandbox is no longer renewed and the broker destroys it when its
    lease lapses; ``close`` can be retried.

    ``toolchain_profile_fingerprint`` and ``toolchain_executables`` bind the
    runner to one coding toolchain profile: its admission evidence then names
    that profile and carries live probes of those executables, which
    ``refresh_execution_admission`` renews in this exact sandbox. Command
    tools holding the same profile admit the runner through that evidence.
    """

    isolation = BROKERED_SANDBOX_CANDIDATE
    default_cwd = BROKERED_SANDBOX_ROOT
    system_execution_mode: RunnerSystemExecutionMode = "shared"

    def __init__(
        self,
        client: SandboxBrokerClient,
        sandbox: SandboxRecord,
        *,
        lease_keeper: SandboxLeaseKeeper | None = None,
        destroy_on_close: bool = True,
        poll_wait_seconds: int = MAX_WAIT_SECONDS,
        toolchain_profile_fingerprint: str | None = None,
        toolchain_executables: Sequence[str] = (),
    ) -> None:
        if not isinstance(client, SandboxBrokerClient):
            raise TypeError("BrokeredSandboxRunner requires a SandboxBrokerClient.")
        if not isinstance(sandbox, SandboxRecord):
            raise TypeError("BrokeredSandboxRunner requires the sandbox's SandboxRecord.")
        if sandbox.status != "ready":
            raise ValueError("BrokeredSandboxRunner requires a ready sandbox.")
        if lease_keeper is not None and lease_keeper.sandbox_id != sandbox.id:
            raise ValueError("The lease keeper belongs to a different sandbox.")
        if type(destroy_on_close) is not bool:
            raise TypeError("destroy_on_close must be a bool.")
        if type(poll_wait_seconds) is not int or not 0 <= poll_wait_seconds <= MAX_WAIT_SECONDS:
            raise ValueError(f"poll_wait_seconds must be from 0 to {MAX_WAIT_SECONDS}.")
        if toolchain_profile_fingerprint is not None and (
            type(toolchain_profile_fingerprint) is not str
            or _SHA256_IDENTITY.fullmatch(toolchain_profile_fingerprint) is None
        ):
            raise ValueError("toolchain_profile_fingerprint must be a lowercase SHA-256 identity.")
        executables = tuple(toolchain_executables)
        if executables and toolchain_profile_fingerprint is None:
            raise ValueError("toolchain_executables require toolchain_profile_fingerprint.")
        if any(type(item) is not str or not item for item in executables):
            raise ValueError("toolchain_executables must be nonempty strings.")
        self._client = client
        self.sandbox = sandbox
        self.lease_keeper = lease_keeper
        self.destroy_on_close = destroy_on_close
        self.poll_wait_seconds = poll_wait_seconds
        self.toolchain_profile_fingerprint = toolchain_profile_fingerprint
        self.toolchain_executables = tuple(sorted(set(executables)))
        self._toolchain_evidence: tuple[ExecutionExecutableEvidence, ...] = ()
        self._toolchain_refresh_lock = asyncio.Lock()
        self._destroyed = False

    @property
    def sandbox_id(self) -> str:
        return self.sandbox.id

    @property
    def broker(self) -> SandboxBrokerClient:
        return self._client

    @property
    def resource_key(self) -> tuple[object, ...]:
        settings = self._client.settings
        return (
            BROKERED_SANDBOX_CANDIDATE,
            settings.api_url,
            settings.application_id,
            self.sandbox.id,
        )

    def output_secret_values_present(self) -> bool:
        # Cayu places nothing secret in a brokered sandbox's environment.
        return False

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
        super().preflight_exec(
            command,
            cwd=cwd,
            env=env,
            env_remove=env_remove,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        owned = copy_exec_command(command)
        environment = remove_runner_env(copy_runner_env(env, inherit_env=False), env_remove)
        validate_command_shape(_argv(owned), environment, validate_stdin(stdin))
        self._timeout_seconds(validate_timeout(timeout_s))

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
        self._ensure_exec_open()
        owned = copy_exec_command(command)
        working_dir = self.resolve_cwd(cwd)
        environment = remove_runner_env(copy_runner_env(env, inherit_env=False), env_remove)
        standard_input = validate_stdin(stdin)
        argv = _argv(owned)
        validate_command_shape(argv, environment, standard_input)
        timeout_seconds = self._timeout_seconds(validate_timeout(timeout_s))
        limit = validate_output_limit(output_limit_bytes)
        requested_limit = MAX_COMMAND_OUTPUT_BYTES if limit is None else limit
        # The broker measures each stream as encoded in its JSON response, where
        # a byte can take up to six; ask for enough to return the caller's raw
        # limit, then cut to it here.
        broker_limit = min(
            MAX_COMMAND_OUTPUT_BYTES,
            max(MIN_COMMAND_OUTPUT_BYTES, requested_limit * JSON_OUTPUT_EXPANSION),
        )
        relative_cwd = posixpath.relpath(working_dir, BROKERED_SANDBOX_ROOT)
        self._require_lease()
        command_id = f"cayu-{secrets.token_hex(16)}"
        dispatch = _Dispatch()
        try:
            finished = await self._run(
                command_id,
                argv,
                dispatch,
                cwd=None if relative_cwd == "." else relative_cwd,
                env=environment,
                timeout_seconds=timeout_seconds,
                stdin=standard_input,
                output_limit_bytes=broker_limit,
            )
        except asyncio.CancelledError as cancellation:
            settled = await self._settle_cancelled(command_id, dispatch)
            attach_cancellation_artifacts(
                cancellation, [_kill_receipt("completed" if settled else "failed")]
            )
            raise
        except (SandboxNotReadyError, SandboxNotFoundError) as error:
            raise self._lease_lost(error) from None
        if finished.status == "failed":
            raise self._lease_lost(None)
        stdout, stdout_cut = _bounded_text(finished.stdout, requested_limit)
        stderr, stderr_cut = _bounded_text(finished.stderr, requested_limit)
        killed = finished.status in {"timed_out", "cancelled"}
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=finished.exit_code if finished.exit_code is not None else -1,
            timed_out=finished.status == "timed_out",
            cancelled=finished.status == "cancelled",
            # The broker reports these only once every process in the command's
            # group has stopped, which is the settlement workspace fencing needs.
            artifacts=[_kill_receipt("completed")] if killed else [],
            stdout_truncated=finished.stdout_truncated or stdout_cut,
            stderr_truncated=finished.stderr_truncated or stderr_cut,
        )

    async def put_archive(self, data: bytes, *, path: str | None = None) -> dict:
        """Extract an uncompressed tar under ``/workspace/<path>`` through the broker."""

        self._ensure_exec_open()
        self._require_lease()
        try:
            return await self._client.put_archive(self.sandbox.id, data, path=path)
        except (SandboxNotReadyError, SandboxNotFoundError) as error:
            raise self._lease_lost(error) from None

    async def get_archive(self, paths: Sequence[str]):
        self._ensure_exec_open()
        self._require_lease()
        try:
            return await self._client.get_archive(self.sandbox.id, paths=paths)
        except (SandboxNotReadyError, SandboxNotFoundError) as error:
            raise self._lease_lost(error) from None

    @property
    def staged_copies(self) -> SandboxStagedCopies | None:
        """What the sandbox offers for staged copies, or None when only archives work."""

        return self.sandbox.staged_copies

    async def copy_in(
        self, source: BinaryIO, *, size: int, sha256: str, path: str | None = None
    ) -> SandboxCopy:
        """Extract a tar of any size the sandbox's staged copies take, through storage."""

        self._ensure_exec_open()
        self._require_lease()
        dispatch = _CopyInDispatch()
        try:
            return await self._client.copy_in(
                self.sandbox.id, source, size=size, sha256=sha256, path=path, _dispatch=dispatch
            )
        except (SandboxNotReadyError, SandboxNotFoundError) as error:
            raise self._lease_lost(error) from None
        except SandboxCopyFailedError as error:
            if error.code == "sandbox_not_ready":
                raise self._lease_lost(error) from None
            raise
        finally:
            if dispatch.mutation_possible and not dispatch.settled:
                self._poison_exec("the broker did not confirm staged workspace extraction stopped")

    async def copy_out(
        self,
        sink: BinaryIO,
        *,
        max_bytes: int,
        paths: Sequence[str] = (),
        excludes: Sequence[str] = (),
    ) -> SandboxCopy:
        """Write a tar of ``paths`` to ``sink`` through storage; see ``SandboxBrokerClient``."""

        self._ensure_exec_open()
        self._require_lease()
        try:
            return await self._client.copy_out(
                self.sandbox.id, sink, max_bytes=max_bytes, paths=paths, excludes=excludes
            )
        except (SandboxNotReadyError, SandboxNotFoundError) as error:
            raise self._lease_lost(error) from None
        except SandboxCopyFailedError as error:
            if error.code == "sandbox_not_ready":
                raise self._lease_lost(error) from None
            raise

    async def refresh(self) -> SandboxRecord:
        self.sandbox = await self._client.get(self.sandbox.id)
        return self.sandbox

    def interactive_process_capability(self) -> RunnerInteractiveProcessCapability | None:
        """Interactive commands through the broker, each stopped with its whole process tree.

        The broker runs each one in its own PID namespace and reports it ended only
        once every process in it has exited, so ``runner_agent_launch`` can run a
        vendor CLI here. Stdio crosses the broker as offset-addressed requests.
        ``None`` unless the sandbox record says the broker runs interactive
        commands (``interactive_commands``); a broker that doesn't refuses them.
        """

        if not self.sandbox.interactive_commands:
            return None
        return _BrokeredInteractiveProcessCapability(self)

    # Admission --------------------------------------------------------------

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return _admission_candidate(self, self._toolchain_evidence)

    async def refresh_execution_admission(self) -> None:
        """Probe the bound toolchain's executables again in this exact sandbox."""

        if not self.toolchain_executables:
            return
        from cayu.environments.admission import ExecutionRequirements

        async with self._toolchain_refresh_lock:
            observer = _BrokeredAdmissionObserver(
                self,
                ExecutionRequirements.trusted(required_executables=self.toolchain_executables),
            )
            candidate = await observer.collect()
            tool_requirements = candidate.evidence.tool_requirements
            self._toolchain_evidence = (
                () if tool_requirements is None else tool_requirements.executables
            )

    async def collect_execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        await self.refresh_execution_admission()
        return self.execution_admission_candidate()

    def execution_admission_observer(self, requirements):
        if not requirements.executable_names():
            return super().execution_admission_observer(requirements)
        return _BrokeredAdmissionObserver(self, requirements)

    def execution_admission_candidate_for(self, requirements):
        if not requirements.executable_names():
            return self.execution_admission_candidate()
        return _BrokeredAdmissionObserver(self, requirements).snapshot()

    # Lifecycle --------------------------------------------------------------

    async def close(self) -> None:
        if self._closed:
            return
        self._close_exec("BrokeredSandboxRunner is closed")
        if self.lease_keeper is not None:
            await self.lease_keeper.stop()
        if self.destroy_on_close and not self._destroyed:
            with contextlib.suppress(SandboxNotFoundError):
                self.sandbox = await self._client.destroy(self.sandbox.id)
            self._destroyed = True
        self._closed = True

    async def _run(
        self, command_id: str, argv: list[str], dispatch: _Dispatch, **options: Any
    ) -> SandboxCommand:
        loop = asyncio.get_running_loop()
        command = await self._start(command_id, argv, dispatch, **options)
        # Polls survive a broker outage for as long as the command itself may run.
        deadline = loop.time() + options["timeout_seconds"] + _POLL_RECOVERY_GRACE_SECONDS
        while not command.finished:
            try:
                command = await self._client.get_command(
                    self.sandbox.id, command_id, wait_seconds=self.poll_wait_seconds
                )
            except SandboxBrokerUnavailableError:
                if loop.time() >= deadline:
                    raise
                await asyncio.sleep(1.0)
        return command

    async def _start(
        self, command_id: str, argv: list[str], dispatch: _Dispatch, **options: Any
    ) -> SandboxCommand:
        loop = asyncio.get_running_loop()
        busy_deadline = loop.time() + _BUSY_RETRY_SECONDS
        recovery_deadline = loop.time() + _START_RECOVERY_SECONDS
        delay = 0.25
        lifetime_retried = False
        while True:
            # Every dispatch, retries included, must pass the exec fence: another
            # command may have poisoned the runner while this one waited.
            try:
                self._ensure_exec_open()
                self._require_lease()
            except BaseException:
                if dispatch.outcome_unknown:
                    await self._settle_cancelled(command_id, dispatch)
                raise
            # Each attempt is one HTTP request, run shielded, so every dispatch
            # that may have reached the broker is visible here and a caller's
            # cancellation never hides one.
            attempt = asyncio.ensure_future(
                self._client.start_command(
                    self.sandbox.id, command_id, argv, retry=False, **options
                )
            )
            dispatch.attempt = attempt
            try:
                # Replaying the same id and body returns the command already started.
                return await asyncio.shield(attempt)
            except SandboxBusyError:
                # This attempt did not start the command; wait for a slot.
                if loop.time() >= busy_deadline:
                    if dispatch.outcome_unknown:
                        await self._settle_cancelled(command_id, dispatch)
                    raise
            except SandboxBrokerRefusedError as error:
                # The broker checks a timeout against the remaining lifetime by
                # its own clock, and only for a command id it has not stored,
                # so this refusal started nothing. Ask once more, for the
                # lifetime it named.
                remaining = None if lifetime_retried else _refused_remaining_lifetime(error)
                if remaining is not None and remaining < options["timeout_seconds"]:
                    lifetime_retried = True
                    options = {**options, "timeout_seconds": remaining}
                    continue
                # A refused replay says nothing about an earlier attempt that
                # may have started the command, so stop that one first.
                if dispatch.outcome_unknown:
                    await self._settle_cancelled(command_id, dispatch)
                raise
            except SandboxBrokerError as error:
                # Anything but a refusal may follow an applied start: a lost or
                # unparseable answer, or one over the response bound. Only a
                # connection the broker never accepted proves nothing was sent.
                dispatch.outcome_unknown = dispatch.outcome_unknown or (
                    error.outcome_unknown
                    if isinstance(error, SandboxBrokerUnavailableError)
                    else True
                )
                if loop.time() >= recovery_deadline:
                    # Gives up on this command; poisons the runner if it may still run.
                    await self._settle_cancelled(command_id, dispatch)
                    raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 2.0)

    async def _cancel_until_finished(self, command_id: str, deadline: float) -> SandboxCommand:
        command = await self._client.cancel_command(self.sandbox.id, command_id)
        while not command.finished and asyncio.get_running_loop().time() < deadline:
            command = await self._client.get_command(
                self.sandbox.id, command_id, wait_seconds=min(5, self.poll_wait_seconds)
            )
        return command

    async def _settle_cancelled(self, command_id: str, dispatch: _Dispatch) -> bool:
        """Stop a command the caller abandoned; poison the runner unless the broker confirms.

        Returns whether nothing from this command can still run.
        """

        async def settle() -> bool:
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _CANCEL_SETTLE_SECONDS
            attempt = dispatch.attempt
            if attempt is None:
                return True  # Nothing was ever sent.
            if not attempt.done():
                # Let the in-flight start finish so its outcome is known.
                with contextlib.suppress(BaseException):
                    await asyncio.wait_for(asyncio.shield(attempt), _CANCEL_SETTLE_SECONDS)
            if dispatch.definitively_refused:
                return True  # Every start was definitively refused.
            while loop.time() < deadline:
                try:
                    command = await self._cancel_until_finished(command_id, deadline)
                    return command.finished
                except SandboxNotFoundError:
                    # A start may still be in flight, so absence proves nothing.
                    return False
                except SandboxNotReadyError:
                    # The sandbox is gone; nothing in it keeps running.
                    return True
                except SandboxBrokerError:
                    await asyncio.sleep(1.0)
            return False

        task = asyncio.create_task(settle())
        try:
            settled = await asyncio.shield(task)
        except BaseException:
            settled = False
        dispatch.cleanup_required = not settled
        if not settled:
            self._poison_exec("the broker could not confirm that an abandoned command stopped")
        return settled

    def _timeout_seconds(self, timeout: int | None) -> int:
        remaining = self._remaining_lifetime_seconds()
        if timeout is None:
            if remaining is None:
                return _DEFAULT_TIMEOUT_SECONDS
            return max(1, remaining - _LIFETIME_CLOCK_MARGIN_SECONDS)
        if remaining is not None and timeout > remaining:
            raise ValueError(
                f"Command timeout of {timeout} seconds exceeds the sandbox's remaining "
                f"lifetime of {remaining} seconds."
            )
        return timeout

    def _remaining_lifetime_seconds(self) -> int | None:
        expires_at = self.sandbox.max_expires_at
        if expires_at is None:
            return None
        return int((expires_at - datetime.now(UTC)).total_seconds())

    def _lease_lost(self, error: SandboxBrokerError | None) -> SandboxLeaseLostError:
        # A sandbox that stopped being ready never becomes ready again, so the
        # runner can't be reopened: poison it, which admission treats as terminal.
        self._record_loss("lease_lost")
        self._poison_exec("the brokered sandbox is no longer ready")
        return SandboxLeaseLostError(
            "The brokered sandbox is no longer ready.",
            status=None if error is None else error.status,
            code=None if error is None else error.code,
        )

    def _require_lease(self) -> None:
        keeper = self.lease_keeper
        if keeper is None:
            return
        try:
            keeper.check()
        except SandboxLeaseLostError:
            self._record_loss("lease_lost")
            self._poison_exec("the brokered sandbox's lease was lost")
            raise


class _BrokeredInteractiveProcessCapability(RunnerInteractiveProcessCapability):
    """Interactive processes in one brokered sandbox; the runner keeps its lifecycle."""

    def __init__(self, runner: BrokeredSandboxRunner) -> None:
        self._runner = runner

    async def open_interactive_process(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> RunnerInteractiveProcess:
        runner = self._runner
        runner._ensure_exec_open()
        argv = _argv(copy_exec_command(command))
        working_dir = runner.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=False)
        validate_command_shape(argv, environment, None)
        timeout_seconds = runner._timeout_seconds(validate_timeout(timeout_s))
        relative_cwd = posixpath.relpath(working_dir, BROKERED_SANDBOX_ROOT)
        runner._require_lease()
        command_id = f"cayu-{secrets.token_hex(16)}"
        dispatch = _Dispatch()
        process = _BrokeredInteractiveProcess(runner, command_id, timeout_seconds, dispatch)
        try:
            await runner._start(
                command_id,
                argv,
                dispatch,
                cwd=None if relative_cwd == "." else relative_cwd,
                env=environment,
                timeout_seconds=timeout_seconds,
                interactive=True,
            )
        except asyncio.CancelledError as cancellation:
            if not await runner._settle_cancelled(command_id, dispatch):
                self.retain_failed_start(cancellation, process)
            raise
        except (SandboxNotReadyError, SandboxNotFoundError) as error:
            lost = runner._lease_lost(error)
            if dispatch.cleanup_required:
                self.retain_failed_start(lost, process)
            raise lost from None
        except BaseException as error:
            # The start gave up without confirming that what it may have started stopped.
            if dispatch.cleanup_required:
                self.retain_failed_start(error, process)
            raise
        process.started = True
        process._deadline = (
            asyncio.get_running_loop().time() + timeout_seconds + _POLL_RECOVERY_GRACE_SECONDS
        )
        return process


class _BrokeredInteractiveProcess(RunnerInteractiveProcess):
    """One interactive command's stdio, addressed by byte offsets through the broker.

    Every stdin write and stdout read names its offset, so a request whose answer was
    lost is sent again without duplicating or losing bytes.
    """

    def __init__(
        self, runner: BrokeredSandboxRunner, command_id: str, timeout_s: int, dispatch: _Dispatch
    ) -> None:
        self._runner = runner
        self._dispatch = dispatch
        self.command_id = command_id
        self.started = False
        # Broker calls keep retrying an outage for as long as the command may run.
        self._deadline = (
            asyncio.get_running_loop().time() + timeout_s + _POLL_RECOVERY_GRACE_SECONDS
        )
        self._stdin_uncertain = False
        self._stdin_offset = 0
        self._stdout_offset = 0
        self._stdin_closed = False
        self._stdout_eof = False
        self._closed = False
        self._write_lock = asyncio.Lock()
        self._read_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()

    async def write(self, data: bytes) -> None:
        async with self._write_lock:
            if self._stdin_uncertain:
                raise SandboxBrokerError("Previous stdin delivery is uncertain; close the process.")
            if self._stdin_closed:
                raise BrokenPipeError("Interactive process stdin is closed.")
            data = bytes(data)
            for start in range(0, len(data), MAX_STDIO_CHUNK_BYTES):
                await self._write_chunk(data[start : start + MAX_STDIO_CHUNK_BYTES], eof=False)
                if self._stdin_closed:
                    raise BrokenPipeError("Interactive process stdin is closed.")

    async def close_stdin(self) -> None:
        async with self._write_lock:
            if self._stdin_uncertain:
                raise SandboxBrokerError("Previous stdin delivery is uncertain; close the process.")
            if not self._stdin_closed:
                await self._write_chunk(b"", eof=True)
                self._stdin_closed = True

    async def _write_chunk(self, chunk: bytes, *, eof: bool) -> None:
        runner = self._runner
        try:
            written = await self._call(
                lambda: runner.broker.write_command_stdin(
                    runner.sandbox_id,
                    self.command_id,
                    offset=self._stdin_offset,
                    data=chunk,
                    eof=eof,
                )
            )
        except SandboxNotReadyError:
            # The sandbox is gone, and the process with it.
            self._stdin_closed = True
            return
        except BaseException:
            self._stdin_uncertain = True
            raise
        if written.closed:
            self._stdin_closed = True
            return
        if written.offset != self._stdin_offset + len(chunk):
            self._stdin_uncertain = True
            raise SandboxBrokerError("The broker's stdin offset is inconsistent.")
        self._stdin_offset = written.offset

    async def read(self) -> bytes:
        async with self._read_lock:
            runner = self._runner
            while not self._stdout_eof:
                chunk = await self._call(
                    lambda: runner.broker.read_command_stdout(
                        runner.sandbox_id,
                        self.command_id,
                        offset=self._stdout_offset,
                        wait_seconds=runner.poll_wait_seconds,
                    )
                )
                self._stdout_offset += len(chunk.data)
                self._stdout_eof = chunk.eof
                if chunk.data:
                    return chunk.data
            return b""

    async def wait(self) -> int:
        runner = self._runner
        while True:
            command = await self._call(
                lambda: runner.broker.get_command(
                    runner.sandbox_id, self.command_id, wait_seconds=runner.poll_wait_seconds
                )
            )
            if command.finished:
                break
        if command.status == "failed":
            raise runner._lease_lost(None)
        return command.exit_code if command.exit_code is not None else -1

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._stdin_closed = True
            runner = self._runner
            loop = asyncio.get_running_loop()
            deadline = loop.time() + _CANCEL_SETTLE_SECONDS
            while True:
                if not self.started and self._dispatch.definitively_refused:
                    break
                try:
                    command = await runner._cancel_until_finished(self.command_id, deadline)
                    # The broker reports an interactive command ended only once its whole
                    # process tree has exited, or destroys the sandbox.
                    if command.finished:
                        break
                except SandboxNotReadyError:
                    break
                except SandboxNotFoundError:
                    # A start whose answer was lost may still register it.
                    if self.started:
                        break
                except SandboxBrokerError:
                    pass
                if loop.time() >= deadline:
                    raise SandboxBrokerError(
                        "The broker could not confirm that the interactive process tree exited."
                    )
                await asyncio.sleep(1.0)
            self._closed = True

    async def _call(self, request: Callable[[], Awaitable[_T]]) -> _T:
        """One broker request, retried through an outage while the command may still run.

        The offsets make every retry idempotent.
        """

        runner = self._runner
        loop = asyncio.get_running_loop()
        while True:
            runner._require_lease()
            try:
                return await request()
            except SandboxNotFoundError as error:
                raise runner._lease_lost(error) from None
            except SandboxBrokerUnavailableError:
                if loop.time() >= self._deadline:
                    raise
            await asyncio.sleep(1.0)


class BrokeredSandboxWorkspace(RunnerWorkspace):
    """``RunnerWorkspace`` whose bulk copies go through the broker's archive routes.

    File tools run through commands like any runner workspace. Copy-in
    uploads the tree in archives of at most ``MAX_ARCHIVE_BYTES`` each, after
    checking every member against the workspace's exclusions; copy-back reads
    the named files in batches with the selective archive read, splitting a
    batch the broker reports as too large.

    ``staged_copies`` chooses when staged copies are used. With ``"auto"`` (the
    default), where the sandbox offers them, a tree larger than one archive
    goes in as one staged copy, and a file too large for one archive comes back
    as one; a staged copy-in that fails while the runner stays usable is sent
    again as split archives. With ``"never"`` only archives are used, so one
    file must fit in one archive. ``"always"`` sends every copy-in, whatever
    its size, as one staged copy and refuses a sandbox that offers none.
    """

    def __init__(
        self,
        runner: BrokeredSandboxRunner,
        *,
        staged_copies: StagedCopiesMode = "auto",
        **kwargs: Any,
    ) -> None:
        if not isinstance(runner, BrokeredSandboxRunner):
            raise TypeError("BrokeredSandboxWorkspace requires a BrokeredSandboxRunner.")
        self.staged_copies = validate_staged_copies_mode(staged_copies)
        super().__init__(runner, **kwargs)
        self._brokered_runner = runner

    def _offered_staged_copies(self) -> SandboxStagedCopies | None:
        """The sandbox's staged copies this workspace may use, or None."""

        if self.staged_copies == "never":
            return None
        staged = self._brokered_runner.staged_copies
        if staged is None and self.staged_copies == "always":
            raise RuntimeError(
                "staged_copies='always', but this sandbox broker offers no staged copies."
            )
        return staged

    def tar_stream_writer(self):
        return self

    def bounded_tar_stream_reader(self):
        return self

    async def write_tar_stream(self, source: BinaryIO, *, archive_bytes: int) -> None:
        if isinstance(source, (str, bytes, bytearray, memoryview)) or not callable(
            getattr(source, "read", None)
        ):
            raise TypeError("BrokeredSandboxWorkspace tar source must be a binary stream.")
        if type(archive_bytes) is not int or archive_bytes < 0:
            raise ValueError("BrokeredSandboxWorkspace archive_bytes must be a non-negative int.")
        staged = self._offered_staged_copies()
        if staged is not None and self.staged_copies == "always":
            await self._write_staged(source, archive_bytes, staged)
            return
        # Include the splitter's conservative per-member allowance when selecting
        # staging. Preserve split uploads when the tree exceeds the staged cap.
        if (
            staged is not None
            and archive_bytes <= staged.max_bytes
            and _sealed_size(archive_bytes + _ARCHIVE_MEMBER_OVERHEAD) > MAX_ARCHIVE_BYTES
        ):
            await self._write_staged(source, archive_bytes, staged)
            return
        await self._write_split(source, archive_bytes)

    async def _write_split(self, source: BinaryIO, archive_bytes: int) -> None:
        chunks = await asyncio.to_thread(self._split_archive, source, archive_bytes)
        for chunk in chunks:
            await self._brokered_runner.put_archive(chunk, path=self.cwd)

    async def read_tar_stream(
        self,
        paths: Sequence[str],
        destination: BinaryIO,
        *,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        max_archive_bytes: int | None = None,
    ) -> TarStreamReadResult:
        if isinstance(destination, (str, bytes, bytearray, memoryview)) or not callable(
            getattr(destination, "write", None)
        ):
            raise TypeError("BrokeredSandboxWorkspace tar destination must be a binary stream.")
        requested = tuple(posixpath.normpath(path) for path in paths)
        if len(set(requested)) != len(requested):
            raise ValueError("Workspace tar paths must be unique.")
        for path in requested:
            if posixpath.isabs(path) or path == ".." or path.startswith("../"):
                raise ValueError("Workspace tar paths must be relative.")
            self._require_path_allowed(path)
        budget = _CopyBackBudget(
            per_file=self.default_read_limit_bytes if max_file_bytes is None else max_file_bytes,
            total_limit=max_total_bytes,
        )
        sink = _CountingWriter(destination, limit=max_archive_bytes)
        with tarfile.open(fileobj=sink, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for batch in _query_batches(requested, prefix=self.cwd):
                await self._stream_batch(batch, archive, budget)
        return TarStreamReadResult(archive_bytes=sink.written)

    async def _stream_batch(
        self,
        batch: tuple[str, ...],
        destination: tarfile.TarFile,
        budget: _CopyBackBudget,
    ) -> None:
        """Copy one batch into ``destination`` under the transfer limits.

        At most one broker archive (``MAX_ARCHIVE_BYTES``) is held at a time,
        and every member is checked against the per-file and total limits
        before its content is read, so files grown in the sandbox after the
        manifest can't make the Agent buffer more than one archive plus the
        limits allow.
        """

        prefix = "" if self.cwd is None else self.cwd.rstrip("/") + "/"
        try:
            result = await self._brokered_runner.get_archive([prefix + path for path in batch])
        except SandboxArchiveTooLargeError:
            # Split and keep going: members are checked against the limits as
            # they arrive, and tar framing makes the reported size no measure
            # of file content.
            if len(batch) == 1:
                staged = self._offered_staged_copies()
                if staged is None:
                    raise RuntimeError(
                        f"Workspace file is larger than one broker archive: {batch[0]}"
                    ) from None
                await self._stream_staged(batch[0], destination, budget, staged)
                return
            middle = len(batch) // 2
            await self._stream_batch(batch[:middle], destination, budget)
            await self._stream_batch(batch[middle:], destination, budget)
            return
        self._absorb(io.BytesIO(result.data), result.missing_paths, batch, destination, budget)

    async def _stream_staged(
        self,
        path: str,
        destination: tarfile.TarFile,
        budget: _CopyBackBudget,
        staged: SandboxStagedCopies,
    ) -> None:
        """Copy back one file too large for an archive as a staged copy, within the limits."""

        prefix = "" if self.cwd is None else self.cwd.rstrip("/") + "/"
        file_limit = budget.per_file + STAGED_FILE_OVERHEAD_BYTES
        limit = min(staged.max_bytes, file_limit)
        with tempfile.TemporaryFile() as spool:
            try:
                copy = await self._brokered_runner.copy_out(
                    spool, max_bytes=limit, paths=[prefix + path]
                )
            except SandboxArchiveTooLargeError as error:
                if error.code == "copy_too_large" or limit < file_limit:
                    # The sandbox's staged copies, not the transfer limits, stopped it.
                    raise RuntimeError(
                        "Workspace file is larger than the sandbox's staged copies allow "
                        f"({staged.max_bytes} bytes per copy, {staged.sandbox_max_bytes} per "
                        f"sandbox): {path}"
                    ) from None
                raise RuntimeError(
                    f"Workspace file exceeds max_file_bytes={budget.per_file}: {path}"
                ) from None
            spool.seek(0)
            self._absorb(spool, copy.missing, (path,), destination, budget)

    def _absorb(
        self,
        source: BinaryIO,
        missing: Sequence[str],
        batch: tuple[str, ...],
        destination: tarfile.TarFile,
        budget: _CopyBackBudget,
    ) -> None:
        """Move the batch's files from one broker archive into ``destination``."""

        prefix = "" if self.cwd is None else self.cwd.rstrip("/") + "/"
        if missing:
            raise FileNotFoundError(
                "Workspace files disappeared before copy-back: " + ", ".join(sorted(missing))[:500]
            )
        wanted = set(batch)
        found: set[str] = set()
        with tarfile.open(fileobj=source, mode="r:") as archive:
            for member in archive:
                name = posixpath.normpath(member.name)
                if prefix and name.startswith(prefix):
                    name = name[len(prefix) :]
                if member.isdir():
                    continue
                if not member.isreg() or name not in wanted or name in found:
                    raise RuntimeError(
                        f"Broker archive returned an unexpected member: {member.name[:200]}"
                    )
                budget.admit(name, member.size)
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Broker archive member could not be read: {name}")
                info = tarfile.TarInfo(name=name)
                info.size = member.size
                info.mode = 0o755 if member.mode & 0o111 else 0o644
                info.uname = SYNC_GIT_MODE_TAR_OWNER
                destination.addfile(info, extracted)
                found.add(name)
        if found != wanted:
            raise RuntimeError("Broker archive did not return every requested file.")

    async def _write_staged(
        self, source: BinaryIO, archive_bytes: int, staged: SandboxStagedCopies
    ) -> None:
        """Copy the tree in as one staged copy, after checking every member.

        A seekable source (``SyncBinding`` hands over its sealed, capacity-owned
        archive) is checked in one pass and uploaded from its start in a
        second, so no host copy is made beside it. Only a source that can't
        seek is spooled to a host temporary file first; that spool is outside
        any ``staging_capacity``.

        With ``staged_copies="auto"``, a staged copy that fails while the runner
        stays usable is sent again as split archives. A failed copy can leave
        part of the same tree extracted; the archives write every member again
        over it. A copy whose stop the broker can't confirm closes the runner,
        so it is never raced.
        """

        if archive_bytes > staged.max_bytes:
            raise RuntimeError(
                f"Workspace archive of {archive_bytes} bytes exceeds the broker's staged copy "
                f"limit of {staged.max_bytes} bytes."
            )
        if source.seekable():
            origin = source.tell()
            digest = await asyncio.to_thread(self._spool_checked, source, archive_bytes, None)
            # The broker checks the upload against this digest, so a source that
            # changed between the passes is refused rather than extracted.
            source.seek(origin)
            await self._copy_in_or_split(source, origin, archive_bytes, digest)
            return
        with tempfile.TemporaryFile() as spool:
            digest = await asyncio.to_thread(self._spool_checked, source, archive_bytes, spool)
            spool.seek(0)
            await self._copy_in_or_split(spool, 0, archive_bytes, digest)

    async def _copy_in_or_split(
        self, source: BinaryIO, origin: int, archive_bytes: int, digest: str
    ) -> None:
        try:
            await self._brokered_runner.copy_in(
                source, size=archive_bytes, sha256=digest, path=self.cwd
            )
            return
        except SandboxLeaseLostError:
            raise
        except SandboxBrokerError as error:
            if self.staged_copies != "auto" or not _exec_usable(self._brokered_runner):
                raise
            _LOGGER.warning(
                "Staged copy-in to brokered sandbox %s failed (%s); sending the workspace "
                "as split archives instead.",
                self._brokered_runner.sandbox_id,
                error.code or type(error).__name__,
            )
            failure = error
        source.seek(origin)
        try:
            await self._write_split(source, archive_bytes)
        except Exception as split_error:
            # A file too large for one archive can only move staged; report why
            # staging failed too.
            split_error.add_note(
                f"The workspace was sent as split archives after its staged copy failed "
                f"({failure.code or type(failure).__name__})."
            )
            raise

    def _spool_checked(self, source: BinaryIO, archive_bytes: int, spool: BinaryIO | None) -> str:
        """Read the archive, checking its members as the split path does, and return its
        SHA-256. With a ``spool`` every byte is written there; without one ``source``
        must be seekable, and is read again to check what follows its last member."""

        origin = source.tell() if spool is None else 0
        limited = _LimitedReader(source, archive_bytes)
        tee = _TeeReader(limited, spool)
        with tarfile.open(fileobj=tee, mode="r|") as incoming:
            for member in incoming:
                name = posixpath.normpath(member.name)
                if (
                    posixpath.isabs(member.name)
                    or name == ".."
                    or name.startswith("../")
                    or not (member.isreg() or member.isdir())
                ):
                    raise ValueError("Workspace archive members must be relative files.")
                if member.isreg():
                    self._require_path_allowed(name)
            parsed_bytes = incoming.offset
            while tee.read(1 << 16):
                pass
        if limited.read_bytes != archive_bytes:
            raise RuntimeError("Workspace archive stream did not match its declared size.")
        # tarfile can stop at a zero block or corrupt header without raising.
        # Readahead may already have read later headers, so inspect from the
        # last parsed member boundary rather than the reader's position.
        rest = spool if spool is not None else source
        rest.seek(origin + parsed_bytes)
        remaining = archive_bytes - parsed_bytes
        while remaining > 0 and (trailing := rest.read(min(1 << 16, remaining))):
            remaining -= len(trailing)
            if any(trailing):
                raise ValueError("Workspace archive has unvalidated trailing content.")
        return tee.hexdigest()

    def _split_archive(self, source: BinaryIO, archive_bytes: int) -> list[bytes]:
        chunks: list[bytes] = []
        pending: list[tuple[tarfile.TarInfo, bytes | None]] = []
        pending_bytes = 0
        limited = _LimitedReader(source, archive_bytes)
        with tarfile.open(fileobj=limited, mode="r|") as incoming:
            for member in incoming:
                name = posixpath.normpath(member.name)
                if (
                    posixpath.isabs(member.name)
                    or name == ".."
                    or name.startswith("../")
                    or not (member.isreg() or member.isdir())
                ):
                    raise ValueError("Workspace archive members must be relative files.")
                if member.isreg():
                    self._require_path_allowed(name)
                needed = _ARCHIVE_MEMBER_OVERHEAD + _padded(member.size)
                if _sealed_size(needed) > MAX_ARCHIVE_BYTES:
                    raise RuntimeError(
                        f"Workspace file is larger than one broker archive: {name[:200]}"
                    )
                if pending and _sealed_size(pending_bytes + needed) > MAX_ARCHIVE_BYTES:
                    chunks.append(_seal(pending))
                    pending, pending_bytes = [], 0
                extracted = incoming.extractfile(member) if member.isreg() else None
                pending.append((member, None if extracted is None else extracted.read()))
                pending_bytes += needed
            # Consume the end-of-archive padding so the size check sees every byte.
            while limited.read(1 << 16):
                pass
        if pending:
            chunks.append(_seal(pending))
        if limited.read_bytes != archive_bytes:
            raise RuntimeError("Workspace archive stream did not match its declared size.")
        return chunks


def _seal(entries: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:", format=tarfile.PAX_FORMAT) as archive:
        for member, content in entries:
            archive.addfile(member, None if content is None else io.BytesIO(content))
    return buffer.getvalue()


@dataclass
class _CopyBackBudget:
    per_file: int
    total_limit: int | None
    total: int = 0

    @property
    def remaining(self) -> float:
        return float("inf") if self.total_limit is None else self.total_limit - self.total

    def total_exceeded(self) -> RuntimeError:
        return RuntimeError(f"Workspace transfer exceeds max_total_bytes={self.total_limit}.")

    def admit(self, path: str, size: int) -> None:
        if size > self.per_file:
            raise RuntimeError(f"Workspace file exceeds max_file_bytes={self.per_file}: {path}")
        if size > self.remaining:
            raise self.total_exceeded()
        self.total += size


class _LimitedReader(io.RawIOBase):
    def __init__(self, source: BinaryIO, limit: int) -> None:
        self._source = source
        self._remaining = limit
        self.read_bytes = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        size = self._remaining if size is None or size < 0 else min(size, self._remaining)
        data = self._source.read(size)
        self._remaining -= len(data)
        self.read_bytes += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


class _TeeReader(io.RawIOBase):
    """Reads through ``source``, hashing every byte and copying it into ``sink`` if given."""

    def __init__(self, source: io.RawIOBase, sink: BinaryIO | None) -> None:
        self._source = source
        self._sink = sink
        self._digest = hashlib.sha256()

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self._source.read(size)
        if data:
            if self._sink is not None:
                self._sink.write(data)
            self._digest.update(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class _CountingWriter(io.RawIOBase):
    def __init__(self, destination: BinaryIO, *, limit: int | None) -> None:
        self._destination = destination
        self._limit = limit
        self.written = 0

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        size = len(data)
        if self._limit is not None and self.written + size > self._limit:
            raise RuntimeError(f"Workspace transfer exceeds max_archive_bytes={self._limit}.")
        self._destination.write(bytes(data))
        self.written += size
        return size


def validate_staged_copies_mode(value: object) -> StagedCopiesMode:
    if value == "auto" and type(value) is str:
        return "auto"
    if value == "never" and type(value) is str:
        return "never"
    if value == "always" and type(value) is str:
        return "always"
    raise ValueError("staged_copies must be 'auto', 'never' or 'always'.")


def _exec_usable(runner: BrokeredSandboxRunner) -> bool:
    try:
        runner._ensure_exec_open()
    except RuntimeError:
        return False
    return True


def _refused_remaining_lifetime(error: SandboxBrokerRefusedError) -> int | None:
    """The remaining lifetime a broker named when it refused a command's timeout."""

    if error.status != 422 or error.code != "sandbox_invalid":
        return None
    match = _REMAINING_LIFETIME_REFUSAL.search(str(error))
    if match is None:
        return None
    remaining = int(match.group(1))
    return remaining if remaining >= 1 else None


@dataclass
class _Dispatch:
    """What one command's start attempts sent to the broker."""

    attempt: asyncio.Future[SandboxCommand] | None = None
    outcome_unknown: bool = False
    cleanup_required: bool = False

    @property
    def definitively_refused(self) -> bool:
        attempt = self.attempt
        return (
            not self.outcome_unknown
            and attempt is not None
            and attempt.done()
            and not attempt.cancelled()
            and isinstance(attempt.exception(), SandboxBrokerRefusedError)
        )


def _kill_receipt(status: str) -> dict[str, Any]:
    return _cleanup_artifact(
        adapter=BROKERED_SANDBOX_CANDIDATE,
        action="kill_command",
        status=status,
        timeout_s=_CANCEL_SETTLE_SECONDS,
    )


def _padded(size: int) -> int:
    return (size + 511) // 512 * 512


def _sealed_size(member_bytes: int) -> int:
    """Upper bound of a sealed archive: members, terminator, then padding to a tar record."""

    return -(-(member_bytes + _ARCHIVE_END_BYTES) // tarfile.RECORDSIZE) * tarfile.RECORDSIZE


def _query_batches(paths: tuple[str, ...], *, prefix: str | None) -> list[tuple[str, ...]]:
    """Group paths so each archive read stays within the broker's path and URL bounds."""

    batches: list[tuple[str, ...]] = []
    current: list[str] = []
    budget = 0
    for path in paths:
        cost = len(quote(f"{prefix or ''}/{path}", safe="")) + len("&path=")
        if current and (len(current) >= MAX_ARCHIVE_PATHS or budget + cost > _ARCHIVE_QUERY_BUDGET):
            batches.append(tuple(current))
            current, budget = [], 0
        current.append(path)
        budget += cost
    if current:
        batches.append(tuple(current))
    return batches


def _argv(command: ExecCommand) -> list[str]:
    if command.kind == "process" and command.argv is not None:
        return list(command.argv)
    if command.shell is not None:
        return ["/bin/sh", "-c", command.shell]
    raise ValueError("Brokered sandbox commands need argv or a shell script.")


def _bounded_text(text: str, limit: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= limit:
        return text, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _fingerprint(runner: BrokeredSandboxRunner) -> str:
    settings = runner.broker.settings
    sandbox = runner.sandbox
    identity: dict[str, Any] = {
        "api_url": settings.api_url,
        "application_id": settings.application_id,
        "sandbox_id": sandbox.id,
        "image": sandbox.image,
        "egress": sandbox.egress.mode,
        "guarantees": None
        if sandbox.guarantees is None
        else sandbox.guarantees.model_dump(mode="json"),
        "root": runner.default_cwd,
    }
    if sandbox.egress.hosts is not None:
        # Only allowlist records carry hosts; the others keep their earlier fingerprint.
        identity["egress_hosts"] = list(sandbox.egress.hosts)
    if sandbox.image_identity is not None:
        # Records without an image identity keep their earlier fingerprint.
        identity["image_identity"] = sandbox.image_identity.model_dump(mode="json")
    return (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(identity, "brokered_sandbox_execution_identity")
        ).hexdigest()
    )


def _admission_candidate(
    runner: BrokeredSandboxRunner,
    executables: tuple[ExecutionExecutableEvidence, ...],
) -> ExecutionAdmissionCandidate:
    fingerprint = _fingerprint(runner)
    sandbox = runner.sandbox
    image_fingerprint = (
        None if sandbox.image_identity is None else sandbox.image_identity.fingerprint
    )
    return ExecutionAdmissionCandidate(
        candidate=BROKERED_SANDBOX_CANDIDATE,
        evidence=ExecutionCapabilityEvidence(
            subject=BROKERED_SANDBOX_CANDIDATE,
            claims=sandbox_capability_claims(
                sandbox.guarantees, egress_mode=sandbox.egress.mode, declared=False
            ),
            environment_fingerprint=fingerprint,
            image_fingerprint=image_fingerprint,
            toolchain_profile_fingerprint=runner.toolchain_profile_fingerprint,
            tool_requirements=ExecutionToolRequirementEvidence(
                environment_fingerprint=fingerprint,
                image_fingerprint=image_fingerprint,
                executables=executables,
            ),
        ),
    )


@dataclass(frozen=True, repr=False)
class _BrokeredAdmissionObserver(RunnerExecutionAdmissionObserver):
    """Probe executables in one exact, ready brokered sandbox.

    The broker's record is read before and after the probes; a different
    sandbox, image or guarantee set fails closed. Probes run through
    ``runner.exec``.
    """

    runner: BrokeredSandboxRunner
    state: dict[str, Any] = field(default_factory=dict)

    def snapshot(self):
        candidate = self.state.get("candidate")
        if candidate is not None and not self.runner.is_closed:
            return candidate
        probes = {probe.executable: probe for probe in self.requirements.executable_probes()}
        return _admission_candidate(
            self.runner,
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

    async def _observe(self) -> SandboxRecord:
        self.runner._ensure_exec_open()
        record = await self.runner.refresh()
        if record.status != "ready":
            raise SandboxLeaseLostError("Brokered sandbox is not ready for admission.")
        return record

    async def collect(self):
        from cayu.environments.admission import EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS

        self.state.pop("candidate", None)
        observed_at = datetime.now(UTC)
        valid_until = observed_at + timedelta(seconds=EXECUTION_LIVE_EVIDENCE_MAX_TTL_SECONDS)
        initial = await self._observe()
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
            result = await self.runner.exec(
                command,
                timeout_s=_ADMISSION_PROBE_TIMEOUT_SECONDS,
                output_limit_bytes=MIN_COMMAND_OUTPUT_BYTES,
            )
            if result.timed_out or result.cancelled:
                raise SandboxBrokerUnavailableError("Brokered admission probe did not complete.")
            accepted = (0,) if probe is None else probe.accepted_exit_codes
            available = result.exit_code in accepted
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
        final = await self._observe()
        if (final.id, final.image, final.egress, final.guarantees, final.image_identity) != (
            initial.id,
            initial.image,
            initial.egress,
            initial.guarantees,
            initial.image_identity,
        ):
            raise SandboxLeaseLostError("Brokered sandbox identity changed during admission.")
        candidate = _admission_candidate(self.runner, tuple(claims))
        self.state["candidate"] = candidate
        return candidate

    async def refresh(self):
        await self.collect()
