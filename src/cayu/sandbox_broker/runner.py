"""Run commands and file operations in a brokered sandbox through the broker's API."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import posixpath
import secrets
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, BinaryIO
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
    RunnerSystemExecutionMode,
    attach_cancellation_artifacts,
    copy_exec_command,
)
from cayu.sandbox_broker.client import (
    JSON_OUTPUT_EXPANSION,
    MAX_ARCHIVE_BYTES,
    MAX_ARCHIVE_PATHS,
    MAX_COMMAND_OUTPUT_BYTES,
    MAX_WAIT_SECONDS,
    MIN_COMMAND_OUTPUT_BYTES,
    SandboxArchiveTooLargeError,
    SandboxBrokerClient,
    SandboxBrokerError,
    SandboxBrokerRefusedError,
    SandboxBrokerUnavailableError,
    SandboxBusyError,
    SandboxCommand,
    SandboxGuarantees,
    SandboxLeaseLostError,
    SandboxNotFoundError,
    SandboxNotReadyError,
    SandboxRecord,
    validate_command_shape,
)
from cayu.sandbox_broker.lease import SandboxLeaseKeeper
from cayu.workspaces import RunnerWorkspace
from cayu.workspaces.base import TarStreamReadResult

__all__ = [
    "BROKERED_SANDBOX_CANDIDATE",
    "BrokeredSandboxRunner",
    "BrokeredSandboxWorkspace",
    "sandbox_capability_claims",
]

_LOGGER = logging.getLogger(__name__)
BROKERED_SANDBOX_ROOT = "/workspace"
#: Admission candidate and evidence subject for brokered sandboxes.
BROKERED_SANDBOX_CANDIDATE = "cayu-cloud-sandbox"
_BUSY_RETRY_SECONDS = 120.0
_START_RECOVERY_SECONDS = 120.0
_POLL_RECOVERY_GRACE_SECONDS = 120.0
_CANCEL_SETTLE_SECONDS = 30.0
_DEFAULT_TIMEOUT_SECONDS = 300
_ADMISSION_PROBE_TIMEOUT_SECONDS = 10
# Leave room in each archive for tar headers, long-name records and padding.
_ARCHIVE_MEMBER_OVERHEAD = 4 * 1024
_ARCHIVE_END_BYTES = 1024
_ARCHIVE_QUERY_BUDGET = 6 * 1024
#: The largest file the brokered workspace can move in one archive.
MAX_TRANSFER_FILE_BYTES = MAX_ARCHIVE_BYTES - 64 * 1024


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

    The network is denied by default only with ``egress_mode="none"``.
    ``public_web`` reaches the open internet through a proxy that logs but
    denies nothing, so it never backs ``deny_by_default_network``. Neither
    mode backs ``brokered_egress``, which needs a Runtime policy broker in the
    network path.
    """

    items = set() if guarantees is None or guarantees.version < 1 else set(guarantees.items)
    backing = {
        "untrusted_code_isolation": "untrusted_code_isolation",
        "host_filesystem_isolation": "untrusted_code_isolation",
        "real_credential_non_possession": "no_provider_credentials",
        "guest_privilege_containment": "unprivileged_commands",
        "unprivileged_guest": "unprivileged_commands",
    }
    if egress_mode == "none":
        backing["deny_by_default_network"] = "egress_only_via_broker"
    claims: list[ExecutionCapabilityClaim] = []
    for capability, item in backing.items():
        if declared:
            claims.append(ExecutionCapabilityClaim.declared(capability))
        elif item in items:
            claims.append(ExecutionCapabilityClaim.available(capability))
        else:
            claims.append(
                ExecutionCapabilityClaim.unsupported(
                    capability,
                    reason_code="sandbox_guarantee_missing",
                    remediation_code="select_guaranteed_sandbox",
                )
            )
    if egress_mode != "none":
        claims.append(
            ExecutionCapabilityClaim.unsupported(
                "deny_by_default_network",
                reason_code="sandbox_egress_public_web",
                remediation_code="select_no_egress",
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
    lifetime (a command without a timeout may run until the sandbox's maximum
    lifetime). A dropped connection is recovered by replaying the same id,
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
        self._client = client
        self.sandbox = sandbox
        self.lease_keeper = lease_keeper
        self.destroy_on_close = destroy_on_close
        self.poll_wait_seconds = poll_wait_seconds
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

    async def refresh(self) -> SandboxRecord:
        self.sandbox = await self._client.get(self.sandbox.id)
        return self.sandbox

    # Admission --------------------------------------------------------------

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        return _admission_candidate(self, ())

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
            except SandboxBrokerRefusedError:
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
            if attempt.done() and not attempt.cancelled():
                error = attempt.exception()
                if isinstance(error, SandboxBrokerRefusedError) and not dispatch.outcome_unknown:
                    return True  # Every start was definitively refused.
            while loop.time() < deadline:
                try:
                    command = await self._client.cancel_command(self.sandbox.id, command_id)
                    while not command.finished and loop.time() < deadline:
                        command = await self._client.get_command(
                            self.sandbox.id,
                            command_id,
                            wait_seconds=min(5, self.poll_wait_seconds),
                        )
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
        if not settled:
            self._poison_exec("the broker could not confirm that an abandoned command stopped")
        return settled

    def _timeout_seconds(self, timeout: int | None) -> int:
        remaining = self._remaining_lifetime_seconds()
        if timeout is None:
            return _DEFAULT_TIMEOUT_SECONDS if remaining is None else max(1, remaining)
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
        self._close_exec("the brokered sandbox is no longer ready")
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
            self._close_exec("the brokered sandbox's lease was lost")
            raise


class BrokeredSandboxWorkspace(RunnerWorkspace):
    """``RunnerWorkspace`` whose bulk copies go through the broker's archive routes.

    File tools run through commands like any runner workspace. Copy-in
    uploads the tree in archives of at most ``MAX_ARCHIVE_BYTES`` each, after
    checking every member against the workspace's exclusions; copy-back reads
    the named files in batches with the selective archive read, splitting a
    batch the broker reports as too large. One file must fit in one archive.
    """

    def __init__(self, runner: BrokeredSandboxRunner, **kwargs: Any) -> None:
        if not isinstance(runner, BrokeredSandboxRunner):
            raise TypeError("BrokeredSandboxWorkspace requires a BrokeredSandboxRunner.")
        super().__init__(runner, **kwargs)
        self._brokered_runner = runner

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
                raise RuntimeError(
                    f"Workspace file is larger than one broker archive: {batch[0]}"
                ) from None
            middle = len(batch) // 2
            await self._stream_batch(batch[:middle], destination, budget)
            await self._stream_batch(batch[middle:], destination, budget)
            return
        if result.missing_paths:
            raise FileNotFoundError(
                "Workspace files disappeared before copy-back: "
                + ", ".join(sorted(result.missing_paths))[:500]
            )
        wanted = set(batch)
        found: set[str] = set()
        with tarfile.open(fileobj=io.BytesIO(result.data), mode="r:") as archive:
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


@dataclass
class _Dispatch:
    """What one command's start attempts sent to the broker."""

    attempt: asyncio.Future[SandboxCommand] | None = None
    outcome_unknown: bool = False


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
    return (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(
                {
                    "api_url": settings.api_url,
                    "application_id": settings.application_id,
                    "sandbox_id": sandbox.id,
                    "image": sandbox.image,
                    "egress": sandbox.egress.mode,
                    "guarantees": None
                    if sandbox.guarantees is None
                    else sandbox.guarantees.model_dump(mode="json"),
                    "root": runner.default_cwd,
                },
                "brokered_sandbox_execution_identity",
            )
        ).hexdigest()
    )


def _admission_candidate(
    runner: BrokeredSandboxRunner,
    executables: tuple[ExecutionExecutableEvidence, ...],
) -> ExecutionAdmissionCandidate:
    fingerprint = _fingerprint(runner)
    sandbox = runner.sandbox
    return ExecutionAdmissionCandidate(
        candidate=BROKERED_SANDBOX_CANDIDATE,
        evidence=ExecutionCapabilityEvidence(
            subject=BROKERED_SANDBOX_CANDIDATE,
            claims=sandbox_capability_claims(
                sandbox.guarantees, egress_mode=sandbox.egress.mode, declared=False
            ),
            environment_fingerprint=fingerprint,
            tool_requirements=ExecutionToolRequirementEvidence(
                environment_fingerprint=fingerprint,
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
        if (final.id, final.image, final.guarantees) != (
            initial.id,
            initial.image,
            initial.guarantees,
        ):
            raise SandboxLeaseLostError("Brokered sandbox identity changed during admission.")
        candidate = _admission_candidate(self.runner, tuple(claims))
        self.state["candidate"] = candidate
        return candidate

    async def refresh(self):
        await self.collect()
