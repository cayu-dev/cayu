"""Send every command another runner runs through a caller-chosen launcher.

``LaunchedRunner`` wraps any ``Runner`` and rewrites each ``ExecCommand``
before the wrapped runner sees it: either by putting an argv prefix in front
(``sudo -u sandbox --``, ``bwrap ... --``, ``firejail --``, ``docker exec ...``)
or by calling a function that returns the command to run. Every command path
goes through the same rewrite, so wrapping one public method can no longer
leave the others outside the launcher.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import TYPE_CHECKING, Any, BinaryIO, TypeAlias

from cayu._validation import canonical_durable_json_bytes
from cayu.runners._admission_probes import EXECUTABLE_AVAILABILITY_SCRIPT
from cayu.runners._subprocess import copy_runner_env, validate_runner_env_remove
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerBinaryStreamCapability,
    RunnerExecutionAdmissionObserver,
    RunnerInteractiveProcess,
    RunnerInteractiveProcessCapability,
    RunnerLifecycleState,
    RunnerPortForwardCapability,
    RunnerSystemExecutionMode,
    RunnerWorkloadAuthority,
    RunnerWorkspaceCapabilityT,
    _clean_runner_preflight,
    copy_exec_command,
    runner_pending_command_settlement_cancellation_safe,
)

if TYPE_CHECKING:
    from cayu.environments.admission import (
        ExecutionAdmissionCandidate,
        ExecutionCapabilityClaim,
        ExecutionEnvironmentAuthority,
        ExecutionExecutableEvidence,
        ExecutionRequirements,
    )
    from cayu.vaults.redaction import SecretRedactor

CommandLauncher: TypeAlias = Sequence[str] | Callable[[ExecCommand], ExecCommand]
"""An argv prefix, or a function that returns the command to run instead."""

LAUNCHER_UNVERIFIED_REASON = "launcher_unverified"
"""Reason code on a wrapped runner's positive claim the launcher may not keep."""

LAUNCHER_UNVERIFIED_REMEDIATION = "declare_launcher_capability"

_DEFAULT_SHELL = ("sh", "-c")
_PROBE_TIMEOUT_SECONDS = 30
_PROBE_OUTPUT_LIMIT_BYTES = 4096
_PROBE_VALIDITY = timedelta(seconds=300)
_NEGATIVE_CLAIM_STATES = frozenset({"unsupported", "unverified"})
_LIFECYCLE_PRECEDENCE: tuple[RunnerLifecycleState, ...] = (
    "closed",
    "poisoned",
    "closing",
    "fenced",
)


class LaunchedRunner(Runner):
    """Run every command of another runner through a launcher.

    ``launcher`` is either an argv prefix or a function ``ExecCommand ->
    ExecCommand``. A prefix goes in front of process commands as they are; a
    shell command becomes ``[*prefix, *shell, script]``, with ``shell``
    defaulting to ``("sh", "-c")``. A prefix that doesn't end with ``--``
    refuses a command whose first argument starts with ``-``, so a command
    can't pass options to the launcher. A function receives a private copy of
    each command and must return an ``ExecCommand``; it decides that itself.
    It can be called more than once for one command (preflight, then
    dispatch), so keep it deterministic and free of side effects.

    A command's ``env`` and ``env_remove`` never reach the launcher's own
    process. They are applied inside it, by prefixing the command with
    ``env -u NAME ... -- NAME=VALUE ...`` (a function launcher receives that
    ``env`` command). The wrapped runner starts the launcher with its own
    environment, so a command can't point ``PATH`` or ``LD_PRELOAD`` at a
    workspace file and replace or hijack the launcher. The values do appear
    in the launcher's argument list; keep secrets in the wrapped runner's
    secret environment, which the launcher inherits.

    The rewrite covers ``exec``, ``exec_redacted``, ``exec_stream`` (when the
    wrapped runner supports binary streams), interactive processes (when it
    offers them), ``preflight_exec`` and the executable probes of admission,
    so built-in tools cannot reach the wrapped runner around the launcher.
    Durable command receipts are not offered: tools fall back to the ordinary
    redacted ``exec`` path, which goes through the launcher.

    ``exec_system`` runs Cayu's own control-plane commands (for example the
    ephemeral Git baseline of a coding sandbox). It goes to the wrapped
    runner's ``exec_system`` without the launcher, so this runner declares
    ``system_execution_mode = "separate"``.

    Admission evidence never claims more than the launcher can show. The
    wrapped runner's negative claims (unsupported, unverified) pass through;
    each positive claim becomes ``unverified`` with reason
    ``launcher_unverified``, because a launcher can change where and as whom a
    command runs. Name the capabilities the launcher keeps in
    ``preserve_claims`` to pass those claims through unchanged, and add your
    own declarations for what the launcher itself provides in ``claims``
    (for example ``deny_by_default_network`` for ``bwrap --unshare-net``).
    Those are your statements; Cayu does not check them. Executables are
    probed through the launcher, and the wrapped runner's image and toolchain
    fingerprints are folded into this runner's environment fingerprint rather
    than passed through.

    The wrapper owns the runner it wraps: ``close()`` closes it.
    """

    system_execution_mode: RunnerSystemExecutionMode = "separate"
    # The waiter delegates only to a wrapped runner that declares the same, and
    # fails closed otherwise (see await_pending_command_settlement).
    pending_command_settlement_cancellation_safe = True

    def __init__(
        self,
        runner: Runner,
        launcher: CommandLauncher,
        *,
        shell: Sequence[str] = _DEFAULT_SHELL,
        preserve_claims: Iterable[str] = (),
        claims: Iterable[ExecutionCapabilityClaim] = (),
    ) -> None:
        from cayu.environments.admission import ExecutionCapabilityClaim

        if not isinstance(runner, Runner):
            raise TypeError("LaunchedRunner runner must be a Runner.")
        self._runner = runner
        self._shell = _argv_tuple(shell, "shell")
        self._prefix: tuple[str, ...] | None
        self._launcher: Callable[[ExecCommand], ExecCommand] | None
        if isinstance(launcher, str | bytes | Sequence):
            self._prefix = _argv_tuple(launcher, "launcher")
            self._launcher = None
            launcher_identity: dict[str, object] = {
                "prefix": list(self._prefix),
                "shell": list(self._shell),
            }
        elif callable(launcher):
            self._prefix = None
            self._launcher = launcher
            launcher_identity = {"callable": secrets.token_hex(16)}
        else:
            raise TypeError("LaunchedRunner launcher must be an argv prefix or a function.")
        self._launcher_identity = launcher_identity
        # Distinguishes wrappers of runners that publish no fingerprint of their own.
        self._instance_identity = secrets.token_hex(16)
        if isinstance(preserve_claims, str):
            raise TypeError("preserve_claims must be an iterable of capability names.")
        preserved: set[str] = set()
        for name in preserve_claims:
            if type(name) is not str or not name.strip():
                raise TypeError("preserve_claims entries must be capability names.")
            preserved.add(name)
        self._preserve_claims = frozenset(preserved)
        declared: dict[str, ExecutionCapabilityClaim] = {}
        for claim in claims:
            if type(claim) is not ExecutionCapabilityClaim:
                raise TypeError("LaunchedRunner claims must be ExecutionCapabilityClaim values.")
            if claim.state == "live_verified":
                raise ValueError(
                    "A launcher declaration cannot be live-verified evidence; "
                    "declare it as declared or available."
                )
            if claim.capability in declared:
                raise ValueError(f"Capability {claim.capability!r} is claimed more than once.")
            declared[claim.capability] = claim
        self._claims = tuple(declared[name] for name in sorted(declared))
        self.isolation = runner.isolation
        self.default_cwd = runner.default_cwd

    def launch(
        self,
        command: ExecCommand,
        *,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
    ) -> ExecCommand:
        """Return the command the wrapped runner receives for ``command``.

        ``env`` and ``env_remove`` are applied inside the launcher by an
        ``env`` command, never to the launcher's own process: a command that
        set ``PATH`` or ``LD_PRELOAD`` could otherwise replace the launcher, or
        run code in it, before the sandbox starts. The wrapped runner starts
        the launcher with its own environment only.
        """

        owned = copy_exec_command(command)
        owned_env = copy_runner_env(env, inherit_env=False)
        removed = validate_runner_env_remove(env_remove)
        if owned_env or removed:
            # The wrapped runners apply env_remove after env, so removal wins.
            assignments = [
                f"{name}={value}" for name, value in owned_env.items() if name not in removed
            ]
            owned = ExecCommand.process(
                "env",
                *(item for name in removed for item in ("-u", name)),
                "--",
                *assignments,
                *_argv(owned, self._shell),
            )
        if self._prefix is not None:
            argv = _argv(owned, self._shell)
            if self._prefix[-1] != "--" and argv[0].startswith("-"):
                raise ValueError(
                    "A launched command cannot start with an option unless the launcher "
                    "prefix ends with '--'."
                )
            return ExecCommand.process(*self._prefix, *argv)
        assert self._launcher is not None
        launched = self._launcher(owned)
        if type(launched) is not ExecCommand:
            raise TypeError("LaunchedRunner launcher must return an ExecCommand.")
        return copy_exec_command(launched)

    # Lifecycle

    @property
    def lifecycle_state(self) -> RunnerLifecycleState:
        """Include the wrapped runner's execution eligibility."""

        outer = super().lifecycle_state
        inner = self._runner.lifecycle_state
        for state in _LIFECYCLE_PRECEDENCE:
            if state in (outer, inner):
                return state
        return "reusable"

    @property
    def loss_reason(self) -> str | None:
        """Report the wrapper's own loss reason, else the wrapped runner's."""

        own = super().loss_reason
        return own if own is not None else self._runner.loss_reason

    @property
    def is_closed(self) -> bool:
        return self._closed or self._runner.is_closed

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        return self._runner.resource_key

    async def close(self) -> None:
        """Close the wrapped runner, then this one."""

        if self._closed:
            return
        self._close_exec("runner finalization has started")
        await self._runner.close()
        self._closed = True

    async def await_pending_command_settlement(self) -> bool:
        """Delegate to the wrapped runner's waiter only when it is cancellation-safe."""

        if not runner_pending_command_settlement_cancellation_safe(self._runner):
            return False
        return await self._runner.await_pending_command_settlement()

    def reopen_exec(self) -> None:
        self._runner.reopen_exec()
        super().reopen_exec()

    def resolve_cwd(self, cwd: str | None = None) -> str:
        return self._runner.resolve_cwd(cwd)

    # Commands

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
        """Validate the launched command against the wrapped runner's rules."""

        self._ensure_exec_open()
        self._runner.preflight_exec(
            self.launch(command, env=env, env_remove=env_remove),
            **_command_kwargs(cwd, None, (), timeout_s, stdin, output_limit_bytes),
        )

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
        return await self._runner.exec(
            self.launch(command, env=env, env_remove=env_remove),
            **_command_kwargs(cwd, None, (), timeout_s, stdin, output_limit_bytes),
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
        self._ensure_exec_open()
        return await self._runner.exec_redacted(
            self.launch(command, env=env, env_remove=env_remove),
            redactor=redactor,
            **_command_kwargs(cwd, None, (), timeout_s, stdin, output_limit_bytes),
        )

    async def exec_system(
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
        """Run a Cayu control-plane command on the wrapped runner, unlaunched."""

        self._ensure_exec_open()
        return await self._runner.exec_system(
            command,
            **_command_kwargs(cwd, env, env_remove, timeout_s, stdin, output_limit_bytes),
        )

    def binary_stream_capability(self) -> RunnerBinaryStreamCapability | None:
        if self._runner.binary_stream_capability() is None:
            return None
        return _LaunchedBinaryStreamCapability(self)

    async def exec_stream(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: BinaryIO | None = None,
        stdout: BinaryIO | None = None,
        stdout_limit_bytes: int | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        """Run a launched command with binary stdin or stdout."""

        self._ensure_exec_open()
        inner = self._runner.binary_stream_capability()
        if inner is None:
            raise RuntimeError("The wrapped runner does not support binary streams.")
        kwargs: dict[str, Any] = {
            **_command_kwargs(cwd, None, (), timeout_s, None, output_limit_bytes),
            "stdin": stdin,
            "stdout": stdout,
            "stdout_limit_bytes": stdout_limit_bytes,
        }
        return await inner.exec_stream(
            self.launch(command, env=env, env_remove=env_remove), **kwargs
        )

    def interactive_process_capability(self) -> RunnerInteractiveProcessCapability | None:
        inner = self._runner.interactive_process_capability()
        if inner is None:
            return None
        return _LaunchedInteractiveProcessCapability(self, inner)

    def port_forward_capability(self) -> RunnerPortForwardCapability | None:
        """Forward the wrapped runner's guest listener; it runs no command."""

        return self._runner.port_forward_capability()

    def workspace_capability(
        self,
        capability_type: type[RunnerWorkspaceCapabilityT],
    ) -> RunnerWorkspaceCapabilityT | None:
        """Forward native file capabilities; they move files and run no command."""

        return self._runner.workspace_capability(capability_type)

    # Declarations

    def workload_authority(self, name: str) -> RunnerWorkloadAuthority | None:
        return self._runner.workload_authority(name)

    def output_secret_values_present(self) -> bool | None:
        return self._runner.output_secret_values_present()

    def browser_control_endpoint_reachable(self, endpoint: str) -> bool | None:
        """Keep the wrapped runner's denials; a launcher may change what is reachable."""

        reachable = self._runner.browser_control_endpoint_reachable(endpoint)
        return False if reachable is False else None

    def browser_recording_supported(self) -> bool:
        return self._runner.browser_recording_supported()

    def execution_environment_authority(self) -> ExecutionEnvironmentAuthority:
        """Keep the boundary that owns the wrapped runner, such as its factory."""

        return self._runner.execution_environment_authority()

    # Admission

    async def refresh_execution_admission(self) -> None:
        await self._runner.refresh_execution_admission()

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        return self._launched_candidate(self._runner.execution_admission_candidate())

    async def collect_execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None:
        return self._launched_candidate(await self._runner.collect_execution_admission_candidate())

    def execution_admission_candidate_for(
        self, requirements: ExecutionRequirements
    ) -> ExecutionAdmissionCandidate | None:
        return self.execution_admission_observer(requirements).snapshot()

    def execution_admission_observer(
        self, requirements: ExecutionRequirements
    ) -> RunnerExecutionAdmissionObserver:
        return _LaunchedAdmissionObserver(
            self,
            requirements,
            inner=self._runner.execution_admission_observer(
                _requirements_without_executables(requirements)
            ),
        )

    async def collect_execution_admission_candidate_for(
        self, requirements: ExecutionRequirements
    ) -> ExecutionAdmissionCandidate | None:
        return await self.execution_admission_observer(requirements).collect()

    def _launched_candidate(
        self,
        supplied: object,
        executables: tuple[ExecutionExecutableEvidence, ...] | None = None,
    ) -> ExecutionAdmissionCandidate | None:
        """Restate the wrapped runner's evidence for launched commands."""

        from cayu.environments.admission import (
            ExecutionAdmissionCandidate,
            ExecutionCapabilityClaim,
            ExecutionCapabilityEvidence,
            ExecutionToolRequirementEvidence,
            _copy_execution_admission_candidate,
        )

        if supplied is None:
            return None
        candidate = _copy_execution_admission_candidate(supplied)
        if candidate is None:
            raise RuntimeError("The wrapped runner returned malformed admission evidence.")
        evidence = candidate.evidence
        claims: dict[str, ExecutionCapabilityClaim] = {}
        for claim in evidence.claims:
            if claim.capability in self._preserve_claims or claim.state in _NEGATIVE_CLAIM_STATES:
                claims[claim.capability] = claim
            else:
                claims[claim.capability] = ExecutionCapabilityClaim.unverified(
                    claim.capability,
                    reason_code=LAUNCHER_UNVERIFIED_REASON,
                    remediation_code=LAUNCHER_UNVERIFIED_REMEDIATION,
                )
        for claim in self._claims:
            claims[claim.capability] = claim
        inner_fingerprint = evidence.environment_fingerprint
        fingerprint = None
        if inner_fingerprint is not None or executables is not None:
            material: dict[str, object] = {
                "launcher": self._launcher_identity,
                "environment_fingerprint": inner_fingerprint,
                "image_fingerprint": evidence.image_fingerprint,
                "toolchain_profile_fingerprint": evidence.toolchain_profile_fingerprint,
            }
            if inner_fingerprint is None:
                material["wrapper"] = self._instance_identity
            fingerprint = (
                "sha256:"
                + sha256(
                    canonical_durable_json_bytes(material, "launched_execution_identity")
                ).hexdigest()
            )
        ordered = tuple(claims[name] for name in sorted(claims))
        return ExecutionAdmissionCandidate(
            candidate=candidate.candidate,
            evidence=ExecutionCapabilityEvidence(
                subject=evidence.subject,
                claims=ordered,
                unclaimed_reason_code=None
                if ordered
                else evidence.unclaimed_reason_code or "capabilities_unclaimed",
                environment_fingerprint=fingerprint,
                tool_requirements=None
                if executables is None or fingerprint is None
                else ExecutionToolRequirementEvidence(
                    environment_fingerprint=fingerprint,
                    executables=executables,
                ),
            ),
        )

    async def _probe_executable(
        self, name: str, *, arguments: tuple[str, ...] | None, accepted: tuple[int, ...]
    ) -> bool:
        command = (
            ExecCommand.process("sh", "-c", EXECUTABLE_AVAILABILITY_SCRIPT, "cayu-admission", name)
            if arguments is None
            else ExecCommand.process(name, *arguments)
        )
        result = await self.exec(
            command,
            timeout_s=_PROBE_TIMEOUT_SECONDS,
            output_limit_bytes=_PROBE_OUTPUT_LIMIT_BYTES,
        )
        return not result.timed_out and not result.cancelled and result.exit_code in accepted


@dataclass
class _LaunchedAdmissionState:
    collected: bool = False
    inner_fingerprint: str | None = None
    executables: tuple[ExecutionExecutableEvidence, ...] | None = None


def _environment_fingerprint(candidate: object) -> str | None:
    evidence = getattr(candidate, "evidence", None)
    fingerprint = getattr(evidence, "environment_fingerprint", None)
    return fingerprint if type(fingerprint) is str else None


@dataclass(frozen=True, repr=False)
class _LaunchedAdmissionObserver(RunnerExecutionAdmissionObserver):
    """Probe executables through the launcher; restate the wrapped runner's claims."""

    runner: LaunchedRunner
    inner: RunnerExecutionAdmissionObserver
    state: _LaunchedAdmissionState = field(default_factory=_LaunchedAdmissionState)

    def snapshot(self) -> ExecutionAdmissionCandidate | None:
        """Restate the wrapped runner's current evidence.

        Claims always come from the wrapped runner's current snapshot.
        Executables observed through the launcher are reused only while the
        wrapped environment keeps the fingerprint it had when they were
        probed; otherwise, and before collection, they are only declared.
        """

        from cayu.environments.admission import ExecutionExecutableEvidence

        current = self.inner.snapshot()
        names = self.requirements.executable_names()
        if not names:
            return self.runner._launched_candidate(current, None)
        if (
            self.state.collected
            and current is not None
            and _environment_fingerprint(current) == self.state.inner_fingerprint
        ):
            return self.runner._launched_candidate(current, self.state.executables)
        probes = {probe.executable: probe for probe in self.requirements.executable_probes()}
        return self.runner._launched_candidate(
            current,
            tuple(
                ExecutionExecutableEvidence(
                    executable=name,
                    state="declared",
                    requirement_fingerprint=None
                    if name not in probes
                    else probes[name].fingerprint,
                )
                for name in names
            ),
        )

    async def collect(self) -> ExecutionAdmissionCandidate | None:
        """Collect the wrapped runner's claims, then probe executables through the launcher."""

        from cayu.environments.admission import ExecutionExecutableEvidence

        self.state.collected = False
        self.state.executables = None
        supplied = await self.inner.collect()
        names = self.requirements.executable_names()
        executables: tuple[ExecutionExecutableEvidence, ...] | None = None
        if names and supplied is not None:
            probes = {probe.executable: probe for probe in self.requirements.executable_probes()}
            observed_at = datetime.now(UTC)
            claims = []
            for name in names:
                probe = probes.get(name)
                available = await self.runner._probe_executable(
                    name,
                    arguments=None if probe is None else probe.probe_arguments,
                    accepted=(0,) if probe is None else probe.accepted_exit_codes,
                )
                claims.append(
                    ExecutionExecutableEvidence(
                        executable=name,
                        state="live_verified" if available else "unavailable",
                        observed_at=observed_at if available else None,
                        valid_until=observed_at + _PROBE_VALIDITY if available else None,
                        requirement_fingerprint=None if probe is None else probe.fingerprint,
                        reason_code=None if available else "executable_unavailable",
                        remediation_code=None if available else "install_executable",
                    )
                )
            executables = tuple(claims)
        candidate = self.runner._launched_candidate(supplied, executables)
        if supplied is not None:
            self.state.inner_fingerprint = _environment_fingerprint(supplied)
            self.state.executables = executables
            self.state.collected = True
        return candidate

    async def refresh(self) -> None:
        await self.collect()


class _LaunchedBinaryStreamCapability(RunnerBinaryStreamCapability):
    def __init__(self, runner: LaunchedRunner) -> None:
        self._runner = runner

    async def exec_stream(self, command: ExecCommand, **kwargs: Any) -> ExecResult:
        return await self._runner.exec_stream(command, **kwargs)


class _LaunchedInteractiveProcessCapability(RunnerInteractiveProcessCapability):
    def __init__(self, runner: LaunchedRunner, inner: RunnerInteractiveProcessCapability) -> None:
        self._runner = runner
        self._inner = inner

    @property
    def secret_values(self) -> tuple[str, ...]:
        return self._inner.secret_values

    async def open_interactive_process(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> RunnerInteractiveProcess:
        self._runner._ensure_exec_open()
        return await self._inner.open_interactive_process(
            self._runner.launch(command, env=env), cwd=cwd, timeout_s=timeout_s
        )


def _argv(command: ExecCommand, shell: tuple[str, ...]) -> list[str]:
    if command.kind == "process":
        return list(command.argv or ())
    return [*shell, command.shell or ""]


def _argv_tuple(value: object, name: str) -> tuple[str, ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"LaunchedRunner {name} must be a sequence of argv strings.")
    items: list[str] = []
    for item in value:
        if type(item) is not str:
            raise TypeError(f"LaunchedRunner {name} entries must be strings.")
        items.append(item)
    if not items:
        raise ValueError(f"LaunchedRunner {name} must not be empty.")
    # ExecCommand applies the same argv rules the runners enforce.
    ExecCommand.process(*items)
    return tuple(items)


def _command_kwargs(
    cwd: str | None,
    env: dict[str, str] | None,
    env_remove: tuple[str, ...],
    timeout_s: int | None,
    stdin: str | None,
    output_limit_bytes: int | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "timeout_s": timeout_s,
        "stdin": stdin,
        "output_limit_bytes": output_limit_bytes,
    }
    # Custom runners written before env_remove existed only receive it when used.
    if type(env_remove) is not tuple or env_remove:
        kwargs["env_remove"] = env_remove
    return kwargs


def _requirements_without_executables(
    requirements: ExecutionRequirements,
) -> ExecutionRequirements:
    """The same policy minus executables, which the wrapper probes itself."""

    from cayu.environments.admission import ExecutionRequirements

    payload = requirements.model_dump(mode="python", warnings=False)
    payload["required_executables"] = ()
    clauses = []
    for clause in payload["tool_requirements"]:
        alternatives = [
            alternative
            for alternative in clause["requirement"]["alternatives"]
            if alternative.get("kind") != "executable"
        ]
        if alternatives:
            clause["requirement"]["alternatives"] = alternatives
            clauses.append(clause)
    payload["tool_requirements"] = clauses
    return ExecutionRequirements.model_validate(payload)


def _runs_from_local_host_root(runner: object) -> bool:
    """Whether ``runner`` starts its commands in a ``LocalRunner``'s host directories.

    True for a ``LocalRunner`` and for ``LaunchedRunner`` wrappers around one,
    whose ``resolve_cwd`` delegates to it. Cayu uses this to pair a
    ``LocalWorkspace`` with the runner; the launcher decides what the command
    then sees, as it does for every cwd.
    """

    from cayu.runners.local import LocalRunner

    while isinstance(runner, LaunchedRunner):
        runner = runner._runner
    return isinstance(runner, LocalRunner)


__all__ = [
    "LAUNCHER_UNVERIFIED_REASON",
    "LAUNCHER_UNVERIFIED_REMEDIATION",
    "CommandLauncher",
    "LaunchedRunner",
]
