from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from cayu._validation import require_durable_nonblank
from cayu.credentials import CredentialMode, CredentialModeInput, normalize_credential_mode
from cayu.runners._secrets import (
    merge_secret_env_values,
    normalize_runner_secret_env,
    redact_exec_result,
    resolved_secret_redactor,
    validate_secret_env_collisions,
)
from cayu.runners._subprocess import (
    SubprocessCommand,
    copy_runner_env,
    merge_runner_env_overrides,
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
    copy_exec_command,
)
from cayu.vaults import (
    SecretEnv,
    SecretRedactor,
    SecretRef,
    SecretResolver,
    resolve_secret_env,
)

if TYPE_CHECKING:
    from cayu.environments.admission import (
        ExecutionAdmissionCandidate,
        ExecutionCapabilityEvidence,
    )

# Non-secret operational host variables forwarded when inherit_env is False so
# commands still resolve binaries and locale without seeing arbitrary host
# secrets (API keys, tokens, cloud credentials).
SAFE_LOCAL_ENV_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    # Windows equivalents.
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
)


class LocalRunner(Runner):
    """Executes local commands with cwd restricted under one root.

    This is not a sandbox. Commands still run with the permissions of the
    current OS user and can access absolute paths allowed by the OS.

    Host environment inheritance is fail-closed: by default commands see only
    a minimal operational base env (``SAFE_LOCAL_ENV_KEYS``) plus explicit
    per-call ``env``, so host secrets are not leaked into agent commands.
    Pass ``inherit_env=True`` to opt in to the full host environment.
    Declared ``secret_env`` entries are resolved through ``secret_resolver``
    at exec time and their values are redacted from captured output.
    """

    isolation = "local"

    def execution_capability_evidence(self) -> ExecutionCapabilityEvidence:
        """Declare the local-process boundary without representing it as isolation."""

        from cayu.environments.admission import (
            ExecutionCapabilityClaim,
            ExecutionCapabilityEvidence,
        )

        unsupported = {
            "untrusted_code_isolation": "local_process_isolation_unsupported",
            "real_credential_non_possession": "local_credential_boundary_unsupported",
            "deny_by_default_network": "local_network_boundary_unsupported",
            "brokered_egress": "local_network_boundary_unsupported",
            "guest_privilege_containment": "local_privilege_boundary_unsupported",
            "unprivileged_guest": "local_privilege_boundary_unsupported",
            "host_filesystem_isolation": "local_host_filesystem_boundary_unsupported",
            "read_only_host_inputs": "local_host_filesystem_boundary_unsupported",
            "reconnect": "reconnect_unsupported",
        }
        return ExecutionCapabilityEvidence(
            subject="local",
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
                            else "select_isolated_execution"
                        ),
                    )
                    for capability, reason_code in unsupported.items()
                ),
            ),
        )

    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate:
        """Expose the local runner's explicit non-isolation evidence to Cayu."""

        from cayu.environments.admission import ExecutionAdmissionCandidate

        return ExecutionAdmissionCandidate(
            candidate="local",
            evidence=self.execution_capability_evidence(),
        )

    def __init__(
        self,
        root: str | Path,
        *,
        inherit_env: bool = False,
        secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef] = (),
        secret_resolver: SecretResolver | None = None,
        credential_mode: CredentialModeInput = CredentialMode.RAW_ENV,
        allow_raw_secret_env: bool = True,
    ) -> None:
        if not isinstance(root, str | PathLike):
            raise TypeError("LocalRunner root must be a string or Path.")
        if not isinstance(inherit_env, bool):
            raise TypeError("LocalRunner inherit_env must be a bool.")
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise FileNotFoundError(f"Runner root does not exist: {root_path}")
        if not root_path.is_dir():
            raise NotADirectoryError(f"Runner root is not a directory: {root_path}")
        self.root = root_path
        self.inherit_env = inherit_env
        self.credential_mode = normalize_credential_mode(credential_mode)
        self._allow_raw_secret_env = allow_raw_secret_env
        case_sensitive_env = _local_environment_names_case_sensitive()
        self.secret_env, self.secret_resolver = normalize_runner_secret_env(
            secret_env,
            secret_resolver,
            credential_mode=self.credential_mode,
            allow_raw_secret_env=allow_raw_secret_env,
            case_sensitive_env_names=case_sensitive_env,
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
            raise TypeError("LocalRunner redactor must be a SecretRedactor.")
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
        """Validate Local's complete environment ownership contract."""

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
        SubprocessCommand,
        str,
        dict[str, str],
        tuple[str, ...],
        dict[str, SecretRef],
        SecretResolver | None,
        int | None,
        str | None,
        int | None,
        bool,
    ]:
        """Own one complete Local request before secret-state consultation."""

        if type(command) is not ExecCommand:
            raise TypeError("LocalRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        case_sensitive_env = _local_environment_names_case_sensitive()
        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(
            env,
            inherit_env=self.inherit_env,
            case_sensitive=case_sensitive_env,
        )
        validated_env_remove = validate_runner_env_remove(env_remove)
        timeout = validate_timeout(timeout_s)
        standard_input = validate_stdin(stdin)
        output_limit = validate_output_limit(output_limit_bytes)
        subprocess_command = _subprocess_command(command)
        if not self.inherit_env:
            safe_host_env = copy_runner_env(
                _safe_host_env(),
                inherit_env=False,
                case_sensitive=case_sensitive_env,
            )
            environment = merge_runner_env_overrides(
                safe_host_env,
                environment,
                case_sensitive=case_sensitive_env,
            )
        declared_secret_env, secret_resolver = normalize_runner_secret_env(
            self.secret_env,
            self.secret_resolver,
            credential_mode=self.credential_mode,
            allow_raw_secret_env=self._allow_raw_secret_env,
            case_sensitive_env_names=case_sensitive_env,
        )
        validate_secret_env_collisions(
            environment,
            declared_secret_env,
            case_sensitive=case_sensitive_env,
        )
        return (
            subprocess_command,
            working_dir,
            environment,
            validated_env_remove,
            declared_secret_env,
            secret_resolver,
            timeout,
            standard_input,
            output_limit,
            case_sensitive_env,
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
                subprocess_command,
                working_dir,
                environment,
                validated_env_remove,
                declared_secret_env,
                secret_resolver,
                timeout,
                standard_input,
                output_limit,
                case_sensitive_env,
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
            subprocess_command = None
            working_dir = ""
            environment = {}
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
        environment = remove_runner_env(
            environment,
            validated_env_remove,
            case_sensitive=case_sensitive_env,
        )
        invocation_redactor = resolved_secret_redactor(resolved_secrets).merged_with(
            output_redactor
        )
        subprocess_run = run_subprocess(
            subprocess_command,
            cwd=working_dir,
            env=environment,
            timeout_s=timeout,
            stdin=standard_input,
            output_limit_bytes=output_limit,
            output_redactor=invocation_redactor,
        )
        # Transfer the raw environment into the subprocess coroutine before
        # crossing an await; this frame must not retain it on cancellation.
        environment = {}
        result = await subprocess_run
        return redact_exec_result(result, resolved_secrets)

    def resolve_cwd(self, cwd: str | None = None) -> str:
        root_value = require_durable_nonblank(os.fspath(self.root), "root")
        root = Path(root_value)
        if not root.is_absolute():
            raise ValueError("LocalRunner root must be an absolute path.")
        root = root.resolve()
        if not root.exists():
            raise FileNotFoundError(f"Runner root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Runner root is not a directory: {root}")
        if cwd is None:
            return str(root)
        cwd = require_durable_nonblank(cwd, "cwd")
        candidate = Path(cwd)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            outside_message = "Runner cwd is outside the runner root."
        else:
            resolved = (root / candidate).resolve()
            outside_message = "Runner cwd escapes the runner root."
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(outside_message) from exc
        if not resolved.exists():
            raise FileNotFoundError(f"Runner cwd does not exist: {cwd}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Runner cwd is not a directory: {cwd}")
        return str(resolved)


def _safe_host_env() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_LOCAL_ENV_KEYS if key in os.environ}


def _local_environment_names_case_sensitive() -> bool:
    return os.name != "nt"


def _subprocess_command(command: ExecCommand) -> SubprocessCommand:
    try:
        owned_command = copy_exec_command(command)
    finally:
        del command
    if owned_command.kind == "process":
        if owned_command.argv is None:
            raise ValueError("Process commands require argv.")
        return SubprocessCommand(argv=owned_command.argv)
    if owned_command.shell is None:
        raise ValueError("Shell commands require a script.")
    return SubprocessCommand(shell=owned_command.shell)
