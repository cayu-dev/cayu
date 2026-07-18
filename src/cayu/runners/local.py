from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

from cayu._validation import require_nonblank
from cayu.credentials import CredentialMode, CredentialModeInput, normalize_credential_mode
from cayu.runners._secrets import (
    merge_secret_env_values,
    normalize_runner_secret_env,
    redact_exec_result,
)
from cayu.runners._subprocess import (
    SubprocessCommand,
    copy_runner_env,
    run_subprocess,
)
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
)
from cayu.vaults import SecretEnv, SecretRef, SecretResolver, resolve_secret_env

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
        self.secret_env, self.secret_resolver = normalize_runner_secret_env(
            secret_env,
            secret_resolver,
            credential_mode=self.credential_mode,
            allow_raw_secret_env=allow_raw_secret_env,
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
            raise TypeError("LocalRunner command must be an ExecCommand.")
        self._ensure_exec_open()
        working_dir = self.resolve_cwd(cwd)
        environment = copy_runner_env(env, inherit_env=self.inherit_env)
        if not self.inherit_env:
            environment = {**_safe_host_env(), **environment}
        resolved_secrets = (
            await resolve_secret_env(self.secret_env, self.secret_resolver)
            if self.secret_env and self.secret_resolver is not None
            else {}
        )
        environment = merge_secret_env_values(environment, resolved_secrets)
        subprocess_command = _subprocess_command(command)
        result = await run_subprocess(
            subprocess_command,
            cwd=working_dir,
            env=environment,
            timeout_s=timeout_s,
            stdin=stdin,
            output_limit_bytes=output_limit_bytes,
        )
        return redact_exec_result(result, resolved_secrets)

    def resolve_cwd(self, cwd: str | None = None) -> str:
        if cwd is None:
            return str(self.root)
        cwd = require_nonblank(cwd, "cwd")
        candidate = Path(cwd)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            outside_message = "Runner cwd is outside the runner root."
        else:
            resolved = (self.root / candidate).resolve()
            outside_message = "Runner cwd escapes the runner root."
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(outside_message) from exc
        if not resolved.exists():
            raise FileNotFoundError(f"Runner cwd does not exist: {cwd}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Runner cwd is not a directory: {cwd}")
        return str(resolved)


def _safe_host_env() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_LOCAL_ENV_KEYS if key in os.environ}


def _subprocess_command(command: ExecCommand) -> SubprocessCommand:
    if command.kind == "process":
        if command.argv is None:
            raise ValueError("Process commands require argv.")
        return SubprocessCommand(argv=command.argv)
    if command.shell is None:
        raise ValueError("Shell commands require a script.")
    return SubprocessCommand(shell=command.shell)
