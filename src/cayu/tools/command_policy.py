from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Mapping

from cayu._validation import require_clean_nonblank
from cayu.runners.base import is_same_or_child
from cayu.tools.base import ToolContext
from cayu.tools.commands import (
    MAX_TIMEOUT_SECONDS,
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
)
from cayu.tools.process_diagnostics import (
    ProcessCommandCapabilities,
    ProcessCommandDiagnostic,
)
from cayu.tools.process_diagnostics import (
    ProcessCommandDenialCode as Code,
)

DEFAULT_MAX_ENV_VALUE_BYTES = 4096
DEFAULT_MAX_STDIN_BYTES = 64 * 1024

_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


class ProcessCommandPolicy(CommandPolicy):
    """Deny-by-default policy for model-controlled command execution.

    Executable identities are matched exactly. Working-directory roots are
    compared against the runner-resolved ``canonical_cwd`` supplied by
    :class:`~cayu.tools.commands.ExecCommandTool`.

    ``public_executable_names`` and ``public_environment_names`` explicitly attest
    names safe for tool discovery and durable diagnostics; they grant no authority.
    Unlisted names and all cwd paths and input values remain withheld.
    ``diagnostic_profile_id`` is an optional public application label, not an
    execution-profile fingerprint. Never put private values in publication fields.
    """

    def __init__(
        self,
        *,
        allowed_executables: Iterable[str] = (),
        public_executable_names: Iterable[str] = (),
        public_environment_names: Iterable[str] = (),
        diagnostic_profile_id: str | None = None,
        approval_required_executables: Iterable[str] = (),
        allowed_cwds: Iterable[str] = (),
        allowed_env_names: Iterable[str] = (),
        allowed_env_values: Mapping[str, str] | None = None,
        max_env_value_bytes: int = DEFAULT_MAX_ENV_VALUE_BYTES,
        allow_stdin: bool = False,
        max_stdin_bytes: int = DEFAULT_MAX_STDIN_BYTES,
        max_timeout_s: int = MAX_TIMEOUT_SECONDS,
        shell_decision: CommandPolicyDecision = CommandPolicyDecision.DENY,
    ) -> None:
        self._allowed_executables = _validated_strings(
            allowed_executables,
            field_name="allowed_executables",
        )
        self._approval_required_executables = _validated_strings(
            approval_required_executables,
            field_name="approval_required_executables",
        )
        if self._allowed_executables & self._approval_required_executables:
            raise ValueError("An executable cannot be both allowed and approval-required.")
        self._allowed_cwds = _validated_canonical_roots(allowed_cwds)
        self._allowed_env_values = _validated_env_values(allowed_env_values)
        self._allowed_env_names = _validated_env_names(allowed_env_names) | frozenset(
            self._allowed_env_values
        )
        self._max_env_value_bytes = _nonnegative_int(
            max_env_value_bytes,
            field_name="max_env_value_bytes",
        )
        if any(
            len(value.encode("utf-8")) > self._max_env_value_bytes
            for value in self._allowed_env_values.values()
        ):
            raise ValueError("An allowed environment value exceeds max_env_value_bytes.")
        if type(allow_stdin) is not bool:
            raise TypeError("allow_stdin must be a boolean.")
        self._allow_stdin = allow_stdin
        self._max_stdin_bytes = _nonnegative_int(
            max_stdin_bytes,
            field_name="max_stdin_bytes",
        )
        self._max_timeout_s = _positive_int(max_timeout_s, field_name="max_timeout_s")
        if self._max_timeout_s > MAX_TIMEOUT_SECONDS:
            raise ValueError(f"max_timeout_s cannot exceed {MAX_TIMEOUT_SECONDS}.")
        if type(shell_decision) is not CommandPolicyDecision:
            raise TypeError("shell_decision must be a CommandPolicyDecision.")
        self._shell_decision = shell_decision
        self._public_executable_names = _public_names(public_executable_names)
        self._public_environment_names = _public_names(public_environment_names, environment=True)
        self._diagnostic_profile_id = diagnostic_profile_id
        # Validate the declared public profile at configuration time.
        _ = self.process_capabilities

    @property
    def process_capabilities(self) -> ProcessCommandCapabilities:
        allowed = tuple(sorted(self._allowed_executables & self._public_executable_names))[:64]
        approval = tuple(
            sorted(self._approval_required_executables & self._public_executable_names)
        )[:64]
        env = self.allowed_environment_names[:64]
        return ProcessCommandCapabilities(
            profile_id=self._diagnostic_profile_id,
            allowed_executables=allowed,
            approval_required_executables=approval,
            executable_names_withheld=(
                len(allowed) + len(approval)
                != len(self._allowed_executables | self._approval_required_executables)
            ),
            allowed_env_names=env,
            environment_names_withheld=len(env) != len(self._allowed_env_names),
            max_env_value_bytes=self._max_env_value_bytes,
            allow_stdin=self._allow_stdin,
            max_stdin_bytes=self._max_stdin_bytes,
            max_timeout_s=self._max_timeout_s,
            shell_decision=self._shell_decision.value,
        )

    def _deny(self, reason: str, code: Code, *, name: str | None = None) -> CommandPolicyResult:
        public_names = (
            self._public_executable_names
            if code in {Code.EXECUTABLE, Code.COMMAND_APPROVAL}
            else self._public_environment_names
        )
        published = name is not None and name in public_names
        named = code in {
            Code.EXECUTABLE,
            Code.COMMAND_APPROVAL,
            Code.ENVIRONMENT_NAME,
            Code.ENVIRONMENT_VALUE,
            Code.ENVIRONMENT_VALUE_SIZE,
        }
        diagnostic = ProcessCommandDiagnostic(
            code=code,
            rejected_name=name if published else None,
            value_state="published_by_policy"
            if published
            else (
                "withheld_not_declared_public"
                if named
                else (
                    "not_applicable"
                    if code in {Code.TIMEOUT, Code.COMMAND_KIND}
                    else "withheld_private_value"
                )
            ),
            capabilities=self.process_capabilities,
        )
        return CommandPolicyResult(
            decision=CommandPolicyDecision.DENY, reason=reason, process_diagnostic=diagnostic
        )

    @property
    def allowed_environment_names(self) -> tuple[str, ...]:
        """Discover allowed override names without exposing any configured values."""
        return tuple(sorted(self._allowed_env_names & self._public_environment_names))

    def _execution_profile_material(self) -> dict[str, object] | None:
        """Return public policy inputs without exposing exact environment values."""

        # Exact values commonly contain credentials. Hashing them into a public
        # component fingerprint would create a durable offline guessing oracle.
        if self._allowed_env_values:
            return None
        return {
            "public_executable_names": sorted(self._public_executable_names),
            "public_environment_names": sorted(self._public_environment_names),
            "diagnostic_profile_id": self._diagnostic_profile_id,
            "allowed_executables": sorted(self._allowed_executables),
            "approval_required_executables": sorted(self._approval_required_executables),
            "allowed_cwds": sorted(self._allowed_cwds),
            "allowed_env_names": sorted(self._allowed_env_names),
            "max_env_value_bytes": self._max_env_value_bytes,
            "allow_stdin": self._allow_stdin,
            "max_stdin_bytes": self._max_stdin_bytes,
            "max_timeout_s": self._max_timeout_s,
            "shell_decision": self._shell_decision.value,
        }

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ) -> CommandPolicyResult:
        del ctx
        decision, decision_reason = self._command_decision(request)
        if decision is CommandPolicyDecision.DENY:
            code = (
                Code.SHELL
                if request.command.kind == "shell"
                else (Code.EXECUTABLE if request.command.kind == "process" else Code.COMMAND_KIND)
            )
            name = (
                request.command.argv[0]
                if request.command.kind == "process" and request.command.argv
                else None
            )
            return self._deny(decision_reason, code, name=name)

        canonical_cwd = request.canonical_cwd
        if (
            canonical_cwd is None
            or "\0" in canonical_cwd
            or not posixpath.isabs(canonical_cwd)
            or posixpath.normpath(canonical_cwd) != canonical_cwd
            or not any(is_same_or_child(canonical_cwd, root) for root in self._allowed_cwds)
        ):
            return self._deny(
                "Canonical cwd is not allowed by the process policy.", Code.WORKING_DIRECTORY
            )

        environment_denial = self._environment_denial(request.env)
        if environment_denial is not None:
            reason, name, code = environment_denial
            result = self._deny(reason, code, name=name)
            # Keep legacy name-only fields, now gated by explicit publication consent.
            result.denied_env_name = (
                name if name in self._public_environment_names else "<withheld-name>"
            )
            result.allowed_env_names = self.process_capabilities.allowed_env_names
            result.allowed_env_names_truncated = (
                self.process_capabilities.environment_names_withheld
            )
            return result

        if request.stdin is not None:
            if not self._allow_stdin:
                return self._deny("Stdin is not allowed by the process policy.", Code.STDIN)
            if len(request.stdin.encode("utf-8")) > self._max_stdin_bytes:
                return self._deny("Stdin exceeds the process policy byte limit.", Code.STDIN_SIZE)

        if request.timeout_s > self._max_timeout_s:
            return self._deny("Timeout exceeds the process policy ceiling.", Code.TIMEOUT)

        if decision is CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL:
            name = (
                request.command.argv[0]
                if request.command.kind == "process" and request.command.argv
                else None
            )
            result = self._deny(
                decision_reason,
                Code.SHELL if request.command.kind == "shell" else Code.COMMAND_APPROVAL,
                name=name,
            )
            result.decision = decision
            return result
        return CommandPolicyResult(
            decision=decision,
            reason=decision_reason or None,
        )

    def _command_decision(
        self,
        request: CommandRequest,
    ) -> tuple[CommandPolicyDecision, str]:
        if request.command.kind == "shell":
            return (
                self._shell_decision,
                "Shell command requires an explicit process policy decision.",
            )
        if request.command.kind != "process":
            return (
                CommandPolicyDecision.DENY,
                "Command kind is not allowed by the process policy.",
            )
        argv = request.command.argv
        executable = None if not argv else argv[0]
        if executable in self._allowed_executables:
            return CommandPolicyDecision.ALLOW, ""
        if executable in self._approval_required_executables:
            return (
                CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL,
                "Executable requires command approval by the process policy.",
            )
        return (
            CommandPolicyDecision.DENY,
            "Executable is not allowed by the process policy.",
        )

    def _environment_denial(self, env: Mapping[str, str] | None) -> tuple[str, str, Code] | None:
        for name, value in (env or {}).items():
            if name not in self._allowed_env_names:
                return (
                    "Environment name is not allowed by the process policy.",
                    name,
                    Code.ENVIRONMENT_NAME,
                )
            if len(value.encode("utf-8")) > self._max_env_value_bytes:
                return (
                    "Environment value exceeds the process policy byte limit.",
                    name,
                    Code.ENVIRONMENT_VALUE_SIZE,
                )
            expected = self._allowed_env_values.get(name)
            if expected is not None and value != expected:
                return (
                    "Environment value does not satisfy the process policy.",
                    name,
                    Code.ENVIRONMENT_VALUE,
                )
        return None


def _validated_strings(values: Iterable[str], *, field_name: str) -> frozenset[str]:
    if isinstance(values, str | bytes):
        raise TypeError(f"{field_name} must be an iterable of strings, not a string.")
    validated: set[str] = set()
    for value in values:
        value = require_clean_nonblank(value, f"{field_name} item")
        if "\0" in value:
            raise ValueError(f"{field_name} entries cannot contain NUL bytes.")
        validated.add(value)
    return frozenset(validated)


def _validated_canonical_roots(values: Iterable[str]) -> frozenset[str]:
    roots = _validated_strings(values, field_name="allowed_cwds")
    for root in roots:
        if not posixpath.isabs(root):
            raise ValueError("allowed_cwds entries must be absolute POSIX paths.")
        if posixpath.normpath(root) != root:
            raise ValueError("allowed_cwds entries must be normalized POSIX paths.")
    return roots


def _validated_env_names(values: Iterable[str]) -> frozenset[str]:
    names = _validated_strings(values, field_name="allowed_env_names")
    for name in names:
        if _ENV_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("allowed_env_names entries must be valid environment names.")
    return names


def _validated_env_values(values: Mapping[str, str] | None) -> dict[str, str]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("allowed_env_values must be a mapping of strings to strings.")
    copied: dict[str, str] = {}
    for name, value in values.items():
        if type(name) is not str or _ENV_NAME_PATTERN.fullmatch(name) is None:
            raise ValueError("allowed_env_values keys must be valid environment names.")
        if type(value) is not str:
            raise TypeError("allowed_env_values values must be strings.")
        if "\0" in value:
            raise ValueError("allowed_env_values values cannot contain NUL bytes.")
        copied[name] = value
    return copied


def _nonnegative_int(value: int, *, field_name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return value


def _positive_int(value: int, *, field_name: str) -> int:
    value = _nonnegative_int(value, field_name=field_name)
    if value == 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    return value


def _public_names(values: Iterable[str], *, environment: bool = False) -> frozenset[str]:
    names = (
        _validated_env_names(values)
        if environment
        else _validated_strings(values, field_name="public_executable_names")
    )
    if len(names) > 64 or any(len(name) > 128 or not name.isprintable() for name in names):
        raise ValueError(
            "Public names must contain at most 64 printable names of at most 128 characters."
        )
    return names
