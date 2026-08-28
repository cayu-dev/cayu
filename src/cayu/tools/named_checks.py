from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import islice
from typing import TypeVar

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_json_value,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.artifacts import ArtifactMetadata, ArtifactScope
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.core.tools import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec
from cayu.runners import ExecCommand, RunnerExecutionError
from cayu.runners.base import runner_workspace_mutation_settlement
from cayu.tools._errors import structured_invalid_arguments, tool_argument_validation
from cayu.tools.commands import (
    DEFAULT_OUTPUT_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_LIMIT_BYTES,
    MAX_TIMEOUT_SECONDS,
    CommandPolicy,
    ExecCommandTool,
)

NAMED_CHECK_DECLARATION_BEHAVIOR_VERSION = "1"
RUN_CHECK_RESULT_PROJECTION_VERSION = "1"
DEFAULT_CHECK_MODEL_OUTPUT_BYTES = 16_000
MAX_CHECK_MODEL_OUTPUT_BYTES = 50_000
MAX_NAMED_CHECKS = 128
MAX_NAMED_CHECK_NAME_BYTES = 128
MAX_NAMED_CHECK_DESCRIPTION_BYTES = 4 * 1024
MAX_NAMED_CHECK_ARGV_ENTRIES = 64
MAX_NAMED_CHECK_ARGV_ENTRY_BYTES = 4 * 1024
MAX_NAMED_CHECK_ARGV_BYTES = 32 * 1024
MAX_NAMED_CHECK_EXECUTABLE_REQUIREMENTS = 32

_OUTPUT_TRUNCATION_MARKER = "\n[check output preview truncated]"
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True, init=False)
class NamedCheck:
    """Immutable application-owned declaration for one bounded process check."""

    name: str
    description: str
    timeout_s: int
    max_output_bytes: int
    execution_profile_identity: ExecutionProfileBehaviorIdentity
    required_executables: tuple[str, ...]
    _argv: tuple[str, ...]

    def __init__(
        self,
        *,
        name: str,
        description: str,
        command: ExecCommand,
        timeout_s: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
        execution_profile_identity: ExecutionProfileBehaviorIdentity,
        required_executables: Iterable[str] = (),
    ) -> None:
        owned_name = require_durable_clean_nonblank(name, "name")
        if len(owned_name.encode("utf-8")) > MAX_NAMED_CHECK_NAME_BYTES:
            raise ValueError(f"name must not exceed {MAX_NAMED_CHECK_NAME_BYTES} bytes.")
        owned_description = require_durable_clean_nonblank(description, "description")
        if len(owned_description.encode("utf-8")) > MAX_NAMED_CHECK_DESCRIPTION_BYTES:
            raise ValueError(
                f"description must not exceed {MAX_NAMED_CHECK_DESCRIPTION_BYTES} bytes."
            )
        if type(command) is not ExecCommand:
            raise TypeError("command must be an exact ExecCommand instance.")
        if command.kind != "process" or command.argv is None or command.shell is not None:
            raise ValueError("Named checks require an exact process-form command.")
        argv = tuple(command.argv)
        _validate_argv(argv)
        owned_timeout = _bounded_positive_int(
            timeout_s,
            field_name="timeout_s",
            maximum=MAX_TIMEOUT_SECONDS,
        )
        owned_output_limit = _bounded_positive_int(
            max_output_bytes,
            field_name="max_output_bytes",
            maximum=MAX_OUTPUT_LIMIT_BYTES,
        )
        owned_identity = copy_execution_profile_behavior_identity(execution_profile_identity)
        if owned_identity is None:
            raise TypeError(
                "execution_profile_identity must be an ExecutionProfileBehaviorIdentity."
            )
        executables = _validated_executable_requirements(required_executables)
        executables.add(argv[0])
        if len(executables) > MAX_NAMED_CHECK_EXECUTABLE_REQUIREMENTS:
            raise ValueError(
                "required_executables must contain at most "
                f"{MAX_NAMED_CHECK_EXECUTABLE_REQUIREMENTS} entries."
            )
        object.__setattr__(self, "name", owned_name)
        object.__setattr__(self, "description", owned_description)
        object.__setattr__(self, "timeout_s", owned_timeout)
        object.__setattr__(self, "max_output_bytes", owned_output_limit)
        object.__setattr__(self, "execution_profile_identity", owned_identity)
        object.__setattr__(self, "required_executables", tuple(sorted(executables)))
        object.__setattr__(self, "_argv", argv)

    @property
    def command(self) -> ExecCommand:
        """Return a detached command so callers cannot mutate the declaration."""

        return ExecCommand.process(*self._argv)

    @property
    def profile_fingerprint(self) -> str:
        """Return the stable secret-free fingerprint for this declaration."""

        return (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(self._profile_material(), "named_check")
            ).hexdigest()
        )

    def _profile_material(self) -> dict[str, object]:
        command_sha256 = sha256(
            canonical_durable_json_bytes(list(self._argv), "named_check.command")
        ).hexdigest()
        return {
            "declaration_behavior_version": NAMED_CHECK_DECLARATION_BEHAVIOR_VERSION,
            "name": self.name,
            "description": self.description,
            "command_kind": "process",
            "command_sha256": f"sha256:{command_sha256}",
            "timeout_s": self.timeout_s,
            "max_output_bytes": self.max_output_bytes,
            "required_executables": list(self.required_executables),
            "execution_profile_identity": self.execution_profile_identity.model_dump(mode="json"),
        }


class RunCheckTool(Tool):
    """Run one application-owned check selected by its finite public name."""

    spec = ToolSpec(
        name="run_check",
        description="Run one application-defined bounded check by name.",
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        workspace_mutation=True,
        input_schema={
            "type": "object",
            "properties": {"check": {"type": "string", "enum": []}},
            "required": ["check"],
            "additionalProperties": False,
        },
    )

    def __init__(
        self,
        *,
        checks: Iterable[NamedCheck],
        command_policy: CommandPolicy,
        max_model_output_bytes: int = DEFAULT_CHECK_MODEL_OUTPUT_BYTES,
    ) -> None:
        if isinstance(checks, str | bytes):
            raise TypeError("checks must be an iterable of NamedCheck declarations.")
        owned_checks = _bounded_iterable_snapshot(
            checks,
            field_name="checks",
            maximum=MAX_NAMED_CHECKS,
        )
        if not owned_checks:
            raise ValueError("checks must contain at least one NamedCheck declaration.")
        if any(type(check) is not NamedCheck for check in owned_checks):
            raise TypeError("checks entries must be exact NamedCheck declarations.")
        names = [check.name for check in owned_checks]
        if len(names) != len(set(names)):
            raise ValueError("checks must not contain duplicate names.")
        if not isinstance(command_policy, CommandPolicy):
            raise TypeError("command_policy must implement CommandPolicy.")
        owned_model_limit = _bounded_positive_int(
            max_model_output_bytes,
            field_name="max_model_output_bytes",
            maximum=MAX_CHECK_MODEL_OUTPUT_BYTES,
        )
        if owned_model_limit < len(_OUTPUT_TRUNCATION_MARKER.encode("utf-8")):
            raise ValueError("max_model_output_bytes is too small for the truncation marker.")
        ordered_checks = tuple(sorted(owned_checks, key=lambda item: item.name))
        schema = type(self).spec.input_schema
        schema["properties"]["check"]["enum"] = [check.name for check in ordered_checks]
        super().__init__(type(self).spec.model_copy(update={"input_schema": schema}))
        self._checks = ordered_checks
        self._checks_by_name = {check.name: check for check in ordered_checks}
        self._command_policy = command_policy
        self._executor = ExecCommandTool(policy=command_policy)
        self._max_model_output_bytes = owned_model_limit

    @property
    def checks(self) -> tuple[NamedCheck, ...]:
        """Return immutable named declarations in canonical name order."""

        return self._checks

    @property
    def command_policy(self) -> CommandPolicy:
        """Return the mandatory defense-in-depth command policy."""

        return self._command_policy

    def _execution_profile_material(self) -> dict[str, object]:
        """Return complete built-in behavior inputs; policy is profiled separately."""

        return {
            "result_projection_version": RUN_CHECK_RESULT_PROJECTION_VERSION,
            "max_model_output_bytes": self._max_model_output_bytes,
            "checks": [check._profile_material() for check in self._checks],
        }

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            if type(args) is not dict:
                raise ValueError("Tool arguments must be an object.")
            if set(args) != {"check"}:
                raise ValueError("Tool arguments must contain only the required `check` field.")
            check_name = args.get("check")
            if type(check_name) is not str:
                raise ValueError("Tool argument `check` must be a string.")
            check = self._checks_by_name.get(check_name)
            if check is None:
                raise ValueError("Tool argument `check` must name a declared check.")

        try:
            raw_result = await self._executor._execute_resolved_command(
                ctx,
                command=check.command,
                cwd=None,
                canonical_cwd=None,
                env=None,
                timeout_s=check.timeout_s,
                stdin=None,
                max_output_bytes=check.max_output_bytes,
                policy_source=self,
                include_runner_evidence=True,
            )
        except RunnerExecutionError as exc:
            return _execution_failure_result(check, exc)
        except TypeError as exc:
            if str(exc) != "Runner returned invalid result type.":
                raise
            return _malformed_execution_result(check)
        try:
            return await self._project_result(ctx, check=check, raw_result=raw_result)
        except TypeError as exc:
            if str(exc) != "Runner returned invalid result type.":
                raise
            return _malformed_execution_result(check)

    async def _project_result(
        self,
        ctx: ToolContext,
        *,
        check: NamedCheck,
        raw_result: ToolResult,
    ) -> ToolResult:
        structured = raw_result.structured
        error = None if structured is None else structured.get("error")
        if error == "command_denied":
            return _policy_result(check, raw_result, status="policy_denied")
        if error == "command_approval_required":
            return _policy_result(check, raw_result, status="approval_required")
        if error == "runner_unavailable" or structured is None:
            return _runner_unavailable_result(check, raw_result)

        stdout = _required_result_text(structured, "stdout")
        stderr = _required_result_text(structured, "stderr")
        stdout_preview, stdout_projection_truncated = _truncate_utf8(
            stdout,
            maximum=self._max_model_output_bytes,
        )
        stderr_preview, stderr_projection_truncated = _truncate_utf8(
            stderr,
            maximum=self._max_model_output_bytes,
        )
        stdout_runner_truncated = _required_result_bool(structured, "stdout_truncated")
        stderr_runner_truncated = _required_result_bool(structured, "stderr_truncated")
        timed_out = _required_result_bool(structured, "timed_out")
        cancelled = _required_result_bool(structured, "cancelled")
        exit_code = _required_result_int(structured, "exit_code")
        runner_artifacts = _result_artifacts(structured)
        output_record = canonical_durable_json_bytes(
            {
                "schema_version": RUN_CHECK_RESULT_PROJECTION_VERSION,
                "check": check.name,
                "check_profile_fingerprint": check.profile_fingerprint,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_runner_truncated": stdout_runner_truncated,
                "stderr_runner_truncated": stderr_runner_truncated,
            },
            "check_output",
        )
        output_sha256 = "sha256:" + sha256(output_record).hexdigest()
        output_artifact: dict[str, object] | None = None
        artifact_status = "not_needed"
        if stdout_projection_truncated or stderr_projection_truncated:
            artifact_status, output_artifact = await _store_output_artifact(
                ctx,
                check=check,
                content=output_record,
                content_sha256=output_sha256,
            )
        artifacts = list(runner_artifacts)
        if output_artifact is not None:
            artifacts.append(output_artifact)
        status = (
            "timed_out"
            if timed_out
            else "cancelled"
            if cancelled
            else "passed"
            if exit_code == 0
            else "failed"
        )
        mutation_settlement = _required_mutation_settlement(structured)
        projected = {
            "check": check.name,
            "check_profile_fingerprint": check.profile_fingerprint,
            "status": status,
            "exit_code": exit_code,
            "stdout": stdout_preview,
            "stderr": stderr_preview,
            "stdout_truncated": stdout_runner_truncated or stdout_projection_truncated,
            "stderr_truncated": stderr_runner_truncated or stderr_projection_truncated,
            "stdout_runner_truncated": stdout_runner_truncated,
            "stderr_runner_truncated": stderr_runner_truncated,
            "stdout_projection_truncated": stdout_projection_truncated,
            "stderr_projection_truncated": stderr_projection_truncated,
            "stdout_bytes": _optional_result_nonnegative_int(structured, "stdout_bytes"),
            "stderr_bytes": _optional_result_nonnegative_int(structured, "stderr_bytes"),
            "timed_out": timed_out,
            "cancelled": cancelled,
            "timeout_s": check.timeout_s,
            "max_output_bytes": check.max_output_bytes,
            "required_executables": list(check.required_executables),
            "output_sha256": output_sha256,
            "output_artifact_status": artifact_status,
            "workspace_mutation_settlement": mutation_settlement,
            "cleanup_uncertain": mutation_settlement in {"deferred", "uncertain"},
            "artifacts": artifacts,
        }
        return ToolResult(
            content=_model_content(
                check=check,
                status=status,
                exit_code=exit_code,
                stdout=stdout_preview,
                stderr=stderr_preview,
                stdout_truncated=stdout_runner_truncated or stdout_projection_truncated,
                stderr_truncated=stderr_runner_truncated or stderr_projection_truncated,
            ),
            structured=projected,
            artifacts=artifacts,
            is_error=timed_out or cancelled,
        )


def _validate_argv(argv: tuple[str, ...]) -> None:
    if not argv:
        raise ValueError("Named check process commands require non-empty argv.")
    if len(argv) > MAX_NAMED_CHECK_ARGV_ENTRIES:
        raise ValueError(
            f"command argv must contain at most {MAX_NAMED_CHECK_ARGV_ENTRIES} entries."
        )
    total_bytes = 0
    for entry in argv:
        if type(entry) is not str:
            raise TypeError("command argv entries must be strings.")
        require_durable_text(entry, "command argv entry")
        size = len(entry.encode("utf-8"))
        if size > MAX_NAMED_CHECK_ARGV_ENTRY_BYTES:
            raise ValueError(
                f"command argv entries must not exceed {MAX_NAMED_CHECK_ARGV_ENTRY_BYTES} bytes."
            )
        total_bytes += size
    if total_bytes > MAX_NAMED_CHECK_ARGV_BYTES:
        raise ValueError(f"command argv must not exceed {MAX_NAMED_CHECK_ARGV_BYTES} bytes.")


def _validated_executable_requirements(values: Iterable[str]) -> set[str]:
    if isinstance(values, str | bytes):
        raise TypeError("required_executables must be an iterable of strings.")
    snapshot = _bounded_iterable_snapshot(
        values,
        field_name="required_executables",
        maximum=MAX_NAMED_CHECK_EXECUTABLE_REQUIREMENTS,
    )
    executables: set[str] = set()
    for value in snapshot:
        executable = require_durable_clean_nonblank(value, "required_executables item")
        if len(executable.encode("utf-8")) > MAX_NAMED_CHECK_ARGV_ENTRY_BYTES:
            raise ValueError(
                "required_executables entries must not exceed "
                f"{MAX_NAMED_CHECK_ARGV_ENTRY_BYTES} bytes."
            )
        executables.add(executable)
    return executables


def _bounded_iterable_snapshot(
    values: Iterable[_T],
    *,
    field_name: str,
    maximum: int,
) -> tuple[_T, ...]:
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable.") from exc
    snapshot = tuple(islice(iterator, maximum + 1))
    if len(snapshot) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} entries.")
    return snapshot


def _bounded_positive_int(value: int, *, field_name: str, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than zero.")
    if value > maximum:
        raise ValueError(f"{field_name} must be at most {maximum}.")
    return value


def _required_result_text(structured: Mapping[str, object], field_name: str) -> str:
    value = structured.get(field_name)
    if type(value) is not str:
        raise TypeError("Runner returned invalid result type.")
    return value


def _required_result_bool(structured: Mapping[str, object], field_name: str) -> bool:
    value = structured.get(field_name)
    if type(value) is not bool:
        raise TypeError("Runner returned invalid result type.")
    return value


def _required_result_int(structured: Mapping[str, object], field_name: str) -> int:
    value = structured.get(field_name)
    if type(value) is not int:
        raise TypeError("Runner returned invalid result type.")
    return value


def _optional_result_nonnegative_int(
    structured: Mapping[str, object],
    field_name: str,
) -> int | None:
    value = structured.get(field_name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise TypeError("Runner returned invalid result type.")
    return value


def _required_mutation_settlement(structured: Mapping[str, object]) -> str:
    value = structured.get("workspace_mutation_settlement")
    if type(value) is not str or value not in {
        "complete",
        "runner_quiescent",
        "deferred",
        "uncertain",
    }:
        raise TypeError("Runner returned invalid result type.")
    return value


def _result_artifacts(structured: Mapping[str, object]) -> list[dict]:
    value = structured.get("artifacts", [])
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TypeError("Runner returned invalid result type.")
    return copy_json_value(list(value), "artifacts")


def _truncate_utf8(value: str, *, maximum: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value, False
    marker = _OUTPUT_TRUNCATION_MARKER.encode("utf-8")
    prefix = encoded[: maximum - len(marker)]
    return prefix.decode("utf-8", errors="ignore").rstrip() + _OUTPUT_TRUNCATION_MARKER, True


async def _store_output_artifact(
    ctx: ToolContext,
    *,
    check: NamedCheck,
    content: bytes,
    content_sha256: str,
) -> tuple[str, dict[str, object] | None]:
    artifact_store = ctx.artifact_store
    if artifact_store is None:
        return "unavailable", None
    artifact_id = None
    if ctx.idempotency_key is not None:
        identity = sha256(
            b"cayu-check-output-v1\0"
            + ctx.session_id.encode("utf-8")
            + b"\0"
            + ctx.idempotency_key.encode("utf-8")
            + b"\0"
            + check.name.encode("utf-8")
        ).hexdigest()[:32]
        artifact_id = f"art_{identity}"
    try:
        artifact = await artifact_store.put_bytes(
            content,
            artifact_id=artifact_id,
            filename=f"check-{check.name}-output.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            environment_name=ctx.environment_name,
            metadata={
                "operation": "run_check",
                "check": check.name,
                "check_profile_fingerprint": check.profile_fingerprint,
                "content_sha256": content_sha256,
                "result_projection_version": RUN_CHECK_RESULT_PROJECTION_VERSION,
            },
        )
    except Exception:
        return "failed", None
    if type(artifact) is not ArtifactMetadata:
        return "failed", None
    return "stored", artifact.model_dump(mode="json")


def _model_content(
    *,
    check: NamedCheck,
    status: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    stdout_truncated: bool,
    stderr_truncated: bool,
) -> str:
    first = (
        f"Check {check.name!r} {status}."
        if status in {"passed", "failed"}
        else f"Check {check.name!r} {status.replace('_', ' ')}."
    )
    if status == "failed":
        first = f"{first[:-1]} with exit code {exit_code}."
    sections = [first]
    if stdout.strip():
        sections.append(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        sections.append(f"stderr:\n{stderr.strip()}")
    if stdout_truncated or stderr_truncated:
        sections.append("Check output was truncated; inspect the structured artifact evidence.")
    return "\n\n".join(sections)


def _base_error_evidence(check: NamedCheck, *, status: str) -> dict[str, object]:
    return {
        "check": check.name,
        "check_profile_fingerprint": check.profile_fingerprint,
        "status": status,
        "timeout_s": check.timeout_s,
        "max_output_bytes": check.max_output_bytes,
        "required_executables": list(check.required_executables),
    }


def _policy_result(check: NamedCheck, raw_result: ToolResult, *, status: str) -> ToolResult:
    structured = {
        **_base_error_evidence(check, status=status),
        **({} if raw_result.structured is None else dict(raw_result.structured)),
    }
    return ToolResult(
        content=f"Check {check.name!r} was refused. {raw_result.content}",
        structured=structured,
        artifacts=raw_result.artifacts,
        is_error=True,
    )


def _runner_unavailable_result(check: NamedCheck, raw_result: ToolResult) -> ToolResult:
    structured = {
        **_base_error_evidence(check, status="runner_unavailable"),
        **({} if raw_result.structured is None else dict(raw_result.structured)),
    }
    return ToolResult(
        content=f"Check {check.name!r} could not run. {raw_result.content}",
        structured=structured,
        artifacts=raw_result.artifacts,
        is_error=True,
    )


def _execution_failure_result(check: NamedCheck, exc: RunnerExecutionError) -> ToolResult:
    mutation_settlement = runner_workspace_mutation_settlement(result=None, error=exc)
    return ToolResult(
        content=f"Check {check.name!r} failed at the runner execution boundary.",
        structured={
            **_base_error_evidence(check, status="execution_failed"),
            "error": "runner_execution_failed",
            "diagnostic": copy_json_value(exc.diagnostic, "diagnostic"),
            "workspace_mutation_settlement": mutation_settlement,
            "cleanup_uncertain": mutation_settlement in {"deferred", "uncertain"},
        },
        artifacts=exc.artifacts,
        is_error=True,
    )


def _malformed_execution_result(check: NamedCheck) -> ToolResult:
    return ToolResult(
        content=f"Check {check.name!r} returned malformed runner evidence.",
        structured={
            **_base_error_evidence(check, status="malformed_execution"),
            "error": "malformed_runner_result",
        },
        is_error=True,
    )
