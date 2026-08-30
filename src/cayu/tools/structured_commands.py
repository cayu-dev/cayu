"""Closed selector-plus-argv command execution for admitted Docker toolchains."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from time import perf_counter_ns
from typing import Any, Protocol, runtime_checkable

from cayu._validation import canonical_durable_json_bytes, copy_json_value
from cayu.artifacts import ArtifactMetadata, ArtifactScope
from cayu.core.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.core.tools import (
    DurableToolRecoveryAuthority,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    WorkspaceHandle,
    _runtime_tool_invocation_authority,
)
from cayu.environments.admission import ExecutionAdmissionCandidate
from cayu.environments.docker_toolchains import (
    DockerCodingCommandAuthority,
    DockerCodingToolchainError,
    DockerCodingToolchainProfile,
    docker_coding_toolchain_runner_admission_failure,
    verify_docker_coding_toolchain_dependencies,
)
from cayu.runners import ExecCommand, ExecResult, RunnerExecutionError
from cayu.runners.base import runner_workspace_mutation_settlement
from cayu.runtime.tool_policy import (
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.tools._errors import structured_invalid_arguments, tool_argument_validation
from cayu.tools.commands import (
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    ExecCommandTool,
)
from cayu.workspaces import WorkspaceGitEntry, WorkspaceGitEntryListResult, WorkspaceReadResult

RUN_COMMAND_RESULT_SCHEMA = "cayu.run_command_result.v1"
STRUCTURED_COMMAND_TOOL_POLICY_SCHEMA = "cayu.structured_command_tool_policy.v1"
_OUTPUT_TRUNCATION_MARKER = "\n[command output preview truncated]"
_COMMAND_MANIFEST_MAX_PATHS = 10_000
_COMMAND_MANIFEST_MAX_FILE_BYTES = 8 * 1024 * 1024
_COMMAND_MANIFEST_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_COMMAND_JOURNAL_RECORD_TYPE = "cayu.run_command.journal"
_COMMAND_JOURNAL_SCHEMA_VERSION = 1
_DURABLE_RUNNER_OPERATION_SCHEMA = "cayu.durable_runner_operation.v1"


class _WorkspaceManifestError(RuntimeError):
    """A complete bounded command mutation observation was unavailable."""


class _CommandJournalError(RuntimeError):
    """A durable structured-command dispatch boundary could not be advanced safely."""


@dataclass(frozen=True)
class _WorkspaceCommandManifest:
    entries: tuple[tuple[str, str, int, str], ...]
    total_bytes: int

    @property
    def fingerprint(self) -> str:
        return (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    {
                        "entries": [list(entry) for entry in self.entries],
                        "total_bytes": self.total_bytes,
                    },
                    "structured_command_workspace_manifest",
                )
            ).hexdigest()
        )

    @property
    def paths(self) -> dict[str, tuple[str, int, str]]:
        return {
            path: (content_sha256, size, git_mode)
            for path, content_sha256, size, git_mode in self.entries
        }


class _StructuredCommandJournal:
    """Runtime-owned durable boundary for one structured command dispatch."""

    def __init__(self, authority: Any, storage_key: str, record: dict[str, Any]) -> None:
        self.authority = authority
        self.storage_key = storage_key
        self.record = record

    async def mark_dispatching(self) -> None:
        if self.record.get("state") != "prepared":
            raise _CommandJournalError("Structured-command journal is not prepared.")
        desired = _copy_command_journal_record(self.record)
        desired["state"] = "dispatching"
        await self._advance(desired)

    async def terminal(self, result: ToolResult) -> None:
        if self.record.get("state") != "dispatching":
            raise _CommandJournalError("Structured-command journal is not dispatching.")
        desired = _copy_command_journal_record(self.record)
        desired["state"] = "terminal"
        desired["terminal_result"] = result.model_dump(mode="json")
        try:
            desired = self.authority.seal_durable_output(desired)
        except Exception as exc:
            raise _CommandJournalError("Structured-command terminal sealing failed.") from exc
        await self._advance(desired)

    async def checkpoint_runner_terminal(
        self,
        result: ToolResult,
        *,
        timing: Mapping[str, object],
    ) -> None:
        """Persist redacted process/output/cleanup evidence before projection."""

        if self.record.get("state") != "dispatching":
            raise _CommandJournalError("Structured-command journal is not dispatching.")
        desired = _copy_command_journal_record(self.record)
        desired["runner_terminal_result"] = result.model_dump(mode="json")
        desired["runner_terminal_timing"] = copy_json_value(dict(timing), "runner_timing")
        desired["runner_terminal_identity"] = _sha256_identity(
            result.model_dump(mode="json"),
            purpose="structured_command_runner_terminal",
        )
        await self._advance(desired)

    async def _advance(self, desired: dict[str, Any]) -> None:
        expected = self.record
        try:
            persisted = await self.authority.compare_and_set_durable_operation(
                self.storage_key,
                expected,
                desired,
                {},
            )
        except Exception as exc:
            persisted = await self.authority.load_durable_operation(self.storage_key)
            if persisted != desired:
                raise _CommandJournalError(
                    "Structured-command journal publication failed."
                ) from exc
        if persisted != desired:
            raise _CommandJournalError(
                "Structured-command journal publication returned conflicting evidence."
            )
        self.record = desired


@runtime_checkable
class _ReceiptAdmissionRunner(Protocol):
    def execution_admission_candidate(self) -> ExecutionAdmissionCandidate | None: ...


class _AdmittedStructuredCommandPolicy(CommandPolicy):
    """Defense in depth for commands already resolved by :class:`RunCommandTool`."""

    def __init__(
        self,
        profile: DockerCodingToolchainProfile,
        authority: DockerCodingCommandAuthority,
    ) -> None:
        self._profile = profile
        self._authority = authority
        self._identity = ExecutionProfileBehaviorIdentity(
            name="cayu.structured_command_policy",
            behavior_version="1",
            implementation_version=profile.fingerprint,
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ) -> CommandPolicyResult:
        del ctx
        command = request.command
        if command.kind != "process" or command.argv is None or command.shell is not None:
            return _denied("Structured command authority permits only direct process argv.")
        argv = tuple(command.argv)
        authority = self._authority
        prefix = (authority.executable, *authority.fixed_arguments)
        suffix = authority.suffix_arguments
        if len(argv) < len(prefix) + len(suffix) or argv[: len(prefix)] != prefix:
            return _denied("Resolved command does not match its selected authority.")
        if suffix and argv[-len(suffix) :] != suffix:
            return _denied("Resolved command does not match its selected authority.")
        end = len(argv) - len(suffix) if suffix else len(argv)
        arguments = argv[len(prefix) : end]
        try:
            authority.validate_model_arguments(arguments)
            working_directory = _relative_working_directory(
                request.canonical_cwd,
                workspace_path=self._profile.workspace_path,
            )
            authority.validate_working_directory(working_directory)
        except ValueError:
            return _denied("Resolved command does not match its selected authority.")
        if (
            request.cwd != (None if working_directory == "." else working_directory)
            or request.env != authority.environment
            or request.stdin is not None
            or request.timeout_s > authority.timeout_seconds
        ):
            return _denied("Resolved command does not match its selected authority.")
        return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)


class _AdmittedStructuredCommandCataloguePolicy(CommandPolicy):
    """Identity-bearing exact policy for every selector in one profile."""

    def __init__(self, profile: DockerCodingToolchainProfile) -> None:
        self._policies = tuple(
            _AdmittedStructuredCommandPolicy(profile, authority)
            for authority in profile.structured_command_authorities
        )
        self._identity = ExecutionProfileBehaviorIdentity(
            name="cayu.structured_command_catalogue_policy",
            behavior_version="1",
            implementation_version=profile.fingerprint,
        )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return self._identity

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ) -> CommandPolicyResult:
        for policy in self._policies:
            verdict = await policy.evaluate(ctx, request)
            if verdict.decision is CommandPolicyDecision.ALLOW:
                return verdict
        return _denied("Resolved command does not match the admitted selector catalogue.")


class StructuredCommandToolPolicy(ToolPolicy):
    """Resolve structured commands before runtime policy and durable approval."""

    def __init__(
        self,
        *,
        toolchain_profile: DockerCodingToolchainProfile,
        base_policy: ToolPolicy | None = None,
    ) -> None:
        if type(toolchain_profile) is not DockerCodingToolchainProfile:
            raise TypeError("toolchain_profile must be an exact DockerCodingToolchainProfile.")
        if base_policy is not None and not isinstance(base_policy, ToolPolicy):
            raise TypeError("base_policy must implement ToolPolicy or be None.")
        self._profile = DockerCodingToolchainProfile.model_validate(
            toolchain_profile.model_dump(mode="python", by_alias=True)
        )
        self._authorities = {
            item.selector: item for item in self._profile.structured_command_authorities
        }
        if not self._authorities:
            raise ValueError("toolchain_profile must expose at least one structured command.")
        self._base_policy = base_policy
        base_identity = None if base_policy is None else base_policy.execution_profile_identity
        if base_policy is not None and base_identity is None:
            self._identity = None
        else:
            material = {
                "schema": STRUCTURED_COMMAND_TOOL_POLICY_SCHEMA,
                "toolchain_profile_fingerprint": self._profile.fingerprint,
                "base_policy_identity": (
                    None if base_identity is None else base_identity.model_dump(mode="json")
                ),
            }
            self._identity = ExecutionProfileBehaviorIdentity(
                name="cayu.structured_command_tool_policy",
                behavior_version="1",
                implementation_version="sha256:"
                + sha256(
                    canonical_durable_json_bytes(material, "structured_command_tool_policy")
                ).hexdigest(),
            )

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        return self._identity

    def _execution_profile_material(self) -> dict[str, object] | None:
        if self._identity is None:
            return None
        return {
            "schema": STRUCTURED_COMMAND_TOOL_POLICY_SCHEMA,
            "profile_fingerprint": self._profile.fingerprint,
            "identity": self._identity.model_dump(mode="json"),
        }

    async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
        if self._base_policy is not None:
            base_result = await self._base_policy.authorize(request)
            if type(base_result) is not ToolPolicyResult:
                raise TypeError("base_policy must return an exact ToolPolicyResult.")
            if base_result.decision is not ToolPolicyDecision.ALLOW:
                return base_result
        if request.tool_name != "run_command":
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)
        try:
            authority, arguments, working_directory, timeout_seconds, output_mode = (
                _resolve_structured_command_request(self._authorities, request.arguments)
            )
        except ValueError:
            return ToolPolicyResult(
                decision=ToolPolicyDecision.DENY,
                reason="Structured command request does not match its admitted authority.",
                metadata={
                    "schema": STRUCTURED_COMMAND_TOOL_POLICY_SCHEMA,
                    "error": "structured_command_authority_mismatch",
                    "toolchain_profile_fingerprint": self._profile.fingerprint,
                },
            )
        argv = authority.command_argv(arguments)
        metadata = {
            "schema": STRUCTURED_COMMAND_TOOL_POLICY_SCHEMA,
            "selector": authority.selector,
            "selector_revision": authority.revision,
            "selector_fingerprint": authority.fingerprint,
            "argv_sha256": _argv_digest(argv),
            "argument_count": len(arguments),
            "working_directory_sha256": "sha256:"
            + sha256(working_directory.encode("utf-8")).hexdigest(),
            "effect": authority.effect,
            "parallel_safe": authority.parallel_safe,
            "idempotent": authority.idempotent,
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": authority.max_output_bytes,
            "output_mode": output_mode,
            "environment_identity": _environment_identity(authority),
            "mutation_scope_identity": _mutation_scope_identity(authority),
            **self._profile.evidence(),
        }
        if authority.approval == "required":
            return ToolPolicyResult(
                decision=ToolPolicyDecision.REQUIRE_APPROVAL,
                reason="Structured command selector requires durable application approval.",
                metadata=metadata,
                approval_expires_in_seconds=authority.approval_expires_in_seconds,
            )
        return ToolPolicyResult(
            decision=ToolPolicyDecision.ALLOW,
            metadata=metadata,
        )


def _resolve_structured_command_request(
    authorities: Mapping[str, DockerCodingCommandAuthority],
    args: dict,
) -> tuple[DockerCodingCommandAuthority, tuple[str, ...], str, int, str]:
    if type(args) is not dict:
        raise ValueError("Tool arguments must be an object.")
    allowed = {"selector", "args", "workingDirectory", "timeoutSeconds", "outputMode"}
    if set(args) - allowed or "selector" not in args:
        raise ValueError("Tool arguments contain unknown fields or omit selector.")
    selector = args.get("selector")
    if type(selector) is not str:
        raise ValueError("Tool argument `selector` must be a string.")
    authority = authorities.get(selector)
    if authority is None:
        raise ValueError("Tool argument `selector` is not admitted by the active profile.")
    raw_arguments = args.get("args", [])
    if type(raw_arguments) is not list or any(type(item) is not str for item in raw_arguments):
        raise ValueError("Tool argument `args` must be an array of strings.")
    arguments = authority.validate_model_arguments(tuple(raw_arguments))
    raw_working_directory = args.get("workingDirectory")
    if raw_working_directory is not None and type(raw_working_directory) is not str:
        raise ValueError("Tool argument `workingDirectory` must be a string.")
    working_directory = authority.validate_working_directory(raw_working_directory)
    timeout_seconds = args.get("timeoutSeconds", authority.timeout_seconds)
    if type(timeout_seconds) is not int or isinstance(timeout_seconds, bool):
        raise ValueError("Tool argument `timeoutSeconds` must be an integer.")
    if not 1 <= timeout_seconds <= authority.timeout_seconds:
        raise ValueError("Tool argument `timeoutSeconds` must narrow the selector timeout ceiling.")
    output_mode = args.get("outputMode", "summary")
    if type(output_mode) is not str or output_mode not in {
        "summary",
        "summary_and_artifact",
    }:
        raise ValueError("Tool argument `outputMode` is invalid.")
    return authority, arguments, working_directory, timeout_seconds, output_mode


class RunCommandTool(Tool):
    """Run one bounded structured command selected from an admitted toolchain profile."""

    spec = ToolSpec(
        name="run_command",
        description=(
            "Run a bounded application-admitted command selector with structured arguments. "
            "This is not a shell."
        ),
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        workspace_mutation=True,
        input_schema={
            "type": "object",
            "properties": {
                "selector": {"type": "string", "enum": []},
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 64,
                    "default": [],
                },
                "workingDirectory": {"type": "string"},
                "timeoutSeconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                },
                "outputMode": {
                    "type": "string",
                    "enum": ["summary", "summary_and_artifact"],
                    "default": "summary",
                },
            },
            "required": ["selector"],
            "additionalProperties": False,
        },
    )

    def __init__(self, *, toolchain_profile: DockerCodingToolchainProfile) -> None:
        if type(toolchain_profile) is not DockerCodingToolchainProfile:
            raise TypeError("toolchain_profile must be an exact DockerCodingToolchainProfile.")
        owned_profile = DockerCodingToolchainProfile.model_validate(
            toolchain_profile.model_dump(mode="python", by_alias=True)
        )
        authorities = owned_profile.structured_command_authorities
        if not authorities:
            raise ValueError("toolchain_profile must expose at least one structured command.")
        schema = copy_json_value(type(self).spec.input_schema, "run_command.input_schema")
        schema["properties"]["selector"]["enum"] = [item.selector for item in authorities]
        schema["properties"]["args"]["maxItems"] = max(item.max_arguments for item in authorities)
        super().__init__(type(self).spec.model_copy(update={"input_schema": schema}))
        self._profile = owned_profile
        self._authorities = {item.selector: item for item in authorities}
        self._command_policies = {
            item.selector: _AdmittedStructuredCommandPolicy(owned_profile, item)
            for item in authorities
        }
        self._executors = {
            selector: ExecCommandTool(policy=policy)
            for selector, policy in self._command_policies.items()
        }
        self._command_policy = _AdmittedStructuredCommandCataloguePolicy(owned_profile)

    @property
    def toolchain_profile(self) -> DockerCodingToolchainProfile:
        return DockerCodingToolchainProfile.model_validate(
            self._profile.model_dump(mode="python", by_alias=True)
        )

    @property
    def selectors(self) -> tuple[DockerCodingCommandAuthority, ...]:
        return self._profile.structured_command_authorities

    @property
    def command_policy(self) -> CommandPolicy:
        """Return the exact identity-bearing policy for the complete selector catalogue."""

        return self._command_policy

    @property
    def _publish_arguments(self) -> bool:
        return False

    def _execution_profile_material(self) -> dict[str, object]:
        return {
            "result_schema": RUN_COMMAND_RESULT_SCHEMA,
            "toolchain_profile_fingerprint": self._profile.fingerprint,
            "selector_fingerprints": [
                authority.fingerprint for authority in self._profile.structured_command_authorities
            ],
        }

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        with tool_argument_validation():
            authority, arguments, working_directory, timeout_seconds, output_mode = (
                self._resolve_arguments(args)
            )
        runner_failure = docker_coding_toolchain_runner_admission_failure(
            ctx.runner,
            profile=self._profile,
        )
        if runner_failure is not None:
            return self._error_result(
                authority,
                status="unavailable",
                error=runner_failure,
                content="Structured commands require the exact admitted Docker toolchain.",
            )
        if ctx.workspace is None:
            return self._error_result(
                authority,
                status="unavailable",
                error="workspace_unavailable",
                content="Structured command requires an active admitted workspace.",
            )
        if authority.dependency_sensitive:
            try:
                await verify_docker_coding_toolchain_dependencies(
                    self._profile,
                    ctx.workspace,
                )
            except DockerCodingToolchainError as exc:
                return self._error_result(
                    authority,
                    status=(
                        "stale_toolchain"
                        if exc.code == "dependency_inputs_changed"
                        else "unavailable"
                    ),
                    error=exc.code,
                    content=str(exc),
                    extra={
                        "dependency_path_count": exc.path_count,
                        "dependency_paths_fingerprint": exc.paths_fingerprint,
                    },
                )
        try:
            before_manifest = await _capture_workspace_command_manifest(ctx.workspace)
        except _WorkspaceManifestError:
            return self._error_result(
                authority,
                status="unavailable",
                error="workspace_observation_unavailable",
                content=(
                    "Structured command requires a complete bounded workspace "
                    "observation before dispatch."
                ),
            )
        runner_failure = docker_coding_toolchain_runner_admission_failure(
            ctx.runner,
            profile=self._profile,
        )
        if runner_failure is not None:
            return self._error_result(
                authority,
                status="unavailable",
                error=runner_failure,
                content=(
                    "Structured command admission changed before dispatch; no command was run."
                ),
            )
        command_argv = authority.command_argv(arguments)
        command = ExecCommand.process(*command_argv)
        requested_cwd = None if working_directory == "." else working_directory
        try:
            journal = await _start_command_journal(
                ctx,
                profile=self._profile,
                authority=authority,
                command_argv=command_argv,
                working_directory=working_directory,
                timeout_seconds=timeout_seconds,
                output_mode=output_mode,
                before_manifest=before_manifest,
            )
            if journal is not None:
                _bind_durable_runner_operation(ctx.runner, journal.record.get("runner_operation"))
                await journal.mark_dispatching()
        except _CommandJournalError:
            return self._error_result(
                authority,
                status="unavailable",
                error="durable_command_journal_unavailable",
                content="Structured command durable dispatch evidence was unavailable.",
            )
        runner_failure = docker_coding_toolchain_runner_admission_failure(
            ctx.runner,
            profile=self._profile,
        )
        if runner_failure is not None:
            result = self._error_result(
                authority,
                status="unavailable",
                error=runner_failure,
                content=("Structured command admission changed at dispatch; no command was run."),
            )
            return await self._finish_durable_result(journal, result)
        started_at = datetime.now(UTC)
        started_ns = perf_counter_ns()
        try:
            raw_result = await self._executors[authority.selector]._execute_resolved_command(
                ctx,
                command=command,
                cwd=requested_cwd,
                canonical_cwd=None,
                env=authority.environment,
                timeout_s=timeout_seconds,
                stdin=None,
                max_output_bytes=authority.max_output_bytes,
                policy_source=self,
                include_runner_evidence=True,
            )
            if journal is not None:
                runner_finished_at = datetime.now(UTC)
                try:
                    await journal.checkpoint_runner_terminal(
                        raw_result,
                        timing=_timing_evidence(
                            started_at,
                            runner_finished_at,
                            started_ns,
                        ),
                    )
                except _CommandJournalError:
                    result = self._error_result(
                        authority,
                        status="ambiguous",
                        error="durable_runner_terminal_unsettled",
                        content=(
                            "Structured command finished at the runner boundary but its durable "
                            "terminal evidence could not be settled."
                        ),
                        extra={"reconstruction_required": True},
                    )
                    return await self._finish_durable_result(journal, result)
        except RunnerExecutionError as exc:
            mutation_settlement = runner_workspace_mutation_settlement(result=None, error=exc)
            try:
                failed_after_manifest = await _capture_workspace_command_manifest(ctx.workspace)
            except _WorkspaceManifestError:
                failed_after_manifest = None
            finished_at = datetime.now(UTC)
            result = self._error_result(
                authority,
                status="failed",
                error="runner_execution_failed",
                content="Structured command failed at the runner execution boundary.",
                extra={
                    "diagnostic": copy_json_value(exc.diagnostic, "diagnostic"),
                    "workspace_mutation_settlement": mutation_settlement,
                    "cleanup_uncertain": mutation_settlement in {"deferred", "uncertain"},
                    "workspace_mutation_evidence": _command_mutation_evidence(
                        authority,
                        before=before_manifest,
                        after=failed_after_manifest,
                    ),
                    **_timing_evidence(started_at, finished_at, started_ns),
                },
                artifacts=exc.artifacts,
            )
            return await self._finish_durable_result(journal, result)
        except TypeError as exc:
            if str(exc) != "Runner returned invalid result type.":
                raise
            result = self._error_result(
                authority,
                status="failed",
                error="malformed_runner_result",
                content="Structured command returned malformed runner evidence.",
            )
            return await self._finish_durable_result(journal, result)
        try:
            finished_at = datetime.now(UTC)
            try:
                after_manifest = await _capture_workspace_command_manifest(ctx.workspace)
            except _WorkspaceManifestError:
                after_manifest = None
            result = await self._project_result(
                ctx,
                authority=authority,
                arguments=arguments,
                working_directory=working_directory,
                timeout_seconds=timeout_seconds,
                output_mode=output_mode,
                command_argv=command_argv,
                raw_result=raw_result,
                timing=_timing_evidence(started_at, finished_at, started_ns),
                before_manifest=before_manifest,
                after_manifest=after_manifest,
            )
            return await self._finish_durable_result(journal, result)
        except TypeError as exc:
            if str(exc) != "Runner returned invalid result type.":
                raise
            result = self._error_result(
                authority,
                status="failed",
                error="malformed_runner_result",
                content="Structured command returned malformed runner evidence.",
            )
            return await self._finish_durable_result(journal, result)

    async def _finish_durable_result(
        self,
        journal: _StructuredCommandJournal | None,
        result: ToolResult,
    ) -> ToolResult:
        if journal is None:
            return result
        structured = {} if result.structured is None else dict(result.structured)
        structured.setdefault("argv_sha256", journal.record["argv_sha256"])
        structured.setdefault(
            "working_directory_sha256",
            journal.record["working_directory_sha256"],
        )
        structured.setdefault("environment_identity", journal.record["environment_identity"])
        structured.setdefault(
            "mutation_scope_identity",
            journal.record["mutation_scope_identity"],
        )
        if journal.record.get("runner_operation") is not None:
            structured.setdefault("runner_operation", journal.record["runner_operation"])
        if journal.record.get("runner_terminal_identity") is not None:
            structured.setdefault(
                "runner_terminal_identity",
                journal.record["runner_terminal_identity"],
            )
        result = result.model_copy(update={"structured": structured})
        authority = next(
            (
                item
                for item in self._profile.structured_command_authorities
                if item.selector == journal.record.get("selector")
            ),
            None,
        )
        if (
            authority is not None
            and type(structured.get("stdout")) is str
            and type(structured.get("stderr")) is str
        ):
            result = _bound_command_result_publication(
                result,
                authority=authority,
                stdout=structured["stdout"],
                stderr=structured["stderr"],
                maximum=self._profile.result_publication_max_bytes,
            )
        try:
            await journal.terminal(result)
        except _CommandJournalError:
            structured.update(
                {
                    "status": "ambiguous",
                    "error": "durable_command_journal_unsettled",
                    "reconstruction_required": True,
                }
            )
            unsettled = result.model_copy(
                update={
                    "content": (
                        "Structured command acknowledgement is ambiguous; recover its durable "
                        "operation before considering any replay."
                    ),
                    "structured": structured,
                    "is_error": True,
                }
            )
            if (
                authority is not None
                and type(structured.get("stdout")) is str
                and type(structured.get("stderr")) is str
            ):
                unsettled = _bound_command_result_publication(
                    unsettled,
                    authority=authority,
                    stdout=structured["stdout"],
                    stderr=structured["stderr"],
                    maximum=self._profile.result_publication_max_bytes,
                )
            return unsettled
        return result

    async def reconcile_durable_tool_call(
        self,
        *,
        parent_session_id: str,
        parent_run_epoch: int,
        execution_profile_fingerprint: str | None,
        environment_name: str | None,
        environment_allocation_fingerprint: str | None,
        model_step_id: str,
        model_attempt_id: str,
        tool_round_id: str,
        tool_call_id: str,
        idempotency_key: str,
        arguments: dict[str, Any],
        started: bool,
        load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
        recovery_authority: DurableToolRecoveryAuthority | None = None,
    ) -> ToolResult | None:
        """Reconcile durable command evidence without ever replaying the command."""

        del started
        storage_key = _command_journal_key(parent_session_id, idempotency_key)
        record = await load_operation(storage_key)
        if record is None:
            return None
        return await _recover_command_journal_result(
            record,
            tool=self,
            parent_session_id=parent_session_id,
            parent_run_epoch=parent_run_epoch,
            execution_profile_fingerprint=execution_profile_fingerprint,
            environment_name=environment_name,
            environment_allocation_fingerprint=environment_allocation_fingerprint,
            model_step_id=model_step_id,
            model_attempt_id=model_attempt_id,
            tool_round_id=tool_round_id,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            arguments=arguments,
            recovery_authority=recovery_authority,
        )

    def _resolve_arguments(
        self,
        args: dict,
    ) -> tuple[DockerCodingCommandAuthority, tuple[str, ...], str, int, str]:
        return _resolve_structured_command_request(self._authorities, args)

    async def _project_result(
        self,
        ctx: ToolContext,
        *,
        authority: DockerCodingCommandAuthority,
        arguments: tuple[str, ...],
        working_directory: str,
        timeout_seconds: int,
        output_mode: str,
        command_argv: tuple[str, ...],
        raw_result: ToolResult,
        timing: Mapping[str, object],
        before_manifest: _WorkspaceCommandManifest,
        after_manifest: _WorkspaceCommandManifest | None,
    ) -> ToolResult:
        structured = raw_result.structured
        error = None if structured is None else structured.get("error")
        if error in {"command_denied", "command_approval_required"}:
            return self._error_result(
                authority,
                status=("denied" if error == "command_denied" else "approval_required"),
                error=str(error),
                content=raw_result.content,
                extra={} if structured is None else dict(structured),
                artifacts=raw_result.artifacts,
            )
        if error == "runner_unavailable" or structured is None:
            return self._error_result(
                authority,
                status="unavailable",
                error="runner_unavailable",
                content=raw_result.content,
                extra={} if structured is None else dict(structured),
                artifacts=raw_result.artifacts,
            )
        stdout = _required_text(structured, "stdout")
        stderr = _required_text(structured, "stderr")
        stdout_preview, stdout_projection_truncated = _truncate_utf8(
            stdout,
            maximum=authority.max_model_output_bytes,
        )
        stderr_preview, stderr_projection_truncated = _truncate_utf8(
            stderr,
            maximum=authority.max_model_output_bytes,
        )
        stdout_runner_truncated = _required_bool(structured, "stdout_truncated")
        stderr_runner_truncated = _required_bool(structured, "stderr_truncated")
        timed_out = _required_bool(structured, "timed_out")
        cancelled = _required_bool(structured, "cancelled")
        exit_code = _required_int(structured, "exit_code")
        mutation_settlement = _required_text(structured, "workspace_mutation_settlement")
        argv_digest = _argv_digest(command_argv)
        output_record = canonical_durable_json_bytes(
            {
                "schema": RUN_COMMAND_RESULT_SCHEMA,
                "selector": authority.selector,
                "selector_fingerprint": authority.fingerprint,
                "argv_sha256": argv_digest,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_runner_truncated": stdout_runner_truncated,
                "stderr_runner_truncated": stderr_runner_truncated,
            },
            "structured_command_output",
        )
        output_sha256 = "sha256:" + sha256(output_record).hexdigest()
        artifacts = _result_artifacts(structured)
        store_output = (
            output_mode == "summary_and_artifact"
            or stdout_projection_truncated
            or stderr_projection_truncated
        )
        artifact_status = "not_requested"
        if store_output:
            artifact_status, output_artifact = await _store_output_artifact(
                ctx,
                authority=authority,
                content=output_record,
                content_sha256=output_sha256,
            )
            if output_artifact is not None:
                artifacts.append(output_artifact)
        process_status = (
            "timed_out"
            if timed_out
            else "cancelled"
            if cancelled
            else "succeeded"
            if exit_code == 0
            else "nonzero"
        )
        mutation_evidence = _command_mutation_evidence(
            authority,
            before=before_manifest,
            after=after_manifest,
        )
        mutation_observation_complete = bool(mutation_evidence["complete"])
        mutation_scope_admitted = bool(mutation_evidence["scope_admitted"])
        cleanup_complete = mutation_settlement in {"complete", "runner_quiescent"}
        status = process_status
        collection_complete = not stdout_runner_truncated and not stderr_runner_truncated
        publication_complete = not store_output or artifact_status == "stored"
        if (
            not timed_out
            and not cancelled
            and (
                not collection_complete
                or not publication_complete
                or not mutation_observation_complete
            )
        ):
            status = "partial"
        if mutation_observation_complete and not mutation_scope_admitted:
            status = "failed"
        if status not in {"cancelled", "failed", "timed_out"}:
            if mutation_settlement == "deferred":
                status = "partial"
            elif mutation_settlement == "uncertain":
                status = "ambiguous"
        exit_code_admitted = exit_code in authority.allowed_exit_codes
        projected = {
            "schema": RUN_COMMAND_RESULT_SCHEMA,
            "status": status,
            "process_status": process_status,
            "selector": authority.selector,
            "selector_revision": authority.revision,
            "selector_fingerprint": authority.fingerprint,
            "executable": authority.executable,
            "argv_sha256": argv_digest,
            "argument_count": len(arguments),
            "working_directory_sha256": "sha256:"
            + sha256(working_directory.encode("utf-8")).hexdigest(),
            "effect": authority.effect,
            "parallel_safe": authority.parallel_safe,
            "idempotent": authority.idempotent,
            "environment_identity": _environment_identity(authority),
            "mutation_scope_identity": _mutation_scope_identity(authority),
            "timeout_seconds": timeout_seconds,
            "max_output_bytes": authority.max_output_bytes,
            "max_model_output_bytes": authority.max_model_output_bytes,
            "allowed_exit_codes": list(authority.allowed_exit_codes),
            "exit_code": exit_code,
            "exit_code_admitted": exit_code_admitted,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "stdout": stdout_preview,
            "stderr": stderr_preview,
            "stdout_truncated": stdout_runner_truncated or stdout_projection_truncated,
            "stderr_truncated": stderr_runner_truncated or stderr_projection_truncated,
            "stdout_runner_truncated": stdout_runner_truncated,
            "stderr_runner_truncated": stderr_runner_truncated,
            "stdout_projection_truncated": stdout_projection_truncated,
            "stderr_projection_truncated": stderr_projection_truncated,
            "stdout_bytes": _optional_nonnegative_int(structured, "stdout_bytes"),
            "stderr_bytes": _optional_nonnegative_int(structured, "stderr_bytes"),
            "output_sha256": output_sha256,
            "output_artifact_status": artifact_status,
            "output_collection_complete": collection_complete,
            "output_publication_complete": publication_complete,
            "workspace_mutation_settlement": mutation_settlement,
            "cleanup_uncertain": mutation_settlement in {"deferred", "uncertain"},
            "workspace_mutation_evidence": mutation_evidence,
            "session_id": ctx.session_id,
            "workspace_id": ctx.workspace_id,
            **dict(timing),
            **_context_receipt_evidence(ctx),
            **_runner_receipt_evidence(ctx.runner),
            **self._profile.evidence(),
            "artifacts": artifacts,
        }
        if not mutation_observation_complete:
            projected["error"] = "workspace_observation_incomplete"
        elif not mutation_scope_admitted:
            projected["error"] = "unexpected_workspace_mutation"
        elif mutation_settlement == "deferred":
            projected["error"] = "workspace_cleanup_deferred"
        elif mutation_settlement == "uncertain":
            projected["error"] = "workspace_cleanup_uncertain"
        result = ToolResult(
            content=_model_content(
                selector=authority.selector,
                status=status,
                exit_code=exit_code,
                stdout=stdout_preview,
                stderr=stderr_preview,
                truncated=projected["stdout_truncated"] or projected["stderr_truncated"],
            ),
            structured=projected,
            artifacts=list(artifacts),
            is_error=(
                timed_out
                or cancelled
                or not exit_code_admitted
                or not collection_complete
                or not publication_complete
                or not mutation_observation_complete
                or not mutation_scope_admitted
                or not cleanup_complete
            ),
        )
        publication_ceiling = self._profile.result_publication_max_bytes
        if _tool_result_publication_bytes(result) > publication_ceiling and not store_output:
            artifact_status, output_artifact = await _store_output_artifact(
                ctx,
                authority=authority,
                content=output_record,
                content_sha256=output_sha256,
            )
            if output_artifact is not None:
                artifacts.append(output_artifact)
            publication_complete = artifact_status == "stored"
            if (
                not timed_out
                and not cancelled
                and not publication_complete
                and status not in {"failed", "ambiguous"}
            ):
                status = "partial"
            projected["status"] = status
            projected["output_artifact_status"] = artifact_status
            projected["output_publication_complete"] = publication_complete
            projected["artifacts"] = artifacts
            if not publication_complete and "error" not in projected:
                projected["error"] = "output_publication_incomplete"
            result = ToolResult(
                content=_model_content(
                    selector=authority.selector,
                    status=status,
                    exit_code=exit_code,
                    stdout=stdout_preview,
                    stderr=stderr_preview,
                    truncated=(projected["stdout_truncated"] or projected["stderr_truncated"]),
                ),
                structured=projected,
                artifacts=list(artifacts),
                is_error=(result.is_error or not publication_complete),
            )
        return _bound_command_result_publication(
            result,
            authority=authority,
            stdout=stdout,
            stderr=stderr,
            maximum=publication_ceiling,
        )

    def _error_result(
        self,
        authority: DockerCodingCommandAuthority,
        *,
        status: str,
        error: str,
        content: str,
        extra: Mapping[str, object] | None = None,
        artifacts: Sequence[Mapping[str, object]] = (),
    ) -> ToolResult:
        result = ToolResult(
            content=content,
            structured={
                "schema": RUN_COMMAND_RESULT_SCHEMA,
                "status": status,
                "error": error,
                "selector": authority.selector,
                "selector_revision": authority.revision,
                "selector_fingerprint": authority.fingerprint,
                **self._profile.evidence(),
                **({} if extra is None else dict(extra)),
            },
            artifacts=list(artifacts),
            is_error=True,
        )
        if _tool_result_publication_bytes(result) <= self._profile.result_publication_max_bytes:
            return result
        return _minimal_publication_ceiling_result(
            authority,
            maximum=self._profile.result_publication_max_bytes,
        )


def _relative_working_directory(canonical: str | None, *, workspace_path: str) -> str:
    if canonical == workspace_path:
        return "."
    if canonical is None or not canonical.startswith(workspace_path + "/"):
        raise ValueError("Canonical command cwd escaped the admitted workspace.")
    return canonical.removeprefix(workspace_path + "/")


def _denied(reason: str) -> CommandPolicyResult:
    return CommandPolicyResult(decision=CommandPolicyDecision.DENY, reason=reason)


def _argv_digest(argv: tuple[str, ...]) -> str:
    return (
        "sha256:"
        + sha256(canonical_durable_json_bytes(list(argv), "structured_command_argv")).hexdigest()
    )


def _environment_identity(authority: DockerCodingCommandAuthority) -> str:
    return (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(
                [item.model_dump(mode="json") for item in authority.fixed_environment],
                "structured_command_environment",
            )
        ).hexdigest()
    )


def _mutation_scope_identity(authority: DockerCodingCommandAuthority) -> str:
    return (
        "sha256:"
        + sha256(
            canonical_durable_json_bytes(
                list(authority.mutation_path_prefixes),
                "structured_command_mutation_scope",
            )
        ).hexdigest()
    )


async def _capture_workspace_command_manifest(
    workspace: WorkspaceHandle,
) -> _WorkspaceCommandManifest:
    """Capture one complete content manifest within the coding copy bounds."""

    try:
        list_git_entries = getattr(workspace, "list_git_entries", None)
        if not callable(list_git_entries):
            raise _WorkspaceManifestError
        listing = await list_git_entries(limit=_COMMAND_MANIFEST_MAX_PATHS + 1)
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise _WorkspaceManifestError from None
    if (
        type(listing) is not WorkspaceGitEntryListResult
        or listing.truncated
        or listing.total_count != len(listing.entries)
        or len(listing.entries) > _COMMAND_MANIFEST_MAX_PATHS
    ):
        raise _WorkspaceManifestError

    entries: list[tuple[str, str, int, str]] = []
    total_bytes = 0
    for observed_entry in listing.entries:
        if type(observed_entry) is not WorkspaceGitEntry:
            raise _WorkspaceManifestError
        path = observed_entry.path
        if observed_entry.git_mode == "120000":
            target_sha256 = observed_entry.symlink_target_sha256
            target_bytes = observed_entry.symlink_target_bytes
            if target_sha256 is None or target_bytes is None:
                raise _WorkspaceManifestError
            total_bytes += target_bytes
            if total_bytes > _COMMAND_MANIFEST_MAX_TOTAL_BYTES:
                raise _WorkspaceManifestError
            entries.append((path, target_sha256, target_bytes, "120000"))
            continue
        try:
            result = await workspace.read_bytes(
                path,
                offset=0,
                max_bytes=_COMMAND_MANIFEST_MAX_FILE_BYTES + 1,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise _WorkspaceManifestError from None
        if (
            type(result) is not WorkspaceReadResult
            or result.truncated
            or result.offset != 0
            or result.redaction_truncated
            or result.source_bytes_read != result.total_bytes
            or result.sha256 is None
            or result.total_bytes > _COMMAND_MANIFEST_MAX_FILE_BYTES
            or result.git_mode != observed_entry.git_mode
        ):
            raise _WorkspaceManifestError
        total_bytes += result.total_bytes
        if total_bytes > _COMMAND_MANIFEST_MAX_TOTAL_BYTES:
            raise _WorkspaceManifestError
        entries.append((path, result.sha256, result.total_bytes, result.git_mode))
    return _WorkspaceCommandManifest(entries=tuple(entries), total_bytes=total_bytes)


def _copy_command_journal_record(value: object) -> dict[str, Any]:
    copied = copy_json_value(value, "structured_command_journal")
    if type(copied) is not dict:
        raise _CommandJournalError("Structured-command journal must be an object.")
    return copied


def _command_journal_key(parent_session_id: str, idempotency_key: str) -> str:
    material = canonical_durable_json_bytes(
        [parent_session_id, idempotency_key],
        "structured_command_journal_key",
    )
    return "run-command:v1:" + sha256(material).hexdigest()


def _sha256_identity(value: object, *, purpose: str) -> str:
    return "sha256:" + sha256(canonical_durable_json_bytes(value, purpose)).hexdigest()


def _durable_runner_resource_identity(runner: object) -> str | None:
    resolver = getattr(runner, "durable_resource_identity", None)
    if not callable(resolver):
        return None
    try:
        identity = resolver()
    except (RuntimeError, TypeError, ValueError):
        return None
    if (
        type(identity) is not str
        or len(identity) != 71
        or not identity.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in identity[7:])
    ):
        return None
    return identity


def _command_runner_operation_identity(
    *,
    storage_key: str,
    runner_resource_identity: str | None,
    argv_sha256: str,
    working_directory_sha256: str,
    environment_identity: str,
    timeout_seconds: int,
) -> dict[str, Any] | None:
    if runner_resource_identity is None:
        return None
    request_identity = _sha256_identity(
        {
            "argv_sha256": argv_sha256,
            "working_directory_sha256": working_directory_sha256,
            "environment_identity": environment_identity,
            "timeout_seconds": timeout_seconds,
        },
        purpose="structured_command_runner_request",
    )
    operation_id = _sha256_identity(
        [storage_key, runner_resource_identity, request_identity],
        purpose="structured_command_runner_operation",
    )
    return {
        "schema": _DURABLE_RUNNER_OPERATION_SCHEMA,
        "operation_id": operation_id,
        "runner_resource_identity": runner_resource_identity,
        "request_identity": request_identity,
        "process_identity": _sha256_identity(
            [operation_id, "process"], purpose="structured_command_process_identity"
        ),
        "output_identity": _sha256_identity(
            [operation_id, "output"], purpose="structured_command_output_identity"
        ),
        "artifact_identity": _sha256_identity(
            [operation_id, "artifact"], purpose="structured_command_artifact_identity"
        ),
        "cleanup_identity": _sha256_identity(
            [operation_id, "cleanup"], purpose="structured_command_cleanup_identity"
        ),
    }


def _bind_durable_runner_operation(runner: object, identity: object) -> None:
    if identity is None:
        return
    binder = getattr(runner, "bind_durable_command_operation", None)
    if not callable(binder):
        raise _CommandJournalError(
            "Structured-command runner cannot bind its durable operation identity."
        )
    try:
        binder(copy_json_value(identity, "runner_operation"))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise _CommandJournalError(
            "Structured-command runner rejected its durable operation identity."
        ) from exc


def _optional_command_identity_sha256(value: str | None) -> str | None:
    return None if value is None else "sha256:" + sha256(value.encode("utf-8")).hexdigest()


async def _start_command_journal(
    ctx: ToolContext,
    *,
    profile: DockerCodingToolchainProfile,
    authority: DockerCodingCommandAuthority,
    command_argv: tuple[str, ...],
    working_directory: str,
    timeout_seconds: int,
    output_mode: str,
    before_manifest: _WorkspaceCommandManifest,
) -> _StructuredCommandJournal | None:
    runtime_authority = _runtime_tool_invocation_authority(ctx)
    if runtime_authority is None:
        return None
    if (
        runtime_authority.tool_name != "run_command"
        or ctx.idempotency_key != runtime_authority.idempotency_key
    ):
        raise _CommandJournalError(
            "Structured-command durable authority does not match the invocation."
        )
    storage_key = _command_journal_key(ctx.session_id, runtime_authority.idempotency_key)
    runner_resource_identity = _durable_runner_resource_identity(ctx.runner)
    runner_operation = _command_runner_operation_identity(
        storage_key=storage_key,
        runner_resource_identity=runner_resource_identity,
        argv_sha256=_argv_digest(command_argv),
        working_directory_sha256=(
            "sha256:" + sha256(working_directory.encode("utf-8")).hexdigest()
        ),
        environment_identity=_environment_identity(authority),
        timeout_seconds=timeout_seconds,
    )
    record: dict[str, Any] = {
        "record_type": _COMMAND_JOURNAL_RECORD_TYPE,
        "schema_version": _COMMAND_JOURNAL_SCHEMA_VERSION,
        "state": "prepared",
        "parent_session_id": ctx.session_id,
        "parent_run_epoch": runtime_authority.parent_run_epoch,
        "environment_name": ctx.environment_name,
        "environment_allocation_fingerprint": (
            runtime_authority.environment_allocation_fingerprint
        ),
        "model_step_id": runtime_authority.model_step_id,
        "model_attempt_id": runtime_authority.model_attempt_id,
        "tool_round_id": runtime_authority.tool_round_id,
        "tool_call_id": runtime_authority.tool_call_id,
        "idempotency_key": runtime_authority.idempotency_key,
        "execution_profile_fingerprint": runtime_authority.execution_profile_fingerprint,
        "effective_arguments_sha256": runtime_authority.effective_arguments_sha256,
        "workspace_id_sha256": _optional_command_identity_sha256(ctx.workspace_id),
        "toolchain_profile_fingerprint": profile.fingerprint,
        "selector": authority.selector,
        "selector_fingerprint": authority.fingerprint,
        "argv_sha256": _argv_digest(command_argv),
        "working_directory_sha256": (
            "sha256:" + sha256(working_directory.encode("utf-8")).hexdigest()
        ),
        "environment_identity": _environment_identity(authority),
        "mutation_scope_identity": _mutation_scope_identity(authority),
        "effect": authority.effect,
        "idempotent": authority.idempotent,
        "timeout_seconds": timeout_seconds,
        "output_mode": output_mode,
        "before_manifest_fingerprint": before_manifest.fingerprint,
        "before_manifest_path_count": len(before_manifest.entries),
        "before_manifest_total_bytes": before_manifest.total_bytes,
        "before_manifest_entries": [list(entry) for entry in before_manifest.entries],
        "runner_operation": runner_operation,
        "runner_terminal_result": None,
        "runner_terminal_timing": None,
        "runner_terminal_identity": None,
        "terminal_result": None,
    }
    existing = await runtime_authority.load_durable_operation(storage_key)
    if existing is not None:
        raise _CommandJournalError(
            "A durable record already exists for this structured-command invocation."
        )
    try:
        persisted = await runtime_authority.compare_and_set_durable_operation(
            storage_key,
            None,
            record,
            {},
        )
    except Exception as exc:
        persisted = await runtime_authority.load_durable_operation(storage_key)
        if persisted != record:
            raise _CommandJournalError(
                "Structured-command dispatch intent publication failed."
            ) from exc
    if persisted != record:
        raise _CommandJournalError(
            "Structured-command dispatch intent returned conflicting evidence."
        )
    return _StructuredCommandJournal(runtime_authority, storage_key, record)


async def _recover_command_journal_result(
    raw_record: object,
    *,
    tool: RunCommandTool,
    parent_session_id: str,
    parent_run_epoch: int,
    execution_profile_fingerprint: str | None,
    environment_name: str | None,
    environment_allocation_fingerprint: str | None,
    model_step_id: str,
    model_attempt_id: str,
    tool_round_id: str,
    tool_call_id: str,
    idempotency_key: str,
    arguments: dict[str, Any],
    recovery_authority: DurableToolRecoveryAuthority | None,
) -> ToolResult:
    try:
        record = _copy_command_journal_record(raw_record)
    except (TypeError, ValueError, _CommandJournalError):
        return _command_recovery_refusal("durable_command_journal_invalid")
    expected_identity = {
        "record_type": _COMMAND_JOURNAL_RECORD_TYPE,
        "schema_version": _COMMAND_JOURNAL_SCHEMA_VERSION,
        "parent_session_id": parent_session_id,
        "parent_run_epoch": parent_run_epoch,
        "environment_name": environment_name,
        "model_step_id": model_step_id,
        "model_attempt_id": model_attempt_id,
        "tool_round_id": tool_round_id,
        "tool_call_id": tool_call_id,
        "idempotency_key": idempotency_key,
    }
    if any(record.get(field) != expected for field, expected in expected_identity.items()):
        return _command_recovery_refusal("durable_command_journal_identity_mismatch")
    if (
        execution_profile_fingerprint is None
        or record.get("execution_profile_fingerprint") != execution_profile_fingerprint
        or record.get("environment_allocation_fingerprint") != environment_allocation_fingerprint
    ):
        return _command_recovery_refusal("execution_profile_drift")
    try:
        argument_digest = sha256(
            canonical_durable_json_bytes(arguments, "run_command_recovery.arguments")
        ).hexdigest()
        authority, resolved_arguments, working_directory, timeout_seconds, output_mode = (
            tool._resolve_arguments(arguments)
        )
        command_argv = authority.command_argv(resolved_arguments)
    except (TypeError, ValueError, UnicodeError):
        return _command_recovery_refusal("durable_command_journal_argument_mismatch")
    expected_request = {
        "effective_arguments_sha256": argument_digest,
        "toolchain_profile_fingerprint": tool._profile.fingerprint,
        "selector": authority.selector,
        "selector_fingerprint": authority.fingerprint,
        "argv_sha256": _argv_digest(command_argv),
        "working_directory_sha256": (
            "sha256:" + sha256(working_directory.encode("utf-8")).hexdigest()
        ),
        "environment_identity": _environment_identity(authority),
        "mutation_scope_identity": _mutation_scope_identity(authority),
        "effect": authority.effect,
        "idempotent": authority.idempotent,
        "timeout_seconds": timeout_seconds,
        "output_mode": output_mode,
    }
    if any(record.get(field) != expected for field, expected in expected_request.items()):
        return _command_recovery_refusal("durable_command_journal_argument_mismatch")
    state = record.get("state")
    if state == "terminal":
        return _recover_terminal_command_result(
            record,
            authority=authority,
            profile=tool._profile,
        )
    if state == "prepared":
        result = tool._error_result(
            authority,
            status="reconstruction_required",
            error="command_not_dispatched",
            content=(
                "Recovered durable structured-command intent before dispatch; the command "
                "was not replayed."
            ),
            extra={
                "recovered": True,
                "replayed": False,
                "dispatch": "not_started",
                "reconstruction_required": True,
            },
        )
        return result
    if state != "dispatching" or record.get("terminal_result") is not None:
        return _command_recovery_refusal("durable_command_journal_invalid")

    runner_operation = record.get("runner_operation")
    runner_terminal_result = record.get("runner_terminal_result")
    runner_terminal_timing = record.get("runner_terminal_timing")
    runner_observation_state = "unknown"
    if (
        runner_terminal_result is None
        and type(runner_operation) is dict
        and recovery_authority is not None
        and recovery_authority.runner_resource_identity
        == runner_operation.get("runner_resource_identity")
        and recovery_authority.reconcile_runner_operation is not None
    ):
        try:
            observation = await recovery_authority.reconcile_runner_operation(
                copy_json_value(runner_operation, "runner_operation")
            )
        except Exception:
            observation = None
        if type(observation) is dict and observation.get("identity") == runner_operation:
            runner_observation_state = str(observation.get("state"))
            if observation.get("state") == "terminal":
                try:
                    observed_result = ExecResult.model_validate(observation.get("result"))
                except Exception:
                    return _command_recovery_refusal("durable_runner_terminal_result_invalid")
                runner_terminal_result = _raw_command_tool_result(observed_result).model_dump(
                    mode="json"
                )
                runner_terminal_timing = {
                    "started_at": None,
                    "finished_at": datetime.now(UTC).isoformat(),
                    "duration_ms": None,
                }
    if runner_terminal_result is not None:
        return await _recover_runner_terminal_command_result(
            record,
            raw_result=runner_terminal_result,
            timing=runner_terminal_timing,
            tool=tool,
            authority=authority,
            arguments=resolved_arguments,
            working_directory=working_directory,
            timeout_seconds=timeout_seconds,
            output_mode=output_mode,
            command_argv=command_argv,
            parent_session_id=parent_session_id,
            environment_name=environment_name,
            tool_call_id=tool_call_id,
            idempotency_key=idempotency_key,
            execution_profile_fingerprint=execution_profile_fingerprint,
            recovery_authority=recovery_authority,
        )

    after_manifest: _WorkspaceCommandManifest | None = None
    workspace_identity_matches = False
    if recovery_authority is not None and recovery_authority.workspace is not None:
        workspace = recovery_authority.workspace
        workspace_identity_matches = (
            recovery_authority.environment_name == environment_name
            and _optional_command_identity_sha256(getattr(workspace, "id", None))
            == record.get("workspace_id_sha256")
        )
        if workspace_identity_matches:
            try:
                after_manifest = await _capture_workspace_command_manifest(workspace)
            except _WorkspaceManifestError:
                after_manifest = None
    before_fingerprint = record.get("before_manifest_fingerprint")
    after_fingerprint = None if after_manifest is None else after_manifest.fingerprint
    workspace_changed = (
        None if after_fingerprint is None else after_fingerprint != before_fingerprint
    )
    mutation_evidence = {
        "complete": after_manifest is not None,
        "scope_admitted": False,
        "effect": authority.effect,
        "scope_identity": _mutation_scope_identity(authority),
        "before_manifest_fingerprint": before_fingerprint,
        "after_manifest_fingerprint": after_fingerprint,
        "changed_path_count": None,
        "changed_paths_identity": None,
        "unexpected_path_count": None,
        "unexpected_paths_identity": None,
        "workspace_changed": workspace_changed,
    }
    return tool._error_result(
        authority,
        status="ambiguous",
        error="command_acknowledgement_lost",
        content=(
            "Structured command dispatch was durable but its terminal acknowledgement was lost. "
            "The command was not replayed; inspect fresh workspace evidence before continuing."
        ),
        extra={
            "recovered": True,
            "replayed": False,
            "dispatch": "started",
            "process_status": (
                "running" if runner_observation_state in {"bound", "dispatching"} else "unknown"
            ),
            "runner_operation": runner_operation,
            "runner_observation_state": runner_observation_state,
            "output_collection_complete": False,
            "output_publication_complete": False,
            "workspace_identity_matches": workspace_identity_matches,
            "workspace_mutation_settlement": "uncertain",
            "cleanup_uncertain": True,
            "workspace_mutation_evidence": mutation_evidence,
            "reconstruction_required": True,
        },
    )


def _raw_command_tool_result(result: ExecResult) -> ToolResult:
    settlement = runner_workspace_mutation_settlement(result=result, error=None)
    structured = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "artifacts": copy_json_value(result.artifacts, "artifacts"),
        "stdout_bytes": result.stdout_bytes,
        "stderr_bytes": result.stderr_bytes,
        "workspace_mutation_settlement": settlement,
    }
    return ToolResult(
        structured=structured,
        artifacts=result.artifacts,
        is_error=result.timed_out or result.cancelled,
    )


def _recover_before_command_manifest(
    record: Mapping[str, object],
) -> _WorkspaceCommandManifest | None:
    raw_entries = record.get("before_manifest_entries")
    if type(raw_entries) is not list or len(raw_entries) > _COMMAND_MANIFEST_MAX_PATHS:
        return None
    entries: list[tuple[str, str, int, str]] = []
    total_bytes = 0
    for raw_entry in raw_entries:
        if type(raw_entry) is not list or len(raw_entry) != 4:
            return None
        path, content_sha256, byte_count, git_mode = raw_entry
        if (
            type(path) is not str
            or type(content_sha256) is not str
            or len(content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in content_sha256)
            or type(byte_count) is not int
            or byte_count < 0
        ):
            return None
        if git_mode == "120000":
            normalized_git_mode = "120000"
        elif git_mode == "100644":
            normalized_git_mode = "100644"
        elif git_mode == "100755":
            normalized_git_mode = "100755"
        else:
            return None
        try:
            if normalized_git_mode == "120000":
                WorkspaceGitEntry(
                    path=path,
                    git_mode="120000",
                    symlink_target_sha256=content_sha256,
                    symlink_target_bytes=byte_count,
                )
            else:
                WorkspaceGitEntry(path=path, git_mode=normalized_git_mode)
        except (TypeError, ValueError):
            return None
        total_bytes += byte_count
        if total_bytes > _COMMAND_MANIFEST_MAX_TOTAL_BYTES:
            return None
        entries.append((path, content_sha256, byte_count, normalized_git_mode))
    manifest = _WorkspaceCommandManifest(entries=tuple(entries), total_bytes=total_bytes)
    if (
        manifest.fingerprint != record.get("before_manifest_fingerprint")
        or len(entries) != record.get("before_manifest_path_count")
        or total_bytes != record.get("before_manifest_total_bytes")
    ):
        return None
    return manifest


async def _recover_runner_terminal_command_result(
    record: dict[str, Any],
    *,
    raw_result: object,
    timing: object,
    tool: RunCommandTool,
    authority: DockerCodingCommandAuthority,
    arguments: tuple[str, ...],
    working_directory: str,
    timeout_seconds: int,
    output_mode: str,
    command_argv: tuple[str, ...],
    parent_session_id: str,
    environment_name: str | None,
    tool_call_id: str,
    idempotency_key: str,
    execution_profile_fingerprint: str,
    recovery_authority: DurableToolRecoveryAuthority | None,
) -> ToolResult:
    if recovery_authority is None or recovery_authority.workspace is None:
        return _command_recovery_refusal("durable_runner_recovery_authority_unavailable")
    try:
        runner_result = ToolResult.model_validate(raw_result)
    except Exception:
        return _command_recovery_refusal("durable_runner_terminal_result_invalid")
    terminal_identity = _sha256_identity(
        runner_result.model_dump(mode="json"),
        purpose="structured_command_runner_terminal",
    )
    recorded_terminal_identity = record.get("runner_terminal_identity")
    if recorded_terminal_identity is not None and recorded_terminal_identity != terminal_identity:
        return _command_recovery_refusal("durable_runner_terminal_result_invalid")
    before_manifest = _recover_before_command_manifest(record)
    if before_manifest is None:
        return _command_recovery_refusal("durable_command_before_manifest_invalid")
    workspace = recovery_authority.workspace
    if recovery_authority.environment_name != environment_name or _optional_command_identity_sha256(
        getattr(workspace, "id", None)
    ) != record.get("workspace_id_sha256"):
        return _command_recovery_refusal("durable_command_workspace_identity_mismatch")
    try:
        after_manifest = await _capture_workspace_command_manifest(workspace)
    except _WorkspaceManifestError:
        after_manifest = None
    recovered_timing = (
        copy_json_value(timing, "runner_terminal_timing") if type(timing) is dict else {}
    )
    if not {
        "started_at",
        "finished_at",
        "duration_ms",
    }.issubset(recovered_timing):
        observed_at = datetime.now(UTC).isoformat()
        recovered_timing = {
            "started_at": observed_at,
            "finished_at": observed_at,
            "duration_ms": 0,
        }
    context = ToolContext(
        session_id=parent_session_id,
        agent_name=recovery_authority.agent_name,
        environment_name=environment_name,
        workspace_id=getattr(workspace, "id", None),
        idempotency_key=idempotency_key,
        workspace=workspace,
        metadata={
            "tool_call_id": tool_call_id,
            "idempotency_key": idempotency_key,
            "execution_profile_fingerprint": execution_profile_fingerprint,
        },
    )
    try:
        result = await tool._project_result(
            context,
            authority=authority,
            arguments=arguments,
            working_directory=working_directory,
            timeout_seconds=timeout_seconds,
            output_mode=output_mode,
            command_argv=command_argv,
            raw_result=runner_result,
            timing=recovered_timing,
            before_manifest=before_manifest,
            after_manifest=after_manifest,
        )
    except (TypeError, ValueError):
        return _command_recovery_refusal("durable_runner_terminal_result_invalid")
    structured = {} if result.structured is None else dict(result.structured)
    structured.update(
        {
            "runner_operation": record.get("runner_operation"),
            "runner_terminal_identity": terminal_identity,
            "recovered": True,
            "replayed": False,
            "dispatch": "runner_terminal_evidence",
            "runner_observation_state": "terminal",
            "reconstruction_required": False,
        }
    )
    result = result.model_copy(update={"structured": structured})
    raw_structured = runner_result.structured
    if raw_structured is None:
        return _command_recovery_refusal("durable_runner_terminal_result_invalid")
    result = _bound_command_result_publication(
        result,
        authority=authority,
        stdout=_required_text(raw_structured, "stdout"),
        stderr=_required_text(raw_structured, "stderr"),
        maximum=tool._profile.result_publication_max_bytes,
    )
    desired = _copy_command_journal_record(record)
    desired.update(
        {
            "state": "terminal",
            "runner_terminal_result": runner_result.model_dump(mode="json"),
            "runner_terminal_timing": recovered_timing,
            "runner_terminal_identity": terminal_identity,
            "terminal_result": result.model_dump(mode="json"),
        }
    )
    try:
        persisted = await recovery_authority.compare_and_set_operation(
            _command_journal_key(parent_session_id, idempotency_key),
            record,
            desired,
            {},
        )
    except Exception:
        return _command_recovery_refusal("durable_command_recovery_settlement_failed")
    if persisted != desired:
        return _command_recovery_refusal("durable_command_recovery_settlement_failed")
    return result


def _recover_terminal_command_result(
    record: dict[str, Any],
    *,
    authority: DockerCodingCommandAuthority,
    profile: DockerCodingToolchainProfile,
) -> ToolResult:
    raw_result = record.get("terminal_result")
    try:
        result = ToolResult.model_validate(raw_result)
    except Exception:
        return _command_recovery_refusal("durable_command_terminal_result_invalid")
    structured = None if result.structured is None else dict(result.structured)
    if structured is None or any(
        structured.get(field) != expected
        for field, expected in {
            "selector": authority.selector,
            "selector_fingerprint": authority.fingerprint,
            "toolchain_profile_fingerprint": profile.fingerprint,
            "argv_sha256": record.get("argv_sha256"),
        }.items()
    ):
        return _command_recovery_refusal("durable_command_terminal_result_invalid")
    structured.update(
        {
            "recovered": True,
            "replayed": False,
            "dispatch": "terminal_evidence",
        }
    )
    recovered = result.model_copy(update={"structured": structured})
    if type(structured.get("stdout")) is str and type(structured.get("stderr")) is str:
        recovered = _bound_command_result_publication(
            recovered,
            authority=authority,
            stdout=structured["stdout"],
            stderr=structured["stderr"],
            maximum=profile.result_publication_max_bytes,
        )
    return recovered


def _command_recovery_refusal(error: str) -> ToolResult:
    return ToolResult(
        content=(
            "Structured command recovery could not authenticate complete durable evidence; "
            "the command was not replayed."
        ),
        structured={
            "schema": RUN_COMMAND_RESULT_SCHEMA,
            "status": "reconstruction_required",
            "error": error,
            "recovered": True,
            "replayed": False,
            "reconstruction_required": True,
        },
        is_error=True,
    )


def _command_mutation_evidence(
    authority: DockerCodingCommandAuthority,
    *,
    before: _WorkspaceCommandManifest,
    after: _WorkspaceCommandManifest | None,
) -> dict[str, object]:
    scope_identity = _mutation_scope_identity(authority)
    if after is None:
        return {
            "complete": False,
            "scope_admitted": False,
            "effect": authority.effect,
            "scope_identity": scope_identity,
            "before_manifest_fingerprint": before.fingerprint,
            "after_manifest_fingerprint": None,
            "changed_path_count": None,
            "changed_paths_identity": None,
            "unexpected_path_count": None,
            "unexpected_paths_identity": None,
        }

    before_paths = before.paths
    after_paths = after.paths
    changed_paths = tuple(
        sorted(
            path
            for path in set(before_paths) | set(after_paths)
            if before_paths.get(path) != after_paths.get(path)
        )
    )
    unexpected_paths = tuple(
        path
        for path in changed_paths
        if authority.effect == "read_only"
        or not _path_is_within_mutation_scope(path, authority.mutation_path_prefixes)
    )
    return {
        "complete": True,
        "scope_admitted": not unexpected_paths,
        "effect": authority.effect,
        "scope_identity": scope_identity,
        "before_manifest_fingerprint": before.fingerprint,
        "after_manifest_fingerprint": after.fingerprint,
        "changed_path_count": len(changed_paths),
        "changed_paths_identity": _path_set_identity(
            changed_paths,
            purpose="structured_command_changed_paths",
        ),
        "unexpected_path_count": len(unexpected_paths),
        "unexpected_paths_identity": _path_set_identity(
            unexpected_paths,
            purpose="structured_command_unexpected_paths",
        ),
    }


def _path_is_within_mutation_scope(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix + "/") for prefix in prefixes)


def _path_set_identity(paths: tuple[str, ...], *, purpose: str) -> str:
    return "sha256:" + sha256(canonical_durable_json_bytes(list(paths), purpose)).hexdigest()


def _context_receipt_evidence(ctx: ToolContext) -> dict[str, object]:
    evidence: dict[str, object] = {}
    for source, target in (
        ("tool_call_id", "tool_call_id"),
        ("approval_id", "approval_id"),
        ("idempotency_key", "idempotency_key"),
        ("execution_profile_fingerprint", "execution_profile_fingerprint"),
    ):
        value = ctx.metadata.get(source)
        if type(value) is str:
            evidence[target] = value
    return evidence


def _timing_evidence(
    started_at: datetime,
    finished_at: datetime,
    started_ns: int,
) -> dict[str, object]:
    return {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_ms": max(0, (perf_counter_ns() - started_ns) // 1_000_000),
    }


def _runner_receipt_evidence(runner: object) -> dict[str, object]:
    if not isinstance(runner, _ReceiptAdmissionRunner):
        return {}
    try:
        candidate = runner.execution_admission_candidate()
    except (RuntimeError, TypeError, ValueError):
        return {}
    if candidate is None or type(candidate) is not ExecutionAdmissionCandidate:
        return {}
    evidence = candidate.evidence
    material: dict[str, object] = {
        "docker_environment_fingerprint": evidence.environment_fingerprint,
        "docker_image_fingerprint": evidence.image_fingerprint,
    }
    if evidence.tool_requirements is not None:
        material["docker_tool_requirements_identity"] = (
            "sha256:"
            + sha256(
                canonical_durable_json_bytes(
                    evidence.tool_requirements.model_dump(mode="json"),
                    "docker_tool_requirements",
                )
            ).hexdigest()
        )
    return material


def _required_text(structured: Mapping[str, object], field_name: str) -> str:
    value = structured.get(field_name)
    if type(value) is not str:
        raise TypeError("Runner returned invalid result type.")
    return value


def _required_bool(structured: Mapping[str, object], field_name: str) -> bool:
    value = structured.get(field_name)
    if type(value) is not bool:
        raise TypeError("Runner returned invalid result type.")
    return value


def _required_int(structured: Mapping[str, object], field_name: str) -> int:
    value = structured.get(field_name)
    if type(value) is not int:
        raise TypeError("Runner returned invalid result type.")
    return value


def _optional_nonnegative_int(
    structured: Mapping[str, object],
    field_name: str,
) -> int | None:
    value = structured.get(field_name)
    if value is None:
        return None
    if type(value) is not int or value < 0:
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
    if maximum < len(marker):
        return "", bool(encoded)
    prefix = encoded[: maximum - len(marker)]
    return prefix.decode("utf-8", errors="ignore").rstrip() + _OUTPUT_TRUNCATION_MARKER, True


def _tool_result_publication_bytes(result: ToolResult) -> int:
    """Measure the exact compact JSON representation admitted for publication."""

    return len(result.model_dump_json().encode("utf-8"))


def _bound_command_result_publication(
    result: ToolResult,
    *,
    authority: DockerCodingCommandAuthority,
    stdout: str,
    stderr: str,
    maximum: int,
) -> ToolResult:
    """Fit the complete duplicated ToolResult beneath the profile publication ceiling."""

    if _tool_result_publication_bytes(result) <= maximum:
        return result
    if result.structured is None:
        return _minimal_publication_ceiling_result(authority, maximum=maximum)

    original = dict(result.structured)

    def candidate(preview_maximum: int) -> ToolResult:
        stdout_preview, stdout_publication_truncated = _truncate_utf8(
            stdout,
            maximum=preview_maximum,
        )
        stderr_preview, stderr_publication_truncated = _truncate_utf8(
            stderr,
            maximum=preview_maximum,
        )
        projected = dict(original)
        projected.update(
            {
                "stdout": stdout_preview,
                "stderr": stderr_preview,
                "stdout_truncated": bool(projected.get("stdout_runner_truncated"))
                or stdout_publication_truncated,
                "stderr_truncated": bool(projected.get("stderr_runner_truncated"))
                or stderr_publication_truncated,
                "stdout_projection_truncated": stdout_publication_truncated,
                "stderr_projection_truncated": stderr_publication_truncated,
                "result_publication_ceiling_applied": True,
            }
        )
        return ToolResult(
            content=_model_content(
                selector=authority.selector,
                status=str(projected["status"]),
                exit_code=int(projected["exit_code"]),
                stdout=stdout_preview,
                stderr=stderr_preview,
                truncated=(projected["stdout_truncated"] or projected["stderr_truncated"]),
            ),
            structured=projected,
            artifacts=list(result.artifacts),
            is_error=result.is_error,
        )

    lower = 0
    upper = authority.max_model_output_bytes
    bounded = candidate(0)
    if _tool_result_publication_bytes(bounded) > maximum:
        return _minimal_publication_ceiling_result(authority, maximum=maximum)
    while lower < upper:
        midpoint = (lower + upper + 1) // 2
        attempted = candidate(midpoint)
        if _tool_result_publication_bytes(attempted) <= maximum:
            lower = midpoint
            bounded = attempted
        else:
            upper = midpoint - 1
    return bounded


def _minimal_publication_ceiling_result(
    authority: DockerCodingCommandAuthority,
    *,
    maximum: int,
) -> ToolResult:
    """Fail closed when receipt metadata alone cannot fit the admitted ceiling."""

    result = ToolResult(
        structured={
            "schema": RUN_COMMAND_RESULT_SCHEMA,
            "status": "partial",
            "error": "result_publication_ceiling_exceeded",
            "selector": authority.selector,
            "result_publication_max_bytes": maximum,
        },
        is_error=True,
    )
    if _tool_result_publication_bytes(result) <= maximum:
        return result
    result = ToolResult(
        structured={"error": "result_publication_ceiling_exceeded"},
        is_error=True,
    )
    if _tool_result_publication_bytes(result) <= maximum:
        return result
    return ToolResult(is_error=True)


async def _store_output_artifact(
    ctx: ToolContext,
    *,
    authority: DockerCodingCommandAuthority,
    content: bytes,
    content_sha256: str,
) -> tuple[str, dict[str, object] | None]:
    artifact_store = ctx.artifact_store
    if artifact_store is None:
        return "unavailable", None
    artifact_id = None
    if ctx.idempotency_key is not None:
        identity = sha256(
            b"cayu-structured-command-output-v1\0"
            + ctx.session_id.encode("utf-8")
            + b"\0"
            + ctx.idempotency_key.encode("utf-8")
            + b"\0"
            + authority.selector.encode("utf-8")
        ).hexdigest()[:32]
        artifact_id = f"art_{identity}"
    try:
        artifact = await artifact_store.put_bytes(
            content,
            artifact_id=artifact_id,
            filename=f"command-{authority.selector}-output.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            environment_name=ctx.environment_name,
            metadata={
                "operation": "run_command",
                "selector": authority.selector,
                "selector_fingerprint": authority.fingerprint,
                "content_sha256": content_sha256,
                "result_schema": RUN_COMMAND_RESULT_SCHEMA,
            },
        )
    except Exception:
        return "failed", None
    if type(artifact) is not ArtifactMetadata:
        return "failed", None
    return "stored", artifact.model_dump(mode="json")


def _model_content(
    *,
    selector: str,
    status: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    truncated: bool,
) -> str:
    first = f"Command selector {selector!r} {status}."
    if status == "nonzero":
        first = f"Command selector {selector!r} settled with exit code {exit_code}."
    sections = [first]
    if stdout.strip():
        sections.append(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        sections.append(f"stderr:\n{stderr.strip()}")
    if truncated:
        sections.append("Command output was truncated; inspect retained artifact evidence.")
    return "\n\n".join(sections)


__all__ = [
    "RUN_COMMAND_RESULT_SCHEMA",
    "STRUCTURED_COMMAND_TOOL_POLICY_SCHEMA",
    "RunCommandTool",
    "StructuredCommandToolPolicy",
]
