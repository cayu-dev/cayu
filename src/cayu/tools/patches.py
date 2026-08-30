"""Bounded structured multi-file workspace patching."""

from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import posixpath
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast

from cayu._validation import (
    canonical_durable_json_bytes,
    require_nonblank,
    require_unicode_scalar_text,
)
from cayu.artifacts import ArtifactMetadata, ArtifactScope
from cayu.core.tools import (
    DurableToolRecoveryAuthority,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _runtime_tool_invocation_authority,
)
from cayu.tools._errors import (
    reject_unknown_tool_arguments,
    structured_invalid_arguments,
    tool_argument_validation,
)
from cayu.tools._redaction import (
    InvocationRedactorSnapshot,
    active_secret_redactor_snapshot,
    record_ambiguous_secret_output,
)
from cayu.vaults import SecretRedactor
from cayu.workspaces import (
    Workspace,
    WorkspaceMoveAmbiguousError,
    WorkspaceMoveResult,
    WorkspaceMoveUnsupportedError,
    WorkspaceMutationResult,
    WorkspacePreconditionUnsupportedError,
    WorkspaceReadResult,
    WorkspaceRevisionMismatchError,
)

MAX_PATCH_OPERATIONS = 100
MAX_PATCH_PATHS = 200
MAX_PATCH_EDITS = 100
MAX_PATCH_REPLACEMENTS_PER_EDIT = 1_000
MAX_PATCH_REPLACEMENTS = MAX_PATCH_EDITS * MAX_PATCH_REPLACEMENTS_PER_EDIT
MAX_PATCH_PATH_BYTES = 4_096
MAX_PATCH_INPUT_BYTES = 2 * 1024 * 1024
MAX_PATCH_FILE_BYTES = 4 * 1024 * 1024
MAX_PATCH_SOURCE_BYTES = 8 * 1024 * 1024
MAX_PATCH_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_PATCH_RETAINED_DIFF_BYTES = 2 * 1024 * 1024
DEFAULT_PATCH_DIFF_PREVIEW_BYTES = 64 * 1024
MAX_PATCH_DIFF_PREVIEW_BYTES = 200 * 1024
DEFAULT_PATCH_FILE_DIFF_PREVIEW_BYTES = 32 * 1024
MAX_PATCH_ARTIFACT_BYTES = 3 * 1024 * 1024
MAX_PATCH_PROJECTED_PATH_BYTES = 512
MAX_PATCH_PROJECTED_REVISION_BYTES = 256
MAX_PATCH_RESULT_BYTES = 1024 * 1024
PATCH_RESULT_VERSION = 2
DEFAULT_PATCH_PROTECTED_ENTRY_NAMES = (".cayu", ".git", ".runtime")
_PATCH_JOURNAL_RECORD_TYPE = "cayu.apply_patch.journal"
_PATCH_JOURNAL_SCHEMA_VERSION = 2

_APPLY_PATCH_ARGUMENTS = frozenset({"operations"})
_OPERATION_FIELDS = frozenset(
    {"type", "path", "from_path", "to_path", "expected_revision", "content", "edits"}
)
_EDIT_FIELDS = frozenset({"old_text", "new_text", "expected_replacements"})
_PATCH_TYPES = frozenset({"create", "update", "delete", "move"})

PatchOperationType = Literal["create", "update", "delete", "move"]
PatchOperationStatus = Literal["applied", "not_started", "conflict", "failed", "unknown"]
PatchOutcome = Literal[
    "applied",
    "precondition_failed",
    "partial",
    "ambiguous",
    "unsupported",
    "cancelled",
    "failed",
]


@dataclass(frozen=True, slots=True)
class _PatchEdit:
    old_text: str
    new_text: str
    expected_replacements: int


@dataclass(frozen=True, slots=True)
class _PatchIntent:
    index: int
    operation_type: PatchOperationType
    source_path: str | None
    destination_path: str | None
    expected_revision: str | None
    content: str | None = None
    edits: tuple[_PatchEdit, ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedOperation:
    intent: _PatchIntent
    before: bytes | None
    before_revision: str | None
    before_sha256: str | None
    after: bytes | None
    after_sha256: str | None
    replacement_count: int
    diff: str


@dataclass(frozen=True, slots=True)
class _PreparedPatch:
    patch_id: str
    operations: tuple[_PreparedOperation, ...]
    application_plan: tuple[int, ...]
    counts: Mapping[str, int]
    total_source_bytes: int
    total_output_bytes: int
    aggregate_diff: str


class _PatchPreflightError(Exception):
    def __init__(
        self,
        category: str,
        *,
        operation_index: int | None = None,
        path: str | None = None,
    ) -> None:
        self.category = category
        self.operation_index = operation_index
        self.path = path
        super().__init__(category)


class _PatchJournalError(RuntimeError):
    """Durable patch-boundary evidence could not be advanced safely."""


class _PatchDurableJournal:
    """Content-free durable boundary for one runtime-owned patch dispatch."""

    def __init__(self, authority: Any, storage_key: str, record: dict[str, Any]) -> None:
        self.authority = authority
        self.storage_key = storage_key
        self.record = record

    async def mark_dispatching(self, operation_index: int) -> None:
        desired = _copy_json_object(self.record)
        operation = _journal_operation(desired, operation_index)
        if operation.get("status") != "not_started" or desired.get("active_operation") is not None:
            raise _PatchJournalError("Patch journal operation boundary is invalid.")
        operation["status"] = "unknown"
        desired["state"] = "applying"
        desired["active_operation"] = operation_index
        await self._advance(desired)

    async def settle(
        self,
        operation_index: int,
        status: PatchOperationStatus,
        *,
        move_fidelity: str | None = None,
        after_revision: str | None = None,
        after_sha256: str | None = None,
        after_bytes: int | None = None,
    ) -> None:
        desired = _copy_json_object(self.record)
        operation = _journal_operation(desired, operation_index)
        if desired.get("active_operation") != operation_index:
            raise _PatchJournalError("Patch journal settlement boundary is invalid.")
        operation["status"] = status
        operation["move_fidelity"] = move_fidelity
        operation["after_revision_sha256"] = _optional_identity_sha256(after_revision)
        operation["after_sha256"] = after_sha256
        operation["after_bytes"] = after_bytes
        desired["state"] = "in_progress"
        desired["active_operation"] = None
        await self._advance(desired)

    async def terminal(self, outcome: PatchOutcome, failure_category: str | None) -> None:
        desired = _copy_json_object(self.record)
        desired["state"] = "terminal"
        desired["terminal_outcome"] = outcome
        desired["failure_category"] = failure_category
        try:
            desired = self.authority.seal_durable_output(desired)
        except Exception as exc:
            raise _PatchJournalError("Patch journal terminal sealing failed.") from exc
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
                raise _PatchJournalError("Patch journal publication failed.") from exc
        if persisted != desired:
            raise _PatchJournalError("Patch journal publication returned conflicting evidence.")
        self.record = desired


def _apply_patch_tool_spec(*, max_operations: int) -> ToolSpec:
    edit_schema = {
        "type": "object",
        "properties": {
            "old_text": {"type": "string", "minLength": 1},
            "new_text": {"type": "string"},
            "expected_replacements": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_PATCH_REPLACEMENTS_PER_EDIT,
                "default": 1,
            },
        },
        "required": ["old_text", "new_text"],
        "additionalProperties": False,
    }
    operation_schema = {
        "type": "object",
        "properties": {
            "type": {"type": "string", "enum": sorted(_PATCH_TYPES)},
            "path": {"type": "string", "minLength": 1},
            "from_path": {"type": "string", "minLength": 1},
            "to_path": {"type": "string", "minLength": 1},
            "expected_revision": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4_096,
            },
            "content": {"type": "string"},
            "edits": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_PATCH_EDITS,
                "items": edit_schema,
            },
        },
        "required": ["type"],
        "additionalProperties": False,
    }
    return ToolSpec(
        name="apply_patch",
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        workspace_mutation=True,
        description=(
            "Apply one bounded structured multi-file UTF-8 patch to the active workspace. "
            "Supports create, exact update, conditional delete, and conditional no-overwrite "
            "move. The complete patch is preflighted before mutation, but application is a "
            "deterministic sequence rather than a cross-file transaction. A partial, ambiguous, "
            "or cancelled result requires fresh reads before repair."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": max_operations,
                    "items": operation_schema,
                }
            },
            "required": ["operations"],
            "additionalProperties": False,
        },
    )


class ApplyPatchTool(Tool):
    """Apply one fully preflighted, bounded multi-file patch."""

    spec = _apply_patch_tool_spec(max_operations=MAX_PATCH_OPERATIONS)

    def __init__(
        self,
        *,
        max_operations: int = MAX_PATCH_OPERATIONS,
        max_input_bytes: int = MAX_PATCH_INPUT_BYTES,
        max_file_bytes: int = MAX_PATCH_FILE_BYTES,
        max_source_bytes: int = MAX_PATCH_SOURCE_BYTES,
        max_output_bytes: int = MAX_PATCH_OUTPUT_BYTES,
        max_retained_diff_bytes: int = MAX_PATCH_RETAINED_DIFF_BYTES,
        max_diff_preview_bytes: int = DEFAULT_PATCH_DIFF_PREVIEW_BYTES,
        max_file_diff_preview_bytes: int = DEFAULT_PATCH_FILE_DIFF_PREVIEW_BYTES,
        max_artifact_bytes: int = MAX_PATCH_ARTIFACT_BYTES,
        protected_entry_names: Iterable[str] = DEFAULT_PATCH_PROTECTED_ENTRY_NAMES,
        spec: ToolSpec | None = None,
    ) -> None:
        self.max_operations = _configuration_int(
            max_operations,
            "max_operations",
            maximum=MAX_PATCH_OPERATIONS,
        )
        self.max_input_bytes = _configuration_int(
            max_input_bytes,
            "max_input_bytes",
            maximum=MAX_PATCH_INPUT_BYTES,
        )
        self.max_file_bytes = _configuration_int(
            max_file_bytes,
            "max_file_bytes",
            maximum=MAX_PATCH_FILE_BYTES,
        )
        self.max_source_bytes = _configuration_int(
            max_source_bytes,
            "max_source_bytes",
            maximum=MAX_PATCH_SOURCE_BYTES,
        )
        self.max_output_bytes = _configuration_int(
            max_output_bytes,
            "max_output_bytes",
            maximum=MAX_PATCH_OUTPUT_BYTES,
        )
        self.max_retained_diff_bytes = _configuration_int(
            max_retained_diff_bytes,
            "max_retained_diff_bytes",
            maximum=MAX_PATCH_RETAINED_DIFF_BYTES,
        )
        self.max_diff_preview_bytes = _configuration_int(
            max_diff_preview_bytes,
            "max_diff_preview_bytes",
            maximum=MAX_PATCH_DIFF_PREVIEW_BYTES,
        )
        self.max_file_diff_preview_bytes = _configuration_int(
            max_file_diff_preview_bytes,
            "max_file_diff_preview_bytes",
            maximum=MAX_PATCH_DIFF_PREVIEW_BYTES,
        )
        self.max_artifact_bytes = _configuration_int(
            max_artifact_bytes,
            "max_artifact_bytes",
            maximum=MAX_PATCH_ARTIFACT_BYTES,
        )
        self.protected_entry_names = _validate_protected_entry_names(protected_entry_names)
        super().__init__(
            spec=(
                _apply_patch_tool_spec(max_operations=self.max_operations) if spec is None else spec
            )
        )
        self.behavior_profile_id = _behavior_profile_id(self._execution_profile_material())

    @property
    def _publish_arguments(self) -> bool:
        # Patch bodies contain source text. Runtime policy still evaluates and
        # commits to the private effective envelope, but terminal/public event
        # surfaces receive the quarantined argument state rather than content.
        return False

    def _execution_profile_material(self) -> dict[str, object]:
        return {
            "version": PATCH_RESULT_VERSION,
            "max_operations": self.max_operations,
            "max_edits": MAX_PATCH_EDITS,
            "max_replacements_per_edit": MAX_PATCH_REPLACEMENTS_PER_EDIT,
            "max_replacements": MAX_PATCH_REPLACEMENTS,
            "max_input_bytes": self.max_input_bytes,
            "max_file_bytes": self.max_file_bytes,
            "max_source_bytes": self.max_source_bytes,
            "max_output_bytes": self.max_output_bytes,
            "max_retained_diff_bytes": self.max_retained_diff_bytes,
            "max_diff_preview_bytes": self.max_diff_preview_bytes,
            "max_file_diff_preview_bytes": self.max_file_diff_preview_bytes,
            "max_artifact_bytes": self.max_artifact_bytes,
            "max_projected_path_bytes": MAX_PATCH_PROJECTED_PATH_BYTES,
            "max_projected_revision_bytes": MAX_PATCH_PROJECTED_REVISION_BYTES,
            "max_result_bytes": MAX_PATCH_RESULT_BYTES,
            "protected_entry_names": list(self.protected_entry_names),
            "application_order": ["move", "create", "update", "delete"],
            "cross_file_atomic": False,
        }

    @structured_invalid_arguments
    async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
        initial_snapshot = active_secret_redactor_snapshot(ctx)
        with tool_argument_validation():
            reject_unknown_tool_arguments(args, allowed=_APPLY_PATCH_ARGUMENTS)
            raw_operations = args.get("operations")
            if not isinstance(raw_operations, list):
                raise ValueError("Tool argument `operations` must be an array.")
            if not raw_operations:
                raise ValueError("Tool argument `operations` cannot be empty.")
            if len(raw_operations) > self.max_operations:
                raise ValueError(
                    f"Tool argument `operations` must contain at most {self.max_operations} items."
                )
            try:
                input_bytes = len(
                    json.dumps(
                        args,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                )
            except (TypeError, UnicodeEncodeError) as exc:
                raise ValueError("Tool arguments must contain valid JSON Unicode values.") from exc
            if input_bytes > self.max_input_bytes:
                raise ValueError(f"Tool arguments exceed max_input_bytes={self.max_input_bytes}.")

        observation_workspace = _require_observation_workspace(ctx)
        mutation_workspace = _require_mutation_workspace(ctx)
        if observation_workspace is None or mutation_workspace is None:
            return ToolResult(
                content="No workspace configured for this tool call.",
                structured={"outcome": "unsupported", "reason": "workspace_missing"},
                is_error=True,
            )
        try:
            intents = _validate_patch_intents(
                raw_operations,
                protected_entry_names=self.protected_entry_names,
            )
            prepared = await self._preflight(observation_workspace, intents)
        except _PatchPreflightError as error:
            return _preflight_failure_result(ctx, error, initial_snapshot)

        projection_snapshot = _stable_projection_snapshot(ctx, initial_snapshot)
        if projection_snapshot is None:
            return _unstable_projection_result(outcome="precondition_failed", mutated=False)
        try:
            journal = await _start_patch_journal(
                ctx,
                prepared,
                behavior_profile_id=self.behavior_profile_id,
            )
        except _PatchJournalError:
            return _journal_failure_result(mutated=False)
        return await self._apply(
            ctx,
            observation_workspace,
            mutation_workspace,
            prepared,
            projection_snapshot,
            journal=journal,
        )

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
        """Reconstruct durable patch boundaries without dispatching mutations."""

        del started
        storage_key = _patch_journal_key(parent_session_id, idempotency_key)
        record = await load_operation(storage_key)
        if record is None:
            return None
        return await _recover_patch_journal_result(
            record,
            behavior_profile_id=self.behavior_profile_id,
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
            load_operation=load_operation,
            storage_key=storage_key,
            max_file_bytes=self.max_file_bytes,
            protected_entry_names=self.protected_entry_names,
        )

    async def _preflight(
        self,
        workspace: Workspace,
        intents: tuple[_PatchIntent, ...],
    ) -> _PreparedPatch:
        try:
            read_limit = workspace.bounded_read_limit(self.max_file_bytes)
        except Exception as exc:
            raise _PatchPreflightError("workspace_read_bound_unsupported") from exc
        if type(read_limit) is not int or not 0 < read_limit <= self.max_file_bytes:
            raise _PatchPreflightError("workspace_read_bound_invalid")

        prepared: list[_PreparedOperation] = []
        total_source_bytes = 0
        total_output_bytes = 0
        total_diff_bytes = 0
        for intent in intents:
            before: bytes | None = None
            before_revision: str | None = None
            before_sha256: str | None = None
            if intent.source_path is not None and intent.operation_type != "create":
                read = await _read_preflight_source(
                    workspace,
                    intent,
                    max_bytes=read_limit,
                )
                before = read.content
                before_revision = read.revision
                before_sha256 = read.sha256 or hashlib.sha256(before).hexdigest()
                total_source_bytes += len(before)
                if total_source_bytes > self.max_source_bytes:
                    raise _PatchPreflightError(
                        "aggregate_source_too_large",
                        operation_index=intent.index,
                        path=intent.source_path,
                    )

            if intent.operation_type == "create":
                assert intent.content is not None
                after = intent.content.encode("utf-8")
                replacement_count = 0
            elif intent.operation_type == "update":
                assert before is not None and intent.source_path is not None
                after, replacement_count = _apply_exact_edits(before, intent)
            elif intent.operation_type == "move":
                assert before is not None
                after = before
                replacement_count = 0
            else:
                after = None
                replacement_count = 0

            if after is not None and len(after) > self.max_file_bytes:
                raise _PatchPreflightError(
                    "result_file_too_large",
                    operation_index=intent.index,
                    path=intent.destination_path,
                )
            total_output_bytes += len(after) if after is not None else 0
            if total_output_bytes > self.max_output_bytes:
                raise _PatchPreflightError(
                    "aggregate_output_too_large",
                    operation_index=intent.index,
                    path=intent.destination_path,
                )
            after_sha256 = hashlib.sha256(after).hexdigest() if after is not None else None
            diff = _operation_diff(intent, before=before, after=after)
            total_diff_bytes += len(diff.encode("utf-8"))
            if total_diff_bytes > self.max_retained_diff_bytes:
                raise _PatchPreflightError(
                    "aggregate_diff_too_large",
                    operation_index=intent.index,
                    path=intent.destination_path or intent.source_path,
                )
            prepared.append(
                _PreparedOperation(
                    intent=intent,
                    before=before,
                    before_revision=before_revision,
                    before_sha256=before_sha256,
                    after=after,
                    after_sha256=after_sha256,
                    replacement_count=replacement_count,
                    diff=diff,
                )
            )

        for operation in prepared:
            intent = operation.intent
            if intent.operation_type not in {"create", "move"}:
                continue
            assert intent.destination_path is not None
            try:
                await workspace.require_absent(intent.destination_path)
            except FileExistsError as exc:
                raise _PatchPreflightError(
                    "destination_exists",
                    operation_index=intent.index,
                    path=intent.destination_path,
                ) from exc
            except WorkspacePreconditionUnsupportedError as exc:
                raise _PatchPreflightError(
                    "absence_precondition_unsupported",
                    operation_index=intent.index,
                    path=intent.destination_path,
                ) from exc
            except (FileNotFoundError, IsADirectoryError) as exc:
                raise _PatchPreflightError(
                    "destination_not_creatable",
                    operation_index=intent.index,
                    path=intent.destination_path,
                ) from exc

        application_plan = tuple(
            operation.intent.index
            for operation in sorted(
                prepared,
                key=lambda item: (
                    {"move": 0, "create": 1, "update": 2, "delete": 3}[item.intent.operation_type],
                    item.intent.destination_path or item.intent.source_path or "",
                    item.intent.index,
                ),
            )
        )
        intent_identity = [
            {
                "index": item.intent.index,
                "type": item.intent.operation_type,
                "source_path": item.intent.source_path,
                "destination_path": item.intent.destination_path,
                "expected_revision_sha256": (
                    hashlib.sha256(item.intent.expected_revision.encode("utf-8")).hexdigest()
                    if item.intent.expected_revision is not None
                    else None
                ),
                "before_sha256": item.before_sha256,
                "after_sha256": item.after_sha256,
                "replacement_count": item.replacement_count,
            }
            for item in prepared
        ]
        digest = hashlib.sha256(_canonical_json_bytes(intent_identity)).hexdigest()
        counts = {
            operation_type: sum(item.intent.operation_type == operation_type for item in prepared)
            for operation_type in sorted(_PATCH_TYPES)
        }
        return _PreparedPatch(
            patch_id=f"patch_{digest[:32]}",
            operations=tuple(prepared),
            application_plan=application_plan,
            counts=counts,
            total_source_bytes=total_source_bytes,
            total_output_bytes=total_output_bytes,
            aggregate_diff="".join(item.diff for item in prepared),
        )

    async def _apply(
        self,
        ctx: ToolContext,
        observation_workspace: Workspace,
        mutation_workspace: Workspace,
        prepared: _PreparedPatch,
        projection_snapshot: InvocationRedactorSnapshot,
        *,
        journal: _PatchDurableJournal | None,
    ) -> ToolResult:
        by_index = {item.intent.index: item for item in prepared.operations}
        application_order = {
            operation_index: sequence
            for sequence, operation_index in enumerate(prepared.application_plan)
        }
        evidence = [
            _initial_operation_evidence(item, application_order[item.intent.index])
            for item in prepared.operations
        ]
        outcome: PatchOutcome = "applied"
        failure_category: str | None = None
        applied_count = 0

        for operation_index in prepared.application_plan:
            operation = by_index[operation_index]
            try:
                await asyncio.sleep(0)
                if journal is not None:
                    await journal.mark_dispatching(operation_index)
            except asyncio.CancelledError:
                outcome = "cancelled"
                failure_category = "cancelled_between_operations"
                break
            except _PatchJournalError:
                evidence[operation_index]["status"] = "failed"
                outcome = "partial" if applied_count else "failed"
                failure_category = "durable_journal_unavailable"
                break
            try:
                mutation = await _apply_operation(mutation_workspace, operation)
                _settle_success_evidence(evidence[operation_index], operation, mutation)
                applied_count += 1
                if journal is not None:
                    try:
                        await journal.settle(
                            operation_index,
                            "applied",
                            move_fidelity=evidence[operation_index]["move_fidelity"],
                            after_revision=evidence[operation_index]["after_revision"],
                            after_sha256=evidence[operation_index]["after_sha256"],
                            after_bytes=evidence[operation_index]["after_bytes"],
                        )
                    except (asyncio.CancelledError, _PatchJournalError):
                        outcome = "ambiguous"
                        failure_category = "durable_settlement_unavailable"
                        break
            except asyncio.CancelledError:
                status, observations = await _reconcile_operation(
                    observation_workspace,
                    operation,
                    self.max_file_bytes,
                )
                _record_operation_observations(evidence[operation_index], observations)
                _settle_reconciled_evidence(
                    evidence[operation_index],
                    operation,
                    status,
                    observations=observations,
                )
                if status == "applied":
                    applied_count += 1
                outcome = "ambiguous" if status == "unknown" else "cancelled"
                failure_category = "cancelled_during_operation"
                if not await _settle_patch_journal(
                    journal,
                    operation_index,
                    status,
                    evidence=evidence[operation_index],
                ):
                    outcome = "ambiguous"
                    failure_category = "durable_settlement_unavailable"
                break
            except WorkspaceMoveAmbiguousError as exc:
                if exc.result is not None:
                    _settle_move_result(evidence[operation_index], operation, exc.result)
                evidence[operation_index]["status"] = "unknown"
                outcome = "ambiguous"
                failure_category = "move_settlement_ambiguous"
                await _settle_patch_journal(
                    journal,
                    operation_index,
                    "unknown",
                    evidence=evidence[operation_index],
                )
                break
            except (WorkspaceRevisionMismatchError, FileExistsError, FileNotFoundError):
                evidence[operation_index]["status"] = "conflict"
                observations = await _observe_operation_paths(
                    observation_workspace,
                    operation,
                    self.max_file_bytes,
                )
                _record_operation_observations(evidence[operation_index], observations)
                outcome = "partial" if applied_count else "precondition_failed"
                failure_category = "concurrent_precondition_conflict"
                if not await _settle_patch_journal(
                    journal,
                    operation_index,
                    "conflict",
                    evidence=evidence[operation_index],
                ):
                    outcome = "ambiguous"
                    failure_category = "durable_settlement_unavailable"
                break
            except (WorkspaceMoveUnsupportedError, WorkspacePreconditionUnsupportedError):
                evidence[operation_index]["status"] = "failed"
                outcome = "partial" if applied_count else "unsupported"
                failure_category = "workspace_capability_unsupported"
                if not await _settle_patch_journal(
                    journal,
                    operation_index,
                    "failed",
                    evidence=evidence[operation_index],
                ):
                    outcome = "ambiguous"
                    failure_category = "durable_settlement_unavailable"
                break
            except Exception:
                status, observations = await _reconcile_operation(
                    observation_workspace,
                    operation,
                    self.max_file_bytes,
                )
                _record_operation_observations(evidence[operation_index], observations)
                _settle_reconciled_evidence(
                    evidence[operation_index],
                    operation,
                    status,
                    observations=observations,
                )
                if status == "applied":
                    applied_count += 1
                    has_remaining = application_order[operation_index] + 1 < len(
                        prepared.application_plan
                    )
                    outcome = "partial" if has_remaining else "applied"
                elif status == "conflict":
                    outcome = "partial" if applied_count else "precondition_failed"
                elif status == "unknown":
                    outcome = "ambiguous"
                else:
                    outcome = "partial" if applied_count else "failed"
                failure_category = "operation_failed"
                if not await _settle_patch_journal(
                    journal,
                    operation_index,
                    status,
                    evidence=evidence[operation_index],
                ):
                    outcome = "ambiguous"
                    failure_category = "durable_settlement_unavailable"
                break

        # Result projection records the exact invocation-secret revision that
        # bounds the public output. Capture it before terminal journaling seals
        # that scope; the runtime cannot publish the result until run() returns.
        result = await self._result(
            ctx,
            prepared,
            evidence,
            outcome=outcome,
            failure_category=failure_category,
            projection_snapshot=projection_snapshot,
        )
        if journal is not None:
            try:
                await journal.terminal(outcome, failure_category)
            except (asyncio.CancelledError, _PatchJournalError):
                return _terminal_journal_failure_result(
                    prepared,
                    workspace_outcome=outcome,
                )
        return result

    async def _result(
        self,
        ctx: ToolContext,
        prepared: _PreparedPatch,
        evidence: list[dict[str, Any]],
        *,
        outcome: PatchOutcome,
        failure_category: str | None,
        projection_snapshot: InvocationRedactorSnapshot,
    ) -> ToolResult:
        current = _stable_projection_snapshot(ctx, projection_snapshot)
        if current is None:
            return _unstable_projection_result(outcome=outcome, mutated=True)
        redacted = _redact_patch_output_stably(
            ctx,
            evidence,
            prepared.aggregate_diff,
            current,
        )
        if redacted is None:
            return _unstable_projection_result(outcome=outcome, mutated=True)
        safe_projection, current = redacted
        full_safe_operations = safe_projection["operations"]
        safe_diff = safe_projection["diff"]
        result_identities = _patch_result_identities(ctx, current.redactor)
        safe_operations, manifest_truncated, manifest_truncation_reasons = (
            _bounded_operation_manifest(full_safe_operations)
        )
        changed_paths = _changed_path_manifest(safe_operations)
        full_changed_paths = _changed_path_manifest(full_safe_operations)
        diff_preview, diff_truncated, truncation_reasons = _bounded_diff_preview(
            safe_diff,
            prepared.operations,
            redactor=current.redactor,
            max_file_bytes=self.max_file_diff_preview_bytes,
            max_total_bytes=self.max_diff_preview_bytes,
        )
        record_ambiguous_secret_output(ctx, current)
        artifact_status = "not_needed"
        artifact_reference: dict[str, Any] | None = None
        if diff_truncated or manifest_truncated:
            artifact_projection_snapshot = _seal_patch_artifact_projection(ctx, current)
            if artifact_projection_snapshot is None:
                return _unstable_projection_result(outcome=outcome, mutated=True)
            artifact_status, artifact_reference = await _store_patch_artifact(
                ctx,
                patch_id=prepared.patch_id,
                behavior_profile_id=self.behavior_profile_id,
                outcome=outcome,
                operations=full_safe_operations,
                changed_paths=full_changed_paths,
                diff=safe_diff,
                identities=result_identities,
                max_bytes=self.max_artifact_bytes,
                projection_snapshot=artifact_projection_snapshot,
            )
            if artifact_status in {"secret_scope_unstable", "secret_scope_cleanup_failed"}:
                return _unstable_projection_result(outcome=outcome, mutated=True)
        structured: dict[str, Any] = {
            "version": PATCH_RESULT_VERSION,
            "patch_id": prepared.patch_id,
            "outcome": outcome,
            "failure_category": failure_category,
            **result_identities,
            "behavior_profile_id": self.behavior_profile_id,
            "cross_file_atomic": False,
            "operation_count": len(prepared.operations),
            "operation_counts": dict(prepared.counts),
            "application_plan": list(prepared.application_plan),
            "operations": safe_operations,
            "changed_paths": changed_paths,
            "manifest_truncated": manifest_truncated,
            "manifest_truncation_reasons": manifest_truncation_reasons,
            "total_source_bytes": prepared.total_source_bytes,
            "total_output_bytes": prepared.total_output_bytes,
            "diff": diff_preview,
            "diff_truncated": diff_truncated,
            "diff_truncation_reasons": truncation_reasons,
            "artifact_status": artifact_status,
            "artifact": artifact_reference,
            "requires_fresh_read": outcome in {"partial", "ambiguous", "cancelled"},
        }
        if len(_canonical_json_bytes(structured)) > MAX_PATCH_RESULT_BYTES:
            if artifact_status == "not_needed":
                artifact_projection_snapshot = _seal_patch_artifact_projection(ctx, current)
                if artifact_projection_snapshot is None:
                    return _unstable_projection_result(outcome=outcome, mutated=True)
                artifact_status, artifact_reference = await _store_patch_artifact(
                    ctx,
                    patch_id=prepared.patch_id,
                    behavior_profile_id=self.behavior_profile_id,
                    outcome=outcome,
                    operations=full_safe_operations,
                    changed_paths=full_changed_paths,
                    diff=safe_diff,
                    identities=result_identities,
                    max_bytes=self.max_artifact_bytes,
                    projection_snapshot=artifact_projection_snapshot,
                )
                if artifact_status in {
                    "secret_scope_unstable",
                    "secret_scope_cleanup_failed",
                }:
                    return _unstable_projection_result(outcome=outcome, mutated=True)
                structured["artifact_status"] = artifact_status
                structured["artifact"] = artifact_reference
            structured["operations"] = _compact_operation_manifest(safe_operations)
            structured["changed_paths"] = _compact_changed_path_manifest(changed_paths)
            structured["manifest_truncated"] = True
            structured["manifest_truncation_reasons"] = sorted(
                {*manifest_truncation_reasons, "result_bytes"}
            )
        if len(_canonical_json_bytes(structured)) > MAX_PATCH_RESULT_BYTES:
            return ToolResult(
                content=(
                    "Patch outcome publication exceeded its fail-closed result bound. "
                    "Inspect fresh workspace evidence before continuing."
                ),
                structured={
                    "error": "patch_result_bound_exceeded",
                    "patch_id": prepared.patch_id,
                    "workspace_outcome": outcome,
                    "workspace_may_have_changed": True,
                    "requires_fresh_read": True,
                    "artifact_status": structured["artifact_status"],
                    "artifact": structured["artifact"],
                },
                is_error=True,
            )
        summary = (
            f"Patch {prepared.patch_id} {outcome}: "
            f"{sum(item['status'] == 'applied' for item in safe_operations)}/"
            f"{len(safe_operations)} operations applied."
        )
        if outcome in {"partial", "ambiguous", "cancelled"}:
            summary += " Re-read every affected path before proposing a repair."
        content = f"{summary}\n\n{diff_preview}" if diff_preview else summary
        return ToolResult(
            content=content,
            structured=structured,
            is_error=outcome != "applied",
        )


def _seal_patch_artifact_projection(
    ctx: ToolContext,
    snapshot: InvocationRedactorSnapshot,
) -> InvocationRedactorSnapshot | None:
    """Freeze a dynamic secret scope before durable artifact publication."""

    if ctx.invocation_secret_snapshot_provider is None:
        # Direct callers without revision evidence retain the historical static
        # redactor contract. Runtime contexts always provide exact revisions.
        return snapshot
    authority = _runtime_tool_invocation_authority(ctx)
    if authority is None:
        return None
    try:
        publication = authority.secret_publication_sealer()
        current = active_secret_redactor_snapshot(ctx)
    except Exception:
        return None
    publication_redactor = getattr(publication, "redactor", None)
    if (
        getattr(publication, "unsafe_output", True) is not False
        or getattr(publication, "secret_scope_incomplete", True) is not False
        or not isinstance(publication_redactor, SecretRedactor)
        or current.revision != snapshot.revision
        or not current.redactor.has_same_registry(snapshot.redactor)
        or not current.redactor.has_same_registry(publication_redactor)
    ):
        return None
    return current


async def _start_patch_journal(
    ctx: ToolContext,
    prepared: _PreparedPatch,
    *,
    behavior_profile_id: str,
) -> _PatchDurableJournal | None:
    authority = _runtime_tool_invocation_authority(ctx)
    if authority is None:
        return None
    if authority.tool_name != "apply_patch" or ctx.idempotency_key != authority.idempotency_key:
        raise _PatchJournalError("Patch durable authority does not match the invocation.")
    storage_key = _patch_journal_key(ctx.session_id, authority.idempotency_key)
    operation_order = {
        operation_index: sequence
        for sequence, operation_index in enumerate(prepared.application_plan)
    }
    record: dict[str, Any] = {
        "record_type": _PATCH_JOURNAL_RECORD_TYPE,
        "schema_version": _PATCH_JOURNAL_SCHEMA_VERSION,
        "state": "prepared",
        "parent_session_id": ctx.session_id,
        "parent_run_epoch": authority.parent_run_epoch,
        "environment_name": ctx.environment_name,
        "environment_allocation_fingerprint": authority.environment_allocation_fingerprint,
        "model_step_id": authority.model_step_id,
        "model_attempt_id": authority.model_attempt_id,
        "tool_round_id": authority.tool_round_id,
        "tool_call_id": authority.tool_call_id,
        "idempotency_key": authority.idempotency_key,
        "execution_profile_fingerprint": authority.execution_profile_fingerprint,
        "effective_arguments_sha256": authority.effective_arguments_sha256,
        "behavior_profile_id": behavior_profile_id,
        "workspace_id_sha256": _optional_identity_sha256(ctx.workspace_id),
        "patch_id": prepared.patch_id,
        "cross_file_atomic": False,
        "application_plan": list(prepared.application_plan),
        "active_operation": None,
        "terminal_outcome": None,
        "failure_category": None,
        "operations": [
            _patch_journal_operation(
                item,
                application_order=operation_order[item.intent.index],
                patch_id=prepared.patch_id,
            )
            for item in prepared.operations
        ],
    }
    existing = await authority.load_durable_operation(storage_key)
    if existing is not None:
        raise _PatchJournalError("A durable record already exists for this patch invocation.")
    try:
        persisted = await authority.compare_and_set_durable_operation(
            storage_key,
            None,
            record,
            {},
        )
    except Exception as exc:
        persisted = await authority.load_durable_operation(storage_key)
        if persisted != record:
            raise _PatchJournalError("Patch journal intent publication failed.") from exc
    if persisted != record:
        raise _PatchJournalError("Patch journal intent publication returned conflicting evidence.")
    return _PatchDurableJournal(authority, storage_key, record)


def _patch_journal_operation(
    operation: _PreparedOperation,
    *,
    application_order: int,
    patch_id: str,
) -> dict[str, Any]:
    intent = operation.intent
    return {
        "index": intent.index,
        "application_order": application_order,
        "type": intent.operation_type,
        "source_path_identity": _salted_identity_sha256(patch_id, intent.source_path),
        "destination_path_identity": _salted_identity_sha256(
            patch_id,
            intent.destination_path,
        ),
        "expected_revision_sha256": _optional_identity_sha256(intent.expected_revision),
        "before_revision_sha256": _optional_identity_sha256(operation.before_revision),
        "after_revision_sha256": None,
        "before_sha256": operation.before_sha256,
        "after_sha256": None,
        "projected_after_sha256": operation.after_sha256,
        "before_bytes": len(operation.before) if operation.before is not None else None,
        "after_bytes": None,
        "projected_after_bytes": len(operation.after) if operation.after is not None else None,
        "replacement_count": operation.replacement_count,
        "status": "not_started",
        "move_fidelity": None,
        "observed_source_state": None,
        "observed_source_revision_sha256": None,
        "observed_source_sha256": None,
        "observed_source_bytes": None,
        "observed_destination_state": None,
        "observed_destination_revision_sha256": None,
        "observed_destination_sha256": None,
        "observed_destination_bytes": None,
    }


def _journal_operation(record: dict[str, Any], operation_index: int) -> dict[str, Any]:
    operations = record.get("operations")
    if type(operations) is not list:
        raise _PatchJournalError("Patch journal operations are invalid.")
    for operation in operations:
        if type(operation) is dict and operation.get("index") == operation_index:
            return operation
    raise _PatchJournalError("Patch journal operation is missing.")


async def _settle_patch_journal(
    journal: _PatchDurableJournal | None,
    operation_index: int,
    status: PatchOperationStatus,
    *,
    evidence: Mapping[str, Any] | None = None,
) -> bool:
    if journal is None:
        return True
    try:
        await journal.settle(
            operation_index,
            status,
            move_fidelity=(None if evidence is None else evidence.get("move_fidelity")),
            after_revision=(None if evidence is None else evidence.get("after_revision")),
            after_sha256=(None if evidence is None else evidence.get("after_sha256")),
            after_bytes=(None if evidence is None else evidence.get("after_bytes")),
        )
    except (asyncio.CancelledError, _PatchJournalError):
        return False
    return True


def _patch_journal_key(parent_session_id: str, idempotency_key: str) -> str:
    digest = hashlib.sha256(
        b"cayu-apply-patch-journal-v2\0"
        + parent_session_id.encode("utf-8")
        + b"\0"
        + idempotency_key.encode("utf-8")
    ).hexdigest()
    return f"apply-patch:v2:{digest}"


def _copy_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    copied = json.loads(_canonical_json_bytes(value))
    if type(copied) is not dict:  # pragma: no cover - input is a mapping
        raise _PatchJournalError("Patch journal copy is invalid.")
    return copied


def _optional_identity_sha256(value: str | None) -> str | None:
    return None if value is None else hashlib.sha256(value.encode("utf-8")).hexdigest()


def _salted_identity_sha256(salt: str, value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(f"{salt}\0{value}".encode()).hexdigest()


def _journal_failure_result(*, mutated: bool) -> ToolResult:
    return ToolResult(
        content=(
            "Patch execution stopped because its durable operation boundary could not be "
            "published. No patch mutation was dispatched."
            if not mutated
            else "Patch settlement could not be published durably; inspect fresh workspace state."
        ),
        structured={
            "outcome": "failed" if not mutated else "ambiguous",
            "failure_category": "durable_journal_unavailable",
            "workspace_may_have_changed": mutated,
            "requires_fresh_read": mutated,
        },
        is_error=True,
    )


def _terminal_journal_failure_result(
    prepared: _PreparedPatch,
    *,
    workspace_outcome: PatchOutcome,
) -> ToolResult:
    return ToolResult(
        content=(
            "Patch terminal settlement could not be published durably. Re-read every "
            "affected path before proposing a repair."
        ),
        structured={
            "version": PATCH_RESULT_VERSION,
            "patch_id": prepared.patch_id,
            "outcome": "ambiguous",
            "workspace_outcome": workspace_outcome,
            "failure_category": "durable_terminal_unavailable",
            "workspace_may_have_changed": True,
            "requires_fresh_read": True,
        },
        is_error=True,
    )


async def _recover_patch_journal_result(
    raw_record: Any,
    *,
    behavior_profile_id: str,
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
    load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
    storage_key: str,
    max_file_bytes: int,
    protected_entry_names: tuple[str, ...],
) -> ToolResult:
    if type(raw_record) is not dict:
        return _patch_recovery_refusal("durable_journal_invalid")
    try:
        record = _copy_json_object(raw_record)
    except (TypeError, ValueError, UnicodeError):
        return _patch_recovery_refusal("durable_journal_invalid")
    expected_identity = {
        "record_type": _PATCH_JOURNAL_RECORD_TYPE,
        "schema_version": _PATCH_JOURNAL_SCHEMA_VERSION,
        "parent_session_id": parent_session_id,
        "parent_run_epoch": parent_run_epoch,
        "environment_name": environment_name,
        "model_step_id": model_step_id,
        "model_attempt_id": model_attempt_id,
        "tool_round_id": tool_round_id,
        "tool_call_id": tool_call_id,
        "idempotency_key": idempotency_key,
        "behavior_profile_id": behavior_profile_id,
    }
    for field_name, expected in expected_identity.items():
        if record.get(field_name) != expected:
            category = (
                "execution_profile_drift"
                if field_name == "behavior_profile_id"
                else "durable_journal_identity_mismatch"
            )
            return _patch_recovery_refusal(category)
    if (
        execution_profile_fingerprint is None
        or record.get("execution_profile_fingerprint") != execution_profile_fingerprint
        or record.get("environment_allocation_fingerprint") != environment_allocation_fingerprint
    ):
        return _patch_recovery_refusal("execution_profile_drift")
    try:
        argument_digest = hashlib.sha256(
            canonical_durable_json_bytes(arguments, "apply_patch_recovery.arguments")
        ).hexdigest()
    except (TypeError, ValueError, UnicodeError):
        return _patch_recovery_refusal("durable_journal_argument_mismatch")
    if record.get("effective_arguments_sha256") != argument_digest:
        return _patch_recovery_refusal("durable_journal_argument_mismatch")

    raw_operations = arguments.get("operations")
    if type(raw_operations) is not list:
        return _patch_recovery_refusal("durable_journal_argument_mismatch")
    try:
        intents = _validate_patch_intents(
            raw_operations,
            protected_entry_names=protected_entry_names,
        )
    except (TypeError, ValueError, _PatchPreflightError):
        return _patch_recovery_refusal("durable_journal_argument_mismatch")

    operations = _validated_recovery_operations(record)
    if operations is None:
        return _patch_recovery_refusal("durable_journal_invalid")
    application_plan = [
        operation["index"]
        for operation in sorted(operations, key=lambda operation: operation["application_order"])
    ]
    patch_id = record.get("patch_id")
    workspace_id_sha256 = record.get("workspace_id_sha256")
    if (
        type(patch_id) is not str
        or len(patch_id) != 38
        or not patch_id.startswith("patch_")
        or not _is_sha256_prefix(patch_id.removeprefix("patch_"))
        or record.get("application_plan") != application_plan
        or not _is_optional_sha256(workspace_id_sha256)
        or record.get("state") not in {"prepared", "applying", "in_progress", "terminal"}
    ):
        return _patch_recovery_refusal("durable_journal_invalid")
    if not _recovery_intents_match_journal(intents, operations, patch_id=patch_id):
        return _patch_recovery_refusal("durable_journal_argument_mismatch")

    unknown_operations = [operation for operation in operations if operation["status"] == "unknown"]
    active_operation = record.get("active_operation")
    if (
        len(unknown_operations) > 1
        or (record.get("state") == "applying" and len(unknown_operations) != 1)
        or (active_operation is not None and record.get("state") != "applying")
        or (
            record.get("state") == "applying" and active_operation != unknown_operations[0]["index"]
        )
    ):
        return _patch_recovery_refusal("durable_journal_invalid")
    if unknown_operations:
        reconciled = await _reconcile_unknown_recovery_operation(
            record,
            operations=operations,
            intents=intents,
            recovery_authority=recovery_authority,
            load_operation=load_operation,
            storage_key=storage_key,
            max_file_bytes=max_file_bytes,
            expected_workspace_id_sha256=workspace_id_sha256,
        )
        if isinstance(reconciled, ToolResult):
            return reconciled
        record = reconciled
        operations = _validated_recovery_operations(record)
        if operations is None:
            return _patch_recovery_refusal("durable_journal_invalid")

    statuses = [operation["status"] for operation in operations]
    stored_outcome = record.get("terminal_outcome")
    if (
        record.get("state") == "terminal"
        and "unknown" not in statuses
        and stored_outcome
        in {
            "applied",
            "precondition_failed",
            "partial",
            "ambiguous",
            "unsupported",
            "cancelled",
            "failed",
        }
    ):
        outcome = stored_outcome
    elif "unknown" in statuses or record.get("state") == "applying":
        outcome = "ambiguous"
    elif statuses and all(status == "applied" for status in statuses):
        outcome = "applied"
    elif "applied" in statuses:
        outcome = "partial"
    elif "conflict" in statuses:
        outcome = "precondition_failed"
    elif "failed" in statuses:
        outcome = "failed"
    else:
        outcome = "cancelled"
    requires_fresh_read = outcome != "applied"
    failure_category = _safe_recovery_failure_category(record.get("failure_category"))
    if failure_category is None and outcome != "applied":
        failure_category = "recovered_after_worker_loss"
    structured = {
        "version": PATCH_RESULT_VERSION,
        "recovered": True,
        "patch_id": patch_id,
        "outcome": outcome,
        "failure_category": failure_category,
        "session_id": parent_session_id,
        "run_epoch": parent_run_epoch,
        "model_step_id": model_step_id,
        "model_attempt_id": model_attempt_id,
        "tool_round_id": tool_round_id,
        "tool_call_id": tool_call_id,
        "tool_call_identity": idempotency_key,
        "workspace_id_sha256": workspace_id_sha256,
        "behavior_profile_id": behavior_profile_id,
        "execution_profile_fingerprint": execution_profile_fingerprint,
        "cross_file_atomic": False,
        "application_plan": application_plan,
        "operations": operations,
        "diff": "",
        "diff_unavailable": True,
        "artifact_status": "unavailable_during_recovery",
        "artifact": None,
        "requires_fresh_read": requires_fresh_read,
    }
    content = (
        f"Recovered patch {patch_id} as {outcome} from its durable operation "
        "boundary; the patch was not replayed."
    )
    if requires_fresh_read:
        content += " Re-read every affected path before proposing a repair."
    return ToolResult(content=content, structured=structured, is_error=outcome != "applied")


def _recovery_intents_match_journal(
    intents: tuple[_PatchIntent, ...],
    operations: list[dict[str, Any]],
    *,
    patch_id: str,
) -> bool:
    if len(intents) != len(operations):
        return False
    for intent, operation in zip(intents, operations, strict=True):
        if (
            intent.index != operation["index"]
            or intent.operation_type != operation["type"]
            or _salted_identity_sha256(patch_id, intent.source_path)
            != operation["source_path_identity"]
            or _salted_identity_sha256(patch_id, intent.destination_path)
            != operation["destination_path_identity"]
            or _optional_identity_sha256(intent.expected_revision)
            != operation["expected_revision_sha256"]
        ):
            return False
    return True


async def _reconcile_unknown_recovery_operation(
    record: dict[str, Any],
    *,
    operations: list[dict[str, Any]],
    intents: tuple[_PatchIntent, ...],
    recovery_authority: DurableToolRecoveryAuthority | None,
    load_operation: Callable[[str], Awaitable[dict[str, Any] | None]],
    storage_key: str,
    max_file_bytes: int,
    expected_workspace_id_sha256: str | None,
) -> dict[str, Any] | ToolResult:
    if recovery_authority is None or not isinstance(recovery_authority.workspace, Workspace):
        return _patch_recovery_refusal("workspace_recovery_capability_unavailable")
    workspace = recovery_authority.workspace
    if (
        recovery_authority.environment_name != record.get("environment_name")
        or _optional_identity_sha256(workspace.id) != expected_workspace_id_sha256
    ):
        return _patch_recovery_refusal("workspace_recovery_identity_mismatch")

    unknown = next(operation for operation in operations if operation["status"] == "unknown")
    intent = intents[unknown["index"]]
    observations = await _observe_recovery_intent(
        workspace,
        intent,
        max_file_bytes=max_file_bytes,
    )
    status = _classify_recovery_observations(intent, unknown, observations)
    desired = _copy_json_object(record)
    desired_operation = _journal_operation(desired, intent.index)
    desired_operation["status"] = status
    _record_recovery_journal_observations(desired_operation, observations)
    if status == "applied":
        _settle_recovered_after_evidence(desired_operation, intent, observations)
    desired["active_operation"] = None
    desired["state"] = "terminal"
    desired_statuses = [
        operation.get("status")
        for operation in desired.get("operations", [])
        if type(operation) is dict
    ]
    desired_outcome = _recovery_outcome(desired_statuses)
    desired["terminal_outcome"] = desired_outcome
    desired["failure_category"] = (
        None if desired_outcome == "applied" else "recovered_after_worker_loss"
    )
    try:
        persisted = await recovery_authority.compare_and_set_operation(
            storage_key,
            record,
            desired,
            {},
        )
    except Exception:
        persisted = await load_operation(storage_key)
        if persisted != desired:
            return _patch_recovery_refusal("durable_journal_reconciliation_conflict")
    if persisted != desired:
        return _patch_recovery_refusal("durable_journal_reconciliation_conflict")
    return desired


async def _observe_recovery_intent(
    workspace: Workspace,
    intent: _PatchIntent,
    *,
    max_file_bytes: int,
) -> dict[str, tuple[str, str | None, str | None, int | None]]:
    if intent.operation_type == "move":
        assert intent.source_path is not None and intent.destination_path is not None
        source, destination = await asyncio.gather(
            _observe_path(workspace, intent.source_path, max_file_bytes),
            _observe_path(workspace, intent.destination_path, max_file_bytes),
        )
        return {"source": source, "destination": destination}
    path = intent.destination_path or intent.source_path
    assert path is not None
    role = "destination" if intent.operation_type == "create" else "source"
    return {role: await _observe_path(workspace, path, max_file_bytes)}


def _classify_recovery_observations(
    intent: _PatchIntent,
    operation: Mapping[str, Any],
    observations: Mapping[str, tuple[str, str | None, str | None, int | None]],
) -> PatchOperationStatus:
    before_sha256 = operation.get("before_sha256")
    after_sha256 = operation.get("projected_after_sha256")
    if intent.operation_type == "move":
        source = observations["source"]
        destination = observations["destination"]
        if source[0] == "absent" and _observation_matches(destination, after_sha256):
            return "applied"
        if _observation_matches(source, before_sha256) and destination[0] == "absent":
            return "not_started"
        if "unknown" in {source[0], destination[0]}:
            return "unknown"
        return "conflict"

    role = "destination" if intent.operation_type == "create" else "source"
    observed = observations[role]
    if intent.operation_type == "create":
        if _observation_matches(observed, after_sha256):
            return "applied"
        if observed[0] == "absent":
            return "not_started"
    elif intent.operation_type == "delete":
        if observed[0] == "absent":
            return "applied"
        if _observation_matches(observed, before_sha256):
            return "not_started"
    else:
        if _observation_matches(observed, after_sha256):
            return "applied"
        if _observation_matches(observed, before_sha256):
            return "not_started"
    return "unknown" if observed[0] == "unknown" else "conflict"


def _record_recovery_journal_observations(
    operation: dict[str, Any],
    observations: Mapping[str, tuple[str, str | None, str | None, int | None]],
) -> None:
    for role, (state, revision, digest, size) in observations.items():
        operation[f"observed_{role}_state"] = state
        operation[f"observed_{role}_revision_sha256"] = _optional_identity_sha256(revision)
        operation[f"observed_{role}_sha256"] = digest
        operation[f"observed_{role}_bytes"] = size


def _settle_recovered_after_evidence(
    operation: dict[str, Any],
    intent: _PatchIntent,
    observations: Mapping[str, tuple[str, str | None, str | None, int | None]],
) -> None:
    if intent.operation_type == "delete":
        return
    role = "destination" if intent.operation_type in {"create", "move"} else "source"
    state, revision, digest, size = observations[role]
    if state != "present" or revision is None or digest is None or size is None:
        raise _PatchJournalError("Recovered applied operation lacks complete after evidence.")
    operation["after_revision_sha256"] = _optional_identity_sha256(revision)
    operation["after_sha256"] = digest
    operation["after_bytes"] = size


def _recovery_outcome(statuses: list[object]) -> PatchOutcome:
    if statuses and all(status == "applied" for status in statuses):
        return "applied"
    if "unknown" in statuses:
        return "ambiguous"
    if "applied" in statuses:
        return "partial"
    if "conflict" in statuses:
        return "precondition_failed"
    if "failed" in statuses:
        return "failed"
    return "cancelled"


def _validated_recovery_operations(record: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    raw = record.get("operations")
    if type(raw) is not list or not 1 <= len(raw) <= MAX_PATCH_OPERATIONS:
        return None
    operations: list[dict[str, Any]] = []
    indices: set[int] = set()
    application_orders: set[int] = set()
    for item in raw:
        if type(item) is not dict:
            return None
        index = item.get("index")
        status = item.get("status")
        operation_type = item.get("type")
        application_order = item.get("application_order")
        if (
            type(index) is not int
            or index < 0
            or index in indices
            or type(application_order) is not int
            or not 0 <= application_order < len(raw)
            or application_order in application_orders
            or status not in {"applied", "not_started", "conflict", "failed", "unknown"}
            or operation_type not in _PATCH_TYPES
        ):
            return None
        digest_fields = (
            "source_path_identity",
            "destination_path_identity",
            "expected_revision_sha256",
            "before_revision_sha256",
            "after_revision_sha256",
            "before_sha256",
            "after_sha256",
            "projected_after_sha256",
            "observed_source_revision_sha256",
            "observed_source_sha256",
            "observed_destination_revision_sha256",
            "observed_destination_sha256",
        )
        if any(not _is_optional_sha256(item.get(field_name)) for field_name in digest_fields):
            return None
        byte_fields = (
            "before_bytes",
            "after_bytes",
            "projected_after_bytes",
            "observed_source_bytes",
            "observed_destination_bytes",
        )
        if any(
            not _is_optional_nonnegative_int(item.get(field_name)) for field_name in byte_fields
        ):
            return None
        replacement_count = item.get("replacement_count")
        move_fidelity = item.get("move_fidelity")
        observation_state_fields = (
            "observed_source_state",
            "observed_destination_state",
        )
        if (
            type(replacement_count) is not int
            or replacement_count < 0
            or replacement_count > MAX_PATCH_REPLACEMENTS
            or move_fidelity not in {None, "atomic_rename", "link_unlink"}
            or any(
                item.get(field_name) not in {None, "absent", "present", "unknown"}
                for field_name in observation_state_fields
            )
        ):
            return None
        indices.add(index)
        application_orders.add(application_order)
        operations.append(
            {
                "index": index,
                "application_order": application_order,
                "type": operation_type,
                **{field_name: item.get(field_name) for field_name in digest_fields},
                **{field_name: item.get(field_name) for field_name in byte_fields},
                **{field_name: item.get(field_name) for field_name in observation_state_fields},
                "replacement_count": replacement_count,
                "status": status,
                "move_fidelity": move_fidelity,
            }
        )
    operations.sort(key=lambda item: item["index"])
    if [item["index"] for item in operations] != list(range(len(operations))):
        return None
    return operations


def _is_optional_sha256(value: object) -> bool:
    return value is None or (type(value) is str and _is_lower_hex(value, length=64))


def _is_sha256_prefix(value: str) -> bool:
    return _is_lower_hex(value, length=32)


def _is_lower_hex(value: str, *, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _is_optional_nonnegative_int(value: object) -> bool:
    return value is None or (type(value) is int and value >= 0)


def _safe_recovery_failure_category(value: object) -> str | None:
    if (
        type(value) is str
        and 0 < len(value) <= 128
        and all(
            character.isascii() and (character.isalnum() or character == "_") for character in value
        )
    ):
        return value
    return None


def _patch_recovery_refusal(category: str) -> ToolResult:
    return ToolResult(
        content=(
            "Patch recovery refused because its durable identity or execution profile no "
            "longer matches. The patch was not replayed."
        ),
        structured={
            "outcome": "unsupported",
            "failure_category": category,
            "recovered": True,
            "replayed": False,
            "requires_fresh_read": True,
        },
        is_error=True,
    )


def _validate_patch_intents(
    raw_operations: list[object],
    *,
    protected_entry_names: tuple[str, ...],
) -> tuple[_PatchIntent, ...]:
    intents: list[_PatchIntent] = []
    logical_paths: dict[str, tuple[int, str]] = {}
    total_edits = 0
    total_paths = 0
    for index, item in enumerate(raw_operations):
        if type(item) is not dict:
            raise _PatchPreflightError("operation_not_object", operation_index=index)
        operation = cast("dict[str, object]", item)
        if set(operation) - _OPERATION_FIELDS:
            raise _PatchPreflightError("unknown_operation_fields", operation_index=index)
        operation_type = operation.get("type")
        if type(operation_type) is not str or operation_type not in _PATCH_TYPES:
            raise _PatchPreflightError("unknown_operation_type", operation_index=index)
        operation_type = cast("PatchOperationType", operation_type)

        expected_fields: set[str]
        if operation_type == "create":
            expected_fields = {"type", "path", "content"}
        elif operation_type == "update":
            expected_fields = {"type", "path", "expected_revision", "edits"}
        elif operation_type == "delete":
            expected_fields = {"type", "path", "expected_revision"}
        else:
            expected_fields = {"type", "from_path", "to_path", "expected_revision"}
        if set(operation) != expected_fields:
            raise _PatchPreflightError("invalid_operation_shape", operation_index=index)

        if operation_type == "move":
            source_path = _validate_patch_path(
                operation.get("from_path"),
                index=index,
                protected_entry_names=protected_entry_names,
            )
            destination_path = _validate_patch_path(
                operation.get("to_path"),
                index=index,
                protected_entry_names=protected_entry_names,
            )
        else:
            path = _validate_patch_path(
                operation.get("path"),
                index=index,
                protected_entry_names=protected_entry_names,
            )
            source_path = None if operation_type == "create" else path
            destination_path = None if operation_type == "delete" else path

        expected_revision = (
            _validate_patch_revision(operation.get("expected_revision"), index=index)
            if operation_type != "create"
            else None
        )
        content = (
            _validate_patch_text(
                operation.get("content"),
                category="invalid_create_content",
                index=index,
                allow_blank=True,
            )
            if operation_type == "create"
            else None
        )
        edits = (
            _validate_patch_edits(operation.get("edits"), index=index)
            if operation_type == "update"
            else ()
        )
        total_edits += len(edits)
        if total_edits > MAX_PATCH_EDITS:
            raise _PatchPreflightError("too_many_edits", operation_index=index)

        paths = (
            (source_path, destination_path)
            if operation_type == "move"
            else (source_path or destination_path,)
        )
        for path in paths:
            assert path is not None
            total_paths += 1
            if total_paths > MAX_PATCH_PATHS:
                raise _PatchPreflightError("too_many_paths", operation_index=index)
            logical_key = _logical_path_key(path)
            previous = logical_paths.get(logical_key)
            if previous is not None:
                category = "move_chain_or_path_collision"
                raise _PatchPreflightError(category, operation_index=index, path=path)
            logical_paths[logical_key] = (index, path)

        intents.append(
            _PatchIntent(
                index=index,
                operation_type=operation_type,
                source_path=source_path,
                destination_path=destination_path,
                expected_revision=expected_revision,
                content=content,
                edits=edits,
            )
        )
    return tuple(intents)


def _validate_patch_path(
    value: object,
    *,
    index: int,
    protected_entry_names: tuple[str, ...],
) -> str:
    if type(value) is not str:
        raise _PatchPreflightError("invalid_path", operation_index=index)
    try:
        path = require_unicode_scalar_text(require_nonblank(value, "path"), "path")
    except ValueError as exc:
        raise _PatchPreflightError("invalid_path", operation_index=index) from exc
    if "\0" in path or "\\" in path:
        raise _PatchPreflightError("invalid_path", operation_index=index)
    if posixpath.isabs(path):
        raise _PatchPreflightError("absolute_path", operation_index=index)
    raw_parts = tuple(part for part in path.split("/") if part not in {"", "."})
    if (
        raw_parts
        and len(raw_parts[0]) == 2
        and raw_parts[0][0].isalpha()
        and raw_parts[0][1] == ":"
    ):
        raise _PatchPreflightError("absolute_path", operation_index=index)
    if ".." in raw_parts:
        raise _PatchPreflightError("path_traversal", operation_index=index)
    normalized = posixpath.normpath(path)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise _PatchPreflightError("invalid_path", operation_index=index)
    if len(normalized.encode("utf-8")) > MAX_PATCH_PATH_BYTES:
        raise _PatchPreflightError("path_too_large", operation_index=index)
    protected = {entry.casefold() for entry in protected_entry_names}
    if any(part.rstrip(" .").casefold() in protected for part in normalized.split("/")):
        raise _PatchPreflightError(
            "protected_path",
            operation_index=index,
            path=normalized,
        )
    return normalized


def _logical_path_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _validate_patch_revision(value: object, *, index: int) -> str:
    if type(value) is not str:
        raise _PatchPreflightError("invalid_revision", operation_index=index)
    try:
        revision = require_unicode_scalar_text(
            require_nonblank(value, "expected_revision"),
            "expected_revision",
        )
    except ValueError as exc:
        raise _PatchPreflightError("invalid_revision", operation_index=index) from exc
    if "\0" in revision or len(revision) > 4_096:
        raise _PatchPreflightError("invalid_revision", operation_index=index)
    return revision


def _validate_patch_text(
    value: object,
    *,
    category: str,
    index: int,
    allow_blank: bool,
) -> str:
    if type(value) is not str or (not allow_blank and not value):
        raise _PatchPreflightError(category, operation_index=index)
    try:
        text = require_unicode_scalar_text(value, category)
    except ValueError as exc:
        raise _PatchPreflightError(category, operation_index=index) from exc
    if "\0" in text:
        raise _PatchPreflightError(category, operation_index=index)
    return text


def _validate_patch_edits(value: object, *, index: int) -> tuple[_PatchEdit, ...]:
    if not isinstance(value, list) or not value or len(value) > MAX_PATCH_EDITS:
        raise _PatchPreflightError("invalid_edits", operation_index=index)
    edits: list[_PatchEdit] = []
    for item in value:
        if type(item) is not dict:
            raise _PatchPreflightError("invalid_edit_shape", operation_index=index)
        edit = cast("dict[str, object]", item)
        if set(edit) - _EDIT_FIELDS:
            raise _PatchPreflightError("invalid_edit_shape", operation_index=index)
        if not {"old_text", "new_text"}.issubset(edit):
            raise _PatchPreflightError("invalid_edit_shape", operation_index=index)
        old_text = _validate_patch_text(
            edit.get("old_text"),
            category="invalid_edit_text",
            index=index,
            allow_blank=False,
        )
        new_text = _validate_patch_text(
            edit.get("new_text"),
            category="invalid_edit_text",
            index=index,
            allow_blank=True,
        )
        if old_text == new_text:
            raise _PatchPreflightError("edit_does_not_change_text", operation_index=index)
        expected = edit.get("expected_replacements", 1)
        if type(expected) is not int or not 1 <= expected <= MAX_PATCH_REPLACEMENTS_PER_EDIT:
            raise _PatchPreflightError(
                "invalid_expected_replacements",
                operation_index=index,
            )
        edits.append(
            _PatchEdit(
                old_text=old_text,
                new_text=new_text,
                expected_replacements=expected,
            )
        )
    return tuple(edits)


async def _read_preflight_source(
    workspace: Workspace,
    intent: _PatchIntent,
    *,
    max_bytes: int,
) -> WorkspaceReadResult:
    assert intent.source_path is not None and intent.expected_revision is not None
    try:
        read = await workspace.read_bytes(intent.source_path, max_bytes=max_bytes)
    except FileNotFoundError as exc:
        raise _PatchPreflightError(
            "source_not_found",
            operation_index=intent.index,
            path=intent.source_path,
        ) from exc
    if read.truncated:
        raise _PatchPreflightError(
            "source_too_large",
            operation_index=intent.index,
            path=intent.source_path,
        )
    if read.offset != 0 or read.revision is None:
        raise _PatchPreflightError(
            "source_revision_unavailable",
            operation_index=intent.index,
            path=intent.source_path,
        )
    if read.revision != intent.expected_revision:
        raise _PatchPreflightError(
            "stale_source_revision",
            operation_index=intent.index,
            path=intent.source_path,
        )
    try:
        read.content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _PatchPreflightError(
            "source_not_utf8",
            operation_index=intent.index,
            path=intent.source_path,
        ) from exc
    return read


def _apply_exact_edits(before: bytes, intent: _PatchIntent) -> tuple[bytes, int]:
    assert intent.source_path is not None
    original = before.decode("utf-8")
    replacements: list[tuple[int, int, str, int]] = []
    replacement_count = 0
    for edit_index, edit in enumerate(intent.edits):
        starts = _exact_match_offsets(original, edit.old_text)
        if len(starts) != edit.expected_replacements:
            raise _PatchPreflightError(
                "replacement_count_mismatch",
                operation_index=intent.index,
                path=intent.source_path,
            )
        replacements.extend(
            (start, start + len(edit.old_text), edit.new_text, edit_index) for start in starts
        )
        replacement_count += len(starts)
    replacements.sort()
    for previous, current in pairwise(replacements):
        if current[0] < previous[1]:
            raise _PatchPreflightError(
                "overlapping_edits",
                operation_index=intent.index,
                path=intent.source_path,
            )
    parts: list[str] = []
    cursor = 0
    for start, end, replacement, _edit_index in replacements:
        parts.extend((original[cursor:start], replacement))
        cursor = end
    parts.append(original[cursor:])
    return "".join(parts).encode("utf-8"), replacement_count


def _exact_match_offsets(text: str, needle: str) -> list[int]:
    offsets: list[int] = []
    cursor = 0
    while True:
        offset = text.find(needle, cursor)
        if offset < 0:
            return offsets
        offsets.append(offset)
        cursor = offset + len(needle)


def _operation_diff(
    intent: _PatchIntent,
    *,
    before: bytes | None,
    after: bytes | None,
) -> str:
    if intent.operation_type == "move":
        assert intent.source_path is not None and intent.destination_path is not None
        return (
            f"diff --git a/{intent.source_path} b/{intent.destination_path}\n"
            "similarity index 100%\n"
            f"rename from {intent.source_path}\n"
            f"rename to {intent.destination_path}\n"
        )
    before_text = before.decode("utf-8") if before is not None else ""
    after_text = after.decode("utf-8") if after is not None else ""
    path = intent.destination_path or intent.source_path
    assert path is not None
    fromfile = "/dev/null" if before is None else f"a/{path}"
    tofile = "/dev/null" if after is None else f"b/{path}"
    body = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


async def _apply_operation(
    workspace: Workspace,
    operation: _PreparedOperation,
) -> WorkspaceMutationResult | WorkspaceMoveResult:
    intent = operation.intent
    if intent.operation_type == "create":
        assert intent.destination_path is not None and operation.after is not None
        return await workspace.create_bytes(intent.destination_path, operation.after)
    if intent.operation_type == "update":
        assert (
            intent.destination_path is not None
            and intent.expected_revision is not None
            and operation.after is not None
        )
        return await workspace.replace_bytes(
            intent.destination_path,
            operation.after,
            expected_revision=intent.expected_revision,
        )
    if intent.operation_type == "delete":
        assert intent.source_path is not None and intent.expected_revision is not None
        return await workspace.delete_if_revision(
            intent.source_path,
            expected_revision=intent.expected_revision,
        )
    assert (
        intent.source_path is not None
        and intent.destination_path is not None
        and intent.expected_revision is not None
    )
    return await workspace.move_if_revision(
        intent.source_path,
        intent.destination_path,
        expected_source_revision=intent.expected_revision,
        require_destination_absent=True,
    )


def _initial_operation_evidence(
    operation: _PreparedOperation,
    application_order: int,
) -> dict[str, Any]:
    intent = operation.intent
    return {
        "index": intent.index,
        "application_order": application_order,
        "type": intent.operation_type,
        "source_path": intent.source_path,
        "destination_path": intent.destination_path,
        "status": "not_started",
        "expected_revision": intent.expected_revision,
        "before_revision": operation.before_revision,
        "after_revision": None,
        "before_sha256": operation.before_sha256,
        "after_sha256": None,
        "projected_after_sha256": operation.after_sha256,
        "before_bytes": len(operation.before) if operation.before is not None else None,
        "after_bytes": None,
        "projected_after_bytes": len(operation.after) if operation.after is not None else None,
        "edit_count": len(intent.edits),
        "replacement_count": operation.replacement_count,
        "move_fidelity": None,
    }


def _settle_success_evidence(
    evidence: dict[str, Any],
    operation: _PreparedOperation,
    mutation: WorkspaceMutationResult | WorkspaceMoveResult,
) -> None:
    intent = operation.intent
    if intent.operation_type == "move":
        if type(mutation) is not WorkspaceMoveResult:
            raise RuntimeError("Workspace move returned invalid evidence.")
        _settle_move_result(evidence, operation, mutation)
        evidence["status"] = "applied"
        return
    if type(mutation) is not WorkspaceMutationResult:
        raise RuntimeError("Workspace mutation returned invalid evidence.")
    expected_operation = {
        "create": "create",
        "update": "replace",
        "delete": "delete",
    }[intent.operation_type]
    if mutation.operation != expected_operation:
        raise RuntimeError("Workspace mutation returned the wrong operation evidence.")
    if operation.before_sha256 != mutation.before_sha256:
        raise RuntimeError("Workspace mutation before identity did not match preflight.")
    if operation.after_sha256 != mutation.after_sha256:
        raise RuntimeError("Workspace mutation after identity did not match preflight.")
    evidence.update(
        {
            "status": "applied",
            "before_revision": mutation.before_revision,
            "after_revision": mutation.after_revision,
            "before_sha256": mutation.before_sha256,
            "after_sha256": mutation.after_sha256,
            "before_bytes": mutation.before_bytes,
            "after_bytes": mutation.after_bytes,
        }
    )


def _settle_move_result(
    evidence: dict[str, Any],
    operation: _PreparedOperation,
    mutation: WorkspaceMoveResult,
) -> None:
    intent = operation.intent
    if (
        mutation.source_before_revision != intent.expected_revision
        or mutation.source_before_sha256 != operation.before_sha256
        or mutation.destination_after_sha256 != operation.after_sha256
    ):
        raise RuntimeError("Workspace move evidence did not match preflight.")
    evidence.update(
        {
            "before_revision": mutation.source_before_revision,
            "after_revision": mutation.destination_after_revision,
            "before_sha256": mutation.source_before_sha256,
            "after_sha256": mutation.destination_after_sha256,
            "before_bytes": mutation.source_before_bytes,
            "after_bytes": mutation.destination_after_bytes,
            "move_fidelity": mutation.fidelity,
        }
    )


async def _reconcile_operation(
    workspace: Workspace,
    operation: _PreparedOperation,
    max_file_bytes: int,
) -> tuple[
    PatchOperationStatus,
    dict[str, tuple[str, str | None, str | None, int | None]],
]:
    intent = operation.intent
    observations = await _observe_operation_paths(workspace, operation, max_file_bytes)
    if intent.operation_type == "move":
        source = observations["source"]
        destination = observations["destination"]
        if source[0] == "absent" and _observation_matches(destination, operation.after_sha256):
            return "applied", observations
        if _observation_matches(source, operation.before_sha256) and destination[0] == "absent":
            return "failed", observations
        if "unknown" in {source[0], destination[0]}:
            return "unknown", observations
        return "conflict", observations

    role = "destination" if intent.operation_type == "create" else "source"
    observed = observations[role]
    if intent.operation_type == "create":
        if _observation_matches(observed, operation.after_sha256):
            return "applied", observations
        status: PatchOperationStatus = "failed" if observed[0] == "absent" else "conflict"
        return status, observations
    if intent.operation_type == "delete":
        if observed[0] == "absent":
            return "applied", observations
        if _observation_matches(observed, operation.before_sha256):
            return "failed", observations
        status = "unknown" if observed[0] == "unknown" else "conflict"
        return status, observations
    if _observation_matches(observed, operation.after_sha256):
        return "applied", observations
    if _observation_matches(observed, operation.before_sha256):
        return "failed", observations
    status = "unknown" if observed[0] == "unknown" else "conflict"
    return status, observations


async def _observe_operation_paths(
    workspace: Workspace,
    operation: _PreparedOperation,
    max_file_bytes: int,
) -> dict[str, tuple[str, str | None, str | None, int | None]]:
    intent = operation.intent
    if intent.operation_type == "move":
        assert intent.source_path is not None and intent.destination_path is not None
        source, destination = await asyncio.gather(
            _observe_path(workspace, intent.source_path, max_file_bytes),
            _observe_path(workspace, intent.destination_path, max_file_bytes),
        )
        return {"source": source, "destination": destination}
    path = intent.destination_path or intent.source_path
    assert path is not None
    observed = await _observe_path(workspace, path, max_file_bytes)
    role = "destination" if intent.operation_type == "create" else "source"
    return {role: observed}


def _record_operation_observations(
    evidence: dict[str, Any],
    observations: Mapping[str, tuple[str, str | None, str | None, int | None]],
) -> None:
    for role, observed in observations.items():
        state, revision, digest, size = observed
        evidence[f"observed_{role}_state"] = state
        evidence[f"observed_{role}_revision"] = revision
        evidence[f"observed_{role}_sha256"] = digest
        evidence[f"observed_{role}_bytes"] = size


async def _observe_path(
    workspace: Workspace,
    path: str,
    max_file_bytes: int,
) -> tuple[str, str | None, str | None, int | None]:
    try:
        read_limit = workspace.bounded_read_limit(max_file_bytes)
        read = await workspace.read_bytes(path, max_bytes=read_limit)
    except FileNotFoundError:
        return ("absent", None, None, None)
    except BaseException:
        return ("unknown", None, None, None)
    if read.truncated or read.offset != 0:
        return ("unknown", None, None, None)
    digest = read.sha256 or hashlib.sha256(read.content).hexdigest()
    return ("present", read.revision, digest, len(read.content))


def _observation_matches(
    observation: tuple[str, str | None, str | None, int | None],
    sha256: str | None,
) -> bool:
    return sha256 is not None and observation[0] == "present" and observation[2] == sha256


def _settle_reconciled_evidence(
    evidence: dict[str, Any],
    operation: _PreparedOperation,
    status: PatchOperationStatus,
    *,
    observations: Mapping[str, tuple[str, str | None, str | None, int | None]],
) -> None:
    evidence["status"] = status
    if status == "applied":
        evidence["after_sha256"] = operation.after_sha256
        role = "destination" if operation.intent.operation_type in {"create", "move"} else "source"
        observed = observations.get(role)
        if observed is not None and observed[0] == "present":
            evidence["after_revision"] = observed[1]
            evidence["after_sha256"] = observed[2]
            evidence["after_bytes"] = observed[3]


def _preflight_failure_result(
    ctx: ToolContext,
    error: _PatchPreflightError,
    initial_snapshot: InvocationRedactorSnapshot,
) -> ToolResult:
    raw = {
        "outcome": "precondition_failed",
        "operation_index": error.operation_index,
        "path": error.path,
        "category": error.category,
        "mutated": False,
    }
    redacted = _redact_projection_stably(ctx, raw, initial_snapshot)
    if redacted is None:
        return _unstable_projection_result(outcome="precondition_failed", mutated=False)
    safe, snapshot = redacted
    record_ambiguous_secret_output(ctx, snapshot)
    index_note = (
        "" if safe["operation_index"] is None else f" at operation {safe['operation_index']}"
    )
    path_note = "" if safe["path"] is None else f" ({safe['path']})"
    return ToolResult(
        content=(
            f"Patch preflight refused{index_note}{path_note}: {safe['category']}. "
            "No changes were written."
        ),
        structured=safe,
        is_error=True,
    )


def _redact_projection_stably(
    ctx: ToolContext,
    raw: Any,
    initial_snapshot: InvocationRedactorSnapshot,
) -> tuple[Any, InvocationRedactorSnapshot] | None:
    snapshot = initial_snapshot
    for _ in range(2):
        safe = snapshot.redactor.redact_json_values(raw)
        current = active_secret_redactor_snapshot(ctx)
        if current.revision == snapshot.revision and current.redactor.has_same_registry(
            snapshot.redactor
        ):
            return safe, current
        snapshot = current
    return None


def _redact_patch_output_stably(
    ctx: ToolContext,
    operations: list[dict[str, Any]],
    diff: str,
    initial_snapshot: InvocationRedactorSnapshot,
) -> tuple[dict[str, Any], InvocationRedactorSnapshot] | None:
    snapshot = initial_snapshot
    sensitive_fields = {
        "source_path",
        "destination_path",
        "expected_revision",
        "before_revision",
        "after_revision",
        "observed_source_revision",
        "observed_destination_revision",
    }
    for _ in range(2):
        safe_operations: list[dict[str, Any]] = []
        for operation in operations:
            safe = dict(operation)
            for field_name in sensitive_fields:
                value = safe.get(field_name)
                if type(value) is str:
                    safe[field_name] = snapshot.redactor.redact_text(value)
            safe_operations.append(safe)
        safe = {
            "operations": safe_operations,
            "diff": snapshot.redactor.redact_text(diff),
        }
        current = active_secret_redactor_snapshot(ctx)
        if current.revision == snapshot.revision and current.redactor.has_same_registry(
            snapshot.redactor
        ):
            return safe, current
        snapshot = current
    return None


def _stable_projection_snapshot(
    ctx: ToolContext,
    initial_snapshot: InvocationRedactorSnapshot,
) -> InvocationRedactorSnapshot | None:
    current = active_secret_redactor_snapshot(ctx)
    if current.revision == initial_snapshot.revision and current.redactor.has_same_registry(
        initial_snapshot.redactor
    ):
        return current
    second = active_secret_redactor_snapshot(ctx)
    if second.revision == current.revision and second.redactor.has_same_registry(current.redactor):
        return second
    return None


def _projection_snapshot_is_current(
    ctx: ToolContext,
    snapshot: InvocationRedactorSnapshot,
) -> bool:
    current = active_secret_redactor_snapshot(ctx)
    return current.revision == snapshot.revision and current.redactor.has_same_registry(
        snapshot.redactor
    )


def _unstable_projection_result(*, outcome: str, mutated: bool) -> ToolResult:
    return ToolResult(
        content=(
            "Patch publication was omitted because its secret-redaction scope changed "
            "repeatedly. Inspect fresh workspace evidence before continuing."
        ),
        structured={
            "error": "secret_redaction_scope_unstable",
            "workspace_outcome": outcome,
            "workspace_may_have_changed": mutated,
        },
        is_error=True,
    )


def _bounded_diff_preview(
    safe_diff: str,
    operations: tuple[_PreparedOperation, ...],
    *,
    redactor: Any,
    max_file_bytes: int,
    max_total_bytes: int,
) -> tuple[str, bool, list[str]]:
    per_file_parts: list[str] = []
    per_file_truncated = False
    for operation in operations:
        safe_operation_diff = redactor.redact_text(operation.diff)
        bounded, truncated = _truncate_utf8(
            safe_operation_diff,
            max_file_bytes,
            marker="\n[patch file diff truncated]\n",
        )
        per_file_parts.append(bounded)
        per_file_truncated = per_file_truncated or truncated
    joined = "".join(per_file_parts)
    # Redacting the complete aggregate is an additional boundary. Use it when
    # no per-file truncation changed the concatenation; otherwise each retained
    # file segment has already been independently redacted.
    if not per_file_truncated:
        joined = safe_diff
    preview, total_truncated = _truncate_utf8(
        joined,
        max_total_bytes,
        marker="\n[patch aggregate diff truncated]\n",
    )
    reasons = []
    if per_file_truncated:
        reasons.append("per_file_bytes")
    if total_truncated:
        reasons.append("aggregate_bytes")
    return preview, bool(reasons), reasons


def _truncate_utf8(value: str, maximum: int, *, marker: str) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= maximum:
        return value, False
    marker_bytes = marker.encode("utf-8")
    if len(marker_bytes) >= maximum:
        return marker_bytes[:maximum].decode("utf-8", errors="ignore"), True
    prefix = encoded[: maximum - len(marker_bytes)]
    return prefix.decode("utf-8", errors="ignore").rstrip() + marker, True


def _patch_result_identities(ctx: ToolContext, redactor: Any) -> dict[str, Any]:
    authority = _runtime_tool_invocation_authority(ctx)
    metadata_tool_call_id = ctx.metadata.get("tool_call_id")
    return {
        "session_id": _safe_identity_projection(redactor, ctx.session_id),
        "run_epoch": None if authority is None else authority.parent_run_epoch,
        "model_step_id": _safe_identity_projection(
            redactor,
            None if authority is None else authority.model_step_id,
        ),
        "model_attempt_id": _safe_identity_projection(
            redactor,
            None if authority is None else authority.model_attempt_id,
        ),
        "tool_round_id": _safe_identity_projection(
            redactor,
            None if authority is None else authority.tool_round_id,
        ),
        "tool_call_id": _safe_identity_projection(
            redactor,
            (
                authority.tool_call_id
                if authority is not None
                else metadata_tool_call_id
                if type(metadata_tool_call_id) is str
                else None
            ),
        ),
        "tool_call_identity": _safe_identity_projection(redactor, ctx.idempotency_key),
        "workspace_id": _safe_identity_projection(redactor, ctx.workspace_id),
        "agent_name": _safe_identity_projection(redactor, ctx.agent_name),
        "environment_name": _safe_identity_projection(redactor, ctx.environment_name),
    }


def _safe_identity_projection(redactor: Any, value: str | None) -> str | None:
    if value is None:
        return None
    projected = redactor.redact_text(value)
    bounded, _ = _truncate_utf8(projected, 4_096, marker="[identity truncated]")
    return bounded


def _bounded_operation_manifest(
    operations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool, list[str]]:
    path_fields = ("source_path", "destination_path")
    revision_fields = (
        "expected_revision",
        "before_revision",
        "after_revision",
        "observed_source_revision",
        "observed_destination_revision",
    )
    projected: list[dict[str, Any]] = []
    reasons: set[str] = set()
    for operation in operations:
        safe = dict(operation)
        for field_name in path_fields:
            value = safe.get(field_name)
            safe[f"{field_name}_sha256"] = _optional_identity_sha256(
                value if type(value) is str else None
            )
            if type(value) is str:
                safe[field_name], truncated = _truncate_utf8(
                    value,
                    MAX_PATCH_PROJECTED_PATH_BYTES,
                    marker="[path truncated]",
                )
                if truncated:
                    reasons.add("path_bytes")
        for field_name in revision_fields:
            value = safe.get(field_name)
            safe[f"{field_name}_sha256"] = _optional_identity_sha256(
                value if type(value) is str else None
            )
            if type(value) is str:
                safe[field_name], truncated = _truncate_utf8(
                    value,
                    MAX_PATCH_PROJECTED_REVISION_BYTES,
                    marker="[revision truncated]",
                )
                if truncated:
                    reasons.add("revision_bytes")
        projected.append(safe)
    ordered_reasons = sorted(reasons)
    return projected, bool(ordered_reasons), ordered_reasons


def _compact_operation_manifest(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "index",
        "application_order",
        "type",
        "source_path_sha256",
        "destination_path_sha256",
        "status",
        "expected_revision_sha256",
        "before_revision_sha256",
        "after_revision_sha256",
        "before_sha256",
        "after_sha256",
        "projected_after_sha256",
        "before_bytes",
        "after_bytes",
        "projected_after_bytes",
        "edit_count",
        "replacement_count",
        "move_fidelity",
        "observed_source_state",
        "observed_source_revision_sha256",
        "observed_source_sha256",
        "observed_source_bytes",
        "observed_destination_state",
        "observed_destination_revision_sha256",
        "observed_destination_sha256",
        "observed_destination_bytes",
    )
    return [
        {field_name: operation.get(field_name) for field_name in fields} for operation in operations
    ]


def _changed_path_manifest(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for operation in operations:
        operation_type = operation["type"]
        source_path = operation.get("source_path")
        destination_path = operation.get("destination_path")
        if source_path is not None and source_path == destination_path:
            changed.append(
                {
                    "operation_index": operation["index"],
                    "path": source_path,
                    "path_sha256": operation.get("source_path_sha256"),
                    "role": "source_and_destination",
                    "before_revision": operation.get("before_revision"),
                    "before_revision_sha256": operation.get("before_revision_sha256"),
                    "after_revision": operation.get("after_revision"),
                    "after_revision_sha256": operation.get("after_revision_sha256"),
                    "before_sha256": operation.get("before_sha256"),
                    "after_sha256": operation.get("after_sha256"),
                    "before_bytes": operation.get("before_bytes"),
                    "after_bytes": operation.get("after_bytes"),
                    "status": operation["status"],
                }
            )
            continue
        if source_path is not None:
            changed.append(
                {
                    "operation_index": operation["index"],
                    "path": source_path,
                    "path_sha256": operation.get("source_path_sha256"),
                    "role": "source",
                    "before_revision": operation.get("before_revision"),
                    "before_revision_sha256": operation.get("before_revision_sha256"),
                    "after_revision": None
                    if operation_type in {"delete", "move"}
                    else operation.get("after_revision"),
                    "after_revision_sha256": None
                    if operation_type in {"delete", "move"}
                    else operation.get("after_revision_sha256"),
                    "before_sha256": operation.get("before_sha256"),
                    "after_sha256": None
                    if operation_type in {"delete", "move"}
                    else operation.get("after_sha256"),
                    "before_bytes": operation.get("before_bytes"),
                    "after_bytes": None
                    if operation_type in {"delete", "move"}
                    else operation.get("after_bytes"),
                    "status": operation["status"],
                }
            )
        if destination_path is not None:
            changed.append(
                {
                    "operation_index": operation["index"],
                    "path": destination_path,
                    "path_sha256": operation.get("destination_path_sha256"),
                    "role": "destination",
                    "before_revision": None,
                    "before_revision_sha256": None,
                    "after_revision": operation.get("after_revision"),
                    "after_revision_sha256": operation.get("after_revision_sha256"),
                    "before_sha256": None,
                    "after_sha256": operation.get("after_sha256"),
                    "before_bytes": None,
                    "after_bytes": operation.get("after_bytes"),
                    "status": operation["status"],
                }
            )
    return changed


def _compact_changed_path_manifest(changed_paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "operation_index",
        "path_sha256",
        "role",
        "before_revision_sha256",
        "after_revision_sha256",
        "before_sha256",
        "after_sha256",
        "before_bytes",
        "after_bytes",
        "status",
    )
    return [{field_name: item.get(field_name) for field_name in fields} for item in changed_paths]


async def _store_patch_artifact(
    ctx: ToolContext,
    *,
    patch_id: str,
    behavior_profile_id: str,
    outcome: PatchOutcome,
    operations: Any,
    changed_paths: Any,
    diff: str,
    identities: Mapping[str, Any],
    max_bytes: int,
    projection_snapshot: InvocationRedactorSnapshot,
) -> tuple[str, dict[str, Any] | None]:
    artifact_store = ctx.artifact_store
    if artifact_store is None:
        return "unavailable", None
    content = _canonical_json_bytes(
        {
            "version": PATCH_RESULT_VERSION,
            "patch_id": patch_id,
            "outcome": outcome,
            "behavior_profile_id": behavior_profile_id,
            "cross_file_atomic": False,
            "identities": identities,
            "operations": operations,
            "changed_paths": changed_paths,
            "diff": diff,
        }
    )
    if len(content) > max_bytes:
        return "too_large", None
    content_sha256 = hashlib.sha256(content).hexdigest()
    artifact_id = None
    if ctx.idempotency_key is not None:
        identity = hashlib.sha256(
            b"cayu-apply-patch-v2\0"
            + ctx.session_id.encode("utf-8")
            + b"\0"
            + ctx.idempotency_key.encode("utf-8")
            + b"\0"
            + patch_id.encode("ascii")
        ).hexdigest()[:32]
        artifact_id = f"art_{identity}"
    try:
        artifact = await artifact_store.put_bytes(
            content,
            artifact_id=artifact_id,
            filename=f"{patch_id}-manifest.json",
            content_type="application/json",
            scope=ArtifactScope.SESSION,
            session_id=ctx.session_id,
            agent_name=ctx.agent_name,
            environment_name=ctx.environment_name,
            metadata={
                "operation": "apply_patch",
                "patch_id": patch_id,
                "outcome": outcome,
                "behavior_profile_id": behavior_profile_id,
                "content_sha256": content_sha256,
                "result_projection_version": PATCH_RESULT_VERSION,
            },
        )
    except Exception:
        return "failed", None
    if not isinstance(artifact, ArtifactMetadata):
        return "failed", None
    if not _projection_snapshot_is_current(ctx, projection_snapshot):
        try:
            await artifact_store.delete(artifact.id)
        except Exception:
            return "secret_scope_cleanup_failed", None
        return "secret_scope_unstable", None
    return (
        "stored",
        {
            "artifact_id": artifact.id,
            "content_type": artifact.content_type,
            "size_bytes": artifact.size_bytes,
            "sha256": content_sha256,
        },
    )


def _require_observation_workspace(ctx: ToolContext) -> Workspace | None:
    authority = ctx._authoritative_workspace_for_builtin()
    workspace = authority if authority is not None else ctx.workspace
    if workspace is None:
        return None
    if not isinstance(workspace, Workspace):
        raise TypeError("Tool context workspace must implement Workspace.")
    return workspace


def _require_mutation_workspace(ctx: ToolContext) -> Workspace | None:
    workspace = ctx.workspace
    if workspace is None:
        return None
    if not isinstance(workspace, Workspace):
        raise TypeError("Patch mutations require a Workspace invocation facade.")
    return workspace


def _configuration_int(value: int, field_name: str, *, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an integer.")
    if not 0 < value <= maximum:
        raise ValueError(f"{field_name} must be from 1 through {maximum}.")
    return value


def _validate_protected_entry_names(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes):
        raise TypeError("protected_entry_names must be an iterable of strings.")
    try:
        copied = tuple(values)
    except TypeError as exc:
        raise TypeError("protected_entry_names must be an iterable of strings.") from exc
    validated: list[str] = []
    seen: set[str] = set()
    for value in copied:
        if type(value) is not str:
            raise TypeError("protected_entry_names entries must be strings.")
        value = require_unicode_scalar_text(
            require_nonblank(value, "protected entry name"),
            "protected entry name",
        )
        if value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
            raise ValueError("protected_entry_names entries must be single path segments.")
        key = value.casefold()
        if key in seen:
            raise ValueError("protected_entry_names entries must be unique.")
        seen.add(key)
        validated.append(value)
    return tuple(validated)


def _behavior_profile_id(material: Mapping[str, object]) -> str:
    return "apply_patch_v1:" + hashlib.sha256(_canonical_json_bytes(material)).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
