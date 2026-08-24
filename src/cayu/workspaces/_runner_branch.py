"""Durable portable workspace branches retained inside a remote runner."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from cayu.runners import ExecCommand
from cayu.runners.base import (
    RunnerWorkspaceMutationSettlement,
    runner_pending_command_settlement_cancellation_safe,
    runner_workspace_mutation_settlement,
)
from cayu.workspaces._runner_branch_guest import RUNNER_WORKSPACE_BRANCH_PROGRAM
from cayu.workspaces.base import (
    Workspace,
    WorkspaceListResult,
    WorkspaceMutationResult,
    WorkspaceReadOffsetError,
    WorkspaceReadResult,
    WorkspaceRevisionMismatchError,
    _validate_workspace_offset,
    _validate_workspace_positive_limit,
    _validate_workspace_relative_path,
    _validate_workspace_revision,
    translate_list_pattern,
    validate_list_pattern,
)
from cayu.workspaces.branches import (
    RemoteWorkspaceBranchAuthorityProvider,
    WorkspaceBranch,
    WorkspaceBranchAuthority,
    WorkspaceBranchBindingAuthority,
    WorkspaceBranchBindingAuthorityClaim,
    WorkspaceBranchBindingAuthorityClaimScope,
    WorkspaceBranchChange,
    WorkspaceBranchChangeSet,
    WorkspaceBranchClosedError,
    WorkspaceBranchConflict,
    WorkspaceBranchContentIdentity,
    WorkspaceBranchCreationResult,
    WorkspaceBranchDurableState,
    WorkspaceBranchFencedError,
    WorkspaceBranchLifecycleStatus,
    WorkspaceBranchOperationConflict,
    WorkspaceBranchOutcomeStatus,
    WorkspaceBranchPublicationRequest,
    WorkspaceBranchPublicationResult,
    WorkspaceBranchRecoveryRequest,
    WorkspaceBranchRecoveryResult,
    WorkspaceBranchRequest,
    WorkspaceBranchResourceExhaustedError,
    WorkspaceBranchRollbackRequest,
    WorkspaceBranchRollbackResult,
    _bounded_workspace_branch_evidence,
    _copy_workspace_branch_authority,
    _copy_workspace_branch_request_envelope,
    _workspace_branch_fixed_authority_evidence_limit_violation,
    copy_workspace_branch_request,
    workspace_branch_change_set_digest,
)
from cayu.workspaces.revisions import WorkspaceIdentity

if TYPE_CHECKING:
    from cayu.workspaces.runner import RunnerWorkspace


_OUTPUT_HEADROOM_BYTES = 16 * 1024
_MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
# Claims in this registry still own a runner command whose settlement is
# unknown. They must never be released merely because another operation starts.
_RETAINED_BINDING_CLAIMS: dict[int, WorkspaceBranchBindingAuthorityClaim] = {}
# These claims reached positive runner quiescence; only their final release
# failed, so a later operation may safely retry that release.
_RETRYABLE_BINDING_CLAIM_RELEASES: dict[int, WorkspaceBranchBindingAuthorityClaim] = {}
_REMOTE_DETAIL_CODES = frozenset(
    {
        "active_branch_capacity_unavailable",
        "active_branch_limit_exceeded",
        "baseline_byte_limit_exceeded",
        "branch_record_limit_exceeded",
        "change_evidence_limit_exceeded",
        "changed_path_limit_exceeded",
        "conflict_evidence_limit_exceeded",
        "file_byte_limit_exceeded",
        "file_count_limit_exceeded",
        "list_evidence_limit_exceeded",
        "overlay_byte_limit_exceeded",
        "path_byte_limit_exceeded",
        "path_count_limit_exceeded",
        "publication_attempt_limit_exceeded",
        "workspace_branch_allocation_changed",
        "workspace_branch_already_committed",
        "workspace_branch_binding_authority_changed",
        "workspace_branch_binding_generation_changed",
        "workspace_branch_change_set_mismatch",
        "workspace_branch_committed",
        "workspace_branch_creation_identity_reused",
        "workspace_branch_expired",
        "workspace_branch_guest_guard_unsupported",
        "workspace_branch_guest_operation_failed",
        "workspace_branch_guard_failed",
        "workspace_branch_identity_mismatch",
        "workspace_branch_identity_terminal",
        "workspace_branch_not_open",
        "workspace_branch_operation_authority_changed",
        "workspace_branch_private_cleanup_invalid",
        "workspace_branch_private_content_changed",
        "workspace_branch_private_content_invalid",
        "workspace_branch_private_content_missing",
        "workspace_branch_private_filesystem_changed",
        "workspace_branch_private_identity_collision",
        "workspace_branch_private_identity_invalid",
        "workspace_branch_private_index_invalid",
        "workspace_branch_private_path_invalid",
        "workspace_branch_private_root_changed",
        "workspace_branch_private_root_invalid",
        "workspace_branch_publication_attempts_invalid",
        "workspace_branch_publication_authority_missing",
        "workspace_branch_publication_authority_unexpected",
        "workspace_branch_publication_changes_missing",
        "workspace_branch_publication_directories_missing",
        "workspace_branch_publication_failed",
        "workspace_branch_publication_identity_reused",
        "workspace_branch_publication_record_missing",
        "workspace_branch_publication_rollback_failed",
        "workspace_branch_publication_source_ambiguous",
        "workspace_branch_publication_unsettled",
        "workspace_branch_publication_verification_failed",
        "workspace_branch_record_corrupt",
        "workspace_branch_record_integrity_failed",
        "workspace_branch_record_invalid",
        "workspace_branch_record_missing",
        "workspace_branch_record_oversized",
        "workspace_branch_recovery_authority_changed",
        "workspace_branch_recovery_not_durable",
        "workspace_branch_rollback_authority_changed",
        "workspace_branch_rollback_identity_reused",
        "workspace_branch_rolled_back",
        "workspace_branch_run_epoch_changed",
        "workspace_branch_source_conflict",
        "workspace_branch_source_changed_during_capture",
        "workspace_branch_source_root_changed",
        "workspace_branch_unexpected_binding_authority",
    }
)


class _RemoteBranchSourceConflict(RuntimeError):
    pass


def _remote_detail_code(value: object, *, fallback: str) -> str:
    return value if type(value) is str and value in _REMOTE_DETAIL_CODES else fallback


def _validated_change_set_digest(value: object) -> str:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise WorkspaceBranchFencedError("Remote workspace branch change-set digest is invalid.")
    return value


def _identity(value: object) -> WorkspaceBranchContentIdentity | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise WorkspaceBranchFencedError("Remote workspace branch returned invalid identity.")
    try:
        return WorkspaceBranchContentIdentity.model_validate(value)
    except Exception as exc:
        raise WorkspaceBranchFencedError(
            "Remote workspace branch returned invalid identity."
        ) from exc


def _changes(
    values: object,
) -> tuple[WorkspaceBranchChange, ...]:
    if type(values) is not list:
        raise WorkspaceBranchFencedError("Remote workspace branch returned invalid changes.")
    changes: list[WorkspaceBranchChange] = []
    for value in values:
        if type(value) is not dict:
            raise WorkspaceBranchFencedError("Remote workspace branch returned invalid changes.")
        try:
            changes.append(WorkspaceBranchChange.model_validate(value))
        except Exception as exc:
            raise WorkspaceBranchFencedError(
                "Remote workspace branch returned invalid changes."
            ) from exc
    if tuple(change.path for change in changes) != tuple(sorted(change.path for change in changes)):
        raise WorkspaceBranchFencedError("Remote workspace branch changes are unordered.")
    return tuple(changes)


def _conflicts(values: object) -> tuple[WorkspaceBranchConflict, ...]:
    if values is None:
        return ()
    if type(values) is not list:
        raise WorkspaceBranchFencedError("Remote workspace branch returned invalid conflicts.")
    conflicts: list[WorkspaceBranchConflict] = []
    for value in values:
        if type(value) is not dict:
            raise WorkspaceBranchFencedError("Remote workspace branch returned invalid conflicts.")
        try:
            conflicts.append(WorkspaceBranchConflict.model_validate(value))
        except Exception as exc:
            raise WorkspaceBranchFencedError(
                "Remote workspace branch returned invalid conflicts."
            ) from exc
    if tuple(conflict.path for conflict in conflicts) != tuple(
        sorted(conflict.path for conflict in conflicts)
    ):
        raise WorkspaceBranchFencedError("Remote workspace branch conflicts are unordered.")
    return tuple(conflicts)


def _validate_conflict_evidence_limit(
    conflicts: tuple[WorkspaceBranchConflict, ...],
    *,
    evidence,
    max_bytes: int,
) -> None:
    serialized_bytes = 2 + len(evidence.model_dump_json().encode("utf-8"))
    for index, conflict in enumerate(conflicts):
        if index:
            serialized_bytes += 1
        serialized_bytes += len(conflict.model_dump_json().encode("utf-8"))
        if serialized_bytes > max_bytes:
            raise WorkspaceBranchResourceExhaustedError("conflict_evidence_limit_exceeded")


def _change_set(
    *,
    branch_id: str,
    source: WorkspaceIdentity,
    baseline_revision: str,
    payload: dict[str, Any],
) -> WorkspaceBranchChangeSet:
    changes = _changes(payload.get("changes"))
    expected = workspace_branch_change_set_digest(
        branch_id=branch_id,
        source=source,
        baseline_revision=baseline_revision,
        changes=changes,
    )
    if payload.get("digest") != expected:
        raise WorkspaceBranchFencedError("Remote workspace branch digest is invalid.")
    return WorkspaceBranchChangeSet(
        branch_id=branch_id,
        source=source,
        baseline_revision=baseline_revision,
        changes=changes,
        digest=expected,
    )


def _mutation(
    value: object,
    *,
    expected_operation: Literal["create", "replace", "delete"],
) -> WorkspaceMutationResult | None:
    if value is None:
        return None
    if type(value) is not dict:
        raise WorkspaceBranchFencedError("Remote workspace branch returned invalid mutation.")
    owned = cast("dict[str, object]", value)
    before = _identity(owned.get("before"))
    after = _identity(owned.get("after"))
    operation = owned.get("operation")
    if operation != expected_operation:
        raise WorkspaceBranchFencedError("Remote workspace branch returned invalid mutation.")
    try:
        return WorkspaceMutationResult(
            operation=cast("Literal['create', 'replace', 'delete']", operation),
            before_revision=None if before is None else f"sha256:{before.sha256}",
            after_revision=None if after is None else f"sha256:{after.sha256}",
            before_sha256=None if before is None else before.sha256,
            after_sha256=None if after is None else after.sha256,
            before_bytes=None if before is None else before.bytes,
            after_bytes=None if after is None else after.bytes,
        )
    except (TypeError, ValueError):
        raise WorkspaceBranchFencedError(
            "Remote workspace branch returned invalid mutation."
        ) from None


def _binding_authority(authority: WorkspaceBranchAuthority) -> WorkspaceBranchBindingAuthority:
    return WorkspaceBranchBindingAuthority(
        environment_name=authority.environment_name,
        binding_generation=authority.binding_generation,
        binding_identity=authority.binding_identity,
    )


def _copy_binding_authority(value: object) -> WorkspaceBranchBindingAuthority:
    if type(value) is not WorkspaceBranchBindingAuthority:
        raise TypeError("Workspace branch binding authority is invalid.")
    fields = (
        value.environment_name,
        value.binding_generation,
        value.binding_identity,
    )
    if any(type(field) is not str for field in fields):
        raise TypeError("Workspace branch binding authority fields are invalid.")
    return WorkspaceBranchBindingAuthority(
        environment_name=value.environment_name,
        binding_generation=value.binding_generation,
        binding_identity=value.binding_identity,
    )


def _copy_publication_request(
    request: object,
) -> WorkspaceBranchPublicationRequest:
    if type(request) is not WorkspaceBranchPublicationRequest:
        raise TypeError("Workspace branch publication request is invalid.")
    values = (
        request.branch_id,
        request.baseline_revision,
        request.change_set_digest,
        request.idempotency_key,
        request.binding_generation,
    )
    if any(value is not None and type(value) is not str for value in values):
        raise TypeError("Workspace branch publication request fields are invalid.")
    if request.expected_run_epoch is not None and type(request.expected_run_epoch) is not int:
        raise TypeError("Workspace branch publication run epoch is invalid.")
    return WorkspaceBranchPublicationRequest(
        branch_id=request.branch_id,
        baseline_revision=request.baseline_revision,
        change_set_digest=request.change_set_digest,
        idempotency_key=request.idempotency_key,
        expected_run_epoch=request.expected_run_epoch,
        binding_generation=request.binding_generation,
    )


def _copy_rollback_request(request: object) -> WorkspaceBranchRollbackRequest:
    if type(request) is not WorkspaceBranchRollbackRequest:
        raise TypeError("Workspace branch rollback request is invalid.")
    values = (
        request.branch_id,
        request.idempotency_key,
        request.binding_generation,
        request.reason,
    )
    if any(type(value) is not str for value in values):
        raise TypeError("Workspace branch rollback request fields are invalid.")
    if type(request.expected_run_epoch) is not int:
        raise TypeError("Workspace branch rollback run epoch is invalid.")
    return WorkspaceBranchRollbackRequest(
        branch_id=request.branch_id,
        idempotency_key=request.idempotency_key,
        expected_run_epoch=request.expected_run_epoch,
        binding_generation=request.binding_generation,
        reason=request.reason,
    )


def _copy_recovery_request(request: object) -> WorkspaceBranchRecoveryRequest:
    if type(request) is not WorkspaceBranchRecoveryRequest:
        raise TypeError("Workspace branch recovery request is invalid.")
    values = (
        request.branch_id,
        request.session_id,
        request.binding_generation,
        request.binding_identity,
        request.recovery_id,
    )
    if any(type(value) is not str for value in values):
        raise TypeError("Workspace branch recovery request fields are invalid.")
    if type(request.expected_run_epoch) is not int:
        raise TypeError("Workspace branch recovery run epoch is invalid.")
    return WorkspaceBranchRecoveryRequest(
        branch_id=request.branch_id,
        session_id=request.session_id,
        expected_run_epoch=request.expected_run_epoch,
        binding_generation=request.binding_generation,
        binding_identity=request.binding_identity,
        recovery_id=request.recovery_id,
    )


def _claim_binding(
    source: RunnerWorkspace,
    authority: WorkspaceBranchAuthority | None,
) -> WorkspaceBranchBindingAuthorityClaim | None:
    if authority is None:
        return None
    resolver = source._branch_authority_resolver
    if (
        not isinstance(resolver, RemoteWorkspaceBranchAuthorityProvider)
        or source._branch_claim_scope is not WorkspaceBranchBindingAuthorityClaimScope.DURABLE
    ):
        raise RuntimeError(
            "Durable remote workspace branches require complete cross-process invocation "
            "authority claims."
        )
    _retry_retained_claim_releases()
    return resolver.claim_operation(authority)


def _release_claim(
    claim: WorkspaceBranchBindingAuthorityClaim | None,
    *,
    primary: BaseException | None = None,
) -> None:
    if claim is None:
        return
    try:
        claim.release()
    except BaseException as release_error:
        _retain_claim_release(claim)
        if primary is not None:
            raise primary from release_error
        raise


@dataclass(slots=True)
class _RunnerOperationSettlement:
    outcome: RunnerWorkspaceMutationSettlement | Literal["not_started"] = "not_started"


def _retain_claim(claim: WorkspaceBranchBindingAuthorityClaim) -> None:
    _RETAINED_BINDING_CLAIMS[id(claim)] = claim


def _retain_claim_release(claim: WorkspaceBranchBindingAuthorityClaim) -> None:
    _RETRYABLE_BINDING_CLAIM_RELEASES[id(claim)] = claim


def _retry_retained_claim_releases() -> None:
    for key, claim in tuple(_RETRYABLE_BINDING_CLAIM_RELEASES.items()):
        try:
            claim.release()
        except Exception:
            continue
        except BaseException:
            raise
        else:
            if _RETRYABLE_BINDING_CLAIM_RELEASES.get(key) is claim:
                del _RETRYABLE_BINDING_CLAIM_RELEASES[key]


async def _settle_and_release_claim(
    source: RunnerWorkspace,
    claim: WorkspaceBranchBindingAuthorityClaim | None,
    settlement: _RunnerOperationSettlement,
    *,
    primary: BaseException | None,
) -> bool:
    if claim is None:
        return True
    outcome = settlement.outcome
    if outcome == "deferred":
        if not runner_pending_command_settlement_cancellation_safe(source._runner):
            _retain_claim(claim)
            return False
        try:
            settled = await source._runner.await_pending_command_settlement()
            if type(settled) is not bool:
                raise TypeError("Runner command settlement must return a bool.")
        except BaseException as settlement_error:
            _retain_claim(claim)
            if isinstance(settlement_error, asyncio.CancelledError) or not isinstance(
                settlement_error, Exception
            ):
                if primary is not None:
                    raise settlement_error from primary
                raise
            if primary is not None:
                raise primary from settlement_error
            raise
        if not settled:
            _retain_claim(claim)
            return False
        outcome = "runner_quiescent"
    if outcome == "uncertain":
        _retain_claim(claim)
        return False
    _release_claim(claim, primary=primary)
    return True


def _authority_payload(authority: WorkspaceBranchAuthority | None) -> dict[str, Any] | None:
    if authority is None:
        return None
    return authority.model_dump(mode="json", warnings=False)


def _binding_payload(authority: WorkspaceBranchAuthority | None) -> dict[str, Any] | None:
    if authority is None:
        return None
    return _binding_authority(authority).model_dump(mode="json", warnings=False)


async def _run(
    source: RunnerWorkspace,
    operation: str,
    payload: dict[str, Any],
    *,
    output_limit_bytes: int,
    settlement: _RunnerOperationSettlement,
) -> dict[str, Any]:
    command = ExecCommand.process(
        source.python_executable,
        "-c",
        RUNNER_WORKSPACE_BRANCH_PROGRAM,
        operation,
    )
    standard_input = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    settlement.outcome = "uncertain"
    try:
        result = await source._runner.exec(
            command,
            cwd=source.cwd,
            timeout_s=source.branch_operation_timeout_s,
            stdin=standard_input,
            output_limit_bytes=output_limit_bytes,
        )
    except BaseException as error:
        settlement.outcome = runner_workspace_mutation_settlement(
            result=None,
            error=error,
        )
        standard_input = ""
        payload = {}
        raise
    settlement.outcome = runner_workspace_mutation_settlement(result=result, error=None)
    if result.timed_out:
        del result
        raise WorkspaceBranchFencedError("workspace_branch_guest_operation_timed_out")
    if result.cancelled:
        del result
        raise WorkspaceBranchFencedError("workspace_branch_guest_operation_cancelled")
    if result.stdout_truncated:
        del result
        raise WorkspaceBranchFencedError(
            "Remote workspace branch result exceeded its transfer limit."
        )
    exit_code = result.exit_code
    parsed_response = _parse_remote_response(result.stdout)
    del result
    if type(parsed_response) is not dict:
        parsed_response = None
        raise WorkspaceBranchFencedError(
            "Remote workspace branch returned invalid bounded evidence."
        )
    response = cast("dict[str, Any]", parsed_response)
    if response.get("ok") is not True:
        safe_error = {
            name: response.get(name)
            for name in (
                "actual_revision",
                "detail_code",
                "error_type",
                "expected_revision",
                "total_bytes",
            )
        }
        response = {}
        _raise_remote_error(safe_error)
    if exit_code != 0:
        # Never forward provider stderr/stdout: either may contain guest-private
        # paths, environment data, or command output.
        response = {}
        raise WorkspaceBranchFencedError("Remote workspace branch operation failed.")
    return response


def _parse_remote_response(value: str) -> object | None:
    try:
        return json.loads(value)
    except Exception:
        return None


def _raise_remote_error(payload: dict[str, Any]) -> None:
    error_type = payload.get("error_type")
    detail_code = _remote_detail_code(
        payload.get("detail_code"),
        fallback="workspace_branch_remote_failure",
    )
    if error_type == "resource_exhausted":
        raise WorkspaceBranchResourceExhaustedError(detail_code)
    if error_type == "operation_conflict":
        raise WorkspaceBranchOperationConflict(detail_code)
    if error_type == "branch_closed":
        raise WorkspaceBranchClosedError(detail_code)
    if error_type == "fenced":
        raise WorkspaceBranchFencedError(detail_code)
    if error_type == "conflicted":
        raise _RemoteBranchSourceConflict(detail_code)
    if error_type == "not_found":
        raise FileNotFoundError("Workspace branch file not found.")
    if error_type == "exists":
        raise FileExistsError("Workspace branch file already exists.")
    if error_type == "not_file":
        raise IsADirectoryError("Workspace branch path is not a file.")
    if error_type in {"invalid_path", "invalid_request"}:
        raise ValueError("Workspace branch request is invalid.")
    if error_type == "offset":
        total = payload.get("total_bytes")
        if type(total) is not int or total < 0:
            raise WorkspaceBranchFencedError("Remote workspace branch offset is invalid.")
        raise WorkspaceReadOffsetError(-1, total)
    if error_type == "stale":
        expected = payload.get("expected_revision")
        actual = payload.get("actual_revision")
        if type(expected) is not str or type(actual) is not str:
            raise WorkspaceBranchFencedError("Remote workspace branch revision is invalid.")
        try:
            expected = _validate_workspace_revision(expected, owner="expected_revision")
            actual = _validate_workspace_revision(actual, owner="actual_revision")
        except (TypeError, ValueError):
            raise WorkspaceBranchFencedError(
                "Remote workspace branch revision is invalid."
            ) from None
        raise WorkspaceRevisionMismatchError(expected, actual)
    if error_type == "unsupported":
        raise RuntimeError("Remote workspace branch guest capability is unsupported.")
    raise WorkspaceBranchFencedError("Remote workspace branch operation failed.")


class RunnerWorkspaceBranch(WorkspaceBranch):
    """Ordinary workspace view backed by a retained remote branch overlay."""

    def __init__(
        self,
        *,
        source: RunnerWorkspace,
        branch_id: str,
        source_identity: WorkspaceIdentity,
        baseline_revision: str,
        limits,
        authority: WorkspaceBranchAuthority | None,
    ) -> None:
        self._source = source
        self._branch_id = branch_id
        self._source_identity = WorkspaceIdentity.model_validate(source_identity)
        self._baseline_revision = baseline_revision
        self._limits = limits
        self._authority = authority
        self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
        self.id = f"{source.id}:branch:{branch_id}"

    @property
    def branch_id(self) -> str:
        return self._branch_id

    @property
    def lifecycle_status(self) -> WorkspaceBranchLifecycleStatus:
        return self._lifecycle

    @property
    def resource_key(self) -> tuple[object, ...] | None:
        source_key = self._source.resource_key
        if source_key is None:
            return None
        return ("runner-workspace-branch", source_key, self._branch_id)

    def branch_capabilities(self):
        return self._source.branch_capabilities()

    def bounded_read_limit(self, max_bytes: int) -> int:
        return min(
            self._source.bounded_read_limit(max_bytes),
            self._limits.max_file_bytes,
        )

    def _validated_path(self, path: str) -> str:
        path = _validate_workspace_relative_path(path)
        try:
            path_bytes = len(path.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("Workspace branch path is not portable text.") from None
        if path_bytes > self._limits.max_path_bytes:
            raise WorkspaceBranchResourceExhaustedError("path_byte_limit_exceeded")
        return path

    def _validated_content(self, content: bytes) -> bytes:
        if type(content) is not bytes:
            raise TypeError("Workspace write content must be bytes.")
        if len(content) > self._limits.max_file_bytes:
            raise WorkspaceBranchResourceExhaustedError("file_byte_limit_exceeded")
        return content

    def _invalid_remote_evidence(self, message: str) -> WorkspaceBranchFencedError:
        self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
        return WorkspaceBranchFencedError(message)

    def _envelope(self) -> dict[str, Any]:
        capability = self._source._branch_capability
        if capability is None:
            raise WorkspaceBranchFencedError("Remote workspace branch capability disappeared.")
        return {
            "branch_id": self._branch_id,
            "allocation_fingerprint": capability.allocation_fingerprint,
            "binding_authority": _binding_payload(self._authority),
            "operation_authority": _authority_payload(self._authority),
        }

    async def _operation(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        output_limit_bytes: int | None = None,
    ) -> dict[str, Any]:
        try:
            claim = _claim_binding(self._source, self._authority)
        except WorkspaceBranchOperationConflict:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        settlement = _RunnerOperationSettlement()
        primary: BaseException | None = None
        try:
            return await _run(
                self._source,
                operation,
                {**self._envelope(), **payload},
                output_limit_bytes=(
                    output_limit_bytes
                    if output_limit_bytes is not None
                    else self._limits.max_evidence_bytes + _OUTPUT_HEADROOM_BYTES
                ),
                settlement=settlement,
            )
        except BaseException as error:
            primary = error
            if isinstance(error, WorkspaceBranchFencedError):
                self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            elif isinstance(error, WorkspaceBranchClosedError):
                self._lifecycle = (
                    WorkspaceBranchLifecycleStatus.ROLLED_BACK
                    if str(error) == "workspace_branch_expired"
                    else WorkspaceBranchLifecycleStatus.FENCED
                )
            raise
        finally:
            try:
                claim_released = await _settle_and_release_claim(
                    self._source,
                    claim,
                    settlement,
                    primary=primary,
                )
            except BaseException:
                self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
                raise
            if not claim_released:
                self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED

    async def read_bytes(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int | None = None,
    ) -> WorkspaceReadResult:
        path = self._validated_path(path)
        offset = _validate_workspace_offset(offset, owner="RunnerWorkspaceBranch")
        limit = (
            self.bounded_read_limit(self._limits.max_file_bytes)
            if max_bytes is None
            else self.bounded_read_limit(
                _validate_workspace_positive_limit(
                    max_bytes,
                    "max_bytes",
                    owner="RunnerWorkspaceBranch",
                )
            )
        )
        try:
            payload = await self._operation(
                "read",
                {"path": path, "offset": offset, "limit": limit},
                output_limit_bytes=(4 * ((limit + 2) // 3)) + _OUTPUT_HEADROOM_BYTES,
            )
        except WorkspaceReadOffsetError as error:
            raise WorkspaceReadOffsetError(offset, error.total_bytes) from None
        encoded = payload.get("content_base64")
        if type(encoded) is not str:
            raise self._invalid_remote_evidence("Remote workspace branch read is invalid.")
        try:
            content = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise self._invalid_remote_evidence("Remote workspace branch read is invalid.") from exc
        total = payload.get("total_bytes")
        if (
            type(total) is not int
            or total < offset + len(content)
            or total > self._limits.max_file_bytes
            or len(content) > limit
        ):
            raise self._invalid_remote_evidence("Remote workspace branch read is invalid.")
        truncated = offset + len(content) < total
        revision = payload.get("revision")
        content_sha256 = payload.get("sha256")
        if not truncated and offset == 0:
            expected_sha256 = hashlib.sha256(content).hexdigest()
            if content_sha256 != expected_sha256 or revision != f"sha256:{expected_sha256}":
                raise self._invalid_remote_evidence(
                    "Remote workspace branch read identity is invalid."
                )
        elif revision is not None or content_sha256 is not None:
            raise self._invalid_remote_evidence(
                "Remote workspace branch partial read identity is invalid."
            )
        return WorkspaceReadResult(
            content=content,
            total_bytes=total,
            truncated=truncated,
            offset=offset,
            revision=revision,
            sha256=content_sha256,
        )

    async def write_bytes(self, path: str, content: bytes) -> None:
        path = self._validated_path(path)
        content = self._validated_content(content)
        payload = await self._operation(
            "write",
            {
                "path": path,
                "content_base64": base64.b64encode(content).decode("ascii"),
            },
        )
        mutation = payload.get("mutation")
        operation = mutation.get("operation") if type(mutation) is dict else None
        if operation not in {"create", "replace"}:
            raise self._invalid_remote_evidence(
                "Remote workspace branch write evidence is invalid."
            )
        try:
            result = _mutation(
                mutation,
                expected_operation=cast("Literal['create', 'replace']", operation),
            )
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        expected_sha256 = hashlib.sha256(content).hexdigest()
        if (
            result is None
            or result.after_sha256 != expected_sha256
            or result.after_bytes != len(content)
            or (operation == "create" and result.before_revision is not None)
            or (operation == "replace" and result.before_revision is None)
        ):
            raise self._invalid_remote_evidence(
                "Remote workspace branch write evidence is invalid."
            )

    async def delete(self, path: str) -> None:
        payload = await self._operation(
            "delete",
            {"path": self._validated_path(path)},
        )
        mutation = payload.get("mutation")
        if mutation is None:
            return
        try:
            result = _mutation(mutation, expected_operation="delete")
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        if result is None or result.before_revision is None or result.after_revision is not None:
            raise self._invalid_remote_evidence(
                "Remote workspace branch delete evidence is invalid."
            )

    async def create_bytes(self, path: str, content: bytes) -> WorkspaceMutationResult:
        path = self._validated_path(path)
        content = self._validated_content(content)
        payload = await self._operation(
            "create_file",
            {
                "path": path,
                "content_base64": base64.b64encode(content).decode("ascii"),
            },
        )
        try:
            result = _mutation(payload.get("mutation"), expected_operation="create")
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        if result is None:
            raise self._invalid_remote_evidence("Remote workspace branch lost create evidence.")
        expected_sha256 = hashlib.sha256(content).hexdigest()
        if (
            result.before_revision is not None
            or result.after_sha256 != expected_sha256
            or result.after_bytes != len(content)
        ):
            raise self._invalid_remote_evidence(
                "Remote workspace branch create evidence is invalid."
            )
        return result

    async def replace_bytes(
        self,
        path: str,
        content: bytes,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        path = self._validated_path(path)
        content = self._validated_content(content)
        if type(expected_revision) is not str or not expected_revision.strip():
            raise ValueError("Workspace expected_revision must be nonblank.")
        payload = await self._operation(
            "replace",
            {
                "path": path,
                "content_base64": base64.b64encode(content).decode("ascii"),
                "expected_revision": expected_revision,
            },
        )
        try:
            result = _mutation(payload.get("mutation"), expected_operation="replace")
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        if result is None:
            raise self._invalid_remote_evidence("Remote workspace branch lost replace evidence.")
        expected_sha256 = hashlib.sha256(content).hexdigest()
        if (
            result.before_revision != expected_revision
            or result.after_sha256 != expected_sha256
            or result.after_bytes != len(content)
        ):
            raise self._invalid_remote_evidence(
                "Remote workspace branch replace evidence is invalid."
            )
        return result

    async def delete_if_revision(
        self,
        path: str,
        *,
        expected_revision: str,
    ) -> WorkspaceMutationResult:
        if type(expected_revision) is not str or not expected_revision.strip():
            raise ValueError("Workspace expected_revision must be nonblank.")
        payload = await self._operation(
            "delete_if",
            {
                "path": self._validated_path(path),
                "expected_revision": expected_revision,
            },
        )
        try:
            result = _mutation(payload.get("mutation"), expected_operation="delete")
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        if result is None:
            raise self._invalid_remote_evidence("Remote workspace branch lost delete evidence.")
        if result.before_revision != expected_revision or result.after_revision is not None:
            raise self._invalid_remote_evidence(
                "Remote workspace branch delete evidence is invalid."
            )
        return result

    async def list(
        self,
        pattern: str = "**/*",
        *,
        limit: int | None = None,
    ) -> WorkspaceListResult:
        pattern = validate_list_pattern(pattern)
        effective = (
            self._source.default_list_limit
            if limit is None
            else _validate_workspace_positive_limit(
                limit,
                "limit",
                owner="RunnerWorkspaceBranch",
            )
        )
        payload = await self._operation(
            "list",
            {"pattern": translate_list_pattern(pattern), "limit": effective},
        )
        paths = payload.get("paths")
        total = payload.get("total_count")
        if (
            type(paths) is not list
            or type(total) is not int
            or total < len(paths)
            or len(paths) > effective
            or total > self._limits.max_files
        ):
            raise self._invalid_remote_evidence("Remote workspace branch list is invalid.")
        try:
            validated = tuple(self._validated_path(path) for path in paths)
        except (TypeError, ValueError, WorkspaceBranchResourceExhaustedError):
            raise self._invalid_remote_evidence(
                "Remote workspace branch list is invalid."
            ) from None
        if validated != tuple(sorted(validated)) or len(set(validated)) != len(validated):
            raise self._invalid_remote_evidence("Remote workspace branch list is invalid.")
        return WorkspaceListResult(
            paths=validated,
            total_count=total,
            truncated=total > len(validated),
        )

    async def changes(self) -> WorkspaceBranchChangeSet:
        payload = await self._operation("changes", {})
        try:
            return _change_set(
                branch_id=self._branch_id,
                source=self._source_identity,
                baseline_revision=self._baseline_revision,
                payload=payload,
            )
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise

    async def publish(
        self,
        request: WorkspaceBranchPublicationRequest,
    ) -> WorkspaceBranchPublicationResult:
        copied = _copy_publication_request(request)
        if self._authority is not None and copied.idempotency_key is None:
            raise ValueError("Durable publication requires explicit authority.")
        self._lifecycle = WorkspaceBranchLifecycleStatus.PUBLISHING
        try:
            payload = await self._operation(
                "publish",
                {"request": copied.model_dump(mode="json", warnings=False)},
                output_limit_bytes=(2 * self._limits.max_evidence_bytes) + _OUTPUT_HEADROOM_BYTES,
            )
        except BaseException:
            if self._lifecycle not in {
                WorkspaceBranchLifecycleStatus.FENCED,
                WorkspaceBranchLifecycleStatus.ROLLED_BACK,
            }:
                self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
            raise
        try:
            change_set = _change_set(
                branch_id=self._branch_id,
                source=self._source_identity,
                baseline_revision=self._baseline_revision,
                payload=payload,
            )
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        try:
            status = WorkspaceBranchOutcomeStatus(payload.get("status"))
        except Exception as exc:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise WorkspaceBranchFencedError(
                "Remote workspace branch publication status is invalid."
            ) from exc
        try:
            conflicts = _conflicts(payload.get("conflicts"))
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        if status is WorkspaceBranchOutcomeStatus.COMMITTED:
            self._lifecycle = WorkspaceBranchLifecycleStatus.COMMITTED
        elif status is WorkspaceBranchOutcomeStatus.CONFLICTED:
            self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
        elif status in {
            WorkspaceBranchOutcomeStatus.FAILED,
            WorkspaceBranchOutcomeStatus.AMBIGUOUS,
        }:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
        else:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise WorkspaceBranchFencedError(
                "Remote workspace branch publication status is invalid."
            )
        evidence = _bounded_workspace_branch_evidence(
            source=self._source_identity,
            outcome=status,
            baseline_revision=self._baseline_revision,
            branch_id=self._branch_id,
            change_set_digest=change_set.digest,
            paths=(
                tuple(conflict.path for conflict in conflicts)
                if status is WorkspaceBranchOutcomeStatus.CONFLICTED
                else tuple(change.path for change in change_set.changes)
            ),
            detail_code=(
                "workspace_branch_source_conflict"
                if status is WorkspaceBranchOutcomeStatus.CONFLICTED
                else _remote_detail_code(
                    payload.get("detail_code"),
                    fallback=f"workspace_branch_{status.value}",
                )
            ),
            max_bytes=self._limits.max_evidence_bytes,
        )
        try:
            _validate_conflict_evidence_limit(
                conflicts,
                evidence=evidence,
                max_bytes=self._limits.max_evidence_bytes,
            )
        except WorkspaceBranchResourceExhaustedError as error:
            if status is not WorkspaceBranchOutcomeStatus.CONFLICTED:
                raise
            return WorkspaceBranchPublicationResult(
                status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                evidence=_bounded_workspace_branch_evidence(
                    source=self._source_identity,
                    outcome=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                    baseline_revision=self._baseline_revision,
                    branch_id=self._branch_id,
                    change_set_digest=change_set.digest,
                    detail_code=error.detail_code,
                    max_bytes=self._limits.max_evidence_bytes,
                ),
            )
        return WorkspaceBranchPublicationResult(
            status=status,
            evidence=evidence,
            conflicts=conflicts,
        )

    async def rollback(
        self,
        request: WorkspaceBranchRollbackRequest | None = None,
    ) -> WorkspaceBranchRollbackResult:
        if self._authority is not None and request is None:
            raise ValueError("Durable workspace rollback requires explicit authority.")
        if request is None:
            request_payload: dict[str, Any] = {
                "branch_id": self._branch_id,
                "idempotency_key": "process-local-rollback",
                "expected_run_epoch": 0,
                "binding_generation": "process-local",
                "reason": "explicit",
            }
        elif type(request) is WorkspaceBranchRollbackRequest:
            request_payload = _copy_rollback_request(request).model_dump(
                mode="json",
                warnings=False,
            )
        else:
            raise TypeError("Workspace branch rollback request is invalid.")
        self._lifecycle = WorkspaceBranchLifecycleStatus.ROLLING_BACK
        try:
            payload = await self._operation("rollback", {"request": request_payload})
        except BaseException:
            if self._lifecycle not in {
                WorkspaceBranchLifecycleStatus.FENCED,
                WorkspaceBranchLifecycleStatus.ROLLED_BACK,
            }:
                self._lifecycle = WorkspaceBranchLifecycleStatus.ACTIVE
            raise
        try:
            status = WorkspaceBranchOutcomeStatus(payload.get("status"))
        except Exception as exc:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise WorkspaceBranchFencedError(
                "Remote workspace branch rollback status is invalid."
            ) from exc
        if status not in {
            WorkspaceBranchOutcomeStatus.ROLLED_BACK,
            WorkspaceBranchOutcomeStatus.EXPIRED,
        }:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise WorkspaceBranchFencedError("Remote workspace branch rollback status is invalid.")
        try:
            digest = _validated_change_set_digest(payload.get("digest"))
        except WorkspaceBranchFencedError:
            self._lifecycle = WorkspaceBranchLifecycleStatus.FENCED
            raise
        evidence = _bounded_workspace_branch_evidence(
            source=self._source_identity,
            outcome=status,
            baseline_revision=self._baseline_revision,
            branch_id=self._branch_id,
            change_set_digest=digest,
            detail_code=f"workspace_branch_{status.value}",
            max_bytes=self._limits.max_evidence_bytes,
        )
        self._lifecycle = WorkspaceBranchLifecycleStatus.ROLLED_BACK
        return WorkspaceBranchRollbackResult(status=status, evidence=evidence)


def _creation_without_branch(
    *,
    source: WorkspaceIdentity,
    baseline_revision: str,
    max_evidence_bytes: int,
    status: WorkspaceBranchOutcomeStatus,
    branch_id: str | None,
    detail_code: str,
    conflicts: tuple[WorkspaceBranchConflict, ...] = (),
) -> WorkspaceBranchCreationResult:
    return WorkspaceBranchCreationResult(
        status=status,
        branch=None,
        evidence=_bounded_workspace_branch_evidence(
            source=source,
            outcome=status,
            baseline_revision=baseline_revision,
            branch_id=branch_id,
            paths=tuple(conflict.path for conflict in conflicts),
            detail_code=detail_code,
            max_bytes=max_evidence_bytes,
            hash_fixed_identity_on_overflow=(
                status is WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED
            ),
        ),
        conflicts=conflicts,
    )


async def create_runner_workspace_branch(
    source: RunnerWorkspace,
    request: WorkspaceBranchRequest,
) -> WorkspaceBranchCreationResult:
    from cayu.workspaces.runner import RunnerWorkspace

    if type(source) is not RunnerWorkspace:
        return await Workspace.create_branch(source, request)
    if source._branch_capability is None:
        return await Workspace.create_branch(source, request)
    envelope = _copy_workspace_branch_request_envelope(request)
    if envelope.source.workspace_id != source.id:
        raise ValueError("Workspace branch baseline belongs to a different workspace.")
    try:
        copied = copy_workspace_branch_request(request)
    except WorkspaceBranchResourceExhaustedError as error:
        return _creation_without_branch(
            source=envelope.source,
            baseline_revision=envelope.baseline_revision,
            max_evidence_bytes=envelope.limits.max_evidence_bytes,
            status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            branch_id=envelope.branch_id,
            detail_code=error.detail_code,
        )
    if copied.authority is not None and (
        not isinstance(
            source._branch_authority_resolver,
            RemoteWorkspaceBranchAuthorityProvider,
        )
        or source._branch_claim_scope is not WorkspaceBranchBindingAuthorityClaimScope.DURABLE
    ):
        raise RuntimeError(
            "Durable remote workspace branches require complete cross-process invocation "
            "authority claims."
        )
    baseline_revision = copied.baseline.revision
    if baseline_revision is None:  # pragma: no cover - WorkspaceBranchRequest validates this
        raise AssertionError("Workspace branch baseline revision disappeared.")
    branch_id = copied.branch_id or f"wsb_remote_{secrets.token_hex(16)}"
    evidence_limit_violation = _workspace_branch_fixed_authority_evidence_limit_violation(
        source=copied.baseline.identity,
        baseline_revision=baseline_revision,
        branch_id=branch_id,
        limits=copied.limits,
        created_detail_code="remote_workspace_branch_created",
    )
    if evidence_limit_violation is not None:
        return _creation_without_branch(
            source=copied.baseline.identity,
            baseline_revision=baseline_revision,
            max_evidence_bytes=copied.limits.max_evidence_bytes,
            status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
            branch_id=branch_id,
            detail_code=evidence_limit_violation,
        )
    baseline = {entry.path: {"sha256": entry.content_sha256} for entry in copied.baseline.paths}
    capability = source._branch_capability
    if capability is None:  # pragma: no cover - checked above
        raise AssertionError("Remote workspace branch capability disappeared.")
    payload = {
        "branch_id": branch_id,
        "source": copied.baseline.identity.model_dump(mode="json", warnings=False),
        "baseline_revision": baseline_revision,
        "baseline": baseline,
        "limits": copied.limits.model_dump(mode="json", warnings=False),
        "allocation_fingerprint": capability.allocation_fingerprint,
        "idempotency_key": copied.idempotency_key,
        "authority": _authority_payload(copied.authority),
    }
    claim = _claim_binding(source, copied.authority)
    settlement = _RunnerOperationSettlement()
    primary: BaseException | None = None
    try:
        try:
            result = await _run(
                source,
                "create",
                payload,
                output_limit_bytes=copied.limits.max_evidence_bytes + _OUTPUT_HEADROOM_BYTES,
                settlement=settlement,
            )
        except _RemoteBranchSourceConflict:
            return _creation_without_branch(
                source=copied.baseline.identity,
                baseline_revision=baseline_revision,
                max_evidence_bytes=copied.limits.max_evidence_bytes,
                status=WorkspaceBranchOutcomeStatus.CONFLICTED,
                branch_id=branch_id,
                detail_code="workspace_branch_baseline_conflicted",
            )
        except WorkspaceBranchResourceExhaustedError as error:
            return _creation_without_branch(
                source=copied.baseline.identity,
                baseline_revision=baseline_revision,
                max_evidence_bytes=copied.limits.max_evidence_bytes,
                status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                branch_id=branch_id,
                detail_code=error.detail_code,
            )
    except BaseException as error:
        primary = error
        raise
    finally:
        await _settle_and_release_claim(
            source,
            claim,
            settlement,
            primary=primary,
        )
    status = result.get("status")
    if status == "conflicted":
        conflicts = _conflicts(result.get("conflicts"))
        conflicted = _creation_without_branch(
            source=copied.baseline.identity,
            baseline_revision=baseline_revision,
            max_evidence_bytes=copied.limits.max_evidence_bytes,
            status=WorkspaceBranchOutcomeStatus.CONFLICTED,
            branch_id=branch_id,
            detail_code="workspace_branch_baseline_conflicted",
            conflicts=conflicts,
        )
        try:
            _validate_conflict_evidence_limit(
                conflicts,
                evidence=conflicted.evidence,
                max_bytes=copied.limits.max_evidence_bytes,
            )
        except WorkspaceBranchResourceExhaustedError as error:
            return _creation_without_branch(
                source=copied.baseline.identity,
                baseline_revision=baseline_revision,
                max_evidence_bytes=copied.limits.max_evidence_bytes,
                status=WorkspaceBranchOutcomeStatus.RESOURCE_EXHAUSTED,
                branch_id=branch_id,
                detail_code=error.detail_code,
            )
        return conflicted
    if status != "created" or result.get("branch_id") != branch_id:
        raise WorkspaceBranchFencedError("Remote workspace branch creation result is invalid.")
    branch = RunnerWorkspaceBranch(
        source=source,
        branch_id=branch_id,
        source_identity=copied.baseline.identity,
        baseline_revision=baseline_revision,
        limits=copied.limits,
        authority=copied.authority,
    )
    return WorkspaceBranchCreationResult(
        status=WorkspaceBranchOutcomeStatus.CREATED,
        branch=branch,
        evidence=_bounded_workspace_branch_evidence(
            source=copied.baseline.identity,
            outcome=WorkspaceBranchOutcomeStatus.CREATED,
            baseline_revision=baseline_revision,
            branch_id=branch_id,
            detail_code="remote_workspace_branch_created",
            max_bytes=copied.limits.max_evidence_bytes,
        ),
    )


async def recover_runner_workspace_branch(
    source: RunnerWorkspace,
    request: WorkspaceBranchRecoveryRequest,
) -> WorkspaceBranchRecoveryResult:
    from cayu.workspaces.runner import RunnerWorkspace

    if type(source) is not RunnerWorkspace:
        raise RuntimeError("Remote workspace branch recovery is unsupported.")
    if source._branch_capability is None:
        raise RuntimeError("Remote workspace branch recovery is unsupported.")
    copied_request = _copy_recovery_request(request)
    resolver = source._branch_authority_resolver
    if (
        not isinstance(resolver, RemoteWorkspaceBranchAuthorityProvider)
        or source._branch_claim_scope is not WorkspaceBranchBindingAuthorityClaimScope.DURABLE
    ):
        raise RuntimeError(
            "Durable remote workspace branches require complete cross-process invocation "
            "authority claims."
        )
    _retry_retained_claim_releases()
    current_authority = _copy_workspace_branch_authority(
        resolver.current_operation_authority(copied_request.session_id)
    )
    current = _binding_authority(current_authority)
    if (
        current_authority.expected_run_epoch != copied_request.expected_run_epoch
        or current.binding_generation != copied_request.binding_generation
        or current.binding_identity != copied_request.binding_identity
    ):
        raise WorkspaceBranchOperationConflict(
            "Workspace branch recovery invocation authority is no longer current."
        )
    claim = resolver.claim_operation(current_authority)
    capability = source._branch_capability
    payload = {
        "branch_id": copied_request.branch_id,
        "allocation_fingerprint": capability.allocation_fingerprint,
        "binding_authority": current.model_dump(mode="json", warnings=False),
        "operation_authority": current_authority.model_dump(mode="json", warnings=False),
        "request": copied_request.model_dump(mode="json", warnings=False),
    }
    settlement = _RunnerOperationSettlement()
    primary: BaseException | None = None
    try:
        result = await _run(
            source,
            "recover",
            payload,
            output_limit_bytes=_MAX_EVIDENCE_BYTES + _OUTPUT_HEADROOM_BYTES,
            settlement=settlement,
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        await _settle_and_release_claim(
            source,
            claim,
            settlement,
            primary=primary,
        )
    try:
        state = WorkspaceBranchDurableState(result.get("state"))
        source_identity = WorkspaceIdentity.model_validate(result.get("source"))
        authority = WorkspaceBranchAuthority.model_validate(result.get("authority"))
        from cayu.workspaces.branches import WorkspaceBranchLimits

        limits = WorkspaceBranchLimits.model_validate(result.get("limits"))
    except Exception as exc:
        raise WorkspaceBranchFencedError(
            "Remote workspace branch recovery result is invalid."
        ) from exc
    baseline_revision = result.get("baseline_revision")
    returned_branch_id = result.get("branch_id")
    if (
        returned_branch_id != copied_request.branch_id
        or source_identity.workspace_id != source.id
        or type(baseline_revision) is not str
        or not baseline_revision
        or authority.session_id != copied_request.session_id
        or authority.expected_run_epoch != copied_request.expected_run_epoch
        or authority != current_authority
    ):
        raise WorkspaceBranchFencedError("Remote workspace branch recovery result is invalid.")
    evidence_limit_violation = _workspace_branch_fixed_authority_evidence_limit_violation(
        source=source_identity,
        baseline_revision=baseline_revision,
        branch_id=copied_request.branch_id,
        limits=limits,
        created_detail_code="remote_workspace_branch_recovered",
    )
    if evidence_limit_violation is not None:
        raise WorkspaceBranchFencedError(
            "Remote workspace branch recovery evidence exceeds its durable limit."
        )
    branch: RunnerWorkspaceBranch | None = None
    publication: WorkspaceBranchPublicationResult | None = None
    rollback: WorkspaceBranchRollbackResult | None = None
    if state in {WorkspaceBranchDurableState.OPEN, WorkspaceBranchDurableState.CONFLICTED}:
        branch = RunnerWorkspaceBranch(
            source=source,
            branch_id=copied_request.branch_id,
            source_identity=source_identity,
            baseline_revision=baseline_revision,
            limits=limits,
            authority=authority,
        )
        outcome = WorkspaceBranchOutcomeStatus.CREATED
        digest = None
        paths: tuple[str, ...] = ()
    elif state is WorkspaceBranchDurableState.COMMITTED:
        change_set = _change_set(
            branch_id=copied_request.branch_id,
            source=source_identity,
            baseline_revision=baseline_revision,
            payload=result,
        )
        outcome = WorkspaceBranchOutcomeStatus.COMMITTED
        digest = change_set.digest
        paths = tuple(change.path for change in change_set.changes)
    elif state in {WorkspaceBranchDurableState.ROLLED_BACK, WorkspaceBranchDurableState.EXPIRED}:
        outcome = WorkspaceBranchOutcomeStatus(state.value)
        digest = _validated_change_set_digest(result.get("digest"))
        paths = ()
    elif state is WorkspaceBranchDurableState.AMBIGUOUS:
        outcome = WorkspaceBranchOutcomeStatus.AMBIGUOUS
        digest = None
        paths = ()
    else:
        outcome = WorkspaceBranchOutcomeStatus.FAILED
        digest = None
        paths = ()
    evidence = _bounded_workspace_branch_evidence(
        source=source_identity,
        outcome=outcome,
        baseline_revision=baseline_revision,
        branch_id=copied_request.branch_id,
        change_set_digest=digest,
        paths=paths,
        detail_code=_remote_detail_code(
            result.get("detail_code"),
            fallback="remote_workspace_branch_recovered",
        ),
        max_bytes=limits.max_evidence_bytes,
    )
    if state is WorkspaceBranchDurableState.COMMITTED:
        publication = WorkspaceBranchPublicationResult(status=outcome, evidence=evidence)
    elif state in {WorkspaceBranchDurableState.ROLLED_BACK, WorkspaceBranchDurableState.EXPIRED}:
        rollback = WorkspaceBranchRollbackResult(status=outcome, evidence=evidence)
    return WorkspaceBranchRecoveryResult(
        state=state,
        evidence=evidence,
        branch=branch,
        publication=publication,
        rollback=rollback,
    )


__all__ = [
    "RunnerWorkspaceBranch",
    "create_runner_workspace_branch",
    "recover_runner_workspace_branch",
]
