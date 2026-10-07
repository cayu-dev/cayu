"""Runtime-owned deterministic completion-verifier execution boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from time import monotonic
from typing import ClassVar, NoReturn, TypeVar, cast
from uuid import uuid4

from pydantic import BaseModel

from cayu._exception_groups import exception_group_children
from cayu._task_wait import (
    CapturedAwaitableOutcome,
    await_shielded_task_outcome,
    capture_awaitable_outcome,
    restore_task_cancellation_requests,
)
from cayu._validation import require_durable_clean_nonblank, revalidate_model_input
from cayu.approvals.tools import ResolutionActor
from cayu.deadlines import ExecutionDeadline, effective_deadline
from cayu.execution_profiles import (
    EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS,
    ExecutionProfileAdoptionIntent,
    ExecutionProfileAuthorityDecision,
    ExecutionProfilePolicyAction,
    ExecutionProfilePolicyResult,
    copy_execution_profile_policy_result,
)
from cayu.runtime._diagnostics import (
    MAX_DIAGNOSTIC_UTF8_BYTES,
    credential_safe_runtime_exception,
    credential_safe_runtime_exception_group,
    exception_diagnostic,
)
from cayu.runtime._execution_profile_identity_validation import (
    copy_secret_free_execution_profile_behavior_identity,
)
from cayu.runtime._runtime_records import RegisteredProvider
from cayu.runtime._task_store_operation_boundary import (
    TaskStoreOperationOutcome,
    capture_sensitive_validation,
    capture_task_store_operation,
    raise_task_store_operation_failure,
    task_store_completion_verifier_dispatch_capability_is_complete,
)
from cayu.tasks._verified_work_authority import (
    completion_decision_claim_authority_matches,
    completion_decision_request_from_record,
    require_completion_decision_integrity,
    require_completion_proposal_integrity,
    require_completion_verifier_profile_integrity,
)
from cayu.tasks.base import TaskClaimLost, TaskStore
from cayu.tasks.completion_verifier_profiles import (
    CompletionVerifierExecutionProfile,
    CompletionVerifierProfileAdoptionDecision,
    CompletionVerifierProfileComponentDeclaration,
    CompletionVerifierProfilePolicy,
    CompletionVerifierProfilePolicyRequest,
    CompletionVerifierProfilePreparationRequest,
    CompletionVerifierProfileRecord,
    build_completion_verifier_execution_profile,
    changed_completion_verifier_profile_components,
    completion_verifier_profile_adoption_request_sha256,
    completion_verifier_profile_preparation_request_sha256,
    copy_completion_verifier_profile_record,
)
from cayu.tasks.contracts import (
    CompletionDecision,
    CompletionDecisionCreate,
    CompletionProposal,
    CompletionVerificationClaim,
    CompletionVerificationClaimLost,
    CompletionVerificationClaimRequest,
    CompletionVerifierDecision,
    CompletionVerifierKind,
    CompletionVerifierRef,
    TaskCompletionDecisionRequired,
    WorkAttempt,
    WorkCompletionConflict,
    WorkContract,
    WorkContractConflict,
    completion_decision_request_sha256,
    completion_gap_fingerprint,
    completion_verification_claim_request_sha256,
    copy_completion_decision,
    copy_completion_proposal,
    copy_completion_verification_claim,
    copy_completion_verifier_decision,
    copy_work_attempt,
    copy_work_contract,
    validate_completion_decision_contract,
)
from cayu.vaults.redaction import SecretRedactor
from cayu.verification._leased_adapter_heartbeat import run_leased_adapter_heartbeat
from cayu.verification._leased_adapter_runner import (
    LeasedAdapterLease,
    LeasedAdapterRunner,
    LeasedAdapterSettlement,
)
from cayu.verification._provider_completion_verifier_profile import (
    RUNTIME_COMPONENT_PREFIX,
    provider_verifier_runtime_components,
)
from cayu.verification._provider_completion_verifier_runtime import (
    ProviderCompletionVerifierRuntime,
    ProviderVerifierExecutionAuthority,
)
from cayu.verification.completion_verifiers import (
    CompletionVerifierExecutionError,
    CompletionVerifierExecutionRequest,
    CompletionVerifierRequest,
    CompletionVerifierUnavailable,
    DeterministicCompletionVerifier,
    copy_completion_verifier_execution_request,
)
from cayu.verification.provider_completion_verifiers import (
    ProviderCompletionVerifier,
    ProviderCompletionVerifierBudgetExhausted,
    ProviderCompletionVerifierDecodingError,
    ProviderCompletionVerifierDispatchError,
    ProviderCompletionVerifierTarget,
    copy_provider_completion_verifier_target,
)
from cayu.workspaces.observation_recovery import (
    retain_workspace_observation_pending_cancellation_requests,
    workspace_observation_pending_cancellation_requests,
)

_RUNTIME_AUTHORED_VERIFIER_FAILURES = frozenset(
    {
        CompletionVerifierUnavailable,
        ProviderCompletionVerifierBudgetExhausted,
        ProviderCompletionVerifierDecodingError,
        ProviderCompletionVerifierDispatchError,
    }
)
_ModelT = TypeVar("_ModelT", bound=BaseModel)
_ExecutionKey = str
_MAX_ACTIVE_COMPLETION_VERIFIERS = 64
_BASE_EXCEPTION_ARGS_DESCRIPTOR = BaseException.__dict__["args"]
_BASE_EXCEPTION_CAUSE_DESCRIPTOR = BaseException.__dict__["__cause__"]


class _ClaimHeartbeatCancellationMarker:
    """Per-heartbeat provenance for cancellation forwarded to its adapter."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<completion-verifier-heartbeat-cancellation>"

    __str__ = __repr__


class _ClaimHeartbeatShutdownMarker:
    """Per-heartbeat provenance for runtime-owned heartbeat shutdown."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<completion-verifier-heartbeat-shutdown>"

    __str__ = __repr__


class _AdapterDrainCancellationMarker:
    """Identity of the shutdown request owned by one retained adapter."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<completion-verifier-adapter-shutdown>"

    __str__ = __repr__


@dataclass(frozen=True, slots=True)
class _VerifierAuthority:
    proposal: CompletionProposal = field(repr=False)
    attempt: WorkAttempt = field(repr=False)
    contract: WorkContract = field(repr=False)
    adapter_request: CompletionVerifierRequest = field(repr=False)


@dataclass(frozen=True, slots=True)
class _RegisteredVerifier:
    adapter: DeterministicCompletionVerifier | ProviderCompletionVerifier = field(repr=False)
    profile: CompletionVerifierExecutionProfile
    target: ProviderCompletionVerifierTarget | None = None


@dataclass(frozen=True, slots=True)
class _TaskStoreOwner:
    store: TaskStore = field(repr=False)


@dataclass(slots=True)
class _ClaimHeartbeatState:
    ownership_lost: asyncio.Future[BaseException]
    late_settlement_failure: BaseException | None = None


@dataclass(slots=True)
class _ClaimLeaseHeartbeat:
    coordinator: CompletionVerifierCoordinator = field(repr=False)
    store_owner: _TaskStoreOwner = field(repr=False)
    request: CompletionVerificationClaimRequest = field(repr=False)
    execution: CompletionVerifierExecutionRequest = field(repr=False)
    claim: CompletionVerificationClaim = field(repr=False)
    deadline: float
    state: _ClaimHeartbeatState = field(repr=False)

    renewal_task_name: ClassVar[str] = "cayu-completion-verifier-claim-renewal"
    cancel_renewal_on_shutdown: ClassVar[bool] = True
    deadline_can_extend: ClassVar[bool] = False

    @property
    def interval_seconds(self) -> float:
        return min(max(self.request.lease_seconds / 4.0, 0.05), 30.0)

    async def renew(self) -> tuple[CompletionVerificationClaim, float]:
        return await self.coordinator._renew_claim(
            self.store_owner, self.request, self.execution, prior_claim=self.claim
        )

    def acknowledge(
        self, result: tuple[CompletionVerificationClaim, float] | None, started: float
    ) -> None:
        if result is None:
            raise WorkCompletionConflict(
                "Completion verification claim renewal returned no result."
            ) from None
        self.claim, self.deadline = result

    def expiry_failure(self, *, renewing: bool) -> BaseException:
        return CompletionVerificationClaimLost(
            "Completion verification claim renewal was not acknowledged before "
            "its local lease deadline."
            if renewing
            else "Completion verification claim acknowledgement expired before renewal."
        )

    def safe_cancellation(self, cancellation: asyncio.CancelledError) -> asyncio.CancelledError:
        return _safe_caller_cancellation(cancellation, redactor=self.coordinator._secret_redactor)

    def raise_late_failure(self, expiry: BaseException, failure: BaseException) -> NoReturn:
        detached = _detached_verifier_failure(failure, redactor=self.coordinator._secret_redactor)
        self.state.late_settlement_failure = detached
        raise expiry from detached

    def raise_cancellation(self, cancellation: asyncio.CancelledError) -> NoReturn:
        try:
            evidence = _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__get__(cancellation, BaseException)
        except BaseException:
            evidence = RuntimeError(
                "Completion-verification claim renewal settlement evidence was inaccessible."
            )
        if evidence is None:
            raise cancellation
        # Retain detached cleanup evidence before Task cancellation can erase
        # its cause. Claim settlement prunes only its owned shutdown signal.
        cancellation.__cause__ = None
        raise BaseExceptionGroup(
            "Completion-verification claim renewal was cancelled during settlement.",
            [evidence, cancellation],
        ) from None


@dataclass(slots=True)
class _ClaimHeartbeat:
    stop: asyncio.Event
    task: asyncio.Task[None]
    state: _ClaimHeartbeatState
    capacity_reservation: object
    cancellation_marker: _ClaimHeartbeatCancellationMarker
    shutdown_marker: _ClaimHeartbeatShutdownMarker
    retained_for_drain: bool = False
    observed_failure_id: int | None = None
    shutdown_requested: bool = False

    @property
    def ownership_lost(self) -> asyncio.Future[BaseException]:
        return self.state.ownership_lost


@dataclass(frozen=True, slots=True)
class _ClaimHeartbeatSettlement:
    failure: BaseException | None = None
    caller_cancellation: asyncio.CancelledError | None = None
    cancellation_requests_consumed: int = 0


@dataclass(slots=True)
class _DrainingAdapter(LeasedAdapterSettlement[_ClaimHeartbeatSettlement]):
    operation_key: _ExecutionKey
    task: asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]]
    heartbeat: _ClaimHeartbeat
    cancellation_marker: _AdapterDrainCancellationMarker
    observed_adapter_failures: tuple[BaseException, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class CompletionVerifierOwnedExecution:
    """Private exact invocation handle; primary and late outcomes are separate.

    Cancelling a settlement waiter does not cancel the original operation,
    adapter, or heartbeat. Keep this handle and dependent stores until settlement.
    """

    operation: asyncio.Task[CapturedAwaitableOutcome[CompletionDecision]] = field(repr=False)
    _drain: asyncio.Future[_DrainingAdapter | None] = field(repr=False)
    _coordinator: CompletionVerifierCoordinator = field(repr=False)

    def acknowledge_settlement(self) -> None:
        """Consume only this execution's observed cleanup, never a successor's."""
        if not self.operation.done() or not self._drain.done():
            raise RuntimeError("Completion verification has not settled.")
        draining = self._drain.result()
        if draining is not None:
            self._coordinator._acknowledge_owned_drain(draining)

    async def settlement(self) -> _ClaimHeartbeatSettlement:
        async def wait_for_terminal(task):
            outcome = await await_shielded_task_outcome(task, timeout_after_cancellation_s=0)
            if outcome.cancellation is not None:
                restore_task_cancellation_requests(
                    outcome.cancellation_requests_consumed, cancellation=outcome.cancellation
                )
                raise outcome.cancellation

        # Primary task cancellation is historical here. A new cancellation of
        # this waiter remains a real cancellation, without cancelling its owner.
        await wait_for_terminal(self.operation)
        draining = await asyncio.shield(self._drain)
        if draining is None:
            return _ClaimHeartbeatSettlement()
        await wait_for_terminal(draining.task)
        settled = self._coordinator._ensure_draining_adapter_settlement(
            draining.operation_key, draining
        )
        return await asyncio.shield(settled)


def _verifier_key(reference: CompletionVerifierRef) -> tuple[str, str, str, str]:
    return (
        reference.verifier_id,
        reference.version,
        reference.kind.value,
        reference.configuration_fingerprint,
    )


def _contains_workload_secret(
    values: tuple[str, ...],
    *,
    redactor: SecretRedactor,
) -> bool:
    return any(redactor.redact_text(value) != value for value in values)


def _copy_exact_model(
    value: object,
    model_type: type[_ModelT],
    *,
    operation_name: str,
    redactor: SecretRedactor,
) -> TaskStoreOperationOutcome[_ModelT]:
    if type(value) is not model_type:
        return TaskStoreOperationOutcome()
    copier: Callable[[object], object]
    if model_type is WorkContract:
        copier = cast("Callable[[object], object]", copy_work_contract)
    elif model_type is WorkAttempt:
        copier = cast("Callable[[object], object]", copy_work_attempt)
    elif model_type is CompletionProposal:
        copier = cast("Callable[[object], object]", copy_completion_proposal)
    elif model_type is CompletionVerificationClaim:
        copier = cast("Callable[[object], object]", copy_completion_verification_claim)
    elif model_type is CompletionDecision:
        copier = cast("Callable[[object], object]", copy_completion_decision)
    else:

        def copier(candidate: object) -> object:
            return revalidate_model_input(candidate, model_type)

    return capture_sensitive_validation(
        lambda: cast("_ModelT", copier(value)),
        operation_name=operation_name,
        redactor=redactor,
    )


def _validate_decision_contract(
    contract: WorkContract,
    request: CompletionDecisionCreate,
) -> bool:
    validate_completion_decision_contract(contract, request)
    return True


def _detached_verifier_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException:
    """Detach an adapter failure from its traceback, context, and argument graph."""

    if isinstance(error, BaseExceptionGroup):
        return credential_safe_runtime_exception_group(
            error,
            group_message="Completion verifier reported multiple failures.",
            leaf_mapper=lambda leaf: _detached_verifier_failure(leaf, redactor=redactor),
            invalid_leaf_factory=lambda: CompletionVerifierExecutionError(
                "Completion verifier reported invalid failure evidence."
            ),
            truncated_leaf_factory=lambda: CompletionVerifierExecutionError(
                "Additional completion verifier failures were omitted."
            ),
            fallback_leaf_mapper=lambda leaf: _generic_verifier_failure(
                leaf,
                redactor=redactor,
            ),
            redactor=redactor,
        )
    diagnostic = exception_diagnostic(
        error,
        empty_message="completion verifier failed",
        nonportable_message="Completion verifier failed with a non-portable diagnostic.",
        redactor=redactor,
    )
    message = diagnostic.message
    if isinstance(error, GeneratorExit):
        return credential_safe_runtime_exception(
            GeneratorExit,
            message,
            redactor=redactor,
            fallback_message="Completion verifier process-control diagnostic was redacted.",
        )
    if isinstance(error, KeyboardInterrupt):
        return credential_safe_runtime_exception(
            KeyboardInterrupt,
            message,
            redactor=redactor,
            fallback_message="Completion verifier process-control diagnostic was redacted.",
        )
    if isinstance(error, SystemExit):
        return credential_safe_runtime_exception(
            SystemExit,
            message,
            redactor=redactor,
            fallback_message="Completion verifier process-control diagnostic was redacted.",
        )
    if isinstance(error, asyncio.CancelledError):
        return credential_safe_runtime_exception(
            CompletionVerifierExecutionError,
            "Completion verifier was cancelled without caller cancellation.",
            redactor=redactor,
            fallback_message="Completion verifier cancellation diagnostic was redacted.",
        )
    if type(error) in _RUNTIME_AUTHORED_VERIFIER_FAILURES:
        # Provider-verifier failures carry runtime-authored messages only.
        return credential_safe_runtime_exception(
            type(error),
            message,
            redactor=redactor,
            fallback_message="Completion verifier failure diagnostic was redacted.",
        )
    return credential_safe_runtime_exception(
        CompletionVerifierExecutionError,
        redactor.redact_text_bounded(
            f"{diagnostic.error_type}: {message}",
            max_bytes=MAX_DIAGNOSTIC_UTF8_BYTES,
        ),
        redactor=redactor,
        fallback_message="Completion verifier failure diagnostic was redacted.",
    )


def _generic_verifier_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException:
    """Retain runtime failure classification without retaining diagnostic text."""

    if isinstance(error, GeneratorExit):
        safe_type: type[BaseException] = GeneratorExit
    elif isinstance(error, KeyboardInterrupt):
        safe_type = KeyboardInterrupt
    elif isinstance(error, SystemExit):
        safe_type = SystemExit
    elif isinstance(error, asyncio.CancelledError):
        safe_type = asyncio.CancelledError
    elif type(error) in _RUNTIME_AUTHORED_VERIFIER_FAILURES or type(error) in {
        CompletionVerifierExecutionError,
        CompletionVerifierUnavailable,
        TaskClaimLost,
        CompletionVerificationClaimLost,
        TaskCompletionDecisionRequired,
        WorkCompletionConflict,
        WorkContractConflict,
        TimeoutError,
        ConnectionError,
        ValueError,
        TypeError,
        NotImplementedError,
        RuntimeError,
    }:
        safe_type = type(error)
    elif isinstance(error, Exception):
        safe_type = CompletionVerifierExecutionError
    else:
        safe_type = BaseException
    return credential_safe_runtime_exception(
        safe_type,
        "Completion verifier failure diagnostic was redacted.",
        redactor=redactor,
        fallback_message="Completion verifier failure diagnostic was withheld.",
    )


def _is_claim_heartbeat_cancellation(
    error: BaseException | None,
    marker: _ClaimHeartbeatCancellationMarker,
) -> bool:
    """Recognize only the cancellation minted for one exact heartbeat failure."""

    return isinstance(error, asyncio.CancelledError) and _is_exception_marker(error, marker)


def _is_exception_marker(error: BaseException, marker: object) -> bool:
    """Read one base-owned exception argument without extension dispatch."""

    if not isinstance(error, BaseException):
        return False
    try:
        args = _BASE_EXCEPTION_ARGS_DESCRIPTOR.__get__(error, BaseException)
    except BaseException:
        return False
    return type(args) is tuple and len(args) == 1 and args[0] is marker


def _exception_tree_has_cancellation(error: BaseException) -> bool:
    pending = [error]
    observed: set[int] = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in observed:
            continue
        observed.add(id(candidate))
        if isinstance(candidate, asyncio.CancelledError):
            return True
        if isinstance(candidate, BaseExceptionGroup):
            children = exception_group_children(candidate)
            if children is not None:
                pending.extend(children)
    return False


def _adapter_failure_after_heartbeat(
    error: BaseException | None,
    *,
    heartbeat: _ClaimHeartbeat,
    heartbeat_failure: BaseException | None,
) -> BaseException | None:
    """Discard only the control signal caused by the preserved heartbeat failure."""

    if heartbeat_failure is not None and error is not None:
        return _prune_exception_graph(
            error,
            should_prune=lambda leaf: _is_claim_heartbeat_cancellation(
                leaf,
                heartbeat.cancellation_marker,
            ),
        )
    return error


def _prune_exception_graph(
    error: BaseException,
    *,
    should_prune: Callable[[BaseException], bool],
    pruned_leaf_replacement: Callable[[BaseException], BaseException | None] | None = None,
) -> BaseException | None:
    """Remove authenticated leaves without recursive or extension-owned traversal."""

    next_token = 1
    pending: list[tuple[int, BaseException, bool]] = [(0, error, False)]
    child_tokens_by_group: dict[int, tuple[int, ...]] = {}
    rebuilt_by_token: dict[int, BaseException | None] = {}
    observed: set[int] = set()
    while pending:
        token, candidate, expanded = pending.pop()
        if expanded:
            children = [
                rebuilt_by_token.pop(child_token, None)
                for child_token in child_tokens_by_group.pop(token, ())
            ]
            retained = [child for child in children if child is not None]
            rebuilt_by_token[token] = (
                BaseExceptionGroup(
                    "Completion verifier reported multiple failures.",
                    retained,
                )
                if retained
                else None
            )
            continue

        candidate_id = id(candidate)
        if candidate_id in observed:
            rebuilt_by_token[token] = None
            continue
        observed.add(candidate_id)
        if not isinstance(candidate, BaseExceptionGroup):
            try:
                prune = should_prune(candidate)
            except BaseException:
                prune = False
            replacement: BaseException | None = None
            if prune and pruned_leaf_replacement is not None:
                try:
                    replacement = pruned_leaf_replacement(candidate)
                except BaseException:
                    replacement = None
            rebuilt_by_token[token] = replacement if prune else candidate
            continue
        children = exception_group_children(candidate)
        if children is None:
            rebuilt_by_token[token] = candidate
            continue
        child_tokens = tuple(range(next_token, next_token + len(children)))
        next_token += len(children)
        child_tokens_by_group[token] = child_tokens
        pending.append((token, candidate, True))
        pending.extend(
            (child_token, child, False)
            for child_token, child in reversed(tuple(zip(child_tokens, children, strict=True)))
        )
    return rebuilt_by_token.get(0)


def _ordered_execution_failures(
    *,
    adapter_failures: tuple[BaseException | None, ...] = (),
    boundary_failures: tuple[BaseException | None, ...] = (),
    redactor: SecretRedactor,
) -> tuple[BaseException, ...]:
    """Detach adapter failures and preserve every distinct boundary failure once."""

    ordered: list[BaseException] = []
    observed: set[int] = set()
    for failure in adapter_failures:
        if failure is None or id(failure) in observed:
            continue
        observed.add(id(failure))
        ordered.append(_detached_verifier_failure(failure, redactor=redactor))
    for failure in boundary_failures:
        if failure is None or id(failure) in observed:
            continue
        observed.add(id(failure))
        ordered.append(failure)
    return tuple(ordered)


def _safe_caller_cancellation(
    cancellation: asyncio.CancelledError,
    *,
    redactor: SecretRedactor,
) -> asyncio.CancelledError:
    diagnostic = exception_diagnostic(
        cancellation,
        empty_message="completion verifier execution cancelled",
        nonportable_message="Completion verifier execution cancellation had a non-portable diagnostic.",
        redactor=redactor,
    )
    return credential_safe_runtime_exception(
        asyncio.CancelledError,
        diagnostic.message,
        redactor=redactor,
        fallback_message="Completion verifier cancellation diagnostic was redacted.",
    )


def _ordered_failure_group(
    message: str,
    *failures: BaseException,
    redactor: SecretRedactor,
) -> BaseExceptionGroup:
    """Preserve distinct failures in operation order and cancellation authority."""

    ordered: list[BaseException] = []
    for failure in failures:
        if all(failure is not existing for existing in ordered):
            ordered.append(failure)
    group = (
        ExceptionGroup(message, cast("list[Exception]", ordered))
        if all(isinstance(failure, Exception) for failure in ordered)
        else BaseExceptionGroup(message, ordered)
    )
    pending_cancellation_requests = max(
        (workspace_observation_pending_cancellation_requests(failure) for failure in ordered),
        default=0,
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            group,
            pending_cancellation_requests,
        )
    group = _credential_safe_verifier_group(
        group,
        group_message=message,
        redactor=redactor,
    )
    if pending_cancellation_requests:
        retain_workspace_observation_pending_cancellation_requests(
            group,
            pending_cancellation_requests,
        )
    return group


def _credential_safe_verifier_group(
    error: BaseExceptionGroup,
    *,
    group_message: str,
    redactor: SecretRedactor,
) -> BaseExceptionGroup:
    """Validate one group after every runtime-owned composition step."""

    return credential_safe_runtime_exception_group(
        error,
        group_message=group_message,
        leaf_mapper=lambda leaf: leaf,
        invalid_leaf_factory=lambda: CompletionVerifierExecutionError(
            "Concurrent failure group was invalid."
        ),
        truncated_leaf_factory=lambda: CompletionVerifierExecutionError(
            "Additional concurrent failures were omitted."
        ),
        fallback_leaf_mapper=lambda leaf: _generic_verifier_failure(
            leaf,
            redactor=redactor,
        ),
        redactor=redactor,
    )


def _copy_verifier_deadline(value: ExecutionDeadline | None) -> ExecutionDeadline:
    if value is None:
        return ExecutionDeadline()
    if type(value) is not ExecutionDeadline:
        raise TypeError("Verifier deadline must be an ExecutionDeadline.")
    copied = ExecutionDeadline.model_validate(value.model_dump(mode="python", warnings=False))
    return effective_deadline(value, copied)


class CompletionVerifierCoordinator:
    """Resolve application adapters and bind their output to durable authority."""

    def __init__(
        self,
        *,
        task_store: TaskStore | None,
        secret_redactor: SecretRedactor,
        profile_policy: CompletionVerifierProfilePolicy | None = None,
        provider_registrations: Callable[[], Mapping[str, RegisteredProvider]] | None = None,
        execution_profile_process_identity: str | None = None,
    ) -> None:
        if task_store is not None and not isinstance(task_store, TaskStore):
            raise TypeError("Completion verifier coordinator requires a TaskStore.")
        if not isinstance(secret_redactor, SecretRedactor):
            raise TypeError("Completion verifier coordinator requires a SecretRedactor.")
        if profile_policy is not None and not isinstance(
            profile_policy,
            CompletionVerifierProfilePolicy,
        ):
            raise TypeError(
                "Completion verifier profile policy must be a CompletionVerifierProfilePolicy."
            )
        self._task_store = task_store
        self._secret_redactor = secret_redactor
        self._profile_policy = profile_policy
        self._profile_policy_identity: str | None = None
        if profile_policy is not None:
            policy_identity_validation = capture_sensitive_validation(
                lambda policy=profile_policy: self._copy_profile_policy_identity(policy),
                operation_name="Completion verifier profile-policy identity validation",
                redactor=self._secret_redactor,
            )
            if policy_identity_validation.failure is not None:
                raise_task_store_operation_failure(policy_identity_validation.failure)
            profile_policy_identity = policy_identity_validation.result
            if profile_policy_identity is None:
                raise ValueError(
                    "Completion verifier profile-policy identity is invalid."
                ) from None
            self._profile_policy_identity = profile_policy_identity
        self._execution_owner_id = f"cver_{uuid4().hex}"
        self._verifiers: dict[tuple[str, str, str, str], _RegisteredVerifier] = {}
        self._adapter_runner = LeasedAdapterRunner[_DrainingAdapter]()
        self._provider_registrations = provider_registrations
        self._process_identity = execution_profile_process_identity or uuid4().hex
        self._provider_runtime = ProviderCompletionVerifierRuntime(
            resolve_provider=self._resolve_provider,
            secret_redactor=secret_redactor,
        )

    def _resolve_provider(self, provider_name: str) -> RegisteredProvider:
        registrations = (
            None if self._provider_registrations is None else self._provider_registrations()
        )
        registered = None if registrations is None else registrations.get(provider_name)
        if registered is None:
            raise CompletionVerifierUnavailable(
                "The provider required by the provider-backed completion verifier is not "
                "registered."
            ) from None
        return registered

    def _provider_runtime_components(
        self,
        target: ProviderCompletionVerifierTarget,
    ) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
        return provider_verifier_runtime_components(
            target=target,
            registered_provider=self._resolve_provider(target.provider_name),
            process_identity=self._process_identity,
            redactor=self._secret_redactor,
        )

    @property
    def pending(self) -> bool:
        """Whether retained adapters, heartbeats or settlements are still running."""

        return self._adapter_runner.pending

    async def drain(self, *, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for retained work, without cancelling it."""

        return await self._adapter_runner.wait_idle(timeout_s)

    def report_retained_failures(self) -> list[BaseException]:
        """Return, once each, retained executions whose settlement failed."""

        return [
            failure
            if failure is not None
            else CompletionVerifierExecutionError(
                "A retained completion-verifier execution ended without a settlement."
            )
            for failure in self._adapter_runner.report_settlement_failures()
        ]

    def _copy_profile_policy_identity(
        self,
        policy: CompletionVerifierProfilePolicy,
    ) -> str:
        identity = require_durable_clean_nonblank(
            policy.identity,
            "completion_verifier_profile_policy.identity",
        )
        if len(identity.encode("utf-8")) > 256:
            raise ValueError(
                "completion_verifier_profile_policy.identity must be at most 256 UTF-8 bytes."
            )
        if self._secret_redactor.redact_text(identity) != identity:
            raise ValueError(
                "completion_verifier_profile_policy.identity contains a workload secret."
            )
        return identity

    def _ensure_process_local_generation(self) -> None:
        """Refresh quiescent runtime ownership after a process fork."""

        if self._adapter_runner.ensure_process_local(
            lambda: CompletionVerifierExecutionError(
                "Completion verifier coordinator inherited active execution state "
                "across a process boundary; rebuild the application in this worker."
            )
        ):
            self._execution_owner_id = f"cver_{uuid4().hex}"

    def register(
        self,
        reference: CompletionVerifierRef,
        verifier: DeterministicCompletionVerifier | ProviderCompletionVerifier,
    ) -> CompletionVerifierRef:
        self._ensure_process_local_generation()
        if type(reference) is not CompletionVerifierRef:
            del reference, verifier
            raise TypeError("Completion verifier registration requires a CompletionVerifierRef.")
        if not isinstance(verifier, (DeterministicCompletionVerifier, ProviderCompletionVerifier)):
            del reference, verifier
            raise TypeError(
                "Completion verifier registration requires a DeterministicCompletionVerifier "
                "or a ProviderCompletionVerifier."
            )
        verifier_value = verifier
        del verifier
        reference_value = reference
        del reference
        validation = capture_sensitive_validation(
            lambda value=reference_value: cast(
                "CompletionVerifierRef",
                revalidate_model_input(value, CompletionVerifierRef),
            ),
            operation_name="Completion verifier reference validation",
            redactor=self._secret_redactor,
        )
        del reference_value
        if validation.failure is not None:
            failure = validation.failure
            del validation, verifier_value
            raise_task_store_operation_failure(failure)
        copied = validation.result
        del validation
        if copied is None:
            del verifier_value
            raise ValueError("Completion verifier reference is invalid.") from None
        expected_kind = (
            CompletionVerifierKind.PROVIDER
            if isinstance(verifier_value, ProviderCompletionVerifier)
            else CompletionVerifierKind.DETERMINISTIC
        )
        if copied.kind is not expected_kind:
            del copied, verifier_value
            raise ValueError(
                "Completion verifier reference kind does not match the registered verifier: "
                "provider references require a ProviderCompletionVerifier and deterministic "
                "references a DeterministicCompletionVerifier."
            )
        if _contains_workload_secret(
            (
                copied.verifier_id,
                copied.version,
                copied.configuration_fingerprint,
            ),
            redactor=self._secret_redactor,
        ):
            del copied, verifier_value
            raise ValueError(
                "Completion verifier identity contains a workload secret and cannot be "
                "registered as durable authority."
            ) from None
        key = _verifier_key(copied)
        if key in self._verifiers:
            del copied, key, verifier_value
            raise credential_safe_runtime_exception(
                ValueError,
                "Completion verifier identity is already registered.",
                redactor=self._secret_redactor,
                fallback_message="Completion verifier registration conflict.",
            ) from None
        identity_validation = capture_sensitive_validation(
            lambda value=verifier_value: copy_secret_free_execution_profile_behavior_identity(
                value.execution_profile_identity,
                redactor=self._secret_redactor,
                field_name="completion_verifier.execution_profile_identity",
            ),
            operation_name="Completion verifier execution-profile identity validation",
            redactor=self._secret_redactor,
        )
        if identity_validation.failure is not None:
            failure = identity_validation.failure
            del copied, key, verifier_value, identity_validation
            raise_task_store_operation_failure(failure)
        adapter_identity = identity_validation.result
        del identity_validation
        if adapter_identity is None:
            del copied, key, verifier_value
            raise ValueError(
                "Completion verifier registration requires stable execution-profile identity."
            ) from None
        component_validation = capture_sensitive_validation(
            lambda value=verifier_value: self._copy_registration_components(
                value.execution_profile_components
            ),
            operation_name="Completion verifier component identity validation",
            redactor=self._secret_redactor,
        )
        if component_validation.failure is not None:
            failure = component_validation.failure
            del copied, key, verifier_value, adapter_identity, component_validation
            raise_task_store_operation_failure(failure)
        components = component_validation.result
        del component_validation
        if components is None:
            del copied, key, verifier_value, adapter_identity
            raise ValueError("Completion verifier component identities are invalid.") from None
        target: ProviderCompletionVerifierTarget | None = None
        if isinstance(verifier_value, ProviderCompletionVerifier):
            if any(item.component_id.startswith(RUNTIME_COMPONENT_PREFIX) for item in components):
                del copied, key, verifier_value, adapter_identity, components
                raise ValueError(
                    "Provider verifier component IDs starting with 'runtime.' are reserved."
                ) from None
            target_validation = capture_sensitive_validation(
                lambda value=verifier_value: copy_provider_completion_verifier_target(value.target),
                operation_name="Provider completion verifier target validation",
                redactor=self._secret_redactor,
            )
            if target_validation.failure is not None:
                failure = target_validation.failure
                del copied, key, verifier_value, adapter_identity, components, target_validation
                raise_task_store_operation_failure(failure)
            target = target_validation.result
            del target_validation
            if target is None:
                del copied, key, verifier_value, adapter_identity, components
                raise ValueError("Provider completion verifier target is invalid.") from None
            components = tuple(
                sorted(
                    (*components, *self._provider_runtime_components(target)),
                    key=lambda item: item.component_id,
                )
            )
        profile = build_completion_verifier_execution_profile(
            verifier=copied,
            adapter_identity=adapter_identity,
            component_declarations=components,
        )
        self._verifiers[key] = _RegisteredVerifier(
            adapter=verifier_value, profile=profile, target=target
        )
        del verifier_value
        return copied

    def _copy_registration_components(
        self,
        value: object,
    ) -> tuple[CompletionVerifierProfileComponentDeclaration, ...]:
        if type(value) is not tuple:
            raise TypeError("execution_profile_components must be a tuple.")
        if len(value) > 64:
            raise ValueError("execution_profile_components contains too many values.")
        copied: list[CompletionVerifierProfileComponentDeclaration] = []
        for index, component in enumerate(value):
            if type(component) is not CompletionVerifierProfileComponentDeclaration:
                raise TypeError(
                    f"execution_profile_components[{index}] must be a "
                    "CompletionVerifierProfileComponentDeclaration."
                )
            component_id = component.component_id
            if self._secret_redactor.redact_text(component_id) != component_id:
                raise ValueError(
                    "Completion verifier component identity contains a workload secret."
                )
            identity = copy_secret_free_execution_profile_behavior_identity(
                component.identity,
                redactor=self._secret_redactor,
                field_name=f"completion_verifier.execution_profile_components[{index}].identity",
            )
            if identity is None:  # pragma: no cover - exact declaration requires it
                raise ValueError("Completion verifier component identity is required.")
            copied.append(
                CompletionVerifierProfileComponentDeclaration(
                    component_id=component_id,
                    identity=identity,
                )
            )
        copied.sort(key=lambda item: item.component_id)
        if len({item.component_id for item in copied}) != len(copied):
            raise ValueError("Completion verifier component IDs must be unique.")
        return tuple(copied)

    def _require_live_registered_profile(
        self,
        registered: _RegisteredVerifier,
    ) -> None:
        validation = capture_sensitive_validation(
            lambda adapter=registered.adapter: (
                copy_secret_free_execution_profile_behavior_identity(
                    adapter.execution_profile_identity,
                    redactor=self._secret_redactor,
                    field_name="completion_verifier.execution_profile_identity",
                ),
                self._copy_registration_components(adapter.execution_profile_components),
            ),
            operation_name="Completion verifier live-profile validation",
            redactor=self._secret_redactor,
        )
        if validation.failure is not None:
            raise CompletionVerifierUnavailable(
                "Completion verifier live execution-profile identity is invalid."
            ) from None
        live_profile = validation.result
        if live_profile is None or live_profile[0] is None:
            raise CompletionVerifierUnavailable(
                "Completion verifier execution-profile identity is unavailable."
            ) from None
        components = live_profile[1]
        if registered.target is not None:
            target_validation = capture_sensitive_validation(
                lambda adapter=registered.adapter: copy_provider_completion_verifier_target(
                    cast("ProviderCompletionVerifier", adapter).target
                ),
                operation_name="Provider completion verifier live-target validation",
                redactor=self._secret_redactor,
            )
            if target_validation.result != registered.target:
                raise CompletionVerifierUnavailable(
                    "Provider completion verifier target changed after registration."
                ) from None
            components = tuple(
                sorted(
                    (*components, *self._provider_runtime_components(registered.target)),
                    key=lambda item: item.component_id,
                )
            )
        candidate = build_completion_verifier_execution_profile(
            verifier=registered.profile.verifier,
            adapter_identity=live_profile[0],
            component_declarations=components,
        )
        if candidate != registered.profile:
            raise CompletionVerifierUnavailable(
                "Completion verifier execution-profile identity changed after registration."
            ) from None

    async def verify(
        self,
        request: CompletionVerifierExecutionRequest,
        execution_deadline: ExecutionDeadline | None = None,
    ) -> CompletionDecision:
        operation = self._verify(request, execution_deadline)
        del request, execution_deadline
        return await operation

    def start_owned_verification(
        self,
        request: CompletionVerifierExecutionRequest,
        execution_deadline: ExecutionDeadline | None = None,
    ) -> CompletionVerifierOwnedExecution:
        validation = capture_sensitive_validation(
            lambda value=request, boundary=execution_deadline: (
                copy_completion_verifier_execution_request(value),
                _copy_verifier_deadline(boundary),
            ),
            operation_name="Owned completion verifier request validation",
            redactor=self._secret_redactor,
        )
        del request, execution_deadline
        if validation.failure is not None:
            raise validation.failure from None
        if validation.result is None:
            raise ValueError("Owned completion verifier request is invalid.") from None
        copied, deadline = validation.result
        retained: asyncio.Future[_DrainingAdapter | None] = (
            asyncio.get_running_loop().create_future()
        )

        async def run(value=copied, boundary=deadline):
            operation = self._verify(value, boundary, retained_drain=retained)
            del value, boundary
            return await operation

        def finished(_task):
            if not retained.done():
                retained.set_result(None)

        operation = asyncio.create_task(
            capture_awaitable_outcome(run), name="cayu-owned-completion-verification"
        )
        operation.add_done_callback(finished)
        return CompletionVerifierOwnedExecution(operation, retained, self)

    async def _verify(
        self,
        request: CompletionVerifierExecutionRequest,
        execution_deadline: ExecutionDeadline | None = None,
        *,
        retained_drain: asyncio.Future[_DrainingAdapter | None] | None = None,
    ) -> CompletionDecision:
        self._ensure_process_local_generation()
        request_value = request
        deadline_value = execution_deadline
        del request, execution_deadline
        validation = capture_sensitive_validation(
            lambda value=request_value, deadline=deadline_value: (
                copy_completion_verifier_execution_request(value),
                _copy_verifier_deadline(deadline),
            ),
            operation_name="Completion verifier execution request validation",
            redactor=self._secret_redactor,
        )
        del request_value, deadline_value
        if validation.failure is not None:
            raise validation.failure from None
        prepared = validation.result
        del validation
        if prepared is None:
            raise ValueError("Completion verifier execution request is invalid.") from None
        copied, deadline = prepared
        del prepared
        if _contains_workload_secret(
            (deadline.source, deadline.scope), redactor=self._secret_redactor
        ):
            del copied, deadline
            raise ValueError("Completion-verifier deadline contains a workload secret.") from None
        if copied.profile_adoption is not None:
            adoption_validation = capture_sensitive_validation(
                lambda intent=copied.profile_adoption: self._require_safe_adoption_intent(intent),
                operation_name="Completion verifier profile-adoption validation",
                redactor=self._secret_redactor,
            )
            if adoption_validation.failure is not None:
                failure = adoption_validation.failure
                del adoption_validation, copied
                raise_task_store_operation_failure(failure)
            if adoption_validation.result is not True:
                del adoption_validation, copied
                raise WorkCompletionConflict(
                    "Completion-verifier profile adoption audit fields contain a workload secret."
                ) from None
            del adoption_validation
        if _contains_workload_secret(
            (
                copied.proposal_id,
                copied.claim_id,
                copied.decision_id,
                copied.worker_id,
            ),
            redactor=self._secret_redactor,
        ):
            del copied
            raise ValueError(
                "Completion-verifier execution identity contains a workload secret."
            ) from None
        key = copied.proposal_id
        with self._adapter_runner.single_flight(key) as lock:
            safe_cancellation: asyncio.CancelledError | None = None
            try:
                await lock.acquire()
            except asyncio.CancelledError as cancellation:
                safe_cancellation = _safe_caller_cancellation(
                    cancellation,
                    redactor=self._secret_redactor,
                )
                del cancellation, copied
            if safe_cancellation is not None:
                raise safe_cancellation from None
            try:
                return await self._verify_locked(copied, deadline, retained_drain=retained_drain)
            finally:
                lock.release()

    async def _verify_locked(
        self,
        request: CompletionVerifierExecutionRequest,
        deadline: ExecutionDeadline,
        *,
        retained_drain: asyncio.Future[_DrainingAdapter | None] | None = None,
    ) -> CompletionDecision:
        store_owner = _TaskStoreOwner(self._require_store())
        authority = await self._load_verifier_authority(store_owner, request)

        existing = await self._load_converged_decision(
            store_owner,
            decision_id=request.decision_id,
            proposal_id=request.proposal_id,
        )
        if existing is not None:
            claim: CompletionVerificationClaim | None = None
            profile: CompletionVerifierProfileRecord | None = None
            existing_validation = None
            try:
                profile = await self._load_profile(store_owner, request.proposal_id)
                if profile is None:
                    raise WorkCompletionConflict(
                        "Durable completion decision has no verifier-profile authority."
                    ) from None
                self._require_profile_authority(profile, authority)
                prior_profile = await self._load_prior_profile(store_owner, authority)
                self._require_profile_transition(profile, prior_profile)
                self._require_profile_adoption_replay(
                    request=request,
                    authority=authority,
                    profile=profile,
                    prior=prior_profile,
                )
                claim = await self._load_required_claim(
                    store_owner,
                    request.proposal_id,
                )
                self._require_exact_claim(
                    claim,
                    request,
                    authority.contract.verifier,
                    execution_owner_id=claim.execution_owner_id,
                    verifier_profile_fingerprint=profile.profile.fingerprint,
                    accept_legacy_timeout=True,
                )
                existing_validation = capture_sensitive_validation(
                    lambda: self._require_existing_decision(
                        existing,
                        request=request,
                        proposal=authority.proposal,
                        attempt=authority.attempt,
                        contract=authority.contract,
                        claim=cast("CompletionVerificationClaim", claim),
                    ),
                    operation_name="Existing completion decision validation",
                    redactor=self._secret_redactor,
                )
                if existing_validation.failure is not None:
                    raise existing_validation.failure from None
                if existing_validation.result is not True:
                    raise WorkCompletionConflict(
                        "Durable completion decision has conflicting authority."
                    ) from None
                return existing
            except BaseException as failure:
                del existing
                del claim, profile, existing_validation, authority, store_owner
                raise_task_store_operation_failure(failure)

        deadline.require_admission("completion_verifier")
        provider_backed = authority.contract.verifier.kind is CompletionVerifierKind.PROVIDER
        if provider_backed and not task_store_completion_verifier_dispatch_capability_is_complete(
            store_owner.store
        ):
            raise credential_safe_runtime_exception(
                CompletionVerifierUnavailable,
                "Provider-backed completion verifiers require a task store with durable "
                "provider-verifier dispatch support.",
                redactor=self._secret_redactor,
                fallback_message="The required completion verifier is unavailable.",
            ) from None
        verifier_key = _verifier_key(authority.contract.verifier)
        if verifier_key not in self._verifiers:
            raise credential_safe_runtime_exception(
                CompletionVerifierUnavailable,
                "The exact completion verifier required by the work contract is not registered.",
                redactor=self._secret_redactor,
                fallback_message="The required completion verifier is unavailable.",
            ) from None

        operation_key = self._execution_key(request)
        draining = self._adapter_runner.draining(operation_key)
        if draining is not None and not draining.task.done():
            raise CompletionVerifierExecutionError(
                "The prior exact completion-verifier execution is still draining."
            ) from None
        if draining is not None:
            settlement_task = self._ensure_draining_adapter_settlement(
                operation_key,
                draining,
            )
            if not settlement_task.done():
                raise CompletionVerifierExecutionError(
                    "The prior exact completion-verifier execution is still draining."
                ) from None
            if not self._adapter_runner.finalize_settlement(
                operation_key,
                draining,
                settlement_task,
                failure_for=lambda settlement: settlement.failure,
            ):
                raise CompletionVerifierExecutionError(
                    "The prior exact completion-verifier execution is still draining."
                ) from None
            settlement_failure = draining.settlement_failure
            if settlement_failure is not None:
                self._adapter_runner.acknowledge_drain(operation_key, draining)
                raise_task_store_operation_failure(settlement_failure)

        registered = self._verifiers[verifier_key]
        self._require_live_registered_profile(registered)
        profile = await self._prepare_profile(
            store_owner,
            request,
            authority,
            registered,
        )
        self._require_live_registered_profile(registered)
        verifier = registered.adapter
        capacity_reservation = self._reserve_adapter_capacity()
        heartbeat: _ClaimHeartbeat | None = None
        try:
            claim_request = CompletionVerificationClaimRequest(
                claim_id=request.claim_id,
                proposal_id=request.proposal_id,
                worker_id=request.worker_id,
                execution_owner_id=self._execution_owner_id,
                verifier=authority.contract.verifier,
                verifier_profile_fingerprint=profile.profile.fingerprint,
                lease_seconds=request.lease_seconds,
                execution_timeout_seconds=request.execution_timeout_seconds,
            )
            claim_outcome = await capture_task_store_operation(
                lambda: store_owner.store.claim_completion_verification(claim_request),
                operation_name="Completion verification claim",
                redactor=self._secret_redactor,
                mutation_store=store_owner.store,
                mutation_method_name="claim_completion_verification",
            )
            if claim_outcome.failure is not None:
                raise_task_store_operation_failure(claim_outcome.failure)
            claim_validation = _copy_exact_model(
                claim_outcome.result,
                CompletionVerificationClaim,
                operation_name="Completion verification claim result validation",
                redactor=self._secret_redactor,
            )
            del claim_outcome
            if claim_validation.failure is not None:
                raise claim_validation.failure from None
            claim = claim_validation.result
            del claim_validation
            if claim is None:
                raise WorkCompletionConflict(
                    "Task store returned an invalid completion verification claim."
                ) from None
            self._require_exact_claim(
                claim,
                request,
                authority.contract.verifier,
                execution_owner_id=self._execution_owner_id,
                verifier_profile_fingerprint=profile.profile.fingerprint,
            )
            claim, claim_deadline_monotonic = await self._renew_claim(
                store_owner,
                claim_request,
                request,
                prior_claim=claim,
            )
            heartbeat = self._start_claim_heartbeat(
                store_owner,
                claim_request,
                request,
                claim,
                claim_deadline_monotonic=claim_deadline_monotonic,
                capacity_reservation=capacity_reservation,
            )

            self._require_live_registered_profile(registered)

            if registered.target is None:
                deterministic = cast("DeterministicCompletionVerifier", verifier)
                invoke = deterministic.verify
            else:
                invoke = self._provider_invocation(
                    store_owner.store,
                    cast("ProviderCompletionVerifier", verifier),
                    registered.target,
                    ProviderVerifierExecutionAuthority(
                        proposal_id=claim.proposal_id,
                        claim_id=claim.claim_id,
                        worker_id=claim.worker_id,
                        execution_owner_id=self._execution_owner_id,
                        claim_attempt_number=claim.attempt_number,
                        verifier=claim.verifier,
                        verifier_profile_fingerprint=claim.verifier_profile_fingerprint,
                    ),
                )
            outcome = await self._invoke_adapter(
                invoke,
                authority.adapter_request,
                operation_key=operation_key,
                timeout_seconds=request.execution_timeout_seconds,
                heartbeat=heartbeat,
                deadline=deadline,
                retained_drain=retained_drain,
            )
        except BaseException as failure:
            if heartbeat is not None and not heartbeat.retained_for_drain:
                settlement = await self._settle_claim_heartbeat(heartbeat)
                propagated = self._failure_after_heartbeat_settlement(
                    failure,
                    settlement,
                    group_message=(
                        "Completion verifier execution and heartbeat settlement failed."
                    ),
                )
                del settlement
            else:
                propagated = failure
            del verifier, registered, profile, authority, store_owner
            if propagated is None:  # pragma: no cover - primary failure is authoritative
                raise AssertionError("Completion verifier failure was lost.") from None
            raise_task_store_operation_failure(propagated)
        finally:
            if heartbeat is None:
                self._release_adapter_capacity_reservation(capacity_reservation)
        try:
            decision = await self._publish_adapter_outcome(
                store_owner=store_owner,
                request=request,
                proposal=authority.proposal,
                attempt=authority.attempt,
                contract=authority.contract,
                verifier_profile_fingerprint=profile.profile.fingerprint,
                outcome=outcome,
                deadline=deadline,
            )
        except BaseException as failure:
            if heartbeat is not None:
                settlement = await self._settle_claim_heartbeat(heartbeat)
                propagated = self._failure_after_heartbeat_settlement(
                    failure,
                    settlement,
                    group_message=(
                        "Completion decision publication and heartbeat settlement failed."
                    ),
                )
                del settlement
            else:
                propagated = failure
            del outcome, verifier, registered, profile, claim, claim_request, authority, store_owner
            if propagated is None:  # pragma: no cover - primary failure is authoritative
                raise AssertionError("Completion decision publication failure was lost.") from None
            raise_task_store_operation_failure(propagated)
        if heartbeat is not None:
            settlement = await self._settle_claim_heartbeat(heartbeat)
            propagated = self._failure_after_heartbeat_settlement(
                None,
                settlement,
                group_message="Completion verifier heartbeat settlement failed.",
            )
            del settlement
            if propagated is not None:
                del (
                    decision,
                    outcome,
                    verifier,
                    registered,
                    profile,
                    claim,
                    claim_request,
                    authority,
                    store_owner,
                )
                raise_task_store_operation_failure(propagated)
        return decision

    async def _load_verifier_authority(
        self,
        store_owner: _TaskStoreOwner,
        request: CompletionVerifierExecutionRequest,
    ) -> _VerifierAuthority:
        proposal: CompletionProposal | None = None
        attempt: WorkAttempt | None = None
        contract: WorkContract | None = None
        context_validation = None
        adapter_request: CompletionVerifierRequest | None = None
        try:
            proposal = await self._load_required(
                lambda owner=store_owner, proposal_id=request.proposal_id: (
                    owner.store.load_completion_proposal(proposal_id)
                ),
                CompletionProposal,
                identity=request.proposal_id,
                identity_field="proposal_id",
                operation_name="Completion proposal lookup",
            )
            attempt = await self._load_required(
                lambda owner=store_owner, attempt_id=proposal.attempt_id: (
                    owner.store.load_work_attempt(attempt_id)
                ),
                WorkAttempt,
                identity=proposal.attempt_id,
                identity_field="attempt_id",
                operation_name="Work attempt lookup",
            )
            contract = await self._load_required(
                lambda owner=store_owner, reference=proposal.contract: (
                    owner.store.load_work_contract(reference)
                ),
                WorkContract,
                identity=proposal.contract.contract_id,
                identity_field="contract_id",
                operation_name="Work contract lookup",
            )
            if _contains_workload_secret(
                (
                    contract.contract_id,
                    contract.verifier.verifier_id,
                    contract.verifier.version,
                    contract.verifier.configuration_fingerprint,
                    contract.result_resolver.resolver_id,
                    contract.result_resolver.version,
                    contract.result_resolver.configuration_fingerprint,
                    *(criterion.criterion_id for criterion in contract.criteria),
                    *(constraint.constraint_id for constraint in contract.constraints),
                    *(requirement.requirement_id for requirement in contract.evidence_requirements),
                ),
                redactor=self._secret_redactor,
            ):
                raise ValueError(
                    "Durable completion-verifier context contains a workload secret in public "
                    "identity."
                ) from None
            context_validation = capture_sensitive_validation(
                lambda contract_value=contract, attempt_value=attempt, proposal_value=proposal: (
                    CompletionVerifierRequest(
                        contract=contract_value,
                        attempt=attempt_value,
                        proposal=proposal_value,
                    )
                ),
                operation_name="Completion verifier context validation",
                redactor=self._secret_redactor,
            )
            if context_validation.failure is not None:
                raise context_validation.failure from None
            adapter_request = context_validation.result
            context_validation = None
            if adapter_request is None:
                raise WorkCompletionConflict(
                    "Durable verifier context has conflicting authority."
                ) from None
            self._require_context_integrity(
                proposal=proposal,
                attempt=attempt,
            )
            return _VerifierAuthority(
                proposal=proposal,
                attempt=attempt,
                contract=contract,
                adapter_request=adapter_request,
            )
        except BaseException as failure:
            del proposal, attempt, contract, context_validation, adapter_request, store_owner
            raise_task_store_operation_failure(failure)

    async def _load_optional_profile(
        self,
        operation: Callable[[], Awaitable[object]],
        *,
        proposal_id: str,
        operation_name: str,
    ) -> CompletionVerifierProfileRecord | None:
        outcome = await capture_task_store_operation(
            operation,
            operation_name=operation_name,
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            return None
        raw_result = outcome.result
        del outcome
        validation = capture_sensitive_validation(
            lambda value=raw_result: copy_completion_verifier_profile_record(
                cast("CompletionVerifierProfileRecord", value)
            ),
            operation_name=f"{operation_name} result validation",
            redactor=self._secret_redactor,
        )
        del raw_result
        if validation.failure is not None:
            raise validation.failure from None
        profile = validation.result
        del validation
        if profile is None or profile.proposal_id != proposal_id:
            raise WorkCompletionConflict(
                "Task store returned completion-verifier profile authority for another proposal."
            ) from None
        return profile

    async def _load_profile(
        self,
        store_owner: _TaskStoreOwner,
        proposal_id: str,
    ) -> CompletionVerifierProfileRecord | None:
        return await self._load_optional_profile(
            lambda: store_owner.store.load_completion_verifier_profile(proposal_id),
            proposal_id=proposal_id,
            operation_name="Completion-verifier profile lookup",
        )

    async def _load_prior_profile(
        self,
        store_owner: _TaskStoreOwner,
        authority: _VerifierAuthority,
    ) -> CompletionVerifierProfileRecord | None:
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.load_prior_completion_verifier_profile(
                authority.proposal.proposal_id
            ),
            operation_name="Prior completion-verifier profile lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            if authority.attempt.ordinal != 1:
                raise WorkCompletionConflict(
                    "Prior work attempt has no verifier-profile authority."
                ) from None
            return None
        raw_result = outcome.result
        del outcome
        validation = capture_sensitive_validation(
            lambda value=raw_result: copy_completion_verifier_profile_record(
                cast("CompletionVerifierProfileRecord", value)
            ),
            operation_name="Prior completion-verifier profile result validation",
            redactor=self._secret_redactor,
        )
        del raw_result
        if validation.failure is not None:
            raise validation.failure from None
        profile = validation.result
        if profile is None:
            raise WorkCompletionConflict(
                "Task store returned invalid prior verifier-profile authority."
            ) from None
        prior_proposal = await self._load_required(
            lambda: store_owner.store.load_completion_proposal(profile.proposal_id),
            CompletionProposal,
            identity=profile.proposal_id,
            identity_field="proposal_id",
            operation_name="Prior completion proposal lookup",
        )
        prior_attempt = await self._load_required(
            lambda: store_owner.store.load_work_attempt(profile.attempt_id),
            WorkAttempt,
            identity=profile.attempt_id,
            identity_field="attempt_id",
            operation_name="Prior work attempt lookup",
        )
        prior_contract = await self._load_required(
            lambda: store_owner.store.load_work_contract(profile.contract),
            WorkContract,
            identity=profile.contract.contract_id,
            identity_field="contract_id",
            operation_name="Prior work contract lookup",
        )
        require_completion_verifier_profile_integrity(
            profile=profile,
            proposal=prior_proposal,
            attempt=prior_attempt,
            contract=prior_contract,
        )
        if (
            prior_attempt.task_id != authority.attempt.task_id
            or prior_attempt.ordinal != authority.attempt.ordinal - 1
            or prior_proposal.attempt_id != prior_attempt.attempt_id
            or prior_contract.reference() != authority.contract.reference()
        ):
            raise WorkCompletionConflict(
                "Task store returned a non-preceding verifier profile as adoption authority."
            ) from None
        return profile

    def _require_profile_authority(
        self,
        profile: CompletionVerifierProfileRecord,
        authority: _VerifierAuthority,
    ) -> None:
        require_completion_verifier_profile_integrity(
            profile=profile,
            proposal=authority.proposal,
            attempt=authority.attempt,
            contract=authority.contract,
        )

    async def _prepare_profile(
        self,
        store_owner: _TaskStoreOwner,
        request: CompletionVerifierExecutionRequest,
        authority: _VerifierAuthority,
        registered: _RegisteredVerifier,
    ) -> CompletionVerifierProfileRecord:
        existing = await self._load_profile(store_owner, request.proposal_id)
        if existing is not None:
            self._require_profile_authority(existing, authority)
            prior = await self._load_prior_profile(store_owner, authority)
            self._require_profile_transition(existing, prior)
            if existing.profile != registered.profile:
                raise CompletionVerifierUnavailable(
                    "The registered completion verifier does not match the durable profile."
                ) from None
            self._require_profile_adoption_replay(
                request=request,
                authority=authority,
                profile=existing,
                prior=prior,
            )
            return existing

        prior = await self._load_prior_profile(store_owner, authority)
        adoption: CompletionVerifierProfileAdoptionDecision | None = None
        expected_prior_proposal_id = None if prior is None else prior.proposal_id
        expected_prior = None if prior is None else prior.profile.fingerprint
        if prior is None:
            if request.profile_adoption is not None:
                raise WorkCompletionConflict(
                    "Initial completion-verifier profile cannot carry adoption intent."
                ) from None
        elif prior.profile == registered.profile:
            if request.profile_adoption is not None:
                raise WorkCompletionConflict(
                    "Exact completion-verifier profile reuse cannot carry adoption intent."
                ) from None
        else:
            intent = request.profile_adoption
            policy = self._profile_policy
            if intent is None or policy is None:
                raise WorkCompletionConflict(
                    "Changed completion-verifier profile requires explicit authorized adoption."
                ) from None
            policy_identity = self._profile_policy_identity
            if policy_identity is None:
                raise WorkCompletionConflict(
                    "Completion-verifier profile policy identity is unavailable."
                ) from None
            changed_components = changed_completion_verifier_profile_components(
                prior.profile,
                registered.profile,
            )
            policy_request = CompletionVerifierProfilePolicyRequest(
                task_id=authority.proposal.task_id,
                proposal_id=authority.proposal.proposal_id,
                attempt_id=authority.attempt.attempt_id,
                expected_profile=prior.profile,
                candidate_profile=registered.profile,
                changed_component_ids=changed_components,
                intent=intent,
            )
            policy_outcome = await capture_task_store_operation(
                lambda: policy.decide(policy_request),
                operation_name="Completion verifier profile-policy decision",
                redactor=self._secret_redactor,
            )
            if policy_outcome.failure is not None:
                raise_task_store_operation_failure(policy_outcome.failure)
            raw_policy_result = policy_outcome.result
            del policy_outcome
            result_validation = capture_sensitive_validation(
                lambda value=raw_policy_result: copy_execution_profile_policy_result(
                    cast("ExecutionProfilePolicyResult", value)
                ),
                operation_name="Completion verifier profile-policy result validation",
                redactor=self._secret_redactor,
            )
            del raw_policy_result
            if result_validation.failure is not None:
                raise result_validation.failure from None
            result = result_validation.result
            del result_validation
            if result is None:
                raise WorkCompletionConflict(
                    "Completion-verifier profile policy returned an invalid decision."
                ) from None
            adoption_rejected = (
                result.action is not ExecutionProfilePolicyAction.ADOPT
                or result.authority_decision is not ExecutionProfileAuthorityDecision.AUTHORIZED
            )
            policy_reason = self._secret_redactor.redact_text_bounded(
                result.reason,
                max_bytes=EXECUTION_PROFILE_ADOPTION_TEXT_MAX_CHARS,
            )
            del result
            current_policy_identity = capture_sensitive_validation(
                lambda: self._copy_profile_policy_identity(policy),
                operation_name="Completion verifier profile-policy identity revalidation",
                redactor=self._secret_redactor,
            )
            if current_policy_identity.failure is not None:
                raise current_policy_identity.failure from None
            if current_policy_identity.result != policy_identity:
                raise WorkCompletionConflict(
                    "Completion-verifier profile policy identity changed during authorization."
                ) from None
            if adoption_rejected:
                raise WorkCompletionConflict(
                    "Completion-verifier profile adoption was not explicitly authorized."
                ) from None
            adoption_request_sha256 = completion_verifier_profile_adoption_request_sha256(
                policy_request,
                policy_identity=policy_identity,
            )
            adoption = CompletionVerifierProfileAdoptionDecision(
                expected_profile_fingerprint=prior.profile.fingerprint,
                candidate_profile_fingerprint=registered.profile.fingerprint,
                changed_component_ids=changed_components,
                policy_identity=policy_identity,
                authority_decision=ExecutionProfileAuthorityDecision.AUTHORIZED,
                idempotency_key=intent.idempotency_key,
                requested_by=ResolutionActor(
                    subject=intent.requested_by.subject,
                    tenant=intent.requested_by.tenant,
                    source=intent.requested_by.source,
                ),
                reason=intent.reason,
                policy_reason=policy_reason,
                request_sha256=adoption_request_sha256,
            )

        preparation = CompletionVerifierProfilePreparationRequest(
            proposal_id=authority.proposal.proposal_id,
            task_id=authority.proposal.task_id,
            attempt_id=authority.attempt.attempt_id,
            attempt_request_sha256=authority.attempt.request_sha256,
            source_execution_profile_fingerprint=(authority.attempt.execution_profile_fingerprint),
            proposal_request_sha256=authority.proposal.request_sha256,
            contract=authority.contract.reference(),
            profile=registered.profile,
            expected_prior_proposal_id=expected_prior_proposal_id,
            expected_prior_profile_fingerprint=expected_prior,
            adoption=adoption,
        )
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.prepare_completion_verifier_profile(preparation),
            operation_name="Completion-verifier profile preparation",
            redactor=self._secret_redactor,
            mutation_store=store_owner.store,
            mutation_method_name="prepare_completion_verifier_profile",
        )
        if outcome.failure is not None:
            failure = outcome.failure
            if isinstance(failure, Exception):
                try:
                    reconciled = await self._load_profile(store_owner, request.proposal_id)
                    if reconciled is not None:
                        self._require_profile_authority(reconciled, authority)
                        self._require_profile_transition(reconciled, prior)
                        if reconciled.request_sha256 == (
                            completion_verifier_profile_preparation_request_sha256(preparation)
                        ):
                            return reconciled
                except BaseException as reconciliation_failure:
                    combined = _ordered_failure_group(
                        "Completion-verifier profile preparation and reconciliation failed.",
                        failure,
                        reconciliation_failure,
                        redactor=self._secret_redactor,
                    )
                    del failure, reconciliation_failure
                    raise combined from None
            raise_task_store_operation_failure(failure)
        raw_result = outcome.result
        del outcome
        validation = capture_sensitive_validation(
            lambda value=raw_result: copy_completion_verifier_profile_record(
                cast("CompletionVerifierProfileRecord", value)
            ),
            operation_name="Completion-verifier profile preparation result validation",
            redactor=self._secret_redactor,
        )
        del raw_result
        if validation.failure is not None:
            raise validation.failure from None
        profile = validation.result
        if profile is None:
            raise WorkCompletionConflict(
                "Task store returned invalid completion-verifier profile authority."
            ) from None
        self._require_profile_authority(profile, authority)
        self._require_profile_transition(profile, prior)
        if profile.request_sha256 != completion_verifier_profile_preparation_request_sha256(
            preparation
        ):
            raise WorkCompletionConflict(
                "Task store returned a different completion-verifier profile preparation."
            ) from None
        return profile

    def _require_profile_adoption_replay(
        self,
        *,
        request: CompletionVerifierExecutionRequest,
        authority: _VerifierAuthority,
        profile: CompletionVerifierProfileRecord,
        prior: CompletionVerifierProfileRecord | None,
    ) -> None:
        """Accept only an exact replay of already-authorized adoption intent."""

        intent = request.profile_adoption
        adoption = profile.adoption
        if adoption is None:
            if intent is not None:
                raise WorkCompletionConflict(
                    "Exact verifier-profile replay cannot introduce adoption intent."
                ) from None
            return

        policy_identity = adoption.policy_identity
        if prior is None:
            raise WorkCompletionConflict(
                "Verifier-profile adoption has no prior profile authority."
            ) from None
        if intent is None:
            return
        changed_components = changed_completion_verifier_profile_components(
            prior.profile,
            profile.profile,
        )
        policy_request = CompletionVerifierProfilePolicyRequest(
            task_id=authority.proposal.task_id,
            proposal_id=authority.proposal.proposal_id,
            attempt_id=authority.attempt.attempt_id,
            expected_profile=prior.profile,
            candidate_profile=profile.profile,
            changed_component_ids=changed_components,
            intent=intent,
        )
        replay_actor = ResolutionActor(
            subject=intent.requested_by.subject,
            tenant=intent.requested_by.tenant,
            source=intent.requested_by.source,
        )
        if (
            adoption.idempotency_key != intent.idempotency_key
            or adoption.requested_by != replay_actor
            or adoption.reason != intent.reason
            or adoption.request_sha256
            != completion_verifier_profile_adoption_request_sha256(
                policy_request,
                policy_identity=policy_identity,
            )
        ):
            raise WorkCompletionConflict(
                "Verifier-profile adoption retry conflicts with durable authority."
            ) from None

    def _require_profile_transition(
        self,
        profile: CompletionVerifierProfileRecord,
        prior: CompletionVerifierProfileRecord | None,
    ) -> None:
        expected_proposal_id = None if prior is None else prior.proposal_id
        expected_fingerprint = None if prior is None else prior.profile.fingerprint
        if (
            profile.expected_prior_proposal_id != expected_proposal_id
            or profile.expected_prior_profile_fingerprint != expected_fingerprint
        ):
            raise WorkCompletionConflict(
                "Durable completion-verifier profile conflicts with prior profile authority."
            ) from None
        adoption = profile.adoption
        if prior is None or prior.profile == profile.profile:
            if adoption is not None:
                raise WorkCompletionConflict(
                    "Verifier-profile reuse has unexpected adoption authority."
                ) from None
            return
        if adoption is None or adoption.changed_component_ids != (
            changed_completion_verifier_profile_components(
                prior.profile,
                profile.profile,
            )
        ):
            raise WorkCompletionConflict(
                "Changed verifier profile has invalid adoption authority."
            ) from None

    def _require_safe_adoption_intent(
        self,
        intent: ExecutionProfileAdoptionIntent,
    ) -> bool:
        document = intent.model_dump(mode="json", warnings=False)
        self._secret_redactor.require_no_secret_keys(
            document,
            field_name="completion_verifier_profile_adoption",
            match_short_substrings=True,
        )
        if self._secret_redactor.redact_json_values(document) != document:
            raise WorkCompletionConflict(
                "Completion-verifier profile adoption audit fields contain a workload secret."
            ) from None
        return True

    async def _publish_adapter_outcome(
        self,
        *,
        store_owner: _TaskStoreOwner,
        request: CompletionVerifierExecutionRequest,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        verifier_profile_fingerprint: str,
        outcome: CompletionVerifierDecision,
        deadline: ExecutionDeadline,
    ) -> CompletionDecision:
        decision_request = CompletionDecisionCreate(
            decision_id=request.decision_id,
            proposal_id=request.proposal_id,
            claim_id=request.claim_id,
            worker_id=request.worker_id,
            verifier=contract.verifier,
            verifier_profile_fingerprint=verifier_profile_fingerprint,
            verdict=outcome.verdict,
            criterion_outcomes=outcome.criterion_outcomes,
            constraint_outcomes=outcome.constraint_outcomes,
            gaps=outcome.gaps,
            evidence_references=outcome.evidence_references,
        )
        del outcome
        record_outcome = None
        reconciled: CompletionDecision | None = None
        decision_validation = None
        decision: CompletionDecision | None = None
        indexed_decision: CompletionDecision | None = None
        published_claim: CompletionVerificationClaim | None = None
        contract_validation = None
        try:
            contract_validation = capture_sensitive_validation(
                lambda: _validate_decision_contract(contract, decision_request),
                operation_name="Completion verifier contract validation",
                redactor=self._secret_redactor,
            )
            if contract_validation.failure is not None:
                raise contract_validation.failure from None
            if contract_validation.result is not True:
                raise WorkCompletionConflict(
                    "Completion verifier decision conflicts with the frozen contract."
                ) from None
            contract_validation = None

            async def publish_decision(owner=store_owner, value=decision_request):
                deadline.require_admission("completion_decision")
                return await owner.store.record_completion_decision(value)

            record_outcome = await capture_task_store_operation(
                publish_decision,
                operation_name="Completion decision publication",
                redactor=self._secret_redactor,
                mutation_store=store_owner.store,
                mutation_method_name="record_completion_decision",
            )
            if record_outcome.failure is not None:
                failure = record_outcome.failure
                record_outcome = None
                # Read-only acknowledgement-loss reconciliation is valid only
                # for ordinary failures.  Cancellation and process-control
                # signals remain authoritative even if the mutation committed.
                if not isinstance(failure, Exception):
                    raise_task_store_operation_failure(failure)
                try:
                    reconciled = await self._load_converged_decision(
                        store_owner,
                        decision_id=request.decision_id,
                        proposal_id=request.proposal_id,
                    )
                    if reconciled is not None:
                        published_claim = await self._load_required_claim(
                            store_owner,
                            request.proposal_id,
                        )
                        if self._decision_matches_request(
                            reconciled,
                            decision_request,
                            claim=published_claim,
                            proposal=proposal,
                            attempt=attempt,
                            contract=contract,
                        ):
                            return reconciled
                except BaseException as reconciliation_failure:
                    combined = _ordered_failure_group(
                        "Completion decision publication and reconciliation failed.",
                        failure,
                        reconciliation_failure,
                        redactor=self._secret_redactor,
                    )
                    del failure, reconciliation_failure
                    raise combined from None
                raise_task_store_operation_failure(failure)
            decision_validation = _copy_exact_model(
                record_outcome.result,
                CompletionDecision,
                operation_name="Completion decision result validation",
                redactor=self._secret_redactor,
            )
            record_outcome = None
            if decision_validation.failure is not None:
                raise decision_validation.failure from None
            decision = decision_validation.result
            decision_validation = None
            published_claim = await self._load_required_claim(
                store_owner,
                request.proposal_id,
            )
            if decision is None or not self._decision_matches_request(
                decision,
                decision_request,
                claim=published_claim,
                proposal=proposal,
                attempt=attempt,
                contract=contract,
            ):
                raise WorkCompletionConflict(
                    "Task store returned a completion decision other than the exact publication."
                ) from None
            indexed_decision = await self._load_converged_decision(
                store_owner,
                decision_id=request.decision_id,
                proposal_id=request.proposal_id,
            )
            if indexed_decision != decision:
                raise WorkCompletionConflict(
                    "Completion decision publication did not converge in both durable indexes."
                ) from None
            return decision
        except BaseException as failure:
            del decision_request, contract, attempt, proposal, store_owner
            del record_outcome, reconciled, decision_validation, decision, indexed_decision
            del published_claim, contract_validation
            raise_task_store_operation_failure(failure)

    def _require_store(self) -> TaskStore:
        store = self._task_store
        if store is None:
            raise RuntimeError("task_store is required to execute completion verifiers.") from None
        if not store.supports_verified_work_contracts:
            raise NotImplementedError(
                f"{type(store).__name__} does not support verified work contracts."
            ) from None
        return store

    async def _load_required(
        self,
        operation: Callable[[], Awaitable[object]],
        model_type: type[_ModelT],
        *,
        identity: str,
        identity_field: str,
        operation_name: str,
    ) -> _ModelT:
        outcome = await capture_task_store_operation(
            operation,
            operation_name=operation_name,
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            raise KeyError(f"{operation_name.removesuffix(' lookup')} not found.") from None
        validation = _copy_exact_model(
            outcome.result,
            model_type,
            operation_name=f"{operation_name} result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        value = validation.result
        del validation
        if value is None or getattr(value, identity_field, None) != identity:
            del value, operation
            raise WorkCompletionConflict(
                f"Task store returned {operation_name.lower()} authority for another identity."
            ) from None
        return value

    async def _load_required_claim(
        self,
        store_owner: _TaskStoreOwner,
        proposal_id: str,
    ) -> CompletionVerificationClaim:
        return await self._load_required(
            lambda: store_owner.store.load_completion_verification_claim(proposal_id),
            CompletionVerificationClaim,
            identity=proposal_id,
            identity_field="proposal_id",
            operation_name="Completion verification claim lookup",
        )

    async def _load_optional_decision(
        self,
        store_owner: _TaskStoreOwner,
        decision_id: str,
    ) -> CompletionDecision | None:
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.load_completion_decision(decision_id),
            operation_name="Completion decision lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            return None
        validation = _copy_exact_model(
            outcome.result,
            CompletionDecision,
            operation_name="Completion decision lookup result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        decision = validation.result
        del validation
        if decision is None or decision.decision_id != decision_id:
            del decision, store_owner
            raise WorkCompletionConflict(
                "Task store returned completion decision authority for another identity."
            ) from None
        return decision

    async def _load_optional_decision_for_proposal(
        self,
        store_owner: _TaskStoreOwner,
        proposal_id: str,
    ) -> CompletionDecision | None:
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.load_completion_decision_for_proposal(proposal_id),
            operation_name="Completion proposal decision lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        if outcome.result is None:
            return None
        validation = _copy_exact_model(
            outcome.result,
            CompletionDecision,
            operation_name="Completion proposal decision lookup result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        decision = validation.result
        del validation
        if decision is None or decision.proposal_id != proposal_id:
            del decision, store_owner
            raise WorkCompletionConflict(
                "Task store returned completion decision authority for another proposal."
            ) from None
        return decision

    async def _load_converged_decision(
        self,
        store_owner: _TaskStoreOwner,
        *,
        decision_id: str,
        proposal_id: str,
    ) -> CompletionDecision | None:
        by_id = await self._load_optional_decision(store_owner, decision_id)
        by_proposal = await self._load_optional_decision_for_proposal(store_owner, proposal_id)
        if by_id is None and by_proposal is not None:
            if by_proposal.decision_id != decision_id:
                del by_id, by_proposal, store_owner
                raise WorkCompletionConflict(
                    "Completion proposal has conflicting authority from a different durable "
                    "decision."
                ) from None
            # Both indexes are insert-only and must publish atomically, but
            # separate reads can straddle a concurrent exact publication.
            # Confirm the first miss once before classifying durable corruption.
            by_id = await self._load_optional_decision(store_owner, decision_id)
            if by_id is not None:
                if by_id != by_proposal:
                    del by_id, by_proposal, store_owner
                    raise WorkCompletionConflict(
                        "Task store returned non-convergent completion decision indexes with "
                        "conflicting authority."
                    ) from None
                return by_id
            del by_id, by_proposal, store_owner
            raise WorkCompletionConflict(
                "Task store omitted the completion decision from its identity index."
            ) from None
        if by_id is not None and by_proposal is None:
            by_proposal = await self._load_optional_decision_for_proposal(
                store_owner,
                proposal_id,
            )
            if by_proposal is not None:
                if by_id != by_proposal:
                    del by_id, by_proposal, store_owner
                    raise WorkCompletionConflict(
                        "Task store returned non-convergent completion decision indexes with "
                        "conflicting authority."
                    ) from None
                return by_id
            del by_id, by_proposal, store_owner
            raise WorkCompletionConflict(
                "Task store omitted the completion decision from its proposal index."
            ) from None
        if by_id is not None and by_id != by_proposal:
            del by_id, by_proposal, store_owner
            raise WorkCompletionConflict(
                "Task store returned non-convergent completion decision indexes with "
                "conflicting authority."
            ) from None
        return by_id

    def _require_exact_claim(
        self,
        claim: CompletionVerificationClaim,
        request: CompletionVerifierExecutionRequest,
        verifier: CompletionVerifierRef,
        *,
        execution_owner_id: str | None,
        verifier_profile_fingerprint: str,
        accept_legacy_timeout: bool = False,
    ) -> None:
        if execution_owner_id is not None and _contains_workload_secret(
            (execution_owner_id,),
            redactor=self._secret_redactor,
        ):
            del claim, request, verifier, execution_owner_id
            raise WorkCompletionConflict(
                "Completion verification claim contains unsafe execution authority."
            ) from None
        expected_timeout = (
            None
            if accept_legacy_timeout and claim.execution_timeout_seconds is None
            else request.execution_timeout_seconds
        )
        expected = CompletionVerificationClaimRequest(
            claim_id=request.claim_id,
            proposal_id=request.proposal_id,
            worker_id=request.worker_id,
            execution_owner_id=execution_owner_id,
            verifier=verifier,
            verifier_profile_fingerprint=verifier_profile_fingerprint,
            lease_seconds=request.lease_seconds,
            execution_timeout_seconds=expected_timeout,
        )
        if (
            claim.claim_id != request.claim_id
            or claim.proposal_id != request.proposal_id
            or claim.worker_id != request.worker_id
            or claim.execution_owner_id != execution_owner_id
            or claim.execution_timeout_seconds != expected_timeout
            or claim.verifier != verifier
            or claim.verifier_profile_fingerprint != verifier_profile_fingerprint
            or claim.request_sha256 != completion_verification_claim_request_sha256(expected)
        ):
            del claim, request, verifier, expected, expected_timeout
            raise WorkCompletionConflict(
                "Completion verification claim conflicts with the exact execution request."
            ) from None

    def _require_existing_decision(
        self,
        decision: CompletionDecision,
        *,
        request: CompletionVerifierExecutionRequest,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
        claim: CompletionVerificationClaim,
    ) -> bool:
        if (
            decision.decision_id != request.decision_id
            or decision.proposal_id != proposal.proposal_id
            or decision.claim_id != claim.claim_id
            or decision.worker_id != request.worker_id
            or decision.verifier != contract.verifier
            or decision.task_id != proposal.task_id
            or decision.attempt_id != attempt.attempt_id
            or decision.contract != contract.reference()
        ):
            raise WorkCompletionConflict(
                "Durable completion decision conflicts with the exact execution request."
            ) from None
        require_completion_decision_integrity(
            decision=decision,
            proposal=proposal,
            attempt=attempt,
            contract=contract,
        )
        if not completion_decision_claim_authority_matches(
            decision=decision,
            claim=claim,
            proposal=proposal,
            contract=contract,
        ):
            raise WorkCompletionConflict(
                "Durable completion decision conflicts with its verification-claim authority."
            ) from None
        return True

    def _decision_matches_request(
        self,
        decision: CompletionDecision,
        request: CompletionDecisionCreate,
        *,
        claim: CompletionVerificationClaim,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
        contract: WorkContract,
    ) -> bool:
        returned_request = completion_decision_request_from_record(decision)
        validate_completion_decision_contract(contract, returned_request)
        return (
            returned_request == request
            and completion_decision_claim_authority_matches(
                decision=decision,
                claim=claim,
                proposal=proposal,
                contract=contract,
            )
            and decision.request_sha256 == completion_decision_request_sha256(returned_request)
            and decision.gap_fingerprint == completion_gap_fingerprint(returned_request)
            and decision.proposal_id == proposal.proposal_id
            and decision.task_id == proposal.task_id
            and decision.attempt_id == attempt.attempt_id
            and decision.contract == contract.reference()
        )

    @staticmethod
    def _require_context_integrity(
        *,
        proposal: CompletionProposal,
        attempt: WorkAttempt,
    ) -> None:
        require_completion_proposal_integrity(proposal=proposal, attempt=attempt)

    async def _renew_claim(
        self,
        store_owner: _TaskStoreOwner,
        claim_request: CompletionVerificationClaimRequest,
        execution_request: CompletionVerifierExecutionRequest,
        *,
        prior_claim: CompletionVerificationClaim,
    ) -> tuple[CompletionVerificationClaim, float]:
        renewal_started_monotonic = monotonic()
        outcome = await capture_task_store_operation(
            lambda: store_owner.store.renew_completion_verification_claim(claim_request),
            operation_name="Completion verification claim renewal",
            redactor=self._secret_redactor,
            mutation_store=store_owner.store,
            mutation_method_name="renew_completion_verification_claim",
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        validation = _copy_exact_model(
            outcome.result,
            CompletionVerificationClaim,
            operation_name="Completion verification claim renewal result validation",
            redactor=self._secret_redactor,
        )
        del outcome
        if validation.failure is not None:
            raise validation.failure from None
        renewed = validation.result
        del validation
        if renewed is None:
            raise WorkCompletionConflict(
                "Task store returned an invalid renewed completion verification claim."
            ) from None
        self._require_exact_claim(
            renewed,
            execution_request,
            claim_request.verifier,
            execution_owner_id=claim_request.execution_owner_id,
            verifier_profile_fingerprint=claim_request.verifier_profile_fingerprint,
        )
        if (
            renewed.attempt_number != prior_claim.attempt_number
            or renewed.claimed_at != prior_claim.claimed_at
            or renewed.lease_expires_at < prior_claim.lease_expires_at
        ):
            del renewed, prior_claim, store_owner, claim_request, execution_request
            raise WorkCompletionConflict(
                "Completion verification claim renewal did not compare-and-extend "
                "the existing authority."
            ) from None
        claim_deadline_monotonic = renewal_started_monotonic + claim_request.lease_seconds
        if monotonic() >= claim_deadline_monotonic:
            raise CompletionVerificationClaimLost(
                "Completion verification claim renewal acknowledgement consumed its lease."
            ) from None
        return renewed, claim_deadline_monotonic

    def _start_claim_heartbeat(
        self,
        store_owner: _TaskStoreOwner,
        claim_request: CompletionVerificationClaimRequest,
        execution_request: CompletionVerifierExecutionRequest,
        claim: CompletionVerificationClaim,
        *,
        claim_deadline_monotonic: float,
        capacity_reservation: object,
    ) -> _ClaimHeartbeat:
        stop = asyncio.Event()
        ownership_lost: asyncio.Future[BaseException] = asyncio.get_running_loop().create_future()
        state = _ClaimHeartbeatState(ownership_lost=ownership_lost)
        task = self._adapter_runner.start_heartbeat(
            self._heartbeat_claim(
                store_owner,
                claim_request,
                execution_request,
                claim,
                stop,
                claim_deadline_monotonic,
                state,
            ),
            name="cayu-completion-verifier-claim-heartbeat",
        )
        control = _ClaimHeartbeat(
            stop=stop,
            task=task,
            state=state,
            capacity_reservation=capacity_reservation,
            cancellation_marker=_ClaimHeartbeatCancellationMarker(),
            shutdown_marker=_ClaimHeartbeatShutdownMarker(),
        )

        def settled(completed: asyncio.Task[None]) -> None:
            with suppress(BaseException):
                completed.result()

        task.add_done_callback(settled)
        return control

    async def _heartbeat_claim(
        self,
        store_owner: _TaskStoreOwner,
        claim_request: CompletionVerificationClaimRequest,
        execution_request: CompletionVerifierExecutionRequest,
        claim: CompletionVerificationClaim,
        stop: asyncio.Event,
        claim_deadline_monotonic: float,
        state: _ClaimHeartbeatState,
    ) -> None:
        await run_leased_adapter_heartbeat(
            _ClaimLeaseHeartbeat(
                self,
                store_owner,
                claim_request,
                execution_request,
                claim,
                claim_deadline_monotonic,
                state,
            ),
            stop=stop,
            ownership_lost=state.ownership_lost,
            clock=monotonic,
        )

    @staticmethod
    def _request_claim_heartbeat_stop(heartbeat: _ClaimHeartbeat) -> bool:
        """Request owned shutdown and retain the task until it settles."""

        heartbeat.stop.set()
        if heartbeat.task.done():
            return False
        if heartbeat.ownership_lost.done():
            # Ownership loss already put the heartbeat into owned settlement.
            # Cancelling it here can replace a later renewal-store failure
            # before the drain owner has observed that evidence.
            return False
        heartbeat.shutdown_requested = True
        return heartbeat.task.cancel(heartbeat.shutdown_marker)

    async def _settle_claim_heartbeat(
        self,
        heartbeat: _ClaimHeartbeat,
    ) -> _ClaimHeartbeatSettlement:
        try:
            return await self._settle_claim_heartbeat_owned(heartbeat)
        finally:
            self._release_adapter_capacity_reservation(heartbeat.capacity_reservation)

    async def _settle_claim_heartbeat_owned(
        self,
        heartbeat: _ClaimHeartbeat,
    ) -> _ClaimHeartbeatSettlement:
        cancel_requested = self._request_claim_heartbeat_stop(heartbeat)
        owned_shutdown_requested = heartbeat.shutdown_requested
        shielded = await await_shielded_task_outcome(
            heartbeat.task,
            timeout_s=None,
            timeout_after_cancellation_s=None,
        )
        failure = shielded.error
        if failure is not None and id(failure) == heartbeat.observed_failure_id:
            failure = heartbeat.state.late_settlement_failure
        elif failure is not None and (cancel_requested or owned_shutdown_requested):
            removed_shutdown_cancellation = False

            def is_owned_shutdown(leaf: BaseException) -> bool:
                nonlocal removed_shutdown_cancellation
                if not isinstance(leaf, asyncio.CancelledError):
                    return False
                if _is_exception_marker(leaf, heartbeat.shutdown_marker):
                    removed_shutdown_cancellation = True
                    return True
                # The task-store boundary detaches cancellation arguments.  A
                # successful Task.cancel() against this private task therefore
                # authenticates the one resulting cancellation leaf.
                if not removed_shutdown_cancellation:
                    removed_shutdown_cancellation = True
                    return True
                return False

            def shutdown_settlement_evidence(
                leaf: BaseException,
            ) -> BaseException | None:
                try:
                    cause = _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__get__(leaf, BaseException)
                except BaseException:
                    return RuntimeError(
                        "Completion-verification claim renewal settlement "
                        "evidence was inaccessible."
                    )
                return cause if isinstance(cause, BaseException) else None

            failure = _prune_exception_graph(
                failure,
                should_prune=is_owned_shutdown,
                pruned_leaf_replacement=shutdown_settlement_evidence,
            )
            if isinstance(failure, BaseExceptionGroup):
                failure = _credential_safe_verifier_group(
                    failure,
                    group_message="Completion verifier heartbeat settlement failed.",
                    redactor=self._secret_redactor,
                )
        caller_cancellation = (
            None
            if shielded.cancellation is None
            else _safe_caller_cancellation(
                shielded.cancellation,
                redactor=self._secret_redactor,
            )
        )
        return _ClaimHeartbeatSettlement(
            failure=failure,
            caller_cancellation=caller_cancellation,
            cancellation_requests_consumed=shielded.cancellation_requests_consumed,
        )

    def _failure_after_heartbeat_settlement(
        self,
        primary: BaseException | None,
        settlement: _ClaimHeartbeatSettlement,
        *,
        group_message: str,
    ) -> BaseException | None:
        failures: list[BaseException] = []
        if primary is not None:
            failures.append(primary)
        if settlement.failure is not None and all(
            settlement.failure is not failure for failure in failures
        ):
            failures.append(settlement.failure)
        caller_cancellation = settlement.caller_cancellation
        if caller_cancellation is not None and (
            primary is None or not _exception_tree_has_cancellation(primary)
        ):
            failures.append(caller_cancellation)

        if not failures:
            propagated: BaseException | None = None
        elif len(failures) == 1:
            propagated = failures[0]
        else:
            propagated = _ordered_failure_group(
                group_message,
                *failures,
                redactor=self._secret_redactor,
            )
        if settlement.cancellation_requests_consumed:
            if propagated is not None:
                retain_workspace_observation_pending_cancellation_requests(
                    propagated,
                    max(
                        settlement.cancellation_requests_consumed,
                        workspace_observation_pending_cancellation_requests(propagated),
                    ),
                )
            restore_task_cancellation_requests(
                settlement.cancellation_requests_consumed,
                cancellation=caller_cancellation,
            )
        return propagated

    @staticmethod
    def _claim_heartbeat_failure(heartbeat: _ClaimHeartbeat) -> BaseException | None:
        if heartbeat.ownership_lost.done():
            failure = heartbeat.ownership_lost.result()
            heartbeat.observed_failure_id = id(failure)
            return failure
        if not heartbeat.task.done() or heartbeat.task.cancelled():
            return None
        try:
            heartbeat.task.result()
        except BaseException as failure:
            heartbeat.observed_failure_id = id(failure)
            return failure
        return None

    def _provider_invocation(
        self,
        store: TaskStore,
        verifier: ProviderCompletionVerifier,
        target: ProviderCompletionVerifierTarget,
        authority: ProviderVerifierExecutionAuthority,
    ) -> Callable[[CompletionVerifierRequest], Awaitable[CompletionVerifierDecision]]:
        runtime = self._provider_runtime

        async def invoke(request: CompletionVerifierRequest) -> CompletionVerifierDecision:
            return await runtime.evaluate(
                store=store,
                verifier=verifier,
                target=target,
                request=request,
                authority=authority,
            )

        return invoke

    async def _invoke_adapter(
        self,
        verifier: Callable[[CompletionVerifierRequest], Awaitable[CompletionVerifierDecision]],
        request: CompletionVerifierRequest,
        *,
        operation_key: _ExecutionKey,
        timeout_seconds: float,
        heartbeat: _ClaimHeartbeat,
        deadline: ExecutionDeadline,
        retained_drain: asyncio.Future[_DrainingAdapter | None] | None = None,
    ) -> CompletionVerifierDecision:
        deadline.require_admission("completion_verifier")
        remaining = deadline.remaining_seconds()
        if remaining is not None:
            timeout_seconds = min(timeout_seconds, remaining)
        try:
            invocation = await self._adapter_runner.run(
                operation_key,
                lambda adapter=verifier, value=request: adapter(value),
                name="cayu-completion-verifier",
                timeout_seconds=timeout_seconds,
                lease=LeasedAdapterLease(
                    ownership_lost=heartbeat.ownership_lost,
                    heartbeat=heartbeat.task,
                    marker=heartbeat.cancellation_marker,
                ),
                on_settled=self._release_adapter_task,
            )
        finally:
            del verifier, request
        task, shielded = invocation.task, invocation.outcome
        del invocation
        if shielded.cancellation is not None:
            cancellation = shielded.cancellation
            cancellation_requests_consumed = shielded.cancellation_requests_consumed
            captured_during_cancellation = shielded.result
            heartbeat_failure = self._claim_heartbeat_failure(heartbeat)
            execution_failures = _ordered_execution_failures(
                adapter_failures=(
                    _adapter_failure_after_heartbeat(
                        shielded.error,
                        heartbeat=heartbeat,
                        heartbeat_failure=heartbeat_failure,
                    ),
                    _adapter_failure_after_heartbeat(
                        (
                            None
                            if captured_during_cancellation is None
                            else captured_during_cancellation.error
                        ),
                        heartbeat=heartbeat,
                        heartbeat_failure=heartbeat_failure,
                    ),
                ),
                boundary_failures=(heartbeat_failure,),
                redactor=self._secret_redactor,
            )
            self._cancel_and_retain_adapter_task(
                operation_key,
                task,
                heartbeat,
                observed_adapter_failures=tuple(
                    error
                    for error in (
                        shielded.error,
                        None
                        if captured_during_cancellation is None
                        else captured_during_cancellation.error,
                    )
                    if error is not None
                ),
                retained_drain=retained_drain,
            )
            safe = _safe_caller_cancellation(
                cancellation,
                redactor=self._secret_redactor,
            )
            if not execution_failures:
                propagated: BaseException = safe
            else:
                propagated = _ordered_failure_group(
                    "Completion verifier failed concurrently with caller cancellation.",
                    *execution_failures,
                    safe,
                    redactor=self._secret_redactor,
                )
                retain_workspace_observation_pending_cancellation_requests(
                    propagated,
                    max(cancellation_requests_consumed, 1),
                )
            del cancellation, captured_during_cancellation, heartbeat_failure
            del execution_failures
            del shielded, task
            restore_task_cancellation_requests(
                cancellation_requests_consumed,
                cancellation=safe,
            )
            raise propagated from None
        if shielded.timed_out:
            heartbeat_failure = self._claim_heartbeat_failure(heartbeat)
            self._cancel_and_retain_adapter_task(
                operation_key, task, heartbeat, retained_drain=retained_drain
            )
            timeout_failure = CompletionVerifierExecutionError(
                "Completion verifier exceeded its bounded execution timeout."
            )
            execution_failures = _ordered_execution_failures(
                boundary_failures=(timeout_failure, heartbeat_failure),
                redactor=self._secret_redactor,
            )
            del heartbeat_failure, timeout_failure
            del shielded, task
            if len(execution_failures) == 1:
                raise execution_failures[0] from None
            raise _ordered_failure_group(
                "Completion verifier timed out while claim renewal failed.",
                *execution_failures,
                redactor=self._secret_redactor,
            ) from None
        task_error = shielded.error
        captured = shielded.result
        self._release_adapter_task(operation_key, task)
        del shielded, task
        heartbeat_failure = self._claim_heartbeat_failure(heartbeat)
        captured_error = None if captured is None else captured.error
        captured_result = None if captured is None else captured.result
        captured_was_missing = captured is None
        execution_failures = _ordered_execution_failures(
            adapter_failures=(
                _adapter_failure_after_heartbeat(
                    task_error,
                    heartbeat=heartbeat,
                    heartbeat_failure=heartbeat_failure,
                ),
                _adapter_failure_after_heartbeat(
                    captured_error,
                    heartbeat=heartbeat,
                    heartbeat_failure=heartbeat_failure,
                ),
            ),
            boundary_failures=(heartbeat_failure,),
            redactor=self._secret_redactor,
        )
        del task_error, captured_error, captured, heartbeat_failure
        if execution_failures:
            del captured_result, captured_was_missing
            if len(execution_failures) == 1:
                raise execution_failures[0] from None
            raise _ordered_failure_group(
                "Completion verifier execution and claim renewal failed.",
                *execution_failures,
                redactor=self._secret_redactor,
            ) from None
        del execution_failures
        if captured_was_missing:
            del captured_result
            raise CompletionVerifierExecutionError(
                "Completion verifier returned no captured outcome."
            ) from None
        del captured_was_missing
        validation = capture_sensitive_validation(
            lambda value=captured_result: copy_completion_verifier_decision(
                cast("CompletionVerifierDecision", value)
            ),
            operation_name="Completion verifier decision validation",
            redactor=self._secret_redactor,
        )
        del captured_result
        if validation.failure is not None:
            raise validation.failure from None
        decision = validation.result
        del validation
        if decision is None:
            raise CompletionVerifierExecutionError(
                "Completion verifier returned an invalid decision."
            ) from None
        deadline.require_admission("completion_verifier_result")
        return decision

    def _reserve_adapter_capacity(self) -> object:
        return self._adapter_runner.reserve_capacity(
            _MAX_ACTIVE_COMPLETION_VERIFIERS,
            lambda: CompletionVerifierExecutionError(
                "Completion verifier execution capacity is exhausted."
            ),
        )

    def _release_adapter_capacity_reservation(self, reservation: object) -> None:
        self._adapter_runner.release_capacity(reservation)

    def _cancel_and_retain_adapter_task(
        self,
        operation_key: _ExecutionKey,
        task: asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]],
        heartbeat: _ClaimHeartbeat,
        *,
        observed_adapter_failures: tuple[BaseException, ...] = (),
        retained_drain: asyncio.Future[_DrainingAdapter | None] | None = None,
    ) -> None:
        heartbeat.retained_for_drain = True
        draining = _DrainingAdapter(
            operation_key=operation_key,
            task=task,
            heartbeat=heartbeat,
            cancellation_marker=_AdapterDrainCancellationMarker(),
            observed_adapter_failures=observed_adapter_failures,
        )
        self._adapter_runner.retain_drain(operation_key, draining)
        if retained_drain is not None:
            # Its owner observes and reports the settlement failure itself.
            draining.failure_reported = True
            retained_drain.set_result(draining)
        if task.done():
            self._ensure_draining_adapter_settlement(operation_key, draining)
            return
        task.cancel(draining.cancellation_marker)

    async def _settle_draining_adapter(
        self, draining: _DrainingAdapter
    ) -> _ClaimHeartbeatSettlement:
        # Only the terminal adapter callback (or an equivalent terminal read)
        # starts settlement. Cancellation delivery is not this evidence.
        if not draining.task.done():
            raise AssertionError("Cannot settle a running completion verifier.")
        try:
            captured = draining.task.result()
            adapter_failure = captured.error
        except BaseException as failure:
            adapter_failure = failure
        if any(adapter_failure is observed for observed in draining.observed_adapter_failures):
            adapter_failure = None
        if adapter_failure is not None:
            adapter_failure = _prune_exception_graph(
                adapter_failure,
                should_prune=lambda leaf: (
                    isinstance(leaf, asyncio.CancelledError)
                    and (
                        _is_exception_marker(leaf, draining.cancellation_marker)
                        or _is_exception_marker(leaf, draining.heartbeat.cancellation_marker)
                    )
                ),
                pruned_leaf_replacement=lambda leaf: _BASE_EXCEPTION_CAUSE_DESCRIPTOR.__get__(
                    leaf, BaseException
                ),
            )
            if adapter_failure is not None:
                # A shutdown cause can also be an explicit group sibling.
                # Canonicalize that overlap by identity after replacements.
                adapter_failure = _prune_exception_graph(
                    adapter_failure, should_prune=lambda _leaf: False
                )
        failures = _ordered_execution_failures(
            adapter_failures=(adapter_failure,), redactor=self._secret_redactor
        )
        del adapter_failure
        settlement = await self._settle_claim_heartbeat(draining.heartbeat)
        if not failures:
            return settlement
        combined = (
            failures[0]
            if settlement.failure is None
            else _ordered_failure_group(
                "Completion verifier adapter and heartbeat drain failed.",
                *failures,
                settlement.failure,
                redactor=self._secret_redactor,
            )
        )
        return _ClaimHeartbeatSettlement(
            failure=combined,
            caller_cancellation=settlement.caller_cancellation,
            cancellation_requests_consumed=settlement.cancellation_requests_consumed,
        )

    def _ensure_draining_adapter_settlement(
        self,
        operation_key: _ExecutionKey,
        draining: _DrainingAdapter,
    ) -> asyncio.Task[_ClaimHeartbeatSettlement]:
        return self._adapter_runner.start_settlement(
            operation_key,
            draining,
            lambda: self._settle_draining_adapter(draining),
            name="cayu-completion-verifier-drain-settlement",
            failure_for=lambda settlement: settlement.failure,
        )

    def _acknowledge_owned_drain(self, draining: _DrainingAdapter) -> None:
        completed = draining.settlement_task
        if (
            not draining.task.done()
            or completed is None
            or not completed.done()
            or not self._adapter_runner.finalize_settlement(
                draining.operation_key,
                draining,
                completed,
                failure_for=lambda settlement: settlement.failure,
            )
        ):
            raise RuntimeError("Completion verifier cleanup has not settled.")
        self._adapter_runner.acknowledge_drain(draining.operation_key, draining)

    def _release_adapter_task(
        self,
        operation_key: _ExecutionKey,
        completed: asyncio.Task[CapturedAwaitableOutcome[CompletionVerifierDecision]],
    ) -> None:
        draining = self._adapter_runner.draining(operation_key)
        if draining is not None and draining.task is completed:
            self._ensure_draining_adapter_settlement(operation_key, draining)

    @staticmethod
    def _execution_key(request: CompletionVerifierExecutionRequest) -> _ExecutionKey:
        return request.proposal_id


__all__ = ["CompletionVerifierCoordinator"]
