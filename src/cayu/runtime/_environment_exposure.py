"""Process-local authority for one runtime-admitted environment exposure."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, SupportsIndex

from cayu._task_wait import await_shielded_task_outcome, restore_task_cancellation_requests
from cayu.environments import (
    ExecutionAdmissionCandidate,
    ExecutionAdmissionDecision,
    ExecutionAdmissionError,
    ExecutionEnvironmentAuthority,
    evaluate_execution_admission,
)
from cayu.environments.admission import (
    _copy_execution_admission_candidate,
    _structured_execution_refusal,
)
from cayu.environments.factory import (
    attach_environment_factory_cleanup_settlement_task,
    combine_environment_factory_cleanup_settlement_tasks,
    environment_factory_cleanup_settlement_tasks,
    retry_environment_factory_cleanup_settlement_task,
)
from cayu.runtime import _environment_operation_boundary as environment_operation_boundary
from cayu.vaults import SecretRedactor

if TYPE_CHECKING:
    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity
    from cayu.runtime.sessions import Session


_ENVIRONMENT_EXPOSURE_AUTHORITY_TOKEN = object()


@dataclass(slots=True, repr=False)
class _EnvironmentExposureAdmission:
    """Mutable dispatch ownership and freshness for one authenticated exposure."""

    interaction_id: str | None
    decision: ExecutionAdmissionDecision
    renewal_lock: asyncio.Lock
    settlement_task: asyncio.Task[None] | None = None


@dataclass(frozen=True, slots=True, repr=False)
class _EnvironmentExposure:
    """Unserializable proof that one exact live environment crossed the lifecycle gate."""

    token: object
    session_id: str
    session_instance_id: str
    run_epoch: int
    registered_agent: Any
    execution_profile: Any
    candidate: str
    execution_environment_authority: ExecutionEnvironmentAuthority | None
    binding_generation_id: str
    environment: Any
    runner: Any
    workspace: Any
    bound_workspace: Any
    binding: Any
    artifact_store: Any
    vault: Any
    proxy: Any
    knowledge_store: Any
    decision: ExecutionAdmissionDecision
    admission: _EnvironmentExposureAdmission

    def __repr__(self) -> str:
        return "_EnvironmentExposure(<authenticated>)"

    def __reduce_ex__(self, _protocol: SupportsIndex, /) -> str | tuple[Any, ...]:
        raise TypeError("Environment exposure authority has no serialization form.")


def expose_registered_environment(
    registered_environment: runtime_records.RegisteredEnvironment,
    *,
    session: Session,
    invocation_context: InvocationContext | None,
    registered_agent: runtime_records.RegisteredAgentState,
    execution_profile: ExecutionProfileIdentity | None,
    decision: ExecutionAdmissionDecision,
) -> runtime_records.RegisteredEnvironment:
    """Mint the only runtime-owned admitted/exposed form of a live environment."""

    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity
    from cayu.runtime.sessions import Session

    if type(registered_environment) is not runtime_records.RegisteredEnvironment:
        raise TypeError("registered_environment must be a RegisteredEnvironment.")
    if type(session) is not Session:
        raise TypeError("session must be a Session.")
    if type(registered_agent) is not runtime_records.RegisteredAgentState:
        raise TypeError("registered_agent must be a RegisteredAgentState.")
    if execution_profile is not None and type(execution_profile) is not ExecutionProfileIdentity:
        raise TypeError("execution_profile must be an ExecutionProfileIdentity or None.")
    if type(decision) is not ExecutionAdmissionDecision or decision.status != "admitted":
        raise TypeError("Environment exposure requires an admitted execution decision.")
    if registered_environment.environment_exposure is not None:
        raise RuntimeError("Environment exposure authority was already minted.")
    if registered_environment.execution_candidate != decision.candidate:
        raise RuntimeError("Environment exposure decision changed the selected candidate.")

    interaction_id = None
    if invocation_context is not None:
        if type(invocation_context) is not InvocationContext:
            raise TypeError("invocation_context must be an InvocationContext or None.")
        if (
            invocation_context.binding.session_id != session.id
            or invocation_context.binding.session_instance_id != session.instance_id
            or invocation_context.binding.run_epoch != session.run_epoch
            or invocation_context.registered_agent is not registered_agent
            or invocation_context.profile is not execution_profile
        ):
            raise RuntimeError("Environment exposure lost frozen invocation authority.")
        interaction_id = invocation_context.binding.interaction_id

    environment = registered_environment.environment
    exposure = _EnvironmentExposure(
        token=_ENVIRONMENT_EXPOSURE_AUTHORITY_TOKEN,
        session_id=session.id,
        session_instance_id=session.instance_id,
        run_epoch=session.run_epoch,
        registered_agent=registered_agent,
        execution_profile=execution_profile,
        candidate=decision.candidate,
        execution_environment_authority=(registered_environment.execution_environment_authority),
        binding_generation_id=registered_environment.binding_generation_id,
        environment=environment,
        runner=environment.runner,
        workspace=environment.workspace,
        bound_workspace=registered_environment.bound_workspace,
        binding=environment.binding,
        artifact_store=environment.artifact_store,
        vault=environment.vault,
        proxy=environment.proxy,
        knowledge_store=environment.knowledge_store,
        decision=decision,
        admission=_EnvironmentExposureAdmission(
            interaction_id=interaction_id,
            decision=decision,
            renewal_lock=asyncio.Lock(),
        ),
    )
    return replace(registered_environment, environment_exposure=exposure)


def _environment_exposure(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    session: Session,
    invocation_context: InvocationContext,
    registered_agent: runtime_records.RegisteredAgentState,
    execution_profile: ExecutionProfileIdentity,
) -> _EnvironmentExposure | None:
    """Return the exact lifecycle-minted authority after structural validation."""

    from cayu.runtime import _runtime_records as runtime_records
    from cayu.runtime._invocation_lifecycle import InvocationContext
    from cayu.runtime.execution_profiles import ExecutionProfileIdentity
    from cayu.runtime.sessions import Session

    if registered_environment is None:
        if invocation_context.registered_environment is not None:
            raise RuntimeError("Environment-free dispatch conflicts with invocation authority.")
        return None
    if type(registered_environment) is not runtime_records.RegisteredEnvironment:
        raise TypeError("registered_environment must be a RegisteredEnvironment or None.")
    if type(session) is not Session:
        raise TypeError("session must be a Session.")
    if type(invocation_context) is not InvocationContext:
        raise TypeError("invocation_context must be an InvocationContext.")
    if type(registered_agent) is not runtime_records.RegisteredAgentState:
        raise TypeError("registered_agent must be a RegisteredAgentState.")
    if type(execution_profile) is not ExecutionProfileIdentity:
        raise TypeError("execution_profile must be an ExecutionProfileIdentity.")

    exposure = registered_environment.environment_exposure
    environment = registered_environment.environment
    if (
        type(exposure) is not _EnvironmentExposure
        or exposure.token is not _ENVIRONMENT_EXPOSURE_AUTHORITY_TOKEN
        or invocation_context.registered_environment is not registered_environment
        or invocation_context.registered_agent is not registered_agent
        or invocation_context.profile is not execution_profile
        or exposure.session_id != session.id
        or exposure.session_instance_id != session.instance_id
        or exposure.run_epoch != session.run_epoch
        or exposure.registered_agent is not registered_agent
        or exposure.execution_profile is not execution_profile
        or exposure.candidate != registered_environment.execution_candidate
        or exposure.execution_environment_authority
        is not registered_environment.execution_environment_authority
        or exposure.binding_generation_id != registered_environment.binding_generation_id
        or exposure.environment is not environment
        or exposure.runner is not environment.runner
        or exposure.workspace is not environment.workspace
        or exposure.bound_workspace is not registered_environment.bound_workspace
        or exposure.binding is not environment.binding
        or exposure.artifact_store is not environment.artifact_store
        or exposure.vault is not environment.vault
        or exposure.proxy is not environment.proxy
        or exposure.knowledge_store is not environment.knowledge_store
        or exposure.decision.status != "admitted"
        or exposure.decision.candidate != exposure.candidate
        or type(exposure.admission) is not _EnvironmentExposureAdmission
        or exposure.admission.interaction_id != invocation_context.binding.interaction_id
        or type(exposure.admission.renewal_lock) is not asyncio.Lock
        or type(exposure.admission.decision) is not ExecutionAdmissionDecision
        or (
            exposure.admission.settlement_task is not None
            and not isinstance(exposure.admission.settlement_task, asyncio.Task)
        )
        or exposure.admission.decision.status != "admitted"
        or exposure.admission.decision.candidate != exposure.candidate
        or exposure.admission.decision.requirements != exposure.decision.requirements
        or _admission_identity(exposure.admission.decision)
        != _admission_identity(exposure.decision)
    ):
        raise RuntimeError(
            "Environment execution requires the exact runtime-admitted exposure authority."
        )
    return exposure


def transfer_queued_environment_exposure(
    *,
    session: Session,
    predecessor: InvocationContext,
    successor: InvocationContext,
) -> None:
    """Move dispatch ownership after an authenticated same-epoch queued handoff.

    Retain the exact environment and admission state, including evidence age and
    outstanding settlement. The predecessor can no longer authorize dispatch.
    """

    if (
        replace(predecessor.binding, interaction_id=successor.binding.interaction_id)
        != successor.binding
        or predecessor.binding.interaction_id == successor.binding.interaction_id
        or predecessor.profile is not successor.profile
        or predecessor.registered_agent is not successor.registered_agent
        or predecessor.registered_environment is not successor.registered_environment
    ):
        raise RuntimeError("Queued environment exposure lost its exact invocation successor.")
    exposure = _environment_exposure(
        predecessor.registered_environment,
        session=session,
        invocation_context=predecessor,
        registered_agent=predecessor.registered_agent,
        execution_profile=predecessor.profile,
    )
    if exposure is not None:
        exposure.admission.interaction_id = successor.binding.interaction_id


def _admission_identity(
    decision: ExecutionAdmissionDecision,
) -> tuple[str | None, str | None, str | None, str | None]:
    evidence = decision.evidence
    if evidence is None:
        return (decision.evidence_schema, None, None, None)
    return (
        evidence.schema_version,
        evidence.environment_fingerprint,
        evidence.image_fingerprint,
        evidence.toolchain_profile_fingerprint,
    )


def _evaluate_exposure_decision(
    exposure: _EnvironmentExposure,
) -> ExecutionAdmissionDecision:
    return evaluate_execution_admission(
        candidate=exposure.candidate,
        requirements=exposure.admission.decision.requirements,
        evidence=exposure.admission.decision.evidence,
        stage="pre_exposure",
    )


def require_environment_exposed(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    session: Session,
    invocation_context: InvocationContext,
    registered_agent: runtime_records.RegisteredAgentState,
    execution_profile: ExecutionProfileIdentity,
) -> None:
    """Fail closed unless the caller owns lifecycle-minted exposure authority."""

    _environment_exposure(
        registered_environment,
        session=session,
        invocation_context=invocation_context,
        registered_agent=registered_agent,
        execution_profile=execution_profile,
    )


def _retain_exposure_admission_settlement(
    exposure: _EnvironmentExposure,
    settlement_tasks: tuple[asyncio.Task[None], ...],
) -> None:
    """Keep opaque renewal ownership reachable by later terminal cleanup."""

    existing = exposure.admission.settlement_task
    tasks = settlement_tasks if existing is None else (existing, *settlement_tasks)
    settlement = combine_environment_factory_cleanup_settlement_tasks(
        tasks,
        task_name="cayu-environment-admission-renewal-settlement",
        failure_message="Environment admission renewal settlement failed.",
    )
    if settlement is not None:
        exposure.admission.settlement_task = settlement


@dataclass(frozen=True, slots=True)
class _RunnerAdmissionSnapshot:
    candidate: ExecutionAdmissionCandidate | None
    candidate_supplied: bool
    authority: object | None
    authority_failed: bool
    settlement_tasks: tuple[asyncio.Task[None], ...]


async def _runner_admission_snapshot(
    exposure: _EnvironmentExposure,
    *,
    redactor: SecretRedactor,
) -> _RunnerAdmissionSnapshot:
    runner = exposure.runner

    async def snapshot() -> _RunnerAdmissionSnapshot:
        candidate: ExecutionAdmissionCandidate | None = None
        candidate_supplied = False
        authority: object | None = None
        authority_failed = False
        settlement_tasks: tuple[asyncio.Task[None], ...] = ()
        try:
            supplied = runner.execution_admission_candidate()
        except Exception as error:
            settlement_tasks = environment_factory_cleanup_settlement_tasks(error)
            del error
        else:
            candidate_supplied = supplied is not None
            candidate = _copy_execution_admission_candidate(supplied)
            del supplied
        if exposure.execution_environment_authority is not None:
            try:
                authority = runner.execution_environment_authority()
            except Exception as error:
                authority_failed = True
                settlement_tasks = (
                    *settlement_tasks,
                    *environment_factory_cleanup_settlement_tasks(error),
                )
                del error
        return _RunnerAdmissionSnapshot(
            candidate=candidate,
            candidate_supplied=candidate_supplied,
            authority=authority,
            authority_failed=authority_failed,
            settlement_tasks=settlement_tasks,
        )

    try:
        return await environment_operation_boundary.await_environment_operation(
            snapshot,
            operation_name="Environment admission evidence snapshot",
            redactor=redactor,
        )
    except BaseException as error:
        settlement_tasks = environment_factory_cleanup_settlement_tasks(error)
        if not isinstance(error, Exception):
            _retain_exposure_admission_settlement(exposure, settlement_tasks)
            raise
        del error
        return _RunnerAdmissionSnapshot(
            candidate=None,
            candidate_supplied=False,
            authority=None,
            authority_failed=exposure.execution_environment_authority is not None,
            settlement_tasks=settlement_tasks,
        )


def _snapshot_decision(
    exposure: _EnvironmentExposure,
    snapshot: _RunnerAdmissionSnapshot,
) -> ExecutionAdmissionDecision:
    candidate = snapshot.candidate
    if candidate is None:
        return _structured_execution_refusal(
            candidate=exposure.candidate,
            requirements=exposure.decision.requirements,
            evidence=None,
            code=(
                "malformed_evidence" if snapshot.candidate_supplied else "missing_final_evidence"
            ),
        )
    if candidate.candidate != exposure.candidate:
        return _structured_execution_refusal(
            candidate=exposure.candidate,
            requirements=exposure.decision.requirements,
            evidence=candidate.evidence,
            code="evidence_candidate_mismatch",
        )
    if snapshot.authority_failed or (
        exposure.execution_environment_authority is not None
        and snapshot.authority is not exposure.execution_environment_authority
    ):
        return _structured_execution_refusal(
            candidate=exposure.candidate,
            requirements=exposure.decision.requirements,
            evidence=candidate.evidence,
            code="environment_authority_mismatch",
        )
    decision = evaluate_execution_admission(
        candidate=exposure.candidate,
        requirements=exposure.decision.requirements,
        evidence=candidate.evidence,
        stage="pre_exposure",
    )
    if _admission_identity(decision) != _admission_identity(exposure.decision):
        return _structured_execution_refusal(
            candidate=exposure.candidate,
            requirements=exposure.decision.requirements,
            evidence=candidate.evidence,
            code="environment_authority_mismatch",
        )
    return decision


def _only_stale_refusals(decision: ExecutionAdmissionDecision) -> bool:
    return (
        decision.status == "refused"
        and bool(decision.refusals)
        and all(refusal.code == "stale_evidence" for refusal in decision.refusals)
    )


async def _await_exposure_admission_settlement(exposure: _EnvironmentExposure) -> None:
    task = exposure.admission.settlement_task
    if task is None:
        return
    if task.done():
        try:
            task.result()
        except BaseException:
            replacement = retry_environment_factory_cleanup_settlement_task(task)
            if replacement is task:
                raise RuntimeError(
                    "Environment admission renewal settlement remains unproven."
                ) from None
            exposure.admission.settlement_task = replacement
            task = replacement
        else:
            exposure.admission.settlement_task = None
            return
    outcome = await await_shielded_task_outcome(task)
    cancellation = outcome.cancellation or outcome.subsequent_cancellation
    if outcome.error is None:
        exposure.admission.settlement_task = None
    cleanup_failure = (
        None
        if outcome.error is None
        else RuntimeError("Environment admission renewal settlement remains unproven.")
    )
    if cancellation is not None:
        restore_task_cancellation_requests(
            outcome.cancellation_requests_consumed,
            cancellation=cancellation,
        )
        if cleanup_failure is not None:
            raise cancellation from cleanup_failure
        raise cancellation
    if cleanup_failure is not None:
        raise cleanup_failure from None


async def await_environment_exposure_settlement(
    registered_environment: runtime_records.RegisteredEnvironment | None,
) -> None:
    """Fence terminal cleanup behind any authenticated renewal handoff."""

    if registered_environment is None:
        return
    exposure = registered_environment.environment_exposure
    if (
        type(exposure) is not _EnvironmentExposure
        or exposure.token is not _ENVIRONMENT_EXPOSURE_AUTHORITY_TOKEN
        or type(exposure.admission) is not _EnvironmentExposureAdmission
    ):
        return
    async with exposure.admission.renewal_lock:
        await _await_exposure_admission_settlement(exposure)


def _settlement_refusal(
    exposure: _EnvironmentExposure,
    decision: ExecutionAdmissionDecision,
) -> ExecutionAdmissionDecision:
    if decision.status == "refused":
        return decision
    return _structured_execution_refusal(
        candidate=exposure.candidate,
        requirements=exposure.decision.requirements,
        evidence=decision.evidence,
        code="missing_final_evidence",
    )


async def _settle_dispatch_owners(
    exposure: _EnvironmentExposure,
    decision: ExecutionAdmissionDecision,
    settlement_tasks: tuple[asyncio.Task[None], ...],
) -> None:
    settlement = combine_environment_factory_cleanup_settlement_tasks(
        settlement_tasks,
        task_name="cayu-environment-admission-renewal-settlement",
        failure_message="Environment admission renewal settlement failed.",
    )
    if settlement is not None:
        exposure.admission.settlement_task = settlement
        try:
            await _await_exposure_admission_settlement(exposure)
        except asyncio.CancelledError:
            raise
        except Exception as settlement_error:
            error = ExecutionAdmissionError(_settlement_refusal(exposure, decision))
            attach_environment_factory_cleanup_settlement_task(error, settlement)
            raise error from settlement_error


async def _raise_dispatch_refusal(
    exposure: _EnvironmentExposure,
    decision: ExecutionAdmissionDecision,
    settlement_tasks: tuple[asyncio.Task[None], ...],
) -> None:
    await _settle_dispatch_owners(
        exposure,
        decision,
        settlement_tasks,
    )
    decision.require_admitted()
    raise AssertionError("A dispatch refusal unexpectedly evaluated as admitted.")


async def refresh_and_require_environment_exposed(
    registered_environment: runtime_records.RegisteredEnvironment | None,
    *,
    session: Session,
    invocation_context: InvocationContext,
    registered_agent: runtime_records.RegisteredAgentState,
    execution_profile: ExecutionProfileIdentity,
    redactor: SecretRedactor,
) -> None:
    """Renew stale runner evidence, then authorize one exact dispatch."""

    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    exposure = _environment_exposure(
        registered_environment,
        session=session,
        invocation_context=invocation_context,
        registered_agent=registered_agent,
        execution_profile=execution_profile,
    )
    if exposure is None:
        return
    decision = _evaluate_exposure_decision(exposure)
    if decision.status == "admitted":
        return
    if not _only_stale_refusals(decision):
        decision.require_admitted()

    async with exposure.admission.renewal_lock:
        exposure = _environment_exposure(
            registered_environment,
            session=session,
            invocation_context=invocation_context,
            registered_agent=registered_agent,
            execution_profile=execution_profile,
        )
        assert exposure is not None
        await _await_exposure_admission_settlement(exposure)
        decision = _evaluate_exposure_decision(exposure)
        if decision.status == "admitted":
            return
        if not _only_stale_refusals(decision):
            decision.require_admitted()

        settlement_tasks: tuple[asyncio.Task[None], ...] = ()
        snapshot = await _runner_admission_snapshot(exposure, redactor=redactor)
        settlement_tasks = (*settlement_tasks, *snapshot.settlement_tasks)
        current = _snapshot_decision(exposure, snapshot)
        if settlement_tasks:
            # A candidate read may itself report an opaque operation still in
            # flight. Do not renew, release, or dispatch concurrently with it.
            await _settle_dispatch_owners(
                exposure,
                current,
                settlement_tasks,
            )
            snapshot = await _runner_admission_snapshot(exposure, redactor=redactor)
            settlement_tasks = snapshot.settlement_tasks
            current = _snapshot_decision(exposure, snapshot)
        if current.status == "admitted" and not settlement_tasks:
            exposure.admission.decision = current
            require_environment_exposed(
                registered_environment,
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                execution_profile=execution_profile,
            )
            return
        if not _only_stale_refusals(current):
            await _raise_dispatch_refusal(exposure, current, settlement_tasks)

        try:
            await environment_operation_boundary.await_environment_operation(
                exposure.runner.refresh_execution_admission,
                operation_name="Environment admission renewal",
                redactor=redactor,
            )
        except BaseException as error:
            transferred = environment_factory_cleanup_settlement_tasks(error)
            if not isinstance(error, Exception):
                _retain_exposure_admission_settlement(exposure, transferred)
                raise
            settlement_tasks = (*settlement_tasks, *transferred)
            del error

        snapshot = await _runner_admission_snapshot(exposure, redactor=redactor)
        settlement_tasks = (*settlement_tasks, *snapshot.settlement_tasks)
        renewed = _snapshot_decision(exposure, snapshot)
        if settlement_tasks:
            # A renewal can publish its candidate before acknowledgement while
            # handing off the dispatched probe. Settle that exact owner, then
            # reconcile from a new side-effect-free snapshot. This admits a
            # completed operation after acknowledgement loss without ever
            # treating an in-flight mutation as reusable.
            await _settle_dispatch_owners(
                exposure,
                renewed,
                settlement_tasks,
            )
            snapshot = await _runner_admission_snapshot(exposure, redactor=redactor)
            settlement_tasks = snapshot.settlement_tasks
            renewed = _snapshot_decision(exposure, snapshot)
        if renewed.status == "admitted" and not settlement_tasks:
            exposure.admission.decision = renewed
            require_environment_exposed(
                registered_environment,
                session=session,
                invocation_context=invocation_context,
                registered_agent=registered_agent,
                execution_profile=execution_profile,
            )
            return
        await _raise_dispatch_refusal(exposure, renewed, settlement_tasks)
