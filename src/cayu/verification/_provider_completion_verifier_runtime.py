"""Runtime-owned provider dispatch for provider-backed completion verifiers.

This owner runs inside the completion-verifier coordinator's adapter slot, so
claim heartbeats, execution timeout, caller cancellation, cancellation-resistant
draining and claim-fenced decision publication are the coordinator's. What it
adds is one governed provider invocation per attempt: the intent is persisted in
the task store under the live claim before the provider is entered, and every
attempt is settled with its observed outcome and usage, including after
cancellation. Nothing is written to a worker session.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import aclosing
from dataclasses import dataclass
from hashlib import sha256

from cayu._task_wait import await_shielded_task_outcome
from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.budgets.billing import BillingIdentity
from cayu.budgets.usage import UsageMetrics, normalize_usage_metrics
from cayu.deadlines import ExecutionDeadline, execution_deadline_scope, expired_execution_deadline
from cayu.providers import ModelProviderError, ModelRequest, ModelStreamEvent, ModelStreamEventType
from cayu.providers._credential_boundary import (
    release_provider_stream_cleanup,
    reserve_provider_stream_cleanup,
)
from cayu.providers.base import ModelStreamDeadlineError, _copy_auxiliary_request
from cayu.providers.deadlines import ProviderStreamDeadlineAdmission
from cayu.providers.response import ModelResponse
from cayu.runtime._auxiliary_response import AuxiliaryResponseCollector
from cayu.runtime._model_errors import (
    model_provider_error_from_payload,
    resolve_request_billing_identity,
)
from cayu.runtime._provider_stream import (
    _admitted_model_provider_events,
    _owned_model_provider_events,
)
from cayu.runtime._runtime_records import RegisteredProvider
from cayu.runtime._task_store_operation_boundary import (
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.runtime.retry_policy import RetryDecision, retry_decision
from cayu.tasks.completion_verifier_dispatches import (
    CompletionVerifierDecodeStatus,
    CompletionVerifierDispatch,
    CompletionVerifierDispatchBudgetExhausted,
    CompletionVerifierDispatchFailure,
    CompletionVerifierDispatchOutcome,
    CompletionVerifierDispatchRequest,
    CompletionVerifierDispatchSettlementRequest,
    CompletionVerifierUsageStatus,
    completion_verifier_dispatch_id,
)
from cayu.tasks.contracts import (
    CompletionVerifierDecision,
    CompletionVerifierRef,
    copy_completion_verifier_decision,
)
from cayu.tasks.store import TaskStore
from cayu.vaults.redaction import SecretRedactor
from cayu.verification.completion_verifiers import CompletionVerifierRequest
from cayu.verification.provider_completion_verifiers import (
    ProviderCompletionVerifier,
    ProviderCompletionVerifierBudgetExhausted,
    ProviderCompletionVerifierDecodingError,
    ProviderCompletionVerifierDispatchError,
    ProviderCompletionVerifierTarget,
    compose_provider_completion_verifier_messages,
    decode_provider_completion_verifier_decision,
)


@dataclass(frozen=True, slots=True)
class ProviderVerifierExecutionAuthority:
    """The live claim tuple one provider-verifier execution runs under."""

    proposal_id: str
    claim_id: str
    worker_id: str
    execution_owner_id: str
    claim_attempt_number: int
    verifier: CompletionVerifierRef
    verifier_profile_fingerprint: str


@dataclass(frozen=True, slots=True)
class _AttemptOutcome:
    settlement: CompletionVerifierDispatchSettlementRequest
    decision: CompletionVerifierDecision | None = None
    provider_failure: ModelProviderError | None = None
    retry: RetryDecision | None = None
    failure: BaseException | None = None


def reusable_provider_decision(
    dispatches: tuple[CompletionVerifierDispatch, ...],
    *,
    verifier_profile_fingerprint: str,
) -> CompletionVerifierDecision | None:
    """Return a durably settled decoded decision, so recovery never re-judges.

    A provider outcome that already committed for this proposal and profile is
    authoritative evidence: a later claim publishes it instead of paying for and
    possibly contradicting it with a second judge.
    """

    for dispatch in reversed(dispatches):
        settlement = dispatch.settlement
        if (
            settlement is not None
            and dispatch.request.verifier_profile_fingerprint == verifier_profile_fingerprint
            and settlement.outcome is CompletionVerifierDispatchOutcome.COMPLETED
            and settlement.decode_status is CompletionVerifierDecodeStatus.DECODED
            and settlement.decision is not None
        ):
            return copy_completion_verifier_decision(settlement.decision)
    return None


def _failure_code(error: ModelProviderError) -> str:
    status = error.status_code
    if status == 429:
        return "provider.rate_limited"
    if status is not None and 500 <= status <= 599:
        return "provider.server_error"
    if status is not None and 400 <= status <= 499:
        return "provider.request_rejected"
    return "provider.error"


class ProviderCompletionVerifierRuntime:
    """Dispatch provider attempts for one registered provider verifier."""

    def __init__(
        self,
        *,
        resolve_provider: Callable[[str], RegisteredProvider],
        secret_redactor: SecretRedactor,
    ) -> None:
        self._resolve_provider = resolve_provider
        self._secret_redactor = secret_redactor

    async def evaluate(
        self,
        *,
        store: TaskStore,
        verifier: ProviderCompletionVerifier,
        target: ProviderCompletionVerifierTarget,
        request: CompletionVerifierRequest,
        authority: ProviderVerifierExecutionAuthority,
    ) -> CompletionVerifierDecision:
        prior = await self._list_dispatches(store, authority.proposal_id)
        reusable = reusable_provider_decision(
            prior, verifier_profile_fingerprint=authority.verifier_profile_fingerprint
        )
        if reusable is not None:
            return reusable
        registered = self._resolve_provider(target.provider_name)
        provider = registered.provider
        messages = compose_provider_completion_verifier_messages(
            await verifier.build_messages(request), request
        )
        prepared, request_sha256 = self._prepare_request(provider, target, messages)
        billing_identity = await resolve_request_billing_identity(
            provider,
            prepared.model_copy(deep=True),
            provider_name=registered.name,
        )
        pricing_provider_name = provider.billing_provider_name or registered.name
        attempts = target.retry_policy.max_attempts
        same_execution = tuple(
            dispatch
            for dispatch in prior
            if dispatch.request.claim_id == authority.claim_id
            and dispatch.request.claim_attempt_number == authority.claim_attempt_number
        )
        for dispatch in same_execution:
            if dispatch.settlement is None:
                # Only this owner can hold this live claim, and the coordinator
                # rejects overlapping executions, so this attempt is no longer
                # observed by anyone: record that its outcome is unknown.
                await self._settle(
                    store,
                    CompletionVerifierDispatchSettlementRequest(
                        dispatch_id=dispatch.dispatch_id,
                        outcome=CompletionVerifierDispatchOutcome.OUTCOME_UNKNOWN,
                        usage_status=CompletionVerifierUsageStatus.MISSING,
                        latency_ms=0,
                        failure=CompletionVerifierDispatchFailure(code="verifier.settlement_lost"),
                    ),
                    cancellation=None,
                )
        first_attempt = len(same_execution) + 1
        if first_attempt > attempts:
            raise ProviderCompletionVerifierDispatchError(
                "The provider verifier execution already used its provider retries."
            )
        for provider_attempt in range(first_attempt, attempts + 1):
            dispatch = await self._record_intent(
                store,
                CompletionVerifierDispatchRequest(
                    dispatch_id=completion_verifier_dispatch_id(
                        proposal_id=authority.proposal_id,
                        claim_id=authority.claim_id,
                        claim_attempt_number=authority.claim_attempt_number,
                        provider_attempt=provider_attempt,
                    ),
                    proposal_id=authority.proposal_id,
                    claim_id=authority.claim_id,
                    worker_id=authority.worker_id,
                    execution_owner_id=authority.execution_owner_id,
                    claim_attempt_number=authority.claim_attempt_number,
                    verifier=authority.verifier,
                    verifier_profile_fingerprint=authority.verifier_profile_fingerprint,
                    provider_attempt=provider_attempt,
                    provider_name=registered.name,
                    pricing_provider_name=pricing_provider_name,
                    model=target.model,
                    request_sha256=request_sha256,
                    max_input_tokens=target.max_input_tokens,
                    max_output_tokens=target.max_output_tokens,
                    timeout_seconds=target.attempt_timeout_seconds,
                    budget=target.budget,
                ),
            )
            if dispatch.settlement is not None:
                # Exact replay of an attempt that already settled: never enter
                # the provider again for the same durable attempt identity.
                raise ProviderCompletionVerifierDispatchError(
                    "The provider verifier attempt already settled without a decision."
                )
            attempt = asyncio.create_task(
                self._run_attempt(
                    registered=registered,
                    prepared=prepared,
                    target=target,
                    request=request,
                    dispatch=dispatch,
                    billing_identity=billing_identity,
                    pricing_provider_name=pricing_provider_name,
                    provider_attempt=provider_attempt,
                ),
                name="cayu-provider-verifier-attempt",
            )
            outcome, cancellation = await self._await_attempt(attempt, dispatch)
            await self._settle(store, outcome.settlement, cancellation=cancellation)
            if isinstance(outcome.failure, asyncio.CancelledError):
                raise ProviderCompletionVerifierDispatchError(
                    "The provider verifier attempt was cancelled without caller cancellation."
                ) from None
            if outcome.failure is not None:
                raise outcome.failure
            if outcome.decision is not None:
                return outcome.decision
            retry = outcome.retry
            if retry is None or not retry.retry or provider_attempt == attempts:
                raise ProviderCompletionVerifierDispatchError(
                    "The provider verifier attempt failed without a decision."
                ) from None
            await asyncio.sleep(retry.delay_seconds)
        raise AssertionError("Provider verifier retry policy ended without an outcome.")

    def _prepare_request(
        self,
        provider,
        target: ProviderCompletionVerifierTarget,
        messages,
    ) -> tuple[ModelRequest, str]:
        copied = ModelRequest(model=target.model, messages=messages)
        canonical_bounded_durable_json_bytes(
            copied.model_dump(mode="json"),
            "provider_verifier_request",
            max_bytes=target.max_request_bytes,
            max_nodes=target.max_request_bytes,
        )
        prepared = provider.prepare_auxiliary_request(
            ModelRequest(model=copied.model, messages=copied.messages),
            max_output_tokens=target.max_output_tokens,
        )
        if type(prepared) is not ModelRequest:
            raise TypeError("Provider auxiliary preparation must return a ModelRequest.")
        prepared = _copy_auxiliary_request(prepared)
        if (
            prepared.model != copied.model
            or prepared.messages != copied.messages
            or prepared.tools
            or prepared.hosted_tools
        ):
            raise ValueError("Provider auxiliary preparation changed the verifier request.")
        encoded = canonical_bounded_durable_json_bytes(
            prepared.model_dump(mode="json"),
            "provider_verifier_request",
            max_bytes=target.max_request_bytes,
            max_nodes=target.max_request_bytes,
        )
        return prepared, sha256(encoded).hexdigest()

    async def _list_dispatches(
        self, store: TaskStore, proposal_id: str
    ) -> tuple[CompletionVerifierDispatch, ...]:
        outcome = await capture_task_store_operation(
            lambda: store.list_completion_verifier_dispatches(proposal_id),
            operation_name="Completion verifier dispatch lookup",
            redactor=self._secret_redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)
        result = outcome.result
        if type(result) is not tuple or any(
            type(item) is not CompletionVerifierDispatch or item.proposal_id != proposal_id
            for item in result
        ):
            raise ProviderCompletionVerifierDispatchError(
                "Task store returned invalid provider verifier dispatches."
            )
        return result

    async def _record_intent(
        self, store: TaskStore, request: CompletionVerifierDispatchRequest
    ) -> CompletionVerifierDispatch:
        outcome = await capture_task_store_operation(
            lambda: store.record_completion_verifier_dispatch(request),
            operation_name="Completion verifier dispatch intent",
            redactor=self._secret_redactor,
            mutation_store=store,
            mutation_method_name="record_completion_verifier_dispatch",
        )
        failure = outcome.failure
        if failure is not None:
            if type(failure) is CompletionVerifierDispatchBudgetExhausted:
                raise ProviderCompletionVerifierBudgetExhausted(str(failure)) from None
            raise_task_store_operation_failure(failure)
        dispatch = outcome.result
        if (
            type(dispatch) is not CompletionVerifierDispatch
            or dispatch.dispatch_id != request.dispatch_id
            or dispatch.request != request
        ):
            raise ProviderCompletionVerifierDispatchError(
                "Task store returned a dispatch other than the exact intent."
            )
        return dispatch

    async def _await_attempt(
        self,
        attempt: asyncio.Task[_AttemptOutcome],
        dispatch: CompletionVerifierDispatch,
    ) -> tuple[_AttemptOutcome, asyncio.CancelledError | None]:
        """Wait for the provider attempt; caller cancellation cancels only the attempt.

        The attempt runs in its own task so the cancellation delivered here keeps
        the coordinator's identity, while the provider stream boundary is free
        to sanitize the cancellation it observes. The attempt always reports an
        outcome, so its accounting can still be settled.
        """

        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                return await asyncio.shield(attempt), cancellation
            except asyncio.CancelledError as delivered:
                if cancellation is None:
                    cancellation = delivered
                if attempt.done():
                    break
                attempt.cancel()
        try:
            return attempt.result(), cancellation
        except asyncio.CancelledError:
            # Cancelled before the attempt started: the provider was never entered.
            return (
                _AttemptOutcome(
                    settlement=CompletionVerifierDispatchSettlementRequest(
                        dispatch_id=dispatch.dispatch_id,
                        outcome=CompletionVerifierDispatchOutcome.CANCELLED,
                        usage_status=CompletionVerifierUsageStatus.MISSING,
                        latency_ms=0,
                        failure=CompletionVerifierDispatchFailure(code="verifier.cancelled"),
                    ),
                    failure=cancellation,
                ),
                cancellation,
            )

    async def _settle(
        self,
        store: TaskStore,
        request: CompletionVerifierDispatchSettlementRequest,
        *,
        cancellation: asyncio.CancelledError | None,
    ) -> None:
        """Write accounting in its own task so cancellation cannot skip it.

        Caller cancellation stays authoritative once accounting is durable.
        """

        async def settle() -> CompletionVerifierDispatch | None:
            outcome = await capture_task_store_operation(
                lambda: store.settle_completion_verifier_dispatch(request),
                operation_name="Completion verifier dispatch settlement",
                redactor=self._secret_redactor,
                mutation_store=store,
                mutation_method_name="settle_completion_verifier_dispatch",
            )
            if outcome.failure is not None:
                raise_task_store_operation_failure(outcome.failure)
            return outcome.result

        task = asyncio.create_task(settle(), name="cayu-provider-verifier-settlement")
        shielded = await await_shielded_task_outcome(task)
        settlement_failure = shielded.error
        if settlement_failure is None:
            recorded = shielded.result
            if (
                type(recorded) is not CompletionVerifierDispatch
                or recorded.dispatch_id != request.dispatch_id
                or recorded.settlement is None
            ):
                settlement_failure = ProviderCompletionVerifierDispatchError(
                    "Task store returned an invalid provider verifier settlement."
                )
        cancellation = cancellation or shielded.cancellation
        if cancellation is not None:
            if settlement_failure is not None:
                raise cancellation from settlement_failure
            raise cancellation
        if settlement_failure is not None:
            raise settlement_failure

    async def _run_attempt(
        self,
        *,
        registered: RegisteredProvider,
        prepared: ModelRequest,
        target: ProviderCompletionVerifierTarget,
        request: CompletionVerifierRequest,
        dispatch: CompletionVerifierDispatch,
        billing_identity: BillingIdentity | None,
        pricing_provider_name: str,
        provider_attempt: int,
    ) -> _AttemptOutcome:
        provider = registered.provider
        redactor = self._secret_redactor
        admission = ProviderStreamDeadlineAdmission(provider.stream_deadlines)
        cleanup = reserve_provider_stream_cleanup(admission.max_concurrent_streams)
        transferred = False
        collector = AuxiliaryResponseCollector(max_bytes=target.max_response_bytes)
        metrics: UsageMetrics | None = None
        usage_status = CompletionVerifierUsageStatus.MISSING
        provider_failure: ModelProviderError | None = None
        failure: BaseException | None = None
        response: ModelResponse | None = None
        terminal_seen = False
        deadline_expired = False
        started = time.monotonic()

        async def refresh() -> None:
            return None

        try:
            async with execution_deadline_scope(
                ExecutionDeadline.after(
                    target.attempt_timeout_seconds,
                    source="completion_verifier",
                    scope="model",
                )
            ):
                task = asyncio.current_task()
                events = _owned_model_provider_events(
                    lambda: _admitted_model_provider_events(
                        provider, prepared, admission, refresh, error_redactor=redactor
                    ),
                    cancellation_baseline=0 if task is None else task.cancelling(),
                    max_concurrent_streams=admission.max_concurrent_streams,
                    cleanup_ownership=cleanup,
                )
                transferred = True
                async with aclosing(events):
                    async for event in events:
                        if (
                            type(event) is ModelStreamEvent
                            and event.type
                            in {ModelStreamEventType.COMPLETED, ModelStreamEventType.ERROR}
                            and not terminal_seen
                        ):
                            terminal_seen = True
                            raw_usage = (
                                event.payload.get("usage") if type(event.payload) is dict else None
                            )
                            metrics = normalize_usage_metrics(
                                provider_name=pricing_provider_name,
                                model=target.model,
                                requested_model=target.model,
                                raw_usage=raw_usage,
                                usage_dialect=registered.usage_dialect,
                                billing_identity=billing_identity,
                            )
                            usage_status = (
                                CompletionVerifierUsageStatus.OBSERVED
                                if metrics is not None
                                else CompletionVerifierUsageStatus.MISSING
                                if raw_usage is None
                                else CompletionVerifierUsageStatus.MALFORMED
                            )
                        if (
                            type(event) is ModelStreamEvent
                            and event.type is ModelStreamEventType.ERROR
                        ):
                            accepted = collector.accept_error(event)
                            provider_failure = model_provider_error_from_payload(
                                redactor.redact_json_values(accepted.payload),
                                fallback_provider=registered.name,
                            ) or ModelProviderError(
                                "Provider verifier model reported a failure.",
                                provider=registered.name,
                            )
                            if isinstance(provider_failure, ModelStreamDeadlineError):
                                raise provider_failure
                        else:
                            collector.add(event)
                if provider_failure is None:
                    response = collector.finish()
        except BaseException as error:
            failure = error
            deadline_expired = expired_execution_deadline() is not None or isinstance(
                error, (TimeoutError, ModelStreamDeadlineError)
            )
        finally:
            admission.close()
            if not transferred:
                release_provider_stream_cleanup(cleanup)
        latency_ms = max(0, int((time.monotonic() - started) * 1000))
        usage = metrics if usage_status is CompletionVerifierUsageStatus.OBSERVED else None

        def unsuccessful(
            outcome: CompletionVerifierDispatchOutcome,
            code: str,
            *,
            status_code: int | None = None,
            retryable: bool | None = None,
        ) -> CompletionVerifierDispatchSettlementRequest:
            return CompletionVerifierDispatchSettlementRequest(
                dispatch_id=dispatch.dispatch_id,
                outcome=outcome,
                usage_status=usage_status,
                usage=usage,
                latency_ms=latency_ms,
                failure=CompletionVerifierDispatchFailure(
                    code=code, status_code=status_code, retryable=retryable
                ),
            )

        if failure is not None:
            if isinstance(failure, asyncio.CancelledError):
                settlement = unsuccessful(
                    CompletionVerifierDispatchOutcome.CANCELLED
                    if not terminal_seen
                    else CompletionVerifierDispatchOutcome.OUTCOME_UNKNOWN,
                    "verifier.cancelled",
                )
                return _AttemptOutcome(settlement=settlement, failure=failure)
            if deadline_expired:
                settlement = unsuccessful(
                    CompletionVerifierDispatchOutcome.TIMED_OUT, "provider.timed_out"
                )
                return _AttemptOutcome(
                    settlement=settlement,
                    failure=ProviderCompletionVerifierDispatchError(
                        "The provider verifier attempt exceeded its timeout."
                    ),
                )
            if isinstance(failure, ModelProviderError) and not terminal_seen:
                # A provider failure raised before any terminal event: the
                # provider rejected or failed the call without stream evidence.
                provider_failure = failure
            else:
                settlement = unsuccessful(
                    CompletionVerifierDispatchOutcome.OUTCOME_UNKNOWN, "provider.stream_failed"
                )
                return _AttemptOutcome(
                    settlement=settlement,
                    failure=ProviderCompletionVerifierDispatchError(
                        "The provider verifier attempt ended without a usable outcome."
                    ),
                )
        if provider_failure is not None:
            decision = retry_decision(
                policy=target.retry_policy,
                attempt=provider_attempt,
                error=str(provider_failure),
                status_code=provider_failure.status_code,
                retryable=provider_failure.retryable,
                retry_after_s=provider_failure.retry_after_s,
                unknown_provider_error=(
                    provider_failure.status_code is None and provider_failure.retryable is None
                ),
            )
            settlement = unsuccessful(
                CompletionVerifierDispatchOutcome.FAILED,
                _failure_code(provider_failure),
                status_code=provider_failure.status_code,
                retryable=provider_failure.retryable,
            )
            return _AttemptOutcome(
                settlement=settlement, provider_failure=provider_failure, retry=decision
            )
        if response is None:  # pragma: no cover - success requires a finished response
            raise AssertionError("Provider verifier attempt has no response.")
        text = response.text
        response_sha256 = sha256(text.encode("utf-8")).hexdigest()
        try:
            decoded = decode_provider_completion_verifier_decision(text, request)
        except ProviderCompletionVerifierDecodingError as decoding_failure:
            settlement = CompletionVerifierDispatchSettlementRequest(
                dispatch_id=dispatch.dispatch_id,
                outcome=CompletionVerifierDispatchOutcome.COMPLETED,
                usage_status=usage_status,
                usage=usage,
                latency_ms=latency_ms,
                response_sha256=response_sha256,
                decode_status=CompletionVerifierDecodeStatus.INVALID,
                failure=CompletionVerifierDispatchFailure(code="decode.invalid_decision"),
            )
            return _AttemptOutcome(settlement=settlement, failure=decoding_failure)
        settlement = CompletionVerifierDispatchSettlementRequest(
            dispatch_id=dispatch.dispatch_id,
            outcome=CompletionVerifierDispatchOutcome.COMPLETED,
            usage_status=usage_status,
            usage=usage,
            latency_ms=latency_ms,
            response_sha256=response_sha256,
            decode_status=CompletionVerifierDecodeStatus.DECODED,
            decision=decoded,
        )
        return _AttemptOutcome(settlement=settlement, decision=decoded)


__all__ = [
    "ProviderCompletionVerifierRuntime",
    "ProviderVerifierExecutionAuthority",
    "reusable_provider_decision",
]
