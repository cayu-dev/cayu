"""Frozen executable registrations for application-owned effect reconciliation."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from cayu._task_wait import unexpected_child_cancellation_error
from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.core.tools import ToolEffect
from cayu.runtime._tool_effect_state import ToolEffectConflict, ToolEffectRecord, _copy_model
from cayu.runtime.tool_effects import (
    ToolEffectReceipt,
    ToolEffectReconcilerSpec,
    ToolEffectReconciliationContext,
    ToolEffectReconciliationRegistration,
    ToolEffectReconciliationRequest,
    ToolEffectReconciliationResult,
)
from cayu.tools._operation_boundary import (
    BoundedInvocationOperationRegistry,
    await_invocation_operation,
)
from cayu.vaults import SecretRedactor


@dataclass(frozen=True, slots=True)
class RegisteredToolEffectReconciler:
    descriptor_json: bytes
    fingerprint: str
    operation: Callable[..., Awaitable[ToolEffectReconciliationResult]]

    def material(self) -> dict[str, Any]:
        """Return detached declarative profile material, never an executable object."""
        return ToolEffectReconcilerSpec.model_validate_json(self.descriptor_json).model_dump(
            mode="json"
        )


def register_tool_effect_reconciler(
    registration: ToolEffectReconciliationRegistration,
    *,
    effect: ToolEffect,
    redactor: SecretRedactor,
) -> RegisteredToolEffectReconciler:
    if type(registration) is not ToolEffectReconciliationRegistration:
        raise TypeError("Effect reconciliation requires an exact registration.")
    if effect is not ToolEffect.EXTERNAL:
        raise ValueError("Receipt reconciliation applies only to external-effect tools.")
    original = registration.spec
    if type(original) is not ToolEffectReconcilerSpec:
        raise TypeError("Effect reconciliation requires an exact specification.")
    spec = ToolEffectReconcilerSpec(
        **{name: getattr(original, name) for name in ToolEffectReconcilerSpec.model_fields}
    )
    operation = getattr(registration.reconciler, "reconcile", None)
    if not inspect.iscoroutinefunction(operation):
        raise TypeError("Effect reconciliation requires an actual async reconcile operation.")
    try:
        inspect.signature(operation).bind(context=None, receipt=None)
    except (TypeError, ValueError):
        raise TypeError("Effect reconcile operation has an incompatible signature.") from None
    material = spec.model_dump(mode="json")
    if redactor.redact_json(material) != material:
        raise ValueError("Effect reconciliation specification must not contain secrets.")
    encoded = canonical_bounded_durable_json_bytes(
        material,
        "effect_reconciler",
        max_bytes=96 * 1024,
        max_nodes=8192,
        max_nesting=40,
    )
    return RegisteredToolEffectReconciler(
        descriptor_json=encoded,
        fingerprint=sha256(encoded).hexdigest(),
        operation=operation,
    )


def validate_reconciliation_result(
    result: ToolEffectReconciliationResult,
    *,
    registered: RegisteredToolEffectReconciler,
    context: ToolEffectReconciliationContext,
) -> ToolEffectReconciliationResult:
    """Validate an application callback's returned evidence against its frozen contract."""
    from jsonschema import Draft202012Validator
    from referencing import Registry

    if type(result) is not ToolEffectReconciliationResult:
        raise TypeError("Reconciler must return an exact ToolEffectReconciliationResult.")
    copied = ToolEffectReconciliationResult(
        **{name: getattr(result, name) for name in ToolEffectReconciliationResult.model_fields}
    )
    receipt: ToolEffectReceipt | None = copied.receipt
    spec = registered.material()
    resource_fields = set(copied.resource_versions)
    if receipt is not None:
        resource_fields.update(receipt.resource_versions)
    if resource_fields - set(spec["resource_version_fields"]):
        raise ValueError("Reconciled resource evidence violates its registered allow-list.")
    if receipt is None:
        return copied
    if (
        receipt.tool_name != context.tool_name
        or receipt.tool_call_id != context.tool_call_id
        or receipt.idempotency_key != context.idempotency_key
        or receipt.receipt_schema != spec["receipt_schema"]
        or receipt.receipt_schema_version != spec["receipt_schema_version"]
        or set(receipt.integrity) - set(spec["integrity_fields"])
    ):
        raise ValueError("Reconciled receipt conflicts with the exact call or registered schema.")
    if receipt.structured is not None:
        try:
            valid = Draft202012Validator(
                spec["result_schema"],
                registry=Registry(),
            ).is_valid(receipt.structured)
        except Exception:
            raise ValueError("Reconciled receipt result could not be validated.") from None
        if not valid:
            raise ValueError("Reconciled receipt result violates its registered schema.")
    # Both accepted representations describe the same effect. Bind their union
    # into the durable receipt so terminal settlement cannot drop partial evidence.
    return ToolEffectReconciliationResult(
        outcome=copied.outcome,
        observation=copied.observation,
        retryable=copied.retryable,
        resource_versions=copied.resource_versions,
        receipt=receipt.model_copy(
            update={
                "resource_versions": {**receipt.resource_versions, **copied.resource_versions},
            }
        ),
    )


@dataclass(frozen=True, slots=True)
class AcceptedToolEffectReconciliation:
    """Internal handoff minted only after the owned application callback returns."""

    context: ToolEffectReconciliationContext
    request_digest: str
    result: ToolEffectReconciliationResult


def project_accepted_reconciliation(
    accepted: AcceptedToolEffectReconciliation,
    *,
    registered: RegisteredToolEffectReconciler | None,
    redactor: SecretRedactor,
) -> ToolEffectReconciliationResult:
    """Redact accepted data, then revalidate the exact contract before publication."""
    original = accepted.result
    copied = ToolEffectReconciliationResult(
        **{name: getattr(original, name) for name in ToolEffectReconciliationResult.model_fields}
    )
    projected = ToolEffectReconciliationResult(
        outcome=copied.outcome,
        observation=copied.observation,
        retryable=copied.retryable,
        resource_versions=redactor.redact_json(copied.resource_versions),
        receipt=None
        if copied.receipt is None
        else ToolEffectReceipt.model_validate(
            # Field names and validated enums are schema controls. Identities,
            # messages and nested application data still pass through redaction.
            {
                name: value if name in {"outcome", "source"} else redactor.redact_json(value)
                for name, value in copied.receipt.model_dump(mode="json").items()
            }
        ),
    )
    if registered is None:
        if projected != _unsupported():
            raise ToolEffectConflict("Unregistered reconciliation has unexpected evidence.")
        return projected
    return validate_reconciliation_result(
        projected, registered=registered, context=accepted.context
    )


class ToolEffectReconciliationTimeout(TimeoutError):
    """The lookup wait expired; the effect stays unresolved and never retryable."""


@dataclass(frozen=True, slots=True)
class _PreparedToolEffectReconciliation:
    request: ToolEffectReconciliationRequest
    context: ToolEffectReconciliationContext
    request_digest: str


class ToolEffectReconciliationOwner:
    """Own a bounded number of read-only application lookups through real settlement."""

    def __init__(self, *, max_operations: int = 32) -> None:
        self._operations = BoundedInvocationOperationRegistry(max_operations=max_operations)

    @property
    def pending_operations(self) -> int:
        return len(self._operations)

    async def aclose(self, *, timeout_seconds: float = 1.0) -> bool:
        return await self._operations.aclose(timeout_s=timeout_seconds)

    @staticmethod
    def prepare(
        *,
        request: ToolEffectReconciliationRequest,
        record: ToolEffectRecord,
        run_epoch: int,
        registered: RegisteredToolEffectReconciler | None,
    ) -> _PreparedToolEffectReconciliation:
        """Validate exact unresolved authority without dispatch or reservation.

        Recovery uses this before extension setup; reconcile repeats it against
        current durable authority immediately before invoking application code.
        """
        if type(request) is not ToolEffectReconciliationRequest:
            raise TypeError("Reconciliation requires an exact request.")
        request = ToolEffectReconciliationRequest(
            **{
                name: getattr(request, name)
                for name in ToolEffectReconciliationRequest.model_fields
            }
        )
        record = _copy_model(record, ToolEffectRecord)
        if type(run_epoch) is not int or run_epoch < 0:
            raise ValueError("Reconciliation requires an exact current run epoch.")
        intent = record.intent
        response = request.user_input_response
        if (intent.pause_id is None) != (response is None) or (
            response is not None
            and (
                response.input_id != intent.pause_id
                or response.session_id != request.session_id
                or response.task_worker_id != request.task_worker_id
                or response.task_handoff_id != request.task_handoff_id
            )
        ):
            raise ToolEffectConflict("Reconciliation has conflicting user-input authority.")
        if (
            record.state != "outcome_unknown"
            or request.expected_revision != record.revision
            or request.expected_run_epoch != run_epoch
            or any(
                getattr(request, name) != getattr(intent, name)
                for name in (
                    "session_id",
                    "session_instance_id",
                    "tool_round_id",
                    "tool_call_id",
                    "tool_name",
                    "idempotency_key",
                )
            )
            or intent.reconciler_fingerprint
            != (None if registered is None else registered.fingerprint)
        ):
            raise ToolEffectConflict("Reconciliation does not match the current uncertain call.")
        context = ToolEffectReconciliationContext(
            **{
                name: getattr(intent, name)
                for name in (
                    "session_id",
                    "session_instance_id",
                    "tool_round_id",
                    "tool_call_id",
                    "tool_name",
                    "idempotency_key",
                    "arguments_digest",
                )
            },
            intent_digest=_reconciliation_digest(intent.model_dump(mode="json")),
            record_revision=record.revision,
        )
        request_digest = reconciliation_request_digest(request)
        if registered is not None and request.receipt is not None:
            spec = registered.material()
            if (
                request.receipt.receipt_schema != spec["receipt_schema"]
                or request.receipt.receipt_schema_version != spec["receipt_schema_version"]
            ):
                raise ToolEffectConflict("Receipt schema does not match the registered reconciler.")
        return _PreparedToolEffectReconciliation(request, context, request_digest)

    async def reconcile(
        self,
        *,
        request: ToolEffectReconciliationRequest,
        record: ToolEffectRecord,
        run_epoch: int,
        registered: RegisteredToolEffectReconciler | None,
    ) -> AcceptedToolEffectReconciliation:
        prepared = self.prepare(
            request=request, record=record, run_epoch=run_epoch, registered=registered
        )
        request, context, request_digest = (
            prepared.request,
            prepared.context,
            prepared.request_digest,
        )
        if registered is None:
            return AcceptedToolEffectReconciliation(context, request_digest, _unsupported())
        spec = registered.material()
        if request.lookup and not spec["supports_lookup"]:
            return AcceptedToolEffectReconciliation(context, request_digest, _unsupported())

        async def invoke() -> ToolEffectReconciliationResult:
            result = await registered.operation(
                context=ToolEffectReconciliationContext(
                    **{
                        name: getattr(context, name)
                        for name in ToolEffectReconciliationContext.model_fields
                    }
                ),
                receipt=None
                if request.receipt is None
                else ToolEffectReceipt(
                    **{
                        name: getattr(request.receipt, name)
                        for name in ToolEffectReceipt.model_fields
                    }
                ),
            )
            return validate_reconciliation_result(result, registered=registered, context=context)

        deadline = asyncio.timeout(spec["timeout_seconds"])
        try:
            async with deadline:
                outcome = await await_invocation_operation(
                    invoke,
                    request_child_cancellation=False,
                    abandon_on_caller_cancellation=True,
                    operation_registry=self._operations,
                )
                if outcome.cancellation is not None:
                    if outcome.error is not None:
                        raise outcome.cancellation from outcome.error
                    raise outcome.cancellation
                if outcome.error is not None:
                    if isinstance(outcome.error, asyncio.CancelledError):
                        raise unexpected_child_cancellation_error(
                            outcome.error,
                            operation="Effect reconciliation",
                        ) from outcome.error
                    raise outcome.error
        except TimeoutError as error:
            if not deadline.expired():
                raise
            raise ToolEffectReconciliationTimeout("Effect reconciliation wait expired.") from error
        if outcome.result is None:
            raise RuntimeError("Effect reconciliation produced no outcome.")
        return AcceptedToolEffectReconciliation(context, request_digest, outcome.result)


def _unsupported() -> ToolEffectReconciliationResult:
    return ToolEffectReconciliationResult(outcome="unsupported", observation="outcome_unknown")


def reconciliation_request_digest(request: ToolEffectReconciliationRequest) -> str:
    """Bind every decision-bearing field before any replay or callback dispatch."""
    if type(request) is not ToolEffectReconciliationRequest:
        raise TypeError("Reconciliation requires an exact request.")
    copied = ToolEffectReconciliationRequest(
        **{name: getattr(request, name) for name in ToolEffectReconciliationRequest.model_fields}
    )
    from cayu.runtime.user_input import user_input_resolution_request_digest

    document = copied.model_dump(mode="json", exclude={"user_input_response"})
    document["user_input_response"] = (
        None
        if copied.user_input_response is None
        else user_input_resolution_request_digest(copied.user_input_response)
    )
    return _reconciliation_digest(document)


def _reconciliation_digest(value: object) -> str:
    return sha256(
        canonical_bounded_durable_json_bytes(
            value,
            "effect_reconciliation",
            max_bytes=128 * 1024,
            max_nodes=16384,
            max_nesting=40,
        )
    ).hexdigest()
