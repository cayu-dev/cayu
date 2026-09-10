"""Bounded external-effect evidence; receipt construction does not authenticate it."""

from __future__ import annotations

from datetime import UTC
from hashlib import sha256
from typing import Any, Literal, Protocol

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from cayu._validation import (
    canonical_bounded_durable_json_bytes,
    copy_bounded_durable_json_value,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.core.execution_identity import (
    ExecutionProfileBehaviorIdentity,
    copy_execution_profile_behavior_identity,
)
from cayu.runtime.config import MAX_STEPS
from cayu.runtime.human_review import HumanReviewContext, HumanReviewReference
from cayu.runtime.user_input import UserInputResponse, copy_user_input_response

TOOL_EFFECT_RECEIPT_MAX_BYTES = 96 * 1024
TOOL_EFFECT_RESULT_MAX_BYTES = 64 * 1024
_RECEIPT_MAX_NODES = 8192


class ToolEffectConflict(RuntimeError):
    """Exact effect identity or state conflicts with the requested operation."""


def _bounded_text(value: str, field: str, *, maximum: int, identifier: bool = False) -> str:
    value = (
        require_durable_clean_nonblank(value, field)
        if identifier
        else require_durable_text(value, field)
    )
    if len(value) > maximum or len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{field} exceeds its byte limit.")
    return value


def _copy_string_map(value: object, field: str, *, maximum_items: int) -> dict[str, str]:
    if type(value) is not dict or len(value) > maximum_items:
        raise ValueError(f"{field} must be a bounded object.")
    copied: dict[str, str] = {}
    for key, item in value.items():
        if type(key) is not str or type(item) is not str:
            raise ValueError(f"{field} must contain string identities and values.")
        copied[_bounded_text(key, field, maximum=256, identifier=True)] = _bounded_text(
            item, field, maximum=1024, identifier=True
        )
    return copied


class ToolEffectReceipt(BaseModel):
    """Untrusted, versioned evidence submitted to an application receipt validator.

    Neither this type nor its digest proves that an external effect happened.
    The registered application validator must authenticate and normalize the
    receipt for the exact durable call before runtime settlement. The runtime
    additionally validates its registered result schema and redacts publication.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    receipt_id: StrictStr
    receipt_schema: StrictStr
    receipt_schema_version: StrictInt = Field(ge=1, le=2**31 - 1)
    tool_call_id: StrictStr
    idempotency_key: StrictStr
    tool_name: StrictStr
    external_system: StrictStr | None = None
    outcome: Literal["completed", "failed"]
    message: StrictStr
    structured: dict[str, Any] | None = None
    resource_versions: dict[str, str] = Field(default_factory=dict)
    observed_at: AwareDatetime
    source: Literal["adapter", "reconciler", "operator"]
    integrity: dict[str, str] = Field(default_factory=dict)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Unsupported tool-effect receipt version.")
        return value

    @field_validator("receipt_id", "receipt_schema", "tool_call_id", "idempotency_key", "tool_name")
    @classmethod
    def validate_identity(cls, value: str, info) -> str:
        return _bounded_text(value, info.field_name, maximum=256, identifier=True)

    @field_validator("external_system")
    @classmethod
    def validate_system(cls, value: str | None) -> str | None:
        return (
            None
            if value is None
            else _bounded_text(value, "external_system", maximum=256, identifier=True)
        )

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        return _bounded_text(value, "message", maximum=4096)

    @field_validator("structured", mode="before")
    @classmethod
    def copy_structured(cls, value: object) -> dict[str, Any] | None:
        if value is None:
            return None
        if type(value) is not dict:
            raise ValueError("structured must be a bounded JSON object.")
        return copy_bounded_durable_json_value(
            value,
            "tool_effect_receipt.structured",
            max_bytes=TOOL_EFFECT_RESULT_MAX_BYTES,
            max_nodes=4096,
            max_nesting=32,
        )

    @field_validator("resource_versions", "integrity", mode="before")
    @classmethod
    def copy_evidence(cls, value: object, info) -> dict[str, str]:
        return _copy_string_map(
            value,
            info.field_name,
            maximum_items=32 if info.field_name == "resource_versions" else 16,
        )

    @field_validator("observed_at")
    @classmethod
    def normalize_observation(cls, value: AwareDatetime):
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_envelope_bound(self) -> ToolEffectReceipt:
        _receipt_bytes(self)
        return self


def copy_tool_effect_receipt(receipt: ToolEffectReceipt) -> ToolEffectReceipt:
    """Revalidate declared fields without serializing a caller-owned model."""

    if type(receipt) is not ToolEffectReceipt:
        raise TypeError("receipt must be an exact ToolEffectReceipt.")
    # Frozen models can still contain caller-mutated nested containers, and
    # model_construct/model_copy can bypass validation. Never model_dump here.
    return ToolEffectReceipt(
        **{name: getattr(receipt, name) for name in ToolEffectReceipt.model_fields}
    )


def tool_effect_receipt_digest(receipt: ToolEffectReceipt) -> str:
    """Content identity only, not an authenticity check or replay authorization."""

    return sha256(_receipt_bytes(copy_tool_effect_receipt(receipt))).hexdigest()


def _receipt_bytes(receipt: ToolEffectReceipt) -> bytes:
    fields = {name: getattr(receipt, name) for name in ToolEffectReceipt.model_fields}
    fields["observed_at"] = receipt.observed_at.isoformat()
    return canonical_bounded_durable_json_bytes(
        fields,
        "tool_effect_receipt",
        max_bytes=TOOL_EFFECT_RECEIPT_MAX_BYTES,
        max_nodes=_RECEIPT_MAX_NODES,
        max_nesting=34,
    )


class ToolEffectReconciliationContext(BaseModel):
    """Detached exact-call identity supplied by runtime, without executable arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: StrictStr
    session_instance_id: StrictStr
    tool_round_id: StrictStr
    tool_call_id: StrictStr
    tool_name: StrictStr
    idempotency_key: StrictStr
    intent_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    arguments_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    record_revision: StrictInt = Field(ge=0)

    @field_validator("*", mode="after")
    @classmethod
    def validate_identity(cls, value, info):
        from cayu.runtime.sessions import MAX_SESSION_ID_BYTES

        if isinstance(value, str):
            return _bounded_text(
                value,
                info.field_name,
                maximum=MAX_SESSION_ID_BYTES if info.field_name == "session_id" else 256,
                identifier=True,
            )
        return value


class ToolEffectReconciliationResult(BaseModel):
    """Application observation, subject to runtime binding/schema validation.

    Retryability refers only to a new explicit reconciliation lookup, never to
    another tool invocation. Partial observations cannot authorize replay.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    outcome: Literal["completed", "failed", "not_found", "conflict", "unsupported"]
    receipt: ToolEffectReceipt | None = None
    observation: Literal["sent", "not_sent", "outcome_unknown", "partial"]
    retryable: StrictBool = False
    resource_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("receipt", mode="before")
    @classmethod
    def copy_receipt(cls, value):
        return None if value is None else copy_tool_effect_receipt(value)

    @field_validator("resource_versions", mode="before")
    @classmethod
    def copy_resources(cls, value):
        return _copy_string_map(value, "resource_versions", maximum_items=32)

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolEffectReconciliationResult:
        terminal = self.outcome in {"completed", "failed"}
        if terminal != (self.receipt is not None):
            raise ValueError("Terminal reconciliation requires exactly one validated receipt.")
        if self.receipt is not None and self.receipt.outcome != self.outcome:
            raise ValueError("Reconciliation outcome conflicts with its receipt.")
        if self.outcome == "completed" and self.observation != "sent":
            raise ValueError("Completed reconciliation requires positive effect evidence.")
        if terminal and self.retryable:
            raise ValueError("Terminal reconciliation cannot request another lookup.")
        if terminal and self.observation == "outcome_unknown":
            raise ValueError("Unknown effect evidence cannot become terminal reconciliation.")
        if self.receipt is not None:
            versions = self.receipt.resource_versions
            if any(
                key in versions and versions[key] != value
                for key, value in self.resource_versions.items()
            ):
                raise ValueError("Reconciliation contains conflicting resource versions.")
            if len(set(versions) | set(self.resource_versions)) > 32:
                raise ValueError("Combined reconciliation resources exceed the bound.")
        return self


class ToolEffectReconciler(Protocol):
    """Application-owned authenticity/lookup boundary; must never dispatch the tool."""

    async def reconcile(
        self,
        *,
        context: ToolEffectReconciliationContext,
        receipt: ToolEffectReceipt | None,
    ) -> ToolEffectReconciliationResult: ...


class ToolEffectReconcilerSpec(BaseModel):
    """Registration-time, profile-bound contract for one external receipt schema."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    execution_profile_identity: ExecutionProfileBehaviorIdentity
    effect: Literal["external"] = "external"
    receipt_schema: StrictStr
    receipt_schema_version: StrictInt = Field(ge=1, le=2**31 - 1)
    required: StrictBool = False
    supports_lookup: StrictBool = False
    timeout_seconds: StrictFloat = Field(default=30.0, gt=0, le=300)
    result_schema: dict[str, Any]
    integrity_fields: tuple[StrictStr, ...] = ()
    resource_version_fields: tuple[StrictStr, ...] = ()

    @field_validator("execution_profile_identity", mode="before")
    @classmethod
    def copy_identity(cls, value):
        if type(value) is dict:
            value = ExecutionProfileBehaviorIdentity.model_validate(value)
        if value is None:
            raise ValueError("Reconciler requires an application behavior identity.")
        return copy_execution_profile_behavior_identity(value)

    @field_validator("receipt_schema")
    @classmethod
    def validate_schema_id(cls, value: str) -> str:
        return _bounded_text(value, "receipt_schema", maximum=256, identifier=True)

    @field_validator("result_schema", mode="before")
    @classmethod
    def copy_result_schema(cls, value):
        from jsonschema import Draft202012Validator

        copied = ToolEffectReceipt.copy_structured(value)
        if (
            copied is None
            or copied.get("type") != "object"
            or copied.get("additionalProperties") is not False
        ):
            raise ValueError("Receipt result schema must declare a closed object.")
        pending = [copied]
        while pending:
            item = pending.pop()
            if type(item) is dict:
                for keyword in ("$ref", "$dynamicRef"):
                    reference = item.get(keyword)
                    if reference is not None and (
                        type(reference) is not str or not reference.startswith("#")
                    ):
                        raise ValueError(
                            "Receipt result schema cannot resolve external references."
                        )
                pending.extend(item.values())
            elif type(item) is list:
                pending.extend(item)
        try:
            Draft202012Validator.check_schema(copied)
        except Exception:
            raise ValueError("Receipt result schema is invalid.") from None
        return copied

    @field_validator("integrity_fields", "resource_version_fields", mode="before")
    @classmethod
    def validate_evidence_fields(cls, value, info):
        maximum = 16 if info.field_name == "integrity_fields" else 32
        if type(value) not in {tuple, list} or len(value) > maximum:
            raise ValueError("Reconciler evidence allow-list must be bounded.")
        fields = tuple(
            _bounded_text(item, info.field_name, maximum=256, identifier=True) for item in value
        )
        if len(fields) != len(set(fields)):
            raise ValueError("Reconciler evidence allow-list contains duplicates.")
        return tuple(sorted(fields))


class ToolEffectReconciliationRegistration:
    """Pair application logic with its executable, versioned declaration."""

    def __init__(self, *, reconciler: ToolEffectReconciler, spec: ToolEffectReconcilerSpec) -> None:
        self.reconciler = reconciler
        self.spec = spec


class ToolEffectReconciliationTarget(BaseModel):
    """Read-only call identity/version snapshot, not permission to reconcile.

    Public inspection aliases private key/incarnation fields. Copy these fields
    into a reconciliation request with a receipt or lookup instruction. The
    runtime still validates current state, authority and external evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    session_id: StrictStr
    session_instance_id: StrictStr
    tool_round_id: StrictStr
    tool_call_id: StrictStr
    tool_name: StrictStr
    idempotency_key: StrictStr
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_revision: StrictInt = Field(ge=0)

    @field_validator("*", mode="after")
    @classmethod
    def validate_identity(cls, value, info):
        return ToolEffectReconciliationContext.validate_identity(value, info)


class ToolEffectReconciliationRequest(BaseModel):
    """An explicit receipt or lookup for one existing uncertain external call.

    Version fields are required optimistic-concurrency evidence, not optional
    hints. Continuation retains the existing invocation's limits unless a
    smaller step bound is supplied. Worker/handoff identities remain subject
    to the runtime's existing task-continuation admission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    review_reference: HumanReviewReference | None = Field(default=None, repr=False)
    session_id: StrictStr
    session_instance_id: StrictStr
    tool_round_id: StrictStr
    tool_call_id: StrictStr
    tool_name: StrictStr
    idempotency_key: StrictStr
    expected_run_epoch: StrictInt = Field(ge=0)
    expected_revision: StrictInt = Field(ge=0)
    receipt: ToolEffectReceipt | None = None
    lookup: StrictBool = False
    max_steps: StrictInt | None = Field(default=None, ge=1, le=MAX_STEPS)
    task_worker_id: StrictStr | None = None
    task_handoff_id: StrictStr | None = None

    user_input_response: UserInputResponse | None = Field(default=None, repr=False)

    @field_validator("user_input_response", mode="before")
    @classmethod
    def detach_user_input_response(cls, value):
        if value is None:
            return None
        if type(value) is dict:
            value = UserInputResponse.model_validate(value)
        if type(value) is not UserInputResponse:
            raise TypeError("Effect reconciliation requires an exact user-input response.")
        value = value.model_copy(
            update={"review_reference": cls.detach_review_reference(value.review_reference)}
        )
        return copy_user_input_response(value)

    @field_validator("review_reference", mode="before")
    @classmethod
    def detach_review_reference(cls, value):
        if value is None:
            return None
        if type(value) is dict:
            value = HumanReviewReference.model_validate(value)
        if type(value) is not HumanReviewReference or type(value.context) is not HumanReviewContext:
            raise TypeError("Effect reconciliation requires an exact human-review reference.")
        return HumanReviewReference(
            context=HumanReviewContext(
                **{name: getattr(value.context, name) for name in HumanReviewContext.model_fields}
            ),
            policy_version=value.policy_version,
            content_tag=value.content_tag,
        )

    @field_validator(
        "session_id",
        "session_instance_id",
        "tool_round_id",
        "tool_call_id",
        "tool_name",
        "idempotency_key",
        "task_worker_id",
        "task_handoff_id",
    )
    @classmethod
    def validate_identity(cls, value, info):
        return (
            None
            if value is None
            else ToolEffectReconciliationContext.validate_identity(value, info)
        )

    @field_validator("receipt", mode="before")
    @classmethod
    def detach_receipt(cls, value):
        if value is None:
            return None
        if type(value) is dict:
            return ToolEffectReceipt.model_validate(value)
        return copy_tool_effect_receipt(value)

    @model_validator(mode="after")
    def validate_request(self) -> ToolEffectReconciliationRequest:
        if self.lookup == (self.receipt is not None):
            raise ValueError("Reconciliation requires exactly one receipt or lookup.")
        if self.task_handoff_id is not None and self.task_worker_id is None:
            raise ValueError("Task handoff identity requires its worker identity.")
        if self.receipt is not None and any(
            getattr(self.receipt, name) != getattr(self, name)
            for name in ("tool_call_id", "tool_name", "idempotency_key")
        ):
            raise ValueError("Supplied receipt conflicts with the requested call.")
        return self
