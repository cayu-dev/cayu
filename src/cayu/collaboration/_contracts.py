"""Bounded values, not authorization tokens or a collaboration state machine."""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Generic, Literal, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)
from typing_extensions import TypeAliasType

from cayu._validation import (
    MAX_PORTABLE_JSON_INTEGER,
    canonical_bounded_durable_json_bytes,
    inspect_bounded_durable_json,
    require_durable_clean_nonblank,
)

MAX_ENVELOPE_BYTES = 64 * 1024
MAX_NODES = 8192
MAX_DEPTH = 64
MAX_ENTRIES = 64
MAX_ID_BYTES = 512
MAX_CODE_BYTES = 128


class CollaborationContractError(ValueError):
    """Content-free rejection of malformed collaboration contract material."""


class CollaborationConflict(ValueError):
    """An operation key was reused with different expected material."""


def _identifier(value: str) -> str:
    require_durable_clean_nonblank(value, "identity")
    if len(value.encode("utf-8")) > MAX_ID_BYTES:
        raise ValueError("Identity exceeds the byte limit.")
    return value


def _code(value: str) -> str:
    _identifier(value)
    if (
        not value.isascii()
        or len(value) > MAX_CODE_BYTES
        or any(not (char.isalnum() or char in "_.:-") for char in value)
    ):
        raise ValueError("Code must contain bounded ASCII identifier characters.")
    return value


Identifier = Annotated[StrictStr, AfterValidator(_identifier)]
Code = Annotated[StrictStr, AfterValidator(_code)]
Generation = Annotated[StrictInt, Field(ge=1, le=MAX_PORTABLE_JSON_INTEGER)]


def snapshot_input(value: object) -> Any:
    """Detach only known model fields and primitive containers, without serializers.

    This adapter bounds model traversal before the existing JSON walker validates
    text/numbers and aggregate bytes. It never invokes repr or model_dump on input.
    """
    remaining = MAX_NODES

    def visit(item: object, depth: int) -> Any:
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > MAX_DEPTH:
            raise CollaborationContractError("Contract traversal limit exceeded.")
        if item is None or type(item) in (bool, int, float, str):
            inspect_bounded_durable_json(
                item,
                "contract",
                max_bytes=MAX_ENVELOPE_BYTES,
                max_nodes=1,
                canonical_numbers=False,
            )
            return item
        if isinstance(item, ContractValue):
            # Fields excluded by serializers are still comparison inputs.
            raw = object.__getattribute__(item, "__dict__")
            extra = object.__getattribute__(item, "__pydantic_extra__")
            if extra or type(raw) is not dict or set(raw) != set(type(item).model_fields):
                raise CollaborationContractError("Malformed contract value.")
            item = raw
        if type(item) is dict:
            if len(item) > MAX_ENTRIES:
                raise CollaborationContractError("Contract entry limit exceeded.")
            copied = {}
            for key, child in item.items():
                if type(key) is not str:
                    raise CollaborationContractError("Contract keys must be strings.")
                copied[visit(key, depth + 1)] = visit(child, depth + 1)
            return copied
        if type(item) is list or type(item) is tuple:
            if len(item) > MAX_ENTRIES:
                raise CollaborationContractError("Contract entry limit exceeded.")
            return [visit(child, depth + 1) for child in item]
        raise CollaborationContractError("Unsupported contract value.")

    result = visit(value, 0)
    inspect_bounded_durable_json(
        result,
        "contract",
        max_bytes=MAX_ENVELOPE_BYTES,
        max_nodes=MAX_NODES,
        max_nesting=MAX_DEPTH,
        max_object_entries=MAX_ENTRIES,
        max_array_entries=MAX_ENTRIES,
        canonical_numbers=False,
    )
    return result


class ContractValue(BaseModel):
    """Internal owner schemas extend this value base; they do not inherit trust.

    Use typed scalar/model/tuple fields, never mutable arbitrary metadata. Owner
    schemas define their own complete intent and explicit-absence semantics.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        revalidate_instances="always",
        validate_default=True,
    )
    unordered_fields: ClassVar[frozenset[str]] = frozenset()

    @model_validator(mode="before")
    @classmethod
    def bounded_input(cls, value: object) -> Any:
        failed = False
        try:
            value = snapshot_input(value)
            if type(value) is not dict or not set(value) <= set(cls.model_fields):
                failed = True
        except (ValueError, TypeError, AttributeError):
            failed = True
        if failed:
            raise CollaborationContractError("Invalid or oversized contract input.") from None
        return value

    @model_validator(mode="after")
    def immutable_fields(self) -> ContractValue:
        def require_immutable(value: object) -> None:
            if (
                isinstance(value, ContractValue)
                or value is None
                or type(value)
                in (
                    str,
                    int,
                    float,
                    bool,
                )
            ):
                return
            if type(value) is tuple:
                for child in value:
                    require_immutable(child)
                return
            raise CollaborationContractError("Contract fields must be immutable typed values.")

        for value in self.__dict__.values():
            require_immutable(value)
        if not self.unordered_fields <= type(self).model_fields.keys():
            raise CollaborationContractError("Unknown unordered contract field.")
        for name in self.unordered_fields:
            values = self.__dict__[name]
            if type(values) is not tuple:
                raise CollaborationContractError("Unordered fields must be tuples.")
            indexed = [
                (
                    canonical_bounded_durable_json_bytes(
                        snapshot_input(item),
                        "contract",
                        max_bytes=MAX_ENVELOPE_BYTES,
                        max_nodes=MAX_NODES,
                        max_nesting=MAX_DEPTH,
                    ),
                    item,
                )
                for item in values
            ]
            if len({key for key, _ in indexed}) != len(indexed):
                raise CollaborationContractError("Unordered contract entries must be unique.")
            object.__setattr__(
                self, name, tuple(item for _, item in sorted(indexed, key=lambda x: x[0]))
            )
        return self


class OperationRef(ContractValue):
    application_scope: Identifier
    namespace_incarnation: Identifier
    generation: Generation
    caller_key: Identifier


class OwnerRef(ContractValue):
    application_scope: Identifier
    owner_id: Identifier
    incarnation: Identifier


class ObjectRef(ContractValue):
    owner: OwnerRef
    kind: Code
    object_id: Identifier
    incarnation: Identifier
    revision: Generation | None = None


class InitiatorBinding(ContractValue):
    issuer: OwnerRef
    principal: Identifier
    participant: ObjectRef | None
    mandate: ObjectRef | None
    invocation_id: Identifier | None
    interaction_id: Identifier | None


class TransportClaimRef(ContractValue):
    """Current servicing reference, not business identity or a live lease proof."""

    owner: OwnerRef
    worker_id: Identifier
    claim_id: Identifier
    generation: Generation


class HandoffSlot(ContractValue):
    source: OwnerRef
    parent: OperationRef
    slot: Identifier


IntentT = TypeVar("IntentT", bound=ContractValue)
ReceiptT = TypeVar("ReceiptT", bound=ContractValue)


class ExpectedOperation(ContractValue, Generic[IntentT]):
    """Owner-typed intent includes all operands and already-frozen selections.

    Unknown owner-generated selections are validated from the original receipt,
    not filled with fresh defaults during lookup. Transport claims are separate.
    """

    operation: OperationRef
    kind: Code
    schema_version: Generation
    mode: Code
    source: OwnerRef
    destination: OwnerRef
    initiator: InitiatorBinding
    receipt_stage: Code
    intent: IntentT

    @model_validator(mode="after")
    def operation_scope(self) -> ExpectedOperation[IntentT]:
        if self.operation.application_scope != self.source.application_scope:
            raise ValueError("Operation namespace conflicts with its source scope.")
        return self


class HandoffIntent(ContractValue, Generic[IntentT]):
    slot: HandoffSlot
    child: ExpectedOperation[IntentT]

    @model_validator(mode="after")
    def consistent_source(self) -> HandoffIntent[IntentT]:
        if self.slot.source != self.child.source:
            raise ValueError("Handoff source conflicts with its child command.")
        if self.slot.parent.application_scope != self.slot.source.application_scope:
            raise ValueError("Handoff parent conflicts with its source scope.")
        return self


class ExactMatch(ContractValue, Generic[ReceiptT]):
    """The receiving owner must authenticate and compare this receipt."""

    status: Literal["match"] = "match"
    receipt: ReceiptT


class ExactNotFound(ContractValue):
    status: Literal["not_found"] = "not_found"


class ExactConflict(ContractValue):
    status: Literal["conflict"] = "conflict"


class ExactUnavailable(ContractValue):
    status: Literal["unavailable"] = "unavailable"


ExactLookup = TypeAliasType(
    "ExactLookup",
    "ExactMatch[ReceiptT] | ExactNotFound | ExactConflict | ExactUnavailable",
    type_params=(ReceiptT,),
)
