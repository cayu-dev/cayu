"""Pure preparation and comparison; no policy resolution or durable effects."""

from __future__ import annotations

from types import UnionType
from typing import Annotated, Literal, TypeVar, Union, get_args, get_origin

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.collaboration._contracts import (
    MAX_DEPTH,
    MAX_ENVELOPE_BYTES,
    MAX_NODES,
    CollaborationConflict,
    CollaborationContractError,
    ContractValue,
    ExactConflict,
    ExactLookup,
    ExactMatch,
    ExactNotFound,
    ExactUnavailable,
    snapshot_input,
)
from cayu.vaults.redaction import SecretRedactor

ValueT = TypeVar("ValueT", bound=ContractValue)


def _literal_controls(annotation: object) -> tuple[object, ...] | None:
    """Only closed literal/None unions are controls; a text branch is not."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _literal_controls(args[0])
    if origin is Literal:
        return args
    if annotation is type(None):
        return (None,)
    if origin in (Union, UnionType):
        controls = []
        for branch in args:
            members = _literal_controls(branch)
            if members is None:
                return None
            controls.extend(members)
        return tuple(controls)
    return None


def _structural_annotation(annotation: object) -> object:
    """Unwrap metadata and a single nullable branch, never choose a text union."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Annotated:
        return _structural_annotation(args[0])
    if origin in (Union, UnionType):
        members = tuple(branch for branch in args if branch is not type(None))
        if len(members) == 1:
            return _structural_annotation(members[0])
    return annotation


def _tuple_member_annotation(annotation: object, index: int) -> object:
    annotation = _structural_annotation(annotation)
    args = get_args(annotation)
    if get_origin(annotation) is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        if index < len(args):
            return args[index]
    return object


def _require_raw_secret_free(
    annotation: object,
    value: object,
    redactor: SecretRedactor,
) -> None:
    """Inspect original bounded values before owner validators can transform them."""
    controls = _literal_controls(annotation)
    if controls is not None and any(
        type(value) is type(item) and value == item for item in controls
    ):
        return
    annotation = _structural_annotation(annotation)
    origin = get_origin(annotation)
    args = get_args(annotation)
    if type(value) is str:
        if redactor.redact_text(value) != value:
            raise CollaborationContractError("Contract contains a workload secret.")
        return
    if type(value) is dict:
        # Pydantic's generic models do not retain a free TypeVar inside an
        # ordinary runtime union. The named lookup alias preserves that binding;
        # expand its trusted schema here without invoking receipt validators.
        candidates = (
            (
                ExactMatch.__class_getitem__(args[0]),
                ExactNotFound,
                ExactConflict,
                ExactUnavailable,
            )
            if origin is ExactLookup
            else (annotation, *args)
        )
        compatible = tuple(
            candidate
            for candidate in candidates
            if isinstance(candidate, type)
            and issubclass(candidate, ContractValue)
            and set(value) <= candidate.model_fields.keys()
            and all(
                name in value or not field.is_required()
                for name, field in candidate.model_fields.items()
            )
            and all(
                name not in value
                or (field_controls := _literal_controls(field.annotation)) is None
                or any(
                    type(value[name]) is type(control) and value[name] == control
                    for control in field_controls
                )
                for name, field in candidate.model_fields.items()
            )
        )
        # Never run owner validators to select a branch before inspecting raw
        # secrets. Only an unambiguous structural match grants schema-key and
        # literal-control exemptions; ambiguity is inspected as untrusted data.
        schema = compatible[0] if len(compatible) == 1 else None
        for key, item in value.items():
            if schema is None:
                _require_raw_secret_free(str, key, redactor)
                child_annotation = object
            else:
                child_annotation = schema.model_fields[key].annotation
            _require_raw_secret_free(child_annotation, item, redactor)
    elif type(value) is list or type(value) is tuple:
        for index, item in enumerate(value):
            _require_raw_secret_free(_tuple_member_annotation(annotation, index), item, redactor)


def _require_secret_free(annotation: object, value: object, redactor: SecretRedactor) -> None:
    controls = _literal_controls(annotation)
    if controls is not None:
        # After-validators can replace a value after its literal validation.
        if not any(type(value) is type(control) and value == control for control in controls):
            raise CollaborationContractError("Invalid contract control.")
        return
    if isinstance(value, ContractValue):
        for name, field in type(value).model_fields.items():
            child = object.__getattribute__(value, "__dict__")[name]
            _require_secret_free(field.annotation, child, redactor)
    elif type(value) is str:
        if redactor.redact_text(value) != value:
            raise CollaborationContractError("Contract contains a workload secret.")
    elif type(value) is tuple:
        for index, child in enumerate(value):
            _require_secret_free(_tuple_member_annotation(annotation, index), child, redactor)
    elif type(value) in (dict, list):
        # Owner schemas must not admit mutable payloads as frozen authority.
        raise CollaborationContractError("Contract requires immutable typed fields.")
    elif value is not None and type(value) not in (bool, int, float):
        raise CollaborationContractError("Unsupported contract field.")


def prepare_contract(
    schema: type[ValueT],
    value: object,
    *,
    redactor: SecretRedactor,
) -> ValueT:
    """Revalidate before encoding; never return raw framework validation errors.

    The supplied schema/redactor are trusted application code. This returns a
    detached value, not authorization. No external operation is called here.
    """
    if not isinstance(schema, type) or not issubclass(schema, ContractValue):
        raise TypeError("A collaboration contract schema is required.")
    if type(redactor) is not SecretRedactor:
        raise TypeError("A trusted secret redactor is required.")
    failed = False
    result: ValueT | None = None
    try:
        plain = snapshot_input(value)
        _require_raw_secret_free(schema, plain, redactor)
        result = schema.model_validate(plain)
        _require_secret_free(schema, result, redactor)
        canonical_bounded_durable_json_bytes(
            snapshot_input(result),
            "contract",
            max_bytes=MAX_ENVELOPE_BYTES,
            max_nodes=MAX_NODES,
            max_nesting=MAX_DEPTH,
        )
    except (ValueError, TypeError, AttributeError):
        failed = True
    if failed:
        raise CollaborationContractError("Invalid or unsafe collaboration contract.") from None
    assert result is not None
    return result


def contract_bytes(value: ValueT, *, redactor: SecretRedactor) -> bytes:
    """Canonical complete material; callers must retain it, not only a digest."""
    if not isinstance(value, ContractValue):
        raise CollaborationContractError("A typed collaboration contract is required.")
    checked = prepare_contract(type(value), value, redactor=redactor)
    return canonical_bounded_durable_json_bytes(
        snapshot_input(checked),
        "contract",
        max_bytes=MAX_ENVELOPE_BYTES,
        max_nodes=MAX_NODES,
        max_nesting=MAX_DEPTH,
    )


def require_exact_contract(
    expected: ValueT,
    observed: ValueT,
    *,
    redactor: SecretRedactor,
) -> None:
    """Full expected equality, not authentication or a current permission check."""
    if type(expected) is not type(observed):
        raise CollaborationConflict("Contract schema conflicts with expected input.")
    if contract_bytes(expected, redactor=redactor) != contract_bytes(observed, redactor=redactor):
        raise CollaborationConflict("Contract conflicts with expected input.")
