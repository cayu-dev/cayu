from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationInfo, field_validator

from cayu._validation import (
    DurableValueError,
    copy_durable_json_value,
    durable_json_object_from_pairs,
    escape_json_pointer_segment,
    parse_durable_json_integer_literal,
    reject_nonportable_json_constant,
    require_durable_clean_nonblank,
    require_durable_text,
)
from cayu.providers import ModelProvider
from cayu.vaults.redaction import SecretRedactor

STRUCTURED_OUTPUT_TOOL_NAME = "__cayu_submit_structured_output"
_STRUCTURED_OUTPUT_TEXT_FIELD = "structured output"
_INVALID_PORTABLE_JSON_MESSAGE = "Final assistant output is not valid portable JSON."

# JSON Schema object members are typed protocol structure, including when they
# recur below another schema-valued member. Names below map-valued members such
# as ``properties`` and ``$defs`` remain caller-controlled and must never gain
# this exemption.
_JSON_SCHEMA_KEYWORDS = frozenset(
    {
        "$anchor",
        "$comment",
        "$defs",
        "$dynamicAnchor",
        "$dynamicRef",
        "$id",
        "$recursiveAnchor",
        "$recursiveRef",
        "$ref",
        "$schema",
        "$vocabulary",
        "additionalItems",
        "additionalProperties",
        "allOf",
        "anyOf",
        "const",
        "contains",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "default",
        "definitions",
        "dependencies",
        "dependentRequired",
        "dependentSchemas",
        "deprecated",
        "description",
        "disallow",
        "divisibleBy",
        "else",
        "enum",
        "examples",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "extends",
        "format",
        "id",
        "if",
        "items",
        "maxContains",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minContains",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "multipleOf",
        "not",
        "nullable",
        "oneOf",
        "pattern",
        "patternProperties",
        "prefixItems",
        "properties",
        "propertyNames",
        "readOnly",
        "required",
        "then",
        "title",
        "type",
        "unevaluatedItems",
        "unevaluatedProperties",
        "uniqueItems",
        "writeOnly",
    }
)
_JSON_SCHEMA_MAP_OF_SCHEMAS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)
_JSON_SCHEMA_SINGLE_SCHEMAS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_JSON_SCHEMA_ARRAYS_OF_SCHEMAS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_JSON_SCHEMA_LEGACY_SCHEMA_OR_ARRAY_MEMBERS = frozenset({"disallow", "extends", "type"})


class StructuredOutputStrategy(StrEnum):
    TOOL = "tool"
    NATIVE = "native"


class NativeStructuredOutputUnsupported(ValueError):
    """``strategy=NATIVE`` was requested but the resolved provider does not
    support native structured output.

    Raised before any session is created or transitioned, so the caller can
    retry with ``strategy="tool"`` (same JSON contract, provider-neutral
    transport) or route to a provider that sets
    ``supports_native_structured_output``. Subclasses ``ValueError`` so
    existing handlers (including the server's 4xx mapping) keep working.
    """


class StructuredOutputSpec(BaseModel):
    """Provider-neutral JSON structured output requirement."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    json_schema: dict[str, Any]
    name: str | None = None
    max_retries: StrictInt = Field(default=2, ge=0, le=8)
    repair_prompt: str | None = None
    strategy: StructuredOutputStrategy = StructuredOutputStrategy.TOOL

    @field_validator("strategy", mode="before")
    @classmethod
    def validate_strategy(cls, value: object) -> StructuredOutputStrategy:
        if isinstance(value, StructuredOutputStrategy):
            return value
        if not isinstance(value, str):
            raise ValueError("Structured output strategy must be a string.")
        return StructuredOutputStrategy(require_durable_clean_nonblank(value, "strategy"))

    @field_validator("json_schema", mode="before")
    @classmethod
    def copy_and_validate_json_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_durable_json_value(value, "json_schema")
        if type(copied) is not dict:
            raise ValueError("Structured output JSON Schema must be an object.")
        try:
            Draft202012Validator.check_schema(copied)
        except SchemaError as exc:
            raise ValueError(f"Invalid structured output JSON Schema: {exc.message}") from exc
        return copied

    @field_validator("name", "repair_prompt")
    @classmethod
    def validate_optional_nonblank_strings(
        cls,
        value: str | None,
        info: ValidationInfo,
    ) -> str | None:
        if value is None:
            return None
        field_name = info.field_name or "value"
        return require_durable_clean_nonblank(value, field_name)


class StructuredOutputError(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    path: str
    message: str
    schema_path: str

    @field_validator("path", "message", "schema_path")
    @classmethod
    def validate_durable_text(cls, value: str, info: ValidationInfo) -> str:
        return require_durable_text(value, info.field_name or "value")


class StructuredOutputValidation(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    valid: bool
    output: Any | None = None
    errors: list[StructuredOutputError] = Field(default_factory=list)

    @field_validator("output", mode="before")
    @classmethod
    def copy_output(cls, value: Any) -> Any:
        if value is None:
            return None
        return copy_durable_json_value(value, "output")

    @field_validator("errors", mode="before")
    @classmethod
    def detach_errors(cls, value):
        if not isinstance(value, list | tuple):
            return value
        return [
            StructuredOutputError.model_validate(error.model_dump(mode="python", warnings=False))
            if isinstance(error, StructuredOutputError)
            else error
            for error in value
        ]


def _require_native_structured_output_support(
    structured_output: StructuredOutputSpec | None,
    *,
    provider_name: str,
    provider: ModelProvider,
) -> None:
    """Reject native structured output before durable session state changes."""

    if structured_output is None or structured_output.strategy != StructuredOutputStrategy.NATIVE:
        return
    if not provider.supports_native_structured_output:
        raise NativeStructuredOutputUnsupported(
            f"Native structured output is not supported by provider: {provider_name}"
        )
    provider.preflight_native_structured_output_schema(
        copy_durable_json_value(structured_output.json_schema, "json_schema")
    )


def copy_structured_output_spec(
    spec: StructuredOutputSpec | None,
) -> StructuredOutputSpec | None:
    if spec is None:
        return None
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    return StructuredOutputSpec(
        json_schema=copy_durable_json_value(spec.json_schema, "json_schema"),
        name=spec.name,
        max_retries=spec.max_retries,
        repair_prompt=spec.repair_prompt,
        strategy=spec.strategy,
    )


def require_secret_free_json_schema_keys(
    schema: dict[str, Any],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> None:
    """Reject secrets in data-owned JSON Schema keys without rejecting keywords."""

    copied = copy_durable_json_value(schema, field_name)
    if type(copied) is not dict:
        raise TypeError("schema must be a JSON object.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")

    def require_untrusted_key(key: str, *, path: str) -> None:
        redactor.require_no_secret_keys(
            {key: None},
            field_name=path,
            match_short_substrings=True,
        )

    def inspect_untyped(value: Any, *, path: str) -> None:
        redactor.require_no_secret_keys(
            value,
            field_name=path,
            match_short_substrings=True,
        )

    def inspect_schema(value: Any, *, path: str) -> None:
        if type(value) is bool:
            return
        if type(value) is not dict:
            inspect_untyped(value, path=path)
            return
        for key, item in value.items():
            member_path = f"{path}.{key}"
            if key not in _JSON_SCHEMA_KEYWORDS:
                require_untrusted_key(key, path=path)
                inspect_untyped(item, path=member_path)
                continue
            if key in _JSON_SCHEMA_MAP_OF_SCHEMAS:
                if type(item) is not dict:
                    inspect_untyped(item, path=member_path)
                    continue
                for dynamic_key, child_schema in item.items():
                    require_untrusted_key(dynamic_key, path=member_path)
                    inspect_schema(
                        child_schema,
                        path=f"{member_path}.{dynamic_key}",
                    )
                continue
            if key == "items" and type(item) is list:
                for index, child_schema in enumerate(item):
                    inspect_schema(child_schema, path=f"{member_path}[{index}]")
                continue
            if key in _JSON_SCHEMA_SINGLE_SCHEMAS:
                inspect_schema(item, path=member_path)
                continue
            if key in _JSON_SCHEMA_ARRAYS_OF_SCHEMAS:
                if type(item) is list:
                    for index, child_schema in enumerate(item):
                        inspect_schema(child_schema, path=f"{member_path}[{index}]")
                else:
                    inspect_untyped(item, path=member_path)
                continue
            if key in _JSON_SCHEMA_LEGACY_SCHEMA_OR_ARRAY_MEMBERS:
                if type(item) is dict:
                    inspect_schema(item, path=member_path)
                elif type(item) is list:
                    for index, member in enumerate(item):
                        if type(member) is dict:
                            inspect_schema(member, path=f"{member_path}[{index}]")
                        else:
                            inspect_untyped(member, path=f"{member_path}[{index}]")
                else:
                    inspect_untyped(item, path=member_path)
                continue
            if key == "dependencies" and type(item) is dict:
                for dynamic_key, dependency in item.items():
                    require_untrusted_key(dynamic_key, path=member_path)
                    if type(dependency) in {dict, bool}:
                        inspect_schema(
                            dependency,
                            path=f"{member_path}.{dynamic_key}",
                        )
                    else:
                        inspect_untyped(
                            dependency,
                            path=f"{member_path}.{dynamic_key}",
                        )
                continue
            inspect_untyped(item, path=member_path)

    inspect_schema(copied, path=field_name)


def json_schema_contains_secret(
    schema: dict[str, Any],
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> bool:
    """Return whether schema-owned keys or any schema value contain a secret."""

    try:
        require_secret_free_json_schema_keys(
            schema,
            redactor=redactor,
            field_name=field_name,
        )
    except ValueError:
        return True
    return redactor.redact_json_values(schema) != schema


def require_secret_free_structured_output_spec(
    spec: StructuredOutputSpec | None,
    *,
    redactor: SecretRedactor,
    field_name: str,
) -> None:
    """Reject a spec that redaction would change at an execution boundary."""

    if spec is None:
        return
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    if not isinstance(redactor, SecretRedactor):
        raise TypeError("redactor must be a SecretRedactor.")
    payload = spec.model_dump(mode="json")
    public_structure = {
        "json_schema",
        "max_retries",
        "name",
        "repair_prompt",
        "strategy",
    }
    redactor.require_no_secret_keys(
        {
            "json_schema": None,
            "max_retries": payload["max_retries"],
            "name": payload["name"],
            "repair_prompt": payload["repair_prompt"],
            "strategy": payload["strategy"],
        },
        field_name=field_name,
        preserve_keys=public_structure,
        match_short_substrings=True,
    )
    require_secret_free_json_schema_keys(
        spec.json_schema,
        redactor=redactor,
        field_name=f"{field_name}.json_schema",
    )
    for authority_field in ("name", "strategy"):
        value = payload.get(authority_field)
        if type(value) is str and redactor.redact_text(value) != value:
            raise ValueError(
                f"{field_name}.{authority_field} contains a workload secret and cannot "
                "be used as structured-output execution authority."
            )
    redacted_payload = redactor.redact_json_values(
        payload,
        preserve_string_fields={"name", "strategy"},
    )
    if redacted_payload != payload:
        raise ValueError(
            f"{field_name} contains a workload secret and cannot cross the boundary "
            "without changing execution semantics."
        )


def validate_structured_output_text(
    text: str,
    spec: StructuredOutputSpec,
) -> StructuredOutputValidation:
    if type(text) is not str:
        raise TypeError("Structured output text must be a string.")
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")

    try:
        stripped = require_durable_text(text, _STRUCTURED_OUTPUT_TEXT_FIELD).strip()
    except DurableValueError:
        return _invalid_portable_json_validation()
    if not stripped:
        return StructuredOutputValidation(
            valid=False,
            errors=[
                StructuredOutputError(
                    path="$",
                    message="Final assistant output is empty.",
                    schema_path="$",
                )
            ],
        )
    try:
        output = json.loads(
            stripped,
            parse_constant=_reject_structured_output_json_constant,
            parse_int=_parse_structured_output_json_integer,
            object_pairs_hook=_reject_structured_output_duplicate_keys,
        )
        return validate_structured_output_value(output, spec)
    except (DurableValueError, json.JSONDecodeError, RecursionError):
        return _invalid_portable_json_validation()


def _parse_structured_output_json_integer(value: str) -> int:
    return parse_durable_json_integer_literal(value, _STRUCTURED_OUTPUT_TEXT_FIELD)


def _reject_structured_output_json_constant(value: str) -> None:
    reject_nonportable_json_constant(value, _STRUCTURED_OUTPUT_TEXT_FIELD)


def _reject_structured_output_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    return durable_json_object_from_pairs(pairs, _STRUCTURED_OUTPUT_TEXT_FIELD)


def _invalid_portable_json_validation() -> StructuredOutputValidation:
    return StructuredOutputValidation(
        valid=False,
        errors=[
            StructuredOutputError(
                path="$",
                message=_INVALID_PORTABLE_JSON_MESSAGE,
                schema_path="$",
            )
        ],
    )


def validate_structured_output_value(
    output: Any,
    spec: StructuredOutputSpec,
) -> StructuredOutputValidation:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    copied_output = copy_durable_json_value(output, "output")
    validator = Draft202012Validator(spec.json_schema)
    errors = sorted(
        validator.iter_errors(copied_output),
        key=lambda error: (list(error.path), list(error.schema_path), error.message),
    )
    if not errors:
        return StructuredOutputValidation(valid=True, output=copied_output)

    return StructuredOutputValidation(
        valid=False,
        errors=[
            StructuredOutputError(
                path=_json_path(error.path),
                message=error.message,
                schema_path=_json_path(error.schema_path),
            )
            for error in errors[:8]
        ],
    )


def structured_output_repair_prompt(
    *,
    spec: StructuredOutputSpec,
    validation: StructuredOutputValidation,
) -> str:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    if type(validation) is not StructuredOutputValidation:
        raise TypeError("Structured output validation must be a StructuredOutputValidation.")

    lead = structured_output_repair_lead(spec)
    schema_text = json.dumps(spec.json_schema, indent=2, sort_keys=True)
    error_lines = "\n".join(f"- {error.path}: {error.message}" for error in validation.errors)
    return f"{lead}\n\nSchema:\n{schema_text}\n\nValidation errors:\n{error_lines}"


def structured_output_repair_lead(spec: StructuredOutputSpec) -> str:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    if spec.repair_prompt is not None:
        return spec.repair_prompt
    if spec.strategy == StructuredOutputStrategy.NATIVE:
        return (
            "Your previous response did not satisfy the required structured output contract. "
            "Return only valid JSON that matches the schema. Do not include Markdown fences "
            "or explanatory text."
        )
    return (
        "Your previous response did not satisfy the required structured output contract. "
        f"Call the `{STRUCTURED_OUTPUT_TOOL_NAME}` tool with an `output` argument that "
        "matches the schema. Do not return the final structured output as plain text."
    )


def structured_output_tool_instruction(spec: StructuredOutputSpec) -> str:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    schema_text = json.dumps(spec.json_schema, indent=2, sort_keys=True)
    return (
        "When you have the final answer for this request, call the "
        f"`{STRUCTURED_OUTPUT_TOOL_NAME}` tool. Put the final structured value in the "
        "`output` argument. Do not use that tool for intermediate work, and do not call "
        "it in the same tool round as any other tool.\n\n"
        f"Required output JSON Schema:\n{schema_text}"
    )


def structured_output_tool_spec(spec: StructuredOutputSpec) -> dict[str, Any]:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    return {
        "name": STRUCTURED_OUTPUT_TOOL_NAME,
        "description": (
            "Submit the final structured output for this run. Use this only when the "
            "final answer is ready. The value must be provided in the `output` field."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "output": copy_durable_json_value(spec.json_schema, "json_schema"),
            },
            "required": ["output"],
            "additionalProperties": False,
        },
    }


def validate_structured_output_tool_arguments(
    arguments: dict[str, Any],
    spec: StructuredOutputSpec,
) -> StructuredOutputValidation:
    if type(arguments) is not dict:
        raise TypeError("Structured output tool arguments must be an object.")
    if "output" not in arguments:
        return StructuredOutputValidation(
            valid=False,
            errors=[
                StructuredOutputError(
                    path="$/output",
                    message="Structured output tool arguments require `output`.",
                    schema_path="$/required",
                )
            ],
        )
    return validate_structured_output_value(arguments["output"], spec)


def structured_output_tool_required_validation() -> StructuredOutputValidation:
    return StructuredOutputValidation(
        valid=False,
        errors=[
            StructuredOutputError(
                path="$",
                message=(
                    "Final structured output must be submitted with the "
                    f"`{STRUCTURED_OUTPUT_TOOL_NAME}` tool."
                ),
                schema_path="$",
            )
        ],
    )


def structured_output_spec_payload(spec: StructuredOutputSpec) -> dict[str, Any]:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    strategy = (
        spec.strategy.value
        if isinstance(spec.strategy, StructuredOutputStrategy)
        else spec.strategy
    )
    return {
        "name": spec.name,
        "schema": copy_durable_json_value(spec.json_schema, "json_schema"),
        "max_retries": spec.max_retries,
        "repair_prompt": spec.repair_prompt,
        "strategy": strategy,
    }


def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        if type(part) is int:
            path = f"{path}[{part}]"
        else:
            path = f"{path}/{escape_json_pointer_segment(str(part))}"
    return path
