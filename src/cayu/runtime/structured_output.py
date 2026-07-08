from __future__ import annotations

import json
from enum import StrEnum
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationInfo, field_validator

from cayu._validation import copy_json_value, escape_json_pointer_segment, require_clean_nonblank

STRUCTURED_OUTPUT_TOOL_NAME = "__cayu_submit_structured_output"


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

    model_config = ConfigDict(extra="forbid")

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
        return StructuredOutputStrategy(require_clean_nonblank(value, "strategy"))

    @field_validator("json_schema", mode="before")
    @classmethod
    def copy_and_validate_json_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_json_value(value, "json_schema")
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
        return require_clean_nonblank(value, field_name)


class StructuredOutputError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    message: str
    schema_path: str


class StructuredOutputValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    output: Any | None = None
    errors: list[StructuredOutputError] = Field(default_factory=list)

    @field_validator("output", mode="before")
    @classmethod
    def copy_output(cls, value: Any) -> Any:
        if value is None:
            return None
        return copy_json_value(value, "output")


def copy_structured_output_spec(
    spec: StructuredOutputSpec | None,
) -> StructuredOutputSpec | None:
    if spec is None:
        return None
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    return StructuredOutputSpec(
        json_schema=copy_json_value(spec.json_schema, "json_schema"),
        name=spec.name,
        max_retries=spec.max_retries,
        repair_prompt=spec.repair_prompt,
        strategy=spec.strategy,
    )


def validate_structured_output_text(
    text: str,
    spec: StructuredOutputSpec,
) -> StructuredOutputValidation:
    if type(text) is not str:
        raise TypeError("Structured output text must be a string.")
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")

    stripped = text.strip()
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
        output = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return StructuredOutputValidation(
            valid=False,
            errors=[
                StructuredOutputError(
                    path="$",
                    message=f"Final assistant output is not valid JSON: {exc.msg}.",
                    schema_path="$",
                )
            ],
        )

    return validate_structured_output_value(output, spec)


def validate_structured_output_value(
    output: Any,
    spec: StructuredOutputSpec,
) -> StructuredOutputValidation:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    copied_output = copy_json_value(output, "output")
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
                "output": copy_json_value(spec.json_schema, "json_schema"),
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
        "schema": copy_json_value(spec.json_schema, "json_schema"),
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
