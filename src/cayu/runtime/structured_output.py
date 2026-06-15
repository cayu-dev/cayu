from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationInfo, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank


class StructuredOutputSpec(BaseModel):
    """Provider-neutral JSON structured output requirement."""

    model_config = ConfigDict(extra="forbid")

    json_schema: dict[str, Any]
    name: str | None = None
    max_retries: StrictInt = Field(default=1, ge=0, le=8)
    repair_prompt: str | None = None

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

    validator = Draft202012Validator(spec.json_schema)
    errors = sorted(
        validator.iter_errors(output),
        key=lambda error: (list(error.path), list(error.schema_path), error.message),
    )
    if not errors:
        return StructuredOutputValidation(valid=True, output=output)

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

    lead = spec.repair_prompt or (
        "Your previous final response did not match the required structured output "
        "schema. Return only valid JSON that matches the schema. Do not include "
        "Markdown fences or explanatory text."
    )
    schema_text = json.dumps(spec.json_schema, indent=2, sort_keys=True)
    error_lines = "\n".join(f"- {error.path}: {error.message}" for error in validation.errors)
    return f"{lead}\n\nSchema:\n{schema_text}\n\nValidation errors:\n{error_lines}"


def structured_output_spec_payload(spec: StructuredOutputSpec) -> dict[str, Any]:
    if type(spec) is not StructuredOutputSpec:
        raise TypeError("Structured output spec must be a StructuredOutputSpec instance.")
    return {
        "name": spec.name,
        "schema": copy_json_value(spec.json_schema, "json_schema"),
        "max_retries": spec.max_retries,
        "repair_prompt": spec.repair_prompt,
    }


def _json_path(parts: Any) -> str:
    path = "$"
    for part in parts:
        if type(part) is int:
            path = f"{path}[{part}]"
        else:
            escaped = str(part).replace("~", "~0").replace("/", "~1")
            path = f"{path}/{escaped}"
    return path
