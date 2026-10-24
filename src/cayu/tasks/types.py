"""Host declarations for task types exposed through the control plane."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator, SchemaError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from referencing import Registry
from referencing.jsonschema import DRAFT202012

if TYPE_CHECKING:
    from jsonschema.protocols import Validator

from cayu._validation import copy_durable_json_object, require_durable_clean_nonblank


class TaskTypeSpec(BaseModel):
    """A host-declared task input contract; execution remains worker-owned."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str = Field(max_length=256)
    description: str = Field(default="", max_length=4096)
    agent_name: str | None = Field(default=None, max_length=256)
    input_schema: dict[str, Any]

    @field_validator("name", "agent_name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_durable_clean_nonblank(value, "task type identity")

    @field_validator("input_schema", mode="before")
    @classmethod
    def validate_schema(cls, value: Any) -> dict[str, Any]:
        schema = copy_durable_json_object(value, "task type schema")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError:
            raise ValueError("Task input schema must be a valid Draft 2020-12 schema.") from None
        pending: list[Any] = [schema]
        while pending:
            item = pending.pop()
            if type(item) is dict:
                if any(
                    key in {"$ref", "$dynamicRef"}
                    and (type(value) is not str or not value.startswith("#"))
                    for key, value in item.items()
                ):
                    raise ValueError("Task input schemas must use local references only.")
                if "$schema" in item and item["$schema"] not in {
                    "https://json-schema.org/draft/2020-12/schema",
                    "https://json-schema.org/draft/2020-12/schema#",
                }:
                    raise ValueError("Task input schemas must use Draft 2020-12.")
                pending.extend(DRAFT202012.subresources_of(item))
        return schema


def task_input_validator(spec: TaskTypeSpec) -> Validator:
    """Never retrieve remote schema resources during input validation."""
    return Draft202012Validator(spec.input_schema, registry=Registry())


def copy_task_type(spec: TaskTypeSpec) -> TaskTypeSpec:
    if type(spec) is not TaskTypeSpec:
        raise TypeError("Task type registration requires a TaskTypeSpec.")
    return TaskTypeSpec(
        name=spec.name,
        description=spec.description,
        agent_name=spec.agent_name,
        input_schema=spec.input_schema,
    )
