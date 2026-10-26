"""Presentation-only tool labels; never execution or authorization evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from cayu._validation import require_durable_clean_nonblank


class ToolDisplay(BaseModel):
    """A faithful summary verb, optional singular noun and optional irregular plural.

    Without a noun, consumers render the verb with a call count (for example,
    ``Searched 3 times``). With a noun, they render ``Read 3 files``; an omitted
    plural defaults to the noun plus ``s``. Missing display uses the tool name.
    Labels are public presentation metadata, not instructions sent to a model.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    verb: StrictStr
    noun: StrictStr | None = None
    plural: StrictStr | None = None

    @field_validator("verb", "noun", "plural")
    @classmethod
    def validate_label(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        assert info.field_name is not None
        value = require_durable_clean_nonblank(value, info.field_name)
        if any(
            ord(character) < 32 or 127 <= ord(character) <= 159 or character in "\u2028\u2029"
            for character in value
        ):
            raise ValueError("Tool display labels must be single-line text without controls.")
        return value

    @model_validator(mode="after")
    def validate_plural(self) -> ToolDisplay:
        if self.plural is not None and self.noun is None:
            raise ValueError("A tool display plural requires a noun.")
        return self

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> ToolDisplay:
        values = {name: getattr(self, name) for name in ToolDisplay.model_fields}
        if update:
            values.update(update)
        return type(self).model_validate(values)


def copy_tool_display(value: object) -> ToolDisplay | None:
    """Validate before serialization and own a snapshot of the declared strings."""
    if value is None:
        return None
    if type(value) is ToolDisplay:
        return ToolDisplay.model_copy(value)
    if type(value) is not dict:
        raise ValueError("Tool display must be a ToolDisplay or an object.")
    return ToolDisplay.model_validate(value)
