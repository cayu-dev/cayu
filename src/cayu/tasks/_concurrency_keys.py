"""Shared creation and persistence contract for optional task concurrency keys."""

from typing import Annotated, TypeAlias

from pydantic import Field

ConcurrencyKey: TypeAlias = Annotated[
    str | None,
    Field(
        strict=True,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:/@#=+-]*$",
        frozen=True,
        exclude_if=lambda value: value is None,
    ),
]
