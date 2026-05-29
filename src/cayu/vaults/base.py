from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from cayu._validation import copy_json_value, require_nonblank


class SecretRef(BaseModel):
    """Reference to a secret value.

    The raw value should be injected into tools/runners by the runtime and
    should not be placed in model prompt text.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    handle: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("handle")
    @classmethod
    def validate_nonblank_handle(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)


def copy_secret_ref(ref: SecretRef) -> SecretRef:
    if type(ref) is not SecretRef:
        raise TypeError("Secret references must be SecretRef instances.")
    return SecretRef(
        name=ref.name,
        handle=ref.handle,
        metadata=copy_json_value(ref.metadata, "metadata"),
    )


class ResolvedSecret(BaseModel):
    """Resolved secret value for runtime injection only.

    `value` uses SecretStr so accidental dumps/logs do not reveal the raw
    secret. Runtime code must explicitly call `get_secret_value()` at the last
    possible moment before injecting into a tool/runner environment.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: SecretStr
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)


class Vault(ABC):
    """Secrets lookup contract."""

    @abstractmethod
    async def get(self, name: str, *, scope: dict[str, Any] | None = None) -> SecretRef:
        """Resolve a secret reference."""

    @abstractmethod
    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        """Resolve a secret reference to a masked value for runtime injection."""
