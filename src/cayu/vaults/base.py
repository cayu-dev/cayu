from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class SecretRef(BaseModel):
    """Reference to a secret value.

    The raw value should be injected into tools/runners by the runtime and
    should not be placed in model prompt text.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    handle: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


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
