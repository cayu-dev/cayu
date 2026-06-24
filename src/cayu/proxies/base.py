from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.vaults import ResolvedSecret, SecretRef


class ProxyAuthorizationResult(BaseModel):
    """Result of a credential proxy outbound authorization check."""

    model_config = ConfigDict(extra="forbid")

    allowed: StrictBool
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("reason")
    @classmethod
    def validate_optional_reason(
        cls,
        value: str | None,
        info,
    ) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @model_validator(mode="after")
    def require_reason_for_denial(self) -> ProxyAuthorizationResult:
        if not self.allowed and self.reason is None:
            raise ValueError("Denied proxy authorizations require a reason.")
        return self


class CredentialProxy(ABC):
    """Credential boundary for trusted tools and environment integrations.

    A proxy can resolve scoped secret references and authorize outbound actions
    without making raw credentials part of model-visible context. Generic
    sandbox command execution is not automatically routed through this contract.
    """

    @abstractmethod
    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret:
        """Resolve a secret reference at the credential boundary."""

    @abstractmethod
    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ProxyAuthorizationResult:
        """Return whether an outbound action is allowed."""


def copy_proxy_authorization_result(
    result: ProxyAuthorizationResult,
) -> ProxyAuthorizationResult:
    if type(result) is not ProxyAuthorizationResult:
        raise TypeError("Proxy authorization results must be ProxyAuthorizationResult instances.")
    return ProxyAuthorizationResult(
        allowed=result.allowed,
        reason=result.reason,
        metadata=copy_json_value(result.metadata, "metadata"),
    )
