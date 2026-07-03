from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank


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
        return require_clean_nonblank(value, info.field_name)

    @field_validator("handle")
    @classmethod
    def validate_nonblank_handle(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class SecretEnv(BaseModel):
    """Environment variable whose value must be resolved from a secret ref."""

    model_config = ConfigDict(extra="forbid")

    name: str
    ref: SecretRef
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("ref")
    @classmethod
    def copy_ref(cls, value: SecretRef) -> SecretRef:
        return copy_secret_ref(value)


def copy_secret_ref(ref: SecretRef) -> SecretRef:
    if type(ref) is not SecretRef:
        raise TypeError("Secret references must be SecretRef instances.")
    return SecretRef(
        name=ref.name,
        handle=ref.handle,
        metadata=copy_json_value(ref.metadata, "metadata"),
    )


def copy_secret_env(secret_env: SecretEnv) -> SecretEnv:
    if type(secret_env) is not SecretEnv:
        raise TypeError("Secret environment entries must be SecretEnv instances.")
    return SecretEnv(
        name=secret_env.name,
        ref=copy_secret_ref(secret_env.ref),
        metadata=copy_json_value(secret_env.metadata, "metadata"),
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
        return require_clean_nonblank(value, info.field_name)

    @field_validator("value")
    @classmethod
    def validate_nonblank_value(cls, value: SecretStr, info) -> SecretStr:
        require_nonblank(value.get_secret_value(), info.field_name)
        return value


def copy_resolved_secret(secret: ResolvedSecret) -> ResolvedSecret:
    if type(secret) is not ResolvedSecret:
        raise TypeError("Resolved secrets must be ResolvedSecret instances.")
    return ResolvedSecret(
        name=secret.name,
        value=SecretStr(secret.value.get_secret_value()),
        metadata=copy_json_value(secret.metadata, "metadata"),
    )


class VaultError(RuntimeError):
    """Base error for vault resolution failures."""


class SecretNotFound(VaultError):
    """Raised when a vault cannot resolve a requested secret."""


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


@runtime_checkable
class SecretResolver(Protocol):
    """Async credential source shared by ``Vault`` and ``CredentialProxy``.

    Anything with an async ``resolve(ref, *, scope=None) -> ResolvedSecret``
    method satisfies this seam, so runners, MCP clients, and providers can
    consume secrets from a vault directly or through a credential proxy.
    """

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> ResolvedSecret: ...


def validate_secret_resolver(resolver: object, field_name: str = "secret_resolver") -> None:
    """Reject objects that cannot asynchronously resolve secret references."""

    resolve = getattr(resolver, "resolve", None)
    if resolve is None or not callable(resolve):
        raise TypeError(f"{field_name} must provide an async resolve(ref, *, scope=) method.")
    if not inspect.iscoroutinefunction(resolve):
        raise TypeError(f"{field_name} resolve() must be an async method.")


def secret_env_refs(
    secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef],
) -> dict[str, SecretRef]:
    """Normalize declared secret environment entries to name -> SecretRef.

    Accepts either a sequence of ``SecretEnv`` entries or a plain
    ``{env_var_name: SecretRef}`` mapping. Duplicate environment variable
    names are rejected because injection would silently drop one secret.
    """

    refs: dict[str, SecretRef] = {}
    if isinstance(secret_env, Mapping):
        for name, ref in cast("Mapping[str, SecretRef]", secret_env).items():
            env_name = require_clean_nonblank(name, "secret_env name")
            refs[env_name] = copy_secret_ref(ref)
        return refs
    if not isinstance(secret_env, Sequence) or isinstance(secret_env, str | bytes):
        raise TypeError("secret_env must be a sequence of SecretEnv or a name->SecretRef mapping.")
    for entry in secret_env:
        if type(entry) is not SecretEnv:
            raise TypeError("secret_env entries must be SecretEnv instances.")
        if entry.name in refs:
            raise ValueError(f"secret_env declares duplicate environment variable: {entry.name}")
        refs[entry.name] = copy_secret_ref(entry.ref)
    return refs


async def resolve_secret_env(
    secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef],
    resolver: SecretResolver,
    *,
    scope: dict[str, Any] | None = None,
) -> dict[str, ResolvedSecret]:
    """Resolve declared secret environment entries to injection-ready values.

    The returned values stay wrapped in ``ResolvedSecret`` (SecretStr) so
    callers unwrap them only at the last moment before injection and can feed
    the same values into a ``SecretRedactor`` for output scrubbing.
    """

    validate_secret_resolver(resolver, "resolver")
    resolved: dict[str, ResolvedSecret] = {}
    for name, ref in secret_env_refs(secret_env).items():
        secret = await resolver.resolve(ref, scope=scope)
        if type(secret) is not ResolvedSecret:
            raise TypeError("Secret resolvers must return ResolvedSecret instances.")
        resolved[name] = secret
    return resolved
