"""Declarative evidence for Cayu's maintained public-agent service boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

PUBLIC_SERVICE_MANIFEST_SCHEMA_VERSION = "2"

# Names of the settings (environment variables, for example) a deployment provides so
# a fail-closed placeholder access policy becomes configured authentication. Names
# only: never their values.
AccessSettingNames = Annotated[
    tuple[
        Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$", max_length=128)],
        ...,
    ],
    Field(max_length=16),
]

# Why a maintained factory chose a fail-closed placeholder: its settings are unset
# (a fresh checkout), present but malformed, or reusing a credential another access
# policy already accepts, such as an operator token that is also a product token.
PlaceholderReason = Literal["unset", "malformed", "reused"]


class ServiceMode(StrEnum):
    """The exposure posture requested from a maintained service factory."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class ServiceIdentityStoreKind(StrEnum):
    """Whether application-owned public/private identity state survives a process."""

    DEVELOPMENT = "development"
    DURABLE = "durable"


class RuntimeStoreDurability(StrEnum):
    """Durability evidence declared by a Cayu runtime store implementation."""

    DEVELOPMENT = "development"
    DURABLE = "durable"
    READ_ONLY = "read_only"
    UNVERIFIED = "unverified"


class PublicServiceManifest(BaseModel):
    """Content-free facts emitted by Cayu's maintained service assembler.

    This does not claim that arbitrary host routes are authorized correctly. It
    describes only the application assembled by ``create_agent_service``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["2"] = PUBLIC_SERVICE_MANIFEST_SCHEMA_VERSION
    mode: ServiceMode
    product_access: Literal["authenticated", "development", "placeholder"]
    operator_access: Literal["authenticated", "open", "placeholder"]
    # The settings a placeholder product or operator access policy names as missing.
    product_access_settings: AccessSettingNames = ()
    operator_access_settings: AccessSettingNames = ()
    # Why each placeholder was chosen; None when that access policy is not a placeholder.
    product_access_reason: PlaceholderReason | None = None
    operator_access_reason: PlaceholderReason | None = None
    identity_store: ServiceIdentityStoreKind
    runtime_session_store: RuntimeStoreDurability
    runtime_task_store: RuntimeStoreDurability | Literal["missing"]
    product_api_path: str
    control_plane_path: str
    product_docs: Literal["disabled"] = "disabled"
    product_cors: Literal["disabled"] = "disabled"
    host_routing: Literal["maintained"] = "maintained"

    @model_validator(mode="after")
    def _reasons_belong_to_placeholders(self) -> PublicServiceManifest:
        if self.product_access_reason is not None and self.product_access != "placeholder":
            raise ValueError("product_access_reason applies only to placeholder product access.")
        if self.operator_access_reason is not None and self.operator_access != "placeholder":
            raise ValueError("operator_access_reason applies only to placeholder operator access.")
        return self
