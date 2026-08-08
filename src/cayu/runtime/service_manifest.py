"""Declarative evidence for Cayu's maintained public-agent service boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

PUBLIC_SERVICE_MANIFEST_SCHEMA_VERSION = "2"


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
    identity_store: ServiceIdentityStoreKind
    runtime_session_store: RuntimeStoreDurability
    runtime_task_store: RuntimeStoreDurability | Literal["missing"]
    product_api_path: str
    control_plane_path: str
    product_docs: Literal["disabled"] = "disabled"
    product_cors: Literal["disabled"] = "disabled"
    host_routing: Literal["maintained"] = "maintained"
