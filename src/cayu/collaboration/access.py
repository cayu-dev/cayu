"""Trusted application access policy, separate from participant request material."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from cayu.collaboration._contracts import ContractValue, Identifier
from cayu.collaboration.participants import (
    CollaborationBootstrap,
    ParticipantConfiguration,
    ParticipantRef,
)

ParticipantAction = Literal[
    "discover",
    "inspect",
    "readback",
    "create",
    "configure",
    "alias",
    "namespace_inspect",
    "namespace_seal",
    "namespace_rotate",
    "namespace_retire",
    "namespace_prune",
    "participant_lifecycle",
    "obligations",
    "request_accept",
    "request_readback",
    "request_control",
]


class CollaborationAccessDenied(PermissionError):
    """The trusted access boundary did not authorize this operation."""


class CollaborationAccessContext(ContractValue):
    """Supplied by trusted SDK host code, never taken from a model/JSON request."""

    principal: Identifier


class CollaborationAccessGrant(ContractValue):
    """Policy-owned result. None means this entire scope; empty means no participants."""

    application_scope: Identifier
    participants: tuple[ParticipantRef, ...] | None = Field(default=(), max_length=64)


class CollaborationAccessPolicy(ABC):
    @abstractmethod
    def authorize(
        self,
        context: CollaborationAccessContext,
        *,
        application_scope: str,
        action: ParticipantAction,
    ) -> CollaborationAccessGrant:
        """Resolve application-owned permissions without external side effects."""


@dataclass(frozen=True)
class CollaborationRegistration:
    bootstrap: CollaborationBootstrap
    access_policy: CollaborationAccessPolicy
    configurations: tuple[ParticipantConfiguration, ...]
