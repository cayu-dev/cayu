"""Versioned application source registration for runtime-owned automatic recall."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from cayu._validation import require_durable_clean_nonblank
from cayu.context.base import ContextRequest
from cayu.memory.recall import RecallSituation, RecallSource, RecallSourceResult
from cayu.storage.memory import KnowledgeAccessScope


@dataclass(frozen=True, kw_only=True)
class AutomaticRecallSourceContext:
    """Read-only construction scope; not a mutable model/transcript request.

    Store handles remain trusted application capabilities. The factory must not
    widen the supplied access scope or mutate those stores.
    """

    session_id: str
    interaction_id: str | None
    environment_name: str | None
    knowledge_namespace: str
    knowledge_access_scope: KnowledgeAccessScope | None
    knowledge_store: Any
    session_store: Any


class AutomaticRecallSourceDescriptor(BaseModel):
    """Durable identity and declared lanes of a trusted read-only extension.

    The application must change configuration_version whenever source behavior
    or access configuration changes. Credentials and callable identities do not
    belong in this descriptor, which is persisted in recall evidence.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    name: str
    channel_names: tuple[str, ...]
    configuration_version: str
    required: bool = True
    candidate_limit: int = 20
    continuation_channels: tuple[str, ...] = ()

    @field_validator("name", "configuration_version", mode="before")
    @classmethod
    def validate_name(cls, value: Any, info) -> str:
        value = require_durable_clean_nonblank(value, info.field_name)
        if len(value.encode("utf-8")) > 256:
            raise ValueError(f"`{info.field_name}` exceeds 256 UTF-8 bytes.")
        return value

    @field_validator("channel_names", "continuation_channels", mode="before")
    @classmethod
    def validate_channels(cls, value: Any, info) -> tuple[str, ...]:
        if type(value) not in (tuple, list) or len(value) > 100:
            raise ValueError(f"`{info.field_name}` must contain at most 100 channel names.")
        channels = tuple(require_durable_clean_nonblank(item, info.field_name) for item in value)
        if any(len(item.encode("utf-8")) > 256 for item in channels):
            raise ValueError("A channel name exceeds 256 UTF-8 bytes.")
        if len(set(channels)) != len(channels):
            raise ValueError("Channel names cannot repeat.")
        return channels

    @field_validator("required", mode="before")
    @classmethod
    def validate_required(cls, value: Any) -> bool:
        if type(value) is not bool:
            raise ValueError("`required` must be a boolean.")
        return value

    @field_validator("candidate_limit", mode="before")
    @classmethod
    def validate_limit(cls, value: Any) -> int:
        if type(value) is not int or not 1 <= value <= 100:
            raise ValueError("`candidate_limit` must be between 1 and 100.")
        return value

    @model_validator(mode="after")
    def validate_membership(self) -> AutomaticRecallSourceDescriptor:
        if not self.channel_names:
            raise ValueError("A source must declare at least one channel.")
        if not set(self.continuation_channels).issubset(self.channel_names):
            raise ValueError("Continuation channels must belong to the source.")
        return self


@dataclass(frozen=True, kw_only=True)
class AutomaticRecallSourceRegistration:
    """Construct a request-scoped source inside RecallEngine's work deadline.

    Factories and sources are trusted application code, not sandboxed plugins.
    They must cooperate with cancellation, avoid blocking the event loop, honor
    the request/situation access scope, and own any resources they acquire.
    Factories must not perform mutations or allocate resources requiring cleanup
    after retrieve returns. The runtime owns receipts and delivery evidence.
    """

    descriptor: AutomaticRecallSourceDescriptor
    factory: Callable[[AutomaticRecallSourceContext], Awaitable[RecallSource]]

    def __post_init__(self) -> None:
        if type(self.descriptor) is not AutomaticRecallSourceDescriptor:
            raise TypeError("descriptor must be an AutomaticRecallSourceDescriptor.")
        object.__setattr__(
            self,
            "descriptor",
            AutomaticRecallSourceDescriptor.model_validate(
                self.descriptor.model_dump(mode="python")
            ),
        )
        if not callable(self.factory):
            raise TypeError("factory must be callable.")


class _FactoryRecallSource(RecallSource):
    """Keep lazy construction within the engine's normal timeout/failure path."""

    def __init__(
        self, registration: AutomaticRecallSourceRegistration, request: ContextRequest | None
    ) -> None:
        descriptor = registration.descriptor
        self.name = descriptor.name
        self.channel_names = descriptor.channel_names
        self.continuation_channels = descriptor.continuation_channels
        super().__init__(required=descriptor.required, candidate_limit=descriptor.candidate_limit)
        self._registration = registration
        self._request = request

    async def retrieve(self, situation: RecallSituation) -> RecallSourceResult:
        if self._request is None:
            raise RuntimeError("A declaration-only recall source cannot retrieve.")
        context = AutomaticRecallSourceContext(
            session_id=self._request.session.id,
            interaction_id=self._request.interaction_id,
            environment_name=self._request.environment_name,
            knowledge_namespace=situation.knowledge_namespace,
            knowledge_access_scope=situation.knowledge_access_scope,
            knowledge_store=self._request.knowledge_store,
            session_store=self._request.session_store,
        )
        source = await self._registration.factory(context)
        if not isinstance(source, RecallSource):
            raise TypeError("An automatic recall factory must return a RecallSource.")
        if (
            source.name != self.name
            or type(source.channel_names) is not tuple
            or source.channel_names != self.channel_names
            or type(source.continuation_channels) is not tuple
            or source.continuation_channels != self.continuation_channels
            or source.required is not self.required
            or type(source.candidate_limit) is not int
            or source.candidate_limit != self.candidate_limit
        ):
            raise ValueError("An automatic recall source does not match its descriptor.")
        return await source.retrieve(situation)
