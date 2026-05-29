from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from cayu.core.events import Event
from cayu.core.messages import Message


class ModelRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[Message]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class ModelStreamEvent(BaseModel):
    """Provider-native stream event.

    Provider adapters may expose this lower-level shape while normalizing SDK
    responses. Runtime code must convert these events into framework `Event`
    records before persisting, dashboarding, or forwarding them.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    delta: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ModelProvider(ABC):
    """Normalizes provider-specific model streams."""

    name: str

    @abstractmethod
    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        """Stream model events for one request."""

    @abstractmethod
    def to_event(
        self,
        stream_event: ModelStreamEvent,
        *,
        session_id: str,
        agent_name: str | None = None,
    ) -> Event:
        """Convert a provider-native stream event into a framework event."""
