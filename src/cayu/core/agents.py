from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.core.tools import Tool


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Agent(ABC):
    """Base contract for agents.

    Concrete implementations will turn messages into an event stream, using
    providers, tools, memory, and runtime services.
    """

    spec: AgentSpec
    tools: list[Tool]

    @abstractmethod
    async def run(self, messages: list[Message]) -> AsyncIterator[Event]:
        """Run the agent and stream structured events."""
