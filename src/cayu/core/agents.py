from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.core.tools import Tool


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("name", "model")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)


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
