from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.events import Event
from cayu.core.messages import Message
from cayu.core.thinking import ThinkingConfig
from cayu.core.tools import Tool


class AgentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    # Registered provider this agent should run on. Falls back to the app's
    # default provider when unset; a RunRequest.provider_name overrides both.
    provider_name: str | None = None
    system_prompt: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provider_options: dict[str, Any] = Field(default_factory=dict)
    thinking: ThinkingConfig | None = None

    @field_validator("metadata", "provider_options", mode="before")
    @classmethod
    def copy_json_mapping(cls, value: dict[str, Any], info) -> dict[str, Any]:
        return copy_json_value(value, info.field_name)

    @field_validator("name", "model")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("provider_name")
    @classmethod
    def validate_optional_provider_name(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_clean_nonblank(value, info.field_name)


class Agent(ABC):
    """Base contract for agents.

    Concrete implementations will turn messages into an event stream, using
    providers, tools, memory, and runtime services.
    """

    spec: AgentSpec
    tools: list[Tool]

    @abstractmethod
    def run(self, messages: list[Message]) -> AsyncIterator[Event]:
        """Run the agent and return a stream of structured events.

        Declared non-async on purpose: implementations are expected to be
        async generators (``async def run(...): yield ...``), which are plain
        callables returning an ``AsyncIterator``. Keeping the abstract method
        non-async gives every implementation the same calling convention —
        ``async for event in agent.run(messages)`` — instead of some callers
        needing ``await agent.run(...)`` first.
        """
