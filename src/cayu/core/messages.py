from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"] = "text"
    text: str


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: list[TextPart] = Field(default_factory=list)

    @classmethod
    def text(cls, role: MessageRole | str, text: str) -> "Message":
        return cls(role=MessageRole(role), content=[TextPart(text=text)])
