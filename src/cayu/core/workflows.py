from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import copy_json_value, require_nonblank
from cayu.core.events import Event


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("name")
    @classmethod
    def validate_nonblank_name(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)


class Workflow(ABC):
    """Deterministic or agent-assisted multi-step orchestration."""

    spec: WorkflowSpec

    @abstractmethod
    async def run(self, session_id: str) -> AsyncIterator[Event]:
        """Run the workflow and stream structured events."""
