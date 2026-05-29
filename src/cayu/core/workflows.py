from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from pydantic import BaseModel, ConfigDict, Field

from cayu.core.events import Event


class WorkflowSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Workflow(ABC):
    """Deterministic or agent-assisted multi-step orchestration."""

    spec: WorkflowSpec

    @abstractmethod
    async def run(self, session_id: str) -> AsyncIterator[Event]:
        """Run the workflow and stream structured events."""
