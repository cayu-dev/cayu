from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: KnowledgeItem
    score: float | None = None
    reason: str | None = None


class KnowledgeStore(ABC):
    """Searchable memory/knowledge contract."""

    @abstractmethod
    async def upsert(self, item: KnowledgeItem) -> None:
        """Insert or update one knowledge item."""

    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[KnowledgeHit]:
        """Search memory/knowledge."""
