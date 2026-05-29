from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import (
    copy_json_value,
    require_finite,
    require_nonblank,
)


class KnowledgeItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @field_validator("id", "text")
    @classmethod
    def validate_nonblank_fields(cls, value: str, info) -> str:
        return require_nonblank(value, info.field_name)

    @field_validator("source")
    @classmethod
    def validate_nonblank_source(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return require_nonblank(value, info.field_name)


class KnowledgeHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: KnowledgeItem
    score: float | None = None
    reason: str | None = None

    @field_validator("item")
    @classmethod
    def copy_item(cls, value):
        return copy_knowledge_item(value)

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, value, info):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"`{info.field_name}` must be a number.")
        return require_finite(value, info.field_name)


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


def copy_knowledge_item(item: KnowledgeItem) -> KnowledgeItem:
    if type(item) is not KnowledgeItem:
        raise TypeError("Knowledge hits must contain KnowledgeItem instances.")
    return KnowledgeItem(
        id=item.id,
        text=item.text,
        source=item.source,
        metadata=copy_json_value(item.metadata, "metadata"),
    )
