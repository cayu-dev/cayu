from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank, require_nonblank


class TextEmbeddingRequest(BaseModel):
    """Provider-neutral request to embed one or more text inputs."""

    model_config = ConfigDict(extra="forbid")

    model: str
    texts: list[str]
    dimensions: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: list[str], info) -> list[str]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return [require_nonblank(text, f"{info.field_name}[{index}]") for index, text in enumerate(value)]

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, value: int | None, info) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value <= 0:
            raise ValueError(f"`{info.field_name}` must be greater than 0.")
        return value

    @field_validator("options", mode="before")
    @classmethod
    def copy_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_json_value(value, "options")
        if type(copied) is not dict:
            raise ValueError("`options` must be a dictionary.")
        return copied


class TextEmbedding(BaseModel):
    """One embedding vector returned by a provider."""

    model_config = ConfigDict(extra="forbid")

    index: int
    vector: list[float]

    @field_validator("index")
    @classmethod
    def validate_index(cls, value: int, info) -> int:
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be greater than or equal to 0.")
        return value

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: list[float], info) -> list[float]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        result: list[float] = []
        for index, item in enumerate(value):
            if isinstance(item, bool) or not isinstance(item, int | float):
                raise ValueError(f"`{info.field_name}[{index}]` must be a number.")
            number = float(item)
            if not isfinite(number):
                raise ValueError(f"`{info.field_name}[{index}]` must be finite.")
            result.append(number)
        return result


class TextEmbeddingUsage(BaseModel):
    """Provider-reported token usage for an embedding request when available."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("input_tokens", "total_tokens", mode="before")
    @classmethod
    def validate_token_count(cls, value: int | None, info) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or type(value) is not int:
            raise ValueError(f"`{info.field_name}` must be an integer.")
        if value < 0:
            raise ValueError(f"`{info.field_name}` must be greater than or equal to 0.")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_json_value(value, "metadata")
        if type(copied) is not dict:
            raise ValueError("`metadata` must be a dictionary.")
        return copied


class TextEmbeddingResult(BaseModel):
    """Provider-neutral embedding response."""

    model_config = ConfigDict(extra="forbid")

    model: str
    embeddings: list[TextEmbedding]
    usage: TextEmbeddingUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        copied = copy_json_value(value, "metadata")
        if type(copied) is not dict:
            raise ValueError("`metadata` must be a dictionary.")
        return copied

    @model_validator(mode="after")
    def validate_embeddings(self) -> TextEmbeddingResult:
        if not self.embeddings:
            raise ValueError("`embeddings` cannot be empty.")
        indexes = [embedding.index for embedding in self.embeddings]
        if len(indexes) != len(set(indexes)):
            raise ValueError("Embedding indexes must be unique.")
        dimensions = {len(embedding.vector) for embedding in self.embeddings}
        if len(dimensions) != 1:
            raise ValueError("Embedding vectors must have the same dimension.")
        return self


class TextEmbeddingProvider(ABC):
    """Provider-neutral text embedding contract."""

    name: str

    @abstractmethod
    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        """Embed text inputs and return one vector for each input."""


def copy_text_embedding_request(request: TextEmbeddingRequest) -> TextEmbeddingRequest:
    if type(request) is not TextEmbeddingRequest:
        raise TypeError("TextEmbeddingRequest instances must not be subclasses.")
    return TextEmbeddingRequest(
        model=request.model,
        texts=list(request.texts),
        dimensions=request.dimensions,
        options=copy_json_value(request.options, "options"),
    )


def copy_text_embedding_result(result: TextEmbeddingResult) -> TextEmbeddingResult:
    if type(result) is not TextEmbeddingResult:
        raise TypeError("TextEmbeddingResult instances must not be subclasses.")
    usage = None
    if result.usage is not None:
        usage = TextEmbeddingUsage(
            input_tokens=result.usage.input_tokens,
            total_tokens=result.usage.total_tokens,
            metadata=copy_json_value(result.usage.metadata, "usage.metadata"),
        )
    return TextEmbeddingResult(
        model=result.model,
        embeddings=[
            TextEmbedding(index=embedding.index, vector=list(embedding.vector))
            for embedding in result.embeddings
        ],
        usage=usage,
        metadata=copy_json_value(result.metadata, "metadata"),
    )
