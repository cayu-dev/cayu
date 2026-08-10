from __future__ import annotations

from abc import ABC, abstractmethod
from math import isfinite
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    copy_durable_json_object,
    copy_json_value,
    require_durable_clean_nonblank,
    require_durable_nonblank,
)


class TextEmbeddingRequest(BaseModel):
    """Provider-neutral request to embed one or more text inputs."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model: str
    texts: list[str]
    dimensions: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, value: list[str], info) -> list[str]:
        if not value:
            raise ValueError(f"`{info.field_name}` cannot be empty.")
        return [
            require_durable_nonblank(text, f"{info.field_name}[{index}]")
            for index, text in enumerate(value)
        ]

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
        return copy_durable_json_object(value, "options")


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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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
        return copy_durable_json_object(value, "metadata")


class TextEmbeddingResult(BaseModel):
    """Provider-neutral embedding response."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    model: str
    embeddings: list[TextEmbedding]
    usage: TextEmbeddingUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("embeddings")
    @classmethod
    def copy_embeddings(cls, value: list[TextEmbedding]) -> list[TextEmbedding]:
        return [copy_text_embedding(embedding) for embedding in value]

    @field_validator("usage")
    @classmethod
    def copy_usage(
        cls,
        value: TextEmbeddingUsage | None,
    ) -> TextEmbeddingUsage | None:
        return copy_text_embedding_usage(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_durable_json_object(value, "metadata")

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


def copy_text_embedding(embedding: TextEmbedding) -> TextEmbedding:
    if type(embedding) is not TextEmbedding:
        raise TypeError("Embedding copies require TextEmbedding instances.")
    return TextEmbedding(index=embedding.index, vector=list(embedding.vector))


def copy_text_embedding_usage(
    usage: TextEmbeddingUsage | None,
) -> TextEmbeddingUsage | None:
    if usage is None:
        return None
    if type(usage) is not TextEmbeddingUsage:
        raise TypeError("Embedding usage copies require TextEmbeddingUsage instances.")
    return TextEmbeddingUsage(
        input_tokens=usage.input_tokens,
        total_tokens=usage.total_tokens,
        metadata=copy_json_value(usage.metadata, "usage.metadata"),
    )


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
    return TextEmbeddingResult(
        model=result.model,
        embeddings=[copy_text_embedding(embedding) for embedding in result.embeddings],
        usage=copy_text_embedding_usage(result.usage),
        metadata=copy_json_value(result.metadata, "metadata"),
    )
