"""Typed provider-hosted execution authority."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}\Z)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)


class HostedToolCapabilityError(ValueError):
    """A provider target cannot honor admitted provider-hosted authority."""


class OpenAIWebSearch(BaseModel):
    """Registration-time authority for OpenAI's hosted Responses web search."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    type: Literal["openai_web_search"] = "openai_web_search"
    search_context_size: Literal["low", "medium", "high"] = "medium"
    external_web_access: StrictBool = True
    allowed_domains: tuple[str, ...] = Field(default=(), max_length=100)
    blocked_domains: tuple[str, ...] = Field(default=(), max_length=100)
    return_token_budget: Literal["default", "unlimited"] = "default"
    include_sources: StrictBool = True

    @field_validator("allowed_domains", "blocked_domains", mode="before")
    @classmethod
    def validate_domains(cls, value: object, info) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{info.field_name} must be a sequence of domain names.")
        domains = tuple(value)
        normalized: list[str] = []
        for domain in domains:
            if type(domain) is not str:
                raise ValueError(f"{info.field_name} entries must be strings.")
            candidate = domain.strip().lower().removesuffix(".")
            if not _DOMAIN_RE.fullmatch(candidate):
                raise ValueError(
                    f"{info.field_name} entries must be bare ASCII domain names without a scheme."
                )
            if candidate in normalized:
                raise ValueError(f"{info.field_name} must not contain duplicate domains.")
            normalized.append(candidate)
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_filters(self) -> OpenAIWebSearch:
        overlap = set(self.allowed_domains).intersection(self.blocked_domains)
        if overlap:
            raise ValueError(
                "OpenAI web search domains cannot be both allowed and blocked: "
                + ", ".join(sorted(overlap))
            )
        return self


def copy_openai_web_search(value: OpenAIWebSearch) -> OpenAIWebSearch:
    """Revalidate caller-owned hosted authority at a trust boundary."""

    if type(value) is not OpenAIWebSearch:
        raise TypeError("Hosted tools must be OpenAIWebSearch instances.")
    return OpenAIWebSearch.model_validate(value.model_dump(mode="python"))
