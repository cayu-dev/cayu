from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

from cayu import Message
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

CapabilityState = Literal["supported", "unsupported", "not_applicable"]
ProviderScenario = Literal[
    "text",
    "tool_round_trip",
    "typed_error",
    "context_overflow",
    "malformed",
    "malformed_terminal",
    "malformed_usage",
    "unfinished",
    "unfinished_reasoning",
    "cancellation",
    "idle_timeout",
    "close",
    "token_counting",
    "native_structured_output",
    "attachments",
    "reasoning",
    "provider_cache_observation",
]
MAX_CAPABILITY_REASON_LENGTH = 240


class ProviderConformanceFailure(AssertionError):
    """A shared scenario failed with adapter- and capability-level context."""


def require_conformance(
    condition: bool,
    *,
    registration: ProviderConformanceRegistration,
    scenario: ProviderScenario,
    observed: object,
    capability: str | None = None,
) -> None:
    if condition:
        return
    capability_detail = "" if capability is None else f" capability={capability}"
    raise ProviderConformanceFailure(
        "ModelProvider conformance failed: "
        f"adapter={registration.name} scenario={scenario}{capability_detail} "
        f"observed={observed!r}"
    )


@dataclass(frozen=True)
class CapabilityClaim:
    state: CapabilityState
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state == "supported" and self.reason is not None:
            raise ValueError("Supported provider capabilities cannot define a skip reason.")
        if self.state != "supported" and not (self.reason and self.reason.strip()):
            raise ValueError("Unsupported and not-applicable capabilities require a reason.")
        if self.reason is not None and len(self.reason) > MAX_CAPABILITY_REASON_LENGTH:
            raise ValueError(
                "Provider capability reasons must be at most "
                f"{MAX_CAPABILITY_REASON_LENGTH} characters."
            )

    @classmethod
    def supported(cls) -> CapabilityClaim:
        return cls("supported")

    @classmethod
    def unsupported(cls, reason: str) -> CapabilityClaim:
        return cls("unsupported", reason)

    @classmethod
    def not_applicable(cls, reason: str) -> CapabilityClaim:
        return cls("not_applicable", reason)


@dataclass(frozen=True)
class ProviderCapabilities:
    token_counting: CapabilityClaim
    native_structured_output: CapabilityClaim
    attachments: CapabilityClaim
    reasoning: CapabilityClaim
    provider_cache_observation: CapabilityClaim


@dataclass
class ProviderHarness:
    provider: ModelProvider
    model: str
    close: Callable[[], Awaitable[None]] | None = None
    wait_started: Callable[[], Awaitable[None]] | None = None
    wait_stopped: Callable[[], Awaitable[None]] | None = None
    is_closed: Callable[[], bool] | None = None

    async def collect(self, request: ModelRequest | None = None) -> list[ModelStreamEvent]:
        if request is None:
            request = ModelRequest(
                model=self.model,
                messages=[Message.text("user", "Say hello.")],
            )
        return [event async for event in self.provider.stream(request)]

    async def aclose(self) -> None:
        if self.close is not None:
            await self.close()
            return
        provider_close = getattr(self.provider, "aclose", None)
        if provider_close is not None:
            await provider_close()


ProviderFactory = Callable[[ProviderScenario], Awaitable[ProviderHarness]]


@dataclass(frozen=True)
class ProviderConformanceRegistration:
    name: str
    provider_type: type[ModelProvider]
    factory: ProviderFactory
    capabilities: ProviderCapabilities
    error_provider: str | None = None
    reports_model_identity: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Provider conformance registration name must be nonblank.")
        if not issubclass(self.provider_type, ModelProvider):
            raise TypeError("Provider conformance registration type must implement ModelProvider.")
        if self.error_provider is not None and not self.error_provider.strip():
            raise ValueError("Provider conformance error provider must be nonblank.")
        if type(self.reports_model_identity) is not bool:
            raise TypeError("Provider conformance model-identity claim must be a boolean.")

    @property
    def expected_error_provider(self) -> str:
        return self.error_provider or self.name
