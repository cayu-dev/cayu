from __future__ import annotations

from collections.abc import Awaitable, Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

from cayu.runners import Runner, RunnerCleanupPolicy, RunnerSystemExecutionMode

CapabilityState = Literal["supported", "unsupported", "not_applicable"]
MAX_CAPABILITY_REASON_LENGTH = 240


@dataclass(frozen=True)
class CapabilityClaim:
    state: CapabilityState
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.state == "supported" and self.reason is not None:
            raise ValueError("Supported runner capabilities cannot define a skip reason.")
        if self.state != "supported" and not (self.reason and self.reason.strip()):
            raise ValueError("Unsupported and not-applicable capabilities require a reason.")
        if self.reason is not None and len(self.reason) > MAX_CAPABILITY_REASON_LENGTH:
            raise ValueError(
                "Runner capability reasons must be at most "
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
class RunnerCapabilities:
    command_cleanup: CapabilityClaim
    sandbox_cleanup: CapabilityClaim
    no_cleanup: CapabilityClaim
    ambiguous_start: CapabilityClaim
    remote_protocol: CapabilityClaim
    suspend_resume: CapabilityClaim


@dataclass
class RunnerHarness:
    runner: Runner
    root: Path
    finalize: Callable[[], Awaitable[None]] | None = None
    system_execution_profiles: list[str] | None = None

    async def aclose(self) -> None:
        try:
            await self.runner.close()
        finally:
            if self.finalize is not None:
                await self.finalize()


@dataclass
class ConformanceEvidence:
    scenario: str
    adapter: str
    capability: str
    observed: object = "not recorded"
    cleanup_artifact: object = None

    @contextmanager
    def reporting(self) -> Generator[None, None, None]:
        try:
            yield
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            raise AssertionError(
                f"scenario={self.scenario} adapter={self.adapter} "
                f"capability={self.capability} "
                f"observed={_concise_repr(self.observed)} "
                f"cleanup_artifact={_concise_repr(self.cleanup_artifact)}: {exc}"
            ) from exc


RunnerFactory = Callable[[Path, pytest.MonkeyPatch], Awaitable[RunnerHarness]]
RunnerProbe = Callable[[Path, pytest.MonkeyPatch], Awaitable[None]]
RemoteProtocolProbe = Callable[
    [Path, pytest.MonkeyPatch],
    Awaitable[dict[str, bool]],
]
CleanupRunnerFactory = Callable[
    [Path, pytest.MonkeyPatch, RunnerCleanupPolicy],
    Awaitable[RunnerHarness],
]


@dataclass(frozen=True)
class RunnerConformanceRegistration:
    name: str
    runner_type: type[Runner]
    factory: RunnerFactory
    capabilities: RunnerCapabilities
    system_execution_mode: RunnerSystemExecutionMode
    cleanup_factory: CleanupRunnerFactory | None = None
    ambiguous_start_probe: RunnerProbe | None = None
    remote_protocol_probe: RemoteProtocolProbe | None = None
    suspend_resume_probe: RunnerProbe | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Runner conformance registration name must be nonblank.")
        if not issubclass(self.runner_type, Runner):
            raise TypeError("Runner conformance registration type must implement Runner.")
        if self.system_execution_mode not in {"shared", "separate"}:
            raise ValueError("Runner system execution mode must be shared or separate.")
        cleanup_claims = (
            self.capabilities.command_cleanup,
            self.capabilities.sandbox_cleanup,
            self.capabilities.no_cleanup,
        )
        if (
            any(claim.state == "supported" for claim in cleanup_claims)
            and self.cleanup_factory is None
        ):
            raise ValueError("Registrations claiming cleanup support require a cleanup factory.")
        for claim, probe, capability in (
            (self.capabilities.ambiguous_start, self.ambiguous_start_probe, "ambiguous start"),
            (self.capabilities.remote_protocol, self.remote_protocol_probe, "remote protocol"),
            (self.capabilities.suspend_resume, self.suspend_resume_probe, "suspend/resume"),
        ):
            if claim.state == "supported" and probe is None:
                raise ValueError(
                    f"Registrations claiming {capability} support require a scenario probe."
                )


def _concise_repr(value: object, *, limit: int = 1200) -> str:
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}...<truncated>"
