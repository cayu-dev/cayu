from __future__ import annotations

from collections.abc import Iterable, Mapping
from importlib.metadata import version
from typing import Any

from cayu.runtime.execution_profiles import build_execution_profile_identity
from cayu.runtime.sessions import SessionIdentity


def profiled_session_identity(
    *,
    provider_name: str,
    model: str,
    durable_system_prompt: str | None = None,
    direct_tools: Iterable[Mapping[str, Any]] = (),
) -> SessionIdentity:
    """Build the identity used by low-level tests that later enter public resume."""

    runtime_version = version("cayu")
    return SessionIdentity(
        provider_name=provider_name,
        model=model,
        runtime_name="cayu",
        runtime_version=runtime_version,
        execution_profile=build_execution_profile_identity(
            runtime_name="cayu",
            runtime_version=runtime_version,
            provider_name=provider_name,
            model=model,
            durable_system_prompt=durable_system_prompt,
            direct_tools=direct_tools,
        ),
    )
