from __future__ import annotations

import os


def resolve_api_key(
    *,
    api_key: str | None,
    env_var: str,
    provider_name: str,
    missing_hint: str,
) -> str:
    """Resolve an explicit API key or environment fallback with provider-specific guidance."""
    if api_key is not None and type(api_key) is not str:
        raise TypeError("api_key must be a string.")
    resolved = api_key if api_key is not None else os.environ.get(env_var, "")
    if not resolved.strip():
        raise ValueError(f"{provider_name} requires an API key: {missing_hint}")
    return resolved
