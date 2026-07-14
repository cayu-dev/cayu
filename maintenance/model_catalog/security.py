"""Trust boundaries for browser-backed catalog maintenance."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from cayu.core.tools import ToolContext
from maintenance.model_catalog.policy import OFFICIAL_HOSTS

PROVIDER_METADATA_KEY = "model_catalog_provider"


def provider_from_context(ctx: ToolContext) -> str:
    provider_name = ctx.metadata.get(PROVIDER_METADATA_KEY)
    if not isinstance(provider_name, str) or provider_name not in OFFICIAL_HOSTS:
        raise ValueError("catalog browser tools require a supported provider in trusted metadata")
    return provider_name


def allowed_hosts(provider_name: str) -> frozenset[str]:
    try:
        return OFFICIAL_HOSTS[provider_name]
    except KeyError as exc:
        raise ValueError(f"unsupported catalog provider: {provider_name}") from exc


def validate_official_url(url: str, *, provider_name: str) -> str:
    """Return a normalized official HTTPS URL or reject it before browser navigation."""

    if not isinstance(url, str):
        raise ValueError("browser URL must be a string")
    parsed = urlsplit(url.strip())
    hostname = parsed.hostname.lower() if parsed.hostname else None
    if parsed.scheme.lower() != "https":
        raise ValueError("catalog browser URLs must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("catalog browser URLs must not contain credentials")
    if parsed.port not in {None, 443}:
        raise ValueError("catalog browser URLs must use the default HTTPS port")
    if hostname not in allowed_hosts(provider_name):
        raise ValueError(f"URL host is not official for {provider_name}: {hostname or '<missing>'}")
    netloc = hostname if parsed.port is None else f"{hostname}:443"
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))
