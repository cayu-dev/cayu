"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "AllowlistProxy": ("cayu.proxies.passthrough", "AllowlistProxy"),
    "CredentialProxy": ("cayu.proxies.base", "CredentialProxy"),
    "PassthroughProxy": ("cayu.proxies.passthrough", "PassthroughProxy"),
    "ProxyAuthorizationResult": ("cayu.proxies.base", "ProxyAuthorizationResult"),
    "copy_proxy_authorization_result": ("cayu.proxies.base", "copy_proxy_authorization_result"),
}

PUBLIC_NAMES = [
    "AllowlistProxy",
    "CredentialProxy",
    "PassthroughProxy",
    "ProxyAuthorizationResult",
    "copy_proxy_authorization_result",
]
