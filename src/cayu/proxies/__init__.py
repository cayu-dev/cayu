"""Credential proxy contracts."""

from cayu.proxies.base import (
    CredentialProxy,
    ProxyAuthorizationResult,
    copy_proxy_authorization_result,
)
from cayu.proxies.passthrough import PassthroughProxy

__all__ = [
    "CredentialProxy",
    "PassthroughProxy",
    "ProxyAuthorizationResult",
    "copy_proxy_authorization_result",
]
