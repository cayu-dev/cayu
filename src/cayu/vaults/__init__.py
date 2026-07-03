"""Vault contracts."""

from cayu.vaults.base import (
    ResolvedSecret,
    SecretEnv,
    SecretNotFound,
    SecretRef,
    SecretResolver,
    Vault,
    VaultError,
    copy_resolved_secret,
    copy_secret_env,
    copy_secret_ref,
    resolve_secret_env,
    secret_env_refs,
    validate_secret_resolver,
)
from cayu.vaults.composite import ChainVault, RoutedVault
from cayu.vaults.local_env import LocalEnvVault
from cayu.vaults.redaction import REDACTED_SECRET, SecretRedactor
from cayu.vaults.static import StaticVault

__all__ = [
    "REDACTED_SECRET",
    "ChainVault",
    "LocalEnvVault",
    "ResolvedSecret",
    "RoutedVault",
    "SecretEnv",
    "SecretNotFound",
    "SecretRedactor",
    "SecretRef",
    "SecretResolver",
    "StaticVault",
    "Vault",
    "VaultError",
    "copy_resolved_secret",
    "copy_secret_env",
    "copy_secret_ref",
    "resolve_secret_env",
    "secret_env_refs",
    "validate_secret_resolver",
]
