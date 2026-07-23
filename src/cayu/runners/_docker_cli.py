"""Bounded environment for trusted host-side Docker CLI helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence

_DOCKER_CLI_ENV_KEYS = (
    "COMSPEC",
    "DOCKER_API_VERSION",
    "DOCKER_CERT_PATH",
    "DOCKER_CONFIG",
    "DOCKER_CONTEXT",
    "DOCKER_HOST",
    "DOCKER_TLS_VERIFY",
    "HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "LOGNAME",
    "NO_PROXY",
    "PATH",
    "PATHEXT",
    "SSH_AUTH_SOCK",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMDRIVE",
    "SYSTEMROOT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "USER",
    "USERPROFILE",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def normalize_docker_cli_env_allowlist(names: Sequence[str]) -> tuple[str, ...]:
    """Validate explicit host variables granted to trusted Docker CLI helpers."""

    if isinstance(names, str) or not isinstance(names, Sequence):
        raise TypeError("docker_cli_env_allowlist must be a sequence of environment names.")
    normalized: list[str] = []
    for name in names:
        if type(name) is not str or not name.strip():
            raise ValueError(
                "docker_cli_env_allowlist entries must be non-empty environment names."
            )
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


def docker_cli_env(extra_allowlist: Sequence[str] = ()) -> dict[str, str]:
    """Return the Docker operational env plus explicit trusted host grants."""

    names = (*_DOCKER_CLI_ENV_KEYS, *normalize_docker_cli_env_allowlist(extra_allowlist))
    return {name: os.environ[name] for name in names if name in os.environ}


__all__ = ["docker_cli_env", "normalize_docker_cli_env_allowlist"]
