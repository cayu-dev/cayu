"""Shared secret-injection seam for runner adapters.

Runners declare secrets as ``SecretEnv`` entries and resolve them through an
async ``SecretResolver`` (a ``Vault`` or ``CredentialProxy``) at exec time.
Raw values are unwrapped only at the injection point, never appear in
host-visible argv, and are scrubbed from ``ExecResult`` output before it
reaches model-visible context.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import cast

from cayu._validation import require_durable_text
from cayu.credentials import (
    CredentialMode,
    CredentialModeInput,
    is_agent_readable,
    normalize_credential_mode,
)
from cayu.runners._subprocess import runner_env_name_identity, validate_runner_env_name
from cayu.runners.base import ExecResult
from cayu.vaults import (
    ResolvedSecret,
    SecretEnv,
    SecretRedactor,
    SecretRef,
    SecretResolver,
    secret_env_refs,
    validate_secret_resolver,
)

DOCKER_ENV_FILE_MAX_LINE_BYTES = 64 * 1024 - 1


def normalize_runner_secret_env(
    secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef],
    secret_resolver: SecretResolver | None,
    *,
    credential_mode: CredentialModeInput = CredentialMode.RAW_ENV,
    allow_raw_secret_env: bool = True,
    case_sensitive_env_names: bool = True,
) -> tuple[dict[str, SecretRef], SecretResolver | None]:
    """Validate a runner's declared secret env, resolver, and credential mode.

    ``secret_env`` is raw injection (``raw_env``): the value is readable by the
    sandbox process. It is refused for any non-agent-readable credential mode
    (``trusted_tool`` or ``virtual_egress``) or when ``allow_raw_secret_env`` is
    opted out — the feature fails closed rather than injecting a raw value.
    """

    mode = normalize_credential_mode(credential_mode)
    owned_secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef]
    if isinstance(secret_env, Sequence) and not isinstance(secret_env, str | bytes):
        owned_entries = tuple(secret_env)
        for entry in owned_entries:
            if type(entry) is SecretEnv:
                validate_runner_env_name(entry.name, "Runner secret_env name")
        owned_secret_env = cast("Sequence[SecretEnv]", owned_entries)
    else:
        owned_secret_env = secret_env
    refs = {
        validate_runner_env_name(name, "Runner secret_env name"): ref
        for name, ref in secret_env_refs(owned_secret_env).items()
    }
    identities: set[str] = set()
    for name in refs:
        identity = runner_env_name_identity(
            name,
            case_sensitive=case_sensitive_env_names,
        )
        if identity in identities:
            raise ValueError("Runner secret_env contains duplicate environment variable names.")
        identities.add(identity)
    if secret_resolver is not None:
        validate_secret_resolver(secret_resolver)
    if refs and secret_resolver is None:
        raise ValueError("Runners with secret_env require a secret_resolver (Vault or proxy).")
    if refs and not is_agent_readable(mode):
        raise ValueError(
            "secret_env (raw injection) cannot be combined with "
            f"credential_mode={mode.value}; non-agent-readable credential modes must not "
            "receive a raw secret."
        )
    if refs and not allow_raw_secret_env:
        raise ValueError(
            "secret_env injects a raw, agent-readable secret; pass "
            "allow_raw_secret_env=True to acknowledge this on an untrusted runner."
        )
    return refs, secret_resolver


def validate_secret_env_collisions(
    env: Mapping[str, str],
    secret_env: Mapping[str, SecretRef] | Mapping[str, ResolvedSecret],
    *,
    case_sensitive: bool = True,
) -> None:
    """Reject ambiguous explicit/declared environment ownership without lookup."""

    explicit_names = {runner_env_name_identity(name, case_sensitive=case_sensitive) for name in env}
    secret_names = {
        runner_env_name_identity(name, case_sensitive=case_sensitive) for name in secret_env
    }
    if not explicit_names.isdisjoint(secret_names):
        raise ValueError("Runner env key collides with declared secret_env.")


def merge_secret_env_values(
    env: dict[str, str],
    resolved: Mapping[str, ResolvedSecret],
) -> dict[str, str]:
    """Merge resolved secret values into a per-call env, rejecting collisions.

    A per-call ``env`` entry silently shadowing a declared secret (or the
    reverse) is ambiguous, so collisions fail closed.
    """

    validate_secret_env_collisions(env, resolved)
    merged = dict(env)
    for name, secret in resolved.items():
        merged[name] = secret.value.get_secret_value()
    return merged


def redact_exec_result(
    result: ExecResult,
    resolved: Mapping[str, ResolvedSecret],
) -> ExecResult:
    """Scrub resolved secret values from an ExecResult's captured output."""

    redactor = resolved_secret_redactor(resolved)
    if not redactor.has_values:
        return result
    return result.model_copy(
        update={
            "stdout": redactor.redact_text(result.stdout),
            "stderr": redactor.redact_text(result.stderr),
        }
    )


def resolved_secret_redactor(
    resolved: Mapping[str, ResolvedSecret],
) -> SecretRedactor:
    """Build the invocation-scoped redactor without exposing its secret registry."""

    return SecretRedactor(tuple(resolved.values()))


@contextmanager
def runner_env_file(environment: Mapping[str, str]) -> Iterator[str | None]:
    """Write a runner's container env to a private temp file for ``--env-file``.

    Container env values (including the model-supplied ``env`` of a tool call) must never
    be merged into the runner CLI's OWN process environment: a prompt-injected agent could
    otherwise set ``DOCKER_HOST``/``LD_PRELOAD``/credential-helper vars and hijack the host
    CLI (connect to an attacker daemon, load a shared object, etc.). Passing them via
    ``--env-file`` keeps container env fully separate from the CLI's environment.

    The file is created ``0600`` (``mkstemp`` default) and unlinked on exit. Yields
    ``None`` when there is no env to pass.
    """

    validate_runner_env_file_environment(environment)
    if not environment:
        yield None
        return
    fd, path = tempfile.mkstemp(prefix="cayu-runner-env-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for key, value in environment.items():
                handle.write(f"{key}={value}\n")
        # The file now owns the transport representation. Do not keep the raw
        # mapping alive in this generator while the subprocess is awaited.
        environment = {}
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)


def validate_runner_env_file_environment(environment: Mapping[str, str]) -> None:
    """Validate Docker's line-oriented key/value transport without exposing content."""

    for key, value in environment.items():
        key = require_durable_text(key, "Runner env-file key")
        value = require_durable_text(value, "Runner env-file value")
        if (
            not key
            or key.startswith("#")
            or key.startswith("\ufeff")
            or "=" in key
            or " " in key
            or "\t" in key
            or "\n" in key
            or "\r" in key
        ):
            raise ValueError("Runner env-file key cannot be represented by Docker.")
        if "\n" in value or "\r" in value:
            raise ValueError("Runner env-file value cannot contain line breaks.")
        line_size = len(f"{key}={value}".encode())
        if line_size > DOCKER_ENV_FILE_MAX_LINE_BYTES:
            raise ValueError("Runner env-file line exceeds Docker's supported size.")
