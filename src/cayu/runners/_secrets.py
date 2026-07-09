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

from cayu.credentials import (
    CredentialMode,
    CredentialModeInput,
    is_agent_readable,
    normalize_credential_mode,
)
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


def normalize_runner_secret_env(
    secret_env: Sequence[SecretEnv] | Mapping[str, SecretRef],
    secret_resolver: SecretResolver | None,
    *,
    credential_mode: CredentialModeInput = CredentialMode.RAW_ENV,
    allow_raw_secret_env: bool = True,
) -> tuple[dict[str, SecretRef], SecretResolver | None]:
    """Validate a runner's declared secret env, resolver, and credential mode.

    ``secret_env`` is raw injection (``raw_env``): the value is readable by the
    sandbox process. It is refused for any non-agent-readable credential mode
    (``trusted_tool`` or ``virtual_egress``) or when ``allow_raw_secret_env`` is
    opted out — the feature fails closed rather than injecting a raw value.
    """

    mode = normalize_credential_mode(credential_mode)
    refs = secret_env_refs(secret_env)
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


def merge_secret_env_values(
    env: dict[str, str],
    resolved: Mapping[str, ResolvedSecret],
) -> dict[str, str]:
    """Merge resolved secret values into a per-call env, rejecting collisions.

    A per-call ``env`` entry silently shadowing a declared secret (or the
    reverse) is ambiguous, so collisions fail closed.
    """

    merged = dict(env)
    for name, secret in resolved.items():
        if name in merged:
            raise ValueError(f"Runner env key collides with declared secret_env: {name}")
        merged[name] = secret.value.get_secret_value()
    return merged


def redact_exec_result(
    result: ExecResult,
    resolved: Mapping[str, ResolvedSecret],
) -> ExecResult:
    """Scrub resolved secret values from an ExecResult's captured output."""

    if not resolved:
        return result
    redactor = SecretRedactor(tuple(resolved.values()))
    if not redactor.has_values:
        return result
    return result.model_copy(
        update={
            "stdout": redactor.redact_text(result.stdout),
            "stderr": redactor.redact_text(result.stderr),
        }
    )


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

    if not environment:
        yield None
        return
    fd, path = tempfile.mkstemp(prefix="cayu-runner-env-")
    try:
        with os.fdopen(fd, "w") as handle:
            for key, value in environment.items():
                if "\n" in key or "=" in key or "\n" in value:
                    raise ValueError(
                        f"Runner env var {key!r} cannot be passed via env-file: names "
                        "cannot contain '=' and neither names nor values may contain newlines."
                    )
                handle.write(f"{key}={value}\n")
        yield path
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(path)
