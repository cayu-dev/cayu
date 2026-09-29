"""Experimental extension seams: adapter identity registry and public re-exports."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

import cayu.extensions.egress as extension_egress
import cayu.extensions.runners as extension_runners
from cayu.runners import _adapter_identity
from cayu.runners._cleanup import sanitize_runner_artifacts
from cayu.runners._diagnostics import trusted_runner_exception_type_name
from cayu.runners.base import (
    DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ExecCommand,
    ExecResult,
    Runner,
    RunnerExecutionError,
    RunnerUnavailableError,
    runner_execution_error,
)
from cayu.tools._runner import (
    _safe_leaf_adapter,
    _safe_runner_adapter,
    _safe_runner_unavailable_error_for_adapter,
)

EXTERNAL = "acme-sandbox"


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Registration is process-wide and permanent; restore the snapshot so other
    # tests keep observing only built-in identities.
    for name in ("_identities", "_adapter_names", "_extension_error_types"):
        monkeypatch.setattr(_adapter_identity, name, getattr(_adapter_identity, name))


class AcmeSandboxError(RuntimeError):
    pass


class _ExternalRunner(Runner):
    isolation = EXTERNAL

    async def exec(
        self,
        command: ExecCommand,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        env_remove: tuple[str, ...] = (),
        timeout_s: int | None = None,
        stdin: str | None = None,
        output_limit_bytes: int | None = DEFAULT_EXEC_OUTPUT_LIMIT_BYTES,
    ) -> ExecResult:
        return ExecResult()


class _KillRaises:
    async def kill(self) -> bool:
        raise AcmeSandboxError("secret provider text")


def test_builtin_identities_are_seeded_without_extra_error_types() -> None:
    identities = extension_runners.registered_runner_adapter_identities()
    assert set(identities) == {"docker", "e2b", "lambda-microvm", "local", "microsandbox"}
    for name, identity in identities.items():
        assert identity == extension_runners.RunnerAdapterIdentity(name=name)
    with pytest.raises(TypeError):
        identities["acme"] = extension_runners.RunnerAdapterIdentity(name="acme")  # type: ignore[index]


@pytest.mark.parametrize(
    "name",
    ["", "Docker", "acme_sandbox", "-acme", "acme-", "acme--x", "1acme", "acmé", "a" * 64],
)
def test_adapter_names_must_be_bounded_lowercase_slugs(name: str) -> None:
    with pytest.raises(ValueError, match="slug"):
        extension_runners.register_runner_adapter_identity(name)


def test_adapter_name_type_and_reserved_value_are_rejected() -> None:
    with pytest.raises(TypeError):
        extension_runners.register_runner_adapter_identity(b"acme")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reserved"):
        extension_runners.register_runner_adapter_identity("unknown")
    assert extension_runners.register_runner_adapter_identity("a" * 63).name == "a" * 63


@pytest.mark.parametrize(
    ("error_types", "error"),
    [
        ("AcmeError", TypeError),
        ({"AcmeError": 1}, TypeError),
        ([1], TypeError),
        (["acme error"], ValueError),
        (["1Acme"], ValueError),
        (["A" * 129], ValueError),
        ([f"E{index}" for index in range(65)], ValueError),
    ],
)
def test_trusted_error_type_names_are_validated(error_types: object, error: type) -> None:
    with pytest.raises(error):
        extension_runners.register_runner_adapter_identity(
            EXTERNAL,
            trusted_error_types=error_types,  # type: ignore[arg-type]
        )
    assert EXTERNAL not in extension_runners.registered_runner_adapter_identities()


def test_identical_registration_is_idempotent_and_differing_registration_conflicts() -> None:
    first = extension_runners.register_runner_adapter_identity(
        EXTERNAL, trusted_error_types=["AcmeSandboxError", "AcmeQuotaError"]
    )
    again = extension_runners.register_runner_adapter_identity(
        EXTERNAL, trusted_error_types=("AcmeQuotaError", "AcmeSandboxError", "AcmeQuotaError")
    )
    assert again is first
    assert first.trusted_error_types == frozenset({"AcmeSandboxError", "AcmeQuotaError"})
    with pytest.raises(ValueError, match="already registered"):
        extension_runners.register_runner_adapter_identity(EXTERNAL)
    assert extension_runners.registered_runner_adapter_identities()[EXTERNAL] is first


def test_builtin_identity_can_be_restated_but_not_widened() -> None:
    docker = extension_runners.registered_runner_adapter_identities()["docker"]
    assert extension_runners.register_runner_adapter_identity("docker") is docker
    with pytest.raises(ValueError, match="already registered"):
        extension_runners.register_runner_adapter_identity(
            "docker", trusted_error_types=["AcmeSandboxError"]
        )
    assert trusted_runner_exception_type_name(AcmeSandboxError()) == "Exception"


def test_unregistered_adapter_fails_closed_to_unknown() -> None:
    failure = runner_execution_error(RuntimeError("secret"), adapter=EXTERNAL)
    assert failure.diagnostic["adapter"] == "unknown"
    assert _safe_runner_adapter(_ExternalRunner()) == "unknown"
    [artifact] = sanitize_runner_artifacts(
        [
            {
                "type": "cayu.runner_cleanup.v1",
                "adapter": EXTERNAL,
                "action": "kill_command",
                "status": "completed",
                "timeout_s": 1.0,
            }
        ]
    )
    assert artifact["adapter"] == "unknown"


def test_registered_external_adapter_survives_in_diagnostics_and_cleanup_receipts() -> None:
    extension_runners.register_runner_adapter_identity(EXTERNAL)

    failure = runner_execution_error(RuntimeError("secret"), adapter=EXTERNAL)
    assert failure.diagnostic["adapter"] == EXTERNAL
    assert runner_execution_error(failure, adapter=EXTERNAL).diagnostic == failure.diagnostic
    assert RunnerExecutionError(diagnostic=dict(failure.diagnostic)).diagnostic["adapter"] == (
        EXTERNAL
    )
    assert _safe_leaf_adapter(failure) == EXTERNAL
    assert _safe_runner_adapter(_ExternalRunner()) == EXTERNAL
    unavailable = _safe_runner_unavailable_error_for_adapter(
        RunnerUnavailableError("secret", diagnostic={"adapter": EXTERNAL}),
        adapter=EXTERNAL,
    )
    assert unavailable.diagnostic["adapter"] == EXTERNAL

    result = asyncio.run(
        extension_runners.cleanup_runner_command_with_diagnostic(
            object(),
            handle=_KillRaises(),
            adapter=EXTERNAL,
            timeout_s=1.0,
            policy="command",
        )
    )
    [artifact] = sanitize_runner_artifacts(result.artifacts)
    assert artifact == {
        "type": extension_runners.RUNNER_CLEANUP_ARTIFACT_TYPE,
        "adapter": EXTERNAL,
        "action": "kill_command",
        "status": "failed",
        "timeout_s": 1.0,
        "error_type": "Exception",
    }


def test_trusted_error_type_is_retained_only_after_registration() -> None:
    raw = AcmeSandboxError("secret provider text")
    assert trusted_runner_exception_type_name(raw) == "Exception"
    assert runner_execution_error(raw, adapter="docker").diagnostic["error_type"] == "Exception"

    extension_runners.register_runner_adapter_identity(
        EXTERNAL, trusted_error_types=["AcmeSandboxError"]
    )

    diagnostic = runner_execution_error(raw, adapter=EXTERNAL).diagnostic
    assert diagnostic["error_type"] == "AcmeSandboxError"
    assert "secret" not in repr(diagnostic)
    result = asyncio.run(
        extension_runners.cleanup_runner_command_with_diagnostic(
            object(),
            handle=_KillRaises(),
            adapter=EXTERNAL,
            timeout_s=1.0,
            policy="command",
        )
    )
    [artifact] = sanitize_runner_artifacts(result.artifacts)
    assert artifact["error_type"] == "AcmeSandboxError"
    # Unrelated names stay generic.
    assert trusted_runner_exception_type_name(type("AcmeOtherError", (Exception,), {})()) == (
        "Exception"
    )


def test_builtin_diagnostics_are_unchanged() -> None:
    for adapter in ("docker", "e2b", "lambda-microvm", "local", "microsandbox"):
        assert runner_execution_error(TimeoutError(), adapter=adapter).diagnostic == {
            "type": "cayu.runner_execution_error.v1",
            "adapter": adapter,
            "status": "failed",
            "error_type": "TimeoutError",
            "errno": None,
            "errno_code": None,
            "execution_phase": "unknown",
            "timed_out": False,
            "cancelled": False,
        }


def test_runner_re_exports_are_the_private_objects() -> None:
    from cayu.runners import _cleanup, _redacted_output, _subprocess, base

    assert extension_runners.register_runner_adapter_identity is (
        _adapter_identity.register_runner_adapter_identity
    )
    assert extension_runners.RunnerAdapterIdentity is _adapter_identity.RunnerAdapterIdentity
    assert extension_runners.RunnerCleanupResult is _cleanup.RunnerCleanupResult
    assert extension_runners.cleanup_runner_command_with_diagnostic is (
        _cleanup.cleanup_runner_command_with_diagnostic
    )
    assert extension_runners.validate_runner_cleanup_policy is (
        _cleanup.validate_runner_cleanup_policy
    )
    assert extension_runners.validate_cancel_timeout is _cleanup.validate_cancel_timeout
    assert extension_runners.DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS == 5.0
    assert extension_runners.RedactedOutputCapture is _redacted_output.RedactedOutputCapture
    assert extension_runners.redact_completed_exec_result is (
        _redacted_output.redact_completed_exec_result
    )
    for name in (
        "validate_timeout",
        "validate_output_limit",
        "validate_stdin",
        "copy_runner_env",
        "remove_runner_env",
    ):
        assert getattr(extension_runners, name) is getattr(_subprocess, name)
    assert extension_runners.copy_exec_command is base.copy_exec_command


def test_egress_re_exports_are_the_private_objects() -> None:
    from cayu.egress import _remote_adapter, proxy_exposure

    for name in (
        "prepare_exposed_proxy_binding",
        "run_enforcement_preflight",
        "run_setup_commands",
        "ProxyServerFactory",
        "DEFAULT_PROXY_SERVER_FACTORY",
        "DEFAULT_REMOTE_SETUP_COMMAND_TIMEOUT_SECONDS",
    ):
        assert getattr(extension_egress, name) is getattr(_remote_adapter, name)
    assert extension_egress.ProxyExposure is proxy_exposure.ProxyExposure
    assert extension_egress.ExposedProxy is proxy_exposure.ExposedProxy


def test_public_capability_evidence_builder_matches_the_adapter_builder() -> None:
    from cayu.egress.adapter import _virtual_egress_execution_capability_evidence

    arguments = {
        "runner_kind": EXTERNAL,
        "runner_ready": True,
        "preflight_observed_at": datetime(2026, 1, 1, tzinfo=UTC),
        "untrusted_isolation": True,
        "credential_non_possession_posture": "available",
        "guest_privilege": "live_verified",
        "unprivileged_guest": "unsupported",
        "host_filesystem_isolation": True,
        "reconnect": False,
        "cancellation_confirmed": False,
    }
    public = extension_egress.virtual_egress_execution_capability_evidence(**arguments)
    assert public == _virtual_egress_execution_capability_evidence(**arguments)
    assert public.subject == EXTERNAL
    with pytest.raises(TypeError, match="reconnect"):
        extension_egress.virtual_egress_execution_capability_evidence(
            **{**arguments, "reconnect": 1}
        )
    with pytest.raises(ValueError, match="guest_privilege"):
        extension_egress.virtual_egress_execution_capability_evidence(
            **{**arguments, "guest_privilege": "verified"}
        )
    with pytest.raises(TypeError, match="preflight_observed_at"):
        extension_egress.virtual_egress_execution_capability_evidence(
            **{**arguments, "preflight_observed_at": "2026-01-01T00:00:00Z"}
        )


def test_microsandbox_proxy_exposure_keeps_its_former_import_path() -> None:
    from cayu.egress import microsandbox_adapter, proxy_exposure

    assert proxy_exposure.MICROSANDBOX_HOST == microsandbox_adapter.MICROSANDBOX_HOST
    assert proxy_exposure.MicrosandboxHostProxyExposure is (
        microsandbox_adapter.MicrosandboxHostProxyExposure
    )
    with pytest.raises(AttributeError):
        _ = proxy_exposure.NotAProxyExposure
