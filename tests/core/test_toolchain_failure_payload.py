"""Toolchain refusals reach failure events as a stable code, path count and fingerprint."""

from __future__ import annotations

from cayu.environments import CodingToolchainError, DockerCodingToolchainError
from cayu.runtime._environment_lifecycle import exception_failure_payload


def test_a_toolchain_refusal_publishes_its_code_count_and_fingerprint_not_its_paths() -> None:
    error = DockerCodingToolchainError(
        "dependency_inputs_unavailable",
        "Toolchain dependency inputs could not be verified.",
        paths=("Dockerfile.coding", "private/uv.lock"),
    )

    payload = exception_failure_payload(error)

    assert payload["error_type"] == "DockerCodingToolchainError"
    assert payload["toolchain_error"] == {
        "code": "dependency_inputs_unavailable",
        "path_count": 2,
        "paths_fingerprint": error.paths_fingerprint,
    }
    assert "private/uv.lock" not in str(payload)


def test_a_wrapped_or_grouped_neutral_refusal_is_found() -> None:
    refusal = CodingToolchainError("toolchain_image_unverified", "Image is not verified.")
    try:
        try:
            raise refusal
        except CodingToolchainError as exc:
            raise RuntimeError("Environment creation failed.") from exc
    except RuntimeError as wrapped:
        assert exception_failure_payload(wrapped)["toolchain_error"]["code"] == (
            "toolchain_image_unverified"
        )
    group = BaseExceptionGroup("creation failed", [ValueError("other"), refusal])
    assert exception_failure_payload(group)["toolchain_error"]["path_count"] == 0


def test_lookalike_errors_publish_nothing_extra() -> None:
    class LookAlike(CodingToolchainError):
        pass

    assert "toolchain_error" not in exception_failure_payload(
        LookAlike("dependency_inputs_changed", "changed", paths=("secret.txt",))
    )
    assert "toolchain_error" not in exception_failure_payload(ValueError("plain"))
