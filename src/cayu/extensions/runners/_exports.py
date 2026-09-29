"""Explicit public exports; implementations load on first access."""

EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS": (
        "cayu.runners._cleanup",
        "DEFAULT_RUNNER_CANCEL_TIMEOUT_SECONDS",
    ),
    "RUNNER_CLEANUP_ARTIFACT_TYPE": ("cayu.runners._cleanup", "RUNNER_CLEANUP_ARTIFACT_TYPE"),
    "RedactedOutputCapture": ("cayu.runners._redacted_output", "RedactedOutputCapture"),
    "RunnerAdapterIdentity": ("cayu.runners._adapter_identity", "RunnerAdapterIdentity"),
    "RunnerCleanupResult": ("cayu.runners._cleanup", "RunnerCleanupResult"),
    "cleanup_runner_command_with_diagnostic": (
        "cayu.runners._cleanup",
        "cleanup_runner_command_with_diagnostic",
    ),
    "copy_exec_command": ("cayu.runners.base", "copy_exec_command"),
    "copy_runner_env": ("cayu.runners._subprocess", "copy_runner_env"),
    "redact_completed_exec_result": (
        "cayu.runners._redacted_output",
        "redact_completed_exec_result",
    ),
    "register_runner_adapter_identity": (
        "cayu.runners._adapter_identity",
        "register_runner_adapter_identity",
    ),
    "registered_runner_adapter_identities": (
        "cayu.runners._adapter_identity",
        "registered_runner_adapter_identities",
    ),
    "remove_runner_env": ("cayu.runners._subprocess", "remove_runner_env"),
    "validate_cancel_timeout": ("cayu.runners._cleanup", "validate_cancel_timeout"),
    "validate_output_limit": ("cayu.runners._subprocess", "validate_output_limit"),
    "validate_runner_cleanup_policy": ("cayu.runners._cleanup", "validate_runner_cleanup_policy"),
    "validate_stdin": ("cayu.runners._subprocess", "validate_stdin"),
    "validate_timeout": ("cayu.runners._subprocess", "validate_timeout"),
}

PUBLIC_NAMES = sorted(EXPORTS)
