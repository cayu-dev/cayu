"""Bounded diagnostic labels; these never grant retry or settlement authority."""

API_CLASSIFICATION_REASONS = frozenset(
    {
        "explicit_status_conflict",
        "identity_conflict",
        "status_identity_conflict",
        "recognized_identity",
        "explicit_status",
        "unsupported_identity",
        "absent_identity",
    }
)
API_CLASSIFICATION_ORIGINS = frozenset({"http", "stream"})


def api_error_diagnostic_fields(
    exc: Exception, *, credential_values: tuple[str, ...] = ()
) -> dict[str, str]:
    fields = {}
    for attribute, vocabulary in (
        ("classification_reason", API_CLASSIFICATION_REASONS),
        ("classification_origin", API_CLASSIFICATION_ORIGINS),
    ):
        value = getattr(exc, attribute, None)
        if (
            type(value) is str
            and value in vocabulary
            and not any(secret and secret in value for secret in credential_values)
        ):
            fields["provider_api_" + attribute] = value
    return fields
