"""Schema fields shared by event projection and tool-result attestation."""

from __future__ import annotations

SHARED_ARTIFACT_RESULT_AUTHORITY_FIELD = "shared_artifact_result_authority"


_REFERENCE_KEYS = frozenset(
    {
        "schema_version",
        "artifact_store_id",
        "artifact_id",
        "content_digest",
        "size_bytes",
        "source_session_id",
        "access_grant_id",
    }
)


_PUBLICATION_RECEIPT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "operation_id",
        "reference",
        "source_workspace_id",
        "source_path_sha256",
        "content_type",
        "policy_fingerprint",
        "retention_class",
        "terminal_disposition",
        "published_at",
    }
)


_MATERIALIZATION_RECEIPT_KEYS = frozenset(
    {
        "record_type",
        "schema_version",
        "operation_id",
        "reference",
        "source_workspace_id",
        "destination_session_id",
        "destination_workspace_id",
        "destination_path_sha256",
        "policy_fingerprint",
        "bytes_written",
        "terminal_disposition",
        "materialized_at",
    }
)


def _event_path(*segments: str) -> tuple[str, ...]:
    return ("result", "structured", *segments)


SHARED_ARTIFACT_RESULT_EVENT_SCHEMA_PATHS = frozenset(
    {
        *(
            _event_path(key)
            for key in {
                "shared_artifact_kind",
                "opaque_ref",
                "shared_artifact_ref",
                "publication_receipt",
                "materialization_receipt",
                "recovered_from_durable_receipt",
            }
        ),
        *(_event_path("shared_artifact_ref", key) for key in _REFERENCE_KEYS),
        *(_event_path("publication_receipt", key) for key in _PUBLICATION_RECEIPT_KEYS),
        *(_event_path("publication_receipt", "reference", key) for key in _REFERENCE_KEYS),
        *(_event_path("materialization_receipt", key) for key in _MATERIALIZATION_RECEIPT_KEYS),
        *(_event_path("materialization_receipt", "reference", key) for key in _REFERENCE_KEYS),
    }
)
