"""Detached authority for an exact, permanently fenced artifact closure set."""

from __future__ import annotations

import json
from dataclasses import dataclass

from cayu._validation import (
    MAX_DURABLE_JSON_INTEGER,
    canonical_durable_json_bytes,
    durable_json_object_from_pairs,
    require_durable_clean_nonblank,
)

ARTIFACT_CLOSURE_MAX_RECORDS = 100_000
ARTIFACT_CLOSURE_MAX_BYTES = 16 * 1024 * 1024
# A caller may grant a larger inventory budget without enlarging the durable
# claim format. Validate actual claim bytes independently of this policy bound.
ARTIFACT_CLOSURE_MAX_POLICY_BYTES = 256 * 1024 * 1024


def _identity(value: object, limit: int) -> str:
    if type(value) is not str or not 0 < len(value) <= limit:
        raise ValueError("Invalid artifact closure identity.")
    return require_durable_clean_nonblank(value, "artifact closure identity")


def _digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("Invalid artifact closure digest.")
    return value


@dataclass(frozen=True)
class ArtifactClosureItem:
    artifact_id: str
    size_bytes: int
    metadata_sha256: str

    def __post_init__(self) -> None:
        _identity(self.artifact_id, 4096)
        _digest(self.metadata_sha256)
        if type(self.size_bytes) is not int or not 0 <= self.size_bytes <= MAX_DURABLE_JSON_INTEGER:
            raise ValueError("Invalid artifact closure size.")


@dataclass(frozen=True)
class ArtifactClosureClaim:
    store_id: str
    session_id: str
    plan_id: str
    artifacts: tuple[ArtifactClosureItem, ...]

    def __post_init__(self) -> None:
        _identity(self.store_id, 4096)
        _identity(self.session_id, 256)
        _digest(self.plan_id)
        if type(self.artifacts) is not tuple or len(self.artifacts) > ARTIFACT_CLOSURE_MAX_RECORDS:
            raise ValueError("Invalid artifact closure set.")
        size = len(
            canonical_durable_json_bytes(
                {
                    "store_id": self.store_id,
                    "session_id": self.session_id,
                    "plan_id": self.plan_id,
                    "artifacts": [],
                },
                "artifact closure claim",
            )
        )
        previous = None
        total_content = 0
        for index, item in enumerate(self.artifacts):
            if type(item) is not ArtifactClosureItem:
                raise ValueError("Invalid artifact closure item.")
            # Reconstruct explicit fields: frozen instances can still be mutated
            # through object.__setattr__, and serializers are not a validation seam.
            validated = ArtifactClosureItem(item.artifact_id, item.size_bytes, item.metadata_sha256)
            if previous is not None and previous >= validated.artifact_id:
                raise ValueError("Artifact closure set must be sorted and unique.")
            previous = validated.artifact_id
            total_content += validated.size_bytes
            if total_content > MAX_DURABLE_JSON_INTEGER:
                raise ValueError("Artifact closure content total exceeds its bound.")
            size += bool(index) + len(
                canonical_durable_json_bytes(
                    {
                        "artifact_id": validated.artifact_id,
                        "size_bytes": validated.size_bytes,
                        "metadata_sha256": validated.metadata_sha256,
                    },
                    "artifact closure item",
                    max_bytes=ARTIFACT_CLOSURE_MAX_BYTES,
                )
            )
            if size > ARTIFACT_CLOSURE_MAX_BYTES:
                raise ValueError("Artifact closure claim exceeds its byte bound.")


def copy_artifact_closure_claim(value: ArtifactClosureClaim) -> ArtifactClosureClaim:
    if type(value) is not ArtifactClosureClaim:
        raise TypeError("A typed artifact closure claim is required.")
    # Validate the complete input before allocating the detached collection.
    ArtifactClosureClaim(value.store_id, value.session_id, value.plan_id, value.artifacts)
    return ArtifactClosureClaim(
        value.store_id,
        value.session_id,
        value.plan_id,
        tuple(
            ArtifactClosureItem(item.artifact_id, item.size_bytes, item.metadata_sha256)
            for item in value.artifacts
        ),
    )


def encode_artifact_closure_claim(value: ArtifactClosureClaim) -> bytes:
    value = copy_artifact_closure_claim(value)
    return canonical_durable_json_bytes(
        {
            "store_id": value.store_id,
            "session_id": value.session_id,
            "plan_id": value.plan_id,
            "artifacts": [
                {
                    "artifact_id": item.artifact_id,
                    "size_bytes": item.size_bytes,
                    "metadata_sha256": item.metadata_sha256,
                }
                for item in value.artifacts
            ],
        },
        "artifact closure claim",
        max_bytes=ARTIFACT_CLOSURE_MAX_BYTES,
        max_nodes=1_000_000,
    )


def decode_artifact_closure_claim(value: bytes) -> ArtifactClosureClaim:
    if type(value) is not bytes or len(value) > ARTIFACT_CLOSURE_MAX_BYTES:
        raise ValueError("Invalid artifact closure record size.")
    try:
        raw = json.loads(
            value,
            object_pairs_hook=lambda pairs: durable_json_object_from_pairs(
                pairs, "artifact closure claim"
            ),
        )
        if type(raw) is not dict or set(raw) != {"store_id", "session_id", "plan_id", "artifacts"}:
            raise ValueError
        if (
            type(raw["artifacts"]) is not list
            or len(raw["artifacts"]) > ARTIFACT_CLOSURE_MAX_RECORDS
        ):
            raise ValueError
        items = []
        for item in raw["artifacts"]:
            if type(item) is not dict or set(item) != {
                "artifact_id",
                "size_bytes",
                "metadata_sha256",
            }:
                raise ValueError
            items.append(
                ArtifactClosureItem(
                    item["artifact_id"], item["size_bytes"], item["metadata_sha256"]
                )
            )
        return ArtifactClosureClaim(
            raw["store_id"], raw["session_id"], raw["plan_id"], tuple(items)
        )
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ValueError("Invalid durable artifact closure claim.") from None
