"""Exact source selection and bounded, content-free knowledge closure evidence."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_bounded_durable_json_value,
    inspect_bounded_durable_json,
    require_durable_clean_nonblank,
)

KNOWLEDGE_CLOSURE_CLASSES = (
    "knowledge_revisions",
    "knowledge_evidence",
    "knowledge_projections",
    "knowledge_index_readiness",
)


def validate_knowledge_closure_inventory(
    raw: object, query: KnowledgeClosureQuery
) -> dict[str, Any]:
    query = copy_knowledge_closure_query(query)
    value = copy_bounded_durable_json_value(
        raw, "knowledge closure inventory", max_bytes=query.max_bytes, max_nodes=1_000_000
    )
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "records", "counts", "complete"}
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["complete"] is not True
        or type(value["records"]) is not dict
        or type(value["counts"]) is not dict
        or set(value["records"]) != set(KNOWLEDGE_CLOSURE_CLASSES)
        or set(value["counts"]) != set(KNOWLEDGE_CLOSURE_CLASSES)
    ):
        raise ValueError("Invalid knowledge closure inventory.")
    count = 0
    for record_class in KNOWLEDGE_CLOSURE_CLASSES:
        rows = value["records"][record_class]
        expected = value["counts"][record_class]
        if type(rows) is not list or type(expected) is not int or expected != len(rows):
            raise ValueError("Conflicting knowledge closure inventory count.")
        count += expected
        if count > query.max_records:
            raise ValueError("Knowledge closure inventory exceeds its bounds.")
        seen = set()
        for row in rows:
            if type(row) is not dict or set(row) != {"identity_sha256", "disposition"}:
                raise ValueError("Invalid knowledge closure reference.")
            digest = row["identity_sha256"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or row["disposition"] != "retained"
                or digest in seen
            ):
                raise ValueError("Invalid knowledge closure reference.")
            seen.add(digest)
    return cast("dict[str, Any]", value)


@dataclass(frozen=True)
class KnowledgeClosureQuery:
    sources: tuple[tuple[str, str], ...]
    max_records: int = 10_000
    max_bytes: int = 16 * 1024 * 1024
    source_uris: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if type(self.sources) is not tuple or len(self.sources) > 100_000:
            raise ValueError("Invalid knowledge closure source set.")
        for pair in self.sources:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("Invalid knowledge closure source identity.")
            for value in pair:
                if type(value) is not str or not 0 < len(value) <= 256:
                    raise ValueError("Invalid knowledge closure source identity.")
                require_durable_clean_nonblank(value, "knowledge closure source")
        if type(self.source_uris) is not tuple or len(self.source_uris) > 100_000:
            raise ValueError("Invalid knowledge closure URI set.")
        for pair in self.source_uris:
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError("Invalid knowledge closure URI identity.")
            for value, limit in zip(pair, (256, 4096), strict=True):
                if type(value) is not str or not 0 < len(value) <= limit:
                    raise ValueError("Invalid knowledge closure URI identity.")
                require_durable_clean_nonblank(value, "knowledge closure URI")
        if type(self.max_records) is not int or not 0 < self.max_records <= 100_000:
            raise ValueError("Invalid knowledge closure record bound.")
        if type(self.max_bytes) is not int or not 0 < self.max_bytes <= 256 * 1024 * 1024:
            raise ValueError("Invalid knowledge closure byte bound.")
        inspect_bounded_durable_json(
            (self.sources, self.source_uris),
            "knowledge closure sources",
            max_bytes=self.max_bytes,
            max_nodes=600_003,
            allow_tuples=True,
        )
        object.__setattr__(self, "sources", tuple(sorted(set(self.sources))))
        object.__setattr__(self, "source_uris", tuple(sorted(set(self.source_uris))))


def copy_knowledge_closure_query(query: KnowledgeClosureQuery) -> KnowledgeClosureQuery:
    if type(query) is not KnowledgeClosureQuery:
        raise TypeError("A typed knowledge closure query is required.")
    return KnowledgeClosureQuery(
        query.sources, query.max_records, query.max_bytes, query.source_uris
    )


class KnowledgeClosureInventory:
    """One bounded set of references, never knowledge content or vectors."""

    def __init__(self, query: KnowledgeClosureQuery):
        self.query = copy_knowledge_closure_query(query)
        self._records: dict[str, list[dict[str, object]]] = {
            key: [] for key in KNOWLEDGE_CLOSURE_CLASSES
        }
        self._keys: dict[tuple[str, str], str] = {}
        self._bytes = 0
        self._failed = False

    @property
    def count(self) -> int:
        return len(self._keys)

    def add_revision(
        self, entry_id, revision, source_type, source_id, source_uri, source_hash
    ) -> tuple[str, int]:
        if type(revision) is not int or not 0 < revision <= 2_147_483_647:
            raise ValueError("Invalid knowledge closure revision.")
        for value in (entry_id, source_type):
            if type(value) is not str or not value:
                raise ValueError("Invalid knowledge closure revision identity.")
        for value in (source_id, source_uri, source_hash):
            if value is not None and type(value) is not str:
                raise ValueError("Invalid knowledge closure revision source.")
        self.add(
            "knowledge_revisions",
            {
                "entry_id": entry_id,
                "revision": revision,
                "source_type": source_type,
                "source_id": source_id,
                "source_uri": source_uri,
                "source_hash": source_hash,
            },
        )
        return entry_id, revision

    def add_evidence(self, evidence) -> tuple[str, int]:
        from cayu.storage.memory import copy_knowledge_evidence

        copied = copy_knowledge_evidence(evidence)
        self.add(
            "knowledge_evidence",
            {
                "id": copied.id,
                "entry_id": copied.entry_id,
                "entry_revision": copied.entry_revision,
                "source_type": copied.source_type,
                "source_id": copied.source_id,
                "source_revision": copied.source_revision,
                "source_hash": copied.source_hash,
                "source_uri": copied.source_uri,
                "chunk_id": copied.chunk_id,
                "role": copied.role.value,
                "locator": copied.locator,
                "disposition": copied.disposition.value,
                "created_at": copied.created_at.isoformat(),
                "metadata": copied.metadata,
            },
        )
        return copied.entry_id, copied.entry_revision

    def add(self, record_class: str, identity: dict[str, object]) -> None:
        if self._failed:
            raise ValueError("Knowledge closure inventory already failed.")
        try:
            if record_class not in self._records:
                raise ValueError("Unknown knowledge closure record class.")
            encoded = canonical_durable_json_bytes(
                identity,
                "knowledge closure identity",
                max_bytes=self.query.max_bytes,
                max_nodes=100_000,
            )
            digest = sha256(encoded).hexdigest()
            stable_identity = (
                {"entry_id": identity["entry_id"], "revision": identity["revision"]}
                if record_class == "knowledge_revisions"
                else {"id": identity["id"]}
                if "id" in identity
                else {"identity": identity["identity"], "attempt_id": identity["attempt_id"]}
            )
            key = (
                record_class,
                sha256(
                    canonical_durable_json_bytes(
                        stable_identity,
                        "knowledge closure record identity",
                        max_bytes=self.query.max_bytes,
                        max_nodes=100,
                    )
                ).hexdigest(),
            )
            if key in self._keys:
                if self._keys[key] != digest:
                    raise ValueError("Knowledge closure record identity has conflicting content.")
                return
            row: dict[str, object] = {"identity_sha256": digest, "disposition": "retained"}
            size = len(canonical_durable_json_bytes(row, "knowledge closure reference"))
            if (
                len(self._keys) >= self.query.max_records
                or self._bytes + size > self.query.max_bytes
            ):
                raise ValueError("Knowledge closure inventory exceeds its bounds.")
            self._records[record_class].append(row)
            self._keys[key] = digest
            self._bytes += size
        except BaseException:
            self._failed = True
            raise

    def add_readiness(self, readiness) -> None:
        from cayu.runtime._session_closure_records import ClosureRecordsBuilder
        from cayu.storage.memory import KnowledgeEmbeddingIdentity, KnowledgeIndexReadiness

        if type(readiness) is not KnowledgeIndexReadiness:
            raise ValueError("Invalid knowledge closure readiness record.")
        # Inspect owned fields before reconstruction: model_dump can emit an
        # extension-mutated value in a serializer warning before validation.
        builder = ClosureRecordsBuilder(max_records=1, max_bytes=self.query.max_bytes)
        builder.add_class("readiness", (readiness,))
        row = builder.records["readiness"][0]
        reconstructed = KnowledgeIndexReadiness(
            **{**row, "identity": KnowledgeEmbeddingIdentity(**row["identity"])}
        )
        self.add(
            "knowledge_index_readiness",
            {"id": reconstructed.operation_id, **row},
        )

    def document(self) -> dict[str, object]:
        if self._failed:
            raise ValueError("Knowledge closure inventory already failed.")
        document: dict[str, object] = {
            "schema_version": 1,
            "records": {
                key: [
                    dict(row) for row in sorted(rows, key=lambda item: str(item["identity_sha256"]))
                ]
                for key, rows in self._records.items()
            },
            "counts": {key: len(rows) for key, rows in self._records.items()},
            "complete": True,
        }
        canonical_durable_json_bytes(
            document,
            "knowledge closure inventory",
            max_bytes=self.query.max_bytes,
            max_nodes=1_000_000,
        )
        return document
