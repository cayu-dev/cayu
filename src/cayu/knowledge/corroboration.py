"""Corroboration: another session recorded as a source on an existing knowledge entry.

When a model finds that the fact it was about to remember is already proposed or kept,
``remember_knowledge(same_as=ENTRY_ID)`` records the calling session as another source of
that entry instead of proposing a duplicate. The entry gets one successor revision with the
same text, kind, labels and status. Its provenance grows in two places, written together in
that one revision:

- a ``supporting`` :class:`~cayu.knowledge.records.KnowledgeEvidence` record for the
  session, beside the evidence the entry already had (approval and review carry evidence
  to later revisions, so it stays with the entry once it is kept); and
- a summary in the entry's metadata under :data:`KNOWLEDGE_CORROBORATION_METADATA_KEY`::

      {"source_count": 3,
       "sources": [{"session_id": "...", "recorded_at": "...", "reason": "..."}, ...]}

``source_count`` counts the session that first wrote the entry plus every corroborating
session, so an entry nobody corroborated has a count of 1 and no metadata key. Read it
with :func:`knowledge_source_count`, which is what review queues sort by, and the sources
with :func:`knowledge_corroborations`.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from cayu._validation import canonical_durable_json_bytes
from cayu.knowledge._revision_rules import _copy_chunks_for_revision, _copy_evidence_for_revision
from cayu.knowledge.records import (
    KnowledgeChunk,
    KnowledgeEntry,
    KnowledgeEvidence,
    KnowledgeEvidenceRole,
    _next_knowledge_revision,
    copy_knowledge_entry,
    require_clean_nonblank,
)

#: The entry metadata key that holds the corroboration summary.
KNOWLEDGE_CORROBORATION_METADATA_KEY = "cayu:corroboration"
#: The most corroborating sessions one entry records; later ones are refused.
MAX_KNOWLEDGE_CORROBORATIONS = 50
#: The longest reason a corroborating session may give, in characters.
MAX_KNOWLEDGE_CORROBORATION_REASON_CHARS = 280
#: The ``source_type`` of a corroborating session's evidence record.
KNOWLEDGE_CORROBORATION_SOURCE_TYPE = "tool"


class KnowledgeCorroborationRefused(ValueError):
    """An entry can't record one more corroborating session.

    ``code`` is ``already_recorded`` (the session wrote the entry or already corroborates
    it), ``source_limit_reached`` or ``malformed_sources`` (the metadata summary is not one
    Cayu wrote).
    """

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Knowledge corroboration refused: {code}.")


class KnowledgeCorroboration(BaseModel):
    """One session recorded as a source of an entry it did not write."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    recorded_at: datetime
    reason: str | None = None

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        value = require_clean_nonblank(value, "session_id")
        if len(value.encode("utf-8")) > 256:
            raise ValueError("`session_id` must be at most 256 UTF-8 bytes.")
        return value

    @field_validator("recorded_at")
    @classmethod
    def validate_recorded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("`recorded_at` must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = require_clean_nonblank(value, "reason")
        if len(value) > MAX_KNOWLEDGE_CORROBORATION_REASON_CHARS:
            raise ValueError(
                f"`reason` must be at most {MAX_KNOWLEDGE_CORROBORATION_REASON_CHARS} characters."
            )
        return value

    def metadata_value(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "session_id": self.session_id,
            "recorded_at": self.recorded_at.isoformat(),
        }
        if self.reason is not None:
            value["reason"] = self.reason
        return value


def knowledge_corroborations(entry: KnowledgeEntry) -> tuple[KnowledgeCorroboration, ...]:
    """The sessions recorded as corroborating sources of ``entry``, oldest first.

    An entry without the metadata key, or with one Cayu did not write, has none.
    """

    return knowledge_corroborations_in_metadata(entry.metadata)


def knowledge_corroborations_in_metadata(
    metadata: Mapping[str, Any],
) -> tuple[KnowledgeCorroboration, ...]:
    """:func:`knowledge_corroborations` for an entry's metadata alone, as an API returns it."""

    parsed = _parse_corroborations(metadata)
    return () if parsed is None else parsed


def knowledge_source_count(entry: KnowledgeEntry) -> int:
    """How many sessions are sources of ``entry``: the one that wrote it plus each
    corroborating one. An entry nobody corroborated counts 1."""

    return 1 + len(knowledge_corroborations(entry))


def knowledge_entry_has_source_session(entry: KnowledgeEntry, session_id: str) -> bool:
    """Whether ``session_id`` wrote ``entry`` or is already recorded as corroborating it."""

    if entry.source_id == session_id or entry.metadata.get("session_id") == session_id:
        return True
    return any(item.session_id == session_id for item in knowledge_corroborations(entry))


def corroborated_knowledge_revision(
    entry: KnowledgeEntry,
    chunks: list[KnowledgeChunk],
    evidence: list[KnowledgeEvidence],
    *,
    corroboration: KnowledgeCorroboration,
) -> tuple[KnowledgeEntry, list[KnowledgeChunk], list[KnowledgeEvidence]]:
    """The successor of ``entry`` that records one more corroborating session.

    ``chunks`` and ``evidence`` must be the complete chunks and evidence of ``entry``'s
    exact revision. The successor keeps the text, chunks, status and every other field,
    carries the evidence forward, and adds the session's ``supporting`` evidence and its
    line in the metadata summary. Raises :class:`KnowledgeCorroborationRefused` when the
    entry already records the session, has reached :data:`MAX_KNOWLEDGE_CORROBORATIONS`,
    or holds a summary Cayu did not write.
    """

    existing = _parse_corroborations(entry.metadata)
    if existing is None:
        raise KnowledgeCorroborationRefused("malformed_sources")
    if knowledge_entry_has_source_session(entry, corroboration.session_id):
        raise KnowledgeCorroborationRefused("already_recorded")
    if len(existing) >= MAX_KNOWLEDGE_CORROBORATIONS:
        raise KnowledgeCorroborationRefused("source_limit_reached")
    sources = [*existing, corroboration]
    metadata = {
        **entry.metadata,
        KNOWLEDGE_CORROBORATION_METADATA_KEY: {
            "source_count": 1 + len(sources),
            "sources": [item.metadata_value() for item in sources],
        },
    }
    successor = KnowledgeEntry.model_validate(
        {
            **copy_knowledge_entry(entry).model_dump(),
            "revision": _next_knowledge_revision(entry.revision),
            "updated_at": max(corroboration.recorded_at, entry.created_at, entry.updated_at),
            "metadata": metadata,
        }
    )
    successor_chunks = _copy_chunks_for_revision(chunks, successor) if chunks else []
    successor_evidence = _copy_evidence_for_revision(
        evidence,
        entry=successor,
        previous_chunks=chunks,
        chunks=successor_chunks,
    )
    session_uri = f"cayu://sessions/{corroboration.session_id}"
    successor_evidence.append(
        KnowledgeEvidence(
            id=_corroboration_evidence_id(successor, corroboration.session_id),
            entry_id=successor.id,
            entry_revision=successor.revision,
            role=KnowledgeEvidenceRole.SUPPORTING,
            source_type=KNOWLEDGE_CORROBORATION_SOURCE_TYPE,
            source_id=corroboration.session_id,
            source_uri=session_uri,
            # The session vouches for this exact text.
            source_hash=successor.source_hash
            or "sha256:" + sha256(successor.text.encode("utf-8")).hexdigest(),
            created_at=corroboration.recorded_at,
            metadata=(
                {"corroboration": True}
                if corroboration.reason is None
                else {"corroboration": True, "reason": corroboration.reason}
            ),
        )
    )
    return successor, successor_chunks, successor_evidence


def _corroboration_evidence_id(entry: KnowledgeEntry, session_id: str) -> str:
    material = canonical_durable_json_bytes(
        {
            "contract": "cayu-knowledge-corroboration-evidence-v1",
            "entry_id": entry.id,
            "entry_revision": entry.revision,
            "session_id": session_id,
        },
        "knowledge corroboration evidence identity",
    )
    return "ke_" + sha256(material).hexdigest()


def _parse_corroborations(
    metadata: Mapping[str, Any],
) -> tuple[KnowledgeCorroboration, ...] | None:
    """The recorded corroborations, ``()`` without a summary, ``None`` for a malformed one."""

    summary = metadata.get(KNOWLEDGE_CORROBORATION_METADATA_KEY)
    if summary is None:
        return ()
    if not isinstance(summary, dict):
        return None
    sources = summary.get("sources")
    if (
        set(summary) != {"source_count", "sources"}
        or not isinstance(sources, list)
        or len(sources) > MAX_KNOWLEDGE_CORROBORATIONS
        or summary.get("source_count") != 1 + len(sources)
    ):
        return None
    parsed: list[KnowledgeCorroboration] = []
    try:
        for item in sources:
            if not isinstance(item, dict):
                return None
            parsed.append(KnowledgeCorroboration.model_validate(item))
    except ValueError:
        return None
    if len({item.session_id for item in parsed}) != len(parsed):
        return None
    return tuple(parsed)
