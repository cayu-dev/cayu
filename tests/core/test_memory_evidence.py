from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import get_args

import pytest
from pydantic import ValidationError

from cayu import (
    ContextExposure,
    ContextExposureTransition,
    KeyedEvidenceFingerprintDomain,
    KnowledgeChunkEvidenceLocator,
    KnowledgeEntryEvidenceLocator,
    OpaqueRecallEvidenceLocator,
    RecallEvidenceQuery,
    RecallItemSelectionReason,
    RecallReceipt,
    RecallReceiptPage,
    RecallSourceCoverage,
    RetrievalCandidateIdentity,
    keyed_evidence_fingerprint,
)
from cayu.memory import _RecallOfferReason
from cayu.memory_evidence import (
    MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES,
    MAX_RECALL_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_RECALL_KNOWLEDGE_CHUNK_INDEX,
    MAX_RECALL_KNOWLEDGE_ENTRY_ID_BYTES,
    MAX_RECALL_KNOWLEDGE_REVISION,
    decode_recall_evidence_cursor,
    encode_recall_evidence_cursor,
)
from cayu.runtime.sessions import MAX_SESSION_ID_BYTES
from cayu.storage.memory import (
    MAX_KNOWLEDGE_CHUNK_ID_BYTES,
    MAX_KNOWLEDGE_CHUNK_INDEX,
    MAX_KNOWLEDGE_ENTRY_ID_BYTES,
    MAX_KNOWLEDGE_REVISION,
)


def _digest(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _fingerprint(value: str, domain: str) -> dict[str, str]:
    return {
        "algorithm": "hmac-sha256",
        "domain": domain,
        "key_id": "memory-evidence-test",
        "digest": _digest(value),
    }


def _receipt_document() -> dict[str, object]:
    return {
        "receipt_id": "receipt-contract",
        "session_id": "session-contract",
        "interaction_id": "interaction-contract",
        "model_step_id": f"mstep_{'1' * 32}",
        "created_at": datetime(2026, 8, 22, tzinfo=UTC),
        "situation_fingerprint": _fingerprint("situation", "recall_situation"),
        "engine_version": "cayu.recall.v1",
        "source_configuration_fingerprint": _fingerprint("sources", "recall_source_configuration"),
        "admission_policy_fingerprint": _fingerprint("policy", "recall_admission_policy"),
        "access_scope_fingerprint": _fingerprint("scope", "recall_access_scope"),
        "frontier_fingerprint": _fingerprint("frontier", "recall_frontier"),
        "sources": [
            {
                "source": "knowledge",
                "required": True,
                "channels": ["knowledge.lexical"],
                "state": "complete",
                "inspected_count": 1,
                "candidate_limit": 10,
            }
        ],
        "inspected_count": 1,
        "eligible_count": 1,
        "admitted_count": 1,
        "offered_count": 0,
        "silent_count": 0,
        "omitted_count": 0,
        "truncated": False,
        "items": [
            {
                "ordinal": 0,
                "identity": {
                    "record_type": "knowledge_entry",
                    "record_id": "entry-contract",
                    "revision": "1",
                },
                "representation_id": "entry_text",
                "content_sha256": _digest("content"),
                "locator": {
                    "kind": "knowledge_entry",
                    "entry_id": "entry-contract",
                    "entry_revision": 1,
                },
                "admission": "admitted",
                "selection_reason": "calibrated_strong_match",
                "fused_rank": 1,
                "match_channels": ["knowledge.lexical"],
            }
        ],
    }


def test_receipt_selection_reasons_preserve_automatic_admission_values() -> None:
    offer_reasons = set(get_args(_RecallOfferReason))
    evidence_reasons = {reason.value for reason in RecallItemSelectionReason}

    assert offer_reasons <= evidence_reasons


def test_memory_evidence_source_identity_bounds_match_built_in_sources() -> None:
    assert MAX_MEMORY_EVIDENCE_SESSION_ID_BYTES == MAX_SESSION_ID_BYTES
    assert MAX_RECALL_KNOWLEDGE_ENTRY_ID_BYTES == MAX_KNOWLEDGE_ENTRY_ID_BYTES
    assert MAX_RECALL_KNOWLEDGE_CHUNK_ID_BYTES == MAX_KNOWLEDGE_CHUNK_ID_BYTES
    assert MAX_RECALL_KNOWLEDGE_REVISION == MAX_KNOWLEDGE_REVISION
    assert MAX_RECALL_KNOWLEDGE_CHUNK_INDEX == MAX_KNOWLEDGE_CHUNK_INDEX


@pytest.mark.parametrize("state", ["unavailable", "failed"])
def test_unavailable_recall_source_is_explicitly_incomplete(state: str) -> None:
    with pytest.raises(ValidationError, match="explicitly incomplete"):
        RecallSourceCoverage(
            source="knowledge",
            required=False,
            channels=("knowledge.lexical",),
            state=state,
            failure_code="source_unavailable",
            inspected_count=0,
            candidate_limit=10,
            truncated=False,
        )

    coverage = RecallSourceCoverage(
        source="knowledge",
        required=False,
        channels=("knowledge.lexical",),
        state=state,
        failure_code="source_unavailable",
        inspected_count=0,
        candidate_limit=10,
        truncated=True,
    )
    assert coverage.truncated is True
    assert coverage.continuation_available is False


def test_recall_receipt_requires_exhaustive_outcome_counts() -> None:
    document = _receipt_document()
    document["eligible_count"] = 2

    with pytest.raises(ValidationError, match="eligible count does not match"):
        RecallReceipt.model_validate(document)


def test_recall_receipt_requires_declared_match_provenance_and_bounded_ranks() -> None:
    undeclared_channel = _receipt_document()
    undeclared_items = undeclared_channel["items"]
    assert isinstance(undeclared_items, list)
    undeclared_items[0]["match_channels"] = ["transcript.lexical"]
    with pytest.raises(ValidationError, match="declared by source coverage"):
        RecallReceipt.model_validate(undeclared_channel)

    impossible_rank = _receipt_document()
    impossible_items = impossible_rank["items"]
    assert isinstance(impossible_items, list)
    impossible_items[0]["fused_rank"] = 2
    with pytest.raises(ValidationError, match="rank exceeds"):
        RecallReceipt.model_validate(impossible_rank)


def test_recall_receipt_rejects_unkeyed_or_cross_domain_private_fingerprints() -> None:
    unkeyed = _receipt_document()
    unkeyed["situation_fingerprint"] = _digest("guessable-situation")
    with pytest.raises(ValidationError):
        RecallReceipt.model_validate(unkeyed)

    cross_domain = _receipt_document()
    cross_domain["situation_fingerprint"] = _fingerprint(
        "guessable-situation", "recall_access_scope"
    )
    with pytest.raises(ValidationError, match="wrong keyed-fingerprint domain"):
        RecallReceipt.model_validate(cross_domain)


def test_recall_receipt_requires_one_key_identity() -> None:
    mixed_top_level = _receipt_document()
    frontier = mixed_top_level["frontier_fingerprint"]
    assert isinstance(frontier, dict)
    frontier["key_id"] = "rotated-memory-evidence-test"
    with pytest.raises(ValidationError, match="must use one key identity"):
        RecallReceipt.model_validate(mixed_top_level)

    mixed_locator = _receipt_document()
    items = mixed_locator["items"]
    assert isinstance(items, list)
    items[0]["identity"] = {
        "record_type": "custom_memory_record",
        "record_id": "custom-contract",
        "revision": "1",
    }
    items[0]["locator"] = {
        "kind": "opaque",
        "fingerprint": {
            **_fingerprint("private-custom-locator", "recall_source_locator"),
            "key_id": "rotated-memory-evidence-test",
        },
    }
    with pytest.raises(ValidationError, match="must use one key identity"):
        RecallReceipt.model_validate(mixed_locator)


def test_keyed_evidence_fingerprint_is_deterministic_and_domain_separated() -> None:
    source_digest = _digest("private low-entropy material")
    key = bytes(range(32))
    situation = keyed_evidence_fingerprint(
        source_digest,
        domain=KeyedEvidenceFingerprintDomain.SITUATION,
        key_id="test-key",
        key=key,
    )
    replay = keyed_evidence_fingerprint(
        source_digest,
        domain=KeyedEvidenceFingerprintDomain.SITUATION,
        key_id="test-key",
        key=key,
    )
    access_scope = keyed_evidence_fingerprint(
        source_digest,
        domain=KeyedEvidenceFingerprintDomain.ACCESS_SCOPE,
        key_id="test-key",
        key=key,
    )

    assert situation == replay
    assert situation.digest != source_digest
    assert situation.digest != access_scope.digest
    assert key.hex() not in str(situation.model_dump(mode="json"))


def test_recall_receipt_rejects_raw_query_material() -> None:
    document = _receipt_document()
    document["query_text"] = "private user request"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecallReceipt.model_validate(document)


def test_recall_receipt_rejects_arbitrary_or_mismatched_locator_material() -> None:
    document = _receipt_document()
    items = document["items"]
    assert isinstance(items, list)
    locator = items[0]["locator"]
    assert isinstance(locator, dict)
    locator["api_key"] = "sk-private-locator-secret"

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecallReceipt.model_validate(document)

    mismatched = _receipt_document()
    mismatched_items = mismatched["items"]
    assert isinstance(mismatched_items, list)
    mismatched_locator = mismatched_items[0]["locator"]
    assert isinstance(mismatched_locator, dict)
    mismatched_locator["entry_id"] = "different-entry"
    with pytest.raises(ValidationError, match="conflicts with its canonical identity"):
        RecallReceipt.model_validate(mismatched)


def test_recall_receipt_bounds_candidate_identity_and_rejects_model_subclasses() -> None:
    oversized = _receipt_document()
    oversized_items = oversized["items"]
    assert isinstance(oversized_items, list)
    oversized_items[0]["identity"]["record_id"] = "x" * 4_097
    with pytest.raises(ValidationError, match="UTF-8 byte bound"):
        RecallReceipt.model_validate(oversized)

    class PrivateIdentity(RetrievalCandidateIdentity):
        api_key: str

    subclassed = _receipt_document()
    subclassed_items = subclassed["items"]
    assert isinstance(subclassed_items, list)
    subclassed_items[0]["identity"] = PrivateIdentity(
        record_type="knowledge_entry",
        record_id="entry-contract",
        revision="1",
        api_key="sk-private-identity-secret",
    )
    with pytest.raises(TypeError, match="must not be a RetrievalCandidateIdentity subclass"):
        RecallReceipt.model_validate(subclassed)


def test_recall_receipt_accepts_full_built_in_locator_identity_bounds() -> None:
    chunk_document = _receipt_document()
    chunk_items = chunk_document["items"]
    assert isinstance(chunk_items, list)
    chunk_id = "c" * 512
    chunk_items[0]["identity"] = {
        "record_type": "knowledge_chunk",
        "record_id": chunk_id,
        "revision": "1",
    }
    chunk_items[0]["locator"] = {
        "kind": "knowledge_chunk",
        "entry_id": "e" * 256,
        "entry_revision": 1,
        "chunk_id": chunk_id,
        "chunk_index": 0,
    }
    chunk_receipt = RecallReceipt.model_validate(chunk_document)
    assert isinstance(chunk_receipt.items[0].locator, KnowledgeChunkEvidenceLocator)

    oversized_chunk = _receipt_document()
    oversized_chunk_items = oversized_chunk["items"]
    assert isinstance(oversized_chunk_items, list)
    oversized_chunk_items[0]["identity"] = {
        "record_type": "knowledge_chunk",
        "record_id": "c" * 513,
        "revision": "1",
    }
    oversized_chunk_items[0]["locator"] = {
        "kind": "knowledge_chunk",
        "entry_id": "entry-contract",
        "entry_revision": 1,
        "chunk_id": "c" * 513,
        "chunk_index": 0,
    }
    with pytest.raises(ValidationError, match="UTF-8 byte bound"):
        RecallReceipt.model_validate(oversized_chunk)

    transcript_document = _receipt_document()
    transcript_items = transcript_document["items"]
    assert isinstance(transcript_items, list)
    transcript_session_id = "s" * 2_048
    content_sha256 = _digest("full-bound-transcript")
    transcript_items[0]["identity"] = {
        "record_type": "transcript_message",
        "record_id": f"{transcript_session_id}:0",
        "revision": content_sha256,
    }
    transcript_items[0]["content_sha256"] = content_sha256
    transcript_items[0]["locator"] = {
        "kind": "transcript_message",
        "session_id": transcript_session_id,
        "transcript_index": 0,
        "text_part_indexes": [0],
    }
    RecallReceipt.model_validate(transcript_document)


def test_opaque_recall_locator_retains_only_a_domain_separated_fingerprint() -> None:
    document = _receipt_document()
    items = document["items"]
    assert isinstance(items, list)
    items[0]["identity"] = {
        "record_type": "custom_memory_record",
        "record_id": "custom-contract",
        "revision": "1",
    }
    items[0]["locator"] = {
        "kind": "opaque",
        "fingerprint": _fingerprint("private-custom-locator", "recall_source_locator"),
    }

    receipt = RecallReceipt.model_validate(document)

    locator = receipt.items[0].locator
    assert isinstance(locator, OpaqueRecallEvidenceLocator)
    assert set(locator.model_dump(mode="json")) == {"kind", "fingerprint"}

    reserved_identity = _receipt_document()
    reserved_items = reserved_identity["items"]
    assert isinstance(reserved_items, list)
    reserved_items[0]["locator"] = {
        "kind": "opaque",
        "fingerprint": _fingerprint("private-custom-locator", "recall_source_locator"),
    }
    with pytest.raises(ValidationError, match="conflicts with its canonical identity"):
        RecallReceipt.model_validate(reserved_identity)

    wrong_domain = _receipt_document()
    wrong_domain_items = wrong_domain["items"]
    assert isinstance(wrong_domain_items, list)
    wrong_domain_items[0]["locator"] = {
        "kind": "opaque",
        "fingerprint": _fingerprint("private-custom-locator", "recall_situation"),
    }
    with pytest.raises(ValidationError, match="wrong keyed-fingerprint domain"):
        RecallReceipt.model_validate(wrong_domain)

    embedded_payload = _receipt_document()
    embedded_payload_items = embedded_payload["items"]
    assert isinstance(embedded_payload_items, list)
    embedded_payload_items[0]["locator"] = {
        "kind": "opaque",
        "fingerprint": _fingerprint("private-custom-locator", "recall_source_locator"),
        "payload": {"api_key": "sk-private-locator-secret"},
    }
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RecallReceipt.model_validate(embedded_payload)


def test_recall_receipt_freezes_closed_safe_locator_values() -> None:
    receipt = RecallReceipt.model_validate(_receipt_document())
    locator = receipt.items[0].locator
    assert isinstance(locator, KnowledgeEntryEvidenceLocator)

    with pytest.raises(ValidationError, match="Instance is frozen"):
        locator.entry_id = "changed"  # type: ignore[misc]


def test_recall_receipt_rejects_locator_model_subclasses() -> None:
    class PrivateLocator(KnowledgeEntryEvidenceLocator):
        api_key: str

    document = _receipt_document()
    items = document["items"]
    assert isinstance(items, list)
    items[0]["locator"] = PrivateLocator(
        entry_id="entry-contract",
        entry_revision=1,
        api_key="sk-private-locator-secret",
    )

    with pytest.raises(TypeError, match="must not be a RecallEvidenceLocator subclass"):
        RecallReceipt.model_validate(document)


def test_provider_acknowledgement_does_not_require_an_unsafe_provider_id() -> None:
    occurred_at = datetime(2026, 8, 22, tzinfo=UTC)
    transition = ContextExposureTransition(
        transition_id="transition-ack",
        revision=3,
        state="acknowledged",
        occurred_at=occurred_at,
        evidence_kind="provider_acknowledgement",
        evidence_ref="provider-operation-evidence",
    )
    assert transition.provider_request_id is None

    bounded_references = ContextExposureTransition(
        transition_id="transition-bounded-ack",
        revision=3,
        state="acknowledged",
        occurred_at=occurred_at,
        evidence_kind="provider_acknowledgement",
        evidence_ref="e" * 512,
        provider_request_id="p" * 512,
    )
    assert len(bounded_references.evidence_ref) == 512

    with pytest.raises(ValidationError, match="Only provider acknowledgement"):
        ContextExposureTransition(
            transition_id="transition-planned",
            revision=0,
            state="planned",
            occurred_at=occurred_at,
            evidence_kind="composition_planned",
            evidence_ref="composition-plan",
            provider_request_id="provider-request-premature",
        )


def test_context_exposure_cannot_combine_different_provider_requests() -> None:
    occurred_at = datetime(2026, 8, 22, tzinfo=UTC)
    transitions = (
        ContextExposureTransition(
            transition_id="transition-planned",
            revision=0,
            state="planned",
            occurred_at=occurred_at,
            evidence_kind="composition_planned",
            evidence_ref="composition-plan",
        ),
        ContextExposureTransition(
            transition_id="transition-prepared",
            revision=1,
            state="prepared",
            occurred_at=occurred_at,
            evidence_kind="request_prepared",
            evidence_ref="prepared-request",
        ),
        ContextExposureTransition(
            transition_id="transition-dispatch",
            revision=2,
            state="dispatch_started",
            occurred_at=occurred_at,
            evidence_kind="dispatch_intent_committed",
            evidence_ref="dispatch-intent",
        ),
        ContextExposureTransition(
            transition_id="transition-ack",
            revision=3,
            state="acknowledged",
            occurred_at=occurred_at,
            evidence_kind="provider_acknowledgement",
            evidence_ref="provider-ack",
            provider_request_id="provider-request-a",
        ),
        ContextExposureTransition(
            transition_id="transition-completed",
            revision=4,
            state="completed",
            occurred_at=occurred_at,
            evidence_kind="provider_completion",
            evidence_ref="provider-completion",
            provider_request_id="provider-request-b",
        ),
    )
    with pytest.raises(ValidationError, match="different provider requests"):
        ContextExposure(
            exposure_id="exposure-contract",
            session_id="session-contract",
            interaction_id="interaction-contract",
            model_step_id=f"mstep_{'1' * 32}",
            model_attempt_id=f"matt_{'2' * 32}",
            provider_attempt_id=f"patt_{'3' * 32}",
            provider_name="scripted",
            model_name="scripted-model",
            composition_fingerprint=_fingerprint("composition", "context_composition"),
            execution_profile_fingerprint=_fingerprint("profile", "execution_profile"),
            context_policy_fingerprint=_fingerprint("context", "context_policy"),
            tool_exposure_fingerprint=_fingerprint("tools", "tool_exposure"),
            request_contract_fingerprint=_fingerprint("request", "provider_request_contract"),
            created_at=occurred_at,
            updated_at=occurred_at,
            state="completed",
            state_revision=4,
            transitions=transitions,
        )


def test_context_exposure_requires_one_key_identity() -> None:
    occurred_at = datetime(2026, 8, 22, tzinfo=UTC)
    planned = ContextExposureTransition(
        transition_id="transition-planned",
        revision=0,
        state="planned",
        occurred_at=occurred_at,
        evidence_kind="composition_planned",
        evidence_ref="composition-plan",
    )
    with pytest.raises(ValidationError, match="must use one key identity"):
        ContextExposure(
            exposure_id="exposure-contract",
            session_id="session-contract",
            interaction_id="interaction-contract",
            model_step_id=f"mstep_{'1' * 32}",
            model_attempt_id=f"matt_{'2' * 32}",
            provider_attempt_id=f"patt_{'3' * 32}",
            provider_name="scripted",
            model_name="scripted-model",
            composition_fingerprint=_fingerprint("composition", "context_composition"),
            execution_profile_fingerprint={
                **_fingerprint("profile", "execution_profile"),
                "key_id": "rotated-memory-evidence-test",
            },
            context_policy_fingerprint=_fingerprint("context", "context_policy"),
            tool_exposure_fingerprint=_fingerprint("tools", "tool_exposure"),
            request_contract_fingerprint=_fingerprint("request", "provider_request_contract"),
            created_at=occurred_at,
            updated_at=occurred_at,
            state="planned",
            state_revision=0,
            transitions=(planned,),
        )


def test_memory_evidence_accepts_the_session_store_id_byte_contract() -> None:
    maximum_session_id = "é" * 1_024
    query = RecallEvidenceQuery(session_id=maximum_session_id)
    assert query.session_id == maximum_session_id

    with pytest.raises(ValidationError, match="UTF-8 byte bound"):
        RecallEvidenceQuery(session_id=f"{maximum_session_id}é")


def test_recall_receipt_page_rejects_contradictory_pagination_state() -> None:
    receipt = RecallReceipt.model_validate(_receipt_document())

    with pytest.raises(ValidationError, match="cursor does not match"):
        RecallReceiptPage(
            items=(receipt,),
            truncated=False,
            next_cursor="cursor",
        )
    with pytest.raises(ValidationError, match="cannot advance an empty"):
        RecallReceiptPage(
            items=(),
            truncated=True,
            next_cursor="cursor",
        )


def test_recall_evidence_cursor_rejects_boolean_version_alias() -> None:
    query = RecallEvidenceQuery(session_id="session-contract")
    cursor = encode_recall_evidence_cursor(
        record_kind="receipt",
        query_fingerprint=query.fingerprint("receipt"),
        created_at=datetime(2026, 8, 22, tzinfo=UTC),
        record_id="receipt-contract",
    )
    payload = json.loads(base64.urlsafe_b64decode(cursor))
    payload["version"] = True
    malformed = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")

    with pytest.raises(ValueError, match="invalid"):
        decode_recall_evidence_cursor(
            malformed,
            record_kind="receipt",
            query_fingerprint=query.fingerprint("receipt"),
        )
