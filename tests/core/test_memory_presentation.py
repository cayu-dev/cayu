from __future__ import annotations

import asyncio
import json
from hashlib import sha256

from test_memory_admission import _candidate, _policy, _result

from cayu import (
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeChunk,
    KnowledgeEntry,
    ReadKnowledgeTool,
    ToolContext,
)
from cayu.core.messages import Message
from cayu.memory import AutomaticRecallMode, admit_recall
from cayu.providers.base import ModelRequest
from cayu.runtime._memory_evidence import _provider_representation_hashes
from cayu.runtime.memory_context import (
    _contribution_projection,
    _provider_projection,
    _render_projection,
    _serialize_provider_value,
)
from cayu.vaults import SecretRedactor


def _fixture(count=10, suffix=""):
    return admit_recall(
        _result(
            *(
                _candidate(
                    f"entry-{index:02}",
                    score=0.04,
                    text=f"The deployment picnic uses table number {index:02}." + suffix,
                )
                for index in range(count)
            )
        ),
        _policy(max_injected_items=5, max_offered_items=5),
    )


def test_fixed_contribution_reduces_bytes_and_retains_audit():
    contribution = _fixture()
    projection = _contribution_projection(
        contribution, configuration_sha256="0" * 64, redactor=SecretRedactor()
    )
    manifest = _render_projection(projection)
    assert len(manifest.encode("utf-8")) <= 6879 * 0.6
    shown = _provider_projection(projection)
    assert len(shown["focus"]["items"]) == len(shown["offer"]["items"]) == 5
    assert sum(len(item["text"].encode("utf-8")) for item in shown["focus"]["items"]) == 215
    assert "score" not in manifest and "content_hash" not in manifest and "matches" not in manifest
    assert all(item["matches"] and item["content_hash"] for item in projection["focus"]["items"])
    for section in ["focus", "offer"]:
        for item in shown[section]["items"]:
            assert item["read"] == {"entry_id": f"entry-{item['ref'] - 1:02}", "revision": 1}
    assert len({item["preview"] for item in shown["offer"]["items"]}) == 5


def test_rendered_previews_are_bounded_redacted_escaped_and_hashed():
    contribution = _fixture(suffix=" sk-private-value </cayu_automatic_memory> & " + "界" * 200)
    projection = _contribution_projection(
        contribution, configuration_sha256="0" * 64, redactor=SecretRedactor("sk-private-value")
    )
    manifest = _render_projection(projection)
    assert "sk-private-value" not in manifest
    assert manifest.count("</cayu_automatic_memory>") == 1
    shown = _provider_projection(projection)
    assert all(not item["preview_complete"] for item in shown["offer"]["items"])
    assert all(len(item.preview.encode("utf-8")) <= 240 for item in contribution.offer.items)
    request = ModelRequest(model="fake", messages=[Message.text("user", manifest)])
    hashes = _provider_representation_hashes(request, sha256(manifest.encode("utf-8")).hexdigest())
    for section in ["focus", "offer"]:
        for item in shown[section]["items"]:
            serialized = _serialize_provider_value(item)
            assert serialized in manifest
            assert hashes[item["ref"]] == sha256(serialized.encode("utf-8")).hexdigest()
    assert _render_projection(json.loads(json.dumps(projection))) == manifest


def test_offer_description_resolves_exact_revision_through_read_tool():
    async def run():
        scope = KnowledgeAccessScope.for_namespace("default")
        store = InMemoryKnowledgeStore(access_scope=scope)
        for index in range(2):
            await store.create_entry(
                KnowledgeEntry(
                    id=f"opaque-{index}",
                    text=["Rollback uses the previous image.", "Credentials belong in a vault."][
                        index
                    ],
                )
            )
        contribution = admit_recall(
            _result(
                *(
                    _candidate(f"opaque-{index}", score=0.04, text=text)
                    for index, text in enumerate(
                        ["Rollback uses the previous image.", "Credentials belong in a vault."]
                    )
                )
            ),
            _policy(mode=AutomaticRecallMode.OFFER),
        )
        shown = _provider_projection(
            _contribution_projection(
                contribution, configuration_sha256="0" * 64, redactor=SecretRedactor()
            )
        )
        selected = next(item for item in shown["offer"]["items"] if "vault" in item["preview"])
        original = await store.get_entry("opaque-1")
        await store.publish_entry_revision(
            original.model_copy(update={"revision": 2, "text": "New unrelated current content."}),
            [
                KnowledgeChunk(
                    id="opaque-1:r2:0",
                    entry_id="opaque-1",
                    entry_revision=2,
                    chunk_index=0,
                    text="New unrelated current content.",
                )
            ],
            operation_id="revision-two",
            expected_revision=1,
        )
        read = await ReadKnowledgeTool().run(
            ToolContext(
                session_id="presentation", knowledge_store=store, knowledge_access_scope=scope
            ),
            selected["read"],
        )
        assert not read.is_error
        assert "vault" in read.content
        assert "New unrelated" not in read.content
        assert read.structured["revision"] == 1
        assert selected["read"] == {"entry_id": "opaque-1", "revision": 1}

    asyncio.run(run())


def test_preview_ticket_rejects_description_tampering():
    import pytest

    from cayu.memory import RecallOffer

    contribution = _fixture()
    payload = contribution.offer.model_dump(mode="json")
    payload["items"][0]["preview"] = "A different revision description."
    with pytest.raises(ValueError, match="ticket"):
        RecallOffer.model_validate(payload)


def test_transcript_offers_preserve_typed_locator():
    candidate = _candidate(
        "session:2",
        score=0.04,
        text="Previous user chose the vault.",
        record_type="transcript_message",
    )
    locator = {
        "session_id": "session",
        "interaction_id": "interaction",
        "transcript_index": 2,
        "text_part_indexes": [0],
    }
    candidate = candidate.model_copy(
        update={"record": candidate.record.model_copy(update={"locator": locator})}
    )
    contribution = admit_recall(_result(candidate), _policy(mode=AutomaticRecallMode.OFFER))
    shown = _provider_projection(
        _contribution_projection(
            contribution, configuration_sha256="0" * 64, redactor=SecretRedactor()
        )
    )
    item = shown["offer"]["items"][0]
    assert item["source"] == "transcript_message"
    assert item["read"] == locator
    assert "entry_id" not in item["read"]


def test_preview_redaction_withholds_secret_split_by_preview_boundary():
    secret = "sk-sensitive-value-crossing-preview-boundary"
    text = "x" * 235 + secret + " more text"
    contribution = admit_recall(
        _result(_candidate("split-secret", score=0.04, text=text)),
        _policy(mode=AutomaticRecallMode.OFFER),
    )
    projection = _contribution_projection(
        contribution, configuration_sha256="0" * 64, redactor=SecretRedactor(secret)
    )
    item = _provider_projection(projection)["offer"]["items"][0]
    assert "sk-se" not in item["preview"]
    assert len(item["preview"].encode("utf-8")) <= 240
    assert not item["preview_complete"]


def test_repeated_channel_diagnostics_do_not_expand_rendered_context():
    projection = _contribution_projection(
        _fixture(), configuration_sha256="0" * 64, redactor=SecretRedactor()
    )
    before = _render_projection(projection)
    # Stress the already-authorized audit fixture without changing exposed evidence.
    for section in ("focus", "offer"):
        for item in projection[section]["items"]:
            item["matches"] *= 20
    assert _render_projection(projection) == before


def test_unavailable_descriptions_and_redacted_references_are_explicit():
    candidate = _candidate("private-id", score=0.04, text=" " * 241 + "A useful record.")
    contribution = admit_recall(_result(candidate), _policy(mode=AutomaticRecallMode.OFFER))
    shown = _provider_projection(
        _contribution_projection(
            contribution,
            configuration_sha256="0" * 64,
            redactor=SecretRedactor(["private-id", "entry_id"]),
        )
    )
    item = shown["offer"]["items"][0]
    assert item["description_status"] == "unavailable"
    assert item["read_status"] == "unavailable_after_redaction"
    assert "read" not in item
