from __future__ import annotations

import pytest

from cayu.core.messages import Message, MessageRole, TextPart, ThinkingPart
from cayu.runtime.sessions import (
    RunRequest,
    SessionIdentity,
    SessionStore,
    TranscriptSearchQuery,
    transcript_search_score,
)


async def assert_transcript_search_conformance(store: SessionStore) -> None:
    """Exercise the portable, access-safe transcript retrieval contract."""

    assert store.supports_transcript_search is True
    long_token = "x" * 3_000
    for session_id in ("recall-alpha", "recall-beta"):
        await store.create(
            RunRequest(agent_name="recall-agent", session_id=session_id, messages=[]),
            identity=SessionIdentity(provider_name="fake", model="fake-model"),
        )
    await store.append_transcript_messages(
        "recall-alpha",
        [
            Message.text("user", "Project Atlas launches Friday"),
            Message(
                role=MessageRole.ASSISTANT,
                content=(
                    ThinkingPart(text="Atlas secret reasoning marker"),
                    TextPart(text="The public Atlas schedule says Monday"),
                ),
            ),
            Message.text("system", "Atlas system-only instruction"),
        ],
        interaction_id="interaction-alpha",
    )
    await store.append_transcript_messages(
        "recall-beta",
        [
            Message.text("assistant", "Atlas belongs to another selected session"),
            Message.text("user", "Straße release marker"),
            Message.text("assistant", "foo bar boundary marker"),
            Message.text("user", "foo_bar exact marker"),
            Message.text("assistant", "Orion launch date"),
            Message.text("user", long_token),
        ],
        interaction_id="interaction-beta",
    )
    await store.append_transcript_messages(
        "recall-alpha",
        [
            Message.text("assistant", "Orion details and unrelated launch date"),
            Message.text("assistant", " ".join(["Orion"] * 1_500)),
        ],
        interaction_id="interaction-alpha",
    )

    first_query = TranscriptSearchQuery(
        text="Atlas",
        session_ids=("recall-alpha",),
        limit=1,
    )
    first = await store.search_transcript(first_query)
    assert len(first.hits) == 1
    assert first.hits[0].session_id == "recall-alpha"
    assert first.hits[0].interaction_id == "interaction-alpha"
    assert first.hits[0].transcript_index == 1
    assert first.hits[0].text == "The public Atlas schedule says Monday"
    assert first.hits[0].text_part_indexes == (1,)
    assert first.hits[0].text_complete is True
    assert first.hits[0].raw_score == transcript_search_score(
        "The public Atlas schedule says Monday",
        first_query,
    )
    assert first.matched_records_examined == 2
    assert first.truncated is True
    assert first.coverage_complete is True
    assert first.next_cursor is not None

    second = await store.search_transcript(
        first_query.model_copy(update={"cursor": first.next_cursor})
    )
    assert [hit.transcript_index for hit in second.hits] == [0]
    assert second.hits[0].text == "Project Atlas launches Friday"
    assert second.truncated is False
    assert second.coverage_complete is True
    assert second.next_cursor is None
    assert second.matched_records_examined == 2

    bounded_history = await store.search_transcript(
        TranscriptSearchQuery(
            text="Atlas",
            session_ids=("recall-alpha",),
            before_transcript_indexes={"recall-alpha": 1},
        )
    )
    assert [hit.transcript_index for hit in bounded_history.hits] == [0]
    assert bounded_history.matched_records_examined == 1
    assert bounded_history.truncated is False

    empty_history = await store.search_transcript(
        TranscriptSearchQuery(
            text="Atlas",
            session_ids=("recall-alpha",),
            before_transcript_indexes={"recall-alpha": 0},
        )
    )
    assert empty_history.hits == ()
    assert empty_history.matched_records_examined == 0
    assert empty_history.truncated is False

    thinking_only = await store.search_transcript(
        TranscriptSearchQuery(text="secret reasoning", session_ids=("recall-alpha",))
    )
    assert thinking_only.hits == ()
    assert thinking_only.truncated is False

    outside_scope = await store.search_transcript(
        TranscriptSearchQuery(text="another selected", session_ids=("recall-alpha",))
    )
    missing_scope = await store.search_transcript(
        TranscriptSearchQuery(text="another selected", session_ids=("missing-session",))
    )
    assert outside_scope.hits == missing_scope.hits == ()

    user_only = await store.search_transcript(
        TranscriptSearchQuery(
            text="Atlas",
            session_ids=("recall-alpha",),
            roles=(MessageRole.USER,),
        )
    )
    assert [hit.transcript_index for hit in user_only.hits] == [0]

    partial = await store.search_transcript(
        TranscriptSearchQuery(
            text="Atlas",
            session_ids=("recall-alpha",),
            max_bytes=8,
        )
    )
    assert partial.hits[0].text == "The publ"
    assert partial.hits[0].text_complete is False
    assert len(partial.hits[0].content_hash) == 64
    assert partial.truncated is True

    unicode_fold = await store.search_transcript(
        TranscriptSearchQuery(text="STRASSE", session_ids=("recall-beta",))
    )
    assert [hit.transcript_index for hit in unicode_fold.hits] == [1]

    lexical_boundary = await store.search_transcript(
        TranscriptSearchQuery(text="foo_bar", session_ids=("recall-beta",))
    )
    assert [hit.transcript_index for hit in lexical_boundary.hits] == [3]

    long_lexeme = await store.search_transcript(
        TranscriptSearchQuery(text=long_token, session_ids=("recall-beta",))
    )
    assert [hit.transcript_index for hit in long_lexeme.hits] == [5]

    exact_byte_fit = await store.search_transcript(
        TranscriptSearchQuery(
            text=long_token,
            session_ids=("recall-beta",),
            max_bytes=len(long_token),
        )
    )
    assert exact_byte_fit.hits[0].text_complete is True
    assert exact_byte_fit.truncated is False
    assert exact_byte_fit.next_cursor is None

    final_hit_preview = await store.search_transcript(
        TranscriptSearchQuery(
            text=long_token,
            session_ids=("recall-beta",),
            max_bytes=4,
        )
    )
    assert final_hit_preview.hits[0].text_complete is False
    assert final_hit_preview.truncated is True
    assert final_hit_preview.next_cursor is None

    relevance_ranked = await store.search_transcript(
        TranscriptSearchQuery(
            text="Orion launch date",
            session_ids=("recall-alpha", "recall-beta"),
            limit=1,
        )
    )
    assert [(hit.session_id, hit.transcript_index) for hit in relevance_ranked.hits] == [
        ("recall-beta", 4)
    ]
    assert relevance_ranked.hits[0].raw_score == transcript_search_score(
        "Orion launch date",
        relevance_ranked.query,
    )
    assert relevance_ranked.truncated is True
    assert relevance_ranked.next_cursor is not None

    scan_limited = await store.search_transcript(
        TranscriptSearchQuery(
            text="Atlas",
            session_ids=("recall-alpha",),
            limit=2,
            max_records_scanned=1,
        )
    )
    assert scan_limited.hits == ()
    assert scan_limited.matched_records_examined == 1
    assert scan_limited.truncated is True
    assert scan_limited.coverage_complete is False
    assert scan_limited.next_cursor is None

    complete_retry = await store.search_transcript(
        scan_limited.query.model_copy(update={"max_records_scanned": 2})
    )
    assert [hit.transcript_index for hit in complete_retry.hits] == [1, 0]
    assert complete_retry.coverage_complete is True

    indexed_scan_limited = await store.search_transcript(
        TranscriptSearchQuery(
            text="Friday",
            session_ids=("recall-alpha",),
            max_records_scanned=1,
        )
    )
    assert [hit.transcript_index for hit in indexed_scan_limited.hits] == [0]
    assert indexed_scan_limited.matched_records_examined == 1
    assert indexed_scan_limited.coverage_complete is True

    with pytest.raises(ValueError, match="at least one explicit session id"):
        TranscriptSearchQuery(text="Atlas", session_ids=())
    with pytest.raises(ValueError, match="between 4"):
        TranscriptSearchQuery(text="Atlas", session_ids=("recall-alpha",), max_bytes=3)
    with pytest.raises(ValueError, match="Invalid transcript search cursor"):
        await store.search_transcript(
            TranscriptSearchQuery(
                text="Friday",
                session_ids=("recall-alpha",),
                cursor=first.next_cursor,
            )
        )


__all__ = ["assert_transcript_search_conformance"]
