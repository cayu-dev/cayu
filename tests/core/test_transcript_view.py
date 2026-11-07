"""Properties of the durable-to-model transcript view."""

from __future__ import annotations

from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cayu.messages import (
    Message,
    MessageRole,
    ProviderStatePart,
    TextPart,
    ThinkingPart,
)
from cayu.runtime import _model_target as model_target
from cayu.runtime import _transcript
from cayu.runtime._transcript_view import (
    DurableTranscript,
    ModelTranscriptView,
    model_transcript_view,
    projection_position_map,
    remap_model_positions_in_checkpoint,
)
from cayu.sessions import TranscriptRecord, TranscriptSnapshot

_KINDS = ("user", "text", "opaque", "thinking", "mixed")


def _message(kind: str, index: int) -> Message:
    state = ProviderStatePart(provider="source", state={"id": f"r{index}"})
    if kind == "user":
        return Message.text("user", f"user {index}")
    parts = {
        "text": (TextPart(text=f"answer {index}"),),
        "opaque": (state,),
        "thinking": (ThinkingPart(text=f"reasoning {index}"),),
        "mixed": (ThinkingPart(text="reasoning"), TextPart(text=f"answer {index}"), state),
    }[kind]
    return Message(role=MessageRole.ASSISTANT, content=parts)


@st.composite
def _histories(draw):
    kinds = draw(st.lists(st.sampled_from(_KINDS), max_size=12))
    retained = [
        index for index in range(len(kinds)) if draw(st.booleans()) or index == len(kinds) - 1
    ]
    # Physical retention drops the oldest rows; the property also holds for any gap.
    if draw(st.booleans()):
        start = draw(st.integers(min_value=0, max_value=len(kinds)))
        retained = list(range(start, len(kinds)))
    snapshot = TranscriptSnapshot(
        records=[
            TranscriptRecord(index=index, message=_message(kinds[index], index))
            for index in retained
        ],
        cursor=len(kinds),
    )
    cursors = sorted(
        draw(st.lists(st.integers(min_value=0, max_value=len(kinds)), min_size=1, max_size=3))
    )
    return kinds, snapshot, cursors


def _omitted(kind: str) -> bool:
    return kind in {"opaque", "thinking"}


@settings(max_examples=300, deadline=None)
@given(_histories())
def test_view_matches_the_snapshot_projection_and_keeps_durable_identity(history) -> None:
    kinds, snapshot, cursors = history
    durable = DurableTranscript.from_snapshot(snapshot)
    retained = {record.index for record in snapshot.records}
    for cursor in cursors:
        view = model_transcript_view(durable, cursor)
        assert (
            view.messages == _transcript._project_model_target_snapshot(snapshot, cursor).messages
        )
        for index in range(snapshot.cursor):
            position = view.model_position(index)
            if index not in retained or (index < cursor and _omitted(kinds[index])):
                assert position is None
            else:
                assert position is not None and view.durable_index(position) == index
                assert view.messages[position].role.value == (
                    "user" if kinds[index] == "user" else "assistant"
                )
        assert view.model_position(snapshot.cursor) == len(view.messages)


@settings(max_examples=300, deadline=None)
@given(_histories(), st.integers(min_value=0, max_value=12))
def test_tail_start_is_the_durable_start_of_a_contiguous_tail(history, count) -> None:
    _, snapshot, cursors = history
    view = model_transcript_view(DurableTranscript.from_snapshot(snapshot), cursors[0])
    tail = view.durable_indexes[len(view.durable_indexes) - count :] if count else ()
    if count <= len(view.durable_indexes) and tail == tuple(
        range(snapshot.cursor - count, snapshot.cursor)
    ):
        assert view.tail_start(count) == snapshot.cursor - count
    else:
        with pytest.raises(ValueError):
            view.tail_start(count)


@settings(max_examples=300, deadline=None)
@given(_histories())
def test_position_maps_compose_and_match_the_user_row_zip(history) -> None:
    _, snapshot, cursors = history
    durable = DurableTranscript.from_snapshot(snapshot)
    views = [model_transcript_view(durable, cursor) for cursor in (*cursors, snapshot.cursor)]
    for earlier, later in pairwise(views):
        forward = earlier.position_map_to(later)

        def users(view: ModelTranscriptView) -> list[int]:
            return [
                position
                for position, message in enumerate(view.messages)
                if message.role is MessageRole.USER
            ]

        user_zip = dict(zip(users(earlier), users(later), strict=True))
        assert {position: forward[position] for position in user_zip} == user_zip
        assert forward[len(earlier.messages)] == len(later.messages)
        for position, mapped in forward.items():
            if position == len(earlier.messages):
                continue
            assert later.durable_index(mapped) == earlier.durable_index(position)
    first, middle, last = views[0], views[len(views) // 2], views[-1]
    through = first.position_map_to(middle)
    onward = middle.position_map_to(last)
    composed = {old: onward[new] for old, new in through.items() if new in onward}
    assert composed == {
        old: new for old, new in first.position_map_to(last).items() if old in through
    }


@settings(max_examples=300, deadline=None)
@given(_histories())
def test_in_memory_reprojection_map_matches_the_durable_map(history) -> None:
    _, snapshot, cursors = history
    durable = DurableTranscript.from_snapshot(snapshot)
    previous = model_transcript_view(durable, cursors[0])
    reprojected = model_target.project_portable_transcript(list(previous.messages))
    full = model_transcript_view(durable, snapshot.cursor)
    assert reprojected.messages == full.messages
    assert projection_position_map(
        reprojected, source_length=len(previous.messages)
    ) == previous.position_map_to(full)


def test_extended_rows_take_the_next_durable_indexes() -> None:
    snapshot = TranscriptSnapshot(
        records=[
            TranscriptRecord(index=3, message=Message.text("user", "kept")),
            TranscriptRecord(index=4, message=_message("opaque", 4)),
        ],
        cursor=5,
    )
    extended = DurableTranscript.from_snapshot(snapshot).extended(
        [Message.text("user", "appended")]
    )
    assert extended.indexes == (3, 4, 5) and extended.cursor == 6
    view = model_transcript_view(extended, 5)
    assert view.durable_indexes == (3, 5)
    assert view.tail_start(1) == 5
    assert view.model_position(4) is None
    with pytest.raises(ValueError):
        model_transcript_view(extended, 7)


def test_durable_transcript_rejects_model_facing_shapes() -> None:
    with pytest.raises(TypeError):
        DurableTranscript(messages=[Message.text("user", "x")], indexes=(0,), cursor=1)
    with pytest.raises(ValueError):
        DurableTranscript(messages=(Message.text("user", "x"),), indexes=(1,), cursor=1)
    with pytest.raises(TypeError):
        model_transcript_view([Message.text("user", "x")], 0)


@pytest.mark.parametrize(
    ("cursor", "expected"),
    [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (5, 3), (6, -1), (-1, -1)],
)
def test_compaction_cursor_moves_after_the_last_covered_kept_row(cursor, expected) -> None:
    # Five old rows; old positions 1 and 4 were omitted by the new projection.
    position_map = {0: 0, 2: 1, 3: 2, 5: 3}
    checkpoint = {
        "context_compaction": {"version": 1, "summary": "s", "compacted_transcript_cursor": cursor},
        "automatic_recall": {"anchor_transcript_index": 1},
    }
    remapped = remap_model_positions_in_checkpoint(checkpoint, position_map=position_map)
    assert remapped["context_compaction"]["compacted_transcript_cursor"] == expected
    assert remapped["automatic_recall"]["anchor_transcript_index"] == -1
    assert checkpoint["context_compaction"]["compacted_transcript_cursor"] == cursor


def test_reverse_remap_does_not_mark_omitted_rows_as_compacted() -> None:
    # Back from three new rows to five old ones, where old rows 1 and 4 are omitted.
    reverse = {0: 0, 1: 2, 2: 3, 3: 5}
    checkpoint = {"context_compaction": {"compacted_transcript_cursor": 1}}
    remapped = remap_model_positions_in_checkpoint(checkpoint, position_map=reverse)
    assert remapped["context_compaction"]["compacted_transcript_cursor"] == 1
