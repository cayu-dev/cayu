"""The one boundary between durable transcript rows and model-facing positions.

A transcript row has a permanent durable index (``TranscriptRecord.index``).
After a model switch, the model-facing list omits assistant rows left with only
provider state or thinking, so a position in that list is not a durable index.
Every model-facing list built from durable rows comes from
:func:`model_transcript_view`, which keeps the durable index of each message,
and every conversion between the two spaces goes through the view.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NewType

from cayu._validation import copy_durable_record
from cayu.context.base import _COMPACTION_CHECKPOINT_KEY
from cayu.messages import Message, detach_message
from cayu.runtime import _model_target as model_target
from cayu.sessions._execution_profile_rules import session_model_projection_cursor
from cayu.sessions._model_failover import ModelFailoverProgress, ModelFailoverSelection
from cayu.sessions.checkpoints import (
    AUTOMATIC_RECALL_CHECKPOINT_KEY,
    AUTOMATIC_RECALL_HISTORY_CHECKPOINT_KEY,
    RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_KEY,
)
from cayu.sessions.records import Session
from cayu.sessions.transcript_queries import TranscriptSnapshot

# Documentation only: both cross into checkpoints and events as plain ints.
DurableIndex = NewType("DurableIndex", int)
ModelPosition = NewType("ModelPosition", int)


@dataclass(frozen=True, slots=True)
class DurableTranscript:
    """Retained durable rows with their indexes and the append cursor. Never model-facing."""

    messages: tuple[Message, ...]
    indexes: tuple[DurableIndex, ...]
    cursor: DurableIndex

    def __post_init__(self) -> None:
        if type(self.messages) is not tuple or type(self.indexes) is not tuple:
            raise TypeError("A durable transcript requires tuples of messages and indexes.")
        if len(self.messages) != len(self.indexes):
            raise ValueError("A durable transcript needs one index per message.")
        if any(type(message) is not Message for message in self.messages):
            raise TypeError("A durable transcript requires exact Message instances.")
        if list(self.indexes) != sorted(set(self.indexes)) or (
            self.indexes and not 0 <= self.indexes[0] <= self.indexes[-1] < self.cursor
        ):
            raise ValueError("Durable transcript indexes must ascend below the cursor.")

    @classmethod
    def from_snapshot(cls, snapshot: TranscriptSnapshot) -> DurableTranscript:
        if type(snapshot) is not TranscriptSnapshot:
            raise TypeError("A durable transcript is read from an exact TranscriptSnapshot.")
        return cls(
            messages=tuple(detach_message(record.message) for record in snapshot.records),
            indexes=tuple(DurableIndex(record.index) for record in snapshot.records),
            cursor=DurableIndex(snapshot.cursor),
        )

    def extended(self, appended: Sequence[Message]) -> DurableTranscript:
        """Add rows that are durably appended at the cursor after this read."""

        return DurableTranscript(
            messages=(*self.messages, *(detach_message(message) for message in appended)),
            indexes=(
                *self.indexes,
                *(DurableIndex(self.cursor + offset) for offset in range(len(appended))),
            ),
            cursor=DurableIndex(self.cursor + len(appended)),
        )

    def message_list(self) -> list[Message]:
        """Detached copies of the retained rows, for durable (not model-facing) uses."""

        return [detach_message(message) for message in self.messages]


@dataclass(frozen=True, slots=True)
class ModelTranscriptView:
    """The model-facing list for one projection cursor, with each row's durable index."""

    messages: tuple[Message, ...]
    durable_indexes: tuple[DurableIndex, ...]
    durable_cursor: DurableIndex
    provider_state_parts_dropped: int
    thinking_parts_dropped: int

    def durable_index(self, position: int) -> DurableIndex:
        if type(position) is not int or not 0 <= position < len(self.durable_indexes):
            raise ValueError("Model position is outside the model-facing transcript.")
        return self.durable_indexes[position]

    def model_position(self, index: int) -> ModelPosition | None:
        """The model position of a durable row; None for an omitted or retained-away row.

        The append cursor maps to the end of the view.
        """

        if type(index) is not int or not 0 <= index <= self.durable_cursor:
            raise ValueError("Durable index exceeds the transcript cursor.")
        if index == self.durable_cursor:
            return ModelPosition(len(self.durable_indexes))
        position = bisect_left(self.durable_indexes, index)
        if position < len(self.durable_indexes) and self.durable_indexes[position] == index:
            return ModelPosition(position)
        return None

    def tail_start(self, count: int) -> DurableIndex:
        """Durable index of the first of the last ``count`` rows, which end at the cursor."""

        if type(count) is not int or not 0 <= count <= len(self.durable_indexes):
            raise ValueError("Transcript tail exceeds the model-facing transcript.")
        start = self.durable_cursor - count
        if self.durable_indexes[len(self.durable_indexes) - count :] != tuple(
            range(start, self.durable_cursor)
        ):
            raise ValueError("Transcript tail is not the contiguous end of the durable rows.")
        return DurableIndex(start)

    def position_map_to(self, other: ModelTranscriptView) -> dict[int, int]:
        """Map this view's model positions to ``other``'s through durable identity.

        A row that ``other`` omits has no entry; callers treat it as invalid. The
        end of this view maps to the end of ``other``.
        """

        if type(other) is not ModelTranscriptView:
            raise TypeError("A position map needs another ModelTranscriptView.")
        target = {index: position for position, index in enumerate(other.durable_indexes)}
        mapped = {
            position: target[index]
            for position, index in enumerate(self.durable_indexes)
            if index in target
        }
        mapped[len(self.durable_indexes)] = len(other.durable_indexes)
        return mapped


def projection_position_map(
    projection: model_target.PortableTranscriptProjection,
    *,
    source_length: int,
) -> dict[int, int]:
    """Source-list position to projected position, for an in-memory model-facing list.

    The end of the source list maps to the end of the projection.
    """

    mapped = {source: position for position, source in enumerate(projection.source_positions)}
    mapped[source_length] = len(projection.messages)
    return mapped


def effective_projection_cursor(
    session: Session,
    model_failover: ModelFailoverProgress | ModelFailoverSelection | None,
) -> DurableIndex:
    """The session's switch cursor, raised by in-run model failover progress."""

    cursor = session_model_projection_cursor(session)
    if model_failover is not None:
        cursor = max(cursor, model_failover.projection_cursor)
    return DurableIndex(cursor)


def model_transcript_view(
    transcript: DurableTranscript,
    projection_cursor: int,
) -> ModelTranscriptView:
    """The only producer of a model-facing list from durable rows."""

    if type(transcript) is not DurableTranscript:
        raise TypeError("A model transcript view is built from a DurableTranscript.")
    if type(projection_cursor) is not int or not 0 <= projection_cursor <= transcript.cursor:
        raise ValueError("Model-target projection cursor exceeds the session transcript.")
    projection = model_target.project_portable_transcript_prefix(
        list(transcript.messages),
        bisect_left(transcript.indexes, projection_cursor),
    )
    return ModelTranscriptView(
        messages=projection.messages,
        durable_indexes=tuple(transcript.indexes[source] for source in projection.source_positions),
        durable_cursor=transcript.cursor,
        provider_state_parts_dropped=projection.provider_state_parts_dropped,
        thinking_parts_dropped=projection.thinking_parts_dropped,
    )


def remap_model_positions_in_checkpoint(
    checkpoint: dict[str, Any],
    *,
    position_map: Mapping[int, int],
) -> dict[str, Any]:
    """Keep checkpoint state stored in model positions on the same rows after a projection.

    ``position_map`` maps old model positions to new ones and the old end to the new
    end (from :meth:`ModelTranscriptView.position_map_to` or
    :func:`projection_position_map`). A recall anchor addresses one row: an unmapped
    anchor becomes -1 and stays invalid. The compaction cursor is a boundary after the
    rows it covers: it moves to just after the last covered row that is still mapped,
    so a row the old list did not show is never marked as summarised. A cursor beyond
    the old end becomes -1 and stays invalid.
    """

    end = max(position_map)
    updated = copy_durable_record(checkpoint, "checkpoint")

    def remap(anchor: Any) -> None:
        if type(anchor) is dict and type(anchor.get("anchor_transcript_index")) is int:
            index = anchor["anchor_transcript_index"]
            anchor["anchor_transcript_index"] = position_map.get(index, -1) if index < end else -1

    state = updated.get(AUTOMATIC_RECALL_CHECKPOINT_KEY)
    remap(state)
    if type(state) is dict and type(state.get("runtime_authored_anchors")) is list:
        for anchor in state["runtime_authored_anchors"]:
            remap(anchor)
    remap(updated.get(RUNTIME_AUTHORED_USER_MESSAGE_CHECKPOINT_KEY))
    history = updated.get(AUTOMATIC_RECALL_HISTORY_CHECKPOINT_KEY)
    if type(history) is dict:
        for name in ("current_placements", "retained_placements"):
            if type(history.get(name)) is list:
                for placement in history[name]:
                    remap(placement)
    compaction = updated.get(_COMPACTION_CHECKPOINT_KEY)
    if type(compaction) is dict:
        cursor = compaction.get("compacted_transcript_cursor")
        if type(cursor) is int and 0 <= cursor <= end:
            compaction["compacted_transcript_cursor"] = 1 + max(
                (mapped for position, mapped in position_map.items() if position < cursor),
                default=-1,
            )
        elif type(cursor) is int:
            compaction["compacted_transcript_cursor"] = -1
    return updated
