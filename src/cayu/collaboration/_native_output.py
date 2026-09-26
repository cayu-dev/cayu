"""Native output evidence; never an execution, publication or disclosure grant."""

from __future__ import annotations

from dataclasses import dataclass

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._session_export_store import source_digest
from cayu.sessions.base import SessionStore, TranscriptRecord, runtime_publication_request_digest


@dataclass(frozen=True, slots=True)
class NativeOutputEvidence:
    stage_id: str
    publication_id: str
    publication_sha256: str
    invocation_id: str
    source_commitment: str


def validate_output_selection(stage_id: str, source_indices: tuple[int, ...]) -> None:
    if (
        type(stage_id) is not str
        or not stage_id.strip()
        or len(stage_id) > 256
        or type(source_indices) is not tuple
        or not 0 < len(source_indices) <= 16
        or any(type(index) is not int or not 0 <= index < 2**53 for index in source_indices)
        or tuple(sorted(set(source_indices))) != source_indices
    ):
        raise CollaborationConflict("Reply production selection is invalid.")


async def read_native_output(
    store: SessionStore,
    *,
    session_id: str,
    session_instance_id: str,
    invocation_id: str,
    run_epoch: int,
    stage_id: str,
    source_indices: tuple[int, ...],
) -> NativeOutputEvidence:
    """Read exact committed evidence for an independently authenticated attachment.

    Callers own attachment authentication, retention and disclosure authorization.
    This read-only helper never selects latest output or authorizes redispatch.
    Private message parts are removed before computing the export commitment.
    """
    validate_output_selection(stage_id, source_indices)
    if (
        any(
            type(value) is not str or not value.strip()
            for value in (session_id, session_instance_id, invocation_id)
        )
        or type(run_epoch) is not int
        or not 0 < run_epoch < 2**53
    ):
        raise CollaborationConflict("Native production identity is invalid.")
    session = await store.load(session_id)
    if session is None or session.instance_id != session_instance_id:
        raise CollaborationConflict("Reply production session incarnation is unavailable.")
    stage = await store.load_model_completion_stage(session_id, stage_id)
    dispatch = await store.load_model_completion_stage_dispatch(session_id, stage_id)
    if (
        stage is None
        or stage.purpose != "assistant-turn"
        or stage.state != "completed"
        or stage.publication is None
        or dispatch is None
        or dispatch.preparation_digest != stage.preparation_digest
        or dispatch.interaction_id != invocation_id
        or dispatch.source_run_epoch != run_epoch
        or stage.source_run_epoch != run_epoch
        or stage.publication.interaction_id != invocation_id
    ):
        raise CollaborationConflict("Reply lacks exact completed model production evidence.")
    publication = stage.publication
    receipt = await store.load_runtime_publication_receipt(session_id, publication.publication_id)
    if (
        receipt is None
        or receipt.request_digest != runtime_publication_request_digest(session_id, publication)
        or receipt.interaction_id != invocation_id
        or receipt.source_run_epoch != run_epoch
        or receipt.transcript_end_cursor - receipt.transcript_start_cursor
        != len(publication.transcript_messages)
    ):
        raise CollaborationConflict("Reply model publication is not durably committed.")
    rows = []
    for index in source_indices:
        offset = index - receipt.transcript_start_cursor
        if not 0 <= offset < len(publication.transcript_messages):
            raise CollaborationConflict("Reply selects content outside its model publication.")
        message = publication.transcript_messages[offset]
        if message.role != "assistant" or any(
            part.type not in {"text", "provider_state", "thinking"} for part in message.content
        ):
            raise CollaborationConflict("Reply requires a visible assistant text projection.")
        visible = tuple(part for part in message.content if part.type == "text")
        if not visible:
            raise CollaborationConflict("Reply model publication has no visible text.")
        rows.append(
            TranscriptRecord(
                index=index,
                interaction_id=invocation_id,
                message=message.model_copy(update={"content": visible}, deep=True),
            )
        )
    return NativeOutputEvidence(
        stage_id=stage_id,
        publication_id=publication.publication_id,
        publication_sha256=receipt.publication_digest,
        invocation_id=invocation_id,
        source_commitment=source_digest(rows),
    )
