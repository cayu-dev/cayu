"""Authenticate reply production from native service and model publication owners."""

from __future__ import annotations

from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration._native_output import (
    NativeOutputEvidence,
    read_native_output,
    validate_output_selection,
)
from cayu.sessions._temporary_continuation import TemporaryServiceRecord
from cayu.sessions.base import SessionStore


async def authenticate_reply_production(
    store: SessionStore,
    service: TemporaryServiceRecord,
    *,
    stage_id: str,
    source_indices: tuple[int, ...],
) -> NativeOutputEvidence:
    """Authenticate service authority before reading its exact native output.

    Source export and current disclosure authorization remain separate. Ordinary
    output must authenticate its own attachment, not fabricate service evidence.
    """
    validate_output_selection(stage_id, source_indices)
    retained = await store._load_temporary_continuation_service(service.admission)
    if retained != service or service.state != "returned" or service.execution is None:
        raise CollaborationConflict("Reply requires authenticated returned service execution.")
    execution = service.execution
    return await read_native_output(
        store,
        session_id=execution.session_id,
        session_instance_id=execution.session_instance_id,
        invocation_id=execution.invocation_id,
        run_epoch=execution.run_epoch,
        stage_id=stage_id,
        source_indices=source_indices,
    )
