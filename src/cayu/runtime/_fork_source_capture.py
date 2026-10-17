"""Store-detached fork-source snapshots and their public failures."""

from __future__ import annotations

import asyncio
import traceback as traceback_module
from typing import Any

from cayu.runtime import _session_request_boundary as session_request_boundary
from cayu.runtime._fork_source_snapshot import (
    fork_source_checkpoint_projection,
    fork_source_checkpoint_sha256,
)
from cayu.sessions.authority import SessionRunFenced
from cayu.sessions.forks import ForkSourceSnapshot
from cayu.sessions.records import Session, _fork_source_session_instance_fingerprint
from cayu.sessions.transcript_queries import TranscriptSnapshot, fork_source_transcript_sha256
from cayu.vaults.redaction import SecretRedactor


def _clear_untrusted_exception_traceback(error: BaseException) -> None:
    """Best-effort frame clearing that cannot replace the public failure."""

    try:
        exception_traceback = error.__traceback__
        if exception_traceback is not None:
            traceback_module.clear_frames(exception_traceback)
    except BaseException:
        return


def _render_fork_source_snapshot_failure(
    value: object,
    *,
    redactor: SecretRedactor,
) -> str:
    """Best-effort public rendering for an untrusted store failure."""

    try:
        return redactor.redact_text(str(value)).strip()
    except BaseException as rendering_error:
        _clear_untrusted_exception_traceback(rendering_error)
        return ""


def _render_fork_source_snapshot_key_failure(
    error: KeyError,
    *,
    redactor: SecretRedactor,
    fallback: str,
) -> str:
    """Render a conventional KeyError argument without trusting its subclass."""

    try:
        arguments = error.args
        if len(arguments) == 1 and type(arguments[0]) is str:
            return _render_fork_source_snapshot_failure(arguments[0], redactor=redactor)
    except BaseException as rendering_error:
        _clear_untrusted_exception_traceback(rendering_error)
    return fallback


def _detached_fork_source_snapshot_failure(
    error: BaseException,
    *,
    redactor: SecretRedactor,
) -> BaseException:
    """Return a public failure without retaining source authority or store frames."""

    _clear_untrusted_exception_traceback(error)
    if isinstance(error, asyncio.CancelledError):
        return asyncio.CancelledError()
    message = _render_fork_source_snapshot_failure(error, redactor=redactor)
    if isinstance(error, SessionRunFenced):
        return SessionRunFenced(message or "Fork source changed during exact snapshot capture.")
    if isinstance(error, KeyError):
        key_message = _render_fork_source_snapshot_key_failure(
            error,
            redactor=redactor,
            fallback=message,
        )
        return KeyError(key_message or "Fork source session was not found.")
    if isinstance(error, ValueError):
        return ValueError(message or "Fork source snapshot authority is invalid.")
    if isinstance(error, RuntimeError):
        return RuntimeError(message or "Fork source snapshot failed.")
    return RuntimeError("Fork source snapshot failed.")


def _fork_source_snapshot_from_material(
    source: Session,
    checkpoint: dict[str, Any] | None,
    transcript: TranscriptSnapshot,
    *,
    redactor: SecretRedactor,
) -> ForkSourceSnapshot:
    """Validate and digest one store-detached fork-source materialization."""

    messages = tuple(record.message for record in transcript.records)
    checkpoint_projection: dict[str, Any] = {}
    try:
        if not session_request_boundary.fork_transcript_is_secret_free(
            messages,
            redactor=redactor,
        ):
            raise ValueError(
                "Fork source transcript contains a workload secret and cannot be copied."
            ) from None
        checkpoint_projection = fork_source_checkpoint_projection(checkpoint)
        if not session_request_boundary.fork_checkpoint_is_secret_free(
            checkpoint_projection,
            redactor=redactor,
        ):
            raise ValueError(
                "Fork source checkpoint contains a workload secret and cannot be copied."
            ) from None
        _, profile, _ = session_request_boundary.prepare_fork_source_execution_profile(
            source,
            checkpoint,
        )
        return ForkSourceSnapshot(
            source_session_id=source.id,
            source_instance_fingerprint=_fork_source_session_instance_fingerprint(source),
            status=source.status,
            run_epoch=source.run_epoch,
            transcript_cursor=transcript.cursor,
            transcript_sha256=fork_source_transcript_sha256(transcript),
            checkpoint_sha256=fork_source_checkpoint_sha256(checkpoint),
            execution_profile_fingerprint=profile.fingerprint,
            causal_budget_id=source.causal_budget_id,
        )
    finally:
        messages = ()
        checkpoint_projection.clear()
