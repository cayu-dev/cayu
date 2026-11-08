"""`GET /sessions/{id}/runtime-timing`: one process's recent runtime timing records.

The runtime times each tool round (`ToolRoundTiming`) and the gap before each model step
(`ModelStepPreparationTiming`), with the database time inside every phase, and keeps the
most recent records in the memory of the process that ran them
(`CayuApp.inspect_recent_tool_round_timing`). This route returns what the answering
process holds for one session, so an operator can read them without code in that process.

`public_runtime_timing_record` projects one record the way the public session APIs show
tool calls: a call is named by its `tool_call_correlation_id`, tool names are redacted,
and model step, attempt and round ids are withheld.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Response
from pydantic import StringConstraints

from cayu.observability.timing import ModelStepPreparationTiming, RuntimeTimingRecord
from cayu.runtime._phase_timing import RuntimeTimingRecorder
from cayu.runtime._session_execution_presence import process_owner_id
from cayu.runtime._tool_call_correlation import public_tool_call_handle
from cayu.runtime.public_authority import PublicAuthorityAliasCodec
from cayu.server.contracts import SessionRuntimeTimingResponse
from cayu.sessions.store import SessionStore

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
#: The most records one read returns; a recorder keeps at most 1024 for all sessions.
MAX_RUNTIME_TIMING_RECORDS = 1024
DEFAULT_RUNTIME_TIMING_RECORDS = 50


def public_runtime_timing_record(
    record: RuntimeTimingRecord,
    *,
    private_session_id: str,
    private_round_id: str | None,
    private_call_ids: Sequence[str],
    redact: Callable[[str], str],
    codec: PublicAuthorityAliasCodec | None,
) -> dict[str, Any]:
    """One record as the API returns it, without private model step, attempt or round ids.

    ``private_round_id`` and ``private_call_ids`` (in the order of ``record.calls``) are the
    identities the record was built from; a call's public handle is derived from them, since
    the record may carry redaction references in their place.
    """

    phases = [phase.model_dump(mode="json") for phase in record.phases]
    common = {
        "kind": record.kind,
        "started_at": record.started_at.isoformat(),
        "completed_at": record.completed_at.isoformat(),
        "duration_seconds": record.duration_seconds,
        "incomplete": record.incomplete,
        "phases": phases,
    }
    if isinstance(record, ModelStepPreparationTiming):
        return {**common, "after_tool_round": record.after_tool_round_id is not None}
    if private_round_id is None or len(private_call_ids) != len(record.calls):
        raise ValueError("A tool round's timing needs its round id and one id per call.")
    return {
        **common,
        "recovered": record.recovered,
        "calls_truncated": record.calls_truncated,
        "calls": [
            {
                "tool_call_id": public_tool_call_handle(
                    session_id=private_session_id,
                    round_id=private_round_id,
                    call_id=call_id,
                    codec=codec,
                ),
                "tool_name": redact(call.tool_name),
                "phases": [phase.model_dump(mode="json") for phase in call.phases],
                "tool_effect_completed_at": _instant(call.tool_effect_completed_at),
                "tool_terminal_staged_at": _instant(call.tool_terminal_staged_at),
                "tool_terminal_publication_started_at": _instant(
                    call.tool_terminal_publication_started_at
                ),
            }
            for call, call_id in zip(record.calls, private_call_ids, strict=True)
        ],
    }


def register_runtime_timing_route(
    router: APIRouter,
    *,
    protected: list[Any],
    cayu_app: Any,
    store: SessionStore,
    timing_recorder: Callable[[], RuntimeTimingRecorder],
    resolve_session: Callable[[str], Awaitable[str]],
    redact: Callable[[str], str],
    codec: Callable[[], PublicAuthorityAliasCodec | None],
) -> None:
    @router.get(
        "/sessions/{session_id}/runtime-timing",
        response_model=SessionRuntimeTimingResponse,
        responses={404: {"description": "The session does not exist."}},
        dependencies=protected,
        description=(
            "The recent runtime timing records this process holds for one session, oldest "
            "first, as `cayu session runtime-timing` reports them: each tool round and each "
            "model step's preparation, split into phases with the database transactions, "
            "lock and pool waits, SQL and commit time inside them. Records are kept only in "
            "the memory of the process that ran the work, in a bounded buffer shared by all "
            "sessions, so a session run by another process (such as a task worker) has none "
            "here; `session_execution` says which process runs it now."
        ),
    )
    async def get_session_runtime_timing(
        session_id: NonBlankString,
        response: Response,
        limit: Annotated[
            int, Query(ge=1, le=MAX_RUNTIME_TIMING_RECORDS)
        ] = DEFAULT_RUNTIME_TIMING_RECORDS,
    ):
        private_id = await resolve_session(session_id)
        if await store.load_state(private_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        recorder = timing_recorder()
        # The newest `limit` of both kinds, oldest first, so the table reads as a timeline.
        entries = sorted(
            recorder.inspect_entries(private_id, limit), key=lambda entry: entry.record.started_at
        )
        execution = await cayu_app.inspect_session_execution(private_id)
        here = process_owner_id()
        status = recorder.status()
        config = recorder.config
        alias_codec = codec()
        response.headers["Cache-Control"] = "private, no-store"
        return {
            "session_id": execution.session_id,
            "scope": "process",
            "process": {"owner_id": here},
            "session_execution": {
                "state": execution.state,
                "owner_id": execution.owner_id,
                "in_this_process": execution.owner_id == here,
            },
            "recorder": {
                "enabled": config.enabled,
                "recent_capacity": config.recent_capacity,
                **status.model_dump(mode="json"),
            },
            "limit": limit,
            "records": [
                public_runtime_timing_record(
                    entry.record,
                    private_session_id=private_id,
                    private_round_id=entry.round_id,
                    private_call_ids=entry.call_ids,
                    redact=redact,
                    codec=alias_codec,
                )
                for entry in entries
            ],
        }


def _instant(value: Any) -> str | None:
    return None if value is None else value.isoformat()
