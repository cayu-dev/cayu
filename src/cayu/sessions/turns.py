"""Optional create-or-resume convenience over the existing execution APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu.sessions.outcomes import RunOutcome, _run_to_completion
from cayu.sessions.requests import ResumeRequest, RunRequest

if TYPE_CHECKING:
    from cayu.applications import CayuApp
    from cayu.collaboration.access import CollaborationAccessContext


class TurnRequest(RunRequest):
    """Send messages to a new or existing session using normal run controls.

    Reuses RunRequest's validated fields. A missing id creates a generated
    session; a supplied id selects creation or resume once. Creation-only fields
    must be omitted when the session already exists, even if their values match.
    Omitted resume controls retain their normal inheritance semantics.
    """


async def send_turn(
    app: CayuApp, request: TurnRequest, *, context: CollaborationAccessContext | None = None
) -> RunOutcome:
    """Send one turn using native admission, authorization and abandoned-run recovery.

    Native exceptions propagate, including admission conflicts and pending actions.
    Model/tool failures emitted as terminal events return a failed RunOutcome.
    Existence is only a selection hint: native atomic creation and resume claims
    decide admission. A lost creation race is never retried as a resume.
    """
    if type(request) is not TurnRequest:
        raise TypeError("send_turn requires a TurnRequest.")
    # Revalidate and detach before awaiting the store. Include mutable defaults
    # and the deadline captured at construction, but preserve omitted controls.
    # Serializing here would lose in-process loop-policy callbacks.
    supplied_fields = request.model_fields_set.copy()
    native = RunRequest(**{name: getattr(request, name) for name in RunRequest.model_fields})
    object.__setattr__(native, "__pydantic_fields_set__", supplied_fields)
    session = None
    if native.session_id is not None:
        session_id, _ = await app._resolve_public_session_authority(native.session_id)
        session = await app.session_store.load(session_id)
    if session is None:
        return await _run_to_completion(app, native, propagate_exceptions=True)
    await app._require_participant_execution(session, context)
    if session.agent_name != native.agent_name:
        raise ValueError("agent_name must match the existing session's agent.")
    # A turn's deadline bounds a new session only; an existing session keeps its
    # own, and replacing it is an explicit ResumeRequest.execution_deadline.
    creation_only = (
        set(RunRequest.model_fields) - set(ResumeRequest.model_fields) - {"agent_name"}
    ) | {"execution_deadline"}
    supplied = creation_only & supplied_fields
    if native.labels:
        supplied.add("labels")
    if supplied:
        raise ValueError(
            "Existing sessions cannot receive creation-only settings: "
            + ", ".join(sorted(supplied))
            + ". Omit them or create a new session."
        )
    values = {
        name: getattr(native, name)
        for name in (RunRequest.model_fields.keys() & ResumeRequest.model_fields.keys())
        - creation_only
    }
    values["expected_session_instance_id"] = session.instance_id
    values["expected_run_epoch"] = session.run_epoch
    resume = ResumeRequest(**values)
    # Let native resume own which omitted controls inherit durable settings.
    object.__setattr__(
        resume,
        "__pydantic_fields_set__",
        (supplied_fields & ResumeRequest.model_fields.keys() - creation_only)
        | {"expected_session_instance_id", "expected_run_epoch"},
    )
    return await _run_to_completion(app, resume, context=context, propagate_exceptions=True)
