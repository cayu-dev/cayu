"""Bounded, credential-safe view of the model call a session's run is making.

A session's run can stop inside a model call; its active model-completion stage
is then the call's durable record. ``inspect_model_stage`` reports it by
identity and state only. It never reads a provider request, a stream delta,
recovery metadata or a publication, so it carries nothing a redactor hides.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from cayu.sessions._model_stage import ModelCompletionPurpose, ModelCompletionStage
from cayu.sessions.store import SessionStore

_MAX_IDENTIFIER_CHARS = 256


class ActiveModelStageInspection(BaseModel):
    """The model call a session's run is making, or made and hasn't published."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage_id: str = Field(max_length=_MAX_IDENTIFIER_CHARS)
    purpose: ModelCompletionPurpose
    # ``in_flight`` while no answer is recorded; ``completed`` once the provider's
    # answer is recorded but not yet published to the session.
    state: Literal["in_flight", "completed"]
    # Whether the request reached the provider (a dispatch receipt is recorded).
    dispatched: StrictBool
    # A background provider operation (for example OpenAI background mode) owns it.
    background: StrictBool
    model_step_id: str = Field(max_length=_MAX_IDENTIFIER_CHARS)
    model_attempt_id: str | None = Field(default=None, max_length=_MAX_IDENTIFIER_CHARS)
    dispatch_ordinal: StrictInt = Field(ge=0)
    provider_name: str | None = Field(default=None, max_length=_MAX_IDENTIFIER_CHARS)
    requested_model: str | None = Field(default=None, max_length=_MAX_IDENTIFIER_CHARS)
    source_run_epoch: StrictInt = Field(ge=0)
    reservation_count: StrictInt = Field(ge=0)
    prepared_at: datetime
    completed_at: datetime | None = None


def _identifier(value: Any) -> str | None:
    if type(value) is not str or not value or len(value) > _MAX_IDENTIFIER_CHARS:
        return None
    return value


async def _stage_inspection(
    store: SessionStore, stage: ModelCompletionStage
) -> ActiveModelStageInspection:
    intent = stage.intent
    dispatch = await store.load_model_completion_stage_dispatch(stage.session_id, stage.stage_id)
    return ActiveModelStageInspection(
        stage_id=stage.stage_id,
        purpose=stage.purpose,
        state=stage.state,
        dispatched=dispatch is not None,
        background=intent.get("provider_operation_start") is not None,
        # The step the call's events name. An auxiliary call's logical step is its
        # publication identity, so its intent carries the step; older stages may not.
        model_step_id=_identifier(intent.get("model_step_id")) or stage.logical_step_id,
        model_attempt_id=_identifier(intent.get("model_attempt_id")),
        dispatch_ordinal=stage.dispatch_ordinal,
        provider_name=_identifier(intent.get("provider_name")),
        requested_model=_identifier(intent.get("requested_model")),
        source_run_epoch=stage.source_run_epoch,
        reservation_count=len(stage.reservation_ids),
        prepared_at=stage.prepared_at,
        completed_at=stage.completed_at,
    )


async def inspect_model_stage(
    store: SessionStore, session_id: str
) -> ActiveModelStageInspection | None:
    """Report the session's active model stage, or ``None`` when there is none.

    Reads the active-stage marker and its dispatch receipt only, never the
    checkpoint, so it fits the bounded session state route and a read-only store.
    """

    active = await store.load_active_model_completion_stage(session_id)
    return None if active is None else await _stage_inspection(store, active.stage)
