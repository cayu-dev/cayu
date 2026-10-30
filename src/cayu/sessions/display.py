"""Minimal bounded session-title reads for public inventory joins."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import require_durable_clean_nonblank
from cayu.session_metadata import validate_title

SESSION_TITLE_BATCH_MAX = 500


class SessionTitleQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    session_ids: tuple[str, ...] = Field(min_length=1, max_length=SESSION_TITLE_BATCH_MAX)

    @field_validator("session_ids")
    @classmethod
    def validate_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            require_durable_clean_nonblank(value, "session id")
        if len(set(values)) != len(values):
            raise ValueError("Session title IDs must be distinct.")
        return values


def copy_title_query(query: SessionTitleQuery) -> SessionTitleQuery:
    if type(query) is not SessionTitleQuery:
        raise TypeError("Session titles require a typed query.")
    return SessionTitleQuery(session_ids=query.session_ids)


def bounded_session_title(value: object) -> str | None:
    try:
        return validate_title(value)
    except ValueError:
        return None


class SessionTitleRow(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    session_id: str
    title: str | None

    @field_validator("session_id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "session id")

    @field_validator("title", mode="before")
    @classmethod
    def validate_text(cls, value: object) -> str | None:
        return validate_title(value)


class SessionTitleSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    sessions: tuple[SessionTitleRow, ...] = Field(max_length=SESSION_TITLE_BATCH_MAX)

    def validate_for_query(self, query: SessionTitleQuery) -> SessionTitleSnapshot:
        query = copy_title_query(query)
        if type(self.sessions) is not tuple or len(self.sessions) != len(query.session_ids):
            raise ValueError("Session titles did not cover the requested IDs.")
        rows = []
        for expected, row in zip(query.session_ids, self.sessions, strict=True):
            if type(row) is not SessionTitleRow or row.session_id != expected:
                raise ValueError("Session titles returned unexpected IDs.")
            rows.append(SessionTitleRow(session_id=row.session_id, title=row.title))
        return SessionTitleSnapshot(sessions=tuple(rows))


def title_snapshot(query: SessionTitleQuery, values: dict[str, object]) -> SessionTitleSnapshot:
    return SessionTitleSnapshot(
        sessions=tuple(
            SessionTitleRow(session_id=key, title=bounded_session_title(values.get(key)))
            for key in query.session_ids
        )
    )
