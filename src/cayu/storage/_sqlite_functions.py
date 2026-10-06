"""Cayu scalar functions and exact accounting aggregation for SQLite."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, cast

from cayu._validation import require_clean_nonblank, require_execution_unit_id
from cayu.messages import Message
from cayu.sessions.base import (
    TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
    transcript_search_document,
    transcript_search_session_token,
)


def _register_sqlite_functions(connection: sqlite3.Connection) -> None:
    from cayu.sessions.pending_actions import pending_action_lookup_key

    def lookup_key(value: object) -> str | None:
        return pending_action_lookup_key(value) if type(value) is str else None

    def is_clean_nonblank_text(value: object) -> int:
        if type(value) is not str:
            return 0
        try:
            require_clean_nonblank(value, "value")
        except ValueError:
            return 0
        return 1

    def is_execution_unit_id(value: object, field_name: object) -> int:
        if type(value) is not str or type(field_name) is not str:
            return 0
        try:
            require_execution_unit_id(value, field_name)
        except (TypeError, ValueError):
            return 0
        return 1

    def transcript_text(message_json: object) -> str:
        if type(message_json) is not str:
            raise ValueError("Transcript message JSON must be text.")
        try:
            payload = json.loads(message_json)
            message = Message.model_validate(payload)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Transcript message JSON is invalid.") from exc
        return transcript_search_document(message)

    def transcript_session_token(session_id: object) -> str:
        if type(session_id) is not str:
            raise ValueError("Transcript session id must be text.")
        return transcript_search_session_token(session_id)

    connection.create_function(
        "cayu_pending_action_lookup_key",
        1,
        lookup_key,
        deterministic=True,
    )
    connection.create_function(
        "cayu_is_clean_nonblank_text",
        1,
        is_clean_nonblank_text,
        deterministic=True,
    )
    connection.create_function(
        "cayu_is_execution_unit_id",
        2,
        is_execution_unit_id,
        deterministic=True,
    )
    connection.create_function(
        "cayu_transcript_search_document",
        1,
        transcript_text,
        deterministic=True,
    )
    connection.create_function(
        "cayu_transcript_session_token",
        1,
        transcript_session_token,
        deterministic=True,
    )
    connection.create_function(
        "cayu_transcript_search_tokenizer_version",
        0,
        lambda: TRANSCRIPT_SEARCH_TOKENIZER_VERSION,
        deterministic=True,
    )
    connection.create_function(
        "cayu_canonical_accounting_json",
        1,
        lambda value: (
            None
            if value is None
            else json.dumps(
                json.loads(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        ),
        deterministic=True,
    )
    connection.create_aggregate(
        "cayu_exact_usage_sum",
        13,
        cast("Any", _ExactUsageSum),
    )


class _ExactUsageSum:
    """Sum normalized counters and canonical outputs from a prior exact sum."""

    # A SQLite table has at most 2**63 - 1 rows and each accepted JSON integer is
    # at most 2**63 - 1, so every possible sum fits in 38 decimal digits.
    _DECIMAL_WIDTH = 38

    def __init__(self) -> None:
        self._totals = [0] * 13

    def step(self, *values: object) -> None:
        for index, value in enumerate(values):
            if type(value) is int and value >= 0:
                self._totals[index] += value
            elif (
                type(value) is str
                and len(value) == self._DECIMAL_WIDTH
                and value.isascii()
                and value.isdecimal()
            ):
                self._totals[index] += int(value)

    def finalize(self) -> str:
        return json.dumps(
            [str(total).zfill(self._DECIMAL_WIDTH) for total in self._totals],
            separators=(",", ":"),
        )
