"""Re-redact the failure text a task store kept, after a redaction fix lands.

A task's ``error`` (and the ``error`` a pending cancellation keeps in its
``status_payload``) is redacted when it is written, with the redaction of that
time. When a later fix masks a credential shape the earlier code missed, records
written before it still hold the text. ``scrub_task_failure`` applies the
current redaction to that stored text again.

The transformation is deterministic for a given task snapshot, redaction and
time, so a store applies it to the task row and to every receipt that keeps a
copy of the task (terminalization, interrupted-handoff and retry-settlement
receipts): a snapshot equal to the row before the scrub stays equal after it,
and replaying the receipt still matches the row.

Each field that changed records the scrub under ``SCRUB_RECORD_KEY``: when, by
which redaction, and how many text values it masked, never the text itself.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from cayu._validation import copy_durable_json_object
from cayu.tasks.records import Task
from cayu.vaults import REDACTED_SECRET
from cayu.vaults._credential_shapes import mask_credential_shapes

#: The key a scrubbed field records its scrubs under.
SCRUB_RECORD_KEY = "cayu_scrub"
#: The redaction a scrub applies: the application's registered secrets, then credential shapes.
SCRUB_REDACTION = "secret_redactor+credential_shapes"
#: Scrub records kept per field; older ones are dropped first.
MAX_SCRUB_RECORDS = 20
#: The most tasks one sweep page examines.
MAX_TASK_FAILURE_SCRUB_PAGE = 100

# An Authorization value that is a whole marker, which a scrub leaves as it is.
_MASKED_AUTHORIZATION = re.compile(
    rf"(?i)authorization[\"']?[ \t]*+[:=]\s*+[\"']?{re.escape(REDACTED_SECRET)}"
    rf"(?![^\s'\",;)\]}}])"
)
# Private-use characters stand for a held value while the shapes are masked.
_HOLD_START, _HOLD_END = "\ue000", "\ue001"
_HELD = re.compile(f"{_HOLD_START}([0-9]+){_HOLD_END}")

#: Why a task was left as it was: a worker holds it (``leased``); its error is the payload
#: of a session's recorded runtime failure, which the session's append-only events and
#: checkpoint bind exactly (``session_failure``); or a cancellation reconciliation receipt
#: binds a digest of it (``cancellation_reconciliation``). Rewriting either of the last two
#: would make their replay conflict, so the text stays until those records go.
TaskFailureScrubSkip = Literal["leased", "session_failure", "cancellation_reconciliation"]
# The error key a session's runtime task failure keeps its bound evidence under
# (`cayu.runtime._continuation_task_failure`).
_RUNTIME_TASK_FAILURE_KEY = "runtime_task_failure"
# The status-payload key a reconciled cancellation keeps its digest-bound evidence under.
_CANCELLATION_RECONCILIATION_KEY = "cancellation_reconciliation"


@dataclass(frozen=True, slots=True)
class TaskFailureScrub:
    """How to scrub: the text redaction, the time to record, and whether to write."""

    redact: Callable[[str], str]
    scrubbed_at: datetime
    dry_run: bool = False


class TaskFailureScrubResult(BaseModel):
    """What a scrub of one task changed, or would change on a dry run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    #: The fields whose text changed: ``error`` and ``status_payload.error``.
    fields: tuple[str, ...] = ()
    #: How many text values the redaction changed in the task row.
    masked_values: StrictInt = Field(default=0, ge=0)
    #: How many stored receipt copies of the task were rewritten with it.
    receipts: StrictInt = Field(default=0, ge=0)
    #: Whether the store wrote the change (never on a dry run).
    applied: StrictBool = False
    skipped_reason: TaskFailureScrubSkip | None = None


def _scrub_value(value: Any, redact: Callable[[str], str]) -> tuple[Any, int]:
    if isinstance(value, str):
        scrubbed = redact(value)
        return scrubbed, int(scrubbed != value)
    if isinstance(value, list):
        items = [_scrub_value(item, redact) for item in value]
        return [item for item, _ in items], sum(count for _, count in items)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        total = 0
        for key, item in value.items():
            if key == SCRUB_RECORD_KEY:
                result[key] = item
                continue
            result[key], count = _scrub_value(item, redact)
            total += count
        return result, total
    return value, 0


def _scrub_object(value: dict[str, Any], scrub: TaskFailureScrub) -> tuple[dict[str, Any], int]:
    scrubbed, count = _scrub_value(value, scrub.redact)
    if not count:
        return value, 0
    prior = value.get(SCRUB_RECORD_KEY)
    records = list(prior.get("scrubs", ())) if isinstance(prior, dict) else []
    records.append(
        {
            "scrubbed_at": scrub.scrubbed_at.isoformat(),
            "redaction": SCRUB_REDACTION,
            "masked_values": count,
        }
    )
    scrubbed[SCRUB_RECORD_KEY] = {"scrubs": records[-MAX_SCRUB_RECORDS:]}
    return copy_durable_json_object(scrubbed, "error"), count


def scrub_task_failure(task: Task, scrub: TaskFailureScrub) -> tuple[Task, tuple[str, ...], int]:
    """``task`` with its failure text re-redacted, the fields that changed, and how many
    text values changed. A task with nothing to mask comes back unchanged."""

    update: dict[str, Any] = {}
    fields: list[str] = []
    total = 0
    if task.error is not None:
        error, count = _scrub_object(task.error, scrub)
        if count:
            update["error"] = error
            fields.append("error")
            total += count
    payload = task.status_payload
    if payload is not None and isinstance(payload.get("error"), dict):
        error, count = _scrub_object(payload["error"], scrub)
        if count:
            update["status_payload"] = {**payload, "error": error}
            fields.append("status_payload.error")
            total += count
    if not update:
        return task, (), 0
    return task.model_copy(update=update, deep=True), tuple(fields), total


def failure_text_redaction(
    redact_secrets: Callable[[Any], Any] | None = None,
) -> Callable[[str], str]:
    """The redaction a scrub applies to each text value, the one exception diagnostics use:
    the registered secrets ``redact_secrets`` removes (such as ``CayuApp.redact_json``)
    first, so masking a shape never cuts one apart, then credential shapes."""

    def redact(text: str) -> str:
        secretless = text if redact_secrets is None else redact_secrets(text)
        if type(secretless) is not str:
            raise TypeError("Secret redaction must return text.")
        if _HOLD_START in secretless:
            return mask_credential_shapes(secretless, REDACTED_SECRET)
        # An Authorization value that is already exactly a marker is set aside while the
        # shapes are masked: the shape would take the marker, and the word after it, for a
        # scheme and credential, so scrubbing again would eat the diagnostic a word at a
        # time. Everything else is masked whole, so a marker inside a longer credential
        # word (a registered secret's prefix) still takes the rest of that word with it.
        held: list[str] = []

        def hold(match: re.Match[str]) -> str:
            held.append(match.group())
            return f"{_HOLD_START}{len(held) - 1}{_HOLD_END}"

        masked = mask_credential_shapes(
            _MASKED_AUTHORIZATION.sub(hold, secretless), REDACTED_SECRET
        )
        return _HELD.sub(lambda match: held[int(match[1])], masked)

    return redact


def scrub_result_row(result: TaskFailureScrubResult, dry_run: bool) -> dict[str, Any]:
    """``result`` as the routes and the CLI report it."""

    return {**result.model_dump(mode="json"), "dry_run": dry_run}


def scrub_snapshot(task: Task, scrub: TaskFailureScrub) -> Task | None:
    """A receipt's copy of a task, scrubbed; ``None`` when it has nothing to mask."""

    scrubbed, fields, _ = scrub_task_failure(task, scrub)
    return scrubbed if fields else None


def failure_scrub_page_limit(limit: int) -> int:
    """``limit`` checked against ``MAX_TASK_FAILURE_SCRUB_PAGE``."""

    if type(limit) is not int or not 1 <= limit <= MAX_TASK_FAILURE_SCRUB_PAGE:
        raise ValueError(f"limit must be an integer from 1 to {MAX_TASK_FAILURE_SCRUB_PAGE}.")
    return limit


def scrub_skip_reason(task: Task) -> TaskFailureScrubSkip | None:
    """Why ``task`` can't be scrubbed: a worker holds its lease and may still write it, or
    another durable record binds its error exactly, so a rewrite would break that record's
    replay (see ``TaskFailureScrubSkip``)."""

    if task.worker_id is not None:
        return "leased"
    if isinstance(task.error, dict) and _RUNTIME_TASK_FAILURE_KEY in task.error:
        return "session_failure"
    payload = task.status_payload
    if isinstance(payload, dict) and _CANCELLATION_RECONCILIATION_KEY in payload:
        return "cancellation_reconciliation"
    return None
