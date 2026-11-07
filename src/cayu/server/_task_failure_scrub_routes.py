"""Re-redact the failure text a task store kept, for full-access callers.

`cayu task scrub-error TASK_ID` and `cayu task scrub-error --all` reach these routes on a
running server or a Cloud Agent. They apply the redaction exception diagnostics use now
(the application's registered secrets, then credential shapes) to a task's stored
``error`` and to the ``error`` a pending cancellation keeps in its ``status_payload``,
together with every receipt that keeps a copy of the task. ``ControlPlaneRoles`` keeps
them full-access only: they aren't in the read-and-decide list.

A field that changed records the scrub under ``cayu_scrub`` (when, which redaction, how
many values), and each applied scrub writes one JSON audit line to standard error,
``{"event": "cayu.task.failure_scrubbed", ...}``, naming the task, the fields, the counts
and the caller's subject. Neither ever holds the text that was masked.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Annotated, cast

from fastapi import HTTPException
from pydantic import StringConstraints

from cayu.server.auth import AuthContext
from cayu.server.contracts import (
    ApiTaskFailureScrub,
    ApiTaskFailureSweep,
    TaskFailureScrubRequest,
    TaskFailureSweepRequest,
)
from cayu.tasks._failure_scrub import (
    TaskFailureScrub,
    TaskFailureScrubResult,
    failure_text_redaction,
    scrub_result_row,
)

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp
    from cayu.tasks.store import TaskStore

NonBlankString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
AUDIT_EVENT = "cayu.task.failure_scrubbed"


def register_task_failure_scrub_routes(
    router: APIRouter,
    *,
    cayu_app: CayuApp,
    protected: list[Depends],
    optional_auth_context: Depends,
) -> None:
    auth_default = cast("AuthContext | None", optional_auth_context)

    def store() -> TaskStore:
        task_store = cayu_app.task_store
        if task_store is None:
            raise HTTPException(404, "Task store is not configured.")
        return task_store

    def scrub(dry_run: bool) -> TaskFailureScrub:
        return TaskFailureScrub(
            redact=failure_text_redaction(cayu_app.redact_json),
            scrubbed_at=datetime.now(UTC),
            dry_run=dry_run,
        )

    @router.post(
        "/tasks/{task_id}/scrub-error",
        response_model=ApiTaskFailureScrub,
        dependencies=protected,
    )
    async def scrub_task_error(
        task_id: NonBlankString,
        body: TaskFailureScrubRequest | None = None,
        auth_context: AuthContext | None = auth_default,
    ):
        dry_run = body is not None and body.dry_run
        task_store = store()
        try:
            result = await task_store.scrub_task_failure_text(task_id, scrub(dry_run))
        except (KeyError, PermissionError):
            raise HTTPException(404, "Task was not found.") from None
        except NotImplementedError:
            raise HTTPException(501, "This task store can't scrub failure text.") from None
        except ValueError:
            raise HTTPException(409, "The task's stored failure text can't be scrubbed.") from None
        _audit(result, auth_context)
        return scrub_result_row(result, dry_run)

    @router.post(
        "/tasks/scrub-errors",
        response_model=ApiTaskFailureSweep,
        dependencies=protected,
    )
    async def scrub_task_errors(
        body: TaskFailureSweepRequest | None = None,
        auth_context: AuthContext | None = auth_default,
    ):
        request = body or TaskFailureSweepRequest()
        task_store = store()
        try:
            task_ids = await task_store.list_task_ids_with_failure_text(
                after_task_id=request.after_task_id, limit=request.limit
            )
        except NotImplementedError:
            raise HTTPException(501, "This task store can't scrub failure text.") from None
        rows = []
        for task_id in task_ids:
            try:
                result = await task_store.scrub_task_failure_text(task_id, scrub(request.dry_run))
            except (KeyError, PermissionError):
                # Deleted, or hidden from this caller, since the page was read.
                continue
            except ValueError:
                raise HTTPException(
                    409, "A task's stored failure text can't be scrubbed."
                ) from None
            _audit(result, auth_context)
            if result.fields or result.receipts or result.skipped_reason:
                rows.append(scrub_result_row(result, request.dry_run))
        return {
            "dry_run": request.dry_run,
            "examined": len(task_ids),
            "tasks": rows,
            "next_after_task_id": task_ids[-1] if len(task_ids) == request.limit else None,
        }


def _audit(result: TaskFailureScrubResult, auth_context: AuthContext | None) -> None:
    if not result.applied:
        return
    record = {
        "at": datetime.now(UTC).isoformat(),
        "event": AUDIT_EVENT,
        "task_id": result.task_id,
        "fields": list(result.fields),
        "masked_values": result.masked_values,
        "receipts": result.receipts,
        "subject": None if auth_context is None else auth_context.subject,
    }
    # ASCII escapes keep an identifier's control characters from forging a log line.
    print(json.dumps(record, sort_keys=True), file=sys.stderr, flush=True)
