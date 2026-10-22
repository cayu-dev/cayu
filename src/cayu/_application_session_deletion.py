"""Application-level dependent-first session deletion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cayu._validation import require_durable_clean_nonblank
from cayu.runtime._session_closure_projection import (
    project_closure_export,
    project_closure_manifest,
    project_closure_report,
)
from cayu.runtime.session_closure import (
    SessionClosureExport,
    SessionClosureExportIncomplete,
    SessionClosurePolicy,
    SessionClosureReport,
)
from cayu.sessions.records import SessionStatus

if TYPE_CHECKING:
    from cayu.applications import CayuApp

_DELETABLE_FINISHED_SESSION_STATUSES = frozenset({SessionStatus.COMPLETED, SessionStatus.FAILED})


async def delete_app_session(
    self: CayuApp, session_id: str, *, finished_only: bool
) -> SessionClosureReport | None:
    session_id = require_durable_clean_nonblank(session_id, "session_id")
    session = await self.session_store.load(session_id)
    if session is None:
        return None
    if finished_only and session.status not in _DELETABLE_FINISHED_SESSION_STATUSES:
        raise ValueError(
            f"Only completed or failed sessions can be deleted; this session is "
            f"{session.status.value}. Pass finished_only=False to delete it anyway."
        )
    if (
        finished_only
        and getattr(self.session_store, "supports_session_closure_finished_only", False) is not True
    ):
        raise ValueError("Session store cannot enforce finished-only closure admission.")
    inspection = await self.validate_session_closure(session_id)
    if not inspection.complete:
        raise ValueError(
            "Session closure inventory is incomplete; no destructive work was started."
        )
    if not await self.discard_parked_egress_allocations(session_id):
        raise ValueError("Session has a parked egress allocation whose cleanup remains in flight.")
    report = project_closure_report(
        await self._environment_registry.session_closure.erase(
            session_id, finished_only=finished_only
        ),
        redactor=self._secret_redactor,
        project_session_id=self.project_session_id_for_exposure,
    )
    if not report.complete:
        raise ValueError(
            report.error or "Session closure is incomplete; no deletion was acknowledged."
        )
    return report


async def export_app_session_closure(
    self: CayuApp,
    session_id: str,
    *,
    policy: SessionClosurePolicy | None,
    allow_partial: bool,
) -> SessionClosureExport:
    try:
        export = await self._environment_registry.session_closure.export(
            session_id, policy=policy, allow_partial=allow_partial
        )
    except SessionClosureExportIncomplete as exc:
        raise SessionClosureExportIncomplete(
            project_closure_manifest(
                exc.manifest,
                redactor=self._secret_redactor,
                project_session_id=self.project_session_id_for_exposure,
            )
        ) from None
    return project_closure_export(
        export,
        redactor=self._secret_redactor,
        project_session_id=self.project_session_id_for_exposure,
    )
