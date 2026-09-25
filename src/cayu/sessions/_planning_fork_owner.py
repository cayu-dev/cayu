"""Application-owned FORK resolution and retention cleanup; no caller callbacks."""

from cayu.collaboration._contracts import ExactMatch
from cayu.collaboration._planning_fork_types import (
    RequestViewStageCommand,
    RequestViewStageReceipt,
    _ForkPreparationReadback,
    _ViewReadback,
)
from cayu.collaboration._preparation import prepare_contract
from cayu.collaboration.participants import CollaborationUnavailable
from cayu.sessions._planning_view_owner import NativePlanningViewOwner
from cayu.sessions._recipient_preparation import resolve_fork_recipient


class NativePlanningForkOwner:
    def __init__(self, application):
        self._app = application
        self._views = NativePlanningViewOwner(application)

    def _command(self, value):
        return prepare_contract(RequestViewStageCommand, value, redactor=self._app._secret_redactor)

    async def reserve(self, expected, *, context):
        from cayu.collaboration._planning_view_reservation import (
            _ViewReservationReadback,
        )
        from cayu.sessions._context_selection_fence import _CONTEXT_SELECTION_AUTHORITY

        command = self._command(expected)
        blueprint = command.preparation
        self._views._access(blueprint.view, context)
        reservation = await self._app.session_store._prepare_context_view_selection_target(
            blueprint.view, authority=_CONTEXT_SELECTION_AUTHORITY
        )
        return _ViewReservationReadback(command, reservation)

    async def prepare(self, expected, *, context):
        command = self._command(expected)
        blueprint = command.preparation
        await self._views.select(blueprint.view, context=context)
        retained = await self._views.adopt(blueprint.view, context=context)
        if not isinstance(retained, ExactMatch):
            raise CollaborationUnavailable("FORK preparation lacks exact adopted history.")
        resolved = await resolve_fork_recipient(
            self._app, blueprint, retained.receipt, context=context
        )
        return _ForkPreparationReadback(command, resolved)

    async def discharge(self, expected, *, context):
        command = self._command(expected)
        found = await self._views.discharge(command.preparation.view, context=context)
        if not isinstance(found, ExactMatch):
            raise CollaborationUnavailable("FORK history responsibility is unresolved.")
        terminal = found.receipt
        selection = None if terminal.state == "excluded" else terminal.selection
        receipt = RequestViewStageReceipt(
            command=command,
            state=terminal.state,
            view_id=None if selection is None else selection.view_id,
            manifest_commitment=None if selection is None else selection.manifest_commitment,
            pin_commitment=None if selection is None else selection.pin_commitment,
        )
        return _ViewReadback(
            prepare_contract(RequestViewStageReceipt, receipt, redactor=self._app._secret_redactor)
        )
