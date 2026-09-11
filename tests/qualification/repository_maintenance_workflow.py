"""Coding-stage workflow; delivery and human approval remain separate phases."""

from cayu import WorkflowBase, WorkflowSpec
from tests.qualification.repository_maintenance_acceptance import MaintenanceAcceptanceRejected
from tests.qualification.repository_maintenance_request import copy_request, invalid_request


class MaintenanceCodingWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="repository-maintenance-coding")

    def __init__(self, application, task, *, accepted):
        super().__init__(application.app)
        self.application = application
        self.task = task
        self.accepted = copy_request(accepted)

    async def run(self, session_id):
        # This module is resolved in the emitted consumer, not the fixture tree.
        from operations.maintenance_requests import (  # ty: ignore[unresolved-import]
            capture_accepted_request,
        )

        if await capture_accepted_request(self.application, self.task) != copy_request(
            self.accepted
        ):
            raise invalid_request()
        if self.task.parent_session_id != session_id or self.task.causal_budget_id != session_id:
            raise ValueError("Coding task is not bound to this workflow root.")
        ctx = self.context(session_id)
        yield await ctx.start()
        publication = await self.application.run(self.task)
        try:
            verified = await self.application.verify(self.task, publication)
        except MaintenanceAcceptanceRejected:
            verdict = "rejected"
            digest = publication.result_reference.digest
        else:
            verdict = "verified"
            digest = verified.result_digest
        yield await ctx.completed(
            {
                "product_run_id": self.task.product_run_id,
                "result_digest": digest,
                "verdict": verdict,
            }
        )
