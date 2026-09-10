"""Typed handler for an application that can resolve immutable candidate references.

Configure the CayuApp's provider, agent, completion verifier, and result resolver,
publish a WorkContract, and enqueue a TaskCreate referring to it before running
VerifiedTaskWorker(app, handler, worker_id=...). The candidate reader below must
only read application-owned evidence; it must not perform the domain operation.
Use the worker as an async context manager and retain it if close reports
VerifiedTaskWorkerDraining. This module does not launch services on import.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from cayu import (
    CompletionProposalCreate,
    CompletionResultReference,
    Message,
    RunRequest,
    VerifiedTaskHandler,
    VerifiedTaskHandlerReport,
    VerifiedTaskPreparationContext,
    VerifiedTaskProposalContext,
)


class ReferencedResultHandler(VerifiedTaskHandler):
    """Bridge read-only domain evidence into an explicit completion proposal."""

    def __init__(
        self,
        agent_name: str,
        candidate: Callable[[VerifiedTaskProposalContext], Awaitable[CompletionResultReference]],
    ) -> None:
        self.agent_name = agent_name
        self.candidate = candidate

    async def prepare(self, context: VerifiedTaskPreparationContext) -> RunRequest:
        # The worker supplies task/session/lease authority after this callback.
        return RunRequest(
            agent_name=self.agent_name,
            messages=[Message.text("user", context.contract.objective)],
        )

    async def propose(self, context: VerifiedTaskProposalContext) -> VerifiedTaskHandlerReport:
        reference = await self.candidate(context)
        # The digest-bound candidate is a claim for the registered verifier to
        # assess, not proof that the task succeeded. Never derive a verdict by
        # parsing the model's final prose.
        return VerifiedTaskHandlerReport(
            proposal=CompletionProposalCreate(
                proposal_id=context.proposal_id,
                attempt_id=context.attempt.attempt_id,
                result=reference,
            )
        )
