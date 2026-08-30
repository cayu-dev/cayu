"""Credential-free reviewed learning from a completed Cayu session."""

from __future__ import annotations

import asyncio
from hashlib import sha256

from cayu import (
    InMemoryKnowledgeStore,
    InMemorySessionStore,
    KnowledgeAccessScope,
    KnowledgeCurator,
    KnowledgeCuratorConfig,
    KnowledgeQuery,
    KnowledgeReviewWorkflow,
    KnowledgeStatus,
    LearningBatch,
    LearningCandidate,
    LearningDecision,
    LearningSignal,
    LearningSourceReference,
    LearningVerdict,
    Message,
    RunRequest,
    Session,
    SessionIdentity,
    SessionStatus,
)

_NAMESPACE = "project:cayu"
_LABELS = {"project": "cayu"}


class DeploymentCandidateGenerator:
    """An application-owned deterministic candidate generator."""

    async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
        if not any(signal.kind == "deployment_failure" for signal in batch.signals):
            return []
        return [
            LearningCandidate(
                proposal_key="deploy-migrations-before-service-start",
                signal_ids=tuple(signal.id for signal in batch.signals),
                title="Run migrations before service startup",
                text="Run database migrations before starting the new service revision.",
                kind="procedure",
                aspects=("deployment", "migrations"),
            )
        ]


class DeploymentEvidenceEvaluator:
    """A distinct application-owned evaluator for generated candidates."""

    async def evaluate_candidate(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> LearningDecision:
        supported = all(
            signal.kind == "deployment_failure" and signal.source_references for signal in signals
        )
        return LearningDecision(
            verdict=LearningVerdict.ACCEPTED if supported else LearningVerdict.REJECTED,
            code="source_supported" if supported else "source_unsupported",
            notes="The terminal deployment evidence supports the proposed ordering.",
            confidence=0.95 if supported else None,
        )


def extract_learning_signal(completed: Session) -> LearningSignal:
    """Extract one bounded signal after the application observes terminal success."""

    if completed.status is not SessionStatus.COMPLETED:
        raise ValueError("Learning extraction requires a completed session.")
    observation_id = str(completed.metadata["learning_observation_id"])
    summary = str(completed.metadata["learning_observation"])
    return LearningSignal(
        id=f"{completed.id}:{observation_id}",
        deduplication_key=f"{completed.id}:{observation_id}",
        kind="deployment_failure",
        scope=_NAMESPACE,
        summary=summary,
        source_references=(
            LearningSourceReference(
                source_type="cayu_session",
                source_id=completed.id,
                source_revision=completed.updated_at.isoformat(),
                source_hash=f"sha256:{sha256(summary.encode('utf-8')).hexdigest()}",
                locator={"observation_id": observation_id},
            ),
        ),
        occurred_at=completed.updated_at,
    )


async def main() -> None:
    sessions = InMemorySessionStore()
    run = await sessions.create(
        RunRequest(
            agent_name="deployment-agent",
            session_id="deployment-run-17",
            messages=[Message.text("user", "Verify the staging deployment.")],
            metadata={
                "learning_observation_id": "startup-ordering",
                "learning_observation": (
                    "Staging failed because the service started before migrations completed."
                ),
            },
        ),
        identity=SessionIdentity(provider_name="hermetic", model="none"),
    )
    completed = await sessions.update_status(run.id, SessionStatus.COMPLETED)
    signal = extract_learning_signal(completed)

    access_scope = KnowledgeAccessScope.for_namespace(
        _NAMESPACE,
        required_labels=_LABELS,
        allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE],
    )
    knowledge = InMemoryKnowledgeStore(access_scope=access_scope)
    async with KnowledgeCurator(
        knowledge,
        candidate_generator=DeploymentCandidateGenerator(),
        evaluator=DeploymentEvidenceEvaluator(),
        config=KnowledgeCuratorConfig(
            candidate_generator_identity="example.deployment-generator.v1",
            evaluator_identity="example.deployment-evaluator.v1",
            namespace=_NAMESPACE,
            labels=_LABELS,
        ),
    ) as curator:
        curation = await curator.curate(LearningBatch(id="deployment-run-17", signals=(signal,)))
    pending = await KnowledgeReviewWorkflow(
        knowledge,
        namespace=_NAMESPACE,
        labels=_LABELS,
    ).list_pending()
    print("curation:", curation.candidates[0].outcome.value)
    print("pending for review:", [item.entry.id for item in pending.entries])

    reviewer = KnowledgeReviewWorkflow(
        knowledge,
        namespace=_NAMESPACE,
        labels=_LABELS,
    )
    approved = await reviewer.approve(
        pending.entries[0].entry.id,
        operation_id="example-reviewed-curation",
        reviewer_identity="deployment-reviewer",
        reviewer_version="1",
    )

    later_run = await sessions.create(
        RunRequest(
            agent_name="deployment-agent",
            session_id="deployment-run-18",
            messages=[Message.text("user", "How should the next deployment start?")],
        ),
        identity=SessionIdentity(provider_name="hermetic", model="none"),
    )
    recalled = await knowledge.search(
        KnowledgeQuery(text="migration service startup", namespace=_NAMESPACE)
    )
    print("approved:", approved.entry.status.value)
    print("later run:", later_run.id)
    print("recalled:", [hit.entry.text for hit in recalled.hits])


if __name__ == "__main__":
    asyncio.run(main())
