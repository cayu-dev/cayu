"""Credential-free durable knowledge enrichment across a fresh worker process.

Usage:
    PYTHONPATH=src .venv/bin/python examples/durable_knowledge_enrichment.py

The producer explicitly submits one bounded signal and exits. A fresh process
reopens the SQLite task and knowledge stores, runs one enrichment job, and
commits pending reviewed knowledge. Nothing scans a transcript or starts a
background worker implicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from cayu import (
    KnowledgeAccessScope,
    KnowledgeCurator,
    KnowledgeCuratorConfig,
    KnowledgeEnrichmentJobStatus,
    KnowledgeEnrichmentQueue,
    KnowledgeEnrichmentRequest,
    KnowledgeEnrichmentTrigger,
    KnowledgeEnrichmentWorker,
    KnowledgeQuery,
    KnowledgeStatus,
    LearningBatch,
    LearningCandidate,
    LearningDecision,
    LearningSignal,
    LearningSourceReference,
    LearningVerdict,
    SQLiteKnowledgeStore,
    SQLiteTaskStore,
    TaskExecutionSource,
)

_NAMESPACE = "project:cayu"
_LABELS = {"project": "cayu"}


class DeploymentCandidateGenerator:
    async def generate_candidates(self, batch: LearningBatch) -> list[LearningCandidate]:
        return [
            LearningCandidate(
                proposal_key="run-migrations-before-startup",
                text="Run database migrations before starting the new service revision.",
                signal_ids=tuple(signal.id for signal in batch.signals),
                kind="procedure",
            )
        ]


class DeploymentEvidenceEvaluator:
    async def evaluate_candidate(
        self,
        candidate: LearningCandidate,
        signals: tuple[LearningSignal, ...],
    ) -> LearningDecision:
        return LearningDecision(
            verdict=(
                LearningVerdict.ACCEPTED
                if candidate.signal_ids == tuple(signal.id for signal in signals)
                else LearningVerdict.REJECTED
            ),
            code="bounded_source_supported",
        )


def _scope() -> KnowledgeAccessScope:
    return KnowledgeAccessScope.for_namespace(
        _NAMESPACE,
        required_labels=_LABELS,
        allowed_statuses=[KnowledgeStatus.PENDING, KnowledgeStatus.ACTIVE],
    )


def _curator_config() -> KnowledgeCuratorConfig:
    return KnowledgeCuratorConfig(
        candidate_generator_identity="example.deployment-generator.v1",
        evaluator_identity="example.deployment-evaluator.v1",
        namespace=_NAMESPACE,
        labels=_LABELS,
    )


def _queue(store: SQLiteTaskStore) -> KnowledgeEnrichmentQueue:
    return KnowledgeEnrichmentQueue(
        store,
        curator_config=_curator_config(),
        access_scope=_scope(),
    )


async def _submit(tasks_path: Path) -> str:
    tasks = SQLiteTaskStore(tasks_path)
    try:
        queue = _queue(tasks)
        occurred_at = datetime.now(UTC)
        source = LearningSourceReference(
            source_type="deployment_event",
            source_id="deployment-42",
            source_revision="event-7",
            source_hash="sha256:deployment-42-event-7",
            locator={"event_id": "event-7"},
        )
        signal = LearningSignal(
            id="deployment-42:startup-order",
            deduplication_key="deployment-42:startup-order",
            kind="deployment_failure",
            scope=_NAMESPACE,
            summary="The service started before its database migration completed.",
            source_references=(source,),
            occurred_at=occurred_at,
        )
        request = KnowledgeEnrichmentRequest(
            operation_id="deployment-42-enrichment",
            batch=LearningBatch(id="deployment-42", signals=(signal,)),
            trigger=KnowledgeEnrichmentTrigger.completed_interaction(
                session_id="deployment-session-42",
                interaction_id="deployment-interaction-42",
                terminal_event_id="event-7",
                occurred_at=occurred_at,
            ),
            profile=queue.profile,
            submitted_at=occurred_at,
            execution_source=TaskExecutionSource.SDK_TASK,
        )
        return (await queue.submit(request)).operation_id
    finally:
        await tasks.close()


async def _work(tasks_path: Path, knowledge_path: Path) -> str:
    tasks = SQLiteTaskStore(tasks_path)
    knowledge = SQLiteKnowledgeStore(knowledge_path, access_scope=_scope())
    curator = KnowledgeCurator(
        knowledge,
        candidate_generator=DeploymentCandidateGenerator(),
        evaluator=DeploymentEvidenceEvaluator(),
        config=_curator_config(),
        access_scope=_scope(),
    )
    try:
        job = await KnowledgeEnrichmentWorker(_queue(tasks), curator).process_next(
            worker_id="example-fresh-worker",
            lease_seconds=30,
        )
        if job is None:
            raise RuntimeError("The fresh worker found no enrichment job.")
        return job.status.value
    finally:
        await curator.aclose()
        await knowledge.close()
        await tasks.close()


def _fresh_worker(tasks_path: Path, knowledge_path: Path) -> str:
    completed = subprocess.run(
        [
            sys.executable,
            __file__,
            "--worker",
            str(tasks_path),
            str(knowledge_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
    )
    return completed.stdout.strip()


async def _inspect(tasks_path: Path, knowledge_path: Path, operation_id: str) -> None:
    tasks = SQLiteTaskStore(tasks_path)
    knowledge = SQLiteKnowledgeStore(knowledge_path, access_scope=_scope())
    try:
        job = await _queue(tasks).load(operation_id)
        if job is None or job.status is not KnowledgeEnrichmentJobStatus.COMPLETED:
            raise RuntimeError("The durable enrichment job did not complete.")
        pending = await knowledge.search(
            KnowledgeQuery(
                text="migration service startup",
                namespace=_NAMESPACE,
                statuses=[KnowledgeStatus.PENDING],
            )
        )
        print("job:", job.id, job.status.value)
        print("pending reviewed knowledge:", [hit.entry.text for hit in pending.hits])
    finally:
        await knowledge.close()
        await tasks.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("TASKS_DB", "KNOWLEDGE_DB"))
    arguments = parser.parse_args()
    if arguments.worker is not None:
        tasks_path, knowledge_path = map(Path, arguments.worker)
        print(asyncio.run(_work(tasks_path, knowledge_path)))
        return

    with tempfile.TemporaryDirectory(prefix="cayu-durable-enrichment-") as directory:
        root = Path(directory)
        tasks_path = root / "tasks.sqlite"
        knowledge_path = root / "knowledge.sqlite"
        operation_id = asyncio.run(_submit(tasks_path))
        worker_status = _fresh_worker(tasks_path, knowledge_path)
        if worker_status != KnowledgeEnrichmentJobStatus.COMPLETED.value:
            raise RuntimeError(f"Fresh worker returned {worker_status!r}.")
        asyncio.run(_inspect(tasks_path, knowledge_path, operation_id))


if __name__ == "__main__":
    main()
