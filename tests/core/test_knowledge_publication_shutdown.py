from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cayu._knowledge_publication_owner import RetainedKnowledgePublicationOwner

_SHUTDOWN_PROGRAM = r"""
import asyncio
import sys
from contextlib import suppress
from datetime import UTC, datetime

from cayu import (
    AgentSpec,
    CayuApp,
    InMemoryKnowledgeStore,
    KnowledgeAccessScope,
    KnowledgeCurator,
    KnowledgeCuratorConfig,
    LearningBatch,
    LearningCandidate,
    LearningDecision,
    LearningSignal,
    LearningSourceReference,
    LearningVerdict,
    RememberKnowledgeTool,
    ToolContext,
)
from cayu.server import DashboardConfig, ServerConfig, ServerLifecycleConfig, create_server


MODE = sys.argv[1]
SETTLES_WITHIN_GRACE = sys.argv[2] == "settles"
STALLS_RECONCILIATION = MODE.endswith("-reconciliation")
SCOPE = KnowledgeAccessScope.privileged()


class StalledStore(InMemoryKnowledgeStore):
    def __init__(self):
        super().__init__(access_scope=SCOPE)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = asyncio.Event()
        self.publication_returned = False

    async def load_entry_publication_receipt(self, operation_id, *, access_scope=None):
        if STALLS_RECONCILIATION and self.publication_returned and not self.release.is_set():
            self.started.set()
            try:
                await self.release.wait()
            finally:
                self.stopped.set()
        return await super().load_entry_publication_receipt(
            operation_id,
            access_scope=access_scope,
        )

    async def publish_entry_revision(
        self,
        entry,
        chunks,
        *,
        evidence=None,
        access_scope=None,
        operation_id,
        expected_revision=None,
        activation_authority=None,
    ):
        if STALLS_RECONCILIATION:
            receipt = await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
                activation_authority=activation_authority,
            )
            self.publication_returned = True
            return receipt
        self.started.set()
        try:
            await self.release.wait()
            return await super().publish_entry_revision(
                entry,
                chunks,
                evidence=evidence,
                access_scope=access_scope,
                operation_id=operation_id,
                expected_revision=expected_revision,
                activation_authority=activation_authority,
            )
        finally:
            self.stopped.set()


class Generator:
    async def generate_candidates(self, batch):
        return [
            LearningCandidate(
                proposal_key="shutdown-proposal",
                text="Run migrations before service startup.",
                kind="procedure",
                signal_ids=("shutdown-signal",),
            )
        ]


class Evaluator:
    async def evaluate_candidate(self, candidate, signals):
        return LearningDecision(
            verdict=LearningVerdict.ACCEPTED,
            code="supported",
            notes="The evidence supports this procedure.",
            confidence=0.95,
        )


def batch():
    source = LearningSourceReference(
        source_type="session",
        source_id="shutdown-session",
        source_hash="sha256:shutdown-source",
        locator={"event_id": "shutdown-event"},
    )
    signal = LearningSignal(
        id="shutdown-signal",
        deduplication_key="shutdown-deduplication",
        kind="deployment_failure",
        scope="project:cayu",
        summary="The service started before its migration completed.",
        source_references=(source,),
        occurred_at=datetime(2026, 8, 24, tzinfo=UTC),
    )
    return LearningBatch(id="shutdown-batch", signals=(signal,))


async def cancel_after_dispatch(invocation, store):
    await asyncio.wait_for(store.started.wait(), timeout=1)
    invocation.cancel("caller left after durable dispatch")
    with suppress(asyncio.CancelledError):
        await invocation
    if SETTLES_WITHIN_GRACE:
        asyncio.get_running_loop().call_later(0.01, store.release.set)


async def run_remember(*, server_lifecycle):
    store = StalledStore()
    tool = RememberKnowledgeTool()
    context = ToolContext(
        session_id="shutdown-session",
        idempotency_key="shutdown-operation",
        knowledge_store=store,
    )
    if server_lifecycle:
        app = CayuApp(enable_logging=False)
        app.register_agent(AgentSpec(name="assistant", model="fake"), tools=[tool])
        server = create_server(
            app,
            config=ServerConfig.local_development(
                dashboard=DashboardConfig(enabled=False),
                lifecycle=ServerLifecycleConfig(
                    knowledge_publication_shutdown_grace_seconds=(
                        0.2 if SETTLES_WITHIN_GRACE else 0.01
                    )
                ),
            ),
        )
        async with server.router.lifespan_context(server):
            invocation = asyncio.create_task(
                tool.run(context, {"text": "shutdown-secret-material"})
            )
            await cancel_after_dispatch(invocation, store)
        drained = await tool.aclose(timeout_s=1)
    else:
        invocation = asyncio.create_task(
            tool.run(context, {"text": "shutdown-secret-material"})
        )
        await cancel_after_dispatch(invocation, store)
        drained = await tool.aclose(
            timeout_s=0.2 if SETTLES_WITHIN_GRACE else 0.01
        )
    await asyncio.wait_for(store.stopped.wait(), timeout=1)
    while tool._publication_owner or tool._read_operations:
        await asyncio.sleep(0)
    assert drained is SETTLES_WITHIN_GRACE


async def run_curator():
    store = StalledStore()
    curator = KnowledgeCurator(
        store,
        candidate_generator=Generator(),
        evaluator=Evaluator(),
        config=KnowledgeCuratorConfig(
            candidate_generator_identity="shutdown.generator.v1",
            evaluator_identity="shutdown.evaluator.v1",
            namespace="project:cayu",
            labels={"project": "cayu"},
        ),
    )
    invocation = asyncio.create_task(curator.curate(batch()))
    await cancel_after_dispatch(invocation, store)
    drained = await curator.aclose(
        timeout_s=0.2 if SETTLES_WITHIN_GRACE else 0.01
    )
    await asyncio.wait_for(store.stopped.wait(), timeout=1)
    while curator._publication_owner:
        await asyncio.sleep(0)
    assert drained is SETTLES_WITHIN_GRACE


if MODE in {"sdk-remember", "sdk-remember-reconciliation"}:
    asyncio.run(run_remember(server_lifecycle=False))
elif MODE == "sdk-curator":
    asyncio.run(run_curator())
elif MODE in {"server-remember", "server-remember-reconciliation"}:
    asyncio.run(run_remember(server_lifecycle=True))
else:
    raise AssertionError(f"unknown mode: {MODE}")
"""


@pytest.mark.parametrize(
    ("mode", "settlement"),
    [
        ("sdk-remember", "stalled"),
        ("sdk-remember", "settles"),
        ("sdk-curator", "stalled"),
        ("sdk-curator", "settles"),
        ("server-remember", "stalled"),
        ("server-remember", "settles"),
        ("sdk-remember-reconciliation", "stalled"),
        ("sdk-remember-reconciliation", "settles"),
        ("server-remember-reconciliation", "stalled"),
        ("server-remember-reconciliation", "settles"),
    ],
)
@pytest.mark.process
def test_retained_publication_shutdown_has_a_real_process_bound(
    mode: str,
    settlement: str,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    environment = {
        **os.environ,
        "PYTHONPATH": str(repository / "src"),
        "PYTHONASYNCIODEBUG": "1",
    }

    completed = subprocess.run(
        [sys.executable, "-c", _SHUTDOWN_PROGRAM, mode, settlement],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Task was destroyed but it is pending" not in completed.stderr
    assert "exception was never retrieved" not in completed.stderr
    assert "shutdown-secret-material" not in completed.stderr


def test_close_caller_cancellation_does_not_cancel_the_shared_drain() -> None:
    async def run() -> tuple[bool, str]:
        owner = RetainedKnowledgePublicationOwner[str](max_publications=1)
        started = asyncio.Event()
        release = asyncio.Event()

        async def publication() -> str:
            started.set()
            await release.wait()
            return "published"

        invocation = asyncio.create_task(owner.run("operation", "fingerprint", publication))
        await asyncio.wait_for(started.wait(), timeout=1)

        close_caller = asyncio.create_task(owner.aclose(timeout_s=1))
        await asyncio.sleep(0)
        close_caller.cancel("shutdown caller left")
        with pytest.raises(asyncio.CancelledError, match="shutdown caller left"):
            await close_caller

        release.set()
        publication_result = await invocation
        return await owner.aclose(timeout_s=0.01), publication_result.value

    drained, value = asyncio.run(run())

    assert drained is True
    assert value == "published"
