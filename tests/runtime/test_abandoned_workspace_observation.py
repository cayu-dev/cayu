"""Abandonment closes the workspace observations of the tool round it retires.

A worker lost inside a tool that changes the workspace, or as it finalizes that
call's workspace observation, leaves the observation open beside its pending round.
Abandoning the session retires the round into its abandonment record. The
observation is then closed with that retained round as its authority, with the
environment where it is registered and without it where it isn't. Repeated closes,
concurrent ones and later sweeps leave the closed observation as it is.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tests.core.test_workspace_mutation_receipts import (
    _portable_environment_spec,
    _SingleToolProvider,
)

from cayu import (
    AgentSpec,
    CayuApp,
    EventType,
    Message,
    RunRequest,
    SessionStatus,
    Tool,
    ToolResult,
    ToolSpec,
)
from cayu.environments.base import Environment
from cayu.environments.bindings import DeterministicWorkspaceBinding
from cayu.runtime import _incomplete_session_recovery as incomplete_recovery
from cayu.runtime._abandoned_tool_result_publication import AbandonedToolResults
from cayu.runtime._incomplete_session_recovery import IncompleteSessionRecovery
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions import IncompleteSessionRecoveryAction
from cayu.storage import SQLiteSessionStore
from cayu.workspaces import LocalWorkspace
from cayu.workspaces.observation_recovery import workspace_observations_from_checkpoint

_ENVIRONMENT = "run-1-slot-1"


class _CrashingWrite(Tool):
    """A workspace write whose worker dies inside it, as a deploy kills it mid-command,
    or (``finalization``) right after it, as it finalizes the write's observation."""

    spec = ToolSpec(
        name="blocking_workspace_mutation",
        parallel_safe=False,
        workspace_mutation=True,
        max_terminal_payload_bytes=1024,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="tests:abandoned-observation:write",
            behavior_version="1",
            implementation_version="1",
        ),
    )

    async def run(self, ctx, args):
        await ctx.workspace.create_bytes("written.txt", b"written")
        if os.environ["CAYU_OBSERVATION_TEST_CRASH"] == "tool":
            os._exit(137)
        return ToolResult(content="written")


def _application(store, root: Path | None = None) -> CayuApp:
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(
        _SingleToolProvider(tool_name="blocking_workspace_mutation", arguments={}), default=True
    )
    if root is not None:
        app.register_environment(
            Environment(
                _portable_environment_spec(_ENVIRONMENT),
                workspace=LocalWorkspace(root, workspace_id="slot-workspace"),
                binding=DeterministicWorkspaceBinding(),
            )
        )
    app.register_agent(AgentSpec(name="fixer", model="scripted-model"), tools=[_CrashingWrite()])
    return app


def _run_worker() -> None:
    """The run's worker: it registers the environment and dies inside the write."""

    async def scenario() -> None:
        store = SQLiteSessionStore(os.environ["CAYU_OBSERVATION_TEST_STORE"])
        if os.environ["CAYU_OBSERVATION_TEST_CRASH"] == "finalization":
            # The tool returned and its outcome is staged; the worker dies as it
            # finalizes the call's workspace observation.
            publish = store.publish_runtime_publication

            async def die_finalizing(session_id, *, request, **kwargs):
                if request.kind == "workspace-observation" and (
                    request.intent.get("phase") == "terminal"
                ):
                    os._exit(137)
                return await publish(session_id, request=request, **kwargs)

            store.publish_runtime_publication = die_finalizing
        app = _application(store, Path(os.environ["CAYU_OBSERVATION_TEST_WORKSPACE"]))
        async for _ in app.run(
            RunRequest(
                agent_name="fixer",
                session_id="fix",
                environment_name=_ENVIRONMENT,
                messages=[Message.text("user", "fix it")],
            )
        ):
            pass
        raise AssertionError("The worker did not die inside the write.")

    asyncio.run(scenario())


@pytest.fixture(params=["tool", "finalization"])
def crash(request) -> str:
    """Where the worker dies: inside the write, or as it finalizes its observation."""

    return request.param


@pytest.fixture
def orphan(tmp_path: Path, crash: str) -> Path:
    """A session its lost worker left running at ``crash``."""

    path = tmp_path / "sessions.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    child = subprocess.run(
        [sys.executable, "-c", f"from {__name__} import _run_worker; _run_worker()"],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(entry for entry in sys.path if entry),
            "CAYU_OBSERVATION_TEST_STORE": str(path),
            "CAYU_OBSERVATION_TEST_WORKSPACE": str(workspace),
            "CAYU_OBSERVATION_TEST_CRASH": crash,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert child.returncode == 137, (child.stdout, child.stderr)
    return path


# How the write's observation ends, by where its worker died.
_WITHOUT_ENVIRONMENT = {
    "tool": ("ambiguous", "worker_lost_before_tool_outcome_was_durable"),
    "finalization": ("incomplete", "workspace_revision_evidence_incomplete"),
}
_WITH_ENVIRONMENT = {
    "tool": ("ambiguous", "worker_lost_before_tool_outcome_was_durable"),
    "finalization": ("complete", None),
}


async def _nothing(self, session_id):
    return ()


async def _nothing_published(*, pending_round, **kwargs):
    return AbandonedToolResults(pending_round=pending_round)


def _as_earlier_release(patch) -> None:
    """Abandon without closing the round's observations, as releases before it did.

    Those releases didn't publish the round's staged result either, which would
    close its observation first.
    """

    patch.setattr(IncompleteSessionRecovery, "finish_abandoned_workspace_observations", _nothing)
    patch.setattr(incomplete_recovery, "publish_abandoned_tool_results", _nothing_published)


async def _open_observations(store) -> dict:
    return workspace_observations_from_checkpoint(await store.load_checkpoint("fix"))


async def _finalized(store) -> list:
    return [
        event
        for event in await store.load_events("fix")
        if event.type is EventType.WORKSPACE_OBSERVATION_FINALIZED
    ]


def test_reconciliation_closes_the_observation_of_the_run_it_abandons(
    orphan: Path, crash: str
) -> None:
    async def scenario() -> None:
        store = SQLiteSessionStore(orphan)
        # A later Release's web process: the run's environment isn't registered here.
        reader = _application(store)
        assert await _open_observations(store)
        assert await _finalized(store) == []

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert result.status is SessionStatus.FAILED
        assert result.actions == (
            # A write whose result was staged has a known outcome.
            IncompleteSessionRecoveryAction.ABANDONED_TOOL_EFFECT
            if crash == "tool"
            else IncompleteSessionRecoveryAction.ABANDONED_ORPHANED_RUN,
            IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION,
        )
        assert await _open_observations(store) == {}
        [finalized] = await _finalized(store)
        # Without the environment nothing verifies what the write changed. A staged
        # result still binds the observation to the write's outcome.
        assert (finalized.payload["status"], finalized.payload.get("detail_code")) == (
            _WITHOUT_ENVIRONMENT[crash]
        )
        # The staged result is published once its observation is closed, withheld since
        # its after-tool hooks hadn't run. A write the worker died inside has none.
        terminals = [
            event
            for event in await store.load_events("fix")
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert [event.payload["tool_call_id"] for event in terminals] == (
            ["call-workspace"] if crash == "finalization" else []
        )
        # Closing the ended session again replays instead of failing.
        closed = await reader.close_session("fix", reason="Worker lost.")
        assert closed.status is SessionStatus.FAILED
        assert IncompleteSessionRecoveryAction.FAILED not in closed.actions
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_close_finishes_an_abandonment_that_left_its_observation_open(
    orphan: Path, monkeypatch
) -> None:
    async def scenario() -> None:
        store = SQLiteSessionStore(orphan)
        reader = _application(store)
        # An earlier release abandoned the run and left its observation open.
        with monkeypatch.context() as earlier:
            _as_earlier_release(earlier)
            [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert result.status is SessionStatus.FAILED
        assert await _open_observations(store)

        closed = await reader.close_session("fix", reason="Worker lost.")

        assert closed.status is SessionStatus.FAILED
        assert IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in closed.actions
        assert await _open_observations(store) == {}
        assert len(await _finalized(store)) == 1
        # Once finished, closing it again finds nothing left to do.
        again = await reader.close_session("fix", reason="Worker lost.")
        assert again.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_an_abandonment_with_the_environment_verifies_the_observation(
    orphan: Path, crash: str, tmp_path: Path, monkeypatch
) -> None:
    async def scenario() -> None:
        store = SQLiteSessionStore(orphan)
        with monkeypatch.context() as earlier:
            _as_earlier_release(earlier)
            reader = _application(store)
            await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
            await reader.aclose()
        assert await _open_observations(store)
        # The worker that registers the environment finishes the abandonment.
        worker = _application(store, tmp_path / "workspace")

        closed = await worker.close_session("fix", reason="Worker lost.")

        assert IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in closed.actions
        assert await _open_observations(store) == {}
        [finalized] = await _finalized(store)
        # The retained stage of a staged result authenticates the write's artifacts.
        assert (finalized.payload["status"], finalized.payload.get("detail_code")) == (
            _WITH_ENVIRONMENT[crash]
        )
        await worker.aclose()
        await store.close()

    asyncio.run(scenario())


def test_the_sweep_finishes_an_abandonment_that_stopped_before_its_interaction(
    orphan: Path, monkeypatch
) -> None:
    from cayu.runtime._session_recovery import SessionRecovery

    async def scenario() -> None:
        store = SQLiteSessionStore(orphan)
        reader = _application(store)
        # An earlier release abandoned the run, left its observation open, and stopped
        # before it settled the run's interaction.
        with monkeypatch.context() as earlier:
            _as_earlier_release(earlier)
            earlier.setattr(SessionRecovery, "settle_released_open_interaction", _nothing)
            await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)
        assert await _open_observations(store)

        [result] = await reader.reconcile_orphaned_sessions(inactive_for_seconds=0)

        assert IncompleteSessionRecoveryAction.FAILED not in result.actions, result.message
        assert IncompleteSessionRecoveryAction.REPAIRED_WORKSPACE_OBSERVATION in result.actions
        assert await _open_observations(store) == {}
        [latest] = [
            event.type
            for event in await store.load_events("fix")
            if event.type.value.startswith("interaction.")
        ][-1:]
        assert latest is EventType.INTERACTION_FAILED
        # Nothing is left for the next sweep.
        assert await reader.reconcile_orphaned_sessions(inactive_for_seconds=0) == ()
        await reader.aclose()
        await store.close()

    asyncio.run(scenario())


def test_concurrent_closes_finish_the_observation_once(orphan: Path, monkeypatch) -> None:
    async def scenario() -> None:
        stores = [SQLiteSessionStore(orphan) for _ in range(2)]
        apps = [_application(store) for store in stores]
        with monkeypatch.context() as earlier:
            _as_earlier_release(earlier)
            await apps[0].reconcile_orphaned_sessions(inactive_for_seconds=0)

        # A person closes it while the sweep, in another process, finishes it.
        results = await asyncio.gather(
            *(app.close_session("fix", reason="Worker lost.") for app in apps)
        )

        for result in results:
            assert result.status is SessionStatus.FAILED
            assert IncompleteSessionRecoveryAction.FAILED not in result.actions, result.message
        assert await _open_observations(stores[0]) == {}
        assert len(await _finalized(stores[0])) == 1
        for app, store in zip(apps, stores, strict=True):
            await app.aclose()
            await store.close()

    asyncio.run(scenario())
