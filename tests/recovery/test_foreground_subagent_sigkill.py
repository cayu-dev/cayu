from __future__ import annotations

import asyncio
import os
import signal

import pytest
from worker_harness import BackendConfig, RecoveryHarness

from cayu.core import ToolResultPart

pytestmark = [
    pytest.mark.process,
    pytest.mark.sigkill_recovery,
    pytest.mark.skipif(os.name != "posix", reason="requires real POSIX SIGKILL"),
]


def test_expired_foreground_worker_cannot_publish_after_recovery(tmp_path):
    backend = BackendConfig.sqlite(tmp_path)
    parent_id = "foreground-expired-owner"
    with RecoveryHarness(tmp_path, backend) as harness:
        old = harness.launch(
            scenario="foreground_subagent",
            action="start",
            session_id=parent_id,
            crash_phase="during_child",
            stale_worker=True,
        )
        reached = old.wait_for_phase("during_child")
        # The recovery worker advances its injected clock beyond the old run's
        # lease, then uses normal child/model settlement. The old process stays
        # alive, with its already-dispatched provider read held at the barrier.
        recovered = harness.launch(
            scenario="foreground_subagent",
            action="recover",
            session_id=parent_id,
            crash_phase="during_child",
            child_session_id=reached["child_session_id"],
        ).wait_success(timeout=60)
        assert recovered["status"] == "completed"
        parent = asyncio.run(harness.load_session_state(parent_id))
        child = asyncio.run(harness.load_session_state(reached["child_session_id"]))
        assert child.session is not None and child.session.status.value == "interrupted"
        assert old.process.poll() is None
        old.signal("release")
        assert old.wait_success(timeout=60)["late_return_observed"]
        late_parent = asyncio.run(harness.load_session_state(parent_id))
        late_child = asyncio.run(harness.load_session_state(reached["child_session_id"]))
        assert late_parent.session == parent.session
        assert late_parent.transcript == parent.transcript
        assert late_child.session == child.session
        assert late_child.transcript == child.transcript
        assert [event for event in late_child.events if event.type == "session.interrupted"] == [
            event for event in child.events if event.type == "session.interrupted"
        ]
        assert not any(event.type == "session.completed" for event in late_child.events)
        assert sum(marker.get("child", False) for marker in harness.read_marker()) == 1
        assert sum(marker.get("late_child_return", False) for marker in harness.read_marker()) == 1


@pytest.mark.parametrize(
    "phase",
    [
        "before_child_creation",
        "after_child_creation",
        "during_child",
        "after_child_completion",
        "after_parent_terminal",
    ],
)
def test_foreground_result_recovery_after_sigkill(tmp_path, phase):
    backend = BackendConfig.sqlite(tmp_path)
    parent_id = "foreground-process-loss"
    with RecoveryHarness(tmp_path, backend) as harness:
        worker = harness.launch(
            scenario="foreground_subagent",
            action="start",
            session_id=parent_id,
            crash_phase=phase,
        )
        reached = worker.wait_for_phase(phase)
        worker.sigkill()
        assert worker.process.returncode == -signal.SIGKILL
        recovered = harness.launch(
            scenario="foreground_subagent",
            action="recover",
            session_id=parent_id,
            crash_phase=phase,
            child_session_id=reached["child_session_id"],
        ).wait_success(timeout=60.0)
        state = asyncio.run(harness.load_session_state(parent_id))
        terminals = [
            event
            for event in state.events
            if event.type in {"tool.call.completed", "tool.call.failed"}
        ]
        markers = harness.read_marker()
        assert sum(marker["child"] for marker in markers) == int(
            phase in {"during_child", "after_child_completion", "after_parent_terminal"}
        )
        if phase == "before_child_creation":
            assert recovered["status"] == "interrupted"
            assert not terminals
            assert len(markers) == 1
            return
        assert recovered["status"] == "completed"
        assert len(terminals) == 1
        assert (
            sum(
                isinstance(part, ToolResultPart)
                for message in state.transcript
                for part in message.content
            )
            == 1
        )
        if reached["live_result"] is not None:
            assert terminals[0].payload["result"] == reached["live_result"]
        else:
            assert terminals[0].type == "tool.call.failed"
            assert terminals[0].payload["result"]["structured"]["status"] == "session.interrupted"
        assert sum(marker["continuation"] for marker in markers) == 1
        replay = harness.launch(
            scenario="foreground_subagent",
            action="recover",
            session_id=parent_id,
            crash_phase=phase,
            child_session_id=reached["child_session_id"],
        ).wait_success(timeout=60.0)
        assert replay["status"] == "completed"
        replay_state = asyncio.run(harness.load_session_state(parent_id))
        assert replay_state.transcript == state.transcript
        assert [
            event
            for event in replay_state.events
            if event.type in {"tool.call.completed", "tool.call.failed"}
        ] == terminals
        assert harness.read_marker() == markers
