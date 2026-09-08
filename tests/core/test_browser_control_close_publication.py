"""Exact close-capability characterization; public race coverage is separate."""

import pytest
from tests.core.test_browser_control import identity

from cayu.runtime._browser_control_checkpoint import (
    BrowserControlCheckpointMutation,
    BrowserControlCloseCheckpointMutation,
    browser_control_checkpoint_mutation_scope,
    project_browser_control_checkpoint,
    require_browser_control_operation_owner,
)
from cayu.runtime.browser_control import (
    BrowserControlCheckpoint,
    BrowserControlConflict,
    BrowserControlRecord,
    closed_browser_control_successor,
)
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY as KEY


def fixture():
    source = BrowserControlRecord(identity=identity())
    before = BrowserControlCheckpoint(records=(source,))
    closed = closed_browser_control_successor(source)
    assert closed is not None
    desired = before.replace_record(expected=source, desired=closed)
    fenced = source.model_copy(
        update={"revision": source.revision + 1, "state": "control_uncertain"}
    )
    return source, before, desired, fenced


@pytest.mark.parametrize("specialized", [False, True])
def test_only_confirmed_close_capability_accepts_exact_disconnect(specialized):
    source, before, desired, fenced = fixture()
    mutation_type = (
        BrowserControlCloseCheckpointMutation if specialized else BrowserControlCheckpointMutation
    )
    current = before.replace_record(expected=source, desired=fenced)
    with browser_control_checkpoint_mutation_scope(mutation_type("session", before, desired)):
        if not specialized:
            with pytest.raises(BrowserControlConflict):
                project_browser_control_checkpoint(
                    {KEY: current.model_dump()}, {KEY: desired.model_dump()}, session_id="session"
                )
            return
        actual = project_browser_control_checkpoint(
            {KEY: current.model_dump()}, {KEY: desired.model_dump()}, session_id="session"
        )
        closed = closed_browser_control_successor(fenced)
        assert closed is not None
        expected = current.replace_record(expected=fenced, desired=closed)
        assert actual == expected.model_dump(mode="json")
        with pytest.raises(BrowserControlConflict):
            require_browser_control_operation_owner("browser-control:unrelated")


@pytest.mark.parametrize(
    "change",
    [
        {"revision": 3},
        {"identity": identity().model_copy(update={"run_epoch": 2})},
        {"capture_restricted": True},
        {"fresh_observation_required": True},
        {"identity": identity().model_copy(update={"worker_instance_id": "other"})},
    ],
)
def test_close_cannot_adopt_different_fenced_authority(change):
    _, before, desired, fenced = fixture()
    current = BrowserControlCheckpoint(records=(fenced.model_copy(update=change),))
    mutation = BrowserControlCloseCheckpointMutation("session", before, desired)
    with browser_control_checkpoint_mutation_scope(mutation), pytest.raises(BrowserControlConflict):
        project_browser_control_checkpoint(
            {KEY: current.model_dump()}, {KEY: desired.model_dump()}, session_id="session"
        )


def test_close_capability_cannot_authorize_an_ordinary_transition():
    source, before, _, fenced = fixture()
    with pytest.raises(BrowserControlConflict):
        BrowserControlCloseCheckpointMutation(
            "session", before, before.replace_record(expected=source, desired=fenced)
        )
