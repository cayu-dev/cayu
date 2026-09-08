"""Epoch preparation and atomic admission must agree; guest checks separately."""

import pytest
from tests.core.test_browser_control import identity

from cayu.runtime._browser_control_model import (
    browser_model_control_epoch,
    validate_browser_model_publication,
)
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlCheckpoint,
    BrowserControlConflict,
    BrowserControlRecord,
)
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.runtime.sessions import Session


@pytest.mark.parametrize("epoch", [None, True, 0, 2, 1])
@pytest.mark.parametrize("state", ["intent", "dispatched"])
def test_atomic_model_admission_requires_prepared_exact_epoch(epoch, state):
    exact = identity()
    record = BrowserControlRecord(identity=exact)
    checkpoint = {
        BROWSER_CONTROLS_CHECKPOINT_KEY: BrowserControlCheckpoint(records=(record,)).model_dump(
            mode="json"
        )
    }
    allocation = BrowserControlAllocation.model_validate(
        exact.model_dump(exclude={"worker_instance_id"})
    )
    assert (
        browser_model_control_epoch(checkpoint, allocation=allocation, operation_name="observe")
        == 1
    )
    operation = {
        "record_type": "cayu.browser-operation",
        "schema_version": 1,
        "state": state,
        "browser_session_id": exact.browser_session_id,
        "parent_session_id": exact.session_id,
        "parent_run_epoch": exact.run_epoch,
        "execution_profile_fingerprint": exact.execution_profile_fingerprint,
        "environment_name": exact.environment_name,
        "allocation_fingerprint": exact.allocation_fingerprint,
        "invocation_control_epoch": epoch,
    }

    def admit():
        validate_browser_model_publication(
            checkpoint,
            session=Session.model_construct(
                id=exact.session_id,
                instance_id=exact.session_instance_id,
                run_epoch=exact.run_epoch,
            ),
            operation_records={"operation": operation},
            operation_name="observe",
        )

    if type(epoch) is int and epoch == 1:
        admit()
    else:
        with pytest.raises(BrowserControlConflict, match="epoch"):
            admit()


def test_model_epoch_rejects_same_browser_under_another_allocation():
    exact = identity()
    checkpoint = {
        BROWSER_CONTROLS_CHECKPOINT_KEY: BrowserControlCheckpoint(
            records=(BrowserControlRecord(identity=exact),)
        ).model_dump(mode="json")
    }
    allocation = BrowserControlAllocation.model_validate(
        {**exact.model_dump(exclude={"worker_instance_id"}), "run_epoch": exact.run_epoch + 1}
    )
    with pytest.raises(BrowserControlConflict):
        browser_model_control_epoch(checkpoint, allocation=allocation, operation_name="observe")
