"""Protocol selection characterization; owners still authenticate every effect."""

import pytest

from cayu.collaboration._contracts import OperationRef
from cayu.collaboration._host_output_selection import select_producer_output
from cayu.collaboration._producer_inspection import (
    ProducerDestinationInspection,
    ProducerOutputInspection,
)
from cayu.collaboration._producer_recovery import ProducerOutputRecovery


def operation(key):
    return OperationRef(
        application_scope="host", namespace_incarnation="one", generation=1, caller_key=key
    )


def snapshot(
    *, request_state="open", disposition="answer", export=None, delivery=None, cleanup=None
):
    return ProducerOutputInspection(
        recovery=ProducerOutputRecovery(
            registration=operation("producer"),
            registration_commitment="sha256:" + "a" * 64,
        ),
        state="launch_claimed",
        completion=operation("completed"),
        cleanup=None,
        cleanup_ack=None,
        exports=(),
        deliveries=(),
        exclusions=(),
        request_state=request_state,
        completion_disposition=disposition,
        answer_destination=None,
        destinations=(
            ProducerDestinationInspection(
                destination=operation("destination"),
                export=export,
                delivery=delivery,
                export_cleanup=cleanup,
            ),
        ),
    )


@pytest.mark.parametrize(
    "values,action",
    [
        ({}, "export"),
        ({"export": "prepared"}, "export"),
        ({"export": "published"}, "publish_answer"),
        ({"export": "rejected"}, "publish_failure"),
        ({"disposition": "failed"}, "publish_failure"),
        ({"request_state": "answered"}, "export"),
        ({"request_state": "answered", "export": "published", "delivery": "pending"}, "deliver"),
        (
            {"request_state": "answered", "export": "published", "delivery": "appended"},
            "release_export",
        ),
        (
            {
                "request_state": "answered",
                "export": "published",
                "delivery": "appended",
                "cleanup": "released",
            },
            "settle",
        ),
        ({"request_state": "answered", "export": "published", "delivery": "excluded"}, "retire"),
        (
            {
                "request_state": "answered",
                "export": "published",
                "delivery": "excluded",
                "cleanup": "retired",
            },
            "settle",
        ),
        ({"request_state": "failed", "disposition": "failed"}, "settle"),
        ({"request_state": "failed", "export": "rejected"}, "exclude"),
        ({"request_state": "failed", "export": "rejected", "delivery": "excluded"}, "retire"),
        (
            {
                "request_state": "failed",
                "export": "rejected",
                "delivery": "excluded",
                "cleanup": "excluded",
            },
            "settle",
        ),
    ],
)
def test_next_step_requires_its_preceding_evidence(values, action):
    observed = snapshot(**values)
    selected = select_producer_output(observed)
    assert selected.blocked is None
    assert selected.intent.action == action
    assert selected.intent.recovery == observed.recovery
    assert selected.intent.destination == (
        None if action in ("publish_failure", "settle") else operation("destination")
    )


def test_one_finished_destination_does_not_settle_the_broadcast():
    observed = snapshot(
        request_state="answered", export="published", delivery="appended", cleanup="released"
    )
    second = ProducerDestinationInspection(
        destination=operation("second"), export=None, delivery=None
    )
    observed = observed.model_copy(update={"destinations": (*observed.destinations, second)})
    selected = select_producer_output(observed)
    assert selected.intent.action == "export"
    assert selected.intent.destination == operation("second")


def test_closure_and_missing_completion_do_not_manufacture_an_answer():
    observed = snapshot(request_state="cancelled")
    assert select_producer_output(observed).blocked == "closure"
    observed = snapshot().model_copy(update={"completion": None, "completion_disposition": None})
    assert select_producer_output(observed).intent.action == "retain_completion"
    observed = observed.model_copy(update={"state": "registered"})
    assert select_producer_output(observed).blocked == "completion"
