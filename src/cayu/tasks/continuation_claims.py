"""Expected attachment and operation identity for an exact continuation claim."""

from __future__ import annotations

from hashlib import sha256

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    require_durable_clean_nonblank,
    revalidate_model_input,
)
from cayu.sessions.invocation import SessionInvocationBinding
from cayu.tasks.records import Task, TaskClaimLost


class TaskContinuationUnsupported(NotImplementedError):
    """The task store cannot authenticate a bound controlled continuation."""


class TaskContinuationExpectation(BaseModel):
    """Bind a new lease to the selected attachment, handoff and operation.

    The operation digest includes the selecting owner's exact pause and decision
    tuple. It binds acknowledgement replay; the runtime still owns pause admission.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    task_id: str
    session_id: str
    session_instance_id: str
    prior_handoff_id: str
    operation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("task_id", "session_id", "prior_handoff_id")
    @classmethod
    def clean_identity(cls, value: str, info) -> str:
        return require_durable_clean_nonblank(value, info.field_name)

    @field_validator("session_instance_id")
    @classmethod
    def clean_instance(cls, value: str) -> str:
        return SessionInvocationBinding.validate_session_instance_id(value)


def prepare_continuation_expectation(
    expected: TaskContinuationExpectation | None, *, task_id: str | None
) -> tuple[TaskContinuationExpectation | None, str | None]:
    if expected is None:
        return None, None
    if type(expected) is not TaskContinuationExpectation:
        raise TypeError("Expected a TaskContinuationExpectation.")
    expected = revalidate_model_input(expected, TaskContinuationExpectation)
    if task_id != expected.task_id:
        raise ValueError("Bound continuation requires its exact task selector.")
    digest = sha256(
        canonical_durable_json_bytes(
            {"schema": "cayu.task-continuation-expectation.v1", **expected.model_dump()},
            "task_continuation_expectation",
        )
    ).hexdigest()
    return expected, digest


def require_expected_continuation(task: Task, expected: TaskContinuationExpectation | None) -> None:
    if expected is not None and (
        task.id != expected.task_id
        or task.session_id != expected.session_id
        or task.session_instance_id != expected.session_instance_id
        or task.interrupted_handoff_id != expected.prior_handoff_id
    ):
        raise TaskClaimLost("Selected task continuation authority changed before claim.")
