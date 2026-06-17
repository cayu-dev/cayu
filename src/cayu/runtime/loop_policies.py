from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_json_value, require_clean_nonblank
from cayu.core.messages import Message, MessageRole, copy_message

if TYPE_CHECKING:
    from cayu.runtime.model_steps import AssistantStepResult, StepClassification
    from cayu.runtime.sessions import Session


class BeforeStopAction(StrEnum):
    COMPLETE = "complete"
    CONTINUE = "continue"
    INTERRUPT = "interrupt"
    FAIL = "fail"


class BeforeStopDecision(BaseModel):
    """Control decision returned before Cayu marks a no-tool-call step complete."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: BeforeStopAction = BeforeStopAction.COMPLETE
    reason: str = "complete"
    message: Message | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def complete(cls, reason: str = "complete", **metadata: Any) -> BeforeStopDecision:
        return cls(action=BeforeStopAction.COMPLETE, reason=reason, metadata=metadata)

    @classmethod
    def continue_with(
        cls,
        message: Message,
        *,
        reason: str = "continue",
        metadata: dict[str, Any] | None = None,
    ) -> BeforeStopDecision:
        return cls(
            action=BeforeStopAction.CONTINUE,
            reason=reason,
            message=message,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def interrupt(
        cls,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> BeforeStopDecision:
        return cls(
            action=BeforeStopAction.INTERRUPT,
            reason=reason,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def fail(
        cls,
        reason: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> BeforeStopDecision:
        return cls(
            action=BeforeStopAction.FAIL,
            reason=reason,
            metadata={} if metadata is None else metadata,
        )

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str, info) -> str:
        return require_clean_nonblank(value, info.field_name)

    @field_validator("message")
    @classmethod
    def copy_message(cls, value: Message | None) -> Message | None:
        if value is None:
            return None
        return copy_message(value)

    @field_validator("metadata", mode="before")
    @classmethod
    def copy_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        return copy_json_value(value, "metadata")

    @model_validator(mode="after")
    def validate_action_payload(self) -> BeforeStopDecision:
        if self.action == BeforeStopAction.CONTINUE:
            if self.message is None:
                raise ValueError("Continue before-stop decisions require a message.")
            if self.message.role != MessageRole.USER:
                raise ValueError("Continue before-stop decisions require a user message.")
        elif self.message is not None:
            raise ValueError("Only continue before-stop decisions can include a message.")
        return self


class BeforeStopContext:
    """Immutable context passed to loop policies at the before-stop boundary."""

    def __init__(
        self,
        *,
        session: Session,
        step_result: AssistantStepResult,
        classification: StepClassification,
        step: int,
        max_steps: int,
        metadata: dict[str, Any],
    ) -> None:
        self._session = session.model_copy(deep=True)
        self._step_result = step_result
        self._classification = classification
        self._step = step
        self._max_steps = max_steps
        self._metadata = copy_json_value(metadata, "metadata")

    @property
    def session(self) -> Session:
        return self._session.model_copy(deep=True)

    @property
    def step_result(self) -> AssistantStepResult:
        return self._step_result

    @property
    def classification(self) -> StepClassification:
        return self._classification

    @property
    def step(self) -> int:
        return self._step

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def metadata(self) -> dict[str, Any]:
        return copy_json_value(self._metadata, "metadata")


class LoopPolicy:
    @property
    def name(self) -> str:
        return type(self).__name__

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        """Run when the model step has no tool calls and Cayu is about to complete."""

        return BeforeStopDecision.complete()


def copy_before_stop_decision(decision: BeforeStopDecision) -> BeforeStopDecision:
    if type(decision) is not BeforeStopDecision:
        raise TypeError("Loop policies must return BeforeStopDecision instances.")
    return BeforeStopDecision(
        action=decision.action,
        reason=decision.reason,
        message=copy_message(decision.message) if decision.message is not None else None,
        metadata=copy_json_value(decision.metadata, "metadata"),
    )


def validate_loop_policies(
    policies: Iterable[LoopPolicy] | None,
    *,
    field_name: str,
) -> tuple[LoopPolicy, ...]:
    if policies is None:
        return ()
    if isinstance(policies, (str, bytes)):
        raise TypeError(f"{field_name} must be an iterable of LoopPolicy instances.")
    try:
        copied = tuple(policies)
    except TypeError as exc:
        raise TypeError(f"{field_name} must be an iterable of LoopPolicy instances.") from exc
    for policy in copied:
        if not isinstance(policy, LoopPolicy):
            raise TypeError(f"{field_name} must contain LoopPolicy instances.")
        require_clean_nonblank(policy.name, f"{field_name}.name")
    return copied
