from __future__ import annotations

from collections.abc import Iterable, Sequence
from enum import StrEnum
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import copy_durable_metadata, copy_json_value, require_clean_nonblank
from cayu.messages import (
    Message,
    MessageRole,
    TextPart,
    ToolResultPart,
    copy_message,
    detach_message,
)
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity

if TYPE_CHECKING:
    from cayu.runtime.model_steps import AssistantStepResult, StepClassification
    from cayu.sessions.base import Session


class BeforeStopAction(StrEnum):
    COMPLETE = "complete"
    CONTINUE = "continue"
    INTERRUPT = "interrupt"
    FAIL = "fail"


class BeforeStopDecision(BaseModel):
    """Control decision returned before Cayu marks a no-tool-call step complete."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

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
        return copy_durable_metadata(value, "metadata")

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
        messages: Sequence[Message] = (),
        continuation_message_indices: Iterable[int] = (),
    ) -> None:
        self._session = session.model_copy(deep=True)
        self._messages = tuple(messages)
        self._continuation_message_indices = frozenset(continuation_message_indices)
        if any(
            type(index) is not int or not 0 <= index < len(self._messages)
            for index in self._continuation_message_indices
        ):
            raise ValueError("Continuation message indices must address context messages.")
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

    @property
    def messages(self) -> tuple[Message, ...]:
        """Durable conversation messages at this boundary, oldest first.

        Includes tool results and runtime-authored continuation messages. An
        empty model step that produced nothing is not a message and is absent.
        """

        return tuple(detach_message(message) for message in self._messages)

    @property
    def continuation_message_indices(self) -> frozenset[int]:
        """Positions of this policy's retained, durably appended continuation sequence."""

        return self._continuation_message_indices


class LoopPolicy:
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity | None:
        """Return a stable application declaration, or ``None`` when non-portable."""

        return None

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def adoption_replay_identity(self) -> str | None:
        """Return stable, versioned identity for explicit profile-adoption replay.

        Stateful request policies must override this property when they can be
        used with ``ResumeRequest.profile_adoption``. The identity must change
        whenever configuration that can affect policy behavior changes. A
        missing identity fails that combination closed without affecting
        ordinary run or resume requests.
        """

        return None

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        """Run when the model step has no tool calls and Cayu is about to complete."""

        return BeforeStopDecision.complete()


class RequireFinalTool(LoopPolicy):
    """Keep a turn going until one of ``tool_names`` has returned without error.

    Some agents must end every turn through a specific tool, for example a
    clarification question or a recorded proposal. Models occasionally stop
    before calling one, sometimes with an empty step. This policy appends a
    reminder and lets the model continue, up to ``max_reminders`` times, and then
    interrupts (or fails) the session with a reason instead of completing it.

    The turn is read from the durable conversation: messages after the latest
    user message other than this policy's own durably receipted reminder. Matching
    reminder text in a real user message still starts a new turn. A tool result counts
    when its ``tool_name`` is listed and ``is_error`` is false, so tools should
    return ``is_error=True`` for failures. The policy keeps no state between
    calls; continuation provenance is checkpointed with each reminder so counts
    survive restarts and recovery.
    """

    def __init__(
        self,
        tool_names: Iterable[str],
        *,
        reminder: str | None = None,
        max_reminders: int = 2,
        on_exhausted: BeforeStopAction = BeforeStopAction.INTERRUPT,
    ) -> None:
        if isinstance(tool_names, (str, bytes)):
            raise TypeError("tool_names must be an iterable of tool names.")
        names = tuple(require_clean_nonblank(name, "tool_names") for name in tool_names)
        if not names:
            raise ValueError("tool_names must name at least one tool.")
        if len(set(names)) != len(names):
            raise ValueError("tool_names must not contain duplicates.")
        if type(max_reminders) is not int or max_reminders < 0:
            raise ValueError("max_reminders must be a non-negative integer.")
        on_exhausted = BeforeStopAction(on_exhausted)
        if on_exhausted not in (BeforeStopAction.INTERRUPT, BeforeStopAction.FAIL):
            raise ValueError("on_exhausted must be interrupt or fail.")
        listed = ", ".join(names)
        self._tool_names = names
        self._reminder = require_clean_nonblank(
            reminder
            if reminder is not None
            else f"You have not finished this turn. End it by calling one of: {listed}.",
            "reminder",
        )
        self._max_reminders = max_reminders
        self._on_exhausted = on_exhausted
        material = "\0".join((*names, self._reminder, str(max_reminders), on_exhausted.value))
        self._digest = sha256(material.encode("utf-8")).hexdigest()

    @property
    def tool_names(self) -> tuple[str, ...]:
        return self._tool_names

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="cayu.loop.require_final_tool",
            behavior_version="1",
            implementation_version=f"sha256:{self._digest}",
        )

    @property
    def adoption_replay_identity(self) -> str:
        return f"cayu.loop.require_final_tool:1:sha256:{self._digest}"

    async def before_stop(self, context: BeforeStopContext) -> BeforeStopDecision:
        reminders = 0
        for index in range(len(context._messages) - 1, -1, -1):
            message = context._messages[index]
            for part in message.content:
                if (
                    type(part) is ToolResultPart
                    and part.tool_name in self._tool_names
                    and not part.is_error
                ):
                    return BeforeStopDecision.complete(
                        "final tool returned", tool_name=part.tool_name
                    )
            if message.role != MessageRole.USER or any(
                type(part) is ToolResultPart for part in message.content
            ):
                continue
            if index in context.continuation_message_indices and self._is_reminder(message):
                reminders += 1
                continue
            break
        if reminders < self._max_reminders:
            return BeforeStopDecision.continue_with(
                Message.text("user", self._reminder),
                reason="final tool required",
                metadata={"reminder": reminders + 1, "tool_names": list(self._tool_names)},
            )
        reason = (
            f"Turn ended without a successful call to one of: {', '.join(self._tool_names)} "
            f"after {reminders} reminder(s)."
        )
        metadata = {"reminders": reminders, "tool_names": list(self._tool_names)}
        if self._on_exhausted == BeforeStopAction.FAIL:
            return BeforeStopDecision.fail(reason, metadata=metadata)
        return BeforeStopDecision.interrupt(reason, metadata=metadata)

    def _is_reminder(self, message: Message) -> bool:
        return (
            len(message.content) == 1
            and type(message.content[0]) is TextPart
            and message.content[0].text == self._reminder
        )


def copy_before_stop_decision(decision: BeforeStopDecision) -> BeforeStopDecision:
    """Return an owned copy of a policy-provided decision.

    `BeforeStopDecision` is frozen and `Message` owns its payloads, so the only
    live reference the policy can still mutate is the `metadata` dict — copy
    just that at this trust boundary instead of rebuilding every field.
    """
    if type(decision) is not BeforeStopDecision:
        raise TypeError("Loop policies must return BeforeStopDecision instances.")
    return decision.model_copy(update={"metadata": copy_json_value(decision.metadata, "metadata")})


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
