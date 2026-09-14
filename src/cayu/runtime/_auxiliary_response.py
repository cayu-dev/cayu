"""Bounded response accumulation; dispatch and stream cleanup remain runtime-owned."""

from __future__ import annotations

from cayu._validation import canonical_bounded_durable_json_bytes
from cayu.providers.base import (
    ModelCompletion,
    ModelFinishReason,
    ModelStreamEvent,
    ModelStreamEventType,
    copy_model_stream_event,
)
from cayu.providers.response import ModelResponse


class AuxiliaryResponseCollector:
    """Retain a bounded detached response, never provider lifecycle authority.

    The caller must capture accounting observations before adding an event: a
    terminal may contain usable usage even when its response content is rejected.
    Feed sanitized events only, after validating untrusted provider observations.
    This object never advances/closes a stream or settles a provider attempt.
    """

    def __init__(self, *, max_bytes: int) -> None:
        if type(max_bytes) is not int or not 0 < max_bytes <= 8 * 1024 * 1024:
            raise ValueError("Response byte limit must be a positive integer at most 8 MiB.")
        self._max_bytes = max_bytes
        self._bytes = len(b'{"events":[]}')
        self._events: list[ModelStreamEvent] = []
        self._failed = False
        self._completed = False

    def add(self, event: ModelStreamEvent) -> None:
        self._accept(event, error=False)

    def accept_error(self, event: ModelStreamEvent) -> ModelStreamEvent:
        """Validate a bounded terminal error without making it a response.

        The stream owner, not this collector, determines whether the provider
        subsequently quiesces and whether the failed attempt may be retried.
        """
        return self._accept(event, error=True)

    def _accept(self, event: ModelStreamEvent, *, error: bool) -> ModelStreamEvent:
        if self._failed or self._completed:
            self._failed = True
            raise ValueError("Auxiliary response collector is already terminal.")
        # A rejected observation permanently prevents success; later valid output
        # cannot erase the failed response boundary.
        self._failed = True
        if type(event) is not ModelStreamEvent:
            raise TypeError("Auxiliary response requires ModelStreamEvent values.")
        allowed = (
            {ModelStreamEventType.ERROR}
            if error
            else {
                ModelStreamEventType.TEXT_DELTA,
                ModelStreamEventType.THINKING,
                ModelStreamEventType.CITATION,
                ModelStreamEventType.COMPLETED,
            }
        )
        if type(event.type) is not ModelStreamEventType or event.type not in allowed:
            raise ValueError("Auxiliary response contains an unsupported event type.")
        if any(
            value is not None
            for value in (
                event.tool_discovery_result,
                event.recovery_metadata,
                event.provider_operation_status,
            )
        ):
            raise ValueError("Auxiliary response cannot contain tool or background authority.")
        completion = event.completion
        completion_data = None
        if completion is not None:
            if (
                type(completion) is not ModelCompletion
                or type(completion.finish_reason) is not ModelFinishReason
            ):
                raise ValueError("Auxiliary response completion metadata is invalid.")
            if completion.finish_reason in {ModelFinishReason.TOOL_CALLS, ModelFinishReason.ERROR}:
                raise ValueError("Auxiliary response cannot complete with tool calls or an error.")
            completion_data = {
                "finish_reason": completion.finish_reason.value,
                "raw_finish_reason": completion.raw_finish_reason,
                "status": completion.status,
                "end_turn": completion.end_turn,
            }
        if (event.type is ModelStreamEventType.COMPLETED) != (completion is not None):
            raise ValueError("Auxiliary response completion must accompany its terminal event.")
        # Raw-field sizing precedes all copies and serializers, including rejected
        # values whose repr/str or Pydantic serializer could disclose credentials.
        raw = {
            "type": event.type.value,
            "delta": event.delta,
            "payload": event.payload,
            "completion": completion_data,
        }
        separator = int(bool(self._events))
        remaining = max(0, self._max_bytes - self._bytes - separator)
        encoded = canonical_bounded_durable_json_bytes(
            raw, "auxiliary_response", max_bytes=remaining, max_nodes=max(1, remaining)
        )
        copied = copy_model_stream_event(event)
        self._events.append(copied)
        self._bytes += len(encoded) + separator
        self._completed = copied.type is ModelStreamEventType.COMPLETED
        self._failed = error
        return copied

    def finish(self) -> ModelResponse:
        if self._failed or not self._completed:
            raise ValueError("Auxiliary response did not complete successfully.")
        return ModelResponse(events=tuple(self._events))
