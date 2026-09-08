from __future__ import annotations

import asyncio
from contextlib import suppress

import pytest

from cayu._validation import DurableValueError
from cayu.providers.base import ModelProviderError
from cayu.runtime.context import (
    _await_owned_compaction_provider_stream,
    _CompactionCompletionObservationError,
    _CompactionCompletionValueError,
    _CompactionToolCallError,
)


@pytest.mark.parametrize(
    "outcome",
    ["success", "tool_call", "malformed", "accounting", "cancelled", "opaque", "subclass"],
)
@pytest.mark.parametrize("repeated", [False, True])
def test_caller_cancellation_preserves_only_governed_completion_metadata(
    outcome: str, repeated: bool
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        settling = asyncio.Event()
        metadata = {"usage": {"input_tokens": 13, "output_tokens": 7}}

        class OpaqueToolError(_CompactionToolCallError):
            pass

        async def operation():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                settling.set()
                if repeated:
                    with suppress(asyncio.CancelledError):
                        await asyncio.Event().wait()
                if outcome == "malformed":
                    raise _CompactionCompletionValueError(
                        error=DurableValueError("invalid_json_type", "metadata"),
                        completed_metadata=metadata,
                    ) from None
                if outcome == "accounting":
                    raise _CompactionCompletionObservationError(
                        error=ModelProviderError("billing failed", provider="test"),
                        completed_metadata=metadata,
                    ) from None
                if outcome == "success":
                    return "summary", metadata
                if outcome == "tool_call":
                    raise _CompactionToolCallError(completed_metadata=metadata) from None
                if outcome == "subclass":
                    raise OpaqueToolError(completed_metadata=metadata) from None
                error = (
                    asyncio.CancelledError("governed")
                    if outcome == "cancelled"
                    else RuntimeError("opaque")
                )
                error.__dict__["completed_metadata"] = metadata
                raise error from None

        task = asyncio.create_task(_await_owned_compaction_provider_stream(operation()))
        await started.wait()
        task.cancel("original caller signal")
        if repeated:
            await settling.wait()
            task.cancel("later caller signal")
        with pytest.raises(asyncio.CancelledError, match="original caller signal") as caught:
            await task
        if outcome in {"success", "tool_call", "malformed", "accounting", "cancelled"}:
            assert caught.value.__dict__["completed_metadata"] == metadata
            assert caught.value.__dict__["completed_metadata"] is not metadata
        else:
            assert "completed_metadata" not in caught.value.__dict__

    asyncio.run(scenario())
