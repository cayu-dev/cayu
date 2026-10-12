"""Compaction support."""

from __future__ import annotations

import hashlib
import os
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

from tests.runtime.runtime_provider_support import FakeProvider

from cayu.budgets.pricing import ModelPrice, PriceBook
from cayu.context.base import CompactionRequest, CompactionResult, ContextCompactor
from cayu.events import Event, EventType
from cayu.providers.base import (
    ModelProvider,
    ModelProviderError,
    ModelRequest,
    ModelStreamEvent,
    UsageDialect,
)
from cayu.sessions.records import Session
from cayu.storage.sqlite import SQLiteSessionStore


class CompactionIdentityMutatingProvider(FakeProvider):
    name = "gateway"
    billing_provider_name = "billco"
    usage_dialect = UsageDialect.ANTHROPIC

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.name = "poisoned\x00provider"
        self.billing_provider_name = "poisoned\ud800billing"
        self.usage_dialect = UsageDialect.GENERIC
        async for event in super().stream(request):
            yield event


class RecordingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.requests: list[CompactionRequest] = []

    def provider_budget_identity(self, session: Session) -> None:
        del session
        return None

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        compacted_text = "|".join(message.content[0].text for message in request.messages)
        if request.existing_summary is not None:
            summary = f"{request.existing_summary}|{compacted_text}"
        else:
            summary = compacted_text
        return CompactionResult(
            summary=summary,
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                if request.existing_summary is not None
                else None
            ),
            metadata={"request_count": len(self.requests)},
        )


def _expected_compaction_failure(
    event: Event,
    *,
    reason: str = "internal_failed",
) -> dict[str, Any]:
    assert event.type == EventType.CONTEXT_COMPACTION_FAILED
    elapsed_ms = event.payload["elapsed_ms"]
    assert type(elapsed_ms) is int and elapsed_ms >= 0
    expected = {
        "elapsed_ms": elapsed_ms,
        "phase": "provider_dispatch",
        "provider_dispatch_disposition": "unknown",
        "reason": reason,
        "recovery_action": "reconcile_completion",
        "retryable": False,
    }
    assert {key: event.payload[key] for key in expected} == expected
    return expected


def compaction_price_book() -> PriceBook:
    return PriceBook(
        prices=(
            ModelPrice.fixed(
                provider_name="fake",
                model="fake-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
            ModelPrice.fixed(
                provider_name="fake",
                model="summary-model",
                input_per_million=Decimal("1"),
                output_per_million=Decimal("10"),
            ),
        )
    )


class FlakyCompactionProvider(ModelProvider):
    name = "flaky"

    def __init__(self, *, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            raise self.error
        yield ModelStreamEvent.text_delta("recovered summary")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class CompletedThenRetryableCompactionProvider(ModelProvider):
    name = "retry-after-completed"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        attempt = len(self.requests)
        yield ModelStreamEvent.text_delta(f"summary {attempt}")
        yield ModelStreamEvent.completed(
            {
                "model": request.model,
                "usage": {"input_tokens": attempt, "output_tokens": 1},
            }
        )
        if attempt == 1:
            raise ModelProviderError(
                "transport failed after completed",
                provider=self.name,
                status_code=503,
                retryable=True,
            )


_AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE = 88


class ProcessLossAutomaticCompactionStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, database: str, *, crash_point: str) -> None:
        super().__init__(database)
        self._crash_point = crash_point

    async def complete_model_completion_stage(self, session_id: str, **kwargs):
        result = await super().complete_model_completion_stage(session_id, **kwargs)
        if (
            self._crash_point == "completed"
            and result.stage.purpose == "context-compaction"
            and not result.replayed
        ):
            os._exit(_AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE)
        return result

    async def promote_model_completion_stage(
        self,
        session_id: str,
        *,
        stage_id: str,
        expected_run_epoch: int,
    ):
        active = await self.load_active_model_completion_stage(session_id)
        result = await super().promote_model_completion_stage(
            session_id,
            stage_id=stage_id,
            expected_run_epoch=expected_run_epoch,
        )
        if (
            self._crash_point == "promoted"
            and active is not None
            and active.stage.purpose == "context-compaction"
        ):
            os._exit(_AUTOMATIC_COMPACTION_PROCESS_EXIT_CODE)
        return result
