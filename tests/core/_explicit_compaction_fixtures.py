"""Shared compaction test doubles and complete session-authority setup."""

from __future__ import annotations

import hashlib
from typing import Any

from tests.core._execution_profile_fixtures import profiled_session_identity

from cayu.applications import CayuApp
from cayu.budgets.base import InMemoryBudgetLedger
from cayu.budgets.billing import BillingIdentity
from cayu.context.base import CompactionRequest, CompactionResult, ContextCompactor
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.providers.bedrock import completed_bedrock_billing_identity
from cayu.runtime._event_projection import public_event_id
from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu.sessions.event_queries import EventQuery
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import RunRequest
from cayu.tools.exposure import ToolCapabilityCeiling


class RecordingCompactor(ContextCompactor):
    def __init__(self) -> None:
        self.requests: list[CompactionRequest] = []

    def provider_budget_identity(self, _session) -> None:
        return None

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:recording-context-compactor",
            behavior_version="1",
            implementation_version="1",
        )

    async def compact(self, request: CompactionRequest) -> CompactionResult:
        self.requests.append(request)
        summary = "durable compact summary"
        if request.existing_summary is not None:
            summary = f"{request.existing_summary}|{summary}"
        return CompactionResult(
            summary=summary,
            covered_message_count=len(request.messages),
            represented_existing_summary_sha256=(
                hashlib.sha256(request.existing_summary.encode("utf-8")).hexdigest()
                if request.existing_summary is not None
                else None
            ),
            metadata={"compactor": type(self).__name__, "mode": "deterministic"},
        )


async def _public_ids_for_private_event_ids(
    store: Any,
    session_id: str,
    event_ids: list[str],
) -> list[str]:
    records = await store.query_events(EventQuery(session_id=session_id, limit=5000))
    sequence_by_id = {record.event.id: record.sequence for record in records}
    return [public_event_id(sequence_by_id[event_id]) for event_id in event_ids]


async def _create_profiled_session(
    app: CayuApp,
    store: Any,
    request: RunRequest,
    *,
    identity: SessionIdentity,
    **kwargs: Any,
):
    """Create a low-level compaction fixture with complete runtime authority."""

    registered_agent = app._agents[request.agent_name]
    ceiling = ToolCapabilityCeiling(tool_names=tuple(registered_agent.tools))
    prepared_request = request.model_copy(update={"tool_capability_ceiling": ceiling})
    if identity.execution_profile is None:
        identity = profiled_session_identity(
            provider_name=identity.provider_name,
            model=identity.model,
            agent_name=request.agent_name,
            app=app,
            invocation_loop_policies=request.loop_policies,
            structured_output=request.structured_output,
            thinking=request.thinking,
            request_budget_limits=request.budget_limits,
            causal_budget_id=(request.causal_budget_id or request.task_id or request.session_id),
            max_steps=request.max_steps,
            limits=request.limits,
            retry_policy=app._effective_retry_policy(request.retry_policy),
        )
    return await store.create(prepared_request, identity=identity, **kwargs)


class CompletingProvider(ModelProvider):
    name = "fake"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:completing-provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest):
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({})


class UsageCompactionProvider(ModelProvider):
    name = "compaction-provider"

    def __init__(self, *, summary: str = "provider summary") -> None:
        self.calls = 0
        self.summary = summary

    async def stream(self, request: ModelRequest):
        self.calls += 1
        yield ModelStreamEvent.text_delta(self.summary)
        yield ModelStreamEvent.completed({"usage": {"input_tokens": 8, "output_tokens": 2}})


class ContextualBedrockCompactionProvider(ModelProvider):
    name = "bedrock"
    billing_provider_name = "bedrock"

    def __init__(self, identity: BillingIdentity, *, report_usage: bool = True) -> None:
        self.identity = identity
        self.report_usage = report_usage
        self.calls = 0

    async def billing_identity_for_request(
        self,
        request: ModelRequest,
    ) -> BillingIdentity:
        assert request.model == self.identity.resource_id
        return self.identity

    def billing_identity_for_completion(
        self,
        identity: BillingIdentity | None,
        payload: dict,
    ) -> BillingIdentity | None:
        assert identity == self.identity
        return completed_bedrock_billing_identity(
            self.identity,
            effective_service_tier="default",
        )

    async def stream(self, request: ModelRequest):
        self.calls += 1
        yield ModelStreamEvent.text_delta("provider summary")
        completed: dict[str, object] = {"model": request.model}
        if self.report_usage:
            completed["usage"] = {"input_tokens": 8, "output_tokens": 2}
        yield ModelStreamEvent.completed(completed)


class FinalRenewalFailureBudgetLedger(InMemoryBudgetLedger):
    def __init__(self) -> None:
        super().__init__()
        self.heartbeat_calls = 0

    async def heartbeat(self, *, reservation_id: str) -> bool:
        self.heartbeat_calls += 1
        if self.heartbeat_calls == 2:
            return False
        return await super().heartbeat(reservation_id=reservation_id)
