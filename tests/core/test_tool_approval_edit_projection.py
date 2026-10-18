"""An externalized edited result keeps an exact projection record and a readable artifact.

The record's byte and token accounting must describe the published content,
notice included; otherwise the runtime no longer recognizes the projection
and stops protecting its artifact reference during publication. (A registered
secret overlapping an artifact id already fails the approval checkpoint secret
check on main, so that case is not exercised end to end here.)
"""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import aclosing
from datetime import UTC, datetime

from pydantic import SecretStr

from cayu import AgentSpec, CayuApp, Message, RunRequest
from cayu.approvals.tools import ToolApprovalDecision, ToolApprovalRequest
from cayu.artifacts import LocalArtifactStore
from cayu.artifacts.settlement import (
    ArtifactWriteSettlementEvidence,
    ArtifactWriteSettlementFailureCode,
    ArtifactWriteSettlementObservation,
    ArtifactWriteSettlementPhase,
    ArtifactWriteSettlementStatus,
    artifact_store_identity_sha256,
    record_artifact_write_settlement,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.events import EventType
from cayu.messages import ToolResultPart
from cayu.providers import ModelProvider, ModelStreamEvent
from cayu.runtime._tool_execution import APPROVED_ARGUMENTS_NOTICE
from cayu.runtime._tool_result_projection import read_runtime_tool_result_projection
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.memory import InMemorySessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.tools.files import ReadFileTool
from cayu.tools.policy import ToolPolicy, ToolPolicyDecision, ToolPolicyResult
from cayu.tools.result_projection import (
    MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES,
    ArtifactExternalizingToolResultPolicy,
    ToolResultProjection,
    ToolResultProjectionPolicy,
    ToolResultProjectionRecord,
    ToolResultProjectionStatus,
    reestimate_tool_result_projection_tokens,
)

_OUTPUT = "row " * 3000 + "TAIL-SENTINEL"


class _ReportTool(Tool):
    spec = ToolSpec(
        name="report",
        description="Build a large report.",
        input_schema={"type": "object", "properties": {"month": {"type": "string"}}},
    )

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self.calls.append(dict(args))
        return ToolResult(content=_OUTPUT)


class _ReportApproval(ToolPolicy):
    async def authorize(self, request):
        if request.tool_name == "report":
            return ToolPolicyResult(decision=ToolPolicyDecision.REQUIRE_APPROVAL)
        return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)


class _ReportProvider(ModelProvider):
    """Calls report, then records the tool result the model received."""

    name = "report"

    def __init__(self) -> None:
        self.requests = []
        self.seen = None

    async def stream(self, request):
        self.requests.append(request)
        results = [
            part
            for message in request.messages
            if message.role == "tool"
            for part in message.content
            if type(part) is ToolResultPart
        ]
        if not results:
            yield ModelStreamEvent.tool_call(id="pause", name="report", arguments={"month": "09"})
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        self.seen = results[-1]
        yield ModelStreamEvent.text_delta("done")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


def test_an_externalized_edited_result_keeps_its_projection_and_artifact(tmp_path):
    key = base64.urlsafe_b64encode(bytes([71]) * 32).decode().rstrip("=")
    store = InMemorySessionStore(
        public_authority_alias_codec=PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(active_key_id="test", keys={"test": SecretStr(key)})
        )
    )
    provider = _ReportProvider()
    tool = _ReportTool()
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=4096, max_inline_token_estimate=None, preview_bytes=64
        ),
    )
    app.register_provider(provider, default=True)
    artifacts = LocalArtifactStore(tmp_path / "artifacts", store_id="edit")
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), artifact_store=artifacts), default=True
    )
    app.register_agent(
        AgentSpec(name="assistant", model="readback-model"),
        tools=[tool, ReadFileTool()],
        tool_policy=_ReportApproval(),
    )

    async def scenario():
        async with aclosing(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id="edit-projection",
                    messages=[Message.text("user", "build the report")],
                )
            )
        ) as stream:
            events = [event async for event in stream]
        paused = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        async with aclosing(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="edit-projection",
                    approval_id=paused.payload["approval_id"],
                    tool_round_id=paused.payload["tool_round_id"],
                    tool_call_id=paused.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    arguments={"month": "10"},
                )
            )
        ) as stream:
            resolved = [event async for event in stream]
        assert resolved[-1].type is EventType.SESSION_COMPLETED, resolved[-1].payload
        return [
            e
            for e in await store.load_events("edit-projection")
            if e.type is EventType.TOOL_CALL_COMPLETED and e.tool_name == "report"
        ]

    [terminal] = asyncio.run(scenario())

    assert tool.calls == [{"month": "10"}]
    projection = read_runtime_tool_result_projection(terminal.payload)
    assert projection is not None
    content = terminal.payload["result"]["content"]
    assert projection.record.projected_bytes == len(content.encode())
    assert content.startswith(APPROVED_ARGUMENTS_NOTICE)
    estimate, method = reestimate_tool_result_projection_tokens(
        content, token_estimation_method=projection.record.token_estimation_method
    )
    assert (
        projection.record.projected_token_estimate,
        projection.record.token_estimation_method,
    ) == (
        estimate,
        method,
    )
    # The runtime-owned artifact reference is published intact.
    assert projection.record.artifact_id in content
    # The model received the same content, with a usable readback instruction.
    assert provider.seen is not None and provider.seen.content == content
    instruction = content.split("Use read_file with ", maxsplit=1)[1]
    arguments = json.loads(instruction.split(" to inspect", maxsplit=1)[0])
    assert arguments["artifact_id"] == projection.record.artifact_id
    # The published reference retrieves the stored output, notice included.
    readback = asyncio.run(
        ReadFileTool().run(
            ToolContext(
                session_id="edit-projection",
                environment_name="local",
                artifact_store=artifacts,
            ),
            {**arguments, "max_bytes": 64 * 1024},
        )
    )
    assert not readback.is_error, readback.content
    assert readback.content.startswith(APPROVED_ARGUMENTS_NOTICE)
    assert "TAIL-SENTINEL" in readback.content


class _BoundFillingPolicy(ToolResultProjectionPolicy):
    """Externalizes like the built-in policy, then fills the projection safety bound."""

    def __init__(self) -> None:
        self.inner = ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=4096, max_inline_token_estimate=None, preview_bytes=64
        )

    @property
    def identity(self) -> str:
        return self.inner.identity

    async def project(self, request):
        projection = await self.inner.project(request)
        padding = MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES - len(projection.result.content.encode())
        content = " " * padding + projection.result.content
        estimate, method = reestimate_tool_result_projection_tokens(
            content, token_estimation_method=projection.record.token_estimation_method
        )
        return ToolResultProjection(
            result=projection.result.model_copy(update={"content": content}),
            record=projection.record.model_copy(
                update={
                    "projected_bytes": len(content.encode()),
                    "projected_token_estimate": estimate,
                    "token_estimation_method": method,
                }
            ),
        )


def _run_edited_report(tmp_path, policy, session_id):
    """Pause the report call, approve it with changed arguments, return its terminal."""

    tool = _ReportTool()
    app = CayuApp(
        session_store=InMemorySessionStore(),
        enable_logging=False,
        tool_result_projection_policy=policy,
    )
    app.register_provider(_ReportProvider(), default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local"),
            artifact_store=LocalArtifactStore(tmp_path / "artifacts", store_id="bound"),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="bound-model"),
        tools=[tool, ReadFileTool()],
        tool_policy=_ReportApproval(),
    )

    async def scenario():
        async with aclosing(
            app.run(
                RunRequest(
                    agent_name="assistant",
                    session_id=session_id,
                    messages=[Message.text("user", "build the report")],
                )
            )
        ) as stream:
            events = [event async for event in stream]
        paused = next(e for e in events if e.type is EventType.TOOL_CALL_APPROVAL_REQUESTED)
        async with aclosing(
            app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id=session_id,
                    approval_id=paused.payload["approval_id"],
                    tool_round_id=paused.payload["tool_round_id"],
                    tool_call_id=paused.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                    arguments={"month": "10"},
                )
            )
        ) as stream:
            resolved = [event async for event in stream]
        assert resolved[-1].type is EventType.SESSION_COMPLETED, resolved[-1].payload
        return [
            e
            for e in await app.session_store.load_events(session_id)
            if e.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            and e.tool_name == "report"
        ]

    [terminal] = asyncio.run(scenario())
    assert tool.calls == [{"month": "10"}]
    return terminal


def test_a_notice_that_would_overflow_the_projection_bound_fails_the_projection_closed(
    tmp_path,
):
    terminal = _run_edited_report(tmp_path, _BoundFillingPolicy(), "edit-bound")

    record = terminal.payload["tool_result_projection"]
    assert record["status"] == "failed"
    assert record["failure_type"] == "approval_notice_exceeds_projection_bound"
    assert terminal.payload["result"]["content"].startswith(APPROVED_ARGUMENTS_NOTICE)


class _UncertainWriteBoundFillingPolicy(ToolResultProjectionPolicy):
    """Loses an upload's commit acknowledgement, then fails within the full bound."""

    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.settlement = ArtifactWriteSettlementEvidence(
            operation_id=f"artifact_write_{'3' * 32}",
            artifact_id=f"art_{'4' * 32}",
            store_identity_sha256=artifact_store_identity_sha256("uncertain-store"),
            status=ArtifactWriteSettlementStatus.RECONCILIATION_REQUIRED,
            phase=ArtifactWriteSettlementPhase.RECONCILIATION,
            observation=ArtifactWriteSettlementObservation.CALLER_BOUNDARY,
            started_at=now,
            observed_at=now,
            elapsed_ms=0,
            failure_codes=(ArtifactWriteSettlementFailureCode.COMMIT_FAILED,),
        )

    @property
    def identity(self) -> str:
        return "tests.uncertain_write_bound_filling.v1"

    async def project(self, request):
        record_artifact_write_settlement(self.settlement)
        content = " " * MAX_PROJECTED_TOOL_RESULT_CONTENT_BYTES
        return ToolResultProjection(
            result=request.result.model_copy(update={"content": content}),
            record=ToolResultProjectionRecord(
                status=ToolResultProjectionStatus.FAILED,
                policy_id=self.identity,
                original_bytes=len(request.result.content.encode("utf-8")),
                projected_bytes=len(content.encode("utf-8")),
                original_token_estimate=0,
                projected_token_estimate=0,
                token_estimation_method="tests_exact_v1",
                failure_type="artifact_store_unavailable",
                artifact_write_settlement=self.settlement,
            ),
        )


def test_a_notice_overflow_keeps_the_uncertain_artifact_write_settlement(tmp_path):
    policy = _UncertainWriteBoundFillingPolicy()
    terminal = _run_edited_report(tmp_path, policy, "edit-bound-settlement")

    record = terminal.payload["tool_result_projection"]
    assert record["failure_type"] == "approval_notice_exceeds_projection_bound"
    settlement = record["artifact_write_settlement"]
    assert settlement["status"] == "reconciliation_required"
    assert settlement["operation_id"] == policy.settlement.operation_id
    assert settlement["artifact_id"] == policy.settlement.artifact_id
    assert terminal.payload["result"]["content"].startswith(APPROVED_ARGUMENTS_NOTICE)
