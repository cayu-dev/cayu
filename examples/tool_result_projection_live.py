"""Verified live-provider contract for oversized tool-result projection."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pydantic import SecretStr

from _live_checks import require, require_positive_model_usage, require_successful_terminal
from cayu import (
    REDACTED_SECRET,
    AgentSpec,
    AnthropicProvider,
    ArtifactExternalizingToolResultPolicy,
    CayuApp,
    Environment,
    EnvironmentSpec,
    Event,
    EventType,
    LocalArtifactStore,
    Message,
    OpenAIProvider,
    PublicAuthorityAliasCodec,
    PublicAuthorityAliasKeyring,
    ReadFileTool,
    RunRequest,
    SecretRedactor,
    SQLiteSessionStore,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent

EVIDENCE_PREFIX = "CAYU_NIGHTLY_EVIDENCE="
RECEIPT_ID = "projection-live-receipt-001"
SECRET_CANARY = "cayu-projection-live-secret-canary-7d921"
REPORT_BEGIN_MARKER = "CAYU_PROJECTION_LIVE_BEGIN"
REPORT_END_MARKER = "CAYU_PROJECTION_LIVE_END"
REPORT_TEXT = (
    f"{REPORT_BEGIN_MARKER}\n"
    f"synthetic_secret={SECRET_CANARY}\n"
    + "".join(
        f"row={index:05d} value=abcdefghijklmnopqrstuvwxyz0123456789\n" for index in range(3_000)
    )
    + f"{REPORT_END_MARKER}\n"
)
SESSION_ID = "projection_live_acceptance"


def _require_dict(value: object, message: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(message)
    return cast("dict[str, Any]", value)


def _require_list(value: object, message: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(message)
    return cast("list[Any]", value)


def _require_sqlite_files_exclude(
    database: Path,
    forbidden: dict[str, bytes],
) -> None:
    for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm")):
        if not path.is_file():
            continue
        content = path.read_bytes()
        for label, marker in forbidden.items():
            require(marker not in content, f"{label} leaked into {path.name}")


class LargeReportTool(Tool):
    spec = ToolSpec(
        name="large_report",
        description="Return a deterministic oversized report for projection validation.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        effect=ToolEffect.NONE,
    )

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        del ctx, args
        self.calls += 1
        return ToolResult(
            content=REPORT_TEXT,
            structured={
                "receipt_id": RECEIPT_ID,
                "report_kind": "projection-live",
            },
        )


class RecordingProvider(ModelProvider):
    """Record provider-neutral requests while preserving the live adapter."""

    def __init__(self, delegate: ModelProvider) -> None:
        self.delegate = delegate
        self.name = delegate.name
        self.billing_provider_name = delegate.billing_provider_name
        self.usage_dialect = delegate.usage_dialect
        self.supports_native_structured_output = delegate.supports_native_structured_output
        self.requests: list[ModelRequest] = []

    @property
    def context_pressure_profile(self):
        return self.delegate.context_pressure_profile

    async def billing_identity_for_request(self, request: ModelRequest):
        return await self.delegate.billing_identity_for_request(request)

    def billing_identity_for_completion(self, identity, payload: dict[str, Any]):
        return self.delegate.billing_identity_for_completion(identity, payload)

    def preflight_native_structured_output_schema(self, json_schema: dict[str, Any]) -> None:
        self.delegate.preflight_native_structured_output_schema(json_schema)

    async def count_input_tokens(self, request: ModelRequest):
        return await self.delegate.count_input_tokens(request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request.model_copy(deep=True))
        async for event in self.delegate.stream(request):
            yield event


async def _run_contract(
    *,
    provider: ModelProvider,
    provider_name: str,
    model: str,
    root: Path,
) -> dict[str, Any]:
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Live projection root must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    database = root / "cayu.db"
    alias_keyring = PublicAuthorityAliasKeyring(
        active_key_id="live",
        keys={
            "live": SecretStr(base64.urlsafe_b64encode(os.urandom(32)).decode("ascii").rstrip("="))
        },
    )
    session_store = SQLiteSessionStore(
        database,
        public_authority_alias_codec=PublicAuthorityAliasCodec(alias_keyring),
    )
    artifact_store = LocalArtifactStore(
        root / "artifacts",
        store_id="projection-live-artifacts",
    )
    recording_provider = RecordingProvider(provider)
    large_report = LargeReportTool()
    app = CayuApp(
        enable_logging=False,
        session_store=session_store,
        secret_redactor=SecretRedactor(SECRET_CANARY),
        tool_result_projection_policy=ArtifactExternalizingToolResultPolicy(
            max_inline_bytes=2_048,
            max_inline_token_estimate=None,
            preview_bytes=128,
        ),
    )
    app.register_provider(recording_provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="local-live", metadata={"kind": "projection-live"}),
            artifact_store=artifact_store,
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(
            name="assistant",
            model=model,
            system_prompt=(
                "You are validating Cayu tool-result projection. Call large_report exactly "
                "once. From its projected result, call read_file exactly once with the returned "
                "artifact_id and max_bytes=256. Then answer briefly. Never call large_report "
                "again."
            ),
        ),
        tools=[large_report, ReadFileTool()],
    )

    events: list[Event] = []
    try:
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=SESSION_ID,
                messages=[
                    Message.text(
                        "user",
                        "Run the live projection check and inspect the projected artifact.",
                    )
                ],
            )
        ):
            events.append(event)

        evidence = _validate_runtime_events(
            events,
            provider_name=provider_name,
            model=model,
            large_report_calls=large_report.calls,
        )
        artifact_id = evidence["artifact_id"]
        stored = await artifact_store.read_bytes(artifact_id)
        expected_stored_text = REPORT_TEXT.replace(SECRET_CANARY, REDACTED_SECRET)
        require(
            stored.content.decode("utf-8") == expected_stored_text,
            "artifact store did not retain the complete redacted report",
        )
        require(stored.truncated is False, "full artifact verification was truncated")

        transcript = await session_store.load_transcript(SESSION_ID)
        serialized_transcript = json.dumps(
            [message.model_dump(mode="json") for message in transcript],
            sort_keys=True,
        )
        serialized_provider_requests = json.dumps(
            [request.model_dump(mode="json") for request in recording_provider.requests],
            sort_keys=True,
        )
        for label, serialized in (
            ("transcript", serialized_transcript),
            ("provider requests", serialized_provider_requests),
        ):
            require(SECRET_CANARY not in serialized, f"raw secret leaked into {label}")
            require(REPORT_END_MARKER not in serialized, f"unbounded report leaked into {label}")
        require(
            "cayu.tool_result_artifact.v1" in serialized_provider_requests,
            "provider requests did not receive the typed projection reference",
        )
        require(
            RECEIPT_ID in serialized_provider_requests,
            "provider requests lost the structured receipt",
        )

        _require_sqlite_files_exclude(
            database,
            {
                "raw secret": SECRET_CANARY.encode(),
                "unbounded report": REPORT_END_MARKER.encode(),
            },
        )
        return {
            **evidence,
            "provider_requests": len(recording_provider.requests),
            "artifact_bytes": len(stored.content),
            "root": str(root),
            "status": "ok",
        }
    finally:
        await session_store.close()


def _provider_config() -> tuple[str, str, ModelProvider]:
    requested = os.environ.get("CAYU_PROVIDER")
    if requested is not None:
        requested = requested.strip().lower()
    if not requested:
        if os.environ.get("OPENAI_API_KEY"):
            requested = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            requested = "anthropic"
        else:
            raise RuntimeError(
                "Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run the live projection contract."
            )
    if requested == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY to run the live projection contract.")
        return (
            "openai",
            os.environ.get("CAYU_OPENAI_MODEL", "gpt-5.6"),
            OpenAIProvider(),
        )
    if requested == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set ANTHROPIC_API_KEY to run the live projection contract.")
        return (
            "anthropic",
            os.environ.get("CAYU_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            AnthropicProvider(),
        )
    raise RuntimeError("CAYU_PROVIDER must be openai or anthropic.")


def _live_root(provider_name: str) -> Path:
    configured = os.environ.get("CAYU_PROJECTION_LIVE_ROOT")
    if configured:
        return Path(configured)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return (
        Path(__file__).resolve().parents[1]
        / ".examples-workspaces"
        / "tool-result-projection-live"
        / f"{provider_name}-{timestamp}-{os.getpid()}"
    )


async def main() -> None:
    try:
        provider_name, model, provider = _provider_config()
        evidence = await _run_contract(
            provider=provider,
            provider_name=provider_name,
            model=model,
            root=_live_root(provider_name),
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print("run_root", evidence["root"])
    print(
        EVIDENCE_PREFIX
        + json.dumps(
            {key: value for key, value in evidence.items() if key != "root"},
            sort_keys=True,
        )
    )
    print("status ok")


def _validate_runtime_events(
    events: list[Event],
    *,
    provider_name: str,
    model: str,
    large_report_calls: int,
) -> dict[str, Any]:
    require_successful_terminal(events)
    completed_model_events = require_positive_model_usage(events)
    require(large_report_calls == 1, f"large_report executed {large_report_calls} times")

    projected_events = [
        event
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED and event.tool_name == "large_report"
    ]
    require(
        len(projected_events) == 1,
        f"expected one completed large_report call, got {len(projected_events)}",
    )
    projected_event = projected_events[0]
    projection = _require_dict(
        projected_event.payload.get("tool_result_projection"),
        "large_report event is missing projection evidence",
    )
    require(projection.get("status") == "externalized", "large_report was not externalized")
    artifact_id = projection.get("artifact_id")
    require(isinstance(artifact_id, str), "projection evidence is missing artifact_id")
    original_bytes = projection.get("original_bytes")
    projected_bytes = projection.get("projected_bytes")
    require(
        isinstance(original_bytes, int) and original_bytes > 2_048,
        f"projection original_bytes is invalid: {original_bytes!r}",
    )
    require(
        isinstance(projected_bytes, int) and 0 < projected_bytes < original_bytes,
        f"projection projected_bytes is invalid: {projected_bytes!r}",
    )

    result = _require_dict(
        projected_event.payload.get("result"),
        "large_report event is missing its result",
    )
    structured = _require_dict(
        result.get("structured"),
        "large_report result is missing structured evidence",
    )
    require(
        structured.get("receipt_id") == RECEIPT_ID,
        "large_report structured receipt did not survive projection",
    )
    artifacts = _require_list(
        result.get("artifacts"),
        "large_report result is missing artifact references",
    )
    require(
        any(
            isinstance(item, dict)
            and item.get("type") == "cayu.tool_result_artifact.v1"
            and item.get("artifact_id") == artifact_id
            for item in artifacts
        ),
        "large_report result is missing its typed projection reference",
    )

    read_events = [
        event
        for event in events
        if event.type == EventType.TOOL_CALL_COMPLETED and event.tool_name == "read_file"
    ]
    require(bool(read_events), "the live model did not call read_file")
    read_result = _require_dict(
        read_events[-1].payload.get("result"),
        "read_file event is missing its result",
    )
    read_structured = _require_dict(
        read_result.get("structured"),
        "read_file result is missing structured evidence",
    )
    require(read_structured.get("source") == "artifact", "read_file did not use an artifact")
    require(
        read_structured.get("artifact_id") == artifact_id,
        "read_file inspected a different artifact",
    )
    require(read_structured.get("truncated") is True, "read_file result was not bounded")
    readback_bytes = read_structured.get("bytes")
    require(
        isinstance(readback_bytes, int) and 0 < readback_bytes <= 256,
        f"read_file byte count is invalid: {readback_bytes!r}",
    )

    total_tokens = sum(
        event.payload["usage_metrics"]["total_tokens"] for event in completed_model_events
    )
    return {
        "provider": provider_name,
        "model": model,
        "artifact_id": artifact_id,
        "original_bytes": original_bytes,
        "projected_bytes": projected_bytes,
        "readback_bytes": readback_bytes,
        "readback_truncated": True,
        "large_report_calls": large_report_calls,
        "total_tokens": total_tokens,
    }


if __name__ == "__main__":
    asyncio.run(main())
