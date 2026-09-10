"""Submitted patch evidence survives real execution and durable reconstruction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from cayu import (
    AgentSpec,
    ApplyPatchTool,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
)
from cayu.evals.corpus import EvaluationEvidencePolicySpec
from cayu.evals.evidence import project_assertion_evidence_view
from cayu.evals.models import Trajectory
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.runtime.usage import session_usage_summary
from cayu.vaults import REDACTED_SECRET, SecretRedactor, SecretRef, StaticVault


@pytest.mark.parametrize(
    "case",
    [
        "stale",
        "count",
        "malformed",
        "freeform",
        "success",
        "multi",
        "secret",
        "secret_stale",
        "large",
        "read_failure",
        "cancel",
        "secret_cancel",
    ],
)
def test_patch_input_evidence_survives_sqlite_reopen(tmp_path: Path, case: str) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    original = b"old\nold\n"
    (root / "source.txt").write_bytes(original)
    revision = "sha256:" + hashlib.sha256(original).hexdigest()
    operation = {
        "type": "update",
        "path": "source.txt",
        "expected_revision": "sha256:" + "0" * 64
        if case in {"stale", "secret_stale"}
        else revision,
        "edits": [{"old_text": "old", "new_text": "new", "expected_replacements": 2}],
    }
    arguments = {"operations": [operation]}
    if case == "count":
        operation["edits"][0]["expected_replacements"] = 1
    elif case == "malformed":
        arguments = {"operations": []}
    elif case == "freeform":
        arguments = {"input": "*** Begin Patch\ninvalid\n*** End Patch"}
    elif case == "multi":
        arguments["operations"].append({"type": "create", "path": "extra.txt", "content": "extra"})
    elif case in {"secret", "secret_stale", "secret_cancel", "large"}:
        operation["edits"][0]["new_text"] = (
            "diagnostic-secret-value" if case.startswith("secret") else "z" * 100_000
        )
    redactor = SecretRedactor("diagnostic-secret-value")
    expected = redactor.redact_json(arguments)

    class InterruptedWorkspace(LocalWorkspace):
        async def read_bytes(self, path, *, offset=0, max_bytes=None):
            if case == "read_failure":
                raise OSError("synthetic read failure")
            if case in {"cancel", "secret_cancel"}:
                task = asyncio.current_task()
                assert task is not None
                task.cancel()
                await asyncio.sleep(0)
            return await super().read_bytes(path, offset=offset, max_bytes=max_bytes)

    class EvidencePatchTool(ApplyPatchTool):
        async def run(self, ctx, args):
            if case.startswith("secret"):
                await ctx.vault.resolve(SecretRef(name="patch-secret"))
            return await super().run(ctx, args)

    async def run() -> None:
        database = tmp_path / "sessions.sqlite"
        codec = PublicAuthorityAliasCodec(
            PublicAuthorityAliasKeyring(
                active_key_id="test",
                keys={"test": SecretStr("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")},
            )
        )
        store = SQLiteSessionStore(database, public_authority_alias_codec=codec)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    [
                        ModelStreamEvent.tool_call(
                            id="patch-call", name="apply_patch", arguments=arguments
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ],
                    [
                        ModelStreamEvent.text_delta("done"),
                        ModelStreamEvent.completed({"finish_reason": "stop"}),
                    ],
                ]
            ),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(name="local"),
                workspace=InterruptedWorkspace(root),
                vault=StaticVault({"patch-secret": "diagnostic-secret-value"}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="assistant", model="fake"),
            tools=[EvidencePatchTool() if case.startswith("secret") else ApplyPatchTool()],
        )

        async def invoke():
            return [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="assistant",
                        session_id="patch-evidence",
                        messages=[Message.text("user", "edit")],
                    )
                )
            ]

        try:
            invocation = asyncio.create_task(invoke())
            if case in {"cancel", "secret_cancel"}:
                with pytest.raises(asyncio.CancelledError):
                    await invocation
            else:
                live = await invocation
                assert live[-1].type is (
                    EventType.SESSION_INTERRUPTED
                    if case == "read_failure"
                    else EventType.SESSION_COMPLETED
                )
        finally:
            await store.close()

        reopened = SQLiteSessionStore(database, public_authority_alias_codec=codec)
        try:
            events = await reopened.load_events("patch-evidence")
            transcript = await reopened.load_transcript("patch-evidence")
            if case in {"read_failure", "cancel", "secret_cancel"}:
                # The runtime has dispatched the external tool before its
                # preflight read. An exception is not a settled tool result;
                # retain the recovery fence and unpublished argument scope.
                assert not any(
                    event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_COMPLETED}
                    for event in events
                )
                unknown = [
                    event for event in events if event.type is EventType.TOOL_EFFECT_OUTCOME_UNKNOWN
                ]
                assert len(unknown) == 1
                assert unknown[0].payload["tool_call_id"] == "patch-call"
                started = [event for event in events if event.type is EventType.TOOL_CALL_STARTED]
                assert len(started) == 1
                assert started[0].payload["arguments_state"] == "quarantined"
                assert "arguments" not in started[0].payload
                checkpoint = await reopened.load_checkpoint("patch-evidence")
                assert checkpoint is not None and checkpoint.get("pending_tool_round") is not None
                assert not any(
                    part.type in {"tool_call", "tool_result"}
                    for message in transcript
                    for part in message.content
                )
                assert (root / "source.txt").read_bytes() == original
                public = json.dumps([event.model_dump(mode="json") for event in events])
                assert "diagnostic-secret-value" not in public
                return
            terminal = next(
                event
                for event in events
                if event.type in {EventType.TOOL_CALL_FAILED, EventType.TOOL_CALL_COMPLETED}
            )
            assert terminal.payload["tool_call_id"] == "patch-call"
            assert terminal.payload["arguments_state"] == "finalized"
            assert terminal.payload["arguments_exact"] is (not case.startswith("secret"))
            assert terminal.payload["arguments"] == expected
            call = next(
                part
                for message in transcript
                for part in message.content
                if part.type == "tool_call"
            )
            assert call.arguments == expected
            assert call.tool_call_id == terminal.payload["tool_call_id"]
            assert call.tool_round_id == terminal.payload["tool_round_id"]
            result = terminal.payload["result"]["structured"]
            if case in {"stale", "secret_stale", "count"}:
                assert result["category"] == (
                    "stale_source_revision"
                    if case in {"stale", "secret_stale"}
                    else "replacement_count_mismatch"
                )
                assert result["mutated"] is False
                assert (root / "source.txt").read_bytes() == original
            elif case in {"malformed", "freeform"}:
                assert terminal.type is EventType.TOOL_CALL_FAILED
                assert (root / "source.txt").read_bytes() == original
            else:
                assert result["outcome"] == "applied"
            trajectory = Trajectory(
                session=await reopened.load("patch-evidence"),
                events=tuple(events),
                transcript=tuple(transcript),
                usage_summary=session_usage_summary("patch-evidence", events),
                final_output="done",
            )
            exported = Trajectory.model_validate_json(trajectory.model_dump_json())
            exported_terminal = next(event for event in exported.events if event.id == terminal.id)
            assert exported_terminal.payload["arguments"] == expected
            evidence = project_assertion_evidence_view(
                CayuApp(enable_logging=False),
                exported,
                evidence_policy=EvaluationEvidencePolicySpec.standard(),
            )
            assert evidence.tool_calls[0].arguments.state == (
                "truncated" if case == "large" else "available"
            )
            if case != "large":
                assert evidence.tool_calls[0].arguments.value == expected
            if case.startswith("secret"):
                public = json.dumps([event.model_dump(mode="json") for event in events])
                assert "diagnostic-secret-value" not in public
                assert "diagnostic-secret-value" not in exported.model_dump_json()
                assert REDACTED_SECRET in public
            with pytest.raises(KeyError, match="Session not found"):
                await reopened.load_events("unrelated-session")
        finally:
            await reopened.close()

    asyncio.run(run())

    if case in {"stale", "secret", "large"}:
        # A new interpreter has neither the invocation tracker nor app redactor.
        readback = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import asyncio, json, sys
from cayu import SQLiteSessionStore
from pydantic import SecretStr
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
async def read():
    codec = PublicAuthorityAliasCodec(PublicAuthorityAliasKeyring(
        active_key_id="test", keys={"test": SecretStr("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")}
    ))
    store = SQLiteSessionStore(sys.argv[1], public_authority_alias_codec=codec)
    try:
        transcript = await store.load_transcript("patch-evidence")
        call = next(p for m in transcript for p in m.content if p.type == "tool_call")
        print(json.dumps(call.arguments))
    finally:
        await store.close()
asyncio.run(read())
""",
                str(tmp_path / "sessions.sqlite"),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert json.loads(readback.stdout) == expected
