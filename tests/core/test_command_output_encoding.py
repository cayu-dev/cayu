from __future__ import annotations

import asyncio
import json
import sys
from uuid import uuid4

import pytest
from pydantic import ValidationError
from tests.core.test_builtin_tools import FakeProvider
from tests.runners.test_docker_live import _docker_path_or_skip

from cayu import (
    AgentSpec,
    CayuApp,
    DockerRunner,
    Environment,
    EnvironmentSpec,
    ExecCommandTool,
    ExecResult,
    ExecutionProfileBehaviorIdentity,
    LocalRunner,
    Message,
    ResumeRequest,
    RunRequest,
    SecretRedactor,
    ToolContext,
    ToolResult,
)
from cayu.core import EventType
from cayu.providers import ModelStreamEvent
from cayu.storage.sqlite import SQLiteSessionStore
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle


class EncodingProvider(FakeProvider):
    @property
    def execution_profile_identity(self):
        return ExecutionProfileBehaviorIdentity(
            name="test-command-encoding-provider", behavior_version="1", implementation_version="1"
        )


def decoded(result, stream):
    value = result.structured[stream]
    return json.loads(value) if result.structured[f"{stream}_encoding"] == "json-string" else value


@pytest.mark.parametrize(
    "stdout,stderr", [(b"safe", b""), (b"a\0b", b""), (b"", b"a\0b"), (b"a\0b", b"c\0d")]
)
@pytest.mark.parametrize("exit_code", [0, 3])
def test_local_command_output_round_trip(tmp_path, stdout, stderr, exit_code):
    script = f"import os; os.write(1, {stdout!r}); os.write(2, {stderr!r}); raise SystemExit({exit_code})"
    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="encoding", runner=LocalRunner(tmp_path)),
            {"argv": [sys.executable, "-c", script]},
        )
    )
    assert decoded(result, "stdout") == stdout.decode()
    assert decoded(result, "stderr") == stderr.decode()
    assert result.structured["exit_code"] == exit_code
    assert result.structured["stdout_bytes"] == len(stdout)
    assert result.structured["stderr_bytes"] == len(stderr)
    assert result.is_error is False
    assert "\0" not in result.content
    assert ToolResult.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5, 8, 30])
@pytest.mark.parametrize("raw", [b"\0" * 10, "é\0\\u0000🙂".encode(), b"\xff\0ok"])
def test_output_capture_bound_precedes_encoding(tmp_path, limit, raw):
    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="encoding", runner=LocalRunner(tmp_path)),
            {
                "argv": [sys.executable, "-c", f"import os; os.write(1, {raw!r})"],
                "max_output_bytes": limit,
            },
        )
    )
    normalized = raw.decode("utf-8", "replace").encode()
    expected = normalized[:limit].decode("utf-8", "ignore")
    assert decoded(result, "stdout") == expected
    assert result.structured["stdout_bytes"] == len(raw)
    assert result.structured["stdout_truncated"] == (len(normalized) > limit)
    assert len(result.structured["stdout"].encode()) <= 6 * limit + 2


@pytest.mark.parametrize("timed_out,cancelled", [(True, False), (False, True)])
def test_observed_interruption_survives_encoding(tmp_path, timed_out, cancelled):
    class ObservedRunner(LocalRunner):
        async def exec(self, command, **kwargs):
            return ExecResult(
                stdout="\0",
                stderr="error\0",
                exit_code=-9,
                timed_out=timed_out,
                cancelled=cancelled,
            )

    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="encoding", runner=ObservedRunner(tmp_path)),
            {"argv": ["unused"]},
        )
    )
    assert result.is_error
    assert result.structured["timed_out"] == timed_out
    assert result.structured["cancelled"] == cancelled
    assert result.structured["exit_code"] == -9
    assert decoded(result, "stdout") == "\0"


def test_redaction_before_json_escaping(tmp_path):
    secret = 'token-"-\\-é'
    redactor = SecretRedactor(secret)
    handle = InvocationRunnerHandle(
        LocalRunner(tmp_path),
        redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
            revision=0, redactor=redactor
        ),
    )
    result = asyncio.run(
        ExecCommandTool().run(
            ToolContext(session_id="encoding", runner=handle),
            {
                "argv": [
                    sys.executable,
                    "-c",
                    "import os; os.write(1, (os.environ['TOKEN'] + chr(0)).encode())",
                ],
                "env": {"TOKEN": secret},
            },
        )
    )
    assert decoded(result, "stdout") == "[REDACTED_SECRET]\0"
    assert secret not in result.model_dump_json()
    assert result.artifacts == []


def test_public_durable_text_still_rejects_nul():
    with pytest.raises(ValidationError):
        ToolResult(content="\0")
    with pytest.raises(ValidationError):
        ToolResult(content="safe", structured={"stdout": "\0"})


@pytest.mark.parametrize("max_steps", [1, 20])
def test_native_dispatch_persists_encoded_outcome_in_sqlite(tmp_path, max_steps):
    async def scenario():
        path = tmp_path / "sessions.db"
        store = SQLiteSessionStore(path)
        provider = EncodingProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="nul-call",
                        name="exec_command",
                        arguments={
                            "argv": [
                                sys.executable,
                                "-c",
                                "from pathlib import Path; import os; p=Path('count'); p.write_text(p.read_text()+'x' if p.exists() else 'x'); os.write(1,b'a\\0b'); os.write(2,b'c\\0d')",
                            ]
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [ModelStreamEvent.completed({"finish_reason": "stop"})],
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        environment = Environment(
            EnvironmentSpec(
                name="local",
                execution_profile_identity=ExecutionProfileBehaviorIdentity(
                    name="test-command-encoding-environment",
                    behavior_version="1",
                    implementation_version="1",
                ),
            ),
            runner=LocalRunner(tmp_path),
        )
        app.register_environment(environment, default=True)
        app.register_agent(AgentSpec(name="assistant", model="fake"), tools=[ExecCommandTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="nul-native",
                    agent_name="assistant",
                    max_steps=max_steps,
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert events[-1].type == (
            EventType.SESSION_INTERRUPTED if max_steps == 1 else EventType.SESSION_COMPLETED
        )
        terminal = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
        assert len(terminal) == 1
        result = ToolResult.model_validate(terminal[0].payload["result"])
        assert decoded(result, "stdout") == "a\0b"
        assert decoded(result, "stderr") == "c\0d"
        assert result.structured["exit_code"] == 0
        await store.close()
        reopened = SQLiteSessionStore(path)
        restored = await reopened.load_events("nul-native")
        recovered = [e for e in restored if e.type == EventType.TOOL_CALL_COMPLETED]
        assert recovered[0].payload["result"] == terminal[0].payload["result"]
        assert await reopened.load("nul-native") is not None
        assert await reopened.load_transcript("nul-native")
        provider.event_batches.append([ModelStreamEvent.completed({"finish_reason": "stop"})])
        restarted = CayuApp(session_store=reopened, enable_logging=False)
        restarted.register_provider(provider, default=True)
        restarted.register_environment(environment, default=True)
        restarted.register_agent(
            AgentSpec(name="assistant", model="fake"), tools=[ExecCommandTool()]
        )
        resumed = [
            e
            async for e in restarted.resume(
                ResumeRequest(
                    session_id="nul-native",
                    max_steps=max_steps,
                    messages=[Message.text("user", "continue")],
                )
            )
        ]
        assert resumed[-1].type == EventType.SESSION_COMPLETED
        assert not any(e.type == EventType.TOOL_CALL_STARTED for e in resumed)
        assert "json-string" in repr(provider.requests[-1].messages)
        assert (tmp_path / "count").read_text() == "x"
        await reopened.close()

    asyncio.run(scenario())


@pytest.mark.process
def test_docker_command_nul_output():
    docker_path = _docker_path_or_skip()

    async def scenario():
        async with await DockerRunner.create(
            f"cayu-nul-{uuid4().hex[:12]}",
            image="alpine:3.20",
            docker_path=docker_path,
            close_action="remove",
        ) as runner:
            result = await ExecCommandTool().run(
                ToolContext(session_id="encoding", runner=runner),
                {"argv": ["sh", "-c", "printf 'a\\000b'; printf 'c\\000d' >&2"]},
            )
            assert decoded(result, "stdout") == "a\0b"
            assert decoded(result, "stderr") == "c\0d"
            assert result.structured["exit_code"] == 0
            assert not result.is_error

    asyncio.run(scenario())


@pytest.mark.parametrize("tool_kind", ["check", "command"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_downstream_output_encoding_distinguishes_literal_json(tmp_path, tool_kind, stream):
    outputs = []
    for value in ["a\0b", '"a\\u0000b"']:
        result, _ = _run_downstream_output(tmp_path, tool_kind, {stream: value})
        outputs.append(result)
        assert not result.is_error
        assert result.structured["exit_code"] == 0
        assert decoded(result, stream) == value
    encoded, literal = outputs
    assert encoded.structured[stream] == literal.structured[stream]
    assert encoded.structured[f"{stream}_encoding"] == "json-string"
    assert literal.structured[f"{stream}_encoding"] == "text"
    assert f"{stream} (JSON string):" in encoded.content
    assert "JSON string" not in literal.content
    assert encoded.structured["output_sha256"] != literal.structured["output_sha256"]


@pytest.mark.parametrize("tool_kind", ["check", "command"])
def test_downstream_encoded_preview_retains_decodable_artifact(tmp_path, tool_kind):
    from hashlib import sha256

    raw = "é\0\\u0000" * 100
    result, store = _run_downstream_output(
        tmp_path, tool_kind, {"stdout": raw, "stderr": raw}, preview_limit=256
    )
    assert not result.is_error
    assert result.structured["output_artifact_status"] == "stored"
    stored = asyncio.run(store.read_bytes(result.artifacts[-1]["id"]))
    record = json.loads(stored.content)
    assert result.structured["output_sha256"] == "sha256:" + sha256(stored.content).hexdigest()
    for stream in ("stdout", "stderr"):
        assert record[f"{stream}_encoding"] == "json-string"
        assert json.loads(record[stream]) == raw
        assert result.structured[f"{stream}_encoding"] == "json-string-preview"
        assert result.structured[f"{stream}_projection_truncated"]
        assert len(result.structured[stream].encode()) <= 256
        assert f"{stream} (JSON string preview; not a complete JSON literal):" in result.content


def test_publication_ceiling_keeps_encoded_preview_metadata(tmp_path):
    raw = "\0" * 2_000
    result, store = _run_downstream_output(tmp_path, "command", {"stdout": raw, "stderr": raw})
    assert result.structured["result_publication_ceiling_applied"]
    from tests.core.test_structured_commands import _profile

    assert len(result.model_dump_json().encode()) <= _profile().result_publication_max_bytes
    for stream in ("stdout", "stderr"):
        assert result.structured[f"{stream}_encoding"] == "json-string-preview"
        assert "not a complete JSON literal" in result.content
    stored = asyncio.run(store.read_bytes(result.artifacts[-1]["id"]))
    assert json.loads(json.loads(stored.content)["stdout"]) == raw


def _run_downstream_output(tmp_path, tool_kind, streams, preview_limit=16_384):
    from tests.core.test_named_checks import RecordingRunner, _check, _policy
    from tests.core.test_structured_commands import _AdmittedRunner, _profile

    from cayu import LocalArtifactStore, LocalWorkspace, RunCheckTool, RunCommandTool

    store = LocalArtifactStore(tmp_path / "artifacts", store_id="encoding")
    if tool_kind == "check":
        tool = RunCheckTool(
            checks=[_check()], command_policy=_policy(), max_model_output_bytes=preview_limit
        )
        ctx = ToolContext(
            session_id="encoded-check",
            runner=RecordingRunner(ExecResult(**streams)),
            artifact_store=store,
        )
        args = {"check": "test"}
    else:
        workspace_path = tmp_path / "workspace"
        workspace_path.mkdir(exist_ok=True)
        (workspace_path / "uv.lock").write_bytes(b"locked\n")
        workspace = LocalWorkspace(workspace_path, workspace_id="workspace")
        profile = _profile()
        authority = profile.structured_command_authorities[0].model_copy(
            update={"max_model_output_bytes": preview_limit}
        )
        profile = profile.model_copy(update={"command_authorities": (authority,)})
        tool = RunCommandTool(toolchain_profile=profile)
        ctx = ToolContext(
            session_id="encoded-command",
            agent_name="agent",
            environment_name="coding",
            workspace_id=workspace.id,
            workspace=workspace,
            runner=_AdmittedRunner(profile, result=ExecResult(**streams)),
            artifact_store=store,
        )
        args = {"selector": "focused-test", "args": ["tests/test_unit.py"]}
    return asyncio.run(tool.run(ctx, args)), store
