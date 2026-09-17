from __future__ import annotations

import asyncio
import json
import sys

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventQuery,
    EventType,
    ExecCommandTool,
    ExecutionProfileBehaviorIdentity,
    LocalRunner,
    Message,
    ModelStreamEvent,
    ProcessCommandPolicy,
    ResumeRequest,
    RunRequest,
    SQLiteSessionStore,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.storage.jsonl_export import import_sessions


def _identity(name):
    return ExecutionProfileBehaviorIdentity(
        name=name, behavior_version="1", implementation_version="1"
    )


class _RestartableScriptedProvider(ScriptedModelProvider):
    @property
    def execution_profile_identity(self):
        return _identity("process-diagnostics-provider")


def test_process_denials_survive_tool_round_restart_inspection_export(tmp_path, sqlite_resources):
    async def scenario():
        async with sqlite_resources as resources:
            path = resources.path()
            store = resources.own(SQLiteSessionStore(path))
            admitted = tmp_path / "admitted"
            admitted.mkdir()
            private = tmp_path / "SECRET-private-cwd"
            private.mkdir()
            marker = admitted / "must-not-execute"
            published_commands = {f"command_{i:02}_" + "界" * 100 for i in range(63)}
            policy = ProcessCommandPolicy(
                allowed_executables={sys.executable, *published_commands},
                allowed_cwds={str(admitted)},
                allowed_env_names={"CI", "SECRET_CONFIGURED_NAME"},
                public_environment_names={"CI", "PYTHONPATH"},
                public_executable_names={"denied-public-command", *published_commands},
                diagnostic_profile_id="test-process-v1",
                max_env_value_bytes=8,
                max_timeout_s=30,
            )
            dangerous_argv = [
                sys.executable,
                "-c",
                f"open({str(marker)!r}, 'w').close() # SECRET-argv",
            ]
            cases = [
                ("executable", {"argv": ["denied-public-command", "SECRET-argv"]}),
                ("executable", {"argv": ["SECRET-private-executable", "SECRET-argv"]}),
                ("environment_name", {"env": {"PYTHONPATH": "SECRET-value"}}),
                ("environment_name", {"env": {"SECRET_REQUEST_NAME": "SECRET-value"}}),
                ("working_directory", {"cwd": str(private)}),
                ("stdin", {"stdin": "SECRET-stdin"}),
                ("timeout", {"timeout_s": 31}),
                ("environment_value_size", {"env": {"CI": "SECRET-too-long"}}),
                ("shell", {"argv": None, "shell": "echo SECRET-shell"}),
            ]
            scripts = [
                [
                    ModelStreamEvent.tool_call(
                        id=f"denied-{index}",
                        name="exec_command",
                        arguments={
                            "argv": dangerous_argv,
                            "cwd": str(admitted),
                            "timeout_s": 30,
                            **overrides,
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
                for index, (_, overrides) in enumerate(cases)
            ]
            # Both successful and nonzero process exits are completed executions.
            for code in (0, 7):
                scripts.append(
                    [
                        ModelStreamEvent.tool_call(
                            id=f"exit-{code}",
                            name="exec_command",
                            arguments={
                                "argv": [sys.executable, "-c", f"raise SystemExit({code})"],
                                "cwd": str(admitted),
                                "timeout_s": 30,
                            },
                        ),
                        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                    ]
                )
            scripts.append(
                [
                    ModelStreamEvent.text_delta("Done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]
            )

            def app_for(session_store, provider):
                app = CayuApp(session_store=session_store, enable_logging=False)
                app.register_provider(provider, default=True)
                app.register_environment(
                    Environment(
                        EnvironmentSpec(
                            name="local",
                            execution_profile_identity=_identity("local-test-environment"),
                        ),
                        runner=LocalRunner(tmp_path),
                    ),
                    default=True,
                )
                app.register_agent(
                    AgentSpec(name="agent", model="scripted-model"),
                    tools=[ExecCommandTool(policy=policy)],
                )
                return app

            app = app_for(store, _RestartableScriptedProvider(scripts))
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        agent_name="agent",
                        session_id="diagnostics",
                        messages=[Message.text("user", "test")],
                    )
                )
            ]
            assert events[-1].type is EventType.SESSION_COMPLETED
            blocked = [event for event in events if event.type is EventType.TOOL_CALL_BLOCKED]
            assert len(blocked) == len(cases)
            for event, (code, _) in zip(blocked, cases, strict=True):
                assert event.payload["arguments_state"] == "unavailable"
                assert event.payload["arguments_exact"] is False
                result = event.payload["result"]["structured"]
                diagnostic = result["process_diagnostic"]
                assert diagnostic["code"] == code
                assert diagnostic["capabilities"]["profile_id"] == "test-process-v1"
                assert diagnostic["capabilities"]["max_timeout_s"] == 30
                assert "Do not replay" in result["recovery_instruction"]
                # OpenAI and other adapters may send only ToolResultPart.content.
                content = event.payload["result"]["content"]
                assert f'"code":"{code}"' in content
                assert '"max_timeout_s":30' in content
                text_diagnostic = json.loads(content.split("\n", 1)[1])
                assert text_diagnostic["capability_names_truncated"] is True
                assert text_diagnostic["capabilities"]["max_env_value_bytes"] == 8
                assert len(content.encode("utf-8")) <= 4096
                assert len(diagnostic["capabilities"]["allowed_executables"]) == 63
                assert "Do not replay" in content
                assert "SECRET" not in event.model_dump_json()
            assert (
                blocked[0].payload["result"]["structured"]["process_diagnostic"]["rejected_name"]
                == "denied-public-command"
            )
            assert (
                blocked[1].payload["result"]["structured"]["process_diagnostic"]["value_state"]
                == "withheld_not_declared_public"
            )
            assert (
                blocked[2].payload["result"]["structured"]["process_diagnostic"]["rejected_name"]
                == "PYTHONPATH"
            )
            completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
            assert [event.payload["result"]["structured"]["exit_code"] for event in completed] == [
                0,
                7,
            ]
            assert not marker.exists()
            await store.close()

            reopened = resources.own(SQLiteSessionStore(path))
            loaded = await reopened.load_events("diagnostics")
            assert [
                e.payload["result"] for e in loaded if e.type is EventType.TOOL_CALL_BLOCKED
            ] == [e.payload["result"] for e in blocked]
            inspection = await reopened.query_events(
                EventQuery(session_id="diagnostics", event_type=EventType.TOOL_CALL_BLOCKED)
            )
            inspected = json.dumps([record.model_dump(mode="json") for record in inspection])
            assert "process_diagnostic" in inspected
            assert "SECRET" not in inspected
            snapshot = await reopened.load_session_export_snapshot("diagnostics")
            assert snapshot is not None
            document = json.dumps(snapshot.document())
            assert "SECRET" not in document
            [restored] = import_sessions([document])
            exported = [e for e in restored.events if e.type is EventType.TOOL_CALL_BLOCKED]
            assert [e.payload["result"] for e in exported] == [e.payload["result"] for e in blocked]
            seen = []

            def continuation(request):
                seen.append(request.model_dump_json())
                return [
                    ModelStreamEvent.text_delta("Continued"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ]

            restarted = app_for(
                reopened, _RestartableScriptedProvider(response_factory=continuation)
            )
            continued = [
                event
                async for event in restarted.resume(
                    ResumeRequest(
                        session_id="diagnostics",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
            assert continued[-1].type is EventType.SESSION_COMPLETED
            assert seen and "process_diagnostic" in seen[0]
            assert "SECRET" not in seen[0]
            assert not marker.exists()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("configuration", "override", "code"),
    [
        ({"allow_stdin": True, "max_stdin_bytes": 2}, {"stdin": "SECRET-stdin"}, "stdin_size"),
        (
            {"allowed_env_values": {"TOKEN": "SECRET-configured"}},
            {"env": {"TOKEN": "SECRET-other"}},
            "environment_value",
        ),
    ],
)
def test_process_value_diagnostics_never_publish_values(configuration, override, code):
    from cayu import CommandRequest, ExecCommand
    from cayu.tools.base import ToolContext

    policy = ProcessCommandPolicy(
        allowed_executables={"python"}, allowed_cwds={"/workspace"}, **configuration
    )
    request = CommandRequest(
        command=ExecCommand.process("python"), canonical_cwd="/workspace", timeout_s=30, **override
    )
    result = asyncio.run(policy.evaluate(ToolContext(session_id="test"), request))
    assert result.process_diagnostic.code.value == code
    assert "SECRET" not in result.model_dump_json()


def test_process_discovery_requires_explicit_publication():
    policy = ProcessCommandPolicy(
        allowed_executables={"SECRET-private", "python"},
        allowed_cwds={"/SECRET-path"},
        allowed_env_names={"SECRET_NAME", "CI"},
        public_executable_names={"python"},
        public_environment_names={"CI"},
    )
    schema = ExecCommandTool(policy=policy).spec.input_schema
    assert "SECRET" not in json.dumps(schema)
    assert '"python"' in schema["description"]
    assert policy.process_capabilities.executable_names_withheld
    assert policy.process_capabilities.environment_names_withheld
    assert policy.process_capabilities.profile_id is None


@pytest.mark.parametrize(
    "configuration",
    [
        {"public_executable_names": "python"},
        {"public_executable_names": {"x" * 129}},
        {"public_executable_names": {"private\ntext"}},
        {"public_environment_names": {"BAD=NAME"}},
        {"public_environment_names": {f"NAME_{i}" for i in range(65)}},
        {"diagnostic_profile_id": ""},
        {"diagnostic_profile_id": "x" * 129},
    ],
)
def test_process_publication_configuration_is_bounded(configuration):
    with pytest.raises((ValueError, TypeError)):
        ProcessCommandPolicy(**configuration)


def test_publication_does_not_grant_execution_and_types_are_public():
    from cayu import (
        CommandPolicyDecision,
        CommandRequest,
        ExecCommand,
        ProcessCommandCapabilities,
        ProcessCommandDenialCode,
        ProcessCommandDiagnostic,
    )
    from cayu.tools import ProcessCommandDiagnostic as ToolsDiagnostic
    from cayu.tools.base import ToolContext

    policy = ProcessCommandPolicy(public_executable_names={"python"})
    result = asyncio.run(
        policy.evaluate(
            ToolContext(session_id="test"),
            CommandRequest(
                command=ExecCommand.process("python"),
                canonical_cwd="/workspace",
                timeout_s=30,
            ),
        )
    )
    assert result.decision is CommandPolicyDecision.DENY
    assert type(result.process_diagnostic) is ProcessCommandDiagnostic is ToolsDiagnostic
    assert result.process_diagnostic.code is ProcessCommandDenialCode.EXECUTABLE
    assert type(result.process_diagnostic.capabilities) is ProcessCommandCapabilities
    assert result.process_diagnostic.rejected_name == "python"
    assert result.process_diagnostic.capabilities.allowed_executables == ()
