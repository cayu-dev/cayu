from __future__ import annotations

import asyncio
import sys

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    CommandPolicy,
    CommandPolicyDecision,
    CommandRequest,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecCommand,
    ExecCommandTool,
    LocalRunner,
    Message,
    ModelStreamEvent,
    ProcessCommandPolicy,
    RunRequest,
    ScriptedModelProvider,
)
from cayu.core.tools import ToolContext
from cayu.tools import ProcessCommandPolicy as ToolsProcessCommandPolicy


def _request(
    *argv: str,
    canonical_cwd: str | None = "/workspace",
) -> CommandRequest:
    return CommandRequest(
        command=ExecCommand.process(*argv),
        cwd=None,
        canonical_cwd=canonical_cwd,
        timeout_s=60,
    )


def _evaluate(
    policy: ProcessCommandPolicy,
    request: CommandRequest,
):
    return asyncio.run(policy.evaluate(ToolContext(session_id="sess_1"), request))


class _RecordingPolicy(CommandPolicy):
    def __init__(self, delegate: ProcessCommandPolicy) -> None:
        self.delegate = delegate
        self.requests: list[CommandRequest] = []

    async def evaluate(
        self,
        ctx: ToolContext,
        request: CommandRequest,
    ):
        self.requests.append(request)
        return await self.delegate.evaluate(ctx, request)


def test_process_command_policy_is_exported_from_public_surfaces() -> None:
    assert ToolsProcessCommandPolicy is ProcessCommandPolicy


def test_process_command_policy_denies_everything_by_default() -> None:
    result = _evaluate(ProcessCommandPolicy(), _request("git", "status"))

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Executable is not allowed by the process policy."


def test_process_command_policy_allows_exact_executable_in_canonical_root() -> None:
    policy = ProcessCommandPolicy(
        allowed_executables={"git"},
        allowed_cwds={"/workspace"},
    )

    result = _evaluate(policy, _request("git", "status", canonical_cwd="/workspace/repo"))

    assert result.decision is CommandPolicyDecision.ALLOW
    assert result.reason is None


@pytest.mark.parametrize(
    "executable",
    ["./git", "/usr/bin/git", " git", "git ", "git-lookalike"],
)
def test_process_command_policy_matches_executable_identity_exactly(executable: str) -> None:
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
        ),
        _request(executable, "status"),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Executable is not allowed by the process policy."


@pytest.mark.parametrize(
    "canonical_cwd",
    [None, "workspace", "/workspace/../escape", "/workspace-lookalike"],
)
def test_process_command_policy_rejects_uncanonical_or_uncontained_cwd(
    canonical_cwd: str | None,
) -> None:
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
        ),
        _request("git", canonical_cwd=canonical_cwd),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Canonical cwd is not allowed by the process policy."


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"allowed_executables": "git"}, TypeError, "iterable of strings"),
        ({"allowed_executables": {" git"}}, ValueError, "must not start or end"),
        ({"allowed_cwds": "workspace"}, TypeError, "iterable of strings"),
        ({"allowed_cwds": {"workspace"}}, ValueError, "absolute POSIX"),
        ({"allowed_cwds": {"/workspace/../repo"}}, ValueError, "normalized POSIX"),
    ],
)
def test_process_command_policy_rejects_invalid_host_configuration(
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ProcessCommandPolicy(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        (
            {
                "allowed_executables": {"git"},
                "approval_required_executables": {"git"},
            },
            ValueError,
            "both allowed and approval-required",
        ),
        ({"allowed_env_names": "LANG"}, TypeError, "iterable of strings"),
        ({"allowed_env_names": {"BAD=NAME"}}, ValueError, "valid environment"),
        ({"allowed_env_values": []}, TypeError, "must be a mapping"),
        ({"allowed_env_values": {"TOKEN": 1}}, TypeError, "must be strings"),
        (
            {"allowed_env_values": {"TOKEN": "too-long"}, "max_env_value_bytes": 3},
            ValueError,
            "exceeds max_env_value_bytes",
        ),
        ({"max_env_value_bytes": True}, TypeError, "must be an integer"),
        ({"allow_stdin": 1}, TypeError, "must be a boolean"),
        ({"max_stdin_bytes": -1}, ValueError, "cannot be negative"),
        ({"max_timeout_s": 0}, ValueError, "greater than zero"),
        ({"max_timeout_s": 601}, ValueError, "cannot exceed 600"),
        ({"shell_decision": "allow"}, TypeError, "CommandPolicyDecision"),
    ],
)
def test_process_command_policy_validates_capability_configuration(
    kwargs: dict,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        ProcessCommandPolicy(**kwargs)


def test_process_command_policy_fails_closed_on_malformed_process_argv() -> None:
    malformed = ExecCommand.model_construct(kind="process", argv=[], shell=None)
    request = _request("git").model_copy(update={"command": malformed})

    result = _evaluate(
        ProcessCommandPolicy(allowed_executables={"git"}, allowed_cwds={"/workspace"}),
        request,
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Executable is not allowed by the process policy."


def test_process_command_policy_uses_only_canonical_cwd_for_authorization() -> None:
    request = _request("git", canonical_cwd="/workspace/repo").model_copy(
        update={"cwd": "../../escape"}
    )

    result = _evaluate(
        ProcessCommandPolicy(allowed_executables={"git"}, allowed_cwds={"/workspace"}),
        request,
    )

    assert result.decision is CommandPolicyDecision.ALLOW


def test_process_command_policy_denies_model_environment_by_default() -> None:
    secret = "secret-that-must-not-be-persisted"
    request = _request("git")
    request = request.model_copy(update={"env": {"AWS_SECRET_ACCESS_KEY": secret}})

    result = _evaluate(
        ProcessCommandPolicy(allowed_executables={"git"}, allowed_cwds={"/workspace"}),
        request,
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Environment name is not allowed by the process policy."
    assert secret not in result.reason


@pytest.mark.parametrize("env", [None, {}, {"LANG": "C.UTF-8"}])
def test_process_command_policy_allows_selected_environment_names(
    env: dict[str, str] | None,
) -> None:
    request = _request("git").model_copy(update={"env": env})
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            allowed_env_names={"LANG"},
        ),
        request,
    )

    assert result.decision is CommandPolicyDecision.ALLOW


def test_process_command_policy_enforces_exact_environment_values_without_disclosure() -> None:
    configured = "configured-secret"
    supplied = "different-secret"
    request = _request("git").model_copy(update={"env": {"TOKEN": supplied}})
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            allowed_env_values={"TOKEN": configured},
        ),
        request,
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Environment value does not satisfy the process policy."
    assert configured not in result.reason
    assert supplied not in result.reason


@pytest.mark.parametrize("env", [None, {}, {"GIT_TERMINAL_PROMPT": "0"}])
def test_process_command_policy_allows_absent_or_exact_constrained_environment_value(
    env: dict[str, str] | None,
) -> None:
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            allowed_env_values={"GIT_TERMINAL_PROMPT": "0"},
        ),
        _request("git").model_copy(update={"env": env}),
    )

    assert result.decision is CommandPolicyDecision.ALLOW


def test_process_command_policy_bounds_environment_value_bytes() -> None:
    request = _request("git").model_copy(update={"env": {"LANG": "éé"}})
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            allowed_env_names={"LANG"},
            max_env_value_bytes=3,
        ),
        request,
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Environment value exceeds the process policy byte limit."


def test_process_command_policy_denies_stdin_by_default_without_disclosure() -> None:
    secret = "stdin-secret"
    result = _evaluate(
        ProcessCommandPolicy(allowed_executables={"git"}, allowed_cwds={"/workspace"}),
        _request("git").model_copy(update={"stdin": secret}),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Stdin is not allowed by the process policy."
    assert secret not in result.reason


@pytest.mark.parametrize("stdin", ["", "é"])
def test_process_command_policy_allows_bounded_stdin_when_enabled(stdin: str) -> None:
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            allow_stdin=True,
            max_stdin_bytes=2,
        ),
        _request("git").model_copy(update={"stdin": stdin}),
    )

    assert result.decision is CommandPolicyDecision.ALLOW


def test_process_command_policy_rejects_stdin_over_byte_limit() -> None:
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            allow_stdin=True,
            max_stdin_bytes=1,
        ),
        _request("git").model_copy(update={"stdin": "é"}),
    )

    assert result.decision is CommandPolicyDecision.DENY
    assert result.reason == "Stdin exceeds the process policy byte limit."


@pytest.mark.parametrize(
    ("timeout_s", "decision"),
    [
        (1, CommandPolicyDecision.ALLOW),
        (10, CommandPolicyDecision.ALLOW),
        (11, CommandPolicyDecision.DENY),
    ],
)
def test_process_command_policy_enforces_host_timeout_ceiling(
    timeout_s: int,
    decision: CommandPolicyDecision,
) -> None:
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            max_timeout_s=10,
        ),
        _request("git").model_copy(update={"timeout_s": timeout_s}),
    )

    assert result.decision is decision
    if decision is CommandPolicyDecision.DENY:
        assert result.reason == "Timeout exceeds the process policy ceiling."


@pytest.mark.parametrize(
    ("shell_decision", "expected"),
    [
        (CommandPolicyDecision.DENY, CommandPolicyDecision.DENY),
        (CommandPolicyDecision.ALLOW, CommandPolicyDecision.ALLOW),
        (
            CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL,
            CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL,
        ),
    ],
)
def test_process_command_policy_treats_shell_as_a_separate_capability(
    shell_decision: CommandPolicyDecision,
    expected: CommandPolicyDecision,
) -> None:
    request = _request("git").model_copy(update={"command": ExecCommand.bash("git status")})
    result = _evaluate(
        ProcessCommandPolicy(
            allowed_executables={"git"},
            allowed_cwds={"/workspace"},
            shell_decision=shell_decision,
        ),
        request,
    )

    assert result.decision is expected
    if expected is not CommandPolicyDecision.ALLOW:
        assert result.reason == "Shell command requires an explicit process policy decision."


def test_process_command_policy_can_require_approval_for_an_executable() -> None:
    result = _evaluate(
        ProcessCommandPolicy(
            approval_required_executables={"python"},
            allowed_cwds={"/workspace"},
        ),
        _request("python", "script.py"),
    )

    assert result.decision is CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL
    assert result.reason == "Executable requires command approval by the process policy."


def test_process_command_policy_runs_allowed_and_blocks_denied_command_in_cayu_app(
    tmp_path,
) -> None:
    work = tmp_path / "repo"
    work.mkdir()
    denied_marker = tmp_path / "denied-command-ran"
    policy = _RecordingPolicy(
        ProcessCommandPolicy(
            allowed_executables={sys.executable},
            allowed_cwds={str(tmp_path)},
            max_timeout_s=30,
        )
    )
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="allowed_call",
                    name="exec_command",
                    arguments={
                        "argv": [
                            sys.executable,
                            "-c",
                            "import os; print(os.getcwd())",
                        ],
                        "cwd": "repo",
                        "timeout_s": 30,
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="denied_call",
                    name="exec_command",
                    arguments={
                        "argv": [
                            "/bin/sh",
                            "-c",
                            f"touch {denied_marker}",
                        ],
                        "cwd": "repo",
                    },
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp()
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="local-dev"), runner=LocalRunner(tmp_path)),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="assistant", model="scripted-model"),
        tools=[ExecCommandTool(policy=policy)],
    )

    events = asyncio.run(
        _collect_events(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "run the deterministic workflow")],
            ),
        )
    )

    assert events[-1].type is EventType.SESSION_COMPLETED
    assert [(request.cwd, request.canonical_cwd) for request in policy.requests] == [
        ("repo", str(work)),
        ("repo", str(work)),
    ]
    completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
    assert completed[0].payload["result"]["structured"]["stdout"].strip() == str(work)
    denied = next(event for event in events if event.type is EventType.TOOL_CALL_FAILED)
    assert denied.payload["result"]["structured"] == {
        "error": "command_denied",
        "decision": "deny",
        "reason": "Executable is not allowed by the process policy.",
    }
    assert not denied_marker.exists()


async def _collect_events(app: CayuApp, request: RunRequest):
    return [event async for event in app.run(request)]
