from __future__ import annotations

import asyncio
import io
import shutil
import sys

import pytest

from cayu.agents import AgentSpec
from cayu.applications import CayuApp
from cayu.environments import LocalWorkspaceEnvironmentFactory
from cayu.environments.admission import (
    ExecutionCapabilityClaim,
    ExecutionRequirements,
    ExecutionToolRequirement,
    evaluate_execution_admission,
)
from cayu.environments.base import Environment, EnvironmentSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.events import EventType
from cayu.messages import Message
from cayu.providers.base import ModelStreamEvent
from cayu.runners import (
    ExecCommand,
    ExecResult,
    LaunchedRunner,
    LocalRunner,
    Runner,
    RunnerBinaryStreamCapability,
    RunnerInteractiveProcess,
    RunnerInteractiveProcessCapability,
)
from cayu.runners.launched import LAUNCHER_UNVERIFIED_REASON
from cayu.sessions.requests import RunRequest
from cayu.tools import ExecCommandTool, SearchTextTool, StaticToolPolicy
from cayu.tools.base import ToolExecutableRequirement, ToolExecutionRequirement
from cayu.vaults.redaction import SecretRedactor
from cayu.workspaces import LocalWorkspace

PREFIX = ("launch", "--as", "sandbox", "--")


class _Process(RunnerInteractiveProcess):
    async def write(self, data: bytes) -> None:
        del data

    async def close_stdin(self) -> None:
        return None

    async def read(self) -> bytes:
        return b""

    async def wait(self) -> int:
        return 0

    async def close(self) -> None:
        return None


class _Interactive(RunnerInteractiveProcessCapability):
    def __init__(self, calls: list[tuple[str, list[str]]]) -> None:
        self.calls = calls

    async def open_interactive_process(self, command, *, cwd=None, env=None, timeout_s=None):
        self.calls.append(("interactive", list(command.argv)))
        return _Process()


class RecordingRunner(Runner, RunnerBinaryStreamCapability):
    """Records the command every entry point receives."""

    isolation = "local"

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []
        self.interactive = _Interactive(self.calls)

    def _record(self, path: str, command: ExecCommand) -> ExecResult:
        assert command.kind == "process"
        self.calls.append((path, list(command.argv)))
        return ExecResult(stdout="ok")

    def preflight_exec(self, command, **kwargs):
        self.calls.append(("preflight", list(command.argv)))

    async def exec(self, command, **kwargs):
        return self._record("exec", command)

    async def exec_redacted(self, command, *, redactor, **kwargs):
        return self._record("exec_redacted", command)

    async def exec_system(self, command, **kwargs):
        self.calls.append(("exec_system", list(command.argv)))
        return ExecResult()

    async def exec_stream(self, command, **kwargs):
        return self._record("exec_stream", command)

    def interactive_process_capability(self):
        return self.interactive


def test_every_command_path_goes_through_the_launcher() -> None:
    inner = RecordingRunner()
    runner = LaunchedRunner(inner, PREFIX)
    command = ExecCommand.process("rg", "needle")

    async def exercise() -> None:
        runner.preflight_exec(command)
        await runner.exec(command)
        await runner.exec_redacted(command, redactor=SecretRedactor())
        capability = runner.binary_stream_capability()
        assert capability is not None
        await capability.exec_stream(command, stdout=io.BytesIO())
        interactive = runner.interactive_process_capability()
        assert interactive is not None
        await interactive.open_interactive_process(command)

    asyncio.run(exercise())
    launched = [*PREFIX, "rg", "needle"]
    assert inner.calls == [
        ("preflight", launched),
        ("exec", launched),
        ("exec_redacted", launched),
        ("exec_stream", launched),
        ("interactive", launched),
    ]


def test_system_commands_use_the_wrapped_runner_without_the_launcher() -> None:
    inner = RecordingRunner()
    runner = LaunchedRunner(inner, PREFIX)

    asyncio.run(runner.exec_system(ExecCommand.process("git", "status")))

    assert runner.system_execution_mode == "separate"
    assert inner.calls == [("exec_system", ["git", "status"])]


def test_shell_commands_run_through_the_configured_shell() -> None:
    default = LaunchedRunner(RecordingRunner(), PREFIX)
    bash = LaunchedRunner(RecordingRunner(), PREFIX, shell=("bash", "-lc"))

    assert default.launch(ExecCommand.bash("echo hi")).argv == [*PREFIX, "sh", "-c", "echo hi"]
    assert bash.launch(ExecCommand.bash("echo hi")).argv == [*PREFIX, "bash", "-lc", "echo hi"]


def test_function_launcher_receives_a_copy_and_must_return_a_command() -> None:
    seen: list[ExecCommand] = []

    def launcher(command: ExecCommand) -> ExecCommand:
        seen.append(command)
        argv = command.argv if command.kind == "process" else ["bash", "-c", command.shell]
        return ExecCommand.process("cmd", *argv)

    original = ExecCommand.process("echo", "hi")
    runner = LaunchedRunner(RecordingRunner(), launcher)

    assert runner.launch(original).argv == ["cmd", "echo", "hi"]
    assert seen[0] == original
    assert seen[0] is not original

    broken = LaunchedRunner(RecordingRunner(), lambda command: ["not", "a", "command"])
    with pytest.raises(TypeError, match="must return an ExecCommand"):
        broken.launch(original)


@pytest.mark.parametrize(
    ("launcher", "error"),
    [
        ("sudo -u sandbox", TypeError),
        ((), ValueError),
        (("sudo", ""), ValueError),
    ],
)
def test_launcher_must_be_an_argv_prefix_or_function(launcher, error) -> None:
    with pytest.raises(error):
        LaunchedRunner(RecordingRunner(), launcher)


def test_wrapped_runner_must_be_a_runner() -> None:
    with pytest.raises(TypeError, match="must be a Runner"):
        LaunchedRunner(object(), PREFIX)  # type: ignore[arg-type]


def test_positive_claims_are_unverified_unless_preserved_or_declared(tmp_path) -> None:
    inner = LocalRunner(tmp_path)
    plain = LaunchedRunner(inner, ("env",)).execution_admission_candidate()
    assert plain is not None
    claims = {claim.capability: claim for claim in plain.evidence.claims}
    # LocalRunner proves cancellation and cleanup of its own process group; a
    # launcher such as sudo can take the command out of that group.
    assert claims["confirmed_cleanup"].state == "unverified"
    assert claims["confirmed_cleanup"].reason_code == LAUNCHER_UNVERIFIED_REASON
    assert claims["confirmed_cancellation"].state == "unverified"
    # Negative claims are already honest and pass through.
    assert claims["deny_by_default_network"].state == "unsupported"
    assert claims["deny_by_default_network"].reason_code == "local_network_boundary_unsupported"

    declared = LaunchedRunner(
        LocalRunner(tmp_path),
        ("env",),
        preserve_claims=("confirmed_cleanup",),
        claims=(ExecutionCapabilityClaim.declared("deny_by_default_network"),),
    ).execution_admission_candidate()
    assert declared is not None
    claims = {claim.capability: claim for claim in declared.evidence.claims}
    assert claims["confirmed_cleanup"].state == "available"
    assert claims["confirmed_cancellation"].state == "unverified"
    assert claims["deny_by_default_network"].state == "declared"
    requirements = ExecutionRequirements(network_access="deny_by_default", cleanup="confirmed")
    assert (
        evaluate_execution_admission(
            candidate=declared.candidate,
            requirements=requirements,
            evidence=declared.evidence,
        ).status
        == "admitted"
    )
    assert (
        evaluate_execution_admission(
            candidate=plain.candidate,
            requirements=requirements,
            evidence=plain.evidence,
        ).status
        == "refused"
    )


def test_launcher_declarations_cannot_be_live_verified(tmp_path) -> None:
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    with pytest.raises(ValueError, match="live-verified"):
        LaunchedRunner(
            LocalRunner(tmp_path),
            ("env",),
            claims=(
                ExecutionCapabilityClaim.live_verified(
                    "deny_by_default_network",
                    observation="denied",
                    observed_at=now,
                    valid_until=now + timedelta(seconds=60),
                ),
            ),
        )


def _executable_requirements(executable: str) -> ExecutionRequirements:
    return ExecutionRequirements(
        tool_requirements=(
            ExecutionToolRequirement(
                tool_name="program",
                requirement=ToolExecutionRequirement(
                    name="program",
                    alternatives=(ToolExecutableRequirement(executable=executable),),
                ),
            ),
        )
    )


def test_executables_are_probed_through_the_launcher(tmp_path) -> None:
    present = sys.executable
    requirements = _executable_requirements(present)
    host = LocalRunner(tmp_path).execution_admission_candidate_for(requirements)
    assert host is not None
    assert host.evidence.tool_requirements.executable_for(present).state == "live_verified"

    # The launcher hides every executable from the command, so the host's own
    # lookup must not be reported as availability for launched commands.
    hidden = LaunchedRunner(
        LocalRunner(tmp_path),
        lambda command: ExecCommand.process("sh", "-c", "exit 1"),
    )
    visible = LaunchedRunner(LocalRunner(tmp_path), ("env",))

    async def collect(runner: LaunchedRunner):
        observer = runner.execution_admission_observer(requirements)
        before = observer.snapshot()
        collected = await observer.collect()
        return before, collected, observer.snapshot()

    before, collected, after = asyncio.run(collect(hidden))
    assert before.evidence.tool_requirements.executable_for(present).state == "declared"
    assert collected.evidence.tool_requirements.executable_for(present).state == "unavailable"
    assert after == collected

    _, collected, _ = asyncio.run(collect(visible))
    evidence = collected.evidence
    assert evidence.tool_requirements.executable_for(present).state == "live_verified"
    assert evidence.environment_fingerprint == evidence.tool_requirements.environment_fingerprint
    assert evidence.environment_fingerprint != host.evidence.environment_fingerprint


def test_binary_streams_run_through_the_launcher_on_a_local_runner(tmp_path) -> None:
    runner = LaunchedRunner(LocalRunner(tmp_path), ("env", "CAYU_LAUNCHED=yes"))
    capability = runner.binary_stream_capability()
    assert capability is not None
    stdout = io.BytesIO()

    result = asyncio.run(
        capability.exec_stream(
            ExecCommand.bash('printf "%s:" "$CAYU_LAUNCHED"; cat'),
            stdin=io.BytesIO(b"bytes"),
            stdout=stdout,
        )
    )

    assert result.exit_code == 0
    assert stdout.getvalue() == b"yes:bytes"


def test_lifecycle_follows_the_wrapped_runner(tmp_path) -> None:
    inner = LocalRunner(tmp_path)
    runner = LaunchedRunner(inner, ("env",))
    assert runner.lifecycle_state == "reusable"
    inner._close_exec("fenced for the test")
    assert runner.lifecycle_state == "fenced"
    runner.reopen_exec()
    assert inner.lifecycle_state == "reusable"

    asyncio.run(runner.close())
    assert inner.is_closed
    assert runner.is_closed
    assert runner.lifecycle_state == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(runner.exec(ExecCommand.process("true")))


def test_browser_endpoint_denials_pass_through_and_reachability_does_not() -> None:
    class Declaring(RecordingRunner):
        def browser_control_endpoint_reachable(self, endpoint: str) -> bool | None:
            return endpoint == "reachable"

    runner = LaunchedRunner(Declaring(), PREFIX)
    assert runner.browser_control_endpoint_reachable("reachable") is None
    assert runner.browser_control_endpoint_reachable("other") is False


def test_built_in_tools_run_through_the_launcher(tmp_path) -> None:
    if shutil.which("rg") is None:
        pytest.skip("search_text needs ripgrep on PATH.")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")
    launched: list[list[str]] = []

    def launcher(command: ExecCommand) -> ExecCommand:
        argv = command.argv if command.kind == "process" else ["sh", "-c", command.shell]
        launched.append(list(argv))
        return ExecCommand.process("env", "CAYU_LAUNCHED=yes", *argv)

    def step(index: int, name: str, arguments: dict[str, object]):
        return [
            ModelStreamEvent.tool_call(id=f"c{index}", name=name, arguments=arguments),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    async def exercise():
        app = CayuApp(enable_logging=False)
        app.register_provider(
            ScriptedModelProvider(
                [
                    step(1, "exec_command", {"shell": "echo launched=$CAYU_LAUNCHED"}),
                    step(2, "search_text", {"pattern": "x = 1"}),
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
                EnvironmentSpec(name="ws"),
                workspace=LocalWorkspace(tmp_path, workspace_id="ws"),
                runner=LaunchedRunner(LocalRunner(tmp_path), launcher),
            ),
            default=True,
        )
        tools = [ExecCommandTool(), SearchTextTool()]
        app.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=tools,
            tool_policy=StaticToolPolicy(allow=[tool.name for tool in tools]),
        )
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="launched",
                    max_steps=4,
                    messages=[Message.text("user", "go")],
                )
            )
        ]

    events = asyncio.run(exercise())
    assert any(event.type is EventType.SESSION_COMPLETED for event in events)
    completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 2
    assert "launched=yes" in str(completed[0].payload["result"])
    assert completed[1].payload["result"]["structured"]["matches"] == [{"path": "app.py"}]
    # The admission probe for rg, then both tool commands, went through the
    # launcher. Tool commands are launched twice: preflight, then dispatch.
    probe, exec_preflight, exec_dispatch, search_preflight, search_dispatch = launched
    assert probe[:2] == ["sh", "-c"] and probe[-1] == "rg"
    assert exec_preflight == exec_dispatch == ["sh", "-c", "echo launched=$CAYU_LAUNCHED"]
    assert search_preflight == search_dispatch
    assert search_dispatch[0] == "rg"


def test_composes_with_a_local_workspace_factory(tmp_path) -> None:
    if shutil.which("rg") is None:
        pytest.skip("search_text needs ripgrep on PATH.")
    (tmp_path / "app.py").write_text("x = 1\n", encoding="utf-8")

    def runner_factory(root, request) -> LaunchedRunner:
        del request
        return LaunchedRunner(LocalRunner(root), ("env", "CAYU_LAUNCHED=yes"))

    def response(request):
        if any(message.role == "tool" for message in request.messages):
            return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
        return [
            ModelStreamEvent.tool_call(
                id="c1", name="exec_command", arguments={"shell": "echo $CAYU_LAUNCHED"}
            ),
            ModelStreamEvent.tool_call(id="c2", name="search_text", arguments={"pattern": "x"}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    async def exercise():
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider(response_factory=response), default=True)
        tools = [ExecCommandTool(), SearchTextTool()]
        app.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=tools,
            tool_policy=StaticToolPolicy(allow=[tool.name for tool in tools]),
        )
        app.register_environment_factory(
            EnvironmentSpec(name="local"),
            LocalWorkspaceEnvironmentFactory(
                allowed_roots=[tmp_path],
                root_from_request=lambda request: tmp_path,
                runner_factory=runner_factory,
            ),
            default=True,
        )
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="factory",
                    messages=[Message.text("user", "go")],
                )
            )
        ]

    events = asyncio.run(exercise())
    assert any(event.type is EventType.SESSION_COMPLETED for event in events)
    completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 2
    assert any("yes" in str(event.payload["result"]) for event in completed)


def test_command_path_cannot_replace_the_launcher(tmp_path) -> None:
    marker = tmp_path / "hijacked"
    fake = tmp_path / "cayu-launcher-for-test"
    fake.write_text(f'#!/bin/sh\ntouch "{marker}"\nexec "$@"\n', encoding="utf-8")
    fake.chmod(0o755)
    runner = LaunchedRunner(LocalRunner(tmp_path), ["cayu-launcher-for-test", "--"])

    async def exercise():
        return await runner.exec(
            ExecCommand.process("true"), env={"PATH": f"{tmp_path}:/usr/bin:/bin"}
        )

    try:
        result = asyncio.run(exercise())
    except Exception:
        result = None
    # The launcher is looked up with the wrapped runner's own PATH, where the
    # workspace file doesn't exist, so it never runs.
    assert not marker.exists()
    assert result is None or result.exit_code != 0


def test_command_environment_is_applied_inside_the_launcher(tmp_path) -> None:
    seen_by_launcher = tmp_path / "launcher-env"
    prefix = (
        "/bin/sh",
        "-c",
        'printf "%s" "${PROBE-unset}" > "$0"; exec "$@"',
        str(seen_by_launcher),
    )
    runner = LaunchedRunner(LocalRunner(tmp_path), prefix)

    result = asyncio.run(
        runner.exec(
            ExecCommand.bash('printf "%s:%s" "$PROBE" "${HOME-removed}"'),
            env={"PROBE": "inside", "HOME": "/kept"},
            env_remove=("HOME",),
        )
    )

    assert result.exit_code == 0, result.stderr
    assert result.stdout == "inside:removed"
    assert seen_by_launcher.read_text(encoding="utf-8") == "unset"
    launched = runner.launch(ExecCommand.process("id"), env={"A": "1"}, env_remove=("B",))
    assert launched.argv == [*prefix, "env", "-u", "B", "--", "A=1", "id"]


def test_prefix_without_option_terminator_refuses_option_shaped_commands() -> None:
    open_prefix = LaunchedRunner(RecordingRunner(), ("bwrap", "--unshare-all"))
    with pytest.raises(ValueError, match="cannot start with an option"):
        open_prefix.launch(ExecCommand.process("--bind", "/", "/", "sh"))
    assert open_prefix.launch(ExecCommand.process("sh")).argv == ["bwrap", "--unshare-all", "sh"]

    terminated = LaunchedRunner(RecordingRunner(), ("bwrap", "--unshare-all", "--"))
    assert terminated.launch(ExecCommand.process("--weird-name")).argv[-1] == "--weird-name"


def test_deferred_settlement_follows_the_wrapped_runner_declaration() -> None:
    from cayu.runners.base import runner_pending_command_settlement_cancellation_safe

    class SafeRunner(RecordingRunner):
        pending_command_settlement_cancellation_safe = True

        async def await_pending_command_settlement(self) -> bool:
            self.calls.append(("settle", []))
            return True

    class UnsafeRunner(RecordingRunner):
        async def await_pending_command_settlement(self) -> bool:
            self.calls.append(("settle", []))
            return True

    safe_inner = SafeRunner()
    safe = LaunchedRunner(safe_inner, PREFIX)
    assert runner_pending_command_settlement_cancellation_safe(safe)
    assert asyncio.run(safe.await_pending_command_settlement()) is True
    assert safe_inner.calls == [("settle", [])]

    unsafe_inner = UnsafeRunner()
    unsafe = LaunchedRunner(unsafe_inner, PREFIX)
    assert asyncio.run(unsafe.await_pending_command_settlement()) is False
    assert unsafe_inner.calls == []


def test_snapshot_follows_changes_to_the_wrapped_environment(tmp_path) -> None:
    class ChangingRunner(LocalRunner):
        fingerprint = "sha256:" + "1" * 64
        network_available = True

        def execution_admission_candidate(self):
            candidate = super().execution_admission_candidate()
            claims = [
                claim
                for claim in candidate.evidence.claims
                if claim.capability != "deny_by_default_network"
            ]
            claims.append(
                ExecutionCapabilityClaim.available("deny_by_default_network")
                if self.network_available
                else ExecutionCapabilityClaim.unsupported(
                    "deny_by_default_network",
                    reason_code="network_withdrawn",
                    remediation_code="select_isolated_execution",
                )
            )
            evidence = candidate.evidence.model_copy(
                update={"environment_fingerprint": self.fingerprint, "claims": tuple(claims)}
            )
            return candidate.model_copy(update={"evidence": evidence})

    inner = ChangingRunner(tmp_path)
    runner = LaunchedRunner(inner, ("env",), preserve_claims=("deny_by_default_network",))
    requirements = _executable_requirements(sys.executable)
    observer = runner.execution_admission_observer(requirements)

    def states(candidate):
        return (
            candidate.evidence.claim_for("deny_by_default_network").state,
            candidate.evidence.tool_requirements.executable_for(sys.executable).state,
        )

    collected = asyncio.run(observer.collect())
    assert states(collected) == ("available", "live_verified")
    assert states(observer.snapshot()) == ("available", "live_verified")

    # A withdrawn capability shows at once; probed executables stay valid
    # while the wrapped environment keeps its fingerprint.
    inner.network_available = False
    assert states(observer.snapshot()) == ("unsupported", "live_verified")

    inner.fingerprint = "sha256:" + "2" * 64
    changed = observer.snapshot()
    assert states(changed) == ("unsupported", "declared")
    assert changed.evidence.environment_fingerprint != collected.evidence.environment_fingerprint


def test_git_changes_accepts_a_local_workspace_behind_a_launcher(tmp_path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git_changes needs git on PATH.")
    import subprocess

    from cayu.tools import GitChangesTool

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "new.txt").write_text("hello\n", encoding="utf-8")

    def response(request):
        if any(message.role == "tool" for message in request.messages):
            return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed()]
        return [
            ModelStreamEvent.tool_call(id="c1", name="git_changes", arguments={}),
            ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
        ]

    async def exercise():
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider(response_factory=response), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="ws"),
                workspace=LocalWorkspace(tmp_path, workspace_id="ws"),
                runner=LaunchedRunner(LocalRunner(tmp_path), ("env",)),
            ),
            default=True,
        )
        tool = GitChangesTool()
        app.register_agent(
            AgentSpec(name="agent", model="scripted"),
            tools=[tool],
            tool_policy=StaticToolPolicy(allow=[tool.name]),
        )
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent", session_id="git", messages=[Message.text("user", "go")]
                )
            )
        ]

    events = asyncio.run(exercise())
    completed = [event for event in events if event.type is EventType.TOOL_CALL_COMPLETED]
    assert len(completed) == 1
    result = str(completed[0].payload["result"])
    assert "workspace_runner_mismatch" not in result
    assert "new.txt" in result
