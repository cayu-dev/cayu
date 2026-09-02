from __future__ import annotations

import asyncio
import json
import sys
from hashlib import sha256

import pytest

import cayu.tools.named_checks as named_checks
from cayu import (
    REDACTED_SECRET,
    AgentSpec,
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    CommandPolicyDecision,
    DockerCodingCommandAuthority,
    DockerCodingDependencyInput,
    DockerCodingToolchainProfile,
    DockerImageIdentity,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecCommand,
    ExecResult,
    ExecutionProfileBehaviorIdentity,
    ExecutionProfileMismatchError,
    InMemorySessionStore,
    LocalArtifactStore,
    LocalRunner,
    LocalWorkspace,
    Message,
    ModelStreamEvent,
    NamedCheck,
    ProcessCommandPolicy,
    ResumeRequest,
    RunCheckTool,
    RunRequest,
    ScriptedModelProvider,
    SecretRedactor,
    StaticToolPolicy,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolEffect,
)
from cayu.core.tools import ToolContext
from cayu.runners import RunnerExecutionError, RunnerUnavailableError
from cayu.runtime.checks import check_manifest
from cayu.tools._redaction import InvocationRedactorSnapshot
from cayu.tools._runner import InvocationRunnerHandle
from cayu.workspaces.revisions import (
    WorkspaceRevisionObservationLimits,
    observe_deterministic_workspace,
)


def _identity(name: str = "project-check") -> ExecutionProfileBehaviorIdentity:
    return ExecutionProfileBehaviorIdentity(
        name=name,
        behavior_version="1",
        implementation_version="1",
    )


def _check(
    name: str = "test",
    *,
    command: ExecCommand | None = None,
    timeout_s: int = 60,
    max_output_bytes: int = 50_000,
    description: str = "Run the deterministic unit tests.",
) -> NamedCheck:
    return NamedCheck(
        name=name,
        description=description,
        command=command or ExecCommand.process("pytest", "-q"),
        timeout_s=timeout_s,
        max_output_bytes=max_output_bytes,
        execution_profile_identity=_identity(name),
    )


def _policy(
    *,
    allowed: tuple[str, ...] = ("pytest",),
    approval_required: tuple[str, ...] = (),
) -> ProcessCommandPolicy:
    return ProcessCommandPolicy(
        allowed_executables=allowed,
        approval_required_executables=approval_required,
        allowed_cwds=("/workspace",),
    )


class RecordingRunner:
    def __init__(self, result: object | BaseException | None = None) -> None:
        self.result = ExecResult() if result is None else result
        self.resolved: list[str | None] = []
        self.preflights: list[tuple[ExecCommand, dict[str, object]]] = []
        self.executions: list[tuple[ExecCommand, dict[str, object]]] = []

    def resolve_cwd(self, cwd: str | None = None) -> str:
        self.resolved.append(cwd)
        return "/workspace" if cwd is None else cwd

    def preflight_exec(self, command: ExecCommand, **kwargs: object) -> None:
        self.preflights.append((command, kwargs))

    async def exec(self, command: ExecCommand, **kwargs: object) -> object:
        self.executions.append((command, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _run(
    tool: RunCheckTool,
    runner: RecordingRunner,
    args: dict[str, object] | None = None,
    *,
    artifact_store: LocalArtifactStore | None = None,
):
    return asyncio.run(
        tool.run(
            ToolContext(
                session_id="sess_checks",
                agent_name="coding",
                environment_name="coding",
                idempotency_key="idem_check_1",
                runner=runner,
                artifact_store=artifact_store,
            ),
            {"check": "test"} if args is None else args,
        )
    )


def test_named_check_is_public_immutable_and_snapshots_process_argv() -> None:
    command = ExecCommand.process("pytest", "-q")
    check = _check(command=command)

    command.argv.append("mutated")
    returned = check.command
    returned.argv.append("also-mutated")

    assert check.command == ExecCommand.process("pytest", "-q")
    assert check.required_executables == ("pytest",)
    with pytest.raises(AttributeError):
        check.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"name": " "}, ValueError, "cannot be blank"),
        ({"description": ""}, ValueError, "cannot be blank"),
        ({"command": ExecCommand.bash("pytest -q")}, ValueError, "process-form"),
        ({"timeout_s": 0}, ValueError, "greater than zero"),
        ({"timeout_s": 601}, ValueError, "at most 600"),
        ({"max_output_bytes": 0}, ValueError, "greater than zero"),
        ({"max_output_bytes": 200_001}, ValueError, "at most 200000"),
        ({"execution_profile_identity": None}, TypeError, "ExecutionProfileBehaviorIdentity"),
    ],
)
def test_named_check_rejects_invalid_declarations(
    overrides: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    values: dict[str, object] = {
        "name": "test",
        "description": "Run tests.",
        "command": ExecCommand.process("pytest", "-q"),
        "timeout_s": 60,
        "max_output_bytes": 50_000,
        "execution_profile_identity": _identity(),
    }
    values.update(overrides)

    with pytest.raises(error, match=message):
        NamedCheck(**values)  # type: ignore[arg-type]


def test_named_check_validates_and_canonicalizes_executable_requirements() -> None:
    check = NamedCheck(
        name="test",
        description="Run tests.",
        command=ExecCommand.process("uv", "run", "pytest", "-q"),
        execution_profile_identity=_identity(),
        required_executables=("pytest", "uv", "pytest"),
    )

    assert check.required_executables == ("pytest", "uv")
    with pytest.raises(TypeError, match="iterable of strings"):
        NamedCheck(
            name="bad",
            description="Bad requirements.",
            command=ExecCommand.process("uv", "run", "pytest"),
            execution_profile_identity=_identity(),
            required_executables="uv",  # type: ignore[arg-type]
        )


def test_run_check_schema_is_closed_and_contains_only_finite_names() -> None:
    tool = RunCheckTool(
        checks=[_check("test"), _check("format")],
        command_policy=_policy(),
    )

    assert tool.schema == {
        "type": "object",
        "properties": {
            "check": {"type": "string", "enum": ["format", "test"]},
        },
        "required": ["check"],
        "additionalProperties": False,
    }
    assert tool.spec.parallel_safe is False
    assert tool.spec.effect is ToolEffect.EXTERNAL
    assert tool.spec.workspace_mutation is True
    assert not {
        "argv",
        "shell",
        "cwd",
        "env",
        "stdin",
        "timeout_s",
        "max_output_bytes",
        "network",
        "image",
        "runner",
    }.intersection(tool.schema["properties"])


def test_run_check_rejects_empty_duplicate_mutable_or_unguarded_configuration() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RunCheckTool(checks=[], command_policy=_policy())
    with pytest.raises(ValueError, match="duplicate names"):
        RunCheckTool(checks=[_check(), _check()], command_policy=_policy())
    with pytest.raises(TypeError, match="NamedCheck"):
        RunCheckTool(checks=[object()], command_policy=_policy())  # type: ignore[list-item]
    with pytest.raises(TypeError, match="CommandPolicy"):
        RunCheckTool(checks=[_check()], command_policy=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"check": "unknown"},
        {"check": ["test"]},
        {"check": "test", "argv": ["pytest"]},
        {"check": "test", "timeout_s": 1},
    ],
)
def test_run_check_rejects_every_non_selector_model_input(args: dict[str, object]) -> None:
    runner = RecordingRunner()

    result = _run(RunCheckTool(checks=[_check()], command_policy=_policy()), runner, args)

    assert result.is_error is True
    assert result.structured == {"error": "invalid_arguments"}
    assert runner.resolved == []
    assert runner.preflights == []
    assert runner.executions == []


def test_run_check_reuses_exact_command_preflight_policy_and_runner_boundary() -> None:
    runner = RecordingRunner(ExecResult(stdout="ok\n", exit_code=0, stdout_bytes=3))
    check = _check(timeout_s=123, max_output_bytes=45_678)
    tool = RunCheckTool(checks=[check], command_policy=_policy())

    result = _run(tool, runner)

    assert result.structured["status"] == "passed"
    assert type(result.structured["duration_ms"]) is int
    assert result.structured["duration_ms"] >= 0
    assert runner.resolved == [None]
    assert len(runner.preflights) == 1
    assert len(runner.executions) == 1
    preflight_command, preflight_options = runner.preflights[0]
    executed_command, executed_options = runner.executions[0]
    assert preflight_command == check.command
    assert executed_command == check.command
    expected_options = {
        "cwd": "/workspace",
        "env": None,
        "timeout_s": 123,
        "stdin": None,
        "output_limit_bytes": 45_678,
    }
    assert preflight_options == expected_options
    assert executed_options == expected_options


def test_run_check_binds_a_pass_to_the_complete_post_check_workspace_revision(
    tmp_path,
) -> None:
    (tmp_path / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    workspace = LocalWorkspace(tmp_path, workspace_id="check-workspace")
    command = ExecCommand.process(sys.executable, "-c", "pass")
    tool = RunCheckTool(
        checks=[_check(command=command)],
        command_policy=ProcessCommandPolicy(
            allowed_executables=(sys.executable,),
            allowed_cwds=(str(tmp_path),),
        ),
    )
    result = asyncio.run(
        tool.run(
            ToolContext(
                session_id="sess_revision_check",
                workspace_id=workspace.id,
                workspace=workspace,
                runner=LocalRunner(tmp_path, inherit_env=False),
            ),
            {"check": "test"},
        )
    )
    observed = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="test-check-revision",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )

    assert result.structured["status"] == "passed"
    assert result.structured["workspace_observation_status"] == "supported"
    assert result.structured["workspace_revision"] == observed.revision


def test_run_check_rejects_a_pass_that_changes_toolchain_dependencies(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "uv.lock"
    lock_path.write_bytes(b"locked\n")
    script = "from pathlib import Path; Path('uv.lock').write_bytes(b'changed\\n')"
    authority = DockerCodingCommandAuthority(
        selector="test",
        revision="1",
        description="Run a dependency-sensitive test.",
        exposure="named_check",
        executable=sys.executable,
        fixed_arguments=("-c", script),
        max_arguments=0,
    )
    profile = DockerCodingToolchainProfile(
        profile_id="test-python",
        revision="1",
        image_identity=DockerImageIdentity(
            reference="test-python@sha256:" + ("a" * 64),
        ),
        platform_architecture="arm64",
        command_authorities=(authority,),
        dependency_inputs=(
            DockerCodingDependencyInput(
                path="uv.lock",
                content_sha256="sha256:" + sha256(b"locked\n").hexdigest(),
            ),
        ),
    )
    workspace = LocalWorkspace(tmp_path, workspace_id="dependency-check-workspace")
    monkeypatch.setattr(
        named_checks,
        "docker_coding_toolchain_runner_admission_failure",
        lambda _runner, **_kwargs: None,
    )
    tool = RunCheckTool(
        checks=[
            _check(
                command=ExecCommand.process(*authority.command_argv()),
            )
        ],
        command_policy=ProcessCommandPolicy(
            allowed_executables=(sys.executable,),
            allowed_cwds=(str(tmp_path),),
        ),
        toolchain_profile=profile,
    )

    result = asyncio.run(
        tool.run(
            ToolContext(
                session_id="sess_dependency_mutating_check",
                workspace_id=workspace.id,
                workspace=workspace,
                runner=LocalRunner(tmp_path, inherit_env=False),
            ),
            {"check": "test"},
        )
    )
    observed = asyncio.run(
        observe_deterministic_workspace(
            workspace,
            observer="test-check-revision",
            limits=WorkspaceRevisionObservationLimits(),
        )
    )

    assert lock_path.read_bytes() == b"changed\n"
    assert result.is_error is True
    assert result.structured["status"] == "stale_toolchain"
    assert result.structured["error"] == "dependency_inputs_changed"
    assert result.structured["exit_code"] == 0
    assert result.structured["workspace_revision"] == observed.revision
    assert result.structured["dependency_path_count"] == 1


def test_run_check_rejects_a_mismatched_workspace_observation_identity(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path, workspace_id="actual-workspace")
    command = ExecCommand.process(sys.executable, "-c", "pass")
    tool = RunCheckTool(
        checks=[_check(command=command)],
        command_policy=ProcessCommandPolicy(
            allowed_executables=(sys.executable,),
            allowed_cwds=(str(tmp_path),),
        ),
    )

    result = asyncio.run(
        tool.run(
            ToolContext(
                session_id="sess_mismatched_check_workspace",
                workspace_id="different-workspace",
                workspace=workspace,
                runner=LocalRunner(tmp_path, inherit_env=False),
            ),
            {"check": "test"},
        )
    )

    assert result.is_error is True
    assert result.structured["status"] == "partial"
    assert result.structured["error"] == "workspace_observation_incomplete"
    assert result.structured["workspace_observation_status"] == "unavailable"
    assert result.structured["workspace_revision"] is None


@pytest.mark.parametrize(
    ("policy", "status", "error"),
    [
        (_policy(allowed=()), "policy_denied", "command_denied"),
        (
            _policy(allowed=(), approval_required=("pytest",)),
            "approval_required",
            "command_approval_required",
        ),
    ],
)
def test_run_check_preserves_inline_command_policy_refusals(
    policy: ProcessCommandPolicy,
    status: str,
    error: str,
) -> None:
    runner = RecordingRunner()
    ctx = ToolContext(session_id="sess_policy", runner=runner)
    tool = RunCheckTool(checks=[_check()], command_policy=policy)

    result = asyncio.run(tool.run(ctx, {"check": "test"}))

    assert result.is_error is True
    assert result.structured["status"] == status
    assert type(result.structured["duration_ms"]) is int
    assert result.structured["duration_ms"] >= 0
    assert result.structured["error"] == error
    assert ctx._policy_denial_for(tool) is not None
    assert runner.executions == []


@pytest.mark.parametrize(
    ("exec_result", "status", "is_error"),
    [
        (ExecResult(exit_code=0), "passed", False),
        (ExecResult(stderr="failure\n", exit_code=3), "failed", False),
        (ExecResult(exit_code=-9, timed_out=True), "timed_out", True),
        (ExecResult(exit_code=-9, cancelled=True), "cancelled", True),
    ],
)
def test_run_check_classifies_completed_and_interrupted_results(
    exec_result: ExecResult,
    status: str,
    is_error: bool,
) -> None:
    result = _run(
        RunCheckTool(checks=[_check()], command_policy=_policy()),
        RecordingRunner(exec_result),
    )

    assert result.structured["status"] == status
    assert result.is_error is is_error
    assert result.structured["exit_code"] == exec_result.exit_code
    if status == "failed":
        assert "exit code 3" in result.content
    if status in {"timed_out", "cancelled"}:
        assert result.structured["workspace_mutation_settlement"] == "uncertain"
        assert result.structured["cleanup_uncertain"] is True


def test_run_check_refuses_success_with_uncertain_cleanup() -> None:
    result = _run(
        RunCheckTool(checks=[_check()], command_policy=_policy()),
        RecordingRunner(
            ExecResult(
                exit_code=0,
                artifacts=[
                    {
                        "type": "cayu.runner_cleanup.v1",
                        "adapter": "docker",
                        "action": "kill_command",
                        "status": "failed",
                    }
                ],
            )
        ),
    )

    assert result.structured["status"] == "ambiguous"
    assert result.structured["error"] == "workspace_cleanup_uncertain"
    assert result.structured["cleanup_uncertain"] is True
    assert result.is_error is True


def test_run_check_distinguishes_unavailable_failed_and_malformed_runners() -> None:
    unavailable = RunnerUnavailableError(
        "Runner unavailable.",
        diagnostic={
            "type": "cayu.runner_unavailable.v1",
            "adapter": "docker",
            "status": "unavailable",
        },
    )
    execution_failure = RunnerExecutionError(
        diagnostic={
            "type": "cayu.runner_execution_error.v1",
            "adapter": "docker",
            "status": "failed",
            "error_type": "RuntimeError",
            "timed_out": False,
            "cancelled": False,
        }
    )
    tool = RunCheckTool(checks=[_check()], command_policy=_policy())

    unavailable_result = _run(tool, RecordingRunner(unavailable))
    failed_result = _run(tool, RecordingRunner(execution_failure))
    malformed_result = _run(tool, RecordingRunner({"exit_code": 0}))

    assert unavailable_result.structured["status"] == "runner_unavailable"
    assert failed_result.structured["status"] == "execution_failed"
    assert failed_result.structured["workspace_mutation_settlement"] == "uncertain"
    assert failed_result.structured["cleanup_uncertain"] is True
    assert malformed_result.structured["status"] == "malformed_execution"
    assert unavailable_result.is_error is True
    assert failed_result.is_error is True
    assert malformed_result.is_error is True


def test_run_check_bounds_model_output_and_retains_captured_log_as_artifact(tmp_path) -> None:
    stdout = "é" * 100
    artifact_store = LocalArtifactStore(tmp_path / "artifacts", store_id="checks")
    tool = RunCheckTool(
        checks=[_check(max_output_bytes=1_000)],
        command_policy=_policy(),
        max_model_output_bytes=64,
    )

    result = _run(
        tool,
        RecordingRunner(
            ExecResult(
                stdout=stdout,
                exit_code=0,
                stdout_bytes=len(stdout.encode("utf-8")),
            )
        ),
        artifact_store=artifact_store,
    )

    assert result.structured["status"] == "passed"
    assert result.structured["stdout_projection_truncated"] is True
    assert len(result.structured["stdout"].encode("utf-8")) <= 64
    assert result.structured["output_artifact_status"] == "stored"
    assert len(result.artifacts) == 1
    artifact_id = result.artifacts[0]["id"]
    stored = asyncio.run(artifact_store.read_bytes(artifact_id))
    assert stdout.encode("utf-8") in stored.content
    assert result.structured["output_sha256"].startswith("sha256:")


def test_named_check_and_tool_profile_identity_is_canonical_and_behavior_sensitive() -> None:
    format_check = _check("format", command=ExecCommand.process("ruff", "format", "--check"))
    test_check = _check("test")
    forward = RunCheckTool(checks=[format_check, test_check], command_policy=_policy())
    reversed_tool = RunCheckTool(checks=[test_check, format_check], command_policy=_policy())

    assert forward._execution_profile_material() == reversed_tool._execution_profile_material()
    assert forward.schema == reversed_tool.schema
    assert _check().profile_fingerprint == _check().profile_fingerprint
    assert _check(
        command=ExecCommand.process("pytest", "-q", "tests/core")
    ).profile_fingerprint != (_check().profile_fingerprint)
    assert _check(timeout_s=61).profile_fingerprint != _check().profile_fingerprint
    assert _check(max_output_bytes=49_999).profile_fingerprint != _check().profile_fingerprint
    assert _check(description="Run another suite.").profile_fingerprint != (
        _check().profile_fingerprint
    )


def test_manifest_exposes_bounded_check_descriptors_and_static_diagnostics() -> None:
    app = CayuApp(enable_logging=False)
    app.register_provider(ScriptedModelProvider([], name="scripted"), default=True)
    app.register_agent(
        AgentSpec(
            name="coding",
            model="test-model",
            workflow_tool_names=("run_check",),
        ),
        tools=[RunCheckTool(checks=[_check()], command_policy=_policy())],
        tool_policy=StaticToolPolicy(allow=("run_check",)),
    )

    manifest = app.describe()
    described = manifest.agents[0].tools[0]
    report = check_manifest(manifest)

    assert described.name == "run_check"
    assert described.command_policy == "ProcessCommandPolicy"
    assert len(described.named_checks) == 1
    assert described.named_checks[0].name == "test"
    assert described.named_checks[0].required_executables == ("pytest",)
    assert described.named_checks[0].profile_fingerprint.startswith("sha256:")
    assert "argv" not in described.named_checks[0].model_dump(mode="json")
    assert "AGENT_WORKFLOW_RUNNER_NOT_REGISTERED" in {
        diagnostic.code for diagnostic in report.diagnostics
    }


def test_public_agent_path_exposes_only_selector_and_returns_failing_check_evidence(
    tmp_path,
) -> None:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_test",
                    name="run_check",
                    arguments={"check": "test"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("The check failed and needs repair."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    command = ExecCommand.process(sys.executable, "-c", "raise SystemExit(7)")
    check = _check(command=command)
    tool = RunCheckTool(
        checks=[check],
        command_policy=ProcessCommandPolicy(
            allowed_executables=(sys.executable,),
            allowed_cwds=(str(tmp_path),),
        ),
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(
            EnvironmentSpec(name="coding"),
            runner=LocalRunner(tmp_path),
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="coding", model="test-model"),
        tools=[tool],
        tool_policy=StaticToolPolicy(allow=("run_check",)),
    )

    async def run() -> list[object]:
        return [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="coding",
                    session_id="sess_public_check",
                    messages=[Message.text("user", "Run the tests.")],
                )
            )
        ]

    asyncio.run(run())

    assert len(provider.requests) == 2
    assert provider.requests[0].tools == [
        {
            "name": "run_check",
            "description": "Run one application-defined bounded check by name.",
            "input_schema": tool.schema,
        }
    ]
    tool_message = next(
        message
        for message in provider.requests[1].messages
        if message.role == "tool" and message.content[0].tool_call_id == "call_test"
    )
    assert "Check 'test' failed with exit code 7." in tool_message.content[0].content


def test_run_check_output_is_redacted_at_runner_capture_before_projection(tmp_path) -> None:
    secret = "named-check-secret"
    prefix = "p" * 40

    def redactor_provider() -> SecretRedactor:
        return SecretRedactor(secret)

    command = ExecCommand.process(
        sys.executable,
        "-c",
        f"import sys; sys.stdout.write({(prefix + secret)!r})",
    )
    tool = RunCheckTool(
        checks=[_check(command=command, max_output_bytes=50)],
        command_policy=ProcessCommandPolicy(
            allowed_executables=(sys.executable,),
            allowed_cwds=(str(tmp_path),),
        ),
    )
    ctx = ToolContext(
        session_id="sess_redacted_check",
        runner=InvocationRunnerHandle(
            LocalRunner(tmp_path),
            redactor_snapshot_provider=lambda: InvocationRedactorSnapshot(
                revision=0,
                redactor=redactor_provider(),
            ),
        ),
        invocation_secret_redactor=redactor_provider,
    )

    result = asyncio.run(tool.run(ctx, {"check": "test"}))
    serialized = json.dumps(result.model_dump(mode="json"))

    assert result.structured["status"] == "passed"
    assert secret not in serialized
    assert REDACTED_SECRET not in serialized
    assert result.structured["stdout"] == prefix
    assert result.structured["stdout_runner_truncated"] is True


def test_run_check_uses_runtime_durable_tool_approval_before_execution(tmp_path) -> None:
    marker = tmp_path / "approved.txt"
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="call_approved_test",
                    name="run_check",
                    arguments={"check": "test"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("Approved check completed."),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    command = ExecCommand.process(
        sys.executable,
        "-c",
        f"from pathlib import Path; Path({str(marker)!r}).write_text('approved')",
    )
    tool = RunCheckTool(
        checks=[_check(command=command)],
        command_policy=ProcessCommandPolicy(
            allowed_executables=(sys.executable,),
            allowed_cwds=(str(tmp_path),),
        ),
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_environment(
        Environment(EnvironmentSpec(name="coding"), runner=LocalRunner(tmp_path)),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="coding", model="test-model"),
        tools=[tool],
        tool_policy=AlwaysRequireApprovalToolPolicy(tools=("run_check",)),
    )

    async def exercise() -> tuple[list[object], list[object]]:
        first = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="coding",
                    session_id="sess_approved_check",
                    messages=[Message.text("user", "Run the check after approval.")],
                )
            )
        ]
        assert not marker.exists()
        approval = next(
            event for event in first if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
        )
        continued = [
            event
            async for event in app.resolve_tool_approval(
                ToolApprovalRequest(
                    session_id="sess_approved_check",
                    approval_id=approval.payload["approval"]["approval_id"],
                    tool_round_id=approval.payload["tool_round_id"],
                    tool_call_id=approval.payload["tool_call_id"],
                    decision=ToolApprovalDecision.APPROVE,
                )
            )
        ]
        return first, continued

    first, continued = asyncio.run(exercise())

    assert first[-1].type == EventType.SESSION_INTERRUPTED
    assert marker.read_text() == "approved"
    completed = next(event for event in continued if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["result"]["structured"]["status"] == "passed"


def test_resume_rejects_changed_named_check_profile_before_provider_work() -> None:
    store = InMemorySessionStore()

    def configured_app(command: ExecCommand) -> tuple[CayuApp, ScriptedModelProvider]:
        provider = ScriptedModelProvider(
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ]
        )
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="coding", model="test-model"),
            tools=[
                RunCheckTool(
                    checks=[_check(command=command)],
                    command_policy=_policy(),
                )
            ],
            tool_policy=StaticToolPolicy(allow=("run_check",)),
        )
        return app, provider

    async def exercise() -> ScriptedModelProvider:
        original, _ = configured_app(ExecCommand.process("pytest", "-q"))
        _ = [
            event
            async for event in original.run(
                RunRequest(
                    agent_name="coding",
                    session_id="sess_named_check_drift",
                    messages=[Message.text("user", "first")],
                )
            )
        ]
        replacement, provider = configured_app(ExecCommand.process("pytest", "-q", "tests/core"))
        with pytest.raises(ExecutionProfileMismatchError) as caught:
            _ = [
                event
                async for event in replacement.resume(
                    ResumeRequest(
                        session_id="sess_named_check_drift",
                        messages=[Message.text("user", "continue")],
                    )
                )
            ]
        assert "tool_implementations" in {
            component.value for component in caught.value.changed_component_classes
        }
        return provider

    replacement_provider = asyncio.run(exercise())

    assert replacement_provider.requests == []


def test_command_policy_decision_names_remain_distinct_from_durable_approval() -> None:
    assert CommandPolicyDecision.REQUIRE_COMMAND_APPROVAL.value == "require_command_approval"
