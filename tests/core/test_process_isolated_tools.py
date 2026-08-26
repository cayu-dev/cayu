from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from cayu._validation import DurableValueError, copy_bounded_durable_json_value
from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    ToolResultPart,
)
from cayu.core.isolated_tools import (
    ProcessIsolatedTool,
    ProcessIsolatedToolContext,
    ProcessIsolatedToolContextProjection,
    ProcessIsolatedToolFactoryRef,
    ProcessIsolatedToolLimits,
)
from cayu.core.tools import (
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
    _bind_runtime_tool_invocation_authority,
)
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runtime import (
    AlwaysRequireApprovalToolPolicy,
    CayuApp,
    InMemorySessionStore,
    InMemoryTaskStore,
    InterruptSessionRequest,
    PublicAuthorityAliasKeyring,
    ResumeRequest,
    RunRequest,
    SessionStatus,
    Task,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    ToolApprovalDecision,
    ToolApprovalRequest,
    ToolExecutionContract,
    run_task_worker,
)
from cayu.runtime import _isolated_tool_process as isolated_process
from cayu.runtime import _tool_execution as tool_execution
from cayu.runtime._isolated_tool_process import (
    IsolatedToolDeadlineExceeded,
    IsolatedToolFailure,
    IsolatedToolInvalidOutput,
    IsolatedToolPreDispatchFailure,
    execute_process_isolated_tool,
    isolated_tool_execution_contract,
    validate_process_isolated_tool_registration,
)
from cayu.runtime._isolated_tool_protocol import (
    IsolatedToolProtocolError,
    IsolatedToolTerminalEnvelope,
    build_isolated_tool_request,
    decode_isolated_tool_request,
    decode_isolated_tool_response,
    encode_isolated_tool_success,
)
from cayu.runtime.tool_policy import (
    StaticToolPolicy,
    ToolPolicy,
    ToolPolicyDecision,
    ToolPolicyRequest,
    ToolPolicyResult,
)
from cayu.server import ServerConfig, create_server
from cayu.vaults import SecretRedactor

_SCHEMA = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "additionalProperties": False,
}


def _factory_ref(
    *,
    implementation_version: str = "1",
) -> ProcessIsolatedToolFactoryRef:
    return ProcessIsolatedToolFactoryRef(
        module="cayu.testing_isolated_tools",
        qualname="build_deterministic_isolated_tool",
        identity=ExecutionProfileBehaviorIdentity(
            name="cayu:testing:deterministic-isolated-tool",
            behavior_version="1",
            implementation_version=implementation_version,
        ),
    )


def _tool(
    *,
    mode: str = "success",
    deadline_seconds: float = 5.0,
    factory_config: dict[str, Any] | None = None,
    max_request_bytes: int = 1 << 20,
    max_response_bytes: int = 1 << 20,
    max_stdout_bytes: int = 64 << 10,
    max_stderr_bytes: int = 64 << 10,
    effect: ToolEffect = ToolEffect.NONE,
    factory: ProcessIsolatedToolFactoryRef | None = None,
) -> ProcessIsolatedTool:
    config = {"mode": mode, **({} if factory_config is None else factory_config)}
    return ProcessIsolatedTool(
        ToolSpec(
            name="isolated_fixture",
            description="Exercise the real isolated process boundary.",
            input_schema=_SCHEMA,
            effect=effect,
            execution_profile_identity=ExecutionProfileBehaviorIdentity(
                name="cayu:testing:isolated-fixture",
                behavior_version="1",
                implementation_version="1",
            ),
        ),
        factory=_factory_ref() if factory is None else factory,
        limits=ProcessIsolatedToolLimits(
            deadline_seconds=deadline_seconds,
            term_grace_seconds=0.1,
            kill_grace_seconds=0.5,
            max_request_bytes=max_request_bytes,
            max_response_bytes=max_response_bytes,
            max_stdout_bytes=max_stdout_bytes,
            max_stderr_bytes=max_stderr_bytes,
        ),
        factory_config=config,
        context_projection=ProcessIsolatedToolContextProjection(fields=("session_id",)),
        environment={"CAYU_TEST_MARKER": "declared"},
    )


async def _load_operation(_storage_key: str) -> dict[str, Any] | None:
    return None


async def _compare_and_set_operation(
    _storage_key: str,
    _expected: dict[str, Any] | None,
    desired: dict[str, Any],
    _secondary: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    return desired


def _context(arguments: dict[str, Any]) -> ToolContext:
    context = ToolContext(
        session_id="isolated-session",
        idempotency_key="tool-execution-1",
        metadata={"unprojected": "must-not-cross"},
    )
    _bind_runtime_tool_invocation_authority(
        context,
        parent_task_id="task-1",
        parent_run_epoch=3,
        model_step_id="model-step-1",
        model_attempt_id="model-attempt-1",
        tool_round_id="tool-round-1",
        tool_call_id="tool-call-1",
        tool_name="isolated_fixture",
        idempotency_key="tool-execution-1",
        effective_arguments=arguments,
        execution_profile_fingerprint="e" * 64,
        environment_allocation_fingerprint="a" * 64,
        load_durable_operation=_load_operation,
        compare_and_set_durable_operation=_compare_and_set_operation,
        seal_durable_output=lambda value: dict(value),
        secret_publication_sealer=lambda: None,
    )
    return context


async def _execute(
    tool: ProcessIsolatedTool,
    arguments: dict[str, Any] | None = None,
) -> ToolResult:
    effective_arguments = {"text": "hello"} if arguments is None else arguments
    return await execute_process_isolated_tool(
        tool=tool,
        context=_context(effective_arguments),
        arguments=effective_arguments,
        registered_schema=_SCHEMA,
        redactor=SecretRedactor(),
    )


def test_isolated_tool_contract_is_explicit_callable_free_and_not_a_sandbox() -> None:
    caller_config = {"nested": {"value": "original"}}
    tool = _tool(factory_config=caller_config)
    caller_config["nested"]["value"] = "mutated"

    contract = isolated_tool_execution_contract(tool)

    assert contract == {
        "boundary": "posix_process",
        "timeout_strength": "hard_process_deadline",
        "sandboxed": False,
        "adapter_identity": {
            "name": "cayu:testing:deterministic-isolated-tool",
            "behavior_version": "1",
            "implementation_version": "1",
        },
        "adapter_configuration_sha256": contract["adapter_configuration_sha256"],
        "hard_deadline_seconds": 5.0,
        "protocol": "cayu.isolated-tool",
        "protocol_version": 1,
    }
    assert contract["adapter_configuration_sha256"].startswith("sha256:")
    assert "factory_config" not in contract
    assert "environment" not in contract
    assert tool.factory_config_copy()["nested"] == {"value": "original"}
    assert (
        isolated_tool_execution_contract(
            tool,
            runtime_timeout_seconds=1,
        )["hard_deadline_seconds"]
        == 1
    )

    with pytest.raises(RuntimeError, match="runtime-owned"):
        asyncio.run(tool.run(ToolContext(session_id="direct"), {}))

    with pytest.raises(ValueError, match="complete hard-deadline evidence"):
        ToolExecutionContract(
            boundary="posix_process",
            timeout_strength="hard_process_deadline",
            adapter_identity=_factory_ref().identity,
            adapter_configuration_sha256="sha256:" + "a" * 64,
            hard_deadline_seconds=1,
            protocol="foreign-protocol",
            protocol_version=1,
        )


def test_isolated_tool_rejects_nonportable_or_interpreter_affecting_configuration() -> None:
    class MappingInput(dict[str, Any]):
        pass

    with pytest.raises(TypeError, match="factory_config"):
        ProcessIsolatedTool(
            ToolSpec(name="invalid", input_schema={"type": "object"}),
            factory=_factory_ref(),
            limits=ProcessIsolatedToolLimits(deadline_seconds=1),
            factory_config=MappingInput(),
        )
    interpreter_secret = "PYTHON_SECRET_CANARY"
    with pytest.raises(ValueError, match="child interpreter") as environment_error:
        ProcessIsolatedTool(
            ToolSpec(name="invalid", input_schema={"type": "object"}),
            factory=_factory_ref(),
            limits=ProcessIsolatedToolLimits(deadline_seconds=1),
            environment={interpreter_secret: "/caller/path"},
        )
    assert interpreter_secret not in str(environment_error.value)
    with pytest.raises(DurableValueError, match="JSON-compatible"):
        ProcessIsolatedTool(
            ToolSpec(name="invalid", input_schema={"type": "object"}),
            factory=_factory_ref(),
            limits=ProcessIsolatedToolLimits(deadline_seconds=1),
            factory_config={"live_object": object()},
        )
    with pytest.raises(TypeError, match="list or tuple"):
        ProcessIsolatedToolContextProjection(fields=(item for item in ["session_id"]))
    with pytest.raises(ValueError, match="dotted Python identifier"):
        ProcessIsolatedToolFactoryRef(
            module="my_app.handlers",
            qualname="factory.<locals>.handler",
            identity=_factory_ref().identity,
        )


def test_registration_rejects_secrets_and_unsupported_platforms_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_tool = ProcessIsolatedTool(
        _tool().spec,
        factory=_factory_ref(),
        limits=ProcessIsolatedToolLimits(deadline_seconds=1),
        factory_config={"endpoint": "registered-secret"},
    )
    with pytest.raises(ValueError, match="cannot contain registered secrets"):
        validate_process_isolated_tool_registration(
            secret_tool,
            redactor=SecretRedactor("registered-secret"),
        )

    monkeypatch.setattr(isolated_process.os, "name", "nt")
    with pytest.raises(RuntimeError, match="POSIX"):
        validate_process_isolated_tool_registration(
            _tool(),
            redactor=SecretRedactor(),
        )


def test_registration_revalidates_isolated_tool_workspace_authority() -> None:
    tool = _tool()
    tool.spec = ToolSpec(
        name=tool.spec.name,
        description=tool.spec.description,
        input_schema=tool.spec.input_schema,
        parallel_safe=False,
        effect=ToolEffect.EXTERNAL,
        workspace_mutation=True,
        execution_profile_identity=tool.spec.execution_profile_identity,
    )
    app = CayuApp(enable_logging=False)

    with pytest.raises(ValueError, match="workspace mutation authority"):
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"),
            tools=[tool],
        )


def test_manifest_distinguishes_unbounded_and_cooperative_ordinary_tools() -> None:
    class OrdinaryTool(Tool):
        def __init__(self) -> None:
            super().__init__(
                ToolSpec(
                    name="ordinary",
                    input_schema={"type": "object"},
                    effect=ToolEffect.NONE,
                    execution_profile_identity=ExecutionProfileBehaviorIdentity(
                        name="tests:ordinary-tool",
                        behavior_version="1",
                        implementation_version="1",
                    ),
                )
            )

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            return ToolResult(content="ordinary")

    observed = []
    for timeout in (None, 1.0):
        app = CayuApp(enable_logging=False, tool_timeout_seconds=timeout)
        app.register_agent(
            AgentSpec(name="assistant", model="test-model"),
            tools=[OrdinaryTool()],
        )
        tool_manifest = app.describe().agents[0].tools[0]
        descriptor = app._agents["assistant"].tool_catalogue.descriptors[0]
        capability = app._agents["assistant"].tool_capabilities[0]
        observed.append(
            (
                tool_manifest.timeout_strength,
                descriptor.execution_contract.timeout_strength,
                capability.execution_contract.timeout_strength,
            )
        )
        assert tool_manifest.execution_boundary == "in_process"
        assert tool_manifest.sandboxed is False

        inconsistent_manifest = tool_manifest.model_dump(mode="python")
        inconsistent_manifest.update(
            {
                "execution_boundary": "posix_process",
                "timeout_strength": "hard_process_deadline",
            }
        )
        with pytest.raises(ValueError, match="manifest evidence is incomplete"):
            type(tool_manifest).model_validate(inconsistent_manifest)

        malformed_identity_manifest = dict(inconsistent_manifest)
        malformed_identity_manifest.update(
            {
                "adapter_identity": {},
                "adapter_configuration_sha256": "sha256:" + "a" * 64,
                "hard_deadline_seconds": 1,
            }
        )
        with pytest.raises(ValueError, match="adapter identity is invalid"):
            type(tool_manifest).model_validate(malformed_identity_manifest)

    assert observed == [
        ("none", "none", "none"),
        (
            "cooperative_in_process",
            "cooperative_in_process",
            "cooperative_in_process",
        ),
    ]


def test_bounded_json_preflight_rejects_depth_cycles_nodes_and_bytes_before_copy() -> None:
    deep: Any = "leaf"
    for _ in range(70):
        deep = [deep]
    with pytest.raises(DurableValueError) as depth_error:
        copy_bounded_durable_json_value(
            deep,
            "payload",
            max_bytes=10_000,
            max_nodes=1_000,
            max_nesting=64,
        )
    assert depth_error.value.code == "nesting_too_deep"

    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(DurableValueError) as cycle_error:
        copy_bounded_durable_json_value(
            cyclic,
            "payload",
            max_bytes=10_000,
            max_nodes=1_000,
        )
    assert cycle_error.value.code == "circular_reference"

    with pytest.raises(DurableValueError) as node_error:
        copy_bounded_durable_json_value(
            [None, None],
            "payload",
            max_bytes=10,
            max_nodes=2,
        )
    assert node_error.value.code == "too_many_json_nodes"

    assert copy_bounded_durable_json_value(
        {"a": "b"},
        "payload",
        max_bytes=9,
        max_nodes=3,
    ) == {"a": "b"}
    with pytest.raises(DurableValueError) as size_error:
        copy_bounded_durable_json_value(
            {"a": "b"},
            "payload",
            max_bytes=8,
            max_nodes=3,
        )
    assert size_error.value.code == "json_value_too_large"


def test_protocol_rejects_noncanonical_trailing_and_identity_mismatched_frames() -> None:
    tool = _tool()
    envelope, encoded = build_isolated_tool_request(
        session_id="session",
        authority={
            "parent_task_id": None,
            "parent_run_epoch": 0,
            "model_step_id": "step",
            "model_attempt_id": "attempt",
            "tool_round_id": "round",
            "tool_call_id": "call",
            "tool_name": "isolated_fixture",
            "idempotency_key": "key",
            "effective_arguments_sha256": "a" * 64,
            "execution_profile_fingerprint": "b" * 64,
            "environment_allocation_fingerprint": None,
        },
        factory=tool.factory,
        limits=tool.limits,
        factory_config=tool.factory_config_copy(),
        arguments={"text": "hello"},
        context=ProcessIsolatedToolContext(),
        environment=tool.environment_copy(),
    )
    assert decode_isolated_tool_request(encoded) == envelope
    with pytest.raises(IsolatedToolProtocolError):
        decode_isolated_tool_request(encoded + b"\n")
    tampered_request = encoded.replace(
        b'"tool_name":"isolated_fixture"', b'"tool_name":"foreign_fixture"'
    )
    with pytest.raises(IsolatedToolProtocolError):
        decode_isolated_tool_request(tampered_request)

    response = encode_isolated_tool_success(
        request_sha256=envelope.request_sha256,
        result=ToolResult(content="ok"),
        max_bytes=tool.limits.max_response_bytes,
    )
    assert decode_isolated_tool_response(
        response,
        expected_request_sha256=envelope.request_sha256,
        max_bytes=tool.limits.max_response_bytes,
    ) == ToolResult(content="ok")
    with pytest.raises(IsolatedToolProtocolError):
        decode_isolated_tool_response(
            response,
            expected_request_sha256="sha256:" + "0" * 64,
            max_bytes=tool.limits.max_response_bytes,
        )
    with pytest.raises(IsolatedToolProtocolError):
        decode_isolated_tool_response(
            response + response,
            expected_request_sha256=envelope.request_sha256,
            max_bytes=tool.limits.max_response_bytes * 2,
        )
    with pytest.raises(IsolatedToolProtocolError):
        decode_isolated_tool_response(
            b"\xff",
            expected_request_sha256=envelope.request_sha256,
            max_bytes=tool.limits.max_response_bytes,
        )


def test_deep_child_response_is_rejected_before_recursive_model_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deep: Any = "leaf"
    for _ in range(130):
        deep = [deep]
    encoded = json.dumps(
        {
            "protocol": "cayu.isolated-tool",
            "version": 1,
            "request_sha256": "sha256:" + "a" * 64,
            "status": "ok",
            "result": {
                "content": "untrusted",
                "structured": deep,
                "artifacts": [],
                "is_error": False,
            },
            "error_code": None,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    reconstruction_called = False

    def forbidden_reconstruction(*_args, **_kwargs):
        nonlocal reconstruction_called
        reconstruction_called = True
        raise AssertionError("recursive reconstruction must follow bounded preflight")

    monkeypatch.setattr(
        IsolatedToolTerminalEnvelope,
        "model_validate",
        forbidden_reconstruction,
    )

    with pytest.raises(IsolatedToolProtocolError) as caught:
        decode_isolated_tool_response(
            encoded,
            expected_request_sha256="sha256:" + "a" * 64,
            max_bytes=1 << 20,
        )

    assert caught.value.code == "response_invalid"
    assert reconstruction_called is False


@pytest.mark.process
def test_real_process_success_uses_only_projected_context_and_declared_environment() -> None:
    result = asyncio.run(_execute(_tool()))

    assert result.content == "hello"
    assert result.structured == {
        "session_id": "isolated-session",
        "environment_marker": "declared",
    }

    environment_result = asyncio.run(_execute(_tool(mode="environment")))
    assert environment_result.structured == {
        "environment": {
            "CAYU_TEST_MARKER": "declared",
            "LC_ALL": "C",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    }

    descriptor_result = asyncio.run(_execute(_tool(mode="file_descriptors")))
    opened = descriptor_result.structured["file_descriptors"]
    assert opened[:3] == [0, 1, 2]
    assert len(opened) == 4


@pytest.mark.process
@pytest.mark.parametrize(
    ("mode", "expected_exception", "expected_code"),
    [
        ("exception", IsolatedToolFailure, "child_child_exception"),
        ("invalid_result", IsolatedToolFailure, "child_invalid_result"),
        ("crash", IsolatedToolFailure, "child_exited_without_terminal_output"),
        ("signal", IsolatedToolFailure, "child_signaled"),
    ],
)
def test_real_process_child_failures_are_typed_and_bounded(
    mode: str,
    expected_exception: type[BaseException],
    expected_code: str,
) -> None:
    with pytest.raises(expected_exception) as caught:
        asyncio.run(_execute(_tool(mode=mode)))

    assert getattr(caught.value, "code", None) == expected_code
    assert "isolated failure" not in str(caught.value)


@pytest.mark.process
def test_real_child_rejects_a_loaded_factory_with_conflicting_identity() -> None:
    with pytest.raises(IsolatedToolFailure) as caught:
        asyncio.run(
            _execute(
                _tool(
                    factory=_factory_ref(implementation_version="different"),
                )
            )
        )

    assert caught.value.code == "child_factory_identity_mismatch"


@pytest.mark.process
@pytest.mark.parametrize(
    ("mode", "limit_name", "expected_code"),
    [
        ("stdout", "max_stdout_bytes", "stdout_too_large"),
        ("stderr", "max_stderr_bytes", "stderr_too_large"),
    ],
)
def test_real_process_diagnostic_overflow_is_bounded(
    mode: str,
    limit_name: str,
    expected_code: str,
) -> None:
    with pytest.raises(IsolatedToolFailure) as caught:
        asyncio.run(
            _execute(
                _tool(
                    mode=mode,
                    deadline_seconds=3,
                    factory_config={"bytes": 4096},
                    **{limit_name: 128},
                )
            )
        )

    assert caught.value.code == expected_code


@pytest.mark.process
def test_terminal_response_cannot_race_past_buffered_stdout_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        isolated_process,
        "_WORKER_MODULE",
        "cayu.testing_isolated_worker_faults",
    )

    with pytest.raises(IsolatedToolFailure) as caught:
        asyncio.run(
            _execute(
                _tool(
                    mode="terminal_then_stdout_overflow",
                    deadline_seconds=3,
                    max_stdout_bytes=128,
                )
            )
        )

    assert caught.value.code == "stdout_too_large"


@pytest.mark.process
@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [
        ("malformed_wire", "response_invalid"),
        ("multiple_wire", "response_invalid"),
        ("oversized_wire", "response_too_large"),
    ],
)
def test_real_malformed_child_protocol_is_typed_and_bounded(
    mode: str,
    expected_code: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        isolated_process,
        "_WORKER_MODULE",
        "cayu.testing_isolated_worker_faults",
    )

    with pytest.raises(IsolatedToolInvalidOutput) as caught:
        asyncio.run(
            _execute(
                _tool(
                    mode=mode,
                    deadline_seconds=3,
                    max_response_bytes=1024,
                )
            )
        )

    assert caught.value.code == expected_code


@pytest.mark.process
def test_malformed_child_output_is_detached_from_every_diagnostic_channel(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "ISOLATED_WIRE_SECRET_CANARY"
    monkeypatch.setattr(
        isolated_process,
        "_WORKER_MODULE",
        "cayu.testing_isolated_worker_faults",
    )
    tool = _tool(mode="secret_invalid_wire", deadline_seconds=3)
    arguments = {"text": "hello"}

    with (
        warnings.catch_warnings(record=True) as captured_warnings,
        pytest.raises(IsolatedToolInvalidOutput) as caught,
    ):
        asyncio.run(
            execute_process_isolated_tool(
                tool=tool,
                context=_context(arguments),
                arguments=arguments,
                registered_schema=_SCHEMA,
                redactor=SecretRedactor(canary),
            )
        )

    captured_output = capsys.readouterr()
    diagnostic_values = [
        str(caught.value),
        repr(caught.value),
        caplog.text,
        captured_output.out,
        captured_output.err,
        *(str(item.message) for item in captured_warnings),
    ]
    current: BaseException | None = caught.value
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        diagnostic_values.extend((str(current), repr(current), repr(current.args)))
        traceback = current.__traceback__
        while traceback is not None:
            if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
                diagnostic_values.extend(
                    repr(value) for value in traceback.tb_frame.f_locals.values()
                )
            traceback = traceback.tb_next
        current = current.__cause__ or current.__context__

    assert canary not in "\n".join(diagnostic_values)


@pytest.mark.process
def test_real_gil_holding_child_is_killed_by_the_hard_wall_deadline(tmp_path: Path) -> None:
    async def scenario(started_path: Path) -> None:
        await _execute(
            _tool(
                mode="gil_block",
                deadline_seconds=2.5,
                factory_config={"seconds": 30, "started_path": str(started_path)},
            )
        )

    started_path = tmp_path / "started"
    started = time.monotonic()
    try:
        with pytest.raises(IsolatedToolDeadlineExceeded):
            asyncio.run(scenario(started_path))
        assert started_path.read_text(encoding="utf-8") == "started"
    finally:
        started_path.unlink(missing_ok=True)

    assert time.monotonic() - started < 4
    assert asyncio.run(_execute(_tool())).content == "hello"


@pytest.mark.process
def test_real_caller_cancellation_kills_the_child_and_remains_cancellation(
    tmp_path: Path,
) -> None:
    async def scenario(started_path: Path) -> tuple[int, bool]:
        task = asyncio.create_task(
            _execute(
                _tool(
                    mode="ignore_term",
                    deadline_seconds=10,
                    factory_config={"seconds": 30, "started_path": str(started_path)},
                )
            )
        )
        async with asyncio.timeout(5):
            while not started_path.exists():
                await asyncio.sleep(0.01)
        task.cancel("caller-stop")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError, match="caller-stop"):
            await task
        return cancelling, task.cancelled()

    cancelling, cancelled = asyncio.run(scenario(tmp_path / "started"))

    assert cancelling == 1
    assert cancelled is True


def test_ordinary_tool_cannot_attach_a_secret_cause_to_caller_cancellation(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    canary = "TOOL_CANCELLATION_CAUSE_SECRET_CANARY"

    class CauseReplacingTool(Tool):
        def __init__(self, started: asyncio.Event) -> None:
            super().__init__(ToolSpec(name="cause_replacing", input_schema={"type": "object"}))
            self.started = started

        async def run(self, ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
            del ctx, args
            self.started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                raise asyncio.CancelledError("extension replacement") from RuntimeError(canary)

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        started = asyncio.Event()
        task = asyncio.create_task(
            tool_execution.run_tool(
                tool=CauseReplacingTool(started),
                effect=ToolEffect.EXTERNAL,
                ctx=ToolContext(session_id="cancellation-cause"),
                arguments={},
                redactor=lambda: SecretRedactor(canary),
                timeout_seconds=30,
            )
        )
        await started.wait()
        task.cancel("caller cancellation")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        return caught.value, cancelling, task.cancelled()

    with warnings.catch_warnings(record=True) as captured_warnings:
        cancellation, cancelling, cancelled = asyncio.run(scenario())

    captured_output = capsys.readouterr()
    diagnostics = [
        str(cancellation),
        repr(cancellation),
        caplog.text,
        captured_output.out,
        captured_output.err,
        *(str(item.message) for item in captured_warnings),
    ]
    current: BaseException | None = cancellation
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        diagnostics.extend((str(current), repr(current), repr(current.args)))
        traceback = current.__traceback__
        while traceback is not None:
            if "/src/cayu/" in traceback.tb_frame.f_code.co_filename:
                diagnostics.extend(repr(value) for value in traceback.tb_frame.f_locals.values())
            traceback = traceback.tb_next
        current = current.__cause__ or current.__context__

    assert cancelling == 1
    assert cancelled is True
    assert cancellation.__cause__ is None
    assert canary not in "\n".join(diagnostics)


@pytest.mark.process
def test_caller_cancellation_during_cleanup_retains_the_preceding_child_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_wait = isolated_process._wait_for_group_and_child

    async def scenario() -> tuple[asyncio.CancelledError, int, bool]:
        cleanup_entered = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def blocked_wait(*args, **kwargs):
            cleanup_entered.set()
            await release_cleanup.wait()
            return await original_wait(*args, **kwargs)

        monkeypatch.setattr(isolated_process, "_wait_for_group_and_child", blocked_wait)
        task = asyncio.create_task(_execute(_tool(mode="crash", deadline_seconds=5)))
        await asyncio.wait_for(cleanup_entered.wait(), timeout=5)
        task.cancel("cancel during process cleanup")
        cancelling = task.cancelling()
        release_cleanup.set()
        with pytest.raises(
            asyncio.CancelledError,
            match="cancel during process cleanup",
        ) as raised:
            await task
        return raised.value, cancelling, task.cancelled()

    cancellation, cancelling, cancelled = asyncio.run(scenario())

    assert cancelling == 1
    assert cancelled is True
    assert isinstance(cancellation.__cause__, IsolatedToolFailure)
    assert cancellation.__cause__.code == "child_exited_without_terminal_output"


def test_parent_process_control_remains_authoritative_when_cleanup_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = isolated_process._IsolatedToolProcessOwner(
        tool=_tool(),
        request_bytes=b"{}",
        request_sha256="sha256:" + "a" * 64,
    )

    async def spawn(*, deadline: float) -> None:
        del deadline

    async def exchange(*, deadline: float):
        del deadline
        raise SystemExit("operator shutdown")

    async def settle(*, cancellation=None):
        del cancellation
        raise isolated_process.IsolatedToolCleanupUnproven("process_cleanup_unproven")

    monkeypatch.setattr(owner, "_spawn", spawn)
    monkeypatch.setattr(owner, "_exchange", exchange)
    monkeypatch.setattr(owner, "_settle", settle)

    with pytest.raises(SystemExit, match="operator shutdown") as caught:
        asyncio.run(owner.run())

    assert isinstance(caught.value.__cause__, isolated_process.IsolatedToolCleanupUnproven)
    assert caught.value.__cause__.code == "process_cleanup_unproven"


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cleanup_process_control_remains_scalar_after_execution_failure(
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
) -> None:
    async def scenario() -> None:
        owner = isolated_process._IsolatedToolProcessOwner(
            tool=_tool(),
            request_bytes=b"{}",
            request_sha256="sha256:" + "a" * 64,
        )
        primary = IsolatedToolInvalidOutput("response_invalid")
        cleanup_evidence = RuntimeError("cleanup interrupted")

        async def spawn(*, deadline: float) -> None:
            del deadline

        async def exchange(*, deadline: float):
            del deadline
            raise primary

        async def settle(*, cancellation=None):
            del cancellation
            raise signal_type("supervisor signal") from cleanup_evidence

        monkeypatch.setattr(owner, "_spawn", spawn)
        monkeypatch.setattr(owner, "_exchange", exchange)
        monkeypatch.setattr(owner, "_settle", settle)

        with pytest.raises(signal_type, match="supervisor signal") as caught:
            await owner.run()

        evidence = caught.value.__cause__
        assert isinstance(evidence, BaseExceptionGroup)
        assert evidence.exceptions == (primary, cleanup_evidence)

    asyncio.run(scenario())


def test_cleanup_process_control_preserves_existing_primary_evidence_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        owner = isolated_process._IsolatedToolProcessOwner(
            tool=_tool(),
            request_bytes=b"{}",
            request_sha256="sha256:" + "a" * 64,
        )
        primary = IsolatedToolInvalidOutput("response_invalid")
        secondary = RuntimeError("cleanup interrupted")
        prior_evidence = ExceptionGroup("existing cleanup evidence", [primary, secondary])

        async def spawn(*, deadline: float) -> None:
            del deadline

        async def exchange(*, deadline: float):
            del deadline
            raise primary

        async def settle(*, cancellation=None):
            del cancellation
            raise SystemExit("supervisor signal") from prior_evidence

        monkeypatch.setattr(owner, "_spawn", spawn)
        monkeypatch.setattr(owner, "_exchange", exchange)
        monkeypatch.setattr(owner, "_settle", settle)

        with pytest.raises(SystemExit, match="supervisor signal") as caught:
            await owner.run()

        assert caught.value.__cause__ is prior_evidence
        assert prior_evidence.exceptions == (primary, secondary)

    asyncio.run(scenario())


@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_process_control_during_settlement_retains_the_live_cleanup_owner(
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
) -> None:
    async def scenario() -> None:
        owner = isolated_process._IsolatedToolProcessOwner(
            tool=_tool(),
            request_bytes=b"{}",
            request_sha256="sha256:" + "a" * 64,
        )
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def blocked_cleanup():
            cleanup_started.set()
            await release_cleanup.wait()
            return None

        cleanup_task = asyncio.create_task(blocked_cleanup())
        owner._cleanup_task = cleanup_task

        async def deliver_process_control(task, **_kwargs):
            assert task is cleanup_task
            await cleanup_started.wait()
            raise signal_type("supervisor signal")

        monkeypatch.setattr(
            isolated_process,
            "await_shielded_task_outcome",
            deliver_process_control,
        )
        try:
            with pytest.raises(signal_type, match="supervisor signal"):
                await owner._settle()
            assert cleanup_task in isolated_process._RETAINED_ISOLATED_TOOL_TASKS
            assert cleanup_task.done() is False
        finally:
            release_cleanup.set()
            await cleanup_task
            await asyncio.sleep(0)
        assert cleanup_task not in isolated_process._RETAINED_ISOLATED_TOOL_TASKS

    asyncio.run(scenario())


def test_late_spawn_owns_passed_descriptor_until_spawn_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        spawn_entered = asyncio.Event()
        release_spawn = asyncio.Event()
        passed_descriptor: int | None = None
        retained: list[asyncio.Task[Any]] = []

        async def delayed_spawn(*_args, **kwargs):
            nonlocal passed_descriptor
            passed_descriptor = kwargs["pass_fds"][0]
            spawn_entered.set()
            await release_spawn.wait()
            os.fstat(passed_descriptor)
            raise OSError("deterministic late spawn failure")

        def retain(task: asyncio.Task[Any], **_kwargs: Any) -> None:
            retained.append(task)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
        monkeypatch.setattr(isolated_process, "_retain_task", retain)
        owner = isolated_process._IsolatedToolProcessOwner(
            tool=_tool(deadline_seconds=0.01),
            request_bytes=b"{}",
            request_sha256="sha256:" + "a" * 64,
        )

        with pytest.raises(IsolatedToolDeadlineExceeded):
            await owner.run()
        assert spawn_entered.is_set()
        assert passed_descriptor is not None
        os.fstat(passed_descriptor)
        assert len(retained) == 1

        release_spawn.set()
        await retained[0]
        with pytest.raises(OSError):
            os.fstat(passed_descriptor)

    asyncio.run(scenario())


@pytest.mark.process
@pytest.mark.parametrize("signal_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("spawn_phase", ["pending", "completed"])
def test_process_control_during_spawn_retains_and_settles_the_late_child(
    monkeypatch: pytest.MonkeyPatch,
    signal_type: type[BaseException],
    spawn_phase: str,
) -> None:
    async def scenario() -> int:
        original_spawn = asyncio.create_subprocess_exec
        original_wait = isolated_process.await_shielded_task_outcome
        process_ready = asyncio.Event()
        release_spawn = asyncio.Event()
        process_id: int | None = None
        signal_delivered = False

        async def delayed_spawn(*args, **kwargs):
            nonlocal process_id
            process = await original_spawn(*args, **kwargs)
            process_id = process.pid
            process_ready.set()
            if spawn_phase == "pending":
                await release_spawn.wait()
            return process

        async def deliver_process_control(task, **kwargs):
            nonlocal signal_delivered
            if task.get_name() == "cayu-isolated-tool-spawn" and not signal_delivered:
                await process_ready.wait()
                signal_delivered = True
                if spawn_phase == "completed":
                    await original_wait(task, **kwargs)
                else:
                    release_spawn.set()
                raise signal_type("supervisor signal during spawn")
            return await original_wait(task, **kwargs)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", delayed_spawn)
        monkeypatch.setattr(
            isolated_process,
            "await_shielded_task_outcome",
            deliver_process_control,
        )
        owner = isolated_process._IsolatedToolProcessOwner(
            tool=_tool(deadline_seconds=5),
            request_bytes=b"{}",
            request_sha256="sha256:" + "a" * 64,
        )

        with pytest.raises(signal_type, match="supervisor signal during spawn"):
            await owner.run()

        assert process_id is not None
        assert owner._spawn_task is None
        assert owner._process is None
        assert owner._wait_task is None
        await asyncio.sleep(0)
        assert not isolated_process._RETAINED_ISOLATED_TOOL_TASKS
        return process_id

    process_id = asyncio.run(scenario())
    _assert_process_gone(process_id)


@pytest.mark.process
def test_unproven_cleanup_retains_ownership_and_publishes_bounded_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_attempts = 0

    async def never_proven(*_args, **_kwargs) -> bool:
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        return False

    monkeypatch.setattr(isolated_process, "_wait_for_group_and_child", never_proven)

    async def scenario():
        arguments = {"text": "hello"}
        tool = _tool(mode="exception", deadline_seconds=5)
        return await tool_execution.run_tool(
            tool=tool,
            effect=ToolEffect.EXTERNAL,
            ctx=_context(arguments),
            arguments=arguments,
            registered_schema=_SCHEMA,
            registered_execution_contract=isolated_tool_execution_contract(tool),
            redactor=SecretRedactor,
        )

    try:
        outcome = asyncio.run(scenario())

        assert cleanup_attempts >= 2
        assert outcome.result.is_error is True
        assert outcome.result.structured["terminal_outcome"] == "tool_execution_error"
        assert outcome.result.structured["outcome_unknown"] is True
        assert outcome.result.structured["isolated_tool_failure_code"] == (
            "process_cleanup_unproven"
        )
        assert outcome.result.structured["isolated_tool_cleanup_failure_code"] == (
            "process_cleanup_deadline_exceeded"
        )
        assert isolated_process._retained_isolated_tool_cleanup_pending() is True
    finally:
        isolated_process._RETAINED_ISOLATED_TOOL_TASKS.clear()
        isolated_process._RETAINED_ISOLATED_TOOL_RETRIES.clear()


@pytest.mark.parametrize("termination", ["cancelled", "failed"])
def test_abnormal_retained_cleanup_keeps_later_isolated_dispatch_fenced(
    monkeypatch: pytest.MonkeyPatch,
    termination: str,
) -> None:
    retry_loops: list[int] = []

    async def retry_cleanup() -> None:
        retry_loops.append(id(asyncio.get_running_loop()))
        await asyncio.sleep(0)

    async def establish_failed_owner() -> None:
        started = asyncio.Event()

        async def retained_cleanup() -> None:
            started.set()
            if termination == "failed":
                raise isolated_process.IsolatedToolCleanupUnproven("retained_cleanup_failed")
            await asyncio.Future()

        cleanup_task = asyncio.create_task(retained_cleanup())
        isolated_process._retain_task(cleanup_task, retry_factory=retry_cleanup)
        await started.wait()
        if termination == "cancelled":
            cleanup_task.cancel("retained owner cancelled")
        with pytest.raises(
            asyncio.CancelledError
            if termination == "cancelled"
            else isolated_process.IsolatedToolCleanupUnproven
        ):
            await cleanup_task
        await asyncio.sleep(0)

    async def unexpected_spawn(*_args, **_kwargs):
        raise AssertionError("A child was spawned while retained cleanup was unresolved.")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_spawn)

    async def reject_and_reconcile() -> None:
        arguments = {"text": "later"}
        tool = _tool()
        outcome = await tool_execution.run_tool(
            tool=tool,
            effect=ToolEffect.NONE,
            ctx=_context(arguments),
            arguments=arguments,
            registered_schema=_SCHEMA,
            registered_execution_contract=isolated_tool_execution_contract(tool),
            redactor=SecretRedactor,
        )

        assert isolated_process._retained_isolated_tool_cleanup_pending() is True
        assert outcome.result.structured["isolated_tool_failure_code"] == (
            "prior_process_cleanup_pending"
        )
        async with asyncio.timeout(1):
            while isolated_process._retained_isolated_tool_cleanup_pending():
                await asyncio.sleep(0)

    try:
        asyncio.run(establish_failed_owner())
        assert isolated_process._retained_isolated_tool_cleanup_pending() is True
        recovery_loop = asyncio.new_event_loop()
        try:
            recovery_loop.run_until_complete(reject_and_reconcile())
            assert retry_loops == [id(recovery_loop)]
        finally:
            recovery_loop.close()
    finally:
        isolated_process._RETAINED_ISOLATED_TOOL_TASKS.clear()
        isolated_process._RETAINED_ISOLATED_TOOL_RETRIES.clear()


@pytest.mark.process
def test_unproven_cleanup_fences_later_isolated_dispatch_until_owner_settles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        release_cleanup = asyncio.Event()
        cleanup_finished = asyncio.Event()
        original_settle_group = isolated_process._settle_owned_process_group
        original_spawn = isolated_process._IsolatedToolProcessOwner._spawn
        spawn_count = 0

        async def blocked_settle_group(*args, **kwargs):
            try:
                await release_cleanup.wait()
                await original_settle_group(*args, **kwargs)
            finally:
                cleanup_finished.set()

        async def counted_spawn(owner, *, deadline: float):
            nonlocal spawn_count
            spawn_count += 1
            await original_spawn(owner, deadline=deadline)

        monkeypatch.setattr(
            isolated_process,
            "_settle_owned_process_group",
            blocked_settle_group,
        )
        monkeypatch.setattr(
            isolated_process._IsolatedToolProcessOwner,
            "_spawn",
            counted_spawn,
        )
        monkeypatch.setattr(
            isolated_process,
            "_CLEANUP_SETTLEMENT_HEADROOM_SECONDS",
            0.01,
        )
        started_path = tmp_path / "unsettled-child-started"
        tool = _tool(
            mode="conditional_gil_block",
            deadline_seconds=2.5,
            factory_config={"seconds": 30, "started_path": str(started_path)},
            effect=ToolEffect.EXTERNAL,
        )

        async def invoke(text: str):
            arguments = {"text": text}
            return await tool_execution.run_tool(
                tool=tool,
                effect=ToolEffect.EXTERNAL,
                ctx=_context(arguments),
                arguments=arguments,
                registered_schema=_SCHEMA,
                registered_execution_contract=isolated_tool_execution_contract(tool),
                redactor=SecretRedactor,
            )

        try:
            first = await invoke("block")
            assert started_path.read_text(encoding="utf-8") == "started"
            assert first.result.structured["isolated_tool_failure_code"] == (
                "process_cleanup_unproven"
            )
            assert isolated_process._retained_isolated_tool_cleanup_pending() is True

            fenced = await invoke("later")
            assert fenced.result.structured["isolated_tool_failure_code"] == (
                "prior_process_cleanup_pending"
            )
            assert spawn_count == 1

            release_cleanup.set()
            await asyncio.wait_for(cleanup_finished.wait(), timeout=3)
            async with asyncio.timeout(3):
                while isolated_process._retained_isolated_tool_cleanup_pending():
                    await asyncio.sleep(0.01)

            successful = await invoke("later")
            assert successful.result.is_error is False
            assert successful.result.content == "later"
            assert spawn_count == 2
        finally:
            release_cleanup.set()
            pending = [
                task
                for task in tuple(isolated_process._RETAINED_ISOLATED_TOOL_TASKS)
                if not task.done()
            ]
            if pending:
                await asyncio.wait(pending, timeout=3)

    asyncio.run(scenario())


def test_cleanup_fence_appearing_during_dispatch_publication_blocks_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        publication_started = asyncio.Event()
        release_publication = asyncio.Event()
        release_cleanup = asyncio.Event()
        published_records: list[dict[str, Any]] = []

        async def delayed_compare_and_set(
            _storage_key: str,
            _expected: dict[str, Any] | None,
            desired: dict[str, Any],
            _secondary: Mapping[str, dict[str, Any]],
        ) -> dict[str, Any]:
            publication_started.set()
            await release_publication.wait()
            published_records.append(desired)
            return desired

        arguments = {"text": "later"}
        context = ToolContext(
            session_id="isolated-session",
            idempotency_key="tool-execution-1",
        )
        _bind_runtime_tool_invocation_authority(
            context,
            parent_task_id="task-1",
            parent_run_epoch=3,
            model_step_id="model-step-1",
            model_attempt_id="model-attempt-1",
            tool_round_id="tool-round-1",
            tool_call_id="tool-call-1",
            tool_name="isolated_fixture",
            idempotency_key="tool-execution-1",
            effective_arguments=arguments,
            execution_profile_fingerprint="e" * 64,
            environment_allocation_fingerprint="a" * 64,
            load_durable_operation=_load_operation,
            compare_and_set_durable_operation=delayed_compare_and_set,
            seal_durable_output=lambda value: dict(value),
            secret_publication_sealer=lambda: None,
        )

        async def retained_cleanup() -> None:
            await release_cleanup.wait()

        async def forbidden_spawn(*_args, **_kwargs):
            raise AssertionError("A child crossed a newly established cleanup fence.")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
        invocation = asyncio.create_task(
            execute_process_isolated_tool(
                tool=_tool(),
                context=context,
                arguments=arguments,
                registered_schema=_SCHEMA,
                redactor=SecretRedactor(),
            )
        )
        await publication_started.wait()
        cleanup_task = asyncio.create_task(retained_cleanup())
        isolated_process._retain_task(cleanup_task)
        release_publication.set()
        try:
            with pytest.raises(IsolatedToolPreDispatchFailure) as caught:
                await invocation
            assert caught.value.code == "prior_process_cleanup_pending"
            assert len(published_records) == 1
        finally:
            release_cleanup.set()
            await cleanup_task
            await asyncio.sleep(0)
        assert not isolated_process._RETAINED_ISOLATED_TOOL_TASKS

    asyncio.run(scenario())


@pytest.mark.process
def test_success_is_not_published_while_temporary_directory_cleanup_is_unproven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retained: list[asyncio.Task[Any]] = []

    async def scenario():
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()

        async def blocked_removal(_directory: str) -> None:
            cleanup_started.set()
            await release_cleanup.wait()

        def retain(task: asyncio.Task[Any], **_kwargs: Any) -> None:
            retained.append(task)

        monkeypatch.setattr(isolated_process, "_remove_temporary_directory", blocked_removal)
        monkeypatch.setattr(
            isolated_process,
            "_TEMPORARY_DIRECTORY_SETTLEMENT_SECONDS",
            0.01,
        )
        monkeypatch.setattr(isolated_process, "_retain_task", retain)
        arguments = {"text": "hello"}
        tool = _tool(effect=ToolEffect.EXTERNAL)
        outcome = await tool_execution.run_tool(
            tool=tool,
            effect=ToolEffect.EXTERNAL,
            ctx=_context(arguments),
            arguments=arguments,
            registered_schema=_SCHEMA,
            registered_execution_contract=isolated_tool_execution_contract(tool),
            redactor=SecretRedactor,
        )
        assert cleanup_started.is_set()
        release_cleanup.set()
        await retained[0]
        return outcome

    outcome = asyncio.run(scenario())

    assert len(retained) == 1
    assert outcome.result.is_error is True
    assert outcome.result.structured["outcome_unknown"] is True
    assert outcome.result.structured["manual_reconciliation_required"] is True
    assert outcome.result.structured["isolated_tool_failure_code"] == ("process_cleanup_unproven")
    assert outcome.result.structured["isolated_tool_cleanup_failure_code"] == (
        "temporary_directory_cleanup_deadline_exceeded"
    )


@pytest.mark.process
def test_global_timeout_does_not_discard_unproven_process_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cleanup_attempts = 0

    async def never_proven(*_args, **_kwargs) -> bool:
        nonlocal cleanup_attempts
        cleanup_attempts += 1
        return False

    monkeypatch.setattr(isolated_process, "_wait_for_group_and_child", never_proven)

    async def scenario():
        arguments = {"text": "hello"}
        tool = _tool(
            mode="gil_block",
            deadline_seconds=30,
            factory_config={
                "seconds": 30,
                "started_path": str(tmp_path / "started"),
            },
            effect=ToolEffect.EXTERNAL,
        )
        return await tool_execution.run_tool(
            tool=tool,
            effect=ToolEffect.EXTERNAL,
            ctx=_context(arguments),
            arguments=arguments,
            registered_schema=_SCHEMA,
            registered_execution_contract=isolated_tool_execution_contract(
                tool,
                runtime_timeout_seconds=2.5,
            ),
            redactor=SecretRedactor,
            timeout_seconds=2.5,
        )

    try:
        outcome = asyncio.run(scenario())

        assert cleanup_attempts >= 2
        assert outcome.result.is_error is True
        assert outcome.result.structured["terminal_outcome"] == "tool_execution_timeout"
        assert outcome.result.structured["outcome_unknown"] is True
        assert outcome.result.structured["manual_reconciliation_required"] is True
        assert outcome.result.structured["isolated_tool_failure_code"] == (
            "process_cleanup_unproven"
        )
        assert outcome.result.structured["isolated_tool_cleanup_failure_code"] == (
            "process_cleanup_deadline_exceeded"
        )
        assert isolated_process._retained_isolated_tool_cleanup_pending() is True
    finally:
        isolated_process._RETAINED_ISOLATED_TOOL_TASKS.clear()
        isolated_process._RETAINED_ISOLATED_TOOL_RETRIES.clear()


@pytest.mark.process
def test_real_timeout_kills_the_complete_child_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "grandchild.pid"
    with pytest.raises(IsolatedToolDeadlineExceeded):
        asyncio.run(
            _execute(
                _tool(
                    mode="grandchild",
                    deadline_seconds=2.5,
                    factory_config={
                        "pid_path": str(pid_path),
                        "seconds": 30,
                        "started_path": str(tmp_path / "started"),
                    },
                )
            )
        )
    grandchild_pid = int(pid_path.read_text(encoding="utf-8"))

    _assert_process_gone(grandchild_pid)


@pytest.mark.process
def test_successful_frame_does_not_wait_for_descendant_held_result_descriptor(
    tmp_path: Path,
) -> None:
    pid_path = tmp_path / "forked-child.pid"

    result = asyncio.run(
        _execute(
            _tool(
                mode="fork_then_success",
                deadline_seconds=3,
                factory_config={"pid_path": str(pid_path)},
            )
        )
    )

    assert result == ToolResult(content="forked handler completed")
    _assert_process_gone(int(pid_path.read_text(encoding="utf-8")))


@pytest.mark.process
def test_real_cancellation_kills_the_complete_child_process_group(tmp_path: Path) -> None:
    pid_path = tmp_path / "grandchild.pid"
    started_path = tmp_path / "started"

    async def scenario() -> tuple[int, bool]:
        task = asyncio.create_task(
            _execute(
                _tool(
                    mode="grandchild",
                    deadline_seconds=10,
                    factory_config={
                        "pid_path": str(pid_path),
                        "seconds": 30,
                        "started_path": str(started_path),
                    },
                )
            )
        )
        async with asyncio.timeout(5):
            while not started_path.exists():
                await asyncio.sleep(0.01)
        task.cancel("cancel descendant owner")
        cancelling = task.cancelling()
        with pytest.raises(asyncio.CancelledError, match="cancel descendant owner"):
            await task
        return cancelling, task.cancelled()

    cancelling, cancelled = asyncio.run(scenario())
    assert cancelling == 1
    assert cancelled is True
    _assert_process_gone(int(pid_path.read_text(encoding="utf-8")))


def _assert_process_gone(process_id: int) -> None:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("isolated tool grandchild survived process-group cleanup")


def test_predispatch_validation_fails_without_starting_a_child() -> None:
    with pytest.raises(IsolatedToolPreDispatchFailure) as caught:
        asyncio.run(_execute(_tool(), {"unexpected": "value"}))

    assert caught.value.code == "arguments_invalid"

    mismatched_arguments = {"text": "dispatched"}
    with pytest.raises(IsolatedToolPreDispatchFailure) as mismatch:
        asyncio.run(
            execute_process_isolated_tool(
                tool=_tool(),
                context=_context({"text": "authorized"}),
                arguments=mismatched_arguments,
                registered_schema=_SCHEMA,
                redactor=SecretRedactor(),
            )
        )
    assert mismatch.value.code == "runtime_authority_mismatch"

    split_boundary_arguments = {"text": "x" * 900}
    with pytest.raises(IsolatedToolPreDispatchFailure) as aggregate_limit:
        asyncio.run(
            execute_process_isolated_tool(
                tool=_tool(max_request_bytes=1024),
                context=_context(split_boundary_arguments),
                arguments=split_boundary_arguments,
                registered_schema=_SCHEMA,
                redactor=SecretRedactor(),
            )
        )
    assert aggregate_limit.value.code == "request_invalid_or_too_large"


def test_runtime_rejects_post_registration_adapter_configuration_drift() -> None:
    arguments = {"text": "hello"}
    tool = _tool()
    registered_contract = isolated_tool_execution_contract(tool)
    object.__setattr__(
        tool,
        "_factory_config",
        {"mode": "success", "post_registration_drift": True},
    )

    outcome = asyncio.run(
        tool_execution.run_tool(
            tool=tool,
            effect=ToolEffect.NONE,
            ctx=_context(arguments),
            arguments=arguments,
            registered_schema=_SCHEMA,
            registered_execution_contract=registered_contract,
            redactor=SecretRedactor,
        )
    )

    assert outcome.result.is_error is True
    assert outcome.result.structured["isolated_tool_failure_code"] == (
        "registered_execution_contract_mismatch"
    )
    assert outcome.result.structured["tool_execution_boundary"] == "posix_process"
    assert outcome.result.structured["tool_timeout_strength"] == "hard_process_deadline"
    assert outcome.terminal_payload_fields() == {
        "isolated_tool_failure_code": "registered_execution_contract_mismatch",
        "tool_execution_boundary": "posix_process",
        "tool_timeout_strength": "hard_process_deadline",
    }


@pytest.mark.parametrize("failure_stage", ["temporary_directory", "result_pipe"])
def test_process_boundary_setup_failures_are_typed_before_child_creation(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_mkdtemp = isolated_process.tempfile.mkdtemp
    temporary_directories: list[Path] = []

    def tracked_mkdtemp(*args, **kwargs) -> str:
        if failure_stage == "temporary_directory":
            raise OSError("host temporary directory unavailable")
        directory = original_mkdtemp(*args, **kwargs)
        temporary_directories.append(Path(directory))
        return directory

    def failing_pipe() -> tuple[int, int]:
        raise OSError("host descriptor allocation unavailable")

    async def forbidden_spawn(*_args, **_kwargs):
        raise AssertionError("process setup failure must precede child creation")

    monkeypatch.setattr(isolated_process.tempfile, "mkdtemp", tracked_mkdtemp)
    if failure_stage == "result_pipe":
        monkeypatch.setattr(isolated_process.os, "pipe", failing_pipe)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)

    async def scenario():
        arguments = {"text": "hello"}
        tool = _tool()
        return await tool_execution.run_tool(
            tool=tool,
            effect=ToolEffect.NONE,
            ctx=_context(arguments),
            arguments=arguments,
            registered_schema=_SCHEMA,
            registered_execution_contract=isolated_tool_execution_contract(tool),
            redactor=SecretRedactor,
        )

    outcome = asyncio.run(scenario())

    assert outcome.result.is_error is True
    assert outcome.result.structured == {
        "error": "tool_unavailable",
        "isolated_tool_failure_code": "process_boundary_setup_failed",
        "tool_execution_boundary": "posix_process",
        "tool_timeout_strength": "hard_process_deadline",
    }
    assert outcome.terminal_payload_fields() == {
        "isolated_tool_failure_code": "process_boundary_setup_failed",
        "tool_execution_boundary": "posix_process",
        "tool_timeout_strength": "hard_process_deadline",
    }
    assert all(not directory.exists() for directory in temporary_directories)


def _public_app(
    tool: ProcessIsolatedTool,
    *,
    tool_policy=None,
    tool_timeout_seconds: float | None = None,
    secret_redactor: SecretRedactor | None = None,
    public_authority_alias_keyring: PublicAuthorityAliasKeyring | None = None,
) -> CayuApp:
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="isolated-call-1",
                    name="isolated_fixture",
                    arguments={"text": "public hello"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("done"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        enable_logging=False,
        tool_timeout_seconds=tool_timeout_seconds,
        secret_redactor=secret_redactor,
        public_authority_alias_keyring=public_authority_alias_keyring,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="test-model"),
        tools=[tool],
        tool_policy=tool_policy,
    )
    return app


async def _run_public(app: CayuApp, *, session_id: str) -> list[Any]:
    return [
        event
        async for event in app.run(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "run the isolated tool")],
            )
        )
    ]


async def _resume_public(app: CayuApp, *, session_id: str) -> list[Any]:
    return [
        event
        async for event in app.resume(
            ResumeRequest(
                session_id=session_id,
                messages=[Message.text("user", "continue after recovery")],
            )
        )
    ]


class _FailingFirstCompletedToolEventStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self.failed_terminal_once = False

    async def append_events(self, session_id: str, events: list[Event]) -> None:
        if not self.failed_terminal_once and any(
            event.type == EventType.TOOL_CALL_COMPLETED for event in events
        ):
            self.failed_terminal_once = True
            raise RuntimeError("terminal tool event unavailable")
        await super().append_events(session_id, events)


@pytest.mark.process
def test_public_runtime_executes_registered_isolated_tool_and_exposes_truthful_evidence() -> None:
    app = _public_app(_tool(deadline_seconds=3))

    manifest_tool = app.describe().agents[0].tools[0]
    assert manifest_tool.execution_boundary == "posix_process"
    assert manifest_tool.timeout_strength == "hard_process_deadline"
    assert manifest_tool.sandboxed is False
    assert manifest_tool.adapter_identity == {
        "name": "cayu:testing:deterministic-isolated-tool",
        "behavior_version": "1",
        "implementation_version": "1",
    }
    assert manifest_tool.adapter_configuration_sha256 is not None
    assert manifest_tool.hard_deadline_seconds == 3

    events = asyncio.run(_run_public(app, session_id="public-isolated-success"))

    completed = next(event for event in events if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["result"]["content"] == "public hello"
    assert events[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.process
def test_public_runtime_applies_the_ordinary_secret_safe_result_boundary() -> None:
    canary = "ISOLATED_RESULT_SECRET_CANARY"
    keyring = PublicAuthorityAliasKeyring(
        active_key_id="isolated",
        keys={
            "isolated": SecretStr(
                base64.urlsafe_b64encode(bytes(range(32))).decode("ascii").rstrip("=")
            )
        },
    )
    app = _public_app(
        _tool(mode="secret_output", deadline_seconds=3),
        secret_redactor=SecretRedactor(canary),
        public_authority_alias_keyring=keyring,
    )

    events = asyncio.run(_run_public(app, session_id="public-isolated-secret-result"))
    transcript = asyncio.run(app.session_store.load_transcript("public-isolated-secret-result"))

    assert events[-1].type == EventType.SESSION_COMPLETED
    assert canary not in repr([event.model_dump(mode="json") for event in events])
    assert canary not in repr([message.model_dump(mode="json") for message in transcript])


@pytest.mark.process
def test_public_hard_timeout_preserves_external_effect_uncertainty_and_no_replay(
    tmp_path: Path,
) -> None:
    started_path = tmp_path / "started"
    app = _public_app(
        _tool(
            mode="gil_block",
            deadline_seconds=2.5,
            factory_config={"seconds": 30, "started_path": str(started_path)},
            effect=ToolEffect.EXTERNAL,
        ),
        secret_redactor=SecretRedactor("UNRELATED_REGISTERED_SECRET"),
    )

    events = asyncio.run(_run_public(app, session_id="public-isolated-timeout"))

    assert started_path.read_text(encoding="utf-8") == "started"
    failed = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert failed.payload["terminal_outcome"] == "tool_execution_timeout"
    assert failed.payload["outcome_unknown"] is True
    assert failed.payload["manual_reconciliation_required"] is True
    assert failed.payload["isolated_tool_failure_code"] == "hard_process_deadline_exceeded"
    assert failed.payload["tool_execution_boundary"] == "posix_process"
    assert failed.payload["tool_timeout_strength"] == "hard_process_deadline"
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in events) == 1
    assert sum(event.type == EventType.TOOL_CALL_FAILED for event in events) == 1


@pytest.mark.process
def test_recovery_of_started_isolated_call_never_launches_a_duplicate_child(
    tmp_path: Path,
) -> None:
    count_path = tmp_path / "recovery-child-count"
    store = _FailingFirstCompletedToolEventStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="isolated-recovery-call",
                    name="isolated_fixture",
                    arguments={"text": "execute once"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("recovered without replay"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="test-model"),
        tools=[
            _tool(
                mode="counted_success",
                deadline_seconds=3,
                factory_config={"count_path": str(count_path)},
                effect=ToolEffect.EXTERNAL,
            )
        ],
    )

    initial = asyncio.run(_run_public(app, session_id="isolated-recovery-no-replay"))
    recovered = asyncio.run(_resume_public(app, session_id="isolated-recovery-no-replay"))
    durable_events = asyncio.run(store.load_events("isolated-recovery-no-replay"))

    assert initial[-1].type == EventType.SESSION_FAILED
    assert recovered[-1].type == EventType.SESSION_COMPLETED
    assert count_path.read_text(encoding="utf-8").splitlines() == ["started"]
    recovered_failure = next(
        event
        for event in recovered
        if event.type == EventType.TOOL_CALL_FAILED and event.payload.get("recovered") is True
    )
    assert recovered_failure.payload["result"]["is_error"] is True
    terminal_events = [
        event
        for event in durable_events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0].payload["result"]["structured"]["outcome_unknown"] is True


@pytest.mark.process
def test_public_policy_denial_occurs_before_isolated_child_creation(tmp_path: Path) -> None:
    started_path = tmp_path / "must-not-exist"
    app = _public_app(
        _tool(
            mode="gil_block",
            deadline_seconds=3,
            factory_config={"seconds": 30, "started_path": str(started_path)},
        ),
        tool_policy=StaticToolPolicy(deny={"isolated_fixture"}),
    )

    events = asyncio.run(_run_public(app, session_id="public-isolated-policy-denial"))

    assert not started_path.exists()
    assert any(event.type == EventType.TOOL_CALL_BLOCKED for event in events)


@pytest.mark.process
def test_public_approval_precedes_child_creation_and_exact_retry_does_not_reexecute(
    tmp_path: Path,
) -> None:
    count_path = tmp_path / "child-start-count"
    app = _public_app(
        _tool(
            mode="counted_success",
            deadline_seconds=3,
            factory_config={"count_path": str(count_path)},
        ),
        tool_policy=AlwaysRequireApprovalToolPolicy(),
    )

    initial = asyncio.run(_run_public(app, session_id="public-isolated-approval"))
    approval_event = next(
        event for event in initial if event.type == EventType.TOOL_CALL_APPROVAL_REQUESTED
    )
    assert not count_path.exists()
    approval = approval_event.payload["approval"]
    request = ToolApprovalRequest(
        session_id="public-isolated-approval",
        approval_id=approval["approval_id"],
        tool_round_id=approval_event.payload["tool_round_id"],
        tool_call_id=approval_event.payload["tool_call_id"],
        decision=ToolApprovalDecision.APPROVE,
        reason="approved by isolated-boundary test",
    )

    approved = asyncio.run(_collect_approval_events(app, request))
    replayed = asyncio.run(_collect_approval_events(app, request))

    assert count_path.read_text(encoding="utf-8").splitlines() == ["started"]
    assert sum(event.type == EventType.TOOL_CALL_STARTED for event in approved) == 1
    assert not any(event.type == EventType.TOOL_CALL_STARTED for event in replayed)


@pytest.mark.process
@pytest.mark.parametrize(
    "registered_secret",
    ["UNRELATED_REGISTERED_SECRET", "external", "process_interrupted"],
)
def test_public_runtime_interruption_settles_one_child_and_one_provider_result(
    tmp_path: Path,
    registered_secret: str,
) -> None:
    session_id = "public-isolated-interruption"
    started_path = tmp_path / "interrupted-child-started"
    count_path = tmp_path / "interrupted-child-count"
    store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="interrupted-isolated-call",
                    name="isolated_fixture",
                    arguments={"text": "block"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("continued after interruption"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=SecretRedactor(registered_secret),
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="test-model"),
        tools=[
            _tool(
                mode="counted_gil_block",
                deadline_seconds=30,
                factory_config={
                    "count_path": str(count_path),
                    "seconds": 30,
                    "started_path": str(started_path),
                },
                effect=ToolEffect.EXTERNAL,
            )
        ],
    )

    async def scenario():
        run_task = asyncio.create_task(_run_public(app, session_id=session_id))
        async with asyncio.timeout(5):
            while not started_path.exists():
                await asyncio.sleep(0.01)
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="operator drain",
                )
            )
        ]
        run_events = await asyncio.wait_for(run_task, timeout=5)
        interrupted_session = await store.load(session_id)
        resumed_events = await _resume_public(app, session_id=session_id)
        durable_events = await store.load_events(session_id)
        return run_events, interrupt_events, interrupted_session, resumed_events, durable_events

    run_events, interrupt_events, interrupted_session, resumed_events, durable_events = asyncio.run(
        scenario()
    )

    assert count_path.read_text(encoding="utf-8").splitlines() == ["started"]
    assert interrupted_session is not None
    assert interrupted_session.status is SessionStatus.INTERRUPTED
    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert resumed_events[-1].type is EventType.SESSION_COMPLETED
    terminal_events = [
        event
        for event in durable_events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    ]
    assert len(terminal_events) == 1
    assert terminal_events[0].type is EventType.TOOL_CALL_FAILED
    assert terminal_events[0].payload["interrupted"] is True
    assert terminal_events[0].payload["result"]["structured"]["interrupted"] is True
    expected_controls = {
        "terminal_outcome": "tool_execution_error",
        "tool_effect": "external",
        "outcome_unknown": True,
        "manual_reconciliation_required": True,
        "isolated_tool_failure_code": "process_interrupted",
        "tool_execution_boundary": "posix_process",
        "tool_timeout_strength": "hard_process_deadline",
    }
    for key, value in expected_controls.items():
        assert terminal_events[0].payload[key] == value
        assert terminal_events[0].payload["result"]["structured"][key] == value
    provider_results = [
        part
        for message in provider.requests[1].messages
        for part in message.content
        if isinstance(part, ToolResultPart)
    ]
    assert len(provider_results) == 1
    assert provider_results[0].is_error is True
    for key, value in expected_controls.items():
        assert provider_results[0].structured[key] == value


@pytest.mark.process
def test_policy_interruption_does_not_claim_isolated_child_dispatch(tmp_path: Path) -> None:
    class BlockingPolicy(ToolPolicy):
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def authorize(self, request: ToolPolicyRequest) -> ToolPolicyResult:
            self.entered.set()
            await self.release.wait()
            return ToolPolicyResult(decision=ToolPolicyDecision.ALLOW)

    session_id = "public-isolated-policy-interruption"
    count_path = tmp_path / "child-count"
    policy = BlockingPolicy()
    store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="policy-interrupted-isolated-call",
                    name="isolated_fixture",
                    arguments={"text": "must-not-run"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("continued after pre-dispatch interruption"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="test-model"),
        tools=[
            _tool(
                mode="counted_success",
                deadline_seconds=3,
                factory_config={"count_path": str(count_path)},
                effect=ToolEffect.EXTERNAL,
            )
        ],
        tool_policy=policy,
    )

    async def scenario():
        run_task = asyncio.create_task(_run_public(app, session_id=session_id))
        await asyncio.wait_for(policy.entered.wait(), timeout=5)
        interrupt_events = [
            event
            async for event in app.interrupt_session(
                InterruptSessionRequest(
                    session_id=session_id,
                    reason="interrupt while policy owns pre-dispatch work",
                )
            )
        ]
        run_events = await asyncio.wait_for(run_task, timeout=5)
        resumed_events = await _resume_public(app, session_id=session_id)
        durable_events = await store.load_events(session_id)
        return run_events, interrupt_events, resumed_events, durable_events

    run_events, interrupt_events, resumed_events, durable_events = asyncio.run(scenario())

    assert not count_path.exists()
    assert run_events[-1].type is EventType.SESSION_INTERRUPTED
    assert [event.type for event in interrupt_events] == [EventType.SESSION_INTERRUPTED]
    assert resumed_events[-1].type is EventType.SESSION_COMPLETED
    terminal = next(
        event
        for event in durable_events
        if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
    )
    assert terminal.type is EventType.TOOL_CALL_FAILED
    assert terminal.payload["interrupted"] is True
    assert terminal.payload["result"]["structured"]["interrupted"] is True
    for field_name in (
        "isolated_tool_failure_code",
        "manual_reconciliation_required",
        "outcome_unknown",
        "terminal_outcome",
        "tool_execution_boundary",
        "tool_timeout_strength",
    ):
        assert field_name not in terminal.payload
        assert field_name not in terminal.payload["result"]["structured"]


@pytest.mark.process
def test_task_worker_remains_live_and_renews_lease_while_isolated_child_blocks_gil() -> None:
    class HeartbeatObservingTaskStore(InMemoryTaskStore):
        verified_work_mutations_are_cancellation_quiescent = True

        def __init__(self) -> None:
            super().__init__()
            self.heartbeat_count = 0
            self.heartbeat_seen = asyncio.Event()

        async def heartbeat(
            self,
            task_id: str,
            worker_id: str,
            *,
            extend_seconds: int = 300,
        ) -> Task:
            task = await super().heartbeat(
                task_id,
                worker_id,
                extend_seconds=extend_seconds,
            )
            self.heartbeat_count += 1
            self.heartbeat_seen.set()
            return task

    task_store = HeartbeatObservingTaskStore()
    session_store = InMemorySessionStore()
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="worker-blocking-call",
                    name="isolated_fixture",
                    arguments={"text": "block"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("continued after hard timeout"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="worker-healthy-call",
                    name="isolated_fixture",
                    arguments={"text": "healthy"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta("continued after healthy tool"),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    app = CayuApp(
        session_store=session_store,
        task_store=task_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="worker-agent", model="test-model"),
        tools=[
            _tool(
                mode="conditional_gil_block",
                deadline_seconds=2.5,
                factory_config={"seconds": 30},
            )
        ],
    )
    server = create_server(app, config=ServerConfig.local_development())

    async def handler(app: CayuApp, task: Task, worker_id: str) -> None:
        async for _event in app.run(
            RunRequest(
                agent_name="worker-agent",
                session_id=f"worker-session-{task.id}",
                task_id=task.id,
                task_worker_id=worker_id,
                messages=[Message.text("user", "run the isolated tool")],
            )
        ):
            pass

    async def scenario():
        blocking_task = await task_store.create_task(
            TaskCreate(task_id="isolated-worker-blocking", type="isolated-worker-job")
        )
        healthy_task = await task_store.create_task(
            TaskCreate(task_id="isolated-worker-healthy", type="isolated-worker-job")
        )
        worker_task = asyncio.create_task(
            run_task_worker(
                app,
                task_store,
                handler,
                worker_id="isolated-worker",
                query=TaskQuery(type="isolated-worker-job"),
                lease_seconds=1,
                poll_interval_s=0.01,
                reclaim=False,
                max_tasks=2,
            )
        )
        await asyncio.wait_for(task_store.heartbeat_seen.wait(), timeout=2)
        assert not worker_task.done()
        async with asyncio.timeout(5):
            while True:
                active_events = await session_store.load_events(
                    f"worker-session-{blocking_task.id}"
                )
                if any(event.type is EventType.TOOL_CALL_STARTED for event in active_events):
                    break
                await asyncio.sleep(0.01)
        assert any(event.type is EventType.TOOL_CALL_STARTED for event in active_events)
        assert not any(
            event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
            for event in active_events
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server),
            base_url="http://cayu.test",
        ) as client:
            health_response = await asyncio.wait_for(
                client.get("/api/health"),
                timeout=1,
            )
        handled = await asyncio.wait_for(worker_task, timeout=8)
        return (
            handled,
            health_response,
            await task_store.load_task(blocking_task.id),
            await task_store.load_task(healthy_task.id),
            await session_store.load_events(f"worker-session-{blocking_task.id}"),
            await session_store.load_events(f"worker-session-{healthy_task.id}"),
        )

    (
        handled,
        health_response,
        blocking_task,
        healthy_task,
        blocking_events,
        healthy_events,
    ) = asyncio.run(scenario())

    assert handled == 2
    assert health_response.status_code == 200
    assert health_response.json() == {"ok": True}
    assert task_store.heartbeat_count >= 1
    assert blocking_task is not None
    assert blocking_task.status is TaskStatus.COMPLETED
    assert healthy_task is not None
    assert healthy_task.status is TaskStatus.COMPLETED
    blocking_failure = next(
        event for event in blocking_events if event.type is EventType.TOOL_CALL_FAILED
    )
    assert blocking_failure.payload["isolated_tool_failure_code"] == (
        "hard_process_deadline_exceeded"
    )
    healthy_completion = next(
        event for event in healthy_events if event.type is EventType.TOOL_CALL_COMPLETED
    )
    assert healthy_completion.payload["result"]["content"] == "healthy"


async def _collect_approval_events(
    app: CayuApp,
    request: ToolApprovalRequest,
) -> list[Any]:
    return [event async for event in app.resolve_tool_approval(request)]


@pytest.mark.process
def test_application_tool_timeout_remains_a_hard_deadline_for_isolated_adapter(
    tmp_path: Path,
) -> None:
    started_path = tmp_path / "global-timeout-started"
    app = _public_app(
        _tool(
            mode="gil_block",
            deadline_seconds=30,
            factory_config={"seconds": 30, "started_path": str(started_path)},
            effect=ToolEffect.IDEMPOTENT,
        ),
        tool_timeout_seconds=5,
    )

    assert app.describe().agents[0].tools[0].hard_deadline_seconds == 5
    started_at = time.monotonic()
    events = asyncio.run(_run_public(app, session_id="public-global-hard-deadline"))
    elapsed = time.monotonic() - started_at

    failed = next(event for event in events if event.type == EventType.TOOL_CALL_FAILED)
    assert started_path.read_text(encoding="utf-8") == "started"
    assert elapsed < 9
    assert failed.payload["terminal_outcome"] == "tool_execution_timeout"
    assert failed.payload["outcome_unknown"] is True
    assert failed.payload["tool_execution_boundary"] == "posix_process"
    assert failed.payload["tool_timeout_strength"] == "hard_process_deadline"


@pytest.mark.process
def test_same_runtime_completes_a_later_session_after_killing_wedged_child(
    tmp_path: Path,
) -> None:
    started_path = tmp_path / "started"
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="wedged-call",
                    name="isolated_fixture",
                    arguments={"text": "block"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
            [
                ModelStreamEvent.tool_call(
                    id="healthy-call",
                    name="isolated_fixture",
                    arguments={"text": "healthy"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="test-model"),
        tools=[
            _tool(
                mode="conditional_gil_block",
                deadline_seconds=2.5,
                factory_config={"seconds": 30, "started_path": str(started_path)},
            )
        ],
    )

    first = asyncio.run(_run_public(app, session_id="wedged-session"))
    second = asyncio.run(_run_public(app, session_id="healthy-session"))

    assert any(event.type == EventType.TOOL_CALL_FAILED for event in first)
    completed = next(event for event in second if event.type == EventType.TOOL_CALL_COMPLETED)
    assert completed.payload["result"]["content"] == "healthy"
    assert second[-1].type == EventType.SESSION_COMPLETED


@pytest.mark.process
def test_parallel_isolated_process_groups_settle_independently(tmp_path: Path) -> None:
    started_path = tmp_path / "parallel-blocked-started"
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(
                    id="parallel-blocked-call",
                    name="isolated_fixture",
                    arguments={"text": "block"},
                ),
                ModelStreamEvent.tool_call(
                    id="parallel-healthy-call",
                    name="isolated_fixture",
                    arguments={"text": "healthy"},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [ModelStreamEvent.completed({"finish_reason": "stop"})],
        ]
    )
    app = CayuApp(enable_logging=False, max_parallel_tool_calls=2)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="test-model"),
        tools=[
            _tool(
                mode="conditional_gil_block",
                deadline_seconds=5.0,
                factory_config={"seconds": 30, "started_path": str(started_path)},
                effect=ToolEffect.IDEMPOTENT,
            )
        ],
    )

    events = asyncio.run(_run_public(app, session_id="parallel-isolated-groups"))

    completed = [event for event in events if event.type == EventType.TOOL_CALL_COMPLETED]
    failed = [event for event in events if event.type == EventType.TOOL_CALL_FAILED]
    assert started_path.read_text(encoding="utf-8") == "started"
    assert len(completed) == 1
    assert completed[0].payload["result"]["content"] == "healthy"
    assert len(failed) == 1
    assert failed[0].payload["terminal_outcome"] == "tool_execution_timeout"
    assert events[-1].type == EventType.SESSION_COMPLETED
