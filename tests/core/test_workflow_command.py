"""Journaled ``command`` workflow steps on a local runner."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from cayu import CayuApp, EventType
from cayu.environments import (
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
)
from cayu.runners import LocalRunner
from cayu.sessions import InMemorySessionStore
from cayu.vaults import SecretRedactor
from cayu.workflows import (
    CommandResult,
    GateOutcome,
    StepError,
    StepResult,
    WorkflowBase,
    WorkflowSpec,
    command,
    gated_loop,
    normalize_gate_outcome,
)

PY = sys.executable


def _app(root: Path, **kwargs) -> CayuApp:
    app = CayuApp(enable_logging=False, **kwargs)
    app.register_environment(
        Environment(EnvironmentSpec(name="local"), runner=LocalRunner(root)),
        default=True,
    )
    return app


class _CommandWorkflow(WorkflowBase):
    spec = WorkflowSpec(name="command-workflow")

    def __init__(self, app: CayuApp, script: str, **command_kwargs) -> None:
        super().__init__(app)
        self.script = script
        self.command_kwargs = command_kwargs
        self.result: CommandResult | None = None

    async def run(self, session_id: str):
        ctx = self.context(session_id)
        yield await ctx.start()
        self.result = await command(
            ctx, step_id="check", argv=[PY, "-c", self.script], **self.command_kwargs
        )
        yield await ctx.completed({"passed": self.result.passed})


async def _drain(workflow: WorkflowBase, session_id: str) -> list:
    return [event async for event in workflow.run(session_id)]


async def _step_events(app: CayuApp, session_id: str, event_type: EventType) -> list:
    return [
        event
        for event in await app.session_store.load_events(session_id)
        if event.type == event_type
    ]


_COUNTING_SCRIPT = (
    "import pathlib, sys; p = pathlib.Path('runs.txt'); "
    "p.write_text(p.read_text() + 'x' if p.exists() else 'x'); "
    "print('ran', len(p.read_text())); sys.stderr.write('warn'); sys.exit({code})"
)


def test_command_journals_outcome_and_returns_typed_result(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        workflow = _CommandWorkflow(app, _COUNTING_SCRIPT.format(code=3), timeout_s=30)
        await _drain(workflow, "run-1")
        result = workflow.result
        assert result is not None
        assert result.exit_code == 3
        assert not result.passed
        assert not result.timed_out
        assert result.stdout.strip() == "ran 1"
        assert result.stderr == "warn"
        assert result.environment_name == "local"
        assert result.argv[0] == PY
        assert not result.replayed

        (started,) = await _step_events(app, "run-1", EventType.WORKFLOW_STEP_STARTED)
        assert started.payload["kind"] == "command"
        assert started.payload["step_id"] == "check"
        assert started.payload["environment_name"] == "local"
        (completed,) = await _step_events(app, "run-1", EventType.WORKFLOW_STEP_COMPLETED)
        payload = completed.payload
        assert payload["kind"] == "command"
        assert payload["exit_code"] == 3
        assert payload["passed"] is False
        assert payload["timed_out"] is False
        assert payload["stdout"].strip() == "ran 1"
        assert payload["stderr"] == "warn"
        assert isinstance(payload["duration_ms"], int)

    asyncio.run(scenario())


def test_resume_returns_journaled_result_without_rerunning(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        script = _COUNTING_SCRIPT.format(code=0)
        first = _CommandWorkflow(app, script)
        await _drain(first, "run-resume")
        second = _CommandWorkflow(app, script)
        await _drain(second, "run-resume")

        assert (tmp_path / "runs.txt").read_text() == "x"
        assert first.result is not None and second.result is not None
        assert second.result.replayed
        assert second.result.passed
        assert second.result.stdout == first.result.stdout
        assert second.result.exit_code == first.result.exit_code
        assert second.result.duration_ms == first.result.duration_ms

        rerun = _CommandWorkflow(app, script, rerun=True)
        await _drain(rerun, "run-resume")
        assert (tmp_path / "runs.txt").read_text() == "xx"
        assert rerun.result is not None and not rerun.result.replayed
        assert rerun.result.stdout.strip() == "ran 2"

        # A later resume replays the newest completion.
        latest = _CommandWorkflow(app, script)
        await _drain(latest, "run-resume")
        assert latest.result is not None and latest.result.replayed
        assert latest.result.stdout.strip() == "ran 2"
        assert (tmp_path / "runs.txt").read_text() == "xx"

    asyncio.run(scenario())


def test_command_redacts_and_bounds_output(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(
            tmp_path,
            session_store=InMemorySessionStore(),
            secret_redactor=SecretRedactor("hunter2-canary"),
        )
        script = "print('token=hunter2-canary'); print('y' * 5000)"
        workflow = _CommandWorkflow(app, script, output_limit_bytes=1000)
        await _drain(workflow, "run-redact")
        result = workflow.result
        assert result is not None
        assert "hunter2-canary" not in result.stdout
        assert result.stdout.startswith("token=")
        assert result.stdout_truncated
        assert len(result.stdout.encode()) <= 1000 + 64

        events = await app.session_store.load_events("run-redact")
        assert all("hunter2-canary" not in str(event.payload) for event in events)

    asyncio.run(scenario())


def test_command_timeout_is_a_result_not_an_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        workflow = _CommandWorkflow(app, "import time; time.sleep(30)", timeout_s=1)
        await _drain(workflow, "run-timeout")
        result = workflow.result
        assert result is not None
        assert result.timed_out
        assert not result.passed
        (completed,) = await _step_events(app, "run-timeout", EventType.WORKFLOW_STEP_COMPLETED)
        assert completed.payload["timed_out"] is True

    asyncio.run(scenario())


def test_command_passes_stdin(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        workflow = _CommandWorkflow(
            app, "import sys; print(sys.stdin.read().upper())", stdin="hello"
        )
        await _drain(workflow, "run-stdin")
        assert workflow.result is not None
        assert workflow.result.stdout.strip() == "HELLO"

    asyncio.run(scenario())


def test_command_result_is_a_gate_verdict() -> None:
    passing = CommandResult(step_id="s", argv=("true",), exit_code=0)
    failing = CommandResult(step_id="s", argv=("false",), exit_code=1)
    timed_out = CommandResult(step_id="s", argv=("sleep",), exit_code=0, timed_out=True)
    assert normalize_gate_outcome(passing) == passing.gate_outcome()
    assert normalize_gate_outcome(passing).passed
    assert not normalize_gate_outcome(failing).passed
    assert not normalize_gate_outcome(timed_out).passed
    assert normalize_gate_outcome(failing).detail["exit_code"] == 1


def test_command_as_gated_loop_gate(tmp_path: Path) -> None:
    class LoopWorkflow(WorkflowBase):
        spec = WorkflowSpec(name="command-gate")

        def __init__(self, app: CayuApp) -> None:
            super().__init__(app)
            self.passed: list[str] = []
            self.failed: list[str] = []

        async def run(self, session_id: str):
            ctx = self.context(session_id)
            yield await ctx.start()

            async def do(item: str) -> StepResult:
                return StepResult(step_id=f"do-{item}", session_id=session_id)

            async def gate(item: str, _result: StepResult) -> CommandResult:
                code = 0 if item == "good" else 1
                return await command(
                    ctx,
                    step_id=f"check-{item}",
                    argv=[PY, "-c", f"raise SystemExit({code})"],
                )

            async def on_pass(item: str, _r: StepResult, outcome: GateOutcome) -> None:
                self.passed.append(item)

            async def on_fail(item: str, _r: StepResult, outcome: GateOutcome) -> None:
                assert outcome.detail["exit_code"] == 1
                self.failed.append(item)

            async for event in gated_loop(
                ctx,
                ["good", "bad"],
                do=do,
                gate=gate,
                on_pass=on_pass,
                on_fail=on_fail,
                key=str,
            ):
                yield event
            yield await ctx.completed()

    async def scenario() -> None:
        app = _app(tmp_path)
        workflow = LoopWorkflow(app)
        await _drain(workflow, "run-loop")
        assert workflow.passed == ["good"]
        assert workflow.failed == ["bad"]

    asyncio.run(scenario())


def test_command_rejects_duplicate_step_id(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        workflow = _CommandWorkflow(app, "pass")
        ctx = workflow.context("run-dup")
        await ctx.start()
        await command(ctx, step_id="same", argv=[PY, "-c", "pass"])
        with pytest.raises(ValueError, match="Duplicate step_id"):
            await command(ctx, step_id="same", argv=[PY, "-c", "pass"])

    asyncio.run(scenario())


def test_command_runner_failure_raises_step_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        runner = LocalRunner(tmp_path)
        await runner.close()
        app = CayuApp(enable_logging=False)
        app.register_environment(
            Environment(EnvironmentSpec(name="closed"), runner=runner), default=True
        )
        ctx = _CommandWorkflow(app, "pass").context("run-closed")
        await ctx.start()
        with pytest.raises(StepError) as raised:
            await command(ctx, step_id="closed", argv=[PY, "-c", "pass"])
        assert raised.value.step_id == "closed"
        assert raised.value.workflow_attempt_id == ctx.attempt_id
        assert await _step_events(app, "run-closed", EventType.WORKFLOW_STEP_COMPLETED) == []

    asyncio.run(scenario())


def test_command_needs_a_concrete_environment(tmp_path: Path) -> None:
    class Factory(EnvironmentFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            raise AssertionError("not called")

    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        app.register_environment_factory(EnvironmentSpec(name="per-session"), Factory())
        ctx = _CommandWorkflow(app, "pass").context("run-factory")
        await ctx.start()
        with pytest.raises(ValueError, match="factory-backed"):
            await command(
                ctx, step_id="check", argv=[PY, "-c", "pass"], environment_name="per-session"
            )
        with pytest.raises(ValueError, match="needs an environment"):
            await command(ctx, step_id="check-default", argv=[PY, "-c", "pass"])
        with pytest.raises(ValueError, match="not registered"):
            await command(ctx, step_id="check-x", argv=[PY, "-c", "pass"], environment_name="x")

    asyncio.run(scenario())


def test_large_argv_shrinks_the_output_bound_so_the_completion_fits(tmp_path: Path) -> None:
    from cayu.workflows.commands import MAX_COMMAND_OUTPUT_LIMIT_BYTES

    async def scenario() -> None:
        app = _app(tmp_path)
        ctx = _CommandWorkflow(app, "pass").context("run-big-argv")
        await ctx.start()
        script = (
            "import sys; sys.stdout.write('o' * 1_100_000); sys.stderr.write('e' * 1_100_000)"
            "#" + "a" * 100_000
        )
        result = await command(
            ctx,
            step_id="big",
            argv=[PY, "-c", script],
            output_limit_bytes=MAX_COMMAND_OUTPUT_LIMIT_BYTES,
            timeout_s=60,
        )
        assert result.passed
        assert result.stdout_truncated and result.stderr_truncated
        assert len(result.stdout) < MAX_COMMAND_OUTPUT_LIMIT_BYTES
        (completed,) = await _step_events(app, "run-big-argv", EventType.WORKFLOW_STEP_COMPLETED)
        assert completed.payload["exit_code"] == 0

    asyncio.run(scenario())


def test_oversized_argv_is_refused_before_running(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        ctx = _CommandWorkflow(app, "pass").context("run-huge-argv")
        await ctx.start()
        script = "open('ran.txt', 'w').write('x')#" + "a" * 1_100_000
        with pytest.raises(ValueError, match="argv and step metadata"):
            await command(ctx, step_id="huge", argv=[PY, "-c", script])
        assert not (tmp_path / "ran.txt").exists()
        assert await _step_events(app, "run-huge-argv", EventType.WORKFLOW_STEP_STARTED) == []

    asyncio.run(scenario())


def test_sealed_application_refuses_a_new_command(tmp_path: Path) -> None:
    from cayu.runtime.application_lifecycle import ApplicationAdmissionsSealed

    async def scenario() -> None:
        app = _app(tmp_path)
        ctx = _CommandWorkflow(app, "pass").context("run-sealed")
        await ctx.start()
        app.seal_admissions()
        with pytest.raises(ApplicationAdmissionsSealed):
            await command(ctx, step_id="sealed", argv=[PY, "-c", "open('ran.txt', 'w').write('x')"])
        assert not (tmp_path / "ran.txt").exists()
        assert await _step_events(app, "run-sealed", EventType.WORKFLOW_STEP_STARTED) == []

    asyncio.run(scenario())


def test_shutdown_waits_for_a_running_command(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = _app(tmp_path)
        ctx = _CommandWorkflow(app, "pass").context("run-shutdown")
        await ctx.start()
        running = asyncio.create_task(
            command(
                ctx,
                step_id="slow",
                argv=[PY, "-c", "import time; time.sleep(1.5); print('done')"],
                timeout_s=30,
            )
        )
        while app._admission.in_flight == 0:
            assert not running.done()
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        outcome = await app.aclose(timeout_s=20)
        assert running.done()
        result = running.result()
        assert result.passed and result.stdout.strip() == "done"
        assert outcome.open_operations == 0

    asyncio.run(scenario())


def test_command_refuses_a_binding_that_prepares_the_workspace(tmp_path: Path) -> None:
    from cayu.environments import NativeBinding
    from cayu.workspaces import LocalWorkspace

    class CheckoutBinding(NativeBinding):
        """Stands in for a binding whose bind() checks out or copies files."""

    async def scenario() -> None:
        app = CayuApp(enable_logging=False)
        workspace = LocalWorkspace(tmp_path, workspace_id="w")
        for name, binding in (("native", NativeBinding()), ("checkout", CheckoutBinding())):
            app.register_environment(
                Environment(
                    EnvironmentSpec(name=name),
                    workspace=workspace,
                    runner=LocalRunner(tmp_path),
                    binding=binding,
                )
            )
        ctx = _CommandWorkflow(app, "pass").context("run-binding")
        await ctx.start()
        ok = await command(
            ctx, step_id="native", argv=[PY, "-c", "pass"], environment_name="native"
        )
        assert ok.passed
        with pytest.raises(ValueError, match="CheckoutBinding"):
            await command(
                ctx,
                step_id="checkout",
                argv=[PY, "-c", "open('ran.txt', 'w').write('x')"],
                environment_name="checkout",
            )
        assert not (tmp_path / "ran.txt").exists()

    asyncio.run(scenario())
