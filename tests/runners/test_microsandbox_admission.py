from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import suppress
from types import SimpleNamespace

import pytest

import cayu.runners.microsandbox as microsandbox
from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    EventType,
    ExecutionRequirements,
    ExecutionToolRequirement,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SearchTextTool,
    Tool,
    ToolExecutableRequirement,
    ToolExecutionRequirement,
    ToolSpec,
)
from cayu.environments import BoundWorkspace, WorkspaceBinding
from cayu.environments.factory import environment_factory_cleanup_settlement_tasks
from cayu.runners import ExecCommand, MicrosandboxRunner


class Guest:
    created_at = 123.0
    registry_name = "search-admission"
    running_name = "search-admission"
    config_json = '{"image":"fixture@sha256:123"}'

    def __init__(self, code=0):
        self.code = code
        self.calls = []

    @property
    def name(self):
        async def read_name():
            return self.running_name

        return read_name()

    async def exec_stream(self, program, arguments, **kwargs):
        self.calls.append((program, arguments))

        async def events():
            if program == "rg":
                yield SimpleNamespace(event_type="stdout", data=b"src/example.py\x001\x1fneedle\n")
            yield SimpleNamespace(event_type="exited", code=self.code)

        return events()


def sdk(guest):
    async def get(name):
        return SimpleNamespace(
            name=guest.registry_name, created_at=guest.created_at, config_json=guest.config_json
        )

    return SimpleNamespace(Sandbox=SimpleNamespace(get=get))


@pytest.mark.skipif(os.name == "nt", reason="POSIX guest probe semantics")
@pytest.mark.parametrize("backend", ["microsandbox", "docker"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("proof", ["builtin", "file", "directory", "absolute", "relative", "empty"])
def test_public_admission_requires_an_executable_not_only_a_shell_builtin(
    tmp_path, monkeypatch, backend, proof, explicit
):
    import cayu.runners.docker as docker_module
    from cayu.runners import DockerRunner, ExecResult

    bin_dir = tmp_path / ("bin" if proof == "absolute" else "bin with spaces")
    bin_dir.mkdir()
    executable = "echo"
    search_path = str(bin_dir)
    if proof == "directory":
        (bin_dir / executable).mkdir()
    elif proof != "builtin":
        (bin_dir / executable).symlink_to("/bin/true")
    if proof == "absolute":
        executable = str(bin_dir / executable)
    elif proof == "relative":
        search_path = bin_dir.name
    elif proof == "empty":
        search_path = ""
        (tmp_path / executable).symlink_to("/bin/true")
    admitted = proof not in {"builtin", "directory"}

    def execute_lookup(arguments):
        return subprocess.run(
            ["/bin/sh", *arguments],
            env={"PATH": search_path},
            cwd=tmp_path,
            capture_output=True,
            timeout=2,
            check=False,
        )

    class BuiltinTool(Tool):
        spec = ToolSpec(
            name="needs_program",
            execution_requirements=(
                ToolExecutionRequirement(
                    name="program",
                    alternatives=(
                        ToolExecutableRequirement(
                            executable=executable,
                            probe_arguments=() if explicit else None,
                        ),
                    ),
                ),
            ),
        )

        async def run(self, ctx, args):
            raise AssertionError("Admission must refuse before any tool dispatch")

    class ShellGuest(Guest):
        async def exec_stream(self, program, arguments, **kwargs):
            # Execute the actual generated lookup, with no external programs in
            # the negative fixture PATH. This is not live backend coverage.
            if program == "sh":
                result = execute_lookup(arguments)
            else:
                try:
                    result = subprocess.run(
                        [program, *arguments],
                        env={"PATH": search_path},
                        cwd=tmp_path,
                        capture_output=True,
                        timeout=2,
                        check=False,
                    )
                except (FileNotFoundError, PermissionError):
                    result = SimpleNamespace(returncode=127)

            async def events():
                yield SimpleNamespace(event_type="exited", code=result.returncode)

            return events()

    async def exercise():
        guest = ShellGuest()
        runner = MicrosandboxRunner(guest, name="search-admission", sandbox_module=sdk(guest))
        if backend == "docker":
            container_id = "a" * 64

            async def inspect(*args, **kwargs):
                return {
                    "Id": container_id,
                    "Image": "sha256:" + "b" * 64,
                    "State": {"Running": True},
                }

            async def probe(*args, script, argv=(), **kwargs):
                result = execute_lookup(["-c", script, *argv])
                return ExecResult(exit_code=result.returncode)

            monkeypatch.setattr(docker_module, "_inspect_strict_container", inspect)
            monkeypatch.setattr(docker_module, "_run_docker_admission_probe", probe)
            runner = DockerRunner(
                "probe-semantics", _container_id=container_id, close_action="none"
            )
        provider = ScriptedModelProvider([[ModelStreamEvent.completed({"finish_reason": "stop"})]])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="guest"), runner=runner), default=True
        )
        app.register_agent(AgentSpec(name="agent", model="fake"), tools=[BuiltinTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="builtin-is-not-executable",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        assert len(provider.requests) == int(admitted)
        assert any(event.type is EventType.SESSION_FAILED for event in events) is not admitted
        assert any(event.type is EventType.SESSION_COMPLETED for event in events) is admitted
        if not admitted:
            assert not any(str(event.type).startswith("model.") for event in events)
            failure = next(event for event in events if event.type is EventType.SESSION_FAILED)
            assert any(
                refusal["tool_name"] == "needs_program" and refusal["executable"] == executable
                for refusal in failure.payload["execution_admission"]["refusals"]
            ), failure.payload["execution_admission"]["refusals"]

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "proof", ["present", "missing", "unknown_identity", "wrong_name", "boolean_incarnation"]
)
def test_public_search_text_admission_observes_exact_microsandbox_guest(proof):
    guest = Guest(1 if proof == "missing" else 0)
    if proof == "unknown_identity":
        guest.created_at = None
    elif proof == "wrong_name":
        guest.registry_name = "another-allocation"
    elif proof == "boolean_incarnation":
        guest.created_at = True
    runner = MicrosandboxRunner(guest, name="search-admission", sandbox_module=sdk(guest))

    async def exercise():
        provider = ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="search",
                        name="search_text",
                        arguments={"pattern": "needle", "mode": "content"},
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="guest"), runner=runner), default=True
        )
        app.register_agent(AgentSpec(name="agent", model="fake"), tools=[SearchTextTool()])
        events = [
            event
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id=f"msb_{proof}",
                    messages=[Message.text("user", "run")],
                )
            )
        ]
        return events, provider

    events, provider = asyncio.run(exercise())
    assert len(provider.requests) == (2 if proof == "present" else 0)
    if proof == "present":
        assert any(program == "rg" for program, _ in guest.calls)
        completed = next(event for event in events if event.type is EventType.TOOL_CALL_COMPLETED)
        assert "src/example.py" in str(completed.payload)
    if proof != "present":
        assert EventType.SESSION_COMPLETED not in {event.type for event in events}
        assert EventType.SESSION_FAILED in {event.type for event in events}
    if proof in {"unknown_identity", "wrong_name", "boolean_incarnation"}:
        assert guest.calls == []
    else:
        assert guest.calls[0][0] == "sh"
        assert guest.calls[0][1][-1] == "rg"


@pytest.mark.parametrize(
    "lookup,phase", [("name", 1), ("name", 2), ("get", 1), ("get", 2), ("probe", 1)]
)
@pytest.mark.parametrize(
    "failure,bound",
    [
        ("timeout", False),
        ("cancel", False),
        ("timeout", True),
        ("cancel", True),
        ("cancel_settlement", True),
    ],
)
def test_public_identity_lookup_deadline_retains_pending_sdk_read(
    monkeypatch, lookup, phase, failure, bound
):
    from cayu.runtime import _environment_lifecycle as lifecycle

    monkeypatch.setattr(lifecycle, "_PRE_EXPOSURE_ADMISSION_SETTLEMENT_TIMEOUT_SECONDS", 0.03)
    monkeypatch.setattr(microsandbox, "MICROSANDBOX_ADMISSION_PROBE_TIMEOUT_SECONDS", 0.05)

    async def exercise():
        dispatched = asyncio.Event()
        finish = asyncio.Event()
        settlement_entered = asyncio.Event()
        if failure == "cancel_settlement":
            original_wait = lifecycle.EnvironmentLifecycle._await_pre_exposure_admission_settlement

            async def observe_settlement(self, session_id):
                settlement_entered.set()
                return await original_wait(self, session_id)

            monkeypatch.setattr(
                lifecycle.EnvironmentLifecycle,
                "_await_pre_exposure_admission_settlement",
                observe_settlement,
            )
        calls = 0
        finalized = []

        class Binding(WorkspaceBinding):
            async def bind(self, workspace, runner, **kwargs):
                return BoundWorkspace(
                    workspace=workspace, source_workspace=workspace, runner=runner
                )

            async def finalize(self, bound, **kwargs):
                finalized.append("released")

        async def barrier():
            nonlocal calls
            calls += 1
            if calls == phase:
                dispatched.set()
                await finish.wait()

        class IdentityGuest(Guest):
            async def exec_stream(self, program, arguments, **kwargs):
                if lookup == "probe":
                    await barrier()
                return await super().exec_stream(program, arguments, **kwargs)

            async def stop_and_wait(self):
                # Stop acknowledgement cannot settle a still-blocked SDK call.
                return (0, True)

            @property
            def name(self):
                async def read():
                    if lookup == "name":
                        await barrier()
                    return self.running_name

                return read()

        guest = IdentityGuest()
        module = sdk(guest)
        original_get = module.Sandbox.get

        async def get(name):
            if lookup == "get":
                await barrier()
            return await original_get(name)

        module.Sandbox.get = get
        runner = MicrosandboxRunner(guest, name="search-admission", sandbox_module=module)
        provider = ScriptedModelProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="guest"), runner=runner, binding=Binding() if bound else None
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="agent", model="fake"), tools=[SearchTextTool()])
        events = []

        async def run():
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id="identity_deadline",
                    messages=[Message.text("user", "run")],
                )
            ):
                events.append(event)

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(dispatched.wait(), 5)
            if failure == "cancel_settlement":
                await asyncio.wait_for(settlement_entered.wait(), 5)
            if failure in {"cancel", "cancel_settlement"}:
                task.cancel("cancel identity lookup")
                assert task.cancelling() == 1
            # Observe bounded return without wait_for injecting another cancel
            # or waiting forever for a cancellation-suppressing implementation.
            done, _ = await asyncio.wait((task,), timeout=2)
            assert task in done, "Admission must return while the SDK barrier remains closed"
            if failure in {"cancel", "cancel_settlement"}:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()
                assert task.cancelling() == 1
            else:
                await task
                assert EventType.SESSION_FAILED in {event.type for event in events}
            assert provider.requests == []
            assert EventType.SESSION_COMPLETED not in {event.type for event in events}
            persisted = await app.session_store.load_events("identity_deadline")
            assert EventType.SESSION_COMPLETED not in {event.type for event in persisted}
            # Cancellation during failure finalization must not rewrite the
            # already-committed failed interaction as a new interruption.
            expected_event = {
                "timeout": EventType.SESSION_FAILED,
                "cancel": EventType.SESSION_INTERRUPTED,
                "cancel_settlement": EventType.INTERACTION_FAILED,
            }[failure]
            assert expected_event in {event.type for event in persisted}
            assert finalized == []
            if bound:
                owner = app._environment_lifecycle._active_environment_setups["identity_deadline"]
                assert owner.admission_settlement_task is not None
                assert not owner.admission_settlement_task.done()
                assert owner.cleanup_ready_for_retry
                with pytest.raises(RuntimeError, match="incomplete environment cleanup"):
                    app._environment_lifecycle._require_no_retained_cleanup_for_session(
                        "identity_deadline"
                    )
            assert not await app.drain_environment_cleanups(timeout_s=0.01)
            with pytest.raises(RuntimeError, match="still pending"):
                await MicrosandboxRunner.from_existing(runner.name, sandbox_module=module)
            assert len(guest.calls) == (0 if phase == 1 else 1)
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)
            assert await app.drain_environment_cleanups(timeout_s=5)
            assert finalized == (["released"] if bound else [])
            assert "identity_deadline" not in app._environment_lifecycle._active_environment_setups

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["cancel", "timeout", "transport", "child_cancel"])
def test_microsandbox_probe_retains_settlement_and_fences_peer_dispatch(monkeypatch, failure):
    monkeypatch.setattr(microsandbox, "MICROSANDBOX_ADMISSION_PROBE_TIMEOUT_SECONDS", 0.03)

    async def exercise():
        dispatched = asyncio.Event()
        allow_guest = asyncio.Event()
        stopping = asyncio.Event()
        allow_stop = asyncio.Event()

        class BlockedGuest(Guest):
            blocking = True

            async def exec_stream(self, program, arguments, **kwargs):
                if not self.blocking:
                    return await super().exec_stream(program, arguments, **kwargs)
                dispatched.set()
                if failure == "transport":
                    raise ConnectionError("connection lost after dispatch")
                if failure == "child_cancel":
                    raise asyncio.CancelledError("SDK cancelled its operation")
                await allow_guest.wait()
                return await super().exec_stream(program, arguments, **kwargs)

            async def stop_and_wait(self):
                stopping.set()
                await allow_stop.wait()
                return (0, True)

        guest = BlockedGuest()
        guest.registry_name = f"probe-{failure}"
        guest.running_name = f"probe-{failure}"
        runner = MicrosandboxRunner(guest, name=f"probe-{failure}", sandbox_module=sdk(guest))
        peer = MicrosandboxRunner(guest, name=runner.name, sandbox_module=SimpleNamespace())
        requirements = ExecutionRequirements(
            tool_requirements=(
                ExecutionToolRequirement(
                    tool_name="search_text",
                    requirement=SearchTextTool().spec.execution_requirements[0],
                ),
            )
        )
        task = asyncio.create_task(runner.execution_admission_observer(requirements).collect())
        await asyncio.wait_for(dispatched.wait(), timeout=2)
        if failure == "cancel":
            task.cancel("cancel guest admission")
            assert task.cancelling() == 1
        try:
            await task
        except asyncio.CancelledError as error:
            assert failure == "cancel"
            assert task.cancelled()
            assert task.cancelling() == 1
            owners = environment_factory_cleanup_settlement_tasks(error)
        except Exception as error:
            assert failure != "cancel"
            owners = environment_factory_cleanup_settlement_tasks(error)
        else:
            raise AssertionError("Probe must not return successful evidence.")
        assert len(owners) == 1
        await asyncio.wait_for(stopping.wait(), timeout=2)
        with pytest.raises(RuntimeError, match="still pending"):
            await peer.exec(ExecCommand.process("true"))
        with pytest.raises(RuntimeError, match="still pending"):
            await MicrosandboxRunner.from_existing(runner.name, sandbox_module=SimpleNamespace())
        allow_stop.set()
        await asyncio.sleep(0)
        if failure not in {"transport", "child_cancel"}:
            assert not owners[0].done()
        allow_guest.set()
        await asyncio.wait_for(owners[0], timeout=2)
        guest.blocking = False
        # Both the stop and delayed original dispatch are now terminal.
        await asyncio.sleep(0)
        result = await peer.exec(ExecCommand.process("true"))
        assert result.exit_code == 0

    asyncio.run(exercise())


@pytest.mark.parametrize(
    "failure,binding_enabled",
    [
        (failure, True)
        for failure in (
            "cancel",
            "timeout",
            "transport",
            "stop_retry",
            "stop_drain",
            "cancel_cleanup",
            "cancel_cleanup_twice",
            "valid_invalid",
            "invalid_valid",
            "invalid_only",
            "conflicting_exits",
        )
    ]
    + [(failure, False) for failure in ("stop_drain", "cancel", "timeout", "transport")],
)
def test_public_run_waits_for_guest_probe_settlement_before_binding_release(
    monkeypatch, failure, binding_enabled
):
    monkeypatch.setattr(microsandbox, "MICROSANDBOX_ADMISSION_PROBE_TIMEOUT_SECONDS", 0.03)

    async def exercise():
        dispatched = asyncio.Event()
        allow_guest = asyncio.Event()
        stopping = asyncio.Event()
        allow_stop = asyncio.Event()
        finalized = []
        events = []
        guest_events = []
        stop_calls = []

        class BlockedGuest(Guest):
            async def exec_stream(self, program, arguments, **kwargs):
                dispatched.set()
                if failure in {
                    "transport",
                    "stop_retry",
                    "stop_drain",
                    "cancel_cleanup",
                    "cancel_cleanup_twice",
                }:
                    raise ConnectionError("guest dispatch acknowledgement lost")
                invalid_exits = {
                    "valid_invalid": (0, True),
                    "invalid_valid": (True, 0),
                    "invalid_only": (True,),
                    "conflicting_exits": (1, 0),
                }
                if failure in invalid_exits:

                    async def events():
                        for code in invalid_exits[failure]:
                            yield SimpleNamespace(event_type="exited", code=code)

                    return events()
                try:
                    await allow_guest.wait()
                except BaseException as error:
                    guest_events.append(type(error).__name__)
                    raise
                guest_events.append("guest_returned")
                return await super().exec_stream(program, arguments, **kwargs)

            async def stop_and_wait(self):
                stop_calls.append("stop")
                if failure == "stop_retry" and len(stop_calls) == 1:
                    raise ConnectionError("guest stop acknowledgement lost")
                stopping.set()
                if failure == "stop_drain" and not allow_stop.is_set():
                    raise RuntimeError("guest stop requires operator recovery")
                await allow_stop.wait()
                return (0, True)

        class Binding(WorkspaceBinding):
            async def bind(self, workspace, runner, **kwargs):
                return BoundWorkspace(
                    workspace=workspace, source_workspace=workspace, runner=runner
                )

            async def finalize(self, bound, **kwargs):
                finalized.append("released")
                return None

        class CallerOwnedRunner(MicrosandboxRunner):
            async def close(self):
                raise AssertionError("Admission cleanup must not close a caller-owned runner.")

        guest = BlockedGuest()
        runner = CallerOwnedRunner(guest, name="search-admission", sandbox_module=sdk(guest))
        provider = ScriptedModelProvider([ModelStreamEvent.completed({"finish_reason": "stop"})])
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="guest"),
                runner=runner,
                binding=Binding() if binding_enabled else None,
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="agent", model="fake"), tools=[SearchTextTool()])

        async def run():
            async for event in app.run(
                RunRequest(
                    agent_name="agent",
                    session_id=f"public_msb_{failure}",
                    messages=[Message.text("user", "run")],
                )
            ):
                events.append(event)

        task = asyncio.create_task(run())
        try:
            await asyncio.wait_for(dispatched.wait(), timeout=5)
            if failure == "cancel":
                task.cancel("cancel public guest admission")
                assert task.cancelling() == 1
            await asyncio.wait_for(stopping.wait(), timeout=5)
            if failure == "stop_drain":
                done, _ = await asyncio.wait((task,), timeout=5)
                assert task in done
                result = (await asyncio.gather(task, return_exceptions=True))[0]
                assert result is None or isinstance(result, Exception)
                assert not task.cancelled()
                assert finalized == []
                assert provider.requests == []
                assert not await app.drain_environment_cleanups(timeout_s=0.05)
                assert finalized == []
                with pytest.raises(RuntimeError, match="still pending"):
                    await MicrosandboxRunner.from_existing(runner.name, sandbox_module=sdk(guest))
                allow_stop.set()
                assert await app.drain_environment_cleanups(timeout_s=5)
                assert finalized == (["released"] if binding_enabled else [])
                assert EventType.SESSION_COMPLETED not in {event.type for event in events}
                return
            if failure in {"cancel_cleanup", "cancel_cleanup_twice"}:
                task.cancel("cancel while probe cleanup is pending")
                assert task.cancelling() == 1
                if failure == "cancel_cleanup_twice":
                    await asyncio.sleep(0)
                    task.cancel("new cancellation while probe cleanup is pending")
            assert provider.requests == []
            assert finalized == []
            if binding_enabled:
                assert not task.done()
            else:
                assert not await app.drain_environment_cleanups(timeout_s=0.01)
            with pytest.raises(RuntimeError, match="still pending"):
                await MicrosandboxRunner.from_existing(runner.name, sandbox_module=sdk(guest))
            allow_stop.set()
            await asyncio.sleep(0.02)
            if failure in {"cancel", "timeout"}:
                assert finalized == [], guest_events
                if binding_enabled:
                    assert not task.done()
                else:
                    assert not await app.drain_environment_cleanups(timeout_s=0.01)
            allow_guest.set()
            if failure in {"cancel", "cancel_cleanup", "cancel_cleanup_twice"}:
                with pytest.raises(asyncio.CancelledError) as cancelled:
                    await asyncio.wait_for(task, timeout=5)
                assert task.cancelled()
                assert task.cancelling() == (2 if failure == "cancel_cleanup_twice" else 1)
                assert cancelled.value.args == (
                    "cancel public guest admission"
                    if failure == "cancel"
                    else "cancel while probe cleanup is pending",
                )
            else:
                await asyncio.wait_for(task, timeout=5)
                assert EventType.SESSION_FAILED in {event.type for event in events}
            assert await app.drain_environment_cleanups(timeout_s=5)
            assert finalized == (["released"] if binding_enabled else [])
            assert EventType.SESSION_COMPLETED not in {event.type for event in events}
            assert provider.requests == []
            assert len(stop_calls) == (2 if failure == "stop_retry" else 1)
        finally:
            allow_stop.set()
            allow_guest.set()
            with suppress(asyncio.CancelledError, Exception):
                await asyncio.wait_for(task, timeout=5)
            if failure == "stop_drain" or not binding_enabled:
                await app.drain_environment_cleanups(timeout_s=5)

    asyncio.run(exercise())
