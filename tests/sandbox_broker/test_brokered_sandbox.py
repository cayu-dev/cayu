"""Brokered sandboxes: the Runtime client of Cloud's sandbox broker, against a fake broker."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.core.test_workspace_mutation_receipts import _SingleToolProvider
from tests.sandbox_broker.fake_broker import API_URL, APPLICATION, TOKEN, FakeBroker

from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import EnvironmentFactoryRequest, EnvironmentSpec
from cayu.environments._ephemeral_git import initialize_ephemeral_git_baseline
from cayu.environments.admission import ExecutionRequirements, evaluate_execution_admission
from cayu.runners import ExecCommand
from cayu.runners.base import runner_workspace_mutation_settlement
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sandbox_broker import (
    BrokeredSandboxCodingBinding,
    BrokeredSandboxCodingEnvironmentFactory,
    BrokeredSandboxRunner,
    BrokeredSandboxWorkspace,
    SandboxAllocationFailedError,
    SandboxBrokerClient,
    SandboxBrokerRefusedError,
    SandboxBrokerSettings,
    SandboxBrokerUnavailableError,
    SandboxLeaseKeeper,
    SandboxLeaseLostError,
    SandboxNotFoundError,
    SandboxQuotaExceededError,
)
from cayu.sandbox_broker import runner as runner_module
from cayu.sandbox_broker.client import MAX_COMMAND_OUTPUT_BYTES
from cayu.storage import SQLiteSessionStore
from cayu.tools import GitChangesTool
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import LocalWorkspace

_ENV = {
    "CAYU_CLOUD_API_URL": API_URL,
    "CAYU_CLOUD_SANDBOX_TOKEN": TOKEN,
    "CAYU_CLOUD_APPLICATION_ID": APPLICATION,
}


def _request(session_id: str = "coding", **options) -> EnvironmentFactoryRequest:
    return EnvironmentFactoryRequest(
        session_id=session_id, agent_name="coder", environment_name="sandbox", **options
    )


def _project(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    (source / "run.sh").write_bytes(b"#!/bin/sh\necho ok\n")
    (source / "run.sh").chmod(0o755)
    return source


def _factory(
    broker: FakeBroker, source: Path, **overrides
) -> BrokeredSandboxCodingEnvironmentFactory:
    options = {
        "broker": broker.client(),
        "source_workspace": LocalWorkspace(source, workspace_id="project"),
        "egress": "public_web",
    }
    options.update(overrides)
    return BrokeredSandboxCodingEnvironmentFactory(**options)


async def _ready_runner(
    broker: FakeBroker, *, egress: str = "none", **keeper_options
) -> BrokeredSandboxRunner:
    client = broker.client()
    created = await client.create(
        idempotency_key=f"k-{len(broker.sandboxes)}",
        image="coding-python",
        egress_mode=egress,
        lease_seconds=60,
    )
    # The lease starts at ready, never during provisioning.
    assert created.status == "provisioning" and created.lease_expires_at is None
    sandbox = await client.wait_until_ready(created.id, timeout_s=5, poll_interval_s=0)
    assert sandbox.lease_expires_at is not None
    keeper = None
    if keeper_options:
        keeper = SandboxLeaseKeeper(client, sandbox, lease_seconds=60, **keeper_options).start()
    return BrokeredSandboxRunner(client, sandbox, lease_keeper=keeper, poll_wait_seconds=1)


# Settings ------------------------------------------------------------------


def test_settings_come_from_the_variables_cloud_injects() -> None:
    settings = SandboxBrokerSettings.from_env(_ENV)
    assert (settings.api_url, settings.application_id) == (API_URL, APPLICATION)
    assert TOKEN not in repr(settings)
    assert TOKEN not in repr(SandboxBrokerClient(settings))
    for name in _ENV:
        with pytest.raises(ValueError, match=name):
            SandboxBrokerSettings.from_env({**_ENV, name: ""})
    with pytest.raises(ValueError, match="HTTPS"):
        SandboxBrokerSettings.from_env({**_ENV, "CAYU_CLOUD_API_URL": "http://cloud.example"})
    # Plain HTTP only for a local broker.
    assert SandboxBrokerSettings.from_env({**_ENV, "CAYU_CLOUD_API_URL": "http://127.0.0.1:8080"})


def test_factory_from_env_requires_an_explicit_egress_mode(tmp_path: Path) -> None:
    source = LocalWorkspace(tmp_path)
    with pytest.raises(ValueError, match="CAYU_CLOUD_SANDBOX_EGRESS"):
        BrokeredSandboxCodingEnvironmentFactory.from_env(source_workspace=source, environ=_ENV)
    factory = BrokeredSandboxCodingEnvironmentFactory.from_env(
        source_workspace=source,
        environ={
            **_ENV,
            "CAYU_CLOUD_SANDBOX_EGRESS": "none",
            "CAYU_CLOUD_SANDBOX_LEASE_SECONDS": "300",
        },
    )
    assert (factory.egress, factory.image, factory.lease_seconds) == ("none", "coding-python", 300)
    with pytest.raises(ValueError, match="CAYU_CLOUD_SANDBOX_EGRESS"):
        BrokeredSandboxCodingEnvironmentFactory.from_env(
            source_workspace=source, environ={**_ENV, "CAYU_CLOUD_SANDBOX_EGRESS": "open"}
        )


# Allocation and use ----------------------------------------------------------


def test_a_session_copies_in_runs_tools_through_the_broker_and_copies_back(
    tmp_path: Path,
) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    class EditTool(Tool):
        spec = ToolSpec(name="edit_project", parallel_safe=False, workspace_mutation=True)

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            assert ctx.workspace is not None
            content = (await ctx.workspace.read_bytes("calc.py")).content
            await ctx.workspace.create_bytes("calc_fixed.py", content.replace(b"a - b", b"a + b"))
            return ToolResult(content="edited")

    async def scenario():
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(
            _SingleToolProvider(tool_name="edit_project", arguments={}), default=True
        )
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )
        app.register_agent(AgentSpec(name="coder", model="scripted-model"), tools=[EditTool()])
        try:
            return [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="coding",
                        agent_name="coder",
                        messages=[Message.text("user", "fix it")],
                    )
                )
            ]
        finally:
            await store.close()

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    (sandbox,) = broker.sandboxes.values()
    # The tool edited the sandbox copy, its edit reached the source, and the
    # sandbox was destroyed when the session ended.
    assert (sandbox.root / "calc.py").exists()
    assert (source / "calc_fixed.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert (source / "calc.py").read_bytes() == b"def add(a, b):\n    return a - b\n"
    assert sandbox.status == "destroyed" and sandbox.destroy_reason == "requested"
    assert sandbox.body["egress"] == {"mode": "public_web"}
    # Copy-in used one archive upload with the executable bit, and copy-back
    # read the changed file with the selective archive read.
    assert [call for call in broker.calls if call[0] == "PUT"] == [
        ("PUT", f"/{sandbox.id}/archive")
    ]
    assert os.access(sandbox.root / "run.sh", os.X_OK)
    assert broker.archive_gets == 1


def test_factory_release_and_binding_destroy_the_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source, renew_every_s=0.05)
        result = await factory.create(_request())
        runner = result.environment.runner
        assert isinstance(runner, BrokeredSandboxRunner)
        assert result.metadata["sandbox_id"] == runner.sandbox_id
        await asyncio.sleep(0.2)
        assert runner.lease_keeper is not None and runner.lease_keeper.renewals >= 2
        bound = await result.environment.binding.bind(
            result.environment.workspace,
            runner,
            session_id="coding",
            agent_name="coder",
            environment_name="sandbox",
        )
        assert (await bound.workspace.read_bytes("calc.py")).content.startswith(b"def add")
        await result.environment.binding.finalize(bound, outcome="interrupted")
        assert runner.is_closed and not runner.lease_keeper.running
        renewals = broker.renewals
        await asyncio.sleep(0.15)
        assert broker.renewals == renewals

    asyncio.run(scenario())
    assert broker.live() == []


def test_completion_recovery_never_copies_back_from_a_replacement(tmp_path: Path) -> None:
    from cayu.environments import EnvironmentFactoryOperation

    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    factory = _factory(broker, source)
    reconnect = {"operation": EnvironmentFactoryOperation.RECONNECT}

    async def scenario():
        first = await factory.create(_request())
        bound = await first.environment.binding.bind(
            first.environment.workspace,
            first.environment.runner,
            session_id="coding",
            agent_name="coder",
            environment_name="sandbox",
        )
        state = first.environment.binding._completion_finalization_recovery_state(bound)
        # The process that bound it is gone: its in-process ownership goes too.
        assert first.environment.binding.abandon(bound)
        await first.release(None)
        # After a crash the runtime asks again with RECONNECT and no metadata,
        # then recovers the binding against whatever it got.
        second = await factory.create(_request(**reconnect))
        with pytest.raises(RuntimeError, match="can't be reattached"):
            await second.environment.binding._recover_completion_finalization(
                second.environment.workspace,
                second.environment.runner,
                session_id="coding",
                agent_name="coder",
                environment_name="sandbox",
                recovery_state=state,
            )
        await second.release(None)
        # Metadata this factory never wrote is refused before allocating.
        with pytest.raises(RuntimeError, match="do not reconnect"):
            await factory.create(_request(**reconnect, reconnect_metadata={"sandbox_id": "x"}))

    asyncio.run(scenario())
    assert sorted(path.name for path in source.iterdir()) == ["calc.py", "run.sh"]
    assert len(broker.sandboxes) == 2 and broker.live() == []


async def _bind(result):
    return await result.environment.binding.bind(
        result.environment.workspace,
        result.environment.runner,
        session_id="coding",
        agent_name="coder",
        environment_name="sandbox",
    )


def test_git_changes_sees_what_the_session_changed_since_copy_in(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        result = await _factory(broker, source).create(_request())
        bound = await _bind(result)
        context = ToolContext(session_id="coding", runner=bound.runner)
        before = await GitChangesTool().run(context, {"mode": "status"})
        await bound.workspace.create_bytes("calc_fixed.py", b"def add(a, b):\n    return a + b\n")
        after = await GitChangesTool().run(context, {"mode": "status"})
        await result.environment.binding.finalize(bound, outcome="completed")
        return before, after

    before, after = asyncio.run(scenario())
    # The copy-in is the baseline, so only the session's own change shows.
    assert not before.is_error and before.structured["changes"] == []
    assert not after.is_error
    assert [change["path"] for change in after.structured["changes"]] == ["calc_fixed.py"]
    # The change reached the source; the sandbox's repository didn't.
    assert (source / "calc_fixed.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert not (source / ".git").exists()
    assert broker.live() == []


def test_the_baseline_fits_an_index_larger_than_one_command_output(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    # `git ls-files --stage -z` lists each of these files in 57 bytes, so the
    # index outgrows what one broker command returns.
    files = MAX_COMMAND_OUTPUT_BYTES // 57 + 1

    async def scenario():
        runner = await _ready_runner(broker)
        (sandbox,) = broker.sandboxes.values()
        for index in range(files):
            (sandbox.root / f"f{index:05}").write_bytes(b"")
        await initialize_ephemeral_git_baseline(runner)
        listed = await runner.exec(
            ExecCommand.process("git", "ls-files", "--stage", "-z"),
            output_limit_bytes=MAX_COMMAND_OUTPUT_BYTES,
        )
        (sandbox.root / "f00000").write_bytes(b"changed\n")
        changed = await GitChangesTool().run(
            ToolContext(session_id="coding", runner=runner), {"mode": "status"}
        )
        await runner.close()
        return listed, changed

    listed, changed = asyncio.run(scenario())
    assert listed.exit_code == 0 and listed.stdout_truncated
    assert not changed.is_error
    assert [change["path"] for change in changed.structured["changes"]] == ["f00000"]


def test_a_git_baseline_requires_source_and_target_to_exclude_git(tmp_path: Path) -> None:
    source = _project(tmp_path)
    (source / ".git").mkdir()
    (source / ".git" / "HEAD").write_bytes(b"ref: refs/heads/main\n")
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        other = await _ready_runner(broker)
        protected = (".git",)
        views = (
            (
                LocalWorkspace(source),
                BrokeredSandboxWorkspace(runner, excluded_directory_names=protected),
                "source",
            ),
            (
                LocalWorkspace(source, excluded_directory_names=protected),
                BrokeredSandboxWorkspace(runner),
                "target",
            ),
            (
                LocalWorkspace(source, excluded_directory_names=protected),
                BrokeredSandboxWorkspace(other, excluded_directory_names=protected),
                "target",
            ),
        )
        for source_view, target, role in views:
            binding = BrokeredSandboxCodingBinding(
                target_workspace=target, max_file_bytes=1024 * 1024
            )
            with pytest.raises(ValueError, match=f"requires a {role} .* that excludes .git"):
                await binding.bind(source_view, runner, session_id="coding")
        await runner.close()
        await other.close()

    asyncio.run(scenario())
    # Refused before copy-in: the source's repository never entered a sandbox.
    assert all(list(sandbox.root.iterdir()) == [] for sandbox in broker.sandboxes.values())
    assert (source / ".git" / "HEAD").read_bytes() == b"ref: refs/heads/main\n"


def test_a_subdirectory_target_gets_its_own_baseline(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        (sandbox,) = broker.sandboxes.values()
        (sandbox.root / "project").mkdir()
        protected = (".git",)
        binding = BrokeredSandboxCodingBinding(
            target_workspace=BrokeredSandboxWorkspace(
                runner, cwd="project", excluded_directory_names=protected
            ),
        )
        bound = await binding.bind(
            LocalWorkspace(source, excluded_directory_names=protected),
            runner,
            session_id="coding",
        )
        await bound.workspace.create_bytes("calc_fixed.py", b"def add(a, b):\n    return a + b\n")
        changed = await GitChangesTool().run(
            ToolContext(session_id="coding", runner=runner, workspace=bound.workspace),
            {"mode": "status"},
        )
        await binding.finalize(bound, outcome="completed")
        return changed

    changed = asyncio.run(scenario())
    assert not changed.is_error
    assert [change["path"] for change in changed.structured["changes"]] == ["calc_fixed.py"]
    # The repository is the target's own, not the sandbox root's.
    (sandbox,) = broker.sandboxes.values()
    assert (sandbox.root / "project" / ".git").is_dir()
    assert not (sandbox.root / ".git").exists()
    assert (source / "calc_fixed.py").read_bytes() == b"def add(a, b):\n    return a + b\n"
    assert not (source / ".git").exists()
    assert broker.live() == []


def test_without_a_git_baseline_the_sandbox_runs_no_git(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        result = await _factory(broker, source, git_baseline=False).create(_request())
        bound = await _bind(result)
        await result.environment.binding.finalize(bound, outcome="completed")

    asyncio.run(scenario())
    (sandbox,) = broker.sandboxes.values()
    assert not (sandbox.root / ".git").exists()
    assert not any(argv[0] == "git" for argv in broker.started)
    assert broker.live() == []


def test_a_baseline_that_fails_fails_the_bind(tmp_path: Path, monkeypatch) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    start = broker._start

    async def without_git(sandbox, body):
        # An image without git: every git command exits nonzero.
        if body["argv"][0] == "git":
            body = {**body, "argv": ["false"]}
        return await start(sandbox, body)

    monkeypatch.setattr(broker, "_start", without_git)

    async def scenario():
        result = await _factory(broker, source).create(_request())
        with pytest.raises(RuntimeError, match="could not establish its ephemeral Git baseline"):
            await _bind(result)
        # Binding never adopted the sandbox, so the factory's release destroys it.
        await result.release(None)

    asyncio.run(scenario())
    assert broker.live() == []


def test_git_baseline_must_be_a_bool(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="git_baseline must be a bool"):
        _factory(FakeBroker(tmp_path / "cloud"), _project(tmp_path), git_baseline="yes")


def test_a_resumed_session_gets_a_fresh_sandbox_each_turn(tmp_path: Path) -> None:
    from cayu import ResumeRequest

    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(_SingleToolProvider(tool_name="noop", arguments={}), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("brokered")),
            _factory(broker, source),
            default=True,
        )

        class Noop(Tool):
            spec = ToolSpec(name="noop")

            async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
                return ToolResult(content="ok")

        app.register_agent(AgentSpec(name="coder", model="scripted-model"), tools=[Noop()])
        try:
            first = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="coding",
                        agent_name="coder",
                        messages=[Message.text("user", "one")],
                    )
                )
            ]
            second = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id="coding", messages=[Message.text("user", "two")])
                )
            ]
            return first, second
        finally:
            await store.close()

    first, second = asyncio.run(scenario())
    assert first[-1].type == EventType.SESSION_COMPLETED, first[-1].payload
    assert second[-1].type == EventType.SESSION_COMPLETED, second[-1].payload
    assert len(broker.sandboxes) == 2 and broker.live() == []


def test_large_trees_move_in_several_archives(tmp_path: Path, monkeypatch) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(runner_module, "MAX_ARCHIVE_BYTES", limit)
    broker = FakeBroker(tmp_path / "cloud")
    broker.archive_limit = limit
    source = tmp_path / "source"
    source.mkdir()
    names = [f"data/part-{index}.bin" for index in range(10)]
    (source / "data").mkdir()
    for index, name in enumerate(names):
        (source / name).write_bytes(bytes([index]) * 20_000)

    async def scenario():
        runner = await _ready_runner(broker)
        workspace = BrokeredSandboxWorkspace(runner)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            for name in names:
                archive.add(source / name, arcname=name)
        data = buffer.getvalue()
        await workspace.write_tar_stream(io.BytesIO(data), archive_bytes=len(data))
        puts = [call for call in broker.calls if call[0] == "PUT"]
        assert len(puts) >= 4
        # Every upload, padding to the tar record included, fit the bound.
        sizes = [len(request.content) for request in broker.requests if request.method == "PUT"]
        assert max(sizes) <= limit
        for name in names:
            assert (broker.sandboxes[runner.sandbox_id].root / name).read_bytes() == (
                source / name
            ).read_bytes()
        # Reading them back splits the batch the broker reports as too large.
        sink = io.BytesIO()
        result = await workspace.read_tar_stream(names, sink, max_file_bytes=limit)
        assert result.archive_bytes == len(sink.getvalue())
        assert broker.archive_gets > 1
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            members = {member.name: member for member in archive}
            assert sorted(members) == sorted(names)
            assert members[names[3]].uname == "cayu.git-mode.v1"
            assert archive.extractfile(names[3]).read() == bytes([3]) * 20_000
        with pytest.raises(FileNotFoundError, match="data/gone.txt"):
            await workspace.read_tar_stream(["data/gone.txt"], io.BytesIO())
        await runner.close()

    asyncio.run(scenario())


def test_copy_back_enforces_the_limits_before_buffering(tmp_path: Path, monkeypatch) -> None:
    limit = 64 * 1024
    monkeypatch.setattr(runner_module, "MAX_ARCHIVE_BYTES", limit)
    broker = FakeBroker(tmp_path / "cloud")
    names = [f"grown-{index}.bin" for index in range(10)]

    async def scenario():
        runner = await _ready_runner(broker)
        root = broker.sandboxes[runner.sandbox_id].root
        for name in names:
            # Files the sandbox grew after the manifest was taken.
            (root / name).write_bytes(b"x" * 20_000)
        workspace = BrokeredSandboxWorkspace(runner)
        # Split batches are streamed one archive at a time, and the read stops
        # at the first member past the total budget.
        broker.archive_limit = limit
        with pytest.raises(RuntimeError, match="max_total_bytes=30000"):
            await workspace.read_tar_stream(names, io.BytesIO(), max_total_bytes=30_000)
        assert broker.archive_gets < len(names)
        # Tar framing never counts as content: files that fit the limits
        # exactly are read even when their combined archive needs a split.
        exact = names[:3]
        broker.archive_gets = 0
        sink = io.BytesIO()
        await workspace.read_tar_stream(
            exact, sink, max_file_bytes=20_000, max_total_bytes=3 * 20_000
        )
        assert broker.archive_gets > 1
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            assert sorted(member.name for member in archive) == sorted(exact)
        # Within one archive, a member over the budget is refused before it is read.
        broker.archive_limit = 8 * 1024 * 1024
        sink = io.BytesIO()
        with pytest.raises(RuntimeError, match="max_total_bytes=30000"):
            await workspace.read_tar_stream(names[:3], sink, max_total_bytes=30_000)
        assert len(sink.getvalue()) < 2 * 20_000
        with pytest.raises(RuntimeError, match="max_file_bytes=1000"):
            await workspace.read_tar_stream(names[:1], io.BytesIO(), max_file_bytes=1000)
        await runner.close()

    asyncio.run(scenario())


def test_paths_keep_their_whitespace(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        root = broker.sandboxes[runner.sandbox_id].root
        (root / "report.txt").write_bytes(b"plain")
        (root / " report.txt ").write_bytes(b"padded")
        (root / " sub ").mkdir()
        (root / "sub").mkdir()
        sink = io.BytesIO()
        await BrokeredSandboxWorkspace(runner).read_tar_stream([" report.txt "], sink)
        with tarfile.open(fileobj=io.BytesIO(sink.getvalue()), mode="r:") as archive:
            assert archive.extractfile(" report.txt ").read() == b"padded"
        result = await runner.exec(ExecCommand.process("pwd"), cwd=" sub ")
        assert result.stdout.rstrip("\n").endswith("/ sub ")
        await runner.close()

    asyncio.run(scenario())


def test_commands_run_asynchronously_in_the_workspace(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        (broker.sandboxes[runner.sandbox_id].root / "sub").mkdir()
        result = await runner.exec(
            ExecCommand.bash("pwd; echo $GREETING"), cwd="sub", env={"GREETING": "hi"}
        )
        assert result.exit_code == 0
        assert result.stdout.splitlines()[-1] == "hi"
        assert result.stdout.splitlines()[0].endswith("/sub")
        # A command outlasting one poll is polled until it finishes; long
        # timeouts are accepted, up to the sandbox's lifetime.
        slow = await runner.exec(ExecCommand.bash("sleep 1.5; echo done"), timeout_s=1800)
        assert slow.stdout.strip() == "done"
        timed_out = await runner.exec(ExecCommand.process("sleep", "5"), timeout_s=1)
        assert timed_out.timed_out and timed_out.exit_code == -1
        # The broker killed the process group, so workspace fencing can settle.
        assert runner_workspace_mutation_settlement(result=timed_out, error=None) == "complete"
        long = await runner.exec(ExecCommand.process("printf", "x" * 3000), output_limit_bytes=10)
        assert long.stdout == "x" * 10 and long.stdout_truncated
        with pytest.raises(ValueError, match="remaining lifetime"):
            await runner.exec(ExecCommand.process("true"), timeout_s=9 * 3600)
        with pytest.raises(ValueError, match="stdin"):
            await runner.exec(ExecCommand.process("cat"), stdin="x" * (1024 * 1024 + 1))
        # Control characters take six bytes each in the response JSON, so the
        # runner asks for enough encoded output to return the raw limit.
        control = await runner.exec(
            ExecCommand.bash("head -c 3000 /dev/zero | tr '\\0' '\\1'"), output_limit_bytes=1000
        )
        assert control.stdout == "\x01" * 1000 and control.stdout_truncated
        started = len(broker.started)
        with pytest.raises(ValueError, match="131071 bytes each"):
            await runner.exec(ExecCommand.process("echo", "x" * 131_072))
        with pytest.raises(ValueError, match="argv exceeds"):
            await runner.exec(ExecCommand.process("echo", *(["x" * 100_000] * 3)))
        assert len(broker.started) == started
        with pytest.raises(ValueError, match="env"):
            await runner.exec(ExecCommand.process("true"), env={"BIG": "x" * (64 * 1024)})
        with pytest.raises(ValueError, match="outside the runner root"):
            await runner.exec(ExecCommand.process("true"), cwd="/etc")
        # A sandbox at its command limit is retried; the command had not started.
        broker.busy_responses = 2
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        await runner.close()

    asyncio.run(scenario())
    assert broker.started[0] == ["/bin/sh", "-c", "pwd; echo $GREETING"]


def test_a_dropped_start_is_recovered_by_replaying_its_id(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        marker = broker.sandboxes[runner.sandbox_id].root / "runs"
        broker.drop_start_responses = 1
        broker.unknown_errors = 1
        result = await runner.exec(ExecCommand.bash("echo run >> runs; echo ok"))
        assert result.stdout.strip() == "ok"
        # The replay returned the started command; it never ran twice.
        assert marker.read_text() == "run\n"
        assert runner.lifecycle_state == "reusable"
        await runner.close()

    asyncio.run(scenario())


@pytest.fixture
def quick_cancel_settlement(monkeypatch):
    monkeypatch.setattr(runner_module, "_CANCEL_SETTLE_SECONDS", 1.0)


async def _cancel(task: asyncio.Task) -> asyncio.CancelledError:
    task.cancel()
    try:
        await task
    except asyncio.CancelledError as cancellation:
        return cancellation
    raise AssertionError("the command was not cancelled")


def _receipts(error: BaseException) -> list[str]:
    return [artifact["status"] for artifact in getattr(error, "artifacts", [])]


def test_a_cancelled_command_is_cancelled_in_the_sandbox(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.3)
        cancellation = await _cancel(task)
        # The broker confirmed the stop: the runner keeps working, and the
        # cancellation carries the receipt workspace fencing settles on.
        assert broker.running() == []
        assert runner.lifecycle_state == "reusable"
        assert _receipts(cancellation) == ["completed"]
        assert runner_workspace_mutation_settlement(result=None, error=cancellation) == "complete"
        assert (await runner.exec(ExecCommand.process("true"))).exit_code == 0
        # Without a confirmed stop, no new work may race the old command.
        broker.cancel_confirms = False
        abandoned = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.3)
        cancellation = await _cancel(abandoned)
        assert runner.lifecycle_state == "poisoned"
        assert _receipts(cancellation) == ["failed"]
        await runner.close()

    asyncio.run(scenario())


def test_a_cancel_overtaking_its_start_waits_for_the_start(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        broker.start_delay = 0.5
        task = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.1)
        # The start is still in flight, so a cancel now would answer 404.
        cancellation = await _cancel(task)
        broker.start_delay = 0.0
        # The runner waited for the start, then cancelled what it started.
        assert len(broker.started) == 1 and broker.running() == []
        assert runner.lifecycle_state == "reusable"
        assert _receipts(cancellation) == ["completed"]
        await runner.close()

    asyncio.run(scenario())


def _lost_then_refused_start(broker: FakeBroker, *, garbled: bool = False) -> SandboxBrokerClient:
    """A transport where the first start applies but its answer is lost, then a 401.

    With ``garbled`` the answer arrives as a 2xx body that isn't JSON, as a
    proxy or ingress might send.
    """

    starts = {"count": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/commands"):
            starts["count"] += 1
            if starts["count"] == 1:
                await broker.handle(request)
                if garbled:
                    return httpx.Response(202, text="<html>proxy</html>")
                raise httpx.ReadError("connection reset", request=request)
            if starts["count"] == 2:
                return httpx.Response(401, text="")
        return await broker.transport(request)

    return SandboxBrokerClient(
        SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
        transport=httpx.MockTransport(handler),
    )


def test_a_refused_replay_never_certifies_a_started_command(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        ready = await _ready_runner(broker)
        # The replay is refused, but the first start already runs the command.
        runner = BrokeredSandboxRunner(_lost_then_refused_start(broker), ready.sandbox)
        with pytest.raises(SandboxBrokerRefusedError):
            await runner.exec(ExecCommand.process("sleep", "30"))
        assert len(broker.started) == 1 and broker.running() == []
        assert runner.lifecycle_state == "reusable"
        # A caller cancelling between the lost start and its replay gets the
        # same treatment: the command is stopped before a receipt is issued.
        cancelled = BrokeredSandboxRunner(_lost_then_refused_start(broker), ready.sandbox)
        task = asyncio.create_task(cancelled.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.1)
        cancellation = await _cancel(task)
        assert len(broker.started) == 2 and broker.running() == []
        assert _receipts(cancellation) == ["completed"]
        # An unparseable 2xx answer means the start was applied, too.
        garbled = BrokeredSandboxRunner(
            _lost_then_refused_start(broker, garbled=True), ready.sandbox
        )
        with pytest.raises(SandboxBrokerRefusedError):
            await garbled.exec(ExecCommand.process("sleep", "30"))
        assert len(broker.started) == 3 and broker.running() == []
        assert garbled.lifecycle_state == "reusable"
        await ready.close()

    asyncio.run(scenario())


def test_waiting_for_a_busy_slot_never_dispatches_after_poisoning(
    tmp_path: Path, quick_cancel_settlement
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        first = asyncio.create_task(runner.exec(ExecCommand.process("sleep", "30")))
        await asyncio.sleep(0.3)
        broker.busy_responses = 1_000
        broker.cancel_confirms = False
        waiting = asyncio.create_task(runner.exec(ExecCommand.process("echo", "after-poison")))
        await asyncio.sleep(0.1)
        await _cancel(first)
        assert runner.lifecycle_state == "poisoned"
        broker.busy_responses = 0
        with pytest.raises(RuntimeError, match="closed"):
            await waiting
        assert ["echo", "after-poison"] not in broker.started
        await runner.close()

    asyncio.run(scenario())


# Admission -----------------------------------------------------------------


def test_untrusted_work_needs_every_guarantee_item_and_no_network(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    untrusted = ExecutionRequirements.untrusted()

    def decision(candidate, *, stage="pre_exposure"):
        return evaluate_execution_admission(
            candidate=candidate.candidate,
            requirements=untrusted,
            evidence=candidate.evidence,
            stage=stage,
        )

    async def scenario():
        request = _request(execution_requirements=untrusted)
        closed = _factory(broker, _project(tmp_path), egress="none")
        pre_create = decision(closed.execution_admission_candidate(request), stage="pre_create")
        assert pre_create.status == "admitted", pre_create.refusals
        runner = await _ready_runner(broker, egress="none")
        admitted = decision(runner.execution_admission_candidate())
        assert admitted.status == "admitted", admitted.refusals
        await runner.close()
        # The open internet is never deny-by-default, before or after allocation.
        open_web = _factory(broker, tmp_path / "source", egress="public_web")
        refused = decision(open_web.execution_admission_candidate(request), stage="pre_create")
        assert {refusal.capability for refusal in refused.refusals} == {"deny_by_default_network"}
        runner = await _ready_runner(broker, egress="public_web")
        refused = decision(runner.execution_admission_candidate())
        assert {refusal.capability for refusal in refused.refusals} == {"deny_by_default_network"}
        await runner.close()
        broker.guarantees = ("untrusted_code_isolation", "egress_only_via_broker")
        weaker = await _ready_runner(broker)
        refused = decision(weaker.execution_admission_candidate())
        assert refused.status == "refused"
        assert {refusal.capability for refusal in refused.refusals} == {
            "real_credential_non_possession",
            "guest_privilege_containment",
        }
        await weaker.close()

    asyncio.run(scenario())


def test_executable_admission_probes_the_exact_sandbox(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        requirements = ExecutionRequirements(required_executables=("cayu-missing-tool", "sh"))
        observer = runner.execution_admission_observer(requirements)
        declared = observer.snapshot()
        assert {item.state for item in declared.evidence.tool_requirements.executables} == {
            "declared"
        }
        candidate = await observer.collect()
        states = {
            item.executable: item.state for item in candidate.evidence.tool_requirements.executables
        }
        assert states == {"sh": "live_verified", "cayu-missing-tool": "unavailable"}
        assert observer.snapshot() == candidate
        await runner.close()

    asyncio.run(scenario())


# Refusals ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "code", "error"),
    [
        (429, "sandbox_quota_exceeded", SandboxQuotaExceededError),
        (403, "scope_required", SandboxBrokerRefusedError),
        (422, "egress_mode_unsupported", SandboxBrokerRefusedError),
        (503, "sandboxes_unavailable", SandboxBrokerRefusedError),
    ],
)
def test_broker_refusals_reach_the_caller_and_leave_nothing(
    tmp_path: Path, status: int, code: str, error: type[Exception]
) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    broker.refuse_create = (status, code)
    with pytest.raises(error) as raised:
        asyncio.run(_factory(broker, _project(tmp_path)).create(_request()))
    assert raised.value.code == code
    if isinstance(raised.value, SandboxQuotaExceededError):
        assert (raised.value.scope, raised.value.limit) == ("agent", 8)
    # A refusal with a contract code is not retried.
    assert [call for call in broker.calls if call[0] == "POST"] == [("POST", "/")]
    assert broker.live() == []


def test_unknown_image_and_failed_allocation_are_refused(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    with pytest.raises(SandboxBrokerRefusedError, match="image_unknown"):
        asyncio.run(_factory(broker, _project(tmp_path), image="gpu").create(_request()))
    broker.fail_allocation = True
    with pytest.raises(SandboxAllocationFailedError, match="allocation_failed"):
        asyncio.run(_factory(broker, tmp_path / "source").create(_request()))


def test_errors_without_a_contract_code_are_judged_by_status(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    failures = {"left": 2}

    async def flaky(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and failures["left"]:
            failures["left"] -= 1
            # The broker applied the create, but the answer was lost.
            await broker.handle(request)
            raise httpx.ReadTimeout("lost", request=request)
        return await broker.transport(request)

    async def scenario():
        client = SandboxBrokerClient(
            SandboxBrokerSettings(api_url=API_URL, application_id=APPLICATION, token=TOKEN),
            transport=httpx.MockTransport(flaky),
        )
        record = await client.create(
            idempotency_key="same", image="coding-python", egress_mode="none", lease_seconds=60
        )
        assert record.id == "sbx_0001"
        broker.unknown_errors = 2
        assert (await client.get(record.id)).id == record.id
        broker.unknown_errors = 10
        with pytest.raises(SandboxBrokerUnavailableError):
            await client.get(record.id)
        broker.unknown_errors = 0
        # Without a contract code, 408, 429 and 5xx are retried ...
        broker.bare_statuses = [429, 408, 503]
        assert (await client.get(record.id)).id == record.id
        # ... and every other 4xx is terminal.
        for status, error in ((401, SandboxBrokerRefusedError), (404, SandboxNotFoundError)):
            broker.bare_statuses = [status]
            before = len(broker.requests)
            with pytest.raises(error) as raised:
                await client.get(record.id)
            assert raised.value.status == status and raised.value.code is None
            assert len(broker.requests) == before + 1

    asyncio.run(scenario())
    assert len(broker.sandboxes) == 1


# Leases ----------------------------------------------------------------------


def test_lease_expiry_stops_the_runner(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    lost: list[SandboxLeaseLostError] = []

    async def scenario():
        runner = await _ready_runner(broker, renew_every_s=0.05, on_lost=lost.append)
        await asyncio.sleep(0.12)
        assert broker.renewals >= 1
        broker.expire(runner.sandbox_id)
        await asyncio.sleep(0.15)
        assert lost and runner.lease_keeper.lost is not None
        with pytest.raises(SandboxLeaseLostError):
            await runner.exec(ExecCommand.process("true"))
        await runner.close()

    asyncio.run(scenario())
    assert len(lost) == 1


def test_the_first_renewal_follows_the_remaining_lease(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta

    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        # Only 0.3 s of a 60 s lease is left when the keeper starts.
        sandbox = runner.sandbox.model_copy(
            update={"lease_expires_at": datetime.now(UTC) + timedelta(seconds=0.3)}
        )
        keeper = SandboxLeaseKeeper(broker.client(), sandbox, lease_seconds=60).start()
        await asyncio.sleep(0.25)
        assert keeper.renewals == 1 and keeper.lost is None
        await keeper.stop()
        await runner.close()

    asyncio.run(scenario())


def test_the_factory_reserves_its_own_labels(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    eight = {f"k{index}": "v" for index in range(8)}

    async def create_and_release():
        result = await _factory(broker, _project(tmp_path), labels=eight).create(_request())
        await result.release(None)

    asyncio.run(create_and_release())
    (sandbox,) = broker.sandboxes.values()
    assert len(sandbox.body["labels"]) == 10
    with pytest.raises(ValueError, match="at most 8"):
        _factory(broker, tmp_path / "source", labels={**eight, "k8": "v"})
    with pytest.raises(ValueError, match="set by the factory"):
        _factory(broker, tmp_path / "source", labels={"cayu.session": "mine"})


def test_a_lapsed_lease_found_by_a_command_closes_the_runner(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker)
        broker.expire(runner.sandbox_id)
        with pytest.raises(SandboxLeaseLostError):
            await runner.exec(ExecCommand.process("true"))
        assert runner.lifecycle_state == "fenced"
        await runner.close()

    asyncio.run(scenario())


# Cleanup ---------------------------------------------------------------------


def test_close_destroys_once_and_a_failed_destroy_leaves_the_lease_to_lapse(
    tmp_path: Path,
) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _ready_runner(broker, renew_every_s=0.05)
        await runner.close()
        await runner.close()
        assert [call for call in broker.calls if call[0] == "DELETE"] == [
            ("DELETE", f"/{runner.sandbox_id}")
        ]
        second = await _ready_runner(broker, renew_every_s=0.05)
        broker.fail_destroy = True
        with pytest.raises(SandboxBrokerUnavailableError):
            await second.close()
        # Renewal stopped, so the broker destroys it when the lease lapses.
        assert not second.lease_keeper.running and not second.is_closed
        broker.fail_destroy = False
        await second.close()
        assert second.is_closed

    asyncio.run(scenario())
    assert broker.live() == []


def test_failed_setup_destroys_the_sandbox(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")
    factory = _factory(broker, _project(tmp_path), setup_commands=["exit 3"])
    with pytest.raises(RuntimeError, match="exit code 3"):
        asyncio.run(factory.create(_request()))
    (sandbox,) = broker.sandboxes.values()
    assert sandbox.status == "destroyed"


def test_the_client_covers_the_read_and_archive_routes(tmp_path: Path) -> None:
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        client = broker.client()
        (image,) = await client.list_images()
        assert image.name == "coding-python" and image.guarantees.has("unprivileged_commands")
        runner = await _ready_runner(broker)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:") as archive:
            info = tarfile.TarInfo("notes/a.txt")
            info.size = 2
            archive.addfile(info, io.BytesIO(b"hi"))
        await runner.put_archive(buffer.getvalue())
        result = await client.get_archive(runner.sandbox_id, paths=["notes", "missing.txt"])
        assert result.missing_paths == ("missing.txt",)
        with tarfile.open(fileobj=io.BytesIO(result.data), mode="r:") as archive:
            assert archive.extractfile("notes/a.txt").read() == b"hi"
        sandboxes, cursor = await client.list_sandboxes(status="ready")
        assert [record.id for record in sandboxes] == [runner.sandbox_id] and cursor is None
        assert await client.events(runner.sandbox_id) == []
        with pytest.raises(ValueError, match="inside /workspace"):
            await client.get_archive(runner.sandbox_id, paths=["../etc"])
        await runner.close()
        await client.aclose()

    asyncio.run(scenario())


# No cloud credentials -------------------------------------------------------------


def test_the_agent_needs_no_cloud_provider_credential(tmp_path: Path, monkeypatch) -> None:
    for name in list(os.environ):
        if name.startswith(("AWS_", "GOOGLE_", "AZURE_", "E2B_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "missing"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing"))
    monkeypatch.setitem(sys.modules, "boto3", None)
    monkeypatch.setitem(sys.modules, "botocore", None)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        result = await _factory(broker, _project(tmp_path)).create(_request())
        await result.release(None)

    asyncio.run(scenario())
    # Only the broker credential ever leaves the Agent.
    assert {request.headers["authorization"] for request in broker.requests} == {f"Bearer {TOKEN}"}
    assert all(request.url.host == "cloud.example.test" for request in broker.requests)
    # Importing the package loads no provider SDK or provider runner.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, cayu.sandbox_broker; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in {'boto3', 'botocore', 'e2b'}"
            " or m == 'cayu.runners.aws_lambda_microvm'))",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert probe.stdout.strip() == "[]", probe.stdout
