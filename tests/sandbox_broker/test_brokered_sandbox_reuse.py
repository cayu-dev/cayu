"""Brokered sandboxes kept between sessions with ``BrokeredSandboxReuse``, against a fake broker."""

from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path

import pytest
from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.core.test_workspace_mutation_receipts import _SingleToolProvider
from tests.sandbox_broker.fake_broker import FakeBroker
from tests.sandbox_broker.test_brokered_sandbox import _project

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.environments import (
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentSpec,
)
from cayu.environments.admission import ExecutionRequirements
from cayu.runners import ExecCommand
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sandbox_broker import (
    BrokeredCodingBackendOptions,
    BrokeredSandboxCodingEnvironmentFactory,
    BrokeredSandboxReuse,
    BrokeredSandboxRunner,
)
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import LocalWorkspace


def _request(session_id: str = "coding", **options) -> EnvironmentFactoryRequest:
    options.setdefault("agent_name", "coder")
    return EnvironmentFactoryRequest(session_id=session_id, environment_name="sandbox", **options)


def _factory(
    broker: FakeBroker, source: Path, **overrides
) -> BrokeredSandboxCodingEnvironmentFactory:
    options = {
        "broker": broker.client(),
        "source_workspace": LocalWorkspace(source, workspace_id="project"),
        "egress": "none",
        "sandbox_excluded_directory_names": ("node_modules",),
        "reuse": BrokeredSandboxReuse(),
    }
    options.update(overrides)
    return BrokeredSandboxCodingEnvironmentFactory(**options)


async def _bind(result, session_id: str = "coding"):
    return await result.environment.binding.bind(
        result.environment.workspace,
        result.environment.runner,
        session_id=session_id,
        agent_name="coder",
        environment_name="sandbox",
    )


async def _session(
    factory, session_id: str, outcome: str = "completed", work=None, **request_options
):
    """Create, bind, run ``work`` on the bound workspace, and finalize one session."""

    result = await factory.create(_request(session_id, **request_options))
    bound = await _bind(result, session_id)
    if work is not None:
        try:
            await work(bound)
        except BaseException:
            # Free the sandbox's workspace for the tests that follow.
            result.environment.binding.abandon(bound)
            await result.release(EnvironmentFactoryReleaseAction.DISCARD)
            raise
    await result.environment.binding.finalize(bound, outcome=outcome)
    return result, bound


async def _shell(runner: BrokeredSandboxRunner, script: str) -> str:
    executed = await runner.exec(ExecCommand.bash(script), cwd="/workspace")
    assert executed.exit_code == 0, executed.stderr
    return executed.stdout


def test_the_next_session_reuses_the_sandbox_with_nothing_of_the_last_one(
    tmp_path: Path,
) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def leave_traces(bound) -> None:
        await bound.workspace.write_bytes("calc.py", b"changed\n")
        await bound.workspace.write_bytes("added.py", b"added\n")
        await bound.workspace.delete("run.sh")
        await _shell(
            bound.runner,
            "mkdir -p node_modules/pkg .cayu empty/dir locked/sub && echo x > node_modules/pkg/i.js"
            " && echo x > .cayu/note && echo x > locked/sub/f && chmod 0555 empty"
            " && chmod 000 locked",
        )

    async def scenario():
        factory = _factory(broker, source)
        first, _ = await _session(factory, "one", outcome="failed", work=leave_traces)
        second = await factory.create(_request("two"))
        bound = await _bind(second, "two")
        listing = await _shell(bound.runner, "find . -path ./.git -prune -o -print | sort")
        calc = (await bound.workspace.read_bytes("calc.py")).content
        await second.environment.binding.finalize(bound, outcome="completed")
        await factory.close_idle_resources()
        return first, second, listing, calc

    first, second, listing, calc = asyncio.run(scenario())
    # One sandbox served both sessions; the second holds exactly the source.
    assert len(broker.sandboxes) == 1 and broker.live() == []
    assert first.metadata["sandbox_id"] == second.metadata["sandbox_id"]
    assert (first.metadata["sandbox_reused"], second.metadata["sandbox_reused"]) == (False, True)
    assert listing.split() == [".", "./calc.py", "./run.sh"]
    assert calc == b"def add(a, b):\n    return a - b\n"
    (sandbox,) = broker.sandboxes.values()
    assert os.access(sandbox.root / "run.sh", os.X_OK)


def test_a_reused_sandbox_skips_allocation_and_setup_commands(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source, setup_commands=("echo setup >> /tmp/setup-runs",))
        await _session(factory, "one")
        creates = broker.calls.count(("POST", "/"))
        await _session(factory, "two")
        await factory.close_idle_resources()
        return creates, broker.calls.count(("POST", "/"))

    first_creates, creates = asyncio.run(scenario())
    assert first_creates == creates == 1
    setups = [argv for argv in broker.started if "setup-runs" in " ".join(argv)]
    assert len(setups) == 1


def test_concurrent_sessions_never_share_a_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        one = await factory.create(_request("one"))
        two = await factory.create(_request("two"))
        assert one.metadata["sandbox_id"] != two.metadata["sandbox_id"]
        bound = await _bind(one, "one")
        await one.environment.binding.finalize(bound, outcome="completed")
        bound = await _bind(two, "two")
        # max_idle=1: the first sandbox already waits, so this one is destroyed.
        await two.environment.binding.finalize(bound, outcome="completed")
        assert [sandbox.id for sandbox in broker.live()] == [one.metadata["sandbox_id"]]
        three = await factory.create(_request("three"))
        four = await factory.create(_request("four"))
        await factory.close_idle_resources()
        return one, three, four

    one, three, four = asyncio.run(scenario())
    assert three.metadata["sandbox_id"] == one.metadata["sandbox_id"]
    assert four.metadata["sandbox_reused"] is False
    assert len(broker.sandboxes) == 3
    # Sessions three and four still hold theirs; closing the factory leaves them be.
    assert len(broker.live()) == 2


def test_an_idle_sandbox_keeps_its_lease_until_it_expires(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(
            broker, source, reuse=BrokeredSandboxReuse(idle_seconds=1), renew_every_s=0.05
        )
        await _session(factory, "one")
        renewals = broker.renewals
        await asyncio.sleep(0.3)
        assert len(broker.live()) == 1 and broker.renewals > renewals
        await asyncio.sleep(1.0)
        assert broker.live() == []
        renewals = broker.renewals
        await asyncio.sleep(0.15)
        assert broker.renewals == renewals
        result = await factory.create(_request("two"))
        assert result.metadata["sandbox_reused"] is False
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())
    assert len(broker.sandboxes) == 2 and broker.live() == []


def test_a_sandbox_gone_while_idle_is_replaced(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        first, _ = await _session(factory, "one")
        broker.expire(first.metadata["sandbox_id"])
        second, _ = await _session(factory, "two")
        await factory.close_idle_resources()
        return first, second

    first, second = asyncio.run(scenario())
    assert second.metadata["sandbox_reused"] is False
    assert first.metadata["sandbox_id"] != second.metadata["sandbox_id"]
    assert broker.live() == []


def test_a_sandbox_near_its_maximum_lifetime_is_not_kept(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        # The fake broker's sandboxes live eight hours.
        factory = _factory(
            broker, source, reuse=BrokeredSandboxReuse(min_lifetime_seconds=9 * 3600)
        )
        await _session(factory, "one")
        assert broker.live() == []

    asyncio.run(scenario())


def test_a_failed_setup_destroys_a_reused_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        await _session(factory, "one")
        result = await factory.create(_request("two"))
        assert result.metadata["sandbox_reused"] is True
        await result.release(EnvironmentFactoryReleaseAction.PRESERVE)
        assert broker.live() == []
        third = await factory.create(_request("three"))
        assert third.metadata["sandbox_reused"] is False
        await third.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())
    assert broker.live() == []


def test_a_closed_factory_keeps_no_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        result = await factory.create(_request("one"))
        bound = await _bind(result, "one")
        await factory.close_idle_resources()
        await result.environment.binding.finalize(bound, outcome="completed")

    asyncio.run(scenario())
    assert len(broker.sandboxes) == 1 and broker.live() == []


def test_an_old_sessions_runner_cant_reach_the_reused_sandbox(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        first, _ = await _session(factory, "one")
        old = first.environment.runner
        assert isinstance(old, BrokeredSandboxRunner) and old.is_closed
        with pytest.raises(RuntimeError, match="closed"):
            await old.exec(ExecCommand.process("true"))
        second = await factory.create(_request("two"))
        # Closing the old runner, as a late release would, destroys nothing.
        await old.close()
        await first.release(EnvironmentFactoryReleaseAction.DISCARD)
        assert len(broker.live()) == 1
        bound = await _bind(second, "two")
        await second.environment.binding.finalize(bound, outcome="completed")
        await factory.close_idle_resources()

    asyncio.run(scenario())
    assert broker.live() == []


def test_unreachable_discard_spares_a_sandbox_a_session_holds(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        first, _ = await _session(factory, "one")
        second = await factory.create(_request("two"))
        assert not await factory.discard_unreachable_allocation(_request("one"), first.metadata)
        assert len(broker.live()) == 1
        bound = await _bind(second, "two")
        await second.environment.binding.finalize(bound, outcome="completed")
        # Now it only waits for reuse, so it goes.
        assert await factory.discard_unreachable_allocation(_request("one"), first.metadata)
        assert broker.live() == []
        third = await factory.create(_request("three"))
        assert third.metadata["sandbox_reused"] is False
        await third.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())


def test_detach_refuses_a_runner_whose_lease_was_lost(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source, renew_every_s=0.05)
        result = await factory.create(_request("one"))
        runner = result.environment.runner
        broker.expire(result.metadata["sandbox_id"])
        await asyncio.sleep(0.2)
        assert runner.lease_keeper.lost is not None
        with pytest.raises(RuntimeError, match="lease"):
            runner.detach()
        assert not runner.is_closed
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())


def test_reuse_is_part_of_the_configuration_and_the_backend_options() -> None:
    with pytest.raises(ValueError, match="max_idle"):
        BrokeredSandboxReuse(max_idle=0)
    # Shorter than the workspace reset may take.
    with pytest.raises(ValueError, match="min_lifetime_seconds"):
        BrokeredSandboxReuse(min_lifetime_seconds=60)
    with pytest.raises(ValueError, match="idle_seconds"):
        BrokeredSandboxReuse(idle_seconds=0.5)
    with pytest.raises(TypeError, match="reuse"):
        BrokeredCodingBackendOptions(reuse=True)


def test_reuse_changes_the_factory_fingerprint(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    plain = _factory(broker, source, reuse=None)
    reused = _factory(broker, source)
    assert plain.execution_profile_identity != reused.execution_profile_identity
    assert (
        reused.with_source_workspace(LocalWorkspace(source, workspace_id="other")).reuse
        == reused.reuse
    )
    with pytest.raises(TypeError, match="reuse"):
        _factory(broker, source, reuse=True)


def test_a_resumed_session_reuses_its_sandbox_through_the_app(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    class Noop(Tool):
        spec = ToolSpec(name="noop")

        async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
            return ToolResult(content="ok")

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
            live_between = len(broker.live())
            second = [
                event
                async for event in app.resume(
                    ResumeRequest(session_id="coding", messages=[Message.text("user", "two")])
                )
            ]
            assert await app.close_idle_environment_resources()
            return first, live_between, second
        finally:
            await store.close()

    first, live_between, second = asyncio.run(scenario())
    assert first[-1].type == EventType.SESSION_COMPLETED, first[-1].payload
    assert second[-1].type == EventType.SESSION_COMPLETED, second[-1].payload
    assert live_between == 1
    assert len(broker.sandboxes) == 1 and broker.live() == []
    (factory_completed,) = [
        event for event in second if event.type == EventType.ENVIRONMENT_FACTORY_COMPLETED
    ]
    phases = [step["phase"] for step in factory_completed.payload["phase_timings"]]
    assert phases == ["sandbox_reuse"]


def test_sandboxes_are_kept_per_agent_and_never_for_untrusted_work(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source, reuse=BrokeredSandboxReuse(max_idle=2))
        first, _ = await _session(factory, "one")
        other, _ = await _session(factory, "two", agent_name="reviewer")
        mine, _ = await _session(factory, "three")
        untrusted = ExecutionRequirements.untrusted(network_access="unrestricted")
        strict = await factory.create(_request("four", execution_requirements=untrusted))
        await strict.release(EnvironmentFactoryReleaseAction.DISCARD)
        await factory.close_idle_resources()
        return first, other, mine, strict

    first, other, mine, strict = asyncio.run(scenario())
    assert other.metadata["sandbox_reused"] is False
    assert other.metadata["sandbox_id"] != first.metadata["sandbox_id"]
    assert mine.metadata["sandbox_id"] == first.metadata["sandbox_id"]
    assert strict.metadata["sandbox_reused"] is False
    assert broker.live() == []


def test_a_sandbox_being_taken_is_spared_by_unreachable_discard(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        first, _ = await _session(factory, "one")
        key = factory._reuse_key(_request("two"))
        assert key is not None
        taken = await factory._take_idle(key)
        assert taken is not None
        assert not await factory.discard_unreachable_allocation(_request("one"), first.metadata)
        assert len(broker.live()) == 1
        sandbox, keeper = taken
        factory._checking_out.discard(sandbox.id)
        await keeper.stop()
        await factory.broker.destroy(sandbox.id)

    asyncio.run(scenario())
    assert broker.live() == []


def test_releasing_idle_sandboxes_skips_one_taken_meanwhile(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source, reuse=BrokeredSandboxReuse(max_idle=2))
        one = await factory.create(_request("one"))
        two = await factory.create(_request("two"))
        for result, session_id in ((one, "one"), (two, "two")):
            bound = await _bind(result, session_id)
            await result.environment.binding.finalize(bound, outcome="completed")
        destroy = factory._destroy_idle

        async def slow_destroy(entry):
            # While the first goes, a session takes the second.
            if entry.sandbox.id == one.metadata["sandbox_id"]:
                taken.append(await factory.create(_request("three")))
            await destroy(entry)

        taken = []
        factory._destroy_idle = slow_destroy
        await factory.release_idle_resources()
        factory._destroy_idle = destroy
        assert taken[0].metadata["sandbox_reused"] is True
        await taken[0].release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())
    assert broker.live() == []


def test_detach_refuses_while_a_cancelled_command_is_settling(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        result = await _factory(broker, source).create(_request("one"))
        runner = result.environment.runner
        runner._cancellations_settling = 1
        with pytest.raises(RuntimeError, match="unknown state"):
            runner.detach()
        runner._cancellations_settling = 0
        await result.release(EnvironmentFactoryReleaseAction.DISCARD)

    asyncio.run(scenario())
    assert broker.live() == []


# Delta copy-in ----------------------------------------------------------------


def _uploads(broker: FakeBroker) -> int:
    return len([call for call in broker.calls if call[0] == "PUT"])


def test_a_reused_sandbox_receives_only_the_files_that_changed(tmp_path: Path) -> None:
    source = _project(tmp_path)
    (source / "pkg").mkdir()
    for index in range(20):
        (source / "pkg" / f"m{index}.py").write_text(f"VALUE = {index}\n")
    broker = FakeBroker(tmp_path / "cloud")

    async def edit(bound) -> None:
        await bound.workspace.write_bytes("pkg/m1.py", b"VALUE = 'session'\n")

    async def scenario():
        factory = _factory(broker, source)
        _, first = await _session(factory, "one", work=edit)
        uploads = _uploads(broker)
        # Copy-back published m1.py, so the sandbox holds it as the source now does.
        (source / "pkg" / "m2.py").write_text("VALUE = 'outside'\n")
        second = await factory.create(_request("two"))
        bound = await _bind(second, "two")
        content = (await bound.workspace.read_bytes("pkg/m2.py")).content
        await second.environment.binding.finalize(bound, outcome="completed")
        await factory.close_idle_resources()
        return first, bound, uploads, content

    first, bound, uploads, content = asyncio.run(scenario())
    assert "copy_in_scope" not in first.metadata["sync_binding"]
    sync = bound.metadata["sync_binding"]
    assert (sync["copy_in_scope"], sync["transferred_files"]) == ("changed", 1)
    assert sync["unchanged_files"] == 21
    assert sync["copied_files"] == 22
    assert content == b"VALUE = 'outside'\n"
    assert _uploads(broker) == uploads + 1
    assert broker.live() == []


def test_a_reset_recopies_kept_files_a_session_tampered_with(tmp_path: Path) -> None:
    source = _project(tmp_path)
    (source / "pkg").mkdir()
    (source / "pkg" / "a.py").write_text("A = 1\n")
    (source / "pkg" / "b.py").write_text("B = 1\n")
    broker = FakeBroker(tmp_path / "cloud")

    async def tamper(bound) -> None:
        await _shell(
            bound.runner,
            # A read-only file, a second link to another, and a directory swapped for a
            # symlink to an identical copy outside the workspace.
            "chmod 0444 calc.py && ln run.sh ../run-link-$$ && cp -r pkg ../pkg-copy-$$"
            " && rm -rf pkg && ln -s ../pkg-copy-$$ pkg"
            # A module that would shadow the reset's own imports.
            " && : > json.py",
        )

    async def scenario():
        factory = _factory(broker, source)
        await _session(factory, "one", outcome="failed", work=tamper)
        second = await factory.create(_request("two"))
        bound = await _bind(second, "two")
        await _shell(bound.runner, "test ! -e json.py")
        listing = await _shell(
            bound.runner,
            "python3 -c 'import os, stat, sys\n"
            "for p in sys.argv[1:]:\n"
            "    i = os.lstat(p)\n"
            "    d = stat.S_ISDIR(i.st_mode)\n"
            "    print(p, oct(i.st_mode & 0o777)[2:], 0 if d else i.st_nlink)'"
            " calc.py run.sh pkg pkg/a.py pkg/b.py",
        )
        await bound.workspace.write_bytes("calc.py", b"writable\n")
        await second.environment.binding.finalize(bound, outcome="failed")
        await factory.close_idle_resources()
        return bound, listing

    bound, listing = asyncio.run(scenario())
    assert listing.splitlines() == [
        "calc.py 644 1",
        "run.sh 755 1",
        "pkg 755 0",
        "pkg/a.py 644 1",
        "pkg/b.py 644 1",
    ]
    # Whatever the target could still vouch for was kept, and everything else crossed.
    sync = bound.metadata["sync_binding"]
    assert sync["unchanged_files"] + sync["transferred_files"] == 4
    assert sync["transferred_files"] >= 3


def test_a_kept_file_gets_back_the_permissions_copy_in_gives_it(tmp_path: Path) -> None:
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        factory = _factory(broker, source)
        await _session(
            factory,
            "one",
            outcome="failed",
            work=lambda bound: _shell(bound.runner, "chmod 0444 calc.py && chmod 0700 run.sh"),
        )
        _, bound = await _session(
            factory,
            "two",
            work=lambda bound: bound.workspace.write_bytes("calc.py", b"writable\n"),
        )
        await factory.close_idle_resources()
        return bound

    sync = asyncio.run(scenario()).metadata["sync_binding"]
    assert (sync["unchanged_files"], sync["transferred_files"]) == (2, 0)
    assert (source / "calc.py").read_bytes() == b"writable\n"
    assert broker.live() == []


def test_kept_paths_too_many_for_one_command_are_all_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cayu.sandbox_broker import coding

    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")
    monkeypatch.setattr(
        coding, "MAX_COMMAND_STDIN_BYTES", coding._WORKSPACE_RESET_STDIN_MARGIN_BYTES
    )

    async def scenario():
        factory = _factory(broker, source)
        await _session(factory, "one")
        _, bound = await _session(factory, "two")
        await factory.close_idle_resources()
        return bound

    sync = asyncio.run(scenario()).metadata["sync_binding"]
    assert (sync["transferred_files"], sync["unchanged_files"]) == (2, 0)
    assert broker.live() == []
