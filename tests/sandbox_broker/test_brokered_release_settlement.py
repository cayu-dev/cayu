"""A brokered session whose release can't finish keeps its workspace only while its sandbox lives.

A failed finalize keeps the source workspace reserved for a retry, and a
workspace mutation that never settled keeps the environment fenced. Neither
can settle once the sandbox is destroyed, so both are released then, and not
before.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import SecretStr
from tests.core.test_human_review import identity
from tests.sandbox_broker.fake_broker import FakeBroker, _error

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.environments import EnvironmentLifecyclePolicy, EnvironmentSpec
from cayu.providers.base import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sandbox_broker import (
    BrokeredSandboxCodingBinding,
    BrokeredSandboxCodingEnvironmentFactory,
    BrokeredSandboxRunner,
    SandboxLeaseKeeper,
)
from cayu.sandbox_broker import runner as runner_module
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import LocalWorkspace

if TYPE_CHECKING:
    import httpx


class _EditOncePerSession(ModelProvider):
    name = "edit-once-per-session"

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        if request.messages and request.messages[-1].role == "tool":
            yield ModelStreamEvent.text_delta("done")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        yield ModelStreamEvent.tool_call(id="call-edit", name="edit_project", arguments={})
        yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})


class _EditTool(Tool):
    spec = ToolSpec(name="edit_project", parallel_safe=False, workspace_mutation=True)

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes(f"{ctx.session_id}.txt", b"edited\n")
        return ToolResult(content="edited")


class _RefusingBroker(FakeBroker):
    """Refuses every copy-back read with Cloud's 403 ``scope_required``."""

    refuse_archive_reads = True

    async def handle(self, request: httpx.Request) -> httpx.Response:
        if (
            self.refuse_archive_reads
            and request.method == "GET"
            and request.url.path.endswith("/archive")
        ):
            self.requests.append(request)
            return _error(403, "scope_required")
        return await super().handle(request)


def _project(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_bytes(b"def add(a, b):\n    return a - b\n")
    return source


class _Sessions:
    def __init__(self, tmp_path: Path, broker: FakeBroker, source: Path) -> None:
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        self.store = SQLiteSessionStore(
            tmp_path / "sessions.sqlite",
            public_authority_alias_codec=PublicAuthorityAliasCodec(keyring),
        )
        self.app = CayuApp(
            session_store=self.store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        self.app.register_provider(_EditOncePerSession(), default=True)
        self.app.register_environment_factory(
            EnvironmentSpec(
                name="sandbox",
                execution_profile_identity=identity("brokered"),
                lifecycle_policy=EnvironmentLifecyclePolicy(),
            ),
            BrokeredSandboxCodingEnvironmentFactory(
                broker=broker.client(),
                source_workspace=LocalWorkspace(source, workspace_id="project"),
                egress="none",
            ),
            default=True,
        )
        self.app.register_agent(
            AgentSpec(name="coder", model="scripted-model"), tools=[_EditTool()]
        )

    async def run(self, session_id: str, *, resume: bool = False):
        messages = [Message.text("user", "edit")]
        stream = (
            self.app.resume(ResumeRequest(session_id=session_id, messages=messages))
            if resume
            else self.app.run(
                RunRequest(session_id=session_id, agent_name="coder", messages=messages)
            )
        )
        events = []
        async with asyncio.timeout(30):
            async for event in stream:
                events.append(event)
        return events


def test_a_refused_finalize_holds_the_workspace_only_until_the_sandbox_is_gone(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner_module, "_DESTROYED_POLL_SECONDS", 0.02)
    source = _project(tmp_path)
    broker = _RefusingBroker(tmp_path / "cloud")

    async def scenario():
        sessions = _Sessions(tmp_path, broker, source)
        try:
            first = await sessions.run("first")
            (sandbox,) = broker.live()
            # The sandbox may still take a finalize retry, so the source stays
            # reserved for it and the next session is refused.
            assert not await sessions.app.drain_environment_cleanups(timeout_s=0.3)
            second = await sessions.run("second")
            assert broker.sandboxes[sandbox.id].status == "ready"
            renewals = broker.renewals
            # Its lease lapses unrenewed, Cloud destroys it, and the reservation goes.
            broker.expire(sandbox.id)
            assert await sessions.app.drain_environment_cleanups(timeout_s=5)
            progress = [
                (event.payload["operation"], event.payload["status"])
                for event in await sessions.store.load_events("first")
                if event.type == EventType.ENVIRONMENT_LIFECYCLE_PROGRESS
            ]
            broker.refuse_archive_reads = False
            third = await sessions.run("third")
            # The failed session itself isn't left with a completion it can't finish.
            resumed = await sessions.run("first", resume=True)
            return first, second, third, resumed, renewals, progress
        finally:
            await sessions.store.close()

    first, second, third, resumed, renewals, progress = asyncio.run(scenario())

    assert first[-1].type == EventType.SESSION_FAILED, first[-1].payload
    assert first[-1].payload["binding_finalize_error"]["error_type"] == (
        "SandboxBrokerRefusedError"
    )
    assert second[-1].type == EventType.SESSION_FAILED
    assert "already bound by an active session" in second[-1].payload["error"]
    assert third[-1].type == EventType.SESSION_COMPLETED, third[-1].payload
    assert resumed[-1].type == EventType.SESSION_COMPLETED, resumed[-1].payload
    assert broker.renewals == renewals
    # The retirement is recorded like any settled retained cleanup.
    assert progress[-1] == ("retained_cleanup", "completed")
    assert sorted(path.name for path in source.iterdir()) == ["calc.py", "first.txt", "third.txt"]


def test_a_timed_out_final_observation_fences_only_until_the_sandbox_is_gone(
    tmp_path: Path, monkeypatch
) -> None:
    from cayu.runtime import _environment_lifecycle as lifecycle

    monkeypatch.setattr(runner_module, "_DESTROYED_POLL_SECONDS", 0.02)
    monkeypatch.setattr(lifecycle, "_FINAL_WORKSPACE_OBSERVATION_TIMEOUT_SECONDS", 0.2)
    observe_revision = BrokeredSandboxCodingBinding.observe_revision
    stuck = asyncio.Event()
    observations: list[str] = []

    async def observe(self, bound):
        observations.append(bound.workspace.id)
        # The first session's tool observations answer; its final one hangs.
        if observations == ["brokered-sandbox:first"] * 3:
            await stuck.wait()
        return await observe_revision(self, bound)

    monkeypatch.setattr(BrokeredSandboxCodingBinding, "observe_revision", observe)
    source = _project(tmp_path)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        stuck_event = stuck
        sessions = _Sessions(tmp_path, broker, source)
        try:
            first = await sessions.run("first")
            (sandbox,) = broker.live()
            broker.expire(sandbox.id)
            assert await sessions.app.drain_environment_cleanups(timeout_s=5)
            second = await sessions.run("second")
            return first, second
        finally:
            stuck_event.set()
            await sessions.store.close()

    first, second = asyncio.run(scenario())

    assert first[-1].type == EventType.SESSION_FAILED, first[-1].payload
    assert first[-1].payload["error_type"] == "WorkspaceMutationSettlementError"
    assert second[-1].type == EventType.SESSION_COMPLETED, second[-1].payload
    assert sorted(path.name for path in source.iterdir()) == ["calc.py", "second.txt"]


async def _kept_runner(broker: FakeBroker) -> BrokeredSandboxRunner:
    client = broker.client()
    created = await client.create(
        idempotency_key=f"k-{len(broker.sandboxes)}",
        image="coding-python",
        egress_mode="none",
        lease_seconds=60,
    )
    sandbox = await client.wait_until_ready(created.id, timeout_s=5, poll_interval_s=0)
    keeper = SandboxLeaseKeeper(client, sandbox, lease_seconds=60, renew_every_s=0.05).start()
    return BrokeredSandboxRunner(client, sandbox, lease_keeper=keeper, poll_wait_seconds=1)


def test_a_retained_sandbox_stops_renewing_and_ends_when_cloud_destroys_it(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner_module, "_DESTROYED_POLL_SECONDS", 0.02)
    broker = FakeBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _kept_runner(broker)
        ended = asyncio.create_task(runner._await_destroyed())
        await asyncio.sleep(0.2)
        # Still alive, so a finalize retry could still use it; nothing renews it.
        assert not ended.done() and not runner.lease_keeper.running
        renewals = broker.renewals
        await asyncio.sleep(0.1)
        assert broker.renewals == renewals
        broker.expire(runner.sandbox_id)
        return await asyncio.wait_for(ended, 1)

    reason = asyncio.run(scenario())
    assert "destroyed (lease_expired)" in reason


def test_an_unanswered_lookup_waits_past_any_renewal_that_may_have_landed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(runner_module, "_DESTROYED_POLL_SECONDS", 0.02)
    monkeypatch.setattr(runner_module, "_LAPSE_CLOCK_GRACE_SECONDS", 0.0)
    broker = _RefusingBroker(tmp_path / "cloud")

    async def scenario():
        runner = await _kept_runner(broker)
        await asyncio.sleep(0.12)
        keeper = runner.lease_keeper
        assert broker.renewals >= 1
        # A renewal was sent just now, so the lease may run past what the
        # cached answers say, even though their expiry has gone by.
        assert keeper.latest_possible_expiry > datetime.now(UTC) + timedelta(seconds=50)
        broker.bare_statuses = [403] * 10_000
        lapsed = datetime.now(UTC) - timedelta(seconds=1)
        keeper.lease_expires_at = lapsed
        runner.sandbox = runner.sandbox.model_copy(update={"lease_expires_at": lapsed})
        ended = asyncio.create_task(runner._await_destroyed())
        await asyncio.sleep(0.2)
        assert not ended.done()
        # Once even that renewal would have run out, the lease has lapsed.
        keeper.latest_possible_expiry = lapsed
        return await asyncio.wait_for(ended, 2)

    reason = asyncio.run(scenario())
    assert "lease lapsed unrenewed" in reason


class _SlowRenewalBroker(FakeBroker):
    async def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/renew"):
            await asyncio.sleep(10)
        return await super().handle(request)


def test_the_lease_bound_covers_a_renewal_still_in_flight_when_stopped(tmp_path: Path) -> None:
    broker = _SlowRenewalBroker(tmp_path / "cloud")

    async def scenario():
        client = broker.client()
        created = await client.create(
            idempotency_key="k", image="coding-python", egress_mode="none", lease_seconds=60
        )
        sandbox = await client.wait_until_ready(created.id, timeout_s=5, poll_interval_s=0)
        keeper = SandboxLeaseKeeper(client, sandbox, lease_seconds=60, renew_every_s=30)
        renewal = asyncio.create_task(keeper.renew_now())
        await asyncio.sleep(1.5)
        renewal.cancel()
        stopped_at = datetime.now(UTC)
        with contextlib.suppress(asyncio.CancelledError):
            await renewal
        return keeper.latest_possible_expiry, stopped_at

    bound, stopped_at = asyncio.run(scenario())
    # Whatever the client sent until it stopped may still extend the lease.
    assert bound >= stopped_at + timedelta(seconds=59.5)


def test_only_a_target_inside_the_sandbox_ends_with_it(tmp_path: Path) -> None:
    from cayu.environments import BoundWorkspace
    from cayu.sandbox_broker import BrokeredSandboxWorkspace

    broker = FakeBroker(tmp_path / "cloud")
    binding = BrokeredSandboxCodingBinding(git_baseline=False)

    async def scenario():
        runner = await _kept_runner(broker)
        other = await _kept_runner(broker)
        try:
            inside = binding._retained_release_end(
                BoundWorkspace(workspace=BrokeredSandboxWorkspace(runner), runner=runner)
            )
            assert inside is not None
            inside.close()
            # Destroying the runner's sandbox neither destroys these targets nor
            # stops what may still be changing them.
            for target in (BrokeredSandboxWorkspace(other), LocalWorkspace(tmp_path)):
                bound = BoundWorkspace(workspace=target, runner=runner)
                assert binding._retained_release_end(bound) is None
        finally:
            await runner.close()
            await other.close()

    asyncio.run(scenario())
