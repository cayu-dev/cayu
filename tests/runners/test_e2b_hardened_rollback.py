from __future__ import annotations

import asyncio

import pytest
from tests.runners.test_e2b import (
    FakeAsyncSandbox,
    FakeE2BModule,
    FakeSandbox,
    FakeSandboxInfo,
    FakeSandboxPaginator,
    FakeSandboxQuery,
    reset_fake_e2b,
)

from cayu.runners import E2BRunner, ExecCommand


@pytest.mark.anyio
@pytest.mark.parametrize("entrance", ["acknowledged", "late", "metadata", "both"])
@pytest.mark.parametrize("deletion_fails", [False, True])
async def test_hardened_rollback_remains_drainable(entrance, deletion_fails):
    reset_fake_e2b()
    sandbox = FakeSandbox("owned-hardened")
    allocated = asyncio.Event()
    acknowledge = asyncio.Event()
    deleting = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    repaired = False
    retained = []
    visible = entrance in {"metadata", "both"}

    async def kill():
        nonlocal calls, visible
        calls += 1
        deleting.set()
        await release.wait()
        if deletion_fails and not repaired:
            raise PermissionError("deletion denied")
        visible = False
        return True

    sandbox.kill = kill
    FakeAsyncSandbox.next_sandbox = sandbox

    class LateSandbox:
        @classmethod
        async def create(cls, **kwargs):
            allocated.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if entrance == "metadata":
                    raise
                await acknowledge.wait()
                return sandbox

        @classmethod
        def list(cls, **kwargs):
            return FakeSandboxPaginator(
                [FakeSandboxInfo(sandbox.sandbox_id, {})] if visible else []
            )

        @classmethod
        async def kill(cls, sandbox_id, **kwargs):
            assert sandbox_id == sandbox.sandbox_id
            return await kill()

    class LateModule:
        AsyncSandbox = LateSandbox
        SandboxQuery = FakeSandboxQuery

    async def setup(runner):
        retained.append(runner)
        raise RuntimeError("setup failed after allocation")

    task = asyncio.create_task(
        E2BRunner.create_hardened(
            guest_setup=setup,
            cleanup_timeout_s=0.03,
            e2b_module=FakeE2BModule if entrance == "acknowledged" else LateModule,
        )
    )
    try:
        if entrance != "acknowledged":
            await asyncio.wait_for(allocated.wait(), 1)
            task.cancel("caller abandoned allocation")
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()
            acknowledge.set()
        await asyncio.wait_for(deleting.wait(), 1)
        if entrance == "acknowledged":
            with pytest.raises(ExceptionGroup) as caught:
                await task
            assert isinstance(caught.value.exceptions[0], RuntimeError)
            assert isinstance(caught.value.exceptions[1], TimeoutError)
            with pytest.raises(RuntimeError):
                await retained[0].exec(ExecCommand.process("true"))
        else:
            # The late handoff observer also expires; deletion must remain owned.
            await asyncio.sleep(0.06)
            assert sandbox.commands.calls == []
        assert await E2BRunner.drain_failed_creations(timeout_s=0.01) == 1
        assert calls == 1
        release.set()
        remaining = await E2BRunner.drain_failed_creations(timeout_s=1)
        assert remaining == (1 if deletion_fails else 0)
        repaired = True
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0
        assert calls == (2 if deletion_fails else 1)
    finally:
        repaired = True
        acknowledge.set()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("deletion_fails", [False, True])
async def test_hardened_rollback_fences_exact_reattachment(recovery, deletion_fails):
    reset_fake_e2b()
    sandbox = FakeSandbox("fenced-hardened")
    deleting = asyncio.Event()
    release = asyncio.Event()
    repaired = False
    calls = 0

    async def kill():
        nonlocal calls
        calls += 1
        deleting.set()
        await release.wait()
        if deletion_fails and not repaired:
            raise PermissionError("deletion denied")
        return True

    sandbox.kill = kill

    class Sandboxes(FakeAsyncSandbox):
        next_sandbox = sandbox

        @classmethod
        def list(cls, **kwargs):
            return FakeSandboxPaginator([FakeSandboxInfo(sandbox.sandbox_id, {})])

    class Module:
        AsyncSandbox = Sandboxes
        SandboxQuery = FakeSandboxQuery

    async def setup(runner):
        raise RuntimeError("setup failed")

    sibling = E2BRunner(sandbox)
    task = asyncio.create_task(
        E2BRunner.create_hardened(
            guest_setup=setup,
            cleanup_timeout_s=0.02,
            e2b_module=Module,
        )
    )

    async def require_fenced():
        before = len(sandbox.commands.calls)
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            if recovery:
                await E2BRunner.recover_hardened("handoff", e2b_module=Module)
            else:
                await E2BRunner.from_existing(sandbox.sandbox_id, e2b_module=Module)
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await sibling.exec(ExecCommand.process("true"))
        assert len(sandbox.commands.calls) == before
        assert Sandboxes.connected == []

    try:
        await asyncio.wait_for(deleting.wait(), 1)
        with pytest.raises(ExceptionGroup):
            await task
        await require_fenced()
        # Unrelated allocations are not fenced.
        other = E2BRunner(FakeSandbox("independent"))
        await other.exec(ExecCommand.process("true"))
        release.set()
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == int(deletion_fails)
        if deletion_fails:
            await require_fenced()
        repaired = True
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0
        assert calls == (2 if deletion_fails else 1)
    finally:
        repaired = True
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0


@pytest.mark.anyio
async def test_attachment_acknowledgement_cannot_cross_rollback_fence():
    reset_fake_e2b()
    sandbox = FakeSandbox("attachment-race")
    connecting = asyncio.Event()
    acknowledge = asyncio.Event()
    deleting = asyncio.Event()
    release = asyncio.Event()

    async def kill():
        deleting.set()
        await release.wait()
        return True

    sandbox.kill = kill

    class Sandboxes(FakeAsyncSandbox):
        next_sandbox = sandbox

        @classmethod
        async def connect(cls, sandbox_id, **kwargs):
            connecting.set()
            await acknowledge.wait()
            return sandbox

    class Module:
        AsyncSandbox = Sandboxes

    async def setup(runner):
        raise RuntimeError("setup failed")

    attaching = asyncio.create_task(
        E2BRunner.from_existing(
            sandbox.sandbox_id,
            e2b_module=Module,
        )
    )
    await asyncio.wait_for(connecting.wait(), 1)
    creating = asyncio.create_task(
        E2BRunner.create_hardened(
            guest_setup=setup,
            cleanup_timeout_s=0.02,
            e2b_module=Module,
        )
    )
    try:
        await asyncio.wait_for(deleting.wait(), 1)
        before = len(sandbox.commands.calls)
        acknowledge.set()
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await attaching
        assert len(sandbox.commands.calls) == before
        with pytest.raises(ExceptionGroup):
            await creating
    finally:
        acknowledge.set()
        release.set()
        await asyncio.gather(attaching, creating, return_exceptions=True)
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0


@pytest.mark.anyio
@pytest.mark.parametrize(
    "fault", ["delete_failed", "delete_blocked", "page_failed", "invalid_item"]
)
async def test_metadata_rollback_retains_every_discovered_allocation(fault):
    allocated = asyncio.Event()
    release = asyncio.Event()
    active = {"first", "second"}
    calls = []
    repaired = False

    class Paginator:
        has_next = True
        page = 0

        async def next_items(self):
            self.page += 1
            if self.page == 2:
                raise RuntimeError("later metadata page failed")
            self.has_next = fault == "page_failed"
            entries = [FakeSandboxInfo(identity, {}) for identity in ("first", "second", "second")]
            if fault == "invalid_item":
                entries.append(FakeSandboxInfo("", {}))
            return entries

    class Sandboxes:
        @classmethod
        async def create(cls, **kwargs):
            allocated.set()
            await asyncio.Event().wait()

        @classmethod
        def list(cls, **kwargs):
            return Paginator()

        @classmethod
        async def kill(cls, sandbox_id, **kwargs):
            calls.append(sandbox_id)
            if sandbox_id == "first" and not repaired:
                if fault == "delete_blocked":
                    await release.wait()
                elif fault == "delete_failed":
                    raise PermissionError("first deletion failed")
            active.discard(sandbox_id)
            return True

    class Module:
        AsyncSandbox = Sandboxes
        SandboxQuery = FakeSandboxQuery

    task = asyncio.create_task(
        E2BRunner.create_hardened(
            e2b_module=Module,
            cleanup_timeout_s=0.03,
        )
    )
    try:
        await asyncio.wait_for(allocated.wait(), 1)
        task.cancel("allocation acknowledgement lost")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled()
        for identity in active:
            with pytest.raises(RuntimeError, match="rollback is still pending"):
                await E2BRunner.from_existing(identity, e2b_module=Module)
        assert "second" not in calls
        repaired = True
        release.set()
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0
        assert active == set()
        assert calls.count("second") == 1
        assert calls.count("first") == (2 if fault == "delete_failed" else 1)
    finally:
        repaired = True
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0


@pytest.mark.anyio
@pytest.mark.parametrize("repeated", [False, True])
@pytest.mark.parametrize("failure_kind", ["ordinary", "grouped_fatal", "direct_fatal"])
async def test_metadata_rollback_preserves_cancellation_and_fatal_outcome(repeated, failure_kind):
    allocated = asyncio.Event()
    listing = asyncio.Event()
    release = asyncio.Event()
    failure = {
        "ordinary": RuntimeError("list failed"),
        "grouped_fatal": BaseExceptionGroup("provider stopped", [SystemExit(23)]),
        "direct_fatal": SystemExit(23),
    }[failure_kind]

    class Paginator:
        has_next = True

        async def next_items(self):
            listing.set()
            await release.wait()
            raise failure

    class Sandboxes:
        @classmethod
        async def create(cls, **kwargs):
            allocated.set()
            await asyncio.Event().wait()

        @classmethod
        def list(cls, **kwargs):
            return Paginator()

    class Module:
        AsyncSandbox = Sandboxes
        SandboxQuery = FakeSandboxQuery

    task = asyncio.create_task(E2BRunner.create_hardened(e2b_module=Module, cleanup_timeout_s=1))
    try:
        await asyncio.wait_for(allocated.wait(), 1)
        task.cancel("owner stopped")
        await asyncio.wait_for(listing.wait(), 1)
        if repeated:
            task.cancel("owner stopped again")
            await asyncio.sleep(0)
        release.set()
        if failure_kind != "ordinary":
            with pytest.raises(BaseExceptionGroup) as caught:
                await task
            assert caught.value.exceptions[1] is failure
            assert isinstance(caught.value.exceptions[0], asyncio.CancelledError)
            assert not task.cancelled()
        else:
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            assert caught.value.args == ("owner stopped",)
            assert caught.value.__cause__ is failure
            assert task.cancelled()
        assert task.cancelling() == (2 if repeated else 1)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_metadata_child_cancellation_does_not_reclaim_historical_caller_cancellation():
    primary = RuntimeError("allocation failed")

    class Paginator:
        has_next = True

        async def next_items(self):
            raise asyncio.CancelledError("provider child cancellation")

    class Sandboxes:
        @classmethod
        async def create(cls, **kwargs):
            raise primary

        @classmethod
        def list(cls, **kwargs):
            return Paginator()

    class Module:
        AsyncSandbox = Sandboxes
        SandboxQuery = FakeSandboxQuery

    async def run():
        current = asyncio.current_task()
        assert current is not None
        current.cancel("already handled")
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError as handled:
            assert handled.args == ("already handled",)
        await E2BRunner.create_hardened(e2b_module=Module, cleanup_timeout_s=0.1)

    task = asyncio.create_task(run())
    with pytest.raises(ExceptionGroup) as caught:
        await task
    assert caught.value.exceptions[0] is primary
    assert isinstance(caught.value.exceptions[1], RuntimeError)
    assert not task.cancelled()
    assert task.cancelling() == 1


@pytest.mark.anyio
@pytest.mark.parametrize("configured_timeout", [None, 1.0])
async def test_retained_deletion_retry_does_not_inherit_discovery_timeout(configured_timeout):
    allocated = asyncio.Event()
    retry_started = asyncio.Event()
    calls = []
    discovery_timeouts = []
    connections = []
    deleted = False
    finish_cleanup = False

    class Paginator:
        has_next = True

        async def next_items(self):
            await asyncio.sleep(0.08)
            self.has_next = False
            return [FakeSandboxInfo("late-visible", {})]

    class Sandboxes:
        @classmethod
        async def create(cls, **kwargs):
            allocated.set()
            await asyncio.Event().wait()

        @classmethod
        def list(cls, **kwargs):
            discovery_timeouts.append(kwargs["request_timeout"])
            return Paginator()

        @classmethod
        async def kill(cls, sandbox_id, **kwargs):
            nonlocal deleted
            calls.append((sandbox_id, kwargs))
            if len(calls) == 1:
                raise ConnectionError("temporary provider failure")
            retry_started.set()
            if not finish_cleanup:
                # Enforce the SDK timeout, rather than silently accepting kwargs.
                async with asyncio.timeout(kwargs.get("request_timeout", 1.0)):
                    await asyncio.sleep(0.25)
            deleted = True
            return True

        @classmethod
        async def connect(cls, sandbox_id, **kwargs):
            connections.append(sandbox_id)
            raise KeyError("provider confirms sandbox is absent")

    class Module:
        AsyncSandbox = Sandboxes
        SandboxQuery = FakeSandboxQuery

    options = {} if configured_timeout is None else {"request_timeout": configured_timeout}
    task = asyncio.create_task(
        E2BRunner.create_hardened(
            e2b_module=Module,
            cleanup_timeout_s=0.15,
            **options,
        )
    )
    try:
        await asyncio.wait_for(allocated.wait(), 1)
        task.cancel("allocation acknowledgement lost")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled() and task.cancelling() == 1
        assert calls and 0 < discovery_timeouts[-1] < 0.25
        with pytest.raises(RuntimeError, match="rollback is still pending"):
            await E2BRunner.from_existing("late-visible", e2b_module=Module)
        assert connections == []
        assert await E2BRunner.drain_failed_creations(timeout_s=0.01) == 1
        await asyncio.wait_for(retry_started.wait(), 1)
        assert await E2BRunner.drain_failed_creations(timeout_s=0.01) == 1
        assert len(calls) == 2
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0
        assert deleted and len(calls) == 2
        assert all(kwargs == options for _, kwargs in calls)
        with pytest.raises(KeyError, match="sandbox is absent"):
            await E2BRunner.from_existing("late-visible", e2b_module=Module)
        assert connections == ["late-visible"]
    finally:
        finish_cleanup = True
        await asyncio.gather(task, return_exceptions=True)
        assert await E2BRunner.drain_failed_creations(timeout_s=1) == 0
