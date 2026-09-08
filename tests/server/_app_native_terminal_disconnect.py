"""Cut the real guest socket around an app-produced browser terminal receipt."""

import asyncio

from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime.browser_control import BrowserControlCheckpoint
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.server._browser_guest_routes import _GuestSocket


class NativeTerminalDisconnect:
    def __init__(self, phase):
        self.operation, self.boundary = phase.split("_")
        self.idle = asyncio.Event()
        self.release = asyncio.Event()
        self.dispatched = []
        self.publications = 0
        self.cut = False
        self.identity = None

    async def arm(self, *, runtime, store, daemon, monkeypatch):
        self.runtime, self.store, self.daemon = runtime, store, daemon
        self.identity = next(iter(runtime.service._owners.values())).bound.result().record.identity
        self.before = await self.record()
        assert self.before.state == "agent_controlled" and self.before.fresh_observation_required, (
            "Handback did not establish the observation fence"
        )
        self.connection = daemon._operator_connection
        assert self.connection is not None, "Native channel missing"
        idle = _GuestSocket.idle

        async def held_idle(socket):
            self.idle.set()
            await self.release.wait()
            return await idle(socket)

        monkeypatch.setattr(_GuestSocket, "idle", held_idle)
        await asyncio.wait_for(self.idle.wait(), 5)
        publish = store.publish_session_operation

        async def publication(*args, operation_transform, **kwargs):
            terminal = False

            def transform(*values):
                nonlocal terminal
                result = operation_transform(*values)
                terminal = any(
                    record.get("operation") == self.operation
                    and record.get("state") == "terminal"
                    and record.get(
                        "close_confirmed" if self.operation == "close" else "observation_confirmed"
                    )
                    is True
                    for record in result.operation_records.values()
                )
                return result

            result = await publish(*args, operation_transform=transform, **kwargs)
            if terminal:
                self.publications += 1
                if self.boundary == "committed" and not self.cut:
                    await self.disconnect()
            return result

        monkeypatch.setattr(store, "publish_session_operation", publication)

    async def record(self):
        assert self.identity is not None
        with browser_control_checkpoint_read_scope(self.identity.session_id):
            checkpoint = await self.store.load_checkpoint(self.identity.session_id)
        return BrowserControlCheckpoint.model_validate(
            checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY]
        ).records[0]

    async def after_native(self, request, result):
        self.dispatched.append(request.operation_id)
        if request.operation_id != "fresh":
            return
        assert request.operation == self.operation
        # Do not replace/construct the returned browser result or its provenance.
        assert result["kind"] == ("closed" if self.operation == "close" else "success"), (
            "Native terminal operation did not succeed"
        )
        if self.operation == "observe":
            assert not self.daemon.control.fresh_observation_required, (
                "Native observation did not settle"
            )
        else:
            assert self.daemon.browser is None and self.daemon.context is None, (
                "Native close did not retire Chromium"
            )
        if self.boundary == "pending":
            assert await self.record() == self.before, "Control changed before terminal publication"
            await self.disconnect()

    async def disconnect(self):
        self.cut = True
        self.at_cut = await self.record()
        if self.boundary == "committed":
            assert self.at_cut.revision == self.before.revision + 1, (
                "Terminal publication did not advance control"
            )
            assert self.at_cut.state == (
                "closed" if self.operation == "close" else "agent_controlled"
            ), "Terminal publication produced unexpected control state"
            assert not self.at_cut.fresh_observation_required, (
                "Published terminal retained observation fence"
            )
        await self.connection.close()
        self.release.set()
        # Let actual route cleanup run; do not use drain's cancellation as proof.
        async with asyncio.timeout(10):
            while any(not task.done() for task in self.runtime.service.channels._tasks):
                await asyncio.sleep(0.01)
        assert not self.runtime.service.channels._tasks, "Guest route retained failed cleanup"
        current = await self.record()
        if self.operation == "close" and self.boundary == "committed":
            assert current == self.at_cut, "Disconnect changed the closed terminal"
        else:
            assert current == self.at_cut.model_copy(
                update={
                    "state": "control_uncertain",
                    "revision": self.at_cut.revision + 1,
                }
            ), "Disconnect did not fence the exact terminal generation"
        self.fenced = current

    async def verify(self):
        assert self.cut, "Guest connection was not cut"
        assert self.publications == 1, f"Terminal publication count differs: {self.publications}"
        assert await self.record() == self.fenced, "Later work changed the disconnect fence"
        assert self.dispatched == ["fresh"], "Unexpected native dispatch after disconnect"

    def release_cleanup(self):
        self.release.set()
