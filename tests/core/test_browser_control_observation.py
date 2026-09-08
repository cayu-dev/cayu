"""Actual app publication after a supplied settled handback checkpoint.

Native handback is not performed by this fixture. Model dispatch, tool sealing,
session publication and persistent readback use their production entrances.
"""

import asyncio
import time
from dataclasses import replace

import pytest
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control import request as takeover_request
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_session import _FakeBrowserBackend
from tests.core.test_environment_allocation_recovery import _FakeRemoteFactory, _FakeRemoteProvider

from cayu import (
    AgentSpec,
    CayuApp,
    EnvironmentSpec,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    run_to_completion,
)
from cayu.runtime._browser_control_bootstrap import BrowserGuestBootstrap
from cayu.runtime._browser_control_checkpoint import (
    BrowserControlCheckpointMutation,
    browser_control_checkpoint_read_scope,
)
from cayu.runtime._browser_control_publication import BrowserControlPublication
from cayu.runtime._browser_control_service import BrowserControlService
from cayu.runtime.browser_control import (
    BrowserControlCheckpoint,
    BrowserControlConflict,
    BrowserControlPrincipal,
    BrowserOperatorPageOperations,
    BrowserTakeoverIntent,
)
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.tools.browser_session import (
    BrowserBackendFailure,
    BrowserSessionTool,
    _durable_browser_operation_key,
    _RunnerBrowserSessionBackend,
)


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("mode", ["protected", "ack_loss", "cancel_ack"])
def test_runtime_handback_preserves_nonzero_operator_accounting(
    tmp_path, monkeypatch, persistent, mode
):
    test_runtime_observation_atomically_releases_handback_fence(
        tmp_path, monkeypatch, persistent, mode, operator_count=3
    )


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize(
    "mode", ["protected", "unprotected", "failure", "ack_loss", "precommit_failure", "cancel_ack"]
)
def test_runtime_observation_atomically_releases_handback_fence(
    tmp_path,
    monkeypatch,
    persistent,
    mode,
    terminal_operation="observe",
    takeover_race=False,
    operator_count=0,
):
    async def scenario():
        store = (
            SQLiteSessionStore(tmp_path / "sessions.db") if persistent else InMemorySessionStore()
        )
        publish = store.publish_session_operation
        injected = []
        committed, release_ack = asyncio.Event(), asyncio.Event()
        close_dispatched, release_close = asyncio.Event(), asyncio.Event()
        publication_task = None

        async def faulted_publication(*args, operation_transform, **kwargs):
            nonlocal publication_task
            observation = False

            def transform(*values):
                nonlocal observation
                result = operation_transform(*values)
                observation = any(
                    r.get("observation_confirmed") is True or r.get("close_confirmed") is True
                    for r in result.operation_records.values()
                )
                if observation and mode == "precommit_failure" and not injected:
                    injected.append(mode)
                    raise ConnectionError("observation transaction rejected before commit")
                return result

            result = await publish(*args, operation_transform=transform, **kwargs)
            if observation and mode == "ack_loss" and not injected:
                injected.append(mode)
                raise ConnectionError("observation acknowledgement lost after commit")
            if observation and mode == "cancel_ack" and not injected:
                injected.append(mode)
                publication_task = asyncio.current_task()
                committed.set()
                await release_ack.wait()
            return result

        monkeypatch.setattr(store, "publish_session_operation", faulted_publication)
        app = CayuApp(
            enable_logging=False,
            session_store=store,
            browser_control=BrowserControlConfig(
                purpose=operator_purpose(),
                policy=Policy(True),
                guest_endpoint="wss://control.test/guest",
            ),
        )
        fake = _FakeBrowserBackend()
        bound = None

        async def preflight(backend, ctx, args):
            return await fake.preflight(ctx, args)

        async def execute(backend, ctx, args):
            if args["operation"] == terminal_operation:
                assert args["invocation_control_epoch"] == 3
                assert args["_operator_page_operations"] == (
                    ((fake.page_id, operator_count),) if operator_count else ()
                )
                if mode == "failure":
                    fake.failure = BrowserBackendFailure("browser_crash")
            response = await fake.execute(ctx, args)
            if takeover_race and args["operation"] == "close":
                close_dispatched.set()
                await release_close.wait()
            return (
                replace(response, profile_output_protected=True)
                if response.observation is not None and mode != "unprotected"
                else response
            )

        async def bootstrap(service, ctx, *, backend, browser_session_id, arguments):
            nonlocal bound
            if bound is not None:
                return
            runtime = app._browser_control_runtime
            assert runtime is not None
            coordinator = runtime.coordinator
            allocation = BrowserGuestBootstrap.allocation_for_invocation(
                ctx,
                purpose=operator_purpose(),
                browser_session_id=browser_session_id,
                arguments=arguments,
            )
            initial = await coordinator.bind_guest(
                allocation=allocation, worker_instance_id="vw_" + "a" * 32
            )
            controls, initial = await coordinator._load(initial.identity)
            request = takeover_request().model_copy(update={"identity": initial.identity})
            assert fake.page_id is not None
            bound = initial.model_copy(
                update={
                    "revision": initial.revision + 1,
                    "control_epoch": 3,
                    "request": request,
                    "fresh_observation_required": True,
                    "capture_restricted": True,
                    "settled_input_sequence": operator_count,
                    "operator_page_operations": (
                        (
                            BrowserOperatorPageOperations(
                                page_id=fake.page_id, operations=operator_count
                            ),
                        )
                        if operator_count
                        else ()
                    ),
                }
            )
            await coordinator._publisher.publish(
                BrowserControlPublication(
                    BrowserControlCheckpointMutation(
                        initial.identity.session_id,
                        controls,
                        controls.replace_record(expected=initial, desired=bound),
                    )
                )
            )
            fake.operation_count += operator_count

        monkeypatch.setattr(_RunnerBrowserSessionBackend, "preflight", preflight)
        monkeypatch.setattr(_RunnerBrowserSessionBackend, "execute", execute)
        monkeypatch.setattr(BrowserControlService, "bootstrap", bootstrap)

        class Provider(ScriptedModelProvider):
            count = 0

            async def stream(self, request):
                self.count += 1
                if self.count == 1:
                    args = {
                        "operation": "navigate",
                        "url": "https://example.test/",
                        "operation_id": "open",
                    }
                elif self.count == 2:
                    args = {
                        "operation": terminal_operation,
                        "session_id": fake.session_id,
                        "page_id": fake.page_id,
                        "operation_id": "fresh",
                    }
                    if terminal_operation == "close":
                        args.pop("page_id")
                else:
                    assert bound is not None
                    with browser_control_checkpoint_read_scope(bound.identity.session_id):
                        checkpoint = await store.load_checkpoint(bound.identity.session_id)
                    current = BrowserControlCheckpoint.model_validate(
                        checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY]
                    ).records[0]
                    released = mode in {"protected", "ack_loss"}
                    assert current.fresh_observation_required is (not released)
                    assert current.revision == bound.revision + int(released)
                    assert current.operator_page_operations == bound.operator_page_operations
                    if operator_count and released:
                        receipt = await store.load_session_operation(
                            bound.identity.session_id, _durable_browser_operation_key("fresh")
                        )
                        assert receipt is not None and receipt["state"] == "terminal"
                        assert receipt["operator_page_operations"] == [
                            [fake.page_id, operator_count]
                        ]
                    if terminal_operation == "close":
                        assert current.state == ("closed" if released else "agent_controlled")
                        if released:
                            coordinator = app._browser_control_runtime.coordinator
                            with pytest.raises(BrowserControlConflict):
                                await coordinator._load(bound.identity)
                            assert (
                                await coordinator.mark_channel_uncertain(expected=bound) == current
                            )
                    yield ModelStreamEvent.text_delta("done")
                    yield ModelStreamEvent.completed({"finish_reason": "stop"})
                    return
                yield ModelStreamEvent.tool_call(
                    id=f"call-{self.count}", name="browser_session", arguments=args
                )
                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})

        provider = Provider([])
        app.register_provider(provider, default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="browser"), _FakeRemoteFactory(_FakeRemoteProvider()), default=True
        )
        app.register_agent(AgentSpec(name="agent", model="model"), tools=[BrowserSessionTool()])
        try:
            operation = run_to_completion(
                app, RunRequest(agent_name="agent", messages=[Message.text("user", "observe")])
            )
            if takeover_race:
                running = asyncio.create_task(operation)
                operation = running
                try:
                    async with asyncio.timeout(20):
                        await close_dispatched.wait()
                    assert bound is not None and app._browser_control_runtime is not None
                    assert not running.done()
                    now = int(time.time() * 1000)
                    intent = BrowserTakeoverIntent(
                        identity=bound.identity,
                        request_id="bt_" + "2" * 32,
                        expected_record_revision=bound.revision,
                        expected_control_epoch=bound.control_epoch,
                        pages=takeover_request().pages,
                        purpose_code="login",
                        requested_at_ms=now,
                        expires_at_ms=now + 10_000,
                        maximum_until_ms=now + 30_000,
                    )
                    bound = await app._browser_control_runtime.coordinator.request_takeover(
                        principal=BrowserControlPrincipal(subject="operator", tenant="tenant"),
                        operator_session_id="competing-operator-session",
                        intent=intent,
                    )
                    assert bound.state == "takeover_requested" and bound.lease_until_ms is None
                    assert not running.done()
                    release_close.set()
                    await running
                finally:
                    release_close.set()
                    if not running.done():
                        running.cancel()
                    await asyncio.gather(running, return_exceptions=True)
            if mode == "cancel_ack":
                owner = asyncio.create_task(operation)
                try:
                    async with asyncio.timeout(20):
                        await committed.wait()
                    assert publication_task is not None and not publication_task.done()
                    assert bound is not None
                    with browser_control_checkpoint_read_scope(bound.identity.session_id):
                        checkpoint = await store.load_checkpoint(bound.identity.session_id)
                    assert checkpoint is not None
                    current = BrowserControlCheckpoint.model_validate(
                        checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY]
                    ).records[0]
                    assert not current.fresh_observation_required
                    assert current.revision == bound.revision + 1
                    receipt = await store.load_session_operation(
                        bound.identity.session_id, _durable_browser_operation_key("fresh")
                    )
                    assert receipt is not None and receipt["state"] == "terminal"
                    assert receipt["invocation_control_epoch"] == current.control_epoch
                    assert current.operator_page_operations == bound.operator_page_operations
                    assert receipt["operator_page_operations"] == (
                        [[fake.page_id, operator_count]] if operator_count else []
                    )
                    assert owner.cancel("cancel observation acknowledgement")
                    await asyncio.sleep(0)
                    assert not publication_task.done()
                    release_ack.set()
                    with pytest.raises(asyncio.CancelledError):
                        await owner
                    assert owner.cancelled() and owner.cancelling() == 1
                    assert publication_task.done() and not publication_task.cancelled()
                    assert [call["operation"] for call in fake.calls] == [
                        "navigate",
                        terminal_operation,
                    ]
                    assert injected == ["cancel_ack"]
                    return
                finally:
                    release_ack.set()
                    if not owner.done():
                        owner.cancel()
                        await asyncio.gather(owner, return_exceptions=True)
            result = await operation
            assert result.ok
            assert provider.count == 3
            assert [call["operation"] for call in fake.calls] == ["navigate", terminal_operation]
            assert injected == ([mode] if mode in {"ack_loss", "precommit_failure"} else [])
            if terminal_operation == "close" and mode in {"protected", "ack_loss"}:
                from cayu.runtime._browser_control_channel import (
                    BoundBrowserGuest,
                    BrowserGuestCommandOwner,
                )

                assert app._browser_control_runtime is not None and bound is not None
                coordinator = app._browser_control_runtime.coordinator
                channel = BrowserGuestCommandOwner(
                    coordinator=coordinator,
                    connection=object(),
                    bound=BoundBrowserGuest(bound, "channel", "a" * 64),
                )
                assert await channel.step() is False
                assert channel.bound.record.state == "closed" and channel.sequence == 0
                await channel.disconnect()
                # Terminal cleanup readback must never reopen live permission.
                with pytest.raises(BrowserControlConflict):
                    await coordinator._load(bound.identity)
        finally:
            if isinstance(store, SQLiteSessionStore):
                await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize(
    "mode", ["protected", "failure", "ack_loss", "precommit_failure", "cancel_ack"]
)
def test_runtime_close_atomically_retires_browser_control(tmp_path, monkeypatch, persistent, mode):
    test_runtime_observation_atomically_releases_handback_fence(
        tmp_path, monkeypatch, persistent, mode, terminal_operation="close"
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_dispatched_close_settles_concurrent_takeover_without_grant(
    tmp_path, monkeypatch, persistent
):
    test_runtime_observation_atomically_releases_handback_fence(
        tmp_path,
        monkeypatch,
        persistent,
        "protected",
        terminal_operation="close",
        takeover_race=True,
    )


@pytest.mark.parametrize("protected", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
def test_guest_fresh_observation_uses_retained_protected_result(monkeypatch, protected, exhausted):
    from tests.core.test_browser_session import _interactive_request

    from cayu.tools import _browser_guest as guest

    async def scenario():
        daemon = guest._InteractiveDaemon("bs_test")
        daemon.context = object()
        daemon.profile_output_values = ()
        daemon.control.bind("a" * 64)
        daemon.control.fresh_observation_required = True
        daemon.control.capture_restricted = True
        if exhausted:
            daemon.operation_ledger_bytes = guest._INTERACTIVE_OPERATION_LEDGER_BYTES - 1

        async def configure(request):
            pass

        async def observe(request):
            return {
                "kind": "success",
                "observation": {"revision": "fresh"},
                "profile_output_protected": protected,
            }

        monkeypatch.setattr(daemon, "_ensure_configuration", configure)
        monkeypatch.setattr(daemon, "_execute_locked", observe)
        result = await daemon.execute(
            replace(_interactive_request("observe"), invocation_control_epoch=1)
        )
        assert daemon.control.fresh_observation_required is (exhausted or not protected)
        assert result["kind"] == ("error" if exhausted else "success")
        receipt = next(iter(daemon.operations.values()))
        assert receipt.response == result

    asyncio.run(scenario())


@pytest.mark.parametrize("transition", ["observe", "close", "conflicting_close"])
def test_guest_channel_accepts_only_exact_runtime_terminal_successors(transition):
    from types import SimpleNamespace
    from typing import cast

    from tests.core.test_browser_control import identity

    from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
    from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
    from cayu.runtime.browser_control import BrowserControlRecord

    async def scenario():
        previous = BrowserControlRecord(
            identity=identity(),
            revision=5,
            control_epoch=3,
            request=takeover_request(),
            fresh_observation_required=True,
        )
        current = previous.model_copy(update={"revision": 6, "fresh_observation_required": False})
        if transition != "observe":
            current = current.model_copy(
                update={
                    "state": "closed",
                    "revision": 7 if transition == "conflicting_close" else 6,
                }
            )

        async def load(identity):
            return None, current

        owner = BrowserGuestCommandOwner(
            coordinator=cast(
                "BrowserControlCoordinator", SimpleNamespace(_load_channel_record=load)
            ),
            connection=object(),
            bound=BoundBrowserGuest(previous, "channel", "a" * 64),
        )
        if transition == "conflicting_close":
            with pytest.raises(BrowserControlConflict):
                await owner.step()
            assert owner.bound.record == previous and owner.sequence == 0
            return
        outcome = await owner.step()
        assert outcome is (False if transition == "close" else None)
        assert owner.bound.record == owner.expected == current
        assert owner.sequence == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "corruption",
    [
        None,
        {"fresh_observation_required": 0},
        {"control_epoch": 4},
        {"state": "operator_controlled"},
        {"pending_sequence": 1},
        {"worker_instance": "different-worker"},
        {"extra": True},
    ],
)
def test_guest_observation_status_cannot_clear_unpublished_host_fence(corruption):
    from types import SimpleNamespace
    from typing import cast

    from tests.core.test_browser_control import identity

    from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
    from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
    from cayu.runtime._browser_control_model import browser_model_control_epoch
    from cayu.runtime.browser_control import BrowserControlAllocation, BrowserControlRecord

    async def scenario():
        record = BrowserControlRecord(
            identity=identity(),
            revision=5,
            control_epoch=3,
            request=takeover_request(),
            fresh_observation_required=True,
        )
        controls = BrowserControlCheckpoint(records=(record,))

        async def load(_identity):
            return controls, record

        class Connection:
            async def send(self, value):
                import json

                self.command = json.loads(value)

            async def recv(self):
                import json

                return json.dumps(
                    {
                        **self.command,
                        "kind": "settled",
                        "control_epoch": 3,
                        "state": "agent_controlled",
                        "settled_sequence": 0,
                        "pending_sequence": None,
                        "fresh_observation_required": False,
                        **(corruption or {}),
                    }
                )

        owner = BrowserGuestCommandOwner(
            coordinator=cast(
                "BrowserControlCoordinator", SimpleNamespace(_load_channel_record=load)
            ),
            connection=Connection(),
            bound=BoundBrowserGuest(record, "channel", "a" * 64),
        )
        if corruption is not None:
            with pytest.raises(BrowserControlConflict):
                await owner.step()
        else:
            await owner.step()
        assert owner.bound.record == owner.expected == record
        assert owner.sequence == 1
        allocation = BrowserControlAllocation.model_validate(
            record.identity.model_dump(exclude={"worker_instance_id"})
        )
        with pytest.raises(BrowserControlConflict):
            browser_model_control_epoch(
                {BROWSER_CONTROLS_CHECKPOINT_KEY: controls.model_dump(mode="json")},
                allocation=allocation,
                operation_name="click",
            )

    asyncio.run(scenario())
