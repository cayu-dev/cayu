"""Service ownership of delivery and guest acknowledgement (not UI acceptance)."""

import asyncio

import pytest
from tests.core.test_browser_control import identity, operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.core.tools import (
    ToolContext,
    _bind_runtime_tool_invocation_authority,
    _RuntimeBrowserAllocationAuthority,
)
from cayu.runtime._browser_control_channel import (
    BoundBrowserGuest,
    BrowserGuestCommandOwner,
    browser_allocation_digest,
)
from cayu.runtime._browser_control_checkpoint import BrowserControlCheckpointMutation
from cayu.runtime._browser_control_publication import BrowserControlPublication
from cayu.runtime._browser_control_service import (
    BrowserControlBootstrapPending,
    BrowserControlService,
    _BootstrapOwner,
)
from cayu.runtime._invocation_secrets import InvocationPublicationSnapshot
from cayu.runtime.browser_control import (
    BrowserControlAllocation,
    BrowserControlConflict,
    BrowserControlRecord,
    closed_browser_control_successor,
)
from cayu.runtime.sessions import SessionStatus
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend
from cayu.vaults.redaction import SecretRedactor


def bound_browser_context(exact, *, runner=None, redactor=None, sealer=None):
    context = ToolContext(
        session_id=exact.session_id,
        environment_name=exact.environment_name,
        idempotency_key="call",
        runner=runner,
    )

    async def unused(*args):
        raise AssertionError("No unrelated durable operation is expected.")

    _bind_runtime_tool_invocation_authority(
        context,
        parent_task_id=None,
        parent_run_epoch=exact.run_epoch,
        model_step_id="step",
        model_attempt_id="attempt",
        tool_round_id="round",
        tool_call_id="call",
        tool_name="browser_session",
        idempotency_key="call",
        effective_arguments={"operation": "navigate"},
        execution_profile_fingerprint=exact.execution_profile_fingerprint,
        environment_allocation_fingerprint=exact.allocation_fingerprint,
        load_durable_operation=unused,
        compare_and_set_durable_operation=unused,
        seal_durable_output=lambda value: value,
        secret_publication_sealer=sealer
        if sealer is not None
        else lambda: InvocationPublicationSnapshot(
            redactor if redactor is not None else SecretRedactor(), False
        ),
        browser_allocation=_RuntimeBrowserAllocationAuthority(
            **exact.model_dump(
                exclude={"worker_instance_id", "browser_session_id", "operator_purpose"}
            )
        ),
    )
    return context


@pytest.mark.parametrize(
    "invalid", ["unsafe", "incomplete", "wrong_type", "wrong_flag", "wrong_redactor"]
)
def test_invalid_sealed_scope_fails_before_bootstrap_issuance(
    monkeypatch, invalid, capsys, caplog, recwarn
):
    canary = "scope-diagnostic-canary"

    class InvalidValue:
        def __repr__(self):
            return canary

    async def scenario():
        exact = identity()
        service = BrowserControlService(
            purpose=operator_purpose(), guest_endpoint="wss://control.example/guest"
        )
        backend = BrowserSessionTool()._backend
        assert type(backend) is _RunnerBrowserSessionBackend
        calls = []

        def forbidden(*args, **kwargs):
            calls.append(1)
            raise AssertionError("Invalid scope reached external preparation.")

        monkeypatch.setattr(service._capabilities, "issue", forbidden)
        monkeypatch.setattr(service, "settle_bootstrap_retirements", forbidden)
        snapshot = InvocationPublicationSnapshot(SecretRedactor(canary), False)
        if invalid == "unsafe":
            object.__setattr__(snapshot, "unsafe_output", True)
        elif invalid == "incomplete":
            object.__setattr__(snapshot, "secret_scope_incomplete", True)
        elif invalid == "wrong_flag":
            object.__setattr__(snapshot, "unsafe_output", 0)
        elif invalid == "wrong_redactor":
            object.__setattr__(snapshot, "redactor", InvalidValue())
        returned = InvalidValue() if invalid == "wrong_type" else snapshot
        context = bound_browser_context(exact, sealer=lambda: returned)
        with pytest.raises(BrowserControlConflict) as raised:
            await service.bootstrap(
                context,
                backend=backend,
                browser_session_id=exact.browser_session_id,
                arguments={"operation": "navigate"},
            )
        assert canary not in str(raised.value) + repr(raised.value)
        assert not calls and not service._owners

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn


def test_concurrent_bootstrap_scopes_merge_under_exact_existing_owner(monkeypatch):
    async def scenario():
        exact = identity()
        service = BrowserControlService(
            purpose=operator_purpose(), guest_endpoint="wss://control.example/guest"
        )
        backend = BrowserSessionTool()._backend
        assert type(backend) is _RunnerBrowserSessionBackend
        entered, release = asyncio.Event(), asyncio.Event()
        arrivals, deliveries = [], []

        async def pause_retirement(**kwargs):
            arrivals.append(1)
            if len(arrivals) == 2:
                entered.set()
            await release.wait()
            return True

        async def deliver(self, ctx, **kwargs):
            deliveries.append(1)
            service.authenticate_guest(kwargs["credential"])
            service.confirm_guest(
                BoundBrowserGuest(BrowserControlRecord(identity=exact), "bc_" + "a" * 32, "b" * 64)
            )

        monkeypatch.setattr(service, "settle_bootstrap_retirements", pause_retirement)
        monkeypatch.setattr(_RunnerBrowserSessionBackend, "bootstrap_control", deliver)

        async def bootstrap(secret):
            return await service.bootstrap(
                bound_browser_context(exact, redactor=SecretRedactor(secret)),
                backend=backend,
                browser_session_id=exact.browser_session_id,
                arguments={"operation": "navigate"},
            )

        tasks = [asyncio.create_task(bootstrap(secret)) for secret in ("scope-one", "scope-two")]
        try:
            await asyncio.wait_for(entered.wait(), 5)
            release.set()
            first, second = await asyncio.gather(*tasks)
            assert first == second
            assert deliveries == [1]
            for secret in ("scope-one", "scope-two"):
                assert service.invocation_origin_redactor(exact).redact_text(secret) != secret
            await bootstrap("scope-three")
            for secret in ("scope-one", "scope-two", "scope-three"):
                assert service.invocation_origin_redactor(exact).redact_text(secret) != secret
            for changed in (
                exact.model_copy(update={"worker_instance_id": "vw_" + "f" * 32}),
                exact.model_copy(update={"run_epoch": exact.run_epoch + 1}),
            ):
                with pytest.raises(BrowserControlConflict):
                    service.invocation_origin_redactor(changed)
            assert deliveries == [1]
            assert await service.drain_bootstrap_deliveries()
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


@pytest.mark.parametrize("cancel", [False, True])
def test_bootstrap_reuses_exact_delivery_after_waiter_loss(monkeypatch, cancel):
    async def scenario():
        exact = identity()
        context = bound_browser_context(exact)
        service = BrowserControlService(
            purpose=operator_purpose(), guest_endpoint="wss://control.example/guest"
        )
        backend = BrowserSessionTool()._backend
        assert type(backend) is _RunnerBrowserSessionBackend
        dispatched = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def deliver(self, ctx, **kwargs):
            calls.append(1)
            allocation = service.authenticate_guest(kwargs["credential"])
            assert allocation.browser_session_id == exact.browser_session_id
            service.confirm_guest(
                BoundBrowserGuest(BrowserControlRecord(identity=exact), "bc_" + "a" * 32, "b" * 64)
            )
            dispatched.set()
            await release.wait()
            raise OSError("runner acknowledgement lost after guest binding")

        monkeypatch.setattr(_RunnerBrowserSessionBackend, "bootstrap_control", deliver)

        async def bootstrap(timeout):
            return await service.bootstrap(
                context,
                backend=backend,
                browser_session_id=exact.browser_session_id,
                arguments={"operation": "navigate"},
                timeout_s=timeout,
            )

        caller = asyncio.create_task(bootstrap(0.05 if not cancel else 1))
        await dispatched.wait()
        if cancel:
            assert caller.cancel("caller disconnected")
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert caller.cancelled() and caller.cancelling() == 1
        else:
            with pytest.raises(BrowserControlBootstrapPending):
                await caller
        assert calls == [1]
        release.set()
        assert (await bootstrap(1)).record.identity == exact
        assert calls == [1]
        assert await service.drain_bootstrap_deliveries()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("delivery_pending", [False, True])
@pytest.mark.parametrize("termination", ["running", "closed", "completed"])
@pytest.mark.parametrize("read_failure", [False, True])
def test_only_ended_settled_channels_retire_bootstrap_owners(
    tmp_path, monkeypatch, backend, delivery_pending, termination, read_failure
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, initial):
            control = coordinator(store, Policy(True))
            record = await control._publisher.publish(initial)
            service = BrowserControlService(
                purpose=operator_purpose(), guest_endpoint="wss://control.example/guest"
            )
            native = BrowserSessionTool()._backend
            assert type(native) is _RunnerBrowserSessionBackend
            release, bound_seen = asyncio.Event(), asyncio.Event()
            if not delivery_pending:
                release.set()

            async def deliver(self, ctx, **kwargs):
                service.authenticate_guest(kwargs["credential"])
                service.confirm_guest(BoundBrowserGuest(record, "bc_" + "a" * 32, "b" * 64))
                bound_seen.set()
                await release.wait()

            monkeypatch.setattr(_RunnerBrowserSessionBackend, "bootstrap_control", deliver)
            caller = asyncio.create_task(
                service.bootstrap(
                    bound_browser_context(record.identity),
                    backend=native,
                    browser_session_id=record.identity.browser_session_id,
                    arguments={"operation": "navigate"},
                )
            )
            try:
                await bound_seen.wait()
                retained = next(iter(service._owners.values()))
                commands = BrowserGuestCommandOwner(
                    coordinator=control, connection=None, bound=retained.bound.result()
                )
                service.attach_commands(commands)
                await service.suspend_viewer_delivery(record.identity)
                assert record.identity in service._suspended_viewer_identities
                if termination == "closed":
                    terminal = closed_browser_control_successor(record)
                    assert terminal is not None
                    # The runtime close publication has its own app-level corpus;
                    # this fixture starts at that committed boundary.
                    await control._publisher.publish(
                        BrowserControlPublication(
                            BrowserControlCheckpointMutation(
                                record.identity.session_id,
                                initial.mutation.desired,
                                initial.mutation.desired.replace_record(
                                    expected=record, desired=terminal
                                ),
                            )
                        )
                    )
                    assert await commands.step() is False
                elif termination == "completed":
                    await store.update_status(record.identity.session_id, SessionStatus.COMPLETED)
                await commands.disconnect()
                read = control.bootstrap_retirement_record
                if read_failure:

                    async def unavailable(identity):
                        raise OSError("retirement snapshot unavailable")

                    monkeypatch.setattr(control, "bootstrap_retirement_record", unavailable)
                service.detach_commands(commands)
                if read_failure:
                    assert not await service.settle_bootstrap_retirements()
                    assert retained.commands is None
                    assert len(service._owners) == 1
                    assert not await service.settle_bootstrap_retirements()
                    assert len(service._owners) == 1
                    monkeypatch.setattr(control, "bootstrap_retirement_record", read)
                assert await service.settle_bootstrap_retirements()
                retain_owner = termination == "running"
                assert bool(service._owners) == (delivery_pending or retain_owner)
                assert (record.identity in service._suspended_viewer_identities) == (
                    delivery_pending or retain_owner
                )
                release.set()
                await caller
                # Delivery completion callbacks run before the waiting caller.
                assert bool(service._owners) == retain_owner
                assert (record.identity in service._suspended_viewer_identities) == retain_owner
                assert await service.drain()
            finally:
                release.set()
                await asyncio.gather(caller, return_exceptions=True)
                await control.drain()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("ended", [False, True])
@pytest.mark.parametrize("viewer_closes_first", [False, True])
def test_viewer_retirement_requires_ended_authority_and_route_close(
    tmp_path, backend, ended, viewer_closes_first
):
    async def scenario():
        async with publication_fixture(backend, tmp_path) as (store, initial):
            control = coordinator(store, Policy(True))
            record = await control._publisher.publish(initial)
            service = BrowserControlService(
                purpose=operator_purpose(), guest_endpoint="wss://control.example/guest"
            )
            native = BrowserSessionTool()._backend
            assert type(native) is _RunnerBrowserSessionBackend
            bound = BoundBrowserGuest(record, "bc_" + "a" * 32, "b" * 64)
            future = asyncio.get_running_loop().create_future()
            future.set_result(bound)

            async def delivered():
                return ()

            delivery = asyncio.create_task(delivered())
            await delivery
            allocation = BrowserControlAllocation.model_validate(
                record.identity.model_dump(exclude={"worker_instance_id"})
            )
            owner = _BootstrapOwner(allocation, native, future, delivery)
            service._owners[browser_allocation_digest(allocation)] = owner
            commands = BrowserGuestCommandOwner(coordinator=control, connection=None, bound=bound)
            service.attach_commands(commands)
            viewer = service.register_viewer(record.identity)
            viewer.begin_send()
            viewer.finish_send()
            assert viewer.may_hold_frames and not viewer.purge_settled
            if viewer_closes_first:
                service.retire_viewer(viewer)
                assert viewer in service._viewers
            if ended:
                await store.update_status(record.identity.session_id, SessionStatus.COMPLETED)
            await commands.disconnect()
            service.detach_commands(commands)
            assert await service.settle_bootstrap_retirements()
            if not viewer_closes_first:
                assert viewer in service._viewers and service._owners
                service.retire_viewer(viewer)
            assert (viewer in service._viewers) is (not ended)
            assert bool(service._owners) is (not ended)
            # No purge acknowledgement or native-quiescence evidence was forged.
            assert viewer.may_hold_frames and not viewer.purge_settled
            _, _, _, durable = await control._read_record(record.identity)
            assert durable.state == "control_uncertain"
            assert await service.drain()
            assert await control.drain()

    asyncio.run(scenario())
