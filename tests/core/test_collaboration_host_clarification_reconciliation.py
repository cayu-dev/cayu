"""A failed observer must not strand an exactly returned clarification turn."""

import asyncio

import pytest
from tests.core.test_collaboration_host_clarifications import drive_pending_maintenance
from tests.core.test_participant_identity import CONTEXT

from cayu import CollaborationHost, HostClarificationRule, HostOwnershipLimits, HostRegistration
from cayu.collaboration._contracts import CollaborationConflict
from cayu.collaboration.access import CollaborationAccessContext, CollaborationAccessDenied
from cayu.collaboration.participants import CollaborationUnavailable


@pytest.mark.anyio
@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("side_session", [False, True])
async def test_host_clarification_ack_loss_drains_exact_return(
    backend, side_session, tmp_path, request, monkeypatch, capsys, caplog
):
    from tests.core.test_clarification_public import test_public_question_uses_real_assistant_export

    final_admissions = []

    async def service_driver(app, candidate, *, context, delivery_context, **_):
        import warnings
        from contextvars import ContextVar

        from tests.core.collaboration_preparation_assertions import (
            assert_safe_preparation_failure,
            private_preparation_failure,
        )

        from cayu.collaboration import _host as host_adapter
        from cayu.runtime._temporary_continuation import temporary_service_key
        from cayu.sessions import _session_continuation_store as native_store

        original = app._clarification_coordinator._service_owned
        primary = ExceptionGroup("clarification acknowledgement lost", [OSError("lost ack")])
        calls = []

        async def committed_then_failed(*args, **kwargs):
            receipt = await original(*args, **kwargs)
            assert receipt.state == "returned"
            calls.append(receipt)
            raise primary

        host = CollaborationHost(
            app,
            HostRegistration(
                limits=HostOwnershipLimits(1, 1, 4, 262144),
                producer_sources=(),
                producer_rules=(),
                clarification_rules=(HostClarificationRule(candidate, context, delivery_context),),
                observation_timeout_s=30,
                shutdown_timeout_s=0.01,
            ),
        )
        failures = []
        with monkeypatch.context() as patch:
            select = host_adapter.start_clarification

            def select_with_capacity(app, ownership, *args, **kwargs):
                # Exercise the real host passes while service and lost-ack
                # reconciliation retain the only execution slot. No repeated
                # full-tuple validation is useful until that owner settles.
                assert ownership.has_slot("execution")
                return select(app, ownership, *args, **kwargs)

            patch.setattr(host_adapter, "start_clarification", select_with_capacity)
            entering = ContextVar("clarification_initial_read", default=False)
            validate_history = native_store.require_history
            initial_failure, secret = private_preparation_failure(app._request_coordinator, patch)
            failed = False

            def fail_initial_read(*args, **kwargs):
                nonlocal failed
                if entering.get() and not failed:
                    failed = True
                    raise initial_failure
                return validate_history(*args, **kwargs)

            async def native_entry(*args, **kwargs):
                token = entering.set(True)
                try:
                    return await original(*args, **kwargs)
                finally:
                    entering.reset(token)

            patch.setattr(native_store, "require_history", fail_initial_read)
            patch.setattr(app._clarification_coordinator, "_service_owned", native_entry)
            async with asyncio.timeout(120):
                with warnings.catch_warnings(record=True) as observed:
                    with pytest.raises(Exception) as caught:
                        await host.run()
            assert_safe_preparation_failure(
                caught.value, initial_failure, secret, observed, capsys, caplog
            )
            assert failed and not calls
            assert host._owned.has_slot("execution")
            assert host.inspect().failed == host.inspect().uncertain == 0
            assert (
                await app.session_store.load_session_operation(
                    candidate.ticket.session_id, temporary_service_key(candidate.operation)
                )
                is None
            )
            patch.setattr(native_store, "require_history", validate_history)
            patch.setattr(app._clarification_coordinator, "_service_owned", committed_then_failed)
            async with asyncio.timeout(120):
                while not host.inspect().failed:
                    await host.service_once()
            if not calls:
                # Report the actual sanitized causal graph, not a truncated
                # repr of the public wrapper inside an assertion's list.
                for outcome in host._owned.inspect().completed:
                    if outcome.error is not None:
                        raise outcome.error
            assert len(calls) == 1
            # The real native receiving owner has already returned and settled;
            # the host still has to authenticate this exact handoff, not retry.
            retained = await app.reconcile_clarification_service(candidate, context=CONTEXT)
            assert retained == calls[0]
            read = app._clarification_coordinator._inspect_service_owned
            assert await read(app, candidate, context=CONTEXT) == retained
            with pytest.raises(CollaborationConflict):
                await read(
                    app,
                    candidate.model_copy(update={"instruction": "Different exact input."}),
                    context=CONTEXT,
                )
            with pytest.raises(CollaborationAccessDenied):
                await read(app, candidate, context=CollaborationAccessContext(principal="foreign"))
            unavailable_observed = asyncio.Event()
            available = asyncio.Event()
            read_entered, release_read = asyncio.Event(), asyncio.Event()
            native_reads = []

            async def inspect_return(*args, **kwargs):
                if not available.is_set():
                    unavailable_observed.set()
                    return None
                native_reads.append(True)
                read_entered.set()
                await release_read.wait()
                return await read(*args, **kwargs)

            patch.setattr(app._clarification_coordinator, "_inspect_service_owned", inspect_return)
            async with asyncio.timeout(10):
                while not unavailable_observed.is_set():
                    try:
                        assert (await host.aclose()).pending
                    except ExceptionGroup as error:
                        failures.append(error)
                    await asyncio.sleep(0.01)
            assert host.inspect().failed == 1
            assert len(calls) == 1
            available.set()
            observer = asyncio.create_task(host.aclose())
            try:
                await asyncio.wait_for(read_entered.wait(), 10)
                observer.cancel()
                assert observer.cancelling() == 1
                with pytest.raises(asyncio.CancelledError):
                    await observer
                assert observer.cancelled() and observer.cancelling() == 1
                assert (await host.aclose()).pending
                assert native_reads == [True]
            finally:
                release_read.set()
                if not observer.done():
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
            async with asyncio.timeout(10):
                while True:
                    try:
                        if not (await host.aclose()).pending:
                            break
                    except ExceptionGroup as error:
                        failures.append(error)
                    await asyncio.sleep(0.01)
        assert calls == [retained]
        assert failures == [primary]
        assert host.inspect().failed == 0
        admit = app.admit_collaboration_request

        async def lose_final_admission_ack(command, *, context):
            result = await admit(command, context=context)
            if command.operation.caller_key == "final-admission" and not final_admissions:
                final_admissions.append(result)
                raise CollaborationUnavailable("Final admission acknowledgement was lost.")
            return result

        monkeypatch.setattr(app, "admit_collaboration_request", lose_final_admission_ack)
        return retained

    await test_public_question_uses_real_assistant_export(
        backend,
        tmp_path,
        request,
        monkeypatch,
        temporary_service=True,
        public_reply=True,
        post_admission=True,
        side_session=side_session,
        one_slot=True,
        continue_questioner=True,
        finish_request=True,
        service_driver=service_driver,
        maintenance_driver=drive_pending_maintenance,
        journey_ttl_ms=900_000,
    )
    assert len(final_admissions) == 1
