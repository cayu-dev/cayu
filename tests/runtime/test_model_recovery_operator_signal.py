"""Worker-loss model settlement preserves a previously delivered operator stop."""

from __future__ import annotations

import asyncio
import base64
import os
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr
from tests.runtime.runtime_provider_support import VersionedFakeProvider
from tests.runtime.test_orphaned_session_reconciliation import (
    _budget,
    _orphan,
    _slot,
    _stores,
)
from tests.runtime.test_orphaned_session_reconciliation import database_url as database_url

from cayu import AgentSpec, CayuApp, Event, EventType, Message, RunRequest, SessionStatus
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions.requests import InterruptSessionRequest
from cayu.storage import open_application_stores
from cayu.vaults.redaction import SecretRedactor


class _CancelledModel(VersionedFakeProvider):
    def __init__(self, *, crash):
        super().__init__([])
        self.crash = crash
        self.started = asyncio.Event()

    async def stream(self, request):
        if not self.crash:
            raise AssertionError("Recovery redispatched the model")
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            os._exit(137)
        yield  # pragma: no cover


def _application(store, *, crash=False, secret_redactor=None):
    app = CayuApp(
        session_store=store,
        budget_policy=_budget(),
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    provider = _CancelledModel(crash=crash)
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="scanner", model="fake-model"))
    slot = _slot()
    app.register_environment_factory(slot.environment.spec, slot)
    return app, provider


def _crash_during_operator_interrupt():
    async def scenario():
        stores = _stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"], os.environ["CAYU_ORPHAN_TEST_STORE"]
        )
        app, provider = _application(stores.session_store, crash=True)

        async def interrupt_started_model():
            await provider.started.wait()
            async for _ in app.interrupt_session(
                InterruptSessionRequest(
                    session_id="scan",
                    reason="operator requested stop",
                    metadata={"operator_request": "keep this original decision"},
                )
            ):
                pass

        interruption = asyncio.create_task(interrupt_started_model())
        async for _ in app.run(
            RunRequest(
                session_id="scan",
                agent_name="scanner",
                environment_name=_slot().environment.spec.name,
                messages=[Message.text("user", "go")],
            )
        ):
            pass
        await interruption
        raise AssertionError("The provider did not receive the operator cancellation")

    asyncio.run(scenario())


@pytest.mark.parametrize("invalid_marker", [None, "missing_request_id", "unknown_type"])
def test_recovery_preserves_operator_signal_after_model_worker_dies(
    tmp_path, database_url, invalid_marker
):
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        database_url=database_url,
        module=__name__,
        entry="_crash_during_operator_interrupt",
    )

    async def scenario():
        stores = _stores(database_url, tmp_path / "cayu.db")
        app, provider = _application(stores.session_store)
        try:
            session = await stores.session_store.load("scan")
            assert session.status is SessionStatus.INTERRUPTING
            checkpoint = await stores.session_store.load_checkpoint("scan")
            original = checkpoint["pending_session_interrupt"]
            assert original["reason"] == "operator requested stop"
            assert original["interruption_request_id"]
            if invalid_marker is not None:
                if invalid_marker == "missing_request_id":
                    original.pop("interruption_request_id")
                else:
                    original["interruption_type"] = "future_unrecognized_signal"
                await stores.session_store.checkpoint("scan", checkpoint)
                active = await stores.session_store.load_active_model_completion_stage("scan")
                reservations = tuple(
                    [
                        await stores.budget_ledger.load_reservation(reservation_id)
                        for reservation_id in active.stage.reservation_ids
                    ]
                )
            await app.reconcile_orphaned_sessions()
            if invalid_marker is not None:
                assert (
                    await stores.session_store.load("scan")
                ).status is SessionStatus.INTERRUPTING
                assert (
                    await stores.session_store.load_model_completion_stage_settlement(
                        "scan", active.stage.stage_id
                    )
                    is None
                )
                for before in reservations:
                    assert (
                        await stores.budget_ledger.load_reservation(before.reservation_id) == before
                    )
                assert not any(
                    event.type is EventType.SESSION_INTERRUPTED
                    for event in await stores.session_store.load_events("scan")
                )
                assert provider.requests == []
                return
            assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
            events = await stores.session_store.load_events("scan")
            [terminal] = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
            for key in ("reason", "metadata", "interruption_type", "interruption_request_id"):
                assert terminal.payload[key] == original[key]
            assert provider.requests == []
            assert await app.reconcile_orphaned_sessions() == ()
            assert await stores.session_store.load_events("scan") == events
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())


def _redacting_stores(database_url, path):
    codec = PublicAuthorityAliasCodec(
        PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={
                "test": SecretStr(base64.urlsafe_b64encode(bytes(range(32))).decode().rstrip("="))
            },
        )
    )
    return open_application_stores(
        database_url,
        sqlite_path=path,
        tasks=False,
        event_watchers=False,
        public_authority_alias_codec=codec,
    )


def _crash_after_operator_model_settlement():
    import cayu.runtime._recovery_ownership as recovery_ownership

    recovery_ownership._INCOMPLETE_RECOVERY_CLAIM_LEASE = timedelta(seconds=1)

    async def scenario():
        stores = _redacting_stores(
            os.environ["CAYU_ORPHAN_TEST_DATABASE_URL"], os.environ["CAYU_ORPHAN_TEST_STORE"]
        )
        checkpoint = await stores.session_store.load_checkpoint("scan")
        request_id = checkpoint["pending_session_interrupt"]["interruption_request_id"]
        app, _ = _application(stores.session_store, secret_redactor=SecretRedactor(request_id[:8]))
        fanout = app._event_writer.fan_out_persisted

        async def die_after_commit(events, *args, **kwargs):
            if any(event.type is EventType.INTERACTION_INTERRUPTED for event in events):
                assert (await stores.session_store.load("scan")).status is SessionStatus.INTERRUPTED
                os._exit(137)
            return await fanout(events, *args, **kwargs)

        app._event_writer.fan_out_persisted = die_after_commit
        await app.reconcile_orphaned_sessions()
        raise AssertionError("Owned model settlement did not reach its native commit")

    asyncio.run(scenario())


@pytest.mark.parametrize("restart", [False, True])
def test_owned_operator_identity_survives_redaction_and_durable_reconstruction(
    tmp_path, database_url, restart, monkeypatch
):
    _orphan(
        tmp_path / "cayu.db",
        "scan",
        database_url=database_url,
        module=__name__,
        entry="_crash_during_operator_interrupt",
    )
    if restart:
        _orphan(
            tmp_path / "cayu.db",
            "scan",
            database_url=database_url,
            module=__name__,
            entry="_crash_after_operator_model_settlement",
        )

    async def scenario():
        from cayu.sessions.recovery import _incomplete_recovery_claim_from_checkpoint

        stores = _redacting_stores(database_url, tmp_path / "cayu.db")
        checkpoint = await stores.session_store.load_checkpoint("scan")
        original = checkpoint["pending_session_interrupt"]
        request_id = original["interruption_request_id"]
        app, provider = _application(
            stores.session_store, secret_redactor=SecretRedactor(request_id[:8])
        )
        failures = []
        settle = app._manual_model_completion_recovery.recover_model_completion_stage

        async def inspect_settlement(*args, **kwargs):
            try:
                return await settle(*args, **kwargs)
            except Exception as error:
                failures.append(error)
                raise

        monkeypatch.setattr(
            app._manual_model_completion_recovery,
            "recover_model_completion_stage",
            inspect_settlement,
        )
        try:
            async with asyncio.timeout(10):
                while True:
                    claim = _incomplete_recovery_claim_from_checkpoint(
                        await stores.session_store.load_checkpoint("scan")
                    )
                    if claim is None or claim[1] <= datetime.now(UTC):
                        break
                    await asyncio.sleep(0.02)
            # Identical caller-owned text on the real public producer entrance
            # cannot borrow the internal interrupt request's provenance.
            await app.emit_events(
                "scan",
                [
                    Event(
                        type="custom.raw_identity_probe",
                        session_id="scan",
                        payload={"interruption_request_id": request_id},
                    )
                ],
            )
            [probe] = [
                event
                for event in await stores.session_store.load_events("scan")
                if event.type == "custom.raw_identity_probe"
            ]
            assert probe.payload["interruption_request_id"] == SecretRedactor(
                request_id[:8]
            ).redact_text(request_id)
            assert probe.payload["interruption_request_id"] != request_id
            await app.reconcile_orphaned_sessions()
            if failures:
                raise failures[0]
            events = await stores.session_store.load_events("scan")
            [terminal] = [event for event in events if event.type is EventType.SESSION_INTERRUPTED]
            assert terminal.payload["interruption_request_id"] == request_id
            for key in ("reason", "metadata", "interruption_type"):
                assert terminal.payload[key] == original[key]
            assert provider.requests == []
            assert await app.reconcile_orphaned_sessions() == ()
            assert await stores.session_store.load_events("scan") == events
        finally:
            await app.aclose()
            await stores.close()

    asyncio.run(scenario())
