"""Authenticated control admission through native boundary evidence and publication."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_coordinator import coordinator, intent_for
from tests.core.test_browser_control_publisher import publication_fixture

from cayu.runtime._browser_control_channel import BoundBrowserGuest, BrowserGuestCommandOwner
from cayu.runtime._browser_control_publisher import BrowserControlPublisher
from cayu.server._browser_control_routes import (
    BrowserOperatorSessionTokens,
    create_browser_control_router,
)
from cayu.server.auth import BasicAuth
from cayu.tools._browser_control_guest import GuestControlChannel, GuestControlFence
from cayu.tools._browser_guest import _InteractiveDaemon, _InteractivePage
from cayu.vaults.redaction import SecretRedactor

_MALFORMED_CASES = (
    "malformed_origin",
    "malformed_missing",
    "malformed_request",
    "malformed_phase",
    "malformed_epoch",
    "malformed_page",
    "malformed_revision",
    "malformed_page_epoch",
)
_MALFORMED_HANDBACK_CASES = (
    "handback_malformed_origin",
    "handback_malformed_missing",
    "handback_malformed_request",
    "handback_malformed_phase",
    "handback_malformed_epoch",
)


@asynccontextmanager
async def audit_publication_fixture(backend, tmp_path, postgres_dsn):
    if backend != "postgres":
        async with publication_fixture(backend, tmp_path) as fixture:
            yield fixture
        return
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from cayu import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    schema_name = f"browser_audit_{uuid4().hex}"
    async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
        await connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema_name)))
        await connection.commit()
    try:
        raw = PostgresSessionStore(
            make_conninfo(postgres_dsn, options=f"-c search_path={schema_name}"),
            schema_mode=SchemaMode.CREATE,
        )
        try:
            async with publication_fixture(backend, tmp_path, raw_store=raw) as fixture:
                yield fixture
        finally:
            await raw.close()
    finally:
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema_name))
            )
            await connection.commit()


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize(
    "case",
    [
        "safe",
        "host_secret",
        "guest_secret",
        *_MALFORMED_CASES,
        *_MALFORMED_HANDBACK_CASES,
        "cancel_handback",
        "lost_handback_ack",
        "failed_handback_readback",
    ],
)
def test_http_boundary_audit_remains_owned_and_private(
    tmp_path, monkeypatch, backend, case, recwarn, caplog, capsys, request
):
    canary = "audit-private-canary"
    postgres_dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None

    async def scenario():
        async with audit_publication_fixture(backend, tmp_path, postgres_dsn) as (store, bootstrap):
            record = await BrowserControlPublisher(store).publish(bootstrap)
            control = coordinator(store, Policy(True))
            if backend == "postgres":
                # PostgreSQL's commit guard uses its real database clock.
                control._clock = lambda: datetime.now(UTC)
            if case == "host_secret":
                control._redactor = SecretRedactor(canary)
            daemon = _InteractiveDaemon(record.identity.browser_session_id)

            async def storage_state(*, indexed_db):
                assert indexed_db is False
                return {
                    "cookies": [],
                    "origins": [
                        {
                            "origin": "https://site.test",
                            "localStorage": [{"name": "token", "value": canary}],
                        }
                    ],
                }

            daemon.context = SimpleNamespace(storage_state=storage_state)
            daemon.visual_worker_instance = record.identity.worker_instance_id
            daemon.control = GuestControlFence(
                worker_instance=record.identity.worker_instance_id,
                monotonic=lambda: 1.0,
                wall_clock=time.time if backend == "postgres" else lambda: 1.0,
            )
            if case == "guest_secret":
                daemon.profile_output_values = ()
                daemon.profile_plaintext_limit = 65536
                daemon.profile_timeout_seconds = 5
                daemon.control.capture_restricted = True
            channel = GuestControlChannel(daemon, scope_sha256="a" * 64)
            channel._binding = "b" * 64
            daemon.claim_operator_channel(channel._nonce)
            await daemon.bind_operator_control("b" * 64)
            origin = f"https://{canary}.test" if case.endswith("secret") else "https://site.test"
            daemon.pages["page"] = _InteractivePage(
                page=SimpleNamespace(url=f"{origin}/private-path?token={canary}"),
                session_id=daemon.session_id,
                page_id="page",
                lifecycle="active",
                revision="revision",
            )
            daemon.active_page_id = "page"
            queue = asyncio.Queue()
            native_commands = []

            class Connection:
                async def send(self, raw):
                    message = json.loads(raw)
                    result = await channel._command(message, self)
                    native_commands.append(message["kind"])
                    corrupt_reply = (
                        case in _MALFORMED_CASES and message["kind"] == "takeover"
                    ) or (case in _MALFORMED_HANDBACK_CASES and message["kind"] == "handback")
                    if corrupt_reply:
                        malformed_case = case.removeprefix("handback_")
                        audit = result["audit"]
                        if malformed_case == "malformed_origin":
                            audit["locations"][0]["origin"] = f"https://site.test/{canary}"
                        elif malformed_case == "malformed_missing":
                            del result["audit"]
                        elif malformed_case == "malformed_request":
                            audit["request_id"] = "bt_" + "f" * 32
                        elif malformed_case == "malformed_phase":
                            audit["phase"] = (
                                "acquired" if message["kind"] == "handback" else "handed_back"
                            )
                        elif malformed_case == "malformed_epoch":
                            audit["control_epoch"] = True
                        else:
                            field, value = {
                                "malformed_page": ("page_id", "another-page"),
                                "malformed_revision": ("revision", "another-revision"),
                                "malformed_page_epoch": ("control_epoch", 2),
                            }[case]
                            audit["locations"][0]["page"][field] = value
                    await queue.put(
                        json.dumps(
                            {
                                "kind": "settled",
                                "channel_id": channel._nonce,
                                "sequence": channel._sequence,
                                **result,
                            }
                        )
                    )

                async def recv(self):
                    return await queue.get()

            commands = BrowserGuestCommandOwner(
                coordinator=control,
                connection=Connection(),
                bound=BoundBrowserGuest(record, channel._nonce, "b" * 64),
            )
            server = FastAPI()
            server.include_router(
                create_browser_control_router(
                    coordinator=control,
                    auth=BasicAuth(username="operator", password="password", tenant="tenant"),
                    sessions=BrowserOperatorSessionTokens(b"k" * 32),
                    allowed_origin="https://operator.test",
                )
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server),
                base_url="https://operator.test",
                auth=httpx.BasicAuth("operator", "password"),
            ) as client:
                token = (await client.post("/browser-control/operator-session")).json()[
                    "operator_session_token"
                ]
                headers = {"X-Cayu-Browser-Operator": token}
                takeover = intent_for(bootstrap).model_copy(update={"maximum_until_ms": 60_000})
                if backend == "postgres":
                    now_ms = int(time.time() * 1000)
                    takeover = takeover.model_copy(
                        update={
                            "requested_at_ms": now_ms,
                            "expires_at_ms": now_ms + 30_000,
                            "maximum_until_ms": now_ms + 60_000,
                        }
                    )
                response = await client.post(
                    "/browser-control/takeover",
                    headers=headers,
                    json=takeover.model_dump(mode="json"),
                )
                assert response.status_code == 200
                if case in _MALFORMED_CASES:
                    with pytest.raises(ValueError) as raised:
                        await commands.step()
                    assert canary not in str(raised.value) + repr(raised.value)
                    await commands.disconnect()
                    _, current = await control._load(record.identity)
                    assert current.state == "control_uncertain"
                    assert current.acquisition_audit is None
                    assert current.control_epoch == record.control_epoch
                    assert canary not in current.model_dump_json()
                    assert native_commands == ["takeover"]
                    assert await control.drain()
                    return
                await commands.step()
                acquired = commands.bound.record
                assert acquired.acquisition_audit is not None
                expected_origin = None if case.endswith("secret") else origin
                assert acquired.acquisition_audit.locations[0].origin == expected_origin
                response = await client.post(
                    "/browser-control/handback",
                    headers=headers,
                    json={
                        "identity": acquired.identity.model_dump(mode="json"),
                        "expected_record_revision": acquired.revision,
                        "expected_control_epoch": acquired.control_epoch,
                        "request_id": takeover.request_id,
                    },
                )
                assert response.status_code == 200
                if case in _MALFORMED_HANDBACK_CASES:
                    with pytest.raises(ValueError) as raised:
                        await commands.step()
                    assert canary not in str(raised.value) + repr(raised.value)
                    assert commands.publication_transition is None
                    await commands.disconnect()
                    _, current = await coordinator(store, Policy(True))._load(record.identity)
                    assert current.state == "control_uncertain"
                    assert current.acquisition_audit == acquired.acquisition_audit
                    assert current.handback_audit is None
                    assert current.control_epoch == acquired.control_epoch
                    assert canary not in current.model_dump_json()
                    assert native_commands == ["takeover", "handback"]
                    assert await control.drain()
                    return
                entered, release = asyncio.Event(), asyncio.Event()
                original = store.publish_session_operation_guarded_with_store_time
                original_read = store.load_session_operation
                writes = []
                publication_failure = OSError("publication acknowledgement lost")
                readback_failure = OSError("receipt read unavailable")
                readback_failures = []

                async def readback(*args, **kwargs):
                    if case == "failed_handback_readback" and writes and not readback_failures:
                        readback_failures.append(readback_failure)
                        raise readback_failure
                    return await original_read(*args, **kwargs)

                async def publication(*args, **kwargs):
                    result = await original(*args, **kwargs)
                    writes.append(result)
                    if len(writes) == 1:
                        entered.set()
                        if case == "cancel_handback":
                            await release.wait()
                        elif case in {"lost_handback_ack", "failed_handback_readback"}:
                            raise publication_failure
                    return result

                monkeypatch.setattr(
                    store, "publish_session_operation_guarded_with_store_time", publication
                )
                monkeypatch.setattr(store, "load_session_operation", readback)
                task = asyncio.create_task(commands.step())
                try:
                    if case == "cancel_handback":
                        async with asyncio.timeout(5):
                            await entered.wait()
                        assert commands.publication_transition is not None
                        assert commands.publication_transition[1].handback_audit is not None
                        assert task.cancel("operator connection ended")
                        release.set()
                        with pytest.raises(asyncio.CancelledError):
                            await task
                        assert task.cancelled() and task.cancelling() == 1
                        await commands.disconnect()
                    elif case == "failed_handback_readback":
                        with pytest.raises(ExceptionGroup) as raised:
                            await task
                        assert raised.value.exceptions == (publication_failure, readback_failure)
                        assert not task.cancelled() and task.cancelling() == 0
                        assert commands.publication_transition is not None
                        assert commands.publication_transition[1].handback_audit is not None
                        await commands.disconnect()
                    else:
                        await task
                        assert commands.publication_transition is None
                finally:
                    release.set()
                    await asyncio.gather(task, return_exceptions=True)
                _, current = await coordinator(store, Policy(True))._load(record.identity)
                assert current.state == (
                    "control_uncertain"
                    if case in {"cancel_handback", "failed_handback_readback"}
                    else "agent_controlled"
                )
                assert current.acquisition_audit == acquired.acquisition_audit
                assert current.handback_audit is not None
                assert current.handback_audit.locations[0].origin == expected_origin
                assert current.handback_audit.control_epoch == current.control_epoch
                assert current.fresh_observation_required
                assert canary not in current.model_dump_json()
                assert native_commands == ["takeover", "handback"]
                assert len(writes) == (
                    2 if case in {"cancel_handback", "failed_handback_readback"} else 1
                )
                assert await control.drain()

    asyncio.run(scenario())
    captured = capsys.readouterr()
    assert canary not in captured.out + captured.err + caplog.text
    assert not recwarn
