import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

from cayu import AgentSpec, CayuApp, Message, ModelStreamEvent, RunRequest, ScriptedModelProvider
from cayu.model_policy import (
    InMemoryModelPolicyStore,
    ModelPolicy,
    ModelPolicyController,
    PolicyChannel,
)
from cayu.runtime._policy_wire import canonical, decode
from cayu.sessions.base import ModelTarget
from cayu.sessions.outcomes import run_to_completion

VECTORS = json.loads(
    (Path(__file__).parents[1] / "fixtures/model_policy/contract.json").read_text()
)
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class Channel(PolicyChannel):
    def __init__(self):
        self._scope = {**VECTORS["effective"]["scope"], "instance_id": uuid4().hex}
        self.model = "model-a"
        self.allowed_models = ["model-a", "model-b"]
        self.revision = 1
        self.receipts = {}
        self.fail_ack = False
        self.unavailable = False

    @property
    def scope(self):
        return self._scope.copy()

    @property
    def incarnation(self):
        return "incarnation-1", 1

    async def read_snapshot(self):
        if self.unavailable:
            raise OSError("management offline")
        now = datetime.now(UTC)
        effective = {
            **VECTORS["effective"],
            "scope": self.scope,
            "effective_revision": self.revision,
            "default_model": self.model,
            "allowed_models": self.allowed_models,
            "default_state": "absent"
            if self.model is None
            else ("eligible" if self.model in self.allowed_models else "ineligible"),
        }
        return canonical(
            {
                "schema_version": 1,
                "kind": "policy_snapshot",
                "snapshot_id": uuid4().hex,
                "incarnation_id": "incarnation-1",
                "incarnation_epoch": 1,
                "effective": effective,
                "config_sha256": sha256(canonical(effective)).hexdigest(),
                "issued_at": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "valid_until": (now + timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            }
        )

    async def report(self, report):
        if self.unavailable:
            raise OSError("management offline")
        value = decode(report)
        key = value["operation_id"]
        if key not in self.receipts:
            now = datetime.now(UTC)
            self.receipts[key] = canonical(
                {
                    "schema_version": 1,
                    "kind": "refusal_receipt"
                    if value["kind"] == "adoption_refusal"
                    else "report_receipt",
                    "receipt_id": uuid4().hex,
                    "report": value,
                    "report_sha256": sha256(report).hexdigest(),
                    "accepted_at": now.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                    "knowledge_valid_until": (now + timedelta(seconds=300)).strftime(
                        "%Y-%m-%dT%H:%M:%S.%fZ"
                    ),
                }
            )
        assert decode(self.receipts[key])["report"] == value
        if self.fail_ack:
            raise OSError("lost acknowledgement")
        return self.receipts[key]


class CatalogProvider(ScriptedModelProvider):
    async def get_models(self):
        return [{"id": "model-a"}, {"id": "model-b"}]


def make_app(store, channel, *, session_store=None, interval=20, offline=False):
    provider = CatalogProvider(
        response_factory=lambda request: [
            ModelStreamEvent.text_delta("ok"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    controller = ModelPolicyController(
        store=store,
        agent_name="assistant",
        provider_name=provider.name,
        scope=channel.scope,
        incarnation=channel.incarnation,
        channel=None if offline else channel,
    )
    app = CayuApp(
        model_policy=ModelPolicy([controller], poll_interval=interval),
        session_store=session_store,
        enable_logging=False,
    )
    app.register_provider(provider, default=True)
    app.register_agent(AgentSpec(name="assistant", model="spec-model"))
    return app, controller, provider


@pytest.fixture(params=["memory", "sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def store_factory(request, tmp_path):
    if request.param == "memory":
        store = InMemoryModelPolicyStore()
        return lambda: store
    if request.param == "sqlite":
        from cayu.storage.model_policy_sqlite import SQLiteModelPolicyStore

        return lambda: SQLiteModelPolicyStore(tmp_path / "policy.db")
    from cayu.storage.migrations import SchemaMode
    from cayu.storage.model_policy_postgres import PostgresModelPolicyStore

    dsn = request.getfixturevalue("postgres_dsn")
    return lambda: PostgresModelPolicyStore(dsn, schema_mode=SchemaMode.CREATE)


async def test_public_run_records_policy_and_does_not_rewrite_existing_session(
    store_factory, request, tmp_path
):
    store = store_factory()
    channel = Channel()
    session_store = None
    if type(store).__name__ == "SQLiteModelPolicyStore":
        from cayu.storage.sqlite import SQLiteSessionStore

        session_store = SQLiteSessionStore(tmp_path / "sessions.sqlite")
    elif type(store).__name__ == "PostgresModelPolicyStore":
        from cayu.storage.migrations import SchemaMode
        from cayu.storage.postgres import PostgresSessionStore

        session_store = PostgresSessionStore(
            request.getfixturevalue("postgres_dsn"), schema_mode=SchemaMode.CREATE
        )
    app, controller, provider = make_app(store, channel, session_store=session_store)
    await app.start_model_policy()
    try:
        await controller.poll_once()
        outcome = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "hello")])
        )
        assert outcome.ok, outcome.error
        session = await app.session_store.load(outcome.session_id)
        assert session.model == "model-a"
        policy_events = [e for e in outcome.events if "model_policy" in e.payload]
        assert policy_events and policy_events[-1].payload["model_policy"]["model"] == "model-a"
        channel.model, channel.revision = "model-b", 2
        await controller.poll_once()
        assert (
            app.resolve_run_model_target(RunRequest(agent_name="assistant", messages=[])).model
            == "model-b"
        )
        second = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "again")])
        )
        assert second.ok, second.error
        assert (await app.session_store.load(second.session_id)).model == "model-b"
        assert (await app.session_store.load(outcome.session_id)).model == "model-a"
        channel.unavailable = True
        explicit = RunRequest(
            agent_name="assistant",
            messages=[],
            target=ModelTarget(provider_name=provider.name, model="explicit"),
        )
        assert app.resolve_run_model_target(explicit).model == "explicit"
        assert len(channel.receipts) == 2
    finally:
        await app.stop_model_policy()
        await store.close()
        if session_store is not None:
            await session_store.close()


async def test_restart_resends_exact_report_and_restores_default(store_factory):
    channel = Channel()
    store = store_factory()
    app, controller, _ = make_app(store, channel)
    channel.fail_ack = True
    await app.start_model_policy()
    try:
        channel.fail_ack = True
        with pytest.raises(OSError):
            await controller.poll_once()
        assert controller.selection().target.model == "model-a"
        assert len(channel.receipts) == 1
    finally:
        await app.stop_model_policy()
        await store.close()
    channel.fail_ack = False
    store = store_factory()
    app, controller, _ = make_app(store, channel)
    await app.start_model_policy()
    try:
        assert controller.selection().target.model == "model-a"
        await controller.adopt_cached()
        await controller.poll_once()
        assert not controller._owner.pending()
        assert len(channel.receipts) == 1
    finally:
        await app.stop_model_policy()
        await store.close()


async def test_background_poll_and_offline_override_pause():
    store, channel = InMemoryModelPolicyStore(), Channel()
    app, controller, _ = make_app(store, channel, interval=0.02)
    await app.start_model_policy()
    try:
        async with asyncio.timeout(3):
            while not channel.receipts:
                await asyncio.sleep(0.01)
        await controller.override_default(model="local")
        channel.model, channel.revision = "model-b", 2
        await controller.poll_once()
        assert controller.selection().target.model == "local"
        await controller.resume_adoption()
        await controller.poll_once()
        assert controller.selection().target.model == "model-b"
        channel.model, channel.revision = None, 3
        await controller.poll_once()
        assert decode(list(channel.receipts.values())[-1])["report"]["reason"] == "default_absent"
        assert controller.selection().target.model == "model-b"
    finally:
        await app.stop_model_policy()


async def test_offline_override_and_withdraw():
    store, channel = InMemoryModelPolicyStore(), Channel()
    app, controller, _ = make_app(store, channel, offline=True)
    await app.start_model_policy()
    try:
        await controller.override_default(model="offline")
        await controller.withdraw()
        assert controller.selection().target.model == "offline"
    finally:
        await app.stop_model_policy()


@pytest.mark.parametrize(
    "reason", ["default_absent", "default_ineligible", "model_unknown", "model_unsupported"]
)
async def test_startup_reports_refusal_without_changing_agent_default(reason, monkeypatch):
    channel = Channel()
    if reason == "default_absent":
        channel.model = None
    elif reason == "default_ineligible":
        channel.allowed_models = ["model-b"]
    elif reason == "model_unknown":
        channel.model = "unknown"
        channel.allowed_models.append("unknown")
    app, controller, provider = make_app(InMemoryModelPolicyStore(), channel)
    if reason == "model_unsupported":

        def reject(*, model):
            if model == "model-a":
                raise ValueError("unsupported target")

        monkeypatch.setattr(provider, "preflight_model_target", reject)
    async with app.model_policy_lifespan():
        assert controller.status == "ready"
        assert controller.selection() is None
        assert (
            app.resolve_run_model_target(RunRequest(agent_name="assistant", messages=[])).model
            == "spec-model"
        )
        receipt = decode(next(iter(channel.receipts.values())))
        assert receipt["report"]["reason"] == reason
        assert not controller._owner.pending()


async def test_multiple_agents_select_independent_defaults():
    store = InMemoryModelPolicyStore()
    channels = [Channel(), Channel()]
    channels[1].model = "model-b"
    provider = CatalogProvider(
        response_factory=lambda request: [
            ModelStreamEvent.text_delta("ok"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]
    )
    controllers = [
        ModelPolicyController(
            store=store,
            agent_name=name,
            provider_name=provider.name,
            scope=channel.scope,
            incarnation=channel.incarnation,
            channel=channel,
        )
        for name, channel in zip(("first", "second"), channels, strict=True)
    ]
    app = CayuApp(model_policy=ModelPolicy(controllers), enable_logging=False)
    app.register_provider(provider, default=True)
    for name in ("first", "second"):
        app.register_agent(AgentSpec(name=name, model="spec-model"))
    async with app.model_policy_lifespan():
        for name, expected in (("first", "model-a"), ("second", "model-b")):
            outcome = await run_to_completion(
                app,
                RunRequest(
                    agent_name=name,
                    messages=[Message.text("user", "hello")],
                ),
            )
            assert outcome.ok, outcome.error
            assert (await app.session_store.load(outcome.session_id)).model == expected


async def test_lifespan_preserves_real_cancellation_and_cleanup_failure():
    class FailingRelease(InMemoryModelPolicyStore):
        fail_release = True

        async def execute(self, command):
            if command.action == "release" and self.fail_release:
                raise RuntimeError("release failed")
            return await super().execute(command)

    store = FailingRelease()
    app, _, _ = make_app(store, Channel())
    entered = asyncio.Event()

    async def work():
        async with app.model_policy_lifespan():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(work())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert task.cancelled() and task.cancelling() == 1
    assert isinstance(caught.value.__cause__, BaseExceptionGroup)

    def leaves(error):
        if isinstance(error, BaseExceptionGroup):
            return [leaf for child in error.exceptions for leaf in leaves(child)]
        return [error]

    assert [str(error) for error in leaves(caught.value.__cause__)] == ["release failed"]
    store.fail_release = False
    await app.stop_model_policy()


@pytest.mark.parametrize("action", ["claim", "renew"])
async def test_read_only_maintenance_keeps_unexpired_default_available(action):
    class DelayedStore(InMemoryModelPolicyStore):
        delayed = False
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def execute(self, command):
            if self.delayed and command.action == action:
                self.entered.set()
                await self.finish.wait()
            return await super().execute(command)

    store = DelayedStore()
    app, controller, _ = make_app(store, Channel())
    await app.start_model_policy()
    store.delayed = True
    task = asyncio.create_task(
        controller._owner.reconcile() if action == "claim" else controller._owner.renew()
    )
    try:
        await store.entered.wait()
        assert (
            app.resolve_run_model_target(RunRequest(agent_name="assistant", messages=[])).model
            == "model-a"
        )
    finally:
        store.finish.set()
        await task
        await app.stop_model_policy()


async def test_cancellation_during_cleanup_keeps_body_failure():
    class DelayedRelease(InMemoryModelPolicyStore):
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def execute(self, command):
            if command.action == "release":
                self.entered.set()
                await self.finish.wait()
            return await super().execute(command)

    store = DelayedRelease()
    app, _, _ = make_app(store, Channel())

    async def work():
        async with app.model_policy_lifespan():
            raise ValueError("body failed")

    task = asyncio.create_task(work())
    await store.entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    store.finish.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await task
    assert task.cancelled() and task.cancelling() == 1
    assert isinstance(caught.value.__cause__, BaseExceptionGroup)
    assert [str(error) for error in caught.value.__cause__.exceptions] == ["body failed"]
    assert not app.model_policy._started


async def test_server_lifespan_owns_policy_workers():
    from cayu.model_policy import PolicyContractError
    from cayu.server import ServerConfig, create_server

    app, controller, _ = make_app(InMemoryModelPolicyStore(), Channel())
    server = create_server(app, config=ServerConfig.local_development())
    async with server.router.lifespan_context(server):
        assert controller.selection().target.model == "model-a"
    assert controller.status == "stopped"
    assert not app.model_policy._tasks
    with pytest.raises(PolicyContractError):
        controller.selection()


async def test_fresh_process_restores_installed_default_and_pending_report(tmp_path):
    from cayu.storage.model_policy_sqlite import SQLiteModelPolicyStore

    path = tmp_path / "policy.sqlite"
    store, channel = SQLiteModelPolicyStore(path), Channel()
    channel.fail_ack = True
    app, controller, _ = make_app(store, channel)
    await app.start_model_policy()
    pending = controller._owner.pending()
    assert len(pending) == 1
    await app.stop_model_policy()
    await store.close()
    script = """
import asyncio, json, sys
from cayu import ScriptedModelProvider
from cayu.model_policy import ModelPolicyController
from cayu.storage.model_policy_sqlite import SQLiteModelPolicyStore
async def main():
    store = SQLiteModelPolicyStore(sys.argv[1])
    controller = ModelPolicyController(store=store, agent_name='assistant',
        provider_name='scripted', scope=json.loads(sys.argv[2]),
        incarnation=('incarnation-1', 1))
    await controller.start(ScriptedModelProvider([]))
    try:
        assert controller.selection().target.model == 'model-a'
        assert controller._owner.pending() == (sys.argv[3].encode(),)
        print('restored')
    finally:
        await controller.close()
        await store.close()
asyncio.run(main())
"""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(path),
        json.dumps(channel.scope),
        pending[0].decode(),
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(30):
            stdout, stderr = await process.communicate()
        assert process.returncode == 0, stderr.decode()
        assert stdout == b"restored\n" and not stderr
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()
