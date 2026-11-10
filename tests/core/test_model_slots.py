"""Model slots: declared by the application, bound in Cayu Cloud, resolved when a session starts."""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import warnings
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from tests.core._execution_profile_fixtures import versioned_test_provider_identity
from tests.core.test_active_invocation_execution_profiles import RecordingAdoptionPolicy
from tests.core.test_cloud_inventory import PRICE, _Cloud, _config, _usage_app
from tests.core.test_durable_subagents import (
    _DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
    _collect,
    _DurableSubagentProvider,
    _RecordingPreparedSubagentDispatcher,
)
from tests.core.test_gateway_provider import chunk, provider_for, response, wire
from tests.core.test_managed_settings_controller import RELEASE, FakeCloud, _controller
from tests.core.test_model_failover_recovery import _RecoveryProvider
from tests.core.test_model_policy_runtime import VECTORS

from cayu import AgentSpec, CayuApp, EventType, Message, ResumeRequest, RunRequest
from cayu.approvals.tools import ResolutionActor, ResolutionActorSource
from cayu.cloud_inventory import CloudInventoryPublisher
from cayu.connections import (
    CloudConnectionBroker,
    ConnectionCallError,
    ModelSlot,
    ModelSlots,
    ModelSlotUnavailable,
)
from cayu.connections.models import parse_model_slots, session_model_slot
from cayu.evals import ScriptedModelProvider
from cayu.managed_settings import model_setting
from cayu.providers import ModelStreamEvent
from cayu.providers.base import ModelRequest
from cayu.providers.retry_policy import RetryPolicy
from cayu.releases import ReleaseIdentity
from cayu.runtime.checks import check_manifest
from cayu.runtime.execution_profiles import ExecutionProfileAdoptionIntent
from cayu.server import ServerConfig, create_server
from cayu.sessions import (
    ForkExecutionProfileSelection,
    ForkSessionRequest,
    InMemorySessionStore,
    ModelFailoverPolicy,
    ModelTarget,
    SessionQuery,
)
from cayu.sessions.outcomes import run_to_completion
from cayu.sessions.records import SessionIdentity
from cayu.sessions.requests import bind_runtime_model_slot
from cayu.tasks import InMemoryTaskStore
from cayu.tools.subagents import SubagentExecutionMode, SubagentSpec, SubagentTool


class FakeBroker:
    """The broker's listing of one Agent's connections, with its model slots."""

    def __init__(self) -> None:
        self.answer: dict = {"items": [], "models": [], "approved_models": None}
        self.unreachable = False
        self.delay: float | None = None
        self.reads = 0
        self.releases: list[str | None] = []

    async def listing(self, *, release: str | None = None) -> dict:
        self.reads += 1
        self.releases.append(release)
        if self.unreachable:
            raise ConnectionCallError("The connection broker is unreachable (ConnectError).")
        if self.delay is not None:
            await asyncio.sleep(self.delay)
        return copy.deepcopy(self.answer)

    def bind(self, slot: str, model: str, *, approved: bool | None = True) -> None:
        self.answer["models"] = [item for item in self.answer["models"] if item["slot"] != slot] + [
            {"slot": slot, "model": model, "source": "bound", "approved": approved}
        ]


class _Provider(ScriptedModelProvider):
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        super().__init__(response_factory=self._respond, name="scripted")

    @property
    def execution_profile_identity(self):
        # Stable across apps, as a redeployed Release's provider is.
        return versioned_test_provider_identity(self)

    def _respond(self, request: ModelRequest):
        self.requests.append(request)
        return [
            ModelStreamEvent.text_delta("ok"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]


@pytest.fixture(autouse=True)
def _outside_cloud(monkeypatch):
    for name in ("CAYU_CONNECTIONS_URL", "CAYU_CONNECTIONS_TOKEN", "CAYU_CLOUD_REGISTRY_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def _app(broker: FakeBroker | None = None, **kwargs) -> tuple[CayuApp, _Provider]:
    kwargs.setdefault("model_slots", None if broker is None else ModelSlots(broker))
    app = CayuApp(enable_logging=False, **kwargs)
    provider = _Provider()
    app.register_provider(provider, default=True)
    app.require_model("primary", default="model-a", label="Primary model")
    app.register_agent(AgentSpec(name="assistant", model="model-x"), model_slot="primary")
    return app, provider


def _started(outcome) -> dict:
    return next(e for e in outcome.events if e.type is EventType.SESSION_STARTED).payload


async def _run(app: CayuApp, **fields):
    return await run_to_completion(
        app, RunRequest(agent_name="assistant", messages=[Message.text("user", "go")], **fields)
    )


def test_declarations_are_validated_and_slot_names_unique():
    app = CayuApp(enable_logging=False)
    declared = app.require_model("fast", default="bedrock/claude-haiku-4-5", label="Fast")
    assert declared.declaration() == {
        "name": "fast",
        "default": "bedrock/claude-haiku-4-5",
        "label": "Fast",
    }
    assert app.require_model("review", default=None).declaration() == {
        "name": "review",
        "default": None,
    }
    for name, default, label in (
        ("Fast", "m", None),
        ("-fast", "m", None),
        ("x" * 64, "m", None),
        ("other", "has space", None),
        ("other", "m" * 129, None),
        ("other", "m", "two\nlines"),
        ("other", "m", "x" * 81),
    ):
        with pytest.raises(ValueError):
            app.require_model(name, default=default, label=label)
    with pytest.raises(ValueError, match="more than once"):
        app.require_model("fast", default="m")
    with pytest.raises(ValueError, match="lowercase"):
        app.register_agent(AgentSpec(name="a", model="m"), model_slot="Fast")
    assert [slot.name for slot in app.model_slots()] == ["fast", "review"]


def test_an_application_declares_at_most_64_slots():
    app = CayuApp(enable_logging=False)
    for index in range(64):
        app.require_model(f"slot-{index}", default="m")
    with pytest.raises(ValueError, match="at most 64"):
        app.require_model("one-more", default="m")


def test_an_undeclared_slot_is_refused_at_describe_and_at_session_start():
    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(_Provider(), default=True)
        app.register_agent(AgentSpec(name="assistant", model="m"), model_slot="primary")
        with pytest.raises(ValueError, match="doesn't declare: primary"):
            app.describe()
        outcome = await _run(app)
        assert not outcome.ok
        assert "require_model" in outcome.error
        app.require_model("primary", default="m")
        assert app.describe().agents[0].model_slot == "primary"
        await app.aclose()

    asyncio.run(scenario())


def test_outside_cloud_a_slot_uses_its_default_and_requests_name_it():
    async def scenario():
        app, provider = _app()
        await app.start_model_slots()
        outcome = await _run(app)
        assert outcome.ok, outcome.error
        assert (await app.session_store.load(outcome.session_id)).model == "model-a"
        assert _started(outcome)["model_slot"] == {"slot": "primary", "source": "default"}
        assert {request.model_slot for request in provider.requests} == {"primary"}
        await app.aclose()

    asyncio.run(scenario())


def test_a_binding_beats_the_default_and_a_change_applies_to_new_sessions_only():
    async def scenario():
        broker = FakeBroker()
        broker.bind("primary", "model-b")
        app, provider = _app(broker)
        await app.start_model_slots()
        first = await _run(app)
        assert first.ok, first.error
        assert (await app.session_store.load(first.session_id)).model == "model-b"
        assert _started(first)["model_slot"] == {"slot": "primary", "source": "bound"}

        broker.bind("primary", "model-c")
        await app._model_slots_reader.poll_once()
        provider.requests.clear()
        resumed = await run_to_completion(
            app,
            ResumeRequest(session_id=first.session_id, messages=[Message.text("user", "more")]),
        )
        assert resumed.ok, resumed.error
        assert {request.model for request in provider.requests} == {"model-b"}

        second = await _run(app)
        assert (await app.session_store.load(second.session_id)).model == "model-c"
        await app.aclose()

    asyncio.run(scenario())


def test_an_explicit_target_wins_over_the_slot():
    async def scenario():
        broker = FakeBroker()
        broker.bind("primary", "model-b")
        app, provider = _app(broker)
        await app.start_model_slots()
        outcome = await _run(app, target=ModelTarget(provider_name="scripted", model="model-z"))
        assert outcome.ok, outcome.error
        session = await app.session_store.load(outcome.session_id)
        assert session.model == "model-z"
        assert "model_slot" not in _started(outcome)
        assert "cayu:model_slot" not in session.metadata
        # Its model didn't come from the slot, so its requests don't name it.
        assert {request.model_slot for request in provider.requests} == {None}
        await app.aclose()

    asyncio.run(scenario())


def test_bindings_never_read_use_the_default_and_say_so():
    async def scenario():
        broker = FakeBroker()
        broker.unreachable = True
        app, _ = _app(broker)
        await app.start_model_slots()
        assert app._model_slots_reader.status == "unavailable"
        outcome = await _run(app)
        assert outcome.ok, outcome.error
        assert (await app.session_store.load(outcome.session_id)).model == "model-a"
        assert _started(outcome)["model_slot"] == {"slot": "primary", "source": "unread"}
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("models", "approved_models", "code"),
    [
        (
            [{"slot": "primary", "model": "model-b", "source": "bound", "approved": False}],
            None,
            "model_not_approved",
        ),
        (
            [{"slot": "primary", "model": "model-a", "source": "default", "approved": False}],
            ["model-b"],
            "model_not_approved",
        ),
    ],
)
def test_a_model_that_isnt_approved_refuses_new_sessions(models, approved_models, code):
    async def scenario():
        broker = FakeBroker()
        broker.answer.update(models=models, approved_models=approved_models)
        app, provider = _app(broker)
        await app.start_model_slots()
        request = RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
        with pytest.raises(ModelSlotUnavailable) as refused:
            app.resolve_run_model_target(request)
        assert refused.value.code == code
        assert refused.value.slot == "primary"
        outcome = await _run(app)
        assert outcome.error_detail.error_type.endswith("ModelSlotUnavailable")
        assert "isn't approved" in outcome.error
        assert provider.requests == []
        assert (await app.session_store.list_sessions(SessionQuery())).sessions == []
        await app.aclose()

    asyncio.run(scenario())


def test_a_slot_without_a_default_needs_a_binding():
    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(_Provider(), default=True)
        app.require_model("primary", default=None)
        app.register_agent(AgentSpec(name="assistant", model="m"), model_slot="primary")
        request = RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
        with pytest.raises(ModelSlotUnavailable) as refused:
            app.resolve_run_model_target(request)
        assert refused.value.code == "model_slot_unbound"
        assert not (await _run(app)).ok
        await app.aclose()

    asyncio.run(scenario())


def test_the_reader_keeps_its_last_good_snapshot():
    async def scenario():
        broker = FakeBroker()
        broker.bind("primary", "model-b")
        reader = ModelSlots(broker, timeout_seconds=0.05)
        await reader.poll_once()
        good = reader.snapshot
        assert good is not None and good.bindings["primary"].model == "model-b"

        broker.answer["models"] = [{"slot": "primary", "source": "bound"}]
        await reader.poll_once()
        assert (reader.status, reader.snapshot) == ("unavailable", good)

        broker.bind("primary", "model-c")
        broker.delay = 1.0
        await reader.poll_once()
        assert (reader.status, reader.snapshot) == ("unavailable", good)

        broker.delay = None
        broker.unreachable = True
        await reader.poll_once()
        assert reader.snapshot is good
        await reader.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "answer",
    [
        {"models": {}},
        {"models": [{"slot": "Primary", "model": "m", "source": "bound", "approved": True}]},
        {"models": [{"slot": "p", "model": "m", "source": 1, "approved": True}]},
        {"models": [{"slot": "p", "model": "m", "source": "bound", "approved": "yes"}]},
        {"models": [{"slot": "p", "model": "has space", "source": "bound", "approved": True}]},
        {
            "models": [
                {"slot": "p", "model": "m", "source": "bound", "approved": True},
                {"slot": "p", "model": "n", "source": "bound", "approved": True},
            ]
        },
        {"models": [], "approved_models": "m"},
        {"models": [], "approved_models": [1]},
    ],
)
def test_a_malformed_listing_is_refused(answer):
    with pytest.raises(ValueError):
        parse_model_slots(answer)


def test_a_listing_from_a_cloud_without_model_slots_binds_nothing():
    snapshot = parse_model_slots({"items": []})
    assert (dict(snapshot.bindings), snapshot.approved_models) == ({}, None)
    snapshot = parse_model_slots(
        {
            "models": [{"slot": "p", "model": None, "source": "unbound", "approved": None}],
            "approved_models": ["m"],
        }
    )
    assert snapshot.bindings["p"].source == "unbound"
    assert snapshot.approved_models == frozenset({"m"})


def test_the_reader_is_configured_by_cloud_and_polls_in_bounds(monkeypatch, caplog):
    assert ModelSlots.from_environment({}) is None
    reader = ModelSlots.from_environment(
        {
            "CAYU_CONNECTIONS_URL": "https://cloud.example/v1/agent-connections",
            "CAYU_CONNECTIONS_TOKEN": "token",
        }
    )
    assert reader is not None and (reader.poll_interval, reader.timeout_seconds) == (25.0, 10.0)
    assert (
        ModelSlots.from_environment(
            {"CAYU_CONNECTIONS_URL": "https://cloud.example/v1/agent-connections"}
        )
        is None
    )
    assert "without its token" in caplog.text
    for interval in (0, 301, True):
        with pytest.raises(ValueError):
            ModelSlots(FakeBroker(), poll_interval=interval)


def test_a_queued_durable_child_keeps_the_model_it_was_prepared_with():
    async def scenario():
        sessions, tasks = InMemorySessionStore(), InMemoryTaskStore()
        broker = FakeBroker()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=_RecordingPreparedSubagentDispatcher(tasks),
            enable_logging=False,
            model_slots=ModelSlots(broker),
        )
        provider = _DurableSubagentProvider()
        app.register_provider(provider, default=True)
        app.require_model("review", default="model")
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer", mode=SubagentExecutionMode.DURABLE
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"), model_slot="review")
        await app.start_model_slots()
        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        assert events[-1].type == EventType.SESSION_COMPLETED
        (child,) = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-parent"))
        ).sessions
        # The slot is bound before a worker runs the queued child: it keeps its model.
        broker.bind("review", "model-2")
        await app._model_slots_reader.poll_once()
        provider.requests.clear()
        handle = await app.dispatcher.process_next(app, worker_id="worker")
        assert not handle.metadata.get("requeued", False)
        child = await sessions.load(child.id)
        assert child.model == "model"
        assert child.metadata["cayu:model_slot"] == {
            "slot": "review",
            "source": "default",
            "model": "model",
        }
        # The child names the slot it was prepared with, and records it when it starts.
        assert {request.model_slot for request in provider.requests} == {"review"}
        started = [
            event
            for event in await sessions.load_events(child.id)
            if event.type is EventType.SESSION_STARTED
        ]
        assert started[0].payload["model_slot"] == {"slot": "review", "source": "default"}
        await app.aclose()

    asyncio.run(scenario())


def test_gateway_requests_carry_the_slot_only_when_the_agent_has_one():
    bodies: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return response(
            wire(chunk({"content": "Hello"}), chunk(finish="stop")),
            content_type="text/event-stream",
        )

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(provider_for(handle), default=True)
        app.require_model("primary", default="bedrock/claude-sonnet-4-5")
        app.register_agent(AgentSpec(name="slotted", model="example/model"), model_slot="primary")
        app.register_agent(AgentSpec(name="plain", model="example/model"))
        for agent in ("slotted", "plain"):
            outcome = await run_to_completion(
                app, RunRequest(agent_name=agent, messages=[Message.text("user", "go")])
            )
            assert outcome.ok, outcome.error
        await app.aclose()

    asyncio.run(scenario())
    slotted, plain = bodies
    assert slotted["model"] == "bedrock/claude-sonnet-4-5"
    assert slotted["cayu"] == {"schema_version": 1, "model_slot": "primary"}
    assert plain["model"] == "example/model"
    assert "cayu" not in plain


def test_a_gateway_refusal_is_final_for_the_session():
    models: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        return response(
            json.dumps({"error": {"code": "model_not_allowed", "message": "no"}}).encode(),
            status=403,
        )

    async def scenario():
        app = CayuApp(enable_logging=False)
        app.register_provider(provider_for(handle), default=True)
        app.require_model("primary", default="bedrock/claude-sonnet-4-5")
        app.register_agent(AgentSpec(name="assistant", model="example/model"), model_slot="primary")
        outcome = await _run(app)
        assert not outcome.ok
        await app.aclose()

    asyncio.run(scenario())
    assert models == ["bedrock/claude-sonnet-4-5"]


def test_model_setting_is_deprecated_and_aliased_to_a_slot_named_after_the_agent():
    with pytest.warns(DeprecationWarning, match="require_model"):
        setting = model_setting()

    async def scenario():
        cloud = FakeCloud()
        cloud.values = {"agent.triage_bot.model": "model-setting"}
        broker = FakeBroker()
        app = CayuApp(
            enable_logging=False,
            managed_settings=_controller(cloud),
            release_identity=ReleaseIdentity(id=RELEASE),
            model_slots=ModelSlots(broker),
        )
        provider = _Provider()
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="triage_bot", model="model-x"), settings=[setting])
        assert app.model_slots() == (ModelSlot(name="triage-bot", default="model-x"),)
        assert app.describe().agents[0].model_slot == "triage-bot"
        await app.start_managed_settings()
        await app.start_model_slots()

        async def run():
            return await run_to_completion(
                app, RunRequest(agent_name="triage_bot", messages=[Message.text("user", "go")])
            )

        # Until the slot is bound, the setting's value applies, and requests name no slot.
        first = await run()
        assert (await app.session_store.load(first.session_id)).model == "model-setting"
        assert "model_slot" not in _started(first)
        assert {request.model_slot for request in provider.requests} == {None}
        # Cloud moved the value to the slot's binding: it wins.
        broker.bind("triage-bot", "model-bound")
        await app._model_slots_reader.poll_once()
        second = await run()
        assert (await app.session_store.load(second.session_id)).model == "model-bound"
        assert _started(second)["model_slot"] == {"slot": "triage-bot", "source": "bound"}

        manifest = app.describe()
        codes = [item.code for item in check_manifest(manifest).diagnostics]
        assert "MODEL_SETTING_DEPRECATED" in codes
        await app.aclose()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.run(scenario())


def test_model_setting_and_a_model_slot_are_not_both_allowed():
    app = CayuApp(enable_logging=False)
    app.require_model("primary", default="m")
    with pytest.warns(DeprecationWarning):
        setting = model_setting()
    with pytest.raises(ValueError, match="drop the deprecated"):
        app.register_agent(
            AgentSpec(name="assistant", model="m"), settings=[setting], model_slot="primary"
        )


def test_an_agent_on_a_slot_has_no_deprecation_diagnostic():
    app, _ = _app()
    codes = [item.code for item in check_manifest(app.describe()).diagnostics]
    assert "MODEL_SETTING_DEPRECATED" not in codes


def test_the_gateway_model_policy_is_deprecated(monkeypatch, tmp_path):
    from cayu.model_policy import (
        InMemoryModelPolicyStore,
        ModelPolicy,
        ModelPolicyController,
        configured_model_policy,
    )

    with pytest.warns(DeprecationWarning, match="ModelPolicyController is deprecated"):
        controller = ModelPolicyController(
            store=InMemoryModelPolicyStore(),
            agent_name="assistant",
            provider_name="scripted",
            scope={**VECTORS["effective"]["scope"], "instance_id": "instance"},
            incarnation=("incarnation", 1),
        )
    with pytest.warns(DeprecationWarning, match="model_policy=...") as caught:
        CayuApp(enable_logging=False, model_policy=ModelPolicy([controller]))
    assert caught[0].filename == __file__
    monkeypatch.setenv("CAYU_MODEL_POLICY_CONFIG", str(tmp_path / "missing.json"))
    with pytest.warns(DeprecationWarning, match="CAYU_MODEL_POLICY_CONFIG"), pytest.raises(OSError):
        configured_model_policy(InMemoryModelPolicyStore())


def test_a_model_request_slot_is_a_slot_name():
    assert ModelRequest(model="m", messages=[], model_slot="primary").model_slot == "primary"
    assert "model_slot" not in ModelRequest(model="m", messages=[]).model_dump()
    with pytest.raises(ValueError):
        ModelRequest(model="m", messages=[], model_slot="Primary")


def test_a_session_keeps_the_slot_it_started_with_across_releases():
    async def scenario():
        sessions = InMemorySessionStore()
        app, _ = _app(session_store=sessions)
        first = await _run(app)
        await app.aclose()

        # A later Release runs the agent without a slot: the session still names its own.
        later = CayuApp(enable_logging=False, session_store=sessions)
        provider = _Provider()
        later.register_provider(provider, default=True)
        later.register_agent(AgentSpec(name="assistant", model="model-x"))
        resumed = await run_to_completion(
            later,
            ResumeRequest(session_id=first.session_id, messages=[Message.text("user", "more")]),
        )
        assert resumed.ok, resumed.error
        assert {request.model_slot for request in provider.requests} == {"primary"}
        plain = await _run(later)
        await later.aclose()

        # And a Release that adds a slot doesn't label a session started without one.
        again, provider = _app(session_store=sessions)
        resumed = await run_to_completion(
            again,
            ResumeRequest(session_id=plain.session_id, messages=[Message.text("user", "more")]),
        )
        assert resumed.ok, resumed.error
        assert {request.model_slot for request in provider.requests} == {None}
        await again.aclose()

    asyncio.run(scenario())


_RECORD = {"slot": "primary", "source": "bound", "model": "m"}


def test_a_caller_cant_forge_the_sessions_slot():
    for request_type, fields in (
        (RunRequest, {"agent_name": "assistant", "messages": []}),
        (ForkSessionRequest, {"source_session_id": "source"}),
    ):
        with pytest.raises(ValueError, match="runtime-owned model-slot authority"):
            request_type(**fields, metadata={"cayu:model_slot": _RECORD})

    async def scenario():
        store = InMemorySessionStore()
        identity = SessionIdentity(provider_name="scripted", model="m")
        # A value set on the request by anything but session preparation is ignored.
        forged = RunRequest(agent_name="assistant", messages=[], session_id="forged")
        forged._runtime_model_slot = dict(_RECORD)
        assert "cayu:model_slot" not in (await store.create(forged, identity=identity)).metadata
        # A bound record is written only for the model the session is created with.
        other = RunRequest(agent_name="assistant", messages=[], session_id="other")
        bind_runtime_model_slot(other, {**_RECORD, "model": "n"})
        assert "cayu:model_slot" not in (await store.create(other, identity=identity)).metadata
        kept = RunRequest(agent_name="assistant", messages=[], session_id="kept")
        bind_runtime_model_slot(kept, _RECORD)
        assert (await store.create(kept, identity=identity)).metadata["cayu:model_slot"] == _RECORD

    asyncio.run(scenario())


def test_a_session_recorded_without_its_slots_model_names_no_slot():
    def session(record):
        return SimpleNamespace(metadata={"cayu:model_slot": record})

    assert session_model_slot(session(_RECORD)) == _RECORD
    assert session_model_slot(session({"slot": "primary", "source": "bound"})) is None
    assert session_model_slot(session({**_RECORD, "source": "chosen"})) is None


def test_a_resumed_session_on_another_model_names_no_slot():
    async def scenario():
        app, provider = _app()
        first = await _run(app)
        provider.requests.clear()
        resumed = await run_to_completion(
            app,
            ResumeRequest(
                session_id=first.session_id,
                messages=[Message.text("user", "more")],
                target=ModelTarget(provider_name="scripted", model="model-z"),
            ),
        )
        assert resumed.ok, resumed.error
        assert {(r.model, r.model_slot) for r in provider.requests} == {("model-z", None)}
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("override", "label"),
    [({}, "primary"), ({"model": "model-z"}, None), ({"agent_name": "other"}, None)],
)
def test_a_fork_names_the_slot_only_while_it_keeps_the_agent_and_the_slots_model(override, label):
    async def scenario():
        app, provider = _app(execution_profile_policy=RecordingAdoptionPolicy())
        # Another agent on the same model as the slot's: a fork onto it names no slot.
        app.register_agent(AgentSpec(name="other", model="model-a"))
        source = await _run(app)
        adoption = (
            {}
            if not override
            else {
                "execution_profile_selection": ForkExecutionProfileSelection.CURRENT_CHILD,
                "profile_adoption": ExecutionProfileAdoptionIntent(
                    idempotency_key="fork-change",
                    reason="Fork onto another agent or model.",
                    requested_by=ResolutionActor(
                        subject="operator", source=ResolutionActorSource.REQUEST
                    ),
                ),
            }
        )
        request = ForkSessionRequest(
            source_session_id=source.session_id, session_id="fork", **override, **adoption
        )
        assert [event async for event in app.fork_session(request)]
        fork = await app.session_store.load("fork")
        assert fork.metadata.get("cayu:model_slot") == (None if label is None else _RECORD_A)
        provider.requests.clear()
        resumed = await run_to_completion(
            app, ResumeRequest(session_id="fork", messages=[Message.text("user", "more")])
        )
        assert resumed.ok, resumed.error
        assert {request.model_slot for request in provider.requests} == {label}
        await app.aclose()

    asyncio.run(scenario())


def test_the_run_route_answers_a_refused_slot_with_its_code():
    broker = FakeBroker()
    broker.bind("primary", "model-b", approved=False)
    app, provider = _app(broker)
    with TestClient(create_server(app, config=ServerConfig.local_development())) as client:
        response = client.post("/api/run", json={"agent": "assistant", "prompt": "go"})
    assert response.status_code == 409
    assert response.json()["detail"].startswith("model_not_approved: ")
    assert provider.requests == []


_RECORD_A = {"slot": "primary", "source": "default", "model": "model-a"}


def test_a_failover_fallback_doesnt_name_the_slot():
    async def scenario():
        app = CayuApp(enable_logging=False)
        primary, backup = _RecoveryProvider("primary"), _RecoveryProvider("backup")
        app.register_provider(primary, default=True)
        app.register_provider(backup)
        app.require_model("primary", default="model-p")
        app.register_agent(
            AgentSpec(name="assistant", model="m", provider_name="primary"), model_slot="primary"
        )
        outcome = await _run(
            app,
            retry_policy=RetryPolicy(max_attempts=1),
            failover=ModelFailoverPolicy(
                fallbacks=(ModelTarget(provider_name="backup", model="large"),),
                max_total_attempts=2,
            ),
        )
        assert outcome.ok, outcome.error
        assert {request.model_slot for request in primary.requests} == {"primary"}
        assert {(request.model, request.model_slot) for request in backup.requests} == {
            ("large", None)
        }
        await app.aclose()

    asyncio.run(scenario())


def test_an_undeclared_slot_fails_startup_but_never_registration():
    async def scenario():
        app = await _usage_app()
        app.register_agent(AgentSpec(name="slotted", model=PRICE.model), model_slot="primary")
        with pytest.raises(ValueError, match="doesn't declare: primary"):
            await app.start_model_slots()
        cloud = _Cloud()
        async with httpx.AsyncClient(transport=httpx.MockTransport(cloud)) as client:
            await CloudInventoryPublisher(app, _config(), client=client).publish_once()
        body = json.loads(next(r for r in cloud.requests if r.method == "PUT").content)
        assert "model_slots" not in body
        assert cloud.current is not None
        await app.aclose()

    asyncio.run(scenario())


def test_the_reader_doesnt_hold_up_startup_and_reads_in_the_background():
    async def scenario():
        broker = FakeBroker()
        broker.bind("primary", "model-b")
        broker.delay = 0.3
        reader = ModelSlots(broker, startup_wait_seconds=0.01)
        app, _ = _app(model_slots=reader)
        await asyncio.wait_for(asyncio.gather(app.start_model_slots(), reader.start()), 0.2)
        assert broker.reads == 1  # one background read, however often start is called
        early = await _run(app)
        assert _started(early)["model_slot"] == {"slot": "primary", "source": "unread"}
        await asyncio.sleep(0.4)
        later = await _run(app)
        assert _started(later)["model_slot"] == {"slot": "primary", "source": "bound"}
        await app.aclose()
        assert reader.status == "stopped"

    asyncio.run(scenario())


def test_the_reader_logs_once_per_outage(caplog):
    async def scenario():
        broker = FakeBroker()
        reader = ModelSlots(broker)
        with caplog.at_level(logging.WARNING, logger="cayu.connections.models"):
            await reader.poll_once()
            broker.unreachable = True
            for _ in range(3):
                await reader.poll_once()
            broker.unreachable = False
            await reader.poll_once()
            broker.unreachable = True
            await reader.poll_once()
        messages = [record.getMessage() for record in caplog.records]
        assert len(messages) == 2
        assert all("(ConnectionCallError)" in message for message in messages)
        assert all("unreachable" not in message for message in messages)

    asyncio.run(scenario())


def test_an_unknown_source_binds_nothing():
    snapshot = parse_model_slots(
        {
            "models": [
                {"slot": "primary", "model": "m", "source": "inherited", "approved": True},
                {"slot": "fast", "model": "f", "source": "bound", "approved": True},
            ]
        }
    )
    assert list(snapshot.bindings) == ["fast"]


def test_require_model_refuses_the_slot_of_an_agent_on_model_setting():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        setting = model_setting()
    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="triage_bot", model="m"), settings=[setting])
    with pytest.raises(ValueError, match="already the slot of agent 'triage_bot'"):
        app.require_model("triage-bot", default="m")

    app = CayuApp(enable_logging=False)
    app.require_model("triage-bot", default="m")
    with pytest.raises(ValueError, match="model_slot='triage-bot' instead"):
        app.register_agent(AgentSpec(name="triage_bot", model="m"), settings=[setting])


def test_a_first_read_that_fails_at_once_doesnt_hold_up_startup():
    async def scenario():
        broker = FakeBroker()
        broker.unreachable = True
        reader = ModelSlots(broker, startup_wait_seconds=5)
        await asyncio.wait_for(reader.start(), 0.5)
        assert reader.status == "unavailable"
        await reader.close()

    asyncio.run(scenario())


def test_the_reader_names_its_release_and_keeps_its_snapshot_when_cloud_refuses_it(
    monkeypatch, caplog
):
    requests: list[httpx.Request] = []
    answers = [
        httpx.Response(200, json={"items": [], "models": [], "approved_models": ["m"]}),
        httpx.Response(
            422, json={"detail": {"code": "release_unknown", "message": "Unknown Release."}}
        ),
        httpx.Response(
            422, json={"detail": {"code": "release_unknown", "message": "Unknown Release."}}
        ),
        httpx.Response(200, json={"items": []}),
    ]

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return answers.pop(0)

    async def scenario():
        broker = CloudConnectionBroker(
            url="https://cloud.example/v1/agent-connections",
            token="token",
            transport=httpx.MockTransport(handle),
        )
        reader = ModelSlots(broker, release_id="rel_1")
        await reader.poll_once()
        good = reader.snapshot
        with caplog.at_level(logging.WARNING, logger="cayu.connections.models"):
            await reader.poll_once()
            await reader.poll_once()
        assert (reader.status, reader.snapshot) == ("unavailable", good)
        assert len(caplog.records) == 1
        assert "release_unknown" not in caplog.text
        await broker.listing()

    asyncio.run(scenario())
    assert [request.url.params.get("release") for request in requests] == [
        "rel_1",
        "rel_1",
        "rel_1",
        None,
    ]


def test_cloud_bound_reader_names_the_apps_release(monkeypatch):
    monkeypatch.setenv("CAYU_CONNECTIONS_URL", "https://cloud.example/v1/agent-connections")
    monkeypatch.setenv("CAYU_CONNECTIONS_TOKEN", "token")
    app = CayuApp(enable_logging=False, release_identity=ReleaseIdentity(id="rel_1"))
    assert app._model_slots_reader is not None
    assert app._model_slots_reader.release_id == "rel_1"
    assert CayuApp(enable_logging=False)._model_slots_reader.release_id is None
