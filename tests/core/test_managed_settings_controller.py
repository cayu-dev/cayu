"""The Agent reads its settings from a fake Cayu Cloud, applies them and reports back."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from cayu import AgentSpec, CayuApp, CayuConfig, Message, ResumeRequest, RunRequest
from cayu.budgets import BudgetLimit, BudgetPolicy
from cayu.configuration import RunDefaults
from cayu.evals import ScriptedModelProvider
from cayu.managed_settings import (
    MAX_REPORT_BYTES,
    HttpManagedSettingsChannel,
    ManagedSetting,
    ManagedSettings,
    fit_report_settings,
    model_setting,
    provider_option_setting,
    thinking_effort_setting,
)
from cayu.providers import ModelStreamEvent, RetryPolicy
from cayu.releases import ReleaseIdentity
from cayu.sessions.outcomes import run_to_completion
from cayu.tools import Tool, ToolContext, ToolResult, ToolSpec

TOKEN = "cayu_reg_settings-token-canary"
RELEASE = "rel_1"


class FakeCloud:
    """Cloud's effective-values and reports routes for one Agent and Release."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.revision = 1
        self.valid_for = timedelta(minutes=5)
        self.schema_missing = False
        self.unreachable = False
        self.reports: list[dict] = []
        self.reads = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if self.unreachable:
            raise httpx.ConnectError("offline")
        if request.method == "GET":
            assert request.url.path == "/v1/applications/scout/settings/effective"
            assert request.url.params["release"] == RELEASE
            self.reads += 1
            if self.schema_missing:
                return httpx.Response(409, json={"code": "settings_schema_missing"})
            return httpx.Response(
                200,
                headers={"Cache-Control": "no-store"},
                json={
                    "contract": "cayu.managed-settings-values/v1",
                    "application_id": "scout",
                    "release_id": RELEASE,
                    "schema_digest": "sha256:" + "0" * 64,
                    "revision": self.revision,
                    "valid_until": (datetime.now(UTC) + self.valid_for).isoformat(),
                    "values": dict(self.values),
                    "conflicts": [],
                    "added_later": True,
                },
            )
        assert request.method == "POST"
        assert request.url.path == "/v1/applications/scout/settings/reports"
        body = json.loads(request.content)
        self.reports.append(body)
        return httpx.Response(202, json={"sequence": body["sequence"]})

    def set(self, **values: object) -> None:
        self.values.update({key.replace("__", "."): value for key, value in values.items()})
        self.revision += 1


def _controller(cloud: FakeCloud, **kwargs) -> ManagedSettings:
    transport = httpx.MockTransport(cloud)
    channel = HttpManagedSettingsChannel(
        api_url="https://cloud.example",
        application_id="scout",
        release_id=RELEASE,
        token=TOKEN,
        transport=transport,
        build_transport=transport,
    )
    return ManagedSettings(channel, **kwargs)


class _Provider(ScriptedModelProvider):
    def __init__(self) -> None:
        self.models: list[str] = []
        super().__init__(response_factory=self._respond, name="scripted")

    def _respond(self, request):
        self.models.append(request.model)
        if request.tools and not any(message.role.value == "tool" for message in request.messages):
            return [
                ModelStreamEvent.tool_call(
                    id="ask", name="ask_user", arguments={"question": "Which one?"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [
            ModelStreamEvent.text_delta("ok"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]

    def preflight_model_target(self, *, model: str) -> None:
        if model.startswith("unknown"):
            raise ValueError(f"unknown model {model}")


def _app(cloud: FakeCloud, *, tools=(), **kwargs) -> tuple[CayuApp, _Provider]:
    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
        **kwargs,
    )
    provider = _Provider()
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="model-a"),
        settings=[model_setting()],
        tools=list(tools),
    )
    return app, provider


def _outcomes(report: dict) -> dict[str, tuple]:
    return {item["key"]: (item["outcome"], item.get("reason")) for item in report["settings"]}


def test_outside_cloud_nothing_polls_and_the_code_defaults_apply(monkeypatch):
    for name in ("CAYU_CLOUD_REGISTRY_TOKEN", "CAYU_CLOUD_APPLICATION_ID", "CAYU_CLOUD_API_URL"):
        monkeypatch.delenv(name, raising=False)
    app = CayuApp(enable_logging=False, config=CayuConfig(run=RunDefaults(max_steps=40)))
    assert app.managed_settings_controller is None
    assert app.config.run.max_steps == 40
    assert ManagedSettings.from_environment(release_id=RELEASE, environ={}) is None
    asyncio.run(app.start_managed_settings())
    asyncio.run(app.aclose())


def test_from_environment_uses_the_registry_credential_and_refuses_a_bad_destination(caplog):
    environ = {
        "CAYU_CLOUD_REGISTRY_TOKEN": TOKEN,
        "CAYU_CLOUD_APPLICATION_ID": "scout",
        "CAYU_CLOUD_API_URL": "https://cloud.example/",
    }
    controller = ManagedSettings.from_environment(release_id=RELEASE, environ=environ)
    assert controller is not None and controller.release_id == RELEASE
    with caplog.at_level(logging.WARNING, logger="cayu.managed_settings"):
        insecure = {**environ, "CAYU_CLOUD_API_URL": "http://cloud.example"}
        assert ManagedSettings.from_environment(release_id=RELEASE, environ=insecure) is None
        assert ManagedSettings.from_environment(release_id=None, environ=environ) is None
    assert TOKEN not in caplog.text


def test_restart_settings_are_read_once_when_the_app_is_built():
    cloud = FakeCloud()
    cloud.values = {
        "config.run.thinking.effort": "high",
        "budget.app.max_estimated_cost": 25,
        "agent.assistant.thinking.effort": "low",
        "agent.assistant.provider_options.temperature": 0.7,
    }
    policy = BudgetPolicy(limits=[BudgetLimit.usd(10, scope="app")])
    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
        budget_policy=policy,
    )
    app.register_provider(_Provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="model-a", provider_options={"temperature": 0.2}),
        settings=[
            thinking_effort_setting(),
            provider_option_setting("temperature", type="number", label="T", maximum=1),
        ],
    )
    assert cloud.reads == 1
    assert app.config.run.thinking is not None and app.config.run.thinking.effort == "high"
    assert app._config_sources["run.thinking"] == "managed"
    assert app.budget_policy is not None
    assert app.budget_policy.limits[0].max_estimated_cost == Decimal(25)
    spec = app.get_agent("assistant").spec
    assert spec.thinking is not None and spec.thinking.effort == "low"
    assert spec.provider_options["temperature"] == 0.7
    # The schema still publishes the code's values as defaults.
    defaults = {item.key: item.default for item in app.managed_settings()}
    assert defaults["budget.app.max_estimated_cost"] == 10
    assert defaults["agent.assistant.provider_options.temperature"] == 0.2
    assert defaults["config.run.thinking.effort"] is None

    async def scenario():
        # A later change waits for the restart Cloud triggers; the report says so.
        cloud.set(config__run__thinking__effort="max")
        await app.start_managed_settings()
        assert app.config.run.thinking.effort == "high"
        report = _outcomes(cloud.reports[-1])
        assert report["config.run.thinking.effort"] == (
            "refused",
            "takes effect when the service restarts",
        )
        assert report["budget.app.max_estimated_cost"] == ("adopted", None)
        await app.aclose()

    asyncio.run(scenario())


def test_a_next_session_change_applies_to_the_next_session_not_a_running_one():
    cloud = FakeCloud()
    cloud.values = {"config.run.max_steps": 40, "agent.assistant.model": "model-a2"}
    app, provider = _app(cloud)

    async def scenario():
        await app.start_managed_settings()
        first = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
        )
        started = next(e for e in first.events if e.type.value == "session.started")
        assert started.payload["managed_settings"] == {
            "revision": 1,
            "values": {"agent.assistant.model": "model-a2", "config.run.max_steps": 40},
        }
        session = await app.session_store.load(first.session_id)
        assert session.model == "model-a2"
        assert first.ok, first.error

        # Cloud changes both before the first session's next turn.
        cloud.set(config__run__max_steps=60, agent__assistant__model="model-b")
        await app.managed_settings_controller.poll_once()
        effective = await app.inspect_effective_run_configuration(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "x")])
        )
        assert effective.max_steps.value == 60
        assert effective.max_steps.source == "managed"
        assert app.resolve_run_model_target(
            RunRequest(agent_name="assistant", messages=[])
        ).model == ("model-b")
        # An explicit request still wins over Cloud.
        explicit = await app.inspect_effective_run_configuration(
            RunRequest(agent_name="assistant", messages=[], max_steps=7)
        )
        assert explicit.max_steps.value == 7

        # The running session resumes with what it started with: its model, and an
        # execution profile that still matches.
        provider.models.clear()
        resumed = await run_to_completion(
            app,
            ResumeRequest(session_id=first.session_id, messages=[Message.text("user", "more")]),
        )
        assert resumed.ok, resumed.error
        assert set(provider.models) == {"model-a2"}
        assert (await app.session_store.load(first.session_id)).model == "model-a2"

        second = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
        )
        assert (await app.session_store.load(second.session_id)).model == "model-b"
        started = next(e for e in second.events if e.type.value == "session.started")
        assert started.payload["managed_settings"]["revision"] == 2
        assert started.payload["managed_settings"]["values"]["config.run.max_steps"] == 60

        report = cloud.reports[-1]
        assert report["revision"] == 2
        assert report["state"] == "applied"
        assert _outcomes(report)["config.run.max_steps"] == ("adopted", None)
        await app.aclose()

    asyncio.run(scenario())


def test_reports_say_what_each_setting_did_and_are_sent_only_when_something_changes():
    cloud = FakeCloud()
    cloud.values = {"agent.assistant.model": "unknown-model", "config.run.max_steps": 0}
    app, _ = _app(cloud)
    app.declare_settings([ManagedSetting(key="search_key", type="secret_ref", label="Key")])
    cloud.set(app__search_key="MISSING_SEARCH_KEY")

    async def scenario():
        controller = app.managed_settings_controller
        await app.start_managed_settings(role="worker")
        report = cloud.reports[-1]
        assert report["contract"] == "cayu.managed-settings-report/v1"
        assert report["release_id"] == RELEASE
        assert report["process"] == {"id": controller.process_id, "role": "worker"}
        assert report["sequence"] == 1
        assert datetime.fromisoformat(report["reported_at"]).tzinfo is not None
        outcomes = _outcomes(report)
        assert outcomes["agent.assistant.model"] == (
            "refused",
            "the agent's provider does not accept this model",
        )
        assert outcomes["config.run.max_steps"][0] == "refused"
        assert outcomes["app.search_key"] == (
            "refused",
            "the environment variable is not set on the Agent",
        )
        assert outcomes["config.run.limits.max_tool_calls"] == ("default", None)
        # Refused values fall back to the code defaults.
        assert app.resolve_run_model_target(
            RunRequest(agent_name="assistant", messages=[])
        ).model == ("model-a")
        assert app.config.run.max_steps == 1000
        assert app.managed_setting_value("app.search_key") is None

        await controller.poll_once()
        assert len(cloud.reports) == 1
        cloud.set(agent__assistant__model="model-c")
        await controller.poll_once()
        assert len(cloud.reports) == 2
        assert cloud.reports[-1]["sequence"] == 2
        assert _outcomes(cloud.reports[-1])["agent.assistant.model"] == ("adopted", None)
        await app.aclose()

    asyncio.run(scenario())


def test_without_a_schema_the_defaults_apply_and_the_process_reports_unreported():
    cloud = FakeCloud()
    cloud.schema_missing = True
    app, _ = _app(cloud)

    async def scenario():
        await app.start_managed_settings()
        controller = app.managed_settings_controller
        assert controller.status == "schema_missing"
        report = cloud.reports[-1]
        assert report["state"] == "unreported"
        assert report["revision"] is None
        assert {item["outcome"] for item in report["settings"]} == {"default"}
        # Once the Release's schema reaches Cloud, the next poll picks the values up.
        cloud.schema_missing = False
        cloud.set(config__run__max_steps=12)
        await controller.poll_once()
        assert app.config.run.max_steps == 12
        await app.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("budget", "expected"), [({}, None), ({"max_retry_elapsed_s": 300.0}, 300.0)]
)
def test_managed_attempts_keep_a_retry_budget_only_when_the_code_set_one(budget, expected):
    # Backoff of 5, 10, 20 and 40 s fits the default budget; Cloud's ten
    # attempts make it about 17 minutes.
    code = RetryPolicy(initial_delay_s=5.0, max_delay_s=240.0, **budget)
    assert code.max_retry_elapsed_s == 300.0
    cloud = FakeCloud()
    cloud.values = {"config.run.retry_policy.max_attempts": 10}
    app, _ = _app(cloud, config=CayuConfig(run=RunDefaults(retry_policy=code)))

    # The app applies Cloud's values to its own copy of the config, which must
    # still know which retry fields the code set.
    async def scenario():
        await app.start_managed_settings()
        policy = app.config.run.retry_policy
        assert (policy.max_attempts, policy.initial_delay_s) == (10, 5.0)
        assert policy.max_retry_elapsed_s == expected
        await app.aclose()

    asyncio.run(scenario())


def test_when_cloud_cannot_be_reached_the_last_good_snapshot_stays():
    cloud = FakeCloud()
    cloud.values = {"config.run.max_steps": 30}
    now = [datetime.now(UTC)]
    transport = httpx.MockTransport(cloud)
    controller = ManagedSettings(
        HttpManagedSettingsChannel(
            api_url="https://cloud.example",
            application_id="scout",
            release_id=RELEASE,
            token=TOKEN,
            transport=transport,
            build_transport=transport,
        ),
        clock=lambda: now[0],
    )
    app = CayuApp(
        enable_logging=False,
        managed_settings=controller,
        release_identity=ReleaseIdentity(id=RELEASE),
    )

    async def scenario():
        await app.start_managed_settings()
        assert app.config.run.max_steps == 30
        cloud.unreachable = True
        await controller.poll_once()
        assert controller.status == "unavailable"
        assert app.config.run.max_steps == 30
        # Past valid_until the values still apply; the report says they are stale.
        now[0] += timedelta(minutes=10)
        cloud.unreachable = False
        cloud.valid_for = timedelta(minutes=-20)
        await controller.poll_once()
        assert app.config.run.max_steps == 30
        assert cloud.reports[-1]["state"] == "stale"
        await app.aclose()

    asyncio.run(scenario())


def test_cloud_unreachable_at_build_leaves_the_code_defaults_and_unreported(caplog):
    cloud = FakeCloud()
    cloud.unreachable = True
    with caplog.at_level(logging.WARNING, logger="cayu.managed_settings"):
        app, _ = _app(cloud, config=CayuConfig(run=RunDefaults(max_steps=40)))
    assert app.config.run.max_steps == 40
    assert app._config_sources["run.max_steps"] == "application"
    assert app.managed_settings_controller.state() == "unreported"
    assert "could not be read" in caplog.text and TOKEN not in caplog.text


def test_an_agent_whose_model_the_gateway_policy_manages_ignores_the_managed_model():
    cloud = FakeCloud()
    cloud.values = {"agent.assistant.model": "model-b"}
    app, _ = _app(cloud)
    request = RunRequest(agent_name="assistant", messages=[])
    assert app.resolve_run_model_target(request).model == "model-b"
    app.model_policy = SimpleNamespace(
        controllers=(SimpleNamespace(agent_name="assistant"),), selection=lambda name: None
    )
    assert app.resolve_run_model_target(request).model == "model-a"


def test_the_channel_refuses_another_agents_values_and_never_follows_redirects():
    def other_release(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "contract": "cayu.managed-settings-values/v1",
                "application_id": "scout",
                "release_id": "rel_other",
                "schema_digest": "sha256:" + "0" * 64,
                "revision": 3,
                "valid_until": datetime.now(UTC).isoformat(),
                "values": {},
            },
        )

    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://elsewhere.example/"})

    async def scenario():
        for handler in (other_release, redirect):
            controller = ManagedSettings(
                HttpManagedSettingsChannel(
                    api_url="https://cloud.example",
                    application_id="scout",
                    release_id=RELEASE,
                    token=TOKEN,
                    transport=httpx.MockTransport(handler),
                )
            )
            await controller.poll_once()
            assert controller.status == "unavailable"
            assert controller.snapshot is None
            await controller.close()

    asyncio.run(scenario())


def test_reports_stay_within_the_contract_limits():
    settings = [
        {"key": f"app.setting_{index:03d}_" + "k" * 100, "outcome": "refused", "reason": "r" * 200}
        for index in range(500)
    ]
    fitted = fit_report_settings([*settings, {"key": "app.extra", "outcome": "default"}])
    assert len(fitted) <= 500
    assert len(json.dumps(fitted, separators=(",", ":")).encode()) + 512 <= MAX_REPORT_BYTES
    assert all(item["outcome"] == "refused" for item in fitted)


def test_poll_interval_is_bounded():
    cloud = FakeCloud()
    with pytest.raises(ValueError, match="at most 30"):
        _controller(cloud, poll_interval=60)


class _ChangeSettings(Tool):
    """Changes the settings in Cloud and polls, in the middle of a running session."""

    def __init__(self, cloud: FakeCloud, app_ref: list[CayuApp]) -> None:
        super().__init__(ToolSpec(name="change_settings", description="Change settings."))
        self._cloud, self._app_ref = cloud, app_ref

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        self._cloud.set(agent__assistant__model="model-b", config__run__max_steps=60)
        await self._app_ref[0].managed_settings_controller.poll_once()
        return ToolResult(content="changed")


def test_a_running_session_keeps_its_values_when_cloud_changes_them_mid_run():
    cloud = FakeCloud()
    cloud.values = {"agent.assistant.model": "model-a2", "config.run.max_steps": 40}
    app_ref: list[CayuApp] = []
    calls: list[str] = []

    def respond(request):
        calls.append(request.model)
        if request.tools and not any(m.role.value == "tool" for m in request.messages):
            return [
                ModelStreamEvent.tool_call(id="c1", name="change_settings", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ]
        return [
            ModelStreamEvent.text_delta("ok"),
            ModelStreamEvent.completed({"finish_reason": "stop"}),
        ]

    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
    )
    app_ref.append(app)
    app.register_provider(ScriptedModelProvider(response_factory=respond), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="model-a"),
        settings=[model_setting()],
        tools=[_ChangeSettings(cloud, app_ref)],
    )

    async def scenario():
        await app.start_managed_settings()
        running = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
        )
        assert running.ok, running.error
        # Both model steps of the running session used the model it started with.
        assert calls == ["model-a2", "model-a2"]
        assert app.config.run.max_steps == 60
        report = cloud.reports[-1]
        assert report["revision"] == 2
        assert _outcomes(report)["agent.assistant.model"] == ("adopted", None)
        calls.clear()
        await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
        )
        assert calls[0] == "model-b"
        await app.aclose()

    asyncio.run(scenario())


def test_the_server_and_cayu_worker_start_the_controller_with_their_role():
    TestClient = pytest.importorskip("fastapi.testclient").TestClient

    from cayu.cli.worker import _invoke_worker
    from cayu.server import ServerConfig, create_server

    cloud = FakeCloud()
    app, _ = _app(cloud)
    with TestClient(create_server(app, config=ServerConfig.local_development())):
        assert cloud.reports[-1]["process"]["role"] == "web"
    assert app.managed_settings_controller.status == "stopped"

    worker_cloud = FakeCloud()
    worker_app, _ = _app(worker_cloud)

    async def target(app: CayuApp, stop: asyncio.Event) -> None:
        assert app.managed_settings_controller.status == "ready"

    assert asyncio.run(_invoke_worker(target, worker_app, name="w", shutdown_grace_seconds=5)) == 0
    assert worker_cloud.reports[-1]["process"]["role"] == "worker"
    assert worker_app.managed_settings_controller.status == "stopped"


@pytest.mark.parametrize("value", ["abc", True, "5", 0])
def test_a_malformed_budget_amount_is_refused_without_stopping_the_app(value):
    cloud = FakeCloud()
    cloud.values = {"budget.app.max_estimated_cost": value}
    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
        budget_policy=BudgetPolicy(limits=[BudgetLimit.usd(10, scope="app")]),
    )
    assert app.budget_policy.limits[0].max_estimated_cost == Decimal(10)
    outcome = {item["key"]: item for item in app._managed_outcomes()}
    assert outcome["budget.app.max_estimated_cost"]["outcome"] == "refused"


def test_a_restart_model_the_provider_refuses_is_not_applied():
    cloud = FakeCloud()
    cloud.values = {"agent.assistant.model": "unknown-x"}
    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
    )
    app.register_provider(_Provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="model-a"),
        settings=[
            ManagedSetting(key="model", type="string", max_length=64, label="M", applies="restart")
        ],
    )
    request = RunRequest(agent_name="assistant", messages=[])
    assert app.resolve_run_model_target(request).model == "model-a"
    outcome = {item["key"]: item for item in app._managed_outcomes()}
    assert outcome["agent.assistant.model"]["outcome"] == "refused"


def test_descriptor_defaults_stay_the_codes_when_cloud_overrides_inherited_thinking():
    from cayu.context.thinking import ThinkingConfig

    cloud = FakeCloud()
    cloud.values = {"config.run.thinking.effort": "high"}
    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
        config=CayuConfig(run=RunDefaults(thinking=ThinkingConfig(effort="medium"))),
    )
    app.register_provider(_Provider(), default=True)
    app.register_agent(
        AgentSpec(name="assistant", model="model-a"),
        settings=[thinking_effort_setting(choices=["low", "medium"])],
    )
    defaults = {item.key: item.default for item in app.managed_settings()}
    assert defaults["agent.assistant.thinking.effort"] == "medium"
    assert defaults["config.run.thinking.effort"] == "medium"
    # The running app still uses Cloud's application-wide effort.
    assert app.get_agent("assistant").spec.thinking.effort == "high"


def test_a_new_run_resolves_its_retry_policy_once():
    cloud = FakeCloud()
    cloud.values = {"config.run.retry_policy.max_attempts": 3}
    app, _ = _app(cloud)

    async def scenario():
        request = app._with_application_run_defaults(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "x")])
        )
        assert request.retry_policy is not None and request.retry_policy.max_attempts == 3
        assert "retry_policy" not in request.model_fields_set
        effective = await app.inspect_effective_run_configuration(
            RunRequest(agent_name="assistant", messages=[Message.text("user", "x")])
        )
        assert effective.retry_policy.source == "managed"
        await app.aclose()

    asyncio.run(scenario())


def test_session_started_records_cloud_retry_values_unless_the_request_sets_them():
    cloud = FakeCloud()
    cloud.values = {"config.run.retry_policy.max_attempts": 3}
    app, _ = _app(cloud)

    async def scenario():
        await app.start_managed_settings()
        outcome = await run_to_completion(
            app, RunRequest(agent_name="assistant", messages=[Message.text("user", "go")])
        )
        started = next(e for e in outcome.events if e.type.value == "session.started")
        assert started.payload["managed_settings"]["values"] == {
            "config.run.retry_policy.max_attempts": 3
        }
        from cayu.providers import RetryPolicy

        explicit = await run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "go")],
                retry_policy=RetryPolicy(max_attempts=2),
            ),
        )
        started = next(e for e in explicit.events if e.type.value == "session.started")
        assert started.payload["managed_settings"]["values"] == {}
        # An explicit value equal to Cloud's is still the request's, not Cloud's.
        same = await run_to_completion(
            app,
            RunRequest(
                agent_name="assistant",
                messages=[Message.text("user", "go")],
                retry_policy=RetryPolicy(max_attempts=3),
            ),
        )
        started = next(e for e in same.events if e.type.value == "session.started")
        assert started.payload["managed_settings"]["values"] == {}
        await app.aclose()

    asyncio.run(scenario())


def test_a_restart_model_applies_when_the_provider_registers_after_the_agent():
    cloud = FakeCloud()
    cloud.values = {"agent.assistant.model": "model-b"}
    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
    )
    app.register_agent(
        AgentSpec(name="assistant", model="model-a"),
        settings=[
            ManagedSetting(key="model", type="string", max_length=64, label="M", applies="restart")
        ],
    )
    app.register_provider(_Provider(), default=True)
    request = RunRequest(agent_name="assistant", messages=[])
    assert app.resolve_run_model_target(request).model == "model-b"
    outcome = {item["key"]: item for item in app._managed_outcomes()}
    assert outcome["agent.assistant.model"]["outcome"] == "adopted"
    # A model the provider refuses, once it is registered, leaves the code's model in use.
    cloud.values = {"agent.assistant.model": "unknown-model"}
    late = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
    )
    late.register_agent(
        AgentSpec(name="assistant", model="model-a"),
        settings=[
            ManagedSetting(key="model", type="string", max_length=64, label="M", applies="restart")
        ],
    )
    late.register_provider(_Provider(), default=True)
    assert late.resolve_run_model_target(request).model == "model-a"


@pytest.mark.parametrize("value", ["", "  model-b  "])
def test_a_model_id_the_agent_spec_would_reject_is_refused(value):
    cloud = FakeCloud()
    cloud.values = {"agent.assistant.model": value}
    app = CayuApp(
        enable_logging=False,
        managed_settings=_controller(cloud),
        release_identity=ReleaseIdentity(id=RELEASE),
    )
    app.register_provider(_Provider(), default=True)
    # A pinned provider routes any model, so only the identity check can refuse it.
    app.register_agent(
        AgentSpec(name="assistant", model="model-a", provider_name="scripted"),
        settings=[model_setting()],
    )
    request = RunRequest(agent_name="assistant", messages=[])
    assert app.resolve_run_model_target(request).model == "model-a"
    outcome = {item["key"]: item for item in app._managed_outcomes()}
    assert outcome["agent.assistant.model"]["outcome"] == "refused"
