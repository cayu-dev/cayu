"""Real app allocation/bootstrap into native Chrome using a local test runner."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretBytes
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy
from tests.core.test_browser_control_transport import control_tls as _control_tls
from tests.core.test_browser_session import _WireRunner
from tests.core.test_environment_allocation_recovery import _FakeRemoteFactory, _FakeRemoteProvider
from tests.server._app_native_allocation_loss import lose_browser_allocation
from tests.server._app_native_operator import handback_app_browser
from tests.server._app_native_takeover_race import NativeTakeoverRace
from tests.server._app_native_terminal_disconnect import NativeTerminalDisconnect
from tests.server._browser_control_tls_server import browser_control_tls_server

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentSpec,
    InMemorySessionStore,
    Message,
    ModelStreamEvent,
    RunRequest,
    ScriptedModelProvider,
    SQLiteSessionStore,
    run_to_completion,
)
from cayu.browser_profiles import (
    AESGCMBrowserProfileKeyAuthority,
    BrowserProfileBinding,
    BrowserProfileCheckpointPolicy,
    BrowserProfileDestinationPolicy,
    BrowserProfileScope,
    SQLiteBrowserProfileStore,
)
from cayu.runners import ExecResult, Runner
from cayu.runtime._browser_control_checkpoint import browser_control_checkpoint_read_scope
from cayu.runtime.browser_control import BrowserControlCheckpoint
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.runtime.checkpoints import BROWSER_CONTROLS_CHECKPOINT_KEY
from cayu.server import (
    BasicAuth,
    BrowserControlServerConfig,
    DashboardConfig,
    ServerConfig,
    create_server,
)
from cayu.tools import _browser_guest
from cayu.tools._browser_control_transport import open_guest_control_channel
from cayu.tools._redaction import active_secret_redactor
from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend
from cayu.vaults import SecretRef, StaticVault

control_tls = _control_tls
pytestmark = pytest.mark.skipif(
    os.environ.get("CAYU_BROWSER_CONTROL_LIVE") != "1", reason="Opt-in local Chromium acceptance."
)


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("viewer_only", [False, True])
def test_app_allocates_and_bootstraps_native_browser(
    tmp_path,
    monkeypatch,
    control_tls,
    persistent,
    caplog,
    capfd,
    recwarn,
    viewer_only,
    rendered=False,
    dynamic_secret=False,
    profile_consent=None,
    takeover_race=False,
    allocation_loss=False,
    racing_loss_poll=False,
    keyboard_only=False,
    terminal_disconnect=None,
):
    from playwright.async_api import async_playwright

    async def scenario():
        rendering_failures = []
        store = (
            SQLiteSessionStore(tmp_path / "app.sqlite") if persistent else InMemorySessionStore()
        )
        daemons = {}
        private_deliveries = []
        private_text = "app-native-private-input-canary"
        dynamic_canary = "invocation-only-origin-canary"
        guest_api = FastAPI()
        _, client_tls = control_tls
        race = (
            NativeTakeoverRace(tls=client_tls, private_text=private_text) if takeover_race else None
        )
        terminal = NativeTerminalDisconnect(terminal_disconnect) if terminal_disconnect else None
        profile_path = tmp_path / "profiles.sqlite"
        profile_store = None
        profile_binding = None
        profile_operations = []
        if profile_consent is not None:
            profile_store = SQLiteBrowserProfileStore(profile_path, store_id="operator-profiles")
            profile_binding = BrowserProfileBinding.build(
                scope=BrowserProfileScope.build(
                    application_id="operator-test", tenant_id="tenant", sharing_scope="agent"
                ),
                destination_policy=BrowserProfileDestinationPolicy.build(
                    ("https://app-native.test",)
                ),
                browser_protocol="cayu.browser-session.v4",
                browser_worker_version="12",
                store=profile_store,
                key_authority=AESGCMBrowserProfileKeyAuthority(
                    authority_id="operator-test-key", key=b"p" * 32
                ),
                profile_id="bprof_operator_test",
                checkpoint_policy=BrowserProfileCheckpointPolicy.ON_CLOSE,
                lease_seconds=120,
            )
            await profile_binding.initialize()

        async def connect(*, endpoint, credential):
            return await open_guest_control_channel(
                endpoint=endpoint, credential=credential, tls=client_tls
            )

        monkeypatch.setattr(_browser_guest, "open_guest_control_channel", connect)

        async def network_app(scope, receive, send):
            await guest_api(scope, receive, send)

        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(
                executable_path=os.environ.get("CAYU_BROWSER_CONTROL_CHROMIUM"), headless=True
            )

            class NativeRunner(_WireRunner, Runner):
                preflight_exec = Runner.preflight_exec

                async def _exec_private_browser_profile(self, command, **kwargs):
                    return await self.exec(command, **kwargs)

                async def exec(self, command, **kwargs):
                    raw = json.loads(kwargs["stdin"])
                    if raw.get("protocol_version") == "cayu.browser-control-bootstrap.v1":
                        private_deliveries.append(1)
                        daemon = daemons[raw.pop("session_id")]
                        raw.pop("protocol_version")
                        result = await daemon.bootstrap_operator_channel(raw)
                    else:
                        request = _browser_guest._interactive_request_from_json(raw)
                        if request.operation in {"profile_restore", "profile_checkpoint"}:
                            profile_operations.append(request.operation)
                        daemon = daemons.get(request.session_id)
                        if daemon is None:
                            daemon = _browser_guest._InteractiveDaemon(request.session_id)
                            daemon.browser = browser
                            if race is not None:
                                race.instrument(daemon, monkeypatch)
                            if request.operation == "profile_restore":
                                result = await daemon.execute(request)
                            else:
                                await daemon._ensure_context(None)
                                await daemon._ensure_configuration(request)
                            await daemon.context.route(
                                "https://app-native.test/**",
                                lambda route: route.fulfill(
                                    body=(
                                        "<label>Password <input type='password' "
                                        "oninput=\"localStorage.setItem('session',this.value)\">"
                                        "</label>"
                                        if profile_consent is not None
                                        else "<label>Password <input type='password'></label>"
                                    ),
                                    content_type="text/html",
                                ),
                            )
                            await daemon.context.route(
                                "https://login-native.test/**",
                                lambda route: route.fulfill(
                                    body="<label>Password <input type='password'></label>",
                                    content_type="text/html",
                                ),
                            )
                            await daemon.context.route(
                                f"https://{dynamic_canary}.test/**",
                                lambda route: route.fulfill(
                                    body="<label>Password <input type='password'></label>",
                                    content_type="text/html",
                                ),
                            )
                            daemons[request.session_id] = daemon
                            if request.operation == "profile_restore":
                                return ExecResult(stdout=json.dumps(result))
                        result = await daemon.execute(request)
                        if terminal is not None and terminal.identity is not None:
                            await terminal.after_native(request, result)
                        if race is not None and request.operation_id == "fresh":
                            await race.before_fresh_publication()
                    return ExecResult(stdout=json.dumps(result))

            runner = NativeRunner()

            class Factory(_FakeRemoteFactory):
                def _result(self, request, resource, *, allocation=None):
                    result = super()._result(request, resource, allocation=allocation)
                    return replace(
                        result,
                        environment=Environment(
                            result.environment.spec,
                            runner=runner,
                            vault=StaticVault({"dynamic": dynamic_canary})
                            if dynamic_secret
                            else None,
                        ),
                    )

            try:
                async with browser_control_tls_server(network_app, tmp_path) as port:
                    endpoint = f"wss://127.0.0.1:{port}/api/browser-control/guest"
                    app = CayuApp(
                        enable_logging=False,
                        session_store=store,
                        browser_control=BrowserControlConfig(
                            purpose=operator_purpose(),
                            policy=race or Policy(True),
                            guest_endpoint=endpoint,
                        ),
                    )
                    runtime = app._browser_control_runtime
                    assert runtime is not None
                    retirement_read = runtime.coordinator.bootstrap_retirement_record
                    retirement_reads = []
                    retirement_failed = asyncio.Event()

                    async def fail_once_retirement(identity):
                        retirement_reads.append(identity)
                        if (
                            len(retirement_reads) == 1
                            and profile_consent is None
                            and terminal is None
                        ):
                            retirement_failed.set()
                            raise OSError("retirement read temporarily unavailable")
                        return await retirement_read(identity)

                    monkeypatch.setattr(
                        runtime.coordinator, "bootstrap_retirement_record", fail_once_retirement
                    )
                    guest_api = create_server(
                        app,
                        config=ServerConfig.protected(
                            race.authenticate
                            if race is not None
                            else BasicAuth(username="operator", password="password"),
                            dashboard=DashboardConfig(
                                path="/operator",
                                directory=(
                                    Path(os.environ["CAYU_BROWSER_DASHBOARD_BUILD"])
                                    if os.environ.get("CAYU_BROWSER_DASHBOARD_BUILD")
                                    else None
                                ),
                            ),
                            browser_control=BrowserControlServerConfig(
                                operator_origin="https://operator.test",
                                signing_key=SecretBytes(b"k" * 32),
                            ),
                        ),
                    )
                    verified = []

                    if dynamic_secret:
                        original_preflight = _RunnerBrowserSessionBackend.preflight

                        async def resolving_preflight(self, ctx, request):
                            assert ctx.vault is not None
                            resolved = await ctx.vault.resolve(SecretRef(name="dynamic"))
                            assert resolved.value.get_secret_value() == dynamic_canary
                            assert (
                                active_secret_redactor(ctx).redact_text(dynamic_canary)
                                != dynamic_canary
                            )
                            assert (
                                app._secret_redactor.redact_text(dynamic_canary) == dynamic_canary
                            )
                            return await original_preflight(self, ctx, request)

                        monkeypatch.setattr(
                            _RunnerBrowserSessionBackend, "preflight", resolving_preflight
                        )

                    class Provider(ScriptedModelProvider):
                        count = 0

                        async def stream(self, request):
                            assert private_text not in repr(request)
                            if dynamic_secret:
                                assert dynamic_canary not in repr(request)
                            self.count += 1
                            if terminal is not None and self.count >= 3:
                                await terminal.verify()
                                assert terminal.identity is not None
                                if self.count == 3:
                                    yield ModelStreamEvent.tool_call(
                                        id="after-disconnect",
                                        name="browser_session",
                                        arguments={
                                            "operation": "observe",
                                            "operation_id": "after-disconnect",
                                            "session_id": terminal.identity.browser_session_id,
                                            "page_id": terminal.daemon.active_page_id
                                            or "closed-page",
                                        },
                                    )
                                    yield ModelStreamEvent.completed(
                                        {"finish_reason": "tool_calls"}
                                    )
                                else:
                                    assert self.count == 4
                                    yield ModelStreamEvent.text_delta("disconnect fence verified")
                                    yield ModelStreamEvent.completed()
                                return
                            if race is not None and self.count > 1:
                                owner = next(iter(runtime.service._owners.values()))
                                bound = owner.bound.result()
                                identity = bound.record.identity
                                if self.count == 2:
                                    verified.append(identity)
                                    await race.prepare(
                                        endpoint=endpoint,
                                        session_id=identity.session_id,
                                        daemon=daemons[identity.browser_session_id],
                                        store=store,
                                        commands=owner.commands,
                                    )
                                    arguments = {
                                        "operation": "list_pages",
                                        "operation_id": "held-model",
                                        "session_id": identity.browser_session_id,
                                    }
                                elif self.count == 3:
                                    assert race.task is not None
                                    await race.task
                                    arguments = {
                                        "operation": "observe",
                                        "operation_id": "blocked-model",
                                        "session_id": identity.browser_session_id,
                                        "page_id": race.pages[0]["page_id"],
                                    }
                                elif self.count == 4:
                                    if allocation_loss:
                                        await lose_browser_allocation(
                                            race,
                                            browser,
                                            runtime.coordinator,
                                            identity,
                                            racing_poll=racing_loss_poll,
                                        )
                                    else:
                                        await race.finish()
                                    arguments = {
                                        "operation": "observe",
                                        "operation_id": "after-loss"
                                        if allocation_loss
                                        else "fresh",
                                        "session_id": identity.browser_session_id,
                                        "page_id": race.pages[0]["page_id"],
                                    }
                                else:
                                    assert self.count == 5
                                    _, current = await runtime.coordinator._load(identity)
                                    if allocation_loss:
                                        assert current.state == "control_uncertain"
                                        assert current.settled_input_sequence == 0
                                        assert race.dispatched == ["open", "held-model"]
                                    else:
                                        assert not current.fresh_observation_required
                                        assert current.settled_input_sequence == 2
                                        assert race.dispatched == ["open", "held-model", "fresh"]
                                    yield ModelStreamEvent.text_delta(
                                        "allocation loss fenced"
                                        if allocation_loss
                                        else "exclusive handback verified"
                                    )
                                    yield ModelStreamEvent.completed()
                                    return
                                yield ModelStreamEvent.tool_call(
                                    id=f"race-{self.count}",
                                    name="browser_session",
                                    arguments=arguments,
                                )
                                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                                return
                            if self.count == 4 and profile_consent is not None:
                                assert profile_store is not None and profile_binding is not None
                                inspection = await profile_store.inspect_profile(
                                    profile_binding.access
                                )
                                assert inspection.generation == int(profile_consent == "allow")
                                assert inspection.storage_entry_count == int(
                                    profile_consent == "allow"
                                )
                                assert not inspection.active_writer
                                assert private_text not in repr(inspection)
                                yield ModelStreamEvent.text_delta("checkpoint consent honored")
                                yield ModelStreamEvent.completed()
                                return
                            if self.count == 1:
                                yield ModelStreamEvent.tool_call(
                                    id="open",
                                    name="browser_session",
                                    arguments={
                                        "operation": "navigate",
                                        "url": "https://app-native.test/",
                                        "operation_id": "open",
                                    },
                                )
                                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                                return
                            assert private_deliveries == [1]
                            assert len(runtime.service._owners) == 1
                            owner = next(iter(runtime.service._owners.values()))
                            assert owner.commands is not None
                            bound = owner.bound.result()
                            with browser_control_checkpoint_read_scope(
                                bound.record.identity.session_id
                            ):
                                checkpoint = await store.load_checkpoint(
                                    bound.record.identity.session_id
                                )
                            assert checkpoint is not None
                            current = BrowserControlCheckpoint.model_validate(
                                checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY]
                            ).records[0]
                            assert current.identity.browser_session_id in daemons
                            assert current.state == "agent_controlled"
                            if self.count == 2:
                                assert current == bound.record
                                verified.append(current.identity)
                                if dynamic_secret:
                                    live_page = next(
                                        iter(
                                            daemons[
                                                current.identity.browser_session_id
                                            ].pages.values()
                                        )
                                    )
                                    await live_page.page.goto(f"https://{dynamic_canary}.test/")
                                if rendered:
                                    from tests.server._browser_operator_rendering import (
                                        rendered_native_operator_journey,
                                    )

                                    async with httpx.AsyncClient(
                                        base_url=f"https://127.0.0.1:{port}",
                                        verify=client_tls,
                                        trust_env=False,
                                    ) as client:
                                        _, takeover = await rendered_native_operator_journey(
                                            browser,
                                            client,
                                            guest_api,
                                            owner.commands,
                                            daemons[current.identity.browser_session_id],
                                            mounted_api=True,
                                            keyboard_only=keyboard_only,
                                            failures=rendering_failures,
                                            clock_ms=None,
                                            session_id=current.identity.session_id,
                                            private_text=private_text,
                                            network_socket_origin=f"wss://127.0.0.1:{port}",
                                            network_tls=client_tls,
                                            expected_page_origin="https://app-native.test",
                                            changed_page_origin="https://login-native.test",
                                            dashboard_path=(
                                                "/operator"
                                                if os.environ.get("CAYU_BROWSER_DASHBOARD_BUILD")
                                                else None
                                            ),
                                        )
                                    page_id = takeover.pages[0].page_id
                                else:
                                    page_id = await handback_app_browser(
                                        session_id=current.identity.session_id,
                                        input_endpoint=endpoint.removesuffix("/guest") + "/input",
                                        tls=client_tls,
                                        private_text=private_text,
                                        viewer_only=viewer_only,
                                        checkpoint_consent=profile_consent or "deny",
                                    )
                                if viewer_only:
                                    assert runtime.service._viewers
                                    yield ModelStreamEvent.text_delta("viewed")
                                    yield ModelStreamEvent.completed()
                                    return
                                if dynamic_secret:
                                    yield ModelStreamEvent.text_delta("handed back")
                                    yield ModelStreamEvent.completed()
                                    return
                                page = daemons[current.identity.browser_session_id].pages[page_id]
                                assert (
                                    await page.page.locator("input").input_value() == private_text
                                )
                                if profile_consent is not None:
                                    assert (
                                        await page.page.evaluate("localStorage.getItem('session')")
                                        == private_text
                                    )
                                if terminal is not None:
                                    await terminal.arm(
                                        runtime=runtime,
                                        store=store,
                                        daemon=daemons[current.identity.browser_session_id],
                                        monkeypatch=monkeypatch,
                                    )
                                yield ModelStreamEvent.tool_call(
                                    id="fresh",
                                    name="browser_session",
                                    arguments={
                                        "operation": terminal.operation
                                        if terminal is not None
                                        else "observe",
                                        "operation_id": "fresh",
                                        "session_id": current.identity.browser_session_id,
                                        **(
                                            {"page_id": page_id}
                                            if terminal is None or terminal.operation == "observe"
                                            else {}
                                        ),
                                    },
                                )
                                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                                return
                            assert self.count == 3
                            assert not current.fresh_observation_required
                            assert current.checkpoint_consent == (profile_consent or "deny")
                            assert current.settled_input_sequence == 2
                            if profile_consent is not None:
                                yield ModelStreamEvent.tool_call(
                                    id="close",
                                    name="browser_session",
                                    arguments={
                                        "operation": "close",
                                        "operation_id": "close",
                                        "session_id": current.identity.browser_session_id,
                                    },
                                )
                                yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
                                return
                            yield ModelStreamEvent.text_delta("ready")
                            yield ModelStreamEvent.completed()

                    app.register_provider(Provider([]), default=True)
                    app.register_environment_factory(
                        EnvironmentSpec(name="browser"),
                        Factory(_FakeRemoteProvider()),
                        default=True,
                    )
                    app.register_agent(
                        AgentSpec(name="agent", model="model"),
                        tools=[
                            BrowserSessionTool()
                            if profile_binding is None
                            else BrowserSessionTool(
                                browser_profile=profile_binding,
                                max_sessions=1,
                                idle_timeout_seconds=60,
                                max_wait_ms=1000,
                            )
                        ],
                    )
                    try:
                        async with asyncio.timeout(60 if keyboard_only else 30):
                            outcome = await run_to_completion(
                                app,
                                RunRequest(
                                    agent_name="agent", messages=[Message.text("user", "open")]
                                ),
                            )
                        assert not rendering_failures, rendering_failures
                        assert outcome.ok, outcome.error
                        assert len(verified) == 1
                        with browser_control_checkpoint_read_scope(outcome.session_id):
                            checkpoint = await store.load_checkpoint(outcome.session_id)
                        events = await store.load_events(outcome.session_id)
                        assert checkpoint is not None
                        if not viewer_only:
                            terminal_control = BrowserControlCheckpoint.model_validate(
                                checkpoint[BROWSER_CONTROLS_CHECKPOINT_KEY]
                            ).records[0]
                            assert terminal_control.acquisition_audit is not None
                            if allocation_loss:
                                assert terminal_control.handback_audit is None
                            else:
                                assert terminal_control.handback_audit is not None
                            assert terminal_control.request is not None
                            for audit in (
                                terminal_control.acquisition_audit,
                                terminal_control.handback_audit,
                            ):
                                if audit is None:
                                    assert allocation_loss
                                    continue
                                assert audit.request_id == terminal_control.request.request_id
                                assert tuple(item.origin for item in audit.locations) == (
                                    None
                                    if dynamic_secret
                                    else "https://login-native.test"
                                    if rendered
                                    else "https://app-native.test",
                                )
                        assert private_text not in json.dumps(checkpoint)
                        if profile_store is not None:
                            assert profile_operations.count("profile_checkpoint") == int(
                                profile_consent == "allow"
                            )
                            assert private_text.encode() not in profile_path.read_bytes()
                        if dynamic_secret:
                            assert dynamic_canary not in json.dumps(checkpoint)
                        assert private_text not in json.dumps(
                            [event.model_dump(mode="json") for event in events]
                        )
                        if dynamic_secret:
                            assert dynamic_canary not in json.dumps(
                                [event.model_dump(mode="json") for event in events]
                            )
                        captured = capfd.readouterr()
                        assert private_text not in captured.out + captured.err + caplog.text
                        assert all(private_text not in str(item.message) for item in recwarn)
                        if dynamic_secret:
                            assert dynamic_canary not in captured.out + captured.err + caplog.text
                            assert all(dynamic_canary not in str(item.message) for item in recwarn)
                    finally:
                        if terminal is not None:
                            terminal.release_cleanup()
                        if race is not None:
                            await race.close()
                        for daemon in daemons.values():
                            assert await daemon.close(), (
                                rendering_failures,
                                {
                                    name: (task.done(), task.cancelled())
                                    for name, task in daemon.session_cleanup_tasks.items()
                                },
                            )
                        assert await runtime.service.channels.drain()
                        if verified and profile_consent is None and terminal is None:
                            await asyncio.wait_for(retirement_failed.wait(), 5)
                            assert runtime.service._owners
                            assert await runtime.service.settle_bootstrap_retirements()
                            assert not runtime.service._owners
                            assert not runtime.service._viewers
                            assert len(retirement_reads) == 2
                        elif verified:
                            assert await runtime.service.settle_bootstrap_retirements()
                            assert not runtime.service._owners
                            assert not runtime.service._viewers
            finally:
                await browser.close()
                if isinstance(store, SQLiteSessionStore):
                    await store.close()
                if profile_store is not None:
                    await profile_store.close()
                    for candidate in (profile_path, Path(str(profile_path) + "-wal")):
                        if candidate.exists():
                            assert private_text.encode() not in candidate.read_bytes()

    asyncio.run(scenario())


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize(
    "phase", ["observe_pending", "observe_committed", "close_pending", "close_committed"]
)
def test_app_native_terminal_publication_disconnect(
    tmp_path, monkeypatch, control_tls, persistent, caplog, capfd, recwarn, phase
):
    test_app_allocates_and_bootstraps_native_browser(
        tmp_path,
        monkeypatch,
        control_tls,
        persistent,
        caplog,
        capfd,
        recwarn,
        viewer_only=False,
        terminal_disconnect=phase,
    )


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("racing_poll", [False, True])
def test_app_native_browser_process_loss_fences_operator_and_model(
    tmp_path, monkeypatch, control_tls, persistent, caplog, capfd, recwarn, racing_poll
):
    test_app_allocates_and_bootstraps_native_browser(
        tmp_path,
        monkeypatch,
        control_tls,
        persistent,
        caplog,
        capfd,
        recwarn,
        viewer_only=False,
        takeover_race=True,
        allocation_loss=True,
        racing_loss_poll=racing_poll,
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_app_native_competing_takeovers_wait_for_dispatched_model(
    tmp_path, monkeypatch, control_tls, persistent, caplog, capfd, recwarn
):
    test_app_allocates_and_bootstraps_native_browser(
        tmp_path,
        monkeypatch,
        control_tls,
        persistent,
        caplog,
        capfd,
        recwarn,
        viewer_only=False,
        takeover_race=True,
    )


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("consent", ["allow", "deny"])
def test_app_native_handback_honors_profile_checkpoint_consent(
    tmp_path, monkeypatch, control_tls, persistent, consent, caplog, capfd, recwarn
):
    test_app_allocates_and_bootstraps_native_browser(
        tmp_path,
        monkeypatch,
        control_tls,
        persistent,
        caplog,
        capfd,
        recwarn,
        viewer_only=False,
        profile_consent=consent,
    )


@pytest.mark.parametrize("persistent", [False, True])
def test_invocation_only_secret_is_not_published_in_browser_audit(
    tmp_path, monkeypatch, control_tls, persistent, caplog, capfd, recwarn
):
    test_app_allocates_and_bootstraps_native_browser(
        tmp_path,
        monkeypatch,
        control_tls,
        persistent,
        caplog,
        capfd,
        recwarn,
        viewer_only=False,
        dynamic_secret=True,
    )


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.skipif(
    not (
        os.environ.get("CAYU_BROWSER_DASHBOARD_URL")
        or os.environ.get("CAYU_BROWSER_DASHBOARD_BUILD")
    ),
    reason="Requires local Vite or a compiled dashboard directory.",
)
@pytest.mark.parametrize("keyboard_only", [False, True])
def test_rendered_operator_controls_app_created_native_browser(
    tmp_path, monkeypatch, control_tls, persistent, caplog, capfd, recwarn, keyboard_only
):
    test_app_allocates_and_bootstraps_native_browser(
        tmp_path,
        monkeypatch,
        control_tls,
        persistent,
        caplog,
        capfd,
        recwarn,
        viewer_only=False,
        rendered=True,
        keyboard_only=keyboard_only,
    )
