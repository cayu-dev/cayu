"""Finite private bootstrap ownership; bearer contents never claim authority."""

import asyncio
from datetime import UTC, datetime

import pytest
from tests.core.test_browser_control import identity, operator_purpose

from cayu.runtime._browser_control_authorization import BrowserControlPermissionDenied
from cayu.runtime._browser_control_bootstrap import BrowserGuestBootstrap
from cayu.runtime.browser_control import BrowserControlAllocation


@pytest.mark.parametrize("durable_allocation", [False, True])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
@pytest.mark.parametrize("control_enabled", [False, True])
def test_real_runtime_requires_acknowledged_allocation(
    tmp_path, monkeypatch, durable_allocation, backend, control_enabled
):
    _run_runtime_allocation(tmp_path, monkeypatch, durable_allocation, backend, control_enabled)


@pytest.mark.parametrize("profile_policy", ["disabled", "on_close", "after_terminal_operation"])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_registered_profile_policy_reaches_durable_control_identity(
    tmp_path, monkeypatch, backend, profile_policy
):
    _run_runtime_allocation(tmp_path, monkeypatch, True, backend, True, profile_policy)


def _run_runtime_allocation(
    tmp_path, monkeypatch, durable_allocation, backend, control_enabled, profile_policy=None
):
    from dataclasses import replace

    from tests.core.test_browser_control_authorization import Policy
    from tests.core.test_browser_session import _browser_profile_binding, _FakeBrowserBackend, _tool

    from cayu import (
        AgentSpec,
        CayuApp,
        Environment,
        EnvironmentSpec,
        LocalArtifactStore,
        Message,
        ModelStreamEvent,
        RunRequest,
        ScriptedModelProvider,
        SQLiteSessionStore,
        run_to_completion,
    )
    from cayu.core.tools import ToolContext
    from cayu.runtime._browser_control_service import BrowserControlService
    from cayu.runtime.browser_control_config import BrowserControlConfig
    from cayu.tools.browser_session import BrowserSessionTool, _RunnerBrowserSessionBackend

    owner = BrowserGuestBootstrap()
    captured = []
    bootstrap_calls = []
    fake_backend = _FakeBrowserBackend()
    binding = None
    if profile_policy is not None:
        from cayu.browser_profiles import (
            BrowserProfileCheckpointPolicy,
            BrowserProfileStateV1,
            InMemoryBrowserProfileStore,
        )

        binding = _browser_profile_binding(
            InMemoryBrowserProfileStore(store_id="control-profile-policy"),
            checkpoint_policy=BrowserProfileCheckpointPolicy(profile_policy),
        )

        async def restore_profile(backend, context, **kwargs):
            return None

        async def capture_profile(backend, context, **kwargs):
            return BrowserProfileStateV1()

        monkeypatch.setattr(_RunnerBrowserSessionBackend, "restore_profile", restore_profile)
        monkeypatch.setattr(_RunnerBrowserSessionBackend, "capture_profile", capture_profile)

    async def bootstrap(service, context, *, backend, browser_session_id, arguments, **unused):
        runtime = app._browser_control_runtime
        assert runtime is not None
        assert service is runtime.service
        assert type(backend) is _RunnerBrowserSessionBackend
        assert len(fake_backend.calls) == 1
        from cayu.core.tools import _runtime_tool_invocation_authority
        from cayu.tools.browser_session import _durable_browser_operation_key

        authority = _runtime_tool_invocation_authority(context)
        assert authority is not None
        receipt = await authority.load_durable_operation(
            _durable_browser_operation_key(arguments["operation_id"])
        )
        assert receipt is not None and receipt["state"] == "terminal"
        assert receipt["browser_session_id"] == browser_session_id
        allocation = owner.allocation_for_invocation(
            context,
            purpose=operator_purpose(),
            browser_session_id=browser_session_id,
            arguments=arguments,
        )
        assert allocation.browser_session_id == fake_backend.session_id
        assert allocation.profile_checkpoint_policy == (profile_policy or "unavailable")
        bootstrap_calls.append(allocation)

    monkeypatch.setattr(BrowserControlService, "bootstrap", bootstrap)
    if control_enabled:

        async def preflight(backend, context, request):
            return await fake_backend.preflight(context, request)

        async def execute(backend, context, request):
            response = await fake_backend.execute(context, request)
            if binding is None:
                return response
            assert response.page_set is not None
            protected_pages = response.page_set.model_copy(
                update={
                    "pages": tuple(
                        page.model_copy(update={"title": None, "url": None})
                        for page in response.page_set.pages
                    )
                }
            )
            return replace(response, profile_output_protected=True, page_set=protected_pages)

        monkeypatch.setattr(_RunnerBrowserSessionBackend, "preflight", preflight)
        monkeypatch.setattr(_RunnerBrowserSessionBackend, "execute", execute)
    original_run = BrowserSessionTool.run

    async def run(tool, context, args):
        copied = ToolContext.model_validate(context.model_dump())
        with pytest.raises(BrowserControlPermissionDenied):
            owner.issue_for_invocation(
                copied, purpose=operator_purpose(), browser_session_id="bs_runtime", arguments=args
            )
        with pytest.raises(BrowserControlPermissionDenied):
            owner.issue_for_invocation(
                context,
                purpose=operator_purpose(),
                browser_session_id="bs_runtime",
                arguments={**args, "operation": "close"},
            )
        result = await original_run(tool, context, args)
        assert not result.is_error
        if durable_allocation:
            token = owner.issue_for_invocation(
                context, purpose=operator_purpose(), browser_session_id="bs_runtime", arguments=args
            )
            resolved = owner.consume(token)
            assert resolved.session_id == context.session_id
            assert resolved.environment_name == "browser"
            assert resolved.browser_session_id == "bs_runtime"
            from tests.core.test_browser_control_authorization import Policy

            from cayu.runtime._browser_control_coordinator import BrowserControlCoordinator
            from cayu.vaults.redaction import SecretRedactor

            coordinator = BrowserControlCoordinator(
                purpose=operator_purpose(),
                store=app.session_store,
                policy=Policy(True),
                redactor=SecretRedactor(),
                clock=lambda: datetime.now(UTC),
            )
            record = await coordinator.bind_guest(
                allocation=resolved, worker_instance_id="vw_" + "a" * 32
            )
            assert record.identity.allocation_fingerprint == resolved.allocation_fingerprint
            assert record.identity.interaction_id == resolved.interaction_id
            assert record.identity.profile_checkpoint_policy == (profile_policy or "unavailable")
            # Reconstruct from the supported store, not only the publication return.
            _, reloaded = await coordinator._load(record.identity)
            assert reloaded == record
            assert await coordinator.drain()
        else:
            with pytest.raises(BrowserControlPermissionDenied):
                owner.issue_for_invocation(
                    context,
                    purpose=operator_purpose(),
                    browser_session_id="bs_runtime",
                    arguments=args,
                )
        captured.append(True)
        return result

    monkeypatch.setattr(BrowserSessionTool, "run", run)
    store = SQLiteSessionStore(tmp_path / "sessions.sqlite") if backend == "sqlite" else None
    app = CayuApp(
        enable_logging=False,
        session_store=store,
        browser_control=(
            BrowserControlConfig(
                purpose=operator_purpose(),
                policy=Policy(True),
                guest_endpoint="wss://control.test/guest",
            )
            if control_enabled
            else None
        ),
    )
    if durable_allocation:
        from tests.core.test_environment_allocation_recovery import (
            _FakeRemoteFactory,
            _FakeRemoteProvider,
        )

        app.register_environment_factory(
            EnvironmentSpec(name="browser"), _FakeRemoteFactory(_FakeRemoteProvider()), default=True
        )
    else:
        app.register_environment(
            Environment(
                EnvironmentSpec(name="browser"), artifact_store=LocalArtifactStore(tmp_path)
            ),
            default=True,
        )
    app.register_provider(
        ScriptedModelProvider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="call",
                        name="browser_session",
                        arguments={
                            "operation": "navigate",
                            "url": "https://example.test",
                            "operation_id": "navigate-1",
                        },
                    ),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ],
                [
                    ModelStreamEvent.text_delta("done"),
                    ModelStreamEvent.completed({"finish_reason": "stop"}),
                ],
            ]
        ),
        default=True,
    )
    app.register_agent(
        AgentSpec(name="agent", model="model"),
        tools=[
            BrowserSessionTool(
                browser_profile=binding, max_sessions=1, idle_timeout_seconds=60, max_wait_ms=1_000
            )
            if control_enabled
            else _tool(fake_backend)
        ],
    )

    async def scenario():
        try:
            if binding is not None:
                await binding.initialize()
            return await run_to_completion(
                app,
                RunRequest(agent_name="agent", messages=[Message.text("user", "Open the page")]),
            )
        finally:
            if store is not None:
                await store.close()

    outcome = asyncio.run(scenario())
    assert outcome.ok
    assert captured == [True], "\n".join(
        str(event.payload) for event in outcome.events if "result" in event.payload
    )
    assert len(bootstrap_calls) == int(control_enabled and durable_allocation)


def allocation():
    return BrowserControlAllocation.model_validate(
        identity().model_dump(exclude={"worker_instance_id"})
    )


def test_bootstrap_is_single_use_detached_and_restart_invalidates_it():
    owner = BrowserGuestBootstrap()
    original = allocation()
    token = owner.issue(original)
    object.__setattr__(original, "browser_session_id", "mutated")
    with pytest.raises(BrowserControlPermissionDenied):
        BrowserGuestBootstrap().consume(token)
    assert owner.consume(token) == allocation()
    with pytest.raises(BrowserControlPermissionDenied):
        owner.consume(token)


def test_bootstrap_expiry_capacity_revocation_and_shutdown():
    clock = [1.0]
    owner = BrowserGuestBootstrap(clock=lambda: clock[0])
    tokens = [owner.issue(allocation()) for _ in range(32)]
    with pytest.raises(BrowserControlPermissionDenied):
        owner.issue(allocation())
    owner.revoke(tokens[0])
    with pytest.raises(BrowserControlPermissionDenied):
        owner.consume(tokens[0])
    owner.issue(allocation())
    clock[0] = 61.0
    for token in tokens:
        with pytest.raises(BrowserControlPermissionDenied):
            owner.consume(token)
    new_token = owner.issue(allocation())
    owner.close()
    with pytest.raises(BrowserControlPermissionDenied):
        owner.consume(new_token)
    with pytest.raises(BrowserControlPermissionDenied):
        owner.issue(allocation())
