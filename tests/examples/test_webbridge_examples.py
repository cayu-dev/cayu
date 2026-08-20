from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from examples.webbridge.daily_check import (
    _settle_daily_task_from_terminal_session,
    _settle_ownerless_terminal_daily_checks,
    daily_check_worker,
    external_cron_tick,
    handle_daily_check,
    load_durable_daily_result,
    register_daily_checker,
)
from examples.webbridge.research import browse_extract_verify
from tests.core._execution_profile_fixtures import create_admitted_session

from cayu import (
    CayuApp,
    CredentialProxy,
    Environment,
    EnvironmentSpec,
    EventQuery,
    EventType,
    ExecutionProfileBehaviorIdentity,
    IncompleteSessionRecoveryRequest,
    InMemoryTaskStore,
    Message,
    ProxyAuthorizationResult,
    RunRequest,
    SecretRef,
    Session,
    SessionInvocationBinding,
    SessionStatus,
    SQLiteSessionStore,
    SQLiteTaskStore,
    Task,
    TaskQuery,
    TaskStatus,
    ToolContext,
    ToolResult,
    WebBridge,
    WebBridgeCredentialAuthority,
    session_invocation_from_task,
)
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime.execution_profiles import (
    active_invocation_execution_profile_from_checkpoint,
    active_invocation_execution_profile_is_released,
)
from cayu.tools import WebFetchAdapterRequest, WebSearchAdapterRequest
from cayu.vaults import ResolvedSecret


class _ConcurrentSettlementSQLiteTaskStore(SQLiteTaskStore):
    def __init__(self, path: Path, *, terminal_status: TaskStatus) -> None:
        super().__init__(path)
        self._terminal_status = terminal_status
        self._terminal_calls = 0
        self._terminal_calls_ready = asyncio.Event()

    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any],
        *,
        worker_id: str | None = None,
    ) -> Task:
        if self._terminal_status is TaskStatus.COMPLETED:
            await self._await_concurrent_terminal_calls()
        return await super().complete_task(task_id, result, worker_id=worker_id)

    async def fail_task(
        self,
        task_id: str,
        error: dict[str, Any],
        *,
        worker_id: str | None = None,
    ) -> Task:
        if self._terminal_status is TaskStatus.FAILED:
            await self._await_concurrent_terminal_calls()
        return await super().fail_task(task_id, error, worker_id=worker_id)

    async def _await_concurrent_terminal_calls(self) -> None:
        self._terminal_calls += 1
        if self._terminal_calls == 2:
            self._terminal_calls_ready.set()
        await self._terminal_calls_ready.wait()


class _ExampleCredentialProxy(CredentialProxy):
    def supports_webbridge_credential_authority(
        self,
        authority: WebBridgeCredentialAuthority,
    ) -> bool:
        return authority == WebBridgeCredentialAuthority(
            provider="example",
            origin="https://provider.example",
            secret_refs=(SecretRef(name="example_key"),),
        )

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, object] | None = None,
    ) -> ResolvedSecret:
        del ref, scope
        raise AssertionError("The fake hosted adapter must not resolve credentials.")

    async def authorize_request(
        self,
        *,
        destination: str,
        credential: SecretRef | None = None,
        action: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ProxyAuthorizationResult:
        del destination, credential, action, metadata
        raise AssertionError("The fake hosted adapter must not authorize requests.")


_HOSTED_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="example-hosted-web",
    behavior_version="1",
    implementation_version="test",
)
_HOSTED_ENVIRONMENT_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="example-hosted-environment",
    behavior_version="1",
    implementation_version="test",
)


class _ResearchAdapter:
    def __init__(self, results: list[object] | None = None) -> None:
        self.results = results
        self.fetches = 0

    def webbridge_credential_authority(self) -> WebBridgeCredentialAuthority:
        return WebBridgeCredentialAuthority(
            provider="example",
            origin="https://provider.example",
            secret_refs=(SecretRef(name="example_key"),),
        )

    async def search(
        self,
        ctx: ToolContext,
        request: WebSearchAdapterRequest,
    ) -> ToolResult:
        del ctx, request
        return ToolResult(
            content="untrusted search",
            structured={
                "results": self.results
                or [
                    {"rank": 1, "url": "https://one.example/", "title": "One"},
                    {"rank": 2, "url": "https://two.example/", "title": "Two"},
                ],
            },
        )

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
        del ctx
        self.fetches += 1
        if request.requested_url == "https://two.example/":
            return ToolResult(
                content="denied",
                structured={"error": "destination_denied"},
                is_error=True,
            )
        return ToolResult(
            content="<untrusted_web_content>page</untrusted_web_content>",
            structured={
                "requested_url": request.requested_url,
                "final_url": "https://one.example/reference",
                "title": "One",
                "content": "page",
                "truncated": False,
            },
        )


class _DailyProvider(ModelProvider):
    name = "daily-fake"

    def __init__(self) -> None:
        self.requests = 0

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return ExecutionProfileBehaviorIdentity(
            name="tests:webbridge_daily_provider",
            behavior_version="1",
            implementation_version="1",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        del request
        self.requests += 1
        if self.requests == 1:
            yield ModelStreamEvent.tool_call(
                id="daily-fetch",
                name="web_fetch",
                arguments={"url": "https://status.example/"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("No material change; canonical source retained.")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _MalformedFetchAdapter(_ResearchAdapter):
    def __init__(self, result: ToolResult) -> None:
        super().__init__(results=[{"rank": 1, "url": "https://one.example/"}])
        self.result = result

    async def fetch(
        self,
        ctx: ToolContext,
        request: WebFetchAdapterRequest,
    ) -> ToolResult:
        del ctx, request
        return self.result


def test_research_recipe_retains_sources_and_isolates_page_failure() -> None:
    evidence = asyncio.run(
        browse_extract_verify(
            WebBridge.hosted(adapter=_ResearchAdapter()),
            ToolContext(session_id="research", proxy=_ExampleCredentialProxy()),
            "bounded runtimes",
        )
    )

    assert [(page.rank, page.source_url, page.final_url) for page in evidence.pages] == [
        (1, "https://one.example/", "https://one.example/reference")
    ]
    assert [(failure.rank, failure.source_url, failure.error) for failure in evidence.failures] == [
        (2, "https://two.example/", "destination_denied")
    ]


def test_research_recipe_isolates_malformed_result_and_keeps_valid_page() -> None:
    evidence = asyncio.run(
        browse_extract_verify(
            WebBridge.hosted(
                adapter=_ResearchAdapter(
                    results=[
                        "malformed",
                        {"rank": 1, "url": "https://one.example/", "title": "One"},
                    ]
                )
            ),
            ToolContext(session_id="research", proxy=_ExampleCredentialProxy()),
            "bounded runtimes",
        )
    )

    assert [page.source_url for page in evidence.pages] == ["https://one.example/"]
    assert [failure.error for failure in evidence.failures] == ["malformed_search_result"]


@pytest.mark.parametrize(
    "result, expected_error",
    [
        (ToolResult(content="missing evidence"), "malformed_fetch_result"),
        (ToolResult(content="failed", is_error=True), "web_operation_failed"),
    ],
)
def test_research_recipe_isolates_fetch_results_without_structured_evidence(
    result: ToolResult,
    expected_error: str,
) -> None:
    evidence = asyncio.run(
        browse_extract_verify(
            WebBridge.hosted(adapter=_MalformedFetchAdapter(result)),
            ToolContext(session_id="research", proxy=_ExampleCredentialProxy()),
            "bounded runtimes",
        )
    )

    assert evidence.pages == ()
    assert [failure.error for failure in evidence.failures] == [expected_error]


def test_external_cron_tick_is_idempotent_for_same_day_and_target() -> None:
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        first = await external_cron_tick(
            app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 20),
        )
        second = await external_cron_tick(
            app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 20),
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert first.id == second.id
    assert first.input == second.input


def test_daily_recipe_runs_cron_task_worker_agent_and_durable_result() -> None:
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        provider = _DailyProvider()
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="hosted",
                    execution_profile_identity=_HOSTED_ENVIRONMENT_IDENTITY,
                ),
                proxy=_ExampleCredentialProxy(),
            ),
            default=True,
        )
        register_daily_checker(
            app,
            WebBridge.hosted(
                adapter=_ResearchAdapter(),
                execution_profile_identity=_HOSTED_PROFILE_IDENTITY,
            ),
            model="daily-model",
        )
        task = await external_cron_tick(
            app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 20),
        )
        handled = await daily_check_worker(
            app,
            store,
            worker_id="daily-worker",
            max_tasks=1,
        )
        durable = await load_durable_daily_result(store, task.id)
        return handled, durable, provider.requests

    handled, durable, provider_requests = asyncio.run(scenario())

    assert handled == 1
    assert durable.status.value == "completed"
    assert durable.session_id == f"session_{durable.id}"
    assert provider_requests == 2


def test_daily_recipe_carries_nondefault_environment_from_task_to_run() -> None:
    async def scenario():
        store = InMemoryTaskStore()
        app = CayuApp(task_store=store, enable_logging=False)
        provider = _DailyProvider()
        app.register_provider(provider, default=True)
        app.register_environment(
            Environment(EnvironmentSpec(name="default-without-proxy")),
            default=True,
        )
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="hosted",
                    execution_profile_identity=_HOSTED_ENVIRONMENT_IDENTITY,
                ),
                proxy=_ExampleCredentialProxy(),
            )
        )
        register_daily_checker(
            app,
            WebBridge.hosted(
                adapter=_ResearchAdapter(),
                execution_profile_identity=_HOSTED_PROFILE_IDENTITY,
            ),
            model="daily-model",
            environment_name="hosted",
        )
        task = await external_cron_tick(
            app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 23),
            environment_name="hosted",
        )

        await daily_check_worker(
            app,
            store,
            worker_id="daily-worker",
            max_tasks=1,
        )
        durable = await load_durable_daily_result(store, task.id)
        session = await app.session_store.load(f"session_{task.id}")
        return durable, session, provider.requests

    durable, session, provider_requests = asyncio.run(scenario())

    assert durable.status.value == "completed"
    assert session is not None
    assert session.environment_name == "hosted"
    assert provider_requests == 2


def test_daily_recipe_does_not_recover_expired_attached_task_from_stale_evidence(
    tmp_path,
) -> None:
    async def scenario():
        session_path = tmp_path / "daily-sessions.sqlite"
        task_path = tmp_path / "daily-tasks.sqlite"
        first_sessions = SQLiteSessionStore(session_path)
        first_tasks = SQLiteTaskStore(task_path)
        first_app = CayuApp(
            session_store=first_sessions,
            task_store=first_tasks,
            enable_logging=False,
        )
        first_app.register_provider(_DailyProvider(), default=True)
        first_app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="hosted",
                    execution_profile_identity=_HOSTED_ENVIRONMENT_IDENTITY,
                ),
                proxy=_ExampleCredentialProxy(),
            ),
            default=True,
        )
        register_daily_checker(
            first_app,
            WebBridge.hosted(
                adapter=_ResearchAdapter(),
                execution_profile_identity=_HOSTED_PROFILE_IDENTITY,
            ),
            model="daily-model",
        )
        created = await external_cron_tick(
            first_app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 21),
        )
        claimed = await first_tasks.claim_task(
            "daily-worker-a",
            TaskQuery(type="webbridge_daily_public_page"),
            lease_seconds=1,
        )
        assert claimed is not None
        prepared = await first_app._session_engine._prepare_initial_run(
            RunRequest(
                agent_name="daily_web_checker",
                session_id=f"session_{created.id}",
                task_id=created.id,
                task_worker_id="daily-worker-a",
                messages=[Message.text("user", "Begin the daily page check.")],
            )
        )
        admitted = await create_admitted_session(
            first_sessions,
            request=prepared.request,
            provider_name=prepared.registered_provider.name,
            model=prepared.registered_agent.spec.model,
            execution_profile=prepared.execution_profile,
        )
        attached = await first_tasks.attach_task(
            created.id,
            session_id=admitted.session.id,
            session_invocation=SessionInvocationBinding(
                id=admitted.session.id,
                invocation=admitted.session.invocation,
            ),
            worker_id="daily-worker-a",
        )
        assert attached.status is TaskStatus.RUNNING
        assert attached.session_id == admitted.session.id
        await first_sessions.close()
        await first_tasks.close()

        await asyncio.sleep(1.05)

        restarted_sessions = SQLiteSessionStore(session_path)
        restarted_tasks = SQLiteTaskStore(task_path)
        restarted_app = CayuApp(
            session_store=restarted_sessions,
            task_store=restarted_tasks,
            enable_logging=False,
        )
        restarted_app.register_provider(_DailyProvider(), default=True)
        restarted_app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="hosted",
                    execution_profile_identity=_HOSTED_ENVIRONMENT_IDENTITY,
                ),
                proxy=_ExampleCredentialProxy(),
            ),
            default=True,
        )
        register_daily_checker(
            restarted_app,
            WebBridge.hosted(
                adapter=_ResearchAdapter(),
                execution_profile_identity=_HOSTED_PROFILE_IDENTITY,
            ),
            model="daily-model",
        )
        stale_scan_handled = await _settle_ownerless_terminal_daily_checks(
            restarted_app,
            restarted_tasks,
            max_tasks=1,
        )
        still_owned = await restarted_tasks.load_task(created.id)
        recovery = await restarted_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(session_id=admitted.session.id)
        )
        recovered_session = await restarted_sessions.load(admitted.session.id)
        await restarted_sessions.close()
        await restarted_tasks.close()
        return stale_scan_handled, still_owned, recovery, recovered_session

    stale_scan_handled, still_owned, recovery, recovered_session = asyncio.run(scenario())

    assert stale_scan_handled == 0
    assert still_owned is not None
    assert still_owned.status is TaskStatus.RUNNING
    assert still_owned.worker_id == "daily-worker-a"
    assert still_owned.session_id == f"session_{still_owned.id}"
    assert recovery.status is SessionStatus.INTERRUPTED
    assert recovered_session is not None
    assert recovered_session.status is SessionStatus.INTERRUPTED


@pytest.mark.parametrize(
    ("terminal_session_status", "terminal_task_status", "terminal_event_type"),
    [
        (SessionStatus.COMPLETED, TaskStatus.COMPLETED, EventType.SESSION_COMPLETED),
        (SessionStatus.FAILED, TaskStatus.FAILED, EventType.SESSION_FAILED),
    ],
)
def test_daily_recipe_settles_ownerless_terminal_session_after_restart(
    tmp_path,
    terminal_session_status: SessionStatus,
    terminal_task_status: TaskStatus,
    terminal_event_type: EventType,
) -> None:
    async def scenario():
        session_path = tmp_path / "ownerless-daily-sessions.sqlite"
        task_path = tmp_path / "ownerless-daily-tasks.sqlite"
        first_sessions = SQLiteSessionStore(session_path)
        first_tasks = SQLiteTaskStore(task_path)
        first_app = CayuApp(
            session_store=first_sessions,
            task_store=first_tasks,
            enable_logging=False,
        )
        first_app.register_provider(_DailyProvider(), default=True)
        first_app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="hosted",
                    execution_profile_identity=_HOSTED_ENVIRONMENT_IDENTITY,
                ),
                proxy=_ExampleCredentialProxy(),
            ),
            default=True,
        )
        register_daily_checker(
            first_app,
            WebBridge.hosted(
                adapter=_ResearchAdapter(),
                execution_profile_identity=_HOSTED_PROFILE_IDENTITY,
            ),
            model="daily-model",
        )
        created = await external_cron_tick(
            first_app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 24),
        )
        claimed = await first_tasks.claim_task(
            "daily-worker-a",
            TaskQuery(type="webbridge_daily_public_page"),
        )
        assert claimed is not None
        prepared = await first_app._session_engine._prepare_initial_run(
            RunRequest(
                agent_name="daily_web_checker",
                session_id=f"session_{created.id}",
                task_id=created.id,
                task_worker_id="daily-worker-a",
                messages=[Message.text("user", "Begin the daily page check.")],
            )
        )
        admitted = await create_admitted_session(
            first_sessions,
            request=prepared.request,
            provider_name=prepared.registered_provider.name,
            model=prepared.registered_agent.spec.model,
            execution_profile=prepared.execution_profile,
        )
        await first_tasks.attach_task(
            created.id,
            session_id=admitted.session.id,
            session_invocation=SessionInvocationBinding(
                id=admitted.session.id,
                invocation=admitted.session.invocation,
            ),
            worker_id="daily-worker-a",
        )
        await first_sessions.transition_status(
            admitted.session.id,
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.INTERRUPTED,
        )
        released = await first_tasks.release_attached_task_worker(
            created.id,
            "daily-worker-a",
        )
        assert released.status is TaskStatus.RUNNING
        assert released.worker_id is None
        assert released.lease_expires_at is None
        await first_sessions.close()
        await first_tasks.close()

        restarted_sessions = SQLiteSessionStore(session_path)
        restarted_tasks = SQLiteTaskStore(task_path)
        restarted_app = CayuApp(
            session_store=restarted_sessions,
            task_store=restarted_tasks,
            enable_logging=False,
        )
        interrupted_handled = await _settle_ownerless_terminal_daily_checks(
            restarted_app,
            restarted_tasks,
            max_tasks=1,
        )
        still_ownerless = await restarted_tasks.load_task(created.id)
        assert still_ownerless is not None
        assert still_ownerless.status is TaskStatus.RUNNING
        assert still_ownerless.worker_id is None
        assert still_ownerless.lease_expires_at is None
        await restarted_sessions.transition_status(
            admitted.session.id,
            from_statuses={SessionStatus.INTERRUPTED},
            to_status=terminal_session_status,
        )
        await restarted_sessions.close()
        await restarted_tasks.close()

        terminal_sessions = SQLiteSessionStore(session_path)
        terminal_tasks = SQLiteTaskStore(task_path)
        terminal_app = CayuApp(
            session_store=terminal_sessions,
            task_store=terminal_tasks,
            enable_logging=False,
        )
        terminal_handled = await daily_check_worker(
            terminal_app,
            terminal_tasks,
            worker_id="daily-worker-c",
            max_tasks=1,
        )
        durable = await terminal_tasks.load_task(created.id)
        recovered_session = await terminal_sessions.load(admitted.session.id)
        checkpoint = await terminal_sessions.load_checkpoint(admitted.session.id)
        active_profile = active_invocation_execution_profile_from_checkpoint(checkpoint)
        terminal_events = await terminal_sessions.query_events(
            EventQuery(
                session_id=admitted.session.id,
                event_type=terminal_event_type,
            )
        )
        await terminal_sessions.close()
        await terminal_tasks.close()
        return (
            interrupted_handled,
            terminal_handled,
            durable,
            recovered_session,
            active_profile,
            terminal_events,
        )

    (
        interrupted_handled,
        terminal_handled,
        durable,
        recovered_session,
        active_profile,
        terminal_events,
    ) = asyncio.run(scenario())

    assert interrupted_handled == 0
    assert terminal_handled == 1
    assert durable is not None
    assert durable.status is terminal_task_status
    assert durable.worker_id is None
    assert durable.lease_expires_at is None
    assert recovered_session is not None
    assert active_profile is not None
    assert active_invocation_execution_profile_is_released(
        active_profile,
        session_id=recovered_session.id,
        run_epoch=recovered_session.run_epoch,
    )
    assert len(terminal_events) == 1
    if terminal_session_status is SessionStatus.COMPLETED:
        assert durable.result == {
            "session_id": f"session_{durable.id}",
            "agent_name": "daily_web_checker",
            "environment_name": "hosted",
        }
        assert durable.error is None
    else:
        assert durable.result is None
        assert durable.error == {
            "session_id": f"session_{durable.id}",
            "message": "The attached daily check session failed.",
        }


@pytest.mark.parametrize("terminal_status", [SessionStatus.COMPLETED, SessionStatus.FAILED])
def test_daily_recipe_concurrently_settles_the_same_ownerless_terminal_task(
    tmp_path,
    terminal_status: SessionStatus,
) -> None:
    async def scenario() -> Task:
        task_status = (
            TaskStatus.COMPLETED
            if terminal_status is SessionStatus.COMPLETED
            else TaskStatus.FAILED
        )
        task_store = _ConcurrentSettlementSQLiteTaskStore(
            tmp_path / "concurrent-ownerless-daily-tasks.sqlite",
            terminal_status=task_status,
        )
        app = CayuApp(task_store=task_store, enable_logging=False)
        created = await external_cron_tick(
            app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 25),
        )
        claimed = await task_store.claim_task(
            "daily-worker-a",
            TaskQuery(type="webbridge_daily_public_page"),
        )
        assert claimed is not None
        session_id = f"session_{created.id}"
        session = Session(
            id=session_id,
            agent_name="daily_web_checker",
            provider_name="daily-provider",
            model="daily-model",
            status=terminal_status,
            invocation=session_invocation_from_task(
                claimed.invocation,
                session_id=session_id,
            ),
        )
        await task_store.attach_task(
            created.id,
            session_id=session_id,
            session_invocation=SessionInvocationBinding(
                id=session_id,
                invocation=session.invocation,
            ),
            worker_id="daily-worker-a",
        )
        await task_store.release_attached_task_worker(created.id, "daily-worker-a")
        first_snapshot = await task_store.load_task(created.id)
        second_snapshot = await task_store.load_task(created.id)
        assert first_snapshot is not None
        assert second_snapshot is not None

        await asyncio.gather(
            _settle_daily_task_from_terminal_session(
                task_store,
                first_snapshot,
                worker_id=None,
                session=session,
                environment_name=None,
            ),
            _settle_daily_task_from_terminal_session(
                task_store,
                second_snapshot,
                worker_id=None,
                session=session,
                environment_name=None,
            ),
        )
        durable = await task_store.load_task(created.id)
        await task_store.close()
        assert durable is not None
        return durable

    durable = asyncio.run(scenario())

    if terminal_status is SessionStatus.COMPLETED:
        assert durable.status is TaskStatus.COMPLETED
        assert durable.result == {
            "session_id": f"session_{durable.id}",
            "agent_name": "daily_web_checker",
            "environment_name": None,
        }
        assert durable.error is None
    else:
        assert durable.status is TaskStatus.FAILED
        assert durable.result is None
        assert durable.error == {
            "session_id": f"session_{durable.id}",
            "message": "The attached daily check session failed.",
        }


def test_daily_recipe_reconciles_a_terminal_session_before_recreating_it() -> None:
    async def scenario():
        task_store = InMemoryTaskStore()
        app = CayuApp(task_store=task_store, enable_logging=False)
        app.register_provider(_DailyProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="hosted",
                    execution_profile_identity=_HOSTED_ENVIRONMENT_IDENTITY,
                ),
                proxy=_ExampleCredentialProxy(),
            ),
            default=True,
        )
        register_daily_checker(
            app,
            WebBridge.hosted(
                adapter=_ResearchAdapter(),
                execution_profile_identity=_HOSTED_PROFILE_IDENTITY,
            ),
            model="daily-model",
        )
        created = await external_cron_tick(
            app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 22),
        )
        claimed = await task_store.claim_task(
            "daily-worker",
            TaskQuery(type="webbridge_daily_public_page"),
        )
        assert claimed is not None
        prepared = await app._session_engine._prepare_initial_run(
            RunRequest(
                agent_name="daily_web_checker",
                session_id=f"session_{created.id}",
                task_id=created.id,
                task_worker_id="daily-worker",
                messages=[Message.text("user", "Begin the daily page check.")],
            )
        )
        await create_admitted_session(
            app.session_store,
            request=prepared.request,
            provider_name=prepared.registered_provider.name,
            model=prepared.registered_agent.spec.model,
            execution_profile=prepared.execution_profile,
        )
        await app.session_store.transition_status(
            f"session_{created.id}",
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.COMPLETED,
        )

        outcome = await handle_daily_check(app, claimed, "daily-worker")
        return outcome, await task_store.load_task(created.id)

    outcome, task = asyncio.run(scenario())

    assert outcome is None
    assert task is not None
    assert task.status.value == "completed"
    assert task.result == {
        "session_id": f"session_{task.id}",
        "agent_name": "daily_web_checker",
        "environment_name": "hosted",
    }


def test_daily_recipe_rejects_unrelated_terminal_session_before_task_settlement() -> None:
    async def scenario():
        task_store = InMemoryTaskStore()
        app = CayuApp(task_store=task_store, enable_logging=False)
        app.register_provider(_DailyProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(
                    name="hosted",
                    execution_profile_identity=_HOSTED_ENVIRONMENT_IDENTITY,
                ),
                proxy=_ExampleCredentialProxy(),
            ),
            default=True,
        )
        register_daily_checker(
            app,
            WebBridge.hosted(
                adapter=_ResearchAdapter(),
                execution_profile_identity=_HOSTED_PROFILE_IDENTITY,
            ),
            model="daily-model",
        )
        created = await external_cron_tick(
            app,
            target_url="https://status.example/",
            scheduled_day=date(2026, 8, 24),
        )
        claimed = await task_store.claim_task(
            "daily-worker",
            TaskQuery(type="webbridge_daily_public_page"),
        )
        assert claimed is not None
        session_id = f"session_{created.id}"
        unrelated = await app._session_engine._prepare_initial_run(
            RunRequest(
                agent_name="daily_web_checker",
                session_id=session_id,
                messages=[Message.text("user", "Unrelated colliding session.")],
            )
        )
        await create_admitted_session(
            app.session_store,
            request=unrelated.request,
            provider_name=unrelated.registered_provider.name,
            model=unrelated.registered_agent.spec.model,
            execution_profile=unrelated.execution_profile,
        )
        await app.session_store.transition_status(
            session_id,
            from_statuses={SessionStatus.RUNNING},
            to_status=SessionStatus.COMPLETED,
        )

        with pytest.raises(ValueError, match="provenance conflict"):
            await handle_daily_check(app, claimed, "daily-worker")
        return await task_store.load_task(created.id)

    task = asyncio.run(scenario())

    assert task is not None
    assert task.status.value == "claimed"
    assert task.result is None
