"""Tool inventory and usage through real runtime and control-plane reads."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from tests.core.test_cloud_inventory import _Tool
from tests.core.test_tool_completion import FinalTool, app_for, call
from tests.core.test_tool_round_publication_failure_matrix import _TwoCallProvider

from cayu import AgentSpec, CayuApp, Message, RunRequest, Tool, ToolResult
from cayu.cloud_inventory import cloud_registration
from cayu.evals.testing import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.server import ServerConfig, create_server
from cayu.sessions import InMemorySessionStore
from cayu.storage import SQLiteSessionStore
from cayu.tools.policy import AlwaysRequireApprovalToolPolicy


class _ExecutableTool(Tool):
    def __init__(self, name):
        super().__init__(FinalTool().spec.model_copy(update={"name": name}))

    async def run(self, ctx, args):
        return ToolResult(content="done")


def test_control_plane_tool_sources_and_approval_match_cloud():
    async def run():
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(
            AgentSpec(name="specialist", model="scripted"),
            tools=[_Tool()],
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        cloud = await cloud_registration(app)
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            response = await client.get("/api/agents")
            assert response.status_code == 200
            tool = response.json()["agents"][0]["tools"][0]
            assert tool["source"] == cloud.tools[0].source.model_dump(mode="json")
            assert tool["approval"] == cloud.tools[0].approval == "always"
            assert tool["input_schema"]
        await app.aclose()

    asyncio.run(run())


def test_local_inventory_has_no_cloud_registration_tool_cap():
    async def run():
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(
            AgentSpec(name="specialist", model="scripted"),
            tools=[_Tool(f"tool_{index:04}") for index in range(501)],
        )
        assert len((await cloud_registration(app)).tools) == 500
        server = create_server(app, config=ServerConfig.local_development())
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="http://test"
        ) as client:
            tools = (await client.get("/api/agents")).json()["agents"][0]["tools"]
            assert len(tools) == 501
            assert all(tool["source"]["kind"] for tool in tools)
        await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_tool_usage_counts_real_repeated_calls_and_returns_exact_recent_links(
    tmp_path, backend, store_override=None
):
    async def run():
        store = store_override or (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "usage.sqlite")
        )
        app = app_for(
            store,
            _TwoCallProvider(
                [
                    call(),
                    call(),
                    [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
                ]
            ),
            FinalTool(),
        )
        _ = [
            event
            async for event in app.run(
                RunRequest(
                    session_id="tool-history",
                    agent_name="support",
                    messages=[Message.text("user", "Go")],
                )
            )
        ]
        server = create_server(app, config=ServerConfig.local_development())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get("/api/tools/usage")
                assert response.status_code == 200
                rows = response.json()["tools"]
                used = [row for row in rows if row["calls"]]
                assert len(used) == 1
                assert used[0]["calls"] == 2
                assert used[0]["failures"] == 0
                first = await client.get(
                    "/api/tools/calls", params={"name": used[0]["name"], "limit": 1}
                )
                assert first.status_code == 200, first.text
                first_page = first.json()
                item = first_page["calls"][0]
                session = await store.load("tool-history")
                assert item["session_instance_id"] == session.instance_id
                exact = await client.get(
                    "/api/sessions/tool-history/events", params={"event_id": item["event_id"]}
                )
                assert exact.status_code == 200
                assert [event["id"] for event in exact.json()["events"]] == [item["event_id"]]
                second = await client.get(
                    "/api/tools/calls",
                    params={
                        "name": used[0]["name"],
                        "limit": 1,
                        "cursor": first_page["next_cursor"],
                    },
                )
                assert second.status_code == 200, second.text
                assert second.json()["calls"][0]["sequence"] < item["sequence"]
                assert second.json()["through_sequence"] == first_page["through_sequence"]
                changed = await client.get(
                    "/api/tools/calls",
                    params={"name": "different", "cursor": first_page["next_cursor"]},
                )
                assert changed.status_code == 409
                invalid = await client.get("/api/tools/calls", params={"cursor": "not-a-cursor"})
                assert invalid.status_code == 422 and isinstance(invalid.json()["detail"], list)

                long_name = "🙂" * 4096
                empty = (await client.get("/api/tools/calls", params={"name": long_name})).json()
                assert empty["calls"] == []
                assert len(empty["next_cursor"]) < 8192
                continued = await client.get(
                    "/api/tools/calls", params={"name": long_name, "cursor": empty["next_cursor"]}
                )
                assert continued.status_code == 200, continued.text

                assert not {"arguments", "result", "tool_call_id", "model_step_id"}.intersection(
                    item
                )
                if backend == "sqlite":
                    await app.aclose()
                    app = app_for(
                        SQLiteSessionStore(tmp_path / "usage.sqlite"),
                        _TwoCallProvider([]),
                        FinalTool(),
                    )
                    restarted = create_server(app, config=ServerConfig.local_development())
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=restarted), base_url="http://test"
                    ) as fresh:
                        resumed = await fresh.get(
                            "/api/tools/calls",
                            params={
                                "name": used[0]["name"],
                                "limit": 1,
                                "cursor": first_page["next_cursor"],
                            },
                        )
                        assert resumed.status_code == 200, resumed.text
                        assert resumed.json() == second.json()
        finally:
            await app.aclose()

    asyncio.run(run())


def test_projected_collisions_preserve_declarations_and_do_not_create_raw_name_oracle():
    from cayu.vaults.redaction import SecretRedactor

    async def run():
        private_names = ("secret_alpha", "secret_beta")
        app = CayuApp(enable_logging=False)
        provider = _TwoCallProvider(
            [
                [
                    ModelStreamEvent.tool_call(name=name, arguments={}, id=f"call_{index}"),
                    ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
                ]
                for index, name in enumerate(private_names)
            ]
            + [[ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]]
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="specialist", model="scripted"),
            tools=[_ExecutableTool(name) for name in private_names],
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="safe-session",
                        agent_name="specialist",
                        messages=[Message.text("user", "Go")],
                    )
                )
            ]
            assert events[-1].type == "session.completed", str(events[-1].payload)
            # A later process/configuration can discover new workload secrets
            # in retained history; admission correctly forbids executing them.
            original = app
            app = CayuApp(
                session_store=original.session_store,
                secret_redactor=SecretRedactor(private_names),
                enable_logging=False,
            )
            app.register_provider(_TwoCallProvider([]), default=True)
            app.register_agent(
                AgentSpec(name="specialist", model="scripted"),
                tools=[_ExecutableTool(name) for name in private_names],
            )
            await original.aclose()
            assert not (await cloud_registration(app)).tools
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                inventory = (await client.get("/api/agents")).json()
                tools = inventory["agents"][0]["tools"]
                assert len(tools) == 2
                projected = app.redact_json(private_names[0])
                assert [tool["name"] for tool in tools] == [projected, projected]
                usage = (await client.get("/api/tools/usage")).json()["tools"]
                assert len(usage) == 1 and usage[0]["calls"] == 2
                public = await client.get("/api/tools/calls", params={"name": projected})
                assert public.status_code == 200, public.text
                assert len(public.json()["calls"]) == 2
                for private in private_names:
                    negative = await client.get("/api/tools/calls", params={"name": private})
                    assert negative.status_code == 200
                    assert negative.json()["calls"] == []
                    assert private not in public.text
                    assert private not in str(inventory)
        finally:
            await app.aclose()

    asyncio.run(run())


def test_hosted_terminal_usage_matches_cloud_and_excludes_in_progress():
    from cayu.cloud_inventory import _build_registration, _cloud_usage_report
    from cayu.providers.hosted import OpenAIWebSearch

    class HostedProvider(_TwoCallProvider):
        def preflight_hosted_tools(self, *, model, hosted_tools, options):
            pass

    async def run():
        statuses = [
            "in_progress",
            "searching",
            "completed",
            "failed",
            "incomplete",
            "outcome_unknown",
        ]
        provider = HostedProvider(
            [
                [
                    *[
                        ModelStreamEvent.hosted_tool_call(
                            {
                                "tool_type": "web_search",
                                "call_id": f"hosted-{index}",
                                "status": status,
                            }
                        )
                        for index, status in enumerate(statuses)
                    ],
                    ModelStreamEvent.text_delta("Done"),
                    ModelStreamEvent.completed(),
                ]
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="specialist", model="scripted"), hosted_tools=[OpenAIWebSearch()]
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="hosted-session",
                        agent_name="specialist",
                        messages=[Message.text("user", "Go")],
                    )
                )
            ]
            assert events[-1].type == "session.completed", str(events[-1].payload)
            _, registered = await _build_registration(app, project_root=None, start_cursor=None)
            report, _ = await _cloud_usage_report(app, registered, after=None, limit=1000)
            assert report is not None and len(report.tools) == 1
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                tools = (await client.get("/api/agents")).json()["agents"][0]["tools"]
                assert len(tools) == 1
                assert tools[0]["source"]["kind"] == "provider"
                assert tools[0]["source"]["provider"] == "openai"
                assert tools[0]["approval"] == "never"
                usage = (await client.get("/api/tools/usage")).json()["tools"][0]
                assert usage["calls"] == report.tools[0].calls == 4
                assert usage["failures"] == report.tools[0].failures == 1
                recent = (await client.get("/api/tools/calls")).json()["calls"]
                assert {item["outcome"] for item in recent} == {
                    "completed",
                    "failed",
                    "incomplete",
                    "outcome_unknown",
                }
                assert len(recent) == 4
        finally:
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("phase", [1, 2])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recent_calls_reject_session_replacement_during_link_enrichment(
    tmp_path, backend, monkeypatch, phase
):
    async def run():
        store = (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "replacement.sqlite")
        )
        app = app_for(
            store,
            _TwoCallProvider(
                [
                    call(),
                    [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
                    [ModelStreamEvent.text_delta("New"), ModelStreamEvent.completed()],
                ]
            ),
            FinalTool(),
        )
        request = RunRequest(
            session_id="reused", agent_name="support", messages=[Message.text("user", "Go")]
        )
        events = [event async for event in app.run(request)]
        assert events[-1].type == "session.completed"
        old = await store.load("reused")
        entered, release = asyncio.Event(), asyncio.Event()
        original_load = store.load_invocation_snapshot
        observations = 0

        async def blocked_load(session_id):
            nonlocal observations
            observations += 1
            if observations == phase:
                entered.set()
                await release.wait()
            return await original_load(session_id)

        monkeypatch.setattr(store, "load_invocation_snapshot", blocked_load)
        server = create_server(app, config=ServerConfig.local_development())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                pending = asyncio.create_task(client.get("/api/tools/calls"))
                try:
                    # A base without the endpoint fails here directly rather than hanging.
                    gate = asyncio.create_task(entered.wait())
                    done, _ = await asyncio.wait(
                        {pending, gate}, timeout=10, return_when=asyncio.FIRST_COMPLETED
                    )
                    assert gate in done, (
                        pending.result().text if pending in done else "lookup did not enter"
                    )
                    monkeypatch.delattr(store, "load_invocation_snapshot")
                    await store.delete_session("reused")
                    replacement = [event async for event in app.run(request)]
                    assert replacement[-1].type == "session.completed", str(replacement[-1].payload)
                    assert (await original_load("reused")).session_instance_id != old.instance_id
                    release.set()
                    response = await pending
                    assert response.status_code == 409, response.text
                finally:
                    release.set()
                    gate.cancel()
                    await asyncio.gather(gate, return_exceptions=True)
                    await asyncio.gather(pending, return_exceptions=True)
        finally:
            await app.aclose()

    asyncio.run(run())


def test_member_can_read_tool_inventory_and_usage_but_unauthenticated_cannot():
    from cryptography.hazmat.primitives.asymmetric import ec
    from tests.server.test_proxy_identity import FakeJwks, _as, _auth, _server, _token

    key = ec.generate_private_key(ec.SECP256R1())
    with _server(_auth(FakeJwks(key))) as client:
        for path in ("/api/agents", "/api/tools/usage", "/api/tools/calls"):
            allowed = client.get(path, headers=_as(_token(key, role="member")))
            assert allowed.status_code == 200, allowed.text
            assert client.get(path).status_code == 401
        contract = client.get("/api/contract", headers=_as(_token(key, role="member"))).json()
        assert contract["capabilities"]["surfaces"]["tool_usage"]["read"]["enabled"] is True


def test_specialist_variants_keep_schema_approval_and_shared_package_mcp_sources(monkeypatch):
    from tests.core.test_mcp import FakeMcpSession, _fake_server_spec, _fake_tool_definitions

    from cayu.mcp import McpToolset
    from cayu.tools.policy import StaticToolPolicy

    monkeypatch.setattr(_Tool, "__module__", "invoice_tools.lookup")
    monkeypatch.setattr(
        "cayu.tool_inventory.metadata.packages_distributions",
        lambda: {"invoice_tools": ["invoice-tools"]},
    )
    monkeypatch.setattr("cayu.tool_inventory.metadata.version", lambda _: "1.2.3")

    async def run():
        app = CayuApp(enable_logging=False)
        definitions = _fake_tool_definitions("echo")
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "inventory-test"}),
            session=FakeMcpSession(definitions=definitions),
            definitions=definitions,
        )
        app.register_agent(
            AgentSpec(name="first", model="model"),
            tools=[_Tool()],
            mcp_toolsets=(toolset,),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        app.register_agent(
            AgentSpec(name="second", model="model"),
            tools=[_Tool()],
            tool_policy=StaticToolPolicy(deny=["lookup"]),
        )
        try:
            cloud = await cloud_registration(app)
            merged = next(tool for tool in cloud.tools if tool.name == "lookup")
            assert merged.approval == "unknown"
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                inventory = (await client.get("/api/agents")).json()["agents"]
                first = next(tool for tool in inventory[0]["tools"] if tool["name"] == "lookup")
                second = inventory[1]["tools"][0]
                assert first["approval"] == "always" and second["approval"] == "blocked"
                assert first["source"] == second["source"] == merged.source.model_dump(mode="json")
                assert first["source"]["kind"] == "package"
                assert first["input_schema"]["properties"] == second["input_schema"]["properties"]
                mcp = next(
                    tool for tool in inventory[0]["tools"] if tool["source"]["kind"] == "mcp"
                )
                assert mcp["source"]["server"] == toolset.server.name
                detail = (await client.get("/api/agents/first")).json()["agents"][0]
                assert detail == inventory[0]
        finally:
            await app.aclose()

    asyncio.run(run())


def test_tool_usage_postgres_exact_links(tmp_path, postgres_dsn):
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    test_tool_usage_counts_real_repeated_calls_and_returns_exact_recent_links(
        tmp_path, "postgres", PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
    )


@pytest.mark.parametrize("approve", [False, True])
def test_approval_decisions_are_separate_from_executed_calls(approve):
    from cayu.approvals import ToolApprovalDecision, ToolApprovalRequest
    from cayu.cloud_inventory import _build_registration, _cloud_usage_report

    async def run():
        app = app_for(
            InMemorySessionStore(),
            _TwoCallProvider(
                [call(), [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()]]
            ),
            FinalTool(),
            tool_policy=AlwaysRequireApprovalToolPolicy(),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="approval",
                        agent_name="support",
                        messages=[Message.text("user", "Go")],
                        tool_completion={"tool_names": ["ask_customer"]},
                    )
                )
            ]
            approval = next(
                event for event in events if event.type == "tool.call.approval_requested"
            )
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                before = await client.get("/api/tools/usage")
                assert before.status_code == 200, before.text
                assert before.json()["tools"][0]["calls"] == 0
                _ = [
                    event
                    async for event in app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id="approval",
                            approval_id=approval.payload["approval"]["approval_id"],
                            tool_call_id=approval.payload["tool_call_id"],
                            tool_round_id=approval.payload["tool_round_id"],
                            decision=ToolApprovalDecision.APPROVE
                            if approve
                            else ToolApprovalDecision.DENY,
                        )
                    )
                ]
                row = (await client.get("/api/tools/usage")).json()["tools"][0]
                assert (row["calls"], row["approved"], row["rejected"]) == (
                    (1, 1, 0) if approve else (0, 0, 1)
                )
                _, registered = await _build_registration(app, project_root=None, start_cursor=None)
                report, _ = await _cloud_usage_report(app, registered, after=None, limit=1000)
                assert report.tools[0].calls == row["calls"]
                assert report.tools[0].approved == row["approved"]
                assert report.tools[0].rejected == row["rejected"]
                assert len((await client.get("/api/tools/calls")).json()["calls"]) == int(approve)
        finally:
            await app.aclose()

    asyncio.run(run())


def test_usage_scans_multiple_batches_and_filtered_calls_continue_empty_pages():
    async def run():
        calls = [ModelStreamEvent.tool_call(name="older", arguments={}, id="older")]
        calls.extend(
            ModelStreamEvent.tool_call(name="newer", arguments={}, id=f"newer_{index}")
            for index in range(110)
        )
        provider = _TwoCallProvider(
            [
                [*calls, ModelStreamEvent.completed({"finish_reason": "tool_calls"})],
                [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
            ]
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="specialist", model="model"),
            tools=[_ExecutableTool("older"), _ExecutableTool("newer")],
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="many",
                        agent_name="specialist",
                        messages=[Message.text("user", "Go")],
                    )
                )
            ]
            assert events[-1].type == "session.completed", str(events[-1].payload)
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get("/api/tools/usage")
                assert response.status_code == 200, response.text
                rows = {row["name"]: row for row in response.json()["tools"]}
                assert rows["newer"]["calls"] == 110 and rows["older"]["calls"] == 1
                first = (await client.get("/api/tools/calls", params={"name": "older"})).json()
                assert first["calls"] == [] and first["next_cursor"] is not None
                second = (
                    await client.get(
                        "/api/tools/calls", params={"name": "older", "cursor": first["next_cursor"]}
                    )
                ).json()
                assert len(second["calls"]) == 1 and second["calls"][0]["name"] == "older"
                assert second["through_sequence"] == first["through_sequence"]
        finally:
            await app.aclose()

    asyncio.run(run())


def test_cancelled_usage_read_propagates_cancellation_and_allows_fresh_read(monkeypatch):
    async def run():
        app = CayuApp(enable_logging=False)
        entered = asyncio.Event()
        original = app.session_store.fold_event_snapshot

        async def blocked(*args, **kwargs):
            await original(*args, **kwargs)
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(app.session_store, "fold_event_snapshot", blocked)
        server = create_server(app, config=ServerConfig.local_development())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                pending = asyncio.create_task(client.get("/api/tools/usage"))
                gate = asyncio.create_task(entered.wait())
                try:
                    done, _ = await asyncio.wait(
                        {pending, gate}, timeout=10, return_when=asyncio.FIRST_COMPLETED
                    )
                    assert gate in done, (
                        pending.result().text if pending in done else "read did not enter"
                    )
                    pending.cancel()
                    assert pending.cancelling() == 1
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert pending.cancelled() and pending.cancelling() == 1
                    monkeypatch.delattr(app.session_store, "fold_event_snapshot")
                    assert (await client.get("/api/tools/usage")).json()["tools"] == []
                finally:
                    gate.cancel()
                    pending.cancel()
                    await asyncio.gather(gate, pending, return_exceptions=True)
        finally:
            await app.aclose()

    asyncio.run(run())


def test_unsupported_store_reads_fail_explicitly_and_discovery_is_honest():
    from cayu.sessions.store import SessionStore

    class UnsupportedStore(InMemorySessionStore):
        fold_event_snapshot = SessionStore.fold_event_snapshot
        query_events_bounded = SessionStore.query_events_bounded
        read_store_identity = SessionStore.read_store_identity

    async def run():
        app = CayuApp(session_store=UnsupportedStore(), enable_logging=False)
        try:
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                contract = (await client.get("/api/contract")).json()
                assert (
                    contract["capabilities"]["surfaces"]["tool_usage"]["read"]["enabled"] is False
                )
                assert (
                    contract["capabilities"]["surfaces"]["tool_inventory"]["read"]["enabled"]
                    is True
                )
                assert (await client.get("/api/tools/usage")).status_code == 501
                assert (await client.get("/api/tools/calls")).status_code == 501
        finally:
            await app.aclose()

    asyncio.run(run())


def test_inventory_does_not_redact_and_revalidate_unpublished_manifest_controls():
    from cayu.vaults.redaction import SecretRedactor

    async def run():
        app = CayuApp(secret_redactor=SecretRedactor("in_process"), enable_logging=False)
        app.register_agent(AgentSpec(name="specialist", model="model"), tools=[_Tool()])
        try:
            cloud = await cloud_registration(app)
            assert cloud.tools[0].approval == "never"
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get("/api/agents")
                assert response.status_code == 200, response.text
                assert response.json()["agents"][0]["tools"][0]["approval"] == "never"
                assert "in_process" not in response.text
        finally:
            await app.aclose()

    asyncio.run(run())


def test_large_event_batches_shrink_without_partial_usage_totals(monkeypatch):
    from cayu.sessions.event_queries import EventQueryResultTooLarge

    async def run():
        app = app_for(
            InMemorySessionStore(),
            _TwoCallProvider(
                [
                    call(),
                    call(),
                    [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
                ]
            ),
            FinalTool(),
        )
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="bounded",
                        agent_name="support",
                        messages=[Message.text("user", "Go")],
                    )
                )
            ]
            from cayu.sessions import _memory_event_reads

            original = _memory_event_reads._query_events_unlocked
            limits = []

            def byte_limited(state, query, *, max_bytes=None):
                limits.append(query.limit)
                records = original(state, query, max_bytes=max_bytes)
                if query.limit > 1 and records:
                    raise EventQueryResultTooLarge(max_bytes)
                return records

            monkeypatch.setattr(_memory_event_reads, "_query_events_unlocked", byte_limited)
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get("/api/tools/usage")
                assert response.status_code == 200, response.text
                assert response.json()["tools"][0]["calls"] == 2
                assert 100 in limits and limits.count(1) >= 3
                calls = (await client.get("/api/tools/calls")).json()
                assert len(calls["calls"]) == 1 and calls["next_cursor"] is not None
                following = (
                    await client.get("/api/tools/calls", params={"cursor": calls["next_cursor"]})
                ).json()
                assert len(following["calls"]) == 1
                assert following["calls"][0]["sequence"] < calls["calls"][0]["sequence"]
        finally:
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_recent_calls_use_bounded_identity_and_safe_cutoff(
    tmp_path, monkeypatch, backend, store_override=None
):
    async def run():
        store = store_override or (
            InMemorySessionStore()
            if backend == "memory"
            else SQLiteSessionStore(tmp_path / "bounded.sqlite")
        )
        app = app_for(
            store,
            _TwoCallProvider(
                [
                    call(),
                    call(),
                    [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
                ]
            ),
            FinalTool(),
        )
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="large",
                        agent_name="support",
                        messages=[Message.text("user", "Go")],
                    )
                )
            ]
            assert events[-1].type == "session.completed"
            await store.update_metadata("large", {"large": "x" * 200_000})
            await store.update_labels("large", {f"label{i}": "x" * 500 for i in range(100)})

            async def forbidden_load(session_id):
                pytest.fail("Recent calls hydrated full session metadata")

            original = store.fold_event_snapshot
            cutoffs = []

            async def bounded(query, consume, *, max_bytes, max_pages=None):
                cutoffs.append(max_pages)
                return await original(query, consume, max_bytes=max_bytes, max_pages=max_pages)

            monkeypatch.setattr(store, "load", forbidden_load)
            monkeypatch.setattr(store, "fold_event_snapshot", bounded)
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                totals = await client.get("/api/tools/usage")
                assert totals.status_code == 200, totals.text
                assert cutoffs == [None]
                response = await client.get("/api/tools/calls")
                assert response.status_code == 200, response.text
                assert len(response.json()["calls"]) == 2
                assert cutoffs == [None, 1]
        finally:
            await app.aclose()

    asyncio.run(run())


def test_tool_usage_postgres_snapshot_includes_later_commits(tmp_path, postgres_dsn):
    from tests.core.postgres_contention_support import drop_cayu_tables

    from cayu import Event, EventType
    from cayu.sessions import SessionIdentity
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        await drop_cayu_tables(postgres_dsn)
        store = PostgresSessionStore(
            postgres_dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE
        )
        app = app_for(
            store,
            _TwoCallProvider(
                [
                    call(),
                    call(),
                    [ModelStreamEvent.text_delta("Done"), ModelStreamEvent.completed()],
                ]
            ),
            FinalTool(),
        )
        release = asyncio.Event()
        pending = None
        try:
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="older",
                        agent_name="support",
                        messages=[Message.text("user", "Go")],
                    )
                )
            ]
            assert events[-1].type == "session.completed"
            for session_id in ("slow", "fast"):
                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="support",
                        messages=[Message.text("user", "Go")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="fake-model"),
                )
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                baseline = (await client.get("/api/tools/usage")).json()
                inserted = asyncio.Event()
                pending = asyncio.create_task(
                    _append_event_without_commit(
                        postgres_dsn,
                        session_id="slow",
                        event=Event(
                            id="slow-call",
                            type=EventType.TOOL_CALL_COMPLETED,
                            session_id="slow",
                            tool_name=FinalTool().spec.name,
                        ),
                        inserted=inserted,
                        release=release,
                    )
                )
                await asyncio.wait_for(inserted.wait(), 10)
                await store.append_event(
                    "fast",
                    Event(
                        id="fast-call",
                        type=EventType.TOOL_CALL_COMPLETED,
                        session_id="fast",
                        tool_name=FinalTool().spec.name,
                    ),
                )
                during = await client.get("/api/tools/usage")
                assert during.status_code == 200, during.text
                assert during.json()["tools"][0]["calls"] == baseline["tools"][0]["calls"] + 1
                first = await client.get("/api/tools/calls", params={"limit": 1})
                assert first.status_code == 200, first.text
                assert first.json()["through_sequence"] > baseline["through_sequence"]
                assert len(first.json()["calls"]) == 1
                cursor = first.json()["next_cursor"]
                release.set()
                await asyncio.wait_for(pending, 10)
                second = await client.get("/api/tools/calls", params={"limit": 1, "cursor": cursor})
                assert second.status_code == 200, second.text
                assert len(second.json()["calls"]) == 1
                assert second.json()["calls"][0]["sequence"] < first.json()["calls"][0]["sequence"]
                fresh = await client.get("/api/tools/calls")
                assert fresh.status_code == 200, fresh.text
                assert len(fresh.json()["calls"]) == 4
                totals = (await client.get("/api/tools/usage")).json()
                assert totals["tools"][0]["calls"] == 4
                assert totals["through_sequence"] > baseline["through_sequence"]
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await app.aclose()

    asyncio.run(run())


def test_recent_calls_postgres_bounded_identity(tmp_path, monkeypatch, postgres_dsn):
    from tests.core.postgres_contention_support import drop_cayu_tables

    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    asyncio.run(drop_cayu_tables(postgres_dsn))
    test_recent_calls_use_bounded_identity_and_safe_cutoff(
        tmp_path,
        monkeypatch,
        "postgres",
        PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE),
    )


async def _append_event_without_commit(
    dsn: str,
    *,
    session_id: str,
    event,
    inserted: asyncio.Event,
    release: asyncio.Event,
    xid_ready: asyncio.Event | None = None,
    insert_gate: asyncio.Event | None = None,
) -> None:
    import json

    import psycopg

    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            if xid_ready is not None:
                await cur.execute("SELECT pg_current_xact_id()")
                xid_ready.set()
                assert insert_gate is not None
                await insert_gate.wait()
            await cur.execute(
                """
                UPDATE cayu_sessions
                SET event_seq = event_seq + 1
                WHERE id = %s
                RETURNING event_seq
                """,
                (session_id,),
            )
            row = await cur.fetchone()
            assert row is not None
            session_order = row[0]
            await cur.execute(
                """
                INSERT INTO cayu_events (
                    session_id,
                    session_order,
                    event_id,
                    event_type,
                    timestamp,
                    agent_name,
                    environment_name,
                    workflow_name,
                    tool_name,
                    payload,
                    event
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    session_id,
                    session_order,
                    event.id,
                    str(event.type),
                    event.timestamp,
                    event.agent_name,
                    event.environment_name,
                    event.workflow_name,
                    event.tool_name,
                    json.dumps(event.payload),
                    json.dumps(event.model_dump(mode="json")),
                ),
            )
            inserted.set()
            await release.wait()
        await conn.commit()


def test_tool_source_metadata_is_reused_per_build_and_refreshed_between_builds(monkeypatch):
    monkeypatch.setattr(_Tool, "__module__", "inventory_package.tools")
    scans = []
    versions = []
    current_package = "first-package"

    def distributions():
        scans.append(current_package)
        return {"inventory_package": [current_package]}

    def version(package):
        versions.append(package)
        return "1.0" if package == "first-package" else "2.0"

    monkeypatch.setattr("cayu.tool_inventory.metadata.packages_distributions", distributions)
    monkeypatch.setattr("cayu.tool_inventory.metadata.version", version)

    async def run():
        nonlocal current_package
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        app.register_agent(
            AgentSpec(name="specialist", model="scripted"),
            tools=[_Tool(f"tool_{i}") for i in range(20)],
        )
        try:
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get("/api/agents")
                assert response.status_code == 200, response.text
                assert scans == versions == ["first-package"]
                assert {
                    tool["source"]["package"] for tool in response.json()["agents"][0]["tools"]
                } == {"first-package"}
                current_package = "second-package"
                response = await client.get("/api/agents/specialist")
                assert response.status_code == 200, response.text
                assert scans == versions == ["first-package", "second-package"]
                assert {
                    tool["source"]["version"] for tool in response.json()["agents"][0]["tools"]
                } == {"2.0"}
                registration = await cloud_registration(app)
                assert scans == versions == ["first-package", "second-package", "second-package"]
                assert {tool.source.package for tool in registration.tools} == {"second-package"}
        finally:
            await app.aclose()

    asyncio.run(run())


def test_postgres_usage_snapshot_does_not_confuse_xid_and_sequence_order(postgres_dsn):
    import psycopg
    from tests.core.postgres_contention_support import drop_cayu_tables

    from cayu import Event, EventType
    from cayu.sessions import SessionIdentity
    from cayu.storage import PostgresSessionStore
    from cayu.storage.migrations import SchemaMode

    async def run():
        await drop_cayu_tables(postgres_dsn)
        store = PostgresSessionStore(postgres_dsn, schema_mode=SchemaMode.CREATE)
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(AgentSpec(name="agent", model="model"), tools=[FinalTool()])
        ready, insert_gate, inserted, release = (asyncio.Event() for _ in range(4))
        release.set()
        pending = None
        try:
            for session_id in ("older-xid", "newer-xid"):
                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="agent",
                        messages=[Message.text("user", "go")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="model"),
                )
            pending = asyncio.create_task(
                _append_event_without_commit(
                    postgres_dsn,
                    session_id="older-xid",
                    event=Event(
                        id="unrelated", type=EventType.SESSION_COMPLETED, session_id="older-xid"
                    ),
                    inserted=inserted,
                    release=release,
                    xid_ready=ready,
                    insert_gate=insert_gate,
                )
            )
            await asyncio.wait_for(ready.wait(), 10)
            async with await psycopg.AsyncConnection.connect(postgres_dsn) as middle:
                await middle.execute("SELECT pg_current_xact_id()")
                await store.append_event(
                    "newer-xid",
                    Event(
                        id="committed-call",
                        type=EventType.TOOL_CALL_COMPLETED,
                        session_id="newer-xid",
                        tool_name=FinalTool().spec.name,
                    ),
                )
                insert_gate.set()
                await asyncio.wait_for(pending, 10)
                server = create_server(app, config=ServerConfig.local_development())
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=server), base_url="http://test"
                ) as client:
                    usage = await client.get("/api/tools/usage")
                    assert usage.status_code == 200, usage.text
                    assert usage.json()["tools"][0]["calls"] == 1
                    calls = await client.get("/api/tools/calls")
                    assert calls.status_code == 200, calls.text
                    assert len(calls.json()["calls"]) == 1
                    assert calls.json()["calls"][0]["sequence"] < usage.json()["through_sequence"]
        finally:
            insert_gate.set()
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await app.aclose()

    asyncio.run(run())


def test_sqlite_http_usage_does_not_mix_retention_between_pages(tmp_path):
    import threading

    from cayu import Event, EventType
    from cayu.sessions import SessionIdentity

    async def run():
        store = SQLiteSessionStore(tmp_path / "retained-snapshot.sqlite")
        app = CayuApp(session_store=store, enable_logging=False)
        app.register_agent(AgentSpec(name="agent", model="model"), tools=[FinalTool()])
        entered = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        pending = None
        try:
            for session_id, count in (("first", 100), ("last", 1)):
                await store.create(
                    RunRequest(
                        session_id=session_id,
                        agent_name="agent",
                        messages=[Message.text("user", "go")],
                    ),
                    identity=SessionIdentity(provider_name="fake", model="model"),
                )
                for index in range(count):
                    await store.append_event(
                        session_id,
                        Event(
                            id=f"call-{index}",
                            type=EventType.TOOL_CALL_COMPLETED,
                            session_id=session_id,
                            tool_name=FinalTool().spec.name,
                        ),
                    )

            def before_second_page(statement):
                if (
                    "bounded_candidates" in statement
                    and "cayu_events.sequence > 100" in statement
                    and not entered.is_set()
                ):
                    loop.call_soon_threadsafe(entered.set)
                    release.wait(10)

            for _, connection in store._readers:
                connection.set_trace_callback(before_second_page)
            server = create_server(app, config=ServerConfig.local_development())
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                pending = asyncio.create_task(client.get("/api/tools/usage"))
                await asyncio.wait_for(entered.wait(), 5)
                await store.delete_session("last")
                release.set()
                response = await asyncio.wait_for(pending, 5)
                assert response.status_code == 200, response.text
                assert response.json()["tools"][0]["calls"] == 101
                assert response.json()["through_sequence"] == 101
                fresh = await client.get("/api/tools/usage")
                assert fresh.status_code == 200, fresh.text
                assert fresh.json()["tools"][0]["calls"] == 100
        finally:
            release.set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            for _, connection in store._readers:
                connection.set_trace_callback(None)
            await app.aclose()

    asyncio.run(run())
