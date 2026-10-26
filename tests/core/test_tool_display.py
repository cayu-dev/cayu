"""Presentation declarations stay faithful through public ToolSpec ownership."""

import asyncio
from copy import deepcopy

import httpx
import pytest

from cayu import AgentSpec, CayuApp, Tool, ToolResult, ToolSpec
from cayu.evals.testing import ScriptedModelProvider
from cayu.server import ServerConfig, create_server


class DisplayTool(Tool):
    async def run(self, ctx, args):
        return ToolResult(content="unchanged")


def test_display_round_trips_and_defensive_copies():
    value = {"verb": "Read", "noun": "entry", "plural": "entries"}
    spec = ToolSpec(name="custom", display=value)
    value["verb"] = "Changed"
    assert spec.display.verb == "Read"
    assert spec.model_copy().display == spec.display
    assert deepcopy(spec).display == spec.display
    assert ToolSpec.model_validate_json(spec.model_dump_json()).display == spec.display
    changed = spec.model_copy(update={"display": {"verb": "Searched"}})
    assert changed.display.verb == "Searched"
    assert changed.display.noun is None
    assert ToolSpec(name="unlabelled").display is None


@pytest.mark.parametrize(
    "display",
    [
        {},
        {"verb": ""},
        {"verb": " Read"},
        {"verb": "Read\nfiles"},
        {"verb": 1},
        {"verb": True},
        {"verb": "Read", "noun": ""},
        {"verb": "Read", "plural": "entries"},
        {"verb": "Read", "extra": "forbidden"},
        {"verb": "\ud800"},
    ],
)
def test_invalid_display_rejected_without_normalizing(display):
    with pytest.raises((TypeError, ValueError)):
        ToolSpec(name="custom", display=display)


def test_registered_display_snapshot_and_unlabelled_fallback_over_http():
    async def run():
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([]), default=True)
        labelled = DisplayTool(
            ToolSpec(name="custom", display={"verb": "Read", "noun": "entry", "plural": "entries"})
        )
        app.register_agent(
            AgentSpec(name="reader", model="scripted"),
            tools=[labelled, DisplayTool(ToolSpec(name="read_file"))],
        )
        labelled.spec = labelled.spec.model_copy(update={"display": {"verb": "Changed"}})
        server = create_server(app, config=ServerConfig.local_development())
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://test"
            ) as client:
                response = await client.get("/api/agents")
                assert response.status_code == 200
                tools = {row["name"]: row for row in response.json()["agents"][0]["tools"]}
                assert tools["custom"]["display"] == {
                    "verb": "Read",
                    "noun": "entry",
                    "plural": "entries",
                }
                assert tools["read_file"]["display"] is None
        finally:
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "name",
    [
        "ReadFileTool",
        "EditFileTool",
        "DeleteFileTool",
        "WriteFileTool",
        "ListFilesTool",
        "ListArtifactsTool",
        "ExecCommandTool",
        "UserInputTool",
        "WebFetchTool",
        "ApplyPatchTool",
        "GitChangesTool",
        "SearchTextTool",
        "ScreenshotPageTool",
        "SearchKnowledgeTool",
        "RememberKnowledgeTool",
        "ListKnowledgeTool",
        "ReadKnowledgeTool",
        "BrowserSessionTool",
    ],
)
def test_builtin_default_constructors_ship_labels(name):
    import cayu.tools as tools

    tool = getattr(tools, name)()
    assert tool.spec.display is not None
    assert tool.spec.display.verb


def test_session_and_child_tools_keep_labels_when_renamed():
    from cayu.sessions import InMemorySessionStore
    from cayu.tools import (
        ChildSessionResultTool,
        ListSessionsTool,
        OwnTreeSessionAccess,
        ReadSessionTool,
        SubagentResultTool,
    )

    store = InMemorySessionStore()
    tools = [
        ListSessionsTool(store, access=OwnTreeSessionAccess(), name="custom_sessions"),
        ReadSessionTool(store, access=OwnTreeSessionAccess(), name="custom_transcript"),
        ChildSessionResultTool(store, name="child_result"),
        SubagentResultTool(store, name="delegate_result"),
    ]
    assert [tool.spec.display.verb for tool in tools] == [
        "Listed sessions",
        "Read",
        "Read child session results",
        "Read subagent results",
    ]
    assert tools[1].spec.display.noun == "session"


def test_remaining_builtin_constructor_labels():
    from tests.core.test_coding_toolchain_admission import _profile
    from tests.core.test_named_checks import _check, _policy
    from tests.core.test_shared_artifacts import _policy as artifact_policy
    from tests.core.test_web_access_routing import _bridge, _FetchAdapter
    from tests.core.test_web_search_tool import _SearchAdapter

    from cayu.tools import (
        MaterializeSharedArtifactTool,
        PublishWorkspaceArtifactTool,
        RunCheckTool,
        RunCommandTool,
        SubagentTool,
        WebAccessRoutePolicy,
        WebAccessRoutingTool,
        WebBridgeRoute,
        WebSearchTool,
    )
    from cayu.tools.discovery import SearchToolsTool

    app = CayuApp(enable_logging=False)
    try:
        tools = [
            RunCommandTool(toolchain_profile=_profile()),
            RunCheckTool(checks=[_check()], command_policy=_policy()),
            PublishWorkspaceArtifactTool(artifact_policy()),
            MaterializeSharedArtifactTool(artifact_policy()),
            SubagentTool(app, agents={"reader": "reader"}, name="renamed_delegate"),
            SearchToolsTool(),
            WebSearchTool(adapter=_SearchAdapter(ToolResult(content="unused"))),
            WebAccessRoutingTool(
                routes=[
                    WebBridgeRoute(
                        route_id="main", bridge=_bridge(_FetchAdapter(ToolResult(content="unused")))
                    )
                ],
                policy=WebAccessRoutePolicy(entry_route_id="main"),
            ),
        ]
        assert [tool.spec.display.verb for tool in tools] == [
            "Ran",
            "Ran",
            "Published",
            "Materialized",
            "Started",
            "Searched tools",
            "Searched the web",
            "Fetched",
        ]
    finally:
        asyncio.run(app.aclose())


@pytest.mark.parametrize("character", ["\x85", "\x9f", "\u2028", "\u2029"])
def test_display_single_line_contract_rejects_unicode_controls(character):
    with pytest.raises(ValueError):
        ToolSpec(name="custom", display={"verb": f"Read{character}files"})


def test_typed_display_snapshot_rejects_mutation_without_diagnostic_leaks(caplog, capsys):
    import warnings

    from cayu.tools import ToolDisplay
    from cayu.tools._execution_requirements import with_intrinsic_execution_requirements

    canary = "PRIVATE_DISPLAY_DIAGNOSTIC"

    class Hostile:
        def __repr__(self):
            return canary

        def __str__(self):
            return canary

    spec = ToolSpec(name="custom", display=ToolDisplay(verb="Read"))
    object.__setattr__(spec.display, "verb", Hostile())
    app = CayuApp(enable_logging=False)
    try:
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            for operation in (
                lambda: spec.model_copy(),
                lambda: with_intrinsic_execution_requirements(spec, ()),
                lambda: app.register_agent(
                    AgentSpec(name="reader", model="model"), tools=[DisplayTool(spec)]
                ),
            ):
                with pytest.raises(ValueError) as error:
                    operation()
                assert canary not in str(error.value)
                assert canary not in repr(error.value)
            assert not app.list_agents()
        captured = capsys.readouterr()
        assert canary not in captured.out + captured.err + caplog.text
        assert canary not in "".join(str(item.message) for item in recorded)
    finally:
        asyncio.run(app.aclose())


def test_display_redaction_replay_and_profile_identity_over_public_owners():
    from cayu import Message, RunRequest
    from cayu._application_registration import _registered_tool_descriptor
    from cayu.evals.runtime_replay import _RecordedTool, _RecordedToolTracker
    from cayu.vaults import SecretRedactor

    async def run():
        descriptors = []
        for label in (
            {"verb": "PRIVATE_LABEL", "noun": "PRIVATE_LABEL", "plural": "PRIVATE_LABEL"},
            None,
        ):
            app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor("PRIVATE_LABEL"))
            app.register_provider(ScriptedModelProvider([]), default=True)
            app.register_agent(
                AgentSpec(name="reader", model="scripted"),
                tools=[DisplayTool(ToolSpec(name="custom", display=label))],
            )
            try:
                request = RunRequest(agent_name="reader", messages=[Message.text("user", "hi")])
                fingerprint = await app.inspect_run_execution_profile(request)
                registered = app.get_agent("reader").tools["custom"]
                replay = _RecordedTool(
                    registered_tool=registered, tracker=_RecordedToolTracker(outcomes={})
                )
                assert replay.spec.display == registered.display
                assert "display" not in replay.schema
                descriptors.append(_registered_tool_descriptor(registered))
                if label:
                    object.__setattr__(registered.display, "verb", "Changed returned snapshot")
                    assert app.get_agent("reader").tools["custom"].display.verb == "PRIVATE_LABEL"
                    assert await app.inspect_run_execution_profile(request) == fingerprint
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(
                        app=create_server(app, config=ServerConfig.local_development())
                    ),
                    base_url="http://test",
                ) as client:
                    response = await client.get("/api/agents")
                    assert response.status_code == 200
                    display = response.json()["agents"][0]["tools"][0]["display"]
                    if label:
                        assert "PRIVATE_LABEL" not in response.text
                        assert all(
                            display[field] != "PRIVATE_LABEL"
                            for field in ("verb", "noun", "plural")
                        )
                    else:
                        assert display is None
            finally:
                await app.aclose()
        assert descriptors[0] == descriptors[1]

    asyncio.run(run())


def test_builtin_mcp_hosted_labels_and_capability_through_agents_http():
    from tests.core.test_mcp import FakeMcpSession, _fake_server_spec, _fake_tool_definitions

    from cayu.mcp import McpToolset
    from cayu.providers.hosted import OpenAIWebSearch
    from cayu.tools import OwnTreeSessionAccess, ReadFileTool, ReadSessionTool

    async def run():
        app = CayuApp(enable_logging=False)
        definitions = _fake_tool_definitions("read_file")
        toolset = McpToolset(
            server=_fake_server_spec().model_copy(update={"connection_id": "display-test"}),
            session=FakeMcpSession(definitions=definitions),
            definitions=definitions,
        )
        app.register_agent(
            AgentSpec(name="builtin", model="model"),
            tools=[
                ReadFileTool(),
                ReadSessionTool(
                    app.session_store, access=OwnTreeSessionAccess(), name="custom_session_reader"
                ),
            ],
        )
        app.register_agent(
            AgentSpec(name="external", model="model"),
            mcp_toolsets=[toolset],
            hosted_tools=[OpenAIWebSearch()],
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development())
                ),
                base_url="http://test",
            ) as client:
                response = await client.get("/api/agents")
                assert response.status_code == 200
                agents = {row["name"]: row for row in response.json()["agents"]}
                builtin = {row["name"]: row for row in agents["builtin"]["tools"]}
                assert builtin["read_file"]["display"] == {
                    "verb": "Read",
                    "noun": "file",
                    "plural": None,
                }
                assert builtin["custom_session_reader"]["display"] == {
                    "verb": "Read",
                    "noun": "session",
                    "plural": None,
                }
                external = agents["external"]["tools"]
                assert len(external) == 2
                assert {row["source"]["kind"] for row in external} == {"mcp", "provider"}
                assert all(row["display"] is None for row in external)
                contract = (await client.get("/api/contract")).json()
                capability = contract["capabilities"]["surfaces"]["tool_display"]
                assert capability["read"]["enabled"] is True
                assert capability["mutate"]["enabled"] is False
        finally:
            await app.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("secret", ["verb", "noun", "plural"])
@pytest.mark.parametrize("path", ["/api/agents", "/api/agents/reader"])
def test_display_redaction_preserves_protocol_keys_through_agents_http(secret, path):
    from cayu.tools import ReadFileTool
    from cayu.vaults import SecretRedactor

    async def run():
        canary = "PRIVATE_DISPLAY_VALUE"
        value = f"Read {secret} {canary}"
        app = CayuApp(enable_logging=False, secret_redactor=SecretRedactor([secret, canary]))
        app.register_agent(
            AgentSpec(name="reader", model="model"),
            tools=[
                ReadFileTool(),
                DisplayTool(ToolSpec(name="custom")),
                DisplayTool(ToolSpec(name="short", display={"verb": value})),
                DisplayTool(
                    ToolSpec(
                        name="complete",
                        display={"verb": value, "noun": value, "plural": value},
                    )
                ),
            ],
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(
                    app=create_server(app, config=ServerConfig.local_development()),
                    raise_app_exceptions=False,
                ),
                base_url="http://test",
            ) as client:
                response = await client.get(path)
                assert response.status_code == 200, response.text
                tools = {row["name"]: row for row in response.json()["agents"][0]["tools"]}
                assert tools["custom"]["display"] is None
                assert tools["read_file"]["display"] == {
                    "verb": "Read",
                    "noun": "file",
                    "plural": None,
                }
                for name in ("short", "complete"):
                    display = tools[name]["display"]
                    assert set(display) == {"verb", "noun", "plural"}
                    for text in display.values():
                        if text is not None:
                            assert secret not in text
                            assert canary not in text
                            assert text == "Read [REDACTED_SECRET] [REDACTED_SECRET]"
                assert tools["short"]["display"]["noun"] is None
                assert tools["short"]["display"]["plural"] is None
                assert canary not in response.text
        finally:
            await app.aclose()

    asyncio.run(run())
