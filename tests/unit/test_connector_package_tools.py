"""Connector package tools (`cayu.connections.tools`): each catalog source over a mock transport,
startup retries and failures, naming, slot declarations, the approval floor and each run."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from tests.connector_broker_support import context, policy_request
from tests.core._execution_profile_fixtures import profiled_session_identity

from cayu import AgentSpec, CayuApp
from cayu.cli import _release_check
from cayu.cli import cloud as cloud_cli
from cayu.cli._cloud_api import CloudApiClient, CloudApiError
from cayu.cli._cloud_auth import CloudAuthError
from cayu.cloud_inventory import CloudToolSource, cloud_registration
from cayu.connections import (
    CONNECTOR_TOOLS_API,
    ConnectionSlot,
    ConnectorPackageTool,
    ConnectorToolsError,
    _tool_catalog,
    tools,
)
from cayu.connectors import slack
from cayu.releases import ReleaseIdentity
from cayu.runtime import check_manifest
from cayu.runtime._execution_profile_admission import _cayu_policy_material
from cayu.tools import (
    ApprovalFloorToolPolicy,
    StaticToolPolicy,
    Tool,
    ToolEffect,
    ToolPolicyDecision,
    ToolResult,
)

AGENT_URL = "https://cloud.example/v1/agent-connections"
MEMBER_URL = "https://cloud.example"
_POST = "Post a message to {channel}."


def _descriptor(name: str, effect: str = "none", **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "description": f"The {name} tool.",
        "input_schema": {
            "type": "object",
            "properties": {"channel": {"type": "string"}, "text": {"type": "string"}},
            "additionalProperties": False,
        },
        "effect": effect,
        "scopes": [],
        "approval": None,
        "max_answer_bytes": 60 * 1024,
        "digest": digest(name),
        **extra,
    }


def digest(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def catalog(toolset: str = "sample", **extra: Any) -> dict[str, Any]:
    return {
        "api": CONNECTOR_TOOLS_API,
        "toolset": toolset,
        "package": f"example-connector-{toolset}",
        "version": "0.1.0",
        "connectors": [toolset, f"{toolset}-sandbox"],
        "default_slot": toolset,
        "description": "Sample items and messages",
        "tools": [
            _descriptor(f"{toolset}_list_items", scopes=["items:read"]),
            _descriptor(
                f"{toolset}_post_message", "external", scopes=["messages:write"], approval=_POST
            ),
        ],
        **extra,
    }


class FakeCloud:
    """Cayu Cloud's connector tools routes: each answer by (method, path), or a function."""

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        answer = self.routes[(request.method, request.url.path)]
        return answer(request) if callable(answer) else answer

    def sent(self, method: str = "POST") -> list[dict[str, Any]]:
        return [
            json.loads(request.content) for request in self.requests if request.method == method
        ]


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Callable[..., FakeCloud]]:
    """Install a fake Cayu Cloud; the process starts in an empty project, outside Cloud,
    signed in nowhere, with its own cache directory and a clock the retries move."""

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("[project]\nname = 'sample'\n")
    monkeypatch.chdir(project)
    monkeypatch.delenv("CAYU_CONNECTIONS_URL", raising=False)
    monkeypatch.delenv("CAYU_CONNECTIONS_TOKEN", raising=False)
    monkeypatch.setattr(_release_check, "user_cache_directory", lambda: tmp_path / "cache")
    monkeypatch.setattr(cloud_cli, "member_client", lambda **_: None)
    monkeypatch.setattr(_tool_catalog, "_read", {})
    now = [0.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(_tool_catalog, "_sleep", sleep)
    monkeypatch.setattr(_tool_catalog, "_clock", lambda: now[0])

    def install(routes: dict[tuple[str, str], Any]) -> FakeCloud:
        fake = FakeCloud(routes)
        monkeypatch.setattr(_tool_catalog, "_transport", httpx.MockTransport(fake))
        return fake

    install.slept = slept  # type: ignore[attr-defined]
    yield install


@pytest.fixture
def in_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAYU_CONNECTIONS_URL", AGENT_URL)
    monkeypatch.setenv("CAYU_CONNECTIONS_TOKEN", "cayu_conn_test")
    monkeypatch.setenv("CAYU_CLOUD_APPLICATION_ID", "app-1")


def _agent_catalog(document: dict[str, Any] | None = None) -> dict[tuple[str, str], Any]:
    return {
        ("GET", "/v1/agent-connections/toolsets/sample"): httpx.Response(
            200, json=document or catalog()
        )
    }


def _run(tool: Tool, **arguments: Any) -> ToolResult:
    return asyncio.run(tool.run(context(metadata={"tool_call_id": "call_1"}), arguments))


def _by_name(loaded: list[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in loaded}


def test_an_agent_reads_its_catalog_and_runs_each_tool_on_its_binding(cloud, in_cloud) -> None:
    fake = cloud(
        {
            **_agent_catalog(),
            ("POST", "/v1/agent-connections/sample/tools/sample_list_items"): httpx.Response(
                200,
                json={
                    "result": {
                        "content": '{"items":[]}',
                        "structured": {"items": []},
                        "truncated": False,
                    }
                },
            ),
        }
    )

    loaded = _by_name(tools("sample"))

    assert list(loaded) == ["sample_list_items"]
    listing = loaded["sample_list_items"]
    assert isinstance(listing, ConnectorPackageTool)
    assert listing.spec.effect is ToolEffect.NONE
    assert listing.approval_reason is None
    assert listing.connection_slots == (
        ConnectionSlot(
            name="sample",
            connectors=("sample", "sample-sandbox"),
            scopes=("items:read",),
            description="Sample items and messages",
        ),
    )
    assert _run(listing) == ToolResult(content='{"items":[]}', structured={"items": []})
    catalog_read, run = fake.requests
    assert catalog_read.headers["authorization"] == "Bearer cayu_conn_test"
    assert run.headers["authorization"] == "Bearer cayu_conn_test"
    assert json.loads(run.content) == {
        "arguments": {},
        "digest": digest("sample_list_items"),
        "call_id": "call_1",
    }
    # A process reads a toolset's catalog once, and an Agent never keeps it on disk.
    tools("sample", writes=True)
    assert len(fake.requests) == 2
    assert _tool_catalog.cache_path("sample") is not None
    assert not _tool_catalog.cache_path("sample").exists()  # type: ignore[union-attr]
    assert _tool_catalog.summary() == (
        f"Connector tools: sample 0.1.0 from {AGENT_URL}, Agent app-1"
    )


def test_a_start_retries_an_unreachable_cloud_then_serves(cloud, in_cloud) -> None:
    answers = iter(
        [httpx.ConnectError("down"), httpx.Response(503), httpx.Response(200, json=catalog())]
    )

    def answer(request: httpx.Request) -> httpx.Response:
        given = next(answers)
        if isinstance(given, Exception):
            raise given
        return given

    cloud({("GET", "/v1/agent-connections/toolsets/sample"): answer})

    assert [tool.name for tool in tools("sample")] == ["sample_list_items"]
    assert cloud.slept == [0.5, 1.0]


def test_a_start_fails_after_twenty_seconds_without_cloud(cloud, in_cloud) -> None:
    cloud({("GET", "/v1/agent-connections/toolsets/sample"): httpx.Response(502)})

    with pytest.raises(ConnectorToolsError) as raised:
        tools("sample")

    assert raised.value.code == "connector_catalog_unreachable"
    assert AGENT_URL in str(raised.value)
    assert cloud.slept == [0.5, 1.0, 2.0, 4.0, 8.0, 4.5]
    assert sum(cloud.slept) == _tool_catalog.STARTUP_RETRY_SECONDS


def test_a_refused_or_malformed_catalog_fails_at_once(cloud, in_cloud) -> None:
    cloud(
        {
            ("GET", "/v1/agent-connections/toolsets/sample"): httpx.Response(
                401, json={"detail": {"code": "agent_credential_invalid", "message": "No."}}
            )
        }
    )
    with pytest.raises(ConnectorToolsError) as refused:
        tools("sample")
    assert refused.value.code == "agent_credential_invalid"
    assert cloud.slept == []

    _tool_catalog._read.clear()
    cloud(_agent_catalog(catalog(api="cayu.connector-tools/v2")))
    with pytest.raises(ConnectorToolsError) as invalid:
        tools("sample")
    assert invalid.value.code == "connector_catalog_invalid"

    unapproved = catalog()
    unapproved["tools"][1]["approval"] = None
    _tool_catalog._read.clear()
    cloud(_agent_catalog(unapproved))
    with pytest.raises(ConnectorToolsError, match="has an effect and no approval"):
        tools("sample", writes=True)


def test_a_toolset_cloud_no_longer_offers_starts_without_tools_and_declares_nothing(
    cloud, in_cloud, caplog: pytest.LogCaptureFixture
) -> None:
    cloud(
        {
            ("GET", "/v1/agent-connections/toolsets/sample"): httpx.Response(
                404, json={"detail": {"code": "connector_unavailable", "message": "Gone."}}
            )
        }
    )

    with caplog.at_level(logging.WARNING, logger="cayu.connections"):
        loaded = tools("sample", slot="sample-eu")

    assert type(loaded) is list
    assert loaded == []
    assert "connector_tools_unavailable" in caplog.text
    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=loaded)
    assert app.connection_slots() == ()
    assert _tool_catalog.summary() == (
        f"Connector tools: sample (unavailable) from {AGENT_URL}, Agent app-1"
    )


def test_a_second_slot_renames_the_tools_and_runs_on_it(cloud, in_cloud) -> None:
    fake = cloud(
        {
            **_agent_catalog(),
            ("POST", "/v1/agent-connections/sample-eu/tools/sample_post_message"): httpx.Response(
                200, json={"result": {"content": "Posted.", "structured": None}}
            ),
        }
    )

    loaded = _by_name(tools("sample", slot="sample-eu", writes=True))

    assert set(loaded) == {"sample_eu_list_items", "sample_eu_post_message"}
    post = loaded["sample_eu_post_message"]
    assert post.connection_slots[0].name == "sample-eu"
    assert post.connection_slots[0].scopes == ("items:read", "messages:write")
    assert _run(post, channel="C1", text="hi").content == "Posted."
    assert fake.sent()[0]["digest"] == digest("sample_post_message")
    with pytest.raises(ValueError, match="longer than 64"):
        tools("sample", slot="s" * 60, writes=True)


def test_a_toolset_slot_merges_with_a_toolkits(cloud, in_cloud) -> None:
    document = catalog("slack")
    document["connectors"] = ["slack"]
    document["tools"] = [_descriptor("slack_search_messages", scopes=["search:read"])]
    cloud({("GET", "/v1/agent-connections/toolsets/slack"): httpx.Response(200, json=document)})

    app = CayuApp(enable_logging=False)
    app.register_agent(
        AgentSpec(name="assistant", model="fake"),
        tools=[*slack.tools(), *tools("slack")],
    )

    (declared,) = app.connection_slots()
    assert declared.name == "slack"
    assert declared.connectors == ("slack",)
    assert {"search:read", "channels:read"} <= set(declared.scopes)


def _described(loaded: list[Tool], policy: Any = None) -> tuple[CayuApp, dict[str, Any], dict]:
    app = CayuApp(release_identity=ReleaseIdentity(id="release-a"), enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=loaded, tool_policy=policy)
    manifest = app.describe()
    flagged = {
        str(diagnostic.parameters["tool"]): diagnostic.code
        for diagnostic in check_manifest(manifest, deploy_only=True).diagnostics
        if diagnostic.code.startswith("EXTERNAL_TOOL_")
    }
    return app, {tool.name: tool for tool in manifest.agents[0].tools}, flagged


def test_a_write_asks_for_approval_itself_and_cayu_check_sees_it(cloud, in_cloud) -> None:
    cloud(_agent_catalog())
    loaded = tools("sample", writes=True)

    app, entries, flagged = _described(loaded)

    assert flagged == {}
    assert entries["sample_post_message"].policy_coverage == "approval_required"
    assert entries["sample_list_items"].policy_coverage == "allowed"
    policy = app._get_registered_agent("assistant").tool_policy
    decided = asyncio.run(
        policy.authorize(policy_request("sample_post_message", channel="#ops\nnow", text="x"))
    )
    assert decided.decision is ToolPolicyDecision.REQUIRE_APPROVAL
    assert decided.reason == "Post a message to #opsnow."
    # The Agent's own policy stays in charge: what it denies stays denied.
    _, denied, _ = _described(loaded, StaticToolPolicy(deny=["sample_post_message"]))
    assert denied["sample_post_message"].policy_coverage == "denied"


def test_cayu_check_shows_a_package_writes_floor_as_a_toolkits_and_a_stable_identity(
    cloud, in_cloud
) -> None:
    cloud(_agent_catalog())
    package = tools("sample", writes=True)
    toolkit = slack.tools(include_writes=True)
    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="package", model="fake"), tools=package)
    app.register_agent(
        AgentSpec(name="toolkit", model="fake"),
        tools=toolkit,
        tool_policy=slack.approval_policy(),
    )
    manifest = app.describe()
    entries = {(agent.name, tool.name): tool for agent in manifest.agents for tool in agent.tools}
    findings = check_manifest(manifest).diagnostics

    write = entries[("package", "sample_post_message")]
    toolkit_write = entries[("toolkit", "slack_post_message")]
    assert write.policy_coverage == toolkit_write.policy_coverage == "approval_required"
    assert [
        finding.subject
        for finding in findings
        if finding.code == "EVAL_EXECUTION_IDENTITY_PROCESS_LOCAL"
        and finding.subject.startswith("tool:package/")
    ] == []
    assert not [
        finding
        for finding in check_manifest(manifest, deploy_only=True).diagnostics
        if finding.code.startswith("EXTERNAL_TOOL_")
        and str(finding.parameters.get("tool")).startswith("sample_")
    ]
    identity = package[1].execution_profile_identity
    assert identity is not None
    assert (
        identity.name == "cayu.connector-tools/example-connector-sample/sample/sample_post_message"
    )
    assert identity.behavior_version == digest("sample_post_message")


def test_an_identity_follows_the_digest_not_the_version(cloud, in_cloud) -> None:
    newer = catalog(version="0.2.0")
    newer["tools"][0]["description"] = "A clearer description."
    cloud(_agent_catalog(newer))
    (fixed,) = tools("sample")
    _tool_catalog._read.clear()
    changed = catalog()
    changed["tools"][0]["digest"] = digest("other")
    cloud(_agent_catalog(changed))
    (rescoped,) = tools("sample")
    _tool_catalog._read.clear()
    cloud(_agent_catalog())
    (original,) = tools("sample")

    assert fixed.execution_profile_identity == original.execution_profile_identity
    assert rescoped.execution_profile_identity != original.execution_profile_identity


def test_a_tool_reports_its_package_as_its_source(cloud, in_cloud) -> None:
    cloud(_agent_catalog())
    app, _, _ = _described(tools("sample"))

    registration = asyncio.run(cloud_registration(app))

    (listed,) = registration.tools
    assert listed.source == CloudToolSource(
        kind="package", package="example-connector-sample", version="0.1.0"
    )


def test_each_refusal_and_tool_error_is_a_tool_error(
    cloud, in_cloud, caplog: pytest.LogCaptureFixture
) -> None:
    route = "/v1/agent-connections/sample/tools/sample_post_message"
    answers = iter(
        [
            httpx.Response(409, json={"detail": {"code": "tool_changed", "message": "Changed."}}),
            httpx.Response(409, json={"detail": {"code": "tool_changed", "message": "Changed."}}),
            httpx.Response(
                404, json={"detail": {"code": "connection_unbound", "message": "Unbound."}}
            ),
            httpx.Response(
                200,
                json={
                    "error": {
                        "code": "invalid_arguments",
                        "message": "channel is required.",
                        "advice": "Give the channel.",
                        "retryable": False,
                        "retry_after_seconds": None,
                        "ambiguous": False,
                        "path": "$.channel",
                        "schema_path": "$.required",
                    }
                },
            ),
            httpx.Response(502, json={"detail": {"code": "tool_timed_out", "message": "Late."}}),
            httpx.Response(
                200,
                json={
                    "error": {
                        "code": "upstream_failed",
                        "message": "Lost.",
                        "advice": "Retry.",
                        "ambiguous": True,
                    }
                },
            ),
        ]
    )
    cloud({**_agent_catalog(), ("POST", route): lambda request: next(answers)})
    post = _by_name(tools("sample", writes=True))["sample_post_message"]

    with caplog.at_level(logging.WARNING, logger="cayu.connections"):
        changed = [_run(post, channel="C1"), _run(post, channel="C1")]
    unbound, invalid, late, partial = (_run(post, channel="C1") for _ in range(4))

    assert {result.structured["error"] for result in changed} == {"tool_changed"}  # type: ignore[index]
    assert "restart the Agent" in changed[0].content
    assert caplog.text.count("tool_changed") == 1
    assert unbound.structured["error"] == "connection_unbound"  # type: ignore[index]
    assert "An admin must bind the slot sample" in unbound.content
    assert invalid.is_error
    assert invalid.structured == {
        "error": "invalid_arguments",
        "message": "channel is required. (at $.channel)",
        "advice": "Give the channel.",
        "retryable": False,
        "ambiguous": False,
        "path": "$.channel",
        "schema_path": "$.required",
    }
    assert "$.channel" in invalid.content
    # A write that timed out may have taken effect: never advised to be retried unchecked.
    assert late.structured["ambiguous"] is True  # type: ignore[index]
    assert "check whether it did" in late.content
    # A write's own error marked ambiguous is never advised to be retried either.
    assert partial.structured["ambiguous"] is True  # type: ignore[index]
    assert "Retry." not in partial.content
    assert "check whether it did" in partial.content


def test_a_reads_timeout_is_temporary_and_an_unknown_agent_says_how_to_link(
    cloud, in_cloud
) -> None:
    route = "/v1/agent-connections/sample/tools/sample_list_items"
    answers = iter(
        [
            httpx.Response(502, json={"detail": {"code": "tool_timed_out", "message": "Late."}}),
            httpx.Response(404, json={"detail": {"code": "agent_not_found", "message": "No."}}),
        ]
    )
    cloud({**_agent_catalog(), ("POST", route): lambda request: next(answers)})
    listing = _by_name(tools("sample"))["sample_list_items"]

    late, unknown = _run(listing), _run(listing)

    assert "ambiguous" not in late.structured  # type: ignore[operator]
    assert "retry later" in late.content
    assert "cayu-cloud.toml" in unknown.content
    assert "ambiguous" not in unknown.structured  # type: ignore[operator]


def test_a_call_id_cloud_cant_keep_is_replaced(cloud, in_cloud) -> None:
    fake = cloud(
        {
            **_agent_catalog(),
            ("POST", "/v1/agent-connections/sample/tools/sample_list_items"): httpx.Response(
                200, json={"result": {"content": "[]"}}
            ),
        }
    )
    listing = _by_name(tools("sample"))["sample_list_items"]

    asyncio.run(listing.run(context(metadata={"tool_call_id": "call with spaces"}), {}))

    sent = fake.sent()[0]["call_id"]
    assert sent != "call with spaces"
    assert len(sent) == 36 and " " not in sent


def test_tools_load_inside_a_running_loop_and_refuse_a_slot_for_several_toolsets(
    cloud, in_cloud
) -> None:
    fake = cloud(_agent_catalog())

    async def inside() -> list[Tool]:
        return tools("sample")

    assert [tool.name for tool in asyncio.run(inside())] == ["sample_list_items"]
    assert len(fake.requests) == 1
    with pytest.raises(ValueError, match="one toolset"):
        tools("sample", "other", slot="sample-eu")


def test_a_snapshot_loads_inside_a_running_loop(cloud, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CAYU_CONNECTOR_TOOLS", "snapshot")
    snapshot = Path(".cayu/connector-tools/sample.json")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(catalog()))

    async def inside() -> list[Tool]:
        return tools("sample")

    assert [tool.name for tool in asyncio.run(inside())] == ["sample_list_items"]


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({"approval": "Post to {room}."}, "isn't an argument"),
        ({"approval": "x" * 513}, "over 512"),
        ({"description": "x" * 1025}, "description is over"),
        ({"digest": "sha256:abc"}, "SHA-256"),
        (
            {"input_schema": {"type": "object", "properties": {"channel": {"type": "string"}}}},
            "additionalProperties",
        ),
    ],
)
def test_a_tool_cloud_describes_out_of_bounds_is_refused(
    cloud, in_cloud, change: dict[str, Any], refusal: str
) -> None:
    document = catalog()
    document["tools"][1].update(change)
    cloud(_agent_catalog(document))

    with pytest.raises(ConnectorToolsError, match=refusal):
        tools("sample", writes=True)


def test_an_approval_keeps_literal_braces(cloud, in_cloud) -> None:
    document = catalog()
    document["tools"][1]["approval"] = "Post {{json}} to {channel}."
    cloud(_agent_catalog(document))
    post = _by_name(tools("sample", writes=True))["sample_post_message"]

    assert post.approval_reason is not None and not isinstance(post.approval_reason, str)
    assert post.approval_reason({"channel": "C1"}) == "Post {json} to C1."


def test_a_catalog_refused_without_a_code_and_a_retry_after(cloud, in_cloud) -> None:
    answers = iter(
        [
            httpx.Response(429, headers={"retry-after": "3"}),
            httpx.Response(200, json=catalog()),
        ]
    )
    cloud({("GET", "/v1/agent-connections/toolsets/sample"): lambda request: next(answers)})
    assert [tool.name for tool in tools("sample")] == ["sample_list_items"]
    assert cloud.slept == [3.0]

    _tool_catalog._read.clear()
    cloud({("GET", "/v1/agent-connections/toolsets/sample"): httpx.Response(403)})
    with pytest.raises(ConnectorToolsError) as refused:
        tools("sample")
    assert refused.value.code == "connector_catalog_refused"


@pytest.fixture
def signed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cloud_cli,
        "member_client",
        lambda **_: CloudApiClient(api_url=MEMBER_URL, api_key="login-token"),
    )
    monkeypatch.setattr(
        cloud_cli, "_listed_agents", lambda client: [{"id": "app-1", "name": "Sample Agent"}]
    )


def test_a_signed_in_developer_runs_the_linked_agents_tools_as_a_member(
    cloud, signed_in, tmp_path: Path
) -> None:
    Path("cayu-cloud.toml").write_text('application = "sample-agent"\n')
    fake = cloud(
        {
            ("GET", "/v1/connector-tools/sample"): httpx.Response(200, json=catalog()),
            ("POST", "/v1/applications/app-1/connections/sample/tools/sample_list_items"): (
                httpx.Response(200, json={"result": {"content": "[]", "structured": None}})
            ),
        }
    )

    listing = _by_name(tools("sample"))["sample_list_items"]

    assert _run(listing).content == "[]"
    assert [request.headers["authorization"] for request in fake.requests] == [
        "Bearer login-token",
        "Bearer login-token",
    ]
    cached = tmp_path / "cache" / "connector-tools" / "sample.json"
    assert json.loads(cached.read_text()) == catalog()
    assert not Path(".cayu").exists()
    assert _tool_catalog.summary() == (
        f"Connector tools: sample 0.1.0 from {MEMBER_URL}, Agent sample-agent"
    )


def test_a_member_falls_back_to_the_snapshot_while_cloud_is_unreachable(
    cloud, signed_in, caplog: pytest.LogCaptureFixture
) -> None:
    Path("cayu-cloud.toml").write_text('application = "sample-agent"\n')
    snapshot = Path(".cayu/connector-tools/sample.json")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(catalog()))
    fake = cloud(
        {
            ("GET", "/v1/connector-tools/sample"): httpx.Response(503),
            ("POST", "/v1/applications/app-1/connections/sample/tools/sample_list_items"): (
                httpx.Response(200, json={"result": {"content": "[]"}})
            ),
        }
    )

    with caplog.at_level(logging.WARNING, logger="cayu.connections"):
        listing = _by_name(tools("sample"))["sample_list_items"]

    assert cloud.slept == []
    assert "saved catalog" in caplog.text and str(snapshot.resolve()) in caplog.text
    assert _run(listing).content == "[]"
    assert len(fake.requests) == 2
    assert "was unreachable" in (_tool_catalog.summary() or "")


def test_a_member_without_a_saved_catalog_fails_while_cloud_is_unreachable(
    cloud, signed_in
) -> None:
    cloud({("GET", "/v1/connector-tools/sample"): httpx.Response(503)})

    with pytest.raises(ConnectorToolsError) as raised:
        tools("sample")

    assert raised.value.code == "connector_catalog_unreachable"
    assert cloud.slept == []


def test_a_login_that_cant_refresh_is_a_tool_error(cloud, monkeypatch: pytest.MonkeyPatch) -> None:
    Path("cayu-cloud.toml").write_text('application = "sample-agent"\n')
    calls = []

    def token() -> str:
        calls.append(1)
        if len(calls) > 1:
            raise CloudAuthError("cloud_auth_unavailable", "Could not read the login.")
        return "login-token"

    monkeypatch.setattr(
        cloud_cli,
        "member_client",
        lambda **_: CloudApiClient(
            api_url=MEMBER_URL, api_key="login-token", api_key_provider=token
        ),
    )
    monkeypatch.setattr(
        cloud_cli, "_listed_agents", lambda client: [{"id": "app-1", "name": "Sample Agent"}]
    )
    cloud({("GET", "/v1/connector-tools/sample"): httpx.Response(200, json=catalog())})

    result = _run(_by_name(tools("sample"))["sample_list_items"])

    assert result.structured["error"] == "cloud_login_required"  # type: ignore[index]
    assert "cayu cloud login" in result.content


def test_an_unlinked_project_loads_the_catalog_and_its_runs_say_how_to_bind(
    cloud, signed_in
) -> None:
    fake = cloud({("GET", "/v1/connector-tools/sample"): httpx.Response(200, json=catalog())})

    result = _run(_by_name(tools("sample"))["sample_list_items"])

    assert result.structured["error"] == "connection_unbound"  # type: ignore[index]
    assert "cayu cloud connections bind sample CONNECTION --application AGENT" in result.content
    assert len(fake.requests) == 1


def test_the_snapshot_switch_pins_a_process_to_saved_catalogs(
    cloud, in_cloud, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = cloud({})
    monkeypatch.setenv("CAYU_CONNECTOR_TOOLS", "snapshot")
    with pytest.raises(ValueError, match="--save"):
        tools("sample")

    snapshot = Path(".cayu/connector-tools/sample.json")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(catalog()))
    listing = _by_name(tools("sample"))["sample_list_items"]

    assert fake.requests == []
    offline = _run(listing)
    assert offline.structured["error"] == "no_connection_client"  # type: ignore[index]
    assert "CAYU_CONNECTOR_TOOLS=snapshot" in offline.content
    assert "LocalConnections" not in offline.content
    monkeypatch.setenv("CAYU_CONNECTOR_TOOLS", "online")
    _tool_catalog._read.clear()
    with pytest.raises(ValueError, match="CAYU_CONNECTOR_TOOLS"):
        tools("sample")


def test_offline_a_saved_snapshot_then_the_cache_declare_tools_that_cant_run(
    cloud, tmp_path: Path
) -> None:
    cloud({})
    with pytest.raises(ValueError) as nothing:
        tools("sample")
    for way in ("CAYU_CONNECTIONS_URL", "cayu cloud login", "--save"):
        assert way in str(nothing.value)

    cached = tmp_path / "cache" / "connector-tools" / "sample.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps(catalog()))
    assert [tool.name for tool in tools("sample")] == ["sample_list_items"]

    _tool_catalog._read.clear()
    saved = catalog()
    saved["version"] = "0.2.0"
    snapshot = Path(".cayu/connector-tools/sample.json")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(saved))
    listing = _by_name(tools("sample"))["sample_list_items"]

    assert isinstance(listing, ConnectorPackageTool) and listing.version == "0.2.0"
    assert _run(listing).structured["error"] == "no_connection_client"  # type: ignore[index]
    assert "offline" in (_tool_catalog.summary() or "")


def _profile(loaded: list[Tool], policy: Any = None) -> Any:
    """The agent's execution profile in a fresh application: a restarted process."""

    app = CayuApp(enable_logging=False)
    app.register_agent(AgentSpec(name="assistant", model="fake"), tools=loaded, tool_policy=policy)
    return profiled_session_identity(provider_name="fake", model="fake", app=app).execution_profile


def test_writes_keep_the_agents_execution_profile_across_a_restart(cloud, in_cloud) -> None:
    cloud(_agent_catalog())
    first = _profile(tools("sample", writes=True))
    _tool_catalog._read.clear()
    restarted = _profile(tools("sample", writes=True))
    _tool_catalog._read.clear()
    changed = catalog()
    changed["tools"][1]["approval"] = "Send a message to {channel}."
    cloud(_agent_catalog(changed))
    reworded = _profile(tools("sample", writes=True))

    assert first == restarted
    assert reworded != first
    # The floor is part of the profile: the same tools without it differ.
    unfloored = tools("sample", writes=True)
    for tool in unfloored:
        tool.approval_reason = None
    assert _profile(unfloored) != reworded


def test_the_floor_composes_its_reasons_and_inner_policy_into_stable_material() -> None:
    floor = ApprovalFloorToolPolicy({"send": "Send it."}, StaticToolPolicy(deny=["drop"]))

    assert _cayu_policy_material(floor) == _cayu_policy_material(
        ApprovalFloorToolPolicy({"send": "Send it."}, StaticToolPolicy(deny=["drop"]))
    )
    for changed in (
        ApprovalFloorToolPolicy({"send": "Send that."}, StaticToolPolicy(deny=["drop"])),
        ApprovalFloorToolPolicy({"send": "Send it."}, StaticToolPolicy(deny=["other"])),
        ApprovalFloorToolPolicy({"post": "Send it."}, StaticToolPolicy(deny=["drop"])),
    ):
        assert _cayu_policy_material(changed) != _cayu_policy_material(floor)
    assert _cayu_policy_material(ApprovalFloorToolPolicy({"send": lambda _: "x"})) is None


_REAL_MEMBER_CLIENT = cloud_cli.member_client


@pytest.fixture
def unusable_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """The real member client, configured with an API key file that can't be read yet."""

    monkeypatch.setattr(cloud_cli, "member_client", _REAL_MEMBER_CLIENT)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)
    monkeypatch.delenv("CAYU_CLOUD_API_KEY", raising=False)
    monkeypatch.setenv("CAYU_CLOUD_API_URL", MEMBER_URL)
    key = tmp_path / "missing-key"
    monkeypatch.setenv("CAYU_CLOUD_API_KEY_FILE", str(key))
    monkeypatch.setattr(
        cloud_cli, "_listed_agents", lambda client: [{"id": "app-1", "name": "Sample Agent"}]
    )
    return key


def test_an_unusable_sign_in_serves_the_saved_catalog_and_signs_in_when_a_tool_runs(
    cloud, unusable_key: Path, caplog: pytest.LogCaptureFixture
) -> None:
    Path("cayu-cloud.toml").write_text('application = "sample-agent"\n')
    snapshot = Path(".cayu/connector-tools/sample.json")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(json.dumps(catalog()))
    fake = cloud(
        {
            ("POST", "/v1/applications/app-1/connections/sample/tools/sample_list_items"): (
                httpx.Response(200, json={"result": {"content": "[]"}})
            ),
        }
    )

    with caplog.at_level(logging.WARNING, logger="cayu.connections"):
        listing = _by_name(tools("sample"))["sample_list_items"]

    assert "api_key_unavailable" in caplog.text and "saved catalog" in caplog.text
    assert "sign-in can't be used" in (_tool_catalog.summary() or "")
    offline = _run(listing)
    assert offline.structured["error"] == "cloud_login_required"  # type: ignore[index]
    assert fake.requests == []
    # Back online: the same tool signs in and runs.
    unusable_key.write_text("cayu_example_key\n")
    assert _run(listing).content == "[]"
    assert fake.requests[0].headers["authorization"] == "Bearer cayu_example_key"


def test_an_unusable_sign_in_without_a_saved_catalog_fails(cloud, unusable_key: Path) -> None:
    cloud({})

    with pytest.raises(CloudApiError):
        tools("sample")


def test_a_login_that_cant_refresh_at_start_serves_the_cached_catalog(
    cloud, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cloud_cli, "member_client", _REAL_MEMBER_CLIENT)
    monkeypatch.setenv("CAYU_CLOUD_CONFIG", str(tmp_path / "cloud-config.json"))
    monkeypatch.delenv("CAYU_CLOUD_CONTEXT", raising=False)

    def expired(*args: Any, **kwargs: Any) -> CloudApiClient:
        raise CloudAuthError("login_refresh_unavailable", "Sign in again.")

    monkeypatch.setattr(cloud_cli, "_cloud_client", expired)
    Path("cayu-cloud.toml").write_text('application = "sample-agent"\n')
    cached = tmp_path / "cache" / "connector-tools" / "sample.json"
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps(catalog()))
    cloud({})

    listing = _by_name(tools("sample"))["sample_list_items"]

    assert _run(listing).structured["error"] == "cloud_login_required"  # type: ignore[index]


@pytest.mark.parametrize(
    ("agents", "code"),
    [
        ([], "agent_not_found"),
        (
            [{"id": "app-1", "name": "Sample Agent"}, {"id": "app-2", "name": "Sample Agent"}],
            "agent_ambiguous",
        ),
    ],
)
def test_an_agent_cloud_cant_name_once_says_how_to_link_it(
    cloud, signed_in, monkeypatch: pytest.MonkeyPatch, agents: list[Any], code: str
) -> None:
    Path("cayu-cloud.toml").write_text('application = "sample-agent"\n')
    monkeypatch.setattr(cloud_cli, "_listed_agents", lambda client: agents)
    cloud({("GET", "/v1/connector-tools/sample"): httpx.Response(200, json=catalog())})

    result = _run(_by_name(tools("sample"))["sample_list_items"])

    assert result.structured["error"] == code  # type: ignore[index]
    assert "cayu-cloud.toml" in result.content
