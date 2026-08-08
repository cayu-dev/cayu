from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

import cayu.mcp.http as mcp_http_module
from cayu import (
    Environment,
    EnvironmentSpec,
    HttpMcpClient,
    McpClient,
    McpInitializeResult,
    McpResourceDefinition,
    McpResourceResult,
    McpServerSpec,
    McpSession,
    McpToolDefinition,
    McpToolResult,
    SecretRef,
    StdioMcpClient,
    connect_mcp_toolset,
    extract_durable_value_error,
    mcp_tool_manifest_identity,
)
from cayu._validation import MAX_DURABLE_JSON_INTEGER, MIN_DURABLE_JSON_INTEGER

_SENSITIVE_CANARY = "private-mcp-value-canary"


class _ForbiddenSecretResolver:
    def __init__(self) -> None:
        self.called = False

    async def resolve(
        self,
        ref: SecretRef,
        *,
        scope: dict[str, Any] | None = None,
    ) -> Any:
        del ref, scope
        self.called = True
        raise AssertionError("secret resolution must not run for invalid MCP configuration")


class _ForbiddenMcpClient(McpClient):
    def __init__(self) -> None:
        self.calls = 0

    async def connect(self, server: McpServerSpec) -> McpSession:
        del server
        self.calls += 1
        raise AssertionError("client connection must not run for invalid MCP configuration")


class _SnapshotMcpSession(McpSession):
    def __init__(self) -> None:
        self.connected_server: McpServerSpec | None = None
        self.closed = False
        self._initialize_result = McpInitializeResult(protocol_version="2025-06-18")
        self._definitions = (McpToolDefinition(name="echo", input_schema={"type": "object"}),)

    @property
    def initialize_result(self) -> McpInitializeResult:
        return self._initialize_result

    async def list_tools(self) -> tuple[McpToolDefinition, ...]:
        return self._definitions

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> McpToolResult:
        del name, arguments
        return McpToolResult()

    async def list_resources(self) -> tuple[McpResourceDefinition, ...]:
        return ()

    async def read_resource(self, uri: str) -> McpResourceResult:
        del uri
        return McpResourceResult()

    async def close(self) -> None:
        self.closed = True


class _BarrierMcpClient(McpClient):
    def __init__(self, session: _SnapshotMcpSession) -> None:
        self.session = session
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.received_server: McpServerSpec | None = None

    async def connect(self, server: McpServerSpec) -> McpSession:
        self.received_server = server
        self.session.connected_server = server.model_copy(deep=True)
        self.started.set()
        await self.release.wait()
        return self.session


@dataclass(frozen=True)
class _McpJsonFieldCase:
    name: str
    capture: Callable[[Any], Any]


@dataclass(frozen=True)
class _McpTextFieldCase:
    name: str
    capture: Callable[[str], str]


_JSON_FIELD_CASES = (
    _McpJsonFieldCase(
        "server.metadata",
        lambda value: McpServerSpec(
            name="server",
            command=["server"],
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _McpJsonFieldCase(
        "resource.metadata",
        lambda value: McpResourceDefinition(
            uri="file:///resource",
            metadata={"probe": value},
        ).metadata["probe"],
    ),
    _McpJsonFieldCase(
        "tool.annotations",
        lambda value: McpToolDefinition(
            name="tool",
            annotations={"probe": value},
        ).annotations["probe"],
    ),
    _McpJsonFieldCase(
        "tool.input_schema",
        lambda value: McpToolDefinition(
            name="tool",
            input_schema={"probe": value},
        ).input_schema["probe"],
    ),
    _McpJsonFieldCase(
        "tool_result.content",
        lambda value: McpToolResult(content=[{"probe": value}]).content[0]["probe"],
    ),
    _McpJsonFieldCase(
        "tool_result.structured_content",
        lambda value: McpToolResult(structured_content={"probe": value}).structured_content[
            "probe"
        ],  # type: ignore[index]
    ),
    _McpJsonFieldCase(
        "resource_result.contents",
        lambda value: McpResourceResult(contents=[{"probe": value}]).contents[0]["probe"],
    ),
    _McpJsonFieldCase(
        "initialize.capabilities",
        lambda value: McpInitializeResult(
            protocol_version="2025-06-18",
            capabilities={"probe": value},
        ).capabilities["probe"],
    ),
)

_TEXT_FIELD_CASES = (
    _McpTextFieldCase(
        "server.name",
        lambda value: McpServerSpec(name=value, command=["server"]).name,
    ),
    _McpTextFieldCase(
        "server.connection_id",
        lambda value: (
            McpServerSpec(
                name="server",
                connection_id=value,
                command=["server"],
            ).connection_id
        ),  # type: ignore[arg-type,return-value]
    ),
    _McpTextFieldCase(
        "server.url",
        lambda value: McpServerSpec(name="server", url=value).url,  # type: ignore[arg-type,return-value]
    ),
    _McpTextFieldCase(
        "server.command[]",
        lambda value: McpServerSpec(name="server", command=[value]).command[0],  # type: ignore[index,return-value]
    ),
    _McpTextFieldCase(
        "server.env.key",
        lambda value: next(
            iter(McpServerSpec(name="server", command=["server"], env={value: "value"}).env)
        ),
    ),
    _McpTextFieldCase(
        "server.env.value",
        lambda value: McpServerSpec(
            name="server",
            command=["server"],
            env={"NAME": value},
        ).env["NAME"],
    ),
    _McpTextFieldCase(
        "server.headers.key",
        lambda value: next(
            iter(
                McpServerSpec(
                    name="server", url="https://example.test", headers={value: "value"}
                ).headers
            )
        ),
    ),
    _McpTextFieldCase(
        "server.headers.value",
        lambda value: McpServerSpec(
            name="server",
            url="https://example.test",
            headers={"x-name": value},
        ).headers["x-name"],
    ),
    _McpTextFieldCase(
        "server.secret_env.key",
        lambda value: next(
            iter(
                McpServerSpec(
                    name="server",
                    command=["server"],
                    secret_env={value: SecretRef(name="secret")},
                ).secret_env
            )
        ),
    ),
    _McpTextFieldCase(
        "server.secret_headers.key",
        lambda value: next(
            iter(
                McpServerSpec(
                    name="server",
                    url="https://example.test",
                    secret_headers={value: SecretRef(name="secret")},
                ).secret_headers
            )
        ),
    ),
    _McpTextFieldCase(
        "resource.uri",
        lambda value: McpResourceDefinition(uri=value).uri,
    ),
    _McpTextFieldCase(
        "resource.name",
        lambda value: McpResourceDefinition(uri="file:///resource", name=value).name,  # type: ignore[arg-type,return-value]
    ),
    _McpTextFieldCase(
        "resource.description",
        lambda value: McpResourceDefinition(uri="file:///resource", description=value).description,  # type: ignore[arg-type,return-value]
    ),
    _McpTextFieldCase(
        "resource.mime_type",
        lambda value: McpResourceDefinition(uri="file:///resource", mime_type=value).mime_type,  # type: ignore[arg-type,return-value]
    ),
    _McpTextFieldCase(
        "tool.name",
        lambda value: McpToolDefinition(name=value).name,
    ),
    _McpTextFieldCase(
        "tool.description",
        lambda value: McpToolDefinition(name="tool", description=value).description,
    ),
    _McpTextFieldCase(
        "initialize.protocol_version",
        lambda value: McpInitializeResult(protocol_version=value).protocol_version,
    ),
    _McpTextFieldCase(
        "initialize.server_name",
        lambda value: (
            McpInitializeResult(
                protocol_version="2025-06-18",
                server_name=value,
            ).server_name
        ),  # type: ignore[arg-type,return-value]
    ),
    _McpTextFieldCase(
        "initialize.server_version",
        lambda value: (
            McpInitializeResult(
                protocol_version="2025-06-18",
                server_version=value,
            ).server_version
        ),  # type: ignore[arg-type,return-value]
    ),
    _McpTextFieldCase(
        "initialize.instructions",
        lambda value: (
            McpInitializeResult(
                protocol_version="2025-06-18",
                instructions=value,
            ).instructions
        ),  # type: ignore[arg-type,return-value]
    ),
)

_EXPECTED_JSON_FIELD_NAMES = {
    "initialize.capabilities",
    "resource.metadata",
    "resource_result.contents",
    "server.metadata",
    "tool.annotations",
    "tool.input_schema",
    "tool_result.content",
    "tool_result.structured_content",
}
_EXPECTED_TEXT_FIELD_NAMES = {
    "initialize.instructions",
    "initialize.protocol_version",
    "initialize.server_name",
    "initialize.server_version",
    "resource.description",
    "resource.mime_type",
    "resource.name",
    "resource.uri",
    "server.command[]",
    "server.connection_id",
    "server.env.key",
    "server.env.value",
    "server.headers.key",
    "server.headers.value",
    "server.name",
    "server.secret_env.key",
    "server.secret_headers.key",
    "server.url",
    "tool.description",
    "tool.name",
}


def test_mcp_conformance_matrix_covers_every_issue_listed_field() -> None:
    assert {case.name for case in _JSON_FIELD_CASES} == _EXPECTED_JSON_FIELD_NAMES
    assert {case.name for case in _TEXT_FIELD_CASES} == _EXPECTED_TEXT_FIELD_NAMES


def _too_deep_value() -> Any:
    value: Any = "leaf"
    for _ in range(129):
        value = [value]
    return value


@pytest.mark.parametrize("case", _JSON_FIELD_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize(
    ("value", "expected_code"),
    [
        (MAX_DURABLE_JSON_INTEGER + 1, "integer_out_of_range"),
        (MIN_DURABLE_JSON_INTEGER - 1, "integer_out_of_range"),
        (float(2**63), "integral_float_out_of_range"),
        (float(MIN_DURABLE_JSON_INTEGER - 2048), "integral_float_out_of_range"),
        (float("nan"), "non_finite_number"),
        (f"{_SENSITIVE_CANARY}\x00", "nul_character"),
        (f"{_SENSITIVE_CANARY}\ud800", "unicode_surrogate"),
        ({f"{_SENSITIVE_CANARY}\x00": "value"}, "nul_character"),
        ({f"{_SENSITIVE_CANARY}\ud800": "value"}, "unicode_surrogate"),
        (_too_deep_value(), "nesting_too_deep"),
    ],
)
def test_mcp_json_fields_reject_nonportable_values(
    case: _McpJsonFieldCase,
    value: Any,
    expected_code: str,
) -> None:
    with pytest.raises(ValueError) as raised:
        case.capture(value)

    durable_error = extract_durable_value_error(raised.value)
    assert durable_error is not None
    assert durable_error.code == expected_code
    assert _SENSITIVE_CANARY not in str(raised.value)


@pytest.mark.parametrize("case", _JSON_FIELD_CASES, ids=lambda case: case.name)
def test_mcp_json_fields_normalize_and_copy_portable_values(case: _McpJsonFieldCase) -> None:
    value: dict[str, Any] = {
        "bounds": [MIN_DURABLE_JSON_INTEGER, MAX_DURABLE_JSON_INTEGER],
        "integral": 42.0,
        "negative_zero": -0.0,
        "fractional": 1.25,
        "unicode": "Zażółć 😀",
        "nested": [{"status": "original"}],
    }

    captured = case.capture(value)

    assert captured == {
        "bounds": [MIN_DURABLE_JSON_INTEGER, MAX_DURABLE_JSON_INTEGER],
        "integral": 42,
        "negative_zero": 0,
        "fractional": 1.25,
        "unicode": "Zażółć 😀",
        "nested": [{"status": "original"}],
    }
    assert type(captured["integral"]) is int
    assert type(captured["negative_zero"]) is int
    assert type(captured["fractional"]) is float

    value["nested"][0]["status"] = "mutated"
    assert captured["nested"][0]["status"] == "original"


@pytest.mark.parametrize("case", _TEXT_FIELD_CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("suffix", ["\x00", "\ud800"], ids=["nul", "surrogate"])
def test_mcp_text_fields_reject_nonportable_text(
    case: _McpTextFieldCase,
    suffix: str,
) -> None:
    with pytest.raises(ValueError) as raised:
        case.capture(f"{_SENSITIVE_CANARY}{suffix}")

    assert extract_durable_value_error(raised.value) is not None
    assert _SENSITIVE_CANARY not in str(raised.value)


@pytest.mark.parametrize("case", _TEXT_FIELD_CASES, ids=lambda case: case.name)
def test_mcp_text_fields_preserve_ordinary_unicode(case: _McpTextFieldCase) -> None:
    assert case.capture("Zażółć 😀") == "Zażółć 😀"


def test_mcp_server_spec_defensively_copies_transport_configuration() -> None:
    command = ["python", "server.py"]
    env = {"MODE": "original"}
    headers = {"x-mode": "original"}
    secret_ref = SecretRef(
        name="secret",
        metadata={"nested": [{"status": "original"}]},
    )
    secret_env = {"TOKEN": secret_ref}
    secret_headers = {"authorization": secret_ref}

    spec = McpServerSpec(
        name="server",
        command=command,
        env=env,
        headers=headers,
        secret_env=secret_env,
        secret_headers=secret_headers,
    )

    command[0] = "mutated"
    env["MODE"] = "mutated"
    headers["x-mode"] = "mutated"
    secret_env.clear()
    secret_headers.clear()
    secret_ref.metadata["nested"][0]["status"] = "mutated"

    assert spec.command == ["python", "server.py"]
    assert spec.env == {"MODE": "original"}
    assert spec.headers == {"x-mode": "original"}
    assert tuple(spec.secret_env) == ("TOKEN",)
    assert tuple(spec.secret_headers) == ("authorization",)
    assert spec.secret_env["TOKEN"].metadata == {"nested": [{"status": "original"}]}
    assert spec.secret_headers["authorization"].metadata == {"nested": [{"status": "original"}]}


def test_connect_mcp_toolset_revalidates_server_before_custom_client_connection() -> None:
    client = _ForbiddenMcpClient()
    server = McpServerSpec(name="server", command=["server"])
    server.command[0] = f"{_SENSITIVE_CANARY}\x00"

    with pytest.raises(ValueError) as raised:
        asyncio.run(connect_mcp_toolset(server, client=client))

    assert extract_durable_value_error(raised.value) is not None
    assert _SENSITIVE_CANARY not in str(raised.value)
    assert client.calls == 0


def test_connect_mcp_toolset_owns_one_server_snapshot_across_connection_await() -> None:
    async def run() -> None:
        expected = McpServerSpec(
            name="original-server",
            connection_id="original-connection",
            command=["original-command"],
            env={"MODE": "original"},
        )
        server = expected.model_copy(deep=True)
        session = _SnapshotMcpSession()
        client = _BarrierMcpClient(session)
        connection = asyncio.create_task(connect_mcp_toolset(server, client=client))

        await client.started.wait()
        assert client.received_server is not None
        client_server = client.received_server
        assert client_server is not server
        server.name = "caller-mutated-server"
        server.connection_id = "caller-mutated-connection"
        server.command[0] = "caller-mutated-command"
        server.env["MODE"] = "caller-mutated"
        client_server.connection_id = "client-mutated-connection"
        client.release.set()

        toolset = await connection
        try:
            assert session.connected_server is not None
            assert server.connection_id == "caller-mutated-connection"
            assert client_server.connection_id == "client-mutated-connection"
            assert session.connected_server == expected
            assert toolset.server == expected
            assert toolset.manifest_identity == mcp_tool_manifest_identity(server=expected)
        finally:
            await toolset.close()
        assert session.closed is True

    asyncio.run(run())


def test_stdio_connect_revalidates_server_before_secret_resolution_or_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_called = False

    async def forbidden_spawn(*args: Any, **kwargs: Any) -> Any:
        nonlocal spawn_called
        del args, kwargs
        spawn_called = True
        raise AssertionError("subprocess creation must not run for invalid MCP configuration")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden_spawn)
    resolver = _ForbiddenSecretResolver()
    server = McpServerSpec(
        name="server",
        command=["server"],
        secret_env={"TOKEN": SecretRef(name="secret")},
    )
    server.command[0] = f"{_SENSITIVE_CANARY}\x00"

    with pytest.raises(ValueError) as raised:
        asyncio.run(StdioMcpClient(secret_resolver=resolver).connect(server))

    assert extract_durable_value_error(raised.value) is not None
    assert _SENSITIVE_CANARY not in str(raised.value)
    assert resolver.called is False
    assert spawn_called is False


def test_environment_revalidates_server_before_registration() -> None:
    server = McpServerSpec(name="server", command=["server"])
    server.command[0] = f"{_SENSITIVE_CANARY}\x00"

    with pytest.raises(ValueError) as raised:
        Environment(EnvironmentSpec(name="environment"), mcp_servers=[server])

    assert extract_durable_value_error(raised.value) is not None
    assert _SENSITIVE_CANARY not in str(raised.value)


def test_environment_rejects_mcp_server_subclasses_before_registration() -> None:
    class McpServerSpecSubclass(McpServerSpec):
        pass

    server = McpServerSpecSubclass(name="server", command=["server"])

    with pytest.raises(TypeError, match="McpServerSpec"):
        Environment(EnvironmentSpec(name="environment"), mcp_servers=[server])


def test_http_connect_revalidates_server_before_secret_resolution_or_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_called = False

    def forbidden_client(*args: Any, **kwargs: Any) -> Any:
        nonlocal client_called
        del args, kwargs
        client_called = True
        raise AssertionError("HTTP setup must not run for invalid MCP configuration")

    monkeypatch.setattr(mcp_http_module.httpx, "AsyncClient", forbidden_client)
    resolver = _ForbiddenSecretResolver()
    server = McpServerSpec(
        name="server",
        url="https://example.test/mcp",
        secret_headers={"authorization": SecretRef(name="secret")},
    )
    server.url = f"{_SENSITIVE_CANARY}\ud800"

    with pytest.raises(ValueError) as raised:
        asyncio.run(HttpMcpClient(secret_resolver=resolver).connect(server))

    assert extract_durable_value_error(raised.value) is not None
    assert _SENSITIVE_CANARY not in str(raised.value)
    assert resolver.called is False
    assert client_called is False
