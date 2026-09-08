"""Production server composition must not create independent browser authority."""

import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import SecretBytes, ValidationError
from tests.core.test_browser_control import operator_purpose
from tests.core.test_browser_control_authorization import Policy

from cayu.runtime.app import CayuApp
from cayu.runtime.browser_control_config import BrowserControlConfig
from cayu.server import AuthenticatedAccess, OpenAccess, ServerConfig, create_server
from cayu.server._browser_control_config import BrowserControlServerConfig
from cayu.server.auth import BasicAuth


def transport():
    return BrowserControlServerConfig(
        operator_origin="https://operator.test", signing_key=SecretBytes(b"k" * 32)
    )


def test_documented_operator_setup_builds_a_protected_server():
    async def scenario():
        documentation = (
            Path(__file__).resolve().parents[2] / "docs" / "server-configuration.md"
        ).read_text()
        section = documentation.split("## Browser operator transport", 1)[1]
        example = section.split("```python\n", 1)[1].split("```", 1)[0]
        namespace = {
            "application_browser_policy": Policy(False),
            "require_operator": BasicAuth(username="operator", password="password"),
            "browser_control_signing_key": b"k" * 32,
        }
        exec(compile(example, "documented-browser-operator-setup", "exec"), namespace)
        app = namespace["cayu_app"]
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=namespace["server"]),
                base_url="https://control.example",
            ) as client:
                path = "/api/browser-control/operator-session"
                assert (await client.post(path)).status_code == 401
                client.auth = httpx.BasicAuth("operator", "password")
                assert (await client.post(path)).status_code == 200
        finally:
            assert await app._browser_control_runtime.service.drain()

    asyncio.run(scenario())


def test_protected_server_requires_application_and_authentication():
    from cayu import BrowserControlConfig as PublicBrowserControlConfig
    from cayu.server import BrowserControlServerConfig as PublicTransportConfig

    assert PublicBrowserControlConfig is BrowserControlConfig
    assert PublicTransportConfig is BrowserControlServerConfig
    with pytest.raises(ValidationError):
        ServerConfig(access=OpenAccess(), browser_control=transport())
    config = ServerConfig(
        access=AuthenticatedAccess(dependency=BasicAuth(username="operator", password="password")),
        browser_control=transport(),
    )
    assert "signing_key" not in config.model_dump()["browser_control"]
    assert config.safe_summary()["browser_control"] == {"configured": True}
    assert "operator.test" not in repr(config.safe_summary())
    assert (
        ServerConfig.protected(
            BasicAuth(username="operator", password="password"), browser_control=transport()
        ).browser_control
        == transport()
    )
    with pytest.raises(ValueError, match="application policy"):
        create_server(CayuApp(enable_logging=False), config=config)


@pytest.mark.parametrize(
    "origin",
    ["http://operator.test", "https://operator.test/path", "https://user:secret@operator.test"],
)
def test_invalid_operator_origin_is_rejected(origin):
    with pytest.raises(ValidationError):
        BrowserControlServerConfig(operator_origin=origin, signing_key=SecretBytes(b"k" * 32))


def test_create_server_mounts_private_routes_with_auth_and_tls():
    async def scenario():
        app = CayuApp(
            enable_logging=False,
            browser_control=BrowserControlConfig(
                purpose=operator_purpose(),
                policy=Policy(False),
                guest_endpoint="wss://operator.test/api/browser-control/guest",
            ),
        )
        server = create_server(
            app,
            config=ServerConfig(
                access=AuthenticatedAccess(
                    dependency=BasicAuth(username="operator", password="password")
                ),
                browser_control=transport(),
            ),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server), base_url="https://operator.test"
        ) as client:
            path = "/api/browser-control/operator-session"
            assert (await client.post(path)).status_code == 401
            client.auth = httpx.BasicAuth("operator", "password")
            response = await client.post(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert "operator_session_token" in response.json()
            assert (
                await client.post(path, headers={"Origin": "https://elsewhere.test"})
            ).status_code == 403
            assert (await client.post("http://operator.test" + path)).status_code == 403
        assert app._browser_control_runtime is not None
        assert not app._browser_control_runtime.service._owners
        assert await app._browser_control_runtime.service.drain()

    asyncio.run(scenario())
