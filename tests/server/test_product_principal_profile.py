"""A product principal's display name and email: carried, never keyed on."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from tests.core.test_resource_execution_access import Policy
from tests.server.test_public_service import _build_service

from cayu import CayuApp, RunRequest
from cayu.server import AuthenticatedProductAccess, ProductPrincipal, ProxyIdentityAuth
from cayu.server.development_identity import DevelopmentIdentityProxy
from cayu.server.resource_access import create_resource_router
from cayu.server.service import _product_auth_dependency
from cayu.sessions import SessionIdentity


def test_name_and_email_are_optional_display_fields() -> None:
    principal = ProductPrincipal(
        tenant_id="acme", subject_id="alice", name="Alice Example", email="alice@example.com"
    )
    assert (principal.name, principal.email) == ("Alice Example", "alice@example.com")
    bare = ProductPrincipal(tenant_id="acme", subject_id="alice")
    assert (bare.name, bare.email) == (None, None)
    # Display only: neither shows in repr, where logs would pick it up.
    assert "alice@example.com" not in repr(principal)
    assert "Alice Example" not in repr(principal)


@pytest.mark.parametrize(
    ("name", "email"),
    [
        ("x" * 257, "a" * 310 + "@example.com"),
        ("Alice\nExample", "alice@example.com\r\n"),
        ("Alice\u2028", "alice\x00@example.com"),
        ("Admin\u202egpj.exe", "a\u2066@example.com"),
        ("   ", ""),
        (7, ["alice@example.com"]),
    ],
)
def test_an_unusable_profile_becomes_none_without_refusing(name: object, email: object) -> None:
    principal = ProductPrincipal(tenant_id="acme", subject_id="alice", name=name, email=email)

    assert (principal.subject_id, principal.name, principal.email) == ("alice", None, None)


def test_a_name_written_with_a_zero_width_joiner_is_kept() -> None:
    name = "Mehr\u200cnaz"
    assert ProductPrincipal(tenant_id="acme", subject_id="m", name=name).name == name


def test_the_longest_usable_values_are_kept() -> None:
    principal = ProductPrincipal(
        tenant_id="acme", subject_id="alice", name="n" * 256, email="e" * 308 + "@example.com"
    )
    assert len(principal.name or "") == 256 and len(principal.email or "") == 320


def test_the_product_auth_adapter_keeps_the_profile() -> None:
    async def from_principal(_request):
        return ProductPrincipal(tenant_id="acme", subject_id="alice", name="Alice", email="a@x.io")

    async def from_mapping(_request):
        return {"tenant_id": "acme", "subject_id": "bob", "name": "Bob"}

    principal = asyncio.run(_product_auth_dependency(from_principal)(None))  # type: ignore[arg-type]
    assert (principal.name, principal.email) == ("Alice", "a@x.io")
    mapped = asyncio.run(_product_auth_dependency(from_mapping)(None))  # type: ignore[arg-type]
    assert (mapped.subject_id, mapped.name, mapped.email) == ("bob", "Bob", None)


def test_product_operations_are_the_same_with_and_without_a_profile() -> None:
    async def product_auth(request: Request) -> ProductPrincipal:
        named = request.headers.get("x-named") == "yes"
        return ProductPrincipal(
            tenant_id="tenant-a",
            subject_id="alice",
            name="Alice" if named else None,
            email="alice@example.com" if named else None,
        )

    service, store, provider = _build_service(
        product_access=AuthenticatedProductAccess(dependency=product_auth)
    )
    client = TestClient(service.asgi_app)
    first = client.post(
        "/api/operations", headers={"Idempotency-Key": "one"}, json={"request": "work"}
    )
    # The same request with a profile is the same operation, replayed, not a conflict.
    again = client.post(
        "/api/operations",
        headers={"Idempotency-Key": "one", "x-named": "yes"},
        json={"request": "work"},
    )

    assert first.status_code == 201
    assert again.status_code in {200, 201}
    assert again.json()["id"] == first.json()["id"]
    assert len(provider.requests) == 1
    stored = [operation.model_dump_json() for operation in store.by_work_id.values()]
    assert stored and not any("alice@example.com" in text for text in stored)


def test_proxy_product_dependency_carries_the_signed_name() -> None:
    proxy = DevelopmentIdentityProxy()
    auth = ProxyIdentityAuth(
        issuer=proxy.issuer, audience=proxy.audience, signing_keys=proxy.signing_keys()
    )
    dependency = auth.product_dependency()
    server = FastAPI()

    @server.get("/me")
    async def me(principal: ProductPrincipal = Depends(dependency)):  # noqa: B008
        return principal.model_dump()

    client = TestClient(server)
    named = client.get("/me", headers=proxy.headers("alice", org="acme", name="Alice"))
    assert named.json() == {
        "tenant_id": "acme",
        "subject_id": "alice",
        "name": "Alice",
        "email": None,
    }


def _acme_subject(principal: ProductPrincipal) -> str:
    # This app serves one tenant; anyone else's person is nobody here.
    if principal.tenant_id != "acme":
        raise HTTPException(status_code=403, detail="Not this app's organization.")
    return principal.subject_id


def _router_client(app: CayuApp, server: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server), base_url="http://test")


async def _app_with_acme_session() -> CayuApp:
    app = CayuApp(resource_access_policy=Policy())
    await app.session_store.create(
        RunRequest(
            agent_name="shared", session_id="own", messages=[], labels={"organization": "acme"}
        ),
        identity=SessionIdentity(provider_name="fake", model="fake"),
    )
    return app


def test_the_resource_router_takes_a_principal_through_principal_subject() -> None:
    async def run() -> None:
        app = await _app_with_acme_session()
        proxy = DevelopmentIdentityProxy()
        auth = ProxyIdentityAuth(
            issuer=proxy.issuer, audience=proxy.audience, signing_keys=proxy.signing_keys()
        )
        server = FastAPI()
        server.include_router(
            create_resource_router(
                app, authenticate=auth.product_dependency(), principal_subject=_acme_subject
            )
        )
        async with _router_client(app, server) as client:
            own = await client.get(
                "/resources/sessions/own", headers=proxy.headers("alice", org="acme", name="A")
            )
            assert own.status_code == 200
            assert (await client.get("/resources/sessions/own")).status_code == 401
            # The same subject in another tenant is not acme's alice.
            other = await client.get(
                "/resources/sessions/own", headers=proxy.headers("alice", org="attacker")
            )
            assert other.status_code == 403 and "own" not in other.text

    asyncio.run(run())


def test_a_principal_without_principal_subject_is_refused() -> None:
    async def run() -> None:
        app = await _app_with_acme_session()

        async def authenticate(request: Request) -> ProductPrincipal:
            return ProductPrincipal(tenant_id="attacker", subject_id="alice")

        server = FastAPI()
        server.include_router(create_resource_router(app, authenticate=authenticate))
        async with _router_client(app, server) as client:
            # Never reduced to the bare subject: the tenant would be lost.
            with pytest.raises(TypeError, match="principal_subject"):
                await client.get("/resources/sessions/own")

    asyncio.run(run())
