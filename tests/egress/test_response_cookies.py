"""Repeated cookies survive real TLS proxying, scrubbing and response copying."""

import asyncio
import ssl
from http.cookies import SimpleCookie

import httpx
import pytest

from cayu.egress import (
    ApprovedEgressDestination,
    CapturedRequest,
    CapturedResponse,
    EgressUpstreamOperation,
    HttpEgressPolicy,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.broker import CAYU_EGRESS_ERROR_HEADER, _scrub_response
from cayu.egress.proxy_server import TransparentEgressProxyServer, _serialize_response_head
from cayu.vaults import SecretRef, StaticVault


@pytest.mark.parametrize("credentialed", [False, True])
def test_login_cookies_survive_upstream_broker_and_real_tls_proxy(
    tmp_path, caplog, capfd, recwarn, credentialed
):
    credential = "resolved-upstream-credential-canary"
    cookies = (
        "challenge=; Expires=Thu, 01 Jan 1970 00:00:00 GMT; Max-Age=0; Path=/; Secure; HttpOnly",
        "authenticated=local-cookie-canary; Path=/; Secure; HttpOnly; SameSite=Strict",
    )
    seen = []
    decisions = []

    async def upstream(request):
        seen.append(request.url.path)
        if request.url.path == "/login":
            return httpx.Response(200, headers={"Set-Cookie": "challenge=pending; Path=/; Secure"})
        if request.url.path == "/mfa":
            assert "challenge=pending" in request.headers.get("cookie", "")
            fields = [("sEt-CoOkIe", item) for item in cookies]
            if credentialed:
                fields.append(("Set-Cookie", f"echo={credential}; Path=/; Secure"))
            return httpx.Response(303, headers=[*fields, ("Location", "/member")])
        jar = SimpleCookie(request.headers.get("cookie", ""))
        authenticated = jar.get("authenticated")
        allowed = authenticated is not None and authenticated.value == "local-cookie-canary"
        assert "challenge" not in jar
        return httpx.Response(200 if allowed else 401, text="member" if allowed else "login")

    async def run():
        policy = HttpEgressPolicy(
            name="local-login",
            allowed_hosts=("login.example.test",),
            allowed_endpoints=(("GET", "/login"), ("POST", "/mfa"), ("GET", "/member")),
        )
        registry = VirtualCredentialRegistry()
        broker = TransparentEgressBroker(
            registry=registry,
            resolver=StaticVault({"cookie-test": credential}),
            policies={policy.name: policy},
            approved_destinations=(
                ApprovedEgressDestination(
                    destination="login.example.test", policy_name=policy.name
                ),
            ),
            upstream=HttpxUpstream(
                transport=httpx.MockTransport(upstream),
                routes={"login.example.test": "http://192.0.2.10"},
            ),
            audit=decisions.append,
        )
        request_headers = {}
        if credentialed:
            grant = registry.mint(
                session_id="cookie-test",
                env_name="TOKEN",
                secret=SecretRef(name="cookie-test"),
                destination="login.example.test",
                credential_kind="opaque_bearer",
                policy_name=policy.name,
            )
            request_headers = {"Authorization": f"Bearer {grant.presented_value}"}
        server = TransparentEgressProxyServer(broker, loop=asyncio.get_running_loop())
        port = await server.start()
        ca = tmp_path / "ca.pem"
        ca.write_bytes(server.authority.ca_cert_pem())
        try:
            async with httpx.AsyncClient(
                proxy=f"http://127.0.0.1:{port}",
                verify=ssl.create_default_context(cafile=str(ca)),
                trust_env=False,
                timeout=10,
                headers=request_headers,
            ) as client:
                assert (await client.get("https://login.example.test/login")).status_code == 200
                response = await client.post("https://login.example.test/mfa")
                assert response.status_code == 303
                assert response.headers.get_list("set-cookie") == list(cookies) + (
                    ["echo=[REDACTED_SECRET]; Path=/; Secure"] if credentialed else []
                )
                assert (await client.get("https://login.example.test/member")).status_code == 200
        finally:
            await server.close()

    asyncio.run(run())
    assert seen == ["/login", "/mfa", "/member"]
    assert len(decisions) == 3 and all(item.allowed for item in decisions)
    captured = capfd.readouterr()
    diagnostic = (
        captured.out
        + captured.err
        + caplog.text
        + repr(decisions)
        + "\n".join(str(item.message) for item in recwarn)
    )
    assert "local-cookie-canary" not in diagnostic
    assert credential not in diagnostic


@pytest.mark.parametrize("count", [0, 1, 2, 64])
def test_cookie_copy_roundtrip_and_scrubbing_preserve_order(count):
    original = [f"cookie{index}=credential-canary; Path=/" for index in range(count)]
    response = CapturedResponse(
        status_code=200,
        headers={"Set-Cookie": "single=credential-canary"},
        set_cookie_headers=original,
    )
    original.clear()
    response = CapturedResponse.model_validate_json(response.model_dump_json())
    scrubbed = _scrub_response(response, secrets=("credential-canary",), max_body_bytes=1024)
    wire = _serialize_response_head(scrubbed).decode("latin-1")
    assert "credential-canary" not in wire
    assert wire.count("Set-Cookie:") == count + 1
    for index in range(count):
        assert scrubbed.set_cookie_headers[index] == f"cookie{index}=[REDACTED_SECRET]; Path=/"
    assert len(response.set_cookie_headers) == count


@pytest.mark.parametrize(
    "value",
    [
        "not-a-list",
        [True],
        ["x=\r\nInjected: value"],
        ["x=\0"],
        ["x=\x01"],
        ["x=\x7f"],
        ["x=\ud800"],
        ["x=\u0100"],
        ["x=y"] * 65,
        ["x" * (64 * 1024 + 1)],
    ],
)
def test_invalid_cookie_fields_rejected_without_echo_and_at_final_serialization(value):
    with pytest.raises(ValueError) as failure:
        CapturedResponse(
            status_code=200, headers={"Private": "sibling-canary"}, set_cookie_headers=value
        )
    assert "sibling-canary" not in str(failure.value) + repr(failure.value)
    malformed = CapturedResponse(status_code=200).model_copy(update={"set_cookie_headers": value})
    with pytest.raises(ValueError):
        _serialize_response_head(malformed)


def test_cookie_latin1_and_exact_byte_limit():
    value = "x=" + "\xff" * (64 * 1024 - 2)
    response = CapturedResponse(status_code=200, set_cookie_headers=(value,))
    assert value.encode("latin-1") in _serialize_response_head(response)


def test_mutated_custom_response_fails_closed_at_broker():
    class Upstream:
        def prepare(self, request, *, limits):
            async def send():
                return CapturedResponse(status_code=200).model_copy(
                    update={"set_cookie_headers": ("secret-canary\r\nInjected: true",)}
                )

            return EgressUpstreamOperation(send)

    async def run():
        policy = HttpEgressPolicy(
            name="local", allowed_hosts=("login.example.test",), allowed_endpoints=(("GET", "/"),)
        )
        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies={policy.name: policy},
            upstream=Upstream(),
            approved_destinations=(
                ApprovedEgressDestination(
                    destination="login.example.test", policy_name=policy.name
                ),
            ),
        )
        return await broker.handle_request(
            CapturedRequest(method="GET", host="login.example.test", path="/")
        )

    response = asyncio.run(run())
    assert response.status_code == 502
    assert "secret-canary" not in response.model_dump_json()
    assert response.headers[CAYU_EGRESS_ERROR_HEADER] == "fetch_failed"
