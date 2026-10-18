"""Credentialless open internet egress: any public HTTPS host, every request audited."""

from __future__ import annotations

import asyncio

import pytest

from cayu.egress import (
    ApprovedEgressDestination,
    CapturedRequest,
    CapturedResponse,
    HttpEgressPolicy,
    HttpxUpstream,
    OpenInternetEgressPolicy,
    PublicWebEgressPolicy,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
    VirtualCredentialSpec,
)
from cayu.egress.authority import EgressAuthorityChangeKind, compare_egress_authority
from cayu.egress.runtime import VirtualEgressEnvironmentFactory
from cayu.vaults import SecretRef, StaticVault


def factory(name="internet", **kwargs):
    return VirtualEgressEnvironmentFactory(
        policies={name: OpenInternetEgressPolicy(name=name)},
        open_internet_policy=name,
        runner_kind="docker",
        **kwargs,
    )


def test_open_internet_is_explicit_and_durable():
    first = factory()._egress_authority_identity
    assert first.bindings == ()
    assert first.policies[0].kind == "open_internet"
    assert type(first).model_validate_json(first.model_dump_json()) == first
    assert (
        compare_egress_authority(first, factory()._egress_authority_identity)
        == EgressAuthorityChangeKind.UNCHANGED
    )
    assert (
        compare_egress_authority(first, factory("changed")._egress_authority_identity)
        == EgressAuthorityChangeKind.INCOMPARABLE
    )
    research = VirtualEgressEnvironmentFactory(
        policies={"internet": PublicWebEgressPolicy(name="internet")},
        public_web_policy="internet",
        runner_kind="docker",
    )._egress_authority_identity
    assert research.fingerprint != first.fingerprint


def test_open_internet_refuses_credentials_and_mixed_authority():
    internet = {"internet": OpenInternetEgressPolicy(name="internet")}
    with pytest.raises(ValueError, match="explicit open_internet_policy"):
        VirtualEgressEnvironmentFactory(policies=internet, runner_kind="docker")
    for wrong in (PublicWebEgressPolicy(name="internet"), None):
        with pytest.raises(ValueError, match="open_internet_policy|public_web_policy admission"):
            VirtualEgressEnvironmentFactory(
                policies={} if wrong is None else {"internet": wrong},
                open_internet_policy="internet",
                runner_kind="docker",
            )
    with pytest.raises(ValueError, match="not both"):
        VirtualEgressEnvironmentFactory(
            policies=internet,
            open_internet_policy="internet",
            public_web_policy="internet",
            runner_kind="docker",
        )
    receiver = HttpEgressPolicy(
        name="receiver",
        allowed_hosts=["receiver.example"],
        allowed_endpoints=[("POST", "/v1/actions")],
    )
    for extra in (
        {
            "credentials": [
                VirtualCredentialSpec(
                    env_name="RECEIVER_TOKEN",
                    secret=SecretRef(name="receiver"),
                    destination="receiver.example",
                    policy_name="receiver",
                )
            ],
            "resolver": StaticVault({"receiver": "secret"}),
            "policies": {**internet, "receiver": receiver},
        },
        {"resolver": StaticVault({})},
        {
            "approved_destinations": [
                ApprovedEgressDestination(destination="receiver.example", policy_name="receiver")
            ],
            "policies": {**internet, "receiver": receiver},
        },
        {"policies": {**internet, "receiver": receiver}},
    ):
        options = {"policies": internet, "open_internet_policy": "internet", **extra}
        with pytest.raises(ValueError, match="separate credentialless environment"):
            VirtualEgressEnvironmentFactory(runner_kind="docker", **options)
    with pytest.raises(ValueError, match="routing"):
        factory(upstream=HttpxUpstream(routes={"example.com": "http://127.0.0.1"}))
    with pytest.raises(ValueError, match="explicit open_internet_policy"):
        TransparentEgressBroker(registry=VirtualCredentialRegistry(), policies=internet)
    with pytest.raises(ValueError, match="separate credentialless broker"):
        TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies=internet,
            open_internet_policy="internet",
            resolver=StaticVault({}),
        )


def test_broker_forwards_any_public_host_unchanged_and_audits_every_request(monkeypatch):
    async def scenario():
        answers = ("140.82.112.3",)
        sent = []
        decisions = []

        async def resolve(host, port):
            return answers

        async def send(self, request, *, target, limits, timeout_s):
            sent.append((request, target))
            return CapturedResponse(status_code=200, body=b"ok")

        monkeypatch.setattr(HttpxUpstream, "_send_to_target", send)
        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies={"internet": OpenInternetEgressPolicy(name="internet")},
            open_internet_policy="internet",
            upstream=HttpxUpstream(destination_resolver=resolve),
            audit=decisions.append,
        )
        assert broker.has_credentialless_destinations
        headers = {
            "Content-Type": "application/x-git-upload-pack-request",
            "Git-Protocol": "version=2",
            "Authorization": "Bearer guest-supplied",
        }
        requests = [
            CapturedRequest(method="GET", host="pypi.org", path="/simple/requests/"),
            CapturedRequest(
                method="POST",
                host="github.com",
                path="/org/repo.git/git-upload-pack",
                headers=headers,
                body=b"0014command=ls-refs\n0000",
            ),
            CapturedRequest(method="PUT", host="registry.npmjs.org", path="/pkg", body=b"{}"),
            CapturedRequest(method="DELETE", host="api.example.com", path="/item/1"),
        ]
        for request in requests:
            assert await broker.authorize_connect_destination(host=request.host, port=443)
            response = await broker.handle_request(request)
            assert response.status_code == 200
            forwarded, target = sent[-1]
            # Method, path, headers and body reach the upstream unchanged.
            assert forwarded.method == request.method
            assert forwarded.path == request.path
            assert forwarded.body == request.body
            assert forwarded.headers == request.headers
            assert target.url == f"https://140.82.112.3{request.path}"
            assert target.sni_hostname == request.host
        assert [(item.allowed, item.destination, item.method) for item in decisions] == [
            (True, request.host, request.method) for request in requests
        ]
        assert {item.policy_name for item in decisions} == {"internet"}
        assert {item.authorization_kind for item in decisions} == {"credentialless"}

        before = len(sent)
        for host in ("127.0.0.1", "0x7f.0.0.1", "2130706433", "localhost", "[::1]"):
            assert not await broker.authorize_connect_destination(host=host, port=443)
        assert not await broker.authorize_connect_destination(host="pypi.org", port=8443)
        for answer in (("10.0.0.1",), ("169.254.169.254",), ("::1",), ("8.8.8.8", "192.168.1.1")):
            answers = answer
            response = await broker.handle_request(
                CapturedRequest(method="GET", host="pypi.org", path="/")
            )
            assert response.status_code == 403
        assert len(sent) == before
        assert sum(not item.allowed for item in decisions) == 4

        await broker.revoke_authority_and_wait(())
        assert not await broker.authorize_connect_destination(host="pypi.org", port=443)
        answers = ("140.82.112.3",)
        assert (
            await broker.handle_request(CapturedRequest(method="GET", host="pypi.org", path="/"))
        ).status_code == 403
        assert len(sent) == before

    asyncio.run(scenario())


@pytest.mark.parametrize("stored", [None, "a" * 64])
def test_open_internet_reconnect_refuses_missing_or_stale_authority(stored):
    from cayu.egress.runtime import InvalidEgressReconnectMetadataError
    from cayu.environments import EnvironmentFactoryOperation, EnvironmentFactoryRequest

    async def scenario():
        metadata = {} if stored is None else {"public_web_authority_fingerprint": stored}
        with pytest.raises(InvalidEgressReconnectMetadataError, match="authority fingerprint"):
            await factory().create(
                EnvironmentFactoryRequest(
                    session_id="session",
                    agent_name="agent",
                    environment_name="builder",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=metadata,
                )
            )

    asyncio.run(scenario())
