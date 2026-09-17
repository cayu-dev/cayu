from __future__ import annotations

import asyncio

import pytest

from cayu import PublicWebEgressPolicy
from cayu.egress import (
    CapturedRequest,
    CapturedResponse,
    HttpxUpstream,
    TransparentEgressBroker,
    VirtualCredentialRegistry,
)
from cayu.egress.authority import EgressAuthorityChangeKind, compare_egress_authority
from cayu.egress.runtime import VirtualEgressEnvironmentFactory


def factory(name="research", **kwargs):
    return VirtualEgressEnvironmentFactory(
        policies={name: PublicWebEgressPolicy(name=name)},
        public_web_policy=name,
        runner_kind="docker",
        **kwargs,
    )


def test_public_web_is_explicit_and_durable():
    first = factory()._egress_authority_identity
    assert first.bindings == ()
    assert first.policies[0].kind == "public_web"
    assert (
        compare_egress_authority(first, factory()._egress_authority_identity)
        == EgressAuthorityChangeKind.UNCHANGED
    )
    assert (
        compare_egress_authority(first, factory("changed")._egress_authority_identity)
        == EgressAuthorityChangeKind.INCOMPARABLE
    )
    assert type(first).model_validate_json(first.model_dump_json()) == first
    with pytest.raises(ValueError, match="explicit"):
        VirtualEgressEnvironmentFactory(
            policies={"research": PublicWebEgressPolicy(name="research")}, runner_kind="docker"
        )
    with pytest.raises(ValueError, match="routing"):
        factory(upstream=HttpxUpstream(routes={"example.com": "http://127.0.0.1"}))


def test_broker_public_admission_rebinding_methods_secrets_and_revocation(monkeypatch):
    async def scenario():
        answers = ("93.184.216.34",)
        sent = []
        resolved = []
        decisions = []

        async def resolve(host, port):
            resolved.append((host, port))
            return answers

        async def send(self, request, *, target, limits, timeout_s):
            sent.append((request, target))
            return CapturedResponse(status_code=200, body=b"public")

        monkeypatch.setattr(HttpxUpstream, "_send_to_target", send)
        broker = TransparentEgressBroker(
            registry=VirtualCredentialRegistry(),
            policies={"research": PublicWebEgressPolicy(name="research")},
            public_web_policy="research",
            upstream=HttpxUpstream(destination_resolver=resolve),
            audit=decisions.append,
        )
        for host in ("first.example", "discovered.example", "redirect.example"):
            assert await broker.authorize_connect_destination(host=host, port=443)
            response = await broker.handle_request(
                CapturedRequest(
                    method="GET",
                    host=host,
                    path="/",
                    headers={
                        "Authorization": "Bearer secret",
                        "Cookie": "secret",
                        "X-Key": "secret",
                    },
                )
            )
            assert response.status_code == 200
            assert "secret" not in str(sent[-1][0].headers)
            assert sent[-1][1].url == "https://93.184.216.34/"
            assert sent[-1][1].sni_hostname == host
        before = len(sent)
        for method in ("POST", "PUT", "DELETE", "PATCH", "CONNECT", "OPTIONS"):
            response = await broker.handle_request(
                CapturedRequest(method=method, host="first.example", path="/")
            )
            assert response.status_code == 403
        for body in (b"mutation",):
            assert (
                await broker.handle_request(
                    CapturedRequest(method="GET", host="first.example", path="/", body=body)
                )
            ).status_code == 403
        for host in (
            "127.0.0.1",
            "127.1",
            "0x7f.0.0.1",
            "127.0.0.01",
            "2130706433",
            "localhost",
            "foo@bar.example",
            "[::1]",
            "a.example:443",
        ):
            assert not await broker.authorize_connect_destination(host=host, port=443)
        assert not await broker.authorize_connect_destination(host="first.example", port=8443)
        for answer in (
            ("127.0.0.1",),
            ("10.0.0.1",),
            ("169.254.169.254",),
            ("::1",),
            ("93.184.216.34", "192.168.1.1"),
        ):
            answers = answer
            response = await broker.handle_request(
                CapturedRequest(method="GET", host="first.example", path="/")
            )
            assert response.status_code == 403
        assert len(sent) == before
        assert len(resolved) == 8
        await broker.revoke_authority_and_wait(())
        assert not await broker.authorize_connect_destination(host="new.example", port=443)
        assert (
            await broker.handle_request(
                CapturedRequest(method="GET", host="first.example", path="/")
            )
        ).status_code == 403
        assert any(item.allowed for item in decisions)
        assert any(not item.allowed for item in decisions)

    asyncio.run(scenario())


@pytest.mark.parametrize("stored", [None, "a" * 64])
def test_public_reconnect_refuses_missing_or_stale_authority_before_adapter_access(stored):
    from cayu.egress.runtime import InvalidEgressReconnectMetadataError
    from cayu.environments import EnvironmentFactoryOperation, EnvironmentFactoryRequest

    async def scenario():
        metadata = {} if stored is None else {"public_web_authority_fingerprint": stored}
        with pytest.raises(InvalidEgressReconnectMetadataError, match="authority fingerprint"):
            await factory().create(
                EnvironmentFactoryRequest(
                    session_id="session",
                    agent_name="agent",
                    environment_name="browser",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=metadata,
                )
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "source_public,target_public", [(True, True), (False, True), (True, False)]
)
@pytest.mark.parametrize("reconcile", [False, True])
def test_public_authority_adoption_preserves_reconnect(
    source_public, target_public, reconcile, monkeypatch
):
    from tests.core.test_execution_profiles import (
        _adoption_handoff_factory,
        _AdoptionHandoffAdapter,
    )
    from tests.runtime.test_egress_authority_transitions import _create_session, _decision_between

    from cayu.egress.transitions import (
        EgressAuthorityTransitionCoordinator,
        SessionCheckpointEgressAuthorityTransitionStore,
        authorized_egress_authority_transition,
        egress_authority_owner_fingerprint,
    )
    from cayu.environments import EnvironmentFactoryOperation, EnvironmentFactoryRequest
    from cayu.sessions.base import InMemorySessionStore

    class Adapter(_AdoptionHandoffAdapter):
        supports_reconnect = True

        def reconnect_metadata(self, runner):
            return {"allocation": self.fingerprint}

        def validate_reconnect_metadata(self, metadata):
            assert metadata == {"allocation": self.fingerprint}
            return dict(metadata)

        async def prepare_reconnect(
            self, *, session_id, environment_name, grants, broker, reconnect_metadata
        ):
            self.validate_reconnect_metadata(reconnect_metadata)
            return await self.prepare(session_id=session_id, grants=grants, broker=broker)

        async def cutover_authority(self, request):
            result = await super().cutover_authority(request)
            self.receipt = result.receipt
            return result

        async def reconcile_authority_cutover(self, request):
            assert self.receipt.to_fingerprint == request.target_authority.fingerprint
            return self.receipt

    async def scenario():
        adapter = Adapter()

        def make_factory(public, generation):
            if not public:
                return _adoption_handoff_factory(
                    adapter=adapter, generation=generation, allow_post=False
                )
            return VirtualEgressEnvironmentFactory(
                policies={"research": PublicWebEgressPolicy(name="research")},
                public_web_policy="research",
                adapter=adapter,
                egress_authority_generation=generation,
            )

        source = make_factory(source_public, 1)
        target = make_factory(target_public, 2)
        store = InMemorySessionStore()
        await _create_session(store, "public-adoption")
        transition_store = SessionCheckpointEgressAuthorityTransitionStore(store)
        coordinator = EgressAuthorityTransitionCoordinator(transition_store)
        original = await source.create(
            EnvironmentFactoryRequest(
                session_id="public-adoption",
                agent_name="assistant",
                environment_name="egress",
            )
        )
        reconnected = None
        try:
            original_metadata = dict(original.reconnect_metadata)
            decision = _decision_between(
                source.egress_authority_identity,
                target.egress_authority_identity,
                session_id="public-adoption",
                idempotency_identity="public-adoption",
            )
            authorized = authorized_egress_authority_transition(
                decision=decision,
                transition_id="public-adoption",
                environment_name="egress",
                owner_fingerprint=egress_authority_owner_fingerprint("owner"),
                source_environment_fingerprint=original_metadata["allocation_fingerprint"],
            )
            if reconcile:

                def interrupted_handoff(*args):
                    raise RuntimeError("interrupted before handoff")

                with monkeypatch.context() as patch:
                    patch.setattr(target, "_bind_adopted_authority", interrupted_handoff)
                    with pytest.raises(RuntimeError, match="interrupted before handoff"):
                        await target.adopt_authority(
                            factory_result=original,
                            authorized=authorized,
                            coordinator=coordinator,
                            owner_token="owner",
                        )
                assert original.reconnect_metadata == original_metadata

            handoff = await target.adopt_authority(
                factory_result=original,
                authorized=authorized,
                coordinator=coordinator,
                owner_token="owner",
            )
            adopted = handoff._claim_factory_result()
            assert adopted is original
            assert handoff.transition.state.value == "active"
            persisted = await transition_store.load("public-adoption")
            assert (
                persisted is not None
                and persisted.target_authority == target.egress_authority_identity
            )
            expected_metadata = dict(original_metadata)
            if target_public:
                expected_metadata["public_web_authority_fingerprint"] = (
                    target.egress_authority_identity.fingerprint
                )
            else:
                expected_metadata.pop("public_web_authority_fingerprint", None)
            assert adopted.reconnect_metadata == expected_metadata
            await adopted.environment.runner.close()
            reconnected = await target.create(
                EnvironmentFactoryRequest(
                    session_id="public-adoption",
                    agent_name="assistant",
                    environment_name="egress",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=adopted.reconnect_metadata,
                )
            )
            assert reconnected.reconnect_metadata == expected_metadata
        finally:
            await original.environment.runner.close()
            if reconnected is not None:
                await reconnected.environment.runner.close()

    asyncio.run(scenario())
