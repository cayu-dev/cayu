from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from cayu.egress import (
    EgressAdapterRegistry,
    EgressBinding,
    HttpEgressPolicy,
    SandboxEgressAdapter,
    TransparentEgressBroker,
    UnsupportedEgressAdapter,
    UnsupportedEgressError,
    VirtualCredentialGrant,
    VirtualCredentialRegistry,
)
from cayu.vaults import StaticVault


def _broker() -> TransparentEgressBroker:
    return TransparentEgressBroker(
        registry=VirtualCredentialRegistry(),
        resolver=StaticVault({"stripe_test_key": "sk_test_real"}),
        policies={
            "provider-example": HttpEgressPolicy(
                name="provider-example",
                allowed_hosts=["api.stripe.com"],
                allowed_endpoints=[("POST", "/v1/customers")],
            )
        },
    )


def test_unregistered_runner_resolves_to_fail_closed_adapter() -> None:
    registry = EgressAdapterRegistry()

    for kind in ("local", "future-runner"):
        adapter = registry.resolve(kind)
        assert isinstance(adapter, UnsupportedEgressAdapter)


def test_prepare_on_unsupported_adapter_raises() -> None:
    adapter = UnsupportedEgressAdapter("local")

    with pytest.raises(UnsupportedEgressError, match="cannot enforce virtual egress"):
        asyncio.run(adapter.prepare(session_id="sess_1", grants=(), broker=_broker()))


def test_error_message_names_no_downgrade() -> None:
    adapter = UnsupportedEgressAdapter("e2b")

    with pytest.raises(UnsupportedEgressError, match="refuse to downgrade to raw secret injection"):
        asyncio.run(adapter.prepare(session_id="sess_1", grants=(), broker=_broker()))


class _FakeEnforcedAdapter(SandboxEgressAdapter):
    runner_kind = "fake"

    def __init__(self) -> None:
        self.torn_down = 0

    async def prepare(
        self,
        *,
        session_id: str,
        grants: Sequence[VirtualCredentialGrant],
        broker: TransparentEgressBroker,
    ) -> EgressBinding:
        async def teardown() -> None:
            self.torn_down += 1

        return EgressBinding(
            env={"HTTPS_PROXY": "http://cayu-egress:8080"},
            ca_cert_pem=b"-----BEGIN CERTIFICATE-----\n",
            teardown=teardown,
        )

    async def create_runner(self, request):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def test_registered_adapter_is_used() -> None:
    registry = EgressAdapterRegistry()
    adapter = _FakeEnforcedAdapter()
    registry.register(adapter)

    resolved = registry.resolve("fake")
    assert resolved is adapter

    binding = asyncio.run(resolved.prepare(session_id="sess_1", grants=(), broker=_broker()))
    assert binding.env["HTTPS_PROXY"] == "http://cayu-egress:8080"


def test_binding_close_is_idempotent() -> None:
    adapter = _FakeEnforcedAdapter()

    async def run() -> None:
        binding = await adapter.prepare(session_id="sess_1", grants=(), broker=_broker())
        await binding.close()
        await binding.close()

    asyncio.run(run())
    assert adapter.torn_down == 1


def test_binding_close_does_not_mark_closed_when_teardown_is_cancelled() -> None:
    calls = {"count": 0}

    async def teardown() -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise asyncio.CancelledError()

    async def run() -> None:
        binding = EgressBinding(teardown=teardown)
        with pytest.raises(asyncio.CancelledError):
            await binding.close()
        await binding.close()
        await binding.close()

    asyncio.run(run())
    assert calls["count"] == 2


def test_binding_close_finishes_teardown_before_reporting_caller_cancellation() -> None:
    async def run() -> tuple[list[str], bool]:
        started = asyncio.Event()
        finish = asyncio.Event()
        order: list[str] = []

        async def teardown() -> None:
            order.append("started")
            started.set()
            await finish.wait()
            order.append("finished")

        binding = EgressBinding(teardown=teardown, teardown_timeout_s=1)
        close_task = asyncio.create_task(binding.close())
        await started.wait()
        close_task.cancel()
        await asyncio.sleep(0)
        assert close_task.done() is False
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        await binding.close()
        return order, binding._closed

    order, closed = asyncio.run(run())
    assert order == ["started", "finished"]
    assert closed is True


def test_binding_close_timeout_is_bounded_and_retryable() -> None:
    async def run() -> tuple[int, bool]:
        started = asyncio.Event()
        finish = asyncio.Event()
        calls = 0

        async def teardown() -> None:
            nonlocal calls
            calls += 1
            started.set()
            await finish.wait()

        binding = EgressBinding(teardown=teardown, teardown_timeout_s=0.01)
        with pytest.raises(TimeoutError, match="did not complete"):
            await binding.close()
        assert started.is_set()
        assert binding._closed is False
        finish.set()
        await binding.close()
        await binding.close()
        return calls, binding._closed

    calls, closed = asyncio.run(run())
    assert calls == 1
    assert closed is True


def test_binding_validates_typed_core_fields() -> None:
    with pytest.raises(ValueError, match="network"):
        EgressBinding(network=" ")
    with pytest.raises(ValueError, match="proxy_port"):
        EgressBinding(proxy_port=0)
    with pytest.raises(ValueError, match="proxy_url"):
        EgressBinding(proxy_url="https://cayu-egress.example:8443")
    with pytest.raises(ValueError, match="teardown_timeout_s"):
        EgressBinding(teardown_timeout_s=0)

    binding = EgressBinding(proxy_url="http://cayu-egress.example:8443")
    assert binding.proxy_url == "http://cayu-egress.example:8443"


def test_registry_rejects_non_adapter() -> None:
    registry = EgressAdapterRegistry()
    bad_adapter: Any = object()

    with pytest.raises(TypeError):
        registry.register(bad_adapter)


def test_registry_rejects_blank_adapter_runner_kind() -> None:
    class _BlankKindAdapter(_FakeEnforcedAdapter):
        runner_kind = " "

    registry = EgressAdapterRegistry()

    with pytest.raises(ValueError, match="runner_kind"):
        registry.register(_BlankKindAdapter())
