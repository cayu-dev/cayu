"""Caller epochs reject late owner claims without touching the accepted owner."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from examples.aws.lambda_microvm_sidecar.supervisor import (
    OwnerFence,
    OwnerLifecycleLeasedError,
    StaleOwnerEpochError,
)
from fastapi import HTTPException
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel, SupervisorTransport
from tests.runners.test_aws_lambda_microvm_owner_fence import _payload, _wait_for_state
from tests.runners.test_aws_lambda_microvm_owner_fence import sidecar as sidecar

from cayu.runners import (
    ExecCommand,
    HttpxLambdaMicroVMEndpointTransport,
    LambdaMicroVMProtocolError,
    LambdaMicroVMRunner,
    LambdaMicroVMStaleOwnerEpoch,
)
from cayu.runners.aws_lambda_microvm import LAMBDA_MICROVM_PROTOCOL_VERSION

A, B, C = ("a" * 64, "b" * 64, "c" * 64)


@pytest.mark.anyio
async def test_late_claim_cannot_interrupt_the_newer_owner(sidecar, tmp_path, monkeypatch):
    pending = asyncio.Event()
    release = asyncio.Event()

    async def delayed_claim():
        pending.set()
        await release.wait()
        return await sidecar.claim_owner({"claim_id": A, "owner_epoch": 1})

    old = asyncio.create_task(delayed_claim())
    await pending.wait()
    accepted = await sidecar.claim_owner({"claim_id": B, "owner_epoch": 2})
    command = _payload(tmp_path, B, sys.executable, "-c", "import time; time.sleep(30)")
    await sidecar.start_command(command)
    await asyncio.to_thread(_wait_for_state, sidecar, command["command_id"], "running")
    touched = []
    monkeypatch.setattr(sidecar, "_close_guest_relays", lambda: touched.append("relay"))
    release.set()
    with pytest.raises(HTTPException) as caught:
        await old
    assert caught.value.status_code == 409
    assert touched == []
    assert (await sidecar.check_owner({"claim_id": B}))["current"] is True
    assert sidecar.SUPERVISOR.get(command["command_id"])["state"] == "running"
    assert await sidecar.claim_owner({"claim_id": B, "owner_epoch": 2}) == {
        "generation": accepted["generation"],
        "superseded_previous": False,
    }


def test_equal_and_unnumbered_claims_keep_legacy_behavior_without_lowering_maximum():
    owner = OwnerFence()
    assert owner.claim(A, owner_epoch=10) == (1, False)
    assert owner.claim(B, owner_epoch=10) == (2, True)
    assert owner.claim(C) == (3, True)
    with pytest.raises(StaleOwnerEpochError):
        owner.claim(A, owner_epoch=9)
    assert owner.is_current(C)
    assert owner.claim(C, owner_epoch=11) == (3, False)
    with pytest.raises(StaleOwnerEpochError):
        owner.claim(C, owner_epoch=10)
    assert owner.is_current(C)


def test_refused_lifecycle_claim_does_not_advance_epoch():
    owner = OwnerFence()
    owner.claim(A, owner_epoch=1)
    owner.acquire_lifecycle(A, "suspend")
    with pytest.raises(OwnerLifecycleLeasedError):
        owner.claim(B, owner_epoch=100)
    owner.release_lifecycle(A)
    assert owner.claim(B, owner_epoch=2) == (2, True)


def test_stale_claim_does_not_retry_pending_takeover_cleanup():
    owner = OwnerFence()
    owner.claim(A, owner_epoch=1)
    calls = []

    def pending_cleanup():
        calls.append("cleanup")
        raise RuntimeError("cleanup pending")

    with pytest.raises(RuntimeError, match="cleanup pending"):
        owner.claim(B, owner_epoch=3, on_supersede=pending_cleanup)
    with pytest.raises(StaleOwnerEpochError):
        owner.claim(C, owner_epoch=2)
    assert calls == ["cleanup"]


@pytest.mark.parametrize("invalid", [-1, True, False, 1.5, "2", {}, []])
def test_sidecar_rejects_invalid_epochs_without_changing_owner(sidecar, invalid):
    asyncio.run(sidecar.claim_owner({"claim_id": A, "owner_epoch": 1}))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(sidecar.claim_owner({"claim_id": B, "owner_epoch": invalid}))
    assert caught.value.status_code == 400
    assert sidecar.OWNER.is_current(A)


@pytest.mark.anyio
async def test_runner_creation_and_attach_send_epoch_and_refuse_stale_attach(tmp_path: Path):
    model = ClientTokenLambdaModel()
    endpoint = SupervisorTransport(tmp_path)
    options: dict[str, Any] = dict(
        client=model,
        endpoint_transport=endpoint,
        default_cwd=str(tmp_path),
        poll_interval_s=0,
        close_action="none",
    )
    first = await LambdaMicroVMRunner.create("image", owner_epoch=4, **options)
    successor = await LambdaMicroVMRunner.from_existing(first.microvm_id, owner_epoch=6, **options)
    with pytest.raises(LambdaMicroVMStaleOwnerEpoch):
        await LambdaMicroVMRunner.from_existing(first.microvm_id, owner_epoch=5, **options)
    assert model.terminate_calls == []
    result = await successor.exec(ExecCommand.process(sys.executable, "-c", "print('current')"))
    assert result.stdout == "current\n"
    await first.close()
    await successor.close()


@pytest.mark.anyio
@pytest.mark.parametrize("capabilities", [None, [], "owner_epoch", ["other"]])
async def test_direct_runner_checks_epoch_capability_before_claim(
    tmp_path, monkeypatch, capabilities
):
    endpoint = SupervisorTransport(tmp_path)

    async def health(**kwargs):
        return {
            "status": "ok",
            "protocol_version": LAMBDA_MICROVM_PROTOCOL_VERSION,
            "capabilities": capabilities,
        }

    monkeypatch.setattr(endpoint, "health", health)
    runner = LambdaMicroVMRunner(
        ClientTokenLambdaModel(),
        microvm_id="vm",
        endpoint="https://example.test",
        endpoint_transport=endpoint,
        owner_epoch=1,
    )
    with pytest.raises(LambdaMicroVMProtocolError, match="owner_epoch"):
        await runner.exec(ExecCommand.process("echo", "hello"))
    assert endpoint.owner_fence._generation == 0
    await runner.close()


@pytest.mark.anyio
async def test_http_claim_transmits_epoch_and_exposes_distinct_refusal():
    requests = []

    def reply(request):
        requests.append(json.loads(request.content))
        return httpx.Response(409, text="untrusted response body")

    client = httpx.AsyncClient(transport=httpx.MockTransport(reply))

    class ClientOwner:
        def get(self):
            return client

        async def aclose(self):
            await client.aclose()

    transport = HttpxLambdaMicroVMEndpointTransport()
    transport._client = ClientOwner()
    try:
        for epoch in (None, 0, 12):
            with pytest.raises(LambdaMicroVMStaleOwnerEpoch) as caught:
                await transport.claim_owner(
                    endpoint="https://example.test",
                    token="token",
                    claim_id=A,
                    timeout_s=1,
                    owner_epoch=epoch,
                )
            assert "untrusted" not in str(caught.value)
        assert requests == [
            {"claim_id": A},
            {"claim_id": A, "owner_epoch": 0},
            {"claim_id": A, "owner_epoch": 12},
        ]
    finally:
        await transport.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize("invalid", [-1, True, 1.5, "2"])
async def test_invalid_epoch_is_refused_before_allocating_or_attaching(invalid):
    class NeverCallProvider:
        def __getattr__(self, name):
            raise AssertionError("provider must not be called")

    for method, identifier in (
        (LambdaMicroVMRunner.create, "image"),
        (LambdaMicroVMRunner.from_existing, "vm"),
    ):
        with pytest.raises(ValueError, match="owner_epoch"):
            await method(identifier, client=NeverCallProvider(), owner_epoch=invalid)
