"""Cross-process owner fencing between Lambda MicroVM runners and the sidecar."""

from __future__ import annotations

import asyncio
import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from tests.runners.lambda_microvm_harness import (
    ClientTokenLambdaModel,
    FakeLambdaClientError,
    SupervisorTransport,
)

import cayu.runners.aws_lambda_microvm as lambda_microvm_module
from cayu import ExecCommand, LambdaMicroVMRunner
from cayu.runners import (
    LambdaMicroVMLifecycleInProgress,
    LambdaMicroVMOwnershipSuperseded,
    LambdaMicroVMOwnershipUnverified,
)

_CLAIM_A = "a" * 64
_CLAIM_B = "b" * 64


@pytest.fixture
def sidecar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CAYU_MICROVM_WORKSPACE_ROOT", str(tmp_path))
    sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)
    module = importlib.import_module("examples.aws.lambda_microvm_sidecar.app")
    try:
        yield module
    finally:
        module.SUPERVISOR.cancel_all()
        sys.modules.pop("examples.aws.lambda_microvm_sidecar.app", None)


def _payload(tmp_path: Path, owner_claim: str | None, *argv: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "command_id": f"cmd-{len(argv)}-{owner_claim or 'none'}"[:64],
        "kind": "process",
        "argv": list(argv) or [sys.executable, "-c", "print('ok')"],
        "cwd": str(tmp_path),
        "env": {},
        "stdin_base64": None,
        "timeout_s": 30,
        "output_limit_bytes": 1024,
        # The host test platform lacks the agent namespace and setpriv tools.
        "execution_profile": "trusted",
    }
    if owner_claim is not None:
        payload["owner_claim"] = owner_claim
    return payload


def test_sidecar_rejects_commands_until_an_owner_claims(sidecar, tmp_path: Path) -> None:
    with pytest.raises(HTTPException) as raised:
        asyncio.run(sidecar.start_command(_payload(tmp_path, _CLAIM_A)))
    assert raised.value.status_code == sidecar.OWNER_SUPERSEDED_STATUS

    assert asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A})) == {
        "generation": 1,
        "superseded_previous": False,
    }
    accepted = asyncio.run(sidecar.start_command(_payload(tmp_path, _CLAIM_A)))
    assert accepted["state"] == "accepted"


def test_new_claim_fences_the_previous_owner_and_cancels_its_commands(
    sidecar, tmp_path: Path
) -> None:
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A}))
    running = _payload(tmp_path, _CLAIM_A, sys.executable, "-c", "import time; time.sleep(30)")
    asyncio.run(sidecar.start_command(dict(running)))

    claimed = asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_B}))

    assert claimed == {"generation": 2, "superseded_previous": True}
    assert sidecar.SUPERVISOR.get(running["command_id"])["state"] == "cancelled"
    with pytest.raises(HTTPException) as raised:
        asyncio.run(sidecar.start_command(_payload(tmp_path, _CLAIM_A)))
    assert raised.value.status_code == sidecar.OWNER_SUPERSEDED_STATUS
    assert asyncio.run(sidecar.check_owner({"claim_id": _CLAIM_A})) == {"current": False}
    assert asyncio.run(sidecar.check_owner({"claim_id": _CLAIM_B})) == {"current": True}


def test_repeated_claim_is_idempotent_and_does_not_cancel(sidecar, tmp_path: Path) -> None:
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A}))
    running = _payload(tmp_path, _CLAIM_A, sys.executable, "-c", "import time; time.sleep(30)")
    asyncio.run(sidecar.start_command(dict(running)))
    # The supervisor starts the process on its own thread; wait for it so the
    # assertion below observes the re-claim, not the start race.
    _wait_for_state(sidecar, running["command_id"], "running")

    assert asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A})) == {
        "generation": 1,
        "superseded_previous": False,
    }
    assert sidecar.SUPERVISOR.get(running["command_id"])["state"] == "running"


def _wait_for_state(sidecar, command_id: str, state: str, timeout_s: float = 10) -> None:
    deadline = time.monotonic() + timeout_s
    while sidecar.SUPERVISOR.get(command_id)["state"] != state:
        assert time.monotonic() < deadline, f"command never reached {state}"
        time.sleep(0.01)


@pytest.mark.parametrize("claim", [None, "", "short", "x" * 129, "é" * 40, 42])
def test_malformed_claims_are_rejected(sidecar, claim: Any) -> None:
    with pytest.raises(HTTPException) as raised:
        asyncio.run(sidecar.claim_owner({"claim_id": claim}))
    assert raised.value.status_code == 400
    assert asyncio.run(sidecar.check_owner({"claim_id": claim})) == {"current": False}


def test_new_claim_resets_the_agent_proxy_relay(sidecar) -> None:
    class Relay:
        def __init__(self, host: str, port: int) -> None:
            self.target = (host, port)
            self.proxy_url = "http://192.0.2.1:18080"
            self.closed = False

        def close(self) -> None:
            self.closed = True

    boundary = sidecar.SUPERVISOR.execution_boundary
    boundary.agent_netns = "cayu-agent"
    boundary._relay_factory = Relay
    boundary.environment_for({"HTTPS_PROXY": "http://10.0.1.20:9443"}, execution_profile="agent")
    first = boundary._relay

    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A}))
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_B}))
    boundary.environment_for({"HTTPS_PROXY": "http://10.0.2.30:9555"}, execution_profile="agent")

    assert first.closed
    assert boundary._relay.target == ("10.0.2.30", 9555)


async def _runner(model: ClientTokenLambdaModel, transport: SupervisorTransport, microvm_id=None):
    if microvm_id is None:
        return await LambdaMicroVMRunner.create(
            model.image_arn,
            client=model,
            endpoint_transport=transport,
            default_cwd=str(transport.supervisor.root),
            poll_interval_s=0,
            close_action="none",
        )
    return await LambdaMicroVMRunner.from_existing(
        microvm_id,
        client=model,
        endpoint_transport=transport,
        default_cwd=str(transport.supervisor.root),
        poll_interval_s=0,
        close_action="none",
    )


@pytest.mark.anyio
async def test_stale_runner_cannot_execute_after_a_successor_attaches(tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    assert (await stale.exec(ExecCommand.process("python3", "-c", "print(1)"))).exit_code == 0

    successor = await _runner(model, sidecar, stale.microvm_id)

    with pytest.raises(LambdaMicroVMOwnershipSuperseded):
        await stale.exec(ExecCommand.process("python3", "-c", "print(2)"))
    with pytest.raises(RuntimeError, match="poisoned|closed"):
        await stale.exec(ExecCommand.process("python3", "-c", "print(3)"))
    result = await successor.exec(ExecCommand.process("python3", "-c", "print('successor')"))
    assert result.stdout == "successor\n"
    await successor.close()


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["terminate", "suspend"])
async def test_stale_runner_cannot_mutate_its_successors_microvm(
    action: str, tmp_path: Path
) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    successor = await _runner(model, sidecar, stale.microvm_id)

    stale.close_action = action
    with pytest.raises(LambdaMicroVMOwnershipSuperseded):
        await stale.close()

    assert model.microvms[stale.microvm_id]["state"] == "RUNNING"
    assert model.terminate_calls == []
    successor.close_action = "terminate"
    await successor.close()
    assert model.terminate_calls == [stale.microvm_id]


@pytest.mark.anyio
async def test_stale_runner_resume_does_not_reclaim(tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    await stale.suspend()
    model.microvms[stale.microvm_id]["state"] = "SUSPENDED"
    successor = await _runner(model, sidecar, stale.microvm_id)

    with pytest.raises(BaseException) as raised:
        await stale.resume()

    assert any(
        isinstance(item, LambdaMicroVMOwnershipSuperseded) for item in _exception_tree(raised.value)
    )
    result = await successor.exec(ExecCommand.process("python3", "-c", "print('still-mine')"))
    assert result.stdout == "still-mine\n"
    await successor.close()


@pytest.mark.anyio
@pytest.mark.parametrize("successor_claimed", [False, True])
async def test_unreachable_sidecar_refuses_mutation_and_leaves_cleanup_pending(
    successor_claimed: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    runner = await _runner(model, sidecar)
    successor = await _runner(model, sidecar, runner.microvm_id) if successor_claimed else None

    async def unreachable(**_kwargs: Any) -> dict[str, Any]:
        raise lambda_microvm_module.LambdaMicroVMEndpointTransientError("unreachable")

    monkeypatch.setattr(sidecar, "check_owner", unreachable)
    monkeypatch.setattr(sidecar, "acquire_lifecycle", unreachable)
    runner.close_action = "terminate"
    with pytest.raises(LambdaMicroVMOwnershipUnverified):
        await runner.close()

    assert model.terminate_calls == []
    assert model.microvms[runner.microvm_id]["state"] == "RUNNING"
    if successor is not None:
        monkeypatch.undo()
        result = await successor.exec(ExecCommand.process("python3", "-c", "print('mine')"))
        assert result.stdout == "mine\n"
        await successor.close()
    else:
        # The authoritative route (the runtime's allocation reap under the
        # durable allocation fence) still disposes the MicroVM.
        await lambda_microvm_module.terminate_microvm_confirmed(
            model, runner.microvm_id, timeout_s=5, poll_interval_s=0
        )
        assert model.microvms[runner.microvm_id]["state"] == "TERMINATED"


def _exception_tree(error: BaseException):
    yield error
    if isinstance(error, BaseExceptionGroup):
        for item in error.exceptions:
            yield from _exception_tree(item)
    if error.__cause__ is not None:
        yield from _exception_tree(error.__cause__)
    if error.__context__ is not None and error.__context__ is not error.__cause__:
        yield from _exception_tree(error.__context__)


@pytest.mark.anyio
async def test_in_flight_command_of_stale_runner_reports_supersession(tmp_path: Path) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    in_flight = asyncio.create_task(
        stale.exec(ExecCommand.process("python3", "-c", "import time; time.sleep(30)"))
    )
    while not sidecar.payloads:
        await asyncio.sleep(0.01)

    successor = await _runner(model, sidecar, stale.microvm_id)

    with pytest.raises(LambdaMicroVMOwnershipSuperseded, match="cancelled this command"):
        await asyncio.wait_for(in_flight, 10)
    result = await successor.exec(ExecCommand.process("python3", "-c", "print('next')"))
    assert result.stdout == "next\n"
    await successor.close()


@pytest.mark.anyio
@pytest.mark.parametrize("action", ["suspend", "terminate", "close-suspend", "close-terminate"])
async def test_stale_runner_cannot_mutate_without_a_prior_rejection(
    action: str, tmp_path: Path
) -> None:
    # The stale runner has never seen a rejection, so only the sidecar's
    # lifecycle lease can stop it.
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    successor = await _runner(model, sidecar, stale.microvm_id)

    with pytest.raises(LambdaMicroVMOwnershipSuperseded):
        if action.startswith("close-"):
            stale.close_action = action.removeprefix("close-")
            await stale.close()
        else:
            await getattr(stale, action)()

    assert model.suspend_calls == []
    assert model.terminate_calls == []
    assert model.microvms[stale.microvm_id]["state"] == "RUNNING"
    result = await successor.exec(ExecCommand.process("python3", "-c", "print('successor')"))
    assert result.stdout == "successor\n"
    successor.close_action = "terminate"
    await successor.close()
    assert model.terminate_calls == [stale.microvm_id]


@pytest.mark.anyio
async def test_stale_runner_cannot_terminate_a_microvm_its_successor_suspended(
    tmp_path: Path,
) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    await stale.suspend()
    successor = await _runner(model, sidecar, stale.microvm_id)
    await successor.suspend()

    with pytest.raises(BaseException) as raised:
        await stale.terminate()

    assert any(
        isinstance(item, LambdaMicroVMOwnershipSuperseded) for item in _exception_tree(raised.value)
    )
    assert model.terminate_calls == []
    await successor.resume()
    result = await successor.exec(ExecCommand.process("python3", "-c", "print('kept')"))
    assert result.stdout == "kept\n"
    await successor.close()


@pytest.mark.anyio
async def test_lifecycle_lease_is_not_used_after_its_dispatch_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    runner = await _runner(model, sidecar)
    monkeypatch.setattr(lambda_microvm_module, "_LAMBDA_LIFECYCLE_DISPATCH_WINDOW_SECONDS", -1.0)

    with pytest.raises(LambdaMicroVMOwnershipUnverified, match="lifecycle lease"):
        await runner.terminate()

    assert model.terminate_calls == []
    # Nothing was sent, so the lease was released and another host may attach.
    successor = await _runner(model, sidecar, runner.microvm_id)
    await successor.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "released"),
    [
        (lambda: FakeLambdaClientError("ValidationException", "rejected"), True),
        (lambda: RuntimeError("connection reset after send"), False),
    ],
    ids=["definitive", "ambiguous"],
)
async def test_lease_is_released_only_after_a_definitive_provider_rejection(
    failure, released: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    runner = await _runner(model, sidecar)

    def failing_terminate(**_kwargs: Any) -> dict[str, Any]:
        raise failure()

    monkeypatch.setattr(model, "terminate_microvm", failing_terminate)
    with pytest.raises(Exception):
        await runner.terminate()

    if released:
        successor = await _runner(model, sidecar, runner.microvm_id)
        await successor.close()
    else:
        # An ambiguous failure may still be applied, so no other host may claim
        # until the guest's lifecycle hook settles the lease.
        with pytest.raises(LambdaMicroVMLifecycleInProgress):
            await sidecar.claim_owner(claim_id="c" * 64)


def test_lifecycle_send_guard_refuses_late_signed_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    boto3 = pytest.importorskip("boto3")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    client = boto3.client(
        "lambda-microvms", region_name="us-east-1", endpoint_url="http://127.0.0.1:9"
    )
    assert lambda_microvm_module._install_lifecycle_send_guard(client)
    dispatch = lambda_microvm_module._LifecycleDispatch(deadline=time.monotonic() - 1)
    lambda_microvm_module._LIFECYCLE_DISPATCH.value = dispatch
    try:
        with pytest.raises(LambdaMicroVMOwnershipUnverified):
            client.terminate_microvm(microvmIdentifier="microvm-late")
    finally:
        lambda_microvm_module._LIFECYCLE_DISPATCH.value = None
    assert dispatch.sent == 0


class _RawBody:
    def stream(self, *_args: Any, **_kwargs: Any):
        yield b"{}"


def _botocore_model(
    model: ClientTokenLambdaModel,
    monkeypatch: pytest.MonkeyPatch,
    send: Any,
    operation: str,
) -> Any:
    """Route one lifecycle operation of the model through real botocore.

    Signing, the retry handler, and event hooks are botocore's own; ``send``
    replaces only the HTTP transport. Standard retries are enabled, so any
    retry the runner fails to prevent would reach ``send``.
    """

    boto3 = pytest.importorskip("boto3")
    from botocore.config import Config

    monkeypatch.setattr("botocore.retries.standard.ExponentialBackoff.delay_amount", lambda *_: 0)
    client = boto3.Session(
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="example-secret",
        region_name="us-east-1",
    ).client(
        "lambda-microvms",
        endpoint_url="https://lambda.invalid",
        config=Config(retries={"mode": "standard", "max_attempts": 3}),
    )
    client._endpoint.http_session.send = send
    monkeypatch.setattr(model, "meta", client.meta, raising=False)
    monkeypatch.setattr(model, operation, getattr(client, operation))
    return client


def _ok_response(request: Any) -> Any:
    from botocore.awsrequest import AWSResponse

    return AWSResponse(request.url, 200, {"Content-Type": "application/json"}, _RawBody())


def _rejection_response(request: Any) -> Any:
    from botocore.awsrequest import AWSResponse

    class _Body:
        def stream(self, *_args: Any, **_kwargs: Any):
            yield b'{"message": "rejected"}'

    return AWSResponse(
        request.url,
        400,
        {"Content-Type": "application/json", "x-amzn-ErrorType": "ValidationException"},
        _Body(),
    )


@pytest.mark.anyio
@pytest.mark.parametrize("second_attempt", ["deadline_refused", "definitive_rejection"])
async def test_timed_out_lifecycle_request_keeps_the_lease_and_is_never_retried(
    second_attempt: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The reviewer's schedule: the first terminate is sent and times out while
    # still pending. Whether its retry would be refused at the deadline or
    # definitively rejected, it must not be sent, and the lease must stay
    # held until the pending request's effect is observed.
    pytest.importorskip("boto3")
    from botocore.exceptions import ReadTimeoutError

    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    actual_terminate = model.terminate_microvm
    sends: list[Any] = []

    def send(request: Any) -> Any:
        sends.append(request)
        if len(sends) == 1:
            if second_attempt == "deadline_refused":
                # Any retry would now be signed after the lease's window.
                monkeypatch.setattr(
                    lambda_microvm_module, "_LAMBDA_LIFECYCLE_DISPATCH_WINDOW_SECONDS", -1.0
                )
            raise ReadTimeoutError(endpoint_url=request.url)
        return _rejection_response(request)

    client = _botocore_model(model, monkeypatch, send, "terminate_microvm")
    try:
        with pytest.raises(Exception):
            await stale.terminate()

        assert len(sends) == 1
        with pytest.raises(LambdaMicroVMLifecycleInProgress):
            await sidecar.claim_owner(claim_id="c" * 64)
        # The pending request lands late; its hook settles the lease.
        actual_terminate(microvmIdentifier=stale.microvm_id)
        assert model.microvms[stale.microvm_id]["state"] == "TERMINATING"
        assert (await sidecar.claim_owner(claim_id="c" * 64))["generation"] == 2
    finally:
        client.close()


@pytest.mark.anyio
async def test_lost_suspend_acknowledgement_is_never_retried_into_a_successor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first suspend is applied and fires the guest hook, but its response
    # is lost. A successor resumes and claims the MicroVM before the old call
    # finishes; the old SDK call must not retry into the successor.
    pytest.importorskip("boto3")
    from botocore.exceptions import ReadTimeoutError

    model = ClientTokenLambdaModel()
    sidecar = SupervisorTransport(tmp_path)
    stale = await _runner(model, sidecar)
    actual_suspend = model.suspend_microvm
    applied = threading.Event()
    release = threading.Event()
    sends: list[Any] = []

    def send(request: Any) -> Any:
        sends.append(request)
        actual_suspend(microvmIdentifier=stale.microvm_id)
        if len(sends) == 1:
            applied.set()
            assert release.wait(10)
            raise ReadTimeoutError(endpoint_url=request.url)
        return _ok_response(request)

    client = _botocore_model(model, monkeypatch, send, "suspend_microvm")
    try:
        task = asyncio.create_task(stale.suspend())
        assert await asyncio.to_thread(applied.wait, 5)
        successor = await _runner(model, sidecar, stale.microvm_id)
        assert model.microvms[stale.microvm_id]["state"] == "RUNNING"
        release.set()
        with pytest.raises(Exception):
            await task

        assert len(sends) == 1
        assert model.microvms[stale.microvm_id]["state"] == "RUNNING"
        result = await successor.exec(ExecCommand.process("python3", "-c", "print('kept')"))
        assert result.stdout == "kept\n"
        await successor.close()
    finally:
        release.set()
        client.close()


@pytest.mark.anyio
async def test_unclaimed_constructor_does_not_dispose_of_a_token_shared_microvm(
    tmp_path: Path,
) -> None:
    # Creator A allocates with token T and stalls before claiming. Creator B
    # replays T, receives the same MicroVM, and claims it. A's readiness then
    # fails; A never owned the MicroVM and must not terminate B's.
    model = ClientTokenLambdaModel()
    entered = asyncio.Event()
    release = asyncio.Event()

    class FailingBeforeClaim(SupervisorTransport):
        async def health(self, **kwargs: Any) -> dict[str, Any]:
            entered.set()
            await release.wait()
            raise OSError("host A cannot reach the sidecar")

    first = FailingBeforeClaim(tmp_path)
    good = SupervisorTransport(tmp_path)

    async def create(transport: SupervisorTransport) -> LambdaMicroVMRunner:
        return await LambdaMicroVMRunner.create(
            model.image_arn,
            client=model,
            endpoint_transport=transport,
            default_cwd=str(tmp_path),
            poll_interval_s=0,
            close_action="none",
            client_token="c" * 64,
            ready_timeout_s=0.1,
        )

    stale = asyncio.create_task(create(first))
    await entered.wait()
    successor = await create(good)
    await asyncio.sleep(0.15)
    release.set()
    with pytest.raises(Exception) as raised:
        await stale

    assert model.terminate_calls == []
    assert model.suspend_calls == []
    assert model.microvms[successor.microvm_id]["state"] == "RUNNING"
    assert any(
        "cleanup is deferred" in note
        for item in _exception_tree(raised.value)
        for note in getattr(item, "__notes__", ())
    )
    result = await successor.exec(ExecCommand.process("python3", "-c", "print('successor')"))
    assert result.stdout == "successor\n"
    successor.close_action = "terminate"
    await successor.close()
    first.supervisor.cancel_all()


def test_owner_check_and_command_registration_are_atomic_with_takeover(
    sidecar, tmp_path: Path
) -> None:
    # The reviewer's barrier: owner A's start has passed admission and is held
    # immediately before registration when owner B claims.
    async def scenario() -> None:
        await sidecar.claim_owner({"claim_id": _CLAIM_A})
        entered = threading.Event()
        release = threading.Event()
        original = sidecar.SUPERVISOR.start

        def delayed(*args: Any, **kwargs: Any) -> dict[str, Any]:
            entered.set()
            assert release.wait(10)
            return original(*args, **kwargs)

        sidecar.SUPERVISOR.start = delayed
        marker = tmp_path / "stale-command-ran"
        stale = _payload(
            tmp_path,
            _CLAIM_A,
            sys.executable,
            "-c",
            f"import time; from pathlib import Path; time.sleep(0.5); "
            f"Path({str(marker)!r}).write_text('stale')",
        )
        start = asyncio.create_task(sidecar.start_command(dict(stale)))
        assert await asyncio.to_thread(entered.wait, 5)
        claim = asyncio.create_task(sidecar.claim_owner({"claim_id": _CLAIM_B}))
        await asyncio.sleep(0.3)
        assert not claim.done(), "takeover completed while an admitted start was unregistered"

        release.set()
        sidecar.SUPERVISOR.start = original
        assert (await start)["state"] == "accepted"
        assert await claim == {"generation": 2, "superseded_previous": True}
        # The takeover returned only after cancelling the stale command.
        cancelled = sidecar.SUPERVISOR.get(stale["command_id"])
        assert cancelled["state"] == "cancelled"
        assert cancelled["cancel_reason"] == sidecar.OWNER_SUPERSEDED_REASON
        await asyncio.sleep(1.0)
        assert not marker.exists()

        fresh = _payload(tmp_path, _CLAIM_B, sys.executable, "-c", "print('successor')")
        fresh["command_id"] = "successor-command"
        assert (await sidecar.start_command(fresh))["state"] == "accepted"
        _wait_for_state(sidecar, "successor-command", "completed")

    asyncio.run(scenario())


def test_takeover_cancels_only_earlier_owners_commands(sidecar, tmp_path: Path) -> None:
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A}))
    older = _payload(tmp_path, _CLAIM_A, sys.executable, "-c", "import time; time.sleep(30)")
    asyncio.run(sidecar.start_command(dict(older)))
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_B}))
    newer = _payload(tmp_path, _CLAIM_B, sys.executable, "-c", "import time; time.sleep(30)")
    newer["command_id"] = "newer-command"
    asyncio.run(sidecar.start_command(dict(newer)))
    _wait_for_state(sidecar, "newer-command", "running")

    # A late cancellation for the generation-2 takeover must not touch generation 2.
    sidecar.SUPERVISOR.cancel_all(reason="owner_superseded", before_generation=2)

    assert sidecar.SUPERVISOR.get(older["command_id"])["state"] == "cancelled"
    assert sidecar.SUPERVISOR.get("newer-command")["state"] == "running"


def test_lifecycle_lease_blocks_takeover_and_commands_until_released(
    sidecar, tmp_path: Path
) -> None:
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A}))
    lease = asyncio.run(sidecar.acquire_lifecycle({"claim_id": _CLAIM_A, "action": "suspend"}))
    assert lease["generation"] == 1 and lease["action"] == "suspend"

    for request in (
        lambda: sidecar.claim_owner({"claim_id": _CLAIM_B}),
        lambda: sidecar.start_command(_payload(tmp_path, _CLAIM_A)),
    ):
        with pytest.raises(HTTPException) as raised:
            asyncio.run(request())
        assert raised.value.status_code == sidecar.OWNER_LIFECYCLE_LEASED_STATUS
    assert asyncio.run(sidecar.check_owner({"claim_id": _CLAIM_A})) == {"current": True}

    asyncio.run(sidecar.release_lifecycle({"claim_id": _CLAIM_A}))
    assert asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_B}))["superseded_previous"]
    with pytest.raises(HTTPException) as raised:
        asyncio.run(sidecar.acquire_lifecycle({"claim_id": _CLAIM_A, "action": "terminate"}))
    assert raised.value.status_code == sidecar.OWNER_SUPERSEDED_STATUS


@pytest.mark.parametrize(
    ("action", "hook", "settles"),
    [
        ("suspend", "suspend_hook", True),
        ("suspend", "resume_hook", True),
        ("suspend", "terminate_hook", True),
        ("terminate", "suspend_hook", False),
        ("terminate", "resume_hook", False),
        ("terminate", "terminate_hook", True),
    ],
)
def test_guest_lifecycle_hooks_settle_only_a_matching_lease(
    sidecar, action: str, hook: str, settles: bool
) -> None:
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A}))
    asyncio.run(sidecar.acquire_lifecycle({"claim_id": _CLAIM_A, "action": action}))

    asyncio.run(getattr(sidecar, hook)())

    if settles:
        assert asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_B}))["superseded_previous"]
    else:
        with pytest.raises(HTTPException) as raised:
            asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_B}))
        assert raised.value.status_code == sidecar.OWNER_LIFECYCLE_LEASED_STATUS


def test_lifecycle_lease_never_expires_by_time(monkeypatch: pytest.MonkeyPatch) -> None:
    from examples.aws.lambda_microvm_sidecar import supervisor
    from examples.aws.lambda_microvm_sidecar.supervisor import (
        OwnerFence,
        OwnerLifecycleLeasedError,
    )

    fence = OwnerFence()
    fence.claim(_CLAIM_A)
    fence.acquire_lifecycle(_CLAIM_A, "terminate")
    # No elapsed interval proves an accepted request has finished.
    later = time.time() + 365 * 24 * 3600
    monkeypatch.setattr(supervisor.time, "time", lambda: later)
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: later)
    with pytest.raises(OwnerLifecycleLeasedError):
        fence.claim(_CLAIM_B)
    fence.settle_lifecycle("terminate")
    assert fence.claim(_CLAIM_B) == (2, True)


@pytest.mark.parametrize("action", [None, "resume", "delete"])
def test_lifecycle_lease_rejects_unknown_actions(sidecar, action: Any) -> None:
    asyncio.run(sidecar.claim_owner({"claim_id": _CLAIM_A}))
    with pytest.raises(HTTPException) as raised:
        asyncio.run(sidecar.acquire_lifecycle({"claim_id": _CLAIM_A, "action": action}))
    assert raised.value.status_code == 400
