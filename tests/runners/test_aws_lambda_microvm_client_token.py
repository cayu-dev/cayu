"""Deterministic faults for idempotent Lambda MicroVM allocation by client token."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from tests.runners.lambda_microvm_harness import (
    ClientTokenLambdaModel,
    FakeLambdaClientError,
    OwnerFencedTransport,
)

import cayu.runners.aws_lambda_microvm as lambda_microvm_module
from cayu import LambdaMicroVMRunner
from cayu.runners import (
    LambdaMicroVMAllocationTerminated,
    LambdaMicroVMClientTokenConflict,
    LambdaMicroVMError,
    LambdaMicroVMSubmissionClosed,
)
from cayu.runners.aws_lambda_microvm import (
    lambda_microvm_run_options,
    run_microvm_with_client_token,
    terminate_microvm_confirmed,
)

IMAGE = ClientTokenLambdaModel.image_arn
TOKEN = "cayu-" + "a" * 64


class HealthyTransport(OwnerFencedTransport):
    def __init__(self) -> None:
        self.health_calls = 0

    async def health(self, **_kwargs: Any) -> dict[str, str]:
        self.health_calls += 1
        return {
            "status": "ok",
            "protocol_version": lambda_microvm_module.LAMBDA_MICROVM_PROTOCOL_VERSION,
        }

    async def start_command(self, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("allocation tests never dispatch guest work")

    get_command = cancel_command = start_command


@pytest.fixture(autouse=True)
def _no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        lambda_microvm_module, "_LAMBDA_CLIENT_TOKEN_RETRY_DELAYS_SECONDS", (0.0, 0.0, 0.0)
    )


async def _create(model: ClientTokenLambdaModel, **options: Any) -> LambdaMicroVMRunner:
    return await LambdaMicroVMRunner.create(
        IMAGE,
        client=model,
        endpoint_transport=HealthyTransport(),
        image_version="3",
        maximum_duration_in_seconds=600,
        client_token=TOKEN,
        close_action="none",
        poll_interval_s=0,
        **options,
    )


@pytest.mark.anyio
async def test_lost_acknowledgement_replays_same_token_and_adopts_one_allocation() -> None:
    model = ClientTokenLambdaModel()
    model.lose_acknowledgements = 2

    runner = await _create(model)

    assert model.created_ids() == [runner.microvm_id]
    assert [call["clientToken"] for call in model.run_calls] == [TOKEN] * 3
    assert len({repr(sorted(call.items())) for call in model.run_calls}) == 1
    await runner.close()


@pytest.mark.anyio
async def test_throttling_retries_are_bounded_and_never_change_the_token() -> None:
    model = ClientTokenLambdaModel()
    model.run_failures = [FakeLambdaClientError("ThrottlingException", "slow down")] * 4

    with pytest.raises(FakeLambdaClientError, match="ThrottlingException"):
        await _create(model)

    assert len(model.run_calls) == 4
    assert {call["clientToken"] for call in model.run_calls} == {TOKEN}
    assert model.created_ids() == []


@pytest.mark.anyio
async def test_nontransient_failure_is_not_retried() -> None:
    model = ClientTokenLambdaModel()
    model.run_failures = [FakeLambdaClientError("AccessDeniedException", "denied")]

    with pytest.raises(FakeLambdaClientError, match="AccessDeniedException"):
        await _create(model)

    assert len(model.run_calls) == 1


@pytest.mark.anyio
async def test_changed_parameters_raise_conflict_without_creating_a_replacement() -> None:
    model = ClientTokenLambdaModel()
    first = await _create(model)
    await first.close()

    with pytest.raises(LambdaMicroVMClientTokenConflict):
        await LambdaMicroVMRunner.create(
            IMAGE,
            client=model,
            endpoint_transport=HealthyTransport(),
            image_version="3",
            maximum_duration_in_seconds=601,
            client_token=TOKEN,
            poll_interval_s=0,
        )

    assert model.created_ids() == [first.microvm_id]
    assert len(model.run_calls) == 2


@pytest.mark.anyio
async def test_replay_reads_current_state_instead_of_stale_acknowledgement() -> None:
    model = ClientTokenLambdaModel()
    first = await _create(model)
    model.suspend_microvm(microvmIdentifier=first.microvm_id)

    adopted = await _create(model)

    assert adopted.microvm_id == first.microvm_id
    assert model.microvms[first.microvm_id]["state"] == "RUNNING"
    assert model.created_ids() == [first.microvm_id]
    await adopted.close()


@pytest.mark.anyio
@pytest.mark.parametrize("state", ["TERMINATING", "TERMINATED", "FAILED"])
async def test_replay_of_terminated_allocation_fails_closed(state: str) -> None:
    model = ClientTokenLambdaModel()
    first = await _create(model)
    await first.close()
    model.microvms[first.microvm_id]["state"] = state

    with pytest.raises(LambdaMicroVMAllocationTerminated) as raised:
        await _create(model)

    assert raised.value.microvm_id == first.microvm_id
    assert raised.value.state == state
    assert model.created_ids() == [first.microvm_id]
    assert model.terminate_calls == []


@pytest.mark.anyio
async def test_concurrent_recovery_workers_converge_on_one_allocation() -> None:
    model = ClientTokenLambdaModel()

    runners = await asyncio.gather(*(_create(model) for _ in range(4)))

    assert {runner.microvm_id for runner in runners} == set(model.created_ids())
    assert len(model.created_ids()) == 1
    for runner in runners:
        await runner.close()


@pytest.mark.anyio
async def test_image_version_drift_is_a_protocol_error() -> None:
    model = ClientTokenLambdaModel()
    first = await _create(model)
    await first.close()
    model.microvms[first.microvm_id]["imageVersion"] = "4"

    with pytest.raises(LambdaMicroVMError, match="image version"):
        await _create(model)


@pytest.mark.anyio
@pytest.mark.parametrize("token", ["", " ", "x" * 129, "café"])
async def test_invalid_client_token_rejected_before_provider_call(token: str) -> None:
    model = ClientTokenLambdaModel()

    with pytest.raises(ValueError, match="client_token"):
        await LambdaMicroVMRunner.create(IMAGE, client=model, client_token=token)
    with pytest.raises(ValueError, match="client_token"):
        await run_microvm_with_client_token(model, {"imageIdentifier": IMAGE}, client_token=token)

    assert model.run_calls == []


@pytest.mark.anyio
async def test_confirmed_termination_waits_for_terminal_readback() -> None:
    model = ClientTokenLambdaModel()
    model.terminal_after_polls = 2
    runner = await _create(model)
    await runner.close()

    outcome = await terminate_microvm_confirmed(
        model, runner.microvm_id, timeout_s=5, poll_interval_s=0
    )

    assert outcome == "terminated"
    assert model.microvms[runner.microvm_id]["state"] == "TERMINATED"
    assert await terminate_microvm_confirmed(model, runner.microvm_id, timeout_s=5) == (
        "terminated"
    )
    assert await terminate_microvm_confirmed(model, "microvm-unknown", timeout_s=5) == "absent"


@pytest.mark.anyio
async def test_confirmed_termination_timeout_leaves_cleanup_retryable() -> None:
    model = ClientTokenLambdaModel()
    model.terminal_after_polls = 1_000_000
    runner = await _create(model)
    await runner.close()

    with pytest.raises(LambdaMicroVMError, match="did not reach TERMINATED"):
        await terminate_microvm_confirmed(
            model, runner.microvm_id, timeout_s=0.05, poll_interval_s=0
        )

    model.terminal_after_polls = 0
    model.microvms[runner.microvm_id]["polls_until_terminated"] = 0
    assert await terminate_microvm_confirmed(model, runner.microvm_id, timeout_s=5) == (
        "terminated"
    )


_CUTOFF = datetime(2027, 1, 15, 12, 0, 0, tzinfo=UTC)


class _BotocoreSends:
    """A real botocore client whose HTTP layer records instead of sending."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, statuses: list[int]) -> None:
        boto3 = pytest.importorskip("boto3")
        from botocore.awsrequest import AWSResponse
        from botocore.config import Config

        self.sent: list[Any] = []
        self.client = boto3.Session(
            aws_access_key_id="AKIDEXAMPLE",
            aws_secret_access_key="example-secret",
            region_name="us-east-1",
        ).client(
            "lambda-microvms",
            config=Config(retries={"mode": "standard", "max_attempts": len(statuses)}),
        )
        body = json.dumps(
            {"microvmId": "microvm-0001", "endpoint": "microvm-0001.invalid", "state": "PENDING"}
        ).encode()

        class _Raw:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def stream(self, *_args: Any, **_kwargs: Any):
                yield self._data

        def send(request: Any) -> Any:
            self.sent.append(request)
            status = statuses[len(self.sent) - 1]
            return AWSResponse(
                request.url, status, {"Content-Type": "application/json"}, _Raw(body)
            )

        monkeypatch.setattr(self.client._endpoint.http_session, "send", send)
        monkeypatch.setattr(
            "botocore.retries.standard.ExponentialBackoff.delay_amount", lambda *_: 0
        )

    def sign_at(self, monkeypatch: pytest.MonkeyPatch, *times: datetime) -> None:
        # The signer reads the clock once per attempt; later attempts reuse the last time.
        def now(remove_tzinfo: bool = True) -> datetime:
            signed_at = times[min(len(self.sent), len(times) - 1)]
            return signed_at.replace(tzinfo=None) if remove_tzinfo else signed_at

        # botocore signs with awscrt when it is installed.
        import botocore.auth
        import botocore.crt.auth

        monkeypatch.setattr(botocore.auth, "get_current_datetime", now)
        monkeypatch.setattr(botocore.crt.auth, "get_current_datetime", now)


def _run_options() -> dict[str, Any]:
    return lambda_microvm_run_options(IMAGE, image_version="3", maximum_duration_in_seconds=600)


@pytest.mark.anyio
async def test_request_signed_before_the_deadline_is_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sends = _BotocoreSends(monkeypatch, statuses=[200])
    sends.sign_at(monkeypatch, _CUTOFF - timedelta(seconds=5))

    response = await run_microvm_with_client_token(
        sends.client,
        _run_options(),
        client_token=TOKEN,
        submit_not_after=_CUTOFF.timestamp(),
    )

    assert response["microvmId"] == "microvm-0001"
    assert len(sends.sent) == 1


@pytest.mark.anyio
async def test_request_signed_after_the_deadline_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The worker passed its local check, then stalled until after the deadline
    # before botocore signed the request.
    monkeypatch.setattr(
        lambda_microvm_module, "_submission_clock", lambda: _CUTOFF.timestamp() - 60
    )
    sends = _BotocoreSends(monkeypatch, statuses=[200])
    sends.sign_at(monkeypatch, _CUTOFF + timedelta(seconds=1))

    with pytest.raises(LambdaMicroVMSubmissionClosed):
        await run_microvm_with_client_token(
            sends.client,
            _run_options(),
            client_token=TOKEN,
            submit_not_after=_CUTOFF.timestamp(),
        )

    assert sends.sent == []


@pytest.mark.anyio
async def test_botocore_retry_signed_after_the_deadline_is_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sends = _BotocoreSends(monkeypatch, statuses=[500, 200, 200])
    sends.sign_at(monkeypatch, _CUTOFF - timedelta(seconds=5), _CUTOFF + timedelta(seconds=5))

    with pytest.raises(LambdaMicroVMSubmissionClosed):
        await run_microvm_with_client_token(
            sends.client,
            _run_options(),
            client_token=TOKEN,
            submit_not_after=_CUTOFF.timestamp(),
        )

    assert len(sends.sent) == 1


@pytest.mark.anyio
async def test_cayu_retry_after_the_deadline_is_never_submitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ClientTokenLambdaModel()
    now = [_CUTOFF.timestamp() - 1]
    monkeypatch.setattr(lambda_microvm_module, "_submission_clock", lambda: now[0])

    class _SlowFailure(ClientTokenLambdaModel):
        def run_microvm(self, **kwargs: Any) -> dict[str, Any]:
            now[0] = _CUTOFF.timestamp() + 1
            model.run_calls.append(kwargs)
            raise FakeLambdaClientError("ThrottlingException", "slow down")

    with pytest.raises(LambdaMicroVMSubmissionClosed):
        await run_microvm_with_client_token(
            _SlowFailure(),
            _run_options(),
            client_token=TOKEN,
            submit_not_after=_CUTOFF.timestamp(),
        )

    assert len(model.run_calls) == 1
