"""A newer owner carries the earlier owner's commands over instead of cancelling them."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from examples.aws.lambda_microvm_sidecar.supervisor import CommandSupervisor, OwnerFence
from fastapi import HTTPException
from fastapi.testclient import TestClient
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel, SupervisorTransport
from tests.runners.test_aws_lambda_microvm_owner_fence import _payload, _wait_for_state
from tests.runners.test_aws_lambda_microvm_owner_fence import sidecar as sidecar

from cayu.runners import (
    ExecCommand,
    LambdaMicroVMError,
    LambdaMicroVMOwnershipSuperseded,
    LambdaMicroVMRunner,
    LambdaMicroVMStaleOwnerEpoch,
)
from cayu.runners.aws_lambda_microvm import LambdaMicroVMLifecycleInProgress

A, B, C = ("a" * 64, "b" * 64, "c" * 64)
_SLEEP = "import time; time.sleep(30)"


@pytest.fixture
def plain_sidecar(sidecar, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # The host test platform lacks the agent namespace and setpriv tools, so agent
    # commands run as the test user, as in the shared supervisor transport.
    supervisor = CommandSupervisor(root=tmp_path)
    monkeypatch.setattr(sidecar, "SUPERVISOR", supervisor)
    try:
        yield sidecar
    finally:
        supervisor.cancel_all()


def _start(sidecar, tmp_path: Path, claim: str, command_id: str, code: str = _SLEEP) -> None:
    payload = _payload(tmp_path, claim, sys.executable, "-c", code)
    payload["command_id"] = command_id
    asyncio.run(sidecar.start_command(payload))
    _wait_for_state(sidecar, command_id, "running")


def _state(sidecar, command_id: str) -> str:
    return sidecar.SUPERVISOR.get(command_id)["state"]


def _status(call: Any) -> int:
    with pytest.raises(HTTPException) as raised:
        asyncio.run(call)
    return raised.value.status_code


def test_a_newer_epoch_carries_the_listed_commands_and_cancels_the_rest(
    plain_sidecar, tmp_path: Path
) -> None:
    sidecar = plain_sidecar
    asyncio.run(sidecar.claim_owner({"claim_id": A, "owner_epoch": 1}))
    _start(sidecar, tmp_path, A, "kept")
    _start(sidecar, tmp_path, A, "other")

    claimed = asyncio.run(
        sidecar.claim_owner(
            {"claim_id": B, "owner_epoch": 2, "carry_commands": ["kept", "unknown"]}
        )
    )

    assert claimed == {"generation": 2, "superseded_previous": True, "carried_commands": ["kept"]}
    assert _state(sidecar, "kept") == "running"
    other = sidecar.SUPERVISOR.get("other")
    assert (other["state"], other["cancel_reason"]) == ("cancelled", "owner_superseded")
    # Asked again, the same claim reports the same commands and stops nothing.
    again = asyncio.run(
        sidecar.claim_owner({"claim_id": B, "owner_epoch": 2, "carry_commands": ["kept"]})
    )
    assert again == {"generation": 2, "superseded_previous": False, "carried_commands": ["kept"]}
    assert _state(sidecar, "kept") == "running"


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"owner_epoch": 2}, {"owner_epoch": 2}),
        ({}, {"owner_epoch": 1}),
    ],
    ids=["equal_epoch", "unnumbered_owner"],
)
def test_only_a_strictly_newer_epoch_than_a_numbered_owner_carries(
    plain_sidecar, tmp_path: Path, first: dict[str, Any], second: dict[str, Any]
) -> None:
    sidecar = plain_sidecar
    asyncio.run(sidecar.claim_owner({"claim_id": A, **first}))
    _start(sidecar, tmp_path, A, "job")

    claimed = asyncio.run(sidecar.claim_owner({"claim_id": B, **second, "carry_commands": ["job"]}))

    assert claimed["carried_commands"] == []
    assert _state(sidecar, "job") == "cancelled"


def test_an_older_epoch_cannot_take_commands_back(plain_sidecar, tmp_path: Path) -> None:
    sidecar = plain_sidecar
    asyncio.run(sidecar.claim_owner({"claim_id": A, "owner_epoch": 1}))
    _start(sidecar, tmp_path, A, "job")
    asyncio.run(sidecar.claim_owner({"claim_id": B, "owner_epoch": 3, "carry_commands": ["job"]}))

    stale = sidecar.claim_owner({"claim_id": C, "owner_epoch": 2, "carry_commands": ["job"]})

    assert _status(stale) == sidecar.STALE_OWNER_EPOCH_STATUS
    assert _state(sidecar, "job") == "running"
    assert sidecar.OWNER.is_current(B)


def test_carrying_needs_an_epoch_and_a_list_of_ids(plain_sidecar) -> None:
    sidecar = plain_sidecar
    for payload in (
        {"claim_id": A, "carry_commands": ["job"]},
        {"claim_id": A, "owner_epoch": 1, "carry_commands": "job"},
        {"claim_id": A, "owner_epoch": 1, "carry_commands": [7]},
        {"claim_id": A, "owner_epoch": 1, "carry_commands": ["job"] * 257},
    ):
        assert _status(sidecar.claim_owner(payload)) == 400
    assert not sidecar.OWNER.is_current(A)


def test_only_the_new_owner_can_cancel_or_release_a_carried_command(
    plain_sidecar, tmp_path: Path
) -> None:
    sidecar = plain_sidecar
    asyncio.run(sidecar.claim_owner({"claim_id": A, "owner_epoch": 1}))
    _start(sidecar, tmp_path, A, "job")
    _start(sidecar, tmp_path, A, "done", "import time; time.sleep(0.5); print('done')")
    asyncio.run(
        sidecar.claim_owner({"claim_id": B, "owner_epoch": 2, "carry_commands": ["job", "done"]})
    )
    superseded = sidecar.OWNER_SUPERSEDED_STATUS

    # The earlier owner, or a caller naming no owner, can neither stop it nor drop its result.
    assert _status(sidecar.cancel_command("job")) == superseded
    assert _status(sidecar.cancel_command("job", {"owner_claim": A})) == superseded
    assert _state(sidecar, "job") == "running"
    _wait_for_state(sidecar, "done", "completed")
    assert _status(sidecar.release_command("done", {"owner_claim": A})) == superseded
    assert sidecar.SUPERVISOR.get("done")["stdout_base64"]

    released = asyncio.run(sidecar.release_command("done", {"owner_claim": B}))
    assert released["state"] == "released"
    cancelled = asyncio.run(sidecar.cancel_command("job", {"owner_claim": B}))
    assert cancelled["state"] == "cancelled"


def test_a_later_claim_that_does_not_carry_cancels_carried_commands(
    plain_sidecar, tmp_path: Path
) -> None:
    sidecar = plain_sidecar
    asyncio.run(sidecar.claim_owner({"claim_id": A, "owner_epoch": 1}))
    _start(sidecar, tmp_path, A, "job")
    asyncio.run(sidecar.claim_owner({"claim_id": B, "owner_epoch": 2, "carry_commands": ["job"]}))

    asyncio.run(sidecar.claim_owner({"claim_id": C, "owner_epoch": 3}))

    assert _state(sidecar, "job") == "cancelled"


def test_a_command_the_earlier_owner_is_cancelling_is_not_carried(tmp_path: Path) -> None:
    fence = OwnerFence()
    supervisor = CommandSupervisor(root=tmp_path)
    fence.claim(A, owner_epoch=1)
    with fence.admitted(A) as generation:
        supervisor.start(
            "job",
            {
                "kind": "process",
                "argv": [sys.executable, "-c", _SLEEP],
                "cwd": str(tmp_path),
                "timeout_s": 30,
            },
            owner_generation=generation,
        )
    supervisor.cancel("job")
    generation, superseded = fence.claim(B, owner_epoch=2, carry=True)
    with fence.carrying(generation) as granted:
        assert (granted, superseded) == (True, True)
        assert supervisor.carry(["job"], generation=generation) == []


def test_an_unnumbered_owner_never_hands_its_commands_over() -> None:
    fence = OwnerFence()
    fence.claim(A, owner_epoch=1)
    fence.claim(B)
    generation, superseded = fence.claim(C, owner_epoch=2, carry=True)
    with fence.carrying(generation) as granted:
        assert (granted, superseded) == (False, True)


def _started(supervisor: CommandSupervisor, root: Path, generation: int, **extra: Any) -> None:
    supervisor.start(
        "job",
        {
            "kind": "process",
            "argv": [sys.executable, "-c", _SLEEP],
            "cwd": str(root),
            "env": {},
            "timeout_s": 30,
            **extra,
        },
        owner_generation=generation,
    )


def test_a_takeover_still_cancelling_leaves_what_a_newer_owner_carried(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    _started(supervisor, tmp_path, 1)
    try:
        # Owner 3 carries the job while owner 2's takeover, which listed it, is still cancelling.
        assert supervisor.carry(["job"], generation=3) == ["job"]
        supervisor._cancel(
            "job",
            reason="owner_superseded",
            owner_generation=None,
            authorized=True,
            before_generation=2,
        )
        assert supervisor.get("job")["state"] in {"accepted", "running"}
    finally:
        supervisor.cancel_all()


def test_an_interactive_command_is_never_carried(tmp_path: Path) -> None:
    supervisor = CommandSupervisor(root=tmp_path)
    _started(supervisor, tmp_path, 1, interactive=True)
    try:
        assert supervisor.carry(["job"], generation=2) == []
    finally:
        supervisor.cancel_all()


def test_a_carry_grant_lapses_once_a_newer_claim_lands() -> None:
    fence = OwnerFence()
    fence.claim(A, owner_epoch=1)
    carrying, _ = fence.claim(B, owner_epoch=2, carry=True)
    fence.claim(C, owner_epoch=3)
    with fence.carrying(carrying) as granted:
        assert granted is False


def test_cancel_and_release_take_an_owner_claim_in_their_body(plain_sidecar, tmp_path) -> None:
    sidecar = plain_sidecar
    client = TestClient(sidecar.app)
    asyncio.run(sidecar.claim_owner({"claim_id": A, "owner_epoch": 1}))
    _start(sidecar, tmp_path, A, "job")
    _start(sidecar, tmp_path, A, "open")
    asyncio.run(sidecar.claim_owner({"claim_id": B, "owner_epoch": 2, "carry_commands": ["job"]}))
    _start(sidecar, tmp_path, B, "plain")

    assert client.request("DELETE", "/v1/commands/job").status_code == 412
    assert client.request("DELETE", "/v1/commands/job", json={"owner_claim": A}).status_code == 412
    stopped = client.request("DELETE", "/v1/commands/job", json={"owner_claim": B})
    assert (stopped.status_code, stopped.json()["state"]) == (200, "cancelled")
    # A command that was never carried keeps its open cancel.
    assert client.request("DELETE", "/v1/commands/plain").json()["state"] == "cancelled"
    assert client.post("/v1/commands/plain/release").json()["state"] == "released"
    health = client.get("/health").json()
    assert health["capabilities"] == ["owner_epoch", "carry_commands"]


class AppTransport:
    """The real sidecar's routes as an endpoint transport, mapping refusals as HTTPS does."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.claims: list[dict[str, Any]] = []

    async def _call(self, call: Any, *, owner: bool = False) -> Any:
        try:
            return await call
        except HTTPException as error:
            if owner and error.status_code == 409:
                raise LambdaMicroVMStaleOwnerEpoch(str(error.detail)) from None
            if error.status_code == 412:
                raise LambdaMicroVMOwnershipSuperseded(str(error.detail)) from None
            if error.status_code == 423:
                raise LambdaMicroVMLifecycleInProgress(str(error.detail)) from None
            raise LambdaMicroVMError(f"HTTP {error.status_code}") from None

    async def health(self, **_kwargs: Any) -> dict[str, Any]:
        return await self.app.health()

    async def claim_owner(
        self,
        *,
        claim_id: str,
        owner_epoch: int | None = None,
        carry_commands: list[str] | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"claim_id": claim_id}
        if owner_epoch is not None:
            payload["owner_epoch"] = owner_epoch
        if carry_commands is not None:
            payload["carry_commands"] = list(carry_commands)
        self.claims.append(payload)
        return await self._call(self.app.claim_owner(payload), owner=True)

    async def check_owner(self, *, claim_id: str, **_kwargs: Any) -> dict[str, Any]:
        return await self.app.check_owner({"claim_id": claim_id})

    async def start_command(
        self, *, command_id: str, payload: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        return await self._call(self.app.start_command({"command_id": command_id, **payload}))

    async def get_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        return await self._call(self.app.get_command(command_id))

    async def cancel_command(
        self, *, command_id: str, owner_claim: str | None = None, **_kwargs: Any
    ) -> dict[str, Any]:
        body = None if owner_claim is None else {"owner_claim": owner_claim}
        return await self._call(self.app.cancel_command(command_id, body))

    async def release_command(
        self, *, command_id: str, owner_claim: str | None = None, **_kwargs: Any
    ) -> dict[str, Any]:
        body = None if owner_claim is None else {"owner_claim": owner_claim}
        return await self._call(self.app.release_command(command_id, body))


def _options(model: ClientTokenLambdaModel, transport: Any, tmp_path: Path) -> dict[str, Any]:
    return dict(
        client=model,
        endpoint_transport=transport,
        default_cwd=str(tmp_path),
        poll_interval_s=0,
        close_action="none",
    )


async def _until_running(sidecar, command_id: str) -> None:
    await asyncio.to_thread(_wait_for_state, sidecar, command_id, "running")


@pytest.mark.anyio
async def test_a_successor_runner_carries_a_named_command_and_returns_its_result(
    plain_sidecar, tmp_path: Path
) -> None:
    sidecar = plain_sidecar
    model = ClientTokenLambdaModel()
    transport = AppTransport(sidecar)
    options = _options(model, transport, tmp_path)
    first = await LambdaMicroVMRunner.create("image", owner_epoch=1, **options)
    assert first.command_carry_supported is True
    code = "import time; time.sleep(1); print('carried'); raise SystemExit(3)"
    waiting = asyncio.create_task(
        first.exec(ExecCommand.process(sys.executable, "-c", code), command_id="job-1")
    )
    await _until_running(sidecar, "job-1")

    # The stopping owner lets go without stopping it.
    first.detach_commands()
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert _state(sidecar, "job-1") == "running"
    with pytest.raises(RuntimeError, match="detached"):
        await first.exec(ExecCommand.process("true"))
    # Detaching is final: the runner can't be reopened for commands of its own.
    with pytest.raises(RuntimeError, match="detached"):
        first.reopen_exec()
    first._open_exec()
    assert first.lifecycle_state == "fenced"

    successor = await LambdaMicroVMRunner.from_existing(
        first.microvm_id, owner_epoch=2, carry_commands=["job-1", "job-2"], **options
    )
    assert successor.carried_commands == ("job-1",)
    assert transport.claims[-1]["carry_commands"] == ["job-1", "job-2"]
    result = await successor.resume_exec("job-1")

    assert (result.exit_code, result.stdout) == (3, "carried\n")
    assert _state(sidecar, "job-1") == "released"
    await first.close()
    await successor.close()


@pytest.mark.anyio
async def test_cancelling_a_resumed_wait_stops_the_carried_command(
    plain_sidecar, tmp_path: Path
) -> None:
    sidecar = plain_sidecar
    options = _options(ClientTokenLambdaModel(), AppTransport(sidecar), tmp_path)
    first = await LambdaMicroVMRunner.create("image", owner_epoch=1, **options)
    waiting = asyncio.create_task(
        first.exec(ExecCommand.process(sys.executable, "-c", _SLEEP), command_id="long")
    )
    await _until_running(sidecar, "long")
    first.detach_commands()
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)
    successor = await LambdaMicroVMRunner.from_existing(
        first.microvm_id, owner_epoch=2, carry_commands=["long"], **options
    )

    resumed = asyncio.create_task(successor.resume_exec("long"))
    await asyncio.sleep(0.05)
    resumed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resumed

    assert _state(sidecar, "long") == "cancelled"
    await first.close()
    await successor.close()


@pytest.mark.anyio
async def test_a_sidecar_without_the_capability_is_never_asked_to_carry(tmp_path: Path) -> None:
    transport = SupervisorTransport(tmp_path)
    claims: list[dict[str, Any]] = []
    claim = transport.claim_owner

    async def recorded(**kwargs: Any) -> dict[str, Any]:
        claims.append(kwargs)
        return await claim(**kwargs)

    transport.claim_owner = recorded  # type: ignore[method-assign]
    options = _options(ClientTokenLambdaModel(), transport, tmp_path)
    first = await LambdaMicroVMRunner.create("image", owner_epoch=1, **options)
    successor = await LambdaMicroVMRunner.from_existing(
        first.microvm_id, owner_epoch=2, carry_commands=["job"], **options
    )

    assert (first.command_carry_supported, successor.carried_commands) == (False, ())
    assert "carry_commands" not in claims[-1]
    with pytest.raises(ValueError, match="carried over"):
        await successor.resume_exec("job")
    await first.close()
    await successor.close()


@pytest.mark.anyio
async def test_carry_arguments_are_checked_before_anything_is_called(tmp_path: Path) -> None:
    class NeverCallProvider:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError("provider must not be called")

    with pytest.raises(ValueError, match="owner_epoch"):
        await LambdaMicroVMRunner.from_existing(
            "vm", client=NeverCallProvider(), carry_commands=["job"]
        )
    for invalid in (["a/b"], [""], ["x" * 129], ["."], [".."]):
        with pytest.raises(ValueError, match="command_id"):
            await LambdaMicroVMRunner.from_existing(
                "vm", client=NeverCallProvider(), owner_epoch=1, carry_commands=invalid
            )
    with pytest.raises(TypeError):
        await LambdaMicroVMRunner.from_existing(
            "vm", client=NeverCallProvider(), owner_epoch=1, carry_commands="job"
        )
    runner = LambdaMicroVMRunner(
        ClientTokenLambdaModel(),
        microvm_id="vm",
        endpoint="https://example.test",
        endpoint_transport=SupervisorTransport(tmp_path),
    )
    with pytest.raises(ValueError, match="command_id"):
        await runner.exec(ExecCommand.process("true"), command_id="no spaces")
    await runner.close()
