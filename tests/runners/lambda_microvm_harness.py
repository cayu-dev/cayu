from __future__ import annotations

import asyncio
import base64
import copy
import email.utils
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.aws.lambda_microvm_sidecar.supervisor import (
    CommandSupervisor,
    OwnerFence,
    OwnerLifecycleLeasedError,
    OwnerSupersededError,
)

from cayu.runners import LambdaMicroVMOwnershipSuperseded
from cayu.runners.aws_lambda_microvm import (
    LAMBDA_MICROVM_PROTOCOL_VERSION,
    LambdaMicroVMLifecycleInProgress,
)

_FENCED_TRANSPORTS: weakref.WeakSet[OwnerFencedTransport] = weakref.WeakSet()


def guest_lifecycle_hook(hook: str) -> None:
    """Deliver the guest's suspend, resume, or terminate hook to every fake sidecar.

    Fake control planes call this when they change a MicroVM's state, as AWS
    delivers those hooks to the real sidecar, which settles a matching lease.
    """

    for transport in list(_FENCED_TRANSPORTS):
        transport.owner_fence.settle_lifecycle(hook)


class OwnerFencedTransport:
    """The sidecar's real owner-fence semantics for fake endpoint transports."""

    _fence: OwnerFence | None = None

    @property
    def owner_fence(self) -> OwnerFence:
        if self._fence is None:
            self._fence = OwnerFence()
            _FENCED_TRANSPORTS.add(self)
        return self._fence

    def on_owner_superseded(self, generation: int) -> None:
        """Cancel earlier owners' commands, as the sidecar does on a new claim."""

    async def claim_owner(self, *, claim_id: str, **_kwargs: Any) -> dict[str, Any]:
        try:
            generation, superseded = self.owner_fence.claim(claim_id)
        except OwnerLifecycleLeasedError as exc:
            raise LambdaMicroVMLifecycleInProgress(str(exc)) from exc
        if superseded:
            self.on_owner_superseded(generation)
        return {"generation": generation, "superseded_previous": superseded}

    async def check_owner(self, *, claim_id: str, **_kwargs: Any) -> dict[str, bool]:
        return {"current": self.owner_fence.is_current(claim_id)}

    async def acquire_lifecycle(
        self, *, claim_id: str, action: str, **_kwargs: Any
    ) -> dict[str, Any]:
        try:
            return self.owner_fence.acquire_lifecycle(claim_id, action)
        except OwnerSupersededError as exc:
            raise LambdaMicroVMOwnershipSuperseded(str(exc)) from exc

    async def release_lifecycle(self, *, claim_id: str, **_kwargs: Any) -> dict[str, str]:
        try:
            self.owner_fence.release_lifecycle(claim_id)
        except OwnerSupersededError as exc:
            raise LambdaMicroVMOwnershipSuperseded(str(exc)) from exc
        return {"status": "released"}

    def admit_start(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.admit_start_with_generation(payload)[0]

    def admit_start_with_generation(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        admitted = dict(payload)
        try:
            generation = self.owner_fence.require(admitted.pop("owner_claim", None))
        except OwnerSupersededError as exc:
            raise LambdaMicroVMOwnershipSuperseded("fake sidecar reports a newer owner") from exc
        except OwnerLifecycleLeasedError as exc:
            raise LambdaMicroVMLifecycleInProgress(str(exc)) from exc
        return admitted, generation


class ConformanceLambdaClient:
    def __init__(self) -> None:
        self.suspend_calls = 0
        self.resume_calls = 0
        self.terminate_calls = 0
        self.state = "RUNNING"

    def run_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        return {
            "microvmId": "mvm-conformance",
            "endpoint": "conformance.lambda-microvm.invalid",
            "state": "PENDING",
            "imageArn": "arn:aws:lambda:us-east-1:123:microvm-image:conformance",
            "imageVersion": "1",
        }

    def create_microvm_auth_token(self, **_kwargs: Any) -> dict[str, Any]:
        return {"authToken": {"X-aws-proxy-auth": "conformance-token"}}

    def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "microvmId": kwargs.get("microvmIdentifier", "mvm-conformance"),
            "endpoint": "conformance.lambda-microvm.invalid",
            "state": self.state,
            "imageArn": "arn:aws:lambda:us-east-1:123:microvm-image:conformance",
            "imageVersion": "1",
        }

    def suspend_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.suspend_calls += 1
        self.state = "SUSPENDED"
        guest_lifecycle_hook("suspend")
        return {}

    def resume_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.resume_calls += 1
        self.state = "RUNNING"
        guest_lifecycle_hook("resume")
        return {}

    def terminate_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.terminate_calls += 1
        self.state = "TERMINATED"
        guest_lifecycle_hook("terminate")
        return {}


DEFAULT_GUEST_BOOT_ID = "0b1c2d3e-4f50-4617-8293-a4b5c6d7e8f9"


def is_boot_id_read(payload: dict[str, Any]) -> bool:
    """Whether a command reads the guest kernel boot id for admission identity."""

    return any("/proc/sys/kernel/random/boot_id" in part for part in payload.get("argv") or ())


class SupervisorTransport(OwnerFencedTransport):
    """Lambda sidecar transport shared by runner conformance and composition tests.

    Guest boot-id reads are answered from ``boot_id`` so admission identity is
    deterministic on hosts without Linux ``/proc`` (every other command runs in
    the real supervisor).
    """

    def __init__(
        self,
        root: Path,
        *,
        scripted_exit_code: Callable[[dict[str, Any]], int | None] | None = None,
        boot_id: str | Callable[[], str] = DEFAULT_GUEST_BOOT_ID,
    ) -> None:
        self.supervisor = CommandSupervisor(root=root)
        self.scripted_exit_code = scripted_exit_code
        self.boot_id = boot_id
        self.execution_profiles: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self._scripted_results: dict[str, dict[str, Any]] = {}

    async def health(self, **_kwargs: Any) -> dict[str, str]:
        return {"status": "ok", "protocol_version": LAMBDA_MICROVM_PROTOCOL_VERSION}

    async def start_command(
        self,
        *,
        command_id: str,
        payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        payload, generation = self.admit_start_with_generation(payload)
        copied = copy.deepcopy(payload)
        self.execution_profiles.append(copied["execution_profile"])
        self.payloads.append(copied)
        if is_boot_id_read(copied):
            boot_id = self.boot_id() if callable(self.boot_id) else self.boot_id
            self._scripted_results[command_id] = _terminal_result(
                command_id, exit_code=0, stdout=boot_id
            )
            return {"command_id": command_id, "state": "accepted"}
        if self.scripted_exit_code is not None:
            exit_code = self.scripted_exit_code(copied)
            if exit_code is not None:
                self._scripted_results[command_id] = _terminal_result(
                    command_id,
                    exit_code=exit_code,
                )
                return {"command_id": command_id, "state": "accepted"}
        return await asyncio.to_thread(
            self.supervisor.start, command_id, payload, owner_generation=generation
        )

    async def get_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        scripted = self._scripted_results.get(command_id)
        if scripted is not None:
            return dict(scripted)
        return await asyncio.to_thread(self.supervisor.get, command_id)

    async def cancel_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        if command_id in self._scripted_results:
            return {"command_id": command_id, "state": "cancelled"}
        return await asyncio.to_thread(self.supervisor.cancel, command_id)

    def on_owner_superseded(self, generation: int) -> None:
        self.supervisor.cancel_all(reason="owner_superseded", before_generation=generation)


def _terminal_result(command_id: str, *, exit_code: int, stdout: str = "") -> dict[str, Any]:
    encoded = stdout.encode("utf-8")
    return {
        "command_id": command_id,
        "state": "completed",
        "stdout_base64": base64.b64encode(encoded).decode("ascii"),
        "stderr_base64": "",
        "exit_code": exit_code,
        "timed_out": False,
        "cancelled": False,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_bytes": len(encoded),
        "stderr_bytes": 0,
    }


class FakeLambdaClientError(Exception):
    """Botocore-shaped client error; tests must not require the AWS SDK."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}")
        self.response = {"Error": {"Code": code, "Message": message}}


class ClientTokenLambdaModel:
    """Control-plane model of the RunMicrovm client-token semantics observed live.

    Verified against AWS in us-east-1 (see the recoverable-allocation live probe):
    an identical replay returns the original acknowledgement unchanged, including
    its original ``PENDING`` state, even after termination; changed parameters are
    rejected with ``ValidationException``; concurrent submissions converge on one
    MicroVM; termination is idempotent; unknown identifiers are not found.

    With ``clock`` set, the model also enforces what AWS guarantees in time: a
    request that arrives more than 15 minutes after it was signed is rejected,
    a MicroVM is terminated once its ``maximumDurationInSeconds`` elapses, and
    read responses carry AWS's clock in a ``Date`` header.
    """

    sigv4_validity_s = 900

    image_arn = "arn:aws:lambda:us-east-1:123:microvm-image:cayu"

    def __init__(self) -> None:
        self.microvms: dict[str, dict[str, Any]] = {}
        self.tokens: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.run_calls: list[dict[str, Any]] = []
        self.terminate_calls: list[str] = []
        self.suspend_calls: list[str] = []
        self.resume_calls: list[str] = []
        self.image_calls: list[str] = []
        self.run_failures: list[BaseException] = []
        self.lose_acknowledgements = 0
        self.terminal_after_polls: int = 0
        self.latest_image_version = "3"
        self.clock: Callable[[], float] | None = None

    def created_ids(self) -> list[str]:
        return list(self.microvms)

    def run_microvm(self, **kwargs: Any) -> dict[str, Any]:
        return self.arrive(self.clock() if self.clock is not None else None, **kwargs)

    def arrive(self, signed_at: float | None, **kwargs: Any) -> dict[str, Any]:
        """Apply one RunMicrovm request signed at ``signed_at`` as it reaches AWS."""

        self.run_calls.append(copy.deepcopy(kwargs))
        if self.run_failures:
            raise self.run_failures.pop(0)
        if (
            self.clock is not None
            and signed_at is not None
            and self.clock() - signed_at > self.sigv4_validity_s
        ):
            raise FakeLambdaClientError("InvalidSignatureException", "Signature expired")
        params = {key: value for key, value in kwargs.items() if key != "clientToken"}
        token = kwargs.get("clientToken")
        if token is not None and token in self.tokens:
            original_params, acknowledgement = self.tokens[token]
            if original_params != params:
                raise FakeLambdaClientError(
                    "ValidationException",
                    "The provided clientToken was used with different request parameters.",
                )
            response = copy.deepcopy(acknowledgement)
        else:
            microvm_id = f"microvm-{len(self.microvms) + 1:04d}"
            self.microvms[microvm_id] = {
                "microvmId": microvm_id,
                "endpoint": f"{microvm_id}.lambda-microvm.invalid",
                "state": "RUNNING",
                "imageArn": self.image_arn,
                "imageVersion": kwargs.get("imageVersion", self.latest_image_version),
                "polls_until_terminated": None,
                "expires_at": (
                    None
                    if self.clock is None or kwargs.get("maximumDurationInSeconds") is None
                    else self.clock() + kwargs["maximumDurationInSeconds"]
                ),
            }
            response = self._public(self.microvms[microvm_id])
            response["state"] = "PENDING"
            if token is not None:
                self.tokens[token] = (params, copy.deepcopy(response))
        if self.lose_acknowledgements:
            self.lose_acknowledgements -= 1
            raise FakeLambdaClientError("InternalServerException", "acknowledgement lost")
        return response

    def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
        microvm = self.microvms.get(kwargs["microvmIdentifier"])
        if microvm is None:
            raise FakeLambdaClientError("ResourceNotFoundException", "not found")
        remaining = microvm["polls_until_terminated"]
        if remaining is not None:
            if remaining <= 0:
                microvm["state"] = "TERMINATED"
            else:
                microvm["polls_until_terminated"] = remaining - 1
        return self._public(microvm)

    def live_ids(self) -> list[str]:
        """Identifiers AWS would still report as not terminated."""

        return [
            identifier
            for identifier, microvm in self.microvms.items()
            if self._public(microvm)["state"] != "TERMINATED"
        ]

    def _public(self, microvm: dict[str, Any]) -> dict[str, Any]:
        expires_at = microvm.get("expires_at")
        if expires_at is not None and self.clock is not None and self.clock() >= expires_at:
            microvm["state"] = "TERMINATED"
        return {
            key: value
            for key, value in microvm.items()
            if key not in {"polls_until_terminated", "expires_at"}
        }

    def get_microvm_image(self, **kwargs: Any) -> dict[str, Any]:
        self.image_calls.append(kwargs["imageIdentifier"])
        response: dict[str, Any] = {
            "imageArn": self.image_arn,
            "state": "CREATED",
            "latestActiveImageVersion": self.latest_image_version,
        }
        if self.clock is not None:
            response["ResponseMetadata"] = {
                "HTTPHeaders": {"date": email.utils.formatdate(self.clock(), usegmt=True)}
            }
        return response

    def create_microvm_auth_token(self, **_kwargs: Any) -> dict[str, Any]:
        return {"authToken": {"X-aws-proxy-auth": "model-token"}}

    def suspend_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.suspend_calls.append(kwargs["microvmIdentifier"])
        self.microvms[kwargs["microvmIdentifier"]]["state"] = "SUSPENDED"
        guest_lifecycle_hook("suspend")
        return {}

    def resume_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.resume_calls.append(kwargs["microvmIdentifier"])
        self.microvms[kwargs["microvmIdentifier"]]["state"] = "RUNNING"
        guest_lifecycle_hook("resume")
        return {}

    def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
        identifier = kwargs["microvmIdentifier"]
        self.terminate_calls.append(identifier)
        microvm = self.microvms.get(identifier)
        if microvm is None:
            raise FakeLambdaClientError("ResourceNotFoundException", "not found")
        if microvm["state"] != "TERMINATED":
            microvm["state"] = "TERMINATING"
            microvm["polls_until_terminated"] = self.terminal_after_polls
        guest_lifecycle_hook("terminate")
        return {}
