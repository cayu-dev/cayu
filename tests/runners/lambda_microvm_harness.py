from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable
from pathlib import Path
from typing import Any

from examples.aws.lambda_microvm_sidecar.supervisor import CommandSupervisor


class ConformanceLambdaClient:
    def __init__(self) -> None:
        self.suspend_calls = 0
        self.resume_calls = 0
        self.terminate_calls = 0

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
            "state": "RUNNING",
            "imageArn": "arn:aws:lambda:us-east-1:123:microvm-image:conformance",
            "imageVersion": "1",
        }

    def suspend_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.suspend_calls += 1
        return {}

    def resume_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.resume_calls += 1
        return {}

    def terminate_microvm(self, **_kwargs: Any) -> dict[str, Any]:
        self.terminate_calls += 1
        return {}


class SupervisorTransport:
    """Lambda sidecar transport shared by runner conformance and composition tests."""

    def __init__(
        self,
        root: Path,
        *,
        scripted_exit_code: Callable[[dict[str, Any]], int | None] | None = None,
    ) -> None:
        self.supervisor = CommandSupervisor(root=root)
        self.scripted_exit_code = scripted_exit_code
        self.execution_profiles: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self._scripted_results: dict[str, dict[str, Any]] = {}

    async def health(self, **_kwargs: Any) -> dict[str, str]:
        return {"status": "ok", "protocol_version": "1"}

    async def start_command(
        self,
        *,
        command_id: str,
        payload: dict[str, Any],
        **_kwargs: Any,
    ) -> dict[str, Any]:
        copied = copy.deepcopy(payload)
        self.execution_profiles.append(copied["execution_profile"])
        self.payloads.append(copied)
        if self.scripted_exit_code is not None:
            exit_code = self.scripted_exit_code(copied)
            if exit_code is not None:
                self._scripted_results[command_id] = _terminal_result(
                    command_id,
                    exit_code=exit_code,
                )
                return {"command_id": command_id, "state": "accepted"}
        return await asyncio.to_thread(self.supervisor.start, command_id, payload)

    async def get_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        scripted = self._scripted_results.get(command_id)
        if scripted is not None:
            return dict(scripted)
        return await asyncio.to_thread(self.supervisor.get, command_id)

    async def cancel_command(self, *, command_id: str, **_kwargs: Any) -> dict[str, Any]:
        if command_id in self._scripted_results:
            return {"command_id": command_id, "state": "cancelled"}
        return await asyncio.to_thread(self.supervisor.cancel, command_id)


def _terminal_result(command_id: str, *, exit_code: int) -> dict[str, Any]:
    return {
        "command_id": command_id,
        "state": "completed",
        "stdout_base64": "",
        "stderr_base64": "",
        "exit_code": exit_code,
        "timed_out": False,
        "cancelled": False,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
    }
