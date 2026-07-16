"""Per-session AWS Lambda MicroVM environment with durable reconnect metadata."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from cayu import (
    BoundWorkspace,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    LambdaMicroVMRunner,
    NativeBinding,
    RunnerWorkspace,
    WorkspaceBinding,
    WorkspaceSnapshot,
)
from cayu.runners import LambdaMicroVMEndpointTransport, LambdaMicroVMProtocolError


class LambdaMicroVMLifecycleBinding(WorkspaceBinding):
    """Native binding that maps terminal session outcomes to MicroVM lifecycle actions."""

    def __init__(self, runner: LambdaMicroVMRunner) -> None:
        if not isinstance(runner, LambdaMicroVMRunner):
            raise TypeError("LambdaMicroVMLifecycleBinding requires a LambdaMicroVMRunner")
        self.runner = runner
        self._native = NativeBinding(default_path=runner.default_cwd)
        self._finalized = False
        self._finalize_lock = asyncio.Lock()

    async def bind(
        self,
        workspace,
        runner,
        *,
        session_id: str,
        agent_name: str | None = None,
        environment_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BoundWorkspace:
        return await self._native.bind(
            workspace,
            runner,
            session_id=session_id,
            agent_name=agent_name,
            environment_name=environment_name,
            metadata=metadata,
        )

    async def finalize(
        self,
        bound: BoundWorkspace,
        *,
        outcome: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WorkspaceSnapshot | None:
        await self._native.finalize(bound, outcome=outcome, metadata=metadata)
        async with self._finalize_lock:
            if self._finalized:
                return None
            if outcome == "interrupted":
                await self.runner.suspend()
            else:
                await self.runner.terminate()
            await self.runner.close()
            self._finalized = True
        return None


class LambdaMicroVMEnvironmentFactory(EnvironmentFactory):
    """Create one MicroVM per session and reattach on resume/recovery.

    Forks always allocate a fresh MicroVM even when inherited checkpoint state
    contains the parent's reconnect metadata.
    """

    def __init__(
        self,
        *,
        image_identifier: str,
        region_name: str,
        client: Any | None = None,
        endpoint_transport_factory: Callable[[], LambdaMicroVMEndpointTransport] | None = None,
        create_options: dict[str, Any] | None = None,
        runner_options: dict[str, Any] | None = None,
    ) -> None:
        self.image_identifier = image_identifier
        self.region_name = region_name
        self.client = client
        self.endpoint_transport_factory = endpoint_transport_factory
        self.create_options = dict(create_options or {})
        self.runner_options = dict(runner_options or {})
        reserved = {"client", "endpoint_transport", "close_action", "region_name"}
        conflicts = sorted(
            reserved.intersection(self.runner_options) | reserved.intersection(self.create_options)
        )
        if conflicts:
            raise ValueError(f"factory options cannot override: {', '.join(conflicts)}")
        overlap = sorted(self.create_options.keys() & self.runner_options.keys())
        if overlap:
            raise ValueError(f"create_options and runner_options overlap: {', '.join(overlap)}")

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        if not isinstance(request, EnvironmentFactoryRequest):
            raise TypeError("LambdaMicroVMEnvironmentFactory requires EnvironmentFactoryRequest")
        endpoint_transport = (
            self.endpoint_transport_factory()
            if self.endpoint_transport_factory is not None
            else None
        )
        reconnect: dict[str, str | None] | None = None
        if request.operation is EnvironmentFactoryOperation.RECONNECT:
            if not request.reconnect_metadata:
                raise ValueError("MicroVM reconnect requires durable reconnect metadata")
            reconnect = {
                "microvm_id": _required_reconnect_string(request.reconnect_metadata, "microvm_id"),
                "endpoint": _required_reconnect_string(request.reconnect_metadata, "endpoint"),
                "region": _required_reconnect_string(request.reconnect_metadata, "region"),
                "image_identifier": _required_reconnect_string(
                    request.reconnect_metadata, "image_identifier"
                ),
                "image_version": _optional_reconnect_string(
                    request.reconnect_metadata, "image_version"
                ),
            }
        reconnect_id = reconnect["microvm_id"] if reconnect is not None else None
        if reconnect_id is None:
            runner = await LambdaMicroVMRunner.create(
                self.image_identifier,
                region_name=self.region_name,
                client=self.client,
                endpoint_transport=endpoint_transport,
                close_action="none",
                **self.create_options,
                **self.runner_options,
            )
        else:
            assert reconnect is not None
            runner = await LambdaMicroVMRunner.from_existing(
                reconnect_id,
                region_name=reconnect["region"],
                client=self.client,
                endpoint_transport=endpoint_transport,
                close_action="none",
                **self.runner_options,
            )
            mismatch = _reconnect_identity_mismatch(runner, reconnect)
            if mismatch is not None:
                await runner.close()
                raise LambdaMicroVMProtocolError(
                    f"Lambda MicroVM reconnect {mismatch} changed from durable metadata."
                )
        region = runner.region_name or self.region_name
        workspace = RunnerWorkspace(
            runner,
            workspace_id=f"lambda-microvm:{runner.microvm_id}:{runner.default_cwd}",
        )
        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(
                    name=request.environment_name,
                    metadata={"kind": "lambda-microvm", "region": region},
                ),
                workspace=workspace,
                runner=runner,
                binding=LambdaMicroVMLifecycleBinding(runner),
            ),
            reconnect_metadata={
                "microvm_id": runner.microvm_id,
                "endpoint": runner.endpoint,
                "region": region,
                "image_identifier": runner.image_identifier or self.image_identifier,
                "image_version": runner.image_version,
            },
        )


def _required_reconnect_string(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Lambda MicroVM reconnect metadata requires nonblank {key}.")
    return value.strip()


def _optional_reconnect_string(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Lambda MicroVM reconnect metadata {key} must be nonblank or null.")
    return value.strip()


def _reconnect_identity_mismatch(
    runner: LambdaMicroVMRunner,
    reconnect: dict[str, str | None],
) -> str | None:
    if runner.endpoint != reconnect["endpoint"]:
        return "endpoint"
    if runner.image_identifier != reconnect["image_identifier"]:
        return "image_identifier"
    expected_version = reconnect["image_version"]
    if expected_version is not None and runner.image_version != expected_version:
        return "image_version"
    return None
