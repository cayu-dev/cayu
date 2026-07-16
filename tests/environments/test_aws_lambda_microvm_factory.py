from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from cayu import EnvironmentFactoryOperation, EnvironmentFactoryRequest, RunnerWorkspace
from cayu.runners import LambdaMicroVMProtocolError

_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "examples" / "aws" / "environments" / "lambda_microvm.py"
)
_SPEC = importlib.util.spec_from_file_location("lambda_microvm_environment_example", _EXAMPLE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
LambdaMicroVMEnvironmentFactory = _MODULE.LambdaMicroVMEnvironmentFactory


class FakeControlClient:
    def __init__(self) -> None:
        self.run_calls = 0
        self.get_calls = 0
        self.suspend_calls = 0
        self.terminate_calls = 0

    def run_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.run_calls += 1
        return {
            "microvmId": f"mvm-{self.run_calls}",
            "endpoint": f"mvm-{self.run_calls}.example.test",
            "imageArn": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
            "imageVersion": "3",
        }

    def get_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls += 1
        return {
            "microvmId": kwargs["microvmIdentifier"],
            "endpoint": f"{kwargs['microvmIdentifier']}.example.test",
            "state": "RUNNING",
            "imageArn": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
            "imageVersion": "3",
        }

    def create_microvm_auth_token(self, **kwargs: Any) -> dict[str, Any]:
        return {"authToken": {"X-aws-proxy-auth": "token"}}

    def suspend_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.suspend_calls += 1
        return {}

    def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.terminate_calls += 1
        return {}


class FailOnceTerminateControlClient(FakeControlClient):
    def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
        self.terminate_calls += 1
        if self.terminate_calls == 1:
            raise RuntimeError("temporary termination failure")
        return {}


class HealthyTransport:
    async def health(self, *, endpoint: str, token: str, timeout_s: float) -> dict[str, str]:
        return {"status": "ok", "protocol_version": "1"}


@pytest.mark.anyio
async def test_lambda_microvm_factory_reconnects_and_applies_terminal_lifecycle() -> None:
    client = FakeControlClient()
    factory = LambdaMicroVMEnvironmentFactory(
        image_identifier="arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        region_name="us-west-2",
        client=client,
        endpoint_transport_factory=HealthyTransport,
        runner_options={"poll_interval_s": 0},
    )
    request = EnvironmentFactoryRequest(
        session_id="session-1",
        agent_name="assistant",
        environment_name="aws-sandbox",
    )

    created = await factory.create(request)
    runner = created.environment.runner
    binding = created.environment.binding
    assert runner is not None and binding is not None
    assert isinstance(created.environment.workspace, RunnerWorkspace)
    assert created.environment.spec.name == "aws-sandbox"
    assert created.reconnect_metadata == {
        "microvm_id": "mvm-1",
        "endpoint": "mvm-1.example.test",
        "region": "us-west-2",
        "image_identifier": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        "image_version": "3",
    }
    bound = await binding.bind(
        created.environment.workspace,
        runner,
        session_id="session-1",
        environment_name="aws-sandbox",
    )
    await binding.finalize(bound, outcome="interrupted")
    assert client.suspend_calls == 1
    assert client.terminate_calls == 0

    resumed = await factory.create(
        EnvironmentFactoryRequest(
            session_id="session-1",
            agent_name="assistant",
            environment_name="aws-sandbox",
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=created.reconnect_metadata,
        )
    )
    resumed_runner = resumed.environment.runner
    resumed_binding = resumed.environment.binding
    assert resumed_runner is not None and resumed_binding is not None
    assert resumed_runner.region_name == "us-west-2"
    assert resumed.environment.spec.metadata["region"] == "us-west-2"
    resumed_bound = await resumed_binding.bind(
        resumed.environment.workspace,
        resumed_runner,
        session_id="session-1",
        environment_name="aws-sandbox",
    )
    await resumed_binding.finalize(resumed_bound, outcome="completed")

    assert client.get_calls == 1
    assert client.terminate_calls == 1

    await factory.create(
        EnvironmentFactoryRequest(
            session_id="session-child",
            parent_session_id="session-1",
            agent_name="assistant",
            environment_name="aws-sandbox",
            reconnect_metadata=created.reconnect_metadata,
        )
    )
    assert client.run_calls == 2


@pytest.mark.anyio
async def test_lambda_microvm_factory_reconnect_uses_durable_region_and_validates_identity() -> (
    None
):
    client = FakeControlClient()
    factory = LambdaMicroVMEnvironmentFactory(
        image_identifier="arn:aws:lambda:us-east-1:123:microvm-image:replacement",
        region_name="us-east-1",
        client=client,
        endpoint_transport_factory=HealthyTransport,
    )
    reconnect_metadata = {
        "microvm_id": "mvm-1",
        "endpoint": "mvm-1.example.test",
        "region": "us-west-2",
        "image_identifier": "arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        "image_version": "3",
    }

    attached = await factory.create(
        EnvironmentFactoryRequest(
            session_id="session-1",
            agent_name="assistant",
            environment_name="aws-sandbox",
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=reconnect_metadata,
        )
    )

    assert attached.environment.runner is not None
    assert attached.environment.runner.region_name == "us-west-2"
    assert attached.reconnect_metadata == reconnect_metadata

    with pytest.raises(LambdaMicroVMProtocolError, match="endpoint changed"):
        await factory.create(
            EnvironmentFactoryRequest(
                session_id="session-1",
                agent_name="assistant",
                environment_name="aws-sandbox",
                operation=EnvironmentFactoryOperation.RECONNECT,
                reconnect_metadata={**reconnect_metadata, "endpoint": "other.example.test"},
            )
        )


@pytest.mark.anyio
async def test_lambda_microvm_lifecycle_binding_retries_failed_finalization() -> None:
    client = FailOnceTerminateControlClient()
    factory = LambdaMicroVMEnvironmentFactory(
        image_identifier="arn:aws:lambda:us-west-2:123:microvm-image:cayu",
        region_name="us-west-2",
        client=client,
        endpoint_transport_factory=HealthyTransport,
    )
    created = await factory.create(
        EnvironmentFactoryRequest(
            session_id="session-1",
            agent_name="assistant",
            environment_name="aws-sandbox",
        )
    )
    runner = created.environment.runner
    binding = created.environment.binding
    assert runner is not None and binding is not None
    bound = await binding.bind(
        created.environment.workspace,
        runner,
        session_id="session-1",
        environment_name="aws-sandbox",
    )

    with pytest.raises(RuntimeError, match="temporary termination failure"):
        await binding.finalize(bound, outcome="completed")
    await binding.finalize(bound, outcome="completed")

    assert client.terminate_calls == 2
