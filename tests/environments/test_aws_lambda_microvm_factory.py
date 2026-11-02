"""The library's Lambda MicroVM environment factory, against the fake control plane and sidecar.

The control plane is ``ClientTokenLambdaModel`` (the client-token semantics
verified live) and every MicroVM's sidecar is the real ``CommandSupervisor``
with the real owner fence, one per endpoint, so a reconnect from another
runner (or another app) fences the runner that owned the MicroVM before.
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr
from tests.core.test_environment_allocation_recovery import (
    _create_session,
    _pending_record,
    _resolve,
    _SimulatedProcessDeath,
)
from tests.core.test_human_review import identity
from tests.egress.test_aws_lambda_microvm_recoverable_allocation import _CrashingModel, _reap
from tests.runners.lambda_microvm_harness import ClientTokenLambdaModel, SupervisorTransport

import cayu.runners.aws_lambda_microvm as runner_module
from cayu import AgentSpec, CayuApp, EventType, Message, RunRequest
from cayu.environments import (
    Environment,
    EnvironmentAllocationContext,
    EnvironmentAllocationIntent,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentAllocationUnsupportedError,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentSpec,
    LambdaMicroVMEnvironmentFactory,
    LambdaMicroVMLifecycleBinding,
)
from cayu.evals import ScriptedModelProvider
from cayu.providers import ModelStreamEvent
from cayu.runners import (
    ExecCommand,
    LambdaMicroVMOwnershipSuperseded,
    LambdaMicroVMProtocolError,
    LambdaMicroVMRunner,
)
from cayu.runtime.public_authority import PublicAuthorityAliasCodec, PublicAuthorityAliasKeyring
from cayu.sessions import InMemorySessionStore
from cayu.sessions.recovery import (
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
)
from cayu.storage import SQLiteSessionStore
from cayu.tools.base import Tool, ToolContext, ToolResult, ToolSpec
from cayu.workspaces import RunnerWorkspace

REGION = "us-east-1"
CREATE = EnvironmentFactoryOperation.CREATE
RECONNECT = EnvironmentFactoryOperation.RECONNECT


@pytest.fixture(autouse=True)
def _fast_client_token_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "_LAMBDA_CLIENT_TOKEN_RETRY_DELAYS_SECONDS", (0.0,) * 3)


class Sidecars:
    """One fake sidecar per MicroVM endpoint, shared by every runner and app.

    A real MicroVM has one sidecar whatever process talks to it; its owner
    fence is what rejects a superseded runner. Keying the fake by endpoint
    gives every runner of one MicroVM the same fence.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self.by_endpoint: dict[str, SupervisorTransport] = {}

    def _for(self, endpoint: str) -> SupervisorTransport:
        transport = self.by_endpoint.get(endpoint)
        if transport is None:
            # The supervisor's root is the workspace the guest commands run in.
            transport = SupervisorTransport(self._root)
            self.by_endpoint[endpoint] = transport
        return transport

    async def health(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).health(endpoint=endpoint, **kwargs)

    async def start_command(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).start_command(endpoint=endpoint, **kwargs)

    async def get_command(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).get_command(endpoint=endpoint, **kwargs)

    async def cancel_command(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).cancel_command(endpoint=endpoint, **kwargs)

    async def release_command(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).release_command(endpoint=endpoint, **kwargs)

    async def claim_owner(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).claim_owner(endpoint=endpoint, **kwargs)

    async def check_owner(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).check_owner(endpoint=endpoint, **kwargs)

    async def acquire_lifecycle(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).acquire_lifecycle(endpoint=endpoint, **kwargs)

    async def release_lifecycle(self, *, endpoint: str, **kwargs: Any):
        return await self._for(endpoint).release_lifecycle(endpoint=endpoint, **kwargs)

    async def configure_control_relay(self, *, endpoint: str, **kwargs: Any):
        raise NotImplementedError("These tests run no browser control relay.")


class LambdaWorld:
    """One control plane, one set of sidecars, one MicroVM disk and one session database."""

    def __init__(self, tmp_path: Path) -> None:
        self.model = ClientTokenLambdaModel()
        self.disk = tmp_path / "microvm-disk"
        self.disk.mkdir()
        self.sidecars = Sidecars(self.disk)
        self.database = tmp_path / "sessions.sqlite"
        self.stores: list[SQLiteSessionStore] = []

    def factory(self, **options: Any) -> LambdaMicroVMEnvironmentFactory:
        return LambdaMicroVMEnvironmentFactory(
            image_identifier=ClientTokenLambdaModel.image_arn,
            region_name=REGION,
            client=self.model,
            endpoint_transport_factory=lambda: self.sidecars,
            runner_options={"poll_interval_s": 0, "default_cwd": str(self.disk)},
            **options,
        )

    def app(
        self, *turns, agents: dict[str, list[Tool]], allocation_scope: str = "session"
    ) -> CayuApp:
        keyring = PublicAuthorityAliasKeyring(
            active_key_id="test",
            keys={"test": SecretStr(base64.urlsafe_b64encode(b"k" * 32).decode().rstrip("="))},
        )
        store = SQLiteSessionStore(
            self.database, public_authority_alias_codec=PublicAuthorityAliasCodec(keyring)
        )
        self.stores.append(store)
        app = CayuApp(
            session_store=store, enable_logging=False, public_authority_alias_keyring=keyring
        )
        app.register_provider(ScriptedModelProvider(list(turns)), default=True)
        app.register_environment_factory(
            EnvironmentSpec(name="sandbox", execution_profile_identity=identity("lambda-sandbox")),
            self.factory(),
            default=True,
            allocation_scope=allocation_scope,
        )
        for name, tools in agents.items():
            app.register_agent(AgentSpec(name=name, model="scripted"), tools=tools)
        return app

    async def close(self) -> None:
        for store in self.stores:
            await store.close()

    def state(self, microvm_id: str) -> str:
        return self.model.get_microvm(microvmIdentifier=microvm_id)["state"]


def call(tool: str) -> list[ModelStreamEvent]:
    return [
        ModelStreamEvent.tool_call(id=f"call-{tool}", name=tool, arguments={}),
        ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
    ]


def done() -> list[ModelStreamEvent]:
    return [ModelStreamEvent.text_delta("done"), ModelStreamEvent.completed({})]


class WritePlan(Tool):
    spec = ToolSpec(
        name="write_plan",
        parallel_safe=False,
        workspace_mutation=True,
        execution_profile_identity=identity("write-plan"),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        assert ctx.workspace is not None
        await ctx.workspace.create_bytes("plan.md", b"return a + b\n")
        return ToolResult(content="planned")


def _request(**overrides: Any) -> EnvironmentFactoryRequest:
    options: dict[str, Any] = {
        "session_id": "session-1",
        "agent_name": "assistant",
        "environment_name": "sandbox",
    }
    options.update(overrides)
    return EnvironmentFactoryRequest(**options)


def test_the_factory_declares_a_recoverable_scope_and_never_creates_outside_it(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)
    factory = world.factory()

    assert factory.allocation_scope(_request()) == EnvironmentAllocationScope(
        provider="aws-lambda-microvm", adapter_generation="environment-client-token-v1"
    )
    assert factory.allocation_scope(_request(operation=EnvironmentFactoryOperation.RECONNECT)) is (
        None
    )
    with pytest.raises(EnvironmentAllocationUnsupportedError, match="create_recoverable"):
        asyncio.run(factory.create(_request()))
    assert world.model.run_calls == []


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"create_options": {"client_token": "x"}}, "cannot override: client_token"),
        ({"runner_options": {"owner_epoch": 1}}, "cannot override: owner_epoch"),
        ({"runner_options": {"egress_network_connectors": []}}, "in create_options"),
        ({"create_options": {"maximum_duration_in_seconds": 0}}, "maximum_duration"),
        ({"client_token_replay_window_s": 3601}, "client_token_replay_window_s"),
    ],
)
def test_options_that_would_change_the_allocation_protocol_are_refused(
    tmp_path: Path, options: dict[str, Any], message: str
) -> None:
    world = LambdaWorld(tmp_path)
    with pytest.raises(ValueError, match=message):
        LambdaMicroVMEnvironmentFactory(
            image_identifier=ClientTokenLambdaModel.image_arn,
            region_name=REGION,
            client=world.model,
            **options,
        )


def test_reconnect_refuses_metadata_of_another_session_or_a_changed_identity(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)
    factory = world.factory()
    identity = _identity(world)

    async def scenario() -> None:
        reconnect = _request(
            operation=EnvironmentFactoryOperation.RECONNECT, reconnect_metadata=identity
        )
        attached = await factory.create(reconnect)
        assert attached.reconnect_metadata == identity
        assert isinstance(attached.environment.workspace, RunnerWorkspace)
        assert isinstance(attached.environment.binding, LambdaMicroVMLifecycleBinding)
        assert attached.release is not None
        await attached.release(EnvironmentFactoryReleaseAction.PRESERVE)
        with pytest.raises(ValueError, match="another session"):
            await factory.create(
                _request(
                    session_id="session-2",
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata=identity,
                )
            )
        with pytest.raises(LambdaMicroVMProtocolError, match="endpoint changed"):
            await factory.create(
                _request(
                    operation=EnvironmentFactoryOperation.RECONNECT,
                    reconnect_metadata={**identity, "endpoint": "other.lambda-microvm.invalid"},
                )
            )

    asyncio.run(scenario())
    # Reconnect never asked AWS for a new MicroVM.
    assert len(world.model.created_ids()) == 1


def test_the_binding_composes_into_a_directly_registered_environment(tmp_path: Path) -> None:
    """The runner, workspace and binding stay usable without the factory."""

    world = LambdaWorld(tmp_path)
    microvm_id = _identity(world)["microvm_id"]

    async def scenario():
        runner = await LambdaMicroVMRunner.from_existing(
            microvm_id,
            region_name=REGION,
            client=world.model,
            endpoint_transport=world.sidecars,
            close_action="none",
            poll_interval_s=0,
            default_cwd=str(world.disk),
        )
        app = CayuApp(enable_logging=False)
        app.register_provider(ScriptedModelProvider([call("write_plan"), done()]), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="direct"),
                workspace=RunnerWorkspace(runner, workspace_id=f"lambda-microvm:{microvm_id}"),
                runner=runner,
                binding=LambdaMicroVMLifecycleBinding(runner),
            ),
            default=True,
        )
        app.register_agent(AgentSpec(name="planner", model="scripted"), tools=[WritePlan()])
        return [
            event
            async for event in app.run(
                RunRequest(
                    session_id="direct",
                    agent_name="planner",
                    messages=[Message.text("user", "plan")],
                )
            )
        ]

    events = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    assert (world.disk / "plan.md").read_bytes() == b"return a + b\n"
    assert world.state(microvm_id) == "TERMINATED"


def test_a_session_allocates_through_cayuapp_and_terminates_on_completion(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)

    async def scenario():
        try:
            app = world.app(call("write_plan"), done(), agents={"planner": [WritePlan()]})
            events = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id="solo",
                        agent_name="planner",
                        messages=[Message.text("user", "plan")],
                    )
                )
            ]
            checkpoint = await app.session_store.load_checkpoint("solo")
            return events, checkpoint
        finally:
            await world.close()

    events, checkpoint = asyncio.run(scenario())
    assert events[-1].type == EventType.SESSION_COMPLETED, events[-1].payload
    (microvm_id,) = world.model.created_ids()
    # Created once, under the intent's client token, with the pinned version and lifetime.
    (submission,) = world.model.run_calls
    assert submission["clientToken"].startswith("cayu-")
    assert submission["imageVersion"] == "3"
    assert submission["maximumDurationInSeconds"] == 28_800
    assert (world.disk / "plan.md").read_bytes() == b"return a + b\n"
    # Completion terminated it, proven by readback.
    assert world.state(microvm_id) == "TERMINATED"
    # Nothing durable carries submission or endpoint authority.
    durable = json.dumps(checkpoint)
    assert submission["clientToken"] not in durable
    assert "model-token" not in durable


def test_the_example_is_a_thin_user_of_the_library_factory() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "aws"
        / "environments"
        / "lambda_microvm.py"
    )
    spec = importlib.util.spec_from_file_location("lambda_microvm_environment_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.LambdaMicroVMEnvironmentFactory is LambdaMicroVMEnvironmentFactory
    assert module.LambdaMicroVMLifecycleBinding is LambdaMicroVMLifecycleBinding
    app = CayuApp(enable_logging=False)
    factory = module.register_lambda_microvm_environment(
        app,
        name="sandbox",
        image_identifier=ClientTokenLambdaModel.image_arn,
        region_name=REGION,
        client=ClientTokenLambdaModel(),
        allocation_scope="parent",
    )
    assert isinstance(factory, LambdaMicroVMEnvironmentFactory)


# Crash recovery of one session's MicroVM. ``_resolve`` drives the runtime's
# environment lifecycle in a fresh CayuApp on the given store each time, so
# every call below is a separate process as far as Cayu is concerned.


class _FakeProvider(ScriptedModelProvider):
    """The provider the crashed sessions were created with."""

    def __init__(self) -> None:
        super().__init__([])
        self.name = "fake"


class _CountingModel(_CrashingModel):
    def __init__(self, crash: str = "") -> None:
        super().__init__(crash)
        self.auth_tokens = 0

    def create_microvm_auth_token(self, **kwargs: Any) -> dict[str, Any]:
        self.auth_tokens += 1
        return super().create_microvm_auth_token(**kwargs)


def _session_factory(world: LambdaWorld, model: ClientTokenLambdaModel):
    return LambdaMicroVMEnvironmentFactory(
        image_identifier=ClientTokenLambdaModel.image_arn,
        region_name=REGION,
        client=model,
        endpoint_transport_factory=lambda: world.sidecars,
        runner_options={"poll_interval_s": 0, "default_cwd": str(world.disk)},
    )


def _runner(resolution) -> LambdaMicroVMRunner:
    assert resolution.error is None, resolution.error
    runner = resolution.registered_environment.environment.runner
    assert isinstance(runner, LambdaMicroVMRunner)
    return runner


def _identity(world: LambdaWorld) -> dict[str, str]:
    created = world.model.run_microvm(imageIdentifier=ClientTokenLambdaModel.image_arn)
    return {
        "microvm_id": created["microvmId"],
        "endpoint": created["endpoint"],
        "region": REGION,
        "image_identifier": created["imageArn"],
        "image_version": created["imageVersion"],
        "session_id": "session-1",
        "environment_name": "sandbox",
    }


def test_a_lost_acknowledgement_is_adopted_by_client_token_replay_in_a_new_app(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)
    model = _CountingModel("after_provider")

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(store, session, _session_factory(world, model), operation=CREATE)
        pending = _pending_record(await store.load_checkpoint(session.id) or {})
        assert pending is not None and pending["state"] == "dispatched"
        pinned = pending["intent"]["provider_metadata"]["lambda"]
        assert pinned["image_version"] == "3" and pinned["maximum_duration_s"] == 28_800
        assert len(model.created_ids()) == 1

        model.latest_image_version = "4"  # a newer image must not leak into recovery
        runner = _runner(
            await _resolve(store, session, _session_factory(world, model), operation=CREATE)
        )
        assert [runner.microvm_id] == model.created_ids()
        assert len({submission["clientToken"] for submission in model.run_calls}) == 1
        assert {submission["imageVersion"] for submission in model.run_calls} == {"3"}
        assert _pending_record(await store.load_checkpoint(session.id) or {}) is None

    asyncio.run(run())


def test_a_new_app_reconnects_with_fresh_access_and_fences_the_previous_owner(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)
    model = _CountingModel()

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        first = _runner(
            await _resolve(store, session, _session_factory(world, model), operation=CREATE)
        )
        assert (await first.exec(ExecCommand.process("true"))).exit_code == 0
        tokens_after_create = model.auth_tokens

        second = _runner(
            await _resolve(store, session, _session_factory(world, model), operation=RECONNECT)
        )
        # Same MicroVM, reached with a newly minted endpoint token.
        assert second.microvm_id == first.microvm_id
        assert model.auth_tokens > tokens_after_create
        assert len(model.created_ids()) == 1
        # The previous owner's lifecycle calls and commands are rejected; it
        # hasn't seen a rejection before its terminate, so only the sidecar's
        # lifecycle fence stops it.
        with pytest.raises(LambdaMicroVMOwnershipSuperseded):
            await first.terminate()
        with pytest.raises((LambdaMicroVMOwnershipSuperseded, RuntimeError)):
            await first.exec(ExecCommand.process("true"))
        assert (await second.exec(ExecCommand.process("true"))).exit_code == 0
        assert model.get_microvm(microvmIdentifier=second.microvm_id)["state"] == "RUNNING"
        assert "model-token" not in json.dumps(await store.load_checkpoint(session.id))

    asyncio.run(run())


def test_an_ownership_epoch_keeps_an_older_owner_from_reattaching(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    factory = world.factory()
    identity = _identity(world)

    def reconnect(epoch: int) -> EnvironmentFactoryRequest:
        return _request(
            operation=EnvironmentFactoryOperation.RECONNECT,
            reconnect_metadata=identity,
            ownership_epoch=epoch,
        )

    async def run() -> None:
        newer = (await factory.create(reconnect(5))).environment.runner
        assert isinstance(newer, LambdaMicroVMRunner)
        assert (await newer.exec(ExecCommand.process("true"))).exit_code == 0
        # An owner that decided to reattach before the takeover arrives late.
        with pytest.raises(LambdaMicroVMOwnershipSuperseded):
            stale = (await factory.create(reconnect(4))).environment.runner
            assert stale is not None
            await stale.exec(ExecCommand.process("true"))
        assert (await newer.exec(ExecCommand.process("true"))).exit_code == 0

    asyncio.run(run())


def test_reaping_an_unacknowledged_intent_terminates_the_token_owned_microvm(
    tmp_path: Path,
) -> None:
    world = LambdaWorld(tmp_path)
    model = _CountingModel("after_provider")

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(store, session, _session_factory(world, model), operation=CREATE)
        allocation, pending = await _reap(store, session, _session_factory(world, model))
        assert allocation.state is EnvironmentAllocationState.REAPED
        assert pending is not None and pending["state"] == "reaped"
        assert len(model.created_ids()) == 1
        assert model.live_ids() == []
        assert len({submission["clientToken"] for submission in model.run_calls}) == 1

    asyncio.run(run())


def test_incomplete_session_recovery_reaps_a_crashed_creation(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    # The worker dies right after AWS created the MicroVM, before Cayu learned its id.
    world.model = _CountingModel("after_provider")

    async def run():
        try:
            crashed = world.app(call("write_plan"), done(), agents={"planner": [WritePlan()]})
            with pytest.raises(BaseException) as died:
                async for _ in crashed.run(
                    RunRequest(
                        session_id="solo",
                        agent_name="planner",
                        messages=[Message.text("user", "plan")],
                    )
                ):
                    pass
            assert _died_of(died.value, _SimulatedProcessDeath)
            assert len(world.model.live_ids()) == 1
            recovering = world.app(agents={"planner": [WritePlan()]})
            return await recovering.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="solo", reason="worker_lost")
            )
        finally:
            await world.close()

    result = asyncio.run(run())
    assert IncompleteSessionRecoveryAction.REAPED_ALLOCATION in result.actions, result
    # The one token-owned MicroVM was found by replay and terminated; none was added.
    assert len(world.model.created_ids()) == 1
    assert world.model.live_ids() == []
    assert len({submission["clientToken"] for submission in world.model.run_calls}) == 1


def _died_of(error: BaseException, kind: type[BaseException]) -> bool:
    seen: BaseException | None = error
    while seen is not None:
        if isinstance(seen, kind):
            return True
        if isinstance(seen, BaseExceptionGroup):
            return any(_died_of(inner, kind) for inner in seen.exceptions)
        seen = seen.__cause__ or seen.__context__
    return False


def test_disposal_is_proven_only_from_terminal_readback(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    factory = world.factory()
    identity = _identity(world)
    request = _request(operation=EnvironmentFactoryOperation.RECONNECT, reconnect_metadata=identity)
    microvm_id = identity["microvm_id"]

    async def run() -> list[bool]:
        alive = await factory.is_allocation_disposed(request)
        world.model.suspend_microvm(microvmIdentifier=microvm_id)
        suspended = await factory.is_allocation_disposed(request)
        world.model.terminate_microvm(microvmIdentifier=microvm_id)
        world.model.get_microvm(microvmIdentifier=microvm_id)
        terminated = await factory.is_allocation_disposed(request)
        unknown = await factory.is_allocation_disposed(
            replace(request, reconnect_metadata={**identity, "microvm_id": "microvm-9999"})
        )
        return [alive, suspended, terminated, unknown]

    assert asyncio.run(run()) == [False, False, True, True]


class _Allocation(EnvironmentAllocationContext):
    """An in-memory allocation context, for driving ``create_recoverable`` directly."""

    def __init__(self, request: EnvironmentFactoryRequest, scope: EnvironmentAllocationScope):
        self._intent = EnvironmentAllocationIntent(
            allocation_id="ealloc_" + "0" * 32,
            provider=scope.provider,
            adapter_generation=scope.adapter_generation,
            session_id=request.session_id,
            environment_name=request.environment_name,
            requested_operation=CREATE,
        )
        self._state = EnvironmentAllocationState.UNPREPARED
        self._acknowledged: dict[str, Any] | None = None

    @property
    def intent(self) -> EnvironmentAllocationIntent:
        return self._intent

    @property
    def state(self) -> EnvironmentAllocationState:
        return self._state

    @property
    def acknowledged_reconnect_metadata(self) -> dict[str, Any] | None:
        return self._acknowledged

    async def prepare(self, provider_metadata) -> EnvironmentAllocationIntent:
        self._intent = self._intent.with_provider_metadata(provider_metadata)
        self._state = EnvironmentAllocationState.PREPARED
        return self._intent

    async def mark_dispatched(self) -> None:
        self._state = EnvironmentAllocationState.DISPATCHED

    async def acknowledge(self, reconnect_metadata) -> None:
        self._acknowledged = dict(reconnect_metadata)
        self._state = EnvironmentAllocationState.ACKNOWLEDGED

    async def mark_reaping(self) -> bool:
        self._state = EnvironmentAllocationState.REAPING
        return True

    async def mark_reaped(self) -> None:
        self._state = EnvironmentAllocationState.REAPED


def _replacement(world: LambdaWorld, predecessor: dict[str, str]):
    factory = world.factory()
    request = _request(replacement_predecessor=predecessor)
    scope = factory.allocation_scope(request)
    assert scope is not None
    return factory.create_recoverable(request, _Allocation(request, scope))


def test_a_replacement_keeps_its_predecessors_exact_image_version(tmp_path: Path) -> None:
    world = LambdaWorld(tmp_path)
    predecessor = _identity(world)
    world.model.terminate_microvm(microvmIdentifier=predecessor["microvm_id"])
    # A newer version became active while the retired MicroVM ran.
    world.model.latest_image_version = "4"

    result = asyncio.run(_replacement(world, predecessor))

    assert result.reconnect_metadata["image_version"] == "3"
    assert world.model.run_calls[-1]["imageVersion"] == "3"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image_identifier": "arn:aws:lambda:us-east-1:123:microvm-image:other"}, "image"),
        ({"region": "us-west-2"}, "region"),
    ],
)
def test_a_replacement_of_another_image_or_region_is_refused_before_dispatch(
    tmp_path: Path, change: dict[str, str], message: str
) -> None:
    world = LambdaWorld(tmp_path)
    predecessor = {**_identity(world), **change}
    submitted = len(world.model.run_calls)

    with pytest.raises(ValueError, match=message):
        asyncio.run(_replacement(world, predecessor))
    assert len(world.model.run_calls) == submitted


class _TerminateFault(ClientTokenLambdaModel):
    """Kill the worker around the first terminate call."""

    def __init__(self, fault: str) -> None:
        super().__init__()
        self.fault = fault

    def terminate_microvm(self, **kwargs: Any) -> dict[str, Any]:
        if self.fault == "before_terminate":
            self.fault = ""
            raise _SimulatedProcessDeath("the worker died before AWS saw the terminate")
        response = super().terminate_microvm(**kwargs)
        if self.fault == "after_terminate":
            self.fault = ""
            # Let AWS finish terminating before the worker dies.
            self.get_microvm(microvmIdentifier=kwargs["microvmIdentifier"])
            raise _SimulatedProcessDeath("AWS terminated it; the worker died before recording that")
        return response


@pytest.mark.parametrize("fault", ["before_disposal", "before_terminate", "after_terminate"])
def test_a_worker_lost_during_completion_teardown_still_terminates_the_microvm(
    tmp_path: Path, fault: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = LambdaWorld(tmp_path)
    world.model = _TerminateFault(fault)
    if fault == "before_disposal":
        original = LambdaMicroVMLifecycleBinding.finalize
        died: list[bool] = []

        async def die_once(self, bound, *, outcome=None, metadata=None):
            if not died:
                died.append(True)
                raise _SimulatedProcessDeath("the worker died before teardown began")
            return await original(self, bound, outcome=outcome, metadata=metadata)

        monkeypatch.setattr(LambdaMicroVMLifecycleBinding, "finalize", die_once)

    async def run():
        try:
            crashed = world.app(call("write_plan"), done(), agents={"planner": [WritePlan()]})
            with pytest.raises(BaseException) as raised:
                async for _ in crashed.run(
                    RunRequest(
                        session_id="solo",
                        agent_name="planner",
                        messages=[Message.text("user", "plan")],
                    )
                ):
                    pass
            assert _died_of(raised.value, _SimulatedProcessDeath)
            recovering = world.app(agents={"planner": [WritePlan()]})
            return await recovering.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(session_id="solo", reason="worker_lost")
            )
        finally:
            await world.close()

    result = asyncio.run(run())
    (microvm_id,) = world.model.created_ids()
    assert world.model.live_ids() == [], (fault, world.state(microvm_id), result)
