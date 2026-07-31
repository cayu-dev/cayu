from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast
from uuid import uuid4

import pytest

from cayu import (
    AgentSpec,
    CayuApp,
    Environment,
    EnvironmentAllocationContext,
    EnvironmentAllocationIntent,
    EnvironmentAllocationScope,
    EnvironmentAllocationState,
    EnvironmentFactory,
    EnvironmentFactoryOperation,
    EnvironmentFactoryReleaseAction,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    Event,
    EventType,
    InMemorySessionStore,
    Message,
    PostgresSessionStore,
    RunRequest,
    SessionIdentity,
    SessionStore,
    SQLiteSessionStore,
)
from cayu.runtime._environment_allocation import (
    EnvironmentAllocationCoordinator,
    EnvironmentAllocationReceipt,
    EnvironmentAllocationRecord,
)
from cayu.runtime._environment_lifecycle import (
    ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY,
    ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY,
)
from cayu.runtime.sessions import CheckpointTransform, Session
from cayu.storage.migrations import SchemaMode
from cayu.vaults import SecretRedactor

_SESSION_ID = "recoverable-allocation-session"
_ENVIRONMENT_NAME = "remote"
_PROVIDER = "fake-remote"
_ADAPTER_GENERATION = "fake-remote-v1"


class _SimulatedProcessDeath(BaseException):
    pass


@dataclass(frozen=True)
class _FakeResource:
    resource_name: str
    allocation_id: str
    session_id: str
    environment_name: str
    adapter_generation: str

    def reconnect_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "provider": _PROVIDER,
            "adapter_generation": self.adapter_generation,
            "resource_name": self.resource_name,
            "allocation_id": self.allocation_id,
            "session_id": self.session_id,
            "environment_name": self.environment_name,
        }


class _FakeRemoteProvider:
    def __init__(self) -> None:
        self.resources: dict[str, _FakeResource] = {}
        self.create_calls: list[str] = []
        self.lookup_calls: list[str] = []
        self.reap_calls: list[str] = []

    def create(self, intent: EnvironmentAllocationContext) -> _FakeResource:
        resource_name = _resource_name(intent.intent.allocation_id)
        self.create_calls.append(resource_name)
        existing = self.resources.get(resource_name)
        if existing is not None:
            if (
                existing.allocation_id != intent.intent.allocation_id
                or existing.session_id != intent.intent.session_id
                or existing.environment_name != intent.intent.environment_name
                or existing.adapter_generation != intent.intent.adapter_generation
            ):
                raise RuntimeError("Fake provider allocation identity belongs to another owner.")
            return existing
        resource = _FakeResource(
            resource_name=resource_name,
            allocation_id=intent.intent.allocation_id,
            session_id=intent.intent.session_id,
            environment_name=intent.intent.environment_name,
            adapter_generation=intent.intent.adapter_generation,
        )
        self.resources[resource_name] = resource
        return resource

    def lookup(
        self,
        resource_name: str,
        *,
        intent: EnvironmentAllocationContext,
    ) -> _FakeResource | None:
        self.lookup_calls.append(resource_name)
        resource = self.resources.get(resource_name)
        if resource is None:
            return None
        if (
            resource.allocation_id != intent.intent.allocation_id
            or resource.session_id != intent.intent.session_id
            or resource.environment_name != intent.intent.environment_name
            or resource.adapter_generation != intent.intent.adapter_generation
        ):
            raise RuntimeError("Fake provider resource belongs to another allocation owner.")
        return resource

    def reap(
        self,
        resource_name: str,
        *,
        allocation_id: str,
        session_id: str,
        environment_name: str,
        adapter_generation: str,
    ) -> None:
        self.reap_calls.append(resource_name)
        resource = self.resources.get(resource_name)
        if resource is None:
            return
        if (
            resource.allocation_id != allocation_id
            or resource.session_id != session_id
            or resource.environment_name != environment_name
            or resource.adapter_generation != adapter_generation
        ):
            raise RuntimeError("Refusing to reap a fake resource owned by another intent.")
        del self.resources[resource_name]


class _FakeRemoteFactory(EnvironmentFactory):
    def __init__(
        self,
        provider: _FakeRemoteProvider,
        *,
        crash_at: str | None = None,
        provider_metadata: Mapping[str, Any] | None = None,
        recovery_entered: asyncio.Event | None = None,
        recovery_release: asyncio.Event | None = None,
        result_environment_name: str | None = None,
        scope_provider: str = _PROVIDER,
        scope_adapter_generation: str = _ADAPTER_GENERATION,
    ) -> None:
        self._provider = provider
        self._crash_at = crash_at
        self._provider_metadata = provider_metadata
        self._recovery_entered = recovery_entered
        self._recovery_release = recovery_release
        self._result_environment_name = result_environment_name
        self._scope_provider = scope_provider
        self._scope_adapter_generation = scope_adapter_generation

    def allocation_scope(
        self,
        request: EnvironmentFactoryRequest,
    ) -> EnvironmentAllocationScope | None:
        if request.operation is not EnvironmentFactoryOperation.CREATE:
            return None
        return EnvironmentAllocationScope(
            provider=self._scope_provider,
            adapter_generation=self._scope_adapter_generation,
        )

    async def create_recoverable(
        self,
        request: EnvironmentFactoryRequest,
        allocation: EnvironmentAllocationContext,
    ) -> EnvironmentFactoryResult:
        if self._crash_at == "before_intent":
            raise _SimulatedProcessDeath("before durable allocation intent")

        if allocation.state is EnvironmentAllocationState.UNPREPARED:
            provider_metadata = (
                self._provider_metadata
                if self._provider_metadata is not None
                else {"resource_name": _resource_name(allocation.intent.allocation_id)}
            )
            await allocation.prepare(provider_metadata)
            if self._crash_at == "after_intent":
                raise _SimulatedProcessDeath("after durable allocation intent")

        metadata = allocation.intent.provider_metadata
        resource_name = metadata.get("resource_name")
        if type(resource_name) is not str:
            raise RuntimeError("Fake allocation intent is missing its resource name.")

        if (
            allocation.state
            in {
                EnvironmentAllocationState.PREPARED,
                EnvironmentAllocationState.DISPATCHED,
            }
            and self._recovery_entered is not None
            and self._recovery_release is not None
        ):
            self._recovery_entered.set()
            await self._recovery_release.wait()

        if allocation.state is EnvironmentAllocationState.PREPARED:
            await allocation.mark_dispatched()
            if self._crash_at == "after_dispatch_before_provider":
                raise _SimulatedProcessDeath("after dispatch before provider creation")
            resource = self._provider.create(allocation)
            if self._crash_at == "after_remote_create":
                raise _SimulatedProcessDeath("after provider creation")
        elif allocation.state is EnvironmentAllocationState.DISPATCHED:
            resource = self._provider.lookup(resource_name, intent=allocation)
            if resource is None:
                resource = self._provider.create(allocation)
        elif allocation.state is EnvironmentAllocationState.ACKNOWLEDGED:
            reconnect_metadata = allocation.acknowledged_reconnect_metadata
            if reconnect_metadata is None:
                raise RuntimeError("Fake provider acknowledgement is missing.")
            resource = self._provider.lookup(resource_name, intent=allocation)
            if resource is None or resource.reconnect_metadata() != reconnect_metadata:
                raise RuntimeError("Fake provider acknowledgement no longer identifies a resource.")
        elif allocation.state is EnvironmentAllocationState.REAPING:
            self._provider.reap(
                resource_name,
                allocation_id=allocation.intent.allocation_id,
                session_id=allocation.intent.session_id,
                environment_name=allocation.intent.environment_name,
                adapter_generation=allocation.intent.adapter_generation,
            )
            await allocation.mark_reaped()
            raise RuntimeError("A reaping fake allocation cannot be replaced.")
        elif allocation.state is EnvironmentAllocationState.REAPED:
            raise RuntimeError("A reaped fake allocation cannot be replaced.")
        else:
            raise AssertionError(f"Unexpected allocation state: {allocation.state}")

        reconnect_metadata = resource.reconnect_metadata()
        if allocation.state is EnvironmentAllocationState.DISPATCHED:
            await allocation.acknowledge(reconnect_metadata)
            if self._crash_at == "after_acknowledgement":
                raise _SimulatedProcessDeath("after provider acknowledgement")
        return self._result(request, resource, allocation=allocation)

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        if request.operation is not EnvironmentFactoryOperation.RECONNECT:
            raise RuntimeError("Fake remote creation requires a durable allocation context.")
        resource_name = request.reconnect_metadata.get("resource_name")
        if type(resource_name) is not str:
            raise RuntimeError("Fake reconnect metadata is missing its resource name.")
        resource = self._provider.resources.get(resource_name)
        if resource is None or resource.reconnect_metadata() != request.reconnect_metadata:
            raise RuntimeError("Fake reconnect refused to allocate a replacement resource.")
        return self._result(request, resource)

    def _result(
        self,
        request: EnvironmentFactoryRequest,
        resource: _FakeResource,
        *,
        allocation: EnvironmentAllocationContext | None = None,
    ) -> EnvironmentFactoryResult:
        async def release(action: EnvironmentFactoryReleaseAction) -> None:
            if action is EnvironmentFactoryReleaseAction.DISCARD:
                self._provider.reap(
                    resource.resource_name,
                    allocation_id=resource.allocation_id,
                    session_id=resource.session_id,
                    environment_name=resource.environment_name,
                    adapter_generation=resource.adapter_generation,
                )
                if allocation is not None:
                    await allocation.mark_reaped()

        return EnvironmentFactoryResult(
            environment=Environment(
                EnvironmentSpec(name=self._result_environment_name or request.environment_name)
            ),
            reconnect_metadata=resource.reconnect_metadata(),
            release=release,
        )


class _AcknowledgementLossStore(InMemorySessionStore):
    def __init__(self, fail_after_transform: int) -> None:
        super().__init__()
        self._fail_after_transform = fail_after_transform
        self._armed = False
        self._transform_count = 0

    def arm(self) -> None:
        self._armed = True

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        await super().transform_checkpoint(session_id, checkpoint_transform)
        if self._armed:
            self._transform_count += 1
            if self._transform_count == self._fail_after_transform:
                raise TimeoutError("checkpoint acknowledgement was lost")


class _PublicationReadbackLossStore(InMemorySessionStore):
    def __init__(self) -> None:
        super().__init__()
        self._armed = False
        self._transform_count = 0
        self._fail_next_load = False

    def arm(self) -> None:
        self._armed = True

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        await super().transform_checkpoint(session_id, checkpoint_transform)
        if self._armed:
            self._transform_count += 1
            if self._transform_count == 4:
                self._fail_next_load = True

    async def load_checkpoint(self, session_id: str) -> dict[str, Any] | None:
        if self._fail_next_load:
            self._fail_next_load = False
            raise TimeoutError("publication readback was unavailable")
        return await super().load_checkpoint(session_id)


class _FatalSignalAfterTransformStore(InMemorySessionStore):
    def __init__(self, fail_after_transform: int) -> None:
        super().__init__()
        self._fail_after_transform = fail_after_transform
        self._armed = False
        self._transform_count = 0

    def arm(self) -> None:
        self._armed = True

    async def transform_checkpoint(
        self,
        session_id: str,
        checkpoint_transform: CheckpointTransform,
    ) -> None:
        await super().transform_checkpoint(session_id, checkpoint_transform)
        if self._armed:
            self._transform_count += 1
            if self._transform_count == self._fail_after_transform:
                raise SystemExit("fatal signal after durable checkpoint commit")


def _resource_name(allocation_id: str) -> str:
    return f"resource-{allocation_id}"


async def _create_session(
    store: SessionStore,
    *,
    session_id: str = _SESSION_ID,
    parent_session_id: str | None = None,
) -> Session:
    await store.create(
        RunRequest(
            agent_name="agent",
            session_id=session_id,
            parent_session_id=parent_session_id,
            environment_name=_ENVIRONMENT_NAME,
            messages=[Message.text("user", "run")],
        ),
        identity=SessionIdentity(provider_name="fake", model="fake-model"),
    )
    session = await store.load(session_id)
    assert session is not None
    return session


async def _resolve(
    store: SessionStore,
    session: Session,
    factory: EnvironmentFactory,
    *,
    operation: EnvironmentFactoryOperation,
    secret_redactor: SecretRedactor | None = None,
    crash_before_completed_event: bool = False,
):
    app = CayuApp(
        session_store=store,
        enable_logging=False,
        secret_redactor=secret_redactor,
    )
    app.register_agent(AgentSpec(name="agent", model="fake-model"))
    app.register_environment_factory(
        EnvironmentSpec(name=_ENVIRONMENT_NAME),
        factory,
    )
    registered_agent = app._get_registered_agent("agent")
    registered_environment = app._get_registered_environment(_ENVIRONMENT_NAME)
    lifecycle = app._environment_lifecycle
    started = await lifecycle.emit_factory_started(
        session=session,
        registered_agent=registered_agent,
        registered_environment=registered_environment,
    )
    if crash_before_completed_event:
        original_emit = lifecycle._event_writer.emit

        async def emit(event: Event) -> Event:
            if event.type is EventType.ENVIRONMENT_FACTORY_COMPLETED:
                raise _SimulatedProcessDeath("after allocation publication")
            return await original_emit(event)

        cast("Any", lifecycle._event_writer).emit = emit
    return await lifecycle.resolve_factory(
        session=session,
        registered_agent=registered_agent,
        registered_environment=registered_environment,
        started_event=started,
        operation=operation,
    )


def _pending_record(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    records = checkpoint.get(ENVIRONMENT_FACTORY_ALLOCATION_INTENTS_CHECKPOINT_KEY, {})
    return records.get(_ENVIRONMENT_NAME)


@pytest.fixture(params=["memory", "sqlite", "postgres"])
def allocation_store_case(request, tmp_path):
    postgres_dsn = request.getfixturevalue("postgres_dsn") if request.param == "postgres" else None
    return request.param, tmp_path, postgres_dsn


async def _open_store(case) -> SessionStore:
    kind, tmp_path, postgres_dsn = case
    if kind == "memory":
        return InMemorySessionStore()
    if kind == "sqlite":
        return SQLiteSessionStore(tmp_path / "allocation-recovery.sqlite")
    return PostgresSessionStore(
        postgres_dsn,
        min_size=1,
        max_size=2,
        schema_mode=SchemaMode.CREATE,
    )


async def _reopen_store(case, store: SessionStore) -> SessionStore:
    kind, _tmp_path, _postgres_dsn = case
    if kind == "memory":
        return store
    await _close_store(store)
    return await _open_store(case)


async def _close_store(store: SessionStore) -> None:
    close = getattr(store, "close", None)
    if close is not None:
        await close()


@pytest.mark.parametrize(
    ("crash_at", "created_before_recovery"),
    [
        ("after_dispatch_before_provider", 0),
        ("after_remote_create", 1),
    ],
)
def test_allocation_recovery_conforms_across_session_stores(
    allocation_store_case,
    crash_at: str,
    created_before_recovery: int,
) -> None:
    async def run() -> None:
        store = await _open_store(allocation_store_case)
        provider = _FakeRemoteProvider()
        session_id = f"{_SESSION_ID}-{crash_at}-{uuid4().hex}"
        try:
            session = await _create_session(store, session_id=session_id)
            with pytest.raises(_SimulatedProcessDeath):
                await _resolve(
                    store,
                    session,
                    _FakeRemoteFactory(provider, crash_at=crash_at),
                    operation=EnvironmentFactoryOperation.CREATE,
                )
            checkpoint = await store.load_checkpoint(session_id)
            assert checkpoint is not None
            pending = _pending_record(checkpoint)
            assert pending is not None
            allocation_id = pending["intent"]["allocation_id"]
            assert pending["state"] == "dispatched"
            assert len(provider.create_calls) == created_before_recovery

            store = await _reopen_store(allocation_store_case, store)
            recovered_session = await store.load(session_id)
            assert recovered_session is not None
            resolution = await _resolve(
                store,
                recovered_session,
                _FakeRemoteFactory(provider),
                operation=EnvironmentFactoryOperation.CREATE,
            )
            assert resolution.error is None
            assert len(provider.create_calls) == 1
            published = await store.load_checkpoint(session_id)
            assert published is not None
            receipt = published[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
                _ENVIRONMENT_NAME
            ]
            assert receipt["intent"]["allocation_id"] == allocation_id
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize(
    ("crash_at", "expected_state", "created_before_recovery"),
    [
        ("before_intent", None, 0),
        ("after_intent", "prepared", 0),
        ("after_dispatch_before_provider", "dispatched", 0),
        ("after_remote_create", "dispatched", 1),
        ("after_acknowledgement", "acknowledged", 1),
    ],
)
def test_remote_allocation_recovers_each_prepublication_crash_window(
    crash_at: str,
    expected_state: str | None,
    created_before_recovery: int,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at=crash_at),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        checkpoint = await store.load_checkpoint(_SESSION_ID) or {}
        pending = _pending_record(checkpoint)
        assert (None if pending is None else pending["state"]) == expected_state
        allocation_id = None if pending is None else pending["intent"]["allocation_id"]
        assert len(provider.create_calls) == created_before_recovery

        resolution = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert resolution.error is None
        assert len(provider.create_calls) == 1

        published = await store.load_checkpoint(_SESSION_ID)
        assert published is not None
        assert _pending_record(published) is None
        receipt = published[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
            _ENVIRONMENT_NAME
        ]
        if allocation_id is not None:
            assert receipt["intent"]["allocation_id"] == allocation_id
        assert (
            published[ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY][_ENVIRONMENT_NAME]
            == _SESSION_ID
        )
        assert (
            published[ENVIRONMENT_FACTORY_RECONNECT_CHECKPOINT_KEY][_ENVIRONMENT_NAME]
            == receipt["reconnect_metadata"]
        )

    asyncio.run(run())


def test_failed_prepublication_validation_reaps_exact_acknowledged_allocation() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()

        failed = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider, result_environment_name="wrong-environment"),
            operation=EnvironmentFactoryOperation.CREATE,
        )

        assert isinstance(failed.error, ValueError)
        assert provider.resources == {}
        assert len(provider.create_calls) == 1
        assert provider.reap_calls == provider.create_calls
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        pending = _pending_record(checkpoint)
        assert pending is not None
        assert pending["state"] == "reaped"

        recovered = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert isinstance(recovered.error, RuntimeError)
        assert "reaped fake allocation cannot be replaced" in str(recovered.error)
        assert len(provider.create_calls) == 1

    asyncio.run(run())


def test_published_allocation_wins_against_concurrent_discard(
    allocation_store_case,
) -> None:
    async def run() -> None:
        store = await _open_store(allocation_store_case)
        provider = _FakeRemoteProvider()
        session_id = f"{_SESSION_ID}-publication-cleanup-race-{uuid4().hex}"
        try:
            session = await _create_session(store, session_id=session_id)
            with pytest.raises(_SimulatedProcessDeath):
                await _resolve(
                    store,
                    session,
                    _FakeRemoteFactory(provider, crash_at="after_remote_create"),
                    operation=EnvironmentFactoryOperation.CREATE,
                )

            entered = [asyncio.Event(), asyncio.Event()]
            release_recovery = asyncio.Event()
            invalid_result_ready = asyncio.Event()
            release_invalid_result = asyncio.Event()

            class DelayedInvalidFactory(_FakeRemoteFactory):
                async def create_recoverable(
                    self,
                    request: EnvironmentFactoryRequest,
                    allocation: EnvironmentAllocationContext,
                ) -> EnvironmentFactoryResult:
                    result = await super().create_recoverable(request, allocation)
                    invalid_result_ready.set()
                    await release_invalid_result.wait()
                    return result

            published_task = asyncio.create_task(
                _resolve(
                    store,
                    session,
                    _FakeRemoteFactory(
                        provider,
                        recovery_entered=entered[0],
                        recovery_release=release_recovery,
                    ),
                    operation=EnvironmentFactoryOperation.CREATE,
                )
            )
            discarded_task = asyncio.create_task(
                _resolve(
                    store,
                    session,
                    DelayedInvalidFactory(
                        provider,
                        recovery_entered=entered[1],
                        recovery_release=release_recovery,
                        result_environment_name="wrong-environment",
                    ),
                    operation=EnvironmentFactoryOperation.CREATE,
                )
            )
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in entered)),
                timeout=1,
            )
            release_recovery.set()
            await asyncio.wait_for(invalid_result_ready.wait(), timeout=1)

            published = await asyncio.wait_for(published_task, timeout=1)
            assert published.error is None
            release_invalid_result.set()
            discarded = await asyncio.wait_for(discarded_task, timeout=1)

            assert isinstance(discarded.error, ValueError)
            assert provider.reap_calls == []
            assert len(provider.resources) == 1
            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint is not None
            assert _pending_record(checkpoint) is None
            receipt = checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
                _ENVIRONMENT_NAME
            ]
            resource_name = receipt["reconnect_metadata"]["resource_name"]
            assert resource_name in provider.resources
        finally:
            await _close_store(store)

    asyncio.run(run())


@pytest.mark.parametrize("delete_before_recovery", [False, True])
def test_reaping_fence_recovers_cleanup_crash_windows(
    allocation_store_case,
    delete_before_recovery: bool,
) -> None:
    async def run() -> None:
        store = await _open_store(allocation_store_case)
        provider = _FakeRemoteProvider()
        session_id = f"{_SESSION_ID}-reaping-{delete_before_recovery}-{uuid4().hex}"
        try:
            session = await _create_session(store, session_id=session_id)
            with pytest.raises(_SimulatedProcessDeath):
                await _resolve(
                    store,
                    session,
                    _FakeRemoteFactory(provider, crash_at="after_acknowledgement"),
                    operation=EnvironmentFactoryOperation.CREATE,
                )

            checkpoint = await store.load_checkpoint(session.id)
            assert checkpoint is not None
            coordinator = EnvironmentAllocationCoordinator(
                session_store=store,
                checkpoint_transform=lambda candidate: lambda _session, _current: candidate,
                secret_redactor=SecretRedactor(),
            )
            record = coordinator.record_from_checkpoint(
                checkpoint,
                environment_name=_ENVIRONMENT_NAME,
            )
            assert record is not None
            allocation = coordinator.context(
                session_id=session.id,
                parent_session_id=None,
                environment_name=_ENVIRONMENT_NAME,
                scope=record.intent.scope,
                existing=record,
            )
            stale_publisher = coordinator.context(
                session_id=session.id,
                parent_session_id=None,
                environment_name=_ENVIRONMENT_NAME,
                scope=record.intent.scope,
                existing=record,
            )
            assert await allocation.mark_reaping()
            with pytest.raises(
                RuntimeError,
                match="publication lost its acknowledged intent",
            ):
                await stale_publisher.publish()
            resource_name = _resource_name(record.intent.allocation_id)
            if delete_before_recovery:
                provider.reap(
                    resource_name,
                    allocation_id=record.intent.allocation_id,
                    session_id=record.intent.session_id,
                    environment_name=record.intent.environment_name,
                    adapter_generation=record.intent.adapter_generation,
                )

            store = await _reopen_store(allocation_store_case, store)
            recovered_session = await store.load(session.id)
            assert recovered_session is not None
            recovered = await _resolve(
                store,
                recovered_session,
                _FakeRemoteFactory(provider),
                operation=EnvironmentFactoryOperation.CREATE,
            )

            assert isinstance(recovered.error, RuntimeError)
            assert "reaping fake allocation cannot be replaced" in str(recovered.error)
            assert provider.resources == {}
            durable = await store.load_checkpoint(session.id)
            assert durable is not None
            pending = _pending_record(durable)
            assert pending is not None
            assert pending["state"] == "reaped"
        finally:
            await _close_store(store)

    asyncio.run(run())


def test_reaping_fence_reconstructs_lost_store_acknowledgement() -> None:
    async def run() -> None:
        store = _AcknowledgementLossStore(fail_after_transform=1)
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at="after_acknowledgement"),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        coordinator = EnvironmentAllocationCoordinator(
            session_store=store,
            checkpoint_transform=lambda candidate: lambda _session, _current: candidate,
            secret_redactor=SecretRedactor(),
        )
        record = coordinator.record_from_checkpoint(
            checkpoint,
            environment_name=_ENVIRONMENT_NAME,
        )
        assert record is not None
        allocation = coordinator.context(
            session_id=session.id,
            parent_session_id=None,
            environment_name=_ENVIRONMENT_NAME,
            scope=record.intent.scope,
            existing=record,
        )
        store.arm()

        assert await allocation.mark_reaping()
        assert allocation.state is EnvironmentAllocationState.REAPING
        durable = await store.load_checkpoint(session.id)
        assert durable is not None
        pending = _pending_record(durable)
        assert pending is not None
        assert pending["state"] == "reaping"

    asyncio.run(run())


def test_remote_allocation_recovers_after_publication_before_completion_event() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider),
                operation=EnvironmentFactoryOperation.CREATE,
                crash_before_completed_event=True,
            )

        checkpoint = await store.load_checkpoint(_SESSION_ID) or {}
        assert _pending_record(checkpoint) is None
        assert len(provider.create_calls) == 1

        resolution = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.RECONNECT,
        )
        assert resolution.error is None
        assert len(provider.create_calls) == 1

    asyncio.run(run())


def test_reconnect_cannot_change_a_published_allocation_receipt() -> None:
    class DriftingReconnectFactory(_FakeRemoteFactory):
        async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
            result = await super().create(request)
            return EnvironmentFactoryResult(
                environment=result.environment,
                metadata=result.metadata,
                reconnect_metadata={
                    **result.reconnect_metadata,
                    "resource_name": "different-resource",
                },
                release=result.release,
                release_timeout_s=result.release_timeout_s,
            )

    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        created = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert created.error is None
        published = await store.load_checkpoint(session.id)
        assert published is not None

        reconnect = await _resolve(
            store,
            session,
            DriftingReconnectFactory(provider),
            operation=EnvironmentFactoryOperation.RECONNECT,
        )

        assert isinstance(reconnect.error, RuntimeError)
        assert "changed its immutable reconnect identity" in str(reconnect.error)
        assert await store.load_checkpoint(session.id) == published
        assert len(provider.resources) == 1
        assert provider.reap_calls == []

    asyncio.run(run())


def test_fork_replaces_only_its_copied_parent_allocation_receipt() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _FakeRemoteProvider()
        parent_id = "allocation-parent"
        child_id = "allocation-child"
        parent = await _create_session(store, session_id=parent_id)
        parent_result = await _resolve(
            store,
            parent,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert parent_result.error is None
        parent_checkpoint = await store.load_checkpoint(parent_id)
        assert parent_checkpoint is not None
        parent_receipt = parent_checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
            _ENVIRONMENT_NAME
        ]

        child = await _create_session(
            store,
            session_id=child_id,
            parent_session_id=parent_id,
        )
        await store.checkpoint(child_id, parent_checkpoint)
        child_result = await _resolve(
            store,
            child,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.RECONNECT,
        )
        assert child_result.error is None
        assert len(provider.create_calls) == 2

        child_checkpoint = await store.load_checkpoint(child_id)
        assert child_checkpoint is not None
        child_receipt = child_checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
            _ENVIRONMENT_NAME
        ]
        assert child_receipt["intent"]["session_id"] == child_id
        assert child_receipt["intent"]["allocation_id"] != parent_receipt["intent"]["allocation_id"]
        assert (
            child_checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_OWNER_CHECKPOINT_KEY][_ENVIRONMENT_NAME]
            == child_id
        )
        assert (
            parent_checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
                _ENVIRONMENT_NAME
            ]
            == parent_receipt
        )

    asyncio.run(run())


def test_fork_refuses_ordinary_create_with_an_inherited_remote_receipt() -> None:
    class _OrdinaryFactory(EnvironmentFactory):
        def __init__(self) -> None:
            self.create_calls = 0

        async def create(
            self,
            request: EnvironmentFactoryRequest,
        ) -> EnvironmentFactoryResult:
            self.create_calls += 1
            return EnvironmentFactoryResult(
                environment=Environment(EnvironmentSpec(name=request.environment_name)),
                reconnect_metadata={"resource_name": "ordinary-replacement"},
            )

    async def run() -> None:
        store = InMemorySessionStore()
        provider = _FakeRemoteProvider()
        parent_id = "downgrade-parent"
        child_id = "downgrade-child"
        parent = await _create_session(store, session_id=parent_id)
        parent_result = await _resolve(
            store,
            parent,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert parent_result.error is None
        parent_checkpoint = await store.load_checkpoint(parent_id)
        assert parent_checkpoint is not None

        child = await _create_session(
            store,
            session_id=child_id,
            parent_session_id=parent_id,
        )
        await store.checkpoint(child_id, parent_checkpoint)
        factory = _OrdinaryFactory()
        result = await _resolve(
            store,
            child,
            factory,
            operation=EnvironmentFactoryOperation.RECONNECT,
        )

        assert isinstance(result.error, RuntimeError)
        assert "published remote allocation receipt" in str(result.error)
        assert factory.create_calls == 0
        assert await store.load_checkpoint(child_id) == parent_checkpoint
        assert len(provider.resources) == 1

    asyncio.run(run())


def test_fork_does_not_adopt_or_reap_its_copied_parent_pending_allocation() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        provider = _FakeRemoteProvider()
        parent_id = "pending-allocation-parent"
        child_id = "pending-allocation-child"
        parent = await _create_session(store, session_id=parent_id)
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                parent,
                _FakeRemoteFactory(provider, crash_at="after_remote_create"),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        parent_checkpoint = await store.load_checkpoint(parent_id)
        assert parent_checkpoint is not None
        parent_pending = _pending_record(parent_checkpoint)
        assert parent_pending is not None
        parent_allocation_id = parent_pending["intent"]["allocation_id"]

        child = await _create_session(
            store,
            session_id=child_id,
            parent_session_id=parent_id,
        )
        await store.checkpoint(child_id, parent_checkpoint)
        child_result = await _resolve(
            store,
            child,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.RECONNECT,
        )
        assert child_result.error is None
        assert len(provider.create_calls) == 2
        assert provider.reap_calls == []
        assert _resource_name(parent_allocation_id) in provider.resources

        unchanged_parent = await store.load_checkpoint(parent_id)
        assert unchanged_parent is not None
        assert _pending_record(unchanged_parent) == parent_pending
        child_checkpoint = await store.load_checkpoint(child_id)
        assert child_checkpoint is not None
        assert _pending_record(child_checkpoint) is None
        child_receipt = child_checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
            _ENVIRONMENT_NAME
        ]
        assert child_receipt["intent"]["session_id"] == child_id
        assert child_receipt["intent"]["allocation_id"] != parent_allocation_id

    asyncio.run(run())


@pytest.mark.parametrize("failed_transform", [1, 2, 3, 4])
def test_remote_allocation_reconstructs_lost_checkpoint_acknowledgements(
    failed_transform: int,
) -> None:
    async def run() -> None:
        store = _AcknowledgementLossStore(failed_transform)
        session = await _create_session(store)
        store.arm()
        provider = _FakeRemoteProvider()
        resolution = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert resolution.error is None
        assert len(provider.create_calls) == 1
        checkpoint = await store.load_checkpoint(_SESSION_ID) or {}
        assert _pending_record(checkpoint) is None
        assert (
            _ENVIRONMENT_NAME in checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY]
        )

    asyncio.run(run())


def test_ambiguous_publication_readback_preserves_exact_allocation_for_reconnect() -> None:
    async def run() -> None:
        store = _PublicationReadbackLossStore()
        session = await _create_session(store)
        store.arm()
        provider = _FakeRemoteProvider()

        first = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert first.error is not None
        assert len(provider.create_calls) == 1
        assert provider.reap_calls == []

        checkpoint = await store.load_checkpoint(_SESSION_ID)
        assert checkpoint is not None
        assert _pending_record(checkpoint) is None
        assert (
            _ENVIRONMENT_NAME in checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY]
        )

        second = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.RECONNECT,
        )
        assert second.error is None
        assert len(provider.create_calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize("failed_transform", [1, 3, 4])
def test_checkpoint_reconciliation_never_swallows_fatal_control_signals(
    failed_transform: int,
) -> None:
    async def run() -> None:
        store = _FatalSignalAfterTransformStore(failed_transform)
        session = await _create_session(store)
        store.arm()
        provider = _FakeRemoteProvider()
        with pytest.raises(SystemExit, match="fatal signal"):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        checkpoint = await store.load_checkpoint(_SESSION_ID)
        assert checkpoint is not None
        if failed_transform == 1:
            pending = _pending_record(checkpoint)
            assert pending is not None
            assert pending["state"] == "prepared"
            retry_operation = EnvironmentFactoryOperation.CREATE
            assert provider.create_calls == []
        elif failed_transform == 3:
            pending = _pending_record(checkpoint)
            assert pending is not None
            assert pending["state"] == "acknowledged"
            retry_operation = EnvironmentFactoryOperation.CREATE
            assert len(provider.create_calls) == 1
            assert provider.reap_calls == []
        else:
            assert _pending_record(checkpoint) is None
            assert (
                _ENVIRONMENT_NAME
                in checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY]
            )
            retry_operation = EnvironmentFactoryOperation.RECONNECT
            assert len(provider.create_calls) == 1
            assert provider.reap_calls == []

        recovered = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=retry_operation,
        )
        assert recovered.error is None
        assert len(provider.create_calls) == 1

    asyncio.run(run())


def test_remote_allocation_fences_concurrent_recovery_workers() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at="after_intent"),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        next_index = 0

        class ContendedFactory(_FakeRemoteFactory):
            async def create_recoverable(
                self,
                request: EnvironmentFactoryRequest,
                allocation: EnvironmentAllocationContext,
            ) -> EnvironmentFactoryResult:
                nonlocal next_index
                index = next_index
                next_index += 1
                self._recovery_entered = entered[index]
                self._recovery_release = release
                return await super().create_recoverable(request, allocation)

        first = asyncio.create_task(
            _resolve(
                store,
                session,
                ContendedFactory(provider),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        )
        second = asyncio.create_task(
            _resolve(
                store,
                session,
                ContendedFactory(provider),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        )
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered)),
            timeout=1,
        )
        release.set()
        results = await asyncio.gather(first, second)
        assert sum(result.error is None for result in results) == 1
        assert len(provider.create_calls) == 1

    asyncio.run(run())


def test_duplicate_dispatched_recovery_acknowledgement_reconstructs_publication() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at="after_remote_create"),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        entered = [asyncio.Event(), asyncio.Event()]
        release = asyncio.Event()
        next_index = 0

        class ContendedFactory(_FakeRemoteFactory):
            async def create_recoverable(
                self,
                request: EnvironmentFactoryRequest,
                allocation: EnvironmentAllocationContext,
            ) -> EnvironmentFactoryResult:
                nonlocal next_index
                index = next_index
                next_index += 1
                self._recovery_entered = entered[index]
                self._recovery_release = release
                return await super().create_recoverable(request, allocation)

        recoveries = [
            asyncio.create_task(
                _resolve(
                    store,
                    session,
                    ContendedFactory(provider),
                    operation=EnvironmentFactoryOperation.CREATE,
                )
            )
            for _ in range(2)
        ]
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered)),
            timeout=1,
        )
        release.set()
        results = await asyncio.gather(*recoveries)
        assert all(result.error is None for result in results)
        assert len(provider.create_calls) == 1
        checkpoint = await store.load_checkpoint(_SESSION_ID)
        assert checkpoint is not None
        assert _pending_record(checkpoint) is None
        assert (
            _ENVIRONMENT_NAME in checkpoint[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY]
        )

    asyncio.run(run())


def test_late_duplicate_acknowledgement_reconstructs_already_published_receipt() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at="after_remote_create"),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None

        def checkpoint_transform(checkpoint: dict[str, Any]) -> CheckpointTransform:
            def transform(
                _session: Session,
                _current: dict[str, Any] | None,
            ) -> dict[str, Any]:
                return checkpoint

            return transform

        coordinator = EnvironmentAllocationCoordinator(
            session_store=store,
            checkpoint_transform=checkpoint_transform,
            secret_redactor=SecretRedactor(),
        )
        record = coordinator.record_from_checkpoint(
            checkpoint,
            environment_name=_ENVIRONMENT_NAME,
        )
        assert record is not None
        scope = EnvironmentAllocationScope(
            provider=_PROVIDER,
            adapter_generation=_ADAPTER_GENERATION,
        )
        contexts = [
            coordinator.context(
                session_id=session.id,
                parent_session_id=None,
                environment_name=_ENVIRONMENT_NAME,
                scope=scope,
                existing=record,
            )
            for _ in range(2)
        ]
        resource = provider.lookup(
            _resource_name(record.intent.allocation_id),
            intent=contexts[0],
        )
        assert resource is not None
        reconnect_metadata = resource.reconnect_metadata()

        await contexts[0].acknowledge(reconnect_metadata)
        await contexts[0].publish()
        await contexts[1].acknowledge(reconnect_metadata)
        await contexts[1].publish()

        published = await store.load_checkpoint(session.id)
        assert published is not None
        assert _pending_record(published) is None
        receipt = published[ENVIRONMENT_FACTORY_ALLOCATION_RECEIPTS_CHECKPOINT_KEY][
            _ENVIRONMENT_NAME
        ]
        assert receipt["reconnect_metadata"] == reconnect_metadata

    asyncio.run(run())


def test_recovery_refuses_foreign_resource_at_the_exact_provider_identity() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at="after_remote_create"),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        resource_name, resource = next(iter(provider.resources.items()))
        provider.resources[resource_name] = _FakeResource(
            resource_name=resource.resource_name,
            allocation_id=resource.allocation_id,
            session_id="different-session",
            environment_name=resource.environment_name,
            adapter_generation=resource.adapter_generation,
        )
        resolution = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert resolution.error is not None
        assert len(provider.create_calls) == 1
        assert provider.reap_calls == []
        assert provider.resources[resource_name].session_id == "different-session"

    asyncio.run(run())


def test_missing_dispatched_allocation_retries_same_logical_provider_operation() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at="after_remote_create"),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        provider.resources.clear()

        recovered = await _resolve(
            store,
            session,
            _FakeRemoteFactory(provider),
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert recovered.error is None
        checkpoint = await store.load_checkpoint(_SESSION_ID)
        assert checkpoint is not None
        assert _pending_record(checkpoint) is None
        assert len(provider.create_calls) == 2
        assert provider.create_calls[0] == provider.create_calls[1]

    asyncio.run(run())


@pytest.mark.parametrize(
    "provider_metadata",
    [
        pytest.param({"api_key": "credential"}, id="secret"),
        pytest.param({"resource_name": float("nan")}, id="nonportable"),
        pytest.param(
            {"resource_name": "x" * (16 * 1024 + 1)},
            id="oversized",
        ),
        pytest.param(
            cast("Any", [("resource_name", "remote-1")]),
            id="non-object",
        ),
    ],
)
def test_allocation_intent_rejects_secret_nonportable_or_oversized_metadata(
    provider_metadata: Mapping[str, Any],
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        resolution = await _resolve(
            store,
            session,
            _FakeRemoteFactory(
                provider,
                provider_metadata=provider_metadata,
            ),
            operation=EnvironmentFactoryOperation.CREATE,
            secret_redactor=SecretRedactor("credential"),
        )
        assert resolution.error is not None
        assert provider.create_calls == []
        checkpoint = await store.load_checkpoint(_SESSION_ID) or {}
        assert _pending_record(checkpoint) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    ("scope_provider", "scope_adapter_generation"),
    [
        pytest.param(
            "credential",
            _ADAPTER_GENERATION,
            id="provider",
        ),
        pytest.param(
            _PROVIDER,
            "credential",
            id="adapter-generation",
        ),
    ],
)
def test_allocation_scope_rejects_secrets_before_persistence(
    scope_provider: str,
    scope_adapter_generation: str,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        resolution = await _resolve(
            store,
            session,
            _FakeRemoteFactory(
                provider,
                scope_provider=scope_provider,
                scope_adapter_generation=scope_adapter_generation,
            ),
            operation=EnvironmentFactoryOperation.CREATE,
            secret_redactor=SecretRedactor("credential"),
        )

        assert isinstance(resolution.error, ValueError)
        assert "credential" not in str(resolution.error)
        assert provider.create_calls == []
        checkpoint = await store.load_checkpoint(session.id) or {}
        assert _pending_record(checkpoint) is None

    asyncio.run(run())


@pytest.mark.parametrize(
    ("scope_provider", "scope_adapter_generation"),
    [
        pytest.param(
            "credential",
            _ADAPTER_GENERATION,
            id="provider",
        ),
        pytest.param(
            _PROVIDER,
            "credential",
            id="adapter-generation",
        ),
    ],
)
def test_allocation_scope_rejects_secrets_on_durable_readback(
    scope_provider: str,
    scope_adapter_generation: str,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(
                    provider,
                    crash_at="after_intent",
                    scope_provider=scope_provider,
                    scope_adapter_generation=scope_adapter_generation,
                ),
                operation=EnvironmentFactoryOperation.CREATE,
            )
        persisted = await store.load_checkpoint(session.id)
        assert persisted is not None
        assert _pending_record(persisted) is not None

        resolution = await _resolve(
            store,
            session,
            _FakeRemoteFactory(
                provider,
                scope_provider=scope_provider,
                scope_adapter_generation=scope_adapter_generation,
            ),
            operation=EnvironmentFactoryOperation.CREATE,
            secret_redactor=SecretRedactor("credential"),
        )

        assert isinstance(resolution.error, ValueError)
        assert "credential" not in str(resolution.error)
        assert provider.create_calls == []
        assert await store.load_checkpoint(session.id) == persisted

    asyncio.run(run())


@pytest.mark.parametrize(
    ("scope_provider", "scope_adapter_generation"),
    [
        pytest.param(
            "credential",
            _ADAPTER_GENERATION,
            id="provider",
        ),
        pytest.param(
            _PROVIDER,
            "credential",
            id="adapter-generation",
        ),
    ],
)
def test_published_allocation_scope_rejects_secrets_on_durable_readback(
    scope_provider: str,
    scope_adapter_generation: str,
) -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        factory = _FakeRemoteFactory(
            provider,
            scope_provider=scope_provider,
            scope_adapter_generation=scope_adapter_generation,
        )
        created = await _resolve(
            store,
            session,
            factory,
            operation=EnvironmentFactoryOperation.CREATE,
        )
        assert created.error is None
        published = await store.load_checkpoint(session.id)
        assert published is not None
        assert _pending_record(published) is None

        resolution = await _resolve(
            store,
            session,
            factory,
            operation=EnvironmentFactoryOperation.RECONNECT,
            secret_redactor=SecretRedactor("credential"),
        )

        assert isinstance(resolution.error, ValueError)
        assert "credential" not in str(resolution.error)
        assert len(provider.create_calls) == 1
        assert await store.load_checkpoint(session.id) == published

    asyncio.run(run())


def test_allocation_acknowledgement_rejects_a_non_object_without_advancing_state() -> None:
    async def run() -> None:
        store = InMemorySessionStore()
        session = await _create_session(store)
        provider = _FakeRemoteProvider()
        with pytest.raises(_SimulatedProcessDeath):
            await _resolve(
                store,
                session,
                _FakeRemoteFactory(provider, crash_at="after_remote_create"),
                operation=EnvironmentFactoryOperation.CREATE,
            )

        checkpoint = await store.load_checkpoint(session.id)
        assert checkpoint is not None
        coordinator = EnvironmentAllocationCoordinator(
            session_store=store,
            checkpoint_transform=lambda candidate: lambda _session, _current: candidate,
            secret_redactor=SecretRedactor(),
        )
        record = coordinator.record_from_checkpoint(
            checkpoint,
            environment_name=_ENVIRONMENT_NAME,
        )
        assert record is not None
        allocation = coordinator.context(
            session_id=session.id,
            parent_session_id=None,
            environment_name=_ENVIRONMENT_NAME,
            scope=record.intent.scope,
            existing=record,
        )

        with pytest.raises(TypeError, match="reconnect_metadata must be a mapping"):
            await allocation.acknowledge(cast("Any", [("resource_id", "remote-1")]))

        unchanged = await store.load_checkpoint(session.id)
        assert unchanged == checkpoint
        assert allocation.state is EnvironmentAllocationState.DISPATCHED

    asyncio.run(run())


@pytest.mark.parametrize(
    "parser,payload",
    [
        pytest.param(
            EnvironmentAllocationIntent.from_payload,
            EnvironmentAllocationIntent(
                allocation_id="ealloc_0123456789abcdef0123456789abcdef",
                provider=_PROVIDER,
                adapter_generation=_ADAPTER_GENERATION,
                session_id=_SESSION_ID,
                environment_name=_ENVIRONMENT_NAME,
                requested_operation=EnvironmentFactoryOperation.CREATE,
            ).to_payload(),
            id="intent",
        ),
        pytest.param(
            EnvironmentAllocationRecord.from_payload,
            EnvironmentAllocationRecord(
                intent=EnvironmentAllocationIntent(
                    allocation_id="ealloc_0123456789abcdef0123456789abcdef",
                    provider=_PROVIDER,
                    adapter_generation=_ADAPTER_GENERATION,
                    session_id=_SESSION_ID,
                    environment_name=_ENVIRONMENT_NAME,
                    requested_operation=EnvironmentFactoryOperation.CREATE,
                ),
                state=EnvironmentAllocationState.PREPARED,
            ).to_payload(),
            id="record",
        ),
        pytest.param(
            EnvironmentAllocationReceipt.from_payload,
            EnvironmentAllocationReceipt(
                intent=EnvironmentAllocationIntent(
                    allocation_id="ealloc_0123456789abcdef0123456789abcdef",
                    provider=_PROVIDER,
                    adapter_generation=_ADAPTER_GENERATION,
                    session_id=_SESSION_ID,
                    environment_name=_ENVIRONMENT_NAME,
                    requested_operation=EnvironmentFactoryOperation.CREATE,
                ),
                reconnect_metadata={"resource_id": "remote-1"},
            ).to_payload(),
            id="receipt",
        ),
    ],
)
def test_allocation_payload_parsers_reject_a_non_object_json_container(
    parser: Any,
    payload: dict[str, Any],
) -> None:
    with pytest.raises(TypeError, match="payload must be a mapping"):
        parser(list(payload.items()))
