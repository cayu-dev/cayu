"""Application-owned authorization and request-local durable execution bounds."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from cayu._resource_access_binding import ResourceExecutionBinding
from cayu._resource_access_errors import ResourceAccessDenied

if TYPE_CHECKING:
    from cayu.sessions.access import SessionAccessScope, _SessionAccessBounds
    from cayu.sessions.base import SessionStore


def encode_scope(scope) -> str:
    from cayu.sessions.access import SessionAccessScope

    if type(scope) is not SessionAccessScope:
        raise TypeError("Policy must resolve a SessionAccessScope.")
    return json.dumps(asdict(scope), sort_keys=True, separators=(",", ":"))


def decode_scope(value):
    from cayu.sessions.access import (
        _ACTIONS,
        SessionAccessRule,
        SessionAccessScope,
        SessionAccessSelector,
    )

    if type(value) is not dict or set(value) != {*_ACTIONS, "protected_label_keys"}:
        raise ValueError("Invalid durable resource scope.")
    return SessionAccessScope(
        **{
            action: tuple(
                SessionAccessRule(
                    selectors=tuple(
                        SessionAccessSelector(**selector) for selector in rule["selectors"]
                    ),
                    allow_all=rule["allow_all"],
                )
                for rule in value[action]
            )
            for action in _ACTIONS
        },
        protected_label_keys=tuple(value["protected_label_keys"]),
    )


@dataclass(frozen=True, slots=True)
class ResourceAccessGrant:
    """Separate application grants for each resource family; omitted families deny."""

    sessions: SessionAccessScope | None = None
    tasks: SessionAccessScope | None = None
    artifacts: SessionAccessScope | None = None
    knowledge: SessionAccessScope | None = None

    def __post_init__(self):
        from cayu.sessions.access import SessionAccessScope

        for name in ("sessions", "tasks", "artifacts", "knowledge"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, SessionAccessScope())
            elif type(value) is not SessionAccessScope:
                raise TypeError("Resource grants require bounded action scopes.")


def decode_grant(value):
    if type(value) is dict and set(value) == {"sessions", "tasks", "artifacts", "knowledge"}:
        return ResourceAccessGrant(**{kind: decode_scope(scope) for kind, scope in value.items()})
    scope = decode_scope(value)
    return ResourceAccessGrant(sessions=scope, tasks=scope, artifacts=scope, knowledge=scope)


def encode_grant(grant):
    from cayu.sessions.access import SessionAccessScope

    if type(grant) is SessionAccessScope:
        return encode_scope(grant)
    if type(grant) is not ResourceAccessGrant:
        raise TypeError(
            "Policy must return ResourceAccessGrant or an explicit universal action scope."
        )
    return json.dumps(
        {
            kind: json.loads(encode_scope(getattr(grant, kind)))
            for kind in ("sessions", "tasks", "artifacts", "knowledge")
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def admitted_scope(binding, kind="sessions"):
    return getattr(decode_grant(json.loads(binding.admitted_json)), kind)


@dataclass(frozen=True, slots=True)
class ResourceAccessDecision:
    """A versioned, optionally expiring decision from the trusted policy service."""

    grant: ResourceAccessGrant | SessionAccessScope
    revision: int
    valid_until: datetime | None = None

    def __post_init__(self):
        encode_grant(self.grant)
        if type(self.revision) is not int or not 0 <= self.revision <= 9007199254740991:
            raise ValueError("Policy revisions must be nonnegative portable integers.")
        if self.valid_until is not None and self.valid_until.utcoffset() is None:
            raise ValueError("Decision expiry requires an aware timestamp.")


def current_decision(value, *, minimum_revision=0):
    decision = value if type(value) is ResourceAccessDecision else ResourceAccessDecision(value, 0)
    if decision.revision < minimum_revision or (
        decision.valid_until is not None and datetime.now(UTC) >= decision.valid_until
    ):
        raise ResourceAccessDenied()
    return decision


class ResourceAccessPolicy(ABC):
    """Trusted host authority. Resolve on every boundary; no Runtime policy cache."""

    @property
    @abstractmethod
    def authority(self) -> str: ...

    @abstractmethod
    async def resolve(
        self, subject: str
    ) -> ResourceAccessDecision | ResourceAccessGrant | SessionAccessScope: ...


@dataclass(frozen=True, slots=True)
class _ExecutionAccess:
    binding: ResourceExecutionBinding
    policy: ResourceAccessPolicy
    labels: tuple[tuple[str, str], ...]
    store: SessionStore | None = None
    session_id: str | None = None
    creation_bounds: _SessionAccessBounds | None = None
    kind: str = "sessions"


_active: ContextVar[_ExecutionAccess | None] = ContextVar("cayu_resource_execution", default=None)


def current_execution_labels():
    active = _active.get()
    return {} if active is None else dict(active.labels)


def current_binding() -> ResourceExecutionBinding | None:
    active = _active.get()
    return None if active is None else active.binding


async def effective_bounds(binding, policy, kind="sessions"):
    from cayu.sessions.access import SessionAccessScope, _SessionAccessBounds

    if not isinstance(policy, ResourceAccessPolicy) or binding.authority != policy.authority:
        raise ResourceAccessDenied()
    current = current_decision(
        await policy.resolve(binding.subject), minimum_revision=binding.policy_revision
    ).grant
    if type(current) is ResourceAccessGrant:
        current = getattr(current, kind)
    elif type(current) is not SessionAccessScope:
        raise ResourceAccessDenied()
    return _SessionAccessBounds(admitted_scope(binding, kind), current, binding)


async def require_dispatch() -> None:
    active = _active.get()
    if active is None:
        return
    bounds = await effective_bounds(active.binding, active.policy, active.kind)
    labels = dict(active.labels)
    if active.store is not None:
        if active.session_id is None:
            raise ResourceAccessDenied()
        token = _model_data_access.set(False)
        try:
            session = await active.store.load(active.session_id)
        finally:
            _model_data_access.reset(token)
        if session is None or session.invocation.resource_access != active.binding:
            raise ResourceAccessDenied()
        labels = session.labels
    if not bounds.matches(labels) or not bounds.matches(labels, "execute"):
        raise ResourceAccessDenied()


@asynccontextmanager
async def execution_access(
    binding, policy, labels, *, store=None, session_id=None, kind="sessions", action="execute"
):
    """Reconstruct only from stored authority or a trusted host admission."""
    if store is not None and type(store).__dict__.get("session_access_version") != 1:
        raise NotImplementedError("Store cannot enforce resource execution access.")
    outer = _active.get()
    if binding is None:
        if outer is None:
            yield
            return
        binding = outer.binding
    elif outer is not None and binding != outer.binding:
        # A child cannot switch to another subject or admitted maximum.
        raise ResourceAccessDenied()
    bounds = await effective_bounds(binding, policy, kind)
    if not bounds.matches(labels) or not bounds.matches(labels, action):
        raise ResourceAccessDenied()
    token = _active.set(
        _ExecutionAccess(binding, policy, tuple(labels.items()), store, session_id, bounds, kind)
    )
    try:
        yield
    finally:
        _active.reset(token)


async def guard_stream(stream, *, binding, policy, labels, store=None, session_id=None):
    """Bind only while driving the producer, never across a caller-facing yield."""
    iterator = aiter(stream)
    try:
        while True:
            async with execution_access(
                binding, policy, labels, store=store, session_id=session_id
            ):
                data_token = _model_data_access.set(False)
                try:
                    try:
                        event = await anext(iterator)
                    except StopAsyncIteration:
                        return
                    await require_dispatch()
                finally:
                    _model_data_access.reset(data_token)
            yield event
    finally:
        # Closing may settle work already dispatched. It must not require a new
        # grant just to release leases, but any new dispatch still revalidates.
        token = _active.set(
            _ExecutionAccess(binding, policy, tuple(labels.items()), store, session_id)
        )
        data_token = _model_data_access.set(False)
        try:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        finally:
            _model_data_access.reset(data_token)
            _active.reset(token)


def current_creation_bounds():
    active = _active.get()
    if active is not None and active.creation_bounds is None:
        raise ResourceAccessDenied()
    return None if active is None else active.creation_bounds


class ScopedCayuAccess:
    """Application-admitted access; construct through ``await app.access(subject)``.

    The subject must already be authenticated by the host. HTTP bodies, model
    arguments and serialized bindings are never inputs to this constructor.
    """

    def __init__(self, app, binding):
        self.__app = app
        self.__binding = binding

    async def _current(self, kind="sessions"):
        return (
            await effective_bounds(self.__binding, self.__app.resource_access_policy, kind)
        ).current

    @property
    def sessions(self):
        from cayu.sessions.access import ScopedSessionAccess

        return ScopedSessionAccess(
            self.__app.session_store,
            admitted=admitted_scope(self.__binding),
            resolve=self._current,
            execution_binding=self.__binding,
        )

    @property
    def tasks(self):
        from cayu.tasks.access import ScopedTaskAccess

        return ScopedTaskAccess(
            self.__app.task_store,
            admitted=admitted_scope(self.__binding, "tasks"),
            resolve=lambda: self._current("tasks"),
            execution_binding=self.__binding,
            policy=self.__app.resource_access_policy,
        )

    def knowledge(self, *, access_scope=None):
        from cayu.knowledge.access import ScopedKnowledgeAccess

        return ScopedKnowledgeAccess(
            self.__app.knowledge_store,
            binding=self.__binding,
            policy=self.__app.resource_access_policy,
            access_scope=access_scope,
        )

    def artifacts(self, store, *, environment_name: str):
        from cayu.artifacts.access import ScopedArtifactAccess

        return ScopedArtifactAccess(
            store,
            admitted=admitted_scope(self.__binding, "artifacts"),
            resolve=lambda: self._current("artifacts"),
            environment_name=environment_name,
        )

    async def run(self, request):
        from contextlib import aclosing
        from uuid import uuid4

        from cayu.sessions.base import RunRequest

        if type(request) is not RunRequest:
            raise TypeError("Scoped runs require RunRequest.")
        request = request.model_copy(deep=True)
        if request.session_id is None:
            request = request.model_copy(update={"session_id": str(uuid4())})
        async with aclosing(
            guard_stream(
                self.__app.run(request),
                binding=self.__binding,
                policy=self.__app.resource_access_policy,
                labels=request.labels,
            )
        ) as stream:
            async for event in stream:
                yield event

    async def resume(self, request):
        from contextlib import aclosing

        from cayu.sessions.base import ResumeRequest

        if type(request) is not ResumeRequest:
            raise TypeError("Scoped resume requires ResumeRequest.")
        private_id = await self.__app._resolve_public_session_id(request.session_id)
        request = request.model_copy(update={"session_id": private_id})
        session = await self.sessions.load(private_id)
        stored_binding = session.invocation.resource_access
        if stored_binding is None or (stored_binding.authority, stored_binding.subject) != (
            self.__binding.authority,
            self.__binding.subject,
        ):
            raise ResourceAccessDenied()
        bounds = await effective_bounds(self.__binding, self.__app.resource_access_policy)
        if not bounds.matches(session.labels, "execute"):
            raise ResourceAccessDenied()
        async with aclosing(
            guard_stream(
                self.__app.resume(request),
                binding=stored_binding,
                policy=self.__app.resource_access_policy,
                labels=session.labels,
            )
        ) as stream:
            async for event in stream:
                yield event

    async def recover(self, request):
        from cayu.sessions.recovery import IncompleteSessionRecoveryRequest

        if type(request) is not IncompleteSessionRecoveryRequest:
            raise TypeError("Scoped recovery requires IncompleteSessionRecoveryRequest.")
        private_id = await self.__app._resolve_public_session_id(request.session_id)
        session = await self.sessions.load(private_id)
        binding = session.invocation.resource_access
        if binding is None or (binding.authority, binding.subject) != (
            self.__binding.authority,
            self.__binding.subject,
        ):
            raise ResourceAccessDenied()
        bounds = await effective_bounds(self.__binding, self.__app.resource_access_policy)
        bounds.require_action(session, "execute")
        async with execution_access(
            binding,
            self.__app.resource_access_policy,
            session.labels,
            store=self.__app.session_store,
            session_id=private_id,
        ):
            result = await self.__app.recover_incomplete_session(
                request.model_copy(update={"session_id": private_id})
            )
            await require_dispatch()
            return result

    async def enqueue_session_message(self, request, *, context):
        return await self.sessions._query(
            lambda: self.__app.enqueue_session_message(request, context=context)
        )

    async def inspect_session_messages(self, query, *, context):
        return await self.sessions._query(
            lambda: self.__app.inspect_session_messages(query, context=context)
        )

    async def apply_session_message_action(self, request, *, context):
        return await self.sessions._query(
            lambda: self.__app.apply_session_message_action(request, context=context)
        )

    async def snapshot_session_message_source(self, session_id, *, context):
        return await self.sessions._query(
            lambda: self.__app.snapshot_session_message_source(session_id, context=context)
        )

    async def append_peer_content(self, request, *, context):
        return await self.sessions._query(
            lambda: self.__app.append_peer_content(request, context=context)
        )

    async def revalidate_delivery(self, event):
        private_id = await self.__app._resolve_public_session_id(event.session_id)
        session = await self.sessions.load(private_id)
        bounds = await effective_bounds(self.__binding, self.__app.resource_access_policy)
        bounds.require_action(session, "execute")
        binding = session.invocation.resource_access
        if binding is None:
            raise ResourceAccessDenied()
        durable = await effective_bounds(binding, self.__app.resource_access_policy)
        durable.require_action(session, "execute")

    async def fork(self, request):
        from contextlib import aclosing

        from cayu.sessions.base import ForkSessionRequest

        if type(request) is not ForkSessionRequest:
            raise TypeError("Scoped forks require ForkSessionRequest.")
        private_id = await self.__app._resolve_public_session_id(request.source_session_id)
        request = request.model_copy(update={"source_session_id": private_id})
        source = await self.sessions.load(private_id)
        binding = source.invocation.resource_access
        if binding is None or (binding.authority, binding.subject) != (
            self.__binding.authority,
            self.__binding.subject,
        ):
            raise ResourceAccessDenied()
        bounds = await effective_bounds(self.__binding, self.__app.resource_access_policy)
        if not bounds.matches(source.labels, "create"):
            raise ResourceAccessDenied()
        async with aclosing(
            guard_stream(
                self.__app.fork_session(request),
                binding=binding,
                policy=self.__app.resource_access_policy,
                labels=source.labels,
            )
        ) as stream:
            async for event in stream:
                yield event


async def admit_access(app, subject):
    if type(app.session_store).__dict__.get("session_access_version") != 1:
        raise NotImplementedError("Store cannot enforce scoped application access.")
    if (
        app.knowledge_store is not None
        and type(app.knowledge_store).__dict__.get("resource_knowledge_access_version") != 1
    ):
        raise NotImplementedError("Knowledge store cannot enforce scoped application access.")
    policy = app.resource_access_policy
    if not isinstance(policy, ResourceAccessPolicy):
        raise ResourceAccessDenied()
    decision = current_decision(await policy.resolve(subject))
    scope = decision.grant
    binding = ResourceExecutionBinding(
        authority=policy.authority,
        subject=subject,
        admitted_json=encode_grant(scope),
        policy_revision=decision.revision,
    )
    return ScopedCayuAccess(app, binding)


_model_data_access: ContextVar[bool] = ContextVar("resource_model_data_access", default=False)


@asynccontextmanager
async def model_data_access():
    token = _model_data_access.set(True)
    try:
        yield
    finally:
        _model_data_access.reset(token)


async def current_data_bounds(kind="sessions"):
    execution = _active.get()
    if execution is None or not _model_data_access.get():
        return None
    await require_dispatch()
    return await effective_bounds(execution.binding, execution.policy, kind)


@asynccontextmanager
async def recovery_access(session, policy, store):
    """Retain authority during settlement; new work still requires dispatch admission.

    Installing the stored bound does not authorize business work. This lets
    recovery release leases and record already-known outcomes after revocation.
    """
    binding = session.invocation.resource_access
    outer = _active.get()
    if binding is None:
        if outer is not None:
            raise ResourceAccessDenied()
        yield
        return
    if outer is not None and outer.binding != binding:
        raise ResourceAccessDenied()
    token = _active.set(
        _ExecutionAccess(binding, policy, tuple(session.labels.items()), store, session.id)
    )
    data_token = _model_data_access.set(False)
    try:
        yield
    finally:
        _model_data_access.reset(data_token)
        _active.reset(token)


def resource_recovery(operation):
    from functools import wraps

    @wraps(operation)
    async def recover(self, *, session, **kwargs):
        async with recovery_access(session, self._resource_access_policy, self._session_store):
            return await operation(self, session=session, **kwargs)

    return recover


def runtime_stream_entrance(operation):
    """Separate runtime bookkeeping from model-controlled native store calls."""
    from functools import wraps

    @wraps(operation)
    async def stream(self, *args, **kwargs):
        try:
            iterator = aiter(operation(self, *args, **kwargs))
        finally:
            # The owning entrance sanitizes rejected inputs. Do not retain a
            # second raw copy in this wrapper's propagated traceback.
            del args, kwargs
        try:
            advance = anext(iterator)
            while True:
                token = _model_data_access.set(False)
                try:
                    try:
                        event = await advance
                    except StopAsyncIteration:
                        return
                finally:
                    _model_data_access.reset(token)
                try:
                    yield event
                except GeneratorExit:
                    raise
                except BaseException as error:
                    # Forward injected failures to the real stream owner so
                    # its cleanup preserves the original signal and causes.
                    advance = iterator.athrow(error)
                else:
                    advance = anext(iterator)
        finally:
            token = _model_data_access.set(False)
            try:
                await iterator.aclose()
            finally:
                _model_data_access.reset(token)

    return stream


async def restore_session_stream(stream, *, session, policy, store):
    """Restore the exact owner on every internal continuation, including forks.

    Internal settlement can advance after revocation. Dispatch and scoped data
    operations still require fresh permission; public delivery has its own guard.
    """
    iterator = aiter(stream)
    try:
        while True:
            async with recovery_access(session, policy, store):
                try:
                    event = await anext(iterator)
                except StopAsyncIteration:
                    return
            yield event
    finally:
        async with recovery_access(session, policy, store):
            await iterator.aclose()
