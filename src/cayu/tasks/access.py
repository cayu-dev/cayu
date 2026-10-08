"""Task operations with immutable classification and native query enforcement."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass

from cayu._resource_access_errors import ResourceAccessDenied
from cayu._validation import copy_label_map
from cayu.sessions.access import SessionAccessScope, _SessionAccessBounds
from cayu.tasks.base import TaskStore
from cayu.tasks.creation import TaskCreate, copy_task_create
from cayu.tasks.queries import TaskQuery, copy_task_query


@dataclass(frozen=True, slots=True)
class _TaskCreationAccess:
    bounds: _SessionAccessBounds
    labels: tuple[tuple[str, str], ...]


_creation: ContextVar[_TaskCreationAccess | None] = ContextVar("task_access_creation", default=None)


def visible(task, bounds):
    return task is not None and bounds.matches(task.invocation.access_labels)


def require_read(task, bounds):
    if not visible(task, bounds):
        raise ResourceAccessDenied()
    return task


def creation_invocation(invocation, request, parent):
    from cayu.resource_access import _active as execution_active
    from cayu.sessions.invocation import TaskInvocation

    access = _creation.get()
    execution = execution_active.get()
    if access is None and execution is None:
        return invocation
    if access is not None:
        bounds, labels = access.bounds, dict(access.labels)
    else:
        assert execution is not None
        bounds, labels = execution.creation_bounds, dict(execution.labels)
    if bounds is None or not bounds.matches(labels) or not bounds.matches(labels, "create"):
        raise ResourceAccessDenied()
    if execution is not None and invocation.resource_access != execution.binding:
        raise ResourceAccessDenied()
    if parent is not None:
        require_read(parent, bounds)
    if request.session_id is not None or request._runtime_session_binding is not None:
        if execution is None or request.session_id != execution.session_id:
            raise ResourceAccessDenied()
        if invocation.resource_access != execution.binding:
            raise ResourceAccessDenied()
    return TaskInvocation.model_validate({**invocation.model_dump(), "access_labels": labels})


def sql_predicate(bounds, *, postgres: bool):
    """Compile both bounds before ORDER/LIMIT; keys and values are parameters."""
    params = []
    marker = "%s" if postgres else "?"
    table = (
        "jsonb_each_text(COALESCE(invocation->'access_labels', '{}'::jsonb))"
        if postgres
        else "json_each(COALESCE(json_extract(invocation_json, '$.access_labels'), '{}'))"
    )
    intersections = []
    for scope in (bounds.admitted, bounds.current):
        alternatives = []
        for rule in scope.read:
            if rule.allow_all:
                alternatives.append("1=1")
                continue
            requirements = []
            for selector in rule.selectors:
                params.append(selector.key)
                inner = f"access_label.key = {marker}"
                if selector.operator in {"in", "not_in"}:
                    params.extend(selector.values)
                    inner += (
                        f" AND access_label.value IN ({', '.join([marker] * len(selector.values))})"
                    )
                expression = f"EXISTS (SELECT 1 FROM {table} AS access_label WHERE {inner})"
                if selector.operator in {"not_in", "not_exists"}:
                    expression = "NOT " + expression
                requirements.append(expression)
            alternatives.append("(" + " AND ".join(requirements) + ")")
        intersections.append("(" + " OR ".join(alternatives or ["1=0"]) + ")")
    return " AND ".join(intersections), params


class ScopedTaskAccess:
    """Trusted application handle; lifecycle worker authority remains separate."""

    def __init__(
        self,
        store: TaskStore,
        *,
        admitted: SessionAccessScope,
        resolve: Callable[[], Awaitable[SessionAccessScope]],
        execution_binding=None,
        policy=None,
    ) -> None:
        if not isinstance(store, TaskStore) or type(store).__dict__.get("task_access_version") != 1:
            raise NotImplementedError("Store does not support native scoped task access v1.")
        if type(admitted) is not SessionAccessScope or not callable(resolve):
            raise TypeError("Task access requires an explicit scope and trusted resolver.")
        self.__store, self.__admitted, self.__resolve = store, admitted, resolve
        self.__binding, self.__policy = execution_binding, policy

    async def _bounds(self):
        current = await self.__resolve()
        if type(current) is not SessionAccessScope:
            raise TypeError("Task resolver must return SessionAccessScope.")
        return _SessionAccessBounds(self.__admitted, current)

    async def load(self, task_id: str):
        return await self.__store.load_task(task_id, _access_bounds=await self._bounds())

    async def list(self, query: TaskQuery | None = None):
        return await self.__store.list_tasks(
            copy_task_query(query), _access_bounds=await self._bounds()
        )

    async def _collection(self, operation):
        token = _collection_bounds.set(await self._bounds())
        try:
            result = await operation()
            if result is None:
                raise ResourceAccessDenied()
            return result
        except KeyError:
            raise ResourceAccessDenied() from None
        finally:
            _collection_bounds.reset(token)

    async def graph(self, graph_id: str):
        return await self._collection(lambda: self.__store.load_task_graph(graph_id))

    async def group(self, group_id: str):
        return await self._collection(lambda: self.__store.load_task_group(group_id))

    async def graph_events(self, graph_id: str, *, after_sequence: int = 0, limit: int = 100):
        return await self._collection(
            lambda: self.__store.list_task_graph_events(
                graph_id, after_sequence=after_sequence, limit=limit
            )
        )

    async def group_events(self, group_id: str, *, after_sequence: int = 0, limit: int = 100):
        return await self._collection(
            lambda: self.__store.list_task_group_events(
                group_id, after_sequence=after_sequence, limit=limit
            )
        )

    async def _create_collection(self, operation, labels):
        labels = copy_label_map(labels, "collection access labels", allow_reserved=False)
        bounds = await self._bounds()
        if not bounds.matches(labels) or not bounds.matches(labels, "create"):
            raise ResourceAccessDenied()
        token = _creation.set(_TaskCreationAccess(bounds, tuple(labels.items())))
        try:
            if self.__binding is None:
                return await self._collection(operation)
            from cayu.resource_access import execution_access

            async with execution_access(
                self.__binding, self.__policy, labels, kind="tasks", action="create"
            ):
                return await self._collection(operation)
        finally:
            _creation.reset(token)

    async def create_graph(self, request, *, labels: dict[str, str]):
        return await self._create_collection(
            lambda: self.__store.create_task_graph(request), labels
        )

    async def create_group(self, request, *, labels: dict[str, str]):
        return await self._create_collection(
            lambda: self.__store.create_task_group(request), labels
        )

    async def _mutate(self, operation):
        token = _mutation.set(await self._bounds())
        try:
            return await operation()
        finally:
            _mutation.reset(token)

    async def cancel(self, task_id: str):
        return await self._mutate(lambda: self.__store.cancel_task(task_id))

    async def pause(self, task_id: str):
        return await self._mutate(lambda: self.__store.pause_task(task_id))

    async def resume(self, task_id: str):
        return await self._mutate(lambda: self.__store.resume_task(task_id))

    async def create(self, request: TaskCreate, *, labels: dict[str, str]):
        request = copy_task_create(request)
        labels = copy_label_map(labels, "task access labels", allow_reserved=False)
        bounds = await self._bounds()
        if not bounds.matches(labels) or not bounds.matches(labels, "create"):
            raise ResourceAccessDenied()
        token = _creation.set(_TaskCreationAccess(bounds, tuple(labels.items())))
        try:
            if self.__binding is None:
                task = await self.__store.create_task(request)
            else:
                from cayu.resource_access import execution_access

                async with execution_access(
                    self.__binding, self.__policy, labels, kind="tasks", action="create"
                ):
                    task = await self.__store.create_task(request)
            require_read(task, bounds)
            if task.invocation.access_labels != labels:
                raise ResourceAccessDenied()
            return task
        finally:
            _creation.reset(token)


_mutation: ContextVar[_SessionAccessBounds | None] = ContextVar(
    "task_access_mutation", default=None
)


def require_mutation(task):
    bounds = _mutation.get()
    if bounds is None:
        return
    require_read(task, bounds)
    if not bounds.matches(task.invocation.access_labels, "modify"):
        raise ResourceAccessDenied()


def runtime_task_mutation(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.resource_access import current_data_bounds

        bounds = await current_data_bounds("tasks")
        if bounds is None or _mutation.get() is not None:
            return await operation(self, *args, **kwargs)
        token = _mutation.set(bounds)
        try:
            return await operation(self, *args, **kwargs)
        finally:
            _mutation.reset(token)

    return guarded


_collection_bounds: ContextVar[_SessionAccessBounds | None] = ContextVar(
    "task_collection_access", default=None
)


def require_collection(receipt):
    bounds = _collection_bounds.get()
    if bounds is None:
        return
    if receipt is None or receipt.access_invocation is None:
        raise ResourceAccessDenied()
    if not bounds.matches(receipt.access_invocation.access_labels):
        raise ResourceAccessDenied()
    creation = _creation.get()
    if creation is not None and dict(creation.labels) != receipt.access_invocation.access_labels:
        raise ResourceAccessDenied()


def graph_access_invocation(tasks):
    values = list(tasks)
    first = values[0].invocation
    if not first.access_labels and first.resource_access is None:
        return None
    if any(
        task.invocation.access_labels != first.access_labels
        or task.invocation.resource_access != first.resource_access
        for task in values
    ):
        raise ResourceAccessDenied()
    return first


def runtime_task_creation(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.resource_access import _active, effective_bounds, require_dispatch

        execution = _active.get()
        if execution is None or _creation.get() is not None:
            return await operation(self, *args, **kwargs)
        await require_dispatch()
        bounds = await effective_bounds(execution.binding, execution.policy, "tasks")
        labels = dict(execution.labels)
        if not bounds.matches(labels) or not bounds.matches(labels, "create"):
            raise ResourceAccessDenied()
        creation_token = _creation.set(_TaskCreationAccess(bounds, tuple(labels.items())))
        collection_token = _collection_bounds.set(bounds)
        try:
            result = await operation(self, *args, **kwargs)
            invocation = getattr(result, "invocation", None)
            if invocation is not None:
                require_read(result, bounds)
            return result
        finally:
            _creation.reset(creation_token)
            _collection_bounds.reset(collection_token)

    return guarded


def runtime_collection_read(operation):
    from functools import wraps

    @wraps(operation)
    async def guarded(self, *args, **kwargs):
        from cayu.resource_access import current_data_bounds

        bounds = await current_data_bounds("tasks")
        if bounds is None or _collection_bounds.get() is not None:
            return await operation(self, *args, **kwargs)
        token = _collection_bounds.set(bounds)
        try:
            return await operation(self, *args, **kwargs)
        finally:
            _collection_bounds.reset(token)

    return guarded
