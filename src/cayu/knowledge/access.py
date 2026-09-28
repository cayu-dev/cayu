"""Intersection of existing knowledge constraints and resource execution policy."""

from __future__ import annotations

import json
from contextvars import ContextVar
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cayu.sessions.access import _SessionAccessBounds

from cayu._resource_access_errors import ResourceAccessDenied

_bounds: ContextVar[_SessionAccessBounds | None] = ContextVar(
    "knowledge_resource_bounds", default=None
)

_constraints: ContextVar[tuple[str, ...] | None] = ContextVar(
    "knowledge_resource_constraints", default=None
)


def validate_constraints(values):
    from cayu.resource_access import decode_scope, encode_scope

    if len(values) > 4:
        raise ValueError("At most four knowledge resource predicates are supported.")
    return tuple(encode_scope(decode_scope(json.loads(value))) for value in values)


def intersect_scope(scope):
    constraints = _constraints.get()
    if constraints is None:
        from cayu.resource_access import current_binding

        if current_binding() is not None:
            raise ResourceAccessDenied()
        return scope
    # A caller may already have narrower resource predicates. Retain all of
    # them; reject overflow rather than dropping an intersection.
    combined = tuple(dict.fromkeys((*scope.resource_constraints, *constraints)))
    if len(combined) > 4:
        raise ResourceAccessDenied()
    return scope.model_copy(update={"resource_constraints": combined})


def matches(scope, labels):
    from cayu.resource_access import decode_scope

    return all(
        decode_scope(json.loads(value)).matches("read", labels)
        for value in scope.resource_constraints
    )


def predicate_sql(scope, *, postgres, table, correlation):
    from cayu.resource_access import decode_scope

    marker = "%s" if postgres else "?"
    params, intersections = [], []
    for serialized in scope.resource_constraints:
        predicate = decode_scope(json.loads(serialized))
        alternatives = []
        for rule in predicate.read:
            if rule.allow_all:
                alternatives.append("1=1")
                continue
            selectors = []
            for selector in rule.selectors:
                params.append(selector.key)
                inner = f"{correlation} AND resource_label.key = {marker}"
                if selector.operator in {"in", "not_in"}:
                    params.extend(selector.values)
                    inner += f" AND resource_label.value IN ({', '.join([marker] * len(selector.values))})"
                clause = f"EXISTS (SELECT 1 FROM {table} AS resource_label WHERE {inner})"
                if selector.operator in {"not_in", "not_exists"}:
                    clause = "NOT " + clause
                selectors.append(clause)
            alternatives.append("(" + " AND ".join(selectors) + ")")
        intersections.append("(" + " OR ".join(alternatives or ["1=0"]) + ")")
    return " AND ".join(intersections or ["1=1"]), params


def runtime_knowledge_operation(action):
    def decorate(operation):
        @wraps(operation)
        async def guarded(self, *args, **kwargs):
            from cayu.resource_access import (
                _active,
                effective_bounds,
                encode_scope,
                require_dispatch,
            )
            from cayu.sessions.access import SessionAccessScope

            execution = _active.get()
            if execution is None:
                return await operation(self, *args, **kwargs)
            await require_dispatch()
            bounds = await effective_bounds(execution.binding, execution.policy, "knowledge")
            constraints = tuple(
                dict.fromkeys(
                    encode_scope(SessionAccessScope(read=getattr(scope, required_action)))
                    for scope in (bounds.admitted, bounds.current)
                    for required_action in (("read",) if action == "read" else ("read", action))
                )
            )
            bounds_token = _bounds.set(bounds)
            token = _constraints.set(constraints)
            try:
                return await operation(self, *args, **kwargs)
            finally:
                _constraints.reset(token)
                _bounds.reset(bounds_token)

        return guarded

    return decorate


def require_relabel(old_labels, new_labels):
    bounds = _bounds.get()
    if bounds is None:
        return
    protected = bounds.admitted.protected_keys() | bounds.current.protected_keys()
    if any(old_labels.get(key) != new_labels.get(key) for key in protected) and (
        not bounds.matches(old_labels, "relabel") or not bounds.matches(new_labels, "relabel")
    ):
        raise ResourceAccessDenied()


class ScopedKnowledgeAccess:
    """Explicit data operations intersected with a host-selected knowledge scope."""

    def __init__(self, store, *, binding, policy, access_scope=None):
        from cayu.knowledge.scopes import copy_knowledge_access_scope

        if type(store).__dict__.get("resource_knowledge_access_version") != 1:
            raise NotImplementedError("Knowledge store cannot enforce resource access.")
        self.__store = store
        self.__binding = binding
        self.__policy = policy
        self.__scope = None if access_scope is None else copy_knowledge_access_scope(access_scope)

    async def _call(self, action, operation):
        from cayu.resource_access import effective_bounds, encode_scope
        from cayu.sessions.access import SessionAccessScope

        bounds = await effective_bounds(self.__binding, self.__policy, "knowledge")
        constraints = tuple(
            dict.fromkeys(
                encode_scope(SessionAccessScope(read=getattr(scope, required)))
                for scope in (bounds.admitted, bounds.current)
                for required in (("read",) if action == "read" else ("read", action))
            )
        )
        token = _constraints.set(constraints)
        bounds_token = _bounds.set(bounds)
        try:
            return await operation()
        finally:
            _bounds.reset(bounds_token)
            _constraints.reset(token)

    async def get_entry(self, entry_id, *, revision=None, max_bytes=1_048_576):
        return await self._call(
            "read",
            lambda: self.__store.get_entry(
                entry_id, revision=revision, max_bytes=max_bytes, access_scope=self.__scope
            ),
        )

    async def list_entries(self, query):
        return await self._call(
            "read", lambda: self.__store.list_entries(query, access_scope=self.__scope)
        )

    async def search(self, query):
        return await self._call(
            "read", lambda: self.__store.search(query, access_scope=self.__scope)
        )

    async def create_entry(self, entry, chunks=None, *, evidence=None):
        return await self._call(
            "create",
            lambda: self.__store.create_entry(
                entry, chunks, evidence=evidence, access_scope=self.__scope
            ),
        )

    async def append_entry_revision(self, entry, chunks=None, *, expected_revision, evidence=None):
        return await self._call(
            "modify",
            lambda: self.__store.append_entry_revision(
                entry,
                chunks,
                expected_revision=expected_revision,
                evidence=evidence,
                access_scope=self.__scope,
            ),
        )

    async def delete_entry(self, entry_id, *, expected_revision):
        return await self._call(
            "delete",
            lambda: self.__store.delete_entry(
                entry_id, expected_revision=expected_revision, access_scope=self.__scope
            ),
        )

    async def read_chunks(self, entry_id, *, revision=None, max_chunks=100, max_bytes=1_048_576):
        return await self._call(
            "read",
            lambda: self.__store.read_chunks(
                entry_id,
                revision=revision,
                max_chunks=max_chunks,
                max_bytes=max_bytes,
                access_scope=self.__scope,
            ),
        )

    async def read_evidence(self, entry_id, *, revision=None, max_records=100, max_bytes=1_048_576):
        return await self._call(
            "read",
            lambda: self.__store.read_evidence(
                entry_id,
                revision=revision,
                max_records=max_records,
                max_bytes=max_bytes,
                access_scope=self.__scope,
            ),
        )
