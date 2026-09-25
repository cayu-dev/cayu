"""Application-scoped artifact access using immutable stored classification."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Literal

from cayu._validation import copy_label_map, require_clean_nonblank
from cayu.artifacts.base import ArtifactMetadata, ArtifactScope, ArtifactStore
from cayu.sessions.access import SessionAccessDenied, SessionAccessScope, _SessionAccessBounds


@dataclass(frozen=True, slots=True)
class _ArtifactAccess:
    bounds: _SessionAccessBounds
    action: Literal["read", "create", "delete", "list"]
    labels: tuple[tuple[str, str], ...] = ()


_active: ContextVar[_ArtifactAccess | None] = ContextVar("cayu_artifact_access", default=None)


def creation_labels() -> dict[str, str]:
    access = _active.get()
    return {} if access is None else dict(access.labels)


def visible(artifact: ArtifactMetadata) -> bool:
    access = _active.get()
    return access is None or access.bounds.matches(artifact.labels)


def require_artifact(artifact: ArtifactMetadata) -> None:
    access = _active.get()
    if access is None or access.action == "list":
        return
    if not access.bounds.matches(artifact.labels):
        raise SessionAccessDenied()
    if access.action in ("create", "delete") and not access.bounds.matches(
        artifact.labels, access.action
    ):
        raise SessionAccessDenied()


class ScopedArtifactAccess:
    """Explicit artifact operations; raw stores remain trusted operator handles."""

    def __init__(
        self,
        store: ArtifactStore,
        *,
        admitted: SessionAccessScope,
        environment_name: str,
        resolve: Callable[[], Awaitable[SessionAccessScope]],
    ) -> None:
        if (
            not isinstance(store, ArtifactStore)
            or type(store).__dict__.get("artifact_access_version") != 1
        ):
            raise NotImplementedError("Store does not support native artifact access v1.")
        if type(admitted) is not SessionAccessScope or not callable(resolve):
            raise TypeError("Artifact access requires a scope and trusted resolver.")
        self.__environment_name = require_clean_nonblank(environment_name, "environment_name")
        self.__store, self.__admitted, self.__resolve = store, admitted, resolve

    async def _call(self, action, operation, *, labels=None):
        current = await self.__resolve()
        if type(current) is not SessionAccessScope:
            raise TypeError("Artifact access resolver must return SessionAccessScope.")
        bounds = _SessionAccessBounds(self.__admitted, current)
        if action == "create" and (
            not bounds.matches(labels or {}, "create") or not bounds.matches(labels or {})
        ):
            raise SessionAccessDenied()
        token = _active.set(_ArtifactAccess(bounds, action, tuple((labels or {}).items())))
        try:
            return await operation()
        except FileNotFoundError:
            raise SessionAccessDenied() from None
        finally:
            _active.reset(token)

    async def put_bytes(
        self,
        content: bytes,
        *,
        labels: dict[str, str],
        filename: str,
        content_type: str | None = None,
        artifact_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        # Standalone classification does not claim access to a session or environment.
        labels = copy_label_map(labels, "artifact labels", allow_reserved=False)
        return await self._call(
            "create",
            lambda: self.__store.put_bytes(
                content,
                artifact_id=artifact_id,
                filename=filename,
                content_type=content_type,
                scope=ArtifactScope.ENVIRONMENT,
                environment_name=self.__environment_name,
                metadata=metadata,
            ),
            labels=labels,
        )

    async def read_bytes(self, artifact_id: str, *, max_bytes: int = 1_048_576):
        if type(max_bytes) is not int or not 1 <= max_bytes <= 4_194_304:
            raise ValueError("Scoped artifact reads require a bound between 1 and 4194304 bytes.")
        return await self._call(
            "read", lambda: self.__store.read_bytes(artifact_id, max_bytes=max_bytes)
        )

    async def read_range(self, artifact_id: str, *, offset: int, max_bytes: int):
        if type(max_bytes) is not int or not 1 <= max_bytes <= 4_194_304:
            raise ValueError("Scoped artifact reads require a bound between 1 and 4194304 bytes.")
        return await self._call(
            "read", lambda: self.__store.read_range(artifact_id, offset=offset, max_bytes=max_bytes)
        )

    async def list(self, *, limit: int = 100):
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("Artifact listing limit must be between 1 and 1000.")
        return await self._call("list", lambda: self.__store.list(limit=limit))

    async def delete(self, artifact_id: str) -> None:
        await self._call("delete", lambda: self.__store.delete(artifact_id))


def runtime_artifact_operation(action):
    """Revalidate native artifact calls made by an admitted execution."""
    from functools import wraps

    def decorate(operation):
        @wraps(operation)
        async def guarded(self, *args, **kwargs):
            from cayu.resource_access import (
                _active as execution_active,
            )
            from cayu.resource_access import (
                effective_bounds,
                require_dispatch,
            )

            execution = execution_active.get()
            if execution is None or _active.get() is not None:
                return await operation(self, *args, **kwargs)
            await require_dispatch()
            bounds = await effective_bounds(execution.binding, execution.policy, "artifacts")
            labels = dict(execution.labels)
            if execution.store is not None:
                if execution.session_id is None:
                    raise SessionAccessDenied()
                session = await execution.store.load(execution.session_id)
                if session is None or session.invocation.resource_access != execution.binding:
                    raise SessionAccessDenied()
                labels = dict(session.labels)
            if action == "create":
                if not bounds.matches(labels, "create") or not bounds.matches(labels):
                    raise SessionAccessDenied()
                owner_id = kwargs.get("session_id")
                if owner_id is not None and owner_id != execution.session_id:
                    raise SessionAccessDenied()
            token = _active.set(_ArtifactAccess(bounds, action, tuple(labels.items())))
            try:
                return await operation(self, *args, **kwargs)
            except FileNotFoundError:
                raise SessionAccessDenied() from None
            finally:
                _active.reset(token)

        return guarded

    return decorate
