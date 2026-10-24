"""Application-owned task declarations, detached at every public boundary."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cayu.sessions.store import SessionStore
    from cayu.tasks.creation import TaskCreate
    from cayu.tasks.operations import TaskCreationOperation
    from cayu.tasks.records import Task
    from cayu.tasks.store import TaskStore

from cayu.tasks.types import TaskTypeSpec, copy_task_type
from cayu.vaults.redaction import SecretRedactor


class ApplicationTaskTypes:
    def __init__(self, redactor: SecretRedactor) -> None:
        self._redactor = redactor
        self._types: dict[str, TaskTypeSpec] = {}

    def register(self, spec: TaskTypeSpec) -> TaskTypeSpec:
        copied = copy_task_type(spec)
        document = copied.model_dump(mode="json", warnings="error")
        if self._redactor.redact_json(document) != document:
            raise ValueError("Public task type declarations must not contain workload secrets.")
        if copied.name in self._types:
            raise ValueError("Task type is already registered.")
        self._types[copied.name] = copied
        return copy_task_type(copied)

    async def create_operation(
        self,
        request: TaskCreate,
        operation: TaskCreationOperation,
        *,
        agent_lookup: Callable[[str], object],
        task_store: TaskStore | None,
        session_store: SessionStore,
    ) -> Task:
        from cayu._application_task_creation import create_task

        request, operation = self.prepare_operation(request, operation, agent_lookup=agent_lookup)
        return await create_task(
            request,
            task_store=task_store,
            session_store=session_store,
            redactor=self._redactor,
            creation_operation=operation,
        )

    def prepare_operation(
        self,
        request: TaskCreate,
        operation: TaskCreationOperation,
        *,
        agent_lookup: Callable[[str], object],
    ) -> tuple[TaskCreate, TaskCreationOperation]:
        from cayu._validation import canonical_durable_json_bytes
        from cayu.tasks.creation import copy_task_create
        from cayu.tasks.operations import TaskCreationConflict, copy_creation_operation

        request = copy_task_create(request)
        operation = copy_creation_operation(operation)
        try:
            declared = self.get(request.type)
        except KeyError:
            raise ValueError("Task type is not declared.") from None
        if declared.agent_name is not None:
            try:
                agent_lookup(declared.agent_name)
            except KeyError:
                raise ValueError("Task specialist is not available.") from None
        if canonical_durable_json_bytes(declared.model_dump(mode="json"), "task type") != (
            canonical_durable_json_bytes(operation.declaration.model_dump(mode="json"), "task type")
        ):
            raise TaskCreationConflict("Task declaration changed.")
        return request, operation

    def get(self, name: str) -> TaskTypeSpec:
        return copy_task_type(self._types[name])

    def list_types(self) -> list[TaskTypeSpec]:
        return [copy_task_type(self._types[name]) for name in sorted(self._types)]
