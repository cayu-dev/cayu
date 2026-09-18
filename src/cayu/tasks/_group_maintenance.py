"""Bounded barrier observation for existing durable worker loops."""

from __future__ import annotations

from dataclasses import dataclass

from cayu.runtime._task_store_operation_boundary import (
    capture_task_store_operation,
    raise_task_store_operation_failure,
)
from cayu.tasks.base import TaskStore
from cayu.tasks.graphs import graph_identifier
from cayu.tasks.groups import TaskGroupUnavailable
from cayu.vaults.redaction import SecretRedactor


@dataclass
class TaskGroupMaintenance:
    """The cursor schedules observations; only the store clock decides expiry."""

    after_group_id: str | None = None
    next_scan_at: float = 0.0

    async def step(self, store: TaskStore, redactor: SecretRedactor, *, now: float) -> None:
        outcome = await capture_task_store_operation(
            lambda: self.advance(store, now=now),
            operation_name="Task group maintenance",
            redactor=redactor,
        )
        if outcome.failure is not None:
            raise_task_store_operation_failure(outcome.failure)

    async def advance(self, store: TaskStore, *, now: float) -> None:
        """Run under the caller's existing store-operation/error owner.

        Dispatchers already own their TaskStore boundary and expose a separate
        diagnostic interface, not the application's private secret registry.
        """
        if not store.supports_task_group_quiescence or now < self.next_scan_at:
            return
        identities = await store.list_task_group_reconciliation_candidates(
            after_group_id=self.after_group_id,
            limit=32,
        )
        if (
            type(identities) is not list
            or len(identities) > 32
            or any(type(identity) is not str for identity in identities)
            or identities != sorted(set(identities))
        ):
            raise TaskGroupUnavailable("Task group maintenance returned invalid identities.")
        for identity in identities:
            graph_identifier(identity)
            if self.after_group_id is not None and identity <= self.after_group_id:
                raise TaskGroupUnavailable("Task group maintenance cursor did not advance.")
            await store.reconcile_task_group(identity)
            # Advance only after acknowledgement. A lost acknowledgement is
            # replayed on the next maintenance turn without duplicate events.
            self.after_group_id = identity
        if len(identities) < 32:
            self.after_group_id = None
            self.next_scan_at = now + 1.0
