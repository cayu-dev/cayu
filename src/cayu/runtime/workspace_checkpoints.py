"""Runtime-owned checkpoint publication before mutating tool success is staged."""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cayu._validation import canonical_durable_json_bytes
from cayu._workspace_mutation import workspace_mutation_task_settlement_probe
from cayu.core.events import Event, EventType
from cayu.runtime._runtime_records import RegisteredEnvironment
from cayu.runtime.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION_KEY,
    CURRENT_CHECKPOINT_SCHEMA_VERSION,
)
from cayu.runtime.sessions import Session, SessionStore
from cayu.workspaces.checkpoints import (
    WorkspaceCheckpointError,
    capture_workspace_checkpoint,
    load_workspace_checkpoint,
    pin_workspace_checkpoint,
    require_checkpoint_store,
    require_exclusive,
    restore_workspace_checkpoint,
    workspace_checkpoint_revision,
)

WORKSPACE_CHECKPOINTS_KEY = "workspace_checkpoints"


class WorkspaceCheckpointReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    phase: Literal["durable", "mutating", "checkpointing"]
    session_id: str
    session_instance_id: str
    environment_name: str
    artifact_store_id: str
    window_id: str
    binding_generation_id: str
    isolation_mechanism: str
    isolation_generation: str
    tool_call_id: str | None = None
    interaction_id: str | None = None
    before_revision: str | None = None
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_artifact_id: str = Field(pattern=r"^art_[0-9a-f]{32}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    pin_owner: str


def _isolation(registered: RegisteredEnvironment):
    # The observation contract belongs to the binding, not to tool dispatch.
    from cayu.runtime._tool_round_executor import _workspace_writer_isolation

    return _workspace_writer_isolation(registered)


async def _receipt(
    store: SessionStore, session: Session, name: str
) -> WorkspaceCheckpointReceipt | None:
    checkpoint = await store.load_checkpoint(session.id)
    records = (checkpoint or {}).get(WORKSPACE_CHECKPOINTS_KEY, {})
    if type(records) is not dict:
        raise WorkspaceCheckpointError("Invalid workspace checkpoint registry.")
    raw = records.get(name)
    return None if raw is None else WorkspaceCheckpointReceipt.model_validate(raw)


async def _publish(
    store: SessionStore,
    session: Session,
    previous: WorkspaceCheckpointReceipt | None,
    current: WorkspaceCheckpointReceipt,
) -> None:
    value = current.model_dump(mode="json")
    old_value = None if previous is None else previous.model_dump(mode="json")

    def transform(current_session: Session, checkpoint: dict[str, Any] | None) -> dict[str, Any]:
        if current_session.instance_id != session.instance_id:
            raise WorkspaceCheckpointError("Workspace checkpoint session incarnation changed.")
        updated = dict(checkpoint or {})
        updated[CHECKPOINT_SCHEMA_VERSION_KEY] = CURRENT_CHECKPOINT_SCHEMA_VERSION
        records = dict(updated.get(WORKSPACE_CHECKPOINTS_KEY, {}))
        old = records.get(current.environment_name)
        if old != old_value and old != value:
            raise WorkspaceCheckpointError("Workspace checkpoint publication lost its predecessor.")
        records[current.environment_name] = value
        updated[WORKSPACE_CHECKPOINTS_KEY] = records
        return updated

    event = Event(
        id="evt_wcp_"
        + hashlib.sha256(canonical_durable_json_bytes(value, "workspace_checkpoint")).hexdigest(),
        type=EventType.WORKSPACE_CHECKPOINT_UPDATED,
        session_id=session.id,
        environment_name=current.environment_name,
        interaction_id=current.interaction_id,
        payload={
            key: value[key]
            for key in (
                "phase",
                "window_id",
                "tool_call_id",
                "interaction_id",
                "before_revision",
                "revision",
                "bytes",
                "duration_ms",
            )
        },
    )
    await store.publish_checkpoint_and_events(
        session.id,
        checkpoint_transform=transform,
        events=[event],
        expected_run_epoch=session.run_epoch,
    )
    if await _receipt(store, session, current.environment_name) != current:
        raise WorkspaceCheckpointError("Workspace checkpoint publication readback mismatch.")


async def _ensure_workspace_checkpoint(
    store: SessionStore,
    session: Session,
    registered: RegisteredEnvironment,
) -> None:
    """Restore/verify the last published revision before exposing an environment."""
    policy = registered.spec.workspace_checkpoint_policy
    if policy is None:
        return
    workspace = registered.environment.workspace
    if workspace is None:
        raise WorkspaceCheckpointError("Workspace checkpoint policy requires a workspace.")
    artifacts = require_checkpoint_store(registered.environment.artifact_store)
    isolation = _isolation(registered)
    require_exclusive(isolation)
    current = await _receipt(store, session, registered.spec.name)
    if current is None:
        owner = f"workspace:{session.instance_id}:baseline:{registered.spec.name}"
        started = time.monotonic()
        artifact_id, manifest = await capture_workspace_checkpoint(
            workspace,
            artifacts,
            policy=policy,
            environment_name=registered.spec.name,
            owner=owner,
            isolation=lambda: _isolation(registered),
        )
        encoded = canonical_durable_json_bytes(
            manifest.model_dump(mode="json"), "workspace_manifest"
        )
        await _publish(
            store,
            session,
            None,
            WorkspaceCheckpointReceipt(
                phase="durable",
                session_id=session.id,
                session_instance_id=session.instance_id,
                environment_name=registered.spec.name,
                artifact_store_id=artifacts.id,
                window_id="baseline",
                binding_generation_id=registered.binding_generation_id,
                isolation_mechanism=isolation.mechanism,
                isolation_generation=isolation.generation,
                revision=manifest.revision,
                manifest_artifact_id=artifact_id,
                manifest_sha256=hashlib.sha256(encoded).hexdigest(),
                bytes=sum(entry.size_bytes for entry in manifest.files),
                duration_ms=int((time.monotonic() - started) * 1000),
                pin_owner=owner,
            ),
        )
        return
    if current.phase != "durable":
        raise WorkspaceCheckpointError(
            "Workspace mutation outcome is unknown; checkpoint recovery is blocked."
        )
    inherited = current.session_id == session.parent_session_id
    if not inherited and (
        current.session_id != session.id or current.session_instance_id != session.instance_id
    ):
        raise WorkspaceCheckpointError("Workspace checkpoint owner does not match the session.")
    if (
        current.artifact_store_id != artifacts.id
        or current.environment_name != registered.spec.name
    ):
        raise WorkspaceCheckpointError("Workspace checkpoint storage authority changed.")
    manifest = await load_workspace_checkpoint(
        artifacts,
        current.manifest_artifact_id,
        policy=policy,
        expected_manifest_sha256=current.manifest_sha256,
    )
    if manifest.revision != current.revision:
        raise WorkspaceCheckpointError("Workspace checkpoint receipt conflicts with manifest.")
    if inherited:
        owner = f"workspace:{session.instance_id}:inherited:{current.manifest_artifact_id}"
        await pin_workspace_checkpoint(
            artifacts,
            current.manifest_artifact_id,
            owner=owner,
            policy=policy,
            expected_manifest_sha256=current.manifest_sha256,
        )
        adopted = current.model_copy(
            update={
                "session_id": session.id,
                "session_instance_id": session.instance_id,
                "pin_owner": owner,
            }
        )
        await _publish(store, session, current, adopted)
        current = adopted
    if await workspace_checkpoint_revision(workspace, policy=policy) != current.revision:
        if current.binding_generation_id == registered.binding_generation_id and not inherited:
            raise WorkspaceCheckpointError(
                "Live workspace changed outside its checkpointed mutation."
            )
        await restore_workspace_checkpoint(
            workspace,
            artifacts,
            manifest,
            policy=policy,
            isolation=lambda: _isolation(registered),
        )

    if (
        current.binding_generation_id != registered.binding_generation_id
        or current.isolation_mechanism != isolation.mechanism
        or current.isolation_generation != isolation.generation
    ):
        await _publish(
            store,
            session,
            current,
            current.model_copy(
                update={
                    "binding_generation_id": registered.binding_generation_id,
                    "isolation_mechanism": isolation.mechanism,
                    "isolation_generation": isolation.generation,
                }
            ),
        )


async def begin_workspace_checkpoint_mutation(
    store: SessionStore,
    session: Session,
    registered: RegisteredEnvironment,
    *,
    window_id: str,
    tool_call_id: str,
    interaction_id: str | None,
) -> None:
    if registered.spec.workspace_checkpoint_policy is None:
        return
    await ensure_workspace_checkpoint(store, session, registered)
    current = await _receipt(store, session, registered.spec.name)
    if current is None:
        raise WorkspaceCheckpointError("Workspace mutation lost its durable baseline.")
    await _publish(
        store,
        session,
        current,
        current.model_copy(
            update={
                "phase": "mutating",
                "window_id": window_id,
                "tool_call_id": tool_call_id,
                "interaction_id": interaction_id,
                "before_revision": current.revision,
            }
        ),
    )


async def _complete_workspace_checkpoint_mutation(
    store: SessionStore,
    session: Session,
    registered: RegisteredEnvironment,
    *,
    window_id: str,
    successful: bool,
    window_exclusive: Callable[[], bool],
) -> None:
    policy = registered.spec.workspace_checkpoint_policy
    if policy is None:
        return
    current = await _receipt(store, session, registered.spec.name)
    if current is None or current.window_id != window_id:
        raise WorkspaceCheckpointError("Workspace mutation lost its checkpoint intent.")
    if current.phase == "durable":
        return
    if not successful:
        raise WorkspaceCheckpointError(
            "Unsuccessful or interrupted workspace mutation requires reconciliation."
        )
    if current.phase != "checkpointing":
        checkpointing = current.model_copy(update={"phase": "checkpointing", "duration_ms": 0})
        await _publish(store, session, current, checkpointing)
        current = checkpointing
    workspace = registered.environment.workspace
    if workspace is None:
        raise WorkspaceCheckpointError("Workspace disappeared before checkpoint publication.")
    artifacts = require_checkpoint_store(registered.environment.artifact_store)
    owner = f"workspace:{session.instance_id}:{window_id}"

    def isolation():
        evidence = _isolation(registered)
        if (
            not window_exclusive()
            or evidence.mechanism != current.isolation_mechanism
            or evidence.generation != current.isolation_generation
        ):
            raise WorkspaceCheckpointError("Workspace mutation isolation is ambiguous.")
        return evidence

    started = time.monotonic()
    artifact_id, manifest = await capture_workspace_checkpoint(
        workspace,
        artifacts,
        policy=policy,
        environment_name=registered.spec.name,
        owner=owner,
        isolation=isolation,
    )
    encoded = canonical_durable_json_bytes(manifest.model_dump(mode="json"), "workspace_manifest")
    await _publish(
        store,
        session,
        current,
        current.model_copy(
            update={
                "phase": "durable",
                "revision": manifest.revision,
                "manifest_artifact_id": artifact_id,
                "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": sum(entry.size_bytes for entry in manifest.files),
                "duration_ms": int((time.monotonic() - started) * 1000),
                "pin_owner": owner,
            }
        ),
    )


async def _bounded_checkpoint_operation(
    registered: RegisteredEnvironment,
    operation: Callable[[], Awaitable[None]],
) -> None:
    policy = registered.spec.workspace_checkpoint_policy
    if policy is None:
        return
    await registered.workspace_mutation_fence.wait_until_available()
    task = asyncio.ensure_future(operation())
    try:
        done, _ = await asyncio.wait({task}, timeout=policy.timeout_seconds)
        if not done:
            raise WorkspaceCheckpointError("Workspace checkpoint deadline exceeded.")
        await task
    except BaseException:
        if not task.done():
            # The environment cannot be released/reused while a delegated
            # restore or artifact operation can still finish. Do not cancel an
            # opaque operation and mistake coroutine cancellation for settlement.
            registered.workspace_mutation_fence.fail_closed(
                workspace_mutation_task_settlement_probe(task)
            )
        raise


async def ensure_workspace_checkpoint(
    store: SessionStore,
    session: Session,
    registered: RegisteredEnvironment,
) -> None:
    await _bounded_checkpoint_operation(
        registered, lambda: _ensure_workspace_checkpoint(store, session, registered)
    )


async def complete_workspace_checkpoint_mutation(
    store: SessionStore,
    session: Session,
    registered: RegisteredEnvironment,
    *,
    window_id: str,
    successful: bool,
    window_exclusive: Callable[[], bool] = lambda: True,
) -> None:
    await _bounded_checkpoint_operation(
        registered,
        lambda: _complete_workspace_checkpoint_mutation(
            store,
            session,
            registered,
            window_id=window_id,
            successful=successful,
            window_exclusive=window_exclusive,
        ),
    )
