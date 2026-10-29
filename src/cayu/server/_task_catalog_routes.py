"""Declared task creation and exact replay; workers retain execution ownership."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Literal, cast

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cayu._validation import (
    canonical_durable_json_bytes,
    copy_durable_json_object,
    require_durable_clean_nonblank,
)
from cayu.server.auth import AuthContext
from cayu.sessions.invocation import (
    InvocationOrigin,
    InvocationOriginTrust,
    SessionExecutionSource,
    TaskExecutionSource,
)
from cayu.tasks.creation import TaskCreate, task_create_with_runtime_invocation
from cayu.tasks.graphs import (
    TASK_GRAPH_MAX_NODES,
    TaskGraphConflict,
    TaskGraphCreate,
    TaskGraphNode,
)
from cayu.tasks.operations import (
    TaskCreationConflict,
    TaskCreationOperation,
    TaskCreationResultDeleted,
    TaskReplaySource,
    task_replay_source,
)
from cayu.tasks.types import TaskTypeSpec, task_input_validator

if TYPE_CHECKING:
    from fastapi import APIRouter
    from fastapi.params import Depends

    from cayu.applications import CayuApp


class TaskTypesResponse(BaseModel):
    task_types: list[TaskTypeSpec]


class _TaskOperationBody(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    @field_validator("operation_id", "type")
    @classmethod
    def clean_identity(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "task operation identity")

    operation_id: str = Field(min_length=1, max_length=256)
    type: str = Field(min_length=1, max_length=256)


class DeclaredTaskCreateBody(_TaskOperationBody):
    input: dict[str, Any]

    @field_validator("input", mode="before")
    @classmethod
    def copy_input(cls, value: Any) -> dict[str, Any]:
        return copy_durable_json_object(value, "task input")


class DeclaredTaskGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    key: str = Field(min_length=1, max_length=256)
    type: str = Field(min_length=1, max_length=256)
    input: dict[str, Any]
    prerequisite_keys: tuple[str, ...] = Field(default=(), max_length=TASK_GRAPH_MAX_NODES)

    @field_validator("key", "type")
    @classmethod
    def clean_identity(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "graph node identity")

    @field_validator("input", mode="before")
    @classmethod
    def copy_input(cls, value: Any) -> dict[str, Any]:
        return copy_durable_json_object(value, "task input")


class DeclaredTaskGraphCreateBody(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    operation_id: str = Field(min_length=1, max_length=256)
    nodes: tuple[DeclaredTaskGraphNode, ...] = Field(min_length=1, max_length=TASK_GRAPH_MAX_NODES)

    @field_validator("operation_id")
    @classmethod
    def clean_identity(cls, value: str) -> str:
        return require_durable_clean_nonblank(value, "graph operation identity")

    @model_validator(mode="after")
    def validate_keys(self) -> DeclaredTaskGraphCreateBody:
        keys = {node.key for node in self.nodes}
        if len(keys) != len(self.nodes):
            raise ValueError("Graph node keys must be unique.")
        for node in self.nodes:
            if len(set(node.prerequisite_keys)) != len(node.prerequisite_keys) or any(
                key not in keys or key == node.key for key in node.prerequisite_keys
            ):
                raise ValueError("Graph prerequisites must name distinct other nodes.")
        return self


class TaskGraphCreationResponse(BaseModel):
    graph_id: str
    task_ids: dict[str, str]


class TaskReplayBody(_TaskOperationBody):
    source: TaskReplaySource


class TaskCreationResponse(BaseModel):
    task_id: str


class TaskReplayAvailability(BaseModel):
    available: bool
    unavailable_reason: (
        Literal[
            "unsupported",
            "task_type_not_declared",
            "specialist_unavailable",
            "input_invalid",
            "source_identity_unavailable",
            "not_task_session",
            "task_source_unavailable",
            "task_source_ambiguous",
        ]
        | None
    ) = None
    source: TaskReplaySource | None = None


def register_task_catalog_routes(
    router: APIRouter,
    *,
    mutation_router: APIRouter,
    cayu_app: CayuApp,
    protected: list[Depends],
    optional_auth_context: Depends,
    resolve_public_session_id: Callable[[str], Awaitable[str]],
) -> None:
    auth_default = cast("AuthContext | None", optional_auth_context)

    def declaration(
        name: str,
    ) -> TaskTypeSpec | Literal["task_type_not_declared", "specialist_unavailable"]:
        try:
            spec = cayu_app.get_task_type(name)
        except KeyError:
            return "task_type_not_declared"
        if spec.agent_name is not None:
            try:
                cayu_app.get_agent(spec.agent_name)
            except KeyError:
                return "specialist_unavailable"
        return spec

    @router.get("/task-types", response_model=TaskTypesResponse, dependencies=protected)
    async def task_types() -> TaskTypesResponse:
        return TaskTypesResponse(task_types=cayu_app.list_task_types())

    @router.get(
        "/tasks/{task_id}/replay", response_model=TaskReplayAvailability, dependencies=protected
    )
    async def replay_availability(task_id: str) -> TaskReplayAvailability:
        store = cayu_app.task_store
        if store is None or not store.supports_task_creation_operations:
            return TaskReplayAvailability(available=False, unavailable_reason="unsupported")
        task = await store.load_task(task_id)
        if task is None:
            raise HTTPException(404, detail="Task not found.")
        spec = declaration(task.type)
        if isinstance(spec, str):
            return TaskReplayAvailability(available=False, unavailable_reason=spec)
        if task.assigned_agent_name != spec.agent_name:
            return TaskReplayAvailability(
                available=False, unavailable_reason="specialist_unavailable"
            )
        try:
            valid = task_input_validator(spec).is_valid(task.input)
        except Exception:
            valid = False
        if not valid:
            return TaskReplayAvailability(available=False, unavailable_reason="input_invalid")
        if (task.session_id is None) != (task.session_instance_id is None):
            return TaskReplayAvailability(
                available=False, unavailable_reason="source_identity_unavailable"
            )
        source = task_replay_source(task)
        document = source.model_dump(mode="json")
        if cayu_app.redact_json(document) != document:
            return TaskReplayAvailability(
                available=False, unavailable_reason="source_identity_unavailable"
            )
        return TaskReplayAvailability(available=True, source=source)

    @router.get(
        "/sessions/{session_id}/task-replay",
        response_model=TaskReplayAvailability,
        dependencies=protected,
    )
    async def session_replay_availability(session_id: str) -> TaskReplayAvailability:
        store = cayu_app.task_store
        if store is None or not store.supports_task_creation_operations:
            return TaskReplayAvailability(available=False, unavailable_reason="unsupported")
        private_id = await resolve_public_session_id(session_id)
        session = await cayu_app.session_store.load(private_id)
        if session is None:
            raise HTTPException(404, detail="Session not found.")
        if session.invocation.source is not SessionExecutionSource.TASK:
            return TaskReplayAvailability(available=False, unavailable_reason="not_task_session")
        try:
            candidates = await store.task_replay_sources_for_session(
                session.id,
                session.instance_id,
                session.invocation.root_invocation_id,
            )
        except NotImplementedError:
            return TaskReplayAvailability(available=False, unavailable_reason="unsupported")
        if len(candidates) != 1:
            return TaskReplayAvailability(
                available=False,
                unavailable_reason=(
                    "task_source_ambiguous" if candidates else "task_source_unavailable"
                ),
            )
        current = await cayu_app.session_store.load(private_id)
        if current is None or current.instance_id != session.instance_id:
            return TaskReplayAvailability(
                available=False, unavailable_reason="task_source_unavailable"
            )
        selected = candidates[0]
        try:
            availability = await replay_availability(selected.id)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            return TaskReplayAvailability(
                available=False, unavailable_reason="task_source_unavailable"
            )
        if availability.available and availability.source != task_replay_source(selected):
            return TaskReplayAvailability(
                available=False, unavailable_reason="task_source_unavailable"
            )
        return availability

    async def create(
        *,
        operation_id: str,
        type_name: str,
        task_input: dict[str, Any],
        source: TaskReplaySource | None,
        actor: AuthContext | None,
    ) -> TaskCreationResponse:
        store = cayu_app.task_store
        if store is None or not store.supports_task_creation_operations:
            raise HTTPException(501, detail={"code": "task_creation_unsupported"})
        spec = declaration(type_name)
        if isinstance(spec, str):
            raise HTTPException(422, detail={"code": spec})
        origin = (
            None
            if actor is None
            else InvocationOrigin(
                trust=InvocationOriginTrust.SERVER_VERIFIED,
                subject=actor.subject,
                tenant=actor.tenant,
            )
        )
        # A refreshed token's incidental claims do not change the authenticated
        # actor. The same actor/operation always selects the same task identity.
        identity = sha256(
            canonical_durable_json_bytes(
                {
                    "operation_id": operation_id,
                    "actor": None if origin is None else origin.model_dump(mode="json"),
                },
                "task operation identity",
            )
        ).hexdigest()
        task_id = f"task-operation-{identity}"
        request = task_create_with_runtime_invocation(
            TaskCreate(
                task_id=task_id,
                type=spec.name,
                assigned_agent_name=spec.agent_name,
                input=task_input,
            ),
            source=TaskExecutionSource.HTTP_TASK,
            verified_origin=origin,
        )
        operation = TaskCreationOperation(
            operation_id=operation_id, declaration=spec, replay_source=source
        )
        try:
            result = await cayu_app.create_task_operation(request, operation)
        except TaskCreationResultDeleted:
            raise HTTPException(409, detail={"code": "task_creation_result_deleted"}) from None
        except TaskCreationConflict:
            raise HTTPException(409, detail={"code": "task_creation_conflict"}) from None
        except NotImplementedError:
            raise HTTPException(501, detail={"code": "task_creation_unsupported"}) from None
        except (TypeError, ValueError):
            raise HTTPException(422, detail={"code": "task_creation_invalid"}) from None
        if result.id != task_id:
            raise HTTPException(503, detail={"code": "task_creation_result_invalid"})
        return TaskCreationResponse(task_id=result.id)

    @mutation_router.post(
        "/task-graphs", response_model=TaskGraphCreationResponse, dependencies=protected
    )
    async def create_graph(
        body: DeclaredTaskGraphCreateBody,
        actor: AuthContext | None = auth_default,
    ) -> TaskGraphCreationResponse:
        store = cayu_app.task_store
        if store is None or not store.supports_task_graphs:
            raise HTTPException(501, detail={"code": "task_graph_creation_unsupported"})
        if any(cayu_app.redact_json(node.key) != node.key for node in body.nodes):
            raise HTTPException(422, detail={"code": "task_graph_identity_invalid"})
        origin = (
            None
            if actor is None
            else InvocationOrigin(
                trust=InvocationOriginTrust.SERVER_VERIFIED,
                subject=actor.subject,
                tenant=actor.tenant,
            )
        )
        identity = sha256(
            canonical_durable_json_bytes(
                {
                    "operation_id": body.operation_id,
                    "actor": None if origin is None else origin.model_dump(mode="json"),
                },
                "graph operation identity",
            )
        ).hexdigest()
        graph_id = f"task-graph-operation-{identity}"
        task_ids = {
            node.key: "graph-task-"
            + sha256(
                canonical_durable_json_bytes(
                    {"graph_id": graph_id, "key": node.key}, "graph node identity"
                )
            ).hexdigest()
            for node in body.nodes
        }
        nodes = []
        for node in sorted(body.nodes, key=lambda item: item.key):
            spec = declaration(node.type)
            if isinstance(spec, str):
                raise HTTPException(422, detail={"code": spec})
            try:
                valid = task_input_validator(spec).is_valid(node.input)
            except Exception:
                valid = False
            if not valid:
                raise HTTPException(422, detail={"code": "task_graph_input_invalid"})
            # Bind exact replay to the declaration, just as single-task operations do.
            declaration_sha256 = sha256(
                canonical_durable_json_bytes(spec.model_dump(mode="json"), "task declaration")
            ).hexdigest()
            request = task_create_with_runtime_invocation(
                TaskCreate(
                    task_id=task_ids[node.key],
                    type=spec.name,
                    assigned_agent_name=spec.agent_name,
                    input=node.input,
                    metadata={"task_type_declaration_sha256": declaration_sha256},
                ),
                source=TaskExecutionSource.HTTP_TASK,
                verified_origin=origin,
            )
            nodes.append(
                TaskGraphNode(
                    task=request,
                    prerequisite_task_ids=tuple(task_ids[key] for key in node.prerequisite_keys),
                )
            )
        try:
            result = await cayu_app.create_task_graph(
                TaskGraphCreate(graph_id=graph_id, nodes=tuple(nodes))
            )
        except TaskGraphConflict:
            raise HTTPException(409, detail={"code": "task_graph_creation_conflict"}) from None
        except NotImplementedError:
            raise HTTPException(501, detail={"code": "task_graph_creation_unsupported"}) from None
        except (TypeError, ValueError):
            raise HTTPException(422, detail={"code": "task_graph_creation_invalid"}) from None
        return TaskGraphCreationResponse(graph_id=result.graph_id, task_ids=task_ids)

    @mutation_router.post("/tasks", response_model=TaskCreationResponse, dependencies=protected)
    async def create_task(
        body: DeclaredTaskCreateBody,
        actor: AuthContext | None = auth_default,
    ) -> TaskCreationResponse:
        return await create(
            operation_id=body.operation_id,
            type_name=body.type,
            task_input=body.input,
            source=None,
            actor=actor,
        )

    @mutation_router.post(
        "/tasks/{task_id}/replay", response_model=TaskCreationResponse, dependencies=protected
    )
    async def replay_task(
        task_id: str,
        body: TaskReplayBody,
        actor: AuthContext | None = auth_default,
    ) -> TaskCreationResponse:
        if task_id != body.source.task_id:
            raise HTTPException(409, detail={"code": "task_replay_source_conflict"})
        return await create(
            operation_id=body.operation_id,
            type_name=body.type,
            task_input={},
            source=body.source,
            actor=actor,
        )
