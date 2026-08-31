from __future__ import annotations

import asyncio
import json
import multiprocessing
import warnings
from collections.abc import AsyncIterator
from copy import deepcopy

import psycopg
import pytest
from tests.core._execution_profile_fixtures import versioned_test_provider_identity

from cayu.core import (
    AgentSpec,
    Event,
    EventType,
    ExecutionProfileBehaviorIdentity,
    Message,
    ToolContext,
    ToolResult,
)
from cayu.environments import Environment, EnvironmentSpec
from cayu.providers import ModelProvider, ModelRequest, ModelStreamEvent
from cayu.runtime import (
    AfterToolCallDecision,
    BeforeToolCallDecision,
    BeforeToolCallHookContext,
    CayuApp,
    DispatchRequest,
    IncompleteSessionRecoveryAction,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionsRecoveryRequest,
    InMemorySessionStore,
    InMemoryTaskStore,
    InterruptSessionRequest,
    RunRequest,
    RuntimeHook,
    SessionQuery,
    SessionStatus,
    TaskClaimLost,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStoreDispatcher,
)
from cayu.runtime._durable_subagent_coordinator import DurableSubagentCoordinator
from cayu.runtime._durable_subagents import (
    DurableSubagentAuthority,
    DurableSubagentSubmissionIntent,
    DurableSubagentSubmissionSeed,
    durable_subagent_authority_rejected,
    durable_subagent_request_sha256,
    durable_subagent_submission_from_checkpoint,
    durable_subagent_submission_receipt_from_checkpoint,
    durable_subagent_submission_seed_from_checkpoint,
    durable_subagent_submission_unsettled,
    is_durable_subagent_authority_rejected,
    is_durable_subagent_submission_unsettled,
    new_durable_subagent_submission_intent,
    new_durable_subagent_submission_seed,
    require_durable_subagent_intent_matches_seed,
)
from cayu.runtime.dispatch import _queued_dispatch_request_sha256
from cayu.runtime.tool_discovery import (
    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
    ToolDiscoveryViewState,
)
from cayu.storage import (
    PostgresSessionStore,
    PostgresTaskStore,
    SQLiteSessionStore,
    SQLiteTaskStore,
)
from cayu.storage.migrations import SchemaMode
from cayu.tools import (
    SubagentExecutionMode,
    SubagentResultTool,
    SubagentSpec,
    SubagentTool,
    project_terminal_subagent_result,
)
from cayu.vaults import SecretRedactor, SecretRef, StaticVault

_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="tests:durable-subagent-tool",
    behavior_version="1",
    implementation_version="1",
)
_MODIFY_DURABLE_SUBAGENT_TASK_HOOK_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="tests:modify-durable-subagent-task-hook",
    behavior_version="1",
    implementation_version="1",
)
_MODIFY_DURABLE_SUBAGENT_TASK_AND_RESULT_HOOK_PROFILE_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="tests:modify-durable-subagent-task-and-result-hook",
    behavior_version="1",
    implementation_version="1",
)


async def _collect(stream: AsyncIterator[Event]) -> list[Event]:
    return [event async for event in stream]


class _SimulatedWorkerLoss(BaseException):
    pass


def test_unsettled_submission_signal_uses_private_immutable_provenance() -> None:
    signal = durable_subagent_submission_unsettled(
        parent_session_id="parent",
        tool_name="spawn_subagent",
        idempotency_key="operation",
        failure=ConnectionError("transient store failure"),
    )
    assert is_durable_subagent_submission_unsettled(
        signal,
        parent_session_id="parent",
        tool_name="spawn_subagent",
        idempotency_key="operation",
    )
    object.__setattr__(signal, "parent_session_id", "other-parent")
    object.__setattr__(signal, "tool_name", "other-tool")
    object.__setattr__(signal, "idempotency_key", "other-operation")
    assert not is_durable_subagent_submission_unsettled(
        signal,
        parent_session_id="other-parent",
        tool_name="other-tool",
        idempotency_key="other-operation",
    )
    assert is_durable_subagent_submission_unsettled(
        ExceptionGroup("wrapped", [signal]),
        parent_session_id="parent",
        tool_name="spawn_subagent",
        idempotency_key="operation",
    )


def test_durable_subagent_authority_rejection_requires_runtime_provenance() -> None:
    signal = durable_subagent_authority_rejected()
    assert is_durable_subagent_authority_rejected(signal)
    assert not is_durable_subagent_authority_rejected(
        RuntimeError("Prepared durable subagent authority is invalid.")
    )


class _DurableSubagentProvider(ModelProvider):
    name = "durable-subagent-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        first_text = request.messages[0].content[0].text
        if first_text == "parent task" and len(request.messages) == 1:
            yield ModelStreamEvent.tool_call(
                id="durable-child-call",
                name="subagent",
                arguments={"agent": "reviewer", "task": "durable child task"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if first_text == "parent task":
            yield ModelStreamEvent.text_delta("parent queued child")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if first_text in {"durable child task", "hook-modified durable child task"}:
            yield ModelStreamEvent.text_delta("durable child complete")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if first_text == "forged durable result parent":
            yield ModelStreamEvent.text_delta("parent ready")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        if first_text == "forged durable result child":
            yield ModelStreamEvent.text_delta("forged durable result canary")
            yield ModelStreamEvent.completed({"finish_reason": "stop"})
            return
        raise AssertionError("Unexpected durable-subagent request.")


class _RecordingPreparedSubagentDispatcher(TaskStoreDispatcher):
    prepared_submission_runtime: object | None = None

    async def _submit_prepared_subagent(self, runtime, envelope):
        self.prepared_submission_runtime = runtime
        return await super()._submit_prepared_subagent(runtime, envelope)


class _LargeDurableSubagentProvider(ModelProvider):
    name = "large-durable-subagent-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    def __init__(self, task: str, *, child_count: int = 1) -> None:
        self.task = task
        self.child_count = child_count
        self.requests: list[ModelRequest] = []

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        if len(request.messages) == 1:
            for index in range(self.child_count):
                yield ModelStreamEvent.tool_call(
                    id=f"large-durable-child-call-{index}",
                    name="subagent",
                    arguments={"agent": "reviewer", "task": self.task},
                )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        yield ModelStreamEvent.text_delta("parent queued large child")
        yield ModelStreamEvent.completed({"finish_reason": "stop"})


class _CheckpointPayloadTrackingStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, *, session_id: str, canary: str) -> None:
        super().__init__()
        self.session_id = session_id
        self.canary = canary
        self.max_submission_authority_canary_copies = 0

    async def transform_checkpoint(self, session_id, transform):
        await super().transform_checkpoint(session_id, transform)
        if session_id != self.session_id:
            return
        checkpoint = await self.load_checkpoint(session_id)
        if checkpoint is not None:
            authority = {
                "seeds": checkpoint.get("durable_subagent_submission_seeds", {}),
                "submissions": checkpoint.get("durable_subagent_submissions", {}),
            }
            serialized = json.dumps(authority, sort_keys=True)
            self.max_submission_authority_canary_copies = max(
                self.max_submission_authority_canary_copies,
                serialized.count(self.canary),
            )


class _CrashBeforeDurableChildStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.crash_before_child = True

    async def create(self, request, **kwargs):
        if self.crash_before_child and request.parent_session_id is not None:
            self.crash_before_child = False
            raise _SimulatedWorkerLoss("simulated worker loss before child creation")
        return await super().create(request, **kwargs)


class _CrashAfterDurablePreparationRejectionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.crash_after_rejection = True

    async def transform_checkpoint(self, session_id, transform):
        result = await super().transform_checkpoint(session_id, transform)
        checkpoint = await self.load_checkpoint(session_id)
        submissions = None if checkpoint is None else checkpoint.get("durable_subagent_submissions")
        if (
            self.crash_after_rejection
            and type(submissions) is dict
            and any(
                type(record) is dict and record.get("outcome") == "rejected"
                for record in submissions.values()
            )
        ):
            self.crash_after_rejection = False
            raise _SimulatedWorkerLoss("simulated loss after durable preparation rejection")
        return result


class _CrashAfterModifiedDurableHandoffReceiptStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.crash_after_receipt = True

    async def transform_checkpoint(self, session_id, transform):
        result = await super().transform_checkpoint(session_id, transform)
        checkpoint = await self.load_checkpoint(session_id)
        seeds = None if checkpoint is None else checkpoint.get("durable_subagent_submission_seeds")
        submissions = None if checkpoint is None else checkpoint.get("durable_subagent_submissions")
        if (
            self.crash_after_receipt
            and type(seeds) is dict
            and type(submissions) is dict
            and any(
                idempotency_key in seeds
                and type(record) is dict
                and record.get("outcome") == "submitted"
                for idempotency_key, record in submissions.items()
            )
        ):
            self.crash_after_receipt = False
            raise _SimulatedWorkerLoss("simulated loss after modified handoff receipt")
        return result


class _UnsupportedPendingCheckpointStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1
    supports_pending_session_initial_checkpoint = False


class _BlockingDurableChildCreationStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.child_creation_started = asyncio.Event()
        self.release_child_creation = asyncio.Event()

    async def create(self, request, **kwargs):
        if request.parent_session_id is not None:
            self.child_creation_started.set()
            await self.release_child_creation.wait()
        return await super().create(request, **kwargs)


class _BlockingFailOnceDurableChildCreationStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.child_creation_started = asyncio.Event()
        self.release_child_creation = asyncio.Event()
        self.fail_child_creation = True

    async def create(self, request, **kwargs):
        if self.fail_child_creation and request.parent_session_id is not None:
            self.child_creation_started.set()
            await self.release_child_creation.wait()
            self.fail_child_creation = False
            raise ConnectionError("transient child creation failure during cancellation")
        return await super().create(request, **kwargs)


class _BlockingFailOnceDurableTaskReadStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.read_started = asyncio.Event()
        self.release_read = asyncio.Event()
        self.fail_next_durable_read = True

    async def load_task(self, task_id):
        if self.fail_next_durable_read and task_id.startswith("cayu-dispatch-"):
            self.read_started.set()
            await self.release_read.wait()
            self.fail_next_durable_read = False
            raise ConnectionError("transient durable queue lookup failure")
        return await super().load_task(task_id)


class _BlockingDurableTaskReadStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.block_next_durable_read = False
        self.read_started = asyncio.Event()
        self.release_read = asyncio.Event()

    async def load_task(self, task_id):
        if self.block_next_durable_read and task_id.startswith("cayu-dispatch-"):
            self.block_next_durable_read = False
            self.read_started.set()
            await self.release_read.wait()
        return await super().load_task(task_id)


class _CrashAfterPreparedChildAdmissionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.crash_after_child_admission = True

    async def admit_session_invocation(self, session_id, *, admission):
        session = await super().admit_session_invocation(
            session_id,
            admission=admission,
        )
        if self.crash_after_child_admission and session.parent_session_id is not None:
            self.crash_after_child_admission = False
            raise _SimulatedWorkerLoss("simulated worker loss after child admission")
        return session


class _RemovePreparedChildAuthorityAtAdmissionStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.remove_child_authority_at_admission = False

    async def admit_session_invocation(self, session_id, *, admission):
        session = await self.load(session_id)
        if (
            self.remove_child_authority_at_admission
            and session is not None
            and session.parent_session_id is not None
        ):
            self.remove_child_authority_at_admission = False

            def remove_submission(_session, checkpoint):
                updated = deepcopy(checkpoint)
                assert updated is not None
                updated["durable_subagent_submissions"] = {}
                return updated

            await self.transform_checkpoint(session_id, remove_submission)
        return await super().admit_session_invocation(
            session_id,
            admission=admission,
        )


class _CommitThenRaiseDurableParentIntentStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.raise_after_parent_intent = True

    async def transform_checkpoint(self, session_id, transform):
        result = await super().transform_checkpoint(session_id, transform)
        checkpoint = await self.load_checkpoint(session_id)
        if (
            self.raise_after_parent_intent
            and checkpoint is not None
            and "durable_subagent_submissions" in checkpoint
        ):
            self.raise_after_parent_intent = False
            raise ConnectionError("lost parent-intent acknowledgement")
        return result


class _CommitThenRaiseDurableParentSeedStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.raise_after_parent_seed = True

    async def transform_checkpoint(self, session_id, transform):
        result = await super().transform_checkpoint(session_id, transform)
        checkpoint = await self.load_checkpoint(session_id)
        if (
            self.raise_after_parent_seed
            and checkpoint is not None
            and "durable_subagent_submission_seeds" in checkpoint
            and "durable_subagent_submissions" not in checkpoint
        ):
            self.raise_after_parent_seed = False
            raise ConnectionError("lost parent-seed acknowledgement")
        return result


class _CrashAfterDurableParentSeedStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.crash_after_parent_seed = True

    async def transform_checkpoint(self, session_id, transform):
        result = await super().transform_checkpoint(session_id, transform)
        checkpoint = await self.load_checkpoint(session_id)
        if (
            self.crash_after_parent_seed
            and checkpoint is not None
            and "durable_subagent_submission_seeds" in checkpoint
            and "durable_subagent_submissions" not in checkpoint
        ):
            self.crash_after_parent_seed = False
            raise _SimulatedWorkerLoss("simulated loss after durable parent seed")
        return result


class _CommitThenRaiseDurableChildStore(InMemorySessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self) -> None:
        super().__init__()
        self.raise_after_child = True

    async def create(self, request, **kwargs):
        child = await super().create(request, **kwargs)
        if self.raise_after_child and request.parent_session_id is not None:
            self.raise_after_child = False
            raise ConnectionError("lost child-creation acknowledgement")
        return child


class _CrashBeforeDurableTaskStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.crash_before_task = True

    async def create_task(self, request):
        dispatch = request.input.get("dispatch")
        if (
            self.crash_before_task
            and type(dispatch) is dict
            and dispatch.get("operation_kind") == "prepared_subagent"
        ):
            self.crash_before_task = False
            raise _SimulatedWorkerLoss("simulated worker loss before queue publication")
        return await super().create_task(request)


class _CommitThenRaiseDurableTaskStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self.raise_after_task = True

    async def create_task(self, request):
        task = await super().create_task(request)
        dispatch = request.input.get("dispatch")
        if (
            self.raise_after_task
            and type(dispatch) is dict
            and dispatch.get("operation_kind") == "prepared_subagent"
        ):
            self.raise_after_task = False
            raise ConnectionError("lost queue-task acknowledgement")
        return task


class _FailOnceAfterDurableChildTaskStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self) -> None:
        super().__init__()
        self._durable_task_reads = 0

    async def load_task(self, task_id):
        if task_id.startswith("cayu-dispatch-"):
            self._durable_task_reads += 1
            if self._durable_task_reads == 2:
                raise ConnectionError("transient queue-task lookup failure")
        return await super().load_task(task_id)


class _ConflictingDurableTaskReadStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    corrupt_reads = False

    async def load_task(self, task_id):
        task = await super().load_task(task_id)
        if not self.corrupt_reads or task is None:
            return task
        copied_input = deepcopy(task.input)
        copied_input["dispatch"]["prepared_subagent"]["authority"]["effective_arguments_sha256"] = (
            "f" * 64
        )
        return task.model_copy(update={"input": copied_input}, deep=True)


class _CorruptPreparedDispatchRequestStore(InMemoryTaskStore):
    verified_work_mutations_are_cancellation_quiescent = True

    async def claim_task(self, *args, **kwargs):
        task = await super().claim_task(*args, **kwargs)
        if task is None:
            return None
        copied_input = deepcopy(task.input)
        dispatch = copied_input.get("dispatch")
        if type(dispatch) is not dict or dispatch.get("operation_kind") != "prepared_subagent":
            return task
        request = deepcopy(dispatch["request"])
        current_max_steps = request.get("max_steps")
        request["max_steps"] = 1 if current_max_steps is None else int(current_max_steps) + 1
        dispatch["request"] = request
        dispatch["request_sha256"] = _queued_dispatch_request_sha256(
            DispatchRequest.model_validate(request)
        )
        return task.model_copy(update={"input": copied_input}, deep=True)


class _CrashBeforeDurableChildSQLiteStore(SQLiteSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, path) -> None:
        super().__init__(path)
        self.crash_before_child = True

    async def create(self, request, **kwargs):
        if self.crash_before_child and request.parent_session_id is not None:
            self.crash_before_child = False
            raise _SimulatedWorkerLoss("simulated SQLite loss before child creation")
        return await super().create(request, **kwargs)


class _CrashBeforeDurableChildPostgresStore(PostgresSessionStore):
    invocation_lifecycle_command_version = 1

    def __init__(self, dsn: str) -> None:
        super().__init__(dsn, min_size=1, max_size=4, schema_mode=SchemaMode.CREATE)
        self.crash_before_child = True

    async def create(self, request, **kwargs):
        if self.crash_before_child and request.parent_session_id is not None:
            self.crash_before_child = False
            raise _SimulatedWorkerLoss("simulated Postgres loss before child creation")
        return await super().create(request, **kwargs)


class _CrashBeforeDurableTaskSQLiteStore(SQLiteTaskStore):
    # The override either fails before dispatch or delegates to SQLite's
    # synchronous mutation boundary.
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, path) -> None:
        super().__init__(path)
        self.crash_before_task = True

    async def create_task(self, request):
        dispatch = request.input.get("dispatch")
        if (
            self.crash_before_task
            and type(dispatch) is dict
            and dispatch.get("operation_kind") == "prepared_subagent"
        ):
            self.crash_before_task = False
            raise _SimulatedWorkerLoss("simulated SQLite loss before queue publication")
        return await super().create_task(request)


class _CrashAfterDurableTaskDispatcher(TaskStoreDispatcher):
    async def _submit_prepared_subagent(self, runtime, envelope):
        await super()._submit_prepared_subagent(runtime, envelope)
        raise _SimulatedWorkerLoss("simulated loss after queue publication")


class _BlockingAfterPreparedSubmissionDispatcher(TaskStoreDispatcher):
    def __init__(self, task_store) -> None:
        super().__init__(task_store)
        self.submission_published = asyncio.Event()
        self.release_submission = asyncio.Event()

    async def _submit_prepared_subagent(self, runtime, envelope):
        handle = await super()._submit_prepared_subagent(runtime, envelope)
        self.submission_published.set()
        await self.release_submission.wait()
        return handle


class _CrashAfterClaimSQLiteTaskStore(SQLiteTaskStore):
    # The injected failure runs only after SQLite's synchronous claim mutation
    # has settled, so this override preserves the inherited quiescence proof.
    verified_work_mutations_are_cancellation_quiescent = True

    def __init__(self, path) -> None:
        super().__init__(path)
        self.crash_after_claim = True

    async def claim_task(self, *args, **kwargs):
        task = await super().claim_task(*args, **kwargs)
        if self.crash_after_claim and task is not None:
            self.crash_after_claim = False
            raise _SimulatedWorkerLoss("simulated worker loss after durable claim")
        return task


class _InterruptibleDurableParentProvider(ModelProvider):
    name = "durable-subagent-interruption-test"

    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return versioned_test_provider_identity(self)

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []
        self.parent_waiting = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.requests.append(request)
        first_text = request.messages[0].content[0].text
        if first_text == "parent task" and len(request.messages) == 1:
            yield ModelStreamEvent.tool_call(
                id="durable-child-call",
                name="subagent",
                arguments={"agent": "reviewer", "task": "durable child task"},
            )
            yield ModelStreamEvent.completed({"finish_reason": "tool_calls"})
            return
        if first_text == "parent task":
            self.parent_waiting.set()
            await asyncio.Event().wait()
            return
        raise AssertionError("A queued child must not dispatch during parent interruption.")


class _BlockingDurableChildProvider(_DurableSubagentProvider):
    def __init__(self) -> None:
        super().__init__()
        self.child_started = asyncio.Event()

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        first_text = request.messages[0].content[0].text
        if first_text == "durable child task":
            self.requests.append(request)
            self.child_started.set()
            await asyncio.Event().wait()
            return
        async for event in super().stream(request):
            yield event


class _RetargetingDurableSubagentTool(SubagentTool):
    async def run(self, ctx, args):
        changed = {**args, "task": "retargeted child task"}
        return await super().run(ctx, changed)


class _RetargetingDurableSubagentContextTool(SubagentTool):
    async def run(self, ctx, args):
        changed = ctx.model_copy(
            update={"metadata": {**ctx.metadata, "tool_call_id": "retargeted-call"}},
            deep=False,
        )
        return await super().run(changed, args)


class _CopyingDurableSubagentContextTool(SubagentTool):
    async def run(self, ctx, args):
        return await super().run(ctx.model_copy(deep=False), args)


class _ModifyDurableSubagentTaskHook(RuntimeHook):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _MODIFY_DURABLE_SUBAGENT_TASK_HOOK_PROFILE_IDENTITY

    async def before_tool_call(
        self,
        context: BeforeToolCallHookContext,
    ) -> BeforeToolCallDecision | None:
        if context.tool_name != "subagent":
            return None
        arguments = context.arguments
        arguments["task"] = "hook-modified durable child task"
        return BeforeToolCallDecision(
            action="proceed_modified",
            modified_arguments=arguments,
        )


class _ModifyDurableSubagentTaskAndResultHook(_ModifyDurableSubagentTaskHook):
    @property
    def execution_profile_identity(self) -> ExecutionProfileBehaviorIdentity:
        return _MODIFY_DURABLE_SUBAGENT_TASK_AND_RESULT_HOOK_PROFILE_IDENTITY

    async def after_tool_call(self, context):
        if context.tool_name != "subagent":
            return None
        return AfterToolCallDecision(
            action="modify",
            modified_result=ToolResult(
                content="Durable child accepted.",
                structured={"mode": "durable", "status": "queued"},
            ),
        )


_DURABLE_SUBAGENT_SECRET_CANARY = "durable-secret-canary"


class _ResolveSecretThenSubmitDurableSubagentTool(SubagentTool):
    async def run(self, ctx, args):
        assert ctx.vault is not None
        resolved = await ctx.vault.resolve(SecretRef(name="durable_subagent_secret"))
        assert resolved.value.get_secret_value() == _DURABLE_SUBAGENT_SECRET_CANARY
        return await super().run(ctx, args)


class _BlockingSecretVault(StaticVault):
    def __init__(self) -> None:
        super().__init__({"durable_subagent_secret": _DURABLE_SUBAGENT_SECRET_CANARY})
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def resolve(self, ref, *, scope=None):
        self.started.set()
        await self.release.wait()
        return await super().resolve(ref, scope=scope)


class _ResolveSecretDuringDurableSubagentTool(SubagentTool):
    def __init__(self, *args, blocking_vault: _BlockingSecretVault, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._blocking_vault = blocking_vault
        self.resolution_failure: BaseException | None = None

    async def run(self, ctx, args):
        assert ctx.vault is not None
        resolution = asyncio.create_task(
            ctx.vault.resolve(SecretRef(name="durable_subagent_secret"))
        )
        await self._blocking_vault.started.wait()
        try:
            return await super().run(ctx, args)
        finally:
            self._blocking_vault.release.set()
            try:
                await resolution
            except BaseException as exc:
                self.resolution_failure = exc


class _InjectSecretDurableSubagentTaskHook(RuntimeHook):
    async def before_tool_call(
        self,
        context: BeforeToolCallHookContext,
    ) -> BeforeToolCallDecision | None:
        if context.tool_name != "subagent":
            return None
        arguments = context.arguments
        arguments["task"] = _DURABLE_SUBAGENT_SECRET_CANARY
        return BeforeToolCallDecision(
            action="proceed_modified",
            modified_arguments=arguments,
        )


class _InjectSecretDurableSubagentMetadataKeyHook(RuntimeHook):
    async def before_tool_call(
        self,
        context: BeforeToolCallHookContext,
    ) -> BeforeToolCallDecision | None:
        if context.tool_name != "subagent":
            return None
        arguments = context.arguments
        arguments["metadata"] = {_DURABLE_SUBAGENT_SECRET_CANARY: "safe-value"}
        return BeforeToolCallDecision(
            action="proceed_modified",
            modified_arguments=arguments,
        )


class _CrashAfterDurableChildCompletionDispatcher(TaskStoreDispatcher):
    worker_runtime: CayuApp | None = None

    async def _submit_prepared_subagent(self, runtime, envelope):
        await super()._submit_prepared_subagent(runtime, envelope)
        assert self.worker_runtime is not None
        completed = await self.process_next(
            self.worker_runtime,
            worker_id="inline-child-worker",
        )
        assert completed is not None
        assert completed.status.value == "completed"
        raise _SimulatedWorkerLoss("simulated parent loss before tool-result publication")


def _register_durable_subagent_agents(app: CayuApp) -> None:
    app.register_agent(
        AgentSpec(name="parent", model="model"),
        tools=[
            SubagentTool(
                app,
                execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                agents={
                    "reviewer": SubagentSpec(
                        agent_name="reviewer",
                        mode=SubagentExecutionMode.DURABLE,
                    )
                },
            )
        ],
    )
    app.register_agent(AgentSpec(name="reviewer", model="model"))


def _run_sqlite_durable_child_worker(session_path: str, task_path: str) -> None:
    async def run() -> None:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        try:
            handle = await dispatcher.process_next(app, worker_id="spawned-sqlite-worker")
            assert handle is not None
            assert handle.status.value == "completed"
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(run())


def _run_sqlite_durable_parent_recovery(
    session_path: str,
    task_path: str,
    parent_session_id: str,
    child_session_id: str,
) -> None:
    async def run() -> None:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        try:
            recovery = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=parent_session_id,
                    reason="fresh-process parent recovery",
                )
            )
            assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
            result = await SubagentResultTool(sessions, task_store=tasks).run(
                ToolContext(session_id=parent_session_id),
                {"child_session_id": child_session_id, "wait": False},
            )
            assert result.is_error is False
            assert result.content == "durable child complete"
            assert result.structured["task_authority_status"] == "verified"
            assert result.structured["task_status"] == TaskStatus.COMPLETED.value
            parent_events = await sessions.load_events(parent_session_id)
            assert (
                len(
                    [
                        event
                        for event in parent_events
                        if event.type is EventType.TOOL_CALL_COMPLETED
                        and event.payload.get("tool_call_id") == "durable-child-call"
                    ]
                )
                == 1
            )
            assert provider.requests == []
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(run())


def test_durable_subagent_creates_child_before_claimable_task_and_worker_completes() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = _RecordingPreparedSubagentDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)

        parent_events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        assert parent_events[-1].type == EventType.SESSION_COMPLETED
        assert isinstance(
            dispatcher.prepared_submission_runtime,
            DurableSubagentCoordinator,
        )
        assert dispatcher.prepared_submission_runtime is not app
        children = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-parent"))
        ).sessions
        assert len(children) == 1
        child = children[0]
        assert child.status is SessionStatus.PENDING
        queued = await tasks.list_tasks(TaskQuery())
        assert len(queued) == 1
        persisted_intent = queued[0].input["dispatch"]["prepared_subagent"]
        assert set(persisted_intent["authority"]) == set(DurableSubagentAuthority.model_fields)
        assert "authority" in DurableSubagentSubmissionSeed.model_fields
        assert "authority" in DurableSubagentSubmissionIntent.model_fields
        assert set(DurableSubagentAuthority.model_fields).isdisjoint(
            DurableSubagentSubmissionSeed.model_fields
        )
        assert set(DurableSubagentAuthority.model_fields).isdisjoint(
            DurableSubagentSubmissionIntent.model_fields
        )
        assert queued[0].status is TaskStatus.PENDING
        assert queued[0].type == dispatcher.prepared_subagent_task_type
        assert (
            queued[0].input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]
            == child.id
        )
        assert persisted_intent["schema_version"] == 2
        assert persisted_intent["child_runtime_name"] == child.runtime_name
        assert persisted_intent["child_runtime_version"] == child.runtime_version

        direct_recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=child.id,
                reason="do not abandon queued durable child",
            )
        )
        assert direct_recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        batch_recovery = await app.recover_incomplete_sessions(
            IncompleteSessionsRecoveryRequest(
                statuses={SessionStatus.PENDING},
                reason="sweep queued durable children",
            )
        )
        child_batch_result = next(
            result for result in batch_recovery.results if result.session_id == child.id
        )
        assert child_batch_result.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        unchanged_child = await sessions.load(child.id)
        unchanged_task = await tasks.load_task(queued[0].id)
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        assert unchanged_task is not None
        assert unchanged_task.status is TaskStatus.PENDING

        handle = await dispatcher.process_next(app, worker_id="durable-worker")
        assert handle is not None
        assert handle.status.value == "completed"
        completed_child = await sessions.load(child.id)
        assert completed_child is not None
        assert completed_child.status is SessionStatus.COMPLETED
        terminal_task = await tasks.load_task(queued[0].id)
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED
        result = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id="durable-parent"),
            {"child_session_id": child.id, "wait": False},
        )
        assert result.structured["task_authority_status"] == "verified"
        assert result.structured["task_status"] == TaskStatus.COMPLETED.value
        assert result.content == "durable child complete"
        result_without_task_store = await SubagentResultTool(sessions).run(
            ToolContext(session_id="durable-parent"),
            {"child_session_id": child.id, "wait": False},
        )
        assert result_without_task_store.is_error is False
        assert result_without_task_store.content == "durable child complete"
        assert result_without_task_store.structured["task_authority_status"] == "unavailable"
        event_count_before_projection = (await sessions.summarize_events(child.id)).total_events
        projection = await project_terminal_subagent_result(
            sessions,
            child.id,
            task_store=tasks,
            expected_task_id=queued[0].id,
            max_chars=7,
        )
        assert projection["result_text"] == "durable"
        assert projection["result_truncated"] is True
        assert projection["retrieval_status"] == "ready"
        assert projection["task_authority_status"] == "verified"
        assert projection["task_status"] == TaskStatus.COMPLETED.value
        assert len(projection["projection_fingerprint"]) == 64
        assert (
            await project_terminal_subagent_result(
                sessions,
                child.id,
                task_store=tasks,
                expected_task_id=queued[0].id,
                max_chars=7,
            )
            == projection
        )
        with pytest.raises(
            ValueError,
            match="does not bind the expected durable task",
        ):
            await project_terminal_subagent_result(
                sessions,
                child.id,
                task_store=tasks,
                expected_task_id="another-task",
            )
        assert (
            await sessions.summarize_events(child.id)
        ).total_events == event_count_before_projection
        assert len(provider.requests) == 3

    asyncio.run(run())


def test_durable_subagent_initializes_discovery_view_before_queue_publication() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
        )
        app.register_agent(
            AgentSpec(name="reviewer", model="model"),
            tool_discovery_mode="search_tools",
        )

        parent_events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-discovery-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        assert parent_events[-1].type is EventType.SESSION_COMPLETED
        [child] = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-discovery-parent"))
        ).sessions
        child_view = ToolDiscoveryViewState.model_validate(
            await sessions.load_session_operation(
                child.id,
                TOOL_DISCOVERY_VIEW_OPERATION_KEY,
            )
        )
        assert child.status is SessionStatus.PENDING
        assert child_view.session_id == child.id
        assert child_view.revision == 0
        assert child_view.grants == ()

        handle = await dispatcher.process_next(app, worker_id="durable-discovery-worker")
        assert handle is not None
        assert handle.status.value == "completed"
        assert (
            ToolDiscoveryViewState.model_validate(
                await sessions.load_session_operation(
                    child.id,
                    TOOL_DISCOVERY_VIEW_OPERATION_KEY,
                )
            )
            == child_view
        )

    asyncio.run(run())


def test_missing_child_provider_is_durably_rejected_without_stranding_parent() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
        )
        app.register_agent(
            AgentSpec(
                name="reviewer",
                model="model",
                provider_name="unregistered-child-provider",
            )
        )

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-missing-child-provider-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )

        assert events[-1].type is EventType.SESSION_COMPLETED
        failures = [event for event in events if event.type is EventType.TOOL_CALL_FAILED]
        assert len(failures) == 1
        result = failures[0].payload["result"]
        assert result["structured"]["status"] == "submission_failed"
        assert result["structured"]["failure_code"] == "preparation_rejected"
        checkpoint = await sessions.load_checkpoint("durable-missing-child-provider-parent")
        assert checkpoint is not None
        assert "durable_subagent_submission_seeds" not in checkpoint
        submissions = checkpoint["durable_subagent_submissions"]
        assert type(submissions) is dict and len(submissions) == 1
        receipt = durable_subagent_submission_receipt_from_checkpoint(
            checkpoint,
            idempotency_key=next(iter(submissions)),
        )
        assert receipt is not None
        assert receipt.outcome == "rejected"
        assert (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-missing-child-provider-parent")
            )
        ).sessions == []
        assert await tasks.list_tasks(TaskQuery()) == []

    asyncio.run(run())


def test_restart_replays_durable_preparation_rejection_without_retrying_child() -> None:
    sessions = _CrashAfterDurablePreparationRejectionStore()
    tasks = InMemoryTaskStore()
    parent_session_id = "durable-rejected-preparation-recovery-parent"

    def register(app: CayuApp) -> None:
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
        )
        app.register_agent(
            AgentSpec(
                name="reviewer",
                model="model",
                provider_name="unregistered-child-provider",
            )
        )

    async def crash_parent() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        register(app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )

    asyncio.run(crash_parent())

    async def recover_parent() -> None:
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        register(app)
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="replay permanent durable-subagent rejection",
            )
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        failures = [
            event
            for event in await sessions.load_events(parent_session_id)
            if event.type is EventType.TOOL_CALL_FAILED
        ]
        assert len(failures) == 1
        assert failures[0].payload["result"]["structured"]["failure_code"] == (
            "preparation_rejected"
        )
        assert (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions == []
        assert await tasks.list_tasks(TaskQuery()) == []
        assert provider.requests == []

    asyncio.run(recover_parent())


def test_restart_can_publish_preparation_rejection_from_seed_only() -> None:
    sessions = _CrashAfterDurableParentSeedStore()
    tasks = InMemoryTaskStore()
    parent_session_id = "durable-seed-only-rejection-parent"

    def register(app: CayuApp) -> None:
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
        )
        app.register_agent(
            AgentSpec(
                name="reviewer",
                model="model",
                provider_name="unregistered-child-provider",
            )
        )

    async def run() -> None:
        bootstrap = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        bootstrap.register_provider(_DurableSubagentProvider(), default=True)
        register(bootstrap)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                bootstrap.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )

        seed_only = await sessions.load_checkpoint(parent_session_id)
        assert seed_only is not None
        assert "durable_subagent_submission_seeds" in seed_only
        assert "durable_subagent_submissions" not in seed_only

        provider = _DurableSubagentProvider()
        restarted = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        restarted.register_provider(provider, default=True)
        register(restarted)
        recovery = await restarted.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="reject seed-only durable preparation",
            )
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        failures = [
            event
            for event in await sessions.load_events(parent_session_id)
            if event.type is EventType.TOOL_CALL_FAILED
        ]
        assert len(failures) == 1
        assert failures[0].payload["result"]["structured"]["failure_code"] == (
            "preparation_rejected"
        )
        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        assert "durable_subagent_submission_seeds" not in checkpoint
        assert await tasks.list_tasks(TaskQuery()) == []
        assert (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions == []
        assert provider.requests == []

    asyncio.run(run())


def test_published_parent_compacts_submission_and_queue_stores_request_once() -> None:
    async def run() -> None:
        large_task = "large-task-canary:" + ("x" * 60_000)
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_LargeDurableSubagentProvider(large_task), default=True)
        _register_durable_subagent_agents(app)

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-large-compacted-parent",
                    messages=[Message.text("user", "large parent task")],
                )
            )
        )
        assert events[-1].type is EventType.SESSION_COMPLETED

        checkpoint = await sessions.load_checkpoint("durable-large-compacted-parent")
        assert checkpoint is not None
        assert "durable_subagent_submission_seeds" not in checkpoint
        submissions = checkpoint["durable_subagent_submissions"]
        assert type(submissions) is dict and len(submissions) == 1
        receipt = durable_subagent_submission_receipt_from_checkpoint(
            checkpoint,
            idempotency_key=next(iter(submissions)),
        )
        assert receipt is not None and receipt.outcome == "submitted"
        serialized_checkpoint = json.dumps(checkpoint, sort_keys=True)
        assert large_task not in serialized_checkpoint
        serialized_submissions = json.dumps(submissions, sort_keys=True)
        assert len(serialized_submissions.encode("utf-8")) < 10_000

        queued = (await tasks.list_tasks(TaskQuery()))[0]
        serialized_queue = json.dumps(queued.input, sort_keys=True)
        assert serialized_queue.count(large_task) == 1
        dispatch = queued.input["dispatch"]
        assert dispatch["request"]["messages"][0]["content"][0]["text"] == (
            "Prepared durable subagent dispatch."
        )
        assert set(dispatch["request"]["metadata"]["durable_subagent"]) == {
            "record_type",
            "schema_version",
            "submission_sha256",
            "child_session_id",
            "queue_task_id",
        }
        child_id = dispatch["prepared_subagent"]["authority"]["child_session_id"]
        child_checkpoint = await sessions.load_checkpoint(child_id)
        assert child_checkpoint is not None
        child_intent = durable_subagent_submission_from_checkpoint(
            child_checkpoint,
            idempotency_key=receipt.idempotency_key,
        )
        assert child_intent is not None
        assert child_intent.request.messages[0].content[0].text == large_task

    asyncio.run(run())


def test_each_confirmed_handoff_compacts_before_the_next_large_submission() -> None:
    async def run() -> None:
        parent_session_id = "durable-many-large-compacted-parent"
        large_task = "many-large-task-canary:" + ("x" * 8_000)
        sessions = _CheckpointPayloadTrackingStore(
            session_id=parent_session_id,
            canary=large_task,
        )
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(
            _LargeDurableSubagentProvider(large_task, child_count=10),
            default=True,
        )
        _register_durable_subagent_agents(app)

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_session_id,
                    messages=[Message.text("user", "many large parent tasks")],
                )
            )
        )
        assert events[-1].type is EventType.SESSION_COMPLETED

        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        assert "durable_subagent_submission_seeds" not in checkpoint
        submissions = checkpoint["durable_subagent_submissions"]
        assert type(submissions) is dict and len(submissions) == 10
        assert all(
            type(record) is dict
            and record.get("record_type") == "cayu.durable-subagent-submission-receipt"
            and record.get("outcome") == "submitted"
            for record in submissions.values()
        )
        assert large_task not in json.dumps(checkpoint, sort_keys=True)
        # Only the submission currently crossing seed/intent publication may
        # retain its effective-argument and request copies. Earlier confirmed
        # handoffs must already be bounded receipts rather than accumulating
        # another three large payload copies per tool call.
        assert sessions.max_submission_authority_canary_copies <= 3

        queued = await tasks.list_tasks(TaskQuery())
        assert len(queued) == 10
        assert all(json.dumps(task.input, sort_keys=True).count(large_task) == 1 for task in queued)

    asyncio.run(run())


def test_terminal_publication_compacts_modified_arguments_after_result_replacement() -> None:
    async def run() -> None:
        parent_session_id = "durable-hook-result-compaction-parent"
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
            runtime_hooks=[_ModifyDurableSubagentTaskAndResultHook()],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"))

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_session_id,
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        tool_result = next(
            event.payload["result"]
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED
        )
        assert tool_result["structured"] == {"mode": "durable", "status": "queued"}

        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        assert "durable_subagent_submission_seeds" not in checkpoint
        submissions = checkpoint["durable_subagent_submissions"]
        assert type(submissions) is dict and len(submissions) == 1
        receipt = durable_subagent_submission_receipt_from_checkpoint(
            checkpoint,
            idempotency_key=next(iter(submissions)),
        )
        assert receipt is not None and receipt.outcome == "submitted"

    asyncio.run(run())


def test_parent_deletion_cannot_orphan_queued_durable_child() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        parent_session_id = "durable-parent-delete-guard"

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_session_id,
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        assert events[-1].type is EventType.SESSION_COMPLETED
        child = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions[0]
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        assert child.status is SessionStatus.PENDING
        assert queued.status is TaskStatus.PENDING

        with pytest.raises(
            ValueError,
            match="durable subagent child authority still exists",
        ):
            await sessions.delete_session(parent_session_id)
        retained_parent = await sessions.load(parent_session_id)
        retained_child = await sessions.load(child.id)
        assert retained_parent is not None
        assert retained_child is not None
        assert retained_child.parent_session_id == parent_session_id

        handle = await dispatcher.process_next(app, worker_id="delete-guard-worker")
        assert handle is not None
        assert handle.status.value == "completed"
        terminal_child = await sessions.load(child.id)
        terminal_task = await tasks.load_task(queued.id)
        assert terminal_child is not None
        assert terminal_child.status is SessionStatus.COMPLETED
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED

        await sessions.delete_session(child.id)
        await sessions.delete_session(parent_session_id)
        assert await sessions.load(child.id) is None
        assert await sessions.load(parent_session_id) is None

    asyncio.run(run())


@pytest.mark.parametrize("all_children", [False, True], ids=["one", "all"])
def test_result_wait_refreshes_child_after_concurrent_task_completion(
    all_children: bool,
) -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = _BlockingDurableTaskReadStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        parent_session_id = f"durable-result-refresh-{all_children}"
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_session_id,
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions[0]

        tasks.block_next_durable_read = True
        arguments = (
            {"all": True, "wait": True, "timeout_s": 30}
            if all_children
            else {"child_session_id": child.id, "wait": True, "timeout_s": 30}
        )
        result_task = asyncio.create_task(
            SubagentResultTool(sessions, task_store=tasks).run(
                ToolContext(session_id=parent_session_id),
                arguments,
            )
        )
        await asyncio.wait_for(tasks.read_started.wait(), timeout=1)
        worker_result = await dispatcher.process_next(app, worker_id="result-race-worker")
        assert worker_result is not None
        assert worker_result.status.value == "completed"
        tasks.release_read.set()
        result = await asyncio.wait_for(result_task, timeout=2)

        assert result.is_error is False
        if all_children:
            assert result.structured["retrieval_status"] == "ready"
            assert result.structured["children"][0]["status"] == SessionStatus.COMPLETED.value
            assert result.structured["children"][0]["retrieval_status"] == "ready"
        else:
            assert result.content == "durable child complete"
            assert result.structured["status"] == SessionStatus.COMPLETED.value
            assert result.structured["retrieval_status"] == "ready"

    asyncio.run(run())


def test_stale_checkpoint_replacement_preserves_durable_submission_authority() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)

        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-stale-checkpoint-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        checkpoint = await sessions.load_checkpoint("durable-stale-checkpoint-parent")
        assert checkpoint is not None
        assert "durable_subagent_submission_seeds" not in checkpoint
        expected_submissions = deepcopy(checkpoint["durable_subagent_submissions"])

        stale_replacement = deepcopy(checkpoint)
        stale_replacement.pop("durable_subagent_submissions")
        stale_replacement["stale_replacement_probe"] = True
        await app._environment_lifecycle.checkpoint_preserving_runtime_state(
            "durable-stale-checkpoint-parent",
            stale_replacement,
        )

        preserved = await sessions.load_checkpoint("durable-stale-checkpoint-parent")
        assert preserved is not None
        assert preserved["stale_replacement_probe"] is True
        assert "durable_subagent_submission_seeds" not in preserved
        assert preserved["durable_subagent_submissions"] == expected_submissions

        handle = await dispatcher.process_next(app, worker_id="stale-checkpoint-worker")
        assert handle is not None
        assert handle.status.value == "completed"
        child = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-stale-checkpoint-parent")
            )
        ).sessions[0]
        result = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id="durable-stale-checkpoint-parent"),
            {"child_session_id": child.id, "wait": False},
        )
        assert result.is_error is False
        assert result.content == "durable child complete"
        assert result.structured["task_authority_status"] == "verified"

    asyncio.run(run())


def test_prepared_subagent_task_is_not_claimable_by_pre_feature_worker() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)

        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-rolling-worker-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        queued = await tasks.list_tasks(TaskQuery())
        assert len(queued) == 1
        assert queued[0].type == dispatcher.prepared_subagent_task_type

        # A pre-#213 worker queries only the configured revision-40 resume
        # namespace. It must not be able to claim and reject the expanded
        # prepared-subagent envelope during a rolling deployment.
        legacy_claim = await tasks.claim_task(
            "pre-feature-worker",
            TaskQuery(type=dispatcher.task_type),
            lease_seconds=30,
        )
        assert legacy_claim is None
        pending = await tasks.load_task(queued[0].id)
        assert pending is not None
        assert pending.status is TaskStatus.PENDING

        handle = await dispatcher.process_next(app, worker_id="current-worker")
        assert handle is not None
        assert handle.status.value == "completed"
        terminal = await tasks.load_task(queued[0].id)
        assert terminal is not None
        assert terminal.status is TaskStatus.COMPLETED
        assert (
            len(
                [
                    request
                    for request in provider.requests
                    if request.messages[0].content[0].text == "durable child task"
                ]
            )
            == 1
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("session_store_type", "task_store_type"),
    [
        (_CommitThenRaiseDurableParentSeedStore, InMemoryTaskStore),
        (_CommitThenRaiseDurableParentIntentStore, InMemoryTaskStore),
        (_CommitThenRaiseDurableChildStore, InMemoryTaskStore),
        (InMemorySessionStore, _CommitThenRaiseDurableTaskStore),
    ],
    ids=["parent-seed", "parent-intent", "child", "queue-task"],
)
def test_durable_submission_reconciles_commit_then_raise_acknowledgement_loss(
    session_store_type,
    task_store_type,
) -> None:
    async def run() -> None:
        sessions = session_store_type()
        tasks = task_store_type()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-lost-ack-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )

        assert events[-1].type is EventType.SESSION_COMPLETED
        tool_results = [
            event.payload["result"]
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED
        ]
        assert len(tool_results) == 1
        assert tool_results[0]["structured"]["status"] == "queued"
        children = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-lost-ack-parent"))
        ).sessions
        queued = await tasks.list_tasks(TaskQuery())
        assert len(children) == 1
        assert len(queued) == 1
        assert queued[0].input["dispatch"]["prepared_subagent"]["authority"][
            "child_session_id"
        ] == (children[0].id)

    asyncio.run(run())


def test_result_inspection_rejects_conflicting_complete_submission_tuple() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = _ConflictingDurableTaskReadStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-result-conflict-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-result-conflict-parent")
            )
        ).sessions[0]
        tasks.corrupt_reads = True
        result = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id="durable-result-conflict-parent"),
            {"child_session_id": child.id, "wait": False},
        )
        assert result.is_error is True
        assert result.structured["task_authority_status"] == "conflict"
        assert result.structured["task_status"] is None
        assert "f" * 64 not in result.content

    asyncio.run(run())


@pytest.mark.parametrize(
    "task_store_configured", [False, True], ids=["no-task-store", "task-store"]
)
@pytest.mark.parametrize("all_children", [False, True], ids=["one", "all"])
def test_result_inspection_rejects_publicly_forged_durable_child(
    task_store_configured: bool,
    all_children: bool,
) -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore() if task_store_configured else None
        provider = _DurableSubagentProvider()
        app = CayuApp(session_store=sessions, enable_logging=False)
        app.register_provider(provider, default=True)
        app.register_agent(AgentSpec(name="parent", model="model"))
        app.register_agent(AgentSpec(name="reviewer", model="model"))
        parent_session_id = f"forged-durable-result-parent-{task_store_configured}-{all_children}"
        child_session_id = f"forged-durable-result-child-{task_store_configured}-{all_children}"
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_session_id,
                    messages=[Message.text("user", "forged durable result parent")],
                )
            )
        )
        await _collect(
            app.run(
                RunRequest(
                    agent_name="reviewer",
                    session_id=child_session_id,
                    parent_session_id=parent_session_id,
                    messages=[Message.text("user", "forged durable result child")],
                    metadata={
                        "subagent": {
                            "agent": "reviewer",
                            "mode": "durable",
                            "idempotency_key": "forged-durable-operation",
                            "durable_dispatch": {
                                "dispatch_id": "forged-durable-dispatch",
                                "queue_task_id": "forged-durable-task",
                                "queue_task_type": "cayu.dispatch.prepared-subagent.v1",
                            },
                        }
                    },
                )
            )
        )

        arguments = (
            {"all": True, "wait": False}
            if all_children
            else {"child_session_id": child_session_id, "wait": False}
        )
        result = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id=parent_session_id),
            arguments,
        )

        assert result.is_error is True
        assert "forged durable result canary" not in str(result.model_dump(mode="json"))
        summary = result.structured["children"][0] if all_children else result.structured
        assert summary["task_authority_status"] == "conflict"
        assert summary["retrieval_status"] == "authority_conflict"
        assert summary["result_text"] == ""

    asyncio.run(run())


def test_parent_recovery_finishes_marker_committed_before_child_creation() -> None:
    sessions = _CrashBeforeDurableChildStore()
    tasks = InMemoryTaskStore()

    async def crash_parent() -> None:
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        try:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-recovery-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        except _SimulatedWorkerLoss:
            return
        raise AssertionError("Simulated worker loss did not reach the child-create boundary.")

    asyncio.run(crash_parent())
    assert (
        asyncio.run(
            sessions.list_sessions(SessionQuery(parent_session_id="durable-recovery-parent"))
        ).sessions
        == []
    )
    assert asyncio.run(tasks.list_tasks(TaskQuery())) == []

    async def recover_parent() -> None:
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-recovery-parent",
                reason="fresh worker restart",
            )
        )
        assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        children = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-recovery-parent"))
        ).sessions
        assert len(children) == 1
        assert children[0].status is SessionStatus.PENDING
        queued = await tasks.list_tasks(TaskQuery())
        assert len(queued) == 1
        assert queued[0].status is TaskStatus.PENDING
        assert (
            queued[0].input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]
            == children[0].id
        )

    asyncio.run(recover_parent())


def test_parent_recovery_rejects_intent_with_request_conflicting_with_seed() -> None:
    sessions = _CrashBeforeDurableChildStore()
    tasks = InMemoryTaskStore()
    parent_session_id = "durable-seed-request-conflict-parent"

    async def crash_parent() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )

    asyncio.run(crash_parent())

    async def replace_intent_request() -> None:
        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        submissions = checkpoint["durable_subagent_submissions"]
        assert type(submissions) is dict and len(submissions) == 1
        idempotency_key = next(iter(submissions))
        intent = durable_subagent_submission_from_checkpoint(
            checkpoint,
            idempotency_key=idempotency_key,
        )
        assert intent is not None
        conflicting_request = intent.request.model_copy(
            update={"messages": [Message.text("user", "conflicting durable child task")]},
            deep=True,
        )
        authority = intent.authority.model_dump(mode="python")
        authority.update(
            intent.model_dump(
                mode="python",
                exclude={"authority", "submission_sha256"},
            )
        )
        authority.pop("request")
        authority.pop("request_sha256")
        conflicting_intent = new_durable_subagent_submission_intent(
            **authority,
            request=conflicting_request,
            request_sha256=durable_subagent_request_sha256(conflicting_request),
        )

        def replace_intent(_session, current_checkpoint):
            updated = deepcopy(current_checkpoint)
            updated["durable_subagent_submissions"][idempotency_key] = (
                conflicting_intent.model_dump(mode="json")
            )
            return updated

        await sessions.transform_checkpoint(parent_session_id, replace_intent)

    asyncio.run(replace_intent_request())

    async def recover_parent() -> None:
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        with pytest.raises(
            RuntimeError,
            match="Durable subagent intent conflicts with its preparation seed",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=parent_session_id,
                    reason="reject conflicting durable child request",
                )
            )
        assert (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions == []
        assert await tasks.list_tasks(TaskQuery()) == []
        assert provider.requests == []

    asyncio.run(recover_parent())


def test_staged_child_consumers_reject_intent_conflicting_with_parent_seed() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = _CrashAfterDurableTaskDispatcher(tasks)
        provider = _DurableSubagentProvider()
        parent_session_id = "durable-staged-seed-conflict-parent"
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        child = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions[0]
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        child_checkpoint = await sessions.load_checkpoint(child.id)
        intent = durable_subagent_submission_from_checkpoint(
            child_checkpoint,
            idempotency_key=queued.input["dispatch"]["prepared_subagent"]["authority"][
                "idempotency_key"
            ],
        )
        assert intent is not None
        seed = durable_subagent_submission_seed_from_checkpoint(
            checkpoint,
            idempotency_key=intent.idempotency_key,
        )
        assert seed is not None
        assert intent.authority == seed.authority
        require_durable_subagent_intent_matches_seed(intent, seed)
        conflicting_request = seed.request.model_copy(
            update={"messages": [Message.text("user", "conflicting durable child task")]},
            deep=True,
        )
        authority = seed.authority.model_dump(mode="python")
        authority.update(
            seed.model_dump(
                mode="python",
                exclude={"authority", "seed_sha256"},
            )
        )
        authority.pop("request")
        authority.pop("request_sha256")
        conflicting_seed = new_durable_subagent_submission_seed(
            **authority,
            request=conflicting_request,
            request_sha256=durable_subagent_request_sha256(conflicting_request),
        )

        def replace_seed(_session, current_checkpoint):
            updated = deepcopy(current_checkpoint)
            updated["durable_subagent_submission_seeds"][intent.idempotency_key] = (
                conflicting_seed.model_dump(mode="json")
            )
            return updated

        await sessions.transform_checkpoint(parent_session_id, replace_seed)
        provider_request_count = len(provider.requests)

        with pytest.raises(
            RuntimeError,
            match="intent conflicts with its preparation seed",
        ):
            await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=child.id,
                    reason="reject parent seed conflict",
                )
            )

        result = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id=parent_session_id),
            {"child_session_id": child.id, "wait": False},
        )
        assert result.is_error is True
        assert result.structured["task_authority_status"] == "conflict"
        assert result.structured["task_status"] is None

        handle = await dispatcher.process_next(app, worker_id="seed-conflict-worker")
        assert handle is not None
        assert handle.status.value == "failed"
        unchanged_child = await sessions.load(child.id)
        failed_task = await tasks.load_task(queued.id)
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        assert failed_task is not None
        assert failed_task.status is TaskStatus.FAILED
        assert len(provider.requests) == provider_request_count

    asyncio.run(run())


@pytest.mark.parametrize("parent_intent_state", ["missing", "conflicting"])
def test_worker_rejects_incomplete_or_conflicting_parent_intent(
    parent_intent_state: str,
) -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        parent_session_id = f"durable-parent-intent-{parent_intent_state}"
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_session_id,
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions[0]
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        child_checkpoint = await sessions.load_checkpoint(child.id)
        intent = durable_subagent_submission_from_checkpoint(
            child_checkpoint,
            idempotency_key=queued.input["dispatch"]["prepared_subagent"]["authority"][
                "idempotency_key"
            ],
        )
        assert intent is not None
        conflicting_request = intent.request.model_copy(
            update={"messages": [Message.text("user", "conflicting durable child task")]},
            deep=True,
        )
        authority = intent.authority.model_dump(mode="python")
        authority.update(
            intent.model_dump(
                mode="python",
                exclude={"authority", "submission_sha256"},
            )
        )
        authority.pop("request")
        authority.pop("request_sha256")
        conflicting_intent = new_durable_subagent_submission_intent(
            **authority,
            request=conflicting_request,
            request_sha256=durable_subagent_request_sha256(conflicting_request),
        )

        def replace_parent_intent(_session, current_checkpoint):
            updated = deepcopy(current_checkpoint)
            submissions = updated["durable_subagent_submissions"]
            if parent_intent_state == "missing":
                del submissions[intent.idempotency_key]
            else:
                submissions[intent.idempotency_key] = conflicting_intent.model_dump(mode="json")
            return updated

        await sessions.transform_checkpoint(parent_session_id, replace_parent_intent)
        provider_request_count = len(provider.requests)

        handle = await dispatcher.process_next(app, worker_id="parent-intent-conflict-worker")
        assert handle is not None
        assert handle.status.value == "failed"
        unchanged_child = await sessions.load(child.id)
        failed_task = await tasks.load_task(queued.id)
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        assert failed_task is not None
        assert failed_task.status is TaskStatus.FAILED
        assert len(provider.requests) == provider_request_count

    asyncio.run(run())


def test_cancellation_reconciles_marker_committed_before_failed_child_creation() -> None:
    async def run() -> None:
        sessions = _BlockingFailOnceDurableChildCreationStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)

        parent = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-cancel-marker-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        )
        await asyncio.wait_for(sessions.child_creation_started.wait(), timeout=1)
        parent.cancel("cancel after durable marker")
        sessions.release_child_creation.set()

        with pytest.raises(asyncio.CancelledError, match="cancel after durable marker"):
            await asyncio.wait_for(parent, timeout=2)
        assert parent.cancelling() == 1
        assert parent.cancelled() is True
        children = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-cancel-marker-parent")
            )
        ).sessions
        queued = await tasks.list_tasks(TaskQuery())
        assert len(children) == 1
        assert children[0].status is SessionStatus.PENDING
        assert len(queued) == 1
        assert queued[0].status is TaskStatus.PENDING
        assert queued[0].input["dispatch"]["prepared_subagent"]["authority"][
            "child_session_id"
        ] == (children[0].id)

    asyncio.run(run())


def test_parent_recovery_refreshes_child_after_concurrent_worker_completion() -> None:
    sessions = InMemorySessionStore()
    tasks = _CrashBeforeDurableTaskStore()
    provider = _DurableSubagentProvider()
    parent_session_id = "durable-concurrent-recovery-parent"

    async def run() -> None:
        bootstrap = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        bootstrap.register_provider(provider, default=True)
        _register_durable_subagent_agents(bootstrap)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                bootstrap.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )

        recovery_dispatcher = _BlockingAfterPreparedSubmissionDispatcher(tasks)
        recovery_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=recovery_dispatcher,
            enable_logging=False,
        )
        recovery_app.register_provider(provider, default=True)
        _register_durable_subagent_agents(recovery_app)
        worker_dispatcher = TaskStoreDispatcher(tasks)
        worker_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=worker_dispatcher,
            enable_logging=False,
        )
        worker_app.register_provider(provider, default=True)
        _register_durable_subagent_agents(worker_app)

        recovery_task = asyncio.create_task(
            recovery_app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=parent_session_id,
                    reason="race parent recovery with child completion",
                )
            )
        )
        await asyncio.wait_for(recovery_dispatcher.submission_published.wait(), timeout=1)
        worker_result = await worker_dispatcher.process_next(
            worker_app,
            worker_id="concurrent-recovery-worker",
        )
        assert worker_result is not None
        assert worker_result.status.value == "completed"
        recovery_dispatcher.release_submission.set()
        recovery = await asyncio.wait_for(recovery_task, timeout=2)

        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        child = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions[0]
        assert child.status is SessionStatus.COMPLETED
        queue_task = (await tasks.list_tasks(TaskQuery()))[0]
        assert queue_task.status is TaskStatus.COMPLETED

    asyncio.run(run())


def test_parent_recovery_uses_durable_effective_arguments_after_hook_modification() -> None:
    sessions = _CrashBeforeDurableChildStore()
    tasks = InMemoryTaskStore()

    def register(app: CayuApp) -> None:
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
            runtime_hooks=[_ModifyDurableSubagentTaskHook()],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"))

    async def crash_parent() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        register(app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-hook-recovery-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )

    asyncio.run(crash_parent())

    async def recover_parent() -> None:
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        register(app)
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-hook-recovery-parent",
                reason="recover hook-modified durable submission",
            )
        )
        assert recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        intent = queued.input["dispatch"]["prepared_subagent"]
        assert intent["authority"]["request"]["messages"][0]["content"][0]["text"] == (
            "hook-modified durable child task"
        )
        children = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-hook-recovery-parent")
            )
        ).sessions
        assert len(children) == 1
        assert children[0].status is SessionStatus.PENDING
        handle = await dispatcher.process_next(app, worker_id="hook-modified-child-worker")
        assert handle is not None
        assert handle.status.value == "completed"
        completed_child = await sessions.load(children[0].id)
        assert completed_child is not None
        assert completed_child.status is SessionStatus.COMPLETED

    asyncio.run(recover_parent())

    async def reattach_completed_child() -> None:
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        register(app)
        recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-hook-recovery-parent",
                reason="reattach hook-modified durable child",
            )
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        completed_events = [
            event
            for event in await sessions.load_events("durable-hook-recovery-parent")
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("tool_call_id") == "durable-child-call"
        ]
        assert len(completed_events) == 1
        recovered_result = completed_events[0].payload["result"]
        assert recovered_result["structured"]["status"] == SessionStatus.COMPLETED.value
        child_session_id = recovered_result["structured"]["child_session_id"]
        fetched = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id="durable-hook-recovery-parent"),
            {"child_session_id": child_session_id, "wait": False},
        )
        assert fetched.content == "durable child complete"
        assert provider.requests == []

    asyncio.run(reattach_completed_child())


def test_modified_handoff_receipt_retains_arguments_until_recovery_publication() -> None:
    sessions = _CrashAfterModifiedDurableHandoffReceiptStore()
    tasks = InMemoryTaskStore()
    parent_session_id = "durable-modified-receipt-recovery-parent"

    def register(app: CayuApp) -> None:
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
            runtime_hooks=[_ModifyDurableSubagentTaskHook()],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"))

    async def run() -> None:
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        register(app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )

        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        seeds = checkpoint["durable_subagent_submission_seeds"]
        submissions = checkpoint["durable_subagent_submissions"]
        assert type(seeds) is dict and len(seeds) == 1
        assert type(submissions) is dict and len(submissions) == 1
        idempotency_key = next(iter(submissions))
        assert submissions[idempotency_key]["outcome"] == "submitted"
        assert "request" not in submissions[idempotency_key]

        restarted_dispatcher = TaskStoreDispatcher(tasks)
        restarted = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=restarted_dispatcher,
            enable_logging=False,
        )
        restarted.register_provider(provider, default=True)
        register(restarted)
        pending = await restarted.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="recover modified handoff receipt",
            )
        )
        assert pending.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        handle = await restarted_dispatcher.process_next(
            restarted,
            worker_id="modified-handoff-worker",
        )
        assert handle is not None and handle.status.value == "completed"
        recovered = await restarted.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="publish recovered modified handoff",
            )
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovered.actions

        compacted = await sessions.load_checkpoint(parent_session_id)
        assert compacted is not None
        assert "durable_subagent_submission_seeds" not in compacted
        receipt = durable_subagent_submission_receipt_from_checkpoint(
            compacted,
            idempotency_key=idempotency_key,
        )
        assert receipt is not None and receipt.outcome == "submitted"
        assert len(await tasks.list_tasks(TaskQuery())) == 1
        assert (
            len(
                [
                    request
                    for request in provider.requests
                    if request.messages[0].content[0].text == "hook-modified durable child task"
                ]
            )
            == 1
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "hook",
    [
        _InjectSecretDurableSubagentTaskHook(),
        _InjectSecretDurableSubagentMetadataKeyHook(),
    ],
    ids=["value", "key"],
)
def test_durable_submission_rejects_post_hook_secret_before_checkpoint_or_queue(
    hook: RuntimeHook,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    async def run() -> tuple[list[Event], dict | None]:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            secret_redactor=SecretRedactor(_DURABLE_SUBAGENT_SECRET_CANARY),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                SubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
            runtime_hooks=[hook],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"))
        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-secret-seed-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        return events, await sessions.load_checkpoint("durable-secret-seed-parent")

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        events, checkpoint = asyncio.run(run())

    children_page = asyncio.run(
        sessions.list_sessions(SessionQuery(parent_session_id="durable-secret-seed-parent"))
    )
    assert children_page.sessions == []
    assert asyncio.run(tasks.list_tasks(TaskQuery())) == []
    assert checkpoint is None or "durable_subagent_submission_seeds" not in checkpoint
    stdout, stderr = capsys.readouterr()
    diagnostic_surfaces = (
        repr(events),
        repr(checkpoint),
        repr(captured_warnings),
        "\n".join(record.getMessage() for record in caplog.records),
        stdout,
        stderr,
    )
    assert all(_DURABLE_SUBAGENT_SECRET_CANARY not in surface for surface in diagnostic_surfaces)


def test_durable_submission_rejects_invocation_scoped_secret_before_persistence(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    async def run() -> tuple[list[Event], dict | None, list[Event], list[Message]]:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        app.register_environment(
            Environment(
                EnvironmentSpec(name="dynamic-secrets"),
                vault=StaticVault({"durable_subagent_secret": _DURABLE_SUBAGENT_SECRET_CANARY}),
            ),
            default=True,
        )
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                _ResolveSecretThenSubmitDurableSubagentTool(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
            runtime_hooks=[_InjectSecretDurableSubagentTaskHook()],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"))
        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-dynamic-secret-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        return (
            events,
            await sessions.load_checkpoint("durable-dynamic-secret-parent"),
            await sessions.load_events("durable-dynamic-secret-parent"),
            await sessions.load_transcript("durable-dynamic-secret-parent"),
        )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        events, checkpoint, stored_events, transcript = asyncio.run(run())

    children_page = asyncio.run(
        sessions.list_sessions(SessionQuery(parent_session_id="durable-dynamic-secret-parent"))
    )
    queued_tasks = asyncio.run(tasks.list_tasks(TaskQuery()))
    assert children_page.sessions == []
    assert queued_tasks == []
    assert checkpoint is None or "durable_subagent_submission_seeds" not in checkpoint
    stdout, stderr = capsys.readouterr()
    diagnostic_surfaces = (
        repr(events),
        repr(checkpoint),
        repr(stored_events),
        repr(transcript),
        repr(children_page.sessions),
        repr(queued_tasks),
        repr(captured_warnings),
        "\n".join(record.getMessage() for record in caplog.records),
        stdout,
        stderr,
    )
    assert all(_DURABLE_SUBAGENT_SECRET_CANARY not in surface for surface in diagnostic_surfaces)


def test_durable_submission_seals_before_active_secret_resolution_can_race_seed(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    observed_resolution_failure: BaseException | None = None

    async def run() -> tuple[list[Event], dict | None, list[Event], list[Message]]:
        nonlocal observed_resolution_failure
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        blocking_vault = _BlockingSecretVault()
        app.register_environment(
            Environment(EnvironmentSpec(name="dynamic-secrets"), vault=blocking_vault),
            default=True,
        )
        tool = _ResolveSecretDuringDurableSubagentTool(
            app,
            execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
            agents={
                "reviewer": SubagentSpec(
                    agent_name="reviewer",
                    mode=SubagentExecutionMode.DURABLE,
                )
            },
            blocking_vault=blocking_vault,
        )
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[tool],
            runtime_hooks=[_InjectSecretDurableSubagentTaskHook()],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"))
        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-racing-secret-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        observed_resolution_failure = tool.resolution_failure
        return (
            events,
            await sessions.load_checkpoint("durable-racing-secret-parent"),
            await sessions.load_events("durable-racing-secret-parent"),
            await sessions.load_transcript("durable-racing-secret-parent"),
        )

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always")
        events, checkpoint, stored_events, transcript = asyncio.run(run())

    children_page = asyncio.run(
        sessions.list_sessions(SessionQuery(parent_session_id="durable-racing-secret-parent"))
    )
    queued_tasks = asyncio.run(tasks.list_tasks(TaskQuery()))
    assert type(observed_resolution_failure) is RuntimeError
    assert str(observed_resolution_failure) == (
        "Secret resolution completed after tool publication."
    )
    assert children_page.sessions == []
    assert queued_tasks == []
    assert checkpoint is None or "durable_subagent_submission_seeds" not in checkpoint
    stdout, stderr = capsys.readouterr()
    diagnostic_surfaces = (
        repr(events),
        repr(checkpoint),
        repr(stored_events),
        repr(transcript),
        repr(children_page.sessions),
        repr(queued_tasks),
        repr(observed_resolution_failure),
        repr(captured_warnings),
        "\n".join(record.getMessage() for record in caplog.records),
        stdout,
        stderr,
    )
    assert all(_DURABLE_SUBAGENT_SECRET_CANARY not in surface for surface in diagnostic_surfaces)


def test_parent_recovery_finishes_child_committed_before_queue_task() -> None:
    sessions = InMemorySessionStore()
    tasks = _CrashBeforeDurableTaskStore()

    async def crash_parent() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        try:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-task-gap-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        except _SimulatedWorkerLoss:
            return
        raise AssertionError("Simulated worker loss did not reach queue publication.")

    asyncio.run(crash_parent())
    children = asyncio.run(
        sessions.list_sessions(SessionQuery(parent_session_id="durable-task-gap-parent"))
    ).sessions
    assert len(children) == 1
    child_id = children[0].id
    assert children[0].status is SessionStatus.PENDING
    assert asyncio.run(tasks.list_tasks(TaskQuery())) == []

    async def recover_parent() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        child_recovery = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=child_id,
                reason="preserve child before queue publication recovery",
            )
        )
        assert child_recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        preserved_child = await sessions.load(child_id)
        assert preserved_child is not None
        assert preserved_child.status is SessionStatus.PENDING
        assert await tasks.list_tasks(TaskQuery()) == []
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-task-gap-parent",
                reason="fresh worker restart",
            )
        )
        assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        queued = await tasks.list_tasks(TaskQuery())
        assert len(queued) == 1
        assert queued[0].status is TaskStatus.PENDING
        assert (
            queued[0].input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]
            == child_id
        )

    asyncio.run(recover_parent())


def test_transient_task_read_after_seed_preserves_parent_round_for_recovery() -> None:
    sessions = InMemorySessionStore()
    tasks = _FailOnceAfterDurableChildTaskStore()
    parent_session_id = "durable-transient-task-read-parent"

    async def fail_after_child_publication() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        with pytest.raises(RuntimeError) as raised:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        assert is_durable_subagent_submission_unsettled(
            raised.value,
            parent_session_id=parent_session_id,
        )
        parent = await sessions.load(parent_session_id)
        assert parent is not None
        assert parent.status is SessionStatus.RUNNING
        checkpoint = await sessions.load_checkpoint(parent_session_id)
        assert checkpoint is not None
        assert "pending_tool_round" in checkpoint
        assert "durable_subagent_submission_seeds" in checkpoint
        assert "durable_subagent_submissions" in checkpoint
        children = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions
        assert len(children) == 1
        assert children[0].status is SessionStatus.PENDING
        assert await tasks.list_tasks(TaskQuery()) == []
        assert not any(
            event.type is EventType.TOOL_CALL_FAILED
            for event in await sessions.load_events(parent_session_id)
        )

    asyncio.run(fail_after_child_publication())

    async def recover_on_fresh_app() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        result = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="recover transient durable submission failure",
            )
        )
        assert result.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        children = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions
        queued = await tasks.list_tasks(TaskQuery())
        assert len(children) == 1
        assert len(queued) == 1
        assert queued[0].status is TaskStatus.PENDING
        assert queued[0].input["dispatch"]["prepared_subagent"]["authority"][
            "child_session_id"
        ] == (children[0].id)

    asyncio.run(recover_on_fresh_app())


@pytest.mark.parametrize("terminal_task_status", [TaskStatus.FAILED, TaskStatus.CANCELLED])
def test_parent_recovery_settles_terminal_durable_queue_task(
    terminal_task_status: TaskStatus,
) -> None:
    sessions = InMemorySessionStore()
    tasks = _CrashBeforeDurableTaskStore()
    parent_session_id = f"durable-{terminal_task_status.value}-task-parent"

    async def crash_parent() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        try:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        except _SimulatedWorkerLoss:
            return
        raise AssertionError("Simulated worker loss did not reach queue publication.")

    asyncio.run(crash_parent())

    async def reconcile_then_cancel() -> None:
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        first = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="repair durable submission",
            )
        )
        assert first.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        queued = await tasks.list_tasks(TaskQuery())
        assert len(queued) == 1
        if terminal_task_status is TaskStatus.FAILED:
            await tasks.fail_task(queued[0].id, {"reason": "queue execution failed"})
        else:
            await tasks.cancel_task(queued[0].id, {"reason": "operator cancelled queue work"})
        result_tool = SubagentResultTool(sessions, task_store=tasks)
        before_recovery = await asyncio.wait_for(
            result_tool.run(
                ToolContext(session_id=parent_session_id),
                {
                    "child_session_id": queued[0].input["dispatch"]["prepared_subagent"][
                        "authority"
                    ]["child_session_id"],
                    "wait": True,
                    "timeout_s": 30,
                },
            ),
            timeout=1,
        )
        assert before_recovery.is_error is True
        assert before_recovery.structured["retrieval_status"] == "queue_terminal"
        assert before_recovery.structured["task_status"] == terminal_task_status.value
        all_results = await asyncio.wait_for(
            result_tool.run(
                ToolContext(session_id=parent_session_id),
                {"all": True, "wait": True, "timeout_s": 30},
            ),
            timeout=1,
        )
        assert all_results.is_error is True
        assert "still running" not in all_results.content
        assert terminal_task_status.value in all_results.content

        second = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="settle cancelled durable child",
            )
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in second.actions
        children = (
            await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
        ).sessions
        assert len(children) == 1
        assert children[0].status is SessionStatus.INTERRUPTED
        parent_events = await sessions.load_events(parent_session_id)
        recovered = [
            event
            for event in parent_events
            if event.type is EventType.TOOL_CALL_FAILED
            and event.payload.get("tool_call_id") == "durable-child-call"
        ]
        assert len(recovered) == 1

    asyncio.run(reconcile_then_cancel())


@pytest.mark.parametrize("terminal_task_status", [TaskStatus.FAILED, TaskStatus.CANCELLED])
def test_child_recovery_settles_terminal_durable_queue_task(
    terminal_task_status: TaskStatus,
) -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        parent_id = f"direct-child-{terminal_task_status.value}-parent"
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_id,
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (await sessions.list_sessions(SessionQuery(parent_session_id=parent_id))).sessions[
            0
        ]
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        if terminal_task_status is TaskStatus.FAILED:
            await tasks.fail_task(queued.id, {"reason": "queue execution failed"})
        else:
            await tasks.cancel_task(queued.id, {"reason": "queue execution cancelled"})

        recovered = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=child.id,
                reason="settle terminal durable queue owner",
            )
        )

        assert recovered.status is SessionStatus.INTERRUPTED
        assert recovered.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
        settled_child = await sessions.load(child.id)
        settled_task = await tasks.load_task(queued.id)
        assert settled_child is not None
        assert settled_child.status is SessionStatus.INTERRUPTED
        assert settled_task is not None
        assert settled_task.status is terminal_task_status
        child_events = await sessions.load_events(child.id)
        assert (
            len([event for event in child_events if event.type is EventType.SESSION_INTERRUPTED])
            == 1
        )
        assert len(provider.requests) == 2

        replay = await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=child.id,
                reason="repeat terminal durable child recovery",
            )
        )
        assert replay.actions == (IncompleteSessionRecoveryAction.SKIPPED_TERMINAL,)
        replay_events = await sessions.load_events(child.id)
        assert (
            len([event for event in replay_events if event.type is EventType.SESSION_INTERRUPTED])
            == 1
        )
        assert len(provider.requests) == 2

    asyncio.run(run())


def test_sqlite_restart_sweep_settles_failed_durable_queue_task(tmp_path) -> None:
    async def run() -> None:
        session_path = str(tmp_path / "terminal-child-sessions.db")
        task_path = str(tmp_path / "terminal-child-tasks.db")
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        bootstrap = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        bootstrap.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(bootstrap)
        try:
            await _collect(
                bootstrap.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="sqlite-terminal-child-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
            child = (
                await sessions.list_sessions(
                    SessionQuery(parent_session_id="sqlite-terminal-child-parent")
                )
            ).sessions[0]
            queued = (await tasks.list_tasks(TaskQuery()))[0]
            await tasks.fail_task(queued.id, {"reason": "worker rejected child"})
            child_id = child.id
            task_id = queued.id
        finally:
            await sessions.close()
            await tasks.close()

        restarted_sessions = SQLiteSessionStore(session_path)
        restarted_tasks = SQLiteTaskStore(task_path)
        restarted_provider = _DurableSubagentProvider()
        restarted = CayuApp(
            session_store=restarted_sessions,
            task_store=restarted_tasks,
            dispatcher=TaskStoreDispatcher(restarted_tasks),
            enable_logging=False,
        )
        restarted.register_provider(restarted_provider, default=True)
        _register_durable_subagent_agents(restarted)
        try:
            page = await restarted.recover_incomplete_sessions(
                IncompleteSessionsRecoveryRequest(
                    statuses={SessionStatus.PENDING},
                    reason="restart sweep terminal durable child",
                )
            )
            child_result = next(result for result in page.results if result.session_id == child_id)
            assert child_result.status is SessionStatus.INTERRUPTED
            assert child_result.actions == (IncompleteSessionRecoveryAction.INTERRUPTED_ABANDONED,)
            settled_child = await restarted_sessions.load(child_id)
            settled_task = await restarted_tasks.load_task(task_id)
            assert settled_child is not None
            assert settled_child.status is SessionStatus.INTERRUPTED
            assert settled_task is not None
            assert settled_task.status is TaskStatus.FAILED
            assert restarted_provider.requests == []
        finally:
            await restarted_sessions.close()
            await restarted_tasks.close()

    asyncio.run(run())


def test_result_inspection_rejects_completed_task_with_nonterminal_child() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-premature-task-completion-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        child_id = queued.input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]
        await tasks.complete_task(queued.id, {"status": "completed"})

        result = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id="durable-premature-task-completion-parent"),
            {"child_session_id": child_id, "wait": False},
        )

        assert result.is_error is True
        assert result.structured["task_authority_status"] == "verified"
        assert result.structured["task_status"] == TaskStatus.COMPLETED.value
        assert result.structured["retrieval_status"] == "queue_conflict"
        projection = await project_terminal_subagent_result(
            sessions,
            child_id,
            task_store=tasks,
            expected_task_id=queued.id,
        )
        assert projection["task_authority_status"] == "verified"
        assert projection["task_status"] == TaskStatus.COMPLETED.value
        assert projection["retrieval_status"] == "queue_conflict"
        assert projection["is_error"] is True
        await sessions.update_status(child_id, SessionStatus.INTERRUPTED)
        interrupted = await project_terminal_subagent_result(
            sessions,
            child_id,
            task_store=tasks,
            expected_task_id=queued.id,
        )
        assert interrupted["status"] == SessionStatus.INTERRUPTED.value
        assert interrupted["task_status"] == TaskStatus.COMPLETED.value
        assert interrupted["retrieval_status"] == "ready"
        assert interrupted["is_error"] is True

    asyncio.run(run())


def test_parent_recovery_preserves_paused_durable_queue_task() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=_CrashAfterDurableTaskDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-paused-task-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        paused = await tasks.pause_task(queued.id, reason="operator hold")
        assert paused.status is TaskStatus.PAUSED

        recovery_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        recovery_app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(recovery_app)
        recovery = await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-paused-task-parent",
                reason="inspect paused durable child",
            )
        )
        assert recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        child_id = queued.input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]
        child = await sessions.load(child_id)
        assert child is not None
        assert child.status is SessionStatus.PENDING
        still_paused = await tasks.load_task(queued.id)
        assert still_paused is not None
        assert still_paused.status is TaskStatus.PAUSED
        result = await SubagentResultTool(sessions, task_store=tasks).run(
            ToolContext(session_id="durable-paused-task-parent"),
            {"child_session_id": child_id, "wait": False},
        )
        assert result.is_error is False
        assert result.structured["task_status"] == TaskStatus.PAUSED.value
        assert result.structured["retrieval_status"] == "not_ready"

    asyncio.run(run())


def test_queue_task_without_referenced_child_fails_closed_before_provider_dispatch() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-missing-child-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-missing-child-parent")
            )
        ).sessions[0]
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        await sessions.delete_session(child.id)

        handle = await dispatcher.process_next(app, worker_id="durable-worker")
        assert handle is not None
        assert handle.status.value == "failed"
        failed = await tasks.load_task(queued.id)
        assert failed is not None
        assert failed.status is TaskStatus.FAILED
        assert await sessions.load(child.id) is None
        assert len(provider.requests) == 2

    asyncio.run(run())


def test_worker_rejects_corrupted_prepared_dispatch_request_before_provider_dispatch() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = _CorruptPreparedDispatchRequestStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-corrupt-request-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-corrupt-request-parent")
            )
        ).sessions[0]

        handle = await dispatcher.process_next(app, worker_id="durable-worker")

        assert handle is None
        assert len(provider.requests) == 2
        unchanged_child = await sessions.load(child.id)
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        failed_task = (await tasks.list_tasks(TaskQuery()))[0]
        assert failed_task.status is TaskStatus.FAILED

    asyncio.run(run())


def test_worker_terminally_rejects_malformed_child_submission_authority() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        parent_id = "durable-malformed-child-authority-parent"
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_id,
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (await sessions.list_sessions(SessionQuery(parent_session_id=parent_id))).sessions[
            0
        ]
        queued = (await tasks.list_tasks(TaskQuery()))[0]

        def corrupt_submission(_session, checkpoint):
            updated = deepcopy(checkpoint)
            assert updated is not None
            updated["durable_subagent_submissions"] = "malformed"
            return updated

        await sessions.transform_checkpoint(child.id, corrupt_submission)

        handle = await dispatcher.process_next(app, worker_id="durable-worker")

        assert handle is not None
        assert handle.status.value == "failed"
        failed = await tasks.load_task(queued.id)
        unchanged_child = await sessions.load(child.id)
        assert failed is not None
        assert failed.status is TaskStatus.FAILED
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        assert len(provider.requests) == 2
        assert await dispatcher.process_next(app, worker_id="replacement-worker") is None

    asyncio.run(run())


def test_worker_terminally_rejects_child_authority_removed_during_admission() -> None:
    async def run() -> None:
        sessions = _RemovePreparedChildAuthorityAtAdmissionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        parent_id = "durable-atomic-child-authority-parent"
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=parent_id,
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (await sessions.list_sessions(SessionQuery(parent_session_id=parent_id))).sessions[
            0
        ]
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        sessions.remove_child_authority_at_admission = True

        handle = await dispatcher.process_next(app, worker_id="durable-worker")

        assert handle is not None
        assert handle.status.value == "failed"
        failed = await tasks.load_task(queued.id)
        unchanged_child = await sessions.load(child.id)
        assert failed is not None
        assert failed.status is TaskStatus.FAILED
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        assert len(provider.requests) == 2
        assert await dispatcher.process_next(app, worker_id="replacement-worker") is None

    asyncio.run(run())


def test_parent_interruption_cascades_to_unclaimed_durable_child() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _InterruptibleDurableParentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        parent_task = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-interrupt-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        )
        await asyncio.wait_for(provider.parent_waiting.wait(), timeout=5)
        child = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-interrupt-parent"))
        ).sessions[0]
        assert child.status is SessionStatus.PENDING

        interrupt_events = await _collect(
            app.interrupt_session(
                InterruptSessionRequest(
                    session_id="durable-interrupt-parent",
                    reason="operator stopped the parent",
                )
            )
        )
        parent_events = await asyncio.wait_for(parent_task, timeout=2)
        assert interrupt_events[-1].type is EventType.SESSION_INTERRUPTED
        assert parent_events[-1].type is EventType.SESSION_INTERRUPTED
        assert await app.drain_background_interruptions(timeout_s=1) is True
        interrupted_child = await sessions.load(child.id)
        assert interrupted_child is not None
        assert interrupted_child.status is SessionStatus.INTERRUPTED

        handle = await dispatcher.process_next(app, worker_id="durable-worker")
        assert handle is not None
        assert handle.status.value == "failed"
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        assert queued.status is TaskStatus.FAILED
        assert len(provider.requests) == 2

    asyncio.run(run())


def test_durable_queue_task_preserves_real_parent_task_lineage() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        parent_task = await tasks.create_task(
            TaskCreate(task_id="durable-parent-task", type="parent-work")
        )
        claimed = await tasks.claim_task("parent-worker", lease_seconds=300)
        assert claimed is not None
        assert claimed.id == parent_task.id
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-task-parent-session",
                    task_id=parent_task.id,
                    task_worker_id="parent-worker",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        queued = [task for task in await tasks.list_tasks(TaskQuery()) if task.id != parent_task.id]
        assert len(queued) == 1
        assert queued[0].parent_task_id == parent_task.id
        intent = queued[0].input["dispatch"]["prepared_subagent"]
        assert intent["authority"]["parent_task_id"] == parent_task.id
        completed_parent = await tasks.load_task(parent_task.id)
        assert completed_parent is not None
        assert completed_parent.status is TaskStatus.COMPLETED

    asyncio.run(run())


def test_sqlite_restart_executes_prepared_child_once(tmp_path) -> None:
    async def run() -> None:
        session_path = tmp_path / "durable-subagent-sessions.sqlite"
        task_path = tmp_path / "durable-subagent-tasks.sqlite"
        producer_sessions = SQLiteSessionStore(session_path)
        producer_tasks = SQLiteTaskStore(task_path)
        producer_dispatcher = TaskStoreDispatcher(producer_tasks)
        producer = CayuApp(
            session_store=producer_sessions,
            task_store=producer_tasks,
            dispatcher=producer_dispatcher,
            enable_logging=False,
        )
        producer_provider = _DurableSubagentProvider()
        producer.register_provider(producer_provider, default=True)
        _register_durable_subagent_agents(producer)
        try:
            await _collect(
                producer.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-sqlite-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
            queued = await producer_tasks.list_tasks(TaskQuery())
            assert len(queued) == 1
            queue_task_id = queued[0].id
            child_session_id = queued[0].input["dispatch"]["prepared_subagent"]["authority"][
                "child_session_id"
            ]
            assert len(producer_provider.requests) == 2
        finally:
            await producer_sessions.close()
            await producer_tasks.close()

        worker_sessions = SQLiteSessionStore(session_path)
        worker_tasks = SQLiteTaskStore(task_path)
        worker_dispatcher = TaskStoreDispatcher(worker_tasks)
        worker = CayuApp(
            session_store=worker_sessions,
            task_store=worker_tasks,
            dispatcher=worker_dispatcher,
            enable_logging=False,
        )
        worker_provider = _DurableSubagentProvider()
        worker.register_provider(worker_provider, default=True)
        _register_durable_subagent_agents(worker)
        try:
            recovery = await worker.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=child_session_id,
                    reason="restart sweep before durable child claim",
                )
            )
            assert recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
            handle = await worker_dispatcher.process_next(
                worker,
                worker_id="sqlite-durable-worker",
            )
            assert handle is not None
            assert handle.status.value == "completed"
            child = await worker_sessions.load(child_session_id)
            assert child is not None
            assert child.status is SessionStatus.COMPLETED
            task = await worker_tasks.load_task(queue_task_id)
            assert task is not None
            assert task.status is TaskStatus.COMPLETED
            assert len(worker_provider.requests) == 1
        finally:
            await worker_sessions.close()
            await worker_tasks.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "crash_phase",
    [
        "during_preparation",
        "before_child",
        "between_child_and_task",
        "after_task",
        "after_child",
    ],
)
def test_sqlite_submission_crash_boundaries_reconcile_one_child_and_task(
    tmp_path,
    crash_phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_path = tmp_path / f"durable-{crash_phase}-sessions.sqlite"
    task_path = tmp_path / f"durable-{crash_phase}-tasks.sqlite"
    parent_session_id = f"durable-{crash_phase}-parent"

    async def crash_parent() -> None:
        sessions = (
            _CrashBeforeDurableChildSQLiteStore(session_path)
            if crash_phase == "before_child"
            else SQLiteSessionStore(session_path)
        )
        tasks = (
            _CrashBeforeDurableTaskSQLiteStore(task_path)
            if crash_phase == "between_child_and_task"
            else SQLiteTaskStore(task_path)
        )
        if crash_phase == "after_task":
            dispatcher = _CrashAfterDurableTaskDispatcher(tasks)
        elif crash_phase == "after_child":
            dispatcher = _CrashAfterDurableChildCompletionDispatcher(tasks)
        else:
            dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        if isinstance(dispatcher, _CrashAfterDurableChildCompletionDispatcher):
            dispatcher.worker_runtime = app
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        if crash_phase == "during_preparation":
            original_prepare = app._session_engine._prepare_initial_run
            preparation_calls = 0

            async def lose_worker_during_child_preparation(*args, **kwargs):
                nonlocal preparation_calls
                result = await original_prepare(*args, **kwargs)
                preparation_calls += 1
                if preparation_calls == 2:
                    raise _SimulatedWorkerLoss(
                        "simulated SQLite loss during child profile preparation"
                    )
                return result

            monkeypatch.setattr(
                app._session_engine,
                "_prepare_initial_run",
                lose_worker_during_child_preparation,
            )
        try:
            with pytest.raises(_SimulatedWorkerLoss):
                await _collect(
                    app.run(
                        RunRequest(
                            agent_name="parent",
                            session_id=parent_session_id,
                            messages=[Message.text("user", "parent task")],
                        )
                    )
                )
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(crash_parent())

    async def recover() -> None:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        try:
            first = await app.recover_incomplete_session(
                IncompleteSessionRecoveryRequest(
                    session_id=parent_session_id,
                    reason="recover durable submission crash boundary",
                )
            )
            children = (
                await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
            ).sessions
            queued = await tasks.list_tasks(TaskQuery())
            assert len(children) == 1
            assert len(queued) == 1
            child_id = children[0].id
            queue_task_id = queued[0].id
            if children[0].status is not SessionStatus.COMPLETED:
                assert first.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
                handle = await dispatcher.process_next(
                    app,
                    worker_id=f"{crash_phase}-replacement-worker",
                )
                assert handle is not None
                assert handle.status.value == "completed"
                second = await app.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(
                        session_id=parent_session_id,
                        reason="attach terminal durable child",
                    )
                )
                assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in second.actions
                assert len(provider.requests) == 1
            else:
                assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in first.actions
                assert provider.requests == []

            final_children = (
                await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
            ).sessions
            final_tasks = await tasks.list_tasks(TaskQuery())
            assert [child.id for child in final_children] == [child_id]
            assert [task.id for task in final_tasks] == [queue_task_id]
            assert final_children[0].status is SessionStatus.COMPLETED
            assert final_tasks[0].status is TaskStatus.COMPLETED
            parent_events = await sessions.load_events(parent_session_id)
            attached = [
                event
                for event in parent_events
                if event.type is EventType.TOOL_CALL_COMPLETED
                and event.payload.get("tool_call_id") == "durable-child-call"
            ]
            assert len(attached) == 1
            assert attached[0].payload["result"]["structured"]["child_session_id"] == child_id
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(recover())


def test_sqlite_worker_claim_loss_reclaims_same_durable_child(tmp_path) -> None:
    session_path = tmp_path / "durable-claim-loss-sessions.sqlite"
    task_path = tmp_path / "durable-claim-loss-tasks.sqlite"

    async def produce() -> tuple[str, str]:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        try:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-claim-loss-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
            queued = (await tasks.list_tasks(TaskQuery()))[0]
            return (
                queued.id,
                queued.input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"],
            )
        finally:
            await sessions.close()
            await tasks.close()

    queue_task_id, child_session_id = asyncio.run(produce())

    async def lose_claim() -> None:
        sessions = SQLiteSessionStore(session_path)
        tasks = _CrashAfterClaimSQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks, lease_seconds=1)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        try:
            with pytest.raises(_SimulatedWorkerLoss):
                await dispatcher.process_next(app, worker_id="lost-durable-worker")
            claimed = await tasks.load_task(queue_task_id)
            assert claimed is not None
            assert claimed.status is TaskStatus.CLAIMED
            child = await sessions.load(child_session_id)
            assert child is not None
            assert child.status is SessionStatus.PENDING
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(lose_claim())

    async def reclaim_and_run() -> None:
        await asyncio.sleep(1.1)
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks, lease_seconds=1)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        try:
            reclaimed = await tasks.reclaim_expired(
                query=TaskQuery(type=dispatcher.prepared_subagent_task_type)
            )
            assert [task.id for task in reclaimed] == [queue_task_id]
            handle = await dispatcher.process_next(
                app,
                worker_id="replacement-durable-worker",
            )
            assert handle is not None
            assert handle.status.value == "completed"
            child = await sessions.load(child_session_id)
            assert child is not None
            assert child.status is SessionStatus.COMPLETED
            task = await tasks.load_task(queue_task_id)
            assert task is not None
            assert task.status is TaskStatus.COMPLETED
            assert len(provider.requests) == 1
            child_events = await sessions.load_events(child_session_id)
            assert (
                len([event for event in child_events if event.type is EventType.MODEL_STARTED]) == 1
            )
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(reclaim_and_run())


def test_reclaimed_worker_recovers_child_admitted_before_worker_loss() -> None:
    async def run() -> None:
        sessions = _CrashAfterPreparedChildAdmissionStore()
        tasks = InMemoryTaskStore()
        first_dispatcher = TaskStoreDispatcher(tasks, lease_seconds=1)
        first_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=first_dispatcher,
            enable_logging=False,
        )
        first_app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(first_app)
        await _collect(
            first_app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-admitted-loss-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        queue_task = (await tasks.list_tasks(TaskQuery()))[0]
        child_id = queue_task.input["dispatch"]["prepared_subagent"]["authority"][
            "child_session_id"
        ]

        with pytest.raises(_SimulatedWorkerLoss, match="after child admission"):
            await first_dispatcher.process_next(first_app, worker_id="lost-worker")
        claimed = await tasks.load_task(queue_task.id)
        child = await sessions.load(child_id)
        assert claimed is not None and claimed.status is TaskStatus.CLAIMED
        assert child is not None and child.status is SessionStatus.RUNNING

        await asyncio.sleep(1.1)
        assert [task.id for task in await tasks.reclaim_expired(query=TaskQuery())] == [
            queue_task.id
        ]
        replacement_dispatcher = TaskStoreDispatcher(
            tasks,
            lease_seconds=1,
            recover_stalled_sessions_after_seconds=0,
        )
        replacement_provider = _DurableSubagentProvider()
        replacement_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=replacement_dispatcher,
            enable_logging=False,
        )
        replacement_app.register_provider(replacement_provider, default=True)
        _register_durable_subagent_agents(replacement_app)

        recovery = await replacement_dispatcher.process_next(
            replacement_app,
            worker_id="replacement-worker",
        )
        assert recovery is not None
        assert recovery.status.value == "submitted"
        assert recovery.metadata["requeued"] is True
        assert recovery.metadata["recovered_session"] is True
        assert replacement_provider.requests == []

        terminal = await replacement_dispatcher.process_next(
            replacement_app,
            worker_id="replacement-worker",
        )
        assert terminal is not None
        assert terminal.status.value == "interrupted"
        settled_child = await sessions.load(child_id)
        settled_task = await tasks.load_task(queue_task.id)
        assert settled_child is not None
        assert settled_child.status is SessionStatus.INTERRUPTED
        assert settled_task is not None
        assert settled_task.status is TaskStatus.COMPLETED
        assert replacement_provider.requests == []

    asyncio.run(run())


def test_recovery_attaches_child_completed_before_parent_tool_result() -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()

    async def crash_parent() -> tuple[str, int]:
        dispatcher = _CrashAfterDurableChildCompletionDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        dispatcher.worker_runtime = app
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        try:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-child-first-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        except _SimulatedWorkerLoss:
            children = (
                await sessions.list_sessions(
                    SessionQuery(parent_session_id="durable-child-first-parent")
                )
            ).sessions
            assert len(children) == 1
            assert children[0].status is SessionStatus.COMPLETED
            queued = await tasks.list_tasks(TaskQuery())
            assert len(queued) == 1
            assert queued[0].status is TaskStatus.COMPLETED
            return children[0].id, len(provider.requests)
        raise AssertionError("Simulated parent loss did not reach the result gap.")

    child_id, original_provider_calls = asyncio.run(crash_parent())
    assert original_provider_calls == 2

    async def recover_parent() -> None:
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        await app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-child-first-parent",
                reason="recover completed child handoff",
            )
        )
        child = await sessions.load(child_id)
        assert child is not None
        assert child.status is SessionStatus.COMPLETED
        queued = await tasks.list_tasks(TaskQuery())
        assert len(queued) == 1
        assert queued[0].status is TaskStatus.COMPLETED
        parent_events = await sessions.load_events("durable-child-first-parent")
        recovered_tool_events = [
            event
            for event in parent_events
            if event.type is EventType.TOOL_CALL_COMPLETED
            and event.payload.get("tool_call_id") == "durable-child-call"
        ]
        assert len(recovered_tool_events) == 1
        assert (
            recovered_tool_events[0].payload["result"]["structured"]["child_session_id"] == child_id
        )
        assert provider.requests == []

    asyncio.run(recover_parent())


def test_fresh_process_executes_sqlite_prepared_child_once(tmp_path) -> None:
    session_path = tmp_path / "fresh-durable-subagent-sessions.sqlite"
    task_path = tmp_path / "fresh-durable-subagent-tasks.sqlite"

    async def produce() -> tuple[str, str]:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        provider = _DurableSubagentProvider()
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        try:
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="fresh-durable-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
            queued = (await tasks.list_tasks(TaskQuery()))[0]
            return (
                queued.id,
                queued.input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"],
            )
        finally:
            await sessions.close()
            await tasks.close()

    queue_task_id, child_session_id = asyncio.run(produce())
    process = multiprocessing.get_context("spawn").Process(
        target=_run_sqlite_durable_child_worker,
        args=(str(session_path), str(task_path)),
    )
    process.start()
    process.join(timeout=20)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        raise AssertionError("Fresh durable-subagent worker did not exit.")
    assert process.exitcode == 0

    async def inspect() -> None:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        try:
            child = await sessions.load(child_session_id)
            assert child is not None
            assert child.status is SessionStatus.COMPLETED
            task = await tasks.load_task(queue_task_id)
            assert task is not None
            assert task.status is TaskStatus.COMPLETED
            child_events = await sessions.load_events(child_session_id)
            assert (
                len([event for event in child_events if event.type is EventType.MODEL_STARTED]) == 1
            )
            assert (
                len([event for event in child_events if event.type is EventType.SESSION_COMPLETED])
                == 1
            )
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(inspect())


def test_fresh_process_recovers_parent_and_retrieves_terminal_child(tmp_path) -> None:
    session_path = tmp_path / "fresh-parent-recovery-sessions.sqlite"
    task_path = tmp_path / "fresh-parent-recovery-tasks.sqlite"
    parent_session_id = "fresh-parent-recovery"

    async def produce_crash_boundary() -> tuple[str, str]:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=_CrashAfterDurableTaskDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        try:
            with pytest.raises(_SimulatedWorkerLoss):
                await _collect(
                    app.run(
                        RunRequest(
                            agent_name="parent",
                            session_id=parent_session_id,
                            messages=[Message.text("user", "parent task")],
                        )
                    )
                )
            queued = (await tasks.list_tasks(TaskQuery()))[0]
            return (
                queued.id,
                queued.input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"],
            )
        finally:
            await sessions.close()
            await tasks.close()

    queue_task_id, child_session_id = asyncio.run(produce_crash_boundary())
    spawn = multiprocessing.get_context("spawn")
    worker = spawn.Process(
        target=_run_sqlite_durable_child_worker,
        args=(str(session_path), str(task_path)),
    )
    worker.start()
    worker.join(timeout=20)
    if worker.is_alive():
        worker.terminate()
        worker.join(timeout=5)
        raise AssertionError("Fresh durable-subagent worker did not exit.")
    assert worker.exitcode == 0

    recovery = spawn.Process(
        target=_run_sqlite_durable_parent_recovery,
        args=(
            str(session_path),
            str(task_path),
            parent_session_id,
            child_session_id,
        ),
    )
    recovery.start()
    recovery.join(timeout=20)
    if recovery.is_alive():
        recovery.terminate()
        recovery.join(timeout=5)
        raise AssertionError("Fresh durable-subagent parent recovery did not exit.")
    assert recovery.exitcode == 0

    async def inspect() -> None:
        sessions = SQLiteSessionStore(session_path)
        tasks = SQLiteTaskStore(task_path)
        try:
            children = (
                await sessions.list_sessions(SessionQuery(parent_session_id=parent_session_id))
            ).sessions
            queued = await tasks.list_tasks(TaskQuery())
            assert [child.id for child in children] == [child_session_id]
            assert [task.id for task in queued] == [queue_task_id]
            assert children[0].status is SessionStatus.COMPLETED
            assert queued[0].status is TaskStatus.COMPLETED
        finally:
            await sessions.close()
            await tasks.close()

    asyncio.run(inspect())


def test_parent_recovery_acknowledges_terminal_prepared_task_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = InMemorySessionStore()
    tasks = InMemoryTaskStore()
    parent_session_id = "durable-parent-terminal-receipt-recovery"

    async def scenario() -> None:
        parent_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=_CrashAfterDurableTaskDispatcher(tasks),
            enable_logging=False,
        )
        parent_app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(parent_app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                parent_app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id=parent_session_id,
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        child_id = queued.input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]

        worker_dispatcher = TaskStoreDispatcher(tasks)
        worker_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=worker_dispatcher,
            enable_logging=False,
        )
        worker_app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(worker_app)

        async def lose_terminal_acknowledgement(*args, **kwargs) -> None:
            del args, kwargs
            raise ConnectionError("prepared task terminal acknowledgement lost")

        monkeypatch.setattr(
            worker_app,
            "_acknowledge_queued_dispatch",
            lose_terminal_acknowledgement,
        )
        with pytest.raises(
            ConnectionError,
            match="prepared task terminal acknowledgement lost",
        ):
            await worker_dispatcher.process_next(
                worker_app,
                worker_id="durable-terminal-receipt-worker",
            )
        terminal_task = await tasks.load_task(queued.id)
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED
        retained_child_checkpoint = await sessions.load_checkpoint(child_id)
        assert retained_child_checkpoint is not None
        assert "queued_dispatch_terminal_receipts" in retained_child_checkpoint

        recovery_provider = _DurableSubagentProvider()
        recovery_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        recovery_app.register_provider(recovery_provider, default=True)
        _register_durable_subagent_agents(recovery_app)
        recovery = await recovery_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id=parent_session_id,
                reason="recover parent after prepared-task acknowledgement loss",
            )
        )

        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in recovery.actions
        settled_child_checkpoint = await sessions.load_checkpoint(child_id)
        assert settled_child_checkpoint is not None
        assert "queued_dispatch_terminal_receipts" not in settled_child_checkpoint
        assert recovery_provider.requests == []

    asyncio.run(scenario())


def test_worker_rejects_changed_child_execution_profile_before_provider_dispatch() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-profile-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (
            await sessions.list_sessions(SessionQuery(parent_session_id="durable-profile-parent"))
        ).sessions[0]
        worker_dispatcher = TaskStoreDispatcher(tasks)
        worker_provider = _DurableSubagentProvider()
        worker = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=worker_dispatcher,
            enable_logging=False,
        )
        worker.register_provider(worker_provider, default=True)
        worker.register_agent(AgentSpec(name="parent", model="model"))
        worker.register_agent(AgentSpec(name="reviewer", model="changed-model"))

        handle = await worker_dispatcher.process_next(
            worker,
            worker_id="changed-profile-worker",
        )
        assert handle is not None
        assert handle.status.value == "submitted"
        unchanged_child = await sessions.load(child.id)
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        assert queued.status is TaskStatus.PENDING
        assert len(provider.requests) == 2
        assert worker_provider.requests == []

        compatible = await dispatcher.process_next(app, worker_id="compatible-profile-worker")
        assert compatible is not None
        assert compatible.status.value == "completed"
        completed_child = await sessions.load(child.id)
        assert completed_child is not None
        assert completed_child.status is SessionStatus.COMPLETED
        assert len(provider.requests) == 3

    asyncio.run(run())


@pytest.mark.parametrize("missing_component", ["agent", "provider"])
def test_incompatible_worker_requeues_prepared_child_for_compatible_worker(
    missing_component: str,
) -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        compatible_dispatcher = TaskStoreDispatcher(tasks)
        compatible_provider = _DurableSubagentProvider()
        compatible_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=compatible_dispatcher,
            enable_logging=False,
        )
        compatible_app.register_provider(compatible_provider, default=True)
        _register_durable_subagent_agents(compatible_app)
        await _collect(
            compatible_app.run(
                RunRequest(
                    agent_name="parent",
                    session_id=f"durable-missing-{missing_component}-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        child = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id=f"durable-missing-{missing_component}-parent")
            )
        ).sessions[0]

        incompatible_dispatcher = TaskStoreDispatcher(tasks)
        incompatible_provider = _DurableSubagentProvider()
        incompatible_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=incompatible_dispatcher,
            enable_logging=False,
        )
        if missing_component != "provider":
            incompatible_app.register_provider(incompatible_provider, default=True)
        if missing_component != "agent":
            _register_durable_subagent_agents(incompatible_app)

        requeued = await incompatible_dispatcher.process_next(
            incompatible_app,
            worker_id=f"missing-{missing_component}-worker",
        )
        assert requeued is not None
        assert requeued.status.value == "submitted"
        assert requeued.metadata["requeued"] is True
        unchanged_child = await sessions.load(child.id)
        assert unchanged_child is not None
        assert unchanged_child.status is SessionStatus.PENDING
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        assert queued.status is TaskStatus.PENDING
        assert queued.worker_id is None
        assert incompatible_provider.requests == []

        completed = await compatible_dispatcher.process_next(
            compatible_app,
            worker_id=f"compatible-after-missing-{missing_component}",
        )
        assert completed is not None
        assert completed.status.value == "completed"
        completed_child = await sessions.load(child.id)
        assert completed_child is not None
        assert completed_child.status is SessionStatus.COMPLETED
        assert len(compatible_provider.requests) == 3

    asyncio.run(run())


def test_unsupported_session_store_rejects_before_submission_state() -> None:
    async def run() -> None:
        sessions = _UnsupportedPendingCheckpointStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-unsupported-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )
        tool_failures = [event for event in events if event.type is EventType.TOOL_CALL_FAILED]
        assert len(tool_failures) == 1
        result = tool_failures[0].payload["result"]
        assert result["structured"]["status"] == "submission_failed"
        assert result["structured"]["error_type"] == "NotImplementedError"
        assert "error" not in result["structured"]
        checkpoint = await sessions.load_checkpoint("durable-unsupported-parent")
        assert checkpoint is not None
        assert "durable_subagent_submissions" not in checkpoint
        assert (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-unsupported-parent")
            )
        ).sessions == []
        assert await tasks.list_tasks(TaskQuery()) == []

    asyncio.run(run())


def test_mismatched_dispatcher_task_store_rejects_before_submission_state() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        app_tasks = InMemoryTaskStore()
        dispatcher_tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=app_tasks,
            dispatcher=TaskStoreDispatcher(dispatcher_tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-mismatched-task-store-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )

        failures = [event for event in events if event.type is EventType.TOOL_CALL_FAILED]
        assert len(failures) == 1
        assert failures[0].payload["result"]["structured"]["status"] == ("submission_failed")
        checkpoint = await sessions.load_checkpoint("durable-mismatched-task-store-parent")
        assert checkpoint is not None
        assert "durable_subagent_submissions" not in checkpoint
        assert (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-mismatched-task-store-parent")
            )
        ).sessions == []
        assert await app_tasks.list_tasks(TaskQuery()) == []
        assert await dispatcher_tasks.list_tasks(TaskQuery()) == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "tool_type",
    [
        _RetargetingDurableSubagentTool,
        _RetargetingDurableSubagentContextTool,
        _CopyingDurableSubagentContextTool,
    ],
    ids=["arguments", "retargeted-context", "copied-context"],
)
def test_durable_submission_rejects_retargeted_genuine_authority(tool_type) -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        app.register_agent(
            AgentSpec(name="parent", model="model"),
            tools=[
                tool_type(
                    app,
                    execution_profile_identity=_DURABLE_SUBAGENT_TOOL_PROFILE_IDENTITY,
                    agents={
                        "reviewer": SubagentSpec(
                            agent_name="reviewer",
                            mode=SubagentExecutionMode.DURABLE,
                        )
                    },
                )
            ],
        )
        app.register_agent(AgentSpec(name="reviewer", model="model"))

        events = await _collect(
            app.run(
                RunRequest(
                    agent_name="parent",
                    session_id="durable-retargeted-arguments-parent",
                    messages=[Message.text("user", "parent task")],
                )
            )
        )

        failures = [event for event in events if event.type is EventType.TOOL_CALL_FAILED]
        assert len(failures) == 1
        result = failures[0].payload["result"]
        assert result["structured"]["status"] == "submission_failed"
        assert result["structured"]["error_type"] == "RuntimeError"
        assert (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-retargeted-arguments-parent")
            )
        ).sessions == []
        assert await tasks.list_tasks(TaskQuery()) == []

    asyncio.run(run())


@pytest.mark.parametrize(
    "tool_timeout_seconds",
    [None, 1.0],
    ids=["without-timeout", "before-timeout"],
)
def test_parent_task_cancellation_waits_for_durable_submission_settlement(
    tool_timeout_seconds: float | None,
) -> None:
    async def run() -> None:
        sessions = _BlockingDurableChildCreationStore()
        tasks = InMemoryTaskStore()
        dispatcher = TaskStoreDispatcher(tasks)
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            tool_timeout_seconds=tool_timeout_seconds,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        parent = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-submission-cancel-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        )
        await asyncio.wait_for(sessions.child_creation_started.wait(), timeout=1)
        parent.cancel("cancel during durable submission")
        await asyncio.sleep(0)
        assert parent.done() is False
        assert (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-submission-cancel-parent")
            )
        ).sessions == []
        assert await tasks.list_tasks(TaskQuery()) == []

        sessions.release_child_creation.set()
        with pytest.raises(asyncio.CancelledError, match="cancel during durable submission"):
            await parent
        assert parent.cancelling() == 1
        assert parent.cancelled() is True
        children = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-submission-cancel-parent")
            )
        ).sessions
        queued = await tasks.list_tasks(TaskQuery())
        assert len(children) == 1
        assert len(queued) == 1
        assert (
            queued[0].input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]
            == children[0].id
        )

    asyncio.run(run())


def test_tool_timeout_reports_queued_child_when_durable_submission_commits_late() -> None:
    async def run() -> None:
        sessions = _BlockingDurableChildCreationStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            tool_timeout_seconds=0.01,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        parent = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-submission-timeout-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        )
        await asyncio.wait_for(sessions.child_creation_started.wait(), timeout=1)
        await asyncio.sleep(0.02)
        sessions.release_child_creation.set()

        events = await asyncio.wait_for(parent, timeout=2)

        queued_results = [
            event.payload["result"]
            for event in events
            if event.type is EventType.TOOL_CALL_COMPLETED
        ]
        assert len(queued_results) == 1, [
            (str(event.type), event.payload.get("result"))
            for event in events
            if event.type in {EventType.TOOL_CALL_COMPLETED, EventType.TOOL_CALL_FAILED}
        ]
        assert queued_results[0]["structured"]["status"] == "queued"
        assert not any(
            event.type is EventType.TOOL_CALL_FAILED
            and event.payload.get("tool_call_id") == "durable-child-call"
            for event in events
        )
        children = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-submission-timeout-parent")
            )
        ).sessions
        queued = await tasks.list_tasks(TaskQuery())
        assert len(children) == 1
        assert len(queued) == 1
        assert queued[0].status is TaskStatus.PENDING

    asyncio.run(run())


def test_tool_timeout_preserves_unsettled_submission_for_exact_recovery() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = _BlockingFailOnceDurableTaskReadStore()
        dispatcher = TaskStoreDispatcher(tasks)
        provider = _DurableSubagentProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            tool_timeout_seconds=0.01,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        parent = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-timeout-unsettled-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        )
        await asyncio.wait_for(tasks.read_started.wait(), timeout=1)
        await asyncio.sleep(0.02)
        tasks.release_read.set()

        with pytest.raises(RuntimeError) as raised:
            await asyncio.wait_for(parent, timeout=2)
        assert is_durable_subagent_submission_unsettled(
            raised.value,
            parent_session_id="durable-timeout-unsettled-parent",
        )
        persisted_parent = await sessions.load("durable-timeout-unsettled-parent")
        assert persisted_parent is not None
        assert persisted_parent.status is SessionStatus.RUNNING
        checkpoint = await sessions.load_checkpoint("durable-timeout-unsettled-parent")
        assert checkpoint is not None
        assert "pending_tool_round" in checkpoint
        assert not any(
            event.type is EventType.TOOL_CALL_FAILED
            for event in await sessions.load_events("durable-timeout-unsettled-parent")
        )

        restarted_dispatcher = TaskStoreDispatcher(tasks)
        restarted_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=restarted_dispatcher,
            tool_timeout_seconds=0.01,
            enable_logging=False,
        )
        restarted_app.register_provider(provider, default=True)
        _register_durable_subagent_agents(restarted_app)
        recovery = await restarted_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-timeout-unsettled-parent",
                reason="recover timeout during ambiguous durable publication",
            )
        )
        assert recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        children = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-timeout-unsettled-parent")
            )
        ).sessions
        queued = await tasks.list_tasks(TaskQuery())
        assert len(children) == 1
        assert len(queued) == 1
        assert queued[0].type == restarted_dispatcher.prepared_subagent_task_type

        handle = await restarted_dispatcher.process_next(
            restarted_app,
            worker_id="recovery-worker",
        )
        assert handle is not None
        assert handle.status.value == "completed"
        assert (
            len(
                [
                    request
                    for request in provider.requests
                    if request.messages[0].content[0].text == "durable child task"
                ]
            )
            == 1
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    ("tool_timeout_seconds", "settlement_delay_s"),
    [(1.0, 0.0), (0.02, 0.03)],
    ids=["before-timeout", "after-timeout"],
)
def test_external_cancellation_remains_authoritative_over_unsettled_timeout(
    tool_timeout_seconds: float,
    settlement_delay_s: float,
) -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = _BlockingFailOnceDurableTaskReadStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            tool_timeout_seconds=tool_timeout_seconds,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        parent = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-timeout-unsettled-cancel-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        )
        await asyncio.wait_for(tasks.read_started.wait(), timeout=1)
        parent.cancel("external cancellation during unsettled durable submission")
        await asyncio.sleep(settlement_delay_s)
        tasks.release_read.set()

        with pytest.raises(
            asyncio.CancelledError,
            match="external cancellation during unsettled durable submission",
        ) as raised:
            await asyncio.wait_for(parent, timeout=2)
        assert parent.cancelling() == 1
        assert parent.cancelled() is True
        assert str(raised.value) == "external cancellation during unsettled durable submission"
        checkpoint = await sessions.load_checkpoint("durable-timeout-unsettled-cancel-parent")
        assert checkpoint is not None
        assert "durable_subagent_submission_seeds" not in checkpoint
        assert "durable_subagent_submissions" in checkpoint
        submissions = checkpoint["durable_subagent_submissions"]
        receipt = durable_subagent_submission_receipt_from_checkpoint(
            checkpoint,
            idempotency_key=next(iter(submissions)),
        )
        assert receipt is not None and receipt.outcome == "submitted"
        failed = [
            event
            for event in await sessions.load_events("durable-timeout-unsettled-cancel-parent")
            if event.type is EventType.TOOL_CALL_FAILED
        ]
        assert len(failed) == 1
        assert failed[0].payload["interrupted"] is True

    asyncio.run(run())


def test_external_cancellation_wins_when_durable_submission_outlasts_tool_timeout() -> None:
    async def run() -> None:
        sessions = _BlockingDurableChildCreationStore()
        tasks = InMemoryTaskStore()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=TaskStoreDispatcher(tasks),
            tool_timeout_seconds=0.02,
            enable_logging=False,
        )
        app.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(app)
        parent = asyncio.create_task(
            _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-submission-timeout-cancel-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        )
        await asyncio.wait_for(sessions.child_creation_started.wait(), timeout=1)
        parent.cancel("external cancellation during durable submission")
        await asyncio.sleep(0.03)
        parent.cancel("repeated external cancellation during durable submission")
        await asyncio.sleep(0)
        sessions.release_child_creation.set()

        with pytest.raises(
            asyncio.CancelledError,
            match="external cancellation during durable submission",
        ):
            await parent
        assert parent.cancelling() == 2
        assert parent.cancelled() is True
        children = (
            await sessions.list_sessions(
                SessionQuery(parent_session_id="durable-submission-timeout-cancel-parent")
            )
        ).sessions
        queued = await tasks.list_tasks(TaskQuery())
        assert len(children) == 1
        assert len(queued) == 1

    asyncio.run(run())


def test_cancelled_queue_worker_reclaims_child_without_second_provider_dispatch() -> None:
    async def run() -> None:
        sessions = InMemorySessionStore()
        tasks = InMemoryTaskStore()
        dispatcher = _CrashAfterDurableTaskDispatcher(tasks)
        provider = _BlockingDurableChildProvider()
        app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=dispatcher,
            enable_logging=False,
        )
        app.register_provider(provider, default=True)
        _register_durable_subagent_agents(app)
        with pytest.raises(_SimulatedWorkerLoss):
            await _collect(
                app.run(
                    RunRequest(
                        agent_name="parent",
                        session_id="durable-worker-cancel-parent",
                        messages=[Message.text("user", "parent task")],
                    )
                )
            )
        worker_dispatcher = TaskStoreDispatcher(tasks)
        worker_app = CayuApp(
            session_store=sessions,
            task_store=tasks,
            dispatcher=worker_dispatcher,
            enable_logging=False,
        )
        worker_app.register_provider(provider, default=True)
        _register_durable_subagent_agents(worker_app)
        queued = (await tasks.list_tasks(TaskQuery()))[0]
        child_id = queued.input["dispatch"]["prepared_subagent"]["authority"]["child_session_id"]

        processing = asyncio.create_task(
            worker_dispatcher.process_next(
                worker_app,
                worker_id="cancelled-child-worker",
            )
        )
        await asyncio.wait_for(provider.child_started.wait(), timeout=1)
        pending_recovery = await worker_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-worker-cancel-parent",
                reason="inspect claimed running durable child",
            )
        )
        assert pending_recovery.actions == (IncompleteSessionRecoveryAction.SKIPPED_ACTIVE,)
        processing.cancel("worker shutdown")
        try:
            await processing
        except asyncio.CancelledError as exc:
            # Provider-boundary cancellation text is credential-safe and canonical.
            assert str(exc) == "Provider operation cancelled"
        else:
            raise AssertionError("Queue worker cancellation did not propagate.")
        assert processing.cancelling() == 1
        assert processing.cancelled() is True
        claimed = await tasks.load_task(queued.id)
        assert claimed is not None
        assert claimed.status is TaskStatus.CLAIMED
        await tasks.release_task(queued.id, "cancelled-child-worker")

        replayed = await worker_dispatcher.process_next(
            worker_app,
            worker_id="replacement-child-worker",
        )
        assert replayed is not None
        assert replayed.status.value == "interrupted"
        child = await sessions.load(child_id)
        assert child is not None
        assert child.status is SessionStatus.INTERRUPTED
        terminal_task = await tasks.load_task(queued.id)
        assert terminal_task is not None
        assert terminal_task.status is TaskStatus.COMPLETED
        terminal_recovery = await worker_app.recover_incomplete_session(
            IncompleteSessionRecoveryRequest(
                session_id="durable-worker-cancel-parent",
                reason="attach interrupted durable child",
            )
        )
        assert IncompleteSessionRecoveryAction.REPAIRED_TOOL_ROUND in (terminal_recovery.actions)
        assert (
            len(
                [
                    request
                    for request in provider.requests
                    if request.messages[0].content[0].text == "durable child task"
                ]
            )
            == 1
        )

    asyncio.run(run())


def test_postgres_concurrent_submission_and_stale_worker_converge(
    postgres_dsn: str,
) -> None:
    async def run() -> None:
        bootstrap_sessions = _CrashBeforeDurableChildPostgresStore(postgres_dsn)
        bootstrap_tasks = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        await bootstrap_sessions.list_sessions(SessionQuery())
        await bootstrap_tasks.list_tasks(TaskQuery())
        async with await psycopg.AsyncConnection.connect(postgres_dsn) as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(
                    "TRUNCATE TABLE cayu_sessions, cayu_tasks RESTART IDENTITY CASCADE"
                )
            await conn.commit()
        task_type = "test.durable-subagent.postgres"
        bootstrap = CayuApp(
            session_store=bootstrap_sessions,
            task_store=bootstrap_tasks,
            dispatcher=TaskStoreDispatcher(bootstrap_tasks, task_type=task_type),
            enable_logging=False,
        )
        bootstrap.register_provider(_DurableSubagentProvider(), default=True)
        _register_durable_subagent_agents(bootstrap)
        try:
            with pytest.raises(_SimulatedWorkerLoss):
                await _collect(
                    bootstrap.run(
                        RunRequest(
                            agent_name="parent",
                            session_id="postgres-durable-parent",
                            messages=[Message.text("user", "parent task")],
                        )
                    )
                )
        finally:
            await bootstrap_sessions.close()
            await bootstrap_tasks.close()

        sessions_a = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        sessions_b = PostgresSessionStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        tasks_a = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        tasks_b = PostgresTaskStore(
            postgres_dsn,
            min_size=1,
            max_size=4,
            schema_mode=SchemaMode.CREATE,
        )
        dispatcher_a = TaskStoreDispatcher(tasks_a, task_type=task_type, lease_seconds=1)
        dispatcher_b = TaskStoreDispatcher(tasks_b, task_type=task_type, lease_seconds=1)
        app_a = CayuApp(
            session_store=sessions_a,
            task_store=tasks_a,
            dispatcher=dispatcher_a,
            enable_logging=False,
        )
        app_b = CayuApp(
            session_store=sessions_b,
            task_store=tasks_b,
            dispatcher=dispatcher_b,
            enable_logging=False,
        )
        provider = _DurableSubagentProvider()
        app_a.register_provider(provider, default=True)
        app_b.register_provider(provider, default=True)
        _register_durable_subagent_agents(app_a)
        _register_durable_subagent_agents(app_b)
        try:
            checkpoint = await sessions_a.load_checkpoint("postgres-durable-parent")
            assert checkpoint is not None
            submissions = checkpoint.get("durable_subagent_submissions", {})
            assert type(submissions) is dict and len(submissions) == 1
            intent = durable_subagent_submission_from_checkpoint(
                checkpoint,
                idempotency_key=next(iter(submissions)),
            )
            assert intent is not None
            first, second = await asyncio.gather(
                app_a._ensure_durable_subagent_submission(intent),
                app_b._ensure_durable_subagent_submission(intent),
            )
            assert first[0].id == second[0].id == intent.child_session_id
            assert first[1].metadata["queue_task_id"] == intent.queue_task_id
            assert second[1].metadata["queue_task_id"] == intent.queue_task_id
            children = (
                await sessions_a.list_sessions(
                    SessionQuery(parent_session_id=intent.parent_session_id)
                )
            ).sessions
            queued = await tasks_a.list_tasks(TaskQuery(type=intent.queue_task_type))
            assert [child.id for child in children] == [intent.child_session_id]
            assert [task.id for task in queued] == [intent.queue_task_id]

            claimed_a = await tasks_a.claim_task(
                "stale-worker",
                TaskQuery(type=intent.queue_task_type),
                lease_seconds=1,
            )
            assert claimed_a is not None
            await asyncio.sleep(1.1)
            reclaimed = await tasks_b.reclaim_expired(query=TaskQuery(type=intent.queue_task_type))
            assert [task.id for task in reclaimed] == [intent.queue_task_id]
            claimed_b = await tasks_b.claim_task(
                "replacement-worker",
                TaskQuery(type=intent.queue_task_type),
                lease_seconds=30,
            )
            assert claimed_b is not None
            with pytest.raises(TaskClaimLost):
                await tasks_a.fail_task(
                    intent.queue_task_id,
                    {"reason": "stale worker must not win"},
                    worker_id="stale-worker",
                )
            await tasks_b.release_task(intent.queue_task_id, "replacement-worker")

            completed = await dispatcher_b.process_next(
                app_b,
                worker_id="terminal-worker",
            )
            assert completed is not None
            assert completed.status.value == "completed"
            child = await sessions_a.load(intent.child_session_id)
            task = await tasks_a.load_task(intent.queue_task_id)
            assert child is not None and child.status is SessionStatus.COMPLETED
            assert task is not None and task.status is TaskStatus.COMPLETED
            assert len(provider.requests) == 1
        finally:
            await sessions_a.close()
            await sessions_b.close()
            await tasks_a.close()
            await tasks_b.close()

    asyncio.run(run())
