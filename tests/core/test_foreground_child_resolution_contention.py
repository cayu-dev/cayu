"""Competing public child resolutions cannot duplicate a parent continuation."""

import asyncio
from uuid import uuid4

import pytest
from tests.core.test_foreground_subagent_recovery import _identity, _Provider
from tests.core.test_tool_round_execution_identities import _RecordingTool

from cayu import (
    AgentSpec,
    CayuApp,
    InMemorySessionStore,
    Message,
    PostgresSessionStore,
    RunRequest,
    SessionQuery,
    SessionStatus,
    SQLiteSessionStore,
    SubagentSpec,
    SubagentTool,
    ToolApprovalDecision,
)
from cayu.providers import ModelStreamEvent
from cayu.runtime import (
    IncompleteSessionRecoveryRequest,
    InvocationLifecycleCommandConflict,
    ToolApprovalRequest,
    UserInputResponse,
)
from cayu.runtime._event_projection import public_event_id
from cayu.runtime._invocation_lifecycle import RebindInvocationCommand
from cayu.runtime.sessions import (
    PersistedEventSideEffectStatus,
    SessionRuntimePublicationConflict,
    SessionStatusConflict,
)
from cayu.runtime.tool_policy import AlwaysRequireApprovalToolPolicy
from cayu.storage.migrations import SchemaMode
from cayu.tools.user_input import UserInputTool


class _ContendedTool(_RecordingTool):
    spec = _RecordingTool.spec.model_copy(
        update={"execution_profile_identity": _identity("contention-record")}
    )


def _app(store, provider, protected, action):
    app = CayuApp(session_store=store, enable_logging=False)
    app.register_provider(provider, default=True)
    app.register_agent(
        AgentSpec(name="parent", model="test"),
        tools=[
            SubagentTool(
                app,
                agents={"child": SubagentSpec(agent_name="child")},
                execution_profile_identity=_identity("child-resolution-contention"),
            ),
        ],
    )
    app.register_agent(
        AgentSpec(name="child", model="test"),
        tools=[UserInputTool()] if action == "input" else [protected],
        tool_policy=None
        if action == "input"
        else AlwaysRequireApprovalToolPolicy(tools=["record"]),
    )
    return app


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize("conflicting", [False, True], ids=["duplicate", "conflicting"])
def test_competing_child_resolutions_attach_one_parent_outcome(
    tmp_path, monkeypatch, backend, action, conflicting, request
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    parent_id = f"resolution-parent-{uuid4().hex}"

    async def scenario():
        def new_store():
            if backend == "memory":
                return InMemorySessionStore()
            if backend == "sqlite":
                return SQLiteSessionStore(tmp_path / "resolution-contention.sqlite")
            return PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

        store = new_store()
        second_store = store if backend == "memory" else new_store()
        protected = _ContendedTool()
        provider = _Provider(
            [
                [
                    ModelStreamEvent.tool_call(
                        id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                    ),
                    ModelStreamEvent.completed(),
                ],
                [
                    ModelStreamEvent.tool_call(
                        id="action",
                        name="ask_user" if action == "input" else "record",
                        arguments={"question": "Which value?"}
                        if action == "input"
                        else {"value": 7},
                    ),
                    ModelStreamEvent.completed(),
                ],
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )
        app = _app(store, provider, protected, action)
        peer = _app(second_store, provider, protected, action)
        ready = asyncio.Event()
        proceed = asyncio.Event()
        tasks = []
        try:
            _ = [
                event
                async for event in app.run(
                    RunRequest(
                        session_id=parent_id,
                        agent_name="parent",
                        messages=[Message.text("user", "go")],
                    )
                )
            ]
            children = await store.list_sessions(SessionQuery(parent_session_id=parent_id))
            assert len(children.sessions) == 1
            child = children.sessions[0]
            parent_checkpoint = await store.load_checkpoint(parent_id)
            assert parent_checkpoint is not None
            wait = parent_checkpoint["foreground_child_wait"]
            child_events = await store.load_events(child.id)
            pending = next(
                event
                for event in child_events
                if event.type
                == (
                    "session.awaiting_user_input"
                    if action == "input"
                    else "tool.call.approval_requested"
                )
            )
            claims = []

            async def overlap_claims(command, apply_command):
                if isinstance(command, RebindInvocationCommand) and command.session_id == child.id:
                    claims.append(command.expected_run_epoch)
                    if len(claims) == 1:
                        ready.set()
                        await proceed.wait()
                    else:
                        proceed.set()
                return await apply_command(command)

            for target in (app, peer):
                apply_command = target._runtime_session_store.apply_invocation_lifecycle_command

                async def wrapped(command, apply_command=apply_command):
                    return await overlap_claims(command, apply_command)

                monkeypatch.setattr(
                    target._runtime_session_store, "apply_invocation_lifecycle_command", wrapped
                )

            async def resolve(index):
                try:
                    resolver = (app, peer)[index]
                    if action == "input":
                        stream = resolver.resolve_user_input(
                            UserInputResponse(
                                session_id=child.id,
                                input_id=pending.payload["input_id"],
                                answer="second" if conflicting and index else "first",
                            )
                        )
                    else:
                        stream = resolver.resolve_tool_approval(
                            ToolApprovalRequest(
                                session_id=child.id,
                                approval_id=pending.payload["approval"]["approval_id"],
                                tool_round_id=pending.payload["tool_round_id"],
                                tool_call_id=pending.payload["tool_call_id"],
                                decision=ToolApprovalDecision.DENY
                                if conflicting and index
                                else ToolApprovalDecision.APPROVE,
                            )
                        )
                    return [event async for event in stream]
                finally:
                    # An early typed conflict may reject the second caller before
                    # dispatch. It must still release the deliberately held first.
                    if index:
                        proceed.set()

            tasks.append(asyncio.create_task(resolve(0)))
            await asyncio.wait_for(ready.wait(), timeout=20)
            assert not tasks[0].done()
            tasks.append(asyncio.create_task(resolve(1)))
            outcomes = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=40
            )
            assert len(claims) == 2 and claims[0] == claims[1]
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            assert len(failures) <= 1, failures
            assert all(
                isinstance(
                    error,
                    (
                        SessionStatusConflict,
                        SessionRuntimePublicationConflict,
                        InvocationLifecycleCommandConflict,
                    ),
                )
                for error in failures
            ), failures
            winners = [
                index
                for index, outcome in enumerate(outcomes)
                if not isinstance(outcome, BaseException)
            ]
            if conflicting:
                assert len(winners) == 1
            assert await app.drain_background_interruptions(timeout_s=10)
            assert await peer.drain_background_interruptions(timeout_s=10)
            parent = await store.load(parent_id)
            finished_child = await store.load(child.id)
            assert parent is not None and parent.status is SessionStatus.COMPLETED
            assert finished_child is not None and finished_child.status is SessionStatus.COMPLETED
            assert len(provider.requests) == 4
            expected_value = "second" if conflicting and winners == [1] else "first"
            assert protected.values == (
                [7] if action == "approval" and expected_value == "first" else []
            )
            child_events = await store.load_events(child.id)
            child_tools = [
                event
                for event in child_events
                if event.type in {"tool.call.completed", "tool.call.failed", "tool.call.blocked"}
            ]
            denied = action == "approval" and expected_value == "second"
            # Denial has its own terminal approval event; no tool ran, so the
            # normal contract emits no completed/failed tool-execution event.
            assert len(child_tools) == (0 if denied else 1)
            assert (
                len(
                    [
                        message
                        for message in await store.load_transcript(child.id)
                        if message.role == "tool"
                    ]
                )
                == 1
            )
            if action == "input":
                assert child_tools[0].payload["result"]["content"] == expected_value
            else:
                assert (
                    sum(
                        event.type in {"tool.call.approved", "tool.call.approval_denied"}
                        for event in child_events
                    )
                    == 1
                )
            assert sum(event.type == "interaction.started" for event in child_events) == 1
            assert sum(event.type == "interaction.completed" for event in child_events) == 1
            parent_events = await store.load_events(parent_id)
            terminal_tools = [
                event
                for event in parent_events
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(terminal_tools) == 1
            assert (
                terminal_tools[0].payload["tool_round_id"] == wait["parent_effect"]["tool_round_id"]
            )
            assert (
                len(
                    [
                        message
                        for message in await store.load_transcript(parent_id)
                        if message.role == "tool"
                    ]
                )
                == 1
            )
            assert sum(event.type == "interaction.started" for event in parent_events) == 1
            assert sum(event.type == "interaction.completed" for event in parent_events) == 1
            # Re-delivery is idempotent after both competing callers have left.
            await app.recover_persisted_event_side_effects()
            assert await store.load_events(parent_id) == parent_events
            assert len(provider.requests) == 4
        finally:
            proceed.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            assert await app.drain_background_interruptions(timeout_s=10)
            assert await peer.drain_background_interruptions(timeout_s=10)
            if backend != "memory":
                await store.close()
                await second_store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("backend", ["memory", "sqlite", "postgres"])
@pytest.mark.parametrize("action", ["approval", "input"])
@pytest.mark.parametrize("race", ["delivery", "recovery"])
def test_competing_child_wakeups_and_parent_recovery_preserve_exact_attachment(
    tmp_path, monkeypatch, backend, action, race, request
):
    dsn = request.getfixturevalue("postgres_dsn") if backend == "postgres" else None
    parent_id = f"delivery-parent-{uuid4().hex}"
    sibling_id = f"delivery-sibling-{uuid4().hex}"

    async def scenario():
        def new_store():
            if backend == "memory":
                return InMemorySessionStore()
            if backend == "sqlite":
                return SQLiteSessionStore(tmp_path / "wakeups.sqlite")
            return PostgresSessionStore(dsn, schema_mode=SchemaMode.CREATE)

        store = new_store()
        stores = [store] + [store if backend == "memory" else new_store() for _ in range(2)]
        opening = [
            [
                ModelStreamEvent.tool_call(
                    id="spawn", name="subagent", arguments={"agent": "child", "task": "work"}
                ),
                ModelStreamEvent.completed(),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="action",
                    name="ask_user" if action == "input" else "record",
                    arguments={"question": "Which value?"} if action == "input" else {"value": 7},
                ),
                ModelStreamEvent.completed(),
            ],
        ]
        provider = _Provider(
            opening
            + opening
            + [
                [ModelStreamEvent.text_delta("child finished"), ModelStreamEvent.completed()],
                [ModelStreamEvent.text_delta("parent finished"), ModelStreamEvent.completed()],
            ]
        )
        protected = _ContendedTool()
        apps = [_app(target, provider, protected, action) for target in stores]
        app, first, second = apps
        all_claimants_ready = asyncio.Event()
        parent_claimed = asyncio.Event()
        release_parent = asyncio.Event()
        tasks = []
        try:
            # Identical provider call IDs in unrelated parents are deliberately
            # insufficient to select a continuation.
            for session_id in (parent_id, sibling_id):
                events = [
                    event
                    async for event in app.run(
                        RunRequest(
                            session_id=session_id,
                            agent_name="parent",
                            messages=[Message.text("user", "go")],
                        )
                    )
                ]
                assert events[-1].type == "session.interrupted"
            checkpoint = await store.load_checkpoint(parent_id)
            wait = checkpoint["foreground_child_wait"]
            child_id = wait["child_session_id"]
            sibling = await store.load(sibling_id)
            sibling_checkpoint = await store.load_checkpoint(sibling_id)
            sibling_events = await store.load_events(sibling_id)
            sibling_child_id = sibling_checkpoint["foreground_child_wait"]["child_session_id"]
            sibling_child = await store.load(sibling_child_id)
            sibling_child_events = await store.load_events(sibling_child_id)
            child_events = await store.load_events(child_id)
            pending = next(
                event
                for event in child_events
                if event.type
                == (
                    "session.awaiting_user_input"
                    if action == "input"
                    else "tool.call.approval_requested"
                )
            )
            continue_parent = app._event_writer._continue_foreground_parent

            async def defer_terminal(claim):
                if claim.session_id == child_id and claim.event.type == "session.completed":
                    return False
                return await continue_parent(claim)

            # Stop only post-commit delivery. The public resolution still owns
            # action closure, child execution, terminal publication, and release.
            with monkeypatch.context() as patch:
                patch.setattr(app._event_writer, "_continue_foreground_parent", defer_terminal)
                if action == "input":
                    stream = app.resolve_user_input(
                        UserInputResponse(
                            session_id=child_id, input_id=pending.payload["input_id"], answer="yes"
                        )
                    )
                else:
                    stream = app.resolve_tool_approval(
                        ToolApprovalRequest(
                            session_id=child_id,
                            approval_id=pending.payload["approval"]["approval_id"],
                            tool_round_id=pending.payload["tool_round_id"],
                            tool_call_id=pending.payload["tool_call_id"],
                            decision=ToolApprovalDecision.APPROVE,
                        )
                    )
                resolved = [event async for event in stream]
                assert resolved[-1].type == "session.completed"
                assert await app.drain_background_interruptions(timeout_s=10)
            assert len(provider.requests) == 5
            assert (await store.load(parent_id)).status is SessionStatus.INTERRUPTED
            terminal = next(
                event
                for event in await store.load_events(child_id)
                if event.type == "session.completed"
            )
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=child_id, event_id=terminal.id
            )
            assert delivery.status is PersistedEventSideEffectStatus.PENDING
            attempts = []
            claims = []
            owners = []

            for index, contender in enumerate((first, second)):
                runtime_store = contender._runtime_session_store
                claim_delivery = runtime_store.claim_persisted_event_side_effect
                list_deliveries = runtime_store.list_persisted_event_side_effect_deliveries
                deliver = contender._event_writer._deliver_persisted_side_effect_claim

                async def overlap_candidates(
                    *, list_deliveries=list_deliveries, index=index, **kwargs
                ):
                    candidates = await list_deliveries(**kwargs)
                    if any(
                        item.session_id == child_id and item.event_id == terminal.id
                        for item in candidates
                    ):
                        attempts.append(index)
                        if len(attempts) == 2:
                            all_claimants_ready.set()
                        await all_claimants_ready.wait()
                    return candidates

                async def overlap_delivery(*, claim_delivery=claim_delivery, **kwargs):
                    exact = kwargs["session_id"] == child_id and kwargs["event_id"] == terminal.id
                    claim = await claim_delivery(**kwargs)
                    if exact:
                        claims.append(claim)
                    return claim

                async def hold_delivery(claim, deliver=deliver, index=index):
                    if claim.session_id == child_id and claim.event_id == terminal.id:
                        owners.append(index)
                        parent_claimed.set()
                        await release_parent.wait()
                    return await deliver(claim)

                if race == "delivery":
                    monkeypatch.setattr(
                        runtime_store,
                        "list_persisted_event_side_effect_deliveries",
                        overlap_candidates,
                    )
                    monkeypatch.setattr(
                        runtime_store, "claim_persisted_event_side_effect", overlap_delivery
                    )
                    monkeypatch.setattr(
                        contender._event_writer,
                        "_deliver_persisted_side_effect_claim",
                        hold_delivery,
                    )

            if race == "delivery":
                tasks = [
                    asyncio.create_task(contender.recover_persisted_event_side_effects())
                    for contender in (first, second)
                ]
                await asyncio.wait_for(parent_claimed.wait(), timeout=30)
                assert sorted(attempts) == [0, 1]
                assert len(owners) == 1
                await asyncio.wait_for(asyncio.shield(tasks[1 - owners[0]]), timeout=20)
                assert len(claims) == 2
                assert sum(claim is not None for claim in claims) == 1
            else:
                reserve = first._runtime_session_store.reserve_stalled_run_recovery

                async def hold_recovery(*args, **kwargs):
                    claim = await reserve(*args, **kwargs)
                    if claim is not None:
                        parent_claimed.set()
                        await release_parent.wait()
                    return claim

                monkeypatch.setattr(
                    first._runtime_session_store, "reserve_stalled_run_recovery", hold_recovery
                )
                recovery = IncompleteSessionRecoveryRequest(session_id=parent_id)
                tasks = [asyncio.create_task(first.recover_incomplete_session(recovery))]
                await asyncio.wait_for(parent_claimed.wait(), timeout=30)
                competing = await asyncio.wait_for(
                    second.recover_incomplete_session(recovery), timeout=20
                )
                assert competing.actions == ("skipped_active",)
            assert len(provider.requests) == 5
            assert not any(
                event.type in {"tool.call.completed", "tool.call.failed"}
                for event in await store.load_events(parent_id)
            )
            release_parent.set()
            outcomes = await asyncio.wait_for(asyncio.gather(*tasks), timeout=40)
            if race == "delivery":
                assert (
                    sum(
                        event.session_id == child_id
                        and event.id == public_event_id(delivery.event_sequence)
                        for outcome in outcomes
                        for event in outcome
                    )
                    == 1
                )
            else:
                await second.recover_persisted_event_side_effects()
            for target in apps:
                assert await target.drain_background_interruptions(timeout_s=10)
            assert (await store.load(parent_id)).status is SessionStatus.COMPLETED
            assert (await store.load(child_id)).status is SessionStatus.COMPLETED
            assert len(provider.requests) == 6
            assert protected.values == ([7] if action == "approval" else [])
            parent_events = await store.load_events(parent_id)
            parent_transcript = await store.load_transcript(parent_id)
            tool_events = [
                event
                for event in parent_events
                if event.type in {"tool.call.completed", "tool.call.failed"}
            ]
            assert len(tool_events) == 1
            assert tool_events[0].type == "tool.call.completed"
            assert tool_events[0].interaction_id == wait["parent_effect"]["interaction_id"]
            assert tool_events[0].payload["tool_round_id"] == wait["parent_effect"]["tool_round_id"]
            assert tool_events[0].payload["tool_call_id"] == wait["parent_effect"]["tool_call_id"]
            assert tool_events[0].payload["result"]["structured"]["child_session_id"] == child_id
            assert sum(message.role == "tool" for message in parent_transcript) == 1
            assert sum(event.type == "interaction.started" for event in parent_events) == 1
            assert sum(event.type == "interaction.completed" for event in parent_events) == 1
            delivery = await store.get_persisted_event_side_effect_delivery(
                session_id=child_id, event_id=terminal.id
            )
            assert delivery.status is PersistedEventSideEffectStatus.DELIVERED
            for contender in (first, second):
                await contender.recover_persisted_event_side_effects()
                await contender.recover_incomplete_session(
                    IncompleteSessionRecoveryRequest(session_id=parent_id)
                )
            assert await store.load_events(parent_id) == parent_events
            assert await store.load_transcript(parent_id) == parent_transcript
            assert len(provider.requests) == 6
            assert await store.load(sibling_id) == sibling
            assert await store.load_checkpoint(sibling_id) == sibling_checkpoint
            assert await store.load_events(sibling_id) == sibling_events
            assert await store.load(sibling_child_id) == sibling_child
            assert await store.load_events(sibling_child_id) == sibling_child_events
        finally:
            all_claimants_ready.set()
            release_parent.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for target in apps:
                assert await target.drain_background_interruptions(timeout_s=10)
            if backend != "memory":
                for target in stores:
                    await target.close()

    asyncio.run(scenario())
