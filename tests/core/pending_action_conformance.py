from __future__ import annotations

from typing import Any

from cayu.core import Event, EventType, Message
from cayu.runtime import RunRequest, SessionIdentity, SessionStatus, SessionStore
from cayu.runtime.sessions import (
    MAX_PENDING_ACTION_TOOL_CALLS,
    PendingActionKind,
    PendingActionQuery,
)


def _identity() -> SessionIdentity:
    return SessionIdentity(provider_name="fake", model="fake-model")


def _pending_call(
    tool_call_id: str,
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "arguments": arguments or {},
        "policy_decision": None,
        "reason": None,
        "metadata": {},
        "active_taint_labels": [],
    }


def _approval_checkpoint(
    approval_id: str,
    tool_call_id: str,
    tool_name: str,
    *,
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "pending_tool_approval": {
            "approval_id": approval_id,
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "arguments": arguments or {},
            "agent_name": "assistant",
            "tool_calls": [_pending_call(tool_call_id, tool_name, arguments=arguments)],
        }
    }


def _input_checkpoint(
    input_id: str,
    tool_call_id: str,
    question: str,
) -> dict[str, Any]:
    return {
        "pending_user_input": {
            "input_id": input_id,
            "tool_call_id": tool_call_id,
            "tool_name": "ask_user",
            "question": question,
            "options": ["yes", "no"],
            "arguments": {},
            "agent_name": "assistant",
            "tool_calls": [_pending_call(tool_call_id, "ask_user")],
        }
    }


def _round_checkpoint(round_id: str, tool_call_id: str) -> dict[str, Any]:
    return {
        "pending_tool_round": {
            "round_id": round_id,
            "agent_name": "assistant",
            "tool_calls": [_pending_call(tool_call_id, "charge")],
        }
    }


async def assert_pending_action_store_conformance(store: SessionStore) -> None:
    async def create(
        session_id: str,
        *,
        status: SessionStatus,
        events: list[Event],
        checkpoint: dict[str, Any],
    ) -> None:
        await store.create(
            RunRequest(
                agent_name="assistant",
                session_id=session_id,
                messages=[Message.text("user", "hello")],
            ),
            identity=_identity(),
        )
        await store.append_events(session_id, events)
        await store.checkpoint(session_id, checkpoint)
        await store.update_status(session_id, status)

    await create(
        "conformance_approval",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_approval_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="conformance_approval",
                payload={
                    "approval": {
                        "approval_id": "conformance_approval_id",
                        "tool_name": "deploy",
                        "arguments": {},
                    }
                },
            ),
            Event(
                id="conformance_alternate_approval_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="conformance_approval",
                payload={
                    "approval": {
                        "approval_id": "conformance_alternate_approval_id",
                        "tool_name": "rollback",
                        "arguments": {},
                    }
                },
            ),
        ],
        checkpoint=_approval_checkpoint(
            "conformance_approval_id",
            "conformance_approval_call",
            "deploy",
        ),
    )
    await create(
        "conformance_input",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_input_event",
                type=EventType.SESSION_AWAITING_USER_INPUT,
                session_id="conformance_input",
                payload={
                    "input_id": "conformance_input_id",
                    "tool_call_id": "conformance_input_call",
                    "question": "Deploy?",
                    "options": ["yes", "no"],
                },
            )
        ],
        checkpoint=_input_checkpoint(
            "conformance_input_id",
            "conformance_input_call",
            "Deploy?",
        ),
    )
    await create(
        "conformance_approval_recovery",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_approval_recovery_event",
                type=EventType.SESSION_INTERRUPTED,
                session_id="conformance_approval_recovery",
                payload={
                    "manual_recovery_required": True,
                    "approval_id": "conformance_recovery_approval_id",
                    "tool_call_id": "conformance_recovery_approval_call",
                    "tool_name": "refund",
                },
            )
        ],
        checkpoint=_approval_checkpoint(
            "conformance_recovery_approval_id",
            "conformance_recovery_approval_call",
            "refund",
        ),
    )
    await create(
        "conformance_input_recovery",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_input_recovery_event",
                type=EventType.SESSION_INTERRUPTED,
                session_id="conformance_input_recovery",
                payload={
                    "manual_recovery_required": True,
                    "user_input": {
                        "input_id": "conformance_recovery_input_id",
                        "tool_call_id": "conformance_recovery_input_call",
                    },
                    "tool_call_id": "conformance_recovery_input_call",
                    "tool_name": "ask_user",
                },
            )
        ],
        checkpoint=_input_checkpoint(
            "conformance_recovery_input_id",
            "conformance_recovery_input_call",
            "Continue?",
        ),
    )
    await create(
        "conformance_round_recovery",
        status=SessionStatus.FAILED,
        events=[
            Event(
                id="conformance_round_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_round_recovery",
                payload={
                    "tool_round_id": "conformance_round_id",
                    "tool_call_id": "conformance_round_call",
                },
            ),
            Event(
                id="conformance_round_failed",
                type=EventType.SESSION_FAILED,
                session_id="conformance_round_recovery",
                payload={
                    "manual_recovery_required": True,
                    "tool_round_id": "conformance_round_id",
                    "tool_call_id": "conformance_round_call",
                },
            ),
        ],
        checkpoint=_round_checkpoint("conformance_round_id", "conformance_round_call"),
    )
    await create(
        "conformance_stale_approval",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_stale_approval_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="conformance_stale_approval",
                payload={
                    "approval": {
                        "approval_id": "conformance_stale_approval_id",
                        "tool_name": "deploy",
                    }
                },
            ),
            Event(
                id="conformance_stale_barrier",
                type=EventType.SESSION_RESUMED,
                session_id="conformance_stale_approval",
            ),
        ],
        checkpoint=_approval_checkpoint(
            "conformance_stale_approval_id",
            "conformance_stale_approval_call",
            "deploy",
        ),
    )
    await create(
        "conformance_malformed",
        status=SessionStatus.INTERRUPTED,
        events=[],
        checkpoint={"pending_tool_round": {"tool_calls": "not-an-array"}},
    )
    await create(
        "conformance_blank_approval_id",
        status=SessionStatus.INTERRUPTED,
        events=[],
        checkpoint={"pending_tool_approval": {"approval_id": "   "}},
    )
    await create(
        "conformance_blank_input_id",
        status=SessionStatus.INTERRUPTED,
        events=[],
        checkpoint={"pending_user_input": {"input_id": "   "}},
    )
    await create(
        "conformance_blank_round_id",
        status=SessionStatus.FAILED,
        events=[],
        checkpoint={
            "pending_tool_round": {
                "round_id": "   ",
                "agent_name": "assistant",
                "tool_calls": [],
            }
        },
    )
    await create(
        "conformance_malformed_terminal",
        status=SessionStatus.FAILED,
        events=[
            Event(
                id="conformance_malformed_terminal_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_malformed_terminal",
                payload={
                    "tool_round_id": "conformance_malformed_terminal_round",
                    "tool_call_id": "conformance_malformed_terminal_call",
                },
            ),
            Event(
                id="conformance_malformed_terminal_completed",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_malformed_terminal",
                payload={
                    "tool_round_id": "conformance_malformed_terminal_round",
                    "tool_call_id": "conformance_malformed_terminal_call",
                },
            ),
            Event(
                id="conformance_malformed_terminal_other_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_malformed_terminal",
                payload={
                    "tool_round_id": "conformance_malformed_terminal_round",
                    "tool_call_id": "conformance_malformed_terminal_other_call",
                },
            ),
        ],
        checkpoint={
            "pending_tool_round": {
                "round_id": "conformance_malformed_terminal_round",
                "agent_name": "assistant",
                "tool_calls": [
                    _pending_call("conformance_malformed_terminal_call", "charge"),
                    _pending_call("conformance_malformed_terminal_other_call", "charge"),
                ],
            }
        },
    )
    await create(
        "conformance_normalized_lookup",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_normalized_lookup_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="conformance_normalized_lookup",
                payload={
                    "approval_id": "   ",
                    "approval": {
                        "approval_id": "conformance_normalized_lookup_id",
                        "tool_name": "deploy",
                    },
                },
            )
        ],
        checkpoint=_approval_checkpoint(
            "conformance_normalized_lookup_id",
            "conformance_normalized_lookup_call",
            "deploy",
        ),
    )
    await create(
        "conformance_completed_with_pending_state",
        status=SessionStatus.COMPLETED,
        events=[
            Event(
                id="conformance_completed_approval_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="conformance_completed_with_pending_state",
                payload={
                    "approval": {
                        "approval_id": "conformance_completed_approval_id",
                        "tool_name": "deploy",
                    }
                },
            )
        ],
        checkpoint=_approval_checkpoint(
            "conformance_completed_approval_id",
            "conformance_completed_approval_call",
            "deploy",
        ),
    )

    actions = []
    cursor: str | None = None
    while True:
        page = await store.query_pending_actions(PendingActionQuery(limit=2, cursor=cursor))
        actions.extend(page.actions)
        if not page.has_more:
            break
        assert page.next_cursor is not None
        cursor = page.next_cursor

    by_session = {action.session.id: action for action in actions}
    assert set(by_session) == {
        "conformance_approval",
        "conformance_input",
        "conformance_approval_recovery",
        "conformance_input_recovery",
        "conformance_round_recovery",
        "conformance_normalized_lookup",
    }
    assert by_session["conformance_approval"].kind == PendingActionKind.TOOL_APPROVAL
    assert by_session["conformance_input"].kind == PendingActionKind.USER_INPUT
    assert by_session["conformance_approval_recovery"].kind == PendingActionKind.MANUAL_RECOVERY
    assert by_session["conformance_input_recovery"].kind == PendingActionKind.MANUAL_RECOVERY
    assert by_session["conformance_round_recovery"].kind == PendingActionKind.MANUAL_RECOVERY
    assert by_session["conformance_normalized_lookup"].kind == PendingActionKind.TOOL_APPROVAL
    assert len({action.id for action in actions}) == len(actions)
    assert all(action.event.event.payload == {} for action in actions)

    malformed = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_malformed")
    )
    assert malformed.actions == []
    assert [issue.code for issue in malformed.issues] == ["source_invalid"]

    for session_id in (
        "conformance_blank_approval_id",
        "conformance_blank_input_id",
        "conformance_blank_round_id",
    ):
        blank_identifier = await store.query_pending_actions(
            PendingActionQuery(session_id=session_id)
        )
        assert blank_identifier.actions == []
        assert [issue.code for issue in blank_identifier.issues] == ["source_invalid"]

    malformed_terminal = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_malformed_terminal")
    )
    assert malformed_terminal.actions == []
    assert [issue.code for issue in malformed_terminal.issues] == ["source_invalid"]

    completed = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_completed_with_pending_state")
    )
    assert completed.actions == []
    assert [issue.code for issue in completed.issues] == ["source_invalid"]
    assert completed.issues[0].status == SessionStatus.COMPLETED

    repeated = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_input_recovery")
    )
    assert [action.id for action in repeated.actions] == [
        by_session["conformance_input_recovery"].id
    ]
    manual = await store.query_pending_actions(
        PendingActionQuery(kind=PendingActionKind.MANUAL_RECOVERY, q="recovery")
    )
    assert {action.session.id for action in manual.actions} == {
        "conformance_approval_recovery",
        "conformance_input_recovery",
        "conformance_round_recovery",
    }

    # Checkpoint state is replaceable. Clearing and then reintroducing the same
    # durable identifier must behave identically in memory and in SQL stores.
    await store.checkpoint("conformance_approval", {})
    assert not (
        await store.query_pending_actions(PendingActionQuery(session_id="conformance_approval"))
    ).actions
    await store.checkpoint(
        "conformance_approval",
        _approval_checkpoint(
            "conformance_approval_id",
            "conformance_approval_call",
            "deploy",
        ),
    )
    restored = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_approval")
    )
    assert len(restored.actions) == 1
    assert restored.actions[0].approval_id == "conformance_approval_id"
    await store.checkpoint(
        "conformance_approval",
        _approval_checkpoint(
            "conformance_alternate_approval_id",
            "conformance_alternate_approval_call",
            "rollback",
        ),
    )
    alternate = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_approval")
    )
    assert len(alternate.actions) == 1
    assert alternate.actions[0].approval_id == "conformance_alternate_approval_id"

    await create(
        "conformance_oversized_resolved",
        status=SessionStatus.FAILED,
        events=[
            Event(
                id="conformance_oversized_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_oversized_resolved",
                payload={
                    "tool_round_id": "conformance_oversized_round",
                    "tool_call_id": "conformance_oversized_call",
                },
            ),
            Event(
                id="conformance_oversized_completed",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_oversized_resolved",
                payload={
                    "tool_round_id": "conformance_oversized_round",
                    "tool_call_id": "conformance_oversized_call",
                    "result": {"content": "x" * 4096},
                },
            ),
            Event(
                id="conformance_oversized_failed",
                type=EventType.SESSION_FAILED,
                session_id="conformance_oversized_resolved",
                payload={"error": "worker stopped after recording the result"},
            ),
        ],
        checkpoint=_round_checkpoint(
            "conformance_oversized_round",
            "conformance_oversized_call",
        ),
    )
    oversized_resolved = await store.query_pending_actions(
        PendingActionQuery(
            session_id="conformance_oversized_resolved",
            max_result_bytes=1024,
        )
    )
    assert oversized_resolved.actions == []
    assert [issue.session_id for issue in oversized_resolved.issues] == [
        "conformance_oversized_resolved"
    ]

    overcomplex_session_id = "conformance_overcomplex_round"
    await create(
        overcomplex_session_id,
        status=SessionStatus.FAILED,
        events=[],
        checkpoint={
            "pending_tool_round": {
                "round_id": "conformance_overcomplex_round_id",
                "agent_name": "assistant",
                "tool_calls": [
                    _pending_call(f"conformance_overcomplex_call_{index}", "charge")
                    for index in range(MAX_PENDING_ACTION_TOOL_CALLS + 1)
                ],
            }
        },
    )
    overcomplex = await store.query_pending_actions(
        PendingActionQuery(session_id=overcomplex_session_id)
    )
    assert overcomplex.actions == []
    assert [issue.session_id for issue in overcomplex.issues] == [overcomplex_session_id]
    assert overcomplex.issues[0].code == "source_too_complex"

    async def create_approval(
        session_id: str,
        *,
        argument_bytes: int = 0,
    ) -> None:
        approval_id = f"{session_id}_approval"
        call_id = f"{session_id}_call"
        arguments = {"blob": "x" * argument_bytes} if argument_bytes else {}
        await create(
            session_id,
            status=SessionStatus.INTERRUPTED,
            events=[
                Event(
                    id=f"{session_id}_event",
                    type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                    session_id=session_id,
                    payload={
                        "approval": {
                            "approval_id": approval_id,
                            "tool_name": "deploy",
                            "arguments": arguments,
                        }
                    },
                )
            ],
            checkpoint=_approval_checkpoint(
                approval_id,
                call_id,
                "deploy",
                arguments=arguments,
            ),
        )

    # A visible issue consumes one page slot and advances the cursor, allowing
    # operators to reach a normal action immediately behind it.
    await create_approval("conformance_small_behind")
    await create_approval("conformance_oversized_head", argument_bytes=16 * 1024)
    first = await store.query_pending_actions(PendingActionQuery(limit=1, max_result_bytes=4096))
    assert first.actions == []
    assert [issue.session_id for issue in first.issues] == ["conformance_oversized_head"]
    assert first.has_more is True
    assert first.next_cursor is not None
    second = await store.query_pending_actions(
        PendingActionQuery(
            limit=1,
            cursor=first.next_cursor,
            max_result_bytes=4096,
        )
    )
    assert [action.session.id for action in second.actions] == ["conformance_small_behind"]
    assert second.issues == []
    assert second.total_count is None

    # The fifth row exists only to answer has_more for limit=1. Its payload must
    # not enter source measurement or materialization on this page.
    await create_approval("conformance_oversized_lookahead", argument_bytes=128 * 1024)
    for index in range(4):
        await create_approval(f"conformance_small_visible_{index}")
    lookahead = await store.query_pending_actions(
        PendingActionQuery(limit=1, max_result_bytes=32 * 1024)
    )
    assert len(lookahead.actions) == 1
    assert lookahead.issues == []
    assert lookahead.has_more is True
    assert lookahead.next_cursor is not None
