from __future__ import annotations

from typing import Any

from cayu.core import Event, EventType, Message
from cayu.runtime import RunRequest, SessionIdentity, SessionStatus, SessionStore
from cayu.runtime.sessions import (
    MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL,
    MAX_PENDING_ACTION_RESULT_BYTES,
    MAX_PENDING_ACTION_TOOL_CALLS,
    PendingActionKind,
    PendingActionQuery,
)

_MODEL_STEP_ID = f"mstep_{'1' * 32}"
_MODEL_ATTEMPT_ID = f"matt_{'2' * 32}"
_TOOL_ROUND_ID = f"tround_{'3' * 32}"
_MALFORMED_TERMINAL_ROUND_ID = f"tround_{'4' * 32}"
_OVERSIZED_ROUND_ID = f"tround_{'5' * 32}"
_OVERCOMPLEX_ROUND_ID = f"tround_{'6' * 32}"
_DUPLICATE_TERMINAL_ROUND_ID = f"tround_{'7' * 32}"
_RECONCILED_TERMINAL_ROUND_ID = f"tround_{'8' * 32}"


def _tool_round_identity_payload(tool_round_id: str = _TOOL_ROUND_ID) -> dict[str, str]:
    return {
        "model_step_id": _MODEL_STEP_ID,
        "model_attempt_id": _MODEL_ATTEMPT_ID,
        "tool_round_id": tool_round_id,
    }


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
            **_tool_round_identity_payload(),
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
            **_tool_round_identity_payload(),
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
            **_tool_round_identity_payload(round_id),
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
                    **_tool_round_identity_payload(),
                    "approval_id": "conformance_approval_id",
                    "tool_call_id": "conformance_approval_call",
                    "approval": {
                        "approval_id": "conformance_approval_id",
                        **_tool_round_identity_payload(),
                        "tool_call_id": "conformance_approval_call",
                        "tool_name": "deploy",
                        "arguments": {},
                    },
                },
            ),
            Event(
                id="conformance_alternate_approval_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="conformance_approval",
                payload={
                    **_tool_round_identity_payload(),
                    "approval_id": "conformance_alternate_approval_id",
                    "tool_call_id": "conformance_alternate_approval_call",
                    "approval": {
                        "approval_id": "conformance_alternate_approval_id",
                        **_tool_round_identity_payload(),
                        "tool_call_id": "conformance_alternate_approval_call",
                        "tool_name": "rollback",
                        "arguments": {},
                    },
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
                    **_tool_round_identity_payload(),
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
                    **_tool_round_identity_payload(),
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
                    **_tool_round_identity_payload(),
                    "manual_recovery_required": True,
                    "user_input": {
                        "input_id": "conformance_recovery_input_id",
                        **_tool_round_identity_payload(),
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
                    **_tool_round_identity_payload(),
                    "tool_call_id": "conformance_round_call",
                },
            ),
            Event(
                id="conformance_round_failed",
                type=EventType.SESSION_FAILED,
                session_id="conformance_round_recovery",
                payload={
                    **_tool_round_identity_payload(),
                    "manual_recovery_required": True,
                    "tool_call_id": "conformance_round_call",
                },
            ),
        ],
        checkpoint=_round_checkpoint(_TOOL_ROUND_ID, "conformance_round_call"),
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
                    **_tool_round_identity_payload(),
                    "approval_id": "conformance_stale_approval_id",
                    "tool_call_id": "conformance_stale_approval_call",
                    "approval": {
                        "approval_id": "conformance_stale_approval_id",
                        **_tool_round_identity_payload(),
                        "tool_call_id": "conformance_stale_approval_call",
                        "tool_name": "deploy",
                    },
                },
            ),
            Event(
                id="conformance_stale_barrier",
                type=EventType.SESSION_RESUMED,
                session_id="conformance_stale_approval",
                payload=_tool_round_identity_payload(),
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
                "tool_round_id": "   ",
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
                    **_tool_round_identity_payload(_MALFORMED_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_malformed_terminal_call",
                },
            ),
            Event(
                id="conformance_malformed_terminal_completed",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_malformed_terminal",
                payload={
                    **_tool_round_identity_payload(_MALFORMED_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_malformed_terminal_call",
                },
            ),
            Event(
                id="conformance_malformed_terminal_other_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_malformed_terminal",
                payload={
                    **_tool_round_identity_payload(_MALFORMED_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_malformed_terminal_other_call",
                },
            ),
        ],
        checkpoint={
            "pending_tool_round": {
                **_tool_round_identity_payload(_MALFORMED_TERMINAL_ROUND_ID),
                "agent_name": "assistant",
                "tool_calls": [
                    _pending_call("conformance_malformed_terminal_call", "charge"),
                    _pending_call("conformance_malformed_terminal_other_call", "charge"),
                ],
            }
        },
    )
    await create(
        "conformance_duplicate_terminal",
        status=SessionStatus.FAILED,
        events=[
            Event(
                id="conformance_duplicate_terminal_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_duplicate_terminal",
                payload={
                    **_tool_round_identity_payload(_DUPLICATE_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_duplicate_terminal_call",
                },
            ),
            Event(
                id="conformance_duplicate_terminal_completed_first",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_duplicate_terminal",
                payload={
                    **_tool_round_identity_payload(_DUPLICATE_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_duplicate_terminal_call",
                    "result": {"content": "first", "is_error": False},
                },
            ),
            Event(
                id="conformance_duplicate_terminal_completed_second",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_duplicate_terminal",
                payload={
                    **_tool_round_identity_payload(_DUPLICATE_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_duplicate_terminal_call",
                    "result": {"content": "second", "is_error": False},
                },
            ),
        ],
        checkpoint=_round_checkpoint(
            _DUPLICATE_TERMINAL_ROUND_ID,
            "conformance_duplicate_terminal_call",
        ),
    )
    await create(
        "conformance_reconciled_terminal",
        status=SessionStatus.FAILED,
        events=[
            Event(
                id="conformance_reconciled_terminal_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_reconciled_terminal",
                payload={
                    **_tool_round_identity_payload(_RECONCILED_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_reconciled_terminal_call",
                },
            ),
            Event(
                id="conformance_reconciled_terminal_malformed",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_reconciled_terminal",
                payload={
                    **_tool_round_identity_payload(_RECONCILED_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_reconciled_terminal_call",
                },
            ),
            Event(
                id="conformance_reconciled_terminal_manual",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_reconciled_terminal",
                payload={
                    **_tool_round_identity_payload(_RECONCILED_TERMINAL_ROUND_ID),
                    "tool_call_id": "conformance_reconciled_terminal_call",
                    "manual_recovery": True,
                    "result": {"content": "verified", "is_error": False},
                },
            ),
        ],
        checkpoint=_round_checkpoint(
            _RECONCILED_TERMINAL_ROUND_ID,
            "conformance_reconciled_terminal_call",
        ),
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
                    **_tool_round_identity_payload(),
                    "approval_id": "   ",
                    "tool_call_id": "conformance_normalized_lookup_call",
                    "approval": {
                        "approval_id": "conformance_normalized_lookup_id",
                        **_tool_round_identity_payload(),
                        "tool_call_id": "conformance_normalized_lookup_call",
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
        "conformance_conflicting_input_identity",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_conflicting_input_event",
                type=EventType.SESSION_AWAITING_USER_INPUT,
                session_id="conformance_conflicting_input_identity",
                payload={
                    **_tool_round_identity_payload(),
                    "model_attempt_id": f"matt_{'9' * 32}",
                    "input_id": "conformance_conflicting_input_id",
                    "tool_call_id": "conformance_conflicting_input_call",
                    "question": "Deploy?",
                },
            )
        ],
        checkpoint=_input_checkpoint(
            "conformance_conflicting_input_id",
            "conformance_conflicting_input_call",
            "Deploy?",
        ),
    )
    await create(
        "conformance_sibling_approval_call",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_sibling_approval_event",
                type=EventType.TOOL_CALL_APPROVAL_REQUESTED,
                session_id="conformance_sibling_approval_call",
                payload={
                    **_tool_round_identity_payload(),
                    "approval_id": "conformance_sibling_approval_id",
                    "tool_call_id": "conformance_sibling_call",
                    "approval": {
                        "approval_id": "conformance_sibling_approval_id",
                        **_tool_round_identity_payload(),
                        "tool_call_id": "conformance_sibling_call",
                        "tool_name": "rollback",
                        "arguments": {},
                    },
                },
            )
        ],
        checkpoint={
            "pending_tool_approval": {
                "approval_id": "conformance_sibling_approval_id",
                **_tool_round_identity_payload(),
                "tool_call_id": "conformance_gating_approval_call",
                "tool_name": "deploy",
                "arguments": {},
                "agent_name": "assistant",
                "tool_calls": [
                    _pending_call("conformance_gating_approval_call", "deploy"),
                    _pending_call("conformance_sibling_call", "rollback"),
                ],
            }
        },
    )
    await create(
        "conformance_sibling_input_call",
        status=SessionStatus.INTERRUPTED,
        events=[
            Event(
                id="conformance_sibling_input_event",
                type=EventType.SESSION_AWAITING_USER_INPUT,
                session_id="conformance_sibling_input_call",
                payload={
                    **_tool_round_identity_payload(),
                    "input_id": "conformance_sibling_input_id",
                    "tool_call_id": "conformance_input_sibling_call",
                    "question": "Wrong question?",
                },
            )
        ],
        checkpoint={
            "pending_user_input": {
                "input_id": "conformance_sibling_input_id",
                **_tool_round_identity_payload(),
                "tool_call_id": "conformance_gating_input_call",
                "tool_name": "ask_user",
                "question": "Deploy?",
                "options": ["yes", "no"],
                "arguments": {},
                "agent_name": "assistant",
                "tool_calls": [
                    _pending_call("conformance_gating_input_call", "ask_user"),
                    _pending_call("conformance_input_sibling_call", "read_file"),
                ],
            }
        },
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
                    **_tool_round_identity_payload(),
                    "approval_id": "conformance_completed_approval_id",
                    "tool_call_id": "conformance_completed_approval_call",
                    "approval": {
                        "approval_id": "conformance_completed_approval_id",
                        **_tool_round_identity_payload(),
                        "tool_call_id": "conformance_completed_approval_call",
                        "tool_name": "deploy",
                    },
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
        "conformance_malformed_terminal",
        "conformance_duplicate_terminal",
    }
    assert by_session["conformance_approval"].kind == PendingActionKind.TOOL_APPROVAL
    assert by_session["conformance_approval"].round_id == _TOOL_ROUND_ID
    assert by_session["conformance_approval"].tool_call_id == "conformance_approval_call"
    assert by_session["conformance_input"].kind == PendingActionKind.USER_INPUT
    assert by_session["conformance_input"].round_id == _TOOL_ROUND_ID
    assert by_session["conformance_input"].tool_call_id == "conformance_input_call"
    assert by_session["conformance_approval_recovery"].kind == PendingActionKind.MANUAL_RECOVERY
    assert by_session["conformance_input_recovery"].kind == PendingActionKind.MANUAL_RECOVERY
    assert by_session["conformance_round_recovery"].kind == PendingActionKind.MANUAL_RECOVERY
    assert (
        by_session["conformance_malformed_terminal"].tool_call_id
        == "conformance_malformed_terminal_call"
    )
    assert (
        by_session["conformance_duplicate_terminal"].tool_call_id
        == "conformance_duplicate_terminal_call"
    )
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
        "conformance_normalized_lookup",
        "conformance_conflicting_input_identity",
        "conformance_sibling_approval_call",
        "conformance_sibling_input_call",
    ):
        blank_identifier = await store.query_pending_actions(
            PendingActionQuery(session_id=session_id)
        )
        assert blank_identifier.actions == []
        assert [issue.code for issue in blank_identifier.issues] == ["source_invalid"]

    malformed_terminal = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_malformed_terminal")
    )
    assert len(malformed_terminal.actions) == 1
    assert malformed_terminal.actions[0].kind == PendingActionKind.MANUAL_RECOVERY
    assert malformed_terminal.issues == []

    duplicate_terminal = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_duplicate_terminal")
    )
    assert len(duplicate_terminal.actions) == 1
    assert duplicate_terminal.actions[0].kind == PendingActionKind.MANUAL_RECOVERY
    assert duplicate_terminal.issues == []

    reconciled_terminal = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_reconciled_terminal")
    )
    assert reconciled_terminal.actions == []
    assert reconciled_terminal.issues == []

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
        "conformance_malformed_terminal",
        "conformance_duplicate_terminal",
    }

    # Resumed tool events carry both their call id and the pause id. Once the
    # pause checkpoint is cleared, the surviving round must still discover its
    # ledger through the call identity.
    for identity_digit, pause_key in zip(("e", "f"), ("approval_id", "input_id"), strict=True):
        session_id = f"conformance_{pause_key}_tagged_ledger"
        tool_call_id = f"{session_id}_call"
        tool_round_id = f"tround_{identity_digit * 32}"
        await create(
            session_id,
            status=SessionStatus.FAILED,
            events=[
                Event(
                    id=f"{session_id}_started",
                    type=EventType.TOOL_CALL_STARTED,
                    session_id=session_id,
                    payload={
                        **_tool_round_identity_payload(tool_round_id),
                        "tool_call_id": tool_call_id,
                        pause_key: f"{session_id}_pause",
                    },
                )
            ],
            checkpoint=_round_checkpoint(tool_round_id, tool_call_id),
        )
        tagged_ledger = await store.query_pending_actions(PendingActionQuery(session_id=session_id))
        assert len(tagged_ledger.actions) == 1
        assert tagged_ledger.actions[0].kind == PendingActionKind.MANUAL_RECOVERY
        assert tagged_ledger.actions[0].tool_call_id == tool_call_id
        assert tagged_ledger.issues == []

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
                    "tool_round_id": _OVERSIZED_ROUND_ID,
                    "tool_call_id": "conformance_oversized_call",
                },
            ),
            Event(
                id="conformance_oversized_completed",
                type=EventType.TOOL_CALL_COMPLETED,
                session_id="conformance_oversized_resolved",
                payload={
                    "tool_round_id": _OVERSIZED_ROUND_ID,
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
            _OVERSIZED_ROUND_ID,
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
                **_tool_round_identity_payload(_OVERCOMPLEX_ROUND_ID),
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

    # A provider may reuse a tool-call id in a later execution. Historical
    # evidence for that id must be excluded by execution identity before byte
    # accounting and per-call complexity limits are applied.
    reused_call_id = "conformance_reused_call"
    reused_round_id = f"tround_{'9' * 32}"
    old_identity = {
        "model_step_id": f"mstep_{'a' * 32}",
        "model_attempt_id": f"matt_{'b' * 32}",
        "tool_round_id": f"tround_{'c' * 32}",
    }
    await create(
        "conformance_reused_call_identity",
        status=SessionStatus.FAILED,
        events=[
            *[
                Event(
                    id=f"conformance_old_terminal_{index}",
                    type=EventType.TOOL_CALL_COMPLETED,
                    session_id="conformance_reused_call_identity",
                    payload={
                        **old_identity,
                        "tool_call_id": reused_call_id,
                    },
                )
                for index in range(40)
            ],
            Event(
                id="conformance_current_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_reused_call_identity",
                payload={
                    **_tool_round_identity_payload(reused_round_id),
                    "tool_call_id": reused_call_id,
                },
            ),
        ],
        checkpoint=_round_checkpoint(reused_round_id, reused_call_id),
    )
    reused = await store.query_pending_actions(
        PendingActionQuery(
            session_id="conformance_reused_call_identity",
            max_result_bytes=8192,
        )
    )
    assert len(reused.actions) == 1
    assert reused.actions[0].kind == PendingActionKind.MANUAL_RECOVERY
    assert reused.issues == []

    # An oversized event still retains a bounded execution identity. Reusing
    # the provider call id in a later round must therefore exclude the stale
    # event before its size can suppress the current recovery action.
    oversized_reused_call_id = "conformance_oversized_reused_call"
    oversized_reused_round_id = f"tround_{'e' * 32}"
    await create(
        "conformance_oversized_reused_call_identity",
        status=SessionStatus.FAILED,
        events=[
            Event(
                id="conformance_oversized_old_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_oversized_reused_call_identity",
                payload={
                    **old_identity,
                    "tool_call_id": oversized_reused_call_id,
                    "manual_recovery": "x" * MAX_PENDING_ACTION_RESULT_BYTES,
                },
            ),
            Event(
                id="conformance_oversized_current_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_oversized_reused_call_identity",
                payload={
                    **_tool_round_identity_payload(oversized_reused_round_id),
                    "tool_call_id": oversized_reused_call_id,
                },
            ),
        ],
        checkpoint=_round_checkpoint(
            oversized_reused_round_id,
            oversized_reused_call_id,
        ),
    )
    oversized_reused = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_oversized_reused_call_identity")
    )
    assert len(oversized_reused.actions) == 1
    assert oversized_reused.actions[0].kind == PendingActionKind.MANUAL_RECOVERY
    assert oversized_reused.actions[0].tool_call_id == oversized_reused_call_id
    assert oversized_reused.issues == []

    # The same bounded envelope keeps oversized evidence for the active round
    # visible as an explicit issue instead of silently treating the call as
    # never started.
    oversized_current_call_id = "conformance_oversized_current_call"
    oversized_current_round_id = f"tround_{'f' * 32}"
    await create(
        "conformance_oversized_current_identity",
        status=SessionStatus.FAILED,
        events=[
            Event(
                id="conformance_oversized_current_identity_started",
                type=EventType.TOOL_CALL_STARTED,
                session_id="conformance_oversized_current_identity",
                payload={
                    **_tool_round_identity_payload(oversized_current_round_id),
                    "tool_call_id": oversized_current_call_id,
                    "manual_recovery": "x" * MAX_PENDING_ACTION_RESULT_BYTES,
                },
            )
        ],
        checkpoint=_round_checkpoint(
            oversized_current_round_id,
            oversized_current_call_id,
        ),
    )
    oversized_current = await store.query_pending_actions(
        PendingActionQuery(session_id="conformance_oversized_current_identity")
    )
    assert oversized_current.actions == []
    assert [issue.session_id for issue in oversized_current.issues] == [
        "conformance_oversized_current_identity"
    ]
    assert oversized_current.issues[0].code == "source_too_large"

    # Current-identity evidence remains bounded even when durable history is
    # corrupted or adversarial. The query fails closed without materializing an
    # unbounded ledger.
    ledger_overcomplex_session_id = "conformance_overcomplex_ledger"
    ledger_overcomplex_call_id = "conformance_overcomplex_ledger_call"
    ledger_overcomplex_round_id = f"tround_{'d' * 32}"
    await create(
        ledger_overcomplex_session_id,
        status=SessionStatus.FAILED,
        events=[
            Event(
                id=f"conformance_overcomplex_ledger_{index}",
                type=EventType.TOOL_CALL_STARTED,
                session_id=ledger_overcomplex_session_id,
                payload={
                    **_tool_round_identity_payload(ledger_overcomplex_round_id),
                    "tool_call_id": ledger_overcomplex_call_id,
                },
            )
            for index in range(MAX_PENDING_ACTION_LEDGER_EVENTS_PER_CALL + 1)
        ],
        checkpoint=_round_checkpoint(
            ledger_overcomplex_round_id,
            ledger_overcomplex_call_id,
        ),
    )
    ledger_overcomplex = await store.query_pending_actions(
        PendingActionQuery(session_id=ledger_overcomplex_session_id)
    )
    assert ledger_overcomplex.actions == []
    assert [issue.session_id for issue in ledger_overcomplex.issues] == [
        ledger_overcomplex_session_id
    ]
    assert ledger_overcomplex.issues[0].code == "source_too_complex"
    assert "event per-call" in ledger_overcomplex.issues[0].detail

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
                        **_tool_round_identity_payload(),
                        "approval_id": approval_id,
                        "tool_call_id": call_id,
                        "approval": {
                            "approval_id": approval_id,
                            **_tool_round_identity_payload(),
                            "tool_call_id": call_id,
                            "tool_name": "deploy",
                            "arguments": arguments,
                        },
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
