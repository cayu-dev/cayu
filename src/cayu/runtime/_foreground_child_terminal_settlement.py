"""Settle an exact terminal child's leftover invocation through normal recovery."""

from collections.abc import Awaitable, Callable
from typing import Protocol

from cayu.core.events import Event, EventType
from cayu.runtime._child_session_identity import (
    ChildSessionKind,
    ChildSessionRecoveryMatcher,
    generate_child_session_id,
)
from cayu.runtime._foreground_child_wait import (
    ForegroundChildWait,
    foreground_child_state_from_checkpoint,
)
from cayu.runtime._tool_effect_state import ToolEffectStateOwner
from cayu.runtime.sessions import (
    EventQuery,
    IncompleteSessionRecoveryRequest,
    IncompleteSessionRecoveryResult,
    SessionRunFenced,
    SessionStore,
)


class _RecoverTerminalSession(Protocol):
    def __call__(
        self,
        request: IncompleteSessionRecoveryRequest,
        *,
        before_mutation: Callable[[], Awaitable[None]],
    ) -> Awaitable[IncompleteSessionRecoveryResult]: ...


async def settle_foreground_child_terminal(
    wait: ForegroundChildWait,
    event: Event,
    *,
    store: SessionStore,
    matcher: ChildSessionRecoveryMatcher,
    recover: _RecoverTerminalSession,
    before_mutation: Callable[[], Awaitable[None]],
) -> None:
    """Never execute a child: require its original interaction already closed."""

    async def require_exact_child() -> None:
        await before_mutation()
        parent = await store.load(wait.parent_effect.session_id)
        child = await store.load(wait.child_session_id)
        if (
            parent is None
            or child is None
            or parent.instance_id != wait.parent_effect.session_instance_id
            or child.instance_id != wait.child_session_instance_id
            or child.id
            != generate_child_session_id(
                kind=ChildSessionKind.SUBAGENT,
                parent_session_id=parent.id,
                logical_spawn_id=wait.parent_effect.idempotency_key,
            )
        ):
            raise SessionRunFenced("Foreground terminal cleanup lost its exact child incarnation.")
        retained, selected = foreground_child_state_from_checkpoint(
            await store.load_checkpoint(parent.id)
        )
        subagent = child.metadata.get("subagent")
        if (
            retained != wait
            or selected is not None
            or type(subagent) is not dict
            or subagent.get("spawn_fingerprint") != wait.child_spawn_fingerprint
        ):
            raise SessionRunFenced("Foreground terminal cleanup lost its exact waiting spawn.")
        effect = await ToolEffectStateOwner(store).resolve_call(
            parent,
            tool_round_id=wait.parent_effect.tool_round_id,
            tool_call_id=wait.parent_effect.tool_call_id,
        )
        if (
            effect is None
            or effect.intent != wait.parent_effect
            or effect.state != "outcome_unknown"
            or effect.child_recovery_arguments is None
            or not matcher.matches_recoverable_child(
                child,
                parent_invocation=parent.invocation,
                parent_session_id=parent.id,
                causal_budget_id=parent.causal_budget_id,
                environment_name=parent.environment_name,
                tool_call_id=effect.intent.tool_call_id,
                idempotency_key=effect.intent.idempotency_key,
                arguments=effect.child_recovery_arguments,
                require_fingerprint=True,
            )
        ):
            raise SessionRunFenced("Foreground terminal cleanup lacks its original spawn evidence.")
        outcome = await store.summarize_outcome(child.id)
        if outcome.terminal_event is None or outcome.terminal_event.event != event:
            raise SessionRunFenced("Foreground terminal cleanup lost its exact outcome.")
        closed = await store.query_events(
            EventQuery(
                session_id=child.id,
                interaction_id=wait.child_interaction_id,
                event_types=(
                    EventType.INTERACTION_COMPLETED,
                    EventType.INTERACTION_FAILED,
                    EventType.INTERACTION_INTERRUPTED,
                ),
                limit=1,
            )
        )
        if not closed:
            raise SessionRunFenced(
                "Foreground terminal cleanup requires a closed child interaction."
            )

    await require_exact_child()
    await recover(
        IncompleteSessionRecoveryRequest(
            session_id=wait.child_session_id,
            inactive_for_seconds=0,
            reason="Settle the exact terminal foreground child before parent continuation.",
        ),
        before_mutation=require_exact_child,
    )
