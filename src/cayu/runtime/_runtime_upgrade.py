"""Whether an interrupted invocation started under a different Cayu runtime build."""

from __future__ import annotations

from cayu.build_provenance import current_runtime_build_provenance
from cayu.execution_profiles import (
    ExecutionProfileComponentClass,
    execution_profile_runtime_component,
)
from cayu.sessions._execution_profile_checkpoint import (
    active_invocation_execution_profile_from_checkpoint,
)
from cayu.sessions.records import Session
from cayu.sessions.store import SessionStore


async def invocation_runtime_changed(store: SessionStore, session: Session) -> bool:
    """Compare the active invocation's runtime component with the running build."""

    from cayu.runtime._execution_profile_continuation import _runtime_version

    active = active_invocation_execution_profile_from_checkpoint(
        await store.load_checkpoint(session.id)
    )
    if active is None:
        return False
    return active.profile.component(
        ExecutionProfileComponentClass.RUNTIME
    ) != execution_profile_runtime_component(
        session.runtime_name, _runtime_version(), current_runtime_build_provenance()
    )


def is_takeover_of(session: Session) -> bool:
    """Whether this recovery belongs to a continuation's takeover of ``session``."""

    from cayu.runtime._abandoned_session_recovery import _ABANDONED_EXECUTION

    takeover = _ABANDONED_EXECUTION.get()
    return takeover is not None and (takeover.session_id, takeover.session_instance_id) == (
        session.id,
        session.instance_id,
    )
