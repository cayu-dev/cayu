"""Same-producer epoch evidence; never a grant to start or disclose work."""

from cayu.sessions._execution_profile_checkpoint import (
    ActiveInvocationExecutionProfile,
)


def require_producer_epoch(index, checkpoint, profile, run_epoch):
    from cayu.sessions._invocation_lifecycle import (
        require_invocation_rebind_lineage,
    )

    original = index.invocation
    if original is None or "sha256:" + profile.fingerprint != original.profile_commitment:
        raise ValueError("Producer epoch lacks its original profile.")
    origin = ActiveInvocationExecutionProfile(
        session_id=index.session_id,
        interaction_id=original.interaction_id,
        run_epoch=original.run_epoch,
        profile=profile,
    )
    target = ActiveInvocationExecutionProfile(
        session_id=origin.session_id,
        interaction_id=origin.interaction_id,
        run_epoch=run_epoch,
        profile=profile,
    )
    require_invocation_rebind_lineage(
        checkpoint,
        session_instance_id=index.session_instance_id,
        original=origin,
        current=target,
    )
    return target
