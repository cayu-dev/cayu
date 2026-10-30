"""A task's claim is authority, not part of the request an external wait replays.

A task handler re-run after its worker died, and a continuation retried under a new
claim, present the same request under another worker id and lease. The digests an
external wait binds must not change with them, or the replay is refused.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cayu import CayuApp, Message, RunRequest
from cayu.sessions.requests import ResumeRequest

_LEASE = datetime(2026, 10, 30, 12, tzinfo=UTC)


def _run(**claim) -> RunRequest:
    return RunRequest(
        agent_name="root",
        session_id="s",
        messages=[Message.text("user", "render it")],
        **claim,
    )


def _resume(**claim) -> ResumeRequest:
    return ResumeRequest(session_id="s", messages=[Message.text("user", "{}")], **claim)


def test_a_new_claim_keeps_a_task_run_digest() -> None:
    digest = CayuApp(
        enable_logging=False
    )._work_attempt_execution.work_attempt_source_request_sha256

    def claim_free(request):
        return digest(request, kind="initial", without_task_claim=True)

    first = _run(task_id="t", task_worker_id="worker-1", task_lease_expires_at=_LEASE)
    retried = _run(
        task_id="t", task_worker_id="worker-2", task_lease_expires_at=_LEASE + timedelta(minutes=5)
    )
    assert digest(first, kind="initial") != digest(retried, kind="initial")
    assert claim_free(first) == claim_free(retried)
    # The task stays bound: another task's run is another request.
    assert claim_free(first) != claim_free(
        _run(task_id="u", task_worker_id="worker-1", task_lease_expires_at=_LEASE)
    )


def test_a_new_claim_keeps_a_task_continuation_digest() -> None:
    digest = CayuApp(
        enable_logging=False
    )._work_attempt_execution.work_attempt_source_request_sha256

    def claim_free(request):
        return digest(request, kind="continuation", without_task_claim=True)

    first = _resume(task_worker_id="w:continue:a", task_handoff_id="h1")
    retried = _resume(task_worker_id="w2:continue:a", task_handoff_id="h2")
    assert claim_free(first) == claim_free(retried) == claim_free(_resume())


def test_an_unclaimed_request_keeps_its_digest() -> None:
    """Waits retained before this change still replay: nothing changes without a claim."""

    digest = CayuApp(
        enable_logging=False
    )._work_attempt_execution.work_attempt_source_request_sha256
    for request, kind in ((_run(), "initial"), (_resume(), "continuation")):
        assert digest(request, kind=kind, without_task_claim=True) == digest(request, kind=kind)


# Computed at 4f11a004a, before claim-free digests: waits retained then must still replay.
_BASE_RUN_DIGEST = "bfb8658a380f7e8fc826e89c2eef52b57913f42648386b511956af076e2d82cf"
_BASE_RESUME_DIGEST = "16e3d4d477122378b1ac9a4581c7ea9a9b14bd93d8d6bce8b3736e913841dd6e"


def test_an_unclaimed_request_with_explicit_none_claims_keeps_its_retained_digest() -> None:
    digest = CayuApp(
        enable_logging=False
    )._work_attempt_execution.work_attempt_source_request_sha256
    run = _run(task_worker_id=None, task_lease_expires_at=None)
    resume = _resume(task_worker_id=None, task_handoff_id=None)
    assert digest(run, kind="initial", without_task_claim=True) == _BASE_RUN_DIGEST
    assert digest(resume, kind="continuation", without_task_claim=True) == _BASE_RESUME_DIGEST
    # A claimed request still binds as if it carried none.
    claimed = _resume(task_worker_id="w:continue:a", task_handoff_id="h1")
    assert digest(claimed, kind="continuation", without_task_claim=True) == digest(
        _resume(), kind="continuation"
    )
