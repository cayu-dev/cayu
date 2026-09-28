"""Finite application policies composed from the same explicit host operations.

Roles, input selection and ordering belong here, not in CollaborationHost.
These credential-free examples use the real provider adapter with local transport.
"""

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from hashlib import sha256

from examples.collaboration.explicit_host import (
    JOURNEY_TIMEOUT_S,
    create_demo_team,
    run_question,
)


async def sequential_specialists(*, report=None):
    """One coordinator explicitly forwards a design result for verification.

    The second check starts only after the first answer was delivered, exposed
    and settled. A separate disclosure grant selects its visible answer, not
    private specialist history. Order alone never grants forwarding authority.
    """
    team = await create_demo_team(roles=("coordinator", "design", "verification"))
    coordinator, designer, verifier = team.participants
    async with asyncio.timeout(JOURNEY_TIMEOUT_S):
        design = await run_question(
            team,
            key="design-check",
            sender=coordinator,
            recipient=designer,
            task="PUBLIC_DRAFT: bounded local service. Check the ownership design.",
            report=report,
        )
        assert design["producer"] == "settled"
        handoff = await team.disclosure.allow_question_input(
            source_receipt=design["source_receipt"],
            text=design["answer"],
            question_key="verification-check",
            recipient=verifier,
            forward_from=coordinator,
        )
        verification = await run_question(
            team,
            key="verification-check",
            sender=coordinator,
            recipient=verifier,
            task="Verify the preceding design answer for the bounded local service.",
            report=report,
            input_handoff=handoff,
        )
    assert len(team.transport.requests) == 6
    actual = json.dumps(verification["serialized_requests"][1]["input"])
    assert design["answer"] in actual
    assert design["source_receipt"] in actual and handoff.content_commitment in actual
    return design, verification


async def supervisor_workers(*, report=None):
    """A supervisor requests two bounded subtasks and collects both replies.

    Each obligation has its own parked supervisor invocation. There is no
    permanent manager model, task-group loser policy or deployment-wide quota.
    """
    team = await create_demo_team(roles=("supervisor", "worker-a", "worker-b"))
    supervisor, worker_a, worker_b = team.participants
    async with asyncio.timeout(JOURNEY_TIMEOUT_S), asyncio.TaskGroup() as group:
        first = group.create_task(
            run_question(
                team,
                key="worker-a",
                sender=supervisor,
                recipient=worker_a,
                task="SUBTASK_A: inspect input validation.",
                report=report,
            )
        )
        second = group.create_task(
            run_question(
                team,
                key="worker-b",
                sender=supervisor,
                recipient=worker_b,
                task="SUBTASK_B: inspect cleanup ownership.",
                report=report,
            )
        )
    results = first.result(), second.result()
    assert all(result["producer"] == "settled" for result in results)
    assert len(team.transport.requests) == 6
    return results


async def peer_discussion(*, report=None):
    """Two explicit discussion turns, with an authorized reply-dependent response.

    The peers exchange roles rather than copying each other's live transcript.
    Each reply reaches the asking peer through its exact export and wait. The
    application grants reuse of the first visible reply for the second question;
    neither a receipt nor reciprocal roles implicitly authorizes that reuse.
    """
    team = await create_demo_team(roles=("peer-a", "peer-b"))
    peer_a, peer_b = team.participants
    async with asyncio.timeout(JOURNEY_TIMEOUT_S):
        objection = await run_question(
            team,
            key="discussion-objection",
            sender=peer_a,
            recipient=peer_b,
            task="SHARED_PROPOSAL: retain cleanup ownership. What is the strongest objection?",
            report=report,
        )
        handoff = await team.disclosure.allow_question_input(
            source_receipt=objection["source_receipt"],
            text=objection["answer"],
            question_key="discussion-defense",
            recipient=peer_a,
        )
        defense = await run_question(
            team,
            key="discussion-defense",
            sender=peer_b,
            recipient=peer_a,
            task="Respond to the preceding objection with evidence about retained cleanup ownership.",
            report=report,
            input_handoff=handoff,
        )
    assert objection["request"].intent.request.sender == defense["request"].intent.request.target
    assert len(team.transport.requests) == 6
    actual = json.dumps(defense["serialized_requests"][1]["input"])
    assert objection["answer"] in actual
    assert objection["source_receipt"] in actual and handoff.content_commitment in actual
    return objection, defense


async def independent_parallel_candidates(*, report=None):
    """Concurrent fresh candidates have disjoint private inputs and reply grants.

    The judge receives both answers, but neither candidate is a recipient of
    the other's export. Any later cross-candidate disclosure needs a new grant;
    selecting a winner here does not manufacture one.
    """
    team = await create_demo_team(roles=("judge", "candidate-a", "candidate-b"))
    judge, candidate_a, candidate_b = team.participants
    async with asyncio.timeout(JOURNEY_TIMEOUT_S), asyncio.TaskGroup() as group:
        first = group.create_task(
            run_question(
                team,
                key="candidate-a",
                sender=judge,
                recipient=candidate_a,
                task="PRIVATE_CANDIDATE_A: independently propose a bounded design.",
                report=report,
            )
        )
        second = group.create_task(
            run_question(
                team,
                key="candidate-b",
                sender=judge,
                recipient=candidate_b,
                task="PRIVATE_CANDIDATE_B: independently propose a bounded design.",
                report=report,
            )
        )
    results = first.result(), second.result()
    for index, result in enumerate(results):
        own, other = ("A", "B") if index == 0 else ("B", "A")
        actual = json.dumps(result["serialized_requests"][1])
        assert "PRIVATE_CANDIDATE_" + own in actual
        assert "PRIVATE_CANDIDATE_" + other not in actual
        assert "Specialist analysis completed" not in actual
    assert len(team.transport.requests) == 6
    # A deterministic application choice among completed replies, not execution
    # authority or a cancellation/exclusion decision for the other candidate.
    return {"selected": results[0]["request"].operation, "results": results}


@dataclass(frozen=True, slots=True)
class DraftRevision:
    """Bounded application work product; not a Cayu acquisition or launch grant."""

    revision: int
    text: str

    def __post_init__(self):
        if (
            type(self.revision) is not int
            or not 1 <= self.revision <= 2
            or type(self.text) is not str
            or not self.text
            or len(self.text.encode()) > 512
        ):
            raise ValueError("The finite review example requires one of two bounded revisions.")

    @property
    def commitment(self):
        value = json.dumps([self.revision, self.text], separators=(",", ":"), ensure_ascii=False)
        return sha256(value.encode()).hexdigest()

    @property
    def review_task(self):
        return "DEMO_REVIEW_REVISION:" + self.commitment + "\nDRAFT: " + self.text


def review_accepts(result, revision):
    """Interpret an already authenticated review reply for this exact draft.

    This is application business acceptance, not a callable authority receiver.
    Callers obtain result through run_question's native export/readback path.
    An old positive answer is insufficient when the draft or revision changes.
    """
    return (
        type(revision) is DraftRevision
        and result["request"].intent.request.content == revision.review_task
        and result["answer"] == "APPROVE_REVISION:" + revision.commitment
        and result["producer"] == "settled"
        and result["delivery"] == "appended"
        and result["continuation"] == "consumed"
    )


async def bounded_review_revision(*, report=None):
    """Two review rounds: changing the draft requires a new exact review.

    These are application work-product reviews, not native human-tool approval
    gates. Ordinary execution and human gates remain under their native owners.
    """
    team = await create_demo_team(roles=("author", "reviewer"))
    author, reviewer = team.participants
    first = DraftRevision(1, "Retain dispatched cleanup until acknowledged.")
    second = DraftRevision(2, "Retain dispatched cleanup and fence retries until acknowledged.")
    async with asyncio.timeout(JOURNEY_TIMEOUT_S):
        first_review = await run_question(
            team,
            key="review-v1",
            sender=author,
            recipient=reviewer,
            task=first.review_task,
            report=report,
        )
        assert review_accepts(first_review, first)
        assert not review_accepts(first_review, second)
        assert len(team.transport.requests) == 3
        second_review = await run_question(
            team,
            key="review-v2",
            sender=author,
            recipient=reviewer,
            task=second.review_task,
            report=report,
        )
    assert review_accepts(second_review, second)
    assert not review_accepts(second_review, first)
    for draft, result in ((first, first_review), (second, second_review)):
        actual = json.dumps(result["serialized_requests"][1]["input"])
        assert draft.commitment in actual and draft.text in actual
    assert len(team.transport.requests) == 6
    return first_review, second_review


async def shared_specialist(*, report=None):
    """Two independent requesters reuse a specialist and one common budget.

    Each question has its own operation, fresh producer session, exact waiting
    recipient and disclosure grant. Reusing the participant neither merges the
    investigations nor authorizes sharing the first request's input/result.
    """
    team = await create_demo_team(roles=("requester-a", "requester-b", "specialist"))
    requester_a, requester_b, specialist = team.participants
    async with asyncio.timeout(JOURNEY_TIMEOUT_S):
        first = await run_question(
            team,
            key="independent-a",
            sender=requester_a,
            recipient=specialist,
            task="CASE_A_INPUT: evaluate the first independent request.",
            report=report,
        )
        second = await run_question(
            team,
            key="independent-b",
            sender=requester_b,
            recipient=specialist,
            task="CASE_B_INPUT: evaluate the second independent request.",
            report=report,
        )
    first_input = json.dumps(first["serialized_requests"][1]["input"])
    second_input = json.dumps(second["serialized_requests"][1]["input"])
    assert "CASE_A_INPUT" in first_input and "CASE_B_INPUT" not in first_input
    assert "CASE_B_INPUT" in second_input and "CASE_A_INPUT" not in second_input
    assert first["request"].operation != second["request"].operation
    assert first["request"].intent.selection.recipient.reference == specialist
    assert second["request"].intent.selection.recipient.reference == specialist
    assert len(team.transport.requests) == 6
    return first, second


async def main():
    patterns = {
        "sequential-specialists": sequential_specialists,
        "supervisor-workers": supervisor_workers,
        "peer-discussion": peer_discussion,
        "independent-parallel-candidates": independent_parallel_candidates,
        "bounded-review-revision": bounded_review_revision,
        "shared-specialist": shared_specialist,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", choices=tuple(patterns))
    selected = parser.parse_args().pattern
    result = await patterns[selected](
        report=lambda phase: print(phase, file=sys.stderr, flush=True)
    )
    results = result["results"] if selected == "independent-parallel-candidates" else result
    print(
        json.dumps(
            {
                "pattern": selected,
                "questions": len(results),
                "provider_calls": sum(item["provider_calls"] for item in results),
                "settled": all(item["producer"] == "settled" for item in results),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
