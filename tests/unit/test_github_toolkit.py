"""The GitHub toolkit on a fake GitHub: each tool's calls and trimmed answer, writes only when
asked and behind approval, head guards, the repository allow-list and errors the model can act
on."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from typing import Any

import httpx
import pytest
from tests.connector_broker_support import assert_within_budgets, context, policy_request

from cayu import AgentSpec, CayuApp
from cayu.approvals import ToolArgumentReview
from cayu.connections import LocalConnections, LocalSlot
from cayu.connectors import github
from cayu.connectors._common import MAX_TERMINAL_PAYLOAD_BYTES
from cayu.connectors.github import client as client_module
from cayu.connectors.github.testing import FakeGitHub
from cayu.runtime import check_manifest
from cayu.tools import (
    AllowAllToolPolicy,
    AlwaysRequireApprovalToolPolicy,
    StaticToolPolicy,
    Tool,
    ToolEffect,
    ToolPolicyDecision,
    ToolResult,
)
from cayu.tools.gateway import tool_argument_validation_error

REPO = "acme/widgets"
BOT = "cayu-shepherd[bot]"
REVIEWER = "cayu-reviewer"
WRITES = (
    "github_comment",
    "github_rerun_failed_jobs",
    "github_update_branch",
    "github_merge_pull",
)
READS = ("github_read_pull", "github_list_pulls", "github_read_issue", "github_read_job_log")


@pytest.fixture(autouse=True)
def waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the client's retry waits."""

    async def skip_wait(seconds: float) -> None:
        pass

    monkeypatch.setattr(client_module, "_sleep", skip_wait)


def _fake() -> FakeGitHub:
    fake = FakeGitHub(login=BOT)
    fake.add_repository(REPO)

    def compare(variables: Any, repository: str) -> dict[str, Any]:
        pull = next(
            pull
            for pull in fake.repository(repository).pulls
            if variables["head"] == f"refs/pull/{pull.number}/head"
        )
        counts = {"aheadBy": pull.ahead_by, "behindBy": pull.behind_by}
        head = {"target": {"oid": pull.head_sha}}
        return {"data": {"repository": {"ref": {"compare": counts}, "head": head}}}

    fake.on_graphql("PullRequestCompare", compare)
    return fake


def _tools(fake: FakeGitHub, **settings: Any) -> dict[str, Tool]:
    made = github.tools(fake, include_writes=True, **settings)
    return {tool.name: tool for tool in made}


def _run(tool: Tool, **args: Any) -> ToolResult:
    return asyncio.run(tool.run(context(), args))


def _answer(result: ToolResult) -> Mapping[str, Any]:
    assert not result.is_error, result.content
    assert result.structured is not None
    return result.structured


def _error(result: ToolResult) -> Mapping[str, Any]:
    assert result.is_error
    assert result.structured is not None
    return result.structured


def test_the_toolkit_has_read_tools_and_write_tools_only_when_asked() -> None:
    fake = _fake()
    reads = github.tools(fake)
    assert [tool.name for tool in reads] == list(READS)
    tools = github.tools(fake, include_writes=True)
    assert {tool.name: tool.spec.effect for tool in tools} == {
        **dict.fromkeys(WRITES, ToolEffect.EXTERNAL),
        **dict.fromkeys(READS, ToolEffect.NONE),
    }
    assert all(tool.spec.max_terminal_payload_bytes == MAX_TERMINAL_PAYLOAD_BYTES for tool in tools)
    assert [tool.name for tool in tools] == [*READS, *WRITES]
    assert [tool.name for tool in tools if not tool.spec.parallel_safe] == list(WRITES)
    assert all("may need a person's approval" in tool.description for tool in tools[4:])
    # The read tools' client refuses writes before they leave, whatever a read tool sends.
    assert all(vars(tool)["_gh"].read_only for tool in tools[:4])
    assert not any(vars(tool)["_gh"].read_only for tool in tools[4:])


def test_read_pull_answers_the_facts_a_model_acts_on() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, behind_by=2, labels=["ready"], author="cayu-scout[bot]")
    ci = fake.add_workflow_run(REPO, pull.head_sha, conclusion="failure")
    tests = fake.add_check_run(REPO, pull.head_sha, "tests", run=ci, conclusion="failure")
    fake.add_check_run(REPO, pull.head_sha, "lint", run=ci, conclusion="success")
    review = fake.add_workflow_run(REPO, pull.head_sha, name="Review")
    fake.add_check_run(
        REPO, pull.head_sha, "Review", run=review, app_slug=REVIEWER, output_title="5/5 · ready"
    )
    fake.add_review(REPO, 7, "alice", state="APPROVED")
    fake.add_comment(REPO, 7, "Looks risky.", author="bob")

    result = _run(
        _tools(fake, reviewer_app=REVIEWER)["github_read_pull"], repository=REPO, number=7
    )

    answer = _answer(result)
    assert json.loads(result.content) == answer
    assert (answer["number"], answer["state"], answer["labels"]) == (7, "open", ["ready"])
    assert (answer["head_sha"], answer["behind_by"], answer["mergeable"]) == (
        pull.head_sha,
        2,
        True,
    )
    checks = answer["checks"]
    assert (checks["state"], checks["complete"], checks["passed"], checks["running"]) == (
        "failed",
        True,
        1,
        [],
    )
    assert checks["failed"] == [
        {
            "name": "tests",
            "conclusion": "failure",
            "job_id": tests.id,
            "run_id": ci.id,
            "attempt": 1,
        }
    ]
    assert [(seen["name"], seen["title"]) for seen in checks["reviewer"]] == [
        ("Review", "5/5 · ready")
    ]
    assert [(r["author"], r["state"]) for r in answer["reviews"]] == [("alice", "APPROVED")]
    assert [(c["author"], c["body"]) for c in answer["comments"]] == [("bob", "Looks risky.")]
    assert (answer["comments_total"], answer["gaps"], answer["reviews_complete"]) == (1, [], True)
    assert not fake.writes()
    assert not fake.unanswered


def test_read_pull_keeps_the_newest_comments_and_shortens_them_to_fit() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)
    for index in range(45):
        fake.add_comment(REPO, 7, f"{index} " + "x" * 60_000, author="bob")

    result = _run(_tools(fake)["github_read_pull"], repository=REPO, number=7)

    answer = _answer(result)
    assert answer["comments_total"] == 45
    comments = answer["comments"]
    assert [comment["body"].split()[0] for comment in comments] == [str(i) for i in range(25, 45)]
    assert all(comment["body_shortened"] for comment in comments)
    assert len(comments[0]["body"]) < 60_000
    assert_within_budgets(result)
    assert not fake.unanswered


def test_read_pull_answers_the_newest_of_thousands_of_comments() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)
    for index in range(5_120):
        fake.add_comment(REPO, 7, f"note {index}", author="bob")

    answer = _answer(_run(_tools(fake)["github_read_pull"], repository=REPO, number=7))

    assert answer["comments_total"] == 5_120
    assert [c["body"] for c in answer["comments"]] == [f"note {i}" for i in range(5_100, 5_120)]
    assert answer["gaps"] == []
    pages = [call.query.get("page") for call in fake.requests if call.path.endswith("/comments")]
    assert pages == ["205"]
    assert not fake.unanswered


def test_comments_too_large_to_read_leave_a_gap_and_the_rest_of_the_facts() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    fake.add_issue(REPO, 3)
    for number in (7, 3):
        fake.add_comment(REPO, number, "long", author="bob")
    fake.refuse("upstream_response_too_large", "GET", "*/comments", times=2)
    tools = _tools(fake)

    facts = _answer(_run(tools["github_read_pull"], repository=REPO, number=7))
    issue = _answer(_run(tools["github_read_issue"], repository=REPO, number=3))

    for answer in (facts, issue):
        assert answer["comments"] == []
        assert answer["comments_total"] == 1
        assert [gap["fact"] for gap in answer["gaps"]] == ["comments"]
    assert facts["head_sha"] == pull.head_sha
    assert not fake.unanswered


def test_list_pulls_answers_each_pull_in_brief() -> None:
    fake = _fake()
    node = {
        "number": 7,
        "title": "Fix the cache",
        "url": f"https://github.com/{REPO}/pull/7",
        "state": "OPEN",
        "isDraft": False,
        "createdAt": "2026-11-12T00:00:00Z",
        "updatedAt": "2026-11-12T00:00:00Z",
        "mergedAt": None,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "baseRefName": "main",
        "headRefName": "fix",
        "headRefOid": "a" * 40,
        "author": {"__typename": "Bot", "login": "cayu-scout"},
        "mergedBy": None,
        "labels": {"nodes": [{"name": "ready"}]},
        "assignees": {"nodes": []},
        "reviewRequests": {"nodes": []},
        "reviews": {"pageInfo": {"hasPreviousPage": False}, "nodes": []},
        "commits": {"nodes": []},
    }
    asked: list[Any] = []

    def listing(variables: Any, repository: str) -> dict[str, Any]:
        asked.append(variables["states"])
        nodes = [{**node, "number": number} for number in (7, 6, 5)]
        # Pull 6 has more reviews than the listing reads: its standing ones may be missing.
        nodes[1]["reviews"] = {"pageInfo": {"hasPreviousPage": True}, "nodes": []}
        page = {"pageInfo": {"hasNextPage": True, "endCursor": "c1"}, "nodes": nodes}
        return {"data": {"repository": {"pullRequests": page}}}

    fake.on_graphql("PullRequestFacts", listing)
    tool = _tools(fake)["github_list_pulls"]

    result = _run(tool, repository=REPO, limit=2)
    merged = _run(tool, repository=REPO, state="merged")

    answer = _answer(result)
    assert answer["complete"] is False
    assert [pull["number"] for pull in answer["pulls"]] == [7, 6]
    assert answer["pulls"][0] == {
        "number": 7,
        "title": "Fix the cache",
        "author": "cayu-scout[bot]",
        "state": "open",
        "draft": False,
        "labels": ["ready"],
        "base": "main",
        "head_ref": "fix",
        "head_sha": "a" * 40,
        "updated_at": "2026-11-12T00:00:00+00:00",
        "mergeable": True,
        "checks": "none",
        "failed_checks": [],
        "reviews": [],
        "reviews_complete": True,
        "gaps": [],
    }
    assert answer["pulls"][1]["reviews_complete"] is False
    assert answer["pulls"][1]["gaps"] == [
        {"fact": "reviews", "reason": "Only the latest 40 reviews were read."}
    ]
    assert not merged.is_error
    assert asked == [["OPEN"], ["MERGED"]]
    assert _error(_run(tool, repository=REPO, state="draft"))["error"] == "invalid_arguments"
    assert not fake.unanswered


def test_read_issue_answers_its_body_and_newest_comments() -> None:
    fake = _fake()
    fake.add_issue(REPO, 3, title="Cache misses", body="b" * 9_000, labels=["bug"], author="ana")
    for index in range(5):
        fake.add_comment(REPO, 3, f"note {index}", author="ben")
    fake.add_pull(REPO, 4)
    tool = _tools(fake)["github_read_issue"]

    result = _run(tool, repository=REPO, number=3, comments=2)
    pull = _run(tool, repository=REPO, number=4, comments=0)

    answer = _answer(result)
    assert (answer["title"], answer["state"], answer["author"], answer["labels"]) == (
        "Cache misses",
        "open",
        "ana",
        ["bug"],
    )
    assert (len(answer["body"]), answer["body_shortened"]) == (8_000, True)
    assert answer["comments_total"] == 5
    assert [comment["body"] for comment in answer["comments"]] == ["note 3", "note 4"]
    assert answer["is_pull_request"] is False
    assert _answer(pull)["is_pull_request"] is True
    assert not fake.unanswered


def test_a_failed_check_leads_to_its_log_tail_and_a_rerun() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    ci = fake.add_workflow_run(REPO, pull.head_sha, conclusion="failure")
    fake.add_check_run(REPO, pull.head_sha, "tests", run=ci, conclusion="failure")
    tools = _tools(fake)
    [failed] = _answer(_run(tools["github_read_pull"], repository=REPO, number=7))["checks"][
        "failed"
    ]
    log = "\n".join(f"\x1b[31mline {index}\x1b[0m" for index in range(500))
    fake.set_log(REPO, failed["job_id"], log)

    tail = _run(tools["github_read_job_log"], repository=REPO, job_id=failed["job_id"], lines=3)
    rerun = _run(
        tools["github_rerun_failed_jobs"],
        repository=REPO,
        run_id=failed["run_id"],
        head_sha=pull.head_sha,
    )

    assert tail.structured == {
        "repository": REPO,
        "job_id": failed["job_id"],
        "log": "line 497\nline 498\nline 499",
        "truncated": True,
    }
    assert rerun.structured == {
        "repository": REPO,
        "run_id": ci.id,
        "head_sha": pull.head_sha,
        "rerun": True,
    }
    assert ci.run_attempt == 2
    [write] = fake.writes()
    assert (write.method, write.path) == (
        "POST",
        f"/repos/{REPO}/actions/runs/{ci.id}/rerun-failed-jobs",
    )
    assert not fake.unanswered


def test_comment_posts_the_body_and_answers_the_new_comment() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)

    result = _run(_tools(fake)["github_comment"], repository=REPO, number=7, body="**Ready.**")

    [comment] = fake.repository(REPO).issues[7].comments
    assert (comment.body, comment.author) == ("**Ready.**", BOT)
    answer = _answer(result)
    assert (answer["id"], answer["number"]) == (comment.id, 7)
    assert answer["url"].endswith(f"#issuecomment-{comment.id}")
    blank = _run(_tools(fake)["github_comment"], repository=REPO, number=7, body="  ")
    assert _error(blank)["error"] == "invalid_arguments"
    assert len(fake.writes()) == 1
    assert not fake.unanswered


def test_a_comment_mentions_no_one_and_hides_nothing_unless_the_application_allows() -> None:
    fake = _fake()
    fake.add_pull(REPO, 7)
    text = "@dependabot merge <!-- @cayu-reviewer review -->"

    _answer(_run(_tools(fake)["github_comment"], repository=REPO, number=7, body=text))
    raw = _tools(fake, plain_comments=False)["github_comment"]
    _answer(_run(raw, repository=REPO, number=7, body=text))

    plain, kept = (comment.body for comment in fake.repository(REPO).issues[7].comments)
    assert plain == github.plain_text(text)
    assert "<!--" not in plain and "@dependabot" not in plain
    assert kept == text
    assert not fake.unanswered


def test_merge_and_update_act_only_at_the_head_they_name() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, behind_by=1)
    tools = _tools(fake)
    stale = "0" * 40

    refused_update = _run(tools["github_update_branch"], repository=REPO, number=7, head_sha=stale)
    updated = _run(tools["github_update_branch"], repository=REPO, number=7, head_sha=pull.head_sha)
    refused_merge = _run(tools["github_merge_pull"], repository=REPO, number=7, head_sha=stale)
    assert not pull.merged
    merged = _run(
        tools["github_merge_pull"],
        repository=REPO,
        number=7,
        head_sha=pull.head_sha,
        merge_method="squash",
    )

    for refused in (refused_update, refused_merge):
        error = _error(refused)
        assert error["error"] == "github_conflict"
        assert "read the pull request" in error["advice"]
    assert updated.structured == {
        "repository": REPO,
        "number": 7,
        "head_sha": pull.head_sha,
        "accepted": True,
    }
    assert _answer(merged)["merged"] is True
    assert (pull.merged, pull.merge_method) == (True, "squash")
    sent = [call.json() for call in fake.writes()]
    assert sent == [
        {"expected_head_sha": stale},
        {"expected_head_sha": pull.head_sha},
        {"sha": stale},
        {"sha": pull.head_sha, "merge_method": "squash"},
    ]
    short = _run(tools["github_merge_pull"], repository=REPO, number=7, head_sha="abc1234")
    assert _error(short)["error"] == "invalid_arguments"
    assert not fake.unanswered


def test_tools_act_only_on_the_repositories_they_were_given() -> None:
    fake = _fake()
    fake.add_pull("acme/other", 1)
    tools = _tools(fake, repositories=["Acme/Widgets"])

    refused = _run(tools["github_comment"], repository="acme/other", number=1, body="hi")
    allowed = _run(tools["github_read_issue"], repository="ACME/widgets", number=1)
    malformed = _run(tools["github_read_issue"], repository="widgets", number=1)

    # The schema names the repositories without an enum: a call's arguments are checked
    # against it before the tool runs, and that check would compare case.
    schema = tools["github_read_issue"].schema
    assert schema["properties"]["repository"]["description"] == "owner/name. One of: Acme/Widgets."
    assert "enum" not in schema["properties"]["repository"]
    assert (
        tool_argument_validation_error({"repository": "ACME/widgets", "number": 1}, schema) is None
    )
    error = _error(refused)
    assert error["error"] == "repository_not_allowed"
    assert "Acme/Widgets" in error["advice"]
    assert _error(allowed)["error"] == "github_not_found"
    assert _error(malformed)["error"] == "invalid_arguments"
    assert [call.path for call in fake.requests] == ["/repos/Acme/Widgets/issues/1"]
    with pytest.raises(ValueError, match="at least one"):
        github.tools(fake, repositories=[])
    with pytest.raises(TypeError):
        github.tools(fake, repositories=REPO)
    assert not fake.unanswered


def test_github_failures_are_errors_the_model_can_act_on() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    tools = _tools(fake)
    merge_path = f"/repos/{REPO}/pulls/7/merge"
    fake.fail(
        "PUT",
        merge_path,
        status=403,
        body={"message": "Resource not accessible by integration"},
        headers={"x-accepted-github-permissions": "contents=write"},
    )
    denied = _error(
        _run(tools["github_merge_pull"], repository=REPO, number=7, head_sha=pull.head_sha)
    )
    fake.fail("POST", f"/repos/{REPO}/issues/7/comments", status=502)
    ambiguous = _error(_run(tools["github_comment"], repository=REPO, number=7, body="hi"))
    fake.rate_limit(retry_after=3_600)
    limited = _error(_run(tools["github_read_issue"], repository=REPO, number=7))
    fake.refuse("scope_not_allowed")
    refused = _error(_run(tools["github_read_issue"], repository=REPO, number=7))
    missing = _error(_run(tools["github_read_pull"], repository=REPO, number=99))

    assert (denied["error"], denied["accepted_permissions"]) == (
        "github_permission_denied",
        "contents=write",
    )
    assert "Resource not accessible by integration" in denied["message"]
    assert (ambiguous["error"], ambiguous["ambiguous"]) == ("upstream_failed", True)
    assert "may have taken effect" in ambiguous["advice"]
    assert limited["error"] == "too_many_calls"
    assert limited["retry_after_seconds"] > 60
    assert (refused["error"], "github" in refused["advice"]) == ("scope_not_allowed", True)
    assert missing["error"] == "github_not_found"
    assert not pull.comments
    assert not fake.unanswered


def test_without_a_client_outside_cloud_the_tools_answer_an_error(monkeypatch) -> None:
    monkeypatch.delenv("CAYU_CONNECTIONS_URL", raising=False)
    *_, comment, _, _, _ = github.tools(include_writes=True)
    result = _run(comment, repository=REPO, number=1, body="hi")
    assert _error(result)["error"] == "no_connection_client"


def test_approval_policy_asks_before_writes_naming_the_target() -> None:
    policy = github.approval_policy()
    head = "1a2b3c4" + "d" * 33

    def decide(tool_name: str, **arguments: Any):
        return asyncio.run(policy.authorize(policy_request(tool_name, **arguments)))

    comment = decide("github_comment", repository=REPO, number=7, body="hi")
    rerun = decide("github_rerun_failed_jobs", repository=REPO, run_id=901)
    headed = decide("github_rerun_failed_jobs", repository=REPO, run_id=901, head_sha=head)
    update = decide("github_update_branch", repository=REPO, number=7, head_sha=head)
    merge = decide(
        "github_merge_pull", repository=REPO, number=7, head_sha=head, merge_method="squash"
    )
    read = decide("github_read_pull", repository=REPO, number=7)

    assert {comment.decision, rerun.decision, update.decision, merge.decision} == {
        ToolPolicyDecision.REQUIRE_APPROVAL
    }
    assert comment.reason == "Comment on acme/widgets#7."
    assert rerun.reason == "Rerun the failed jobs of workflow run 901 in acme/widgets."
    assert headed.reason == (
        "Rerun the failed jobs of workflow run 901 in acme/widgets at head 1a2b3c4."
    )
    assert update.reason == "Update the branch of acme/widgets#7 at head 1a2b3c4."
    assert merge.reason == "Merge acme/widgets#7 at head 1a2b3c4 (squash)."
    assert read.decision == ToolPolicyDecision.ALLOW
    long = decide("github_comment", repository="a/b\n" + "c" * 500, number=1, body="hi")
    assert long.reason is not None and len(long.reason) < 120 and "\n" not in long.reason


def test_approval_policy_leaves_every_other_decision_to_inner() -> None:
    denying = github.approval_policy(StaticToolPolicy(deny=["github_merge_pull"]))
    asking = github.approval_policy(
        AlwaysRequireApprovalToolPolicy(tools=["github_read_job_log"], approval_gate="people")
    )

    def decide(policy, tool_name: str):
        return asyncio.run(policy.authorize(policy_request(tool_name, repository=REPO)))

    assert decide(denying, "github_merge_pull").decision == ToolPolicyDecision.DENY
    assert decide(denying, "github_comment").decision == ToolPolicyDecision.REQUIRE_APPROVAL
    gated = decide(asking, "github_read_job_log")
    assert gated.decision == ToolPolicyDecision.REQUIRE_APPROVAL
    assert gated.metadata == {"cayu:approval_gate": "people"}


def test_review_fields_show_the_repository_number_and_head() -> None:
    fields = github.review_fields()
    assert fields == {
        "github_comment": ("repository", "number", "body"),
        "github_rerun_failed_jobs": ("repository", "run_id", "head_sha"),
        "github_update_branch": ("repository", "number", "head_sha"),
        "github_merge_pull": ("repository", "number", "head_sha", "merge_method"),
    }
    schemas = {
        tool.name: tool.schema["properties"] for tool in github.tools(_fake(), include_writes=True)
    }
    assert all(set(names) <= set(schemas[tool]) for tool, names in fields.items())
    ToolArgumentReview(fields, binding_key=b"k" * 32)


@pytest.mark.parametrize(
    "line",
    ["日本語のログ行です。" * 12, "\\" * 150 + '"'],
    ids=["cjk", "backslashes"],
)
def test_a_log_tail_fits_the_answer_budget_whatever_its_text(line: str) -> None:
    fake = _fake()
    fake.set_log(REPO, 5, "\n".join(f"{index} {line}" for index in range(3_000)))

    result = _run(_tools(fake)["github_read_job_log"], repository=REPO, job_id=5, lines=2_000)

    answer = _answer(result)
    assert answer["truncated"] is True
    assert answer["log"].endswith(f"2999 {line}")
    assert len(answer["log"]) > 1_000
    assert_within_budgets(result)
    assert not fake.unanswered


def test_a_log_the_agent_cant_download_is_unavailable_not_temporary() -> None:
    fake = _fake()
    fake.set_log(REPO, 5, "boom")

    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    fake.download_transport = httpx.MockTransport(unreachable)

    error = _error(_run(_tools(fake)["github_read_job_log"], repository=REPO, job_id=5))

    assert error["error"] == "log_unavailable"
    assert "don't retry" in error["advice"]
    assert "sig=" not in error["message"]
    assert not fake.unanswered


def test_read_pull_lists_a_bounded_number_of_failed_checks() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    ci = fake.add_workflow_run(REPO, pull.head_sha, conclusion="failure")
    for index in range(400):
        fake.add_check_run(
            REPO,
            pull.head_sha,
            f"job {index}",
            run=ci,
            conclusion="failure",
            output_title="t" * 1_000,
        )

    result = _run(_tools(fake)["github_read_pull"], repository=REPO, number=7)

    checks = _answer(result)["checks"]
    assert (len(checks["failed"]), checks["failed_total"]) == (30, 400)
    assert "failed_total" in checks["failed_note"]
    assert (checks["running"], checks["running_total"]) == ([], 0)
    assert "running_note" not in checks
    assert all(len(check["title"]) == 200 for check in checks["failed"])
    assert_within_budgets(result)
    assert not fake.unanswered


def test_read_pull_lists_a_bounded_number_of_running_checks() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    ci = fake.add_workflow_run(REPO, pull.head_sha, status="in_progress", conclusion=None)
    for index in range(400):
        fake.add_check_run(
            REPO,
            pull.head_sha,
            f"{index:03d}" + "x" * 297,
            run=ci,
            status="queued",
            conclusion=None,
        )

    result = _run(_tools(fake)["github_read_pull"], repository=REPO, number=7)

    checks = _answer(result)["checks"]
    assert (checks["state"], len(checks["running"]), checks["running_total"]) == (
        "running",
        30,
        400,
    )
    assert "running_total" in checks["running_note"]
    assert all(len(name) == 200 for name in checks["running"])
    assert_within_budgets(result)
    assert not fake.unanswered


def test_read_pull_shortens_check_texts_until_the_answer_fits() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    ci = fake.add_workflow_run(REPO, pull.head_sha, conclusion="failure")
    review = fake.add_workflow_run(REPO, pull.head_sha, name="Review")
    for index in range(30):
        fake.add_check_run(
            REPO,
            pull.head_sha,
            f"{index:02d}" + "検" * 300,
            run=ci,
            conclusion="failure",
            output_title="失" * 1_000,
        )
        fake.add_check_run(
            REPO,
            pull.head_sha,
            f"{index:02d}" + "審" * 300,
            run=review,
            app_slug=REVIEWER,
            output_title="題" * 1_000,
            output_summary="要" * 5_000,
        )

    result = _run(
        _tools(fake, reviewer_app=REVIEWER)["github_read_pull"], repository=REPO, number=7
    )

    checks = _answer(result)["checks"]
    assert (len(checks["failed"]), len(checks["reviewer"])) == (30, 30)
    assert all(seen["summary_shortened"] for seen in checks["reviewer"])
    assert max(len(seen["summary"]) for seen in checks["reviewer"]) < 500
    assert_within_budgets(result)
    assert not fake.unanswered


def test_a_log_with_nul_and_control_characters_is_answered_with_them_shown() -> None:
    fake = _fake()
    fake.set_log(REPO, 5, "before\x00after\n\x07bell \x1b[31mred\x1b[0m \x1bX\nend\x7f")

    fake.add_issue(REPO, 3, title="nul\x00title")
    fake.add_comment(REPO, 3, "a\x00b", author="ben")

    result = _run(_tools(fake)["github_read_job_log"], repository=REPO, job_id=5)
    issue = _answer(_run(_tools(fake)["github_read_issue"], repository=REPO, number=3))

    assert (issue["title"], issue["comments"][0]["body"]) == ("nul\\x00title", "a\\x00b")

    assert _answer(result)["log"] == "before\\x00after\n\\x07bell red \\x1bX\nend\\x7f"
    assert not fake.unanswered


def test_a_merge_github_cant_make_is_not_mergeable() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7, draft=True)

    error = _error(
        _run(_tools(fake)["github_merge_pull"], repository=REPO, number=7, head_sha=pull.head_sha)
    )

    assert error["error"] == "github_not_mergeable"
    assert "still a draft" in error["message"]
    assert "Read the pull request" in error["advice"]
    assert not pull.merged
    assert not fake.unanswered


def test_a_rerun_named_for_a_head_refuses_a_run_of_another() -> None:
    fake = _fake()
    pull = fake.add_pull(REPO, 7)
    old = fake.add_workflow_run(REPO, "b" * 40, conclusion="failure")
    tool = _tools(fake)["github_rerun_failed_jobs"]

    moved = _error(_run(tool, repository=REPO, run_id=old.id, head_sha=pull.head_sha))
    malformed = _error(_run(tool, repository=REPO, run_id=old.id, head_sha="b"))

    assert moved["error"] == "github_conflict"
    assert "b" * 40 in moved["message"]
    assert malformed["error"] == "invalid_arguments"
    assert (old.run_attempt, fake.writes()) == (1, [])
    assert not fake.unanswered


def test_read_issue_reads_only_the_last_pages_for_its_newest_comments() -> None:
    fake = _fake()
    fake.add_issue(REPO, 3, title="Busy")
    for index in range(5_120):
        fake.add_comment(REPO, 3, f"note {index}", author="ben")

    result = _run(_tools(fake)["github_read_issue"], repository=REPO, number=3, comments=50)

    answer = _answer(result)
    assert answer["comments_total"] == 5_120
    assert [c["body"] for c in answer["comments"]] == [f"note {i}" for i in range(5_070, 5_120)]
    pages = [call.query.get("page") for call in fake.requests if call.path.endswith("/comments")]
    assert pages == ["205", "204", "203"]
    assert not fake.unanswered


def test_write_tools_can_use_their_own_slot() -> None:
    hosts: list[tuple[str, str]] = []

    def github_hosts(request: httpx.Request) -> httpx.Response:
        hosts.append((request.method, request.url.host))
        if request.method == "POST":
            return httpx.Response(201, json={"id": 1, "html_url": "u", "created_at": "t"})
        return httpx.Response(200, json={"number": 1, "comments": 0})

    local = LocalConnections(
        {
            "github-read": LocalSlot("https://read.example"),
            "github-write": LocalSlot("https://write.example"),
        },
        transport=httpx.MockTransport(github_hosts),
    )
    tools = {
        tool.name: tool
        for tool in github.tools(
            local, slot="github-read", write_slot="github-write", include_writes=True
        )
    }

    _answer(_run(tools["github_comment"], repository=REPO, number=1, body="hi"))
    _answer(_run(tools["github_read_issue"], repository=REPO, number=1))

    assert hosts == [("POST", "write.example"), ("GET", "read.example")]


def test_cayu_check_sees_the_github_write_tools_guarded() -> None:
    def described(policy: Any) -> tuple[dict[str, str], dict[str, str]]:
        app = CayuApp(enable_logging=False)
        app.register_agent(
            AgentSpec(name="assistant", model="fake"),
            tools=github.tools(_fake(), include_writes=True),
            tool_policy=policy,
        )
        manifest = app.describe()
        flagged = {
            str(diagnostic.parameters["tool"]): diagnostic.code
            for diagnostic in check_manifest(manifest, deploy_only=True).diagnostics
            if diagnostic.code.startswith("EXTERNAL_TOOL_")
        }
        return {tool.name: tool.policy_coverage for tool in manifest.agents[0].tools}, flagged

    assert described(github.approval_policy()) == (
        {**dict.fromkeys(READS, "allowed"), **dict.fromkeys(WRITES, "approval_required")},
        {},
    )
    assert described(AllowAllToolPolicy())[1] == dict.fromkeys(WRITES, "EXTERNAL_TOOL_UNGUARDED")


def _write_calls(fake: FakeGitHub) -> dict[str, tuple[str, str, dict[str, Any]]]:
    """Each write tool's call on pull 7 (and its failed run): method, path and arguments."""

    pull = fake.add_pull(REPO, 7)
    run = fake.add_workflow_run(REPO, pull.head_sha, conclusion="failure")
    head = {"number": 7, "head_sha": pull.head_sha}
    return {
        "github_comment": ("POST", f"/repos/{REPO}/issues/7/comments", {"number": 7, "body": "hi"}),
        "github_rerun_failed_jobs": (
            "POST",
            f"/repos/{REPO}/actions/runs/{run.id}/rerun-failed-jobs",
            {"run_id": run.id},
        ),
        "github_update_branch": ("PUT", f"/repos/{REPO}/pulls/7/update-branch", head),
        "github_merge_pull": ("PUT", f"/repos/{REPO}/pulls/7/merge", head),
    }


_CHECKS = {
    "github_comment": "read the pull request's or issue's latest comments",
    "github_rerun_failed_jobs": "read the run's attempts",
    "github_update_branch": "read the pull request's head",
    "github_merge_pull": "read whether the pull request was merged",
}


@pytest.mark.parametrize("tool", WRITES)
@pytest.mark.parametrize("failure", ["upstream_response_too_large", "upstream_failed", "502"])
def test_a_write_that_may_have_run_says_how_to_check_and_never_to_retry_later(
    tool: str, failure: str
) -> None:
    fake = _fake()
    method, path, args = _write_calls(fake)[tool]
    if failure == "502":
        fake.fail(method, path, status=502)
    else:
        fake.refuse(failure, method, path)
    error = _error(_run(_tools(fake)[tool], repository=REPO, **args))

    assert error["error"] == ("upstream_failed" if failure == "502" else failure)
    assert error["ambiguous"] is True
    assert error["advice"] == f"It may have taken effect; {_CHECKS[tool]} before retrying."
    assert "retry later" not in error["advice"]
    assert len(fake.writes()) == 1
    assert not fake.unanswered


@pytest.mark.parametrize("tool", WRITES)
def test_a_write_refused_before_sending_and_a_read_are_never_ambiguous(tool: str) -> None:
    fake = _fake()
    method, path, args = _write_calls(fake)[tool]
    fake.refuse("scope_not_allowed", method, path)
    refused = _error(_run(_tools(fake)[tool], repository=REPO, **args))
    fake.refuse("upstream_response_too_large", "GET", f"/repos/{REPO}/issues/7")
    read = _error(_run(_tools(fake)["github_read_issue"], repository=REPO, number=7))

    assert refused["error"] == "scope_not_allowed" and "ambiguous" not in refused
    assert "don't retry" in refused["advice"]
    assert read["error"] == "upstream_response_too_large" and "ambiguous" not in read
    assert "ask for less" in read["advice"]
    assert not fake.unanswered
