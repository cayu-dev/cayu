"""Flagship recipe: a cloud PR-review agent, composed end-to-end on cayu.

This is the canonical answer to "build a cloud code reviewer agent that looks at
PRs, QAs the feature end to end, and posts comments." Every piece below is a
first-class cayu primitive; the value of this file is showing how they *compose*.

PIPELINE
--------
    GitHub webhook (pull_request opened/synchronize)      -> build_webhook_app()
      -> HMAC-verified in production, translated to a Task -> enqueue_pr_review()
      -> a worker claims the Task and starts a session     -> claim_and_run_one()
         bound to a fresh git checkout of the PR head       (PRSandboxFactory)
      -> the "pr-reviewer" agent: get_pr_diff -> read the
         changed code -> run QA commands in the sandbox ->
         post_pr_comment exactly once                       (build_app, tools below)

The trigger is durable (survives restarts, runs on a worker pool) rather than a
synchronous chat reply — that is what "cloud" means here.

GITHUB EGRESS — TWO OPTIONS
---------------------------
cayu ships no GitHub-specific primitive; you have two composable choices:

1. A custom ``Tool`` that calls the GitHub REST API through the environment's
   credential proxy (shown below with ``GetPRDiffTool`` / ``PostPRCommentTool``).
   Full control, zero extra infrastructure.
2. The GitHub MCP server via ``McpServerSpec(url=..., secret_headers=...)`` and
   ``connect_mcp_toolset`` — zero custom tool code. See docs/recipes/pr-reviewer.md
   and the "Streamable HTTP MCP" section of the README.

This file uses option 1 so it is self-contained and runnable with no extra server.

RUN IT
------
    # Deterministic demo (no model key needed; reads a public PR read-only):
    PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py

    # Live review of a real PR (needs a provider key + a GitHub token):
    OPENAI_API_KEY=... GITHUB_TOKEN=... \
      PYTHONPATH=src python examples/github_pr_reviewer/pr_reviewer.py --live owner/repo#123
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import sys
import tempfile
from pathlib import Path

import httpx

from cayu import (
    AgentSpec,
    CayuApp,
    CommandPolicy,
    CommandPolicyDecision,
    CommandPolicyResult,
    CommandRequest,
    Environment,
    EnvironmentFactory,
    EnvironmentFactoryRequest,
    EnvironmentFactoryResult,
    EnvironmentSpec,
    Event,
    EventType,
    ExecCommandTool,
    GitRepositoryBinding,
    ListFilesTool,
    LocalEnvVault,
    LocalRunner,
    LocalWorkspace,
    Message,
    PassthroughProxy,
    ReadFileTool,
    RunLimits,
    RunRequest,
    ScriptedModelProvider,
    SQLiteTaskStore,
    StaticToolPolicy,
    Task,
    TaskCreate,
    TaskQuery,
    TaskStatus,
    TaskStore,
    Tool,
    ToolContext,
    ToolResult,
    ToolSpec,
)
from cayu.providers import ModelStreamEvent
from cayu.vaults import SecretRef

# fastapi is optional (only cayu[server]/[dev] pull it in). It is imported at
# module level -- not lazily inside build_webhook_app -- because
# ``from __future__ import annotations`` defers annotation evaluation to
# typing.get_type_hints(fn), which reads fn.__globals__ (this module's globals),
# not an enclosing function's locals. A lazy ``from fastapi import Request``
# would leave ``Request`` out of module globals, and FastAPI would then treat the
# ``request: Request`` parameter as a required *query* param -- every POST 422s
# with a misleading "field required". (Real gotcha; keep this import here.)
try:
    from fastapi import FastAPI, HTTPException, Request

    _HAS_FASTAPI = True
except ImportError:  # pragma: no cover - only when fastapi is absent
    _HAS_FASTAPI = False

GITHUB_API = "https://api.github.com"
GITHUB_PAGE_SIZE = 100


class DeliveryTaskConflictError(ValueError):
    """Raised when a GitHub delivery id is reused with different task data."""


# --------------------------------------------------------------------------- #
# GitHub egress tools (option 1: custom Tools through the credential proxy).
# --------------------------------------------------------------------------- #


class GetPRDiffTool(Tool):
    """Fetch a PR's metadata + changed files/patches from the GitHub REST API."""

    spec = ToolSpec(
        name="get_pr_diff",
        description=(
            "Fetch a GitHub pull request's title, body, base/head refs, and the list "
            "of changed files with per-file patches. Uses the PR identity from the "
            "session's environment metadata when the caller omits it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repo owner/org."},
                "repo": {"type": "string", "description": "Repo name."},
                "pr_number": {"type": "integer", "description": "Pull request number."},
            },
            "additionalProperties": False,
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        owner = args.get("owner") or ctx.metadata.get("repo_owner")
        repo = args.get("repo") or ctx.metadata.get("repo_name")
        pr_number = args.get("pr_number") or ctx.metadata.get("pr_number")
        if not (owner and repo and pr_number):
            return ToolResult(
                content="Missing owner/repo/pr_number (not in tool args or session metadata).",
                is_error=True,
            )

        headers = {"Accept": "application/vnd.github+json"}
        token = await _resolve_github_token(ctx, tool="get_pr_diff")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        base = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            pr_resp = await client.get(base, headers=headers)
            if pr_resp.status_code != 200:
                return ToolResult(
                    content=f"GitHub API error {pr_resp.status_code} fetching PR: "
                    f"{pr_resp.text[:500]}",
                    is_error=True,
                )
            pr = pr_resp.json()
            files: list[dict] = []
            page = 1
            while True:
                files_resp = await client.get(
                    f"{base}/files",
                    headers=headers,
                    params={"per_page": GITHUB_PAGE_SIZE, "page": page},
                )
                if files_resp.status_code != 200:
                    return ToolResult(
                        content=f"GitHub API error {files_resp.status_code} fetching files: "
                        f"{files_resp.text[:500]}",
                        is_error=True,
                    )
                page_files = files_resp.json()
                files.extend(page_files)
                if len(page_files) < GITHUB_PAGE_SIZE:
                    break
                page += 1

        changed = [
            {
                "path": f["filename"],
                "status": f["status"],
                "additions": f["additions"],
                "deletions": f["deletions"],
                "patch": (f.get("patch") or "")[:4000],
            }
            for f in files
        ]
        summary = [
            f"PR #{pr_number} {pr['title']!r} ({pr['head']['ref']} -> {pr['base']['ref']})",
            f"{len(changed)} files changed.",
            "",
            *[f"- {c['path']} (+{c['additions']}/-{c['deletions']})" for c in changed],
        ]
        return ToolResult(
            content="\n".join(summary),
            structured={
                "title": pr["title"],
                "body": pr.get("body"),
                "head_ref": pr["head"]["ref"],
                "head_sha": pr["head"]["sha"],
                "base_ref": pr["base"]["ref"],
                "files": changed,
            },
        )


class PostPRCommentTool(Tool):
    """Post one review comment on a GitHub pull request (issue-comment style)."""

    spec = ToolSpec(
        name="post_pr_comment",
        description="Post one review comment on the pull request being reviewed.",
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "pr_number": {"type": "integer"},
                "body": {"type": "string", "description": "Markdown comment body."},
            },
            "required": ["body"],
            "additionalProperties": False,
        },
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        owner = args.get("owner") or ctx.metadata.get("repo_owner")
        repo = args.get("repo") or ctx.metadata.get("repo_name")
        pr_number = args.get("pr_number") or ctx.metadata.get("pr_number")
        body = args["body"]
        if not (owner and repo and pr_number):
            return ToolResult(
                content="Missing owner/repo/pr_number (not in tool args or session metadata).",
                is_error=True,
            )
        if ctx.proxy is None:
            # Fail closed: no credential proxy configured -> never reach the network.
            return ToolResult(
                content=(
                    "No credential proxy configured for this environment -- cannot post to "
                    "GitHub. Wire Environment(vault=..., proxy=...) with a github_token secret."
                ),
                is_error=True,
            )

        destination = f"{GITHUB_API}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        credential = SecretRef(name="github_token")
        authorization = await ctx.proxy.authorize_request(
            destination=destination,
            credential=credential,
            action="post_pr_comment",
            metadata={"owner": owner, "repo": repo, "pr_number": pr_number},
        )
        if not authorization.allowed:
            return ToolResult(
                content=f"Blocked by credential proxy: {authorization.reason}", is_error=True
            )

        resolved = await ctx.proxy.resolve(
            credential, scope={"session_id": ctx.session_id, "tool": "post_pr_comment"}
        )
        token = resolved.value.get_secret_value()
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                destination,
                headers={
                    "Accept": "application/vnd.github+json",
                    "Authorization": f"Bearer {token}",
                },
                json={"body": body},
            )
        if resp.status_code >= 300:
            return ToolResult(
                content=f"GitHub API error {resp.status_code}: {resp.text[:500]}", is_error=True
            )
        posted = resp.json()
        return ToolResult(
            content=f"Posted comment {posted.get('html_url', posted.get('id'))}",
            structured={"id": posted.get("id"), "html_url": posted.get("html_url")},
        )


async def _resolve_github_token(ctx: ToolContext, *, tool: str) -> str | None:
    """Resolve the github_token secret via the proxy, or None for anonymous reads."""
    if ctx.proxy is None:
        return None
    try:
        resolved = await ctx.proxy.resolve(
            SecretRef(name="github_token"), scope={"session_id": ctx.session_id, "tool": tool}
        )
        return resolved.value.get_secret_value()
    except Exception:
        return None  # no token configured -> unauthenticated, rate-limited public read


# --------------------------------------------------------------------------- #
# QA safety: restrict exec_command to an allowlist of real test/build tools so
# the model cannot run arbitrary shell while "QAing end-to-end".
# --------------------------------------------------------------------------- #

_ALLOWED_QA_COMMANDS = {
    "pytest",
    "python",
    "python3",
    "uv",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "node",
    "make",
    "go",
    "cargo",
    "tox",
    "ruff",
    "mypy",
    "jest",
}
_DENYLISTED_TOKENS = {"rm", "curl", "wget", "sudo", "ssh", "git", "scp"}


class QaCommandPolicy(CommandPolicy):
    """Allow only recognized test/build invocations; deny raw shell strings."""

    async def evaluate(self, ctx: ToolContext, request: CommandRequest) -> CommandPolicyResult:
        command = request.command
        if command.kind == "shell":
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY,
                reason="Raw shell strings are not allowed for QA; use kind='process' with argv.",
            )
        argv = command.argv or []
        if not argv or argv[0] not in _ALLOWED_QA_COMMANDS:
            got = argv[0] if argv else "<empty>"
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY, reason=f"'{got}' is not an allowed QA command."
            )
        if any(token in _DENYLISTED_TOKENS for token in argv):
            return CommandPolicyResult(
                decision=CommandPolicyDecision.DENY, reason="Command contains a disallowed token."
            )
        return CommandPolicyResult(decision=CommandPolicyDecision.ALLOW)


# --------------------------------------------------------------------------- #
# Per-review sandbox: a fresh workspace + runner + git checkout per session.
# Swap LocalRunner/LocalWorkspace for E2BRunner/E2BWorkspace (cayu[e2b]) to get
# real cloud-sandbox isolation instead of the trusted local-dev runner.
# --------------------------------------------------------------------------- #

_shared_vault = LocalEnvVault({"github_token": "GITHUB_TOKEN"})
_shared_proxy = PassthroughProxy(_shared_vault)


class PRSandboxFactory(EnvironmentFactory):
    """One fresh clone per review session, checked out at the PR's head branch."""

    def __init__(self, base_root: Path, *, with_credentials: bool = True) -> None:
        self._base_root = base_root
        self._with_credentials = with_credentials

    async def create(self, request: EnvironmentFactoryRequest) -> EnvironmentFactoryResult:
        repo_url = request.metadata.get("repo_url")
        ref = request.metadata.get("head_ref")
        if not repo_url or not ref:
            raise ValueError(
                "PRSandboxFactory requires 'repo_url' and 'head_ref' in RunRequest.metadata; "
                "got: " + json.dumps(request.metadata)
            )
        root = self._base_root / request.session_id
        root.mkdir(parents=True, exist_ok=True)
        environment = Environment(
            EnvironmentSpec(name=request.environment_name),
            workspace=LocalWorkspace(root, workspace_id=request.session_id),
            runner=LocalRunner(root),
            binding=GitRepositoryBinding(repo_url=repo_url, ref=ref),
            vault=_shared_vault if self._with_credentials else None,
            proxy=_shared_proxy if self._with_credentials else None,
        )
        return EnvironmentFactoryResult(environment=environment)


REVIEWER_SYSTEM_PROMPT = """\
You are an autonomous pull-request review agent. For the pull request described in
the first user message:

1. Call get_pr_diff to see what changed.
2. Use list_files / read_file to inspect changed files and related code as needed.
3. QA the change by running the project's test suite and relevant checks with
   exec_command (only a fixed allowlist of test/build tools is permitted; raw shell
   strings are rejected).
4. Call post_pr_comment exactly once with a concise, specific review: what you
   checked, what passed/failed, and any concrete concerns. Do not comment twice.
"""


def build_app(
    task_db_path: Path, sandbox_root: Path, *, provider, model: str
) -> tuple[CayuApp, SQLiteTaskStore]:
    """Construct the full runtime: durable task store, per-PR sandbox, agent + tools."""
    task_store = SQLiteTaskStore(task_db_path)
    app = CayuApp(task_store=task_store)
    app.register_provider(provider, default=True)
    app.register_environment_factory(
        EnvironmentSpec(name="pr-sandbox"), PRSandboxFactory(sandbox_root), default=True
    )
    app.register_agent(
        AgentSpec(name="pr-reviewer", model=model, system_prompt=REVIEWER_SYSTEM_PROMPT),
        tools=[
            GetPRDiffTool(),
            PostPRCommentTool(),
            ReadFileTool(),
            ListFilesTool(),
            ExecCommandTool(policy=QaCommandPolicy()),
        ],
        tool_policy=StaticToolPolicy(
            allow=["get_pr_diff", "post_pr_comment", "read_file", "list_files", "exec_command"]
        ),
    )
    return app, task_store


def build_provider() -> tuple[object, str]:
    """Pick a real provider from whichever key is set (used outside the demo)."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from cayu import AnthropicProvider

        return AnthropicProvider(), os.environ.get("CAYU_MODEL", "claude-sonnet-4-6")
    if os.environ.get("OPENAI_API_KEY"):
        from cayu import OpenAIProvider

        return OpenAIProvider(), os.environ.get("CAYU_MODEL", "gpt-5.4-mini")
    raise RuntimeError("Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run a live review.")


# --------------------------------------------------------------------------- #
# Trigger side: durable Task queue + worker. EventWatcher watches cayu's *own*
# event log; it is not an external-webhook trigger, so this half is app code over
# any TaskStore implementation.
# --------------------------------------------------------------------------- #


async def enqueue_pr_review(
    task_store: TaskStore,
    *,
    owner: str,
    repo: str,
    pr_number: int,
    repo_url: str,
    head_ref: str,
    base_ref: str,
    task_id: str | None = None,
    github_delivery_id: str | None = None,
    head_sha: str | None = None,
) -> Task:
    create = TaskCreate(
        task_id=task_id,
        type="review_pr",
        title=f"Review {owner}/{repo}#{pr_number}",
        assigned_agent_name="pr-reviewer",
        input={
            "owner": owner,
            "repo": repo,
            "pr_number": pr_number,
            "repo_url": repo_url,
            "head_ref": head_ref,
            "head_sha": head_sha,
            "base_ref": base_ref,
        },
        metadata=(
            {"github_delivery_id": github_delivery_id} if github_delivery_id is not None else {}
        ),
    )
    try:
        return await task_store.create_task(create)
    except ValueError:
        if task_id is None:
            raise
        existing = await task_store.load_task(task_id)
        if existing is None:
            raise
        if (
            existing.type != create.type
            or existing.input != create.input
            or existing.metadata.get("github_delivery_id") != github_delivery_id
        ):
            raise DeliveryTaskConflictError(
                "GitHub delivery id already exists with different task data"
            ) from None
        return existing


async def claim_and_run_one(
    app: CayuApp, task_store: TaskStore, *, worker_id: str = "worker-1"
) -> Task | None:
    """Claim one pending review_pr task and run the pr-reviewer agent against it."""
    task = await task_store.claim_task(
        worker_id,
        TaskQuery(type="review_pr", assigned_agent_name="pr-reviewer"),
        lease_seconds=900,
    )
    if task is None:
        return None

    owner, repo, pr_number = task.input["owner"], task.input["repo"], task.input["pr_number"]
    session_id = _review_session_id(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        task_id=task.id,
        head_sha=task.input.get("head_sha"),
    )
    request = RunRequest(
        agent_name="pr-reviewer",
        session_id=session_id,
        task_id=task.id,
        task_worker_id=worker_id,
        environment_name="pr-sandbox",
        metadata={
            "repo_owner": owner,
            "repo_name": repo,
            "pr_number": pr_number,
            "repo_url": task.input["repo_url"],
            "head_ref": task.input["head_ref"],
            "head_sha": task.input.get("head_sha"),
            "base_ref": task.input["base_ref"],
        },
        messages=[
            Message.text(
                "user",
                f"Review pull request #{pr_number} in {owner}/{repo} "
                f"({task.input['head_ref']} -> {task.input['base_ref']}).",
            )
        ],
        limits=RunLimits(max_tool_calls=20, max_elapsed_seconds=600),
    )
    terminal_event: Event | None = None
    async for event in app.run(request):
        print(event.type, event.tool_name or "-", str(event.payload)[:160])
        if event.type in {
            EventType.SESSION_COMPLETED,
            EventType.SESSION_FAILED,
            EventType.SESSION_INTERRUPTED,
        }:
            terminal_event = event
    return await _terminalize_claimed_task_if_needed(
        task_store,
        task_id=task.id,
        worker_id=worker_id,
        terminal_event=terminal_event,
    )


async def _terminalize_claimed_task_if_needed(
    task_store: TaskStore,
    *,
    task_id: str,
    worker_id: str,
    terminal_event: Event | None,
) -> Task | None:
    task = await task_store.load_task(task_id)
    if task is None or task.status not in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
        return task
    if terminal_event is None:
        return task
    if terminal_event.type == EventType.SESSION_FAILED:
        return await task_store.fail_task(
            task_id,
            {
                "status": "failed",
                "error": terminal_event.payload.get("error"),
                "error_type": terminal_event.payload.get("error_type"),
            },
            worker_id=worker_id,
        )
    status = "interrupted" if terminal_event.type == EventType.SESSION_INTERRUPTED else "completed"
    return await task_store.complete_task(
        task_id,
        {"status": status},
        worker_id=worker_id,
    )


def build_webhook_app(task_store: TaskStore, *, webhook_secret: str | None):
    """A tiny GitHub webhook receiver that enqueues review_pr tasks."""
    if not _HAS_FASTAPI:
        raise RuntimeError("pip install fastapi (or cayu[server]) to use the webhook app.")
    if webhook_secret == "":
        raise ValueError("webhook_secret must be non-empty, or None to disable signature checks.")
    api = FastAPI(title="pr-reviewer-webhook")

    @api.post("/webhooks/github")
    async def github_webhook(request: Request) -> dict:
        raw = await request.body()
        if webhook_secret:
            signature = request.headers.get("X-Hub-Signature-256", "")
            digest = "sha256=" + hmac.new(webhook_secret.encode(), raw, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, digest):
                raise HTTPException(status_code=401, detail="bad signature")
        if request.headers.get("X-GitHub-Event") != "pull_request":
            return {"ignored": True}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON webhook payload") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="webhook payload must be a JSON object")
        if payload.get("action") not in {"opened", "reopened", "synchronize"}:
            return {"ignored": True}
        try:
            pr = payload["pull_request"]
            repository = payload["repository"]
            owner = repository["owner"]["login"]
            repo = repository["name"]
            pr_number = pr["number"]
            head = pr["head"]
            if not isinstance(head, dict):
                raise ValueError("invalid pull_request webhook payload")
            head_sha = head.get("sha")
            base_ref = pr["base"]["ref"]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=422, detail="invalid pull_request webhook payload"
            ) from exc
        delivery_id = request.headers.get("X-GitHub-Delivery")
        if not delivery_id:
            raise HTTPException(status_code=400, detail="missing X-GitHub-Delivery header")
        task_id = _github_delivery_task_id(delivery_id)
        try:
            repo_url, checkout_ref = _checkout_source_from_pr_payload(pr, payload)
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            task = await enqueue_pr_review(
                task_store,
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                repo_url=repo_url,
                head_ref=checkout_ref,
                head_sha=head_sha,
                base_ref=base_ref,
                task_id=task_id,
                github_delivery_id=delivery_id,
            )
        except DeliveryTaskConflictError as exc:
            raise HTTPException(
                status_code=409,
                detail=str(exc),
            ) from exc
        return {"task_id": task.id}

    return api


def _github_delivery_task_id(delivery_id: str) -> str:
    digest = hashlib.sha256(delivery_id.encode("utf-8")).hexdigest()[:32]
    return f"github-delivery-{digest}"


def _checkout_source_from_pr_payload(pr: dict, payload: dict) -> tuple[str, str]:
    head = pr["head"]
    if not isinstance(head, dict):
        raise ValueError("GitHub PR head payload must be a JSON object.")
    head_repo = head.get("repo")
    if not isinstance(head_repo, dict) or not head_repo.get("clone_url"):
        repository = payload.get("repository")
        full_name = (
            repository.get("full_name", "<unknown>")
            if isinstance(repository, dict)
            else "<unknown>"
        )
        raise ValueError(
            "GitHub PR head repository is unavailable for "
            f"{full_name}#{pr.get('number', '<unknown>')}; fetch refs/pull/N/head "
            "from the base repository before binding if deleted-fork PRs must be reviewed."
        )
    head_ref = head.get("ref")
    if not head_ref:
        raise ValueError("GitHub PR head ref is missing.")
    return head_repo["clone_url"], head_ref


def _review_session_id(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    task_id: str,
    head_sha: str | None,
) -> str:
    sha_part = head_sha[:12] if head_sha else "no-head-sha"
    task_part = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return f"pr-review-{owner}-{repo}-{pr_number}-{sha_part}-{task_part}"


# --------------------------------------------------------------------------- #
# Entry points.
# --------------------------------------------------------------------------- #


async def review_pr(owner: str, repo: str, pr_number: int) -> None:
    """Live review of a real PR. Requires a provider key + GITHUB_TOKEN."""
    provider, model = build_provider()
    async with httpx.AsyncClient(timeout=20.0) as client:
        pr_resp = await client.get(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}",
            headers={
                "Accept": "application/vnd.github+json",
                **(
                    {"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
                    if os.environ.get("GITHUB_TOKEN")
                    else {}
                ),
            },
        )
    if pr_resp.status_code != 200:
        raise RuntimeError(
            f"GitHub API error {pr_resp.status_code} fetching PR: {pr_resp.text[:500]}"
        )
    pr = pr_resp.json()
    repo_url, checkout_ref = _checkout_source_from_pr_payload(
        pr, {"repository": pr["base"]["repo"]}
    )
    scratch = Path(tempfile.mkdtemp(prefix="cayu_pr_reviewer_"))
    app, task_store = build_app(
        scratch / "tasks.sqlite", scratch / "sandboxes", provider=provider, model=model
    )
    await enqueue_pr_review(
        task_store,
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        repo_url=repo_url,
        head_ref=checkout_ref,
        head_sha=pr["head"]["sha"],
        base_ref=pr["base"]["ref"],
    )
    finished = await claim_and_run_one(app, task_store, worker_id="live-worker")
    assert finished is not None  # we just enqueued exactly one task for this worker
    print("task:", finished.id, finished.status, finished.session_id)


async def demo() -> None:
    """No-key demo: drives the real runtime with a scripted (non-live) provider.

    Reads a public PR read-only and exercises the sandbox, the QA command policy
    (including a denied raw-shell attempt), and the fail-closed post path.
    """
    owner, repo, pr_number = "octocat", "Hello-World", 1
    provider = ScriptedModelProvider(
        [
            [
                ModelStreamEvent.tool_call(id="c1", name="get_pr_diff", arguments={}),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="c2",
                    name="exec_command",
                    arguments={"kind": "process", "argv": ["python3", "--version"]},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="c3", name="exec_command", arguments={"kind": "shell", "shell": "rm -rf /"}
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.tool_call(
                    id="c4",
                    name="post_pr_comment",
                    arguments={"body": "Ran python3 --version as a smoke QA check; LGTM."},
                ),
                ModelStreamEvent.completed({"finish_reason": "tool_calls"}),
            ],
            [
                ModelStreamEvent.text_delta(
                    "Review finished; posting was blocked without a token."
                ),
                ModelStreamEvent.completed({"finish_reason": "stop"}),
            ],
        ]
    )
    os.environ.pop("GITHUB_TOKEN", None)  # demo posts fail closed (no token)
    scratch = Path(tempfile.mkdtemp(prefix="cayu_pr_reviewer_demo_"))
    app, task_store = build_app(
        scratch / "tasks.sqlite", scratch / "sandboxes", provider=provider, model="scripted-model"
    )
    await enqueue_pr_review(
        task_store,
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        repo_url=f"https://github.com/{owner}/{repo}.git",
        head_ref="test",
        head_sha="demo",
        base_ref="master",
    )
    finished = await claim_and_run_one(app, task_store, worker_id="demo-worker")
    assert finished is not None  # we just enqueued exactly one task for this worker
    print("task:", finished.id, finished.status, "| model requests:", len(provider.requests))


def main(argv: list[str]) -> None:
    if len(argv) >= 2 and argv[0] == "--live":
        spec = argv[1]  # owner/repo#123
        owner_repo, _, num = spec.partition("#")
        owner, _, repo = owner_repo.partition("/")
        asyncio.run(review_pr(owner, repo, int(num)))
    else:
        asyncio.run(demo())


if __name__ == "__main__":
    main(sys.argv[1:])
